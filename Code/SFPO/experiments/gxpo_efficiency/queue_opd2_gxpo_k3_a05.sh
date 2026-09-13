#!/usr/bin/env bash
#
# Queue the OPD^2+GXPO variant K=3 / alpha=0.5 behind the running K=3 / alpha=0.1 arm.
#
# Waits for GPUs 1,2 to be done, verifies they are free, then launches. It never
# kills processes because those may belong to another queue.
#
# Finish is detected three ways, whichever lands first:
#   1. the opd2_gxpo tmux session is gone          -> definitely done
#   2. GPUs 1 AND 2 below MEM_FREE_MIB            -> clean exit, memory released
#   3. GPUs 1 AND 2 at 0% util for IDLE_NEEDED min -> done but memory piled up
# A periodic eval keeps utilisation non-zero, so (3) does not fire mid-run.
#
#   bash experiments/gxpo_efficiency/queue_opd2_gxpo_k3_a05.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

WATCH_GPUS="1,2"
GUARD_GPUS="0,3,4"          # anything alive here must survive
MEM_FREE_MIB=2000
IDLE_NEEDED=15              # consecutive 0%-utilisation minutes
POLL_S=60
# At 3 minutes, 0% utilisation alone is not proof the run ended -- a checkpoint
# write or an inter-phase stall can idle both GPUs that long mid-training, and a
# false positive kills a healthy run. So the idle path additionally requires the
# training log to have gone stale for the same window. A genuinely finished run
# satisfies both; a mid-run lull keeps writing to the log and does not.
TRAIN_LOG="$CODE_ROOT/runs_opd2_opd2_gxpo.log"
SESSION="opd2_gxpo"
NEW_SESSION="opd2_gxpo_a05"

say() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- wait --------
idle=0
while true; do
  if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
    say "tmux session '$SESSION' is gone -- previous run finished."; break
  fi
  mapfile -t rows < <(nvidia-smi -i "$WATCH_GPUS" \
      --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  if [[ ${#rows[@]} -lt 2 ]]; then say "nvidia-smi read failed; retrying"; sleep "$POLL_S"; continue; fi
  m1=${rows[0]%%,*}; u1=${rows[0]##*, }
  m2=${rows[1]%%,*}; u2=${rows[1]##*, }
  m1=${m1// /}; u1=${u1// /}; m2=${m2// /}; u2=${u2// /}

  if (( u1 == 0 && u2 == 0 )); then idle=$((idle + 1)); else idle=0; fi
  log_age=999999
  [[ -f "$TRAIN_LOG" ]] && log_age=$(( $(date +%s) - $(stat -c %Y "$TRAIN_LOG") ))
  say "gpu1 ${m1}MiB ${u1}%   gpu2 ${m2}MiB ${u2}%   zero-util ${idle}/${IDLE_NEEDED}min   log_age ${log_age}s"

  if (( m1 < MEM_FREE_MIB && m2 < MEM_FREE_MIB )); then
    say "both GPUs below ${MEM_FREE_MIB}MiB -- freed cleanly."; break
  fi
  if (( idle >= IDLE_NEEDED && log_age >= IDLE_NEEDED * 60 )); then
    say "0% utilisation AND log stale for ${IDLE_NEEDED}min -- finished but piled up."; break
  fi
  sleep "$POLL_S"
done

# ---------------------------------------------------------------- free --------
pids=$(nvidia-smi -i "$WATCH_GPUS" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d " " | sort -u)
if [[ -n "$pids" ]]; then
  say "REFUSING to launch: GPUs $WATCH_GPUS still have compute PIDs: $pids"
  exit 1
fi
say "GPUs $WATCH_GPUS free; no processes were killed"

# --------------------------------------------------------------- launch -------
# Recipe values transcribed from launch_opd2_paper_pair.sh's PAPER array; only K,
# alpha and the run name differ. GXPO_RUN_NAME is required because K is unchanged
# at 3 and alpha is NOT part of the derived run name (common.sh:347) -- without it
# resume_mode=auto would silently resume the alpha=0.1 checkpoint.
say "launching K=3 alpha=0.5 in tmux session '$NEW_SESSION'"
tmux new-session -d -s "$NEW_SESSION" -c "$CODE_ROOT" \
  "env TRAIN_BATCH_SIZE=256 PPO_MINI_BATCH_SIZE=256 ROLLOUT_N=1 VAL_N=4 \
       MAX_RESPONSE_LENGTH=8192 MAX_STEPS=100 TRAIN_SEED=42 \
       LR=5e-6 LR_WARMUP_STYLE=cosine LR_WARMUP_RATIO=0.1 LR_MIN_RATIO=0.1 \
       ADAMW_WEIGHT_DECAY=0.0 ROLLOUT_TEMPERATURE=0.7 ROLLOUT_TOP_P=1.0 \
       OPD2_TOPK=1024 OPD2_GEN_LOSS_WEIGHT=0.1 OPD2_REWARDS_BIAS=0.0 \
       LOSS_AGG_MODE=seq-mean-token-mean \
       OPD2_MICRO_BATCH_SIZE=4 OPD2_CHUNK_TOKENS=1024 OPD2_KEEP_ON_GPU=False \
       VLLM_MAX_NUM_SEQS=256 VLLM_MAX_NUM_BATCHED_TOKENS=98304 VLLM_ENABLE_CHUNKED_PREFILL=True \
       TRAINER_TEST_FREQ=5 VAL_BEFORE_TRAIN=True SAVE_FREQ=10 FINAL_EVAL_ENABLED=False \
       WANDB_PROJECT=gxpo-opd2 \
       GPU_IDS=1,2 GPU_COUNT=2 FSDP_SIZE=2 VLLM_GPU_MEMORY_UTILIZATION=0.6 \
       K=3 REPOSITION_ALPHA=0.5 \
       GXPO_MIN_EFFECTIVE_MULTIPLIER=0 GXPO_WARMUP_STEPS=3 ENTROPY_COEFF=0.001 \
       GXPO_ZSCORE_W=20 GXPO_TAU=3.0 \
       GXPO_RUN_NAME=qwen25-1p5b_gxpo_k3_a05_seed42 \
       bash experiments/gxpo_efficiency/qwen25_1p5b_opd2_gxpo_adamw_dir_k10.sh \
       2>&1 | tee runs_opd2_opd2_gxpo_a05.log"

sleep 5
tmux has-session -t "=$NEW_SESSION" 2>/dev/null \
  && say "launched. attach: tmux attach -t $NEW_SESSION | log: runs_opd2_opd2_gxpo_a05.log" \
  || say "LAUNCH FAILED -- session '$NEW_SESSION' did not start"
