#!/usr/bin/env bash
#
# queue_opd2_gxpo_k3_a03.sh
#
# Queues the k=3 / alpha=0.3 OPD^2+GXPO variant on GPUs 1,2 behind the live
# opd2_gxpo run. Fires when EITHER finish indicator holds:
#   A) no compute processes on GPU 1 or 2 (2 consecutive 60s polls), OR
#   B) 0% utilization on both GPUs for 15 consecutive minutes.
# Then it SIGTERMs leftovers on GPUs 1,2 ONLY (never touches 3,4), verifies
# the GPUs are free, and launches tmux session opd2_gxpo_a03.
#
# Run inside tmux; the training log attaches to the new session AND to
# runs_opd2_opd2_gxpo_a03.log:
#   tmux new-session -d -s queue_gxpo_a03 -c Code/SFPO \
#     "bash experiments/gxpo_efficiency/queue_opd2_gxpo_k3_a03.sh 2>&1 | tee runs_queue_gxpo_a03.log"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$CODE_ROOT"

GPUS="1,2"
SESSION="opd2_gxpo_a03"
POLL=60
IDLE_NEED=15
FREE_NEED=2

log() { echo "[$(date '+%F %T')] [queue] $*"; }

gpu_pids() {
  nvidia-smi -i "$GPUS" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | grep -E '^[0-9]+$' || true
}
gpus_busy() { [[ -n "$(gpu_pids)" ]]; }
gpus_idle_now() { ! nvidia-smi -i "$GPUS" --query-gpu=utilization.gpu \
  --format=csv,noheader,nounits 2>/dev/null | grep -qE '[1-9]'; }

log "watching GPUs $GPUS for opd2_gxpo to finish (target session: $SESSION)"
free_streak=0
idle_streak=0
while true; do
  sleep "$POLL"
  if gpus_busy; then
    free_streak=0
    if gpus_idle_now; then
      idle_streak=$((idle_streak + 1))
    else
      idle_streak=0
    fi
    log "busy (pids: $(gpu_pids | tr '\n' ' ')) idle_streak=${idle_streak}/${IDLE_NEED}"
  else
    idle_streak=0
    free_streak=$((free_streak + 1))
    log "no compute processes, free_streak=${free_streak}/${FREE_NEED}"
  fi
  if [[ "$free_streak" -ge "$FREE_NEED" ]]; then
    log "indicator A: GPUs $GPUS freed"
    break
  fi
  if [[ "$idle_streak" -ge "$IDLE_NEED" ]]; then
    log "indicator B: 0% utilization for ${IDLE_NEED} minutes"
    break
  fi
done

# ---- free GPUs 1,2 (leftover PIDs on THESE GPUs only) ----
leftovers="$(gpu_pids)"
if [[ -n "$leftovers" ]]; then
  log "SIGTERM leftovers on GPUs $GPUS: $(echo "$leftovers" | tr '\n' ' ')"
  # shellcheck disable=SC2086
  kill -TERM $leftovers 2>/dev/null || true
  sleep 15
  leftovers="$(gpu_pids)"
  if [[ -n "$leftovers" ]]; then
    log "SIGKILL stragglers: $(echo "$leftovers" | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill -KILL $leftovers 2>/dev/null || true
    sleep 5
    leftovers="$(gpu_pids)"
  fi
fi
if [[ -n "$leftovers" ]]; then
  log "REFUSING to launch: GPUs $GPUS still busy: $(echo "$leftovers" | tr '\n' ' ')" >&2
  exit 1
fi
log "GPUs $GPUS free"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  log "tmux session '$SESSION' already exists - not relaunching" >&2
  exit 1
fi

# ---- same paper recipe as launch_opd2_paper_pair.sh, alpha 0.3 variant ----
PAPER=(
  TRAIN_BATCH_SIZE=256
  PPO_MINI_BATCH_SIZE=256
  ROLLOUT_N=1
  VAL_N=4
  MAX_RESPONSE_LENGTH=8192
  MAX_STEPS=100
  TRAIN_SEED=42
  LR=5e-6
  LR_WARMUP_STYLE=cosine
  LR_WARMUP_RATIO=0.1
  LR_MIN_RATIO=0.1
  ADAMW_WEIGHT_DECAY=0.0
  ROLLOUT_TEMPERATURE=0.7
  ROLLOUT_TOP_P=1.0
  OPD2_TOPK=1024
  OPD2_GEN_LOSS_WEIGHT=0.1
  OPD2_REWARDS_BIAS=0.0
  LOSS_AGG_MODE=seq-mean-token-mean
  OPD2_MICRO_BATCH_SIZE=4
  OPD2_CHUNK_TOKENS=1024
  OPD2_KEEP_ON_GPU=False
  VLLM_MAX_NUM_SEQS=256
  VLLM_MAX_NUM_BATCHED_TOKENS=98304
  VLLM_ENABLE_CHUNKED_PREFILL=True
  TRAINER_TEST_FREQ=5
  VAL_BEFORE_TRAIN=True
  SAVE_FREQ=10
  FINAL_EVAL_ENABLED=False
  WANDB_PROJECT=gxpo-opd2
)
# Own run dir/wandb run: alpha is not part of the default name, so without
# this the variant would resume the alpha=0.8 run's checkpoints under
# resume_mode=auto.
GXPO_RUN_NAME="qwen25-1p5b_gxpo_k3_a03_seed42_adamwdir_opd2_seqmean"

log "launching $SESSION on GPUs $GPUS (K=3, alpha=0.3)"
tmux new-session -d -s "$SESSION" -c "$CODE_ROOT" \
  "env ${PAPER[*]} GXPO_RUN_NAME=$GXPO_RUN_NAME GPU_IDS=$GPUS GPU_COUNT=2 FSDP_SIZE=2 \
   VLLM_GPU_MEMORY_UTILIZATION=0.45 \
   K=3 REPOSITION_ALPHA=0.3 \
   GXPO_MIN_EFFECTIVE_MULTIPLIER=1.0 \
   GXPO_WARMUP_STEPS=12 \
   ENTROPY_COEFF=0.01 \
   bash experiments/gxpo_efficiency/qwen25_1p5b_opd2_gxpo_adamw_dir_k10.sh 2>&1 | tee runs_opd2_$SESSION.log"
log "launched. attach: tmux attach -t $SESSION"
