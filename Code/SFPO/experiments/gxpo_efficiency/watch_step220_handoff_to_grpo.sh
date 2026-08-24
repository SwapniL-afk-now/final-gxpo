#!/usr/bin/env bash
set -euo pipefail

# Wait for the fully written step-220 checkpoint and its validation record,
# stop the GXPO driver, then start a GRPO continuation from step 220. The
# trainer resumes at global step 221 after loading this checkpoint.

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SFPO_ROOT="$PROJECT_ROOT/Code/SFPO"
RUN_DIR="${GXPO_HANDOFF_RUN_DIR:-$PROJECT_ROOT/results/gxpo_efficiency/qwen25_math_1p5b_gxpo_k10_a05_b256_mb64_fsdp2_fp32_noliger_v3_20260823}"
CHECKPOINT_DIR="$RUN_DIR/global_step_220"
VALIDATION_METRICS="$RUN_DIR/greedy_validation.jsonl"
BASE_LAUNCHER="$SFPO_ROOT/experiments/gxpo_efficiency/qwen25_math_1p5b_gxpo_b256_a05.sh"
WATCH_LOG="$RUN_DIR/grpo_handoff_watcher.log"
TRAIN_SESSION="${GXPO_HANDOFF_TRAIN_SESSION:-gxpo-qwen25-fsdp2}"
CONT_SESSION="${GXPO_HANDOFF_CONT_SESSION:-gxpo-qwen25-grpo}"
POLL_SECONDS="${GXPO_HANDOFF_POLL_SECONDS:-15}"
CONT_RUN_NAME="${GXPO_HANDOFF_CONT_RUN_NAME:-qwen25_math_1p5b_grpo_from_step220_20260824}"
SWAPNIL_ROOT="$(cd -- "$PROJECT_ROOT/.." && pwd)"
CONT_RUN_DIR="$PROJECT_ROOT/results/gxpo_efficiency/$CONT_RUN_NAME"

mkdir -p "$RUN_DIR" "$CONT_RUN_DIR"

log() {
  printf '[handoff %s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$WATCH_LOG"
}

required_checkpoint_files=(
  actor/model_world_size_2_rank_0.pt
  actor/model_world_size_2_rank_1.pt
  actor/optim_world_size_2_rank_0.pt
  actor/optim_world_size_2_rank_1.pt
  actor/extra_state_world_size_2_rank_0.pt
  actor/extra_state_world_size_2_rank_1.pt
  data.pt
  data_profiler.pt
)

checkpoint_complete() {
  [[ -d "$CHECKPOINT_DIR" ]] || return 1
  local rel
  for rel in "${required_checkpoint_files[@]}"; do
    [[ -s "$CHECKPOINT_DIR/$rel" ]] || return 1
  done
}

checkpoint_signature() {
  local rel
  for rel in "${required_checkpoint_files[@]}"; do
    stat -c '%n:%s:%Y' "$CHECKPOINT_DIR/$rel"
  done | sort | sha256sum | awk '{print $1}'
}

validation_complete() {
  [[ -s "$VALIDATION_METRICS" ]] || return 1
  rg -q '"eval_greedy/global_step": 220' "$VALIDATION_METRICS"
}

main_training_pid() {
  ps -eo pid=,args= | awk '
    $0 ~ /python -u -m verl\.trainer\.main_ppo/ &&
    $0 ~ /experiment_name=qwen25_math_1p5b_gxpo_k10/ { print $1; exit }
  '
}

stop_current_training() {
  local pid
  pid="$(main_training_pid || true)"
  log "requesting current GXPO tmux job to stop (pid=${pid:-unknown})"
  if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
    tmux send-keys -t "$TRAIN_SESSION" C-c
  elif [[ -n "$pid" ]]; then
    kill -INT "$pid" 2>/dev/null || true
  fi

  for _ in $(seq 1 40); do
    pid="$(main_training_pid || true)"
    [[ -z "$pid" ]] && return 0
    sleep 3
  done

  pid="$(main_training_pid || true)"
  if [[ -n "$pid" ]]; then
    log "GXPO driver did not exit after interrupt; sending TERM to pid $pid"
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      pid="$(main_training_pid || true)"
      [[ -z "$pid" ]] && return 0
      sleep 3
    done
  fi

  pid="$(main_training_pid || true)"
  if [[ -n "$pid" ]]; then
    log "GXPO driver still exists; sending KILL to pid $pid"
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

log "watching for complete checkpoint: $CHECKPOINT_DIR"
last_signature=""
stable_count=0
while true; do
  if checkpoint_complete; then
    current_signature="$(checkpoint_signature)"
    if [[ "$current_signature" == "$last_signature" ]]; then
      stable_count=$((stable_count + 1))
    else
      last_signature="$current_signature"
      stable_count=1
    fi
    log "checkpoint files present; stable polls=$stable_count/2"
    if (( stable_count >= 2 )) && validation_complete; then
      log "step-220 validation record and checkpoint are complete"
      break
    fi
  else
    stable_count=0
    last_signature=""
  fi
  sleep "$POLL_SECONDS"
done

stop_current_training
log "GXPO driver stopped; forcing permanent GXPO shutoff for the continuation"

if tmux has-session -t "$CONT_SESSION" 2>/dev/null; then
  log "continuation session $CONT_SESSION already exists; refusing to start a duplicate"
  exit 2
fi

CONT_RAY_ROOT="$SWAPNIL_ROOT/.grpo"
CONT_RAY_AIR="$SWAPNIL_ROOT/.grair"

# Preserve the dataloader bookkeeping that the trainer reads from the run root.
for state_file in current_epoch.txt sampling_num.txt; do
  if [[ -f "$RUN_DIR/$state_file" ]]; then
    cp "$RUN_DIR/$state_file" "$CONT_RUN_DIR/$state_file"
  fi
done

tmux new-session -d -s "$CONT_SESSION" -c "$SFPO_ROOT" \
  "export METHOD=grpo; \
   export GXPO_RUN_NAME='$CONT_RUN_NAME'; \
   export TRAINER_RESUME_MODE='$CHECKPOINT_DIR'; \
   export TRAINER_RESUME_FROM_PATH=True; \
   export VAL_BEFORE_TRAIN=False; \
   export TRAINER_TEST_FREQ=5; \
   export MAX_STEPS=400; \
   export SAVE_FREQ=20; \
   export TRAIN_BATCH_SIZE=256; \
   export PPO_MINI_BATCH_SIZE=64; \
   export LOG_PROB_MICRO_BATCH_SIZE=8; \
   export FSDP_SIZE=2; \
   export GPU_IDS=0,1; \
   export GPU_COUNT=2; \
   export USE_LIGER=False; \
   export ACTOR_MODEL_DTYPE=float32; \
   export WANDB_PROJECT=gxpo-efficiency-final; \
   export WANDB_GROUP=qwen25-math-1p5b-b256; \
   export WANDB_TAGS='model:qwen25-math-1p5b,method:grpo,source:gxpo-step220,transition:gxpo-to-grpo'; \
   export RAY_TMPDIR='$CONT_RAY_ROOT'; \
   export RAY_AIR_LOCAL_CACHE_DIR='$CONT_RAY_AIR'; \
   export GXPO_CONCISE_LOGS=1; \
   export TRANSFORMERS_VERBOSITY=error; \
   exec bash '$BASE_LAUNCHER'"

log "started GRPO continuation in tmux session $CONT_SESSION"
log "resume source: $CHECKPOINT_DIR"
log "continuation will execute step 221 onward; step 220 will not be revalidated"
