#!/usr/bin/env bash
set -euo pipefail

# Queue one run behind the live low-RAM control run. Fail closed: completion,
# session exit, and a clear GPU check are all required before launch.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CURRENT_SESSION="${CURRENT_SESSION:-opd2-qwen25-3b-7b-gpu012}"
CURRENT_RUN="${CURRENT_RUN:-$CODE_ROOT/results/gxpo_efficiency/qwen25-3b-qmath_teacher7b_gpu012_grpo_seed3407_adamwdir_opd2}"
CURRENT_METRICS="$CURRENT_RUN/train_metrics.jsonl"
NEW_SESSION="${NEW_SESSION:-opd2-gxpo-qwen25-3b-7b-gpu012}"
RUN_LOG="$CODE_ROOT/results/gxpo_efficiency/${NEW_SESSION}-tmux.log"
POLL_SECONDS="${POLL_SECONDS:-30}"

log() { printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }

completed() { rg -q '"train/global_step"[[:space:]]*:[[:space:]]*100([.,}]|$)' "$CURRENT_METRICS" 2>/dev/null; }
session_alive() { tmux has-session -t "=$CURRENT_SESSION" 2>/dev/null; }
gpu_pids() {
  nvidia-smi -i 0,1,2 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | grep -E '^[0-9]+$' | sort -u || true
}

log "waiting for $CURRENT_SESSION to complete 100 steps"
while ! completed || session_alive; do
  step="$(rg -o '"train/global_step"[[:space:]]*:[[:space:]]*[0-9]+' "$CURRENT_METRICS" 2>/dev/null | tail -n1 | grep -o '[0-9][0-9]*' || true)"
  log "control step=${step:-unknown} session=$(session_alive && echo active || echo exited)"
  sleep "$POLL_SECONDS"
done

log "control reached step 100 and its tmux session exited; waiting for GPUs 0,1,2 to clear"
while pids="$(gpu_pids)"; [[ -n "$pids" ]]; do
  log "still busy: $(echo "$pids" | tr '\n' ' ')"
  sleep "$POLL_SECONDS"
done

if tmux has-session -t "=$NEW_SESSION" 2>/dev/null; then
  log "refusing duplicate launch: tmux session $NEW_SESSION already exists"
  exit 2
fi

mkdir -p "$(dirname "$RUN_LOG")"
log "GPUs 0,1,2 are clear; launching GXPO in $NEW_SESSION"
set +e
env GXPO_RUN_NAME=qwen25-3b-qmath_teacher7b_gpu012_opd2_gxpo_k5_a01_queue \
  TRAINER_RESUME_MODE=disable ENTROPY_COEFF=0 MAX_STEPS=100 \
  GPU_IDS=0,1,2 GPU_COUNT=2 FSDP_SIZE=2 \
  OPD2_DEDICATED_GPU=True OPD2_KEEP_ON_GPU=True OPD2_ATTN_IMPL=flash_attention_2 \
  ACTOR_PARAM_OFFLOAD=False ACTOR_OPTIMIZER_OFFLOAD=False \
  DATALOADER_NUM_WORKERS=0 RAY_memory_usage_threshold=0.85 RAY_OBJECT_STORE_MEMORY_GB=8 \
  bash "$SCRIPT_DIR/qwen25_1p5b_opd2_gxpo_adamw_dir_k10.sh" 2>&1 | tee -a "$RUN_LOG"
rc=${PIPESTATUS[0]}
set -e
log "GXPO exited with status $rc"
exit "$rc"
