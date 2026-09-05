#!/usr/bin/env bash
#
# evaluate_checkpoints_greedy.sh — greedy benchmark eval for KD+GXPO checkpoints.
#
# The KD-SFT trainer's TEST_FREQ only scores KD val/loss. This watcher adds the
# real thing: for every new flat global_step_* checkpoint under RUN_DIR it runs
# GREEDY decoding (n=1, temperature=0) over the six GXPO benchmarks
# (math500, aime24, aime25, amc, minerva, olympiadbench) via
# tools/kd_sft/evaluate_greedy.py (flat HF checkpoints), sequentially, one checkpoint at a time.
#
# It never colocates with training: a checkpoint is evaluated only when a GPU
# is actually free (>80GB available and no trainer alive), otherwise it waits.
# Safe to launch alongside training in its own tmux session:
#   bash evaluate_checkpoints_greedy.sh <RUN_DIR>            # one pass + exit
#   bash evaluate_checkpoints_greedy.sh <RUN_DIR> --watch    # poll until DONE
set -euo pipefail

RUN_DIR="${1:?usage: $0 <RUN_DIR> [--watch]}"
WATCH=0
[[ "${2:-}" == "--watch" ]] && WATCH=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then set -a; source "$REPO_ROOT/.env"; set +a; fi

KD_VENV_ROOT="${KD_VENV_ROOT:-$SFPO_ROOT/.venv}"
PYTHON_BIN="$KD_VENV_ROOT/bin/python"
export PATH="$KD_VENV_ROOT/bin:$PATH"
export PYTHONPATH="$SFPO_ROOT:$KD_VENV_ROOT/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_FLASHINFER_SAMPLER=0

GXPO_DATA="${GXPO_DATA_ROOT:-$SFPO_ROOT/data}"
BENCH_FILES=("$GXPO_DATA/math500/test.parquet" "$GXPO_DATA/aime2024/test.parquet"
  "$GXPO_DATA/aime2025/test.parquet" "$GXPO_DATA/amc/test.parquet"
  "$GXPO_DATA/minervamath/test.parquet" "$GXPO_DATA/olympiadbench/test.parquet")
STUDENT_BASE="${STUDENT_MODEL:-$REPO_ROOT/models/Qwen2.5-1.5B-Instruct}"
# Greedy full-bench eval costs ~30-60 min per checkpoint (minerva-heavy), so
# only every EVAL_EVERY-th step is evaluated (plus whatever is newest at the
# end). TEST_FREQ=5 val/loss already covers fine-grained tracking.
EVAL_EVERY="${EVAL_EVERY:-25}"

gpu_free() {
  # Echoes the index of a GPU with >80GB free and no trainer process, else empty.
  if pgrep -f "verl.trainer.kd_sft_trainer" >/dev/null 2>&1; then return 1; fi
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 > 80000) print $1}' | head -n 1
}

eval_step() {
  local step="$1" gpu="$2" ckpt="$RUN_DIR/global_step_$step" marker="$RUN_DIR/.eval_greedy_step${step}.done"
  [[ -d "$ckpt" && ! -e "$marker" ]] || return 1
  echo "[eval-watch] greedy eval step $step on GPU $gpu ..."
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/kd_sft/evaluate_greedy.py \
    --checkpoint-dir "$ckpt" --base-model "$STUDENT_BASE" \
    --data-files "${BENCH_FILES[@]}" --seeds 0 --step "$step" \
    --n 1 --temperature 0 --top-p 1.0 --max-tokens 3072 --tp 1 \
    --gpu-memory-utilization 0.85 --output "$RUN_DIR/eval_greedy_step${step}.json" 2>&1 | tee "$RUN_DIR/eval_greedy_step${step}.log"
  touch "$marker"
  echo "[eval-watch] step $step done."
  return 0
}

cd "$SFPO_ROOT"
pass_once() {
  # Sets DID_EVAL=1 when a checkpoint was evaluated this pass.
  DID_EVAL=0
  local did=0
  for ckpt in "$RUN_DIR"/global_step_*; do
    [[ -d "$ckpt" ]] || continue
    step="${ckpt##*global_step_}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    # Skip off-cadence epoch-end checkpoints (evaluated: every EVAL_EVERY steps).
    [[ $((10#$step % EVAL_EVERY)) -eq 0 ]] || continue
    [[ -e "$RUN_DIR/.eval_greedy_step${step}.done" ]] && continue
    gpu="$(gpu_free || true)"
    if [[ -z "$gpu" ]]; then echo "[eval-watch] no free GPU for step $step; waiting..."; DID_EVAL=0; return 0; fi
    eval_step "$step" "$gpu" && did=1
  done
  DID_EVAL=$did
  return 0
}

if [[ "$WATCH" -eq 1 ]]; then
  echo "[eval-watch] watching $RUN_DIR (Ctrl-C to stop) ..."
  idle_rounds=0
  while true; do
    pass_once
    if [[ "$DID_EVAL" -eq 0 ]] && ! pgrep -f "verl.trainer.kd_sft_trainer" >/dev/null 2>&1 \
       && ls "$RUN_DIR"/.eval_greedy_step*.done >/dev/null 2>&1; then
      idle_rounds=$((idle_rounds + 1))
      [[ "$idle_rounds" -ge 3 ]] && { echo "[eval-watch] trainer gone, all checkpoints evaluated; exiting."; break; }
    else
      idle_rounds=0
    fi
    sleep 60
  done
else
  pass_once
fi
