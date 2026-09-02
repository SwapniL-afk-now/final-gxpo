#!/usr/bin/env bash
# Prepare verified teacher trajectories, then launch the two student jobs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE=/workspace/.env
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
if [[ -v HF_API_KEY && -n "$HF_API_KEY" ]]; then
  if [[ ! -v HF_TOKEN || -z "$HF_TOKEN" ]]; then export HF_TOKEN="$HF_API_KEY"; fi
  if [[ ! -v HUGGINGFACE_HUB_TOKEN || -z "$HUGGINGFACE_HUB_TOKEN" ]]; then export HUGGINGFACE_HUB_TOKEN="$HF_API_KEY"; fi
fi
if [[ -v PYTHONPATH && -n "$PYTHONPATH" ]]; then export PYTHONPATH="$CODE_ROOT:$PYTHONPATH"; else export PYTHONPATH="$CODE_ROOT"; fi
export WANDB_MODE=online
export WANDB_PROJECT=code-distillation
export WANDB_RUN_GROUP=code_distill_fsdppartition
export HF_HOME=/workspace/.hf_home
export PATH="/workspace/gradient-extrapolation-based-policy-optimization/.venv-h200/bin:$PATH"
PYTHON_BIN=/workspace/gradient-extrapolation-based-policy-optimization/.venv-h200/bin/python
DATA_ROOT=/workspace/jepa-grpo-cache/data/code_distill
EVAL_ROOT=/workspace/jepa-grpo-cache/eval_data/code_distill
LOG_ROOT="$CODE_ROOT/runs/code_distill_queue"
mkdir -p "$LOG_ROOT"

"$CODE_ROOT/train-scripts/run_code_teacher_generation.sh" 2>&1 | tee -a "$LOG_ROOT/teacher_generation.log"
"$PYTHON_BIN" "$CODE_ROOT/tools/prepare_code_eval_sources.py" --output-dir "$EVAL_ROOT" 2>&1 | tee -a "$LOG_ROOT/eval_sources.log"
"$PYTHON_BIN" "$CODE_ROOT/tools/prepare_code_eval_manifest.py" \
  --training-manifest "$DATA_ROOT/manifest.json" \
  --output-dir "$EVAL_ROOT" \
  --humanevalplus "$EVAL_ROOT/raw/humanevalplus.parquet" \
  --mbppplus "$EVAL_ROOT/raw/mbppplus.parquet" \
  --livecodebench "$EVAL_ROOT/raw/livecodebench.parquet" \
  --livecodebench-after 2024-07-31 2>&1 | tee -a "$LOG_ROOT/eval_manifest.log"

if tmux has-session -t code-distill-adamw-01 2>/dev/null; then
  echo "code-distill-adamw-01 already exists; refusing duplicate launch" >&2
  exit 3
fi
if tmux has-session -t code-distill-gxpo-23 2>/dev/null; then
  echo "code-distill-gxpo-23 already exists; refusing duplicate launch" >&2
  exit 3
fi

mkdir -p "$CODE_ROOT/runs/code_distill_adamw_gpus01" \
         "$CODE_ROOT/runs/code_distill_gxpo_gpus23"

tmux new-session -d -s code-distill-adamw-01 \
  "cd '$CODE_ROOT' && source /workspace/gradient-extrapolation-based-policy-optimization/.venv-h200/bin/activate && GPU_IDS=0,1 GPU_COUNT=2 FSDP_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 WANDB_RUN_NAME=code_distill_adamw_gpus01 CODE_DISTILL_RUN_ROOT='$CODE_ROOT/runs/code_distill_adamw_gpus01' '$CODE_ROOT/train-scripts/run_code_distill_sft_adamw_fsdp.sh' 2>&1 | tee -a '$CODE_ROOT/runs/code_distill_adamw_gpus01/tmux.log'"
tmux new-session -d -s code-distill-gxpo-23 \
  "cd '$CODE_ROOT' && source /workspace/gradient-extrapolation-based-policy-optimization/.venv-h200/bin/activate && GPU_IDS=2,3 GPU_COUNT=2 FSDP_SIZE=2 CUDA_VISIBLE_DEVICES=2,3 WANDB_RUN_NAME=code_distill_gxpo_gpus23 CODE_DISTILL_RUN_ROOT='$CODE_ROOT/runs/code_distill_gxpo_gpus23' '$CODE_ROOT/train-scripts/run_code_distill_gxpo_sft_fsdp.sh' 2>&1 | tee -a '$CODE_ROOT/runs/code_distill_gxpo_gpus23/tmux.log'"
echo "student sessions started: code-distill-adamw-01 and code-distill-gxpo-23"
