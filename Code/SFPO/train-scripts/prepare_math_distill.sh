#!/usr/bin/env bash
# Build stored DeepSeek-R1 math SFT data, then score traces with teacher logits.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
ENV_FILE="${ENV_FILE:-/workspace/.env}"
if [[ -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; fi
if [[ -n "${HF_API_KEY:-}" ]]; then
  export HF_TOKEN="${HF_TOKEN:-$HF_API_KEY}"
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_API_KEY}"
fi

PYTHON_BIN="${PYTHON_BIN:-/workspace/gradient-extrapolation-based-policy-optimization/.venv-h200/bin/python}"
DATA_ROOT="${MATH_DISTILL_ROOT:-/workspace/jepa-grpo-cache/data/math_distill_r1_7b}"
TEACHER="${MATH_TEACHER_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
STUDENT="${MATH_STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TOPK="${KD_TOPK:-32}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
TRAIN_SIZE="${MATH_TRAIN_SIZE:-63978}"

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'dataset=wangx0t/numina-deepseek-DeepSeek-R1-Distill-Qwen-7B\nteacher=%s\nstudent=%s\noutput=%s\ntrain=%s val=2000 seed=%s\nKD=prompt-logprobs-only topk=%s max_length=%s\n' \
    "$TEACHER" "$STUDENT" "$DATA_ROOT" "$TRAIN_SIZE" "${SPLIT_SEED:-42}" "$TOPK" "$MAX_LENGTH"
  exit 0
fi

"$PYTHON_BIN" tools/prepare_math_distillation.py \
  --output-dir "$DATA_ROOT" --teacher-model "$TEACHER" \
  --train-size "$TRAIN_SIZE" --val-size 2000 --seed "${SPLIT_SEED:-42}"

# max_tokens=1 plus prompt_logprobs scores stored responses; it never samples
# teacher responses.
"$PYTHON_BIN" tools/score_math_teacher_distributions.py \
  --train-input "$DATA_ROOT/math_r1_train.parquet" \
  --val-input "$DATA_ROOT/math_r1_val.parquet" \
  --output-dir "$DATA_ROOT" --teacher-model "$TEACHER" \
  --student-tokenizer "$STUDENT" --topk "$TOPK" --max-length "$MAX_LENGTH" \
  --tensor-parallel-size "${TEACHER_TP:-4}" \
  --gpu-memory-utilization "${TEACHER_GPU_UTIL:-0.85}" --overwrite
