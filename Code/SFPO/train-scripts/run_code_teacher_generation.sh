#!/usr/bin/env bash
# Offline Qwen2.5-Coder-7B teacher trajectory generation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$CODE_ROOT/.." && pwd)"
cd "$CODE_ROOT"
ENV_FILE="${ENV_FILE:-/workspace/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ENV_FILE}"
  set +a
fi
if [[ -n "${HF_API_KEY:-}" ]]; then
  export HF_TOKEN="${HF_TOKEN:-$HF_API_KEY}"
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_API_KEY}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
OUT_DIR="${CODE_DISTILL_ROOT:-/workspace/jepa-grpo-cache/data/code_distill}"
MODEL="${MODEL_QWEN25_CODER_7B:-Qwen/Qwen2.5-Coder-7B-Instruct}"
STUDENT_MODEL="${MODEL_QWEN25_CODER_1P5B:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/gradient-extrapolation-based-policy-optimization/.venv-h200/bin/python}"
PROMPTS="$OUT_DIR/teacher_prompts.parquet"
TRAJECTORIES="$OUT_DIR/teacher_trajectories.jsonl"
CANDIDATES="$OUT_DIR/teacher_candidates.jsonl"
KD_TOPK="${KD_TOPK:-32}"
TEACHER_TP="${TEACHER_TP:-4}"

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'teacher=%s\nstudent=%s\noutput=%s\ntensor_parallel_size=%s\ntemperature=1.0 top_p=1.0 max_tokens=3072\nKD=offline_top%s score_only=%s\n' \
    "$MODEL" "$STUDENT_MODEL" "$OUT_DIR" "$TEACHER_TP" "$KD_TOPK" "${KD_SCORE_ONLY:-0}"
  exit 0
fi

if [[ "${KD_SCORE_ONLY:-0}" != "1" ]]; then
  "$PYTHON_BIN" tools/prepare_code_distillation.py --output-dir "$OUT_DIR"
  if [[ "${TEACHER_VERIFY_ONLY:-0}" == "1" ]]; then
    [[ -s "$CANDIDATES" ]] || { echo "Missing saved teacher candidates: $CANDIDATES" >&2; exit 2; }
    "$PYTHON_BIN" tools/generate_code_teacher_trajectories.py \
      --input "$PROMPTS" --output "$TRAJECTORIES" --candidates-output "$CANDIDATES" \
      --verify-only --verifier-workers "${VERIFIER_WORKERS:-64}"
  else
    "$PYTHON_BIN" tools/generate_code_teacher_trajectories.py \
      --input "$PROMPTS" --output "$TRAJECTORIES" --model "$MODEL" \
      --num-samples 8 --temperature 1.0 --top-p 1.0 --max-tokens 3072 \
      --tensor-parallel-size "$TEACHER_TP" \
      --candidates-output "$CANDIDATES" \
      --verifier-workers "${VERIFIER_WORKERS:-64}"
  fi
  "$PYTHON_BIN" tools/prepare_code_distillation.py \
    --output-dir "$OUT_DIR" --teacher-jsonl "$TRAJECTORIES"
else
  for required in "$OUT_DIR/teacher_sft_train.parquet" "$OUT_DIR/teacher_sft_val.parquet"; do
    [[ -s "$required" ]] || { echo "Missing reusable response corpus: $required" >&2; exit 2; }
  done
fi

KD_OVERWRITE_ARGS=()
if [[ "${KD_SCORE_OVERWRITE:-0}" == "1" ]]; then
  KD_OVERWRITE_ARGS=(--overwrite)
fi
"$PYTHON_BIN" tools/score_code_teacher_distributions.py \
  --train-input "$OUT_DIR/teacher_sft_train.parquet" \
  --val-input "$OUT_DIR/teacher_sft_val.parquet" \
  --output-dir "$OUT_DIR" \
  --teacher-model "$MODEL" --student-tokenizer "$STUDENT_MODEL" \
  --topk "$KD_TOPK" --max-length 4096 \
  --tensor-parallel-size "$TEACHER_TP" \
  --gpu-memory-utilization "${TEACHER_GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-num-batched-tokens "${TEACHER_MAX_BATCHED_TOKENS:-32768}" \
  --max-num-seqs "${TEACHER_MAX_NUM_SEQS:-128}" \
  --request-batch-size "${KD_REQUEST_BATCH_SIZE:-128}" \
  "${KD_OVERWRITE_ARGS[@]}"
