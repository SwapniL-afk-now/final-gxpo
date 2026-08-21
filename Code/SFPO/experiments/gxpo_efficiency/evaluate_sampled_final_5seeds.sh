#!/usr/bin/env bash
set -euo pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="$(cd "$CODE/../.." && pwd)"
OUT="${SAMPLED_EVAL_OUT:-$CODE/results/gxpo_efficiency/sampled_final_16_5seeds_20260820}"
BASE_MODEL="${MODEL_QWEN25_MATH_1P5B:-$CODE/models/Qwen2.5-Math-1.5B-Instruct}"
GRPO_STEP="${GRPO_EVAL_STEP:-400}"
SFPO_STEP="${SFPO_EVAL_STEP:-400}"
GXPO_STEP="${GXPO_EVAL_STEP:-400}"
read -r -a EVAL_SEED_LIST <<< "${EVAL_SEEDS:-0 1 2 3 4}"
read -r -a EVAL_METHOD_LIST <<< "${EVAL_METHODS:-grpo sfpo gxpo}"

set -a
. "$REPO/.env"
set +a

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$CODE/.runtime_deps:/workspace/.gxpo_pydeps:$CODE"
export HF_HOME="${HF_HOME:-$CODE/.hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export VLLM_CACHE_ROOT="$OUT/vllm_cache"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$OUT/flashinfer_autotune_cache"
mkdir -p "$OUT"

if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo "GPU is not free; refusing to start sampled evaluation" >&2
  nvidia-smi
  exit 2
fi

DATA=(
  "$CODE/data/math500/test.parquet"
  "$CODE/data/aime2024/test.parquet"
  "$CODE/data/aime2025/test.parquet"
  "$CODE/data/amc/test.parquet"
  "$CODE/data/minervamath/test.parquet"
  "$CODE/data/olympiadbench/test.parquet"
)

exec /venv/main/bin/python -u "$CODE/tools/evaluate_sampled_5seeds.py" \
  --base-model "$BASE_MODEL" \
  --data-files "${DATA[@]}" \
  --seeds "${EVAL_SEED_LIST[@]}" \
  --methods "${EVAL_METHOD_LIST[@]}" \
  --output-dir "$OUT" \
  --grpo-repo "${GRPO_EVAL_HF_REPO:-swapnil7777/learn-to-predict}" \
  --grpo-step "$GRPO_STEP" \
  --sfpo-run-dir "$CODE/results/gxpo_efficiency/qwen25_math_1p5b_sfpo_k10_a03_b64_mb16_w50_v2_20260819" \
  --sfpo-step "$SFPO_STEP" \
  --gxpo-run-dir "$CODE/results/gxpo_efficiency/qwen25_math_1p5b_gxpo_k10_a03_b64_mb16_w50_tau3_v2_20260819" \
  --gxpo-step "$GXPO_STEP" \
  --n "${EVAL_N:-16}" \
  --temperature "${EVAL_TEMPERATURE:-1.0}" \
  --top-p "${EVAL_TOP_P:-1.0}" \
  --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-tokens "${EVAL_MAX_TOKENS:-3072}"
