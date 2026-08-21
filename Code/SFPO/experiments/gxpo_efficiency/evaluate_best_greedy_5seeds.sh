#!/usr/bin/env bash
set -euo pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="$(cd "$CODE/../.." && pwd)"
OUT="${GREEDY_EVAL_OUT:-$CODE/results/gxpo_efficiency/greedy_best_5seeds_20260820}"
BASE_MODEL="${MODEL_QWEN25_MATH_1P5B:-$CODE/models/Qwen2.5-Math-1.5B-Instruct}"
GRPO_STEP="${GRPO_EVAL_STEP:-295}"
SFPO_STEP="${SFPO_EVAL_STEP:-190}"
GXPO_STEP="${GXPO_EVAL_STEP:-190}"

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
  echo "GPU is not free; refusing to start evaluation" >&2
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

exec /venv/main/bin/python -u "$CODE/tools/evaluate_greedy_5seeds.py" \
  --base-model "$BASE_MODEL" \
  --data-files "${DATA[@]}" \
  --seeds 0 1 2 3 4 \
  --output-dir "$OUT" \
  --grpo-repo "${GRPO_EVAL_HF_REPO:-swapnil7777/learn-to-predict}" \
  --grpo-step "$GRPO_STEP" \
  --sfpo-run-dir "$CODE/results/gxpo_efficiency/qwen25_math_1p5b_sfpo_k10_a03_b64_mb16_w50_v2_20260819" \
  --sfpo-step "$SFPO_STEP" \
  --gxpo-run-dir "$CODE/results/gxpo_efficiency/qwen25_math_1p5b_gxpo_k10_a03_b64_mb16_w50_tau3_v2_20260819" \
  --gxpo-step "$GXPO_STEP" \
  --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-tokens 3072
