#!/usr/bin/env bash
# General post-training evaluation matching examples/tafr_grpo/run_repro_eval.sh.
# Usage: GPU=1 ./run_general_eval.sh /path/to/model [output_dir]
set -euo pipefail

MODEL="${1:?usage: GPU=1 $0 MODEL [OUTPUT_DIR]}"
OUT="${2:-./eval-results/$(basename "${MODEL%/}")}"
GPU="${GPU:?set GPU=0|1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_ROOT="/workspace/Joint-Embedding-Guided-Policy-Optimization"
PY="/workspace/Joint-Embedding-Guided-Policy-Optimization/.venv/bin/python"

export CUDA_VISIBLE_DEVICES="$GPU"
export RAY_ADDRESS=local
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$OUT"

# Match GSPO/repro evaluation: greedy n=1, top-p 1.0, max 3072 tokens.
COMMON=(--model "$MODEL" --n 1 --temperature 0 --top-p 1.0 --max-tokens 3072 --gpu-memory-utilization "${GPU_UTIL:-0.9}")
MATH_CORE=math500,amc23,aime24,aime25,aime26
MATH_EXTRA=olympiadbench,minervamath
CODE=humanevalplus,mbppplus,livecodebench
SEEDS_CORE="${SEEDS_CORE:-3407,31415,27182,16180,42}"
SEEDS_ALL="${SEEDS_ALL:-3407,31415,27182,16180,42}"

cd "$ROOT"
echo "=== core math (seeds $SEEDS_CORE) ==="
"$PY" "$EVAL_ROOT/eval_math_multiseed_batched.py" "${COMMON[@]}" --benchmarks "$MATH_CORE" --seeds "$SEEDS_CORE" --out "$OUT/math_core.json"
echo "=== extra math (seeds $SEEDS_ALL) ==="
"$PY" "$EVAL_ROOT/eval_math_multiseed_batched.py" "${COMMON[@]}" --benchmarks "$MATH_EXTRA" --seeds "$SEEDS_ALL" --out "$OUT/math_extra.json"
echo "=== code (seeds $SEEDS_ALL) ==="
"$PY" "$EVAL_ROOT/eval_code_multiseed_batched.py" "${COMMON[@]}" --benchmarks "$CODE" --seeds "$SEEDS_ALL" --out "$OUT/code.json"
echo "=== done: $OUT ==="
