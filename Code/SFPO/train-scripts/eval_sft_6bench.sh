#!/usr/bin/env bash
# Post-training 6-benchmark vLLM eval for the SFT pair (baseline + GXPO):
# amc23, aime24, aime25, math500, olympiadbench, minervamath.
#
# Same harness the KD/RL arms use (tools/evaluate_greedy_5seeds.py for greedy
# pass@1 + tools/evaluate_sampled_5seeds.py for sampled pass@n/average@n:
# vLLM LLM.generate, verl scorer, same JSON schema) via
# train-scripts/eval_sft_ckpt_vllm.py -- only the decoding flags are the SFT
# pair's locked metrics: n=8, seeds 0,1,2, temp 0.6, top_p 0.95.
#
# Data files default to the same $CODE/data/.../test.parquet paths the KD/RL
# eval scripts use. Set EVAL_DATA_ROOT to fall back to the legacy
# <root>/{amc23,aime24,aime25,math500,olympiadbench,minervamath}.parquet layout,
# or override any benchmark individually.
#
# Usage: GPU=0 ./eval_sft_6bench.sh <hf_checkpoint_dir>
#   (e.g. ./runs/sft_baseline_dapoossmed_seed42/global_step_200)
set -euo pipefail

export RAY_ADDRESS=local   # force an isolated Ray cluster per job -- unaddressed ray.init() auto-attaches to any existing local cluster (via /tmp/ray/session_latest), starving concurrent GPU0/GPU1 jobs of GPUs ("Total available GPUs 0")

GPU="${GPU:?set GPU=0|1}"
CKPT="${1:?usage: GPU=0 $0 <hf_checkpoint_dir>}"

CODE="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -n "${EVAL_DATA_ROOT:-}" ]]; then
  AMC23="${AMC23:-$EVAL_DATA_ROOT/amc23.parquet}"
  AIME24="${AIME24:-$EVAL_DATA_ROOT/aime24.parquet}"
  AIME25="${AIME25:-$EVAL_DATA_ROOT/aime25.parquet}"
  MATH500="${MATH500:-$EVAL_DATA_ROOT/math500.parquet}"
  OLYMPIAD="${OLYMPIAD:-$EVAL_DATA_ROOT/olympiadbench.parquet}"
  MINERVA="${MINERVA:-$EVAL_DATA_ROOT/minervamath.parquet}"
else
  MATH500="${MATH500:-$CODE/data/math500/test.parquet}"
  AIME24="${AIME24:-$CODE/data/aime2024/test.parquet}"
  AIME25="${AIME25:-$CODE/data/aime2025/test.parquet}"
  AMC23="${AMC23:-$CODE/data/amc/test.parquet}"
  MINERVA="${MINERVA:-$CODE/data/minervamath/test.parquet}"
  OLYMPIAD="${OLYMPIAD:-$CODE/data/olympiadbench/test.parquet}"
fi

N="${EVAL_N:-8}"
SEEDS="${EVAL_SEEDS:-0,1,2}"
TEMP="${EVAL_TEMP:-0.7}"
TOP_P="${EVAL_TOP_P:-0.95}"
GPU_UTIL="${EVAL_GPU_UTIL:-0.85}"
# Training responses are now capped at 3072 tokens, so generation only needs
# to cover that plus prompt headroom -- a short explicit MAX_MODEL_LEN (not
# the model's full native context) is what actually buys the throughput, by
# letting vLLM fit far more concurrent sequences in the same KV cache budget.
MAX_TOKENS="${EVAL_RESPONSE_LENGTH:-3072}"
MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${EVAL_MAX_NUM_BATCHED_TOKENS:-65536}"

missing=0
for required in "$MATH500" "$AIME24" "$AIME25" "$AMC23" "$MINERVA" "$OLYMPIAD"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing benchmark dataset: $required" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "Set AMC23/AIME24/AIME25/MATH500/OLYMPIAD/MINERVA (or EVAL_DATA_ROOT) to the 6 benchmark parquets." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$CODE"

OUT="${EVAL_OUTPUT_DIR:-./eval-results/6bench/$(basename "${CKPT%/}")}"
mkdir -p "$OUT"

SKIP_GREEDY_FLAG=()
if [[ "${SFT_VLLM_EVAL_SKIP_GREEDY:-0}" == "1" ]]; then
  SKIP_GREEDY_FLAG+=(--skip-greedy)
fi

if [[ "$GPU" == *,* ]]; then
  # Independent TP=1 workers let every listed GPU evaluate concurrently.
  PYTHONPATH="$CODE" python -u tools/kd_sft/evaluate_greedy.py \
      --checkpoint-dir "$CKPT" \
      --base-model "${EVAL_BASE_MODEL:-$CKPT}" \
      --data-files "$MATH500" "$AIME24" "$AIME25" "$AMC23" "$MINERVA" "$OLYMPIAD" \
      --seed "${SEEDS%%,*}" \
      --n "$N" \
      --temperature "$TEMP" \
      --top-p "$TOP_P" \
      --max-tokens "$MAX_TOKENS" \
      --tp 1 \
      --gpu-devices "$GPU" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --attention-backend "${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}" \
      --sft-output-dir "$OUT" \
      --output "$OUT/eval_pass${N}_6bench.json" \
      2>&1 | tee "$OUT/eval.log"
  exit
fi

PYTHONPATH="$CODE" python -u train-scripts/eval_sft_ckpt_vllm.py \
    --ckpt "$CKPT" \
    --data-files "$MATH500" "$AIME24" "$AIME25" "$AMC23" "$MINERVA" "$OLYMPIAD" \
    --seeds ${SEEDS//,/ } \
    --n "$N" \
    --temperature "$TEMP" \
    --top-p "$TOP_P" \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --output-dir "$OUT" \
    "${SKIP_GREEDY_FLAG[@]}" \
    2>&1 | tee "$OUT/eval.log"
