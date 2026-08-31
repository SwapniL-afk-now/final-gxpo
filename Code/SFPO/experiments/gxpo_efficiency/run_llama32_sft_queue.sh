#!/usr/bin/env bash
# Sequential SFT audit queue: run each arm across GPUs 0 and 1; start GXPO only after baseline completes.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
BASELINE="$SCRIPT_DIR/llama32_3b_sft_baseline_fsdp2.sh"
GXPO="$SCRIPT_DIR/llama32_3b_sft_gxpo_fsdp2.sh"
SFT_TOTAL_STEPS="${SFT_TOTAL_STEPS:-400}"
if [[ "${1:-}" == "--smoke" ]]; then
  SFT_TOTAL_STEPS=2
  shift
fi
if [[ "${1:-}" == "--steps" ]]; then
  SFT_TOTAL_STEPS="${2:?--steps requires a positive integer}"
  shift 2
fi
if ! [[ "$SFT_TOTAL_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SFT_TOTAL_STEPS must be a positive integer; got $SFT_TOTAL_STEPS" >&2
  exit 2
fi
export SFT_TOTAL_STEPS
export SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-10}"
export SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-32}"
export SFT_MICRO_BATCH_SIZE="${SFT_MICRO_BATCH_SIZE:-8}"
export SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-2048}"
export SFT_GXPO_K="${SFT_GXPO_K:-3}"
export SFT_GXPO_ALPHA="${SFT_GXPO_ALPHA:-0.8}"
export SFT_GREEDY_EVAL_FREQ="${SFT_GREEDY_EVAL_FREQ:-5}"
export SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-5}"
export SFT_ATTN_IMPL="${SFT_ATTN_IMPL:-flash_attention_2}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export SFT_EVAL_KIND="${SFT_EVAL_KIND:-knights_and_knaves}"
export SFT_WANDB_PROJECT="${SFT_WANDB_PROJECT:-gxpo-efficiency-final}"
export SFT_WANDB_GROUP="${SFT_WANDB_GROUP:-llama32-3b-knights-and-knaves-b32-fsdp2}"
export WANDB_MODE="${WANDB_MODE:-online}"
export SFT_RESULTS_ROOT="${SFT_RESULTS_ROOT:-$REPO_ROOT/results/gxpo_efficiency/llama32_3b_sft_fsdp2}"
export SFT_SMOKE_SUFFIX="${SFT_SMOKE_SUFFIX:-}"
if [[ "$SFT_TOTAL_STEPS" -le 2 && -z "$SFT_SMOKE_SUFFIX" ]]; then
  export SFT_SMOKE_SUFFIX="_smoke2"
fi
export WANDB_TAGS="${WANDB_TAGS:-model:llama3.2-3b,dataset:knights-and-knaves,framework:sft,fsdp:fsdp2,optimizer:adamw,batch:32,microbatch:8}"
export SFT_RUN_NAME="llama32_3b_sft_baseline_knights_and_knaves_fsdp2_b32_seed42${SFT_SMOKE_SUFFIX}"
echo "[queue] starting baseline: steps=$SFT_TOTAL_STEPS gpus=0,1"
GPU_IDS=0,1 CUDA_VISIBLE_DEVICES=0,1 GPU_COUNT=2 bash "$BASELINE"
echo "[queue] baseline completed successfully; starting GXPO on gpus=0,1"
export SFT_RUN_NAME="llama32_3b_sft_gxpo_knights_and_knaves_fsdp2_k${SFT_GXPO_K}_a${SFT_GXPO_ALPHA}_b32_seed42${SFT_SMOKE_SUFFIX}"
GPU_IDS=0,1 CUDA_VISIBLE_DEVICES=0,1 GPU_COUNT=2 bash "$GXPO"
echo "[queue] baseline and GXPO completed successfully"
