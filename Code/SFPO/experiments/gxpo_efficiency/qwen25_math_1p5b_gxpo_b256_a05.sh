#!/usr/bin/env bash
set -euo pipefail

GXPO_LOCAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export GXPO_LOCAL_ROOT

# Load checkout-local secrets (for example WANDB_API_KEY) without printing them.
if [[ -f "$GXPO_LOCAL_ROOT/.env" ]]; then
  set -a
  source "$GXPO_LOCAL_ROOT/.env"
  set +a
fi

# Keep every artifact, cache, checkpoint, model, dataset, and temporary file under this checkout.
export GXPO_DATA_ROOT="$GXPO_LOCAL_ROOT/Code/SFPO/data"
export GXPO_RESULTS_ROOT="$GXPO_LOCAL_ROOT/results/gxpo_efficiency"
export MODEL_QWEN25_MATH_1P5B="$GXPO_LOCAL_ROOT/models/Qwen2.5-Math-1.5B-Instruct"
export HF_HOME="$GXPO_LOCAL_ROOT/.hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
unset TRANSFORMERS_CACHE
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning:transformers.utils.hub}"
export XDG_CACHE_HOME="$GXPO_LOCAL_ROOT/.cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
# Triton compiles a tiny Python/CUDA helper during FlashInfer warm-up.
# Keep the Python development headers in this checkout instead of requiring a
# system-wide python3.12-dev installation.
GXPO_PYTHON_INCLUDE_DIR="$GXPO_LOCAL_ROOT/.python-dev/usr/include/python3.12"
if [[ -f "$GXPO_PYTHON_INCLUDE_DIR/Python.h" ]]; then
   export C_INCLUDE_PATH="$GXPO_LOCAL_ROOT/.python-dev/usr/include:$GXPO_PYTHON_INCLUDE_DIR${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
   export CPLUS_INCLUDE_PATH="$GXPO_LOCAL_ROOT/.python-dev/usr/include:$GXPO_PYTHON_INCLUDE_DIR${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
fi
# FlashInfer expects a conventional CUDA_HOME layout.
# This workspace-local shim links the pip toolkit bin/include/lib directories.
GXPO_CUDA_HOME="$GXPO_LOCAL_ROOT/.cuda-toolkit"
if [[ -x "$GXPO_CUDA_HOME/bin/nvcc" ]]; then
  export CUDA_HOME="$GXPO_CUDA_HOME"
  export CUDA_PATH="$GXPO_CUDA_HOME"
  export CUDACXX="$GXPO_CUDA_HOME/bin/nvcc"
  export PATH="$GXPO_CUDA_HOME/bin:$PATH"
fi
LOCAL_CUDA_LIB_ROOT="$GXPO_LOCAL_ROOT/.venv/lib/python3.12/site-packages/nvidia"
for LOCAL_CUDA_LIB_DIR in "$LOCAL_CUDA_LIB_ROOT/cu13/lib" "$LOCAL_CUDA_LIB_ROOT/cublas/lib" "$LOCAL_CUDA_LIB_ROOT/cuda_cupti/lib" "$LOCAL_CUDA_LIB_ROOT/cuda_nvrtc/lib" "$LOCAL_CUDA_LIB_ROOT/cuda_runtime/lib" "$LOCAL_CUDA_LIB_ROOT/cudnn/lib" "$LOCAL_CUDA_LIB_ROOT/cufft/lib" "$LOCAL_CUDA_LIB_ROOT/curand/lib" "$LOCAL_CUDA_LIB_ROOT/cusolver/lib" "$LOCAL_CUDA_LIB_ROOT/cusparse/lib" "$LOCAL_CUDA_LIB_ROOT/nccl/lib" "$LOCAL_CUDA_LIB_ROOT/nvjitlink/lib" "$LOCAL_CUDA_LIB_ROOT/nvshmem/lib" "$LOCAL_CUDA_LIB_ROOT/nvtx/lib"; do
  if [[ -d "$LOCAL_CUDA_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="$LOCAL_CUDA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done
export WANDB_CACHE_DIR="$GXPO_LOCAL_ROOT/.wandb/cache"
export WANDB_CONFIG_DIR="$GXPO_LOCAL_ROOT/.wandb/config"
export WANDB_DATA_DIR="$GXPO_LOCAL_ROOT/.wandb/data"
export WANDB_ARTIFACT_DIR="$GXPO_LOCAL_ROOT/.wandb/artifacts"
# Portable scratch dir: prefer GXPO_TMPDIR, then an existing inherited TMPDIR,
# then the original H200 mount, then a fresh mktemp dir. The old unconditional
# export + mkdir under set -euo pipefail killed startup on machines without
# /office/dev_workspace mounted (audit: revision_config_audit.md finding 5).
if [[ -n "${GXPO_TMPDIR:-}" ]]; then
  export TMPDIR="$GXPO_TMPDIR"
elif [[ -z "${TMPDIR:-}" || ! -w "$(dirname "${TMPDIR:-/nonexistent}")" ]]; then
  TMPDIR="/office/dev_workspace/swapnil/.gxpo-tmp"
fi
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
SWAPNIL_ROOT="$(cd -- "$GXPO_LOCAL_ROOT/.." && pwd)"
export SWAPNIL_ROOT
export RAY_TMPDIR="${RAY_TMPDIR:-$SWAPNIL_ROOT/.gxpo-ray}"
export RAY_AIR_LOCAL_CACHE_DIR="${RAY_AIR_LOCAL_CACHE_DIR:-$SWAPNIL_ROOT/.gxpo-ray-air}"
mkdir -p "$GXPO_DATA_ROOT" "$GXPO_RESULTS_ROOT" "$GXPO_LOCAL_ROOT/models" \
  "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
  "$XDG_CACHE_HOME" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" \
  "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$RAY_TMPDIR" \
  "$RAY_AIR_LOCAL_CACHE_DIR" || true   # individual failures surface at use time
if ! mkdir -p "$TMPDIR" 2>/dev/null; then
  # last-resort fallback: never let scratch-dir creation abort the launch
  TMPDIR="$(mktemp -d)"
  export TMPDIR TMP="$TMPDIR" TEMP="$TMPDIR"
fi

# Dedicated 1.5B GXPO run: global train batch 256, k=5, reposition alpha 0.5.
# Keep the original qwen25_math_1p5b_gxpo_k10.sh unchanged for reproducibility.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
export REPOSITION_ALPHA=0.5
export K=5
export GXPO_TAU=1.0
export GXPO_TRIGGER_PATIENCE=1
export GXPO_FALLBACK_MODE=permanent
export GXPO_WARMUP_STEPS=0
export GXPO_ZSCORE_W=30
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-8}"
export FSDP_SIZE=2
export USE_LIGER=True
export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
export GPU_IDS="${GPU_IDS:-0,1}"
export GPU_COUNT="${GPU_COUNT:-2}"
export SAVE_FREQ="${SAVE_FREQ:-20}"
export MAX_STEPS="${MAX_STEPS:-400}"
export WANDB_PROJECT="${WANDB_PROJECT:-gxpo-efficiency-final}"
export WANDB_GROUP="${WANDB_GROUP:-qwen25-math-1p5b-b256}"
export WANDB_TAGS="${WANDB_TAGS:-model:qwen25-math-1p5b,method:gxpo,k:5,alpha:0.5,batch:256,minibatch:64,experiment:custom}"
export WANDB_MODE="${WANDB_MODE:-online}"
export GXPO_RUN_NAME="${GXPO_RUN_NAME:-qwen25_math_1p5b_gxpo_k5_a05_perm_b256_mb64_fsdp2_fp32_liger_v5_20260824}"
export GXPO_CONCISE_LOGS=1
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export ACTOR_MODEL_DTYPE=float32

# A 256-prompt batch with rollout.n=8 creates 2048 responses, split evenly
# across the two rollout ranks. Match the per-rank sequence cap to that share
# so vLLM keeps enough KV-cache blocks for active generations without admitting
# an unnecessarily large concurrent queue.
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1024}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-98304}"
export VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-True}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export VLLM_SLEEP_LEVEL="${VLLM_SLEEP_LEVEL:-2}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.7}"

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen25_math_1p5b_gxpo_k10.sh"
