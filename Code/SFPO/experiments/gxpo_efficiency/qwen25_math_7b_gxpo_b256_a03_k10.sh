#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
# The training dependencies live in the shared project environment on this instance.
GXPO_RUNTIME_ROOT="${GXPO_RUNTIME_ROOT:-/workspace/gradient-extrapolation-based-policy-optimization}"
if [[ -x "$GXPO_RUNTIME_ROOT/.venv-h200/bin/python" ]]; then
  export VIRTUAL_ENV="$GXPO_RUNTIME_ROOT/.venv-h200"
  export PATH="$GXPO_RUNTIME_ROOT/.venv-h200/bin:$PATH"
fi
export GXPO_DATA_ROOT="${GXPO_DATA_ROOT:-/workspace/data}"
ENV_FILE="${GXPO_ENV_FILE:-/workspace/.env}"
if [[ ! -f "$ENV_FILE" && -f "$REPO_ROOT/.env" ]]; then ENV_FILE="$REPO_ROOT/.env"; fi
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi
if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE:-online}" != "offline" ]]; then
  echo "WANDB_API_KEY was not found in $ENV_FILE; refusing online launch." >&2
  exit 2
fi

# Qwen2.5-Math-7B-Instruct | GXPO | batch 256 | PPO minibatch 64 | K=10 | alpha=0.3
# Standalone Qwen 7B GXPO entrypoint. It defines the model-specific
# configuration and invokes only the shared training implementation.

MODEL_ALIAS="qwen2.5-math-7b-instruct"
MODEL_ID="${MODEL_QWEN25_MATH_7B:-Qwen/Qwen2.5-Math-7B-Instruct}"
METHOD="gxpo"

export MODEL_QWEN25_MATH_7B="$MODEL_ID"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export K="${K:-10}"
export REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.3}"
export GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"
# Use the requested SFPO-style entropy trigger for the queued 7B run.
# Entropy is scored with a 30-step rolling z-score window.
export GXPO_TRIGGER_SIGNAL="${GXPO_TRIGGER_SIGNAL:-entropy}"
export GXPO_SHUTOFF_MODE="${GXPO_SHUTOFF_MODE:-trajectory_aware}"
export GXPO_TRIGGER_ABS_THRESHOLD="${GXPO_TRIGGER_ABS_THRESHOLD:-0}"
export GXPO_TRIGGER_ROBUST="${GXPO_TRIGGER_ROBUST:-0}"
export GXPO_TAU="${GXPO_TAU:-1.5}"
export GXPO_ZSCORE_W="${GXPO_ZSCORE_W:-30}"
export GXPO_TRIGGER_PATIENCE="${GXPO_TRIGGER_PATIENCE:-2}"
export GXPO_RESET_ENTROPY_AFTER_WARMUP="${GXPO_RESET_ENTROPY_AFTER_WARMUP:-False}"
export GXPO_TRIGGER_MIN_OBS="${GXPO_TRIGGER_MIN_OBS:-0}"
export GXPO_MAX_ACTIVE_STEPS="${GXPO_MAX_ACTIVE_STEPS:-150}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export GPU_COUNT="${GPU_COUNT:-4}"
export FSDP_SIZE="${FSDP_SIZE:-4}"
export ATTN_IMPL="${ATTN_IMPL:-flash_attention_3}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-8}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-98304}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1024}"
# Keep this conservative for the 7B actor; 0.65 is suitable for the 1.5B run
# but was not safe for the 7B model's memory footprint.
export VLLM_GPU_MEMORY_UTILIZATION="0.48"
export WANDB_PROJECT="${WANDB_PROJECT:-gxpo-efficiency-final}"
export WANDB_GROUP="${WANDB_GROUP:-qwen2.5-math-7b-instruct-gxpo}"
export GXPO_RUN_NAME="${GXPO_RUN_NAME:-qwen25_math_7b_instruct_gxpo_b256_mb64_a0.3_k10_entropy_txopt_tau1.5_p2_m150_tp10_v048}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
