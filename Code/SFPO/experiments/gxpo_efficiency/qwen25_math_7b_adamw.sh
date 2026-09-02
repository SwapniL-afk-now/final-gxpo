#!/usr/bin/env bash
# Self-contained Qwen2.5-Math-7B-Instruct GXPO + AdamW launcher.
# Configuration mirrors qwen25_math_7b_gxpo_b256_a03_k10.sh; only the
# optimizer changes. GXPO_TAU is the entropy-gate threshold.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
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

MODEL_ALIAS="qwen2.5-math-7b-instruct-gxpo-adamw"
MODEL_ID="${MODEL_QWEN25_MATH_7B:-Qwen/Qwen2.5-Math-7B-Instruct}"
METHOD="gxpo"

# Match the effective 7B GXPO launcher configuration.
export MODEL_QWEN25_MATH_7B="$MODEL_ID"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export K="${K:-10}"
export REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.3}"
export OPTIMIZER_NAME="adamw"
export GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"
export GXPO_TRIGGER_SIGNAL="entropy"
export GXPO_TAU="3"
export GXPO_ZSCORE_W="30"
export GXPO_TRIGGER_PATIENCE="2"
export GXPO_RESET_ENTROPY_AFTER_WARMUP="${GXPO_RESET_ENTROPY_AFTER_WARMUP:-False}"
export GXPO_SHUTOFF_MODE="${GXPO_SHUTOFF_MODE:-trajectory_aware}"
export GXPO_FALLBACK_MODE="permanent"
export GXPO_FALLBACK_WINDOW="${GXPO_FALLBACK_WINDOW:-10}"
export GXPO_WARMUP_STEPS="${GXPO_WARMUP_STEPS:-0}"
export GXPO_MAX_ACTIVE_STEPS="150"
export GXPO_DIAG_FREQ="${GXPO_DIAG_FREQ:-10}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
export ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"
export ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-True}"
export ACTOR_KL_LOSS_COEF="${ACTOR_KL_LOSS_COEF:-0.01}"
export ACTOR_KL_LOSS_TYPE="${ACTOR_KL_LOSS_TYPE:-low_var_kl}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export GPU_COUNT="${GPU_COUNT:-4}"
export FSDP_SIZE="${FSDP_SIZE:-4}"
export ATTN_IMPL="${ATTN_IMPL:-flash_attention_3}"
export USE_LIGER="${USE_LIGER:-True}"
export ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-True}"
export USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-True}"
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-8}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-512}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.40}"
export VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-True}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-False}"
export VLLM_FREE_CACHE_ENGINE="${VLLM_FREE_CACHE_ENGINE:-False}"
export MAX_STEPS="${MAX_STEPS:-400}"
# Full resumable checkpoints are ~85 GiB each for this 7B actor and take ~31s to
# write. At the old default of 5 that is 80 saves (~42 min of pure checkpoint
# I/O) across a 400-step run. Validation cadence is unaffected: TRAINER_TEST_FREQ
# still drives eval and best-checkpoint saves every 5 steps.
export SAVE_FREQ="${SAVE_FREQ:-25}"
export TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-5}"
export WANDB_PROJECT="${WANDB_PROJECT:-gxpo-efficiency-final}"
export WANDB_GROUP="${WANDB_GROUP:-qwen2.5-math-7b-instruct-adamw}"
export GXPO_RUN_NAME="${GXPO_RUN_NAME:-qwen25_math_7b_instruct_gxpo_adamw_b256_mb64_k10_a0.3_kl01_noent_r2048_entropy_tau3}"

echo "[qwen 7b GXPO + AdamW launcher]"
echo "model=$MODEL_ID method=$METHOD optimizer=$OPTIMIZER_NAME"
echo "batch/minibatch=$TRAIN_BATCH_SIZE/$PPO_MINI_BATCH_SIZE GPUs=$GPU_IDS fsdp=$FSDP_SIZE"
echo "temperature=$ROLLOUT_TEMPERATURE top_p=$ROLLOUT_TOP_P max_response_length=$MAX_RESPONSE_LENGTH"
echo "attention=$ATTN_IMPL liger=$USE_LIGER vllm_backend=$VLLM_ATTENTION_BACKEND enforce_eager=$VLLM_ENFORCE_EAGER"
echo "vllm_gpu_util=$VLLM_GPU_MEMORY_UTILIZATION max_batched_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS max_seqs=$VLLM_MAX_NUM_SEQS"
echo "KL enabled=$ACTOR_USE_KL_LOSS coef=$ACTOR_KL_LOSS_COEF entropy_coeff=$ENTROPY_COEFF"
echo "gxpo_threshold=$GXPO_TAU signal=$GXPO_TRIGGER_SIGNAL window=$GXPO_ZSCORE_W patience=$GXPO_TRIGGER_PATIENCE"

exec env \
  MODEL_ALIAS="$MODEL_ALIAS" \
  MODEL_ID="$MODEL_ID" \
  METHOD="$METHOD" \
  TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
  PPO_MINI_BATCH_SIZE="$PPO_MINI_BATCH_SIZE" \
  OPTIMIZER_NAME="$OPTIMIZER_NAME" \
  GXPO_TAU="$GXPO_TAU" \
  GXPO_ZSCORE_W="$GXPO_ZSCORE_W" \
  GXPO_TRIGGER_PATIENCE="$GXPO_TRIGGER_PATIENCE" \
  GXPO_OPTIMIZER_STATE_MODE="$GXPO_OPTIMIZER_STATE_MODE" \
  ROLLOUT_TEMPERATURE="$ROLLOUT_TEMPERATURE" \
  ROLLOUT_TOP_P="$ROLLOUT_TOP_P" \
  MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
  MAX_STEPS="$MAX_STEPS" \
  SAVE_FREQ="$SAVE_FREQ" \
  TRAINER_TEST_FREQ="$TRAINER_TEST_FREQ" \
  GXPO_RUN_NAME="$GXPO_RUN_NAME" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  WANDB_GROUP="$WANDB_GROUP" \
  bash "$SCRIPT_DIR/common.sh"
