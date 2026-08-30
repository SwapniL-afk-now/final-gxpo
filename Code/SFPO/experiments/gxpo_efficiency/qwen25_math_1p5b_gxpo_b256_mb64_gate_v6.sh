#!/usr/bin/env bash
#
# qwen25_math_1p5b_gxpo_b256_mb64_gate_v6.sh
#
# Complete entrypoint: Qwen2.5-Math-1.5B-Instruct | GXPO | batch 256 |
# minibatch 64 | K=10 | alpha=0.8 | 4 GPUs (FSDP size 4)
# driven by the Gate-v2 prediction-quality trigger.
#
# Gate configuration: gradient-direction disagreement with cosine shutoff.
#   signal      : disagreement (actor-side; 1 - |cos(g0, g_slow)|)
#   shutoff     : cosine disagreement
#   trigger     : absolute disagreement >= 0.15, with a 10-step z-score window
#   window      : 10 observations
#   hard budget : stop extrapolation after 150 enabled steps
#
# Usage:
#   bash qwen25_math_1p5b_gxpo_b256_mb64_gate_v6.sh            # launch
#   bash qwen25_math_1p5b_gxpo_b256_mb64_gate_v6.sh --dry-run  # print resolved
#                                                              # config, no launch
#
# Every setting below can be overridden from the environment, e.g.:
#   GXPO_TAU=1.5 MAX_STEPS=200 bash qwen25_math_1p5b_gxpo_b256_mb64_gate_v6.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"       # checkout root (holds .env, models/, Code/)
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ---------------------------------------------------------------- secrets ----
# Checkout-local secrets (WANDB_API_KEY); never printed. When this is a
# separate clone without its own .env, use the workspace-level .env.
ENV_FILE="${GXPO_ENV_FILE:-$REPO_ROOT/.env}"
if [[ ! -f "$ENV_FILE" && -f "/workspace/.env" ]]; then
  ENV_FILE="/workspace/.env"
fi
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ENV_FILE"
  set +a
fi

# ------------------------------------------------------------ gate config ----
# Experiment settings owned by this entrypoint.  Defaults are kept for the
# v6 experiment, while explicit environment overrides are honored so controlled
# K/alpha sweeps are actually reflected in the downstream trainer config.
export K="${K:-10}"
export REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.8}"
export MODEL_QWEN25_MATH_1P5B="${MODEL_QWEN25_MATH_1P5B:-/workspace/models/Qwen2.5-Math-1.5B-Instruct}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
export GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"
export GXPO_DATA_ROOT="${GXPO_DATA_ROOT:-/workspace/data}"
export GXPO_WARMUP_STEPS="${GXPO_WARMUP_STEPS:-0}"
export GXPO_RESET_ENTROPY_AFTER_WARMUP="${GXPO_RESET_ENTROPY_AFTER_WARMUP:-False}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export GPU_COUNT="${GPU_COUNT:-4}"
export FSDP_SIZE="${FSDP_SIZE:-4}"
export ATTN_IMPL="${ATTN_IMPL:-flash_attention_3}"
# Keep vLLM larger than the shared defaults, but leave headroom for the
# transactional GXPO actor. The previous 0.75/49k combination OOMed during
# the first backward pass on H200 despite the 1.5B model size.
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.65}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-98304}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1024}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-8}"

# Use the shared trainer-side entropy gate, matching SFPO. The actor remains
# on GXPO while entropy controls when extrapolation is shut off.
export GXPO_TRIGGER_SIGNAL="${GXPO_TRIGGER_SIGNAL:-disagreement}"
export GXPO_SHUTOFF_MODE="${GXPO_SHUTOFF_MODE:-cosine}"
export GXPO_TRIGGER_ABS_THRESHOLD="${GXPO_TRIGGER_ABS_THRESHOLD:-0.15}"
export GXPO_TRIGGER_ROBUST="${GXPO_TRIGGER_ROBUST:-0}"
export GXPO_TAU="${GXPO_TAU:-2.0}"
export GXPO_ZSCORE_W="${GXPO_ZSCORE_W:-10}"
export GXPO_TRIGGER_PATIENCE="${GXPO_TRIGGER_PATIENCE:-2}"
export GXPO_TRIGGER_MIN_OBS="${GXPO_TRIGGER_MIN_OBS:-0}"
export GXPO_MAX_ACTIVE_STEPS="${GXPO_MAX_ACTIVE_STEPS:-150}"
unset GXPO_TRIGGER_SUSTAIN_W
# Ordinary mean/std z-score over the 10-observation window.
# ------------------------------------------------------------- preflight -----
MISSING=0
MODEL_DIR="$MODEL_QWEN25_MATH_1P5B"
DATA_ROOT="${GXPO_DATA_ROOT:-/workspace/data}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "PREFLIGHT FAIL: model weights not found at $MODEL_DIR" >&2
  echo "  (download Qwen/Qwen2.5-Math-1.5B-Instruct there, or point" >&2
  echo "   MODEL_QWEN25_MATH_1P5B at an existing local copy)" >&2
  MISSING=1
fi

for rel in dapo_math/train.parquet lighteval-math/train.parquet \
           math500/test.parquet aime2024/test.parquet aime2025/test.parquet \
           amc/test.parquet minervamath/test.parquet olympiadbench/test.parquet; do
  if [[ ! -f "$DATA_ROOT/$rel" ]]; then
    echo "PREFLIGHT FAIL: missing prepared dataset: $DATA_ROOT/$rel" >&2
    MISSING=1
  fi
done

if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE:-online}" != "offline" ]]; then
  echo "PREFLIGHT WARN: WANDB_API_KEY not set and WANDB_MODE!=offline;" >&2
  echo "                 metrics will fail to upload (training continues)." >&2
fi

if [[ "$MISSING" -ne 0 ]]; then
  echo "Preflight failed - fix the items above and re-run." >&2
  exit 2
fi

# ------------------------------------------------------------ dry run --------
if [[ "$DRY_RUN" -eq 1 ]]; then
  cat <<EOT
[dry-run] resolved launch configuration
  repo_root          : $REPO_ROOT
  model              : $MODEL_DIR
  data_root          : $DATA_ROOT
  method             : gxpo (K=${K:-10}, alpha=${REPOSITION_ALPHA:-0.8})
  batch / minibatch  : ${TRAIN_BATCH_SIZE:-256} / ${PPO_MINI_BATCH_SIZE:-64}
  gpus               : ${GPU_COUNT:-4}  (ids ${GPU_IDS:-0,1,2,3}, FSDP_SIZE=${FSDP_SIZE:-4})
  max_steps          : ${MAX_STEPS:-400}   save_freq ${SAVE_FREQ:-20}
  dtype / liger      : ${ACTOR_MODEL_DTYPE:-float32} / ${USE_LIGER:-True}
  attention          : train ${ATTN_IMPL:-flash_attention_2} | vllm ${VLLM_ATTENTION_BACKEND:-FLASHINFER}
  --- disagreement trigger ---
  trigger_signal     : $GXPO_TRIGGER_SIGNAL (SFPO-style trainer gate)
  shutoff_mode       : $GXPO_SHUTOFF_MODE        (disagreement = 1 - |cos(g0,g_slow)|)
  abs threshold      : $GXPO_TRIGGER_ABS_THRESHOLD
  tau / patience     : $GXPO_TAU / $GXPO_TRIGGER_PATIENCE
  robust statistic   : $GXPO_TRIGGER_ROBUST (ordinary mean/std)
  min_obs            : $GXPO_TRIGGER_MIN_OBS
  max_active_steps   : $GXPO_MAX_ACTIVE_STEPS    (hard runtime ceiling)
  zscore window      : $GXPO_ZSCORE_W
  wandb project      : ${WANDB_PROJECT:-gxpo-efficiency-final}
[dry-run] preflight OK - would launch now.
EOT
  exit 0
fi

# ---------------------------------------------------------------- launch -----
# Run the shared trainer directly; this launcher is self-contained for the 1.5B model.
MODEL_ALIAS="qwen25-math-1p5b"
MODEL_ID="$MODEL_QWEN25_MATH_1P5B"
METHOD="gxpo"
export SAVE_FREQ="${SAVE_FREQ:-20}"
source "$SCRIPT_DIR/common.sh"
