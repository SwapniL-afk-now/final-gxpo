#!/usr/bin/env bash
#
# qwen25_math_1p5b_gxpo_b256_mb64_gate_v6.sh
#
# Complete entrypoint: Llama-3.2-3B-Instruct | GXPO | batch 256 |
# minibatch 64 | K=10 | alpha=0.3 | 4 GPUs (FSDP size 4)
# driven by the Gate-v2 prediction-quality trigger.
#
# Gate configuration: ordinary mean/std z-score of cosine disagreement.
#   signal      : grad (actor-side; disagreement = 1 - |cos(g0, g_slow)|)
#   shutoff     : cosine disagreement
#   trigger     : z-score >= 2.0 for 2 consecutive outer batches
#   window      : 30 observations
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
# Experiment settings owned by this entrypoint.  The downstream environment
# wrapper must preserve these inherited values instead of overriding them.
export K=10
export REPOSITION_ALPHA=0.3

# Actor-side prediction-quality gate. The shared trainer-side entropy gate is
# explicitly disabled for this grad-trigger run; cosine disagreement is the
# only statistical fallback trigger.
export GXPO_TRIGGER_SIGNAL="${GXPO_TRIGGER_SIGNAL:-grad}"
export GXPO_SHUTOFF_MODE="${GXPO_SHUTOFF_MODE:-cosine}"
export GXPO_TRIGGER_ABS_THRESHOLD="${GXPO_TRIGGER_ABS_THRESHOLD:-0}"
export GXPO_TRIGGER_ROBUST="${GXPO_TRIGGER_ROBUST:-0}"
export GXPO_TAU="${GXPO_TAU:-2.0}"
export GXPO_ZSCORE_W="${GXPO_ZSCORE_W:-30}"
export GXPO_TRIGGER_PATIENCE="${GXPO_TRIGGER_PATIENCE:-2}"
export GXPO_TRIGGER_MIN_OBS="${GXPO_TRIGGER_MIN_OBS:-0}"
export GXPO_MAX_ACTIVE_STEPS="${GXPO_MAX_ACTIVE_STEPS:-150}"
unset GXPO_TRIGGER_SUSTAIN_W
# Ordinary mean/std z-score over the 30-observation window.
# ------------------------------------------------------------- preflight -----
MISSING=0
MODEL_DIR="${MODEL_LLAMA32_3B:-/workspace/models/Llama-3.2-3B-Instruct}"
DATA_ROOT="${GXPO_DATA_ROOT:-/workspace/data}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "PREFLIGHT FAIL: model weights not found at $MODEL_DIR" >&2
  echo "  (download meta-llama/Llama-3.2-3B-Instruct there, or point" >&2
  echo "   MODEL_LLAMA32_3B at an existing local copy)" >&2
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
  method             : gxpo (K=${K:-10}, alpha=${REPOSITION_ALPHA:-0.3})
  batch / minibatch  : ${TRAIN_BATCH_SIZE:-256} / ${PPO_MINI_BATCH_SIZE:-64}
  gpus               : ${GPU_COUNT:-4}  (ids ${GPU_IDS:-0,1,2,3}, FSDP_SIZE=${FSDP_SIZE:-4})
  max_steps          : ${MAX_STEPS:-400}   save_freq ${SAVE_FREQ:-20}
  dtype / liger      : ${ACTOR_MODEL_DTYPE:-float32} / ${USE_LIGER:-True}
  attention          : train ${ATTN_IMPL:-flash_attention_2} | vllm ${VLLM_ATTENTION_BACKEND:-FLASHINFER}
  --- gate v2 ---
  trigger_signal     : $GXPO_TRIGGER_SIGNAL
  shutoff_mode       : $GXPO_SHUTOFF_MODE        (disagreement = 1 - |cos(g0,g_slow)|)
  abs threshold      : $GXPO_TRIGGER_ABS_THRESHOLD (disabled; z-score path)
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
# Hands off to the proven b256 wrapper (env setup, caches, common.sh chain).
exec bash "$SCRIPT_DIR/qwen25_math_1p5b_gxpo_b256_a05.sh"
