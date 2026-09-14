#!/usr/bin/env bash
#
# qwen25_math_1p5b_gxpo_adamw_transactional_dir_k10.sh
#
# Qwen2.5-Math-1.5B-Instruct | GXPO + AdamW |
# batch 64 | minibatch 16.
#
#   baseline (that script, GXPO_RETENTION_SPACE=grad)
#       r_i = (c1 * g1_i) / (c0 * g0_i)        raw-gradient retention
#
#   this script (GXPO_RETENTION_SPACE=auto)
#       d_t = ((1 - lr*wd) * theta_t - theta_{t+1}) / lr
#       r_i = d1_i / d0_i                      AdamW optimizer-direction retention
#
# AdamW does not move the parameters along the gradient; it moves them along
# m_hat / (sqrt(v_hat) + eps). Reconstructing d_t from the real probe
# displacement makes the retention signal carry AdamW's moments, its bias
# correction, its epsilon convention and the clipped gradient it actually
# consumed. Everything else -- GRPO loss, PPO clipping, rollout, K, alpha, LR,
# batch sizes, the trigger/gate, the corrective pass -- is identical to the
# baseline, so the two are a controlled A/B on the retention signal alone.
#
# Optimizer-state mode: TRANSACTIONAL. The probe trajectory s0 -> s1 -> s2 is
# rolled back to s0 before the corrective pass, so the retained AdamW step
# counter advances exactly ONCE per PPO minibatch:
#     (theta_next, s_next) = AdamW(theta_tilde, s0, g_corrective)
#
# Run name: common.sh appends "_adamwdir" for adamw+auto, so this can never
# resume the baseline's wandb id, checkpoints or result directory.
#     qwen25-math-1p5b_gxpo_k10_seed3407_adamwdir
#
# Usage:
#   bash qwen25_math_1p5b_gxpo_adamw_transactional_dir_k10.sh            # launch
#   bash qwen25_math_1p5b_gxpo_adamw_transactional_dir_k10.sh --dry-run  # print
#                                                       # resolved config, no launch
#
# Every setting below is overridable from the environment, e.g.:
#   MAX_STEPS=200 GPU_COUNT=2 bash qwen25_math_1p5b_gxpo_adamw_transactional_dir_k10.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"       # checkout root (holds .env, models/, Code/)
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ---------------------------------------------------------------- secrets ----
# Checkout-local secrets (WANDB_API_KEY); never printed.
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

# --------------------------------------------------------- experiment cfg ----
# Deliberately identical to qwen25_math_1p5b_gxpo_k10.sh so the only substantive
# difference between the two runs is the retention estimator.
export K="${K:-3}"
export REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.8}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"

# Optimizer: plain fp32 AdamW (no Muon parameters at all in this run).
export OPTIMIZER_NAME="adamw"

# Transactional GXPO: the two probe steps' moments and step counter are
# snapshotted before probe 1 and rolled back after repositioning, so the
# corrective step is taken from s0 and Adam's counter advances once per minibatch.
export GXPO_OPTIMIZER_STATE_MODE="transactional"

# Optimizer-aware retention. Under a pure AdamW optimizer `auto` classifies every
# trainable parameter as ADAMW_DIRECTION. Set GXPO_RETENTION_SPACE=grad to
# reproduce the legacy raw-gradient arm from this same entrypoint.
# Keep the GRPO run name free of the optimizer-aware GXPO suffix.
export GXPO_RETENTION_SPACE="${GXPO_RETENTION_SPACE:-auto}"
# Keep mixed-response groups by enabling the pre-generation difficulty filter;
# the sampler skips easy/all-correct and hard/all-zero candidates while filling
# each training batch back to TRAIN_BATCH_SIZE.
export GXPO_DYNAMIC_FILTERING="True"
export GXPO_DYNAMIC_FILTERING_STRATEGY="all_probabilistic"
export GXPO_SAMPLING_BATCH_SIZE="${GXPO_SAMPLING_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"

# SLED is disabled for this launcher. Keep rollout and loss paths off so future
# runs cannot re-enable SLED implicitly through this entrypoint.
export SLED_VLLM_ENABLED="0"
export SLED_ENABLED="0"
export SLED_VLLM_EARLY_LAYERS="${SLED_VLLM_EARLY_LAYERS:-14,18,22,26}"
export SLED_VLLM_ALPHA="${SLED_VLLM_ALPHA:-2.0}"
export SLED_VLLM_SCALE="${SLED_VLLM_SCALE:-10}"
export SLED_VLLM_LOWER_BOUND="${SLED_VLLM_LOWER_BOUND:--1000}"

export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
export SAVE_FREQ="${SAVE_FREQ:-20}"

# ------------------------------------------------------------- preflight -----
MISSING=0
# This checkout has neither models/ nor a prepared data/ of its own on this host;
# both live in the sibling gxpo-speed-audit checkout (same files, same layout).
GXPO_ASSET_ROOT="/office/dev_workspace/swapnil/gradient-extrapolation-based-policy-optimization-gxpo-speed-audit"
MODEL_DIR="${MODEL_QWEN25_MATH_1P5B:-$GXPO_ASSET_ROOT/models/Qwen2.5-Math-1.5B-Instruct}"
DATA_ROOT="${GXPO_DATA_ROOT:-$GXPO_ASSET_ROOT/Code/SFPO/data}"
export MODEL_QWEN25_MATH_1P5B="$MODEL_DIR"
export GXPO_DATA_ROOT="$DATA_ROOT"

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
  method             : ${METHOD:-gxpo} + adamw (GXPO enabled)
  K / alpha          : $K / $REPOSITION_ALPHA
  batch / minibatch  : $TRAIN_BATCH_SIZE / $PPO_MINI_BATCH_SIZE
  gpus               : ${GPU_COUNT:-1}  (ids ${GPU_IDS:-<inherited>}, FSDP_SIZE=${FSDP_SIZE:-1})
  max_steps          : ${MAX_STEPS:-400}   save_freq $SAVE_FREQ
  optimizer          : $OPTIMIZER_NAME
  optimizer_state    : $GXPO_OPTIMIZER_STATE_MODE
  retention_space    : $GXPO_RETENTION_SPACE
  dynamic filtering  : $GXPO_DYNAMIC_FILTERING (mixed-response groups retained)
  validation         : every ${TRAINER_TEST_FREQ:-5} steps, greedy n=${VAL_N:-1}
  attention          : train $ATTN_IMPL | vllm ${VLLM_ATTENTION_BACKEND:-FLASHINFER}
  wandb project      : ${WANDB_PROJECT:-gxpo-efficiency-final}
  A/B baseline       : qwen25_math_1p5b_gxpo_k10.sh (pinned GXPO_RETENTION_SPACE=grad)
[dry-run] preflight OK - would launch now.
EOT
  exit 0
fi

# ---------------------------------------------------------------- launch -----
MODEL_ALIAS="qwen25-math-1p5b"
MODEL_ID="$MODEL_QWEN25_MATH_1P5B"
METHOD="${METHOD:-gxpo}"
source "$SCRIPT_DIR/common.sh"
