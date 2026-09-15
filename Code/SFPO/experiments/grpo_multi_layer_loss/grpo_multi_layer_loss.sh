#!/usr/bin/env bash
#
# grpo_multi_layer_loss.sh
#
# Qwen2.5-Math-1.5B-Instruct | GRPO + AdamW | multi-layer policy loss |
# batch 64 | minibatch 16.
#
# The ONLY change vs plain GRPO is the policy loss. For the selected layers S
# (1-based decoder blocks; l = output of block l, l = 28 is the final layer):
#
#     log pi^(l)(y_t) = log_softmax(lm_head(norm(h_t^(l))) / T)[y_t]
#     r_t^(l)         = exp(log pi_theta^(l) - log pi_old^(l))
#     L_policy        = (1/|S|) * sum_{l in S} L_GRPO(r^(l), A)
#
# Shared lm_head + final norm (no new parameters), same advantages, mask, clip
# and loss_agg_mode, one backward and one optimizer step per mini-batch.
# pi_old^(l) comes from the actor's usual no-grad old-log-prob pass, which is
# why SLED vLLM rollouts are OFF here (they would bypass that pass).
#
# Usage:
#   bash grpo_multi_layer_loss.sh                                  # final (28) + 12,14,16
#   SELECTED_POLICY_LAYERS=12,14,16 bash grpo_multi_layer_loss.sh  # intermediate layers only
#   SELECTED_POLICY_LAYERS=14 bash grpo_multi_layer_loss.sh        # single layer
#   SELECTED_POLICY_LAYERS=null bash grpo_multi_layer_loss.sh      # plain-GRPO control
#   bash grpo_multi_layer_loss.sh --dry-run                        # print config, no launch
#
# Run name gets _ml<layers> (e.g. _ml12-14-16) so it never resumes a GRPO run.
# Every setting below is overridable from the environment, e.g.:
#   MAX_STEPS=200 GPU_IDS=0 bash grpo_multi_layer_loss.sh
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
# 1-based decoder layers; empty or null = original GRPO (control arm).
# 28 = num_hidden_layers of Qwen2.5-Math-1.5B = the legacy final-layer GRPO loss
# (it reuses the ordinary final log-probs, no extra projection). The update is
#     L = (L_12 + L_14 + L_16 + L_28) / 4
# Change 28 if MODEL_QWEN25_MATH_1P5B points at a model with a different depth.
export SELECTED_POLICY_LAYERS="${SELECTED_POLICY_LAYERS-12,14,16,28}"

export K="${K:-10}"
export REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.3}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export MAX_STEPS="${MAX_STEPS:-300}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-5}"
export TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-disable}"
export FINAL_EVAL_ENABLED="${FINAL_EVAL_ENABLED:-False}"

# Optimizer: plain fp32 AdamW (no Muon parameters at all in this run).
export OPTIMIZER_NAME="adamw"
# Plain GRPO control: no entropy bonus and no GXPO repositioning.
export ENTROPY_COEFF="0"

# Transactional GXPO: the two probe steps' moments and step counter are
# snapshotted before probe 1 and rolled back after repositioning, so the
# corrective step is taken from s0 and Adam's counter advances once per minibatch.
export GXPO_OPTIMIZER_STATE_MODE="transactional"

# Optimizer-aware retention. Under a pure AdamW optimizer `auto` classifies every
# trainable parameter as ADAMW_DIRECTION. Set GXPO_RETENTION_SPACE=grad to
# reproduce the legacy raw-gradient arm from this same entrypoint.
# Keep the GRPO run name free of the optimizer-aware GXPO suffix.
export GXPO_RETENTION_SPACE="grad"
# Keep mixed-response groups by enabling the pre-generation difficulty filter;
# the sampler skips easy/all-correct and hard/all-zero candidates while filling
# each training batch back to TRAIN_BATCH_SIZE.
export GXPO_DYNAMIC_FILTERING="True"
export GXPO_DYNAMIC_FILTERING_STRATEGY="all_probabilistic"
export GXPO_SAMPLING_BATCH_SIZE="${GXPO_SAMPLING_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"

# Plain vLLM sampling. SLED rollouts attach their own old_log_probs and skip
# the actor pass that computes pi_old^(l); forced off so an inherited env
# cannot turn it back on.
export SLED_VLLM_ENABLED="0"
export ACTOR_PARAM_OFFLOAD="True"
export ACTOR_OPTIMIZER_OFFLOAD="False"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-49152}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"

# Greedy validation, one response per prompt; training keeps eight rollouts.
export ROLLOUT_N="${ROLLOUT_N:-8}"
export VAL_N="${VAL_N:-1}"
export VAL_DO_SAMPLE="False"
export VAL_TEMPERATURE="0"

export SLED_ENABLED="0"
export OPD2_ENABLED="0"

export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
export SAVE_FREQ="${SAVE_FREQ:-5}"

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
  method             : grpo + adamw (GXPO disabled)
  policy layers      : ${SELECTED_POLICY_LAYERS:-null}  (1-based; null = plain GRPO)
  batch / minibatch  : $TRAIN_BATCH_SIZE / $PPO_MINI_BATCH_SIZE
  gpus               : ${GPU_COUNT:-1}  (ids ${GPU_IDS:-<inherited>}, FSDP_SIZE=${FSDP_SIZE:-1})
  max_steps          : ${MAX_STEPS:-300}   save_freq $SAVE_FREQ
  optimizer          : $OPTIMIZER_NAME
  dynamic filtering  : $GXPO_DYNAMIC_FILTERING (mixed-response groups retained)
  attention          : train $ATTN_IMPL | vllm ${VLLM_ATTENTION_BACKEND:-FLASHINFER}
  wandb project      : ${WANDB_PROJECT:-gxpo-efficiency-final}
  A/B baseline       : SELECTED_POLICY_LAYERS=null bash grpo_multi_layer_loss.sh
[dry-run] preflight OK - would launch now.
EOT
  exit 0
fi

# ---------------------------------------------------------------- launch -----
MODEL_ALIAS="qwen25-math-1p5b"
MODEL_ID="$MODEL_QWEN25_MATH_1P5B"
METHOD="grpo"
source "$SCRIPT_DIR/../gxpo_efficiency/common.sh"
