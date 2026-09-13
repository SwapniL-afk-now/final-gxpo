#!/usr/bin/env bash
#
# qwen25_1p5b_grpo_sled_delta_signgate.sh
#
# Qwen2.5-Math-1.5B-Instruct | GRPO + SLED-Delta (OPD^2-style sign gate) | plain AdamW
#
# The new combined method:
#   L_total = grpo_coef * L_grpo + sled_loss_coef * L_sled,
# where L_grpo is the repository's unchanged GRPO loss (sequence-level
# group-relative advantage, PPO clip) and L_sled is the auxiliary
# token-level sign-gated self-distillation loss
# (verl/workers/actor/sled_delta.py). One rollout, one optimizer update.
#
# Configuration source of truth: Code/SFPO/experiments/gxpo_efficiency/common.sh
# (same file the plain-GRPO control qwen25_math_1p5b_grpo.sh sources). This
# launcher only selects the model, enables the default-off SLED block, and
# pins the run name. Every other training hyperparameter resolves to the
# common.sh default, i.e. to the plain-GRPO control value. See README.md for
# the parity table.
#
# Usage:
#   bash qwen25_1p5b_grpo_sled_delta_signgate.sh            # launch
#   bash qwen25_1p5b_grpo_sled_delta_signgate.sh --dry-run  # resolved config, no launch
#
# GPU selection is never hardcoded here: pass through the environment, e.g.
#   GPU_IDS=3 GPU_COUNT=1 bash qwen25_1p5b_grpo_sled_delta_signgate.sh
# (common.sh maps GPU_IDS onto CUDA_VISIBLE_DEVICES). Do not launch on GPUs
# holding other jobs.
#
# Everything is overridable from the environment, e.g.:
#   SLED_LOSS_COEF=0                         # regression: exact plain GRPO
#   SLED_LOSS_COEF=0.5 SLED_ALPHA=1.0         # ablations without code changes
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"   # Code/SFPO
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ---------------------------------------------------------------- secrets ----
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

# ------------------------------------------------------------- SLED config ---
export SLED_ENABLED=1
export SLED_LOSS_COEF="${SLED_LOSS_COEF:-1.0}"
export SLED_ALPHA="${SLED_ALPHA:-0.5}"
export SLED_EARLY_LAYER="${SLED_EARLY_LAYER:--1}"
export SLED_TOPK="${SLED_TOPK:-1024}"
export SLED_MICRO_BATCH_SIZE="${SLED_MICRO_BATCH_SIZE:-2}"
export SLED_CHUNK_TOKENS="${SLED_CHUNK_TOKENS:-512}"

# Keep validation periodic and greedy; skip the separate stochastic terminal eval.
export VAL_N=1
export VAL_DO_SAMPLE=False
export VAL_TEMPERATURE=0.0
export FINAL_EVAL_ENABLED=False

# The SLED frozen path can hold hidden states on CPU; Liger RMSNorm is GPU-only.
export USE_LIGER=False
# FlashAttention 3 is not installed on this host; use the installed backend.
export ATTN_IMPL=flash_attention_2

# ------------------------------------------------------------ dry run --------
if [[ "$DRY_RUN" -eq 1 ]]; then
  cat <<EOT
[dry-run] resolved launch configuration
  repo_root          : $REPO_ROOT
  student            : ${MODEL_QWEN25_MATH_1P5B:-Qwen/Qwen2.5-Math-1.5B-Instruct}
  method             : grpo + SLED-Delta auxiliary loss (OPD^2-style sign gate)
  advantage          : unchanged GRPO group-relative sequence advantage (broadcast)
  sled teacher       : early-exit self-contrast, alpha $SLED_ALPHA, early_layer $SLED_EARLY_LAYER
  sled loss          : L = grpo_coef * L_grpo + $SLED_LOSS_COEF * L_sled (gate acts on SLED only)
  sled scoring       : topk $SLED_TOPK, micro_bsz $SLED_MICRO_BATCH_SIZE, chunk $SLED_CHUNK_TOKENS
  regression         : SLED_LOSS_COEF=0 skips SLED entirely (exact plain GRPO)
  config source      : experiments/gxpo_efficiency/common.sh defaults (GRPO control parity)
  gpus               : ${GPU_COUNT:-1} (ids ${GPU_IDS:-<inherited>}, FSDP_SIZE=${FSDP_SIZE:-1})
  wandb project      : ${WANDB_PROJECT:-gxpo-efficiency-final}
[dry-run] preflight OK - would launch now (common.sh runs its own preflight).
EOT
  exit 0
fi

# ---------------------------------------------------------------- launch -----
# Same common.sh lineage as the plain-GRPO control; only the model (per the
# experiment brief: Qwen2.5-1.5B-Instruct) and the SLED switch differ.
export TRAIN_SEED="${TRAIN_SEED:-3407}"
export GXPO_RUN_NAME="${GXPO_RUN_NAME:-qwen25_math_1p5b_grpo_sled_delta_signgate_seed${TRAIN_SEED}}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen25-math-1p5b}"
MODEL_ID="${MODEL_QWEN25_MATH_1P5B:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
METHOD="grpo"
source "$REPO_ROOT/experiments/gxpo_efficiency/common.sh"
