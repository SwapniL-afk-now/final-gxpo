#!/usr/bin/env bash
#
# llama32_3b_gxpo_muon_faststate_b128_mb32.sh
#
# Complete entrypoint: Llama-3.2-3B-Instruct | GXPO + Muon | batch 128 |
# minibatch 32 | K=10 | alpha=0.3 | 2 GPUs (Blackwell 6000 Pro class, FSDP 2).
#
# Gate configuration - CONSERVATIVE ENTROPY PROFILE (H200 calibration: tau=3.0 /
# patience=3 gives 0/60 false positives on a flat healthy series; slow drift is
# caught by the relative sustained-level criterion, which costs no false positives):
#   signal    : entropy (trainer-side gate)
#   trigger   : z-score >= 3.0, held for 3 consecutive scored batches
#   warmup    : 0, so the frozen level baseline is learned from the early regime
#   budget    : permanent fallback once tripped (no re-arming)
#
# Optimizer: Muon (gather-scatter backend under FSDP) instead of AdamW; every
# optimizer choice stays overridable via OPTIMIZER_NAME=adamw|muon.
#
# Batch is halved vs the 4-GPU H200 muon run (256 -> 128) to keep per-GPU load
# identical on 2 GPUs. Re-tune on the training host if headroom allows.
#
# Usage:
#   bash llama32_3b_gxpo_muon_faststate_b128_mb32.sh            # launch
#   bash llama32_3b_gxpo_muon_faststate_b128_mb32.sh --dry-run  # print resolved
#                                                      # config, no launch
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

# ------------------------------------------------------------ gate config ----
# Experiment settings owned by this entrypoint.  The downstream common.sh chain
# must preserve these inherited values instead of overriding them.
export K=10
export REPOSITION_ALPHA=0.3
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export GPU_IDS="${GPU_IDS:-0,1}"
export GPU_COUNT="${GPU_COUNT:-2}"
export FSDP_SIZE="${FSDP_SIZE:-2}"

# Optimizer selection: Muon with the FSDP gather-scatter backend.
export OPTIMIZER_NAME="${OPTIMIZER_NAME:-muon}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export MUON_NS_STEPS="${MUON_NS_STEPS:-5}"
export MUON_NESTEROV="${MUON_NESTEROV:-True}"
export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-1e-2}"
export MUON_DISTRIBUTED_BACKEND="${MUON_DISTRIBUTED_BACKEND:-gather_scatter}"

# Transactional GXPO: probe optimizer moments are snapshotted and restored, so
# the two probe steps never pollute the moments of the slow correction step.
export GXPO_OPTIMIZER_STATE_MODE="transactional_fast_state"

# Conservative entropy gate (see header for calibration rationale).
export GXPO_TRIGGER_SIGNAL="${GXPO_TRIGGER_SIGNAL:-entropy}"
export GXPO_SHUTOFF_MODE="${GXPO_SHUTOFF_MODE:-trajectory_aware}"
export GXPO_TAU="${GXPO_TAU:-3.0}"
export GXPO_ZSCORE_W="${GXPO_ZSCORE_W:-30}"
export GXPO_TRIGGER_PATIENCE="${GXPO_TRIGGER_PATIENCE:-3}"
export GXPO_FALLBACK_MODE="${GXPO_FALLBACK_MODE:-permanent}"
export GXPO_FALLBACK_WINDOW="${GXPO_FALLBACK_WINDOW:-10}"
export GXPO_WARMUP_STEPS="${GXPO_WARMUP_STEPS:-0}"
export GXPO_RESET_ENTROPY_AFTER_WARMUP="${GXPO_RESET_ENTROPY_AFTER_WARMUP:-True}"

# Conservative memory profile for the 3B model on 2 GPUs (smoke-tune on host).
export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-8}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-512}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"

# ------------------------------------------------------------- preflight -----
MISSING=0
# This checkout has neither models/ nor a prepared data/ of its own on this host;
# both live in the sibling gxpo-speed-audit checkout (same files, same layout).
GXPO_ASSET_ROOT="/office/dev_workspace/swapnil/gradient-extrapolation-based-policy-optimization-gxpo-speed-audit"
MODEL_DIR="${MODEL_LLAMA32_3B:-$GXPO_ASSET_ROOT/models/Llama-3.2-3B-Instruct}"
DATA_ROOT="${GXPO_DATA_ROOT:-$GXPO_ASSET_ROOT/Code/SFPO/data}"
export MODEL_LLAMA32_3B="$MODEL_DIR"
export GXPO_DATA_ROOT="$DATA_ROOT"

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
  method             : gxpo + ${OPTIMIZER_NAME:-muon} (K=${K:-10}, alpha=${REPOSITION_ALPHA:-0.3})
  batch / minibatch  : ${TRAIN_BATCH_SIZE:-128} / ${PPO_MINI_BATCH_SIZE:-32}
  gpus               : ${GPU_COUNT:-2}  (ids ${GPU_IDS:-0,1}, FSDP_SIZE=${FSDP_SIZE:-2})
  max_steps          : ${MAX_STEPS:-400}   save_freq ${SAVE_FREQ:-20}
  optimizer          : ${OPTIMIZER_NAME:-muon} (momentum ${MUON_MOMENTUM:-0.95}, NS ${MUON_NS_STEPS:-5}, backend ${MUON_DISTRIBUTED_BACKEND:-gather_scatter})
  optimizer_state    : $GXPO_OPTIMIZER_STATE_MODE
  attention          : train ${ATTN_IMPL:-flash_attention_2} | vllm ${VLLM_ATTENTION_BACKEND:-FLASHINFER}
  --- gate: conservative entropy (tau=3, patience=3) ---
  trigger_signal     : $GXPO_TRIGGER_SIGNAL
  shutoff_mode       : $GXPO_SHUTOFF_MODE
  tau / patience     : $GXPO_TAU / $GXPO_TRIGGER_PATIENCE
  zscore window      : $GXPO_ZSCORE_W
  max_active_steps   : ${GXPO_MAX_ACTIVE_STEPS:-0}    (0 = no hard budget cap)
  wandb project      : ${WANDB_PROJECT:-gxpo-efficiency-final}
[dry-run] preflight OK - would launch now.
EOT
  exit 0
fi

# ---------------------------------------------------------------- launch -----
MODEL_ALIAS="llama32-3b-muon"
MODEL_ID="$MODEL_LLAMA32_3B"
METHOD="gxpo"
export SAVE_FREQ="${SAVE_FREQ:-20}"
source "$SCRIPT_DIR/common.sh"
