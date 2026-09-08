#!/usr/bin/env bash
#
# qwen25_math_1p5b_gxpo_muon_transactional_b64_mb16.sh
#
# Complete entrypoint: Qwen2.5-Math-1.5B-Instruct | GXPO + Muon | batch 64 |
# minibatch 16 | K=10 | alpha=0.3 | 2 GPUs (Blackwell 6000 Pro class, FSDP 2)
# driven by the z-score cosine-disagreement trigger.
#
# Optimizer-state mode: TRANSACTIONAL -- probe optimizer moments are
# snapshotted before the two probe steps and rolled back after repositioning,
# so the slow correction step is always taken from the moments the minibatch
# started with. This is the companion arm to
# qwen25_math_1p5b_gxpo_muon_faststate_b64_mb16.sh, which keeps the probe
# steps' moments instead (Adam's step counter then advances 3x per minibatch).
#
# Gate configuration - ORDINARY Z-SCORE PROFILE:
#   signal    : grad (actor-side; disagreement = 1 - |cos(g0, g_slow)| from pre-clip grads)
#   trigger   : ordinary mean/std z-score of disagreement >= 2.0,
#               held for 2 consecutive scored batches
#   history   : preceding 30 disagreement observations (ABS_THRESHOLD=0 selects z-path)
#   budget    : hard stop after 150 enabled steps regardless of gate (runtime cap)
#
# Optimizer: Muon (gather-scatter backend under FSDP) instead of AdamW; every
# optimizer choice stays overridable via OPTIMIZER_NAME=adamw|muon.
#
# Usage:
#   bash qwen25_math_1p5b_gxpo_muon_transactional_b64_mb16.sh            # launch
#   bash qwen25_math_1p5b_gxpo_muon_transactional_b64_mb16.sh --dry-run  # print
#                                                          # resolved config, no launch
#
# Every setting below can be overridden from the environment, e.g.:
#   GXPO_TAU=1.5 MAX_STEPS=200 bash qwen25_math_1p5b_gxpo_muon_transactional_b64_mb16.sh
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
export K="${K:-10}"
export REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.3}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
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
export GXPO_OPTIMIZER_STATE_MODE="transactional"

# Retention is read off the two real optimizer steps (per-matrix scalar) for
# Muon-owned matrices; gradient ratios cannot describe Muon's displacement,
# because its step size is independent of gradient magnitude. Set
# GXPO_RETENTION_SPACE=grad to reproduce the pre-fix arm.
export GXPO_RETENTION_SPACE="${GXPO_RETENTION_SPACE:-auto}"

# Actor-side prediction-quality gate (z-score path: ABS_THRESHOLD=0).
export GXPO_TRIGGER_SIGNAL="${GXPO_TRIGGER_SIGNAL:-grad}"
export GXPO_SHUTOFF_MODE="${GXPO_SHUTOFF_MODE:-cosine}"
export GXPO_TRIGGER_ABS_THRESHOLD="${GXPO_TRIGGER_ABS_THRESHOLD:-0}"
export GXPO_TRIGGER_SUSTAIN_W="${GXPO_TRIGGER_SUSTAIN_W:-10}"
# Ordinary mean/std z-score; robust median/MAD is intentionally disabled.
export GXPO_TRIGGER_ROBUST="${GXPO_TRIGGER_ROBUST:-0}"
export GXPO_TAU="${GXPO_TAU:-2.0}"
export GXPO_ZSCORE_W="${GXPO_ZSCORE_W:-30}"
export GXPO_TRIGGER_MIN_OBS="${GXPO_TRIGGER_MIN_OBS:-0}"
export GXPO_MAX_ACTIVE_STEPS="${GXPO_MAX_ACTIVE_STEPS:-150}"
export GXPO_TRIGGER_PATIENCE="${GXPO_TRIGGER_PATIENCE:-2}"
export GXPO_WARMUP_STEPS="${GXPO_WARMUP_STEPS:-0}"
export GXPO_RESET_ENTROPY_AFTER_WARMUP="${GXPO_RESET_ENTROPY_AFTER_WARMUP:-False}"

# Memory profile proven on this host for 1.5B GXPO (see qwen25_math_1p5b_gxpo_b256_a05.sh).
# Muon state (momentum only) is smaller than AdamW's, so this is conservative.
export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"
export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-8}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1024}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-98304}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.7}"

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
  method             : gxpo + ${OPTIMIZER_NAME:-muon} (K=${K:-10}, alpha=${REPOSITION_ALPHA:-0.3})
  batch / minibatch  : ${TRAIN_BATCH_SIZE:-64} / ${PPO_MINI_BATCH_SIZE:-16}
  gpus               : ${GPU_COUNT:-2}  (ids ${GPU_IDS:-0,1}, FSDP_SIZE=${FSDP_SIZE:-2})
  max_steps          : ${MAX_STEPS:-400}   save_freq ${SAVE_FREQ:-20}
  optimizer          : ${OPTIMIZER_NAME:-muon} (momentum ${MUON_MOMENTUM:-0.95}, NS ${MUON_NS_STEPS:-5}, backend ${MUON_DISTRIBUTED_BACKEND:-gather_scatter})
  optimizer_state    : $GXPO_OPTIMIZER_STATE_MODE
  retention_space    : $GXPO_RETENTION_SPACE      (auto = update-space for Muon matrices)
  attention          : train ${ATTN_IMPL:-flash_attention_2} | vllm ${VLLM_ATTENTION_BACKEND:-FLASHINFER}
  --- gate: ordinary z-score (abs=0) ---
  trigger_signal     : $GXPO_TRIGGER_SIGNAL      (must not be 'entropy')
  shutoff_mode       : $GXPO_SHUTOFF_MODE
  tau / patience     : $GXPO_TAU / $GXPO_TRIGGER_PATIENCE
  robust statistic   : $GXPO_TRIGGER_ROBUST      (0 = ordinary mean/std)
  zscore window      : $GXPO_ZSCORE_W
  min_obs age floor  : $GXPO_TRIGGER_MIN_OBS
  max_active_steps   : $GXPO_MAX_ACTIVE_STEPS    (hard runtime ceiling)
  wandb project      : ${WANDB_PROJECT:-gxpo-efficiency-final}
[dry-run] preflight OK - would launch now.
EOT
  exit 0
fi

# ---------------------------------------------------------------- launch -----
MODEL_ALIAS="qwen25-math-1p5b"
MODEL_ID="$MODEL_QWEN25_MATH_1P5B"
METHOD="gxpo"
export SAVE_FREQ="${SAVE_FREQ:-20}"
source "$SCRIPT_DIR/common.sh"
