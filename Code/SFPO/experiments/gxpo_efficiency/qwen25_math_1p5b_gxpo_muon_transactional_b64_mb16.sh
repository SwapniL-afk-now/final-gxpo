#!/usr/bin/env bash
#
# qwen25_math_1p5b_gxpo_muon_transactional_b64_mb16.sh
#
# Complete entrypoint: Qwen2.5-Math-1.5B-Instruct | GXPO + Muon | batch 256 |
# minibatch 64 | K=5 | alpha=0.5 | GPUs 1,2 (FSDP 2, TP 1), entropy trigger.
#
# Optimizer-state mode: TRANSACTIONAL -- probe optimizer moments are
# snapshotted before the two probe steps and rolled back after repositioning,
# so the slow correction step is always taken from the moments the minibatch
# started with. This is the companion arm to
# qwen25_math_1p5b_gxpo_muon_faststate_b64_mb16.sh, which keeps the probe
# steps' moments instead (Adam's step counter then advances 3x per minibatch).
#
# Gate configuration - ENTROPY PROFILE:
#   signal    : entropy
#   trigger   : ordinary mean/std z-score of the entropy signal >= 3.0,
#               held for 3 consecutive scored batches
#   history   : preceding 50 observations
#   budget    : hard stop after 150 enabled steps regardless of gate (runtime cap)
#
# Optimizer: Muon with the FSDP gather-scatter backend.
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
export K="5"
export REPOSITION_ALPHA="0.5"
export TRAIN_BATCH_SIZE="256"
export PPO_MINI_BATCH_SIZE="64"
export GPU_IDS="1,2"
export GPU_COUNT="2"
export FSDP_SIZE="2"

# Optimizer selection: Muon with the FSDP gather-scatter backend.
export OPTIMIZER_NAME="muon"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export MUON_NS_STEPS="${MUON_NS_STEPS:-5}"
export MUON_NESTEROV="${MUON_NESTEROV:-True}"
export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-1e-2}"
export MUON_DISTRIBUTED_BACKEND="${MUON_DISTRIBUTED_BACKEND:-gather_scatter}"

# KL is fully disabled: no actor KL loss, no reward KL penalty, and no
# reference-policy worker/log-prob computation (common.sh/main_ppo enforce this
# when these switches and the KL coefficient are all zero).
export USE_KL_LOSS="False"
export KL_LOSS_COEF="0.0"
export FINAL_EVAL_ENABLED="False"

# Transactional GXPO: probe optimizer moments are snapshotted and restored, so
# the two probe steps never pollute the moments of the slow correction step.
export GXPO_OPTIMIZER_STATE_MODE="transactional"

# Retention is read off the two real optimizer steps (per-matrix scalar) for
# Muon-owned matrices; gradient ratios cannot describe Muon's displacement,
# because its step size is independent of gradient magnitude. Set
# The optimizer-aware Muon update-space estimator is pinned for this run.
export GXPO_RETENTION_SPACE="auto"

# Actor-side entropy gate (z-score path: ABS_THRESHOLD=0).
export GXPO_TRIGGER_SIGNAL="entropy"
export GXPO_SHUTOFF_MODE="trajectory_aware"
export GXPO_TRIGGER_ABS_THRESHOLD="0"
export GXPO_TRIGGER_SUSTAIN_W="10"
# Ordinary mean/std z-score; robust median/MAD is intentionally disabled.
export GXPO_TRIGGER_ROBUST="0"
export GXPO_TAU="3.0"
export GXPO_ZSCORE_W="50"
export GXPO_TRIGGER_MIN_OBS="0"
export GXPO_MAX_ACTIVE_STEPS="150"
export GXPO_TRIGGER_PATIENCE="3"
export GXPO_WARMUP_STEPS="0"
export GXPO_RESET_ENTROPY_AFTER_WARMUP="False"

# Memory profile proven on this host for 1.5B GXPO (see qwen25_math_1p5b_gxpo_b256_a05.sh).
# Muon state (momentum only) is smaller than AdamW's, so this is conservative.
export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"
export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-8}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1024}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-98304}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
# 0.7 left this run at ~97.1 of 97.9 GB per GPU during generation and it OOM'd
# in vLLM's MLP activation asking for 1.64GB. GXPO is not an ordinary RL actor:
# on top of params/grads/optimizer it keeps three model-sized buffers
# (theta0/g0/g1) resident for the whole step, ~9.3GB per rank at 1.5B and
# ~19.3GB at 3B. 0.6 matches what the 3B launchers already use and leaves real
# headroom instead of relying on the run staying under the line.
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"

# System-RAM profile for the rule-based verifier. Each scorer is a spawned
# interpreter that re-imports the trainer's __main__, so it carries torch:
# measured 698MB RSS / 367MB private each. reward_fn and val_reward_fn used to
# hold separate 64-wide pools -> 128 workers, ~49GB of host RAM resident for the
# whole run. naive.py now shares one pool; this pins its width so the box's core
# count can never silently set it again.
export REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-16}"

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
  method             : gxpo + ${OPTIMIZER_NAME:-muon} (K=${K:-5}, alpha=${REPOSITION_ALPHA:-0.5})
  batch / minibatch  : ${TRAIN_BATCH_SIZE:-256} / ${PPO_MINI_BATCH_SIZE:-64}
  gpus               : ${GPU_COUNT:-2}  (ids ${GPU_IDS:-1,2}, FSDP_SIZE=${FSDP_SIZE:-2})
  max_steps          : ${MAX_STEPS:-400}   save_freq ${SAVE_FREQ:-20}
  optimizer          : ${OPTIMIZER_NAME:-muon} (momentum ${MUON_MOMENTUM:-0.95}, NS ${MUON_NS_STEPS:-5}, backend ${MUON_DISTRIBUTED_BACKEND:-gather_scatter})
  optimizer_state    : $GXPO_OPTIMIZER_STATE_MODE
  retention_space    : $GXPO_RETENTION_SPACE      (auto = update-space for Muon matrices)
  kl loss            : $USE_KL_LOSS (coefficient $KL_LOSS_COEF)
  attention          : train ${ATTN_IMPL:-flash_attention_2} | vllm ${VLLM_ATTENTION_BACKEND:-FLASHINFER}
  --- gate: entropy z-score (abs=0) ---
  trigger_signal     : $GXPO_TRIGGER_SIGNAL
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
