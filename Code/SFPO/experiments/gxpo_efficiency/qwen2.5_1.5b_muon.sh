#!/usr/bin/env bash
#
# qwen2.5_1.5b_muon.sh
#
# Complete entrypoint: Qwen2.5-Math-1.5B-Instruct | GXPO + Muon | batch 64 |
# minibatch 16 | K=10 | alpha=0.3 | 4 GPUs (FSDP size 4)
# driven by the simple cosine-disagreement z-score trigger.
#
# Gate configuration - ORDINARY Z-SCORE PROFILE:
#   signal    : grad (actor-side; disagreement = 1 - |cos(g0, g_slow)| from pre-clip grads)
#   trigger   : ordinary mean/std z-score of disagreement >= 2.0,
#               held for 2 consecutive scored batches
#   history   : preceding 30 disagreement observations
#   entropy   : trainer-side entropy trigger disabled; cosine z-score is the sole gate
#   budget    : hard stop after 150 enabled steps regardless of gate (runtime cap)
#
# Usage:
#   bash qwen2.5_1.5b_muon.sh            # launch
#   bash qwen2.5_1.5b_muon.sh --dry-run  # print resolved config, no launch
#
# Every setting below can be overridden from the environment, e.g.:
#   GXPO_TAU=1.5 MAX_STEPS=200 bash qwen25_math_1p5b_gxpo_b256_mb64_gate_v6.sh
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
# Experiment settings owned by this entrypoint.  The downstream environment
# wrapper must preserve these inherited values instead of overriding them.
export K="${K:-10}"
export REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.3}"

# This entrypoint owns the single-GPU launch. The downstream wrapper preserves
# these inherited values instead of reverting to its historical 0,1/FSDP=2 setup.
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_IDS}"
export GPU_COUNT="${GPU_COUNT:-4}"
export FSDP_SIZE="${FSDP_SIZE:-4}"
export TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-disable}"
export TRAINER_RESUME_FROM_PATH="${TRAINER_RESUME_FROM_PATH:-False}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export METHOD="${METHOD:-gxpo}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export OPTIMIZER_NAME="${OPTIMIZER_NAME:-muon}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export MUON_NS_STEPS="${MUON_NS_STEPS:-5}"
export MUON_NESTEROV="${MUON_NESTEROV:-True}"
export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-1e-2}"
export MUON_DISTRIBUTED_BACKEND="${MUON_DISTRIBUTED_BACKEND:-gather_scatter}"
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
export GXPO_RUN_NAME="${GXPO_RUN_NAME:-qwen25_math_1p5b_gxpo_muon_k10_a0.3_b64_mb16_gpu4_fsdp4_fp32_liger_zscore_v6_memsafe}"

# Actor-side prediction-quality gate. With signal=grad, ray_trainer uses only
# the actor-side cosine-disagreement gate; the entropy gate is not evaluated.
export GXPO_TRIGGER_SIGNAL="${GXPO_TRIGGER_SIGNAL:-grad}"
export GXPO_SHUTOFF_MODE="${GXPO_SHUTOFF_MODE:-cosine}"
# Set ABS_THRESHOLD=0 to select the z-score path instead of the direct level path.
export GXPO_TRIGGER_ABS_THRESHOLD="${GXPO_TRIGGER_ABS_THRESHOLD:-0}"
export GXPO_TRIGGER_SUSTAIN_W="${GXPO_TRIGGER_SUSTAIN_W:-10}"
# Ordinary mean/std z-score; robust median/MAD is intentionally disabled.
export GXPO_TRIGGER_ROBUST="${GXPO_TRIGGER_ROBUST:-0}"
export GXPO_TAU="${GXPO_TAU:-2.0}"
export GXPO_TRIGGER_MIN_OBS="${GXPO_TRIGGER_MIN_OBS:-0}"
export GXPO_MAX_ACTIVE_STEPS="${GXPO_MAX_ACTIVE_STEPS:-150}"
export GXPO_TRIGGER_PATIENCE="${GXPO_TRIGGER_PATIENCE:-2}"
export GXPO_ZSCORE_W="${GXPO_ZSCORE_W:-30}"

# Single-GPU memory-safe profile.  The actor, optimizer, GXPO slow-gradient
# state, and vLLM all share physical GPU 2; the previous two-GPU-sized
# 98k-token/1024-sequence vLLM profile exhausted the card during step 1.
# Keep the requested global batch/minibatch unchanged while bounding peak
# rollout and actor-update allocations.
export PPO_MAX_TOKEN_LEN_PER_GPU=16384
export VLLM_MAX_NUM_SEQS=512
export VLLM_MAX_NUM_BATCHED_TOKENS=32768
# Leave enough headroom for the full-precision actor, Muon state, and GXPO
# buffers when vLLM wakes up between rollout and training phases.
export VLLM_GPU_MEMORY_UTILIZATION=0.4

# ------------------------------------------------------------- preflight -----
MISSING=0
MODEL_DIR="${MODEL_QWEN25_MATH_1P5B:-/workspace/models/Qwen2.5-Math-1.5B-Instruct}"
DATA_ROOT="${GXPO_DATA_ROOT:-/workspace/data}"
export MODEL_QWEN25_MATH_1P5B="$MODEL_DIR"
export GXPO_DATA_ROOT="$DATA_ROOT"
export GXPO_RESULTS_ROOT="${GXPO_RESULTS_ROOT:-$REPO_ROOT/results/gxpo_efficiency}"

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

case "$GPU_COUNT" in
  ''|*[!0-9]*) echo "PREFLIGHT FAIL: GPU_COUNT must be a positive integer (got $GPU_COUNT)" >&2; MISSING=1 ;;
  0) echo "PREFLIGHT FAIL: GPU_COUNT must be greater than zero" >&2; MISSING=1 ;;
esac
case "$TRAIN_BATCH_SIZE" in
  ''|*[!0-9]*) echo "PREFLIGHT FAIL: TRAIN_BATCH_SIZE must be an integer (got $TRAIN_BATCH_SIZE)" >&2; MISSING=1 ;;
  0) echo "PREFLIGHT FAIL: TRAIN_BATCH_SIZE must be greater than zero" >&2; MISSING=1 ;;
esac
case "$PPO_MINI_BATCH_SIZE" in
  ''|*[!0-9]*) echo "PREFLIGHT FAIL: PPO_MINI_BATCH_SIZE must be an integer (got $PPO_MINI_BATCH_SIZE)" >&2; MISSING=1 ;;
  0) echo "PREFLIGHT FAIL: PPO_MINI_BATCH_SIZE must be greater than zero" >&2; MISSING=1 ;;
esac
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
  method             : gxpo + ${OPTIMIZER_NAME:-muon} (K=${K:-10}, alpha=${REPOSITION_ALPHA:-1.0})
  batch / minibatch  : ${TRAIN_BATCH_SIZE:-64} / ${PPO_MINI_BATCH_SIZE:-16}
  gpus               : ${GPU_COUNT:-1}  (ids ${GPU_IDS:-2}, FSDP_SIZE=${FSDP_SIZE:-1})
  max_steps          : ${MAX_STEPS:-400}   save_freq ${SAVE_FREQ:-20}
  val before train   : ${VAL_BEFORE_TRAIN}
  optimizer          : ${OPTIMIZER_NAME:-muon} (momentum ${MUON_MOMENTUM:-0.95}, NS ${MUON_NS_STEPS:-5}, nesterov ${MUON_NESTEROV:-True}, wd ${MUON_WEIGHT_DECAY:-1e-2})
  muon backend       : ${MUON_DISTRIBUTED_BACKEND:-gather_scatter}
  dtype / liger      : ${ACTOR_MODEL_DTYPE:-float32} / ${USE_LIGER:-True}
  attention          : train ${ATTN_IMPL:-flash_attention_2} | vllm ${VLLM_ATTENTION_BACKEND:-FLASHINFER}
  memory caps        : PPO ${PPO_MAX_TOKEN_LEN_PER_GPU} tokens | vllm ${VLLM_MAX_NUM_BATCHED_TOKENS} tokens / ${VLLM_MAX_NUM_SEQS} seqs | vllm util ${VLLM_GPU_MEMORY_UTILIZATION}
  --- gate v2 ---
  trigger_signal     : $GXPO_TRIGGER_SIGNAL      (must not be 'entropy')
  shutoff_mode       : $GXPO_SHUTOFF_MODE        (disagreement = 1 - |cos(g0,g_slow)|)
  abs threshold      : $GXPO_TRIGGER_ABS_THRESHOLD (0 = use z-score path)
  tau / patience     : $GXPO_TAU / $GXPO_TRIGGER_PATIENCE (ordinary z-score)
  robust statistic   : $GXPO_TRIGGER_ROBUST      (0 = ordinary mean/std)
  zscore window      : $GXPO_ZSCORE_W           (preceding disagreement values)
  sustain window     : $GXPO_TRIGGER_SUSTAIN_W   (ignored in z-score mode)
  min_obs age floor  : $GXPO_TRIGGER_MIN_OBS
  max_active_steps   : $GXPO_MAX_ACTIVE_STEPS    (hard runtime ceiling)
  wandb project      : ${WANDB_PROJECT:-gxpo-efficiency-final}
[dry-run] preflight OK - would launch now.
EOT
  exit 0
fi

# ---------------------------------------------------------------- launch -----
# Use the shared RL launcher directly; this entrypoint pins OPTIMIZER_NAME=muon.
export MODEL_ALIAS="${MODEL_ALIAS:-qwen25-math-1p5b}"
export MODEL_ID="${MODEL_ID:-$MODEL_DIR}"
export METHOD="${METHOD:-gxpo}"
exec bash "$SCRIPT_DIR/common.sh"
