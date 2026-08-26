#!/usr/bin/env bash
#
# qwen25_math_1p5b_gxpo_b256_mb64_gate_v6.sh
#
# Complete entrypoint: Qwen2.5-Math-1.5B-Instruct | GXPO | batch 256 |
# minibatch 64 | K=10 | alpha=0.3 | 2 GPUs (Blackwell 6000 Pro class)
# driven by the Gate-v2 prediction-quality trigger.
#
# Gate configuration - MODERATE PROFILE (evidence: Code/SFPO/.audit/gxpo_algorithm_findings.md):
#   signal    : grad (actor-side; disagreement = 1 - |cos(g0, g_slow)| from pre-clip grads)
#   primary   : sustained level - rolling median of last 10 observations >= 0.15,
#               held for 2 consecutive scored batches (zero false positives across
#               all 7 production runs in replay; catches failing runs ~5x earlier
#               than the entropy gate, which missed them entirely)
#   backup    : robust median/MAD z-score path, used only if ABS_THRESHOLD=0
#   age floor : no trip before 12 scored post-warmup batches (rides out the
#               volatile window that caused all observed production trips)
#   budget    : hard stop after 150 enabled steps regardless of gate (runtime cap)
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
export K=10
export REPOSITION_ALPHA=0.3

# Actor-side prediction-quality gate. GXPO_TRIGGER_SIGNAL must differ from
# 'entropy' or common.sh warns and the trainer entropy gate keeps control.
export GXPO_TRIGGER_SIGNAL="${GXPO_TRIGGER_SIGNAL:-grad}"
export GXPO_SHUTOFF_MODE="${GXPO_SHUTOFF_MODE:-cosine}"
# PRIMARY criterion (calibrated on 7 production runs, see .audit/gxpo_algorithm_findings.md):
# trip when the rolling median of the last 10 disagreement observations stays >= 0.15
# for 2 consecutive batches. Replay: healthy runs never trip; failing k10 trips @55;
# diverged no-fallback trips @90; zero false positives.
export GXPO_TRIGGER_ABS_THRESHOLD="${GXPO_TRIGGER_ABS_THRESHOLD:-0.15}"
export GXPO_TRIGGER_SUSTAIN_W="${GXPO_TRIGGER_SUSTAIN_W:-10}"
# SECONDARY z-score path (used only when ABS_THRESHOLD=0): robust median/MAD.
export GXPO_TRIGGER_ROBUST="${GXPO_TRIGGER_ROBUST:-1}"
export GXPO_TAU="${GXPO_TAU:-2.0}"
export GXPO_TRIGGER_MIN_OBS="${GXPO_TRIGGER_MIN_OBS:-12}"
export GXPO_MAX_ACTIVE_STEPS="${GXPO_MAX_ACTIVE_STEPS:-150}"
export GXPO_TRIGGER_PATIENCE="${GXPO_TRIGGER_PATIENCE:-2}"
# Window length is inherited from common.sh (GXPO_ZSCORE_W=30).

# ------------------------------------------------------------- preflight -----
MISSING=0
MODEL_DIR="${MODEL_QWEN25_MATH_1P5B:-$REPO_ROOT/models/Qwen2.5-Math-1.5B-Instruct}"
DATA_ROOT="${GXPO_DATA_ROOT:-$REPO_ROOT/Code/SFPO/data}"

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
  method             : gxpo (K=${K:-10}, alpha=${REPOSITION_ALPHA:-0.3})
  batch / minibatch  : ${TRAIN_BATCH_SIZE:-256} / ${PPO_MINI_BATCH_SIZE:-64}
  gpus               : ${GPU_COUNT:-2}  (ids ${GPU_IDS:-0,1}, FSDP_SIZE=${FSDP_SIZE:-2})
  max_steps          : ${MAX_STEPS:-400}   save_freq ${SAVE_FREQ:-20}
  dtype / liger      : ${ACTOR_MODEL_DTYPE:-float32} / ${USE_LIGER:-True}
  attention          : train ${ATTN_IMPL:-flash_attention_2} | vllm ${VLLM_ATTENTION_BACKEND:-FLASHINFER}
  --- gate v2 ---
  trigger_signal     : $GXPO_TRIGGER_SIGNAL      (must not be 'entropy')
  shutoff_mode       : $GXPO_SHUTOFF_MODE        (disagreement = 1 - |cos(g0,g_slow)|)
  abs threshold      : $GXPO_TRIGGER_ABS_THRESHOLD (rolling median of last ${GXPO_TRIGGER_SUSTAIN_W:-10})
  tau / patience     : $GXPO_TAU / $GXPO_TRIGGER_PATIENCE (z-path backup when abs=0)
  robust statistic   : $GXPO_TRIGGER_ROBUST      (median/MAD, sigma floor 10%)
  min_obs age floor  : $GXPO_TRIGGER_MIN_OBS      (moderate profile)
  max_active_steps   : $GXPO_MAX_ACTIVE_STEPS    (hard runtime ceiling)
  zscore window      : ${GXPO_ZSCORE_W:-30}
  wandb project      : ${WANDB_PROJECT:-gxpo-efficiency-final}
[dry-run] preflight OK - would launch now.
EOT
  exit 0
fi

# ---------------------------------------------------------------- launch -----
# Hands off to the proven b256 wrapper (env setup, caches, common.sh chain).
exec bash "$SCRIPT_DIR/qwen25_math_1p5b_gxpo_b256_a05.sh"
