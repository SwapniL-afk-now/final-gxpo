#!/usr/bin/env bash
#
# qwen25_1p5b_opd2_gxpo_adamw_dir_k10.sh
#
# Qwen3-1.7B non-thinking (student) | OPD^2 delta distillation | GXPO + AdamW |
# TRANSACTIONAL optimizer state | OPTIMIZER-DIRECTION retention | K=10 | alpha=0.3
#
# OPD^2 (On-Policy Delta Distillation, arXiv:2607.15161, NAVER AI Lab) replaces
# the verifier advantage with a DENSE PER-TOKEN signal built from the delta
# between a reasoning-tuned teacher and its pre-reasoning-tuning base:
#
#   signal = (teacher_gt - teacher_base_gt) - (E_base[teacher] - E_base[teacher_base])
#   d_base = (teacher_gt - student_gt)      - (E_base[teacher] - E_base[student])
#   signal = 0 wherever signal * d_base < 0        # direction gate
#   advantage = signal * gen_loss_weight
#
# where E_base[X] = sum_v p_student(v) * X_v over the STUDENT's top-K columns.
# This is an ADVANTAGE SOURCE; GXPO is an OPTIMIZER (3-pass probe/reposition/
# correct below the loss). They compose with zero changes to either: the PPO
# clip, the retention estimator and the trigger gate are untouched.
#
#   student      : Qwen/Qwen3-1.7B            (non-thinking mode, ENABLE_THINKING=False)
#   teacher      : Qwen/Qwen3-4B-Instruct-2507
#   teacher_base : Qwen/Qwen3-4B-Base         (the paper's exact pair)
#
# Every hyperparameter below is the paper's own recipe
# (on-policy-delta/opd2/recipes/Qwen3-1.7B/opd2/config_open_nvidia_100k.yaml):
# effective batch 256, one response per prompt, 100 steps, lr 5e-6 with a cosine
# schedule floored at 0.1x and 10% warmup, temperature 0.7, 8192-token responses,
# top-K 1024 instead of the full vocabulary, gen_loss_weight 0.1, reference KL
# fully disabled. Models are the paper's own; only the dataset differs.
#
# Usage:
#   bash qwen25_1p5b_opd2_gxpo_adamw_dir_k10.sh            # launch
#   bash qwen25_1p5b_opd2_gxpo_adamw_dir_k10.sh --dry-run  # resolved config, no launch
#
# Everything is overridable from the environment, e.g.:
#   MAX_STEPS=200 GPU_COUNT=2 bash qwen25_1p5b_opd2_gxpo_adamw_dir_k10.sh
#
# Control arm (same config, GXPO off):
#   qwen25_1p5b_opd2_nogxpo_control.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"       # checkout root (holds .env, models/, Code/)
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ---------------------------------------------------------------- secrets ----
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

# --------------------------------------------------------------- OPD^2 cfg ---
export OPD2_ENABLED=1
export OPD2_TEACHER="${OPD2_TEACHER:-$REPO_ROOT/models/Qwen3-4B-Instruct-2507}"
export OPD2_TEACHER_BASE="${OPD2_TEACHER_BASE:-$REPO_ROOT/models/Qwen3-4B-Base}"
# top-K truncation for the E_base[.] mean corrections. The weight is the
# student's own probability, ~0 outside its own top-K, so this is near-lossless.
# <=0 selects the exact full-vocabulary path through the same code.
export OPD2_TOPK="${OPD2_TOPK:-1024}"
export OPD2_GEN_LOSS_WEIGHT="${OPD2_GEN_LOSS_WEIGHT:-0.1}"
export OPD2_REWARDS_BIAS="${OPD2_REWARDS_BIAS:-0.0}"
# The student prompt is rendered non-thinking (Qwen3 template + enable_thinking=
# False appends an empty <think></think> block); Qwen3-4B-Instruct-2507 has no
# think mode, so its prompt is re-rendered with its own chat_template -- the
# reference's opd_no_think_teacher. teacher_base shares the teacher's prompt.
# The Qwen3 tokenizers share every id, so the id contract is a no-op.
export OPD2_TEACHER_TEMPLATE="${OPD2_TEACHER_TEMPLATE:-True}"
export ENABLE_THINKING="${ENABLE_THINKING:-False}"
# Rows are scored one at a time by default, exactly like the reference trainer.
# Raise for throughput once VRAM headroom is measured (rows are length-sorted and
# right-padded, which is safe under causal attention).
export OPD2_MICRO_BATCH_SIZE="${OPD2_MICRO_BATCH_SIZE:-1}"
export OPD2_CHUNK_TOKENS="${OPD2_CHUNK_TOKENS:-512}"
# Teacher + teacher_base are parked on CPU between scoring phases so they are
# never co-resident with the vLLM rollout engine. Set True to keep ~7GB of bf16
# weights resident and trade VRAM for the per-step transfer.
export OPD2_KEEP_ON_GPU="${OPD2_KEEP_ON_GPU:-False}"

# ------------------------------------------------------- paper hyperparams ---
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
# num_generations=1 in the recipe. The delta signal is dense and per-token, so
# there is no group baseline to estimate and nothing to gain from n>1.
export ROLLOUT_N="${ROLLOUT_N:-1}"
export MAX_STEPS="${MAX_STEPS:-100}"
export LR="${LR:-5e-6}"
export LR_WARMUP_STYLE="${LR_WARMUP_STYLE:-cosine}"
export LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.1}"
export LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
export SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant. Solve the problem carefully and provide a clear final answer.}"
# beta=0.0 in the recipe: no reference-policy KL anywhere. common.sh's OPD^2
# preflight rejects anything else.
export USE_KL_LOSS="False"
export KL_LOSS_COEF="0.0"
export SAVE_FREQ="${SAVE_FREQ:-20}"

# ---------------------------------------------------------------- GXPO cfg ---
export K="${K:-10}"
export REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.3}"
export OPTIMIZER_NAME="${OPTIMIZER_NAME:-adamw}"
export GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"
# auto = AdamW optimizer-direction retention r = d1/d0 (see
# GXPO_OPTIMIZER_AWARE_RETENTION.md). Set to grad for the legacy g1/g0 arm.
export GXPO_RETENTION_SPACE="${GXPO_RETENTION_SPACE:-auto}"
# common.sh defaults the shutoff-gate warmup to 50 steps, which was chosen for
# 400-step runs. This run is 100 steps, so scale it to keep the same 12.5%
# fraction -- otherwise half the run would sit in a regime where the gate cannot
# trip at all.
export GXPO_WARMUP_STEPS="${GXPO_WARMUP_STEPS:-12}"
export GXPO_RESET_ENTROPY_AFTER_WARMUP="${GXPO_RESET_ENTROPY_AFTER_WARMUP:-True}"

export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
# The three frozen/student forwards per response are on top of the usual RL step,
# and responses run to 8192 tokens. Leave vLLM more headroom than the reward-RL
# arms do.
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"

# ------------------------------------------------------------- preflight -----
MISSING=0
GXPO_ASSET_ROOT="/office/dev_workspace/swapnil/gradient-extrapolation-based-policy-optimization-gxpo-speed-audit"
MODEL_DIR="${STUDENT_MODEL:-/office/shared_cache/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
DATA_ROOT="${GXPO_DATA_ROOT:-$GXPO_ASSET_ROOT/Code/SFPO/data}"
export MODEL_QWEN25_1P5B_INSTRUCT="$MODEL_DIR"
export GXPO_DATA_ROOT="$DATA_ROOT"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "PREFLIGHT FAIL: student weights not found at $MODEL_DIR" >&2
  echo "  (point STUDENT_MODEL at a local Qwen3-1.7B copy)" >&2
  MISSING=1
fi
for _label in TEACHER:"$OPD2_TEACHER" TEACHER_BASE:"$OPD2_TEACHER_BASE"; do
  _name="${_label%%:*}"; _path="${_label#*:}"
  if [[ ! -f "$_path/config.json" ]]; then
    echo "PREFLIGHT FAIL: OPD^2 $_name weights not found at $_path" >&2
    if [[ "$_name" == "TEACHER_BASE" ]]; then
      echo "  hf download Qwen/Qwen3-4B-Base --local-dir $_path" >&2
    else
      echo "  hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir $_path" >&2
    fi
    MISSING=1
  fi
done
unset _label _name _path

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
  student            : $MODEL_DIR
  teacher            : $OPD2_TEACHER
  teacher_base       : $OPD2_TEACHER_BASE
  data_root          : $DATA_ROOT
  method             : opd2 + gxpo + adamw (K=$K, alpha=$REPOSITION_ALPHA)
  advantage          : OPD^2 delta signal x $OPD2_GEN_LOSS_WEIGHT (verifier reward is metrics-only)
  opd2 top_k         : $OPD2_TOPK   bias $OPD2_REWARDS_BIAS   teacher_template $OPD2_TEACHER_TEMPLATE
  opd2 scoring       : micro_bsz $OPD2_MICRO_BATCH_SIZE, chunk $OPD2_CHUNK_TOKENS tokens, keep_on_gpu $OPD2_KEEP_ON_GPU
  batch / minibatch  : $TRAIN_BATCH_SIZE / $PPO_MINI_BATCH_SIZE   rollout_n $ROLLOUT_N
  max_response_len   : $MAX_RESPONSE_LENGTH   temperature $ROLLOUT_TEMPERATURE
  lr                 : $LR ($LR_WARMUP_STYLE, warmup $LR_WARMUP_RATIO, min_lr_ratio $LR_MIN_RATIO)
  ref KL             : disabled (use_kl_loss=$USE_KL_LOSS, coef=$KL_LOSS_COEF)
  gpus               : ${GPU_COUNT:-1}  (ids ${GPU_IDS:-<inherited>}, FSDP_SIZE=${FSDP_SIZE:-1})
  max_steps          : $MAX_STEPS   save_freq $SAVE_FREQ
  optimizer          : $OPTIMIZER_NAME
  optimizer_state    : $GXPO_OPTIMIZER_STATE_MODE
  retention_space    : $GXPO_RETENTION_SPACE
  gxpo scale         : clamp [1, $(awk "BEGIN{print $K/2+1}")], min_effective_multiplier ${GXPO_MIN_EFFECTIVE_MULTIPLIER:-0}
  gxpo warmup        : ${GXPO_WARMUP_STEPS:-50} steps (gates the instability shutoff, not GXPO itself)
  entropy_coeff      : ${ENTROPY_COEFF:-0} (OPD^2 reference uses 0; nonzero needs OPD2_ALLOW_ENTROPY=1)
  attention          : train $ATTN_IMPL | vllm ${VLLM_ATTENTION_BACKEND:-FLASHINFER} (util $VLLM_GPU_MEMORY_UTILIZATION)
  wandb project      : ${WANDB_PROJECT:-gxpo-efficiency-final}
  control arm        : qwen25_1p5b_opd2_nogxpo_control.sh
[dry-run] preflight OK - would launch now.
EOT
  exit 0
fi

# ---------------------------------------------------------------- launch -----
MODEL_ALIAS="${MODEL_ALIAS:-qwen3-1p7b}"
MODEL_ID="$MODEL_QWEN25_1P5B_INSTRUCT"
METHOD="gxpo"
source "$SCRIPT_DIR/common.sh"
