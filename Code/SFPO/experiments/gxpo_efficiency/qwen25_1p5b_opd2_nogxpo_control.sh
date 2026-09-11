#!/usr/bin/env bash
#
# qwen25_1p5b_opd2_nogxpo_control.sh
#
# Qwen2.5-1.5B-Instruct (student) | OPD^2 delta distillation | plain AdamW, NO GXPO
#
# The A/B control for qwen25_1p5b_opd2_gxpo_adamw_dir_k10.sh: byte-identical
# OPD^2 configuration and hyperparameters, with METHOD=grpo so the actor takes a
# single ordinary PPO-clipped step per mini-batch instead of GXPO's 3-pass
# extrapolated update. The only substantive difference between the two runs is
# the optimizer-level update rule, which is exactly what the comparison is for.
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
#   student      : Qwen/Qwen2.5-1.5B-Instruct
#   teacher      : Qwen/Qwen2.5-Math-1.5B-Instruct
#   teacher_base : Qwen/Qwen2.5-Math-1.5B     (the teacher's pre-instruct base)
#
# All three share one tokenizer; the chat templates differ only in the default
# system prompt, which SYSTEM_PROMPT overrides, so prompt ids are identical.
#
# Hyperparameters follow the paper's recipe
# (on-policy-delta/opd2/recipes/Qwen3-1.7B/opd2/config_open_nvidia_100k.yaml):
# effective batch 256, one response per prompt, 100 steps, lr 5e-6 with a cosine
# schedule floored at 0.1x and 10% warmup, temperature 0.7, top-K 1024 instead of
# the full vocabulary, gen_loss_weight 0.1, reference KL fully disabled.
# Deviation: 3072-token responses (paper 8192) -- Qwen2.5-Math has 4096
# positions and prompts take up to 1024; common.sh's preflight enforces it.
#
# Usage:
#   bash qwen25_1p5b_opd2_nogxpo_control.sh            # launch
#   bash qwen25_1p5b_opd2_nogxpo_control.sh --dry-run  # resolved config, no launch
#
# Everything is overridable from the environment, e.g.:
#   MAX_STEPS=200 GPU_COUNT=2 bash qwen25_1p5b_opd2_nogxpo_control.sh
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
export OPD2_TEACHER="${OPD2_TEACHER:-/office/dev_workspace/swapnil/gradient-extrapolation-based-policy-optimization-gxpo-speed-audit/models/Qwen2.5-Math-1.5B-Instruct}"
export OPD2_TEACHER_BASE="${OPD2_TEACHER_BASE:-$REPO_ROOT/models/Qwen2.5-Math-1.5B}"
# top-K truncation for the E_base[.] mean corrections. The weight is the
# student's own probability, ~0 outside its own top-K, so this is near-lossless.
# <=0 selects the exact full-vocabulary path through the same code.
export OPD2_TOPK="${OPD2_TOPK:-1024}"
export OPD2_GEN_LOSS_WEIGHT="${OPD2_GEN_LOSS_WEIGHT:-0.1}"
export OPD2_REWARDS_BIAS="${OPD2_REWARDS_BIAS:-0.0}"
# The teacher prompt is re-rendered with the teacher's own chat_template (the
# reference's opd_no_think_teacher); teacher_base shares it. For this trio the
# render is id-identical to the student's and the id contract is a no-op.
# ENABLE_THINKING only affects Qwen3 templates; Qwen2.5 ignores it.
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
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
export SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant. Solve the problem carefully and provide a clear final answer.}"
# beta=0.0 in the recipe: no reference-policy KL anywhere. common.sh's OPD^2
# preflight rejects anything else.
export USE_KL_LOSS="False"
export KL_LOSS_COEF="0.0"
export SAVE_FREQ="${SAVE_FREQ:-20}"

# ------------------------------------------------------------ optimizer ------
# METHOD=grpo below pins +actor_rollout_ref.actor.use_gxpo=False, so K,
# REPOSITION_ALPHA and every GXPO_* knob are inert here by construction.
export OPTIMIZER_NAME="${OPTIMIZER_NAME:-adamw}"

export ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
# The three frozen/student forwards per response are on top of the usual RL step,
# and responses run to 3072 tokens. Leave vLLM more headroom than the reward-RL
# arms do.
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"

# ------------------------------------------------------------- preflight -----
MISSING=0
GXPO_ASSET_ROOT="/office/dev_workspace/swapnil/gradient-extrapolation-based-policy-optimization-gxpo-speed-audit"
MODEL_DIR="${STUDENT_MODEL:-/office/shared_cache/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
DATA_ROOT="${GXPO_DATA_ROOT:-$GXPO_ASSET_ROOT/Code/SFPO/data}"
export MODEL_QWEN25_1P5B_INSTRUCT="$MODEL_DIR"
export GXPO_DATA_ROOT="$DATA_ROOT"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "PREFLIGHT FAIL: student weights not found at $MODEL_DIR" >&2
  echo "  (point STUDENT_MODEL at a local Qwen2.5-1.5B-Instruct copy)" >&2
  MISSING=1
fi
for _label in TEACHER:"$OPD2_TEACHER" TEACHER_BASE:"$OPD2_TEACHER_BASE"; do
  _name="${_label%%:*}"; _path="${_label#*:}"
  if [[ ! -f "$_path/config.json" ]]; then
    echo "PREFLIGHT FAIL: OPD^2 $_name weights not found at $_path" >&2
    if [[ "$_name" == "TEACHER_BASE" ]]; then
      echo "  hf download Qwen/Qwen2.5-Math-1.5B --local-dir $_path" >&2
    else
      echo "  hf download Qwen/Qwen2.5-Math-1.5B-Instruct --local-dir $_path" >&2
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
  method             : opd2 + plain adamw (no GXPO)
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
  attention          : train $ATTN_IMPL | vllm ${VLLM_ATTENTION_BACKEND:-FLASHINFER} (util $VLLM_GPU_MEMORY_UTILIZATION)
  wandb project      : ${WANDB_PROJECT:-gxpo-efficiency-final}
  gxpo arm           : qwen25_1p5b_opd2_gxpo_adamw_dir_k10.sh
[dry-run] preflight OK - would launch now.
EOT
  exit 0
fi

# ---------------------------------------------------------------- launch -----
MODEL_ALIAS="${MODEL_ALIAS:-qwen25-1p5b-qmath}"
MODEL_ID="$MODEL_QWEN25_1P5B_INSTRUCT"
METHOD="grpo"
source "$SCRIPT_DIR/common.sh"
