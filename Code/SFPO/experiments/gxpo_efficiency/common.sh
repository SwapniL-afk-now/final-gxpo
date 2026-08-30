#!/usr/bin/env bash
# Shared launcher for the final 3-model x 3-method GXPO efficiency matrix.
# All fairness-critical settings live here; the nine entrypoints only select
# MODEL_ALIAS, MODEL_ID, and METHOD.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Make every launch reproducible from a fresh tmux shell.  The base image's
# venv is not automatically activated, and FA3/dependency wheels are kept in
# user-writable workspace paths on this unprivileged instance.
GXPO_PROJECT_ROOT="$(cd -- "$REPO_ROOT/../.." && pwd)"
if [[ -x "$GXPO_PROJECT_ROOT/.venv/bin/python" ]]; then
  export VIRTUAL_ENV="$GXPO_PROJECT_ROOT/.venv"
  export PATH="$GXPO_PROJECT_ROOT/.venv/bin:$PATH"
fi
export PYTHONPATH="$REPO_ROOT/.runtime_deps:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_home}"
# A system-provided HF_HOME can exist but still be unwritable by the training
# user.  Fall back to the repository cache instead of failing inside Ray's
# remote main task with an opaque worker shutdown.
if [[ ! -w "$HF_HOME/hub" ]]; then
  export HF_HOME="$REPO_ROOT/.hf_home"
fi
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HUB_CACHE"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_3}"
export ATTN_IMPL
python - "$ATTN_IMPL" <<'PY'
import importlib.util
import sys

attn_impl = sys.argv[1]
required = ['pandas', 'wandb', 'tensordict']
if attn_impl == 'flash_attention_2':
    required.append('flash_attn')
else:
    required.extend(['flash_attn_3', 'flash_attn_interface'])
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing runtime dependencies for {attn_impl}: {', '.join(missing)}")
PY

REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.5}"
K="${K:-5}"
TRAIN_SEED="${TRAIN_SEED:-3407}"
FINAL_EVAL_SEEDS="${FINAL_EVAL_SEEDS:-0 1 2 3}"
MAX_STEPS="${MAX_STEPS:-400}"
SAVE_FREQ="${SAVE_FREQ:-5}"
SFPO_WARMUP_STEPS="${SFPO_WARMUP_STEPS:-50}"
GXPO_WARMUP_STEPS="${GXPO_WARMUP_STEPS:-50}"
GXPO_TAU="${GXPO_TAU:-3.0}"
GXPO_ZSCORE_W="${GXPO_ZSCORE_W:-30}"
GXPO_TRIGGER_PATIENCE="${GXPO_TRIGGER_PATIENCE:-3}"
GXPO_FALLBACK_MODE="${GXPO_FALLBACK_MODE:-permanent}"
GXPO_FALLBACK_WINDOW="${GXPO_FALLBACK_WINDOW:-10}"
GXPO_ACTOR_DUTY_CYCLE="${GXPO_ACTOR_DUTY_CYCLE:-0}"
GXPO_DIAG_FREQ="${GXPO_DIAG_FREQ:-10}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
SFPO_ZSCORE_THRESHOLD="${SFPO_ZSCORE_THRESHOLD:-2.5}"
SFPO_TRIGGER_PATIENCE="${SFPO_TRIGGER_PATIENCE:-3}"
SFPO_RESET_ENTROPY_AFTER_WARMUP="${SFPO_RESET_ENTROPY_AFTER_WARMUP:-True}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
ROLLOUT_N="${ROLLOUT_N:-8}"
LR="${LR:-1e-6}"
USE_LIGER="${USE_LIGER:-True}"
OPTIM_FUSED="${OPTIM_FUSED:-False}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-True}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-True}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-}"
TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-5}"
TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-auto}"
TRAINER_RESUME_FROM_PATH="${TRAINER_RESUME_FROM_PATH:-False}"
GPU_COUNT="${GPU_COUNT:-${N_GPUS:-1}}"
PROJECT="${WANDB_PROJECT:-gxpo-efficiency-final}"

if [[ -z "${MODEL_ALIAS:-}" || -z "${MODEL_ID:-}" || -z "${METHOD:-}" ]]; then
  echo "common.sh requires MODEL_ALIAS, MODEL_ID, and METHOD" >&2
  exit 2
fi
case "$METHOD" in
  grpo|sfpo|gxpo) ;;
  *) echo "Unsupported METHOD=$METHOD" >&2; exit 2 ;;
esac

DATA_ROOT="${GXPO_DATA_ROOT:-$REPO_ROOT/data}"
DAPO_TRAIN="${DAPO_TRAIN:-$DATA_ROOT/dapo_math/train.parquet}"
LIGHTEVAL_TRAIN="${LIGHTEVAL_TRAIN:-$DATA_ROOT/lighteval-math/train.parquet}"
MATH500="${MATH500:-$DATA_ROOT/math500/test.parquet}"
AIME24="${AIME24:-$DATA_ROOT/aime2024/test.parquet}"
AIME25="${AIME25:-$DATA_ROOT/aime2025/test.parquet}"
AMC23="${AMC23:-$DATA_ROOT/amc/test.parquet}"
MINERVA="${MINERVA:-$DATA_ROOT/minervamath/test.parquet}"
OLYMPIAD="${OLYMPIAD:-$DATA_ROOT/olympiadbench/test.parquet}"

missing=0
for required in "$DAPO_TRAIN" "$LIGHTEVAL_TRAIN" "$MATH500" "$AIME24" "$AIME25" "$AMC23" "$MINERVA" "$OLYMPIAD"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing prepared dataset: $required" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "Set GXPO_DATA_ROOT or the individual DAPO_TRAIN/LIGHTEVAL_TRAIN/benchmark variables." >&2
  exit 2
fi

if [[ "$MODEL_ID" == /* && ! -e "$MODEL_ID" ]]; then
  echo "Configured local model path does not exist: $MODEL_ID" >&2
  exit 2
fi

RUN_NAME="${GXPO_RUN_NAME:-${MODEL_ALIAS}_${METHOD}${METHOD:+_}$( [[ "$METHOD" == grpo ]] && echo "" || echo "k${K}_" )seed${TRAIN_SEED}}"
# Remove the accidental doubled separator for GRPO while keeping names explicit.
RUN_NAME="${RUN_NAME//__/_}"
RESULT_ROOT="${GXPO_RESULTS_ROOT:-$REPO_ROOT/results/gxpo_efficiency}"
RUN_DIR="$RESULT_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR"

# Keep vLLM and FlashInfer autotune artifacts writable and isolated per run.
export VLLM_CACHE_ROOT="$RUN_DIR/vllm_cache"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$RUN_DIR/flashinfer_autotune_cache"
export VLLM_SLEEP_LEVEL="${VLLM_SLEEP_LEVEL:-2}"
mkdir -p "$VLLM_CACHE_ROOT" "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"

export GXPO_EFFICIENCY_RUN=1
export GXPO_RUN_NAME="$RUN_NAME"
export GXPO_MODEL_ALIAS="$MODEL_ALIAS"
export TRAIN_SEED
export FINAL_EVAL_SEEDS
export WANDB_PROJECT="$PROJECT"
if [[ ! -v WANDB_GROUP || -z "$WANDB_GROUP" ]]; then
  export WANDB_GROUP="$MODEL_ALIAS"
fi
if [[ ! -v WANDB_TAGS || -z "$WANDB_TAGS" ]]; then
  if [[ "$METHOD" == "grpo" ]]; then
    export WANDB_TAGS="model:$MODEL_ALIAS,method:$METHOD,experiment:final-efficiency"
  else
    export WANDB_TAGS="model:$MODEL_ALIAS,method:$METHOD,k:$K,alpha:$REPOSITION_ALPHA,experiment:final-efficiency"
  fi
fi
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="$RUN_DIR"
export RAY_ADDRESS=local
export MPLBACKEND=Agg

if [[ -n "${GPU_IDS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi

# A duty-cycle sleep reduces sustained load but cannot guarantee an
# instantaneous board-power ceiling. For power-sensitive GXPO launches, fail
# closed unless the physical NVIDIA power limit is already at or below the
# configured maximum. Applying the limit requires an administrator.
if [[ "${GXPO_ENFORCE_POWER_LIMIT:-False}" == "True" || "${GXPO_ENFORCE_POWER_LIMIT:-0}" == "1" ]]; then
  GXPO_MAX_POWER_W="${GXPO_MAX_POWER_W:-500}"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "GXPO power safety: nvidia-smi is unavailable; refusing to launch." >&2
    exit 2
  fi
  gxpo_power_limits="$(nvidia-smi --id="$GPU_IDS" \
    --query-gpu=index,power.limit --format=csv,noheader,nounits 2>/dev/null)" || {
    echo "GXPO power safety: unable to read power limits for GPUs $GPU_IDS; refusing to launch." >&2
    exit 2
  }
  while IFS=',' read -r gxpo_gpu_id gxpo_power_limit; do
    [[ -z "$gxpo_gpu_id" ]] && continue
    if awk -v limit="$gxpo_power_limit" -v maximum="$GXPO_MAX_POWER_W" \
      'BEGIN { exit !(limit+0 > maximum+0) }'; then
      echo "GXPO power safety: GPU ${gxpo_gpu_id//[[:space:]]/} limit=${gxpo_power_limit}W exceeds ${GXPO_MAX_POWER_W}W; refusing to launch." >&2
      echo "Ask an administrator to run: sudo nvidia-smi -i $GPU_IDS -pl $GXPO_MAX_POWER_W" >&2
      exit 2
    fi
  done <<< "$gxpo_power_limits"
  echo "GXPO power safety: physical GPU limits verified <= ${GXPO_MAX_POWER_W}W"
fi

TRAIN_FILES="['$DAPO_TRAIN','$LIGHTEVAL_TRAIN']"
VAL_FILES="['$MATH500','$AIME24','$AIME25','$AMC23','$MINERVA','$OLYMPIAD']"

METHOD_FLAGS=()
# Gate v2 (opt-in; see .audit/gxpo_algorithm_findings.md):
#   GXPO_TRIGGER_ROBUST=1   -> median/MAD z-score (resists early-warmup transient bursts)
#   GXPO_TRIGGER_MIN_OBS=N  -> gate cannot trip until N scored post-warmup observations
# Prediction-quality gating instead of the trainer entropy gate requires a custom
# entrypoint: gxpo_trigger_signal != entropy plus gxpo_shutoff_mode=cosine, which makes
# the actor gate on disagreement = 1 - |cos(g0, g_slow)|.

# Gate v2 passthroughs (defaults preserve the historical behavior exactly).
GXPO_TRIGGER_SIGNAL="${GXPO_TRIGGER_SIGNAL:-entropy}"
GXPO_SHUTOFF_MODE="${GXPO_SHUTOFF_MODE:-trajectory_aware}"
export GXPO_TRIGGER_SIGNAL GXPO_SHUTOFF_MODE

if [[ "$GXPO_SHUTOFF_MODE" == "cosine" && "$GXPO_TRIGGER_SIGNAL" == "entropy" ]]; then
  echo "WARNING: GXPO_SHUTOFF_MODE=cosine is INERT with GXPO_TRIGGER_SIGNAL=entropy:" >&2
  echo "         the trainer entropy gate makes the trip decision; cosine stats are logged only." >&2
fi
# Boolean footgun guard: ${VAR:+..} would fire on '0'/'false'; require an explicit yes.
case "${GXPO_TRIGGER_ROBUST:-}" in
  1|true|True|yes) METHOD_FLAGS+=(+actor_rollout_ref.actor.gxpo_trigger_robust=True) ;;
  ""|0|false|False|no) : ;;
  *) echo "WARNING: GXPO_TRIGGER_ROBUST='$GXPO_TRIGGER_ROBUST' not recognized; ignoring." >&2 ;;
esac
if [[ -n "${GXPO_TRIGGER_MIN_OBS:-}" && "$GXPO_TRIGGER_SIGNAL" == "entropy" ]]; then
  echo "WARNING: GXPO_TRIGGER_MIN_OBS only affects the actor-side gate; inert with signal=entropy." >&2
fi
if [[ -n "${GXPO_TRIGGER_ROBUST:-}" && "$GXPO_TRIGGER_SIGNAL" == "entropy" ]]; then
  echo "WARNING: GXPO_TRIGGER_ROBUST only affects the actor-side gate; inert with signal=entropy." >&2
fi
case "${GXPO_TRIGGER_ABS_THRESHOLD:-}" in
  ""|0|0.0) : ;;
  *) if [[ "$GXPO_TRIGGER_SIGNAL" == "entropy" || "$GXPO_SHUTOFF_MODE" != "cosine" ]]; then
       echo "WARNING: GXPO_TRIGGER_ABS_THRESHOLD only applies to cosine mode with signal!=entropy; inert here." >&2
     else
       METHOD_FLAGS+=(+actor_rollout_ref.actor.gxpo_trigger_abs_threshold="$GXPO_TRIGGER_ABS_THRESHOLD")
     fi ;;
esac
if [[ -n "${GXPO_TRIGGER_SUSTAIN_W:-}" ]]; then
  METHOD_FLAGS+=(+actor_rollout_ref.actor.gxpo_trigger_sustain_w="$GXPO_TRIGGER_SUSTAIN_W")
fi

case "$METHOD" in
  grpo)
    METHOD_FLAGS+=(+actor_rollout_ref.actor.use_gxpo=False)
    ;;
  sfpo)
    METHOD_FLAGS+=(
      +actor_rollout_ref.actor.use_sfpo=True
      +actor_rollout_ref.actor.sfpo_inner_steps="$K"
      +actor_rollout_ref.actor.sfpo_step_size="$REPOSITION_ALPHA"
      +actor_rollout_ref.actor.zscore_w=30
      +actor_rollout_ref.actor.zscore_threshold="$SFPO_ZSCORE_THRESHOLD"
      +actor_rollout_ref.actor.sfpo_warmup_steps="$SFPO_WARMUP_STEPS"
      +actor_rollout_ref.actor.sfpo_trigger_patience="$SFPO_TRIGGER_PATIENCE"
      +actor_rollout_ref.actor.sfpo_reset_entropy_after_warmup="$SFPO_RESET_ENTROPY_AFTER_WARMUP"
    )
    ;;
  gxpo)
    METHOD_FLAGS+=(
      +actor_rollout_ref.actor.use_gxpo=True
      # GXPO owns its trigger; disable the legacy trainer-side SFPO entropy gate.
      +actor_rollout_ref.actor.zscore_w=0
      +actor_rollout_ref.actor.gxpo_k="$K"
      +actor_rollout_ref.actor.gxpo_alpha="$REPOSITION_ALPHA"
      +actor_rollout_ref.actor.gxpo_delta=1e-8
      +actor_rollout_ref.actor.gxpo_tau="$GXPO_TAU"
      +actor_rollout_ref.actor.gxpo_zscore_w="$GXPO_ZSCORE_W"
      +actor_rollout_ref.actor.gxpo_trigger_signal="$GXPO_TRIGGER_SIGNAL"
      +actor_rollout_ref.actor.gxpo_trigger_patience="$GXPO_TRIGGER_PATIENCE"
      +actor_rollout_ref.actor.gxpo_fallback_mode="$GXPO_FALLBACK_MODE"
      +actor_rollout_ref.actor.gxpo_fallback_window="$GXPO_FALLBACK_WINDOW"
      +actor_rollout_ref.actor.gxpo_trigger_granularity=outer
      +actor_rollout_ref.actor.gxpo_warmup_steps="$GXPO_WARMUP_STEPS"
      +actor_rollout_ref.actor.gxpo_reset_entropy_after_warmup="$GXPO_RESET_ENTROPY_AFTER_WARMUP"
      +actor_rollout_ref.actor.gxpo_omega=0.1
      +actor_rollout_ref.actor.gxpo_shutoff_mode="$GXPO_SHUTOFF_MODE"
      +actor_rollout_ref.actor.gxpo_recompute_old_log_probs=False
      +actor_rollout_ref.actor.gxpo_diag_freq="$GXPO_DIAG_FREQ"
      +actor_rollout_ref.actor.gxpo_actor_duty_cycle="$GXPO_ACTOR_DUTY_CYCLE"
      ${GXPO_TRIGGER_MIN_OBS:+\+actor_rollout_ref.actor.gxpo_trigger_min_obs="$GXPO_TRIGGER_MIN_OBS"}
      ${GXPO_MAX_ACTIVE_STEPS:+\+actor_rollout_ref.actor.gxpo_max_active_steps="$GXPO_MAX_ACTIVE_STEPS"}
    )
    ;;
esac

cat <<EOF
[fair comparison config]
model=$MODEL_ID
model_alias=$MODEL_ALIAS
method=$METHOD
K=$K
reposition_alpha=$REPOSITION_ALPHA
train_seed=$TRAIN_SEED
train_batch_size=$TRAIN_BATCH_SIZE
rollout_n=$ROLLOUT_N
learning_rate=$LR
max_steps=$MAX_STEPS
save_freq=$SAVE_FREQ
sfpo_warmup_steps=$SFPO_WARMUP_STEPS
sfpo_zscore_threshold=$SFPO_ZSCORE_THRESHOLD
sfpo_trigger_patience=$SFPO_TRIGGER_PATIENCE
sfpo_reset_entropy_after_warmup=$SFPO_RESET_ENTROPY_AFTER_WARMUP
gxpo_warmup_steps=$GXPO_WARMUP_STEPS
gxpo_tau=$GXPO_TAU
gxpo_zscore_w=$GXPO_ZSCORE_W
gxpo_trigger_signal=$GXPO_TRIGGER_SIGNAL
gxpo_shutoff_mode=$GXPO_SHUTOFF_MODE
gxpo_trigger_robust=${GXPO_TRIGGER_ROBUST:-0}
gxpo_trigger_min_obs=${GXPO_TRIGGER_MIN_OBS:-0}
gxpo_max_active_steps=${GXPO_MAX_ACTIVE_STEPS:-0}
gxpo_trigger_patience=$GXPO_TRIGGER_PATIENCE
gxpo_fallback_mode=$GXPO_FALLBACK_MODE
gxpo_fallback_window=$GXPO_FALLBACK_WINDOW
gxpo_actor_duty_cycle=$GXPO_ACTOR_DUTY_CYCLE
gxpo_diag_freq=$GXPO_DIAG_FREQ
gxpo_trigger_granularity=outer
validation_interval=5
validation_decoding=greedy temperature=0 do_sample=false n=1
final_decoding=stochastic temperature=1.0 top_p=0.7 do_sample=true n=4 seeds=$FINAL_EVAL_SEEDS
gpu_count=$GPU_COUNT
train_files=$TRAIN_FILES
validation_files=$VAL_FILES
run_dir=$RUN_DIR
EOF

python -u -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  +algorithm.norm_adv_by_std_in_grpo=False \
  +algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length=1024 \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.seed="$TRAIN_SEED" \
  data.system_prompt="$SYSTEM_PROMPT" \
  actor_rollout_ref.model.path="$MODEL_ID" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing="$ENABLE_GRADIENT_CHECKPOINTING" \
  actor_rollout_ref.model.attn_implementation="$ATTN_IMPL" \
  +actor_rollout_ref.model.use_liger="$USE_LIGER" \
  actor_rollout_ref.actor.optim.lr="$LR" \
  +actor_rollout_ref.actor.optim.name=adamw \
  +actor_rollout_ref.actor.optim.fused="$OPTIM_FUSED" \
  actor_rollout_ref.actor.use_torch_compile="$USE_TORCH_COMPILE" \
  +actor_rollout_ref.actor.data_loader_seed="$TRAIN_SEED" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-16}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}" \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE:-1}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD" \
  ${ACTOR_MODEL_DTYPE:+\+actor_rollout_ref.actor.fsdp_config.model_dtype="$ACTOR_MODEL_DTYPE"} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE:-8}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_PARALLEL_SIZE:-1}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-98304}" \
  actor_rollout_ref.rollout.max_num_seqs="${VLLM_MAX_NUM_SEQS:-1024}" \
  actor_rollout_ref.rollout.enable_chunked_prefill="${VLLM_ENABLE_CHUNKED_PREFILL:-True}" \
  actor_rollout_ref.rollout.attention_backend="${VLLM_ATTENTION_BACKEND:-FLASHINFER}" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1.0}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P:-1.0}" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE:-8}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward_model.reward_manager=naive \
  algorithm.kl_ctrl.kl_coef=0.000 \
  trainer.critic_warmup=0 \
  trainer.logger="['console','wandb']" \
  trainer.project_name="$PROJECT" \
  trainer.experiment_name="$RUN_NAME" \
  trainer.default_local_dir="$RUN_DIR" \
  trainer.n_gpus_per_node="$GPU_COUNT" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  +trainer.keep_last_ckpts=1 \
  +trainer.keep_all_ckpts=False \
  trainer.resume_mode="$TRAINER_RESUME_MODE" \
  trainer.resume_from_path="$TRAINER_RESUME_FROM_PATH" \
  trainer.test_freq="$TRAINER_TEST_FREQ" \
  +trainer.validation_seeds='[0]' \
  +trainer.keep_last_validations=1 \
  +trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  +trainer.max_steps="$MAX_STEPS" \
  trainer.total_training_steps="$MAX_STEPS" \
  trainer.total_epochs=100 \
  "${METHOD_FLAGS[@]}" \
  | tee "$RUN_DIR/train.log"

TERMINAL_STEP="$MAX_STEPS"
if [[ ! -d "$RUN_DIR/global_step_$TERMINAL_STEP" ]]; then
  TERMINAL_STEP="$(find "$RUN_DIR" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | sed 's/global_step_//' | sort -n | tail -1)"
fi
if [[ -z "$TERMINAL_STEP" || ! -d "$RUN_DIR/global_step_$TERMINAL_STEP" ]]; then
  echo "Training finished without a terminal checkpoint under $RUN_DIR" >&2
  exit 3
fi

python -u tools/evaluate_gxpo_terminal.py \
  --run-dir "$RUN_DIR" \
  --base-model "$MODEL_ID" \
  --data-files "$MATH500" "$AIME24" "$AIME25" "$AMC23" "$MINERVA" "$OLYMPIAD" \
  --seeds $FINAL_EVAL_SEEDS \
  --step "$TERMINAL_STEP" \
  --n 4 --temperature 1.0 --top-p 0.7
