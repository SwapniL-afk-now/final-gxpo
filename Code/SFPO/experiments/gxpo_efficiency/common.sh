#!/usr/bin/env bash
# Shared launcher for the final 3-model x 3-method GXPO efficiency matrix.
# All fairness-critical settings live here; the nine entrypoints only select
# MODEL_ALIAS, MODEL_ID, and METHOD.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.5}"
K="${K:-10}"
TRAIN_SEED="${TRAIN_SEED:-3407}"
FINAL_EVAL_SEEDS="${FINAL_EVAL_SEEDS:-0 1 2 3}"
MAX_STEPS="${MAX_STEPS:-400}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
ROLLOUT_N="${ROLLOUT_N:-8}"
LR="${LR:-1e-6}"
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

export GXPO_EFFICIENCY_RUN=1
export GXPO_RUN_NAME="$RUN_NAME"
export GXPO_MODEL_ALIAS="$MODEL_ALIAS"
export TRAIN_SEED
export FINAL_EVAL_SEEDS
export WANDB_PROJECT="$PROJECT"
export WANDB_GROUP="$MODEL_ALIAS"
export WANDB_TAGS="model:$MODEL_ALIAS,method:$METHOD${METHOD:+,k:$K}${METHOD:+,alpha:$REPOSITION_ALPHA},experiment:final-efficiency"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="$RUN_DIR"
export RAY_ADDRESS=local
export MPLBACKEND=Agg

if [[ -n "${GPU_IDS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi

TRAIN_FILES="['$DAPO_TRAIN','$LIGHTEVAL_TRAIN']"
VAL_FILES="['$MATH500','$AIME24','$AIME25','$AMC23','$MINERVA','$OLYMPIAD']"

METHOD_FLAGS=()
case "$METHOD" in
  grpo)
    ;;
  sfpo)
    METHOD_FLAGS+=(
      +actor_rollout_ref.actor.use_sfpo=True
      +actor_rollout_ref.actor.sfpo_inner_steps="$K"
      +actor_rollout_ref.actor.sfpo_step_size="$REPOSITION_ALPHA"
      +actor_rollout_ref.actor.zscore_w=30
      +actor_rollout_ref.actor.zscore_threshold="${SFPO_ZSCORE_THRESHOLD:-0.5}"
    )
    ;;
  gxpo)
    METHOD_FLAGS+=(
      +actor_rollout_ref.actor.use_gxpo=True
      +actor_rollout_ref.actor.gxpo_k="$K"
      +actor_rollout_ref.actor.gxpo_alpha="$REPOSITION_ALPHA"
      +actor_rollout_ref.actor.gxpo_delta=1e-8
      +actor_rollout_ref.actor.gxpo_tau="${GXPO_TAU:-2.0}"
      +actor_rollout_ref.actor.gxpo_omega=0.1
      +actor_rollout_ref.actor.gxpo_shutoff_mode=trajectory_aware
      +actor_rollout_ref.actor.gxpo_recompute_old_log_probs=False
      +actor_rollout_ref.actor.gxpo_diag_freq=10
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
  data.val_batch_size=128 \
  data.max_prompt_length=1024 \
  data.max_response_length=3072 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.seed="$TRAIN_SEED" \
  actor_rollout_ref.model.path="$MODEL_ID" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.attn_implementation="${ATTN_IMPL:-flash_attention_2}" \
  actor_rollout_ref.actor.optim.lr="$LR" \
  +actor_rollout_ref.actor.optim.name=adamw \
  +actor_rollout_ref.actor.data_loader_seed="$TRAIN_SEED" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-16}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}" \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE:-8}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_PARALLEL_SIZE:-1}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}" \
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
  trainer.save_freq=5 \
  +trainer.keep_last_ckpts=2 \
  +trainer.keep_all_ckpts=False \
  trainer.test_freq=5 \
  +trainer.val_before_train=False \
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
