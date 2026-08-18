#!/usr/bin/env bash
# Storage-safe final main-table run: SFPO vs GXPO, Qwen2.5-1.5B-Instruct, full fine-tune, DAPO-17k train / AMC23 eval.
# Dedicated final main-table run; set GPU=0 or GPU=1.
set -euo pipefail
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

# /etc/environment ships RAY_ADDRESS="127.0.0.1" (no port) which is an invalid
# bootstrap address; clear it so ray.init() starts a fresh local cluster.
export RAY_ADDRESS=local   # force an isolated Ray cluster per job -- unaddressed ray.init() auto-attaches to any existing local cluster (via /tmp/ray/session_latest), starving concurrent GPU0/GPU1 jobs of GPUs ("Total available GPUs 0")

METHOD="sfpo"
K=10
GPU="${GPU:-0}"
ALPHA="${ALPHA:-0.3}"
SFPO_TAU=0.5      # SFPO entropy z-score shutoff threshold
GXPO_TAU=2.0      # GXPO trajectory-aware shutoff threshold
TRAIN_SEED=3407                 # matched FEPO training seed
VAL_SEEDS="[3407]"           # same single greedy evaluation seed as GSPO
LR=1e-6
MAX_STEPS=500
N=8                           # responses per prompt, train and eval
PROJECT=tafr-repro-math15b

MODEL=/workspace/models/Qwen2.5-Math-1.5B-Instruct
TRAIN=/workspace/jepa-grpo-cache/data/dsr_math345/train.parquet   # Hendrycks MATH, Level 3-5
VAL_FILES="[/workspace/jepa-grpo-cache/eval_data/math500.parquet,/workspace/jepa-grpo-cache/eval_data/amc23.parquet,/workspace/jepa-grpo-cache/eval_data/aime24.parquet,/workspace/jepa-grpo-cache/eval_data/aime25.parquet]"
if [ "$METHOD" = "gxpo" ]; then TAU="$GXPO_TAU"; else TAU="$SFPO_TAU"; fi
EXP="efficiency_sfpo_k${K}_a${ALPHA}_tau${TAU}_mathl35_amc23_seed${TRAIN_SEED}_clamped_v1_k${K}"
RUN_DIR="./runs/${EXP}"
mkdir -p "$RUN_DIR"

METHOD_FLAGS=()
case "$METHOD" in
  sfpo)
    METHOD_FLAGS+=(
      +actor_rollout_ref.actor.use_sfpo=True
      +actor_rollout_ref.actor.sfpo_inner_steps="$K"
      +actor_rollout_ref.actor.sfpo_step_size="$ALPHA"
      +actor_rollout_ref.actor.zscore_w=30
      +actor_rollout_ref.actor.zscore_threshold="$SFPO_TAU"
    ) ;;
  gxpo)
    METHOD_FLAGS+=(
      +actor_rollout_ref.actor.use_gxpo=True
      +actor_rollout_ref.actor.gxpo_k="$K"
      +actor_rollout_ref.actor.gxpo_alpha="$ALPHA"
      +actor_rollout_ref.actor.gxpo_delta=1e-8
      +actor_rollout_ref.actor.gxpo_tau="$GXPO_TAU"
      +actor_rollout_ref.actor.gxpo_omega=0.1
      +actor_rollout_ref.actor.gxpo_shutoff_mode=trajectory_aware
      +actor_rollout_ref.actor.gxpo_recompute_old_log_probs=False
      +actor_rollout_ref.actor.gxpo_diag_freq=10
    ) ;;
  *) echo "unknown METHOD=$METHOD"; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_PROJECT="$PROJECT"
export WANDB_MODE=online
export MPLBACKEND=Agg          # force cloud syncing (was defaulting to offline)
export WANDB_DIR="$RUN_DIR"

python -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.norm_adv_by_std_in_grpo=False \
    +algorithm.use_kl_in_reward=False \
    data.train_files="['$TRAIN']" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=64 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    +data.seed=$TRAIN_SEED \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.attn_implementation="$ATTN_IMPL" \
    actor_rollout_ref.actor.optim.lr=$LR \
    +actor_rollout_ref.actor.data_loader_seed=$TRAIN_SEED \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.000 \
    +reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=false \
    +reward.reward_kwargs.overlong_buffer_cfg.len=512 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.max_resp_len=3072 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$PROJECT" \
    trainer.experiment_name="$EXP" \
    trainer.default_local_dir="$RUN_DIR" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    +trainer.keep_last_ckpts=1 \
    +trainer.keep_all_ckpts=False \
    +trainer.remove_previous_ckpt_in_save=False \
    trainer.test_freq=10 \
    +trainer.val_before_train=False \
    +trainer.validation_seeds="$VAL_SEEDS" \
    trainer.total_training_steps=$MAX_STEPS \
    trainer.total_epochs=100 \
    "${METHOD_FLAGS[@]}" \
    | tee "$RUN_DIR/train.log"
