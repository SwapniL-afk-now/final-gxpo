#!/usr/bin/env bash
# GRPO baseline for the gxpo-qwen-1.5B study.
#   model: Qwen2.5-Math-1.5B-Instruct, full fine-tune (no LoRA), lr 1e-6
#   train: DeepScaleR-Preview (40,309 problems)
#   eval:  AMC23 + AIME24/25/26 @ temp 0.6 / top_p 0.95, n=8, 3 eval seeds
#   300 steps on 2 GPUs; best checkpoint selected by macro-mean val pass@1.
#
# NOTE: Qwen2.5-Math-1.5B-Instruct has max_position_embeddings=4096, so
# max_prompt_length + max_response_length must stay <= 4096 (they sum to exactly that).
# Raising the response budget on this model needs RoPE scaling.
#
# Resumable: rerunning the same command picks up the latest global_step_* automatically
# (trainer.resume_mode=auto). Logs append, so the resumed log keeps the earlier history.
#
# Usage: TRAIN_SEED=42 ./train-scripts/run_deepscaler_grpo.sh
set -euo pipefail

# /etc/environment ships RAY_ADDRESS="127.0.0.1" (no port) which is an invalid
# bootstrap address; clear it so ray.init() starts a fresh local cluster.
export RAY_ADDRESS=local

# training runs use the JEPA repo's venv (torch 2.9.1), not /venv/main
source /workspace/Joint-Embedding-Guided-Policy-Optimization/.venv/bin/activate

TRAIN_SEED="${TRAIN_SEED:-42}"
GPUS="${GPUS:-0,1}"
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
VAL_SEEDS="[0,1,2]"           # three evaluation seeds per validation pass
LR=1e-6
MAX_STEPS=300
N=8                           # responses per prompt, train and eval
PROJECT=gxpo-qwen-1.5B

MODEL=/workspace/models/Qwen2.5-Math-1.5B-Instruct
TRAIN=/workspace/jepa-grpo-cache/data/deepscaler_preview_train.parquet
E=/workspace/jepa-grpo-cache/eval_data
VAL="['$E/amc23.parquet','$E/aime24.parquet','$E/aime25.parquet','$E/aime26.parquet']"

EXP="grpo_deepscaler_qwenmath1.5b_seed${TRAIN_SEED}"
RUN_DIR="./runs/${EXP}"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="$GPUS"
export WANDB_PROJECT="$PROJECT"
export WANDB_MODE=online          # force cloud syncing (was defaulting to offline)
export WANDB_DIR="$RUN_DIR"

python -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['$TRAIN']" \
    data.val_files="$VAL" \
    data.train_batch_size=32 \
    data.val_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    +data.seed=$TRAIN_SEED \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$N \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.000 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$PROJECT" \
    trainer.experiment_name="$EXP" \
    trainer.default_local_dir="$RUN_DIR" \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=10 \
    trainer.resume_mode=auto \
    +trainer.keep_last_ckpts=2 \
    +trainer.val_before_train=True \
    +trainer.validation_seeds="$VAL_SEEDS" \
    +trainer.max_steps=$MAX_STEPS \
    trainer.total_epochs=100 \
    2>&1 | tee -a "$RUN_DIR/train.log"
