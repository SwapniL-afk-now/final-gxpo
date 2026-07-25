#!/usr/bin/env bash
# Group-2 ablation: no fallback (shutoff gate disabled; extrapolation never reverts to GRPO)
# Reviewer question answered: mbJD Q2 / Mvsj -- why permanently disable extrapolation vs. no fallback?
# Self-contained -- a reviewer can run this directly:  GPU=1 ./train-scripts/run_ablation_no_fallback.sh
# Identical to the full GXPO rebuttal run (run_rebuttal_seeded.sh METHOD=gxpo) except the
# ablation-specific flags below. Qwen2.5-1.5B, MATH L3-5 train / AMC23 eval, seed 42, 200 steps.
set -euo pipefail
export RAY_ADDRESS=local   # force an isolated Ray cluster per job -- unaddressed ray.init() auto-attaches to any existing local cluster (via /tmp/ray/session_latest), starving concurrent GPU0/GPU1 jobs of GPUs ("Total available GPUs 0")

GPU="${GPU:?set GPU=0|1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
K=5
ALPHA=0.5
GXPO_TAU=2.0
VAL_SEEDS="[0,1,2]"
LR=1e-6
MAX_STEPS=200
N=8
PROJECT=rebuttul

MODEL=/workspace/models/Qwen2.5-1.5B-Instruct
TRAIN=/workspace/jepa-grpo-cache/data/math_l35/train.parquet
VAL=/workspace/jepa-grpo-cache/eval_data/amc23.parquet
EXP="gxpo_k${K}_a${ALPHA}_tau${GXPO_TAU}_mathl35_amc23_seed${TRAIN_SEED}_nofallback"
RUN_DIR="./runs/${EXP}"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_PROJECT="$PROJECT"
export WANDB_MODE=online
export WANDB_CONSOLE=off   # keep metrics online; stop streaming verbose console to wandb filestream (avoids "filestream at capacity" 409)
export WANDB_DIR="$RUN_DIR"

python -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['$TRAIN']" \
    data.val_files="['$VAL']" \
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
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
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
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.000 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$PROJECT" \
    trainer.experiment_name="$EXP" \
    trainer.default_local_dir="$RUN_DIR" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    +trainer.val_before_train=True \
    +trainer.validation_seeds="$VAL_SEEDS" \
    +trainer.max_steps=$MAX_STEPS \
    trainer.total_epochs=100 \
    +actor_rollout_ref.actor.use_gxpo=True \
    +actor_rollout_ref.actor.gxpo_k=$K \
    +actor_rollout_ref.actor.gxpo_alpha=$ALPHA \
    +actor_rollout_ref.actor.gxpo_delta=1e-8 \
    +actor_rollout_ref.actor.gxpo_tau=$GXPO_TAU \
    +actor_rollout_ref.actor.gxpo_omega=0.1 \
    +actor_rollout_ref.actor.gxpo_recompute_old_log_probs=True \
    +actor_rollout_ref.actor.gxpo_diag_freq=10 \
    +actor_rollout_ref.actor.gxpo_shutoff_mode=never \
    | tee "$RUN_DIR/train.log"
