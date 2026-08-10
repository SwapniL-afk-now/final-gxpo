#!/usr/bin/env bash
# Non-math RLVR rebuttal run -- GXPO (answers Mvsj "add >=1 non-math RLVR experiment").
# KodCode-Light-RL-10K (easy+medium, function-based, pytest reward). Easiest RLVR set for a 1.5B.
# Code generation with unit-test (prime_code) rewards, Qwen2.5-1.5B-Instruct, full fine-tune.
#   TRAIN: TACO-verified + APPS(all/train) stdin-only, prompts <=1024 tok (10,113, data_source taco/apps)
#   EVAL : HumanEval+ (164) and MBPP+ (378) -- reviewer-named benchmarks;
#          scored by ported codegen_plus (data_source humanevalplus/mbppplus).
# TRAIN reward: taco/apps -> prime_code (unchanged). EVAL reward: humanevalplus/mbppplus ->
# codegen_plus (ported from the JEPA repo into verl/utils/reward_score, + dispatch branches).
# AdamW optimizer. Identical to run_rebuttal_code_grpo.sh except the GXPO actor flags below.
# Build the data first:  python train-scripts/prep_code_rlvr.py
# Usage: GPU=1 TRAIN_SEED=42 ./train-scripts/run_rebuttal_code_gxpo.sh
set -euo pipefail

# /etc/environment ships RAY_ADDRESS="127.0.0.1" (no port) -> invalid bootstrap addr; clear it.
export RAY_ADDRESS=local   # force an isolated Ray cluster per job -- unaddressed ray.init() auto-attaches to any existing local cluster (via /tmp/ray/session_latest), starving concurrent GPU0/GPU1 jobs of GPUs ("Total available GPUs 0")

GPU="${GPU:?set GPU=0|1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
VAL_SEEDS="[0,1,2]"
K=5
ALPHA=0.5
GXPO_TAU=2.0                  # GXPO trajectory-aware shutoff threshold
LR=1e-6
MAX_STEPS=200
N=8
PROJECT=rebuttul

MODEL=/workspace/models/Qwen2.5-Coder-1.5B-Instruct
TRAIN=/workspace/jepa-grpo-cache/data/kodcode_light/train.parquet
VAL_HE=/workspace/jepa-grpo-cache/eval_data/humanevalplus.parquet
VAL_MBPP=/workspace/jepa-grpo-cache/eval_data/mbppplus.parquet
EXP="gxpo_kodcode_seed${TRAIN_SEED}"
RUN_DIR="./runs/${EXP}"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_PROJECT="$PROJECT"
export WANDB_MODE=online
export WANDB_CONSOLE=off   # keep metrics online; stop streaming verbose console to wandb filestream (avoids "filestream at capacity" 409)
export WANDB_DIR="$RUN_DIR"
export REWARD_NUM_WORKERS=32      # pytest subprocesses are heavy; 128 OOMed host RAM (Ray kill). 32 is safe.
export REWARD_CONTINUOUS_MAX=3    # cap partial-credit per-test probes 10->3 (adv speedup)

python -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['$TRAIN']" \
    data.val_files="['$VAL_HE']" \
    data.train_batch_size=32 \
    data.val_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
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
    +trainer.val_before_train=False \
    +trainer.validation_seeds="$VAL_SEEDS" \
    +trainer.max_steps=$MAX_STEPS \
    trainer.total_epochs=100 \
    +actor_rollout_ref.actor.use_gxpo=True \
    +actor_rollout_ref.actor.gxpo_k="$K" \
    +actor_rollout_ref.actor.gxpo_alpha="$ALPHA" \
    +actor_rollout_ref.actor.gxpo_delta=1e-8 \
    +actor_rollout_ref.actor.gxpo_tau="$GXPO_TAU" \
    +actor_rollout_ref.actor.gxpo_omega=0.1 \
    +actor_rollout_ref.actor.gxpo_shutoff_mode=trajectory_aware \
    +actor_rollout_ref.actor.gxpo_recompute_old_log_probs=False \
    +actor_rollout_ref.actor.gxpo_diag_freq=10 \
    | tee "$RUN_DIR/train.log"
