#!/usr/bin/env bash
# SFT + GXPO-style update: identical to run_sft_baseline.sh except each step is the GXPO
# 3-pass extrapolated update (probe g0, probe g1, reposition, slow correction) applied to a
# plain cross-entropy objective -- no importance ratio, no advantages. K/alpha/tau match the
# GXPO RL arm so the two are directly comparable.
#
# Costs ~3x the baseline's wall time (3 backward passes per step).
# Usage: GPU=1 ./run_sft_gxpo.sh
set -euo pipefail

# /etc/environment ships RAY_ADDRESS="127.0.0.1" (no port), an invalid bootstrap address.
unset RAY_ADDRESS

GPU="${GPU:?set GPU=0|1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
LR=1e-6
MAX_STEPS=200
PROJECT=rebuttul
K=5
ALPHA=0.5
GXPO_TAU=2.0      # same trajectory-aware shutoff threshold as the GXPO RL arm

MODEL=/workspace/models/Qwen2.5-1.5B-Instruct
TRAIN=/workspace/jepa-grpo-cache/data/math_l35_sft/train.parquet
VAL=/workspace/jepa-grpo-cache/data/math_l35_sft/test.parquet

EXP="sft_gxpo_k${K}_a${ALPHA}_tau${GXPO_TAU}_mathl35_seed${TRAIN_SEED}"
RUN_DIR="./runs/${EXP}"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_PROJECT="$PROJECT"
export WANDB_MODE=online
export WANDB_DIR="$RUN_DIR"

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=2048 \
    data.truncation=right \
    model.partial_pretrain="$MODEL" \
    model.enable_gradient_checkpointing=True \
    optim.lr=$LR \
    optim.betas="[0.9,0.999]" \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.0 \
    optim.clip_grad=1.0 \
    +optim.use_gxpo=True \
    +optim.gxpo_k=$K \
    +optim.gxpo_alpha=$ALPHA \
    +optim.gxpo_delta=1e-8 \
    +optim.gxpo_tau=$GXPO_TAU \
    +optim.gxpo_omega=0.1 \
    +optim.gxpo_shutoff_mode=trajectory_aware \
    use_remove_padding=true \
    trainer.project_name="$PROJECT" \
    trainer.experiment_name="$EXP" \
    trainer.default_local_dir="$RUN_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=2 \
    trainer.total_training_steps=$MAX_STEPS \
    +trainer.test_freq=10 \
    +trainer.save_freq=100 \
    +trainer.val_max_batches=50 \
    trainer.seed=$TRAIN_SEED \
    2>&1 | tee "$RUN_DIR/train.log"
