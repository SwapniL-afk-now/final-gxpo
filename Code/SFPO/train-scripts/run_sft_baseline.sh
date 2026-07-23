#!/usr/bin/env bash
# SFT baseline: plain 1-pass supervised finetuning of Qwen2.5-1.5B-Instruct on Hendrycks
# MATH Level 3-5 (human worked solutions as targets). Paired with run_sft_gxpo.sh, which is
# identical except for the GXPO-style 3-pass update -- together they answer whether the
# GXPO update transfers to a supervised (and by extension pretraining) objective.
#
# Hyperparameters mirror the RL rebuttal runs: lr 1e-6, batch 32, 200 steps, seed 42.
# Usage: GPU=0 ./run_sft_baseline.sh
set -euo pipefail

# /etc/environment ships RAY_ADDRESS="127.0.0.1" (no port), an invalid bootstrap address.
unset RAY_ADDRESS

GPU="${GPU:?set GPU=0|1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
LR=1e-6
MAX_STEPS=200
PROJECT=rebuttul

MODEL=/workspace/models/Qwen2.5-1.5B-Instruct
TRAIN=/workspace/jepa-grpo-cache/data/math_l35_sft/train.parquet
VAL=/workspace/jepa-grpo-cache/data/math_l35_sft/test.parquet

EXP="sft_baseline_mathl35_seed${TRAIN_SEED}"
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
