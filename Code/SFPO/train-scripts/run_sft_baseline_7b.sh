#!/usr/bin/env bash
# SFT baseline (7B): plain 1-pass supervised finetuning of Qwen2.5-7B-Instruct on Hendrycks
# MATH Level 3-5 (human worked solutions as targets). Paired with run_sft_gxpo_7b.sh, which is
# identical except for the GXPO-style 3-pass update.
#
# 7B needs FSDP across 2 GPUs (AdamW states alone are ~3x params in fp32); 1-GPU like the
# 1.5B pair cannot fit it. lr 1e-5, batch 32, 500 steps (3 epochs of data, capped at 500),
# validate every 10 steps, seed 42.
# Usage: ./run_sft_baseline_7b.sh
#   GPU_IDS=0,1 NPROC=2 MODEL=/path/to/Qwen2.5-7B-Instruct ./run_sft_baseline_7b.sh
set -euo pipefail

# /etc/environment ships RAY_ADDRESS="127.0.0.1" (no port), an invalid bootstrap address.
export RAY_ADDRESS=local   # force an isolated Ray cluster per job -- unaddressed ray.init() auto-attaches to any existing local cluster (via /tmp/ray/session_latest), starving concurrent jobs of GPUs ("Total available GPUs 0")

GPU_IDS="${GPU_IDS:-0,1}"
NPROC="${NPROC:-2}"
TRAIN_SEED="${TRAIN_SEED:-42}"
LR="${LR:-1e-5}"
PROJECT="${WANDB_PROJECT:-rebuttul}"

MODEL="${MODEL:-/workspace/models/Qwen2.5-7B-Instruct}"
TRAIN="${TRAIN:-/workspace/jepa-grpo-cache/data/math_l35_sft/train.parquet}"
VAL="${VAL:-/workspace/jepa-grpo-cache/data/math_l35_sft/test.parquet}"

EXP="sft_baseline_7b_mathl35_seed${TRAIN_SEED}"
RUN_DIR="./runs/${EXP}"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export WANDB_PROJECT="$PROJECT"
export WANDB_MODE=online
export WANDB_DIR="$RUN_DIR"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=2 \
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
    trainer.total_epochs=3 \
    trainer.total_training_steps=500 \
    +trainer.test_freq=10 \
    +trainer.save_freq=100 \
    +trainer.val_max_batches=50 \
    trainer.seed=$TRAIN_SEED \
    2>&1 | tee "$RUN_DIR/train.log"
