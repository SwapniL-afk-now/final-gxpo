#!/usr/bin/env bash
set -euo pipefail

# Dedicated 1.5B GXPO run: global train batch 256, reposition alpha 0.5.
# Keep the original qwen25_math_1p5b_gxpo_k10.sh unchanged for reproducibility.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
export REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.5}"
export K="${K:-10}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export GPU_IDS="${GPU_IDS:-0}"
export SAVE_FREQ="${SAVE_FREQ:-20}"
export MAX_STEPS="${MAX_STEPS:-400}"
export WANDB_PROJECT="${WANDB_PROJECT:-gxpo-efficiency-final}"
export WANDB_GROUP="${WANDB_GROUP:-qwen25-math-1p5b-b256}"
export WANDB_TAGS="${WANDB_TAGS:-model:qwen25-math-1p5b,method:gxpo,k:10,alpha:0.5,batch:256,minibatch:64,experiment:custom}"
export WANDB_MODE="${WANDB_MODE:-online}"
export GXPO_RUN_NAME="${GXPO_RUN_NAME:-qwen25_math_1p5b_gxpo_k10_a05_b256_mb64_w50_tau3_v1_20260820}"

# A 256-prompt batch with rollout.n=8 can create up to 2048 sequences.
# max_num_batched_tokens remains bounded by the shared launcher default.
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2048}"

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen25_math_1p5b_gxpo_k10.sh"
