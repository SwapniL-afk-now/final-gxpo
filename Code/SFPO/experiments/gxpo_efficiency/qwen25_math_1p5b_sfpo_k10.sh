#!/usr/bin/env bash
set -euo pipefail

# Load W&B credentials from the checkout .env, falling back to the workspace .env.
SFPO_ENV_FILE="${GXPO_ENV_FILE:-/workspace/final-gxpo/.env}"
if [[ ! -f "$SFPO_ENV_FILE" && -f "/workspace/.env" ]]; then
  SFPO_ENV_FILE="/workspace/.env"
fi
if [[ -f "$SFPO_ENV_FILE" ]]; then
  set -a
  source "$SFPO_ENV_FILE"
  set +a
fi

# Use the verified local assets and all four GPUs for FSDP4.
export GXPO_DATA_ROOT="${GXPO_DATA_ROOT:-/workspace/data}"
export GXPO_RESULTS_ROOT="${GXPO_RESULTS_ROOT:-/workspace/final-gxpo/results/gxpo_efficiency}"
export MODEL_LLAMA32_3B="${MODEL_LLAMA32_3B:-/workspace/models/Llama-3.2-3B-Instruct}"
export GPU_IDS=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPU_COUNT=4
export FSDP_SIZE=4
export ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-float32}"
MODEL_ALIAS="llama32-3b-instruct"
MODEL_ID="${MODEL_LLAMA32_3B:-/workspace/models/Llama-3.2-3B-Instruct}"
METHOD="sfpo"
SAVE_FREQ="${SAVE_FREQ:-20}"
K="${K:-3}"
REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
SFPO_ZSCORE_THRESHOLD="${SFPO_ZSCORE_THRESHOLD:-1.0}"
SFPO_WARMUP_STEPS="${SFPO_WARMUP_STEPS:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-gxpo-efficiency-final}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_TAGS="${WANDB_TAGS:-model:llama32-3b-instruct,method:sfpo,k:${K},alpha:${REPOSITION_ALPHA},batch:256,minibatch:64,gpus:4,fsdp:4,experiment:sfpo-efficiency}"
WANDB_GROUP="${WANDB_GROUP:-llama32-3b-instruct-sfpo-b256-fsdp4}"
GXPO_RUN_NAME="${GXPO_RUN_NAME:-llama32_3b_instruct_sfpo_k${K}_a${REPOSITION_ALPHA}_b256_mb64_g4_fsdp4_fp32}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
