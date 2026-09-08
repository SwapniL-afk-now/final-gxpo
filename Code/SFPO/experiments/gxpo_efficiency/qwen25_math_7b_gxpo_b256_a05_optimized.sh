#!/usr/bin/env bash
set -euo pipefail

# Qwen2.5-Math-7B GXPO run using the optimized 1.5B launcher settings, with
# memory-safe limits required by the larger model. Algorithmic settings remain
# unchanged.
export MODEL_QWEN25_MATH_7B="${MODEL_QWEN25_MATH_7B:-Qwen/Qwen2.5-Math-7B-Instruct}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
export REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.8}"
export K="${K:-10}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export GPU_IDS="${GPU_IDS:-0}"
export SAVE_FREQ="${SAVE_FREQ:-20}"
export MAX_STEPS="${MAX_STEPS:-200}"
export GXPO_WARMUP_STEPS="${GXPO_WARMUP_STEPS:-50}"
export GXPO_TAU="${GXPO_TAU:-1.5}"
export GXPO_ZSCORE_W="${GXPO_ZSCORE_W:-30}"
export GXPO_TRIGGER_PATIENCE="${GXPO_TRIGGER_PATIENCE:-3}"
export USE_LIGER="${USE_LIGER:-True}"
export USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-True}"
export ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-True}"
export OPTIM_FUSED="${OPTIM_FUSED:-False}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-4}"
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
export ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-bf16}"
# 0.75 caused the 7B actor/vLLM co-resident worker to die while building the
# vLLM engine.  0.60 leaves a practical headroom margin on the H200 while
# retaining substantially more rollout capacity than the original 0.50.
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.60}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1024}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
export WANDB_PROJECT="${WANDB_PROJECT:-gxpo-efficiency-final}"
export WANDB_GROUP="${WANDB_GROUP:-qwen25-math-7b-b256}"
export WANDB_MODE="${WANDB_MODE:-online}"

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen25_math_7b_gxpo_k10.sh"
