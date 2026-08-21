#!/usr/bin/env bash
set -euo pipefail

# Qwen3 4B Instruct GXPO timing/efficiency run.
MODEL_ALIAS="qwen3-4b"
MODEL_ID="${MODEL_QWEN3_4B:-Qwen/Qwen3-4B-Instruct-2507}"
METHOD="gxpo"
SAVE_FREQ="${SAVE_FREQ:-20}"
K="${K:-10}"
REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.5}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
GXPO_TAU="${GXPO_TAU:-1.5}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-Provide a concise solution with only the essential reasoning steps. Avoid long explanations and finish with the final answer.}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
