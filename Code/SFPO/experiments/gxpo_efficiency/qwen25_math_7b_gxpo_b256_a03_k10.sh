#!/usr/bin/env bash
set -euo pipefail

# Qwen2.5-Math-7B-Instruct | GXPO | batch 256 | PPO minibatch 64 | K=5 | alpha=0.3
# This entrypoint delegates to the dedicated Qwen 7B launcher so the model
# identifier is not accidentally replaced by a Llama/1.5B default.

MODEL_ALIAS="qwen2.5-math-7b-instruct"
MODEL_ID="${MODEL_QWEN25_MATH_7B:-Qwen/Qwen2.5-Math-7B-Instruct}"
METHOD="gxpo"

export MODEL_QWEN25_MATH_7B="$MODEL_ID"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export K="5"
export REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.3}"
# Gate-v2: use gradient-direction disagreement, not the trainer entropy gate.
# Disagreement = 1 - |cos(g0, g_slow)|; score it with a 30-step z-score window.
export GXPO_TRIGGER_SIGNAL="${GXPO_TRIGGER_SIGNAL:-grad}"
export GXPO_SHUTOFF_MODE="${GXPO_SHUTOFF_MODE:-cosine}"
export GXPO_TRIGGER_ABS_THRESHOLD="${GXPO_TRIGGER_ABS_THRESHOLD:-0}"
export GXPO_TRIGGER_ROBUST="${GXPO_TRIGGER_ROBUST:-0}"
export GXPO_TAU="${GXPO_TAU:-2.0}"
export GXPO_ZSCORE_W="${GXPO_ZSCORE_W:-30}"
export GXPO_TRIGGER_PATIENCE="${GXPO_TRIGGER_PATIENCE:-2}"
export GXPO_TRIGGER_MIN_OBS="${GXPO_TRIGGER_MIN_OBS:-0}"
export GXPO_MAX_ACTIVE_STEPS="${GXPO_MAX_ACTIVE_STEPS:-150}"
export ROLLOUT_TEMPERATURE="0.7"
export ROLLOUT_TOP_P="0.8"
export MAX_RESPONSE_LENGTH="3072"
export VLLM_GPU_MEMORY_UTILIZATION="0.48"
export WANDB_PROJECT="${WANDB_PROJECT:-gxpo-efficiency-final}"
export WANDB_GROUP="${WANDB_GROUP:-qwen2.5-math-7b-instruct-gxpo}"
export GXPO_RUN_NAME="${GXPO_RUN_NAME:-qwen2.5_math_7b_instruct_gxpo_b256_mb64_a0.3_k5_gradcos_tau2_p2_m150_tp08_v048}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
