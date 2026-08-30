#!/usr/bin/env bash
set -euo pipefail

set -a
source /workspace/.env
set +a
source /workspace/gradient-extrapolation-based-policy-optimization/.venv-h200/bin/activate
test -n "${WANDB_API_KEY:-}"

export WANDB_MODE=online
export WANDB_RESUME=never
export HYDRA_FULL_ERROR=1
export GXPO_ENFORCE_POWER_LIMIT=False
export GXPO_ACTOR_DUTY_CYCLE=0
export GXPO_RUN_NAME=llama32_3b_instruct_gxpo_gate_v6_k10_a0.3_b256_mb64_g4_fsdp4_fp32_liger_maxpower

cd /workspace/final-gxpo/Code/SFPO
exec bash experiments/gxpo_efficiency/qwen25_math_1p5b_gxpo_b256_mb64_gate_v6.sh \
  2>&1 | tee -a /workspace/final-gxpo/results/gxpo-llama32-3b-gate-v6-tmux.log
