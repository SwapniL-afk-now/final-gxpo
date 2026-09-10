#!/usr/bin/env bash
set -euo pipefail
MODEL_ALIAS="qwen25-math-1p5b"
MODEL_ID="${MODEL_QWEN25_MATH_1P5B:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
METHOD="${METHOD:-gxpo}"
SAVE_FREQ="${SAVE_FREQ:-20}"
K="${K:-10}"
REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.3}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
# A/B BASELINE ARM -- pinned to the legacy raw-gradient retention estimator
# (r = g1/g0) so this entrypoint keeps reproducing exactly what the existing
# qwen25-math-1p5b_gxpo_k10_seed3407 run is. Without this pin it would inherit
# common.sh's `auto`, which is now optimizer-aware and selects the AdamW
# direction ratio r = d1/d0 -- a different algorithm under the same name.
# The optimizer-aware arm is qwen25_math_1p5b_gxpo_adamw_transactional_dir_k10.sh.
GXPO_RETENTION_SPACE="${GXPO_RETENTION_SPACE:-grad}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
