#!/usr/bin/env bash
set -euo pipefail
MODEL_ALIAS="qwen25-math-1p5b"
MODEL_ID="${MODEL_QWEN25_MATH_1P5B:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
METHOD="gxpo"
SAVE_FREQ="${SAVE_FREQ:-20}"
K="${K:-10}"
REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.3}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
