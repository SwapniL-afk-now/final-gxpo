#!/usr/bin/env bash
set -euo pipefail
MODEL_ALIAS="qwen25-math-1p5b"
MODEL_ID="${MODEL_QWEN25_MATH_1P5B:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
METHOD="sfpo"
SAVE_FREQ="${SAVE_FREQ:-20}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
