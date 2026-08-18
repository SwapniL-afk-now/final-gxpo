#!/usr/bin/env bash
set -euo pipefail
MODEL_ALIAS="qwen25-math-7b"
MODEL_ID="${MODEL_QWEN25_MATH_7B:-Qwen/Qwen2.5-Math-7B-Instruct}"
METHOD="sfpo"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
