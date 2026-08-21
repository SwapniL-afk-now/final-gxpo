#!/usr/bin/env bash
set -euo pipefail

# Qwen3 4B GRPO efficiency run.  The shared launcher keeps the comparison
# settings aligned with the Llama 3.2 3B and Qwen2.5 runs.
MODEL_ALIAS="qwen3-4b"
MODEL_ID="${MODEL_QWEN3_4B:-Qwen/Qwen3-4B-Base}"
METHOD="grpo"
SAVE_FREQ="${SAVE_FREQ:-20}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
