#!/usr/bin/env bash
set -euo pipefail
MODEL_ALIAS="llama32-3b"
MODEL_ID="${MODEL_LLAMA32_3B:-/workspace/models/Llama-3.2-3B-Instruct}"
METHOD="gxpo"
SAVE_FREQ="${SAVE_FREQ:-20}"
K="${K:-10}"
REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.3}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
