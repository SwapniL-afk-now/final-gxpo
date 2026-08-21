#!/usr/bin/env bash
set -euo pipefail

# Llama 3.2 3B SFPO efficiency run: K=3, alpha=0.8, batch=64, minibatch=16.
# Keep the existing k10 launcher unchanged and override only this run's settings.
export K="${K:-3}"
export REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.8}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export SFPO_ZSCORE_THRESHOLD="${SFPO_ZSCORE_THRESHOLD:-1.0}"
export SFPO_WARMUP_STEPS="${SFPO_WARMUP_STEPS:-0}"
export GXPO_RUN_NAME="${GXPO_RUN_NAME:-llama32_3b_sfpo_k3_a08_b64_mb16_seed3407_v3_20260821}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/llama32_3b_sfpo_k10.sh"
