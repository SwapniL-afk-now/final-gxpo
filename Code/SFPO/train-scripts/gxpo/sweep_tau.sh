#!/usr/bin/env bash
# Tau sweep at k=5 for Figure 6 (Qwen2.5-1.5B by default).
set -euo pipefail
cd "$(dirname "$0")"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
export MODEL_TAG="${MODEL_TAG:-qwen1.5b}"
export METHOD=gxpo
export K=5

for TAU in 0.7 1.0 1.5 2.0; do
  TAU=$TAU ./run.sh
done
