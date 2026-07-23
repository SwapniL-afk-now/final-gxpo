#!/usr/bin/env bash
# Alpha x k sweep for Tables 5-7 and Figures 2/4/5/9 (Qwen2.5-1.5B by default).
set -euo pipefail
cd "$(dirname "$0")"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
export MODEL_TAG="${MODEL_TAG:-qwen1.5b}"
export METHOD=gxpo

for ALPHA in 0.1 0.5 1.0; do
  for K in 3 5 10; do
    ALPHA=$ALPHA K=$K ./run.sh
  done
done
