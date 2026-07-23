#!/usr/bin/env bash
# Score an SFT checkpoint (or the base model) on AMC23: avg@8 / pass@8 over 3 seeds.
# Usage: GPU=0 ./eval_sft_amc23.sh /path/to/runs/<exp>/global_step_200
set -euo pipefail

unset RAY_ADDRESS

GPU="${GPU:?set GPU=0|1}"
MODEL="${1:?usage: eval_sft_amc23.sh <hf_checkpoint_dir>}"

export CUDA_VISIBLE_DEVICES="$GPU"

cd "$(dirname "$0")/.."
PYTHONPATH="$PWD" python -u train-scripts/eval_amc23.py --model "$MODEL" --n 8 --seeds 0,1,2
