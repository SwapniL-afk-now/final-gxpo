#!/usr/bin/env bash
# Score an SFT checkpoint (or the base model) on AMC23: avg@8 / pass@8 over 3 seeds.
# Usage: GPU=0 ./eval_sft_amc23.sh /path/to/runs/<exp>/global_step_200
set -euo pipefail

export RAY_ADDRESS=local   # force an isolated Ray cluster per job -- unaddressed ray.init() auto-attaches to any existing local cluster (via /tmp/ray/session_latest), starving concurrent GPU0/GPU1 jobs of GPUs ("Total available GPUs 0")

GPU="${GPU:?set GPU=0|1}"
MODEL="${1:?usage: eval_sft_amc23.sh <hf_checkpoint_dir>}"

export CUDA_VISIBLE_DEVICES="$GPU"

cd "$(dirname "$0")/.."
PYTHONPATH="$PWD" python -u train-scripts/eval_amc23.py --model "$MODEL" --n 8 --seeds 0,1,2
