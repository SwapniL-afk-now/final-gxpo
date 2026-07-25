#!/usr/bin/env bash
# MATH500 eval for the SFPO / GXPO 3-seed runs archived in the HF dataset `ismamNur/rebuttal-runs`.
# Those are raw verl NO_SHARD checkpoints, so each run is fetched, converted to HF format,
# evaluated, then deleted before the next one (only ~7G of scratch at a time).
#
# Usage: GPU=1 ./train-scripts/eval_math500_rebuttal.sh [run_name ...]
set -euo pipefail

export RAY_ADDRESS=local
GPU="${GPU:?set GPU=0|1}"

REPO=ismamNur/rebuttal-runs
SCRATCH="${SCRATCH:-/workspace/models/eval}"

RUNS=("$@")
if [ "${#RUNS[@]}" -eq 0 ]; then
    RUNS=(sfpo_k5_a0.5_tau0.5_mathl35_amc23_seed42
          sfpo_k5_a0.5_tau0.5_mathl35_amc23_seed123
          sfpo_k5_a0.5_tau0.5_mathl35_amc23_seed777
          gxpo_k5_a0.5_tau2.0_mathl35_amc23_seed42
          gxpo_k5_a0.5_tau2.0_mathl35_amc23_seed123
          gxpo_k5_a0.5_tau2.0_mathl35_amc23_seed777)
fi

cd "$(dirname "$0")/.."
source /workspace/Joint-Embedding-Guided-Policy-Optimization/.venv/bin/activate
set -a; source /workspace/gradient-extrapolation-based-policy-optimization/.env; set +a

for RUN in "${RUNS[@]}"; do
    echo "=== fetching $RUN ==="
    ACTOR=$(python - "$REPO" "$RUN" "$SCRATCH" <<'EOF'
import sys, os, re
from huggingface_hub import HfApi, snapshot_download
repo, run, scratch = sys.argv[1:4]
api = HfApi()
files = [f for f in api.list_repo_files(repo, repo_type='dataset') if f.startswith(run + '/')]
steps = {int(m.group(1)) for f in files if (m := re.match(rf'{re.escape(run)}/global_step_(\d+)/', f))}
assert steps, f'no checkpoint in {run}'
step = max(steps)                       # each run archives exactly one (its best) checkpoint
local = os.path.join(scratch, run)
snapshot_download(repo_id=repo, repo_type='dataset', local_dir=local,
                  allow_patterns=[f'{run}/global_step_{step}/actor/huggingface/*',
                                  f'{run}/global_step_{step}/actor/model_world_size_*_rank_0.pt'])
print(os.path.join(local, run, f'global_step_{step}', 'actor'))
EOF
)
    ACTOR=$(echo "$ACTOR" | tail -1)
    echo "=== converting $ACTOR ==="
    python train-scripts/push_ckpt.py "$ACTOR"          # no repo id -> convert in place, no upload

    # eval_math500.sh names its log after the model dir's basename -> use the run name
    HFDIR="$SCRATCH/hf/$RUN"
    rm -rf "$HFDIR"; mkdir -p "$SCRATCH/hf"; mv "$ACTOR/huggingface" "$HFDIR"

    GPU="$GPU" ./train-scripts/eval_math500.sh "$HFDIR"
    rm -rf "$SCRATCH/$RUN" "$HFDIR"
done

echo "all done; per-model logs in ./eval-results/math500/"
