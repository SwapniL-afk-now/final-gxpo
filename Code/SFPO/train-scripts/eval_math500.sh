#!/usr/bin/env bash
# MATH500 evaluation pipeline: 3 seeds per model, reports pass@1 / avg@8 / pass@8 as mean +- std.
# Runs on the free GPU at 0.9 vLLM memory utilization, temperature 0.6, top_p 0.95.
#
# Usage:
#   GPU=1 ./train-scripts/eval_math500.sh <model_dir_or_hub_id> [more models...]
# Results land in ./eval-results/math500/<name>.log plus a summary table at the end.
set -euo pipefail

export RAY_ADDRESS=local   # isolated Ray per job -- see run_rebuttal_*.sh

GPU="${GPU:?set GPU=0|1}"
[ "$#" -ge 1 ] || { echo "usage: GPU=1 $0 <model> [model...]" >&2; exit 1; }

DATA="${DATA:-/workspace/jepa-grpo-cache/eval_data/math500.parquet}"
N="${N:-8}"
SEEDS="${SEEDS:-0,1,2}"
TEMP="${TEMP:-0.6}"
TOP_P="${TOP_P:-0.95}"
GPU_UTIL="${GPU_UTIL:-0.9}"

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$(dirname "$0")/.."
source /workspace/Joint-Embedding-Guided-Policy-Optimization/.venv/bin/activate

OUT=./eval-results/$(basename "$DATA" .parquet)   # results grouped per benchmark
mkdir -p "$OUT"

for MODEL in "$@"; do
    NAME=$(basename "${MODEL%/}")
    LOG="$OUT/${NAME}.log"
    echo "=== $NAME -> $LOG ==="
    PYTHONPATH="$PWD" python -u train-scripts/eval_amc23.py \
        --model "$MODEL" \
        --data "$DATA" \
        --n "$N" \
        --seeds "$SEEDS" \
        --temperature "$TEMP" \
        --top-p "$TOP_P" \
        --gpu-memory-utilization "$GPU_UTIL" \
        2>&1 | tee "$LOG"
done

echo
echo "================ MATH500 summary (mean +- std over seeds $SEEDS) ================"
python - "$OUT" <<'EOF'
import glob, json, os, sys
rows = []
for path in sorted(glob.glob(os.path.join(sys.argv[1], '*.log'))):
    for line in open(path):
        if line.startswith('RESULT_JSON '):
            rows.append(json.loads(line[len('RESULT_JSON '):]))
if not rows:
    sys.exit('no results')
keys = [k for k in rows[-1] if k.startswith(('pass@', 'avg@'))]
print(f"{'model':<44}" + ''.join(f'{k:>22}' for k in keys))
for r in rows:
    cells = ''.join(f'{m:>14.4f} ± {s:.4f}' for m, s in (r[k] for k in keys))
    print(f"{os.path.basename(r['model'].rstrip('/')):<44}{cells}")
EOF
