#!/usr/bin/env bash
# Prepare all datasets for the GXPO paper protocol:
#   train: Hendrycks MATH Level 3-5
#   eval:  Math-500, AMC23, GSM8K, MinervaMath, OlympiadBench
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/data/gxpo}"
cd "$(dirname "$0")/.."

python examples/data_preprocess/math_dataset.py --local_dir "$DATA_ROOT/train" --levels 3,4,5
python examples/data_preprocess/test_math500.py --local_dir "$DATA_ROOT/eval/math500"
python examples/data_preprocess/test_amc.py --local_dir "$DATA_ROOT/eval/amc23"
python examples/data_preprocess/test_gsm8k.py --local_dir "$DATA_ROOT/eval/gsm8k"
python examples/data_preprocess/test_minervamath.py --local_dir "$DATA_ROOT/eval/minerva"
python examples/data_preprocess/test_olympiad.py --local_dir "$DATA_ROOT/eval/olympiad"

echo "Done. Parquets under $DATA_ROOT:"
find "$DATA_ROOT" -name '*.parquet'
