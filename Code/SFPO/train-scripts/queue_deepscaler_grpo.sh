#!/usr/bin/env bash
# Run the three training seeds of the gxpo-qwen-1.5B GRPO study back to back on both GPUs.
#
# After a seed finishes, its checkpoints are slimmed to the two the paper needs (best-pass@1
# and final), keeping only the standalone HF weights: 22GB -> 3.4GB each. Without this the
# three seeds want ~200GB and would run the disk out partway through seed 3.
#
# A seed that FAILS is left untouched, so it can be resumed by rerunning the same command.
#
# Usage: ./train-scripts/queue_deepscaler_grpo.sh          (seeds 42 123 777)
#        SEEDS="123 777" ./train-scripts/queue_deepscaler_grpo.sh
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-42 123 777}"

slim() {
  local dir="$1" best final step
  [ -d "$dir" ] || return 0
  best=$(python -c "import json;print(json.load(open('$dir/best_ckpt.json'))['best_step'])" 2>/dev/null || echo "")
  final=$(ls -d "$dir"/global_step_*/ 2>/dev/null | sed 's#.*global_step_##;s#/##' | sort -n | tail -1)
  for d in "$dir"/global_step_*/; do
    [ -d "$d" ] || continue
    step=$(basename "$d" | sed 's/global_step_//')
    if [ "$step" = "$best" ] || [ "$step" = "$final" ]; then
      find "$d/actor" -maxdepth 1 -name '*.pt' -delete   # drop FSDP shards + optimizer state
    else
      rm -rf "$d"
    fi
  done
  echo "[slim] $dir -> kept best=$best final=$final (HF weights only; no longer resumable)"
}

for S in $SEEDS; do
  echo "=== seed $S starting $(date -Is) ==="
  TRAIN_SEED=$S ./train-scripts/run_deepscaler_grpo.sh
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "=== seed $S done $(date -Is) ==="
    slim "./runs/grpo_deepscaler_qwenmath1.5b_seed${S}"
  else
    echo "=== seed $S FAILED rc=$rc -- checkpoints kept so it can be resumed ==="
  fi
  df -h /workspace | tail -1
done
echo "=== all seeds complete $(date -Is) ==="
