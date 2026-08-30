#!/usr/bin/env bash
# Shared Qwen2.5-Math SFT launcher. The argument is baseline or gxpo.
set -euo pipefail

SFT_METHOD="${1:?usage: sft_common.sh baseline|gxpo [--dry-run]}"
shift
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
SFPO_ROOT="$REPO_ROOT/Code/SFPO"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi
TORCHRUN_BIN="${TORCHRUN_BIN:-$REPO_ROOT/.venv/bin/torchrun}"
if [[ ! -x "$TORCHRUN_BIN" ]]; then
  TORCHRUN_BIN="$(command -v torchrun || true)"
fi
if [[ -z "$TORCHRUN_BIN" ]]; then
  echo "SFT preflight failed: torchrun was not found; activate the project environment." >&2
  exit 2
fi

export PYTHONPATH="$SFPO_ROOT/.runtime_deps:$SFPO_ROOT:${PYTHONPATH:-}"
export GPU_IDS="${GPU_IDS:-${GPU:-2}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_IDS}"
export GPU_COUNT="${GPU_COUNT:-1}"
if [[ "$GPU_COUNT" != "1" ]]; then
  echo "This audit SFT profile is single-process; set GPU_COUNT=1." >&2
  exit 2
fi

MODEL="${SFT_MODEL:-${MODEL_QWEN25_MATH_1P5B:-$REPO_ROOT/models/Qwen2.5-Math-1.5B-Instruct}}"
SFT_DATA_ROOT="${SFT_DATA_ROOT:-$SFPO_ROOT/data/math_l35_sft}"
TRAIN_FILE="${SFT_TRAIN_FILE:-$SFT_DATA_ROOT/train.parquet}"
VAL_FILE="${SFT_VAL_FILE:-$SFT_DATA_ROOT/test.parquet}"
RESULT_ROOT="${SFT_RESULTS_ROOT:-${GXPO_RESULTS_ROOT:-$REPO_ROOT/results/gxpo_efficiency}}"

TRAIN_SEED="${TRAIN_SEED:-42}"
SFT_LR="${SFT_LR:-1e-5}"
SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-64}"
SFT_MICRO_BATCH_SIZE="${SFT_MICRO_BATCH_SIZE:-4}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-2048}"
SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-3}"
SFT_TOTAL_STEPS="${SFT_TOTAL_STEPS:-500}"
SFT_TEST_FREQ="${SFT_TEST_FREQ:-10}"
SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-100}"
SFT_VAL_MAX_BATCHES="${SFT_VAL_MAX_BATCHES:-50}"
SFT_ATTN_IMPL="${SFT_ATTN_IMPL:-flash_attention_2}"
SFT_WANDB_PROJECT="${SFT_WANDB_PROJECT:-gxpo-efficiency-sft}"
if [[ -z "${WANDB_MODE+x}" ]]; then
  WANDB_MODE="${WANDB_API_KEY:+online}"
  WANDB_MODE="${WANDB_MODE:-offline}"
fi
export WANDB_MODE WANDB_PROJECT="$SFT_WANDB_PROJECT"

case "$SFT_METHOD" in
  baseline)
    USE_GXPO=False
    METHOD_NAME="sft_baseline"
    ;;
  gxpo)
    USE_GXPO=True
    METHOD_NAME="sft_gxpo"
    ;;
  *)
    echo "Unsupported SFT method: $SFT_METHOD" >&2
    exit 2
    ;;
esac

SFT_GXPO_K="${SFT_GXPO_K:-10}"
SFT_GXPO_ALPHA="${SFT_GXPO_ALPHA:-0.3}"
SFT_GXPO_TAU="${SFT_GXPO_TAU:-5.0}"
SFT_GXPO_WARMUP="${SFT_GXPO_WARMUP:-3}"
SFT_GXPO_ZSCORE_W="${SFT_GXPO_ZSCORE_W:-30}"
SFT_GXPO_OMEGA="${SFT_GXPO_OMEGA:-0.1}"
SFT_GXPO_SHUTOFF_MODE="${SFT_GXPO_SHUTOFF_MODE:-trajectory_aware}"

RUN_NAME="${SFT_RUN_NAME:-qwen25_math_1p5b_${METHOD_NAME}_k${SFT_GXPO_K}_a${SFT_GXPO_ALPHA}_b${SFT_TRAIN_BATCH_SIZE}_seed${TRAIN_SEED}}"
RUN_DIR="$RESULT_ROOT/$RUN_NAME"

missing=0
for required in "$MODEL/config.json" "$TRAIN_FILE" "$VAL_FILE"; do
  if [[ ! -f "$required" ]]; then
    echo "SFT preflight failed: missing $required" >&2
    missing=1
  fi
done
if [[ ! -f "$MODEL/model.safetensors" && ! -f "$MODEL/model.safetensors.index.json" ]]; then
  echo "SFT preflight failed: model weights were not found under $MODEL" >&2
  missing=1
fi

if [[ "$missing" -ne 0 ]]; then
  echo "Run prepare_sft_data.sh first, or set SFT_DATA_ROOT/SFT_TRAIN_FILE/SFT_VAL_FILE." >&2
  exit 2
fi

if [[ "$TRAIN_FILE" == "$VAL_FILE" ]]; then
  echo "SFT preflight failed: train and validation files must differ." >&2
  exit 2
fi

if ! [[ "$SFT_TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$SFT_MICRO_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "SFT preflight failed: batch sizes must be positive integers." >&2
  exit 2
fi
if (( SFT_TRAIN_BATCH_SIZE % SFT_MICRO_BATCH_SIZE != 0 )); then
  echo "SFT preflight failed: train batch must divide evenly by microbatch." >&2
  exit 2
fi
if ! [[ "$SFT_TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ && "$SFT_TOTAL_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SFT preflight failed: epochs and total steps must be positive integers." >&2
  exit 2
fi
"$PYTHON_BIN" "$SFPO_ROOT/tools/validate_sft_data.py" --data-root "$SFT_DATA_ROOT" --train-file "$TRAIN_FILE" --test-file "$VAL_FILE" >/dev/null

mkdir -p "$RUN_DIR"
export WANDB_DIR="$RUN_DIR"
export SFT_RUN_DIR="$RUN_DIR"
export SFT_REPO_ROOT="$REPO_ROOT"
export METHOD_NAME USE_GXPO MODEL SFT_DATA_ROOT TRAIN_FILE VAL_FILE TRAIN_SEED
export SFT_TRAIN_BATCH_SIZE SFT_MICRO_BATCH_SIZE SFT_MAX_LENGTH SFT_LR
export SFT_TOTAL_EPOCHS SFT_TOTAL_STEPS SFT_GXPO_K SFT_GXPO_ALPHA
export SFT_GXPO_TAU SFT_GXPO_WARMUP SFT_GXPO_ZSCORE_W SFT_GXPO_OMEGA
export SFT_GXPO_SHUTOFF_MODE

"$PYTHON_BIN" - "$RUN_DIR/run_manifest.json" <<'PY'
import json
import os
import subprocess
from pathlib import Path

out = Path(os.environ["SFT_RUN_DIR"])
try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=os.environ["SFT_REPO_ROOT"]
    ).decode().strip()
except Exception:
    commit = None

manifest = {
    "schema_version": 1,
    "kind": "sft_training",
    "method": os.environ["METHOD_NAME"],
    "model": os.environ["MODEL"],
    "train_file": os.environ["TRAIN_FILE"],
    "val_file": os.environ["VAL_FILE"],
    "seed": int(os.environ["TRAIN_SEED"]),
    "batch_size": int(os.environ["SFT_TRAIN_BATCH_SIZE"]),
    "micro_batch_size": int(os.environ["SFT_MICRO_BATCH_SIZE"]),
    "max_length": int(os.environ["SFT_MAX_LENGTH"]),
    "learning_rate": float(os.environ["SFT_LR"]),
    "total_epochs": int(os.environ["SFT_TOTAL_EPOCHS"]),
    "total_training_steps": int(os.environ["SFT_TOTAL_STEPS"]),
    "optimizer": "adamw",
    "use_gxpo": os.environ["USE_GXPO"] == "True",
    "gxpo": {
        "k": int(os.environ["SFT_GXPO_K"]),
        "alpha": float(os.environ["SFT_GXPO_ALPHA"]),
        "tau": float(os.environ["SFT_GXPO_TAU"]),
        "warmup": int(os.environ["SFT_GXPO_WARMUP"]),
        "zscore_w": int(os.environ["SFT_GXPO_ZSCORE_W"]),
        "omega": float(os.environ["SFT_GXPO_OMEGA"]),
        "shutoff_mode": os.environ["SFT_GXPO_SHUTOFF_MODE"],
    },
    "benchmark_data_root": str(Path(os.environ["SFT_DATA_ROOT"]).parent),
    "git_commit": commit,
}
(out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

OVERRIDES=(
  "data.train_files=['$TRAIN_FILE']"
  "data.val_files=['$VAL_FILE']"
  "data.prompt_key=prompt"
  "data.response_key=response"
  "data.train_batch_size=$SFT_TRAIN_BATCH_SIZE"
  "data.micro_batch_size_per_gpu=$SFT_MICRO_BATCH_SIZE"
  "data.max_length=$SFT_MAX_LENGTH"
  "data.truncation=error"
  "model.partial_pretrain=$MODEL"
  "model.attn_implementation=$SFT_ATTN_IMPL"
  "model.enable_gradient_checkpointing=True"
  "model.use_liger=False"
  "optim.lr=$SFT_LR"
  "optim.betas=[0.9,0.999]"
  "optim.weight_decay=0.01"
  "optim.warmup_steps_ratio=0.0"
  "optim.clip_grad=1.0"
  "+optim.use_gxpo=$USE_GXPO"
  "+optim.gxpo_k=$SFT_GXPO_K"
  "+optim.gxpo_alpha=$SFT_GXPO_ALPHA"
  "+optim.gxpo_delta=1e-8"
  "+optim.gxpo_tau=$SFT_GXPO_TAU"
  "+optim.gxpo_warmup=$SFT_GXPO_WARMUP"
  "+optim.gxpo_zscore_w=$SFT_GXPO_ZSCORE_W"
  "+optim.gxpo_omega=$SFT_GXPO_OMEGA"
  "+optim.gxpo_shutoff_mode=$SFT_GXPO_SHUTOFF_MODE"
  "use_remove_padding=False"
  "trainer.project_name=$SFT_WANDB_PROJECT"
  "trainer.experiment_name=$RUN_NAME"
  "trainer.default_local_dir=$RUN_DIR"
  "trainer.default_hdfs_dir=null"
  "trainer.logger=['console','wandb']"
  "trainer.total_epochs=$SFT_TOTAL_EPOCHS"
  "trainer.total_training_steps=$SFT_TOTAL_STEPS"
  "+trainer.test_freq=$SFT_TEST_FREQ"
  "+trainer.save_freq=$SFT_SAVE_FREQ"
  "+trainer.val_max_batches=$SFT_VAL_MAX_BATCHES"
  "trainer.seed=$TRAIN_SEED"
)

cat <<EOT
SFT launch configuration
  method             : $METHOD_NAME
  model              : $MODEL
  train              : $TRAIN_FILE
  validation         : $VAL_FILE
  output             : $RUN_DIR
  batch / microbatch : $SFT_TRAIN_BATCH_SIZE / $SFT_MICRO_BATCH_SIZE
  max length         : $SFT_MAX_LENGTH
  epochs / max steps : $SFT_TOTAL_EPOCHS / $SFT_TOTAL_STEPS
  seed / optimizer   : $TRAIN_SEED / AdamW
  gxpo K / alpha     : $SFT_GXPO_K / $SFT_GXPO_ALPHA
  wandb mode         : $WANDB_MODE
EOT

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

"$PYTHON_BIN" - <<'PY'
import importlib
for module in ("pandas", "tensordict", "transformers", "flash_attn"):
    try:
        importlib.import_module(module)
    except Exception as exc:
        raise SystemExit(f"SFT preflight failed importing {module}: {exc}")
PY
cd "$SFPO_ROOT"
"$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node=1 \
  -m verl.trainer.fsdp_sft_trainer \
  "${OVERRIDES[@]}" 2>&1 | tee "$RUN_DIR/train.log"
