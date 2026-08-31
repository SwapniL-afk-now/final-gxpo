#!/usr/bin/env bash
# Shared Qwen2.5 SFT launcher. The argument is baseline or gxpo.
set -euo pipefail

SFT_METHOD="${1:?usage: sft_common.sh baseline|gxpo [--dry-run]}"
shift
DRY_RUN="${DRY_RUN:-0}"
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
if [[ -n "${TORCHRUN_BIN:-}" ]]; then
  TORCHRUN_CMD=("$TORCHRUN_BIN")
else
  # Use the same workspace interpreter for the launcher and its worker.
  TORCHRUN_CMD=("$PYTHON_BIN" -m torch.distributed.run)
fi

export PYTHONPATH="$SFPO_ROOT/.runtime_deps:$SFPO_ROOT:${PYTHONPATH:-}"
export PATH="$REPO_ROOT/.venv/bin:$PATH"
SFT_PYTHON_INCLUDE_DIR="$REPO_ROOT/.python-dev/usr/include/python3.12"
if [[ -f "$SFT_PYTHON_INCLUDE_DIR/Python.h" ]]; then
  export C_INCLUDE_PATH="$REPO_ROOT/.python-dev/usr/include:$SFT_PYTHON_INCLUDE_DIR${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
  export CPLUS_INCLUDE_PATH="$REPO_ROOT/.python-dev/usr/include:$SFT_PYTHON_INCLUDE_DIR${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
fi
SFT_CUDA_HOME="$REPO_ROOT/.cuda-toolkit"
if [[ -x "$SFT_CUDA_HOME/bin/nvcc" ]]; then
  export CUDA_HOME="${CUDA_HOME:-$SFT_CUDA_HOME}"
  export CUDA_PATH="${CUDA_PATH:-$SFT_CUDA_HOME}"
  export CUDACXX="${CUDACXX:-$SFT_CUDA_HOME/bin/nvcc}"
  export PATH="$SFT_CUDA_HOME/bin:$PATH"
fi
SFT_CUDA_LIB_ROOT="$REPO_ROOT/.venv/lib/python3.12/site-packages/nvidia"
for SFT_CUDA_LIB_DIR in "$SFT_CUDA_LIB_ROOT/cu13/lib" "$SFT_CUDA_LIB_ROOT/cublas/lib" "$SFT_CUDA_LIB_ROOT/cuda_nvrtc/lib" "$SFT_CUDA_LIB_ROOT/cuda_runtime/lib" "$SFT_CUDA_LIB_ROOT/cudnn/lib" "$SFT_CUDA_LIB_ROOT/cufft/lib" "$SFT_CUDA_LIB_ROOT/curand/lib" "$SFT_CUDA_LIB_ROOT/cusolver/lib" "$SFT_CUDA_LIB_ROOT/cusparse/lib" "$SFT_CUDA_LIB_ROOT/nccl/lib" "$SFT_CUDA_LIB_ROOT/nvjitlink/lib" "$SFT_CUDA_LIB_ROOT/nvtx/lib"; do
  if [[ -d "$SFT_CUDA_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="$SFT_CUDA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export GPU_IDS="${GPU_IDS:-${GPU:-2}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_IDS}"
if [[ -z "${GPU_COUNT:-}" ]]; then
  IFS=, read -r -a SFT_GPU_LIST <<< "$GPU_IDS"
  GPU_COUNT="${#SFT_GPU_LIST[@]}"
fi
if ! [[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "SFT preflight failed: GPU_COUNT must be a positive integer; got $GPU_COUNT" >&2
  exit 2
fi
IFS=, read -r -a SFT_GPU_LIST <<< "$GPU_IDS"
if [[ "${#SFT_GPU_LIST[@]}" -ne "$GPU_COUNT" ]]; then
  echo "SFT preflight failed: GPU_COUNT=$GPU_COUNT does not match GPU_IDS=$GPU_IDS" >&2
  exit 2
fi

MODEL="${SFT_MODEL:-Qwen/Qwen2.5-1.5B}"
SFT_SOURCE_DATASET="${SFT_SOURCE_DATASET:-K-and-K/knights-and-knaves}"
SFT_DATA_ROOT="${SFT_DATA_ROOT:-$SFPO_ROOT/data/knights_and_knaves_sft}"
TRAIN_FILE="${SFT_TRAIN_FILE:-$SFT_DATA_ROOT/train.parquet}"
VAL_FILE="${SFT_VAL_FILE:-$SFT_DATA_ROOT/test.parquet}"
RESULT_ROOT="${SFT_RESULTS_ROOT:-${GXPO_RESULTS_ROOT:-$REPO_ROOT/results/gxpo_efficiency}}"

TRAIN_SEED="${TRAIN_SEED:-42}"
SFT_LR="${SFT_LR:-1e-5}"
SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-32}"
# 64 caused a reproducible logits-allocation OOM on this 1.5B setup; keep the
# same global batch through gradient accumulation with the validated safe value.
SFT_MICRO_BATCH_SIZE="${SFT_MICRO_BATCH_SIZE:-8}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-2048}"
SFT_TRUNCATION="${SFT_TRUNCATION:-right}"
SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-10}"
SFT_TOTAL_STEPS="${SFT_TOTAL_STEPS:-0}"
SFT_FSDP_STRATEGY="${SFT_FSDP_STRATEGY:-fsdp2}"
SFT_FSDP_CPU_OFFLOAD="${SFT_FSDP_CPU_OFFLOAD:-False}"
SFT_TEST_FREQ="${SFT_TEST_FREQ:-10}"
SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-0}"
SFT_GREEDY_EVAL_FREQ="${SFT_GREEDY_EVAL_FREQ:-5}"
SFT_VAL_MAX_BATCHES="${SFT_VAL_MAX_BATCHES:-50}"
SFT_EVAL_KIND="${SFT_EVAL_KIND:-knights_and_knaves}"
SFT_EVAL_BENCHMARK_ROOT="${SFT_EVAL_BENCHMARK_ROOT:-$SFT_DATA_ROOT}"
SFT_EVAL_GREEDY_MAX_EXAMPLES="${SFT_EVAL_GREEDY_MAX_EXAMPLES:-0}"
SFT_EVAL_GREEDY_BATCH_SIZE="${SFT_EVAL_GREEDY_BATCH_SIZE:-4}"
SFT_EVAL_GREEDY_MAX_NEW_TOKENS="${SFT_EVAL_GREEDY_MAX_NEW_TOKENS:-3072}"
SFT_EVAL_GREEDY_PROMPT_MAX_LENGTH="${SFT_EVAL_GREEDY_PROMPT_MAX_LENGTH:-2048}"
SFT_VLLM_GPU_MEMORY_UTILIZATION="${SFT_VLLM_GPU_MEMORY_UTILIZATION:-0.18}"
SFT_EVAL_TIMEOUT_S="${SFT_EVAL_TIMEOUT_S:-900}"
SFT_ATTN_IMPL="${SFT_ATTN_IMPL:-flash_attention_2}"
export SFT_ATTN_IMPL
SFT_WANDB_PROJECT="${SFT_WANDB_PROJECT:-gxpo-efficiency-final}"
SFT_WANDB_GROUP="${SFT_WANDB_GROUP:-qwen25-1p5b-knights-and-knaves-b32}"
if [[ -z "${WANDB_MODE+x}" ]]; then
  WANDB_MODE="${WANDB_API_KEY:+online}"
  WANDB_MODE="${WANDB_MODE:-offline}"
fi
WANDB_GROUP="${WANDB_GROUP:-$SFT_WANDB_GROUP}"
WANDB_TAGS="${WANDB_TAGS:-model:qwen25-1p5b,dataset:knights-and-knaves,domain:logical-reasoning,framework:sft,method:${SFT_METHOD},batch:${SFT_TRAIN_BATCH_SIZE},minibatch:${SFT_MICRO_BATCH_SIZE},optimizer:adamw}"
export WANDB_MODE WANDB_PROJECT="$SFT_WANDB_PROJECT" WANDB_GROUP WANDB_TAGS

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

SFT_GXPO_K="${SFT_GXPO_K:-3}"
SFT_GXPO_ALPHA="${SFT_GXPO_ALPHA:-0.8}"
SFT_GXPO_TAU="${SFT_GXPO_TAU:-2.0}"
SFT_GXPO_WARMUP="${SFT_GXPO_WARMUP:-0}"
SFT_GXPO_ZSCORE_W="${SFT_GXPO_ZSCORE_W:-30}"
SFT_GXPO_OMEGA="${SFT_GXPO_OMEGA:-0.1}"
SFT_GXPO_SHUTOFF_MODE="${SFT_GXPO_SHUTOFF_MODE:-cosine}"
SFT_GXPO_TRIGGER_ROBUST="${SFT_GXPO_TRIGGER_ROBUST:-False}"
SFT_GXPO_MIN_POST_WARMUP_OBS="${SFT_GXPO_MIN_POST_WARMUP_OBS:-0}"
SFT_GXPO_MAX_ACTIVE_STEPS="${SFT_GXPO_MAX_ACTIVE_STEPS:-150}"
SFT_GXPO_TRIGGER_PATIENCE="${SFT_GXPO_TRIGGER_PATIENCE:-2}"
SFT_GXPO_ABS_THRESHOLD="${SFT_GXPO_ABS_THRESHOLD:-0}"
SFT_GXPO_SUSTAIN_WINDOW="${SFT_GXPO_SUSTAIN_WINDOW:-10}"

RUN_NAME="${SFT_RUN_NAME:-qwen25_1p5b_${METHOD_NAME}_knights_and_knaves_k${SFT_GXPO_K}_a${SFT_GXPO_ALPHA}_b${SFT_TRAIN_BATCH_SIZE}_seed${TRAIN_SEED}}"
RUN_DIR="$RESULT_ROOT/$RUN_NAME"

missing=0
required_files=("$TRAIN_FILE" "$VAL_FILE")
case "$SFT_EVAL_KIND" in
  knights_and_knaves)
    required_files+=("$SFT_EVAL_BENCHMARK_ROOT/iid_test.parquet"
                     "$SFT_EVAL_BENCHMARK_ROOT/ood_test.parquet")
    ;;
  math)
    required_files+=("$SFT_EVAL_BENCHMARK_ROOT/math500/test.parquet"
                     "$SFT_EVAL_BENCHMARK_ROOT/aime2024/test.parquet"
                     "$SFT_EVAL_BENCHMARK_ROOT/aime2025/test.parquet"
                     "$SFT_EVAL_BENCHMARK_ROOT/amc/test.parquet"
                     "$SFT_EVAL_BENCHMARK_ROOT/minervamath/test.parquet"
                     "$SFT_EVAL_BENCHMARK_ROOT/olympiadbench/test.parquet")
    ;;
  *)
    echo "SFT preflight failed: unsupported SFT_EVAL_KIND=$SFT_EVAL_KIND" >&2
    exit 2
    ;;
esac
for required in "${required_files[@]}"; do
  if [[ ! -f "$required" ]]; then
    echo "SFT preflight failed: missing $required" >&2
    missing=1
  fi
done
if [[ "$MODEL" == /* || "$MODEL" == ./* || "$MODEL" == ../* ]]; then
  if [[ ! -f "$MODEL/config.json" ]]; then
    echo "SFT preflight failed: local model config was not found under $MODEL" >&2
    missing=1
  fi
  if [[ ! -f "$MODEL/model.safetensors" && ! -f "$MODEL/model.safetensors.index.json" ]]; then
    echo "SFT preflight failed: local model weights were not found under $MODEL" >&2
    missing=1
  fi
fi

if [[ "$missing" -ne 0 ]]; then
  echo "Run prepare_knights_and_knaves_sft_data.sh first, or set SFT_DATA_ROOT/SFT_TRAIN_FILE/SFT_VAL_FILE." >&2
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
if ! [[ "$SFT_TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ && "$SFT_TOTAL_STEPS" =~ ^[0-9][0-9]*$ ]]; then
  echo "SFT preflight failed: epochs must be positive and total steps must be zero or a positive integer." >&2
  exit 2
fi
"$PYTHON_BIN" "$SFPO_ROOT/tools/validate_sft_data.py" --data-root "$SFT_DATA_ROOT" \
  --train-file "$TRAIN_FILE" --test-file "$VAL_FILE" --source-dataset "$SFT_SOURCE_DATASET" >/dev/null

mkdir -p "$RUN_DIR"
export WANDB_DIR="$RUN_DIR"
export SFT_RUN_DIR="$RUN_DIR"
export SFT_REPO_ROOT="$REPO_ROOT"
export SFT_WANDB_PROJECT SFT_WANDB_GROUP
export METHOD_NAME USE_GXPO MODEL SFT_SOURCE_DATASET SFT_DATA_ROOT TRAIN_FILE VAL_FILE TRAIN_SEED SFT_EVAL_KIND
export SFT_TRAIN_BATCH_SIZE SFT_MICRO_BATCH_SIZE SFT_MAX_LENGTH SFT_LR SFT_FSDP_STRATEGY SFT_FSDP_CPU_OFFLOAD
export SFT_TRUNCATION
export SFT_TOTAL_EPOCHS SFT_TOTAL_STEPS SFT_GXPO_K SFT_GXPO_ALPHA
export SFT_GXPO_TAU SFT_GXPO_WARMUP SFT_GXPO_ZSCORE_W SFT_GXPO_OMEGA
export SFT_GXPO_SHUTOFF_MODE SFT_GXPO_TRIGGER_ROBUST
export SFT_GXPO_MIN_POST_WARMUP_OBS SFT_GXPO_MAX_ACTIVE_STEPS
export SFT_GXPO_TRIGGER_PATIENCE SFT_GXPO_ABS_THRESHOLD SFT_GXPO_SUSTAIN_WINDOW
export SFT_EVAL_BENCHMARK_ROOT SFT_GREEDY_EVAL_FREQ
export SFT_EVAL_GREEDY_MAX_EXAMPLES SFT_EVAL_GREEDY_BATCH_SIZE
export SFT_EVAL_GREEDY_MAX_NEW_TOKENS SFT_EVAL_GREEDY_PROMPT_MAX_LENGTH
export SFT_VLLM_GPU_MEMORY_UTILIZATION SFT_EVAL_TIMEOUT_S

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
    "method": "sft",
    "model": os.environ["MODEL"],
    "source_dataset": os.environ["SFT_SOURCE_DATASET"],
    "eval_kind": os.environ["SFT_EVAL_KIND"],
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
        "trigger_robust": os.environ["SFT_GXPO_TRIGGER_ROBUST"] == "True",
        "min_post_warmup_obs": int(os.environ["SFT_GXPO_MIN_POST_WARMUP_OBS"]),
        "max_active_steps": int(os.environ["SFT_GXPO_MAX_ACTIVE_STEPS"]),
        "trigger_patience": int(os.environ["SFT_GXPO_TRIGGER_PATIENCE"]),
        "abs_threshold": float(os.environ["SFT_GXPO_ABS_THRESHOLD"]),
        "sustain_window": int(os.environ["SFT_GXPO_SUSTAIN_WINDOW"]),
    },
    "attention": os.environ["SFT_ATTN_IMPL"],
    "fsdp_strategy": os.environ["SFT_FSDP_STRATEGY"],
    "fsdp_cpu_offload": os.environ["SFT_FSDP_CPU_OFFLOAD"] == "True",
    "wandb_project": os.environ["SFT_WANDB_PROJECT"],
    "wandb_group": os.environ["WANDB_GROUP"],
    "eval_frequency": int(os.environ["SFT_GREEDY_EVAL_FREQ"]),
    "eval_timeout_s": int(os.environ["SFT_EVAL_TIMEOUT_S"]),
    "sft_data_root": os.environ["SFT_DATA_ROOT"],
    "benchmark_data_root": str(Path(os.environ["SFT_EVAL_BENCHMARK_ROOT"])),
    "git_commit": commit,
}
(out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

OVERRIDES=(
  "data.train_files=$TRAIN_FILE"
  "data.val_files=$VAL_FILE"
  "data.prompt_key=prompt"
  "data.response_key=response"
  "data.train_batch_size=$SFT_TRAIN_BATCH_SIZE"
  "data.micro_batch_size_per_gpu=$SFT_MICRO_BATCH_SIZE"
  "data.max_length=$SFT_MAX_LENGTH"
  "data.truncation=$SFT_TRUNCATION"
  "model.partial_pretrain=$MODEL"
  "model.attn_implementation=$SFT_ATTN_IMPL"
  "model.enable_gradient_checkpointing=True"
  "+model.fsdp_config.strategy=$SFT_FSDP_STRATEGY"
  "model.fsdp_config.cpu_offload=$SFT_FSDP_CPU_OFFLOAD"
  "model.fsdp_config.offload_params=True"
  "model.use_liger=False"
  "optim.lr=$SFT_LR"
  "optim.betas=[0.9,0.999]"
  "optim.weight_decay=0.01"
  "optim.warmup_steps_ratio=0.03"
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
  "+optim.gxpo_trigger_robust=$SFT_GXPO_TRIGGER_ROBUST"
  "+optim.gxpo_min_post_warmup_obs=$SFT_GXPO_MIN_POST_WARMUP_OBS"
  "+optim.gxpo_max_active_steps=$SFT_GXPO_MAX_ACTIVE_STEPS"
  "+optim.gxpo_trigger_patience=$SFT_GXPO_TRIGGER_PATIENCE"
  "+optim.gxpo_abs_threshold=$SFT_GXPO_ABS_THRESHOLD"
  "+optim.gxpo_sustain_window=$SFT_GXPO_SUSTAIN_WINDOW"
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
  "+trainer.greedy_eval_freq=$SFT_GREEDY_EVAL_FREQ"
  "+trainer.eval_benchmark_root=$SFT_EVAL_BENCHMARK_ROOT"
  "+trainer.eval_kind=$SFT_EVAL_KIND"
  "+trainer.eval_greedy_max_examples=$SFT_EVAL_GREEDY_MAX_EXAMPLES"
  "+trainer.eval_greedy_batch_size=$SFT_EVAL_GREEDY_BATCH_SIZE"
  "+trainer.eval_greedy_max_new_tokens=$SFT_EVAL_GREEDY_MAX_NEW_TOKENS"
  "+trainer.eval_greedy_prompt_max_length=$SFT_EVAL_GREEDY_PROMPT_MAX_LENGTH"
  "+trainer.eval_greedy_vllm_gpu_memory_utilization=$SFT_VLLM_GPU_MEMORY_UTILIZATION"
  "+trainer.eval_greedy_timeout_s=$SFT_EVAL_TIMEOUT_S"
  "+trainer.val_max_batches=$SFT_VAL_MAX_BATCHES"
  "+trainer.keep_best_only=True"
  "trainer.seed=$TRAIN_SEED"
)

cat <<EOT
SFT launch configuration
  method             : $METHOD_NAME
  model              : $MODEL
  dataset            : $SFT_SOURCE_DATASET
  train              : $TRAIN_FILE
  validation         : $VAL_FILE
  output             : $RUN_DIR
  batch / microbatch : $SFT_TRAIN_BATCH_SIZE / $SFT_MICRO_BATCH_SIZE
  max length         : $SFT_MAX_LENGTH
  epochs / max steps : $SFT_TOTAL_EPOCHS / $SFT_TOTAL_STEPS
  seed / optimizer   : $TRAIN_SEED / AdamW
  gxpo K / alpha     : $SFT_GXPO_K / $SFT_GXPO_ALPHA
  wandb project / group: $SFT_WANDB_PROJECT / $SFT_WANDB_GROUP
  wandb mode         : $WANDB_MODE
  gpu / train attention : $GPU_IDS / $SFT_ATTN_IMPL
  eval backend / freq   : $VLLM_ATTENTION_BACKEND / $SFT_GREEDY_EVAL_FREQ steps
  eval kind              : $SFT_EVAL_KIND
EOT

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

"$PYTHON_BIN" - <<'PY'
import importlib
import os
modules = ["pandas", "tensordict", "transformers"]
if os.environ.get("SFT_ATTN_IMPL") == "flash_attention_2":
    modules.append("flash_attn")
for module in modules:
    try:
        importlib.import_module(module)
    except Exception as exc:
        raise SystemExit(f"SFT preflight failed importing {module}: {exc}")
PY
cd "$SFPO_ROOT"
"${TORCHRUN_CMD[@]}" --standalone --nnodes=1 --nproc_per_node="$GPU_COUNT" \
  -m verl.trainer.fsdp_sft_trainer \
  "${OVERRIDES[@]}" 2>&1 | tee "$RUN_DIR/train.log"
