#!/usr/bin/env bash
#
# qwen25_math_1p5b_kd_gxpo_b256_mb64_gate_v6.sh
#
# Offline KD + GXPO: Qwen2.5-1.5B-Instruct (student) from
# Qwen2.5-Math-7B-Instruct (teacher): batch 256 | K=10 | alpha=0.3 | 2 GPUs.
#
# This is the final-gxpo home for knowledge distillation. It replaces the
# prototype under Joint-Embedding-Guided-Policy-Optimization, which stacked a
# full [tokens, vocab] FP32 copy on top of two persistent GPU probe buffers
# and OOMed on later mini-batches. Here the dedicated KD-SFT trainer consumes a cached
# top-K forward KL target; ordinary SFT never sees these fields. The GXPO
# 3-pass geometry is the same proven muon/adam-gxpo path (KDSFTTrainer
# _gxpo_training_step), and the FP32 normalizer runs in 2048-token chunks so
# multiple passes reuse one ~1.2GB peak instead of growing memory.
#
# Two stages:
#   1. build one cache per dataset (there is no --phase both mode):
#        python tools/kd_sft/build_teacher_topk.py --mode gen \
#          --train-parquet <prompt-parquet> --teacher-path <7B> \
#          --student-tokenizer <1.5B> --out data/kd/<name>_topk32.parquet
#      For sharded builds, run each shard and then:
#        python tools/kd_sft/build_teacher_topk.py --merge-shards \
#          --num-shards 2 --out data/kd/<name>_topk32.parquet
#   2. launch this script (KD-SFT+GXPO, no rollout, no Ray teacher servers).
#
# Usage:
#   bash qwen25_math_1p5b_kd_gxpo_b256_mb64_gate_v6.sh            # launch
#   bash qwen25_math_1p5b_kd_gxpo_b256_mb64_gate_v6.sh --dry-run  # preflight only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"       # checkout root (holds .env, models/, Code/)
SFPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"             # Code/SFPO
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ---------------------------------------------------------------- secrets ----
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

# ------------------------------------------------- self-contained env --------
# Everything below resolves from KD_VENV_ROOT + repo paths, so a bare
# `bash ...sh` in a fresh tmux shell works: interpreter, CUDA/FT libs,
# Python headers (Triton/FlashInfer JIT), HF + vLLM caches, NCCL, tmpdirs.
KD_VENV_ROOT="${KD_VENV_ROOT:-$SFPO_ROOT/.venv}"
PYTHON_BIN="$KD_VENV_ROOT/bin/python"
KD_SITE_PACKAGES="$KD_VENV_ROOT/lib/python3.12/site-packages"
export VIRTUAL_ENV="$KD_VENV_ROOT"
export PATH="$KD_VENV_ROOT/bin:$PATH"
export PYTHONPATH="$SFPO_ROOT:$KD_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
# CUDA runtime + math libs shipped inside the venv (cu13 stack).
for _kd_lib in "$KD_SITE_PACKAGES/nvidia/cu13/lib" "$KD_SITE_PACKAGES/torch/lib"; do
  [[ -d "$_kd_lib" ]] && export LD_LIBRARY_PATH="$_kd_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
unset _kd_lib
export CUDA_HOME="$KD_SITE_PACKAGES/nvidia/cu13"
# Python headers for Triton/FlashInfer JIT (no system python3.12-dev here).
KD_PY_INCLUDE="${KD_PY_INCLUDE:-$REPO_ROOT/.python-dev/usr/include/python3.12}"
[[ -f "$KD_PY_INCLUDE/Python.h" ]] && export CPATH="$KD_PY_INCLUDE${CPATH:+:$CPATH}"
# FlashInfer version-check bypass (cubin tops out below the pinned wheel;
# same bypass the proven GXPO launchers use) + vLLM sampler setting.
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# Caches: repo-local for HF + per-run for vLLM (set fully after RUN_DIR).
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}"
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
KD_TMPROOT="$(cd -- "$REPO_ROOT/.." && pwd)"
export TMPDIR="${TMPDIR:-$KD_TMPROOT/.gxpo-tmp}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export RAY_TMPDIR="${RAY_TMPDIR:-$KD_TMPROOT/.gxpo-ray}"
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$XDG_CACHE_HOME" 2>/dev/null || true

# ------------------------------------------------------------ kd config ------
# Same K/alpha/gate profile as qwen25_math_1p5b_gxpo_b256_mb64_gate_v6.sh.
export K="${K:-10}"
export REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.3}"
export GXPO_TAU="${GXPO_TAU:-2.0}"
export GXPO_ZSCORE_W="${GXPO_ZSCORE_W:-30}"
export GXPO_WARMUP_STEPS="${GXPO_WARMUP_STEPS:-0}"
export GXPO_SHUTOFF_MODE="${GXPO_SHUTOFF_MODE:-trajectory_aware}"

export STUDENT_MODEL="${STUDENT_MODEL:-$REPO_ROOT/models/Qwen2.5-1.5B-Instruct}"
export TEACHER_MODEL="${TEACHER_MODEL:-$REPO_ROOT/models/Qwen2.5-Math-7B-Instruct}"
export KD_DATA_ROOT="${KD_DATA_ROOT:-$SFPO_ROOT/data/kd}"
export KD_TRAIN="${KD_TRAIN:-$KD_DATA_ROOT/dapo_math_teacher_topk32.parquet}"
export KD_VAL="${KD_VAL:-$KD_DATA_ROOT/math500_teacher_topk32.parquet}"
export KD_TOPK="${KD_TOPK:-32}"
export KD_CHUNK_TOKENS="${KD_CHUNK_TOKENS:-2048}"
export KD_LR="${KD_LR:-1e-5}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"       # 1023 prompt + 3072 response + 1
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}"  # 4*4096=16k tok/GPU peak
# Throughput profile: FA2 actor (model.attn_implementation below), FlashInfer
# vLLM (cache-builder generation), CUDA graphs on (no enforce_eager anywhere).
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export MAX_STEPS="${MAX_STEPS:-400}"
export SAVE_FREQ="${SAVE_FREQ:-20}"
export TEST_FREQ="${TEST_FREQ:-20}"
export GPU_IDS="${GPU_IDS:-0,1}"
export GXPO_RUN_NAME="${GXPO_RUN_NAME:-qwen25_math_1p5b_kd_gxpo_k${K}_a${REPOSITION_ALPHA}_b256_topk${KD_TOPK}}"
export WANDB_PROJECT="${WANDB_PROJECT:-gxpo-efficiency-final}"
export WANDB_GROUP="${WANDB_GROUP:-qwen25-math-1p5b-kd}"
export WANDB_MODE="${WANDB_MODE:-online}"
RUN_DIR="$REPO_ROOT/results/gxpo_efficiency/$GXPO_RUN_NAME"

# ------------------------------------------------------------- preflight -----
MISSING=0
[[ -x "$PYTHON_BIN" ]] || { echo "PREFLIGHT FAIL: no python at $PYTHON_BIN (set KD_VENV_ROOT)" >&2; MISSING=1; }
if ! "$PYTHON_BIN" -c "import torch.distributed.run" 2>/dev/null; then
  echo "PREFLIGHT FAIL: $PYTHON_BIN has no torch.distributed.run" >&2
  MISSING=1
fi
if ! "$PYTHON_BIN" -c "import torch, transformers, flash_attn, flashinfer, liger_kernel, tensordict, hydra, peft; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "PREFLIGHT FAIL: $PYTHON_BIN misses packages (flash_attn/flashinfer/liger/...) or CUDA" >&2
  MISSING=1
fi
[[ -f "$STUDENT_MODEL/config.json" ]] || { echo "PREFLIGHT FAIL: student not found at $STUDENT_MODEL" >&2; MISSING=1; }
[[ -f "$TEACHER_MODEL/config.json" ]] || { echo "PREFLIGHT FAIL: teacher not found at $TEACHER_MODEL" >&2; MISSING=1; }
[[ -f "$KD_TRAIN" ]] || {
  echo "PREFLIGHT FAIL: missing KD train cache: $KD_TRAIN" >&2
  echo "  build it: python tools/kd_sft/build_teacher_topk.py --mode gen --train-parquet <prompt-parquet> \\" >&2
  echo "    --teacher-path $TEACHER_MODEL --student-tokenizer $STUDENT_MODEL --out $KD_TRAIN" >&2
  MISSING=1
}
[[ -f "$KD_VAL" ]] || { echo "PREFLIGHT FAIL: missing KD val cache: $KD_VAL" >&2; MISSING=1; }
if [[ "$MISSING" -eq 0 ]]; then
  # Fast cache-id guard: every cached id must be indexable by the STUDENT
  # embedding table, otherwise the
  # gather/embedding kernel aborts (SIGABRT, no Python traceback). The builder
  # enforces this; this scan fail-closes stale caches.
  if ! "$PYTHON_BIN" - "$STUDENT_MODEL" "$TEACHER_MODEL" "$KD_TRAIN" "$KD_VAL" <<'PY' 2>&1 | tail -n 3; then
import json, sys
import pandas as pd
s_path, t_path, kd_train, kd_val = sys.argv[1:5]
sV = json.load(open(f"{s_path}/config.json"))["vocab_size"]
tV = json.load(open(f"{t_path}/config.json"))["vocab_size"]
print(f"preflight vocabs: student={sV} teacher={tV}")
worst = 0
for p in (kd_train, kd_val):
    df = pd.read_parquet(p, columns=["response_ids", "teacher_topk_ids"])
    for col in ("response_ids", "teacher_topk_ids"):
        for v in df[col].tolist():
            import numpy as np
            a = np.asarray(v)
            if a.dtype == object:
                a = np.concatenate([np.asarray(r).flatten() for r in v])
            worst = max(worst, int(a.max()))
print(f"preflight max cached id={worst} (must be < {sV})")
assert worst < sV, f"stale cache: id {worst} >= student vocab {sV}; rebuild with tools/kd_sft/build_teacher_topk.py"
print("preflight id-range OK")
PY
    echo "PREFLIGHT FAIL: KD cache id-range check failed (see above)" >&2
    MISSING=1
  fi
fi
if [[ "$MISSING" -eq 0 ]]; then
  if ! "$PYTHON_BIN" tools/kd_sft/validate_tokenizers.py \
      --student-tokenizer "$STUDENT_MODEL" --teacher-path "$TEACHER_MODEL" 2>&1 | tail -n 3; then
    echo "PREFLIGHT FAIL: student/teacher tokenizer identity is incompatible" >&2
    MISSING=1
  fi
fi
if [[ "$MISSING" -ne 0 ]]; then echo "Preflight failed - fix the items above and re-run." >&2; exit 2; fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  cat <<EOT
[dry-run] resolved KD+GXPO launch configuration
  repo_root          : $REPO_ROOT
  venv / python      : $PYTHON_BIN
  student            : $STUDENT_MODEL
  teacher            : $TEACHER_MODEL
  kd_train / kd_val  : $KD_TRAIN / $KD_VAL
  method             : kd-gxpo (K=$K, alpha=$REPOSITION_ALPHA, topk=$KD_TOPK, chunk=$KD_CHUNK_TOKENS)
  batch / micro/gpu  : $TRAIN_BATCH_SIZE / $MICRO_BATCH_SIZE_PER_GPU (max_length=$MAX_LENGTH)
  gpus               : $GPU_IDS
  max_steps          : $MAX_STEPS   save_freq $SAVE_FREQ test_freq $TEST_FREQ
  gate               : shutoff=$GXPO_SHUTOFF_MODE tau=$GXPO_TAU zscore_w=$GXPO_ZSCORE_W warmup=$GXPO_WARMUP_STEPS
  run_dir            : $RUN_DIR
[dry-run] preflight OK - would launch now.
EOT
  exit 0
fi

mkdir -p "$RUN_DIR"
export WANDB_DIR="$RUN_DIR" WANDB_PROJECT WANDB_GROUP WANDB_MODE
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
# Per-run vLLM/FlashInfer autotune caches (writable, isolated).
export VLLM_CACHE_ROOT="$RUN_DIR/vllm_cache"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$RUN_DIR/flashinfer_autotune_cache"
mkdir -p "$VLLM_CACHE_ROOT" "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"

cd "$SFPO_ROOT"
# NOTE: launched via "$PYTHON_BIN -m torch.distributed.run" (not the torchrun
# shim) so the workers are guaranteed to run under KD_VENV_ROOT's interpreter.
exec "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node=2 -m verl.trainer.kd_sft_trainer \
  data.train_files="['$KD_TRAIN']" \
  data.val_files="['$KD_VAL']" \
  data.prompt_key=prompt \
  data.response_key=response \
  data.max_length="$MAX_LENGTH" \
  data.truncation=error \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
  +data.teacher_topk="$KD_TOPK" \
  +data.teacher_topk_log_probs_key=teacher_topk_log_probs \
  +data.teacher_topk_ids_key=teacher_topk_ids \
  +data.response_ids_key=response_ids \
  +data.kd_chunk_tokens="$KD_CHUNK_TOKENS" \
  model.partial_pretrain="$STUDENT_MODEL" \
  model.trust_remote_code=True \
  model.attn_implementation=flash_attention_2 \
  model.enable_gradient_checkpointing=True \
  model.use_liger=True \
  use_remove_padding=False \
  ulysses_sequence_parallel_size=1 \
  optim.lr="$KD_LR" \
  optim.clip_grad=1.0 \
  +optim.use_gxpo=True \
  +optim.gxpo_k="$K" \
  +optim.gxpo_alpha="$REPOSITION_ALPHA" \
  +optim.gxpo_delta=1e-8 \
  +optim.gxpo_tau="$GXPO_TAU" \
  +optim.gxpo_zscore_w="$GXPO_ZSCORE_W" \
  +optim.gxpo_shutoff_mode="$GXPO_SHUTOFF_MODE" \
  +optim.gxpo_warmup="$GXPO_WARMUP_STEPS" \
  trainer.default_local_dir="$RUN_DIR" \
  trainer.project_name="$WANDB_PROJECT" \
  trainer.experiment_name="$GXPO_RUN_NAME" \
  trainer.logger="['console','wandb']" \
  trainer.total_epochs=100 \
  trainer.total_training_steps="$MAX_STEPS" \
  +trainer.test_freq="$TEST_FREQ" \
  +trainer.save_freq="$SAVE_FREQ" \
  2>&1 | tee "$RUN_DIR/train.log"
