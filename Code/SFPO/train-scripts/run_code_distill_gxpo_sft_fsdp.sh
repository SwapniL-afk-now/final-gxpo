#!/usr/bin/env bash
# Two-GPU FSDP1 partition (override GPU_IDS/GPU_COUNT for another placement) GXPO-SFT distillation run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$CODE_ROOT/.." && pwd)"
cd "$CODE_ROOT"
ENV_FILE="${ENV_FILE:-/workspace/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a; source "${ENV_FILE}"; set +a
fi
if [[ -n "${HF_API_KEY:-}" ]]; then
  export HF_TOKEN="${HF_TOKEN:-$HF_API_KEY}"
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_API_KEY}"
fi

export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-math-distillation}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-math_distill_gxpo_gpus0123}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-math_distill_fsdppartition}"

MODEL="${MATH_STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DATA_ROOT="${MATH_DISTILL_ROOT:-/workspace/jepa-grpo-cache/data/math_distill_r1_7b}"
EVAL_ROOT="${MATH_EVAL_ROOT:-/workspace/data/sdc_validation_normalized}"
RUN_DIR="${CODE_DISTILL_RUN_ROOT:-$CODE_ROOT/runs/math_distill_gxpo_sft}"
MAX_STEPS="${MAX_STEPS:-400}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
KD_ENABLED="${KD_ENABLED:-1}"
KD_TOPK="${KD_TOPK:-32}"
KD_WEIGHT="${KD_WEIGHT:-0.5}"
KD_TEMPERATURE="${KD_TEMPERATURE:-2.0}"
GXPO_ALPHA="${GXPO_ALPHA:-0.8}"
GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-legacy}"
IFS=',' read -r -a DISTILL_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT="${#DISTILL_GPU_LIST[@]}"
FSDP_SIZE="$GPU_COUNT"
mkdir -p "$RUN_DIR"
RESUME_ARGS=()
if [[ -n "${RESUME_PATH:-}" ]]; then
  RESUME_ARGS=("trainer.resume_path=$RESUME_PATH")
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'model=%s\ntrain=%s/teacher_kd_train.parquet\nval=%s/teacher_kd_val.parquet\nGPUs=%s FSDP_SIZE=%s global_batch=256 microbatch=4\nKD=offline_top%s weight=%s temperature=%s\nGXPO K=3 alpha=%s state=%s disagreement_window=30 tau=1.5 patience=2 metrics=pass@1\n' "$MODEL" "$DATA_ROOT" "$DATA_ROOT" "$CUDA_VISIBLE_DEVICES" "$FSDP_SIZE" "$KD_TOPK" "$KD_WEIGHT" "$KD_TEMPERATURE" "$GXPO_ALPHA" "$GXPO_OPTIMIZER_STATE_MODE"
  exit 0
fi

for required in "$DATA_ROOT/teacher_kd_train.parquet" "$DATA_ROOT/teacher_kd_val.parquet" \
                "$DATA_ROOT/teacher_kd_train.sidecar/manifest.json" "$DATA_ROOT/teacher_kd_val.sidecar/manifest.json" \
                "$EVAL_ROOT/math500.parquet" "$EVAL_ROOT/aime2024.parquet" "$EVAL_ROOT/aime2025.parquet"; do
  [[ -e "$required" ]] || { echo "Missing required study asset: $required" >&2; exit 2; }
done
if [[ "$MODEL" == /* || "$MODEL" == ./* ]]; then
  [[ -e "$MODEL" ]] || { echo "Missing local model snapshot: $MODEL" >&2; exit 2; }
fi

torchrun --standalone --nnodes=1 --nproc_per_node="$GPU_COUNT" -m verl.trainer.fsdp_sft_trainer \
  data.train_files="['$DATA_ROOT/teacher_kd_train.parquet']" \
  data.val_files="['$DATA_ROOT/teacher_kd_val.parquet']" \
  data.prompt_key=prompt data.response_key=response \
  data.train_batch_size=256 data.micro_batch_size_per_gpu=4 data.max_length=4096 data.truncation=right \
  model.partial_pretrain="$MODEL" model.model_dtype=fp32 \
  model.attn_implementation=flash_attention_2 model.enable_gradient_checkpointing=True \
  model.use_liger=True model.fsdp_config.strategy=fsdp1 \
  model.fsdp_config.cpu_offload=False model.fsdp_config.offload_params=False \
  optim.lr=1e-5 optim.betas='[0.9,0.999]' optim.weight_decay=0.01 \
  optim.warmup_steps_ratio=0.05 optim.clip_grad=1.0 \
  optim.kd_enabled="$KD_ENABLED" optim.kd_topk="$KD_TOPK" \
  optim.kd_weight="$KD_WEIGHT" optim.kd_temperature="$KD_TEMPERATURE" \
  +optim.use_gxpo=True +optim.gxpo_k=3 +optim.gxpo_alpha="$GXPO_ALPHA" +optim.gxpo_delta=1e-8 \
  +optim.gxpo_optimizer_state_mode="$GXPO_OPTIMIZER_STATE_MODE" \
  +optim.gxpo_tau=1.5 +optim.gxpo_zscore_w=30 +optim.gxpo_trigger_patience=2 \
  +optim.gxpo_trigger_signal=disagreement +optim.gxpo_shutoff_mode=trajectory_aware \
  trainer.project_name="$WANDB_PROJECT" trainer.experiment_name="$WANDB_RUN_NAME" \
  trainer.default_local_dir="$RUN_DIR" trainer.default_hdfs_dir=null \
  trainer.logger="['console','wandb']" trainer.total_epochs=100 trainer.total_training_steps="$MAX_STEPS" \
  trainer.test_freq=5 trainer.greedy_eval_freq=5 trainer.save_freq=50 trainer.resumable_save_freq=50 \
  trainer.keep_best_only=True trainer.eval_kind=math_distill trainer.eval_benchmark_root="$EVAL_ROOT" \
  trainer.eval_greedy_max_new_tokens="$MAX_RESPONSE_LENGTH" \
  trainer.eval_greedy_vllm_gpu_memory_utilization=0.70 \
  trainer.eval_greedy_vllm_tensor_parallel_size=1 trainer.eval_greedy_vllm_data_parallel_size=1 \
  trainer.eval_greedy_vllm_max_num_batched_tokens=32768 trainer.eval_greedy_vllm_max_num_seqs=256 \
  trainer.seed="${TRAIN_SEED:-42}" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$RUN_DIR/train.log"
