#!/usr/bin/env bash
# FSDP1 GXPO distillation: same stored math traces and KD objective as AdamW.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CODE_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/workspace/gradient-extrapolation-based-policy-optimization/.venv-h200/bin/python}"
ENV_FILE="${ENV_FILE:-/workspace/.env}"
if [[ -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; fi
if [[ -n "${HF_API_KEY:-}" ]]; then
  export HF_TOKEN="${HF_TOKEN:-$HF_API_KEY}" HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_API_KEY}"
fi
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-math-distillation}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-math_distill_qwen25_1p5b_gxpo_r1traces}"

MODEL="${MATH_STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DATA_ROOT="${MATH_DISTILL_ROOT:-/workspace/jepa-grpo-cache/data/math_distill_r1_7b}"
EVAL_ROOT="${MATH_EVAL_ROOT:-/workspace/data/sdc_validation_normalized}"
RUN_DIR="${MATH_DISTILL_RUN_ROOT:-$CODE_ROOT/runs/$WANDB_RUN_NAME}"
MAX_STEPS="${MAX_STEPS:-400}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
KD_ENABLED="${KD_ENABLED:-1}"
KD_TOPK="${KD_TOPK:-32}"
KD_WEIGHT="${KD_WEIGHT:-0.5}"
KD_TEMPERATURE="${KD_TEMPERATURE:-2.0}"
if [[ "$KD_ENABLED" == "1" ]]; then
  TRAIN_FILE="$DATA_ROOT/teacher_kd_train.parquet"
  VAL_FILE="$DATA_ROOT/teacher_kd_val.parquet"
else
  TRAIN_FILE="$DATA_ROOT/math_r1_train.parquet"
  VAL_FILE="$DATA_ROOT/math_r1_val.parquet"
fi
GXPO_K="${GXPO_K:-3}"
GXPO_ALPHA="${GXPO_ALPHA:-0.8}"
GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"
IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT="${#GPU_LIST[@]}"
mkdir -p "$RUN_DIR"

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'model=%s\ntrain=%s\nval=%s\neval=%s (math500,aime24,aime25)\nGPUs=%s FSDP_SIZE=%s batch=256 microbatch=%s steps=%s\nKD=offline_teacher_logits topk=%s weight=%s temperature=%s\nGXPO K=%s alpha=%s state=%s\neval=greedy n=1 temperature=0 top_p=1\nW&B=%s/%s\n' \
    "$MODEL" "$TRAIN_FILE" "$VAL_FILE" "$EVAL_ROOT" "$CUDA_VISIBLE_DEVICES" "$GPU_COUNT" "$MICRO_BATCH_SIZE_PER_GPU" "$MAX_STEPS" "$KD_TOPK" "$KD_WEIGHT" "$KD_TEMPERATURE" "$GXPO_K" "$GXPO_ALPHA" "$GXPO_OPTIMIZER_STATE_MODE" "$WANDB_PROJECT" "$WANDB_RUN_NAME"
  exit 0
fi

for required in "$TRAIN_FILE" "$VAL_FILE" \
                "$EVAL_ROOT/math500.parquet" "$EVAL_ROOT/aime2024.parquet" "$EVAL_ROOT/aime2025.parquet"; do
  [[ -e "$required" ]] || { echo "Missing required math asset: $required" >&2; exit 2; }
done
if [[ "$KD_ENABLED" == "1" ]]; then
  for required in "$DATA_ROOT/teacher_kd_train.sidecar/manifest.json" "$DATA_ROOT/teacher_kd_val.sidecar/manifest.json"; do
    [[ -e "$required" ]] || { echo "Missing required KD asset: $required" >&2; exit 2; }
  done
fi

RESUME_ARGS=()
if [[ -n "${RESUME_PATH:-}" ]]; then RESUME_ARGS=("trainer.resume_path=$RESUME_PATH"); fi
"$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$GPU_COUNT" -m verl.trainer.fsdp_sft_trainer \
  data.train_files="['$TRAIN_FILE']" data.val_files="['$VAL_FILE']" \
  data.prompt_key=prompt data.response_key=response data.train_batch_size=256 data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
  data.max_length="$MAX_RESPONSE_LENGTH" data.truncation=right model.partial_pretrain="$MODEL" model.model_dtype=fp32 \
  model.attn_implementation=flash_attention_2 model.enable_gradient_checkpointing=True model.use_liger=True \
  model.fsdp_config.strategy=fsdp1 model.fsdp_config.cpu_offload=False model.fsdp_config.offload_params=False \
  optim.lr=1e-5 optim.betas='[0.9,0.999]' optim.weight_decay=0.01 optim.warmup_steps_ratio=0.05 optim.clip_grad=1.0 \
  optim.kd_enabled="$KD_ENABLED" optim.kd_topk="$KD_TOPK" optim.kd_weight="$KD_WEIGHT" optim.kd_temperature="$KD_TEMPERATURE" \
  +optim.use_gxpo=True +optim.gxpo_k="$GXPO_K" +optim.gxpo_alpha="$GXPO_ALPHA" +optim.gxpo_delta=1e-8 \
  +optim.gxpo_optimizer_state_mode="$GXPO_OPTIMIZER_STATE_MODE" +optim.gxpo_tau=1.5 +optim.gxpo_zscore_w=30 \
  +optim.gxpo_trigger_patience=2 +optim.gxpo_trigger_signal=disagreement +optim.gxpo_shutoff_mode=trajectory_aware \
  trainer.project_name="$WANDB_PROJECT" trainer.experiment_name="$WANDB_RUN_NAME" trainer.default_local_dir="$RUN_DIR" \
  trainer.default_hdfs_dir=null trainer.logger="['console','wandb']" trainer.total_epochs=100 trainer.total_training_steps="$MAX_STEPS" \
  trainer.test_freq=5 trainer.greedy_eval_freq=5 trainer.save_freq=50 trainer.resumable_save_freq=50 trainer.keep_best_only=True \
  trainer.eval_kind=math_distill trainer.eval_benchmark_root="$EVAL_ROOT" trainer.eval_greedy_max_new_tokens="$MAX_RESPONSE_LENGTH" \
  trainer.eval_greedy_vllm_gpu_memory_utilization=0.70 trainer.eval_greedy_vllm_tensor_parallel_size=1 trainer.eval_greedy_vllm_data_parallel_size=1 \
  trainer.eval_greedy_vllm_max_num_batched_tokens=32768 trainer.eval_greedy_vllm_max_num_seqs=256 trainer.seed="${TRAIN_SEED:-42}" \
  "${RESUME_ARGS[@]}" 2>&1 | tee -a "$RUN_DIR/train.log"
