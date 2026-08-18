#!/usr/bin/env bash
set -euo pipefail

SMOKE_NAME="${1:?smoke name required}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="$REPO_ROOT/models/Qwen2.5-Math-1.5B-Instruct"
DAPO_PATH="$REPO_ROOT/data/dapo_math/train.parquet"
LIGHTEVAL_PATH="$REPO_ROOT/data/lighteval-math/train.parquet"
RUN_DIR="$REPO_ROOT/results/perf_smoke/qwen1p5b/$SMOKE_NAME"

for required in "$MODEL_PATH" "$DAPO_PATH" "$LIGHTEVAL_PATH"; do
  [[ -e "$required" ]] || { echo "Missing smoke asset: $required" >&2; exit 2; }
done
if [[ -e "$RUN_DIR/train_metrics.jsonl" ]]; then
  echo "Refusing to overwrite completed $SMOKE_NAME output: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

# The image-level vLLM cache may be root-owned after the managed service has
# started.  Keep each smoke self-contained and writable by the experiment user.
export VLLM_CACHE_ROOT="$RUN_DIR/vllm_cache"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$RUN_DIR/flashinfer_autotune_cache"
mkdir -p "$VLLM_CACHE_ROOT"
mkdir -p "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"

case "$SMOKE_NAME" in
  smoke_a)
    VLLM_GPU_MEMORY_UTILIZATION=0.50
    VLLM_MAX_NUM_BATCHED_TOKENS=8192
    VLLM_MAX_NUM_SEQS=1024
    VLLM_ATTENTION_BACKEND=null
    VLLM_SLEEP_LEVEL=1
    PPO_MINI_BATCH_SIZE=16
    PPO_MAX_TOKEN_LEN_PER_GPU=24576
    LOG_PROB_MICRO_BATCH_SIZE=8
    LOG_PROB_MAX_TOKEN_LEN_PER_GPU=24576
    ;;
  smoke_b)
    # Keep enough headroom for the actor's retained CUDA working set while
    # increasing scheduler/token capacity and using level-2 sleep.
    VLLM_GPU_MEMORY_UTILIZATION=0.50
    VLLM_MAX_NUM_BATCHED_TOKENS=32768
    VLLM_MAX_NUM_SEQS=1024
    VLLM_ATTENTION_BACKEND=FLASHINFER
    VLLM_SLEEP_LEVEL=2
    PPO_MINI_BATCH_SIZE=32
    PPO_MAX_TOKEN_LEN_PER_GPU=32768
    LOG_PROB_MICRO_BATCH_SIZE=16
    LOG_PROB_MAX_TOKEN_LEN_PER_GPU=65536
    ;;
  *)
    echo "Unknown smoke name: $SMOKE_NAME" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$REPO_ROOT:/workspace/.gxpo_pydeps${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export GXPO_EFFICIENCY_RUN=1
export GXPO_RUN_NAME="qwen1p5b_${SMOKE_NAME}"
export GXPO_GPU_TELEMETRY_INTERVAL=1
export VLLM_SLEEP_LEVEL
export TRAIN_SEED="${TRAIN_SEED:-3407}"
export RAY_ADDRESS=local
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1

TRAIN_FILES="['$DAPO_PATH','$LIGHTEVAL_PATH']"

ARGS=(
  algorithm.adv_estimator=grpo
  +algorithm.norm_adv_by_std_in_grpo=False
  +algorithm.use_kl_in_reward=False
  data.train_files="$TRAIN_FILES"
  data.val_files="[]"
  data.train_batch_size=256
  data.max_prompt_length=1024
  data.max_response_length=3072
  data.filter_overlong_prompts=True
  data.truncation=error
  +data.seed="$TRAIN_SEED"
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.attn_implementation=flash_attention_3
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr=1e-6
  +actor_rollout_ref.actor.optim.name=adamw
  +actor_rollout_ref.actor.data_loader_seed="$TRAIN_SEED"
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.fsdp_config.fsdp_size=1
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE"
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOG_PROB_MAX_TOKEN_LEN_PER_GPU"
  actor_rollout_ref.ref.fsdp_config.param_offload=False
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION"
  actor_rollout_ref.rollout.max_num_batched_tokens="$VLLM_MAX_NUM_BATCHED_TOKENS"
  actor_rollout_ref.rollout.max_num_seqs="$VLLM_MAX_NUM_SEQS"
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.val_kwargs.do_sample=False
  actor_rollout_ref.rollout.val_kwargs.temperature=0
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0
  algorithm.kl_ctrl.kl_coef=0.0
  trainer.logger="['console']"
  trainer.project_name=gxpo-perf-smoke
  trainer.experiment_name="qwen1p5b_${SMOKE_NAME}"
  trainer.default_local_dir="$RUN_DIR"
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.save_freq=-1
  trainer.test_freq=-1
  trainer.resume_mode=disable
  +trainer.val_before_train=False
  +trainer.max_steps=2
  trainer.total_training_steps=2
  trainer.total_epochs=100
)

if [[ "$VLLM_ATTENTION_BACKEND" != "null" ]]; then
  ARGS+=("actor_rollout_ref.rollout.attention_backend=$VLLM_ATTENTION_BACKEND")
fi

echo "[performance smoke] name=$SMOKE_NAME output=$RUN_DIR"
echo "[performance smoke] model=$MODEL_PATH"
echo "[performance smoke] vllm_mem=$VLLM_GPU_MEMORY_UTILIZATION max_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS max_seqs=$VLLM_MAX_NUM_SEQS sleep_level=$VLLM_SLEEP_LEVEL"
echo "[performance smoke] ppo_mini=$PPO_MINI_BATCH_SIZE ppo_max_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU logprob_micro=$LOG_PROB_MICRO_BATCH_SIZE logprob_max_tokens=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU"

/venv/main/bin/python -u -m verl.trainer.main_ppo "${ARGS[@]}" 2>&1 | tee "$RUN_DIR/train.log"

[[ -s "$RUN_DIR/train_metrics.jsonl" ]] || { echo "Missing metrics output: $RUN_DIR/train_metrics.jsonl" >&2; exit 3; }
