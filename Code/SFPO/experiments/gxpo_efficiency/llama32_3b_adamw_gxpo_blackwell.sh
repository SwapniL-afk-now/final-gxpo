#!/usr/bin/env bash
# Self-contained Llama 3.2 3B RL launcher: AdamW + METHOD, Blackwell profile.
# Hardware target: NVIDIA RTX PRO 6000 Blackwell (sm_120, 97 GiB, no NVLink).
# The experiment configuration matches the H200 sibling script; only the
# runtime settings that Blackwell cannot satisfy are different.
# This file owns the complete experiment configuration and does not delegate
# model/method settings to another launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$CODE_ROOT/../.." && pwd)"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
cd "$CODE_ROOT"

# Load W&B credentials and other persistent project settings without printing them.
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

export PYTHONPATH="$CODE_ROOT/.runtime_deps:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HUB_CACHE"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in "$(command -v python 2>/dev/null || true)" \
                   "$PROJECT_ROOT/.venv/bin/python" \
                   "/workspace/gradient-extrapolation-based-policy-optimization/.venv-h200/bin/python" \
                   "/venv/main/bin/python" \
                   "$(command -v python3 2>/dev/null || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN:-}" || ! -x "$PYTHON_BIN" ]]; then
  echo "No usable Python interpreter found" >&2
  exit 2
fi
export PYTHON_BIN

# ---------------------------------------------------------------------------
# Blackwell (RTX PRO 6000, sm_120) runtime profile.
# The H200 sibling script uses FlashAttention-3 and Hopper-sized memory budgets.
# Neither is valid here: FA3 has no sm_120 build, the cards have 97 GiB instead
# of 141 GiB, and nvidia-smi reports "P2P not supported" between all pairs, so
# NCCL must stay on the shared-memory path.
# ---------------------------------------------------------------------------
# flash-attn 2.8.3 in this venv is a CUDA-13 build; without libcudart.so.13 on
# the loader path "import flash_attn" fails at runtime even though the module
# is installed.
RUNTIME_PROBE="$("$PYTHON_BIN" - <<'PY'
import pathlib
import sys
import sysconfig

roots = []
purelib = sysconfig.get_paths().get('purelib')
if purelib:
    roots.append(pathlib.Path(purelib))
roots.extend(pathlib.Path(p) for p in sys.path if p.endswith('site-packages'))
cudart = ''
for root in roots:
    candidate = root / 'nvidia' / 'cu13' / 'lib' / 'libcudart.so.13'
    if candidate.exists():
        cudart = str(candidate.parent)
        break
print(cudart)

# PYTORCH_CUDA_ALLOC_CONF is deprecated from torch 2.9; the new name is
# PYTORCH_ALLOC_CONF. Emit the name this interpreter actually accepts.
try:
    import torch
    version = tuple(int(part) for part in torch.__version__.split('+')[0].split('.')[:2])
except Exception:  # noqa: BLE001
    version = (0, 0)
print('PYTORCH_ALLOC_CONF' if version >= (2, 9) else 'PYTORCH_CUDA_ALLOC_CONF')
PY
)"
CUDART13_DIR="$(printf '%s\n' "$RUNTIME_PROBE" | sed -n '1p')"
ALLOC_CONF_VAR="$(printf '%s\n' "$RUNTIME_PROBE" | sed -n '2p')"
ALLOC_CONF_VAR="${ALLOC_CONF_VAR:-PYTORCH_ALLOC_CONF}"
if [[ -n "$CUDART13_DIR" ]]; then
  export LD_LIBRARY_PATH="$CUDART13_DIR:${LD_LIBRARY_PATH:-}"
fi
# No NVLink and no PCIe peer-to-peer on these cards; keep NCCL off the P2P path.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
if [[ -z "${!ALLOC_CONF_VAR:-}" ]]; then
  export "$ALLOC_CONF_VAR=expandable_segments:True,max_split_size_mb:512"
fi

MODEL_ALIAS="llama32-3b-adamw-gxpo-blackwell"
DEFAULT_MODEL="${LLAMA32_3B_SNAPSHOT:-/workspace/.hf_home/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95}"
MODEL_ID="${MODEL_LLAMA32_3B:-$DEFAULT_MODEL}"
METHOD="gxpo"

# Fixed comparison configuration; every value remains environment-overridable.
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_IDS}"
GPU_COUNT="${GPU_COUNT:-4}"
FSDP_SIZE="${FSDP_SIZE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
MAX_STEPS="${MAX_STEPS:-400}"
SAVE_FREQ="${SAVE_FREQ:-5}"
TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-5}"
TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-disable}"
TRAINER_RESUME_FROM_PATH="${TRAINER_RESUME_FROM_PATH:-False}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
TRAIN_SEED="${TRAIN_SEED:-3407}"
FINAL_EVAL_SEEDS="${FINAL_EVAL_SEEDS:-0 1 2 3}"
LR="${LR:-1e-6}"
# FlashAttention-3 has no sm_120 build (Hopper-only), so FA2 is the fastest
# valid kernel here; "sdpa" is the safe fallback.
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
USE_LIGER="${USE_LIGER:-True}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-True}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-True}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
OPTIMIZER_NAME="${OPTIMIZER_NAME:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.001}"
ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-False}"
ACTOR_KL_LOSS_COEF="${ACTOR_KL_LOSS_COEF:-0.0}"
ACTOR_KL_LOSS_TYPE="${ACTOR_KL_LOSS_TYPE:-low_var_kl}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-8}"
# 97 GiB per card instead of 141 GiB: smaller activation and KV budgets.
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.60}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-512}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-True}"
# vLLM selects FLASH_ATTN (FA2 path) on sm_120; FLASHINFER runs but is not
# the tuned default for this architecture.
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-False}"
VLLM_FREE_CACHE_ENGINE="${VLLM_FREE_CACHE_ENGINE:-False}"
FILTER_MIXED_RESPONSES="${FILTER_MIXED_RESPONSES:-True}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
WANDB_PROJECT="${WANDB_PROJECT:-gxpo-efficiency-final}"
WANDB_MODE="${WANDB_MODE:-online}"
DATA_ROOT="${GXPO_DATA_ROOT:-/workspace/data}"
RESULT_ROOT="${GXPO_RESULTS_ROOT:-$CODE_ROOT/results/gxpo_efficiency}"

if [[ "gxpo" == "gxpo" ]]; then
  K="${K:-10}"
  REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.3}"
  GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"
  GXPO_TRIGGER_SIGNAL="${GXPO_TRIGGER_SIGNAL:-entropy}"
  # Gate calibration (measured against production entropy noise, within-30-window CV ~0.07):
  # tau=3.0/patience=3 on the z-path gives 0/60 false positives on a flat healthy series,
  # while tau=1.5/patience=2 gives 45/60. The z-path stays conservative and only catches
  # genuine spikes; slow drift -- the failure this gate exists for -- is caught by the
  # relative sustained-level criterion (GXPO_OMEGA), which costs no false positives.
  # Warmup is 0 so the frozen level baseline is learned from the early regime, before any
  # drift has had time to redefine "normal".
  GXPO_WARMUP_STEPS="${GXPO_WARMUP_STEPS:-0}"
  GXPO_TAU="${GXPO_TAU:-3.0}"
  GXPO_ZSCORE_W="${GXPO_ZSCORE_W:-30}"
  GXPO_TRIGGER_PATIENCE="${GXPO_TRIGGER_PATIENCE:-3}"
  GXPO_FALLBACK_MODE="${GXPO_FALLBACK_MODE:-permanent}"
  GXPO_FALLBACK_WINDOW="${GXPO_FALLBACK_WINDOW:-10}"
  GXPO_RESET_ENTROPY_AFTER_WARMUP="${GXPO_RESET_ENTROPY_AFTER_WARMUP:-True}"
  GXPO_SHUTOFF_MODE="${GXPO_SHUTOFF_MODE:-trajectory_aware}"
  GXPO_DIAG_FREQ="${GXPO_DIAG_FREQ:-10}"
  RUN_NAME="${GXPO_RUN_NAME:-llama32_3b_adamw_gxpo_k${K}_a${REPOSITION_ALPHA}_b256_mb64_g4_fsdp4_blackwell}"
else
  RUN_NAME="${GXPO_RUN_NAME:-llama32_3b_adamw_grpo_b256_mb64_g4_fsdp4_blackwell}"
fi

DAPO_TRAIN="${DAPO_TRAIN:-$DATA_ROOT/dapo_math/train.parquet}"
LIGHTEVAL_TRAIN="${LIGHTEVAL_TRAIN:-$DATA_ROOT/lighteval-math/train.parquet}"
MATH500="${MATH500:-$DATA_ROOT/math500/test.parquet}"
AIME24="${AIME24:-$DATA_ROOT/aime2024/test.parquet}"
AIME25="${AIME25:-$DATA_ROOT/aime2025/test.parquet}"
AMC23="${AMC23:-$DATA_ROOT/amc/test.parquet}"
MINERVA="${MINERVA:-$DATA_ROOT/minervamath/test.parquet}"
OLYMPIAD="${OLYMPIAD:-$DATA_ROOT/olympiadbench/test.parquet}"
for required in "$DAPO_TRAIN" "$LIGHTEVAL_TRAIN" "$MATH500" "$AIME24" "$AIME25" "$AMC23" "$MINERVA" "$OLYMPIAD"; do
  [[ -f "$required" ]] || { echo "Missing prepared dataset: $required" >&2; exit 2; }
done
[[ -e "$MODEL_ID" ]] || { echo "Model path does not exist: $MODEL_ID" >&2; exit 2; }

RUN_DIR="$RESULT_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR" "$RUN_DIR/vllm_cache" "$RUN_DIR/flashinfer_autotune_cache"
export GXPO_EFFICIENCY_RUN=1 GXPO_RUN_NAME="$RUN_NAME" GXPO_MODEL_ALIAS="$MODEL_ALIAS"
export WANDB_PROJECT WANDB_MODE WANDB_DIR="$RUN_DIR" RAY_ADDRESS=local MPLBACKEND=Agg
export VLLM_CACHE_ROOT="$RUN_DIR/vllm_cache"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$RUN_DIR/flashinfer_autotune_cache"
export VLLM_SLEEP_LEVEL="${VLLM_SLEEP_LEVEL:-2}"
export TRAIN_SEED FINAL_EVAL_SEEDS
export WANDB_GROUP="${WANDB_GROUP:-$MODEL_ALIAS}"
export WANDB_TAGS="${WANDB_TAGS:-model:$MODEL_ALIAS,method:$METHOD,optimizer:adamw,gpu:rtx-pro-6000-blackwell,batch:256,minibatch:64,gpus:4,fsdp:4,experiment:final-efficiency}"

cat <<EOT
[llama AdamW launcher | Blackwell sm_120 profile]
model=$MODEL_ID
method=$METHOD
optimizer=$OPTIMIZER_NAME
batch/minibatch=$TRAIN_BATCH_SIZE/$PPO_MINI_BATCH_SIZE
GPU_IDS=$GPU_IDS GPU_COUNT=$GPU_COUNT FSDP_SIZE=$FSDP_SIZE
attention=$ATTN_IMPL liger=$USE_LIGER
rollout temperature=$ROLLOUT_TEMPERATURE top_p=$ROLLOUT_TOP_P max_response_length=$MAX_RESPONSE_LENGTH
vllm backend=$VLLM_ATTENTION_BACKEND enforce_eager=$VLLM_ENFORCE_EAGER max_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS max_seqs=$VLLM_MAX_NUM_SEQS
filter_mixed_responses=$FILTER_MIXED_RESPONSES reward=exact_match full_batch_metrics=True
EOT
if [[ "$METHOD" == "gxpo" ]]; then
  echo "GXPO K=$K alpha=$REPOSITION_ALPHA optimizer_state=$GXPO_OPTIMIZER_STATE_MODE trigger=$GXPO_TRIGGER_SIGNAL"
fi

"$PYTHON_BIN" - "$ATTN_IMPL" <<'PY'
import importlib
import sys

import torch

required = ['pandas', 'wandb', 'tensordict']
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit('Missing runtime dependencies: ' + ', '.join(missing))

if not torch.cuda.is_available():
    raise SystemExit('No CUDA device visible')
major, minor = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f'[preflight] gpu={name} sm_{major}{minor} count={torch.cuda.device_count()}')
if (major, minor) == (9, 0):
    print('[preflight] WARNING: Hopper detected; the H200 script is faster here.')

attn = sys.argv[1]
if attn == 'flash_attention_3':
    raise SystemExit('flash_attention_3 has no sm_120 build; use flash_attention_2 or sdpa')
if attn == 'flash_attention_2':
    # find_spec is not enough: the wheel links against libcudart.so.13.
    try:
        importlib.import_module('flash_attn')
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            'flash_attention_2 was requested but "import flash_attn" failed: '
            f'{type(exc).__name__}: {exc}. Re-run with ATTN_IMPL=sdpa or fix '
            'LD_LIBRARY_PATH for libcudart.so.13.'
        )
elif attn not in {'sdpa', 'eager'}:
    raise SystemExit(f'Unsupported ATTN_IMPL for this profile: {attn}')
PY

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] preflight OK; no training launched"
  exit 0
fi


TRAIN_FILES="['$DAPO_TRAIN','$LIGHTEVAL_TRAIN']"
VAL_FILES="['$MATH500','$AIME24','$AIME25','$AMC23','$MINERVA','$OLYMPIAD']"
ARGS=(
  algorithm.adv_estimator=grpo
  +algorithm.norm_adv_by_std_in_grpo=False
  +algorithm.use_kl_in_reward=False
  "data.train_files=$TRAIN_FILES"
  "data.val_files=$VAL_FILES"
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  data.val_batch_size=128
  data.max_prompt_length=1024
  "data.max_response_length=$MAX_RESPONSE_LENGTH"
  data.filter_overlong_prompts=True
  data.truncation=error
  "+data.seed=$TRAIN_SEED"
  "data.system_prompt=$SYSTEM_PROMPT"
  "actor_rollout_ref.model.path=$MODEL_ID"
  actor_rollout_ref.model.use_remove_padding=True
  "actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING"
  "actor_rollout_ref.model.attn_implementation=$ATTN_IMPL"
  "+actor_rollout_ref.model.use_liger=$USE_LIGER"
  "actor_rollout_ref.actor.optim.lr=$LR"
  "+actor_rollout_ref.actor.optim.name=$OPTIMIZER_NAME"
  "+actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY"
  "+actor_rollout_ref.actor.optim.fused=False"
  "actor_rollout_ref.actor.use_torch_compile=$USE_TORCH_COMPILE"
  "+actor_rollout_ref.actor.data_loader_seed=$TRAIN_SEED"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU"
  actor_rollout_ref.actor.clip_ratio=0.2
  actor_rollout_ref.actor.grad_clip=1.0
  "actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF"
  "actor_rollout_ref.actor.use_kl_loss=$ACTOR_USE_KL_LOSS"
  "actor_rollout_ref.actor.kl_loss_coef=$ACTOR_KL_LOSS_COEF"
  "actor_rollout_ref.actor.kl_loss_type=$ACTOR_KL_LOSS_TYPE"
  "actor_rollout_ref.actor.fsdp_config.fsdp_size=$FSDP_SIZE"
  "actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD"
  "+actor_rollout_ref.actor.filter_mixed_responses=$FILTER_MIXED_RESPONSES"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE"
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.name=vllm
  "actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION"
  "actor_rollout_ref.rollout.max_num_batched_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS"
  "actor_rollout_ref.rollout.max_num_seqs=$VLLM_MAX_NUM_SEQS"
  "actor_rollout_ref.rollout.enable_chunked_prefill=$VLLM_ENABLE_CHUNKED_PREFILL"
  "actor_rollout_ref.rollout.attention_backend=$VLLM_ATTENTION_BACKEND"
  "actor_rollout_ref.rollout.enforce_eager=$VLLM_ENFORCE_EAGER"
  "actor_rollout_ref.rollout.free_cache_engine=$VLLM_FREE_CACHE_ENGINE"
  "actor_rollout_ref.rollout.n=$ROLLOUT_N"
  "actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE"
  "actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P"
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.val_kwargs.do_sample=False
  actor_rollout_ref.rollout.val_kwargs.temperature=0
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE"
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  reward_model.reward_manager=naive
  algorithm.kl_ctrl.kl_coef=0.000
  trainer.critic_warmup=0
  'trainer.logger=[console,wandb]'
  "trainer.project_name=$WANDB_PROJECT"
  "trainer.experiment_name=$RUN_NAME"
  "trainer.default_local_dir=$RUN_DIR"
  "trainer.n_gpus_per_node=$GPU_COUNT"
  trainer.nnodes=1
  "trainer.save_freq=$SAVE_FREQ"
  +trainer.keep_last_ckpts=1
  +trainer.keep_all_ckpts=False
  "trainer.resume_mode=$TRAINER_RESUME_MODE"
  "trainer.resume_from_path=$TRAINER_RESUME_FROM_PATH"
  "trainer.test_freq=$TRAINER_TEST_FREQ"
  '+trainer.validation_seeds=[0]'
  +trainer.keep_last_validations=1
  "+trainer.val_before_train=$VAL_BEFORE_TRAIN"
  "+trainer.max_steps=$MAX_STEPS"
  "trainer.total_training_steps=$MAX_STEPS"
  trainer.total_epochs=100
  +actor_rollout_ref.actor.use_gxpo=True
)
if [[ "$METHOD" == "gxpo" ]]; then
  ARGS+=(
    +actor_rollout_ref.actor.zscore_w=0
    "+actor_rollout_ref.actor.gxpo_k=$K"
    "+actor_rollout_ref.actor.gxpo_alpha=$REPOSITION_ALPHA"
    +actor_rollout_ref.actor.gxpo_delta=1e-8
    "+actor_rollout_ref.actor.gxpo_tau=$GXPO_TAU"
    "+actor_rollout_ref.actor.gxpo_zscore_w=$GXPO_ZSCORE_W"
    "+actor_rollout_ref.actor.gxpo_trigger_signal=$GXPO_TRIGGER_SIGNAL"
    "+actor_rollout_ref.actor.gxpo_trigger_patience=$GXPO_TRIGGER_PATIENCE"
    "+actor_rollout_ref.actor.gxpo_fallback_mode=$GXPO_FALLBACK_MODE"
    "+actor_rollout_ref.actor.gxpo_optimizer_state_mode=$GXPO_OPTIMIZER_STATE_MODE"
    "+actor_rollout_ref.actor.gxpo_fallback_window=$GXPO_FALLBACK_WINDOW"
    +actor_rollout_ref.actor.gxpo_trigger_granularity=outer
    "+actor_rollout_ref.actor.gxpo_warmup_steps=$GXPO_WARMUP_STEPS"
    "+actor_rollout_ref.actor.gxpo_reset_entropy_after_warmup=$GXPO_RESET_ENTROPY_AFTER_WARMUP"
    +actor_rollout_ref.actor.gxpo_omega=0.1
    "+actor_rollout_ref.actor.gxpo_shutoff_mode=$GXPO_SHUTOFF_MODE"
    +actor_rollout_ref.actor.gxpo_recompute_old_log_probs=False
    "+actor_rollout_ref.actor.gxpo_diag_freq=$GXPO_DIAG_FREQ"
  )
fi
"$PYTHON_BIN" -u -m verl.trainer.main_ppo "${ARGS[@]}" | tee "$RUN_DIR/train.log"
