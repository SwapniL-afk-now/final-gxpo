#!/usr/bin/env bash
# Off-policy SFT knowledge distillation on DeepScaleR problem/solution pairs
# (CE + kl_beta * D_KL(p_teacher || p_student), verl/trainer/fsdp_sft_trainer.py),
# where each batch takes the GXPO 3-pass extrapolated update instead of one plain
# optimizer step. This is the GXPO arm of the pair; see
# run_kd_sft_hybrid_dapo_lighteval_1p5b.sh for the plain-SFT baseline.
#
# The KL is teacher-distribution KD, not the RL-style anchor this file used to
# run: the frozen model is the DeepScaleR-1.5B-Preview teacher rather than a copy
# of the student's init, and the divergence is a FULL forward KL over all 151,936
# logits at every supervised token (data.kl_full=True) rather than the K3
# one-token estimator of the reverse KL. The teacher only runs forward passes --
# no generation, no top-K cache. `solution` remains the hard CE target.
#
# Training data is cached by prep_32b_traces_kd.py: each upstream row of
# sam-12labs/DeepScaleR-Preview-Dataset_DeepSeek-R1-Distill-Qwen-32B_reasoning_traces
# holds one `problem` plus a LIST of 32B reasoning traces with a per-trace
# correctness flag, exploded into one row per CORRECT trace with `problem` as
# prompt and the trace as `solution`. Same-prompt rows are independent samples
# to the row-based SFTDataset. The validation holdout is at PROBLEM level
# (seed-42 shuffled).
#
# Costs ~3x the plain script's wall time (3 forward+backward passes per batch,
# plus one frozen-reference forward per pass for the K3 KL term, which needs
# only the taken token's logprobs -- O(1) per token, no vocabulary sum). That
# 3x is an intrinsic property of the GXPO method, not something
# to optimize away; the levers below are everything that IS controllable:
#   - data.max_token_len_per_gpu, NOT data.max_length. The dataset still pads
#     every sample to a fixed max_length, but data.use_dynamic_bsz=True now sorts
#     rows by real length and cuts micro-batches under a token budget, and the
#     trainer trims each micro-batch to its widest real row. Against a 3,172-token
#     p50 that is what makes max_length=16384 affordable at all, and it moves
#     the wall-clock lever from max_length to the token budget.
#   - the token budget is the size of the single largest forward/backward call
#     (gradient accumulation runs the resulting micro-batches sequentially,
#     discarding activations between them), so it -- not
#     train_batch_size -- sets peak memory. GXPO holds 4 extra full fp32
#     parameter-shaped buffers (theta0/g0/g1 + the slow-pass copy) plus an
#     AdamW-state transaction snapshot on top of the same forward, so it has
#     materially less headroom at a given micro_batch_size_per_gpu than plain
#     SFT does. Verify with a short dry run before trusting it at full scale.
# K=3, alpha=0.1 -- a deliberately conservative operating point, where the effective
# displacement multiplier alpha*scale stays below 1 (a conservative contraction toward
# theta2). See the GXPO_MIN_EFFECTIVE_MULTIPLIER block below for the contraction warning.
# was wrong: it pinned the multiplier at ~0.75, so every "extrapolated" step
# actually landed short of the two probe steps it paid for.
#
# The defaults match the ytr1pu3r SFT-KL baseline: lr=1e-6, kl_beta=0.1,
# max_length=2688, train_batch_size=256, and micro_batch_size_per_gpu=4.
# The frozen reference is the same Qwen2.5-1.5B-Instruct checkpoint.
#
# The entropy shutoff gate is ARMED (tau=3.0, patience=3) and GXPO is budget-capped to
# the first 100 steps; z, trigger_stat and entropy are logged every step regardless.
#
# Validation loss and the six-benchmark eval both run every 10 steps by default.
# The trainer launches one independent TP=1 vLLM worker per listed GPU; the final
# pass below uses the same n=4/temp=.7 settings after training.
#
# Multi-GPU works WITH in-training eval in this tree: every rank offloads to CPU,
# rank 0 runs the vLLM subprocess pinned to its own GPU, the other ranks resume
# onto theirs, and the barrier after it has a 10h timeout. (An older tree raised
# on world_size != 1; that guard is not here.) The pin only holds because the
# trainer narrows BOTH CUDA_VISIBLE_DEVICES and $GPU for the subprocess --
# eval_sft_6bench.sh re-exports CVD from $GPU and would otherwise start a second
# worker on rank 1's GPU.
#
# Two optimizer-state modes, selected by GXPO_OPTIMIZER_STATE_MODE (see the
# variable block below for the full contract). Both reposition to theta_tilde;
# they differ only in the AdamW state the slow correction starts from --
# `transactional` rolls the probe steps' moments back, `transactional_fast_state`
# keeps them. Each mode gets its own EXP/RUN_DIR/wandb run.
#
# Retention estimator, selected by GXPO_RETENTION_SPACE (see its variable block
# below). Default `auto` is the optimizer-aware AdamW rule r = d1/d0, read off
# the two real probe displacements; `grad` is the legacy raw-gradient r = g1/g0
# that every earlier run here used. The `auto` arm is tagged `_adamwdir` in
# EXP/RUN_DIR/wandb so the two never share a run identity.
#
# Usage:
#   GPU=1 ./run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh                    # transactional, adamwdir
#   GPU=1 GXPO_RETENTION_SPACE=grad \
#     ./run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh                        # legacy g1/g0 A/B arm
#   GPU=1 GXPO_OPTIMIZER_STATE_MODE=transactional_fast_state \
#     ./run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh                        # no optim refresh
#   GPU=0,1 MICRO_BATCH_SIZE=4 ./run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh  # 2-GPU FSDP
#
# Multi-GPU: pass a comma list (GPU=0,1). One rank per device; NCCL P2P/SHM are
# left ENABLED and the sharding strategy drops to shard_grad_op, because at
# 2 ranks FULL_SHARD's second all-gather and the crippled NCCL transport
# together cost more wall clock than the whole forward+backward.
#
# MICRO_BATCH_SIZE is worth raising with the extra headroom -- it sets the number
# of FSDP unshard/reshard cycles per pass, not just the activation footprint --
# and 4 is the configured value for 1.5B/len2688 on 2x H200 here. MEASURED: mbs=8 with
# shard_grad_op OOMs at 91.4GB allocated (shard_grad_op keeps the unsharded
# parameters resident through backward, on top of GXPO's three model-sized fp32
# buffers and the AdamW-state transaction snapshot). If you raise it further,
# drop back to FSDP_SHARDING_STRATEGY=full_shard.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_ADDRESS=local

# Bind every launcher to the standalone target environment. Do not inherit a
# deleted/stale venv, CUDA toolkit, or FlashInfer backend from the caller.
VENV_ROOT="${VENV_ROOT:-/office/dev_workspace/swapnil/final-gxpo-h200/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_ROOT/bin/python}"
export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$VENV_ROOT/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-$VENV_ROOT/lib/python3.12/site-packages/nvidia/cu13}"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
GPU="${GPU:?set GPU=0|1, or a comma list such as GPU=0,1 for multi-GPU FSDP}"
export GPU
# One training rank per listed device. GPU=1 -> NPROC=1 (unchanged); GPU=0,1 -> 2.
NPROC="$(awk -F, '{print NF}' <<< "$GPU")"

# NCCL transport. P2P+SHM off is a vLLM-worker workaround and costs nothing
# when training runs on a single rank, but it is catastrophic for multi-rank
# FSDP: measured on this box (4x H200, PCIe/PHB, no NVLink) a 256MB/rank
# all_gather runs at 2.4 GB/s with both disabled and 9.6 GB/s with both
# enabled -- a 4x tax on every collective. GXPO pays that bill three times per
# batch (three forward+backward passes), so the defaults below follow the rank
# count: crippled only where it is free.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-$(( NPROC > 1 ? 0 : 1 ))}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-$(( NPROC > 1 ? 0 : 1 ))}"

TRAIN_SEED="${TRAIN_SEED:-42}"
LR="${LR:-5e-6}"
PROJECT="${PROJECT:-gxpo-efficiency-final}"
K="${K:-3}"
ALPHA="${ALPHA:-0.1}"
# Entropy gate, every knob explicit (no code defaults). The gate is live from
# step 0 (GXPO_WARMUP=0) over a 30-step rolling window, and falls back to plain
# single-pass SFT permanently once the entropy signal fires (tau=3.0 held 3
# consecutive batches). GXPO_MAX_ACTIVE_STEPS=100 is the hard backstop: GXPO
# runs at most the first 100 of 400 steps, then single-pass SFT for the rest.
# This is the knights-and-knaves duty cycle, where GXPO on the first ~1/3 of
# training beat both the always-on variant and the plain-SFT baseline. Note a
# warmup >= the budget would make the gate unreachable -- keep them ordered.
GXPO_TRIGGER_SIGNAL="${GXPO_TRIGGER_SIGNAL:-entropy}"
GXPO_SHUTOFF_MODE="${GXPO_SHUTOFF_MODE:-trajectory_aware}"
GXPO_TAU="${GXPO_TAU:-3.0}"
GXPO_TRIGGER_PATIENCE="${GXPO_TRIGGER_PATIENCE:-3}"
GXPO_WARMUP="${GXPO_WARMUP:-0}"
GXPO_TRIGGER_ROBUST="${GXPO_TRIGGER_ROBUST:-0}"
GXPO_TRIGGER_SUSTAIN_W="${GXPO_TRIGGER_SUSTAIN_W:-10}"
GXPO_ZSCORE_W="${GXPO_ZSCORE_W:-30}"
GXPO_TRIGGER_MIN_OBS="${GXPO_TRIGGER_MIN_OBS:-0}"
GXPO_TRIGGER_ABS_THRESHOLD="${GXPO_TRIGGER_ABS_THRESHOLD:-0}"
GXPO_MAX_ACTIVE_STEPS="${GXPO_MAX_ACTIVE_STEPS:-100}"
GXPO_RELATIVE_THRESHOLD="${GXPO_RELATIVE_THRESHOLD:-0}"
GXPO_FALLBACK_MODE="${GXPO_FALLBACK_MODE:-permanent}"
GXPO_FALLBACK_WINDOW="${GXPO_FALLBACK_WINDOW:-10}"

# Contraction guard. theta_tilde = theta0 + alpha*scale*(theta2-theta0), so the
# EFFECTIVE multiplier is alpha*scale, not alpha: above 1 the reposition lands past
# theta2 (the extrapolation GXPO exists for), below 1 it lands short and the 3-pass
# update makes less progress than the two probe steps it already paid for. The
# trainer logs train/gxpo_effective_multiplier and train/gxpo_contracting every step
# and prints a one-shot warning the first time it drops below 1. Set this to 1.0 to
# clamp the per-coordinate multiplier instead of only warning; 0 leaves it warn-only.
#
# scale is bounded to [1, K/2+1] and tends to K/2 as the retention ratio r -> 1, which
# is what happens when probe steps are tiny (small lr, strong KL anchor). At K=3 that
# ceiling is 1.5, so alpha=0.1 gives a conservative multiplier -- the knights-and-knaves operating point.
# The earlier K=5/alpha=0.3 pair gave 0.3*2.5 = 0.75 and silently contracted every step.
GXPO_MIN_EFFECTIVE_MULTIPLIER="${GXPO_MIN_EFFECTIVE_MULTIPLIER:-0}"

# Optimizer-state handling across the two probe steps. Parameters are repositioned to
# theta_tilde in both modes; only the AdamW state the slow correction starts from differs:
#   transactional            -- snapshot the AdamW state x before probe 1 and roll back to x
#                               after repositioning, so the slow correction is taken from the
#                               moments the batch started with (the probes stay pure probes).
#   transactional_fast_state -- no refresh: the probe steps' moments and step counter are
#                               kept and the slow correction is taken from them, so AdamW's
#                               step counter advances 3x per batch instead of 1x.
# The mode is tagged into the run name so the two arms never share a run dir or wandb run;
# transactional is left untagged because it is the established baseline.
GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"
case "$GXPO_OPTIMIZER_STATE_MODE" in
  transactional)            OPT_STATE_TAG="" ;;
  transactional_fast_state) OPT_STATE_TAG="_optkeep" ;;
  *) echo "PREFLIGHT FAIL: GXPO_OPTIMIZER_STATE_MODE must be transactional or transactional_fast_state, got '$GXPO_OPTIMIZER_STATE_MODE'" >&2; exit 2 ;;
esac

# Which space the retention ratio is measured in. GXPO models OPTIMIZER-induced
# motion, and this trainer's optimizer is always AdamW -- which does NOT move the
# parameters along the gradient, it moves them along m_hat/(sqrt(v_hat)+eps).
#   auto -- optimizer-aware: r = d1/d0, the ratio of the two adaptive directions
#           AdamW actually applied, reconstructed from the real probe
#           displacements as d_t = ((1 - lr*wd)*theta_t - theta_{t+1})/lr. That
#           inversion carries AdamW's moments, bias correction, epsilon
#           convention and the CLIPPED gradient it really consumed.
#   grad -- legacy raw-gradient r = g1/g0. This is what every pre-existing
#           GXPO-SFT/KD run used, and it is the A/B control arm.
# 'update' is the per-matrix Muon estimator and is rejected here: there is no
# Muon-owned matrix in an AdamW SFT run. See GXPO_OPTIMIZER_AWARE_RETENTION.md.
# The trainer itself still defaults to grad, so this launcher is the explicit
# opt-in; RETENTION_TAG below keeps the two arms on separate run identities.
GXPO_RETENTION_SPACE="${GXPO_RETENTION_SPACE:-auto}"
case "$GXPO_RETENTION_SPACE" in
  auto)   RETENTION_TAG="_adamwdir" ;;
  grad)   RETENTION_TAG="" ;;
  update) echo "PREFLIGHT FAIL: GXPO_RETENTION_SPACE=update is the Muon per-matrix estimator and is not supported by the AdamW SFT/KD trainer; use auto or grad" >&2; exit 2 ;;
  *) echo "PREFLIGHT FAIL: GXPO_RETENTION_SPACE must be auto or grad, got '$GXPO_RETENTION_SPACE'" >&2; exit 2 ;;
esac

MODEL="${MODEL:-/office/shared_cache/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
export EVAL_BASE_MODEL="$MODEL"
MAX_LENGTH="${MAX_LENGTH:-16384}"
DATASET_ID="${DATASET_ID:-sam-12labs/DeepScaleR-Preview-Dataset_DeepSeek-R1-Distill-Qwen-32B_reasoning_traces}"
DATASET_DIR="${DATASET_DIR:-$CODE_ROOT/data/r1_32b_traces}"
VAL_SIZE="${VAL_SIZE:-1000}"
DATA_SEED="${DATA_SEED:-42}"
TRAIN="${TRAIN:-$DATASET_DIR/train_32b_correct_max${MAX_LENGTH}.parquet}"
VAL="${VAL:-$DATASET_DIR/val_32b_correct_max${MAX_LENGTH}_${VAL_SIZE}.parquet}"
if [[ ! -f "$TRAIN" || ! -f "$VAL" ]]; then
  mkdir -p "$DATASET_DIR"
  "$PYTHON_BIN" "$(dirname "${BASH_SOURCE[0]}")/prep_32b_traces_kd.py" \
    "$DATASET_ID" "$DATASET_DIR" "$MODEL" "$MAX_LENGTH" "$VAL_SIZE" "$DATA_SEED"
fi

# The distillation teacher. DeepScaleR-1.5B-Preview shares the student's Qwen2
# architecture and its 151,936 vocabulary, so one tokenization serves both and
# the trainer's vocab_size preflight passes. The trainer builds it from ITS OWN
# config, not the student's -- the two disagree on rope_theta (1e4 vs 1e6) and
# tie_word_embeddings (False vs True), and the student config would silently
# drop the teacher's untied lm_head.
KL_REF_MODEL="${KL_REF_MODEL:-/office/dev_workspace/swapnil/final-gxpo/models/DeepScaleR-1.5B-Preview}"
# Distillation weight: loss = CE + KL_BETA * KL(teacher || student). 1.0 is plain
# CE+KL. NOT comparable to the 0.1 the old K3 anchor used -- that anchor started
# at exactly 0 (reference == student init) while a full teacher KL starts at
# several nats/token. Check train/kl on a smoke run before a long one.
KL_BETA="${KL_BETA:-1.0}"
# Token budget per micro-batch. The binding constraint is the [tokens, 151936]
# logits tensor in the CE path (~3.7GB bf16 at 12288) plus ~1.5GB for the fp32
# KL chunk. Must stay above the longest real sequence -- the trainer RAISES if
# one row exceeds the budget -- and the 32B correct traces run long: longest
# 10,425, p99 7,575, p50 3,172 student tokens (prompt + trace + eos), so 8192
# is too small and 12288 is the setting. GXPO
# holds three extra model-sized fp32 buffers plus an AdamW snapshot on top of the
# same forward, so it has less headroom here than the plain arm at a given budget.
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-12288}"
# FSDP sharding strategy. FULL_SHARD (ZeRO-3) all-gathers every flat parameter
# in forward and AGAIN in backward; SHARD_GRAD_OP (ZeRO-2) holds the forward
# all-gather through backward, halving parameter traffic for one extra
# unsharded bf16 copy of the model (~3GB for 1.5B). On a PCIe-only box with a
# 1.5B model that is the right trade -- FULL_SHARD is saving ~1.5GB/rank of
# parameters while spending tens of seconds per step to do it. Single-rank runs
# keep full_shard so nothing about the existing GPU=<n> arm changes.
if (( NPROC > 1 )); then
  FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY:-shard_grad_op}"
else
  FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY:-full_shard}"
fi
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
# TRAIN_BATCH_SIZE is global: verl divides it by the data-parallel rank count
# first. It must still divide evenly across ranks, but the per-rank split into
# micro-batches is now by token budget rather than by MICRO_BATCH_SIZE rows, so
# the second divisibility requirement no longer applies.
if (( TRAIN_BATCH_SIZE % NPROC != 0 )); then
  echo "PREFLIGHT FAIL: TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE) must be a multiple of NPROC ($NPROC)" >&2
  exit 2
fi

# RETENTION_TAG must reach the run name: 'auto' is a different algorithm from the
# raw-gradient runs already sitting in ./runs, and an untagged name would let this
# arm reuse their directory and wandb history.
# The rank count changes throughput, memory and the per-rank batch, so a 1-GPU
# and a 2-GPU run of otherwise identical config are different runs and must not
# share a wandb id or checkpoint directory. Single-GPU keeps the untagged name
# every existing run already uses.
NPROC_TAG=""
if (( NPROC > 1 )); then
  NPROC_TAG="_${NPROC}gpu"
fi
EXP="${EXP:-sftkd_gxpo_fullkl_k${K}_a${ALPHA}_b${KL_BETA}_r1_32b_b${TRAIN_BATCH_SIZE}_lr${LR}_len${MAX_LENGTH}_seed${TRAIN_SEED}${OPT_STATE_TAG}${RETENTION_TAG}${NPROC_TAG}}"
RUN_DIR="${RUN_DIR:-./runs/${EXP}}"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_PROJECT="$PROJECT"
export WANDB_MODE=online
export WANDB_DIR="$RUN_DIR"
export MATH500="${MATH500:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO/data/eval_sft_6bench/math500.parquet}"
export AIME24="${AIME24:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO/data/eval_sft_6bench/aime24.parquet}"
export AIME25="${AIME25:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO/data/eval_sft_6bench/aime25.parquet}"
export AMC23="${AMC23:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO/data/eval_sft_6bench/amc23.parquet}"
export MINERVA="${MINERVA:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO/data/eval_sft_6bench/minervamath.parquet}"
export OLYMPIAD="${OLYMPIAD:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO/data/eval_sft_6bench/olympiadbench.parquet}"
# In-training six-benchmark eval reads the eval_sft_6bench layout (present);
# the legacy dapo layout it falls back to does not exist in this checkout.
export EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO/data/eval_sft_6bench}"
# The harness resolves its own location relative to the caller's cwd, which is
# wrong when the trainer launches it -- pin the SFPO root explicitly.
export EVAL_CODE_ROOT="${EVAL_CODE_ROOT:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO}"
# In-training eval shares the GPU with the suspended training state (~29GB
# held: fp32 params + AdamW + grads + bf16 ref), so its vLLM budget must fit
# alongside -- 0.6 (~58GB) fits, the default 0.85 does not. The post-training
# eval below runs after the trainer exits and keeps 0.85 on the free GPU.
export SFT_VLLM_EVAL_GPU_UTIL="${SFT_VLLM_EVAL_GPU_UTIL:-0.6}"
# Cadence evals generate four sampled responses per prompt (temp 0.7, top_p 1.0,
# no greedy pass); the trainer reads the merged multi-GPU result directly.
# via eval_sample_n/eval_seed_count/eval_skip_greedy.
export EVAL_N="${EVAL_N:-4}"
export EVAL_TOP_P="${EVAL_TOP_P:-1.0}"
export EVAL_SEEDS="${EVAL_SEEDS:-0}"
export SFT_VLLM_EVAL_SKIP_GREEDY="${SFT_VLLM_EVAL_SKIP_GREEDY:-1}"

for f in "$TRAIN" "$VAL" "$KL_REF_MODEL"; do
  [[ -f "$f" || -d "$f" ]] || { echo "PREFLIGHT FAIL: missing $f" >&2; exit 2; }
done

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  for f in "$MATH500" "$AIME24" "$AIME25" "$AMC23" "$MINERVA" "$OLYMPIAD"; do
    [[ -f "$f" ]] || { echo "PREFLIGHT FAIL: missing six-benchmark file $f (set SKIP_EVAL=1 to train without evaluation)" >&2; exit 2; }
  done
fi

"$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    data.prompt_key=problem \
    data.response_key=solution \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
    data.max_length="$MAX_LENGTH" \
    data.truncation=error \
    data.normalize_by_sequence=True \
    data.kl_beta="$KL_BETA" \
    data.kl_ref_model="$KL_REF_MODEL" \
    data.kl_full=True \
    ++data.kl_full_chunk_tokens="${KL_CHUNK_TOKENS:-2048}" \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
    ++data.val_micro_batch_size_per_gpu="${VAL_MICRO_BATCH_SIZE:-2}" \
    model.partial_pretrain="$MODEL" \
    model.enable_gradient_checkpointing="${GRAD_CKPT:-True}" \
    ++model.fsdp_config.sharding_strategy="$FSDP_SHARDING_STRATEGY" \
    model.use_liger=True \
    optim.lr="$LR" \
    optim.betas="[0.9,0.999]" \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.0 \
    optim.clip_grad=1.0 \
    +optim.use_gxpo=True \
    +optim.gxpo_k="$K" \
    +optim.gxpo_alpha="$ALPHA" \
    +optim.gxpo_delta=1e-8 \
    +optim.gxpo_trigger_signal="$GXPO_TRIGGER_SIGNAL" \
    +optim.gxpo_shutoff_mode="$GXPO_SHUTOFF_MODE" \
    +optim.gxpo_tau="$GXPO_TAU" \
    +optim.gxpo_trigger_patience="$GXPO_TRIGGER_PATIENCE" \
    +optim.gxpo_warmup="$GXPO_WARMUP" \
    +optim.gxpo_trigger_robust="$GXPO_TRIGGER_ROBUST" \
    +optim.gxpo_trigger_sustain_w="$GXPO_TRIGGER_SUSTAIN_W" \
    +optim.gxpo_zscore_w="$GXPO_ZSCORE_W" \
    +optim.gxpo_trigger_min_obs="$GXPO_TRIGGER_MIN_OBS" \
    +optim.gxpo_trigger_abs_threshold="$GXPO_TRIGGER_ABS_THRESHOLD" \
    +optim.gxpo_max_active_steps="$GXPO_MAX_ACTIVE_STEPS" \
    +optim.gxpo_relative_threshold="$GXPO_RELATIVE_THRESHOLD" \
    +optim.gxpo_fallback_mode="$GXPO_FALLBACK_MODE" \
    +optim.gxpo_fallback_window="$GXPO_FALLBACK_WINDOW" \
    +optim.gxpo_optimizer_state_mode="$GXPO_OPTIMIZER_STATE_MODE" \
    +optim.gxpo_retention_space="$GXPO_RETENTION_SPACE" \
    +optim.gxpo_param_buffer_device="${GXPO_PARAM_BUFFER_DEVICE:-cpu}" \
    +optim.gxpo_min_effective_multiplier="$GXPO_MIN_EFFECTIVE_MULTIPLIER" \
    +optim.gxpo_omega=0.1 \
    trainer.project_name="$PROJECT" \
    trainer.experiment_name="$EXP" \
    trainer.default_local_dir="$RUN_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs="${TOTAL_EPOCHS:-25}" \
    trainer.total_training_steps="${MAX_STEPS:-300}" \
    ++trainer.test_freq="${TEST_FREQ:-25}" \
    ++trainer.save_freq="${SAVE_FREQ:-0}" \
    ++trainer.benchmark_eval_freq="${BENCHMARK_EVAL_FREQ:-50}" \
    ++trainer.eval_sample_n="$EVAL_N" \
    ++trainer.eval_seed_count=1 \
    ++trainer.eval_skip_greedy=1 \
    trainer.seed="$TRAIN_SEED" \
    2>&1 | tee "$RUN_DIR/train.log"

if [[ "${SKIP_EVAL:-0}" != 1 ]]; then
  CODE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
  CKPT_ABS="$(ls -d "$RUN_DIR"/global_step_* | sort -t_ -k3 -n | tail -1)"
  CKPT_ABS="$(cd "$CKPT_ABS" && pwd)"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$CODE_ROOT/tools/kd_sft/evaluate_greedy.py" \
    --checkpoint-dir "$CKPT_ABS" --base-model "$MODEL" \
    --data-files "$MATH500" "$AIME24" "$AIME25" "$AMC23" "$MINERVA" "$OLYMPIAD" \
    --seed "${EVAL_SEED:-0}" --n "${EVAL_FINAL_N:-4}" --temperature 0.7 --top-p 1.0 \
    --max-tokens "${EVAL_MAX_TOKENS:-3072}" --tp 1 --gpu-devices "$GPU" --gpu-memory-utilization 0.85 \
    --max-model-len "${EVAL_MAX_MODEL_LEN:-4096}" --max-num-seqs 256 --max-num-batched-tokens 65536 \
    --attention-backend "$VLLM_ATTENTION_BACKEND" \
    --log-wandb --wandb-project "$PROJECT" --wandb-run "$EXP-post-eval" \
    --output "$RUN_DIR/eval_pass4_6bench.json" 2>&1 | tee "$RUN_DIR/eval_pass4_6bench.log"
  rm -rf "$CKPT_ABS"
fi
