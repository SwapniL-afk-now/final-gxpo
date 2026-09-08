#!/usr/bin/env bash
# SFT + reference-KL version of run_kd_sft_hybrid_dapo_lighteval_1p5b.sh: identical
# supervised objective on flat teacher responses (CE + kl_beta * D_KL(pi_theta ||
# pi_ref), verl/trainer/fsdp_sft_trainer.py), but each batch takes the GXPO
# 3-pass extrapolated update (probe g0 + step, probe g1 + step, reposition,
# slow correction + step) instead of one plain optimizer step. The KL term here
# is the plain RL-style anchor against a frozen reference policy -- not a
# distillation term: no teacher top-K cache is needed or used.
#
# Training data is the flat verified-correct teacher-response corpus
# (prompt + teacher_response + teacher_response_ids); the deleted top-K cache
# (dapo_lighteval_topk16_*) is no longer referenced anywhere in this file.
#
# Costs ~3x the plain script's wall time (3 forward+backward passes per batch,
# plus one frozen-reference forward per pass for the K3 KL term, which needs
# only the taken token's logprobs -- O(1) per token, no vocabulary sum). That
# 3x is an intrinsic property of the GXPO method, not something
# to optimize away; the levers below are everything that IS controllable:
#   - data.max_length=4096, not 16384 (KDSFTDataset pads every sample to a FIXED
#     max_length, so this is the single biggest wall-clock lever).
#   - micro_batch_size_per_gpu is the size of the single largest forward/backward
#     call (gradient accumulation runs train_batch_size/micro_batch_size_per_gpu
#     of these sequentially, discarding activations between them), so it -- not
#     train_batch_size -- sets peak memory. GXPO holds 4 extra full fp32
#     parameter-shaped buffers (theta0/g0/g1 + the slow-pass copy) plus an
#     AdamW-state transaction snapshot on top of the same forward, so it has
#     materially less headroom at a given micro_batch_size_per_gpu than plain
#     SFT does. Verify with a short dry run before trusting it at full scale.
# K=3, alpha=0.8 -- the knights-and-knaves operating point, where the effective
# displacement multiplier alpha*scale sits near 1.2 (a real extrapolation past
# theta2). See the GXPO_MIN_EFFECTIVE_MULTIPLIER block below for why K=5/alpha=0.3
# was wrong: it pinned the multiplier at ~0.75, so every "extrapolated" step
# actually landed short of the two probe steps it paid for.
#
# lr=1e-5 and kl_beta=0 (pure SFT). At lr=1e-6 with a kl_beta=0.1 anchor pointed at
# the student's own init, neither this arm nor the plain-SFT baseline learned at all
# -- train loss moved 7% over 400 steps and the six-benchmark average sat flat inside
# its own noise band, so there was no headroom for GXPO to accelerate into. kl_beta=0
# also skips building the frozen reference model entirely.
#
# The entropy shutoff gate is ARMED (tau=3.0, patience=3) and GXPO is budget-capped to
# the first 150 steps; z, trigger_stat and entropy are logged every step regardless.
#
# Evaluation: six-benchmark vLLM eval runs in-training after every 5 steps
# (trainer.benchmark_eval_freq=5, same eval_greedy/* wandb keys as the final
# post-eval) plus one post-training pass below.
#
# Two optimizer-state modes, selected by GXPO_OPTIMIZER_STATE_MODE (see the
# variable block below for the full contract). Both reposition to theta_tilde;
# they differ only in the AdamW state the slow correction starts from --
# `transactional` rolls the probe steps' moments back, `transactional_fast_state`
# keeps them. Each mode gets its own EXP/RUN_DIR/wandb run.
#
# Usage:
#   GPU=1 ./run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh                    # transactional
#   GPU=1 GXPO_OPTIMIZER_STATE_MODE=transactional_fast_state \
#     ./run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh                        # no optim refresh
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
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd):${PYTHONPATH:-}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"

GPU="${GPU:?set GPU=0|1}"
export GPU
TRAIN_SEED="${TRAIN_SEED:-42}"
LR="${LR:-1e-5}"
PROJECT="${PROJECT:-gxpo-efficiency-final}"
K="${K:-3}"
ALPHA="${ALPHA:-0.8}"
# Entropy gate, every knob explicit (no code defaults). The gate is live from
# step 0 (GXPO_WARMUP=0) over a 30-step rolling window, and falls back to plain
# single-pass SFT permanently once the entropy signal fires (tau=3.0 held 3
# consecutive batches). GXPO_MAX_ACTIVE_STEPS=150 is the hard backstop: GXPO
# runs at most the first 150 of 400 steps, then single-pass SFT for the rest.
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
GXPO_MAX_ACTIVE_STEPS="${GXPO_MAX_ACTIVE_STEPS:-150}"
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
# ceiling is 1.5, so alpha=0.8 gives ~1.2 -- the knights-and-knaves operating point.
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

MODEL="${MODEL:-/office/shared_cache/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
# Flat verified-correct teacher responses (prompt + teacher_response[_ids]).
# No top-K cache columns are required: the KL term is D_KL(pi_theta || pi_ref)
# against a frozen reference policy, not a distillation term.
FLAT_TRAIN="${FLAT_TRAIN:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO/data/teacher_responses/qwen25_math7b_dapo_lighteval_train_n1_correct_only.parquet}"
TRAIN="${TRAIN:-$FLAT_TRAIN}"
VAL="${VAL:-$TRAIN}"

# Reference policy for the KL anchor. Defaults to the student's own starting
# weights (the standard RL KL reference); point at a frozen teacher to anchor
# to the teacher instead.
KL_REF_MODEL="${KL_REF_MODEL:-$MODEL}"
# KL anchor weight. Tune on a smoke run: too high stalls learning on the
# teacher traces, too low lets the policy drift off them.
KL_BETA="${KL_BETA:-0}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-96}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
if (( TRAIN_BATCH_SIZE % MICRO_BATCH_SIZE != 0 )); then
  echo "PREFLIGHT FAIL: TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE) must be a multiple of MICRO_BATCH_SIZE ($MICRO_BATCH_SIZE)" >&2
  exit 2
fi

EXP="sftkl_gxpo_k${K}_a${ALPHA}_b${KL_BETA}_flat14k_lr${LR}_seed${TRAIN_SEED}${OPT_STATE_TAG}"
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
# Cadence evals generate ONE sampled response per prompt (temp 0.7, top_p 1.0,
# no greedy pass); the trainer reads back the matching sft_sampled_1_1seed.json
# via eval_sample_n/eval_seed_count/eval_skip_greedy.
export EVAL_N="${EVAL_N:-1}"
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

"$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    data.prompt_key=prompt \
    data.response_key=teacher_response \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
    data.max_length="$MAX_LENGTH" \
    data.truncation=error \
    data.normalize_by_sequence=True \
    data.kl_beta="$KL_BETA" \
    data.kl_ref_model="$KL_REF_MODEL" \
    model.partial_pretrain="$MODEL" \
    model.enable_gradient_checkpointing=True \
    ++model.fsdp_config.strategy=fsdp2 \
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
    +optim.gxpo_param_buffer_device="${GXPO_PARAM_BUFFER_DEVICE:-cpu}" \
    +optim.gxpo_min_effective_multiplier="$GXPO_MIN_EFFECTIVE_MULTIPLIER" \
    +optim.gxpo_omega=0.1 \
    trainer.project_name="$PROJECT" \
    trainer.experiment_name="$EXP" \
    trainer.default_local_dir="$RUN_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=3 \
    trainer.total_training_steps="${MAX_STEPS:-400}" \
    ++trainer.test_freq=0 \
    ++trainer.save_freq=0 \
    ++trainer.benchmark_eval_freq=5 \
    ++trainer.eval_sample_n=1 \
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
    --seed "${EVAL_SEED:-0}" --n 1 --temperature 0.7 --top-p 1.0 \
    --max-tokens "${EVAL_MAX_TOKENS:-3072}" --tp 1 --gpu-memory-utilization 0.85 \
    --max-model-len "${EVAL_MAX_MODEL_LEN:-4096}" --max-num-seqs 256 --max-num-batched-tokens 65536 \
    --attention-backend "$VLLM_ATTENTION_BACKEND" \
    --log-wandb --wandb-project "$PROJECT" --wandb-run "$EXP-post-eval" \
    --output "$RUN_DIR/eval_pass1_6bench.json" 2>&1 | tee "$RUN_DIR/eval_pass1_6bench.log"
fi
