#!/usr/bin/env bash
# SFT + GXPO-style update: identical to run_sft_baseline_stratos_1p5b.sh except
# each batch takes the GXPO 3-pass extrapolated update (probe g0 + optimizer
# step, probe g1 + optimizer step, reposition, slow correction + optimizer
# step = 3 optimizer steps per batch) applied to a plain cross-entropy
# objective -- no importance ratio, no advantages.
# K=3, alpha=0.1, tau=5.0, warmup=3.
#
# Costs ~3x the baseline's wall time (3 forward+backward passes per batch).
# NOTE: the loss path now trims each micro-batch to its real tokens rather than
# the padded data.max_length. Against a ~3.5k median sequence that is ~3.4x less
# compute per pass (measured), so a GXPO step is ~100s here, not ~226s. A
# baseline arm must run the same trimmed code for eff/step_time_s to compare.
# Config: Qwen2.5-1.5B-Instruct, Bespoke-Stratos-17k, batch 96, micro-batch 1,
# max_length 16384, lr 1e-5, liger kernels, 160 steps (1 epoch). GXPO
# 3-pass runs on every step: the shutoff gate is configured but never armed
# (tau=1e9, patience=2) -- see the GXPO_TAU block below for why. Memory: GXPO holds 4 extra full-model
# fp32 buffers (~24GB) plus the AdamW probe transaction (~12GB), which OOMs the
# 16k transient. FSDP `cpu_offload=True` used to paper over that, but it moves
# the whole backward and optimizer onto the CPU -- the GPU sat at 4% and a step
# took >20x the baseline. Instead the two buffers that are only touched in bulk
# (theta0/theta2) and the AdamW snapshot are staged in pinned host memory
# (gxpo_param_buffer_device=cpu) while every compute pass stays on the GPU;
# placement changes, not math. No
# mid-run benchmark evals or checkpoints; only the final global_step_160 HF
# checkpoint is kept (fsdp2 => no resumable training-state dir). Two
# unavoidable trainer behaviors, both disclosed: (a) cheap val-loss logging
# at the 3 epoch ends (val split required at startup; no generations, no
# checkpoints); (b) the trainer unconditionally runs its own repo-harness
# greedy eval once AT THE END (temp 0.0, 16k budget) and would delete the
# final checkpoint if that eval crashed -- it is wired to succeed so the
# checkpoint is retained for the real reporting below.
# Post-training pass@1 on all 6 benchmarks: temp 0.7, 1 response/prompt,
# 16k generation budget.
# Usage: GPU=1 ./run_sft_gxpo_stratos_1p5b.sh
set -euo pipefail

# /etc/environment ships RAY_ADDRESS="127.0.0.1" (no port), an invalid bootstrap address.
# Variable-length micro-batches (the loss path now trims each one to its real
# tokens instead of the padded 16384) fragment the caching allocator badly:
# 90.7 GiB reserved without this, 66.7 GiB with it, same step time.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_ADDRESS=local   # force an isolated Ray cluster per job -- unaddressed ray.init() auto-attaches to any existing local cluster (via /tmp/ray/session_latest), starving concurrent GPU0/GPU1 jobs of GPUs ("Total available GPUs 0")

GPU="${GPU:?set GPU=0|1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
LR=1e-5
PROJECT="${PROJECT:-gxpo-efficiency-final}"
K=3
ALPHA="${ALPHA:-0.1}"
# SHUTOFF DISABLED (tau=1e9), deliberately. The earlier tau=5.0 was calibrated on
# SFT *grad-norm* logs, but fsdp_sft_trainer feeds the gate `stat_override=disagreement`
# (= 1 - |cos(g0, g_slow)|), a different statistic on a different scale, so that
# calibration does not transfer. With trigger_robust=False there is no sigma floor
# (the floor exists only in the robust branch, added precisely because a near-constant
# window drives the scale to ~0 and manufactures huge z from tiny wobbles), and
# relative_threshold defaults to 0 while the abs_threshold path requires
# shutoff_mode='cosine' -- so the raw z-path is the ONLY live criterion, and
# fallback_mode defaults to 'permanent'.
#
# Measured on this exact config: disagreement ran 0.000, 0.055, 0.002, 0.002, 0.003
# over the first five healthy steps. Once the 30-observation window fills with
# ~0.002-0.003 (sigma ~5e-4), a repeat of that isolated 27x spike scores z ~ 105,
# i.e. 21x over tau=5.0 -> permanent shutoff on a healthy run.
#
# tau=1e9 makes check_trigger unreachable; trigger_patience=2 additionally requires a
# sustained excursion, so the gate is configured but never armed. z, trigger_stat and
# disagreement are still computed and logged every step, so what the gate WOULD have
# done stays fully auditable after the fact.
GXPO_TAU="${GXPO_TAU:-1e9}"
GXPO_TRIGGER_PATIENCE="${GXPO_TRIGGER_PATIENCE:-2}"
# The EMA gate starts cold (mu=1); step-0 z~30 for norms~30 would instantly, permanently
# shut off extrapolation. The EMA tracks within ~2 steps, so suppress the trigger for 3.
GXPO_WARMUP=3

MODEL="${MODEL:-/office/shared_cache/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
DATA_DIR="${DATA_DIR:-/office/dev_workspace/swapnil/data/Bespoke-Stratos-17k}"
TRAIN="${TRAIN:-$DATA_DIR/train.parquet}"
VAL="${VAL:-$DATA_DIR/val.parquet}"
# Symlink layout for the trainer's mandatory end-of-run harness eval
# (math500/test.parquet, aime2024/test.parquet, ...).
EVAL_ROOT="${EVAL_ROOT:-/office/dev_workspace/swapnil/data/eval_sft_6bench_terminal}"

# GXPO optimizer state across the two probe steps. Parameters are repositioned to
# theta_tilde in both modes; only the AdamW state the slow correction starts from differs:
#   transactional            -- snapshot the AdamW state before probe 1 and roll back to it
#                               after repositioning, so the slow correction is taken from
#                               the moments the batch started with (probes stay probes).
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

EXP="sft_gxpo_k${K}_a${ALPHA}_tau${GXPO_TAU}_stratos1p5b_b96_e3_lr1e-5_seed${TRAIN_SEED}${OPT_STATE_TAG}"
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

"${PYTHON_BIN:-python}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=96 \
    data.micro_batch_size_per_gpu="${SFT_MICRO_BATCH_SIZE:-1}" \
    data.max_length=16384 \
    data.truncation=right \
    model.partial_pretrain="$MODEL" \
    model.enable_gradient_checkpointing=True \
    ++model.fsdp_config.strategy=fsdp2 \
    model.use_liger=True \
    optim.lr=$LR \
    optim.betas="[0.9,0.999]" \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.0 \
    optim.clip_grad=1.0 \
    +optim.use_gxpo=True \
    +optim.gxpo_k=$K \
    +optim.gxpo_alpha=$ALPHA \
    +optim.gxpo_delta=1e-8 \
    +optim.gxpo_tau=$GXPO_TAU \
    +optim.gxpo_warmup=$GXPO_WARMUP \
    +optim.gxpo_trigger_patience=$GXPO_TRIGGER_PATIENCE \
    +optim.gxpo_optimizer_state_mode="$GXPO_OPTIMIZER_STATE_MODE" \
    +optim.gxpo_param_buffer_device="${GXPO_PARAM_BUFFER_DEVICE:-cpu}" \
    +optim.gxpo_omega=0.1 \
    +optim.gxpo_shutoff_mode=trajectory_aware \
    use_remove_padding=true \
    trainer.project_name="$PROJECT" \
    trainer.experiment_name="$EXP" \
    trainer.default_local_dir="$RUN_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=3 \
    trainer.total_training_steps=160 \
    ++trainer.test_freq=0 \
    ++trainer.save_freq=0 \
    ++trainer.greedy_eval_freq=0 \
    trainer.eval_kind=math \
    trainer.eval_benchmark_root="$EVAL_ROOT" \
    trainer.eval_greedy_max_new_tokens=16384 \
    +trainer.eval_greedy_max_examples=0 \
    +trainer.eval_greedy_prompt_max_length=2048 \
    trainer.eval_greedy_vllm_gpu_memory_utilization=0.6 \
    trainer.eval_greedy_vllm_max_num_batched_tokens=32768 \
    trainer.eval_greedy_vllm_max_num_seqs=16 \
    +trainer.eval_greedy_timeout_s=0 \
    trainer.seed=$TRAIN_SEED \
    2>&1 | tee "$RUN_DIR/train.log"

# Post-training pass@1 six-benchmark eval on the final checkpoint.
# SKIP_EVAL=1 to skip. temp 0.7, n=1, 16k generation budget with vLLM sized to
# match (max_model_len covers prompt headroom + 16k; 16 seqs keep 16k-KV in 96GB).
if [[ "${SKIP_EVAL:-0}" != 1 ]]; then
  CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  CKPT_ABS="$(ls -d "$RUN_DIR"/global_step_* | sort -t_ -k3 -n | tail -1)"
  CKPT_ABS="$(cd "$CKPT_ABS" && pwd)"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "${PYTHON_BIN:-python}" "$CODE_ROOT/tools/kd_sft/evaluate_greedy.py" \
    --checkpoint-dir "$CKPT_ABS" --base-model "$MODEL" \
    --data-files "$MATH500" "$AIME24" "$AIME25" "$AMC23" "$MINERVA" "$OLYMPIAD" \
    --seed "${EVAL_SEED:-0}" --n 1 --temperature 0.7 --top-p 1.0 \
    --max-tokens 16384 --tp 1 --gpu-memory-utilization 0.85 \
    --max-model-len 20480 --max-num-seqs 16 --max-num-batched-tokens 32768 \
    --output "$RUN_DIR/eval_pass1_6bench.json" 2>&1 | tee "$RUN_DIR/eval_pass1_6bench.log"
fi
