#!/usr/bin/env bash
#
# launch_opd2_paper_pair.sh
#
# The OPD^2 A/B pair at the paper's recipe, side by side in two tmux sessions:
#   opd2_only  GPU 0  -- OPD^2 advantage, plain AdamW  (qwen25_1p5b_opd2_nogxpo_control.sh)
#   opd2_gxpo  GPU 1  -- OPD^2 advantage + GXPO        (qwen25_1p5b_opd2_gxpo_adamw_dir_k10.sh)
#
# GPUs 3 and 4 hold a FOREIGN run -- never touch them.
#
# Every recipe value below is transcribed from
# on-policy-delta/opd2/recipes/Qwen3-1.7B/opd2/config_open_nvidia_100k.yaml.
# Models (set in the arm launchers): Qwen2.5-1.5B-Instruct student,
# Qwen2.5-Math-1.5B-Instruct teacher, Qwen2.5-Math-1.5B teacher_base -- one
# shared tokenizer, short-CoT answers. The paper's Qwen3-1.7B / Qwen3-4B trio ran
# ~4000-token responses (30%+ truncated at 8192) at ~18h per arm.
# The earlier Qwen2.5-1.5B-Instruct student with a DeepScaleR - R1-Distill delta
# (two long-CoT <think> models) was pushed into R1-style reasoning it could not
# finish: every response hit 8192 by step 13 (gxpo-opd2 l4zbui0k / 4rwyfa3w).
#
#   per_device_train_batch_size 2 x grad_accum 16 x 8 GPUs = 256   -> TRAIN_BATCH_SIZE=256
#   ...and that is ONE optimizer step per batch in trl              -> PPO_MINI_BATCH_SIZE=256
#   num_generations 1                                              -> ROLLOUT_N=1
#   periodic validation decodes 1 response/prompt                    -> VAL_N=1
#   max_completion_length 8192 / max_prompt_length 1024            -> MAX_RESPONSE_LENGTH=3072
#     (DEVIATION: Qwen2.5-Math has 4096 positions, so 1024 + 3072 is the hard
#      ceiling; common.sh's preflight enforces it)
#   learning_rate 5.0e-06                                          -> LR=5e-6
#   lr_scheduler cosine_with_min_lr, min_lr_rate 0.1, warmup 0.1   -> LR_WARMUP_STYLE/RATIO/MIN
#   temperature 0.7                                                -> ROLLOUT_TEMPERATURE=0.7
#   opd2_rewards_top_k 1024                                        -> OPD2_TOPK=1024
#   opd_gen_loss_weight 0.1  /  opd_rewards_bias 0.0               -> OPD2_GEN_LOSS_WEIGHT / _BIAS
#   beta 0.0 (no reference KL)                                     -> USE_KL_LOSS=False, coef 0
#   max_steps 100  /  seed 42                                      -> MAX_STEPS=100, TRAIN_SEED=42
#   loss_type grpo (per-sequence token-mean)                       -> LOSS_AGG_MODE=seq-mean-token-mean
#   epsilon 0.2 / max_grad_norm 1.0 / gradient_checkpointing true  -> common.sh defaults
#   weight_decay unset -> HF default 0.0                           -> ADAMW_WEIGHT_DECAY=0.0
#   scale_rewards true                                             -> inert: the reference
#                                       hardcodes disable_adv_norm, advantages = raw signal
#
# THROUGHPUT knobs below are implementation details, NOT recipe values -- they do
# not change a single number the algorithm sees (verified bit-exact by
# tools/opd2/verify_signal.py). Tune them freely.
#
# Usage:
#   bash experiments/gxpo_efficiency/launch_opd2_paper_pair.sh          # both
#   bash experiments/gxpo_efficiency/launch_opd2_paper_pair.sh --only   # OPD^2-only arm
#   bash experiments/gxpo_efficiency/launch_opd2_paper_pair.sh --gxpo   # OPD^2+GXPO arm
#   tmux attach -t opd2_only   /   tmux attach -t opd2_gxpo
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

WHICH="${1:---both}"

PAPER=(
  # ---- recipe: change nothing here without changing the paper ----
  TRAIN_BATCH_SIZE=256
  PPO_MINI_BATCH_SIZE=256
  ROLLOUT_N=1
  VAL_N=1
  MAX_RESPONSE_LENGTH=3072
  MAX_STEPS=100
  TRAIN_SEED=42
  LR=5e-6
  LR_WARMUP_STYLE=cosine
  LR_WARMUP_RATIO=0.1
  LR_MIN_RATIO=0.1
  ADAMW_WEIGHT_DECAY=0.0
  ROLLOUT_TEMPERATURE=0.7
  ROLLOUT_TOP_P=1.0
  OPD2_TOPK=1024
  OPD2_GEN_LOSS_WEIGHT=0.1
  OPD2_REWARDS_BIAS=0.0
  # The reference loss has no entropy term; common.sh refuses nonzero under OPD^2.
  ENTROPY_COEFF=0
  # loss_type: grpo  -> mean within each sequence, then across sequences.
  # verl's historical reduction is a single token-mean over the whole batch,
  # which weights long responses more; at an 8192-token budget with widely
  # varying response lengths the two objectives are materially different
  # (tools/opd2/test_loss_agg.py measures the gap). Every non-OPD^2 arm keeps
  # token-mean -- this opts in only here. Run names get an _seqmean tag.
  LOSS_AGG_MODE=seq-mean-token-mean

  # ---- throughput: implementation only, algorithm-neutral ----
  # Score several rows per teacher forward instead of one. Rows are length-sorted
  # and right-padded, which is exact under causal attention; the reference's
  # one-row loop is an artifact, not a hyperparameter. 4 x [b, 8192, V] bf16 is
  # ~10GB transient -- lower this first on OOM.
  # 8 x 3072 tokens is below the 4 x 8192 peak that ran fine.
  OPD2_MICRO_BATCH_SIZE=8
  # Bigger fp32 log_softmax chunk = fewer kernel launches. Peak is chunk*V*4B.
  OPD2_CHUNK_TOKENS=1024
  # False = teacher + teacher_base are loaded disk->GPU for each scoring phase and
  # freed after, so they cost no host RAM (parking two 4B models on CPU pinned
  # ~16GB) and no VRAM during rollout. True keeps ~16GB resident on the GPU.
  OPD2_KEEP_ON_GPU=False
  # Low host-RAM footprint: each dataloader worker copies the dataset (~0.8GB).
  DATALOADER_NUM_WORKERS=2
  # Each reward worker is a spawned interpreter that re-imports torch (~0.7GB);
  # the default 16 held ~12GB. Math verification is milliseconds, so 4 is plenty.
  REWARD_NUM_WORKERS=4
  # Teacher + teacher_base (2 x 1.5B) share the arm's one GPU with training;
  # a dedicated scorer GPU would sit mostly idle.
  OPD2_DEDICATED_GPU=False
  # Cap Ray's shared-memory object store (default ~30% of RAM). OPD^2 streams
  # ~25-130MB per scoring micro-batch through it and frees each one.
  RAY_OBJECT_STORE_MEMORY_GB=8
  VLLM_MAX_NUM_SEQS=256
  VLLM_MAX_NUM_BATCHED_TOKENS=98304
  VLLM_ENABLE_CHUNKED_PREFILL=True

  # ---- evaluation ----
  TRAINER_TEST_FREQ=5
  VAL_BEFORE_TRAIN=True
  SAVE_FREQ=10
  FINAL_EVAL_ENABLED=False
  WANDB_PROJECT=gxpo-opd2
)

# Layout: 1 GPU per arm, both arms at once on identical hardware, so the
# wall-clock A/B stays fair (Qwen3 measured 222s/step on 1 GPU vs 181s on 3).
ONLY_GPU="${OPD2_ONLY_GPU:-0}"
GXPO_GPU="${OPD2_GXPO_GPU:-1}"
arm_cmd () {  # arm_cmd <log name> <gpu> <script> <extra env...>
  local name="$1" gpu="$2" script="$3"; shift 3
  echo "env ${PAPER[*]} GPU_IDS=$gpu GPU_COUNT=1 FSDP_SIZE=1 $* ${OPD2_EXTRA_ENV:-} \
bash experiments/gxpo_efficiency/$script 2>&1 | tee runs_opd2_$name.log"
}

launch () {  # launch <session> <command>
  local session="$1"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session '$session' already exists - kill it first: tmux kill-session -t $session" >&2
    return 1
  fi
  echo "[launch] $session"
  tmux new-session -d -s "$session" -c "$CODE_ROOT" "$2"
}

# No GXPO buffers competing for VRAM, so vLLM gets the larger KV cache.
ONLY_CMD="$(arm_cmd opd2_only "$ONLY_GPU" qwen25_1p5b_opd2_nogxpo_control.sh \
  VLLM_GPU_MEMORY_UTILIZATION=0.55)"

  # GXPO keeps three model-sized fp32 buffers resident per rank on top of
  # params/grads/AdamW state, so vLLM gets less.
  # Three runs died at step 7-8 at k10/a0.3, k5/a0.5 and k3/a0.8 alike, so K and
  # alpha were never the controlling variable. What actually happened (2026-09-10):
  # OPD^2's advantages are unnormalized and tiny (advantage_std ~0.014, pg_loss
  # ~0.000), so entropy_loss*entropy_coeff was the ONLY term with magnitude in
  # policy_loss = pg_loss - entropy_loss*entropy_coeff. The objective was, in
  # effect, "maximize entropy": entropy climbed 0.14->0.53, the policy went
  # diffuse, stopped emitting EOS, response_length ran 992->1979->7604 into the
  # 8192 cap and ppo_kl hit 0.463. ENTROPY_COEFF=0.01 was the accelerant; the
  # reference (config_open_nvidia_100k.yaml) uses no entropy bonus at all.
  #
  # 0.001 was tried next and failed the same way, only slower (entropy 0.07 ->
  # 3.9 by step 25 once the looping student zeroed the OPD^2 signal), so both
  # arms now take ENTROPY_COEFF=0 from PAPER above.
  #
  # MIN_EFFECTIVE_MULTIPLIER=0 is REQUIRED at alpha=0.1: scale is clamped to
  # [1, K/2+1], so alpha*scale <= 0.25 and the old 1.0 floor would round every
  # coordinate back up to 1.0 -- alpha would do nothing. 0 also lets GXPO damp a
  # step when retention says the direction isn't persisting, so expect (and
  # ignore) the "contracting, not extrapolating" warning from dp_actor.py.
  #
  # WARMUP_STEPS=3, not 12: warmup gates the instability SHUTOFF, and at 12 the
  # safety net could not arm until after every one of those runs was already dead
  # (gxpo/trigger_warmup_active was still 1 at step 8).
GXPO_CMD="$(arm_cmd opd2_gxpo "$GXPO_GPU" qwen25_1p5b_opd2_gxpo_adamw_dir_k10.sh \
  VLLM_GPU_MEMORY_UTILIZATION=0.45 \
  K=3 REPOSITION_ALPHA=0.1 \
  GXPO_MIN_EFFECTIVE_MULTIPLIER=0 \
  GXPO_WARMUP_STEPS=3)"

case "$WHICH" in
  --only) launch opd2_only "$ONLY_CMD" ;;
  --gxpo) launch opd2_gxpo "$GXPO_CMD" ;;
  --both) launch opd2_only "$ONLY_CMD"; launch opd2_gxpo "$GXPO_CMD" ;;
  *) echo "usage: $0 [--both|--only|--gxpo]" >&2; exit 2 ;;
esac

echo
echo "attach:  tmux attach -t opd2_only | opd2_gxpo"
echo "logs:    $CODE_ROOT/runs_opd2_opd2_only.log  /  runs_opd2_opd2_gxpo.log"
