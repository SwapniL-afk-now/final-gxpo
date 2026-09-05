# On-Policy KD Launcher Families

This document records the related on-policy KD launchers for the Qwen2.5
1.5B and 3B students. Each model has two variants. The only algorithmic
difference between the variants is the actor update method.

## Launcher matrix

| Student | Mode | Bash file | Actor update | GXPO |
|---|---|---|---|---|
| Qwen2.5-Math-1.5B-Instruct | On-policy KD | `experiments/gxpo_efficiency/qwen25_math_1p5b_onpolicy_kd.sh` | Normal `update_actor()` optimizer update | Explicitly disabled |
| Qwen2.5-Math-1.5B-Instruct | On-policy KD + GXPO | `experiments/gxpo_efficiency/qwen25_math_1p5b_onpolicy_kd_gxpo.sh` | Transactional `gxpo_update_actor()` predict/reposition update | Enabled; forced active by default |
| Qwen2.5-3B-Instruct | On-policy KD | `experiments/gxpo_efficiency/qwen25_3b_onpolicy_kd.sh` | Normal `update_actor()` optimizer update | Explicitly disabled |
| Qwen2.5-3B-Instruct | On-policy KD + GXPO | `experiments/gxpo_efficiency/qwen25_3b_onpolicy_kd_gxpo.sh` | Transactional `gxpo_update_actor()` predict/reposition update | Enabled; forced active by default |

The naming rule is intentional:

- `_onpolicy_kd.sh` means normal actor update without GXPO.
- `_onpolicy_kd_gxpo.sh` means the transactional GXPO actor update.
- `qwen25_math_1p5b` and `qwen25_3b` identify the student model family.

## Shared on-policy KD flow

For every training step, all four launchers:

1. Generate fresh responses with the student vLLM engine.
2. Put the student engine to sleep.
3. Wake the frozen plain-HuggingFace Qwen2.5-Math-7B teacher.
4. Score those exact student responses.
5. Attach teacher top-k tensors to the current batch.
6. Run the selected actor update.

The teacher never generates responses and is never run through vLLM.

## Update-path details

### KD-only variants

The 1.5B and 3B KD-only launchers explicitly set:

```text
+actor_rollout_ref.actor.use_gxpo=False
+actor_rollout_ref.actor.use_kd=True
+actor_rollout_ref.actor.kd_use_pg=False
```

They use the normal actor optimizer update. They do not run GXPO prediction,
extrapolation, or reposition logic. `kd_use_pg=False` preserves the current
KD loss-only behavior.

### KD + GXPO variants

The GXPO launchers explicitly enable the transactional path. Their key settings
include:

```text
+actor_rollout_ref.actor.use_gxpo=True
+actor_rollout_ref.actor.gxpo_k=10
+actor_rollout_ref.actor.gxpo_force_all_steps=True
+actor_rollout_ref.actor.use_kd=True
```

The 3B launchers default to a more conservative actor mini-batch/token profile
(`ppo_mini_batch_size=32`, `ppo_max_token_len_per_gpu=12288`) because the student
is larger. These values can be overridden through environment variables.

## Shared implementation files

- `verl/trainer/ppo/teacher_kd.py` — teacher-group lifecycle and batch scoring
- `verl/workers/teacher_scoring_worker.py` — frozen HF teacher scoring
- `verl/trainer/ppo/ray_trainer.py` — on-policy scoring and update-path dispatch
- `verl/workers/actor/dp_actor.py` — KD loss and normal/GXPO actor updates

`experiments/gxpo_efficiency/common.sh` remains unchanged and is not used as a
wrapper for this launcher family.

## Launch checklist

Each launcher requires:

```text
TRAIN_FILES
VAL_FILES
KD_TEACHER_PATH
```

Run a preflight first, changing only the launcher filename for the desired
student/update variant:

```bash
TRAIN_FILES="['/path/train.parquet']" \
VAL_FILES="['/path/val.parquet']" \
KD_TEACHER_PATH="/path/to/Qwen2.5-Math-7B-Instruct" \
bash experiments/gxpo_efficiency/qwen25_3b_onpolicy_kd.sh --dry-run
```

Use `qwen25_3b_onpolicy_kd_gxpo.sh` for the 3B GXPO variant, or the two
`qwen25_math_1p5b_*` names for the 1.5B variants.
