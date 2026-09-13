# Optimizer-aware GXPO retention

This document describes how GXPO measures *retention* -- the quantity it
extrapolates over K steps -- and why that measurement depends on which optimizer
owns the parameter. It complements `GXPO_MUON_ALGORITHM.md`, which describes the
Muon arm in full; the three-pass structure, the trigger/shutoff gate, the
corrective pass and the transactional semantics are unchanged and documented
there.

Implementation:

- `verl/workers/actor/dp_actor.py`
- `verl/workers/actor/gxpo_state.py`
- `verl/workers/actor/optimizer_transaction.py`
- `verl/workers/muon.py`

## The principle

~~~text
GXPO models optimizer-induced update dynamics,
not necessarily raw-gradient dynamics.
~~~

GXPO's reposition rule extrapolates the trajectory the optimizer would follow if
it kept taking steps. What "keeping going" means is a property of the optimizer:

| Optimizer | Direction it actually applies | Retention |
| --- | --- | --- |
| SGD | the gradient itself | reduces to gradient retention |
| AdamW | adaptive moment-preconditioned direction | coordinatewise `r = d1 / d0` |
| Muon | matrix-coupled, orthogonalized, magnitude-normalized | per-matrix scalar `rho = <u0,u1> / <u0,u0>` |

The raw-gradient rule `r = g1 / g0` is the SGD case. It was GXPO's only
estimator originally, and it is correct exactly when the step is proportional to
the gradient.

## Why raw `g1 / g0` is only an approximation for AdamW

AdamW does not move the parameters along the gradient. It transforms the
gradient through its moments,

~~~text
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
~~~

and steps along

~~~text
d_t = m_hat_t / (sqrt(v_hat_t) + eps)
~~~

subject to whatever bias-correction and epsilon ordering the implementation
uses. With beta1 = 0.9 and beta2 = 0.999 the moments are heavily smoothed, so a
gradient that changes sharply between the two probe steps produces a *much*
smaller change in `d`. A worked case from
`tests/gxpo/test_gxpo_adamw_direction.py` (settled moments, `g1 = 4 * g0`):

~~~text
raw gradient        r = 3.00 (clipped from 4)   scale = 6.00  (the K/2+1 ceiling)
AdamW direction     r = 0.87                    scale = 3.05
~~~

The legacy estimator pins the scale at its ceiling and extrapolates to a
displacement AdamW will never produce. This is not a small correction; it is a
different prediction.

## The new AdamW rule

For every AdamW-owned coordinate:

~~~text
r_i = d1_i / d0_i
q_i = S_K(r_i) / S_2(r_i) = S_K(r_i) / (1 + r_i)
~~~

with `S_K(r) = 1 + r + ... + r^(K-1)`.

The reposition equation is unchanged from the existing GXPO baseline -- only the
meaning of `q` changed:

~~~text
theta_tilde = theta0 + alpha * q * (theta2 - theta0)
~~~

No decay-aware `T_K(r, c)` predictor is introduced here. That is a separate
methodological question; mixing it in would make this patch uncontrolled.

## How `d_t` is reconstructed

Both AdamW implementations in this tree -- `torch.optim.AdamW` and the AdamW
branch of `verl.workers.muon.Muon` -- write the parameter as a *decoupled* step:

~~~text
theta_{t+1} = (1 - lr * weight_decay) * theta_t - lr * d_t
~~~

So with `c = 1 - lr * weight_decay`, the direction that actually produced the
step is exactly

~~~text
d_t = (c * theta_t - theta_{t+1}) / lr
~~~

GXPO uses this inversion rather than re-deriving `m_hat / (sqrt(v_hat) + eps)`
from optimizer state. That choice buys, for free and by construction:

- first-moment dynamics;
- second-moment dynamics;
- the exact bias correction that implementation uses;
- the exact epsilon convention and its placement;
- the **clipped** gradient the optimizer actually consumed, not the raw
  pre-clip gradient GXPO captured for the gate;
- independence from state-key names. `torch.optim.AdamW` stores
  `exp_avg` / `exp_avg_sq` / `step`; Muon's branch stores
  `moment1` / `moment2` / `step` and applies its correction as
  `(1 - b1^t) / sqrt(1 - b2^t)` on the write-back rather than to the moments.
  A state-based implementation would have to special-case each of them and would
  silently rot if either changed.

`lr` and `weight_decay` are read from the parameter's own param group after each
probe step, never assumed globally constant: LR is schedule-driven and groups may
differ. At `lr == 0` (an LR-warmup step) there is no direction to read and the
parameter falls back to neutral retention.

The two probe directions are

~~~text
d0 = (c0 * theta0 - theta1) / lr0
d1 = (c1 * theta1 - theta2) / lr1
~~~

`theta1` is never stored. `u0 = theta1 - theta0` lives in the `g1` buffer slot
(see *Memory* below), so `theta1 = theta0 + u0`.

### Numerical note

`c * theta_t - theta_{t+1}` is a difference of two nearly equal FP32 numbers, so
`d` carries a relative error of roughly `eps(theta) / (lr * |d|)`. At production
magnitudes (weights ~1e-2, lr 1e-6, `|d|` ~ 1 for AdamW) that is ~1e-3 -- about
three significant digits, far finer than `S_K(r)/S_2(r)` can distinguish.
`tests/gxpo/test_gxpo_adamw_direction.py::test_fp32_reconstruction_is_accurate_enough_at_production_magnitudes`
pins this against an FP64 reference.

## Stabilization

Deliberately identical to the gradient-space estimator, so the retention
*signal* is the only thing that changed:

- a coordinate is active only when `abs(d0_i) > 1e-3 * RMS(d0)` -- thresholded on
  `d0`, the quantity being divided by, not on `g0`, which is a different
  quantity once the preconditioner is in the loop;
- inactive coordinates get neutral retention (`r = 1`, `scale = 1`);
- active ratios are clipped to `[-2, 3]`; non-finite ratios become neutral;
- the scale is bounded to `[1, K/2 + 1]`.

Under FSDP the `RMS(d0)` threshold is reduced over the sharding process group
(never the replica dimension), exactly as the `g0` RMS is, so it is invariant to
`FSDP_SIZE`. One collective covers every parameter.

## How AdamW differs from Muon

Muon's retention stays a **scalar per matrix**. Its step size is independent of
gradient magnitude -- it normalizes the momentum matrix before Newton--Schulz and
scales the write-back by parameter shape alone -- so the coordinatewise rule is
not merely inaccurate for it, it would distort the orthogonalized direction Muon
chose. AdamW's step, by contrast, *is* coordinatewise: its preconditioner is
diagonal, so a coordinatewise ratio is the natural object.

This patch does not touch the Muon path: not the `rho` estimator, not its
clamping, not its FSDP dot-product reductions, not Newton--Schulz.

## Classification

`verl/workers/actor/gxpo_state.py` defines three retention kinds
(`RetentionKind.LEGACY_GRAD`, `ADAMW_DIRECTION`, `MUON_UPDATE`). The actor
classifies each parameter once and caches the result
(`_gxpo_retention_kinds`).

`gxpo_retention_space` (env `GXPO_RETENTION_SPACE`):

| Setting | Muon-owned matrix | AdamW-owned parameter | Unknown optimizer |
| --- | --- | --- | --- |
| `auto` (default) | `MUON_UPDATE` | `ADAMW_DIRECTION` | `LEGACY_GRAD` + one warning |
| `grad` | `LEGACY_GRAD` | `LEGACY_GRAD` | `LEGACY_GRAD` |
| `update` | `MUON_UPDATE` | `MUON_UPDATE` | `MUON_UPDATE` |

"AdamW-owned" means the optimizer is recognized as taking a decoupled-AdamW
step: `torch.optim.AdamW`, or the non-Muon branch of `verl.workers.muon.Muon`
(`state[p]['use_muon'] is False`). Anything else -- SGD, `torch.optim.Adam`
(whose weight decay is coupled into the gradient, so the inversion above does not
hold), an unrecognized wrapper -- is **not** silently modelled as AdamW; it falls
back to the legacy estimator and prints one warning.

## Transactional semantics are unchanged

The probe optimizer state is intentionally discarded. In `transactional` mode:

~~~text
(theta1, s1) = AdamW(theta0, s0, g0)     probe 1
(theta2, s2) = AdamW(theta1, s1, g1)     probe 2
theta_tilde  = theta0 + alpha * q * (theta2 - theta0)
s2 -> s0                                  rollback
g_c          = grad L(theta_tilde)
(theta_next, s_next) = AdamW(theta_tilde, s0, g_c)     the ONE retained step
~~~

The probes exist only to measure the local trajectory; letting their moments and
step counter persist would make the retained update depend on a trajectory the
run does not actually take. Consequently AdamW's step counter advances **once**
per PPO mini-batch, not three times. `transactional_fast_state` is the explicit
ablation that keeps the probe state (counter advances 3x) and is unaffected by
this change.

## Memory

No additional persistent model-sized buffer. GXPO still allocates exactly three
(`theta0`, `g0`, `g1`). Neither optimizer-aware estimator needs raw `g1`, so both
store `u0 = theta1 - theta0` in the `g1` slot and skip its gradient capture; the
AdamW path recovers `theta1 = theta0 + u0` from it and derives `d0` and `d1`
transiently, one parameter at a time. The `d0` RMS pass streams its directions
through a generator so they are never all resident at once. The shutoff/cosine
gate is unaffected -- it reads `g0` and the corrective gradient, never `g1`.

`gxpo_retention_space=grad` still needs real raw `g1`, so that legacy capture
path is preserved intact.

## Diagnostics

The three retention families are reported separately and are never averaged
together -- their denominators and their meanings differ.

AdamW optimizer-direction path (denominator: AdamW-owned coordinates only):

- `actor/gxpo_adamw_direction_params`
- `actor/gxpo_adamw_r_mean`, `actor/gxpo_adamw_r_std`
- `actor/gxpo_adamw_scale_mean`, `actor/gxpo_adamw_scale_max`
- `actor/gxpo_adamw_inactive_frac`
- `actor/gxpo_adamw_ratio_clip_frac`

Muon update-space path (denominator: Muon-owned matrices):

- `actor/gxpo_update_space_params`
- `actor/gxpo_retention_rho_mean`, `actor/gxpo_retention_rho_negative_frac`
- `actor/gxpo_update_scale_mean`

Legacy gradient-space path (denominator: `n_grad_coords`, the coordinates that
actually took that path -- zero on an `auto` AdamW run):

- `actor/gxpo_r_mean`, `actor/gxpo_r_std`
- `actor/gxpo_inactive_frac`, `actor/gxpo_ratio_clip_frac`
- `actor/gxpo_cos_g0_g1`

Displacement aggregates (`actor/gxpo_scale_mean`,
`actor/gxpo_effective_multiplier`, `actor/gxpo_disp2_norm`,
`actor/gxpo_dispK_norm`) remain coordinate-weighted over *all* parameters
regardless of kind, since they describe the reposition itself.

On a hybrid Muon run with `auto`, the Muon and AdamW subsets can therefore be
read independently in W&B.

## Reproducing the old behavior

~~~bash
GXPO_RETENTION_SPACE=grad   # legacy raw-gradient retention, r = g1/g0
GXPO_RETENTION_SPACE=auto   # optimizer-aware retention
~~~

The legacy implementation (`compute_gxpo_retention_scale`) is not deleted; both
estimators call one shared, stabilization-identical core.

`common.sh` tags the run name `_adamwdir` whenever `OPTIMIZER_NAME=adamw` and
`GXPO_RETENTION_SPACE=auto`, because that combination used to be a no-op and now
selects a different algorithm. Without the tag, `trainer.resume_mode=auto` would
let the new arm resume the old raw-gradient run's wandb id, checkpoints and
result directory. `grad` stays untagged: it is what every existing AdamW run name
already means.

Launchers for the Qwen2.5-Math-1.5B A/B:

| Arm | Launcher | Run name |
| --- | --- | --- |
| legacy `g1/g0` | `experiments/gxpo_efficiency/qwen25_math_1p5b_gxpo_k10.sh` | `qwen25-math-1p5b_gxpo_k10_seed3407` |
| AdamW direction | `experiments/gxpo_efficiency/qwen25_math_1p5b_gxpo_adamw_transactional_dir_k10.sh` | `qwen25-math-1p5b_gxpo_k10_seed3407_adamwdir` |

## The SFT / SFT-KD arm

`verl/trainer/fsdp_sft_trainer.py` carries its **own** GXPO implementation --
the SFT and KD launchers never reach `dp_actor.py`. `KDSFTTrainer`
(`verl/trainer/kd_sft_trainer.py`) subclasses it and inherits the same
`_gxpo_training_step`. The optimizer there is always `torch.optim.AdamW`, and
the same rule applies: `d_t = ((1 - lr*wd) * theta_t - theta_{t+1}) / lr`,
`r = d1/d0`, `u0` parked in the `g1` slot, three buffers not four.

Two differences from the actor:

- **`update` is rejected**, not aliased. There is no Muon-owned matrix in an
  AdamW SFT run for a per-matrix update-space scalar to describe, so
  `optim.gxpo_retention_space=update` raises rather than quietly doing something
  else.
- **The default is `grad`, not `auto`.** In `dp_actor.py`, `auto` was already a
  live setting (it selected Muon's update-space estimator), so making it
  optimizer-aware extended an existing choice. In the SFT trainer `auto` never
  existed: every SFT and SFT-KD launcher in this tree was written against the
  raw-gradient estimator and none passes the flag, so defaulting to `auto` would
  change the algorithm under all of them at once with no run-name change to show
  for it. Launchers opt in explicitly.

Diagnostics mirror the actor's under the `train/` prefix:
`train/gxpo_adamw_direction_params`, `train/gxpo_adamw_r_mean`,
`train/gxpo_adamw_r_std`, `train/gxpo_adamw_scale_mean`,
`train/gxpo_adamw_scale_max`, `train/gxpo_adamw_inactive_frac`,
`train/gxpo_adamw_ratio_clip_frac`. The legacy `train/gxpo_r_*` and
`train/gxpo_cos_g0_g1` family is normalized over `n_grad_coords` and reads zero
on an `auto` run.

### KD launchers

Each opts in with `GXPO_RETENTION_SPACE` (default `auto`), validates the value,
forwards the validated variable, and appends `_adamwdir` to its run name so the
optimizer-aware arm cannot inherit a raw-gradient run's directory or wandb id.
`GXPO_RETENTION_SPACE=grad` reproduces the legacy arm under the original name.

| Launcher | Trainer | Config prefix |
| --- | --- | --- |
| `train-scripts/off-policy-sft-kd/run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh` | `FSDPSFTTrainer` | `+optim.` |
| `experiments/gxpo_efficiency/qwen25_math_1p5b_kd_gxpo_b256_mb64_gate_v6.sh` | `KDSFTTrainer` | `+optim.` |
| `experiments/gxpo_efficiency/qwen25_math_1p5b_onpolicy_kd_gxpo.sh` | RL actor | `+actor_rollout_ref.actor.` |
| `experiments/gxpo_efficiency/qwen25_3b_onpolicy_kd_gxpo.sh` | RL actor | `+actor_rollout_ref.actor.` |

(`qwen25_math_1p5b_offpolicy_kd_gxpo_k10_a03.sh` execs the 1.5B on-policy
launcher, so it inherits the block.)

The two RL-actor KD launchers pass the flag **explicitly** even though the actor
already defaults to `auto`: they build their own run names rather than going
through `common.sh`, so an inherited default would have changed the algorithm
without changing the name.

## Tests

`tests/gxpo/test_gxpo_adamw_direction.py` covers the reconstruction against
torch's own state and against Muon's AdamW branch, per-group `lr`/`weight_decay`,
FP32 accuracy at production magnitudes, two-step retention, the regression that
`d1/d0` is not `g1/g0`, neutral behavior on a vanished denominator, transactional
state restoration and the single retained step-counter advance, the fast-state
ablation, the unchanged Muon scalar rule, hybrid and pure-AdamW classification,
the unknown-optimizer fallback, and the no-fourth-buffer memory invariant.

`tests/gxpo/test_gxpo_sft_adamw_direction.py` covers the second implementation:
the `grad` default and the launchers that depend on it, the `update` rejection,
`KDSFTTrainer` inheriting rather than forking the step, classification and the
non-AdamW fallback, per-group `lr`/`wd`, the skippable `g1` capture, the
three-buffer invariant, the `d0` RMS and ratio on a real two-probe sequence, the
separated metric denominators, and the opt-in/validate/tag/forward contract for
every KD launcher.

## Metrics: a key is logged only when its estimator ran

GXPO reports three retention families, and on any given run at most two of them
mean anything. Under `auto` with AdamW the `g1` buffer holds `u0 = θ1 − θ0`, not
a gradient, so `g1_norm`, `r_mean`, `r_std` and `cos_g0_g1` are not "zero" —
they are undefined. They used to be emitted as `0.000`, which wandb cannot
distinguish from a measurement and which draws a flat line through the middle of
every retention plot.

They are now **omitted**. `reduce_metrics` is a per-key `np.mean` with no key
union, `Tracking.log` passes the dict through unchanged, and the console
formatter iterates `dict.items()`, so an absent key produces a gap rather than a
zero everywhere downstream.

| Key (`train/` in SFT, `actor/` in RL) | Defined when |
| --- | --- |
| `gxpo_retention_kind` (0 grad, 1 adamw, 2 muon, 3 mixed) | always |
| `gxpo_legacy_grad_params`, `gxpo_adamw_direction_params`, `gxpo_update_space_params` | always (sum to the parameter count) |
| `gxpo_g0_norm`, `gxpo_gslow_norm`, `gxpo_scale_mean`, `gxpo_effective_multiplier`, `gxpo_contracting`, `gxpo_disp2_norm`, `gxpo_dispK_norm`, `gxpo_dispK_over_disp2`, `gxpo_clip_scale_g0` | always — these describe the reposition itself, not a ratio |
| `gxpo_g1_norm`, `gxpo_r_mean`, `gxpo_r_std`, `gxpo_scale_max`, `gxpo_cos_g0_g1`, `gxpo_inactive_frac`, `gxpo_ratio_clip_frac`, `gxpo_clip_scale_g1` | a parameter took the legacy gradient path |
| `gxpo_adamw_r_mean`, `_r_std`, `_scale_mean`, `_scale_max`, `_inactive_frac`, `_ratio_clip_frac`, `_d0_norm`, `_d1_norm`, `_cos_d0_d1` | a parameter took the AdamW-direction path |
| `gxpo_retention_rho_mean`, `gxpo_update_scale_mean`, `gxpo_retention_rho_negative_frac` | a Muon matrix took the update-space path (RL arm only) |

`gxpo_adamw_d0_norm` / `d1_norm` / `cos_d0_d1` are the direct analogues of
`g0_norm` / `g1_norm` / `cos_g0_g1` for the directions AdamW actually applied.
They exist because without them an `auto` run reported no magnitude or turn
information whatsoever about what it was extrapolating.

`_gxpo_default_metrics` (RL, non-GXPO step) now carries only the four gate keys.
A fallback step measured no retention, and zero-filling it made that step
indistinguishable from a real step whose retention happened to be zero.

Report-side coercions were removed to match: `make_figures.py` no longer
`fillna(0.0)`s a retention column (matplotlib breaks the line at NaN, which is
the honest picture) and falls back to the AdamW column when the legacy one was
never measured; `make_tables.py` reports the family that was measured and `NaN`
for one that was not.
