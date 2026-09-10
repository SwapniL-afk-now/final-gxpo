# Current GXPO algorithm for Muon

This document describes the implementation currently used by final-gxpo for
GXPO (Gradient Extrapolation-based Policy Optimization) when the actor uses
Muon. The relevant implementation is in:

- verl/workers/actor/dp_actor.py
- verl/workers/actor/gxpo_state.py
- verl/workers/actor/optimizer_transaction.py
- verl/workers/muon.py

For how GXPO chooses a retention estimator per optimizer -- in particular the
AdamW optimizer-direction rule that non-Muon parameters now use under
`gxpo_retention_space=auto` -- see `GXPO_OPTIMIZER_AWARE_RETENTION.md`.

## Summary

For each PPO mini-batch, GXPO performs two real optimizer steps as probes,
measures how the second step retains the direction of the first, repositions
the model along the resulting two-step displacement, and then performs one
slow corrective step. For Muon-owned matrices, retention is measured in
parameter-update space as one scalar per matrix. This preserves the direction
chosen by Muon's Newton--Schulz orthogonalization. Non-Muon parameters use a
coordinatewise estimator: under `gxpo_retention_space=auto` that is the AdamW
optimizer-direction ratio `r = d1/d0`, and under `grad` it is the legacy
raw-gradient ratio `r = g1/g0`.

## Notation

- theta0: parameters at the beginning of the mini-batch.
- g0: raw, pre-clipping policy gradient at theta0.
- theta1: parameters after the first optimizer step.
- g1: raw gradient at theta1 for the same mini-batch.
- theta2: parameters after the second optimizer step.
- gslow: raw corrective gradient at the repositioned parameters.
- K: GXPO extrapolation horizon, configured by gxpo_k.
- alpha: reposition coefficient, configured by gxpo_alpha.
- delta: geometric-series denominator guard, normally 1e-8.

The two observed optimizer displacements are:

~~~text
u0    = theta1 - theta0
u1    = theta2 - theta1
disp2 = theta2 - theta0 = u0 + u1
~~~

## Three-pass update

When GXPO is enabled and its shutoff gate is open, one PPO mini-batch follows
these phases.

### Pass 1: first probe

1. Copy every trainable parameter to theta0.
2. Snapshot the local optimizer state.
3. Backpropagate at theta0 and save the raw gradient as g0.
4. Clip the gradient only for the optimizer, then perform the first real
   optimizer step, producing theta1.

The raw gradient is captured before clipping. The clip multiplier is retained
for the gradient-space retention calculation.

### Pass 2: second probe

1. Backpropagate the same mini-batch at theta1.
2. Save g1 for gradient-space parameters.
3. For each Muon-owned parameter, reuse the g1 buffer to store u0 =
   theta1 - theta0; g1 is not used for Muon retention.
4. Clip the second optimizer gradient and perform the second real optimizer
   step, producing theta2.

If a probe has an invalid or non-finite global gradient norm, all ranks restore
theta0 and the optimizer snapshot, clear gradients, and run one ordinary
single-pass GRPO update for that mini-batch.

## Muon-aware retention

### Why gradient ratios are wrong for Muon

Muon is not a gradient-proportional optimizer. It updates momentum, applies
Newton--Schulz zeroth-power orthogonalization to a matrix direction, and writes
back a shape-dependent step size. Consequently, a coordinatewise g1/g0 ratio
does not describe the evolution of a Muon parameter's displacement.

### Update-space estimator

For each Muon-owned matrix, compute one scalar:

~~~text
rho = <u0, u1> / <u0, u0>
~~~

The dot products describe the whole original matrix. Under FSDP, each rank has
only a shard, so both dot products are summed across the FSDP sharding process
group before rho is formed. A vanished first step gives neutral retention
rho = 0. Non-finite values are replaced with zero and rho is clamped to
[-1, 1].

Define the geometric sum:

~~~text
S_n(rho) = 1 + rho + ... + rho^(n-1)
scale   = S_K(rho) / S_2(rho)
~~~

The implementation evaluates S_n with Horner recurrence. If abs(S_2) is no
greater than delta, scale is 1. Non-finite scales become 1 and the scale is
bounded to [0, K/2 + 1].

The Muon matrix is repositioned as:

~~~text
theta_tilde = theta0 + alpha * scale * (theta2 - theta0)
~~~

The scale is deliberately a scalar for the complete matrix. It changes only
the magnitude of the displacement and never distorts the direction selected by
Muon. The actual multiplier is alpha * scale:

- greater than 1: moves beyond theta2 (extrapolation);
- equal to 1: lands at theta2;
- less than 1: contracts between theta0 and theta2.

The optional gxpo_min_effective_multiplier can clamp this multiplier from below;
the actor warns if the mean effective multiplier is below one.

### Coordinatewise estimator for other parameters

Embeddings, output heads, norms, and other parameters not owned by Muon use a
coordinatewise estimator. Which one depends on `gxpo_retention_space`.

Under `auto` those parameters are AdamW-owned (Muon runs its own decoupled-AdamW
branch for them), so they use the AdamW optimizer-direction ratio `r = d1/d0`,
where `d_t = ((1 - lr*wd) * theta_t - theta_{t+1}) / lr` is reconstructed from
the real probe displacement. See `GXPO_OPTIMIZER_AWARE_RETENTION.md`.

Under `grad`, every parameter -- Muon-owned included -- falls back to the legacy
raw-gradient estimator described next. For active coordinates:

~~~text
r     = (c1 * g1) / (c0 * g0)
scale = S_K(r) / S_2(r)
~~~

Here c0 and c1 are the actual gradient-clip multipliers used by the two probe
steps. A coordinate is active only when abs(g0) is greater than 1e-3 times the
RMS of g0 for that parameter tensor. Inactive coordinates use neutral retention
(r = 1 and scale = 1). Active ratios are clipped to [-2, 3], non-finite
values become neutral, and scales are bounded to [1, K/2 + 1]. The same
theta_tilde formula is then applied coordinatewise.

Retention-space selection is controlled by gxpo_retention_space:

| Setting | Muon parameters | Other parameters |
| --- | --- | --- |
| auto (default) | update-space scalar | AdamW optimizer-direction (`d1/d0`) |
| grad | gradient-space | gradient-space (`g1/g0`, legacy) |
| update | update-space | update-space |

With a pure AdamW optimizer, auto selects no update-space parameters; every
trainable parameter takes the AdamW optimizer-direction path. Under `auto` an
optimizer that is neither Muon nor a recognized decoupled AdamW falls back to
gradient space with one warning rather than being modelled incorrectly.

## Pass 3: slow corrective update

After theta_tilde is written into the live parameters, the same mini-batch is
evaluated once more to obtain gslow. The corrective gradient is clipped and one
final optimizer step is performed. This slow step is the retained update; the
two probe steps only estimate the local trajectory and retention.

The default gxpo_optimizer_state_mode=transactional restores the optimizer
snapshot taken before Pass 1 before Pass 3. Probe momentum and step counters
therefore do not pollute the retained optimizer trajectory. The alternate
transactional_fast_state mode keeps the two probe mutations and performs Pass 3
from that fast state; optimizer state then advances through all three passes.

If gxpo_skip_corrective is enabled, Pass 3 is omitted and theta_tilde itself is
the final update.

## Muon optimizer used by the actor

build_muon classifies every trainable parameter using validated FSDP original
parameter metadata:

- Muon: contiguous, global two-dimensional matrices that are not embeddings or
  output heads.
- AdamW: every other trainable parameter.

In distributed mode, configured as muon_distributed_backend=gather_scatter,
each Muon matrix's local gradient and momentum are all-gathered, the dense
Muon update is computed on the reconstructed global matrix, and only the
rank-local update slice is written back. Momentum remains sharded locally.

For a Muon matrix, the update is:

~~~text
momentum <- momentum * muon_momentum + gradient
direction <- gradient + muon_momentum * momentum   # Nesterov path
update <- NewtonSchulz(direction, ns_steps)
parameter <- parameter * (1 - lr * weight_decay)
parameter <- parameter - adjust_lr_for_muon(lr, shape) * update
~~~

The shipped Muon GXPO launchers normally use momentum 0.95, Nesterov enabled,
five Newton--Schulz iterations, weight decay 1e-2, and gather/scatter. These
are launcher-configurable.

## Shutoff and fallback gate

GXPO has a quality gate. The actor records one trigger observation per outer
batch. With gxpo_trigger_signal=entropy, the trainer supplies the SFPO-style
entropy observation. With the gradient signal, the default observation is the
pre-clipping corrective norm abs(gslow); legacy_g0 uses abs(g0). Cosine mode
uses:

~~~text
observation = 1 - abs(cos(g0, gslow))
~~~

The rolling statistic is scored against the preceding window, never a window
that already contains the current observation. Once the window is full, the
z-score path trips on an upward excursion z >= gxpo_tau for
gxpo_trigger_patience consecutive observations. The implementation freezes the
baseline while a streak is open so sustained excursions remain countable.
Optional robust median/MAD and sustained relative-level criteria catch
outliers and slow drift.

When the gate trips, the next step is the first disabled step. With the
default gxpo_fallback_mode=permanent, GXPO buffers are released and all later
updates are ordinary single-pass GRPO. temporary mode falls back for
gxpo_fallback_window steps and then re-arms using the retained baseline. A
non-zero gxpo_max_active_steps is a hard compute budget and disables GXPO after
that many active outer steps regardless of the statistical gate.

Every rank makes the same gate decision through distributed reductions. Failed
probes and explicitly degenerate batches also restore the pre-probe state and
fall back to one standard update.

## Precision and distributed invariants

- Parameters, retained gradients, GXPO buffers, and floating optimizer state
  are required to remain FP32 under the strict precision contract.
- Retention arithmetic is widened before division and geometric recurrences.
- Muon update-space dot products are reduced over the FSDP sharding group, not
  the replica group, so rho is invariant to shard size.
- Update-space retention is scalar per original matrix; a per-coordinate scale
  would destroy Newton--Schulz geometry.
- Probe rollback restores parameters and local optimizer state, including newly
  created optimizer-state entries.

## Diagnostics

Important metrics for a Muon GXPO run include:

- actor/gxpo_retention_rho_mean and actor/gxpo_retention_rho_negative_frac
- actor/gxpo_update_scale_mean
- actor/gxpo_effective_multiplier and actor/gxpo_contracting
- actor/gxpo_disp2_norm, actor/gxpo_dispK_norm, and
  actor/gxpo_dispK_over_disp2
- actor/gxpo_trigger_stat, actor/gxpo_trigger_z,
  actor/gxpo_shutoff_step, and actor/gxpo_fallback_step
- actor/gxpo_optim_state_kept (0 transactional, 1 transactional-fast-state)
- optimizer/muon_update_norm and optimizer/muon_momentum_norm
- optimizer/muon_gather_time, optimizer/muon_newton_schulz_time, and
  optimizer/muon_scatter_time

For Muon runs with retention_space=auto, actor/gxpo_r_mean,
actor/gxpo_r_std, and actor/gxpo_cos_g0_g1 describe only the *legacy
gradient-space* subset, which under auto is empty (the AdamW-owned parameters
now report under actor/gxpo_adamw_r_mean and friends). They are not Muon
retention statistics. The AdamW subset's retention is reported separately -- see
the diagnostics section of `GXPO_OPTIMIZER_AWARE_RETENTION.md` -- so the Muon and
AdamW halves of a hybrid run can be read independently.
