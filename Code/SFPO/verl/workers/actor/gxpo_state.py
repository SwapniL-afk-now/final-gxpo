# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GXPO (Gradient Extrapolation-Based Policy Optimization) trigger state.

Tracks the rolling z-score shutoff gate from Algorithm 1 of the GXPO paper:
once the corrective-gradient norm deviates by Z >= tau from its recent-window
baseline, extrapolation is permanently disabled (s* = step + 1) and all
subsequent updates fall back to single-pass GRPO.
"""

import statistics
from typing import Optional, Tuple

import torch

__all__ = ['GXPOState', 'geometric_sum_horner', 'compute_gxpo_retention_scale']


def geometric_sum_horner(value: torch.Tensor, n: int) -> torch.Tensor:
    """Return ``1 + value + ... + value**(n - 1)`` using Horner's rule."""
    if n < 1:
        raise ValueError(f'geometric sum length must be positive, got {n}')
    result = torch.ones_like(value)
    for _ in range(1, n):
        result.mul_(value).add_(1.0)
    return result


def compute_gxpo_retention_scale(g0: torch.Tensor, g1: torch.Tensor, K: int,
                                 delta: float) -> Tuple[torch.Tensor, torch.Tensor,
                                                         torch.Tensor, torch.Tensor]:
    """Compute production GXPO ratio/scale and diagnostic masks.

    Inactive coordinates retain the observed two-step displacement. Active
    ratios are clipped to [-2, 3], non-finite ratios are replaced with one,
    and the geometric scale is bounded to [1, K / 2 + 1].
    """
    if K < 2:
        raise ValueError(f'GXPO K must be at least two, got {K}')
    if g0.shape != g1.shape:
        raise ValueError(f'g0 and g1 must have the same shape, got {g0.shape} and {g1.shape}')

    one = torch.ones_like(g0)
    active = g0.abs() > delta
    # Branchless active-mask arithmetic avoids a CUDA scalar active.any()
    # synchronization while retaining the old sign convention for g0 == 0.
    sign_g0 = torch.where(g0 >= 0, one, -one)
    denominator = g0.abs().clamp_min(delta) * sign_g0
    candidate = g1 / denominator
    finite = torch.isfinite(candidate)
    ratio_clipped = active & ((~finite) | (candidate < -2.0) | (candidate > 3.0))
    candidate.clamp_(-2.0, 3.0).nan_to_num_(nan=1.0)
    ratio = torch.where(active, candidate, one)

    s_k = geometric_sum_horner(ratio, K)
    s_2 = geometric_sum_horner(ratio, 2)
    active_scale = torch.where(s_2.abs() > delta, s_k / s_2, one)
    active_scale = active_scale.nan_to_num(nan=1.0, posinf=1.0, neginf=1.0)
    active_scale.clamp_(1.0, K / 2.0 + 1.0)
    scale = torch.where(active, active_scale, one)
    return ratio, scale, active, ratio_clipped


class GXPOState:
    """Per-actor GXPO state: hyperparameters + rolling shutoff statistics."""

    VALID_SHUTOFF_MODES = ('trajectory_aware', 'legacy_g0', 'cosine', 'never')
    VALID_FALLBACK_MODES = ('permanent', 'temporary')

    def __init__(
        self,
        K: int = 5,
        alpha: float = 0.5,
        delta: float = 1e-8,
        tau: float = 0.5,
        omega: float = 0.1,
        zscore_w: int = 30,
        shutoff_mode: str = 'trajectory_aware',
        warmup_steps: int = 0,
        fallback_mode: str = 'permanent',
        fallback_window: int = 10,
        trigger_patience: int = 1,
        trigger_robust: bool = False,
        min_post_warmup_obs: int = 0,
        max_active_steps: int = 0,
        abs_threshold: float = 0.0,
        sustain_window: int = 10,
    ):
        if shutoff_mode not in self.VALID_SHUTOFF_MODES:
            raise ValueError(f'Invalid GXPO shutoff mode: {shutoff_mode}. '
                             f'Expected one of {sorted(self.VALID_SHUTOFF_MODES)}')
        if fallback_mode not in self.VALID_FALLBACK_MODES:
            raise ValueError(f'Invalid GXPO fallback mode: {fallback_mode}. '
                             f'Expected one of {sorted(self.VALID_FALLBACK_MODES)}')
        self.K = int(K)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.tau = float(tau)
        self.omega = float(omega)
        if int(zscore_w) < 1:
            raise ValueError('GXPO zscore_w must be at least 1')
        self.zscore_w = int(zscore_w)
        self.shutoff_mode = shutoff_mode
        # After the gate trips, 'permanent' disables extrapolation for the rest of training;
        # 'temporary' reverts to GRPO for `fallback_window` steps then re-arms the gate with the
        # retained rolling baseline (Group-2 ablation). Default 'permanent' matches the shipped method.
        self.fallback_mode = fallback_mode
        self.fallback_window = int(fallback_window)
        if int(trigger_patience) < 1:
            raise ValueError('GXPO trigger_patience must be at least 1')
        self.trigger_patience = int(trigger_patience)
        # Keep the warmup field for compatibility with existing launchers. The rolling window
        # itself also prevents a cold-start trigger until it contains zscore_w observations.
        self.warmup_steps = int(warmup_steps)
        # Robust gating (opt-in): score observations against the window median and a
        # MAD-derived scale instead of mean/std. Production runs showed the classic
        # z-score trips in early-transient bursts right after warmup and goes quiet
        # afterwards; the robust statistic resists exactly that contamination.
        self.trigger_robust = bool(trigger_robust)
        if int(min_post_warmup_obs) < 0:
            raise ValueError('GXPO min_post_warmup_obs must be non-negative')
        # Minimum number of scored post-warmup observations before the gate may trip.
        # Streaks accumulated before this age are discarded, so the first decision
        # after the minimum age is not an instant carry-over trip.
        self.min_post_warmup_obs = int(min_post_warmup_obs)
        # Hard cap on enabled (extrapolating) outer steps. 0 disables the cap.
        # This is a worst-case runtime guard, NOT a stability mechanism: the
        # statistical gate remains responsible for quality-based shutoff.
        if int(max_active_steps) < 0:
            raise ValueError('GXPO max_active_steps must be non-negative')
        self.max_active_steps = int(max_active_steps)
        # Sustained-level criterion (cosine mode): trip when the rolling median of
        # the last `sustain_window` disagreement observations stays >= abs_threshold
        # for `trigger_patience` consecutive scored batches. Calibrated on
        # production curves (.audit/gxpo_algorithm_findings.md): healthy runs sit
        # at median <= 0.06 (p90 <= 0.12); failing ones >= 0.25. Threshold 0.15
        # separates them with zero false trips and catches sustained degradation
        # that windowed z-scores structurally miss (the baseline adapts to the
        # new level and z falls back down). 0 disables the criterion (z-path).
        if float(abs_threshold) < 0:
            raise ValueError('GXPO abs_threshold must be non-negative')
        self.abs_threshold = float(abs_threshold)
        if int(sustain_window) < 2:
            raise ValueError('GXPO sustain_window must be >= 2')
        self.sustain_window = int(sustain_window)
        # True when shutoff came from the hard budget rather than the gate.
        self.budget_stop = None

        self.trigger_index = float('inf')  # s*: first step with extrapolation disabled
        self.trigger_streak = 0
        # Scored observations since the last warmup baseline reset (gate age).
        self.post_warmup_scored = 0
        self.mu = 1.0  # preceding rolling-window mean of the trigger statistic
        self.sigma = 1.0  # preceding rolling-window population std of the trigger statistic
        self.step_count = 0
        self.trigger_history = []
        # The paper's gate uses a rolling history and does not evaluate a z-score until
        # the configured window is full.
        self.observation_count = 0
        # SFPO clears observations collected during warmup and starts a fresh
        # baseline once trigger evaluation is enabled.
        self._warmup_reset_done = self.warmup_steps <= 0

    def is_enabled(self, step: Optional[int] = None) -> bool:
        if step is None:
            step = self.step_count
        # Hard compute budget (F5/F8 worst-case guard): never spend more than
        # max_active_steps outer steps in 3-pass mode, regardless of what the
        # statistical gate does. Bounds the runtime penalty of a mis-tuned gate;
        # recorded via trigger_index so existing shutoff metrics/reporting apply.
        if (self.max_active_steps > 0 and step >= self.max_active_steps
                and self.trigger_index == float('inf') and self.budget_stop is None):
            self.trigger_index = step
            self.budget_stop = True
            return False
        if step < self.trigger_index:
            return True
        # Tripped. Permanent stays off forever; temporary re-arms after the window by clearing the
        # trip. Keep the pre-trip rolling baseline (mu, sigma) -- it tracked the true norm scale
        # and the gate fired on a *deviation*, not a bad baseline; a cold reset would discard
        # useful recent history. During the window the
        # fallback runs standard GRPO steps, which don't touch the rolling baseline, so it stays frozen-valid.
        # budget_stop=True (max_active_steps) is a HARD cap: temporary mode must never re-arm past it.
        if (self.fallback_mode == 'temporary' and not self.budget_stop
                and step >= self.trigger_index + self.fallback_window):
            self.trigger_index = float('inf')
            return True
        return False

    def update_stats(self, H_s: float) -> float:
        """Score against the preceding window, then append the current observation.

        This is intentionally ordered like SFPO's entropy gate: the current
        observation is never part of the mean/std used to score itself.
        """
        history = self.trigger_history[-self.zscore_w:]
        if len(history) < self.zscore_w:
            self.trigger_history.append(float(H_s))
            return 0.0

        if self.trigger_robust:
            # Median/MAD location-scale: a single transient in the window moves the
            # baseline far less than mean/std, so one spike cannot both contaminate
            # the reference and hide the next one.
            ordered = sorted(history)
            mid = len(ordered) // 2
            median = (ordered[mid - 1] + ordered[mid]) / 2.0 if len(ordered) % 2 == 0 else ordered[mid]
            deviations = sorted(abs(value - median) for value in history)
            mad_mid = len(deviations) // 2
            mad = ((deviations[mad_mid - 1] + deviations[mad_mid]) / 2.0
                   if len(deviations) % 2 == 0 else deviations[mad_mid])
            self.mu = median
            # Consistent with std under Gaussian windows, floored at 10% of the
            # location: replay over production runs showed a near-constant window
            # drives raw MAD to ~0 and manufactures enormous z from tiny wobbles
            # (false trips). The floor keeps the robust scale physically meaningful.
            self.sigma = max(1.4826 * mad, 0.10 * abs(median))
        else:
            self.mu = sum(history) / self.zscore_w
            variance = sum((value - self.mu) ** 2 for value in history) / self.zscore_w
            self.sigma = variance ** 0.5
        z_score = (float(H_s) - self.mu) / (self.sigma + 1e-9)
        self.trigger_history.append(float(H_s))
        if len(self.trigger_history) > self.zscore_w:
            del self.trigger_history[:-self.zscore_w]
        return z_score

    def reset_trigger_baseline(self):
        """Discard warmup observations before collecting the SFPO-style window."""
        self.trigger_history.clear()
        self.observation_count = 0
        self.trigger_streak = 0
        self.post_warmup_scored = 0
        self.mu = 1.0
        self.sigma = 1.0

    def resolve_trigger_observation(self, *, g0_norm: float, g_slow_norm: float,
                                    stat_override: Optional[float] = None) -> float:
        """Resolve this step's raw gate observation.

        ``stat_override`` carries an already-resolved statistic (e.g. the outer-batch
        mean of per-minibatch observations, or |cos(g0, g_slow)| for the cosine mode,
        which cannot be recovered from norms alone)."""
        if stat_override is not None:
            return float(stat_override)
        if self.shutoff_mode == 'legacy_g0':
            return float(g0_norm)
        if self.shutoff_mode == 'cosine':
            raise ValueError(
                "shutoff_mode='cosine' requires the caller to pass the pre-computed "
                '|cos(g0, g_slow)| via stat_override; it cannot be derived from norms')
        return float(g_slow_norm)

    def check_trigger(self, Z_s: float, step: int) -> bool:
        if step < self.warmup_steps:
            self.trigger_streak = 0
            return False
        # Gate age floor (F3): every observed trip in production fired in the volatile
        # burst immediately after warmup. Require min_post_warmup_obs scored post-warmup
        # observations before any trip, and discard streaks accumulated before the age
        # threshold so reaching it does not instantly convert old streaks into a trip.
        if self.post_warmup_scored < self.min_post_warmup_obs:
            self.trigger_streak = 0
            return False
        # Algorithm 1 shuts off on an upward instability only.  A low-norm
        # observation is not evidence that extrapolation has become unsafe.
        if self.trigger_index != float('inf'):
            return False
        if Z_s >= self.tau:
            self.trigger_streak += 1
            if self.trigger_streak >= self.trigger_patience:
                self.trigger_index = step + 1
                return True
        else:
            self.trigger_streak = 0
        return False

    def update_trigger_state(self, *, step: int, g0_norm: float,
                             g_slow_norm: float, allow_trigger: bool = True,
                             defer_trigger: bool = False,
                             stat_override: Optional[float] = None) -> Tuple[float, float, bool]:
        """Feed this step's gate observation into the rolling shutoff state.

        ``allow_trigger=False`` is used for an outer training-step warmup: the rolling baseline and
        observation history continue to update, but this observation cannot trip the
        shutoff gate. This keeps the first post-warmup decision statistically useful
        without allowing a warmup spike to disable GXPO.

        ``stat_override`` supplies a pre-resolved observation (outer-batch mean or the
        cosine-mode |cos(g0, g_slow)|); see ``resolve_trigger_observation``.

        Returns ``(z_score, trigger_stat, triggered)``."""
        trigger_stat = self.resolve_trigger_observation(g0_norm=g0_norm, g_slow_norm=g_slow_norm,
                                                        stat_override=stat_override)
        if self.shutoff_mode == 'never':
            return 0.0, float(trigger_stat), False
        if not allow_trigger:
            # Keep warmup observations isolated; the first enabled observation
            # resets this history before the post-warmup rolling window starts.
            self.trigger_history.append(float(trigger_stat))
            if len(self.trigger_history) > self.zscore_w:
                del self.trigger_history[:-self.zscore_w]
            self.observation_count += 1
            self.trigger_streak = 0
            return 0.0, float(trigger_stat), False
        if not self._warmup_reset_done:
            self.reset_trigger_baseline()
            self._warmup_reset_done = True
        if defer_trigger:
            # A deferred caller must aggregate minibatches first and call this
            # method once with the outer-batch mean.
            return 0.0, float(trigger_stat), False
        # Fill the rolling baseline before allowing a shutoff decision. This mirrors
        # SFPO's rolling mean/std gate and avoids a cold-start z-score.
        z_score = self.update_stats(trigger_stat)
        self.observation_count += 1
        self.post_warmup_scored += 1
        if len(self.trigger_history) < self.zscore_w:
            return float(z_score), float(trigger_stat), False
        if self.shutoff_mode == 'cosine' and self.abs_threshold > 0:
            # Sustained-level criterion replaces the z-path for cosine mode when
            # enabled. The rolling median over `sustain_window` scored batches is
            # compared against abs_threshold; streak/patience/warmup/age-floor
            # semantics match check_trigger exactly.
            window = [float(v) for v in self.trigger_history[-self.sustain_window:]]
            med = statistics.median(window)
            eligible = (med >= self.abs_threshold
                        and step >= self.warmup_steps
                        and self.post_warmup_scored >= self.min_post_warmup_obs
                        and self.trigger_index == float('inf'))
            if eligible:
                self.trigger_streak += 1
                if self.trigger_streak >= self.trigger_patience:
                    self.trigger_index = step + 1
                    return float(z_score), float(trigger_stat), True
                return float(z_score), float(trigger_stat), False
            self.trigger_streak = 0
            return float(z_score), float(trigger_stat), False
        triggered = self.check_trigger(z_score, step)
        return float(z_score), float(trigger_stat), bool(triggered)


if __name__ == '__main__':
    # Self-check: over one forward pass (mirroring dp_actor), record enabled-state per step.
    # warmup=3 avoids the cold-start trip; a single spike at step 10 trips the gate.
    def trajectory(mode):
        s = GXPOState(tau=2.0, omega=0.1, zscore_w=10, warmup_steps=3, shutoff_mode='trajectory_aware',
                      fallback_mode=mode, fallback_window=5)
        norms = [30.0] * 10 + [300.0] + [30.0] * 15  # stable ~30, spike at step 10
        enabled, trip = [], None
        for step, h in enumerate(norms):
            on = s.is_enabled(step)          # gate decision for this step
            enabled.append(on)
            if on:
                _, _, fired = s.update_trigger_state(step=step, g0_norm=h, g_slow_norm=h)
                if fired and trip is None:
                    trip = step
            # else: fallback runs a standard step, rolling baseline left frozen (mirrors dp_actor)
        return enabled, trip, s

    en_p, trip_p, sp = trajectory('permanent')
    en_t, trip_t, st = trajectory('temporary')
    assert trip_p == trip_t == 10, f'gate should trip at the spike (step 10), got {trip_p}/{trip_t}'
    # permanent: off for every step after the trip
    assert not any(en_p[trip_p + 1:]), 'permanent must stay disabled after tripping'
    # temporary: off during the window (trip+1 .. trip+window), on again after
    assert not any(en_t[trip_t + 1:trip_t + 1 + 5]), 'temporary must be off inside the window'
    assert en_t[trip_t + 1 + 5], 'temporary must re-enable after fallback_window'
    # re-armed baseline kept (not cold-reset to 1.0, which would instantly re-trip on norm ~30)
    assert st.mu > 10.0, f'temporary should keep the learned rolling baseline, got mu={st.mu}'
    print(f'OK: permanent off for all steps after {trip_p}; '
          f'temporary re-enables at step {trip_t + 1 + 5} (mu kept {st.mu:.1f})')
