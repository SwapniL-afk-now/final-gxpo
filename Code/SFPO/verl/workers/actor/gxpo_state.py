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
once the corrective-gradient norm deviates by |Z| >= tau from its EMA
baseline, extrapolation is permanently disabled (s* = step + 1) and all
subsequent updates fall back to single-pass GRPO.
"""

from typing import Optional, Tuple

__all__ = ['GXPOState']


class GXPOState:
    """Per-actor GXPO state: hyperparameters + rolling shutoff statistics."""

    VALID_SHUTOFF_MODES = ('trajectory_aware', 'legacy_g0', 'never')
    VALID_FALLBACK_MODES = ('permanent', 'temporary')

    def __init__(
        self,
        K: int = 5,
        alpha: float = 0.5,
        delta: float = 1e-8,
        tau: float = 0.5,
        omega: float = 0.1,
        shutoff_mode: str = 'trajectory_aware',
        warmup_steps: int = 0,
        fallback_mode: str = 'permanent',
        fallback_window: int = 10,
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
        self.shutoff_mode = shutoff_mode
        # After the gate trips, 'permanent' disables extrapolation for the rest of training;
        # 'temporary' reverts to GRPO for `fallback_window` steps then re-arms the gate with a
        # fresh EMA (Group-2 ablation). Default 'permanent' matches the shipped method.
        self.fallback_mode = fallback_mode
        self.fallback_window = int(fallback_window)
        # The EMA (mu, sigma) starts cold at (1, 1); the first few steps' z-scores are huge
        # until it tracks the true grad-norm scale (e.g. z~30 at step 0 for norms ~30). Suppress
        # the trigger for `warmup_steps` while the EMA is still updating so a cold-start spike
        # can't permanently disable extrapolation. Default 0 preserves the RL arm's behavior.
        self.warmup_steps = int(warmup_steps)

        self.trigger_index = float('inf')  # s*: first step with extrapolation disabled
        self.mu = 1.0  # EMA mean of the trigger statistic
        self.sigma = 1.0  # EMA std of the trigger statistic
        self.step_count = 0
        # The paper's gate uses a rolling history and does not evaluate a z-score
        # until at least two valid corrective-gradient observations exist.
        self.observation_count = 0

    def is_enabled(self, step: Optional[int] = None) -> bool:
        if step is None:
            step = self.step_count
        if step < self.trigger_index:
            return True
        # Tripped. Permanent stays off forever; temporary re-arms after the window by clearing the
        # trip. Keep the pre-trip EMA (mu, sigma) as the baseline -- it tracked the true norm scale
        # and the gate fired on a *deviation*, not a bad baseline; a cold reset to (1,1) would
        # instantly re-trip on the next real norm (the cold-start spike). During the window the
        # fallback runs standard GRPO steps, which don't touch the EMA, so it stays frozen-valid.
        if self.fallback_mode == 'temporary' and step >= self.trigger_index + self.fallback_window:
            self.trigger_index = float('inf')
            return True
        return False

    def update_stats(self, H_s: float) -> float:
        """EMA update of (mu, sigma); returns the standardized observation Z_s."""
        eps = 1e-8
        Z_s = (H_s - self.mu) / (self.sigma + eps)

        mu_old = self.mu
        sigma_sq_new = (1 - self.omega) * self.sigma**2 + self.omega * (H_s - mu_old)**2
        self.sigma = max(0.01, sigma_sq_new**0.5)
        self.mu = (1 - self.omega) * mu_old + self.omega * H_s

        return Z_s

    def resolve_trigger_observation(self, *, g0_norm: float, g_slow_norm: float) -> float:
        if self.shutoff_mode == 'legacy_g0':
            return float(g0_norm)
        return float(g_slow_norm)

    def check_trigger(self, Z_s: float, step: int) -> bool:
        if step < self.warmup_steps:
            return False
        # Algorithm 1 shuts off on an upward instability only.  A low-norm
        # observation is not evidence that extrapolation has become unsafe.
        if Z_s >= self.tau and self.trigger_index == float('inf'):
            self.trigger_index = step + 1
            return True
        return False

    def update_trigger_state(self, *, step: int, g0_norm: float,
                             g_slow_norm: float) -> Tuple[float, float, bool]:
        """Feed this step's norms into the gate. Returns (z_score, trigger_stat, triggered)."""
        trigger_stat = self.resolve_trigger_observation(g0_norm=g0_norm, g_slow_norm=g_slow_norm)
        if self.shutoff_mode == 'never':
            return 0.0, float(trigger_stat), False
        # Fill the rolling baseline before allowing a shutoff decision.  This
        # mirrors the paper's `if len(B) > 1` gate condition and avoids a
        # cold-start z-score based on the initial (1, 1) EMA.
        if self.observation_count < 2:
            self.update_stats(trigger_stat)
            self.observation_count += 1
            return 0.0, float(trigger_stat), False
        z_score = self.update_stats(trigger_stat)
        self.observation_count += 1
        triggered = self.check_trigger(z_score, step)
        return float(z_score), float(trigger_stat), bool(triggered)


if __name__ == '__main__':
    # Self-check: over one forward pass (mirroring dp_actor), record enabled-state per step.
    # warmup=3 avoids the cold-start trip; a single spike at step 10 trips the gate.
    def trajectory(mode):
        s = GXPOState(tau=2.0, omega=0.1, warmup_steps=3, shutoff_mode='trajectory_aware',
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
            # else: fallback runs a standard step, EMA left frozen (mirrors dp_actor)
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
    assert st.mu > 10.0, f'temporary should keep the learned EMA baseline, got mu={st.mu}'
    print(f'OK: permanent off for all steps after {trip_p}; '
          f'temporary re-enables at step {trip_t + 1 + 5} (mu kept {st.mu:.1f})')
