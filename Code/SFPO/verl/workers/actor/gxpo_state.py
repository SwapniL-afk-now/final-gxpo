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

    def __init__(
        self,
        K: int = 5,
        alpha: float = 0.5,
        delta: float = 1e-8,
        tau: float = 0.5,
        omega: float = 0.1,
        shutoff_mode: str = 'trajectory_aware',
    ):
        if shutoff_mode not in self.VALID_SHUTOFF_MODES:
            raise ValueError(f'Invalid GXPO shutoff mode: {shutoff_mode}. '
                             f'Expected one of {sorted(self.VALID_SHUTOFF_MODES)}')
        self.K = int(K)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.tau = float(tau)
        self.omega = float(omega)
        self.shutoff_mode = shutoff_mode

        self.trigger_index = float('inf')  # s*: first step with extrapolation disabled
        self.mu = 1.0  # EMA mean of the trigger statistic
        self.sigma = 1.0  # EMA std of the trigger statistic
        self.step_count = 0

    def is_enabled(self, step: Optional[int] = None) -> bool:
        if step is None:
            step = self.step_count
        return step < self.trigger_index

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
        if abs(Z_s) >= self.tau and self.trigger_index == float('inf'):
            self.trigger_index = step + 1
            return True
        return False

    def update_trigger_state(self, *, step: int, g0_norm: float,
                             g_slow_norm: float) -> Tuple[float, float, bool]:
        """Feed this step's norms into the gate. Returns (z_score, trigger_stat, triggered)."""
        trigger_stat = self.resolve_trigger_observation(g0_norm=g0_norm, g_slow_norm=g_slow_norm)
        if self.shutoff_mode == 'never':
            return 0.0, float(trigger_stat), False
        z_score = self.update_stats(trigger_stat)
        triggered = self.check_trigger(z_score, step)
        return float(z_score), float(trigger_stat), bool(triggered)
