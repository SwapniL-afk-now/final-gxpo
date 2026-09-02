"""Regression tests for the GXPO entropy shutoff gate.

These cover the defects that made the gate un-trippable in the
`llama32_3b_muon_gxpo` configuration (tau=3.0, patience=3, warmup=50):

* the scored observation used to sit inside its own baseline window, which caps
  abs(z) at sqrt(zscore_w - 1) and biases every score downward;
* `trigger_patience > 1` was self-defeating, because each accepted violation was
  folded into the baseline before the next one was scored;
* a windowed z-score cannot see a slow monotone drift at all -- the rolling mean
  simply follows it. That is the failure actually observed on Llama-3.2-3B.

The false-positive guard replays a real, healthy production entropy series so a
more sensitive gate cannot start shutting off good runs.
"""

import glob
import os
import random
import re

import pytest

from verl.workers.actor.gxpo_state import GXPOState


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_gate(**overrides):
    """A gate configured the way the trainer builds it (see _build_gxpo_entropy_gate)."""
    kwargs = dict(tau=3.0, zscore_w=30, warmup_steps=0, shutoff_mode='legacy_g0',
                  trigger_patience=3, sustain_window=10, relative_threshold=0.10)
    kwargs.update(overrides)
    return GXPOState(**kwargs)


def replay(series, **overrides):
    """Feed a series through the gate; return (trip_step, max_z, gate)."""
    gate = build_gate(**overrides)
    max_z = float('-inf')
    for step, value in enumerate(series):
        if not gate.is_enabled(step):
            continue
        z, _, fired = gate.update_trigger_state(step=step, g0_norm=value, g_slow_norm=value,
                                                stat_override=value)
        max_z = max(max_z, z)
        if fired:
            return step, max_z, gate
    return None, max_z, gate


# Real runs are noisy: the within-30-window coefficient of variation of
# `train/entropy_mean` is ~0.07 on the Qwen-Math production run (p10 0.06, p90 0.13)
# and ~0.04 over the first Llama-3.2-3B steps. Synthetic series must carry comparable
# noise, or a perfectly smooth ramp produces a near-zero window std and trips the
# z-path for reasons that never occur in practice.
PRODUCTION_CV = 0.07


def noisy(level_fn, n, cv=PRODUCTION_CV, seed=1234):
    rng = random.Random(seed)
    return [max(1e-6, level_fn(i) * (1.0 + rng.gauss(0.0, cv))) for i in range(n)]


def load_logged_entropy(pattern='results/gxpo_efficiency/*/train.log', minimum=60):
    """Real `train/entropy_mean` series from completed runs, longest first."""
    series = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, pattern))):
        with open(path, errors='ignore') as handle:
            values = [float(v) for v in re.findall(r'train/entropy_mean:([0-9.]+)', handle.read())]
        if len(values) >= minimum:
            series.append((os.path.basename(os.path.dirname(path)), values))
    return sorted(series, key=lambda item: -len(item[1]))


# --------------------------------------------------------------------------- #
# The reported failure mode: a slow drift the z-score cannot see.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('baseline', [0.12, 4.50])
def test_slow_drift_trips_level_criterion_but_not_zscore(baseline):
    """A +30% ramp over 80 steps must trip, at either entropy scale.

    `baseline=0.12` is the Qwen-Math regime, `4.50` the Llama-3.2-3B regime. The
    relative criterion is scale-free, so both must behave identically -- an
    absolute threshold could not have covered both.
    """
    ramp = noisy(lambda i: baseline * (1.0 + 0.30 * i / 79), 80)

    trip_z_only, _, _ = replay(ramp, relative_threshold=0.0)
    assert trip_z_only is None, 'z-only gate should be blind to a slow ramp (regression guard)'

    trip, _, gate = replay(ramp)
    assert trip is not None, 'relative level criterion must catch a sustained drift'
    # The baseline is the mean of the FIRST full scoring window, so on a ramp it already
    # carries the drift accumulated over those 30 steps -- it is not the initial value.
    assert gate.baseline_level == pytest.approx(sum(ramp[:30]) / 30, rel=0.02), (
        'the frozen baseline must be the mean of the first full post-warmup window')
    assert gate.baseline_level < sum(ramp[-30:]) / 30, (
        'the baseline must stay frozen below the drifted level, not follow it')


def test_relative_criterion_is_scale_free():
    """The same series scaled 40x must trip at exactly the same step."""
    ramp = noisy(lambda i: 1.0 + 0.30 * i / 79, 80)
    small, _, _ = replay([v * 0.12 for v in ramp])
    large, _, _ = replay([v * 4.80 for v in ramp])
    assert small == large is not None, f'{small} != {large}'


def test_drift_below_omega_does_not_trip():
    """A +5% drift is under the 10% omega and must be tolerated."""
    ramp = noisy(lambda i: 0.12 * (1.0 + 0.05 * i / 119), 120)
    trip, _, _ = replay(ramp)
    assert trip is None, f'a +5% drift is within tolerance, tripped at {trip}'


def test_flat_noisy_run_does_not_trip():
    """The calibration that matters: no false positive on a flat, healthy series.

    tau=3.0/patience=3 measured 0/60 trips across seeded flat series at production
    noise; tau=1.5/patience=2 measured 45/60. Enabling the level criterion adds no
    false positives at omega=0.10 (omega=0.05 measured 26/60).
    """
    for seed in range(12):
        trip, _, _ = replay(noisy(lambda i: 0.12, 200, seed=seed))
        assert trip is None, f'flat healthy series tripped at step {trip} (seed {seed})'


def test_sustained_rise_is_caught_within_the_run():
    """A +20% sustained rise must be caught well inside a 400-step budget."""
    trips = []
    for seed in range(12):
        trip, _, _ = replay(noisy(lambda i: 0.12 * (1.0 + 0.20 * i / 199), 200, seed=seed))
        assert trip is not None, f'a +20% sustained rise must trip (seed {seed})'
        trips.append(trip)
    assert max(trips) < 200


# --------------------------------------------------------------------------- #
# The z-path still catches what it was meant to catch.
# --------------------------------------------------------------------------- #

def test_sudden_spike_still_trips_the_zscore_path():
    series = noisy(lambda i: 0.12, 40) + [0.60] * 6 + noisy(lambda i: 0.12, 20, seed=7)
    trip, max_z, _ = replay(series, relative_threshold=0.0)
    assert trip is not None, f'a 5x spike must trip the z-path, max z {max_z:.2f}'
    assert trip >= 40, f'the z-path must not fire on the pre-spike noise, fired at {trip}'


def test_patience_above_one_is_satisfiable():
    """With a self-absorbing baseline, observations 2..N score below the first.

    Regression guard for the frozen-baseline fix: without it, `patience=3` on a
    sustained excursion could not be satisfied.
    """
    series = noisy(lambda i: 0.12, 40) + [0.30] * 10
    trip, _, _ = replay(series, relative_threshold=0.0)
    assert trip is not None, 'patience=3 must be reachable on a sustained excursion'


def test_scored_sample_is_excluded_from_its_own_baseline():
    """z must be computed against the PRECEDING window.

    A constant series followed by one outlier: if the outlier were inside its own
    window, abs(z) would be bounded by sqrt(w-1); scored against the preceding
    (zero-variance) window it is far larger.
    """
    gate = build_gate(zscore_w=30, relative_threshold=0.0, tau=1e9)
    for step, value in enumerate([1.0] * 31):
        gate.update_trigger_state(step=step, g0_norm=value, g_slow_norm=value,
                                  stat_override=value)
    z, _, _ = gate.update_trigger_state(step=31, g0_norm=2.0, g_slow_norm=2.0,
                                        stat_override=2.0)
    assert z > 30.0, (
        f'outlier scored against a zero-variance preceding window should give a huge z, got {z}')


# --------------------------------------------------------------------------- #
# False positives: a real, healthy production run must not be shut off.
# --------------------------------------------------------------------------- #

def test_healthy_production_runs_do_not_trip():
    logged = load_logged_entropy()
    if not logged:
        pytest.skip('no train.log with a long enough entropy series available')
    for name, values in logged:
        # Healthy RL entropy decays; none of these runs should be gated off.
        assert sum(values[-10:]) / 10 <= sum(values[:10]) / 10, (
            f'{name} is not a healthy (non-rising) reference series')
        trip, max_z, _ = replay(values)
        assert trip is None, (
            f'{name}: healthy run tripped at step {trip} (max z {max_z:.2f}) -- false positive')
