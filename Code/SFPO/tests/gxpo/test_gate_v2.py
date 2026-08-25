#!/usr/bin/env python3
"""Tests for GXPO gate v2 additions: cosine shutoff mode, robust z-score,
minimum post-warmup observation age. Run: python3 tests/gxpo/test_gate_v2.py"""
import importlib.util
from pathlib import Path

# Load gxpo_state.py directly by path (mirrors test_gxpo_parity.py): importing the
# verl package would require tensordict etc., absent in CPU-only check environments.
GXPO_STATE_PATH = Path(__file__).resolve().parents[2] / 'verl' / 'workers' / 'actor' / 'gxpo_state.py'
_spec = importlib.util.spec_from_file_location('production_gxpo_state', GXPO_STATE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GXPOState = _mod.GXPOState


def feed(state, observations, start_step=50):
    """Feed observations as enabled outer steps; return list of triggered flags."""
    out = []
    for i, obs in enumerate(observations):
        step = start_step + i if state.warmup_steps else i
        state.is_enabled(step)
        _, _, trig = state.update_trigger_state(step=step, g0_norm=obs, g_slow_norm=obs,
                                                allow_trigger=True,
                                                stat_override=(obs if state.shutoff_mode == 'cosine' else None))
        out.append(trig)
    return out


# 1. cosine mode trips on sustained disagreement
s = GXPOState(tau=2.0, omega=0.1, zscore_w=10, warmup_steps=0, trigger_patience=2,
              shutoff_mode='cosine', fallback_mode='permanent', min_post_warmup_obs=0)
stable = [1 - 0.95] * 12            # |cos|=0.95 -> disagreement 0.05
spike  = [1 - 0.60] * 3             # disagreement jumps to 0.40
trig = feed(s, stable + spike)
assert not any(trig[:12]), 'must stay quiet on healthy cosines'
assert any(trig), 'cosine mode must trip on sustained disagreement'
print('PASS cosine mode trips on sustained disagreement (first trip at obs %d)' % trig.index(True))

# 2. cosine resolve without override must fail loudly
try:
    s.resolve_trigger_observation(g0_norm=1.0, g_slow_norm=1.0)
    raise AssertionError('expected ValueError')
except ValueError:
    print('PASS cosine resolve without override raises ValueError')

# 3. robust z-score resists window contamination; classic does not
def first_trip_z(robust):
    st_ = GXPOState(tau=2.0, omega=0.1, zscore_w=10, warmup_steps=0, trigger_patience=1,
                    shutoff_mode='trajectory_aware', trigger_robust=robust)
    zs = []
    for h in [10.0] * 9 + [200.0] + [60.0]:   # outlier contaminates the window; the
        # follow-up must still register above the floored robust scale (floor=1.0)
        st_.is_enabled(st_.step_count)
        z, _, _ = st_.update_trigger_state(step=st_.step_count, g0_norm=h, g_slow_norm=h)
        zs.append(z)
    return zs[-1]

z_classic, z_robust = first_trip_z(False), first_trip_z(True)
assert abs(z_classic) < 2.0, f'classic z should be diluted by the outlier, got {z_classic}'
assert z_robust > 5.0, f'robust z should still see the spike, got {z_robust}'
print(f'PASS robust vs classic on contaminated window: classic={z_classic:.2f} robust={z_robust:.2f}')

# 4. minimum post-warmup age delays the trip and discards pre-age streaks
s = GXPOState(tau=1.5, omega=0.1, zscore_w=5, warmup_steps=0, trigger_patience=2,
              shutoff_mode='trajectory_aware', min_post_warmup_obs=6)
# early double-spike (obs 1-2) happens before the age floor -> must NOT trip;
# late double-spike (obs 9-10) happens after the floor -> must trip.
series = [30.0] + [300.0, 300.0] + [30.0] * 4 + [300.0, 300.0]
trig = feed(s, series)
assert not any(trig[:7]), f'no trip may happen while the age floor blocks: {trig}'
assert any(trig[7:]), 'gate must trip on the post-age sustained spike'
print('PASS min_post_warmup_obs delays trip (first trip at obs %d)' % trig.index(True))

# 5. backward compatibility: defaults reproduce classic behavior exactly
s_old = GXPOState(tau=2.0, omega=0.1, zscore_w=10, warmup_steps=3,
                  shutoff_mode='trajectory_aware', fallback_mode='permanent')
norms = [30.0] * 10 + [300.0] + [30.0] * 15
trip = None
for step, h in enumerate(norms):
    s_old.is_enabled(step)
    _, _, fired = s_old.update_trigger_state(step=step, g0_norm=h, g_slow_norm=h)
    if fired and trip is None:
        trip = step
assert trip == 10, f'default behavior changed! trip at {trip}, expected 10'
print('PASS backward compatibility: default gate trips at step 10 as before')

# 6. hard active-step budget: force-shutoff even when the statistical gate never trips
s = GXPOState(tau=1.0, omega=0.1, zscore_w=5, warmup_steps=0, trigger_patience=2,
              shutoff_mode='trajectory_aware', max_active_steps=5)
flat = [10.0] * 20                    # perfectly stable -> classic gate never fires
trig_budget = []
for i, h in enumerate(flat):
    on = s.is_enabled(i)
    if on:
        s.update_trigger_state(step=i, g0_norm=h, g_slow_norm=h)
    trig_budget.append(on)
assert all(trig_budget[:5]), 'budget must allow the first 5 steps'
assert not any(trig_budget[5:]), 'budget must disable extrapolation at step 5 and stay off'
assert s.budget_stop is True and s.trigger_index == 5
# default (no budget) keeps old behavior: gate alone decides
s2 = GXPOState(tau=1.0, omega=0.1, zscore_w=5, warmup_steps=0, trigger_patience=99,
               shutoff_mode='trajectory_aware')
on_all = [s2.is_enabled(i) for i in range(20)]
assert all(on_all), 'without a budget, stable runs must stay enabled'
print('PASS max_active_steps hard budget bounds 3-pass runtime; default unchanged')

print('ALL GATE V2 CHECKS PASSED')
