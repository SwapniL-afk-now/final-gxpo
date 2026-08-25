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

# ---------------------------------------------------------------- adversarial additions
import random

def ref_robust_scale(hist):
    """Reference median/MAD implementation incl. the production sigma floor."""
    ordered = sorted(hist)
    mid = len(ordered) // 2
    med = (ordered[mid - 1] + ordered[mid]) / 2.0 if len(ordered) % 2 == 0 else ordered[mid]
    dev = sorted(abs(v - med) for v in hist)
    m = len(dev) // 2
    mad = (dev[m - 1] + dev[m]) / 2.0 if len(dev) % 2 == 0 else dev[m]
    return med, max(1.4826 * mad, 0.10 * abs(med))

# A7. robust sigma floor: near-constant window + tiny wobble keeps z sane
s = GXPOState(shutoff_mode='trajectory_aware', warmup_steps=0, zscore_w=30,
              trigger_robust=True, trigger_patience=2)
max_z = 0.0
for i in range(300):
    obs = 5.2 if i % 30 == 29 else 5.0
    z, _, _ = s.update_trigger_state(step=i, g0_norm=obs, g_slow_norm=obs)
    max_z = max(max_z, abs(z))
assert max_z <= 50.0, f'robust floor failed to bound z: {max_z}'
assert abs(s.sigma - 0.5) < 1e-9, f'sigma floor should be 0.10*median=0.5, got {s.sigma}'
assert s.trigger_index == float('inf'), 'wobble must never trip a healthy series'
print(f'PASS robust sigma floor bounds z on near-constant windows (max|z|={max_z:.3g})')

# A8. MAD odd/even window parity matches the reference scale exactly
base = [100.0, 62.0, 141.0, 83.0, 118.0, 96.0, 104.0, 71.0, 129.0, 88.0]  # asymmetric devs
for w in (10, 11, 20, 21, 30, 31):
    hist = (base * ((w // len(base)) + 1))[:w]
    st_ = GXPOState(shutoff_mode='trajectory_aware', warmup_steps=0, zscore_w=w,
                    trigger_robust=True)
    st_.trigger_history = list(hist)
    st_.update_stats(12.0)
    rmu, rsig = ref_robust_scale(hist)
    assert st_.mu == rmu and abs(st_.sigma - rsig) < 1e-9, \
        f'w={w}: mu/sigma {st_.mu}/{st_.sigma} != reference {rmu}/{rsig}'
print('PASS MAD scale matches reference for even and odd windows (10/11/20/21/30/31)')

# A9. stat_override fully bypasses the norms in every mode
for mode in ('trajectory_aware', 'cosine'):
    st_ = GXPOState(shutoff_mode=mode, warmup_steps=0, zscore_w=5)
    _, stat, _ = st_.update_trigger_state(step=0, g0_norm=0.0, g_slow_norm=0.0, stat_override=0.5)
    assert stat == 0.5, f'{mode}: override not threaded, got {stat}'
    _, stat2, _ = st_.update_trigger_state(step=1, g0_norm=1e9, g_slow_norm=1e18, stat_override=0.7)
    assert stat2 == 0.7 and st_.trigger_history == [0.5, 0.7], \
        f'{mode}: norms leaked into observation: {st_.trigger_history}'
print('PASS stat_override ignores g0_norm/g_slow_norm (trajectory_aware and cosine)')

# A10. age floor delays a deterministic trip to scored #7 (patience=2, age=6);
# tau=-100 makes every scored observation a violation so any age-floor leak trips early.
st_ = GXPOState(shutoff_mode='trajectory_aware', warmup_steps=0, zscore_w=3,
                trigger_patience=2, min_post_warmup_obs=6, tau=-100.0)
trigs = []
for i in range(12):
    st_.is_enabled(i)
    _, _, t = st_.update_trigger_state(step=i, g0_norm=30.0, g_slow_norm=30.0)
    trigs.append(t)
assert not any(trigs[:6]), f'trip before min_post_warmup_obs: {trigs}'
assert trigs[6], f'expected deterministic trip at scored #7, got {trigs}'
print('PASS min_post_warmup_obs discards pre-age streaks deterministically')
st_r = GXPOState(shutoff_mode='trajectory_aware', warmup_steps=0, zscore_w=3,
                 trigger_patience=2, min_post_warmup_obs=6, tau=-100.0)
st_r.update_trigger_state(step=0, g0_norm=30.0, g_slow_norm=30.0)
st_r.update_trigger_state(step=1, g0_norm=30.0, g_slow_norm=30.0)
assert st_r.post_warmup_scored == 2
st_r.reset_trigger_baseline()
assert st_r.post_warmup_scored == 0 and not st_r.trigger_history
assert st_r.mu == 1.0 and st_r.sigma == 1.0
print('PASS reset_trigger_baseline zeroes post_warmup_scored and history')

# A11. cosine end-to-end mini-replay: healthy quiet for 300 steps, degraded trips fast
def cosine_replay(seed, degrade_at=None, n=300):
    rng = random.Random(seed)
    st_ = GXPOState(K=5, alpha=0.5, delta=1e-8, tau=2.0, omega=0.1, zscore_w=30,
                    shutoff_mode='cosine', warmup_steps=0, fallback_mode='permanent',
                    trigger_patience=3, min_post_warmup_obs=0)
    for i in range(n):
        lvl = 0.26 if (degrade_at is not None and i >= degrade_at) else 0.05
        disagreement = max(0.0, rng.gauss(lvl, 0.01))
        st_.is_enabled(i)
        _, _, fired = st_.update_trigger_state(step=i, g0_norm=1.0, g_slow_norm=1.0,
                                               stat_override=disagreement)
        if fired:
            return i
    return None

for seed in range(20):
    assert cosine_replay(seed, None) is None, f'healthy series tripped at seed {seed}'
trip = cosine_replay(0, degrade_at=100)
assert trip is not None and trip - 100 <= 35, f'degraded series tripped too late: {trip}'
print('PASS cosine replay: 20 seeds quiet when healthy (300 steps); '
      f'degraded trips at step {trip} ({trip - 100} obs after onset)')

print('ALL GATE V2 CHECKS PASSED')
