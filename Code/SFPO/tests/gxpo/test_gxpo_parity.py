"""GXPO verl-port parity checks (CPU only, no verl runtime deps).

Run: python tests/gxpo/test_gxpo_parity.py

Checks:
  1. GXPOState trigger gate is bit-identical to the reference CCGRPOState
     (reference class source is extracted by AST from legacy_impl.py when
     available; skipped otherwise).
  2. The verl per-parameter in-place ratio/scale/reposition arithmetic matches
     the reference flat-vector formulation element-wise.
  3. Corollary 2 sanity: on a diagonal quadratic with plain SGD and alpha=1,
     one GXPO outer step lands on the K+1-step plain-GD point.
  4. The Table-6 explicit Horner geometric sum matches the closed form.
"""

import ast
import importlib.util
import os
import sys

import torch

REPO = os.path.join(os.path.dirname(__file__), '..', '..')
REFERENCE_IMPL = os.environ.get(
    'GXPO_REFERENCE_IMPL',
    os.path.expanduser('~/inside-model/gxpo_ral/legacy_impl.py'))

DELTA = 1e-8


def load_gxpo_state():
    path = os.path.join(REPO, 'verl', 'workers', 'actor', 'gxpo_state.py')
    spec = importlib.util.spec_from_file_location('gxpo_state', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GXPOState


def load_reference_ccgrpo_state():
    """Extract the CCGRPOState class from legacy_impl.py without importing the module."""
    if not os.path.exists(REFERENCE_IMPL):
        return None
    tree = ast.parse(open(REFERENCE_IMPL).read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CCGRPOState')
    namespace = {'Optional': __import__('typing').Optional}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), REFERENCE_IMPL, 'exec'), namespace)
    return namespace['CCGRPOState']


def reference_ratio_scale(g0, g1, K, delta):
    """Flat-vector arithmetic exactly as legacy_impl.py lines 6502-6511."""
    sign_g0 = torch.sign(g0)
    sign_g0 = torch.where(sign_g0 == 0, torch.ones_like(sign_g0), sign_g0)
    g0_safe = sign_g0 * torch.clamp(torch.abs(g0), min=delta)
    r = g1 / g0_safe
    r = torch.clamp(r, -2.0, 3.0)
    r = torch.where(torch.isfinite(r), r, torch.ones_like(r))
    s_k = (1.0 - torch.pow(r, K)) / (1.0 - r + delta)
    s_2 = (1.0 - torch.pow(r, 2)) / (1.0 - r + delta)
    scale = torch.clamp(s_k / (s_2 + delta), 1.0, K / 2.0 + 1.0)
    return r, scale


def verl_ratio_scale(g0b, g1b, K, delta):
    """Per-parameter in-place arithmetic exactly as dp_actor._gxpo_minibatch_step."""
    sgn = torch.where(g0b >= 0, 1.0, -1.0)
    r = g1b / (g0b.abs().clamp(min=delta) * sgn)
    r.clamp_(-2.0, 3.0).nan_to_num_(nan=1.0)
    one_minus_r = 1.0 - r
    s_k = (1.0 - r.pow(K)) / (one_minus_r + delta)
    s_2 = (1.0 - r * r) / (one_minus_r + delta)
    scale = (s_k / (s_2 + delta)).clamp_(1.0, K / 2.0 + 1.0)
    return r, scale


def test_trigger_gate_parity():
    Reference = load_reference_ccgrpo_state()
    if Reference is None:
        print('SKIP trigger-gate parity (reference impl not found)')
        return
    GXPOState = load_gxpo_state()
    torch.manual_seed(0)
    for mode in ('trajectory_aware', 'legacy_g0', 'never'):
        ours = GXPOState(K=5, alpha=0.5, delta=DELTA, tau=0.5, omega=0.1, shutoff_mode=mode)
        ref = Reference(K=5, alpha=0.5, delta=DELTA, tau=0.5, omega=0.1, shutoff_mode=mode)
        for step in range(200):
            g0 = float(torch.rand(1)) * 0.05
            gslow = float(torch.rand(1)) * 0.05 + (0.5 if step == 150 else 0.0)
            z_a, s_a, t_a = ours.update_trigger_state(step=step, g0_norm=g0, g_slow_norm=gslow)
            z_b, s_b, t_b = ref.update_trigger_state(step=step, g0_norm=g0, g_slow_norm=gslow)
            assert (z_a, s_a, t_a) == (z_b, s_b, t_b), f'{mode} step {step}: {(z_a, s_a, t_a)} != {(z_b, s_b, t_b)}'
            assert ours.trigger_index == ref.trigger_index
    print('PASS trigger-gate parity (3 shutoff modes, 200 steps each)')


def test_ratio_scale_parity():
    torch.manual_seed(1)
    for K in (3, 5, 10):
        g0 = torch.randn(10000, dtype=torch.float32) * 1e-3
        g1 = torch.randn(10000, dtype=torch.float32) * 1e-3
        g0[::97] = 0.0  # exercise the sign(0) -> +1 branch
        g1[::53] = float('nan')  # exercise the non-finite branch
        r_ref, scale_ref = reference_ratio_scale(g0.clone(), g1.clone(), K, DELTA)
        r_new, scale_new = verl_ratio_scale(g0.clone(), g1.clone(), K, DELTA)
        assert torch.equal(r_ref, r_new), f'K={K}: retention ratios differ'
        assert torch.allclose(scale_ref, scale_new, atol=0, rtol=0), f'K={K}: scales differ'
    print('PASS ratio/scale parity (K in {3,5,10}, zeros + NaNs exercised)')


def gxpo_outer_step_gd(theta0, h, eta, K, alpha, delta):
    """One GXPO outer step on L = 0.5 * sum(h * theta^2) under plain GD."""
    grad = lambda th: h * th
    theta = theta0.clone()
    g0 = grad(theta)
    theta = theta - eta * g0
    g1 = grad(theta)
    theta = theta - eta * g1
    _, scale = verl_ratio_scale(g0.clone(), g1.clone(), K, delta)
    theta = theta0 + alpha * scale * (theta - theta0)
    g_slow = grad(theta)
    return theta - eta * g_slow


def test_corollary2_diagonal_quadratic():
    torch.manual_seed(2)
    h = torch.rand(1000, dtype=torch.float64) * 0.9 + 0.05  # h_i > 0, eta*h_i <= 1
    eta = 1.0
    theta0 = torch.randn(1000, dtype=torch.float64)
    for K in (3, 5, 10):
        theta_gxpo = gxpo_outer_step_gd(theta0, h, eta, K, alpha=1.0, delta=1e-14)
        theta_gd = theta0.clone()
        for _ in range(K + 1):
            theta_gd = theta_gd - eta * h * theta_gd
        err = (theta_gxpo - theta_gd).abs().max().item()
        assert err < 1e-9, f'K={K}: GXPO vs {K + 1}-step GD max err {err:.3e}'
    print('PASS Corollary 2 (one GXPO step == K+1 plain-GD steps on diagonal quadratic)')


def test_explicit_sum_matches_closed_form():
    torch.manual_seed(3)
    for K in (3, 5, 10):
        r = torch.empty(10000, dtype=torch.float64).uniform_(-1.9, 2.9)
        r = r[(r - 1.0).abs() > 1e-3]  # closed form is stabilized near r=1
        r = r[(r + 1.0).abs() > 1e-3]
        s_expl = torch.ones_like(r)
        for _ in range(K - 1):  # Horner accumulation, as in dp_actor diagnostics
            s_expl.mul_(r).add_(1.0)
        scale_expl = s_expl / (1.0 + r)
        s_k = (1.0 - r.pow(K)) / (1.0 - r + DELTA)
        s_2 = (1.0 - r * r) / (1.0 - r + DELTA)
        scale_closed = s_k / (s_2 + DELTA)
        assert torch.allclose(scale_expl, scale_closed, rtol=1e-5, atol=1e-7), f'K={K}'
    print('PASS explicit Horner sum matches closed-form S_K/S_2')


if __name__ == '__main__':
    test_trigger_gate_parity()
    test_ratio_scale_parity()
    test_corollary2_diagonal_quadratic()
    test_explicit_sum_matches_closed_form()
    print('ALL GXPO PARITY CHECKS PASSED')
