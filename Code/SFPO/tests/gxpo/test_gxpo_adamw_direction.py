"""Optimizer-aware AdamW retention for GXPO: r = d1 / d0, not g1 / g0.

Run directly with:
    PYTHONPATH=Code/SFPO python tests/gxpo/test_gxpo_adamw_direction.py

These tests import the exact production helpers used by ``dp_actor.py``; they do
not maintain a second implementation of the arithmetic.
"""

import ast
import copy
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from verl.workers.actor.dp_actor import DataParallelPPOActor  # noqa: E402
from verl.workers.actor.gxpo_state import (  # noqa: E402
    RetentionKind, adamw_direction, adamw_direction_from_step,
    compute_gxpo_adamw_direction_retention_scale,
    compute_gxpo_retention_scale, compute_gxpo_update_retention_scale)
from verl.workers.actor.optimizer_transaction import snapshot_optimizer_state  # noqa: E402
from verl.workers.muon import Muon  # noqa: E402

ACTOR_PATH = REPO / 'verl' / 'workers' / 'actor' / 'dp_actor.py'
K = 10
DELTA = 1e-8

LR = 3e-3
WD = 0.04
BETAS = (0.9, 0.999)
EPS = 1e-8


def _fresh_param(seed=0, n=64, dtype=torch.float32, scale=1.0):
    torch.manual_seed(seed)
    return torch.nn.Parameter(torch.randn(n, dtype=dtype) * scale)


def _adamw(param, lr=LR, weight_decay=WD):
    return torch.optim.AdamW([param], lr=lr, betas=BETAS, eps=EPS,
                             weight_decay=weight_decay)


def _step(param, optimizer, grad):
    """One real optimizer step; returns (theta_before, theta_after)."""
    before = param.detach().clone()
    param.grad = grad.clone()
    optimizer.step()
    return before, param.detach().clone()


def _textbook_adamw_direction(optimizer, param):
    """m_hat / (sqrt(v_hat) + eps) from torch's own state, for cross-checking."""
    state = optimizer.state[param]
    step = float(state['step'])
    beta1, beta2 = BETAS
    m_hat = state['exp_avg'] / (1 - beta1**step)
    v_hat = state['exp_avg_sq'] / (1 - beta2**step)
    return m_hat / (v_hat.sqrt() + EPS)


# --------------------------------------------------------------------------
# Test A -- the reconstructed direction IS the direction the optimizer applied
# --------------------------------------------------------------------------

def test_reconstructed_direction_matches_real_adamw_update():
    """d = ((1 - lr*wd) * theta_before - theta_after) / lr equals m_hat/(sqrt(v_hat)+eps).

    Exercised with nonzero weight decay, nonzero first and second moments, and
    step > 1 -- i.e. every term that the legacy raw-gradient ratio ignores.
    """
    # FP64 first: there the identity is exact algebra, so any disagreement
    # would be a modelling error rather than round-off.
    param = _fresh_param(seed=1, dtype=torch.float64)
    optimizer = _adamw(param)

    torch.manual_seed(2)
    grads = [(torch.randn(64) * scale).double() for scale in (1.0, 0.3, 4.0, 0.7)]
    for index, grad in enumerate(grads):
        before, after = _step(param, optimizer, grad)
        step_number = index + 1
        assert int(optimizer.state[param]['step'].item()) == step_number

        reconstructed = adamw_direction(before, after, LR, WD)
        expected = _textbook_adamw_direction(optimizer, param)
        assert torch.allclose(reconstructed, expected, rtol=1e-12, atol=0.0), (
            f'step {step_number}: max abs err '
            f'{(reconstructed - expected).abs().max().item():.3e}')

        if step_number > 1:
            # The moments are genuinely live: without bias correction and the
            # second moment, the reconstruction would not match.
            assert optimizer.state[param]['exp_avg'].abs().sum() > 0
            assert optimizer.state[param]['exp_avg_sq'].abs().sum() > 0

    # Now the precision production actually runs at. c*theta_t - theta_{t+1} is a
    # difference of two nearly equal FP32 numbers, so d inherits a relative error
    # of about eps(theta) / (lr * |d|). At the magnitudes above that is ~1e-5.
    param32 = _fresh_param(seed=1)
    optimizer32 = _adamw(param32)
    torch.manual_seed(2)
    for scale in (1.0, 0.3, 4.0, 0.7):
        before, after = _step(param32, optimizer32, torch.randn(64) * scale)
    got = adamw_direction(before, after, LR, WD)
    assert torch.allclose(got, _textbook_adamw_direction(optimizer32, param32),
                          rtol=1e-3, atol=1e-4)


def test_fp32_reconstruction_is_accurate_enough_at_production_magnitudes():
    """lr=1e-6 on weights ~1e-2 is the worst cancellation GXPO actually meets.

    d ~ O(1) for AdamW, so lr*d ~ 1e-6 while an FP32 ulp at theta ~ 1e-2 is
    ~1e-9: the reconstruction keeps about three significant digits, and the
    retention ratio r = d1/d0 inherits that. Three digits is far more than
    S_K(r)/S_2(r) can distinguish, which is why the displacement route is usable
    at production LRs at all.
    """
    lr, wd = 1e-6, 1e-2
    param = _fresh_param(seed=20, scale=1e-2)
    reference = torch.nn.Parameter(param.detach().double().clone())
    optimizer = _adamw(param, lr=lr, weight_decay=wd)
    reference_optimizer = _adamw(reference, lr=lr, weight_decay=wd)

    torch.manual_seed(21)
    grads = [torch.randn(64) * scale for scale in (1.0, 0.4, 2.5)]
    for grad in grads:
        before, after = _step(param, optimizer, grad)
        ref_before, ref_after = _step(reference, reference_optimizer, grad.double())
    d_fp32 = adamw_direction(before, after, lr, wd)
    d_fp64 = adamw_direction(ref_before, ref_after, lr, wd)
    relative = ((d_fp32 - d_fp64).abs() / d_fp64.abs().clamp_min(1e-12)).max().item()
    assert relative < 1e-2, relative


def test_step_form_is_exact_and_beats_the_endpoint_form_at_production_lr():
    """d0 is built from u0, which GXPO already holds; forming theta0 + u0 to
    subtract theta0 back off is where FP32 loses digits.

    Both forms are the same quantity -- asserted in FP64 -- but at lr=1e-6 with
    |theta| ~ 1e-2 the step |u| ~ 1e-7 lands only ~200 ulps above theta's, so the
    round-trip through theta0 + u0 costs real precision in d0, the denominator of
    r = d1 / d0. The step form never forms that sum.
    """
    lr, wd = 1e-6, 1e-2
    torch.manual_seed(40)
    theta0 = torch.randn(4096, dtype=torch.float64) * 2e-2
    u0 = torch.randn(4096, dtype=torch.float64) * (lr * 0.26)  # |d| ~ 0.26, as measured

    # Error is measured against RMS(d0), not coordinatewise: a coordinate where u0
    # is near zero has no significant digits to lose in the first place, and the
    # retention gate already treats those as inactive at 1e-3 * RMS.
    exact = adamw_direction_from_step(theta0, u0, lr, wd)
    rms = exact.square().mean().sqrt()
    err = lambda d: ((d.double() - exact).abs().max() / rms).item()

    # Same quantity: the endpoint form's FP64 cancellation is ~6e-11, not zero.
    assert err(adamw_direction(theta0, theta0 + u0, lr, wd)) < 1e-9

    theta0_32, u0_32 = theta0.float(), u0.float()
    step_err = err(adamw_direction_from_step(theta0_32, u0_32, lr, wd))
    endpoint_err = err(adamw_direction(theta0_32, theta0_32 + u0_32, lr, wd))
    # Measured: 1.1e-2 against 2.9e-7. This is the precision the change recovers.
    assert endpoint_err > 1e-3, endpoint_err
    assert step_err < endpoint_err / 1000, (step_err, endpoint_err)


def test_step_form_rejects_a_non_positive_lr_like_the_endpoint_form():
    theta0 = torch.zeros(4)
    with pytest.raises(ValueError):
        adamw_direction_from_step(theta0, torch.zeros(4), 0.0, 1e-2)


def test_reconstruction_holds_for_the_muon_owned_adamw_branch():
    """Muon's own AdamW branch uses different state keys and a different
    bias-correction ordering; the displacement reconstruction is agnostic to both."""
    torch.manual_seed(30)
    matrix = torch.nn.Parameter(torch.randn(8, 8))
    vector = torch.nn.Parameter(torch.randn(32))
    optimizer = Muon(lr=LR, wd=WD, muon_params=[matrix], adamw_params=[vector],
                     adamw_betas=(0.9, 0.95), adamw_eps=1e-8)

    torch.manual_seed(3)
    for scale in (1.0, 0.25, 3.0):
        matrix.grad = torch.randn(8, 8)
        vector.grad = torch.randn(32) * scale
        before = vector.detach().clone()
        optimizer.step()
        after = vector.detach().clone()

        state = optimizer.state[vector]
        # Muon's branch: normalized = m / (eps + sqrt(v)); correction =
        # (1 - b1^t) / sqrt(1 - b2^t); write-back is -lr/correction * normalized.
        beta1, beta2 = 0.9, 0.95
        step = state['step']
        correction = (1 - beta1**step) / (1 - beta2**step)**0.5
        expected = (state['moment1'] / (1e-8 + state['moment2'].sqrt())) / correction

        reconstructed = adamw_direction(before, after, LR, WD)
        assert torch.allclose(reconstructed, expected, rtol=1e-3, atol=1e-4)
        # Muon's keys really are the non-standard ones, so a state-key-based
        # implementation would have had to special-case this optimizer.
        assert 'exp_avg' not in state and 'moment1' in state


def test_direction_is_undefined_at_zero_learning_rate():
    param = _fresh_param(seed=4)
    with pytest.raises(ValueError, match='undefined at lr=0'):
        adamw_direction(param.detach(), param.detach(), 0.0, WD)


def test_direction_rms_skips_zero_learning_rate_parameters():
    """The actor-level RMS helper must neutralize schedule-zero groups."""
    param = _fresh_param(seed=41)
    optimizer = torch.optim.AdamW([{'params': [param], 'lr': 0.0, 'weight_decay': WD}],
                                  betas=BETAS, eps=EPS)
    actor = _BufferActor(optimizer, [param])
    theta0 = [param.detach().clone()]
    u0_bufs = [torch.zeros_like(param)]

    assert actor._gxpo_adamw_direction_rms([0], theta0, u0_bufs, [0.0], [WD]) == {}

def test_per_group_lr_and_weight_decay_are_respected():
    """Two groups with different lr/wd: each must be inverted with its own values."""
    a = _fresh_param(seed=5, n=32)
    b = _fresh_param(seed=6, n=32)
    optimizer = torch.optim.AdamW(
        [{'params': [a], 'lr': 1e-3, 'weight_decay': 0.0},
         {'params': [b], 'lr': 5e-2, 'weight_decay': 0.2}],
        betas=BETAS, eps=EPS)
    a.grad, b.grad = torch.randn(32), torch.randn(32)
    before_a, before_b = a.detach().clone(), b.detach().clone()
    optimizer.step()

    for param, before, lr, wd in ((a, before_a, 1e-3, 0.0), (b, before_b, 5e-2, 0.2)):
        got = adamw_direction(before, param.detach(), lr, wd)
        assert torch.allclose(got, _textbook_adamw_direction(optimizer, param),
                              rtol=1e-3, atol=1e-4)

    # Using the wrong group's hyperparameters is detectably wrong, so the
    # per-group lookup is load-bearing rather than cosmetic.
    wrong = adamw_direction(before_b, b.detach(), 1e-3, 0.0)
    assert not torch.allclose(wrong, _textbook_adamw_direction(optimizer, b),
                              rtol=1e-2, atol=1e-3)


# --------------------------------------------------------------------------
# Test B -- two-step retention from real probe displacements
# --------------------------------------------------------------------------

def _two_probe_steps(g0, g1, lr=LR, wd=WD, seed=7, warm_steps=3, warm_grad=None):
    """Run `warm_steps` warmup steps, then the two GXPO probe steps.

    Returns (theta0, theta1, theta2, d0, d1) with real optimizer state carried in.
    """
    param = _fresh_param(seed=seed)
    optimizer = _adamw(param, lr=lr, weight_decay=wd)
    torch.manual_seed(seed + 100)
    for _ in range(warm_steps):
        _step(param, optimizer,
              torch.randn_like(param) if warm_grad is None else warm_grad)

    theta0, theta1 = _step(param, optimizer, g0)
    _, theta2 = _step(param, optimizer, g1)
    d0 = adamw_direction(theta0, theta1, lr, wd)
    d1 = adamw_direction(theta1, theta2, lr, wd)
    return theta0, theta1, theta2, d0, d1


def test_two_step_adamw_retention_matches_the_production_helper():
    torch.manual_seed(8)
    g0 = torch.randn(64)
    g1 = torch.randn(64) * 2.0
    theta0, theta1, theta2, d0, d1 = _two_probe_steps(g0, g1)

    ratio, scale, active, clipped = compute_gxpo_adamw_direction_retention_scale(
        d0, d1, K, DELTA)

    # The helper's ratio is d1/d0 wherever the coordinate is active.
    reference = (d1 / d0).clamp(-2.0, 3.0)
    assert torch.allclose(ratio[active], reference[active], rtol=1e-5, atol=1e-6)
    assert torch.isfinite(scale).all()
    assert (scale >= 1.0).all() and (scale <= K / 2.0 + 1.0).all()

    # The actor never stores theta1; it recovers it from u0 in the g1 slot.
    u0 = theta1 - theta0
    recovered_d0 = adamw_direction(theta0, theta0 + u0, LR, WD)
    recovered_d1 = adamw_direction(theta0 + u0, theta2, LR, WD)
    assert torch.allclose(recovered_d0, d0, rtol=0, atol=0)
    assert torch.allclose(recovered_d1, d1, rtol=0, atol=0)


# --------------------------------------------------------------------------
# Test C -- regression: this is NOT the raw gradient ratio
# --------------------------------------------------------------------------

def test_adamw_retention_is_not_the_raw_gradient_ratio():
    """A large gradient jump on top of live Adam moments separates the two rules.

    g1 = 4 * g0 makes the raw-gradient estimator report r ~ 4 (clipped to 3) on
    every active coordinate. AdamW moves along m_hat/(sqrt(v_hat)+eps): the first
    moment absorbs only 10% of the jump and the second moment grows in the
    denominator, so the true retention is far closer to 1. Reporting r ~ 3 here
    would extrapolate the K-step trajectory to a displacement AdamW will never
    produce.
    """
    torch.manual_seed(9)
    # A settled, sign-stable regime: the moments are warmed on g0 itself, so d0
    # is a well-conditioned O(1) denominator everywhere and the comparison
    # isolates the estimator rather than near-zero-crossing noise.
    g0 = torch.randn(64).abs() + 0.5
    g1 = 4.0 * g0
    theta0, theta1, theta2, d0, d1 = _two_probe_steps(g0, g1, warm_grad=g0)

    grad_ratio, grad_scale, grad_active, _ = compute_gxpo_retention_scale(
        g0, g1, K, DELTA)
    dir_ratio, dir_scale, dir_active, _ = compute_gxpo_adamw_direction_retention_scale(
        d0, d1, K, DELTA)

    assert grad_active.all() and dir_active.all()
    # Raw-gradient rule: pinned at the clip ceiling.
    assert torch.allclose(grad_ratio, torch.full_like(grad_ratio, 3.0))
    # Optimizer-direction rule: nowhere near it.
    assert dir_ratio.max() < 2.0, dir_ratio.max().item()
    assert dir_ratio.mean() < 1.6, dir_ratio.mean().item()
    assert dir_ratio.min() > 0.5, dir_ratio.min().item()
    # ... and the resulting extrapolation differs materially, which is the whole
    # point: the two arms are not a relabelling of each other.
    assert (grad_scale.mean() - dir_scale.mean()).abs() > 0.5

    # Direct statement of the regression: the production helper returns d1/d0.
    assert not torch.allclose(dir_ratio, grad_ratio, rtol=0.2, atol=0.2)
    expected = (d1 / d0).clamp(-2.0, 3.0)
    assert torch.allclose(dir_ratio, expected, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------
# Test D -- neutral behavior on a vanished denominator
# --------------------------------------------------------------------------

def test_tiny_or_zero_d0_is_neutral_and_finite():
    d0 = torch.tensor([1.0, 0.0, 1e-12, -1.0, 1.0, 1.0, 2.0])
    d1 = torch.tensor([2.0, 5.0, 5.0, -2.0, float('nan'), float('inf'), 0.0])
    ratio, scale, active, clipped = compute_gxpo_adamw_direction_retention_scale(
        d0, d1, K, DELTA)

    assert torch.isfinite(ratio).all() and torch.isfinite(scale).all()
    # 0 and 1e-12 sit far below 1e-3 * RMS(d0): inactive, hence neutral.
    assert not active[1] and not active[2]
    assert ratio[1] == 1.0 and ratio[2] == 1.0
    assert scale[1] == 1.0 and scale[2] == 1.0
    # NaN/Inf numerators on active coordinates are replaced with neutral retention
    # and flagged as clipped.
    assert ratio[4] == 1.0 and ratio[5] == 3.0
    assert clipped[4] and clipped[5]
    assert (scale >= 1.0).all() and (scale <= K / 2.0 + 1.0).all()


def test_zero_learning_rate_leaves_the_actor_on_neutral_retention():
    """The reposition loop must not call adamw_direction at lr == 0."""
    source = ACTOR_PATH.read_text()
    tree = ast.parse(source)
    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == '_gxpo_minibatch_step')
    body = ast.get_source_segment(source, step)
    assert 'if lr0 > 0.0 and lr1 > 0.0:' in body, (
        'the AdamW branch must guard against an LR-warmup step of zero')
    guard_at = body.index('if lr0 > 0.0 and lr1 > 0.0:')
    assert body.index('d0 = adamw_direction_from_step(') > guard_at


# --------------------------------------------------------------------------
# Test E / F -- transactional semantics
# --------------------------------------------------------------------------

def _clone_state(optimizer):
    return {id(p): {k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                    for k, v in state.items()}
            for p, state in optimizer.state.items()}


def _assert_state_equal(actual, expected):
    assert set(actual) == set(expected)
    for key, values in expected.items():
        assert set(actual[key]) == set(values), key
        for name, value in values.items():
            got = actual[key][name]
            if torch.is_tensor(value):
                assert torch.equal(got, value), f'{key}.{name}'
            else:
                assert got == value, f'{key}.{name}'


def test_transactional_probe_state_is_restored_and_the_step_counter_advances_once():
    """Test E: s0 -> s1 -> s2 exists only in the probe; the retained trajectory is
    s0 -> s_next, and AdamW's step counter advances by exactly one per minibatch."""
    param = _fresh_param(seed=11)
    optimizer = _adamw(param)
    torch.manual_seed(12)
    for _ in range(5):                      # warm the moments and the counter
        _step(param, optimizer, torch.randn_like(param))

    before = _clone_state(optimizer)
    step_before = int(optimizer.state[param]['step'].item())
    assert step_before == 5

    transaction = snapshot_optimizer_state(optimizer)
    _step(param, optimizer, torch.randn_like(param))       # probe 1  -> s1
    assert int(optimizer.state[param]['step'].item()) == step_before + 1
    _step(param, optimizer, torch.randn_like(param))       # probe 2  -> s2
    assert int(optimizer.state[param]['step'].item()) == step_before + 2

    transaction.restore()                                   # s2 -> s0
    _assert_state_equal(_clone_state(optimizer), before)
    assert int(optimizer.state[param]['step'].item()) == step_before

    _step(param, optimizer, torch.randn_like(param))        # Pass 3: the ONE
    assert int(optimizer.state[param]['step'].item()) == step_before + 1, (
        'the retained AdamW step counter must advance exactly once per minibatch; '
        'the two probe steps are temporary')


def test_transactional_fast_state_still_advances_through_the_probes():
    """Test F: the fast-state ablation is unchanged -- no rollback, 3x counter."""
    param = _fresh_param(seed=13)
    optimizer = _adamw(param)
    torch.manual_seed(14)
    _step(param, optimizer, torch.randn_like(param))
    step_before = int(optimizer.state[param]['step'].item())

    # transactional_fast_state takes the snapshot (it is the failure-path
    # rollback) but never restores it on the success path.
    snapshot_optimizer_state(optimizer)
    for _ in range(3):
        _step(param, optimizer, torch.randn_like(param))
    assert int(optimizer.state[param]['step'].item()) == step_before + 3


def test_actor_restores_before_pass_three_only_in_transactional_mode():
    source = ACTOR_PATH.read_text()
    tree = ast.parse(source)
    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == '_gxpo_minibatch_step')
    body = ast.get_source_segment(source, step)
    assert "if self.gxpo_optimizer_state_mode == 'transactional':" in body
    assert body.index("optimizer_transaction.restore()") < body.index('# Pass 3: slow correction')
    assert body.count('probe_optimizer_step()\n') == 3


# --------------------------------------------------------------------------
# Test G -- the Muon path is untouched
# --------------------------------------------------------------------------

def test_muon_scalar_retention_is_unchanged():
    """Same u0/u1/K/delta must give the same rho and scale as before the patch.

    The expected values below are the pre-patch algorithm restated in closed
    form, not a copy of the current output.
    """
    torch.manual_seed(15)
    u0 = torch.randn(4, 6) * 1e-6
    u1 = 0.6 * u0 + 0.01 * torch.randn(4, 6) * 1e-6

    rho, scale = compute_gxpo_update_retention_scale(u0, u1, K, DELTA)
    expected_rho = ((u0 * u1).sum() / u0.square().sum()).clamp(-1.0, 1.0)
    expected_scale = (sum(expected_rho**i for i in range(K))
                      / sum(expected_rho**i for i in range(2)))
    assert torch.allclose(rho, expected_rho, rtol=1e-6, atol=0)
    assert torch.allclose(scale, expected_scale, rtol=1e-6, atol=0)
    assert rho.ndim == 0 and scale.ndim == 0, 'Muon retention must stay a per-matrix scalar'

    # Degenerate first step -> rho = 0 -> scale = 1 (neutral), not rho = 1.
    zero_rho, zero_scale = compute_gxpo_update_retention_scale(
        torch.zeros(4, 6), u1, K, DELTA)
    assert zero_rho.item() == 0.0 and zero_scale.item() == 1.0


def test_muon_matrices_never_take_the_coordinatewise_adamw_rule():
    source = ACTOR_PATH.read_text()
    tree = ast.parse(source)
    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == '_gxpo_minibatch_step')
    body = ast.get_source_segment(source, step)
    muon_at = body.index('rho_u, scale = compute_gxpo_update_retention_scale(')
    adamw_at = body.index('compute_gxpo_adamw_direction_retention_scale(')
    # The AdamW rule lives in a strictly later, mutually exclusive branch.
    assert muon_at < adamw_at
    assert 'elif is_adamw_direction:' in body
    assert 'is_update_space = kind == RetentionKind.MUON_UPDATE' in body


# --------------------------------------------------------------------------
# Tests H / I -- retention classification
# --------------------------------------------------------------------------

class _StubActor:
    """Minimal carrier for the classification methods under test.

    They touch only the config, the optimizer, and the GXPO parameter list, so
    binding them to a stub exercises the production code without standing up an
    FSDP model.
    """

    _gxpo_adamw_direction_supported = DataParallelPPOActor._gxpo_adamw_direction_supported
    _gxpo_retention_kinds = DataParallelPPOActor._gxpo_retention_kinds
    _gxpo_u0_slot_mask = staticmethod(DataParallelPPOActor._gxpo_u0_slot_mask)

    def __init__(self, optimizer, params, space):
        self.actor_optimizer = optimizer
        self._gxpo_params = list(params)
        self.config = {'gxpo_retention_space': space}
        self._gxpo_retention_cache = None
        self._gxpo_unsupported_optimizer_warned = False


def _hybrid_muon():
    matrix_a = torch.nn.Parameter(torch.randn(8, 8))
    matrix_b = torch.nn.Parameter(torch.randn(6, 4))
    bias = torch.nn.Parameter(torch.randn(8))
    norm = torch.nn.Parameter(torch.randn(6))
    optimizer = Muon(lr=LR, wd=WD, muon_params=[matrix_a, matrix_b],
                     adamw_params=[bias, norm])
    return optimizer, [matrix_a, matrix_b, bias, norm]


def test_hybrid_muon_classification_under_auto():
    optimizer, params = _hybrid_muon()
    kinds = _StubActor(optimizer, params, 'auto')._gxpo_retention_kinds()
    assert kinds == [RetentionKind.MUON_UPDATE, RetentionKind.MUON_UPDATE,
                     RetentionKind.ADAMW_DIRECTION, RetentionKind.ADAMW_DIRECTION]
    # Every non-legacy kind borrows the g1 slot for u0.
    assert _StubActor._gxpo_u0_slot_mask(kinds) == [True, True, True, True]


def test_hybrid_muon_classification_under_grad_is_legacy():
    optimizer, params = _hybrid_muon()
    actor = _StubActor(optimizer, params, 'grad')
    assert actor._gxpo_retention_kinds() is None, (
        "gxpo_retention_space=grad must keep every parameter on the legacy "
        "gradient estimator (None is the all-legacy fast path)")
    assert _StubActor._gxpo_u0_slot_mask(None) is None


def test_forced_update_space_is_preserved():
    optimizer, params = _hybrid_muon()
    kinds = _StubActor(optimizer, params, 'update')._gxpo_retention_kinds()
    assert kinds == [RetentionKind.MUON_UPDATE] * 4


def test_pure_adamw_classification_under_auto():
    params = [_fresh_param(seed=16), _fresh_param(seed=17, n=8)]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    kinds = _StubActor(optimizer, params, 'auto')._gxpo_retention_kinds()
    assert kinds == [RetentionKind.ADAMW_DIRECTION] * 2

    assert _StubActor(optimizer, params, 'grad')._gxpo_retention_kinds() is None


def test_unknown_optimizer_falls_back_to_legacy_with_a_warning(capsys):
    """auto must never model an unrecognized optimizer as AdamW."""
    params = [_fresh_param(seed=18)]
    for optimizer in (torch.optim.SGD(params, lr=LR),
                      torch.optim.Adam(params, lr=LR)):   # coupled decay: not AdamW
        actor = _StubActor(optimizer, params, 'auto')
        assert actor._gxpo_retention_kinds() is None
        assert 'WARNING' in capsys.readouterr().out


def test_classification_is_cached_and_preserves_parameter_order():
    optimizer, params = _hybrid_muon()
    actor = _StubActor(optimizer, params, 'auto')
    first = actor._gxpo_retention_kinds()
    assert actor._gxpo_retention_kinds() is first, 'classification must be cached'
    assert len(first) == len(params)


def test_invalid_retention_space_is_rejected():
    params = [_fresh_param(seed=19)]
    actor = _StubActor(torch.optim.AdamW(params, lr=LR), params, 'directions')
    with pytest.raises(ValueError, match='auto|grad|update'):
        actor._gxpo_retention_kinds()


# --------------------------------------------------------------------------
# Test J -- no new model-sized buffer
# --------------------------------------------------------------------------

def test_adamw_direction_path_adds_no_persistent_model_sized_buffer():
    """GXPO still allocates exactly theta0/g0/g1, and the AdamW path reuses g1.

    theta1 is never stored: the branch reconstructs it as theta0 + u0 from the g1
    slot, and both directions are derived transiently per parameter. The one
    extra pass (the d0 RMS collective) streams its directions through a generator
    so they are released one at a time.
    """
    source = ACTOR_PATH.read_text()
    tree = ast.parse(source)

    init = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == '_gxpo_init_buffers')
    init_code = '\n'.join(line for line in ast.get_source_segment(source, init).splitlines()
                          if not line.lstrip().startswith('#'))
    assert "('theta0', 'g0', 'g1')" in init_code
    assert 'theta1' not in init_code
    assert init_code.count('torch.empty_like') == 1

    release = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == '_gxpo_release_buffers')
    assert 'self._gxpo_bufs = None' in ast.get_source_segment(source, release)

    rms = next(node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == '_gxpo_adamw_direction_rms')
    rms_code = ast.get_source_segment(source, rms)
    assert 'def directions():' in rms_code and 'yield' in rms_code, (
        'the d0 RMS pass must stream, not materialize a full-model list of directions')

    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == '_gxpo_minibatch_step')
    body = ast.get_source_segment(source, step)
    assert 'theta1 = t0f + u0' in body and 'u0 = g1b.float()' in body, (
        'theta1 must be reconstructed from u0 in the g1 slot, not stored')
    assert 'del theta1' in body and 'del d0, d1' in body


# --------------------------------------------------------------------------
# End-to-end arithmetic: the actor's own helpers, driven through the actor's
# own sequence, on a real AdamW optimizer.
# --------------------------------------------------------------------------

class _BufferActor(_StubActor):
    """Stub carrying the production helpers the reposition loop calls."""

    _gxpo_global_tensor_rms = DataParallelPPOActor._gxpo_global_tensor_rms
    _gxpo_adamw_direction_rms = DataParallelPPOActor._gxpo_adamw_direction_rms
    _gxpo_param_group_hparams = DataParallelPPOActor._gxpo_param_group_hparams

    def __init__(self, optimizer, params, space='auto'):
        super().__init__(optimizer, params, space)
        self.actor_module = None
        self._gxpo_fsdp_invariant_threshold = True


def test_actor_helpers_reproduce_the_target_reposition_on_a_real_adamw_run():
    """theta_tilde = theta0 + alpha * q * (theta2 - theta0) with q = S_K(r)/S_2(r),
    r = d1/d0, reconstructed exactly as the actor does it: theta0 and u0 in the
    g1 slot, theta2 live in the parameter, nothing else retained.
    """
    alpha = 0.3
    torch.manual_seed(40)
    params = [torch.nn.Parameter(torch.randn(48)),
              torch.nn.Parameter(torch.randn(3, 5))]
    optimizer = torch.optim.AdamW(
        [{'params': [params[0]], 'lr': LR, 'weight_decay': WD},
         {'params': [params[1]], 'lr': LR / 2, 'weight_decay': 0.0}],
        betas=BETAS, eps=EPS)
    actor = _BufferActor(optimizer, params)

    kinds = actor._gxpo_retention_kinds()
    assert kinds == [RetentionKind.ADAMW_DIRECTION] * 2
    u0_slots = actor._gxpo_u0_slot_mask(kinds)
    assert u0_slots == [True, True]
    adamw_indices = [i for i, kind in enumerate(kinds)
                     if kind == RetentionKind.ADAMW_DIRECTION]

    # Warm the moments so d0/d1 are genuinely preconditioned, not first-step.
    for _ in range(4):
        for parameter in params:
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    theta0 = [p.detach().clone() for p in params]

    # Probe 1.
    for parameter in params:
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    # u0 goes into the g1 slot; theta1 itself is never stored.
    g1_bufs = [p.detach() - t0 for p, t0 in zip(params, theta0)]
    lrs0, wds0 = actor._gxpo_param_group_hparams()
    assert lrs0 == [LR, LR / 2] and wds0 == [WD, 0.0]

    # Probe 2.
    for parameter in params:
        parameter.grad = torch.randn_like(parameter) * 3.0
    optimizer.step()
    lrs1, wds1 = actor._gxpo_param_group_hparams()

    d0_rms = actor._gxpo_adamw_direction_rms(adamw_indices, theta0, g1_bufs, lrs0, wds0)
    assert set(d0_rms) == {0, 1}

    for index, parameter in enumerate(params):
        t0f = theta0[index].float()
        theta1 = t0f + g1_bufs[index].float()
        d0 = adamw_direction(t0f, theta1, lrs0[index], wds0[index])
        d1 = adamw_direction(theta1, parameter.data.float(), lrs1[index], wds1[index])

        # The RMS helper agrees with a plain local RMS in the single-rank case.
        assert torch.allclose(d0_rms[index], d0.square().mean().sqrt(), rtol=1e-5)

        ratio, scale, active, _ = compute_gxpo_adamw_direction_retention_scale(
            d0, d1, K, DELTA, d0_rms=d0_rms[index])
        assert active.all(), 'a warmed AdamW step should have no inactive coordinates'
        assert torch.allclose(ratio, (d1 / d0).clamp(-2.0, 3.0), rtol=1e-5, atol=1e-6)

        s_k = sum(ratio**i for i in range(K))
        expected_scale = (s_k / (1.0 + ratio)).clamp(1.0, K / 2.0 + 1.0)
        assert torch.allclose(scale, expected_scale, rtol=1e-4, atol=1e-6)

        disp2 = parameter.data.float() - t0f
        theta_tilde = t0f + alpha * scale * disp2
        assert torch.isfinite(theta_tilde).all()
        # The reposition is the existing GXPO rule verbatim; only q changed.
        assert torch.allclose(theta_tilde, t0f + disp2 * (scale * alpha),
                              rtol=1e-6, atol=0)


def test_the_same_sequence_under_grad_keeps_the_legacy_estimator():
    """A/B compatibility: retention_space=grad classifies nothing as AdamW, so the
    actor captures raw g1 and forms r = g1/g0 exactly as before."""
    params = [torch.nn.Parameter(torch.randn(16))]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    actor = _BufferActor(optimizer, params, space='grad')
    assert actor._gxpo_retention_kinds() is None
    assert actor._gxpo_u0_slot_mask(None) is None

    g0 = torch.randn(16)
    g1 = torch.randn(16)
    legacy = compute_gxpo_retention_scale(g0, g1, K, DELTA, clip_scale_g0=0.5,
                                          clip_scale_g1=0.25)
    reference = ((g1 * 0.25) / (g0 * 0.5)).clamp(-2.0, 3.0)
    assert torch.allclose(legacy[0][legacy[2]], reference[legacy[2]], rtol=1e-6)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
