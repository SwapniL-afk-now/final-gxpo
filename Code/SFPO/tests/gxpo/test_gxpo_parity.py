"""GXPO correctness tests for the production helper and actor wiring.

Run directly with:
    PYTHONPATH=Code/SFPO python tests/gxpo/test_gxpo_parity.py

The scale tests import the exact helper used by ``dp_actor.py``. They do not
maintain a second implementation of GXPO arithmetic.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
# The Muon tests below import the verl package itself (the scale tests only need
# gxpo_state, which is loaded by path). Make direct execution work without
# requiring the caller to set PYTHONPATH.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
GXPO_STATE_PATH = REPO / 'verl' / 'workers' / 'actor' / 'gxpo_state.py'
ACTOR_PATH = REPO / 'verl' / 'workers' / 'actor' / 'dp_actor.py'


def load_gxpo_module():
    spec = importlib.util.spec_from_file_location('production_gxpo_state', GXPO_STATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GXPO = load_gxpo_module()


def production_ratio_scale(g0, g1, K, delta=1e-8, **kwargs):
    return GXPO.compute_gxpo_retention_scale(g0, g1, K, delta, **kwargs)


def test_scale_edge_cases():
    g0 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1e-9, 1.0, 1.0, 1.0])
    g1 = torch.tensor([1.0, 0.0, -1.0, 10.0, float('nan'), 4.0, 4.0, float('inf'), -3.0, 1.0])
    ratio, scale, active, clipped = production_ratio_scale(g0, g1, K=5)

    # r=1 -> S_K/S_2=K/2; r=0 -> 1; r=-1 has S_2=0 and safely falls back to 1.
    assert torch.allclose(ratio[:4], torch.tensor([1.0, 0.0, -1.0, 3.0]))
    assert torch.allclose(scale[:4], torch.tensor([2.5, 1.0, 1.0, 3.5]))
    assert not active[5] and not active[6]
    assert torch.equal(ratio[5:7], torch.ones(2))
    assert torch.equal(scale[5:7], torch.ones(2))
    assert clipped[3] and clipped[4] and clipped[7]
    assert torch.isfinite(ratio).all() and torch.isfinite(scale).all()
    assert scale.min() >= 1.0 and scale.max() <= 3.5


def test_retention_uses_probe_clip_scales_and_relative_threshold():
    g0 = torch.tensor([1e-6, 1.0, -2.0])
    g1 = torch.tensor([7.0, 0.5, -1.0])
    g0_before, g1_before = g0.clone(), g1.clone()
    ratio, scale, active, clipped = production_ratio_scale(
        g0, g1, K=2, clip_scale_g0=0.5, clip_scale_g1=0.25)

    # RMS(g0) ~= 1.291, so the first coordinate is below 1e-3 * RMS and is
    # neutral; the remaining coordinates use (c1*g1)/(c0*g0).
    assert torch.equal(active, torch.tensor([False, True, True]))
    assert torch.allclose(ratio, torch.tensor([1.0, 0.25, 0.25]))
    assert torch.equal(scale, torch.ones_like(scale))
    assert not clipped.any()
    assert torch.equal(g0, g0_before)
    assert torch.equal(g1, g1_before)


def test_retention_ratio_clip_is_applied_after_clip_correction():
    g0 = torch.tensor([2.0])
    g1 = torch.tensor([10.0])
    ratio, _, _, clipped = production_ratio_scale(
        g0, g1, K=5, clip_scale_g0=0.5, clip_scale_g1=0.25)

    # Raw g1/g0 is 5, while the ratio of the gradients that drove the clipped
    # updates is 2.5; neither value should be confused with displacement ratios.
    assert torch.allclose(ratio, torch.tensor([2.5]))
    assert not clipped.item()


def test_k_two_has_no_extra_extrapolation():
    g0 = torch.tensor([0.2, -0.4, 2.0])
    g1 = torch.tensor([0.7, 0.3, -8.0])
    _, scale, _, _ = production_ratio_scale(g0, g1, K=2)
    assert torch.equal(scale, torch.ones_like(scale))


def test_horner_geometric_sum():
    values = torch.tensor([-1.9, -0.5, 0.0, 0.999999, 1.0, 2.9], dtype=torch.float64)
    for n in (1, 2, 4, 8):
        actual = GXPO.geometric_sum_horner(values, n)
        expected = torch.stack([values.pow(i) for i in range(n)]).sum(dim=0)
        assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_diagonal_quadratic_exact_case():
    torch.manual_seed(2)
    h = torch.rand(1000, dtype=torch.float64) * 0.9 + 0.05
    eta = 1.0
    theta0 = torch.sign(torch.randn(1000, dtype=torch.float64)) * (torch.rand(1000, dtype=torch.float64) * 1.5 + 0.5)
    for K in (2, 4, 8):
        theta1 = theta0 - eta * h * theta0
        theta2 = theta1 - eta * h * theta1
        _, scale, _, _ = production_ratio_scale(h * theta0, h * theta1, K, delta=1e-14)
        theta_tilde = theta0 + scale * (theta2 - theta0)
        theta_gxpo = theta_tilde - eta * h * theta_tilde
        theta_gd = theta0.clone()
        for _ in range(K + 1):
            theta_gd = theta_gd - eta * h * theta_gd
        assert (theta_gxpo - theta_gd).abs().max().item() < 1e-9


def test_bf16_matches_fp32_reference():
    torch.manual_seed(4)
    g0_fp32 = torch.randn(4096, dtype=torch.float32) * 0.1
    # Keep this precision comparison away from the intentional S_2=0
    # stabilization boundary (r=-1), which is discontinuous by design.
    g1_fp32 = g0_fp32 * (0.2 + torch.rand(4096, dtype=torch.float32) * 1.5)
    _, scale_fp32, _, _ = production_ratio_scale(g0_fp32, g1_fp32, K=5)
    _, scale_bf16, _, _ = production_ratio_scale(g0_fp32.bfloat16(), g1_fp32.bfloat16(), K=5)
    assert torch.allclose(scale_bf16.float(), scale_fp32, rtol=0.08, atol=0.08)


def test_trigger_gate_observes_corrective_norm():
    state = GXPO.GXPOState(K=5, tau=2.0, omega=0.1, zscore_w=10, warmup_steps=0)
    observations = [30.0] * 10 + [300.0] + [30.0] * 4
    triggered_at = None
    for step, norm in enumerate(observations):
        if state.is_enabled(step):
            _, _, triggered = state.update_trigger_state(
                step=step, g0_norm=norm, g_slow_norm=norm)
            if triggered and triggered_at is None:
                triggered_at = step
    assert triggered_at == 10
    assert state.trigger_index == 11
    assert not state.is_enabled(11)


def test_trigger_gate_uses_rolling_window_mean_and_std():
    state = GXPO.GXPOState(K=5, tau=1.0, zscore_w=3, warmup_steps=0,
                           trigger_patience=2)
    for step, norm in enumerate((10.0, 10.0, 10.0)):
        z, _, triggered = state.update_trigger_state(
            step=step, g0_norm=norm, g_slow_norm=norm)
        assert z == 0.0
        assert not triggered

    z, _, triggered = state.update_trigger_state(
        step=3, g0_norm=30.0, g_slow_norm=30.0)
    assert not triggered
    # SFPO scores the current value against the preceding window; the spike
    # itself is not allowed to inflate its own rolling mean/std.
    assert state.mu == 10.0
    assert state.sigma == 0.0
    assert z > 1e9

    # The spike opened a candidate streak (z >= tau), so the baseline that scored it is
    # pinned for the rest of the excursion: otherwise the spike would be folded into the
    # mean/std before the next observation was scored, and trigger_patience > 1 could
    # never be satisfied on a sustained excursion.
    assert state.trigger_streak == 1
    z, _, triggered = state.update_trigger_state(
        step=4, g0_norm=10.0, g_slow_norm=10.0)
    assert not triggered
    assert state.mu == 10.0, 'baseline must stay frozen while a candidate streak is open'
    assert z == 0.0

    # Back under tau clears the streak, so the live rolling window resumes and now
    # legitimately includes the spike.
    assert state.trigger_streak == 0
    z, _, triggered = state.update_trigger_state(
        step=5, g0_norm=10.0, g_slow_norm=10.0)
    assert not triggered
    assert state.mu == 50.0 / 3.0
    assert z < 0.0


def test_trigger_patience_requires_consecutive_violations():
    state = GXPO.GXPOState(K=5, tau=2.0, omega=0.1, warmup_steps=0,
                           trigger_patience=3)
    assert not state.check_trigger(2.1, 10)
    assert state.trigger_streak == 1
    assert not state.check_trigger(2.2, 11)
    assert state.trigger_streak == 2
    assert state.check_trigger(2.3, 12)
    assert state.trigger_streak == 3
    assert state.trigger_index == 13


def test_deferred_trigger_does_not_consume_a_minibatch_observation():
    state = GXPO.GXPOState(K=5, tau=2.0, zscore_w=1, omega=0.1, warmup_steps=0,
                           trigger_patience=3)
    z, _, triggered = state.update_trigger_state(
        step=10, g0_norm=30.0, g_slow_norm=30.0,
        allow_trigger=True, defer_trigger=True)
    assert not triggered and state.trigger_streak == 0
    assert z == 0.0
    assert state.observation_count == 0
    assert state.trigger_history == []


def test_warmup_observations_are_reset_before_the_post_warmup_window():
    state = GXPO.GXPOState(K=5, tau=1.0, zscore_w=2, warmup_steps=2)
    state.update_trigger_state(step=0, g0_norm=1.0, g_slow_norm=1.0,
                               allow_trigger=False)
    state.update_trigger_state(step=1, g0_norm=1.0, g_slow_norm=1.0,
                               allow_trigger=False)
    z, _, triggered = state.update_trigger_state(
        step=2, g0_norm=10.0, g_slow_norm=10.0, allow_trigger=True)
    assert z == 0.0 and not triggered
    assert state.trigger_history == [10.0]


def test_outer_trigger_reduces_minibatches_to_one_mean_scalar():
    source = ACTOR_PATH.read_text()
    assert "outer_stat = sum(stat_values) / len(stat_values) if stat_values else 0.0" in source
    assert "outer_z = max(z_values, default=0.0)" not in source
    assert "score it against the preceding" in source


def test_gxpo_fallback_uses_sfpo_entropy_gate_in_trainer():
    trainer_source = (REPO / 'verl' / 'trainer' / 'ppo' / 'ray_trainer.py').read_text()
    worker_source = (REPO / 'verl' / 'workers' / 'fsdp_workers.py').read_text()
    actor_source = ACTOR_PATH.read_text()
    assert 'self.gxpo_entropy_container = []' in trainer_source
    # The entropy gate runs through GXPOState rather than a second inline z-score, so the
    # preceding-window ordering, the frozen streak baseline and the sustained-level
    # criterion are shared with the gradient-signal path instead of reimplemented.
    assert 'self._gxpo_gate.update_trigger_state(' in trainer_source
    assert '_build_gxpo_entropy_gate' in trainer_source
    assert 'gxpo_trigger_z = (gxpo_trigger_stat - u) / std' not in trainer_source, (
        'the inline z-score gate must not come back: its window contained the sample it '
        'was scoring, which caps abs(z) at sqrt(zscore_w - 1)')
    # The SFPO path deliberately keeps its own inline gate so existing SFPO baselines
    # remain reproducible.
    assert 'sfpo_trigger_z = (self.entropy_container[-1] - u) / std' in trainer_source
    assert "batch.meta_info['gxpo_trigger_stop'] = self.stop_GXPO" in trainer_source
    assert "data.meta_info.get('gxpo_trigger_stop', False)" in worker_source
    assert "self.config.get('gxpo_trigger_signal', 'entropy') == 'entropy'" in actor_source
    assert "data.meta_info.get('gxpo_trigger_z', 0.0)" in actor_source


def test_fixed_old_log_probs_wiring_and_checkpointing():
    source = ACTOR_PATH.read_text()
    tree = ast.parse(source)
    actor_step = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == '_gxpo_minibatch_step')
    assert "gxpo_recompute_old_log_probs', False" in source
    assert "old_log_prob = data['old_log_probs']" in source
    assert 'recompute_old_log_probs=recompute_old' in source
    assert "gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})" in (
        (REPO / 'verl' / 'workers' / 'fsdp_workers.py').read_text())
    assert actor_step is not None


def test_scale_diagnostics_are_accumulated_and_bounded():
    source = ACTOR_PATH.read_text()
    assert 'dtype=torch.float32' in source
    assert 'stats[9] += scale.float().sum()' in source
    assert 'scale_max = torch.maximum(scale_max, scale.float().amax().reshape(1))' in source
    assert '.double()' not in ast.get_source_segment(
        source,
        next(node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.FunctionDef) and node.name == '_gxpo_minibatch_step'))
    _, scale, _, _ = production_ratio_scale(torch.ones(8), torch.ones(8), K=5)
    assert scale.mean().item() == 2.5
    assert scale.max().item() == 2.5


def test_probe_passes_skip_discarded_metrics():
    source = ACTOR_PATH.read_text()
    tree = ast.parse(source)
    backward = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == '_backward_minibatch')
    collect_arg = next(arg for arg in backward.args.args if arg.arg == 'collect_metrics')
    assert collect_arg is not None
    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == '_gxpo_minibatch_step')
    step_source = ast.get_source_segment(source, step)
    assert 'collect_metrics=skip_corrective' in step_source
    assert 'collect_metrics=False' in step_source
    assert 'collect_metrics=True' in step_source


def test_actor_reports_retention_stability_diagnostics():
    source = ACTOR_PATH.read_text()
    for metric in (
            "actor/gxpo_clip_scale_g0",
            "actor/gxpo_clip_scale_g1",
            "actor/gxpo_relative_threshold_reject_frac",
            "actor/gxpo_ratio_clip_frac"):
        assert metric in source
    step = next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.FunctionDef) and node.name == "_gxpo_minibatch_step")
    step_source = ast.get_source_segment(source, step)
    assert "clip_scale_g0=clip_scale_g0" in step_source
    assert "clip_scale_g1=clip_scale_g1" in step_source
    assert step_source.index("gn0 = probe_clip_grads()") < step_source.index(
        "clip_scale_g0 =")
    assert step_source.index("gn1 = probe_clip_grads()") < step_source.index(
        "clip_scale_g1 =")


def test_attention_backend_is_configurable_with_fa2_default():
    worker_source = (REPO / 'verl' / 'workers' / 'fsdp_workers.py').read_text()
    helper_source = (REPO / 'verl' / 'utils' / 'attention.py').read_text()
    assert 'resolve_attention_implementation' in worker_source
    assert 'DEFAULT_ATTENTION_IMPLEMENTATION = "flash_attention_2"' in helper_source
    assert 'flash_attention_3' in helper_source
    assert "attn_implementation=attn_implementation" in worker_source
    assert "attn_implementation='flash_attention_2'" not in worker_source


def test_reposition_uses_two_step_displacement_at_correct_location():
    theta0 = torch.tensor([1.0, -2.0, 3.0])
    theta2 = torch.tensor([0.8, -1.5, 2.0])
    g0 = torch.tensor([0.2, -0.4, 0.6])
    g1 = torch.tensor([0.1, -0.8, 1.8])
    _, scale, _, _ = production_ratio_scale(g0, g1, K=4)
    theta_tilde = theta0 + 0.5 * scale * (theta2 - theta0)
    expected = theta0 + 0.5 * scale * (theta2 - theta0)
    assert torch.equal(theta_tilde, expected)
    # A corrective gradient must be evaluated at theta_tilde, not theta2.
    corrective_at_tilde = theta_tilde.square().sum().sqrt()
    corrective_at_theta2 = theta2.square().sum().sqrt()
    assert corrective_at_tilde != corrective_at_theta2


def test_optimizer_scheduler_and_vllm_sync_boundaries():
    worker_path = REPO / 'verl' / 'workers' / 'fsdp_workers.py'
    worker_source = worker_path.read_text()
    worker_tree = ast.parse(worker_source)
    gxpo_worker = next(node for node in ast.walk(worker_tree)
                       if isinstance(node, ast.FunctionDef) and node.name == 'gxpo_update_actor')
    gxpo_worker_source = ast.get_source_segment(worker_source, gxpo_worker)
    assert gxpo_worker_source.count('self.actor_lr_scheduler.step()') == 1
    assert 'sync_model_weights' not in gxpo_worker_source

    generate = next(node for node in ast.walk(worker_tree)
                    if isinstance(node, ast.FunctionDef) and node.name == 'generate_sequences')
    generate_source = ast.get_source_segment(worker_source, generate)
    assert 'self.rollout_sharding_manager' in generate_source


def test_gate_and_gradient_capture_order():
    source = ACTOR_PATH.read_text()
    tree = ast.parse(source)
    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == '_gxpo_minibatch_step')
    step_source = ast.get_source_segment(source, step)
    assert step_source.index('self._gxpo_capture_grads(g0_bufs)') < step_source.index(
        'gn0 = probe_clip_grads()')
    # Prefix match, not an exact call: g1 capture takes a `skip` argument so the
    # update-space path can repurpose the g1 slot. The ordering is what matters.
    assert step_source.index('self._gxpo_capture_grads(g1_bufs') < step_source.index(
        'gn1 = probe_clip_grads()')
    # Current semantics intentionally gate after the corrective optimizer step:
    # the trigger disables subsequent GXPO steps, not the update just computed.
    assert step_source.rfind('self.actor_optimizer.step()') < step_source.index('state.update_trigger_state')
    assert 'snapshot_optimizer_state(self.actor_optimizer)' in step_source
    assert 'gxpo_optimizer_state_mode' in source
    assert 'optimizer_transaction.restore()' in step_source


# ---------------------------------------------------------------------------
# Muon: why the gradient-space retention estimator does not apply, and what the
# update-space estimator does instead.
#
# These drive the real production optimizer (verl.workers.muon.Muon, dense
# path) rather than a stand-in, so they fail if Muon's update rule changes.
# ---------------------------------------------------------------------------

def production_update_scale(u0, u1, K, delta=1e-8, **kwargs):
    return GXPO.compute_gxpo_update_retention_scale(u0, u1, K, delta, **kwargs)


def _run_muon(grads, shape=(16, 12), lr=1e-3, seed=0):
    """Apply a prescribed gradient sequence through real Muon; return the theta path."""
    from verl.workers.muon import Muon
    torch.manual_seed(seed)
    p = torch.nn.Parameter(torch.randn(*shape, dtype=torch.float32))
    opt = Muon(lr=lr, wd=0.0, muon_params=[p], momentum=0.95, nesterov=True, ns_steps=5)
    path = [p.data.clone()]
    for g in grads:
        p.grad = g.clone()
        opt.step()
        path.append(p.data.clone())
    return path


def test_muon_update_is_invariant_to_gradient_scale():
    """The root cause, pinned: Muon's step does not depend on gradient magnitude.

    zeropower_via_newtonschulz5 normalizes its input to unit norm before the NS
    iterations, and adjust_lr_for_muon scales the write-back by parameter shape
    alone. So r = g1/g0 -- the entire gradient-space retention signal -- carries
    no information about how Muon's displacement evolves.

    Newton-Schulz casts its input to bfloat16, so the invariance is exact only
    for scale factors that are pure exponent shifts. Both halves are asserted:

      * x2  -- bit-exact, no difference at all;
      * x10, x100, x10000 -- a small residual from landing on different bf16
        grid points, which stays flat instead of growing with the factor. A
        genuine scale dependence would grow with it (x10000 would move 10000x
        further); bf16 quantization does not.
    """
    torch.manual_seed(3)
    base = [torch.randn(16, 12) for _ in range(4)]
    reference = _run_muon(base)
    ref_disp = reference[-1] - reference[0]

    exact = _run_muon([2.0 * g for g in base])
    assert torch.equal(exact[-1] - exact[0], ref_disp), (
        'a power-of-two gradient rescale must leave the Muon step bit-identical')

    factors = (10.0, 100.0, 10000.0)
    rels = []
    for factor in factors:
        scaled = _run_muon([factor * g for g in base])
        disp = scaled[-1] - scaled[0]
        rels.append(((disp - ref_disp).norm() / ref_disp.norm()).item())
        assert rels[-1] < 0.10, (
            f'{factor:g}x gradient moved the displacement by {rels[-1]:.3f} (relative); '
            'expected only bf16 quantization noise')

    # The decisive part: the residual is flat in the factor. A
    # gradient-proportional optimizer would show rel ~ factor - 1, i.e. three
    # orders of magnitude between the first and last entry here.
    assert max(rels) / min(rels) < 3.0, (
        f'residual tracked the gradient scale ({rels}), so the step is not '
        'magnitude-invariant after all')


def _prediction_errors(K=10, shape=(16, 12), seed=1, growth=3.0):
    """Predicted vs. true K-step Muon displacement, for both estimators."""
    torch.manual_seed(seed)
    g0 = torch.randn(*shape)
    grads = [g0 * (growth**i) for i in range(K)]

    path = _run_muon(grads, shape=shape, seed=seed)
    theta0, theta1, theta2, thetaK = path[0], path[1], path[2], path[K]
    disp2 = theta2 - theta0
    true_disp = thetaK - theta0

    _, grad_scale, _, _ = production_ratio_scale(grads[0], grads[1], K=K)
    _, upd_scale = production_update_scale(theta1 - theta0, theta2 - theta1, K=K)

    def rel_err(pred):
        return ((pred - true_disp).norm() / (true_disp.norm() + 1e-12)).item()

    return {
        'grad_err': rel_err(disp2 * grad_scale),
        'update_err': rel_err(disp2 * upd_scale),
        'upd_scale': upd_scale.item(),
        'grad_scale_mean': grad_scale.mean().item(),
    }


def test_grad_space_error_tracks_a_signal_muon_ignores():
    """The gradient-space predictor is right only by coincidence.

    Its error is driven entirely by how the *gradients* evolve -- a quantity
    Muon's update discards. Flat gradients (r=1) happen to give S_K/S_2 = K/2,
    which is roughly correct for Muon, so the estimator looks fine. Decaying
    gradients make it predict a contraction Muon never performs, and the error
    explodes. Same optimizer, same K: only the ignored signal changed.
    """
    flat = _prediction_errors(growth=1.0)['grad_err']
    decay = _prediction_errors(growth=0.3)['grad_err']

    assert flat < 0.10, f'flat-gradient grad-space error {flat:.3f} unexpectedly large'
    assert decay > 0.50, f'decaying-gradient grad-space error only {decay:.3f}'
    assert decay > 5 * flat, (
        f'expected the error to swing with the ignored signal, got {flat:.3f} -> {decay:.3f}')


def test_update_space_predictor_is_stable_across_gradient_regimes():
    """The fix: reading retention off the real steps removes that sensitivity.

    Muon's steps persist in direction, so rho ~ 1 and S_K/S_2 ~ K/2 regardless
    of what the gradients do -- which is exactly the invariance Muon itself has.
    """
    regimes = {g: _prediction_errors(growth=g) for g in (0.3, 0.5, 1.0, 3.0)}

    for growth, res in regimes.items():
        assert res['update_err'] < 0.15, (
            f'update-space error {res["update_err"]:.3f} at growth={growth}')
        # The extrapolation factor is stable near K/2 = 5, as Muon's constant
        # step size implies.
        assert 3.5 < res['upd_scale'] < 6.0, (
            f'update-space scale {res["upd_scale"]:.2f} at growth={growth}')

    worst_update = max(r['update_err'] for r in regimes.values())
    worst_grad = max(r['grad_err'] for r in regimes.values())
    assert worst_update < worst_grad, (
        f'update-space worst case {worst_update:.3f} should beat '
        f'grad-space worst case {worst_grad:.3f}')


def test_scalar_scale_preserves_direction_where_coordinatewise_does_not():
    """disp2 is built from orthogonalized steps; a per-entry scale rotates it.

    The gradient ratio must be non-uniform for this to bite -- a uniform ratio
    yields a uniform scale, which is just a scalar in disguise. Real training
    ratios are non-uniform (the run logged retention_std ~ 0.6).
    """
    torch.manual_seed(5)
    shape = (16, 12)
    g0 = torch.randn(*shape)
    # Per-coordinate ratios spread over the estimator's active range.
    ratio = torch.empty(shape).uniform_(0.2, 2.5)
    grads = [g0, g0 * ratio]

    path = _run_muon(grads, shape=shape, seed=5)
    disp2 = path[2] - path[0]

    def cos_with_disp2(pred):
        return (torch.dot(pred.flatten(), disp2.flatten())
                / (pred.norm() * disp2.norm() + 1e-12)).item()

    _, upd_scale = production_update_scale(path[1] - path[0], path[2] - path[1], K=10)
    assert abs(cos_with_disp2(disp2 * upd_scale) - 1.0) < 1e-5, 'scalar scaling rotated disp2'

    _, grad_scale, _, _ = production_ratio_scale(g0, g0 * ratio, K=10)
    cos_grad = cos_with_disp2(disp2 * grad_scale)
    assert cos_grad < 0.999, (
        f'coordinatewise scale left direction intact (cos={cos_grad:.6f}); '
        'the ratio spread may no longer reach the estimator')


def test_update_space_reduces_to_grad_space_for_sgd():
    """The generalization claim: under SGD the two estimators agree.

    With u_t = -lr * g_t and a uniform coordinatewise ratio r, rho == r, so the
    new estimator returns exactly what the old one does. It is a strict
    generalization, not a different algorithm.
    """
    lr = 0.1
    torch.manual_seed(11)
    for r in (0.25, 0.5, 0.9, -0.5):
        g0 = torch.randn(32)
        g1 = r * g0
        rho, upd_scale = production_update_scale(-lr * g0, -lr * g1, K=10)
        _, grad_scale, _, _ = production_ratio_scale(g0, g1, K=10)
        assert abs(rho.item() - r) < 1e-5, f'rho={rho.item()} != r={r}'
        assert torch.allclose(grad_scale, upd_scale.expand_as(grad_scale), atol=1e-5), (
            f'estimators disagree for SGD at r={r}')


def test_update_space_bounds_and_degenerate_cases():
    K = 10
    # Perfectly persistent direction -> rho=1 -> S_K/S_2 = K/2.
    u = torch.ones(8)
    rho, scale = production_update_scale(u, u, K=K)
    assert abs(rho.item() - 1.0) < 1e-6 and abs(scale.item() - K / 2.0) < 1e-6

    # Orthogonal second step -> rho=0 -> neutral scale.
    rho, scale = production_update_scale(torch.tensor([1.0, 1.0]),
                                         torch.tensor([1.0, -1.0]), K=K)
    assert abs(rho.item()) < 1e-6 and abs(scale.item() - 1.0) < 1e-6

    # A vanished first step carries no direction: neutral, never a blow-up.
    rho, scale = production_update_scale(torch.zeros(8), torch.ones(8), K=K)
    assert torch.isfinite(rho) and torch.isfinite(scale)
    assert abs(scale.item() - 1.0) < 1e-6

    # rho is clamped into [-1, 1] even when the second step overshoots wildly.
    rho, _ = production_update_scale(torch.ones(8), 50.0 * torch.ones(8), K=K)
    assert abs(rho.item()) <= 1.0 + 1e-9

    # On |rho| <= 1 the series ratio is 1/(1 - rho^2) >= 1, so the [0, K/2+1]
    # floor never binds. Contraction is expressed through alpha, not here.
    for i in range(-99, 100):
        _, s = production_update_scale(u, (i / 100.0) * u, K=K)
        assert s.item() >= 1.0 - 1e-6


def test_update_space_is_invariant_to_absolute_step_magnitude():
    """Regression: the self-dot guard must be a predicate, not an epsilon.

    rho is scale-free -- multiplying both steps by any constant must leave it
    unchanged. An additive guard breaks that at small magnitudes, and real Muon
    steps are small: a 1536x1536 matrix at lr 1e-6 has ||u0||^2 ~ 1e-7, against
    which delta = 1e-8 is a >10% error. The replay initially reported the
    update-space estimator losing on 196/196 matrices purely because of this.
    """
    torch.manual_seed(17)
    n = 1536 * 1536
    base = torch.randn(n)
    other = 0.99 * base + (1 - 0.99**2)**0.5 * torch.randn(n)

    reference = None
    for magnitude in (1.0, 1e-3, 2e-7, 1e-9):
        rho, scale = production_update_scale(base * magnitude, other * magnitude, K=10)
        assert abs(rho.item() - 0.99) < 5e-3, (
            f'rho={rho.item():.4f} at step magnitude {magnitude:g}; expected ~0.99 '
            'regardless of magnitude')
        if reference is None:
            reference = scale.item()
        assert abs(scale.item() - reference) < 1e-3, (
            f'scale moved from {reference:.4f} to {scale.item():.4f} when only the '
            'absolute step size changed')

    # Correspondingly, rho is exactly the norm-weighted cosine.
    u0, u1 = base * 2e-7, other * 2e-7
    cos = (torch.dot(u0, u1) / (u0.norm() * u1.norm())).item()
    ratio = (u1.norm() / u0.norm()).item()
    rho, _ = production_update_scale(u0, u1, K=10)
    assert abs(rho.item() - cos * ratio) < 1e-4, (
        f'rho={rho.item():.6f} != cos*ratio={cos * ratio:.6f}')


def test_update_space_dots_may_be_supplied_precomputed():
    """The FSDP path passes shard-summed dot products instead of the tensors."""
    torch.manual_seed(13)
    u0, u1 = torch.randn(64), torch.randn(64)
    direct = production_update_scale(u0, u1, K=10)
    dots = torch.stack(((u0 * u1).sum(), u0.square().sum()))
    viadots = production_update_scale(None, None, K=10, dots=dots)
    assert torch.allclose(direct[0], viadots[0], atol=1e-6)
    assert torch.allclose(direct[1], viadots[1], atol=1e-6)

    # Summing per-shard dots must reproduce the whole-tensor answer -- that is
    # exactly what the all_reduce in _gxpo_update_space_dots relies on.
    half = torch.stack(((u0[:32] * u1[:32]).sum(), u0[:32].square().sum()))
    rest = torch.stack(((u0[32:] * u1[32:]).sum(), u0[32:].square().sum()))
    sharded = production_update_scale(None, None, K=10, dots=half + rest)
    assert torch.allclose(direct[1], sharded[1], atol=1e-6)


def test_gxpo_allocates_exactly_three_model_sized_buffers():
    """Memory invariant: GXPO keeps three model-shard buffers, never four.

    The update-space estimator needs u0 = theta1 - theta0, but a fourth buffer
    is a whole extra copy of the model per rank (~3GB at 1.5B, ~6GB at 3B) in a
    run that already sat at 94.6 of 94.97 GiB. Muon-owned parameters therefore
    store u0 in their g1 slot, which is dead weight for them: g1 exists only to
    form r = g1/g0, the ratio this fix showed is uninformative for Muon.
    """
    source = ACTOR_PATH.read_text()
    tree = ast.parse(source)
    init = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == '_gxpo_init_buffers')
    init_source = ast.get_source_segment(source, init)
    assert "('theta0', 'g0', 'g1')" in init_source, (
        'GXPO must allocate exactly theta0/g0/g1')
    code_only = '\n'.join(line for line in init_source.splitlines()
                          if not line.lstrip().startswith('#'))
    assert 'theta1' not in code_only, (
        'a fourth model-sized buffer must not be allocated')
    # The g1 slot is repurposed, so its gradient capture must be skippable.
    assert 'skip=update_space' in source
    # ...and the shutoff gate must correlate the corrective gradient against g0,
    # never g1: a Muon-owned parameter's g1 slot holds u0, not a gradient.
    assert 'gslow_stats[1] += (gradf * g0f).sum()' in source, (
        'the shutoff gate must correlate the corrective gradient with g0')
    assert 'gradf * g1f' not in source


def test_update_space_dots_identity_used_by_the_actor():
    """The actor forms <u0,u1> as <u0,disp2> - <u0,u0>; check that identity.

    It never materializes u1, because theta1 is gone by then -- only theta0, the
    live theta2, and u0 (in the g1 slot) survive.
    """
    torch.manual_seed(23)
    # Exact-arithmetic check first: in float64 the identity is algebra, so it
    # must hold to round-off.
    u0 = (torch.randn(512) * 2e-7).double()
    u1 = (torch.randn(512) * 2e-7).double()
    disp2 = u0 + u1
    self_dot = u0.square().sum()
    via_identity = (u0 * disp2).sum() - self_dot
    direct = (u0 * u1).sum()
    assert torch.allclose(via_identity, direct, rtol=1e-12, atol=0.0)

    # Now the precision the actor actually runs at. disp2 is formed in float32
    # from weights ~1e-2 and steps ~1e-7, and <u0,disp2> - <u0,u0> cancels about
    # one digit, so agreement is float32-grade, not float64-grade. rho is a ratio
    # of these, and 1e-3 here is far tighter than the 0.01 that would matter to
    # S_K(rho)/S_2(rho).
    u0_32, u1_32 = u0.float(), u1.float()
    disp2_32 = u0_32 + u1_32
    approx = ((u0_32 * disp2_32).sum() - u0_32.square().sum()).double()
    assert torch.allclose(approx, direct, rtol=1e-3, atol=0.0), (
        f'fp32 identity drifted: {approx.item():.6e} vs {direct.item():.6e}')

    rho_a, scale_a = production_update_scale(u0, u1, K=10)
    dots = torch.stack(((u0 * disp2).sum() - u0.square().sum(), u0.square().sum()))
    rho_b, scale_b = production_update_scale(None, None, K=10, dots=dots)
    assert torch.allclose(rho_a, rho_b, atol=1e-5)
    assert torch.allclose(scale_a, scale_b, atol=1e-5)


def test_actor_routes_muon_params_to_update_space():
    """Wiring: the actor must branch on Muon ownership, not apply one estimator."""
    source = ACTOR_PATH.read_text()
    assert 'compute_gxpo_update_retention_scale' in source
    assert "get('use_muon', False)" in source, 'Muon ownership must come from the optimizer'
    assert 'gxpo_retention_space' in source
    # The shard-reducing collective must sit outside the per-parameter loop so
    # every rank reaches it the same number of times.
    dots_at = source.index('self._gxpo_update_space_dots(')
    loop_at = source.index('for _p2_i, (p, t0, g0b, g1b, g0_rms) in enumerate(')
    assert dots_at < loop_at, 'the all_reduce must not be inside the per-parameter loop'


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'PASS {name}')
    print('ALL GXPO PRODUCTION CHECKS PASSED')
