"""GXPO correctness tests for the production helper and actor wiring.

Run directly with:
    PYTHONPATH=Code/SFPO python tests/gxpo/test_gxpo_parity.py

The scale tests import the exact helper used by ``dp_actor.py``. They do not
maintain a second implementation of GXPO arithmetic.
"""

import ast
import importlib.util
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
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
    assert step_source.index('self._gxpo_capture_grads(g1_bufs)') < step_source.index(
        'gn1 = probe_clip_grads()')
    # Current semantics intentionally gate after the corrective optimizer step:
    # the trigger disables subsequent GXPO steps, not the update just computed.
    assert step_source.rfind('self.actor_optimizer.step()') < step_source.index('state.update_trigger_state')
    assert 'snapshot_optimizer_state(self.actor_optimizer)' in step_source
    assert 'gxpo_optimizer_state_mode' in source
    assert 'optimizer_transaction.restore()' in step_source


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'PASS {name}')
    print('ALL GXPO PRODUCTION CHECKS PASSED')
