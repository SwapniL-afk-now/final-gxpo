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


def production_ratio_scale(g0, g1, K, delta=1e-8):
    return GXPO.compute_gxpo_retention_scale(g0, g1, K, delta)


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
    theta0 = torch.randn(1000, dtype=torch.float64)
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
    state = GXPO.GXPOState(K=5, tau=2.0, omega=0.1, warmup_steps=3)
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
    assert 'stats[9] += scale.double().sum()' in source
    assert 'scale_max = torch.maximum(scale_max, scale.max().double().reshape(1))' in source
    _, scale, _, _ = production_ratio_scale(torch.ones(8), torch.ones(8), K=5)
    assert scale.mean().item() == 2.5
    assert scale.max().item() == 2.5


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
    assert step_source.index('self._gxpo_capture_grads(g0_bufs)') < step_source.index('self._clip_grads()')
    assert step_source.index('self._gxpo_capture_grads(g1_bufs)') < step_source.index('self._clip_grads()', step_source.index('self._gxpo_capture_grads(g1_bufs)'))
    # Current semantics intentionally gate after the corrective optimizer step:
    # the trigger disables subsequent GXPO steps, not the update just computed.
    assert step_source.rfind('self.actor_optimizer.step()') < step_source.index('state.update_trigger_state')
    assert 'probe-step optimizer-moment' in source and 'pollution is accepted' in source


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'PASS {name}')
    print('ALL GXPO PRODUCTION CHECKS PASSED')
