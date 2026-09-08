"""CPU-only tests for GXPO's local optimizer-state transaction."""

import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[2] / 'verl' / 'workers' / 'actor' / 'optimizer_transaction.py'
spec = importlib.util.spec_from_file_location('optimizer_transaction', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
snapshot_optimizer_state = module.snapshot_optimizer_state


def _state_clone(optimizer, parameter):
    state = optimizer.state.get(parameter, {})
    return {key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()}


@pytest.mark.parametrize('optimizer_factory', [
    lambda parameter: torch.optim.Adam([parameter], lr=0.1),
    lambda parameter: torch.optim.AdamW([parameter], lr=0.1),
    lambda parameter: torch.optim.SGD([parameter], lr=0.1, momentum=0.9),
])
def test_probe_state_is_restored_for_common_optimizers(optimizer_factory):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = optimizer_factory(parameter)

    parameter.grad = torch.tensor([0.25, -0.5])
    optimizer.step()
    baseline = _state_clone(optimizer, parameter)
    baseline_step = baseline.get('step')
    exp_avg_ref = optimizer.state[parameter].get('exp_avg')

    transaction = snapshot_optimizer_state(optimizer)
    parameter.grad = torch.tensor([-3.0, 4.0])
    optimizer.step()
    parameter.grad = torch.tensor([5.0, -6.0])
    optimizer.step()
    transaction.restore()

    restored = optimizer.state[parameter]
    for key, value in baseline.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(restored[key], value)
        else:
            assert restored[key] == value
    assert restored.get('step') == baseline_step
    if exp_avg_ref is not None:
        assert restored['exp_avg'] is exp_avg_ref


def test_probe_updates_are_discarded_but_final_step_commits_once():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    baseline_step = optimizer.state[parameter]['step'].clone()

    transaction = snapshot_optimizer_state(optimizer)
    parameter.grad = torch.full_like(parameter, 2.0)
    optimizer.step()
    parameter.grad = torch.full_like(parameter, 3.0)
    optimizer.step()
    transaction.restore()

    parameter.grad = torch.full_like(parameter, 4.0)
    optimizer.step()
    assert optimizer.state[parameter]['step'] == baseline_step + 1


def test_new_probe_state_entries_are_removed():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    transaction = snapshot_optimizer_state(optimizer)

    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert parameter in optimizer.state
    transaction.restore()
    assert parameter not in optimizer.state


def test_nested_tensor_and_scalar_state_is_restored():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    optimizer.state[parameter]['nested'] = {
        'buffer': torch.tensor([2.0]),
        'counter': 3,
    }
    nested_buffer = optimizer.state[parameter]['nested']['buffer']
    transaction = snapshot_optimizer_state(optimizer)

    optimizer.state[parameter]['nested']['buffer'].fill_(99.0)
    optimizer.state[parameter]['nested']['counter'] = 100
    transaction.restore()

    assert optimizer.state[parameter]['nested']['buffer'] is nested_buffer
    assert torch.equal(nested_buffer, torch.tensor([2.0]))
    assert optimizer.state[parameter]['nested']['counter'] == 3


def test_context_rolls_back_on_error():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    baseline = _state_clone(optimizer, parameter)

    with pytest.raises(RuntimeError):
        with snapshot_optimizer_state(optimizer):
            parameter.grad = torch.full_like(parameter, 4.0)
            optimizer.step()
            raise RuntimeError('probe failed')

    restored = _state_clone(optimizer, parameter)
    assert torch.equal(restored['exp_avg'], baseline['exp_avg'])
    assert torch.equal(restored['exp_avg_sq'], baseline['exp_avg_sq'])
    assert restored['step'] == baseline['step']


def test_muon_probe_state_is_restored():
    from omegaconf import OmegaConf
    from verl.workers.muon import build_muon

    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    config = OmegaConf.create({
        "lr": 0.1, "weight_decay": 0.0, "betas": (0.9, 0.999),
        "muon_momentum": 0.95, "muon_ns_steps": 2, "muon_nesterov": True,
        "muon_distributed_backend": "gather_scatter",
    })
    optimizer = build_muon(model, config)
    loss = model(torch.ones(2, 4)).sum()
    loss.backward()
    optimizer.step()
    baseline = optimizer.state[model[0].weight]["momentum_buffer"].detach().clone()

    transaction = snapshot_optimizer_state(optimizer)
    for value in (2.0, 3.0):
        optimizer.zero_grad(set_to_none=True)
        model(torch.full((2, 4), value)).sum().backward()
        optimizer.step()
    transaction.restore()

    restored = optimizer.state[model[0].weight]["momentum_buffer"]
    assert torch.equal(restored, baseline)


def _run_two_probe_steps(parameter, optimizer):
    """Mimic GXPO's two probe steps so the post-probe state is well defined."""
    for grad in (-3.0, 5.0):
        parameter.grad = torch.full_like(parameter, grad)
        optimizer.step()


def test_keeping_probe_state_needs_no_second_transaction():
    """The `transactional_fast_state` (no-refresh) mode is a no-op on the state.

    The trainer used to snapshot the post-probe state, restore the pre-probe
    snapshot over it, then restore the post-probe snapshot back -- a round trip
    that lands exactly where simply not restoring lands. `_gxpo_training_step`
    relies on that equivalence to skip both the extra snapshot (a full AdamW-state
    clone) and the two copy passes; pin it here so the two can never diverge.
    """
    def build():
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        parameter.grad = torch.tensor([0.25, -0.5])
        optimizer.step()
        return parameter, optimizer

    # Arm A: the old snapshot -> restore(base) -> restore(fast) round trip.
    parameter_a, optimizer_a = build()
    base_state = _state_clone(optimizer_a, parameter_a)
    base_transaction = snapshot_optimizer_state(optimizer_a)
    _run_two_probe_steps(parameter_a, optimizer_a)
    fast_transaction = snapshot_optimizer_state(optimizer_a)
    base_transaction.restore()
    fast_transaction.restore()
    round_tripped = _state_clone(optimizer_a, parameter_a)

    # Arm B: the new path -- the probe mutations are simply left in place.
    parameter_b, optimizer_b = build()
    _run_two_probe_steps(parameter_b, optimizer_b)
    kept = _state_clone(optimizer_b, parameter_b)

    for key, value in kept.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(round_tripped[key], value), key
        else:
            assert round_tripped[key] == value, key

    # ... and this is genuinely the post-probe state, not the pre-probe one:
    # both moments moved and AdamW's step counter advanced by the two probes.
    assert not torch.equal(kept['exp_avg'], base_state['exp_avg'])
    assert not torch.equal(kept['exp_avg_sq'], base_state['exp_avg_sq'])
    assert int(kept['step']) == int(base_state['step']) + 2


def test_keeping_probe_state_preserves_optimizer_tensor_identity():
    """Skipping the round trip keeps the optimizer's own state tensors in place."""
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.tensor([0.25, -0.5])
    optimizer.step()

    transaction = snapshot_optimizer_state(optimizer)
    exp_avg_ref = optimizer.state[parameter]['exp_avg']
    _run_two_probe_steps(parameter, optimizer)

    # The no-refresh mode never calls restore(), so no reallocation can happen.
    assert optimizer.state[parameter]['exp_avg'] is exp_avg_ref
    # The snapshot stays usable as the failure-path rollback either way.
    transaction.restore()
    assert optimizer.state[parameter]['exp_avg'] is exp_avg_ref


def test_keeping_probe_state_matches_round_trip_on_the_very_first_step():
    """Step 0 edge case: the pre-probe snapshot is empty, the probes create the state.

    The old round trip deleted every probe-created entry (none were in the base
    snapshot) and then re-cloned them from the fast snapshot; skipping both must
    land on the same values.
    """
    def build():
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        return parameter, torch.optim.AdamW([parameter], lr=0.1)

    parameter_a, optimizer_a = build()
    assert not optimizer_a.state, 'AdamW should hold no state before the first step'
    base_transaction = snapshot_optimizer_state(optimizer_a)
    _run_two_probe_steps(parameter_a, optimizer_a)
    fast_transaction = snapshot_optimizer_state(optimizer_a)
    base_transaction.restore()
    assert not optimizer_a.state, 'the base restore drops every probe-created entry'
    fast_transaction.restore()
    round_tripped = _state_clone(optimizer_a, parameter_a)

    parameter_b, optimizer_b = build()
    _run_two_probe_steps(parameter_b, optimizer_b)
    kept = _state_clone(optimizer_b, parameter_b)

    assert kept, 'the probe steps must have created optimizer state'
    for key, value in kept.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(round_tripped[key], value), key
        else:
            assert round_tripped[key] == value, key
