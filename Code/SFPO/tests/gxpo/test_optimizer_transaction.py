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
