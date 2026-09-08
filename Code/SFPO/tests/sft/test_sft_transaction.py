"""Static and CPU checks for transactional SFT GXPO wiring."""

import ast
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "verl" / "trainer" / "fsdp_sft_trainer.py"


def _gxpo_step_source():
    source = TRAINER.read_text()
    tree = ast.parse(source)
    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "_gxpo_training_step")
    return ast.get_source_segment(source, step)


def test_sft_gxpo_uses_local_optimizer_transaction():
    step_source = _gxpo_step_source()
    # The pre-probe snapshot is unconditional: both modes need it as the
    # failure-path rollback for a non-finite pass.
    assert "self._snapshot_optimizer_state(self.optimizer)" in step_source
    # Two literal step sites -- standard_step() and probe_optimizer_step(), the
    # latter driving all three GXPO passes.
    assert step_source.count("self.optimizer.step()") == 2
    assert "def probe_optimizer_step()" in step_source
    assert "def restore_probe_state()" in step_source
    # The rollback happens before the slow correction, never after it.
    assert step_source.index("optimizer_transaction.restore()") < step_source.index(
        "# Pass 3: slow correction at theta_tilde")


def test_sft_gxpo_rolls_back_probe_state_only_in_transactional_mode():
    """`transactional` refreshes the optimizer state; `transactional_fast_state` keeps it.

    The no-refresh mode must reach Pass 3 without any restore at all -- no second
    snapshot, no round trip (see tests/gxpo/test_optimizer_transaction.py for why
    those are equivalent).
    """
    step_source = _gxpo_step_source()
    guarded = ("if self.gxpo_optimizer_state_mode == 'transactional':\n"
               "            optimizer_transaction.restore()")
    assert guarded in step_source
    # The post-reposition restore is reached only through that guard: every other
    # `optimizer_transaction.restore()` belongs to restore_probe_state().
    assert step_source.count("optimizer_transaction.restore()") == 2
    assert "fast_optimizer_transaction" not in step_source


def test_sft_scheduler_is_outside_gxpo_and_runs_once_per_training_step():
    source = TRAINER.read_text()
    tree = ast.parse(source)
    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "training_step")
    step_source = ast.get_source_segment(source, step)
    assert step_source.count("self.lr_scheduler.step()") == 1
    assert step_source.index("self.lr_scheduler.step()") > step_source.index(
        "self._gxpo_training_step(batch)")


def test_transaction_preserves_final_adam_update_after_probe_restore():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    baseline_parameter = parameter.detach().clone()
    baseline_state = {key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                      for key, value in optimizer.state[parameter].items()}

    from verl.workers.actor.optimizer_transaction import snapshot_optimizer_state
    transaction = snapshot_optimizer_state(optimizer)
    parameter.grad = torch.full_like(parameter, 2.0)
    optimizer.step()
    parameter.grad = torch.full_like(parameter, 3.0)
    optimizer.step()
    transaction.restore()
    assert not torch.equal(parameter, baseline_parameter)
    for key, value in baseline_state.items():
        actual = optimizer.state[parameter][key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(actual, value)
        else:
            assert actual == value
