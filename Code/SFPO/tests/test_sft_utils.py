#!/usr/bin/env python3
"""Unit checks for the SFT scheduler/fit-loop step-horizon contract."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verl.trainer.sft_utils import resolve_total_training_steps
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
import torch


def assert_raises(fn, message):
    try:
        fn()
    except ValueError as exc:
        assert message in str(exc), (str(exc), message)
    else:
        raise AssertionError("expected ValueError")


def main():
    assert resolve_total_training_steps(10, 3, None) == 30
    assert resolve_total_training_steps(10, 3, 500) == 30
    assert resolve_total_training_steps(10, 3, 0) == 30
    assert resolve_total_training_steps(10, 3, 17) == 17
    assert_raises(lambda: resolve_total_training_steps(0, 3), "steps_per_epoch")
    assert_raises(lambda: resolve_total_training_steps(10, 0), "total_epochs")
    assert_raises(lambda: resolve_total_training_steps(10, 3, -1), "total_training_steps")
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=1.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=4)
    observed = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])
    assert len(observed) == 4
    assert observed[-1] == 0.0, observed
    assert observed[0] > observed[-1]
    print("SFT scheduler termination/final LR: PASS")
    print("SFT scheduler contract: PASS")


if __name__ == "__main__":
    main()
