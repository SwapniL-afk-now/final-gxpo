#!/usr/bin/env python3
"""Unit checks for the SFT scheduler/fit-loop step-horizon contract."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verl.trainer.sft_utils import resolve_total_training_steps


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
    assert resolve_total_training_steps(10, 3, 17) == 17
    assert_raises(lambda: resolve_total_training_steps(0, 3), "steps_per_epoch")
    assert_raises(lambda: resolve_total_training_steps(10, 0), "total_epochs")
    assert_raises(lambda: resolve_total_training_steps(10, 3, 0), "total_training_steps")
    assert_raises(lambda: resolve_total_training_steps(10, 3, -1), "total_training_steps")
    print("SFT scheduler contract: PASS")


if __name__ == "__main__":
    main()
