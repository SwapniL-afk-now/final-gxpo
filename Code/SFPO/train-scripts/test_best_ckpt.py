"""Checks for best-pass@1 checkpoint selection and resume-checkpoint retention.

Run: python train-scripts/test_best_ckpt.py
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ckpt_steps_to_remove


def _score(metrics):
    # _best_ckpt_score touches no instance state, so call it unbound against a bare object.
    return RayPPOTrainer._best_ckpt_score(object(), metrics)


def test_score_uses_exact_wandb_metric():
    m = {
        "eval_greedy/avg_pass1": 0.42,
        "val/pass_at_1/amc23": 0.99,
        "val/pass_at_1/aime24": 0.01,
        "eval_greedy/avg_pass1/seed0": 0.01,
    }
    assert _score(m) == 0.42, _score(m)

    # Missing or non-finite W&B metric must never replace an existing best.
    assert _score({}) == float("-inf")
    assert _score({"eval_greedy/avg_pass1": float("nan")}) == float("-inf")


def test_retention_keeps_latest_and_pins_best():
    # nothing to delete while under the keep budget
    assert ckpt_steps_to_remove([50, 100], keep=2, best_step=50) == []

    # oldest go first, but the best step survives even when it is the oldest
    assert ckpt_steps_to_remove([50, 100, 150, 200], keep=2, best_step=50) == [100]
    assert ckpt_steps_to_remove([50, 100, 150, 200], keep=2, best_step=None) == [50, 100]

    # best inside the keep window is not double-counted
    assert ckpt_steps_to_remove([50, 100, 150], keep=2, best_step=150) == [50]

    # unsorted input still resolves by step order
    assert ckpt_steps_to_remove([200, 50, 150, 100], keep=2, best_step=50) == [100]


def test_best_is_always_recoverable():
    """The pinned best must never be deleted, across a full 300-step run."""
    on_disk, best_step, best_score = [], None, float('-inf')
    rng = np.random.default_rng(0)
    for step in range(10, 301, 10):
        score = float(rng.random())
        if score > best_score:
            best_score, best_step = score, step
        if step % 50 == 0 or step == best_step:
            on_disk.append(step)
            for s in ckpt_steps_to_remove(on_disk, 2, best_step):
                on_disk.remove(s)
        assert best_step in on_disk, f'lost best ckpt {best_step} at step {step}: {on_disk}'
    print(f'  best step {best_step} still on disk; final dirs {on_disk}')


if __name__ == '__main__':
    test_score_uses_exact_wandb_metric()
    test_retention_keeps_latest_and_pins_best()
    test_best_is_always_recoverable()
    print('ok')
