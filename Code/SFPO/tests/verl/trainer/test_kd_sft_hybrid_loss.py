"""Unit tests for the hybrid SFT(CE) + KL top-K distillation loss added to
verl/trainer/kd_sft_loss.py and verl/trainer/kd_sft_trainer.py.

No GPU / FSDP model needed: compute_forward_kl_topk_chunked and
combine_hybrid_kd_loss are plain tensor functions, exercised here on CPU.
"""
import math

import pytest
import torch

from verl.trainer.kd_sft_loss import compute_forward_kl_topk_chunked
from verl.trainer.kd_sft_trainer import combine_hybrid_kd_loss


def _toy_batch(vocab=6, topk=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    student_logits = torch.randn(4, vocab, generator=g)
    teacher_ids = torch.stack([torch.randperm(vocab, generator=g)[:topk] for _ in range(4)])
    teacher_raw = torch.randn(4, topk, generator=g)
    teacher_log_probs = teacher_raw.log_softmax(dim=-1)
    return student_logits, teacher_log_probs, teacher_ids


def test_temperature_default_is_one_and_matches_omitted_arg():
    student_logits, t_lp, t_ids = _toy_batch()
    out_default = compute_forward_kl_topk_chunked(student_logits, t_lp, t_ids)
    out_explicit = compute_forward_kl_topk_chunked(student_logits, t_lp, t_ids, temperature=1.0)
    assert torch.allclose(out_default['distillation_losses'], out_explicit['distillation_losses'])


def test_temperature_matches_manual_hinton_formula():
    """Hand-verify the T-scaled KL against a from-scratch computation for one
    row: temper student logits and the teacher's top-K distribution, forward
    KL, scale by T^2."""
    torch.manual_seed(0)
    V, K, T = 8, 4, 2.0
    student_logits = torch.randn(1, V)
    ids = torch.randperm(V)[:K].unsqueeze(0)
    teacher_raw = torch.randn(1, K)
    teacher_log_probs = teacher_raw.log_softmax(dim=-1)

    out = compute_forward_kl_topk_chunked(student_logits, teacher_log_probs, ids, temperature=T)

    # Manual reference.
    lf = student_logits / T
    lz = torch.logsumexp(lf, dim=-1, keepdim=True)
    s_topk = torch.gather(lf, -1, ids) - lz
    t_topk = teacher_log_probs / T
    t_topk = t_topk - torch.logsumexp(t_topk, dim=-1, keepdim=True)
    p = t_topk.exp()
    ref_kl = (p * (t_topk - s_topk)).sum(dim=-1)
    ref = ref_kl * (T ** 2)

    assert torch.allclose(out['distillation_losses'], ref, atol=1e-5)


def test_temperature_scaling_changes_loss_value():
    student_logits, t_lp, t_ids = _toy_batch()
    out_t1 = compute_forward_kl_topk_chunked(student_logits, t_lp, t_ids, temperature=1.0)
    out_t2 = compute_forward_kl_topk_chunked(student_logits, t_lp, t_ids, temperature=2.0)
    assert not torch.allclose(out_t1['distillation_losses'], out_t2['distillation_losses'])


def test_all_sentinel_row_stays_near_zero_under_temperature():
    """A row where every top-K rank is the shape-preserving OOV sentinel
    (<-100) must not be turned into a spurious uniform target by the
    temperature renormalization -- loss should stay ~0, not spike."""
    V, K = 8, 3
    student_logits = torch.randn(1, V)
    ids = torch.arange(K).unsqueeze(0)
    teacher_log_probs = torch.full((1, K), -1e4)  # all sentinel

    out = compute_forward_kl_topk_chunked(
        student_logits, teacher_log_probs, ids,
        log_prob_min_clamp=-10.0, loss_max_clamp=10.0, temperature=2.0,
    )
    assert torch.isfinite(out['distillation_losses']).all()
    assert out['distillation_losses'].item() < 1e-3
    assert out['teacher_mass'].item() < 1e-3


def test_mixed_sentinel_row_excludes_sentinel_from_renormalization():
    """One real rank + sentinels: after tempering, the real rank should carry
    ~all the mass (not diluted by treating sentinels as valid alternatives)."""
    V, K = 8, 3
    student_logits = torch.zeros(1, V)
    ids = torch.arange(K).unsqueeze(0)
    teacher_log_probs = torch.tensor([[0.0, -1e4, -1e4]])  # one real entry (log p=0 -> p=1)

    out = compute_forward_kl_topk_chunked(
        student_logits, teacher_log_probs, ids,
        log_prob_min_clamp=-10.0, loss_max_clamp=10.0, temperature=2.0,
    )
    assert torch.isfinite(out['distillation_losses']).all()
    assert out['teacher_mass'].item() == pytest.approx(1.0, abs=1e-4)


def test_combine_hybrid_kd_loss_defaults_reproduce_pure_ce():
    ce = torch.tensor([1.0, 2.0, 3.0])
    kd = torch.tensor([10.0, 20.0, 30.0])
    combined = combine_hybrid_kd_loss(ce, kd, kd_alpha=1.0, kd_beta=0.0)
    assert torch.allclose(combined, ce)


def test_combine_hybrid_kd_loss_defaults_reproduce_pure_kd():
    ce = torch.tensor([1.0, 2.0, 3.0])
    kd = torch.tensor([10.0, 20.0, 30.0])
    combined = combine_hybrid_kd_loss(ce, kd, kd_alpha=0.0, kd_beta=1.0)
    assert torch.allclose(combined, kd)


def test_combine_hybrid_kd_loss_weighted_sum():
    ce = torch.tensor([1.0, 2.0])
    kd = torch.tensor([3.0, 4.0])
    combined = combine_hybrid_kd_loss(ce, kd, kd_alpha=0.5, kd_beta=0.5)
    assert torch.allclose(combined, torch.tensor([2.0, 3.0]))


def test_combine_hybrid_kd_loss_shape_mismatch_raises():
    with pytest.raises(AssertionError):
        combine_hybrid_kd_loss(torch.zeros(3), torch.zeros(4), 0.5, 0.5)


def test_combine_hybrid_kd_loss_rejects_negative_weights():
    with pytest.raises(AssertionError):
        combine_hybrid_kd_loss(torch.zeros(3), torch.zeros(3), -0.1, 0.5)
