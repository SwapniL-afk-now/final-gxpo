"""Memory-efficient sparse teacher-distribution losses."""

from __future__ import annotations

import torch


def sparse_topk_kd_loss(
    student_logits: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    teacher_counts: torch.Tensor,
    token_mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return summed T-scaled KD loss, valid-token count, and per-token losses.

    Teacher probabilities are renormalized over their stored top-K support. The
    student normalizer still spans the full vocabulary. This formulation avoids
    materializing a full-vocabulary FP32 log-softmax tensor.
    """
    if temperature <= 0:
        raise ValueError("KD temperature must be positive")
    if student_logits.ndim != 3 or teacher_ids.ndim != 3:
        raise ValueError("KD logits and teacher IDs must be rank-3 tensors")
    if teacher_ids.shape != teacher_logprobs.shape:
        raise ValueError("Teacher IDs and log-probabilities must have the same shape")
    if student_logits.shape[:2] != teacher_ids.shape[:2]:
        raise ValueError("Student and teacher token dimensions do not align")
    if teacher_counts.shape != token_mask.shape or teacher_counts.shape != student_logits.shape[:2]:
        raise ValueError("KD count and token masks must match student token dimensions")

    valid = token_mask.bool() & (teacher_counts > 0)
    if not bool(valid.any()):
        zero = student_logits.sum() * 0.0
        return zero, valid.sum(), torch.zeros_like(token_mask, dtype=torch.float32)

    logits = student_logits[valid]
    ids = teacher_ids[valid].long()
    logprobs = teacher_logprobs[valid].float()
    counts = teacher_counts[valid].long()
    width = ids.shape[-1]
    entry_mask = torch.arange(width, device=ids.device).unsqueeze(0) < counts.unsqueeze(1)
    if bool(((ids < 0) | (ids >= student_logits.shape[-1]))[entry_mask].any()):
        raise ValueError("Teacher top-K contains a token ID outside the student vocabulary")

    scaled_teacher = (logprobs / temperature).masked_fill(~entry_mask, float("-inf"))
    teacher_probs = torch.softmax(scaled_teacher, dim=-1).masked_fill(~entry_mask, 0.0)
    selected_logits = torch.gather(logits, dim=-1, index=ids).float() / temperature
    # Keep the full-vocabulary partition function without allocating full log-probs.
    log_partition = torch.logsumexp(logits / temperature, dim=-1).float()
    valid_losses = (log_partition - (teacher_probs * selected_logits).sum(dim=-1)) * (
        temperature ** 2
    )
    token_losses = torch.zeros_like(token_mask, dtype=torch.float32)
    token_losses[valid] = valid_losses
    return valid_losses.sum(), valid.sum(), token_losses
