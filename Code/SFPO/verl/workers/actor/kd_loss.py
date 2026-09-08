# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline top-K forward-KL distillation loss for the GXPO actor.

Ported from the on-policy KD prototype
(``Joint-Embedding-Guided-Policy-Optimization/verl/trainer/distillation/fsdp/losses.py``)
with one functional change that fixes the multi-update OOM: the student
``[tokens, vocab]`` tensor is never upcast to FP32 in full. The full-vocabulary
normalizer (``logsumexp``) is computed in token chunks (default 2048), so the
transient FP32 peak is ``chunk * vocab * 4B`` (~1.2GB at V=152k) instead of
``tokens * vocab * 4B`` (~5GB at 8192 tokens).

The GXPO 3-pass update (``dp_actor.update_policy_gxpo``) is untouched: this
module only replaces the *loss value* inside ``_backward_minibatch``. Multiple
passes therefore reuse the same peak instead of stacking allocations, matching
the PPO mini-batch memory behaviour.

Expected tensor layout (dense, response positions only):
  student_logits:      [N, V]  (flattened response tokens, any float dtype)
  teacher_topk_logps:  [N, K]  (float; non-finite entries are neutralized)
  teacher_topk_ids:    [N, K]  (long, vocab indices)
  response_mask:       [N]     (bool, True for real response tokens)

Returns per-token ``distillation_losses``, ``student_mass``, ``teacher_mass``.
"""

import torch

# Default token chunk for the FP32 normalizer. 2048 * 152064 * 4B ~= 1.2GB.
KD_TOPK_CHUNK_TOKENS = 2048


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Forward KL(p || q) from log-probs; inputs are [N, K], output [N]."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1)


def reverse_kl_topk_rest(
    s_topk: torch.Tensor,
    t_topk: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Reverse KL(q || p) over teacher top-K dims plus one `rest` bucket.

    Inputs are [C, K]: student top-K log-probs, teacher top-K log-probs, and a
    bool mask that is False for shape-preserving out-of-vocabulary sentinel
    ranks (teacher logprob below -100). Output is [C].

    The `rest` bucket holds each side's off-top-K mass (1 - mass), so the
    student cannot hide probability outside the teacher's top-K: mass the
    student places where the teacher has none is penalized, which is the
    mode-seeking behaviour that keeps on-policy distillation sharp. Sentinel
    ranks are excluded from both the top-K terms and the masses.
    """
    s = s_topk.exp() * valid
    t = t_topk.exp() * valid
    m_s = s.sum(dim=-1, keepdim=True)
    m_t = t.sum(dim=-1, keepdim=True)
    rest_s = (1.0 - m_s).clamp(0.0, 1.0)
    rest_t = (1.0 - m_t).clamp(0.0, 1.0)
    topk_term = (s * (s_topk.clamp_min(-50.0) - t_topk) * valid).sum(dim=-1)
    # log() inputs are eps-guarded so a zero-mass side contributes exactly 0
    # instead of NaN (0 * -inf); clamp the per-token loss at the call site.
    rest_term = (
        rest_s.squeeze(-1)
        * (rest_s.squeeze(-1).clamp_min(eps).log() - rest_t.squeeze(-1).clamp_min(eps).log())
    )
    return topk_term + rest_term


def compute_forward_kl_topk_chunked(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    log_prob_min_clamp: float | None = -10.0,
    loss_max_clamp: float | None = 10.0,
    chunk_tokens: int = KD_TOPK_CHUNK_TOKENS,
) -> dict:
    """Chunked top-K forward KL. Never materializes a full [N, V] FP32 copy."""
    assert student_logits.dim() == 2, f"student_logits must be [N, V], got {tuple(student_logits.shape)}"
    n_tokens, vocab = student_logits.shape
    assert teacher_topk_log_probs.shape == (n_tokens, teacher_topk_log_probs.shape[-1])
    assert teacher_topk_ids.shape == teacher_topk_log_probs.shape
    topk = teacher_topk_log_probs.shape[-1]
    chunk_tokens = max(1, int(chunk_tokens))

    # Neutralize missing/non-finite teacher tail ranks (matches prototype).
    teacher_topk_log_probs = torch.nan_to_num(teacher_topk_log_probs, nan=-1e4, posinf=0.0, neginf=-1e4)

    loss_parts, smass_parts, tmass_parts = [], [], []
    for start in range(0, n_tokens, chunk_tokens):
        end = min(start + chunk_tokens, n_tokens)
        # Only this slice is upcast; freed at the end of each iteration.
        lf = student_logits[start:end].float()  # [C, V]
        lz = torch.logsumexp(lf, dim=-1, keepdim=True)  # [C, 1]
        topk_logits = torch.gather(lf, dim=-1, index=teacher_topk_ids[start:end])  # [C, K]
        s_topk = topk_logits - lz
        t_topk = teacher_topk_log_probs[start:end]
        if log_prob_min_clamp is not None:
            s_topk = s_topk.clamp_min(log_prob_min_clamp)
            # Values below -100 are the shape-preserving out-of-vocabulary
            # sentinel emitted by the HF teacher. Keep them at zero mass;
            # clamping them to -10 would add a spurious 4.5e-5 probability per
            # invalid rank and can duplicate the pad-token id.
            t_topk = torch.where(
                t_topk < -100.0,
                t_topk,
                t_topk.clamp_min(log_prob_min_clamp),
            )
        s_mass = s_topk.exp().sum(dim=-1)
        t_mass = t_topk.exp().sum(dim=-1)
        per_tok = kl_divergence(log_q=s_topk, log_p=t_topk).clamp_min(0.0)
        if loss_max_clamp is not None:
            per_tok = per_tok.clamp_max(loss_max_clamp)
        loss_parts.append(per_tok)
        smass_parts.append(s_mass)
        tmass_parts.append(t_mass)
        del lf, lz, topk_logits, s_topk, t_topk, s_mass, t_mass, per_tok

    out = {
        "distillation_losses": torch.cat(loss_parts, dim=0),
        "student_mass": torch.cat(smass_parts, dim=0),
        "teacher_mass": torch.cat(tmass_parts, dim=0),
        "topk": topk,
    }
    del loss_parts, smass_parts, tmass_parts
    return out


def compute_reverse_kl_topk_chunked(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    log_prob_min_clamp: float | None = -10.0,
    loss_max_clamp: float | None = 10.0,
    chunk_tokens: int = KD_TOPK_CHUNK_TOKENS,
) -> dict:
    """Chunked top-K reverse KL(q_student || p_teacher) + rest bucket.

    Same tensor contract, chunking discipline, and return keys as
    ``compute_forward_kl_topk_chunked``; only the loss value differs. The
    ``student_mass`` / ``teacher_mass`` diagnostics keep their forward-style
    definitions so dashboards stay comparable across the switch.

    Use this for ON-POLICY distillation (student trains on its own rollouts):
    forward KL is mass-covering and pumps entropy once the student's prefixes
    push the teacher into flat, low-top-K-mass regimes, while reverse KL is
    mode-seeking and keeps the student concentrated on the teacher's best
    tokens. Off-policy distillation on fixed teacher data can stay forward.
    """
    assert student_logits.dim() == 2, f"student_logits must be [N, V], got {tuple(student_logits.shape)}"
    n_tokens, vocab = student_logits.shape
    assert teacher_topk_log_probs.shape == (n_tokens, teacher_topk_log_probs.shape[-1])
    assert teacher_topk_ids.shape == teacher_topk_log_probs.shape
    topk = teacher_topk_log_probs.shape[-1]
    chunk_tokens = max(1, int(chunk_tokens))

    teacher_topk_log_probs = torch.nan_to_num(teacher_topk_log_probs, nan=-1e4, posinf=0.0, neginf=-1e4)

    loss_parts, smass_parts, tmass_parts = [], [], []
    for start in range(0, n_tokens, chunk_tokens):
        end = min(start + chunk_tokens, n_tokens)
        lf = student_logits[start:end].float()  # [C, V]
        lz = torch.logsumexp(lf, dim=-1, keepdim=True)  # [C, 1]
        topk_logits = torch.gather(lf, dim=-1, index=teacher_topk_ids[start:end])  # [C, K]
        s_topk = topk_logits - lz
        t_topk = teacher_topk_log_probs[start:end]
        if log_prob_min_clamp is not None:
            s_topk = s_topk.clamp_min(log_prob_min_clamp)
            t_topk = torch.where(
                t_topk < -100.0,
                t_topk,
                t_topk.clamp_min(log_prob_min_clamp),
            )
        valid = (teacher_topk_log_probs[start:end] >= -100.0).to(lf.dtype)
        s_mass = s_topk.exp().sum(dim=-1)
        t_mass = (t_topk.exp() * valid).sum(dim=-1)
        per_tok = reverse_kl_topk_rest(s_topk, t_topk, valid).clamp_min(0.0)
        if loss_max_clamp is not None:
            per_tok = per_tok.clamp_max(loss_max_clamp)
        loss_parts.append(per_tok)
        smass_parts.append(s_mass)
        tmass_parts.append(t_mass)
        del lf, lz, topk_logits, s_topk, t_topk, valid, s_mass, t_mass, per_tok

    out = {
        "distillation_losses": torch.cat(loss_parts, dim=0),
        "student_mass": torch.cat(smass_parts, dim=0),
        "teacher_mass": torch.cat(tmass_parts, dim=0),
        "topk": topk,
    }
    del loss_parts, smass_parts, tmass_parts
    return out
