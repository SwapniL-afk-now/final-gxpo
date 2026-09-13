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
"""GRPO + SLED-Delta: canonical self-distillation signal and sign-gated loss.

SLED-Delta is a TOKEN-LEVEL self-distillation signal built from the SAME
frozen rollout snapshot that GRPO uses. The "virtual teacher" is not an
external model: it is an early-exit contrast of the rollout policy itself
(``logits_sled = (1 + alpha) * F - alpha * E`` with ``F`` the final-layer
logits and ``E`` the early-layer logits at the same position). With
``alpha = 0`` the teacher is bit-identical to the student (``q == p``).

Per response token, with ``log_softmax`` at the rollout temperature and
``E_p[X] = sum_v p(v) * X_v`` truncated to the STUDENT's top-K columns
(``sled_topk``, default 1024 -- the same near-lossless truncation OPD^2
uses: the weight is the student's own probability, ~0 outside its own
top-K)::

    R_delta(v) = log q(v) - log p(v)
    A_delta    = R_delta(y) - E_p[R_delta]            # frozen, token-level
    R_opd(v)   = log q(v) - log pi_theta(v)           # live student
    A_opd      = R_opd(y) - E_p[R_opd]                # live, token-level
    g          = 1[A_delta * A_opd > 0]               # OPD^2-style sign gate
    L_sled     = -mean(g * sg(A_delta) * log pi_theta(y))   # masked mean

The gate applies ONLY to the SLED term. The GRPO term keeps its own
sequence-level advantage and PPO clip untouched::

    L_total = grpo_coef * L_grpo + sled_coef * L_sled      (defaults 1.0 / 1.0)

With ``sled_coef = 0`` the SLED scoring phase is skipped entirely and the
update is bit-identical to ordinary GRPO.

Precision discipline (same lesson as OPD^2): temperature division and
``log_softmax`` run in FP32. The forward itself may be bf16; the contrast,
centering and gate are fp32.

Run ``python -m verl.workers.actor.sled_delta`` for the CPU self-check.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

# Token chunk for full-vocab log_softmax reductions. The transient peak is
# chunk * vocab * 4B (fp32). Same discipline as opd2_signal.py.
SLED_CHUNK_TOKENS = 512


def sled_contrast_logits(final_logits: torch.Tensor, early_logits: torch.Tensor,
                         alpha: float) -> torch.Tensor:
    """Early-exit self-contrast: ``(1 + alpha) * F - alpha * E`` in fp32.

    With ``alpha == 0`` this is bit-identical to ``F`` (IEEE multiply by 1.0
    and subtract of signed zero are exact), so ``q == p`` exactly and the
    downstream delta is exactly zero.
    """
    f = final_logits.float()
    e = early_logits.float()
    a = float(alpha)
    return (1.0 + a) * f - a * e


def log_softmax_fp32(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Full-vocab log_softmax in fp32 with the temperature divided in fp32."""
    return F.log_softmax(logits.float() / float(temperature), dim=-1)


def sled_truncated_frozen(p_topk_lp: torch.Tensor, q_topk_lp: torch.Tensor,
                          logp_gt: torch.Tensor, logq_gt: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Frozen per-token SLED statistics from top-K columns.

    Args:
        p_topk_lp: ``[n, K]`` fp32 student log-probs at the student's top-K
            columns (full-vocabulary-normalized, e.g. from
            ``opd2_signal.topk_from_logits``).
        q_topk_lp: ``[n, K]`` fp32 SLED-teacher log-probs at the SAME columns
            (full-vocabulary-normalized, e.g. from
            ``opd2_signal.gather_from_logits``).
        logp_gt / logq_gt: ``[n]`` fp32 log-probs of the sampled token.

    Returns ``A_delta`` (the centered token delta), both entropies, both KL
    directions, and ``E_p[log q]`` (needed later for the live OPD term). Every
    expectation is truncated to the top-K columns without renormalization --
    the OPD^2 convention, near-lossless because the weight is the student's
    own probability. When ``K >= V`` this is exact.
    """
    p = p_topk_lp.float().exp()
    q = q_topk_lp.float().exp()
    ep_logp = (p * p_topk_lp.float()).sum(-1)
    ep_logq = (p * q_topk_lp.float()).sum(-1)
    a_delta = (logq_gt.float() - logp_gt.float()) - (ep_logq - ep_logp)
    out = {
        'a_delta': torch.nan_to_num(a_delta, nan=0.0, posinf=0.0, neginf=0.0).float(),
        'ent_p': (-ep_logp).float(),
        'ent_q': (-(q * q_topk_lp.float()).sum(-1)).float(),
        'kl_q_p': ((q * (q_topk_lp.float() - p_topk_lp.float())).sum(-1)).float(),
        'kl_p_q': (ep_logp - ep_logq).float(),
        'ep_logq': ep_logq.float(),
    }
    return out


def live_opd_advantage(a_delta: torch.Tensor, logq_gt: torch.Tensor, ep_logq: torch.Tensor,
                       live_logprob_y: torch.Tensor, live_topk_lp: torch.Tensor,
                       p_topk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Live teacher-student OPD advantage and the sign gate.

    ``A_opd = (log q(y) - E_p[log q]) - (log pi(y) - E_p[log pi])``.
    Both brackets use the SAME top-K columns (weights ``p_topk``) -- the
    frozen ``ep_logq`` was accumulated over them at scoring time, and
    ``E_p[log pi]`` is accumulated over them here from the live log-probs
    ``live_topk_lp``. Same near-lossless truncation OPD^2 uses: the weight is
    the student's own probability, ~0 outside its own top-K.

    All inputs are detached target-side quantities. Returns
    ``(a_opd, gate)`` with ``gate = 1[a_delta * a_opd > 0]``.
    """
    ep_live = (p_topk.float() * live_topk_lp.float()).sum(-1)
    a_opd = (logq_gt.float() - ep_logq.float()) - (live_logprob_y.float() - ep_live)
    a_opd = torch.nan_to_num(a_opd, nan=0.0, posinf=0.0, neginf=0.0)
    gate = (a_delta.float() * a_opd > 0).float()
    return a_opd.float(), gate


def sled_gated_per_token_loss(live_logprob_y: torch.Tensor, a_delta: torch.Tensor,
                              gate: torch.Tensor) -> torch.Tensor:
    """``-g * sg(A_delta) * log pi_theta(y)``; only ``log pi`` gets gradient."""
    return -gate * a_delta.detach().float() * live_logprob_y


def agreement_quadrants(grpo_adv: torch.Tensor, a_delta: torch.Tensor,
                        mask: torch.Tensor) -> Dict[str, float]:
    """GRPO (sequence-level, broadcast) vs SLED (token-level) sign agreement.

    Diagnostic only: disagreement is EXPECTED (different credit assignment
    levels) and never gates either loss.
    """
    m = mask.bool()
    if int(m.sum().item()) == 0:
        return {k: float('nan') for k in ('agree', 'disagree', 'pp', 'pn', 'np', 'nn')}
    g = (grpo_adv[m] > 0).float()
    s = (a_delta[m] > 0).float()
    agree = (g == s).float().mean().item()
    return {
        'agree': agree,
        'disagree': 1.0 - agree,
        'pp': ((g == 1) & (s == 1)).float().mean().item(),
        'pn': ((g == 1) & (s == 0)).float().mean().item(),
        'np': ((g == 0) & (s == 1)).float().mean().item(),
        'nn': ((g == 0) & (s == 0)).float().mean().item(),
    }


def per_token_grad_weight_cosine(grpo_token_w: torch.Tensor, sled_token_w: torch.Tensor,
                                 mask: torch.Tensor) -> float:
    """Cosine between per-token ``dL/dlogpi`` weights over masked tokens.

    A cheap first-order proxy for "do the two objectives push the same tokens
    the same way". This is NOT a full-parameter gradient cosine (which would
    need two extra backward passes per step); it is logged as such.
    """
    m = mask.bool()
    a = (grpo_token_w.float() * m.float())[m]
    b = (sled_token_w.float() * m.float())[m]
    denom = a.norm() * b.norm()
    if int(m.sum().item()) == 0 or float(denom.item()) == 0.0:
        return float('nan')
    return float((a @ b / denom).item())


# --------------------------------------------------------------- self-check --
def _truncated_stats_for_check(f_raw, s_raw, gt, k, temperature=1.0):
    """Production truncation route (topk + gather + truncated stats)."""
    from verl.workers.actor.opd2_signal import gather_from_logits, topk_from_logits
    logp_gt, p_topk, idx = topk_from_logits(f_raw, gt, k, temperature=temperature,
                                            chunk_tokens=SLED_CHUNK_TOKENS)
    logq_gt, q_topk = gather_from_logits(s_raw, gt, idx, temperature=temperature,
                                         chunk_tokens=SLED_CHUNK_TOKENS)
    stats = sled_truncated_frozen(p_topk, q_topk, logp_gt, logq_gt)
    stats['logp_gt'] = logp_gt.float()
    stats['logq_gt'] = logq_gt.float()
    return stats


def _self_check():
    torch.manual_seed(0)
    n, v, k = 11, 64, 16

    # 1. alpha=0 is exact: q == p and A_delta == 0.
    f = torch.randn(n, v)
    e = torch.randn(n, v)
    assert torch.equal(sled_contrast_logits(f, e, 0.0), f.float()), 'alpha=0 must be exact'
    stats = _truncated_stats_for_check(f, sled_contrast_logits(f, e, 0.0),
                                       torch.randint(0, v, (n,)), k)
    assert float(stats['a_delta'].abs().max().item()) == 0.0, stats['a_delta'].abs().max()

    # 2. truncated stats == manual top-K centering on a nonzero signal.
    f2, s2 = torch.randn(n, v), torch.randn(n, v)
    gt = torch.randint(0, v, (n,))
    ref = _truncated_stats_for_check(f2, s2, gt, k, temperature=0.7)
    assert float(ref['a_delta'].abs().max().item()) > 1e-3, 'degenerate fixture'
    from verl.workers.actor.opd2_signal import gather_from_logits, topk_from_logits
    logp_gt, p_topk, idx = topk_from_logits(f2, gt, k, temperature=0.7,
                                            chunk_tokens=SLED_CHUNK_TOKENS)
    logq_gt, q_topk = gather_from_logits(s2, gt, idx, temperature=0.7,
                                         chunk_tokens=SLED_CHUNK_TOKENS)
    p = p_topk.exp()
    want = (logq_gt - logp_gt) - ((p * q_topk).sum(-1) - (p * p_topk).sum(-1))
    assert torch.allclose(ref['a_delta'], want, atol=1e-5), (ref['a_delta'] - want).abs().max()

    # 2b. truncation fidelity: at K=V the truncated route is the full-vocab
    # route; below V the error is bounded by the dropped tail (the sampled
    # token is force-included in the top-K, so only the E_p[R] tail matters:
    # |err| <= max_tail|R| * tail_mass).
    full = _truncated_stats_for_check(f2, s2, gt, v, temperature=0.7)
    logp_full = log_softmax_fp32(f2, 0.7)
    logq_full = log_softmax_fp32(s2, 0.7)
    pp = logp_full.exp()
    rr = logq_full - logp_full
    want_full = rr.gather(-1, gt.reshape(n, 1)).squeeze(-1) - (pp * rr).sum(-1)
    assert torch.allclose(full['a_delta'], want_full, atol=1e-5)
    trunc = _truncated_stats_for_check(f2, s2, gt, k, temperature=0.7)
    from verl.workers.actor.opd2_signal import topk_from_logits as _topk
    _, p_topk, idx = _topk(f2, gt, k, temperature=0.7, chunk_tokens=SLED_CHUNK_TOKENS)
    tail_mass = 1.0 - p_topk.exp().sum(-1)
    in_topk = torch.zeros(n, v, dtype=torch.bool).scatter_(-1, idx, True)
    max_tail_r = (rr.masked_fill(in_topk, 0.0)).abs().amax(-1)
    bound = max_tail_r * tail_mass + 1e-5
    err = (trunc['a_delta'] - full['a_delta']).abs()
    assert bool((err <= bound).all().item()), (err - bound).abs().max()
    assert float(tail_mass.max().item()) > 0, 'fixture must actually drop mass'

    # 3. gate truth table, through the real function: with E_p terms at zero,
    # a_opd reduces to logq_gt - live_y, so pin a_opd to the four sign cases.
    ad = torch.tensor([2.0, -2.0, 2.0, -2.0, 0.0])
    ao = torch.tensor([1.0, -1.0, -1.0, 1.0, 5.0])
    got_ao, got_g = live_opd_advantage(ad, ao, torch.zeros(5), torch.zeros(5),
                                       torch.zeros(5, 1), torch.zeros(5, 1))
    assert torch.allclose(got_ao, ao, atol=1e-6), got_ao
    assert torch.equal(got_g, torch.tensor([1., 1., 0., 0., 0.])), got_g

    # 4. gate only gates SLED: zero gate kills the SLED term, not a GRPO stand-in.
    lp = torch.zeros(6, requires_grad=True)
    adv = torch.tensor([1.0, -1.0, 2.0, -0.5, 0.0, 3.0])
    ratio = torch.exp(lp)  # old_logprob = 0; ratio = 1 is inside the clip range
    grpo_tok = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)
    gate = torch.tensor([0., 0., 1., 1., 0., 1.])
    a_delta = torch.tensor([5., -5., 5., -5., 5., 0.5])
    sled_tok = sled_gated_per_token_loss(lp, a_delta, gate)
    (grpo_tok.sum() + sled_tok.sum()).backward()
    # SLED grad is -g*A on gated tokens, 0 elsewhere; GRPO grad is -A everywhere.
    assert torch.allclose(lp.grad, -(adv + gate * a_delta), atol=1e-6), lp.grad

    print('[sled_delta] self-check OK')


if __name__ == '__main__':
    _self_check()
