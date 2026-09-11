#!/usr/bin/env python
"""Checks for core_algos.agg_loss / compute_policy_loss(loss_agg_mode=...).

Run: cd final-gxpo/Code/SFPO && PYTHONPATH=. python tools/opd2/test_loss_agg.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from verl.trainer.ppo.core_algos import LOSS_AGG_MODES, agg_loss, compute_policy_loss


def main():
    torch.manual_seed(0)
    B, R = 6, 20
    loss = torch.randn(B, R)
    # Deliberately uneven lengths: every mode is identical when they are equal,
    # so an equal-length fixture would prove nothing.
    lens = torch.tensor([20, 15, 10, 5, 2, 1])
    mask = (torch.arange(R)[None, :] < lens[:, None]).float()

    # 1. token-mean is the historical behavior, unchanged.
    import verl.utils.torch_functional as verl_F
    assert torch.allclose(agg_loss(loss, mask, 'token-mean'),
                          verl_F.masked_mean(loss, mask)), 'default mode drifted'

    # 2. seq-mean-token-mean == trl's loss_type="grpo", literally.
    trl_grpo = ((loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    assert torch.allclose(agg_loss(loss, mask, 'seq-mean-token-mean'), trl_grpo)

    # 3. seq-mean-token-sum-norm == trl's loss_type="dr_grpo".
    trl_dr = (loss * mask).sum() / (B * R)
    assert torch.allclose(agg_loss(loss, mask, 'seq-mean-token-sum-norm'), trl_dr)

    # 4. The modes actually differ on uneven lengths (guards a silent no-op).
    vals = [agg_loss(loss, mask, m).item() for m in LOSS_AGG_MODES]
    assert len(set(round(v, 6) for v in vals)) == 3, f'modes collapsed: {vals}'

    # 5. ...and agree when every sequence has the full length.
    full = torch.ones(B, R)
    assert torch.allclose(agg_loss(loss, full, 'token-mean'),
                          agg_loss(loss, full, 'seq-mean-token-mean'))

    # 6. seq-mean-token-mean composes across gradient accumulation. dp_actor
    #    scales each micro-batch by len(data)/ppo_mini_batch_size -- a SEQUENCE
    #    count weight -- so the accumulated gradient must equal the whole-batch
    #    loss. This is the property that makes the mode correct under
    #    use_dynamic_bsz, and it is exactly what token-mean does NOT satisfy.
    acc = sum(agg_loss(loss[s:s + 2], mask[s:s + 2], 'seq-mean-token-mean') * (2 / B)
              for s in range(0, B, 2))
    assert torch.allclose(acc, agg_loss(loss, mask, 'seq-mean-token-mean'), atol=1e-6), \
        'seq-mean-token-mean does not compose across micro-batches'

    # 7. compute_policy_loss routes the mode through, and its clipfrac/KL
    #    diagnostics stay mode-independent.
    old_lp = torch.randn(B, R)
    lp = old_lp + 0.01 * torch.randn(B, R)
    adv = torch.randn(B, R)
    outs = {m: compute_policy_loss(old_lp, lp, adv, mask, 0.2, loss_agg_mode=m)
            for m in LOSS_AGG_MODES}
    assert len(set(round(o[0].item(), 6) for o in outs.values())) == 3, 'mode not routed'
    for m in LOSS_AGG_MODES[1:]:
        assert torch.allclose(outs[m][1], outs['token-mean'][1]), 'clipfrac moved with mode'
        assert torch.allclose(outs[m][2], outs['token-mean'][2]), 'ppo_kl moved with mode'

    # 8. An unknown mode fails loudly rather than silently falling back.
    try:
        agg_loss(loss, mask, 'nope')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown loss_agg_mode silently accepted')

    print('loss_agg self-check OK  ' + '  '.join(
        f'{m}={v:+.5f}' for m, v in zip(LOSS_AGG_MODES, vals)))


if __name__ == '__main__':
    main()
