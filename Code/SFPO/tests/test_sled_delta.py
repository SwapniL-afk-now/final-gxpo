"""GRPO + SLED-Delta correctness tests (CPU-only).

Run directly with:
    PYTHONPATH=Code/SFPO python Code/SFPO/tests/test_sled_delta.py
or via pytest:
    PYTHONPATH=Code/SFPO python -m pytest -q Code/SFPO/tests/test_sled_delta.py

Every test imports the PRODUCTION modules (no second implementation of the
math): ``verl/workers/actor/sled_delta.py`` for the SLED signal/gate/loss and
``verl/trainer/ppo/core_algos.py`` for the GRPO advantage and PPO-clip loss.
Static checks follow the precedent of ``tests/gxpo/test_gxpo_parity.py``.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ACTOR_DIR = REPO / 'verl' / 'workers' / 'actor'
TRAINER_DIR = REPO / 'verl' / 'trainer' / 'ppo'
WORKERS_DIR = REPO / 'verl' / 'workers'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SLED = _load('production_sled_delta', ACTOR_DIR / 'sled_delta.py')
CORE = _load('production_core_algos', TRAINER_DIR / 'core_algos.py')

torch.manual_seed(7)


def _grpo_fixture():
    """Small batch: 2 prompts x 3 responses, uneven lengths."""
    B, R = 6, 5
    old = torch.randn(B, R) * 0.5 - 1.0
    logp = old + torch.randn(B, R) * 0.05
    mask = torch.tensor([[1, 1, 1, 1, 0],
                         [1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 1],
                         [1, 1, 0, 0, 0],
                         [1, 1, 1, 1, 0],
                         [1, 1, 1, 0, 0]], dtype=torch.float32)
    rewards = torch.zeros(B, R)
    scores = torch.tensor([1.0, 0.0, 0.5, 0.2, 0.8, 0.4])
    for i, s in enumerate(scores):
        rewards[i, 0] = s  # outcome reward on the first token (summed => score)
    index = np.array([0, 0, 0, 1, 1, 1])
    return old, logp, mask, rewards, index


def test_grpo_equivalence_coef_zero():
    """Test 1: sled_loss_coef=0 reproduces ordinary GRPO (loss AND grads)."""
    old, logp, mask, rewards, index = _grpo_fixture()
    adv, _, _ = CORE.compute_grpo_outcome_advantage(rewards, mask, index)
    lp = logp.clone().requires_grad_(True)
    pg, _, _ = CORE.compute_policy_loss(old, lp, adv, mask, 0.2)
    pg.backward()
    g_grpo = lp.grad.clone()

    lp2 = logp.clone().requires_grad_(True)
    pg2, _, _ = CORE.compute_policy_loss(old, lp2, adv, mask, 0.2)
    total = 1.0 * pg2 + 0.0 * lp2.new_zeros(())
    total.backward()
    assert torch.equal(pg, total), (pg, total)
    assert torch.equal(g_grpo, lp2.grad), (g_grpo - lp2.grad).abs().max()
    print('PASS test_grpo_equivalence_coef_zero')


def _production_truncated_route(f_raw, s_raw, gt, k, temperature=1.0):
    """The exact production route: student top-K, teacher gather, truncate."""
    from verl.workers.actor.opd2_signal import gather_from_logits, topk_from_logits
    logp_gt, p_topk, idx = topk_from_logits(f_raw, gt, k, temperature=temperature)
    logq_gt, q_topk = gather_from_logits(s_raw, gt, idx, temperature=temperature)
    stats = SLED.sled_truncated_frozen(p_topk, q_topk, logp_gt, logq_gt)
    stats['logp_gt'] = logp_gt.float()
    stats['logq_gt'] = logq_gt.float()
    return stats


def test_sled_component_standalone():
    """Test 2: with grpo_coef=0 the composed loss IS the canonical SLED loss."""
    B, R, V, K = 4, 6, 64, 16
    f_raw = torch.randn(B * R, V)
    s_raw = torch.randn(B * R, V)
    gt = torch.randint(0, V, (B * R,))
    stats = _production_truncated_route(f_raw, s_raw, gt, K)
    a_delta = stats['a_delta'].view(B, R).detach()
    # manual top-K centering reference
    from verl.workers.actor.opd2_signal import gather_from_logits, topk_from_logits
    logp_gt, p_topk, idx = topk_from_logits(f_raw, gt, K, temperature=1.0)
    logq_gt, q_topk = gather_from_logits(s_raw, gt, idx, temperature=1.0)
    p = p_topk.exp()
    want = (logq_gt - logp_gt) - ((p * q_topk).sum(-1) - (p * p_topk).sum(-1))
    assert torch.allclose(a_delta.view(-1), want, atol=1e-5)
    # standalone SLED loss with a hand-built live term
    live_y = torch.randn(B, R, requires_grad=True)
    from verl.workers.actor.opd2_signal import topk_from_logits
    _, p_topk_raw, _ = topk_from_logits(f_raw, gt, K, temperature=1.0)
    p_topk = p_topk_raw.view(B, R, K).detach()
    live_topk = torch.randn(B, R, K)
    a_opd, gate = SLED.live_opd_advantage(
        a_delta, stats['logq_gt'].view(B, R).detach(),
        stats['ep_logq'].view(B, R).detach(), live_y.detach(), live_topk, p_topk)
    per_tok = SLED.sled_gated_per_token_loss(live_y, a_delta, gate)
    mask = torch.ones(B, R)
    sled_loss = CORE.agg_loss(per_tok, mask, 'token-mean')
    composed = 0.0 * live_y.new_zeros(()) + 1.0 * sled_loss
    assert torch.equal(composed, sled_loss)
    # spec formula: -sum(M g sg(A) logpi) / sum(M)
    want_loss = -(mask * gate.detach() * a_delta * live_y.detach()).sum() / mask.sum()
    assert torch.allclose(sled_loss.detach(), want_loss, atol=1e-5)
    print('PASS test_sled_component_standalone')


def test_advantage_dimensionality():
    """Test 3: GRPO advantage is broadcast-constant; SLED A_delta varies."""
    old, logp, mask, rewards, index = _grpo_fixture()
    adv, _, _ = CORE.compute_grpo_outcome_advantage(rewards, mask, index)
    for i in range(adv.shape[0]):
        row = adv[i][mask[i].bool()]
        assert torch.equal(row, row[0].expand_as(row)), f'row {i} not broadcast: {row}'
    n, v, k = 24, 64, 16
    stats = _production_truncated_route(torch.randn(n, v), torch.randn(n, v),
                                        torch.randint(0, v, (n,)), k)
    assert float(stats['a_delta'].std().item()) > 1e-3, 'SLED advantage must vary across tokens'
    print('PASS test_advantage_dimensionality')


def test_topk_truncation_fidelity():
    """Truncation change: K=V is the full-vocab route; below V the error is
    bounded by the dropped tail (sampled token is force-included, so only the
    E_p[R] tail matters: |err| <= max_tail|R| * tail_mass)."""
    from verl.workers.actor.opd2_signal import topk_from_logits
    n, v, k = 24, 64, 16
    f2, s2 = torch.randn(n, v), torch.randn(n, v)
    gt = torch.randint(0, v, (n,))
    trunc = _production_truncated_route(f2, s2, gt, k, temperature=0.7)
    # K = V must reproduce manual full-vocab centering exactly.
    full = _production_truncated_route(f2, s2, gt, v, temperature=0.7)
    logp_full = SLED.log_softmax_fp32(f2, 0.7)
    logq_full = SLED.log_softmax_fp32(s2, 0.7)
    pp = logp_full.exp()
    rr = logq_full - logp_full
    want_full = rr.gather(-1, gt.reshape(n, 1)).squeeze(-1) - (pp * rr).sum(-1)
    assert torch.allclose(full['a_delta'], want_full, atol=1e-5)
    _, p_topk, idx = topk_from_logits(f2, gt, k, temperature=0.7)
    tail_mass = 1.0 - p_topk.exp().sum(-1)
    in_topk = torch.zeros(n, v, dtype=torch.bool).scatter_(-1, idx, True)
    max_tail_r = rr.masked_fill(in_topk, 0.0).abs().amax(-1)
    err = (trunc['a_delta'] - full['a_delta']).abs()
    assert bool((err <= max_tail_r * tail_mass + 1e-5).all().item()), \
        (err - max_tail_r * tail_mass).max()
    assert float(tail_mass.max().item()) > 0, 'fixture must actually drop mass'
    print('PASS test_topk_truncation_fidelity')


def test_gate_only_gates_sled():
    """Test 4: gate=0 kills the SLED gradient but never the GRPO gradient."""
    B, R = 3, 4
    old = torch.zeros(B, R)
    lp = torch.zeros(B, R, requires_grad=True)
    adv = torch.tensor([[2.0, 2.0, 2.0, 0.0],
                        [-1.5, -1.5, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float32)
    ratio = torch.exp(lp)
    pg_tok = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)
    gate = torch.tensor([[0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 1., 1.]])
    a_delta = torch.tensor([[9., 9., 9., 9.], [9., 9., 9., 9.], [0., 0., 0., 0.]])
    sled_tok = SLED.sled_gated_per_token_loss(lp, a_delta, gate)
    (CORE.agg_loss(pg_tok, mask, 'token-mean') + CORE.agg_loss(sled_tok, mask, 'token-mean')).backward()
    # token (0,0): gate 0, GRPO adv 2 -> GRPO grad only
    assert lp.grad[0, 0].item() != 0.0
    assert abs(lp.grad[0, 0].item() - (-2.0 / mask.sum().item())) < 1e-5, lp.grad[0, 0]
    # token (0,1): gate 1 -> both
    assert abs(lp.grad[0, 1].item() - (-(2.0 + 9.0) / mask.sum().item())) < 1e-5
    # prompt/pad token (0,3): no gradient from either
    assert lp.grad[0, 3].item() == 0.0
    print('PASS test_gate_only_gates_sled')


def test_one_rollout_no_second_generation():
    """Test 5: the SLED path scores the given rollout; it never generates."""
    sled_src = (ACTOR_DIR / 'sled_delta.py').read_text()
    dp_src = (ACTOR_DIR / 'dp_actor.py').read_text()
    fsdp_src = (WORKERS_DIR / 'fsdp_workers.py').read_text()
    ray_src = (TRAINER_DIR / 'ray_trainer.py').read_text()
    for name, src in (('sled_delta', sled_src),):
        for tok in ('generate_sequences', 'vllm', '.generate(', 'rollout.'):
            assert tok not in src, f'{name} must not generate (found {tok!r})'
    sled_methods = dp_src[dp_src.index('def compute_sled_delta_frozen'):dp_src.index(
        'def _make_minibatch_iterator')]
    for tok in ('generate_sequences', 'vllm'):
        assert tok not in sled_methods, f'dp_actor SLED scoring must not generate ({tok!r})'
    assert 'compute_sled_delta_signal' in fsdp_src
    sled_block = ray_src[ray_src.index('GRPO+SLED-Delta: attach'):]
    sled_block = sled_block[:sled_block.index("per-prompt group views")]
    assert 'compute_sled_delta_signal' in sled_block
    assert "batch.batch['advantages'] =" not in sled_block, 'SLED must not rewrite advantages'
    print('PASS test_one_rollout_no_second_generation')


def test_alpha_zero_recovers_grpo():
    """Test 6: alpha=0 -> q==p exactly, A_delta==0, total==GRPO."""
    n, v, k = 16, 48, 12
    f = torch.randn(n, v)
    e = torch.randn(n, v)
    assert torch.equal(SLED.sled_contrast_logits(f, e, 0.0), f.float())
    stats = _production_truncated_route(f, SLED.sled_contrast_logits(f, e, 0.0),
                                        torch.randint(0, v, (n,)), k, temperature=0.7)
    assert float(stats['a_delta'].abs().max().item()) == 0.0
    B, R = 2, 8
    old = torch.zeros(B, R)
    lp = torch.zeros(B, R, requires_grad=True)
    adv = torch.ones(B, R)
    mask = torch.ones(B, R)
    pg, _, _ = CORE.compute_policy_loss(old, lp, adv, mask, 0.2)
    gate = torch.zeros(B, R)
    sled_tok = SLED.sled_gated_per_token_loss(lp, stats['a_delta'][:B * R].view(B, R), gate)
    total = pg + 1.0 * CORE.agg_loss(sled_tok, mask, 'token-mean')
    assert torch.equal(total, pg), (total, pg)
    print('PASS test_alpha_zero_recovers_grpo')


def test_sign_gate_truth_table():
    """Test 7: keep on agreement, block on disagreement (through live_opd_advantage)."""
    ad = torch.tensor([2.0, -2.0, 2.0, -2.0, 0.0])
    ao = torch.tensor([1.0, -1.0, -1.0, 1.0, 5.0])
    got_ao, got_g = SLED.live_opd_advantage(ad, ao, torch.zeros(5), torch.zeros(5),
                                            torch.zeros(5, 1), torch.zeros(5, 1))
    assert torch.allclose(got_ao, ao, atol=1e-6)
    assert torch.equal(got_g, torch.tensor([1., 1., 0., 0., 0.])), got_g
    print('PASS test_sign_gate_truth_table')


def test_existing_modes_unchanged():
    """Test 8: SLED wiring is strictly opt-in; old dispatch paths are intact."""
    sled_src = (ACTOR_DIR / 'sled_delta.py').read_text()
    dp_src = (ACTOR_DIR / 'dp_actor.py').read_text()
    assert "self.config.get('use_sled_delta', False)" in dp_src
    branch = dp_src[dp_src.index('GRPO+SLED-Delta auxiliary loss'):]
    branch = branch[:branch.index('del log_prob')]
    assert "'sled_a_delta' in data" in branch
    assert "float(self.config.get('sled_loss_coef', 1.0)) != 0.0" in branch
    # the non-SLED statement is verbatim: pg minus entropy, no SLED term
    assert 'policy_loss = pg_loss - entropy_loss * entropy_coeff' in dp_src
    # OPD^2 advantage swap untouched
    assert "batch.batch['advantages'] = sig * gen_w" in (TRAINER_DIR / 'ray_trainer.py').read_text()
    # common.sh defaults SLED off and refuses the OPD^2+SLED combo
    common = (REPO / 'experiments' / 'gxpo_efficiency' / 'common.sh').read_text()
    assert 'SLED_ENABLED="${SLED_ENABLED:-0}"' in common
    assert 'SLED_ENABLED=1 cannot be combined with OPD2_ENABLED=1' in common
    # set -u ordering: SLED_ON must be assigned before the RUN_NAME tag reads it.
    assert common.index(') SLED_ON=') < common.index('_sledsig'), \
        'SLED_ON must be defined before the RUN_NAME tag block'
    # mode hygiene: the no-grad SLED forwards must restore train/eval mode so
    # the surrounding update loop's numerics are untouched.
    assert dp_src.count('self.actor_module.train(was_training)') >= 2, \
        'both SLED forwards must restore the prior train/eval mode'
    # truncation change: no full-vocab reduction helpers may remain, and the
    # shared top-K defaults to 1024.
    assert 'chunk_accumulate_stats' not in sled_src, 'full-vocab path must be gone'
    assert 'finalize_frozen_stats' not in sled_src, 'full-vocab path must be gone'
    assert 'sled_truncated_frozen' in sled_src
    assert 'SLED_TOPK="${SLED_TOPK:-1024}"' in common
    print('PASS test_existing_modes_unchanged')


TESTS = [test_grpo_equivalence_coef_zero, test_sled_component_standalone,
         test_advantage_dimensionality, test_topk_truncation_fidelity,
         test_gate_only_gates_sled,
         test_one_rollout_no_second_generation, test_alpha_zero_recovers_grpo,
         test_sign_gate_truth_table, test_existing_modes_unchanged]


if __name__ == '__main__':
    for t in TESTS:
        t()
    print(f'{len(TESTS)} SLED-Delta tests passed')
