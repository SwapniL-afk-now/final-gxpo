"""CPU regression tests for the multi-layer GRPO policy loss (selected_policy_layers).

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest tests/test_multilayer_policy_loss.py -q
"""
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import Qwen2Config, Qwen2ForCausalLM

from verl import DataProto
from verl.trainer.ppo import core_algos
import verl.utils.torch_functional as verl_F
from verl.workers.actor.dp_actor import DataParallelPPOActor

# flash-attn's triton cross-entropy is CUDA-only; use verl's own fallback on CPU.
verl_F.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False

N_LAYERS, VOCAB, P, R, B = 4, 64, 4, 5, 3


def make_actor(layers, ckpt=False, **extra):
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=N_LAYERS,
                      num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
                      max_position_embeddings=64, attn_implementation='eager')
    model = Qwen2ForCausalLM(cfg).float()
    if ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    actor_cfg = OmegaConf.create({'ulysses_sequence_parallel_size': 1, 'use_remove_padding': False,
                                  'use_torch_compile': False, 'selected_policy_layers': layers,
                                  'use_kl_loss': False, 'ppo_mini_batch_size': B, **extra})
    return DataParallelPPOActor(actor_cfg, model)


def make_batch():
    g = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, VOCAB, (B, P + R), generator=g)
    attention_mask = torch.ones(B, P + R, dtype=torch.long)
    attention_mask[0, :2] = 0  # left-padded prompt
    attention_mask[1, -2:] = 0  # right-padded response
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids,
             'responses': input_ids[:, P:]}
    adv = torch.randn(B, 1, generator=g).expand(B, R).contiguous()
    noise = 0.3 * torch.randn(B, R, N_LAYERS, generator=g)
    return batch, adv, noise


def mask_of(batch):
    return batch['attention_mask'][:, -R:]


def logit_lens(model, batch, layer):
    """Reference: HF hidden_states[layer] (1-based) -> final norm -> lm_head."""
    out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                position_ids=batch['position_ids'], output_hidden_states=True, use_cache=False)
    h = out.hidden_states[layer]
    if layer != N_LAYERS:  # HF already applies the final norm to the last entry
        h = model.model.norm(h)
    logits = model.lm_head(h)[:, -R - 1:-1]
    return F.log_softmax(logits.float(), -1).gather(-1, batch['responses'][..., None]).squeeze(-1)


def multilayer_loss(actor, batch, adv, noise, mode='token-mean'):
    _, lp, layer_lp = actor._forward_micro_batch(batch, temperature=1.0, need_entropy=False, with_layers=True)
    cols = [l - 1 for l in actor.selected_policy_layers]
    old = (layer_lp.detach() + noise[..., cols])
    return actor._multilayer_policy_loss(layer_lp, old, adv, mask_of(batch), 0.2, mode), layer_lp, old


def no_hooks(model):
    return all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def grads(model):
    return {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for n, p in model.named_parameters()}


def test_null_is_original_path():
    actor = make_actor(None)
    batch, adv, noise = make_batch()
    assert actor.selected_policy_layers is None
    out = actor._forward_micro_batch(batch, temperature=1.0)
    assert len(out) == 2 and no_hooks(actor.actor_module)
    assert torch.allclose(out[1], logit_lens(actor.actor_module, batch, N_LAYERS), atol=1e-5)
    data = DataProto.from_dict(tensors={**batch, 'old_log_probs': out[1].detach(), 'advantages': adv,
                                        'old_log_probs_layers': out[1].detach()[..., None]})
    assert 'old_log_probs_layers' not in actor._make_minibatch_iterator(data)[2]
    assert 'old_log_probs_layers' in make_actor([N_LAYERS])._make_minibatch_iterator(data)[2]


def test_last_layer_equals_null_exactly():
    batch, adv, noise = make_batch()
    null, last = make_actor(None), make_actor([N_LAYERS])
    _, lp = null._forward_micro_batch(batch, temperature=1.0, need_entropy=False)
    old = lp.detach() + noise[..., -1]
    null_loss, _, _ = core_algos.compute_policy_loss(old, lp, adv, mask_of(batch), 0.2)
    null_loss.backward()
    _, lp2, layer_lp = last._forward_micro_batch(batch, temperature=1.0, need_entropy=False, with_layers=True)
    ml_loss, _, _, _ = last._multilayer_policy_loss(layer_lp, old[..., None], adv, mask_of(batch), 0.2,
                                                    'token-mean')
    ml_loss.backward()
    assert torch.equal(null_loss, ml_loss)
    g0, g1 = grads(null.actor_module), grads(last.actor_module)
    assert all(torch.equal(g0[n], g1[n]) for n in g0)


@pytest.mark.parametrize('layer', [1, 2, 3, 4])
def test_single_layer_matches_logit_lens(layer):
    actor = make_actor([layer])
    batch, adv, noise = make_batch()
    (loss, _, _, metrics), layer_lp, old = multilayer_loss(actor, batch, adv, noise)
    assert torch.allclose(layer_lp[..., 0], logit_lens(actor.actor_module, batch, layer), atol=1e-5)
    want, _, _ = core_algos.compute_policy_loss(old[..., 0], layer_lp[..., 0], adv, mask_of(batch), 0.2)
    assert torch.equal(loss, want)
    assert set(metrics) == {f'policy_loss/layer_{layer}', f'policy_ratio/layer_{layer}',
                            f'clip_fraction/layer_{layer}', f'approx_kl/layer_{layer}',
                            'policy_loss/multilayer'}
    assert no_hooks(actor.actor_module)


@pytest.mark.parametrize('mode', core_algos.LOSS_AGG_MODES)
def test_multi_layer_is_mean_of_layer_losses(mode):
    actor = make_actor([3, 1, 2])  # unsorted input -> deterministic sorted order
    assert actor.selected_policy_layers == (1, 2, 3)
    batch, adv, noise = make_batch()
    (loss, clip, kl, metrics), layer_lp, old = multilayer_loss(actor, batch, adv, noise, mode)
    per = [core_algos.compute_policy_loss(old[..., k], layer_lp[..., k], adv, mask_of(batch), 0.2, mode)[0]
           for k in range(3)]
    assert torch.allclose(loss, (per[0] + per[1] + per[2]) / 3)
    for k, l in enumerate((1, 2, 3)):
        assert torch.equal(metrics[f'policy_loss/layer_{l}'], per[k].detach())
    assert torch.equal(metrics['policy_loss/multilayer'], loss.detach())


@pytest.mark.parametrize('layer', [1, 2, 3])
def test_layer_loss_does_not_reach_later_blocks(layer):
    actor = make_actor([layer])
    batch, adv, noise = make_batch()
    (loss, _, _, _), _, _ = multilayer_loss(actor, batch, adv, noise)
    loss.backward()
    for i, block in enumerate(actor.actor_module.model.layers):
        norms = [p.grad.abs().sum().item() if p.grad is not None else 0.0 for p in block.parameters()]
        if i >= layer:  # 0-based block i is 1-based layer i+1 > layer
            assert sum(norms) == 0.0, (layer, i)
        else:
            assert sum(norms) > 0.0, (layer, i)


def test_earlier_blocks_accumulate_all_layer_gradients():
    batch, adv, noise = make_batch()
    got = {}
    for layers in ([1], [3], [1, 3]):
        actor = make_actor(layers)
        (loss, _, _, _), _, _ = multilayer_loss(actor, batch, adv, noise)
        loss.backward()
        got[tuple(layers)] = grads(actor.actor_module)
    for n in got[(1, 3)]:
        assert torch.allclose(got[(1, 3)][n], (got[(1,)][n] + got[(3,)][n]) / 2, atol=1e-6), n


def test_old_detached_current_differentiable():
    actor = make_actor([2, 4])
    batch, adv, noise = make_batch()
    data = DataProto.from_dict(tensors=batch, meta_info={'micro_batch_size': 2, 'temperature': 1.0,
                                                         'use_dynamic_bsz': False})
    old_lp, _, old_layers = actor.compute_log_prob(data, with_layers=True)
    assert old_layers.shape == (B, R, 2) and not old_layers.requires_grad and not old_lp.requires_grad
    assert torch.equal(old_layers[..., 1], old_lp)  # l = N reuses the final log-probs
    _, _, layer_lp = actor._forward_micro_batch(batch, temperature=1.0, with_layers=True)
    assert layer_lp.requires_grad and layer_lp.grad_fn is not None
    assert torch.allclose(layer_lp.detach(), old_layers, atol=1e-6)
    assert len(actor.compute_log_prob(data)) == 2  # default call signature unchanged


@pytest.mark.parametrize('mode', core_algos.LOSS_AGG_MODES)
def test_masked_positions_do_not_matter(mode):
    actor = make_actor([1, 2])
    batch, adv, _ = make_batch()
    mask = mask_of(batch)
    g = torch.Generator().manual_seed(2)
    lp = torch.randn(B, R, 2, generator=g, requires_grad=True)
    old = torch.randn(B, R, 2, generator=g)
    loss, _, _, metrics = actor._multilayer_policy_loss(lp, old, adv, mask, 0.2, mode)
    loss.backward()
    assert lp.grad[~mask.bool()].abs().sum() == 0
    junk = (~mask.bool())[..., None] * 1.5  # finite: exp(3) cannot overflow to inf*0
    loss2, _, _, metrics2 = actor._multilayer_policy_loss(lp.detach() + junk, old - junk, adv, mask, 0.2, mode)
    assert torch.equal(loss.detach(), loss2)
    assert all(torch.equal(metrics[k], metrics2[k]) for k in metrics)


def test_gradient_checkpointing_parity_and_no_hook_leak():
    batch, adv, noise = make_batch()
    got = []
    for ckpt in (False, True):
        actor = make_actor([1, 3], ckpt=ckpt)
        (loss, _, _, _), _, _ = multilayer_loss(actor, batch, adv, noise)
        loss.backward()
        assert no_hooks(actor.actor_module)
        got.append(grads(actor.actor_module))
    assert all(torch.allclose(got[0][n], got[1][n], atol=1e-6) for n in got[0])


@pytest.mark.parametrize('bad', [[], [0], [N_LAYERS + 1], [2, 2], [True], '12', 3, [1.0]])
def test_validation_rejects(bad):
    with pytest.raises(ValueError):
        make_actor(bad)


def test_rejects_other_loss_paths():
    with pytest.raises(ValueError, match='use_kd'):
        make_actor([2], use_kd=True)


def test_token_chunking_matches_single_chunk(monkeypatch):
    import verl.workers.actor.dp_actor as dp_actor
    batch, adv, noise = make_batch()
    got = []
    for chunk in (4096, 4):  # 4 tokens forces many checkpointed chunks per layer
        monkeypatch.setattr(dp_actor, 'LAYER_LOG_PROB_CHUNK_TOKENS', chunk)
        actor = make_actor([1, 3, N_LAYERS])
        (loss, _, _, _), layer_lp, _ = multilayer_loss(actor, batch, adv, noise)
        loss.backward()
        got.append((layer_lp.detach(), grads(actor.actor_module)))
    assert torch.allclose(got[0][0], got[1][0], atol=1e-6)
    assert all(torch.allclose(got[0][1][n], got[1][1][n], atol=1e-6) for n in got[0][1])


def test_aux_coef_weights_final_plus_intermediate_mean():
    batch, adv, noise = make_batch()
    plain = make_actor([1, 2, N_LAYERS])
    (_, _, _, m), _, _ = multilayer_loss(plain, batch, adv, noise)
    weighted = make_actor([1, 2, N_LAYERS], multilayer_aux_coef=0.1)
    (loss, _, _, mw), _, _ = multilayer_loss(weighted, batch, adv, noise)
    aux = (m['policy_loss/layer_1'] + m['policy_loss/layer_2']) / 2
    assert torch.allclose(loss.detach(), m[f'policy_loss/layer_{N_LAYERS}'] + 0.1 * aux, atol=1e-6)
    assert torch.allclose(mw['policy_loss/aux_mean'], aux, atol=1e-6)
    assert 'policy_loss/aux_mean' not in m
    with pytest.raises(ValueError, match='final layer'):
        make_actor([1, 2], multilayer_aux_coef=0.1)


def test_kl_mode_k3_loss_matches_reference_and_stops_grad_on_final():
    actor = make_actor([1, 2], multilayer_loss='kl', multilayer_aux_coef=0.5)
    batch, _, _ = make_batch()
    mask = mask_of(batch)
    _, log_prob, layer_lp = actor._forward_micro_batch(batch, temperature=1.0, need_entropy=False,
                                                        with_layers=True)
    for k, layer in enumerate(actor.selected_policy_layers):  # columns are sampled-token lens log-probs
        assert torch.allclose(layer_lp[..., k], logit_lens(actor.actor_module, batch, layer), atol=1e-5)
    loss, metrics = actor._layer_distill_loss(torch.zeros(()), log_prob, layer_lp, mask)
    # k3 = r - log r - 1, r = pi_l / pi_final (verl low_var_kl, clamped to [-10, 10])
    k3 = lambda l: (torch.exp(l - log_prob.detach()) - (l - log_prob.detach()) - 1).clamp(-10, 10)
    expected = 0.5 * (verl_F.masked_mean(k3(layer_lp[..., 0]), mask) + verl_F.masked_mean(k3(layer_lp[..., 1]), mask)) / 2
    assert torch.allclose(loss, expected, atol=1e-6)
    assert set(metrics) >= {'distill_kl/layer_1', 'distill_kl/layer_2', 'policy_loss/aux_mean'}
    loss.backward()
    g = grads(actor.actor_module)
    # The final-policy target is detached: blocks above the highest distilled layer get nothing.
    for n, t in g.items():
        if n.startswith(('model.layers.2.', 'model.layers.3.')):
            assert t.abs().sum() == 0, n
    assert g['model.layers.0.self_attn.q_proj.weight'].abs().sum() > 0


def test_kl_mode_validation():
    with pytest.raises(ValueError, match='intermediate layers only'):
        make_actor([1, N_LAYERS], multilayer_loss='kl', multilayer_aux_coef=0.1)
    with pytest.raises(ValueError, match='needs multilayer_aux_coef'):
        make_actor([1, 2], multilayer_loss='kl')
    with pytest.raises(ValueError, match="'grpo' or 'kl'"):
        make_actor([1, 2], multilayer_loss='bogus')
