#!/usr/bin/env python
"""Frozen-batch replay: does GXPO's extrapolation predict what Muon actually does?

GXPO repositions to ``theta0 + alpha * scale * (theta2 - theta0)``, where
``scale`` is a K-step geometric extrapolation. Two estimators produce that
scale:

  grad-space   coordinatewise ``S_K(r)/S_2(r)`` with ``r = g1/g0`` -- the
               shipped behavior, exact for SGD.
  update-space per-matrix scalar ``S_K(rho)/S_2(rho)`` with
               ``rho = <u0,u1>/<u0,u0>`` read off the two real optimizer steps.

This script measures both against ground truth: it runs the two probe steps,
forms each prediction, then continues from the same starting point for K real
Muon steps and compares against where Muon actually landed.

It is deliberately single-process. The checkpoint is a 2-rank FSDP save, so the
weights are reassembled from both shard files and the model is rebuilt
unsharded on one GPU; Muon then runs its dense path on true full matrices,
which is what the retention question is about.

The objective is next-token cross-entropy on real prompts from the training
mix, not the PPO surrogate: reconstructing rollouts and advantages would add a
great deal of machinery without changing what is being tested. The claim under
test is a property of the optimizer's geometry given a gradient sequence, not
of the loss that produced it. Gradients are nonetheless taken at the real
step-60 weights, at real shapes, across all Muon-owned matrices.

Usage:
    python tools/gxpo_replay_muon.py \
        --checkpoint results/gxpo_efficiency/<run>/global_step_60/actor \
        --data <path>/lighteval-math/train.parquet
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl.workers.actor.gxpo_state import (compute_gxpo_retention_scale,
                                           compute_gxpo_update_retention_scale)
from verl.workers.muon import Muon


def load_full_state_dict(checkpoint_dir, world_size=2):
    """Reassemble an unsharded state dict from the per-rank FSDP DTensor saves."""
    shards = []
    for rank in range(world_size):
        path = os.path.join(checkpoint_dir, f'model_world_size_{world_size}_rank_{rank}.pt')
        shards.append(torch.load(path, map_location='cpu', weights_only=False))

    full = {}
    for key, reference in shards[0].items():
        locals_ = [shard[key].to_local() for shard in shards]
        merged = torch.cat(locals_, dim=0)
        logical = tuple(reference.shape)
        if tuple(merged.shape) != logical:
            # FSDP pads the last shard to a uniform size; drop the padding.
            if merged.shape[0] < logical[0]:
                raise RuntimeError(f'{key}: shards too small, {merged.shape} < {logical}')
            merged = merged[:logical[0]]
        full[key] = merged.contiguous()
    return full


def build_model(checkpoint_dir, base_model, device, dtype=torch.float32):
    """Load the base model, then overwrite its parameters with the checkpoint's.

    Deliberately *not* meta-device + to_empty(): that leaves non-persistent
    buffers -- notably ``rotary_emb.inv_freq``, which no checkpoint stores --
    as uninitialized memory, which silently destroys attention. (Observed:
    LM loss 10.7 vs 4.3 for the same weights.) Starting from a real
    from_pretrained model keeps every buffer correct, and load_state_dict then
    replaces exactly the trained tensors.
    """
    from transformers import AutoModelForCausalLM
    hf_dir = os.path.join(checkpoint_dir, 'huggingface')
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype,
                                                 attn_implementation='sdpa')
    model.to(device)
    state = load_full_state_dict(checkpoint_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Qwen ties the LM head to the embedding; a tied head is legitimately absent
    # from the checkpoint and is re-tied below.
    missing = [k for k in missing if 'lm_head' not in k]
    if missing or unexpected:
        raise RuntimeError(f'state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}')
    model.tie_weights()
    model.train()
    model.config.use_cache = False
    return model, hf_dir


def build_optimizer(model, lr, momentum, ns_steps, weight_decay):
    """Partition parameters exactly as the training run's build_muon does."""
    from verl.workers.muon import _is_embedding_or_head
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and p.is_contiguous() and not _is_embedding_or_head(name):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    opt = Muon(lr=lr, wd=weight_decay, muon_params=muon_params,
               adamw_params=adamw_params, momentum=momentum,
               nesterov=True, ns_steps=ns_steps)
    return opt, muon_params


def make_batch(data_path, hf_dir, batch_size, max_len, device, seed):
    import pandas as pd
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    frame = pd.read_parquet(data_path)
    frame = frame.sample(n=batch_size, random_state=seed)
    texts = []
    for _, row in frame.iterrows():
        messages = list(row['prompt'])
        texts.append(tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True))
    enc = tok(texts, return_tensors='pt', padding=True, truncation=True,
              max_length=max_len)
    return {k: v.to(device) for k, v in enc.items()}


def compute_gradients(model, batch):
    """One backward on the frozen batch; returns the grad-norm for reference."""
    model.zero_grad(set_to_none=True)
    out = model(**batch, labels=batch['input_ids'])
    out.loss.backward()
    total = torch.zeros((), device=out.loss.device)
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.float().square().sum()
    return float(out.loss.item()), float(total.sqrt().item())


def snapshot(params):
    return [p.detach().clone() for p in params]


def restore(params, saved):
    with torch.no_grad():
        for p, s in zip(params, saved):
            p.data.copy_(s)


def clone_optimizer_state(opt):
    out = {}
    for p, state in opt.state.items():
        out[p] = {k: (v.detach().clone() if torch.is_tensor(v) else v)
                  for k, v in state.items()}
    return out


def restore_optimizer_state(opt, saved):
    for p, state in saved.items():
        target = opt.state[p]
        for k, v in state.items():
            if torch.is_tensor(v) and torch.is_tensor(target.get(k)):
                target[k].copy_(v)
            else:
                target[k] = (v.detach().clone() if torch.is_tensor(v) else v)


def report_sweep(params, theta0, theta1, theta2, trajectory, g0, g1, k_values, args):
    """Per-horizon accuracy, and what the reposition actually buys per backward pass.

    GXPO spends 3 backward passes per outer step (two probes plus the corrective
    pass) and moves alpha * scale * (theta2 - theta0). Plain training spends 3
    backward passes and moves 3 steps. Because Muon's steps have constant
    magnitude, |n-step displacement| ~ n * |u|, so |theta2 - theta0| ~ 2|u| and

        speedup ~ alpha * scale * 2 / 3

    which is below 1 whenever alpha * scale < 1.5. That is the number that
    decides whether extrapolation is worth its cost.
    """
    print()
    print(f'{"K":>4}{"true_scale":>12}{"upd_scale":>11}{"upd_err":>9}'
          f'{"grad_scale":>12}{"grad_err":>9}{"a*s@0.3":>9}{"speedup":>9}'
          f'{"a for 2x":>10}')
    results = []
    for K in k_values:
        thetaK = trajectory[K]
        rows = []
        for i in range(len(params)):
            disp2 = (theta2[i] - theta0[i]).float()
            true_disp = (thetaK[i] - theta0[i]).float()
            if true_disp.norm() == 0 or disp2.norm() == 0:
                continue
            _, grad_scale, _, _ = compute_gxpo_retention_scale(
                g0[i].float(), g1[i].float(), K, args.delta)
            _, upd_scale = compute_gxpo_update_retention_scale(
                (theta1[i] - theta0[i]).float(), (theta2[i] - theta1[i]).float(),
                K, args.delta)
            rows.append((
                (true_disp.norm() / disp2.norm()).item(),
                upd_scale.item(),
                ((disp2 * upd_scale - true_disp).norm() / true_disp.norm()).item(),
                grad_scale.mean().item(),
                ((disp2 * grad_scale - true_disp).norm() / true_disp.norm()).item(),
            ))

        def med(j):
            v = sorted(r[j] for r in rows)
            return v[len(v) // 2]

        true_s, upd_s, upd_e, grad_s, grad_e = (med(j) for j in range(5))
        eff = args.alpha * upd_s
        speedup = 2.0 * eff / 3.0
        alpha_2x = 3.0 / upd_s          # alpha giving speedup 2.0
        print(f'{K:>4}{true_s:>12.3f}{upd_s:>11.3f}{upd_e:>9.3f}'
              f'{grad_s:>12.3f}{grad_e:>9.3f}{eff:>9.3f}{speedup:>9.3f}'
              f'{alpha_2x:>10.3f}')
        results.append({'K': K, 'true_scale': true_s, 'update_scale': upd_s,
                        'update_err': upd_e, 'grad_scale': grad_s,
                        'grad_err': grad_e, 'effective_multiplier': eff,
                        'speedup_vs_plain': speedup, 'alpha_for_2x': alpha_2x})
    print()
    print('speedup = alpha*scale*2/3, the progress GXPO buys per backward pass')
    print(f'relative to plain training, at alpha={args.alpha}. Below 1.0 means the')
    print('three-pass step makes less progress than three ordinary steps.')
    if args.json_out:
        with open(args.json_out, 'w') as fh:
            json.dump({'sweep': results, 'args': vars(args)}, fh, indent=2)
        print(f'wrote {args.json_out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--base-model', required=True,
                    help='base HF model the run started from; supplies the '
                         'non-persistent buffers the checkpoint does not store')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--k-sweep', default=None,
                    help='comma-separated K values to evaluate from one shared '
                         'ground-truth trajectory, e.g. 2,4,6,8,10,15,20')
    ap.add_argument('--alpha', type=float, default=0.3)
    ap.add_argument('--delta', type=float, default=1e-8)
    ap.add_argument('--lr', type=float, default=1e-6)
    ap.add_argument('--momentum', type=float, default=0.95)
    ap.add_argument('--ns-steps', type=int, default=5)
    ap.add_argument('--weight-decay', type=float, default=1e-2)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-len', type=int, default=512)
    ap.add_argument('--warmup', type=int, default=5,
                    help='steps to seed Muon momentum before measuring; the '
                         'checkpoint stores it FSDP-sharded and this replay '
                         'runs unsharded, so it is rebuilt rather than loaded')
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--fresh-batch', action='store_true',
                    help='draw a NEW minibatch for each ground-truth step. This '
                         'is the realistic setting: GXPO reuses one minibatch '
                         'across its three passes, but the K ordinary steps it '
                         'replaces would each see different data. A frozen '
                         'ground truth overstates step-to-step alignment and '
                         'therefore overstates what extrapolation can predict.')
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f'loading checkpoint {args.checkpoint}', flush=True)
    model, hf_dir = build_model(args.checkpoint, args.base_model, device)
    opt, muon_params = build_optimizer(model, args.lr, args.momentum,
                                       args.ns_steps, args.weight_decay)
    print(f'  Muon-owned matrices: {len(muon_params)}', flush=True)

    batch = make_batch(args.data, hf_dir, args.batch_size, args.max_len,
                       device, args.seed)
    print(f'  frozen batch: {tuple(batch["input_ids"].shape)}', flush=True)

    for i in range(args.warmup):
        loss, gn = compute_gradients(model, batch)
        opt.step()
        print(f'  warmup {i + 1}/{args.warmup}: loss={loss:.4f} |g|={gn:.4e}', flush=True)

    # ---- theta0 -------------------------------------------------------------
    theta0 = snapshot(muon_params)
    opt_state0 = clone_optimizer_state(opt)

    # ---- probe step 1 -> theta1, capture g0 ---------------------------------
    loss0, gn0 = compute_gradients(model, batch)
    g0 = [p.grad.detach().clone() for p in muon_params]
    opt.step()
    theta1 = snapshot(muon_params)

    # ---- probe step 2 -> theta2, capture g1 ---------------------------------
    loss1, gn1 = compute_gradients(model, batch)
    g1 = [p.grad.detach().clone() for p in muon_params]
    opt.step()
    theta2 = snapshot(muon_params)

    # ---- ground truth: real Muon steps from theta0, recorded at each horizon -
    k_values = ([int(x) for x in args.k_sweep.split(',')] if args.k_sweep
                else [args.K])
    k_max = max(k_values)
    restore(muon_params, theta0)
    restore_optimizer_state(opt, opt_state0)
    trajectory = {}
    for i in range(k_max):
        step_batch = batch
        if args.fresh_batch:
            step_batch = make_batch(args.data, hf_dir, args.batch_size,
                                    args.max_len, device, args.seed + 1000 + i)
        compute_gradients(model, step_batch)
        opt.step()
        if (i + 1) in k_values:
            trajectory[i + 1] = snapshot(muon_params)
        print(f'  ground truth step {i + 1}/{k_max}', flush=True)

    if args.k_sweep:
        report_sweep(muon_params, theta0, theta1, theta2, trajectory,
                     g0, g1, k_values, args)
        return

    thetaK = trajectory[args.K]

    # ---- compare ------------------------------------------------------------
    rows = []
    for i, p in enumerate(muon_params):
        disp2 = (theta2[i] - theta0[i]).float()
        true_disp = (thetaK[i] - theta0[i]).float()
        if true_disp.norm() == 0:
            continue

        _, grad_scale, _, _ = compute_gxpo_retention_scale(
            g0[i].float(), g1[i].float(), args.K, args.delta)
        rho, upd_scale = compute_gxpo_update_retention_scale(
            (theta1[i] - theta0[i]).float(), (theta2[i] - theta1[i]).float(),
            args.K, args.delta)

        def stats(pred):
            err = ((pred - true_disp).norm() / true_disp.norm()).item()
            cos = (torch.dot(pred.flatten(), true_disp.flatten())
                   / (pred.norm() * true_disp.norm() + 1e-12)).item()
            return err, cos

        grad_err, grad_cos = stats(disp2 * grad_scale)
        upd_err, upd_cos = stats(disp2 * upd_scale)

        # What scale would have been exactly right, and what the two candidate
        # models say about it.
        u0 = (theta1[i] - theta0[i]).float()
        u1 = (theta2[i] - theta1[i]).float()
        true_scale = (true_disp.norm() / disp2.norm()).item()
        cos_norm = (torch.dot(u0.flatten(), u1.flatten())
                    / (u0.norm() * u1.norm() + 1e-12)).clamp(-1.0, 1.0)
        # Constant-magnitude rotating walk: |u_t| fixed, direction turning by a
        # constant angle phi each step. Then |sum_{t<K} u_t| / |u_0 + u_1|
        # = sin(K*phi/2) / sin(phi), which tends to K/2 as phi -> 0.
        phi = torch.arccos(cos_norm)
        rot_scale = (torch.sin(args.K * phi / 2) / torch.sin(phi)).abs().item() \
            if phi.item() > 1e-6 else args.K / 2.0
        rot_err, rot_cos = stats(disp2 * rot_scale)
        norm_ratio = (u1.norm() / (u0.norm() + 1e-12)).item()

        rows.append({
            'index': i,
            'shape': tuple(p.shape),
            'grad_err': grad_err, 'grad_cos': grad_cos,
            'update_err': upd_err, 'update_cos': upd_cos,
            'rot_err': rot_err, 'rot_cos': rot_cos,
            'true_scale': true_scale,
            'rot_scale': rot_scale,
            'cos_norm': cos_norm.item(),
            'step_norm_ratio': norm_ratio,
            'rho': rho.item(),
            'update_scale': upd_scale.item(),
            'grad_scale_mean': grad_scale.mean().item(),
        })

    def med(key):
        vals = sorted(r[key] for r in rows)
        return vals[len(vals) // 2] if vals else float('nan')

    better = sum(1 for r in rows if r['update_err'] < r['grad_err'])
    print()
    print(f'matrices compared: {len(rows)}   K={args.K}')
    print(f'{"":<22}{"median":>12}{"mean":>12}')
    for key in ('grad_err', 'update_err', 'rot_err', 'grad_cos', 'update_cos',
                'rot_cos', 'true_scale', 'rot_scale', 'cos_norm', 'step_norm_ratio'):
        mean = sum(r[key] for r in rows) / max(len(rows), 1)
        print(f'{key:<22}{med(key):>12.4f}{mean:>12.4f}')
    print(f'{"rho":<22}{med("rho"):>12.4f}')
    print(f'{"update_scale":<22}{med("update_scale"):>12.4f}')
    print(f'{"grad_scale_mean":<22}{med("grad_scale_mean"):>12.4f}')
    print()
    print(f'update-space closer to truth on {better}/{len(rows)} matrices '
          f'({100.0 * better / max(len(rows), 1):.1f}%)')

    if args.json_out:
        with open(args.json_out, 'w') as fh:
            json.dump({'rows': rows, 'args': vars(args)}, fh, indent=2)
        print(f'wrote {args.json_out}')


if __name__ == '__main__':
    main()
