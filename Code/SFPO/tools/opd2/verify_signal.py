#!/usr/bin/env python
"""Parity check for the OPD^2 port.

``--unit``  CPU tensor checks (same as ``python -m verl.workers.actor.opd2_signal``).
``--live``  Loads student / teacher / teacher_base on one GPU, scores a couple of
            hand-written sequences through :class:`OPD2Scorer` + the verl-side
            helpers, and asserts the result matches a LITERAL transcription of
            NAVER's ``opd2_trainer.OPD2Trainer._compute_opd_rewards`` loop run on
            the same inputs. If this disagrees, the port is wrong and nothing
            downstream matters.

Usage:
    cd final-gxpo/Code/SFPO
    PYTHONPATH=. python tools/opd2/verify_signal.py --unit
    PYTHONPATH=. python tools/opd2/verify_signal.py --live --gpu 0
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_STUDENT = ('/office/shared_cache/.cache/huggingface/hub/'
                   'models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/'
                   '989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
MODELS = os.path.abspath(os.path.join(REPO_ROOT, '..', '..', 'models'))
DEFAULT_TEACHER = os.path.join(MODELS, 'DeepScaleR-1.5B-Preview')
DEFAULT_TEACHER_BASE = os.path.join(MODELS, 'DeepSeek-R1-Distill-Qwen-1.5B')


# ------------------------------------------------------------------ reference --
def reference_rewards(student, teacher, teacher_base, prompt_ids, teacher_prompt_ids,
                      comp_ids, temperature, top_k, bias, device):
    """Transcription of opd2_trainer._compute_opd_rewards for ONE sequence.

    Deliberately naive: one full-vocab [1, T, V] log_softmax per model, no
    chunking, no micro-batching. That is the point -- it is the thing the fast
    path has to agree with.
    """
    def fwd(model, p_ids, gt_ids):
        full = torch.cat([p_ids, comp_ids], dim=1)
        # bf16, NOT the reference's bare torch.autocast(device_type='cuda') --
        # that defaults to FLOAT16, which on a bf16 checkpoint is a lossy
        # downcast worth ~2.5 logits (measured on DeepScaleR). verl's own
        # forwards pin dtype=torch.bfloat16; so does the OPD^2 scorer.
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=device.type == 'cuda'):
            out = model(input_ids=full, use_cache=False)
        p_len, c_len = p_ids.shape[1], comp_ids.shape[1]
        logits = out.logits[:, p_len - 1:p_len - 1 + c_len, :]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        return F.log_softmax(logits.float() / temperature, dim=-1)

    gt_ids = comp_ids.unsqueeze(-1)
    base_lp = fwd(student, prompt_ids, gt_ids)
    base_gt = base_lp.gather(-1, gt_ids).squeeze(-1)
    topk_idx = base_lp.topk(top_k, dim=-1).indices
    gt_in = (topk_idx == gt_ids).any(dim=-1, keepdim=True)
    last = torch.zeros_like(topk_idx, dtype=torch.bool)
    last[..., -1] = True
    topk_idx = torch.where(last & ~gt_in, gt_ids.expand_as(topk_idx), topk_idx)
    base_topk = base_lp.gather(-1, topk_idx)
    del base_lp

    eval_lp = fwd(teacher, teacher_prompt_ids, gt_ids)
    eval_gt = eval_lp.gather(-1, gt_ids).squeeze(-1)
    eval_topk = eval_lp.gather(-1, topk_idx)
    del eval_lp

    tb_lp = fwd(teacher_base, teacher_prompt_ids, gt_ids)
    tb_gt = tb_lp.gather(-1, gt_ids).squeeze(-1)
    tb_topk = tb_lp.gather(-1, topk_idx)
    del tb_lp

    base_prob = base_topk.exp()
    mean_eval = (base_prob * eval_topk).sum(dim=-1)
    mean_tb = (base_prob * tb_topk).sum(dim=-1)
    mean_base = (base_prob * base_topk).sum(dim=-1)
    signal = (eval_gt - tb_gt) - (mean_eval - mean_tb)
    d_base = (eval_gt - base_gt) - (mean_eval - mean_base)
    signal = torch.where((signal * d_base) < 0, torch.zeros_like(signal), signal)
    signal = signal - bias
    signal = torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal.squeeze(0), topk_idx.squeeze(0)


# ----------------------------------------------------------------------- live --
def run_live(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from verl.workers.actor.opd2_signal import (OPD2Scorer, combine_opd2_signal,
                                                topk_from_logits)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    temperature, top_k, bias = args.temperature, args.topk, args.bias

    print(f'[live] device={device} topk={top_k} temperature={temperature}')
    stok = AutoTokenizer.from_pretrained(args.student)
    student = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16, attn_implementation=args.attn).to(device).eval()
    student.requires_grad_(False)

    scorer = OPD2Scorer(args.teacher, args.teacher_base, attn_implementation=args.attn,
                        chunk_tokens=args.chunk, use_teacher_template=True,
                        keep_on_gpu=True, verbose=True)
    torch.cuda.set_device(device)  # OPD2Scorer loads disk->current device
    scorer.to_gpu()

    chats = [
        [{'role': 'user', 'content': 'What is 17 * 23? Show your reasoning.'}],
        [{'role': 'user', 'content': 'If x + 5 = 12, what is 3x?'}],
    ]

    max_abs = 0.0
    for n, chat in enumerate(chats):
        prompt_text = stok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompt_ids = stok(prompt_text, return_tensors='pt',
                          add_special_tokens=False)['input_ids'].to(device)
        with torch.no_grad():
            gen = student.generate(prompt_ids, max_new_tokens=args.tokens,
                                   do_sample=True, temperature=temperature,
                                   top_p=1.0, pad_token_id=stok.pad_token_id or stok.eos_token_id)
        comp_ids = gen[:, prompt_ids.shape[1]:]
        r_len = comp_ids.shape[1]
        t_prompt_ids = scorer.render_teacher_prompt_ids(chat, prompt_ids[0].tolist())
        t_prompt = torch.tensor([t_prompt_ids], dtype=torch.long, device=device)

        # --- reference (naive, one model at a time, full-vocab)
        with torch.no_grad():
            ref_sig, ref_idx = reference_rewards(
                student, scorer.teacher, scorer.teacher_base, prompt_ids, t_prompt,
                comp_ids, temperature, top_k, bias, device)

        # --- port (chunked student top-K + OPD2Scorer + combine_opd2_signal)
        with torch.no_grad():
            full = torch.cat([prompt_ids, comp_ids], dim=1)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == 'cuda'):
                logits = student(input_ids=full, use_cache=False).logits
            p_len = prompt_ids.shape[1]
            logits = logits[:, p_len - 1:p_len - 1 + r_len, :]
            b_gt, b_tk, idx = topk_from_logits(logits.reshape(r_len, -1),
                                               comp_ids.reshape(r_len), top_k,
                                               temperature=temperature,
                                               chunk_tokens=args.chunk)
            del logits
            seq = list(t_prompt_ids) + comp_ids[0].tolist()
            (e_gt, e_tk, t_gt, t_tk), = scorer.score_rows([seq], [r_len],
                                                          [comp_ids[0]], [idx],
                                                          temperature=temperature)
            got = combine_opd2_signal(b_gt, b_tk, e_gt, e_tk, t_gt, t_tk, bias)

        # The reference divides by temperature in fp32, the real pipeline does it
        # in bf16 (dp_actor._forward_kd_micro_batch calls logits.div_ under
        # autocast). Different rounding can reshuffle the columns right at the
        # K-th boundary, where the student's own probability -- the E_base weight
        # -- is ~0. So compare the OVERLAP, not the exact index tensor; the signal
        # equality below is the assertion that actually matters.
        overlap = 0.0
        for t in range(r_len):
            a, b = set(idx[t].tolist()), set(ref_idx[t].tolist())
            overlap += len(a & b) / max(len(a), 1)
        overlap /= max(r_len, 1)
        assert overlap > 0.99, f'top-K column selection diverged: {overlap:.4f} overlap'
        d = (got - ref_sig).abs().max().item()
        max_abs = max(max_abs, d)
        gated = float((got == 0).float().mean())
        print(f'[live] seq {n}: {r_len} tokens | max|port - reference| = {d:.3e} '
              f'| topk overlap {overlap:.4f} '
              f'| signal mean {got.mean():.4f} std {got.std():.4f} | gated {gated:.2%}')
        assert d < args.tol, f'signal mismatch {d} >= tol {args.tol}'

    print(f'[live] OPD^2 parity OK (max abs diff {max_abs:.3e}, tol {args.tol})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', action='store_true')
    ap.add_argument('--live', action='store_true')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--student', default=DEFAULT_STUDENT)
    ap.add_argument('--teacher', default=DEFAULT_TEACHER)
    ap.add_argument('--teacher-base', dest='teacher_base', default=DEFAULT_TEACHER_BASE)
    ap.add_argument('--topk', type=int, default=1024)
    ap.add_argument('--tokens', type=int, default=96)
    ap.add_argument('--chunk', type=int, default=32)
    ap.add_argument('--attn', default='flash_attention_2')
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--bias', type=float, default=0.0)
    # bf16 forwards through two different code paths; 1e-2 is loose enough for
    # kernel-order noise and tight enough to catch a real formula/alignment bug
    # (the signal itself is O(1)).
    ap.add_argument('--tol', type=float, default=1e-2)
    args = ap.parse_args()
    if not args.unit and not args.live:
        args.unit = True

    if args.unit:
        from verl.workers.actor.opd2_signal import _self_check
        _self_check()
    if args.live:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
        args.gpu = 0
        run_live(args)


if __name__ == '__main__':
    main()
