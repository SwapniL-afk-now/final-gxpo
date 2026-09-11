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
"""OPD^2 (On-Policy Delta Distillation, arXiv:2607.15161) per-token signal.

Port of NAVER's ``opd2_trainer.OPD2Trainer._compute_opd_rewards`` onto verl.
The signal is a DENSE PER-TOKEN ADVANTAGE -- it replaces the verifier reward,
it is not an auxiliary loss. Everything below the advantage (PPO clip, GXPO's
3-pass update, the retention estimator) is untouched by design.

Per response token, with ``log_softmax`` over the FULL vocabulary at the rollout
temperature and ``E_base[X] = sum_v p_student(v) * X_v`` truncated to the
STUDENT's top-K columns::

    signal = (eval_gt - tb_gt)   - (E_base[eval] - E_base[tb])
    d_base = (eval_gt - base_gt) - (E_base[eval] - E_base[base])
    signal = 0 where signal * d_base < 0            # direction gate
    signal = signal - rewards_bias

``base``/``eval``/``tb`` are student / teacher / teacher_base.

One deliberate divergence from the reference: it runs ``log_softmax`` (and the
``E_base[.]`` reduction) in bf16 to keep the ``[1, T, V]`` peak small. Here the
log_softmax is fp32 *within a token chunk*, so the peak stays bounded without
paying bf16's precision -- whose spacing at a logit of ~30 is ~0.25, the same
order as the signal itself (sigma ~ 0.1). Everything downstream is fp32.

Why the top-K truncation is near-lossless: the weight is the student's own
probability, which is ~0 outside the student's own top-K, so the dropped terms
contribute ~0 to every expectation.

Run ``python -m verl.workers.actor.opd2_signal`` for the CPU self-check.
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__file__)

# Token chunk for the full-vocab log_softmax. The transient peak is
# chunk * vocab * 4B (fp32) ~= 300MB at chunk=512, V=152k -- the same chunking
# discipline as teacher_scoring_worker.py and kd_loss.py.
OPD2_CHUNK_TOKENS = 512


def topk_from_logits(
    logits: torch.Tensor,
    gt_ids: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    chunk_tokens: int = OPD2_CHUNK_TOKENS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Student side: full-vocab log_softmax -> (gt log-prob, top-K log-probs, top-K ids).

    Args:
        logits: ``[N, V]`` flat token logits, RAW (not yet temperature-scaled).
        gt_ids: ``[N]`` int64 sampled token ids.
        k: number of columns to keep (clamped to V).
        temperature: divided in FP32 inside the chunk loop. verl's own
            ``_forward_micro_batch`` divides in bf16, whose spacing at a logit of
            ~30 is ~0.25 -- the same order as the OPD^2 signal itself. The
            reference (``opd2_trainer._forward_and_get_logp_topk``) divides in
            the log_softmax, so do it here too.

    Returns ``(gt_lp [N], topk_lp [N, K], topk_idx [N, K])``, all full-V-normalized.

    The sampled token is force-included: on the rare top-K miss the smallest
    (last) column is overwritten with ``gt``, matching
    ``opd2_trainer._forward_and_get_logp_topk``. Without it ``E_base`` and the
    gt term would be normalized over inconsistent supports.
    """
    assert logits.dim() == 2, f"logits must be [N, V], got {tuple(logits.shape)}"
    n_tokens, vocab = logits.shape
    k = min(int(k), vocab) if k > 0 else vocab
    chunk_tokens = max(1, int(chunk_tokens))
    gt_col = gt_ids.reshape(n_tokens, 1)

    gt_parts, lp_parts, idx_parts = [], [], []
    for start in range(0, n_tokens, chunk_tokens):
        end = min(start + chunk_tokens, n_tokens)
        lp = F.log_softmax(logits[start:end].float() / temperature, dim=-1)  # [C, V]
        gt_c = gt_col[start:end]
        gt_parts.append(lp.gather(-1, gt_c).squeeze(-1))
        idx = lp.topk(k, dim=-1).indices  # [C, K]
        gt_in_topk = (idx == gt_c).any(dim=-1, keepdim=True)
        last_col = torch.zeros_like(idx, dtype=torch.bool)
        last_col[..., -1] = True
        idx = torch.where(last_col & ~gt_in_topk, gt_c.expand_as(idx), idx)
        lp_parts.append(lp.gather(-1, idx))
        idx_parts.append(idx)
        del lp, idx
    return (torch.cat(gt_parts, 0), torch.cat(lp_parts, 0), torch.cat(idx_parts, 0))


def gather_from_logits(
    logits: torch.Tensor,
    gt_ids: torch.Tensor,
    topk_idx: torch.Tensor,
    temperature: float = 1.0,
    chunk_tokens: int = OPD2_CHUNK_TOKENS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Teacher / teacher_base side: log-probs at the STUDENT's top-K columns.

    Same contract as :func:`topk_from_logits` but the columns are given, not
    chosen. Returns ``(gt_lp [N], topk_lp [N, K])``.
    """
    assert logits.dim() == 2, f"logits must be [N, V], got {tuple(logits.shape)}"
    n_tokens = logits.shape[0]
    chunk_tokens = max(1, int(chunk_tokens))
    gt_col = gt_ids.reshape(n_tokens, 1)

    gt_parts, lp_parts = [], []
    for start in range(0, n_tokens, chunk_tokens):
        end = min(start + chunk_tokens, n_tokens)
        lp = F.log_softmax(logits[start:end].float() / temperature, dim=-1)
        gt_parts.append(lp.gather(-1, gt_col[start:end]).squeeze(-1))
        lp_parts.append(lp.gather(-1, topk_idx[start:end]))
        del lp
    return torch.cat(gt_parts, 0), torch.cat(lp_parts, 0)


def combine_opd2_signal(
    base_gt: torch.Tensor,
    base_topk: torch.Tensor,
    eval_gt: torch.Tensor,
    eval_topk: torch.Tensor,
    tb_gt: torch.Tensor,
    tb_topk: torch.Tensor,
    rewards_bias: float = 0.0,
) -> torch.Tensor:
    """The five lines of OPD^2. All log-prob inputs are ``[N]`` / ``[N, K]``.

    Returns the per-token signal ``[N]`` (float32, NaN-free).
    """
    base_prob = base_topk.exp()
    mean_eval = (base_prob * eval_topk).sum(dim=-1)
    mean_tb = (base_prob * tb_topk).sum(dim=-1)
    mean_base = (base_prob * base_topk).sum(dim=-1)

    signal = (eval_gt - tb_gt) - (mean_eval - mean_tb)
    d_base = (eval_gt - base_gt) - (mean_eval - mean_base)
    signal = torch.where((signal * d_base) < 0, torch.zeros_like(signal), signal)
    signal = signal - rewards_bias
    return torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).float()


class OPD2Scorer:
    """Frozen teacher + teacher_base, parked on CPU between scoring phases.

    Loading, CPU/GPU parking and the two-stage ``model.model(...)`` +
    chunked ``lm_head`` forward are the same pattern as
    ``verl.workers.teacher_scoring_worker.TeacherScoringWorker``; the difference
    is that OPD^2 gathers at the STUDENT's top-K columns instead of taking the
    teacher's own top-K, so the two models must be co-resident with the student
    rather than living in a separate Ray actor (a ``[B, R, 1024]`` index tensor
    is not transportable at the paper's batch/length).
    """

    def __init__(
        self,
        teacher_path: str,
        teacher_base_path: str,
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        chunk_tokens: int = OPD2_CHUNK_TOKENS,
        use_teacher_template: bool = True,
        keep_on_gpu: bool = False,
        verbose: bool = False,
        student_tokenizer=None,
    ):
        from transformers import AutoConfig, AutoTokenizer

        self.chunk_tokens = int(chunk_tokens)
        self.keep_on_gpu = bool(keep_on_gpu)
        self.torch_dtype = getattr(torch, dtype)
        self._on_gpu = False
        self._logged_prompt = False
        self.verbose = verbose

        # Weights are NOT held in host RAM. to_gpu() streams them from disk
        # straight onto the GPU and to_cpu() frees them, so between scoring
        # phases they cost 0 RSS (only reclaimable page cache). Two 4B teachers
        # parked on CPU pinned ~16GB of RAM for the whole run.
        self._paths = (teacher_path, teacher_base_path)
        self._attn = attn_implementation
        self.teacher = None
        self.teacher_base = None

        # The teacher's own chat_template renders the teacher-side prompt; the
        # response ids are the student's and are shared because both tokenizers
        # use the same base BPE. Only the prompt convention differs.
        self.teacher_tokenizer = None
        if use_teacher_template:
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)

        self.vocab_size = min(int(AutoConfig.from_pretrained(p).vocab_size) for p in self._paths)

        # Student response ids are spliced into a teacher-vocabulary sequence, so
        # every id must mean the same thing to both. See _build_id_contract.
        self.id_remap: dict = {}
        self.masked_ids: Tuple[int, ...] = ()
        if student_tokenizer is not None and self.teacher_tokenizer is not None:
            self._build_id_contract(student_tokenizer)

    # ------------------------------------------------------- id contract ----
    def _build_id_contract(self, student_tokenizer, max_divergent: int = 64):
        """Reconcile student and teacher token ids, or refuse to run.

        OPD^2 splices the STUDENT's response ids into a TEACHER-rendered prompt and
        asks the teacher to score them. That is only meaningful where the two
        tokenizers agree on what an id means. For Qwen2.5-Instruct against the
        DeepSeek-R1-Distill family they agree on all 151643 base BPE ids and on
        151650+, and disagree on exactly seven control ids -- among them the
        student's generation EOS ``<|im_end|>`` (151645), which the teachers read
        as ``<|Assistant|>``: a turn-OPENING marker. Scoring it unmapped asks the
        teacher "how likely is a new assistant turn here?" at the end of every
        finished answer, which yields a systematic anti-termination advantage on
        the one token that controls response length.

        Every divergent id is resolved one of two ways, never left to chance:

          * **matched by declared role.** Both tokenizers name their own EOS / PAD /
            BOS / UNK / SEP / CLS / MASK, so a student id holding a role is mapped to
            whatever id holds that same role for the teacher. For this pair that
            resolves EOS (student 151645 ``<|im_end|>`` -> teacher 151643
            ``<|end_of_sentence|>``) and PAD (151643 -> 151643, an identity match
            that must NOT be masked). Roles are read from the tokenizers, so a
            different model pair is handled without editing this function.

          * **masked** otherwise -- its signal is zeroed downstream rather than
            guessed at. The five left over here are Qwen2.5
            ``additional_special_tokens`` (``<|im_start|>``, the object-ref and box
            pairs) for which DeepSeek genuinely has no counterpart; mapping
            ``<|box_start|>`` onto the teacher's ``<think>`` would inject a large
            spurious signal. A well-formed math response contains none of them, and
            ``opd2/unmapped_frac`` reports it at runtime if that ever stops being true.

        A disagreement wider than ``max_divergent`` ids is not a control-token
        mismatch but genuinely incompatible tokenizers, and raises.
        """
        t_tok = self.teacher_tokenizer
        n = min(len(student_tokenizer), len(t_tok), self.vocab_size)
        s_names = student_tokenizer.convert_ids_to_tokens(list(range(n)))
        t_names = t_tok.convert_ids_to_tokens(list(range(n)))
        divergent = [i for i in range(n) if s_names[i] != t_names[i]]
        if len(divergent) > max_divergent:
            raise ValueError(
                f'OPD^2 student and teacher tokenizers disagree on {len(divergent)} of '
                f'{n} ids (limit {max_divergent}). Response ids cannot be scored by the '
                'teacher; these models are not a compatible OPD^2 pair.')

        # student id -> teacher id, for every id both sides give the same ROLE.
        by_role = {}
        for role in ('eos', 'pad', 'bos', 'unk', 'sep', 'cls', 'mask'):
            s_id = getattr(student_tokenizer, f'{role}_token_id', None)
            t_id = getattr(t_tok, f'{role}_token_id', None)
            if s_id is not None and t_id is not None:
                by_role.setdefault(int(s_id), (int(t_id), role))

        remap, masked = {}, []
        for i in divergent:
            if i in by_role:
                t_id, _ = by_role[i]
                if t_id != i:
                    remap[i] = t_id
            else:
                masked.append(i)
        self.id_remap = remap
        self.masked_ids = tuple(masked)

        if self.verbose:
            print(f'[opd2] token-id contract: {n - len(divergent)}/{n} ids identical, '
                  f'{len(remap)} remapped, {len(masked)} masked', flush=True)
            for i in divergent:
                if i in remap:
                    how = f'-> {remap[i]} {t_names[remap[i]]!r} (role {by_role[i][1]})'
                elif i in by_role:
                    how = f'OK (role {by_role[i][1]} on both sides)'
                else:
                    how = 'MASKED (no counterpart)'
                print(f'[opd2]   {i}: student={s_names[i]!r} teacher={t_names[i]!r}  {how}',
                      flush=True)

    def remap_response_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Student ids -> teacher ids for the remappable control tokens."""
        for s_id, t_id in self.id_remap.items():
            ids = ids.masked_fill(ids == s_id, t_id)
        return ids

    def unmapped_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """True where a student id has no honest teacher counterpart."""
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for s_id in self.masked_ids:
            mask |= ids == s_id
        return mask

    def _load(self, path: str):
        from transformers import AutoModelForCausalLM

        # device_map loads each safetensors shard directly to the GPU, so the
        # weights never materialize in host RAM.
        model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=self.torch_dtype,
            attn_implementation=self._attn,
            device_map={"": torch.cuda.current_device()},
        )
        model.eval()
        model.requires_grad_(False)
        return model

    # ---------------------------------------------------------- residency ----
    def to_gpu(self):
        if not self._on_gpu:
            self.teacher = self._load(self._paths[0])
            self.teacher_base = self._load(self._paths[1])
            self._on_gpu = True

    def to_cpu(self):
        # ponytail: re-reads ~2x model size from disk each step (page-cache hit
        # after step 1, a few seconds); set keep_on_gpu=True if VRAM allows.
        if self._on_gpu and not self.keep_on_gpu:
            self.teacher = self.teacher_base = None
            self._on_gpu = False
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    def render_teacher_prompt_ids(self, chat, student_prompt_ids: List[int]) -> List[int]:
        """Teacher-side prompt ids for one row.

        ``chat`` is ``non_tensor_batch['raw_prompt']`` (the message list). With
        no teacher tokenizer configured, the student's own prompt ids are reused
        -- OPD's default for same-template families.
        """
        if self.teacher_tokenizer is None:
            return list(student_prompt_ids)
        text = self.teacher_tokenizer.apply_chat_template(
            list(chat), tokenize=False, add_generation_prompt=True)
        if self.verbose and not self._logged_prompt:
            self._logged_prompt = True
            print(f"[opd2] teacher prompt render (first row):\n{text}\n[opd2] ---", flush=True)
        return self.teacher_tokenizer.encode(text, add_special_tokens=False)

    # ------------------------------------------------------------ scoring ----
    @torch.no_grad()
    def score_rows(
        self,
        seqs: List[List[int]],
        resp_lens: List[int],
        gt_ids: List[torch.Tensor],
        topk_idx: List[torch.Tensor],
        temperature: float = 1.0,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Score a micro-batch of full ``prompt+response`` sequences.

        Each row ``i`` contributes ``resp_lens[i]`` response positions; ``gt_ids[i]``
        is ``[R_i]`` and ``topk_idx[i]`` is ``[R_i, K]`` (from the student).
        ``temperature`` is the ROLLOUT temperature and must be the same one the
        student's columns were produced at -- the reference scores all three
        models at ``self.temperature``.

        Returns per row ``(eval_gt, eval_topk, tb_gt, tb_topk)``.
        """
        if not self._on_gpu:
            raise RuntimeError("OPD2Scorer.score_rows called while parked on CPU; call to_gpu().")
        device = next(self.teacher.parameters()).device
        lengths = [len(s) for s in seqs]
        max_len = max(lengths)
        pad_id = 0

        input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            attention_mask[i, :len(s)] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        out = [None] * len(seqs)
        per_model = {}
        for name, model in (("eval", self.teacher), ("tb", self.teacher_base)):
            rows = self._forward_rows(model, input_ids, attention_mask, lengths,
                                      resp_lens, gt_ids, topk_idx, device, temperature)
            per_model[name] = rows
        for i in range(len(seqs)):
            e_gt, e_tk = per_model["eval"][i]
            t_gt, t_tk = per_model["tb"][i]
            out[i] = (e_gt, e_tk, t_gt, t_tk)
        return out

    @torch.no_grad()
    def score_and_combine(self, seqs, resp_lens, gt_ids, topk_idx, base_gt, base_topk,
                          temperature: float = 1.0, rewards_bias: float = 0.0):
        """Dedicated-GPU entry point (this class wrapped as a Ray actor holding its
        own GPU, see main_ppo): score one micro-batch and return only the per-row
        signal ``[R_i]`` on CPU, so the ``[R, K]`` teacher tensors never leave this
        GPU. ``topk_idx`` may arrive as int32 to halve the transfer."""
        self.to_gpu()
        device = next(self.teacher.parameters()).device
        scored = self.score_rows(seqs, resp_lens, gt_ids, [t.long() for t in topk_idx],
                                 temperature=temperature)
        return [combine_opd2_signal(base_gt[i].to(device), base_topk[i].to(device),
                                    *scored[i], rewards_bias).cpu()
                for i in range(len(seqs))]

    def _forward_rows(self, model, input_ids, attention_mask, lengths, resp_lens,
                      gt_ids, topk_idx, device, temperature):
        """One model's forward + per-row response-span gather.

        Hidden states first ([b, L, H] is small); the ``lm_head`` is applied in
        token chunks so the [C, V] logits peak is bounded regardless of response
        length -- the same discipline as
        ``teacher_scoring_worker._score_micro_batch``.
        """
        base = getattr(model, "model", None) or model.base_model
        hidden = base(input_ids=input_ids, attention_mask=attention_mask,
                      use_cache=False).last_hidden_state
        rows = []
        for i, L in enumerate(lengths):
            r_len = resp_lens[i]
            # Row j of the logits predicts sequence position j+1. Response token
            # t sits at position (L - r_len + t), so it is predicted by row
            # (L - r_len + t - 1). Same alignment as
            # teacher_kd.score_batch_and_attach.
            start = L - r_len - 1
            h = hidden[i, start:start + r_len, :]
            gt_i = gt_ids[i].to(device)
            idx_i = topk_idx[i].to(device)
            gt_parts, lp_parts = [], []
            for c0 in range(0, r_len, self.chunk_tokens):
                c1 = min(c0 + self.chunk_tokens, r_len)
                logits = model.lm_head(h[c0:c1]).float()  # [C, V]
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                lp = F.log_softmax(logits / temperature, dim=-1)
                del logits
                gt_parts.append(lp.gather(-1, gt_i[c0:c1, None]).squeeze(-1))
                lp_parts.append(lp.gather(-1, idx_i[c0:c1]))
                del lp
            rows.append((torch.cat(gt_parts, 0), torch.cat(lp_parts, 0)))
            del h, gt_parts, lp_parts
        del hidden
        return rows


# --------------------------------------------------------------- self-check --
def _reference_signal(base_gt, base_topk, eval_gt, eval_topk, tb_gt, tb_topk, bias):
    """Literal transcription of opd2_trainer.py's five lines, for the diff."""
    base_prob = base_topk.exp()
    mean_eval_logp = (base_prob * eval_topk).sum(dim=-1)
    mean_tb_logp = (base_prob * tb_topk).sum(dim=-1)
    mean_base_logp = (base_prob * base_topk).sum(dim=-1)
    per_token_signal = (eval_gt - tb_gt) - (mean_eval_logp - mean_tb_logp)
    d_base = (eval_gt - base_gt) - (mean_eval_logp - mean_base_logp)
    per_token_signal = torch.where(
        (per_token_signal * d_base) < 0,
        torch.zeros_like(per_token_signal), per_token_signal,
    )
    return per_token_signal - bias


def _self_check():
    torch.manual_seed(0)
    N, V, K = 37, 512, 16

    # 1. combine_opd2_signal == the reference formula.
    b_lp = F.log_softmax(torch.randn(N, K), dim=-1)
    e_lp = F.log_softmax(torch.randn(N, K), dim=-1)
    t_lp = F.log_softmax(torch.randn(N, K), dim=-1)
    b_gt, e_gt, t_gt = torch.randn(N), torch.randn(N), torch.randn(N)
    got = combine_opd2_signal(b_gt, b_lp, e_gt, e_lp, t_gt, t_lp, 0.25)
    want = _reference_signal(b_gt, b_lp, e_gt, e_lp, t_gt, t_lp, 0.25)
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()

    # 2. the direction gate zeroes exactly the sign-disagreeing positions.
    #    (checked pre-bias, since the bias is subtracted after the gate)
    raw = combine_opd2_signal(b_gt, b_lp, e_gt, e_lp, t_gt, t_lp, 0.0)
    bp = b_lp.exp()
    me = (bp * e_lp).sum(-1)
    s = (e_gt - t_gt) - (me - (bp * t_lp).sum(-1))
    d = (e_gt - b_gt) - (me - (bp * b_lp).sum(-1))
    assert torch.equal(raw == 0, (s * d) < 0) or ((s * d) < 0).sum() == 0, "gate mismatch"
    assert 0 < int(((s * d) < 0).sum()) < N, "degenerate gate fixture"

    # 3. topk_from_logits force-includes a gt that misses the top-K.
    logits = torch.randn(N, V)
    worst = logits.argmin(dim=-1)
    gt_lp, tk_lp, tk_idx = topk_from_logits(logits, worst, K, chunk_tokens=8)
    assert (tk_idx == worst.unsqueeze(-1)).any(-1).all(), "gt not force-included"
    ref_lp = F.log_softmax(logits.float(), dim=-1)
    assert torch.allclose(gt_lp, ref_lp.gather(-1, worst[:, None]).squeeze(-1), atol=1e-6)
    assert torch.allclose(tk_lp, ref_lp.gather(-1, tk_idx), atol=1e-6)

    # 4. gather_from_logits agrees with a plain full-vocab log_softmax.
    g_gt, g_tk = gather_from_logits(logits, worst, tk_idx, chunk_tokens=8)
    assert torch.allclose(g_tk, ref_lp.gather(-1, tk_idx), atol=1e-6)
    assert torch.allclose(g_gt, gt_lp, atol=1e-6)

    # 5. K <= 0 degenerates to the exact full-vocab case.
    _, full_lp, full_idx = topk_from_logits(logits, worst, 0, chunk_tokens=8)
    assert full_idx.shape[-1] == V and torch.allclose(full_lp.exp().sum(-1),
                                                      torch.ones(N), atol=1e-4)

    # 6. the student/teacher id contract: remap EOS, mask the rest, refuse a
    #    wholesale mismatch. Mirrors the real Qwen2.5 / DeepSeek-R1 divergence.
    class _Tok:
        def __init__(self, names, **roles):
            self.names = names
            for r in ('eos', 'pad', 'bos', 'unk', 'sep', 'cls', 'mask'):
                setattr(self, f'{r}_token_id', roles.get(r))

        def __len__(self):
            return len(self.names)

        def convert_ids_to_tokens(self, ids):
            return [self.names[i] for i in ids]

    # Mirrors the real pair: id 20 is PAD on both sides (identical role, differing
    # name -> matched, NOT masked); 22 is the student's EOS (-> teacher EOS 20);
    # 21 and 23 are student-only control tokens with no counterpart.
    shared = [f"tok{i}" for i in range(20)]
    student = _Tok(shared + ["<|endoftext|>", "<|im_start|>", "<|im_end|>",
                             "<|box_start|>"], eos=22, pad=20)
    teacher = _Tok(shared + ["<|end_of_sentence|>", "<|User|>", "<|Assistant|>",
                             "<think>"], eos=20, pad=20)

    def _contract(s, t, **kw):
        sc = object.__new__(OPD2Scorer)
        sc.teacher_tokenizer, sc.vocab_size, sc.verbose = t, len(t), False
        sc._build_id_contract(s, **kw)
        return sc

    sc = _contract(student, teacher)
    assert sc.id_remap == {22: 20}, sc.id_remap          # <|im_end|> -> teacher EOS
    assert sc.masked_ids == (21, 23), sc.masked_ids      # 20 matched by PAD role

    ids = torch.tensor([3, 22, 21, 23, 7, 20])
    assert torch.equal(sc.remap_response_ids(ids), torch.tensor([3, 20, 21, 23, 7, 20]))
    assert torch.equal(sc.unmapped_mask(ids),
                       torch.tensor([False, False, True, True, False, False]))
    # content ids survive untouched, including in a [R, K] top-K index tensor
    cols = torch.arange(20).reshape(4, 5)
    assert torch.equal(sc.remap_response_ids(cols), cols)

    # a tokenizer pair that disagrees wholesale is not an OPD^2 pair
    try:
        _contract(student, _Tok([f"x{i}" for i in range(24)], eos=20), max_divergent=4)
    except ValueError:
        pass
    else:
        raise AssertionError("wholesale vocab mismatch must raise")

    print("opd2_signal self-check OK")


if __name__ == "__main__":
    _self_check()
