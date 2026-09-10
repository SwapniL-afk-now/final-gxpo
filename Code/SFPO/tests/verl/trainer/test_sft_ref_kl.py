"""SFT reference-KL anchor (D_KL(pi_theta || pi_ref)) and flat-launcher wiring."""

import sys
from pathlib import Path

import re
import types

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
EFFICIENCY = REPO / 'experiments' / 'gxpo_efficiency'
OFFPOLICY = REPO / 'train-scripts' / 'off-policy-sft-kd'

sys.path.insert(0, str(REPO))
from verl.trainer.fsdp_sft_trainer import (  # noqa: E402
    FSDPSFTTrainer,
    full_kl_per_token,
    k3_kl_per_token,
    mean_over_batch_rows,
    mean_response_entropy,
)


def test_mean_response_entropy_matches_manual_masked_mean():
    torch.manual_seed(3)
    logits = torch.randn(3, 11, 43)
    mask = torch.zeros(3, 11, dtype=torch.bool)
    mask[0, 2:7] = True
    mask[1, :11] = True
    mask[2, 5:6] = True
    got = mean_response_entropy(logits, mask)
    logp = torch.log_softmax(logits.float(), dim=-1)
    ent = -(logp.exp() * logp).sum(dim=-1)
    row_sums = torch.stack([ent[i][mask[i]].sum() for i in range(3)])
    assert torch.allclose(got, row_sums.mean(), atol=1e-5)


def test_mean_response_entropy_ignores_prompt_positions():
    logits = torch.zeros(2, 6, 17)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 3:] = True
    got = mean_response_entropy(logits, mask)
    # Uniform over 17 tokens -> ln(17) per response token x 3 tokens per row,
    # meaned over the 2 rows (per-response sums, consistent with the loss).
    assert torch.allclose(got, torch.tensor(float(3 * __import__('math').log(17))), atol=1e-4)


def test_k3_matches_schulman_formula():
    torch.manual_seed(0)
    student_logp = torch.randn(7) * 2.0
    ref_logp = torch.randn(7) * 2.0
    got = k3_kl_per_token(student_logp, ref_logp)
    r = ref_logp.float() - student_logp.float()
    expected = torch.clamp(torch.exp(r) - r - 1.0, min=-10.0, max=10.0)
    assert torch.allclose(got, expected, atol=1e-6)


def test_k3_is_zero_for_identical_policies():
    logp = torch.randn(5) * 3.0
    got = k3_kl_per_token(logp, logp.clone())
    assert torch.allclose(got, torch.zeros(5), atol=1e-5)


def test_k3_is_nonnegative_and_unbiased_at_zero_drift():
    # Identical policies collide exactly; a symmetric perturbation averages
    # near the analytic KL of the perturbed pair (checked against full-vocab).
    torch.manual_seed(2)
    base = torch.randn(200, 60)
    log_p = torch.log_softmax(base, dim=-1)
    log_q = torch.log_softmax(base + 0.05 * torch.randn(200, 60), dim=-1)
    taken = torch.randint(0, 60, (200,))
    k3 = k3_kl_per_token(log_p[range(200), taken], log_q[range(200), taken])
    assert bool((k3 >= 0.0).all())
    exact = (log_p.exp() * (log_p - log_q)).sum(dim=-1).clamp_min(0.0)
    assert abs(float(k3.mean()) - float(exact.mean())) < 0.05


def test_k3_accepts_bf16_inputs():
    student = torch.randn(4, dtype=torch.bfloat16)
    ref = torch.randn(4, dtype=torch.bfloat16)
    got = k3_kl_per_token(student, ref)
    assert got.shape == (4,) and torch.isfinite(got).all()


def test_mean_over_batch_rows_equal_weights_responses():
    sums = torch.tensor([[1.0, 2.0], [10.0, 20.0]])
    assert torch.allclose(mean_over_batch_rows(sums), torch.tensor(16.5))


def _launcher(name):
    return (OFFPOLICY / name).read_text()


def test_launchers_train_flat_teacher_responses_not_topk_cache():
    flat = 'qwen25_math7b_dapo_lighteval_train_n1_correct_only.parquet'
    for name in ('run_kd_sft_hybrid_dapo_lighteval_1p5b.sh',
                 'run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh'):
        text = _launcher(name)
        assert flat in text, name
        for stale in ('TRAIN_SPLIT', 'teacher_topk_log_probs', 'teacher_topk_ids',
                      'kd_sft_trainer', 'build_teacher_topk', 'kd_temperature'):
            assert stale not in text, (name, stale)
        assert '-m verl.trainer.fsdp_sft_trainer' in text, name
        assert 'data.response_key=teacher_response' in text, name
        assert '++data.val_micro_batch_size_per_gpu="${VAL_MICRO_BATCH_SIZE:-4}"' in text, name
        assert 'data.kl_beta=' in text, name
        assert '++trainer.benchmark_eval_freq="${BENCHMARK_EVAL_FREQ:-0}"' in text, name


def test_launchers_eval_wiring_fixed():
    for name in ('run_kd_sft_hybrid_dapo_lighteval_1p5b.sh',
                 'run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh'):
        text = _launcher(name)
        # CODE_ROOT must land on Code/SFPO (tools live under it), not Code.
        assert 'CODE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"' in text, name
        assert '--max-num-seqs 256' in text, name
        assert '--attention-backend' in text, name
        assert '--n 4 --temperature 0.7' in text, name
        assert '--log-wandb' in text, name


def test_gxpo_launcher_keeps_transactional_pin():
    text = _launcher('run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh')
    assert '+optim.use_gxpo=True' in text
    assert 'GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"' in text
    assert '+optim.gxpo_optimizer_state_mode="$GXPO_OPTIMIZER_STATE_MODE"' in text


def test_gxpo_launcher_offers_both_optimizer_state_modes_under_separate_names():
    """Both arms are runnable from this one launcher and can never collide.

    The transactional EXP stays untagged so the existing baseline run dir and its
    wandb history keep matching; the no-refresh arm gets its own `_optkeep` name.
    """
    text = _launcher('run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh')
    assert 'transactional)            OPT_STATE_TAG="" ;;' in text
    assert 'transactional_fast_state) OPT_STATE_TAG="_optkeep" ;;' in text
    assert 'PREFLIGHT FAIL: GXPO_OPTIMIZER_STATE_MODE must be' in text
    assert ('EXP="sftkl_gxpo_k${K}_a${ALPHA}_b${KL_BETA}_flat14k_lr${LR}'
            '_seed${TRAIN_SEED}${OPT_STATE_TAG}${RETENTION_TAG}${NPROC_TAG}"') in text


def test_sft_trainer_refreshes_optimizer_state_only_in_transactional_mode():
    source = (REPO / 'verl' / 'trainer' / 'fsdp_sft_trainer.py').read_text()
    assert "if mode not in ('transactional', 'transactional_fast_state'):" in source
    assert ("if self.gxpo_optimizer_state_mode == 'transactional':\n"
            "            optimizer_transaction.restore()") in source
    # The mode is logged so the two arms are separable in wandb.
    assert "'train/gxpo_optim_state_kept'" in source


def test_gxpo_launcher_update_and_gate_profile():
    text = _launcher('run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh')
    assert 'K="${K:-3}"' in text
    assert 'ALPHA="${ALPHA:-0.1}"' in text
    assert 'trainer.total_training_steps="${MAX_STEPS:-300}"' in text
    assert 'GXPO_TAU="${GXPO_TAU:-3.0}"' in text
    assert 'GXPO_TRIGGER_PATIENCE="${GXPO_TRIGGER_PATIENCE:-3}"' in text
    assert 'GXPO_WARMUP="${GXPO_WARMUP:-0}"' in text
    assert 'GXPO_ZSCORE_W="${GXPO_ZSCORE_W:-30}"' in text
    # Full gate profile explicit in bash: no code defaults allowed.
    for var in ('GXPO_TRIGGER_SIGNAL', 'GXPO_SHUTOFF_MODE', 'GXPO_TRIGGER_ROBUST',
                'GXPO_TRIGGER_SUSTAIN_W', 'GXPO_TRIGGER_MIN_OBS',
                'GXPO_TRIGGER_ABS_THRESHOLD', 'GXPO_MAX_ACTIVE_STEPS',
                'GXPO_RELATIVE_THRESHOLD', 'GXPO_FALLBACK_MODE', 'GXPO_FALLBACK_WINDOW'):
        assert f'{var}="${{{var}:-' in text, var
    for flag in ('gxpo_trigger_signal', 'gxpo_trigger_robust', 'gxpo_trigger_sustain_w',
                 'gxpo_zscore_w', 'gxpo_trigger_min_obs', 'gxpo_trigger_abs_threshold',
                 'gxpo_max_active_steps', 'gxpo_relative_threshold',
                 'gxpo_fallback_mode', 'gxpo_fallback_window'):
        assert f'+optim.{flag}=' in text, flag


def test_sft_trainer_forwards_full_gate_profile():
    source = (REPO / 'verl' / 'trainer' / 'fsdp_sft_trainer.py').read_text()
    # Gate math knobs reach GXPOState; the signal selector is consumed where
    # the observation is built (entropy needs a forward pass, norms don't).
    for key in ('gxpo_trigger_patience', 'gxpo_trigger_robust',
                'gxpo_trigger_sustain_w', 'gxpo_trigger_min_obs',
                'gxpo_trigger_abs_threshold', 'gxpo_max_active_steps',
                'gxpo_relative_threshold', 'gxpo_fallback_mode', 'gxpo_fallback_window'):
        assert f"'{key}'" in source, key
    assert 'gxpo_trigger_signal' in source
    assert 'stat_override' in source
    assert '_last_entropy_mean' in source


def test_eval_gpu_budget_fits_alongside_training_state():
    for name in ('run_kd_sft_hybrid_dapo_lighteval_1p5b.sh',
                 'run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh'):
        text = _launcher(name)
        assert 'SFT_VLLM_EVAL_GPU_UTIL="${SFT_VLLM_EVAL_GPU_UTIL:-0.6}"' in text, name


def test_launchers_eval_single_sample_cadence():
    for name in ('run_kd_sft_hybrid_dapo_lighteval_1p5b.sh',
                 'run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh'):
        text = _launcher(name)
        assert 'EVAL_N="${EVAL_N:-1}"' in text, name
        assert 'EVAL_SEEDS="${EVAL_SEEDS:-0}"' in text, name
        assert 'SFT_VLLM_EVAL_SKIP_GREEDY="${SFT_VLLM_EVAL_SKIP_GREEDY:-1}"' in text, name
        assert '++trainer.eval_sample_n=1' in text, name
        assert '++trainer.eval_seed_count=1' in text, name
        assert '++trainer.eval_skip_greedy=1' in text, name
        assert '++trainer.benchmark_eval_freq="${BENCHMARK_EVAL_FREQ:-0}"' in text, name
        assert 'EVAL_CODE_ROOT=' in text, name
        # The in-training eval harness runs under `set -u` and reads $GPU;
        # it must be exported or every cadence eval dies unbound.
        assert 'export GPU' in text, name


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'PASS {name}')
    print('ALL SFT REF-KL CHECKS PASSED')


def test_gxpo_launcher_alpha_is_the_requested_conservative_value():
    """The requested alpha 0.1 intentionally keeps GXPO conservative for K=3."""
    text = _launcher('run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh')
    k = int(re.search(r'K="\$\{K:-(\d+)\}"', text).group(1))
    alpha = float(re.search(r'ALPHA="\$\{ALPHA:-([\d.]+)\}"', text).group(1))
    assert k == 3
    assert alpha == 0.1


def test_gxpo_launcher_gate_is_reachable_before_the_budget_cap():
    """A warmup >= max_active_steps makes the shutoff gate unreachable.

    check_trigger() returns early while step < warmup_steps, and is_enabled() hard-stops
    GXPO once step >= max_active_steps, so warmup >= budget means the entropy gate can
    never fire and the run silently loses its safety valve.
    """
    text = _launcher('run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh')
    warmup = int(re.search(r'GXPO_WARMUP="\$\{GXPO_WARMUP:-(\d+)\}"', text).group(1))
    budget = int(re.search(r'GXPO_MAX_ACTIVE_STEPS="\$\{GXPO_MAX_ACTIVE_STEPS:-(\d+)\}"',
                           text).group(1))
    assert budget > 0, 'budget cap restored (knights-and-knaves duty cycle)'
    assert warmup < budget, (warmup, budget)


def test_gxpo_launcher_forwards_the_contraction_guard():
    text = _launcher('run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh')
    assert 'GXPO_MIN_EFFECTIVE_MULTIPLIER="${GXPO_MIN_EFFECTIVE_MULTIPLIER:-0}"' in text
    assert ('+optim.gxpo_min_effective_multiplier="$GXPO_MIN_EFFECTIVE_MULTIPLIER"'
            in text)


def test_sft_trainer_applies_and_reports_the_effective_multiplier():
    source = (REPO / 'verl' / 'trainer' / 'fsdp_sft_trainer.py').read_text()
    # alpha is applied through the guarded multiplier, never bare on dispK.
    assert 'eff = scale.float() * alpha' in source
    assert 'eff.clamp_(min=min_eff)' in source
    assert 'p.data.copy_(disp2.mul_(eff).add_(t0.float()))' in source
    assert 'dispK.mul_(alpha)' not in source
    # Reported every step, and warned about once.
    assert "'train/gxpo_effective_multiplier'" in source
    assert "'train/gxpo_contracting'" in source
    assert '_gxpo_contraction_warned' in source


def test_rl_actor_mirrors_the_contraction_guard():
    source = (REPO / 'verl' / 'workers' / 'actor' / 'dp_actor.py').read_text()
    assert 'eff = scale * alpha' in source
    assert 'eff.clamp_(min=min_eff)' in source
    assert 'p.data.copy_(disp2.mul_(eff).add_(t0.float()))' in source
    assert 'dispK.mul_(alpha)' not in source
    assert "'actor/gxpo_effective_multiplier'" in source
    assert "'actor/gxpo_contracting'" in source
    # The all-reduce split must follow the stats width, not a hardcoded one.
    assert 'full[:stats.numel()], full[stats.numel():]' in source


# --- full-vocabulary teacher KL (data.kl_full=True) ---------------------------


def test_full_kl_is_zero_for_identical_distributions():
    torch.manual_seed(11)
    logits = torch.randn(17, 29)
    got = full_kl_per_token(logits, logits.clone())
    assert torch.allclose(got, torch.zeros(17), atol=1e-6)


def test_full_kl_matches_an_independent_implementation():
    """Cross-check against F.kl_div, which computes KL(target || input)."""
    torch.manual_seed(5)
    teacher = torch.randn(23, 7)
    student = torch.randn(23, 7)
    got = full_kl_per_token(teacher, student)
    expected = torch.nn.functional.kl_div(
        torch.log_softmax(student.float(), dim=-1),
        torch.log_softmax(teacher.float(), dim=-1),
        log_target=True, reduction='none').sum(dim=-1)
    assert torch.allclose(got, expected, atol=1e-6)


def test_full_kl_is_the_forward_direction_not_the_reverse():
    """KL(teacher || student), so swapping the arguments must change the value."""
    torch.manual_seed(7)
    teacher = torch.randn(9, 13) * 3.0
    student = torch.randn(9, 13)
    assert not torch.allclose(full_kl_per_token(teacher, student),
                              full_kl_per_token(student, teacher), atol=1e-3)


def test_full_kl_chunking_does_not_change_the_result():
    torch.manual_seed(13)
    teacher, student = torch.randn(40, 11), torch.randn(40, 11)
    one_shot = full_kl_per_token(teacher, student, chunk_tokens=1000)
    for chunk in (1, 3, 7, 40):
        assert torch.allclose(full_kl_per_token(teacher, student, chunk_tokens=chunk),
                              one_shot, atol=1e-6), chunk


def test_full_kl_is_nonnegative():
    torch.manual_seed(17)
    for _ in range(20):
        got = full_kl_per_token(torch.randn(8, 31), torch.randn(8, 31))
        assert (got >= -1e-6).all()


def test_full_kl_gradient_reaches_the_student_only():
    teacher = torch.randn(6, 19, requires_grad=True)
    student = torch.randn(6, 19, requires_grad=True)
    full_kl_per_token(teacher, student).sum().backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    # The teacher is frozen: detached inside the kernel, so no gradient at all.
    assert teacher.grad is None


def test_full_kl_handles_an_empty_mask():
    assert full_kl_per_token(torch.zeros(0, 5), torch.zeros(0, 5)).shape == (0,)


# --- token-budgeted micro-batches (data.use_dynamic_bsz=True) -----------------


def _splitter(**data_cfg):
    """_split_micro_batches only touches self.config, so a stub is enough."""
    from omegaconf import OmegaConf
    from tensordict import TensorDict

    stub = types.SimpleNamespace(config=OmegaConf.create({'data': data_cfg}))

    def split(lengths, width=None):
        width = width or max(lengths)
        mask = torch.zeros(len(lengths), width, dtype=torch.long)
        for row, length in enumerate(lengths):
            mask[row, :length] = 1
        batch = TensorDict({'attention_mask': mask, 'input_ids': mask.clone()},
                           batch_size=len(lengths))
        return FSDPSFTTrainer._split_micro_batches(stub, batch)

    return split


def test_dynamic_split_respects_the_padded_token_budget():
    lengths = [10, 4000, 25, 3000, 8, 12, 900, 7, 1500, 30]
    groups = _splitter(use_dynamic_bsz=True, max_token_len_per_gpu=8192)(lengths)
    for group in groups:
        real = group['attention_mask'].sum(dim=1)
        # The cost that matters is rows * widest row, because the trainer trims
        # each micro-batch to its widest row rather than packing it.
        assert len(real) * int(real.max()) <= 8192


def test_dynamic_split_keeps_every_row_exactly_once():
    lengths = [10, 4000, 25, 3000, 8, 12, 900, 7, 1500, 30]
    groups = _splitter(use_dynamic_bsz=True, max_token_len_per_gpu=8192)(lengths)
    seen = sorted(int(n) for g in groups for n in g['attention_mask'].sum(dim=1))
    assert seen == sorted(lengths)


def test_dynamic_split_is_deterministic_for_gxpos_three_passes():
    lengths = [10, 4000, 25, 3000, 8, 12, 900, 7, 1500, 30]
    split = _splitter(use_dynamic_bsz=True, max_token_len_per_gpu=8192)
    shapes = [[tuple(g['attention_mask'].shape) for g in split(lengths)] for _ in range(3)]
    assert shapes[0] == shapes[1] == shapes[2]


def test_dynamic_split_rejects_a_budget_below_the_longest_row():
    with pytest.raises(ValueError, match='longest sequence'):
        _splitter(use_dynamic_bsz=True, max_token_len_per_gpu=512)([10, 900])


def test_dynamic_split_ignores_padding_width_not_real_length():
    """Rows padded to max_length must be budgeted by their REAL length."""
    split = _splitter(use_dynamic_bsz=True, max_token_len_per_gpu=8192)
    # 40 rows of 100 real tokens each, padded out to 16384 as SFTDataset does.
    groups = split([100] * 40, width=16384)
    assert len(groups) == 1, 'real cost is 40*100=4000, well inside the budget'


def test_fixed_split_is_unchanged_and_weights_reduce_to_one_over_n():
    """The row-weighted accumulation must reproduce the old 1/n exactly."""
    groups = _splitter(use_dynamic_bsz=False, micro_batch_size_per_gpu=4)([9] * 12)
    assert [int(g.batch_size[0]) for g in groups] == [4, 4, 4]
    total = sum(int(g.batch_size[0]) for g in groups)
    assert all(abs(int(g.batch_size[0]) / total - 1 / len(groups)) < 1e-12 for g in groups)


# --- launcher wiring ---------------------------------------------------------


def test_launchers_distill_from_the_teacher_with_full_kl():
    for name in ('run_kd_sft_hybrid_dapo_lighteval_1p5b.sh',
                 'run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh'):
        text = _launcher(name)
        assert 'data.kl_full=True' in text, name
        assert 'DeepScaleR-1.5B-Preview' in text, name
        # The teacher must NOT be the student's own init any more.
        assert 'KL_REF_MODEL="${KL_REF_MODEL:-$MODEL}"' not in text, name
        assert 'data.use_dynamic_bsz=True' in text, name
        assert 'data.max_token_len_per_gpu=' in text, name


def test_launchers_run_validation_and_benchmarks_every_ten_steps():
    for name in ('run_kd_sft_hybrid_dapo_lighteval_1p5b.sh',
                 'run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh'):
        text = _launcher(name)
        assert 'data.val_files="$VAL"' in text, name
        assert 'data.val_files=null' not in text, name
        assert '++trainer.test_freq="${TEST_FREQ:-10}"' in text, name
        assert '++trainer.benchmark_eval_freq="${BENCHMARK_EVAL_FREQ:-10}"' in text, name


def test_launchers_carry_the_requested_schedule():
    for name in ('run_kd_sft_hybrid_dapo_lighteval_1p5b.sh',
                 'run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh'):
        text = _launcher(name)
        assert 'LR="${LR:-5e-6}"' in text, name
        assert 'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"' in text, name
        assert 'MAX_LENGTH="${MAX_LENGTH:-16384}"' in text, name
        assert 'trainer.total_training_steps="${MAX_STEPS:-300}"' in text, name
        # total_training_steps is a min() cap against total_epochs*steps_per_epoch,
        # so too few epochs would silently end the run early: 6391/512 = 12 steps
        # per epoch, so 300 steps needs 25 epochs to bind.
        assert 'trainer.total_epochs="${TOTAL_EPOCHS:-25}"' in text, name


def test_trainer_builds_a_teacher_from_its_own_config():
    source = (REPO / 'verl' / 'trainer' / 'fsdp_sft_trainer.py').read_text()
    # Reusing the student's config drops an untied teacher lm_head silently.
    assert 'ref_config = AutoConfig.from_pretrained(' in source
    assert 'config=ref_config' in source
    assert 'config=config, torch_dtype=torch.bfloat16' not in source
    assert 'vocab_size' in source
