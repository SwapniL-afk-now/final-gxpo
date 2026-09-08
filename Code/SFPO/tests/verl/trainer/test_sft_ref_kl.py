"""SFT reference-KL anchor (D_KL(pi_theta || pi_ref)) and flat-launcher wiring."""

import sys
from pathlib import Path

import re
import torch

REPO = Path(__file__).resolve().parents[3]
EFFICIENCY = REPO / 'experiments' / 'gxpo_efficiency'
OFFPOLICY = REPO / 'train-scripts' / 'off-policy-sft-kd'

sys.path.insert(0, str(REPO))
from verl.trainer.fsdp_sft_trainer import (  # noqa: E402
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
        assert 'data.kl_beta=' in text, name
        assert 'benchmark_eval_freq=5' in text, name


def test_launchers_eval_wiring_fixed():
    for name in ('run_kd_sft_hybrid_dapo_lighteval_1p5b.sh',
                 'run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh'):
        text = _launcher(name)
        # CODE_ROOT must land on Code/SFPO (tools live under it), not Code.
        assert 'CODE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"' in text, name
        assert '--max-num-seqs 256' in text, name
        assert '--attention-backend' in text, name
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
            '_seed${TRAIN_SEED}${OPT_STATE_TAG}"') in text


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
    assert 'ALPHA="${ALPHA:-0.8}"' in text
    assert 'trainer.total_training_steps="${MAX_STEPS:-400}"' in text
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
        assert 'benchmark_eval_freq=5' in text, name
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


def test_gxpo_launcher_defaults_extrapolate_rather_than_contract():
    """K/alpha must put the effective multiplier alpha*scale above 1.

    theta_tilde = theta0 + alpha*scale*(theta2-theta0), and `scale` is bounded to
    [1, K/2+1], tending to K/2 as the retention ratio r -> 1 (tiny probe steps).
    So alpha*K/2 is the multiplier the launcher actually converges to. Below 1 the
    reposition lands SHORT of theta2 and the 3-pass update contracts -- which is
    what K=5/alpha=0.3 did (0.3 * 2.5 = 0.75).
    """
    text = _launcher('run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh')
    k = int(re.search(r'K="\$\{K:-(\d+)\}"', text).group(1))
    alpha = float(re.search(r'ALPHA="\$\{ALPHA:-([\d.]+)\}"', text).group(1))
    assert alpha * (k / 2.0) > 1.0, (k, alpha, alpha * k / 2.0)
    # And the floor of the scale range must not drag it below 1 by much either.
    assert alpha >= 0.5, alpha


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
