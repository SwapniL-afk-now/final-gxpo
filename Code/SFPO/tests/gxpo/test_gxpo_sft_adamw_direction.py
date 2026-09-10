"""Optimizer-aware AdamW retention in the SFT / SFT-KD trainer.

The SFT and KD launchers do NOT go through `dp_actor.py`; they drive
`verl/trainer/fsdp_sft_trainer.py` (and `KDSFTTrainer`, which subclasses it),
which carries its own GXPO implementation. These tests cover that second copy
and the launcher wiring that reaches it.

Run directly with:
    PYTHONPATH=Code/SFPO python tests/gxpo/test_gxpo_sft_adamw_direction.py
"""

import ast
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer  # noqa: E402
from verl.workers.actor.gxpo_state import RetentionKind, adamw_direction  # noqa: E402

SFT_TRAINER_PATH = REPO / 'verl' / 'trainer' / 'fsdp_sft_trainer.py'
KD_TRAINER_PATH = REPO / 'verl' / 'trainer' / 'kd_sft_trainer.py'
SFT_KD_LAUNCHER = (REPO / 'train-scripts' / 'off-policy-sft-kd'
                   / 'run_kd_sft_gxpo_hybrid_dapo_lighteval_1p5b.sh')
EFFICIENCY = REPO / 'experiments' / 'gxpo_efficiency'

LR = 3e-3
WD = 0.04
K = 10
DELTA = 1e-8


class _StubTrainer:
    """Minimal carrier for the SFT trainer's GXPO classification helpers."""

    _gxpo_adamw_direction_supported = FSDPSFTTrainer._gxpo_adamw_direction_supported
    _gxpo_retention_kinds = FSDPSFTTrainer._gxpo_retention_kinds
    _gxpo_param_group_hparams = FSDPSFTTrainer._gxpo_param_group_hparams
    _gxpo_global_tensor_rms = FSDPSFTTrainer._gxpo_global_tensor_rms
    _gxpo_adamw_direction_rms = FSDPSFTTrainer._gxpo_adamw_direction_rms
    _gxpo_capture_grads = FSDPSFTTrainer._gxpo_capture_grads

    def __init__(self, optimizer, params, space='auto'):
        self.optimizer = optimizer
        self._gxpo_params = list(params)
        self._gxpo_retention_space = space
        self._gxpo_retention_cache = None
        self._gxpo_unsupported_optimizer_warned = False
        self._gxpo_fsdp_invariant_threshold = True


def _params(n=2, size=32, seed=0):
    torch.manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(size)) for _ in range(n)]


# --------------------------------------------------------------------------
# Config surface
# --------------------------------------------------------------------------

def test_sft_trainer_defaults_to_the_legacy_estimator():
    """Unlike the actor, 'auto' never existed here, so the default must stay grad.

    Every pre-existing SFT / SFT-KD launcher was written against r = g1/g0 and
    none of them passes the flag; defaulting to 'auto' would change the algorithm
    under all of them at once, with no run-name change to show for it.
    """
    source = SFT_TRAINER_PATH.read_text()
    assert "self.config.optim.get('gxpo_retention_space', 'grad')" in source, (
        "the SFT trainer's gxpo_retention_space default must be 'grad'")

    # And no SFT-trainer launcher may rely on that default while claiming auto.
    unflagged = []
    for path in sorted((REPO / 'train-scripts').rglob('*.sh')):
        text = path.read_text()
        if '+optim.use_gxpo=True' in text and 'gxpo_retention_space' not in text:
            unflagged.append(path.relative_to(REPO))
    # Those launchers are fine precisely because the default is legacy; this
    # assertion documents the dependency so the default cannot be flipped
    # without someone seeing this test.
    assert unflagged, 'expected legacy SFT launchers that rely on the grad default'


def test_update_space_is_rejected_not_silently_aliased():
    source = SFT_TRAINER_PATH.read_text()
    assert "if space == 'update':" in source
    assert 'is not supported by the SFT ' in source
    assert "must be auto or grad" in source


def test_kd_trainer_inherits_the_sft_gxpo_step():
    """The KD launchers reach this code through KDSFTTrainer, not dp_actor."""
    tree = ast.parse(KD_TRAINER_PATH.read_text())
    kd = next(node for node in ast.walk(tree)
              if isinstance(node, ast.ClassDef) and node.name == 'KDSFTTrainer')
    bases = [b.id if isinstance(b, ast.Name) else getattr(b, 'attr', '') for b in kd.bases]
    assert 'FSDPSFTTrainer' in bases
    assert '_gxpo_training_step' not in {n.name for n in kd.body
                                         if isinstance(n, ast.FunctionDef)}, (
        'KDSFTTrainer must inherit the GXPO step, not fork it')


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def test_auto_classifies_every_parameter_as_adamw_direction():
    params = _params(seed=1)
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    kinds = _StubTrainer(optimizer, params, 'auto')._gxpo_retention_kinds()
    assert kinds == [RetentionKind.ADAMW_DIRECTION] * 2


def test_grad_keeps_the_all_legacy_fast_path():
    params = _params(seed=2)
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    assert _StubTrainer(optimizer, params, 'grad')._gxpo_retention_kinds() is None


def test_non_adamw_optimizer_falls_back_with_a_warning(capsys):
    params = _params(seed=3)
    for optimizer in (torch.optim.SGD(params, lr=LR),
                      torch.optim.Adam(params, lr=LR)):   # coupled decay: not AdamW
        trainer = _StubTrainer(optimizer, params, 'auto')
        assert trainer._gxpo_retention_kinds() is None
        assert 'WARNING' in capsys.readouterr().out


def test_classification_is_cached():
    params = _params(seed=4)
    trainer = _StubTrainer(torch.optim.AdamW(params, lr=LR), params, 'auto')
    first = trainer._gxpo_retention_kinds()
    assert trainer._gxpo_retention_kinds() is first


def test_param_group_hparams_are_read_per_group():
    a, b = _params(n=2, seed=5)
    optimizer = torch.optim.AdamW(
        [{'params': [a], 'lr': 1e-3, 'weight_decay': 0.0},
         {'params': [b], 'lr': 5e-2, 'weight_decay': 0.2}])
    lrs, wds = _StubTrainer(optimizer, [a, b], 'auto')._gxpo_param_group_hparams()
    assert lrs == [1e-3, 5e-2]
    assert wds == [0.0, 0.2]


# --------------------------------------------------------------------------
# Buffer reuse
# --------------------------------------------------------------------------

def test_g1_capture_is_skipped_for_slots_holding_u0():
    """The g1 slot is reused for u0, so its gradient capture must be skippable."""
    params = _params(n=3, size=8, seed=6)
    for parameter in params:
        parameter.grad = torch.full_like(parameter, 7.0)
    trainer = _StubTrainer(torch.optim.AdamW(params, lr=LR), params, 'auto')

    bufs = [torch.full_like(p, -1.0) for p in params]
    trainer._gxpo_capture_grads(bufs, skip=[True, False, True])
    assert torch.equal(bufs[0], torch.full_like(bufs[0], -1.0)), 'skipped slot was clobbered'
    assert torch.equal(bufs[2], torch.full_like(bufs[2], -1.0))
    assert torch.equal(bufs[1], torch.full_like(bufs[1], 7.0))

    # No skip mask -> historical behavior, every slot captured.
    bufs = [torch.full_like(p, -1.0) for p in params]
    trainer._gxpo_capture_grads(bufs)
    assert all(torch.equal(b, torch.full_like(b, 7.0)) for b in bufs)


def test_sft_gxpo_still_allocates_exactly_three_model_sized_buffers():
    source = SFT_TRAINER_PATH.read_text()
    assert "for n in ('theta0', 'g0', 'g1')" in source
    tree = ast.parse(source)
    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == '_gxpo_training_step')
    body = ast.get_source_segment(source, step)
    assert 'theta1 = t0f + u0' in body and 'u0 = g1b.float()' in body, (
        'theta1 must be reconstructed from u0 in the g1 slot, not stored')
    assert 'capture_skip=u0_slots' in body
    assert 'del theta1' in body and 'del d0, d1' in body
    # Exactly one model-sized allocation site: the three-buffer dict above.
    # A fourth buffer is a whole extra copy of the model per rank.
    assert body.count('torch.empty_like') == 1
    alloc = next(line for line in body.splitlines() if 'torch.empty_like' in line)
    assert "('theta0', 'g0', 'g1')" in alloc and 'theta1' not in alloc


# --------------------------------------------------------------------------
# Arithmetic: the trainer's own helpers on a real AdamW run
# --------------------------------------------------------------------------

def test_direction_rms_and_ratio_on_a_real_two_probe_sequence():
    params = _params(n=2, size=48, seed=7)
    optimizer = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=WD)
    trainer = _StubTrainer(optimizer, params, 'auto')
    kinds = trainer._gxpo_retention_kinds()
    adamw_indices = [i for i, k in enumerate(kinds)
                     if k == RetentionKind.ADAMW_DIRECTION]

    torch.manual_seed(8)
    for _ in range(4):                      # warm the moments
        for parameter in params:
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    theta0 = [p.detach().clone() for p in params]
    for parameter in params:
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    g1_bufs = [p.detach() - t0 for p, t0 in zip(params, theta0)]   # u0 in the g1 slot
    lrs0, wds0 = trainer._gxpo_param_group_hparams()

    for parameter in params:
        parameter.grad = torch.randn_like(parameter) * 3.0
    optimizer.step()
    lrs1, wds1 = trainer._gxpo_param_group_hparams()

    d0_rms = trainer._gxpo_adamw_direction_rms(adamw_indices, theta0, g1_bufs, lrs0, wds0)
    assert set(d0_rms) == set(adamw_indices)

    from verl.workers.actor.gxpo_state import compute_gxpo_adamw_direction_retention_scale
    for index in adamw_indices:
        t0f = theta0[index].float()
        theta1 = t0f + g1_bufs[index].float()
        d0 = adamw_direction(t0f, theta1, lrs0[index], wds0[index])
        d1 = adamw_direction(theta1, params[index].data.float(), lrs1[index], wds1[index])
        assert torch.allclose(d0_rms[index], d0.square().mean().sqrt(), rtol=1e-5)

        ratio, scale, active, _ = compute_gxpo_adamw_direction_retention_scale(
            d0, d1, K, DELTA, d0_rms=d0_rms[index])
        assert torch.allclose(ratio, (d1 / d0).clamp(-2.0, 3.0), rtol=1e-5, atol=1e-6)
        assert torch.isfinite(scale).all()
        assert (scale >= 1.0).all() and (scale <= K / 2.0 + 1.0).all()


def test_legacy_and_adamw_stats_use_separate_denominators():
    """train/gxpo_r_mean must not silently average in AdamW-direction ratios."""
    source = SFT_TRAINER_PATH.read_text()
    tree = ast.parse(source)
    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == '_gxpo_training_step')
    body = ast.get_source_segment(source, step)
    assert 'r_mean = sum_r / grad_coords' in body
    assert 'adamw_r_mean = adamw_sum_r / adamw_coords' in body
    for key in ('train/gxpo_adamw_direction_params', 'train/gxpo_adamw_r_mean',
                'train/gxpo_adamw_r_std', 'train/gxpo_adamw_scale_mean',
                'train/gxpo_adamw_scale_max', 'train/gxpo_adamw_inactive_frac',
                'train/gxpo_adamw_ratio_clip_frac'):
        assert key in body, key
    # g1 is squared into a gradient norm only on the legacy path; on the AdamW
    # path that slot holds u0, a displacement, and folding it in would be
    # meaningless.
    assert body.index('if kind == RetentionKind.ADAMW_DIRECTION:') < body.index(
        'stats[1] += torch.linalg.vector_norm(g1b).double()')


# --------------------------------------------------------------------------
# Launcher wiring
# --------------------------------------------------------------------------

KD_LAUNCHERS = (
    SFT_KD_LAUNCHER,
    EFFICIENCY / 'qwen25_math_1p5b_kd_gxpo_b256_mb64_gate_v6.sh',
    EFFICIENCY / 'qwen25_math_1p5b_onpolicy_kd_gxpo.sh',
    EFFICIENCY / 'qwen25_3b_onpolicy_kd_gxpo.sh',
)


@pytest.mark.parametrize('launcher', KD_LAUNCHERS, ids=lambda p: p.name)
def test_kd_launchers_opt_in_validate_and_tag(launcher):
    text = launcher.read_text()
    assert 'GXPO_RETENTION_SPACE="${GXPO_RETENTION_SPACE:-auto}"' in text, (
        f'{launcher.name} must opt in explicitly, not inherit a default')
    assert 'auto)   RETENTION_TAG="_adamwdir" ;;' in text
    assert 'grad)   RETENTION_TAG="" ;;' in text
    # 'update' is the Muon estimator and must be refused, not silently accepted.
    assert 'GXPO_RETENTION_SPACE=update is the Muon per-matrix estimator' in text
    # The tag must reach the run name, or the two arms share a directory.
    assert '${RETENTION_TAG}' in text, (
        f'{launcher.name} defines RETENTION_TAG but never applies it to a run name')
    # The validated variable is forwarded, never re-defaulted at the flag site.
    forwarded = [line for line in text.splitlines()
                 if 'gxpo_retention_space=' in line and line.lstrip().startswith('+')]
    assert forwarded, f'{launcher.name} must forward gxpo_retention_space to the trainer'
    for line in forwarded:
        assert '"$GXPO_RETENTION_SPACE"' in line, line.strip()
        assert '${GXPO_RETENTION_SPACE:-' not in line, line.strip()


@pytest.mark.parametrize('launcher', KD_LAUNCHERS, ids=lambda p: p.name)
def test_kd_launchers_target_the_right_config_prefix(launcher):
    """SFT/KD-trainer launchers use +optim.*; actor launchers use +actor_rollout_ref.*."""
    text = launcher.read_text()
    if '+optim.use_gxpo=True' in text:
        assert '+optim.gxpo_retention_space=' in text
    else:
        assert '+actor_rollout_ref.actor.gxpo_retention_space=' in text


def test_plain_kd_launcher_is_untouched():
    """The non-GXPO KD launcher must not have grown a GXPO flag."""
    plain = (REPO / 'train-scripts' / 'off-policy-sft-kd'
             / 'run_kd_sft_hybrid_dapo_lighteval_1p5b.sh').read_text()
    assert 'gxpo_retention_space' not in plain
    assert 'use_gxpo=True' not in plain


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))


# --------------------------------------------------------------------------
# Multi-rank throughput contract
# --------------------------------------------------------------------------

def test_sft_kd_launcher_multi_gpu_enables_nccl_and_zero2():
    """A comma GPU list must scale ranks AND drop the two settings that make
    multi-rank FSDP slow, while GPU=<single> keeps today's behaviour exactly."""
    text = SFT_KD_LAUNCHER.read_text()
    assert "NPROC=\"$(awk -F, '{print NF}' <<< \"$GPU\")\"" in text
    assert '--nproc_per_node="$NPROC"' in text
    # NCCL transport follows the rank count, not a hardcoded 1.
    assert 'NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-$(( NPROC > 1 ? 0 : 1 ))}"' in text
    assert 'NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-$(( NPROC > 1 ? 0 : 1 ))}"' in text
    # ZeRO-2 for multi-rank, unchanged ZeRO-3 for one rank.
    assert 'FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY:-shard_grad_op}"' in text
    assert 'FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY:-full_shard}"' in text
    assert '++model.fsdp_config.sharding_strategy="$FSDP_SHARDING_STRATEGY"' in text
    # The old flag named a strategy the SFT trainer never reads.
    assert 'fsdp_config.strategy=fsdp2' not in text
    # Accumulation must divide evenly across ranks.
    assert 'TRAIN_BATCH_SIZE % (NPROC * MICRO_BATCH_SIZE)' in text


def test_sft_trainer_honours_sharding_strategy_and_prefetch():
    source = SFT_TRAINER_PATH.read_text()
    assert "self.config.model.fsdp_config.get('sharding_strategy', 'full_shard')" in source
    assert 'sharding_strategy=sharding_strategy,' in source
    assert 'forward_prefetch=bool(' in source
    assert 'limit_all_gathers=True,' in source


def test_gxpo_stats_never_materialize_full_size_fp64():
    """Every reduced statistic is parameter-shaped; .double() on one of those
    allocates a second, 2x-size copy of the whole flat parameter. Accumulate in
    fp64 instead of casting into it."""
    source = SFT_TRAINER_PATH.read_text()
    tree = ast.parse(source)
    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == '_gxpo_training_step')
    body = ast.get_source_segment(source, step)
    for name in ('g0b', 'g1b', 'disp2', 'dispK', 'r', 'scale', 'eff', 'active',
                 'ratio_clipped'):
        assert f'{name}.double()' not in body, name


# --------------------------------------------------------------------------
# Metric honesty: a key is emitted only when its estimator ran
# --------------------------------------------------------------------------

def _sft_step_body():
    source = SFT_TRAINER_PATH.read_text()
    tree = ast.parse(source)
    step = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == '_gxpo_training_step')
    return ast.get_source_segment(source, step)


LEGACY_ONLY_SFT_KEYS = ('train/gxpo_g1_norm', 'train/gxpo_r_mean', 'train/gxpo_r_std',
                        'train/gxpo_cos_g0_g1', 'train/gxpo_inactive_frac',
                        'train/gxpo_ratio_clip_frac', 'train/gxpo_scale_max',
                        'train/gxpo_clip_scale_g1')
ADAMW_ONLY_SFT_KEYS = ('train/gxpo_adamw_r_mean', 'train/gxpo_adamw_r_std',
                       'train/gxpo_adamw_scale_mean', 'train/gxpo_adamw_scale_max',
                       'train/gxpo_adamw_inactive_frac',
                       'train/gxpo_adamw_ratio_clip_frac',
                       'train/gxpo_adamw_d0_norm', 'train/gxpo_adamw_d1_norm',
                       'train/gxpo_adamw_cos_d0_d1')


def test_sft_undefined_metric_families_are_omitted_not_zeroed():
    """Under 'auto' the g1 slot holds u0, so there is no g1 norm and no g1/g0
    ratio. Emitting 0.0 makes a placeholder indistinguishable from a
    measurement; the key must be absent instead."""
    body = _sft_step_body()
    # Both families live behind a coverage guard, not in the unconditional dict.
    assert 'if has_grad:' in body
    assert 'if has_adamw:' in body
    guard_grad = body.index('if has_grad:')
    guard_adamw = body.index('if has_adamw:')
    base = body.index('gxpo_metrics = {')
    for key in LEGACY_ONLY_SFT_KEYS:
        assert key in body, key
        assert body.index(key) > guard_grad, f'{key} must be gated on has_grad'
    for key in ADAMW_ONLY_SFT_KEYS:
        assert key in body, key
        assert body.index(key) > guard_adamw, f'{key} must be gated on has_adamw'
    # ...and the guards come after the always-defined base dict.
    assert base < guard_grad < guard_adamw


def test_sft_always_logs_a_retention_kind_census():
    body = _sft_step_body()
    for key in ('train/gxpo_retention_kind', 'train/gxpo_legacy_grad_params',
                'train/gxpo_adamw_direction_params'):
        assert key in body, key
        assert body.index(key) < body.index('if has_grad:'), \
            f'{key} must be unconditional'
    # Path-agnostic metrics describe the reposition, not a ratio: unconditional.
    for key in ('train/gxpo_g0_norm', 'train/gxpo_scale_mean',
                'train/gxpo_effective_multiplier', 'train/gxpo_disp2_norm',
                'train/gxpo_dispK_norm', 'train/gxpo_clip_scale_g0'):
        assert body.index(key) < body.index('if has_grad:'), key


def test_sft_legacy_path_no_longer_discards_activity_masks():
    """compute_gxpo_retention_scale already returns active/ratio_clipped; the
    SFT arm used to throw them away, which is why it had no inactive_frac."""
    body = _sft_step_body()
    assert 'r, scale, active, ratio_clipped = compute_gxpo_retention_scale(' in body
    assert '_active, _ratio_clipped' not in body
