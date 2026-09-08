"""Static wiring checks for the transactional GXPO path and launchers."""

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ACTOR = REPO / 'verl' / 'workers' / 'actor' / 'dp_actor.py'
EFFICIENCY = REPO / 'experiments' / 'gxpo_efficiency'


def _function_source(path, function_name):
    source = path.read_text()
    tree = ast.parse(source)
    node = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == function_name)
    return source, ast.get_source_segment(source, node)


def test_gxpo_uses_transaction_before_slow_gradient():
    source, step_source = _function_source(ACTOR, '_gxpo_minibatch_step')
    assert 'snapshot_optimizer_state(self.actor_optimizer)' in step_source
    assert step_source.count('probe_optimizer_step()\n') == 3
    assert step_source.index('optimizer_transaction.restore()') < step_source.index(
        '# Pass 3: slow correction')
    assert 'probe-step optimizer-moment' not in source


def test_launchers_enable_transactional_mode_and_keep_requested_settings():
    launcher_15 = (EFFICIENCY / 'qwen25_math_1p5b_gxpo_b256_mb64_gate_v6.sh').read_text()
    wrapper = (EFFICIENCY / 'qwen25_math_1p5b_gxpo_b256_a05.sh').read_text()

    # Experiment-owned settings live in the gate_v6 entrypoint ...
    assert 'MODEL_QWEN25_MATH_1P5B' in launcher_15
    assert 'Qwen2.5-Math-1.5B-Instruct' in launcher_15
    assert 'export K=10' in launcher_15
    assert 'export REPOSITION_ALPHA=0.3' in launcher_15
    assert 'GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"' in launcher_15
    # ... while the batch/hardware envelope lives in the b256 wrapper it execs.
    assert 'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"' in wrapper
    assert 'PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"' in wrapper
    assert 'GXPO_WARMUP_STEPS="${GXPO_WARMUP_STEPS:-0}"' in wrapper
    assert 'GXPO_RESET_ENTROPY_AFTER_WARMUP="${GXPO_RESET_ENTROPY_AFTER_WARMUP:-False}"' in wrapper

    common = (EFFICIENCY / 'common.sh').read_text()
    assert 'GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"' in common
    assert 'actor_rollout_ref.actor.gxpo_optimizer_state_mode="$GXPO_OPTIMIZER_STATE_MODE"' in common

    # KD efficiency paths pin the mode instead of relying on the trainer default.
    kd_gate = (EFFICIENCY / 'qwen25_math_1p5b_kd_gxpo_b256_mb64_gate_v6.sh').read_text()
    assert 'GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"' in kd_gate
    assert '+optim.gxpo_optimizer_state_mode="$GXPO_OPTIMIZER_STATE_MODE"' in kd_gate
    offpolicy = (EFFICIENCY / 'qwen25_math_1p5b_offpolicy_kd_gxpo_k10_a03.sh').read_text()
    assert 'GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"' in offpolicy


def test_optimizer_switch_defaults_to_adamw_and_supports_muon():
    common = (EFFICIENCY / 'common.sh').read_text()
    assert 'OPTIMIZER_NAME="${OPTIMIZER_NAME:-adamw}"' in common
    assert '+actor_rollout_ref.actor.optim.name="$OPTIMIZER_NAME"' in common
    assert 'MUON_DISTRIBUTED_BACKEND="${MUON_DISTRIBUTED_BACKEND:-gather_scatter}"' in common
    assert '+actor_rollout_ref.actor.optim.muon_distributed_backend=' in common
    assert 'Unsupported OPTIMIZER_NAME' in common


def test_muon_gxpo_scripts_pin_muon_and_their_named_optimizer_state_mode():
    # Each model gets one launcher per optimizer-state arm (named, not env-toggled)
    # so `ls` alone tells you which mode a script runs -- no reading the body required.
    scripts_and_modes = (
        ('qwen25_math_1p5b_gxpo_muon_transactional_b64_mb16.sh', 'transactional'),
        ('qwen25_math_1p5b_gxpo_muon_faststate_b64_mb16.sh', 'transactional_fast_state'),
        ('llama32_3b_gxpo_muon_transactional_b128_mb32.sh', 'transactional'),
        ('llama32_3b_gxpo_muon_faststate_b128_mb32.sh', 'transactional_fast_state'),
    )
    for script, mode in scripts_and_modes:
        text = (EFFICIENCY / script).read_text()
        assert 'OPTIMIZER_NAME="${OPTIMIZER_NAME:-muon}"' in text
        assert 'MUON_DISTRIBUTED_BACKEND="${MUON_DISTRIBUTED_BACKEND:-gather_scatter}"' in text
        assert f'export GXPO_OPTIMIZER_STATE_MODE="{mode}"' in text, (
            f'{script} must hardcode its own optimizer-state mode, not leave it '
            f'env-overridable -- that is the point of splitting the file by mode')
        assert 'source "$SCRIPT_DIR/common.sh"' in text


# Every launcher that forwards the mode must offer BOTH optimizer-state arms, not just
# transactional: validate the value, and tag the run name so the two never collide.
MODE_BLOCK_MARKERS = (
    'GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"',
    'transactional_fast_state) OPT_STATE_TAG="_optkeep" ;;',
    'PREFLIGHT FAIL: GXPO_OPTIMIZER_STATE_MODE must be',
)


def _mode_forwarding_launchers():
    """Every shell launcher that passes gxpo_optimizer_state_mode to the trainer."""
    roots = (REPO / 'train-scripts', REPO / 'experiments' / 'gxpo_efficiency')
    found = []
    for root in roots:
        for path in sorted(root.rglob('*.sh')):
            text = path.read_text()
            if 'gxpo_optimizer_state_mode=' in text:
                found.append((path, text))
    assert found, 'no GXPO launchers found -- the glob is wrong'
    return found


def test_every_gxpo_launcher_supports_the_no_refresh_mode():
    for path, text in _mode_forwarding_launchers():
        owns_block = all(marker in text for marker in MODE_BLOCK_MARKERS)
        # A launcher may instead inherit the block from common.sh or from the
        # launcher it execs; those delegates are checked on their own pass.
        delegates = 'source "$SCRIPT_DIR/common.sh"' in text or 'exec ' in text
        assert owns_block or delegates, (
            f'{path.relative_to(REPO)} forwards gxpo_optimizer_state_mode but neither '
            f'validates/tags the mode itself nor delegates to a launcher that does')


def test_mode_owning_launchers_tag_the_run_name():
    """The tag must reach the run name, or the two arms overwrite each other."""
    for path, text in _mode_forwarding_launchers():
        if not all(marker in text for marker in MODE_BLOCK_MARKERS):
            continue
        assert '${OPT_STATE_TAG}' in text, (
            f'{path.relative_to(REPO)} defines OPT_STATE_TAG but never applies it to a '
            f'run name (EXP / RUN_NAME / GXPO_RUN_NAME)')


def test_no_launcher_re_defaults_the_mode_past_its_validation():
    """An inline `${GXPO_OPTIMIZER_STATE_MODE:-transactional}` at the flag site would
    silently bypass the case block's rejection of an unknown value."""
    for path, text in _mode_forwarding_launchers():
        for line in text.splitlines():
            if 'gxpo_optimizer_state_mode=' in line and line.lstrip().startswith('+'):
                assert '${GXPO_OPTIMIZER_STATE_MODE:-' not in line, (
                    f'{path.relative_to(REPO)}: forward the validated variable '
                    f'("$GXPO_OPTIMIZER_STATE_MODE"), not a re-default: {line.strip()}')
