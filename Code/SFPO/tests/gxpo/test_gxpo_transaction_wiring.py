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
    launcher_7 = (EFFICIENCY / 'qwen25_math_7b_gxpo_b256_a03_k10.sh').read_text()


    assert 'MODEL_QWEN25_MATH_1P5B' in launcher_15
    assert 'Qwen2.5-Math-1.5B-Instruct' in launcher_15
    assert 'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"' in launcher_15
    assert 'PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"' in launcher_15
    assert 'ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"' in launcher_15
    assert 'ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"' in launcher_15
    assert 'MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"' in launcher_15
    assert 'GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"' in launcher_15

    assert 'MODEL_QWEN25_MATH_7B' in launcher_7
    assert 'export K="${K:-5}"' in launcher_7
    assert 'PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"' in launcher_7
    assert 'GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"' in launcher_7
    assert 'REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.3}"' in launcher_7
    assert 'ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.8}"' in launcher_7

    common = (EFFICIENCY / 'common.sh').read_text()
    muon = (EFFICIENCY / 'qwen2.5_1.5b_muon.sh').read_text()
    assert 'OPTIMIZER_NAME="${OPTIMIZER_NAME:-adamw}"' in common
    assert 'actor_rollout_ref.actor.optim.name="$OPTIMIZER_NAME"' in common
    assert 'OPTIMIZER_NAME="${OPTIMIZER_NAME:-muon}"' in muon
    assert 'MUON_DISTRIBUTED_BACKEND="${MUON_DISTRIBUTED_BACKEND:-gather_scatter}"' in muon
