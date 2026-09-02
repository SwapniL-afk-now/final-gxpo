import ast
import math
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[2]
TRAINER_PATH = REPO / 'verl' / 'trainer' / 'ppo' / 'ray_trainer.py'
source = TRAINER_PATH.read_text()
tree = ast.parse(source)
helper_node = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef)
                   and node.name == 'mixed_response_indices')
helper_module = ast.Module(body=[helper_node], type_ignores=[])
namespace = {'torch': torch, 'math': math}
exec(compile(ast.fix_missing_locations(helper_module), str(TRAINER_PATH), 'exec'), namespace)
mixed_response_indices = namespace['mixed_response_indices']


def config_for(n):
    return SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(n=n),
        ),
    )


def test_mixed_response_indices_keep_complete_prompt_groups():
    rewards = torch.tensor([1, 1, 1, 0, 0, 0, 0, 1], dtype=torch.float32)
    rows, prompt_count = mixed_response_indices(
        rewards, config_for(2), divisor=2)
    assert prompt_count == 2
    assert rows.tolist() == [2, 3, 6, 7]


def test_mixed_response_indices_respect_distributed_divisibility():
    rewards = torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.float32)
    rows, prompt_count = mixed_response_indices(
        rewards, config_for(2), divisor=4)
    # Three mixed prompts are reduced to two complete prompt groups, so the
    # selected rows can be evenly dispatched across four ranks.
    assert prompt_count == 2
    assert rows.tolist() == [0, 1, 2, 3]


def test_trainer_preserves_full_batch_metrics_when_filtering_actor_updates():
    source = TRAINER_PATH.read_text()
    assert "filter_mixed_responses', actor_cfg.get('use_gxpo', False)" in source
    assert "actor_update_indices = list(range(batch_num))" in source
    assert "examples/mixed_response_filtering_enabled" in source
    assert 'if use_gxpo or (filter_mixed_responses and not use_sfpo):' in source



def test_llama_muon_launchers_are_self_contained_and_configured():
    launcher_dir = REPO / 'experiments' / 'gxpo_efficiency'
    for method in ('grpo', 'gxpo'):
        launcher = (launcher_dir / f'llama32_3b_muon_{method}.sh').read_text()
        assert 'source "$SCRIPT_DIR/common.sh"' not in launcher
        assert 'OPTIMIZER_NAME="${OPTIMIZER_NAME:-muon}"' in launcher
        assert 'GPU_IDS="${GPU_IDS:-0,1,2,3}"' in launcher
        assert 'FSDP_SIZE="${FSDP_SIZE:-4}"' in launcher
        assert 'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"' in launcher
        assert 'PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"' in launcher
        assert 'ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"' in launcher
        assert 'ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.8}"' in launcher
        assert 'MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"' in launcher
        assert 'FILTER_MIXED_RESPONSES="${FILTER_MIXED_RESPONSES:-True}"' in launcher
        assert 'actor_rollout_ref.rollout.enforce_eager=$VLLM_ENFORCE_EAGER' in launcher
    gxpo = (launcher_dir / 'llama32_3b_muon_gxpo.sh').read_text()
    assert 'K="${K:-10}"' in gxpo
    assert 'REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.3}"' in gxpo
    assert 'GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"' in gxpo
