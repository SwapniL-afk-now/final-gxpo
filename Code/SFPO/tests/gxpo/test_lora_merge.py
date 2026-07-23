"""LoRA merge-for-rollout parity check (CPU only).

Run: python tests/gxpo/test_lora_merge.py

Verifies that verl.utils.lora_utils.merge_lora_state_dict (used by the
FSDP->vLLM sharding manager) produces exactly the same weights and key names
as peft's own merge_and_unload on a tiny random Llama.
"""

import importlib.util
import os

import torch

REPO = os.path.join(os.path.dirname(__file__), '..', '..')


def load_merge_fn():
    path = os.path.join(REPO, 'verl', 'utils', 'lora_utils.py')
    spec = importlib.util.spec_from_file_location('lora_utils', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.merge_lora_state_dict


def main():
    from transformers import LlamaConfig, LlamaForCausalLM
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    config = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, vocab_size=256)
    base = LlamaForCausalLM(config)
    original = {k: v.clone() for k, v in base.state_dict().items()}

    lora_config = LoraConfig(task_type='CAUSAL_LM', r=8, lora_alpha=16,
                             target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias='none')
    peft_model = get_peft_model(base, lora_config)
    # non-zero adapters so the merge actually changes weights
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if 'lora_' in name:
                p.normal_(0.0, 0.05)

    merge_lora_state_dict = load_merge_fn()
    scaling = lora_config.lora_alpha / lora_config.r
    ours = merge_lora_state_dict(dict(peft_model.state_dict()), scaling)

    expected = peft_model.merge_and_unload().state_dict()

    assert set(ours.keys()) == set(expected.keys()), (
        f'key mismatch:\n only ours: {sorted(set(ours) - set(expected))[:5]}'
        f'\n only peft: {sorted(set(expected) - set(ours))[:5]}')
    for key in expected:
        assert torch.allclose(ours[key], expected[key], atol=1e-6), f'weight mismatch at {key}'

    changed = sum(1 for k in expected if k in original and not torch.equal(expected[k], original[k]))
    assert changed == config.num_hidden_layers * len(lora_config.target_modules), \
        f'expected every targeted projection to change, got {changed}'
    print(f'PASS LoRA merge parity ({len(expected)} tensors, adapters folded into {changed} projections)')


if __name__ == '__main__':
    main()
