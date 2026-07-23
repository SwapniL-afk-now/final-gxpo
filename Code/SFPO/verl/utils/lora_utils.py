# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""LoRA state-dict merging for rollout weight sync (torch-only, no vLLM deps)."""

import torch

__all__ = ['merge_lora_state_dict']


def _full_tensor(t):
    return t.full_tensor() if hasattr(t, 'full_tensor') else t


def merge_lora_state_dict(params, lora_scaling: float):
    """Fold peft LoRA adapters into base weights and strip peft key prefixes.

    Input keys follow peft naming (base_model.model.<path>.base_layer.weight,
    base_model.model.<path>.lora_A.default.weight, ...); output keys are plain
    HF names (<path>.weight) with W_merged = W_base + scaling * (B @ A), so the
    vLLM weight loaders receive ordinary full model weights.
    """
    lora_A = {k: v for k, v in params.items() if '.lora_A.' in k}
    lora_B = {k: v for k, v in params.items() if '.lora_B.' in k}

    merged = {}
    for key, value in params.items():
        if '.lora_A.' in key or '.lora_B.' in key:
            continue
        out_key = key
        if out_key.startswith('base_model.model.'):
            out_key = out_key[len('base_model.model.'):]
        if '.base_layer.' in key:
            a_key = key.replace('.base_layer.', '.lora_A.default.')
            b_key = key.replace('.base_layer.', '.lora_B.default.')
            out_key = out_key.replace('.base_layer.', '.')
            if a_key in lora_A and b_key in lora_B:
                a = _full_tensor(lora_A[a_key]).to(torch.float32)
                b = _full_tensor(lora_B[b_key]).to(torch.float32)
                delta = (lora_scaling * (b @ a))
                if hasattr(value, 'full_tensor'):  # DTensor shard: distribute delta onto the same layout
                    from torch.distributed.tensor import distribute_tensor
                    delta = distribute_tensor(delta.to(value.dtype), value.device_mesh, value.placements)
                    value = value + delta
                else:
                    value = value + delta.to(value.dtype)
        merged[out_key] = value
    return merged
