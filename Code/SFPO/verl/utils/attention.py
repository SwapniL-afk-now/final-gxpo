"""Attention-backend selection shared by FSDP model builders."""

from __future__ import annotations

from collections.abc import Mapping

import torch

DEFAULT_ATTENTION_IMPLEMENTATION = "flash_attention_2"


def resolve_attention_implementation(model_config, override_config: Mapping | None = None) -> str:
    """Return the validated backend passed to ``from_pretrained``."""
    override_config = override_config or {}
    requested = override_config.get("attn_implementation")
    if requested is None:
        requested = model_config.get("attn_implementation", DEFAULT_ATTENTION_IMPLEMENTATION)
    requested = str(requested)
    validate_attention_implementation(requested)
    return requested


def validate_attention_implementation(attention_implementation: str) -> None:
    """Fail loudly when a requested non-default backend is unavailable."""
    if attention_implementation == "flash_attention_3":
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (9, 0):
            raise RuntimeError("flash_attention_3 requires a Hopper-class CUDA GPU (SM90+)")
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        except Exception as exc:
            raise RuntimeError("The selected Transformers build has no attention registry") from exc
        if attention_implementation not in ALL_ATTENTION_FUNCTIONS:
            raise RuntimeError("Transformers does not register flash_attention_3")
        try:
            try:
                from flash_attn_3 import flash_attn_interface  # noqa: F401
            except ImportError:
                import flash_attn_interface  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "flash_attention_3 was requested but the official FA3 interface is unavailable"
            ) from exc
    elif attention_implementation not in {"eager", "sdpa", "flash_attention_2"}:
        raise ValueError(f"Unsupported attention implementation: {attention_implementation}")
