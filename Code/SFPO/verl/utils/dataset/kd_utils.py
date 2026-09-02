"""Shared utilities for offline sparse token-distribution distillation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


KD_SCHEMA_VERSION = 1


def tokenizer_fingerprint(tokenizer) -> str:
    """Hash every tokenizer property that can change token alignment."""
    payload = {
        "vocab": sorted(tokenizer.get_vocab().items(), key=lambda item: (item[1], item[0])),
        "special_tokens_map": tokenizer.special_tokens_map,
        "chat_template": getattr(tokenizer, "chat_template", None),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def format_prompt_response(tokenizer, prompt, response) -> tuple[str, str]:
    """Match the exact prompt/response formatting used by ``SFTDataset``."""
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)) and all(isinstance(item, dict) for item in prompt):
        prompt_chat = list(prompt)
    else:
        prompt_chat = [{"role": "user", "content": str(prompt)}]
    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=False
        )
    else:
        prompt_text = str(prompt)
        if not prompt_text.endswith(("\n", " ")):
            prompt_text += "\n"
    if tokenizer.eos_token is None:
        raise ValueError("KD requires a tokenizer with an EOS token")
    return prompt_text, str(response) + tokenizer.eos_token


class SparseTeacherStore:
    """Read-only memory-mapped top-K teacher distributions."""

    FILES = {
        "row_offsets": "row_offsets.npy",
        "token_ids": "token_ids.npy",
        "topk_ids": "topk_ids.npy",
        "topk_logprobs": "topk_logprobs.npy",
        "topk_counts": "topk_counts.npy",
        "topk_mass": "topk_mass.npy",
    }

    def __init__(self, root: str | Path, *, expected_topk: int, expected_tokenizer: str):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing KD manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if int(self.manifest.get("schema_version", -1)) != KD_SCHEMA_VERSION:
            raise ValueError(f"Unsupported KD schema in {manifest_path}")
        if int(self.manifest.get("topk", -1)) != int(expected_topk):
            raise ValueError(
                f"KD top-K mismatch: data={self.manifest.get('topk')} config={expected_topk}"
            )
        if self.manifest.get("tokenizer_fingerprint") != expected_tokenizer:
            raise ValueError("KD tokenizer fingerprint does not match the student tokenizer")
        self.arrays = {
            name: np.load(self.root / filename, mmap_mode="r", allow_pickle=False)
            for name, filename in self.FILES.items()
        }
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        rows = int(self.manifest["rows"])
        tokens = int(self.manifest["tokens"])
        topk = int(self.manifest["topk"])
        offsets = self.arrays["row_offsets"]
        if offsets.shape != (rows + 1,) or int(offsets[0]) != 0 or int(offsets[-1]) != tokens:
            raise ValueError(f"Malformed KD row offsets in {self.root}")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError(f"Non-monotonic KD row offsets in {self.root}")
        expected = {
            "token_ids": (tokens,),
            "topk_ids": (tokens, topk),
            "topk_logprobs": (tokens, topk),
            "topk_counts": (tokens,),
            "topk_mass": (tokens,),
        }
        for name, shape in expected.items():
            if self.arrays[name].shape != shape:
                raise ValueError(
                    f"Malformed KD array {name}: {self.arrays[name].shape} != {shape}"
                )
        counts = self.arrays["topk_counts"]
        if np.any(counts <= 0) or np.any(counts > topk):
            raise ValueError(f"Invalid KD top-K counts in {self.root}")

    def row(self, row_index: int) -> dict[str, np.ndarray]:
        offsets = self.arrays["row_offsets"]
        if row_index < 0 or row_index + 1 >= len(offsets):
            raise IndexError(f"KD row index {row_index} is out of range")
        start, end = int(offsets[row_index]), int(offsets[row_index + 1])
        return {name: array[start:end] for name, array in self.arrays.items() if name != "row_offsets"}
