"""CPU tests for offline sparse token-distribution distillation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from verl.trainer.kd_utils import sparse_topk_kd_loss
from verl.utils.dataset.kd_utils import (
    KD_SCHEMA_VERSION, SparseTeacherStore, tokenizer_fingerprint,
)
from verl.utils.dataset.sft_dataset import SFTDataset


ROOT = Path(__file__).resolve().parents[2]


def load_scorer():
    path = ROOT / "tools" / "score_code_teacher_distributions.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sparse_kd_matches_dense_reference_when_topk_covers_vocab():
    student = torch.tensor([[[1.2, -0.4, 0.7, 2.0]]], requires_grad=True)
    teacher_logits = torch.tensor([0.1, 1.5, -0.2, 0.8])
    teacher_logprobs = torch.log_softmax(teacher_logits, dim=-1).view(1, 1, 4)
    teacher_ids = torch.arange(4).view(1, 1, 4)
    counts = torch.tensor([[4]], dtype=torch.uint8)
    mask = torch.tensor([[True]])
    temperature = 2.0

    total, count, per_token = sparse_topk_kd_loss(
        student, teacher_ids, teacher_logprobs, counts, mask, temperature
    )
    teacher_probs = torch.softmax(teacher_logprobs / temperature, dim=-1)
    expected = -(
        teacher_probs * torch.log_softmax(student / temperature, dim=-1)
    ).sum() * temperature ** 2
    assert count.item() == 1
    assert torch.allclose(total, expected, atol=2e-5, rtol=2e-5)
    assert torch.allclose(per_token.sum(), expected, atol=2e-5, rtol=2e-5)
    total.backward()
    assert torch.isfinite(student.grad).all()


def test_teacher_topk_extraction_sorts_and_records_probability_mass():
    module = load_scorer()

    class Value:
        def __init__(self, logprob):
            self.logprob = logprob

    ids, logprobs, count, mass = module.extract_topk(
        {7: Value(-2.5), 3: Value(-0.8), 9: Value(-1.2)}, 2
    )
    assert ids.tolist() == [3, 9]
    assert count == 2
    assert np.allclose(logprobs.astype(np.float32), [-0.8, -1.2], atol=5e-4)
    assert np.isclose(mass, np.exp(-0.8) + np.exp(-1.2))


def test_sparse_teacher_store_validates_and_memory_maps(tmp_path):
    topk = 2
    arrays = {
        "row_offsets": np.asarray([0, 2], dtype=np.int64),
        "token_ids": np.asarray([4, 5], dtype=np.int32),
        "topk_ids": np.asarray([[4, 1], [5, 2]], dtype=np.int32),
        "topk_logprobs": np.asarray([[-0.1, -2.0], [-0.2, -1.5]], dtype=np.float16),
        "topk_counts": np.asarray([2, 2], dtype=np.uint8),
        "topk_mass": np.asarray([0.95, 0.9], dtype=np.float16),
    }
    for name, array in arrays.items():
        np.save(tmp_path / f"{name}.npy", array, allow_pickle=False)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": KD_SCHEMA_VERSION,
        "topk": topk,
        "rows": 1,
        "tokens": 2,
        "tokenizer_fingerprint": "tokenizer-hash",
    }))
    store = SparseTeacherStore(
        tmp_path, expected_topk=topk, expected_tokenizer="tokenizer-hash"
    )
    row = store.row(0)
    assert row["token_ids"].tolist() == [4, 5]
    assert isinstance(store.arrays["topk_ids"], np.memmap)


def test_launchers_enable_kd_and_keep_both_gxpo_modes_configurable():
    adamw = (ROOT / "train-scripts" / "run_code_distill_sft_adamw_fsdp.sh").read_text()
    gxpo = (ROOT / "train-scripts" / "run_code_distill_gxpo_sft_fsdp.sh").read_text()
    teacher = (ROOT / "train-scripts" / "run_code_teacher_generation.sh").read_text()
    for source in (adamw, gxpo):
        assert "teacher_kd_train.parquet" in source
        assert 'KD_TOPK="${KD_TOPK:-32}"' in source
        assert 'KD_WEIGHT="${KD_WEIGHT:-0.5}"' in source
        assert 'KD_TEMPERATURE="${KD_TEMPERATURE:-2.0}"' in source
    assert 'GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"' in gxpo
    assert '+optim.gxpo_optimizer_state_mode="$GXPO_OPTIMIZER_STATE_MODE"' in gxpo
    assert "score_code_teacher_distributions.py" in teacher
    assert 'KD_SCORE_ONLY:-0' in teacher


def test_shared_loss_precedes_optimizer_state_mode_branch():
    source = (ROOT / "verl" / "trainer" / "fsdp_sft_trainer.py").read_text()
    assert "sparse_topk_kd_loss(" in source
    assert "self._compute_loss_and_backward" in source
    assert 'self.gxpo_optimizer_state_mode == "transactional"' in source
    assert source.index("def _compute_loss_and_backward") < source.index("def _gxpo_training_step")


class TinyTokenizer:
    chat_template = None
    eos_token = "~"
    eos_token_id = 9
    pad_token_id = 0
    special_tokens_map = {"eos_token": "~", "pad_token": "_"}

    def get_vocab(self):
        return {"_": 0, "P": 1, "\n": 2, "R": 3, "~": 9}

    def __call__(self, text, return_tensors=None, add_special_tokens=False, **_kwargs):
        ids = [self.get_vocab()[character] for character in text]
        if return_tensors == "pt":
            values = torch.tensor([ids], dtype=torch.long)
            return {"input_ids": values, "attention_mask": torch.ones_like(values)}
        return {"input_ids": ids}


def write_tiny_kd_dataset(tmp_path, token_ids):
    tokenizer = TinyTokenizer()
    fingerprint = tokenizer_fingerprint(tokenizer)
    sidecar = tmp_path / "teacher_kd_train.sidecar"
    sidecar.mkdir()
    topk = 2
    arrays = {
        "row_offsets": np.asarray([0, len(token_ids)], dtype=np.int64),
        "token_ids": np.asarray(token_ids, dtype=np.int32),
        "topk_ids": np.tile(np.asarray([[3, 9]], dtype=np.int32), (len(token_ids), 1)),
        "topk_logprobs": np.tile(np.asarray([[-0.2, -2.0]], dtype=np.float16), (len(token_ids), 1)),
        "topk_counts": np.full(len(token_ids), 2, dtype=np.uint8),
        "topk_mass": np.full(len(token_ids), 0.95, dtype=np.float16),
    }
    for name, array in arrays.items():
        np.save(sidecar / f"{name}.npy", array, allow_pickle=False)
    (sidecar / "manifest.json").write_text(json.dumps({
        "schema_version": KD_SCHEMA_VERSION,
        "topk": topk,
        "rows": 1,
        "tokens": len(token_ids),
        "tokenizer_fingerprint": fingerprint,
    }))
    import pandas as pd
    parquet = tmp_path / "teacher_kd_train.parquet"
    pd.DataFrame([{
        "prompt": "P", "response": "R",
        "source": "taco_dataset_solution_fallback",
        "kd_row_index": 0, "kd_token_count": len(token_ids),
        "kd_sidecar": sidecar.name,
        "teacher_tokenizer_fingerprint": fingerprint,
        "teacher_topk": topk,
    }]).to_parquet(parquet, index=False)
    return tokenizer, parquet


def test_sft_dataset_aligns_kd_to_shifted_response_positions(tmp_path):
    tokenizer, parquet = write_tiny_kd_dataset(tmp_path, [3, 9])
    dataset = SFTDataset(
        parquet_files=str(parquet), tokenizer=tokenizer, max_length=8,
        kd_enabled=True, kd_topk=2,
    )
    row = dataset[0]
    positions = row["kd_token_mask"].nonzero().flatten().tolist()
    assert positions == [1, 2]
    assert row["kd_teacher_ids"][positions].tolist() == [[3, 9], [3, 9]]
    assert row["kd_source_id"].item() == 1


def test_sft_dataset_fails_closed_on_tokenizer_alignment_error(tmp_path):
    tokenizer, parquet = write_tiny_kd_dataset(tmp_path, [9, 3])
    dataset = SFTDataset(
        parquet_files=str(parquet), tokenizer=tokenizer, max_length=8,
        kd_enabled=True, kd_topk=2,
    )
    import pytest
    with pytest.raises(ValueError, match="response token IDs"):
        dataset[0]



def test_offline_scorer_covers_teacher_and_fallback_rows(tmp_path):
    module = load_scorer()
    tokenizer = TinyTokenizer()
    fingerprint = tokenizer_fingerprint(tokenizer)
    import pandas as pd
    input_path = tmp_path / "teacher_sft_train.parquet"
    pd.DataFrame([
        {"prompt": "P", "response": "R", "source": "taco_verified_teacher"},
        {"prompt": "P", "response": "R", "source": "taco_dataset_solution_fallback"},
    ]).to_parquet(input_path, index=False)

    class Logprob:
        def __init__(self, value):
            self.logprob = value

    class Output:
        def __init__(self, ids):
            self.prompt_token_ids = ids
            self.prompt_logprobs = [
                None,
                {1: Logprob(-0.8), 3: Logprob(-1.2)},
                {3: Logprob(-0.4), 9: Logprob(-1.5)},
                {9: Logprob(-0.3), 3: Logprob(-1.7)},
            ]

    class FakeLLM:
        def generate(self, prompts, _sampling, use_tqdm=False):
            assert use_tqdm is False
            return [Output(prompt["prompt_token_ids"]) for prompt in prompts]

    output_path = tmp_path / "teacher_kd_train.parquet"
    module.score_split(
        llm=FakeLLM(), sampling=object(), tokenizer=tokenizer,
        tokenizer_hash=fingerprint, teacher_model="teacher",
        input_path=input_path, output_path=output_path,
        topk=2, max_length=8, request_batch_size=1, overwrite=False,
    )
    scored = pd.read_parquet(output_path)
    assert scored["source"].tolist() == [
        "taco_verified_teacher", "taco_dataset_solution_fallback",
    ]
    store = SparseTeacherStore(
        output_path.with_suffix(".sidecar"), expected_topk=2,
        expected_tokenizer=fingerprint,
    )
    assert store.manifest["source_counts"] == {
        "taco_verified_teacher": 1,
        "taco_dataset_solution_fallback": 1,
    }
    assert store.manifest["tokens"] == 4
