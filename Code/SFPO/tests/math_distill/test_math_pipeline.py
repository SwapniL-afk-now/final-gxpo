from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def load_tool(name):
    path = ROOT / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numina_materializer_uses_stored_generation_and_deduplicates():
    module = load_tool("prepare_math_distillation.py")
    frame = pd.DataFrame([
        {
            "problem": "1+1?", "solution": "not the selected field",
            "generation": "<think>one plus one</think>\\boxed{2}",
            "messages": [{"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "other"}],
            "model_name": module.TEACHER_MODEL,
        },
        {
            "problem": "1+1?", "solution": "duplicate",
            "generation": "duplicate trace", "messages": [{"role": "user", "content": "1+1?"}],
            "model_name": module.TEACHER_MODEL,
        },
        {
            "problem": "2+2?", "solution": "unused",
            "generation": "stored second trace", "messages": [{"role": "user", "content": "2+2?"}],
            "model_name": module.TEACHER_MODEL,
        },
    ])
    rows = module.materialize_rows(frame)
    assert len(rows) == 2
    assert rows.iloc[0]["response"] == "<think>one plus one</think>\\boxed{2}"
    assert rows.iloc[0]["source"] == "numina_r1_distill_teacher_trace"
    assert "generate(" not in (ROOT / "tools" / "prepare_math_distillation.py").read_text()


def test_split_is_seeded_and_disjoint():
    module = load_tool("prepare_math_distillation.py")
    frame = pd.DataFrame({
        "problem_hash": [str(i) for i in range(8)],
        "problem_id": [str(i) for i in range(8)],
    })
    first_train, first_val = module.split_rows(frame, 5, 2, 123)
    second_train, second_val = module.split_rows(frame, 5, 2, 123)
    assert first_train["problem_hash"].tolist() == second_train["problem_hash"].tolist()
    assert first_val["problem_hash"].tolist() == second_val["problem_hash"].tolist()
    assert set(first_train["problem_hash"]).isdisjoint(first_val["problem_hash"])


def test_math_evaluator_loads_only_requested_benchmarks_and_greedy_config(tmp_path):
    module = load_tool("evaluate_math_distillation.py")
    assert module.BENCHMARKS == ("math500", "aime24", "aime25")
    for name in ("math500", "aime2024", "aime2025"):
        (tmp_path / f"{name}.parquet").touch()
    files = module.resolve_benchmark_files(tmp_path)
    assert set(files) == set(module.BENCHMARKS)
    assert "temperature=0.0" in (ROOT / "tools" / "evaluate_math_distillation.py").read_text()
    assert "n=1" in (ROOT / "tools" / "evaluate_math_distillation.py").read_text()


def test_math_answer_equivalence_supports_boxed_fraction_integer_and_symbolic():
    from verl.utils.reward_score import math_verify

    assert math_verify.compute_score(r"\boxed{1/2}", r"\frac{1}{2}") == 1.0
    assert math_verify.compute_score(r"The answer is \boxed{204}", "204") == 1.0
    assert math_verify.compute_score(r"\boxed{x+1}", "1+x") == 1.0


def test_math_launchers_are_self_contained_and_use_math_assets():
    for name in ("run_math_distill_sft_adamw_fsdp.sh", "run_math_distill_gxpo_sft_fsdp.sh"):
        source = (ROOT / "train-scripts" / name).read_text()
        assert "math_distill_r1_7b" in source
        assert "Qwen/Qwen2.5-1.5B-Instruct" in source
        assert "math_distill" in source
        assert "math500.parquet" in source
        assert "aime2024.parquet" in source
        assert "aime2025.parquet" in source
        assert "temperature=0 top_p=1" in source
    gxpo = (ROOT / "train-scripts" / "run_math_distill_gxpo_sft_fsdp.sh").read_text()
    assert "+optim.use_gxpo=True" in gxpo
    assert "gxpo_optimizer_state_mode" in gxpo
