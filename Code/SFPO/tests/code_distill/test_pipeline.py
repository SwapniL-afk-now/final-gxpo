"""CPU tests for the coding-distillation data and FSDP wiring."""

from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[2]


def load_tool(name):
    path = ROOT / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_hash_is_stable_and_ignores_code_instruction():
    module = load_tool("prepare_code_distillation.py")
    prompt = module.make_prompt("Implement binary search.")
    assert module.canonical_prompt(prompt) == "You are an expert competitive programmer. Solve the problem in Python and ensure the program reads stdin and writes stdout. Implement binary search."
    assert module.stable_id(module.canonical_prompt(prompt)) == module.stable_id(
        module.canonical_prompt(prompt)
    )


def test_teacher_trajectory_limit_and_split_support_are_visible():
    source = (ROOT / "tools" / "generate_code_teacher_trajectories.py").read_text()
    assert 'choices=("all", "train", "validation")' in source
    assert "len(records) >= 2" in source
    assert "continuous=False" in source


def test_code_evaluator_reports_required_benchmarks():
    module = load_tool("evaluate_code_distillation.py")
    assert module.BENCHMARKS == ("humanevalplus", "mbppplus", "livecodebench")


def test_code_evaluator_parallel_verification_is_bounded_and_configurable():
    module = load_tool("evaluate_code_distillation.py")
    assert module._resolve_verifier_workers(7) == 7
    assert "ProcessPoolExecutor" in (ROOT / "tools" / "evaluate_code_distillation.py").read_text()
    assert "Verifying generated code" in (ROOT / "tools" / "evaluate_code_distillation.py").read_text()


def test_fsdpsft_uses_fp32_parameter_config_and_code_eval():
    source = (ROOT / "verl" / "trainer" / "fsdp_sft_trainer.py").read_text()
    assert "model_dtype" in source
    assert "evaluate_code_distillation.py" in source
    assert "finite_on_all_ranks" in source
    assert "save_resumable_checkpoint" in source


def test_resumable_manager_converts_generic_optimizer_state_through_fsdp():
    source = (ROOT / "verl" / "utils" / "checkpoint" / "fsdp_checkpoint_manager.py").read_text()
    assert source.count("FSDP.optim_state_dict_to_load(") == 1
    assert source.count("FSDP.optim_state_dict(self.model, self.optimizer)") == 1


def test_launchers_use_configurable_fsdp_and_shared_decoding():
    launchers = [
        ROOT / "train-scripts" / "run_code_distill_sft_adamw_fsdp.sh",
        ROOT / "train-scripts" / "run_code_distill_gxpo_sft_fsdp.sh",
        ROOT / "train-scripts" / "run_code_distill_rl_fsdp.sh",
    ]
    for launcher in launchers:
        source = launcher.read_text()
        assert "CUDA_VISIBLE_DEVICES" in source
        assert "GPU_COUNT" in source or "fsdp_config.fsdp_size=4" in source
        assert "max_response_length=3072" in source or "MAX_RESPONSE_LENGTH" in source
    for launcher in (launchers[2],):
        source = launcher.read_text()
        assert "temperature=0.7" in source
        assert "top_p=1.0" in source
