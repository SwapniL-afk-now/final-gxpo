from pathlib import Path


ROOT = Path(__file__).parents[3]


def test_ordinary_sft_has_no_offline_kd_wiring():
    trainer = (ROOT / "verl/trainer/fsdp_sft_trainer.py").read_text()
    dataset = (ROOT / "verl/utils/dataset/sft_dataset.py").read_text()
    for source in (trainer, dataset):
        assert "teacher_topk" not in source
        assert "use_kd" not in source
        assert "response_ids_key" not in source


def test_offline_kd_has_dedicated_modules():
    assert (ROOT / "verl/trainer/kd_sft_trainer.py").is_file()
    assert (ROOT / "verl/trainer/kd_sft_loss.py").is_file()
    assert (ROOT / "verl/utils/dataset/kd_sft_dataset.py").is_file()
    trainer = (ROOT / "verl/trainer/kd_sft_trainer.py").read_text()
    assert "verl.trainer.kd_sft_loss" in trainer
    assert "verl.workers.actor.kd_loss" not in trainer
