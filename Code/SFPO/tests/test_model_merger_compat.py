"""Regression tests for checkpoint finalization with the installed Transformers API."""

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODEL_MERGER = REPO / 'scripts' / 'model_merger.py'


def test_model_merger_imports_without_optional_vision_auto_model():
    """Causal-LM checkpoint merging must not require the vision auto-model class."""
    spec = importlib.util.spec_from_file_location('production_model_merger', MODEL_MERGER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
