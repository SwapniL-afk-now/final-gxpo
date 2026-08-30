"""Focused tests for the classic-FSDP Muon adapter."""

import ast
import os
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from verl.workers.muon import (FSDPMuonParamInfo, build_fsdp_muon_registry,
                               build_muon, zeropower_via_newtonschulz5)

REPO = Path(__file__).resolve().parents[2]


def config(**overrides):
    values = {
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "betas": (0.9, 0.999),
        "muon_momentum": 0.95,
        "muon_ns_steps": 5,
        "muon_nesterov": True,
        "muon_distributed_backend": "gather_scatter",
    }
    values.update(overrides)
    return OmegaConf.create(values)


class ClassificationToy(nn.Module):

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(13, 8)
        self.block = nn.Linear(8, 16)
        self.proj = nn.Linear(16, 8)
        self.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 13, bias=False)

    def forward(self, ids):
        return self.lm_head(self.norm(self.proj(self.block(self.embed_tokens(ids)))))


class TestMuon(unittest.TestCase):

    def test_newton_schulz_rectangular_shapes_are_finite(self):
        torch.manual_seed(0)
        for shape in ((64, 32), (32, 64)):
            result = zeropower_via_newtonschulz5(torch.randn(*shape), steps=5)
            self.assertEqual(result.shape, shape)
            self.assertTrue(torch.isfinite(result).all())

    def test_registry_ignores_alignment_padding_and_records_offsets(self):
        import torch.distributed as dist
        from types import SimpleNamespace

        if dist.is_initialized():
            self.skipTest("test requires an uninitialized process group")
        fd, init_path = tempfile.mkstemp(prefix="muon-gloo-")
        os.close(fd)
        os.unlink(init_path)
        try:
            dist.init_process_group("gloo", rank=0, world_size=1,
                                    init_method=f"file://{init_path}")
            first = nn.Parameter(torch.zeros(4))
            second = nn.Parameter(torch.zeros(6))
            flat = SimpleNamespace(
                _params=[first, second],
                _fqns=["first", "second"],
                _shapes=[(2, 2), (2, 3)],
                _contiguities=[True, True],
                _numels_with_padding=[4, 2, 6],
                _is_padding_mask=[False, True, False],
                _shard_param_infos=[
                    SimpleNamespace(in_shard=True, intra_param_start_idx=0,
                                   numel_in_shard=4),
                    SimpleNamespace(in_shard=True, intra_param_start_idx=0,
                                   numel_in_shard=6),
                ],
            )
            handle = SimpleNamespace(flat_param=flat,
                                     process_group=dist.group.WORLD)
            registry = build_fsdp_muon_registry(SimpleNamespace(_all_handles=[handle]))
            self.assertEqual(registry[first].global_offset, 0)
            self.assertEqual(registry[second].global_offset, 6)
            self.assertEqual(registry[second].global_numel, 6)
        finally:
            dist.destroy_process_group()
            if os.path.exists(init_path):
                os.unlink(init_path)

    def test_classification_excludes_embeddings_heads_biases_and_norms(self):
        model = ClassificationToy()
        optimizer = build_muon(model, config())
        flags = {name: optimizer.state[param]["use_muon"]
                 for name, param in model.named_parameters()}

        self.assertIs(flags["block.weight"], True)
        self.assertIs(flags["proj.weight"], True)
        for name in ("embed_tokens.weight", "block.bias", "proj.bias",
                     "norm.weight", "norm.bias", "lm_head.weight"):
            self.assertIs(flags[name], False)
        self.assertEqual(optimizer.muon_parameter_count, 2)
        self.assertEqual(optimizer.adamw_parameter_count, 6)
        self.assertEqual(len({id(param) for group in optimizer.param_groups for param in group["params"]}), 8)
        self.assertEqual(len(optimizer.parameter_signature), 64)

    def test_noncontiguous_matrix_is_excluded_but_valid_matrix_uses_muon(self):
        model = nn.Module()
        model.valid = nn.Parameter(torch.randn(8, 8))
        model.noncontiguous = nn.Parameter(torch.randn(8, 8).t())
        optimizer = build_muon(model, config())
        self.assertIs(optimizer.state[model.valid]["use_muon"], True)
        self.assertIs(optimizer.state[model.noncontiguous]["use_muon"], False)

    def test_muon_step_moves_parameters_and_reports_diagnostics(self):
        torch.manual_seed(1)
        model = ClassificationToy()
        optimizer = build_muon(model, config())
        before = model.block.weight.detach().clone()
        loss = model(torch.tensor([[1, 2, 3]])).square().mean()
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.equal(before, model.block.weight))
        diagnostics = optimizer.diagnostics()
        self.assertEqual(diagnostics["optimizer/muon_backend_active"], 0.0)
        self.assertEqual(diagnostics["optimizer/muon_parameter_count"], 2.0)
        self.assertGreater(diagnostics["optimizer/muon_update_norm"], 0.0)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for key, value in diagnostics.items()
                            if key.startswith("optimizer/muon_") and key != "optimizer/muon_backend_active"))

    def test_invalid_all_adamw_model_fails_closed(self):
        model = nn.Module()
        model.vector = nn.Parameter(torch.randn(32))
        with self.assertRaisesRegex(RuntimeError, "no valid 2-D Muon parameters"):
            build_muon(model, config())

    def test_unsupported_distributed_backend_fails_closed(self):
        model = ClassificationToy()
        with self.assertRaisesRegex(ValueError, "unsupported Muon distributed backend"):
            build_muon(model, config(muon_distributed_backend="flex_shard"))

    def test_actor_has_one_authoritative_optimizer_step_boundary(self):
        source = (REPO / "verl" / "workers" / "actor" / "dp_actor.py").read_text()
        tree = ast.parse(source)
        direct_steps = [node for node in ast.walk(tree)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "step"
                        and isinstance(node.func.value, ast.Attribute)
                        and node.func.value.attr == "actor_optimizer"]
        self.assertEqual(len(direct_steps), 1)
        self.assertIn("already_clipped=False", source)
        self.assertIn("self._optimizer_step(already_clipped=True)", source)


if __name__ == "__main__":
    unittest.main()
