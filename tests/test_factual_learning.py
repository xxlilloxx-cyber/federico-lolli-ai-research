from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "factual_learning_lora_symmetric"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT))

from dataset import examples_for, generate_facts
from modeling import (ParallelLoRASymmetricAdapter, branch_displacements,
                      branch_gradient_norms, branch_norms, branch_parameter_snapshot,
                      insert_adaptation, parameter_budget)
from src.adapters import LowRankAdapter

spec = importlib.util.spec_from_file_location("factual_runner", EXP / "run_experiment.py")
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class ConvLike(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__(); self.weight = nn.Parameter(torch.randn(d, d)); self.bias = nn.Parameter(torch.randn(d))

    def forward(self, x): return x @ self.weight + self.bias


class FakeModel(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        block = nn.Module(); block.attn = nn.Module(); block.attn.c_proj = ConvLike(d)
        self.transformer = nn.Module(); self.transformer.h = nn.ModuleList([block])


def adapted(condition: str, rank: int = 2):
    torch.manual_seed(7); model = FakeModel()
    return insert_adaptation(model, condition, lora_rank=rank, symmetric_rank=rank,
                             alpha_lora=4, alpha_symmetric=4)


def train_twice(model):
    params = [p for p in model.parameters() if p.requires_grad]; optimizer = torch.optim.SGD(params, lr=.1)
    for _ in range(2):
        optimizer.zero_grad(); model.transformer.h[0].attn.c_proj(torch.randn(3, 8)).square().mean().backward(); optimizer.step()
    optimizer.zero_grad(); model.transformer.h[0].attn.c_proj(torch.randn(3, 8)).square().mean().backward()


def test_backbone_frozen_and_output_shapes():
    for condition in ("lora", "symmetric", "combined"):
        model = adapted(condition); module = model.transformer.h[0].attn.c_proj
        assert module(torch.randn(2, 5, 8)).shape == (2, 5, 8)
        assert all(not p.requires_grad for p in module.base.parameters())


def test_all_requested_branches_receive_gradients():
    for condition in ("lora", "symmetric", "combined"):
        model = adapted(condition); train_twice(model); gradients = branch_gradient_norms(model)
        if condition in ("lora", "combined"): assert gradients["lora_gradient_norm"] > 0
        if condition in ("symmetric", "combined"): assert gradients["symmetric_gradient_norm"] > 0


def test_combined_branch_zeroing_reproduces_single_branches_and_base():
    torch.manual_seed(11); base = ConvLike(); combined = ParallelLoRASymmetricAdapter(base, 2, 2, 4, 4)
    with torch.no_grad(): combined.B.normal_(); combined.P.normal_()
    lora = LowRankAdapter(base, 2, 4, "lora"); symmetric = LowRankAdapter(base, 2, 4, "symmetric_quadratic")
    with torch.no_grad():
        lora.U.copy_(combined.A); lora.V.copy_(combined.B)
        symmetric.U.copy_(combined.U); symmetric.P.copy_(combined.P)
    x = torch.randn(4, 8)
    combined.symmetric_multiplier = 0; assert torch.allclose(combined(x), lora(x), atol=1e-6)
    combined.symmetric_multiplier = 1; combined.lora_multiplier = 0
    assert torch.allclose(combined(x), symmetric(x), atol=1e-6)
    combined.symmetric_multiplier = 0
    assert torch.allclose(combined(x), base(x), atol=1e-6)


def test_parameter_count_reporting():
    assert parameter_budget(adapted("lora", 2)).lora_parameters == 32
    assert parameter_budget(adapted("symmetric", 2)).symmetric_parameters == 32
    budget = parameter_budget(adapted("combined", 2))
    assert (budget.lora_parameters, budget.symmetric_parameters, budget.trainable_parameters) == (32, 32, 64)


def test_norms_and_displacements_use_current_parameters():
    model = adapted("combined")
    initial = branch_parameter_snapshot(model)
    before = branch_norms(model)
    assert branch_displacements(model, initial)["total_parameter_displacement_l2"] == 0
    train_twice(model)
    after = branch_norms(model); displacement = branch_displacements(model, initial)
    assert after["lora_b_norm"] > 0 and after["symmetric_p_norm"] > 0
    assert after["lora_effective_update_norm"] > 0
    assert displacement["lora_parameter_displacement_l2"] > 0
    assert displacement["symmetric_parameter_displacement_l2"] > 0
    assert displacement["total_parameter_displacement_max_abs"] > 0
    assert after != before


def test_dataset_and_initialization_are_reproducible():
    assert generate_facts(42, 3) == generate_facts(42, 3)
    assert examples_for(generate_facts(42, 1), "paraphrase") == examples_for(generate_facts(42, 1), "paraphrase")
    first, second = adapted("combined"), adapted("combined")
    for a, b in zip(first.parameters(), second.parameters()): assert torch.equal(a, b)


def test_saved_metrics_reload(tmp_path):
    rows = [{key: 0 for key in runner.METRIC_FIELDS}]
    path = tmp_path / "metrics.csv"; runner.write_metrics(path, rows)
    with path.open() as handle: loaded = list(csv.DictReader(handle))
    assert len(loaded) == 1 and set(loaded[0]) == set(runner.METRIC_FIELDS)
