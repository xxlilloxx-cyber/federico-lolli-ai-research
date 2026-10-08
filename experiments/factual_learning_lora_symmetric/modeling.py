"""Adapters for the isolated factual-learning experiment."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from src.adapters import LowRankAdapter, freeze_and_insert


class ParallelLoRASymmetricAdapter(nn.Module):
    """Frozen projection plus independent LoRA and Symmetric branches.

    Row-vector convention:
    base(x) + (alpha_l/r_l)(xA)B + (alpha_s/r_s)((xU)⊙(xU))P.
    """

    def __init__(self, base: nn.Module, lora_rank: int, symmetric_rank: int,
                 alpha_lora: float = 4.0, alpha_symmetric: float = 4.0) -> None:
        super().__init__()
        if lora_rank < 1 or symmetric_rank < 1:
            raise ValueError("branch ranks must be positive")
        self.base = base
        self.lora_rank = int(lora_rank)
        self.symmetric_rank = int(symmetric_rank)
        self.alpha_lora = float(alpha_lora)
        self.alpha_symmetric = float(alpha_symmetric)
        self.lora_multiplier = 1.0
        self.symmetric_multiplier = 1.0
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        d_in, d_out = base.weight.shape
        self.A = nn.Parameter(torch.empty(d_in, self.lora_rank))
        self.B = nn.Parameter(torch.zeros(self.lora_rank, d_out))
        self.U = nn.Parameter(torch.empty(d_in, self.symmetric_rank))
        self.P = nn.Parameter(torch.zeros(self.symmetric_rank, d_out))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))

    def branch_outputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x.float()
        lora = (self.alpha_lora / self.lora_rank) * ((z @ self.A) @ self.B)
        projected = z @ self.U
        symmetric = (self.alpha_symmetric / self.symmetric_rank) * ((projected * projected) @ self.P)
        return lora, symmetric

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora, symmetric = self.branch_outputs(x)
        correction = self.lora_multiplier * lora + self.symmetric_multiplier * symmetric
        return self.base(x) + correction.to(x.dtype)


def _projection(model: nn.Module, layer: int, projection: str) -> nn.Module:
    module = model.transformer.h[layer]
    for part in projection.split("."):
        module = getattr(module, part)
    return module


def _replace_projection(model: nn.Module, layer: int, projection: str, replacement: nn.Module) -> None:
    parent = model.transformer.h[layer]
    parts = projection.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def insert_adaptation(model: nn.Module, condition: str, *, lora_rank: int,
                      symmetric_rank: int, alpha_lora: float,
                      alpha_symmetric: float, layers: Iterable[int] = (0,),
                      projections: Iterable[str] = ("attn.c_proj",)) -> nn.Module:
    """Freeze the backbone and insert one of the four conditions."""
    condition = condition.lower()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers, projections = tuple(layers), tuple(projections)
    if condition == "frozen":
        return model
    if condition == "lora":
        return freeze_and_insert(model, "lora", lora_rank, alpha_lora,
                                 layers, projections, False)
    if condition == "symmetric":
        return freeze_and_insert(model, "symmetric_quadratic", symmetric_rank,
                                 alpha_symmetric, layers, projections, False)
    if condition != "combined":
        raise ValueError(f"unknown condition: {condition}")
    for layer in layers:
        for projection in projections:
            base = _projection(model, layer, projection)
            replacement = ParallelLoRASymmetricAdapter(
                base, lora_rank, symmetric_rank, alpha_lora, alpha_symmetric
            ).to(device=base.weight.device)
            _replace_projection(model, layer, projection, replacement)
    return model


@dataclass(frozen=True)
class ParameterBudget:
    frozen_backbone_parameters: int
    lora_parameters: int
    symmetric_parameters: int
    trainable_parameters: int

    @property
    def trainable_percentage(self) -> float:
        return 100.0 * self.trainable_parameters / self.frozen_backbone_parameters


def parameter_budget(model: nn.Module) -> ParameterBudget:
    lora = symmetric = 0
    for module in model.modules():
        if isinstance(module, ParallelLoRASymmetricAdapter):
            lora += module.A.numel() + module.B.numel()
            symmetric += module.U.numel() + module.P.numel()
        elif isinstance(module, LowRankAdapter) and module.kind == "lora":
            lora += module.U.numel() + module.V.numel()
        elif isinstance(module, LowRankAdapter) and module.kind == "symmetric_quadratic":
            symmetric += module.U.numel() + module.P.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return ParameterBudget(total - lora - symmetric, lora, symmetric, trainable)


def branch_norms(model: nn.Module) -> dict[str, float]:
    """Recompute factor, branch, and effective-update norms from current weights."""
    lora_a_sq = lora_b_sq = symmetric_u_sq = symmetric_p_sq = delta_w_sq = 0.0
    for module in model.modules():
        if isinstance(module, ParallelLoRASymmetricAdapter):
            a, b, u, p = (value.detach().float() for value in (module.A, module.B, module.U, module.P))
            lora_a_sq += float(a.square().sum()); lora_b_sq += float(b.square().sum())
            symmetric_u_sq += float(u.square().sum()); symmetric_p_sq += float(p.square().sum())
            delta_w_sq += float(((module.alpha_lora / module.lora_rank) * (a @ b)).square().sum())
        elif isinstance(module, LowRankAdapter) and module.kind == "lora":
            a, b = module.U.detach().float(), module.V.detach().float()
            lora_a_sq += float(a.square().sum()); lora_b_sq += float(b.square().sum())
            delta_w_sq += float(((module.alpha / module.rank) * (a @ b)).square().sum())
        elif isinstance(module, LowRankAdapter) and module.kind == "symmetric_quadratic":
            u, p = module.U.detach().float(), module.P.detach().float()
            symmetric_u_sq += float(u.square().sum()); symmetric_p_sq += float(p.square().sum())
    lora_sq = lora_a_sq + lora_b_sq
    symmetric_sq = symmetric_u_sq + symmetric_p_sq
    return {"lora_a_norm": math.sqrt(lora_a_sq), "lora_b_norm": math.sqrt(lora_b_sq),
            "lora_effective_update_norm": math.sqrt(delta_w_sq),
            "lora_parameter_norm": math.sqrt(lora_sq),
            "symmetric_u_norm": math.sqrt(symmetric_u_sq),
            "symmetric_p_norm": math.sqrt(symmetric_p_sq),
            "symmetric_parameter_norm": math.sqrt(symmetric_sq),
            "total_adaptation_norm": math.sqrt(lora_sq + symmetric_sq)}


def branch_parameter_snapshot(model: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    """Copy current trainable adapter tensors, separated by functional branch."""
    snapshot: dict[str, dict[str, torch.Tensor]] = {"lora": {}, "symmetric": {}}
    for module_name, module in model.named_modules():
        if isinstance(module, ParallelLoRASymmetricAdapter):
            for name, parameter in (("A", module.A), ("B", module.B)):
                snapshot["lora"][f"{module_name}.{name}"] = parameter.detach().float().cpu().clone()
            for name, parameter in (("U", module.U), ("P", module.P)):
                snapshot["symmetric"][f"{module_name}.{name}"] = parameter.detach().float().cpu().clone()
        elif isinstance(module, LowRankAdapter) and module.kind == "lora":
            snapshot["lora"][f"{module_name}.A"] = module.U.detach().float().cpu().clone()
            snapshot["lora"][f"{module_name}.B"] = module.V.detach().float().cpu().clone()
        elif isinstance(module, LowRankAdapter) and module.kind == "symmetric_quadratic":
            snapshot["symmetric"][f"{module_name}.U"] = module.U.detach().float().cpu().clone()
            snapshot["symmetric"][f"{module_name}.P"] = module.P.detach().float().cpu().clone()
    return snapshot


def branch_displacements(model: nn.Module, initial: dict[str, dict[str, torch.Tensor]]) -> dict[str, float]:
    """Measure current-minus-initial displacement for each branch."""
    current = branch_parameter_snapshot(model)
    result: dict[str, float] = {}
    total_sq = 0.0
    total_max = 0.0
    for branch in ("lora", "symmetric"):
        if set(current[branch]) != set(initial[branch]):
            raise ValueError(f"{branch} snapshot keys changed during training")
        squared = maximum = 0.0
        for key, value in current[branch].items():
            difference = value - initial[branch][key]
            squared += float(difference.square().sum())
            maximum = max(maximum, float(difference.abs().max()) if difference.numel() else 0.0)
        result[f"{branch}_parameter_displacement_l2"] = math.sqrt(squared)
        result[f"{branch}_parameter_displacement_max_abs"] = maximum
        total_sq += squared; total_max = max(total_max, maximum)
    result["total_parameter_displacement_l2"] = math.sqrt(total_sq)
    result["total_parameter_displacement_max_abs"] = total_max
    return result


def branch_gradient_norms(model: nn.Module) -> dict[str, float]:
    squared = {"lora_gradient_norm": 0.0, "symmetric_gradient_norm": 0.0}
    for module in model.modules():
        branches = ()
        if isinstance(module, ParallelLoRASymmetricAdapter):
            branches = (("lora_gradient_norm", (module.A, module.B)),
                        ("symmetric_gradient_norm", (module.U, module.P)))
        elif isinstance(module, LowRankAdapter) and module.kind == "lora":
            branches = (("lora_gradient_norm", (module.U, module.V)),)
        elif isinstance(module, LowRankAdapter) and module.kind == "symmetric_quadratic":
            branches = (("symmetric_gradient_norm", (module.U, module.P)),)
        for key, parameters in branches:
            squared[key] += sum(float(parameter.grad.detach().float().square().sum())
                                for parameter in parameters if parameter.requires_grad and parameter.grad is not None)
    return {key: math.sqrt(value) for key, value in squared.items()}
