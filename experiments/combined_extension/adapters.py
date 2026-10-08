"""Adapters for the Combined extension without changing historical implementations."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from src.adapters import LowRankAdapter


class CombinedAdapter(nn.Module):
    """Frozen projection with independent linear and symmetric branches.

    ``total_rank``: base + alpha/(r_l+r_s) * (xAB + (xU)^2 P)
    ``branch_wise``: base + alpha/r_l*xAB + alpha/r_s*(xU)^2P
    """

    MODES = {"total_rank", "branch_wise"}

    def __init__(self, base: nn.Module, rank_lora: int, rank_symmetric: int,
                 alpha: float = 4.0, scaling_mode: str = "total_rank") -> None:
        super().__init__()
        if rank_lora < 1 or rank_symmetric < 1:
            raise ValueError("both Combined ranks must be positive integers")
        if scaling_mode not in self.MODES:
            raise ValueError(f"scaling_mode must be one of {sorted(self.MODES)}")
        self.base = base
        self.rank_lora, self.rank_symmetric = int(rank_lora), int(rank_symmetric)
        self.alpha, self.scaling_mode = float(alpha), scaling_mode
        self.lora_enabled = True; self.symmetric_enabled = True
        for parameter in base.parameters(): parameter.requires_grad_(False)
        d_in, d_out = base.weight.shape
        self.A = nn.Parameter(torch.empty(d_in, rank_lora))
        self.B = nn.Parameter(torch.zeros(rank_lora, d_out))
        self.U = nn.Parameter(torch.empty(d_in, rank_symmetric))
        self.P = nn.Parameter(torch.zeros(rank_symmetric, d_out))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))

    @property
    def scales(self) -> tuple[float, float]:
        if self.scaling_mode == "total_rank":
            scale = self.alpha / (self.rank_lora + self.rank_symmetric)
            return scale, scale
        return self.alpha / self.rank_lora, self.alpha / self.rank_symmetric

    def branch_outputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x.float(); lora_scale, symmetric_scale = self.scales
        linear = lora_scale * ((z @ self.A) @ self.B)
        projected = z @ self.U
        symmetric = symmetric_scale * ((projected * projected) @ self.P)
        return linear.to(x.dtype), symmetric.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear, symmetric = self.branch_outputs(x)
        if not self.lora_enabled: linear = torch.zeros_like(linear)
        if not self.symmetric_enabled: symmetric = torch.zeros_like(symmetric)
        return self.base(x) + linear + symmetric


def _target(model: nn.Module, layer: int, projection: str):
    parent = model.transformer.h[layer]; parts = projection.split(".")
    for part in parts[:-1]: parent = getattr(parent, part)
    return parent, parts[-1]


def insert_adapter(model: nn.Module, method: str, *, rank: int | None = None,
                   rank_lora: int | None = None, rank_symmetric: int | None = None,
                   alpha: float = 4.0, scaling_mode: str = "total_rank",
                   layers: Iterable[int] = (0,), projections: Iterable[str] = ("attn.c_proj",)) -> nn.Module:
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for layer in layers:
        for projection in projections:
            parent, leaf = _target(model, int(layer), projection); base = getattr(parent, leaf)
            if method == "lora": replacement = LowRankAdapter(base, int(rank), alpha, "lora", True)
            elif method == "symmetric": replacement = LowRankAdapter(base, int(rank), alpha, "symmetric_quadratic", True)
            elif method == "combined":
                replacement = CombinedAdapter(base, int(rank_lora), int(rank_symmetric), alpha, scaling_mode)
            else: raise ValueError(f"unknown method: {method}")
            setattr(parent, leaf, replacement.to(device=base.weight.device))
    return model


@dataclass(frozen=True)
class Counts:
    lora: int; symmetric: int; total_trainable: int; total_model: int


def parameter_counts(model: nn.Module) -> Counts:
    lora = symmetric = 0
    for module in model.modules():
        if isinstance(module, CombinedAdapter):
            lora += module.A.numel() + module.B.numel(); symmetric += module.U.numel() + module.P.numel()
        elif isinstance(module, LowRankAdapter) and module.kind == "lora":
            lora += module.U.numel() + module.V.numel()
        elif isinstance(module, LowRankAdapter) and module.kind == "symmetric_quadratic":
            symmetric += module.U.numel() + module.P.numel()
    return Counts(lora, symmetric, sum(p.numel() for p in model.parameters() if p.requires_grad),
                  sum(p.numel() for p in model.parameters()))


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.named_parameters() if value.requires_grad}
