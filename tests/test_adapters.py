import torch
from torch import nn
from src.adapters import LowRankAdapter, parameter_counts


class ConvLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 12))
        self.bias = nn.Parameter(torch.randn(12))

    def forward(self, x):
        return x @ self.weight + self.bias


def test_identity_and_gradients():
    x = torch.randn(2, 3, 8)
    base = ConvLike()
    expected = base(x).detach()
    model = LowRankAdapter(base, rank=4, alpha=4, kind="quadratic")
    assert torch.allclose(model(x), expected, atol=1e-6)
    model(x).square().mean().backward()
    assert model.V.grad is not None and torch.isfinite(model.V.grad).all()
    assert base.weight.grad is None and base.bias.grad is None
    assert parameter_counts(model)["trainable_parameters"] == 4 * (8 + 12)


def test_equal_parameter_budget():
    lora = LowRankAdapter(ConvLike(), rank=4, kind="lora")
    quadratic = LowRankAdapter(ConvLike(), rank=4, kind="quadratic")
    assert parameter_counts(lora)["trainable_parameters"] == parameter_counts(quadratic)["trainable_parameters"]


def test_signed_and_feature_identity_and_gradients():
    x = torch.randn(2, 3, 8)
    for kind in ("signed_quadratic", "feature_interaction"):
        base = ConvLike()
        expected = base(x).detach()
        model = LowRankAdapter(base, rank=3, kind=kind)
        assert torch.allclose(model(x), expected, atol=1e-6)
        model(x).square().mean().backward()
        assert all(p.grad is None for p in base.parameters())
        assert any(p.grad is not None for n, p in model.named_parameters() if n != "base.weight" and n != "base.bias")


def test_new_adapter_parameter_budgets_and_identity():
    x = torch.randn(2, 3, 8)
    base = ConvLike()
    symmetric = LowRankAdapter(base, rank=4, kind="symmetric_quadratic")
    assert parameter_counts(symmetric)["trainable_parameters"] == 4 * (8 + 12)
    assert torch.allclose(symmetric(x), base(x), atol=1e-6)
    symmetric(x).sum().backward()
    assert symmetric.P.grad is not None and torch.isfinite(symmetric.U.grad).all()
    assert symmetric.U.grad.norm().item() == 0.0

    base2 = ConvLike()
    mixed = LowRankAdapter(base2, kind="linear_quadratic", rank_l=1, rank_q=2)
    assert parameter_counts(mixed)["trainable_parameters"] == 1 * (8 + 12) + 2 * (8 + 8 + 12)
    assert torch.allclose(mixed(x), base2(x), atol=1e-6)
    mixed(x).square().mean().backward()
    assert mixed.B.grad is not None and mixed.P.grad is not None
