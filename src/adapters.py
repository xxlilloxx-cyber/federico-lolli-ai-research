import math
import torch
from torch import nn


class LowRankAdapter(nn.Module):
    """Frozen linear layer plus a zero initialized adapter branch.

    Feature interaction is ((XU) elementwise (XV))P. Expanding each product
    gives (sum_i x_i U_ik)(sum_j x_j V_jk), including x_i*x_j terms.
    """
    KINDS = {"lora", "quadratic", "signed_quadratic", "feature_interaction",
             "symmetric_quadratic", "linear_quadratic"}

    def __init__(self, base, rank=4, alpha=4.0, kind="quadratic", diagnostics=False,
                 rank_l=None, rank_q=None, alpha_l=None, alpha_q=None):
        super().__init__()
        if kind not in self.KINDS or rank < 1:
            raise ValueError(f"kind must be one of {self.KINDS}")
        self.base, self.kind, self.rank, self.alpha = base, kind, rank, alpha
        self.rank_l = rank_l or rank
        self.rank_q = rank_q or rank
        self.alpha_l = alpha_l if alpha_l is not None else alpha
        self.alpha_q = alpha_q if alpha_q is not None else alpha
        self.diagnostics, self.last_stats = diagnostics, {}
        for p in base.parameters():
            p.requires_grad_(False)
        d, out = base.weight.shape
        if kind == "linear_quadratic":
            self.A = nn.Parameter(torch.empty(d, self.rank_l))
            self.B = nn.Parameter(torch.zeros(self.rank_l, out))
            self.U = nn.Parameter(torch.empty(d, self.rank_q))
            self.V = nn.Parameter(torch.empty(d, self.rank_q))
            self.P = nn.Parameter(torch.zeros(self.rank_q, out))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        elif kind == "symmetric_quadratic":
            self.U = nn.Parameter(torch.empty(d, rank))
            self.P = nn.Parameter(torch.zeros(rank, out))
            nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        else:
            self.U = nn.Parameter(torch.empty(d, rank))
            nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        if kind == "feature_interaction":
            self.V = nn.Parameter(torch.empty(d, rank))
            self.P = nn.Parameter(torch.zeros(rank, out))
            nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        elif kind not in {"symmetric_quadratic", "linear_quadratic"}:
            self.V = nn.Parameter(torch.zeros(rank, out))

    def forward(self, x):
        z = x.float()
        if self.kind == "linear_quadratic":
            linear_features = z
            xu, xv = z @ self.U, z @ self.V
            quadratic_features = xu * xv
            linear_branch = (linear_features @ self.A) @ self.B
            quadratic_branch = quadratic_features @ self.P
            branch = (self.alpha_l / self.rank_l) * linear_branch + (self.alpha_q / self.rank_q) * quadratic_branch
            features = quadratic_features
        elif self.kind == "lora":
            features = z
            branch = (features @ self.U) @ self.V
        elif self.kind == "symmetric_quadratic":
            features = z @ self.U
            branch = (features * features) @ self.P
        elif self.kind == "quadratic":
            features = z.square()
            branch = (features @ self.U) @ self.V
        elif self.kind == "signed_quadratic":
            features = z * z.abs()
            branch = (features @ self.U) @ self.V
        else:
            xu, xv = z @ self.U, z @ self.V
            features = xu * xv
            branch = features @ self.P
        if self.diagnostics:
            with torch.no_grad():
                base_out = self.base(x).float()
                branch_out = branch.float()
                self.last_stats = {"mean_x": z.mean().item(), "std_x": z.std().item(),
                                   "max_abs_x": z.abs().max().item(),
                                   "adapter_output_norm": branch_out.norm().item(),
                                   "base_output_norm": base_out.norm().item(),
                                   "adapter_base_ratio": (branch_out.norm() / (base_out.norm() + 1e-8)).item()}
                if self.kind in {"quadratic", "signed_quadratic", "symmetric_quadratic"}:
                    self.last_stats.update(mean_transform=features.mean().item(),
                                           std_transform=features.std().item(),
                                           max_abs_transform=features.abs().max().item())
                elif self.kind == "feature_interaction":
                    self.last_stats.update(mean_xu=xu.mean().item(), std_xu=xu.std().item(),
                                           mean_xv=xv.mean().item(), std_xv=xv.std().item(),
                                           mean_interaction=features.mean().item(),
                                           std_interaction=features.std().item(),
                                           max_abs_interaction=features.abs().max().item())
                elif self.kind == "linear_quadratic":
                    self.last_stats.update(mean_xu=xu.mean().item(), std_xu=xu.std().item(),
                                           mean_xv=xv.mean().item(), std_xv=xv.std().item(),
                                           linear_norm=linear_branch.norm().item(),
                                           quadratic_norm=quadratic_branch.norm().item(),
                                           input_linear_norm=(self.base(x).float().norm().item()))
        return self.base(x) + (self.alpha / self.rank) * branch.to(x.dtype)


def freeze_and_insert(model, kind, rank=4, alpha=4.0, layers=(0,),
                      projections=("attn.c_proj",), diagnostics=False, **kwargs):
    for p in model.parameters():
        p.requires_grad_(False)
    for i in layers:
        block = model.transformer.h[i]
        for name in projections:
            parent = block
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            leaf = parts[-1]
            base = getattr(parent, leaf)
            adapter = LowRankAdapter(base, rank, alpha, kind, diagnostics, **kwargs)
            adapter.to(device=base.weight.device)
            setattr(parent, leaf, adapter)
    return model


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable,
            "trainable_percentage": 100 * trainable / total}


def _adapter_modules(model):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, LowRankAdapter)]


def save_adapter_checkpoint(model, path, config=None, optimizer=None, step=None):
    """Save only adapter weights plus enough metadata to validate a reload."""
    modules = _adapter_modules(model)
    if not modules:
        raise ValueError("model contains no LowRankAdapter modules")
    metadata = dict(config or {})
    metadata.update({"checkpoint_format": 1,
                     "adapter_modules": [name for name, _ in modules],
                     "adapter_kinds": [m.kind for _, m in modules],
                     "adapter_ranks": [m.rank for _, m in modules],
                     "checkpoint_step": step})
    state = {}
    for name, module in modules:
        for key, value in module.state_dict().items():
            state[f"{name}.{key}"] = value.detach().cpu()
    payload = {"metadata": metadata, "adapter_state": state}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    path = str(path)
    torch.save(payload, path)
    return metadata


def load_adapter_checkpoint(model, path, expected_config=None, map_location="cpu"):
    """Load adapter-only weights and reject architecture/rank/layer mismatches."""
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    metadata = payload.get("metadata", {})
    modules = dict(_adapter_modules(model))
    expected_modules = metadata.get("adapter_modules", [])
    if set(expected_modules) != set(modules):
        raise ValueError(f"adapter module mismatch: checkpoint={expected_modules}, model={list(modules)}")
    if expected_config:
        for key in ("kind", "rank", "layers", "projections", "rank_l", "rank_q"):
            if key in expected_config and key in metadata and expected_config[key] != metadata[key]:
                raise ValueError(f"checkpoint {key} mismatch: {metadata[key]} != {expected_config[key]}")
    for name, module in modules.items():
        prefix = name + "."
        state = {k[len(prefix):]: v for k, v in payload["adapter_state"].items() if k.startswith(prefix)}
        if not state:
            raise ValueError(f"missing state for adapter module {name}")
        try:
            module.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"adapter shape mismatch for {name}: {exc}") from exc
    return metadata, payload.get("optimizer_state")
