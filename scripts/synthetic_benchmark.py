"""Small controlled function-class benchmark for the six adapter families."""
import csv, json, math, random, sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0, 'src')
from adapters import LowRankAdapter, parameter_counts


class FrozenLinear(nn.Module):
    def __init__(self, d, out):
        super().__init__(); self.weight = nn.Parameter(torch.zeros(d, out), requires_grad=False); self.bias = nn.Parameter(torch.zeros(out), requires_grad=False)
    def forward(self, x): return x @ self.weight + self.bias


def make(kind, d, out, rank=4):
    base = FrozenLinear(d, out)
    if kind == 'linear_quadratic':
        return LowRankAdapter(base, kind=kind, rank_l=rank, rank_q=rank, alpha_l=1.0, alpha_q=1.0)
    return LowRankAdapter(base, kind=kind, rank=rank, alpha=1.0)


def main():
    torch.manual_seed(42); random.seed(42)
    d, out, n = 16, 4, 2048
    x = torch.randn(n, d)
    a = torch.randn(d, out)
    q = torch.randn(d, d); q = (q + q.T) / 2
    targets = {
        'linear': x @ a,
        'quadratic': torch.einsum('ni,io,no->no', x, q, torch.ones_like(x) @ torch.eye(d)) if False else torch.stack([x @ q[:, j] * x[:, j] for j in range(out)], dim=1),
    }
    # Independent symmetric matrices for a genuine multi-output quadratic target.
    qs = torch.stack([((torch.randn(d,d) + torch.randn(d,d).T) / 2) for _ in range(out)])
    targets['quadratic'] = torch.einsum('ni,oij,nj->no', x, qs, x)
    targets['mixed'] = x @ a + targets['quadratic']
    kinds = ['lora', 'quadratic', 'signed_quadratic', 'feature_interaction', 'symmetric_quadratic', 'linear_quadratic']
    rows = []
    for case, y in targets.items():
        for kind in kinds:
            model = make(kind, d, out, rank=4)
            opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.03)
            for step in range(300):
                pred = model(x); loss = (pred-y).square().mean(); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): final = float((model(x)-y).square().mean())
            rows.append({'case': case, 'model': kind, 'trainable_parameters': parameter_counts(model)['trainable_parameters'], 'mse': final, 'steps': 300})
            print(case, kind, f'mse={final:.6f}', flush=True)
    outdir=Path('results_extended/synthetic'); outdir.mkdir(parents=True, exist_ok=True)
    (outdir/'summary.json').write_text(json.dumps(rows, indent=2))
    with (outdir/'metrics.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


if __name__ == '__main__': main()
