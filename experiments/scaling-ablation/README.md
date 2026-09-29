# Rank-scaling ablation

This exploratory ablation asks whether the controlled rank trend is explained
solely by changing the effective multiplier `alpha/r`.

- **Policy A:** reuses the independent 1,000-step screening with `alpha=4`.
- **Policy B:** keeps `alpha/r=1` using alpha 1, 2, 4, and 8 at ranks 1, 2, 4,
  and 8.
- **Methods:** LoRA and Symmetric Quadratic.
- **Seed/duration:** seed 42, 1,000 steps. The planned design was deliberately
  single seed and remains exploratory.
- **Private originals:** `results_rank_scaling_ablation/`.
- **Protocol:** [PROTOCOL.md](PROTOCOL.md) and
  [`config/scaling/constant-effective-scale-1000.json`](../../config/scaling/constant-effective-scale-1000.json).
- **Canonical tables/figures:** `results/tables/scaling-ablation/` and
  `results/figures/scaling-ablation/`.
