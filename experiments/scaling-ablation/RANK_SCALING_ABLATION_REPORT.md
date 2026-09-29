# Rank-Scaling Ablation Report

## Question

The controlled rank studies use a branch multiplier `alpha/r`. The original screening fixed alpha=4, so effective scale changed from 4 at rank 1 to 0.5 at rank 8. This ablation tests a separate Policy B with `alpha/r=1` at every rank.

## Protocol

| Policy | r=1 | r=2 | r=4 | r=8 |
|---|---:|---:|---:|---:|
| A: current | alpha=4; scale=4 | alpha=4; scale=2 | alpha=4; scale=1 | alpha=4; scale=0.5 |
| B: constant scale | alpha=1; scale=1 | alpha=2; scale=1 | alpha=4; scale=1 | alpha=8; scale=1 |

Both policies use GPT-2, frozen backbone, one block-0 `attn.c_proj` adapter, WikiText-2, sequence length 128, gradient accumulation 4, AdamW 3e-4, seed 42, and 1,000 fresh steps. Policy A values are read from the valid screening runs; only Policy B was newly trained.

## Results

| Rank | LoRA A | Symmetric A | S−L A | LoRA B | Symmetric B | S−L B |
| 1 | 3.6171 | 3.6614 | +0.0443 | 3.6301 | 3.6711 | +0.0410 |
| 2 | 3.5861 | 3.5561 | -0.0300 | 3.5984 | 3.5693 | -0.0292 |
| 4 | 3.5530 | 3.4780 | -0.0750 | 3.5530 | 3.4780 | -0.0750 |
| 8 | 3.5492 | 3.4315 | -0.1177 | 3.5056 | 3.4081 | -0.0975 |

## Interpretation

**Measured result.** Under Policy B, inspect the `S−L B` column. This is one seed and one training duration, so it cannot establish a robust causal explanation.

**Does Symmetric still improve with rank at constant scale?** Yes in this seed-42 screen: its loss falls from 3.6711 at rank 1 to 3.4081 at rank 8. LoRA also improves, from 3.6301 to 3.5056. **Does the gap remain?** Yes: it favours LoRA at rank 1 (+0.0410) and Symmetric at ranks 2, 4, and 8 (-0.0292, -0.0750, -0.0975). The rank-4 values are identical across policies because both use alpha=4 there; the rank-8 Symmetric advantage is smaller under constant scale (-0.0975 versus -0.1177). This is evidence that scaling changes the magnitude of the observed pattern, while the single-seed ablation does not establish how much of it is caused by scale versus rank.

**Scaling confound.** If Policy B differs materially from Policy A, alpha/r is an important contributor to the observed trend. If a pattern remains, that is compatible with a capacity/rank contribution but does not isolate it from optimization. Seeds 123 and 456 would be justified only if the seed-42 result is informative; they were intentionally not launched automatically.

## Files

This directory contains the per-seed and aggregate CSVs and reproducible SVG/PNG/PDF plot.
