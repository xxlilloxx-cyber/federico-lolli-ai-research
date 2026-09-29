# Multi-layer Fixed-budget Study — LoRA vs Symmetric Quadratic

**Author:** Federico Lolli  
**Contact:** xxlilloxx@gmail.com

## Executive summary

This report is a separate continuation of the single-layer and depth-placement studies. It asks whether a fixed total adapter budget is used more effectively when concentrated in one Transformer block or distributed over multiple blocks. The attention output projection `attn.c_proj` was used because it is the original matched-budget reference and has identical 768→768 dimensions at every block.

The total target was 6,144 trainable parameters, equivalent to four rank units because one rank costs `768+768=1,536` parameters. The tested valid configurations therefore use rank 4 in one layer, rank 2 in two layers, or rank 1 in four layers. LoRA and Symmetric use exactly the same layers and total parameter count within each pair.

This document preserves the original phase-1 screening history. The screening was later superseded by complete 6,144- and 12,288-parameter multi-layer confirmations and the 5,000-step campaign in the local-rank and long-convergence reports. The failed data-driven attempt remains an operational artifact and is excluded from all final aggregates.

## 1. Scientific question and design

The single-layer depth scan showed that Symmetric can be competitive at several depths. The present experiment changes only the distribution of the same total rank budget. For LoRA the per-layer correction is `α(xA_l)B_l`; for Symmetric it is `α(xU_l)^(⊙2)P_l`. Parameters are independent across layers and the GPT-2 backbone is frozen.

For each architecture, the distribution gain is defined relative to the concentrated layer-2 run:

`Gain_distribution = Loss_multi-layer − Loss_concentrated`.

A negative value means distribution improved validation loss under the same total budget. The architecture interaction is:

`Interaction = (Loss_SYM,multi − Loss_SYM,single) − (Loss_LoRA,multi − Loss_LoRA,single)`.

Negative interaction means distribution helped Symmetric more than LoRA.

## 2. Parameter accounting and validation

The smoke test verified forward, backward, finite gradients, frozen backbone and exact 6,144 trainable parameters for rank allocations 4, 2+2 and 1+1+1+1. Every valid screen run has 1,000 metric rows, zero NaN/Inf and peak allocation near 348.4 MiB. No checkpoint was produced by the historical screening trainer; metrics and summaries are the source of truth for this historical phase. Later long runs use validated adapter-only checkpoints.

## 3. Valid screening results

| Configuration | Model | Layers | Rank/layer | Steps | Params | Val loss | PPL | AULC | Time s | Peak MiB | Valid |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| concentrated_layer2_lora_seed42 | lora | 2 | 4 | 1000 | 6144 | 3.5744 | 35.6718 | 3.6429 | 149.9688 | 348.3662 | True |
| concentrated_layer2_symmetric_quadratic_seed42 | symmetric_quadratic | 2 | 4 | 1000 | 6144 | 3.4807 | 32.4811 | 3.5800 | 151.8802 | 348.3662 | True |
| distributed_four_lora_seed42 | lora | 0,3,7,11 | 1 | 1000 | 6144 | 3.3962 | 29.8511 | 3.4945 | 159.8082 | 348.3662 | True |
| distributed_four_symmetric_quadratic_seed42 | symmetric_quadratic | 0,3,7,11 | 1 | 1000 | 6144 | 3.4792 | 32.4353 | 3.5796 | 163.4717 | 348.3662 | True |
| distributed_two_lora_seed42 | lora | 0,11 | 2 | 1000 | 6144 | 3.4369 | 31.0903 | 3.5500 | 155.0679 | 348.3662 | True |
| distributed_two_symmetric_quadratic_seed42 | symmetric_quadratic | 0,11 | 2 | 1000 | 6144 | 3.4149 | 30.4132 | 3.5261 | 156.8167 | 348.3662 | True |

For the valid configurations, the data show how the same budget behaves when moved across depth. The exact per-step source is in each run's `metrics.csv`; aggregate rows are in `screen_results.csv`.

## 4. How to read the figures

`validation_curves.png` shows loss versus optimization step; lower is better and curves are directly comparable only within the same step budget. `delta_loss_architecture.png` is the signed Symmetric-minus-LoRA endpoint difference; values below zero favor Symmetric. `loss_vs_insertion_points.png` compares one, two and four insertion points, but the layer sets differ by strategy, so it is a distribution screen rather than a pure N-only causal plot. `parameter_efficiency.png` confirms that all valid points have the same 6,144-parameter target; differences cannot be attributed to a larger trainable budget.

## 5. Observations, interpretations and limitations

**Observation.** The concentrated, two-layer and four-layer pairs all satisfy the fixed total parameter budget and use the same seed and training protocol.

**Interpretation.** This is a clean test of capacity placement within the tested configurations. It does not compare arbitrary rank choices or larger budgets.

**Observation.** The data-driven two-layer LoRA attempt entered an uninterruptible `D` state before producing a valid result. Its directory is retained as an error artifact, but it is excluded from every table and mean.

**Interpretation.** The GPU runtime remains an operational limitation. The failure does not imply an architectural failure because no completed training result exists for that run.

**Limitation.** Early, middle and late regional configurations were not reached after the lockup. Three-seed and 2,000-step confirmations were subsequently executed in the local-rank study, followed by 5,000-step three-seed runs. Therefore this historical report should not be read as the final state of the campaign; the later reports contain the consolidated evidence.

## 6. Reproducibility and file map

- `results_multilayer/screen/`: separate configurations, metrics and summaries.
- `screen_results.csv`: valid aggregate rows.
- `figures/validation_curves.png`
- `figures/delta_loss_architecture.png`
- `figures/loss_vs_insertion_points.png`
- `figures/parameter_efficiency.png`
- `scripts/validation/smoke_multilayer.py`
- `scripts/training/run_multilayer_screen.py`

The earlier single-layer results remain in `results/` and `results_depth/` and were not overwritten.

## 7. Next step

Because the GPU lockup occurred before the requested data-driven and regional comparisons, the scientifically useful next step is to recover the GPU runtime and rerun only the missing phase-1 configurations, preferably in separate processes with a health check between runs. Only after all phase-1 strategies are available should the most informative configurations be promoted to 2,000 steps and three seeds.
