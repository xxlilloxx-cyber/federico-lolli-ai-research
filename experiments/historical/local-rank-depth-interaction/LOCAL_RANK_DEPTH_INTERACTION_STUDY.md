# Local-Rank × Depth Interaction Study

**Author:** Federico Lolli  
**Contact:** xxlilloxx@gmail.com

## Scope and final status

This report is the final account of the fixed-budget local-rank study. It uses the filesystem audit rather than intermediate draft text. The primary experiments are complete:

- **6,144 parameters:** 2 architectures × 3 geometries × 3 seeds = **18 valid runs**, 2,000 steps each.
- **12,288 parameters:** 2 architectures × 3 geometries × 3 seeds = **18 valid runs**, 2,000 steps each.

The six primary configurations at each budget are LoRA and Symmetric Quadratic with concentrated placement, two-layer distributed placement, and four-layer distributed placement. Every valid primary run reached its configured final step with zero NaN/Inf counts. Failed CUDA attempts remain as historical artifacts and are excluded from all aggregates.

The row-level source of truth is [MASTER_RESULTS.csv](../MASTER_RESULTS.csv). The budget-specific source files are `valid_results.csv`, `valid_results_12288.csv`, `aggregate_6144.csv`, and `aggregate_12288.csv`.

## Scientific question

The experiment asks whether a fixed number of trainable parameters should be concentrated in one Transformer block or distributed across several blocks, and whether LoRA and Symmetric Quadratic react differently to that allocation.

For a 768→768 projection, one rank unit costs 1,536 parameters. At 6,144 parameters the available total rank is four; at 12,288 parameters it is eight. This creates the controlled geometries:

| Budget | Concentrated | Two-layer | Four-layer |
|---:|---|---|---|
| 6,144 | 1×rank4 at layer 2 | rank2 at layers [0,11] | rank1 at [0,3,7,11] |
| 12,288 | 1×rank8 at layer 2 | rank4 at [0,11] | rank2 at [0,3,7,11] |

The 12,288 design is especially informative because it tests four-layer rank2 against one-layer rank8. It does not make local rank and depth statistically independent, so it cannot prove that rank2 is a minimum.

## Models and parameter accounting

LoRA adds a frozen linear projection plus a low-rank correction:

\[
\Delta y = \alpha (xA)B.
\]

Symmetric Quadratic adds:

\[
\Delta y = \alpha (xU)^{\odot 2}P,
\]

where \(U\in\mathbb{R}^{d\times r}\) and \(P\in\mathbb{R}^{r\times d_{out}}\). For output coordinate \(j\),

\[
\Delta y_j=x^TQ_jx,\qquad Q_j=\sum_kP_{kj}u_ku_k^T.
\]

Thus the local rank is the number of rank-1 symmetric quadratic components available at one insertion point. Distributing the same total rank over more blocks changes where capacity is applied and, at low local rank, how many components are available in each block.

All adapter parameters are trainable; the GPT-2 backbone is frozen. Both architectures have the same theoretical parameter count at the same rank and projection dimensions.

## Protocol

The experiments use GPT-2 Small, WikiText-2 raw text, the GPT-2 tokenizer, consecutive 128-token blocks, batch size 1, gradient accumulation 4, AdamW at learning rate 3e-4, gradient clipping 1.0, FP16 backbone computation, and FP32 adapter parameters. Each LoRA/Symmetric pair uses the same seed, data order, preprocessing, optimizer settings, and number of optimizer steps.

Validation loss is the mean over the recorded validation subset used by the trainer. Lower validation loss and perplexity are better. AULC is the arithmetic mean of recorded validation losses over the run; no missing points are interpolated.

## Results at 6,144 parameters

| Geometry | LoRA val loss | Symmetric val loss |
|---|---:|---:|
| Concentrated | 3.5039 ± 0.0136 | 3.3745 ± 0.0135 |
| Two-layer | 3.3661 ± 0.0036 | 3.3730 ± 0.0278 |
| Four-layer | 3.3199 ± 0.0063 | 3.3794 ± 0.0138 |

At this budget LoRA improves as the rank budget is distributed. Symmetric is essentially unchanged from concentrated to two layers and slightly worse at four layers. This is the observation that motivated the rank-scaling experiment.

## Results at 12,288 parameters

| Geometry | LoRA val loss | Symmetric val loss |
|---|---:|---:|
| Concentrated | 3.5013 ± 0.0048 | 3.3391 ± 0.0128 |
| Two-layer | 3.3157 ± 0.0087 | 3.2390 ± 0.0272 |
| Four-layer | 3.2578 ± 0.0077 | 3.2356 ± 0.0221 |

At this budget both methods improve when capacity is distributed. Relative to the concentrated geometry, LoRA gains −0.1856 with two layers and −0.2436 with four layers. Symmetric gains −0.1001 and −0.1036. Symmetric remains lower than LoRA in all three matched geometries, but the margin narrows from −0.1622 in the concentrated case to −0.0222 in the four-layer case.

## Convergence and figures

The figures in `figures/` show the complete recorded curves. In convergence plots, lower is better. In \(\Delta\mathrm{Loss}=\mathrm{Symmetric}-\mathrm{LoRA}\) plots, negative values favor Symmetric. Error bars represent seed standard deviation, not confidence intervals.

- `convergence_6144.png` and `convergence_12288_confirmation.png`: validation curves.
- `validation_vs_layers_12288.png`: endpoint loss versus number of insertion points at equal budget.
- `delta_loss_vs_step_12288.png`: the architecture gap throughout training.
- `distribution_gain_12288.png`: multi-layer minus concentrated loss within each architecture.
- `loss_vs_local_rank_12288.png`: local rank and placement at fixed total budget.
- `per_seed_scatter_12288.png`: paired seed endpoints.

The 12,288 curves show that Symmetric starts and remains below LoRA on average, while the gap is smallest for four-layer distributed placement. Both architectures benefit from distribution, but the gain is larger for LoRA.

## Interpretation

**Observation.** At both budgets, LoRA benefits from distributing parameters. Symmetric benefits from distribution at 12,288 parameters but not at 6,144 parameters with rank1 per layer.

**Interpretation.** The sign change is consistent with local quadratic capacity affecting how well Symmetric uses multiple depths. It is also compatible with optimization effects or interactions between placement and rank.

**Limitation.** The two budgets do not form a full factorial design. Therefore the data do not establish a causal minimum local rank, nor do they show that one architecture is universally better.

## Operational history

Several early attempts were interrupted by the known external GPU failure mode. Kernel logs recorded NVIDIA Xid 79, PCIe link loss, and processes waiting in `nvidia_ioctl` in state `D`. The runner was stopped, the machine was rebooted, and subsequent runs skipped all valid directories. No NVIDIA driver, CUDA installation, PCIe setting, or system configuration was modified.

These incidents are operational history only. They are not model results. Incomplete directories and failed processes are retained for diagnosis and excluded from means.

## Conclusions

The fixed-budget evidence supports three limited statements:

1. Distribution across depth is useful for LoRA at both tested budgets.
2. Symmetric Quadratic can benefit from distribution when the four-layer local rank is increased from 1 to 2, while its 6,144 rank1 result did not improve.
3. Symmetric has lower validation loss than LoRA in every 12,288 matched geometry, but its advantage becomes small when the budget is distributed over four layers.

These are replicated over three seeds on GPT-2 Small and WikiText-2 validation data. They do not establish held-out test superiority, generality to other models, or a causal rank threshold. Those questions require checkpointed longer runs and held-out evaluation, which are addressed by the subsequent long-convergence study.
