# Publication-Ready Summary

## Provisional title
**Regularization Strength Controls Stability and Plasticity in Parameter-Efficient Continual Factual Learning**

## Research question and formulation
How does online-EWC strength affect memory and acquisition across LoRA, Symmetric, and Combined adapters?
$$L_{total}=L_{current}+\frac{\lambda}{2}\sum_iF_i(\theta_i-\theta_i^*)^2,\quad F_{AB}=F_A+F_B.$$

## Protocol
Frozen GPT-2 Small; block-0 `attn.c_proj`; A→B→C without replay; 2,500 steps per phase; 6,144 trainable adapter parameters for sequential/EWC; confirmation seeds 42, 123, 456.

## Main three-seed findings
| method | lambda | mean_forgetting_mean | mean_forgetting_sample_sd | mean_new_fact_acquisition_nll_mean | mean_new_fact_acquisition_nll_sample_sd | mean_final_memory_nll_mean | mean_final_memory_nll_sample_sd | mean_final_probability_mean | mean_final_exact_match_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combined | 0.0000 | 5.5329 | 0.2459 | 0.2991 | 0.0922 | 3.9493 | 0.0523 | 0.2553 | 0.1343 |
| combined | 10.0000 | 3.9336 | 0.4809 | 0.5758 | 0.1149 | 2.9572 | 0.3171 | 0.1996 | 0.0648 |
| combined | 30.0000 | 2.7443 | 0.3321 | 1.1637 | 0.2463 | 2.4373 | 0.0694 | 0.1259 | 0.0046 |
| lora | 0.0000 | 4.4847 | 0.3475 | 0.4435 | 0.0853 | 3.6856 | 0.0662 | 0.2285 | 0.0833 |
| lora | 1.0000 | 4.1844 | 0.4172 | 0.4920 | 0.0963 | 3.4855 | 0.0915 | 0.2173 | 0.0787 |
| lora | 10.0000 | 2.5429 | 0.6715 | 0.9805 | 0.2138 | 2.6169 | 0.2279 | 0.1510 | 0.0046 |
| symmetric | 0.0000 | 5.7811 | 0.0958 | 0.1454 | 0.0466 | 4.0117 | 0.1490 | 0.2979 | 0.2778 |
| symmetric | 10.0000 | 4.2774 | 0.3552 | 0.3189 | 0.0660 | 3.0044 | 0.2639 | 0.2465 | 0.1343 |
| symmetric | 30.0000 | 3.1026 | 0.4115 | 0.7046 | 0.1744 | 2.3888 | 0.1854 | 0.1782 | 0.0185 |

## Exploratory sweep
The seven-point seed-42 sweep characterizes the transition and high-λ under-learning; it is not pooled as replication.

## Six figures
- [Figure 1](../figures_main/figure_1_exploratory_stability_plasticity.svg)
- [Figure 2](../figures_main/figure_2_confirmation_stability_plasticity.svg)
- [Figure 3](../figures_main/figure_3_final_balanced_memory.svg)
- [Figure 4](../figures_main/figure_4_forgetting_decomposition.svg)
- [Figure 5](../figures_main/figure_5_plasticity_exact_generation.svg)
- [Figure 6](../figures_main/figure_6_mechanistic_evidence.svg)

## Claims supported
- **SUPPORTED BY CONFIRMATION:** A. EWC strength provides a controllable stability-plasticity trade-off — Paired multi-seed forgetting and acquisition changes.
- **SUPPORTED BY CONFIRMATION:** B. The useful lambda region depends on adapter geometry — Method-specific confirmation grids and Pareto positions.
- **SUPPORTED BY CONFIRMATION:** C. Intermediate regularization can improve balanced memory relative to lambda=0 — Three-seed final mean NLL and paired directions.
- **SUPPORTED EXPLORATORILY ONLY:** D. Very strong EWC reduces forgetting but causes under-learning — Seed-42 lambda 50/100/300 sweep; not in confirmation grid.
- **SUPPORTED BY CONFIRMATION:** E. Adapter geometries respond differently to identical EWC strength — Cross-method mean and seed-wise responses at shared lambdas.
- **SUPPORTED EXPLORATORILY ONLY:** F. Parameter-displacement reduction tracks forgetting reduction — Within-method exploratory correlations; mechanistic consistency, not causality.

## Limitations
Three seeds and one factual-data generator/model/placement constrain inference. Hard/grow controls use 18,432 parameters and are not parameter matched.

## Recommended paper structure
Motivation; continual factual-learning setup; adapter geometries; online EWC; exploratory sweep; confirmation; mechanism; discussion; limitations.

## Recommended additional experiment
Independent fact-set generations and phase-order permutations at selected λ values.
