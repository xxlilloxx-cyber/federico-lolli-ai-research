# EWC Memory Final Analysis

## 1. Executive summary
The primary evidence is the three-seed confirmation; the seven-point λ sweep remains exploratory and the ABC protocol comparison is mechanistic context. The study asks how regularization strength controls stability and plasticity across three adapter geometries.

## 2. Research question
How strongly should adapter parameters resist changes important to previous facts while retaining acquisition of new facts?

## 3. Mathematical formulation
$$L_{total}=L_{current}+\frac{\lambda}{2}\sum_iF_i(\theta_i-\theta_i^*)^2,\qquad F_{AB}=F_A+F_B,$$ with γ=1. Lambda zero is unconstrained plasticity; increasing λ resists movement; very large λ may under-learn new facts.

## 4. Experimental hierarchy
Level A: 27 confirmation cells (three seeds). Level B: 21 exploratory sweep cells (seed 42). Level C: 36 ABC baseline/mechanistic cells. Deduplication yields 72 unique runs.

## 5. ABC continual-learning baseline
Sequential reuses 6,144 parameters. Grow-unfrozen and hard consolidation reach 18,432 active parameters. EWC retains 6,144 trainable parameters plus non-trainable Fisher/reference state.

## 6. Why consolidation is necessary
Sequential learning exhibits parameter overwrite. Capacity growth alone does not guarantee preservation. Hard consolidation prevents tensor overwrite but can retain functional interference from simultaneously active slots. EWC instead applies a soft importance-weighted constraint.

## 7. Exploratory lambda sweep
The single-seed sweep maps λ={0,1,10,30,50,100,300}. It supports the shape of the trade-off and the high-λ under-learning observation, not multi-seed uncertainty claims.

## 8. Multi-seed confirmation
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

## 9. LoRA
LoRA candidate λ values are 1 and 10; paired seed directions are reported in Table 3.

## 10. Symmetric
Symmetric candidate λ values are 10 and 30.

## 11. Combined
Combined candidate λ values are 10 and 30. The analysis does not presume or require Combined superiority.

## 12. Global stability-plasticity comparison
Confirmatory global nondominated configurations: **combined λ=0, combined λ=10, lora λ=1, lora λ=10, symmetric λ=0, symmetric λ=10, symmetric λ=30**.

## 13. Balanced-memory result
- combined λ=10: CONFIRMED, final NLL 2.9572 versus λ=0 3.9493 (3/3).
- combined λ=30: CONFIRMED, final NLL 2.4373 versus λ=0 3.9493 (3/3).
- lora λ=1: CONFIRMED, final NLL 3.4855 versus λ=0 3.6856 (3/3).
- lora λ=10: CONFIRMED, final NLL 2.6169 versus λ=0 3.6856 (3/3).
- symmetric λ=10: CONFIRMED, final NLL 3.0044 versus λ=0 4.0117 (3/3).
- symmetric λ=30: CONFIRMED, final NLL 2.3888 versus λ=0 4.0117 (3/3).

## 14. Exact-match behavior
Acquisition, phase-boundary, final, and paraphrase exact-match metrics remain separate. The exploratory sweep identifies collapse thresholds; confirmation evaluates whether candidate points retain generation across seeds.

## 15. Parameter movement
Displacement is analyzed within method and related descriptively to forgetting. Correlation is mechanistic consistency, not causal proof.

## 16. Fisher analysis
Fisher sums, maxima, support, and factor shares are compared within parameterization. Raw Fisher magnitudes are not treated as directly comparable across geometries.

## 17. Hard vs soft consolidation
Hard/grow protocols are mechanistic controls and are not parameter matched to sequential/EWC.

## 18. Limitations
Three seeds, one deterministic synthetic dataset family, GPT-2 Small, one adapter placement, and one phase order limit generalization. Lambda sweep extremes have one seed.

## 19. Supported / unsupported claims
- **SUPPORTED BY CONFIRMATION:** A. EWC strength provides a controllable stability-plasticity trade-off — Paired multi-seed forgetting and acquisition changes.
- **SUPPORTED BY CONFIRMATION:** B. The useful lambda region depends on adapter geometry — Method-specific confirmation grids and Pareto positions.
- **SUPPORTED BY CONFIRMATION:** C. Intermediate regularization can improve balanced memory relative to lambda=0 — Three-seed final mean NLL and paired directions.
- **SUPPORTED EXPLORATORILY ONLY:** D. Very strong EWC reduces forgetting but causes under-learning — Seed-42 lambda 50/100/300 sweep; not in confirmation grid.
- **SUPPORTED BY CONFIRMATION:** E. Adapter geometries respond differently to identical EWC strength — Cross-method mean and seed-wise responses at shared lambdas.
- **SUPPORTED EXPLORATORILY ONLY:** F. Parameter-displacement reduction tracks forgetting reduction — Within-method exploratory correlations; mechanistic consistency, not causality.

## 20. Publication recommendation
The evidence supports a focused publication about architecture-dependent stability-plasticity control under online EWC, with the hierarchy and parameter-budget cautions stated explicitly.

## 21. Next experiments
Replicate the selected operating points with additional fact-set generations and phase orders before making broader claims about factual continual learning.
