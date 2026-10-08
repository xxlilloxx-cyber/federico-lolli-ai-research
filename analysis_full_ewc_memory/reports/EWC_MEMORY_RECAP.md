# EWC Memory Project Recap

- Question: how λ controls stability and plasticity in parameter-efficient continual factual learning.
- Included: sequential, grow-unfrozen, hard consolidation, EWC, exploratory sweep, and multi-seed confirmation.
- Unique scientific runs: 72.
- Exploratory λ: 0, 1, 10, 30, 50, 100, 300 (seed 42).
- Confirmation: LoRA 0/1/10; Symmetric 0/10/30; Combined 0/10/30; seeds 42/123/456.
- Candidate assessment:
- combined λ=10: CONFIRMED, final NLL 2.9572 versus λ=0 3.9493 (3/3).
- combined λ=30: CONFIRMED, final NLL 2.4373 versus λ=0 3.9493 (3/3).
- lora λ=1: CONFIRMED, final NLL 3.4855 versus λ=0 3.6856 (3/3).
- lora λ=10: CONFIRMED, final NLL 2.6169 versus λ=0 3.6856 (3/3).
- symmetric λ=10: CONFIRMED, final NLL 3.0044 versus λ=0 4.0117 (3/3).
- symmetric λ=30: CONFIRMED, final NLL 2.3888 versus λ=0 4.0117 (3/3).
- Global confirmation Pareto: combined λ=0, combined λ=10, lora λ=1, lora λ=10, symmetric λ=0, symmetric λ=10, symmetric λ=30.
- Major limits: three seeds, one dataset generator, GPT-2 Small, one placement.
- Publication status: justified as a controlled, scoped confirmation; broader generalization requires more datasets/orders.
- Exact next experiment: repeat selected points across independently generated A/B/C sets and permuted phase orders.
- Full report: [EWC_MEMORY_FINAL_ANALYSIS.md](EWC_MEMORY_FINAL_ANALYSIS.md)

## Six main figures
- [Figure 1](../figures_main/figure_1_exploratory_stability_plasticity.svg)
- [Figure 2](../figures_main/figure_2_confirmation_stability_plasticity.svg)
- [Figure 3](../figures_main/figure_3_final_balanced_memory.svg)
- [Figure 4](../figures_main/figure_4_forgetting_decomposition.svg)
- [Figure 5](../figures_main/figure_5_plasticity_exact_generation.svg)
- [Figure 6](../figures_main/figure_6_mechanistic_evidence.svg)

## Main tables
- [Table 1](../tables/TABLE_1_CONFIRMATION_PROTOCOL.csv)
- [Table 2](../tables/TABLE_2_CONFIRMATION_RESULTS.csv)
- [Table 3](../tables/TABLE_3_PAIRED_DIFFERENCES.csv)
- [Table 4](../tables/TABLE_4_GLOBAL_PARETO.csv)
- [Table 5](../tables/TABLE_5_EXPLORATORY_SWEEP.csv)
