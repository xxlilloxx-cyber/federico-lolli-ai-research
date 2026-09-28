# Controlled AG News Results

Six independent GPT-2 classification runs compared LoRA and Symmetric Quadratic at the same `attn.c_proj` placement, rank 4, and 9,216 trainable parameters (including the shared classification head). The three seeds were 42, 123, and 456. Each run used 4,096 training examples, a fixed 1,000-example stratified validation split, 500 optimizer steps, and the original AG News test split.

| Method | Test accuracy (mean ± sample SD) | Test macro-F1 (mean ± sample SD) | Test loss (mean ± sample SD) |
|---|---:|---:|---:|
| LoRA | 0.8201 ± 0.0051 | 0.8156 ± 0.0050 | 0.5445 ± 0.0542 |
| Symmetric Quadratic | 0.8021 ± 0.0240 | 0.7929 ± 0.0315 | 0.6847 ± 0.1919 |

The paired Symmetric-minus-LoRA mean differences are -0.0179 accuracy points, -0.0227 macro-F1 points, and +0.1401 test loss. Symmetric is higher in accuracy and macro-F1 in one of three matched seeds (seed 42), while LoRA is higher in the other two. The method difference is comparable with, and for Symmetric smaller than, the observed seed-to-seed dispersion; with three seeds these data do not establish a statistically reliable improvement.

This controlled downstream result does not reproduce a universal advantage for Symmetric Quadratic. It complements the earlier WikiText-2 evidence: the language-modeling advantage depended on placement, depth allocation, and optimization duration, while this AG News protocol favoured LoRA on average. The results support testing quadratic adaptation as a conditional design choice, not treating it as a replacement for LoRA.

Figures: `docs/assets/publication/downstream_accuracy_comparison.*` and `docs/assets/publication/downstream_macro_f1_comparison.*`. Individual values and paired differences are in `docs/data/downstream_controlled_results.csv` and `docs/data/downstream_controlled_paired_differences.csv`.
