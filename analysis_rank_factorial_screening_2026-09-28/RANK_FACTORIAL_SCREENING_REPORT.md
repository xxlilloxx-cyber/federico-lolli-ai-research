# Controlled Rank-Factorial Screening Report

**Scope.** Fresh WikiText-2 causal-language-modeling screening: two methods × ranks 1/2/4/8 × seeds 42/123/456, with 1,000 optimizer steps per run. This is a controlled screening study; it is not the historical 5,000-step campaign and it does not provide held-out test results.

## Protocol

All runs use frozen GPT-2 (`gpt2`), adapters at `transformer.h[0].attn.c_proj` in block 0, sequence length 128, gradient accumulation 4, AdamW learning rate 3e-4, alpha 4, and fresh deterministic seeds. LoRA and Symmetric use the same rank within each paired comparison. Since a 768→768 adapter has 1,536 trainable parameters per rank, their adapter budgets match at each rank.

## Final validation results

| Rank | LoRA loss (mean ± SD) | Symmetric loss (mean ± SD) | Symmetric − LoRA | LoRA PPL (mean ± SD) | Symmetric PPL (mean ± SD) |
| 1 | 3.6075 ± 0.0143 | 3.6355 ± 0.0230 | +0.0280 ± 0.0155 | 36.88 ± 0.53 | 37.93 ± 0.88 |
| 2 | 3.5798 ± 0.0054 | 3.5460 ± 0.0088 | -0.0338 ± 0.0037 | 35.87 ± 0.19 | 34.68 ± 0.31 |
| 4 | 3.5491 ± 0.0060 | 3.4770 ± 0.0144 | -0.0721 ± 0.0194 | 34.78 ± 0.21 | 32.36 ± 0.47 |
| 8 | 3.5406 ± 0.0077 | 3.4083 ± 0.0209 | -0.1323 ± 0.0148 | 34.49 ± 0.27 | 30.22 ± 0.63 |

Perplexity is calculated separately as `exp(final token-level cross-entropy loss)` for each seed, then averaged. It is valid here because the stored language-modeling loss is token-level cross entropy.

## Observations

- The paired mean loss difference is positive at rank 1 if Symmetric is worse, and negative if Symmetric is lower. Interpret the table rather than a single seed: all three seeds are retained at every rank.
- Rank changes both the number of trainable parameters and the effective scale `alpha/r`; this study does not isolate those factors.
- The result is limited to 1,000 training steps and a small validation sample used by the existing pipeline. It does not establish a 5,000-step or held-out-test rank effect.

## Files

- `rank_factorial_screening_per_seed.csv`: one row per valid run.
- `rank_factorial_screening_aggregate.csv`: mean, sample SD, min and max.
- `rank_factorial_screening_paired_differences.csv`: within-seed Symmetric − LoRA comparisons.
- `figures/`: PNG, SVG and PDF figures; dots are seeds and error bars are sample SD.

## Next controlled step

Run fresh 5,000-step confirmations for ranks 1/2/4/8 and seeds 42/123/456 before claiming a persistent rank effect. The screening data should not be merged with the historical 5,000-step results.
