# Controlled Rank 5,000-Step Confirmation Report

## Scope

24 fresh WikiText-2 runs: LoRA and Symmetric Quadratic, ranks 1/2/4/8, seeds 42/123/456, each trained for 5,000 steps. This is independent of historical work and the 1,000-step screening.

## Final validation and held-out test

| Rank | LoRA final val | Symmetric final val | paired S−L val | LoRA test | Symmetric test | paired S−L test |
| 1 | 3.5734 ± 0.0051 | 3.5314 ± 0.0207 | -0.0419 ± 0.0249 | 3.9256 ± 0.0009 | 3.9131 ± 0.0042 | -0.0125 ± 0.0051 |
| 2 | 3.4773 ± 0.0057 | 3.4340 ± 0.0249 | -0.0434 ± 0.0250 | 3.8492 ± 0.0025 | 3.8305 ± 0.0144 | -0.0187 ± 0.0119 |
| 4 | 3.3483 ± 0.0093 | 3.3101 ± 0.0011 | -0.0382 ± 0.0095 | 3.7656 ± 0.0054 | 3.7429 ± 0.0024 | -0.0227 ± 0.0035 |
| 8 | 3.3054 ± 0.0117 | 3.2446 ± 0.0096 | -0.0608 ± 0.0179 | 3.7319 ± 0.0035 | 3.6938 ± 0.0044 | -0.0381 ± 0.0072 |

Final validation is measured at step 5,000. Test loss uses that final checkpoint. PPL is computed per seed and then averaged. Negative paired differences favour Symmetric. Three seeds show variability but do not establish broad significance. Rank also changes alpha/r (4, 2, 1, 0.5), so this does not isolate pure rank.

## Remaining work

The separate constant-scale ablation, spectral analysis and interaction analysis are not yet executed.
