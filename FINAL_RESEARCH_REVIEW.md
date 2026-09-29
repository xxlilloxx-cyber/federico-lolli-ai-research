# Final Research Review

**Author:** Federico Lolli  
**Status:** exploratory controlled research; local only; not pushed or deployed.

## Abstract

This project evaluates activation-dependent quadratic low-rank corrections for
a frozen GPT-2 projection. Controlled evidence shows task-dependent behavior.
On WikiText-2, Symmetric Quadratic has lower mean final validation and held-out
test loss than LoRA at ranks 1, 2, 4, and 8 after 5,000 steps. On AG News,
LoRA has higher mean test accuracy and macro-F1. A constant-effective-scale
ablation retains the seed-42 rank pattern, though its magnitude changes.
Checkpoint analyses confirm distinct learned local Jacobians and non-zero
second-order adapter interactions, but do not establish these properties as the
cause of the language-modeling result.

## Evidence inventory and metric definitions

All 24 controlled rank-confirmation cells contain configuration, metrics,
summary, final adapter checkpoint and final WikiText-2 test evaluation.
`best_recorded_validation_loss` is the minimum fresh validation result within a
run. `final_validation_loss` is the fresh step-5,000 evaluation.
`final_test_loss` evaluates that same final checkpoint on the public test split.
Test data did not select rank, method, hyperparameters, or checkpoint.

## Controlled 5,000-step WikiText-2 result

| Rank | LoRA final val | Symmetric final val | paired S−L | LoRA test | Symmetric test | paired S−L |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.5734 ± 0.0051 | 3.5314 ± 0.0207 | -0.0419 ± 0.0249 | 3.9256 ± 0.0009 | 3.9131 ± 0.0042 | -0.0125 ± 0.0051 |
| 2 | 3.4773 ± 0.0057 | 3.4340 ± 0.0249 | -0.0434 ± 0.0250 | 3.8492 ± 0.0025 | 3.8305 ± 0.0144 | -0.0187 ± 0.0119 |
| 4 | 3.3483 ± 0.0093 | 3.3101 ± 0.0011 | -0.0382 ± 0.0095 | 3.7656 ± 0.0054 | 3.7429 ± 0.0024 | -0.0227 ± 0.0035 |
| 8 | 3.3054 ± 0.0117 | 3.2446 ± 0.0096 | -0.0608 ± 0.0179 | 3.7319 ± 0.0035 | 3.6938 ± 0.0044 | -0.0381 ± 0.0072 |

Symmetric has lower final validation and test loss for all three seeds at ranks
2, 4, and 8; the per-seed CSV is authoritative for exact consistency counts.
At rank 1, the mean is lower and the variation is larger. Three seeds support a
consistent descriptive result, not a universal significance claim.

## Controlled 1,000-step screening and convergence

The independent three-seed screen found paired mean differences +0.0280,
-0.0338, -0.0721, and -0.1323 at ranks 1, 2, 4, and 8. Fresh trajectories of
the later 5,000-step runs show rank 1 changing from +0.0280 at step 1,000 to
-0.0419 at step 5,000. The selected-checkpoint crossover is between steps
2,000 and 3,000. Ranks 2–8 already favor Symmetric at step 1,000; their gaps do
not increase monotonically through step 5,000.

## Scaling ablation

With seed 42 and 1,000 steps, fixing `alpha/r=1` produces paired differences
+0.0410, -0.0292, -0.0750, and -0.0975 at ranks 1/2/4/8. The sign pattern
persists; the rank-8 gap is smaller than under fixed alpha=4 (-0.1177). Scaling
is therefore a material confound for magnitude, while this one-seed ablation
does not attribute the residual pattern causally to rank.

## AG News downstream result

Across seeds 42/123/456, LoRA test accuracy is 0.8201 ± 0.0051 and macro-F1
0.8156 ± 0.0050. Symmetric accuracy is 0.8021 ± 0.0240 and macro-F1 0.7929 ±
0.0315. Symmetric is higher on one of three paired seeds. The result favors
LoRA on average and demonstrates that the WikiText-2 ordering is task-specific.

## Mechanism analysis

LoRA effective-update entropy ranks increase from 1.00 to 6.57 over nominal
ranks 1→8. Symmetric local-Jacobian effective ranks increase from 1.00 to 5.59.
These quantities are not the same object. Symmetric U/P factors use multiple
learned channels, while output-covariance effective rank reaches 3.59 at rank 8
on the fixed sample. LoRA's adapter Hessian is exactly zero. Symmetric learned
Hessian Frobenius norms are non-zero (means 155.15, 92.13, 52.68, 17.61 across
ranks), under the documented scalar reduction. This is a structural result,
not proof of a performance mechanism.

## Computational cost

At rank 4, sequence length 128 and batch 1, LoRA versus Symmetric measured
forward latency is 12.157 ± 0.243 versus 12.184 ± 0.236 ms; backward latency is
14.805 ± 0.236 versus 14.766 ± 0.188 ms; peak allocated VRAM is 429.05 versus
430.31 MiB. Parameter counts are identical at 6,144. On this device/configuration
the measured overhead is small.

## Related work

LoRA is the direct linear baseline. DoRA modifies weight magnitude/direction;
MoRA and HiRA target higher-rank weight updates; LoRAN nonlinearly transforms a
weight update; PERA polynomially expands low-rank factors in parameter space.
The present method instead squares projected input activations. Published
benchmarks are context only and are not directly compared numerically. See
`RELATED_WORK_METHOD_COMPARISON.md` for references and methodology mapping.

## Limitations and release assessment

The study uses GPT-2 Small, WikiText-2 and AG News, three main seeds, one
optimizer schedule, context length 128, and a focused insertion point in the
rank factorial. Policy B scaling has one seed. Full-block Hessians and a second
Transformer family are absent. Local stash history remains a documented public
release blocker and was not rewritten.

The project is scientifically ready for a **carefully qualified exploratory
release** once the publication-history blocker is resolved. It is not evidence
of universal superiority, state of the art, or causal attribution to quadratic
interactions.

## Recommended next experiment

Repeat the constant-scale ablation for seeds 123 and 456 only if greater
confidence in the scaling interaction is needed, then replicate the controlled
rank comparison on a second language-modeling dataset and model family.
