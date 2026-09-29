# Mechanism Analysis Report

## 1. Motivation

The controlled WikiText-2 confirmation found lower mean final validation and
held-out test loss for Symmetric Quadratic at ranks 1, 2, 4, and 8. The AG News
classification experiment instead favored LoRA on average. This report asks
what structural differences are measurable; it does not claim that those
differences cause either task result.

## 2. Scaling ablation

The primary rule is `scale=4/r`. A separate seed-42, 1,000-step ablation fixed
`scale=1` by setting alpha to 1/2/4/8 at ranks 1/2/4/8.

| Rank | Current S−L loss | Constant-scale S−L loss |
|---:|---:|---:|
| 1 | +0.0443 | +0.0410 |
| 2 | -0.0300 | -0.0292 |
| 4 | -0.0750 | -0.0750 |
| 8 | -0.1177 | -0.0975 |

The sign pattern persists: LoRA is lower at rank 1, Symmetric at ranks 2–8.
Scaling changes magnitude, especially at rank 8, but does not remove the
seed-42 trend. This remains exploratory because Policy B has one seed.

## 3. Analytical distinction

LoRA has `f_L(x)=s(xA)B`, constant Jacobian `sAB`, and zero adapter-only
Hessian. Symmetric has `f_S(x)=s((xU)⊙(xU))P`, local Jacobian
`2s U diag(xU) P`, and for fixed output reduction `c` the Hessian
`2s U diag(Pc) U^T`. Analytical formulas match double-precision autograd tests.

## 4. Spectral and Jacobian evidence

Across three seeds, mean entropy effective ranks of LoRA's effective update are
1.00, 1.92, 3.76, and 6.57 at nominal ranks 1, 2, 4, and 8. Symmetric's mean
local-Jacobian effective ranks on the fixed held-out sample are 1.00, 1.67,
3.09, and 5.59. These are different objects: a constant linear update versus
an activation-dependent local derivative. The data do not support calling
Symmetric a higher-rank weight update.

Symmetric U effective rank grows approximately 1.00, 2.00, 3.99, 7.96; P grows
1.00, 1.98, 3.89, 7.28. The empirical adapter-output covariance effective rank
on 32 held-out token positions grows 1.00, 1.92, 3.00, 3.59. This shows learned
use of multiple channels, while the single-block sample limits generalization.

## 5. Second-order interactions

LoRA's adapter-only Hessian norm is mathematically and numerically zero.
Symmetric's mean Hessian Frobenius norms over seeds are 155.15, 92.13, 52.68,
and 17.61 at ranks 1, 2, 4, and 8 under the fixed residual-direction scalar.
The decline reflects learned factors together with changing `alpha/r`; it is
not evidence that rank reduces useful interactions. Off-diagonal entries are
non-zero and dense under the stated threshold. One Hessian per checkpoint is
used because, with fixed `c`, it is input-independent; repeated token positions
are not treated as independent observations.

Full-block/model Hessians were not executed. A reproducible scalar response and
dimension-reduction protocol must be frozen before incurring the substantial
second-derivative cost.

## 6. Convergence dynamics

Mean paired validation differences (Symmetric minus LoRA) at selected fresh
evaluation steps are:

| Rank | 250 | 500 | 1000 | 2000 | 3000 | 4000 | 5000 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.0066 | +0.0536 | +0.0280 | +0.0096 | -0.0043 | -0.0265 | -0.0419 |
| 2 | -0.0006 | -0.0054 | -0.0367 | -0.0432 | -0.0319 | -0.0311 | -0.0434 |
| 4 | -0.0042 | -0.0254 | -0.0707 | -0.0631 | -0.0803 | -0.0635 | -0.0382 |
| 8 | -0.0299 | -0.0737 | -0.1247 | -0.1156 | -0.0947 | -0.0781 | -0.0608 |

Rank 1 changes sign between 2,000 and 3,000 at the selected checkpoints, with
noisy shorter crossings at the 25-step resolution. Ranks 2–8 favor Symmetric
from early or intermediate training. At ranks 4 and 8 the advantage peaks
before step 5,000 and narrows later. Curves describe optimization behavior, not
its cause.

## 7. Computational cost

On the RTX A500, rank 4, batch 1, sequence 128, after 10 warmups and across 30
iterations: LoRA forward is 12.157 ± 0.243 ms and backward 14.805 ± 0.236 ms;
Symmetric forward is 12.184 ± 0.236 ms and backward 14.766 ± 0.188 ms. Peak
allocated VRAM is 429.05 versus 430.31 MiB. Differences are small in this
focused benchmark and should not be generalized to other hardware or ranks.

## 8. Scientific interpretation

**Measured:** Symmetric has lower controlled WikiText-2 final/test loss at all
four ranks, but LoRA has higher AG News accuracy and macro-F1 on average. The
constant-scale seed-42 trend remains. The adapters have distinct local
Jacobians and only Symmetric has learned non-zero adapter Hessians.

**Mathematically guaranteed:** LoRA is linear in adapter input and its
adapter-only Hessian is zero. Symmetric is quadratic, has an input-dependent
Jacobian, and a generally non-zero rank-at-most-r Hessian for fixed linear
output reductions.

**Hypothesis:** explicit local second-order structure may help the tested
language-modeling configuration. The experiments do not isolate this structure
as the cause and show no universal downstream advantage.
