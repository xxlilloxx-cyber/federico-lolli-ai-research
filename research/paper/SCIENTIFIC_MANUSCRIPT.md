# Symmetric Factorized Quadratic Adaptation: A Controlled Study of Nonlinear Low-Rank Adapters for GPT-2

**Federico Lolli**  
Contact: xxlilloxx@gmail.com

## Abstract

Parameter-efficient fine-tuning adapts a pretrained model while leaving most
weights frozen. We study a symmetric factorized quadratic adapter that maps an
input activation through a low-rank projection, squares the projected channels,
and projects them back. The method has the same parameter count as a matched
LoRA branch, but its correction is quadratic rather than linear in the local
activation. The study combines historical adapter-family and placement screens
with three controlled campaigns: six GPT-2 AG News classification runs, a
24-run 1,000-step WikiText-2 rank screen, and an independent 24-run 5,000-step
WikiText-2 confirmation at ranks 1, 2, 4, and 8. In the 5,000-step campaign,
Symmetric obtains lower mean final validation and held-out test loss at every
rank. In contrast, LoRA obtains higher mean AG News accuracy (0.8201 versus
0.8021) and macro-F1 (0.8156 versus 0.7929). A single-seed constant-scale
ablation preserves the observed rank sign pattern but changes its magnitude.
Local Jacobian spectra and adapter Hessians verify that the trained quadratic
branch has input-dependent first derivatives and nonzero second derivatives;
these structural facts do not establish a causal explanation for performance.
The results support useful, task-dependent quadratic adaptation rather than a
universal advantage over LoRA.

## 1. Introduction

Full fine-tuning updates every model weight, whereas parameter-efficient
fine-tuning keeps most pretrained parameters frozen and learns a small
correction. This reduces task-specific state and makes the correction's
functional form an experimental variable. LoRA is a strong baseline because it
uses few parameters, introduces a controlled rank bottleneck, and is widely
reproducible. Its adapter branch is linear in its incoming activation and
cannot itself represent second-order feature interactions.

This study asks whether an equally sized quadratic branch provides useful
capacity, and how its behavior changes with task, rank, placement, duration,
and scaling. Controlled comparisons are necessary because each of these
choices can alter the observed ordering.

### Research question

Can a low-rank adapter gain useful expressive capacity by replacing the purely
linear LoRA correction with an explicit quadratic function of projected
activations, while retaining a comparable parameter budget and practical
computational cost?

### Contributions

The work provides a precisely defined activation-space quadratic adapter,
matched comparisons with LoRA, rank/scaling/convergence/downstream evidence,
local-Jacobian and Hessian analyses, a matched cost benchmark, and conservative
positioning relative to verified related work. These are empirical and
analytical contributions, not a claim of universal novelty or superiority.

The contribution is an independently evaluated formulation and a controlled
experimental record. It is not a claim that quadratic low-rank networks are
unprecedented. In particular, PERA studies polynomial expansions of low-rank
weight factors and QuadraNet V2 studies factorized quadratic neural layers.
Our experiments instead place a quadratic function of activations beside a
frozen GPT-2 projection.

## 2. Related work

The methods below are compared at the mathematical and architectural levels.
Published benchmark results obtained with different backbones, datasets,
training budgets, or evaluation protocols are not numerically ranked against
the experiments in this study.

LoRA learns the low-rank linear correction `(α/r)(xA)B` and is the direct
experimental baseline. DoRA decomposes weight magnitude and direction. MoRA
uses a square trainable matrix with nonparametric transforms and is designed to
increase effective update rank, while HiRA uses a Hadamard-product high-rank
weight update. LoRAN applies a nonlinear transformation to the low-rank weight
update. These methods operate in weight or parameter space rather than by
squaring the current projected activation. They were not experimentally
reproduced in this study.

### PERA

PERA introduces polynomial expansion in parameter and low-rank-factor space
before factor composition. This study instead applies the nonlinearity directly
to the projected activations:

`x → xU → (xU)⊙(xU) → P`.

Both approaches investigate higher-order structure in parameter-efficient
adaptation, but the polynomial or quadratic operation acts on different
mathematical objects. PERA was not used as an experimental baseline in this
project.

### QuadraNet V2

For one output coordinate, `Δy_j = xQ_jxᵀ`, with
`Q_j = (α/r)U diag(P[:,j])Uᵀ`. This places the method within the family of
factorized quadratic transformations. Across a complete multi-output layer,
all output coordinates share the projection directions contained in `U`, while
`P` provides signed output-specific coefficients.

Consequently, similarity at the level of a single-output quadratic form does
not establish complete architectural equivalence between the Symmetric adapter
and QuadraNet V2. The present study therefore treats QuadraNet V2 as a closely
related quadratic precedent rather than evidence of either exact equivalence
or universal novelty.

### Position of the proposed method

The main distinction of the Symmetric adapter is not simply that it uses low
rank or that it introduces nonlinearity. Its defining feature is that the
low-rank projection is applied to the incoming activation and the projected
coordinates are squared before being mapped back to the output space. This
creates an activation-dependent quadratic correction with an input-dependent
Jacobian, while retaining a parameter structure comparable in size to the
matched LoRA branch.

The experiments in this project evaluate this specific design directly against
LoRA under controlled parameter budgets. Other methods in this section clarify
the relationship to existing parameter-efficient and quadratic adaptation
mechanisms; they are contextual related work rather than experimentally
reproduced baselines. Detailed equivalence and methodology qualifications are
maintained in `comparisons/`.

## 3. Mathematical formulation

The adapter is a trainable correction beside a frozen GPT-2 projection, not a
replacement for the pretrained model. Only the low-rank branch is optimized.
This makes the comparison a test of adapter functional form under matched
matrix shapes and parameter counts.

We use row vectors. For input $x\in\mathbb{R}^{1\times d}$, frozen weight
$W\in\mathbb{R}^{d\times d_{out}}$, and frozen bias $b$, the base projection is

$$y_{base}=xW+b.$$

The LoRA correction is

$$\Delta y_L=\frac{\alpha}{r}(xA)B,$$

where $A\in\mathbb{R}^{d\times r}$ and
$B\in\mathbb{R}^{r\times d_{out}}$. The Symmetric correction is

$$\Delta y_S=\frac{\alpha}{r}((xU)\odot(xU))P,$$

with $U\in\mathbb{R}^{d\times r}$ and
$P\in\mathbb{R}^{r\times d_{out}}$. For output $j$,

$$\Delta y_{S,j}=\frac{\alpha}{r}\sum_kP_{kj}(xu_k)^2=xQ_jx^T,$$

$$Q_j=\frac{\alpha}{r}\sum_kP_{kj}u_ku_k^T.$$

Equivalently,

$$Q_j=\frac{\alpha}{r}U\operatorname{diag}(P_{:,j})U^T.$$

Thus $Q_j$ is symmetric with rank at most $r$. Directions $u_k$ are shared
across outputs, while signed $P_{kj}$ coefficients can yield an indefinite
$Q_j$. The expansion $(ax_1+bx_2)^2=a^2x_1^2+2abx_1x_2+b^2x_2^2$ makes the
pairwise cross interaction explicit: the squared projection is not restricted
to independent squared input features. This is the central structural
difference from LoRA, whose branch remains linear in its input.

Both methods contain $r(d+d_{out})$ trainable adapter parameters. Output
factors $B$ and $P$ are zero initialized,
so both corrections start at zero. At initialization the input factor receives
zero loss gradient until the output factor becomes nonzero.

For the controlled rank study $\alpha=4$, making $\alpha/r$ equal to 4, 2, 1,
and 0.5 for ranks 1, 2, 4, and 8. Rank, parameter count, and effective scale
therefore change together in the primary factorial. This motivated the separate
constant-effective-scale ablation.

![Figure 1. Matched LoRA and Symmetric Quadratic adapter branches beside a frozen row-vector projection.](https://github.com/xxlilloxx-cyber/federico-lolli-ai-research/blob/main/results/figures/architecture/adapter_architecture_comparison.svg)

**Figure 1.** Architecture of the matched LoRA and Symmetric Quadratic
adapters. Both add a low-rank trainable branch beside the frozen projection.
LoRA applies two linear projections, whereas Symmetric inserts an element-wise
square of the projected activation before the output projection, producing an
activation-dependent quadratic correction.

![Figure 2. Adapter placement in the controlled GPT-2 block.](https://github.com/xxlilloxx-cyber/federico-lolli-ai-research/blob/main/docs/assets/figures/gpt2-attn-adapter-path.svg)

**Figure 2.** The controlled adapter and frozen zero-based block 0
`attn.c_proj` receive the same attention output. Their outputs are summed before
attention residual dropout and the residual addition.

## 4. Experimental methodology

The work proceeded through an exploratory stage and a controlled stage. The
exploratory stage compared adapter families, 500/1,000/2,000-step durations,
attention and MLP placements, a full 12-block attention scan, and multilayer
configurations at 6,144- and 12,288-parameter budgets. Those protocols were not
uniform and remain separate from the controlled evidence.

| Campaign | Design | Role |
|---|---|---|
| Historical adapter families | LoRA, quadratic, interaction, Symmetric, and Linear+Quadratic variants | Architecture exploration |
| Depth and placement | Attention/MLP screens and 12-block attention scan | Placement exploration |
| Fixed-budget multilayer | 6,144 and 12,288 parameters distributed across layers | Depth versus local rank |
| AG News | Two methods, three seeds, 500 steps | Controlled classification |
| Rank screening | Two methods, four ranks, three seeds, 1,000 steps | Controlled screening |
| Rank confirmation | 24 fresh runs, 5,000 steps | Central language-model evidence |
| Scaling ablation | Fixed $\alpha=4$ versus $\alpha/r=1$, seed 42 | Exploratory scale check |
| Mechanism and cost | Stored checkpoints and matched rank-4 benchmark | Structural and practical characterization |

The two principal controlled protocols are consolidated below. The independent
1,000-step rank screen is a separate campaign and is not the first segment of
the 5,000-step confirmation trajectories.

| Setting | WikiText-2 confirmation | AG News |
|---|---|---|
| Backbone | GPT-2 Small, frozen | GPT-2 Small, frozen |
| Adapter placement | Block 0 `attn.c_proj` | Block 0 `attn.c_proj` |
| Objective | Causal language modelling | Classification |
| Sequence length | 128 tokens | 128 tokens |
| Training data | 36,718 packed blocks | 4,096 examples |
| Validation | 3,760 packed blocks; eight-block evaluation limit | 1,000-example stratified subset |
| Test | Held-out WikiText-2 test; 2,213 evaluated blocks | Original AG News test split |
| Methods | LoRA, Symmetric | LoRA, Symmetric |
| Rank(s) | 1, 2, 4, 8 | 4 |
| Seeds | 42, 123, 456 | 42, 123, 456 |
| Optimizer / learning rate | AdamW / $3\times10^{-4}$ | AdamW / $3\times10^{-4}$ |
| Batch / accumulation | 1 / 4 | 1 / 4 |
| Gradient clipping | 1.0 | 1.0 |
| Scheduler | None | None |
| Adapter scale | $\alpha=4$ | $\alpha=4$ |
| Training steps | 5,000 | 500 |
| Trainable parameters | Rank dependent; matched by method | 9,216 including classifier |

**Table 3. Controlled experimental protocols.** Fields are taken from the
recorded publication-safe configurations.

The controlled WikiText-2 experiments freeze GPT-2 Small and insert one adapter
beside the input of zero-based block 0 `attn.c_proj`. The frozen projection and
adapter run in parallel; their outputs are summed before attention residual
dropout and residual addition. Text is packed into consecutive 128-token blocks
for causal cross-entropy training. Runs use batch size 1, gradient accumulation
4, AdamW at $3\times10^{-4}$, gradient clipping 1.0, no scheduler, $\alpha=4$,
ranks 1/2/4/8, and seeds 42/123/456. The 1,000-step screen and 5,000-step
confirmation are independent campaigns; they are not concatenated into one
trajectory.

Best recorded validation, final validation at the declared training step, and
held-out test evaluation of the final checkpoint are distinct measurements.
The held-out test set does not select method, rank, hyperparameters, or
checkpoint.

The downstream experiment uses AG News with 4,096 training examples, a fixed
1,000-example stratified validation subset, and the original test split. LoRA
and Symmetric use the same frozen GPT-2 backbone, tokenizer, placement, rank 4,
training schedule, paired seed data order, 500 optimizer steps, and 9,216
trainable parameters including the classification head. This campaign tests
transfer to a classification task.

The controlled phase matches backbone, placement, rank, parameter count,
optimizer, budget, and seed set. Separate scaling, convergence, and downstream
analyses address effective scale, optimization duration, and task dependence
rather than assigning every observed difference to the quadratic operation.

## 5. Controlled results

### 5.1 WikiText-2 rank confirmation

| Rank | LoRA final validation | Symmetric final validation | LoRA test | Symmetric test |
|---:|---:|---:|---:|---:|
| 1 | 3.5734 ± 0.0051 | 3.5314 ± 0.0207 | 3.9256 ± 0.0009 | 3.9131 ± 0.0042 |
| 2 | 3.4773 ± 0.0057 | 3.4340 ± 0.0249 | 3.8492 ± 0.0025 | 3.8305 ± 0.0144 |
| 4 | 3.3483 ± 0.0093 | 3.3101 ± 0.0011 | 3.7656 ± 0.0054 | 3.7429 ± 0.0024 |
| 8 | 3.3054 ± 0.0117 | 3.2446 ± 0.0096 | 3.7319 ± 0.0035 | 3.6938 ± 0.0044 |

Values are means ± sample SD over three seeds. Paired held-out differences are:

| Rank | Symmetric − LoRA test loss | Seeds favouring Symmetric |
|---:|---:|---:|
| 1 | -0.0125 ± 0.0050 | 3/3 |
| 2 | -0.0187 ± 0.0119 | 3/3 |
| 4 | -0.0227 ± 0.0035 | 3/3 |
| 8 | -0.0381 ± 0.0072 | 3/3 |

All twelve paired comparisons favour Symmetric. The conclusion remains
restricted to this backbone, placement, dataset, schedule, and scaling policy.

![Figure 3. Held-out WikiText-2 test loss by rank.](https://github.com/xxlilloxx-cyber/federico-lolli-ai-research/blob/main/docs/assets/publication/rank_5000_final_test_loss.svg)

**Figure 3.** Held-out test loss after 5,000 steps. Lower is better. Individual
seed points, means, and sample SD show Symmetric below LoRA at each tested rank.

### 5.2 AG News

| Method | Test accuracy | Test macro-F1 | Test loss |
|---|---:|---:|---:|
| LoRA | 0.8201 ± 0.0051 | 0.8156 ± 0.0050 | 0.5445 ± 0.0542 |
| Symmetric | 0.8021 ± 0.0240 | 0.7929 ± 0.0315 | 0.6847 ± 0.1919 |

LoRA is higher on average for accuracy and macro-F1, lower on test loss, and
favoured on two of three matched seeds. This negative result is central: the
WikiText-2 ordering does not transfer to the tested classification protocol.

![Figure 4. Controlled AG News accuracy.](https://github.com/xxlilloxx-cyber/federico-lolli-ai-research/blob/main/docs/assets/publication/downstream_accuracy_comparison.svg)

**Figure 4.** AG News test accuracy with every seed and mean ± sample SD. LoRA
is higher on average.

## 6. Scaling ablation and convergence

The seed-42, 1,000-step constant-effective-scale ablation sets $\alpha=r$.

| Rank | LoRA $\alpha=4$ | Sym $\alpha=4$ | $\Delta$ | LoRA $\alpha/r=1$ | Sym $\alpha/r=1$ | $\Delta$ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.6171 | 3.6614 | +0.0443 | 3.6301 | 3.6711 | +0.0410 |
| 2 | 3.5861 | 3.5561 | -0.0300 | 3.5984 | 3.5693 | -0.0292 |
| 4 | 3.5530 | 3.4780 | -0.0750 | 3.5530 | 3.4780 | -0.0750 |
| 8 | 3.5492 | 3.4315 | -0.1177 | 3.5056 | 3.4081 | -0.0975 |

The method differences retain the same signs as the fixed-$\alpha$ screen:
LoRA is lower at rank 1, while Symmetric is lower at ranks 2, 4, and 8. One
seed is insufficient for a stable interaction estimate; scaling influences
magnitude and remains a confound in the primary rank comparison.

![Figure 5. Single-seed scaling ablation.](https://github.com/xxlilloxx-cyber/federico-lolli-ai-research/blob/main/docs/assets/publication/rank_scaling_ablation_loss.svg)

**Figure 5.** Seed-42 comparison of fixed alpha and constant effective scale.
The sign pattern is preserved while effect magnitude changes.

| Rank | Step 1,000 | 2,000 | 3,000 | 4,000 | 5,000 |
|---:|---:|---:|---:|---:|---:|
| 1 | +0.0280 | +0.0096 | -0.0043 | -0.0265 | -0.0419 |
| 2 | -0.0367 | -0.0432 | -0.0319 | -0.0311 | -0.0434 |
| 4 | -0.0707 | -0.0631 | -0.0803 | -0.0635 | -0.0382 |
| 8 | -0.1247 | -0.1156 | -0.0947 | -0.0781 | -0.0608 |

Entries are mean paired validation differences, Symmetric minus LoRA. Fresh
5,000-step trajectories show rank 1 crossing between steps 2,000 and 3,000.
Ranks 2, 4, and 8 favour Symmetric at each listed point, while ranks 4 and 8
show narrower gaps at step 5,000 than at step 1,000. The independent 1,000-step
screen is not merged into these curves, and the trajectories do not establish
asymptotic behavior.

![Figure 6. Fresh rank-1 confirmation trajectories.](https://github.com/xxlilloxx-cyber/federico-lolli-ai-research/blob/main/docs/assets/publication/convergence_rank1.svg)

**Figure 6.** Rank-1 validation trajectories from the independent 5,000-step
confirmation. Bands are sample SD; the mean ordering crosses between the
recorded 2,000- and 3,000-step evaluations.

## 7. Derivatives and learned spectra

These analyses characterize the local behavior of trained adapters. They
establish structural differences between LoRA and Symmetric Quadratic
Adaptation, but do not by themselves explain the observed performance
differences.

### 7.1 Local Jacobians

For the row-vector convention used throughout the study,

$$J_{\mathrm{LoRA}}=\frac{\alpha}{r}AB,$$

and

$$J_{\mathrm{Sym}}(x)=2\frac{\alpha}{r}U\operatorname{diag}(xU)P.$$

LoRA is linear at the adapter level, so its Jacobian is independent of the
input activation and can be represented by one fixed low-rank update matrix.
The Symmetric adapter is quadratic. Its Jacobian depends explicitly on the
current activation through $\operatorname{diag}(xU)$, so the effective local
transformation changes with the input and cannot generally be reduced to one
constant $\Delta W$. Double-precision tests verified both analytical
derivatives against PyTorch autograd. This is a structural result rather than a
causal explanation of performance.

### 7.2 Effective-rank analysis

| Rank | LoRA effective update | Symmetric local Jacobian |
|---:|---:|---:|
| 1 | 1.00 | 1.00 |
| 2 | 1.92 | 1.67 |
| 4 | 3.76 | 3.09 |
| 8 | 6.57 | 5.59 |

| Rank | Symmetric U | Symmetric P | Output covariance |
|---:|---:|---:|---:|
| 1 | 1.00 | 1.00 | 1.00 |
| 2 | 2.00 | 1.98 | 1.92 |
| 4 | 3.99 | 3.89 | 3.00 |
| 8 | 7.96 | 7.28 | 3.59 |

These quantities describe different mathematical objects and should not be
interpreted as interchangeable notions of “weight rank”. In particular, the
Symmetric local-Jacobian rank is not a direct analogue of a fixed LoRA
weight-update rank.

![Figure 7. Effective rank of the LoRA update and Symmetric local Jacobian.](https://github.com/xxlilloxx-cyber/federico-lolli-ai-research/blob/main/docs/assets/publication/jacobian_effective_rank.svg)

**Figure 7.** Entropy effective rank across nominal adapter ranks. The plotted
objects describe related local transformation complexity, not identical
weight-space ranks.

### 7.3 Second-order structure

For a fixed linear reduction of the adapter output by $c$,

$$H=2\frac{\alpha}{r}U\operatorname{diag}(Pc)U^T.$$

LoRA has an exactly zero adapter-only Hessian with respect to its input because
the branch is linear. Symmetric has an explicit second-order response. Mean
Symmetric Hessian Frobenius norms across seeds are 155.15, 92.13, 52.68, and
17.61 at ranks 1, 2, 4, and 8. The decline should not be interpreted as weaker
or less useful interactions at higher rank: the magnitude depends on learned
factors and the protocol's $\alpha/r$ scaling. For fixed $c$, the exact
quadratic-adapter Hessian is input independent, so repeated token activations
are not independent Hessian observations. Double-precision autograd tests
verified the analytical expression. A full Transformer-block Hessian was not
evaluated and is not reported as an empirical result.

![Figure 8. Adapter-only Hessian response.](https://github.com/xxlilloxx-cyber/federico-lolli-ai-research/blob/main/docs/assets/publication/adapter_interaction_hessian.svg)

**Figure 8.** LoRA has zero adapter-only Hessian; Symmetric has an explicit
nonzero quadratic response. This distinction does not prove a causal
performance mechanism.

## 8. Computational cost at rank 4

The matched benchmark uses batch size 1, sequence length 128, 10 warm-up
iterations, and 30 measured iterations. Both adapters contain 6,144 trainable
parameters.

| Method | Forward latency | Backward latency | Peak VRAM | Forward throughput |
|---|---:|---:|---:|---:|
| LoRA | 12.157 ± 0.243 ms | 14.805 ± 0.236 ms | 429.05 MiB | 10,529 tokens/s |
| Symmetric Quadratic | 12.184 ± 0.236 ms | 14.766 ± 0.188 ms | 430.31 MiB | 10,505 tokens/s |

Relative to LoRA, Symmetric changes mean forward latency by approximately
+0.22%, mean backward latency by -0.27%, peak allocated VRAM by +1.25 MiB, and
forward throughput by approximately -0.22%. In this measured configuration,
the methods have nearly identical computational cost. The element-wise square
adds only a small measured overhead in this implementation and hardware
configuration; the result should not be generalized to all hardware or model
scales.

## 9. Discussion

The controlled experiments reveal a more nuanced picture than a simple ranking
between LoRA and Symmetric Quadratic Adaptation. The central WikiText-2
confirmation provides consistent evidence in favor of the quadratic adapter
under the tested language-modeling protocol, whereas the AG News classification
experiment reverses that ordering. The main interpretation of the study is
therefore not that one adapter dominates the other, but that the functional
form of a low-rank correction can materially affect behavior and that this
effect depends on the task, rank, scaling policy, and optimization regime.

### 9.1 WikiText-2 result and rank dependence

The strongest evidence comes from the independent 5,000-step WikiText-2
confirmation campaign. Across ranks 1, 2, 4, and 8, Symmetric achieves lower
mean final validation loss and lower mean held-out test loss than the matched
LoRA baseline. All twelve paired seed-rank held-out comparisons also show lower
loss for Symmetric under this protocol.

The mean held-out test difference increases across the tested ranks, from
approximately -0.0125 at rank 1 to -0.0381 at rank 8. This shows that the
observed difference is not confined to a single rank. Nevertheless, the
present experiments do not establish why the magnitude changes with rank.
Under the primary protocol, increasing rank changes both the adapter capacity
and the effective scale $\alpha/r$, so rank alone cannot be treated as the sole
causal variable.

The appropriate interpretation is therefore configuration specific: the
Symmetric functional form is useful in the tested GPT-2 Small, WikiText-2,
block-0 `attn.c_proj` setting. The experiment does not establish that quadratic
adaptation must outperform linear adaptation for language modeling in general.

### 9.2 Optimization dynamics

The convergence analysis shows that training duration can materially change
the comparison. Rank 1 is particularly informative. LoRA has lower mean
validation loss at the earlier recorded checkpoints, but the ordering changes
between the 2,000- and 3,000-step evaluations. Symmetric subsequently maintains
the lower mean validation loss through the final 5,000-step checkpoint. A
comparison terminated earlier could therefore have produced a different
conclusion.

Ranks 2, 4, and 8 follow a different trajectory. Symmetric already has lower
mean validation loss at the recorded 1,000-step point. At ranks 4 and 8,
however, the magnitude of the difference narrows later in training. These
trajectories show that adapter comparisons should not be interpreted
independently of optimization duration. They do not establish asymptotic
behavior, because the study observes only the finite schedules that were
actually executed.

### 9.3 Scaling and rank

The main rank experiment fixes $\alpha$ at 4, so $\alpha/r$ decreases as rank
increases. The constant-scale ablation tests whether this scaling rule alone
accounts for the observed rank pattern. With $\alpha/r$ fixed to 1, the
qualitative 1,000-step pattern remains unchanged for seed 42: LoRA performs
better at rank 1, while Symmetric performs better at ranks 2, 4, and 8.

The magnitude does change. At rank 8, for example, the
Symmetric-minus-LoRA validation-loss difference changes from -0.1177 under the
primary scaling policy to -0.0975 under constant effective scale. This
indicates that scaling contributes to the magnitude of the observed difference
but does not, for the tested seed, fully explain the rank-dependent sign
pattern. Because the constant-scale experiment was performed for only one
seed, it cannot establish a general rank-scaling relationship.

### 9.4 Task dependence and AG News

AG News provides an important counterexample to the WikiText-2 result. Under
the matched classification protocol, LoRA achieves higher mean test accuracy,
higher macro-F1, and lower test loss. Symmetric also shows substantially
greater seed-to-seed variability.

The two experimental settings differ in several respects. WikiText-2 is a
causal language-modeling task, whereas AG News is classification and introduces
a trainable classification head. The datasets, objectives, and training
budgets also differ. The present experiments do not isolate which of these
factors causes the reversed ordering. The defensible conclusion is therefore
that the relative benefit of the quadratic correction is task and protocol
dependent. This negative counter-result is scientifically important because it
prevents the positive WikiText-2 result from being interpreted as evidence of
universal superiority.

### 9.5 Structural interpretation

The derivative analyses establish that LoRA and Symmetric belong to different
local function classes. LoRA is linear with respect to the adapter input. Its
Jacobian is input independent and its adapter-only Hessian is exactly zero. Its
correction can therefore be represented by a single fixed low-rank update.
Symmetric has an input-dependent local Jacobian and an explicit non-zero
adapter-only second derivative. Its effective local transformation changes
with the incoming activation and cannot, in general, be reduced to one constant
$\Delta W$.

The effective rank of the LoRA update, the effective rank of the Symmetric
local Jacobian, the ranks of $U$ and $P$, and the output-covariance rank describe
different mathematical objects. They should not be interpreted as
interchangeable measurements of one underlying “weight rank.” These structural
results demonstrate that the quadratic adapter has a different functional
form, but they do not establish that Jacobian variability, Hessian magnitude,
or any measured spectral property causes the observed WikiText-2 improvement.
Establishing such a causal relationship would require additional
intervention-based analysis.

### 9.6 Computational implications

The matched rank-4 benchmark shows that the explicit quadratic operation
introduces little measured practical cost in the tested configuration. Forward
latency differs from LoRA by approximately +0.22%, backward latency by
approximately -0.27%, peak allocated VRAM increases by about 1.25 MiB, and
forward throughput remains nearly unchanged.

This indicates that, in this implementation and hardware configuration, the
element-wise square can be introduced without a substantial measured
computational penalty. The benchmark should not be generalized to larger
models, higher ranks, longer sequences, larger batches, or different
accelerators without additional measurement.

### 9.7 Overall interpretation

Taken together, the experiments support the view that adapter functional form
is an important design variable alongside rank and parameter count. Two
adapters with matched trainable matrix shapes and parameter budgets can
implement substantially different local transformations and can produce
different empirical behavior across tasks and optimization regimes.

Within the controlled WikiText-2 study, Symmetric Quadratic Adaptation provides
a consistent advantage over the matched LoRA baseline. The AG News experiment
demonstrates that this advantage does not automatically transfer to another
task. The evidence therefore supports Symmetric Quadratic Adaptation as a
viable task-dependent alternative to LoRA while leaving open the broader
question of when quadratic activation-space interactions are most useful.

![Figure 9. Architectural context for related methods.](https://github.com/xxlilloxx-cyber/federico-lolli-ai-research/blob/main/docs/assets/figures/adapter-family-comparison.svg)

**Figure 9.** Original architectural comparison of LoRA, Symmetric, PERA,
QuadraNet V2, and a nonlinear bottleneck adapter. Similar visual structure does
not imply mathematical equivalence.

## 10. Limitations

This study was designed to provide controlled evidence about one specific
quadratic low-rank adapter formulation rather than a complete benchmark of
parameter-efficient fine-tuning. Several limitations therefore constrain the
generality of the conclusions.

### 10.1 Model scope

The controlled experiments use GPT-2 Small. The results do not establish that
the same behavior will occur in larger GPT-2 variants, other decoder-only
language models, encoder architectures, vision Transformers, or multimodal
systems. Model scale and architecture may alter optimization dynamics,
representation geometry, memory behavior, and the usefulness of explicit
quadratic interactions. Replication across additional model families is
required before drawing broader architectural conclusions.

### 10.2 Task and dataset scope

The main controlled evidence comes from two datasets with different
objectives: WikiText-2 causal language modeling and AG News classification.
The reversed ordering between these tasks demonstrates that task dependence
matters, but two datasets are insufficient to characterize the conditions
under which quadratic activation-space adaptation is beneficial. Additional
generative, classification, reasoning, and domain-specific tasks would be
required to establish broader transfer behavior.

### 10.3 Experimental scope

The central WikiText-2 comparison uses three seeds, four ranks, one primary
controlled adapter placement, one optimizer family, and a fixed training
schedule. Three seeds provide repeated evidence but remain a small basis for
broad statistical inference. The study therefore reports means, sample
standard deviations, and paired differences without claiming general
statistical significance over a larger population of training runs.

The principal controlled placement is zero-based block 0 `attn.c_proj`.
Earlier placement and multilayer studies provide exploratory context, but they
do not constitute a fully controlled evaluation of every Transformer layer
and projection.

### 10.4 Scaling and hyperparameters

The main rank study fixes $\alpha$ at 4, causing $\alpha/r$ to vary with rank.
The constant-scale ablation examines this issue only for seed 42. The result
therefore suggests that scaling alone does not explain the observed rank
pattern for that seed, but it does not provide a complete multi-seed
characterization of the interaction between rank and scale.

The study also does not perform an exhaustive independent hyperparameter
optimization for each adapter. Alternative learning rates, schedules,
initialization policies, regularization settings, or scales could alter the
relative behavior.

### 10.5 Mechanistic interpretation

The Jacobian and Hessian analyses establish analytical and measured structural
differences between LoRA and Symmetric, but they do not demonstrate that these
differences cause the WikiText-2 performance gap. The adapter-only Hessian is
analytically tractable. A full Transformer-block Hessian was not evaluated
because it would require a separate scalar-reduction protocol and substantially
greater computational complexity. More detailed intervention studies would be
needed to connect local second-order structure directly to task performance.

### 10.6 Computational-cost scope

The computational benchmark covers one rank-4 configuration with batch size 1,
sequence length 128, and one recorded hardware/software environment. The
near-equal latency and small memory difference observed here should therefore
be interpreted as an implementation-specific measurement, not as a universal
statement about computational efficiency. Scaling behavior for larger models,
higher ranks, longer sequences, larger batches, and different accelerators
remains unmeasured.

### 10.7 Related-work scope

LoRA is the only related method reproduced as a direct controlled baseline.
DoRA, MoRA, HiRA, LoRAN, PERA, QuadraNet V2, and nonlinear adapter methods are
used to establish mathematical and architectural context, not as
experimentally reproduced comparisons under the same protocol. Published
results from those works are therefore not directly compared numerically with
the present experiments.

The literature review also does not establish universal mathematical novelty
relative to every prior quadratic neural architecture. The contribution should
instead be understood as the formulation, controlled evaluation, and
structural characterization of this specific activation-space quadratic
adapter.

### 10.8 Generalization

The results establish reproducible behavior within the tested conditions. They
do not establish state-of-the-art performance, universal superiority over
LoRA, broad transfer across model families, or a general causal advantage of
second-order adaptation. Broader replication across larger models, additional
tasks, additional placements, and independently repeated scaling conditions is
required to determine how widely the observed behavior generalizes.

## 11. Conclusion

This study investigated whether a simple quadratic transformation in a
low-dimensional activation space can provide a useful alternative to
conventional linear low-rank adaptation. The results show that the Symmetric
Quadratic Adapter is a practical and computationally lightweight mechanism for
introducing explicit second-order interactions into a frozen Transformer,
while retaining the same low-rank parameter structure used for comparison with
LoRA.

The strongest evidence comes from the controlled WikiText-2 experiments.
Across 24 independent 5,000-step runs, covering ranks 1, 2, 4, and 8 and three
random seeds, Symmetric achieved lower mean final validation loss and lower
mean held-out test loss than LoRA at every tested rank. The difference was
largest at rank 8 in the final test evaluation, where mean loss decreased from
3.7319 ± 0.0035 for LoRA to 3.6938 ± 0.0044 for Symmetric. Importantly, the
rank-1 convergence trajectories also show that the comparison depends on
optimization time: LoRA initially performs better, while Symmetric overtakes
it between the recorded 2,000- and 3,000-step checkpoints. These observations
suggest that the behavior of quadratic adaptation cannot be characterized
solely by its final parameter budget; rank and training duration also matter.

The scaling experiment provides an additional qualification. Holding the
effective scale α/r constant changes the magnitude of the differences but
preserves, for the tested seed, the qualitative pattern observed at 1,000
steps: LoRA performs better at rank 1, whereas Symmetric performs better at
ranks 2, 4, and 8. This indicates that the observed rank trend is not explained
exclusively by the changing α/r factor of the primary protocol, although the
single-seed nature of this ablation prevents a broader conclusion.

The downstream AG News experiment provides an important counterexample to any
claim of general superiority. Under the matched classification protocol, LoRA
achieved higher mean accuracy and macro-F1 than Symmetric. Mean test accuracy
was 0.8201 ± 0.0051 for LoRA and 0.8021 ± 0.0240 for Symmetric, with
substantially greater variability for the quadratic adapter. The combined
evidence therefore points to a task-dependent trade-off rather than a
universally better adaptation mechanism.

The structural analyses help clarify why the two adapters are genuinely
different. LoRA remains linear with respect to the adapter input and can be
represented by a fixed low-rank weight update. Symmetric instead produces an
input-dependent local Jacobian and possesses explicit non-zero adapter-level
second derivatives. Its behavior therefore cannot, in general, be reduced to
a single constant ΔW. Spectral measurements further show that the effective
rank of the local Jacobian, the ranks of the learned factors, and the
output-covariance rank describe distinct properties of the learned
transformation and should not be interpreted as interchangeable notions of
weight rank.

These additional expressive properties do not appear to introduce a
substantial computational penalty in the measured configuration. At rank 4,
forward latency and backward latency were almost identical to LoRA, while peak
memory increased by only about 1.25 MiB. In this specific implementation,
explicit quadratic activation interactions can therefore be introduced at
approximately the same practical computational scale as the matched linear
adapter.

Taken together, the results support a narrower but more useful conclusion than
a claim of overall superiority: **low-rank quadratic activation-space
adaptation is a viable design point for parameter-efficient Transformer
adaptation, and in the controlled WikiText-2 setting studied here it
consistently improved final language-model loss relative to the matched LoRA
baseline.** At the same time, the AG News results demonstrate that this
advantage does not automatically transfer across tasks.

The study also suggests that the functional form of a low-rank adapter matters
in addition to its parameter count. Two adapters with matched trainable
dimensions can induce substantially different local transformations: one
linear and input independent, the other quadratic and activation dependent.
This motivates treating adapter architecture—not only rank and parameter
budget—as an experimental variable in parameter-efficient fine-tuning.

The current evidence remains limited to GPT-2 Small, a restricted set of
adapter placements, two datasets, three seeds for the principal controlled
experiments, and a single seed for the scaling ablation. The experiments
therefore do not establish state-of-the-art performance, statistical
significance across broad populations of runs, universal transfer to other
Transformer families, or a causal relationship between second-order structure
and improved language-model performance. Nor do they establish that quadratic
low-rank adaptation itself is mathematically novel relative to the full prior
literature.

The results instead establish a reproducible empirical starting point. Future
work across larger models, additional tasks, broader placements, and
independently replicated scaling configurations can determine whether the
advantages observed here represent a broader property of activation-space
quadratic adaptation or a characteristic of the specific regimes studied.

## Appendix B. Historical evidence

The historical experiments preceded the final controlled protocols and were
used to explore adapter formulation, training duration, placement, and
distribution of a fixed parameter budget across layers. Because these campaigns
were not conducted under one uniform protocol, they are reported here as
exploratory evidence and are not pooled with the controlled rank-confirmation
or AG News results.

### B.1 Adapter-family exploration

| Adapter | Mean best-recorded validation loss |
|---|---:|
| LoRA | 3.5732 |
| Element-wise quadratic | 3.6351 |
| Signed quadratic | 3.6257 |
| Interaction | 3.5581 |
| Symmetric | 3.5526 |
| Linear+Quadratic | 3.5302 |

**Historical 500-step adapter-family screen.** Values are verified three-seed
means of best-recorded validation loss. These exploratory runs are not a
matched final benchmark.

### B.2 Longer exploratory training

| Steps | LoRA | Symmetric | Linear+Quadratic |
|---:|---:|---:|---:|
| 1,000 | 3.5491 | 3.4770 | 3.4928 |
| 2,000 | 3.4598 | 3.3784 | 3.4360 |

**Historical rank-4 block-0 insertion-point summaries.** Values are verified
three-seed means of best-recorded validation loss. They are not final-checkpoint
test results.

### B.3 Placement and depth exploration

A seed-42, 20-step attention scan evaluated all twelve GPT-2 blocks. Symmetric
recorded lower validation loss at 11 of 12 blocks; block 1 was the exception.
At block 2, the largest short-run separation was 3.8243 for LoRA versus 3.8030
for Symmetric. Separate attention-versus-MLP screens found nearly tied
single-module MLP results, while the combined block-0 MLP branch used 15,360
parameters and therefore was not directly budget matched to the 6,144-parameter
attention branches. These short screens established that placement merited
further control; they do not define a stable depth ranking.

### B.4 Fixed-budget multilayer exploration

| Budget | Geometry | Local rank | LoRA | Symmetric |
|---:|---|---:|---:|---:|
| 6,144 | Concentrated [2] | 4 | 3.5039 | 3.3745 |
| 6,144 | Two layers [0,11] | 2 | 3.3661 | 3.3730 |
| 6,144 | Four layers [0,3,7,11] | 1 | 3.3199 | 3.3794 |
| 12,288 | Concentrated [2] | 8 | 3.5013 | 3.3391 |
| 12,288 | Two layers [0,11] | 4 | 3.3157 | 3.2390 |
| 12,288 | Four layers [0,3,7,11] | 2 | 3.2578 | 3.2356 |

**Historical fixed-budget multilayer study.** Values are verified three-seed
means of best-recorded validation loss after 2,000-step runs. Distributing a
fixed budget changes both placement and local rank, so these results are not
merged with the later single-placement rank-confirmation experiment.

### B.5 Role of the historical experiments

These exploratory campaigns served to identify promising adapter structures
and experimental questions. They motivated the later controlled studies but do
not carry the same evidential weight because their protocols differ in
placement, duration, rank allocation, or evaluation definition. The central
conclusions of this manuscript are therefore based on the controlled
WikiText-2 confirmation, AG News classification, scaling, convergence,
mechanism, and computational-cost experiments.

## References

1. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
2. Liu et al. *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML 2024.
3. Jiang et al. *MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning.* 2024.
4. Huang et al. *HiRA: Parameter-Efficient Hadamard High-Rank Adaptation.* ICLR 2025.
5. Li, Song, and Hou. *LoRAN.* Findings of EMNLP 2024.
6. Zhang et al. *Polynomial Expansion Rank Adaptation.* Findings of ACL 2026. DOI: 10.18653/v1/2026.findings-acl.650.
7. Xu et al. *QuadraNet V2.* WACV 2026.
8. Houlsby et al. *Parameter-Efficient Transfer Learning for NLP.* ICML 2019.

The complete verified bibliography and original-source URLs are in
`comparisons/papers/REFERENCES.md`.
