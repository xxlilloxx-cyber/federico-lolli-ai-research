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

LoRA adapts a frozen weight by learning a low-rank linear correction. This is a
strong baseline because it uses few parameters, introduces a controlled rank
bottleneck, and is widely reproducible. A local linear adapter cannot itself
represent second-order interactions among coordinates of its input activation.
We ask whether an equally sized quadratic branch can provide useful adaptation
capacity, and how its behavior changes with task, rank, placement, training
duration, and scaling.

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

### 5.2 AG News

| Method | Test accuracy | Test macro-F1 | Test loss |
|---|---:|---:|---:|
| LoRA | 0.8201 ± 0.0051 | 0.8156 ± 0.0050 | 0.5445 ± 0.0542 |
| Symmetric | 0.8021 ± 0.0240 | 0.7929 ± 0.0315 | 0.6847 ± 0.1919 |

LoRA is higher on average for accuracy and macro-F1, lower on test loss, and
favoured on two of three matched seeds. This negative result is central: the
WikiText-2 ordering does not transfer to the tested classification protocol.

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

## 9. Discussion and limitations

The results establish that the quadratic adapter trains stably at small budgets
and can outperform matched LoRA on controlled WikiText-2 language modelling.
They also show that this ordering is task dependent. The derivative analyses
verify a structural difference but do not isolate its causal contribution.
Limitations include GPT-2 Small, two datasets, three main seeds, one optimizer
schedule, one focused controlled insertion point, short context, a single-seed
scaling ablation, no second model family, and no completed full-block Hessian.

## 10. Conclusion

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
