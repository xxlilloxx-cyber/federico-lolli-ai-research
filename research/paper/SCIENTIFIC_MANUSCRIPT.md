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

LoRA learns a low-rank weight update. DoRA decomposes weight magnitude and
direction. MoRA and HiRA increase the effective rank of weight-space updates by
different parameterizations. LoRAN applies a nonlinear transformation to a
low-rank weight update. PERA expands low-rank factors polynomially in parameter
space. QuadraNet V2 is the closest verified quadratic-form precedent: a
single-output factorized quadratic form is related to our per-output equation,
but complete multi-output equivalence has not been established because our
outputs share projection directions and use output-specific signed coefficients.
These methods are literature context and were not head-to-head baselines here.
Detailed formula and methodology comparisons are maintained in
`comparisons/`.

## 3. Mathematical formulation

We use row vectors. For input $x\in\mathbb{R}^{1\times d}$, frozen weight
$W\in\mathbb{R}^{d\times d_{out}}$, and frozen bias $b$, the projection is

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

Thus $Q_j$ is symmetric with rank at most $r$. Directions $u_k$ are shared
across outputs, while signed $P_{kj}$ coefficients can yield an indefinite
$Q_j$. The expansion $(ax_1+bx_2)^2=a^2x_1^2+2abx_1x_2+b^2x_2^2$ makes the
cross interaction explicit. Output factors $B$ and $P$ are zero initialized,
so both corrections start at zero. At initialization the input factor receives
zero loss gradient until the output factor becomes nonzero.

For the controlled rank study $\alpha=4$, making $\alpha/r$ equal to 4, 2, 1,
and 0.5 for ranks 1, 2, 4, and 8. Rank, parameter count, and effective scale
therefore change together in the primary factorial.

## 4. Experimental methodology

The backbone is frozen GPT-2 Small. Controlled rank experiments insert one
adapter beside the input of zero-based block 0 `attn.c_proj`. The frozen
projection and adapter run in parallel; their outputs are added before GPT-2's
attention residual dropout and residual connection. WikiText-2 text is
tokenized with the GPT-2 tokenizer and packed into consecutive 128-token blocks
for causal language modelling. Runs use batch size 1, gradient accumulation 4,
AdamW at $3\times10^{-4}$, clipping 1.0, no scheduler, and seeds 42, 123, and
456. Fresh validation is kept separate from held-out test evaluation of the
final checkpoint.

The downstream experiment uses AG News with 4,096 training examples, a fixed
1,000-example stratified validation subset, and the original test split. LoRA
and Symmetric use the same GPT-2 backbone, tokenizer, placement, rank 4,
training schedule, data order by paired seed, and 9,216 trainable parameters
including the classification head.

## 5. Controlled results

### 5.1 WikiText-2 rank confirmation

| Rank | LoRA final validation | Symmetric final validation | LoRA test | Symmetric test |
|---:|---:|---:|---:|---:|
| 1 | 3.5734 ± 0.0051 | 3.5314 ± 0.0207 | 3.9256 ± 0.0009 | 3.9131 ± 0.0042 |
| 2 | 3.4773 ± 0.0057 | 3.4340 ± 0.0249 | 3.8492 ± 0.0025 | 3.8305 ± 0.0144 |
| 4 | 3.3483 ± 0.0093 | 3.3101 ± 0.0011 | 3.7656 ± 0.0054 | 3.7429 ± 0.0024 |
| 8 | 3.3054 ± 0.0117 | 3.2446 ± 0.0096 | 3.7319 ± 0.0035 | 3.6938 ± 0.0044 |

Values are means ± sample SD over three seeds. Symmetric-minus-LoRA paired test
differences are -0.0125, -0.0187, -0.0227, and -0.0381. The conclusion is
restricted to this backbone, placement, dataset, schedule, and scaling policy.

### 5.2 AG News

LoRA achieves test accuracy 0.8201 ± 0.0051 and macro-F1 0.8156 ± 0.0050.
Symmetric achieves 0.8021 ± 0.0240 and 0.7929 ± 0.0315. LoRA is higher on
average, and Symmetric is higher on only one of three matched seeds. This
negative result is central: the WikiText-2 ordering does not transfer to the
tested classification protocol.

## 6. Scaling ablation and convergence

The seed-42, 1,000-step constant-effective-scale ablation sets $\alpha=r$.
The method differences retain the same signs as the fixed-$\alpha$ screen:
LoRA is lower at rank 1, while Symmetric is lower at ranks 2, 4, and 8. At rank
8 the gap changes from -0.1177 to -0.0975. One seed is insufficient for a
stable interaction estimate; scaling influences magnitude and remains a
confound in the primary rank comparison.

Fresh 5,000-step trajectories show rank 1 crossing from LoRA-favoured to
Symmetric-favoured between steps 2,000 and 3,000. The independent 1,000-step
screen is not merged into these curves.

## 7. Mechanism analysis

LoRA has an input-independent adapter Jacobian
$J_L=(\alpha/r)AB$. Symmetric has

$$J_S(x)=2\frac{\alpha}{r}U\operatorname{diag}(xU)P.$$

For a fixed linear output reduction $c$, its Hessian is

$$H_S=2\frac{\alpha}{r}U\operatorname{diag}(Pc)U^T,$$

while LoRA's adapter-only Hessian is zero. Analytical Jacobians and Hessians
match autograd in tests. Learned LoRA effective-update entropy rank grows from
1.00 to 6.57 across nominal ranks 1 to 8. Symmetric mean local-Jacobian
effective rank grows from 1.00 to approximately 5.59. These are not the same
object and should not be described as a higher Symmetric weight rank.

The reduced Symmetric Hessian has nonzero learned Frobenius norms, whereas
LoRA's is zero. Because the exact quadratic adapter Hessian is input independent
under this linear reduction, repeated tokens are not independent Hessian
measurements. A full-block Hessian analysis was blocked and is omitted from the
results.

## 8. Computational cost

At rank 4, batch 1, and length 128, LoRA and Symmetric use 6,144 trainable
adapter parameters. Mean forward latency is 12.157 ± 0.243 ms versus 12.184 ±
0.236 ms; backward latency is 14.805 ± 0.236 ms versus 14.766 ± 0.188 ms; peak
allocated VRAM is 429.05 versus 430.31 MiB. Forward throughput is approximately
10.53k versus 10.51k tokens/s. This is consistent with 128 tokens per roughly
12 ms; a prior shorthand of 10.5 tokens/s was a unit omission.

## 9. Discussion and limitations

The results establish that the quadratic adapter trains stably at small budgets
and can outperform matched LoRA on controlled WikiText-2 language modelling.
They also show that this ordering is task dependent. The derivative analyses
verify a structural difference but do not isolate its causal contribution.
Limitations include GPT-2 Small, two datasets, three main seeds, one optimizer
schedule, one focused controlled insertion point, short context, a single-seed
scaling ablation, no second model family, and no completed full-block Hessian.

## 10. Conclusion

Under the controlled WikiText-2 protocol, explicit symmetric quadratic
corrections provide useful adaptation capacity and lower mean held-out loss at
all tested ranks. Under controlled AG News classification, LoRA performs better
on average. The strongest supported interpretation is an interaction among
task, local function class, rank, scaling, placement, and optimization time,
not universal superiority.

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
