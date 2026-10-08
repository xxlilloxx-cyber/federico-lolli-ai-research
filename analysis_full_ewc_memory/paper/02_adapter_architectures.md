# 2. Adapter Architectures

## 2.1 Common formulation

All experiments use a frozen GPT-2 Small backbone [3]. Let

\[
x\in\mathbb{R}^{1\times d}
\]

denote the activation entering a frozen projection

\[
W\in\mathbb{R}^{d\times d_{\mathrm{out}}}.
\]

The adapted transformation is

\[
y=xW+b+\Delta y(x),
\]

where only the parameters defining

\[
\Delta y(x)
\]

are optimized.

Adapters are inserted at zero-based Transformer block 0, `attn.c_proj`. For GPT-2 Small,

\[
d=d_{\mathrm{out}}=768.
\]

All architectures are matched at

\[
N_{\mathrm{trainable}}=6144
\]

trainable parameters.

The comparison therefore holds backbone, adapter placement, optimization protocol, and trainable parameter count fixed while varying the functional form of the adaptation.

## 2.2 LoRA

Low-Rank Adaptation (LoRA) parameterizes the update to a frozen projection through two low-rank factors [2]:

\[
\Delta y_L
=
\frac{\alpha}{r}(xA)B,
\]

with

\[
A\in\mathbb{R}^{d\times r},
\qquad
B\in\mathbb{R}^{r\times d_{\mathrm{out}}}.
\]

Equivalently,

\[
\Delta y_L
=
x\Delta W_L,
\]

where

\[
\Delta W_L
=
\frac{\alpha}{r}AB
\]

and

\[
\operatorname{rank}(\Delta W_L)\leq r.
\]

The trainable parameter count is

\[
N_L=r(d+d_{\mathrm{out}}).
\]

For

\[
d=d_{\mathrm{out}}=768,
\qquad
r=4,
\]

this gives

\[
N_L=6144.
\]

The experiments use

\[
\alpha=4.
\]

Because the adapter contribution is linear in \(x\), its Jacobian is

\[
J_L
=
\frac{\partial\Delta y_L}{\partial x}
=
\frac{\alpha}{r}AB,
\]

which is independent of the input activation. The adapter-only Hessian therefore satisfies

\[
H_L=0.
\]

## 2.3 Symmetric quadratic adapter

The Symmetric adapter replaces the input-linear low-rank correction with a factorized quadratic transformation:

\[
\Delta y_S
=
\frac{\alpha}{r}
\left[(xU)\odot(xU)\right]P,
\]

where

\[
U\in\mathbb{R}^{d\times r},
\qquad
P\in\mathbb{R}^{r\times d_{\mathrm{out}}}.
\]

Its trainable parameter count is

\[
N_S=r(d+d_{\mathrm{out}}),
\]

which gives

\[
N_S=6144
\]

for

\[
r=4.
\]

For output coordinate \(j\),

\[
\Delta y_{S,j}
=
\frac{\alpha}{r}
\sum_{k=1}^{r}
P_{kj}(xu_k)^2,
\]

where \(u_k\) denotes column \(k\) of \(U\).

This can be expressed as a quadratic form,

\[
\Delta y_{S,j}
=
xQ_jx^T,
\]

with

\[
Q_j
=
\frac{\alpha}{r}
\sum_{k=1}^{r}
P_{kj}u_ku_k^T.
\]

Since

\[
(u_ku_k^T)^T=u_ku_k^T,
\]

each effective matrix satisfies

\[
Q_j^T=Q_j,
\]

and

\[
\operatorname{rank}(Q_j)\leq r.
\]

The factorization therefore represents a low-rank symmetric quadratic form without explicitly constructing a full quadratic matrix for each output dimension.

## 2.4 Differential structure

The distinction between LoRA and Symmetric can be expressed directly through their local differential structure.

For Symmetric,

\[
J_S(x)
=
2\frac{\alpha}{r}
U\,\operatorname{diag}(xU)\,P.
\]

Unlike LoRA,

\[
J_S(x)
\]

depends on the input activation.

For output coordinate \(j\),

\[
H_{S,j}
=
2Q_j.
\]

Thus, at the adapter level,

\[
H_L=0,
\]

whereas the trained Symmetric branch can satisfy

\[
H_{S,j}\neq0.
\]

The two adapters therefore differ structurally even under identical parameter budgets: LoRA provides an input-linear low-rank correction [2], whereas Symmetric provides an input-dependent second-order correction.

## 2.5 Combined linear-quadratic adapter

The Combined architecture sums independent linear and quadratic branches:

\[
\Delta y_C
=
\frac{\alpha_L}{r_L}(xA)B
+
\frac{\alpha_S}{r_S}
\left[(xU)\odot(xU)\right]P.
\]

The continual-learning configuration uses

\[
r_L=2,
\qquad
r_S=2,
\]

and

\[
\alpha_L=\alpha_S=4.
\]

The total number of trainable parameters is therefore

\[
N_C
=
(r_L+r_S)(d+d_{\mathrm{out}})
=
6144.
\]

Branch-wise scaling gives

\[
\frac{\alpha_L}{r_L}
=
\frac{\alpha_S}{r_S}
=
2.
\]

The Jacobian is

\[
J_C(x)
=
J_L+J_S(x),
\]

while the adapter-level Hessian contribution is entirely determined by the quadratic branch:

\[
H_C=H_S.
\]

Combined therefore represents a mixed linear–quadratic adaptation under the same total parameter budget as the single-family adapters.

## 2.6 Matched-budget comparison

The evaluated configurations are

| Method | Rank configuration | Scaling | Trainable parameters |
|---|---:|---:|---:|
| LoRA | \(r=4\) | \(\alpha=4\) | 6,144 |
| Symmetric | \(r=4\) | \(\alpha=4\) | 6,144 |
| Combined | \(r_L=2,\ r_S=2\) | \(\alpha_L=\alpha_S=4\) | 6,144 |

Although parameter count is matched, the corresponding function classes are not equivalent.

For LoRA,

\[
\operatorname{rank}(\Delta W_L)\leq r,
\]

whereas for Symmetric,

\[
\operatorname{rank}(Q_j)\leq r.
\]

Equal rank therefore has different functional interpretations across the two parameterizations.

The experimental comparison should consequently be interpreted as parameter-budget matching rather than functional-capacity matching.

## 2.7 Architectural hypothesis

The three architectures define different local adaptation geometries:

\[
\text{LoRA}
\rightarrow
\text{input-linear low-rank correction},
\]

\[
\text{Symmetric}
\rightarrow
\text{input-dependent quadratic correction},
\]

\[
\text{Combined}
\rightarrow
\text{linear + quadratic correction}.
\]

The working hypothesis is that these differences affect the balance between acquisition and retention under sequential learning.

In particular, the experiments test whether LoRA occupies a comparatively stability-oriented regime, whether Symmetric provides greater plasticity, and whether Combined produces an intermediate or synergistic behavior.

These are empirical hypotheses and are not implied by the architectural formulation alone.

## 2.8 Relevance to EWC

Elastic Weight Consolidation constrains movement in parameter space through

\[
\mathcal L_{\mathrm{EWC}}
=
\frac{\lambda}{2}
\sum_i
F_i
(\theta_i-\theta_i^\ast)^2
\]

[1].

However, LoRA, Symmetric, and Combined define different mappings

\[
\theta\mapsto\Delta y(x).
\]

Consequently, identical values of

\[
\lambda
\]

or identical parameter displacements need not correspond to identical functional constraints across architectures.

This motivates the joint analysis of adapter geometry and consolidation strength developed in the following sections.
