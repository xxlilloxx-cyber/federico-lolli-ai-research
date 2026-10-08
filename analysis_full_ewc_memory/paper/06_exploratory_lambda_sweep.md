# 6. Exploratory EWC Lambda Sweep

## 6.1 Sweep design

The exploratory stage evaluates

\[
\lambda\in\{0,1,10,30,50,100,300\}
\]

for LoRA, Symmetric, and Combined using seed 42.

The analysis is performed in the stability–plasticity plane defined by

\[
F_{\mathrm{mean}}
=
\frac{
F_{A\rightarrow B}
+
F_{A\rightarrow C}
+
F_{B\rightarrow C}
}{3}
\]

and

\[
L_{\mathrm{new}}^{\mathrm{acq}}
=
\frac{
L_B^{\mathrm{acq}}
+
L_C^{\mathrm{acq}}
}{2}.
\]

Both quantities are minimized.

![Exploratory stability-plasticity sweep](../analysis_full_ewc_memory/figures_main/figure_1_exploratory_stability_plasticity.svg)

Across all three architectures, increasing \(\lambda\) reduces forgetting while increasing new-fact acquisition NLL. The sweep therefore reveals a continuous stability–plasticity trade-off rather than a uniformly beneficial effect of stronger consolidation.

## 6.2 LoRA

LoRA moves from

\[
F_{\mathrm{mean}}=4.2868
\]

and

\[
L_{\mathrm{new}}^{\mathrm{acq}}=0.5044
\]

at

\[
\lambda=0
\]

to

\[
F_{\mathrm{mean}}=2.1935
\]

and

\[
L_{\mathrm{new}}^{\mathrm{acq}}=1.0628
\]

at

\[
\lambda=10.
\]

The final mean NLL improves from

\[
3.7514
\]

to

\[
2.4909.
\]

At

\[
\lambda=1,
\]

final exact match remains equal to the \(\lambda=0\) value,

\[
0.0833,
\]

while forgetting decreases to

\[
3.9427.
\]

At

\[
\lambda=10,
\]

exact match remains non-zero,

\[
0.0139,
\]

whereas at

\[
\lambda=30
\]

it reaches zero.

Both geometric knee criteria select

\[
\lambda=10.
\]

The LoRA confirmation candidates are therefore

\[
\lambda\in\{1,10\}.
\]

## 6.3 Symmetric

Symmetric provides the strongest unregularized acquisition:

\[
L_{\mathrm{new}}^{\mathrm{acq}}=0.1232
\]

at

\[
\lambda=0,
\]

with final exact match

\[
0.3333.
\]

This operating point is associated with

\[
F_{\mathrm{mean}}=5.6856.
\]

At

\[
\lambda=10,
\]

the method reaches

\[
F_{\mathrm{mean}}=3.8739,
\]

\[
L_{\mathrm{new}}^{\mathrm{acq}}=0.3682,
\]

and final exact match

\[
0.1250.
\]

At

\[
\lambda=30,
\]

the corresponding values are

\[
F_{\mathrm{mean}}=2.7536
\]

and

\[
L_{\mathrm{new}}^{\mathrm{acq}}=0.9028.
\]

The final mean NLL reaches

\[
2.3002,
\]

the lowest value observed in the exploratory sweep.

Exact match is zero at this point.

Both geometric compromise criteria select

\[
\lambda=30,
\]

while

\[
\lambda=10
\]

is retained because it preserves substantially stronger acquisition and non-zero exact generation.

The Symmetric confirmation candidates are therefore

\[
\lambda\in\{10,30\}.
\]

## 6.4 Combined

Combined exhibits intermediate behavior.

At

\[
\lambda=0,
\]

\[
F_{\mathrm{mean}}=5.2562,
\]

\[
L_{\mathrm{new}}^{\mathrm{acq}}=0.4037.
\]

At

\[
\lambda=10,
\]

\[
F_{\mathrm{mean}}=3.3880,
\]

\[
L_{\mathrm{new}}^{\mathrm{acq}}=0.6870.
\]

At

\[
\lambda=30,
\]

\[
F_{\mathrm{mean}}=2.5690,
\]

\[
L_{\mathrm{new}}^{\mathrm{acq}}=1.2213,
\]

with final mean NLL

\[
2.3786
\]

and final exact match

\[
0.0139.
\]

Exact match reaches zero at

\[
\lambda=50.
\]

The endpoint-chord criterion selects

\[
\lambda=10,
\]

whereas normalized distance to the ideal point selects

\[
\lambda=30.
\]

Both values are therefore retained for confirmation.

## 6.5 Cross-method comparison

The unregularized configurations occupy distinct positions in the stability–plasticity plane:

| Method | \(F_{\mathrm{mean}}\) | \(L_{\mathrm{new}}^{\mathrm{acq}}\) |
|---|---:|---:|
| LoRA | 4.2868 | 0.5044 |
| Symmetric | 5.6856 | 0.1232 |
| Combined | 5.2562 | 0.4037 |

LoRA begins from the most stable configuration, whereas Symmetric begins from the most plastic one. Combined lies between them.

At shared values of \(\lambda\), Combined does not simultaneously achieve lower forgetting and lower acquisition NLL than both LoRA and Symmetric.

The exploratory data therefore provide no evidence that combining linear and quadratic branches yields systematic dominance in the primary trade-off plane.

## 6.6 Final balanced memory

The final mean NLL,

\[
L_{\mathrm{final}}^{\mathrm{mean}},
\]

is non-monotonic in \(\lambda\).

The minimum sampled values are

\[
\text{LoRA}: \quad
\lambda=10,\qquad
L_{\mathrm{final}}^{\mathrm{mean}}=2.4909,
\]

\[
\text{Symmetric}: \quad
\lambda=30,\qquad
L_{\mathrm{final}}^{\mathrm{mean}}=2.3002,
\]

and

\[
\text{Combined}: \quad
\lambda=30,\qquad
L_{\mathrm{final}}^{\mathrm{mean}}=2.3786.
\]

Thus, the strongest balanced-memory operating points occur at intermediate regularization rather than at the highest tested \(\lambda\).

## 6.7 High-regularization regime

At

\[
\lambda\in\{100,300\},
\]

all methods continue to reduce forgetting but exhibit substantial degradation in acquisition.

At

\[
\lambda=300,
\]

\[
L_{\mathrm{new}}^{\mathrm{acq}}
=
3.0840
\]

for LoRA,

\[
2.6767
\]

for Symmetric, and

\[
2.9827
\]

for Combined.

Final exact match is zero for all three methods.

These configurations therefore define an over-constrained regime in which additional stability is obtained at excessive plasticity cost.

## 6.8 Pareto structure and candidate selection

All sampled points are Pareto-nondominated within their respective adapter families because reductions in forgetting are consistently accompanied by increased acquisition NLL.

Pareto membership therefore does not identify a unique operating point.

Two geometric criteria are used to characterize compromise regions: normalized distance to the ideal point and maximum distance from the endpoint chord.

The resulting selections are

\[
\text{LoRA}: \lambda=10,
\]

\[
\text{Symmetric}: \lambda=30,
\]

and

\[
\text{Combined}: \lambda\in\{10,30\},
\]

depending on the criterion.

These selections are descriptive and depend on the sampled lambda grid and metric normalization.

## 6.9 Confirmation grid

The exploratory stage yields the following candidate configurations:

| Method | Confirmation \(\lambda\) values |
|---|---:|
| LoRA | \(0,1,10\) |
| Symmetric | \(0,10,30\) |
| Combined | \(0,10,30\) |

The \(\lambda=0\) control is retained for all methods.

The selected configurations are evaluated across seeds

\[
42,\qquad123,\qquad456
\]

in the multi-seed confirmation stage presented in the following section.
