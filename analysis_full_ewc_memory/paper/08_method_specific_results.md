# 8. Method-Specific Results

## 8.1 LoRA

LoRA exhibits the most stability-oriented response among the three adapter families.

In the exploratory sweep, the unregularized configuration starts from

\[
F_{\mathrm{mean}}=4.2868
\]

and

\[
L_{\mathrm{new}}^{\mathrm{acq}}=0.5044.
\]

Increasing EWC strength progressively reduces forgetting, with the main compromise region occurring around

\[
\lambda=10.
\]

The multi-seed confirmation preserves this behavior: LoRA reacts measurably already at

\[
\lambda=1,
\]

and reaches a substantially more stable regime at

\[
\lambda=10.
\]

This response indicates that relatively modest EWC coefficients are sufficient to constrain the low-rank linear update introduced by LoRA [2].

The corresponding cost is reduced plasticity. As \(\lambda\) increases, acquisition NLL rises and exact-generation performance deteriorates.

LoRA therefore favors operating points that prioritize retention at comparatively low regularization strength.

## 8.2 Symmetric

Symmetric occupies the most plastic region of the comparison.

At

\[
\lambda=0,
\]

the exploratory run achieves

\[
L_{\mathrm{new}}^{\mathrm{acq}}=0.1232,
\]

the lowest acquisition NLL among the three unregularized adapters, but also exhibits

\[
F_{\mathrm{mean}}=5.6856,
\]

the highest forgetting value.

Moderate EWC substantially changes this behavior.

At

\[
\lambda=10,
\]

forgetting decreases while the adapter retains comparatively strong acquisition. Increasing to

\[
\lambda=30
\]

moves the model toward a substantially more stable regime.

The multi-seed confirmation preserves this qualitative response and supports a broader plastic regime than that observed for LoRA.

The result is consistent with the functional structure introduced in Section 2. The Symmetric adapter defines an input-dependent quadratic correction rather than an input-linear update. However, the experiments establish only an empirical association between this geometry and the observed stability–plasticity behavior; they do not prove that quadratic structure alone causes the difference.

Within the evaluated configurations, Symmetric provides the strongest evidence that a more plastic unregularized adapter can benefit substantially from intermediate consolidation.

## 8.3 Combined

Combined incorporates both LoRA-like linear and Symmetric quadratic branches under the same total trainable parameter budget.

Its empirical behavior is generally intermediate between the two individual architectures.

At

\[
\lambda=0,
\]

the exploratory configuration yields

\[
F_{\mathrm{mean}}=5.2562
\]

and

\[
L_{\mathrm{new}}^{\mathrm{acq}}=0.4037.
\]

Increasing EWC strength reduces forgetting while progressively weakening acquisition.

The most relevant operating region lies around

\[
\lambda\in\{10,30\}.
\]

The multi-seed results confirm that both settings improve retention relative to the unregularized configuration.

However, Combined does not consistently dominate both LoRA and Symmetric in the primary stability–plasticity plane.

The simultaneous presence of linear and quadratic branches therefore does not, under the tested equal-budget rank allocation,

\[
r_L=r_S=2,
\]

produce clear evidence of synergistic superiority.

## 8.4 Cross-method comparison

The three methods occupy different regions of the stability–plasticity space even before EWC is applied.

The qualitative ordering at low regularization is

\[
\text{LoRA}
\rightarrow
\text{more stability-oriented},
\]

\[
\text{Symmetric}
\rightarrow
\text{more plasticity-oriented},
\]

with Combined generally between them.

EWC shifts every architecture toward greater stability, but the required regularization scale differs.

LoRA reaches a constrained regime at lower \(\lambda\), whereas Symmetric remains comparatively plastic under stronger consolidation.

This prevents a direct interpretation of a shared numerical value of

\[
\lambda
\]

as an equivalent functional constraint across methods.

## 8.5 Balanced likelihood and exact generation

The architecture ranking depends on the evaluation criterion.

Likelihood-based metrics favor intermediate regularization because moderate EWC reduces forgetting while still permitting acquisition of later factual sets.

Exact generation is substantially more fragile.

A configuration may improve

\[
L_{\mathrm{final}}^{\mathrm{mean}}
\]

while producing limited or zero exact-match generation.

This difference is consistent with the distinction between probabilistic support for a factual target and successful discrete generation, which is also relevant in knowledge-editing evaluation [9,10,13].

The results therefore do not support replacing continuous likelihood metrics with exact match, or vice versa. Both capture complementary aspects of factual retention.

## 8.6 Equal parameter count does not imply equal behavior

All three adapters use

\[
6144
\]

trainable parameters.

Nevertheless, their continual-learning trajectories differ substantially.

This demonstrates that trainable parameter count alone is insufficient to characterize parameter-efficient continual adaptation.

LoRA allocates its capacity to an effective low-rank linear transformation,

\[
\Delta W_L=AB,
\]

whereas Symmetric allocates the same parameter budget to low-rank quadratic forms,

\[
Q_j
=
\frac{\alpha}{r}
\sum_k
P_{kj}u_ku_k^T.
\]

Combined partitions the same budget across both transformation families.

The observed differences therefore arise under matched parameter count but unmatched functional geometry.

## 8.7 Architecture-dependent EWC response

EWC constrains adapter parameters according to

\[
\frac{\lambda}{2}
\sum_i
F_i
(\theta_i-\theta_i^\ast)^2
\]

[1].

The same parameter-space penalty can produce different functional effects because the mapping

\[
\theta\mapsto\Delta y(x)
\]

depends on adapter architecture.

The results support this interpretation empirically: each method follows the same general stability–plasticity direction as \(\lambda\) increases, but with different sensitivity to the regularization scale.

This architecture-dependent response is one of the central findings of the study.

## 8.8 Practical operating regimes

Within the tested configurations, the useful operating regions can be summarized as

\[
\text{LoRA}:
\quad
\lambda\approx1\text{--}10,
\]

\[
\text{Symmetric}:
\quad
\lambda\approx10\text{--}30,
\]

and

\[
\text{Combined}:
\quad
\lambda\approx10\text{--}30.
\]

These intervals should not be interpreted as universal hyperparameter recommendations.

They identify the regions in which each architecture provides the most informative balance between retention and acquisition under the present GPT-2 Small factual-learning protocol.

## 8.9 Summary

The method-specific results support three principal conclusions.

LoRA is comparatively easier to stabilize with low EWC strength.

Symmetric provides stronger plasticity and remains adaptable under moderate consolidation.

Combined produces a valid mixed linear–quadratic architecture but does not demonstrate systematic superiority over both constituent methods.

The results therefore favor architecture-specific tuning of consolidation strength rather than a common EWC coefficient across parameter-efficient adaptation methods.
