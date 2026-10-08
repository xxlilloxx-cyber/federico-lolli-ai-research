# 7. Multi-Seed Confirmation

## 7.1 Confirmation design

The exploratory sweep identified architecture-specific regularization regions for subsequent confirmation.

The selected grids are

\[
\text{LoRA}:
\quad
\lambda\in\{0,1,10\},
\]

\[
\text{Symmetric}:
\quad
\lambda\in\{0,10,30\},
\]

and

\[
\text{Combined}:
\quad
\lambda\in\{0,10,30\}.
\]

Each configuration is evaluated using seeds

\[
42,\qquad123,\qquad456.
\]

Seed-42 configurations already available from the exploratory sweep are reused only after validation and are included once in the aggregate analysis.

The objective of this stage is not to identify a statistically universal optimum, but to determine whether the stability–plasticity patterns observed in the exploratory sweep persist across independent training seeds.

## 7.2 Validation of the confirmation set

All confirmation runs satisfy the same experimental constraints defined in Section 5:

- identical GPT-2 Small backbone [3];
- identical adapter placement;
- fixed 6,144-parameter trainable budget;
- identical \(A\rightarrow B\rightarrow C\) factual sequence;
- no replay;
- identical EWC execution path;
- identical dataset;
- identical Fisher-estimation procedure.

The \(\lambda=0\) configurations remain the primary execution-matched controls.

Scientific aggregation is performed over unique canonical runs so that reused seed-42 results are not counted twice.

## 7.3 Stability–plasticity response

The multi-seed results preserve the principal trend observed during exploration:

\[
\lambda\uparrow
\Rightarrow
F_{\mathrm{mean}}\downarrow,
\]

while

\[
\lambda\uparrow
\Rightarrow
L_{\mathrm{new}}^{\mathrm{acq}}\uparrow.
\]

Thus, EWC consistently shifts all three adapter families from plastic toward stable operating regimes, in agreement with its intended role as a continual-learning consolidation mechanism [1].

However, the magnitude and location of this transition remain architecture dependent.

![Multi-seed stability-plasticity confirmation](../analysis_full_ewc_memory/figures_main/figure_2_confirmation_stability_plasticity.svg)

## 7.4 LoRA

LoRA exhibits a comparatively stability-oriented response.

A measurable reduction in forgetting is already obtained at

\[
\lambda=1,
\]

and the shift becomes stronger at

\[
\lambda=10.
\]

The corresponding acquisition loss increases as consolidation becomes stronger.

This confirms the exploratory observation that LoRA responds to comparatively low EWC coefficients.

The result is consistent across the confirmation seeds and indicates that the low-rank linear adapter [2] enters a constrained regime earlier than the quadratic Symmetric adapter under the present parameterization.

## 7.5 Symmetric

Symmetric remains the most plastic architecture at low and moderate consolidation.

At

\[
\lambda=10,
\]

the adapter preserves comparatively strong new-fact acquisition while reducing forgetting relative to its unregularized configuration.

Increasing the coefficient to

\[
\lambda=30
\]

produces a substantially more stable operating point while retaining better plasticity than would be expected from the corresponding high-regularization regime.

The multi-seed analysis therefore confirms that Symmetric requires stronger EWC than LoRA before entering a comparably stability-oriented region.

This behavior is empirical and should not be interpreted as a universal property of quadratic adapters.

## 7.6 Combined

Combined also follows the expected stability–plasticity trajectory.

Both

\[
\lambda=10
\]

and

\[
\lambda=30
\]

reduce forgetting relative to the unregularized configuration, with the stronger setting incurring a larger acquisition cost.

Across the confirmation runs, Combined generally remains between the behavior of LoRA and Symmetric.

The results do not provide evidence that the simultaneous presence of linear and quadratic branches produces a configuration that consistently dominates both individual adapter families.

## 7.7 Final balanced memory

The final balanced-memory metric is

\[
L_{\mathrm{final}}^{\mathrm{mean}}
=
\frac{
L_A(\theta_C)
+
L_B(\theta_C)
+
L_C(\theta_C)
}{3}.
\]

The confirmation results preserve the non-monotonic relationship between consolidation strength and final balanced performance.

![Final balanced memory](../analysis_full_ewc_memory/figures_main/figure_3_final_balanced_memory.svg)

Moderate EWC improves final memory relative to the corresponding

\[
\lambda=0
\]

controls because the reduction in forgetting outweighs the associated loss of plasticity.

The strongest balanced regime is observed for Symmetric around

\[
\lambda=30,
\]

with Combined at

\[
\lambda=30
\]

remaining close and LoRA favoring the lower

\[
\lambda=10
\]

region.

These observations are descriptive across three seeds rather than evidence for a universal ranking.

## 7.8 Forgetting decomposition

Mean forgetting combines three transitions,

\[
F_{A\rightarrow B},
\qquad
F_{A\rightarrow C},
\qquad
F_{B\rightarrow C}.
\]

Their decomposition confirms that the EWC effect is not restricted to a single transition.

![Forgetting decomposition](../analysis_full_ewc_memory/figures_main/figure_4_forgetting_decomposition.svg)

Increasing consolidation generally improves preservation of both the earliest factual set \(A\) and the intermediate set \(B\).

The largest challenge remains long-term preservation of \(A\) through two subsequent adaptation phases,

\[
A\rightarrow B\rightarrow C.
\]

The reduction of this long-range degradation contributes substantially to the improvement in

\[
F_{\mathrm{mean}}.
\]

## 7.9 New-fact acquisition

Acquisition is measured immediately after learning sets \(B\) and \(C\).

The multi-seed results show that the plasticity cost of EWC is present in both phases rather than being confined to the final update.

For all architectures,

\[
L_B^{\mathrm{acq}}
\]

and

\[
L_C^{\mathrm{acq}}
\]

tend to increase as \(\lambda\) increases.

Symmetric retains stronger acquisition over a broader regularization range, whereas LoRA reaches a stability-oriented regime at lower consolidation strength.

This confirms that the architecture-dependent effect is present during learning itself and is not solely an artifact of final evaluation.

## 7.10 Exact generation

Likelihood-based improvements do not translate monotonically into exact generation.

![Plasticity and exact generation](../analysis_full_ewc_memory/figures_main/figure_5_plasticity_exact_generation.svg)

Exact-match accuracy decreases as consolidation becomes strong, and the metric exhibits substantial seed sensitivity.

In particular, the zero exact-match value observed for Symmetric at

\[
\lambda=30
\]

in the exploratory seed is not reproduced as a universal outcome across the confirmation seeds.

This distinction is important because factual likelihood and discrete generated recall capture different aspects of model behavior, as also recognized in knowledge-editing evaluation [9,10,13].

The confirmation therefore supports using both likelihood-based and generation-based metrics when evaluating continual factual adaptation.

## 7.11 Architecture-dependent regularization scale

A common EWC coefficient does not correspond to a common behavioral regime across architectures.

The confirmation results instead support the approximate ordering

\[
\lambda_{\mathrm{effective}}^{\mathrm{LoRA}}
<
\lambda_{\mathrm{effective}}^{\mathrm{Symmetric}},
\]

in the sense that LoRA reaches a stability-oriented region at lower regularization strength.

Combined occupies an intermediate regime under the tested rank allocation.

This result is consistent with the architectural analysis of Section 2: equal parameter count does not imply identical mappings between parameter displacement and functional change.

Consequently, consolidation strength should be tuned jointly with adapter parameterization rather than transferred directly between architectures.

## 7.12 Confirmation of the principal hypotheses

The multi-seed stage supports four principal conclusions from the exploratory sweep.

First,

\[
\lambda\uparrow
\Rightarrow
\text{forgetting decreases}.
\]

Second,

\[
\lambda\uparrow
\Rightarrow
\text{new-fact acquisition degrades}.
\]

Third, intermediate EWC can improve final balanced memory relative to unconstrained sequential adaptation.

Fourth, the regularization strength required to reach a given stability–plasticity regime depends on adapter architecture.

The confirmation therefore supports the central formulation

\[
\text{continual-learning response}
=
f(
\text{adapter geometry},
\lambda
).
\]

## 7.13 Statistical scope

The confirmation stage uses three seeds per selected configuration.

This is sufficient to test whether the exploratory trends are reproducible across independent runs, but not to support strong population-level statistical claims.

The analysis therefore emphasizes consistency of direction and architecture-dependent operating regimes rather than significance testing.

No universal optimal value of

\[
\lambda
\]

is claimed.

The confirmed configurations instead define experimentally supported regions that are examined in greater detail in the following method-specific and mechanistic analyses.
