# 9. Stability–Plasticity Analysis

## 9.1 Multi-objective interpretation

Continual factual learning requires simultaneous control of retention and acquisition.

The primary objective space is

\[
\left(
F_{\mathrm{mean}},
L_{\mathrm{new}}^{\mathrm{acq}}
\right),
\]

with both quantities minimized.

This formulation reflects the classical stability–plasticity tension in continual learning [1,8]: stronger preservation of previous knowledge generally reduces the freedom available for subsequent adaptation.

The present experiments make this trade-off explicit by varying EWC strength while keeping backbone, adapter placement, trainable parameter count, and training protocol fixed.

## 9.2 Within-method Pareto structure

For each adapter family, the sampled EWC configurations form a monotonic trade-off trajectory.

As

\[
\lambda
\]

increases,

\[
F_{\mathrm{mean}}
\]

decreases, whereas

\[
L_{\mathrm{new}}^{\mathrm{acq}}
\]

increases.

Consequently, all sampled exploratory points are Pareto-nondominated within their respective method: no configuration simultaneously improves both retention and acquisition relative to another sampled point from the same architecture.

This result implies that EWC does not identify a single intrinsically optimal solution. It exposes a family of operating points whose preference depends on the relative value assigned to stability and plasticity.

## 9.3 Architecture-dependent trade-off trajectories

The three adapter families do not follow identical trajectories through the stability–plasticity plane.

LoRA reaches a stability-oriented region at comparatively low values of

\[
\lambda.
\]

Symmetric remains substantially more plastic over the same range and requires stronger regularization before reaching a comparable consolidation regime.

Combined generally follows an intermediate trajectory.

Thus,

\[
\lambda
\]

is not a directly transferable measure of functional constraint across architectures.

The relevant object is instead the architecture-specific mapping

\[
\lambda
\longmapsto
\left(
F_{\mathrm{mean}},
L_{\mathrm{new}}^{\mathrm{acq}}
\right).
\]

## 9.4 Compromise regions

Because the sampled configurations are Pareto-nondominated, useful operating points must be identified by an explicit compromise criterion.

Two geometric criteria were applied to the exploratory trajectories: normalized distance to the ideal point and maximum distance from the endpoint chord.

The resulting compromise regions are centered around

\[
\lambda\approx10
\]

for LoRA and

\[
\lambda\approx10\text{--}30
\]

for Symmetric and Combined.

These values should be interpreted as experiment-specific knees rather than universal hyperparameter optima.

Their relevance is supported by the multi-seed confirmation, which preserves the same qualitative architecture-dependent ordering.

## 9.5 Retention gain versus acquisition cost

The effect of regularization can be interpreted as an exchange between retention gain and acquisition cost.

Relative to the execution-matched

\[
\lambda=0
\]

condition, define

\[
G_F(\lambda)
=
F_{\mathrm{mean}}(0)
-
F_{\mathrm{mean}}(\lambda),
\]

and

\[
C_A(\lambda)
=
L_{\mathrm{new}}^{\mathrm{acq}}(\lambda)
-
L_{\mathrm{new}}^{\mathrm{acq}}(0).
\]

A useful regularization regime requires

\[
G_F(\lambda)>0
\]

without an excessive increase in

\[
C_A(\lambda).
\]

The exploratory and confirmation results show that this balance is achieved primarily at intermediate EWC strengths.

## 9.6 Intermediate regularization

Intermediate EWC improves final balanced memory because the reduction in forgetting can exceed the loss in acquisition quality.

This produces a non-monotonic relationship between

\[
\lambda
\]

and

\[
L_{\mathrm{final}}^{\mathrm{mean}}.
\]

The best sampled final balanced-memory regions occur before maximum stability is reached.

This result is central to the interpretation of the experiment: minimizing forgetting alone is not equivalent to maximizing continual-learning performance.

A model can retain old knowledge extremely well while failing to acquire the new factual sets.

## 9.7 Over-constrained regime

At high regularization strengths,

\[
\lambda\in\{100,300\},
\]

forgetting continues to decrease, but new-fact acquisition deteriorates substantially.

This regime corresponds to excessive stabilization.

The adapter remains close to previously consolidated parameter configurations, but the available parameter movement becomes insufficient for effective adaptation to later factual sets.

In the limit, the desired stability mechanism therefore becomes a source of under-learning.

This behavior is consistent with the purpose of EWC as a soft constraint rather than an absolute parameter-freezing mechanism [1].

## 9.8 Cross-method dominance

No adapter architecture uniformly dominates the others across the complete stability–plasticity plane.

LoRA tends to provide stronger stability at low regularization.

Symmetric provides stronger plasticity and remains adaptable at higher consolidation strengths.

Combined occupies a mixed regime but does not consistently outperform both constituent architectures.

Therefore, architecture comparison depends on the operating objective.

A stability-prioritized application may favor a different configuration from one in which rapid acquisition is more important.

## 9.9 Exact generation and continuous metrics

Exact generation introduces an additional dimension beyond the NLL-based stability–plasticity plane.

A configuration can improve

\[
F_{\mathrm{mean}}
\]

and

\[
L_{\mathrm{final}}^{\mathrm{mean}}
\]

while exhibiting weak exact-match generation.

This occurs because target likelihood and discrete generation are related but non-equivalent measurements of factual behavior, a distinction also relevant to knowledge-editing evaluation [9,10,13].

Exact match is also substantially more sensitive to seed and decoding outcomes than the continuous likelihood metrics.

For this reason, the stability–plasticity analysis is primarily based on NLL, while exact generation is retained as a complementary behavioral measurement.

## 9.10 Stability–plasticity versus final balanced memory

The two-objective representation

\[
\left(
F_{\mathrm{mean}},
L_{\mathrm{new}}^{\mathrm{acq}}
\right)
\]

explains why final balanced memory can improve at intermediate regularization.

A decrease in

\[
F_{\mathrm{mean}}
\]

improves retention of \(A\) and \(B\), while an increase in

\[
L_{\mathrm{new}}^{\mathrm{acq}}
\]

reduces the quality of later learning.

The final model reflects both effects simultaneously.

Hence,

\[
L_{\mathrm{final}}^{\mathrm{mean}}
\]

reaches its best sampled region where the marginal retention benefit and marginal acquisition cost are favorably balanced.

## 9.11 Role of adapter geometry

Equal trainable parameter count does not produce equal stability–plasticity behavior.

The three methods implement different maps from adapter parameters to model outputs:

\[
\theta_L
\mapsto
\Delta y_L(x),
\]

\[
\theta_S
\mapsto
\Delta y_S(x),
\]

and

\[
\theta_C
\mapsto
\Delta y_C(x).
\]

Since EWC constrains movement in parameter space [1], these different parameter-to-function mappings can yield different functional responses under equivalent nominal regularization.

The observed architecture-dependent trade-off is therefore compatible with the hypothesis that consolidation strength and adapter geometry should be considered jointly.

The present experiments establish this dependence empirically but do not identify a unique causal mechanism.

## 9.12 Interpretation

The complete stability–plasticity analysis supports the relation

\[
\lambda\uparrow
\Rightarrow
\text{greater stability}
+
\text{lower plasticity},
\]

but the rate of this transition depends on architecture.

The experimental evidence therefore argues against selecting EWC strength independently of adapter design.

A more appropriate formulation is

\[
\text{operating regime}
=
f(
\text{adapter geometry},
\lambda,
\text{retention requirement},
\text{acquisition requirement}
).
\]

The following section examines the parameter-space behavior underlying these empirical trade-offs.
