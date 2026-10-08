# 13. Conclusions

This study investigated how online Elastic Weight Consolidation interacts with different parameter-efficient adapter geometries in continual factual learning.

Using a frozen GPT-2 Small backbone [3], LoRA [2], Symmetric, and Combined adapters were evaluated under the same sequential

\[
A\rightarrow B\rightarrow C
\]

protocol without replay. All methods used the same adapter placement and a matched budget of

\[
6144
\]

trainable parameters.

Across all three architectures, EWC produces a consistent stability–plasticity trade-off:

\[
\lambda\uparrow
\Rightarrow
\text{forgetting}\downarrow,
\]

while

\[
\lambda\uparrow
\Rightarrow
\text{new-fact acquisition quality}\downarrow.
\]

This behavior is consistent with the role of EWC as a soft consolidation mechanism for reducing catastrophic forgetting [1].

The experiments also show that the effective regularization regime depends on adapter geometry. LoRA reaches a stability-oriented region at comparatively low values of

\[
\lambda,
\]

whereas Symmetric remains more plastic under moderate consolidation and requires stronger regularization before entering a similarly constrained regime. Combined generally exhibits intermediate behavior and does not consistently dominate both constituent architectures.

Intermediate regularization provides the most useful balance between retention and acquisition. Strong EWC continues to reduce forgetting, but eventually produces under-learning of later factual sets. Consequently, maximum stability is not equivalent to maximum continual-learning performance.

The mechanistic analysis supports this interpretation. Increasing

\[
\lambda
\]

reduces cumulative adapter displacement, and larger displacement is positively associated with forgetting across all three methods. However, the same parameter-space constraint produces different functional effects across architectures because LoRA, Symmetric, and Combined implement different mappings from trainable parameters to model outputs.

The results therefore support the central formulation

\[
\text{continual-learning behavior}
=
f(
\text{adapter geometry},
\text{consolidation strength}
).
\]

Equal trainable parameter count is not sufficient to predict continual-learning behavior, and a common EWC coefficient cannot be assumed to impose an equivalent functional constraint across different adapter parameterizations.

The study further reinforces the distinction between parameter preservation and functional preservation. Preventing direct modification of previously trained parameters does not necessarily preserve previously learned behavior, because later adaptations can alter the complete network function. Continual-learning evaluation should therefore focus on behavioral retention rather than parameter immutability alone.

Overall, the results indicate that parameter-efficient continual learning should be treated as a joint design problem over adaptation geometry and consolidation strength rather than as two independent choices.

The present conclusions are restricted to GPT-2 Small, a single adapter placement, a fixed parameter budget, synthetic factual updates, and short continual-learning sequences. Future work should extend the analysis to larger models, multi-layer adaptation, longer task sequences, realistic factual updates, broader PEFT baselines, and more structured importance approximations.

A particularly relevant direction is the use of branch-specific consolidation in the Combined architecture,

\[
\lambda_L
\neq
\lambda_S,
\]

together with independent optimization of

\[
r_L
\]

and

\[
r_S.
\]

More generally, future experiments should determine whether the architecture-dependent EWC response observed here persists across model scales and whether quadratic low-rank adaptation continues to occupy a distinct stability–plasticity regime under broader continual-learning conditions.
