# 12. Limitations

## 12.1 Model scale and adapter placement

All experiments are conducted on GPT-2 Small [3] with adaptation restricted to zero-based Transformer block 0 at `attn.c_proj`.

The reported results therefore establish behavior only for this backbone scale and placement.

They do not determine whether the same stability–plasticity ordering persists for larger language models, deeper adapter insertion, multiple adapted layers, or alternative projection sites.

## 12.2 Synthetic factual-learning setting

The continual-learning sequence uses three synthetic factual sets with 12 associations each.

This controlled design is useful for isolating interference and retention, but it does not reproduce the diversity, ambiguity, paraphrastic variability, or contextual complexity of real factual knowledge.

The study should therefore be interpreted as a controlled continual-learning experiment rather than as a direct evaluation of large-scale knowledge editing in realistic deployment conditions [9,10,13].

## 12.3 Sequence length

The experimental sequence contains only three learning phases,

\[
A\rightarrow B\rightarrow C.
\]

This is sufficient to observe repeated interference and online consolidation, but it does not establish the long-term behavior of the method.

For longer sequences, the accumulated Fisher state

\[
F_{1:t}
=
\gamma F_{1:t-1}+F_t
\]

may progressively constrain adaptation, particularly with

\[
\gamma=1.
\]

Long-horizon experiments are therefore required to determine whether the observed compromise regions remain stable as the number of sequential updates increases.

## 12.4 No-replay regime

The study intentionally excludes replay.

This isolates the contribution of regularization and adapter geometry, but it does not compare EWC against rehearsal-based continual-learning strategies.

Replay, memory buffers, generative rehearsal, or hybrid replay–regularization methods may produce different stability–plasticity trade-offs.

The results should therefore not be interpreted as evidence that EWC is superior to replay-based continual learning.

## 12.5 Diagonal Fisher approximation

The EWC implementation uses a diagonal Fisher approximation [1]:

\[
F
\approx
\operatorname{diag}(F_1,\ldots,F_n).
\]

Off-diagonal parameter interactions are ignored.

This approximation is particularly relevant for factorized adapters, where the effective transformation depends jointly on multiple parameter matrices.

The estimated importance structure therefore provides only a local, coordinate-dependent approximation to the geometry of previous-task sensitivity.

## 12.6 Empirical Fisher sample size

The empirical Fisher is estimated using 12 deterministic examples after phases \(A\) and \(B\).

This provides a consistent estimator within the present experimental design but remains a small sample of the model's possible activation and output space.

The resulting Fisher values may therefore be sensitive to the specific factual examples used for consolidation.

Larger or more diverse estimation sets could alter the effective regularization profile.

## 12.7 Parameterization dependence

Fisher values and parameter displacements are computed in the native coordinates of each adapter architecture.

Consequently,

\[
F_i^{\mathrm{LoRA}},
\qquad
F_i^{\mathrm{Symmetric}},
\qquad
F_i^{\mathrm{Combined}}
\]

are not directly comparable as architecture-independent measures of functional importance.

Similarly,

\[
\|\theta-\theta^\ast\|_2
\]

does not have the same functional interpretation across parameterizations.

Cross-method conclusions therefore rely primarily on behavioral metrics rather than direct comparison of raw Fisher or displacement magnitudes.

## 12.8 Fixed online accumulation factor

The online EWC accumulation coefficient is fixed to

\[
\gamma=1.
\]

Thus,

\[
F_{AB}=F_A+F_B.
\]

No decay of previous importance estimates is applied.

This choice maximizes persistence of earlier Fisher information but may become increasingly restrictive over longer learning sequences.

Alternative values

\[
0<\gamma<1
\]

could provide controlled forgetting of old importance estimates and modify the long-term stability–plasticity balance.

## 12.9 Discrete lambda grid

The exploratory study evaluates

\[
\lambda\in
\{0,1,10,30,50,100,300\}.
\]

The resulting knee locations therefore depend on a discrete and logarithmically sparse sampling of the regularization axis.

The selected operating regions,

\[
\lambda\approx10
\]

for LoRA and

\[
\lambda\approx10\text{--}30
\]

for Symmetric and Combined, should not be interpreted as exact optima.

A denser local sweep could identify more precise compromise points.

## 12.10 Number of seeds

The confirmation stage uses three seeds,

\[
42,\qquad123,\qquad456.
\]

This is sufficient to assess whether the main exploratory trends persist across independent runs, but it is insufficient for strong inferential claims.

The study therefore reports descriptive consistency rather than population-level statistical significance.

Additional seeds would be required for reliable confidence intervals, hypothesis testing, or fine-grained ranking between closely performing configurations.

## 12.11 Exact-generation sensitivity

Exact match is highly sensitive to decoding behavior.

A small change in token probability can alter the generated sequence without producing a proportionally large change in NLL.

Conversely, substantial likelihood improvement does not guarantee exact generation.

The observed exact-match variability across seeds and regularization strengths should therefore be interpreted together with continuous likelihood metrics.

Results may also depend on the chosen decoding and answer-normalization procedures.

## 12.12 Parameter matching is not functional matching

The three architectures are matched at

\[
6144
\]

trainable parameters.

However, equal parameter count does not imply equal functional capacity.

LoRA represents a low-rank linear transformation [2], whereas Symmetric represents a collection of low-rank quadratic forms.

Combined divides its capacity between these two function classes.

The experiments therefore compare architectures under equal parameter budget, not under an established measure of equal expressivity.

## 12.13 Combined branch allocation

The Combined adapter uses

\[
r_L=r_S=2
\]

and a single EWC coefficient

\[
\lambda
\]

for both branches.

This configuration does not test whether linear and quadratic components would benefit from different parameter allocations or different consolidation strengths.

A more general formulation could use

\[
\lambda_L
\neq
\lambda_S
\]

and independently vary

\[
r_L
\]

and

\[
r_S.
\]

The absence of systematic Combined dominance in the present experiments therefore does not exclude gains from branch-specific optimization.

## 12.14 Mechanistic evidence is associative

The mechanistic analysis identifies strong associations between regularization strength, adapter displacement, and forgetting.

However,

\[
\lambda
\rightarrow
D_{A\rightarrow C}
\rightarrow
F_{\mathrm{mean}}
\]

is not established as a complete causal chain.

The observed correlations are computed over a small discrete lambda sweep, and parameter movement is measured in architecture-specific coordinates.

The mechanistic results should therefore be interpreted as supporting evidence rather than causal proof.

## 12.15 Scope of architectural claims

The study shows that LoRA, Symmetric, and Combined occupy different empirical stability–plasticity regimes under the tested conditions.

It does not establish that quadratic adaptation is universally more plastic, that LoRA is universally more stable, or that either family is intrinsically superior.

Such claims would require evaluation across multiple model families, parameter budgets, datasets, adapter placements, and optimization regimes.

Similarly, the present work does not claim that the Symmetric formulation is the only possible quadratic or nonlinear low-rank parameterization.

A broader comparison with related nonlinear and higher-order parameter-efficient methods is required to establish its position within the wider PEFT literature.

## 12.16 Generalization

The strongest supported conclusion is restricted to the experimental setting studied here:

\[
\text{adapter geometry}
+
\text{EWC strength}
\]

jointly determine the observed stability–plasticity behavior under fixed-budget continual factual adaptation.

Future validation should extend this analysis to larger models, longer continual-learning sequences, realistic factual updates, multiple adapter placements, broader PEFT baselines, and more structured consolidation methods.

These extensions are necessary before drawing conclusions about the general behavior of quadratic low-rank adaptation or architecture-specific EWC scaling in large language models.
