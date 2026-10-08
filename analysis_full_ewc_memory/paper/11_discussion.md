# 11. Discussion

## 11.1 Main finding

The central result of this study is that EWC strength acts as an explicit stability–plasticity control variable across all three parameter-efficient adapter families.

Increasing

\[
\lambda
\]

consistently reduces forgetting but weakens the acquisition of new factual information.

This behavior is expected from the role of EWC as a soft consolidation mechanism [1], but the experiments show that the magnitude and location of the trade-off depend strongly on adapter geometry.

The relevant design problem is therefore not the independent choice of an adapter or an EWC coefficient, but their joint selection.

## 11.2 Stability and plasticity are coupled

The exploratory sweep shows that, within each architecture, every reduction in forgetting is associated with a corresponding acquisition cost.

The sampled configurations therefore form a Pareto-like trade-off trajectory rather than a sequence in which stronger consolidation is uniformly beneficial.

This observation is important because continual-learning performance cannot be evaluated from forgetting alone.

A model that perfectly preserves previous information but fails to acquire new facts is not an effective continual learner.

The useful operating region is instead located where the retention benefit obtained from regularization remains larger than the resulting loss of plasticity.

## 11.3 Adapter geometry changes the EWC response

LoRA, Symmetric, and Combined use the same frozen backbone, the same adapter location, and the same trainable parameter budget, yet their responses to EWC differ.

LoRA reaches a stability-oriented regime at relatively low

\[
\lambda.
\]

Symmetric remains substantially more plastic over the same regularization range and requires stronger consolidation before reaching a similar retention regime.

Combined generally occupies an intermediate region.

This result indicates that the effective strength of EWC is architecture dependent.

A numerical value of

\[
\lambda
\]

therefore cannot be interpreted independently of the parameterization on which the penalty operates.

## 11.4 Parameter count is not sufficient to characterize adaptation

All evaluated adapters contain

\[
6144
\]

trainable parameters.

Nevertheless, they exhibit substantially different continual-learning behavior.

The result shows that parameter count alone is not an adequate measure of adaptation capacity.

LoRA constrains the update to a low-rank linear transformation [2],

\[
\Delta y_L
=
\frac{\alpha}{r}(xA)B,
\]

whereas Symmetric represents an input-dependent quadratic correction,

\[
\Delta y_S
=
\frac{\alpha}{r}
\left[(xU)\odot(xU)\right]P.
\]

Combined partitions the same parameter budget between the two.

Thus, equal parameter budget does not imply equal functional geometry.

## 11.5 Interpretation of the Symmetric adapter

The Symmetric adapter exhibits the strongest acquisition behavior in the weakly regularized regime and remains plastic under moderate EWC.

At the same time, this plasticity is associated with greater unregularized forgetting.

The result suggests that the quadratic parameterization provides a different adaptation regime from the linear LoRA baseline.

However, the present experiments do not establish that quadratic structure is intrinsically superior.

The comparison demonstrates a different stability–plasticity profile under matched parameter count, not a universal ranking of expressive power.

The strongest result is therefore the interaction between the Symmetric geometry and consolidation strength rather than an isolated claim of architectural superiority.

## 11.6 Interpretation of LoRA

LoRA provides a comparatively stable baseline.

This is consistent with its constrained low-rank linear update structure [2] and with the observation that relatively small EWC coefficients produce substantial reductions in parameter displacement.

The same sensitivity also means that plasticity is lost earlier as

\[
\lambda
\]

increases.

Under the present task, LoRA is therefore attractive when retention is prioritized, but its optimal consolidation region differs from that of the more plastic Symmetric adapter.

## 11.7 Combined does not establish synergistic dominance

The Combined adapter was designed to provide both linear and quadratic adaptation within the same total parameter budget.

The results show that the architecture is viable and produces a distinct intermediate stability–plasticity trajectory.

However, it does not consistently dominate both LoRA and Symmetric.

This suggests that simply dividing a fixed budget between two functional branches is not sufficient to guarantee complementary gains.

One possible explanation is that the two branches compete for a limited parameter budget.

Another is that applying a single shared EWC coefficient does not account for differences in the sensitivity of the linear and quadratic branches.

These interpretations remain hypotheses and require direct experimental testing.

## 11.8 Parameter displacement and functional retention

The mechanistic analysis shows that increasing EWC strength strongly reduces cumulative adapter displacement.

Across all three architectures, larger displacement is also associated with greater forgetting.

This supports the intended role of EWC [1]:

\[
\text{important parameters}
\rightarrow
\text{restricted movement}
\rightarrow
\text{improved retention}.
\]

However, Euclidean parameter displacement is not sufficient to characterize functional change.

The adapter parameterizations define different nonlinear mappings from parameters to model outputs, and identical parameter-space movement can therefore have different behavioral consequences.

This explains why the same

\[
\lambda
\]

does not produce equivalent retention across architectures.

## 11.9 Fisher-weighted versus Euclidean geometry

EWC defines a local weighted distance,

\[
D_F(\theta,\theta^\ast)
=
\sum_i
F_i(\theta_i-\theta_i^\ast)^2,
\]

rather than an isotropic Euclidean constraint [1].

In principle, this allows movement in parameters assigned low importance while restricting movement in parameters considered important for previously learned information.

The present results confirm that increasing this constraint reduces total adapter movement, but the realized scalar EWC penalty is only weakly associated with the final behavioral metrics.

This indicates that neither

\[
\|\theta-\theta^\ast\|_2
\]

nor

\[
D_F(\theta,\theta^\ast)
\]

alone fully characterizes functional retention.

The relevant geometry is ultimately induced by the complete parameter-to-function mapping.

## 11.10 Likelihood and exact recall measure different behavior

The experiments reveal a clear distinction between likelihood-based and generation-based metrics.

Moderate EWC can improve final mean NLL while exact-match generation deteriorates.

This is not contradictory.

NLL evaluates the probability assigned to the correct target sequence, whereas exact match requires the decoding process to recover that sequence exactly.

Knowledge-editing literature similarly distinguishes between successful modification of target likelihood and broader behavioral properties of the edited model [9,10,13].

For continual factual learning, both views are therefore necessary.

Continuous likelihood metrics provide a stable representation of gradual forgetting and acquisition, while exact generation tests whether the learned factual association is strong enough to determine the model output.

## 11.11 Functional preservation versus parameter preservation

The historical hard-consolidation experiments reinforce an important distinction.

Previously trained adapter tensors can remain exactly unchanged while the corresponding factual behavior still degrades after later components are introduced.

Hence,

\[
\text{parameter preservation}
\neq
\text{functional preservation}.
\]

This observation is particularly relevant for modular and parameter-efficient continual learning.

Preventing direct overwriting of earlier parameters does not prevent later components from altering shared hidden representations or the final output.

The present EWC experiments instead study soft consolidation of a single shared adapter, where all parameters remain available for subsequent learning but movement is selectively constrained.

## 11.12 Implications for parameter-efficient continual learning

The results suggest that PEFT methods should not be evaluated solely through static adaptation performance.

A parameterization that learns a task efficiently in isolation may respond very differently when repeatedly updated.

For continual adaptation, at least three properties should be considered jointly:

\[
\text{acquisition},
\qquad
\text{retention},
\qquad
\text{response to consolidation}.
\]

The experiments show that these properties differ even between adapters with identical trainable parameter counts.

This extends the usual PEFT comparison beyond parameter efficiency toward continual adaptation behavior.

## 11.13 Relation to factual model editing

ROME, MEMIT, MEND, and SERAC demonstrate that factual associations in language models can be modified without full-model retraining [9–12].

The present work studies a related but distinct regime.

Rather than performing isolated or externally managed edits, the model sequentially optimizes a fixed parameter-efficient adapter on successive factual sets.

The central question is therefore not only whether a fact can be inserted, but whether new facts can continue to be acquired while preserving earlier updates.

This places the present study at the intersection of parameter-efficient adaptation, continual learning, and factual knowledge modification.

## 11.14 Scope of the architectural conclusions

The results should not be interpreted as evidence that one adapter family is universally preferable.

The observed ordering is specific to the evaluated GPT-2 Small configuration, adapter placement, matched parameter budget, factual datasets, optimization settings, and EWC implementation.

The strongest supported conclusion is narrower:

\[
\boxed{
\text{adapter geometry changes the empirical stability–plasticity response to EWC}
}
\]

under the controlled conditions of this study.

This claim is directly supported by both the exploratory sweep and multi-seed confirmation.

## 11.15 Summary

The experiments indicate that continual parameter-efficient adaptation is governed jointly by functional parameterization and consolidation strength.

LoRA provides a relatively stability-oriented regime, Symmetric retains greater plasticity under moderate consolidation, and Combined does not demonstrate systematic synergistic dominance.

Intermediate EWC provides the most useful balance between retaining previous facts and acquiring new ones, while excessive regularization produces under-learning.

The resulting design principle is

\[
\text{continual PEFT}
\neq
\text{adapter selection}
+
\text{independent regularization tuning}.
\]

Instead,

\[
\text{continual PEFT}
=
\text{joint optimization of adapter geometry and consolidation strength}.
\]
