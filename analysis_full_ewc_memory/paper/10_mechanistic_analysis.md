# 10. Mechanistic Analysis

## 10.1 Parameter displacement

To examine how EWC modifies the optimization trajectory, adapter displacement is measured across learning phases:

\[
D_{A\rightarrow B}
=
\|\theta_B-\theta_A\|_2,
\]

\[
D_{B\rightarrow C}
=
\|\theta_C-\theta_B\|_2,
\]

and

\[
D_{A\rightarrow C}
=
\|\theta_C-\theta_A\|_2.
\]

Across the exploratory lambda sweep, increasing

\[
\lambda
\]

generally reduces cumulative adapter displacement.

For \(D_{A\rightarrow C}\), displacement decreases monotonically across all six adjacent lambda transitions for LoRA and across five of six transitions for both Symmetric and Combined.

This behavior is consistent with the role of EWC as a Fisher-weighted constraint on parameter movement [1].

![Mechanistic evidence](../analysis_full_ewc_memory/figures_main/figure_6_mechanistic_evidence.svg)

## 10.2 Displacement and forgetting

The relationship between cumulative displacement and forgetting is positive for all three architectures.

For the exploratory sweep, the correlations between

\[
D_{A\rightarrow C}
\]

and

\[
F_{\mathrm{mean}}
\]

are:

| Method | Pearson \(r\) | Spearman \(\rho\) |
|---|---:|---:|
| LoRA | 0.7873 | 1.0000 |
| Symmetric | 0.6930 | 0.9643 |
| Combined | 0.7471 | 0.8929 |

Thus, configurations that move farther from the post-\(A\) adapter state tend to exhibit greater forgetting.

The relation is strongest in rank-order terms for LoRA and remains substantial for Symmetric and Combined.

These correlations are descriptive and do not establish that Euclidean parameter displacement is itself the causal source of forgetting.

## 10.3 Lambda and cumulative displacement

The direct relationship between EWC strength and cumulative displacement is strongly negative.

For

\[
\lambda
\]

versus

\[
D_{A\rightarrow C},
\]

the exploratory correlations are:

| Method | Pearson \(r\) | Spearman \(\rho\) |
|---|---:|---:|
| LoRA | -0.9478 | -1.0000 |
| Symmetric | -0.9915 | -0.9643 |
| Combined | -0.9769 | -0.8929 |

The effect is therefore consistent across all adapter families:

\[
\lambda\uparrow
\Rightarrow
D_{A\rightarrow C}\downarrow.
\]

Together with the positive displacement–forgetting association, this provides a parameter-space interpretation of the observed retention gains.

## 10.4 Lambda and behavioral response

The exploratory correlations between

\[
\lambda
\]

and forgetting are negative:

| Method | Pearson \(r\) | Spearman \(\rho\) |
|---|---:|---:|
| LoRA | -0.6321 | -1.0000 |
| Symmetric | -0.7502 | -1.0000 |
| Combined | -0.7358 | -1.0000 |

Conversely, the correlations between

\[
\lambda
\]

and acquisition NLL are positive:

| Method | Pearson \(r\) | Spearman \(\rho\) |
|---|---:|---:|
| LoRA | 0.7717 | 1.0000 |
| Symmetric | 0.9267 | 1.0000 |
| Combined | 0.8992 | 1.0000 |

The perfect Spearman coefficients for the primary behavioral metrics reflect the monotonic ordering of the sampled lambda values.

These results quantify the same stability–plasticity trajectory observed in Sections 6–9.

## 10.5 Fisher structure

The first Fisher estimate,

\[
F_A,
\]

is identical across lambda values within a given architecture because phase \(A\) is completed before EWC becomes active.

After learning \(B\),

\[
F_{AB}
=
F_A+F_B
\]

depends on the trajectory induced by the selected regularization strength.

Consequently, later Fisher states are not independent of

\[
\lambda.
\]

Across the adapter families, most Fisher mass is concentrated in output-side factors rather than being uniformly distributed across all trainable matrices.

However, raw Fisher values are parameterization dependent. Direct comparison of absolute Fisher magnitudes between LoRA, Symmetric, and Combined therefore does not provide an architecture-independent measure of functional importance.

## 10.6 Realized EWC penalty

The realized consolidation cost at a given training state is

\[
P_{\mathrm{EWC}}
=
\frac{\lambda}{2}
\sum_i
F_i
(\theta_i-\theta_i^\ast)^2.
\]

Unlike

\[
\lambda,
\]

this quantity depends jointly on the regularization coefficient, the Fisher profile, and the displacement actually taken by the optimizer.

Its correlation with behavioral metrics is substantially weaker than the direct lambda and displacement relationships.

For the realized phase-\(C\) penalty versus forgetting:

| Method | Pearson \(r\) | Spearman \(\rho\) |
|---|---:|---:|
| LoRA | -0.4391 | -0.1071 |
| Symmetric | -0.5507 | -0.3929 |
| Combined | -0.4043 | -0.2857 |

For the same penalty versus acquisition NLL:

| Method | Pearson \(r\) | Spearman \(\rho\) |
|---|---:|---:|
| LoRA | 0.1581 | 0.1071 |
| Symmetric | 0.1904 | 0.3929 |
| Combined | 0.0287 | 0.2857 |

Thus, the realized scalar penalty is not by itself a sufficient descriptor of the resulting behavioral regime.

## 10.7 Architecture-dependent response

The three adapters show the same qualitative chain,

\[
\lambda\uparrow
\Rightarrow
\text{smaller parameter displacement}
\Rightarrow
\text{less forgetting},
\]

but differ in how rapidly this transition occurs.

LoRA reaches a constrained regime at lower

\[
\lambda,
\]

whereas Symmetric retains stronger plasticity under moderate regularization.

Combined again occupies an intermediate region.

This is consistent with the fact that EWC acts in parameter space [1], while the adapters implement different parameter-to-function mappings.

For LoRA,

\[
\Delta y_L(x)
=
\frac{\alpha}{r}(xA)B,
\]

whereas for Symmetric,

\[
\Delta y_S(x)
=
\frac{\alpha}{r}
\left[(xU)\odot(xU)\right]P.
\]

Equal Euclidean displacement or equal Fisher-weighted displacement therefore need not produce equal functional perturbations.

## 10.8 Functional interference beyond parameter overwrite

Previous hard-consolidation experiments provide an important complementary observation.

Earlier adapter tensors can remain bit-identical while performance on the facts associated with those tensors still degrades after later components are trained.

Thus,

\[
\Delta\theta_{\mathrm{old}}=0
\]

does not imply

\[
\Delta f_{\mathrm{old}}=0.
\]

The network output depends on the complete active computation, so new additive components can alter the function even when previously trained parameters are perfectly preserved.

This supports the distinction introduced in Section 3 between parameter preservation and functional retention.

EWC addresses a different regime: instead of freezing historical parameters, it constrains the movement of a shared adapter while allowing continued optimization.

## 10.9 Interpretation

The mechanistic evidence is consistent with the following sequence:

\[
\lambda
\rightarrow
\text{Fisher-weighted constraint}
\rightarrow
\text{reduced adapter movement}
\rightarrow
\text{reduced functional interference}.
\]

However, the present experiments do not establish this sequence as a complete causal mechanism.

The Fisher approximation is diagonal, parameter displacement is measured in the chosen adapter coordinates, and the mapping from those coordinates to model behavior is architecture dependent.

The mechanistic analysis should therefore be interpreted as associative evidence supporting the behavioral findings rather than as proof of a universal geometric explanation.

## 10.10 Mechanistic conclusion

Three observations are robust across the evaluated architectures.

First,

\[
\lambda
\]

strongly controls adapter displacement.

Second, larger cumulative displacement is associated with greater forgetting.

Third, the same nominal EWC strength produces different behavioral responses across adapter parameterizations.

These results support the central interpretation of the study: continual-learning performance is jointly determined by the geometry of the adaptation mechanism and by the strength of the consolidation constraint.
