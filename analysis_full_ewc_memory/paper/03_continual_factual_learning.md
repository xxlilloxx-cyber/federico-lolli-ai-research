# 3. Continual Factual-Learning Setting

## 3.1 Sequential learning protocol

The continual-learning task consists of three disjoint factual datasets,

\[
\mathcal D_A,\qquad
\mathcal D_B,\qquad
\mathcal D_C,
\]

each containing 12 synthetic factual associations.

Training proceeds sequentially as

\[
A\rightarrow B\rightarrow C,
\]

without replay of previously observed factual sets.

Let

\[
\theta_0
\]

denote the initial adapter parameters and

\[
\theta_A,\qquad
\theta_B,\qquad
\theta_C
\]

the states obtained after the three training phases:

\[
\theta_0
\overset{\mathcal D_A}{\longrightarrow}
\theta_A
\overset{\mathcal D_B}{\longrightarrow}
\theta_B
\overset{\mathcal D_C}{\longrightarrow}
\theta_C.
\]

This setting isolates the central continual-learning problem of acquiring new information while preserving previously learned behavior [8]. In the absence of replay, retention depends entirely on the adaptation dynamics and consolidation mechanism.

## 3.2 Acquisition

For each factual set, acquisition is measured immediately after its corresponding training phase:

\[
L_A^{\mathrm{acq}}
=
L_A(\theta_A),
\]

\[
L_B^{\mathrm{acq}}
=
L_B(\theta_B),
\]

\[
L_C^{\mathrm{acq}}
=
L_C(\theta_C).
\]

The aggregate plasticity metric used in the stability–plasticity analysis is

\[
L_{\mathrm{new}}^{\mathrm{acq}}
=
\frac{
L_B^{\mathrm{acq}}
+
L_C^{\mathrm{acq}}
}{2}.
\]

Lower acquisition NLL indicates stronger adaptation to newly introduced facts.

## 3.3 Forgetting

Forgetting is defined as the increase in loss on a factual set after subsequent training.

For set \(A\) after learning \(B\),

\[
F_{A\rightarrow B}
=
L_A(\theta_B)
-
L_A(\theta_A).
\]

Long-term forgetting of \(A\) after the complete sequence is

\[
F_{A\rightarrow C}
=
L_A(\theta_C)
-
L_A(\theta_A).
\]

For set \(B\) after learning \(C\),

\[
F_{B\rightarrow C}
=
L_B(\theta_C)
-
L_B(\theta_B).
\]

The aggregate retention metric is

\[
F_{\mathrm{mean}}
=
\frac{
F_{A\rightarrow B}
+
F_{A\rightarrow C}
+
F_{B\rightarrow C}
}{3}.
\]

Positive values indicate degradation relative to the post-acquisition state, while values closer to zero indicate stronger retention. This functional definition follows the standard continual-learning interpretation of forgetting as degradation on previously acquired knowledge [1,8].

## 3.4 Stability–plasticity formulation

The primary comparison is expressed in the two-dimensional objective space

\[
\left(
F_{\mathrm{mean}},
L_{\mathrm{new}}^{\mathrm{acq}}
\right),
\]

with both quantities minimized.

The continual-learning problem is therefore treated as

\[
\min
\left(
F_{\mathrm{mean}},
L_{\mathrm{new}}^{\mathrm{acq}}
\right).
\]

This formulation separates stability from plasticity rather than reducing performance to a single scalar quantity.

## 3.5 Final memory

After phase \(C\), the final model is evaluated independently on all three factual sets:

\[
L_A(\theta_C),
\qquad
L_B(\theta_C),
\qquad
L_C(\theta_C).
\]

The balanced final-memory metric is

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

This quantity summarizes the final distribution of likelihood across the complete learning history, while the individual set-level values are retained to distinguish old-fact retention from newest-fact acquisition.

## 3.6 Evaluation metrics

Target negative log-likelihood is the primary continuous evaluation metric. For prompt \(q_i\) and target sequence

\[
a_i=(a_{i,1},\ldots,a_{i,T_i}),
\]

the autoregressive target loss is

\[
\operatorname{NLL}_i
=
-
\sum_{t=1}^{T_i}
\log
p_\theta
\left(
a_{i,t}
\mid
q_i,a_{i,<t}
\right).
\]

The experiments additionally report correct-answer probability and exact-match generation.

These metrics capture different aspects of factual behavior. NLL measures probabilistic support for the target sequence, whereas exact match evaluates whether the target is recovered by generation. This distinction is relevant in factual editing and continual factual adaptation, where improved likelihood does not necessarily imply successful generated recall [9,10,13].

Paraphrased prompts are evaluated separately using the same metric families.

## 3.7 Functional definition of retention

Retention is evaluated through model behavior rather than parameter preservation alone.

For an earlier factual set \(X\), the relevant quantity is

\[
L_X(\theta_{\mathrm{later}})
-
L_X(\theta_{\mathrm{acq}}).
\]

This distinction is important because unchanged parameters do not necessarily imply unchanged network function when additional trainable components modify the same computational pathway.

The experimental protocol therefore treats forgetting as functional interference rather than direct parameter overwrite.

## 3.8 Relation to EWC

Elastic Weight Consolidation introduces a controlled stability constraint through

\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{current}}
+
\frac{\lambda}{2}
\sum_i
F_i
(\theta_i-\theta_i^\ast)^2
\]

[1].

The central experimental hypothesis is that increasing

\[
\lambda
\]

reduces

\[
F_{\mathrm{mean}},
\]

while increasing

\[
L_{\mathrm{new}}^{\mathrm{acq}}.
\]

The resulting trajectory defines the empirical stability–plasticity response of each adapter architecture.

The next section specifies the online EWC formulation used to control this trade-off.
