# 5. Experimental Protocol

## 5.1 Backbone and adapter placement

All experiments use GPT-2 Small [3] with the pretrained backbone frozen throughout training.

Adapters are inserted at zero-based Transformer block 0, `attn.c_proj`.

The three compared architectures are parameter matched:

| Method | Configuration | Trainable parameters |
|---|---:|---:|
| LoRA | \(r=4,\ \alpha=4\) | 6,144 |
| Symmetric | \(r=4,\ \alpha=4\) | 6,144 |
| Combined | \(r_L=2,\ r_S=2,\ \alpha_L=\alpha_S=4\) | 6,144 |

LoRA follows the standard low-rank formulation of Hu et al. [2]. Symmetric and Combined use the formulations defined in Section 2.

For Combined,

\[
\frac{\alpha_L}{r_L}
=
\frac{\alpha_S}{r_S}
=
2.
\]

Thus, all comparisons use identical backbone, placement, and trainable-parameter budget.

## 5.2 Factual datasets and continual sequence

The continual-learning task consists of three disjoint factual sets,

\[
\mathcal D_A,\qquad
\mathcal D_B,\qquad
\mathcal D_C,
\]

with 12 synthetic factual associations per set.

Training proceeds sequentially as

\[
A\rightarrow B\rightarrow C.
\]

No replay is used.

During phase \(B\), examples from \(A\) are excluded, and during phase \(C\), examples from both \(A\) and \(B\) are excluded.

The dataset is deterministic and identical across all adapter and regularization configurations.

The validated dataset SHA-256 is

\[
\texttt{cc505ec681369c5f9aa4c7123cfff11da878bce17c0669890d416b5eb07f1923}.
\]

## 5.3 Optimization

Each phase is trained for

\[
2500
\]

optimizer steps, giving

\[
7500
\]

steps for a complete \(A\rightarrow B\rightarrow C\) run.

Optimization uses AdamW with learning rate

\[
3\times10^{-4},
\]

no learning-rate scheduler, and gradient clipping at

\[
1.0.
\]

The sequence length is

\[
64.
\]

The EWC execution path uses FP16, microbatch size 4, and gradient accumulation 1, corresponding to an effective batch size of 4.

## 5.4 EWC configuration

Consolidation follows the EWC objective of Kirkpatrick et al. [1],

\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{current}}
+
\frac{\lambda}{2}
\sum_i
F_i
(\theta_i-\theta_i^\ast)^2.
\]

Only adapter parameters contribute to the Fisher estimate and regularization term.

The empirical diagonal Fisher is estimated from 12 deterministic examples after phases \(A\) and \(B\).

Online accumulation uses

\[
F_{AB}
=
\gamma F_A+F_B,
\]

with

\[
\gamma=1.
\]

The trainable adapter capacity therefore remains fixed throughout the complete sequence.

## 5.5 Matched \(\lambda=0\) control

The primary baseline uses

\[
\lambda=0
\]

inside the same EWC execution path.

Fisher estimation, reference-state construction, batching, checkpointing, tokenization, and optimizer configuration are retained, while the EWC contribution satisfies

\[
\mathcal L_{\mathrm{EWC}}=0.
\]

This condition is therefore used as the primary numerical control for positive-\(\lambda\) comparisons.

Historical sequential runs executed under a different microbatch configuration are retained only as contextual evidence and are not used as the principal lambda baseline.

## 5.6 Exploratory lambda sweep

The exploratory stage evaluates

\[
\lambda\in
\{0,1,10,30,50,100,300\}
\]

for LoRA, Symmetric, and Combined using seed 42.

The design therefore contains

\[
3\times7=21
\]

configurations.

This stage is used to characterize the stability–plasticity trajectories and identify candidate regularization regions.

No inferential claim is made from the single-seed sweep.

## 5.7 Multi-seed confirmation

Candidate values selected from the exploratory sweep are evaluated across

\[
42,\qquad123,\qquad456
\]

random seeds.

The confirmation grids are

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

Compatible seed-42 runs from the exploratory stage are reused only after configuration validation and contribute once to scientific aggregation.

## 5.8 Evaluation checkpoints

Evaluation is performed after each learning phase:

\[
\theta_A,\qquad
\theta_B,\qquad
\theta_C.
\]

The primary quantities are acquisition losses,

\[
L_B^{\mathrm{acq}},
\qquad
L_C^{\mathrm{acq}},
\]

forgetting terms,

\[
F_{A\rightarrow B},
\qquad
F_{A\rightarrow C},
\qquad
F_{B\rightarrow C},
\]

and final balanced memory,

\[
L_{\mathrm{final}}^{\mathrm{mean}}.
\]

The aggregate stability and plasticity metrics are

\[
F_{\mathrm{mean}}
=
\frac{
F_{A\rightarrow B}
+
F_{A\rightarrow C}
+
F_{B\rightarrow C}
}{3},
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

## 5.9 Behavioral metrics

Target negative log-likelihood is the primary continuous metric.

Correct-answer probability and exact-match generation are retained as complementary measurements.

Paraphrased prompts are evaluated separately using the same metric families.

The distinction between likelihood and exact generation is relevant to factual editing and model-update evaluation, where improved target likelihood does not necessarily imply successful discrete recall [9,10,13].

## 5.10 Mechanistic diagnostics

Saved checkpoints are used to compute parameter displacement across learning phases:

\[
\|\theta_B-\theta_A\|_2,
\]

\[
\|\theta_C-\theta_B\|_2,
\]

and

\[
\|\theta_C-\theta_A\|_2.
\]

Additional diagnostics include Fisher mass, maximum Fisher entry, effective support, factor-level Fisher allocation, and realized EWC cost,

\[
\frac{\lambda}{2}
\sum_i
F_i
(\theta_i-\theta_i^\ast)^2.
\]

These quantities are used to characterize the parameter-space response to consolidation and are not interpreted as causal evidence in isolation.

## 5.11 Statistical treatment

The exploratory sweep is analyzed descriptively.

The confirmation stage reports aggregate results across the three seeds as

\[
\text{mean}\pm\text{sample standard deviation}.
\]

Because the confirmation contains only three seeds, the analysis does not claim population-level statistical significance.

Emphasis is placed on consistency of effect direction, architecture-dependent response, and agreement between exploratory and confirmation stages.

## 5.12 Data validation and aggregation

Scientific aggregation is performed over unique canonical runs rather than raw result directories.

The complete validated study contains 72 canonical scientific runs after duplicate resolution.

Validation checks include:

- dataset identity;
- no-replay phase separation;
- adapter parameter budget;
- execution-path consistency;
- exact zero EWC multiplier for \(\lambda=0\);
- unique method–\(\lambda\)–seed aggregation.

These controls define the experimental basis for the results presented in the following sections.
