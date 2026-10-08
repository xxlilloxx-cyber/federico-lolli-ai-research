# 1. Introduction

Parameter-efficient fine-tuning (PEFT) enables pretrained language models to be adapted while updating only a small fraction of their parameters [4]. Among PEFT methods, Low-Rank Adaptation (LoRA) constrains weight updates through trainable low-rank factors while keeping the pretrained backbone frozen [2]. Such approaches reduce trainable capacity and storage requirements, but sequential adaptation remains susceptible to interference between successive updates.

Continual learning addresses the problem of acquiring new information while preserving previously learned behavior [8]. In neural networks, sequential optimization may produce catastrophic forgetting, whereby adaptation to new data degrades performance on earlier tasks or knowledge [1]. This stability–plasticity trade-off becomes particularly relevant for language models that must incorporate factual information over time.

Knowledge-editing methods have addressed related problems by modifying factual associations stored in pretrained language models. ROME localizes and edits factual associations through targeted parameter modification [9], while MEMIT extends this approach to larger sets of edits [10]. Other methods, including MEND [11] and SERAC [12], investigate learned update transformations and memory-assisted editing. These approaches demonstrate that factual knowledge can be modified without full-model retraining, but repeated sequential updates remain a central challenge for persistent model adaptation [13].

This work studies continual factual learning under fixed parameter-efficient adaptation capacity. A frozen GPT-2 Small backbone [3] is adapted using three matched-budget architectures:

\[
\text{LoRA},
\qquad
\text{Symmetric},
\qquad
\text{Combined}.
\]

LoRA provides a conventional linear low-rank update [2]. Symmetric introduces a factorized quadratic correction, while Combined incorporates both linear and quadratic branches. Each architecture contains 6,144 trainable parameters and is inserted at the same Transformer projection, isolating functional parameterization as the primary architectural variable.

The model learns three disjoint factual sets sequentially,

\[
A\rightarrow B\rightarrow C,
\]

without replay. The objective is to characterize how adapter geometry interacts with consolidation strength in determining retention of previous facts and acquisition of new ones.

## 1.1 Adapter geometry and continual factual learning

LoRA applies the low-rank correction

\[
\Delta y_L
=
\frac{\alpha}{r}(xA)B,
\]

which is linear in the incoming activation \(x\) [2].

The Symmetric adapter instead uses

\[
\Delta y_S
=
\frac{\alpha}{r}
\left[(xU)\odot(xU)\right]P,
\]

producing a factorized quadratic transformation.

The Combined architecture contains both components,

\[
\Delta y_C
=
\frac{\alpha_L}{r_L}(xA)B
+
\frac{\alpha_S}{r_S}
\left[(xU)\odot(xU)\right]P.
\]

Although the three architectures are matched in trainable parameter count, they define different functional classes. The central architectural hypothesis is therefore that adapter geometry affects the stability–plasticity response under sequential factual learning.

## 1.2 From parameter preservation to functional preservation

Continual learning cannot be reduced to prevention of parameter overwrite alone. A parameter subset may remain unchanged while modifications elsewhere alter the total network function and degrade previously acquired behavior.

This motivates the distinction between parameter preservation and functional retention. Rather than allocating and freezing independent adapter capacity for each learning phase, the present study investigates soft consolidation of a fixed shared parameter set.

Elastic Weight Consolidation (EWC) addresses catastrophic forgetting by penalizing changes to parameters estimated to be important for previously learned information [1]. The objective is

\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{current}}
+
\frac{\lambda}{2}
\sum_i
F_i
(\theta_i-\theta_i^\ast)^2,
\]

where \(F_i\) represents a diagonal Fisher importance estimate, \(\theta_i^\ast\) is the consolidated reference value, and \(\lambda\) controls the strength of the stability constraint [1].

We employ an online formulation in which Fisher information is accumulated after sequential learning phases,

\[
F_{AB}
=
\gamma F_A+F_B,
\qquad
\gamma=1.
\]

This preserves a fixed 6,144-parameter adapter throughout the complete learning sequence.

## 1.3 Research question

The central research question is:

> **How does online EWC regularization strength control the stability–plasticity trade-off in continual factual learning across linear, quadratic, and combined low-rank adapters?**

The study examines whether increasing EWC strength consistently reduces forgetting, whether this retention gain is accompanied by reduced acquisition, and whether the resulting trade-off depends on adapter geometry.

The primary behavioral quantities are mean forgetting,

\[
F_{\mathrm{mean}},
\]

and mean acquisition NLL on newly introduced fact sets,

\[
L_{\mathrm{new}}^{\mathrm{acq}}.
\]

The resulting continual-learning problem is therefore treated as

\[
\min
\left(
F_{\mathrm{mean}},
L_{\mathrm{new}}^{\mathrm{acq}}
\right).
\]

## 1.4 Experimental strategy

The study first evaluates

\[
\lambda\in
\{0,1,10,30,50,100,300\}
\]

for all three adapter families using a common seed.

Candidate operating regions identified from this exploratory sweep are subsequently evaluated across seeds

\[
42,\qquad123,\qquad456.
\]

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

The \(\lambda=0\) condition is executed through the same EWC training path and serves as the matched numerical control.

## 1.5 Contributions

This work makes four main contributions.

First, it provides a parameter-matched continual-learning comparison between linear, quadratic, and combined low-rank adapters under identical backbone, placement, data, and training conditions.

Second, it characterizes EWC strength as a continuous stability–plasticity control variable rather than as a binary consolidation mechanism.

Third, it shows that the effective regularization regime depends on adapter geometry: LoRA transitions toward stability at lower \(\lambda\), whereas the quadratic Symmetric adapter retains greater plasticity under moderate consolidation.

Fourth, it relates the observed behavioral trade-off to parameter displacement and Fisher-weighted constraints, providing mechanistic evidence that reduced adapter movement is associated with reduced forgetting.

The results support the formulation

\[
\text{continual-learning behavior}
=
f(
\text{adapter geometry},
\text{consolidation strength}
).
\]

The remainder of the paper defines the adapter architectures and continual-learning protocol, presents the exploratory and multi-seed results, and analyzes the resulting stability–plasticity and mechanistic behavior.
