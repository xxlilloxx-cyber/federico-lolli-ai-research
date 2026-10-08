# Stability–Plasticity Control in Parameter-Efficient Continual Factual Learning: EWC across Linear, Quadratic, and Combined Low-Rank Adapters

**Federico Lolli**

## Abstract

Sequential parameter-efficient adaptation must balance acquisition of new information against retention of earlier updates. We study this problem in a no-replay sequence of three disjoint synthetic factual sets using a frozen GPT-2 Small backbone and three adapters with 6,144 trainable parameters: LoRA, a symmetric factorized quadratic adapter, and a Combined linear–quadratic adapter. Consolidation uses online Elastic Weight Consolidation (EWC) with an empirical diagonal Fisher over adapter parameters. A seven-point sweep on seed 42 selects architecture-specific coefficient ranges; the selected settings are then evaluated on two additional seeds, with descriptive aggregates over all three. Stronger EWC consistently reduces forgetting while increasing new-fact acquisition NLL. Intermediate coefficients improve final balanced memory relative to the execution-matched \(\lambda=0\) control in every tested candidate configuration and each paired seed. The transition scale depends on adapter parameterization: LoRA responds at lower coefficients, Symmetric remains more plastic under moderate regularization, and Combined does not systematically dominate either component architecture. Increasing EWC strength also reduces parameter displacement, which is positively associated with forgetting. Under the evaluated setting, these findings support treating consolidation strength and adapter geometry as coupled design choices. Scope is limited to GPT-2 Small, one placement, three 12-fact sets, and two out-of-selection replication seeds.

**Keywords:** continual learning; EWC; parameter-efficient fine-tuning; LoRA; quadratic adapters; factual learning; GPT-2

# 1. Introduction

Parameter-efficient fine-tuning (PEFT) adapts pretrained language models through a small trainable parameter subset [4]. LoRA uses low-rank factors [2], while prompt- and prefix-based methods provide related alternatives [5,6]. These approaches reduce update and storage costs but do not eliminate interference under sequential adaptation [1,7,8]. The problem is especially relevant when factual associations are introduced over time.

Knowledge-editing methods including ROME, MEMIT, MEND, and SERAC modify factual behavior without full-model retraining [9–12]. We study a related no-replay setting in which one parameter-efficient pathway is optimized sequentially on three disjoint factual sets. The central question is: **How does EWC strength control the stability–plasticity trade-off across linear, quadratic, and combined low-rank adapters with the same trainable-parameter budget?**

We compare 6,144-parameter LoRA, Symmetric quadratic, and Combined linear–quadratic adapters at one projection of a frozen GPT-2 Small model [3]. Training follows \(A\rightarrow B\rightarrow C\) with online EWC [1]. A sweep over \(\lambda\in\{0,1,10,30,50,100,300\}\) on seed 42 characterizes each response curve and determines candidate coefficients. Those candidates are subsequently evaluated on two unseen seeds, 123 and 456; descriptive three-seed aggregates include the selection seed. The \(\lambda=0\) condition follows the same EWC runner and isolates the regularization term.

The study contributes (i) a controlled comparison of three adapter geometries under a common budget and continual-learning protocol; (ii) a characterization of EWC strength as a stability–plasticity control; (iii) out-of-selection replication of candidate regimes on two additional seeds; and (iv) parameter-displacement and Fisher diagnostics that qualify the behavioral results. The exploratory sweep and replicated candidate evaluation are kept distinct throughout.

# 2. Adapter Architectures

## 2.1 Common formulation and matched budget

All experiments use a frozen GPT-2 Small projection with row-vector input \(x\in\mathbb R^{1\times d}\), weight \(W\in\mathbb R^{d\times d_{\mathrm{out}}}\), and adapted output \(y=xW+b+\Delta y(x)\). At block-0 `attn.c_proj`, \(d=d_{\mathrm{out}}=768\). Only the adapter factors are optimized.

| Method | Rank configuration | Scaling | Trainable parameters |
|---|---:|---:|---:|
| LoRA | \(r=4\) | \(\alpha=4\) | 6,144 |
| Symmetric | \(r=4\) | \(\alpha=4\) | 6,144 |
| Combined | \(r_L=2,\ r_S=2\) | \(\alpha_L=\alpha_S=4\) | 6,144 |

The comparison matches backbone, placement, optimizer, and trainable parameter count. It does not equate functional capacity, local rank meaning, or branch-wise scale.

## 2.2 LoRA

LoRA [2] applies

\[
\Delta y_L=\frac{\alpha}{r}(xA)B,
\qquad A\in\mathbb R^{d\times r},\quad B\in\mathbb R^{r\times d_{\mathrm{out}}}.
\]

Equivalently, \(\Delta y_L=x\Delta W_L\), where \(\Delta W_L=(\alpha/r)AB\) and \(\operatorname{rank}(\Delta W_L)\le r\). Its parameter count is \(r(d+d_{\mathrm{out}})=6{,}144\) at rank 4. The correction is input-linear, with constant Jacobian \(J_L=(\alpha/r)AB\) and adapter-only Hessian \(H_L=0\).

## 2.3 Symmetric quadratic adapter

The Symmetric branch applies

\[
\Delta y_S=\frac{\alpha}{r}\big[(xU)\odot(xU)\big]P,
\qquad U\in\mathbb R^{d\times r},\quad P\in\mathbb R^{r\times d_{\mathrm{out}}}.
\]

It has the same parameter count \(r(d+d_{\mathrm{out}})\). For output coordinate \(j\),

\[
\Delta y_{S,j}=\frac{\alpha}{r}\sum_{k=1}^{r}P_{kj}(xu_k)^2=xQ_jx^T,
\qquad Q_j=\frac{\alpha}{r}\sum_{k=1}^{r}P_{kj}u_ku_k^T.
\]

Each \(Q_j\) is symmetric with \(\operatorname{rank}(Q_j)\le r\). The adapter therefore represents output-specific low-rank quadratic forms without constructing dense quadratic matrices. Its local derivatives are

\[
J_S(x)=2\frac{\alpha}{r}U\operatorname{diag}(xU)P,
\qquad H_{S,j}=2Q_j.
\]

Unlike LoRA, the Jacobian depends on the activation, and the trained adapter can have non-zero second derivatives.

## 2.4 Combined linear–quadratic adapter

Combined sums independent branches,

\[
\Delta y_C=\frac{\alpha_L}{r_L}(xA)B+
\frac{\alpha_S}{r_S}\big[(xU)\odot(xU)\big]P.
\]

The study uses \(r_L=r_S=2\) and \(\alpha_L=\alpha_S=4\), for \((r_L+r_S)(d+d_{\mathrm{out}})=6{,}144\) parameters. Its Jacobian is \(J_C(x)=J_L+J_S(x)\), and its adapter-only Hessian comes from the quadratic branch. Combined is thus a mixed function class under the same total budget; it is not a same-local-rank or same-branch-scale comparison with the rank-4 single-family adapters.

# 3. Continual-Learning Formulation

## 3.1 Sequential protocol and acquisition

The task contains disjoint factual datasets \(\mathcal D_A,\mathcal D_B,\mathcal D_C\), each with 12 associations. A shared adapter is trained as \(\theta_0\overset{\mathcal D_A}{\longrightarrow}\theta_A\overset{\mathcal D_B}{\longrightarrow}\theta_B\overset{\mathcal D_C}{\longrightarrow}\theta_C\), without replay. Acquisition is measured immediately after each phase: \(L_A^{\mathrm{acq}}=L_A(\theta_A)\), \(L_B^{\mathrm{acq}}=L_B(\theta_B)\), and \(L_C^{\mathrm{acq}}=L_C(\theta_C)\). The primary plasticity metric is

\[
L_{\mathrm{new}}^{\mathrm{acq}}=\frac{L_B^{\mathrm{acq}}+L_C^{\mathrm{acq}}}{2}.
\]

## 3.2 Retention and forgetting

Absolute retention is evaluated through \(L_A(\theta_A)\), \(L_A(\theta_B)\), \(L_A(\theta_C)\), \(L_B(\theta_B)\), and \(L_B(\theta_C)\). Forgetting is their difference from the corresponding post-acquisition reference:

\[
F_{A\rightarrow B}=L_A(\theta_B)-L_A(\theta_A),\quad
F_{A\rightarrow C}=L_A(\theta_C)-L_A(\theta_A),\quad
F_{B\rightarrow C}=L_B(\theta_C)-L_B(\theta_B).
\]

Mean forgetting is \(F_{\mathrm{mean}}=(F_{A\rightarrow B}+F_{A\rightarrow C}+F_{B\rightarrow C})/3\). Lower NLL and lower forgetting are better. \(F_{\mathrm{mean}}=0\) indicates no average degradation from the post-acquisition references; positive values indicate degradation, and negative values would indicate improvement.

## 3.3 Final memory and behavioral metrics

After phase \(C\), balanced memory is

\[
L_{\mathrm{final}}^{\mathrm{mean}}=\frac{L_A(\theta_C)+L_B(\theta_C)+L_C(\theta_C)}{3}.
\]

For prompt \(q_i\) and target tokens \(a_i=(a_{i,1},\ldots,a_{i,T_i})\), per-example target NLL is

\[
\operatorname{NLL}_i=-\frac{1}{T_i}\sum_{t=1}^{T_i}\log p_\theta(a_{i,t}\mid q_i,a_{i,<t}).
\]

The reported **correct-answer probability** is \(N^{-1}\sum_i\exp(-\operatorname{NLL}_i)\): the mean across examples of the geometric mean target-token probability. It is neither the full target-sequence probability nor the arithmetic mean of token probabilities. Exact match measures exact recovery of the target token sequence. These metrics distinguish graded likelihood from discrete generated recall [9,10,13].

# 4. EWC Method

## 4.1 Online empirical diagonal-Fisher objective

The shared adapter is consolidated with Elastic Weight Consolidation (EWC) [1]. For current-phase loss \(\mathcal L_{\mathrm{current}}\), adapter parameters \(\theta\), consolidated reference \(\theta^\ast\), and diagonal importance values \(F_i\), training minimizes

\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{current}}
+
\frac{\lambda}{2}
\sum_i F_i(\theta_i-\theta_i^\ast)^2.
\]

The implementation estimates an empirical diagonal Fisher over adapter parameters only. For a deterministic set of target examples, it averages squared per-example gradients of the target log probability. It is therefore neither a full Fisher matrix nor an exact expectation over the model distribution.

## 4.2 Sequential consolidation

After phase \(A\), the implementation stores \(\theta_A^\ast\) and estimates \(F_A\). Phase \(B\) minimizes

\[
\mathcal L_B^{\mathrm{total}}
=
\mathcal L_B
+
\frac{\lambda}{2}
\sum_i F_{A,i}(\theta_i-\theta_{A,i}^\ast)^2.
\]

After phase \(B\), a second empirical diagonal estimate \(F_B\) is accumulated online,

\[
F_{AB}=\gamma F_A+F_B,
\qquad \gamma=1,
\]

and the reference is updated to the post-\(B\) parameter state \(\theta_{AB}^\ast\). Phase \(C\) then minimizes

\[
\mathcal L_C^{\mathrm{total}}
=
\mathcal L_C
+
\frac{\lambda}{2}
\sum_i F_{AB,i}(\theta_i-\theta_{AB,i}^\ast)^2.
\]

The Fisher estimates use 12 deterministic phase-\(A\) examples and 12 deterministic phase-\(B\) examples. No previous-phase examples are replayed during subsequent optimization.

## 4.3 Interpretation and controls

At \(\lambda=0\), Fisher and reference states are still computed through the same runner, but the regularizer and its gradient are exactly zero. This is the execution-matched plasticity control. Increasing \(\lambda\) raises resistance to movement along coordinates assigned high previous-fact importance; sufficiently large values can impede acquisition of later facts.

EWC differs from the other ABC controls. `sequential_single` reuses the same 6,144 adapter parameters without consolidation. `grow_unfrozen` adds a fresh slot per phase while keeping all slots trainable. `hard_consolidation` adds a fresh slot and freezes earlier slots exactly. The latter two end with 18,432 active adapter parameters and are mechanistic controls rather than parameter-matched competitors to sequential learning or EWC.

# 5. Experimental Setup

## 5.1 Model, adapters, and initialization

All experiments use a frozen GPT-2 Small backbone [3] with one adapter at zero-based block 0 `attn.c_proj`. LoRA and Symmetric use rank 4 and \(\alpha=4\), giving \(\alpha/r=1\). Combined uses \(r_L=r_S=2\) and \(\alpha_L=\alpha_S=4\), giving branch-wise scales \(\alpha_L/r_L=\alpha_S/r_S=2\). Each architecture has 6,144 trainable adapter parameters, although Combined is not matched to the single-branch methods in branch-wise scale.

For LoRA, \(A\) is initialized with PyTorch Kaiming-uniform initialization using \(a=\sqrt{5}\), and \(B\) is zero. For Symmetric, \(U\) uses the same initializer and \(P\) is zero. Combined independently initializes \(A\) and \(U\) in the same way and sets \(B=P=0\). With the implemented \((d,r)\) factor shapes, the uniform bounds are \(\pm1/\sqrt r\): \(\pm0.5\) at rank 4 and \(\pm1/\sqrt2\) at rank 2. All initial adapter corrections are therefore exactly zero.

## 5.2 Dataset construction

A fixed dataset seed generates three globally disjoint sets of 12 synthetic identifier–code associations. Subjects have the form `QF-` followed by seven uppercase letters; answers include a leading space, `VX-`, and six uppercase letters. Subjects and answers are unique across all sets. Each fact has three training templates, two validation templates, and two paraphrase templates defined in the dataset source. GPT-2 tokenization yields multi-token targets of six to eight tokens in this dataset. The validated dataset SHA-256 is `cc505ec681369c5f9aa4c7123cfff11da878bce17c0669890d416b5eb07f1923`.

## 5.3 Optimization and randomness

Each phase uses 2,500 optimizer steps, for 7,500 steps per \(A\rightarrow B\rightarrow C\) run. AdamW uses learning rate \(3\times10^{-4}\), \((\beta_1,\beta_2)=(0.9,0.999)\), \(\epsilon=10^{-8}\), zero weight decay, gradient clipping at 1.0, and no scheduler. Training uses sequence length 64, FP16 with gradient scaling, microbatch size 4, gradient accumulation 1, and effective batch size 4.

The loss includes target tokens only: prompt labels and padding are set to \(-100\), and cross-entropy is averaged over target tokens per example and then over the microbatch. Prompts and targets are tokenized once into host-memory tensors. GPT-2 remains in training mode during optimization, so its pretrained dropout modules are active even though backbone parameters are frozen; evaluation and Fisher estimation use evaluation mode.

Each training seed controls Python, NumPy, PyTorch CPU, and CUDA RNGs, adapter initialization, dropout, and deterministic per-phase example ordering. PyTorch deterministic algorithms are enabled. There is no DataLoader or separate DataLoader RNG. Dataset generation uses a fixed dataset seed shared by every run. Generation is greedy and therefore introduces no sampling RNG.

## 5.4 EWC and evaluation

The empirical diagonal Fisher uses adapter parameters only and 12 deterministically selected examples after both phases \(A\) and \(B\). Online accumulation uses \(\gamma=1\). The \(\lambda=0\) control computes the same Fisher and reference states through the same runner, while its regularization contribution and gradient are exactly zero.

Evaluation occurs at \(\theta_A,\theta_B,\theta_C\). Exact generation disables sampling, uses one greedy beam (the library default), and sets the maximum continuation length to the target-token length. Generation stops at that limit or at the model EOS rule. The generated suffix is compared directly with the target token IDs. There is no text decoding, answer extraction, whitespace trimming, case folding, or punctuation normalization; the target's leading-space tokenization is part of the exact comparison.

Paraphrase prompts use two held-out templates per fact and the same NLL, correct-answer probability, and exact-token-match definitions. Their results are reported with the confirmation analysis rather than treated as a separate task.

## 5.5 Experimental hierarchy and statistical treatment

The exploratory sweep evaluates \(\lambda\in\{0,1,10,30,50,100,300\}\) for all three methods using seed 42. It selects LoRA \(\lambda\in\{1,10\}\), Symmetric \(\lambda\in\{10,30\}\), and Combined \(\lambda\in\{10,30\}\), retaining \(\lambda=0\) for each. Seeds 123 and 456 are subsequent out-of-selection replications. Tables also give descriptive means and sample standard deviations over seeds 42, 123, and 456; no population-level significance claim is made.

The validated corpus contains 72 unique canonical runs: 9 sequential-single baselines, 9 grow-unfrozen controls, 9 hard-consolidation controls, 9 EWC \(\lambda=100\) mechanistic controls, 18 other exploratory sweep cells, and 18 additional confirmation executions. The nine compatible seed-42 selection cells are referenced in confirmation aggregation rather than duplicated. Grow-unfrozen and hard consolidation end with 18,432 active adapter parameters and serve as auxiliary mechanistic controls, not parameter-matched primary competitors; their complete memory matrices and slot ablations remain in the machine-readable ABC analysis outputs.

# 6. Results

## 6.1 Exploratory lambda sweep

Figure 1 shows the single-seed sweep. Lower NLL and forgetting are better, and both axes are minimized. Across all architectures, increasing \(\lambda\) reduces forgetting while increasing new-fact acquisition NLL.

![Exploratory stability-plasticity sweep](submission_figures/figure_1_exploratory_stability_plasticity.pdf)

### LoRA

LoRA moves from \((F_{\mathrm{mean}}=4.2868,L_{\mathrm{new}}^{\mathrm{acq}}=0.5044)\) at \(\lambda=0\) to \((2.1935,1.0628)\) at \(\lambda=10\); final balanced NLL improves from 3.7514 to 2.4909. At \(\lambda=1\), forgetting falls to 3.9427 while final exact match remains at its \(\lambda=0\) value of 0.0833. Exact match is 0.0139 at \(\lambda=10\) and zero at 30. Both geometric criteria select \(\lambda=10\); \(\lambda=1\) is retained to test the low-coefficient transition.

### Symmetric

At \(\lambda=0\), Symmetric has the lowest exploratory acquisition NLL (0.1232) and the highest mean forgetting (5.6856), with final exact match 0.3333. Increasing \(\lambda\) to 10 changes these metrics to 0.3682 and 3.8739, respectively, while exact match remains 0.1250. At \(\lambda=30\), forgetting falls to 2.7536, acquisition NLL rises to 0.9028, and final balanced NLL reaches the exploratory minimum of 2.3002, although exact match is zero. The selected coefficients are therefore 10 and 30.

### Combined

Combined moves from \((F_{\mathrm{mean}}=5.2562,L_{\mathrm{new}}^{\mathrm{acq}}=0.4037)\) at \(\lambda=0\) to \((3.3880,0.6870)\) at 10 and \((2.5690,1.2213)\) at 30. At \(\lambda=30\), final balanced NLL is 2.3786 and exact match remains 0.0139; exact match reaches zero at 50. The endpoint-chord and ideal-distance criteria select 10 and 30, respectively, so both proceed to replication.

The unregularized operating points summarize the architectural spread:

| Method | \(F_{\mathrm{mean}}\) | \(L_{\mathrm{new}}^{\mathrm{acq}}\) |
|---|---:|---:|
| LoRA | 4.2868 | 0.5044 |
| Symmetric | 5.6856 | 0.1232 |
| Combined | 5.2562 | 0.4037 |

At \(\lambda=300\), acquisition NLL rises to 3.0840 for LoRA, 2.6767 for Symmetric, and 2.9827 for Combined, with zero exact match for every method. The high-coefficient region is thus stable but under-plastic.

For candidate selection, forgetting and acquisition NLL are min–max normalized separately within each architecture: \(z_m=(m-m_{\min})/(m_{\max}-m_{\min})\). The ideal point is \((0,0)\), and the first criterion minimizes \(\sqrt{z_F^2+z_L^2}\). The second criterion takes the perpendicular Euclidean distance from each normalized point to the chord joining that architecture's \(\lambda=0\) and \(\lambda=300\) endpoints, selecting the maximum. `argmin`/`argmax` return the first ascending-\(\lambda\) entry on an exact tie. These descriptive criteria yield:

| Method | Confirmation \(\lambda\) values |
|---|---:|
| LoRA | \(0,1,10\) |
| Symmetric | \(0,10,30\) |
| Combined | \(0,10,30\) |

## 6.2 Multi-seed replication

Candidate coefficients selected on seed 42 were evaluated on the two unseen seeds 123 and 456. Table 4 reports descriptive aggregates over all three seeds. Every non-zero candidate reduces mean forgetting, raises acquisition NLL, and improves final balanced NLL relative to its execution-matched control in 3/3 paired seeds; the latter count includes the selection seed, with the same direction reproduced on both unseen seeds.

| Method | $\lambda$ | Mean forgetting | New acquisition NLL | Final balanced NLL | Final correct-answer probability | Final exact match |
|---|---:|---:|---:|---:|---:|---:|
| LoRA | 0 | 4.4847 ± 0.3475 | 0.4435 ± 0.0853 | 3.6856 ± 0.0662 | 0.2285 ± 0.0129 | 0.0833 ± 0.0139 |
| LoRA | 1 | 4.1844 ± 0.4172 | 0.4920 ± 0.0963 | 3.4855 ± 0.0915 | 0.2173 ± 0.0146 | 0.0787 ± 0.0080 |
| LoRA | 10 | 2.5429 ± 0.6715 | 0.9805 ± 0.2138 | 2.6169 ± 0.2279 | 0.1510 ± 0.0165 | 0.0046 ± 0.0080 |
| Symmetric | 0 | 5.7811 ± 0.0958 | 0.1454 ± 0.0466 | 4.0117 ± 0.1490 | 0.2979 ± 0.0183 | 0.2778 ± 0.0735 |
| Symmetric | 10 | 4.2774 ± 0.3552 | 0.3189 ± 0.0660 | 3.0044 ± 0.2639 | 0.2465 ± 0.0133 | 0.1343 ± 0.0289 |
| Symmetric | 30 | 3.1026 ± 0.4115 | 0.7046 ± 0.1744 | 2.3888 ± 0.1854 | 0.1782 ± 0.0208 | 0.0185 ± 0.0160 |
| Combined | 0 | 5.5329 ± 0.2459 | 0.2991 ± 0.0922 | 3.9493 ± 0.0523 | 0.2553 ± 0.0206 | 0.1343 ± 0.0578 |
| Combined | 10 | 3.9336 ± 0.4809 | 0.5758 ± 0.1149 | 2.9572 ± 0.3171 | 0.1996 ± 0.0231 | 0.0648 ± 0.0424 |
| Combined | 30 | 2.7443 ± 0.3321 | 1.1637 ± 0.2463 | 2.4373 ± 0.0694 | 0.1259 ± 0.0134 | 0.0046 ± 0.0080 |

![Multi-seed stability-plasticity confirmation](submission_figures/figure_2_confirmation_stability_plasticity.pdf)

LoRA exhibits a measurable response at \(\lambda=1\) and a larger stability shift at 10. Symmetric preserves stronger acquisition at moderate regularization and moves to its lowest confirmed balanced NLL at 30. Combined follows an intermediate path without dominating both single-family adapters. At the shared \(\lambda=10\), LoRA has the lowest forgetting, Symmetric the lowest acquisition NLL, and Combined lies between them; nominal \(\lambda\) therefore does not equalize the functional constraint across parameterizations.

## 6.3 Absolute retention trajectories

Figure 3 reports the absolute NLL on set \(A\) immediately after acquisition and after phases \(B\) and \(C\). Lower NLL is better; the plotted differences yield \(F_{A\rightarrow B}\) and \(F_{A\rightarrow C}\) exactly. Regularization preserves more of the original likelihood support but begins from the same post-\(A\) state within each method because EWC is inactive during phase \(A\).

![Absolute retention NLL for factual set A](submission_figures/figure_absolute_retention_nll.pdf)

Table 5 gives the corresponding set-\(B\) trajectory; \(F_{B\rightarrow C}=L_B(\theta_C)-L_B(\theta_B)\).

| Method | $\lambda$ | $L_B(\theta_B)$ | $L_B(\theta_C)$ |
|---|---:|---:|---:|
| LoRA | 0 | 0.4580 ± 0.1142 | 4.3649 ± 0.0749 |
| LoRA | 1 | 0.4835 ± 0.1158 | 3.9718 ± 0.2050 |
| LoRA | 10 | 0.8311 ± 0.2066 | 2.7061 ± 0.2324 |
| Symmetric | 0 | 0.1611 ± 0.0348 | 5.3060 ± 0.2687 |
| Symmetric | 10 | 0.2443 ± 0.0579 | 3.5056 ± 0.4070 |
| Symmetric | 30 | 0.4135 ± 0.1250 | 2.4164 ± 0.1925 |
| Combined | 0 | 0.3000 ± 0.0962 | 4.9151 ± 0.2396 |
| Combined | 10 | 0.4970 ± 0.0529 | 3.3386 ± 0.5716 |
| Combined | 30 | 0.7790 ± 0.2327 | 2.3111 ± 0.3157 |

## 6.4 Balanced memory, exact recall, and paraphrases

Final balanced NLL is non-monotonic because it combines retention and acquisition. Figure 4 shows the best sampled confirmation settings at LoRA \(\lambda=10\), Symmetric \(\lambda=30\), and Combined \(\lambda=30\). Their means are 2.6169, 2.3888, and 2.4373, respectively; three seeds do not support a universal ranking.

![Final balanced memory](submission_figures/figure_3_final_balanced_memory.pdf)

Figure 5 decomposes forgetting. The largest component remains \(F_{A\rightarrow C}\), which spans two subsequent phases.

![Forgetting decomposition](submission_figures/figure_4_forgetting_decomposition.pdf)

Exact generation declines more sharply than continuous likelihood metrics (Figure 6). Symmetric \(\lambda=30\), for example, has final exact match \(0.0185\pm0.0160\) despite its low balanced NLL. The same distinction appears on paraphrases. At the selected strongest candidates, final paraphrase NLL / correct-answer probability / exact match are \(2.6298\pm0.2457/0.1445\pm0.0129/0\) for LoRA \(\lambda=10\), \(2.4907\pm0.1403/0.1670\pm0.0259/0.0231\pm0.0289\) for Symmetric \(\lambda=30\), and \(2.5475\pm0.1150/0.1130\pm0.0071/0\) for Combined \(\lambda=30\). Thus, balanced NLL improvements do not imply uniformly improved exact recall.

![Plasticity and exact generation](submission_figures/figure_5_plasticity_exact_generation.pdf)

# 7. Mechanistic Analysis

## 7.1 Parameter displacement

Adapter movement is measured by \(D_{A\rightarrow B}=\|\theta_B-\theta_A\|_2\), \(D_{B\rightarrow C}=\|\theta_C-\theta_B\|_2\), and \(D_{A\rightarrow C}=\|\theta_C-\theta_A\|_2\). Across the exploratory sweep, cumulative displacement decreases in all six adjacent lambda transitions for LoRA and in five of six for Symmetric and Combined (Figure 7).

![Mechanistic evidence](submission_figures/figure_6_mechanistic_evidence.pdf)

Cumulative displacement is positively associated with forgetting:

| Method | Pearson \(r\) | Spearman \(\rho\) |
|---|---:|---:|
| LoRA | 0.7873 | 1.0000 |
| Symmetric | 0.6930 | 0.9643 |
| Combined | 0.7471 | 0.8929 |

The association is descriptive and does not establish that Euclidean movement causes forgetting.

## 7.2 Lambda, displacement, and behavior

Because the lambda grid is non-uniform, Spearman \(\rho\) is the primary monotonic descriptor. Raw-lambda Pearson coefficients are retained for comparability, and Pearson correlations against \(\log(1+\lambda)\) provide a scale-sensitivity check.

| Method | Pearson \(r(\lambda,D_{{A\rightarrow C}})\) | Pearson \(r(\log(1+\lambda),D_{{A\rightarrow C}})\) | Spearman \(\rho\) |
|---|---:|---:|---:|
| LoRA | -0.9478 | -0.8994 | -1.0000 |
| Symmetric | -0.9915 | -0.7290 | -0.9643 |
| Combined | -0.9769 | -0.7847 | -0.8929 |

| Method | Pearson \(r(\lambda,F_{{\mathrm{{mean}}}})\) | Pearson \(r(\log(1+\lambda),F_{{\mathrm{{mean}}}})\) | Spearman \(\rho\) |
|---|---:|---:|---:|
| LoRA | -0.6321 | -0.9749 | -1.0000 |
| Symmetric | -0.7502 | -0.9964 | -1.0000 |
| Combined | -0.7358 | -0.9967 | -1.0000 |

| Method | Pearson \(r(\lambda,L_{{\mathrm{{new}}}}^{{\mathrm{{acq}}}})\) | Pearson \(r(\log(1+\lambda),L_{{\mathrm{{new}}}}^{{\mathrm{{acq}}}})\) | Spearman \(\rho\) |
|---|---:|---:|---:|
| LoRA | 0.7717 | 0.9797 | 1.0000 |
| Symmetric | 0.9267 | 0.9341 | 1.0000 |
| Combined | 0.8992 | 0.9383 | 1.0000 |

The perfect rank correlations describe the ordered seven-point trajectories; no significance test is applied.

## 7.3 Fisher structure and realized constraint

Within each architecture, \(F_A\) is identical across lambda values because phase \(A\) precedes consolidation. The later state \(F_{AB}=F_A+F_B\) depends on the regularized phase-\(B\) trajectory. Fisher mass is concentrated mainly in output-side factors, but raw magnitudes are coordinate dependent and are not compared as architecture-independent functional importance.

The realized cost, \(P_{\mathrm{EWC}}=(\lambda/2)\sum_iF_i(\theta_i-\theta_i^\ast)^2\), combines the coefficient, Fisher profile, and actual trajectory. Its phase-\(C\) correlations with forgetting are:

| Method | Pearson \(r\) | Spearman \(\rho\) |
|---|---:|---:|
| LoRA | -0.4391 | -0.1071 |
| Symmetric | -0.5507 | -0.3929 |
| Combined | -0.4043 | -0.2857 |

For acquisition NLL, the corresponding coefficients are:

| Method | Pearson \(r\) | Spearman \(\rho\) |
|---|---:|---:|
| LoRA | 0.1581 | 0.1071 |
| Symmetric | 0.1904 | 0.3929 |
| Combined | 0.0287 | 0.2857 |

The scalar realized penalty is therefore insufficient to identify the behavioral regime.

## 7.4 Coordinate dependence and functional interference

EWC and Euclidean displacement operate in native parameter coordinates. For LoRA, \(A\mapsto cA\) and \(B\mapsto B/c\) preserve \(AB\); for the quadratic factorization, \(U\mapsto cU\) and \(P\mapsto P/c^2\) preserve the idealized quadratic form. Functionally equivalent updates can therefore have different parameter magnitudes, diagonal Fisher profiles, and effective EWC strengths. The architecture-dependent response may reflect functional geometry, coordinate scaling, conditioning, or their interaction.

Hard-consolidation controls expose a complementary distinction: old adapter tensors remain bit-identical, yet later active slots can degrade earlier behavior. Parameter preservation does not guarantee functional preservation because the output depends on the complete active computation. Together, these observations support a parameter-space account consistent with the behavioral data, but not a causal or coordinate-invariant mechanism.

# 8. Discussion

## 8.1 Architecture-dependent EWC response

The different transition scales show that a nominal EWC coefficient is not an architecture-independent consolidation measure. EWC constrains parameters, whereas its behavioral effect depends on their map to model outputs. Under the evaluated protocol, LoRA responds at lower coefficients, Symmetric remains plastic over a wider moderate range, and Combined lies between them without systematic dominance. This is an empirical response of the tested parameterizations, not evidence of inherent stability or plasticity for an entire adapter family.

The Combined comparison also couples geometry, rank allocation, and scale. Although all methods contain 6,144 trainable parameters, single branches use \(\alpha/r=1\), while each Combined branch uses scale 2. Its behavior therefore cannot be attributed uniquely to mixing linear and quadratic functions.

## 8.2 Parameter-space versus functional geometry

Matching parameter count controls storage and trainable dimension, not functional capacity or conditioning. The observed reduction in displacement is consistent with EWC's intended operation, but the reparameterization symmetries in Section 7 show why Euclidean distance and a diagonal Fisher are coordinate dependent. Architecture-specific effective lambda may arise from the function class, factor scaling, conditioning, or all three. The correlations provide mechanistic consistency without isolating these causes.

## 8.3 Likelihood, exact recall, and functional preservation

NLL measures graded target support, whereas exact match requires that support to determine every decoded token. Intermediate consolidation can improve balanced likelihood while reducing exact acquisition of the newest facts. Absolute retention trajectories make the trade-off explicit: the same constraint that limits growth in old-fact NLL also raises the post-acquisition loss of later facts.

Hard consolidation further shows that unchanged old parameters do not imply an unchanged function when new components remain active. Behavioral retention, rather than parameter preservation alone, is therefore the relevant endpoint for continual PEFT.

## 8.4 Implications for continual PEFT

Static adaptation quality and trainable-parameter count do not predict behavior under repeated updates. Adapter geometry, factor scaling, and consolidation strength should be selected jointly for the required balance of retention, acquisition, and exact recall. The protocol differs from one-shot factual editing [9–12], but shares its concern with persistence under later changes. The present evidence identifies controlled operating regimes; it does not establish a universal adapter or coefficient.

# 9. Limitations

The experiments use GPT-2 Small, one adapter at zero-based block 0 `attn.c_proj`, three 12-fact synthetic sets, and a three-phase sequence. They exclude replay and do not cover realistic factual diversity, longer histories, other placements, larger models, or alternative PEFT and consolidation methods. Candidate selection uses seed 42; seeds 123 and 456 provide two out-of-selection replications, while the reported three-seed aggregates are descriptive rather than a basis for population-level significance.

The empirical Fisher is diagonal and estimated from 12 deterministic examples after each consolidated phase, with online accumulation fixed at \(\gamma=1\). It ignores parameter correlations and may become restrictive over longer sequences. The lambda grid is sparse and non-uniform; knee locations depend on its endpoints and per-architecture normalization, and raw Pearson correlations depend on the numerical lambda scale. Mechanistic correlations over seven points are associative rather than causal.

Exact match uses strict greedy token equality and can diverge from likelihood-based metrics. Balanced NLL weights the three sets equally and encodes no application-specific preference. Parameter matching is not functional matching: all primary adapters have 6,144 trainable parameters, whereas grow-unfrozen and hard-consolidation controls reach 18,432 active parameters and serve only as mechanistic context.

Combined fixes \(r_L=r_S=2\), uses a shared lambda, and assigns scale 2 to both branches, compared with scale 1 for the single-branch methods. Rank allocation and branch-wise scaling are therefore unresolved confounds. More generally, reparameterization can preserve adapter functions while changing Euclidean magnitudes and diagonal Fisher coordinates. The study cannot isolate whether architecture-dependent response arises from functional geometry, coordinate scale, initialization, conditioning, or another coupled property.

# 10. Conclusions

Online EWC provides a controllable stability–plasticity trade-off for fixed-budget continual factual learning under the evaluated GPT-2 Small protocol. Increasing regularization preserves earlier facts but weakens acquisition of later ones.

The effective coefficient scale depends on adapter parameterization. LoRA shifts toward stability at lower lambda, Symmetric remains more plastic under moderate regularization, and Combined does not systematically dominate. Intermediate coefficients provide the strongest sampled balanced-memory regimes, while high coefficients produce under-learning and weak exact recall.

Smaller adapter displacement accompanies lower forgetting, but this relationship is coordinate dependent and associative. Equal parameter count, and even exact preservation of old parameters, does not imply equal functional memory.

These results support joint selection of adapter geometry and consolidation strength under a specified continual-learning objective. Larger models, realistic and longer fact sequences, more placements and seeds, and direct controls over branch scaling and parameterization are needed before the observed coefficient regimes can be generalized.

# References

[1] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins,
A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska,
D. Hassabis, C. Clopath, D. Kumaran, and R. Hadsell,
“Overcoming catastrophic forgetting in neural networks,”
Proceedings of the National Academy of Sciences,
vol. 114, no. 13, pp. 3521–3526, 2017.
DOI: 10.1073/pnas.1611835114

[2] E. J. Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” International Conference on Learning Representations, 2022. arXiv:2106.09685.

[3] A. Radford et al., “Language Models are Unsupervised Multitask Learners,” OpenAI Technical Report, 2019.

[4] N. Houlsby et al., “Parameter-Efficient Transfer Learning for NLP,” Proceedings of ICML, PMLR 97:2790–2799, 2019.

[5] X. L. Li and P. Liang, “Prefix-Tuning: Optimizing Continuous Prompts for Generation,” Proceedings of ACL-IJCNLP, pp. 4582–4597, 2021. doi:10.18653/v1/2021.acl-long.353.

[6] B. Lester, R. Al-Rfou, and N. Constant, “The Power of Scale for Parameter-Efficient Prompt Tuning,” Proceedings of EMNLP, pp. 3045–3059, 2021. doi:10.18653/v1/2021.emnlp-main.243.

[7] Z. Ke and B. Liu, “Continual Learning of Natural Language Processing Tasks: A Survey,” arXiv:2211.12701, 2022.

[8] M. Biesialska, K. Biesialska, and M. R. Costa-jussà, “Continual Lifelong Learning in Natural Language Processing: A Survey,” Proceedings of COLING, pp. 6523–6541, 2020. doi:10.18653/v1/2020.coling-main.574.

[9] K. Meng et al., “Locating and Editing Factual Associations in GPT,” Advances in Neural Information Processing Systems, vol. 35, 2022. arXiv:2202.05262.

[10] K. Meng et al., “Mass-Editing Memory in a Transformer,” International Conference on Learning Representations, 2023. arXiv:2210.07229.

[11] E. Mitchell et al., “Fast Model Editing at Scale,” International Conference on Learning Representations, 2022. arXiv:2110.11309.

[12] E. Mitchell et al., “Memory-Based Model Editing at Scale,” Proceedings of ICML, PMLR 162:15817–15831, 2022. arXiv:2206.06520.

[13] S. Wang et al., “Knowledge Editing for Large Language Models: A Survey,” ACM Computing Surveys, 57(3), Article 59, 2024. doi:10.1145/3698590.

[14] N. Zhang et al., “A Comprehensive Study of Knowledge Editing for Large Language Models,” arXiv:2401.01286, 2024.
