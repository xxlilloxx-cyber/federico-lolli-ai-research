#!/usr/bin/env python3
"""Create the reviewer-level submission Markdown from the polished manuscript."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "FINAL_PAPER_FINAL.md"
OUTPUT = ROOT / "FINAL_PAPER_SUBMISSION.md"
ANALYSIS = ROOT.parent
METHOD_ORDER = {"lora": 0, "symmetric": 1, "combined": 2}
METHOD_LABEL = {"lora": "LoRA", "symmetric": "Symmetric", "combined": "Combined"}


def replace_section(text: str, start: str, end: str, replacement: str) -> str:
    pattern = re.compile(rf"^{re.escape(start)}\n.*?(?=^{re.escape(end)}\n)", re.M | re.S)
    rendered = replacement.rstrip() + "\n\n"
    result, count = pattern.subn(lambda _: rendered, text)
    if count != 1:
        raise RuntimeError(f"Expected one section {start!r}, found {count}")
    return result


def confirmation_table() -> str:
    data = pd.read_csv(ANALYSIS / "tables" / "confirmation_aggregate.csv")
    data["_order"] = data.method.map(METHOD_ORDER)
    data = data.sort_values(["_order", "lambda"])
    lines = [
        "| Method | $\\lambda$ | Mean forgetting | New acquisition NLL | Final balanced NLL | Final correct-answer probability | Final exact match |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in data.iterrows():
        pm = lambda key: f"{r[key + '_mean']:.4f} ± {r[key + '_sample_sd']:.4f}"
        lines.append(
            f"| {METHOD_LABEL[r.method]} | {int(r['lambda'])} | {pm('mean_forgetting')} | "
            f"{pm('mean_new_fact_acquisition_nll')} | {pm('mean_final_memory_nll')} | "
            f"{pm('mean_final_probability')} | {pm('mean_final_exact_match')} |"
        )
    return "\n".join(lines)


def b_retention_table() -> str:
    data = pd.read_csv(ROOT / "submission_data" / "absolute_retention_aggregate.csv")
    data["_order"] = data.method.map(METHOD_ORDER)
    data = data.sort_values(["_order", "lambda"])
    lines = [
        "| Method | $\\lambda$ | $L_B(\\theta_B)$ | $L_B(\\theta_C)$ |",
        "|---|---:|---:|---:|",
    ]
    for _, r in data.iterrows():
        lines.append(
            f"| {METHOD_LABEL[r.method]} | {int(r['lambda'])} | "
            f"{r.b_theta_b_mean:.4f} ± {r.b_theta_b_sample_sd:.4f} | "
            f"{r.b_theta_c_mean:.4f} ± {r.b_theta_c_sample_sd:.4f} |"
        )
    return "\n".join(lines)


ABSTRACT = r'''## Abstract

Sequential parameter-efficient adaptation must balance acquisition of new information against retention of earlier updates. We study this problem in a no-replay sequence of three disjoint synthetic factual sets using a frozen GPT-2 Small backbone and three adapters with 6,144 trainable parameters: LoRA, a symmetric factorized quadratic adapter, and a Combined linear–quadratic adapter. Consolidation uses online Elastic Weight Consolidation (EWC) with an empirical diagonal Fisher over adapter parameters. A seven-point sweep on seed 42 selects architecture-specific coefficient ranges; the selected settings are then evaluated on two additional seeds, with descriptive aggregates over all three. Stronger EWC consistently reduces forgetting while increasing new-fact acquisition NLL. Intermediate coefficients improve final balanced memory relative to the execution-matched \(\lambda=0\) control in every tested candidate configuration and each paired seed. The transition scale depends on adapter parameterization: LoRA responds at lower coefficients, Symmetric remains more plastic under moderate regularization, and Combined does not systematically dominate either component architecture. Increasing EWC strength also reduces parameter displacement, which is positively associated with forgetting. Under the evaluated setting, these findings support treating consolidation strength and adapter geometry as coupled design choices. Scope is limited to GPT-2 Small, one placement, three 12-fact sets, and two out-of-selection replication seeds.

**Keywords:** continual learning; EWC; parameter-efficient fine-tuning; LoRA; quadratic adapters; factual learning; GPT-2'''


INTRODUCTION = r'''# 1. Introduction

Parameter-efficient fine-tuning (PEFT) adapts pretrained language models through a small trainable parameter subset [4]. LoRA uses low-rank factors [2], while prompt- and prefix-based methods provide related alternatives [5,6]. These approaches reduce update and storage costs but do not eliminate interference under sequential adaptation [1,7,8]. The problem is especially relevant when factual associations are introduced over time.

Knowledge-editing methods including ROME, MEMIT, MEND, and SERAC modify factual behavior without full-model retraining [9–12]. We study a related no-replay setting in which one parameter-efficient pathway is optimized sequentially on three disjoint factual sets. The central question is: **How does EWC strength control the stability–plasticity trade-off across linear, quadratic, and combined low-rank adapters with the same trainable-parameter budget?**

We compare 6,144-parameter LoRA, Symmetric quadratic, and Combined linear–quadratic adapters at one projection of a frozen GPT-2 Small model [3]. Training follows \(A\rightarrow B\rightarrow C\) with online EWC [1]. A sweep over \(\lambda\in\{0,1,10,30,50,100,300\}\) on seed 42 characterizes each response curve and determines candidate coefficients. Those candidates are subsequently evaluated on two unseen seeds, 123 and 456; descriptive three-seed aggregates include the selection seed. The \(\lambda=0\) condition follows the same EWC runner and isolates the regularization term.

The study contributes (i) a controlled comparison of three adapter geometries under a common budget and continual-learning protocol; (ii) a characterization of EWC strength as a stability–plasticity control; (iii) out-of-selection replication of candidate regimes on two additional seeds; and (iv) parameter-displacement and Fisher diagnostics that qualify the behavioral results. The exploratory sweep and replicated candidate evaluation are kept distinct throughout.'''


ARCHITECTURES = r'''# 2. Adapter Architectures

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

The study uses \(r_L=r_S=2\) and \(\alpha_L=\alpha_S=4\), for \((r_L+r_S)(d+d_{\mathrm{out}})=6{,}144\) parameters. Its Jacobian is \(J_C(x)=J_L+J_S(x)\), and its adapter-only Hessian comes from the quadratic branch. Combined is thus a mixed function class under the same total budget; it is not a same-local-rank or same-branch-scale comparison with the rank-4 single-family adapters.'''


CONTINUAL = r'''# 3. Continual-Learning Formulation

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

The reported **correct-answer probability** is \(N^{-1}\sum_i\exp(-\operatorname{NLL}_i)\): the mean across examples of the geometric mean target-token probability. It is neither the full target-sequence probability nor the arithmetic mean of token probabilities. Exact match measures exact recovery of the target token sequence. These metrics distinguish graded likelihood from discrete generated recall [9,10,13].'''


EXPERIMENTAL = r'''# 5. Experimental Setup

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

The validated corpus contains 72 unique canonical runs: 9 sequential-single baselines, 9 grow-unfrozen controls, 9 hard-consolidation controls, 9 EWC \(\lambda=100\) mechanistic controls, 18 other exploratory sweep cells, and 18 additional confirmation executions. The nine compatible seed-42 selection cells are referenced in confirmation aggregation rather than duplicated. Grow-unfrozen and hard consolidation end with 18,432 active adapter parameters and serve as auxiliary mechanistic controls, not parameter-matched primary competitors; their complete memory matrices and slot ablations remain in the machine-readable ABC analysis outputs.'''


def results_section() -> str:
    return rf'''# 6. Results

## 6.1 Exploratory lambda sweep

Figure 1 shows the single-seed sweep. Lower NLL and forgetting are better, and both axes are minimized. Across all architectures, increasing \(\lambda\) reduces forgetting while increasing new-fact acquisition NLL.

![Exploratory stability-plasticity sweep](submission_figures/figure_1_exploratory_stability_plasticity.pdf)

### LoRA

LoRA moves from \((F_{{\mathrm{{mean}}}}=4.2868,L_{{\mathrm{{new}}}}^{{\mathrm{{acq}}}}=0.5044)\) at \(\lambda=0\) to \((2.1935,1.0628)\) at \(\lambda=10\); final balanced NLL improves from 3.7514 to 2.4909. At \(\lambda=1\), forgetting falls to 3.9427 while final exact match remains at its \(\lambda=0\) value of 0.0833. Exact match is 0.0139 at \(\lambda=10\) and zero at 30. Both geometric criteria select \(\lambda=10\); \(\lambda=1\) is retained to test the low-coefficient transition.

### Symmetric

At \(\lambda=0\), Symmetric has the lowest exploratory acquisition NLL (0.1232) and the highest mean forgetting (5.6856), with final exact match 0.3333. Increasing \(\lambda\) to 10 changes these metrics to 0.3682 and 3.8739, respectively, while exact match remains 0.1250. At \(\lambda=30\), forgetting falls to 2.7536, acquisition NLL rises to 0.9028, and final balanced NLL reaches the exploratory minimum of 2.3002, although exact match is zero. The selected coefficients are therefore 10 and 30.

### Combined

Combined moves from \((F_{{\mathrm{{mean}}}}=5.2562,L_{{\mathrm{{new}}}}^{{\mathrm{{acq}}}}=0.4037)\) at \(\lambda=0\) to \((3.3880,0.6870)\) at 10 and \((2.5690,1.2213)\) at 30. At \(\lambda=30\), final balanced NLL is 2.3786 and exact match remains 0.0139; exact match reaches zero at 50. The endpoint-chord and ideal-distance criteria select 10 and 30, respectively, so both proceed to replication.

The unregularized operating points summarize the architectural spread:

| Method | \(F_{{\mathrm{{mean}}}}\) | \(L_{{\mathrm{{new}}}}^{{\mathrm{{acq}}}}\) |
|---|---:|---:|
| LoRA | 4.2868 | 0.5044 |
| Symmetric | 5.6856 | 0.1232 |
| Combined | 5.2562 | 0.4037 |

At \(\lambda=300\), acquisition NLL rises to 3.0840 for LoRA, 2.6767 for Symmetric, and 2.9827 for Combined, with zero exact match for every method. The high-coefficient region is thus stable but under-plastic.

For candidate selection, forgetting and acquisition NLL are min–max normalized separately within each architecture: \(z_m=(m-m_{{\min}})/(m_{{\max}}-m_{{\min}})\). The ideal point is \((0,0)\), and the first criterion minimizes \(\sqrt{{z_F^2+z_L^2}}\). The second criterion takes the perpendicular Euclidean distance from each normalized point to the chord joining that architecture's \(\lambda=0\) and \(\lambda=300\) endpoints, selecting the maximum. `argmin`/`argmax` return the first ascending-\(\lambda\) entry on an exact tie. These descriptive criteria yield:

| Method | Confirmation \(\lambda\) values |
|---|---:|
| LoRA | \(0,1,10\) |
| Symmetric | \(0,10,30\) |
| Combined | \(0,10,30\) |

## 6.2 Multi-seed replication

Candidate coefficients selected on seed 42 were evaluated on the two unseen seeds 123 and 456. Table 4 reports descriptive aggregates over all three seeds. Every non-zero candidate reduces mean forgetting, raises acquisition NLL, and improves final balanced NLL relative to its execution-matched control in 3/3 paired seeds; the latter count includes the selection seed, with the same direction reproduced on both unseen seeds.

{confirmation_table()}

![Multi-seed stability-plasticity confirmation](submission_figures/figure_2_confirmation_stability_plasticity.pdf)

LoRA exhibits a measurable response at \(\lambda=1\) and a larger stability shift at 10. Symmetric preserves stronger acquisition at moderate regularization and moves to its lowest confirmed balanced NLL at 30. Combined follows an intermediate path without dominating both single-family adapters. At the shared \(\lambda=10\), LoRA has the lowest forgetting, Symmetric the lowest acquisition NLL, and Combined lies between them; nominal \(\lambda\) therefore does not equalize the functional constraint across parameterizations.

## 6.3 Absolute retention trajectories

Figure 3 reports the absolute NLL on set \(A\) immediately after acquisition and after phases \(B\) and \(C\). Lower NLL is better; the plotted differences yield \(F_{{A\rightarrow B}}\) and \(F_{{A\rightarrow C}}\) exactly. Regularization preserves more of the original likelihood support but begins from the same post-\(A\) state within each method because EWC is inactive during phase \(A\).

![Absolute retention NLL for factual set A](submission_figures/figure_absolute_retention_nll.pdf)

Table 5 gives the corresponding set-\(B\) trajectory; \(F_{{B\rightarrow C}}=L_B(\theta_C)-L_B(\theta_B)\).

{b_retention_table()}

## 6.4 Balanced memory, exact recall, and paraphrases

Final balanced NLL is non-monotonic because it combines retention and acquisition. Figure 4 shows the best sampled confirmation settings at LoRA \(\lambda=10\), Symmetric \(\lambda=30\), and Combined \(\lambda=30\). Their means are 2.6169, 2.3888, and 2.4373, respectively; three seeds do not support a universal ranking.

![Final balanced memory](submission_figures/figure_3_final_balanced_memory.pdf)

Figure 5 decomposes forgetting. The largest component remains \(F_{{A\rightarrow C}}\), which spans two subsequent phases.

![Forgetting decomposition](submission_figures/figure_4_forgetting_decomposition.pdf)

Exact generation declines more sharply than continuous likelihood metrics (Figure 6). Symmetric \(\lambda=30\), for example, has final exact match \(0.0185\pm0.0160\) despite its low balanced NLL. The same distinction appears on paraphrases. At the selected strongest candidates, final paraphrase NLL / correct-answer probability / exact match are \(2.6298\pm0.2457/0.1445\pm0.0129/0\) for LoRA \(\lambda=10\), \(2.4907\pm0.1403/0.1670\pm0.0259/0.0231\pm0.0289\) for Symmetric \(\lambda=30\), and \(2.5475\pm0.1150/0.1130\pm0.0071/0\) for Combined \(\lambda=30\). Thus, balanced NLL improvements do not imply uniformly improved exact recall.

![Plasticity and exact generation](submission_figures/figure_5_plasticity_exact_generation.pdf)
'''


MECHANISTIC = r'''# 7. Mechanistic Analysis

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

Hard-consolidation controls expose a complementary distinction: old adapter tensors remain bit-identical, yet later active slots can degrade earlier behavior. Parameter preservation does not guarantee functional preservation because the output depends on the complete active computation. Together, these observations support a parameter-space account consistent with the behavioral data, but not a causal or coordinate-invariant mechanism.'''


DISCUSSION = r'''# 8. Discussion

## 8.1 Architecture-dependent EWC response

The different transition scales show that a nominal EWC coefficient is not an architecture-independent consolidation measure. EWC constrains parameters, whereas its behavioral effect depends on their map to model outputs. Under the evaluated protocol, LoRA responds at lower coefficients, Symmetric remains plastic over a wider moderate range, and Combined lies between them without systematic dominance. This is an empirical response of the tested parameterizations, not evidence of inherent stability or plasticity for an entire adapter family.

The Combined comparison also couples geometry, rank allocation, and scale. Although all methods contain 6,144 trainable parameters, single branches use \(\alpha/r=1\), while each Combined branch uses scale 2. Its behavior therefore cannot be attributed uniquely to mixing linear and quadratic functions.

## 8.2 Parameter-space versus functional geometry

Matching parameter count controls storage and trainable dimension, not functional capacity or conditioning. The observed reduction in displacement is consistent with EWC's intended operation, but the reparameterization symmetries in Section 7 show why Euclidean distance and a diagonal Fisher are coordinate dependent. Architecture-specific effective lambda may arise from the function class, factor scaling, conditioning, or all three. The correlations provide mechanistic consistency without isolating these causes.

## 8.3 Likelihood, exact recall, and functional preservation

NLL measures graded target support, whereas exact match requires that support to determine every decoded token. Intermediate consolidation can improve balanced likelihood while reducing exact acquisition of the newest facts. Absolute retention trajectories make the trade-off explicit: the same constraint that limits growth in old-fact NLL also raises the post-acquisition loss of later facts.

Hard consolidation further shows that unchanged old parameters do not imply an unchanged function when new components remain active. Behavioral retention, rather than parameter preservation alone, is therefore the relevant endpoint for continual PEFT.

## 8.4 Implications for continual PEFT

Static adaptation quality and trainable-parameter count do not predict behavior under repeated updates. Adapter geometry, factor scaling, and consolidation strength should be selected jointly for the required balance of retention, acquisition, and exact recall. The protocol differs from one-shot factual editing [9–12], but shares its concern with persistence under later changes. The present evidence identifies controlled operating regimes; it does not establish a universal adapter or coefficient.'''


LIMITATIONS = r'''# 9. Limitations

The experiments use GPT-2 Small, one adapter at zero-based block 0 `attn.c_proj`, three 12-fact synthetic sets, and a three-phase sequence. They exclude replay and do not cover realistic factual diversity, longer histories, other placements, larger models, or alternative PEFT and consolidation methods. Candidate selection uses seed 42; seeds 123 and 456 provide two out-of-selection replications, while the reported three-seed aggregates are descriptive rather than a basis for population-level significance.

The empirical Fisher is diagonal and estimated from 12 deterministic examples after each consolidated phase, with online accumulation fixed at \(\gamma=1\). It ignores parameter correlations and may become restrictive over longer sequences. The lambda grid is sparse and non-uniform; knee locations depend on its endpoints and per-architecture normalization, and raw Pearson correlations depend on the numerical lambda scale. Mechanistic correlations over seven points are associative rather than causal.

Exact match uses strict greedy token equality and can diverge from likelihood-based metrics. Balanced NLL weights the three sets equally and encodes no application-specific preference. Parameter matching is not functional matching: all primary adapters have 6,144 trainable parameters, whereas grow-unfrozen and hard-consolidation controls reach 18,432 active parameters and serve only as mechanistic context.

Combined fixes \(r_L=r_S=2\), uses a shared lambda, and assigns scale 2 to both branches, compared with scale 1 for the single-branch methods. Rank allocation and branch-wise scaling are therefore unresolved confounds. More generally, reparameterization can preserve adapter functions while changing Euclidean magnitudes and diagonal Fisher coordinates. The study cannot isolate whether architecture-dependent response arises from functional geometry, coordinate scale, initialization, conditioning, or another coupled property.'''


CONCLUSIONS = r'''# 10. Conclusions

Online EWC provides a controllable stability–plasticity trade-off for fixed-budget continual factual learning under the evaluated GPT-2 Small protocol. Increasing regularization preserves earlier facts but weakens acquisition of later ones.

The effective coefficient scale depends on adapter parameterization. LoRA shifts toward stability at lower lambda, Symmetric remains more plastic under moderate regularization, and Combined does not systematically dominate. Intermediate coefficients provide the strongest sampled balanced-memory regimes, while high coefficients produce under-learning and weak exact recall.

Smaller adapter displacement accompanies lower forgetting, but this relationship is coordinate dependent and associative. Equal parameter count, and even exact preservation of old parameters, does not imply equal functional memory.

These results support joint selection of adapter geometry and consolidation strength under a specified continual-learning objective. Larger models, realistic and longer fact sequences, more placements and seeds, and direct controls over branch scaling and parameterization are needed before the observed coefficient regimes can be generalized.'''


def main() -> None:
    text = SOURCE.read_text()
    text = replace_section(text, "## Abstract", "# 1. Introduction", ABSTRACT)
    text = replace_section(text, "# 1. Introduction", "# 2. Adapter Architectures", INTRODUCTION)
    text = replace_section(text, "# 2. Adapter Architectures", "# 3. Continual-Learning Formulation", ARCHITECTURES)
    text = replace_section(text, "# 3. Continual-Learning Formulation", "# 4. EWC Method", CONTINUAL)
    text = replace_section(text, "# 5. Experimental Setup", "# 6. Results", EXPERIMENTAL)
    text = replace_section(text, "# 6. Results", "# 7. Mechanistic Analysis", results_section())
    text = replace_section(text, "# 7. Mechanistic Analysis", "# 8. Discussion", MECHANISTIC)
    text = replace_section(text, "# 8. Discussion", "# 9. Limitations", DISCUSSION)
    text = replace_section(text, "# 9. Limitations", "# 10. Conclusions", LIMITATIONS)
    text = replace_section(text, "# 10. Conclusions", "# References", CONCLUSIONS)
    OUTPUT.write_text(text)
    print(OUTPUT)


if __name__ == "__main__":
    main()
