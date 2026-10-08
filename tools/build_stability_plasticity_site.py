#!/usr/bin/env python3
"""Build the additive Stability–Plasticity Control project pages and assets."""
from __future__ import annotations

import csv
import html
import shutil
from pathlib import Path

from build_search_index import build as build_search_index

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PROJECT = DOCS / "projects" / "02-stability-plasticity"
ASSETS = DOCS / "assets" / "stability-plasticity"
DATA = DOCS / "data" / "stability-plasticity"
PAPER = ROOT / "analysis_full_ewc_memory" / "paper"
ANALYSIS = ROOT / "analysis_full_ewc_memory"
REPO = "https://github.com/xxlilloxx-cyber/federico-lolli-ai-research"
TITLE = "Stability–Plasticity Control in Parameter-Efficient Continual Factual Learning: EWC across Linear, Quadratic, and Combined Low-Rank Adapters"
PDF_NAME = "Federico_Lolli_Stability_Plasticity_Control.pdf"
SOURCE_PDF = PAPER / "Federico Lolli Stability–Plasticity Control in Parameter-Efficient Continual Factual Learning.pdf"

ABSTRACT = (
    "Sequential parameter-efficient adaptation must balance acquisition of new information against retention of earlier updates. "
    "We study this problem in a no-replay sequence of three disjoint synthetic factual sets using a frozen GPT-2 Small backbone "
    "and three adapters with 6,144 trainable parameters: LoRA, a symmetric factorized quadratic adapter, and a Combined "
    "linear–quadratic adapter. Consolidation uses online Elastic Weight Consolidation (EWC) with an empirical diagonal Fisher "
    "over adapter parameters. A seven-point sweep on seed 42 selects architecture-specific coefficient ranges; the selected "
    "settings are then evaluated on two additional seeds, with descriptive aggregates over all three. Stronger EWC consistently "
    "reduces forgetting while increasing new-fact acquisition NLL. Intermediate coefficients improve final balanced memory "
    "relative to the execution-matched λ=0 control in every tested candidate configuration and each paired seed. The transition "
    "scale depends on adapter parameterization: LoRA responds at lower coefficients, Symmetric remains more plastic under "
    "moderate regularization, and Combined does not systematically dominate either component architecture. Increasing EWC "
    "strength also reduces parameter displacement, which is positively associated with forgetting. Under the evaluated setting, "
    "these findings support treating consolidation strength and adapter geometry as coupled design choices. Scope is limited to "
    "GPT-2 Small, one placement, three 12-fact sets, and two out-of-selection replication seeds."
)


def global_nav(prefix: str, current: str) -> str:
    entries = [
        ("Home", f"{prefix}index.html", "home"),
        ("Research", f"{prefix}research/index.html", "research"),
        ("Publications", f"{prefix}publications.html", "publications"),
        ("GitHub", REPO, "repository"),
        ("About", f"{prefix}about.html", "about"),
    ]
    links = []
    for label, href, key in entries:
        current_attribute = ' aria-current="page"' if key == current else ""
        links.append(f'<a{current_attribute} href="{href}">{label}</a>')
    return "".join(links)


def page(title: str, body: str, *, prefix: str, current: str, description: str,
         mathjax: bool = False) -> str:
    math = ""
    if mathjax:
        math = '''<script>window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"]],displayMath:[["\\\\[","\\\\]"]]},options:{skipHtmlTags:["script","noscript","style","textarea","pre","code"]}};</script><script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'''
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="{html.escape(description, quote=True)}"><title>{html.escape(title)}</title><link rel="stylesheet" href="{prefix}assets/site.css">{math}</head><body>
<header class="site-header"><nav class="nav" aria-label="Primary navigation"><a class="brand" href="{prefix}index.html">Federico Lolli<small>AI Research</small></a><div class="nav-links">{global_nav(prefix, current)}</div></nav></header>
{body}
<footer class="site-footer"><p>© Federico Lolli · <a href="mailto:xxlilloxx@gmail.com">xxlilloxx@gmail.com</a></p><p>Original code: <a href="{REPO}/blob/main/LICENSE">MIT License</a>. Original site text and figures: <a href="{REPO}/blob/main/LICENSE-CONTENT.md">CC BY 4.0</a>. Third-party materials retain their own terms.</p></footer></body></html>'''


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def confirmation_table() -> str:
    rows = load_rows(ANALYSIS / "tables" / "confirmation_aggregate.csv")
    order = {"lora": 0, "symmetric": 1, "combined": 2}
    labels = {"lora": "LoRA", "symmetric": "Symmetric", "combined": "Combined"}
    rows.sort(key=lambda row: (order[row["method"]], float(row["lambda"])))
    body = []
    for row in rows:
        def value(metric: str) -> str:
            return f'{float(row[metric + "_mean"]):.4f} ± {float(row[metric + "_sample_sd"]):.4f}'
        body.append(
            f'<tr><td>{labels[row["method"]]}</td><td class="number">{float(row["lambda"]):g}</td>'
            f'<td class="number">{value("mean_forgetting")}</td>'
            f'<td class="number">{value("mean_new_fact_acquisition_nll")}</td>'
            f'<td class="number">{value("mean_final_memory_nll")}</td>'
            f'<td class="number">{value("mean_final_probability")}</td>'
            f'<td class="number">{value("mean_final_exact_match")}</td></tr>'
        )
    return '''<div class="table-wrap"><table><caption>Three-seed descriptive aggregates. Seed 42 selected candidate coefficients; seeds 123 and 456 are out-of-selection replications.</caption><thead><tr><th>Method</th><th>λ</th><th>Mean forgetting</th><th>New acquisition NLL</th><th>Final balanced NLL</th><th>Correct-answer probability</th><th>Exact match</th></tr></thead><tbody>''' + "".join(body) + "</tbody></table></div>"


def figure(number: int, filename: str, alt: str, caption: str) -> str:
    return f'''<figure class="figure"><img src="../../assets/stability-plasticity/{filename}" alt="{html.escape(alt, quote=True)}"><figcaption><strong>Figure {number}.</strong> {caption}</figcaption></figure>'''


def project_page() -> str:
    project_nav = '''<nav class="project-nav" aria-label="Stability–Plasticity Control navigation"><div class="inner"><a href="#overview">Overview</a><a href="#formulation">Mathematical formulation</a><a href="#method">Method</a><a href="#protocol">Experimental protocol</a><a href="#results">Results</a><a href="#discussion">Discussion</a><a href="#references">References</a><a href="../../data/Federico_Lolli_Stability_Plasticity_Control.pdf">Full paper PDF</a></div></nav>'''
    body = rf'''{project_nav}<main class="paper-page">
<section id="overview" class="hero compact"><p class="eyebrow">Research Project 02 · Research manuscript · 2026</p><h1>{TITLE}</h1><p class="paper-byline">Federico Lolli</p><p class="lede">How does regularization strength control stability and plasticity when the same continual factual-learning protocol is implemented through different parameter-efficient adapter geometries?</p><div class="button-row"><a class="button" href="../../data/{PDF_NAME}">Download full paper (PDF)</a><a class="button secondary" href="{REPO}/tree/main/experiments/factual_learning_lora_symmetric">Source code</a></div></section>
<section><h2>Abstract</h2><p>{ABSTRACT}</p><p class="keywords">Continual learning · EWC · PEFT · LoRA · quadratic adapters · factual learning · GPT-2</p></section>
<section><h2>Introduction</h2><p>Sequential parameter-efficient adaptation must incorporate new factual associations without erasing earlier updates. The study isolates this problem in a no-replay sequence of three disjoint synthetic fact sets. It compares three adapters with the same trainable-parameter count but different parameter-to-function maps: linear LoRA, a symmetric factorized quadratic adapter, and their Combined form.</p><p>A full seven-point λ sweep on seed 42 identifies candidate operating regions. The selected coefficients are then evaluated on two unseen seeds, 123 and 456. The three-seed summaries therefore combine one selection seed with two out-of-selection replications and are descriptive rather than inferential.</p></section>
<section id="formulation"><h2>Mathematical formulation</h2><h3>Adapter architectures</h3><p>For row-vector activation \(x\), frozen projection \(W\), and bias \(b\), the adapted projection is \(y=xW+b+\Delta y(x)\). The backbone remains frozen.</p><div class="equation">\[\Delta y_L=\frac{{\alpha}}{{r}}(xA)B.\]</div><p>LoRA is input-linear, with \(J_L=(\alpha/r)AB\) and adapter-only Hessian \(H_L=0\).</p><div class="equation">\[\Delta y_S=\frac{{\alpha}}{{r}}\big[(xU)\odot(xU)\big]P.\]</div><p>For output coordinate \(j\), \(\Delta y_{{S,j}}=xQ_jx^T\), where</p><div class="equation">\[Q_j=\frac{{\alpha}}{{r}}\sum_{{k=1}}^r P_{{kj}}u_ku_k^T,\qquad J_S(x)=2\frac{{\alpha}}{{r}}U\operatorname{{diag}}(xU)P,\qquad H_{{S,j}}=2Q_j.\]</div><p>The Combined adapter sums independent branches:</p><div class="equation">\[\Delta y_C=\frac{{\alpha_L}}{{r_L}}(xA)B+\frac{{\alpha_S}}{{r_S}}\big[(xU)\odot(xU)\big]P.\]</div>
<h3>Continual factual-learning protocol</h3><p>Training follows \(A\rightarrow B\rightarrow C\) without replay. Absolute losses are evaluated at each phase boundary. Forgetting is defined by</p><div class="equation">\[F_{{A\rightarrow B}}=L_A(\theta_B)-L_A(\theta_A),\quad F_{{A\rightarrow C}}=L_A(\theta_C)-L_A(\theta_A),\quad F_{{B\rightarrow C}}=L_B(\theta_C)-L_B(\theta_B).\]</div><div class="equation">\[F_{{\mathrm{{mean}}}}=\frac{{F_{{A\rightarrow B}}+F_{{A\rightarrow C}}+F_{{B\rightarrow C}}}}{{3}},\qquad L_{{\mathrm{{new}}}}^{{\mathrm{{acq}}}}=\frac{{L_B(\theta_B)+L_C(\theta_C)}}{{2}}.\]</div><p>Lower NLL and lower forgetting are better. Zero forgetting indicates no degradation from the post-acquisition reference; a negative value would indicate improvement.</p>
<h3>Online empirical diagonal-Fisher EWC</h3><div class="equation">\[\mathcal L_{{\mathrm{{total}}}}=\mathcal L_{{\mathrm{{current}}}}+\frac{{\lambda}}{{2}}\sum_i F_i(\theta_i-\theta_i^*)^2.\]</div><p>After phase A, the method stores \(\theta_A^*\) and estimates \(F_A\). After B, it updates the online Fisher as \(F_{{AB}}=\gamma F_A+F_B\) with \(\gamma=1\), and uses the post-B reference during C. Fisher and reference states are computed even at λ=0; the regularization term and its gradient are then exactly zero.</p></section>
<section id="method"><h2>Stability–plasticity control mechanism</h2><p>The controlled variable is EWC strength λ applied to one shared adapter throughout the three phases. Increasing λ raises resistance to movement in coordinates assigned high importance by the empirical diagonal Fisher. The experiment tests whether the same nominal coefficient induces the same functional trade-off in different adapter parameterizations.</p><div class="table-wrap"><table><caption>Matched trainable-parameter configurations.</caption><thead><tr><th>Method</th><th>Rank</th><th>Scale</th><th>Trainable parameters</th></tr></thead><tbody><tr><td>LoRA</td><td>r=4</td><td>α/r=1</td><td class="number">6,144</td></tr><tr><td>Symmetric</td><td>r=4</td><td>α/r=1</td><td class="number">6,144</td></tr><tr><td>Combined</td><td>r<sub>L</sub>=2, r<sub>S</sub>=2</td><td>α<sub>L</sub>/r<sub>L</sub>=α<sub>S</sub>/r<sub>S</sub>=2</td><td class="number">6,144</td></tr></tbody></table></div><p>Parameter count is matched, but branch-wise scaling and functional geometry are not. Combined therefore couples geometry, rank allocation, and output scale.</p></section>
<section id="protocol"><h2>Experimental methodology</h2><dl class="definition"><dt>Backbone</dt><dd>GPT-2 Small, fully frozen.</dd><dt>Placement</dt><dd>Zero-based block 0, <code>attn.c_proj</code>.</dd><dt>Data</dt><dd>Three deterministic, globally disjoint sets A/B/C; 12 synthetic identifier–code facts per set; no replay.</dd><dt>Schedule</dt><dd>2,500 optimizer steps per phase; 7,500 steps per run.</dd><dt>Optimizer</dt><dd>AdamW, learning rate 3×10<sup>−4</sup>, β=(0.9, 0.999), ε=10<sup>−8</sup>, zero weight decay, gradient clipping 1.0, no scheduler.</dd><dt>Execution</dt><dd>FP16, sequence length 64, microbatch 4, gradient accumulation 1, effective batch 4.</dd><dt>Fisher</dt><dd>Adapter-only empirical diagonal estimate from 12 deterministic examples after A and B; γ=1.</dd><dt>Sweep</dt><dd>λ∈{{0,1,10,30,50,100,300}} on selection seed 42.</dd><dt>Replication</dt><dd>LoRA λ∈{{0,1,10}}; Symmetric and Combined λ∈{{0,10,30}} on seeds 42, 123, 456.</dd><dt>Metrics</dt><dd>Target NLL, geometric mean correct-answer token probability, exact token match, paraphrase metrics, forgetting, balanced final NLL, parameter displacement, Fisher structure, and realized EWC cost.</dd></dl><p>Prompt and padding labels are masked with −100, so only target tokens contribute to training loss. Exact generation is greedy, with no text normalization; the generated suffix must exactly match the target token IDs.</p></section>
<section id="results"><h2>Results</h2><h3>Exploratory experiments</h3><p>Across all three architectures, increasing λ reduces forgetting and raises new-fact acquisition NLL. LoRA moves toward stability at lower coefficients; Symmetric retains stronger plasticity over the moderate range; Combined occupies an intermediate region without systematic dominance.</p>{figure(1, "figure_1_exploratory_stability_plasticity.svg", "Three stability-plasticity trajectories for LoRA, Symmetric, and Combined adapters.", "Exploratory single-seed stability–plasticity trajectories across seven EWC strengths. Lower-left is better on both axes.")}
<h3>Multiseed confirmation</h3>{confirmation_table()}{figure(2, "figure_2_confirmation_stability_plasticity.svg", "Multi-seed stability-plasticity points with error bars.", "Candidate evaluation over seeds 42, 123, and 456. Points show descriptive means and bars show sample standard deviations; seed 42 was used for selection.")}
<p>Every non-zero candidate reduces mean forgetting, increases acquisition NLL, and improves final balanced NLL relative to its execution-matched λ=0 control in all three paired seeds. The direction is also reproduced on both out-of-selection seeds. These are descriptive three-seed results, not a claim of population-level significance.</p>
<h3>Quantitative results and balanced memory</h3>{figure(3, "figure_absolute_retention_nll.svg", "Absolute NLL on factual set A after each training phase.", "Absolute NLL trajectory for factual set A after phases A, B, and C. Curves show three-seed means and sample standard deviations. Lower NLL is better; forgetting values are differences between these absolute losses.")}{figure(4, "figure_3_final_balanced_memory.svg", "Final balanced memory NLL by method and lambda.", "Final balanced memory after phase C. Bars show mean NLL across sets A, B, and C; black points are individual seeds.")}{figure(5, "figure_4_forgetting_decomposition.svg", "Forgetting components A after B, A after C, and B after C.", "Forgetting decomposed into A after B, A after C, and B after C. Points show three-seed means with sample standard deviations.")}
<h3>Forgetting and exact-generation analysis</h3><p>Intermediate coefficients give the lowest sampled balanced final NLL, while exact generation declines more sharply. At the strongest replicated candidates, final balanced NLL is 2.6169±0.2279 for LoRA λ=10, 2.3888±0.1854 for Symmetric λ=30, and 2.4373±0.0694 for Combined λ=30. Their corresponding exact-match means are 0.0046, 0.0185, and 0.0046.</p>{figure(6, "figure_5_plasticity_exact_generation.svg", "Acquisition NLL and exact generation by method and lambda.", "New-fact acquisition NLL and final exact-token-match generation for the repeated candidate configurations.")}
<h3>Mechanistic analysis</h3><p>Stronger EWC generally accompanies smaller cumulative adapter displacement. Across the seven-point sweep, displacement and forgetting are positively associated within every method, but the analysis is coordinate dependent and does not establish causality.</p>{figure(7, "figure_6_mechanistic_evidence.svg", "Parameter displacement, realized EWC penalty, and Fisher support.", "Exploratory relationships among EWC strength, cumulative adapter displacement, realized phase-C penalty, and online Fisher effective support.")}</section>
<section id="discussion"><h2>Discussion</h2><p>The transition scale differs across architectures, so λ cannot be treated as an architecture-independent consolidation measure. EWC constrains parameter coordinates, whereas behavior depends on the parameter-to-function map. Reparameterizations can preserve adapter functions while changing Euclidean displacement and diagonal Fisher values.</p><p>NLL and exact generation answer different questions. Moderate consolidation can improve likelihood across all three fact sets while weakening exact acquisition of the newest targets. Hard-consolidation controls further show that bit-identical old adapter tensors do not guarantee functional retention when later components remain active.</p></section><section id="limitations"><h2>Limitations</h2><div class="note"><strong>Scope.</strong> The evidence is limited to GPT-2 Small, one placement, three 12-fact synthetic sets, a three-phase no-replay sequence, a diagonal empirical Fisher estimated from 12 examples per consolidation point, γ=1, a sparse λ grid, and two out-of-selection replication seeds. Grow-unfrozen and hard-consolidation controls reach 18,432 active parameters and are mechanistic context rather than parameter-matched competitors.</div></section>
<section><h2>Conclusions</h2><p>Under the evaluated protocol, online EWC provides a controllable stability–plasticity trade-off. Its effective scale depends on adapter parameterization, and intermediate coefficients produce the strongest sampled balanced-memory regimes. Combined does not systematically dominate the single-family adapters. The results support joint selection of adapter geometry and consolidation strength, while larger models, realistic fact streams, more placements, and direct controls over branch scaling remain necessary for broader generalization.</p></section>
<section id="reproducibility"><h2>Download and reproducibility</h2><div class="grid two"><article class="card"><h3>Paper</h3><p><a href="../../data/{PDF_NAME}">Full paper PDF</a></p><p><a href="../../data/stability-plasticity/paper.md">Manuscript source (Markdown)</a> · <a href="../../data/stability-plasticity/paper.tex">LaTeX</a></p></article><article class="card"><h3>Code and data</h3><p><a href="{REPO}/tree/main/experiments/factual_learning_lora_symmetric">Experiment implementation</a></p><p><a href="../../data/stability-plasticity/confirmation_aggregate.csv">Confirmation aggregates</a> · <a href="../../data/stability-plasticity/exploratory_lambda_sweep.csv">Exploratory sweep</a> · <a href="../../data/stability-plasticity/absolute_retention_aggregate.csv">Retention losses</a></p></article></div><p><a href="../../data/stability-plasticity/ANALYSIS_MANIFEST.json">Analysis manifest</a> · <a href="../../data/stability-plasticity/DATA_VALIDATION_REPORT.md">Validation report</a> · <a href="../../data/stability-plasticity/FINAL_SUBMISSION_AUDIT.md">Final paper audit</a></p></section>
<section id="references"><h2>References</h2><ol class="references"><li>Kirkpatrick et al. (2017). “Overcoming catastrophic forgetting in neural networks.” <em>PNAS</em> 114(13):3521–3526. doi:10.1073/pnas.1611835114.</li><li>Hu et al. (2022). “LoRA: Low-Rank Adaptation of Large Language Models.” <em>ICLR</em>. arXiv:2106.09685.</li><li>Radford et al. (2019). “Language Models are Unsupervised Multitask Learners.” OpenAI Technical Report.</li><li>Houlsby et al. (2019). “Parameter-Efficient Transfer Learning for NLP.” <em>ICML</em>, PMLR 97:2790–2799.</li><li>Li and Liang (2021). “Prefix-Tuning: Optimizing Continuous Prompts for Generation.” <em>ACL-IJCNLP</em>, 4582–4597. doi:10.18653/v1/2021.acl-long.353.</li><li>Lester, Al-Rfou, and Constant (2021). “The Power of Scale for Parameter-Efficient Prompt Tuning.” <em>EMNLP</em>, 3045–3059. doi:10.18653/v1/2021.emnlp-main.243.</li><li>Ke and Liu (2022). “Continual Learning of Natural Language Processing Tasks: A Survey.” arXiv:2211.12701.</li><li>Biesialska, Biesialska, and Costa-jussà (2020). “Continual Lifelong Learning in Natural Language Processing: A Survey.” <em>COLING</em>, 6523–6541. doi:10.18653/v1/2020.coling-main.574.</li><li>Meng et al. (2022). “Locating and Editing Factual Associations in GPT.” <em>NeurIPS</em> 35. arXiv:2202.05262.</li><li>Meng et al. (2023). “Mass-Editing Memory in a Transformer.” <em>ICLR</em>. arXiv:2210.07229.</li><li>Mitchell et al. (2022). “Fast Model Editing at Scale.” <em>ICLR</em>. arXiv:2110.11309.</li><li>Mitchell et al. (2022). “Memory-Based Model Editing at Scale.” <em>ICML</em>, PMLR 162:15817–15831. arXiv:2206.06520.</li><li>Wang et al. (2024). “Knowledge Editing for Large Language Models: A Survey.” <em>ACM Computing Surveys</em> 57(3), Article 59. doi:10.1145/3698590.</li><li>Zhang et al. (2024). “A Comprehensive Study of Knowledge Editing for Large Language Models.” arXiv:2401.01286.</li></ol></section></main>'''
    return page(TITLE, body, prefix="../../", current="research", description="Federico Lolli's study of architecture-dependent EWC stability–plasticity control in parameter-efficient continual factual learning.", mathjax=True)


def home_page() -> str:
    body = f'''<main><section class="hero"><p class="eyebrow">Independent research archive</p><h1>Federico Lolli’s AI research</h1><p class="lede">A growing collection of reproducible exploratory studies on how machine-learning systems can be adapted, measured, and interpreted. Each project keeps its question, protocol, source data, and limitations visible.</p><p>This site is designed as a research archive rather than a stream of claims. Results are reported with their experimental scope, including negative and mixed findings.</p></section>
<section><h2>Latest research</h2><article class="project-card latest"><img src="assets/stability-plasticity/figure_2_confirmation_stability_plasticity.svg" alt="Stability-plasticity confirmation curves for LoRA, Symmetric, and Combined adapters."><div><p class="project-number">RESEARCH PROJECT 02 · 2026 MANUSCRIPT</p><h3>Stability–Plasticity Control</h3><p><strong>{TITLE}</strong></p><p>Online empirical diagonal-Fisher EWC is evaluated across linear, quadratic, and Combined low-rank adapters in a no-replay A→B→C factual-learning sequence. The study separates a seed-42 coefficient sweep from two out-of-selection replication seeds.</p><p><strong>Federico Lolli</strong></p><p><a href="projects/02-stability-plasticity/index.html">Read the research page →</a> · <a href="data/{PDF_NAME}">Download the full paper</a></p></div></article></section>
<section><h2>Published research projects</h2><div class="project-catalog"><article class="project-card"><img src="assets/publication/adapter_architecture_comparison.svg" alt="Signal-path comparison of LoRA and the Symmetric Quadratic Adapter."><div><p class="project-number">RESEARCH PROJECT 01</p><h3>Symmetric Quadratic Adaptation</h3><p>A controlled investigation of nonlinear low-rank adaptation for frozen GPT-2. The study covers language modelling, AG News classification, rank and scaling controls, convergence, derivative spectra, and computational cost.</p><p class="keywords">Low-rank adaptation · quadratic interactions · GPT-2 · LoRA · parameter efficiency</p><p><a href="projects/01-quadratic-gpt2/index.html">Open Project 01 →</a> · <a href="data/Federico_Lolli_Symmetric_Quadratic_Adaptation_GPT2.pdf">Read the paper (PDF)</a></p></div></article></div></section>
<section><h2>How to read this archive</h2><div class="grid"><article class="card"><h3>Question first</h3><p>Each study defines a narrow empirical question before presenting results.</p></article><article class="card"><h3>Evidence linked</h3><p>Important reported values link to source tables, code, configurations, or a full technical report.</p></article><article class="card"><h3>Limits stated</h3><p>Findings are constrained to the models, data, budgets, and training protocols actually tested.</p></article></div></section></main>'''
    return page("Federico Lolli — AI Research", body, prefix="", current="home", description="Federico Lolli's AI research archive, including Stability–Plasticity Control and Symmetric Quadratic Adaptation.")


def research_page() -> str:
    body = '''<main><section class="hero compact"><p class="eyebrow">Research archive</p><h1>Research</h1><p class="lede">Two distinct research lines, each with its own protocol, evidence, and limitations.</p></section><section><div class="research-tree"><article class="card"><p class="project-number">RESEARCH PROJECT 01</p><h2>LoRA and Symmetric Adapters</h2><p>Controlled comparisons of Linear LoRA and the Symmetric Quadratic Adapter, including the historical Combined formulation, experimental results, mechanism analysis, and technical reports.</p><ul><li><a href="../projects/01-quadratic-gpt2/method.html">Linear LoRA and Symmetric formulation</a></li><li><a href="../projects/01-quadratic-gpt2/experiments.html">Combined and experimental protocols</a></li><li><a href="../projects/01-quadratic-gpt2/results.html">Experimental results</a></li><li><a href="../projects/01-quadratic-gpt2/reproducibility.html">Technical reports and reproducibility</a></li></ul><p><a href="../projects/01-quadratic-gpt2/index.html">Open Project 01 →</a></p></article><article class="card"><p class="project-number">RESEARCH PROJECT 02</p><h2>Stability–Plasticity Control</h2><p>Online empirical diagonal-Fisher EWC across LoRA, Symmetric, and Combined adapters in continual factual learning.</p><ul><li><a href="../projects/02-stability-plasticity/index.html#overview">Paper overview</a></li><li><a href="../projects/02-stability-plasticity/index.html#formulation">Mathematical formulation</a></li><li><a href="../projects/02-stability-plasticity/index.html#results">Experimental results</a></li><li><a href="../data/Federico_Lolli_Stability_Plasticity_Control.pdf">Full paper PDF</a></li></ul><p><a href="../projects/02-stability-plasticity/index.html">Open Project 02 →</a></p></article></div></section></main>'''
    return page("Research — Federico Lolli", body, prefix="../", current="research", description="Research projects by Federico Lolli: low-rank quadratic adaptation and EWC stability-plasticity control.")


def publications_page() -> str:
    body = f'''<main><section class="hero compact"><p class="eyebrow">Scientific archive</p><h1>Publications and manuscripts</h1><p class="lede">Research papers are listed separately from technical reports and experimental documentation. No venue publication is implied unless explicitly stated.</p></section><section><article class="card publication-entry"><p class="project-number">RESEARCH MANUSCRIPT · 2026</p><h2>{TITLE}</h2><p><strong>Federico Lolli</strong></p><p>This manuscript studies how online empirical diagonal-Fisher EWC strength controls retention and acquisition across three parameter-efficient adapter geometries in a no-replay continual factual-learning protocol.</p><p><a href="projects/02-stability-plasticity/index.html">Paper page →</a> · <a href="data/{PDF_NAME}">PDF</a></p></article><article class="card publication-entry"><p class="project-number">RESEARCH PAPER · PROJECT 01</p><h2>Symmetric Factorized Quadratic Adaptation: A Controlled Study of Nonlinear Low-Rank Adapters for GPT-2</h2><p><strong>Federico Lolli</strong></p><p>A controlled study of linear and quadratic low-rank adapters across language modelling, classification, scaling, convergence, mechanism, and computational cost.</p><p><a href="projects/01-quadratic-gpt2/index.html">Paper page →</a> · <a href="data/Federico_Lolli_Symmetric_Quadratic_Adaptation_GPT2.pdf">PDF</a></p></article></section><section><h2>Technical reports</h2><p>Project-specific methodology, result tables, and reproducibility reports remain linked from each research page and are not presented as separate publications.</p></section></main>'''
    return page("Publications — Federico Lolli", body, prefix="", current="publications", description="Research papers and manuscripts by Federico Lolli.")


def copy_public_assets() -> None:
    PROJECT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    if not SOURCE_PDF.is_file():
        raise FileNotFoundError(f"Definitive paper not found: {SOURCE_PDF}")
    shutil.copy2(SOURCE_PDF, DOCS / "data" / PDF_NAME)
    figure_sources = {
        "figure_1_exploratory_stability_plasticity.svg": ANALYSIS / "figures_main" / "figure_1_exploratory_stability_plasticity.svg",
        "figure_2_confirmation_stability_plasticity.svg": ANALYSIS / "figures_main" / "figure_2_confirmation_stability_plasticity.svg",
        "figure_3_final_balanced_memory.svg": ANALYSIS / "figures_main" / "figure_3_final_balanced_memory.svg",
        "figure_4_forgetting_decomposition.svg": ANALYSIS / "figures_main" / "figure_4_forgetting_decomposition.svg",
        "figure_5_plasticity_exact_generation.svg": ANALYSIS / "figures_main" / "figure_5_plasticity_exact_generation.svg",
        "figure_6_mechanistic_evidence.svg": ANALYSIS / "figures_main" / "figure_6_mechanistic_evidence.svg",
        "figure_absolute_retention_nll.svg": PAPER / "submission_figures" / "figure_absolute_retention_nll.svg",
    }
    for name, source in figure_sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, ASSETS / name)
    public_data = {
        "paper.md": PAPER / "FINAL_PAPER_SUBMISSION.md",
        "paper.tex": PAPER / "FINAL_PAPER_SUBMISSION.tex",
        "FINAL_SUBMISSION_AUDIT.md": PAPER / "FINAL_SUBMISSION_AUDIT.md",
        "confirmation_aggregate.csv": ANALYSIS / "tables" / "confirmation_aggregate.csv",
        "exploratory_lambda_sweep.csv": ANALYSIS / "tables" / "exploratory_lambda_sweep.csv",
        "absolute_retention_aggregate.csv": PAPER / "submission_data" / "absolute_retention_aggregate.csv",
        "canonical_run_breakdown.csv": PAPER / "submission_data" / "canonical_run_breakdown.csv",
        "lambda_correlation_sensitivity.csv": PAPER / "submission_data" / "lambda_correlation_sensitivity.csv",
        "ANALYSIS_MANIFEST.json": ANALYSIS / "provenance" / "ANALYSIS_MANIFEST.json",
        "DATA_VALIDATION_REPORT.md": ANALYSIS / "reports" / "DATA_VALIDATION_REPORT.md",
    }
    for name, source in public_data.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, DATA / name)
    # Keep the canonical audit unchanged while removing workstation-specific
    # paths from its public mirror.
    audit = DATA / "FINAL_SUBMISSION_AUDIT.md"
    audit.write_text(
        audit.read_text(encoding="utf-8")
        .replace(str(ROOT), "<repository-root>")
        .replace(str(Path.home() / ".local" / "bin" / "tectonic"), "tectonic"),
        encoding="utf-8",
    )


def build_stability_plasticity_site() -> None:
    copy_public_assets()
    (PROJECT / "index.html").write_text(project_page() + "\n", encoding="utf-8")
    (DOCS / "index.html").write_text(home_page() + "\n", encoding="utf-8")
    (DOCS / "research" / "index.html").write_text(research_page() + "\n", encoding="utf-8")
    (DOCS / "publications.html").write_text(publications_page() + "\n", encoding="utf-8")
    build_search_index(__import__("datetime").date.today().isoformat())
    print("built Stability–Plasticity Control page, archive navigation, and public assets")


if __name__ == "__main__":
    build_stability_plasticity_site()
