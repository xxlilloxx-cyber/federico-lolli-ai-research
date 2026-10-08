# Final Submission Audit

## Scope and provenance

This pass edited and analyzed existing results only. No training, benchmark, dataset generation, checkpoint update, or scientific run was executed. The raw roots `results_abc_memory/`, `results_ewc_lambda_sweep/`, and `results_ewc_confirmation/` were read without modification. Canonical aggregation uses 72 unique run identities and counts reused seed-42 confirmation references once.

Primary evidence was read from `analysis_full_ewc_memory/data/CANONICAL_RUNS.csv`, `analysis_full_ewc_memory/tables/confirmation_full_metrics.csv`, `analysis_full_ewc_memory/tables/confirmation_aggregate.csv`, the exploratory sweep table, and each canonical run's `memory_matrix.csv`, `summary.json`, and `config.json`. Implementation details were verified against `experiments/factual_learning_lora_symmetric/abc_memory.py`, `dataset.py`, `modeling.py`, `run_experiment.py`, and `src/adapters.py`.

## Prose and layout compaction

- Replaced fragmented scalar displays in Results with compact inline comparisons.
- Consolidated the adapter section around the structural equations and converted parameter shapes, ranks, and short derivative statements to inline mathematics.
- Kept display mathematics for adapter definitions, the EWC objective, forgetting identities, and other referenced structural expressions.
- Preserved the compact four-subsection Discussion and removed repeated architecture rankings from Mechanistic Analysis.
- Normalized method order to LoRA, Symmetric, Combined and lambda order to ascending values.
- Renamed the ambiguous table heading `Final probability` to `Final correct-answer probability`.
- Converted the six existing canonical SVG figures to vector PDF without redrawing or changing data. The new retention figure is also vector PDF.
- Final length is 24 pages. The previous polished manuscript had 26 pages; the two-page reduction follows removal of fragmented display math and repeated prose while retaining the requested scientific content.

## Absolute-loss retention analysis

The figure uses full-precision three-seed aggregates computed directly from the 27 canonical confirmation `memory_matrix.csv` files. The manuscript prints four decimals; the underlying values used for plotting are saved in `submission_data/absolute_retention_aggregate.csv`.

| Method | $\lambda$ | $L_A(\theta_A)$ | $L_A(\theta_B)$ | $L_A(\theta_C)$ | $L_B(\theta_B)$ | $L_B(\theta_C)$ |
|---|---:|---:|---:|---:|---:|---:|
| LoRA | 0 | 1.04750500 ± 0.22461033 | 5.37947456 ± 0.45751673 | 6.26267458 ± 0.22134203 | 0.45796022 ± 0.11420370 | 4.36494466 ± 0.07487973 |
| LoRA | 1 | 1.04750500 ± 0.22461033 | 5.17567528 ± 0.44349537 | 5.98428624 ± 0.25817976 | 0.48352552 ± 0.11579398 | 3.97179364 ± 0.20501235 |
| LoRA | 10 | 1.04750500 ± 0.22461033 | 3.83399842 ± 0.52919166 | 4.01462784 ± 0.67171119 | 0.83114560 ± 0.20664478 | 2.70609852 ± 0.23237033 |
| Symmetric | 0 | 0.23988876 ± 0.02218783 | 6.07855729 ± 0.29771337 | 6.59959998 ± 0.27752574 | 0.16114305 ± 0.03482437 | 5.30596522 ± 0.26869378 |
| Symmetric | 10 | 0.23988876 ± 0.02218783 | 4.93635859 ± 0.25338425 | 5.11413578 ± 0.46880608 | 0.24426886 ± 0.05787695 | 3.50561658 ± 0.40699349 |
| Symmetric | 30 | 0.23988876 ± 0.02218783 | 4.03017816 ± 0.45290766 | 3.75439666 ± 0.55810285 | 0.41345198 ± 0.12497070 | 2.41644561 ± 0.19246319 |
| Combined | 0 | 0.37359442 ± 0.12637444 | 6.09613436 ± 0.37109064 | 6.63454258 ± 0.16750469 | 0.29999051 ± 0.09622750 | 4.91510437 ± 0.23960648 |
| Combined | 10 | 0.37359442 ± 0.12637444 | 4.82787984 ± 0.47960186 | 4.87839423 ± 0.50368154 | 0.49695816 ± 0.05292828 | 3.33861753 ± 0.57163629 |
| Combined | 30 | 0.37359442 ± 0.12637444 | 3.99563124 ± 0.45224014 | 3.45238282 ± 0.07234370 | 0.77900619 ± 0.23270948 | 2.31106805 ± 0.31568476 |

For every method, lambda, and seed, the audit recomputed

- $F_{A\rightarrow B}=L_A(\theta_B)-L_A(\theta_A)$,
- $F_{A\rightarrow C}=L_A(\theta_C)-L_A(\theta_A)$, and
- $F_{B\rightarrow C}=L_B(\theta_C)-L_B(\theta_B)$.

The maximum absolute difference from the canonical reported forgetting fields is `8.881784197001252323e-16`. Per-run values and residuals are in `submission_data/absolute_retention_per_run.csv` and `submission_data/forgetting_identity_verification.csv`.

## Reproducibility clarifications

- **Seed selection:** seed 42 is identified as the exploratory selection seed; 123 and 456 are the two out-of-selection replications. Three-seed values are described as descriptive aggregates.
- **Decoding:** exact match uses greedy decoding, sampling disabled, the default single beam, a maximum continuation length equal to the target length, standard EOS handling, and exact suffix-token equality. There is no text normalization or trimming.
- **Probability:** correct-answer probability is the example mean of $\exp(-\operatorname{NLL}_i)$, where each NLL is the mean target-token loss; it is a mean geometric token probability.
- **Training mask:** prompt and padding labels are `-100`; only target tokens contribute to cross-entropy.
- **Initialization:** LoRA $A$ and Symmetric $U$ use Kaiming uniform with $a=\sqrt5$; output factors $B$ and $P$ are zero. Combined initializes its two input factors independently by the same rule and both output factors to zero. Every initial adapter output is exactly zero.
- **Optimizer:** AdamW uses learning rate $3\times10^{-4}$, betas $(0.9,0.999)$, epsilon $10^{-8}$, weight decay 0, clipping 1.0, no scheduler, microbatch 4, accumulation 1, and effective batch 4.
- **Randomness:** the training seed controls Python, NumPy, PyTorch CPU/CUDA RNGs, adapter initialization, dropout, and deterministic phase ordering. PyTorch deterministic algorithms are enabled. GPT-2 is in train mode during optimization and evaluation mode for scoring and Fisher estimation.
- **Paraphrases:** two held-out paraphrase templates per fact are evaluated. A compact confirmation summary is included in Results.
- **Dataset:** each of A, B, and C contains 12 globally disjoint synthetic identifier–code pairs. Targets contain six to eight GPT-2 tokens.

## Canonical-run breakdown

The 72 canonical runs decompose into 9 sequential-single baselines, 9 grow-unfrozen controls, 9 hard-consolidation controls, 9 EWC lambda-100 mechanistic controls, 18 remaining exploratory sweep cells, and 18 additional confirmation executions. The nine compatible seed-42 selection cells are reused by reference and are not counted twice. Grow-unfrozen and hard consolidation are labeled auxiliary mechanistic controls because they reach 18,432 active adapter parameters.

## Scientific caveats added

- LoRA and Symmetric have $\alpha/r=1$; both Combined branches have $\alpha/r=2$. The comparison is parameter-count matched but not branch-scale matched.
- The transformations $A\mapsto cA$, $B\mapsto B/c$ and $U\mapsto cU$, $P\mapsto P/c^2$ preserve the corresponding idealized adapter functions. Euclidean displacement, diagonal Fisher, and EWC therefore depend on parameter coordinates.
- Spearman rho is the primary monotonic sweep descriptor because the lambda grid is non-uniform. Pearson correlations against $\log(1+\lambda)$ were added as a sensitivity check in `submission_data/lambda_correlation_sensitivity.csv`; no significance testing was performed.
- Knee criteria now state the per-architecture min-max normalization, ideal point, normalized Euclidean distance, endpoint chord, perpendicular-distance formula, and first-in-ascending-lambda tie behavior used by the analysis implementation.
- Existing verified references were retained. No unverified citation or DOI was introduced.

## Figures and tables

Seven figures and eleven tables are present. The new figure is `submission_figures/figure_absolute_retention_nll.pdf`; it complements rather than replaces the forgetting-decomposition figure. Every final page was rendered and visually inspected. Figures, legends, formulas, tables, captions, and references are not cropped, and there are no unexplained blank pages.

## Build and validation

Build command:

```bash
cd <repository-root>/analysis_full_ewc_memory/paper
../../.venv/bin/python tools/prepare_submission_data.py
../../.venv/bin/python tools/prepare_submission_manuscript.py
../../.venv/bin/python tools/build_submission_paper.py
tectonic -X compile FINAL_PAPER_SUBMISSION.tex --outdir submission_build_final --keep-logs --keep-intermediates
cp submission_build_final/FINAL_PAPER_SUBMISSION.pdf FINAL_PAPER_SUBMISSION.pdf
../../.venv/bin/python tools/verify_submission_paper.py
```

Validation result: 72 unique canonical runs, 27 unique confirmation observations, seven figures, eleven tables, 14 references, 24 pages, and maximum forgetting-identity error `8.882e-16`. PDF metadata contain the requested title, author `Federico Lolli`, subject, and keywords. The LaTeX build has no fatal errors, unresolved citations/references, missing assets, or overfull boxes. Remaining warnings are benign underfull-box and local-font reproducibility notices.

No experimental value, seed, dataset, hyperparameter, equation, table datum, existing plotted point, citation, or scientific conclusion was changed. Added quantities are deterministic analyses of existing evaluation outputs.
