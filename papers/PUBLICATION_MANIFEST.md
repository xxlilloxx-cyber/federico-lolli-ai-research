# Publication manifest — 2026-10-08

This manifest defines the selective Git archive for the two-project research
release. It supplements the pre-publication working-tree snapshot retained
locally as `PREPUBLICATION_GIT_STATUS.txt` and the safety branch
`safety/pre-stability-publication-20261008`.

## A. Publication files

- Project 01 static pages and established public data under `docs/`.
- Project 02 page, PDF, seven SVG figures, public aggregate tables, manuscript
  mirrors, validation report, and analysis manifest under `docs/`.
- Definitive Project 02 manuscript sources, bibliography, audit, result
  verification, and PDF under `analysis_full_ewc_memory/paper/`.
- Research indexes under `papers/` and the repository documentation.

## B. Reproducibility files

- Project 02 implementation under
  `experiments/factual_learning_lora_symmetric/`.
- Combined-adapter extension source under `experiments/combined_extension/`.
- External runners, master configuration, tests, analysis scripts, validated
  aggregate tables, figures, reports, and provenance.
- The exploratory lambda analysis and the canonical full EWC-memory analysis.

## C. Local experimental artifacts excluded from Git

- `results_abc_memory/`, `results_ewc_lambda_sweep/`,
  `results_ewc_confirmation/`, `results_combined_extension/`, and other raw
  run roots.
- Adapter checkpoints (`*.pt`, `*.pth`, `*.ckpt`), run logs, caches, virtual
  environments, intermediate LaTeX build/render directories, smoke outputs,
  and the large review ZIP under `tmp/duplicate-files/local-review-bundles/`.
- The local duplicate `docs/assets/publication/Pasted image (2).png`.

These files remain on the workstation. Exclusion from Git does not delete or
relocate them. Publication-safe results are represented by the selected
aggregate tables, figures, summaries, source code, configurations, and
provenance records.
