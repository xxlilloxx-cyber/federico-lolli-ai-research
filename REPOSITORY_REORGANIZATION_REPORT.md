# Repository Reorganization Report

**Date:** 2026-09-29  
**Branch:** `public-release-preparation`  
**Remote operations:** none

## Outcome

The repository now separates implementation, execution scripts, protocols,
campaign records, canonical sanitized results, scientific writing, related-work
comparison, public website exports, and recoverable archives. No training was
launched, no raw measurement was edited, and no private checkpoint was added to
Git.

## Canonical layout

| Purpose | Canonical location |
|---|---|
| Adapter/training implementation | `src/` |
| Functional scripts | `scripts/{training,evaluation,aggregation,analysis,figures,validation}/` |
| Publication-safe configurations | `config/` |
| Campaign protocols and reports | `experiments/` |
| Sanitized tables | `results/tables/` |
| Publication figures | `results/figures/` |
| Figure provenance | `results/manifests/` |
| Scientific paper and report | `research/paper/` |
| Mathematical analysis | `research/mathematical-analysis/` |
| Reproducibility and audit | `research/reproducibility/` |
| Related-work/equivalence review | `comparisons/` |
| GitHub Pages payload | `docs/` |
| Recoverable obsolete/duplicate material | `tmp/` |

`src/adapters.py` and `src/train.py` remain at their established paths because
historical runners and imports depend on them. Moving them into deeper package
directories would be a behavior-changing refactor and was intentionally outside
this migration.

## Migration statistics

The principal migration commit records 184 Git renames, 98 additions, three
deletions caused by rename detection, and ten modified tracked files. Ten
tracked artifacts are recorded in `tmp/MANIFEST.csv`: one development note,
one identical report copy, four legacy public-data files, two superseded
readiness/planning reports, and three obsolete one-off scripts. Every archived
entry retains its original path, destination, reason, SHA-256, tracking status,
and migration date.

Three obsolete scripts were archived because they described already completed
campaigns as missing or launched a cell that is now complete. They remain
recoverable. Large ignored review bundles were moved beneath the local ignored
temporary archive without adding them to Git.

## Scientific preservation

- Six AG News runs remain in the ignored private evidence tree; canonical
  sanitized tables are under `results/tables/downstream-ag-news/`.
- All 24 rank-screening runs and all 24 independent rank-confirmation runs
  remain untouched in their ignored private trees; canonical tables are under
  their corresponding `results/tables/` directories.
- The scaling ablation, mechanism analyses, convergence data, and cost benchmark
  have distinct canonical tables and campaign documentation.
- Historical raw result directories remain in their original local locations
  and are ignored. This conservative exception avoids breaking historical
  provenance paths and does not expose them through `docs/`.
- Best-recorded validation, final validation, and final-checkpoint held-out test
  remain distinct throughout controlled result tables.

## Comparisons and originality

`comparisons/` contains verified original-source references and standardized
summaries for LoRA, DoRA, MoRA, HiRA, LoRAN, PERA, QuadraNet V2, and nonlinear
bottleneck adapters. The mathematical assessment finds a related per-output
factorized quadratic form between Symmetric and QuadraNet V2, but does not claim
complete multi-output equivalence. PERA operates on polynomial expansions of
weight factors, whereas Symmetric squares projected activations. The repository
does not claim invention of quadratic low-rank adaptation.

## Reports and website

The canonical extended report is `research/paper/TECHNICAL_REPORT.md`. The
canonical paper sources are `SCIENTIFIC_MANUSCRIPT.md` and `.tex` beside it.
Website copies under `docs/data/` are generated mirrors. The static site keeps a
general multi-project homepage and presents Symmetric Quadratic Adaptation as
Project 01. No placeholder project was published.

## Validation

- Python tests: 11 passed with external pytest plugin autoload disabled.
- Public long-data validator: six endpoints and 3,618 trajectory rows agree.
- Publication validator: canonical controlled-rank values and all local
  page/assets links passed.
- Python sources compile successfully.
- The technical-report public mirror is byte-identical to its canonical source
  after synchronization.
- Current tracked public files contain no raw checkpoint extensions, raw result
  directories, compiled Python artifacts, unapproved public email, local home
  path, or private-key marker.

## Exceptions and blockers

- No manuscript PDF was compiled because the local toolchain contains no
  Pandoc, `pdflatex`, or Tectonic executable. Markdown and LaTeX sources are
  complete.
- `CITATION.cff` is not present. No author or affiliation metadata was invented
  to create one during this migration.
- The full-block/model Hessian analysis is blocked and is not presented as a
  completed result.
- The constant-scale ablation remains single-seed and is labelled accordingly.
- A historical local stash ref previously identified as containing private
  experiment metadata remains unchanged. It is not part of the `docs/` payload
  and a normal branch push does not transfer stash refs, but it remains a risk
  for any all-ref mirror or `.git` archive. Destructive remediation was not
  authorized.

## Local commits

- `9289aac` — inventory repository for conservative reorganization.
- `069e4cb` — reorganize research artifacts and provenance.
- `e0abe6f` — complete multi-project research site and manuscript.

No push, merge, deployment, Pages enablement, remote release, history rewrite,
or stash modification occurred.
