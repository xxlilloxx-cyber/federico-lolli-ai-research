# Publication readiness review

## Completed in this revision

- Audited 184 existing summary artifacts into `docs/EXPERIMENT_AUDIT.md` and `docs/data/experiment_audit.csv` without changing raw results.
- Added original LoRA-versus-Symmetric architecture figure in PNG, SVG, and PDF.
- Added a rank-summary figure based only on existing historical records, explicitly labelled as non-factorial.
- Preserved the distinction between best validation, final validation, and final held-out test metrics.
- Added public-data regression tests and retained canonical public data definitions.

## New derived material

The audit CSV and rank plot are derived from recorded summaries. No new model run, accuracy, F1, timing, memory value, perplexity, or significance result was invented.

## Still required

Downstream task evaluation, controlled rank factorial experiments, spectral/effective-rank analysis from checkpoints, interaction analysis, and inference-latency instrumentation remain unmeasured. Exact commands and constraints are in `docs/REQUIRED_ADDITIONAL_EXPERIMENTS.md`.

## Readiness

The documentation and existing-result presentation are stronger, but the project remains an independent exploratory report. It is not ready to claim task-level superiority or general quadratic-adaptation advantage. Publication must also resolve the local-stash history issue documented in `PUBLIC_RELEASE_AUDIT.md`.
