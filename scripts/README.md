# Script organization

- `training/`: experiment launchers and resume-safe runners.
- `evaluation/`: checkpoint and held-out evaluation.
- `aggregation/`: public-data export, aggregation, and report synchronization.
- `analysis/`: convergence, mechanism, cost, and synthetic analyses.
- `figures/`: plot and publication-asset generation.
- `validation/`: environment checks, smoke tests, and evidence validation.

Run scripts from the repository root. Raw output directories are ignored. The
file names and scientific algorithms are unchanged; this migration only groups
them by function and updates repository-root resolution.
