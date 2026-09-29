# Mechanism analysis

Existing final checkpoints from the 5,000-step rank confirmation were analyzed
without retraining. The analysis covers LoRA effective-update spectra,
Symmetric U/P spectra, empirical output covariance, local input-dependent
Jacobians, and adapter-only Hessians under a fixed scalar reduction.

- **Ranks/seeds:** ranks 1, 2, 4, and 8; seeds 42, 123, and 456.
- **Fixed sample:** the first public WikiText-2 test block, token positions
  0–31; definition in [heldout_sample.json](heldout_sample.json).
- **Private input:** final adapter checkpoints in the ignored confirmation tree.
- **Canonical tables:** `results/tables/mechanism-analysis/`.
- **Canonical figures:** `results/figures/mechanism-analysis/`.
- **Methods:** `research/mathematical-analysis/MECHANISM_ANALYSIS_REPORT.md`
  and `SYMMETRIC_DERIVATIVE_ANALYSIS.md`.
- **Runner:** `scripts/analysis/analyze_adapter_mechanisms.py`.

The nonlinear Symmetric adapter is never represented as a constant `DeltaW`.
Full-block/model Hessian analysis was not performed and remains explicitly
blocked by the absence of a predeclared scalar-response protocol.
