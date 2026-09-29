# Experiment Completion Matrix

| Evidence / analysis | Status | Evidence | Limitation or blocker |
|---|---|---|---|
| Historical adapter families | DONE | `results/`, `results_extended/`, audit CSV | mixed durations; retained as historical |
| AG News controlled downstream | DONE | 6 runs, 3 seeds per method; public aggregate CSVs | one downstream dataset; 500 steps |
| Controlled rank screening | DONE | 24 runs: 2 methods × 4 ranks × 3 seeds × 1,000 steps | screening only |
| Controlled rank confirmation | DONE | 24 fresh 5,000-step runs with final test checkpoints | rank changes parameters and alpha/r |
| Constant-scale ablation | PARTIAL | 8 Policy-B runs, seed 42, 1,000 steps | single seed by predeclared design; seeds 123/456 not run |
| LoRA effective-update spectra | DONE | 12 final checkpoints | applies only to linear adapter matrix |
| Symmetric U/P spectra | DONE | 12 final checkpoints | factor spectra are not a constant DeltaW |
| Symmetric output covariance | DONE | fixed 32-position held-out activation sample | one test block/sample definition |
| Local Jacobian spectra | DONE | 768 rows: 24 checkpoints × 32 positions | local, adapter-only analysis |
| Adapter-only second-order interactions | DONE | one Hessian per checkpoint under fixed residual-direction scalar | scalar reduction is one defensible choice |
| Full-block/model Hessian interaction | BLOCKED | not executed | full 768-dimensional Hessian and scalar definition require a separately frozen protocol and substantial compute |
| Convergence reconstruction | DONE | fresh evaluations every 25 steps; selected checkpoints exported | validation uses existing eight-block protocol |
| Controlled cost benchmark | DONE | 30 timed iterations after 10 warmups, rank 4 seed 42 | one device and batch/sequence configuration |
| Technical/public report integration | DONE | final review and report appendices | exploratory release only |
| Publication history safety | BLOCKED | `PUBLIC_RELEASE_AUDIT.md` | reachable local stash history remains; no destructive rewrite authorized |
