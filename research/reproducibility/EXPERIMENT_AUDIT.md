# Experiment audit

This audit was generated from existing `summary.json` files without rerunning
training. The complete publication-safe per-run table is
[`experiment_audit.csv`](experiment_audit.csv). `NOT AVAILABLE` means the
original summary did not record that field; no value was inferred. Controlled
final held-out values are under `results/tables/`.

| Experiment family | Runs | Adapters observed | Step values observed |
|---|---:|---|---|
| results | 25 | baseline, feature_interaction, lora, quadratic, signed_quadratic | 20, 50, 500 |
| results_depth | 64 | lora, symmetric_quadratic | 100, 20 |
| results_extended | 28 | linear_quadratic, lora, symmetric_quadratic | 1000, 20, 2000, 500 |
| results_local_rank | 42 | lora, symmetric_quadratic | 1000, 2000 |
| results_long_convergence_5000 | 18 | lora, symmetric_quadratic | 5000 |
| results_multilayer | 6 | lora, symmetric_quadratic | 1000 |

The source-result-file field is a local traceability reference; raw artifacts are deliberately not part of the public Pages payload.
