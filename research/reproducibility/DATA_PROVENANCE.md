# Data provenance and single-source policy

## Evidence flow

```text
ignored private run directory
  config.json / launch_config.yaml
  metrics.csv
  summary.json
  final adapter checkpoint
        │ validation and export scripts
        ▼
results/tables/<campaign>/           canonical publication-safe CSVs
        │ figure generator
        ├──► results/figures/<campaign>/   canonical figures
        │
        ├──► research/paper/TECHNICAL_REPORT.md
        │
        └──► docs/data + docs/assets       generated Pages mirrors
```

## Canonical locations

| Artifact | Authoritative location | Mirror or derivative |
|---|---|---|
| per-run raw configuration/metrics/summary | ignored `results*` campaign directory | sanitized protocol JSON under `config/` |
| adapter checkpoints | ignored run `checkpoints/` directory | none |
| public aggregate/per-seed data | `results/tables/<campaign>/` | `docs/data/` when required by Pages |
| publication figures | `results/figures/<campaign>/` | `docs/assets/` when required by Pages |
| complete report | `research/paper/TECHNICAL_REPORT.md` | `docs/data/TECHNICAL_REPORT.md` |
| related-work comparison | `comparisons/` | summarized in report/site |
| experiment protocol/report | `experiments/<campaign>/` | summarized in report/site |

## Metric definitions

- `best_recorded_validation_loss`: minimum fresh validation loss within one
  run; seed minima are aggregated only after the within-run minimum.
- `final_validation_loss`: fresh evaluation at the stated final optimizer
  step.
- `final_test_loss`: held-out evaluation of the same final checkpoint.
- perplexity aggregates: arithmetic mean of `exp(loss_seed)`, not
  `exp(mean(loss))` as the primary statistic.
- uncertainty: sample standard deviation unless a file explicitly says
  otherwise.

Historical tables retain their original names and semantics. Corrections to
interpretation do not replace raw values. Campaign README files document when a
historical field is a best value rather than a final endpoint.

## Mirror synchronization

`scripts/aggregation/generate_technical_report.py` validates the long-study
public data and copies the canonical report into `docs/data/`.
Publication scripts write canonical tables/figures; the static Pages payload
contains explicit byte-identical mirrors because GitHub Pages must resolve them
without access to private results or a custom build step. Mirror hashes are
checked during final migration validation.
