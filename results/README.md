# Validated public results

`results/` is the canonical tracked evidence layer:

- `tables/` contains publication-safe aggregate and per-seed CSVs;
- `figures/` contains canonical publication figures grouped by campaign;
- `manifests/` records figure provenance.

Other children of this directory are historical raw runs and remain ignored.
The static site mirrors selected canonical tables and figures under `docs/`.
Those mirrors are generated artifacts and must not be edited independently.
