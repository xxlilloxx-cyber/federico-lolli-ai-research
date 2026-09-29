# Repository structure

## Implementation

- `src/adapters.py`: all implemented adapter families, insertion utilities,
  parameter counting, and adapter-only checkpoint save/load.
- `src/train.py`: historical and controlled WikiText-2 training entry point.
- `scripts/training/`: campaign launchers. Run from the repository root.
- `scripts/evaluation/`: final-checkpoint and held-out evaluators.
- `scripts/analysis/`: mechanism, convergence, synthetic, and cost analyses.
- `scripts/aggregation/`: validated exports, aggregation, report sync.
- `scripts/figures/`: publication figure generation.
- `scripts/validation/`: environment, smoke, and public-evidence checks.

The two original `src` modules remain in place because historical runners
import `adapters` directly after adding `src/` to `sys.path`. Moving them into
new packages during a documentation migration would risk changing import
resolution and historical behavior.

## Configuration and campaigns

- `config/historical/`: original YAML configurations.
- `config/downstream/`: controlled AG News protocol.
- `config/rank/`: 1,000-step screening and 5,000-step confirmation protocols.
- `config/scaling/`: constant-effective-scale ablation.
- `experiments/historical/`: depth, placement, multi-layer, local-rank, and
  long-convergence historical studies.
- `experiments/downstream-ag-news/`: six-run classification study.
- `experiments/rank-screening-1000/`: 24-run controlled screen.
- `experiments/rank-confirmation-5000/`: independent 24-run confirmation.
- `experiments/scaling-ablation/`: Policy A/B scaling comparison.
- `experiments/mechanism-analysis/`: fixed held-out sample definition and
  checkpoint-analysis provenance.
- `experiments/computational-cost/`: focused timing/memory benchmark.

Raw run directories and checkpoints remain ignored at their original root
paths because runners and provenance records refer to them and they are private.

## Evidence and reports

- `results/tables/`: canonical publication-safe numerical evidence.
- `results/figures/`: canonical figures grouped by campaign.
- `results/manifests/FIGURE_MANIFEST.csv`: per-figure source and aggregation.
- `research/paper/TECHNICAL_REPORT.md`: canonical complete report.
- `research/paper/SCIENTIFIC_MANUSCRIPT.md` and `.tex`: canonical paper sources.
- `research/paper/FINAL_RESEARCH_REVIEW.md`: concise integrated evidence review.
- `research/mathematical-analysis/`: derivative and mechanism reports.
- `research/reproducibility/`: audit, completion matrix, and reproduction.
- `research/limitations/`: publication readiness, limitations, and release audits.

`docs/data/` and `docs/assets/` contain Pages mirrors generated from canonical
sources. They remain in the deployable tree so the static site works without a
build plugin.

## Related work

- `comparisons/papers/`: verified metadata and official links; no redistributed
  third-party PDFs.
- `comparisons/models/`: standardized records for LoRA, DoRA, MoRA, HiRA,
  PERA, QuadraNet V2, LoRAN, and nonlinear bottleneck adapters.
- `comparisons/MATHEMATICAL_EQUIVALENCE.md`: formula-level classification.
- `comparisons/EXPERIMENTAL_METHODOLOGY.md`: protocol mapping.
- `comparisons/NOVELTY_ASSESSMENT.md`: supported and unsupported originality
  claims.
- `comparisons/figures/`: original comparison diagrams.

## Public site, tests, and archives

- `docs/`: English static site designed to work below
  `/federico-lolli-ai-research/`.
- `tests/`: adapter, derivative, checkpoint, and public-data tests.
- `tools/`: repository maintenance tools, including deterministic inventory.
- `tmp/`: recoverable tracked archive with `MANIFEST.csv`; never deployed.

Private ignored ZIPs, previews, environments, raw metrics, logs, and
checkpoints stay in place and are not automatically tracked.
