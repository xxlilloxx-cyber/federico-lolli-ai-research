# Reproducibility

## Environment

Use the pinned/limited dependencies in `requirements.txt`. The measured machine
has Ubuntu, an NVIDIA RTX A500 Embedded GPU with 4 GiB VRAM, 32 GiB host RAM,
and the versions recorded by `scripts/validation/check_environment.py`.

```bash
git clone https://github.com/xxlilloxx-cyber/federico-lolli-ai-research.git
cd federico-lolli-ai-research
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/validation/check_environment.py
```

## Validate published data and tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. .venv/bin/python -m pytest -q
.venv/bin/python scripts/validation/validate_public_long_data.py
.venv/bin/python tools/validate_publication.py
```

## Rebuild analyses without training

```bash
.venv/bin/python scripts/analysis/analyze_adapter_mechanisms.py
.venv/bin/python scripts/figures/generate_mechanism_figures.py
.venv/bin/python scripts/aggregation/export_rank_convergence.py
.venv/bin/python scripts/analysis/benchmark_adapter_cost.py
```

The fixed held-out mechanism sample is documented in
`experiments/mechanism-analysis/heldout_sample.json`. The 5,000-step confirmation
and scaling runners are resume-safe, but rerunning training is unnecessary when
their valid artifacts exist. Raw checkpoints/results remain local and ignored;
sanitized canonical CSVs under `results/tables/` are the public evidence layer;
`docs/data/` contains generated Pages mirrors.

## Controlled campaign commands

Run these only when reproducing training. Existing valid runs are already
complete and must not be overwritten.

```bash
# AG News: individual cells or resume-safe remaining-cell launcher
.venv/bin/python scripts/training/run_controlled_downstream.py --help
bash scripts/training/run_downstream_remaining.sh

# WikiText-2 controlled rank screen
.venv/bin/python scripts/training/run_rank_factorial_controlled.py --help

# Independent 5,000-step rank confirmation
bash scripts/training/run_rank_confirmation_all.sh

# Seed-42 constant-effective-scale ablation
bash scripts/training/run_rank_scaling_ablation.sh
```

The publication-safe protocol records are under `config/downstream/`,
`config/rank/`, and `config/scaling/`. Private per-run configurations are the
authoritative record for each executed cell and remain with ignored raw runs.

## Figure regeneration

```bash
.venv/bin/python scripts/figures/generate_public_long_figures.py
.venv/bin/python scripts/figures/generate_downstream_figures.py
.venv/bin/python scripts/aggregation/generate_rank_screening_report.py
.venv/bin/python scripts/aggregation/generate_rank_5000_report.py
.venv/bin/python scripts/aggregation/export_rank_convergence.py
.venv/bin/python scripts/aggregation/generate_scaling_ablation_report.py
.venv/bin/python scripts/figures/generate_mechanism_figures.py
```

Plot-only scripts read publication-safe CSVs where possible. Checkpoint-only
mechanism analysis and the cost benchmark require the ignored final adapter
checkpoints.

## Build and preview the website

```bash
.venv/bin/python tools/build_public_site.py
.venv/bin/python tools/validate_publication.py
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000/`. The deployed project base path is
`/federico-lolli-ai-research/`; all site-local references are relative so the
same files resolve under that prefix. `research/paper/SCIENTIFIC_MANUSCRIPT.md`
and `.tex` are the canonical paper sources. Build the PDF with:

```bash
cd research/paper
tectonic SCIENTIFIC_MANUSCRIPT.tex
cd ../..
cp research/paper/SCIENTIFIC_MANUSCRIPT.{md,tex,pdf} docs/data/
cp research/paper/Federico_Lolli_Symmetric_Quadratic_Adaptation_GPT2.pdf docs/data/
```

The files under `docs/data/` are generated public mirrors; the project overview
links to the compiled PDF.
