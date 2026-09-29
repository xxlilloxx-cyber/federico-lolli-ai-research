# Reproducibility

## Environment

Use the pinned/limited dependencies in `requirements.txt`. The measured machine
has Ubuntu, an NVIDIA RTX A500 Embedded GPU with 4 GiB VRAM, 32 GiB host RAM,
and the versions recorded by `scripts/check_environment.py`.

```bash
git clone https://github.com/xxlilloxx-cyber/federico-lolli-ai-research.git
cd federico-lolli-ai-research
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/check_environment.py
```

## Validate published data and tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. .venv/bin/python -m pytest -q
.venv/bin/python scripts/validate_public_long_data.py
```

## Rebuild analyses without training

```bash
.venv/bin/python scripts/analyze_adapter_mechanisms.py
.venv/bin/python scripts/generate_mechanism_figures.py
.venv/bin/python scripts/export_rank_convergence.py
.venv/bin/python scripts/benchmark_adapter_cost.py
```

The fixed held-out mechanism sample is documented in
`analysis_mechanisms_2026-09-29/heldout_sample.json`. The 5,000-step confirmation
and scaling runners are resume-safe, but rerunning training is unnecessary when
their valid artifacts exist. Raw checkpoints/results remain local and ignored;
sanitized CSVs under `docs/data/` are the public evidence layer.
