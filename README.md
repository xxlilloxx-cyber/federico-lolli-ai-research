# Federico Lolli’s AI Research

Author: Federico Lolli · Contact: xxlilloxx@gmail.com

This repository is an archive for reproducible exploratory AI research. It currently contains **Research Project 01: Low-Rank Quadratic Adaptation of GPT-2**.

## Project 01

The project asks whether a small, activation-dependent quadratic correction can adapt a frozen GPT-2 Small projection usefully when compared with a parameter-matched linear LoRA correction. It evaluates a frozen GPT-2 backbone on WikiText-2, beginning with several adapter families and ending with a controlled 12,288-parameter, three-seed, 5,000-step LoRA-versus-Symmetric-Quadratic study with held-out test evaluation.

The measured conclusion is conditional: Symmetric Quadratic has a clear advantage with one concentrated insertion point, the advantage narrows for two points, and LoRA has lower held-out test loss with four distributed points. The repository does not claim universal superiority or novelty by assertion.

## Repository map

- `docs/` — English static GitHub Pages source for review. It is not deployed.
- `src/` — adapter implementation, insertion logic, checkpoint save/load, and trainer.
- `scripts/` — environment checks, runners, test evaluation, and analysis.
- `configs/` — experiment configurations.
- `tests/` — unit and checkpoint tests.
- `report/` — technical reports and publication-safe aggregate tables.
- `docs/data/` — site mirrors of final tables and the technical report.

The Pages site is organized for future projects. Project 01 has separate overview, formulation, methods, results, related-work, and reproduction pages. Open `docs/index.html` locally or follow `docs/README.md` for a local preview.

## Key final result

At the matched 12,288-parameter budget, final 5,000-step held-out WikiText-2 test loss is:

| Geometry | LoRA | Symmetric Quadratic |
|---|---:|---:|
| Block [2], rank 8 | 3.7741 ± 0.0022 | 3.7067 ± 0.0060 |
| Blocks [0, 11], rank 4 each | 3.6361 ± 0.0075 | 3.6306 ± 0.0025 |
| Blocks [0, 3, 7, 11], rank 2 each | 3.6192 ± 0.0115 | 3.6452 ± 0.0149 |

Final means and per-seed values are in `report/LONG_CONVERGENCE_AND_TEST/aggregate_long.csv` and `long_results.csv`; the site mirrors are `docs/data/`.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/check_environment.py
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q
```

The final workflow is implemented in `scripts/run_long_convergence_5000.py`, `scripts/evaluate_long_test.py`, and `scripts/analyze_long_convergence.py`. It was designed for a 4 GB NVIDIA GPU using a FP16 backbone, FP32 adapters, 128-token sequences, batch size 1, and gradient accumulation 4.

## Licensing and third-party scope

Original repository code is licensed under the [MIT License](LICENSE). Original written material and original figures are licensed under [CC BY 4.0](LICENSE-CONTENT.md). These licenses do not apply to GPT-2 weights, WikiText-2, third-party libraries, external figures, or cited research; see [NOTICE.md](NOTICE.md).

## Publication status

The site and public-release material are on an unpublished review branch. Do not deploy GitHub Pages, merge to the default branch, or publish without Federico Lolli’s explicit approval.
