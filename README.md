# Federico Lolli’s AI Research

**Author:** Federico Lolli

**Contact:** xxlilloxx@gmail.com

**Status:** independent exploratory research; no claim of peer review or
universal superiority.

## Research objective

Research Project 01 asks whether structured quadratic interactions can improve
the expressive efficiency of low-rank Transformer adaptation under controlled
parameter budgets. A pretrained GPT-2 backbone is frozen, and only small
adapter branches are trained.

LoRA supplies the direct linear baseline:

`Delta y = (alpha/r)(xA)B`.

The Symmetric Quadratic Adapter instead learns:

`Delta y = (alpha/r)((xU) ⊙ (xU))P`.

For output `j`, this is the symmetric quadratic form
`x Q_j x^T`, with `Q_j=(alpha/r)U diag(P[:,j])U^T`. The implementation shares
the projection directions in `U` across outputs and learns signed
output-specific coefficients in `P`.

## Research Project 02 — Stability–Plasticity Control

The second research line studies continual factual learning under a fixed
parameter-efficient budget. It compares LoRA, Symmetric, and Combined adapters
in a no-replay `A → B → C` sequence and measures how online empirical
diagonal-Fisher EWC controls retention and acquisition. The candidate EWC
coefficients selected on seed 42 are evaluated on two additional seeds.

- [Scientific web page](https://xxlilloxx-cyber.github.io/federico-lolli-ai-research/projects/02-stability-plasticity/)
- [Full paper PDF](https://xxlilloxx-cyber.github.io/federico-lolli-ai-research/data/Federico_Lolli_Stability_Plasticity_Control.pdf)
- [Experiment implementation](experiments/factual_learning_lora_symmetric/)
- [Final analysis](analysis_full_ewc_memory/)

## Experimental program

The repository preserves distinct campaigns rather than combining incompatible
protocols:

- historical adapter-family, placement, depth, and fixed-budget studies;
- a six-run controlled GPT-2/AG News downstream comparison;
- a 24-run, 1,000-step WikiText-2 rank screening;
- an independent 24-run, 5,000-step WikiText-2 rank confirmation with
  final-checkpoint held-out testing;
- a separate seed-42 constant-effective-scale ablation;
- checkpoint-only spectral, Jacobian, Hessian, and convergence analyses;
- a focused rank-4 GPU cost benchmark.

## Main measured findings

Under the controlled 5,000-step WikiText-2 block-0 protocol, Symmetric had
lower mean final validation and held-out test loss than LoRA at ranks 1, 2, 4,
and 8 across the three evaluated seeds. At rank 8, mean test loss was
`3.7319 ± 0.0035` for LoRA and `3.6938 ± 0.0044` for Symmetric.

The independent AG News experiment had the opposite mean ordering: LoRA test
accuracy was `0.8201 ± 0.0051`, while Symmetric accuracy was
`0.8021 ± 0.0240`. These results support task-, rank-, placement-, and
training-duration-dependent behavior. They do not establish universal
superiority, state of the art, or broad transfer.

## Repository map

- [`src/`](src/) — adapter and training implementation.
- [`scripts/`](scripts/) — grouped training, evaluation, aggregation, analysis,
  figure, and validation tools.
- [`config/`](config/) — historical and controlled publication-safe protocols.
- [`experiments/`](experiments/) — campaign reports and links to private raw
  evidence.
- [`results/`](results/) — canonical publication-safe tables, figures, and
  provenance manifests.
- [`research/`](research/) — paper, mathematical analysis, reproducibility, and
  limitations.
- [`papers/`](papers/) — stable indexes for publication-oriented research artifacts.
- [`comparisons/`](comparisons/) — verified related work, mathematical
  equivalence, methodology, and originality assessment.
- [`docs/`](docs/) — generated/static GitHub Pages source.
- [`tests/`](tests/) — unit and evidence-consistency tests.
- [`tmp/`](tmp/) — recoverable archive for non-canonical tracked material.

See [`REPOSITORY_STRUCTURE.md`](REPOSITORY_STRUCTURE.md) for the complete map.

## Read and reproduce

- Stability–Plasticity Control paper:
  [`docs/data/Federico_Lolli_Stability_Plasticity_Control.pdf`](docs/data/Federico_Lolli_Stability_Plasticity_Control.pdf),
  with its [research page](docs/projects/02-stability-plasticity/index.html).

- Canonical technical report:
  [`research/paper/TECHNICAL_REPORT.md`](research/paper/TECHNICAL_REPORT.md).
- Scientific manuscript:
  [`research/paper/SCIENTIFIC_MANUSCRIPT.md`](research/paper/SCIENTIFIC_MANUSCRIPT.md)
  with [LaTeX source](research/paper/SCIENTIFIC_MANUSCRIPT.tex).
- Final evidence review:
  [`research/paper/FINAL_RESEARCH_REVIEW.md`](research/paper/FINAL_RESEARCH_REVIEW.md).
- Reproduction guide:
  [`research/reproducibility/REPRODUCIBILITY.md`](research/reproducibility/REPRODUCIBILITY.md).
- Model comparison:
  [`comparisons/MODEL_COMPARISON.md`](comparisons/MODEL_COMPARISON.md).
- Publication-readiness review:
  [`research/limitations/PUBLICATION_READINESS_REPORT.md`](research/limitations/PUBLICATION_READINESS_REPORT.md).
- Static site: `docs/index.html`; intended project URL:
  <https://xxlilloxx-cyber.github.io/federico-lolli-ai-research/>.
- Project 01 website:
  <https://xxlilloxx-cyber.github.io/federico-lolli-ai-research/projects/01-quadratic-gpt2/>.
- Project 02 website:
  <https://xxlilloxx-cyber.github.io/federico-lolli-ai-research/projects/02-stability-plasticity/>.

Minimal validation from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/validation/check_environment.py
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/validation/validate_public_long_data.py
```

Plot reproduction uses publication-safe CSVs and does not require private
checkpoints. Training runners create ignored raw artifacts; consult the
campaign README before launching one.

## Licensing and third-party scope

Original code is licensed under the [MIT License](LICENSE). Original text and
figures are licensed under [CC BY 4.0](LICENSE-CONTENT.md). These licences do
not cover GPT-2 weights, WikiText-2, AG News, third-party libraries, cited
papers, or external assets; see [NOTICE.md](NOTICE.md).

No remote publication, merge, or deployment is performed by repository
reorganization work.
