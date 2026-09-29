# Federico Lolli AI Research — GitHub Pages source

This directory is the English GitHub Pages source for Federico Lolli's AI
research archive. The site introduces Research Project 01, *Symmetric
Quadratic Adaptation*, and links its methods, results, references, and
reproducibility material.

## Local preview

From the repository root, mount `docs/` beneath the eventual project path:

```bash
mkdir -p site-preview/base
ln -s ../../docs site-preview/base/federico-lolli-ai-research
python3 -m http.server 8000 --directory site-preview/base
```

Open `http://localhost:8000/federico-lolli-ai-research/`.

## Research materials

- `data/long_final_per_seed.csv`, `data/long_final_aggregate.csv`, and
  `data/long_validation_trajectories.csv` are generated Pages mirrors of the
  canonical public evidence under `../results/tables/`.
- `data/README.md` defines the data fields; `data/FIGURE_MANIFEST.md` records
  the source and aggregation used for each public figure.
- `data/TECHNICAL_REPORT.md` mirrors the canonical standalone report at
  `../research/paper/TECHNICAL_REPORT.md`.
- `data/Federico_Lolli_Symmetric_Quadratic_Adaptation_GPT2.pdf` is the primary public paper, while
  `data/SCIENTIFIC_MANUSCRIPT.md` and `.tex` mirror its canonical sources; the
  adjacent `.tex` file is its publication-oriented LaTeX version.
- `../scripts/validation/validate_public_long_data.py` validates final
  endpoints, and `../scripts/figures/generate_public_long_figures.py`
  regenerates public figures from
  sanitized CSVs.

## Licences and attribution

Original repository code is available under the [MIT License](https://github.com/xxlilloxx-cyber/federico-lolli-ai-research/blob/main/LICENSE).
Original site text and original figures are available under [CC BY 4.0](https://github.com/xxlilloxx-cyber/federico-lolli-ai-research/blob/main/LICENSE-CONTENT.md).
These terms do not relicense third-party software, pretrained GPT-2 weights,
WikiText-2, cited papers, or linked external materials; see the
[NOTICE](https://github.com/xxlilloxx-cyber/federico-lolli-ai-research/blob/main/NOTICE.md).
