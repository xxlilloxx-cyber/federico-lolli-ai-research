# Federico Lolli AI Research — Pages source

This is a dependency-free, English GitHub Pages source tree prepared for review only. It must not be deployed, published, or merged to a default branch without explicit approval.

## Structure

- `index.html` — archive home page.
- `research/` — research-project index.
- `projects/01-quadratic-gpt2/` — Project 01 overview, formulation, methods, results, related work, and reproduction guide.
- `about.html` — author, approved contact address, and license scope.
- `assets/` — stylesheet, original SVG diagrams, and selected result figures.
- `data/` — publication-safe CSV and Markdown mirrors used by site claims.

The public page text is contained directly in the page files to keep maintenance simple. Figures derived from measurements use the listed source CSV files; original explanatory SVG diagrams are stored beside the site CSS.

## Local review

From the repository root:

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/docs/`.
