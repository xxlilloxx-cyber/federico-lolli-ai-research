# Public-release review audit

**Repository root:** the checked-out repository root (the local checkout name is not a repository subdirectory).
**Branch:** `public-release-preparation`
**Commit at audit start:** `a8bad9a`
**Publication status:** local review only. No push, merge, GitHub Pages enablement, deployment, or public release was performed.

## Canonical sources and mirrors

- Canonical standalone report: `report/TECHNICAL_REPORT.md`.
- Public report mirror: `docs/data/TECHNICAL_REPORT.md`; it is copied from the canonical report after report changes.
- Canonical final public data: `docs/data/long_final_per_seed.csv`, `long_final_aggregate.csv`, and `long_validation_trajectories.csv`.
- Publication scripts: `scripts/export_public_long_data.py`, `scripts/generate_public_long_figures.py`, and `scripts/validate_public_long_data.py`.
- Private original source: ignored `results_long_convergence_5000/` summaries, metrics, checkpoints, and logs. It is retained locally but excluded from the public deployment payload.
- `aggregate_long.csv`, `long_results.csv`, and `report/MASTER_RESULTS.csv` are legacy/historical outputs. Their old `validation_loss`/`final_val_loss` summary naming must not be read as step-5,000 validation.

## Corrected numerical definitions

`best_recorded_validation_loss` is the minimum fresh validation result **within each seed**, then aggregated. `final_validation_loss` is the fresh evaluation at step 5,000. `final_test_loss` is held-out test loss of that same final checkpoint. Perplexity aggregates are means of per-seed perplexities. Test–validation differences are formed within seed, then averaged with sample SD.

The recalculated final validation means are 3.3703/3.2831 (concentrated LoRA/Symmetric), 3.2186/3.1876 (two-layer), and 3.1743/3.1925 (four-layer). The corresponding signed differences are -0.0872, -0.0310, and +0.0182. The older 3.1537/3.1557 four-layer values are best-recorded values, not final endpoints.

## Step-0 and trajectory provenance

`3.8293571472` is an original pre-update validation result stored in every run summary. It is deterministic under zero-preserving initialization and must not be treated as 18 independent baseline observations. The raw metrics have 5,000 rows per run: 201 fresh post-update evaluations (step 1 plus steps 25…5000) and 4,799 cached rows. Step 5000 is a scheduled final evaluation only once. The sanitized trajectory export has 3,618 rows: 18 labelled pre-update rows plus 3,600 scheduled fresh rows. Post-update step-1 rows are deliberately excluded.

## AULC definition

The legacy AULC value is a **simple arithmetic mean** of the 5,000 stored validation columns in a run, including cached repeats and excluding summary step 0; these run-level means are then averaged across seeds. It is not an integrated area. A normalized trapezoidal fresh-point area would be `sum((L_i + L_{i+1})/2 * (t_{i+1}-t_i))/5000`; it has not silently replaced legacy AULC.

## Figure/table checks

`./.venv/bin/python scripts/validate_public_long_data.py` passed: all six mean step-5000 trajectory endpoints match `long_final_aggregate.csv`, and the trajectory row count is 3,618. `docs/data/FIGURE_MANIFEST.md` records inputs, filters, seeds, aggregation, uncertainty, and generator for every final figure. Current PNGs were regenerated with `scripts/generate_public_long_figures.py`; no smoothing is applied and bands/bars are labelled sample SD.

## Link, asset, and public-content checks

- All site-relative HTML asset targets were checked from `docs/`; no missing local asset was found.
- Pages were checked under `/federico-lolli-ai-research/`, not server root: 14/14 tested page, CSS, SVG, PNG, CSV, and report targets returned HTTP 200. The local project-base preview is documented in `docs/projects/01-quadratic-gpt2/reproduce.html`.
- Repository links use the intended public branch path and need to be retained as a stable public branch or changed to a release tag before publication.
- The three source SVG diagrams are present in `docs/assets/figures/`; their public PNG/SVG preview copies are included only in the local review package.

## Privacy review scope and limits

Tracked working-tree files and the intended `docs/` deployment payload were scanned for local home paths, unapproved email addresses, common token markers, and private-key markers. No matches were found. Original experimental artifacts remain locally preserved and ignored.

**Release blocker found during the all-references history scan.** Local stash
references contain historical commits with raw experiment-result directories,
run logs, configuration metadata, and adapter checkpoint artifacts. Those
historical commits are not ancestors of the current public-release branch, but
they are reachable through local stash references and must not be pushed or
otherwise included in a publication ref. No values from those artifacts are
recorded here. Before any publication, remove or isolate the affected local
stash references through an approved history-cleanup process, then repeat the
history scan on the exact branch/tag selected for release. No destructive
rewrite was performed in this review pass.

## Requirement checklist

1. Canonical files and mirrors — complete.
2. Explicit validation fields — complete.
3. Canonical report rewritten — complete.
4. Historical versus fresh comparisons separated — complete.
5. Step-0 provenance and row accounting — complete.
6. Legacy AULC defined; no silent redefinition — complete.
7. Row-vector mathematics/scaling — complete.
8. GPT-2 insertion SVG corrected — complete.
9. Recorded methods and unavailable metadata marked — complete.
10. Final and historical results/conclusions — complete.
11. Reproducible figures plus manifest/checker — complete.
12. Related work: LoRAN/PERA distinction and ACL 2026 record — complete.
13. Public identity/navigation/contact/license scope — complete.
14. Project-base link policy documented; final remote link check remains required before deployment.
15. Reproduction instructions and public data schema — complete.
16. Working-tree/deployment privacy review passed; all-references history scan found the local-stash raw-artifact blocker described above.
17. Review ZIP generation is deferred until the history blocker is resolved or explicitly accepted for a branch-only release procedure.
## September 2026 controlled-evidence update

Tracked/publication-candidate text and data were rescanned after the controlled
rank, scaling, mechanism, and cost analyses. The scan covered tracked working
tree files and `docs/` for local home paths, unapproved email addresses,
private-key markers, common token markers, checkpoints, binaries, and raw result
directories. No matching sensitive value was found in the candidate payload.
Raw controlled results and checkpoints are preserved locally and excluded by
`.gitignore`; sanitized CSVs under `docs/data/` provide figure provenance.

This working-tree scan is not a substitute for history review. `git stash list`
still exposes the previously documented local stash reference
`release-site-revision-before-sanitized-history`. It remains a release blocker
for any operation that would expose all reachable refs. No stash ref or Git
history was modified in this work.
