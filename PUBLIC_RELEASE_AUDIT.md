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
- Pages are intended to be previewed under `/federico-lolli-ai-research/`, not server root. The local project-base preview is documented in `docs/projects/01-quadratic-gpt2/reproduce.html`.
- Repository links use the intended public branch path and need to be retained as a stable public branch or changed to a release tag before publication.
- The three source SVG diagrams are present in `docs/assets/figures/`; their public PNG/SVG preview copies are included only in the local review package.

## Privacy review scope and limits

Tracked working-tree files and the intended `docs/` deployment payload were scanned for local home paths, unapproved email addresses, common token markers, and private-key markers. No matches were found. Original experimental artifacts remain locally preserved and ignored. This is **not** proof that unreachable local reflogs, external backups, or a future remote history are clean. No destructive history rewrite was performed. A reachable-branch history scan must be repeated immediately before any push/release.

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
16. Working-tree/deployment privacy review complete with stated history limitation.  
17. Review ZIP is generated from the current working tree after the final checks.
