# Public-release review audit

Branch: `public-release-preparation`  
Publication status: local review only; no push, merge, Pages deployment, or publishing performed.

## Canonical sources

- Canonical report: `report/TECHNICAL_REPORT.md`; site mirror: `docs/data/TECHNICAL_REPORT.md`.
- Canonical final public data: `docs/data/long_final_per_seed.csv` and `docs/data/long_validation_trajectories.csv`.
- Private source artifacts: ignored `results_long_convergence_5000/` summaries, metrics, checkpoints, and logs.
- Export script: `scripts/export_public_long_data.py`; plot script: `scripts/generate_public_long_figures.py`.

## Definitions and consistency

`best_recorded_validation_loss` is a per-seed minimum. `final_validation_loss` is step 5000. `final_test_loss` is final-checkpoint test loss. The former summary field `validation_loss` is best-recorded loss. Current plots use one original step-0 pre-update measurement and fresh steps 25–5000 only; 18 × 201 = 3,618 rows. Step 1 in original metrics is post-update.

Legacy AULC is a per-run arithmetic mean over all 5,000 stored validation rows, including cached repeats, then a seed mean. It is not a numerical integral and remains historical.

## Checks

- Final table means regenerated from per-seed CSV; expected final validation means match to four decimals.
- Figures derive only from sanitized public CSV fields and show sample SD bands/error bars where applicable.
- `pytest -q tests`: 4 passed.
- Local asset references: 0 failures.
- Base-path page request under `/federico-lolli-ai-research/`: HTTP 200.
- Desktop and mobile previews: 18 images under `site-preview/final/`.

## Privacy scope

Tracked working-tree scan searched for local paths, unapproved email, common token prefixes, and private-key markers; no matches were found. Raw private experiment artifacts remain local and ignored. This scan does not prove that unreachable local reflogs, external backups, or a remote service history are clean. No destructive history rewrite was performed in this revision.

## Open items

- Model and dataset revision hashes were not recorded by original runs.
- PERA publication/acceptance status must be checked against its original source before release language is finalized.
- Historical reports retain their own legacy metric definitions and are explicitly separated from final fresh trajectories.

## Requirement checklist

1. Canonical files: complete. 2. Validation definitions: complete. 3. Canonical report: complete. 4. Historical/fresh separation: complete. 5. Step-0 provenance: complete. 6. Legacy AULC definition: complete. 7. Row-vector mathematics: complete. 8. GPT-2 SVG: complete. 9. Final-campaign methods: complete; revisions unavailable. 10. Results and conclusions: complete in canonical report/site. 11. Reproducible final figures: complete. 12. Related-work distinction: complete; PERA status remains source-verification item. 13. Public navigation: complete. 14. Base-path and external repository links: complete for local assets and tested pages. 15. Reproduction instructions: complete. 16. Tracked-file privacy scope: complete; history limitation documented. 17. Versioned review ZIP: complete.
