# Project 01 public numerical data

These are the publication-safe, authoritative exports for the final 5,000-step
campaign. They contain no local paths, checkpoint payloads, logs, or machine
diagnostics.

* `long_final_per_seed.csv` has one row for each of the 18 completed runs.
  `best_validation_loss` is the minimum fresh validation loss within that run;
  `final_validation_loss` is the loss evaluated at optimizer step 5,000; and
  `test_loss_final_checkpoint` is the held-out loss from that same final
  checkpoint.
* `long_final_aggregate.csv` aggregates per-seed values with the arithmetic
  mean and sample standard deviation (`n=3`). Perplexity is averaged across
  per-seed perplexities; it is not computed by exponentiating a displayed mean
  loss. `test_minus_final_validation_loss` is paired within each seed before
  aggregation.
* `long_validation_trajectories.csv` has 3,618 rows: one original pre-update
  baseline row and 200 fresh scheduled validation points (steps 25–5,000) for
  every run. The post-update step-1 measurements are intentionally excluded so
  curves do not mix the common baseline with a post-update value.

`aggregate_long.csv` and `long_results.csv` are retained legacy exports. Their
historical `validation_loss` field is a best-recorded summary, not necessarily
the step-5,000 value. Use the `long_*` files above for current figures and
final-study tables.

Regenerate these exports privately with `scripts/export_public_long_data.py`.
It requires the ignored original run artifacts. Regenerate public figures from
these CSVs with `scripts/generate_public_long_figures.py`.
