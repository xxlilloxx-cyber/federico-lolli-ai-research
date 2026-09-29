"""Export sanitized final-campaign tables from private run artifacts.

The input directories are intentionally ignored by git.  The output contains
no local paths, checkpoints, or logs and is sufficient to reproduce the
published 5,000-step plots.
"""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "results_long_convergence_5000" / "budget12288_5000"
TEST = ROOT / "results_long_convergence_5000" / "test_eval"
OUT = ROOT / "docs" / "data"


def main():
    rows, trajectory = [], []
    for summary_path in sorted(PRIVATE.glob("*/summary.json")):
        run = summary_path.parent.name
        summary = json.loads(summary_path.read_text())
        metrics = list(csv.DictReader((summary_path.parent / "metrics.csv").open()))
        test = json.loads((TEST / f"{run}.json").read_text())
        final = metrics[-1]
        final_loss = float(final["validation_loss"])
        test_loss = float(test["test_loss"])
        rows.append({
            "run_label": run,
            "geometry": summary["strategy"],
            "architecture": summary["kind"],
            "seed": summary["seed"],
            "local_rank": summary["rank"],
            "layers": ",".join(map(str, summary["layers"])),
            "trainable_parameters": summary["trainable_parameters"],
            "initial_validation_loss": summary["initial_validation_loss"],
            "best_validation_loss": summary["validation_loss"],
            "best_validation_step": summary["best_step"],
            "final_validation_loss": final_loss,
            "final_validation_perplexity": math.exp(final_loss),
            "test_loss_final_checkpoint": test_loss,
            "test_perplexity_final_checkpoint": test["test_perplexity"],
            "test_minus_final_validation_loss": test_loss - final_loss,
        })
        # Step 0 is the actual pre-update validation measurement.  Subsequent
        # values are only fresh validation passes, not cached per-step repeats.
        trajectory.append({"run_label": run, "geometry": summary["strategy"],
                           "architecture": summary["kind"], "seed": summary["seed"],
                           "step": 0, "validation_loss": summary["initial_validation_loss"]})
        for point in metrics:
            step = int(point["step"])
            if step % int(summary.get("eval_every", 25)) == 0 or step == int(summary["steps"]):
                trajectory.append({"run_label": run, "geometry": summary["strategy"],
                                   "architecture": summary["kind"], "seed": summary["seed"],
                                   "step": step, "validation_loss": point["validation_loss"]})
    OUT.mkdir(parents=True, exist_ok=True)
    aggregates = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["geometry"], row["architecture"])].append(row)
    for (geometry, architecture), group in sorted(groups.items()):
        aggregate = {"geometry": geometry, "architecture": architecture,
                     "n_seeds": len(group), "seeds": ",".join(str(x["seed"]) for x in group),
                     "local_rank": group[0]["local_rank"], "layers": group[0]["layers"],
                     "trainable_parameters": group[0]["trainable_parameters"]}
        for field in ["best_validation_loss", "final_validation_loss", "final_validation_perplexity",
                      "test_loss_final_checkpoint", "test_perplexity_final_checkpoint",
                      "test_minus_final_validation_loss"]:
            values = [float(row[field]) for row in group]
            aggregate[f"mean_{field}"] = statistics.mean(values)
            aggregate[f"sample_sd_{field}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregates.append(aggregate)
    for path, data in [(OUT / "long_final_per_seed.csv", rows),
                       (OUT / "long_final_aggregate.csv", aggregates),
                       (OUT / "long_validation_trajectories.csv", trajectory)]:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=data[0].keys())
            writer.writeheader(); writer.writerows(data)
    print(f"wrote {len(rows)} final rows, {len(aggregates)} aggregates and {len(trajectory)} validation points")


if __name__ == "__main__":
    main()
