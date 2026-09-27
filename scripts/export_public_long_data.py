"""Export sanitized final-campaign tables from private run artifacts.

The input directories are intentionally ignored by git.  The output contains
no local paths, checkpoints, or logs and is sufficient to reproduce the
published 5,000-step plots.
"""
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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
            "final_validation_loss": final["validation_loss"],
            "test_loss_final_checkpoint": test["test_loss"],
            "test_perplexity_final_checkpoint": test["test_perplexity"],
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
    for path, data in [(OUT / "long_final_per_seed.csv", rows),
                       (OUT / "long_validation_trajectories.csv", trajectory)]:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=data[0].keys())
            writer.writeheader(); writer.writerows(data)
    print(f"wrote {len(rows)} final rows and {len(trajectory)} validation points")


if __name__ == "__main__":
    main()
