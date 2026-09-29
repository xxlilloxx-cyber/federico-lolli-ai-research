"""Check that canonical final-study tables and plotted trajectory endpoints agree."""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data"


def main():
    aggregate = list(csv.DictReader((DATA / "long_final_aggregate.csv").open()))
    trajectory = list(csv.DictReader((DATA / "long_validation_trajectories.csv").open()))
    final = defaultdict(list)
    for row in trajectory:
        if int(row["step"]) == 5000:
            final[(row["geometry"], row["architecture"])].append(float(row["validation_loss"]))
    failures = []
    for row in aggregate:
        key = (row["geometry"], row["architecture"])
        value = sum(final[key]) / len(final[key])
        expected = float(row["mean_final_validation_loss"])
        if abs(value - expected) > 1e-10:
            failures.append((key, value, expected))
    if failures:
        raise SystemExit(f"endpoint mismatch: {failures}")
    if len(trajectory) != 3618:
        raise SystemExit(f"unexpected trajectory row count: {len(trajectory)}")
    print("OK: 6 final endpoints agree with aggregate table; 3,618 trajectory rows.")


if __name__ == "__main__":
    main()
