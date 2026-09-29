"""Regression tests for the sanitized public final-study evidence."""
import csv
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"


def test_final_trajectory_endpoints_match_aggregate_table():
    aggregate = list(csv.DictReader((DATA / "long_final_aggregate.csv").open()))
    points = list(csv.DictReader((DATA / "long_validation_trajectories.csv").open()))
    final = {}
    for row in points:
        if int(row["step"]) == 5000:
            final.setdefault((row["geometry"], row["architecture"]), []).append(
                float(row["validation_loss"])
            )
    assert len(points) == 3618
    for row in aggregate:
        key = (row["geometry"], row["architecture"])
        mean = sum(final[key]) / len(final[key])
        assert abs(mean - float(row["mean_final_validation_loss"])) < 1e-10


def test_canonical_report_uses_explicit_final_study_terms():
    report = (ROOT / "research" / "paper" / "TECHNICAL_REPORT.md").read_text()
    for term in (
        "best_recorded_validation_loss",
        "final_validation_loss",
        "final_test_loss",
        "3.3703",
        "3.2831",
        "3.1743",
        "3.1925",
    ):
        assert term in report


def test_static_publication_links_and_values():
    subprocess.run([sys.executable, str(ROOT / "tools" / "validate_publication.py")], check=True)
