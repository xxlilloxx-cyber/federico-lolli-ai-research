#!/usr/bin/env python3
"""Create a deterministic repository inventory for the reorganization audit.

The tool is read-only with respect to research artifacts. It excludes Git object
storage and the local virtual environment, hashes regular files, records Git
tracking/ignore state, and applies conservative path-based classifications.
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tmp" / "migration" / "repository_inventory.csv"
EXCLUDED_PARTS = {".git", ".venv", ".pytest_cache", "__pycache__"}


def git_lines(*args: str) -> set[str]:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    )
    return {line for line in result.stdout.splitlines() if line}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def category(path: Path) -> str:
    text = path.as_posix().lower()
    name = path.name.lower()
    suffix = path.suffix.lower()
    if "checkpoint" in text or suffix in {".pt", ".pth", ".ckpt"}:
        return "CHECKPOINT"
    if text.startswith("src/"):
        return "SOURCE CODE"
    if text.startswith("tests/"):
        return "TEST"
    if text.startswith("configs/") or text.startswith("config/") or suffix in {".yaml", ".yml"}:
        return "CONFIGURATION"
    if text.startswith("scripts/"):
        if "plot" in name or "figure" in name:
            return "PLOTTING SCRIPT"
        if "train" in name or name.startswith("run_"):
            return "TRAINING SCRIPT"
        if "eval" in name:
            return "EVALUATION SCRIPT"
        if "analy" in name or "aggregate" in name or "export" in name:
            return "ANALYSIS SCRIPT"
        if "valid" in name or "check" in name or "smoke" in name:
            return "EVALUATION SCRIPT"
        return "ANALYSIS SCRIPT"
    if text.startswith("docs/") and suffix in {".html", ".css"}:
        return "PUBLIC WEBSITE"
    if suffix in {".png", ".svg", ".pdf"}:
        return "FIGURE"
    if suffix == ".csv":
        if any(part.startswith("results") for part in path.parts):
            return "RAW EXPERIMENT DATA"
        return "AGGREGATE RESULTS"
    if suffix == ".json" and any(part.startswith("results") for part in path.parts):
        return "RAW EXPERIMENT DATA"
    if "report" in name or text.startswith("report/") or text.startswith("analysis_"):
        return "SCIENTIFIC REPORT" if "technical_report" in text else "INTERMEDIATE REPORT"
    if name.endswith(".md"):
        if "related" in text or "reference" in text:
            return "BIBLIOGRAPHY"
        if "comparison" in text:
            return "COMPARISON WITH OTHER MODELS"
        return "DEVELOPMENT NOTE"
    if suffix == ".zip" or text.startswith("review-package") or text.startswith("site-preview"):
        return "DUPLICATE"
    return "SOURCE CODE" if suffix in {".py", ".sh"} else "DEVELOPMENT NOTE"


def visibility(path: Path, tracked: bool, ignored: bool) -> str:
    text = path.as_posix()
    if ignored and (text.startswith("results") or "checkpoint" in text or path.suffix in {".pt", ".ckpt", ".pth"}):
        return "PRIVATE RAW EVIDENCE"
    if text.startswith("docs/"):
        return "PUBLIC DEPLOYMENT"
    return "PUBLIC REPOSITORY" if tracked else "LOCAL PRIVATE/WORKING"


def importance(cat: str) -> str:
    if cat in {"RAW EXPERIMENT DATA", "CHECKPOINT", "SOURCE CODE", "CONFIGURATION", "SCIENTIFIC REPORT"}:
        return "CRITICAL"
    if cat in {"AGGREGATE RESULTS", "FIGURE", "TEST", "TRAINING SCRIPT", "EVALUATION SCRIPT", "ANALYSIS SCRIPT"}:
        return "HIGH"
    if cat in {"DUPLICATE", "OBSOLETE MATERIAL"}:
        return "LOW"
    return "MEDIUM"


def proposed_destination(path: Path, cat: str, tracked: bool) -> str:
    text = path.as_posix()
    if not tracked and (text.startswith("results") or cat == "CHECKPOINT"):
        return text + " (preserve privately in place; reference by manifest)"
    if cat == "SOURCE CODE":
        return text if text.startswith("src/") else f"src/utils/{path.name}"
    mapping = {
        "TRAINING SCRIPT": f"scripts/training/{path.name}",
        "EVALUATION SCRIPT": f"scripts/evaluation/{path.name}",
        "ANALYSIS SCRIPT": f"scripts/analysis/{path.name}",
        "PLOTTING SCRIPT": f"scripts/figures/{path.name}",
        "TEST": text,
        "CONFIGURATION": "config/historical/" + path.name,
        "AGGREGATE RESULTS": "results/tables/" + path.name,
        "SCIENTIFIC REPORT": "research/paper/" + path.name,
        "INTERMEDIATE REPORT": "experiments/<campaign>/" + path.name,
        "FIGURE": "results/figures/<campaign>/" + path.name,
        "BIBLIOGRAPHY": "comparisons/papers/" + path.name,
        "COMPARISON WITH OTHER MODELS": "comparisons/" + path.name,
        "PUBLIC WEBSITE": text,
        "DUPLICATE": "tmp/duplicate-files/" + text,
        "DEVELOPMENT NOTE": "research/methodology/" + path.name,
    }
    return mapping.get(cat, text)


def main() -> None:
    tracked_paths = git_lines("ls-files")
    ignored_paths = git_lines("ls-files", "--others", "-i", "--exclude-standard")
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
            continue
        rel = path.relative_to(ROOT)
        rel_text = rel.as_posix()
        cat = category(rel)
        tracked = rel_text in tracked_paths
        ignored = rel_text in ignored_paths
        files.append(
            {
                "current_path": rel_text,
                "git_status": "TRACKED" if tracked else ("IGNORED" if ignored else "UNTRACKED"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "category": cat,
                "scientific_importance": importance(cat),
                "visibility": visibility(rel, tracked, ignored),
                "active_dependency_note": "inspect references before move" if tracked else "private artifact; path used by runners",
                "proposed_destination": proposed_destination(rel, cat, tracked),
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(files[0]))
        writer.writeheader()
        writer.writerows(sorted(files, key=lambda row: row["current_path"]))
    print(f"Wrote {len(files)} records to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
