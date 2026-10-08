#!/usr/bin/env python3
"""Validate the submission manuscript without modifying scientific inputs."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT.parent
REPO = ANALYSIS.parent


def main() -> None:
    md = (ROOT / "FINAL_PAPER_SUBMISSION.md").read_text()
    tex = (ROOT / "FINAL_PAPER_SUBMISSION.tex").read_text()
    pdf = ROOT / "Federico Lolli Stability–Plasticity Control in Parameter-Efficient Continual Factual Learning.pdf"
    assert pdf.is_file() and pdf.stat().st_size > 100_000

    confirmation = pd.read_csv(ANALYSIS / "tables" / "confirmation_full_metrics.csv")
    assert len(confirmation) == 27
    assert confirmation.canonical_run_id.nunique() == 27
    assert not confirmation.duplicated(["method", "lambda", "seed"]).any()
    assert set(confirmation.seed) == {42, 123, 456}
    assert np.isfinite(confirmation.select_dtypes(include="number").to_numpy()).all()

    identities = pd.read_csv(ROOT / "submission_data" / "forgetting_identity_verification.csv")
    max_error = float(identities.filter(like="error_").abs().to_numpy().max())
    assert max_error < 1e-6

    canonical = pd.read_csv(ANALYSIS / "data" / "CANONICAL_RUNS.csv")
    assert len(canonical) == canonical.canonical_run_id.nunique() == 72
    assert canonical.complete.all()

    forbidden = re.compile(r"TODO|TBD|FIXME|XXX|filecite|turn0|turn1|ChatGPT|Codex|INSERT|\?\?")
    assert not forbidden.search(md)
    assert not forbidden.search(tex)
    assert "Federico Lolli" in md and "Federico Lolli" in tex
    assert "A\\rightarrow B\\rightarrow C" in md
    assert "Final correct-answer probability" in md
    assert len(re.findall(r"^!\[", md, re.M)) == 7
    assert len(re.findall(r"^\| Method \|", md, re.M)) == 11
    for match in re.finditer(r"\]\(([^)]+)\)", md):
        target = ROOT / match.group(1)
        assert target.exists(), target

    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    required = {
        "Title:": "Stability–Plasticity Control in Parameter-Efficient Continual Factual Learning: EWC across Linear, Quadratic, and Combined Low-Rank Adapters",
        "Author:": "Federico Lolli",
        "Subject:": "Parameter-efficient continual factual learning with Elastic Weight Consolidation",
        "Pages:": "24",
    }
    for field, value in required.items():
        line = next(x for x in info.splitlines() if x.startswith(field))
        assert line.split(":", 1)[1].strip() == value, line

    log = (ROOT / "submission_build_final" / "FINAL_PAPER_SUBMISSION.log").read_text()
    for token in ["Overfull", "undefined references", "Citation `", "Emergency stop"]:
        assert token not in log, token

    print(json.dumps({
        "status": "valid",
        "canonical_runs": len(canonical),
        "confirmation_runs": len(confirmation),
        "max_forgetting_identity_error": max_error,
        "figures": 7,
        "tables": 11,
        "references": 14,
        "pages": 24,
    }, indent=2))


if __name__ == "__main__":
    main()
