#!/usr/bin/env python3
"""Validate the additive Stability–Plasticity Pages publication."""
from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGE = DOCS / "projects" / "02-stability-plasticity" / "index.html"
SOURCE_PDF = ROOT / "analysis_full_ewc_memory" / "paper" / "Federico Lolli Stability–Plasticity Control in Parameter-Efficient Continual Factual Learning.pdf"
PUBLIC_PDF = DOCS / "data" / "Federico_Lolli_Stability_Plasticity_Control.pdf"
TITLE = "Stability–Plasticity Control in Parameter-Efficient Continual Factual Learning: EWC across Linear, Quadratic, and Combined Low-Rank Adapters"


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"] or "")
        for key in ("href", "src"):
            if values.get(key):
                self.links.append(values[key] or "")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_local_links() -> None:
    failures: list[str] = []
    for page in DOCS.rglob("*.html"):
        parser = Links()
        parser.feed(page.read_text(encoding="utf-8"))
        for raw in parser.links:
            parsed = urlsplit(raw)
            if parsed.scheme or parsed.netloc or raw.startswith(("mailto:", "#")):
                continue
            target = (page.parent / unquote(parsed.path)).resolve()
            if not parsed.path:
                target = page
            if target.is_dir():
                target = target / "index.html"
            if not target.exists():
                failures.append(f"{page.relative_to(ROOT)} -> {raw}")
    assert not failures, "broken local links:\n" + "\n".join(failures)


def main() -> None:
    assert PAGE.is_file()
    text = PAGE.read_text(encoding="utf-8")
    for required in (
        TITLE, "Federico Lolli", "MathJax", "Online empirical diagonal-Fisher EWC",
        "Experimental methodology", "Discussion", "Limitations",
        "Download full paper", "Source code", "References",
    ):
        assert required in text, required
    assert text.count("<figure") == 7
    assert text.count("<li>") >= 14
    assert "WikiText-2" not in text and "AG News" not in text
    assert "\\theta" in text and "\\lambda" in text and "A\\rightarrow B\\rightarrow C" in text
    for public_file in (DOCS / "data" / "stability-plasticity").iterdir():
        if public_file.suffix.lower() in {".md", ".json", ".csv", ".tex"}:
            assert str(Path.home()) not in public_file.read_text(encoding="utf-8"), public_file

    expected_figures = sorted((DOCS / "assets" / "stability-plasticity").glob("*.svg"))
    assert len(expected_figures) == 7
    for figure in expected_figures:
        ET.parse(figure)

    assert digest(SOURCE_PDF) == digest(PUBLIC_PDF)
    info = subprocess.run(["pdfinfo", str(PUBLIC_PDF)], check=True, text=True, capture_output=True).stdout
    assert f"Title:           {TITLE}" in info
    assert "Author:          Federico Lolli" in info
    assert "Pages:           24" in info

    rows = list(csv.DictReader((ROOT / "analysis_full_ewc_memory" / "tables" / "confirmation_aggregate.csv").open()))
    assert len(rows) == 9
    for row in rows:
        for metric in ("mean_forgetting", "mean_new_fact_acquisition_nll", "mean_final_memory_nll", "mean_final_probability", "mean_final_exact_match"):
            formatted = f'{float(row[metric + "_mean"]):.4f} ± {float(row[metric + "_sample_sd"]):.4f}'
            assert formatted in text, (row["method"], row["lambda"], metric)

    for old_url in (
        "index.html", "method.html", "experiments.html", "results.html", "mechanism.html",
        "comparisons.html", "conclusions.html", "reproducibility.html", "references.html",
        "formulation.html", "methods.html", "reproduce.html",
    ):
        assert (DOCS / "projects" / "01-quadratic-gpt2" / old_url).is_file()
    home = (DOCS / "index.html").read_text(encoding="utf-8")
    research = (DOCS / "research" / "index.html").read_text(encoding="utf-8")
    publications = (DOCS / "publications.html").read_text(encoding="utf-8")
    assert "Latest research" in home and "RESEARCH PROJECT 01" in home and "RESEARCH PROJECT 02" in research
    assert "RESEARCH MANUSCRIPT · 2026" in publications
    validate_local_links()
    print("PASS: Stability–Plasticity site, scientific table, PDF, figures, legacy URLs, and local links validated")


if __name__ == "__main__":
    main()
