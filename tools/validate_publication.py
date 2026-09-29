"""Validate tracked publication data, static links, and public-file hygiene."""
from __future__ import annotations

import csv
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_rank() -> None:
    path = ROOT / "results/tables/rank-confirmation-5000/rank_5000_aggregate.csv"
    data = list(csv.DictReader(path.open()))
    require(len(data) == 8, "rank aggregate must contain 2 methods × 4 ranks")
    targets = {("LoRA", 1): 3.9256009909740004, ("Symmetric Quadratic", 1): 3.913083841291101,
               ("LoRA", 2): 3.849203721570904, ("Symmetric Quadratic", 2): 3.830496388972749,
               ("LoRA", 4): 3.7655791677899306, ("Symmetric Quadratic", 4): 3.7429156612247425,
               ("LoRA", 8): 3.731909922976005, ("Symmetric Quadratic", 8): 3.6938370284338924}
    for row in data:
        key = row["method"], int(row["rank"])
        require(math.isclose(float(row["final_test_loss_mean"]), targets[key], abs_tol=1e-12), f"rank value mismatch: {key}")


def validate_links() -> int:
    errors: list[str] = []
    checked = 0
    for page in DOCS.rglob("*.html"):
        text = page.read_text(encoding="utf-8")
        for attr, raw in re.findall(r'''(?:href|src)=(?:"([^"]+)"|'([^']+)')''', text):
            link = attr or raw
            if link.startswith(("http://", "https://", "mailto:", "#", "data:")):
                continue
            parsed = urlparse(link)
            target = (page.parent / unquote(parsed.path)).resolve()
            checked += 1
            if not target.exists():
                errors.append(f"{page.relative_to(ROOT)} -> {link}")
    require(not errors, "broken local links:\n" + "\n".join(errors))
    return checked


def validate_public_hygiene() -> None:
    tracked = subprocess.check_output(["git", "ls-files", "docs"], cwd=ROOT, text=True).splitlines()
    unapproved_email = "federicololli" + "@" + "hotmail.com"
    forbidden = ("/home/", unapproved_email, "BEGIN PRIVATE KEY", "public-release-preparation")
    for name in tracked:
        path = ROOT / name
        if not path.is_file() or path.suffix.lower() in {".png", ".pdf", ".jpg", ".jpeg"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in forbidden:
            require(marker not in text, f"forbidden public marker in {name}: {marker}")


def validate_project_metadata_and_figure() -> None:
    project = DOCS / "projects" / "01-quadratic-gpt2"
    pages = ["index.html", "method.html", "experiments.html", "results.html",
             "mechanism.html", "comparisons.html", "conclusions.html",
             "reproducibility.html", "references.html"]
    titles: set[str] = set()
    descriptions: set[str] = set()
    for name in pages:
        text = (project / name).read_text(encoding="utf-8")
        title = re.search(r"<title>([^<]+)</title>", text)
        description = re.search(r'<meta name="description" content="([^"]+)"', text)
        require(title is not None and title.group(1).strip(), f"missing title: {name}")
        require(description is not None and description.group(1).strip(), f"missing description: {name}")
        require(title.group(1) not in titles, f"duplicate title: {title.group(1)}")
        require(description.group(1) not in descriptions, f"duplicate description: {name}")
        titles.add(title.group(1))
        descriptions.add(description.group(1))
    index = (project / "index.html").read_text(encoding="utf-8")
    require("adapter_architecture_comparison.svg" in index, "canonical Figure 1 missing")
    require("adapter_architecture_comparison.png" not in index, "rejected PNG is referenced")
    svg = DOCS / "assets/publication/adapter_architecture_comparison.svg"
    ET.parse(svg)
    svg_text = svg.read_text(encoding="utf-8")
    for token in ("xW + b", "(α/r)(xA)B", "(α/r)((xU) ⊙ (xU))P", "z ⊙ z", "r(d + d_out)"):
        require(token in svg_text, f"Figure 1 missing canonical token: {token}")
    for forbidden in ("W₀x", "diag(x", "Px", "Vx"):
        require(forbidden not in svg_text, f"Figure 1 contains forbidden notation: {forbidden}")


def main() -> None:
    validate_rank()
    links = validate_links()
    validate_public_hygiene()
    validate_project_metadata_and_figure()
    print(f"OK: canonical rank values, {links} local links/assets, metadata, Figure 1, and public hygiene")


if __name__ == "__main__":
    main()
