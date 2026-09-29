"""Validate tracked publication data, static links, and public-file hygiene."""
from __future__ import annotations

import csv
import math
import re
import subprocess
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
    forbidden = ("/home/", "federicololli@hotmail.com", "BEGIN PRIVATE KEY", "public-release-preparation")
    for name in tracked:
        path = ROOT / name
        if not path.is_file() or path.suffix.lower() in {".png", ".pdf", ".jpg", ".jpeg"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in forbidden:
            require(marker not in text, f"forbidden public marker in {name}: {marker}")


def main() -> None:
    validate_rank()
    links = validate_links()
    validate_public_hygiene()
    print(f"OK: canonical rank values, {links} local links/assets, and public hygiene")


if __name__ == "__main__":
    main()
