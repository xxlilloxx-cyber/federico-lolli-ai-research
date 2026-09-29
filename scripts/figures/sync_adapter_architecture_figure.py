"""Validate and publish the canonical Figure 1 SVG.

The editable source is ``results/figures/architecture/adapter_architecture_comparison.svg``.
The GitHub Pages copy is generated from that source. The separately supplied
PNG failed the mathematical audit and is deliberately not read or overwritten.
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/figures/architecture/adapter_architecture_comparison.svg"
PUBLIC = ROOT / "docs/assets/publication/adapter_architecture_comparison.svg"

REQUIRED = (
    "xW + b",
    "Δy = (α/r)(xA)B",
    "Δy = (α/r)((xU) ⊙ (xU))P",
    "z ⊙ z",
    "element-wise",
    "r(d + d_out)",
)
FORBIDDEN = ("W₀x", "diag(x", "Px", "Vx")


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    missing = [token for token in REQUIRED if token not in text]
    forbidden = [token for token in FORBIDDEN if token in text]
    if missing or forbidden:
        raise SystemExit(f"Figure 1 audit failed: missing={missing}; forbidden={forbidden}")
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, PUBLIC)
    print(f"validated and synchronized {PUBLIC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
