"""Synchronize the canonical technical-report mirror after validating public data.

The canonical source is report/TECHNICAL_REPORT.md. This script deliberately
never reads private result directories and never writes a historical report:
its public workflow is to validate sanitized final-study CSVs and copy the
canonical report into docs/data/ for the Pages download link.
"""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "TECHNICAL_REPORT.md"
MIRROR = ROOT / "docs" / "data" / "TECHNICAL_REPORT.md"
CHECKER = ROOT / "scripts" / "validate_public_long_data.py"


def main() -> None:
    if not REPORT.exists():
        raise SystemExit(f"canonical report is missing: {REPORT}")
    subprocess.run([sys.executable, str(CHECKER)], check=True)
    MIRROR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPORT, MIRROR)
    print(f"validated public data and synchronized {MIRROR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
