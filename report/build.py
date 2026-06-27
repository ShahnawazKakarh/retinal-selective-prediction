"""Build the technical report PDF using WeasyPrint.

Usage:
    pip install 'weasyprint>=63'
    python report/build.py

Output: report/retinal-selective-prediction-v1.0.0.pdf
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPORT_DIR.parent
HTML_IN = REPORT_DIR / "report.html"
CSS_IN = REPORT_DIR / "style.css"
PDF_OUT = REPORT_DIR / "retinal-selective-prediction-v1.2.0.pdf"


def main() -> int:
    try:
        from weasyprint import CSS, HTML
    except ImportError:
        sys.exit("Missing dependency. Install with: pip install 'weasyprint>=63'")

    if not HTML_IN.exists():
        sys.exit(f"Missing {HTML_IN}")
    if not CSS_IN.exists():
        sys.exit(f"Missing {CSS_IN}")

    print(f"Building PDF from {HTML_IN}", flush=True)
    HTML(filename=str(HTML_IN), base_url=str(REPORT_DIR)).write_pdf(
        target=str(PDF_OUT),
        stylesheets=[CSS(filename=str(CSS_IN))],
    )
    size_kb = PDF_OUT.stat().st_size / 1024
    print(f"Wrote {PDF_OUT}  ({size_kb:.1f} KB)")
    print(f"Build date: {date.today().isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
