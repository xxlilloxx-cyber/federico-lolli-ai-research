"""Generate search discovery files and canonical metadata for GitHub Pages."""
from __future__ import annotations

import argparse
import datetime as dt
import html
import re
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SITE = "https://xxlilloxx-cyber.github.io/federico-lolli-ai-research/"
REDIRECTS = {
    "projects/01-quadratic-gpt2/formulation.html": "projects/01-quadratic-gpt2/method.html",
    "projects/01-quadratic-gpt2/methods.html": "projects/01-quadratic-gpt2/experiments.html",
    "projects/01-quadratic-gpt2/reproduce.html": "projects/01-quadratic-gpt2/reproducibility.html",
}
PDFS = (
    "data/Federico_Lolli_Symmetric_Quadratic_Adaptation_GPT2.pdf",
    "data/Federico_Lolli_Stability_Plasticity_Control.pdf",
)


def public_url(relative: str) -> str:
    if relative == "index.html":
        return SITE
    if relative.endswith("/index.html"):
        return urljoin(SITE, relative.removesuffix("index.html"))
    return urljoin(SITE, relative)


def add_metadata(path: Path) -> None:
    relative = path.relative_to(DOCS).as_posix()
    text = path.read_text(encoding="utf-8")
    title_match = re.search(r"<title>(.*?)</title>", text, flags=re.I | re.S)
    description_match = re.search(r'<meta\s+name="description"\s+content="([^"]*)"\s*/?>', text, flags=re.I)
    title = html.unescape(title_match.group(1).strip()) if title_match else "Federico Lolli — AI Research"
    description = html.unescape(description_match.group(1).strip()) if description_match else "Federico Lolli AI research archive."
    canonical = public_url(REDIRECTS.get(relative, relative))
    robots = "noindex,follow" if relative in REDIRECTS else "index,follow,max-image-preview:large"

    patterns = (
        r'<link\s+rel="canonical"\s+href="[^"]*"\s*/?>',
        r'<meta\s+name="robots"\s+content="[^"]*"\s*/?>',
        r'<meta\s+property="og:[^"]+"\s+content="[^"]*"\s*/?>',
        r'<meta\s+name="twitter:card"\s+content="[^"]*"\s*/?>',
    )
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.I)
    metadata = (
        f'<link rel="canonical" href="{html.escape(canonical, quote=True)}">'
        f'<meta name="robots" content="{robots}">'
        '<meta property="og:type" content="website">'
        f'<meta property="og:title" content="{html.escape(title, quote=True)}">'
        f'<meta property="og:description" content="{html.escape(description, quote=True)}">'
        f'<meta property="og:url" content="{html.escape(canonical, quote=True)}">'
        '<meta name="twitter:card" content="summary">'
    )
    marker = "</head>" if "</head>" in text else "<title>"
    if marker == "</head>":
        text = text.replace(marker, metadata + marker, 1)
    elif marker in text:
        text = text.replace(marker, metadata + marker, 1)
    else:
        raise ValueError(f"No HTML head marker in {path}")
    path.write_text(text, encoding="utf-8")


def build(lastmod: str) -> None:
    pages = sorted(DOCS.rglob("*.html"))
    for page in pages:
        add_metadata(page)
    canonical_pages = [p.relative_to(DOCS).as_posix() for p in pages if p.relative_to(DOCS).as_posix() not in REDIRECTS]
    resources = canonical_pages + [pdf for pdf in PDFS if (DOCS / pdf).is_file()]
    entries = "".join(
        f"  <url><loc>{html.escape(public_url(item))}</loc><lastmod>{lastmod}</lastmod></url>\n"
        for item in resources
    )
    (DOCS / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{entries}</urlset>\n",
        encoding="utf-8",
    )
    (DOCS / "robots.txt").write_text(
        "User-agent: *\nAllow: /\n\n" f"Sitemap: {urljoin(SITE, 'sitemap.xml')}\n",
        encoding="utf-8",
    )
    print(f"built search metadata for {len(canonical_pages)} pages and {len(resources) - len(canonical_pages)} PDFs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lastmod", default=dt.date.today().isoformat())
    args = parser.parse_args()
    build(args.lastmod)


if __name__ == "__main__":
    main()
