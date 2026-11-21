"""Build docs/report.pdf from docs/report.md (A4, figures and tables with captions).

    pip install markdown playwright && playwright install chromium
    python docs/build_report.py                       # -> docs/report.pdf
    python docs/build_report.py --subtitle "Organisation · Programme" --out copy.pdf

For a copy handed in elsewhere, --subtitle replaces "Personal project" in the title block and
--insert adds a Markdown file before the References section. Fonts are the page's own
(app/web/fonts, SIL Open Font License).
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

import markdown

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent

CSS = """
@page { size: A4; margin: 17mm 18mm 18mm 18mm; }
:root { --ink: #1d2420; --muted: #5b655f; --rule: #c9cfc8; --accent: #2F6B3F; }
html { font-family: 'Source Serif 4', Georgia, serif; font-size: 9.9pt; line-height: 1.42;
       color: var(--ink); font-variant-numeric: lining-nums; }
body { margin: 0; hyphens: auto; -webkit-hyphens: auto; text-align: justify; }
h1 { font: 600 16.5pt/1.22 'IBM Plex Sans', sans-serif; margin: 0 0 6pt; text-align: left;
     letter-spacing: -0.01em; hyphens: none; }
h1 + p { font: 400 8.9pt/1.5 'IBM Plex Sans', sans-serif; color: var(--muted); text-align: left;
         margin: 0 0 10pt; padding-bottom: 8pt; border-bottom: 0.6pt solid var(--rule); }
h1 + p strong { color: var(--ink); font-weight: 600; }
h2 { font: 600 11pt/1.3 'IBM Plex Sans', sans-serif; margin: 12pt 0 4pt; text-align: left;
     break-after: avoid; }
h3 { font: 600 9.9pt/1.3 'IBM Plex Sans', sans-serif; margin: 9pt 0 3pt; break-after: avoid; }
p { margin: 0 0 5pt; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 6pt; padding-left: 14pt; }
li { margin: 0 0 2.5pt; }
a { color: inherit; text-decoration: none; }
code { font: 400 8.4pt 'IBM Plex Mono', monospace; }
em { font-style: italic; }
figure { margin: 7pt 0 9pt; break-inside: avoid; }
figure img { display: block; width: 100%; height: auto; }
figure.narrow img { width: 62%; margin: 0 auto; }
figcaption { font: 400 8.2pt/1.4 'IBM Plex Sans', sans-serif; color: var(--muted);
             margin-top: 4pt; text-align: left; }
figcaption strong { color: var(--ink); font-weight: 600; }
table { width: 100%; border-collapse: collapse; font: 400 8.1pt/1.35 'IBM Plex Sans', sans-serif;
        font-variant-numeric: tabular-nums; }
th { font-weight: 600; text-align: left; border-bottom: 0.8pt solid var(--ink);
     padding: 2.5pt 4pt; }
td { border-bottom: 0.4pt solid var(--rule); padding: 2.2pt 4pt; vertical-align: top; }
th[style*="right"], td[style*="right"] { text-align: right; }
tbody tr:last-child td { border-bottom: 0.8pt solid var(--ink); }
.abstract { margin: 0 0 4pt; }
h2#references ~ p { font-size: 8.4pt; line-height: 1.38; margin-bottom: 3pt; text-align: left;
                    padding-left: 16pt; text-indent: -16pt; }
"""


def fonts_css() -> str:
    css = (ROOT / "app" / "web" / "fonts.css").read_text(encoding="utf-8")
    return css.replace("url(fonts/", f"url({(ROOT / 'app' / 'web' / 'fonts').as_uri()}/")


def to_html(md_text: str) -> str:
    body = markdown.markdown(md_text, extensions=["tables", "attr_list", "toc"])
    # local images: absolute file URIs so the browser finds them from a temporary file
    body = re.sub(
        r'src="(\.\./[^"]+)"',
        lambda m: f'src="{(DOCS / m.group(1)).resolve().as_uri()}"',
        body,
    )
    # an image paragraph followed by its "Figure N." paragraph -> <figure> with <figcaption>
    body = re.sub(
        r"<p>(<img [^>]+>)</p>\s*<p>(<strong>Figure \d+\.</strong>.*?)</p>",
        lambda m: f"<figure>{m.group(1)}<figcaption>{m.group(2)}</figcaption></figure>",
        body,
        flags=re.S,
    )
    # a table followed by its "Table N." paragraph -> keep together, caption below
    body = re.sub(
        r"(<table>.*?</table>)\s*<p>(<strong>Table \d+\.</strong>.*?)</p>",
        lambda m: f"<figure>{m.group(1)}<figcaption>{m.group(2)}</figcaption></figure>",
        body,
        flags=re.S,
    )
    # the histogram is printed narrower than the full-width figures
    return body.replace(
        '<figure><img alt="Grad-CAM mass on the leaf"',
        '<figure class="narrow"><img alt="Grad-CAM mass on the leaf"',
    )


def build(source: Path, out: Path, insert: Path | None = None, subtitle: str | None = None) -> None:
    from playwright.sync_api import sync_playwright

    text = source.read_text(encoding="utf-8")
    if insert is not None:
        extra = insert.read_text(encoding="utf-8").strip() + "\n\n"
        text = text.replace("\n## References\n", "\n" + extra + "## References\n", 1)
    if subtitle:
        old = "Personal project · Technical report,"
        text = text.replace(old, f"{subtitle} · Technical report,", 1)
    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Plant Disease Detection: technical report (Anar Hummatli)</title>"
        f"<style>{fonts_css()}\n{CSS}</style></head><body>{to_html(text)}</body></html>"
    )
    footer = (
        "<div style=\"width:100%;font:7.5pt 'IBM Plex Sans',sans-serif;color:#5b655f;"
        'padding:0 18mm;display:flex;justify-content:space-between">'
        "<span>Plant disease detection · Anar Hummatli</span>"
        '<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>'
    )
    with tempfile.TemporaryDirectory() as tmp:
        page_file = Path(tmp) / "report.html"
        page_file.write_text(html, encoding="utf-8")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(page_file.as_uri(), wait_until="networkidle")
            page.evaluate("document.fonts.ready")
            page.pdf(
                path=str(out),
                format="A4",
                print_background=True,
                display_header_footer=True,
                header_template="<span></span>",
                footer_template=footer,
                margin={"top": "17mm", "bottom": "18mm", "left": "18mm", "right": "18mm"},
            )
            browser.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DOCS / "report.md")
    parser.add_argument("--out", type=Path, default=DOCS / "report.pdf")
    parser.add_argument("--insert", type=Path, help="Markdown inserted before the References")
    parser.add_argument("--subtitle", help='replaces "Personal project" in the title block')
    args = parser.parse_args(argv)
    build(args.source, args.out, args.insert, args.subtitle)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
