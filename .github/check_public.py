"""Check the student repository and the exact GitHub Pages artifact."""

import argparse
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RULES = [
    r"(?:\.gitignore|README\.md|_quarto\.yml)",
    r"(?:index|syllabus|course-map|resources)\.qmd",
    r"\.github/(?:check_public\.py|workflows/publish\.yml)",
    r"lectures/(?:index|L00-engineering-ml|L01-how-models-learn)\.qmd",
    r"seminars/(?:index|S01-first-classifier|S02-sensor-data-pipeline|S03-gradient-descent|S04-model-evaluation)\.qmd",
    r"project/_[a-z-]+\.md",
    r"assets/(?:logo\.svg|references\.bib|next-class\.js|slides-layout\.js)",
    r"assets/theme/[a-z0-9-]+\.scss",
    r"assets/generated/(?:intro|sep25)/[a-z0-9-]+\.svg",
    r"assets/media/gofman-algorithm-meme\.png",
    r"assets/vendor/katex/(?:LICENSE|README\.md|katex\.min\.(?:js|css)|fonts/[\w-]+\.(?:woff2?|ttf))",
    r"starter/(?:\.gitignore|Makefile|README\.md|pyproject\.toml|uv\.lock)",
    r"starter/\.github/workflows/check\.yml",
    r"starter/(?:configs/[\w-]+\.json|data/[\w-]+\.(?:csv|tsv|txt|md))",
    r"starter/notebooks/(?:S0[1-4]-live-coding|solutions/S0[1-4]-complete)\.ipynb",
    r"starter/(?:lessons/s0[1-4]\.py|src/ml_sau/[\w]+\.py|tests/test_[\w]+\.py|tools/reset_notebooks\.py)",
]
PRIVATE_CONTENT = re.compile(
    r"(?m)^\s*:{3,}\s*\{[^}\n]*\.notes\b"
    r"|<aside\b[^>]*class=[\"'][^\"']*\bnotes\b"
    r"|/Users/ilamaskov/|codex-clipboard-"
)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attrs):
        self.references.extend(value for name, value in attrs if name in {"href", "src"} and value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path)
    args = parser.parse_args()
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    tracked = [name for name in tracked if name]
    errors = []
    for name in tracked:
        if not any(re.fullmatch(rule, name) for rule in SOURCE_RULES):
            errors.append(f"Source outside student manifest: {name}")
        path = ROOT / name
        if path.suffix in {".qmd", ".md", ".ipynb"} and PRIVATE_CONTENT.search(path.read_text()):
            errors.append(f"Private content in student source: {name}")
    if args.site:
        site = args.site.resolve()
        if not (site / "index.html").is_file():
            errors.append("Rendered site is missing index.html")
        for path in site.rglob("*"):
            if not path.is_file():
                continue
            name = path.relative_to(site).as_posix()
            # Resources are either reviewed sources, rendered pages, or Quarto runtime files.
            runtime = name.startswith("site_libs/")
            rendered_page = name.endswith(".html") and name[:-5] + ".qmd" in tracked
            if not (
                name in tracked
                or runtime
                or rendered_page
                or name in {".nojekyll", "sitemap.xml", "robots.txt", "search.json"}
            ):
                errors.append(f"Unexpected published file: {name}")
            if path.suffix not in {".html", ".md", ".ipynb", ".json"} or runtime:
                continue
            content = path.read_text()
            if PRIVATE_CONTENT.search(content):
                errors.append(f"Private content in published file: {name}")
            if path.suffix != ".html":
                continue
            links = Links()
            links.feed(content)
            for reference in links.references:
                url = urlsplit(reference)
                if url.scheme or url.netloc or not url.path:
                    continue
                target = unquote(url.path)
                if target.startswith("/ml-iu1/"):
                    resolved = site / target[len("/ml-iu1/") :]
                elif target.startswith("/"):
                    resolved = site / target.lstrip("/")
                else:
                    resolved = path.parent / target
                if not resolved.exists():
                    errors.append(f"Broken local link: {name} -> {target}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        f"Student publication check passed: {len(tracked)} source files"
        + ("; Pages artifact and links verified" if args.site else "")
    )


if __name__ == "__main__":
    main()
