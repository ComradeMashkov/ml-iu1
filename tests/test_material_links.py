import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
SKIPPED_DIRECTORIES = {".git", ".quarto", ".venv", "_extensions", "_site"}


def source_files():
    for suffix in ("*.qmd", "*.md", "*.html"):
        for path in ROOT.rglob(suffix):
            if not SKIPPED_DIRECTORIES.intersection(path.parts):
                yield path


def local_references(path: Path):
    text = path.read_text(errors="ignore")
    yield from (match[1] for match in re.findall(r"(!?)\[[^\]]*\]\(([^) >]+)", text))
    yield from re.findall(r"(?:src|href)=[\"']([^\"']+)", text)


def reference_exists(path: Path, reference: str) -> bool:
    clean = reference.split("#", 1)[0].split("?", 1)[0]
    if not clean or clean.startswith(("http:", "https:", "mailto:", "data:", "javascript:")):
        return True

    target = (path.parent / clean).resolve()
    if target.exists():
        return True
    return target.suffix == ".html" and target.with_suffix(".qmd").exists()


def test_local_material_references_exist() -> None:
    missing = [
        f"{path.relative_to(ROOT)} -> {reference}"
        for path in source_files()
        for reference in local_references(path)
        if not reference_exists(path, reference)
    ]

    assert not missing, "Missing local references:\n" + "\n".join(missing)
