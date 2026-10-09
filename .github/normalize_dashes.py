"""Keep Quarto's generated page titles consistent with course punctuation."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path)
    args = parser.parse_args()
    output = args.site or Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", "_site"))
    if not output.is_absolute():
        output = ROOT / output
    count = 0
    for path in output.rglob("*.html"):
        if "site_libs" in path.relative_to(output).parts:
            continue
        original = path.read_text(encoding="utf-8")
        normalized = original.replace("\u2014", "-").replace("\u2013", "-")
        if normalized != original:
            path.write_text(normalized, encoding="utf-8")
            count += 1
    print(f"Course punctuation normalized in {count} rendered pages")


if __name__ == "__main__":
    main()
