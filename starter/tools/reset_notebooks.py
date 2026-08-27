"""Return seminar notebooks to their empty-code-cell state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def _cell_text(cell: dict[str, object], path: Path) -> str:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(line, str) for line in source):
        return "".join(source)
    raise TypeError(f"{path}: cell source must be a string or a list of strings")


def _empty_code_cell(section_title: str) -> dict[str, object]:
    cell_id = hashlib.sha256(section_title.encode("utf-8")).hexdigest()[:8]
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": [],
    }


def reset_code_cells(notebook: dict[str, object], path: Path) -> tuple[int, int, int]:
    """Restore one empty code cell after every level-two section heading."""
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise TypeError(f"{path}: notebook must contain a cells list")

    reset_count = 0
    removed_count = 0
    added_count = 0
    restored_cells: list[dict[str, object]] = []
    pending_section: str | None = None

    for cell in cells:
        if not isinstance(cell, dict):
            raise TypeError(f"{path}: every cell must be an object")

        if cell.get("cell_type") == "markdown":
            if pending_section is not None:
                restored_cells.append(_empty_code_cell(pending_section))
                added_count += 1
            restored_cells.append(cell)
            text = _cell_text(cell, path).lstrip()
            pending_section = text.splitlines()[0] if text.startswith("## ") else None
            continue

        if cell.get("cell_type") != "code":
            restored_cells.append(cell)
            continue

        reset_count += 1
        if pending_section is None:
            removed_count += 1
            continue

        cell["source"] = []
        cell["execution_count"] = None
        cell["outputs"] = []
        restored_cells.append(cell)
        pending_section = None

    if pending_section is not None:
        restored_cells.append(_empty_code_cell(pending_section))
        added_count += 1

    notebook["cells"] = restored_cells

    return reset_count, removed_count, added_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clear code and outputs from seminar notebooks, preserving Markdown cells."
    )
    parser.add_argument(
        "--notebooks-dir",
        type=Path,
        default=DEFAULT_NOTEBOOKS_DIR,
        help="directory containing the .ipynb files to reset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook_paths = sorted(args.notebooks_dir.glob("*.ipynb"))
    if not notebook_paths:
        raise SystemExit(f"No .ipynb files found in {args.notebooks_dir}")

    prepared: list[tuple[Path, dict[str, object], int, int, int]] = []
    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(notebook, dict):
            raise TypeError(f"{path}: notebook root must be an object")
        reset_count, removed_count, added_count = reset_code_cells(notebook, path)
        prepared.append((path, notebook, reset_count, removed_count, added_count))

    for path, notebook, _, _, _ in prepared:
        serialized = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
        temporary_path = path.with_suffix(".ipynb.tmp")
        temporary_path.write_text(serialized, encoding="utf-8")
        temporary_path.replace(path)

    total_reset = sum(reset_count for _, _, reset_count, _, _ in prepared)
    total_removed = sum(removed_count for _, _, _, removed_count, _ in prepared)
    total_added = sum(added_count for _, _, _, _, added_count in prepared)
    print(
        f"Reset {total_reset} code cells in {len(prepared)} notebooks; "
        f"removed {total_removed} extra and added {total_added} missing code cells. "
        "Markdown cells were preserved."
    )


if __name__ == "__main__":
    main()
