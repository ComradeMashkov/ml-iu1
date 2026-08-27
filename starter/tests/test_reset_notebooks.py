import json
import subprocess
import sys
from pathlib import Path

STARTER_ROOT = Path(__file__).resolve().parents[1]


def test_reset_notebooks_clears_only_code_cells(tmp_path: Path) -> None:
    notebook_path = tmp_path / "seminar.ipynb"
    original_markdown = ["# Семинар\n", "Теория остаётся на месте."]
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"keep": True},
                "source": ["## Первый блок\n", *original_markdown],
            },
            {
                "cell_type": "code",
                "execution_count": 7,
                "metadata": {"keep": True},
                "outputs": [{"output_type": "stream", "text": ["42\n"]}],
                "source": ["answer = 40 + 2\n", "print(answer)"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Второй блок"],
            },
        ],
        "metadata": {"kernelspec": {"name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(STARTER_ROOT / "tools" / "reset_notebooks.py"),
            "--notebooks-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    reset = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert reset["cells"][0]["source"] == ["## Первый блок\n", *original_markdown]
    assert reset["cells"][0]["metadata"] == {"keep": True}
    assert reset["cells"][1]["source"] == []
    assert reset["cells"][1]["execution_count"] is None
    assert reset["cells"][1]["outputs"] == []
    assert reset["cells"][1]["metadata"] == {"keep": True}
    assert reset["cells"][2]["source"] == ["## Второй блок"]
    assert reset["cells"][3]["cell_type"] == "code"
    assert reset["cells"][3]["source"] == []
    assert reset["cells"][3]["execution_count"] is None
    assert reset["cells"][3]["outputs"] == []
    assert len(reset["cells"]) == 4
    assert reset["metadata"] == notebook["metadata"]
    assert "Reset 2 code cells in 1 notebooks" in completed.stdout
    assert "removed 1 extra and added 1 missing code cells" in completed.stdout
