"""Execute the exact instructor insertions through the distributed notebooks."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "starter"


def completed_cells(seminar: str) -> list[str]:
    script_path = ROOT / "seminars" / f"{seminar}-speaker-script.md"
    if not script_path.exists():
        pytest.skip("instructor scripts are local and excluded from the public repository")
    script = script_path.read_text()
    insertions = {
        int(number): code
        for number, code in re.findall(
            r"<!-- notebook-solution: (\d+) -->\s*```python\n(.*?)```", script, re.DOTALL
        )
    }
    notebook = json.loads((STARTER / "notebooks" / f"{seminar}-live-coding.ipynb").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert set(insertions) == set(range(1, len(cells) + 1))
    result = []
    for number, cell in enumerate(cells, start=1):
        source = "".join(cell["source"])
        placeholder = r"(?m)^raise NotImplementedError\([^\n]*\)[ \t]*$"
        source, count = re.subn(
            placeholder, lambda _, replacement=insertions[number]: replacement.rstrip(), source
        )
        assert count == 1, f"{seminar} block {number}: expected one insertion point"
        compile(source, f"{seminar} block {number}", "exec")
        result.append(source)
    return result


@pytest.mark.parametrize("seminar", ["S01", "S02"])
@pytest.mark.parametrize("working_directory", [".", "notebooks"])
def test_instructor_notebook_matches_cli_from_both_start_directories(
    seminar: str, working_directory: str, tmp_path: Path
) -> None:
    cells = completed_cells(seminar)
    project = tmp_path / "starter"
    for directory in ("src", "configs", "notebooks"):
        shutil.copytree(
            STARTER / directory,
            project / directory,
            ignore=shutil.ignore_patterns("reports", "__pycache__", ".ipynb_checkpoints"),
        )
    shutil.copy2(STARTER / "pyproject.toml", project / "pyproject.toml")
    # Use this copy of the package, independent of an editable local installation.
    environment = {**os.environ, "PYTHONPATH": str(project / "src")}
    notebook_program = "\n\n".join(cells)
    notebook_run = subprocess.run(
        [sys.executable, "-c", notebook_program],
        cwd=project / working_directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert notebook_run.returncode == 0, notebook_run.stdout + notebook_run.stderr

    module, report_name = {
        "S01": ("ml_sau.first_model", "s1-first-model.json"),
        "S02": ("ml_sau.sensor", "s2-quality.json"),
    }[seminar]
    report_path = project / "reports" / report_name
    notebook_path = project / "reports" / "s1-notebook.json" if seminar == "S01" else report_path
    notebook_report = json.loads(notebook_path.read_text())
    report_path.unlink(missing_ok=True)
    csv_artifacts = {}
    if seminar == "S02":
        for filename in ("s2-features.csv", "s2-windows.csv"):
            path = project / "reports" / filename
            csv_artifacts[filename] = path.read_text()
            path.unlink()
    command_run = subprocess.run(
        [sys.executable, "-m", module],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert command_run.returncode == 0, command_run.stdout + command_run.stderr
    assert notebook_report == json.loads(report_path.read_text())
    for filename, content in csv_artifacts.items():
        assert content == (project / "reports" / filename).read_text()
    assert not (project / "notebooks" / "reports").exists()
