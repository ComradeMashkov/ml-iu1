import re
from itertools import pairwise
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]

LECTURES = [
    ROOT / "lectures/L00-engineering-ml.qmd",
    ROOT / "lectures/L01-supervised-validation.qmd",
]

SEMINARS = [
    ROOT / "seminars/S01-engineering-toolchain.qmd",
    ROOT / "seminars/S02-sensor-data-pipeline.qmd",
    ROOT / "seminars/S03-honest-baseline.qmd",
]


@pytest.mark.parametrize(
    ("path", "limit"),
    [(path, 32) for path in LECTURES] + [(path, 20) for path in SEMINARS],
)
def test_ready_lesson_respects_slide_budget(path: Path, limit: int) -> None:
    content_slides = sum(line.startswith("## ") for line in path.read_text().splitlines())

    assert content_slides <= limit


@pytest.mark.parametrize("path", LECTURES + SEMINARS)
def test_ready_lesson_has_non_overlapping_90_minute_timing(path: Path) -> None:
    pattern = re.compile(r"(?<!\d)(\d+):(\d\d)–(\d+):(\d\d)")
    intervals: list[tuple[int, int]] = []
    for match in pattern.finditer(path.read_text()):
        start_h, start_m, end_h, end_m = map(int, match.groups())
        intervals.append((start_h * 60 + start_m, end_h * 60 + end_m))

    assert intervals
    assert intervals[0][0] == 0
    assert max(end for _, end in intervals) == 90
    assert all(
        current_end <= next_start for (_, current_end), (next_start, _) in pairwise(intervals)
    )


@pytest.mark.parametrize("path", LECTURES + SEMINARS)
def test_ready_lesson_has_no_overloaded_text_slide(path: Path) -> None:
    blocks = re.split(r"(?m)^## ", path.read_text())[1:]

    for block in blocks:
        title = block.splitlines()[0]
        visible = block.split("::: {.notes}", 1)[0]
        visible = re.sub(r"```.*?```", "", visible, flags=re.DOTALL)
        visible = re.sub(r"!\[[^]]*\]\([^)]*\)", "", visible)
        words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", visible)
        nonempty_lines = [line for line in visible.splitlines()[1:] if line.strip()]

        assert len(words) <= 115, f"{title}: {len(words)} words"
        assert len(nonempty_lines) <= 20, f"{title}: {len(nonempty_lines)} non-empty lines"


def test_future_lessons_are_marked_as_planned() -> None:
    home = (ROOT / "index.qmd").read_text()

    for lesson in ["L2", "L3", "L4", "L5", "L6", "L7", "L8", "S4", "S17"]:
        schedule_rows = [
            line for line in home.splitlines() if line.startswith("|") and f"{lesson} ·" in line
        ]
        assert schedule_rows
        assert all("запланировано" in line for line in schedule_rows)
