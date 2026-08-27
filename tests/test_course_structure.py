import json
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
    ROOT / "seminars/S01-first-classifier.qmd",
    ROOT / "seminars/S02-sensor-data-pipeline.qmd",
    ROOT / "seminars/S03-honest-baseline.qmd",
]
LIVE_CODING_NOTEBOOKS = [
    ROOT / "starter/notebooks/S01-live-coding.ipynb",
    ROOT / "starter/notebooks/S02-live-coding.ipynb",
    ROOT / "starter/notebooks/S03-live-coding.ipynb",
]
READY_LESSONS = LECTURES + SEMINARS


def content_slides(path: Path) -> list[str]:
    return re.split(r"(?m)^## ", path.read_text())[1:]


def visible_lesson_text(path: Path) -> str:
    return re.sub(r"::: \{\.notes\}.*?\n:::", "", path.read_text(), flags=re.DOTALL)


@pytest.mark.parametrize("path", READY_LESSONS)
def test_ready_lesson_has_contiguous_90_minute_route(path: Path) -> None:
    pattern = re.compile(r"(?<!\d)(\d+):(\d\d)–(\d+):(\d\d)")
    intervals: list[tuple[int, int]] = []
    for match in pattern.finditer(path.read_text()):
        start_h, start_m, end_h, end_m = map(int, match.groups())
        intervals.append((start_h * 60 + start_m, end_h * 60 + end_m))

    assert intervals
    assert intervals[0][0] == 0
    assert intervals[-1][1] == 90
    assert all(end == next_start for (_, end), (next_start, _) in pairwise(intervals))
    assert all(start < end for start, end in intervals)


@pytest.mark.parametrize("path", READY_LESSONS)
def test_ready_lesson_has_only_broad_clipping_safeguard(path: Path) -> None:
    """Catch likely rendering failures without treating word counts as pedagogy."""

    for block in content_slides(path):
        title = block.splitlines()[0]
        visible = block.split("::: {.notes}", 1)[0]
        visible = re.sub(r"```.*?```", "", visible, flags=re.DOTALL)
        visible = re.sub(r"!\[[^]]*\]\([^)]*\)", "", visible)
        words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", visible)
        nonempty_lines = [line for line in visible.splitlines()[1:] if line.strip()]

        assert len(words) <= 190, f"{path.name}: {title}: likely clipped ({len(words)} words)"
        assert len(nonempty_lines) <= 32, (
            f"{path.name}: {title}: likely clipped ({len(nonempty_lines)} lines)"
        )


@pytest.mark.parametrize("path", READY_LESSONS)
def test_ready_lesson_has_a_substantial_visual_layer(path: Path) -> None:
    text = path.read_text()
    source_images = len(re.findall(r"!\[[^]]*\]\(", text))
    reproducible_plots = len(re.findall(r"(?m)^lf\.[a-z_]+\(", text))

    assert source_images + reproducible_plots >= 7, (
        f"{path.name}: only {source_images} source images and "
        f"{reproducible_plots} reproducible plots"
    )


@pytest.mark.parametrize("path", LECTURES)
def test_lecture_follows_concept_introduction_cycle(path: Path) -> None:
    text = path.read_text().lower()

    for marker in [
        "инженерная задача",
        "результат лекции",
        "интуиц",
        "определение",
        "формализац",
        "разобранный пример",
        "самостоятельная практика",
    ]:
        assert marker in text, f"{path.name}: missing {marker}"
    assert text.count("контрольный разбор") >= 2


@pytest.mark.parametrize("path", READY_LESSONS)
def test_ready_lessons_do_not_prompt_the_audience_with_questions(path: Path) -> None:
    text = path.read_text()
    notes = "\n".join(re.findall(r"::: \{\.notes\}(.*?)\n:::", text, re.DOTALL))
    prose = re.sub(r"https?://\S+", "", text)

    assert "?" not in prose, f"{path.name}: audience-facing question remains"
    assert "проверка понимания" not in text.lower()
    for prompt in ["спросить", "попросить", "опросить", "ответ аудитории"]:
        assert prompt not in notes.lower(), f"{path.name}: note still asks for {prompt}"


@pytest.mark.parametrize("path", SEMINARS)
def test_seminar_states_prerequisites_and_result(path: Path) -> None:
    text = path.read_text().lower()
    notes = "\n".join(re.findall(r"::: \{\.notes\}(.*?)\n:::", text, re.DOTALL))

    assert "входные знания" in text
    assert "что построим" in text
    assert "notebook" in text
    assert "свернуть презентац" in notes
    assert "вернуться к слайд" in notes
    assert "преподаватель" not in text
    assert "студент" not in text


@pytest.mark.parametrize("path", SEMINARS)
def test_visible_seminar_slides_contain_no_stage_directions(path: Path) -> None:
    visible = visible_lesson_text(path).lower()

    for phrase in [
        "преподаватель",
        "студент",
        "синхронно",
        "свернуть презентац",
        "вернуться к слайд",
        "перед notebook",
        "после notebook",
        "каркас",
        "результат занятия",
    ]:
        assert phrase not in visible, f"{path.name}: stage direction remains: {phrase}"


@pytest.mark.parametrize("path", SEMINARS)
def test_seminars_do_not_assign_independent_or_fill_in_work(path: Path) -> None:
    text = path.read_text().lower()

    for prompt in [
        "## самостоятель",
        "работать индивидуально",
        "минут индивидуально",
        "заполните",
        "впишите",
        "выберите один эксперимент",
        "создайте `tests",
        "реализуйте `",
        "добавьте проверки",
        "сохраните исходный",
        "нарисуйте без кода",
    ]:
        assert prompt not in text, f"{path.name}: independent prompt remains: {prompt}"
    assert "|  |" not in text, f"{path.name}: blank fill-in table remains on a slide"


def test_live_coding_notebooks_are_empty_scaffolds_with_ordered_blocks() -> None:
    quarto = (ROOT / "_quarto.yml").read_text()

    for expected_blocks, path in zip([5, 4, 5], LIVE_CODING_NOTEBOOKS, strict=True):
        notebook = json.loads(path.read_text())
        markdown = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        ).lower()
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

        assert markdown.count("\n## ") == expected_blocks
        assert len(code_cells) == expected_blocks
        assert all(cell["execution_count"] is None for cell in code_cells)
        assert all(not cell["outputs"] for cell in code_cells)
        assert all(not cell["source"] for cell in code_cells)
        assert "преподаватель" not in path.read_text().lower()
        assert "студент" not in path.read_text().lower()
        assert "живой кодинг" not in path.read_text().lower()
        assert f"starter/notebooks/{path.name}" in quarto


def test_seminars_do_not_end_with_acceptance_slide() -> None:
    for path in SEMINARS:
        headings = re.findall(r"(?m)^## (.+)$", path.read_text())
        assert headings
        assert headings[-1].strip().lower() != "приёмка"


def test_teacher_scripts_are_excluded_from_the_student_site() -> None:
    quarto = (ROOT / "_quarto.yml").read_text()
    gitignore = (ROOT / ".gitignore").read_text()
    lecture_index = (ROOT / "lectures/index.qmd").read_text()
    seminar_index = (ROOT / "seminars/index.qmd").read_text()

    assert "!lectures/L00-speaker-script.md" in quarto
    assert "/lectures/L00-speaker-script.md" in gitignore
    assert "L00-speaker-script" not in lecture_index
    for seminar_number in range(1, 4):
        script = f"S0{seminar_number}-speaker-script.md"
        assert f"!seminars/{script}" in quarto
        assert f"/seminars/{script}" in gitignore
        assert script.removesuffix(".md") not in seminar_index


def test_first_module_defines_core_ml_vocabulary_before_later_lessons() -> None:
    l0 = (ROOT / "lectures/L00-engineering-ml.qmd").read_text().lower()
    l1 = (ROOT / "lectures/L01-supervised-validation.qmd").read_text().lower()
    s1 = (ROOT / "seminars/S01-first-classifier.qmd").read_text().lower()

    for term in [
        "машинное обучение",
        "**объект**",
        "**наблюдение**",
        "обучающий пример",
        "выборка",
        "размер выборки",
        "признак",
        "признаковый вектор",
        "матрица «объекты–признаки»",
        "пространство объектов",
        "пространство ответов",
        "обучение с учителем",
        "бинарная классификация",
        "многоклассовая классификация",
        "регрессия",
        "семейство",
        "параметр",
        "гиперпараметр",
        "алгоритм обучения",
        "обученная модель",
        "вывод (`inference`)",
        "оценка модели",
        "функция потерь",
        "эмпирический риск",
        "базовая модель",
        "обобщающая способность",
        "переобучение",
    ]:
        assert term in l0, f"L0 does not define {term}"

    for term in [
        "единица решения",
        "зависимая группа",
        "объект переноса",
        "разбиение",
        "**train**",
        "**validation**",
        "**test**",
        "утечка данных",
        "агрегация",
        "**score**",
        "калибровка",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "average precision",
        "порог",
    ]:
        assert term in l1, f"L1 does not define {term}"

    for term in ["fit", "score", "accuracy", "ложная тревога", "пропуск"]:
        assert term in s1, f"S1 does not apply {term}"


def test_student_site_calls_practical_classes_seminars() -> None:
    quarto = (ROOT / "_quarto.yml").read_text()
    public_pages = [
        ROOT / "index.qmd",
        ROOT / "syllabus.qmd",
        ROOT / "seminars/index.qmd",
        ROOT / "project/index.qmd",
    ]
    public_text = "\n".join(path.read_text() for path in public_pages)

    assert 'text: "Семинары"' in quarto
    assert "Лаборатории" not in quarto
    assert "Лаборатории" not in public_text


def test_score_is_not_called_a_flight_probability() -> None:
    sources = [
        ROOT / "assets/course_case.py",
        ROOT / "starter/src/ml_sau/baseline.py",
        ROOT / "lectures/L01-supervised-validation.qmd",
        ROOT / "seminars/S03-honest-baseline.qmd",
    ]
    stale_terms = ["aggregate_flight_predictions", "max_window_probability", "flight_probability"]

    for path in sources:
        text = path.read_text()
        for term in stale_terms:
            assert term not in text, f"{path.name}: stale term {term}"


def test_split_manifest_names_dependence_time_and_deployment_axes() -> None:
    sources = [
        ROOT / "lectures/L01-supervised-validation.qmd",
        ROOT / "seminars/S03-honest-baseline.qmd",
        ROOT / "starter/src/ml_sau/baseline.py",
    ]
    required = [
        "decision_unit",
        "dependence_group",
        "validation_policy",
        "test_policy",
        "deployment_group",
        "time_order_key",
    ]

    for path in sources:
        text = path.read_text()
        assert all(field in text for field in required), path.name


def test_starter_leaves_only_two_focused_functions_per_implementation_lab() -> None:
    sensor = (ROOT / "starter/src/ml_sau/sensor.py").read_text()
    baseline = (ROOT / "starter/src/ml_sau/baseline.py").read_text()

    assert sensor.count("NotImplementedError") == 2
    assert baseline.count("NotImplementedError") == 2
    assert "def write_quality_report" in sensor
    assert "def run_baseline" in baseline
    assert "evaluate_test" not in baseline
    assert "--evaluate-test" not in baseline


def test_course_has_mandatory_entry_gate() -> None:
    prerequisites = " ".join((ROOT / "prerequisites.qmd").read_text().lower().split())
    syllabus = " ".join((ROOT / "syllabus.qmd").read_text().lower().split())

    assert "обязательным допуском к s1" in prerequisites
    assert "все четыре пункта" in prerequisites
    assert "число повторных попыток не ограничено" in prerequisites
    assert "четыре отметки `passed`" in syllabus


def test_project_decisions_are_fixed_in_versioned_protocols() -> None:
    datasets = (ROOT / "project/_approved-datasets.md").read_text()
    gate = (ROOT / "project/_test-gate-protocol.md").read_text()
    platform = (ROOT / "project/_target-platform.md").read_text()

    for dataset_id in ["dataset/447", "dataset/791", "dataset/551", "dataset/240"]:
        assert dataset_id in datasets
    assert "Test-A" in gate and "Test-B" in gate
    assert "S12" in gate and "S14" in gate
    assert "20 MiB" in platform
    assert "256 MiB" in platform
    assert "p95 ≤ 50 ms" in platform


def test_schedule_preserves_the_new_dependency_order() -> None:
    home = (ROOT / "index.qmd").read_text()

    for expected in [
        "S1 · Первый классификатор",
        "S4 · Карточка проекта, репозиторий и CI",
        "S6 · Диагностика и регуляризация MLP",
        "S7 · 1D-CNN и сравнение с базовой моделью",
        "L4 · Семейства задач, метрики и границы применения",
        "L5 · Гибридная оценка, мониторинг и безопасный резерв",
        "L6 · Инференс и эксплуатация на целевой платформе",
    ]:
        assert expected in home


def test_calendar_uses_denominator_fridays_and_pair_times() -> None:
    home = (ROOT / "index.qmd").read_text()

    for date in [
        "11 сентября 2026",
        "25 сентября 2026",
        "9 октября 2026",
        "23 октября 2026",
        "6 ноября 2026",
        "20 ноября 2026",
        "4 декабря 2026",
        "18 декабря 2026",
    ]:
        assert date in home

    for pair_time in ["15:55–17:25", "17:35–19:05", "19:15–20:45"]:
        assert pair_time in home

    assert "1 сентября 2026 года, во вторник" in home
    assert "1 января 2027" not in home
    assert "Дата переноса уточняется" not in home
    assert "нового материала в этот день нет" in home


def test_future_lessons_are_marked_as_planned() -> None:
    home = (ROOT / "index.qmd").read_text()

    for lesson in ["L2", "L3", "L4", "L5", "L6", "S4", "S15"]:
        schedule_rows = [
            line for line in home.splitlines() if line.startswith("|") and f"{lesson} ·" in line
        ]
        assert schedule_rows
        assert all("запланировано" in line for line in schedule_rows)

    assert "L7 ·" not in home
    assert "L8 ·" not in home
    assert "S16 ·" not in home
    assert "S17 ·" not in home


def test_removed_flight_simulator_cannot_return_to_the_site() -> None:
    removed_paths = [
        ROOT / "interactive",
        ROOT / "assets/flight-mission.css",
        ROOT / "assets/flight-mission.mjs",
        ROOT / "assets/flight-mission-core.mjs",
        ROOT / "assets/models/nasa-global-hawk.glb",
        ROOT / "tests/flight-mission-core.test.mjs",
    ]
    assert not any(path.exists() for path in removed_paths)

    navigation = "\n".join(
        (ROOT / path).read_text()
        for path in ["_quarto.yml", "index.qmd", "syllabus.qmd", "lectures/index.qmd"]
    ).lower()
    assert "flight-mission" not in navigation
    assert "интерактивная лаборатория" not in navigation
    assert "лётная лаборатория" not in navigation


def test_live_loading_indicator_stays_hidden() -> None:
    theme = (ROOT / "assets/theme/slides.scss").read_text()

    assert ".reveal #exercise-loading-indicator.d-none { display: none !important; }" in theme
