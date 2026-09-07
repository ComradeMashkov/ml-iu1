"""S1 figures drawn only from the reproducible six-versus-five feature experiment."""

import json
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from figstyle import ACCENT, ACCENT2, AMBER, GREEN, INK

STARTER_ROOT = Path(__file__).resolve().parents[1] / "starter"
sys.path.insert(0, str(STARTER_ROOT / "src"))

from ml_sau.first_model import calculate_first_model


@lru_cache(maxsize=1)
def demo_report() -> dict[str, object]:
    settings = json.loads((STARTER_ROOT / "configs/first-model.json").read_text())
    return calculate_first_model(settings)


def sorted_flight_scores() -> plt.Figure:
    report = demo_report()
    rows = sorted(report["flight_results"], key=lambda row: row["score"])
    scores = np.array([row["score"] for row in rows])
    target = np.array([row["target"] for row in rows])
    positions = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    for label, color, marker in ((0, ACCENT, "o"), (1, AMBER, "^")):
        mask = target == label
        ax.scatter(
            positions[mask],
            scores[mask],
            color=color,
            marker=marker,
            s=45,
            label=f"историческая метка {label}",
        )
    ax.axhline(report["demonstration_threshold"], color=ACCENT2, linestyle="--", label="порог 0.5")
    ax.set(
        xlabel="валидационные полёты по возрастанию score",
        ylabel="максимум score окон",
        ylim=(0, 1),
        xlim=(-1, len(rows)),
    )
    ax.legend(fontsize=10, loc="upper left")
    return fig


def feature_comparison() -> plt.Figure:
    report = demo_report()
    values = [report["demonstration_accuracy"], report["feature_comparison"]["reduced_accuracy"]]
    fig, ax = plt.subplots(figsize=(8.0, 3.1))
    ax.barh([1, 0], values, color=[ACCENT, GREEN], height=0.5)
    ax.set_yticks([1, 0], ["6 признаков", "5: без температуры"])
    for position, value in zip([1, 0], values, strict=True):
        correct = round(value * report["n_validation_flights"])
        text_x = max(value + 0.018, report["always_zero_accuracy"] + 0.03)
        ax.text(text_x, position, f"{correct}/36 = {value:.3f}", va="center", fontsize=12)
    ax.axvline(
        report["always_zero_accuracy"],
        color=INK,
        linestyle="--",
        linewidth=1.3,
        label="постоянный ответ 0: 25/36",
    )
    ax.set(xlabel="accuracy по тем же 36 полётам при пороге 0.5", xlim=(0, 1.04), ylim=(-0.6, 1.6))
    ax.legend(loc="lower right", fontsize=10)
    return fig


def actual_error_profiles() -> plt.Figure:
    examples = demo_report()["inspected_examples"]
    labels = ["ошибка слежения", "ток", "температура", "вибрация", "нагрузка", "напряжение"]
    fig, ax = plt.subplots(figsize=(8.1, 4.1))
    positions = np.arange(len(labels))
    for shift, example, color in zip([-0.17, 0.17], examples, [AMBER, ACCENT], strict=True):
        ax.barh(
            positions + shift,
            example["standardized_window_features"],
            height=0.3,
            color=color,
            label=f"{example['result']}: {example['flight_id']}",
        )
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set(xlabel="(значение окна − среднее train) / масштаб train", xlim=(-2, 3.2))
    ax.legend(fontsize=10, loc="lower right")
    return fig
