"""Reproducible explanatory plots used in the first five course sessions.

Photographs and engineering schematics live in ``assets/images`` and come from
the cited primary sources.  This module is intentionally limited to plots whose
geometry or values are derived from the course examples.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from figstyle import ACCENT, ACCENT2, AMBER, GREEN, GRID, INK, MUTED
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from sklearn.metrics import precision_recall_curve


def four_flights() -> plt.Figure:
    """Plot the four-flight example and one possible linear boundary."""

    temperature = np.array([47, 51, 64, 69])
    current = np.array([10.2, 11.0, 14.1, 15.0])
    target = np.array([0, 0, 1, 1])
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    colors = np.where(target == 1, ACCENT2, ACCENT)
    ax.scatter(temperature, current, c=colors, s=110, edgecolor="white", linewidth=1.5)
    for index, (x_value, y_value) in enumerate(zip(temperature, current, strict=True), 1):
        ax.annotate(f"F{index:02d}", (x_value, y_value), xytext=(7, 4), textcoords="offset points")
    boundary_x = np.array([50, 67])
    ax.plot(boundary_x, 21.5 - 0.14 * boundary_x, "--", color=MUTED, label="одно из правил")
    ax.set(xlabel="средняя температура, °C", ylabel="RMS тока, A")
    ax.legend(loc="upper left")
    return fig


def sigmoid_score() -> plt.Figure:
    """Show the mapping from a linear score to the unit interval."""

    z_value = np.linspace(-6, 6, 300)
    score = 1 / (1 + np.exp(-z_value))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(z_value, score, color=ACCENT)
    ax.axhline(0.5, color=MUTED, linestyle="--")
    ax.axvline(0, color=MUTED, linestyle="--")
    ax.fill_between(z_value, 0, score, where=z_value >= 0, color=ACCENT, alpha=0.10)
    ax.annotate(
        r"$z=0 \Rightarrow s=0.5$",
        (0, 0.5),
        xytext=(1.1, 0.30),
        arrowprops={"arrowstyle": "->"},
    )
    ax.set(xlabel="линейная оценка $z=b+w^Tx$", ylabel="score $s(x)$", ylim=(-0.03, 1.03))
    return fig


def learning_curve() -> plt.Figure:
    """Contrast optimization loss with an engineering metric."""

    step = np.arange(1, 41)
    loss = 0.78 * np.exp(-step / 12) + 0.16
    flight_recall = 0.35 + 0.52 * (1 - np.exp(-step / 15))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8))
    axes[0].plot(step, loss, color=ACCENT)
    axes[0].set(xlabel="шаг оптимизации", ylabel="loss по окнам", title="Настройка параметров")
    axes[1].plot(step[::5], flight_recall[::5], "o-", color=GREEN)
    axes[1].axhline(0.85, color=AMBER, linestyle="--", label="требование")
    axes[1].set(xlabel="контрольная точка", ylabel="recall по полётам", title="Инженерная проверка")
    axes[1].legend()
    return fig


def sorted_scores(score: np.ndarray, target: np.ndarray, threshold: float) -> plt.Figure:
    """Plot sorted model scores and the operational threshold."""

    score = np.asarray(score)
    target = np.asarray(target)
    sample = np.linspace(0, len(score) - 1, min(90, len(score)), dtype=int)
    order = np.argsort(score[sample])
    selected_score = score[sample][order]
    selected_target = target[sample][order]
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    colors = np.where(selected_target == 1, ACCENT2, ACCENT)
    ax.scatter(np.arange(len(order)), selected_score, c=colors, s=34)
    ax.axhline(threshold, color=INK, linestyle="--", label=f"порог {threshold:.2f}")
    ax.set(xlabel="окна, отсортированные по score", ylabel="score", ylim=(-0.03, 1.03))
    ax.legend(loc="upper left")
    return fig


def split_matrix() -> plt.Figure:
    """Visualize the two holdout axes used by the course case."""

    matrix = np.zeros((7, 7), dtype=int)
    matrix[:5, 5:] = 1
    matrix[5:, :] = 2
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.imshow(matrix, cmap=ListedColormap([ACCENT, GREEN, MUTED]), vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(7), [f"F{item}" for item in range(7)])
    ax.set_yticks(range(7), ["EMA-00", "EMA-01", "EMA-02", "EMA-03", "EMA-04", "EMA-18", "EMA-19"])
    ax.set(xlabel="порядок полёта привода", ylabel="привод")
    ax.legend(
        handles=[
            Patch(color=ACCENT, label="train"),
            Patch(color=GREEN, label="validation"),
            Patch(color=MUTED, label="test"),
        ],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
    )
    return fig


def window_leakage() -> plt.Figure:
    """Compare a leaky row split with a grouped flight split."""

    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    starts = np.arange(0, 51, 10)
    colors = [ACCENT, GREEN] * 3
    for start, color in zip(starts, colors, strict=True):
        ax.add_patch(Rectangle((start, 1.1), 20, 0.45, color=color, alpha=0.78))
    for start in starts:
        ax.add_patch(Rectangle((start, 0.15), 20, 0.45, color=ACCENT, alpha=0.78))
    ax.text(-3, 1.33, "плохо", ha="right", va="center", weight="bold")
    ax.text(-3, 0.38, "правильно", ha="right", va="center", weight="bold")
    ax.text(72, 1.33, "окна одного полёта попали в train и validation", va="center")
    ax.text(72, 0.38, "весь полёт остаётся в одной выборке", va="center")
    ax.set(xlim=(-5, 132), ylim=(0, 1.8), xlabel="время полёта, с")
    ax.set_yticks([])
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.grid(False)
    return fig


def scaler_leakage() -> plt.Figure:
    """Show why a preprocessing statistic must come from train only."""

    rng = np.random.default_rng(12)
    train = rng.normal(45, 4.2, 500)
    validation = rng.normal(55, 4.5, 220)
    fig, ax = plt.subplots(figsize=(7.8, 3.0))
    ax.hist(train, bins=24, alpha=0.55, color=ACCENT, label="train")
    ax.hist(validation, bins=20, alpha=0.55, color=GREEN, label="validation")
    ax.axvline(train.mean(), color=ACCENT, linewidth=3, label="среднее train")
    ax.axvline(
        np.r_[train, validation].mean(),
        color=ACCENT2,
        linestyle="--",
        linewidth=3,
        label="среднее всех данных — утечка",
    )
    ax.set(xlabel="температура, °C", ylabel="число окон")
    ax.legend(fontsize=10)
    return fig


def flight_aggregation(score: np.ndarray | None = None) -> plt.Figure:
    """Show window scores and max aggregation for three flights."""

    if score is None:
        score = np.array(
            [
                0.18,
                0.22,
                0.31,
                0.28,
                0.15,
                0.34,
                0.48,
                0.82,
                0.51,
                0.39,
                0.77,
                0.68,
                0.72,
                0.91,
                0.74,
            ]
        ).reshape(3, 5)
    else:
        score = np.asarray(score).reshape(3, -1)
    fig, ax = plt.subplots(figsize=(7.8, 3.1))
    for flight_index, row in enumerate(score):
        x_value = np.arange(row.size) + flight_index * (row.size + 1)
        ax.plot(x_value, row, "o", color=ACCENT, alpha=0.75)
        max_index = int(np.argmax(row))
        ax.plot(x_value[max_index], row[max_index], "o", color=ACCENT2, markersize=11)
        ax.text(x_value.mean(), -0.11, f"F{flight_index + 1}", ha="center")
    ax.set(ylabel="window_score", xlabel="окна внутри полёта", ylim=(-0.02, 1.03))
    ax.text(0.99, 0.93, "красная точка = max", transform=ax.transAxes, ha="right", color=ACCENT2)
    return fig


def precision_recall(
    target: np.ndarray, score: np.ndarray, threshold: float | None = None
) -> plt.Figure:
    """Plot a PR curve computed from provided flight-level scores."""

    precision, recall, thresholds = precision_recall_curve(target, score)
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.step(recall, precision, where="post", color=ACCENT)
    if threshold is not None and thresholds.size:
        index = int(np.argmin(np.abs(thresholds - threshold)))
        ax.plot(
            recall[index],
            precision[index],
            "o",
            color=ACCENT2,
            markersize=10,
            label=f"τ≈{threshold:.2f}",
        )
        ax.legend()
    ax.set(xlabel="recall", ylabel="precision", xlim=(0, 1.02), ylim=(0, 1.02))
    return fig


def threshold_tradeoff(target: np.ndarray, score: np.ndarray) -> plt.Figure:
    """Plot precision and recall against the candidate threshold."""

    precision, recall, thresholds = precision_recall_curve(target, score)
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.plot(thresholds, precision[:-1], color=ACCENT, label="precision")
    ax.plot(thresholds, recall[:-1], color=GREEN, label="recall")
    ax.axhline(0.85, color=AMBER, linestyle="--", label="recall ≥ 0.85")
    ax.set(xlabel="порог τ", ylabel="метрика", ylim=(0, 1.03))
    ax.legend(ncol=3, fontsize=10)
    return fig


def confusion(target: np.ndarray, prediction: np.ndarray) -> plt.Figure:
    """Draw a compact binary confusion matrix without hiding counts."""

    target = np.asarray(target)
    prediction = np.asarray(prediction)
    matrix = np.array(
        [
            [np.sum((target == 0) & (prediction == 0)), np.sum((target == 0) & (prediction == 1))],
            [np.sum((target == 1) & (prediction == 0)), np.sum((target == 1) & (prediction == 1))],
        ]
    )
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.imshow(
        matrix, cmap=ListedColormap(["#F3F6FA", "#86A9D8"]), vmin=0, vmax=max(1, matrix.max())
    )
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=24,
                weight="bold",
            )
    ax.set_xticks([0, 1], ["штатно", "осмотр"])
    ax.set_yticks([0, 1], ["штатно", "осмотр"])
    ax.set(xlabel="решение модели", ylabel="истинный исход")
    ax.grid(False)
    return fig


def feature_ablation(
    names: Sequence[str],
    values: Sequence[float],
    xlabel: str = "демонстрационная accuracy по полётам",
) -> plt.Figure:
    """Compare a small sequence of feature-set experiments."""

    fig, ax = plt.subplots(figsize=(7.8, 3.0))
    positions = np.arange(len(names))
    ax.barh(positions, values, color=[MUTED, ACCENT, GREEN, AMBER][: len(names)])
    ax.set_yticks(positions, names)
    ax.set(xlabel=xlabel, xlim=(0, 1))
    for position, value in zip(positions, values, strict=True):
        ax.text(value + 0.015, position, f"{value:.2f}", va="center")
    return fig


def error_profiles() -> plt.Figure:
    """Contrast interpretable feature profiles for FP and FN examples."""

    labels = ["ошибка слежения", "ток", "температура", "вибрация"]
    false_positive = np.array([0.3, 1.1, 1.4, 0.4])
    false_negative = np.array([0.8, -0.2, -0.5, 1.2])
    y_value = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    ax.barh(y_value - 0.18, false_positive, height=0.34, color=AMBER, label="ложная тревога")
    ax.barh(y_value + 0.18, false_negative, height=0.34, color=ACCENT2, label="пропуск")
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(y_value, labels)
    ax.set(xlabel="отклонение от среднего train, σ")
    ax.legend(ncol=2, fontsize=10)
    return fig


def multirate_timeline() -> plt.Figure:
    """Show two clocks with different sample rates and a shared grid."""

    fig, ax = plt.subplots(figsize=(8.2, 2.7))
    imu = np.arange(0, 1.001, 0.01)
    gnss = np.arange(0, 1.001, 0.1) + 0.018
    grid = np.arange(0, 1.001, 0.02)
    ax.vlines(imu, 2.05, 2.35, color=ACCENT, linewidth=0.6)
    ax.vlines(gnss, 1.15, 1.55, color=GREEN, linewidth=2.0)
    ax.vlines(grid, 0.20, 0.55, color=INK, linewidth=0.9)
    ax.text(-0.02, 2.2, "IMU ≈100 Гц", ha="right", va="center")
    ax.text(-0.02, 1.35, "GNSS ≈10 Гц", ha="right", va="center")
    ax.text(-0.02, 0.38, "сетка 50 Гц", ha="right", va="center")
    ax.set(xlim=(-0.04, 1.02), ylim=(0, 2.55), xlabel="время, с")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    return fig


def timestamp_defects() -> plt.Figure:
    """Make duplicate, backward and long-gap timestamps visible."""

    timestamp = np.array([0.00, 0.10, 0.20, 0.20, 0.32, 0.28, 0.40, 0.95, 1.05])
    index = np.arange(timestamp.size)
    fig, ax = plt.subplots(figsize=(8.0, 2.9))
    ax.plot(index, timestamp, "o-", color=ACCENT)
    ax.plot([3], [timestamp[3]], "o", color=AMBER, markersize=11, label="повтор")
    ax.plot([5], [timestamp[5]], "o", color=ACCENT2, markersize=11, label="ход назад")
    ax.plot([7], [timestamp[7]], "o", color=GREEN, markersize=11, label="разрыв")
    ax.set(xlabel="номер пакета", ylabel="timestamp, с")
    ax.legend(ncol=3, fontsize=10)
    return fig


def gap_policy() -> plt.Figure:
    """Illustrate interpolation across short spans and masking a long gap."""

    time = np.array([0.0, 0.1, 0.2, 0.3, 0.55, 0.65, 0.75, 1.45, 1.55, 1.65])
    value = np.sin(2 * np.pi * 0.7 * time)
    dense = np.linspace(0, 1.65, 250)
    interpolated = np.interp(dense, time, value)
    masked = interpolated.copy()
    masked[(dense > 0.75) & (dense < 1.45)] = np.nan
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    ax.plot(dense, interpolated, "--", color=MUTED, label="слепая интерполяция")
    ax.plot(dense, masked, color=ACCENT, label="политика max_gap")
    ax.plot(time, value, "o", color=INK, label="измерения")
    ax.axvspan(0.75, 1.45, color=ACCENT2, alpha=0.12, label="NaN: данных нет")
    ax.set(xlabel="время, с", ylabel="сигнал")
    ax.legend(ncol=2, fontsize=10)
    return fig


def window_validity() -> plt.Figure:
    """Show which windows are rejected after gap masking."""

    fig, ax = plt.subplots(figsize=(8.1, 2.8))
    starts = np.arange(0, 8)
    valid = np.array([True, True, False, False, False, True, True, True])
    for start, is_valid in zip(starts, valid, strict=True):
        color = GREEN if is_valid else ACCENT2
        ax.add_patch(Rectangle((start, 0.35), 2, 0.65, color=color, alpha=0.75))
        ax.text(
            start + 1,
            0.68,
            "OK" if is_valid else "drop",
            ha="center",
            va="center",
            color="white",
            weight="bold",
        )
    ax.axvspan(3.4, 5.1, color=INK, alpha=0.12, label="разрыв сигнала")
    ax.set(xlim=(-0.2, 9.2), ylim=(0.2, 1.25), xlabel="время")
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=10)
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    return fig


def repeatable_runs() -> plt.Figure:
    """Show the contract of a reproducible run without inventing a screenshot."""

    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    rows = [
        ("config.json", "7c8…", "7c8…"),
        ("split.json", "da1…", "da1…"),
        ("validation-metrics.json", "2f4…", "2f4…"),
    ]
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["артефакт", "запуск 1", "запуск 2"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.46, 0.22, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.6)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor("#EAF0F8")
            cell.set_text_props(weight="bold")
    return fig
