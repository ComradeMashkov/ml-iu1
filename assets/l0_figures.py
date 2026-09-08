"""Slide-sized, reproducible illustrations for L0.

All numbers in this module belong to teaching examples. Optimisation and
regression illustrations are computed from the displayed data and fitted models.
"""

import matplotlib.pyplot as plt
import numpy as np
from figstyle import ACCENT, ACCENT2, AMBER, GREEN, GRID, INK, MUTED
from matplotlib.patches import Patch, Rectangle
from numpy.polynomial import Polynomial

TEMPERATURE = np.array([47.0, 51.0, 64.0, 69.0])
CURRENT = np.array([10.2, 11.0, 14.1, 15.0])
TARGET = np.array([0, 0, 1, 1])
FEATURES = np.column_stack(((TEMPERATURE - 60) / 10, (CURRENT - 12) / 2))
IDS = ["F01", "F02", "F03", "F04"]
COLORS = np.where(TARGET == 1, ACCENT2, ACCENT)


def _axes(ncols=1, *, width=12, height=4.5):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), layout="constrained")
    for ax in np.atleast_1d(axes):
        ax.tick_params(labelsize=15)
        ax.xaxis.label.set_fontsize(17)
        ax.yaxis.label.set_fontsize(17)
        ax.title.set_fontsize(19)
    return fig, axes


def _points(ax, *, scaled=False, annotate=True):
    x, y = FEATURES.T if scaled else (TEMPERATURE, CURRENT)
    ax.scatter(x, y, c=COLORS, s=155, edgecolors="white", linewidths=1.8, zorder=5)
    if annotate:
        for label, x_value, y_value in zip(IDS, x, y, strict=True):
            ax.annotate(
                label,
                (x_value, y_value),
                xytext=(9, 7),
                textcoords="offset points",
                fontsize=17,
                zorder=6,
            )
    if scaled:
        ax.set(xlim=(-1.8, 1.65), ylim=(-1.5, 2.1), xlabel="$x_1$", ylabel="$x_2$")
    else:
        ax.set(
            xlim=(43, 75),
            ylim=(9, 16.8),
            xlabel="Средняя температура, °C",
            ylabel="RMS тока, А",
        )


def _class_legend(ax, **kwargs):
    ax.legend(
        handles=[Patch(color=ACCENT, label="$y=0$"), Patch(color=ACCENT2, label="$y=1$")],
        fontsize=16,
        ncol=2,
        **kwargs,
    )


def flight_windows() -> plt.Figure:
    """A generated current signal and three overlapping 20-second windows."""
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 4.8),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [2, 1.5]},
    )
    time = np.linspace(0, 40, 801)
    current = 11 + 0.7 * np.sin(time / 3.7) + 0.25 * np.sin(2.4 * time)
    current += 1.8 * np.exp(-(((time - 24) / 3) ** 2))
    axes[0].plot(time, current, color=INK, lw=2)
    axes[0].set(ylabel="Ток, А", ylim=(9.5, 14), title="Один полёт · flight_id = F05")
    axes[0].set_yticks([10, 12, 14])
    for index, (start, color) in enumerate(zip([0, 10, 20], [ACCENT, GREEN, AMBER])):
        axes[1].add_patch(Rectangle((start, 2 - index - 0.32), 20, 0.64, color=color))
        axes[1].text(
            start + 10,
            2 - index,
            f"W{index + 1}: {start}–{start + 20} с",
            ha="center",
            va="center",
            fontsize=18,
            color="white",
            weight="medium",
        )
    axes[1].set(xlim=(-0.3, 40.3), ylim=(-0.5, 2.5), xlabel="Время, с", yticks=[])
    axes[1].set_xticks([0, 10, 20, 30, 40])
    axes[1].spines[["left", "top", "right"]].set_visible(False)
    for ax in axes:
        ax.tick_params(labelsize=15)
        ax.xaxis.label.set_fontsize(17)
        ax.yaxis.label.set_fontsize(17)
    return fig


def flight_rows() -> plt.Figure:
    """Show 72 rows grouped into three flights, with one label per flight."""
    fig, ax = _axes(height=4.3)
    ax.axis("off")
    for row, (flight, target, color) in enumerate(
        [("F05", 0, ACCENT), ("F06", 1, ACCENT2), ("F07", 0, ACCENT)]
    ):
        y = 2 - row
        ax.text(-1, y, flight, ha="right", va="center", fontsize=21, weight="medium")
        for window in range(24):
            ax.add_patch(Rectangle((window, y - 0.24), 0.8, 0.48, color=color, alpha=0.78))
        ax.text(25.2, y, f"24 окна\nобщая метка $y={target}$", va="center", fontsize=19)
    for window in [1, 8, 16, 24]:
        ax.text(window - 0.6, 2.5, f"W{window}", ha="center", fontsize=16, color=INK)
    ax.text(11.5, -0.8, "3 полёта $\\to$ 72 строки таблицы", ha="center", fontsize=22)
    ax.set(xlim=(-3, 34), ylim=(-1, 3))
    return fig


def feature_table() -> plt.Figure:
    """The full four-row teaching dataset; X and y use distinct column fills."""
    fig, ax = _axes(height=4.4)
    ax.axis("off")
    rows = [
        [name, f"{t:.0f}", f"{current:.1f}", str(y)]
        for name, t, current, y in zip(IDS, TEMPERATURE, CURRENT, TARGET, strict=True)
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Полёт", "Температура, °C", "RMS тока, А", "Метка y"],
        cellLoc="center",
        bbox=[0.02, 0.03, 0.96, 0.78],
        colWidths=[0.16, 0.32, 0.3, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(21)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(3)
        cell.set_facecolor("#EDF3FB" if col in (1, 2) else "#FBEFEE" if col == 3 else "#F2F3F5")
        if row == 0:
            cell.set_text_props(weight="medium", fontsize=18)
            cell.set_facecolor(ACCENT if col in (1, 2) else ACCENT2 if col == 3 else INK)
            cell.get_text().set_color("white")
    ax.text(0.49, 0.9, "Признаки X", ha="center", fontsize=24, color=ACCENT)
    ax.text(0.89, 0.9, "Цель y", ha="center", fontsize=24, color=ACCENT2)
    return fig


def toy_scatter() -> plt.Figure:
    """The same rows plotted as points in a two-feature space."""
    fig, ax = _axes()
    _points(ax)
    _class_legend(ax, loc="upper left")
    return fig


def scale_features() -> plt.Figure:
    """An explicit change of coordinates, preserving the four observations."""
    fig, axes = _axes(2, height=4.7)
    _points(axes[0], annotate=False)
    _points(axes[1], scaled=True, annotate=False)
    axes[0].set_title("Исходные признаки", fontsize=20)
    axes[1].set_title("$x_1=(T-60)/10$,  $x_2=(I-12)/2$", fontsize=19)
    for ax, coordinates, label in [
        (axes[0], (64, 14.1), "F03: (64; 14.1)"),
        (axes[1], (0.4, 1.05), "F03: (0.4; 1.05)"),
    ]:
        ax.annotate(
            label,
            coordinates,
            xytext=(0.05, 0.87),
            textcoords="axes fraction",
            fontsize=19,
            color=ACCENT2,
            arrowprops={"arrowstyle": "->", "color": ACCENT2, "lw": 1.4},
        )
    return fig


def linear_score() -> plt.Figure:
    """The level lines of z=x1+x2, with F03 on its z=1.45 contour."""
    fig, ax = _axes()
    x1 = np.linspace(-1.8, 1.65, 151)
    x2 = np.linspace(-1.5, 2.1, 151)
    xx, yy = np.meshgrid(x1, x2)
    ax.contourf(xx, yy, xx + yy, levels=[-4, 0, 4], colors=[ACCENT, ACCENT2], alpha=0.08)
    contours = ax.contour(xx, yy, xx + yy, levels=[-2, -1, 1, 2], colors=MUTED, linewidths=1.1)
    ax.clabel(contours, inline=True, fmt=lambda value: f"z = {value:g}", fontsize=14)
    ax.plot(x1, -x1, color=INK, lw=2.2, label="$z=0$")
    _points(ax, scaled=True, annotate=False)
    ax.scatter(
        [0.4], [1.05], s=320, facecolors="none", edgecolors=ACCENT2, linewidths=2.3, zorder=7
    )
    ax.annotate(
        "F03: z = 0.4 + 1.05 = 1.45",
        (0.4, 1.05),
        xytext=(-1.65, 1.8),
        fontsize=20,
        color=ACCENT2,
        arrowprops={"arrowstyle": "->", "color": ACCENT2, "lw": 1.4},
    )
    ax.text(-1.55, -1.22, "$z<0$", color=ACCENT, fontsize=23)
    ax.text(1.12, 1.7, "$z>0$", color=ACCENT2, fontsize=23)
    ax.legend(loc="lower right", fontsize=17)
    return fig


def sigmoid() -> plt.Figure:
    """Map two exact linear scores onto sigmoid probabilities."""
    fig, ax = _axes()
    z = np.linspace(-6, 6, 500)
    ax.plot(z, 1 / (1 + np.exp(-z)), color=ACCENT, lw=3)
    for value, color in [(0, MUTED), (1.45, ACCENT2)]:
        score = 1 / (1 + np.exp(-value))
        ax.plot([value, value, -6], [0, score, score], "--", lw=1.4, color=color)
        ax.scatter([value], [score], s=115, color=color, edgecolor="white", zorder=5)
    ax.annotate(
        "z = 1.45 $\\to$ s ≈ 0.81",
        (1.45, 1 / (1 + np.exp(-1.45))),
        xytext=(-4.8, 0.91),
        fontsize=21,
        color=ACCENT2,
        arrowprops={"arrowstyle": "->", "color": ACCENT2},
    )
    ax.annotate(
        "z = 0 $\\to$ s = 0.5",
        (0, 0.5),
        xytext=(1.5, 0.31),
        fontsize=20,
        arrowprops={"arrowstyle": "->", "color": MUTED},
    )
    ax.set(xlim=(-6, 6), ylim=(0, 1.05), xlabel="Линейная оценка z", ylabel="Выход модели s")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    return fig


def compare_rules() -> plt.Figure:
    """Two decision boundaries and all four scores, calculated from their rules."""
    fig, axes = _axes(2, height=4.8)
    for ax, scores, title in zip(
        axes,
        [1 / (1 + np.exp(-FEATURES[:, 0])), 1 / (1 + np.exp(-FEATURES.sum(axis=1)))],
        ["A: z = x₁", "B: z = x₁ + x₂"],
        strict=True,
    ):
        _points(ax, annotate=False)
        ax.set_title(title, fontsize=22)
        for name, t, current, score in zip(IDS, TEMPERATURE, CURRENT, scores, strict=True):
            # Place each label in its own horizontal band to keep neighbouring points distinct.
            offset = (0, -27) if name in ("F01", "F03") else (0, 17)
            ax.annotate(
                f"{name}: {score:.2f}",
                (t, current),
                xytext=offset,
                textcoords="offset points",
                fontsize=15,
                zorder=6,
            )
    axes[0].axvline(60, color=INK, linestyle="--", lw=2)
    boundary = np.array([43, 75])
    axes[1].plot(boundary, 24 - 0.2 * boundary, color=INK, linestyle="--", lw=2)
    axes[1].set_ylabel("")
    return fig


def binary_loss() -> plt.Figure:
    """Binary log-loss for both labels; mark two predictions when y=1."""
    fig, ax = _axes()
    score = np.linspace(0.015, 0.985, 400)
    ax.plot(score, -np.log(score), color=ACCENT2, lw=2.8, label="$y=1$:  $-\\log(s)$")
    ax.plot(score, -np.log1p(-score), color=ACCENT, lw=2.8, label="$y=0$:  $-\\log(1-s)$")
    for logit, location in [(0.4, (0.36, 1.1)), (1.45, (0.70, 0.9))]:
        value = 1 / (1 + np.exp(-logit))
        loss = -np.log(value)
        ax.scatter([value], [loss], s=110, color=ACCENT2, edgecolor="white", zorder=5)
        ax.annotate(
            f"s ≈ {value:.3f}\nℓ ≈ {loss:.3f}",
            (value, loss),
            xytext=location,
            fontsize=18,
            color=ACCENT2,
            arrowprops={"arrowstyle": "->", "color": ACCENT2},
        )
    ax.set(xlim=(0, 1), ylim=(0, 3), xlabel="Оценка класса 1: s", ylabel="Потеря ℓ")
    ax.legend(loc="upper center", ncol=2, fontsize=18)
    return fig


def fit_iterations() -> plt.Figure:
    """Run full-batch gradient descent on logistic loss with L2 lambda=0.02."""
    design = np.column_stack([np.ones(4), FEATURES])
    parameters = np.zeros(3)
    regularization = 0.02
    steps = 101
    objective = np.zeros(steps)
    parameter_history = np.zeros((steps, 3))
    for step in range(steps):
        logits = design @ parameters
        objective[step] = np.mean(np.logaddexp(0, logits) - TARGET * logits)
        objective[step] += regularization * np.dot(parameters[1:], parameters[1:]) / 2
        parameter_history[step] = parameters
        score = 1 / (1 + np.exp(-logits))
        gradient = design.T @ (score - TARGET) / len(TARGET)
        gradient[1:] += regularization * parameters[1:]
        parameters = parameters - 0.6 * gradient
    fig, axes = _axes(2, height=4.6)
    axes[0].plot(np.arange(steps), objective, color=ACCENT, lw=3)
    axes[0].scatter([0, steps - 1], objective[[0, -1]], color=ACCENT, s=80, zorder=4)
    axes[0].annotate(
        f"{objective[0]:.3f}",
        (0, objective[0]),
        xytext=(8, -4),
        textcoords="offset points",
        fontsize=18,
    )
    axes[0].annotate(
        f"{objective[-1]:.3f}",
        (steps - 1, objective[-1]),
        xytext=(-3, 12),
        textcoords="offset points",
        fontsize=18,
        ha="right",
    )
    axes[0].set(xlabel="Шаг оптимизации", ylabel="Целевая функция J", title="Потеря + L2-штраф")
    for index, (label, color) in enumerate(zip(["b", "w₁", "w₂"], [MUTED, ACCENT, ACCENT2])):
        axes[1].plot(
            np.arange(steps), parameter_history[:, index], label=label, color=color, lw=2.5
        )
    axes[1].set(xlabel="Шаг оптимизации", ylabel="Значение параметра", title="Настройка b, w₁, w₂")
    axes[1].legend(ncol=3, fontsize=18, loc="lower right")
    return fig


def flight_aggregation() -> plt.Figure:
    """Take each flight's maximum window score and apply threshold 0.5."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.8),
        layout="constrained",
        gridspec_kw={"width_ratios": [2.3, 1.3]},
    )
    scores = [[0.1, 0.2, 0.35], [0.2, 0.4, 0.8], [0.15, 0.3, 0.6], [0.1, 0.25, 0.45]]
    targets = [0, 1, 0, 1]
    outcomes = [
        "верный прогноз класса 0",
        "верный прогноз класса 1",
        "ложное срабатывание",
        "пропуск",
    ]
    for index, (values, target, outcome) in enumerate(zip(scores, targets, outcomes, strict=True)):
        row = 3 - index
        color = ACCENT if target == 0 else ACCENT2
        axes[0].plot([min(values), max(values)], [row, row], color=GRID, lw=4)
        axes[0].scatter(values, [row] * 3, color=color, s=100, edgecolor="white", zorder=4)
        axes[0].scatter(
            [max(values)], [row], facecolor="none", edgecolor=INK, s=230, lw=1.7, zorder=5
        )
        axes[0].annotate(
            f"{max(values):.2f}",
            (max(values), row),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            fontsize=17,
        )
        prediction = int(max(values) >= 0.5)
        axes[1].text(0, row + 0.10, f"ŷ = {prediction}", fontsize=21, va="center", color=INK)
        axes[1].text(
            0,
            row - 0.20,
            outcome,
            fontsize=16,
            va="center",
            color=GREEN if prediction == target else ACCENT2,
        )
    axes[0].axvline(0.5, color=AMBER, linestyle="--", lw=2)
    axes[0].text(0.515, 3.65, "порог 0.5", color=AMBER, fontsize=17)
    axes[0].set(
        xlim=(0, 1),
        ylim=(-0.5, 3.9),
        xlabel="Оценки трёх окон; обведён максимум",
        yticks=[3, 2, 1, 0],
        yticklabels=[f"F0{i + 5} · y = {y}" for i, y in enumerate(targets)],
    )
    axes[0].tick_params(labelsize=16)
    axes[0].grid(axis="y", visible=False)
    axes[1].set(xlim=(0, 1), ylim=(-0.5, 3.9))
    axes[1].axis("off")
    return fig


def holdout_flights() -> plt.Figure:
    """An illustrative group assignment; identifiers are schematic."""
    fig, ax = _axes(height=4.5)
    ax.axis("off")
    groups = [
        ("Обучение", ["A01", "A02", "A03", "A04"], ACCENT, "fit"),
        ("Валидация", ["B01", "B02"], GREEN, "выбор настроек"),
        ("Тест", ["C01", "C02"], AMBER, "итоговая оценка"),
    ]
    for row, (label, flights, color, purpose) in enumerate(groups):
        y = 2 - row
        ax.text(-0.25, y, label, ha="right", va="center", fontsize=21, color=color)
        for index, flight in enumerate(flights):
            x = index * 2.2
            ax.add_patch(
                Rectangle(
                    (x, y - 0.28), 1.8, 0.56, facecolor=color, alpha=0.12, edgecolor=color, lw=1.4
                )
            )
            for window in range(6):
                ax.add_patch(
                    Rectangle(
                        (x + 0.1 + 0.27 * window, y - 0.19), 0.22, 0.16, facecolor=color, alpha=0.85
                    )
                )
            ax.text(x + 0.9, y + 0.10, flight, ha="center", va="center", fontsize=16)
        ax.text(9.2, y, purpose, va="center", fontsize=20)
    ax.text(4.2, -0.85, "Все окна полёта остаются в одной группе", fontsize=22, ha="center")
    ax.set(xlim=(-3, 14), ylim=(-1.05, 2.7))
    return fig


def overfitting() -> plt.Figure:
    """Fit degree-2 and degree-9 polynomials to identical noisy training data."""
    rng = np.random.default_rng(4)
    train_x = np.linspace(-0.9, 0.9, 10)
    validation_x = np.linspace(-0.85, 0.85, 23)
    true_signal = lambda values: 0.5 + 0.65 * values - 0.7 * values**2
    train_y = true_signal(train_x) + rng.normal(0, 0.17, len(train_x))
    validation_y = true_signal(validation_x) + rng.normal(0, 0.17, len(validation_x))
    plot_x = np.linspace(-0.9, 0.9, 500)
    fig, axes = _axes(2, height=4.8)
    for ax, degree, color in zip(axes, [2, 9], [ACCENT, ACCENT2], strict=True):
        model = Polynomial.fit(train_x, train_y, degree)
        train_mae = np.mean(np.abs(model(train_x) - train_y))
        validation_mae = np.mean(np.abs(model(validation_x) - validation_y))
        ax.plot(plot_x, model(plot_x), color=color, lw=2.4, label="Модель")
        ax.scatter(train_x, train_y, color=INK, s=65, label="Обучение", zorder=4)
        ax.scatter(
            validation_x,
            validation_y,
            color=GREEN,
            marker="x",
            s=60,
            linewidths=1.8,
            label="Новые объекты",
            zorder=3,
        )
        ax.set(
            xlabel="Нормированная нагрузка x",
            ylabel="Отклонение тока, А",
            xlim=(-0.94, 0.94),
            ylim=(-1.5, 1.7),
        )
        ax.set_title(
            f"Степень {degree}\nMAE: обучение {train_mae:.2f} А · новые {validation_mae:.2f} А",
            fontsize=17,
        )
    axes[1].set_ylabel("")
    axes[0].legend(loc="upper left", fontsize=14)
    return fig


def regression_residuals() -> plt.Figure:
    """Three given temperature forecasts, with signed errors +2, -1 and +4 °C."""
    observed = np.array([50.0, 60.0, 65.0])
    predicted = np.array([52.0, 59.0, 69.0])
    absolute_errors = np.abs(predicted - observed)
    positions = np.arange(1, 4)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        layout="constrained",
        gridspec_kw={"width_ratios": [2.1, 1]},
    )
    axes[0].scatter(positions, observed, color=INK, s=150, label="Измерение y", zorder=5)
    axes[0].scatter(
        positions, predicted, color=ACCENT, marker="D", s=135, label="Прогноз ŷ", zorder=5
    )
    axes[0].vlines(positions, predicted, observed, color=ACCENT2, lw=3.3, zorder=4)
    for x_value, actual, forecast in zip(positions, observed, predicted, strict=True):
        axes[0].annotate(
            f"{forecast - actual:+.0f} °C",
            (x_value, (actual + forecast) / 2),
            xytext=(19, 0),
            textcoords="offset points",
            fontsize=23,
            va="center",
            color=ACCENT2,
        )
    axes[0].set(
        xlim=(0.6, 3.6),
        ylim=(47, 74),
        xticks=positions,
        xlabel="Наблюдение",
        ylabel="Температура через 30 с, °C",
    )
    axes[0].legend(loc="upper left", fontsize=17)
    axes[0].tick_params(labelsize=16)
    axes[1].axis("off")
    axes[1].text(0.05, 0.76, "Абсолютные ошибки", fontsize=21)
    axes[1].text(0.05, 0.58, "2 °C    1 °C    4 °C", fontsize=22, color=ACCENT2)
    axes[1].text(0.05, 0.35, "MAE = (2 + 1 + 4) / 3", fontsize=21)
    axes[1].text(0.05, 0.15, f"≈ {absolute_errors.mean():.2f} °C", fontsize=29, color=ACCENT2)
    return fig
