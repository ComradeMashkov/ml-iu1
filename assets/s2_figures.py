"""S2 figures computed from exactly the same source and reference pipeline as the notebook."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter" / "src"))
from figstyle import ACCENT, ACCENT2, GREEN, MUTED
from ml_sau.sensor import prepare_sensor_channels, process_sensor_log
from ml_sau.sensor_source import generate_sensor_log


def raw_timing():
    log = generate_sensor_log()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.8))
    dt = np.diff(log.imu_t_s)
    axes[0].plot(log.imu_t_s[1:], dt, color=ACCENT, lw=1.3)
    axes[0].scatter(log.imu_t_s[1:][dt == 0], dt[dt == 0], color=ACCENT2, zorder=3)
    axes[0].set(xlabel="время IMU, с", ylabel="Δt IMU, с")
    axes[0].annotate("повтор", xy=(12.0, 0), xytext=(9, 0.16), arrowprops={"arrowstyle": "->"})
    common_t = (log.gnss_t_s - log.gnss_clock_offset_s) / log.gnss_clock_rate
    axes[1].plot(common_t, log.gnss_t_s - common_t, color=GREEN)
    axes[1].set(xlabel="общая шкала, с", ylabel=r"$t_{\mathrm{GNSS}}-t$, с")
    return fig


def interpolation_example():
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot([0, 2], [0, 4], "--", color=MUTED)
    ax.scatter([0, 2], [0, 4], color=ACCENT, label="исходные значения", s=70)
    ax.scatter([1], [2], color=GREEN, label="при пределе 2 с: v(1)=2", s=70)
    ax.annotate("при пределе 0.5 с: здесь NaN", (1, 2), xytext=(0.2, 3.5), fontsize=11)
    ax.set(xlim=(-0.2, 2.2), ylim=(-0.2, 4.3), xlabel="время, с", ylabel="значение v")
    ax.legend(fontsize=10, loc="lower right")
    return fig


def gap_and_windows():
    strict = process_sensor_log(max_gap_s=0.10)
    relaxed = process_sensor_log(max_gap_s=0.50)
    log = generate_sensor_log()
    imu_t, imu_si, _, _ = prepare_sensor_channels(log)
    batch, grid = strict[2], strict[3]
    fig, axes = plt.subplots(
        2, 1, figsize=(9, 4.2), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]}
    )
    local = (imu_t >= 15.5) & (imu_t <= 20.5)
    axes[0].plot(imu_t[local], imu_si[local, 0], ".", ms=2, color=MUTED, label="измерения IMU")
    axes[0].plot(grid, relaxed[4][:, 0], "--", color=ACCENT2, label="предел 0.50 с")
    axes[0].plot(grid, strict[4][:, 0], color=ACCENT, label="предел 0.10 с")
    axes[0].set(ylabel="fₓ, м/с²")
    axes[0].legend(fontsize=10, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    for i, start in enumerate([16.0, 17.0, 18.0]):
        idx = int(np.flatnonzero(np.isclose(batch.start_s, start))[0])
        color = GREEN if batch.valid[idx] else ACCENT2
        axes[1].plot([start, start + 1.98], [i, i], color=color, lw=8, solid_capstyle="butt")
        axes[1].text(
            start + 0.99, i + 0.15, f"окно с началом {start:g} с", ha="center", fontsize=10
        )
    gap_i = int(np.argmax(np.diff(imu_t)))
    for ax in axes:
        ax.axvspan(imu_t[gap_i], imu_t[gap_i + 1], color=MUTED, alpha=0.18)
    axes[1].set(xlim=(15.5, 20.5), ylim=(-0.5, 2.6), xlabel="общая шкала времени, с", yticks=[])
    return fig
