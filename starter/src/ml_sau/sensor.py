"""Sensor-data pipeline for S2 with two focused student decisions."""

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ml_sau.sensor_source import G0, generate_sensor_log


def timestamp_report(t_s: NDArray[np.float64]) -> dict[str, int]:
    """Count non-finite, duplicate, and backward timestamps."""

    raise NotImplementedError("Implement on S2")


def interpolate_with_gap_mask(
    t_s: NDArray[np.float64],
    values: NDArray[np.float64],
    grid_s: NDArray[np.float64],
    max_gap_s: float,
) -> NDArray[np.float64]:
    """Interpolate short spans and return NaN across gaps or outside the input."""

    raise NotImplementedError("Implement on S2")


def build_windows(
    values: NDArray[np.float64], window_size: int, stride: int
) -> NDArray[np.float64]:
    """Build fixed-length windows and omit those containing non-finite values."""

    if values.ndim not in (1, 2):
        raise ValueError("values must be a one- or two-dimensional array")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if len(values) < window_size:
        shape = (0, window_size, *values.shape[1:])
        return np.empty(shape, dtype=np.float64)

    starts = range(0, len(values) - window_size + 1, stride)
    candidates = np.stack([values[start : start + window_size] for start in starts])
    finite_axes = tuple(range(1, candidates.ndim))
    return candidates[np.isfinite(candidates).all(axis=finite_axes)]


def write_quality_report(output: Path) -> None:
    """Run the S2 pipeline and write its deterministic JSON report."""

    log = generate_sensor_log()
    time_quality = timestamp_report(log.imu_t_s)

    # Keep the first sample at a duplicated timestamp; interpolation requires a
    # strictly increasing source clock.
    keep = np.r_[True, np.diff(log.imu_t_s) > 0]
    source_t = log.imu_t_s[keep]
    values_si = np.column_stack(
        (
            log.acceleration_g[keep] * G0,
            np.deg2rad(log.angular_rate_dps[keep]),
        )
    )
    sample_period_s = 0.02
    grid_s = np.arange(source_t[0], source_t[-1], sample_period_s)
    aligned = interpolate_with_gap_mask(
        source_t,
        values_si,
        grid_s,
        max_gap_s=0.10,
    )
    windows = build_windows(aligned, window_size=100, stride=50)

    report = {
        "artifact": "s2-quality-and-feature-contract",
        "synthetic_data": True,
        "source": "separate IMU/GNSS teaching log; not the S1/S3 actuator table",
        "timestamp_quality": time_quality,
        "feature_contract": {
            "clock": "IMU",
            "sample_period_s": sample_period_s,
            "columns": ["acceleration_x_mps2", "acceleration_y_mps2", "angular_rate_radps"],
            "max_interpolated_gap_s": 0.10,
            "window_size_samples": 100,
            "stride_samples": 50,
            "windows_with_gaps_are_dropped": True,
        },
        "n_grid_samples": len(grid_s),
        "n_valid_windows": len(windows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    write_quality_report(Path("reports/s2-quality.json"))
