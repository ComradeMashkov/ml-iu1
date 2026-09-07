"""Reference S2 pipeline: sensor clocks, usable windows, and feature rows."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ml_sau.sensor_source import G0, SensorLog, generate_sensor_log

GAP_TOLERANCE_S = 1e-9
CHANNEL_NAMES = (
    "specific_force_x_mps2",
    "specific_force_z_mps2",
    "angular_rate_radps",
    "speed_mps",
)
FEATURE_NAMES = (
    "specific_force_x_rms_mps2",
    "specific_force_z_mean_mps2",
    "angular_rate_std_radps",
    "speed_mean_mps",
)


@dataclass(frozen=True)
class WindowBatch:
    """All candidate windows, their start times, and a separate validity mask."""

    values: NDArray[np.float64]
    start_s: NDArray[np.float64]
    valid: NDArray[np.bool_]


def timestamp_report(t_s: NDArray[np.float64]) -> dict[str, int]:
    """Count non-finite entries and defects in finite adjacent timestamp pairs."""
    t_s = np.asarray(t_s, dtype=np.float64)
    if t_s.ndim != 1:
        raise ValueError("t_s must be one-dimensional")
    finite = np.isfinite(t_s)
    pair_is_finite = finite[:-1] & finite[1:]
    with np.errstate(invalid="ignore"):
        dt = np.diff(t_s)
    return {
        "non_finite": int((~finite).sum()),
        "duplicate": int(((dt == 0) & pair_is_finite).sum()),
        "backward": int(((dt < 0) & pair_is_finite).sum()),
    }


def linear_interpolation(t_s, values, grid_s):
    """Prepared shape checks and NumPy interpolation, without a gap policy."""
    t_s = np.asarray(t_s, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    grid_s = np.asarray(grid_s, dtype=np.float64)
    if t_s.ndim != 1 or grid_s.ndim != 1:
        raise ValueError("t_s and grid_s must be one-dimensional")
    if values.ndim not in (1, 2) or len(values) != len(t_s):
        raise ValueError("values must have shape (n,) or (n, channels)")
    if not len(t_s) or not np.isfinite(t_s).all() or np.any(np.diff(t_s) <= 0):
        raise ValueError("t_s must be nonempty, finite, and strictly increasing")
    if values.ndim == 2 and values.shape[1] == 0:
        raise ValueError("values must contain at least one channel")
    columns = values[:, None] if values.ndim == 1 else values
    result = np.column_stack(
        [np.interp(grid_s, t_s, c, left=np.nan, right=np.nan) for c in columns.T]
    )
    return result[:, 0] if values.ndim == 1 else result


def interpolate_with_gap_mask(t_s, values, grid_s, max_gap_s):
    """Keep measured endpoints; mask only the interiors of long intervals."""
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive and finite")
    t_s = np.asarray(t_s, dtype=np.float64)
    grid_s = np.asarray(grid_s, dtype=np.float64)
    result = linear_interpolation(t_s, values, grid_s)
    for left, right in zip(t_s[:-1], t_s[1:], strict=True):  # noqa: RUF007 — matches the lesson
        if right - left > max_gap_s + GAP_TOLERANCE_S:
            inside = (grid_s > left) & (grid_s < right)
            result[inside] = np.nan
    return result


def prepare_sensor_channels(log: SensorLog):
    """Reference for the clock and unit conversion written in notebook block 2."""
    quality = timestamp_report(log.imu_t_s)
    if quality["non_finite"] or quality["backward"]:
        raise ValueError("repair or reject non-finite/backward IMU time before processing")
    keep = np.r_[True, np.diff(log.imu_t_s) > 0]
    imu_t = log.imu_t_s[keep]
    imu_si = np.column_stack(
        (log.acceleration_g[keep] * G0, np.deg2rad(log.angular_rate_dps[keep]))
    )
    gnss_t = (log.gnss_t_s - log.gnss_clock_offset_s) / log.gnss_clock_rate
    start = max(imu_t[0], gnss_t[0])
    stop = min(imu_t[-1], gnss_t[-1])
    grid_s = np.arange(np.ceil(start / 0.02) * 0.02, stop, 0.02)
    return imu_t, imu_si, gnss_t, grid_s


def build_window_batch(values, grid_s, window_size=100, stride=50):
    """Preserve the start and validity of every complete candidate window."""
    values = np.asarray(values, dtype=np.float64)
    grid_s = np.asarray(grid_s, dtype=np.float64)
    if values.ndim not in (1, 2) or grid_s.ndim != 1 or len(grid_s) != len(values):
        raise ValueError("values and one-dimensional grid_s must have matching lengths")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if not np.isfinite(grid_s).all() or np.any(np.diff(grid_s) <= 0):
        raise ValueError("grid_s must be finite and strictly increasing")
    starts = np.arange(0, max(0, len(values) - window_size + 1), stride)
    if not len(starts):
        windows = np.empty((0, window_size, *values.shape[1:]), dtype=np.float64)
    else:
        windows = np.stack([values[i : i + window_size] for i in starts])
    axes = tuple(range(1, windows.ndim))
    valid = np.isfinite(windows).all(axis=axes)
    return WindowBatch(windows, grid_s[starts], valid)


def build_windows(values, window_size, stride):
    """Compatibility wrapper returning only finite windows."""
    batch = build_window_batch(values, np.arange(len(values)), window_size, stride)
    return batch.values[batch.valid]


def window_features(batch: WindowBatch):
    """Return one row per valid window, in FEATURE_NAMES order."""
    if batch.values.ndim != 3 or batch.values.shape[2] != len(CHANNEL_NAMES):
        raise ValueError("features require four channels: force x, force z, rate, speed")
    w = batch.values[batch.valid]
    force_x_rms = np.sqrt(np.mean(w[:, :, 0] ** 2, axis=1))
    force_z_mean = np.mean(w[:, :, 1], axis=1)
    rate_std = np.std(w[:, :, 2], axis=1)
    speed_mean = np.mean(w[:, :, 3], axis=1)
    return np.column_stack((force_x_rms, force_z_mean, rate_std, speed_mean))


def assemble_quality_report(log, time_quality, grid_s, batch, X, max_gap_s):
    """Prepared serialization schema; computation is visible in the notebook."""
    return {
        "artifact": "s2-quality-and-feature-contract",
        "synthetic_data": True,
        "source": "separate IMU/GNSS teaching log; not the S1/S3 actuator table",
        "timestamp_quality": time_quality,
        "clock_conversion": {
            "reference": "provided common time in seconds",
            "gnss_clock_rate": log.gnss_clock_rate,
            "gnss_clock_offset_s": log.gnss_clock_offset_s,
        },
        "feature_contract": {
            "sample_period_s": 0.02,
            "input_columns": list(CHANNEL_NAMES),
            "columns": list(FEATURE_NAMES),
            "max_interpolated_gap_s": max_gap_s,
            "gap_comparison_tolerance_s": GAP_TOLERANCE_S,
            "window_size_samples": 100,
            "stride_samples": 50,
            "windows_with_gaps_are_dropped": True,
            "standard_deviation_ddof": 0,
            "decision_unit": "telemetry window; target labels are not supplied",
        },
        "n_grid_samples": len(grid_s),
        "n_candidate_windows": len(batch.valid),
        "n_valid_windows": int(batch.valid.sum()),
        "feature_shape": list(X.shape),
        "valid_window_start_s": batch.start_s[batch.valid].tolist(),
        "discarded_window_start_s": batch.start_s[~batch.valid].tolist(),
    }


def process_sensor_log(
    timestamp_fn=timestamp_report, interpolation_fn=interpolate_with_gap_mask, *, max_gap_s=0.10
):
    """Reference reproduction of all six notebook stages."""
    log = generate_sensor_log()
    time_quality = timestamp_fn(log.imu_t_s)
    imu_t, imu_si, gnss_t, grid_s = prepare_sensor_channels(log)
    imu_aligned = interpolation_fn(imu_t, imu_si, grid_s, max_gap_s)
    speed_aligned = interpolation_fn(gnss_t, log.speed_mps, grid_s, max_gap_s)
    aligned = np.column_stack((imu_aligned, speed_aligned))
    batch = build_window_batch(aligned, grid_s)
    X = window_features(batch)
    report = assemble_quality_report(log, time_quality, grid_s, batch, X, max_gap_s)
    return report, X, batch, grid_s, aligned


def make_quality_report(
    timestamp_fn=timestamp_report, interpolation_fn=interpolate_with_gap_mask, *, max_gap_s=0.10
):
    """Return the JSON-compatible report, retaining the original callable API."""
    return process_sensor_log(timestamp_fn, interpolation_fn, max_gap_s=max_gap_s)[0]


def write_sensor_artifacts(output_dir, report, X, batch):
    """Write the report, feature rows, and the provenance of all candidate windows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "s2-quality.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    np.savetxt(
        output_dir / "s2-features.csv",
        np.column_stack((batch.start_s[batch.valid], X)),
        delimiter=",",
        header="window_start_s," + ",".join(FEATURE_NAMES),
        comments="",
    )
    np.savetxt(
        output_dir / "s2-windows.csv",
        np.column_stack((batch.start_s, batch.valid)),
        delimiter=",",
        header="window_start_s,valid",
        comments="",
        fmt=["%.9f", "%d"],
    )


def write_quality_report(output: Path) -> None:
    """Write CLI artifacts from the finished reference implementation."""
    report, X, batch, _, _ = process_sensor_log()
    write_sensor_artifacts(output.parent, report, X, batch)
    if output.name != "s2-quality.json":
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    write_quality_report(Path("reports/s2-quality.json"))
