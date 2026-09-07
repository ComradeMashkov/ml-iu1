"""Executable S2 contracts: clocks, gap endpoints, provenance, and feature rows."""

import json

import numpy as np
import pytest

from ml_sau.sensor import (
    GAP_TOLERANCE_S,
    build_window_batch,
    interpolate_with_gap_mask,
    prepare_sensor_channels,
    process_sensor_log,
    timestamp_report,
    window_features,
    write_quality_report,
)
from ml_sau.sensor_source import generate_sensor_log


@pytest.mark.parametrize(
    ("time", "expected"),
    [
        ([], (0, 0, 0)),
        ([1.0], (0, 0, 0)),
        ([np.nan], (1, 0, 0)),
        ([np.inf], (1, 0, 0)),
        ([0.0, 0.0, -1.0, np.nan, np.inf, np.inf], (3, 1, 1)),
    ],
)
def test_timestamp_contract(time, expected):
    original = np.array(time)
    before = original.copy()
    report = timestamp_report(original)
    assert tuple(report.values()) == expected
    assert all(type(value) is int for value in report.values())
    np.testing.assert_equal(original, before)
    json.dumps(report)


def test_timestamp_rejects_matrix():
    with pytest.raises(ValueError):
        timestamp_report(np.zeros((2, 2)))


def test_gap_preserves_measured_endpoints_and_forbids_extrapolation():
    t = np.array([0.0, 2.0])
    values = np.array([0.0, 4.0])
    grid = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        interpolate_with_gap_mask(t, values, grid, 2.0),
        [np.nan, 0.0, 2.0, 4.0, np.nan],
    )
    np.testing.assert_allclose(
        interpolate_with_gap_mask(t, values, grid, 0.5),
        [np.nan, 0.0, np.nan, 4.0, np.nan],
    )


def test_multichannel_gap_uses_one_time_mask():
    result = interpolate_with_gap_mask(
        np.array([0.0, 0.1, 0.5]),
        np.array([[0.0, 0.0], [1.0, 2.0], [5.0, 10.0]]),
        np.array([0.05, 0.3, 0.5]),
        0.1,
    )
    np.testing.assert_allclose(result, [[0.5, 1.0], [np.nan, np.nan], [5.0, 10.0]])


@pytest.mark.parametrize("time", [[], [0.0, 0.0], [1.0, 0.0], [0.0, np.nan], [np.inf]])
def test_interpolation_requires_finite_strict_source_time(time):
    with pytest.raises(ValueError):
        interpolate_with_gap_mask(np.array(time), np.zeros(len(time)), np.array([0.0]), 0.1)


@pytest.mark.parametrize("limit", [0.0, -1.0, np.nan, np.inf])
def test_interpolation_rejects_invalid_gap_limit(limit):
    with pytest.raises(ValueError):
        interpolate_with_gap_mask(np.array([0.0, 1.0]), np.zeros(2), np.zeros(1), limit)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        interpolate_with_gap_mask(np.array([0.0, 1.0]), np.zeros(3), np.zeros(1), 0.1)


def test_gnss_clock_conversion_and_roundoff_tolerance():
    log = generate_sensor_log()
    _, imu_si, gnss_t, grid = prepare_sensor_channels(log)
    np.testing.assert_allclose(gnss_t, np.arange(301) * 0.1, atol=1e-13)
    assert imu_si.shape[1] == 3
    assert len(grid) == 1500
    speed = interpolate_with_gap_mask(gnss_t, log.speed_mps, grid, 0.1)
    assert np.isfinite(speed).all()
    # The tolerance must not silently admit a materially longer interval.
    t = np.array([0.0, 0.1 + 100 * GAP_TOLERANCE_S])
    assert np.isnan(interpolate_with_gap_mask(t, t, np.array([0.05]), 0.1)[0])


def test_windows_preserve_starts_even_when_some_are_invalid():
    values = np.arange(8, dtype=float)
    values[3] = np.nan
    batch = build_window_batch(values, 10.0 + np.arange(8) * 0.02, 3, 2)
    np.testing.assert_allclose(batch.start_s, [10.0, 10.04, 10.08])
    np.testing.assert_array_equal(batch.valid, [True, False, True])
    np.testing.assert_allclose(batch.values[batch.valid], [[0, 1, 2], [4, 5, 6]])
    short = build_window_batch(np.zeros((2, 4)), np.array([0.0, 0.02]), 3, 1)
    assert short.values.shape == (0, 3, 4)
    assert short.start_s.size == short.valid.size == 0


def test_feature_values_units_and_valid_row_order():
    values = np.array([[3, 9, 1, 40], [4, 11, 3, 44], [np.nan, 0, 0, 0], [0, 0, 0, 0]])
    batch = build_window_batch(values, np.arange(4, dtype=float), 2, 2)
    X = window_features(batch)
    np.testing.assert_allclose(X, [[np.sqrt(12.5), 10.0, 1.0, 42.0]])
    np.testing.assert_allclose(batch.start_s[batch.valid], [0.0])


def test_two_gap_variants_change_only_affected_windows():
    strict, X, batch, _, aligned = process_sensor_log(max_gap_s=0.1)
    relaxed, relaxed_X, relaxed_batch, _, _ = process_sensor_log(max_gap_s=0.5)
    assert X.shape == (26, 4)
    assert relaxed_X.shape == (29, 4)
    assert aligned.shape == (1500, 4)
    assert strict["discarded_window_start_s"] == [16.0, 17.0, 18.0]
    assert relaxed["discarded_window_start_s"] == []
    np.testing.assert_allclose(X, relaxed_X[batch.valid])
    np.testing.assert_allclose(batch.start_s, relaxed_batch.start_s)


def test_cli_artifacts_preserve_feature_provenance(tmp_path):
    write_quality_report(tmp_path / "s2-quality.json")
    report = json.loads((tmp_path / "s2-quality.json").read_text())
    features = np.loadtxt(tmp_path / "s2-features.csv", delimiter=",", skiprows=1)
    windows = np.loadtxt(tmp_path / "s2-windows.csv", delimiter=",", skiprows=1)
    assert features.shape == (26, 5)
    assert windows.shape == (29, 2)
    np.testing.assert_allclose(features[:, 0], report["valid_window_start_s"])
    np.testing.assert_allclose(windows[windows[:, 1] == 0, 0], [16.0, 17.0, 18.0])
