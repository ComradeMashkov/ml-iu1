import numpy as np

from ml_sau.course_case import course_holdout_masks, generate_course_case
from ml_sau.first_model import run_first_model
from ml_sau.sensor import build_windows
from ml_sau.sensor_source import generate_sensor_log


def test_sensor_source_contains_expected_timestamp_defects() -> None:
    log = generate_sensor_log()
    dt = np.diff(log.imu_t_s)

    assert np.sum(dt == 0) == 1
    assert np.max(dt) > 0.4


def test_course_split_keeps_flights_disjoint() -> None:
    case = generate_course_case()
    train, validation, test = course_holdout_masks(case)

    train_flights = set(case.flight_id[train])
    validation_flights = set(case.flight_id[validation])
    test_flights = set(case.flight_id[test])

    assert train_flights.isdisjoint(validation_flights)
    assert train_flights.isdisjoint(test_flights)
    assert validation_flights.isdisjoint(test_flights)
    assert np.all(train | validation | test)

    known_actuators = set(case.actuator_id[train]) | set(case.actuator_id[validation])
    test_actuators = set(case.actuator_id[test])
    assert known_actuators.isdisjoint(test_actuators)


def test_course_target_is_constant_within_flight() -> None:
    case = generate_course_case()

    for flight in np.unique(case.flight_id):
        assert np.unique(case.y[case.flight_id == flight]).size == 1


def test_prepared_window_builder_drops_non_finite_windows() -> None:
    values = np.arange(8, dtype=np.float64)
    values[3] = np.nan

    windows = build_windows(values, window_size=3, stride=2)

    assert windows.tolist() == [[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]]


def test_first_model_writes_preview_report(tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"features": ["current_rms_a", "temperature_c"], "data_seed": 1126}')
    output = tmp_path / "report.json"

    report = run_first_model(config, output)

    assert report["preview_only"] is True
    assert report["decision_unit"] == "flight"
    assert output.exists()
