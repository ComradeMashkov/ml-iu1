import numpy as np

from ml_sau.course_case import course_holdout_masks, generate_course_case
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


def test_course_target_is_constant_within_flight() -> None:
    case = generate_course_case()

    for flight in np.unique(case.flight_id):
        assert np.unique(case.y[case.flight_id == flight]).size == 1
