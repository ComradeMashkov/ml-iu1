import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from assets.course_case import (
    aggregate_flight_predictions,
    course_holdout_masks,
    generate_course_case,
)


def load_starter_course_case():
    """Load the copied student generator without changing the import path."""

    path = Path(__file__).parents[1] / "starter/src/ml_sau/course_case.py"
    spec = importlib.util.spec_from_file_location("starter_course_case", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_course_split_covers_rows_and_keeps_flights_disjoint() -> None:
    case = generate_course_case()
    train, validation, test = course_holdout_masks(case)

    assert np.all(train | validation | test)
    assert not np.any(train & validation)
    assert not np.any(train & test)
    assert not np.any(validation & test)

    flight_sets = [
        set(case.flight_id[train]),
        set(case.flight_id[validation]),
        set(case.flight_id[test]),
    ]
    assert flight_sets[0].isdisjoint(flight_sets[1])
    assert flight_sets[0].isdisjoint(flight_sets[2])
    assert flight_sets[1].isdisjoint(flight_sets[2])


def test_starter_generator_matches_lecture_generator() -> None:
    lecture = generate_course_case()
    starter = load_starter_course_case().generate_course_case()

    np.testing.assert_allclose(starter.X, lecture.X)
    np.testing.assert_array_equal(starter.y, lecture.y)
    np.testing.assert_array_equal(starter.flight_id, lecture.flight_id)


def test_aggregation_uses_maximum_and_sorted_flight_ids() -> None:
    probability = np.asarray([0.2, 0.8, 0.4, 0.3])
    target = np.asarray([1, 1, 0, 0])
    flight_id = np.asarray(["F2", "F2", "F1", "F1"])

    flights, flight_target, flight_probability = aggregate_flight_predictions(
        probability, target, flight_id
    )

    assert flights.tolist() == ["F1", "F2"]
    assert flight_target.tolist() == [0, 1]
    assert flight_probability.tolist() == pytest.approx([0.4, 0.8])


def test_aggregation_rejects_inconsistent_flight_target() -> None:
    with pytest.raises(ValueError, match="inconsistent targets"):
        aggregate_flight_predictions(
            np.asarray([0.2, 0.8]),
            np.asarray([0, 1]),
            np.asarray(["F1", "F1"]),
        )


def test_aggregation_rejects_non_finite_probability() -> None:
    with pytest.raises(ValueError, match="finite"):
        aggregate_flight_predictions(
            np.asarray([np.nan]),
            np.asarray([0]),
            np.asarray(["F1"]),
        )
