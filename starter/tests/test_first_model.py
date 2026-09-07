"""The S1 report must retain notebook results instead of rebuilding a solution."""

import numpy as np

from ml_sau.course_case import PHYSICAL_FEATURES, generate_course_case
from ml_sau.first_model import build_preview_report, make_preview_model


def test_report_keeps_supplied_scores_and_decisions_without_training() -> None:
    case = generate_course_case()
    model = make_preview_model()
    # Deliberately disagree with score >= 0.5. Such a notebook mistake must remain
    # visible in its report so that comparison with the separate CLI can catch it.
    prediction = np.array([True, True, False])
    report = build_preview_report(
        case=case,
        features=PHYSICAL_FEATURES,
        data_seed=1126,
        threshold=0.5,
        model=model,
        flights=np.array(["EMA-00-F05", "EMA-00-F06", "EMA-01-F05"]),
        target=np.array([0, 1, 0]),
        score=np.array([0.2, 0.6, 0.9]),
        prediction=prediction,
        reduced_features=tuple(name for name in PHYSICAL_FEATURES if name != "temperature_c"),
        reduced_score=np.array([0.1, 0.7, 0.8]),
        reduced_prediction=np.array([False, True, False]),
        example_rows=[],
    )

    rows = report["flight_results"]
    assert [row["score"] for row in rows] == [0.2, 0.6, 0.9]
    assert [row["prediction"] for row in rows] == prediction.astype(int).tolist()
    assert report["feature_comparison"]["changed_flights"] == ["EMA-00-F05"]
    assert not hasattr(model[-1], "classes_")
