"""Reproducible S3 baseline with two focused student functions."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_sau.course_case import PHYSICAL_FEATURES, course_holdout_masks, generate_course_case


def aggregate_flight_scores(
    window_score: NDArray[np.float64],
    target: NDArray[np.int64],
    flight_id: NDArray[np.str_],
) -> tuple[NDArray[np.str_], NDArray[np.int64], NDArray[np.float64]]:
    """Apply the documented max-score heuristic and validate one target per flight."""

    raise NotImplementedError("Implement on S3")


def select_threshold(
    target: NDArray[np.int64], score: NDArray[np.float64], min_recall: float
) -> float:
    """Choose the highest-precision point satisfying recall; break ties by threshold."""

    raise NotImplementedError("Implement on S3")


def _scores(estimator: object, case: object, mask: NDArray[np.bool_], feature_idx: np.ndarray):
    window_score = estimator.predict_proba(case.X[mask][:, feature_idx])[:, 1]
    return aggregate_flight_scores(window_score, case.y[mask], case.flight_id[mask])


def run_baseline(config: Path, output: Path) -> None:
    """Train the S3 baseline and save split, model, and validation artifacts."""

    settings = json.loads(config.read_text())
    if settings["aggregation"] != "max_window_score_heuristic":
        raise ValueError("S3 supports only the documented max-score heuristic")

    case = generate_course_case(seed=int(settings["data_seed"]))
    train, validation, test = course_holdout_masks(case)
    feature_idx = np.arange(len(PHYSICAL_FEATURES))
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(case.X[train][:, feature_idx], case.y[train])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    )
    model.fit(case.X[train][:, feature_idx], case.y[train])

    flights, target, score = _scores(model, case, validation, feature_idx)
    _, dummy_target, dummy_score = _scores(dummy, case, validation, feature_idx)
    np.testing.assert_array_equal(target, dummy_target)
    threshold = select_threshold(target, score, float(settings["min_recall"]))
    prediction = score >= threshold

    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    split_manifest = {
        "decision_unit": "flight",
        "dependence_group": "flight_id",
        "validation_policy": "future_flights_of_known_actuators",
        "test_policy": "unseen_actuators",
        "deployment_group": "actuator_id",
        "time_order_key": "flight_order",
        "split_version": settings["split_version"],
        "counts_windows": {
            "train": int(train.sum()),
            "validation": int(validation.sum()),
            "test": int(test.sum()),
        },
    }
    (output / "split.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n"
    )
    metrics = {
        "unit": "flight",
        "score_semantics": "model score; not a calibrated flight probability",
        "aggregation": settings["aggregation"],
        "average_precision": float(average_precision_score(target, score)),
        "dummy_average_precision": float(average_precision_score(target, dummy_score)),
        "threshold": threshold,
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(target, prediction, labels=[0, 1])
        .ravel()
        .tolist(),
        "n_flights": len(flights),
    }
    (output / "validation-metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    joblib.dump(model, output / "model.joblib")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/runs/s3-baseline"))
    args = parser.parse_args()
    run_baseline(args.config, args.output)


if __name__ == "__main__":
    main()
