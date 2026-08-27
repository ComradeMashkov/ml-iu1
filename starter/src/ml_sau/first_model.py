"""Working first classifier used in seminar S1.

The validation policy is provided here so that a beginner can inspect a complete
fit/predict loop before implementing the protocol in S3. The report is explicitly
marked as a preview and must not be used as the final course result.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_sau.course_case import PHYSICAL_FEATURES, course_holdout_masks, generate_course_case


def _flight_table(
    window_score: np.ndarray, target: np.ndarray, flight_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a compact flight table for the S1 preview."""

    flights = np.unique(flight_id)
    target_values = []
    score_values = []
    for flight in flights:
        flight_mask = flight_id == flight
        target_values.append(target[flight_mask][0])
        score_values.append(window_score[flight_mask].max())
    flight_target = np.asarray(target_values)
    flight_score = np.asarray(score_values)
    return flights, flight_target, flight_score


def run_first_model(config: Path, output: Path) -> dict[str, object]:
    """Train a logistic classifier and write a beginner-readable preview report."""

    settings = json.loads(config.read_text())
    selected = tuple(settings["features"])
    unknown = set(selected) - set(PHYSICAL_FEATURES)
    if unknown:
        raise ValueError(f"unknown physical features: {sorted(unknown)}")
    feature_columns = []
    for name in selected:
        feature_columns.append(PHYSICAL_FEATURES.index(name))
    feature_idx = np.asarray(feature_columns)

    case = generate_course_case(seed=int(settings.get("data_seed", 1126)))
    train, validation, _ = course_holdout_masks(case)
    X_selected = case.X[:, feature_idx]
    X_train = X_selected[train]
    y_train = case.y[train]
    X_validation = X_selected[validation]
    y_validation = case.y[validation]
    validation_flight_id = case.flight_id[validation]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    )
    model.fit(X_train, y_train)
    probability_table = model.predict_proba(X_validation)
    window_score = probability_table[:, 1]
    flights, target, score = _flight_table(window_score, y_validation, validation_flight_id)
    threshold = float(settings.get("demonstration_threshold", 0.8))
    prediction = score >= threshold
    error_flights = []
    for index in range(len(flights)):
        if prediction[index] != target[index]:
            error_flights.append(str(flights[index]))

    report: dict[str, object] = {
        "artifact": "s1-first-model",
        "preview_only": True,
        "why_preview_only": "Разбиение, метрика и порог формально изучаются в L1–S3.",
        "row_meaning": "20-second actuator window",
        "decision_unit": "flight",
        "features": list(selected),
        "target": "after-flight inspection flag",
        "model_family": "standardized logistic regression",
        "n_train_windows": int(train.sum()),
        "n_validation_flights": len(flights),
        "demonstration_threshold": threshold,
        "threshold_source": "instructor-provided illustration; selection is taught in L1-S3",
        "demonstration_accuracy": float(accuracy_score(target, prediction)),
        "example_error_flights": error_flights[:2],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/first-model.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/s1-first-model.json"))
    args = parser.parse_args()
    report = run_first_model(args.config, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
