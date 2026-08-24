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
    flight_target = np.asarray([target[flight_id == flight][0] for flight in flights])
    flight_score = np.asarray([window_score[flight_id == flight].max() for flight in flights])
    return flights, flight_target, flight_score


def run_first_model(config: Path, output: Path) -> dict[str, object]:
    """Train a logistic classifier and write a beginner-readable preview report."""

    settings = json.loads(config.read_text())
    selected = tuple(settings["features"])
    unknown = set(selected) - set(PHYSICAL_FEATURES)
    if unknown:
        raise ValueError(f"unknown physical features: {sorted(unknown)}")
    feature_idx = np.asarray([PHYSICAL_FEATURES.index(name) for name in selected])

    case = generate_course_case(seed=int(settings.get("data_seed", 1126)))
    train, validation, _ = course_holdout_masks(case)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    )
    model.fit(case.X[train][:, feature_idx], case.y[train])
    window_score = model.predict_proba(case.X[validation][:, feature_idx])[:, 1]
    flights, target, score = _flight_table(
        window_score, case.y[validation], case.flight_id[validation]
    )
    threshold = float(settings.get("demonstration_threshold", 0.8))
    prediction = score >= threshold
    error_idx = np.flatnonzero(prediction != target)

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
        "example_error_flights": flights[error_idx[:2]].tolist(),
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
