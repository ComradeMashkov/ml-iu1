"""Reproducible S1 demonstration with results supplied by the notebook or CLI.

The split and threshold are teaching inputs. Neither is selected by this module.
Every row is synthetic; the target records a simulated inspection assignment,
not a verified component fault.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_sau.course_case import (
    PHYSICAL_FEATURES,
    ActuatorCase,
    course_holdout_masks,
    generate_course_case,
)


@dataclass(frozen=True)
class PreviewSplit:
    """Arrays used in the first fit/predict demonstration."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    validation_flight_id: np.ndarray


def prepare_preview_split(case: ActuatorCase, features: tuple[str, ...]) -> PreviewSplit:
    """Select named physical features and return the supplied S1 split."""

    unknown = set(features) - set(PHYSICAL_FEATURES)
    if unknown:
        raise ValueError(f"unknown physical features: {sorted(unknown)}")
    if not features:
        raise ValueError("at least one physical feature is required")
    columns = [case.feature_names.index(name) for name in features]
    train, validation, _ = course_holdout_masks(case)
    X = case.X[:, columns]
    return PreviewSplit(
        X_train=X[train],
        y_train=case.y[train],
        X_validation=X[validation],
        y_validation=case.y[validation],
        validation_flight_id=case.flight_id[validation],
    )


def make_preview_model() -> Pipeline:
    """The same explicitly displayed model as in the S1 notebook."""

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def aggregate_preview_scores(
    window_score: np.ndarray, target: np.ndarray, flight_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one historical target and the maximum score for each flight."""

    flights = np.unique(flight_id)
    flight_target = np.empty(len(flights), dtype=int)
    flight_score = np.empty(len(flights))
    for index, flight in enumerate(flights):
        mask = flight_id == flight
        flight_target[index] = target[mask][0]
        flight_score[index] = window_score[mask].max()
    return flights, flight_target, flight_score


def preview_flight_scores(
    case: ActuatorCase, features: tuple[str, ...]
) -> tuple[Pipeline, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the S1 model and return one score per validation flight."""

    split = prepare_preview_split(case, features)
    model = make_preview_model()
    model.fit(split.X_train, split.y_train)
    window_score = model.predict_proba(split.X_validation)[:, 1]
    flights, target, score = aggregate_preview_scores(
        window_score, split.y_validation, split.validation_flight_id
    )
    return model, flights, target, score


def _counts(target: np.ndarray, prediction: np.ndarray) -> dict[str, int]:
    return {
        "TP": int(np.sum((target == 1) & prediction)),
        "TN": int(np.sum((target == 0) & ~prediction)),
        "FP": int(np.sum((target == 0) & prediction)),
        "FN": int(np.sum((target == 1) & ~prediction)),
    }


def build_preview_report(
    *,
    case: ActuatorCase,
    features: tuple[str, ...],
    data_seed: int,
    threshold: float,
    model: Pipeline,
    flights: np.ndarray,
    target: np.ndarray,
    score: np.ndarray,
    prediction: np.ndarray,
    reduced_features: tuple[str, ...],
    reduced_score: np.ndarray,
    reduced_prediction: np.ndarray,
    example_rows: list[int],
) -> dict[str, object]:
    """Package the actual notebook arrays without fitting or predicting again.

    Predictions are retained as supplied, so the comparison with the CLI detects
    mistakes in the notebook's classification as well as in its model scores.
    """

    train, validation, _ = course_holdout_masks(case)
    columns = [case.feature_names.index(name) for name in features]
    examples = []
    for row in example_rows:
        if not validation[row]:
            raise ValueError("an inspected example must belong to validation")
        flight = case.flight_id[row]
        index = int(np.flatnonzero(flights == flight)[0])
        values = case.X[row, columns]
        standardized = model[0].transform(values[None, :])[0]
        label = "TP" if target[index] == 1 else "FP"
        if not prediction[index]:
            label = "FN" if target[index] == 1 else "TN"
        examples.append(
            {
                "flight_id": str(flight),
                "result": label,
                "target": int(target[index]),
                "prediction": int(prediction[index]),
                "flight_score": float(score[index]),
                "window_row": int(row),
                "window_start_s": float(case.window_start_s[row]),
                "window_features": values.tolist(),
                "standardized_window_features": standardized.tolist(),
            }
        )

    return {
        "artifact": "s1-first-model",
        "preview_only": True,
        "why_preview_only": "Синтетические данные; разбиение и порог предоставлены для разбора.",
        "data_source": "synthetic actuator windows, not NASA flight measurements",
        "data_seed": data_seed,
        "row_meaning": "20-second actuator window, step 10 seconds",
        "decision_unit": "flight",
        "features": list(features),
        "target": "simulated historical after-flight inspection assignment",
        "model_family": "standardized logistic regression, no class weighting",
        "n_train_windows": int(train.sum()),
        "n_validation_flights": len(flights),
        "split_policy": "actuators 00-17: flights 00-04 train, 05-06 validation; 18-23 test",
        "demonstration_threshold": threshold,
        "threshold_source": "instructor-provided illustration, not selected for deployment",
        "demonstration_accuracy": float(np.mean(prediction == target)),
        "confusion_counts": _counts(target, prediction),
        "always_zero_accuracy": float(np.mean(target == 0)),
        "feature_comparison": {
            "reduced_features": list(reduced_features),
            "reduced_accuracy": float(np.mean(reduced_prediction == target)),
            "reduced_confusion_counts": _counts(target, reduced_prediction),
            "changed_flights": flights[prediction != reduced_prediction].tolist(),
        },
        "flight_results": [
            {
                "flight_id": str(flight),
                "target": int(target[index]),
                "score": float(score[index]),
                "prediction": int(prediction[index]),
                "reduced_score": float(reduced_score[index]),
                "reduced_prediction": int(reduced_prediction[index]),
            }
            for index, flight in enumerate(flights)
        ],
        "example_error_flights": flights[prediction != target][:2].tolist(),
        "inspected_examples": examples,
    }


def calculate_first_model(settings: dict[str, object]) -> dict[str, object]:
    """Execute the same six-stage experiment as the completed S1 notebook."""

    features = tuple(settings["features"])
    data_seed = int(settings.get("data_seed", 1126))
    threshold = float(settings.get("demonstration_threshold", 0.5))
    case = generate_course_case(seed=data_seed)
    split = prepare_preview_split(case, features)
    model = make_preview_model()
    model.fit(split.X_train, split.y_train)
    window_score = model.predict_proba(split.X_validation)[:, 1]
    flights, target, score = aggregate_preview_scores(
        window_score, split.y_validation, split.validation_flight_id
    )
    prediction = score >= threshold

    reduced_features = tuple(name for name in features if name != "temperature_c")
    if not reduced_features:
        raise ValueError("the comparison requires a feature besides temperature_c")
    keep = [name != "temperature_c" for name in features]
    reduced_model = make_preview_model()
    reduced_model.fit(split.X_train[:, keep], split.y_train)
    reduced_window_score = reduced_model.predict_proba(split.X_validation[:, keep])[:, 1]
    reduced_flights, reduced_target, reduced_score = aggregate_preview_scores(
        reduced_window_score, split.y_validation, split.validation_flight_id
    )
    np.testing.assert_array_equal(flights, reduced_flights)
    np.testing.assert_array_equal(target, reduced_target)
    reduced_prediction = reduced_score >= threshold

    # The fixed lesson contains both FP and TN. Other valid feature configurations
    # may have different error types, so retain real examples without inventing any.
    error_indices = np.flatnonzero((target == 0) & prediction)
    if not len(error_indices):
        error_indices = np.flatnonzero(prediction != target)
    correct_indices = np.flatnonzero((target == 0) & ~prediction)
    if not len(correct_indices):
        correct_indices = np.flatnonzero(prediction == target)
    inspected = list(error_indices[:1]) + list(correct_indices[:1])
    _, validation, _ = course_holdout_masks(case)
    validation_rows = np.flatnonzero(validation)
    example_rows = []
    for index in inspected:
        local_rows = np.flatnonzero(split.validation_flight_id == flights[index])
        local_peak = local_rows[np.argmax(window_score[local_rows])]
        example_rows.append(int(validation_rows[local_peak]))

    return build_preview_report(
        case=case,
        features=features,
        data_seed=data_seed,
        threshold=threshold,
        model=model,
        flights=flights,
        target=target,
        score=score,
        prediction=prediction,
        reduced_features=reduced_features,
        reduced_score=reduced_score,
        reduced_prediction=reduced_prediction,
        example_rows=example_rows,
    )


def run_first_model(config: Path, output: Path) -> dict[str, object]:
    """Reproduce and save the S1 experiment from its fixed configuration."""

    report = calculate_first_model(json.loads(config.read_text(encoding="utf-8")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/first-model.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/s1-first-model.json"))
    args = parser.parse_args()
    print(json.dumps(run_first_model(args.config, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
