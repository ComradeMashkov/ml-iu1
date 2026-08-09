"""Student implementation target for seminar S3."""

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def aggregate_flight_predictions(
    probability: NDArray[np.float64],
    target: NDArray[np.int64],
    flight_id: NDArray[np.str_],
) -> tuple[NDArray[np.str_], NDArray[np.int64], NDArray[np.float64]]:
    """Aggregate window probabilities and validate one target per flight."""

    raise NotImplementedError("Implement on S3")


def select_threshold(
    target: NDArray[np.int64], probability: NDArray[np.float64], min_recall: float
) -> float:
    """Choose the highest-precision validation point satisfying minimum recall."""

    raise NotImplementedError("Implement on S3")


def run_baseline(config: Path, output: Path, *, evaluate_test: bool = False) -> None:
    """Train the S3 baseline and save the split, model, and metrics."""

    raise NotImplementedError("Implement on S3")


def main() -> None:
    """Parse the stable command-line interface used in the seminar."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/runs/s3-baseline"))
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    run_baseline(args.config, args.output, evaluate_test=args.evaluate_test)


if __name__ == "__main__":
    main()
