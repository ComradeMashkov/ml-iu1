"""Student implementation target for seminar S2."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def timestamp_report(t_s: NDArray[np.float64]) -> dict[str, int]:
    """Count non-finite, duplicate, and backward timestamps."""

    raise NotImplementedError("Implement on S2")


def interpolate_with_gap_mask(
    t_s: NDArray[np.float64],
    values: NDArray[np.float64],
    grid_s: NDArray[np.float64],
    max_gap_s: float,
) -> NDArray[np.float64]:
    """Interpolate short spans and return NaN across gaps or outside the input."""

    raise NotImplementedError("Implement on S2")


def build_windows(
    values: NDArray[np.float64], window_size: int, stride: int
) -> NDArray[np.float64]:
    """Build fixed-length windows and omit those containing non-finite values."""

    raise NotImplementedError("Implement on S2")


def write_quality_report(output: Path) -> None:
    """Run the S2 pipeline and write its deterministic JSON report."""

    raise NotImplementedError("Implement on S2")


if __name__ == "__main__":
    write_quality_report(Path("reports/s2-quality.json"))
