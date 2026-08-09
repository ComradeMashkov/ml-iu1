"""Synthetic multi-rate IMU/GNSS source for seminar S2.

The generated arrays are not real flight data. The defects are deliberate and
deterministic so that tests can assert the expected quality report.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

G0 = 9.80665


@dataclass(frozen=True)
class SensorLog:
    """A compact multi-rate sensor record with known timestamp defects."""

    imu_t_s: NDArray[np.float64]
    acceleration_g: NDArray[np.float64]
    angular_rate_dps: NDArray[np.float64]
    gnss_t_s: NDArray[np.float64]
    speed_mps: NDArray[np.float64]


def generate_sensor_log(seed: int = 2026) -> SensorLog:
    """Return a deterministic log with one duplicate and one long IMU gap."""

    rng = np.random.default_rng(seed)
    physical_t = np.arange(0.0, 30.0, 0.01)
    keep = ~((physical_t >= 17.8) & (physical_t < 18.25))
    physical_t = physical_t[keep]
    imu_t = physical_t + rng.normal(0.0, 0.0007, physical_t.size)
    duplicate = int(np.searchsorted(physical_t, 12.0))
    imu_t[duplicate] = imu_t[duplicate - 1]

    acceleration_g = np.column_stack(
        (
            0.08 * np.sin(2 * np.pi * 0.31 * physical_t),
            1.0 + 0.05 * np.sin(2 * np.pi * 0.17 * physical_t),
        )
    )
    acceleration_g += rng.normal(0.0, 0.006, acceleration_g.shape)
    angular_rate_dps = 17.0 * np.sin(2 * np.pi * 0.17 * physical_t)
    angular_rate_dps += rng.normal(0.0, 0.65, physical_t.size)

    gnss_physical_t = np.arange(0.0, 30.0, 0.1)
    gnss_t = 1.00022 * gnss_physical_t + 0.18
    speed = 42.0 + 2.4 * np.sin(2 * np.pi * 0.035 * gnss_physical_t)
    speed += rng.normal(0.0, 0.12, speed.size)

    return SensorLog(
        imu_t_s=imu_t,
        acceleration_g=acceleration_g,
        angular_rate_dps=angular_rate_dps,
        gnss_t_s=gnss_t,
        speed_mps=speed,
    )
