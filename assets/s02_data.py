"""Deterministic synthetic multi-rate flight log used by seminar S2.

The log deliberately contains the defects discussed in the seminar:
different units and sample rates, timestamp jitter, one duplicate, one gap,
GNSS clock offset, and linear clock drift.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

G0 = 9.80665


@dataclass(frozen=True)
class SensorLog:
    """A small, self-contained multi-sensor telemetry record."""

    imu_t_s: NDArray[np.float64]
    accel_g: NDArray[np.float64]
    gyro_dps: NDArray[np.float64]
    gnss_t_s: NDArray[np.float64]
    gnss_yaw_rate_dps: NDArray[np.float64]
    gnss_speed_mps: NDArray[np.float64]
    clock_offset_s: float
    clock_drift: float
    gap_start_s: float
    gap_end_s: float


def yaw_rate_dps(t_s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Smooth but distinctive maneuver signal shared by IMU and GNSS."""

    base = 17.0 * np.sin(2 * np.pi * 0.17 * t_s)
    detail = 6.0 * np.sin(2 * np.pi * 0.47 * t_s + 0.6)
    chirp = 3.0 * np.sin(2 * np.pi * (0.025 * t_s + 0.00055 * t_s**2))
    turns = np.zeros_like(t_s)
    for center, amplitude, width in (
        (18.0, 9.0, 1.6),
        (36.0, 12.0, 2.2),
        (58.0, -10.0, 1.8),
        (83.0, 8.0, 2.6),
        (112.0, -11.0, 3.4),
        (139.0, 10.0, 2.0),
        (161.0, -8.0, 1.7),
    ):
        turns += amplitude * np.exp(-0.5 * ((t_s - center) / width) ** 2)
    return base + detail + chirp + turns


def speed_mps(t_s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reference airframe speed used for the low-rate GNSS channel."""

    return 42.0 + 2.4 * np.sin(2 * np.pi * 0.035 * t_s) + 0.8 * np.sin(2 * np.pi * 0.11 * t_s + 0.3)


def generate_sensor_log(
    duration_s: float = 180.0,
    seed: int = 2026,
) -> SensorLog:
    """Generate an IMU/GNSS record with known, reproducible defects."""

    rng = np.random.default_rng(seed)

    imu_physical_t = np.arange(0.0, duration_s, 0.01)
    imu_t = imu_physical_t + rng.normal(0.0, 0.0007, imu_physical_t.size)

    gap_start_s, gap_end_s = 17.8, 18.25
    keep = ~((imu_physical_t >= gap_start_s) & (imu_physical_t < gap_end_s))
    imu_physical_t = imu_physical_t[keep]
    imu_t = imu_t[keep]

    duplicate_idx = int(np.searchsorted(imu_physical_t, 12.0))
    imu_t[duplicate_idx] = imu_t[duplicate_idx - 1]

    gyro = yaw_rate_dps(imu_physical_t) + rng.normal(0.0, 0.65, imu_physical_t.size)
    accel_x_mps2 = (
        0.75 * np.sin(2 * np.pi * 0.31 * imu_physical_t)
        + 0.22 * np.sin(2 * np.pi * 1.8 * imu_physical_t)
        + rng.normal(0.0, 0.06, imu_physical_t.size)
    )
    accel_z_mps2 = (
        G0
        + 0.55 * np.sin(2 * np.pi * 0.17 * imu_physical_t + 0.2)
        + rng.normal(0.0, 0.08, imu_physical_t.size)
    )
    accel_g = np.column_stack((accel_x_mps2 / G0, accel_z_mps2 / G0))

    gnss_physical_t = np.arange(0.0, duration_s, 0.1)
    clock_offset_s = 0.18
    clock_drift = 220e-6
    gnss_t = (
        (1.0 + clock_drift) * gnss_physical_t
        + clock_offset_s
        + rng.normal(0.0, 0.0025, gnss_physical_t.size)
    )
    gnss_yaw = yaw_rate_dps(gnss_physical_t) + rng.normal(0.0, 0.45, gnss_physical_t.size)
    gnss_speed = speed_mps(gnss_physical_t) + rng.normal(0.0, 0.12, gnss_physical_t.size)

    return SensorLog(
        imu_t_s=imu_t,
        accel_g=accel_g,
        gyro_dps=gyro,
        gnss_t_s=gnss_t,
        gnss_yaw_rate_dps=gnss_yaw,
        gnss_speed_mps=gnss_speed,
        clock_offset_s=clock_offset_s,
        clock_drift=clock_drift,
        gap_start_s=gap_start_s,
        gap_end_s=gap_end_s,
    )
