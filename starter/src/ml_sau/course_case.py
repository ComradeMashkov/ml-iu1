"""Synthetic actuator-health table used on L1 and S3.

Rows are overlapping windows. The maintenance target is defined once per flight
and repeated for all of its windows. This generator matches the one used to
build the lecture figures.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FEATURE_NAMES = (
    "tracking_rmse_deg",
    "current_rms_a",
    "temperature_c",
    "vibration_rms_g",
    "load_rms_kn",
    "supply_voltage_v",
    "flight_signature",
)

PHYSICAL_FEATURES = FEATURE_NAMES[:-1]


@dataclass(frozen=True)
class ActuatorCase:
    """Window features and group identifiers needed for evaluation."""

    X: NDArray[np.float64]
    y: NDArray[np.int64]
    feature_names: tuple[str, ...]
    flight_id: NDArray[np.str_]
    actuator_id: NDArray[np.str_]
    window_start_s: NDArray[np.float64]
    flight_order: NDArray[np.int64]
    regime: NDArray[np.str_]


def _sigmoid(value: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(value)))


def generate_course_case(
    *,
    n_actuators: int = 24,
    flights_per_actuator: int = 7,
    windows_per_flight: int = 24,
    seed: int = 1126,
) -> ActuatorCase:
    """Generate a reproducible actuator monitoring dataset."""

    rng = np.random.default_rng(seed)
    rows: list[list[float]] = []
    labels: list[int] = []
    flight_ids: list[str] = []
    actuator_ids: list[str] = []
    window_starts: list[float] = []
    flight_orders: list[int] = []
    regimes: list[str] = []

    regime_names = np.array(["cruise", "maneuver", "approach"])
    regime_load = {"cruise": 0.25, "maneuver": 1.00, "approach": 0.55}

    for actuator_idx in range(n_actuators):
        actuator = f"EMA-{actuator_idx:02d}"
        manufacture_bias = rng.normal(0.0, 0.16)
        wear_rate = np.clip(rng.normal(0.12, 0.025), 0.07, 0.19)
        cooling_bias = rng.normal(0.0, 1.8)

        for flight_idx in range(flights_per_actuator):
            flight = f"{actuator}-F{flight_idx:02d}"
            regime = str(regime_names[(actuator_idx + 2 * flight_idx) % len(regime_names)])
            load_level = regime_load[regime]
            wear = np.clip(
                0.12
                + wear_rate * flight_idx
                + 0.08 * load_level
                + manufacture_bias
                + rng.normal(0.0, 0.035),
                0.02,
                1.20,
            )
            flight_signature = rng.uniform(-2.5, 2.5)
            flight_effect = rng.normal(0.0, 0.65)
            event_logit = -4.2 + 3.8 * wear + load_level + flight_effect
            flight_degraded = int(rng.random() < _sigmoid(event_logit))

            for window_idx in range(windows_per_flight):
                phase = window_idx / max(1, windows_per_flight - 1)
                local_load = load_level + 0.18 * np.sin(2 * np.pi * phase)
                local_load += rng.normal(0.0, 0.07)
                local_wear = wear + 0.05 * phase

                tracking = 0.20 + 0.75 * local_wear + 0.18 * local_load
                tracking += rng.normal(0.0, 0.13)
                current = 6.2 + 4.5 * local_load + 4.0 * local_wear
                current += rng.normal(0.0, 0.75)
                temperature = (
                    34.0
                    + cooling_bias
                    + 12.0 * local_load
                    + 15.5 * local_wear
                    + 2.5 * phase
                    + rng.normal(0.0, 2.0)
                )
                vibration = 0.018 + 0.026 * local_load + 0.052 * local_wear
                vibration += rng.normal(0.0, 0.010)
                load_rms = 2.5 + 7.0 * local_load + rng.normal(0.0, 0.7)
                voltage = 270.0 - 1.8 * local_load - 1.4 * local_wear
                voltage += rng.normal(0.0, 0.9)

                rows.append(
                    [
                        tracking,
                        current,
                        temperature,
                        vibration,
                        load_rms,
                        voltage,
                        flight_signature,
                    ]
                )
                labels.append(flight_degraded)
                flight_ids.append(flight)
                actuator_ids.append(actuator)
                window_starts.append(window_idx * 10.0)
                flight_orders.append(flight_idx)
                regimes.append(regime)

    return ActuatorCase(
        X=np.asarray(rows, dtype=np.float64),
        y=np.asarray(labels, dtype=np.int64),
        feature_names=FEATURE_NAMES,
        flight_id=np.asarray(flight_ids),
        actuator_id=np.asarray(actuator_ids),
        window_start_s=np.asarray(window_starts, dtype=np.float64),
        flight_order=np.asarray(flight_orders, dtype=np.int64),
        regime=np.asarray(regimes),
    )


def course_holdout_masks(
    case: ActuatorCase,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Split known actuators by time and reserve six unseen actuators for test."""

    actuator_number = np.asarray(
        [int(value.split("-")[1]) for value in case.actuator_id], dtype=np.int64
    )
    known = actuator_number < 18
    train = known & (case.flight_order <= 4)
    validation = known & (case.flight_order >= 5)
    test = ~known
    return train, validation, test
