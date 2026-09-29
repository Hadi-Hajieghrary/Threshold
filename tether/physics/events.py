"""Crossing interpolation and the synthetic Phase 0 benchmark."""

from __future__ import annotations

import numpy as np


def linear_crossing_time(
    first_time: float,
    first_value: float,
    second_time: float,
    second_value: float,
) -> float:
    """Interpolate a zero crossing between samples with opposite signs."""
    if first_value == second_value:
        raise ValueError("crossing interpolation requires distinct sample values")
    fraction = -first_value / (second_value - first_value)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("samples do not bracket a crossing")
    return first_time + fraction * (second_time - first_time)


def linear_crossing_speed(
    first_value: float,
    second_value: float,
    first_speed: float,
    second_speed: float,
) -> float:
    """Interpolate crossing speed using the elongation crossing fraction."""
    if first_value == second_value:
        raise ValueError("crossing interpolation requires distinct sample values")
    fraction = -first_value / (second_value - first_value)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("samples do not bracket a crossing")
    return first_speed + fraction * (second_speed - first_speed)


def synthetic_crossing_benchmark() -> dict[str, object]:
    """Measure convergence on a curved signal with a non-grid-aligned root."""
    root = 0.373
    grid_fraction = 0.37
    lower_indices = np.array([10, 20, 40, 80], dtype=int)
    step_sizes = root / (lower_indices + grid_fraction)
    time_errors = []
    speed_errors = []

    def signal(time):
        offset = time - root
        return offset + 0.8 * offset**2 + 0.3 * offset**3

    def signal_rate(time):
        offset = time - root
        return 1.0 + 1.6 * offset + 0.9 * offset**2

    for lower_index, step_size in zip(lower_indices, step_sizes):
        first_time = lower_index * step_size
        second_time = (lower_index + 1) * step_size
        first_value = signal(first_time)
        second_value = signal(second_time)
        crossing_time = linear_crossing_time(
            first_time, first_value, second_time, second_value
        )
        crossing_speed = linear_crossing_speed(
            first_value,
            second_value,
            signal_rate(first_time),
            signal_rate(second_time),
        )
        time_errors.append(abs(crossing_time - root))
        speed_errors.append(abs(crossing_speed - signal_rate(root)) / abs(signal_rate(root)))

    observed_order = float(
        np.polyfit(np.log(step_sizes), np.log(np.asarray(time_errors)), 1)[0]
    )
    return {
        "root": root,
        "step_sizes": step_sizes.tolist(),
        "time_errors": time_errors,
        "speed_relative_errors": speed_errors,
        "time_order": observed_order,
        "finest_speed_relative_error": speed_errors[-1],
    }