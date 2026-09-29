"""Determinism and statistical acceptance checks for Phase 0 weather."""

import numpy as np

from tether.physics import constants
from tether.physics.plant import run_transit_log
from tether.physics.weather import (
    apply_ar1,
    ar1_95_interval,
    ar1_coefficient,
    estimate_ar1_coefficient,
    generate_weather,
    hill_tail_index,
    standardized_innovations,
)


def test_p0_t4_drake_logs_are_bit_identical_at_same_seed():
    sample_count = 20
    first_weather = generate_weather(4102, sample_count)
    second_weather = generate_weather(4102, sample_count)
    first = run_transit_log(
        constants.NOMINAL_THRUST, 0.12, constants.WEATHER_PERIOD, first_weather.states
    )
    second = run_transit_log(
        constants.NOMINAL_THRUST, 0.12, constants.WEATHER_PERIOD, second_weather.states
    )

    assert np.array_equal(first.truth, second.truth)
    assert np.array_equal(first.tension, second.tension)
    assert np.array_equal(first.marks, second.marks)


def test_p0_t8_weather_statistics():
    sample_count = 1_000_000
    gaussian = standardized_innovations(811, sample_count, 1, "gaussian", "local")[:, 0, 0]
    centered = gaussian - gaussian.mean()
    kurtosis = np.mean(centered**4) / np.mean(centered**2) ** 2
    assert abs(kurtosis - 3.0) <= 0.03

    student = standardized_innovations(812, sample_count, 1, "student_t3", "local")[:, 0, 0]
    tail_index = hill_tail_index(student, upper_order_count=5_000)
    assert 2.7 <= tail_index <= 3.3

    ar_samples = standardized_innovations(813, 300_000, 1, "gaussian", "local")
    states = apply_ar1(ar_samples)
    estimate = estimate_ar1_coefficient(states[1_000:])
    interval = ar1_95_interval(states[1_000:].size - 1, ar1_coefficient())
    assert interval[0] <= estimate <= interval[1]

    common = standardized_innovations(814, 10_000, 6, "gaussian", "common")
    for body in range(1, common.shape[1]):
        assert np.array_equal(common[:, 0], common[:, body])