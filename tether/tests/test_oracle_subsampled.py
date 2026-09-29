"""Oracle a_hat on the estimator's 10 ms grid, and the model-selectable tick hazard."""

from __future__ import annotations

import numpy as np

from tether.monitor.hazard import hazard_generator, precursor_state, tick_hazard
from tether.monitor.linearized import linearized_hazard_mc, model_tick_hazard
from tether.monitor.oracle import SIGMA_A_FLOOR, oracle_precursor, subsampled_acceleration


def _series(onset: float = 1.004, acceleration: float = 3.0, end: float = 3.0):
    """1 ms truth of one cable: taut until ``onset``, then e' = -0.5 + a (t - onset)."""
    time = np.round(np.arange(0.0, end + 1e-9, 1e-3), 9)
    rate = np.where(time < onset, 0.0, -0.5 + acceleration * (time - onset))
    elongation = np.where(time < onset, 0.01, -1e-3 - 0.1 * (time - onset))
    return time, elongation[:, None], rate[:, None]


def test_slope_is_exact_on_the_10ms_grid():
    time, elongation, rate = _series()
    ticks = np.round(np.arange(0.0, 3.0 + 1e-9, 0.1), 9)
    slack = (np.interp(ticks, time, elongation[:, 0]) <= 0.0)[:, None]
    a_hat, sigma_a = subsampled_acceleration(ticks, slack, time, elongation, rate, a0=9.9)
    live = slack[:, 0]
    assert np.all(np.isnan(a_hat[~live, 0]))
    # First slack tick (1.1 s) already has ten 10 ms samples since the 1.004 s onset.
    np.testing.assert_allclose(a_hat[live, 0], 3.0, rtol=1e-9)
    np.testing.assert_allclose(sigma_a[live, 0], SIGMA_A_FLOOR)


def test_fallback_before_three_samples():
    # Onset 1.085 s: at the 1.1 s tick only the 1.09 and 1.10 grid samples are in the interval.
    time, elongation, rate = _series(onset=1.085)
    ticks = np.array([1.0, 1.1, 1.2])
    slack = np.array([[False], [True], [True]])
    a_hat, sigma_a = subsampled_acceleration(ticks, slack, time, elongation, rate, a0=9.9)
    assert a_hat[1, 0] == 9.9 and sigma_a[1, 0] == SIGMA_A_FLOOR
    np.testing.assert_allclose(a_hat[2, 0], 3.0, rtol=1e-9)


def test_oracle_precursor_series_option():
    time, elongation, rate = _series()
    ticks = np.round(np.arange(0.0, 3.0 + 1e-9, 0.1), 9)
    index = np.round(ticks / 1e-3).astype(int)
    slack = elongation[index] <= 0.0
    onset = np.where(slack, 1.004, np.nan)
    output = oracle_precursor(ticks, slack, onset, elongation[index], rate[index], series=(time, elongation, rate))
    np.testing.assert_allclose(output.a_hat[slack], 3.0, rtol=1e-9)


def test_model_tick_hazard_dispatch():
    sigma = np.diag([1e-4, 1e-3])
    args = (-0.05, 0.4, sigma, 2.0, 0.1, 0.5)
    constant = model_tick_hazard("constant_acceleration", *args, hazard_generator(3, 0), n_samples=512)
    assert constant == tick_hazard(*args, hazard_generator(3, 0), n_samples=512)
    mean, covariance = precursor_state(*args[:5])
    linear = model_tick_hazard("linearized", *args, hazard_generator(3, 0), n_samples=512)
    assert linear == linearized_hazard_mc(mean, covariance, 2.0, 0.5, 512, hazard_generator(3, 0))
    assert np.isnan(model_tick_hazard("linearized", np.nan, *args[1:], hazard_generator(3, 0)))
