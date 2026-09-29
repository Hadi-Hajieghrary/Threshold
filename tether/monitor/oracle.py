"""Oracle arm precursor (arm O): the IV.9 precursor evaluated on plant truth.

While the vessel's slack flag is set, (e, e') are the true values at the tick; a_hat is the
least-squares slope of the true e' over the last 0.3 s of the current slack interval (window
inclusive), sigma_a its standard error floored at 0.05 m/s^2; before three samples exist
a_hat = a0 = T0 / m_eff with sigma_a at the floor.  The slope is fitted either on the ticks
(four at 10 Hz) or, given the 1 ms truth, on the 10 ms grid the estimator arms use.  Sigma
is a declared numerical floor, diag(ORACLE_SIGMA_E^2, ORACLE_SIGMA_EDOT^2).  Every field is
NaN while taut.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tether.monitor.outcomes import SlackIntervals, slack_intervals
from tether.theory.excursion import EFFECTIVE_MASS

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

MISSION_PRETENSION = 1000.0
SLOPE_WINDOW = 0.3
SLOPE_MIN_SAMPLES = 3
SIGMA_A_FLOOR = 0.05
ORACLE_SIGMA_E = 1.0e-4
ORACLE_SIGMA_EDOT = 1.0e-3
_WINDOW_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Precursor:
    """EstimatorOutput-like precursor arrays over the ticks (m,) and vessels (N)."""

    time: FloatArray
    slack: BoolArray
    onset_time: FloatArray
    e_hat: FloatArray
    edot_hat: FloatArray
    sigma: FloatArray
    a_hat: FloatArray
    sigma_a: FloatArray


def _slope(times: FloatArray, values: FloatArray) -> tuple[float, float]:
    centred = times - times.mean()
    spread = float(np.dot(centred, centred))
    slope = float(np.dot(centred, values - values.mean()) / spread)
    residual = values - values.mean() - slope * centred
    variance = float(np.dot(residual, residual)) / (times.size - 2)
    return slope, float(np.sqrt(variance / spread))


def acceleration_estimate(
    time: ArrayLike,
    rate: ArrayLike,
    intervals: SlackIntervals,
    a0: float,
    window: float = SLOPE_WINDOW,
    floor: float = SIGMA_A_FLOOR,
    min_samples: int = SLOPE_MIN_SAMPLES,
) -> tuple[FloatArray, FloatArray]:
    """(a_hat, sigma_a), arrays (m, N): least-squares slope of ``rate`` over the ticks of the
    current slack interval within ``window`` of the tick, standard error floored at ``floor``;
    (a0, floor) before ``min_samples`` finite samples exist; NaN while taut."""
    if min_samples < 3:
        raise ValueError("a standard error needs at least three samples")
    ticks = np.asarray(time, dtype=float)
    rates = np.asarray(rate, dtype=float).reshape(intervals.shape)
    a_hat = np.full(intervals.shape, np.nan)
    sigma_a = np.full(intervals.shape, np.nan)
    for vessel, first, last in zip(intervals.vessel, intervals.first, intervals.last):
        times = ticks[first : last + 1]
        values = rates[first : last + 1, vessel]
        lows = np.searchsorted(times, times - window - _WINDOW_TOLERANCE, side="left")
        for offset, low in enumerate(lows):
            span = slice(low, offset + 1)
            usable = np.isfinite(values[span])
            if np.count_nonzero(usable) < min_samples:
                a_hat[first + offset, vessel], sigma_a[first + offset, vessel] = a0, floor
                continue
            slope, error = _slope(times[span][usable], values[span][usable])
            a_hat[first + offset, vessel] = slope
            sigma_a[first + offset, vessel] = max(error, floor)
    return a_hat, sigma_a


def subsampled_acceleration(
    time: ArrayLike,
    slack: ArrayLike,
    series_time: ArrayLike,
    series_elongation: ArrayLike,
    series_rate: ArrayLike,
    a0: float,
    period: float = 0.01,
    window: float = SLOPE_WINDOW,
    floor: float = SIGMA_A_FLOOR,
    min_samples: int = SLOPE_MIN_SAMPLES,
) -> tuple[FloatArray, FloatArray]:
    """(a_hat, sigma_a) at the slack ticks (arrays (m, N)) from the true e' sampled every
    ``period`` since the current true slack onset and within ``window`` of the tick -- the
    sample grid of the estimator core's slope -- standard error floored at ``floor``; (a0,
    floor) before ``min_samples`` samples exist; NaN while taut."""
    ticks = np.asarray(time, dtype=float)
    flags = np.asarray(slack, dtype=bool)
    flags = flags[:, None] if flags.ndim == 1 else flags
    fine_time = np.asarray(series_time, dtype=float)
    elongation = np.asarray(series_elongation, dtype=float).reshape(fine_time.size, -1)
    rate = np.asarray(series_rate, dtype=float).reshape(fine_time.size, -1)
    on_grid = np.flatnonzero(np.abs(fine_time / period - np.round(fine_time / period)) < 1.0e-6)
    tick_index = np.clip(np.searchsorted(fine_time, ticks - 1.0e-9), 0, fine_time.size - 1)
    a_hat = np.full(flags.shape, np.nan)
    sigma_a = np.full(flags.shape, np.nan)
    positions = np.arange(fine_time.size)
    for vessel in range(flags.shape[1]):
        last_taut = np.maximum.accumulate(np.where(elongation[:, vessel] > 0.0, positions, -1))
        for k in np.flatnonzero(flags[:, vessel]):
            start = fine_time[min(last_taut[tick_index[k]] + 1, fine_time.size - 1)]
            low = max(start, ticks[k] - window) - _WINDOW_TOLERANCE
            grid = on_grid[(fine_time[on_grid] >= low) & (fine_time[on_grid] <= ticks[k] + _WINDOW_TOLERANCE)]
            values = rate[grid, vessel]
            usable = np.isfinite(values)
            if np.count_nonzero(usable) < min_samples:
                a_hat[k, vessel], sigma_a[k, vessel] = a0, floor
                continue
            slope, error = _slope(fine_time[grid][usable], values[usable])
            a_hat[k, vessel], sigma_a[k, vessel] = slope, max(error, floor)
    return a_hat, sigma_a


def oracle_precursor(
    time: ArrayLike,
    slack: ArrayLike,
    onset_time: ArrayLike,
    e_true: ArrayLike,
    edot_true: ArrayLike,
    pretension: float = MISSION_PRETENSION,
    sigma_e: float = ORACLE_SIGMA_E,
    sigma_edot: float = ORACLE_SIGMA_EDOT,
    series: tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
) -> Precursor:
    """Oracle precursor from the true e, e' at the ticks (arrays (m, N)) and the shared slack
    flags and onset times of the sensor-driven arms.  With ``series`` (the 1 ms truth
    ``(time, e, e')``) a_hat is fitted on the 10 ms sample grid of the estimator arms
    (``subsampled_acceleration``) instead of on the ticks."""
    ticks = np.asarray(time, dtype=float)
    flags = np.asarray(slack, dtype=bool)
    flags = flags[:, None] if flags.ndim == 1 else flags
    onsets = np.asarray(onset_time, dtype=float).reshape(flags.shape)
    elongation = np.asarray(e_true, dtype=float).reshape(flags.shape)
    rate = np.asarray(edot_true, dtype=float).reshape(flags.shape)
    intervals = slack_intervals(ticks, flags, onsets)
    if series is None:
        a_hat, sigma_a = acceleration_estimate(ticks, rate, intervals, pretension / EFFECTIVE_MASS)
    else:
        a_hat, sigma_a = subsampled_acceleration(ticks, flags, *series, pretension / EFFECTIVE_MASS)
    sigma = np.full(flags.shape + (2, 2), np.nan)
    sigma[flags] = np.diag([sigma_e**2, sigma_edot**2])
    return Precursor(
        time=ticks,
        slack=flags,
        onset_time=np.where(flags, onsets, np.nan),
        e_hat=np.where(flags, elongation, np.nan),
        edot_hat=np.where(flags, rate, np.nan),
        sigma=sigma,
        a_hat=a_hat,
        sigma_a=sigma_a,
    )
