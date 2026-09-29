"""Hill tail-index estimation and log-log slope fits of exceedance rates."""

from __future__ import annotations

import numpy as np
from scipy import stats

from tether.evt._common import check_confidence, finite_1d, two_sided_z


def _descending(samples) -> np.ndarray:
    return np.sort(finite_1d(samples))[::-1]


def _check_order(k: int, ordered: np.ndarray) -> int:
    if int(k) != k or not 1 <= k < ordered.size:
        raise ValueError("k must be an integer in [1, sample_count - 1]")
    if not ordered[int(k)] > 0.0:
        raise ValueError("the k + 1 largest samples must be positive")
    return int(k)


def hill_estimator(
    samples, k: int, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Hill estimate of the upper-tail index from the ``k`` largest samples.

    ``alpha = 1 / mean(log(X_(i) / X_(k+1)))`` over the ``k`` upper order statistics, with the
    asymptotic normal interval ``alpha (1 -/+ z / sqrt(k))``.
    """
    ordered = _descending(samples)
    k = _check_order(k, ordered)
    mean_log_excess = float(np.mean(np.log(ordered[:k] / ordered[k])))
    alpha = 1.0 / mean_log_excess if mean_log_excess > 0.0 else np.inf
    half_width = two_sided_z(confidence) / float(np.sqrt(k))
    return alpha, alpha * (1.0 - half_width), alpha * (1.0 + half_width)


def hill_plot(samples, ks) -> np.ndarray:
    """Hill estimates of the tail index at each number of upper order statistics in ``ks``."""
    ordered = _descending(samples)
    ks = np.asarray(ks, dtype=int).ravel()
    if ks.size == 0:
        return np.empty(0)
    _check_order(int(ks.max()), ordered)
    if ks.min() < 1:
        raise ValueError("every k must be at least 1")
    logs = np.log(ordered[: ks.max() + 1])
    cumulative = np.cumsum(logs)
    mean_log_excess = cumulative[ks - 1] / ks - logs[ks]
    with np.errstate(divide="ignore"):
        return 1.0 / mean_log_excess


def log_slope_fit(
    x,
    y,
    weights,
    confidence: float = 0.95,
    known_variance: bool = False,
) -> tuple[float, float, float, float]:
    """Weighted least-squares fit of ``log y = intercept + slope log x``.

    Points with non-positive ``x``, ``y`` or weight are dropped.  With relative weights the
    slope interval uses the residual variance and a t quantile on ``m - 2`` degrees of
    freedom; with ``known_variance`` the weights are inverse variances of ``log y``, the
    residual variance factor is floored at 1 (inflated only for over-dispersion), and a
    normal quantile is used.  Returns ``(slope, intercept, slope_lo, slope_hi)``.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    if not x.shape == y.shape == weights.shape:
        raise ValueError("x, y and weights must have equal length")
    keep = (x > 0.0) & (y > 0.0) & (weights > 0.0) & np.isfinite(x * y * weights)
    if np.count_nonzero(keep) < 2:
        raise ValueError("at least two usable points are required")
    log_x, log_y, weights = np.log(x[keep]), np.log(y[keep]), weights[keep]
    total = weights.sum()
    mean_x = np.sum(weights * log_x) / total
    mean_y = np.sum(weights * log_y) / total
    spread = np.sum(weights * (log_x - mean_x) ** 2)
    if not spread > 0.0:
        raise ValueError("x must take at least two distinct values")
    slope = float(np.sum(weights * (log_x - mean_x) * (log_y - mean_y)) / spread)
    intercept = float(mean_y - slope * mean_x)
    residual = log_y - intercept - slope * log_x
    dof = log_x.size - 2
    residual_variance = np.sum(weights * residual**2) / dof if dof > 0 else np.nan
    check_confidence(confidence)
    if known_variance:
        variance_factor = max(1.0, residual_variance) if dof > 0 else 1.0
        quantile = two_sided_z(confidence)
    elif dof > 0:
        variance_factor = residual_variance
        quantile = float(stats.t.ppf(0.5 + 0.5 * confidence, dof))
    else:
        return slope, intercept, np.nan, np.nan
    half_width = quantile * float(np.sqrt(variance_factor / spread))
    return slope, intercept, slope - half_width, slope + half_width


def tail_index_from_rates(
    levels, rates, counts, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Tail index ``alpha = -d log(rate) / d log(level)`` by Poisson-weighted least squares.

    Weights are the exceedance counts (inverse variances of ``log rate``); levels with zero
    counts are dropped, and levels are treated as independent although cumulative counts at
    nested levels are positively correlated.
    """
    slope, _, slope_lo, slope_hi = log_slope_fit(
        levels, rates, counts, confidence=confidence, known_variance=True
    )
    return -slope, -slope_hi, -slope_lo
