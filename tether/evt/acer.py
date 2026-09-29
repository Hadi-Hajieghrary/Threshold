"""Naess-Gaidai average conditional exceedance rate (ACER) estimation and extrapolation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize

_OFFSET_BOUNDS = (1e-3, 10.0)
_SHAPE_BOUNDS = (0.25, 10.0)
_OFFSET_GRID = np.linspace(*np.log(_OFFSET_BOUNDS), 41)
_SHAPE_GRID = np.linspace(*np.log(_SHAPE_BOUNDS), 21)


@dataclass(frozen=True)
class ACERFit:
    """Tail model ``epsilon(eta) = q exp(-a (eta - b)^c)``."""

    q: float
    a: float
    b: float
    c: float


def _count_above(ordered: np.ndarray, levels: np.ndarray) -> np.ndarray:
    return ordered.size - np.searchsorted(ordered, levels, side="right")


def acer_estimates(series: Sequence[np.ndarray], levels, k: int) -> np.ndarray:
    """ACER function ``epsilon_k(eta)`` per sample, pooled over the series.

    ``epsilon_k(eta)`` is the fraction of positions ``j >= k`` at which ``X_j > eta`` while
    the ``k - 1`` preceding samples are all ``<= eta``; counts and positions are summed over
    the series before dividing (``k = 1`` is the plain exceedance fraction).
    """
    if k not in (1, 2, 3):
        raise ValueError("k must be 1, 2 or 3")
    levels = np.asarray(levels, dtype=float).ravel()
    counts = np.zeros(levels.size)
    positions = 0
    for values in series:
        values = np.asarray(values, dtype=float).ravel()
        if values.size < k:
            continue
        current = values[k - 1:]
        counts += _count_above(np.sort(current), levels)
        if k > 1:
            preceding = np.maximum.reduce(
                [values[k - 1 - lag: values.size - lag] for lag in range(1, k)]
            )
            counts -= _count_above(np.sort(np.minimum(current, preceding)), levels)
        positions += current.size
    if positions == 0:
        raise ValueError("no series is long enough for order k")
    return counts / positions


def _linear_part(
    levels: np.ndarray, log_epsilon: np.ndarray, weights: np.ndarray, b: float, c: float
) -> tuple[float, float, float]:
    """Weighted least squares of ``log epsilon = log q - a (eta - b)^c`` at fixed ``b, c``."""
    t = (levels - b) ** c
    total = weights.sum()
    mean_t = np.sum(weights * t) / total
    mean_y = np.sum(weights * log_epsilon) / total
    spread = np.sum(weights * (t - mean_t) ** 2)
    if not spread > 0.0:
        return np.inf, np.nan, np.nan
    a = -np.sum(weights * (t - mean_t) * (log_epsilon - mean_y)) / spread
    log_q = mean_y + a * mean_t
    residual = log_epsilon - log_q + a * t
    return float(np.sum(weights * residual**2)), float(log_q), float(a)


def _minimize_on_grid(
    objective: Callable[[float], float], grid: np.ndarray
) -> tuple[float, float]:
    """Minimize on ``[grid[0], grid[-1]]``: grid scan, then bounded Brent around the best node."""
    values = np.array([objective(x) for x in grid])
    best = int(np.argmin(values))
    if not np.isfinite(values[best]):
        return float(grid[best]), np.inf

    def finite(x: float) -> float:
        value = objective(x)
        return value if np.isfinite(value) else 1e300

    result = optimize.minimize_scalar(
        finite,
        bounds=(grid[max(best - 1, 0)], grid[min(best + 1, grid.size - 1)]),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if result.fun < values[best]:
        return float(result.x), float(result.fun)
    return float(grid[best]), float(values[best])


def fit_acer(levels, epsilon, tail_start: float, weights=None) -> ACERFit:
    """Fit ``epsilon ~ q exp(-a (eta - b)^c)`` on levels at or above ``tail_start``.

    Weighted least squares on ``log epsilon``: ``q`` and ``a`` are solved exactly at each
    ``(b, c)``, searched under ``a > 0``, ``0.25 <= c <= 10`` and
    ``1e-3 <= (tail_start - b) / span <= 10`` with ``span`` the width of the fitted level
    range.  The bounds exclude the degenerate power-law limit (``c -> 0``, ``b -> -inf``) in
    which ``q`` overflows.  The residual is nearly flat along the offset when ``c`` is near 1
    and has separate basins, so ``(b, c)`` are found by profiling: ``c`` is minimized at each
    offset (21-point log grid refined by bounded Brent) and the profile is minimized over the
    offset the same way on a 41-point log grid.  Default weights are proportional to
    ``epsilon``, the Poisson inverse variance of ``log epsilon`` for a fixed number of samples.
    """
    levels = np.asarray(levels, dtype=float).ravel()
    epsilon = np.asarray(epsilon, dtype=float).ravel()
    if levels.shape != epsilon.shape:
        raise ValueError("levels and epsilon must have equal length")
    weights = epsilon.copy() if weights is None else np.asarray(weights, dtype=float).ravel()
    keep = (levels >= tail_start) & (epsilon > 0.0) & (weights > 0.0) & np.isfinite(levels)
    if np.count_nonzero(keep) < 4:
        raise ValueError("at least four positive ACER values are required in the tail")
    eta, log_epsilon, weights = levels[keep], np.log(epsilon[keep]), weights[keep]
    weights = weights / weights.max()
    span = max(eta.max() - tail_start, 1e-12)

    def residual(log_offset: float, log_c: float) -> float:
        b = tail_start - span * np.exp(log_offset)
        value, _, a = _linear_part(eta, log_epsilon, weights, b, float(np.exp(log_c)))
        return value if a > 0.0 and np.isfinite(value) else np.inf

    def profile(log_offset: float) -> float:
        return _minimize_on_grid(lambda log_c: residual(log_offset, log_c), _SHAPE_GRID)[1]

    log_offset, best = _minimize_on_grid(profile, _OFFSET_GRID)
    if not np.isfinite(best):
        raise ValueError("no decreasing tail model fits the ACER values")
    log_c = _minimize_on_grid(lambda s: residual(log_offset, s), _SHAPE_GRID)[0]
    b = tail_start - span * np.exp(log_offset)
    c = float(np.exp(log_c))
    _, log_q, a = _linear_part(eta, log_epsilon, weights, b, c)
    return ACERFit(q=float(np.exp(log_q)), a=a, b=float(b), c=c)


def acer_extrapolate(fit: ACERFit, level) -> float | np.ndarray:
    """Evaluate the fitted ACER tail ``q exp(-a (level - b)^c)`` (``q`` at or below ``b``)."""
    level = np.asarray(level, dtype=float)
    value = fit.q * np.exp(-fit.a * np.maximum(level - fit.b, 0.0) ** fit.c)
    return float(value) if value.ndim == 0 else value
