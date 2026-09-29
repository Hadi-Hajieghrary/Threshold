"""Generalized Pareto tail fitting, profile-likelihood intervals, and threshold diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from tether.evt._common import check_confidence, finite_1d, scalar_or_array, two_sided_z

SHAPE_BOUNDS = (-0.5, 1.5)
MIN_EXCEEDANCES = 5
_SHAPE_GRID = np.linspace(SHAPE_BOUNDS[0], SHAPE_BOUNDS[1], 21)
_EXPONENTIAL_LIMIT = 1e-10
_SERIES_LIMIT = 1e-3
_MAX_CUMULATIVE_HAZARD = 700.0
_HAZARD_STEPS = 0.1 * 2.0 ** np.arange(8)


@dataclass(frozen=True)
class GPDFit:
    """GPD fit to the excesses over ``threshold``; ``covariance`` is ordered (shape, scale)."""

    shape: float
    scale: float
    threshold: float
    n_exceedances: int
    n_total: int
    loglik: float
    covariance: np.ndarray


@dataclass(frozen=True)
class _UnitFit:
    excesses: np.ndarray
    unit: float
    shape: float
    loglik: float


def _excesses(values, threshold: float) -> tuple[np.ndarray, int]:
    finite = finite_1d(values)
    excesses = finite[finite > threshold] - threshold
    if excesses.size < MIN_EXCEEDANCES:
        raise ValueError(f"at least {MIN_EXCEEDANCES} exceedances are required")
    return excesses, finite.size


def _loglik(excesses: np.ndarray, shape: float, scale: float) -> float:
    if not scale > 0.0:
        return -np.inf
    z = excesses / scale
    base = -excesses.size * np.log(scale)
    if abs(shape) < _EXPONENTIAL_LIMIT:
        return float(base - z.sum())
    w = shape * z
    if w.min() <= -1.0:
        return -np.inf
    return float(base - (1.0 + 1.0 / shape) * np.log1p(w).sum())


def _profile_scale(excesses: np.ndarray, shape: float) -> float:
    """Scale maximizing the likelihood at fixed shape: the unique root of the scale score.

    The score condition ``mean(y / (scale + shape y)) = 1 / (1 + shape)`` is strictly
    decreasing in the scale on the support, so the root is bracketed in log scale and found
    by safeguarded Newton from the mean excess (the exact root at shape 0), with Brent as
    the fallback.  Returns NaN when no root exists (samples dominated by zero excesses).
    """
    largest = excesses.max()
    mean = excesses.mean()
    target = 1.0 / (1.0 + shape)
    upper = np.log(2.0 * (abs(shape) * largest + (1.0 + abs(shape)) * mean))
    lower = np.log(-shape * largest) if shape < 0.0 else np.log(mean) - 50.0
    log_scale = np.log(mean) if np.log(mean) > lower else 0.5 * (lower + upper)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for _ in range(60):
            scale = np.exp(log_scale)
            denominator = scale + shape * excesses
            ratio = excesses / denominator
            score = ratio.mean() - target
            if abs(score) <= 1e-15 * target:
                return float(scale)
            if score > 0.0:
                lower = log_scale
            else:
                upper = log_scale
            proposal = log_scale + score / (scale * np.mean(ratio / denominator))
            if abs(proposal - log_scale) < 1e-13:
                if abs(score) < 1e-9 * target:
                    return float(np.exp(proposal))
                break
            if not lower < proposal < upper:
                proposal = 0.5 * (lower + upper)
            log_scale = proposal
    return _profile_scale_bracketed(excesses, shape, target)


def _profile_scale_bracketed(excesses: np.ndarray, shape: float, target: float) -> float:
    largest = excesses.max()
    mean = excesses.mean()

    def score(log_scale: float) -> float:
        with np.errstate(divide="ignore", over="ignore"):
            return float(np.mean(excesses / (np.exp(log_scale) + shape * excesses)) - target)

    upper = np.log(2.0 * (abs(shape) * largest + (1.0 + abs(shape)) * mean))
    if shape < 0.0:
        lower, step = np.log(-shape * largest) + 1e-12, 1e-9
    else:
        lower, step = np.log(mean) - 10.0, -10.0
    for _ in range(30):
        if score(lower) > 0.0:
            return float(np.exp(optimize.brentq(score, lower, upper, xtol=1e-12)))
        lower += step
    return np.nan


def _profile_loglik(excesses: np.ndarray, shape: float) -> float:
    return _loglik(excesses, shape, _profile_scale(excesses, shape))


def _maximize(
    objective: Callable[[float], float], grid: np.ndarray
) -> tuple[float, float]:
    """Maximize on ``[grid[0], grid[-1]]``: grid scan, then bounded Brent around the best node."""
    values = np.array([objective(x) for x in grid])
    if not np.any(np.isfinite(values)):
        raise ValueError("the likelihood is not finite anywhere on the search grid")
    best = int(np.argmax(np.where(np.isfinite(values), values, -np.inf)))
    bracket = (grid[max(best - 1, 0)], grid[min(best + 1, grid.size - 1)])

    def negative(x: float) -> float:
        value = objective(x)
        return -value if np.isfinite(value) else 1e300

    result = optimize.minimize_scalar(
        negative, bounds=bracket, method="bounded", options={"xatol": 1e-9}
    )
    if np.isfinite(result.fun) and -result.fun > values[best]:
        return float(result.x), float(-result.fun)
    return float(grid[best]), float(values[best])


def _fit_unit(excesses: np.ndarray) -> _UnitFit:
    """Constrained MLE on excesses rescaled to unit mean, for numerical conditioning."""
    unit = float(excesses.mean())
    if not unit > 0.0:
        raise ValueError("excesses must not all be zero")
    scaled = excesses / unit
    shape, loglik = _maximize(lambda s: _profile_loglik(scaled, s), _SHAPE_GRID)
    return _UnitFit(scaled, unit, shape, loglik)


def _shape_second_derivative(z: np.ndarray, shape: float) -> np.ndarray:
    """Per-observation d^2 loglik / d shape^2, with a series where shape * z is small."""
    x = shape * z
    small = np.abs(x) < _SERIES_LIMIT
    phi = np.empty_like(z)
    xs = x[small]
    phi[small] = -2.0 / 3.0 + xs * (1.5 + xs * (-2.4 + xs * (10.0 / 3.0)))
    xl = x[~small]
    phi[~small] = (
        -2.0 * np.log1p(xl) + 2.0 * xl / (1.0 + xl) + (xl / (1.0 + xl)) ** 2
    ) / xl**3
    return z**3 * phi + (z / (1.0 + x)) ** 2


def _observed_covariance(excesses: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """Inverse observed information in (shape, scale); NaN if not positive definite."""
    z = excesses / scale
    w = 1.0 + shape * z
    k = excesses.size
    d_scale_scale = (k - (1.0 + shape) * np.sum(z / w + z / w**2)) / scale**2
    d_shape_scale = (np.sum(z / w) - (1.0 + shape) * np.sum((z / w) ** 2)) / scale
    d_shape_shape = float(np.sum(_shape_second_derivative(z, shape)))
    information = -np.array(
        [[d_shape_shape, d_shape_scale], [d_shape_scale, d_scale_scale]]
    )
    if not np.all(np.isfinite(information)) or np.any(np.linalg.eigvalsh(information) <= 0.0):
        return np.full((2, 2), np.nan)
    return np.linalg.inv(information)


def _fit_excesses(excesses: np.ndarray, threshold: float, n_total: int) -> GPDFit:
    unit_fit = _fit_unit(excesses)
    scale = _profile_scale(unit_fit.excesses, unit_fit.shape) * unit_fit.unit
    return GPDFit(
        shape=unit_fit.shape,
        scale=scale,
        threshold=float(threshold),
        n_exceedances=int(excesses.size),
        n_total=int(n_total),
        loglik=unit_fit.loglik - excesses.size * np.log(unit_fit.unit),
        covariance=_observed_covariance(excesses, unit_fit.shape, scale),
    )


def fit_gpd(values, threshold: float) -> GPDFit:
    """Maximum-likelihood GPD fit to the excesses of ``values`` over ``threshold``.

    Non-finite values are ignored and ``n_total`` counts the finite ones.  The shape is
    constrained to ``SHAPE_BOUNDS`` = [-0.5, 1.5]; the scale is profiled out
    exactly, the shape found by a grid scan refined with bounded Brent, and shapes within
    1e-10 of zero use the exponential limit.  The covariance is the inverse observed
    information at the estimate (not meaningful when the shape sits on a bound).
    """
    excesses, n_total = _excesses(values, threshold)
    return _fit_excesses(excesses, threshold, n_total)


def _shape_interval(unit_fit: _UnitFit, confidence: float) -> tuple[float, float]:
    drop = 0.5 * float(stats.chi2.ppf(check_confidence(confidence), 1))

    def deficit(shape: float) -> float:
        return unit_fit.loglik - _profile_loglik(unit_fit.excesses, shape) - drop

    limits = []
    for bound in SHAPE_BOUNDS:
        if bound == unit_fit.shape or deficit(bound) <= 0.0:
            limits.append(float(bound))
        else:
            limits.append(float(optimize.brentq(deficit, bound, unit_fit.shape, xtol=1e-8)))
    return limits[0], limits[1]


def profile_shape_interval(
    values, threshold: float, confidence: float = 0.95
) -> tuple[float, float]:
    """Profile-likelihood interval for the GPD shape, clipped to ``SHAPE_BOUNDS``."""
    excesses, _ = _excesses(values, threshold)
    return _shape_interval(_fit_unit(excesses), confidence)


def _survival(excess, shape: float, scale: float) -> np.ndarray:
    z = np.maximum(np.asarray(excess, dtype=float), 0.0) / scale
    if abs(shape) < _EXPONENTIAL_LIMIT:
        return np.exp(-z)
    w = shape * z
    inside = w > -1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        survival = np.exp(-np.log1p(np.where(inside, w, 0.0)) / shape)
    return np.where(inside, survival, 0.0)


def exceedance_probability(fit: GPDFit, x) -> float | np.ndarray:
    """``P(X > x | X > u) = (1 + shape (x - u) / scale)^(-1 / shape)``; 1 for ``x <= u``."""
    excess = np.asarray(x, dtype=float) - fit.threshold
    return scalar_or_array(_survival(excess, fit.shape, fit.scale))


def tail_rate(fit: GPDFit, rate_at_threshold: float, x) -> float | np.ndarray:
    """Rate of exceeding ``x``: the rate at the threshold times the GPD exceedance probability."""
    return scalar_or_array(rate_at_threshold * np.asarray(exceedance_probability(fit, x)))


def _scale_for_hazard(shape: float, distance: float, hazard: float) -> float:
    """Scale giving cumulative hazard ``-log P(Y > distance) = hazard`` at fixed shape."""
    if abs(shape) < _EXPONENTIAL_LIMIT:
        return distance / hazard
    with np.errstate(over="ignore"):
        return float(shape * distance / np.expm1(shape * hazard))


def profile_tail_rate_interval(
    values,
    threshold: float,
    rate_at_threshold: float,
    x: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Profile-likelihood interval for the rate of exceeding ``x``.

    The GPD conditional exceedance probability ``p = P(X > x | X > u)`` is profiled over the
    shape (within ``SHAPE_BOUNDS``) with the scale fixed by ``p``; the interval for ``p`` is
    scaled by ``rate_at_threshold``.  The Poisson uncertainty of ``rate_at_threshold`` is
    ignored, so the interval is conditional on the threshold rate.  The search runs over the
    cumulative hazard ``-log p`` (capped at 700); a limit the likelihood cannot exclude is
    reported as 0 (below) or ``rate_at_threshold`` (above).  When the fitted upper endpoint
    lies below ``x`` the estimate is 0, the lower limit is 0, and the upper limit is the
    largest ``p`` the likelihood still admits (0 if none).
    """
    excesses, _ = _excesses(values, threshold)
    if x <= threshold:
        return float(rate_at_threshold), float(rate_at_threshold)
    unit_fit = _fit_unit(excesses)
    distance = (x - threshold) / unit_fit.unit
    drop = 0.5 * float(stats.chi2.ppf(check_confidence(confidence), 1))
    scale_hat = _profile_scale(unit_fit.excesses, unit_fit.shape)
    p_hat = float(_survival(distance, unit_fit.shape, scale_hat))
    hazard_hat = -np.log(p_hat) if p_hat > 0.0 else np.inf

    def deficit(log_hazard: float) -> float:
        hazard = np.exp(log_hazard)
        profile = _maximize(
            lambda s: _loglik(unit_fit.excesses, s, _scale_for_hazard(s, distance, hazard)),
            _SHAPE_GRID,
        )[1]
        return unit_fit.loglik - profile - drop

    def exit_point(inside: float, direction: float) -> float | None:
        inner = inside
        for step in _HAZARD_STEPS:
            outer = inside + direction * step
            if deficit(outer) > 0.0:
                return float(np.exp(optimize.brentq(deficit, inner, outer, xtol=1e-9)))
            inner = outer
        return None

    start = np.log(min(hazard_hat, _MAX_CUMULATIVE_HAZARD))
    if deficit(start) <= 0.0:
        high_hazard = exit_point(start, 1.0)
        low_hazard = exit_point(start, -1.0)
    else:
        high_hazard = None
        entry = next(
            (start - step for step in _HAZARD_STEPS if deficit(start - step) <= 0.0), None
        )
        if entry is None:
            return 0.0, 0.0
        low_hazard = exit_point(entry, -1.0)
    p_lower = 0.0 if high_hazard is None else float(np.exp(-high_hazard))
    p_upper = 1.0 if low_hazard is None else float(np.exp(-low_hazard))
    return rate_at_threshold * p_lower, rate_at_threshold * p_upper


def mean_residual_life(
    values, thresholds, confidence: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean excess over each threshold with a normal-theory interval (NaN below 2 excesses)."""
    ordered = np.sort(finite_1d(values))
    thresholds = np.asarray(thresholds, dtype=float).ravel()
    z = two_sided_z(confidence)
    mean_excess = np.full(thresholds.size, np.nan)
    half_width = np.full(thresholds.size, np.nan)
    for index, threshold in enumerate(thresholds):
        excesses = ordered[np.searchsorted(ordered, threshold, side="right"):] - threshold
        if excesses.size >= 2:
            mean_excess[index] = excesses.mean()
            half_width[index] = z * excesses.std(ddof=1) / np.sqrt(excesses.size)
    return mean_excess, mean_excess - half_width, mean_excess + half_width


def parameter_stability(
    values,
    thresholds,
    confidence: float = 0.95,
    min_exceedances: int = 10,
) -> dict[str, np.ndarray]:
    """GPD shape (with profile interval) and modified scale ``scale - shape u`` per threshold.

    Thresholds with fewer than ``min_exceedances`` exceedances give NaN.
    """
    finite = finite_1d(values)
    thresholds = np.asarray(thresholds, dtype=float).ravel()
    result = {
        name: np.full(thresholds.size, np.nan)
        for name in ("shape", "shape_lo", "shape_hi", "modified_scale")
    }
    result["threshold"] = thresholds.copy()
    result["n_exceedances"] = np.array([np.count_nonzero(finite > u) for u in thresholds])
    for index, threshold in enumerate(thresholds):
        if result["n_exceedances"][index] < max(min_exceedances, MIN_EXCEEDANCES):
            continue
        excesses = finite[finite > threshold] - threshold
        unit_fit = _fit_unit(excesses)
        scale = _profile_scale(unit_fit.excesses, unit_fit.shape) * unit_fit.unit
        result["shape"][index] = unit_fit.shape
        result["shape_lo"][index], result["shape_hi"][index] = _shape_interval(unit_fit, confidence)
        result["modified_scale"][index] = scale - unit_fit.shape * threshold
    return result


def select_threshold(
    values,
    candidate_quantiles: Sequence[float] = (0.80, 0.85, 0.90, 0.925, 0.95, 0.97),
    min_exceedances: int = 50,
    confidence: float = 0.95,
) -> float:
    """Lowest candidate threshold above which the GPD shape is stable.

    Candidates are the sample quantiles at ``candidate_quantiles``; only those with at least
    ``min_exceedances`` exceedances are eligible.  The selected threshold is the lowest
    eligible candidate whose profile-likelihood shape interval contains the shape estimate
    of every higher eligible candidate.  The highest eligible candidate passes vacuously,
    so it is the fallback when no lower candidate is stable.
    """
    finite = finite_1d(values)
    thresholds = np.quantile(finite, np.sort(np.asarray(candidate_quantiles, dtype=float)))
    required = max(min_exceedances, MIN_EXCEEDANCES)
    eligible = [u for u in thresholds if np.count_nonzero(finite > u) >= required]
    if not eligible:
        raise ValueError("no candidate threshold has enough exceedances")
    fits = [_fit_unit(finite[finite > u] - u) for u in eligible]
    for index, unit_fit in enumerate(fits):
        lower, upper = _shape_interval(unit_fit, confidence)
        if all(lower <= higher.shape <= upper for higher in fits[index + 1:]):
            return float(eligible[index])
    return float(eligible[-1])
