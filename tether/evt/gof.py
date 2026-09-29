"""Goodness of fit with estimated parameters: Anderson-Darling and parametric-bootstrap KS."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats

from tether.evt._common import as_generator, finite_1d
from tether.evt.gpd import _fit_unit, _profile_scale, _survival

KS_FAMILIES = ("rayleigh", "exponential", "gpd")
_AD_VERTEX = 153.0


def _ad_normal_p_value(a_star: float) -> float:
    """D'Agostino and Stephens (1986, Table 4.9) p-value for the modified statistic."""
    if a_star >= 0.6:
        a = min(a_star, _AD_VERTEX)
        return float(np.exp(1.2937 - 5.709 * a + 0.0186 * a * a))
    if a_star >= 0.34:
        return float(np.exp(0.9177 - 4.279 * a_star - 1.38 * a_star * a_star))
    if a_star >= 0.2:
        return float(1.0 - np.exp(-8.318 + 42.796 * a_star - 59.938 * a_star * a_star))
    return float(1.0 - np.exp(-13.436 + 101.14 * a_star - 223.73 * a_star * a_star))


def anderson_darling_normal(x) -> tuple[float, float]:
    """Anderson-Darling test of normality with mean and variance estimated from ``x``.

    Returns the unmodified statistic ``A2`` and the p-value of the D'Agostino-Stephens
    modified statistic ``A* = A2 (1 + 0.75 / n + 2.25 / n^2)``.
    """
    ordered = np.sort(finite_1d(x))
    n = ordered.size
    if n < 3:
        raise ValueError("at least three samples are required")
    spread = ordered.std(ddof=1)
    if not spread > 0.0:
        raise ValueError("samples must not be constant")
    z = (ordered - ordered.mean()) / spread
    weights = 2.0 * np.arange(1, n + 1) - 1.0
    a2 = float(-n - np.sum(weights * (stats.norm.logcdf(z) + stats.norm.logsf(z[::-1]))) / n)
    return a2, _ad_normal_p_value(a2 * (1.0 + 0.75 / n + 2.25 / n**2))


def rayleigh_scale_mle(x) -> float:
    """Maximum-likelihood Rayleigh scale, ``sqrt(sum x^2 / 2n)``."""
    x = finite_1d(x)
    if x.size == 0:
        raise ValueError("at least one sample is required")
    return float(np.sqrt(np.sum(x * x) / (2.0 * x.size)))


def _rayleigh_fit(x: np.ndarray) -> tuple[float, ...]:
    return (rayleigh_scale_mle(x),)


def _rayleigh_cdf(x: np.ndarray, scale: float) -> np.ndarray:
    return -np.expm1(-0.5 * (np.maximum(x, 0.0) / scale) ** 2)


def _rayleigh_draw(generator: np.random.Generator, n: int, scale: float) -> np.ndarray:
    return generator.rayleigh(scale, n)


def _exponential_fit(x: np.ndarray) -> tuple[float, ...]:
    return (float(np.mean(x)),)


def _exponential_cdf(x: np.ndarray, scale: float) -> np.ndarray:
    return -np.expm1(-np.maximum(x, 0.0) / scale)


def _exponential_draw(generator: np.random.Generator, n: int, scale: float) -> np.ndarray:
    return generator.exponential(scale, n)


def _gpd_fit(x: np.ndarray) -> tuple[float, ...]:
    unit_fit = _fit_unit(x)
    return unit_fit.shape, _profile_scale(unit_fit.excesses, unit_fit.shape) * unit_fit.unit


def _gpd_cdf(x: np.ndarray, shape: float, scale: float) -> np.ndarray:
    return 1.0 - _survival(x, shape, scale)


def _gpd_draw(generator: np.random.Generator, n: int, shape: float, scale: float) -> np.ndarray:
    log_uniform = np.log1p(-generator.random(n))
    if abs(shape) < 1e-10:
        return -scale * log_uniform
    return scale * np.expm1(-shape * log_uniform) / shape


_FAMILY_FUNCTIONS: dict[str, tuple[Callable, Callable, Callable]] = {
    "rayleigh": (_rayleigh_fit, _rayleigh_cdf, _rayleigh_draw),
    "exponential": (_exponential_fit, _exponential_cdf, _exponential_draw),
    "gpd": (_gpd_fit, _gpd_cdf, _gpd_draw),
}


def _ks_statistic(x: np.ndarray, cdf: Callable, parameters: tuple[float, ...]) -> float:
    ordered = np.sort(x)
    n = ordered.size
    probabilities = cdf(ordered, *parameters)
    positions = np.arange(1, n + 1) / n
    above = np.max(positions - probabilities)
    below = np.max(probabilities - positions + 1.0 / n)
    return float(max(above, below))


def ks_parametric_bootstrap(
    x,
    family: str,
    n_boot: int = 999,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Kolmogorov-Smirnov test against ``family`` with parameters estimated from ``x``.

    The null distribution of ``D`` is simulated by fitting, drawing samples of the same size
    at the fitted parameters, refitting, and recomputing ``D``; the p-value is
    ``(1 + #{D* >= D}) / (n_boot + 1)``.  Families are ``rayleigh`` (scale MLE),
    ``exponential`` (mean MLE) and ``gpd``, for which ``x`` must be excesses over a threshold
    (fitted with threshold 0, shape within the GPD bounds).
    """
    if family not in _FAMILY_FUNCTIONS:
        raise ValueError(f"family must be one of {KS_FAMILIES}")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    fit, cdf, draw = _FAMILY_FUNCTIONS[family]
    x = finite_1d(x)
    if x.size < 5 or np.any(x < 0.0):
        raise ValueError("at least five non-negative samples are required")
    parameters = fit(x)
    statistic = _ks_statistic(x, cdf, parameters)
    generator = as_generator(rng)
    replicates = np.empty(n_boot)
    for index in range(n_boot):
        simulated = draw(generator, x.size, *parameters)
        replicates[index] = _ks_statistic(simulated, cdf, fit(simulated))
    return statistic, float((1 + np.count_nonzero(replicates >= statistic)) / (n_boot + 1))
