"""Exact Poisson intervals for counts and rates, and zero-event censored bounds."""

from __future__ import annotations

import numpy as np
from scipy import stats

from tether.evt._common import check_confidence


def _check_count(count) -> int:
    if int(count) != count or count < 0:
        raise ValueError("count must be a non-negative integer")
    return int(count)


def poisson_interval(count: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact two-sided Garwood interval for a Poisson mean (chi-square form).

    The lower limit is 0 when ``count`` is 0; each tail carries (1 - confidence) / 2.
    """
    count = _check_count(count)
    alpha = 1.0 - check_confidence(confidence)
    lower = 0.0 if count == 0 else 0.5 * float(stats.chi2.ppf(alpha / 2.0, 2 * count))
    upper = 0.5 * float(stats.chi2.ppf(1.0 - alpha / 2.0, 2 * count + 2))
    return lower, upper


def rate_interval(
    count: int, exposure: float, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Rate ``count / exposure`` with its exact Garwood interval."""
    if not exposure > 0.0:
        raise ValueError("exposure must be positive")
    lower, upper = poisson_interval(count, confidence)
    return _check_count(count) / exposure, lower / exposure, upper / exposure


def censored_upper_bound(exposure: float, confidence: float = 0.95) -> float:
    """One-sided upper confidence bound on a rate after zero events in ``exposure``."""
    if not exposure > 0.0:
        raise ValueError("exposure must be positive")
    return float(-np.log1p(-check_confidence(confidence)) / exposure)
