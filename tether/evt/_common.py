"""Shared helpers for the extreme-value statistics package."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

DEFAULT_SEED = 0


def as_generator(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Return ``rng`` as a Generator; ``None`` gives a fixed-seed Generator so results replay."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(DEFAULT_SEED if rng is None else rng)


def check_confidence(confidence: float) -> float:
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    return confidence


def two_sided_z(confidence: float) -> float:
    """Standard-normal quantile for a two-sided interval at ``confidence``."""
    return float(stats.norm.ppf(0.5 + 0.5 * check_confidence(confidence)))


def percentile_interval(
    replicates: np.ndarray, confidence: float, axis: int | None = None
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Percentile interval of bootstrap replicates, ignoring non-finite replicates."""
    alpha = 1.0 - check_confidence(confidence)
    values = np.asarray(replicates, dtype=float)
    values = np.where(np.isfinite(values), values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lower, upper = np.nanquantile(values, [alpha / 2.0, 1.0 - alpha / 2.0], axis=axis)
    if axis is None:
        return float(lower), float(upper)
    return lower, upper


def finite_1d(values) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def scalar_or_array(values: np.ndarray) -> np.ndarray | float:
    values = np.asarray(values, dtype=float)
    return float(values) if values.ndim == 0 else values
