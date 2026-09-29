"""Forecast calibration and discrimination: recalibration, ECE, reliability bands, AUROC."""

from __future__ import annotations

import numpy as np
from scipy import special, stats

from tether.evt._common import as_generator, percentile_interval, two_sided_z
from tether.evt.proportions import wilson_interval

PROBABILITY_CLIP = 1e-6


def _forecasts(p, y) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if p.shape != y.shape or p.size == 0:
        raise ValueError("forecasts and outcomes must be non-empty and of equal length")
    if np.any((p < 0.0) | (p > 1.0)) or np.any(np.isnan(p)):
        raise ValueError("forecasts must lie in [0, 1]")
    if np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("outcomes must be 0 or 1")
    return p, y


def _binary_loglik(design: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    eta = design @ beta
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def logistic_recalibration(p, y, confidence: float = 0.95) -> dict[str, float]:
    """Fit ``logit P(y = 1) = intercept + slope logit(p)`` by Newton-IRLS.

    Forecasts are clipped to ``[1e-6, 1 - 1e-6]``; Newton steps are halved until the
    log-likelihood does not decrease.  Wald intervals come from the observed information
    at the estimate (equal to the expected information for the logit link).
    """
    p, y = _forecasts(p, y)
    if y.min() == y.max():
        raise ValueError("both outcomes must occur")
    clipped = np.clip(p, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    design = np.column_stack([np.ones(p.size), special.logit(clipped)])
    beta = np.array([0.0, 1.0])
    loglik = _binary_loglik(design, y, beta)
    for _ in range(100):
        mean = special.expit(design @ beta)
        information = design.T @ (design * (mean * (1.0 - mean))[:, np.newaxis])
        step = np.linalg.solve(information, design.T @ (y - mean))
        fraction = 1.0
        while fraction > 1e-10:
            candidate = beta + fraction * step
            candidate_loglik = _binary_loglik(design, y, candidate)
            if candidate_loglik >= loglik - 1e-12 * abs(loglik):
                break
            fraction *= 0.5
        beta, loglik = candidate, candidate_loglik
        if np.max(np.abs(fraction * step)) < 1e-10:
            break
    mean = special.expit(design @ beta)
    information = design.T @ (design * (mean * (1.0 - mean))[:, np.newaxis])
    standard_error = np.sqrt(np.diag(np.linalg.inv(information)))
    half_width = two_sided_z(confidence) * standard_error
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "intercept_lo": float(beta[0] - half_width[0]),
        "intercept_hi": float(beta[0] + half_width[0]),
        "slope_lo": float(beta[1] - half_width[1]),
        "slope_hi": float(beta[1] + half_width[1]),
    }


def _bin_index(p: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    return np.minimum((p * n_bins).astype(int), n_bins - 1)


def _ece(bins: np.ndarray, p: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    forecast_sum = np.bincount(bins, weights=p, minlength=n_bins)
    outcome_sum = np.bincount(bins, weights=y, minlength=n_bins)
    return float(np.sum(np.abs(forecast_sum - outcome_sum)) / p.size)


def expected_calibration_error(p, y, n_bins: int = 10) -> float:
    """Count-weighted mean of ``|mean forecast - observed frequency|`` over equal-width bins."""
    p, y = _forecasts(p, y)
    return _ece(_bin_index(p, n_bins), p, y, n_bins)


def ece_bootstrap(
    p,
    y,
    n_bins: int = 10,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """ECE with a percentile interval from resampling forecast-outcome pairs.

    The ECE is biased upward by sampling noise and its percentile interval inherits the
    bias, so for well-calibrated forecasts the interval can lie above the point estimate.
    """
    p, y = _forecasts(p, y)
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    bins = _bin_index(p, n_bins)
    generator = as_generator(rng)
    replicates = np.empty(n_boot)
    for index in range(n_boot):
        sample = generator.integers(0, p.size, size=p.size)
        replicates[index] = _ece(bins[sample], p[sample], y[sample], n_bins)
    lower, upper = percentile_interval(replicates, confidence)
    return _ece(bins, p, y, n_bins), lower, upper


def reliability_bins(
    p, y, n_bins: int = 10, family_confidence: float = 0.95
) -> list[dict[str, float | int | bool]]:
    """Reliability diagram entries with simultaneous (Bonferroni) Wilson bands.

    Only non-empty equal-width bins are returned; each band is a Wilson interval at level
    ``1 - (1 - family_confidence) / n_nonempty`` and ``inside`` says whether the bin's mean
    forecast lies within it.
    """
    p, y = _forecasts(p, y)
    bins = _bin_index(p, n_bins)
    counts = np.bincount(bins, minlength=n_bins)
    forecast_sum = np.bincount(bins, weights=p, minlength=n_bins)
    outcome_sum = np.bincount(bins, weights=y, minlength=n_bins)
    occupied = np.flatnonzero(counts)
    level = 1.0 - (1.0 - family_confidence) / occupied.size
    entries = []
    for index in occupied:
        observed, lower, upper = wilson_interval(
            int(round(outcome_sum[index])), int(counts[index]), level
        )
        mean_forecast = float(forecast_sum[index] / counts[index])
        entries.append(
            {
                "bin_lo": index / n_bins,
                "bin_hi": (index + 1) / n_bins,
                "count": int(counts[index]),
                "mean_forecast": mean_forecast,
                "observed": observed,
                "lo": lower,
                "hi": upper,
                "inside": bool(lower <= mean_forecast <= upper),
            }
        )
    return entries


def _classes(scores, y) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(y).ravel()
    if scores.shape != y.shape:
        raise ValueError("scores and outcomes must have equal length")
    positive = y.astype(bool)
    if positive.all() or not positive.any():
        raise ValueError("both outcomes must occur")
    return scores[positive], scores[~positive]


def auroc(scores, y) -> float:
    """Area under the ROC curve, the Mann-Whitney probability with ties counted as 1/2."""
    positive, negative = _classes(scores, y)
    ranks = stats.rankdata(np.concatenate([positive, negative]))
    n1, n0 = positive.size, negative.size
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auroc_interval(
    scores,
    y,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """AUROC with a percentile interval from a class-stratified bootstrap.

    Positives and negatives are resampled separately so every replicate keeps both classes;
    each replicate is evaluated from resampling multiplicities in linear time.
    """
    positive, negative = _classes(scores, y)
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    negative = np.sort(negative)
    below = np.searchsorted(negative, positive, side="left")
    at_or_below = np.searchsorted(negative, positive, side="right")
    n1, n0 = positive.size, negative.size
    generator = as_generator(rng)
    replicates = np.empty(n_boot)
    for index in range(n_boot):
        positive_weight = np.bincount(generator.integers(0, n1, size=n1), minlength=n1)
        cumulative = np.concatenate(
            ([0], np.cumsum(np.bincount(generator.integers(0, n0, size=n0), minlength=n0)))
        )
        wins = cumulative[below] + 0.5 * (cumulative[at_or_below] - cumulative[below])
        replicates[index] = np.sum(positive_weight * wins) / (n1 * n0)
    lower, upper = percentile_interval(replicates, confidence)
    return auroc(scores, y), lower, upper
