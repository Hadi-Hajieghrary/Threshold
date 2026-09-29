"""Seed-level, paired, and moving-block bootstrap resampling.

Every resampler in ``tether.evt`` takes ``rng``: a Generator, an integer seed, or ``None``,
which uses a fixed seed so that every interval replays exactly.  Intervals are percentile
intervals; non-finite replicates are dropped.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from tether.evt._common import as_generator, percentile_interval


def _seed_indices(n: int, n_boot: int, rng) -> np.ndarray:
    if n < 1:
        raise ValueError("at least one seed is required")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    return as_generator(rng).integers(0, n, size=(n_boot, n))


def seed_bootstrap_rate(
    counts,
    exposures,
    n_boot: int = 2000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Pooled rate ``sum(counts) / sum(exposures)`` with a seed-resampling percentile interval."""
    counts = np.asarray(counts, dtype=float).ravel()
    exposures = np.asarray(exposures, dtype=float).ravel()
    if counts.shape != exposures.shape:
        raise ValueError("counts and exposures must have one entry per seed")
    if np.any(exposures < 0.0) or not exposures.sum() > 0.0:
        raise ValueError("exposures must be non-negative with a positive total")
    indices = _seed_indices(counts.size, n_boot, rng)
    with np.errstate(divide="ignore", invalid="ignore"):
        replicates = counts[indices].sum(axis=1) / exposures[indices].sum(axis=1)
    lower, upper = percentile_interval(replicates, confidence)
    return float(counts.sum() / exposures.sum()), lower, upper


def seed_bootstrap_statistic(
    per_seed_items: Sequence,
    statistic: Callable[[list], float],
    n_boot: int = 2000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """``statistic`` of the per-seed items with a seed-resampling percentile interval.

    Replicates for which ``statistic`` is not finite are dropped from the interval.
    """
    items = list(per_seed_items)
    indices = _seed_indices(len(items), n_boot, rng)
    replicates = np.array(
        [statistic([items[index] for index in row]) for row in indices], dtype=float
    )
    lower, upper = percentile_interval(replicates, confidence)
    return float(statistic(items)), lower, upper


def paired_bootstrap_difference(
    a,
    b,
    n_boot: int = 2000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Mean paired difference ``mean(a - b)``, resampling seeds jointly so pairs stay intact."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("paired outcomes must have equal length")
    differences = a - b
    indices = _seed_indices(differences.size, n_boot, rng)
    lower, upper = percentile_interval(differences[indices].mean(axis=1), confidence)
    return float(differences.mean()), lower, upper


def moving_block_indices(
    n: int, block_length: int, rng: np.random.Generator | None
) -> np.ndarray:
    """Indices of one moving-block bootstrap series of length ``n``.

    Blocks of ``block_length`` consecutive indices start uniformly on ``[0, n - block_length]``
    and are concatenated, the last block truncated so the series has exactly ``n`` entries.
    """
    if not 1 <= block_length <= n:
        raise ValueError("block_length must lie in [1, n]")
    block_count = -(-n // block_length)
    starts = as_generator(rng).integers(0, n - block_length + 1, size=block_count)
    return (starts[:, np.newaxis] + np.arange(block_length)).ravel()[:n]
