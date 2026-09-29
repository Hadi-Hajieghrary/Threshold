"""Extremal index and cluster-size estimators for a point process of onsets (plan v3 WP0/WP3).

Two classical estimators of theta on a stationary sequence sampled at a fixed period:

* the intervals estimator of Ferro & Segers (2003), which needs no declustering parameter
  and uses the inter-exceedance times ``T_i = S_{i+1} - S_i`` (in samples): with
  ``theta = min(1, 2 (sum T)^2 / ((N-1) sum T^2))`` when ``max T <= 2`` and
  ``theta = min(1, 2 (sum (T-1))^2 / ((N-1) sum (T-1)(T-2)))`` otherwise;
* the runs estimator, ``theta = clusters / exceedances`` with clusters separated by a gap of
  more than ``run_length`` samples.

The branching-process prediction (tether/theory/branching.py) is ``1 / E[cluster size]`` under
the declustering that assigns every offspring to its primary ancestor; ``cluster_sizes_from_forest``
computes exactly those sizes from a parent index.  Everything is pure NumPy.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from tether.evt._common import as_generator, percentile_interval


def interexceedance_times(exceedance_indices: np.ndarray) -> np.ndarray:
    indices = np.sort(np.asarray(exceedance_indices, dtype=np.int64))
    if indices.size < 2:
        return np.zeros(0, dtype=np.int64)
    return np.diff(indices)


def intervals_estimator_from_gaps(gaps: np.ndarray) -> float:
    """Ferro-Segers theta from inter-exceedance times in samples (nan with fewer than one gap)."""
    t = np.asarray(gaps, dtype=float)
    if t.size < 1:
        return math.nan
    if np.max(t) <= 2.0:
        denominator = t.size * np.sum(t * t)
        return math.nan if denominator <= 0.0 else float(min(1.0, 2.0 * np.sum(t) ** 2 / denominator))
    denominator = t.size * np.sum((t - 1.0) * (t - 2.0))
    return math.nan if denominator <= 0.0 else float(min(1.0, 2.0 * np.sum(t - 1.0) ** 2 / denominator))


def intervals_estimator(exceedance_indices: np.ndarray) -> float:
    return intervals_estimator_from_gaps(interexceedance_times(exceedance_indices))


def runs_estimator(exceedance_indices: np.ndarray, run_length: int) -> float:
    """clusters / exceedances, a new cluster starting after a gap of more than ``run_length`` samples."""
    indices = np.sort(np.asarray(exceedance_indices, dtype=np.int64))
    if indices.size == 0:
        return math.nan
    gaps = np.diff(indices)
    clusters = 1 + int(np.sum(gaps > run_length))
    return clusters / indices.size


def theta_point_process(times: np.ndarray, sample_period: float = 1.0e-3, run_length: float = 3.0) -> dict:
    """Both estimators on a list of onset times (seconds), sampled at ``sample_period``."""
    times = np.asarray(times, dtype=float)
    indices = np.unique(np.round(times / sample_period).astype(np.int64))
    return {
        "n": int(indices.size),
        "theta_intervals": intervals_estimator(indices),
        "theta_runs": runs_estimator(indices, int(round(run_length / sample_period))),
    }


def cluster_sizes_from_forest(parent_index: np.ndarray) -> np.ndarray:
    """Size (self included) of the tree rooted at each primary, ``parent_index[i] = -1`` for roots.

    Returns one size per root, in root order.  Raises on a cycle or a dangling parent.
    """
    parent = np.asarray(parent_index, dtype=np.int64)
    n = parent.size
    root_of = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        node, steps = i, 0
        while parent[node] >= 0:
            if parent[node] >= n:
                raise ValueError(f"dangling parent {parent[node]} at {node}")
            node = parent[node]
            steps += 1
            if steps > n:
                raise ValueError("cycle in the parent index")
        root_of[i] = node
    roots = np.flatnonzero(parent < 0)
    return np.array([int(np.sum(root_of == r)) for r in roots], dtype=np.int64)


def empirical_cluster_law(sizes: np.ndarray) -> dict:
    sizes = np.asarray(sizes, dtype=np.int64)
    if sizes.size == 0:
        return {"n": 0, "mean": math.nan, "theta": math.nan, "pmf": {}}
    values, counts = np.unique(sizes, return_counts=True)
    return {
        "n": int(sizes.size),
        "mean": float(sizes.mean()),
        "theta": float(1.0 / sizes.mean()),
        "max": int(sizes.max()),
        "pmf": {int(v): float(c / sizes.size) for v, c in zip(values, counts)},
    }


def bootstrap_theta(
    per_seed_indices: Sequence[np.ndarray],
    estimator: str = "intervals",
    run_length: int | None = None,
    n_boot: int = 2000,
    confidence: float = 0.95,
    rng=None,
) -> tuple[float, float, float]:
    """theta pooled over seeds (gaps pooled, never across seed boundaries) with a seed-resampling interval."""
    items = [np.sort(np.asarray(x, dtype=np.int64)) for x in per_seed_indices]
    if not items:
        raise ValueError("at least one seed is required")

    def statistic(rows: list) -> float:
        if estimator == "intervals":
            gaps = np.concatenate([interexceedance_times(x) for x in rows]) if rows else np.zeros(0)
            return intervals_estimator_from_gaps(gaps)
        if estimator == "runs":
            if run_length is None:
                raise ValueError("runs estimator needs run_length")
            exceedances = sum(int(x.size) for x in rows)
            clusters = sum(1 + int(np.sum(np.diff(x) > run_length)) for x in rows if x.size)
            return math.nan if exceedances == 0 else clusters / exceedances
        raise ValueError(f"unknown estimator: {estimator}")

    generator = as_generator(rng)
    indices = generator.integers(0, len(items), size=(n_boot, len(items)))
    replicates = np.array([statistic([items[i] for i in row]) for row in indices], dtype=float)
    lower, upper = percentile_interval(replicates, confidence)
    return float(statistic(items)), lower, upper
