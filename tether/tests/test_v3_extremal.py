"""Plan v3 WP0: extremal-index and cluster-size estimators."""

from __future__ import annotations

import numpy as np
import pytest

from tether.evt import extremal as E


def _clustered_indices(rng, clusters: int, mean_size: float, gap: int = 2000) -> np.ndarray:
    """Clusters of geometric size (mean ``mean_size``) separated by exponential gaps."""
    indices, position = [], 0
    for _ in range(clusters):
        position += int(rng.exponential(gap)) + 10
        size = rng.geometric(1.0 / mean_size)
        for k in range(size):
            indices.append(position + 2 * k)
    return np.array(indices, dtype=np.int64)


def test_intervals_estimator_near_one_on_iid():
    rng = np.random.default_rng(0)
    indices = np.sort(rng.choice(2_000_000, size=4000, replace=False))
    assert abs(E.intervals_estimator(indices) - 1.0) < 0.05


def test_intervals_estimator_on_synthetic_clusters():
    rng = np.random.default_rng(1)
    indices = _clustered_indices(rng, clusters=6000, mean_size=2.0)
    assert abs(E.intervals_estimator(indices) - 0.5) < 0.05


def test_runs_estimator_equals_cluster_fraction():
    indices = np.array([10, 11, 12, 500, 501, 900])
    assert E.runs_estimator(indices, run_length=5) == 3 / 6
    assert E.runs_estimator(indices, run_length=1000) == 1 / 6
    assert np.isnan(E.runs_estimator(np.array([]), 3))


def test_theta_point_process_converts_times_to_samples():
    out = E.theta_point_process(np.array([1.0, 1.001, 1.002, 5.0]), sample_period=1e-3, run_length=3.0)
    assert out["n"] == 4 and out["theta_runs"] == 0.5


def test_cluster_sizes_from_forest():
    parent = np.array([-1, 0, 0, 1, -1, 4, -1])
    assert E.cluster_sizes_from_forest(parent).tolist() == [4, 2, 1]
    with pytest.raises(ValueError):
        E.cluster_sizes_from_forest(np.array([1, 0]))
    law = E.empirical_cluster_law(np.array([4, 2, 1]))
    assert law["mean"] == 7 / 3 and abs(law["theta"] - 3 / 7) < 1e-12 and law["pmf"][4] == 1 / 3


def test_bootstrap_covers_truth():
    rng = np.random.default_rng(2)
    seeds = [_clustered_indices(rng, clusters=400, mean_size=2.0) for _ in range(20)]
    value, lower, upper = E.bootstrap_theta(seeds, "intervals", n_boot=300, rng=5)
    assert lower <= value <= upper and lower < 0.5 < upper
    runs, lo, hi = E.bootstrap_theta(seeds, "runs", run_length=100, n_boot=300, rng=6)
    assert abs(runs - 0.5) < 0.05 and lo <= runs <= hi
    with pytest.raises(ValueError):
        E.bootstrap_theta(seeds, "runs")
