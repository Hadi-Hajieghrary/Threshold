"""Plan v3 WP0: the multitype branching model of the cascade."""

from __future__ import annotations

import numpy as np
import pytest

from tether.theory import branching as B


def test_single_type_progeny_is_geometric():
    for m in (0.0, 0.3, 0.7):
        k = np.array([[m]])
        assert abs(B.mean_progeny(k)[0, 0] - 1.0 / (1.0 - m)) < 1e-12
        assert abs(B.expected_cluster_size(k, [1.0]) - 1.0 / (1.0 - m)) < 1e-12


def test_theta_equals_one_minus_m_single_type():
    assert abs(B.extremal_index(np.array([[0.4]]), [1.0]) - 0.6) < 1e-12
    assert abs(B.mean_offspring(np.array([[0.4]]), [1.0]) - 0.4) < 1e-12


def test_rho_of_rank_one_kernel_and_criticality():
    u = np.array([1.0, 2.0, 3.0])
    k = np.outer(u, u) / 20.0  # rho = |u|^2 / 20 = 0.7
    assert abs(B.spectral_radius(k) - 0.7) < 1e-12
    with pytest.raises(ValueError):
        B.mean_progeny(np.outer(u, u) / 10.0)  # rho = 1.4
    with pytest.raises(ValueError):
        B.spectral_radius(np.array([[np.nan]]))


def test_total_rate_reduces_to_multiplier():
    k = np.array([[0.0, 0.2], [0.3, 0.0]])
    nu = np.array([1.0, 2.0])
    total = B.total_rate(k, nu)
    assert np.allclose(total, np.linalg.solve(np.eye(2) - k, nu))
    # symmetric single-rate case: every cable sees nu / (1 - m)
    k2 = np.full((5, 5), 0.1) - 0.1 * np.eye(5)
    assert np.allclose(B.total_rate(k2, np.ones(5)), 1.0 / (1.0 - 0.4))


def test_borel_tanner_normalises_and_has_mean_1_over_1_minus_m():
    sizes = np.arange(1, 400)
    for m in (0.2, 0.5, 0.8):
        pmf = B.borel_tanner_pmf(m, sizes)
        assert abs(pmf.sum() - 1.0) < 1e-6
        assert abs(np.sum(pmf * sizes) - 1.0 / (1.0 - m)) < 2e-3
    assert B.borel_tanner_pmf(0.0, np.array([1, 2]))[0] == 1.0
    with pytest.raises(ValueError):
        B.borel_tanner_pmf(1.0, sizes)


def test_monte_carlo_cluster_sizes_match_borel_tanner():
    m = 0.5
    sizes = B.simulate_cluster_sizes(np.array([[m]]), [1.0], 20000, rng=1)
    assert abs(sizes.mean() - 2.0) < 0.1
    values, counts = np.unique(sizes, return_counts=True)
    pmf = counts / sizes.size
    expected = B.borel_tanner_pmf(m, values)
    assert np.max(np.abs(pmf[:5] - expected[:5])) < 0.01


def test_multitype_monte_carlo_matches_mean_progeny():
    k = np.array([[0.0, 0.3, 0.1], [0.2, 0.0, 0.2], [0.1, 0.3, 0.0]])
    pi = np.array([0.5, 0.3, 0.2])
    sizes = B.simulate_cluster_sizes(k, pi, 20000, rng=2)
    assert abs(sizes.mean() - B.expected_cluster_size(k, pi)) < 0.05


def test_cluster_size_uses_the_children_row_convention():
    # every primary is type 0; a type-0 parent has 0.05 type-1 children, a type-1 parent 0.8 type-0 children
    k = np.array([[0.0, 0.8], [0.05, 0.0]])
    pi = np.array([1.0, 0.0])
    expected = 1.0 + 0.05 * (1.0 + 0.8 * (1.0 + 0.05 * (1.0 + 0.8)))  # first four generations
    exact = B.expected_cluster_size(k, pi)
    assert abs(exact - 1.0 / (1.0 - 0.04) * (1.0 + 0.05)) < 1e-12  # 1^T (I-K)^-1 e_0 in closed form
    assert exact > expected - 1e-3
    wrong = float(pi @ B.mean_progeny(k).sum(axis=1))  # rows = parents would give this
    assert abs(wrong - exact) > 0.5
    sizes = B.simulate_cluster_sizes(k, pi, 20000, rng=7)
    assert abs(sizes.mean() - exact) < 0.03


def test_kernel_monotone_in_peak_and_zero_diagonal():
    quantiles = np.tile(np.linspace(0.0, 2000.0, 101), (3, 1))  # margins uniform on [0, 2 kN]
    ratio = np.array([[0.0, 0.4, 0.2], [0.4, 0.0, 0.4], [0.2, 0.4, 0.0]])
    low = B.kernel_from_margin_law(ratio, quantiles, np.array([500.0, 500.0]), np.array([0, 1]))
    high = B.kernel_from_margin_law(ratio, quantiles, np.array([4000.0, 4000.0]), np.array([0, 1]))
    assert low[0, 0] == 0.0 and np.isnan(low[0, 2])  # no parent on cable 2
    assert np.all(high[:, :2][~np.isnan(high[:, :2])] >= low[:, :2][~np.isnan(low[:, :2])])
    assert abs(low[1, 0] - 0.1) < 1e-9  # P(margin < 0.4 * 500 = 200) on uniform [0, 2000]
    assert abs(high[1, 0] - 0.8) < 1e-9 and abs(high[2, 0] - 0.4) < 1e-9  # swings 1600 N and 800 N


def test_empirical_kernel_counts_offspring_per_parent():
    parents = np.array([0, 0, 1])
    counts = np.array([[0, 2, 1], [0, 0, 1], [1, 0, 0]])
    k = B.empirical_kernel(parents, counts)
    assert k[1, 0] == 1.0 and k[2, 0] == 1.0 and k[0, 0] == 0.0 and k[0, 1] == 1.0
    assert np.all(np.isnan(k[:, 2]))


def test_bootstrap_spectral_radius_brackets_the_pooled_value():
    rng = np.random.default_rng(3)
    kernels = [(np.array([[0.0, 0.3 + 0.05 * rng.standard_normal()], [0.2, 0.0]]), 10) for _ in range(20)]
    value, lower, upper = B.bootstrap_spectral_radius(kernels, n_boot=200, rng=4)
    assert lower <= value <= upper and 0.15 < value < 0.35
