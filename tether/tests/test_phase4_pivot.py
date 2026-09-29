"""Phase 4 T1-fail pivot: the five-cable t copula fit and the fleet coincidence."""

from __future__ import annotations

import numpy as np
from scipy import stats

from tether.campaign.phase4 import fit_t_copula, fleet_coincidence, t_copula_draws


def _corr(rho: float, d: int = 5) -> np.ndarray:
    return np.full((d, d), rho) + (1.0 - rho) * np.eye(d)


def _ranks(x: np.ndarray) -> np.ndarray:
    return (stats.rankdata(x, axis=0)) / (x.shape[0] + 1.0)


def test_fit_recovers_heavy_and_gaussian_copulas():
    rng = np.random.default_rng(3)
    heavy = fit_t_copula(_ranks(t_copula_draws(_corr(0.5), 4.0, 6000, rng)))
    assert 3.0 <= heavy["nu"] <= 6.0
    assert heavy["loglik"] > heavy["gaussian_loglik"] + 20.0
    np.testing.assert_allclose(heavy["corr"][0, 1], 0.5, atol=0.05)
    z = rng.standard_normal((6000, 5)) @ np.linalg.cholesky(_corr(0.5)).T
    light = fit_t_copula(_ranks(z))
    assert light["nu"] >= 20.0


def test_fleet_coincidence_decreases_for_gaussian_and_levels_off_for_t():
    rng = np.random.default_rng(4)
    levels = (0.9, 0.99, 0.999)
    z = stats.norm.cdf(rng.standard_normal((400_000, 5)) @ np.linalg.cholesky(_corr(0.3)).T)
    gaussian = fleet_coincidence(z, levels)
    assert gaussian[0] > gaussian[1] > gaussian[2]
    t = fleet_coincidence(t_copula_draws(_corr(0.3), 3.0, 400_000, rng), levels)
    assert t[2] > 2.0 * gaussian[2]
    assert np.isnan(fleet_coincidence(np.zeros((10, 5)), (0.5,))[0])
