"""Extreme-value statistics checked against published values and simulated truth."""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

from tether import evt
from tether.evt.gpd import GPDFit

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([2026, seed]))


def _gpd_sample(shape: float, scale: float, size: int, seed: int) -> np.ndarray:
    return stats.genpareto.rvs(shape, scale=scale, size=size, random_state=_generator(seed))


def test_evt_imports_no_drake():
    environment = dict(os.environ, PYTHONPATH=str(REPOSITORY_ROOT))
    probe = (
        "import sys, tether.evt\n"
        "loaded = [m for m in sys.modules if m == 'pydrake' or m.startswith('pydrake.')]\n"
        "assert not loaded, loaded\n"
        "assert 'tether.physics.fleet' not in sys.modules\n"
    )
    subprocess.run(
        [sys.executable, "-c", probe], cwd=REPOSITORY_ROOT, env=environment, check=True
    )


def test_garwood_known_values():
    assert evt.poisson_interval(0) == (0.0, pytest.approx(3.689, abs=5e-4))
    assert evt.censored_upper_bound(1.0) == pytest.approx(2.996, abs=5e-4)
    assert evt.censored_upper_bound(4.0, confidence=0.9) == pytest.approx(np.log(10.0) / 4.0)
    lower, upper = evt.poisson_interval(10)
    assert lower == pytest.approx(4.7954, abs=1e-4)
    assert upper == pytest.approx(18.3904, abs=1e-4)
    rate, rate_lo, rate_hi = evt.rate_interval(10, 2.0)
    assert (rate, rate_lo, rate_hi) == pytest.approx((5.0, lower / 2.0, upper / 2.0))
    with pytest.raises(ValueError):
        evt.poisson_interval(-1)


def test_garwood_coverage_is_at_least_nominal():
    counts = np.arange(200)
    intervals = np.array([evt.poisson_interval(int(count)) for count in counts])
    for mean in np.linspace(0.2, 40.0, 120):
        covered = (intervals[:, 0] <= mean) & (mean <= intervals[:, 1])
        assert stats.poisson.pmf(counts[covered], mean).sum() >= 0.95 - 1e-9


def test_wilson_and_newcombe_published_values():
    p, lower, upper = evt.wilson_interval(81, 263)
    assert p == pytest.approx(81 / 263)
    assert (lower, upper) == pytest.approx((0.2553, 0.3662), abs=5e-5)
    assert evt.wilson_interval(0, 20)[1:] == pytest.approx((0.0, 0.1611), abs=5e-5)
    assert evt.wilson_interval(1, 29)[1:] == pytest.approx((0.0061, 0.1718), abs=5e-5)
    difference, lower, upper = evt.newcombe_difference(56, 70, 48, 80)
    assert difference == pytest.approx(0.2)
    assert (lower, upper) == pytest.approx((0.0524, 0.3339), abs=5e-5)
    assert evt.newcombe_difference(9, 10, 3, 10)[1:] == pytest.approx((0.1705, 0.8090), abs=5e-5)
    assert evt.newcombe_difference(5, 56, 0, 29)[1:] == pytest.approx((-0.0381, 0.1926), abs=5e-5)


def test_seed_bootstrap_rate_and_statistic_agree():
    generator = _generator(1)
    exposures = generator.uniform(50.0, 150.0, size=20)
    counts = generator.poisson(0.3 * exposures)
    rate, lower, upper = evt.seed_bootstrap_rate(
        counts, exposures, n_boot=400, confidence=0.99, rng=_generator(2)
    )
    assert rate == pytest.approx(counts.sum() / exposures.sum())
    assert lower < rate < upper
    assert lower < 0.3 < upper
    pooled = evt.seed_bootstrap_statistic(
        list(zip(counts, exposures)),
        lambda items: sum(c for c, _ in items) / sum(t for _, t in items),
        n_boot=400,
        confidence=0.99,
        rng=_generator(2),
    )
    assert pooled == pytest.approx((rate, lower, upper))
    assert evt.seed_bootstrap_rate(counts, exposures, n_boot=50) == evt.seed_bootstrap_rate(
        counts, exposures, n_boot=50
    )


def test_paired_bootstrap_preserves_pairing():
    generator = _generator(3)
    baseline = generator.normal(0.0, 10.0, size=30)
    treated = baseline + 0.5 + generator.normal(0.0, 0.05, size=30)
    difference, lower, upper = evt.paired_bootstrap_difference(
        treated, baseline, n_boot=500, confidence=0.99, rng=_generator(4)
    )
    assert difference == pytest.approx(np.mean(treated - baseline))
    assert lower < 0.5 < upper
    assert upper - lower < 0.1


def test_moving_block_indices_are_contiguous_blocks():
    indices = evt.moving_block_indices(103, 10, _generator(5))
    assert indices.shape == (103,)
    assert indices.min() >= 0 and indices.max() < 103
    blocks = [indices[start:start + 10] for start in range(0, 103, 10)]
    for block in blocks:
        assert np.all(np.diff(block) == 1)
        assert block[0] <= 93


HAND_TIMES = np.arange(20.0)
HAND_VALUES = np.array(
    [0, 5, 6, 0, 0, 7, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 9, 9, 0, 3], dtype=float
)


def test_runs_decluster_hand_built_series():
    peak_times, peaks = evt.runs_decluster(HAND_TIMES, HAND_VALUES, 3.0, 2.0)
    assert peak_times.tolist() == [2.0, 5.0, 10.0, 16.0]
    assert peaks.tolist() == [6.0, 7.0, 4.0, 9.0]
    peak_times, peaks = evt.runs_decluster(HAND_TIMES, HAND_VALUES, 3.0, 3.0)
    assert peak_times.tolist() == [5.0, 10.0, 16.0]
    assert peaks.tolist() == [7.0, 4.0, 9.0]
    assert evt.runs_decluster(HAND_TIMES, HAND_VALUES, 10.0, 2.0)[0].size == 0


def test_upcrossing_counts_hand_built_series():
    levels = [3.5, 5.5, 8.0, 9.0, 10.0]
    assert evt.upcrossing_counts(HAND_TIMES, HAND_VALUES, levels, 2.0).tolist() == [4, 3, 1, 1, 0]
    assert evt.upcrossing_counts(HAND_TIMES, HAND_VALUES, levels, 3.0).tolist() == [3, 2, 1, 1, 0]
    valid = np.ones(20, dtype=bool)
    valid[15] = False
    masked = evt.upcrossing_counts(HAND_TIMES, HAND_VALUES, levels, 2.0, valid)
    assert masked.tolist() == [3, 2, 0, 0, 0]
    with_gap = HAND_VALUES.copy()
    with_gap[9] = np.nan
    assert evt.upcrossing_counts(HAND_TIMES, with_gap, [3.5], 2.0).tolist() == [3]


def _naive_upcrossings(times, values, levels, run_length, valid):
    counts = []
    for level in levels:
        count, last_above = 0, None
        for index in range(values.size):
            if not valid[index]:
                continue
            if values[index] >= level:
                crossing = index > 0 and valid[index - 1] and values[index - 1] < level
                if crossing and (last_above is None or times[index] - last_above > run_length):
                    count += 1
                last_above = times[index]
        counts.append(count)
    return counts


def test_upcrossing_counts_match_naive_loop():
    generator = _generator(6)
    times = np.cumsum(generator.uniform(0.5, 1.5, size=1500))
    values = np.round(np.cumsum(generator.normal(size=1500)) * 0.3 + generator.normal(size=1500), 1)
    valid = generator.random(1500) > 0.1
    levels = np.quantile(values, np.linspace(0.05, 0.99, 8))
    for run_length in (0.0, 2.0, 7.5):
        expected = _naive_upcrossings(times, values, levels, run_length, valid)
        assert evt.upcrossing_counts(times, values, levels, run_length, valid).tolist() == expected


def test_upcrossing_counts_scale_to_campaign_series():
    generator = _generator(7)
    times = np.arange(600_000) * 0.01
    values = np.sin(0.7 * times) + 0.05 * generator.normal(size=times.size)
    levels = np.concatenate([[0.0], np.linspace(-0.5, 2.5, 149)])
    start = time.perf_counter()
    counts = evt.upcrossing_counts(times, values, levels, 1.0)
    assert time.perf_counter() - start < 10.0
    assert counts.shape == (150,)
    periods = int(np.floor(times[-1] * 0.7 / (2.0 * np.pi)))
    assert abs(counts[0] - periods) <= 2
    assert counts[-1] == 0


def test_gpd_fit_recovers_shape_and_scale():
    excesses = _gpd_sample(0.3, 2.0, 3000, seed=8)
    body = 10.0 - 5.0 * _generator(9).random(2000)
    values = np.concatenate([body, 10.0 + excesses])
    fit = evt.fit_gpd(values, 10.0)
    assert fit.n_exceedances == 3000 and fit.n_total == 5000
    assert fit.shape == pytest.approx(0.3, abs=0.08)
    assert fit.scale == pytest.approx(2.0, rel=0.1)
    reference_shape, _, reference_scale = stats.genpareto.fit(excesses, floc=0.0)
    reference_loglik = stats.genpareto.logpdf(excesses, reference_shape, 0.0, reference_scale).sum()
    assert fit.loglik >= reference_loglik - 1e-6
    assert np.sqrt(fit.covariance[0, 0]) == pytest.approx(1.3 / np.sqrt(3000), rel=0.2)
    assert np.sqrt(fit.covariance[1, 1]) == pytest.approx(2.0 * np.sqrt(2.6 / 3000), rel=0.2)
    lower, upper = evt.profile_shape_interval(values, 10.0, confidence=0.99)
    assert lower < fit.shape < upper
    assert lower < 0.3 < upper


def test_gpd_fit_handles_exponential_tail():
    values = _generator(10).exponential(1.5, size=5000)
    fit = evt.fit_gpd(values, 0.0)
    assert abs(fit.shape) < 0.05
    assert fit.scale == pytest.approx(1.5, rel=0.06)
    lower, upper = evt.profile_shape_interval(values, 0.0, confidence=0.99)
    assert lower < 0.0 < upper


def test_exceedance_probability_and_tail_rate():
    fit = GPDFit(0.5, 2.0, 1.0, 100, 1000, 0.0, np.eye(2))
    assert evt.exceedance_probability(fit, 5.0) == pytest.approx(0.25)
    assert evt.exceedance_probability(fit, 0.0) == 1.0
    assert evt.tail_rate(fit, 10.0, 5.0) == pytest.approx(2.5)
    exponential = GPDFit(0.0, 2.0, 1.0, 100, 1000, 0.0, np.eye(2))
    assert evt.exceedance_probability(exponential, 5.0) == pytest.approx(np.exp(-2.0))
    bounded = GPDFit(-0.5, 2.0, 1.0, 100, 1000, 0.0, np.eye(2))
    assert evt.exceedance_probability(bounded, 3.0) == pytest.approx(0.25)
    assert evt.exceedance_probability(bounded, 6.0) == 0.0


def test_profile_tail_rate_interval_contains_truth():
    excesses = _gpd_sample(0.3, 1.0, 2000, seed=11)
    fit = evt.fit_gpd(excesses, 0.0)
    true_rate = 0.5 * stats.genpareto.sf(20.0, 0.3)
    lower, upper = evt.profile_tail_rate_interval(excesses, 0.0, 0.5, 20.0, confidence=0.99)
    assert lower < evt.tail_rate(fit, 0.5, 20.0) < upper
    assert lower < true_rate < upper
    assert evt.profile_tail_rate_interval(excesses, 0.0, 0.5, -1.0) == (0.5, 0.5)


def test_mean_residual_life_is_linear_for_gpd():
    values = _gpd_sample(0.2, 1.0, 50_000, seed=12)
    thresholds = np.array([0.0, 1.0, 2.0])
    mean_excess, lower, upper = evt.mean_residual_life(values, thresholds, confidence=0.99)
    theory = (1.0 + 0.2 * thresholds) / 0.8
    assert np.all(lower < theory) and np.all(theory < upper)
    assert np.all(lower < mean_excess) and np.all(mean_excess < upper)
    assert np.isnan(evt.mean_residual_life(values, [1e9])[0][0])


def _mixture_sample(seed: int) -> np.ndarray:
    generator = _generator(seed)
    tail = generator.random(20_000) < 0.15
    excesses = stats.genpareto.rvs(0.3, scale=0.5, size=tail.size, random_state=generator)
    return np.where(tail, 1.0 + excesses, generator.random(tail.size))


def test_parameter_stability_and_threshold_selection():
    values = _mixture_sample(13)
    quantiles = (0.80, 0.85, 0.90, 0.925, 0.95, 0.97)
    thresholds = np.quantile(values, quantiles)
    diagnostics = evt.parameter_stability(values, thresholds, confidence=0.99)
    assert diagnostics["shape"][0] > diagnostics["shape_hi"][1]
    assert diagnostics["shape_lo"][1] < 0.3 < diagnostics["shape_hi"][1]
    assert np.allclose(diagnostics["shape"][1:], 0.3, atol=0.15)
    assert diagnostics["modified_scale"][1] == pytest.approx(0.2, abs=0.1)
    for index, threshold in enumerate(thresholds):
        fit = evt.fit_gpd(values, threshold)
        assert diagnostics["shape"][index] == pytest.approx(fit.shape)
        modified_scale = fit.scale - fit.shape * threshold
        assert diagnostics["modified_scale"][index] == pytest.approx(modified_scale)
        assert diagnostics["n_exceedances"][index] == fit.n_exceedances
    selected = evt.select_threshold(values, quantiles)
    assert selected in thresholds
    assert selected >= thresholds[1]
    small = _gpd_sample(0.3, 1.0, 1000, seed=14)
    chosen = evt.select_threshold(small, quantiles, min_exceedances=100)
    assert chosen <= np.quantile(small, 0.90)
    with pytest.raises(ValueError):
        evt.select_threshold(small, quantiles, min_exceedances=5000)


def test_hill_on_pareto_samples():
    samples = (1.0 - _generator(15).random(20_000)) ** (-1.0 / 3.0)
    alpha, lower, upper = evt.hill_estimator(samples, 1000)
    assert alpha == pytest.approx(3.0, abs=0.3)
    assert upper - alpha == pytest.approx(alpha * 1.96 / np.sqrt(1000), rel=1e-3)
    assert alpha - lower == pytest.approx(upper - alpha)
    wide_lower, wide_upper = evt.hill_estimator(samples, 1000, confidence=0.99)[1:]
    assert wide_lower < 3.0 < wide_upper
    plot = evt.hill_plot(samples, [500, 1000, 4000])
    assert plot[1] == pytest.approx(alpha)
    assert np.all(np.abs(plot - 3.0) < 0.5)


def test_tail_index_from_rates_and_log_slope_fit():
    levels = np.linspace(1.0, 3.0, 9)
    exposure = 100.0
    counts = _generator(16).poisson(50.0 * levels**-3.0 * exposure)
    alpha, lower, upper = evt.tail_index_from_rates(
        levels, counts / exposure, counts, confidence=0.99
    )
    assert lower < 3.0 < upper
    assert alpha == pytest.approx(3.0, abs=0.15)
    slope, intercept, slope_lo, slope_hi = evt.log_slope_fit(levels, 7.0 * levels**-2.5, np.ones(9))
    assert slope == pytest.approx(-2.5)
    assert intercept == pytest.approx(np.log(7.0))
    assert slope_lo == pytest.approx(-2.5) and slope_hi == pytest.approx(-2.5)


APPENDIX_A3_RHO = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8)
APPENDIX_A3_CHI_95 = (0.050, 0.105, 0.189, 0.244, 0.310, 0.392, 0.495)
APPENDIX_A3_CHI_99 = (0.010, 0.034, 0.087, 0.129, 0.188, 0.267, 0.377)


def test_gaussian_copula_chi_reproduces_appendix_a3():
    rho = np.array(APPENDIX_A3_RHO)
    assert np.round(evt.gaussian_copula_chi(0.95, rho), 3).tolist() == list(APPENDIX_A3_CHI_95)
    assert np.round(evt.gaussian_copula_chi(0.99, rho), 3).tolist() == list(APPENDIX_A3_CHI_99)
    assert evt.gaussian_copula_chi(0.999, 0.0) == pytest.approx(0.001)
    assert evt.gaussian_copula_chi(0.99, 1.0) == pytest.approx(1.0)


def test_gaussian_copula_closed_forms():
    assert evt.gaussian_copula_chibar(0.99, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert evt.gaussian_copula_chibar(1.0 - 1e-12, 0.5) == pytest.approx(0.5, abs=0.05)
    assert evt.gaussian_eta(0.5) == 0.75
    assert evt.spearman_to_pearson(1.0) == pytest.approx(1.0)
    assert evt.spearman_to_pearson(0.5) == pytest.approx(2.0 * np.sin(np.pi / 12.0))
    u = 0.975
    assert evt.gaussian_copula_chi(u, 0.3) == pytest.approx(
        stats.multivariate_normal([0.0, 0.0], [[1.0, 0.3], [0.3, 1.0]]).cdf(
            [-stats.norm.ppf(u)] * 2
        )
        / (1.0 - u),
        rel=1e-3,
    )


@pytest.mark.parametrize(
    ("u", "rho"), [(0.95, -0.3), (0.99, -0.9), (0.999, -0.5), (0.999, 0.0), (1.0 - 1e-12, 0.0)]
)
def test_gaussian_copula_tail_for_non_positive_correlation(u, rho):
    z = stats.norm.ppf(u)
    spread = np.sqrt(1.0 - rho * rho)
    joint = integrate.quad(
        lambda x: stats.norm.pdf(x) * stats.norm.sf((z - rho * x) / spread),
        z,
        np.inf,
        epsabs=0.0,
        epsrel=1e-12,
        limit=200,
    )[0]
    assert evt.gaussian_copula_chi(u, rho) == pytest.approx(joint / (1.0 - u), rel=1e-9)
    chibar = 2.0 * np.log1p(-u) / np.log(joint) - 1.0
    assert evt.gaussian_copula_chibar(u, rho) == pytest.approx(chibar, rel=1e-9, abs=1e-12)


@pytest.fixture(scope="module")
def gaussian_pseudo_observations():
    samples = _generator(17).multivariate_normal([0.0, 0.0], [[1.0, 0.5], [0.5, 1.0]], size=200_000)
    observations = evt.pseudo_observations(samples)
    return observations[:, 0], observations[:, 1]


def test_empirical_dependence_matches_gaussian_copula(gaussian_pseudo_observations):
    U, V = gaussian_pseudo_observations
    assert U.min() > 0.0 and U.max() < 1.0
    u = np.array([0.95, 0.99])
    assert np.allclose(evt.chi_empirical(u, U, V), evt.gaussian_copula_chi(u, 0.5), atol=0.025)
    assert np.allclose(evt.chibar_empirical(u, U, V), evt.gaussian_copula_chibar(u, 0.5), atol=0.04)
    eta, lower, upper = evt.eta_ledford_tawn(U, V, tail_fraction=0.05)
    assert eta == pytest.approx(evt.gaussian_eta(0.5), abs=0.05)
    assert lower < eta < upper
    rho_s = stats.spearmanr(U, V).statistic
    assert evt.spearman_to_pearson(rho_s) == pytest.approx(0.5, abs=0.01)


def test_dependence_bootstrap_intervals(gaussian_pseudo_observations):
    U, V = gaussian_pseudo_observations
    U, V = U[:20_000], V[:20_000]
    u = np.array([0.9, 0.95])
    result = evt.dependence_bootstrap(
        U, V, u, block_length=50, n_boot=200, rng=_generator(18), confidence=0.99
    )
    assert np.all(result["chi_lo"] <= result["chi"]) and np.all(result["chi"] <= result["chi_hi"])
    assert np.all(result["chibar_lo"] <= result["chibar_hi"])
    assert np.all(result["chi_lo"] < evt.gaussian_copula_chi(u, 0.5))
    assert np.all(evt.gaussian_copula_chi(u, 0.5) < result["chi_hi"])
    seed = int(_generator(19).integers(2**63))
    indices = evt.moving_block_indices(U.size, 37, np.random.default_rng(seed))
    single = evt.dependence_bootstrap(U, V, u, block_length=37, n_boot=1, rng=_generator(19))
    assert single["chi_lo"] == pytest.approx(evt.chi_empirical(u, U[indices], V[indices]))
    identical = evt.dependence_bootstrap(U, U, u, block_length=50, n_boot=20, rng=_generator(20))
    marginal = np.array([np.mean(U > level) for level in u])
    assert identical["chi"] == pytest.approx(marginal / (1.0 - u))
    sparse = evt.chibar_empirical([0.999], U[:100], 1.0 - U[:100])
    assert np.isnan(sparse[0])


def test_dependence_bootstrap_keeps_replicates_without_joint_exceedances():
    generator = _generator(35)
    U = (generator.permutation(2000) + 1.0) / 2001.0
    V = (generator.permutation(2000) + 1.0) / 2001.0
    level = np.sort(np.minimum(U, V))[-2]
    assert evt.chi_empirical([level], U, V)[0] * (1.0 - level) == pytest.approx(1.0 / 2000.0)
    result = evt.dependence_bootstrap(U, V, [level], block_length=1, n_boot=400, rng=_generator(36))
    assert np.isfinite(result["chibar"][0])
    assert result["chi_lo"][0] == 0.0
    assert result["chibar_lo"][0] == -1.0
    assert result["chibar_hi"][0] > result["chibar"][0]


def test_anderson_darling_normal():
    normal = _generator(21).normal(3.0, 2.0, size=100)
    statistic, p_value = evt.anderson_darling_normal(normal)
    assert statistic == pytest.approx(stats.anderson(normal).statistic)
    assert p_value > 0.05
    assert evt.anderson_darling_normal(_generator(22).exponential(size=200))[1] < 1e-6
    p_values = np.array(
        [evt.anderson_darling_normal(_generator(1000 + i).normal(size=50))[1] for i in range(200)]
    )
    assert 0.01 <= np.mean(p_values < 0.05) <= 0.10
    assert 0.4 <= np.mean(p_values < 0.5) <= 0.6


def test_ks_parametric_bootstrap():
    generator = _generator(23)
    assert evt.rayleigh_scale_mle(generator.rayleigh(2.0, 5000)) == pytest.approx(2.0, rel=0.03)
    rayleigh = generator.rayleigh(2.0, 300)
    statistic, p_value = evt.ks_parametric_bootstrap(
        rayleigh, "rayleigh", n_boot=199, rng=_generator(24)
    )
    assert 0.0 < statistic < 0.1
    assert p_value > 0.01
    exponential = generator.exponential(2.0, 300)
    wrong_family = evt.ks_parametric_bootstrap(
        exponential, "rayleigh", n_boot=199, rng=_generator(25)
    )
    assert wrong_family[1] <= 0.02
    right_family = evt.ks_parametric_bootstrap(
        exponential, "exponential", n_boot=199, rng=_generator(26)
    )
    assert right_family[1] > 0.01
    excesses = _gpd_sample(0.3, 1.0, 300, seed=27)
    assert evt.ks_parametric_bootstrap(excesses, "gpd", n_boot=99, rng=_generator(28))[1] > 0.01
    with pytest.raises(ValueError):
        evt.ks_parametric_bootstrap(rayleigh, "weibull")


@pytest.fixture(scope="module")
def forecasts():
    generator = _generator(29)
    p = generator.uniform(0.02, 0.98, size=20_000)
    y = (generator.random(p.size) < p).astype(float)
    logit = np.log(p / (1.0 - p))
    overconfident = 1.0 / (1.0 + np.exp(-2.0 * logit))
    return p, y, overconfident


def test_logistic_recalibration(forecasts):
    p, y, overconfident = forecasts
    calibrated = evt.logistic_recalibration(p, y, confidence=0.99)
    assert calibrated["slope_lo"] < 1.0 < calibrated["slope_hi"]
    assert calibrated["intercept_lo"] < 0.0 < calibrated["intercept_hi"]
    dispersed = evt.logistic_recalibration(overconfident, y)
    assert dispersed["slope_hi"] < 1.0
    assert dispersed["slope"] == pytest.approx(0.5, abs=0.05)


def test_calibration_error_and_reliability_bands(forecasts):
    p, y, overconfident = forecasts
    assert evt.expected_calibration_error(p, y) < 0.02
    assert evt.expected_calibration_error(overconfident, y) > 0.05
    ece, lower, upper = evt.ece_bootstrap(overconfident, y, n_boot=100, rng=_generator(30))
    assert lower < ece < upper
    calibrated_bins = evt.reliability_bins(p, y, family_confidence=0.99)
    assert len(calibrated_bins) == 10
    assert all(entry["inside"] for entry in calibrated_bins)
    first = calibrated_bins[0]
    in_bin = p < 0.1
    expected = evt.wilson_interval(int(y[in_bin].sum()), int(in_bin.sum()), 1.0 - 0.01 / 10)
    assert (first["observed"], first["lo"], first["hi"]) == pytest.approx(expected)
    assert first["count"] == int(in_bin.sum())
    assert sum(not entry["inside"] for entry in evt.reliability_bins(overconfident, y)) >= 7


def test_auroc_noise_ties_and_bootstrap():
    generator = _generator(31)
    scores = generator.normal(size=4000)
    labels = generator.random(4000) < 0.3
    auc, lower, upper = evt.auroc_interval(
        scores, labels, n_boot=200, rng=_generator(32), confidence=0.99
    )
    assert auc == pytest.approx(0.5, abs=0.03)
    assert lower < 0.5 < upper
    tied = np.round(generator.normal(size=500), 1)
    tied_labels = generator.random(500) < 0.4
    u_statistic = stats.mannwhitneyu(tied[tied_labels], tied[~tied_labels]).statistic
    assert evt.auroc(tied, tied_labels) == pytest.approx(
        u_statistic / (tied_labels.sum() * (~tied_labels).sum())
    )
    assert evt.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    replay = _generator(33)
    positive, negative = tied[tied_labels], np.sort(tied[~tied_labels])
    resampled_positive = positive[replay.integers(0, positive.size, size=positive.size)]
    resampled_negative = negative[replay.integers(0, negative.size, size=negative.size)]
    expected = evt.auroc(
        np.concatenate([resampled_positive, resampled_negative]),
        np.concatenate([np.ones(positive.size), np.zeros(negative.size)]),
    )
    single = evt.auroc_interval(tied, tied_labels, n_boot=1, rng=_generator(33))
    assert single[1] == pytest.approx(expected)


def test_acer_counts_on_hand_built_series():
    series = [np.array([0.0, 0.0, 2.0, 0.0, 0.0, 3.0]), np.array([2.0, 0.0])]
    assert evt.acer_estimates(series[:1], [1.5], 1).tolist() == [pytest.approx(2 / 6)]
    assert evt.acer_estimates(series[:1], [1.5], 2).tolist() == [pytest.approx(2 / 5)]
    assert evt.acer_estimates(series[:1], [1.5], 3).tolist() == [pytest.approx(2 / 4)]
    assert evt.acer_estimates(series, [1.5], 2).tolist() == [pytest.approx(2 / 6)]
    with pytest.raises(ValueError):
        evt.acer_estimates(series, [1.5], 4)


def test_fit_acer_recovers_noise_free_tails():
    levels = np.linspace(3.0, 8.0, 21)
    fit = evt.fit_acer(levels, 0.8 * np.exp(-1.3 * (levels - 2.0) ** 1.7), tail_start=3.0)
    assert (fit.q, fit.a, fit.b, fit.c) == pytest.approx((0.8, 1.3, 2.0, 1.7), rel=1e-6)
    assert evt.acer_extrapolate(fit, 10.0) == pytest.approx(0.8 * np.exp(-1.3 * 8.0**1.7), rel=1e-6)
    exponential = evt.fit_acer(levels, np.exp(-levels), tail_start=3.0)
    assert exponential.c == pytest.approx(1.0, rel=1e-6)
    assert evt.acer_extrapolate(exponential, 14.0) == pytest.approx(np.exp(-14.0), rel=1e-6)


def test_fit_acer_reaches_the_least_squares_optimum_on_noisy_rates():
    samples = _generator(53).exponential(size=100_000)
    levels = np.linspace(0.5, 8.0, 31)
    epsilon = evt.acer_estimates([samples], levels, 1)
    fit = evt.fit_acer(levels, epsilon, tail_start=2.0)
    tail = levels >= 2.0
    eta, log_epsilon = levels[tail], np.log(epsilon[tail])
    weights = epsilon[tail] / epsilon[tail].max()
    fitted = np.sum(weights * (log_epsilon - np.log(fit.q) + fit.a * (eta - fit.b) ** fit.c) ** 2)
    span = eta.max() - 2.0
    offsets = span * np.logspace(-3.0, 1.0, 150)
    shapes = np.logspace(np.log10(0.25), 1.0, 150)
    t = (eta - (2.0 - offsets)[:, None, None]) ** shapes[None, :, None]
    mean_t = np.sum(weights * t, axis=-1, keepdims=True) / weights.sum()
    mean_y = np.sum(weights * log_epsilon) / weights.sum()
    a = -np.sum(weights * (t - mean_t) * (log_epsilon - mean_y), axis=-1, keepdims=True) / np.sum(
        weights * (t - mean_t) ** 2, axis=-1, keepdims=True
    )
    grid = np.sum(weights * (log_epsilon - mean_y + a * (t - mean_t)) ** 2, axis=-1)
    assert fitted <= np.min(np.where(a[..., 0] > 0.0, grid, np.inf)) * (1.0 + 1e-6)


def test_acer_recovers_poisson_rate_of_iid_exponential():
    samples = _generator(34).exponential(size=1_000_000)
    series = [samples[:500_000], samples[500_000:]]
    levels = np.linspace(1.0, 10.0, 37)
    well_sampled = (levels >= 5.0) & (levels <= 7.0)
    for k in (1, 2, 3):
        epsilon = evt.acer_estimates(series, levels, k)
        assert np.allclose(epsilon[well_sampled], np.exp(-levels[well_sampled]), rtol=0.1)
        fit = evt.fit_acer(levels, epsilon, tail_start=5.0)
        assert fit.b < 5.0 and fit.a > 0.0 and np.isfinite(fit.q)
        assert evt.acer_extrapolate(fit, np.array([6.0, 8.0])) == pytest.approx(
            np.exp(-np.array([6.0, 8.0])), rel=0.1
        )
        assert 0.5 < evt.acer_extrapolate(fit, 12.0) / np.exp(-12.0) < 2.0
