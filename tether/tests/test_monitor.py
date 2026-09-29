"""Phase 5 monitor: hazard, oracle precursor, outcome labels, and forecast metrics."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import special, stats

from tether.monitor import hazard as hz
from tether.monitor import metrics as mt
from tether.monitor import oracle as orc
from tether.monitor import outcomes as oc
from tether.physics import constants
from tether.theory.excursion import EFFECTIVE_MASS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SLOW = pytest.mark.skipif(os.environ.get("TETHER_SLOW") != "1", reason="set TETHER_SLOW=1")
TINY = np.diag([1e-12, 1e-12])
SERIES_STEP = 1e-3


def _generator(*entropy: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([2026, 5, *entropy]))


def _ticks(duration: float, step: float = 0.1) -> np.ndarray:
    return np.round(step * np.arange(int(round(duration / step)) + 1), 10)


def _parabolic_truth(excursions, duration: float = 8.0):
    """1 ms e, e' of slack parabolas e = -u tau + a tau^2 / 2 (return speed u at 2u/a)."""
    time = np.round(SERIES_STEP * np.arange(int(round(duration / SERIES_STEP)) + 1), 10)
    elongation = np.full(time.size, 0.01)
    rate = np.zeros(time.size)
    for onset, speed, acceleration in excursions:
        arrival = onset + 2.0 * speed / acceleration
        window = (time >= onset - 0.05) & (time <= arrival + 0.05)
        tau = time[window] - onset
        elongation[window] = -speed * tau + 0.5 * acceleration * tau**2
        rate[window] = -speed + acceleration * tau
    return time, elongation, rate


def _parabola_at(ticks, excursions):
    e = np.full(ticks.size, 0.01)
    edot = np.zeros(ticks.size)
    for onset, speed, acceleration in excursions:
        inside = (ticks > onset) & (ticks < onset + 2.0 * speed / acceleration)
        tau = ticks[inside] - onset
        e[inside] = -speed * tau + 0.5 * acceleration * tau**2
        edot[inside] = -speed + acceleration * tau
    return e, edot


EXCURSIONS = ((1.0005, 0.3, 2.0), (3.0237, 1.5, 1.2))


def test_monitor_imports_no_drake():
    environment = dict(os.environ, PYTHONPATH=str(REPOSITORY_ROOT))
    probe = (
        "import sys\n"
        "import tether.monitor.hazard, tether.monitor.metrics\n"
        "import tether.monitor.oracle, tether.monitor.outcomes\n"
        "loaded = [m for m in sys.modules if m == 'pydrake' or m.startswith('pydrake.')]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", probe], cwd=REPOSITORY_ROOT, env=environment, check=True)


def test_deterministic_precursor_matches_constant_acceleration():
    rng = hz.hazard_generator(3, 0)
    assert hz.tick_hazard(-0.5, 0.0, TINY, 2.0, 1e-9, 1.40, rng) == 1.0
    assert hz.tick_hazard(-0.5, 0.0, TINY, 2.0, 1e-9, 1.43, rng) == 0.0
    assert hz.tick_hazard(-5.0, 0.0, TINY, 2.0, 1e-9, 0.1, rng) == 0.0
    assert hz.tick_hazard(-0.1, 1.0, TINY, 0.0, 1e-9, 0.99, rng) == 1.0
    assert hz.tick_hazard(-0.3, -1.0, TINY, 0.5, 1e-9, 0.1, rng) == 0.0
    assert hz.tick_rice(-0.5, 0.0, TINY, 2.0, 1e-9, 1.40) == pytest.approx(1.0, abs=1e-6)
    depth, v_b, a_hat, sigma_a = 0.3, 1.0, 1.7, 0.2
    threshold = max(v_b**2 / (2.0 * depth), 2.0 * depth / hz.DEFAULT_HORIZON**2)
    expected = stats.norm.sf((threshold - a_hat) / sigma_a)
    value = hz.tick_hazard(-depth, 0.0, TINY, a_hat, sigma_a, v_b, rng)
    assert abs(value - expected) < 4.0 * np.sqrt(expected * (1.0 - expected) / hz.DEFAULT_SAMPLES)
    assert hz.tick_rice(-depth, 0.0, TINY, a_hat, sigma_a, v_b) == pytest.approx(expected, abs=1e-6)
    state = rng.bit_generator.state
    assert np.isnan(hz.tick_hazard(np.nan, 0.0, TINY, 1.0, 0.1, v_b, rng))
    assert np.isnan(hz.tick_hazard(-0.3, 0.0, [[1.0, 2.0], [2.0, 1.0]], 1.0, 0.1, v_b, rng))
    assert rng.bit_generator.state == state


def _synthetic_output(ticks: int = 30) -> dict[str, np.ndarray]:
    slack = np.zeros((ticks, 2), dtype=bool)
    slack[5:15] = True
    slack[20:26] = True
    step = np.arange(ticks, dtype=float)[:, None] * np.ones((1, 2))
    sigma = np.broadcast_to(np.array([[4e-4, 1e-4], [1e-4, 1e-2]]), (ticks, 2, 2, 2))
    return {
        "time": _ticks(0.1 * (ticks - 1)),
        "slack": slack,
        "e_hat": np.where(slack, -0.3 - 0.005 * step, np.nan),
        "edot_hat": np.where(slack, -0.1 + 0.01 * step, np.nan),
        "sigma": np.where(slack[..., None, None], sigma, np.nan),
        "a_hat": np.where(slack, 1.6, np.nan),
        "sigma_a": np.where(slack, 0.4, np.nan),
    }


def test_hazard_series_is_reproducible_and_advances_in_tick_order():
    output = _synthetic_output()
    first = hz.hazard_series(output, 1.0, 11, 0)
    again = hz.hazard_series(SimpleNamespace(**output), 1.0, 11, 0)
    np.testing.assert_array_equal(first.hazard, again.hazard)
    np.testing.assert_array_equal(first.rice, again.rice)
    slack = output["slack"][:, 0]
    assert np.all(first.hazard[~slack] == 0.0) and np.all(first.rice[~slack] == 0.0)
    assert np.all((first.hazard[slack] > 0.05) & (first.hazard[slack] < 0.95))
    rng = hz.hazard_generator(11, 0)
    manual = [
        hz.tick_hazard(
            output["e_hat"][k, 0],
            output["edot_hat"][k, 0],
            output["sigma"][k, 0],
            output["a_hat"][k, 0],
            output["sigma_a"][k, 0],
            1.0,
            rng,
        )
        for k in np.flatnonzero(slack)
    ]
    np.testing.assert_array_equal(first.hazard[slack], manual)
    se = np.sqrt(first.rice[slack] * (1.0 - first.rice[slack]) / hz.DEFAULT_SAMPLES)
    assert np.all(np.abs(first.hazard[slack] - first.rice[slack]) < 5.0 * se + 1e-3)
    fleet = hz.fleet_hazard(output, 1.0, 11, rice=False)
    np.testing.assert_array_equal(fleet.hazard[:, 0], first.hazard)
    assert not np.array_equal(fleet.hazard[:, 1], fleet.hazard[:, 0])
    assert np.all(np.isnan(fleet.rice))
    other = hz.hazard_series(output, 1.0, 12, 0, rice=False)
    assert not np.array_equal(other.hazard, first.hazard)


def test_outcome_labeller_is_exact_on_parabolic_excursions():
    series_time, elongation, rate = _parabolic_truth(EXCURSIONS)
    second = _parabolic_truth(EXCURSIONS[:1])
    ticks = _ticks(8.0)
    e_ticks, _ = _parabola_at(ticks, EXCURSIONS)
    slack = np.column_stack([e_ticks < 0.0, _parabola_at(ticks, EXCURSIONS[:1])[0] < 0.0])
    labels = oc.tick_outcomes(
        ticks,
        slack,
        series_time,
        np.column_stack([elongation, second[1]]),
        np.column_stack([rate, second[2]]),
        v_b=1.0,
    )
    arrivals = [onset + 2.0 * speed / acceleration for onset, speed, acceleration in EXCURSIONS]
    crossings = oc.true_upcrossings(series_time, elongation, rate)
    np.testing.assert_allclose(crossings.time, arrivals, atol=2e-6)
    np.testing.assert_allclose(crossings.speed, [0.3, 1.5], atol=5e-6)
    first_leg = ticks <= 1.3
    second_leg = (ticks >= 3.6) & (ticks <= 5.5)
    np.testing.assert_array_equal(labels.crossed[:, 0], first_leg | second_leg)
    np.testing.assert_array_equal(labels.crossed[:, 1], first_leg)
    np.testing.assert_allclose(labels.crossing_time[first_leg, 0], arrivals[0], atol=2e-6)
    np.testing.assert_allclose(labels.closing_speed[second_leg, 0], 1.5, atol=5e-6)
    np.testing.assert_array_equal(labels.label[:, 0], second_leg)
    assert not labels.label[:, 1].any()
    np.testing.assert_array_equal(labels.benign()[:, 0], ~second_leg)
    intervals = oc.slack_intervals(ticks, slack)
    np.testing.assert_array_equal(intervals.vessel, [0, 0, 1])
    np.testing.assert_array_equal(ticks[intervals.first], [1.1, 3.1, 1.1])
    np.testing.assert_array_equal(ticks[intervals.last], [1.3, 5.5, 1.3])
    truth = oc.interval_outcomes(
        intervals,
        series_time,
        np.column_stack([elongation, second[1]]),
        np.column_stack([rate, second[2]]),
        v_b=1.0,
        breaking_strength=1e9,
    )
    expected = [arrivals[0], arrivals[1], arrivals[0]]
    np.testing.assert_allclose(truth.reengagement_time, expected, atol=2e-6)
    np.testing.assert_allclose(truth.reengagement_speed, [0.3, 1.5, 0.3], atol=5e-6)
    np.testing.assert_array_equal(truth.dangerous, [False, True, False])
    np.testing.assert_array_equal(truth.benign(), [True, False, True])
    assert not truth.severed.any()


def test_interval_reengagement_is_its_own_not_the_next_excursions():
    """A debounced flag outlives the true re-engagement: a short excursion's only slack tick can
    follow its return, and a later dangerous upcrossing must not be attributed to it."""
    short, fast = (1.005, 0.1, 3.0), (1.5, 1.6, 3.2)
    series_time, elongation, rate = _parabolic_truth((short, fast), duration=6.0)
    ticks = _ticks(6.0)
    slack = (np.isclose(ticks, 1.1)) | ((ticks > 1.5) & (ticks <= 2.55))
    onset = np.where(ticks < 1.3, 1.02, 1.52)
    onset = np.where(slack, onset, np.nan)
    intervals = oc.slack_intervals(ticks, slack, onset)
    level = 10_000.0
    truth = oc.interval_outcomes(intervals, series_time, elongation, rate, 1.0, level)
    arrivals = [onset_ + 2.0 * speed / accel for onset_, speed, accel in (short, fast)]
    np.testing.assert_allclose(truth.reengagement_time, arrivals, atol=2e-5)
    np.testing.assert_allclose(truth.reengagement_speed, [0.1, 1.6], atol=2e-5)
    np.testing.assert_array_equal(truth.severed, [False, True])
    np.testing.assert_array_equal(truth.count, [1, 1])
    blind = oc.interval_outcomes(
        oc.slack_intervals(ticks, slack), series_time, elongation, rate, 1.0, level
    )
    assert not blind.reengaged[0] and blind.severed.sum() == 1
    hazard = np.where(slack, 0.9, 0.0)
    labels = oc.tick_outcomes(ticks, slack, series_time, elongation, rate, 1.0)
    table = mt.monitor_table(hazard, labels, intervals, truth)
    leads = mt.lead_times(
        table.forecast, table.time, table.interval, table.reengagement_time, table.severed, 0.5
    )
    assert np.isnan(leads[0]) and leads[1] == pytest.approx(arrivals[1] - 1.6, abs=1e-6)


def test_interval_severs_on_any_of_its_reengagements():
    """A sub-tick severing engagement followed by a slow one inside the same flag run."""
    violent, slow = (5.005, 1.5, 60.0), (5.2, 0.3, 2.0)
    series_time, elongation, rate = _parabolic_truth((violent, slow), duration=8.0)
    ticks = _ticks(8.0)
    slack = (ticks >= 5.1 - 1e-9) & (ticks <= 5.5 + 1e-9)
    intervals = oc.slack_intervals(ticks, slack, np.where(slack, 5.02, np.nan))
    truth = oc.interval_outcomes(intervals, series_time, elongation, rate, 1.0, 10_000.0)
    first = violent[0] + 2.0 * violent[1] / violent[2]
    assert truth.count[0] == 2 and truth.severed[0]
    assert truth.reengagement_time[0] == pytest.approx(first, abs=2e-6)
    assert truth.peak_tension[0] > 10_000.0 and truth.dangerous[0]
    calm = oc.interval_outcomes(intervals, series_time, elongation, rate, 1.0, 1e9)
    last = slow[0] + 2.0 * slow[1] / slow[2]
    assert not calm.severed[0] and calm.reengagement_time[0] == pytest.approx(last, abs=2e-6)


def test_slack_intervals_split_where_the_onset_changes():
    ticks = _ticks(0.9)
    slack = np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 0], dtype=bool)
    onset = np.array([np.nan, 0.05, 0.05, 0.05, np.nan, 0.45, 0.45, 0.72, 0.72, np.nan])
    intervals = oc.slack_intervals(ticks, slack, onset)
    np.testing.assert_array_equal(intervals.first, [1, 5, 7])
    np.testing.assert_array_equal(intervals.last, [3, 6, 8])
    np.testing.assert_array_equal(intervals.onset_time, [0.05, 0.45, 0.72])
    np.testing.assert_array_equal(intervals.tick_index()[:, 0], [-1, 0, 0, 0, -1, 1, 1, 2, 2, -1])
    assert oc.slack_intervals(ticks, slack).count == 2


def test_engagement_peak_matches_the_cable_tracker():
    pytest.importorskip("pydrake")
    from tether.physics.cable import CableEventTracker, CableMode

    stiffness, damping = constants.CABLE_STIFFNESS, constants.CABLE_DAMPING
    omega = np.sqrt(stiffness / EFFECTIVE_MASS)
    zeta = damping / (2.0 * np.sqrt(stiffness * EFFECTIVE_MASS))
    damped = omega * np.sqrt(1.0 - zeta**2)
    time = np.round(SERIES_STEP * np.arange(6001), 10)
    elongation = np.full(time.size, 0.0058)
    rate = np.zeros(time.size)
    tracker = CableEventTracker(
        stiffness=stiffness,
        damping=damping,
        cable=0,
        capacity=64,
        mode=CableMode.RECORDING,
        break_threshold=None,
    )
    for onset, speed, acceleration in ((1.0003, 0.8, 1.0), (3.5007, 1.6, 1.4)):
        arrival = onset + 2.0 * speed / acceleration
        leg = (time >= onset) & (time < arrival)
        tau = time[leg] - onset
        elongation[leg] = -speed * tau + 0.5 * acceleration * tau**2
        rate[leg] = -speed + acceleration * tau
        ring = (time >= arrival) & (time < arrival + 0.4)
        s = time[ring] - arrival
        swing = speed / damped * np.sin(damped * s) - 0.0058 * np.cos(damped * s)
        elongation[ring] = 0.0058 + np.exp(-zeta * omega * s) * swing
        rate[ring] = np.gradient(elongation[ring], SERIES_STEP)
    tension = stiffness * elongation + damping * rate
    for sample in zip(time, elongation, rate, tension):
        tracker.sample(*(float(value) for value in sample))
    records = tracker.state.reengagement_ring.drain()
    crossings = oc.true_upcrossings(time, elongation, rate)
    assert crossings.time.size == len(records) >= 2
    np.testing.assert_allclose(crossings.time, [r.t_up for r in records], rtol=0, atol=1e-12)
    peaks = [
        oc.engagement_peak(elongation, tension, int(index), float(speed))
        for index, speed in zip(crossings.index, crossings.speed)
    ]
    np.testing.assert_allclose(peaks, [r.T_peak for r in records], rtol=1e-12)
    ticks = _ticks(6.0)
    slack = np.interp(ticks, time, elongation) < 0.0
    intervals = oc.slack_intervals(ticks, slack)
    level = 10_000.0
    truth = oc.interval_outcomes(intervals, time, elongation, rate, 1.0, level)
    by_time = {r.t_up: r.T_peak for r in records}
    assert truth.reengaged.sum() >= 2 and not truth.reengaged[-1]
    for when, peak in zip(truth.reengagement_time[truth.reengaged], truth.peak_tension):
        assert peak == pytest.approx(by_time[when], rel=1e-12)
    expected = [by_time.get(when, 0.0) > level for when in truth.reengagement_time]
    np.testing.assert_array_equal(truth.severed, expected)
    assert truth.severed.any() and not truth.severed.all()


def test_oracle_slope_estimate():
    ticks = _ticks(1.9)
    slack = np.zeros(ticks.size, dtype=bool)
    slack[2:16] = True
    onset = np.where(slack, 0.15, np.nan)
    line = 1.7 * (ticks - 0.15) - 0.4
    precursor = orc.oracle_precursor(ticks, slack, onset, -0.1 * ticks, line)
    a0 = orc.MISSION_PRETENSION / EFFECTIVE_MASS
    np.testing.assert_allclose(precursor.a_hat[2:4, 0], a0)
    np.testing.assert_allclose(precursor.a_hat[4:16, 0], 1.7, rtol=1e-9)
    np.testing.assert_allclose(precursor.sigma_a[2:16, 0], orc.SIGMA_A_FLOOR)
    assert np.all(np.isnan(precursor.a_hat[~slack, 0]))
    np.testing.assert_array_equal(precursor.e_hat[slack, 0], -0.1 * ticks[slack])
    np.testing.assert_allclose(
        precursor.sigma[5, 0], np.diag([orc.ORACLE_SIGMA_E**2, orc.ORACLE_SIGMA_EDOT**2])
    )
    noisy = line + 0.3 * _generator(3).standard_normal(ticks.size)
    precursor = orc.oracle_precursor(ticks, slack, onset, -0.1 * ticks, noisy)
    for tick in range(4, 16):
        window = slice(max(2, tick - 3), tick + 1)
        fit = stats.linregress(ticks[window], noisy[window])
        assert precursor.a_hat[tick, 0] == pytest.approx(fit.slope, rel=1e-9)
        expected = max(fit.stderr, orc.SIGMA_A_FLOOR)
        assert precursor.sigma_a[tick, 0] == pytest.approx(expected, rel=1e-9)
    assert np.any(precursor.sigma_a[4:16, 0] > orc.SIGMA_A_FLOOR)


def test_oracle_pipeline_on_parabolic_truth():
    series_time, elongation, rate = _parabolic_truth(EXCURSIONS)
    ticks = _ticks(8.0)
    e_true, edot_true = _parabola_at(ticks, EXCURSIONS)
    slack = e_true < 0.0
    legs = [(ticks > 1.0) & (ticks < 1.4), (ticks > 3.0) & (ticks < 5.6)]
    onset = np.where(slack, np.select(legs, [1.0005, 3.0237], np.nan), np.nan)
    precursor = orc.oracle_precursor(ticks, slack, onset, e_true, edot_true)
    risk = hz.hazard_series(precursor, 1.0, 21, 0, rice=False)
    labels = oc.tick_outcomes(ticks, slack, series_time, elongation, rate, 1.0)
    first_leg = slack & (ticks < 2.0)
    assert np.all(risk.hazard[first_leg] == 0.0) and not labels.label[first_leg, 0].any()
    settled = slack & (ticks >= 3.9)
    assert np.all(risk.hazard[settled] == 1.0) and labels.label[settled, 0].all()
    intervals = oc.slack_intervals(ticks, slack, onset)
    truth = oc.interval_outcomes(intervals, series_time, elongation, rate, 1.0, 1e9)
    table = mt.monitor_table(risk.hazard, labels, intervals, truth, seed=4)
    other = mt.monitor_table(risk.hazard, labels, intervals, truth, seed=9)
    pooled = mt.concatenate_tables([table, other])
    assert pooled.n_intervals == 4 and pooled.forecast.size == 2 * slack.sum()
    np.testing.assert_array_equal(pooled.interval_seed, [4, 4, 9, 9])
    np.testing.assert_array_equal(np.unique(pooled.interval), [0, 1, 2, 3])
    np.testing.assert_array_equal(pooled.outcome[: table.forecast.size], labels.label[slack, 0])


def _prop10_ticks(rng: np.random.Generator, count: int, samples: int):
    """Forecasts from the true conditional law (Prop 10) and labels from the 1 ms truth."""
    sigma = np.diag([0.03**2, 0.15**2])
    spread_a, v_b = 0.3, 1.0
    horizon_grid = np.round(SERIES_STEP * np.arange(2201), 10)
    hazard_rng = hz.hazard_generator(5, 0)
    means = np.column_stack(
        [
            rng.uniform(-0.6, -0.05, count),
            rng.uniform(-1.0, 1.0, count),
            rng.uniform(0.5, 2.5, count),
        ]
    )
    labels, forecasts = np.zeros(count, dtype=bool), np.zeros(count)
    for row, (e0, v0, a0) in enumerate(means):
        forecasts[row] = hz.tick_hazard(e0, v0, sigma, a0, spread_a, v_b, hazard_rng, samples)
        truth = rng.multivariate_normal([e0, v0], sigma)
        acceleration = a0 + spread_a * rng.standard_normal()
        path = truth[0] + horizon_grid * (truth[1] + 0.5 * acceleration * horizon_grid)
        speed = truth[1] + acceleration * horizon_grid
        outcome = oc.tick_outcomes([0.0], [True], horizon_grid, path, speed, v_b)
        labels[row] = outcome.label[0, 0]
    return labels, forecasts


def test_honest_posterior_hazard_is_calibrated_against_labelled_truth():
    labels, forecasts = _prop10_ticks(_generator(4), 300, 1024)
    assert 0.1 < np.mean(labels) < 0.9
    assert abs(np.mean(forecasts) - np.mean(labels)) < 3.0 * np.sqrt(0.25 / labels.size)


def test_calibrated_forecasts_recalibrate_to_identity():
    rng = _generator(1)
    clusters, per = 1500, 8
    logit = rng.normal(-1.5, 1.2, (clusters, 1)) + rng.normal(0.0, 0.5, (clusters, per))
    forecast = special.expit(logit).ravel()
    outcome = rng.random(forecast.size) < forecast
    cluster = np.repeat(np.arange(clusters), per)
    report = mt.calibration_report(forecast, outcome, cluster, n_boot=300, rng=7)
    assert report.slope_lo <= 1.0 <= report.slope_hi
    assert report.intercept_lo <= 0.0 <= report.intercept_hi
    assert report.large_intercept_lo <= 0.0 <= report.large_intercept_hi
    assert report.ece < 0.02 and report.ece_lo <= report.ece <= report.ece_hi < 0.03
    assert report.n_outside <= 1 and report.n_outside_wilson <= 1
    assert report.top_bin_ratio_lo <= 1.0 <= report.top_bin_ratio_hi
    assert report.top_three_ratio_lo <= 1.0 <= report.top_three_ratio_hi
    assert report.auroc_lo < report.auroc < report.auroc_hi and report.auroc > 0.7
    assert report.n_clusters == clusters and report.n_ticks == forecast.size
    clipped = np.clip(forecast, mt.FORECAST_CLIP, 1.0 - mt.FORECAST_CLIP)
    weights = np.ones(forecast.size)
    logit = special.logit(clipped)
    beta = mt._weighted_logistic(logit, outcome.astype(float), weights, np.zeros(2))
    np.testing.assert_allclose(beta, [report.intercept, report.slope], atol=1e-8)
    assert report == mt.calibration_report(forecast, outcome, cluster, n_boot=300, rng=7)


def test_overconfident_posterior_under_forecasts_as_prop12_predicts():
    rng = _generator(2)
    ratio, count, per = 2.0, 40_000, 5
    spread, v_b = 0.4, 1.0
    mean = v_b - rng.uniform(1.4, 1.6, count) * spread
    outcome = mean + spread * rng.standard_normal(count) > v_b
    reported = spread / np.sqrt(ratio)
    forecast = stats.norm.sf((v_b - mean) / reported)
    covariance = np.array([[0.01, 0.002], [0.002, spread**2]])
    errors = rng.multivariate_normal(np.zeros(2), covariance, size=count)
    nees = mt.tick_nees(errors[:, 0], errors[:, 1], covariance / ratio, 0.0, 0.0)
    cluster = np.repeat(np.arange(count // per), per)
    interval = np.bincount(cluster, weights=nees) / per
    summary = mt.nees_summary(interval, np.arange(interval.size) % 40, n_boot=300)
    assert summary.verdict == "above"
    assert summary.mean == pytest.approx(2.0 * ratio, rel=0.05)
    report = mt.calibration_report(forecast, outcome, cluster, n_boot=200, rng=3)
    assert report.large_intercept_lo > 0.0
    assert report.top_bin_ratio_lo > 1.0 and report.top_three_ratio_lo > 1.0
    top = np.argsort(-forecast, kind="stable")[: int(np.ceil(0.1 * count))]
    prediction = mt.overconfidence_prediction(
        summary.mean, v_b, reported, mean=mean[top], weights=forecast[top]
    )
    assert prediction.variance_ratio == pytest.approx(ratio, rel=0.05)
    assert prediction.leading > 1.0 and prediction.agrees(report.top_bin_ratio)
    assert report.top_bin_ratio == pytest.approx(prediction.exact, rel=0.2)


def test_nees_is_consistent_for_an_honest_posterior():
    rng = _generator(6)
    ticks = _ticks(4.9)
    slack = np.zeros((ticks.size, 3), dtype=bool)
    slack[3:15] = slack[20:31] = slack[35:47] = True
    covariance = np.array([[4e-4, 2e-4], [2e-4, 9e-2]])
    values = []
    seeds = []
    for seed in range(40):
        errors = rng.multivariate_normal(np.zeros(2), covariance, size=slack.shape)
        tick = mt.tick_nees(errors[..., 0], errors[..., 1], covariance, 0.0, 0.0)
        intervals = oc.slack_intervals(ticks, slack)
        average = mt.interval_nees(tick, intervals)
        first = intervals.first[0]
        assert average[0] == pytest.approx(tick[first : intervals.last[0] + 1, 0].mean())
        values.append(average)
        seeds.append(np.full(average.size, seed))
    summary = mt.nees_summary(np.concatenate(values), np.concatenate(seeds), n_boot=500)
    assert summary.verdict == "inside" and summary.n_seeds == 40 and summary.n_intervals == 360
    assert summary.lo < summary.mean < summary.hi and summary.band_lo < 2.0 < summary.band_hi
    assert np.isnan(mt.tick_nees(0.1, 0.1, np.zeros((2, 2)), 0.0, 0.0))


def test_false_alarm_threshold_and_lead_time():
    ticks_per, count = 10, 100
    interval = np.repeat(np.arange(count), ticks_per)
    time = np.tile(0.1 * np.arange(ticks_per), count) + 10.0 * interval
    forecast = np.zeros(interval.size)
    benign = np.ones(interval.size, dtype=bool)
    severing = np.arange(20)
    for row in severing:
        span = interval == row
        forecast[span] = np.minimum(1.0, 0.2 * np.arange(ticks_per))
        benign[span] = False
    for row in range(20, count):
        forecast[row * ticks_per + 3] = row / 100.0
    h_crit = mt.false_alarm_threshold(forecast, benign, interval)
    assert h_crit == pytest.approx(0.94)
    assert mt.false_alarm_rate(forecast, benign, interval, h_crit) == pytest.approx(0.05)
    assert mt.false_alarm_rate(forecast, benign, interval, h_crit - 0.005) > 0.05
    arrival = np.full(count, np.nan)
    arrival[severing] = 10.0 * severing + 0.95
    positive = np.isin(np.arange(count), severing)
    forecast[interval == 19] = 0.0
    leads = mt.lead_times(forecast, time, interval, arrival, positive, h_crit)
    np.testing.assert_allclose(leads[:19], 0.45)
    assert np.all(np.isnan(leads[19:]))
    seeds = np.arange(count) % 10
    summary = mt.lead_time_summary(leads, positive, seeds, n_boot=200)
    assert summary.n_positive == 20 and summary.n_detected == 19
    assert summary.median == pytest.approx(0.45) and summary.lo <= 0.45 <= summary.hi
    shifted = np.where(np.isfinite(leads), leads - 0.1, np.nan)
    difference = mt.paired_lead_difference(leads, shifted, positive, seeds, n_boot=200)
    assert difference == pytest.approx((0.1, 0.1, 0.1))


def test_stratified_selection_is_deterministic_and_balanced():
    rng = _generator(7)
    pools = [(0.0, 0.0, 500), (0.0, 0.02, 300), (0.02, 0.05, 300), (0.05, 0.1, 10)]
    pools += [(0.1, 0.2, 300), (0.2, 1.0, 300)]
    hazard = np.concatenate([rng.uniform(low, high, size) for low, high, size in pools])
    hazard[:500] = 0.0
    index, stratum = mt.stratified_selection(hazard, master_seed=5)
    assert index.size == 200 and np.all(hazard[index] > 0.0)
    np.testing.assert_array_equal(np.bincount(stratum, minlength=5), [48, 48, 10, 47, 47])
    again, _ = mt.stratified_selection(hazard, master_seed=5)
    np.testing.assert_array_equal(index, again)
    assert not np.array_equal(index, mt.stratified_selection(hazard, master_seed=6)[0])


def _study_states():
    means = np.array([[-0.3, -0.5, 1.8], [-0.25, -0.4, 1.6], [-0.4, 0.0, 1.5], [-0.3, -0.55, 1.6]])
    covariances = np.zeros((4, 3, 3))
    covariances[:, :2, :2] = np.diag([0.02**2, 0.1**2])
    covariances[:, 2, 2] = 0.3**2
    return means, covariances


def test_rice_study_converges_and_agrees_for_constant_acceleration():
    means, covariances = _study_states()
    study = mt.rice_study(means, covariances, 1.2, master_seed=8, n_samples=20_000)
    assert study.converged and study.convergence < 0.005
    assert np.all((study.h_mc[:, -1] > 0.005) & (study.h_mc[:, -1] < 0.5))
    np.testing.assert_allclose(study.h_exact, study.h_mc[:, -1], atol=0.003)
    assert study.consistent.all() and study.domain_count >= 1 and study.passed
    reference = study.h_mc[:, -1]
    noise = np.sqrt((1.0 - reference) / (study.n_samples * reference))
    assert np.all(np.abs(study.ratio - 1.0) < 4.0 * noise)
    repeat = mt.rice_study(means, covariances, 1.2, master_seed=8, n_samples=20_000)
    np.testing.assert_array_equal(study.h_mc, repeat.h_mc)


def test_rice_study_counts_a_failed_quadrature_as_inconsistent(monkeypatch):
    means, covariances = _study_states()
    quadrature = mt.rice_dangerous_upcrossings_joint

    def failing(mean, cov, horizon, v_b):
        if np.array_equal(mean, means[2]):
            raise ValueError("Rice quadrature did not converge; use Monte Carlo")
        return quadrature(mean, cov, horizon, v_b)

    monkeypatch.setattr(mt, "rice_dangerous_upcrossings_joint", failing)
    study = mt.rice_study(means, covariances, 1.2, master_seed=8, n_samples=5000)
    domain = (study.h_mc[:, -1] > 0.0) & (study.h_mc[:, -1] < mt.RICE_DOMAIN_LIMIT)
    assert domain[2] and np.isnan(study.h_rice[2]) and not study.consistent[2]
    assert study.rice_failures == 1 and study.domain_count == domain.sum() == 2
    assert study.domain_fraction == pytest.approx(0.5) and not study.passed


def test_labels_are_censored_where_the_horizon_leaves_the_truth():
    near_end, beyond = EXCURSIONS[:1] + ((6.9005, 0.3, 2.0),), ((6.5005, 1.5, 1.2),)
    series_time, first, first_rate = _parabolic_truth(near_end)
    _, second, second_rate = _parabolic_truth(beyond)
    elongation, rate = np.column_stack([first, second]), np.column_stack([first_rate, second_rate])
    ticks = _ticks(8.0)
    slack = np.column_stack(
        [_parabola_at(ticks, near_end)[0] < 0.0, _parabola_at(ticks, beyond)[0] < 0.0]
    )
    labels = oc.tick_outcomes(ticks, slack, series_time, elongation, rate, v_b=1.0)
    np.testing.assert_array_equal(labels.observed[:, 0], ticks <= 6.0 + 1e-9)
    late = slack[:, 0] & (ticks > 6.0)
    assert late.sum() == 3 and labels.crossed[late, 0].all() and not labels.censored[late, 0].any()
    assert labels.censored[slack[:, 1], 1].all() and not labels.label[:, 1].any()
    intervals = oc.slack_intervals(ticks, slack)
    truth = oc.interval_outcomes(intervals, series_time, elongation, rate, 1.0, 1e9)
    np.testing.assert_array_equal(truth.censored, [False, False, True])
    table = mt.monitor_table(np.where(slack, 0.5, 0.0), labels, intervals, truth)
    assert table.censored == slack[:, 1].sum() and table.dropped == 0
    assert table.forecast.size == slack[:, 0].sum() and np.all(table.interval != 2)


def test_top_bin_ratios_do_not_depend_on_the_tick_order():
    rng = _generator(11)
    count = 4000
    forecast = np.where(rng.random(count) < 0.75, 0.0, rng.integers(1, 400, count) / 2048)
    outcome = rng.random(count) < np.where(forecast == 0.0, 0.01, 3.0 * forecast)
    cluster = np.repeat(np.arange(count // 8), 8)
    report = mt.calibration_report(forecast, outcome, cluster, n_boot=50, rng=1)
    order = rng.permutation(count)
    shuffled = mt.calibration_report(forecast[order], outcome[order], cluster[order], n_boot=50)
    places, above = int(np.ceil(0.3 * count)), forecast > 0.0
    assert above.sum() < places
    share = (places - above.sum()) / np.count_nonzero(~above)
    expected = (outcome[above].sum() + share * outcome[~above].sum()) / forecast[above].sum()
    assert report.top_three_ratio == pytest.approx(expected, rel=1e-12)
    assert shuffled.top_three_ratio == pytest.approx(expected, rel=1e-12)
    assert shuffled.top_three_observed == pytest.approx(report.top_three_observed, rel=1e-12)
    assert shuffled.top_bin_ratio == pytest.approx(report.top_bin_ratio, rel=1e-12)
    assert mt._top_weights(forecast, 0.3).sum() == pytest.approx(places)


def test_reliability_band_holds_when_intervals_share_their_label():
    """Calibrated forecasts whose ticks share one outcome per slack interval: the simultaneous
    band must not fail them, as a percentile bootstrap does when a bin's clusters all agree."""
    failures, degenerate, detected = 0, 0, 0
    for replicate in range(12):
        rng = _generator(13, replicate)
        count = 300
        selected = rng.random(count) < 0.25
        common, rare = rng.beta(1.2, 1.5, count), rng.beta(0.3, 30, count)
        risk = np.where(selected, common, rare)
        sizes = rng.integers(3, 16, count)
        forecast = np.repeat(risk, sizes)
        outcome = np.repeat(rng.random(count) < risk, sizes)
        cluster = np.repeat(np.arange(count), sizes)
        report = mt.calibration_report(forecast, outcome, cluster, n_boot=40, rng=replicate)
        halved = mt.calibration_report(0.5 * forecast, outcome, cluster, n_boot=10, rng=replicate)
        failures += report.n_outside > 1
        detected += halved.n_outside > 1
        degenerate += any(
            entry["boot_lo"] == entry["boot_hi"] and not entry["boot_inside"]
            for entry in report.bins
        )
        for entry in report.bins:
            members = np.minimum((forecast * 10).astype(int), 9) == round(entry["bin_lo"] * 10)
            sizes_in_bin = np.bincount(cluster[members])
            sizes_in_bin = sizes_in_bin[sizes_in_bin > 0]
            expected = sizes_in_bin.sum() ** 2 / np.sum(sizes_in_bin**2)
            assert entry["effective_count"] == pytest.approx(expected, rel=1e-12)
    assert failures == 0 and degenerate >= 6 and detected >= 8
    rng = _generator(14)
    forecast = rng.uniform(0.0, 1.0, 400)
    outcome = rng.random(400) < forecast
    report = mt.calibration_report(forecast, outcome, np.arange(400), n_boot=20, rng=1)
    for entry in report.bins:
        assert entry["effective_count"] == pytest.approx(entry["count"], rel=1e-12)
        assert (entry["lo"], entry["hi"]) == pytest.approx(
            (entry["wilson_lo"], entry["wilson_hi"]), abs=1e-12
        )


def test_false_alarm_population_ignores_where_empty_intervals_sit():
    rng = _generator(15)
    present = np.arange(40)
    interval = np.repeat(present[present != 10], 5)
    forecast = rng.uniform(0.0, 0.3, interval.size)
    for row, level in ((3, 0.9), (7, 0.8), (21, 0.7)):
        forecast[interval == row] = level
    benign = np.ones(interval.size, dtype=bool)
    interior = mt.false_alarm_threshold(forecast, benign, interval)
    renumbered = np.where(interval > 10, interval - 1, interval)
    trailing = mt.false_alarm_threshold(forecast, benign, renumbered)
    assert interior == trailing == pytest.approx(0.8)
    assert mt.false_alarm_rate(forecast, benign, interval, interior) == pytest.approx(1 / 39)
    explicit = mt.false_alarm_threshold(forecast, benign, interval, n_intervals=40)
    assert explicit == pytest.approx(0.7)


def test_separated_outcomes_have_no_joint_recalibration():
    truth = _generator(12).random(600) < 0.2
    cluster = np.repeat(np.arange(100), 6)
    report = mt.calibration_report(np.where(truth, 1.0, 0.0), truth, cluster, n_boot=50, rng=2)
    assert report.separated and report.auroc == 1.0 and report.ece == 0.0
    fitted = (report.intercept, report.slope, report.slope_lo, report.slope_hi)
    assert np.all(np.isnan(fitted + (report.intercept_lo, report.wald_slope_hi)))
    blurred = np.where(truth, 0.7, 0.2)
    blurred[np.flatnonzero(~truth)[:3]] = 0.7
    assert mt.calibration_report(blurred, truth, cluster, n_boot=20).separated
    blurred[np.flatnonzero(truth)[0]] = 0.1
    report = mt.calibration_report(blurred, truth, cluster, n_boot=20, rng=3)
    assert not report.separated and np.isfinite([report.slope, report.intercept]).all()
    weights = np.ones(truth.size)
    weights[np.flatnonzero(truth)[0]] = 0.0
    logit = special.logit(np.clip(blurred, mt.FORECAST_CLIP, 1.0 - mt.FORECAST_CLIP))
    beta = mt._weighted_logistic(logit, truth.astype(float), weights, np.zeros(2))
    assert np.all(np.isnan(beta))


def test_prop12_leading_order_applies_only_in_the_tail():
    below = mt.overconfidence_prediction(4.0, 1.0, 0.25, mean=0.5)
    assert below.leading == pytest.approx(np.e) and below.exact > below.leading > 1.0
    above = mt.overconfidence_prediction(4.0, 1.0, 0.25, mean=1.3)
    assert np.isnan(above.leading) and 0.0 < above.exact < 1.0 and not above.agrees(1.0)
    mixed = mt.overconfidence_prediction(4.0, 1.0, 0.25, mean=[0.5, 1.3], weights=[1.0, 0.0])
    assert mixed.leading == pytest.approx(below.leading)
    assert mixed.exact == pytest.approx(below.exact)


def test_overconfident_hazard_under_forecasts_through_the_monitor():
    """Posterior Sigma / a, sigma_a / sqrt(a) into the hazard; labels from true 1 ms paths."""
    rng = _generator(11, 0)
    ratio, count, v_b = 2.0, 1200, 1.2
    sigma, spread_a = np.diag([0.02**2, 0.08**2]), 0.3
    horizon_grid = np.round(SERIES_STEP * np.arange(2201), 10)
    e0, v0 = rng.uniform(-0.35, -0.25, count), rng.uniform(-0.1, 0.1, count)
    a0 = rng.uniform(1.4, 1.75, count)
    hazard_rng = hz.hazard_generator(5, 2)
    labels, forecast, errors = np.zeros(count, dtype=bool), np.zeros(count), np.zeros((count, 2))
    reported, reported_a = sigma / ratio, spread_a / np.sqrt(ratio)
    for row in range(count):
        forecast[row] = hz.tick_hazard(
            e0[row], v0[row], reported, a0[row], reported_a, v_b, hazard_rng, 1024
        )
        truth = rng.multivariate_normal([e0[row], v0[row]], sigma)
        acceleration = a0[row] + spread_a * rng.standard_normal()
        errors[row] = truth - (e0[row], v0[row])
        path = truth[0] + horizon_grid * (truth[1] + 0.5 * acceleration * horizon_grid)
        speed = truth[1] + acceleration * horizon_grid
        labels[row] = oc.tick_outcomes([0.0], [True], horizon_grid, path, speed, v_b).label[0, 0]
    nees = mt.tick_nees(errors[:, 0], errors[:, 1], reported, 0.0, 0.0)
    assert nees.mean() == pytest.approx(2.0 * ratio, rel=0.1)
    report = mt.calibration_report(forecast, labels, np.arange(count), n_boot=300, rng=0)
    assert report.large_intercept_lo > 0.0
    assert report.top_bin_ratio_lo > 1.0 and report.top_three_ratio_lo > 1.0
    mean_speed, speed_sd = mt.closing_speed_moments(
        e0, v0, np.broadcast_to(reported, (count, 2, 2)), a0, reported_a
    )
    top = mt._top_weights(forecast, mt.TOP_FRACTIONS[0])
    prediction = mt.overconfidence_prediction(
        nees.mean(), v_b, speed_sd, mean=mean_speed, weights=top * forecast
    )
    assert prediction.leading > 1.5
    assert 0.5 < report.top_bin_ratio / prediction.exact < 2.0


@SLOW
def test_rice_study_at_campaign_size():
    rng = _generator(9)
    count = 1200
    means = np.column_stack(
        [
            rng.uniform(-0.8, -0.05, count),
            rng.uniform(-1.2, 0.6, count),
            rng.uniform(0.8, 2.6, count),
        ]
    )
    covariances = np.zeros((count, 3, 3))
    covariances[:, :2, :2] = np.diag([0.02**2, 0.1**2])
    covariances[:, 2, 2] = 0.2**2
    generator = hz.hazard_generator(9, 0)
    risk = np.array(
        [
            hz.tick_hazard(e, v, c[:2, :2], a, 0.2, 1.2, generator)
            for (e, v, a), c in zip(means, covariances)
        ]
    )
    index, _ = mt.stratified_selection(risk, master_seed=9)
    study = mt.rice_study(means[index], covariances[index], 1.2, 9, state_ids=index)
    assert study.converged and study.passed


@SLOW
def test_honest_posterior_hazard_recalibrates_to_identity():
    rng = _generator(10)
    labels, forecasts = _prop10_ticks(rng, 3000, hz.DEFAULT_SAMPLES)
    report = mt.calibration_report(forecasts, labels, np.arange(labels.size), n_boot=500)
    assert report.slope_lo <= 1.0 <= report.slope_hi
    assert report.large_intercept_lo <= 0.0 <= report.large_intercept_hi
    assert report.ece < 0.05
