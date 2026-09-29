"""Part II excursion, impact, threshold-law, and hazard predictors (Drake-free)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy import linalg, special
from scipy.integrate import solve_ivp

from tether.physics import constants
from tether.theory import excursion as ex
from tether.theory.impact import TabulatedImpact

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
APPENDIX_A1 = {0.0: 1.000, 0.05: 0.931, 0.10: 0.880, 0.15: 0.844, 0.20: 0.821, 0.30: 0.813}
PLAN_CRITICAL_DEPTH = {6e3: 0.14, 9e3: 0.31, 12e3: 0.55, 16e3: 0.97, 25e3: 2.37}
PLAN_SNAP_KN = {0.1: 5.1, 0.25: 8.1, 0.5: 11.5, 1.0: 16.2, 2.0: 22.9}
K = constants.CABLE_STIFFNESS
MEFF = ex.EFFECTIVE_MASS
T0 = ex.NOMINAL_PRETENSION
A0 = ex.RESTORING_ACCELERATION

OSCILLATOR_OMEGA = 2.0 * np.pi
OSCILLATOR_ZETA = 0.3
OSCILLATOR_SIGMA_E = 0.2
OSCILLATOR_SIGMA_EDOT = OSCILLATOR_OMEGA * OSCILLATOR_SIGMA_E
OSCILLATOR_DT = 0.005


def _peak_tension_by_ode(zeta: float) -> float:
    """Max of e + 2 zeta e' for e'' + 2 zeta e' + e = 0, e(0) = 0, e'(0) = 1."""
    span = 4.0 * np.pi
    solution = solve_ivp(
        lambda t, y: [y[1], -2.0 * zeta * y[1] - y[0]],
        (0.0, span),
        [0.0, 1.0],
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        dense_output=True,
    )
    elongation, rate = solution.sol(np.linspace(0.0, span, 400_001))
    return float(np.max(elongation + 2.0 * zeta * rate))


@dataclass(frozen=True)
class _Integrated:
    shutoff_elongation: float
    turning_time: float
    max_depth: float
    return_time: float
    return_speed: float
    total_work: float
    return_work: float


def _integrate_excursion(u: float, pieces: list[tuple[float, float]]) -> _Integrated:
    """Integrate e'' = a_j on consecutive (duration_j, a_j) pieces from e = 0, e' = -u."""

    def returned(t, y):
        return y[0]

    def turned(t, y):
        return y[1]

    returned.terminal, returned.direction = True, 1.0
    turned.direction = 1.0
    state, clock = np.array([0.0, -u]), 0.0
    legs, vertex, crossing, shutoff = [], None, None, None
    for duration, acceleration in pieces:
        solution = solve_ivp(
            lambda t, y, a=acceleration: [y[1], a],
            (clock, clock + duration),
            state,
            method="DOP853",
            rtol=1e-12,
            atol=1e-14,
            events=(returned, turned),
            dense_output=True,
        )
        legs.append((clock, solution.t[-1], acceleration, solution.sol))
        if shutoff is None:
            shutoff = float(solution.y[0, -1])
        if vertex is None and solution.t_events[1].size:
            vertex = (float(solution.t_events[1][0]), float(-solution.y_events[1][0][0]))
        if solution.status == 1:
            crossing = (float(solution.t_events[0][0]), float(solution.y_events[0][0][1]))
            break
        clock, state = solution.t[-1], solution.y[:, -1]
    assert vertex is not None and crossing is not None
    total_work = sum(a * (sol(stop)[0] - sol(start)[0]) for start, stop, a, sol in legs)
    return_work = sum(
        a * (sol(stop)[0] - sol(max(start, vertex[0]))[0])
        for start, stop, a, sol in legs
        if stop > vertex[0]
    )
    return _Integrated(
        shutoff_elongation=shutoff,
        turning_time=vertex[0],
        max_depth=vertex[1],
        return_time=crossing[0],
        return_speed=crossing[1],
        total_work=float(total_work),
        return_work=float(return_work),
    )


@pytest.fixture(scope="module")
def oscillator_paths():
    """Exactly stationary Gaussian (e, e') of a damped oscillator driven by white noise."""
    rng = np.random.default_rng(np.random.SeedSequence([2026, 91]))
    drift = np.array(
        [[0.0, 1.0], [-(OSCILLATOR_OMEGA**2), -2.0 * OSCILLATOR_ZETA * OSCILLATOR_OMEGA]]
    )
    stationary = np.diag([OSCILLATOR_SIGMA_E**2, OSCILLATOR_SIGMA_EDOT**2])
    transition = linalg.expm(drift * OSCILLATOR_DT)
    noise = np.linalg.cholesky(stationary - transition @ stationary @ transition.T)
    paths, steps = 256, 10_000
    state = rng.standard_normal((paths, 2)) @ np.linalg.cholesky(stationary).T
    elongation = np.empty((paths, steps + 1))
    rate = np.empty((paths, steps + 1))
    elongation[:, 0], rate[:, 0] = state.T
    for step in range(steps):
        state = state @ transition.T + rng.standard_normal((paths, 2)) @ noise.T
        elongation[:, step + 1], rate[:, step + 1] = state.T
    return elongation, rate


def test_appendix_a1_impact_factor_table():
    for zeta, expected in APPENDIX_A1.items():
        assert round(ex.impact_factor(zeta), 3) == expected
    np.testing.assert_allclose(
        ex.impact_factor(np.array(list(APPENDIX_A1))),
        [ex.impact_factor(zeta) for zeta in APPENDIX_A1],
    )


@pytest.mark.parametrize("zeta", [0.0, 0.05, 0.0992, 0.2, 0.3, 0.45, 0.6, 1.0, 1.5])
def test_impact_factor_equals_direct_ode_maximum(zeta):
    assert ex.impact_factor(zeta) == pytest.approx(_peak_tension_by_ode(zeta), abs=1e-7)


def test_pinned_plant_values():
    assert MEFF == pytest.approx(600.0 * 2500.0 / 3100.0)
    assert round(MEFF, 2) == 483.87
    assert T0 == pytest.approx(990.0, abs=1e-9)
    assert round(ex.DAMPING_RATIO, 4) == 0.0992
    assert round(float(np.sqrt(K / MEFF)), 2) == 18.74
    assert round(ex.IMPACT_FACTOR, 3) == 0.881
    assert abs(ex.IMPACT_FACTOR - ex.PINNED_IMPACT_FACTOR) < 1e-3
    assert ex.IMPEDANCE == pytest.approx(ex.IMPACT_FACTOR * np.sqrt(K * MEFF))
    assert round(ex.IMPEDANCE / 1e3, 2) == 7.99
    assert round(A0, 2) == 2.05
    assert ex.critical_closing_speed(12e3) == pytest.approx(12e3 / ex.IMPEDANCE)


def test_appendix_a4_tables_reproduce_at_one_kilonewton():
    for level, depth in PLAN_CRITICAL_DEPTH.items():
        assert round(ex.critical_depth(level, 1000.0, f=ex.PINNED_IMPACT_FACTOR), 2) == depth
    for depth, snap in PLAN_SNAP_KN.items():
        tension = ex.snap_tension_from_depth(depth, 1000.0, f=ex.PINNED_IMPACT_FACTOR)
        assert round(tension / 1e3, 1) == snap


def test_appendix_a4_at_pinned_pretension():
    f = ex.PINNED_IMPACT_FACTOR
    depths = ex.critical_depth(np.array(list(PLAN_CRITICAL_DEPTH)), T0, f=f)
    snaps = ex.snap_tension_from_depth(np.array(list(PLAN_SNAP_KN)), T0, f=f) / 1e3
    np.testing.assert_allclose(depths, [0.1381, 0.3107, 0.5524, 0.9821, 2.3977], atol=5e-5)
    np.testing.assert_allclose(snaps, [5.1055, 8.0725, 11.4163, 16.1451, 22.8326], atol=5e-4)
    assert [round(value, 2) for value in depths] == [0.14, 0.31, 0.55, 0.98, 2.40]
    assert [round(value, 1) for value in snaps] == [5.1, 8.1, 11.4, 16.1, 22.8]
    at_one_kilonewton = ex.critical_depth(np.array(list(PLAN_CRITICAL_DEPTH)), 1000.0, f=f)
    np.testing.assert_allclose(depths * T0, at_one_kilonewton * 1000.0)
    assert round(ex.critical_depth(12e3, T0, f=f), 2) == 0.55
    assert round(ex.saturation_snap(T0, 2.0, f=f) / 1e3) == 23
    assert ex.saturation_snap(T0, 2.0) == pytest.approx(ex.snap_tension_from_depth(2.0, T0))


def test_severance_inequality_inverts_corollary_5():
    t_g = 2.0
    coefficient = ex.kappa(T0, t_g)
    levels = np.array([6e3, 12e3, 25e3])
    np.testing.assert_allclose(
        coefficient * levels**2, 2.0 * MEFF * ex.critical_depth(levels, T0) / t_g**2, rtol=1e-12
    )
    np.testing.assert_allclose(
        ex.closed_form_snap(T0 + coefficient * levels**2, T0, t_g), levels, rtol=1e-12
    )


@pytest.mark.parametrize("speed, acceleration", [(0.3, A0), (0.05, 0.4), (1.2, 5.0)])
def test_prop2_ballistic_symmetry_by_direct_integration(speed, acceleration):
    integrated = _integrate_excursion(speed, [(60.0, acceleration)])
    predicted = ex.ballistic_return(speed, acceleration)
    assert integrated.return_time == pytest.approx(predicted["return_time"], rel=1e-9)
    assert integrated.max_depth == pytest.approx(predicted["depth"], rel=1e-9)
    assert integrated.return_speed == pytest.approx(predicted["return_speed"], rel=1e-9)
    vertex_form = ex.return_speed(0.0, acceleration, integrated.max_depth)
    onset_form = ex.return_speed(speed, acceleration, integrated.max_depth)
    assert vertex_form == pytest.approx(integrated.return_speed, rel=1e-9)
    assert onset_form == pytest.approx(np.sqrt(2.0) * integrated.return_speed, rel=1e-9)


@pytest.mark.parametrize(
    "delta_w, t_g, speed",
    [
        (500.0, 0.5, 0.1),
        (100.0, 2.0, 0.0),
        (990.0, 1.0, 0.3),
        (0.0, 1.0, 0.2),
        (-300.0, 1.0, 0.4),
        (-300.0, 3.0, 0.4),
    ],
)
def test_prop3_energy_conversion_by_piecewise_integration(delta_w, t_g, speed):
    integrated = _integrate_excursion(speed, [(t_g, -delta_w / MEFF), (60.0, A0)])
    outcome = ex.two_phase_excursion(delta_w, t_g, speed)
    assert outcome.max_depth == pytest.approx(integrated.max_depth, rel=1e-9, abs=1e-12)
    assert outcome.return_speed == pytest.approx(integrated.return_speed, rel=1e-9)
    assert outcome.return_time == pytest.approx(integrated.return_time, rel=1e-9)
    assert outcome.turning_time == pytest.approx(integrated.turning_time, rel=1e-9)
    if not outcome.returned_during_gust:
        shutoff = max(-integrated.shutoff_elongation, 0.0)
        assert outcome.depth_at_shutoff == pytest.approx(shutoff, abs=1e-12)
    return_mean = integrated.return_work / integrated.max_depth
    assert ex.return_speed(0.0, return_mean, integrated.max_depth) == pytest.approx(
        integrated.return_speed, rel=1e-9
    )
    energy = speed**2 + 2.0 * integrated.total_work
    assert integrated.return_speed**2 == pytest.approx(energy, rel=1e-9)
    if outcome.turning_time >= t_g:
        assert return_mean == pytest.approx(A0, rel=1e-9)
        assert ex.snap_tension_from_depth(integrated.max_depth, T0) == pytest.approx(
            ex.IMPEDANCE * integrated.return_speed, rel=1e-9
        )
    if delta_w > 0.0:
        assert ex.deep_excursion_depth(delta_w, t_g, speed) == pytest.approx(
            -integrated.shutoff_elongation, rel=1e-9
        )


def test_prop4_threshold_at_differential_load_equal_to_pretension():
    t_g = 2.0
    ratios = np.array([0.6, 0.9, 0.99, 1.0, 1.0 + 1e-9, 1.01, 1.1, 1.5, 2.0])
    depths = np.array([ex.two_phase_excursion((r - 1.0) * T0, t_g).max_depth for r in ratios])
    snaps = ex.closed_form_snap(ratios * T0, T0, t_g)
    assert np.all(depths[ratios <= 1.0] == 0.0) and np.all(depths[ratios > 1.0] > 0.0)
    assert np.all(snaps[ratios <= 1.0] == 0.0) and np.all(snaps[ratios > 1.0] > 0.0)
    sub = ex.two_phase_excursion(-0.1 * T0, 4.0, 0.2)
    assert sub.max_depth == pytest.approx(ex.two_phase_excursion(-0.1 * T0, 8.0, 0.2).max_depth)
    assert sub.max_depth == pytest.approx(0.2**2 * MEFF / (2.0 * 0.1 * T0))
    assert sub.max_depth > ex.deep_excursion_depth(-0.1 * T0, 4.0, 0.2)
    assert ex.deep_excursion_depth(-0.1 * T0, 4.0, 0.2) == pytest.approx(0.2**2 / (2.0 * A0))
    deep = [ex.two_phase_excursion(0.1 * T0, duration).max_depth for duration in (2.0, 4.0)]
    assert deep[1] / deep[0] == pytest.approx(4.0)
    for ratio, plan_depth in [(1.5, 3.10), (2.0, 8.27)]:
        outcome = ex.two_phase_excursion((ratio - 1.0) * 1000.0, 2.0, T0=1000.0)
        shutoff = ex.deep_excursion_depth((ratio - 1.0) * 1000.0, 2.0, 0.0, T0=1000.0)
        assert round(outcome.max_depth, 2) == plan_depth
        assert outcome.max_depth / shutoff == pytest.approx(ratio)
    assert ex.two_phase_excursion(100.0, 2.0, T0=1000.0).max_depth == pytest.approx(0.46, abs=0.01)


def test_corollary5_optimum_and_post_gust_closing():
    w_rel, t_g = 2000.0, 1.5
    pretensions = np.linspace(1.0, w_rel, 20_001)
    severity = ex.closed_form_snap(w_rel, pretensions, t_g)
    assert pretensions[np.argmax(severity)] == pytest.approx(ex.pretension_optimum(w_rel), rel=1e-3)
    assert severity[-1] == 0.0
    for pretension in (500.0, T0, 1500.0):
        shutoff = ex.deep_excursion_depth(w_rel - pretension, t_g, 0.0, T0=pretension)
        assert ex.closed_form_snap(w_rel, pretension, t_g) == pytest.approx(
            ex.snap_tension_from_depth(shutoff, pretension)
        )
        exact = ex.two_phase_excursion(w_rel - pretension, t_g, T0=pretension)
        assert ex.two_phase_snap(w_rel, pretension, t_g) == pytest.approx(
            ex.IMPEDANCE * exact.return_speed
        )
        assert ex.two_phase_snap(w_rel, pretension, t_g) / ex.closed_form_snap(
            w_rel, pretension, t_g
        ) == pytest.approx(np.sqrt(w_rel / pretension))
    assert np.all(np.diff(ex.two_phase_snap(w_rel, pretensions, t_g)) <= 0.0)


def test_two_phase_severance_load_and_its_tail_index():
    t_g, alpha = 2.0, 3.0
    coefficient = ex.kappa(T0, t_g)
    levels = np.array([6e3, 12e3, 25e3, 1e5, 1e6])
    loads = ex.two_phase_severance_load(levels, T0, t_g)
    np.testing.assert_allclose(ex.two_phase_snap(loads, T0, t_g), levels, rtol=1e-12)
    assert np.all(loads < T0 + coefficient * levels**2)
    weak = ex.two_phase_severance_load(500.0, T0, t_g) - T0
    first_order = coefficient * 500.0**2
    assert weak == pytest.approx(first_order - first_order**2 / T0, rel=1e-6)
    step = 1e-5
    upper = ex.two_phase_severance_load(levels * np.exp(step), T0, t_g)
    lower = ex.two_phase_severance_load(levels * np.exp(-step), T0, t_g)
    index = alpha * (np.log(upper) - np.log(lower)) / (2.0 * step)
    closed = alpha * (1.0 - 1.0 / np.sqrt(1.0 + 4.0 * coefficient * levels**2 / T0))
    np.testing.assert_allclose(index, closed, rtol=1e-6)
    assert np.all(index < alpha)
    assert np.all(index < ex.rv_snap_local_index(levels, T0, coefficient, alpha))
    assert ex.rv_snap_local_index(1e6, T0, coefficient, alpha) > 1.99 * alpha


def test_theorem6_asymptotic_log_slopes():
    coefficient, sigma_w, levels = 1e-6, 400.0, np.array([1.0e6, 1.001e6])
    snap = ex.gaussian_snap_log_rate(levels, 0.1, T0, coefficient, 0.0, sigma_w)
    assert np.diff(snap)[0] / np.diff(levels**4)[0] == pytest.approx(
        ex.gaussian_snap_log_slope(coefficient, sigma_w), rel=5e-3
    )
    taut = ex.gaussian_taut_log_rate(levels, T0, 2000.0, 2.0e4)
    assert np.diff(taut)[0] / np.diff(levels**2)[0] == pytest.approx(
        ex.gaussian_taut_log_slope(2000.0), rel=5e-3
    )
    moderate = np.array([5e3, 2e4, 6e4])
    np.testing.assert_allclose(
        ex.gaussian_snap_log_rate(moderate, 0.1, T0, coefficient, 0.0, sigma_w),
        np.log(ex.gaussian_snap_rate(moderate, 0.1, T0, coefficient, 0.0, sigma_w)),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        ex.gaussian_taut_log_rate(moderate, T0, 2000.0, 2.0e4),
        np.log(ex.gaussian_taut_rate(moderate, T0, 2000.0, 2.0e4)),
        rtol=1e-10,
    )


def test_theorem7_univariate_indices():
    levels, alpha, coefficient = np.array([1e4, 2e4]), 3.0, 1e-6
    snap = np.log(ex.rv_snap_rate(levels, 0.2, 5.0, coefficient, alpha))
    taut = np.log(ex.rv_taut_rate(levels, 1e9, alpha))
    assert np.diff(snap)[0] / np.diff(np.log(levels))[0] == pytest.approx(-2.0 * alpha)
    assert np.diff(taut)[0] / np.diff(np.log(levels))[0] == pytest.approx(-alpha)
    centre, step = 1.2e4, 1e-4
    bracket = centre * np.exp([-step, step])
    local = np.log(ex.rv_snap_rate(bracket, 0.2, 5.0, coefficient, alpha, T0=T0))
    assert -np.diff(local)[0] / (2.0 * step) == pytest.approx(
        ex.rv_snap_local_index(centre, T0, coefficient, alpha), rel=1e-6
    )


def test_theorem7_fleet_quadrature():
    peak = np.array([[1.0, 0.5], [-1.0, 2.0], [0.0, -3.0], [0.3, 0.3]])
    load = np.array([[0.2, -0.1], [0.5, 0.0], [-1.0, -1.0], [0.9, 0.05]])
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    pretension, coefficient = np.array([990.0, 1200.0]), np.array([1e-6, 2e-6])
    alpha, probability, level = 3.0, 0.01, 1.0e4
    expected_fleet, expected_cables = 0.0, np.zeros(2)
    for s in range(4):
        radii = []
        for i in range(2):
            candidates = [np.inf]
            if peak[s, i] > 0.0:
                candidates.append(level / peak[s, i])
            if load[s, i] > 0.0:
                candidates.append((pretension[i] + coefficient[i] * level**2) / load[s, i])
            radii.append(min(candidates))
            expected_cables[i] += probability * weights[s] * radii[i] ** (-alpha)
        expected_fleet += probability * weights[s] * min(radii) ** (-alpha)
    args = (probability, weights, peak, load, pretension, coefficient, alpha)
    fleet = ex.fleet_rv_rate(level, *args)
    cables = ex.cable_rv_rates(level, *args)
    assert fleet == pytest.approx(expected_fleet, rel=1e-12)
    np.testing.assert_allclose(cables, expected_cables, rtol=1e-12)
    assert cables.max() <= fleet < cables.sum()
    np.testing.assert_allclose(ex.fleet_rv_rate(np.array([level, 2 * level]), *args)[0], fleet)
    blind = ex.fleet_rv_rate(level, 0.01, [1.0], [[-1.0, 0.0]], [[0.0, -2.0]], 990.0, 1e-6, alpha)
    assert blind == 0.0
    doubled = np.array([1e4, 2e4])
    taut_only = ex.fleet_rv_rate(doubled, 1.0, [1.0], [[2.0]], [[-1.0]], 990.0, 1e-6, alpha)
    snap_only = ex.fleet_rv_rate(doubled, 1.0, [1.0], [[-2.0]], [[1.0]], 0.0, 1e-6, alpha)
    assert np.log(taut_only[1] / taut_only[0]) / np.log(2.0) == pytest.approx(-alpha)
    assert np.log(snap_only[1] / snap_only[0]) / np.log(2.0) == pytest.approx(-2.0 * alpha)


def test_corollary71_crossover():
    alpha, coefficient, taut_scale = 3.0, 1e-6, 1.25e5
    expected = (1.0 / taut_scale) ** (1.0 / alpha) / coefficient
    snap = lambda level: ex.rv_snap_rate(level, 1.0, 1.0, coefficient, alpha)  # noqa: E731
    taut = lambda level: ex.rv_taut_rate(level, taut_scale, alpha)  # noqa: E731
    assert ex.crossover_threshold(snap, taut, (1e3, 1e5)) == pytest.approx(expected, rel=1e-8)
    assert ex.crossover_threshold(snap, taut, (1e3, 0.5 * expected)) is None

    gaussian_snap = lambda x: ex.gaussian_snap_log_rate(x, 0.1, T0, 1e-6, 0.0, 400.0)  # noqa: E731
    gaussian_taut = lambda x: ex.gaussian_taut_log_rate(x, T0, 800.0, 8000.0)  # noqa: E731
    grid = np.linspace(6e3, 1e6, 200_001)
    gap = gaussian_snap(grid) - gaussian_taut(grid)
    brute = grid[np.flatnonzero(np.diff(np.sign(gap)))[0]]
    found = ex.crossover_threshold(gaussian_snap, gaussian_taut, (6e3, 1e6), log_rates=True)
    assert found == pytest.approx(brute, abs=grid[1] - grid[0])


def test_prop1_onset_rate_against_simulated_gaussian_process(oscillator_paths):
    elongation, rate = oscillator_paths
    mean = 0.5 * OSCILLATOR_SIGMA_E
    level = mean + elongation
    down = (level[:, :-1] > 0.0) & (level[:, 1:] <= 0.0)
    exposure = down.shape[0] * down.shape[1] * OSCILLATOR_DT
    predicted = ex.onset_rate(mean, OSCILLATOR_SIGMA_E, OSCILLATOR_SIGMA_EDOT)
    assert down.sum() / exposure == pytest.approx(predicted, rel=0.07)
    rows, columns = np.nonzero(down)
    fraction = level[rows, columns] / (level[rows, columns] - level[rows, columns + 1])
    speeds = -(rate[rows, columns] + fraction * (rate[rows, columns + 1] - rate[rows, columns]))
    assert np.mean(speeds) == pytest.approx(OSCILLATOR_SIGMA_EDOT * np.sqrt(np.pi / 2.0), rel=0.05)
    assert np.mean(speeds**2) == pytest.approx(2.0 * OSCILLATOR_SIGMA_EDOT**2, rel=0.08)
    median = OSCILLATOR_SIGMA_EDOT * np.sqrt(2.0 * np.log(2.0))
    assert ex.rayleigh_cdf(median, OSCILLATOR_SIGMA_EDOT) == pytest.approx(0.5)
    assert np.mean(speeds <= median) == pytest.approx(0.5, abs=0.03)


def test_rayleigh_law_and_ballistic_snap_rate():
    grid = np.linspace(0.0, 10.0, 100_001)
    density = ex.rayleigh_pdf(grid, 1.3)
    assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=1e-6)
    inside = grid <= 4.0
    assert ex.rayleigh_cdf(4.0, 1.3) == pytest.approx(
        np.trapezoid(density[inside], grid[inside]), abs=1e-6
    )
    assert ex.rayleigh_pdf(-1.0, 1.3) == 0.0 and ex.rayleigh_cdf(-1.0, 1.3) == 0.0
    onset = ex.onset_rate(0.01, 0.004, 0.05)
    assert ex.ballistic_snap_rate(4e3, 0.01, 0.004, 0.05) == pytest.approx(
        onset * (1.0 - ex.rayleigh_cdf(4e3 / ex.IMPEDANCE, 0.05))
    )


def test_rice_bound_dominates_first_upcrossing_hazard(oscillator_paths):
    elongation, rate = oscillator_paths
    mean, speed_level = -OSCILLATOR_SIGMA_E, OSCILLATOR_SIGMA_EDOT
    level = mean + elongation
    results = {}
    for steps in (20, 800):
        windows = np.lib.stride_tricks.sliding_window_view(level, steps + 1, axis=1)[:, ::steps]
        rates = np.lib.stride_tricks.sliding_window_view(rate, steps + 1, axis=1)[:, ::steps]
        times = OSCILLATOR_DT * np.arange(steps + 1)
        crossings = ex.first_upcrossings_from_paths(
            times, windows.reshape(-1, steps + 1), rates.reshape(-1, steps + 1)
        )
        hazard = float(np.mean(crossings.dangerous(speed_level)))
        bound = ex.rice_hazard_bound(
            times, mean, 0.0, OSCILLATOR_SIGMA_E**2, OSCILLATOR_SIGMA_EDOT**2, 0.0, speed_level
        )
        intensity = np.exp(-1.0) * OSCILLATOR_SIGMA_EDOT / (2.0 * np.pi * OSCILLATOR_SIGMA_E)
        expected = times[-1] * intensity
        assert bound == pytest.approx(expected, rel=1e-9)
        results[steps] = (hazard, bound)
    short_hazard, short_bound = results[20]
    assert 0.95 <= short_bound / short_hazard <= 1.08
    long_hazard, long_bound = results[800]
    assert long_bound > 1.0 and long_hazard < 0.5 * long_bound


HAZARD_STATES = [
    (
        np.array([-0.3, -0.2, 2.0]),
        np.array([[0.05**2, -0.0015, 0.0], [-0.0015, 0.1**2, 0.0], [0.0, 0.0, 0.5**2]]),
    ),
    (
        np.array([-0.8, -0.5, 1.5]),
        np.array([[0.1**2, 0.0, 0.0], [0.0, 0.2**2, 0.016], [0.0, 0.016, 0.4**2]]),
    ),
]


@pytest.mark.parametrize("state", range(len(HAZARD_STATES)))
def test_rice_integral_matches_monte_carlo_for_constant_acceleration(state):
    mean, cov = HAZARD_STATES[state]
    horizon, count = 2.0, 100_000
    samples = ex.sample_constant_acceleration(mean, cov, count, np.random.default_rng([7, state]))
    exact = ex.first_upcrossings_exact(samples, horizon)
    grid = ex.first_upcrossings_on_grid(samples, horizon, 0.005)
    for speed_level in (0.8, 1.2, 1.5, 1.8):
        hazard = float(np.mean(exact.dangerous(speed_level)))
        bound = ex.rice_dangerous_upcrossings_joint(mean, cov, horizon, speed_level)
        error = np.sqrt(max(hazard * (1.0 - hazard), 1.0 / count) / count)
        assert abs(bound - hazard) <= 4.0 * error + 2e-4
        assert bound >= float(np.mean(grid.dangerous(speed_level))) - 4.0 * error
        if 0.03 < hazard < 0.2:
            assert 0.9 <= bound / hazard <= 1.1


TIGHT_HAZARD_STATES = [
    (np.array([-0.4, -0.3, 2.0]), np.diag([2.5e-4**2, 7.5e-5**2, 2e-3**2]), 1.3),
    (np.array([-0.4, -0.3, 2.0]), np.diag([2.5e-4**2, 7.5e-5**2, 2e-3**2]), 1.3002),
    (np.array([-0.05, 0.4, 1.0]), np.diag([1e-5**2, 1e-4**2, 1.5e-2**2]), 0.5),
]


@pytest.mark.parametrize("state", range(len(TIGHT_HAZARD_STATES)))
def test_rice_integral_resolves_tight_posteriors(state):
    mean, cov, speed_level = TIGHT_HAZARD_STATES[state]
    horizon, count = 2.0, 400_000
    samples = ex.sample_constant_acceleration(mean, cov, count, np.random.default_rng([29, state]))
    hazard = float(np.mean(ex.first_upcrossings_exact(samples, horizon).dangerous(speed_level)))
    error = np.sqrt(max(hazard * (1.0 - hazard), 1.0 / count) / count)
    rice = ex.rice_dangerous_upcrossings_joint(mean, cov, horizon, speed_level)
    assert abs(rice - hazard) <= 4.0 * error
    coarse = ex.rice_dangerous_upcrossings_joint(mean, cov, horizon, speed_level, n_grid=21)
    assert coarse == pytest.approx(rice, abs=1e-8)


def test_rice_integral_deterministic_and_onset_states():
    path = np.array([-1.0, -0.2, 2.0])
    closing = np.sqrt(0.2**2 + 2.0 * 2.0 * 1.0)
    for speed_level in (0.5 * closing, 1.5 * closing):
        rice = ex.rice_dangerous_upcrossings_joint(path, np.zeros((3, 3)), 2.0, speed_level)
        simulated = ex.constant_acceleration_hazard_mc(
            path, np.zeros((3, 3)), 2.0, speed_level, 10, np.random.default_rng(0), 0.005
        )
        assert rice == simulated == float(speed_level < closing)
    returns_in_time = special.ndtr((2.0 - 0.3) / 0.3)
    for speed_level in (0.2, 0.4):
        onset = ex.rice_dangerous_upcrossings(
            0.0, -0.3, np.zeros((2, 2)), 2.0, 0.3, 2.0, speed_level
        )
        assert onset == pytest.approx(returns_in_time * float(speed_level < 0.3), abs=1e-9)
    for speed_level in (0.25, 0.33, 0.4):
        onset = ex.rice_dangerous_upcrossings(
            0.0, -0.3, np.diag([0.0, 0.05**2]), 2.0, 0.3, 2.0, speed_level
        )
        assert onset == pytest.approx(special.ndtr((0.3 - speed_level) / 0.05), abs=1e-6)
    with pytest.raises(ValueError):
        ex.constant_acceleration_hazard_mc(
            path, np.eye(3), 2.0, 1.0, 0, np.random.default_rng(0), 0.01
        )


def test_rice_independent_acceleration_wrapper():
    block = np.array([[0.05**2, -0.0015], [-0.0015, 0.1**2]])
    joint = np.zeros((3, 3))
    joint[:2, :2], joint[2, 2] = block, 0.5**2
    assert ex.rice_dangerous_upcrossings(-0.3, -0.2, block, 2.0, 0.5, 2.0, 1.2) == pytest.approx(
        ex.rice_dangerous_upcrossings_joint([-0.3, -0.2, 2.0], joint, 2.0, 1.2), rel=1e-14
    )


def test_monte_carlo_hazard_converges_with_grid_step():
    mean, cov = HAZARD_STATES[0]
    horizon, count, speed_level, seed = 2.0, 50_000, 1.2, [17, 3]
    samples = ex.sample_constant_acceleration(mean, cov, count, np.random.default_rng(seed))
    exact = ex.first_upcrossings_exact(samples, horizon)
    hazards, time_errors = {}, {}
    for step in (0.02, 0.01, 0.005):
        grid = ex.first_upcrossings_on_grid(samples, horizon, step)
        hazards[step] = float(np.mean(grid.dangerous(speed_level)))
        both = grid.crossed & exact.crossed
        time_errors[step] = float(np.median(np.abs(grid.time[both] - exact.time[both])))
        assert abs(hazards[step] - float(np.mean(exact.dangerous(speed_level)))) < 1e-3
    assert abs(hazards[0.02] - hazards[0.005]) < 0.005
    assert time_errors[0.02] / time_errors[0.01] > 3.0
    assert time_errors[0.01] / time_errors[0.005] > 3.0
    direct = ex.constant_acceleration_hazard_mc(
        mean, cov, horizon, speed_level, count, np.random.default_rng(seed), 0.01
    )
    assert direct == hazards[0.01]
    assert direct == ex.constant_acceleration_hazard_mc(
        mean, cov, horizon, speed_level, count, np.random.default_rng(seed), 0.01
    )


def test_exact_first_upcrossing_cases():
    samples = np.array(
        [
            [-1.0, 0.0, 2.0],
            [0.5, -2.0, 2.0],
            [-1.0, 2.0, -2.0],
            [-1.0, 3.0, -2.0],
            [-1.0, 0.5, 0.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, 1.0],
            [-10.0, 0.0, 1.0],
        ]
    )
    result = ex.first_upcrossings_exact(samples, 3.0)
    expected = [True, True, False, True, True, False, True, False]
    np.testing.assert_array_equal(result.crossed, expected)
    crossed = result.crossed
    np.testing.assert_allclose(
        result.time[crossed], [1.0, 1.0 + np.sqrt(0.5), (3.0 - np.sqrt(5.0)) / 2.0, 2.0, 2.0]
    )
    np.testing.assert_allclose(result.speed[crossed], [2.0, np.sqrt(2.0), np.sqrt(5.0), 0.5, 1.0])
    grid = ex.first_upcrossings_on_grid(samples, 3.0, 7e-4)
    np.testing.assert_array_equal(grid.crossed, result.crossed)
    np.testing.assert_allclose(grid.speed[crossed], result.speed[crossed], rtol=1e-5)


def test_prop12_overconfidence_factor():
    assert ex.overconfidence_factor(1.0, 2.0, 0.5) == 1.0
    assert ex.overconfidence_factor(10.0, 2.0, 1.0) == pytest.approx(np.exp(18.0))
    for score in (6.0, 10.0):
        exact = ex.overconfidence_factor_exact(10.0, score, 1.0)
        ratio = special.ndtr(-score) / special.ndtr(-np.sqrt(10.0) * score)
        assert exact == pytest.approx(ratio, rel=1e-9)
        leading = ex.overconfidence_factor(10.0, score, 1.0)
        assert exact / (np.sqrt(10.0) * leading) == pytest.approx(1.0, abs=1.5 / score**2)


def _synthetic_impact_table(**kwargs) -> TabulatedImpact:
    speeds = np.array([3.0, 0.25, 0.5, 1.0, 2.0, 0.5, 3.0])
    scatter = np.array([40.0, 0.0, -50.0, 0.0, 0.0, 50.0, -40.0])
    peaks = 1600.0 + 7000.0 * speeds + 300.0 * speeds**2 + scatter
    return TabulatedImpact.from_measurements(speeds, peaks, **kwargs)


def test_tabulated_impact_interpolation_inverse_and_round_trip():
    table = _synthetic_impact_table(zero_speed_peak=1600.0)
    law = lambda v: 1600.0 + 7000.0 * v + 300.0 * v**2  # noqa: E731
    assert table.speeds == (0.25, 0.5, 1.0, 2.0, 3.0)
    np.testing.assert_allclose(table.peak(np.array(table.speeds)), law(np.array(table.speeds)))
    raw_fast = np.array([3.0, 1.0, 2.0, 3.0])
    raw_peaks = law(raw_fast) + np.array([40.0, 0.0, 0.0, -40.0])
    assert table.tail_slope == pytest.approx(np.sum(raw_fast * raw_peaks) / np.sum(raw_fast**2))
    assert table.peak(4.5) == pytest.approx(law(3.0) + table.tail_slope * 1.5)
    assert table.peak(0.1) == pytest.approx(1600.0 + (law(0.25) - 1600.0) * 0.4)
    dense = np.linspace(0.0, 6.0, 6001)
    assert np.all(np.diff(table.peak(dense)) > 0.0)
    assert table.is_monotone
    speeds = np.array([0.0, 0.1, 0.3, 0.75, 1.0, 2.5, 5.0])
    np.testing.assert_allclose(table.critical_speed(table.peak(speeds)), speeds, atol=1e-9)
    assert table.critical_speed(1000.0) == 0.0
    with pytest.raises(ValueError):
        table.critical_speed(np.array([6e3, np.nan]))
    restored = TabulatedImpact.from_dict(json.loads(json.dumps(table.to_dict())))
    assert restored == table
    np.testing.assert_array_equal(restored.peak(dense), table.peak(dense))
    origin = _synthetic_impact_table()
    assert origin.zero_speed_peak is None and origin.peak(0.0) == 0.0
    assert origin.peak(0.125) == pytest.approx(0.5 * law(0.25))


def test_tabulated_impact_linear_table_and_zero_speed_knot():
    speeds = np.array([0.25, 0.5, 1.0, 2.0, 3.0])
    linear = TabulatedImpact.from_measurements(speeds, ex.IMPEDANCE * speeds)
    assert linear.tail_slope == pytest.approx(ex.IMPEDANCE)
    levels = np.array([1e3, 6e3, 12e3, 25e3, 40e3])
    np.testing.assert_allclose(
        linear.critical_speed(levels), ex.critical_closing_speed(levels), rtol=1e-9
    )
    knot = TabulatedImpact.from_measurements(
        np.array([0.0, 0.25, 1.0, 2.0]), np.array([1600.0, 3500.0, 9000.0, 17000.0])
    )
    assert knot.base_peak == 1600.0 and knot.peak(0.0) == 1600.0
    assert knot.critical_speed(3500.0) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        TabulatedImpact.from_measurements([0.0, 1.0], [1600.0, 9000.0], zero_speed_peak=1600.0)
    with pytest.raises(ValueError):
        TabulatedImpact.from_measurements([0.2, 0.4], [2000.0, 3000.0])
    slow = TabulatedImpact.from_measurements([0.2, 0.4], [2000.0, 3000.0], tail_slope=ex.IMPEDANCE)
    assert slow.peak(0.6) == pytest.approx(3000.0 + 0.2 * ex.IMPEDANCE)


def test_tabulated_impact_refuses_nonmonotone_inverse():
    table = TabulatedImpact.from_measurements(
        [0.5, 1.0, 1.5, 2.0], [5000.0, 9000.0, 8500.0, 17000.0], zero_speed_peak=1600.0
    )
    assert not table.is_monotone
    assert table.peak(1.25) > 0.0
    with pytest.raises(ValueError):
        table.critical_speed(12e3)
    base_above = TabulatedImpact.from_measurements(
        [0.5, 1.0], [3000.0, 9000.0], zero_speed_peak=4000.0
    )
    with pytest.raises(ValueError):
        base_above.critical_speed(5000.0)


def test_theory_modules_are_drake_free():
    code = (
        "import sys, tether.theory.excursion, tether.theory.impact; "
        "print(sorted(m for m in sys.modules if m.startswith(('pydrake', 'tether'))))"
    )
    environment = dict(os.environ, PYTHONPATH=str(REPOSITORY_ROOT))
    loaded = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=environment
    ).stdout
    assert "pydrake" not in loaded
    assert "tether.physics.fleet" not in loaded
