"""Plan v2 weather additions: front(c) lags, the gust class, episode bookkeeping (IV.5, P0-W)."""

import math

import numpy as np
import pytest

from tether.campaign.v2 import phase0_weather as p0w
from tether.physics import constants
from tether.physics.weather import (
    RAISED_COSINE_ENERGY,
    draw_gust_marks,
    front_arrival_delays,
    gust_background_factor,
    gust_class_weather,
    gust_forces,
    gust_second_moments,
    hill_tail_index,
    lagged_front_weather_forces,
    pinned_stationary_std,
    raised_cosine_pulse,
    stationary_weather_forces,
)


def test_exact_common_front_is_bit_identical_to_v1_front():
    positions, _ = p0w.fan_positions()
    for angle in (p0w.BOW_QUARTERING, p0w.BROADSIDE):
        v1 = stationary_weather_forces(31, 20.0, direction="front", front_angle=angle, scale=0.5)
        v2 = lagged_front_weather_forces(31, 20.0, positions=positions, front_angle=angle, speed=math.inf, scale=0.5)
        assert np.array_equal(v1, v2)


def test_v1_weather_defaults_unchanged():
    # the local class is untouched by the v2 additions: same draws, same arithmetic
    a = stationary_weather_forces(5, 3.0)
    b = stationary_weather_forces(5, 3.0, direction="local", scale=1.0)
    assert np.array_equal(a, b)
    assert a.shape == (302, 6, 2)


def test_front_delays_follow_the_frozen_front():
    positions = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    # front pushing toward -x travels toward -x: the body at x = 20 is hit first
    delays = front_arrival_delays(positions, math.pi, 10.0)
    assert delays.tolist() == [200, 100, 0]
    assert front_arrival_delays(positions, 0.0, 5.0).tolist() == [0, 200, 400]
    assert front_arrival_delays(positions, 0.0, math.inf).tolist() == [0, 0, 0]


def test_front_bodies_see_one_stream_delayed_by_integer_samples():
    positions, _ = p0w.fan_positions()
    angle, speed = p0w.BOW_QUARTERING, 10.0
    delays = front_arrival_delays(positions, angle, speed)
    forces = lagged_front_weather_forces(8, 30.0, positions=positions, front_angle=angle, speed=speed)
    sigma = pinned_stationary_std()
    direction = np.array([math.cos(angle), math.sin(angle)])
    scalar = forces @ direction / sigma[None, :]  # (samples, bodies) standardized along the front
    first = int(np.argmin(delays))
    assert np.allclose(forces @ np.array([-direction[1], direction[0]]), 0.0, atol=1e-9)
    for body in range(positions.shape[0]):
        d = int(delays[body])
        assert np.allclose(scalar[500:, body], scalar[500 - d : scalar.shape[0] - d, first], rtol=1e-12, atol=1e-12)


def test_raised_cosine_pulse_shape_and_energy():
    x = np.linspace(0.0, 1.0, 200001)
    e = raised_cosine_pulse(x)
    assert e.max() == pytest.approx(1.0)
    assert raised_cosine_pulse(np.array([-0.1, 0.0, 1.0, 1.2])).tolist() == [0.0, 0.0, 0.0, 0.0]
    from scipy.integrate import trapezoid

    assert trapezoid(e**2, x) == pytest.approx(RAISED_COSINE_ENERGY, rel=1e-6)


def test_variance_preservation_factor():
    moments = gust_second_moments()
    assert moments["amplitude_second_moment"] == pytest.approx(3.0)
    assert moments["mean_duration"] == pytest.approx(2.0 * math.exp(0.045))
    g = 1.5**2 * moments["isotropic_component_variance"]
    assert g == pytest.approx(0.1 * 3.0 * 0.5 * 2.0 * math.exp(0.045) * 0.375 * 2.25)
    assert gust_background_factor() == pytest.approx(math.sqrt(1.0 - g))
    assert gust_background_factor() == pytest.approx(0.857452, abs=1e-6)


def test_gust_marks_are_deterministic_pareto_and_on_their_own_stream():
    a = draw_gust_marks(3, 2, -40.0, 600.0)
    b = draw_gust_marks(3, 2, -40.0, 600.0)
    assert np.array_equal(a.times, b.times) and np.array_equal(a.amplitudes, b.amplitudes)
    assert np.all(a.amplitudes >= 1.0) and np.all(np.diff(a.times) >= 0.0)
    amplitudes = np.concatenate([draw_gust_marks(seed, 0, 0.0, 10000.0).amplitudes for seed in range(60)])
    assert 2.85 <= hill_tail_index(amplitudes, amplitudes.size // 10) <= 3.15
    durations = np.concatenate([draw_gust_marks(seed, 1, 0.0, 10000.0).durations for seed in range(20)])
    assert np.median(durations) == pytest.approx(2.0, rel=0.03)
    assert np.std(np.log(durations)) == pytest.approx(0.3, rel=0.05)


def test_gust_class_shares_its_background_draws_with_the_gaussian_cell():
    forces, background, gusts, marks = gust_class_weather(12, 30.0, return_components=True)
    gaussian = stationary_weather_forces(12, 30.0)
    assert np.allclose(background, gust_background_factor() * gaussian, rtol=1e-12, atol=1e-9)
    assert np.array_equal(forces, background + gusts)
    assert len(marks) == constants.VESSEL_COUNT + 1


def test_gust_term_variance_matches_its_realized_marks():
    """Superposition and sampling: the gust term's second moment equals sum_n (a0 A)^2 s^2 t_g 3/8."""
    a0 = 1.5 * pinned_stationary_std()
    measured = 0.0
    predicted = 0.0
    for seed in range(4):
        gusts, marks = gust_forces(seed, 600.0)
        measured += float(np.sum(gusts**2))
        for body, m in enumerate(marks):
            inside = (m.times >= 0.0) & (m.times < gusts.shape[0] * constants.WEATHER_PERIOD)
            predicted += float(np.sum((a0[body] * m.amplitudes[inside]) ** 2 * m.durations[inside] * RAISED_COSINE_ENERGY)) / constants.WEATHER_PERIOD
    assert measured / predicted == pytest.approx(1.0, abs=0.05)


def test_fleet_wide_gust_structures():
    positions, _ = p0w.fan_positions()
    angle = p0w.BOW_QUARTERING
    common, marks = gust_forces(4, 60.0, structure="common", front_angle=angle)
    assert len(marks) == 1
    a0 = 1.5 * pinned_stationary_std()
    # every body carries the same standardized pulse train along the front
    standardized = common / a0[None, :, None]
    for body in range(1, 6):
        assert np.allclose(standardized[:, body], standardized[:, 0])
    delays = front_arrival_delays(positions, angle, 10.0)
    front, _ = gust_forces(4, 60.0, structure="front", front_angle=angle, delays=delays)
    first = int(np.argmin(delays))
    for body in range(6):
        d = int(delays[body])
        lhs = front[4000:, body] / a0[body]
        rhs = front[4000 - d : front.shape[0] - d, first] / a0[first]
        assert np.allclose(lhs, rhs, atol=1e-9)


def test_relative_load_series_matches_the_plant_formula():
    from tether.physics.fleet import relative_gust_load

    forces = gust_class_weather(2, 2.0)
    _, units = p0w.fan_positions()
    w_rel, w_c = p0w.conjugate_loads(forces, units)
    for k in range(0, forces.shape[0], 37):
        assert np.allclose(w_rel[k], relative_gust_load(forces[k], units), rtol=1e-12, atol=1e-9)
    # W_c by hand
    k = 50
    expected = p0w.DRAG_REDUCED * ((forces[k, 0] / p0w.C_L)[None, :] - forces[k, 1:] / p0w.C_A) @ np.eye(2)
    assert np.allclose(w_c[k], np.sum(expected * units, axis=1))


def test_merge_rule_and_grid_counts():
    series = np.zeros(1000)
    series[100:110] = 5.0
    series[209:215] = 6.0  # gap of 99 samples -> merged
    series[315:320] = 7.0  # gap of 100 samples -> new episode
    starts, stops, maxima = p0w.merged_episodes(series, 1.0)
    assert starts.tolist() == [100, 315]
    assert stops.tolist() == [215, 320]
    assert maxima.tolist() == [6.0, 7.0]
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=20000)) * 0.1
    x = np.abs(x - x.mean())
    levels = np.linspace(0.0, x.max() + 1.0, 57)
    grid = p0w.episode_counts_on_grid(x, levels)
    explicit = [p0w.merged_episodes(x, u)[0].size for u in levels]
    assert grid.tolist() == explicit


def test_background_negligible_level_rule():
    levels = np.arange(6) * 1.0
    gust = np.array([100, 80, 50, 30, 10, 0])
    back = np.array([900, 20, 4, 4, 0, 0])
    # ratio <= 0.1 holds from index 2 on except index 3 (4 > 3): the rule needs all higher levels
    assert p0w.smallest_negligible_level(levels, gust, back) == 4.0
    back2 = np.array([900, 20, 4, 2, 0, 0])
    assert p0w.smallest_negligible_level(levels, gust, back2) == 2.0
    assert p0w.hill_at_threshold(np.array([2.0, 4.0]), 1.0) == pytest.approx(2.0 / (math.log(2) + math.log(4)))


def test_w3_analytic_forms():
    positions, units = p0w.fan_positions()
    configs = {c["name"]: c for c in p0w.w3_configurations()}
    local = p0w.analytic_std(configs["local"], units, positions, "rel")
    assert np.allclose(local, p0w.PAIR_REDUCED_MASS * math.hypot(3500 / 2500, 700 / 600))
    common = p0w.analytic_std(configs["exact common bow_quartering"], units, positions, "rel")
    projection = np.abs(units @ np.array([math.cos(p0w.BOW_QUARTERING), math.sin(p0w.BOW_QUARTERING)]))
    assert np.allclose(common, math.sqrt(2) * p0w.PAIR_REDUCED_MASS * (3500 / 2500 - 700 / 600) * projection)
    # numeric agreement on a short record, loose
    forces = p0w.w3_forces(configs["front(10 m/s) bow_quartering"], 3, positions)
    w_rel, _ = p0w.conjugate_loads(forces, units)
    analytic = p0w.analytic_std(configs["front(10 m/s) bow_quartering"], units, positions, "rel")
    assert np.all(np.abs(w_rel.std(axis=0) / analytic - 1.0) < 0.5)


def test_h4prime_classification():
    t0 = 1000.0
    assert p0w.classify_wc_episode(0.5, 1100.0, t0) == "R1"
    assert p0w.classify_wc_episode(0.5, 1000.0 + 0.3 * (t0 / 350.0) * 600.0 / 0.5 + 1.0, t0) == "transitional"
    assert p0w.classify_wc_episode(3.5, 1100.0, t0) == "R2"
    assert p0w.classify_wc_episode(3.4, 1100.0, t0) == "transitional"
    assert p0w.classify_wc_episode(0.72, 1010.0, t0) == "transitional"
