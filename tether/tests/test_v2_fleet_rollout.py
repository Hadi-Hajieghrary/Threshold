"""Plan v2 P5-T2': the vectorised fleet rollout and the pre-test's event/bounce bookkeeping."""

from __future__ import annotations

import math
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tether.monitor import fleet_rollout as fr
from tether.physics import constants
from tether.physics.fleet import FleetGeometry, equilibrium_state, formation_geometry, operating_point

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MISSION_CACHE = REPOSITORY_ROOT / "records" / "phase5" / "cache" / "phase5_missions.pkl"
FAN = formation_geometry("fan", arc_half_angle=0.55)


def _pair_geometry() -> FleetGeometry:
    return FleetGeometry("pair", np.array([[4.0, 0.0]]), np.array([[-1.5, 0.0]]), np.array([0.0]), 0.0)


def _pair_state(elongation: float, vessel_speed: float = 0.0) -> np.ndarray:
    """Load at the origin, one vessel astern-attached along +x with the given elongation."""
    q = [0.0, 0.0, 0.0, 4.0 + constants.CABLE_REST_LENGTH + elongation + 1.5, 0.0, 0.0]
    v = [0.0, 0.0, 0.0, vessel_speed, 0.0, 0.0]
    return np.array(q + v)


def _no_weather(first_index: int, bodies: int, samples: int = 1, count: int = 4000) -> fr.WeatherFuture:
    return fr.fixed_weather(first_index, np.zeros((count, bodies, 2)), samples)


def test_free_vessel_under_constant_thrust_reaches_thrust_over_drag():
    model = fr.RolloutModel(FAN, stiffness=0.0, damping=0.0, step=1.0e-3)
    state = equilibrium_state(FAN, operating_point(FAN, 1000.0))
    state[18:] = 0.0
    thrust_level = constants.NOMINAL_THRUST
    horizon = 20.0
    steps = int(round(horizon / model.step))
    thrust = np.full((steps, 5), thrust_level)
    result = fr.rollout(model, state, 0.0, horizon, 0, _no_weather(0, 6, count=2002), thrust, record_steps=np.array([steps]))
    final = result.record_state[0]
    headings = state[5:18:3]
    velocities = final[18:].reshape(6, 3)
    terminal = thrust_level / constants.VESSEL_LINEAR_DRAG
    speed_along = velocities[1:, 0] * np.cos(headings) + velocities[1:, 1] * np.sin(headings)
    np.testing.assert_allclose(speed_along, terminal, rtol=1e-4)
    np.testing.assert_allclose(velocities[0], 0.0, atol=1e-12)


def test_taut_cable_pair_oscillates_at_the_pinned_cable_mode():
    geometry = _pair_geometry()
    model = fr.RolloutModel(geometry, damping=0.0, load_linear_drag=0.0, vessel_linear_drag=0.0, load_angular_drag=0.0, vessel_angular_drag=0.0,
                            step=5.0e-4)
    pull = 1000.0
    equilibrium = pull / constants.CABLE_STIFFNESS
    forces = np.zeros((1000, 2, 2))
    forces[:, 0, 0] = -pull
    forces[:, 1, 0] = pull
    horizon = 3.0
    steps = int(round(horizon / model.step))
    result = fr.rollout(model, _pair_state(equilibrium + 0.5 * equilibrium), 0.0, horizon, 0, fr.fixed_weather(0, forces), np.zeros((steps, 1)),
                        record_steps=np.arange(steps + 1), stop_when_all_crossed=False)
    e = result.record_elongation[:, 0, 0] - equilibrium
    assert np.all(result.record_elongation[:, 0, 0] > 0.0)
    t = np.arange(steps + 1) * model.step
    k = np.flatnonzero((e[:-1] <= 0.0) & (e[1:] > 0.0))
    crossings = t[k] - e[k] * model.step / (e[k + 1] - e[k])
    period = float(np.mean(np.diff(crossings)))
    reduced = constants.VESSEL_MASS * constants.LOAD_MASS / (constants.VESSEL_MASS + constants.LOAD_MASS)
    expected = 2.0 * math.pi * math.sqrt(reduced / constants.CABLE_STIFFNESS)
    assert abs(period / expected - 1.0) < 1e-3
    assert abs(expected - 0.335) < 1e-3


@pytest.mark.parametrize("mode", fr.HEADING_MODES)
def test_nothing_moves_without_weather_thrust_or_taut_cables(mode):
    from tether.campaign.mission import MissionSpec, initial_state

    state = initial_state(MissionSpec())
    kinematics_start = state.copy()
    state[3:18:3] -= 0.05  # every chord 5 cm short of its rest length: all cables slack
    model = fr.RolloutModel(FAN, headings=mode, heading_gain=0.0, trim_gain=0.0, step=1.0e-3)
    steps = 2000
    result = fr.rollout(model, state, 10.0, 2.0, 0, _no_weather(1000, 6, count=202), np.zeros((steps, 5)),
                        heading_reference=lambda t: state[5:18:3], record_steps=np.array([0, steps]))
    assert np.array_equal(result.record_state[1], state)
    assert np.all(result.record_elongation[-1, 0] < 0.0)
    assert not result.crossed.any()
    assert kinematics_start is not state


def test_first_upcrossing_time_and_closing_speed_are_interpolated_exactly():
    geometry = _pair_geometry()
    model = fr.RolloutModel(geometry, load_linear_drag=0.0, vessel_linear_drag=0.0, step=1.0e-3)
    result = fr.rollout(model, _pair_state(-0.05, vessel_speed=0.5), 0.0, 0.3, 0, _no_weather(0, 2, count=40), np.zeros((300, 1)))
    assert result.crossed.all()
    assert result.crossing_time[0] == pytest.approx(0.1, abs=1e-9)
    assert result.closing_speed[0] == pytest.approx(0.5, abs=1e-9)
    assert result.hazard(0.4) == 1.0
    assert result.hazard(0.6) == 0.0
    assert result.hazard(0.4, horizon=0.05) == 0.0


def test_steady_tow_is_an_exact_equilibrium_of_the_controlled_rollout():
    from tether.campaign.fleet_run import fan_heading_reference
    from tether.physics.fleet import CableParameters

    operating = operating_point(FAN, 1000.0)
    state = equilibrium_state(FAN, operating)
    reference = fan_heading_reference(FAN, operating, CableParameters(), 500.0, 100.0)
    model = fr.RolloutModel(FAN, headings="controlled", step=1.0e-3)
    steps = 2000
    result = fr.rollout(model, state, 30.0, 2.0, 0, _no_weather(3000, 6, count=202), np.tile(operating.thrusts, (steps, 1)),
                        heading_reference=lambda t: reference, record_steps=np.array([0, steps]))
    np.testing.assert_allclose(result.record_elongation[-1, 0], 1000.0 / constants.CABLE_STIFFNESS, atol=1e-7)
    np.testing.assert_allclose(result.record_state[1][18:], state[18:], atol=1e-7)


def test_controller_hold_uses_the_previous_command_at_an_update_instant():
    assert fr.controller_hold_time(0.0) == 0.0
    assert fr.controller_hold_time(0.02) == pytest.approx(0.0)
    assert fr.controller_hold_time(0.0205) == pytest.approx(0.02)
    assert fr.controller_hold_time(0.04) == pytest.approx(0.02)
    rows = fr.thrust_on_steps(lambda t: np.array([t]), 0.02, 25, 1.0e-3)
    assert rows[0, 0] == pytest.approx(0.0)
    assert rows[1, 0] == pytest.approx(0.02)
    assert rows[20, 0] == pytest.approx(0.02)
    assert rows[21, 0] == pytest.approx(0.04)


def test_common_ar1_future_follows_the_generator_law_and_draw_order():
    phi = 0.99875
    current = np.array([0.3, -1.2])
    draws = fr.common_ar1_future(np.random.default_rng(7), 4, current, 5, phi)
    replay = np.random.default_rng(7)
    gaussian = replay.standard_normal((4, 5, 2))
    radial = np.sqrt(1.0 / replay.chisquare(3.0, size=(4, 5)))
    expected = np.empty((4, 6, 2))
    expected[:, 0] = current
    for j in range(1, 6):
        expected[:, j] = phi * expected[:, j - 1] + math.sqrt(1 - phi**2) * gaussian[:, j - 1] * radial[:, j - 1, None]
    np.testing.assert_allclose(draws, expected, rtol=0, atol=1e-15)
    many = fr.common_ar1_future(np.random.default_rng(8), 200_000, current, 1, phi, distribution="gaussian")
    np.testing.assert_allclose(many[:, 1].mean(axis=0), phi * current, atol=5e-3 * math.sqrt(1 - phi**2) * 3)
    np.testing.assert_allclose(many[:, 1].var(axis=0), 1 - phi**2, rtol=0.02)


def test_weather_future_matches_the_common_mode_generator_structure():
    from tether.physics.weather import stationary_weather_forces

    background = stationary_weather_forces(5001, 30.0, distribution="student_t3", direction="common")
    std = np.array([constants.LOAD_WEATHER_STD] + [constants.VESSEL_WEATHER_STD] * 5)
    standardized = background[:, 0, :] / std[0]
    np.testing.assert_allclose(background, standardized[:, None, :] * std[None, :, None], rtol=1e-12, atol=1e-9)
    future = fr.WeatherFuture(100, standardized[None, 100:110], np.tile(std, (10, 1)), np.zeros((10, 6, 2)))
    fx, fy = future.forces(3)
    np.testing.assert_allclose(fx[0], background[103, :, 0], atol=1e-9)
    np.testing.assert_allclose(fy[0], background[103, :, 1], atol=1e-9)


@pytest.mark.skipif(not MISSION_CACHE.exists(), reason="needs the committed Phase 5 cache")
def test_replay_of_the_recorded_future_reproduces_the_truth():
    from tether.campaign.mission import MissionSpec, mission_geometry, mission_schedules, mission_weather

    with MISSION_CACHE.open("rb") as handle:
        mission = pickle.load(handle)[2]
    truth = mission["truth"]
    spec = MissionSpec()
    schedules = mission_schedules(spec)
    weather = mission_weather(spec, mission["seed"])
    model = fr.RolloutModel(mission_geometry(spec), headings="controlled", step=1.0e-3)
    ticks = np.round(np.arange(200, 1290) * 0.1, 10)
    index = np.round(ticks / 1e-3).astype(int)
    slack = [(t, c) for t, i in zip(ticks, index) for c in range(5) if truth.elongation[i, c] <= 0.0]
    assert len(slack) > 10
    for t, cable in slack[:: len(slack) // 4][:4]:
        k0 = int(round(t / 0.01))
        steps = 500
        result = fr.rollout(model, truth.state[k0], t, 0.5, cable, fr.fixed_weather(k0, weather[k0 : k0 + 52]), fr.thrust_on_steps(schedules.surge, t, steps, 1e-3),
                            heading_reference=schedules.heading, record_steps=np.arange(0, steps + 1, 10), stop_when_all_crossed=False)
        start = int(round(t / 1e-3))
        grid = start + np.arange(0, 501, 10)
        assert np.max(np.abs(result.record_elongation[:, 0, cable] - truth.elongation[grid, cable])) < 0.01
        assert np.max(np.abs(result.record_rate[:, 0, cable] - truth.rate[grid, cable])) < 0.05


# ----------------------------------------------------------------------------- pre-test bookkeeping


def _excursions(spans, duration=10.0):
    """1 ms series slack (e < 0) inside each (onset, re-engagement) span, taut elsewhere."""
    time = np.round(np.arange(int(round(duration / 1e-3)) + 1) * 1e-3, 10)
    e = np.full(time.size, 0.02)
    rate = np.zeros(time.size)
    for onset, back in spans:
        inside = (time > onset) & (time < back)
        middle = 0.5 * (onset + back)
        half = 0.5 * (back - onset)
        e[inside] = -0.1 * (1.0 - ((time[inside] - middle) / half) ** 2)
        rate[inside] = 0.2 * (time[inside] - middle) / half**2
    return time, e, rate


def test_bounce_rule_and_interval_split_at_a_reengagement_between_ticks():
    from tether.campaign.v2.p5_pretest import cable_intervals

    spans = [(1.005, 1.505), (2.005, 3.215), (3.245, 3.555), (6.005, 6.555)]
    series_time, e, rate = _excursions(spans)
    ticks = np.round(np.arange(101) * 0.1, 10)
    slack = np.interp(ticks, series_time, e) <= 0.0
    intervals = cable_intervals(ticks, slack, series_time, e, rate)
    starts = ticks[intervals.first]
    np.testing.assert_allclose(starts, [1.1, 2.1, 3.3, 6.1])
    np.testing.assert_allclose(ticks[intervals.last], [1.5, 3.2, 3.5, 6.5])
    np.testing.assert_allclose(intervals.onset, [1.005, 2.005, 3.245, 6.005], atol=2e-3)
    assert intervals.bounce.tolist() == [False, True, True, False]


def test_declarations_are_written_once_and_never_edited(tmp_path, monkeypatch):
    from tether.campaign.v2 import p5_pretest

    path = tmp_path / "declarations.json"
    monkeypatch.setattr(p5_pretest, "DECLARATIONS_PATH", path)
    first = p5_pretest.write_declarations()
    assert p5_pretest.write_declarations() == first
    monkeypatch.setattr(p5_pretest, "DECLARATIONS", dict(p5_pretest.DECLARATIONS, horizons="changed"))
    with pytest.raises(RuntimeError):
        p5_pretest.write_declarations()
