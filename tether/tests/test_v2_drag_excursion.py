"""Unit tests of the drag-inclusive one-cable excursion (plan v2 II.4-II.6) and of Claim A's
prediction helpers (tether/campaign/v2/claim_a.py).

Every closed form is checked against the reduced ODE in the regime where it is exact: the
vessel-dominant forms of Prop. 4' and the three-piece map against the ODE with the load
pinned; Cor. 5' against the drag-free two-leg integration; Prop. 2 against the drag-free
slack ODE; the drift crossing against the two-lag slack solution.  Pretensions are
deliberately off the committed grid (900 N) so no test re-derives a committed number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tether.theory import drag_excursion as dx

T0 = 900.0


def _square(total: int, on: int, off: int, value: float) -> np.ndarray:
    forcing = np.zeros(total)
    forcing[on:off] = value
    return forcing


# ----------------------------------------------------------------------------- the ODE


def test_taut_equilibrium_is_a_fixed_point_and_records_no_marks():
    result = dx.simulate(dx.OneCableODE(), T0, np.zeros(200), np.zeros(200), trace=True)
    assert np.max(np.abs(result.trace_state[:, 0, 0] - T0 / dx.STIFFNESS)) < 1e-15
    assert np.max(np.abs(result.trace_tension[:, 0] - T0)) < 1e-9
    assert not result.marks.t_up


def test_discretization_is_the_exact_zoh_map():
    ode = dx.OneCableODE()
    phi, gamma = ode.discretized(True, 1e-3)
    phi2, gamma2 = ode.discretized(True, 5e-4)
    assert np.allclose(phi2 @ phi2, phi, rtol=0, atol=1e-12)
    assert np.allclose(phi2 @ gamma2 + gamma2, gamma, rtol=0, atol=1e-15)


def test_slack_mode_matches_the_exact_two_lag_solution():
    ode = dx.OneCableODE()
    x0 = np.array([-2.0, 0.3, -0.1])
    w_a, w_l = -500.0, 800.0
    result = dx.simulate(ode, T0, np.full(100, w_a), np.full(100, w_l), x0=x0, trace=True)
    exact = dx.slack_solution(ode, x0, T0, w_a, w_l, result.trace_time)
    assert np.max(np.abs(result.trace_state.transpose(1, 0, 2)[:, :, 0] - exact)) < 1e-10


def test_mark_interpolation_matches_the_exact_crossing():
    ode = dx.OneCableODE(pinned_load=True)
    x0 = np.array([-0.05, 0.0, 0.0])
    result = dx.simulate(ode, T0, np.zeros(100), np.zeros(100), x0=x0)
    marks = result.marks.arrays()
    # e(t) = -0.05 + v_T (t - tau (1 - exp(-t/tau))): first root
    v_t = T0 / dx.C_A

    def e(t):
        return -0.05 + v_t * (t + dx.TAU_A * math.expm1(-t / dx.TAU_A))

    from scipy.optimize import brentq

    root = brentq(e, 0.01, 1.0, xtol=1e-14)
    assert abs(marks["t_up"][0] - root) < 1e-6
    assert abs(marks["v_up"][0] - v_t * -math.expm1(-root / dx.TAU_A)) < 1e-6
    assert marks["depth"][0] == pytest.approx(0.05, abs=1e-12)


def test_shared_forcing_columns_give_identical_paths():
    rng = np.random.default_rng(3)
    w = rng.normal(0.0, 2000.0, (300, 1))
    single = dx.simulate(dx.OneCableODE(), T0, w, np.zeros((300, 1))).marks.arrays()
    shared = dx.simulate(dx.OneCableODE(), np.array([T0, T0]), w, np.zeros((300, 1)),
                         columns_a=np.array([0, 0]), columns_l=np.array([0, 0])).marks.arrays()
    assert single["t_up"].size > 0
    for p in (0, 1):
        rows = shared["path"] == p
        # identical up to BLAS round-off (the matrix products differ in width)
        assert np.allclose(shared["t_up"][rows], single["t_up"], rtol=0, atol=1e-9)
        assert np.allclose(shared["v_up"][rows], single["v_up"], rtol=0, atol=1e-9)


# ----------------------------------------------------------------------------- Prop. 4' (vessel-dominant, exact)


def test_prop4_drift_depth_post_gust_deepening_and_undriven_return_are_exact_with_the_load_pinned():
    ode = dx.OneCableODE(pinned_load=True)
    v_d, t_x = 1.2, 3.0
    force = T0 + dx.C_A * v_d  # vessel terminal speed v_d under the gust
    samples = int(round(t_x / dx.WEATHER_PERIOD))
    result = dx.simulate(ode, T0, _square(3000, 0, samples, -force), np.zeros(3000), x0=np.array([-1e-12, 0.0, 0.0]), trace=True)
    at_shutoff = int(round(t_x / dx.SAMPLE_PERIOD))
    assert -result.trace_state[at_shutoff, 0, 0] == pytest.approx(dx.shutoff_depth(v_d, t_x), rel=1e-9)
    assert -result.trace_state[at_shutoff, 1, 0] == pytest.approx(dx.shutoff_speed(v_d, t_x), rel=1e-9)
    marks = result.marks.arrays()
    v_s = dx.shutoff_speed(v_d, t_x)
    total = dx.shutoff_depth(v_d, t_x) + dx.post_gust_deepening(v_s, T0)
    assert marks["depth"][0] == pytest.approx(total, rel=1e-6)
    assert marks["t_deep"][0] == pytest.approx(t_x + dx.post_gust_stop_time(v_s, T0), abs=1e-3)
    assert marks["v_up"][0] == pytest.approx(dx.undriven_return_speed(total, T0), rel=1e-5)
    assert marks["t_up"][0] - marks["t_deep"][0] == pytest.approx(dx.undriven_return_time(total, T0), abs=2e-3)


def test_shutoff_depth_is_linear_in_duration_beyond_tau():
    durations = np.array([6.0, 8.0, 10.0, 12.0])
    depths = dx.shutoff_depth(1.0, durations)
    slopes = np.diff(depths) / np.diff(durations)
    assert np.all(np.abs(slopes - 1.0) < 0.04)
    # small durations: quadratic, v_d t^2/(2 tau)
    assert dx.shutoff_depth(1.0, 0.01) == pytest.approx(1e-4 / (2 * dx.TAU_A), rel=1e-2)


def test_undriven_return_small_depth_limit_and_saturation():
    v_t = T0 / dx.C_A
    small = 1e-4
    ballistic = math.sqrt(2 * T0 * small / dx.M_A)
    # II.4: sqrt(2 a_A Delta) (1 - w/3 + O(w^2)), w = V_up/v_T
    assert dx.undriven_return_speed(small, T0) == pytest.approx(ballistic * (1.0 - ballistic / v_t / 3.0), rel=1e-4)
    deep = dx.undriven_return_speed(np.array([2.0, 10.0, 30.0]), T0)
    assert np.all(np.diff(deep) > 0) and np.all(deep <= v_t)
    assert deep[-1] > 0.99 * v_t
    assert dx.undriven_ceiling(T0) == pytest.approx(dx.IMPEDANCE * v_t)


def test_critical_depth_inverts_the_undriven_return():
    v_t = T0 / dx.C_A
    speeds = np.array([0.1, 0.5, 1.0, 2.0, 0.99 * v_t])
    depth = dx.critical_depth(speeds, T0)
    assert np.allclose(dx.undriven_return_speed(depth, T0), speeds, rtol=1e-8)
    assert math.isinf(dx.critical_depth(1.01 * v_t, T0))


def test_drag_inclusive_severance_level_inverts_the_shutoff_depth():
    v_b, t_x = 0.8, 6.0
    level = dx.drag_inclusive_severance_level(v_b, T0, t_x, exact_depth=True)
    assert dx.shutoff_depth(dx.drift_speed(level, T0), t_x) == pytest.approx(dx.critical_depth(v_b, T0), rel=1e-10)
    asymptotic = dx.drag_inclusive_severance_level(v_b, T0, t_x)
    assert asymptotic > level  # the asymptote under-counts the depth, so it demands more excess
    assert math.isnan(dx.drag_inclusive_severance_level(v_b, T0, 1.0))


# ----------------------------------------------------------------------------- R1 three-piece map


def test_three_piece_map_with_its_exact_leg_is_the_pinned_load_ode():
    ode = dx.OneCableODE(pinned_load=True)
    amplitude, t_x = 3000.0, 0.3
    samples = int(round(t_x / dx.WEATHER_PERIOD))
    result = dx.simulate(ode, T0, _square(1500, 0, samples, -amplitude), np.zeros(1500), x0=np.array([-1e-12, 0.0, 0.0]))
    marks = result.marks.arrays()
    exact = dx.three_piece_map(amplitude, T0, t_x, include_impulse_depth=True, drag_in_impulse=True)
    assert marks["v_up"][0] == pytest.approx(exact.return_speed, rel=1e-5)
    assert marks["depth"][0] == pytest.approx(exact.total_depth, rel=1e-5)


def test_three_piece_map_first_order_drag_loss():
    # the impulsive map from a tiny impulse: V_up / v_s = 1 - (2/3) r + O(r^2)
    for r in (0.005, 0.01, 0.02):
        v_t = T0 / dx.C_A
        amplitude = T0 + r * v_t * dx.M_A / 0.01
        mapped = dx.three_piece_map(amplitude, T0, 0.01)
        assert mapped.return_speed / mapped.shutoff_speed == pytest.approx(1.0 - 2.0 * r / 3.0, abs=2.0 * r * r)


def test_required_impulse_inverts_the_committed_map():
    for v_b in (0.3, 0.8, 1.5):
        impulse = dx.required_impulse(v_b, T0)
        for t_x in (0.2, 0.5):
            mapped = dx.three_piece_map(T0 + impulse / t_x, T0, t_x)
            assert mapped.return_speed == pytest.approx(v_b, rel=1e-9)
        assert dx.r1_required_amplitude(v_b, T0, 0.3) == pytest.approx(T0 + impulse / 0.3)
    assert math.isinf(dx.required_impulse(T0 / dx.C_A * 1.01, T0))


def test_r1_rate_curve_is_a_gaussian_in_the_amplitude():
    curve = dx.r1_rate_curve(np.array([T0, 2 * T0]), T0, 250.0, 1e-3)
    assert curve[0] == pytest.approx(1e-3)
    assert math.log(curve[1] / curve[0]) == pytest.approx(-(4 * T0**2 - T0**2) / (2 * 250.0**2))


# ----------------------------------------------------------------------------- Cor. 5' and Props. 2, 3'


def test_cor5_square_gust_return_speed_matches_the_two_leg_integration():
    w, t_g = 2600.0, 0.4
    # forcing leg: free two-body, drag-free; relative acceleration (T0 - W)/m_eff
    forcing = dx.OneCableODE(c_a=0.0, c_l=0.0, m_l=dx.M_L)
    force = w * dx.M_A / dx.M_EFF  # vessel-only force with W_rel = W
    leg = dx.slack_solution(forcing, np.zeros(3), T0, -force, 0.0, np.array([t_g]))[:, 0]
    depth, speed = -leg[0], -(leg[1] - leg[2])
    # return leg: the formation's restoring acceleration a0 = T0/m_fleet
    v_up = math.sqrt(speed**2 + 2.0 * T0 / dx.M_FLEET * depth)
    assert v_up == pytest.approx(dx.square_gust_return_speed(w, T0, t_g), rel=1e-12)


def test_cor5_severance_level_inverts_the_return_speed_and_reduces_at_b0():
    levels = np.array([6000.0, 12000.0, 25000.0])
    for t_g in (0.5, 1.0, 2.0):
        w = dx.square_gust_severance_level(levels, T0, t_g)
        # kappa carries Z = f sqrt(k m_eff) with f = 0.880 (0.15 % below the pinned 7993.5)
        impedance = dx.IMPACT_FACTOR * math.sqrt(dx.STIFFNESS * dx.M_EFF)
        assert np.allclose(impedance * dx.square_gust_return_speed(w, T0, t_g), levels, rtol=1e-12)
        kappa = dx.square_gust_kappa(T0, t_g)
        reduced = dx.square_gust_severance_level(levels, T0, t_g, b=0.0)
        assert np.allclose(reduced, (T0 + np.sqrt(T0**2 + 4 * kappa * T0 * levels**2)) / 2)


def test_prop2_ballistic_excursion_on_the_drag_free_ode():
    ode = dx.OneCableODE(c_a=0.0, pinned_load=True)
    u = 0.4
    result = dx.simulate(ode, T0, np.zeros(200), np.zeros(200), x0=np.array([0.0, -u, 0.0]))
    marks = result.marks.arrays()
    a = T0 / dx.M_A
    expected = dx.ballistic_excursion(u, a)
    assert marks["u_entry"][0] == pytest.approx(u)
    # the 1 ms sample after the crossing already carries up to one physics step of taut
    # dynamics (c V h/m_A), exactly as the plant tracker's interpolation does
    assert marks["v_up"][0] == pytest.approx(expected["return_speed"], rel=1e-3)
    assert marks["t_up"][0] == pytest.approx(expected["return_time"], abs=1e-6)
    assert marks["depth"][0] == pytest.approx(expected["depth"], rel=1e-4)


def test_prop3_identity_from_the_deepest_point_and_the_offset_estimator():
    ode = dx.OneCableODE(pinned_load=True)
    result = dx.simulate(ode, T0, np.zeros(1000), np.zeros(1000), x0=np.array([-1.5, 0.0, 0.0]), trace=True)
    marks = result.marks.arrays()
    v_up, depth = marks["v_up"][0], marks["depth"][0]
    # depth-mean (work) acceleration over the return leg reproduces V_up exactly
    rows = result.trace_time <= marks["t_up"][0]
    e = result.trace_state[rows, 0, 0]
    rate = result.trace_state[rows, 1, 0]
    from scipy.integrate import trapezoid

    work = trapezoid(np.gradient(rate, result.trace_time[rows]), e)
    assert dx.closing_speed(work / depth, depth) == pytest.approx(v_up, rel=2e-3)
    # the time-mean acceleration is not the operative quantity under drag: the undriven
    # return decelerates its own acceleration, so sqrt(2 a_time Delta) overstates V_up
    time_mean = v_up / (marks["t_up"][0] - marks["t_deep"][0])
    assert dx.closing_speed(time_mean, depth) > 1.05 * v_up
    # the offset estimator (V^2 - u^2)/(2 Delta) returns 0 on a ballistic excursion (Prop. 2)
    ballistic = dx.ballistic_excursion(0.4, T0 / dx.M_A)
    assert (ballistic["return_speed"] ** 2 - 0.4**2) / (2 * ballistic["depth"]) == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------------- the drift criterion


@pytest.mark.parametrize("ode", [dx.OneCableODE(), dx.OneCableODE(m_l=dx.M_L, c_l=dx.C_L), dx.OneCableODE(pinned_load=True)])
def test_drift_crossing_is_where_the_terminal_relative_speed_vanishes(ode):
    crossing = ode.drift_crossing_vessel_only
    for lam, sign in ((crossing * 0.99, 1.0), (crossing * 1.01, -1.0)):
        force = lam * T0 * dx.VESSEL_FORCE_PER_WC
        late = dx.slack_solution(ode, np.array([-1.0, 0.0, 0.0]), T0, -force, 0.0, np.array([60.0]))[:, 0]
        assert np.sign(late[1] - late[2]) == sign
    if not ode.pinned_load and ode.c_l == dx.C_L:
        assert crossing == pytest.approx(1.0, abs=1e-12)  # the plan's c_eff reading


def test_onset_acceleration_crossing_is_the_mass_form():
    ode = dx.OneCableODE()
    lam = ode.onset_acceleration_crossing_vessel_only
    force = lam * T0 * dx.VESSEL_FORCE_PER_WC
    a, b = ode.matrices(False)
    acceleration = (a @ np.zeros(3) + b @ np.array([T0, -force, 0.0]))
    assert acceleration[1] - acceleration[2] == pytest.approx(0.0, abs=1e-12)


def test_full_ode_drift_depth_agrees_with_prop4_in_the_sustained_regime():
    ode = dx.OneCableODE()
    lam, t_x = 1.5, 6.0
    force = lam * T0 * dx.VESSEL_FORCE_PER_WC
    result = dx.simulate(ode, T0, _square(3000, 1000, 1600, -force), np.zeros(3000), trace=True)
    depth = -result.trace_state[16000, 0, 0]
    closed = dx.shutoff_depth(dx.drift_speed(lam * T0, T0), t_x)
    assert depth == pytest.approx(closed, rel=0.05)


def test_vessel_only_forcing_realises_the_declared_conjugates():
    w_c = 1000.0
    force = w_c * dx.VESSEL_FORCE_PER_WC
    assert dx.C_EFF * (0.0 / dx.C_L + force / dx.C_A) == pytest.approx(w_c)
    assert dx.M_EFF * force / dx.M_A == pytest.approx(dx.WREL_PER_WC_VESSEL_ONLY * w_c)
    assert dx.TAU_A == pytest.approx(1.714, abs=1e-3) and dx.TAU_LF == pytest.approx(0.710, abs=1e-3)
    assert dx.C_EFF == pytest.approx(329.06, abs=0.01) and dx.B_FLEET == pytest.approx(0.076, abs=1e-3)


# ----------------------------------------------------------------------------- claim_a helpers


def test_classify_mark_reads_the_weather_side_only():
    from tether.campaign.v2 import claim_a as ca
    from tether.campaign.v2.phase0_weather import merged_episodes

    w_c = np.zeros(2000)
    w_c[500:530] = 1.5 * T0  # 0.30 s at 1.5 T0: R1 (v_s = 0.45 T0 * 0.3/600 < 0.3 v_T)
    w_c[1000:1400] = 1.2 * T0  # 4.0 s: R2
    episodes = merged_episodes(w_c, T0)
    assert ca.classify_mark(4.0, 4.9, w_c, episodes, T0)[0] == "N"
    label, t_x, peak = ca.classify_mark(5.1, 5.6, w_c, episodes, T0)
    assert label == "R1" and t_x == pytest.approx(0.30) and peak == pytest.approx(1.5 * T0)
    assert ca.classify_mark(13.0, 15.0, w_c, episodes, T0)[0] == "R2"
    # both episodes inside the interval: the larger W^c_max decides
    assert ca.classify_mark(4.0, 16.0, w_c, episodes, T0)[0] == "R1"


def test_flagged_lti_crossings_equal_brute_force():
    from tether.campaign.v2 import claim_a as ca

    setup = ca.lti_setup(900.0, 700.0)
    levels = {"z1": 1.0 * setup["sigma_e"], "z2": 2.0 * setup["sigma_e"]}
    flagged = ca.lti_crossings(setup, np.random.default_rng(5), 8, 600, levels, chunk=250)
    brute = ca.lti_crossings(setup, np.random.default_rng(5), 8, 600, levels, chunk=250, brute_force=True)
    for name in levels:
        assert flagged[name]["time"].size == brute[name]["time"].size > 0
        assert np.array_equal(flagged[name]["cable"], brute[name]["cable"])
        assert np.array_equal(flagged[name]["direction"], brute[name]["direction"])
        assert np.allclose(flagged[name]["time"], brute[name]["time"], rtol=0, atol=1e-12)


def test_decluster_path_is_the_events_module_rule():
    from tether.campaign.v2 import claim_a as ca
    from tether.campaign.v2 import events as ev

    rng = np.random.default_rng(11)
    times = np.sort(rng.uniform(0.0, 200.0, 300))
    cables = rng.integers(0, 5, 300)
    directions = np.where(rng.random(300) < 0.5, -1, 1)
    result = ca.decluster_path(times, cables, directions, 3.0, 200.0)
    downs = np.flatnonzero((directions < 0) & (times >= 3.0) & (times <= 200.0))
    ups = directions > 0
    reference = ev.primary_onsets(times[downs], cables[downs], times[ups], cables[ups])
    assert np.array_equal(result["primary"], reference.primary)
    assert result["exposure"] == pytest.approx(ev.excluded_exposure(times[ups], 3.0, 200.0))
