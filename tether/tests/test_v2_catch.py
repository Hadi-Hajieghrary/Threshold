"""Plan v2 P6-T0: the catch law (tether.control.catch) and the counterfactual integrator
(tether.campaign.v2.p6_t0) on synthetic excursions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from tether.campaign.v2 import p6_t0
from tether.control.catch import catch_engaged, catch_thrust, landing_profile
from tether.physics import constants
from tether.physics.fleet import cable_kinematics, equilibrium_state, formation_geometry, operating_point

V_B = 0.45377161
F_T = 1294.63


# ----------------------------------------------------------------------------- the law


def test_landing_profile_values():
    assert landing_profile(0.0, V_B) == pytest.approx(0.3 * V_B)
    assert landing_profile(-1.13, V_B) == pytest.approx(math.sqrt((0.3 * V_B) ** 2 + 2 * 0.2 * 1.13))
    # e_hat >= 0: the profile is v_soft
    assert landing_profile(0.2, V_B) == pytest.approx(0.3 * V_B)
    np.testing.assert_allclose(landing_profile(np.array([-2.5, 0.0]), V_B), [math.sqrt((0.3 * V_B) ** 2 + 1.0), 0.3 * V_B])


def test_catch_released_when_taut_below_profile_or_unknown():
    assert catch_thrust(-1.0, 5.0, False, F_T, V_B, 0.0) == F_T  # taut: released
    assert catch_thrust(-1.0, 0.1, True, F_T, V_B, 0.0) == F_T  # below the profile
    assert catch_thrust(np.nan, 5.0, True, F_T, V_B, -1.0) == F_T  # no estimate
    assert catch_thrust(-1.0, 5.0, True, F_T, V_B, -1.0, easing=0.5) == -F_T
    assert not catch_engaged(-1.0, 0.1, True, V_B)
    assert catch_engaged(-1.0, 1.0, True, V_B)


def test_catch_proportional_band_and_clip():
    e, ref = -0.5, landing_profile(-0.5, V_B)
    small = ref + 0.1  # K_v * 0.1 = 200 N below F_T
    assert catch_thrust(e, small, True, F_T, V_B, 0.0) == pytest.approx(F_T - 200.0)
    big = ref + 3.0  # 6000 N below F_T: clipped
    assert catch_thrust(e, big, True, F_T, V_B, 0.0) == 0.0
    assert catch_thrust(e, big, True, F_T, V_B, -1.0) == -F_T
    # reverse and hold agree inside the band
    assert catch_thrust(e, small, True, F_T, V_B, -1.0) == pytest.approx(F_T - 200.0)
    out = catch_thrust(np.array([e, e]), np.array([small, big]), np.array([True, True]), F_T, V_B, -1.0)
    np.testing.assert_allclose(out, [F_T - 200.0, -F_T])


def test_sign_convention_matches_cable_kinematics():
    """A vessel moving away from the load has edot > 0 (the closing slack cable's re-engagement sign)."""
    geometry = formation_geometry("fan", arc_half_angle=0.55)
    operating = operating_point(geometry, 1000.0)
    state = equilibrium_state(geometry, operating)
    base = 3 * (geometry.vessel_count + 1)
    state[base:] = 0.0
    angle = geometry.cable_angles[2]
    state[base + 9 : base + 11] = [math.cos(angle), math.sin(angle)]  # vessel 2 moves outward
    rate = cable_kinematics(state, geometry, constants.CABLE_REST_LENGTH)["rate"]
    assert rate[2] == pytest.approx(1.0)
    assert np.allclose(np.delete(rate, 2), 0.0)


# ----------------------------------------------------------------------------- events and statistics


def _mark(cable, t_up, peak, dwell=3.0, v=1.0):
    return (cable, t_up, 0.5, v, peak, dwell)


def test_declustered_events_anchor_and_dangerous_mark():
    marks = [_mark(1, 10.0, 3000.0), _mark(1, 10.3, 6000.0, dwell=0.2), _mark(1, 11.9, 9000.0, dwell=0.1), _mark(1, 12.1, 5000.0), _mark(2, 10.1, 4600.0)]
    events = p6_t0.declustered_events(marks)
    assert [(e["cable"], e["t_event"], e["marks"]) for e in events] == [(1, 10.0, 3), (2, 10.1, 1), (1, 12.1, 1)]
    first = events[0]
    assert first["dangerous"] and first["dangerous_index"] == 1 and first["t_up"] == 10.3 and first["cluster_max"] == 9000.0
    assert all(e["dangerous"] for e in events)
    assert not p6_t0.declustered_events([_mark(0, 5.0, 4500.0)])[0]["dangerous"]  # strict threshold


def test_severance_type_and_clopper_pearson():
    marks = [(0, 10.0), (1, 20.0)]
    assert p6_t0.severance_type(None, marks)[0] == "none"
    assert p6_t0.severance_type((10.03, 0), marks)[0] == "snap"
    assert p6_t0.severance_type((10.08, 0), marks)[0] == "taut_overload"
    assert p6_t0.severance_type((10.08, 0), marks, 0.335, before_only=True)[0] == "snap"
    assert p6_t0.severance_type((10.03, 1), marks)[0] == "taut_overload"
    low, high = p6_t0.clopper_pearson(16, 20)
    assert low == pytest.approx(0.5634, abs=1e-4) and high == pytest.approx(0.9427, abs=1e-4)
    assert p6_t0.clopper_pearson(20, 20)[1] == 1.0 and p6_t0.clopper_pearson(0, 20)[0] == 0.0


def test_first_crossing():
    t = np.arange(5) * 1e-3
    e = np.zeros((5, 2))
    rate = np.zeros((5, 2))
    e[3, 1] = 0.03  # 5100 N
    assert p6_t0.first_crossing(t, e, rate) == (0.003, 1)
    assert p6_t0.first_crossing(t, np.zeros((5, 2)), rate) is None


# ----------------------------------------------------------------------------- synthetic excursions


class _Ones:
    def __call__(self, time):
        return np.ones_like(np.asarray(time, dtype=float))


def _excursion(depth=1.0, slack_cable=2, weather=0.0, duration=12.0, estimates=None, closing=0.0):
    """Steady fan tow at constant velocity with cable ``slack_cable`` pulled ``depth`` metres slack,
    its vessel moving with the fleet; the load rows follow the steady tow (exact for 'literal')."""
    geometry = formation_geometry("fan", arc_half_angle=0.55)
    operating = operating_point(geometry, 1000.0)
    start = equilibrium_state(geometry, operating)
    n = geometry.vessel_count
    base = 3 * (n + 1)
    angle = geometry.cable_angles[slack_cable]
    j = 3 * (slack_cable + 1)
    extra = depth + operating.tensions[slack_cable] / constants.CABLE_STIFFNESS
    start[j : j + 2] -= extra * np.array([math.cos(angle), math.sin(angle)])
    rows = int(round(duration / p6_t0.STATE_PERIOD)) + 3
    t0 = 40.0
    times = t0 + np.arange(rows) * p6_t0.STATE_PERIOD
    state = np.repeat(start[None], rows, axis=0)
    velocity = start[base : base + 2].copy()
    for body in range(n + 1):
        state[:, 3 * body : 3 * body + 2] += (times - t0)[:, None] * velocity[None]
    # the slack vessel's initial closing speed along its chord (row 0 is the start state)
    state[0, base + j : base + j + 2] += closing * np.array([math.cos(angle), math.sin(angle)])
    forcing = np.zeros((rows + 8, n + 1, 2))
    forcing[:, :, 0] = weather
    log_start = int(round(t0 / 1e-3)) - 400
    kin = cable_kinematics(start, geometry, constants.CABLE_REST_LENGTH)
    x = p6_t0.Excursion(
        seed=0,
        cable=slack_cable,
        t_up=t0 + 1.0e3,  # never 'recorded' re-engaged inside the test window
        v_return=0.0,
        T_peak=0.0,
        dwell=5.0,
        depth=depth,
        critical_speed=V_B,
        thrusts=operating.thrusts.copy(),
        load_offsets=np.asarray(geometry.load_offsets, float),
        vessel_offsets=np.asarray(geometry.vessel_offsets, float),
        start_label=t0,
        state=state,
        weather=forcing,
        weather_start=int(round(t0 / p6_t0.STATE_PERIOD)),
        log_start=log_start,
        e_log=np.full(20000, kin["elongation"][slack_cable]),
        edot_log=np.zeros(20000),
        estimates=estimates or {},
        envelope=_Ones(),
    )
    return x, operating, geometry


def _analytic_crossing(thrust, depth):
    c, m = constants.VESSEL_LINEAR_DRAG, constants.VESSEL_MASS
    v_inf, tau = thrust / c, m / c
    distance = lambda t: v_inf * t - v_inf * tau * (1.0 - math.exp(-t / tau)) - depth
    t = brentq(distance, 1e-6, 100.0)
    return t, v_inf * (1.0 - math.exp(-t / tau))


def test_literal_replay_matches_analytic_free_flight():
    """Load on a constant-velocity track, vessel starting with the fleet's velocity: in the moving
    frame the vessel's relative motion is m dv/dt = F - c v - c V... solved here in the ground frame."""
    x, operating, _ = _excursion(depth=1.0)
    # relative to the load (which moves at V with its own drag balanced by the cables), the free
    # vessel feels F_T - c (V + u): at t = 0, u = 0 and F_T - c V = T0 (the cable it lost)
    out = p6_t0.integrate(x, "literal", "replay", None)
    assert out["outcome"] == "reengaged"
    c, m = constants.VESSEL_LINEAR_DRAG, constants.VESSEL_MASS
    surplus = operating.thrusts[2] - c * operating.speed  # along the chord (hull aligned in the centre cable)
    t, v = _analytic_crossing(surplus, 1.0)
    assert out["v_cf"] == pytest.approx(v, abs=2e-3)
    assert out["t_cf"] - x.start_label == pytest.approx(t, abs=2e-3)


def test_fleet_mode_holds_the_steady_tow():
    x, operating, geometry = _excursion(depth=-1000.0 / constants.CABLE_STIFFNESS)  # no slack: exact equilibrium
    x.t_up = x.start_label + 1.0e3
    # run the fleet a little with the replay and check the tensions of the untouched state stay at T0
    x.state = x.state[:303]
    out = p6_t0.integrate(x, "fleet", "replay", None, horizon=3.0, trace=True)
    assert out["outcome"] == "horizon"
    final = out["final"]
    start = x.state[0]
    base = 3 * (geometry.vessel_count + 1)
    elapsed = final["label"] - x.start_label
    assert elapsed == pytest.approx(3.0)
    expected = start[:base].reshape(-1, 3)[:, :2] + elapsed * start[base : base + 2][None]
    np.testing.assert_allclose(final["position"], expected, atol=1e-6)
    np.testing.assert_allclose(final["velocity"], np.repeat(start[base : base + 2][None], geometry.vessel_count + 1, axis=0), atol=1e-8)
    assert abs(final["load_omega"]) < 1e-9


def test_law_brakes_the_return_and_orders_the_variants():
    x, _, _ = _excursion(depth=1.5)
    replay = p6_t0.integrate(x, "fleet", "replay", None)
    hold = p6_t0.integrate(x, "fleet", "true10", 0.0)
    reverse = p6_t0.integrate(x, "fleet", "true10", -1.0)
    fast = p6_t0.integrate(x, "fleet", "true50", 0.0)
    assert replay["outcome"] == hold["outcome"] == reverse["outcome"] == fast["outcome"] == "reengaged"
    assert replay["v_cf"] > 1.5 and replay["engaged_fraction"] == 0.0
    assert hold["v_cf"] < 0.5 * replay["v_cf"] and hold["delay"] > replay["delay"] and hold["engaged_fraction"] > 0.3
    assert fast["v_cf"] == pytest.approx(hold["v_cf"], abs=0.05)
    # unsaturated (thrust never reaches F_min = 0): hold and reverse coincide, and the proportional
    # law settles above the profile by about (T0 + m a_c - c u) / K_v, so it lands near 0.6-0.7 m/s
    assert hold["min_thrust"] > 0.0 and reverse["v_cf"] == pytest.approx(hold["v_cf"], abs=1e-12)
    assert 0.3 * V_B + 0.3 < hold["v_cf"] < 0.3 * V_B + 0.7
    assert replay["T_peak_cf"] > hold["T_peak_cf"] > 1000.0
    literal = p6_t0.integrate(x, "literal", "true10", 0.0)
    assert literal["outcome"] == "reengaged" and literal["v_cf"] < 0.5 * replay["v_cf"]


def test_saturated_law_reverse_beats_hold():
    x, _, _ = _excursion(depth=1.5, closing=2.5)
    replay = p6_t0.integrate(x, "literal", "replay", None)
    hold = p6_t0.integrate(x, "literal", "true50", 0.0)
    reverse = p6_t0.integrate(x, "literal", "true50", -1.0)
    assert hold["min_thrust"] == 0.0 and reverse["min_thrust"] < 0.0
    assert reverse["v_cf"] < hold["v_cf"] < replay["v_cf"]


def _estimates(e_err, edot_err, rows=400, tick0=395):
    return {"tick0": tick0, "slack": np.ones(rows, bool), "e_err": np.full(rows, e_err), "edot_err": np.full(rows, edot_err), "held": (e_err, edot_err)}


def test_arm_path_with_zero_error_equals_true10_and_is_monotone_in_the_error():
    zero = {"P": _estimates(0.0, 0.0), "B2": _estimates(0.0, 0.5), "L": _estimates(0.0, -0.5)}
    x, _, _ = _excursion(depth=1.5, estimates=zero)
    truth = p6_t0.integrate(x, "fleet", "true10", 0.0)
    p = p6_t0.integrate(x, "fleet", "P", 0.0)
    over = p6_t0.integrate(x, "fleet", "B2", 0.0)  # estimator over-reads the closing speed: brakes harder
    under = p6_t0.integrate(x, "fleet", "L", 0.0)  # under-reads: brakes less
    assert p["v_cf"] == pytest.approx(truth["v_cf"], abs=1e-12)
    assert over["v_cf"] < truth["v_cf"] < under["v_cf"]
    # Prop. 14: an under-read of 0.5 m/s lands near 0.5 m/s above the profile's v_soft
    assert under["v_cf"] > V_B


def test_unflagged_estimates_never_engage():
    silent = {"P": {"tick0": 395, "slack": np.zeros(400, bool), "e_err": np.zeros(400), "edot_err": np.zeros(400), "held": None}}
    x, _, _ = _excursion(depth=1.5, estimates=silent)
    replay = p6_t0.integrate(x, "fleet", "replay", None)
    p = p6_t0.integrate(x, "fleet", "P", 0.0)
    assert p["engaged_fraction"] == 0.0
    assert p["v_cf"] == pytest.approx(replay["v_cf"], abs=1e-3)


def test_closure_is_detected():
    # a strong push toward the load (weather along -x) with the reverse law: the vessel runs into the load
    x, _, _ = _excursion(depth=3.0, weather=-6000.0)
    out = p6_t0.integrate(x, "literal", "true10", -1.0, horizon=10.0)
    assert out["outcome"] in ("closure", "horizon")
    assert not out["caught"]


def test_declarations_are_json_and_name_every_choice():
    import json

    payload = json.loads(p6_t0.json_bytes(p6_t0.DECLARATIONS))
    for key in ("law", "sign_convention", "event_rule", "dangerous", "populations", "turnaround", "heading", "weather", "modes", "controllers", "admissibility_rule", "validation", "taut_overload"):
        assert key in payload


def test_post_contact_rows_do_not_leak_into_the_crossing():
    """Addenda 1-2: rows after the recorded t_up carry the recorded snap; the replay's crossing
    speed must not depend on them (vessel heading, and the literal mode's load track)."""
    for mode in p6_t0.MODES:
        x, _, _ = _excursion(depth=1.0)
        clean = p6_t0.integrate(x, mode, "replay", None)
        x.t_up = clean["t_cf"]  # the 'recorded' contact is this mode's own crossing
        base = 3 * (x.thrusts.size + 1)
        k = int(math.floor((x.t_up - x.start_label) / p6_t0.STATE_PERIOD))
        j = 3 * (x.cable + 1)
        kicked = x.state.copy()
        kicked[k + 1 :, j + 2] += 0.3  # heading kick after contact
        kicked[k + 1 :, base + j + 2] += 5.0  # yaw rate kick
        kicked[k + 1 :, base : base + 2] += 0.4  # load velocity kick
        x.state = kicked
        again = p6_t0.integrate(x, mode, "replay", None)
        assert again["v_cf"] == pytest.approx(clean["v_cf"], abs=1e-9)
        assert again["t_cf"] == pytest.approx(clean["t_cf"], abs=1e-12)
