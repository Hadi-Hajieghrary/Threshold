"""Plan v3 WP2: the interventional integrator on a synthetic steady tow (Drake-free)."""

from __future__ import annotations

import numpy as np
import pytest

from tether.campaign.v3 import intervention as IV
from tether.physics import constants as C
from tether.theory import reduced_lti as R

T0 = 600.0


def _window(cable=2, t_up=10.0, rows=800, weather_scale=0.0):
    geometry = R.tow_geometry("parallel")
    operating = R.tow_operating_point(geometry, T0)
    state = R.steady_tow_state(geometry, operating, C.CABLE_REST_LENGTH, C.CABLE_STIFFNESS)
    # the steady tow translates at the tow speed; rows advance the positions accordingly
    n = geometry.vessel_count
    base = 3 * (n + 1)
    states = np.tile(state, (rows, 1))
    times = 9.5 + np.arange(rows) * IV.STATE_PERIOD
    speed = state[base : base + 2]
    for k in range(rows):
        states[k, 0:base:3] += speed[0] * (times[k] - times[0])
        states[k, 1:base:3] += speed[1] * (times[k] - times[0])
    weather = np.zeros((int(times[-1] / IV.STATE_PERIOD) + 5, n + 1, 2)) + weather_scale
    return IV.ParentWindow(
        cable=cable, t_up=t_up, v_return=0.0, T_peak=T0, event_id=0, x=1.0, thrusts=operating.thrusts,
        load_offsets=geometry.load_offsets, vessel_offsets=geometry.vessel_offsets, start_label=float(times[0]),
        state=states, weather=weather, weather_start=0, stiffness=C.CABLE_STIFFNESS, damping=C.CABLE_DAMPING,
    )


def test_replay_holds_a_synthetic_steady_tow():
    """With every cable active and no weather the steady tow stays taut: no onsets anywhere."""
    x = _window()
    out = IV.integrate(x, None, "recorded", horizon=2.0, window=2.0)
    assert all(v.size == 0 for v in out["onsets"]) and out["closure"] is None
    e = out["e_10ms"]
    assert np.all(e > 0.0) and np.max(np.abs(e - T0 / C.CABLE_STIFFNESS)) < 2e-4


def test_disabled_cable_applies_no_force_and_the_others_move():
    """Disabling the centre cable unloads the payload's pull on it; the branch differs from the factual one."""
    x = _window()
    factual = IV.integrate(x, None, "recorded", horizon=2.0, window=2.0)
    counter = IV.integrate(x, 2, "recorded", horizon=2.0, window=2.0)
    assert np.allclose(factual["e_10ms"][0], counter["e_10ms"][0])
    # the disabled cable's chord opens (its vessel, still thrusting, runs ahead) and is never counted as an onset
    assert counter["e_10ms"][-1, 2] > factual["e_10ms"][-1, 2] + 1e-3
    assert counter["onsets"][2].size == 0
    assert np.max(np.abs(factual["e_10ms"][-1] - counter["e_10ms"][-1])) > 1e-4


def test_branches_identical_before_t_up_and_heading_modes_valid():
    """The counterfactual removes cable j's force from t_up on: before t_up both branches are the record's."""
    x = _window(t_up=10.5)
    a = IV.integrate(x, None, "recorded", horizon=1.0, window=1.0)
    b = IV.integrate(x, 2, "recorded", horizon=1.0, window=1.0)
    before = a["t_10ms"] < 10.5
    assert np.array_equal(a["t_10ms"], b["t_10ms"]) and a["t_10ms"][0] == x.start_label
    assert before.sum() >= 50
    assert np.array_equal(a["e_10ms"][before], b["e_10ms"][before])  # bit-identical before t_up (taut cable here)
    after = a["t_10ms"] > 10.5 + 0.1
    assert np.max(np.abs(a["e_10ms"][after] - b["e_10ms"][after])) > 1e-4
    with pytest.raises(ValueError):
        IV.integrate(x, None, "sideways", horizon=0.5, window=0.5)
    held = IV.integrate(x, None, "held", horizon=1.0, window=1.0)
    assert held["closure"] is None


def test_causal_offspring_counting_excludes_the_parent():
    factual = {"onsets": [np.array([1.0, 2.0]), np.array([1.5]), np.array([]), np.array([3.0]), np.array([])]}
    counter = {"onsets": [np.array([1.0]), np.array([1.5]), np.array([9.0]), np.array([]), np.array([])]}
    out = IV.causal_offspring(factual, counter, parent_cable=2, window=3.0)
    assert out.tolist() == [1, 0, 0, 1, 0]


def test_select_parents_rule_and_cap():
    events = {"T_peak": np.array([300.0, 5000.0, 1500.0, 2000.0, 700.0, 6000.0])}
    spec = IV.InterventionSpec(max_parents=3)
    chosen = IV.select_parents(events, T0, spec)
    assert set(chosen.tolist()) == {1, 5, 3}  # x = 8.3, 10, 3.3 exceed 2.3; the cap keeps the three largest
    wide = IV.select_parents(events, T0, IV.InterventionSpec(max_parents=10))
    assert set(wide.tolist()) == set(range(6))


def test_validation_thresholds_and_verdict():
    rows = [{"cable": 0, "t_up": 1.0, "v_return": 0.5, "T_peak": 1000.0, "onset_set_reproduced": True,
             "factual_parent": {"outcome": "reengaged", "t_cf": 1.001, "v_cf": 0.505, "T_peak_cf": 1005.0}}]
    out = IV.validate(rows)
    assert out["passed"] and out["thresholds"]["dt_max"] == 0.005 and out["onset_set_reproduced_share"] == 1.0
    rows[0]["factual_parent"]["t_cf"] = 1.02
    assert not IV.validate(rows)["passed"]
    rows[0]["factual_parent"] = {"outcome": "horizon", "t_cf": None, "v_cf": None, "T_peak_cf": None}
    out = IV.validate(rows)
    assert not out["passed"] and out["not_reengaged"]


def test_match_tolerance():
    assert IV._match(np.array([1.0, 2.0]), np.array([2.01, 1.0]), 0.02)
    assert not IV._match(np.array([1.0]), np.array([1.0, 2.0]), 0.02)
    assert not IV._match(np.array([1.0]), np.array([1.05]), 0.02)
