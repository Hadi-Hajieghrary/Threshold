"""Plan v3 WP1/WP2: the in-worker reducer on hand-built logs (Drake-free)."""

from __future__ import annotations

import math

import numpy as np

from tether.campaign.v2 import events as ev
from tether.campaign.v2 import regime as rg
from tether.campaign.v3 import reducer as RD
from tether.physics import constants as C
from tether.theory import reduced_lti as R
from tether.theory import transmission as TR

K, CD, T0 = C.CABLE_STIFFNESS, C.CABLE_DAMPING, 600.0


def _steady_geometry():
    geometry = R.tow_geometry("parallel")
    operating = R.tow_operating_point(geometry, T0)
    state = R.steady_tow_state(geometry, operating, C.CABLE_REST_LENGTH, K)
    return geometry, state


def _record(time, elongation, rate, alive, geometry, state, sim_end):
    rows = 3
    state_time = np.linspace(0.0, sim_end, rows)
    return {
        "event_time": time, "elongation": elongation, "rate": rate, "alive": alive,
        "state_time": state_time, "state": np.tile(state, (rows, 1)),
        "load_offsets": geometry.load_offsets, "vessel_offsets": geometry.vessel_offsets,
        "sim_end": sim_end, "closure": None, "stiffness": K, "damping": CD,
        "all_marks": {"t_up": np.zeros(0), "cable": np.zeros(0, dtype=np.int64), "v_return": np.zeros(0)},
    }


def test_tension_series_is_the_plants_law():
    e = np.array([[0.01, -0.01, 0.0, 0.02, 0.02]])
    r = np.array([[0.0, 0.0, 0.0, -5.0, 1.0]])
    alive = np.array([[True, True, True, True, False]])
    q = RD.tension_series(e, r, alive, K, CD)
    assert q[0, 0] == K * 0.01 and q[0, 1] == 0.0 and q[0, 2] == 0.0
    assert q[0, 3] == 0.0  # k e + c edot < 0 is clamped
    assert q[0, 4] == 0.0  # dead cable


def test_transmission_rows_measure_a_hand_built_pulse():
    """Cable 2 re-engages at t = 5 s; cable 0's tension dips by 120 N inside the engagement period."""
    geometry, state = _steady_geometry()
    n, dt = 5, 1e-3
    time = np.arange(0.0, 10.0, dt)
    e = np.full((time.size, n), T0 / K)  # every cable taut at the pretension
    rate = np.zeros((time.size, n))
    alive = np.ones((time.size, n), dtype=bool)
    # cable 2 slack from 3 s, back at 5 s (a mark with T_peak 1800 N = 3 T0)
    e[(time >= 3.0) & (time < 5.0), 2] = -0.05
    e[(time >= 5.0) & (time < 5.05), 2] = 1800.0 / K
    # cable 0 dips by 120 N 0.1 s after the re-engagement, and rises by 40 N at 0.25 s
    dip = (time >= 5.1) & (time < 5.15)
    e[dip, 0] = (T0 - 120.0) / K
    e[(time >= 5.25) & (time < 5.30), 0] = (T0 + 40.0) / K
    # cable 4 goes slack 1 s later (an onset within 3 s)
    e[(time >= 6.0) & (time < 6.5), 4] = -0.02
    marks = {"t_up": np.array([5.0]), "T_peak": np.array([1800.0]), "cable": np.array([2]), "t_down": np.array([3.0])}
    dec = ev.decluster_marks(marks["t_up"], marks["cable"], rule="anchored")
    record = _record(time, e, rate, alive, geometry, state, 10.0)
    interpolant = rg.StateInterpolant.from_log(record["state_time"], record["state"], geometry.load_offsets, geometry.vessel_offsets)
    period = TR.frequencies(K, CD, n)["engagement_period"]
    rows = RD.transmission_rows(record, marks, dec, interpolant, T0, K, CD, period)
    assert rows["parent_mark"].size == 4 and set(rows["neighbour_i"].tolist()) == {0, 1, 3, 4}
    row0 = {k: v[rows["neighbour_i"] == 0][0] for k, v in rows.items()}
    assert abs(row0["T_i_at_t_up"] - T0) < 1e-6 and abs(row0["drop"] - 120.0) < 1e-6 and abs(row0["rise"] - 40.0) < 1e-6
    assert abs(row0["lag_of_min"] - 0.1) < 2e-3 and abs(row0["x"] - 3.0) < 1e-12
    assert abs(row0["drop_over_Tpeak"] - 120.0 / 1800.0) < 1e-9
    # parallel formation at the design state: sigma_i = sigma_j, cos = 1, geometry factor = T_02 / 0.44 = 1 (centre column)
    assert abs(row0["cos_dsigma"] - 1.0) < 1e-6 and abs(row0["geometry_factor_ij"] - 1.0) < 1e-6
    assert abs(row0["drop_norm"] - 120.0 / 1800.0) < 1e-9 and row0["eligible_i"] and row0["all_neighbours_taut"]
    assert not row0["onset_within_3s"]
    row4 = {k: v[rows["neighbour_i"] == 4][0] for k, v in rows.items()}
    assert row4["onset_within_3s"] and abs(row4["onset_lag"] - 1.0) < 2e-3 and row4["drop"] == 0.0


def test_window_is_the_cells_engagement_period_and_eligibility_at_t_up():
    geometry, state = _steady_geometry()
    n, dt = 5, 1e-3
    time = np.arange(0.0, 8.0, dt)
    e = np.full((time.size, n), T0 / K)
    rate = np.zeros((time.size, n))
    alive = np.ones((time.size, n), dtype=bool)
    e[(time >= 2.0) & (time < 4.0), 1] = -0.05  # the mark on cable 1
    e[(time >= 3.5) & (time < 4.5), 3] = -0.01  # cable 3 is slack at t_up: not eligible
    # a dip on cable 0 just outside a 0.2 s window but inside 0.335 s
    e[(time >= 4.25) & (time < 4.30), 0] = (T0 - 200.0) / K
    marks = {"t_up": np.array([4.0]), "T_peak": np.array([1200.0]), "cable": np.array([1]), "t_down": np.array([2.0])}
    dec = ev.decluster_marks(marks["t_up"], marks["cable"], rule="anchored")
    record = _record(time, e, rate, alive, geometry, state, 8.0)
    interpolant = rg.StateInterpolant.from_log(record["state_time"], record["state"], geometry.load_offsets, geometry.vessel_offsets)
    long = RD.transmission_rows(record, marks, dec, interpolant, T0, K, CD, 0.335)
    short = RD.transmission_rows(record, marks, dec, interpolant, T0, K, CD, 0.2)
    drop_long = long["drop"][long["neighbour_i"] == 0][0]
    drop_short = short["drop"][short["neighbour_i"] == 0][0]
    assert abs(drop_long - 200.0) < 1e-6 and drop_short == 0.0
    assert not long["eligible_i"][long["neighbour_i"] == 3][0] and not long["all_neighbours_taut"][0]


def test_offspring_table_attributes_cross_cable_onsets_to_the_latest_reengagement():
    geometry, state = _steady_geometry()
    n, dt = 5, 1e-3
    time = np.arange(0.0, 30.0, dt)
    e = np.full((time.size, n), T0 / K)
    rate = np.zeros((time.size, n))
    alive = np.ones((time.size, n), dtype=bool)
    # cable 2: slack 8-10 s (mark A); cable 0: slack 11-12 s (offspring of A, cross); cable 2: slack 10.5-10.8 (bounce, same event)
    e[(time >= 8.0) & (time < 10.0), 2] = -0.05
    e[(time >= 10.5) & (time < 10.8), 2] = -0.02
    e[(time >= 11.0) & (time < 12.0), 0] = -0.03
    # cable 4: slack 20-21 s with no re-engagement in the 3 s before: primary
    e[(time >= 20.0) & (time < 21.0), 4] = -0.03
    marks = {
        "t_up": np.array([10.0, 10.8, 12.0, 21.0]), "T_peak": np.array([1500.0, 700.0, 900.0, 800.0]),
        "cable": np.array([2, 2, 0, 4]), "t_down": np.array([8.0, 10.5, 11.0, 20.0]),
    }
    dec = ev.decluster_marks(marks["t_up"], marks["cable"], rule="anchored")
    assert dec.event_count == 3  # the bounce joins cable 2's event
    record = _record(time, e, rate, alive, geometry, state, 30.0)
    table = RD.offspring_table(record, marks, dec, warmup=0.0, end=30.0)
    RD._events_primary_flag(table, marks, dec)
    events, onsets = table["events"], table["onsets"]
    a = int(np.flatnonzero(events["cable_j"] == 2)[0])
    assert events["n_cross"][a] == 1 and events["offspring_per_cable"][a].tolist() == [1, 0, 0, 0, 0]
    assert abs(events["first_lag"][a] - 0.2) < 2e-3  # 11.0 - 10.8: the bounce is the latest re-engagement
    assert events["T_peak"][a] == 1500.0 and events["cluster_max_T_peak"][a] == 1500.0 and events["n_marks"][a] == 2
    assert events["primary"][a] and events["primary"][int(np.flatnonzero(events["cable_j"] == 4)[0])]
    assert not events["primary"][int(np.flatnonzero(events["cable_j"] == 0)[0])]
    assert onsets["cable"].size == 4 and int(onsets["cross"].sum()) == 1 and int(onsets["primary"].sum()) == 2


def test_margin_samples_use_clean_taut_samples_only():
    geometry, state = _steady_geometry()
    n, dt = 5, 1e-3
    time = np.arange(0.0, 20.0, dt)
    e = np.full((time.size, n), T0 / K)
    e[:, 1] = 2.0 * T0 / K  # cable 1 carries 1200 N
    rate = np.zeros((time.size, n))
    alive = np.ones((time.size, n), dtype=bool)
    e[(time >= 5.0) & (time < 6.0), 0] = -0.02
    marks = {"t_up": np.array([6.0]), "T_peak": np.array([1000.0]), "cable": np.array([0]), "t_down": np.array([5.0])}
    out = RD.margin_samples({"event_time": time, "elongation": e, "rate": rate, "alive": alive}, marks, 0.0, 20.0, K, CD)
    # every cable: 20 decimated samples minus the (6, 9] exclusion window (3) minus, for cable 0, its slack second
    assert out["counts"][1] == 17 and out["counts"][0] == 16
    assert abs(out["quantiles"][1, 50] - 1200.0) < 1e-6 and abs(out["quantiles"][0, 50] - 600.0) < 1e-6


def test_first_index_at_or_after_handles_rounding():
    time = np.arange(0.0, 1.0, 1e-3)
    assert RD._first_index_at_or_after(time, 0.5) == 500
    assert RD._first_index_at_or_after(time, 0.5 + 1e-12) == 500
    assert RD._first_index_at_or_after(time, 0.5004) == 501


def test_offspring_table_handles_zero_marks_and_same_cable_parents():
    geometry, state = _steady_geometry()
    n, dt = 5, 1e-3
    time = np.arange(0.0, 20.0, dt)
    e = np.full((time.size, n), T0 / K)
    rate = np.zeros((time.size, n))
    alive = np.ones((time.size, n), dtype=bool)
    empty = {"t_up": np.zeros(0), "T_peak": np.zeros(0), "cable": np.zeros(0, dtype=np.int64), "t_down": np.zeros(0)}
    dec = ev.decluster_marks(empty["t_up"], empty["cable"], rule="anchored")
    record = _record(time, e, rate, alive, geometry, state, 20.0)
    table = RD.offspring_table(record, empty, dec, warmup=0.0, end=20.0)
    RD._events_primary_flag(table, empty, dec)
    assert table["events"]["cable_j"].size == 0 and table["events"]["parent_event"].size == 0
    # zero marks but a slack onset inside the window (stage A crashed here: FAN_T600_I1.0 / k x 0.25 runs, addendum 1)
    e_on = e.copy()
    e_on[(time >= 9.0) & (time < 9.4), 3] = -0.01
    table = RD.offspring_table(_record(time, e_on, rate, alive, geometry, state, 20.0), empty, dec, warmup=0.0, end=20.0)
    assert table["onsets"]["cable"].tolist() == [3] and table["onsets"]["primary"].all()
    assert table["onsets"]["parent_event"].tolist() == [-1] and table["onsets"]["parent_cable"].tolist() == [-1]
    interpolant = rg.StateInterpolant.from_log(record["state_time"], record["state"], geometry.load_offsets, geometry.vessel_offsets)
    rows = RD.transmission_rows(record, empty, dec, interpolant, T0, K, CD, 0.335)
    assert rows["parent_mark"].size == 0
    # cable 1 re-engages at 5 s, then goes slack again at 7.5 s (2.5 s later: outside the 2 s burst, inside the 3 s parent window)
    e[(time >= 3.0) & (time < 5.0), 1] = -0.03
    e[(time >= 7.5) & (time < 8.0), 1] = -0.02
    marks = {"t_up": np.array([5.0, 8.0]), "T_peak": np.array([900.0, 700.0]), "cable": np.array([1, 1]), "t_down": np.array([3.0, 7.5])}
    dec = ev.decluster_marks(marks["t_up"], marks["cable"], rule="anchored")
    assert dec.event_count == 2
    record = _record(time, e, rate, alive, geometry, state, 20.0)
    table = RD.offspring_table(record, marks, dec, warmup=0.0, end=20.0)
    RD._events_primary_flag(table, marks, dec)
    ev_t = table["events"]
    second = int(np.argmax(ev_t["t_up"]))
    assert ev_t["primary"][second] and ev_t["same_cable_parent"][second] and ev_t["parent_event"][second] == -1
    assert ev_t["primary"].all() and not ev_t["same_cable_parent"][1 - second]
