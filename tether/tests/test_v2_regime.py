"""Unit tests for the v2 (H4') regime machinery and the v2 summariser (plan v2 II.1, II.2, B.2, B.4)."""

import math

import numpy as np
import pytest

from tether.campaign.v2 import regime as rg
from tether.physics import constants


def test_declared_constants():
    assert rg.TAU_LF == pytest.approx(0.710, abs=1.0e-3)
    assert rg.R2_DURATION == pytest.approx(3.43, abs=1.0e-2)
    assert rg.C_EFF == pytest.approx(329.0, rel=0.01)
    assert rg.M_EFF == pytest.approx(483.9, abs=0.1)


def test_conjugate_loads_weights_and_sign():
    d = np.array([[[1.0, 0.0]]])
    load_push = np.zeros((1, 2, 2))
    load_push[0, 0] = [1.0, 0.0]  # load pushed toward the vessel: closes the gap
    w_rel, w_c = rg.conjugate_loads(load_push, d)
    assert w_rel[0, 0] == pytest.approx(0.19, abs=0.01) and w_c[0, 0] == pytest.approx(0.06, abs=0.01)
    vessel_push = np.zeros((1, 2, 2))
    vessel_push[0, 1] = [-1.0, 0.0]  # vessel pushed back toward the load: closes the gap
    w_rel, w_c = rg.conjugate_loads(vessel_push, d)
    assert w_rel[0, 0] == pytest.approx(0.81, abs=0.01) and w_c[0, 0] == pytest.approx(0.94, abs=0.01)
    lateral = np.zeros((1, 2, 2))
    lateral[0, :, 1] = 500.0
    assert np.allclose(rg.conjugate_loads(lateral, d), 0.0)


def test_relative_load_matches_the_plant_formula():
    from tether.physics.fleet import relative_gust_load

    rng = np.random.default_rng(1)
    weather = rng.normal(0.0, 1000.0, (6, 2))
    direction = rg.unit(rng.normal(size=(5, 2)))
    w_rel, _ = rg.conjugate_loads(weather[None], direction[None])
    assert np.array_equal(w_rel[0], relative_gust_load(weather, direction))


def test_weather_index_is_the_plants_zero_order_hold():
    times = np.array([0.0, 0.009, 0.00999999999, 0.01, 0.0199, 5.0])
    assert list(rg.weather_index(times, 1000)) == [0, 0, 1, 1, 1, 500]
    assert rg.weather_index(np.array([100.0]), 10)[0] == 9


def test_classify_boundaries():
    t0 = 1000.0
    v_t = t0 / constants.VESSEL_LINEAR_DRAG
    assert rg.classify(0.0, float("nan"), t0, False) == "N"
    # R1: short and slow; v_s = (W - T0) t_x / m_A
    assert rg.classify(0.5, t0 + 0.29 * v_t * constants.VESSEL_MASS / 0.5, t0, True) == "R1"
    assert rg.classify(0.5, t0 + 0.31 * v_t * constants.VESSEL_MASS / 0.5, t0, True) == "transitional"
    assert rg.classify(0.72, t0 + 1.0, t0, True) == "transitional"
    assert rg.classify(3.4, t0 + 1.0, t0, True) == "transitional"
    assert rg.classify(3.5, t0 + 5000.0, t0, True) == "R2"


def test_episode_merge_is_strictly_below_one_second():
    values = np.zeros(10_000)
    values[100:200] = 2.0
    values[1199:1300] = 2.0  # gap of 999 samples: merged
    values[2300:2400] = 2.0  # gap of 1000 samples: not merged
    starts, stops = rg.exceedance_episodes(values, 1.0, 1000)
    assert list(starts) == [100, 2300] and list(stops) == [1300, 2400]
    assert rg.exceedance_episodes(np.zeros(5), 1.0, 1000)[0].size == 0


def _synthetic():
    """One cable, 1 ms grid, three slack intervals with known W^c histories."""
    dt = 1.0e-3
    n = 20_001
    time = np.arange(n) * dt
    e = np.full((n, 1), 0.01)
    wc = np.zeros((n, 1))
    t0 = 1000.0
    # mark 0: slack [5001, 5501), deepest at 5200; episode 4000-4800 and 5100-5300 merged (gap 300)
    e[5001:5501, 0] = -0.02
    e[5200, 0] = -0.05
    wc[4000:4800, 0] = 1500.0
    wc[5100:5300, 0] = 1500.0
    # mark 1: slack [8001, 8101), no exceedance (class N)
    e[8001:8101, 0] = -0.01
    wc[8001:8101, 0] = 900.0
    # mark 2: slack [12001, 12301), inside a 5 s episode (class R2); a later episode
    # (1.1 s after the first ends, so not merged) does not touch the slack interval
    e[12001:12301, 0] = -0.03
    wc[10000:15000, 0] = 1200.0
    wc[16100:16200, 0] = 1300.0
    downs = np.array([5001, 8001, 12001])
    ups = np.array([5501, 8101, 12301])
    down_time = time[downs - 1] + (-e[downs - 1, 0] / (e[downs, 0] - e[downs - 1, 0])) * dt
    up_time = time[ups - 1] + (-e[ups - 1, 0] / (e[ups, 0] - e[ups - 1, 0])) * dt
    marks = {
        "t_up": up_time,
        "cable": np.zeros(3, dtype=np.int64),
        "dwell": up_time - down_time,
        "depth": np.array([0.05, 0.01, 0.03]),
        "v_return": np.array([0.5, 0.1, 0.3]),
    }
    states = np.zeros((n // 10 + 1, 12))
    states[:, 3] = 14.0  # vessel 1 m beyond rest length + attachments
    interpolant = rg.StateInterpolant.from_log(time[::10], states, np.array([[2.0, 0.0]]), np.array([[-1.5, 0.0]]))
    return time, e, wc, marks, downs, ups, down_time, interpolant, t0


def test_mark_covariates_on_known_histories():
    time, e, wc, marks, downs, ups, down_time, interpolant, t0 = _synthetic()
    out = rg.mark_covariates(time=time, elongation=e, relative_load=0.8 * wc, wc=wc, marks=marks, up_index=ups,
                             down_index=downs, down_time=down_time, interpolant=interpolant, pretension=t0)
    assert list(out["regime"]) == ["transitional", "N", "R2"]
    # mark 0: the full merged episode [4000, 5300) governs (IV.4); clipped to the slack it is R1
    assert out["t_x"][0] == pytest.approx(1.3)
    assert out["Wc_max"][0] == 1500.0
    assert out["v_s"][0] == pytest.approx(500.0 * 1.3 / 600.0)
    assert out["t_x_clip"][0] == pytest.approx(0.299)
    assert out["regime_clip"][0] == "R1"
    assert out["n_episodes"][0] == 1
    assert out["Wc_slack_max"][0] == 1500.0
    assert out["Wc_return_mean"][0] == pytest.approx(np.mean(wc[5200:5501, 0]))
    assert out["Wrel_return_mean"][0] == pytest.approx(0.8 * np.mean(wc[5200:5501, 0]))
    assert out["a_bar_ret"][0] == pytest.approx(0.5**2 / (2 * 0.05))
    # mark 1: no exceedance
    assert out["t_x"][1] == 0.0 and not out["any_exceedance"][1] and math.isnan(out["Wc_max"][1])
    # mark 2: 5 s episode, the later separate episode is not attached
    assert out["t_x"][2] == pytest.approx(5.0) and out["n_episodes"][2] == 1
    assert out["regime_clip"][2] == "R1"  # 0.3 s clipped at 200 N excess
    # chord angles: vessel dead ahead of the load
    assert np.allclose(out["sigma_up"], 0.0) and np.allclose(out["psi_up"], 0.0)
    # dwell is carried, not used: changing it leaves every class unchanged
    marks["dwell"] = marks["dwell"] * 100.0
    again = rg.mark_covariates(time=time, elongation=e, relative_load=0.8 * wc, wc=wc, marks=marks, up_index=ups,
                               down_index=downs, down_time=down_time, interpolant=interpolant, pretension=t0)
    assert list(again["regime"]) == list(out["regime"])


def test_governing_episode_is_the_one_with_most_in_slack_samples():
    time, e, wc, marks, downs, ups, down_time, interpolant, t0 = _synthetic()
    # runs 250 samples apart merge into one episode, reported with its full duration
    wc[:] = 0.0
    wc[4000:5050, 0] = 1100.0
    wc[5300:5480, 0] = 1400.0
    out = rg.mark_covariates(time=time, elongation=e, relative_load=wc, wc=wc, marks=marks, up_index=ups,
                             down_index=downs, down_time=down_time, interpolant=interpolant, pretension=t0)
    assert out["n_episodes"][0] == 1 and out["t_x"][0] == pytest.approx(1.48) and out["Wc_max"][0] == 1400.0
    # two genuinely separate episodes (gap >= 1 s) inside one long slack interval
    e[5001:7501, 0] = -0.02
    ups2 = ups.copy()
    ups2[0] = 7501
    marks["t_up"][0] = time[7500] + (0.02 / 0.03) * 1.0e-3
    wc[:] = 0.0
    wc[5100:5200, 0] = 1100.0  # 100 in-slack samples
    wc[6300:6600, 0] = 1100.0  # 300 in-slack samples, gap 1100 samples: separate, and governs
    out = rg.mark_covariates(time=time, elongation=e, relative_load=wc, wc=wc, marks=marks, up_index=ups2,
                             down_index=downs, down_time=down_time, interpolant=interpolant, pretension=t0)
    assert out["n_episodes"][0] == 2 and out["t_x"][0] == pytest.approx(0.3)
    assert out["episode_start"][0] == pytest.approx(6.3)


def test_state_interpolant_is_exact_at_rows_and_wraps_psi():
    rng = np.random.default_rng(4)
    states = rng.normal(0.0, 0.3, (5, 36))
    states[:, 3:18:3] += 14.0
    time = np.arange(5) * 0.01
    load = rng.normal(size=(5, 2))
    stern = np.tile([-1.5, 0.0], (5, 1))
    interpolant = rg.StateInterpolant.from_log(time, states, load, stern)
    chords = rg.chord_vectors(states, load, stern)
    assert np.array_equal(interpolant.direction(time[:-1]), rg.unit(chords[:-1]))
    angles = interpolant.angles(time[1:3])
    assert np.all(np.abs(angles["psi"]) <= math.pi)


# ----------------------------------------------------------------------------- Drake-backed checks


def test_chord_vectors_match_the_plant_kinematics():
    from tether.physics.fleet import cable_kinematics, fast_cable_terms, formation_geometry

    geometry = formation_geometry("parallel")
    rng = np.random.default_rng(7)
    for _ in range(5):
        state = rng.normal(0.0, 0.5, 36)
        state[3:18:3] += 14.0
        direction = rg.unit(rg.chord_vectors(state[None], geometry.load_offsets, geometry.vessel_offsets))[0]
        terms = fast_cable_terms(state, geometry, constants.CABLE_REST_LENGTH)
        assert np.allclose(direction[:, 0], terms[4], rtol=0.0, atol=1.0e-14)
        assert np.allclose(direction[:, 1], terms[5], rtol=0.0, atol=1.0e-14)
        kin = cable_kinematics(state, geometry, constants.CABLE_REST_LENGTH)
        assert np.allclose(direction, kin["direction"], rtol=0.0, atol=1.0e-14)


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"weather_scale": 0.35, "weather_direction": "mixed", "weather_rho": 0.5},
        {"weather_direction": "front", "weather_front_angle": 1.0},
        {"weather_scale": 0.0},
    ],
)
def test_regenerated_weather_is_bit_equal_to_build_run(changes):
    from dataclasses import replace

    from tether.campaign.fleet_run import FleetRunSpec, build_run

    spec = replace(FleetRunSpec(pretension=1000.0, heading_gain=500.0, duration=30.0, warmup=5.0), **changes)
    run = build_run(spec, 2003)
    own = run.fleet.cables._weather
    regenerated = rg.regenerate_weather(spec, 2003)
    if own is None:
        assert regenerated is None
    else:
        assert regenerated.dtype == own.dtype and np.array_equal(regenerated, own)


@pytest.fixture(scope="module")
def short_run():
    from tether.campaign.fleet_run import FleetRunSpec, build_run, run_to_end

    spec = FleetRunSpec(pretension=1000.0, heading_gain=500.0, weather_scale=1.0, duration=40.0, warmup=20.0)
    return run_to_end(build_run(spec, 2004))


def test_v2_summary_keeps_v1_fields_bit_for_bit(short_run):
    import copy

    from tether.campaign.summaries import summarize_run
    from tether.campaign.v2.summaries import _bytes_equal, summarize_run_v2

    reference = copy.deepcopy(summarize_run(short_run, 20.0, 1000.0))
    summary = summarize_run_v2(short_run, 20.0, 1000.0)
    assert set(reference) <= set(summary)
    assert _bytes_equal(reference, {key: summary[key] for key in reference}) == []
    # and a second v1 call after the v2 blocks is unchanged too (the v2 blocks do not mutate the run)
    assert _bytes_equal(reference, copy.deepcopy(summarize_run(short_run, 20.0, 1000.0))) == []


def test_v2_summary_blocks_are_consistent(short_run):
    from tether.campaign.common import json_bytes
    from tether.campaign.v2.summaries import summarize_run_v2

    summary = summarize_run_v2(short_run, 20.0, 1000.0)
    diagnostics = summary["v2_diagnostics"]
    assert diagnostics["weather_bit_equal_to_run"] is True
    assert diagnostics["wrel_offline_vs_log"]["passes_1e-6_relative"]
    assert diagnostics["marks_unmatched_to_crossings"] == 0
    assert diagnostics["onset_time_max_residual_s"] <= 1.0e-9
    assert diagnostics["marks_same_order_as_v1"] is True
    marks = summary["marks"]
    extra = summary["marks_v2"]
    n = marks["t_up"].size
    assert n > 0
    for name, values in extra.items():
        assert np.asarray(values).shape[0] == n, name
    assert set(extra["regime"]) <= {"N", "R1", "R2", "transitional"}
    # every window mark belongs to one event; in-window events absorb all their members
    events = summary["events"]
    counted = int(events["n_marks"].sum())
    orphans = diagnostics["window_marks_whose_event_parent_is_outside_window"]
    assert counted + orphans == n
    parents = extra["event_role"] == "parent"
    assert int(parents.sum()) == events["event_id"].size
    # onsets: v1's per-cable count, primary subset, exposures ordered
    onsets = summary["onsets_v2"]
    assert np.array_equal(onsets["per_cable"], summary["onsets"])
    assert onsets["primary_count"] <= onsets["count"]
    assert 0.0 <= onsets["primary_exposure"] <= onsets["exposure"]
    assert summary["clean"]["exposure"] == pytest.approx(onsets["primary_exposure"])
    # the primary flag of a mark is exactly "no up-crossing in (t_down - 3, t_down]"
    lag = extra["onset_parent_lag"]
    assert np.all(np.isnan(lag[extra["primary"]]))
    assert np.all((lag[~extra["primary"]] >= 0.0) & (lag[~extra["primary"]] < 3.0))
    # B.3's windows (t_up, t_up + 3 s], any cable, contain v1's [t_up, t_up + 1 s] same-cable
    # windows except the sample at t_up itself, so B.3 keeps at most one sample per up-crossing more
    ups = diagnostics["up_crossings"]
    assert np.all(summary["clean"]["moments"]["q"][0] <= summary["moments"]["q"][0] + ups)
    assert np.all(summary["clean"]["samples"] <= summary["clean"]["taut_samples_in_window"])
    shape = summary["shape"]
    assert shape["chord_world_std"].shape == (5,) and shape["psi_std"].shape == (5,)
    assert summary["closure"]["terminal"] is False and summary["closure"]["exposure_lost"] == 0.0
    json_bytes(summary)  # serialisable by the campaign's writer


def test_closure_block_reports_exposure_lost():
    from tether.campaign.v2.summaries import summarize_record

    time, e, wc, marks, downs, ups, down_time, interpolant, t0 = _synthetic()

    class Spec:
        weather_scale = 0.0
        warmup = 2.0
        duration = 30.0

    record = {
        "event_time": time, "elongation": e, "rate": np.zeros_like(e), "relative_load": np.zeros_like(e),
        "alive": np.ones_like(e, dtype=bool), "state_time": interpolant.time, "state": np.zeros((interpolant.time.size, 12)),
        "all_marks": {**{k: np.zeros(0) for k in ("t_up", "u_entry", "depth", "dwell", "v_return", "T_peak", "n_maxima",
                                                   "cable", "W_rel_at_onset", "W_rel_max")}},
        "closure": (0, 15.0), "sim_end": 15.0, "load_offsets": np.array([[2.0, 0.0]]), "vessel_offsets": np.array([[-1.5, 0.0]]),
        "stiffness": constants.CABLE_STIFFNESS, "damping": constants.CABLE_DAMPING, "spec": Spec(), "master_seed": 1,
    }
    record["state"][:, 3] = 14.0
    out = summarize_record(record, 2.0, t0)
    assert out["closure"] == {"terminal": True, "time": 15.0, "cable": 0, "exposure_lost": 17.0, "planned_end": 32.0,
                              "in_window": True}
    assert out["onsets_v2"]["exposure"] == pytest.approx(13.0)
