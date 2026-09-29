"""Unit tests for Phase 1(f): the stochastic-cell machinery (tether/campaign/v2/phase1_stochastic.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tether.campaign.v2 import phase1_stochastic as ps


# --------------------------------------------------------------------------- the declared grid


def test_cells_match_the_declared_phase2_grid():
    assert len(ps.CELLS) == 8
    selected = [c for c in ps.CELLS if c.k_sigma == ps.SELECTED_K_SIGMA]
    assert len(selected) == 6
    assert {(c.intensity, c.pretension) for c in selected} == {
        (0.35, 600.0), (0.35, 800.0), (0.35, 1000.0), (0.5, 600.0), (0.5, 1000.0), (0.5, 1400.0)}
    # k_h = 1.5 x the lateral stability boundary 0.318 (1.5 T0 + k_c), k_c = 100
    for cell in ps.CELLS:
        boundary = 0.318 * (1.5 * cell.pretension + 100.0)
        assert cell.heading_gain == pytest.approx(round(1.5 * boundary), abs=1.0)
    # the comparison cells change exactly one thing against the selected 1 kN cell
    for name in ("i035_T1000_ks0", "i050_T1000_ks0"):
        comparison = ps.CELL_BY_NAME[name]
        assert comparison.k_sigma == 0.0 and comparison.sway_limit is None
        assert comparison.heading_gain == ps.CELL_BY_NAME[name.replace("ks0", "ks3")].heading_gain
    assert len(ps.PILOT_SEEDS) == 2 and len(ps.STATISTICS_SEEDS) == 20
    assert ps.ALL_SEEDS == tuple(range(7101, 7123))


def test_specs_differ_only_in_the_declared_fields():
    base = ps.spec_for(ps.CELL_BY_NAME["i035_T1000_ks3"])
    comparison = ps.spec_for(ps.CELL_BY_NAME["i035_T1000_ks0"])
    assert base.k_sigma == 3.0 and base.sway_limit == ps.SWAY_LIMIT_V2
    assert comparison.k_sigma == 0.0 and comparison.sway_limit is None
    assert base.config_hash() != comparison.config_hash()
    assert base.duration == 300.0 and base.warmup == 20.0 and base.cable_mode == "recording"
    assert base.weather_scale == 0.35 and base.formation == "parallel"


def test_declarations_state_the_cost_and_the_literal_readings():
    declarations = ps.build_declarations()
    cost = declarations["cells"]["cost"]
    assert cost["simulated_seconds"] == 8 * 22 * 300.0
    assert cost["core_hours_plan_basis_7000_per_core_hour"] == pytest.approx(52800 / 7000.0)
    assert "FULL merged W^c episode" in declarations["context"]["open_machinery_rulings"]
    assert "ANCHORED" in declarations["context"]["open_machinery_rulings"]
    assert len(declarations["cells"]["grid"]) == 8


# --------------------------------------------------------------------------- small helpers


def test_through_origin_recovers_a_known_slope():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    fit = ps.through_origin(x, 2.5 * x)
    assert fit["slope"] == pytest.approx(2.5)
    assert fit["uncentred_r2"] == pytest.approx(1.0)
    assert fit["n"] == 4
    # non-finite rows are dropped, not propagated
    fit = ps.through_origin(np.array([1.0, np.nan, 3.0]), np.array([2.0, 5.0, 6.0]))
    assert fit["n"] == 2 and fit["slope"] == pytest.approx(2.0)
    assert ps.through_origin(np.array([1.0]), np.array([1.0]))["slope"] is None


def test_threshold_grid_snaps_dedupes_and_survives_an_empty_pilot():
    peaks = np.array([1000.0, 2000.0, 3000.0, 12000.0])
    grid = ps.threshold_grid(peaks)
    assert grid["quantiles"] == [50.0, 80.0, 95.0, 99.0]
    assert grid["pilot_marks"] == 4
    for level in grid["levels_N"]:
        assert level % ps.GRID_STEP_N == 0
    assert grid["levels_N"] == sorted(set(grid["levels_N"]), key=grid["levels_N"].index)
    # two quantiles landing on the same 0.5 kN level collapse to one
    flat = ps.threshold_grid(np.array([1000.0, 1010.0, 1020.0, 1030.0]))
    assert flat["levels_N"] == [1000.0]
    empty = ps.threshold_grid(np.empty(0))
    assert empty["levels_N"] == [] and empty["pilot_marks"] == 0 and "no threshold grid" in empty["note"]


def test_poisson_and_binomial_intervals_bracket_the_estimate():
    low, high = ps.poisson_interval(10, 100.0)
    assert low < 0.1 < high
    assert ps.poisson_interval(0, 100.0)[0] == 0.0
    assert ps.poisson_interval(0, 100.0)[1] > 0.0
    low, high = ps.binomial_interval(0, 22)
    assert low == 0.0 and 0.0 < high < 0.25  # a zero-closure cell is below 0.25 with its interval reported
    assert ps.binomial_interval(22, 22)[1] == 1.0


# --------------------------------------------------------------------------- return leg (P1-T5 / T8 inputs)


def test_return_leg_quantities_read_the_deepest_point_and_the_driving_weather():
    # one cable, one mark: e dips to -0.5 m at sample 4 and re-engages at sample 8.
    time = np.arange(10) * 1.0e-3
    elongation = np.array([0.1, -0.1, -0.3, -0.45, -0.5, -0.4, -0.2, -0.05, 0.05, 0.2])[:, None]
    wc = np.array([0.0, 0.0, 0.0, 0.0, 900.0, 900.0, 0.0, 0.0, 0.0, 0.0])[:, None]
    summary = {
        "marks": {"t_up": np.array([8.0e-3]), "cable": np.array([0]), "v_return": np.array([0.6]),
                  "depth": np.array([0.5])},
        "marks_v2": {"i_down": np.array([1]), "i_up": np.array([8])},
    }
    out = ps.return_leg_quantities(summary, wc, elongation, time, pretension=600.0)
    assert out["t_deep"][0] == pytest.approx(4.0e-3)
    assert out["return_leg_s"][0] == pytest.approx(4.0e-3)
    assert out["a_bar_ret_time"][0] == pytest.approx(0.6 / 4.0e-3)
    # the return leg is samples 4..7; W^c exceeds T0 on two of those four
    assert out["driven_fraction"][0] == pytest.approx(0.5)


def test_return_leg_quantities_skip_a_mark_without_a_slack_interval():
    time = np.arange(5) * 1.0e-3
    elongation = np.zeros((5, 1))
    summary = {"marks": {"t_up": np.array([1.0]), "cable": np.array([0]), "v_return": np.array([1.0]),
                         "depth": np.array([1.0])},
               "marks_v2": {"i_down": np.array([-1]), "i_up": np.array([-1])}}
    out = ps.return_leg_quantities(summary, np.zeros((5, 1)), elongation, time, 1000.0)
    assert math.isnan(out["a_bar_ret_time"][0]) and math.isnan(out["driven_fraction"][0])


# --------------------------------------------------------------------------- P1-T5 and P1-T8


def _marks(depth, a_time, regime, driven=None, t0=1000.0):
    depth = np.asarray(depth, dtype=float)
    a_time = np.asarray(a_time, dtype=float)
    v_up = np.sqrt(2.0 * a_time * depth)
    return {
        "depth": depth, "v_return": v_up, "a_bar_ret_time": a_time,
        "a_bar_ret": v_up**2 / (2.0 * depth),
        "driven_fraction": np.zeros(depth.size) if driven is None else np.asarray(driven, dtype=float),
        "regime": np.asarray(regime), "t_up": np.arange(depth.size, dtype=float),
    }


def test_t5_is_an_identity_on_the_depth_reading_and_passes_on_a_consistent_set():
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.05, 3.0, 40)
    a_time = rng.uniform(0.5, 2.0, 40)
    marks = _marks(depth, a_time, np.full(40, "R2"))
    out = ps.t5_block({"R2": marks})["R2"]
    assert out["depth_mean_reading"]["slope"] == pytest.approx(1.0)
    assert out["time_mean_reading"]["slope"] == pytest.approx(1.0)
    assert out["verdict"] == "PASS" and out["powered_20_marks"]


def test_t5_reports_under_powered_below_twenty_marks_and_fails_a_biased_set():
    marks = _marks(np.full(10, 1.0), np.full(10, 1.0), np.full(10, "R1"))
    assert ps.t5_block({"R1": marks})["R1"]["verdict"] == "UNDER-POWERED"
    biased = _marks(np.full(25, 1.0), np.full(25, 1.0), np.full(25, "R1"))
    biased["v_return"] = biased["v_return"] * 1.5  # V_up half again the law's prediction
    out = ps.t5_block({"R1": biased})["R1"]
    assert out["verdict"] == "FAIL" and out["time_mean_reading"]["slope"] == pytest.approx(1.5)


def test_t5_excludes_zero_depth_marks_and_counts_them():
    rng = np.random.default_rng(1)
    depth = np.concatenate([rng.uniform(0.05, 2.0, 25), np.zeros(5)])
    a_time = rng.uniform(0.5, 2.0, 30)
    marks = _marks(np.where(depth > 0, depth, 1.0), a_time, np.full(30, "N"))
    marks["depth"] = depth
    marks["a_bar_ret"] = np.where(depth > 0, marks["a_bar_ret"], np.nan)
    out = ps.t5_block({"N": marks})["N"]
    assert out["marks"] == 30 and out["marks_with_positive_depth"] == 25
    assert out["marks_zero_depth_excluded"] == 5
    assert out["time_mean_reading"]["n"] == 25
    assert out["powered_20_marks"]


def test_t8_bands_and_a0_normalisation():
    depth = np.array([0.1, 0.5, 2.0, 5.0, 0.2])
    a_time = np.array([1.0, 2.0, 3.0, 4.0, 3.0])
    marks = _marks(depth, a_time, np.full(5, "N"), driven=np.array([0.0, 0.5, 1.0, 1.0, 0.0]))
    out = ps.t8_block({"N": marks}, {"N": np.full(5, 1000.0)})["N"]
    a0 = 1000.0 / ps.M_FLEET
    assert out["bands"]["0.05-0.3"]["n"] == 2
    assert out["bands"]["0.05-0.3"]["median_a_time_over_a0"] == pytest.approx(np.median([1.0, 3.0]) / a0)
    assert out["bands"]["3.0-12.0"]["n"] == 1
    assert out["bands"]["3.0-12.0"]["median_a_time_over_a0"] == pytest.approx(4.0 / a0)
    assert out["bands"]["0.3-1.0"]["median_driven_fraction"] == pytest.approx(0.5)
    assert out["marks_below_0.05m"] == 0 and out["marks_above_12m"] == 0


# --------------------------------------------------------------------------- cells and the phase tests


def _run(cell_name, seed, *, pilot=False, chord_deg=10.0, psi_deg=8.0, marks=0, peaks=(), regimes=(),
         closure=None, onsets=0, primary=0):
    n = len(peaks) if len(peaks) else marks
    peaks = np.asarray(peaks, dtype=float) if len(peaks) else np.zeros(n)
    regimes = np.asarray(regimes) if len(regimes) else np.full(n, "N")
    # circular sums that reproduce a wanted circular std exactly: R = exp(-std^2/2)
    count = 3000

    def sums(deg):
        resultant = math.exp(-math.radians(deg) ** 2 / 2.0)
        return {"cos": np.full(5, resultant * count), "sin": np.zeros(5), "count": count}

    mark_table = {
        "t_up": np.arange(n, dtype=float) + 21.0, "cable": np.zeros(n, dtype=int),
        "u_entry": np.full(n, 0.05), "depth": np.full(n, 0.5), "dwell": np.full(n, 1.0),
        "v_return": np.full(n, 1.0), "T_peak": peaks, "n_maxima": np.ones(n, dtype=int),
        "regime": regimes, "regime_clip": regimes, "t_x": np.zeros(n), "t_x_clip": np.zeros(n),
        "Wc_max": np.zeros(n), "v_s": np.zeros(n), "any_exceedance": np.zeros(n, dtype=bool),
        "n_episodes": np.zeros(n, dtype=int), "a_bar_ret": np.ones(n), "Wc_return_mean": np.zeros(n),
        "Wc_slack_max": np.zeros(n), "primary": np.ones(n, dtype=bool), "event_id": np.arange(n),
        "event_role": np.full(n, "parent"), "t_deep": np.arange(n, dtype=float) + 20.5,
        "a_bar_ret_time": np.ones(n), "driven_fraction": np.zeros(n), "return_leg_s": np.ones(n),
    }
    return {
        "cell": cell_name, "seed": seed, "pilot": pilot, "config_hash": "hash", "end_time": 320.0,
        "exposure_s": 300.0, "window_samples": 300000, "tracker_dropped": 0, "lint_violations": [],
        "closure": ({"terminal": False, "time": None, "cable": None, "exposure_lost": 0.0, "planned_end": 320.0,
                     "in_window": None} if closure is None else
                    {"terminal": True, "time": closure, "cable": 0, "exposure_lost": 320.0 - closure,
                     "planned_end": 320.0, "in_window": True}),
        "slack_samples": np.zeros(5, dtype=int), "onsets_count": onsets, "primary_onsets": primary,
        "primary_onsets_2s_same_cable": primary, "primary_exposure_s": 300.0,
        "onsets_per_cable": np.zeros(5, dtype=int), "marks": mark_table, "n_marks": n,
        "events": {"cluster_max_T_peak": peaks, "regime": regimes, "n_marks": np.ones(n, dtype=int),
                   "n_bounces": np.zeros(n, dtype=int), "primary": np.ones(n, dtype=bool)},
        "n_events": n, "events_chained_sensitivity": n, "regime_summary": {},
        "clean": {"exposure_s": 250.0, "window_exposure_s": 300.0, "excluded_fraction": 1 / 6,
                  "q_count": np.full(5, 1000.0), "q_sum": np.full(5, 1.0e6), "q_square": np.full(5, 1.01e9)},
        "shape_sums": {"chord_world": sums(chord_deg), "chord_load": sums(chord_deg), "psi": sums(psi_deg),
                       "load_yaw": {"cos": np.array([math.exp(-0.02) * count]), "sin": np.array([0.0]), "count": count}},
        "shape_samples": count,
        "shape_std_deg": {"chord_world": np.full(5, chord_deg), "chord_load": np.full(5, chord_deg),
                          "psi": np.full(5, psi_deg), "load_yaw": np.array([11.0])},
        "diagnostics": {"weather_bit_equal": True, "marks_same_order_as_v1": True, "marks_unmatched": 0,
                        "wrel_passes": True, "orphan_marks": 0},
        "wall_seconds": 100.0,
    }


def _cell_runs(cell_name, **kwargs):
    runs = [_run(cell_name, seed, pilot=True, **kwargs) for seed in ps.PILOT_SEEDS]
    runs += [_run(cell_name, seed, **kwargs) for seed in ps.STATISTICS_SEEDS]
    return runs


def test_cell_block_pools_shape_exposure_and_rates():
    runs = _cell_runs("i035_T1000_ks3", chord_deg=25.4, psi_deg=13.0, onsets=10, primary=4)
    block = ps.cell_block(ps.CELL_BY_NAME["i035_T1000_ks3"], runs)
    assert block["P1_T11"]["chord_world_stat_deg"] == pytest.approx(25.4, rel=1e-6)
    assert block["P1_T11"]["psi_stat_deg"] == pytest.approx(13.0, rel=1e-6)
    assert block["P1_T11"]["verdict"] == "FAIL"  # chord misses, psi is inside
    assert block["P1_T11"]["psi_meets_15"] and not block["P1_T11"]["chord_world_meets_15"]
    assert block["exposure_s"] == pytest.approx(20 * 300.0)  # statistics seeds only
    assert block["onsets"] == 200 and block["primary_onsets"] == 80
    assert block["primary_rate_per_cable_s"] == pytest.approx(80 / (5 * 20 * 300.0))
    low, high = block["primary_rate_ci95_per_cable_s"]
    assert low < block["primary_rate_per_cable_s"] < high
    assert block["clean_set"]["exposure_lost_fraction"] == pytest.approx(1 / 6)
    assert block["closure"]["closures"] == 0 and block["closure"]["below_0.25"]
    assert block["diagnostics"]["wrel_checked_runs"] == 22 and block["diagnostics"]["wrel_passed_runs"] == 22


def test_wrel_check_counts_not_checked_runs_separately():
    """A run through the addendum-1 shim carries no C3 verdict; it must not read as a failure."""
    runs = _cell_runs("i035_T1000_ks3")
    for run in runs[:5]:
        run["diagnostics"]["wrel_passes"] = None
    block = ps.cell_block(ps.CELL_BY_NAME["i035_T1000_ks3"], runs)
    assert block["diagnostics"]["wrel_not_checked_runs"] == 5
    assert block["diagnostics"]["wrel_checked_runs"] == 17
    assert block["diagnostics"]["wrel_passes_every_checked_run"]
    runs[6]["diagnostics"]["wrel_passes"] = False
    assert not ps.cell_block(ps.CELL_BY_NAME["i035_T1000_ks3"], runs)["diagnostics"]["wrel_passes_every_checked_run"]


def test_cell_block_grid_comes_from_the_pilots_only_and_counts_events_on_the_statistics_seeds():
    peaks = np.array([1000.0, 2000.0, 3000.0, 12000.0])
    runs = [_run("i050_T1000_ks3", seed, pilot=True, peaks=peaks) for seed in ps.PILOT_SEEDS]
    runs += [_run("i050_T1000_ks3", seed, peaks=np.full(30, 4000.0), regimes=np.full(30, "R2"))
             for seed in ps.STATISTICS_SEEDS]
    block = ps.cell_block(ps.CELL_BY_NAME["i050_T1000_ks3"], runs)
    assert block["pilots"]["marks"] == 8  # two pilot seeds x four marks
    assert block["marks"] == 600 and block["events"] == 600  # pilots excluded from the scored counts
    lowest = block["threshold_grid"]["levels_N"][0]
    assert block["snap_exceedances"][f"{lowest:.0f}"]["events_total"] == 600
    assert block["snap_exceedances"][f"{lowest:.0f}"]["events_by_class"]["R2"] == 600
    top = block["threshold_grid"]["levels_N"][-1]
    assert top > 4000.0 and block["snap_exceedances"][f"{top:.0f}"]["events_total"] == 0


def test_closure_block_reports_the_fraction_time_distribution_and_lost_exposure():
    runs = _cell_runs("i050_T0600_ks3")
    for run in runs[:8]:
        run["closure"] = {"terminal": True, "time": 120.0, "cable": 1, "exposure_lost": 200.0,
                          "planned_end": 320.0, "in_window": True}
    block = ps.cell_block(ps.CELL_BY_NAME["i050_T0600_ks3"], runs)
    assert block["closure"]["closures"] == 8 and block["closure"]["runs"] == 22
    assert block["closure"]["fraction"] == pytest.approx(8 / 22)
    assert not block["closure"]["below_0.25"]
    assert block["closure"]["exposure_lost_s_total"] == pytest.approx(1600.0)
    assert block["closure"]["time_quartiles_s"] == [120.0, 120.0, 120.0]


def test_analyse_scores_T10prime_T12_and_builds_the_trade_off_table():
    runs = []
    for cell in ps.CELLS:
        # every selected cell but one is event-poor; i050_T0600_ks3 carries a populated class
        if cell.name == "i050_T0600_ks3":
            peaks = np.array([3000.0, 5000.0, 7000.0, 9000.0])
            statistics = np.full(40, 12000.0)
            runs += [_run(cell.name, seed, pilot=True, peaks=peaks, regimes=np.full(4, "N"), chord_deg=39.0,
                          psi_deg=17.0) for seed in ps.PILOT_SEEDS]
            runs += [_run(cell.name, seed, peaks=statistics, regimes=np.full(40, "N"), chord_deg=39.0,
                          psi_deg=17.0, onsets=50, primary=20) for seed in ps.STATISTICS_SEEDS]
        else:
            runs += _cell_runs(cell.name, chord_deg=25.4, psi_deg=13.0)
    results = ps.analyse(runs)
    assert set(results["cells"]) == {c.name for c in ps.CELLS}
    assert "_marks" not in results["cells"]["i035_T1000_ks3"]
    population = results["P1_T10prime"]["by_class"]["N"]
    assert population["per_cell"]["i050_T0600_ks3"]["levels_with_20_events"] >= 3
    assert population["cells_with_3_powered_levels"] == 1
    assert not population["meets_population_rule"]  # one cell, the rule asks for three
    assert not results["P1_T10prime"]["any_class_populated"]
    assert results["P1_T12"]["total_closures_all_cells"] == 0
    assert results["P1_T12"]["clause_a_one_cell_below_0.25_at_each_intensity"]
    assert results["P1_T12"]["clause_b_rate_carries_no_interval"]["degenerate"]
    assert not results["P1_T12"]["non_degenerate"]
    assert results["P1_T12"]["verdict"].startswith("DEGENERATE on the plan's SECOND clause")
    assert len(results["trade_off"]) == 8
    row = next(r for r in results["trade_off"] if r["cell"] == "i050_T0600_ks3")
    assert row["meets_H1"] is False and row["meets_population_rule_in_some_class"]
    assert row["chord_gap_ratio"] == pytest.approx(39.0 / 15.0, rel=1e-6)
    assert all(r["k_sigma"] in (0.0, 3.0) for r in results["trade_off"])
    closest = results["closest_to_both"]
    assert len(closest["ranked"]) == 6  # the selected cells only
    assert closest["least_far"] == "i050_T0600_ks3"  # the only cell with events at its third level
    assert closest["ranked"][0]["population_shortfall_factor"] <= 1.0
    assert math.isinf(closest["ranked"][-1]["population_shortfall_factor"])


def test_analyse_survives_the_json_round_trip_the_records_take():
    import json

    from tether.campaign.common import json_bytes

    runs = []
    for cell in ps.CELLS:
        if cell.name == "i050_T1000_ks3":
            runs += _cell_runs(cell.name, peaks=np.linspace(1000.0, 9000.0, 25),
                               regimes=np.array(["N"] * 20 + ["R2"] * 5), onsets=30, primary=12)
        else:
            runs += _cell_runs(cell.name)  # event-free cells: empty mark and event tables
    round_tripped = json.loads(json_bytes({"runs": runs}).decode("ascii"))["runs"]
    results = ps.analyse(round_tripped)
    assert results["cells"]["i050_T1000_ks3"]["marks"] == 20 * 25
    assert results["cells"]["i035_T0600_ks3"]["marks"] == 0
    assert results["cells"]["i035_T0600_ks3"]["threshold_grid"]["levels_N"] == []
    assert results["cells"]["i050_T1000_ks3"]["marks_by_class"]["R2"] == 20 * 5
    assert results["P1_T5"]["pooled_over_cells"]["all"]["marks"] == 20 * 25
    assert len(results["trade_off"]) == 8
    json_bytes(results)  # the record serialises


def test_summary_without_reengagements_matches_the_summariser_on_a_run_that_has_one():
    """The addendum-1 shim, checked against summarize_run_v2 where the summariser works.

    The defect the shim exists for (no re-engagement anywhere, at least one onset) makes
    ``summarize_run_v2`` raise, so the two cannot be compared there; on a quiet run with
    no crossing at all both paths run and must agree block for block.
    """
    from tether.campaign.fleet_run import FleetRunSpec, build_run, run_to_end
    from tether.campaign.v2 import events as ev
    from tether.campaign.v2 import regime as rg
    from tether.campaign.v2.summaries import summarize_run_v2

    spec = FleetRunSpec(formation="parallel", pretension=1000.0, heading_gain=763.0, trim_gain=100.0,
                        weather_distribution="gaussian", weather_direction="local", weather_scale=0.05,
                        duration=15.0, warmup=5.0, cable_mode="recording", log_state=True,
                        k_sigma=3.0, sway_limit=ps.SWAY_LIMIT_V2)
    run = run_to_end(build_run(spec, 9002))
    weather = rg.regenerate_weather(spec, 9002, ps.N_CABLES)
    log = run.fleet.cables.log
    ups, _ = ev.crossings(log.event_time[: log.count], log.elongation[: log.count], log.alive[: log.count])
    assert ups.size == 0, "the fixture must be a run with no re-engagement"

    reference = summarize_run_v2(run, 5.0, 1000.0, weather=weather)
    shim = ps.summary_without_reengagements(run, 5.0, 1000.0, weather)

    assert shim["onsets_v2"]["count"] == reference["onsets_v2"]["count"]
    assert shim["onsets_v2"]["primary_count"] == reference["onsets_v2"]["primary_count"]
    assert shim["onsets_v2"]["primary_exposure"] == pytest.approx(reference["onsets_v2"]["primary_exposure"])
    assert shim["clean"]["exposure"] == pytest.approx(reference["clean"]["exposure"])
    assert shim["clean"]["excluded_fraction"] == pytest.approx(reference["clean"]["excluded_fraction"] or 0.0)
    assert np.allclose(np.asarray(shim["clean"]["moments"]["q"], dtype=float),
                       np.asarray(reference["clean"]["moments"]["q"], dtype=float))
    assert shim["closure"] == reference["closure"]
    for name in ("chord_world", "chord_load", "psi", "load_yaw"):
        assert np.allclose(shim["shape"]["sums"][name]["cos_sum"], reference["shape"]["sums"][name]["cos_sum"])
    assert int(np.asarray(shim["marks"]["t_up"]).size) == int(np.asarray(reference["marks"]["t_up"]).size)
    assert shim["v2_diagnostics"]["no_reengagement_in_run"] is True
    assert shim["v2_diagnostics"]["weather_bit_equal_to_run"] is True


def test_T12_is_degenerate_when_an_intensity_has_no_cell_below_the_bound():
    runs = []
    for cell in ps.CELLS:
        cell_runs = _cell_runs(cell.name)
        if cell.intensity == 0.35 and cell.k_sigma == ps.SELECTED_K_SIGMA:
            for run in cell_runs:
                run["closure"] = {"terminal": True, "time": 100.0, "cable": 0, "exposure_lost": 220.0,
                                  "planned_end": 320.0, "in_window": True}
        runs += cell_runs
    results = ps.analyse(runs)
    assert not results["P1_T12"]["non_degenerate"]
    assert not results["P1_T12"]["clause_a_one_cell_below_0.25_at_each_intensity"]
    assert results["P1_T12"]["0.35"]["cells_below_0.25"] == []
    assert results["P1_T12"]["0.5"]["satisfied"]
    assert results["P1_T12"]["verdict"].startswith("DEGENERATE: some intensity")
