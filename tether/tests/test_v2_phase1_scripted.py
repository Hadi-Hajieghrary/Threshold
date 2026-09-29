"""Unit tests for the Phase 1 deterministic scripted-cell harness (plan v2 Phase 1 (a)-(d))."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tether.campaign.v2 import phase1_scripted as ps
from tether.campaign.v2 import regime as rg
from tether.physics import constants
from tether.theory import drag_excursion as dx


def test_grid_matches_the_plan():
    jobs = ps.gust_jobs()
    c_jobs = [job for job in jobs if job.cell == "c"]
    d_jobs = [job for job in jobs if job.cell == "d"]
    assert len(c_jobs) == 126
    assert len(d_jobs) == 4
    assert sorted({job.lam for job in c_jobs}) == sorted(ps.LAMBDAS)
    assert sorted({job.t_x for job in c_jobs}) == sorted(ps.DURATIONS)
    assert sorted({job.pretension for job in c_jobs}) == sorted(ps.PRETENSIONS)
    assert sorted(job.lam * job.pretension for job in d_jobs) == sorted(ps.D_AMPLITUDES)
    assert all(job.t_x == ps.D_DURATION for job in d_jobs)


def test_every_duration_is_a_whole_number_of_weather_samples():
    for t_x in ps.DURATIONS + (ps.D_DURATION,):
        samples = t_x / rg.WEATHER_PERIOD
        assert abs(samples - round(samples)) < 1.0e-9


def test_gust_weather_realises_the_declared_W_c():
    """W^c = c_eff (W_L/c_L - W_A/c_A).d equals lambda T0 with d = +x, on the test cable only."""
    pretension, lam, t_x = 1000.0, 1.25, 1.5
    weather = ps.gust_weather(pretension, lam, t_x)
    direction = np.zeros((weather.shape[0], constants.VESSEL_COUNT, 2))
    direction[..., 0] = 1.0
    _, w_c = rg.conjugate_loads(weather, direction)
    start = int(round(ps.GUST_START / rg.WEATHER_PERIOD))
    stop = start + int(round(t_x / rg.WEATHER_PERIOD))
    assert np.allclose(w_c[start:stop, ps.TEST_CABLE], lam * pretension)
    assert np.allclose(w_c[:start, :], 0.0)
    assert np.allclose(w_c[stop:, :], 0.0)
    other = [i for i in range(constants.VESSEL_COUNT) if i != ps.TEST_CABLE]
    assert np.allclose(w_c[:, other], 0.0)
    # the load carries no force, so the other four cables are disturbed only through the plant
    assert np.allclose(weather[:, 0, :], 0.0)


def test_gust_weather_force_magnitude_and_sign():
    weather = ps.gust_weather(1000.0, 2.0, 0.2)
    start = int(round(ps.GUST_START / rg.WEATHER_PERIOD))
    force = weather[start, ps.TEST_CABLE + 1, 0]
    assert force == pytest.approx(-1.0636363636363637 * 2000.0)
    assert force < 0.0  # along -d, opposite the tow direction


def test_analytic_class_matches_the_declaration():
    # lambda <= 1: no exceedance, class N
    assert ps.analytic_class(0.6, 10.0, 1000.0) == "N"
    assert ps.analytic_class(1.0, 10.0, 1000.0) == "N"
    # t_x = 3.4 s lies below 2 tau_A = 3.4286 s, so it is transitional, not R2
    assert 2.0 * dx.TAU_A > 3.4
    assert ps.analytic_class(1.5, 3.4, 1000.0) == "transitional"
    assert ps.analytic_class(1.5, 6.0, 1000.0) == "R2"
    # R1 needs t_x < tau_Lf and v_s < 0.3 v_T
    assert ps.analytic_class(1.05, 0.2, 1000.0) == "R1"


def test_through_origin_recovers_a_known_slope():
    x = np.array([1.0, 2.0, 3.0])
    fit = ps.through_origin(x, 4.0 * x)
    assert fit["slope"] == pytest.approx(4.0)
    assert fit["uncentred_r2"] == pytest.approx(1.0)
    assert ps.through_origin(np.array([1.0]), np.array([1.0]))["slope"] is None


def test_aic_prefers_the_linear_fit_on_linear_data():
    x = np.array([1.5, 3.4, 6.0, 10.0])
    y = 2.0 * x + 0.5
    linear = ps.aic(x, y, 1)
    quadratic = ps.aic(x, y, 2)
    assert linear is not None
    assert quadratic is None  # n = 4 <= k = 4 for the quadratic: no AIC, by declaration
    x5 = np.array([1.5, 3.4, 6.0, 10.0, 14.0])
    assert ps.aic(x5, 2.0 * x5 + 0.5, 1) < ps.aic(x5, 2.0 * x5 + 0.5 + x5 * x5, 1)


def test_median_ignores_missing_values():
    assert ps._median([1.0, None, 3.0, float("nan")]) == pytest.approx(2.0)
    assert ps._median([]) is None
    assert ps._median([None]) is None


def test_heading_gains_are_the_owner_ruling():
    assert ps.HEADING_GAINS == {600.0: 477.0, 1000.0: 763.0, 1400.0: 1050.0}
    assert ps.K_SIGMA == 3.0
    assert ps.SWAY_LIMIT == pytest.approx(math.radians(20.0), rel=2.0e-3)


def test_spec_carries_the_declared_controller_and_no_stochastic_weather():
    spec = ps.GustJob("c", 1000.0, 1.5, 3.4).spec()
    assert spec.k_sigma == 3.0
    assert spec.sway_limit == ps.SWAY_LIMIT
    assert spec.weather_scale == 0.0
    assert spec.heading_gain == 763.0
    assert spec.formation == "parallel"
    assert spec.duration == 40.0 and spec.warmup == 0.0
    # the stochastic generator is never called for these specs
    assert rg.regenerate_weather(spec, 1) is None


def test_declarations_are_serialisable_and_complete():
    payload = ps._declarations()
    parsed = json.loads(json.dumps(payload))
    for name in ("P1-T1", "P1-T2", "P1-T3", "P1-T4", "P1-T5", "P1-T6", "P1-T7", "P1-T8", "P1-T9"):
        assert name in parsed["tests"]
    assert parsed["open_machinery_rulings"]["t_x"].startswith("LITERAL")
    assert parsed["open_machinery_rulings"]["bounces"].startswith("LITERAL")
    assert parsed["plant"]["stochastic_weather"].startswith("NONE")


def test_scoring_helpers_on_a_synthetic_record():
    """A hand-built record exercises the T4/T5/T7 scorers without touching Drake."""
    record = {
        "job": {"cell": "c", "pretension": 1000.0, "lam": 0.9, "t_x": 0.71, "cost": 1.0},
        "W_c_N": 900.0, "force_on_vessel_N": 957.3, "analytic_class": "N",
        "main": {"slack": True, "regime": "N", "V_up": 0.10, "u_entry": 0.10, "max_depth": 0.005,
                 "inside_gust": True, "a_bar_ret_time_mean": 1.0, "a_bar_ret_depth_mean": 1.0,
                 "driven_fraction_of_return": 0.0, "open_at_shutoff": False, "Delta_g": 0.0,
                 "post_gust_deepening": None, "closure_terminal": False, "returned_during_gust": True},
        "collateral": {"cable": [], "t_up": [], "u_entry": [], "depth": [], "v_return": [],
                       "T_peak": [], "regime": []},
        "shape": {"chord_world_std_max_deg": 0.2, "chord_load_std_max_deg": 0.2,
                  "psi_std_max_deg": 1.0, "load_yaw_std_deg": 0.03},
        "closure": {"terminal": False, "time": None, "cable": None, "exposure_lost": 0.0},
        "counts": {"marks_total": 1, "marks_after_gust_start": 1, "collateral_marks": 0,
                   "events_anchored": 1, "events_chained_in_window": 1, "onsets": 1, "primary_onsets": 1},
        "diagnostics": {"weather_bit_equal_to_run": True, "wrel_offline_passes_1e-6": True,
                        "marks_same_order_as_v1": True, "marks_unmatched_to_crossings": 0},
        "wall_seconds": 1.0, "end_time": 40.0,
    }
    predictions = {"P1-T4": {"summary": {}}, "P1-T5": {"N": {}}, "P1-T8": {"N": {}}}
    t4 = ps.score_t4([record], predictions)
    assert t4["n"] == 1
    assert t4["marks"][0]["symmetry_error"] == pytest.approx(0.0)
    assert t4["marks"][0]["symmetry_pass"] is True
    # depth 0.005 m against u^2/(2a) with a = 0.1 * 1000/523.9
    a_plan = 0.1 * 1000.0 / ps.M_FLEET
    assert t4["marks"][0]["depth_over_plan_ballistic"] == pytest.approx(0.005 / (0.01 / (2 * a_plan)))
    t5 = ps.score_t5([record], predictions)
    assert t5["per_class"]["N"]["n"] == 1
    assert t5["per_class"]["N"]["powered"] is False
    t8 = ps.score_t8([record], predictions)
    assert t8["per_class"]["N"]["0.005-0.3" if False else "0.05-0.3"]["n"] == 0


def test_t7_marks_a_closed_run_as_censored():
    record = {
        "job": {"cell": "c", "pretension": 1400.0, "lam": 2.0, "t_x": 10.0, "cost": 1.0},
        "main": {"slack": True, "regime": "R2", "closure_terminal": True, "open_at_shutoff": False,
                 "returned_during_gust": False, "max_depth": 11.0},
        "collateral": {"regime": []},
    }
    predictions = {"P1-T7": {}}
    scored = ps.score_t7([record], predictions)
    cell = scored["cells"]["1400@2.0"]
    assert cell["rows"][0]["censored"] is True
    assert cell["rows"][0]["censor_reason"] == "closure"
    assert cell["scored_points"] == 0


@pytest.mark.skipif(not ps.RESULTS_PATH.exists(), reason="the campaign has not been run")
def test_addendum_1_is_derived_from_the_results_and_edits_nothing():
    before = ps.RESULTS_PATH.read_bytes()
    payload = ps.addendum_1()
    assert ps.RESULTS_PATH.read_bytes() == before
    assert payload["results_sha256"] == ps.sha256_file(ps.RESULTS_PATH)
    assert "NOT EVALUABLE" in payload["reason"]
    assert len(payload["cells"]) == len([1 for t0 in ps.PRETENSIONS for lam in ps.LAMBDAS if lam > 1.0])


def test_t9_cell_conserves_energy_and_applies_opposite_forces():
    result = ps.run_t9()
    assert result["passed"] is True
    assert result["max_energy_drift"] < ps.T9_ENERGY_LIMIT
    assert result["max_abs_force_sum_N"] <= ps.T9_IMPULSE_LIMIT
    # at c = 0 the impact factor is 1, so the peak/speed ratio is sqrt(k m_eff)
    reference = math.sqrt(constants.CABLE_STIFFNESS * ps.M_EFF)
    for ratio in result["zero_damping_impedance_check"]["measured_peak_over_speed"]:
        assert ratio == pytest.approx(reference, rel=1.0e-3)
