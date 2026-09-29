"""Focused deterministic Phase 1 Stage 3-4 acceptance tests."""

import numpy as np
import pytest
from pydrake.multibody.plant import DiscreteContactApproximation

from tether.physics import constants
from tether.physics.cable import CableMode
from tether.physics.phase1_deterministic import (
    FullFormationConfig,
    GUST_DURATIONS,
    GUST_RATIOS,
    PHASE1R_ENTRY_SPEEDS,
    PHASE1R_GUST_DURATIONS,
    PHASE1R_GUST_RATIOS,
    PHASE1R_T6_PRIMARY_STEP,
    PHASE1R_T6_PRIMARY_WINDOW,
    PINNED_FLEET_IMPEDANCE,
    PRETENSION_LEVELS,
    build_full_formation_plant,
    deterministic_fleet_acceptance,
    phase1r_excursion_acceptance,
    run_full_formation,
    scripted_acceptance,
)


@pytest.fixture(scope="module")
def fleet_results():
    return deterministic_fleet_acceptance()


@pytest.fixture(scope="module")
def excursion_results():
    return scripted_acceptance()


def test_p1_t1_full_unilateral_plant_reproduces_phase0_transit(fleet_results):
    model = build_full_formation_plant(FullFormationConfig())
    result = fleet_results["P1-T1"]

    assert model.plant.time_step() == pytest.approx(constants.TIME_STEP)
    assert (
        model.plant.get_discrete_contact_approximation()
        == DiscreteContactApproximation.kSap
    )
    assert len(model.cables) == constants.VESSEL_COUNT
    assert all(cable.config.mode is CableMode.RECORDING for cable in model.cables)
    assert result["distance_relative_error"] <= 0.01
    assert result["tension_relative_error"] <= 0.03
    assert result["no_cable_slack"]
    assert result["rotational_excess_kg"] == 0.0
    assert result["passed"]


def test_p1_t2b_full_fleet_peaks_are_trajectory_derived_and_linear(fleet_results):
    result = fleet_results["P1-T2b"]

    assert result["speeds_mps"] == [0.25, 0.5, 1.0, 2.0, 3.0]
    assert all(peak > 990.0 for peak in result["first_peaks_n"])
    assert result["uncentred_r_squared"] > 0.99
    assert result["zero_speed_baseline_n"] >= 0.0
    assert len(result["baseline_subtracted_peaks_n"]) == 5
    assert result["baseline_subtracted_uncentred_r_squared"] > 0.99
    assert result["affine_r_squared"] > 0.99
    assert set(result["low_speed_residuals_n"]) == {
        "required_through_origin",
        "baseline_subtracted",
        "affine",
    }
    assert result["impedance_n_s_per_m"] == pytest.approx(PINNED_FLEET_IMPEDANCE)
    assert result["other_cables_taut_at_pretension"]
    assert result["passed"]


def test_full_plant_excursion_exposes_tracker_records_and_relative_load():
    trajectory = run_full_formation(
        FullFormationConfig(
            pretension=990.0,
            differential_gust_n=0.8 * 990.0,
            gust_duration_s=0.05,
        ),
        0.5,
        common_speed=2.0,
        target_relative_speed=-0.2,
        target_elongation=0.001,
    )

    target_records = [
        record
        for record in trajectory.reengagement_records
        if record.cable == trajectory.target_cable
    ]
    assert target_records
    record = target_records[0]
    assert record.W_rel_at_onset != 0.0
    assert record.W_rel_max != 0.0
    reengagement_index = int(
        next(
            index
            for index, time in enumerate(trajectory.time)
            if time >= record.t_up
        )
    )
    sampled_peak = max(
        trajectory.tension[reengagement_index :, trajectory.target_cable]
    )
    assert sampled_peak > 0.0
    assert record.T_peak == pytest.approx(sampled_peak, rel=0.02)
    assert trajectory.kinetic_energy_j.shape == trajectory.time.shape
    assert trajectory.spring_potential_j.shape == trajectory.time.shape
    assert np.all(trajectory.cable_damping_dissipation_w >= 0.0)
    assert np.all(trajectory.cable_clipping_dissipation_w >= 0.0)


def test_full_plant_decouples_production_physics_and_recording_steps():
    config = FullFormationConfig(
        time_step_s=0.00025,
        recording_step_s=0.001,
    )

    trajectory = run_full_formation(config, 0.005, common_speed=2.0)
    model = build_full_formation_plant(config)

    assert model.plant.time_step() == pytest.approx(0.00025)
    assert np.allclose(np.diff(trajectory.time), 0.001)
    assert all(
        event.time / 0.001 == pytest.approx(round(event.time / 0.001))
        for event in trajectory.event_records
    )


def test_full_plant_conservative_channel_controls_are_explicit():
    config = FullFormationConfig(
        thrust_enabled=False,
        drag_enabled=False,
        weather_enabled=False,
        gust_enabled=False,
        cable_damping_n_s_per_m=0.0,
        differential_gust_n=990.0,
        gust_duration_s=1.0,
        time_step_s=0.00025,
        recording_step_s=0.001,
    )

    trajectory = run_full_formation(config, 0.005, common_speed=2.0)

    assert np.allclose(np.diff(trajectory.time), 0.001)
    assert np.all(trajectory.gust_power_w == 0.0)
    assert np.all(trajectory.thrust_power_w == 0.0)
    assert np.all(trajectory.linear_drag_power_w == 0.0)
    assert np.all(trajectory.angular_drag_power_w == 0.0)
    assert np.all(trajectory.weather_power_w == 0.0)
    assert np.all(trajectory.cable_damping_dissipation_w == 0.0)
    assert np.all(trajectory.relative_load == 0.0)


def test_full_plant_restart_preserves_supplied_turning_state():
    config = FullFormationConfig(time_step_s=0.00025, recording_step_s=0.001)
    reference = run_full_formation(
        config,
        0.001,
        common_speed=2.0,
        target_elongation=-0.01,
    )

    restarted = run_full_formation(
        config,
        0.001,
        common_speed=0.0,
        initial_truth=reference.truth[0],
    )

    assert np.array_equal(restarted.truth[0], reference.truth[0])


def test_p1_t4_reports_literal_power_and_corrected_ballistics(excursion_results):
    tests, outcomes = excursion_results
    result = tests["P1-T4"]

    assert len(outcomes) == len(PRETENSION_LEVELS) * len(GUST_RATIOS) * len(
        GUST_DURATIONS
    )
    assert result["literal_eligible_count"] == 0
    assert result["verdict"] == "UNDER-POWERED"
    assert result["corrected_constant_gust_eligible_count"] > 0
    assert result["corrected_verdict"] == "PASS"


def test_p1_t5_energy_and_pinned_impedance_fits(excursion_results):
    result = excursion_results[0]["P1-T5"]

    assert abs(result["energy_slope"] - 1.0) <= 0.05
    assert result["energy_uncentred_r_squared"] > 0.98
    assert result["maximum_tension_relative_error"] <= 0.06
    assert result["verdict"] == "PASS"


def test_p1_t6_preserves_literal_failure_and_force_consistent_pivot(
    excursion_results,
):
    result = excursion_results[0]["P1-T6"]

    assert result["literal_verdict"] == "FAIL"
    assert result["verdict"] == "PIVOT"
    assert result["decisive_no_go"]
    assert result["force_consistent_verdict"] == "PASS"
    assert result["full_plant_force_consistent_verdict"] == "PASS"
    assert result["force_consistent_acceleration_fit"]["lambda_c"] == pytest.approx(
        1.0, rel=0.05
    )


def test_p1_t7_evaluates_literal_and_post_gust_depth_predictors(excursion_results):
    result = excursion_results[0]["P1-T7"]

    assert result["literal_eligible_count"] > 0
    assert result["literal_maximum_relative_error"] <= 0.15
    assert result["amended_eligible_count"] > 0
    assert result["amended_maximum_relative_error"] <= 0.15
    assert result["verdict"] == "PASS"


def test_p1_t8_reports_observed_complete_depth_and_sensitivity(excursion_results):
    result = excursion_results[0]["P1-T8"]

    assert result["observed_not_global"]
    assert result["observed_maximum_complete_depth_m"] > 0.0
    assert result["implied_maximum_snap_tension_n"] > 0.0
    assert result["complete_count"] > 0
    assert result["censored_count"] > 0
    assert len(result["sensitivity"]) >= 2
    assert result["verdict"] == "PASS"


def test_phase1r_grid_uses_full_plant_measurements_and_tracker_records(
    phase1r_results,
):
    _, outcomes = phase1r_results

    assert len(outcomes) == 60
    assert {item.pretension_n for item in outcomes} == set(PRETENSION_LEVELS)
    assert {item.gust_ratio for item in outcomes} == set(PHASE1R_GUST_RATIOS)
    assert {item.gust_duration_s for item in outcomes} == set(
        PHASE1R_GUST_DURATIONS
    )
    assert {item.requested_entry_speed_mps for item in outcomes} == set(
        PHASE1R_ENTRY_SPEEDS
    )
    assert all(item.status == "MEASURED" for item in outcomes)
    assert all(item.measured_gust_work_j > 0.0 for item in outcomes)
    assert all(item.measured_peak_tension_n > 0.0 for item in outcomes)
    assert any(
        not np.isclose(
            item.measured_peak_tension_n,
            PINNED_FLEET_IMPEDANCE * item.measured_return_speed_mps,
        )
        for item in outcomes
    )
    target_events = {
        event.kind
        for item in outcomes
        for event in item.trajectory.event_records
        if event.cable == item.trajectory.target_cable
    }
    assert {
        "geometric_down",
        "geometric_up",
        "force_onset",
        "force_cessation",
        "turning",
    } <= target_events


def test_phase1r_corrected_falsification_statuses(phase1r_results):
    tests, _ = phase1r_results

    energy = tests["P1-T5"]["full_plant_energy_balance"]
    impact = tests["P1-T5"]["impact_calibration"]
    surrogate = tests["P1-T5"]["reduced_effective_mass_surrogate"]
    assert tests["P1-T5"]["measurement_source"] == (
        "complete full-plant energy and work traces"
    )
    assert energy["status"] == "PASS"
    assert energy["maximum_normalized_residual"] <= energy["relative_tolerance"]
    assert surrogate["criterion"].startswith("diagnostic only")
    assert impact["below_range_count"] == 38
    assert impact["in_range_count"] == 22
    assert impact["above_range_count"] == 0
    assert impact["status"] == "MODEL_INVALID_LOW_SPEED"
    assert impact["in_range_affine_intercept_n"] > 0.0
    assert impact["affine_intercept_fraction_at_min_speed"] > 0.5
    assert tests["P1-T5"]["status"] == "MODEL_INVALID_LOW_SPEED"
    assert tests["P1-T6"]["bracketed_around_one"]
    assert tests["P1-T6"]["status"] == "PASS"
    assert tests["P1-T6"]["primary_estimator"]["time_step_s"] == (
        PHASE1R_T6_PRIMARY_STEP
    )
    assert tests["P1-T6"]["primary_estimator"]["fit_window_s"] == (
        PHASE1R_T6_PRIMARY_WINDOW
    )
    assert tests["P1-T6"]["primary_estimator"]["lambda_c"] == pytest.approx(
        1.0, rel=0.05
    )
    assert tests["P1-T6"]["sensitivity_status"] == "ROBUST_WITHIN_5_PERCENT"
    assert all(row["status"] == "PASS" for row in tests["P1-T6"]["sensitivity"])
    assert all(
        row["initial_elongation_m"] < 0.0
        and row["initial_tension_n"] == 0.0
        and row["maximum_elongation_m"] < 0.0
        and row["maximum_tension_n"] == 0.0
        for row in tests["P1-T6"]["probe_measurements"]
    )
    assert tests["P1-T6"]["withdrawn_original_depth_comparator"] == {
        "status": "WITHDRAWN_SPECIFICATION",
        "physical_verdict": None,
        "threshold_estimate": None,
    }
    assert tests["P1-T7"]["status"] == "PASS"
    assert tests["P1-T7"]["maximum_shutoff_depth_relative_error"] <= 0.15
    assert tests["P1-T7"]["maximum_amended_depth_relative_error"] <= 0.15
    assert tests["P1-T8"]["status"] == "OBSERVED_NOT_GLOBAL"
    assert not tests["P1-T8"]["global_maximum_established"]


def test_phase1r_square_gust_work_and_complete_energy_balance(phase1r_results):
    _, outcomes = phase1r_results
    for outcome in outcomes:
        balance = outcome.energy_balance
        assert balance is not None
        assert balance["gust_work_j"] == pytest.approx(
            outcome.measured_gust_work_j, abs=1.0e-12
        )
        assert balance["delta_mechanical_energy_j"] == pytest.approx(
            balance["delta_kinetic_j"] + balance["delta_spring_potential_j"]
        )
        expected = (
            balance["gust_work_j"]
            + balance["thrust_work_j"]
            + balance["linear_drag_work_j"]
            + balance["angular_drag_work_j"]
            + balance["weather_work_j"]
            - balance["cable_damping_dissipation_j"]
            - balance["cable_clipping_dissipation_j"]
        )
        assert balance["residual_j"] == pytest.approx(
            balance["delta_mechanical_energy_j"] - expected
        )