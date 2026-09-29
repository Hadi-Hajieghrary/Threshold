"""Phase 1 isolated cable mechanics acceptance tests."""

import numpy as np
import pytest

from tether.physics import constants
from tether.physics.phase1_mechanics import (
    IMPACT_SPEEDS,
    PINNED_IMPEDANCE,
    REFINEMENT_STEPS,
    _conservation_metrics,
    mechanics_acceptance,
    p1_t9r_acceptance,
    run_two_body_impact,
)


@pytest.fixture(scope="module")
def mechanics_results():
    return mechanics_acceptance()


@pytest.fixture(scope="module")
def p1_t9r_results():
    return p1_t9r_acceptance()


def test_p1_t2_trajectory_peaks_are_linear_and_match_impedance(mechanics_results):
    result = mechanics_results["P1-T2"]
    assert result["uncentred_r_squared"] > 0.99
    assert result["relative_error"] <= 0.15
    assert result["impedance_n_s_per_m"] == pytest.approx(PINNED_IMPEDANCE)
    assert result["passed"]


def test_p1_t3_impedance_recovers_radial_reduced_mass(mechanics_results):
    result = mechanics_results["P1-T3"]
    assert result["relative_error"] <= 0.10
    assert result["rotational_excess_kg"] == 0.0
    assert result["passed"]


def test_discrete_impact_records_exact_initial_state_before_first_update():
    speed = 1.0
    trajectory = run_two_body_impact(speed, 0.0, discrete=True)
    total_mass = constants.LOAD_MASS + constants.VESSEL_MASS

    assert trajectory.time[0] == 0.0
    assert trajectory.load_position[0] == 0.0
    assert trajectory.vessel_position[0] == constants.CABLE_REST_LENGTH
    assert trajectory.load_velocity[0] == (
        -constants.VESSEL_MASS * speed / total_mass
    )
    assert trajectory.vessel_velocity[0] == (
        constants.LOAD_MASS * speed / total_mass
    )
    assert trajectory.elongation[0] == pytest.approx(0.0, abs=1.0e-15)
    assert trajectory.tension[0] == 0.0


def test_discrete_force_time_level_is_identified_from_state_update():
    trajectory = run_two_body_impact(1.0, 0.0, discrete=True)
    metrics = _conservation_metrics(trajectory)

    assert metrics["body_impulse_evaluation"] == (
        "left_endpoint_discrete_force_verified_from_state_update"
    )
    assert metrics["left_endpoint_per_step_normalized_momentum_residual"] <= 1.0e-9
    assert metrics["right_endpoint_per_step_normalized_momentum_residual"] > 1.0e-3
    assert np.isfinite(metrics["right_endpoint_per_step_normalized_momentum_residual"])


def test_p1_t9_independent_energy_and_impulse_checks(mechanics_results):
    result = mechanics_results["P1-T9"]
    refinement = result["continuous_refinement"]
    assert [row["step_s"] for row in refinement] == [0.001, 0.0005, 0.00025]
    assert all(row["maximum_energy_relative_drift"] <= 0.005 for row in refinement)
    assert all(row["maximum_normalized_body_residual"] <= 0.005 for row in refinement)
    assert result["continuous_refinement_passed"]

    production = result["production_discrete_1ms"]
    assert production["integrator"] == "discrete_sap"
    assert production["body_impulse_evaluation"] == (
        "left_endpoint_discrete_force_verified_from_state_update"
    )
    assert production["maximum_normalized_body_residual"] < 1.0e-12
    assert production["maximum_energy_relative_drift"] > 0.005
    assert result["blocking_failure_basis"] == "discrete_1ms_energy"
    assert not production["passed"]
    assert not result["passed"]


def test_p1_t9r_step_sweep_has_expected_discrete_outcomes(p1_t9r_results):
    steps = p1_t9r_results["step_results"]

    assert [row["step_s"] for row in steps] == list(REFINEMENT_STEPS)
    assert all(len(row["speed_results"]) == len(IMPACT_SPEEDS) for row in steps)
    assert not steps[0]["passes_numerically"]
    assert steps[1]["passes_numerically"]
    assert steps[2]["passes_numerically"]
    assert p1_t9r_results["half_millisecond_passes_numerically"]
    assert not p1_t9r_results["half_millisecond_selected"]
    assert p1_t9r_results["candidate_passes_numerically"]
    assert p1_t9r_results["passed"]


def test_p1_t9r_convergence_and_diagnostics_are_separate(p1_t9r_results):
    assert p1_t9r_results["monotone_decrease_passed"]
    assert p1_t9r_results["convergence_orders_passed"]
    assert p1_t9r_results["continuous_rk3_controls_passed"]
    assert all(
        row["orders_defined"]
        and 0.8 <= row["coarse_to_medium_order"] <= 1.2
        and 0.8 <= row["medium_to_fine_order"] <= 1.2
        for row in p1_t9r_results["convergence"]
    )
    for step in p1_t9r_results["step_results"]:
        for row in step["speed_results"]:
            assert row["initial_state_exact"]
            assert row["maximum_per_step_normalized_body_momentum_residual"] <= 1.0e-9
            assert row["equal_opposite_integrated_impulse_mismatch"] <= 1.0e-9
            assert row["normalized_total_momentum_drift"] <= 1.0e-9
            assert "post_release_endpoint_relative_energy_defect" in row
            assert "maximum_shadow_energy_relative_drift" in row