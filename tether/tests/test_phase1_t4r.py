"""P1-T4R deterministic full-plant study and replay tests."""

import numpy as np
import pytest

from tether.physics.phase1_deterministic import REDUCED_MASS
from tether.physics.phase1_t4r import (
    P1_T4R_BRANCHES,
    P1_T4R_ENTRY_SPEEDS,
    P1_T4R_GUST_RATIOS,
    P1_T4R_PHYSICS_STEPS,
    P1_T4R_PRODUCTION_STEP,
    P1T4RCellSpec,
    analyse_record,
    preregistered_cell_specs,
    record_arrays,
    run_cell,
)


def _copy_arrays(arrays):
    return {name: values.copy() for name, values in arrays.items()}


def _cell_index(arrays, *, step, branch):
    selected = (
        np.isclose(arrays["cell_physics_step_s"], step)
        & (arrays["cell_branch"] == branch)
    )
    return int(np.flatnonzero(selected)[0])


def _perturb_return_rate(arrays, cell_index, delta):
    cable = 2
    count = int(arrays["trace_sample_count"][cell_index])
    start = int(arrays["trace_sample_offset"][cell_index])
    elongation = arrays["trace_all_elongation_m"][start : start + count, cable]
    crossing = np.flatnonzero(
        (elongation[:-1] <= 0.0) & (elongation[1:] > 0.0)
    )[0]
    position_count = arrays["trace_truth"].shape[1] // 2
    target_velocity = position_count + 3 * (cable + 1)
    for sample_index in (crossing, crossing + 1):
        arrays["trace_all_elongation_rate_mps"][
            start + sample_index, cable
        ] += delta
        arrays["trace_truth"][start + sample_index, target_velocity] += delta


@pytest.fixture(scope="module")
def mini_cells():
    specs = [
        P1T4RCellSpec(step, 990.0, 0.8, 0.2, branch)
        for step in P1_T4R_PHYSICS_STEPS
        for branch in P1_T4R_BRANCHES
    ]
    return tuple(run_cell(spec) for spec in specs)


@pytest.fixture(scope="module")
def mini_arrays(mini_cells):
    return record_arrays(mini_cells)


def test_preregistered_grid_covers_domain_and_is_unique():
    specs = preregistered_cell_specs()

    assert len(specs) == 216
    assert len({spec.identifier for spec in specs}) == len(specs)
    assert {spec.physics_step_s for spec in specs} == set(P1_T4R_PHYSICS_STEPS)
    assert {spec.pretension_n for spec in specs} == {742.5, 990.0, 1237.5}
    assert {spec.entry_speed_mps for spec in specs} == set(P1_T4R_ENTRY_SPEEDS)
    assert {spec.gust_ratio for spec in specs} == set(P1_T4R_GUST_RATIOS)
    assert {spec.branch for spec in specs} == set(P1_T4R_BRANCHES)


def test_branches_share_turning_state_and_isolate_forcing(mini_cells):
    production_literal = [
        cell
        for cell in mini_cells
        if np.isclose(cell.spec.physics_step_s, P1_T4R_PRODUCTION_STEP)
    ]
    by_branch = {cell.spec.branch: cell for cell in production_literal}
    initial_states = [cell.trajectory.truth[0] for cell in production_literal]

    assert all(np.array_equal(initial_states[0], state) for state in initial_states[1:])
    isolated = by_branch["isolated_unforced"].trajectory
    constant = by_branch["constant_force_control"].trajectory
    confound = by_branch["forcing_active_confound"].trajectory
    departure = production_literal[0].departure
    assert np.any(departure.relative_load[:, 2] > 0.0)
    assert np.any(departure.gust_power_w != 0.0)
    assert np.all(departure.linear_drag_power_w == 0.0)
    assert np.all(departure.cable_damping_dissipation_w == 0.0)
    assert np.all(isolated.relative_load == 0.0)
    assert np.all(isolated.gust_power_w == 0.0)
    assert np.all(isolated.linear_drag_power_w == 0.0)
    assert np.all(isolated.cable_damping_dissipation_w == 0.0)
    assert np.any(constant.relative_load[:, 2] > 0.0)
    assert np.all(constant.linear_drag_power_w == 0.0)
    assert np.all(constant.cable_damping_dissipation_w == 0.0)
    assert np.any(confound.linear_drag_power_w < 0.0)
    assert np.any(confound.cable_damping_dissipation_w > 0.0)


def test_metrics_are_recomputed_exactly_and_events_are_ordered(mini_arrays):
    study = analyse_record(mini_arrays)

    assert all(
        row["event_order_and_return_valid"] and row["state_channels_consistent"]
        for row in study["measurements"]
    )
    for row in study["measurements"]:
        expected_literal_depth = row["measured_entry_speed_mps"] ** 2 / (
            2.0 * row["pretension_n"] / REDUCED_MASS
        )
        assert row["literal_ballistic_depth_m"] == pytest.approx(
            expected_literal_depth
        )
        assert row["turning_time_s"] > 0.0
        assert row["geometric_up_time_s"] > 0.0
        assert row["geometric_up_time_s"] <= row["force_onset_time_s"] + 1.0e-12


def test_single_worst_cell_and_step_sensitivity_fail_without_averaging(mini_arrays):
    failing_cell = _cell_index(
        mini_arrays,
        step=P1_T4R_PRODUCTION_STEP,
        branch="isolated_unforced",
    )
    perturbed_gate = _copy_arrays(mini_arrays)
    _perturb_return_rate(perturbed_gate, failing_cell, 1.0)
    gate_study = analyse_record(perturbed_gate)
    assert gate_study["literal_table_interpretation"]["status"] == "FAIL"
    assert (
        gate_study["literal_table_interpretation"]["worst_speed_cell"]["identifier"]
        == str(perturbed_gate["cell_identifier"][failing_cell])
    )

    coarse_cell = _cell_index(
        mini_arrays,
        step=P1_T4R_PHYSICS_STEPS[0],
        branch="isolated_unforced",
    )
    perturbed_step = _copy_arrays(mini_arrays)
    _perturb_return_rate(perturbed_step, coarse_cell, 0.1)
    step_study = analyse_record(perturbed_step)
    assert step_study["physics_step_sensitivity"]["status"] == "FAIL"


def test_missing_matched_return_fails_sensitivity_without_crashing(mini_arrays):
    perturbed = _copy_arrays(mini_arrays)
    indices = [
        _cell_index(perturbed, step=step, branch="forcing_active_confound")
        for step in P1_T4R_PHYSICS_STEPS
    ]
    for cell_index in indices:
        count = int(perturbed["trace_sample_count"][cell_index])
        start = int(perturbed["trace_sample_offset"][cell_index])
        trace_slice = slice(start, start + count)
        elongation = perturbed["trace_all_elongation_m"][trace_slice, 2]
        perturbed["trace_all_elongation_m"][trace_slice, 2] = -np.abs(elongation)

    study = analyse_record(perturbed)
    comparison_identifier = str(
        perturbed["cell_comparison_identifier"][indices[0]]
    )
    comparison = next(
        row
        for row in study["physics_step_sensitivity"]["comparisons"]
        if row["comparison_identifier"] == comparison_identifier
    )

    assert not comparison["comparison_valid"]
    assert np.isinf(comparison["speed_error_percentage_point_difference"])
    assert np.isinf(comparison["return_speed_relative_difference"])
    assert study["physics_step_sensitivity"]["status"] == "FAIL"
