"""Deterministic full-plant P1-T4R ballistic-symmetry study."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tether.physics import constants
from tether.physics.phase1_deterministic import (
    GEOMETRIC_DEPTH_LIMIT,
    PRETENSION_LEVELS,
    REDUCED_MASS,
    FullFormationConfig,
    FullFormationTrajectory,
    run_full_formation,
)

P1_T4R_PHYSICS_STEPS = (5.0e-4, 2.5e-4)
P1_T4R_PRODUCTION_STEP = 2.5e-4
P1_T4R_RECORDING_STEP = 1.0e-3
P1_T4R_ENTRY_SPEEDS = np.array([0.10, 0.20, 0.35, 0.50])
P1_T4R_GUST_RATIOS = np.array([0.0, 0.8, 0.95])
P1_T4R_BRANCHES = (
    "isolated_unforced",
    "constant_force_control",
    "forcing_active_confound",
)
P1_T4R_SPEED_ERROR_LIMIT = 0.05
P1_T4R_LITERAL_DEPTH_RATIO_LIMIT = 1.10
P1_T4R_DERIVATION_DEPTH_ERROR_LIMIT = 0.05
P1_T4R_STEP_SENSITIVITY_LIMIT = 0.01
P1_T4R_WORKERS = 12

_POWER_FIELDS = (
    "gust_power_w",
    "thrust_power_w",
    "linear_drag_power_w",
    "angular_drag_power_w",
    "weather_power_w",
    "constant_body_force_power_w",
    "cable_damping_dissipation_w",
    "cable_clipping_dissipation_w",
)


@dataclass(frozen=True)
class P1T4RCellSpec:
    physics_step_s: float
    pretension_n: float
    gust_ratio: float
    entry_speed_mps: float
    branch: str

    @property
    def group_identifier(self) -> str:
        return (
            f"dt{1.0e6 * self.physics_step_s:.0f}_"
            f"p{self.pretension_n:.1f}_l{self.gust_ratio:.2f}_"
            f"u{self.entry_speed_mps:.2f}"
        )

    @property
    def identifier(self) -> str:
        return f"{self.group_identifier}_{self.branch}"

    @property
    def comparison_identifier(self) -> str:
        return (
            f"p{self.pretension_n:.1f}_l{self.gust_ratio:.2f}_"
            f"u{self.entry_speed_mps:.2f}_{self.branch}"
        )


@dataclass(frozen=True)
class P1T4RCell:
    spec: P1T4RCellSpec
    departure: FullFormationTrajectory
    turning_time_s: float
    turning_truth: np.ndarray
    trajectory: FullFormationTrajectory


def preregistered_cell_specs() -> tuple[P1T4RCellSpec, ...]:
    """Return the immutable matched-branch grid declared before scoring."""
    return tuple(
        P1T4RCellSpec(
            physics_step_s=float(physics_step),
            pretension_n=float(pretension),
            gust_ratio=float(gust_ratio),
            entry_speed_mps=float(entry_speed),
            branch=branch,
        )
        for physics_step in P1_T4R_PHYSICS_STEPS
        for pretension in PRETENSION_LEVELS
        for gust_ratio in P1_T4R_GUST_RATIOS
        for entry_speed in P1_T4R_ENTRY_SPEEDS
        for branch in P1_T4R_BRANCHES
    )


def _steady_tow(pretension_n: float) -> tuple[float, float]:
    common_speed = (
        constants.VESSEL_COUNT * pretension_n / constants.LOAD_LINEAR_DRAG
    )
    thrust = common_speed * (
        constants.LOAD_LINEAR_DRAG
        + constants.VESSEL_COUNT * constants.VESSEL_LINEAR_DRAG
    ) / constants.VESSEL_COUNT
    return thrust, common_speed


def _constant_balance_forces(pretension_n: float) -> np.ndarray:
    forces = np.zeros((constants.VESSEL_COUNT + 1, 2))
    forces[0, 0] = -constants.VESSEL_COUNT * pretension_n
    return forces


def _conservative_config(spec: P1T4RCellSpec, *, gust_enabled: bool) -> FullFormationConfig:
    return FullFormationConfig(
        thrusts=spec.pretension_n,
        pretension=spec.pretension_n,
        constant_body_forces=_constant_balance_forces(spec.pretension_n),
        differential_gust_n=spec.gust_ratio * spec.pretension_n,
        gust_duration_s=20.0,
        closure_depth_m=GEOMETRIC_DEPTH_LIMIT,
        time_step_s=spec.physics_step_s,
        recording_step_s=P1_T4R_RECORDING_STEP,
        drag_enabled=False,
        weather_enabled=False,
        gust_enabled=gust_enabled,
        cable_damping_n_s_per_m=0.0,
    )


def _confound_config(spec: P1T4RCellSpec) -> FullFormationConfig:
    thrust, _ = _steady_tow(spec.pretension_n)
    return FullFormationConfig(
        thrusts=thrust,
        pretension=spec.pretension_n,
        differential_gust_n=spec.gust_ratio * spec.pretension_n,
        gust_duration_s=20.0,
        closure_depth_m=GEOMETRIC_DEPTH_LIMIT,
        time_step_s=spec.physics_step_s,
        recording_step_s=P1_T4R_RECORDING_STEP,
        weather_enabled=False,
    )


def _turning_horizon(spec: P1T4RCellSpec) -> float:
    acceleration = (
        spec.pretension_n / REDUCED_MASS * max(1.0 - spec.gust_ratio, 0.05)
    )
    return max(0.25, 1.25 * spec.entry_speed_mps / acceleration + 0.05)


def _turning_from_trace(
    time: np.ndarray,
    truth: np.ndarray,
    elongation: np.ndarray,
    rate: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    candidates = np.flatnonzero(
        (rate[:-1] < 0.0)
        & (rate[1:] >= 0.0)
        & (elongation[:-1] < 0.0)
    )
    if not candidates.size:
        raise RuntimeError("departure did not reach a slack turning state")
    left = int(candidates[0])
    fraction = -rate[left] / (rate[left + 1] - rate[left])
    turning_time = time[left] + fraction * (time[left + 1] - time[left])
    turning_truth = truth[left] + fraction * (truth[left + 1] - truth[left])
    turning_depth = -(
        elongation[left] + fraction * (elongation[left + 1] - elongation[left])
    )
    return float(turning_time), turning_truth, float(turning_depth)


def _simulate_departure(
    spec: P1T4RCellSpec,
) -> tuple[FullFormationTrajectory, float, np.ndarray]:
    config = _conservative_config(spec, gust_enabled=True)
    _, common_speed = _steady_tow(spec.pretension_n)
    departure = run_full_formation(
        config,
        _turning_horizon(spec),
        common_speed=common_speed,
        target_relative_speed=-spec.entry_speed_mps,
        target_elongation=0.0,
    )
    cable = departure.target_cable
    turning_time, turning_truth, _ = _turning_from_trace(
        departure.time,
        departure.truth,
        departure.elongation[:, cable],
        departure.elongation_rate[:, cable],
    )
    return departure, turning_time, turning_truth


def _return_horizon(
    spec: P1T4RCellSpec,
    turning_depth: float,
) -> float:
    acceleration = spec.pretension_n / REDUCED_MASS
    if spec.branch != "isolated_unforced":
        acceleration *= max(1.0 - spec.gust_ratio, 0.05)
    return max(0.25, 1.35 * np.sqrt(2.0 * turning_depth / acceleration) + 0.05)


def _simulate_return(
    spec: P1T4RCellSpec,
    turning_truth: np.ndarray,
    turning_depth: float,
) -> FullFormationTrajectory:
    if spec.branch == "isolated_unforced":
        config = _conservative_config(spec, gust_enabled=False)
    elif spec.branch == "constant_force_control":
        config = _conservative_config(spec, gust_enabled=True)
    elif spec.branch == "forcing_active_confound":
        config = _confound_config(spec)
    else:
        raise ValueError(f"unknown P1-T4R branch: {spec.branch}")
    _, common_speed = _steady_tow(spec.pretension_n)
    return run_full_formation(
        config,
        _return_horizon(spec, turning_depth),
        common_speed=common_speed,
        initial_truth=turning_truth,
    )


def _run_group(spec: P1T4RCellSpec) -> tuple[P1T4RCell, ...]:
    departure, turning_time, turning_truth = _simulate_departure(spec)
    _, _, turning_depth = _turning_from_trace(
        departure.time,
        departure.truth,
        departure.elongation[:, departure.target_cable],
        departure.elongation_rate[:, departure.target_cable],
    )
    cells = []
    for branch in P1_T4R_BRANCHES:
        branch_spec = P1T4RCellSpec(
            physics_step_s=spec.physics_step_s,
            pretension_n=spec.pretension_n,
            gust_ratio=spec.gust_ratio,
            entry_speed_mps=spec.entry_speed_mps,
            branch=branch,
        )
        cells.append(
            P1T4RCell(
                spec=branch_spec,
                departure=departure,
                turning_time_s=turning_time,
                turning_truth=turning_truth.copy(),
                trajectory=_simulate_return(
                    branch_spec,
                    turning_truth,
                    turning_depth,
                ),
            )
        )
    return tuple(cells)


def run_cell(spec: P1T4RCellSpec) -> P1T4RCell:
    """Run one branch, including its measured full-plant departure."""
    departure, turning_time, turning_truth = _simulate_departure(spec)
    _, _, turning_depth = _turning_from_trace(
        departure.time,
        departure.truth,
        departure.elongation[:, departure.target_cable],
        departure.elongation_rate[:, departure.target_cable],
    )
    return P1T4RCell(
        spec=spec,
        departure=departure,
        turning_time_s=turning_time,
        turning_truth=turning_truth,
        trajectory=_simulate_return(spec, turning_truth, turning_depth),
    )


def run_grid(max_workers: int = P1_T4R_WORKERS) -> tuple[P1T4RCell, ...]:
    """Run each outbound group once, then its three matched return branches."""
    group_specs = preregistered_cell_specs()[:: len(P1_T4R_BRANCHES)]
    if max_workers == 1:
        groups = tuple(_run_group(spec) for spec in group_specs)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            groups = tuple(executor.map(_run_group, group_specs))
    return tuple(cell for group in groups for cell in group)


def _add_trace_arrays(
    arrays: dict[str, np.ndarray],
    prefix: str,
    trajectories: Sequence[FullFormationTrajectory],
) -> None:
    counts = np.array([trajectory.time.size for trajectory in trajectories], dtype=np.int64)
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts[:-1])))
    total = int(np.sum(counts))
    state_width = trajectories[0].truth.shape[1]
    arrays[f"{prefix}_sample_offset"] = offsets
    arrays[f"{prefix}_sample_count"] = counts
    arrays[f"{prefix}_time_s"] = np.empty(total)
    arrays[f"{prefix}_truth"] = np.empty((total, state_width))
    for name in (
        "all_tension_n",
        "all_elongation_m",
        "all_elongation_rate_mps",
        "all_relative_load_n",
    ):
        arrays[f"{prefix}_{name}"] = np.empty((total, constants.VESSEL_COUNT))
    arrays[f"{prefix}_kinetic_energy_j"] = np.empty(total)
    arrays[f"{prefix}_spring_potential_j"] = np.empty(total)
    for field in _POWER_FIELDS:
        arrays[f"{prefix}_{field}"] = np.empty(total)
    for index, trajectory in enumerate(trajectories):
        start = int(offsets[index])
        trace_slice = slice(start, start + trajectory.time.size)
        arrays[f"{prefix}_time_s"][trace_slice] = trajectory.time
        arrays[f"{prefix}_truth"][trace_slice] = trajectory.truth
        arrays[f"{prefix}_all_tension_n"][trace_slice] = trajectory.tension
        arrays[f"{prefix}_all_elongation_m"][trace_slice] = trajectory.elongation
        arrays[f"{prefix}_all_elongation_rate_mps"][trace_slice] = (
            trajectory.elongation_rate
        )
        arrays[f"{prefix}_all_relative_load_n"][trace_slice] = trajectory.relative_load
        arrays[f"{prefix}_kinetic_energy_j"][trace_slice] = trajectory.kinetic_energy_j
        arrays[f"{prefix}_spring_potential_j"][trace_slice] = (
            trajectory.spring_potential_j
        )
        for field in _POWER_FIELDS:
            arrays[f"{prefix}_{field}"][trace_slice] = getattr(trajectory, field)


def _add_event_arrays(
    arrays: dict[str, np.ndarray],
    prefix: str,
    trajectories: Sequence[FullFormationTrajectory],
) -> None:
    rows = [
        tuple(
            event
            for event in trajectory.event_records
            if event.cable == trajectory.target_cable
        )
        for trajectory in trajectories
    ]
    maximum = max(1, max(map(len, rows)))
    arrays[f"{prefix}_event_count"] = np.array(list(map(len, rows)), dtype=np.int64)
    arrays[f"{prefix}_event_kind"] = np.full((len(rows), maximum), "", dtype="U20")
    for name in (
        "time_s",
        "elongation_m",
        "elongation_rate_mps",
        "tension_n",
        "relative_load_n",
    ):
        arrays[f"{prefix}_event_{name}"] = np.full((len(rows), maximum), np.nan)
    for row_index, events in enumerate(rows):
        for event_index, event in enumerate(events):
            arrays[f"{prefix}_event_kind"][row_index, event_index] = event.kind
            arrays[f"{prefix}_event_time_s"][row_index, event_index] = event.time
            arrays[f"{prefix}_event_elongation_m"][row_index, event_index] = event.elongation
            arrays[f"{prefix}_event_elongation_rate_mps"][row_index, event_index] = (
                event.elongation_rate
            )
            arrays[f"{prefix}_event_tension_n"][row_index, event_index] = event.tension
            arrays[f"{prefix}_event_relative_load_n"][row_index, event_index] = (
                event.relative_load
            )


def record_arrays(cells: Sequence[P1T4RCell]) -> dict[str, np.ndarray]:
    """Archive unique departures and all matched returns with raw channels."""
    groups: list[P1T4RCell] = []
    group_indices: dict[str, int] = {}
    cell_group_indices = []
    for cell in cells:
        identifier = cell.spec.group_identifier
        if identifier not in group_indices:
            group_indices[identifier] = len(groups)
            groups.append(cell)
        cell_group_indices.append(group_indices[identifier])
    arrays: dict[str, np.ndarray] = {
        "group_identifier": np.array(
            [cell.spec.group_identifier for cell in groups], dtype="U48"
        ),
        "group_physics_step_s": np.array([cell.spec.physics_step_s for cell in groups]),
        "group_pretension_n": np.array([cell.spec.pretension_n for cell in groups]),
        "group_gust_ratio": np.array([cell.spec.gust_ratio for cell in groups]),
        "group_requested_entry_speed_mps": np.array(
            [cell.spec.entry_speed_mps for cell in groups]
        ),
        "group_archived_turning_time_s": np.array(
            [cell.turning_time_s for cell in groups]
        ),
        "group_archived_turning_truth": np.stack(
            [cell.turning_truth for cell in groups]
        ),
        "cell_identifier": np.array([cell.spec.identifier for cell in cells], dtype="U80"),
        "cell_comparison_identifier": np.array(
            [cell.spec.comparison_identifier for cell in cells], dtype="U72"
        ),
        "cell_group_index": np.array(cell_group_indices, dtype=np.int64),
        "cell_physics_step_s": np.array([cell.spec.physics_step_s for cell in cells]),
        "cell_pretension_n": np.array([cell.spec.pretension_n for cell in cells]),
        "cell_gust_ratio": np.array([cell.spec.gust_ratio for cell in cells]),
        "cell_requested_entry_speed_mps": np.array(
            [cell.spec.entry_speed_mps for cell in cells]
        ),
        "cell_branch": np.array([cell.spec.branch for cell in cells], dtype="U32"),
    }
    _add_trace_arrays(arrays, "departure_trace", [cell.departure for cell in groups])
    _add_event_arrays(arrays, "departure", [cell.departure for cell in groups])
    _add_trace_arrays(arrays, "trace", [cell.trajectory for cell in cells])
    _add_event_arrays(arrays, "return", [cell.trajectory for cell in cells])
    return arrays


def _trace_slice(record, prefix: str, index: int) -> slice:
    start = int(record[f"{prefix}_sample_offset"][index])
    count = int(record[f"{prefix}_sample_count"][index])
    return slice(start, start + count)


def _crossing(time: np.ndarray, elongation: np.ndarray, rate: np.ndarray):
    candidates = np.flatnonzero((elongation[:-1] <= 0.0) & (elongation[1:] > 0.0))
    if not candidates.size:
        return None, None
    left = int(candidates[0])
    fraction = -elongation[left] / (elongation[left + 1] - elongation[left])
    return (
        float(time[left] + fraction * (time[left + 1] - time[left])),
        float(rate[left] + fraction * (rate[left + 1] - rate[left])),
    )


def _integrate_to(time: np.ndarray, values: np.ndarray, end_time: float) -> float:
    interior = time < end_time
    integration_time = np.concatenate((time[interior], [end_time]))
    integration_values = np.concatenate(
        (values[interior], [np.interp(end_time, time, values)])
    )
    return float(np.trapezoid(integration_values, integration_time))


def _state_channels_consistent(
    truth: np.ndarray,
    elongation: np.ndarray,
    rate: np.ndarray,
) -> bool:
    cable = constants.VESSEL_COUNT // 2
    position_count = truth.shape[1] // 2
    target_position = 3 * (cable + 1)
    state_elongation = (
        truth[:, target_position] - truth[:, 0] - constants.CABLE_REST_LENGTH
    )
    state_rate = truth[:, position_count + target_position] - truth[:, position_count]
    return bool(
        np.allclose(state_elongation, elongation, rtol=0.0, atol=1.0e-10)
        and np.allclose(state_rate, rate, rtol=0.0, atol=1.0e-10)
    )


def _work_channels(record, prefix: str, trace_slice: slice, end_time: float):
    time = record[f"{prefix}_time_s"][trace_slice]
    return {
        field.replace("_power_w", "_work_j").replace(
            "_dissipation_w", "_dissipation_j"
        ): _integrate_to(time, record[f"{prefix}_{field}"][trace_slice], end_time)
        for field in _POWER_FIELDS
    }


def _measure_group(record, group_index: int) -> dict[str, object]:
    trace_slice = _trace_slice(record, "departure_trace", group_index)
    time = record["departure_trace_time_s"][trace_slice]
    truth = record["departure_trace_truth"][trace_slice]
    cable = constants.VESSEL_COUNT // 2
    elongation = record["departure_trace_all_elongation_m"][trace_slice, cable]
    rate = record["departure_trace_all_elongation_rate_mps"][trace_slice, cable]
    turning_time, turning_truth, turning_depth = _turning_from_trace(
        time, truth, elongation, rate
    )
    event_count = int(record["departure_event_count"][group_index])
    event_kinds = record["departure_event_kind"][group_index, :event_count]
    event_times = record["departure_event_time_s"][group_index, :event_count]
    turning_events = event_times[event_kinds == "turning"]
    entry_speed = max(0.0, -float(rate[0]))
    valid = bool(
        abs(elongation[0]) <= 1.0e-10
        and entry_speed > 0.0
        and turning_time > 0.0
        and turning_events.size == 1
        and abs(turning_events[0] - turning_time) <= 1.0e-9
        and _state_channels_consistent(truth, elongation, rate)
        and np.allclose(
            record["group_archived_turning_truth"][group_index],
            turning_truth,
            rtol=0.0,
            atol=1.0e-10,
        )
        and np.isclose(
            record["group_archived_turning_time_s"][group_index],
            turning_time,
            rtol=0.0,
            atol=1.0e-10,
        )
    )
    return {
        "identifier": str(record["group_identifier"][group_index]),
        "physics_step_s": float(record["group_physics_step_s"][group_index]),
        "pretension_n": float(record["group_pretension_n"][group_index]),
        "gust_ratio": float(record["group_gust_ratio"][group_index]),
        "requested_entry_speed_mps": float(
            record["group_requested_entry_speed_mps"][group_index]
        ),
        "measured_entry_speed_mps": entry_speed,
        "turning_time_s": turning_time,
        "turning_depth_m": turning_depth,
        "turning_truth": turning_truth,
        "departure_event_order_valid": valid,
        "departure_work_channels_j": _work_channels(
            record, "departure_trace", trace_slice, turning_time
        ),
    }


def _measure_cell(
    record,
    cell_index: int,
    group: dict[str, object],
) -> dict[str, object]:
    trace_slice = _trace_slice(record, "trace", cell_index)
    time = record["trace_time_s"][trace_slice]
    truth = record["trace_truth"][trace_slice]
    cable = constants.VESSEL_COUNT // 2
    elongation = record["trace_all_elongation_m"][trace_slice, cable]
    rate = record["trace_all_elongation_rate_mps"][trace_slice, cable]
    up_time, return_speed = _crossing(time, elongation, rate)
    event_count = int(record["return_event_count"][cell_index])
    event_kinds = record["return_event_kind"][cell_index, :event_count]
    event_times = record["return_event_time_s"][cell_index, :event_count]
    geometric_up = event_times[event_kinds == "geometric_up"]
    force_onset = event_times[event_kinds == "force_onset"]
    state_consistent = bool(
        np.allclose(truth[0], group["turning_truth"], rtol=0.0, atol=1.0e-10)
        and _state_channels_consistent(truth, elongation, rate)
    )
    event_valid = bool(
        group["departure_event_order_valid"]
        and up_time is not None
        and abs(rate[0]) <= 1.0e-10
        and geometric_up.size == 1
        and force_onset.size == 1
        and 0.0 < geometric_up[0] <= force_onset[0] + 1.0e-12
        and abs(geometric_up[0] - up_time) <= 1.0e-9
        and state_consistent
    )
    entry_speed = float(group["measured_entry_speed_mps"])
    pretension = float(group["pretension_n"])
    gust_ratio = float(group["gust_ratio"])
    literal_depth = entry_speed**2 / (2.0 * pretension / REDUCED_MASS)
    derivation_depth = literal_depth / (1.0 - gust_ratio)
    speed_error = (
        np.inf if return_speed is None else abs(return_speed / entry_speed - 1.0)
    )
    return {
        "identifier": str(record["cell_identifier"][cell_index]),
        "comparison_identifier": str(
            record["cell_comparison_identifier"][cell_index]
        ),
        "group_identifier": group["identifier"],
        "physics_step_s": float(record["cell_physics_step_s"][cell_index]),
        "pretension_n": pretension,
        "gust_ratio": gust_ratio,
        "requested_entry_speed_mps": float(
            record["cell_requested_entry_speed_mps"][cell_index]
        ),
        "measured_entry_speed_mps": entry_speed,
        "branch": str(record["cell_branch"][cell_index]),
        "turning_time_s": float(group["turning_time_s"]),
        "geometric_up_time_s": up_time,
        "force_onset_time_s": float(force_onset[0]) if force_onset.size == 1 else None,
        "return_speed_mps": return_speed,
        "measured_turning_depth_m": float(group["turning_depth_m"]),
        "literal_ballistic_depth_m": literal_depth,
        "derivation_constant_force_depth_m": derivation_depth,
        "speed_relative_error": float(speed_error),
        "literal_depth_ratio": float(group["turning_depth_m"]) / literal_depth,
        "derivation_depth_relative_error": abs(
            float(group["turning_depth_m"]) / derivation_depth - 1.0
        ),
        "initial_turning_rate_mps": float(rate[0]),
        "event_order_and_return_valid": event_valid,
        "state_channels_consistent": state_consistent,
        "departure_work_channels_j": group["departure_work_channels_j"],
        "return_work_channels_j": (
            {}
            if up_time is None
            else _work_channels(record, "trace", trace_slice, up_time)
        ),
    }


def _worst(measurements: Sequence[dict[str, object]], field: str) -> dict[str, object]:
    row = max(measurements, key=lambda item: float(item[field]))
    return {"identifier": row["identifier"], field: row[field]}


def analyse_record(record) -> dict[str, object]:
    """Recompute all P1-T4R metrics from raw outbound and return channels."""
    groups = [
        _measure_group(record, index)
        for index in range(record["group_identifier"].size)
    ]
    measurements = [
        _measure_cell(
            record,
            index,
            groups[int(record["cell_group_index"][index])],
        )
        for index in range(record["cell_identifier"].size)
    ]
    production = [
        row
        for row in measurements
        if np.isclose(row["physics_step_s"], P1_T4R_PRODUCTION_STEP)
    ]
    literal = [row for row in production if row["branch"] == "isolated_unforced"]
    derivation = [
        row for row in production if row["branch"] == "constant_force_control"
    ]
    literal_checks = {
        "every_speed_error_below_5_percent": all(
            row["speed_relative_error"] < P1_T4R_SPEED_ERROR_LIMIT for row in literal
        ),
        "every_depth_below_1_1_ballistic": all(
            row["literal_depth_ratio"] < P1_T4R_LITERAL_DEPTH_RATIO_LIMIT
            for row in literal
        ),
        "every_event_order_and_return_valid": all(
            row["event_order_and_return_valid"] for row in literal
        ),
    }
    derivation_checks = {
        "every_speed_error_below_5_percent": all(
            row["speed_relative_error"] < P1_T4R_SPEED_ERROR_LIMIT
            for row in derivation
        ),
        "every_depth_error_at_most_5_percent": all(
            row["derivation_depth_relative_error"]
            <= P1_T4R_DERIVATION_DEPTH_ERROR_LIMIT
            for row in derivation
        ),
        "every_literal_table_depth_below_1_1_ballistic": all(
            row["literal_depth_ratio"] < P1_T4R_LITERAL_DEPTH_RATIO_LIMIT
            for row in derivation
        ),
        "every_event_order_and_return_valid": all(
            row["event_order_and_return_valid"] for row in derivation
        ),
    }
    by_step = {
        (row["comparison_identifier"], row["physics_step_s"]): row
        for row in measurements
    }
    comparisons = []
    for identifier in sorted({row["comparison_identifier"] for row in measurements}):
        coarse = by_step[(identifier, P1_T4R_PHYSICS_STEPS[0])]
        fine = by_step[(identifier, P1_T4R_PHYSICS_STEPS[1])]
        return_speeds = (coarse["return_speed_mps"], fine["return_speed_mps"])
        speed_errors = (
            coarse["speed_relative_error"],
            fine["speed_relative_error"],
        )
        comparison_valid = all(value is not None for value in return_speeds)
        comparisons.append(
            {
                "comparison_identifier": identifier,
                "comparison_valid": comparison_valid,
                "speed_error_percentage_point_difference": (
                    abs(speed_errors[0] - speed_errors[1])
                    if comparison_valid
                    else np.inf
                ),
                "return_speed_relative_difference": (
                    abs(return_speeds[0] / return_speeds[1] - 1.0)
                    if comparison_valid
                    else np.inf
                ),
            }
        )
    maximum_step_difference = max(
        row["speed_error_percentage_point_difference"] for row in comparisons
    )
    step_passed = maximum_step_difference <= P1_T4R_STEP_SENSITIVITY_LIMIT
    controls = {}
    for branch in P1_T4R_BRANCHES:
        selected = [row for row in production if row["branch"] == branch]
        controls[branch] = {
            "cell_count": len(selected),
            "maximum_speed_relative_error": max(
                row["speed_relative_error"] for row in selected
            ),
            "worst_speed_cell": _worst(selected, "speed_relative_error"),
            "invalid_event_or_return_cells": [
                row["identifier"]
                for row in selected
                if not row["event_order_and_return_valid"]
            ],
        }
    literal_passed = all(literal_checks.values()) and step_passed
    derivation_core_checks = {
        key: value
        for key, value in derivation_checks.items()
        if key != "every_literal_table_depth_below_1_1_ballistic"
    }
    derivation_passed = all(derivation_core_checks.values()) and step_passed
    exact_table_on_derivation_passed = (
        derivation_passed
        and derivation_checks["every_literal_table_depth_below_1_1_ballistic"]
    )
    return {
        "status": "PASS" if derivation_passed else "FAIL",
        "primary_interpretation": "derivation_constant_force",
        "literal_table_interpretation": {
            "cell_count": len(literal),
            "checks": literal_checks,
            "maximum_speed_relative_error": max(
                row["speed_relative_error"] for row in literal
            ),
            "maximum_literal_depth_ratio": max(
                row["literal_depth_ratio"] for row in literal
            ),
            "worst_speed_cell": _worst(literal, "speed_relative_error"),
            "worst_depth_cell": _worst(literal, "literal_depth_ratio"),
            "status": "PASS" if literal_passed else "FAIL",
        },
        "derivation_interpretation": {
            "cell_count": len(derivation),
            "checks": derivation_checks,
            "maximum_speed_relative_error": max(
                row["speed_relative_error"] for row in derivation
            ),
            "maximum_derivation_depth_relative_error": max(
                row["derivation_depth_relative_error"] for row in derivation
            ),
            "maximum_literal_depth_ratio": max(
                row["literal_depth_ratio"] for row in derivation
            ),
            "worst_speed_cell": _worst(derivation, "speed_relative_error"),
            "worst_derivation_depth_cell": _worst(
                derivation, "derivation_depth_relative_error"
            ),
            "worst_literal_depth_cell": _worst(derivation, "literal_depth_ratio"),
            "core_status": "PASS" if derivation_passed else "FAIL",
            "exact_table_status": (
                "PASS" if exact_table_on_derivation_passed else "FAIL"
            ),
        },
        "physics_step_sensitivity": {
            "limit_percentage_points": P1_T4R_STEP_SENSITIVITY_LIMIT,
            "maximum_speed_error_percentage_point_difference": maximum_step_difference,
            "worst_comparison": max(
                comparisons,
                key=lambda row: row["speed_error_percentage_point_difference"],
            ),
            "comparisons": comparisons,
            "status": "PASS" if step_passed else "FAIL",
        },
        "controls": controls,
        "departure_groups": [
            {key: value for key, value in group.items() if key != "turning_truth"}
            for group in groups
        ],
        "measurements": measurements,
    }
