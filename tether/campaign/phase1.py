"""Execute the implemented Phase 1 Stage 1-2 mechanics campaign."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
from pathlib import Path
import subprocess
import tempfile
import time
import zipfile

import numpy as np

from tether.analysis.phase1_figures import (
    DEFAULT_PHASE1R_F1,
    DEFAULT_PHASE1R_F2,
    generate_phase1r_figures,
)
from tether.physics import constants
from tether.physics.phase1_mechanics import (
    ANALYTIC_IMPEDANCE,
    IMPACT_FACTOR,
    IMPACT_SPEEDS,
    PINNED_IMPEDANCE,
    REFINEMENT_STEPS,
    mechanics_acceptance,
)
from tether.physics.phase1_deterministic import (
    FLEET_IMPACT_SPEEDS,
    GEOMETRIC_DEPTH_LIMIT,
    GUST_DURATIONS,
    GUST_RATIOS,
    PINNED_FLEET_IMPEDANCE,
    PRETENSION_LEVELS,
    PRIMARY_ENTRY_SPEED,
    PHASE1R_ENTRY_SPEEDS,
    PHASE1R_ENERGY_BALANCE_TOLERANCE,
    PHASE1R_GUST_DURATIONS,
    PHASE1R_GUST_RATIOS,
    PHASE1R_HORIZON,
    PHASE1R_IMPACT_MAX_SPEED,
    PHASE1R_IMPACT_MIN_MARKS,
    PHASE1R_IMPACT_MIN_SPEED,
    PHASE1R_INITIAL_EXTENSION,
    PHASE1R_T6_FIT_WINDOWS,
    PHASE1R_T6_PRIMARY_STEP,
    PHASE1R_T6_PRIMARY_WINDOW,
    PHASE1R_T6_PROBE_STEPS,
    PHASE1R_T6_SLACK_EXTENSION,
    PHASE1R_WORKERS,
    PINNED_FLEET_EFFECTIVE_MASS,
    SENSITIVITY_ENTRY_SPEEDS,
    FullFormationConfig,
    deterministic_fleet_acceptance,
    phase1r_excursion_acceptance,
    run_full_formation,
    scripted_acceptance,
)
from tether.physics.plant import steady_state_values

ROOT = Path(__file__).resolve().parents[2]
RECORD_DIR = ROOT / "records" / "phase1"
RESULTS_PATH = RECORD_DIR / "phase1_mechanics.json"
REPORT_PATH = ROOT / "reports" / "phase1_mechanics_report.md"
DETERMINISTIC_RESULTS_PATH = RECORD_DIR / "phase1_deterministic.json"
DETERMINISTIC_RECORD_PATH = RECORD_DIR / "phase1_deterministic.npz"
DETERMINISTIC_REPORT_PATH = ROOT / "reports" / "phase1_deterministic_report.md"
PHASE1R_RESULTS_PATH = RECORD_DIR / "phase1r_results.json"
PHASE1R_RECORD_PATH = RECORD_DIR / "phase1r_records.npz"
PHASE1R_MANIFEST_PATH = RECORD_DIR / "phase1r_manifest.json"
PHASE1R_REPORT_PATH = ROOT / "reports" / "phase1r_report.md"
PHASE1R_SEEDS = (4102,)
PHASE1R_REPLAY_NUMERIC_ATOL = 1.0e-6
REMAINING_TESTS = (
    "P1-T10 excursion duration",
)


def _json_bytes(value: object) -> bytes:
    def numpy_scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    return (
        json.dumps(value, indent=2, sort_keys=True, default=numpy_scalar) + "\n"
    ).encode("ascii")


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer, mode="w", compression=zipfile.ZIP_STORED
    ) as archive:
        for name in sorted(arrays):
            array_buffer = io.BytesIO()
            np.lib.format.write_array(
                array_buffer, np.asarray(arrays[name]), allow_pickle=False
            )
            member = zipfile.ZipInfo(
                f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            member.compress_type = zipfile.ZIP_STORED
            member.external_attr = 0o600 << 16
            archive.writestr(member, array_buffer.getvalue())
    return archive_buffer.getvalue()


def _phase1r_configuration() -> dict[str, object]:
    return {
        "masses_kg": {
            "load": constants.LOAD_MASS,
            "vessel": constants.VESSEL_MASS,
        },
        "yaw_inertias_kg_m2": {
            "load": constants.LOAD_YAW_INERTIA,
            "vessel": constants.VESSEL_YAW_INERTIA,
        },
        "linear_drag_n_s_per_m": {
            "load": constants.LOAD_LINEAR_DRAG,
            "vessel": constants.VESSEL_LINEAR_DRAG,
        },
        "angular_drag_n_m_s": {
            "load": constants.LOAD_ANGULAR_DRAG,
            "vessel": constants.VESSEL_ANGULAR_DRAG,
        },
        "nominal_thrust_n": constants.NOMINAL_THRUST,
        "thrusts_n_per_vessel": [
            (
                constants.VESSEL_COUNT
                * pretension
                / constants.LOAD_LINEAR_DRAG
            )
            * (
                constants.LOAD_LINEAR_DRAG
                + constants.VESSEL_COUNT * constants.VESSEL_LINEAR_DRAG
            )
            / constants.VESSEL_COUNT
            for pretension in PRETENSION_LEVELS
        ],
        "vessel_count": constants.VESSEL_COUNT,
        "cable": {
            "rest_length_m": constants.CABLE_REST_LENGTH,
            "stiffness_n_per_m": constants.CABLE_STIFFNESS,
            "damping_n_s_per_m": constants.CABLE_DAMPING,
            "mode": "recording",
            "closure_depth_m": GEOMETRIC_DEPTH_LIMIT,
        },
        "grids": {
            "pretensions_n": PRETENSION_LEVELS.tolist(),
            "gust_ratios": PHASE1R_GUST_RATIOS.tolist(),
            "gust_durations_s": PHASE1R_GUST_DURATIONS.tolist(),
            "entry_speeds_mps": PHASE1R_ENTRY_SPEEDS.tolist(),
            "impact_speeds_mps": FLEET_IMPACT_SPEEDS.tolist(),
            "mechanics_refinement_steps_s": list(REFINEMENT_STEPS),
        },
        "p1_t6_probe": {
            "initial_target_elongation_m": PHASE1R_T6_SLACK_EXTENSION,
            "initial_target_relative_speed_mps": 0.0,
            "fit_windows_s": list(PHASE1R_T6_FIT_WINDOWS),
            "time_steps_s": list(PHASE1R_T6_PROBE_STEPS),
            "primary_fit_window_s": PHASE1R_T6_PRIMARY_WINDOW,
            "primary_time_step_s": PHASE1R_T6_PRIMARY_STEP,
        },
        "durations_s": {
            "fleet_regression": 0.25,
            "mechanics_impact": 0.25,
            "full_plant_excursion_horizon": PHASE1R_HORIZON,
            "gusts": PHASE1R_GUST_DURATIONS.tolist(),
            "t6_probe": max(PHASE1R_T6_FIT_WINDOWS),
        },
        "thresholds": {
            "energy_balance_relative_residual": PHASE1R_ENERGY_BALANCE_TOLERANCE,
            "impact_calibration_min_speed_mps": PHASE1R_IMPACT_MIN_SPEED,
            "impact_calibration_max_speed_mps": PHASE1R_IMPACT_MAX_SPEED,
            "impact_calibration_min_marks": PHASE1R_IMPACT_MIN_MARKS,
            "impact_slope_relative_error": 0.06,
            "t6_lambda_relative_error": 0.05,
            "depth_relative_error": 0.15,
            "submetre_depth_m": 1.0,
            "p1_t9_relative_error": 0.005,
        },
        "acceptance_criteria": {
            "P1-T1": "distance error <= 1%, tension error <= 3%, no slack",
            "P1-T2": "uncentred R2 > 0.99 and analytic impedance error <= 15%",
            "P1-T2b": "uncentred R2 > 0.99 with other cables taut",
            "P1-T3": "reduced-mass relative error <= 10%",
            "P1-T4": "not rerun; nonblocking in corrected Phase 1R gate",
            "P1-T5_full_plant_energy": (
                f"maximum normalized balance residual <= "
                f"{PHASE1R_ENERGY_BALANCE_TOLERANCE}"
            ),
            "P1-T5_impact": (
                f"at least {PHASE1R_IMPACT_MIN_MARKS} marks in "
                f"[{PHASE1R_IMPACT_MIN_SPEED}, {PHASE1R_IMPACT_MAX_SPEED}] m/s "
                "and in-range slope within 6%; rejection denotes a low-speed "
                "through-origin model/specification failure"
            ),
            "P1-T6": (
                "primary true-slack estimator brackets 1 and "
                "abs(lambda_c - 1) <= 0.05"
            ),
            "P1-T7": "both eligible depth errors <= 15% below 1 m",
            "P1-T8": "finite-grid observation only; no global PASS claim",
            "P1-T9": (
                "discrete 1 ms energy, body impulse, and total momentum "
                "relative errors <= 0.5%; continuous refinement also passes"
            ),
            "P1-T10": "not run; nonblocking in corrected Phase 1R gate",
        },
        "seeds": list(PHASE1R_SEEDS),
        "pinned_impedances_n_s_per_m": {
            "isolated": PINNED_IMPEDANCE,
            "fleet": PINNED_FLEET_IMPEDANCE,
        },
        "pinned_fleet_effective_mass_kg": PINNED_FLEET_EFFECTIVE_MASS,
        "full_plant_integrator": "Drake discrete MultibodyPlant SAP",
        "mechanics_integrator": "continuous fixed-step RK3 plus discrete SAP production check",
        "time_step_s": constants.TIME_STEP,
        "event_period_s": constants.TIME_STEP,
        "full_plant_horizon_s": PHASE1R_HORIZON,
        "grid_workers": PHASE1R_WORKERS,
        "initial_target_extension_m": PHASE1R_INITIAL_EXTENSION,
        "target_cable": 2,
        "stochastic_production_grid": "NOT_RUN",
    }


def _configuration_sha256(configuration: dict[str, object]) -> str:
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _source_state() -> dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "status": "unavailable"}
    return {
        "revision": revision,
        "status": "dirty" if status.strip() else "clean",
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recorded_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _phase1r_record_arrays(
    outcomes,
    excursion_tests: dict[str, object],
) -> dict[str, np.ndarray]:
    def optional(name: str) -> np.ndarray:
        return np.array(
            [
                np.nan if getattr(outcome, name) is None else getattr(outcome, name)
                for outcome in outcomes
            ],
            dtype=float,
        )

    arrays = {
        "pinned_fleet_effective_mass_kg": np.array(
            [PINNED_FLEET_EFFECTIVE_MASS]
        ),
        "cell_pretension_n": np.array([item.pretension_n for item in outcomes]),
        "cell_gust_ratio": np.array([item.gust_ratio for item in outcomes]),
        "cell_gust_duration_s": np.array(
            [item.gust_duration_s for item in outcomes]
        ),
        "cell_requested_entry_speed_mps": np.array(
            [item.requested_entry_speed_mps for item in outcomes]
        ),
        "cell_status": np.array([item.status for item in outcomes], dtype="U16"),
        "cell_censor_reason": np.array(
            [item.censor_reason or "" for item in outcomes], dtype="U40"
        ),
        "cell_geometric_down_time_s": optional("geometric_down_time_s"),
        "cell_geometric_up_time_s": optional("geometric_up_time_s"),
        "cell_measured_entry_speed_mps": optional("measured_entry_speed_mps"),
        "cell_depth_at_gust_shutoff_m": np.array(
            [item.measured_depth_at_gust_shutoff_m for item in outcomes]
        ),
        "cell_maximum_depth_m": np.array(
            [item.measured_maximum_depth_m for item in outcomes]
        ),
        "cell_return_speed_mps": optional("measured_return_speed_mps"),
        "cell_peak_tension_n": optional("measured_peak_tension_n"),
        "cell_gust_work_j": np.array(
            [item.measured_gust_work_j for item in outcomes]
        ),
        "cell_slack_acceleration_mps2": optional(
            "measured_slack_acceleration_mps2"
        ),
        "cell_reduced_shutoff_depth_prediction_m": optional(
            "reduced_shutoff_depth_prediction_m"
        ),
        "cell_amended_maximum_depth_prediction_m": optional(
            "amended_maximum_depth_prediction_m"
        ),
        "cell_impedance_extrapolation": np.array(
            [item.impedance_extrapolation for item in outcomes], dtype=bool
        ),
    }
    balance_fields = (
        "window_start_s",
        "window_end_s",
        "delta_kinetic_j",
        "delta_spring_potential_j",
        "delta_mechanical_energy_j",
        "gust_work_j",
        "thrust_work_j",
        "linear_drag_work_j",
        "angular_drag_work_j",
        "weather_work_j",
        "cable_damping_dissipation_j",
        "cable_clipping_dissipation_j",
        "signed_nonconservative_work_j",
        "residual_j",
        "normalization_j",
        "normalized_residual",
    )
    for field in balance_fields:
        arrays[f"cell_energy_balance_{field}"] = np.array(
            [
                np.nan
                if item.energy_balance is None
                else item.energy_balance[field]
                for item in outcomes
            ]
        )
    sample_count = np.array(
        [item.trajectory.time.size for item in outcomes], dtype=np.int64
    )
    maximum_samples = int(np.max(sample_count))
    arrays["trace_sample_count"] = sample_count
    for name in (
        "trace_time_s",
        "trace_kinetic_energy_j",
        "trace_spring_potential_j",
        "trace_gust_power_w",
        "trace_thrust_power_w",
        "trace_linear_drag_power_w",
        "trace_angular_drag_power_w",
        "trace_weather_power_w",
        "trace_cable_damping_dissipation_w",
        "trace_cable_clipping_dissipation_w",
        "trace_target_elongation_m",
        "trace_target_elongation_rate_mps",
        "trace_target_tension_n",
        "trace_target_relative_load_n",
    ):
        arrays[name] = np.full((len(outcomes), maximum_samples), np.nan)
    for cell_index, outcome in enumerate(outcomes):
        trajectory = outcome.trajectory
        count = trajectory.time.size
        cable = trajectory.target_cable
        arrays["trace_time_s"][cell_index, :count] = trajectory.time
        arrays["trace_target_elongation_m"][cell_index, :count] = (
            trajectory.elongation[:, cable]
        )
        arrays["trace_target_elongation_rate_mps"][cell_index, :count] = (
            trajectory.elongation_rate[:, cable]
        )
        arrays["trace_target_tension_n"][cell_index, :count] = (
            trajectory.tension[:, cable]
        )
        arrays["trace_target_relative_load_n"][cell_index, :count] = (
            trajectory.relative_load[:, cable]
        )
        arrays["trace_kinetic_energy_j"][cell_index, :count] = (
            trajectory.kinetic_energy_j
        )
        arrays["trace_spring_potential_j"][cell_index, :count] = (
            trajectory.spring_potential_j
        )
        arrays["trace_gust_power_w"][cell_index, :count] = trajectory.gust_power_w
        arrays["trace_thrust_power_w"][cell_index, :count] = (
            trajectory.thrust_power_w
        )
        arrays["trace_linear_drag_power_w"][cell_index, :count] = (
            trajectory.linear_drag_power_w
        )
        arrays["trace_angular_drag_power_w"][cell_index, :count] = (
            trajectory.angular_drag_power_w
        )
        arrays["trace_weather_power_w"][cell_index, :count] = (
            trajectory.weather_power_w
        )
        arrays["trace_cable_damping_dissipation_w"][cell_index, :count] = (
            trajectory.cable_damping_dissipation_w
        )
        arrays["trace_cable_clipping_dissipation_w"][cell_index, :count] = (
            trajectory.cable_clipping_dissipation_w
        )

    state_width = outcomes[0].trajectory.truth.shape[1]
    arrays["trace_truth"] = np.full(
        (len(outcomes), maximum_samples, state_width), np.nan
    )
    for name in (
        "trace_all_tension_n",
        "trace_all_elongation_m",
        "trace_all_elongation_rate_mps",
        "trace_all_relative_load_n",
    ):
        arrays[name] = np.full(
            (len(outcomes), maximum_samples, constants.VESSEL_COUNT), np.nan
        )
    for cell_index, outcome in enumerate(outcomes):
        trajectory = outcome.trajectory
        count = trajectory.time.size
        arrays["trace_truth"][cell_index, :count] = trajectory.truth
        arrays["trace_all_tension_n"][cell_index, :count] = trajectory.tension
        arrays["trace_all_elongation_m"][cell_index, :count] = trajectory.elongation
        arrays["trace_all_elongation_rate_mps"][cell_index, :count] = (
            trajectory.elongation_rate
        )
        arrays["trace_all_relative_load_n"][cell_index, :count] = (
            trajectory.relative_load
        )

    probes = excursion_tests["P1-T6"]["probe_measurements"]
    arrays["t6_primary_time_step_s"] = np.array([PHASE1R_T6_PRIMARY_STEP])
    arrays["t6_primary_fit_window_s"] = np.array([PHASE1R_T6_PRIMARY_WINDOW])
    for key, suffix in (
        ("pretension_n", "pretension_n"),
        ("gust_ratio", "gust_ratio"),
        ("time_step_s", "time_step_s"),
        ("fit_window_s", "fit_window_s"),
        ("initial_elongation_m", "initial_elongation_m"),
        ("initial_tension_n", "initial_tension_n"),
        ("maximum_elongation_m", "maximum_elongation_m"),
        ("maximum_tension_n", "maximum_tension_n"),
        ("acceleration_mps2", "acceleration_mps2"),
    ):
        arrays[f"t6_probe_{suffix}"] = np.array([row[key] for row in probes])

    event_rows = [
        (cell_index, record)
        for cell_index, outcome in enumerate(outcomes)
        for record in outcome.trajectory.event_records
    ]
    arrays.update(
        {
            "event_cell_index": np.array(
                [row[0] for row in event_rows], dtype=np.int64
            ),
            "event_kind": np.array(
                [row[1].kind for row in event_rows], dtype="U20"
            ),
            "event_time_s": np.array([row[1].time for row in event_rows]),
            "event_cable": np.array(
                [row[1].cable for row in event_rows], dtype=np.int64
            ),
            "event_elongation_m": np.array(
                [row[1].elongation for row in event_rows]
            ),
            "event_elongation_rate_mps": np.array(
                [row[1].elongation_rate for row in event_rows]
            ),
            "event_tension_n": np.array(
                [row[1].tension for row in event_rows]
            ),
            "event_relative_load_n": np.array(
                [row[1].relative_load for row in event_rows]
            ),
        }
    )
    reengagement_rows = [
        (cell_index, record)
        for cell_index, outcome in enumerate(outcomes)
        for record in outcome.trajectory.reengagement_records
    ]
    arrays.update(
        {
            "reengagement_cell_index": np.array(
                [row[0] for row in reengagement_rows], dtype=np.int64
            ),
            "reengagement_cable": np.array(
                [row[1].cable for row in reengagement_rows], dtype=np.int64
            ),
            "reengagement_time_s": np.array(
                [row[1].t_up for row in reengagement_rows]
            ),
            "reengagement_entry_speed_mps": np.array(
                [row[1].u_entry for row in reengagement_rows]
            ),
            "reengagement_depth_m": np.array(
                [row[1].depth for row in reengagement_rows]
            ),
            "reengagement_return_speed_mps": np.array(
                [row[1].v_return for row in reengagement_rows]
            ),
            "reengagement_peak_tension_n": np.array(
                [row[1].T_peak for row in reengagement_rows]
            ),
            "reengagement_relative_load_onset_n": np.array(
                [row[1].W_rel_at_onset for row in reengagement_rows]
            ),
            "reengagement_relative_load_max_n": np.array(
                [row[1].W_rel_max for row in reengagement_rows]
            ),
        }
    )
    taut_rows = [
        (cell_index, record)
        for cell_index, outcome in enumerate(outcomes)
        for record in outcome.trajectory.taut_records
    ]
    arrays.update(
        {
            "taut_cell_index": np.array(
                [row[0] for row in taut_rows], dtype=np.int64
            ),
            "taut_cable": np.array(
                [row[1].cable for row in taut_rows], dtype=np.int64
            ),
            "taut_start_time_s": np.array([row[1].t_start for row in taut_rows]),
            "taut_end_time_s": np.array([row[1].t_end for row in taut_rows]),
            "taut_peak_n": np.array([row[1].q_peak for row in taut_rows]),
        }
    )
    return arrays


def _metric_subset_matches(computed: object, reported: object) -> bool:
    if isinstance(computed, dict):
        return isinstance(reported, dict) and all(
            key in reported and _metric_subset_matches(value, reported[key])
            for key, value in computed.items()
        )
    if isinstance(computed, (list, tuple)):
        return isinstance(reported, (list, tuple)) and len(computed) == len(
            reported
        ) and all(
            _metric_subset_matches(first, second)
            for first, second in zip(computed, reported)
        )
    if isinstance(computed, (float, int, np.number)) and not isinstance(
        computed, (bool, np.bool_)
    ):
        return bool(
            np.isclose(
                float(computed),
                float(reported),
                rtol=1.0e-12,
                atol=PHASE1R_REPLAY_NUMERIC_ATOL,
                equal_nan=True,
            )
        )
    return computed == reported


def _first_sampled_peak(values: np.ndarray) -> float:
    for index in range(1, values.size):
        if values[index] < values[index - 1]:
            return float(values[index - 1])
    raise RuntimeError("record ended before a sampled peak")


def _record_origin_fit(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    slope = float(np.dot(x_values, y_values) / np.dot(x_values, x_values))
    residual = y_values - slope * x_values
    r_squared = float(
        1.0 - np.dot(residual, residual) / np.dot(y_values, y_values)
    )
    return slope, r_squared


def _record_threshold(x_values: np.ndarray, y_values: np.ndarray) -> dict[str, float]:
    beta, alpha = np.polyfit(x_values, y_values, 1)
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "lambda_c": float(-alpha / beta),
    }


def _record_conservation_metrics(
    record,
    prefix: str,
    *,
    discrete: bool,
) -> dict[str, object]:
    time_values = record[f"{prefix}_time_s"]
    load_velocity = record[f"{prefix}_load_velocity_mps"]
    vessel_velocity = record[f"{prefix}_vessel_velocity_mps"]
    load_force = record[f"{prefix}_load_force_n"]
    vessel_force = record[f"{prefix}_vessel_force_n"]
    step = float(time_values[1] - time_values[0])
    load_delta = constants.LOAD_MASS * (load_velocity[-1] - load_velocity[0])
    vessel_delta = constants.VESSEL_MASS * (
        vessel_velocity[-1] - vessel_velocity[0]
    )
    if discrete:
        load_impulse = float(np.sum(load_force[1:]) * step)
        vessel_impulse = float(np.sum(vessel_force[1:]) * step)
        impulse_evaluation = "right_endpoint_discrete_force"
        integrator = "discrete_sap"
    else:
        load_impulse = float(np.trapezoid(load_force, time_values))
        vessel_impulse = float(np.trapezoid(vessel_force, time_values))
        impulse_evaluation = "trapezoidal_continuous_force"
        integrator = "continuous_fixed_step_rk3"
    load_residual = float(load_delta - load_impulse)
    vessel_residual = float(vessel_delta - vessel_impulse)
    momentum = (
        constants.LOAD_MASS * load_velocity
        + constants.VESSEL_MASS * vessel_velocity
    )
    momentum_scale = max(
        np.max(np.abs(constants.LOAD_MASS * load_velocity)),
        np.max(np.abs(constants.VESSEL_MASS * vessel_velocity)),
        np.finfo(float).tiny,
    )
    impulse_scale = max(
        abs(load_delta),
        abs(vessel_delta),
        abs(load_impulse),
        abs(vessel_impulse),
        np.finfo(float).tiny,
    )
    energy = record[f"{prefix}_kinetic_energy_j"] + record[
        f"{prefix}_spring_potential_j"
    ]
    return {
        "maximum_energy_relative_drift": float(
            np.max(np.abs(energy / energy[0] - 1.0))
        ),
        "load_delta_momentum_kg_mps": float(load_delta),
        "load_impulse_n_s": load_impulse,
        "load_momentum_impulse_residual_kg_mps": load_residual,
        "vessel_delta_momentum_kg_mps": float(vessel_delta),
        "vessel_impulse_n_s": vessel_impulse,
        "vessel_momentum_impulse_residual_kg_mps": vessel_residual,
        "body_impulse_evaluation": impulse_evaluation,
        "maximum_normalized_body_residual": float(
            max(abs(load_residual), abs(vessel_residual)) / impulse_scale
        ),
        "maximum_total_momentum_drift_kg_mps": float(
            np.max(np.abs(momentum - momentum[0]))
        ),
        "normalized_total_momentum_drift": float(
            np.max(np.abs(momentum - momentum[0])) / momentum_scale
        ),
        "step_s": step,
        "integrator": integrator,
    }


def _phase1r_raw_energy_balances(record) -> dict[str, np.ndarray]:
    raw_fields = {
        "cell_status",
        "cell_pretension_n",
        "cell_gust_ratio",
        "cell_gust_duration_s",
        "cell_geometric_down_time_s",
        "cell_geometric_up_time_s",
        "trace_sample_count",
        "trace_time_s",
        "trace_truth",
        "trace_kinetic_energy_j",
        "trace_spring_potential_j",
        "trace_gust_power_w",
        "trace_thrust_power_w",
        "trace_linear_drag_power_w",
        "trace_angular_drag_power_w",
        "trace_weather_power_w",
        "trace_cable_damping_dissipation_w",
        "trace_cable_clipping_dissipation_w",
    }
    missing = sorted(raw_fields - set(record.files))
    if missing:
        raise RuntimeError(
            "Phase 1R record lacks raw energy/work channels: " + ", ".join(missing)
        )

    fields = (
        "window_start_s",
        "window_end_s",
        "delta_kinetic_j",
        "delta_spring_potential_j",
        "delta_mechanical_energy_j",
        "gust_work_j",
        "thrust_work_j",
        "linear_drag_work_j",
        "angular_drag_work_j",
        "weather_work_j",
        "cable_damping_dissipation_j",
        "cable_clipping_dissipation_j",
        "signed_nonconservative_work_j",
        "residual_j",
        "normalization_j",
        "normalized_residual",
    )
    balances = {
        field: np.full(record["cell_status"].shape, np.nan, dtype=float)
        for field in fields
    }

    def interpolate(
        time_values: np.ndarray, values: np.ndarray, sample_time: float
    ) -> np.ndarray | float:
        if values.ndim == 1:
            return float(np.interp(sample_time, time_values, values))
        return np.array(
            [
                np.interp(sample_time, time_values, values[:, column])
                for column in range(values.shape[1])
            ]
        )

    def integrate(
        time_values: np.ndarray,
        values: np.ndarray,
        start_time: float,
        end_time: float,
    ) -> float:
        interior = (time_values > start_time) & (time_values < end_time)
        window_time = np.concatenate(
            ([start_time], time_values[interior], [end_time])
        )
        window_values = np.concatenate(
            (
                [float(interpolate(time_values, values, start_time))],
                values[interior],
                [float(interpolate(time_values, values, end_time))],
            )
        )
        return float(np.trapezoid(window_values, window_time))

    for cell_index, status in enumerate(record["cell_status"]):
        if status != "MEASURED":
            continue
        sample_count = int(record["trace_sample_count"][cell_index])
        time_values = record["trace_time_s"][cell_index, :sample_count]
        truth = record["trace_truth"][cell_index, :sample_count]
        kinetic = record["trace_kinetic_energy_j"][cell_index, :sample_count]
        spring = record["trace_spring_potential_j"][cell_index, :sample_count]
        start_time = float(record["cell_geometric_down_time_s"][cell_index])
        end_time = float(record["cell_geometric_up_time_s"][cell_index])
        start_positions = interpolate(
            time_values, truth[:, : truth.shape[1] // 2], start_time
        )
        end_positions = interpolate(
            time_values, truth[:, : truth.shape[1] // 2], end_time
        )
        delta_kinetic = float(
            interpolate(time_values, kinetic, end_time)
            - interpolate(time_values, kinetic, start_time)
        )
        delta_spring = float(
            interpolate(time_values, spring, end_time)
            - interpolate(time_values, spring, start_time)
        )
        pretension = float(record["cell_pretension_n"][cell_index])
        common_speed = constants.VESSEL_COUNT * pretension / constants.LOAD_LINEAR_DRAG
        thrust_per_vessel = (
            common_speed
            * (
                constants.LOAD_LINEAR_DRAG
                + constants.VESSEL_COUNT * constants.VESSEL_LINEAR_DRAG
            )
            / constants.VESSEL_COUNT
        )
        thrust_work = float(
            thrust_per_vessel
            * np.sum(end_positions[3::3] - start_positions[3::3])
        )
        active_start = max(start_time, 0.0)
        active_end = min(
            end_time, float(record["cell_gust_duration_s"][cell_index])
        )
        if active_end > active_start:
            active_start_positions = interpolate(
                time_values, truth[:, : truth.shape[1] // 2], active_start
            )
            active_end_positions = interpolate(
                time_values, truth[:, : truth.shape[1] // 2], active_end
            )
            target_x = 3 * (2 + 1)
            relative_displacement = (
                active_end_positions[0]
                - active_start_positions[0]
                - active_end_positions[target_x]
                + active_start_positions[target_x]
            )
            gust_work = float(
                pretension
                * float(record["cell_gust_ratio"][cell_index])
                * relative_displacement
            )
        else:
            gust_work = 0.0
        integrated = {
            name: integrate(
                time_values,
                record[trace_name][cell_index, :sample_count],
                start_time,
                end_time,
            )
            for name, trace_name in (
                ("linear_drag_work_j", "trace_linear_drag_power_w"),
                ("angular_drag_work_j", "trace_angular_drag_power_w"),
                ("weather_work_j", "trace_weather_power_w"),
                (
                    "cable_damping_dissipation_j",
                    "trace_cable_damping_dissipation_w",
                ),
                (
                    "cable_clipping_dissipation_j",
                    "trace_cable_clipping_dissipation_w",
                ),
            )
        }
        delta_mechanical = delta_kinetic + delta_spring
        signed_work = (
            gust_work
            + thrust_work
            + integrated["linear_drag_work_j"]
            + integrated["angular_drag_work_j"]
            + integrated["weather_work_j"]
            - integrated["cable_damping_dissipation_j"]
            - integrated["cable_clipping_dissipation_j"]
        )
        residual = delta_mechanical - signed_work
        normalization = max(
            abs(delta_mechanical),
            abs(gust_work)
            + abs(thrust_work)
            + abs(integrated["linear_drag_work_j"])
            + abs(integrated["angular_drag_work_j"])
            + abs(integrated["weather_work_j"])
            + abs(integrated["cable_damping_dissipation_j"])
            + abs(integrated["cable_clipping_dissipation_j"]),
            np.finfo(float).tiny,
        )
        values = {
            "window_start_s": start_time,
            "window_end_s": end_time,
            "delta_kinetic_j": delta_kinetic,
            "delta_spring_potential_j": delta_spring,
            "delta_mechanical_energy_j": delta_mechanical,
            "gust_work_j": gust_work,
            "thrust_work_j": thrust_work,
            **integrated,
            "signed_nonconservative_work_j": signed_work,
            "residual_j": residual,
            "normalization_j": normalization,
            "normalized_residual": abs(residual) / normalization,
        }
        for field, value in values.items():
            balances[field][cell_index] = value
    return balances


def _phase1r_energy_balance_replay_checks(
    record,
    results: dict[str, object],
    *,
    absolute_tolerance: float = PHASE1R_REPLAY_NUMERIC_ATOL,
) -> tuple[dict[str, bool], dict[str, np.ndarray]]:
    balances = _phase1r_raw_energy_balances(record)
    measured = record["cell_status"] == "MEASURED"
    reported = results["tests"]["P1-T5"]["full_plant_energy_balance"]
    normalized = balances["normalized_residual"][measured]
    maximum = float(np.max(normalized))
    status = (
        "PASS"
        if normalized.size == np.count_nonzero(measured)
        and maximum <= PHASE1R_ENERGY_BALANCE_TOLERANCE
        else "FAIL"
    )
    checks = {
        f"aggregate_{field}_matches_raw": bool(
            np.allclose(
                record[f"cell_energy_balance_{field}"][measured],
                values[measured],
                rtol=0.0,
                atol=absolute_tolerance,
                equal_nan=True,
            )
        )
        for field, values in balances.items()
    }
    checks.update(
        {
            "reported_evaluated_count_matches_raw": (
                reported["evaluated_count"] == int(normalized.size)
            ),
            "reported_maximum_residual_matches_raw": bool(
                np.isclose(
                    reported["maximum_normalized_residual"],
                    maximum,
                    rtol=0.0,
                    atol=absolute_tolerance,
                )
            ),
            "reported_status_matches_raw": reported["status"] == status,
        }
    )
    return checks, balances


def _phase1r_record_metric_checks(
    record_path: Path,
    results: dict[str, object],
) -> dict[str, bool]:
    tests = results["tests"]
    computed = {}
    with np.load(record_path, allow_pickle=False) as record:
        energy_replay_checks, raw_energy_balances = (
            _phase1r_energy_balance_replay_checks(record, results)
        )
        bilateral_truth = record["p1t1_bilateral_truth"]
        unilateral_truth = record["p1t1_unilateral_truth"]
        position_count = bilateral_truth.shape[1] // 2
        phase0_distance = float(
            np.mean(bilateral_truth[-1, position_count::3])
            * (
                record["p1t1_bilateral_time_s"][-1]
                - record["p1t1_bilateral_time_s"][0]
            )
        )
        unilateral_distance = float(
            np.mean(
                unilateral_truth[-1, :position_count:3]
                - unilateral_truth[0, :position_count:3]
            )
        )
        phase0_tension = float(np.mean(record["p1t1_bilateral_tension_n"][-1]))
        unilateral_tension = float(np.mean(record["p1t1_unilateral_tension_n"]))
        distance_error = abs(unilateral_distance / phase0_distance - 1.0)
        tension_error = abs(unilateral_tension / phase0_tension - 1.0)
        minimum_elongation = float(np.min(record["p1t1_unilateral_elongation_m"]))
        no_slack = minimum_elongation > 0.0
        t1_passed = distance_error <= 0.01 and tension_error <= 0.03 and no_slack
        computed["P1-T1"] = {
            "phase0_distance_m": phase0_distance,
            "unilateral_distance_m": unilateral_distance,
            "distance_relative_error": distance_error,
            "phase0_mean_tension_n": phase0_tension,
            "unilateral_mean_tension_n": unilateral_tension,
            "tension_relative_error": tension_error,
            "minimum_elongation_m": minimum_elongation,
            "no_cable_slack": no_slack,
            "passed": t1_passed,
            "status": "PASS" if t1_passed else "FAIL",
        }

        impact_speeds = record["mechanics_impact_speeds_mps"]
        isolated_peaks = np.array(
            [
                _first_sampled_peak(values)
                for values in record["mechanics_impact_tension_n"]
            ]
        )
        isolated_impedance, isolated_r_squared = _record_origin_fit(
            impact_speeds, isolated_peaks
        )
        isolated_error = abs(isolated_impedance / ANALYTIC_IMPEDANCE - 1.0)
        t2_passed = isolated_r_squared > 0.99 and isolated_error <= 0.15
        computed["P1-T2"] = {
            "speeds_mps": impact_speeds.tolist(),
            "first_peaks_n": isolated_peaks.tolist(),
            "impedance_n_s_per_m": isolated_impedance,
            "pinned_impedance_n_s_per_m": PINNED_IMPEDANCE,
            "analytic_impedance_n_s_per_m": ANALYTIC_IMPEDANCE,
            "relative_error": isolated_error,
            "uncentred_r_squared": isolated_r_squared,
            "passed": t2_passed,
            "status": "PASS" if t2_passed else "FAIL",
        }
        reduced_mass_estimate = isolated_impedance**2 / (
            constants.CABLE_STIFFNESS * IMPACT_FACTOR**2
        )
        analytic_reduced_mass = (
            constants.VESSEL_MASS
            * constants.LOAD_MASS
            / (constants.VESSEL_MASS + constants.LOAD_MASS)
        )
        mass_error = abs(reduced_mass_estimate / analytic_reduced_mass - 1.0)
        t3_passed = mass_error <= 0.10
        computed["P1-T3"] = {
            "estimated_reduced_mass_kg": reduced_mass_estimate,
            "analytic_reduced_mass_kg": analytic_reduced_mass,
            "relative_error": mass_error,
            "estimation_residual_kg": (
                reduced_mass_estimate - analytic_reduced_mass
            ),
            "rotational_excess_kg": 0.0,
            "passed": t3_passed,
            "status": "PASS" if t3_passed else "FAIL",
        }

        fleet_speeds = record["p1t2b_impact_speeds_mps"]
        fleet_tension = record["p1t2b_impact_tension_n"]
        fleet_peaks = np.array(
            [
                _first_sampled_peak(values[:, 2])
                for values in fleet_tension
            ]
        )
        fleet_impedance, fleet_r_squared = _record_origin_fit(
            fleet_speeds, fleet_peaks
        )
        zero_baseline = float(
            np.max(record["p1t2b_baseline_tension_n"][:, 2])
        )
        baseline_peaks = fleet_peaks - zero_baseline
        baseline_impedance, baseline_r_squared = _record_origin_fit(
            fleet_speeds, baseline_peaks
        )
        affine_slope, affine_intercept = np.polyfit(
            fleet_speeds, fleet_peaks, 1
        )
        fleet_residual = fleet_peaks - fleet_impedance * fleet_speeds
        affine_residual = fleet_peaks - (
            affine_intercept + affine_slope * fleet_speeds
        )
        affine_r_squared = float(
            1.0
            - np.dot(affine_residual, affine_residual)
            / np.dot(
                fleet_peaks - np.mean(fleet_peaks),
                fleet_peaks - np.mean(fleet_peaks),
            )
        )
        other_initial = np.delete(fleet_tension[:, 0, :], 2, axis=1)
        other_taut = bool(
            np.allclose(other_initial, 990.0, rtol=0.0, atol=1.0e-8)
        )
        t2b_passed = fleet_r_squared > 0.99 and other_taut
        computed["P1-T2b"] = {
            "speeds_mps": fleet_speeds.tolist(),
            "first_peaks_n": fleet_peaks.tolist(),
            "target_initial_tension_n": float(fleet_tension[0, 0, 2]),
            "impedance_n_s_per_m": fleet_impedance,
            "pinned_impedance_n_s_per_m": PINNED_FLEET_IMPEDANCE,
            "pin_relative_error": abs(
                fleet_impedance / PINNED_FLEET_IMPEDANCE - 1.0
            ),
            "reference_impedance_n_s_per_m": 9_000.0,
            "reference_relative_difference": abs(
                fleet_impedance / 9_000.0 - 1.0
            ),
            "uncentred_r_squared": fleet_r_squared,
            "zero_speed_baseline_n": zero_baseline,
            "baseline_subtracted_peaks_n": baseline_peaks.tolist(),
            "baseline_subtracted_impedance_n_s_per_m": baseline_impedance,
            "baseline_subtracted_uncentred_r_squared": baseline_r_squared,
            "affine_slope_n_s_per_m": float(affine_slope),
            "affine_intercept_n": float(affine_intercept),
            "affine_r_squared": affine_r_squared,
            "low_speed_residuals_n": {
                "required_through_origin": float(fleet_residual[0]),
                "baseline_subtracted": float(
                    baseline_peaks[0] - baseline_impedance * fleet_speeds[0]
                ),
                "affine": float(affine_residual[0]),
            },
            "other_cables_taut_at_pretension": other_taut,
            "passed": t2b_passed,
            "status": "PASS" if t2b_passed else "FAIL",
        }

        measured = record["cell_status"] == "MEASURED"
        return_speeds = record["cell_return_speed_mps"][measured]
        measured_peaks = record["cell_peak_tension_n"][measured]
        exact_gust_work = record["cell_gust_work_j"][measured]
        entry_speeds = record["cell_measured_entry_speed_mps"][measured]
        reduced_energy_change = 0.5 * PINNED_FLEET_EFFECTIVE_MASS * (
            return_speeds**2 - entry_speeds**2
        )
        surrogate_slope, surrogate_r_squared = _record_origin_fit(
            exact_gust_work, reduced_energy_change
        )
        surrogate_status = (
            "PASS"
            if abs(surrogate_slope - 1.0) <= 0.05
            and surrogate_r_squared > 0.98
            else "FAIL"
        )
        normalized_balance = raw_energy_balances["normalized_residual"][measured]
        maximum_balance = float(np.max(normalized_balance))
        balance_status = (
            "PASS"
            if normalized_balance.size == np.count_nonzero(measured)
            and maximum_balance <= PHASE1R_ENERGY_BALANCE_TOLERANCE
            else "FAIL"
        )
        predicted_peaks = PINNED_FLEET_IMPEDANCE * return_speeds
        all_slope, all_r_squared = _record_origin_fit(
            predicted_peaks, measured_peaks
        )
        in_domain = (
            (return_speeds >= PHASE1R_IMPACT_MIN_SPEED)
            & (return_speeds <= PHASE1R_IMPACT_MAX_SPEED)
        )
        in_domain_count = int(np.count_nonzero(in_domain))
        if in_domain_count >= PHASE1R_IMPACT_MIN_MARKS:
            in_domain_slope, in_domain_r_squared = _record_origin_fit(
                predicted_peaks[in_domain], measured_peaks[in_domain]
            )
            impact_status = (
                "PASS"
                if abs(in_domain_slope - 1.0) <= 0.06
                else "MODEL_INVALID_LOW_SPEED"
            )
            affine_slope, affine_intercept = np.polyfit(
                return_speeds[in_domain], measured_peaks[in_domain], 1
            )
            affine_residual = measured_peaks[in_domain] - (
                affine_intercept + affine_slope * return_speeds[in_domain]
            )
            centred_peaks = measured_peaks[in_domain] - np.mean(
                measured_peaks[in_domain]
            )
            affine_r_squared = float(
                1.0
                - np.dot(affine_residual, affine_residual)
                / np.dot(centred_peaks, centred_peaks)
            )
            affine_intercept_fraction = float(
                affine_intercept
                / (PINNED_FLEET_IMPEDANCE * np.min(return_speeds[in_domain]))
            )
        else:
            in_domain_slope = None
            in_domain_r_squared = None
            affine_slope = None
            affine_intercept = None
            affine_r_squared = None
            affine_intercept_fraction = None
            impact_status = "UNDER-POWERED"
        t5_status = (
            "FAIL"
            if balance_status == "FAIL"
            else "MODEL_INVALID_LOW_SPEED"
            if impact_status == "MODEL_INVALID_LOW_SPEED"
            else "UNDER-POWERED"
            if impact_status == "UNDER-POWERED"
            else "PASS"
        )
        computed["P1-T5"] = {
            "status": t5_status,
            "nonpassing_basis": (
                "impact_model_specification"
                if impact_status == "MODEL_INVALID_LOW_SPEED"
                else "full_plant_energy_balance"
                if balance_status == "FAIL"
                else None
            ),
            "physical_plant_failure": balance_status == "FAIL",
            "complete_count": int(np.count_nonzero(measured)),
            "right_censored_count": int(np.count_nonzero(~measured)),
            "full_plant_energy_balance": {
                "status": balance_status,
                "evaluated_count": int(normalized_balance.size),
                "maximum_normalized_residual": maximum_balance,
            },
            "reduced_effective_mass_surrogate": {
                "status": surrogate_status,
                "effective_mass_kg": PINNED_FLEET_EFFECTIVE_MASS,
                "through_origin_slope": surrogate_slope,
                "uncentred_r_squared": surrogate_r_squared,
            },
            "impact_calibration": {
                "status": impact_status,
                "physical_plant_failure": False,
                "return_speed_range_mps": [
                    float(np.min(return_speeds)),
                    float(np.max(return_speeds)),
                ],
                "below_range_count": int(
                    np.count_nonzero(return_speeds < PHASE1R_IMPACT_MIN_SPEED)
                ),
                "in_range_count": in_domain_count,
                "above_range_count": int(
                    np.count_nonzero(return_speeds > PHASE1R_IMPACT_MAX_SPEED)
                ),
                "in_range_through_origin_slope": in_domain_slope,
                "in_range_uncentred_r_squared": in_domain_r_squared,
                "in_range_affine_slope_n_s_per_m": float(affine_slope),
                "in_range_affine_intercept_n": float(affine_intercept),
                "in_range_affine_r_squared": affine_r_squared,
                "affine_intercept_fraction_at_min_speed": (
                    affine_intercept_fraction
                ),
                "all_mark_diagnostic": {
                    "through_origin_slope": all_slope,
                    "uncentred_r_squared": all_r_squared,
                },
            },
        }

        probes = [
            {
                "pretension_n": float(pretension),
                "gust_ratio": float(gust_ratio),
                "time_step_s": float(time_step),
                "fit_window_s": float(fit_window),
                "initial_elongation_m": float(initial_elongation),
                "initial_tension_n": float(initial_tension),
                "maximum_elongation_m": float(maximum_elongation),
                "maximum_tension_n": float(maximum_tension),
                "acceleration_mps2": float(acceleration),
            }
            for pretension, gust_ratio, time_step, fit_window, initial_elongation,
            initial_tension, maximum_elongation, maximum_tension, acceleration in zip(
                record["t6_probe_pretension_n"],
                record["t6_probe_gust_ratio"],
                record["t6_probe_time_step_s"],
                record["t6_probe_fit_window_s"],
                record["t6_probe_initial_elongation_m"],
                record["t6_probe_initial_tension_n"],
                record["t6_probe_maximum_elongation_m"],
                record["t6_probe_maximum_tension_n"],
                record["t6_probe_acceleration_mps2"],
            )
        ]
        conditions = []
        for time_step, fit_window in sorted(
            {(row["time_step_s"], row["fit_window_s"]) for row in probes},
            reverse=True,
        ):
            rows = [
                row
                for row in probes
                if row["time_step_s"] == time_step
                and row["fit_window_s"] == fit_window
            ]
            x_values = np.array([row["gust_ratio"] for row in rows])
            y_values = np.array(
                [
                    row["acceleration_mps2"]
                    / (row["pretension_n"] / PINNED_FLEET_EFFECTIVE_MASS)
                    for row in rows
                ]
            )
            fit = _record_threshold(x_values, y_values)
            bracketed = any(
                row["gust_ratio"] < 1.0 and row["acceleration_mps2"] > 0.0
                for row in rows
            ) and any(
                row["gust_ratio"] > 1.0 and row["acceleration_mps2"] < 0.0
                for row in rows
            )
            condition_status = (
                "PASS"
                if bracketed and abs(fit["lambda_c"] - 1.0) <= 0.05
                else "FAIL"
            )
            conditions.append(
                {
                    "time_step_s": time_step,
                    "fit_window_s": fit_window,
                    "sample_count": len(rows),
                    "bracketed_around_one": bracketed,
                    **fit,
                    "status": condition_status,
                }
            )
        primary = next(
            row
            for row in conditions
            if np.isclose(row["time_step_s"], record["t6_primary_time_step_s"][0])
            and np.isclose(
                row["fit_window_s"], record["t6_primary_fit_window_s"][0]
            )
        )
        t6_passed = primary["status"] == "PASS"
        sensitivity_passed = all(row["status"] == "PASS" for row in conditions)
        legacy_valid = np.isfinite(record["cell_slack_acceleration_mps2"])
        legacy_fit = _record_threshold(
            record["cell_gust_ratio"][legacy_valid],
            record["cell_slack_acceleration_mps2"][legacy_valid]
            / (
                record["cell_pretension_n"][legacy_valid]
                / PINNED_FLEET_EFFECTIVE_MASS
            ),
        )
        computed["P1-T6"] = {
            "status": "PASS" if t6_passed else "FAIL",
            "primary_estimator": primary,
            "bracketed_around_one": primary["bracketed_around_one"],
            "lambda_c_interval": [
                min(row["lambda_c"] for row in conditions),
                max(row["lambda_c"] for row in conditions),
            ],
            "sensitivity_status": (
                "ROBUST_WITHIN_5_PERCENT"
                if sensitivity_passed
                else "WINDOW_OR_STEP_SENSITIVE"
                if t6_passed
                else "PRIMARY_FAIL"
            ),
            "sensitivity": conditions,
            "probe_measurements": probes,
            "legacy_transient_diagnostic": {
                "sample_count": int(np.count_nonzero(legacy_valid)),
                "acceleration_fit": legacy_fit,
            },
        }

        eligible = (
            measured
            & (record["cell_depth_at_gust_shutoff_m"] < 1.0)
            & np.isfinite(record["cell_reduced_shutoff_depth_prediction_m"])
            & np.isfinite(record["cell_amended_maximum_depth_prediction_m"])
        )
        shutoff_error = np.abs(
            record["cell_depth_at_gust_shutoff_m"][eligible]
            / record["cell_reduced_shutoff_depth_prediction_m"][eligible]
            - 1.0
        )
        maximum_error = np.abs(
            record["cell_maximum_depth_m"][eligible]
            / record["cell_amended_maximum_depth_prediction_m"][eligible]
            - 1.0
        )
        t7_passed = bool(
            np.count_nonzero(eligible)
            and np.max(shutoff_error) <= 0.15
            and np.max(maximum_error) <= 0.15
        )
        computed["P1-T7"] = {
            "status": "PASS" if t7_passed else "FAIL",
            "eligible_count": int(np.count_nonzero(eligible)),
            "maximum_shutoff_depth_relative_error": float(
                np.max(shutoff_error)
            ),
            "maximum_amended_depth_relative_error": float(
                np.max(maximum_error)
            ),
        }
        deepest_index = int(np.argmax(record["cell_maximum_depth_m"]))
        censored_count = int(np.count_nonzero(~measured))
        computed["P1-T8"] = {
            "status": (
                "RIGHT_CENSORED" if censored_count else "OBSERVED_NOT_GLOBAL"
            ),
            "case_status_counts": {
                "MEASURED": int(np.count_nonzero(measured)),
                "RIGHT_CENSORED": censored_count,
            },
            "observed_maximum_depth_m": float(
                record["cell_maximum_depth_m"][deepest_index]
            ),
            "observed_maximum_case": {
                "pretension_n": float(record["cell_pretension_n"][deepest_index]),
                "gust_ratio": float(record["cell_gust_ratio"][deepest_index]),
                "gust_duration_s": float(
                    record["cell_gust_duration_s"][deepest_index]
                ),
                "entry_speed_mps": float(
                    record["cell_requested_entry_speed_mps"][deepest_index]
                ),
                "status": str(record["cell_status"][deepest_index]),
            },
            "global_maximum_established": False,
            "impedance_extrapolation_count": int(
                np.count_nonzero(record["cell_impedance_extrapolation"] & measured)
            ),
        }

        refinement = [
            _record_conservation_metrics(
                record, f"t9_refinement_{index}", discrete=False
            )
            for index in range(len(REFINEMENT_STEPS))
        ]
        production = _record_conservation_metrics(
            record, "t9_production_discrete_1ms", discrete=True
        )
        for row, step in zip(refinement, REFINEMENT_STEPS):
            row["step_s"] = step
        production["step_s"] = constants.TIME_STEP
        production["passed"] = bool(
            production["maximum_energy_relative_drift"] <= 0.005
            and production["maximum_normalized_body_residual"] <= 0.005
            and production["normalized_total_momentum_drift"] <= 0.005
        )
        refinement_passed = all(
            row["maximum_energy_relative_drift"] <= 0.005
            and row["maximum_normalized_body_residual"] <= 0.005
            and row["normalized_total_momentum_drift"] <= 0.005
            for row in refinement
        )
        t9_passed = bool(refinement_passed and production["passed"])
        computed["P1-T9"] = {
            "continuous_refinement": refinement,
            "continuous_refinement_passed": refinement_passed,
            "production_discrete_1ms": production,
            "maximum_energy_relative_drift": production[
                "maximum_energy_relative_drift"
            ],
            "normalized_impulse_mismatch": production[
                "maximum_normalized_body_residual"
            ],
            "blocking_failure_basis": (
                "discrete_1ms_energy"
                if production["maximum_energy_relative_drift"] > 0.005
                else None
            ),
            "passed": t9_passed,
            "status": "PASS" if t9_passed else "FAIL",
        }
    comparisons = {
        test_id: _metric_subset_matches(values, tests[test_id])
        for test_id, values in computed.items()
    }
    comparisons["P1-T5_raw_energy_balance"] = all(
        energy_replay_checks.values()
    )
    return comparisons


def _derive_phase1r_decision(tests: dict[str, dict[str, object]]):
    blocking = ("P1-T1", "P1-T2", "P1-T2b", "P1-T3", "P1-T5", "P1-T6", "P1-T7", "P1-T9")
    failed = [test_id for test_id in blocking if tests[test_id]["status"] != "PASS"]
    phase1_gate = "GO" if not failed else "NO-GO"
    deterministic_verdict = "PASS" if not failed else "FAIL"
    stop_reason = (
        None
        if not failed
        else "blocking criteria not satisfied: "
        + ", ".join(
            f"{test_id} ({tests[test_id]['status']})" for test_id in failed
        )
    )
    return phase1_gate, deterministic_verdict, stop_reason, failed


def _configuration() -> dict[str, object]:
    return {
        "cable_damping_impact_n_s_per_m": constants.CABLE_DAMPING,
        "cable_damping_energy_n_s_per_m": 0.0,
        "cable_rest_length_m": constants.CABLE_REST_LENGTH,
        "cable_stiffness_n_per_m": constants.CABLE_STIFFNESS,
        "event_period_s": constants.TIME_STEP,
        "force_application": "explicit ExternallyAppliedSpatialForce at attachments",
        "impact_factor": IMPACT_FACTOR,
        "impact_speeds_mps": IMPACT_SPEEDS.tolist(),
        "integration": "Drake continuous MultibodyPlant, fixed-step RK3",
        "integration_step_s": constants.TIME_STEP,
        "load_mass_kg": constants.LOAD_MASS,
        "vessel_mass_kg": constants.VESSEL_MASS,
    }


def _format_report(results: dict[str, object]) -> str:
    t2 = results["tests"]["P1-T2"]
    t3 = results["tests"]["P1-T3"]
    t9 = results["tests"]["P1-T9"]
    peak_rows = "\n".join(
        f"| {speed:.2f} | {peak:.9f} |"
        for speed, peak in zip(t2["speeds_mps"], t2["first_peaks_n"])
    )
    remaining = "\n".join(f"- {test}." for test in results["remaining_tests"])
    production = t9["production_discrete_1ms"]
    return f"""> **SUPERSEDED:** Historical Stage 1-2 artifact. The Phase 1R report contains the current gate decision.

# Phase 1 Mechanics Report

## Scope verdict

**{results['mechanics_verdict']}.** This result covers Phase 1 Stage 1-2 mechanics only. It is not a full Phase 1 GO decision because the remaining tests listed below have not been executed.

## Method

The isolated cell contains a 2500 kg load and a 600 kg vessel on independent collinear prismatic joints. Gravity, drag, weather, thrust, and control are absent. Initial velocities impose zero total momentum and the prescribed relative speed at zero cable elongation. A `UnilateralCable` applies equal-and-opposite `ExternallyAppliedSpatialForce` values at the attachment points. Drake advances the continuous plant with a fixed 1 ms Runge-Kutta 3 step.

P1-T2 uses the first local maximum of the cable force sampled from each Drake trajectory; no analytic peak is substituted. The through-origin slope is `sum(V*T_peak)/sum(V^2)`, and the reported coefficient is the uncentred R2. P1-T9 evaluates per-body momentum-impulse residuals, total momentum drift, and energy drift in continuous RK3 cells at 1, 0.5, and 0.25 ms. It separately records the discrete SAP production cell at 1 ms.

## Results

| Test | Measurement | Criterion | Verdict |
|---|---|---|---|
| P1-T2 | Zhat = {t2['impedance_n_s_per_m']:.12f} N s/m; uncentred R2 = {t2['uncentred_r_squared']:.12f}; analytic-value error = {100 * t2['relative_error']:.9f}% | R2 > 0.99; error <= 15% from 7990 N s/m | {'PASS' if t2['passed'] else 'FAIL'} |
| P1-T3 | meff_hat = {t3['estimated_reduced_mass_kg']:.12f} kg; translational meff = {t3['analytic_reduced_mass_kg']:.12f} kg; error = {100 * t3['relative_error']:.9f}% | error <= 10% | {'PASS' if t3['passed'] else 'FAIL'} |
| P1-T9 | discrete 1 ms energy drift = {production['maximum_energy_relative_drift']:.12e}; maximum normalized body residual = {production['maximum_normalized_body_residual']:.12e}; momentum drift = {production['normalized_total_momentum_drift']:.12e} | each <= 0.5%; continuous refinement also passes | {'PASS' if t9['passed'] else 'FAIL'} |

The measured and pinned engagement impedance is **{PINNED_IMPEDANCE:.12f} N s/m**.

### P1-T2 trajectory peaks

| Relative speed [m/s] | First sampled peak [N] |
|---:|---:|
{peak_rows}

### P1-T3 rotational contribution

The collinear centerline cell has no rotational degree of freedom or lever arm. Its rotational excess is therefore {t3['rotational_excess_kg']:.1f} kg. The measured mass residual of {t3['estimation_residual_kg']:.12f} kg is numerical and is not attributed to rotation.

## Deviations

1. The mechanics cell uses a continuous Drake `MultibodyPlant` with fixed-step Runge-Kutta 3 integration at 1 ms. The Phase 0 plant remains discrete. This choice retains the explicit force input while permitting the independent state-time energy audit to satisfy P1-T9.
2. This stage implements the isolated single-cable cell only. The five-vessel unilateral regression, fleet impedance, and excursion campaigns remain outside the present scope.

## Remaining Phase 1 tests

{remaining}

Until these tests are implemented and executed, no full Phase 1 GO is claimed.
"""


def run_mechanics_campaign(
    results_path: Path = RESULTS_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, object]:
    tests = mechanics_acceptance()
    passed = all(result["passed"] for result in tests.values())
    phase1_gate = "NOT_EVALUATED" if REMAINING_TESTS else ("GO" if passed else "NO-GO")
    results = {
        "schema_version": 1,
        "scope": "Phase 1 Stage 1-2 mechanics only",
        "phase1_gate": phase1_gate,
        "mechanics_verdict": "PASS" if passed else "FAIL",
        "configuration": _configuration(),
        "pinned_impedance_n_s_per_m": PINNED_IMPEDANCE,
        "tests": tests,
        "remaining_tests": list(REMAINING_TESTS),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_bytes(_json_bytes(results))
    report_path.write_text(_format_report(results), encoding="ascii")
    print(f"Phase 1 Stage 1-2 mechanics: {results['mechanics_verdict']}")
    for test_id, result in tests.items():
        print(f"{test_id}: {'PASS' if result['passed'] else 'FAIL'}")
    return results


def _deterministic_configuration() -> dict[str, object]:
    return {
        "cable_mode": "recording",
        "cable_damping_n_s_per_m": constants.CABLE_DAMPING,
        "cable_rest_length_m": constants.CABLE_REST_LENGTH,
        "cable_stiffness_n_per_m": constants.CABLE_STIFFNESS,
        "entry_speed_primary_mps": PRIMARY_ENTRY_SPEED,
        "entry_speed_sensitivity_mps": list(SENSITIVITY_ENTRY_SPEEDS),
        "force_application": "explicit spatial forces at body-fixed attachments",
        "full_plant": "Drake discrete MultibodyPlant, SAP, zero gravity",
        "geometry": "collinear Phase 0 specialization",
        "geometric_depth_guard_m": GEOMETRIC_DEPTH_LIMIT,
        "gust_durations_s": GUST_DURATIONS.tolist(),
        "gust_ratios": GUST_RATIOS.tolist(),
        "impact_speeds_mps": FLEET_IMPACT_SPEEDS.tolist(),
        "pinned_fleet_impedance_n_s_per_m": PINNED_FLEET_IMPEDANCE,
        "pretensions_n": PRETENSION_LEVELS.tolist(),
        "pretension_source": "explicit inference from Phase 0 measured set",
        "sample_period_s": constants.TIME_STEP,
        "stochastic_campaign_included": False,
        "target_cable": 2,
        "time_step_s": constants.TIME_STEP,
    }


def _excursion_record_arrays(outcomes) -> dict[str, np.ndarray]:
    def optional(name: str) -> np.ndarray:
        return np.array(
            [
                np.nan if getattr(outcome, name) is None else getattr(outcome, name)
                for outcome in outcomes
            ],
            dtype=float,
        )

    return {
        "excursion_censored": np.array(
            [outcome.censored for outcome in outcomes], dtype=bool
        ),
        "excursion_complete": np.array(
            [outcome.complete for outcome in outcomes], dtype=bool
        ),
        "excursion_depth_at_gust_shutoff_m": np.array(
            [outcome.depth_at_gust_shutoff_m for outcome in outcomes]
        ),
        "excursion_entry_speed_mps": np.array(
            [outcome.entry_speed_mps for outcome in outcomes]
        ),
        "excursion_gust_duration_s": np.array(
            [outcome.gust_duration_s for outcome in outcomes]
        ),
        "excursion_gust_ratio": np.array(
            [outcome.gust_ratio for outcome in outcomes]
        ),
        "excursion_maximum_depth_m": np.array(
            [outcome.maximum_depth_m for outcome in outcomes]
        ),
        "excursion_peak_tension_n": optional("peak_tension_n"),
        "excursion_pretension_n": np.array(
            [outcome.pretension_n for outcome in outcomes]
        ),
        "excursion_reengagement_time_s": optional("reengagement_time_s"),
        "excursion_return_speed_mps": optional("return_speed_mps"),
        "excursion_slack_acceleration_mps2": np.array(
            [outcome.slack_acceleration_mps2 for outcome in outcomes]
        ),
        "excursion_slack_at_gust_shutoff": np.array(
            [outcome.slack_at_gust_shutoff for outcome in outcomes], dtype=bool
        ),
    }


def _format_deterministic_report(results: dict[str, object]) -> str:
    tests = results["tests"]
    t1 = tests["P1-T1"]
    t2b = tests["P1-T2b"]
    t4 = tests["P1-T4"]
    t5 = tests["P1-T5"]
    t6 = tests["P1-T6"]
    t7 = tests["P1-T7"]
    t8 = tests["P1-T8"]
    impact_rows = "\n".join(
        f"| {speed:.2f} | {peak:.9f} |"
        for speed, peak in zip(t2b["speeds_mps"], t2b["first_peaks_n"])
    )
    pretension_rows = "\n".join(
        f"| {pretension} | {values['complete_count']} | {values['slope']:.9f} | "
        f"{values['uncentred_r_squared']:.9f} |"
        for pretension, values in t5["per_pretension"].items()
    )
    sensitivity_rows = "\n".join(
        f"| {row['entry_speed_mps']:.2f} | {row['complete_count']} | "
        f"{row['censored_count']} | {row['observed_maximum_complete_depth_m']:.9f} |"
        for row in t8["sensitivity"]
    )
    return f"""> **SUPERSEDED:** Historical Stage 3-4 artifact based on the withdrawn P1-T6 specification. See `reports/phase1r_report.md` for the corrected gate.

# Phase 1 Deterministic Stage 3-4 Report

## Gate verdict

**{results['phase1_gate']} / {results['deterministic_verdict']}.** {results['stop_reason']}. P1-T6 uses the now-withdrawn literal criterion in this historical artifact. The force-consistent acceleration diagnostic has reduced-model threshold {t6['force_consistent_acceleration_fit']['lambda_c']:.12f} and full-plant threshold {t6['full_plant_acceleration_fit']['lambda_c']:.12f}. The campaign stops before the stochastic 8 x 20 x 300 s stage, and no full Phase 1 GO is claimed.

## Configuration

The full formation uses a discrete Drake `MultibodyPlant` with a 1 ms step, SAP contact approximation, zero gravity, explicit drag, thrust, weather, square-gust, and attachment-point cable forces. Five `UnilateralCable` systems run in recording mode. The geometry is the collinear Phase 0 specialization; consequently, the rotational excess is reported as zero.

The scripted grid uses lambda in `{GUST_RATIOS.tolist()}`, durations in `{GUST_DURATIONS.tolist()}` s, and the explicitly inferred Phase 0 pretensions `{PRETENSION_LEVELS.tolist()}` N. The primary slack-entry speed is {PRIMARY_ENTRY_SPEED:.2f} m/s. Sensitivity runs use `{list(SENSITIVITY_ENTRY_SPEEDS)}` m/s. Reduced trajectories use exact constant-acceleration updates on the same 1 ms record grid and stop at a {GEOMETRIC_DEPTH_LIMIT:.2f} m attachment-coincidence guard.

## Acceptance results

| Test | Measurement | Criterion | Verdict |
|---|---|---|---|
| P1-T1 | distance error {100 * t1['distance_relative_error']:.12e}%; tension error {100 * t1['tension_relative_error']:.12e}%; minimum elongation {t1['minimum_elongation_m']:.12f} m | distance <= 1%; tension <= 3%; no slack | {t1['verdict']} |
| P1-T2b | Zfleet = {t2b['impedance_n_s_per_m']:.12f} N s/m; uncentred R2 = {t2b['uncentred_r_squared']:.12f}; difference from 9000 = {100 * t2b['reference_relative_difference']:.9f}% | R2 > 0.99; reference difference recorded only | {t2b['verdict']} |
| P1-T4 | literal eligible marks {t4['literal_eligible_count']}; corrected constant-gust marks {t4['corrected_constant_gust_eligible_count']}; corrected speed/depth errors {t4['corrected_maximum_speed_relative_error']:.3e}/{t4['corrected_maximum_depth_relative_error']:.3e} | literal only when gust is absent for the complete slack interval; corrected comparator reported separately | {t4['verdict']} |
| P1-T5 | slope {t5['energy_slope']:.12f}; uncentred R2 {t5['energy_uncentred_r_squared']:.12f}; maximum tension error {100 * t5['maximum_tension_relative_error']:.9f}% | slope within 5%; R2 > 0.98; tension error <= 6% | {t5['verdict']} |
| P1-T6 | literal low/high conditions {t6['literal_low_lambda_passed']}/{t6['literal_high_lambda_passed']}; literal threshold {t6['literal_threshold_fit']['lambda_c']:.12f}; force thresholds {t6['force_consistent_acceleration_fit']['lambda_c']:.12f}/{t6['full_plant_acceleration_fit']['lambda_c']:.12f} | literal threshold within 5% of 1; amended force threshold within 5% | {t6['literal_verdict']} / {t6['verdict']} |
| P1-T7 | shutoff-depth maximum error {t7['literal_maximum_relative_error']:.3e}; amended maximum-depth error {t7['amended_maximum_relative_error']:.3e} | each <= 15% where depth < 1 m | {t7['verdict']} |
| P1-T8 | observed maximum complete depth {t8['observed_maximum_complete_depth_m']:.12f} m; implied snap {t8['implied_maximum_snap_tension_n']:.9f} N | measure and pin; no global claim | {t8['verdict']} |

## Fleet impact

The centre cable starts at the slack boundary. The other four cables start at {t1['phase0_mean_tension_n']:.9f} N pretension. The velocity perturbation preserves total fleet momentum and gives the target cable the prescribed relative speed while the four non-target cable rates remain zero. Subsequent force redistribution is retained by the full plant; no isolated peak formula replaces the measured trajectory.

| Relative speed [m/s] | First peak [N] |
|---:|---:|
{impact_rows}

The measured and pinned fleet impedance is **{PINNED_FLEET_IMPEDANCE:.12f} N s/m**.

## Scripted excursions

P1-T4 has no literal eligible mark because every scripted slack interval begins while the square gust is active. Its literal verdict is therefore `UNDER-POWERED`, not `PASS`. For the {t4['corrected_constant_gust_eligible_count']} excursions completed before gust shutoff, the physically corrected comparator uses acceleration `a0(1-lambda)` and passes.

P1-T5 uses {t5['complete_count']} complete and excludes {t5['censored_count']} censored excursions. The peak tension is based on the independently measured isolated engagement response and is checked against the pinned full-fleet impedance.

| Pretension [N] | Complete | Energy slope | Uncentred R2 |
|---:|---:|---:|---:|
{pretension_rows}

P1-T6's literal low-lambda branch fails because any positive gust reduces the restoring acceleration and therefore increases depth above `u^2/(2a0)`, even when `lambda < 1`. In contrast, direct force balance gives `a_slack/a0 = 1 - lambda`; its threshold is one in the reduced model and {t6['full_plant_acceleration_fit']['lambda_c']:.9f} in the full formation. The amended result is diagnostic only and does not reverse the literal NO-GO.

P1-T7 distinguishes depth at gust shutoff from maximum depth. The amended predictor adds the post-gust stopping distance when the shutoff velocity remains inward. Both comparisons satisfy the 15% criterion on their eligible sub-metre cases.

## Observed depth and sensitivity

The {t8['observed_maximum_complete_depth_m']:.9f} m value is the largest complete depth observed on this finite scripted grid, not a global geometric limit. Five primary-grid trajectories reach the {GEOMETRIC_DEPTH_LIMIT:.2f} m guard and are censored.

| Entry speed [m/s] | Complete | Censored | Observed maximum complete depth [m] |
|---:|---:|---:|---:|
{sensitivity_rows}

## Artifacts and scope

- `records/phase1/phase1_deterministic.json`: configuration, metrics, verdicts, and record digest.
- `records/phase1/phase1_deterministic.npz`: full-plant 1 ms trace, fleet impact points, and scripted excursion table.
- `reports/figures/phase1_F1_fleet_impedance.png`: fleet impact fit generated from the NPZ record.
- `reports/figures/phase1_F2_excursion_depth.png`: scripted depth grid generated from the NPZ record.

The stochastic 8 x 20 x 300 s campaign is not implemented or executed.
"""


def run_deterministic_campaign(
    results_path: Path = DETERMINISTIC_RESULTS_PATH,
    record_path: Path = DETERMINISTIC_RECORD_PATH,
    report_path: Path = DETERMINISTIC_REPORT_PATH,
) -> dict[str, object]:
    """Execute and record deterministic Phase 1 Stage 3-4."""
    fleet_tests = deterministic_fleet_acceptance()
    for result in fleet_tests.values():
        result["verdict"] = "PASS" if result["passed"] else "FAIL"
    scripted_tests, outcomes = scripted_acceptance()
    tests = {**fleet_tests, **scripted_tests}
    decisive = [
        test_id
        for test_id, result in tests.items()
        if result.get("decisive_no_go", False) or result.get("verdict") == "FAIL"
    ]
    pivoted = any(result.get("verdict") == "PIVOT" for result in tests.values())
    phase1_gate = "NO-GO" if decisive else "GO"
    deterministic_verdict = "PIVOT" if pivoted else ("FAIL" if decisive else "PASS")
    stop_reason = (
        "blocking tests failed: " + ", ".join(decisive) if decisive else None
    )

    speed, pretension = steady_state_values(constants.NOMINAL_THRUST)
    transit = run_full_formation(
        FullFormationConfig(pretension=pretension),
        0.25,
        common_speed=speed,
    )
    impact = run_full_formation(
        FullFormationConfig(pretension=pretension),
        0.25,
        common_speed=speed,
        target_relative_speed=1.0,
        target_elongation=0.0,
    )
    impact_event_code = np.zeros_like(impact.geometrically_engaged, dtype=np.int8)
    impact_event_code[1:] = np.diff(
        impact.geometrically_engaged.astype(np.int8), axis=0
    )
    arrays = _excursion_record_arrays(outcomes)
    arrays.update(
        {
            "fleet_first_peaks_n": np.asarray(
                fleet_tests["P1-T2b"]["first_peaks_n"]
            ),
            "fleet_impact_speeds_mps": FLEET_IMPACT_SPEEDS,
            "full_impact_elongation_m": impact.elongation,
            "full_impact_elongation_rate_mps": impact.elongation_rate,
            "full_impact_engagement_event_code": impact_event_code,
            "full_impact_force_positive": impact.force_positive,
            "full_impact_geometrically_engaged": impact.geometrically_engaged,
            "full_impact_tension_n": impact.tension,
            "full_impact_time_s": impact.time,
            "full_impact_truth": impact.truth,
            "full_transit_elongation_m": transit.elongation,
            "full_transit_elongation_rate_mps": transit.elongation_rate,
            "full_transit_force_positive": transit.force_positive,
            "full_transit_geometrically_engaged": transit.geometrically_engaged,
            "full_transit_tension_n": transit.tension,
            "full_transit_time_s": transit.time,
            "full_transit_truth": transit.truth,
        }
    )
    record_bytes = _npz_bytes(arrays)
    configuration = _deterministic_configuration()
    configuration_sha256 = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()
    try:
        recorded_path = str(record_path.relative_to(ROOT))
    except ValueError:
        recorded_path = str(record_path)
    results = {
        "schema_version": 1,
        "scope": "Phase 1 deterministic Stage 3-4 only",
        "phase1_gate": phase1_gate,
        "deterministic_verdict": deterministic_verdict,
        "stop_reason": stop_reason,
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "record_path": recorded_path,
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "tests": tests,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_bytes(record_bytes)
    results_path.write_bytes(_json_bytes(results))
    report_path.write_text(_format_deterministic_report(results), encoding="ascii")
    print(
        f"Phase 1 deterministic Stage 3-4: "
        f"{deterministic_verdict} ({phase1_gate})"
    )
    for test_id, result in tests.items():
        print(f"{test_id}: {result['verdict']}")
    return results


def _format_phase1r_report(results: dict[str, object]) -> str:
    tests = results["tests"]
    t1 = tests["P1-T1"]
    t2 = tests["P1-T2"]
    t2b = tests["P1-T2b"]
    t3 = tests["P1-T3"]
    t5 = tests["P1-T5"]
    t6 = tests["P1-T6"]
    t7 = tests["P1-T7"]
    t8 = tests["P1-T8"]
    t9 = tests["P1-T9"]
    energy_balance = t5["full_plant_energy_balance"]
    surrogate = t5["reduced_effective_mass_surrogate"]
    impact = t5["impact_calibration"]
    primary = t6["primary_estimator"]
    production = t9["production_discrete_1ms"]
    impact_slope = (
        "n/a"
        if impact["in_range_through_origin_slope"] is None
        else f"{impact['in_range_through_origin_slope']:.9f}"
    )
    refinement_rows = "\n".join(
        "| {step_s:.6f} | {maximum_energy_relative_drift:.9e} | "
        "{maximum_normalized_body_residual:.9e} | "
        "{normalized_total_momentum_drift:.9e} |".format(**row)
        for row in t9["continuous_refinement"]
    )
    sensitivity_rows = "\n".join(
        f"| {row['time_step_s']:.4f} | {1000 * row['fit_window_s']:.0f} | "
        f"{row['lambda_c']:.9f} | {row['bracketed_around_one']} | {row['status']} |"
        for row in t6["sensitivity"]
    )
    return f"""# Phase 1R Falsification Gate Report

## Recommendation

**{results['phase1_gate']} ({results['deterministic_verdict']}).** {results['stop_reason'] or 'All corrected deterministic blocking tests passed'}. P1-T5 full-plant energy conservation passes; its non-passing plan status is solely a low-speed through-origin impact-model specification failure, not a physical plant failure. **NO-GO is independently sustained by P1-T9 alone** at the production 1 ms step. The stochastic production campaign remains `{results['stochastic_campaign']['status']}`. No 8 x 20 x 300 s campaign was launched.

## Execution

- Full-plant cells: {results['execution']['cell_count']} ({len(PRETENSION_LEVELS)} pretensions x {len(PHASE1R_GUST_RATIOS)} gust ratios x {len(PHASE1R_GUST_DURATIONS)} durations x {len(PHASE1R_ENTRY_SPEEDS)} entry speeds).
- Seeds: {results['execution']['seeds']} (the deterministic grid consumes no random draws).
- Source revision: `{results['source']['revision']}`; worktree status: `{results['source']['status']}`.
- Wall time is observational, recorded in JSON, and excluded from deterministic replay comparison.
- Full plant: discrete Drake SAP at {constants.TIME_STEP:.6f} s; event sampling at {constants.TIME_STEP:.6f} s with interpolated crossings.
- Grid: pretensions {PRETENSION_LEVELS.tolist()} N; lambda {PHASE1R_GUST_RATIOS.tolist()}; durations {PHASE1R_GUST_DURATIONS.tolist()} s; entry speeds {PHASE1R_ENTRY_SPEEDS.tolist()} m/s.

## Corrected gate results

| Test | Measurement | Criterion | Status |
|---|---|---|---|
| P1-T1 | distance error {100 * t1['distance_relative_error']:.6e}%; tension error {100 * t1['tension_relative_error']:.6e}% | <=1%; <=3%; no slack | {t1['status']} |
| P1-T2 | Z = {t2['impedance_n_s_per_m']:.9f} N s/m; R2 = {t2['uncentred_r_squared']:.12f} | R2 > 0.99; analytic error <=15% | {t2['status']} |
| P1-T2b | required Z = {t2b['impedance_n_s_per_m']:.9f}; R2 = {t2b['uncentred_r_squared']:.12f}; baseline-subtracted Z = {t2b['baseline_subtracted_impedance_n_s_per_m']:.9f}; affine slope/intercept = {t2b['affine_slope_n_s_per_m']:.9f}/{t2b['affine_intercept_n']:.9f} | required through-origin R2 >0.99 | {t2b['status']} |
| P1-T3 | inferred mass {t3['estimated_reduced_mass_kg']:.9f} kg; error {100 * t3['relative_error']:.6f}% | <=10% | {t3['status']} |
| P1-T4 | not rerun; prior specification is outside this corrected gate | nonblocking | {tests['P1-T4']['status']} |
| P1-T5 | energy `{energy_balance['status']}` at max normalized residual {energy_balance['maximum_normalized_residual']:.9e}; impact `{impact['status']}` at in-domain through-origin slope {impact_slope} ({impact['in_range_count']} marks) | balance residual <= {energy_balance['relative_tolerance']:.3f}; at least {impact['minimum_mark_count']} calibrated marks and impact slope within 6% | {t5['status']} |
| P1-T6 | primary lambda_c = {primary['lambda_c']:.9f}; bracket = {primary['bracketed_around_one']}; sensitivity range [{t6['lambda_c_interval'][0]:.9f}, {t6['lambda_c_interval'][1]:.9f}] | primary 1 ms/5 ms estimator brackets 1 and abs(lambda_c-1) <=0.05 | {t6['status']} |
| P1-T7 | shutoff-depth max error {100 * t7['maximum_shutoff_depth_relative_error']:.6f}%; amended max-depth error {100 * t7['maximum_amended_depth_relative_error']:.6f}% over {t7['eligible_count']} cells | each <=15% for measured depth <1 m | {t7['status']} |
| P1-T8 | observed maximum {t8['observed_maximum_depth_m']:.9f} m; measured/right-censored = {t8['case_status_counts']['MEASURED']}/{t8['case_status_counts']['RIGHT_CENSORED']} | measurement/censoring only; no PASS claim | {t8['status']} |
| P1-T9 | discrete 1 ms energy drift {100 * production['maximum_energy_relative_drift']:.6f}%; body residual {100 * production['maximum_normalized_body_residual']:.6f}%; momentum drift {100 * production['normalized_total_momentum_drift']:.6f}% | energy <=0.5%; impulse and momentum diagnostics <=0.5% | {t9['status']} |

### P1-T2b low-speed residuals

- Required through-origin: {t2b['low_speed_residuals_n']['required_through_origin']:.9f} N.
- Baseline-subtracted: {t2b['low_speed_residuals_n']['baseline_subtracted']:.9f} N.
- Affine: {t2b['low_speed_residuals_n']['affine']:.9f} N.
- Zero-speed baseline: {t2b['zero_speed_baseline_n']:.9f} N.

### P1-T5 energy/work and impact calibration

The full-plant balance uses all six bodies and all five springs over each geometric-down to geometric-up excursion window. Its sign convention is `{energy_balance['sign_convention']}`. Positive drag work values would add energy; the measured linear and angular drag works are negative. Cable damping and force-clipping losses are reported as positive dissipations and subtracted. Square-gust work is evaluated as exact constant-force displacement over the active segment, without trapezoidal averaging across shutoff.

The reduced effective-mass identity is a separate surrogate diagnostic: slope {surrogate['through_origin_slope']:.9f}, uncentred R2 {surrogate['uncentred_r_squared']:.9f}, status `{surrogate['status']}`. It is not described as full-plant conservation.

Return-speed calibration covers [{impact['return_speed_range_mps'][0]:.9f}, {impact['return_speed_range_mps'][1]:.9f}] m/s. Counts below/in/above the validated [{impact['validated_speed_range_mps'][0]:.2f}, {impact['validated_speed_range_mps'][1]:.2f}] m/s range are {impact['below_range_count']}/{impact['in_range_count']}/{impact['above_range_count']}. Only in-domain marks determine `{impact['status']}`; the all-mark diagnostic slope is {impact['all_mark_diagnostic']['through_origin_slope']:.9f}. Over the observed in-domain range, the affine fit has slope {impact['in_range_affine_slope_n_s_per_m']:.9f} N s/m and intercept {impact['in_range_affine_intercept_n']:.9f} N. The intercept is {100 * impact['affine_intercept_fraction_at_min_speed']:.3f}% of the nominal impedance term at the lowest in-range speed. This rejects the through-origin low-speed model; it does not show energy nonconservation or a generic physical plant failure.

### P1-T6 corrected protocol

The original maximum-depth comparator is **WITHDRAWN_SPECIFICATION**. It has no physical FAIL and no threshold estimate. The corrected probe starts geometrically slack at {t6['initial_state']['target_elongation_m']:.1e} m, with zero target tension and zero target relative speed. The preregistered primary fit uses the 1 ms plant step over 5 ms and yields alpha = {primary['alpha']:.9f}, beta = {primary['beta']:.9f}, and lambda_c = {primary['lambda_c']:.9f}. The prior taut-to-slack transient fit is retained only as a non-primary diagnostic.

| Plant step [s] | Fit window [ms] | lambda_c | Brackets 1 | Status |
|---:|---:|---:|:---:|---|
{sensitivity_rows}

Sensitivity classification: **{t6['sensitivity_status']}**.

### P1-T9 refinement

| Step [s] | Energy drift | Max body residual | Total momentum drift |
|---:|---:|---:|---:|
{refinement_rows}

The production result is a separate discrete SAP run at 1 ms; it is not inferred from continuous RK3. Body impulse uses `{production['body_impulse_evaluation']}`, matching the production state update. The discrete energy failure alone remains blocking and is not overridden by the impulse diagnostic.

## Events and measurement provenance

All five full-plant cables receive connected relative-load inputs. Geometric down/up crossings, force onset/cessation, turning points, and rupture threshold crossings are separately sampled at 1 ms and interpolated. Reengagement and taut records are the actual `UnilateralCable` tracker records. P1-T5 archives full six-body state, all five cable states, total kinetic and spring energy, and gust, thrust, drag, weather, damping, and clipping channels. No peak is constructed as `Z*v_return`.

Replay regenerates into temporary paths without overwriting committed artifacts. It recomputes all per-cell energy/work terms and normalized residuals from the archived raw NPZ state, energy, and power channels, then compares them with the committed aggregate fields and JSON metrics at absolute tolerance {PHASE1R_REPLAY_NUMERIC_ATOL:.1e}. The manifest and report retain the generating revision and dirty-worktree status.

P1-T8 reports the finite-grid status `{t8['status']}`. The finite-geometry closure depth is {t8['closure_depth_m']:.6f} m. A global maximum was not established. Out-of-domain return marks are labeled rather than silently used as validated impedance predictions.

## Deviations and open items

1. The compact Phase 1R grid replaces the original production grid and is intentionally limited to 60 deterministic full-plant cells.
2. P1-T4 is not rerun because the review corrections target P1-T2b and P1-T5 through P1-T9; it is nonblocking here.
3. P1-T10 and the stochastic production campaign are `NOT_RUN`.
4. Wall time is observational and is not included in deterministic byte-identity promises.

## Artifacts

- `records/phase1/phase1r_records.npz`: full-plant target traces and flattened reengagement, taut, geometric, force-transition, turning, and rupture records.
- `records/phase1/phase1r_results.json`: computed metrics and programmatically derived gate fields.
- `records/phase1/phase1r_manifest.json`: complete configuration, hash, seeds, artifact digest, and replay contract.
- `reports/figures/phase1r_F1_acceleration_threshold.png` and `reports/figures/phase1r_F2_depth_comparison.png`: record-only figures.

The historical Phase 1 mechanics and deterministic reports are preserved as explicitly superseded artifacts.
"""


def _phase1r_manifest(
    configuration: dict[str, object],
    source: dict[str, object],
    artifacts: dict[str, dict[str, str]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "driver": "python -m tether.campaign.phase1 phase1r",
        "source": source,
        "runtime": {"python_version": platform.python_version()},
        "configuration": configuration,
        "configuration_sha256": _configuration_sha256(configuration),
        "acceptance_criteria": configuration["acceptance_criteria"],
        "seeds": list(PHASE1R_SEEDS),
        "artifacts": artifacts,
        "committed_record_path": artifacts["record"]["path"],
        "record_sha256": artifacts["record"]["sha256"],
        "probe": {
            "command": "python -m tether.campaign.phase1 replay",
            "numeric_absolute_tolerance": PHASE1R_REPLAY_NUMERIC_ATOL,
            "full_plant_energy_balance": (
                "recomputed from archived raw state, energy, and work channels"
            ),
            "full_plant_energy_fields_compared": 16,
            "event_times_comparison": "exact",
            "deterministic_record_bytes": True,
            "complete_results_excluding": ["execution.wall_seconds"],
            "report_bytes": True,
            "figure_bytes": True,
            "non_overwriting": True,
        },
    }


def run_phase1r_campaign(
    results_path: Path = PHASE1R_RESULTS_PATH,
    record_path: Path = PHASE1R_RECORD_PATH,
    manifest_path: Path = PHASE1R_MANIFEST_PATH,
    report_path: Path = PHASE1R_REPORT_PATH,
    figure_f1_path: Path = DEFAULT_PHASE1R_F1,
    figure_f2_path: Path = DEFAULT_PHASE1R_F2,
    *,
    mechanics_bundle=None,
    fleet_bundle=None,
    excursion_bundle=None,
) -> dict[str, object]:
    """Execute the corrected compact Phase 1R deterministic falsification gate."""
    start = time.perf_counter()
    source = _source_state()
    if mechanics_bundle is None:
        mechanics_bundle = mechanics_acceptance(
            include_evidence=True,
            _legacy_phase1r_sampling=True,
        )
    mechanics_tests, mechanics_evidence = mechanics_bundle
    if fleet_bundle is None:
        fleet_bundle = deterministic_fleet_acceptance(include_evidence=True)
    fleet_tests, fleet_evidence = fleet_bundle
    if excursion_bundle is None:
        excursion_bundle = phase1r_excursion_acceptance()
    excursion_tests, outcomes = excursion_bundle
    for result in mechanics_tests.values():
        result["status"] = "PASS" if result["passed"] else "FAIL"
    for result in fleet_tests.values():
        result["status"] = "PASS" if result["passed"] else "FAIL"
    tests = {
        "P1-T1": fleet_tests["P1-T1"],
        "P1-T2": mechanics_tests["P1-T2"],
        "P1-T2b": fleet_tests["P1-T2b"],
        "P1-T3": mechanics_tests["P1-T3"],
        "P1-T4": {
            "status": "NOT_RUN",
            "blocking": False,
            "reason": "not part of the corrected Phase 1R falsification gate",
        },
        **excursion_tests,
        "P1-T9": mechanics_tests["P1-T9"],
        "P1-T10": {"status": "NOT_RUN", "blocking": False},
    }
    phase1_gate, deterministic_verdict, stop_reason, failed = (
        _derive_phase1r_decision(tests)
    )
    stochastic_status = "NOT_RUN"
    stochastic_reason = (
        stop_reason
        if failed
        else "full stochastic production campaign excluded from Phase 1R"
    )
    arrays = _phase1r_record_arrays(outcomes, excursion_tests)
    arrays.update(mechanics_evidence)
    arrays.update(fleet_evidence)
    record_bytes = _npz_bytes(arrays)
    configuration = _phase1r_configuration()
    record_sha256 = hashlib.sha256(record_bytes).hexdigest()
    results = {
        "schema_version": 1,
        "scope": "Phase 1R corrected deterministic falsification gate",
        "phase1_gate": phase1_gate,
        "deterministic_verdict": deterministic_verdict,
        "stop_reason": stop_reason,
        "independent_no_go_basis": (
            ["P1-T9"] if tests["P1-T9"]["status"] != "PASS" else []
        ),
        "configuration": configuration,
        "configuration_sha256": _configuration_sha256(configuration),
        "record_sha256": record_sha256,
        "source": source,
        "execution": {
            "cell_count": len(outcomes),
            "seeds": list(PHASE1R_SEEDS),
            "wall_seconds": time.perf_counter() - start,
        },
        "tests": tests,
        "stochastic_campaign": {
            "status": stochastic_status,
            "reason": stochastic_reason,
            "production_grid_launched": False,
        },
    }
    for path in (
        results_path,
        record_path,
        manifest_path,
        report_path,
        figure_f1_path,
        figure_f2_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_bytes(record_bytes)
    results_path.write_bytes(_json_bytes(results))
    report_path.write_text(_format_phase1r_report(results), encoding="ascii")
    generate_phase1r_figures(record_path, figure_f1_path, figure_f2_path)
    artifacts = {
        "record": {
            "path": _recorded_path(record_path),
            "sha256": record_sha256,
        },
        "results": {
            "path": _recorded_path(results_path),
            "sha256": _sha256_file(results_path),
        },
        "report": {
            "path": _recorded_path(report_path),
            "sha256": _sha256_file(report_path),
        },
        "figure_f1": {
            "path": _recorded_path(figure_f1_path),
            "sha256": _sha256_file(figure_f1_path),
        },
        "figure_f2": {
            "path": _recorded_path(figure_f2_path),
            "sha256": _sha256_file(figure_f2_path),
        },
    }
    manifest_path.write_bytes(
        _json_bytes(_phase1r_manifest(configuration, source, artifacts))
    )
    print(f"Phase 1R gate: {phase1_gate} ({deterministic_verdict})")
    for test_id, result in tests.items():
        print(f"{test_id}: {result['status']}")
    return results


def replay_phase1r(
    results_path: Path = PHASE1R_RESULTS_PATH,
    record_path: Path = PHASE1R_RECORD_PATH,
    manifest_path: Path = PHASE1R_MANIFEST_PATH,
    report_path: Path = PHASE1R_REPORT_PATH,
    figure_f1_path: Path = DEFAULT_PHASE1R_F1,
    figure_f2_path: Path = DEFAULT_PHASE1R_F2,
    *,
    campaign_inputs: dict[str, object] | None = None,
) -> bool:
    """Regenerate Phase 1R in temporary paths and compare without overwriting."""
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    configuration = _phase1r_configuration()
    if manifest["configuration_sha256"] != _configuration_sha256(configuration):
        raise RuntimeError("current Phase 1R configuration does not match the manifest")
    committed_artifact_paths = {
        "record": PHASE1R_RECORD_PATH,
        "results": results_path,
        "report": report_path,
        "figure_f1": figure_f1_path,
        "figure_f2": figure_f2_path,
    }
    committed_artifact_paths["record"] = record_path
    manifest_digests_match = all(
        manifest["artifacts"][name]["path"] == _recorded_path(path)
        and manifest["artifacts"][name]["sha256"] == _sha256_file(path)
        for name, path in committed_artifact_paths.items()
    )
    with tempfile.TemporaryDirectory(prefix="phase1r-replay-") as directory:
        temporary = Path(directory)
        replay_results = run_phase1r_campaign(
            temporary / "results.json",
            temporary / "records.npz",
            temporary / "manifest.json",
            temporary / "report.md",
            temporary / "F1.png",
            temporary / "F2.png",
            **(campaign_inputs or {}),
        )
        exact = {}
        maximum_errors = {}
        with np.load(record_path, allow_pickle=False) as committed, np.load(
            temporary / "records.npz", allow_pickle=False
        ) as replayed:
            names_match = set(committed.files) == set(replayed.files)
            for name in sorted(set(committed.files) & set(replayed.files)):
                first = committed[name]
                second = replayed[name]
                exact_comparison = (
                    name.startswith(("event_", "reengagement_", "taut_"))
                    or first.dtype.kind in "USb"
                )
                exact[name] = bool(
                    np.array_equal(first, second)
                    if first.dtype.kind in "US"
                    else np.array_equal(first, second, equal_nan=True)
                )
                if first.dtype.kind in "iufc" and first.size:
                    maximum_errors[name] = float(
                        np.nanmax(np.abs(first.astype(float) - second.astype(float)))
                    )
                else:
                    maximum_errors[name] = 0.0
                if not exact_comparison:
                    exact[name] = bool(
                        np.allclose(first, second, rtol=0.0, atol=1.0e-6, equal_nan=True)
                    )
        replay_bytes = (temporary / "records.npz").read_bytes()
        byte_identical = replay_bytes == record_path.read_bytes()
        digest_matches = (
            hashlib.sha256(replay_bytes).hexdigest() == manifest["record_sha256"]
        )
        committed_results = json.loads(results_path.read_text(encoding="ascii"))
        with np.load(record_path, allow_pickle=False) as committed:
            committed_energy_checks, committed_energy_balances = (
                _phase1r_energy_balance_replay_checks(
                    committed,
                    committed_results,
                    absolute_tolerance=PHASE1R_REPLAY_NUMERIC_ATOL,
                )
            )
        committed_energy_metrics_match = all(committed_energy_checks.values())
        replay_comparable = json.loads(json.dumps(replay_results, sort_keys=True))
        committed_comparable = json.loads(
            json.dumps(committed_results, sort_keys=True)
        )
        replay_comparable["execution"].pop("wall_seconds", None)
        committed_comparable["execution"].pop("wall_seconds", None)
        complete_results_match = replay_comparable == committed_comparable
        report_identical = (
            (temporary / "report.md").read_bytes() == report_path.read_bytes()
        )
        figure_f1_identical = (
            (temporary / "F1.png").read_bytes() == figure_f1_path.read_bytes()
        )
        figure_f2_identical = (
            (temporary / "F2.png").read_bytes() == figure_f2_path.read_bytes()
        )
        replay_manifest = json.loads(
            (temporary / "manifest.json").read_text(encoding="ascii")
        )
        record_metric_checks = _phase1r_record_metric_checks(
            temporary / "records.npz", replay_results
        )
        record_metrics_match = all(record_metric_checks.values())
        manifest_contract_fields = (
            "schema_version",
            "driver",
            "source",
            "runtime",
            "configuration",
            "configuration_sha256",
            "acceptance_criteria",
            "seeds",
            "probe",
        )
        manifest_contract_match = all(
            replay_manifest[field] == manifest[field]
            for field in manifest_contract_fields
        )
    passed = bool(
        names_match
        and all(exact.values())
        and byte_identical
        and digest_matches
        and complete_results_match
        and report_identical
        and figure_f1_identical
        and figure_f2_identical
        and manifest_digests_match
        and manifest_contract_match
        and record_metrics_match
        and committed_energy_metrics_match
    )
    evidence = {
        "temporary_regeneration": True,
        "committed_files_overwritten": False,
        "array_names_match": names_match,
        "array_comparisons": exact,
        "maximum_absolute_errors": maximum_errors,
        "event_times_exact": exact.get("event_time_s", False),
        "byte_identical": byte_identical,
        "digest_matches": digest_matches,
        "complete_results_match_excluding_wall_time": complete_results_match,
        "report_byte_identical": report_identical,
        "figure_f1_byte_identical": figure_f1_identical,
        "figure_f2_byte_identical": figure_f2_identical,
        "manifest_digests_match": manifest_digests_match,
        "manifest_contract_match": manifest_contract_match,
        "record_metric_comparisons": record_metric_checks,
        "record_metrics_match": record_metrics_match,
        "raw_energy_balance_recomputation": {
            "source": "committed raw NPZ channels",
            "absolute_tolerance": PHASE1R_REPLAY_NUMERIC_ATOL,
            "recomputed_maximum_normalized_residual": float(
                np.nanmax(committed_energy_balances["normalized_residual"])
            ),
            "comparisons": committed_energy_checks,
            "passed": committed_energy_metrics_match,
        },
        "source_provenance": manifest["source"],
        "committed_files_overwritten": False,
        "passed": passed,
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("Phase 1R replay probe failed")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("mechanics", "deterministic", "phase1r", "replay")
    )
    arguments = parser.parse_args()
    if arguments.command == "mechanics":
        run_mechanics_campaign()
    elif arguments.command == "deterministic":
        run_deterministic_campaign()
    elif arguments.command == "phase1r":
        run_phase1r_campaign()
    elif arguments.command == "replay":
        replay_phase1r()


if __name__ == "__main__":
    main()