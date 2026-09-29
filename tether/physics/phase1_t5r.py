"""Deterministic low-speed full-formation P1-T5R identification study."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tether.physics import constants
from tether.physics.phase1_deterministic import (
    GEOMETRIC_DEPTH_LIMIT,
    PINNED_FLEET_IMPEDANCE,
    PRETENSION_LEVELS,
    FullFormationConfig,
    FullFormationTrajectory,
    run_full_formation,
    run_phase1r_excursion,
)

P1_T5R_PHYSICS_STEPS = (5.0e-4, 2.5e-4)
P1_T5R_PRODUCTION_STEP = 2.5e-4
P1_T5R_RECORDING_STEP = 1.0e-3
P1_T5R_NOMINAL_SPEEDS = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
P1_T5R_FORCING_SPEEDS = np.array([0.14, 0.255, 0.37, 0.485, 0.60])
P1_T5R_ZERO_REPEATS = 2
P1_T5R_GUST_RATIO = 0.0
P1_T5R_GUST_DURATION = 0.05
P1_T5R_HORIZON = 1.0
P1_T5R_BASELINE_DURATION = 0.25
P1_T5R_CALIBRATION_TOLERANCE = 0.06
P1_T5R_R2_MINIMUM = 0.99
P1_T5R_MAXIMUM_RELATIVE_ERROR = 0.06
P1_T5R_WORKERS = 4

MODEL_FORMULAS = {
    "A": "T_peak = Z_pinned * v_return",
    "B": "T_peak = T0(pretension) + Z_dyn * v_return",
    "C": "T_peak = alpha(pretension) + Z_dyn * v_return",
}
PRIMARY_MODEL = "B"
MODEL_C_JUSTIFICATION = (
    "Existing Phase 1R and P1-T2b data show a nonzero low-speed affine "
    "intercept, while the three pretension levels impose distinct static loads."
)

_HELD_OUT_SPEEDS = {
    742.5: (0.20, 0.40),
    990.0: (0.10, 0.30, 0.50),
    1237.5: (0.20, 0.40),
}


@dataclass(frozen=True)
class P1T5RCellSpec:
    physics_step_s: float
    pretension_n: float
    nominal_speed_mps: float
    forcing_entry_speed_mps: float
    repetition: int
    split: str

    @property
    def identifier(self) -> str:
        return (
            f"dt{1.0e6 * self.physics_step_s:.0f}_"
            f"p{self.pretension_n:.1f}_v{self.nominal_speed_mps:.2f}_"
            f"r{self.repetition}"
        )


@dataclass(frozen=True)
class P1T5RCell:
    spec: P1T5RCellSpec
    trajectory: FullFormationTrajectory


def split_for_cell(pretension_n: float, nominal_speed_mps: float) -> str:
    """Return the preregistered checkerboard train/held-out assignment."""
    if np.isclose(nominal_speed_mps, 0.0):
        return "train"
    held_out = _HELD_OUT_SPEEDS[float(pretension_n)]
    return (
        "heldout"
        if any(np.isclose(nominal_speed_mps, speed) for speed in held_out)
        else "train"
    )


def preregistered_cell_specs() -> tuple[P1T5RCellSpec, ...]:
    """Build the immutable P1-T5R grid and split before simulation."""
    specs = []
    for physics_step in P1_T5R_PHYSICS_STEPS:
        for pretension in PRETENSION_LEVELS:
            for nominal_speed, forcing_speed in zip(
                P1_T5R_NOMINAL_SPEEDS,
                P1_T5R_FORCING_SPEEDS,
            ):
                specs.append(
                    P1T5RCellSpec(
                        physics_step_s=float(physics_step),
                        pretension_n=float(pretension),
                        nominal_speed_mps=float(nominal_speed),
                        forcing_entry_speed_mps=float(forcing_speed),
                        repetition=0,
                        split=split_for_cell(float(pretension), float(nominal_speed)),
                    )
                )
            for repetition in range(P1_T5R_ZERO_REPEATS):
                specs.append(
                    P1T5RCellSpec(
                        physics_step_s=float(physics_step),
                        pretension_n=float(pretension),
                        nominal_speed_mps=0.0,
                        forcing_entry_speed_mps=0.0,
                        repetition=repetition,
                        split="train",
                    )
                )
    return tuple(specs)


def _thrust_for_pretension(pretension_n: float) -> tuple[float, float]:
    common_speed = (
        constants.VESSEL_COUNT * pretension_n / constants.LOAD_LINEAR_DRAG
    )
    thrust = common_speed * (
        constants.LOAD_LINEAR_DRAG
        + constants.VESSEL_COUNT * constants.VESSEL_LINEAR_DRAG
    ) / constants.VESSEL_COUNT
    return thrust, common_speed


def _run_cell(spec: P1T5RCellSpec) -> P1T5RCell:
    if np.isclose(spec.nominal_speed_mps, 0.0):
        thrust, common_speed = _thrust_for_pretension(spec.pretension_n)
        trajectory = run_full_formation(
            FullFormationConfig(
                thrusts=thrust,
                pretension=spec.pretension_n,
                closure_depth_m=GEOMETRIC_DEPTH_LIMIT,
                time_step_s=spec.physics_step_s,
                recording_step_s=P1_T5R_RECORDING_STEP,
            ),
            P1_T5R_BASELINE_DURATION,
            common_speed=common_speed,
            target_relative_speed=0.0,
            target_elongation=0.0,
        )
    else:
        outcome = run_phase1r_excursion(
            spec.pretension_n,
            P1_T5R_GUST_RATIO,
            P1_T5R_GUST_DURATION,
            spec.forcing_entry_speed_mps,
            time_step_s=spec.physics_step_s,
            recording_step_s=P1_T5R_RECORDING_STEP,
            horizon_s=P1_T5R_HORIZON,
        )
        trajectory = outcome.trajectory
    return P1T5RCell(spec=spec, trajectory=trajectory)


def run_p1_t5r_grid(max_workers: int = P1_T5R_WORKERS) -> tuple[P1T5RCell, ...]:
    """Run the preregistered full-formation identification and sensitivity grid."""
    specs = preregistered_cell_specs()
    if max_workers == 1:
        return tuple(_run_cell(spec) for spec in specs)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return tuple(executor.map(_run_cell, specs))


def p1_t5r_record_arrays(cells: Sequence[P1T5RCell]) -> dict[str, np.ndarray]:
    """Archive cell membership and full raw channels at the 1 ms cadence."""
    maximum_samples = max(cell.trajectory.time.size for cell in cells)
    state_width = cells[0].trajectory.truth.shape[1]
    count = len(cells)
    arrays: dict[str, np.ndarray] = {
        "cell_identifier": np.array([cell.spec.identifier for cell in cells], dtype="U48"),
        "cell_physics_step_s": np.array(
            [cell.spec.physics_step_s for cell in cells], dtype=float
        ),
        "cell_pretension_n": np.array(
            [cell.spec.pretension_n for cell in cells], dtype=float
        ),
        "cell_nominal_speed_mps": np.array(
            [cell.spec.nominal_speed_mps for cell in cells], dtype=float
        ),
        "cell_forcing_entry_speed_mps": np.array(
            [cell.spec.forcing_entry_speed_mps for cell in cells], dtype=float
        ),
        "cell_repetition": np.array(
            [cell.spec.repetition for cell in cells], dtype=np.int64
        ),
        "cell_split": np.array([cell.spec.split for cell in cells], dtype="U8"),
        "trace_sample_count": np.array(
            [cell.trajectory.time.size for cell in cells], dtype=np.int64
        ),
        "trace_time_s": np.full((count, maximum_samples), np.nan),
        "trace_truth": np.full((count, maximum_samples, state_width), np.nan),
        "trace_all_tension_n": np.full(
            (count, maximum_samples, constants.VESSEL_COUNT), np.nan
        ),
        "trace_all_elongation_m": np.full(
            (count, maximum_samples, constants.VESSEL_COUNT), np.nan
        ),
        "trace_all_elongation_rate_mps": np.full(
            (count, maximum_samples, constants.VESSEL_COUNT), np.nan
        ),
        "trace_all_relative_load_n": np.full(
            (count, maximum_samples, constants.VESSEL_COUNT), np.nan
        ),
    }
    for cell_index, cell in enumerate(cells):
        trajectory = cell.trajectory
        sample_count = trajectory.time.size
        arrays["trace_time_s"][cell_index, :sample_count] = trajectory.time
        arrays["trace_truth"][cell_index, :sample_count] = trajectory.truth
        arrays["trace_all_tension_n"][cell_index, :sample_count] = trajectory.tension
        arrays["trace_all_elongation_m"][cell_index, :sample_count] = (
            trajectory.elongation
        )
        arrays["trace_all_elongation_rate_mps"][cell_index, :sample_count] = (
            trajectory.elongation_rate
        )
        arrays["trace_all_relative_load_n"][cell_index, :sample_count] = (
            trajectory.relative_load
        )
    return arrays


def _crossing_rate(
    first_elongation: float,
    second_elongation: float,
    first_rate: float,
    second_rate: float,
) -> float:
    fraction = -first_elongation / (second_elongation - first_elongation)
    return float(first_rate + fraction * (second_rate - first_rate))


def _first_peak(
    tension: np.ndarray,
    start_index: int,
    crossing_rate: float = 0.0,
) -> float:
    peak = max(
        constants.CABLE_DAMPING * max(0.0, crossing_rate),
        float(tension[start_index]),
    )
    for value in tension[start_index + 1 :]:
        value = float(value)
        if value < peak:
            return peak
        peak = max(peak, value)
    raise RuntimeError("raw trajectory ended before the first tension peak")


def _measure_raw_cell(record, cell_index: int) -> dict[str, object]:
    sample_count = int(record["trace_sample_count"][cell_index])
    cable = constants.VESSEL_COUNT // 2
    time_values = record["trace_time_s"][cell_index, :sample_count]
    elongation = record["trace_all_elongation_m"][
        cell_index, :sample_count, cable
    ]
    rate = record["trace_all_elongation_rate_mps"][
        cell_index, :sample_count, cable
    ]
    tension = record["trace_all_tension_n"][cell_index, :sample_count, cable]
    nominal_speed = float(record["cell_nominal_speed_mps"][cell_index])
    if np.isclose(nominal_speed, 0.0):
        positive = np.flatnonzero(elongation > 0.0)
        if not positive.size:
            raise RuntimeError("zero-speed control never developed positive elongation")
        start_index = int(positive[0])
        return_speed = 0.0
        entry_speed = 0.0
        peak = _first_peak(tension, start_index)
        down_time = None
        up_time = None
    else:
        down_candidates = np.flatnonzero(
            (elongation[:-1] > 0.0) & (elongation[1:] <= 0.0)
        )
        if not down_candidates.size:
            raise RuntimeError("impact cell never entered geometric slack")
        down_left = int(down_candidates[0])
        down_right = down_left + 1
        entry_speed = max(
            0.0,
            -_crossing_rate(
                elongation[down_left],
                elongation[down_right],
                rate[down_left],
                rate[down_right],
            ),
        )
        down_fraction = -elongation[down_left] / (
            elongation[down_right] - elongation[down_left]
        )
        down_time = float(
            time_values[down_left]
            + down_fraction * (time_values[down_right] - time_values[down_left])
        )
        up_candidates = np.flatnonzero(
            (elongation[down_right:-1] <= 0.0)
            & (elongation[down_right + 1 :] > 0.0)
        )
        if not up_candidates.size:
            raise RuntimeError("impact cell never returned from geometric slack")
        up_left = down_right + int(up_candidates[0])
        up_right = up_left + 1
        return_speed = max(
            0.0,
            _crossing_rate(
                elongation[up_left],
                elongation[up_right],
                rate[up_left],
                rate[up_right],
            ),
        )
        up_fraction = -elongation[up_left] / (
            elongation[up_right] - elongation[up_left]
        )
        up_time = float(
            time_values[up_left]
            + up_fraction * (time_values[up_right] - time_values[up_left])
        )
        peak = _first_peak(tension, up_right, return_speed)
    return {
        "identifier": str(record["cell_identifier"][cell_index]),
        "physics_step_s": float(record["cell_physics_step_s"][cell_index]),
        "pretension_n": float(record["cell_pretension_n"][cell_index]),
        "nominal_speed_mps": nominal_speed,
        "forcing_entry_speed_mps": float(
            record["cell_forcing_entry_speed_mps"][cell_index]
        ),
        "repetition": int(record["cell_repetition"][cell_index]),
        "split": str(record["cell_split"][cell_index]),
        "measured_entry_speed_mps": entry_speed,
        "measured_return_speed_mps": return_speed,
        "measured_peak_tension_n": peak,
        "geometric_down_time_s": down_time,
        "geometric_up_time_s": up_time,
        "sample_count": sample_count,
    }


def _prediction_metrics(
    measurements: Sequence[dict[str, object]],
    predictions: np.ndarray,
) -> dict[str, object]:
    observed = np.array(
        [row["measured_peak_tension_n"] for row in measurements], dtype=float
    )
    residual = observed - predictions
    calibration_slope = float(
        np.dot(predictions, observed) / np.dot(predictions, predictions)
    )
    uncentred_r_squared = float(
        1.0 - np.dot(residual, residual) / np.dot(observed, observed)
    )
    relative_errors = np.abs(residual) / observed
    metrics = {
        "count": len(measurements),
        "calibration_slope_through_origin": calibration_slope,
        "calibration_slope_relative_error": abs(calibration_slope - 1.0),
        "uncentred_r_squared": uncentred_r_squared,
        "maximum_relative_peak_error": float(np.max(relative_errors)),
        "mae_n": float(np.mean(np.abs(residual))),
        "rmse_n": float(np.sqrt(np.mean(residual**2))),
        "minimum_prediction_n": float(np.min(predictions)),
    }
    per_pretension = {}
    for pretension in PRETENSION_LEVELS:
        selected = np.array(
            [np.isclose(row["pretension_n"], pretension) for row in measurements]
        )
        per_pretension[str(float(pretension))] = _prediction_metrics_flat(
            observed[selected], predictions[selected]
        )
    metrics["per_pretension"] = per_pretension
    return metrics


def _prediction_metrics_flat(
    observed: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    residual = observed - predictions
    calibration_slope = float(
        np.dot(predictions, observed) / np.dot(predictions, predictions)
    )
    return {
        "count": int(observed.size),
        "calibration_slope_through_origin": calibration_slope,
        "calibration_slope_relative_error": abs(calibration_slope - 1.0),
        "uncentred_r_squared": float(
            1.0 - np.dot(residual, residual) / np.dot(observed, observed)
        ),
        "maximum_relative_peak_error": float(
            np.max(np.abs(residual) / observed)
        ),
        "mae_n": float(np.mean(np.abs(residual))),
        "rmse_n": float(np.sqrt(np.mean(residual**2))),
        "minimum_prediction_n": float(np.min(predictions)),
    }


def _acceptance_checks(
    metrics: dict[str, object],
    slope: float,
) -> dict[str, bool]:
    return {
        "calibration_slope_within_6_percent": bool(
            metrics["calibration_slope_relative_error"]
            <= P1_T5R_CALIBRATION_TOLERANCE
        ),
        "uncentred_r_squared_above_0_99": bool(
            metrics["uncentred_r_squared"] > P1_T5R_R2_MINIMUM
        ),
        "maximum_relative_peak_error_at_most_6_percent": bool(
            metrics["maximum_relative_peak_error"]
            <= P1_T5R_MAXIMUM_RELATIVE_ERROR
        ),
        "nonnegative_dynamic_slope": bool(slope >= 0.0),
        "nonnegative_predictions": bool(metrics["minimum_prediction_n"] >= 0.0),
    }


def _passes(checks: dict[str, bool]) -> bool:
    return all(checks.values())


def fit_candidate_models(
    measurements: Sequence[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Fit on training cells only and score the preregistered held-out cells."""
    production = [
        row
        for row in measurements
        if np.isclose(row["physics_step_s"], P1_T5R_PRODUCTION_STEP)
    ]
    training = [row for row in production if row["split"] == "train"]
    held_out = [
        row
        for row in production
        if row["split"] == "heldout" and row["nominal_speed_mps"] > 0.0
    ]
    training_dynamic = [
        row for row in training if row["nominal_speed_mps"] > 0.0
    ]
    baseline_by_pretension = {
        float(pretension): float(
            np.mean(
                [
                    row["measured_peak_tension_n"]
                    for row in training
                    if np.isclose(row["pretension_n"], pretension)
                    and np.isclose(row["nominal_speed_mps"], 0.0)
                ]
            )
        )
        for pretension in PRETENSION_LEVELS
    }
    held_out_speeds = np.array(
        [row["measured_return_speed_mps"] for row in held_out], dtype=float
    )

    prediction_a = PINNED_FLEET_IMPEDANCE * held_out_speeds
    metrics_a = _prediction_metrics(held_out, prediction_a)
    checks_a = _acceptance_checks(metrics_a, PINNED_FLEET_IMPEDANCE)

    train_speeds = np.array(
        [row["measured_return_speed_mps"] for row in training_dynamic], dtype=float
    )
    train_dynamic_peaks = np.array(
        [
            row["measured_peak_tension_n"]
            - baseline_by_pretension[float(row["pretension_n"])]
            for row in training_dynamic
        ],
        dtype=float,
    )
    slope_b = float(
        np.dot(train_speeds, train_dynamic_peaks)
        / np.dot(train_speeds, train_speeds)
    )
    prediction_b = np.array(
        [
            baseline_by_pretension[float(row["pretension_n"])]
            + slope_b * row["measured_return_speed_mps"]
            for row in held_out
        ],
        dtype=float,
    )
    metrics_b = _prediction_metrics(held_out, prediction_b)
    checks_b = _acceptance_checks(metrics_b, slope_b)

    design = np.zeros((len(training), len(PRETENSION_LEVELS) + 1))
    response = np.array(
        [row["measured_peak_tension_n"] for row in training], dtype=float
    )
    for row_index, row in enumerate(training):
        pretension_index = int(
            np.flatnonzero(np.isclose(PRETENSION_LEVELS, row["pretension_n"]))[0]
        )
        design[row_index, pretension_index] = 1.0
        design[row_index, -1] = row["measured_return_speed_mps"]
    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    intercepts_c = {
        float(pretension): float(coefficients[index])
        for index, pretension in enumerate(PRETENSION_LEVELS)
    }
    slope_c = float(coefficients[-1])
    prediction_c = np.array(
        [
            intercepts_c[float(row["pretension_n"])]
            + slope_c * row["measured_return_speed_mps"]
            for row in held_out
        ],
        dtype=float,
    )
    metrics_c = _prediction_metrics(held_out, prediction_c)
    checks_c = _acceptance_checks(metrics_c, slope_c)

    return {
        "A": {
            "formula": MODEL_FORMULAS["A"],
            "parameters": {"Z_pinned_n_s_per_m": PINNED_FLEET_IMPEDANCE},
            "fit_cell_identifiers": [],
            "heldout_metrics": metrics_a,
            "acceptance_checks": checks_a,
            "heldout_thresholds_satisfied": _passes(checks_a),
            "qualified": False,
            "status": "RETAINED_FAILED_BASELINE",
            "historical_failure_preserved": True,
        },
        "B": {
            "formula": MODEL_FORMULAS["B"],
            "parameters": {
                "baseline_by_pretension_n": {
                    str(key): value for key, value in baseline_by_pretension.items()
                },
                "Z_dyn_n_s_per_m": slope_b,
            },
            "fit_cell_identifiers": [row["identifier"] for row in training],
            "heldout_metrics": metrics_b,
            "acceptance_checks": checks_b,
            "heldout_thresholds_satisfied": _passes(checks_b),
            "qualified": _passes(checks_b),
            "status": "PASS" if _passes(checks_b) else "FAIL",
            "primary": True,
        },
        "C": {
            "formula": MODEL_FORMULAS["C"],
            "parameters": {
                "intercept_by_pretension_n": {
                    str(key): value for key, value in intercepts_c.items()
                },
                "Z_dyn_n_s_per_m": slope_c,
            },
            "fit_cell_identifiers": [row["identifier"] for row in training],
            "heldout_metrics": metrics_c,
            "acceptance_checks": checks_c,
            "heldout_thresholds_satisfied": _passes(checks_c),
            "qualified": _passes(checks_c),
            "status": "PASS" if _passes(checks_c) else "FAIL",
            "included": True,
            "justification": MODEL_C_JUSTIFICATION,
        },
    }


def _range(values: Sequence[float]) -> list[float]:
    return [float(np.min(values)), float(np.max(values))]


def analyse_p1_t5r_record(record) -> dict[str, object]:
    """Recompute cell targets, fits, held-out metrics, and diagnostics from raw data."""
    measurements = [
        _measure_raw_cell(record, cell_index)
        for cell_index in range(record["cell_identifier"].size)
    ]
    candidates = fit_candidate_models(measurements)
    production = [
        row
        for row in measurements
        if np.isclose(row["physics_step_s"], P1_T5R_PRODUCTION_STEP)
    ]
    dynamic = [row for row in production if row["nominal_speed_mps"] > 0.0]
    held_out = [row for row in dynamic if row["split"] == "heldout"]
    baseline_repeatability = []
    for step in P1_T5R_PHYSICS_STEPS:
        for pretension in PRETENSION_LEVELS:
            peaks = np.array(
                [
                    row["measured_peak_tension_n"]
                    for row in measurements
                    if np.isclose(row["physics_step_s"], step)
                    and np.isclose(row["pretension_n"], pretension)
                    and np.isclose(row["nominal_speed_mps"], 0.0)
                ]
            )
            baseline_repeatability.append(
                {
                    "physics_step_s": step,
                    "pretension_n": float(pretension),
                    "repeat_count": int(peaks.size),
                    "mean_peak_tension_n": float(np.mean(peaks)),
                    "peak_range_n": float(np.ptp(peaks)),
                    "maximum_relative_deviation": float(
                        np.max(np.abs(peaks - np.mean(peaks))) / np.mean(peaks)
                    ),
                }
            )
    sensitivity_rows = []
    for fine in production:
        coarse = next(
            row
            for row in measurements
            if np.isclose(row["physics_step_s"], 5.0e-4)
            and np.isclose(row["pretension_n"], fine["pretension_n"])
            and np.isclose(row["nominal_speed_mps"], fine["nominal_speed_mps"])
            and row["repetition"] == fine["repetition"]
        )
        sensitivity_rows.append(
            {
                "pretension_n": fine["pretension_n"],
                "nominal_speed_mps": fine["nominal_speed_mps"],
                "repetition": fine["repetition"],
                "return_speed_absolute_difference_mps": abs(
                    fine["measured_return_speed_mps"]
                    - coarse["measured_return_speed_mps"]
                ),
                "peak_tension_absolute_difference_n": abs(
                    fine["measured_peak_tension_n"]
                    - coarse["measured_peak_tension_n"]
                ),
                "peak_tension_relative_difference": abs(
                    fine["measured_peak_tension_n"]
                    / coarse["measured_peak_tension_n"]
                    - 1.0
                ),
            }
        )
    selected_model = (
        "B"
        if candidates["B"]["qualified"]
        else "C"
        if candidates["C"]["qualified"]
        else None
    )
    return {
        "measurements": measurements,
        "split": {
            "method": "preregistered checkerboard transfer across speed and pretension",
            "training_cell_identifiers": [
                row["identifier"] for row in production if row["split"] == "train"
            ],
            "heldout_cell_identifiers": [
                row["identifier"] for row in production if row["split"] == "heldout"
            ],
        },
        "achieved_domain": {
            "nominal_speed_range_mps": _range(
                [row["nominal_speed_mps"] for row in dynamic]
            ),
            "forcing_entry_speed_range_mps": _range(
                [row["forcing_entry_speed_mps"] for row in dynamic]
            ),
            "measured_entry_speed_range_mps": _range(
                [row["measured_entry_speed_mps"] for row in dynamic]
            ),
            "measured_return_speed_range_mps": _range(
                [row["measured_return_speed_mps"] for row in dynamic]
            ),
            "heldout_measured_entry_speed_range_mps": _range(
                [row["measured_entry_speed_mps"] for row in held_out]
            ),
            "heldout_measured_return_speed_range_mps": _range(
                [row["measured_return_speed_mps"] for row in held_out]
            ),
            "per_pretension": {
                str(float(pretension)): {
                    "measured_entry_speed_range_mps": _range(
                        [
                            row["measured_entry_speed_mps"]
                            for row in dynamic
                            if np.isclose(row["pretension_n"], pretension)
                        ]
                    ),
                    "measured_return_speed_range_mps": _range(
                        [
                            row["measured_return_speed_mps"]
                            for row in dynamic
                            if np.isclose(row["pretension_n"], pretension)
                        ]
                    ),
                }
                for pretension in PRETENSION_LEVELS
            },
        },
        "candidates": candidates,
        "primary_candidate": PRIMARY_MODEL,
        "selected_model": selected_model,
        "status": "PASS" if selected_model is not None else "FAIL",
        "baseline_repeatability": baseline_repeatability,
        "physics_step_sensitivity": {
            "comparisons": sensitivity_rows,
            "maximum_return_speed_absolute_difference_mps": max(
                row["return_speed_absolute_difference_mps"]
                for row in sensitivity_rows
            ),
            "maximum_peak_tension_absolute_difference_n": max(
                row["peak_tension_absolute_difference_n"]
                for row in sensitivity_rows
            ),
            "maximum_peak_tension_relative_difference": max(
                row["peak_tension_relative_difference"]
                for row in sensitivity_rows
            ),
        },
    }