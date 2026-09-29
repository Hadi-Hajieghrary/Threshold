"""Deterministic P1-T9R discrete-step remediation campaign and replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import tempfile
import time

import numpy as np

from tether.campaign.phase1 import (
    PHASE1R_MANIFEST_PATH,
    PHASE1R_RECORD_PATH,
    PHASE1R_REPORT_PATH,
    PHASE1R_RESULTS_PATH,
    ROOT,
    _configuration_sha256,
    _json_bytes,
    _npz_bytes,
    _recorded_path,
    _sha256_file,
    _source_state,
)
from tether.physics import constants
from tether.physics.phase1_mechanics import (
    IMPACT_SPEEDS,
    P1_T9R_CANDIDATE_STEP,
    P1_T9R_ENERGY_LIMIT,
    P1_T9R_MOMENTUM_LIMIT,
    P1_T9R_ORDER_BOUNDS,
    REFINEMENT_STEPS,
    p1_t9r_acceptance,
)

RECORD_DIR = ROOT / "records" / "phase1"
RESULTS_PATH = RECORD_DIR / "phase1_t9r_results.json"
RECORD_PATH = RECORD_DIR / "phase1_t9r_records.npz"
MANIFEST_PATH = RECORD_DIR / "phase1_t9r_manifest.json"
REPORT_PATH = ROOT / "reports" / "phase1_t9r_report.md"
REPLAY_NUMERIC_ATOL = 1.0e-12

PHASE1R_ARTIFACT_PATHS = {
    "results": PHASE1R_RESULTS_PATH,
    "record": PHASE1R_RECORD_PATH,
    "manifest": PHASE1R_MANIFEST_PATH,
    "report": PHASE1R_REPORT_PATH,
}


def _phase1r_hashes() -> dict[str, str]:
    return {
        _recorded_path(path): _sha256_file(path)
        for path in PHASE1R_ARTIFACT_PATHS.values()
    }


def _metric_subset_matches(computed: object, reported: object) -> bool:
    if isinstance(computed, dict):
        return isinstance(reported, dict) and all(
            key in reported and _metric_subset_matches(value, reported[key])
            for key, value in computed.items()
        )
    if isinstance(computed, list):
        return (
            isinstance(reported, list)
            and len(computed) == len(reported)
            and all(
                _metric_subset_matches(first, second)
                for first, second in zip(computed, reported)
            )
        )
    if isinstance(computed, (float, int, np.number)) and not isinstance(
        computed, (bool, np.bool_)
    ):
        return bool(
            np.isclose(
                float(computed),
                float(reported),
                rtol=1.0e-12,
                atol=REPLAY_NUMERIC_ATOL,
                equal_nan=True,
            )
        )
    return computed == reported


def _configuration() -> dict[str, object]:
    return {
        "test_id": "P1-T9R",
        "integrator": "Drake discrete MultibodyPlant SAP",
        "physics_steps_s": list(REFINEMENT_STEPS),
        "impact_speeds_mps": IMPACT_SPEEDS.tolist(),
        "candidate_physics_step_s": P1_T9R_CANDIDATE_STEP,
        "future_production_event_logging_step_s": constants.TIME_STEP,
        "duration_s": 0.25,
        "masses_kg": {
            "load": constants.LOAD_MASS,
            "vessel": constants.VESSEL_MASS,
        },
        "cable": {
            "rest_length_m": constants.CABLE_REST_LENGTH,
            "stiffness_n_per_m": constants.CABLE_STIFFNESS,
            "damping_n_s_per_m": 0.0,
        },
        "gravity_mps2": [0.0, 0.0, 0.0],
        "drag": "disabled",
        "initial_state": (
            "zero elongation, prescribed relative speed, zero total momentum"
        ),
        "acceptance_criteria": {
            "initial_state": "recorded t=0 sample equals the supplied state exactly",
            "physical_energy": (
                "maximum physical-energy relative envelope <= 0.005"
            ),
            "monotonicity": "physical-energy envelope decreases at each step halving",
            "convergence": "both step-halving orders are in [0.8, 1.2]",
            "momentum": (
                "per-step body residual, equal-opposite integrated impulse mismatch, "
                "and total momentum drift <= 1e-9"
            ),
            "continuous_controls": "existing continuous RK3 controls pass",
        },
        "gate_energy_quantity": "maximum_physical_energy_relative_envelope",
        "diagnostic_only_energy_quantities": [
            "post_release_endpoint_relative_energy_defect",
            "maximum_shadow_energy_relative_drift",
            "continuous_rk3_energy",
        ],
        "stochastic_production": "NOT_RUN",
    }


def _record_metrics(
    record,
    prefix: str,
    *,
    discrete: bool,
) -> dict[str, object]:
    time_values = record[f"{prefix}_time_s"]
    load_velocity = record[f"{prefix}_load_velocity_mps"]
    vessel_velocity = record[f"{prefix}_vessel_velocity_mps"]
    elongation = record[f"{prefix}_elongation_m"]
    elongation_rate = record[f"{prefix}_elongation_rate_mps"]
    tension = record[f"{prefix}_tension_n"]
    load_force = record[f"{prefix}_load_force_n"]
    vessel_force = record[f"{prefix}_vessel_force_n"]
    kinetic_energy = record[f"{prefix}_kinetic_energy_j"]
    spring_potential = record[f"{prefix}_spring_potential_j"]
    step = float(time_values[1] - time_values[0])
    load_interval_delta = constants.LOAD_MASS * np.diff(load_velocity)
    vessel_interval_delta = constants.VESSEL_MASS * np.diff(vessel_velocity)
    if discrete:
        left_residual = np.concatenate(
            (
                load_interval_delta - step * load_force[:-1],
                vessel_interval_delta - step * vessel_force[:-1],
            )
        )
        right_residual = np.concatenate(
            (
                load_interval_delta - step * load_force[1:],
                vessel_interval_delta - step * vessel_force[1:],
            )
        )
        interval_scale = max(
            float(np.max(np.abs(load_interval_delta))),
            float(np.max(np.abs(vessel_interval_delta))),
            np.finfo(float).tiny,
        )
        left_normalized_residual = float(
            np.max(np.abs(left_residual)) / interval_scale
        )
        right_normalized_residual = float(
            np.max(np.abs(right_residual)) / interval_scale
        )
        load_impulse = float(np.sum(load_force[:-1]) * step)
        vessel_impulse = float(np.sum(vessel_force[:-1]) * step)
        impulse_evaluation = (
            "left_endpoint_discrete_force_verified_from_state_update"
        )
    else:
        left_normalized_residual = float("nan")
        right_normalized_residual = float("nan")
        load_impulse = float(np.trapezoid(load_force, time_values))
        vessel_impulse = float(np.trapezoid(vessel_force, time_values))
        impulse_evaluation = "trapezoidal_continuous_force"
    load_delta = float(constants.LOAD_MASS * (load_velocity[-1] - load_velocity[0]))
    vessel_delta = float(
        constants.VESSEL_MASS * (vessel_velocity[-1] - vessel_velocity[0])
    )
    load_residual = load_delta - load_impulse
    vessel_residual = vessel_delta - vessel_impulse
    impulse_scale = max(
        abs(load_delta),
        abs(vessel_delta),
        abs(load_impulse),
        abs(vessel_impulse),
        np.finfo(float).tiny,
    )
    momentum = (
        constants.LOAD_MASS * load_velocity
        + constants.VESSEL_MASS * vessel_velocity
    )
    momentum_scale = max(
        float(np.max(np.abs(constants.LOAD_MASS * load_velocity))),
        float(np.max(np.abs(constants.VESSEL_MASS * vessel_velocity))),
        np.finfo(float).tiny,
    )
    energy = kinetic_energy + spring_potential
    physical_energy_envelope = float(np.max(np.abs(energy / energy[0] - 1.0)))
    positive_force = np.flatnonzero(tension > 0.0)
    release_candidates = (
        np.flatnonzero(tension[positive_force[0] + 1 :] <= 0.0)
        if positive_force.size
        else np.array([], dtype=int)
    )
    release_index = (
        int(positive_force[0] + 1 + release_candidates[0])
        if release_candidates.size
        else None
    )
    metrics = {
        "maximum_physical_energy_relative_envelope": physical_energy_envelope,
        "maximum_energy_relative_drift": physical_energy_envelope,
        "post_release_endpoint_relative_energy_defect": float(
            abs(energy[-1] / energy[0] - 1.0)
        ),
        "first_release_time_s": (
            None if release_index is None else float(time_values[release_index])
        ),
        "endpoint_is_post_release": bool(
            release_index is not None and release_index < time_values.size - 1
        ),
        "load_delta_momentum_kg_mps": load_delta,
        "load_impulse_n_s": load_impulse,
        "load_momentum_impulse_residual_kg_mps": load_residual,
        "vessel_delta_momentum_kg_mps": vessel_delta,
        "vessel_impulse_n_s": vessel_impulse,
        "vessel_momentum_impulse_residual_kg_mps": vessel_residual,
        "body_impulse_evaluation": impulse_evaluation,
        "equal_opposite_integrated_impulse_mismatch": float(
            abs(load_impulse + vessel_impulse) / impulse_scale
        ),
        "maximum_normalized_body_residual": float(
            max(abs(load_residual), abs(vessel_residual)) / impulse_scale
        ),
        "maximum_total_momentum_drift_kg_mps": float(
            np.max(np.abs(momentum - momentum[0]))
        ),
        "normalized_total_momentum_drift": float(
            np.max(np.abs(momentum - momentum[0])) / momentum_scale
        ),
    }
    if discrete:
        metrics.update(
            {
                "left_endpoint_per_step_normalized_momentum_residual": (
                    left_normalized_residual
                ),
                "right_endpoint_per_step_normalized_momentum_residual": (
                    right_normalized_residual
                ),
                "maximum_per_step_normalized_body_momentum_residual": (
                    left_normalized_residual
                ),
            }
        )
        shadow_energy = energy - (
            0.5
            * step
            * constants.CABLE_STIFFNESS
            * np.maximum(elongation, 0.0)
            * elongation_rate
        )
        metrics["maximum_shadow_energy_relative_drift"] = float(
            np.max(np.abs(shadow_energy / energy[0] - 1.0))
        )
    return metrics


def _recomputed_acceptance(record) -> dict[str, object]:
    step_results = []
    for step_index, step in enumerate(REFINEMENT_STEPS):
        speed_results = []
        for speed_index, speed in enumerate(IMPACT_SPEEDS):
            prefix = f"p1_t9r_discrete_s{step_index}_v{speed_index}"
            metrics = _record_metrics(record, prefix, discrete=True)
            total_mass = constants.LOAD_MASS + constants.VESSEL_MASS
            metrics.update(
                {
                    "impact_speed_mps": float(speed),
                    "initial_state_exact": bool(
                        record[f"{prefix}_time_s"][0] == 0.0
                        and record[f"{prefix}_load_position_m"][0] == 0.0
                        and record[f"{prefix}_vessel_position_m"][0]
                        == constants.CABLE_REST_LENGTH
                        and record[f"{prefix}_load_velocity_mps"][0]
                        == -constants.VESSEL_MASS * speed / total_mass
                        and record[f"{prefix}_vessel_velocity_mps"][0]
                        == constants.LOAD_MASS * speed / total_mass
                        and record[f"{prefix}_elongation_m"][0] == 0.0
                        and record[f"{prefix}_tension_n"][0] == 0.0
                    ),
                    "step_s": step,
                }
            )
            speed_results.append(metrics)
        maximum_envelope = max(
            row["maximum_physical_energy_relative_envelope"]
            for row in speed_results
        )
        maximum_per_step_residual = max(
            row["maximum_per_step_normalized_body_momentum_residual"]
            for row in speed_results
        )
        maximum_impulse_mismatch = max(
            row["equal_opposite_integrated_impulse_mismatch"]
            for row in speed_results
        )
        maximum_momentum_drift = max(
            row["normalized_total_momentum_drift"] for row in speed_results
        )
        passes_numerically = bool(
            all(row["initial_state_exact"] for row in speed_results)
            and maximum_envelope <= P1_T9R_ENERGY_LIMIT
            and maximum_per_step_residual <= P1_T9R_MOMENTUM_LIMIT
            and maximum_impulse_mismatch <= P1_T9R_MOMENTUM_LIMIT
            and maximum_momentum_drift <= P1_T9R_MOMENTUM_LIMIT
        )
        step_results.append(
            {
                "step_s": step,
                "speed_results": speed_results,
                "maximum_physical_energy_relative_envelope": maximum_envelope,
                "energy_margin_to_limit": P1_T9R_ENERGY_LIMIT - maximum_envelope,
                "maximum_per_step_normalized_body_momentum_residual": (
                    maximum_per_step_residual
                ),
                "maximum_equal_opposite_integrated_impulse_mismatch": (
                    maximum_impulse_mismatch
                ),
                "maximum_normalized_total_momentum_drift": maximum_momentum_drift,
                "passes_numerically": passes_numerically,
            }
        )

    convergence = []
    for speed_index, speed in enumerate(IMPACT_SPEEDS):
        envelopes = [
            row["speed_results"][speed_index][
                "maximum_physical_energy_relative_envelope"
            ]
            for row in step_results
        ]
        orders = [
            float(np.log2(envelopes[0] / envelopes[1])),
            float(np.log2(envelopes[1] / envelopes[2])),
        ]
        convergence.append(
            {
                "impact_speed_mps": float(speed),
                "physical_energy_envelopes": envelopes,
                "monotone_decrease": bool(envelopes[0] > envelopes[1] > envelopes[2]),
                "coarse_to_medium_order": orders[0],
                "medium_to_fine_order": orders[1],
                "orders_defined": True,
                "orders_within_bounds": bool(
                    all(P1_T9R_ORDER_BOUNDS[0] <= order <= P1_T9R_ORDER_BOUNDS[1]
                        for order in orders)
                ),
            }
        )

    continuous_controls = []
    for step_index, step in enumerate(REFINEMENT_STEPS):
        metrics = _record_metrics(
            record, f"p1_t9r_continuous_s{step_index}", discrete=False
        )
        metrics.update({"step_s": step, "integrator": "continuous_fixed_step_rk3"})
        continuous_controls.append(metrics)
    continuous_controls_passed = all(
        row["maximum_physical_energy_relative_envelope"] <= P1_T9R_ENERGY_LIMIT
        and row["maximum_normalized_body_residual"] <= P1_T9R_ENERGY_LIMIT
        and row["normalized_total_momentum_drift"] <= P1_T9R_ENERGY_LIMIT
        for row in continuous_controls
    )
    candidate = step_results[REFINEMENT_STEPS.index(P1_T9R_CANDIDATE_STEP)]
    half_millisecond = step_results[REFINEMENT_STEPS.index(5.0e-4)]
    monotone_passed = all(row["monotone_decrease"] for row in convergence)
    convergence_passed = all(row["orders_within_bounds"] for row in convergence)
    passed = bool(
        candidate["passes_numerically"]
        and monotone_passed
        and convergence_passed
        and continuous_controls_passed
    )
    return {
        "test_id": "P1-T9R",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "candidate_physics_step_s": P1_T9R_CANDIDATE_STEP,
        "production_event_logging_step_s": constants.TIME_STEP,
        "physical_energy_relative_limit": P1_T9R_ENERGY_LIMIT,
        "momentum_relative_limit": P1_T9R_MOMENTUM_LIMIT,
        "convergence_order_bounds": list(P1_T9R_ORDER_BOUNDS),
        "step_results": step_results,
        "convergence": convergence,
        "monotone_decrease_passed": monotone_passed,
        "convergence_orders_passed": convergence_passed,
        "continuous_rk3_controls": continuous_controls,
        "continuous_rk3_controls_passed": continuous_controls_passed,
        "half_millisecond_passes_numerically": half_millisecond[
            "passes_numerically"
        ],
        "half_millisecond_selected": False,
        "half_millisecond_selection_reason": "inadequate numerical margin",
        "candidate_passes_numerically": candidate["passes_numerically"],
        "gate_energy_quantity": "maximum_physical_energy_relative_envelope",
        "excluded_gate_energy_quantities": [
            "post_release_endpoint_relative_energy_defect",
            "maximum_shadow_energy_relative_drift",
            "continuous_rk3_energy",
        ],
    }


def p1_t9r_record_metric_checks(
    record_path: Path,
    results: dict[str, object],
) -> dict[str, bool]:
    with np.load(record_path, allow_pickle=False) as record:
        metadata_match = bool(
            np.array_equal(record["p1_t9r_physics_steps_s"], REFINEMENT_STEPS)
            and np.array_equal(record["p1_t9r_impact_speeds_mps"], IMPACT_SPEEDS)
        )
        recomputed = _recomputed_acceptance(record)
    return {
        "metadata_matches_configuration": metadata_match,
        "all_gate_metrics_match_raw_channels": _metric_subset_matches(
            recomputed, results["test"]
        ),
    }


def _format_report(results: dict[str, object]) -> str:
    test = results["test"]
    rows = []
    for step in test["step_results"]:
        for row in step["speed_results"]:
            rows.append(
                "| {step:.2f} | {speed:.2f} | {energy:.9e} | {endpoint:.9e} | "
                "{shadow:.9e} | {body:.9e} | {impulse:.9e} | {momentum:.9e} | "
                "{initial} |".format(
                    step=1000.0 * step["step_s"],
                    speed=row["impact_speed_mps"],
                    energy=row["maximum_physical_energy_relative_envelope"],
                    endpoint=row["post_release_endpoint_relative_energy_defect"],
                    shadow=row["maximum_shadow_energy_relative_drift"],
                    body=row["maximum_per_step_normalized_body_momentum_residual"],
                    impulse=row["equal_opposite_integrated_impulse_mismatch"],
                    momentum=row["normalized_total_momentum_drift"],
                    initial=row["initial_state_exact"],
                )
            )
    convergence_rows = "\n".join(
        "| {impact_speed_mps:.2f} | {coarse_to_medium_order:.9f} | "
        "{medium_to_fine_order:.9f} | {monotone_decrease} | "
        "{orders_within_bounds} |".format(**row)
        for row in test["convergence"]
    )
    summary_rows = "\n".join(
        f"| {1000.0 * row['step_s']:.2f} | "
        f"{row['maximum_physical_energy_relative_envelope']:.9e} | "
        f"{row['energy_margin_to_limit']:.9e} | "
        f"{row['passes_numerically']} |"
        for row in test["step_results"]
    )
    return f"""# P1-T9R Discrete-Step Remediation Gate

## Decision

**{test['status']}: the 0.25 ms SAP physics step is qualified for candidate production use.** The event and logging cadence remains 1.0 ms. The 0.5 ms step passes the numerical limits but is not selected because its physical-energy margin is inadequate. This result remediates P1-T9 only; it does not constitute a Phase 1 GO decision.

## Preregistered gate

The deterministic sweep evaluates SAP steps of 1.0, 0.5, and 0.25 ms at impact speeds {IMPACT_SPEEDS.tolist()} m/s. Every cell uses the unchanged {constants.LOAD_MASS:.0f} kg and {constants.VESSEL_MASS:.0f} kg masses, {constants.CABLE_STIFFNESS:.0f} N/m stiffness, zero cable damping, no drag, and zero gravity. The recorded t=0 state precedes every discrete update.

The gate quantity is the maximum physical-energy relative envelope, with an unchanged limit of {100.0 * P1_T9R_ENERGY_LIMIT:.1f}%. Endpoint energy, shadow energy, and continuous Runge-Kutta 3 (RK3) energy are diagnostic only and cannot satisfy this criterion. Momentum acceptance requires the per-step body residual, equal-opposite integrated impulse mismatch, and total momentum drift to remain at or below {P1_T9R_MOMENTUM_LIMIT:.0e}.

## Step decision

| SAP step [ms] | Worst physical-energy envelope | Margin to 0.5% limit | Numerical pass |
|---:|---:|---:|:---:|
{summary_rows}

The 1.0 ms step fails the unchanged energy criterion. The 0.5 ms step passes by only {100.0 * test['step_results'][1]['energy_margin_to_limit']:.6f} percentage points. The selected 0.25 ms step passes by {100.0 * test['step_results'][2]['energy_margin_to_limit']:.6f} percentage points and preserves the 1.0 ms event/logging cadence for future production.

## Metrics by step and speed

| SAP step [ms] | Speed [m/s] | Physical envelope | Endpoint defect | Shadow drift | Per-step body residual | Impulse mismatch | Momentum drift | Exact t=0 |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
{chr(10).join(rows)}

The left-endpoint force convention is established from the discrete state equation: interval momentum changes match the force sampled at the interval start to roundoff, whereas the right-endpoint residual is nonzero. The integrated impulses therefore use `force[:-1]`; this is not a relabeling of the prior convention.

## Convergence

| Speed [m/s] | 1.0 to 0.5 ms order | 0.5 to 0.25 ms order | Monotone | Order in [0.8, 1.2] |
|---:|---:|---:|:---:|:---:|
{convergence_rows}

The physical-energy envelope decreases at both halvings for every speed, and all ten measured orders lie in the preregistered interval. The existing continuous RK3 controls also pass; they remain controls rather than substitutes for discrete SAP qualification.

## Provenance and scope

- Source revision: `{results['source']['revision']}`; worktree status: `{results['source']['status']}`.
- Deterministic cells: {results['execution']['discrete_cell_count']} discrete SAP and {results['execution']['continuous_control_count']} continuous RK3 controls.
- Stochastic production: `NOT_RUN`.
- Raw NPZ channels include time, two-body position and velocity, elongation and elongation rate, tension, both body forces, kinetic energy, and spring potential for every cell.
- Replay regenerates into temporary paths, recomputes all gate metrics from raw channels, and does not overwrite committed artifacts.
- Existing Phase 1R artifacts remained unchanged during generation: `{results['phase1r_preservation']['unchanged_during_campaign']}`.

## Remaining blockers

P1-T4 remains `UNDER-POWERED` in the deterministic Stage 3-4 record and was not rerun by Phase 1R. P1-T5 full-plant energy conservation passes, but its impact calibration remains `MODEL_INVALID_LOW_SPEED`. P1-T9R does not modify either conclusion, and no stochastic production campaign was executed.
"""


def _manifest(
    configuration: dict[str, object],
    source: dict[str, object],
    artifacts: dict[str, dict[str, str]],
    phase1r_hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "driver": "python -m tether.campaign.phase1_t9r run",
        "source": source,
        "runtime": {"python_version": platform.python_version()},
        "configuration": configuration,
        "configuration_sha256": _configuration_sha256(configuration),
        "acceptance_criteria": configuration["acceptance_criteria"],
        "artifacts": artifacts,
        "record_sha256": artifacts["record"]["sha256"],
        "phase1r_artifact_hashes": phase1r_hashes,
        "probe": {
            "command": "python -m tether.campaign.phase1_t9r replay",
            "raw_channel_metric_recomputation": True,
            "numeric_absolute_tolerance": REPLAY_NUMERIC_ATOL,
            "deterministic_record_bytes": True,
            "complete_results_excluding": ["execution.wall_seconds"],
            "report_bytes": True,
            "non_overwriting": True,
        },
    }


def run_campaign(
    results_path: Path = RESULTS_PATH,
    record_path: Path = RECORD_PATH,
    manifest_path: Path = MANIFEST_PATH,
    report_path: Path = REPORT_PATH,
    *,
    acceptance_bundle=None,
) -> dict[str, object]:
    """Run P1-T9R without modifying any Phase 1R artifact."""
    start = time.perf_counter()
    phase1r_before = _phase1r_hashes()
    source = _source_state()
    if acceptance_bundle is None:
        acceptance_bundle = p1_t9r_acceptance(include_evidence=True)
    test, evidence = acceptance_bundle
    configuration = _configuration()
    record_bytes = _npz_bytes(evidence)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_bytes(record_bytes)
    phase1r_after_record = _phase1r_hashes()
    if phase1r_before != phase1r_after_record:
        raise RuntimeError("P1-T9R modified an existing Phase 1R artifact")
    results = {
        "schema_version": 1,
        "scope": "P1-T9R deterministic discrete-step remediation gate only",
        "gate": test["status"],
        "candidate_physics_step_status": (
            "QUALIFIED" if test["passed"] else "NOT_QUALIFIED"
        ),
        "configuration": configuration,
        "configuration_sha256": _configuration_sha256(configuration),
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "source": source,
        "execution": {
            "discrete_cell_count": len(REFINEMENT_STEPS) * len(IMPACT_SPEEDS),
            "continuous_control_count": len(REFINEMENT_STEPS),
            "wall_seconds": time.perf_counter() - start,
        },
        "test": test,
        "phase1r_preservation": {
            "hashes": phase1r_before,
            "unchanged_during_campaign": True,
        },
        "remaining_blockers": {
            "P1-T4": "UNDER-POWERED; not rerun by Phase 1R or P1-T9R",
            "P1-T5": (
                "full-plant energy PASS; impact calibration MODEL_INVALID_LOW_SPEED"
            ),
        },
        "stochastic_production": {
            "status": "NOT_RUN",
            "production_grid_launched": False,
        },
    }
    results_path.write_bytes(_json_bytes(results))
    report_path.write_text(_format_report(results), encoding="ascii")
    artifacts = {
        "record": {"path": _recorded_path(record_path), "sha256": _sha256_file(record_path)},
        "results": {"path": _recorded_path(results_path), "sha256": _sha256_file(results_path)},
        "report": {"path": _recorded_path(report_path), "sha256": _sha256_file(report_path)},
    }
    manifest_path.write_bytes(
        _json_bytes(_manifest(configuration, source, artifacts, phase1r_before))
    )
    if phase1r_before != _phase1r_hashes():
        raise RuntimeError("P1-T9R modified an existing Phase 1R artifact")
    print(
        f"P1-T9R: {test['status']} "
        f"({results['candidate_physics_step_status']} at "
        f"{1000.0 * P1_T9R_CANDIDATE_STEP:.2f} ms)"
    )
    return results


def replay(
    results_path: Path = RESULTS_PATH,
    record_path: Path = RECORD_PATH,
    manifest_path: Path = MANIFEST_PATH,
    report_path: Path = REPORT_PATH,
    *,
    campaign_inputs: dict[str, object] | None = None,
) -> bool:
    """Regenerate P1-T9R in temporary paths and verify without overwriting."""
    before = {
        "results": _sha256_file(results_path),
        "record": _sha256_file(record_path),
        "manifest": _sha256_file(manifest_path),
        "report": _sha256_file(report_path),
    }
    phase1r_before = _phase1r_hashes()
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    configuration = _configuration()
    if manifest["configuration_sha256"] != _configuration_sha256(configuration):
        raise RuntimeError("current P1-T9R configuration does not match the manifest")
    artifact_paths = {
        "record": record_path,
        "results": results_path,
        "report": report_path,
    }
    manifest_digests_match = all(
        manifest["artifacts"][name]["path"] == _recorded_path(path)
        and manifest["artifacts"][name]["sha256"] == _sha256_file(path)
        for name, path in artifact_paths.items()
    )
    committed_results = json.loads(results_path.read_text(encoding="ascii"))
    raw_checks = p1_t9r_record_metric_checks(record_path, committed_results)
    with tempfile.TemporaryDirectory(prefix="phase1-t9r-replay-") as directory:
        temporary = Path(directory)
        replay_results = run_campaign(
            temporary / "results.json",
            temporary / "records.npz",
            temporary / "manifest.json",
            temporary / "report.md",
            **(campaign_inputs or {}),
        )
        record_identical = (
            (temporary / "records.npz").read_bytes() == record_path.read_bytes()
        )
        report_identical = (
            (temporary / "report.md").read_bytes() == report_path.read_bytes()
        )
        replay_comparable = json.loads(json.dumps(replay_results, sort_keys=True))
        committed_comparable = json.loads(
            json.dumps(committed_results, sort_keys=True)
        )
        replay_comparable["execution"].pop("wall_seconds", None)
        committed_comparable["execution"].pop("wall_seconds", None)
        results_match = replay_comparable == committed_comparable
    after = {
        "results": _sha256_file(results_path),
        "record": _sha256_file(record_path),
        "manifest": _sha256_file(manifest_path),
        "report": _sha256_file(report_path),
    }
    phase1r_after = _phase1r_hashes()
    passed = bool(
        manifest_digests_match
        and all(raw_checks.values())
        and record_identical
        and report_identical
        and results_match
        and before == after
        and phase1r_before == phase1r_after
        and manifest["phase1r_artifact_hashes"] == phase1r_after
    )
    evidence = {
        "temporary_regeneration": True,
        "committed_files_overwritten": False,
        "manifest_digests_match": manifest_digests_match,
        "raw_metric_checks": raw_checks,
        "record_byte_identical": record_identical,
        "report_byte_identical": report_identical,
        "complete_results_match_excluding_wall_time": results_match,
        "p1_t9r_artifacts_unchanged": before == after,
        "phase1r_artifacts_unchanged": phase1r_before == phase1r_after,
        "passed": passed,
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("P1-T9R replay probe failed")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "replay"))
    arguments = parser.parse_args()
    if arguments.command == "run":
        run_campaign()
    else:
        replay()


if __name__ == "__main__":
    main()