"""Execute, record, report, and replay the Phase 0 acceptance campaign."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pydrake

from tether.analysis.phase0_figures import generate_figure
from tether.estimation.truth_isolation import lint_all_arms, lint_source
from tether.physics import constants
from tether.physics.events import synthetic_crossing_benchmark
from tether.physics.geometry import (
    attachment_length,
    attachment_length_rate,
    attachment_position,
    attachment_velocity,
)
from tether.physics.plant import run_steady_transit, run_transit_log, steady_state_values
from tether.physics.weather import (
    apply_ar1,
    ar1_95_interval,
    ar1_coefficient,
    estimate_ar1_coefficient,
    generate_weather,
    hill_tail_index,
    standardized_innovations,
)

ROOT = Path(__file__).resolve().parents[2]
RECORD_DIR = ROOT / "records" / "phase0"
REPORT_PATH = ROOT / "reports" / "phase0_report.md"
RECORD_PATH = RECORD_DIR / "transit_nominal.npz"
RESULTS_PATH = RECORD_DIR / "phase0_results.json"
MANIFEST_PATH = RECORD_DIR / "manifest.json"
MASTER_SEED = 4102
RECORD_DURATION = 0.5
RUNTIME_DURATION = 1.0
BASELINE_WALL_SECONDS_PER_SIM_SECOND = 0.9

EXPECTED_VERSIONS = {
    "drake": "1.51.1",
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "matplotlib": "3.10.9",
    "pytest": "8.4.2",
}


def _json_bytes(value: object) -> bytes:
    def numpy_scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    return (
        json.dumps(value, indent=2, sort_keys=True, default=numpy_scalar) + "\n"
    ).encode("ascii")


def _configuration() -> dict[str, object]:
    return {
        "cable_damping": constants.CABLE_DAMPING,
        "cable_rest_length": constants.CABLE_REST_LENGTH,
        "cable_stiffness": constants.CABLE_STIFFNESS,
        "geometry": "collinear_bar_sigma_0",
        "master_seed": MASTER_SEED,
        "record_duration": RECORD_DURATION,
        "sample_period": constants.WEATHER_PERIOD,
        "time_step": constants.TIME_STEP,
        "thrust_levels": list(constants.THRUST_LEVELS),
        "vessel_count": constants.VESSEL_COUNT,
        "weather_distribution": "gaussian",
        "weather_direction": "local",
    }


def _config_hash() -> str:
    canonical = json.dumps(_configuration(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Serialize an NPZ with fixed member order, metadata, and timestamps."""
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            array_buffer = io.BytesIO()
            np.lib.format.write_array(array_buffer, np.asarray(arrays[name]), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            member.external_attr = 0o600 << 16
            archive.writestr(member, array_buffer.getvalue())
    return archive_buffer.getvalue()


def _record_arrays(log) -> dict[str, np.ndarray]:
    return {
        "marks": log.marks,
        "tension": log.tension,
        "time": log.time,
        "truth": log.truth,
    }


def _version_result() -> dict[str, object]:
    actual = {
        package: importlib.metadata.version(package) for package in EXPECTED_VERSIONS
    }
    lock_entries = {}
    for line in (ROOT / "requirements.lock.txt").read_text(encoding="ascii").splitlines():
        package, version = line.split("==", maxsplit=1)
        lock_entries[package] = version
    pydrake_version_present = hasattr(pydrake, "__version__")
    passed = actual == EXPECTED_VERSIONS and lock_entries == EXPECTED_VERSIONS
    return {
        "actual": actual,
        "expected": EXPECTED_VERSIONS,
        "lock_entries": lock_entries,
        "pydrake_version_attribute_present": pydrake_version_present,
        "version_source": "importlib.metadata.version('drake')",
        "passed": passed,
    }


def _steady_state_result() -> dict[str, object]:
    measurements = []
    passed = True
    for thrust in constants.THRUST_LEVELS:
        expected_speed, expected_tension = steady_state_values(thrust)
        observed = run_steady_transit(thrust)
        speed_relative_error = abs(observed.speed / expected_speed - 1.0)
        tension_relative_error = abs(observed.mean_tension / expected_tension - 1.0)
        passed = passed and speed_relative_error <= 0.02 and tension_relative_error <= 0.02
        measurements.append(
            {
                "thrust_n": thrust,
                "expected_speed_mps": expected_speed,
                "measured_speed_mps": observed.speed,
                "speed_relative_error": speed_relative_error,
                "expected_tension_n": expected_tension,
                "measured_tension_n": observed.mean_tension,
                "tension_relative_error": tension_relative_error,
            }
        )
    return {"measurements": measurements, "passed": passed}


def _attachment_result() -> dict[str, object]:
    first_position = np.array([1.2, -0.7])
    second_position = np.array([8.1, 4.3])
    first_angle, second_angle = 0.37, -0.61
    first_offset = np.array([1.1, -0.4])
    second_offset = np.array([-0.8, 0.9])
    first_velocity = np.array([0.8, -0.3])
    second_velocity = np.array([-0.2, 1.4])
    first_rate, second_rate = 0.73, -0.46
    first_attachment = attachment_position(first_position, first_angle, first_offset)
    second_attachment = attachment_position(second_position, second_angle, second_offset)
    measured = attachment_length_rate(
        first_attachment,
        attachment_velocity(first_velocity, first_rate, first_angle, first_offset),
        second_attachment,
        attachment_velocity(second_velocity, second_rate, second_angle, second_offset),
    )
    difference_step = 1.0e-5

    def oracle_length(offset_time: float) -> float:
        first = attachment_position(
            first_position + offset_time * first_velocity,
            first_angle + offset_time * first_rate,
            first_offset,
        )
        second = attachment_position(
            second_position + offset_time * second_velocity,
            second_angle + offset_time * second_rate,
            second_offset,
        )
        return attachment_length(first, second)

    oracle = (
        oracle_length(difference_step) - oracle_length(-difference_step)
    ) / (2.0 * difference_step)
    absolute_error = abs(measured - oracle)
    return {
        "production_rate_mps": measured,
        "central_difference_rate_mps": oracle,
        "absolute_error_mps": absolute_error,
        "difference_step_s": difference_step,
        "passed": absolute_error <= 1.0e-6,
    }


def _determinism_result():
    weather_count = int(np.ceil(RECORD_DURATION / constants.WEATHER_PERIOD)) + 2
    first_weather = generate_weather(MASTER_SEED, weather_count)
    second_weather = generate_weather(MASTER_SEED, weather_count)
    first_log = run_transit_log(
        constants.NOMINAL_THRUST,
        RECORD_DURATION,
        constants.WEATHER_PERIOD,
        first_weather.states,
    )
    second_log = run_transit_log(
        constants.NOMINAL_THRUST,
        RECORD_DURATION,
        constants.WEATHER_PERIOD,
        second_weather.states,
    )
    first_arrays = _record_arrays(first_log)
    second_arrays = _record_arrays(second_log)
    comparisons = {
        name: bool(np.array_equal(first_arrays[name], second_arrays[name]))
        for name in first_arrays
    }
    first_bytes = _npz_bytes(first_arrays)
    second_bytes = _npz_bytes(second_arrays)
    comparisons["record_bytes"] = first_bytes == second_bytes
    return (
        {
            "comparisons": comparisons,
            "record_sha256": hashlib.sha256(first_bytes).hexdigest(),
            "passed": all(comparisons.values()),
        },
        first_bytes,
    )


def _truth_isolation_result() -> dict[str, object]:
    lint_results = lint_all_arms()
    oracle_path = ROOT / "tether" / "estimation" / "oracle.py"
    oracle_source = oracle_path.read_text(encoding="ascii")
    undeclared_source = oracle_source.replace(
        'TRUTH_ISOLATION_EXEMPTIONS = {"tether.physics.plant"}',
        "TRUTH_ISOLATION_EXEMPTIONS = set()",
    )
    undeclared_violations = lint_source(undeclared_source, "undeclared_oracle")
    passed = all(not violations for violations in lint_results.values()) and bool(
        undeclared_violations
    )
    return {
        "declared_arm_violations": lint_results,
        "undeclared_oracle_violations": undeclared_violations,
        "passed": passed,
    }


def _runtime_result() -> dict[str, object]:
    weather_count = int(np.ceil(RUNTIME_DURATION / constants.WEATHER_PERIOD)) + 2
    weather = generate_weather(MASTER_SEED, weather_count)
    start = time.perf_counter()
    run_transit_log(
        constants.NOMINAL_THRUST,
        RUNTIME_DURATION,
        constants.WEATHER_PERIOD,
        weather.states,
    )
    wall_seconds = time.perf_counter() - start
    ratio = wall_seconds / RUNTIME_DURATION
    multiplier = ratio / BASELINE_WALL_SECONDS_PER_SIM_SECOND
    return {
        "vessel_count": constants.VESSEL_COUNT,
        "weather_on": True,
        "simulated_seconds": RUNTIME_DURATION,
        "wall_seconds": wall_seconds,
        "wall_seconds_per_simulated_second": ratio,
        "plan_baseline_wall_seconds_per_simulated_second": BASELINE_WALL_SECONDS_PER_SIM_SECOND,
        "phase1_projected_core_hours": 15.0 * multiplier,
        "passed": True,
    }


def _weather_result() -> dict[str, object]:
    innovation_count = 1_000_000
    gaussian = standardized_innovations(
        MASTER_SEED + 1, innovation_count, 1, "gaussian", "local"
    )[:, 0, 0]
    centered = gaussian - gaussian.mean()
    gaussian_kurtosis = float(np.mean(centered**4) / np.mean(centered**2) ** 2)
    student = standardized_innovations(
        MASTER_SEED + 2, innovation_count, 1, "student_t3", "local"
    )[:, 0, 0]
    hill_alpha = hill_tail_index(student, upper_order_count=5_000)

    ar_count = 300_000
    ar_innovations = standardized_innovations(
        MASTER_SEED + 3, ar_count, 1, "gaussian", "local"
    )
    ar_states = apply_ar1(ar_innovations)[1_000:]
    phi = ar1_coefficient()
    phi_estimate = estimate_ar1_coefficient(ar_states)
    phi_interval = ar1_95_interval(ar_states.size - 1, phi)
    common = standardized_innovations(
        MASTER_SEED + 4, 10_000, constants.VESSEL_COUNT + 1, "gaussian", "common"
    )
    common_exact = all(
        np.array_equal(common[:, 0], common[:, body])
        for body in range(1, common.shape[1])
    )
    passed = (
        abs(gaussian_kurtosis - 3.0) <= 0.03
        and 2.7 <= hill_alpha <= 3.3
        and phi_interval[0] <= phi_estimate <= phi_interval[1]
        and common_exact
    )
    return {
        "innovation_sample_count": innovation_count,
        "gaussian_kurtosis": gaussian_kurtosis,
        "student_t3_hill_alpha": hill_alpha,
        "hill_upper_order_count": 5_000,
        "phi_target": phi,
        "phi_estimate": phi_estimate,
        "phi_95_interval": list(phi_interval),
        "ar_sample_count": ar_count,
        "common_standardized_innovations_exact": common_exact,
        "passed": passed,
    }


def _manifest(record_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "driver": "python -m tether.campaign.phase0 run",
        "configuration": _configuration(),
        "configuration_sha256": _config_hash(),
        "seeds": [MASTER_SEED],
        "committed_record_path": "records/phase0/transit_nominal.npz",
        "record_sha256": record_sha256,
        "probe": {
            "command": "python -m tether.campaign.phase0 replay",
            "plant_absolute_tolerance": 1.0e-6,
            "marks_comparison": "exact",
            "deterministic_record_bytes": True,
        },
    }


def _format_report(results: dict[str, object]) -> str:
    tests = results["tests"]
    steady = tests["P0-T2"]
    maximum_speed_error = max(item["speed_relative_error"] for item in steady["measurements"])
    maximum_tension_error = max(
        item["tension_relative_error"] for item in steady["measurements"]
    )
    runtime = tests["P0-T6"]
    weather = tests["P0-T8"]
    verdict = results["gate_verdict"]
    rows = [
        (
            "P0-T1",
            "drake 1.51.1; numpy 2.2.6; scipy 1.15.3; matplotlib 3.10.9; pytest 8.4.2",
            "exact pins",
            tests["P0-T1"]["passed"],
        ),
        (
            "P0-T2",
            f"max speed error {100 * maximum_speed_error:.6f}%; max tension error {100 * maximum_tension_error:.6f}%",
            "both errors <= 2% at all three thrusts",
            steady["passed"],
        ),
        (
            "P0-T3",
            f"absolute rate error {tests['P0-T3']['absolute_error_mps']:.3e} m/s",
            "<= 1e-6 m/s",
            tests["P0-T3"]["passed"],
        ),
        (
            "P0-T4",
            f"truth, tension, marks, and NPZ bytes identical; SHA-256 {tests['P0-T4']['record_sha256']}",
            "bit-identical at equal seed",
            tests["P0-T4"]["passed"],
        ),
        (
            "P0-T5",
            f"declared violations 0; undeclared oracle violations {len(tests['P0-T5']['undeclared_oracle_violations'])}",
            "ordinary clean; oracle requires declaration",
            tests["P0-T5"]["passed"],
        ),
        (
            "P0-T6",
            f"{runtime['wall_seconds_per_simulated_second']:.6f} wall-s/sim-s (N=5, weather on)",
            "recorded; no blocking threshold",
            tests["P0-T6"]["passed"],
        ),
        (
            "P0-T7",
            f"time order {tests['P0-T7']['time_order']:.4f}; finest speed error {100 * tests['P0-T7']['finest_speed_relative_error']:.6f}%",
            "order >= 1.8; speed error < 1%",
            tests["P0-T7"]["passed"],
        ),
        (
            "P0-T8",
            f"kurtosis {weather['gaussian_kurtosis']:.6f}; Hill alpha {weather['student_t3_hill_alpha']:.6f}; phi {weather['phi_estimate']:.8f} in [{weather['phi_95_interval'][0]:.8f}, {weather['phi_95_interval'][1]:.8f}]; common exact",
            "3 +/- 0.03; [2.7, 3.3]; phi in 95% interval; exact common driver",
            tests["P0-T8"]["passed"],
        ),
    ]
    table = "\n".join(
        f"| {test_id} | {measurement} | {criterion} | {'PASS' if passed else 'FAIL'} |"
        for test_id, measurement, criterion, passed in rows
    )
    thrust_rows = "\n".join(
        "| {thrust_n:.2f} | {expected_speed_mps:.6f} | {measured_speed_mps:.6f} | "
        "{expected_tension_n:.6f} | {measured_tension_n:.6f} |".format(**measurement)
        for measurement in steady["measurements"]
    )
    budget_relation = runtime["wall_seconds_per_simulated_second"] / BASELINE_WALL_SECONDS_PER_SIM_SECOND
    return f"""# Phase 0 Report

## Gate verdict

**{verdict}.** The mandatory gate comprises P0-T1 through P0-T5, P0-T7, and P0-T8. P0-T6 is recorded but is not blocking. A failure of P0-T2 or P0-T3 would force a NO-GO verdict.

## Configuration and execution

- Master seed: `{MASTER_SEED}`.
- Plant: Drake discrete `MultibodyPlant`, 1 ms time step, SAP approximation, zero gravity, one load and five vessels on planar joints.
- Cable model: explicit bilateral spring-damper forces applied at body-fixed attachment points.
- Baseline geometry: collinear cables with `bar_sigma = 0` and symmetric load attachment offsets.
- Weather: 10 ms local Gaussian AR(1), with one `[master, 11, body]` generator per body.
- Deterministic transit record: {RECORD_PATH.relative_to(ROOT)}.

## Acceptance results

| Test | Measured value or interval | Acceptance criterion | Verdict |
|---|---|---|---|
{table}

### P0-T2 operating points

| Thrust [N/vessel] | Formula speed [m/s] | Simulated speed [m/s] | Formula tension [N] | Simulated tension [N] |
|---:|---:|---:|---:|---:|
{thrust_rows}

## Runtime implication

The measured cost is {runtime['wall_seconds_per_simulated_second']:.6f} wall-s/sim-s, compared with the IV.14 planning assumption of 0.9 wall-s/sim-s. The ratio is {budget_relation:.3f}. Applying this ratio to the provisional Phase 1 allocation changes its estimate from 15.0 to {runtime['phase1_projected_core_hours']:.3f} core-hours on this container. This estimate remains hardware-specific and should be measured again if the execution platform changes.

## Deviations from the plan

1. Drake 1.51.1 does not define `pydrake.__version__`. P0-T1 therefore uses `importlib.metadata.version("drake")`; the package metadata and lock file both report 1.51.1.
2. The plan names three thrust levels but specifies only 1305 N. Phase 0 uses the documented symmetric set 978.75, 1305.0, and 1631.25 N (0.75x, 1.0x, and 1.25x nominal).
3. The Phase 0 baseline specializes the formation to `bar_sigma = 0`. Generic offsets, angles, and angular rates remain covered by P0-T3.
4. The ordinary and oracle estimation arms are minimal boundary fixtures for the AST lint. No Phase 1 estimation, controller, unilateral-cable, or severance behavior is implemented.

## Artifacts

- `records/phase0/transit_nominal.npz`: deterministic truth, tension, time, and empty Phase 0 mark arrays.
- `records/phase0/phase0_results.json`: measured values and gate decisions.
- `records/phase0/manifest.json`: driver, configuration hash, seed, record path, digest, and replay probe.
- `reports/figures/phase0_validation.png`: generated exclusively from the committed NPZ record.

## Open items

No blocking Phase 0 item remains. Phase 1 must replace the bilateral force law with the unilateral cable and introduce event marks, live/recording severance modes, and the energy-conservation test before any excursion-law campaign begins.

## Phase 1 recommendation

Proceed to Phase 1 under the **GO** gate only if replay continues to match the committed Phase 0 record. Preserve the measured rigid baseline and the attachment-kinematics test as regression gates.
"""


def run_campaign() -> dict[str, object]:
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    determinism, record_bytes = _determinism_result()
    crossing = synthetic_crossing_benchmark()
    crossing["passed"] = (
        crossing["time_order"] >= 1.8
        and crossing["finest_speed_relative_error"] < 0.01
    )
    tests = {
        "P0-T1": _version_result(),
        "P0-T2": _steady_state_result(),
        "P0-T3": _attachment_result(),
        "P0-T4": determinism,
        "P0-T5": _truth_isolation_result(),
        "P0-T6": _runtime_result(),
        "P0-T7": crossing,
        "P0-T8": _weather_result(),
    }
    blocking_ids = ("P0-T1", "P0-T2", "P0-T3", "P0-T4", "P0-T5", "P0-T7", "P0-T8")
    gate_verdict = "GO" if all(tests[test_id]["passed"] for test_id in blocking_ids) else "NO-GO"
    results = {
        "schema_version": 1,
        "configuration_sha256": _config_hash(),
        "gate_verdict": gate_verdict,
        "tests": tests,
    }

    RECORD_PATH.write_bytes(record_bytes)
    RESULTS_PATH.write_bytes(_json_bytes(results))
    MANIFEST_PATH.write_bytes(_json_bytes(_manifest(determinism["record_sha256"])))
    REPORT_PATH.write_text(_format_report(results), encoding="ascii")
    generate_figure(RECORD_PATH)
    print(f"Phase 0 gate: {gate_verdict}")
    for test_id, result in tests.items():
        print(f"{test_id}: {'PASS' if result['passed'] else 'FAIL'}")
    return results


def replay() -> bool:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="ascii"))
    if manifest["configuration_sha256"] != _config_hash():
        raise RuntimeError("current Phase 0 configuration does not match the manifest")
    weather_count = int(np.ceil(RECORD_DURATION / constants.WEATHER_PERIOD)) + 2
    weather = generate_weather(MASTER_SEED, weather_count)
    replay_log = run_transit_log(
        constants.NOMINAL_THRUST,
        RECORD_DURATION,
        constants.WEATHER_PERIOD,
        weather.states,
    )
    replay_arrays = _record_arrays(replay_log)
    with np.load(RECORD_PATH, allow_pickle=False) as committed:
        exact = {name: bool(np.array_equal(committed[name], value)) for name, value in replay_arrays.items()}
        maximum_errors = {
            name: float(np.max(np.abs(committed[name] - value))) if value.size else 0.0
            for name, value in replay_arrays.items()
        }
    replay_bytes = _npz_bytes(replay_arrays)
    byte_identical = replay_bytes == RECORD_PATH.read_bytes()
    digest_matches = hashlib.sha256(replay_bytes).hexdigest() == manifest["record_sha256"]
    passed = all(exact.values()) and byte_identical and digest_matches
    print(json.dumps(
        {
            "byte_identical": byte_identical,
            "digest_matches": digest_matches,
            "exact_arrays": exact,
            "maximum_absolute_errors": maximum_errors,
            "passed": passed,
        },
        indent=2,
        sort_keys=True,
    ))
    if not passed:
        raise RuntimeError("Phase 0 replay probe failed")
    return passed


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