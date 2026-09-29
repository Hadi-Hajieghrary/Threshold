"""Run and replay the deterministic full-plant P1-T4R study."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
from pathlib import Path
import tempfile
import time

import numpy as np

from tether.campaign.phase1 import (
    ROOT,
    _configuration_sha256,
    _json_bytes,
    _npz_bytes,
    _recorded_path,
    _sha256_file,
    _source_state,
)
from tether.campaign.phase1_t5r import PROTECTED_ARTIFACT_PATHS as EARLIER_ARTIFACTS
from tether.physics import constants
from tether.physics.phase1_deterministic import PRETENSION_LEVELS
from tether.physics.phase1_t4r import (
    P1_T4R_BRANCHES,
    P1_T4R_DERIVATION_DEPTH_ERROR_LIMIT,
    P1_T4R_ENTRY_SPEEDS,
    P1_T4R_GUST_RATIOS,
    P1_T4R_LITERAL_DEPTH_RATIO_LIMIT,
    P1_T4R_PHYSICS_STEPS,
    P1_T4R_PRODUCTION_STEP,
    P1_T4R_RECORDING_STEP,
    P1_T4R_SPEED_ERROR_LIMIT,
    P1_T4R_STEP_SENSITIVITY_LIMIT,
    analyse_record,
    preregistered_cell_specs,
    record_arrays,
    run_grid,
)

RECORD_DIR = ROOT / "records" / "phase1"
RESULTS_PATH = RECORD_DIR / "phase1_t4r_results.json"
RECORD_PATH = RECORD_DIR / "phase1_t4r_records.npz"
MANIFEST_PATH = RECORD_DIR / "phase1_t4r_manifest.json"
REPORT_PATH = ROOT / "reports" / "phase1_t4r_report.md"
REPLAY_NUMERIC_ATOL = 1.0e-10

PROTECTED_ARTIFACT_PATHS = {
    **EARLIER_ARTIFACTS,
    "phase1_t5r_results": RECORD_DIR / "phase1_t5r_results.json",
    "phase1_t5r_record": RECORD_DIR / "phase1_t5r_records.npz",
    "phase1_t5r_manifest": RECORD_DIR / "phase1_t5r_manifest.json",
    "phase1_t5r_report": ROOT / "reports" / "phase1_t5r_report.md",
}


def protected_artifact_hashes() -> dict[str, str]:
    return {
        _recorded_path(path): _sha256_file(path)
        for path in PROTECTED_ARTIFACT_PATHS.values()
    }


def _configuration() -> dict[str, object]:
    return {
        "test_id": "P1-T4R",
        "scope": "deterministic matched-turning-state six-body/five-cable study",
        "plant": "existing Drake discrete MultibodyPlant with SAP contact approximation",
        "target_cable": constants.VESSEL_COUNT // 2,
        "plan_definition": {
            "statement": "Ballistic symmetry (Prop. 2): scripted excursions with lambda < 1",
            "speed_inequality": "abs(V_up - u) / u < 0.05",
            "depth_inequality": "depth < 1.1 * u^2 / (2 * a0)",
            "a0": "T0 / m_eff",
        },
        "derivation_definition": {
            "hypothesis": "H4 slow forcing; eta = 0; W_rel constant over the excursion",
            "net_acceleration": "a = a0 - W_rel / m_eff > 0",
            "speed_prediction": "V_up = u",
            "depth_prediction": "depth = u^2 / (2 * a)",
            "primary_interpretation": True,
        },
        "ambiguity_policy": (
            "score the literal table and the constant-force derivation separately; "
            "do not choose the favorable interpretation after observing results"
        ),
        "physics_steps_s": list(P1_T4R_PHYSICS_STEPS),
        "production_physics_step_s": P1_T4R_PRODUCTION_STEP,
        "event_and_recording_step_s": P1_T4R_RECORDING_STEP,
        "pretensions_n": PRETENSION_LEVELS.tolist(),
        "entry_speeds_mps": P1_T4R_ENTRY_SPEEDS.tolist(),
        "gust_ratios": P1_T4R_GUST_RATIOS.tolist(),
        "turning_state_construction": (
            "simulate from e=0 and measured negative entry rate under the "
            "conservative constant-W_rel full plant; interpolate the first "
            "negative-to-positive radial-rate crossing from 1 ms raw samples"
        ),
        "branches": {
            "isolated_unforced": {
                "role": "literal-table primary branch",
                "transient_gust": "disabled",
                "weather": "disabled",
                "drag": "disabled only for this conservative isolation",
                "cable_damping_n_s_per_m": 0.0,
                "nominal_balance": "constant thrust and load balancing force retained",
            },
            "constant_force_control": {
                "role": "primary Proposition 2 derivation branch",
                "W_rel": "constant lambda * T0 over the return",
                "weather": "disabled",
                "drag": "disabled only for this conservative control",
                "cable_damping_n_s_per_m": 0.0,
                "nominal_balance": "constant thrust and load balancing force retained",
            },
            "forcing_active_confound": {
                "role": "diagnostic only",
                "W_rel": "square differential gust remains active over the return",
                "weather": "disabled",
                "drag": "plan values retained",
                "cable_damping_n_s_per_m": constants.CABLE_DAMPING,
                "thrust": "steady-tow calibrated value retained",
            },
        },
        "acceptance_criteria": {
            "literal_speed": f"every eligible cell < {P1_T4R_SPEED_ERROR_LIMIT}",
            "literal_depth": (
                f"every eligible cell depth ratio < {P1_T4R_LITERAL_DEPTH_RATIO_LIMIT}"
            ),
            "derivation_speed": f"every eligible cell < {P1_T4R_SPEED_ERROR_LIMIT}",
            "derivation_depth": (
                "every eligible cell relative error <= "
                f"{P1_T4R_DERIVATION_DEPTH_ERROR_LIMIT}"
            ),
            "step_sensitivity": (
                "every matched speed-error difference <= "
                f"{P1_T4R_STEP_SENSITIVITY_LIMIT} (one percentage point)"
            ),
            "events": "zero-rate turning at t=0 < geometric up-crossing <= force onset",
            "aggregation": "worst-cell gate; no averaging",
        },
        "stochastic_production": "NOT_RUN",
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


def record_metric_checks(
    record_path: Path,
    results: dict[str, object],
    *,
    expected_specs=None,
) -> dict[str, bool]:
    expected = preregistered_cell_specs() if expected_specs is None else expected_specs
    with np.load(record_path, allow_pickle=False) as record:
        metadata_match = bool(
            np.array_equal(
                record["cell_identifier"],
                np.array([spec.identifier for spec in expected]),
            )
            and np.array_equal(
                record["cell_physics_step_s"],
                np.array([spec.physics_step_s for spec in expected]),
            )
        )
        recomputed = analyse_record(record)
    return {
        "metadata_matches_preregistration": metadata_match,
        "metrics_match_raw_state_event_and_work_channels": _metric_subset_matches(
            recomputed, results["study"]
        ),
    }


def _format_report(results: dict[str, object]) -> str:
    study = results["study"]
    literal = study["literal_table_interpretation"]
    derivation = study["derivation_interpretation"]
    sensitivity = study["physics_step_sensitivity"]
    literal_decision = literal["status"]
    derivation_decision = derivation["core_status"]
    return f"""# P1-T4R Full-Plant Ballistic Symmetry

## Decision

**P1-T4R primary result: {results['gate']}.** The constant-force interpretation implied by Proposition 2 is primary because its derivation explicitly assumes `eta = 0` and constant `W_rel`. Its core result is `{derivation_decision}`. The literal Phase 1 table interpretation is `{literal_decision}`. The exact table depth inequality applied to derivation-consistent cells is `{derivation['exact_table_status']}`. Controls are diagnostic and are not pooled with either gate.

This bounded deterministic study does not establish an overall Phase 1 GO. P1-T5R remains a blocker regardless of P1-T4R, and stochastic production was not launched.

## Exact plan criterion and ambiguity

The plan states: "P1-T4 Ballistic symmetry (Prop. 2): scripted excursions with lambda < 1" with `abs(V_up - u) / u < 0.05` and `depth < 1.1 * u^2 / (2 * a0)`, where `a0 = T0 / m_eff`.

Proposition 2 instead assumes H4 slow forcing, `eta = 0`, and constant `W_rel`, giving `a = a0 - W_rel / m_eff`, `V_up = u`, and `depth = u^2 / (2*a)`. For nonzero lambda these depth formulas differ. Both were preregistered and scored; the derivation interpretation is primary.

## Plant, branches, and grid

All cells use the existing six-body/five-cable Drake SAP plant. Production physics is {1000.0 * P1_T4R_PRODUCTION_STEP:.2f} ms; the matched sensitivity step is {1000.0 * P1_T4R_PHYSICS_STEPS[0]:.2f} ms; event tracking and raw logging remain {1000.0 * P1_T4R_RECORDING_STEP:.1f} ms.

The grid crosses pretensions {PRETENSION_LEVELS.tolist()} N, entry speeds {P1_T4R_ENTRY_SPEEDS.tolist()} m/s, and lambda {P1_T4R_GUST_RATIOS.tolist()}. It covers the Phase1R lambda 0.8/0.95 and speed 0.10/0.20 domain and the P1-T5R 0.10-0.50 m/s domain. For each grid point, the full plant departs from `e=0` with measured negative radial speed under conservative constant `W_rel`; the first negative-to-positive radial-rate crossing defines the interpolated turning state. All three returns start from that same archived state. No predicted depth is imposed.

- `isolated_unforced`: transient gust and weather removed; velocity-dependent drag and cable damping are explicitly disabled only for conservative isolation; constant nominal balance forces remain.
- `constant_force_control`: same conservative plant with constant `W_rel = lambda*T0`, as Proposition 2 requires.
- `forcing_active_confound`: original steady thrust, plan drag, cable damping {constants.CABLE_DAMPING:.1f} N s/m, and active square gust retained; diagnostic only.

## Metrics and worst cells

| Interpretation | Cells | Max speed error | Max relevant depth metric | Event/state valid | Result |
|---|---:|---:|---:|:---:|:---:|
| Literal unforced | {literal['cell_count']} | {literal['maximum_speed_relative_error']:.9f} | ratio {literal['maximum_literal_depth_ratio']:.9f} | {literal['checks']['every_event_order_and_return_valid']} | {literal_decision} |
| Constant-force derivation | {derivation['cell_count']} | {derivation['maximum_speed_relative_error']:.9f} | error {derivation['maximum_derivation_depth_relative_error']:.9f} | {derivation['checks']['every_event_order_and_return_valid']} | {derivation_decision} |

Literal worst speed cell: `{literal['worst_speed_cell']['identifier']}` at {literal['worst_speed_cell']['speed_relative_error']:.9f}. Literal worst depth cell: `{literal['worst_depth_cell']['identifier']}` at ratio {literal['worst_depth_cell']['literal_depth_ratio']:.9f}.

Derivation worst speed cell: `{derivation['worst_speed_cell']['identifier']}` at {derivation['worst_speed_cell']['speed_relative_error']:.9f}. Its worst derivation-depth error is {derivation['worst_derivation_depth_cell']['derivation_depth_relative_error']:.9f}; its worst literal-table depth ratio is {derivation['worst_literal_depth_cell']['literal_depth_ratio']:.9f}.

The maximum 0.50/0.25 ms speed-error difference is {sensitivity['maximum_speed_error_percentage_point_difference']:.9f}, against the 0.01 one-percentage-point limit: `{sensitivity['status']}`. No failing cell is averaged away.

## Archived evidence and provenance

The NPZ contains every cell's exact initial/full state, all five cable tensions, elongations, rates and relative loads, kinetic and spring energy, gust/thrust/drag/weather/constant-force power, damping and clipping dissipation, and target event records. Replay recomputes all metrics from these raw channels, regenerates into temporary paths, verifies hashes, and checks non-overwrite behavior.

- Source revision: `{results['source']['revision']}`; worktree status: `{results['source']['status']}`.
- Deterministic cells: {results['execution']['cell_count']}.
- Prior authoritative artifacts unchanged during campaign: `{results['artifact_preservation']['unchanged_during_campaign']}`.
- Stochastic production: `NOT_RUN`.

## Recommendation

Retain overall Phase 1 `NO-GO`. P1-T5R remains an independent blocker regardless of this P1-T4R result. Use the primary constant-force result to decide the Proposition 2 claim, retain the literal-table result as a separate specification check, and do not infer either from the forcing-active confound branch.
"""


def _manifest(
    configuration: dict[str, object],
    source: dict[str, object],
    artifacts: dict[str, dict[str, str]],
    protected_hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "driver": "python -m tether.campaign.phase1_t4r run",
        "source": source,
        "runtime": {"python_version": platform.python_version()},
        "configuration": configuration,
        "configuration_sha256": _configuration_sha256(configuration),
        "acceptance_criteria": configuration["acceptance_criteria"],
        "artifacts": artifacts,
        "record_sha256": artifacts["record"]["sha256"],
        "protected_artifact_hashes": protected_hashes,
        "probe": {
            "command": "python -m tether.campaign.phase1_t4r replay",
            "temporary_regeneration": True,
            "raw_state_event_and_work_recomputation": True,
            "numeric_absolute_tolerance": REPLAY_NUMERIC_ATOL,
            "deterministic_record_bytes": True,
            "complete_results_excluding": ["execution.wall_seconds"],
            "report_bytes": True,
            "manifest_hash_verification": True,
            "non_overwriting": True,
        },
    }


def run_campaign(
    results_path: Path = RESULTS_PATH,
    record_path: Path = RECORD_PATH,
    manifest_path: Path = MANIFEST_PATH,
    report_path: Path = REPORT_PATH,
    *,
    cells=None,
) -> dict[str, object]:
    """Run P1-T4R without modifying existing authoritative artifacts."""
    start = time.perf_counter()
    protected_before = protected_artifact_hashes()
    source = _source_state()
    if cells is None:
        cells = run_grid()
    arrays = record_arrays(cells)
    record_bytes = _npz_bytes(arrays)
    with np.load(io.BytesIO(record_bytes), allow_pickle=False) as record:
        study = analyse_record(record)
    configuration = _configuration()
    for path in (results_path, record_path, manifest_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_bytes(record_bytes)
    if protected_before != protected_artifact_hashes():
        raise RuntimeError("P1-T4R modified a protected authoritative artifact")
    gate = study["derivation_interpretation"]["core_status"]
    results = {
        "schema_version": 1,
        "scope": "P1-T4R deterministic full-plant symmetry study only",
        "gate": gate,
        "phase1_gate": "NO-GO",
        "configuration": configuration,
        "configuration_sha256": _configuration_sha256(configuration),
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "source": source,
        "execution": {
            "cell_count": len(cells),
            "production_cell_count": sum(
                np.isclose(cell.spec.physics_step_s, P1_T4R_PRODUCTION_STEP)
                for cell in cells
            ),
            "wall_seconds": time.perf_counter() - start,
        },
        "study": study,
        "artifact_preservation": {
            "hashes": protected_before,
            "unchanged_during_campaign": True,
        },
        "remaining_blockers": {
            "P1-T5R": (
                "retained as a blocker regardless of the P1-T4R outcome"
            ),
            "overall_phase1": "NO-GO retained; P1-T4R cannot establish overall GO",
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
        _json_bytes(_manifest(configuration, source, artifacts, protected_before))
    )
    if protected_before != protected_artifact_hashes():
        raise RuntimeError("P1-T4R modified a protected authoritative artifact")
    print(f"P1-T4R: {gate} (Phase 1 NO-GO; P1-T5R blocker retained)")
    return results


def replay(
    results_path: Path = RESULTS_PATH,
    record_path: Path = RECORD_PATH,
    manifest_path: Path = MANIFEST_PATH,
    report_path: Path = REPORT_PATH,
    *,
    campaign_inputs: dict[str, object] | None = None,
) -> bool:
    """Regenerate in temporary paths and verify raw-derived P1-T4R results."""
    paths = {
        "results": results_path,
        "record": record_path,
        "manifest": manifest_path,
        "report": report_path,
    }
    before = {name: _sha256_file(path) for name, path in paths.items()}
    protected_before = protected_artifact_hashes()
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    configuration = _configuration()
    if manifest["configuration_sha256"] != _configuration_sha256(configuration):
        raise RuntimeError("current P1-T4R configuration does not match the manifest")
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
    inputs = campaign_inputs or {}
    expected_specs = (
        tuple(cell.spec for cell in inputs["cells"])
        if "cells" in inputs
        else None
    )
    raw_checks = record_metric_checks(
        record_path,
        committed_results,
        expected_specs=expected_specs,
    )
    with tempfile.TemporaryDirectory(prefix="phase1-t4r-replay-") as directory:
        temporary = Path(directory)
        replay_results = run_campaign(
            temporary / "results.json",
            temporary / "records.npz",
            temporary / "manifest.json",
            temporary / "report.md",
            **inputs,
        )
        record_identical = (
            (temporary / "records.npz").read_bytes() == record_path.read_bytes()
        )
        report_identical = (
            (temporary / "report.md").read_bytes() == report_path.read_bytes()
        )
        replay_comparable = json.loads(_json_bytes(replay_results))
        committed_comparable = json.loads(json.dumps(committed_results, sort_keys=True))
        replay_comparable["execution"].pop("wall_seconds", None)
        committed_comparable["execution"].pop("wall_seconds", None)
        results_match = replay_comparable == committed_comparable
    after = {name: _sha256_file(path) for name, path in paths.items()}
    protected_after = protected_artifact_hashes()
    passed = bool(
        manifest_digests_match
        and all(raw_checks.values())
        and record_identical
        and report_identical
        and results_match
        and before == after
        and protected_before == protected_after
        and manifest["protected_artifact_hashes"] == protected_after
    )
    evidence = {
        "temporary_regeneration": True,
        "committed_files_overwritten": False,
        "manifest_digests_match": manifest_digests_match,
        "raw_metric_checks": raw_checks,
        "record_byte_identical": record_identical,
        "report_byte_identical": report_identical,
        "complete_results_match_excluding_wall_time": results_match,
        "p1_t4r_artifacts_unchanged": before == after,
        "protected_artifacts_unchanged": protected_before == protected_after,
        "passed": passed,
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("P1-T4R replay probe failed")
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