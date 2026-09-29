"""Run and replay the deterministic P1-T5R impact-model study."""

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
from tether.physics.phase1_deterministic import PRETENSION_LEVELS
from tether.physics.phase1_t5r import (
    MODEL_C_JUSTIFICATION,
    MODEL_FORMULAS,
    P1_T5R_BASELINE_DURATION,
    P1_T5R_CALIBRATION_TOLERANCE,
    P1_T5R_FORCING_SPEEDS,
    P1_T5R_GUST_DURATION,
    P1_T5R_GUST_RATIO,
    P1_T5R_HORIZON,
    P1_T5R_MAXIMUM_RELATIVE_ERROR,
    P1_T5R_NOMINAL_SPEEDS,
    P1_T5R_PHYSICS_STEPS,
    P1_T5R_PRODUCTION_STEP,
    P1_T5R_R2_MINIMUM,
    P1_T5R_RECORDING_STEP,
    P1_T5R_ZERO_REPEATS,
    PRIMARY_MODEL,
    analyse_p1_t5r_record,
    p1_t5r_record_arrays,
    preregistered_cell_specs,
    run_p1_t5r_grid,
)

RECORD_DIR = ROOT / "records" / "phase1"
RESULTS_PATH = RECORD_DIR / "phase1_t5r_results.json"
RECORD_PATH = RECORD_DIR / "phase1_t5r_records.npz"
MANIFEST_PATH = RECORD_DIR / "phase1_t5r_manifest.json"
REPORT_PATH = ROOT / "reports" / "phase1_t5r_report.md"
REPLAY_NUMERIC_ATOL = 1.0e-10

PROTECTED_ARTIFACT_PATHS = {
    "phase0_manifest": ROOT / "records" / "phase0" / "manifest.json",
    "phase0_results": ROOT / "records" / "phase0" / "phase0_results.json",
    "phase0_record": ROOT / "records" / "phase0" / "transit_nominal.npz",
    "phase0_report": ROOT / "reports" / "phase0_report.md",
    "phase0_figure": ROOT / "reports" / "figures" / "phase0_validation.png",
    "phase1_mechanics_results": RECORD_DIR / "phase1_mechanics.json",
    "phase1_mechanics_report": ROOT / "reports" / "phase1_mechanics_report.md",
    "phase1_deterministic_results": RECORD_DIR / "phase1_deterministic.json",
    "phase1_deterministic_record": RECORD_DIR / "phase1_deterministic.npz",
    "phase1_deterministic_report": ROOT / "reports" / "phase1_deterministic_report.md",
    "phase1r_results": RECORD_DIR / "phase1r_results.json",
    "phase1r_record": RECORD_DIR / "phase1r_records.npz",
    "phase1r_manifest": RECORD_DIR / "phase1r_manifest.json",
    "phase1r_report": ROOT / "reports" / "phase1r_report.md",
    "phase1r_figure_f1": ROOT / "reports" / "figures" / "phase1r_F1_acceleration_threshold.png",
    "phase1r_figure_f2": ROOT / "reports" / "figures" / "phase1r_F2_depth_comparison.png",
    "phase1_t9r_results": RECORD_DIR / "phase1_t9r_results.json",
    "phase1_t9r_record": RECORD_DIR / "phase1_t9r_records.npz",
    "phase1_t9r_manifest": RECORD_DIR / "phase1_t9r_manifest.json",
    "phase1_t9r_report": ROOT / "reports" / "phase1_t9r_report.md",
}


def protected_artifact_hashes() -> dict[str, str]:
    return {
        _recorded_path(path): _sha256_file(path)
        for path in PROTECTED_ARTIFACT_PATHS.values()
    }


def _configuration() -> dict[str, object]:
    production_specs = [
        spec
        for spec in preregistered_cell_specs()
        if np.isclose(spec.physics_step_s, P1_T5R_PRODUCTION_STEP)
    ]
    return {
        "test_id": "P1-T5R",
        "scope": "deterministic low-speed full-formation impact-model study",
        "plant": "existing five-vessel Drake discrete SAP full formation",
        "target_cable": 2,
        "physics_steps_s": list(P1_T5R_PHYSICS_STEPS),
        "production_physics_step_s": P1_T5R_PRODUCTION_STEP,
        "event_logging_step_s": P1_T5R_RECORDING_STEP,
        "nominal_speed_grid_mps": P1_T5R_NOMINAL_SPEEDS.tolist(),
        "adapted_forcing_entry_speeds_mps": P1_T5R_FORCING_SPEEDS.tolist(),
        "pretensions_n": PRETENSION_LEVELS.tolist(),
        "zero_speed_repeats_per_pretension_and_step": P1_T5R_ZERO_REPEATS,
        "zero_speed_baseline_duration_s": P1_T5R_BASELINE_DURATION,
        "excursion": {
            "differential_gust_ratio": P1_T5R_GUST_RATIO,
            "gust_duration_s": P1_T5R_GUST_DURATION,
            "horizon_s": P1_T5R_HORIZON,
            "initial_target_extension_m": 0.002,
            "forcing_adaptation_basis": (
                "deterministic pre-fit endpoint probes; no held-out model score used"
            ),
        },
        "peak_definition": (
            "first sampled local maximum of target-cable applied tension after "
            "the first geometric reengagement; zero-speed controls use the first "
            "sampled local maximum after positive elongation"
        ),
        "return_speed_definition": (
            "linearly interpolated elongation rate at the first geometric up-crossing"
        ),
        "model_candidates": {
            "A": {
                "formula": MODEL_FORMULAS["A"],
                "role": "original fixed through-origin baseline",
                "preregistered_status": "FAILED_BASELINE_RETAINED",
                "qualification_eligible": False,
            },
            "B": {
                "formula": MODEL_FORMULAS["B"],
                "role": "primary simplest nonzero-baseline model",
                "qualification_eligible": True,
            },
            "C": {
                "formula": MODEL_FORMULAS["C"],
                "role": "predeclared secondary model",
                "qualification_eligible": True,
                "justification": MODEL_C_JUSTIFICATION,
            },
        },
        "primary_model": PRIMARY_MODEL,
        "fit_policy": "fit only production-step cells assigned to train",
        "split": {
            "method": "checkerboard transfer across speed and pretension",
            "training_cell_identifiers": [
                spec.identifier for spec in production_specs if spec.split == "train"
            ],
            "heldout_cell_identifiers": [
                spec.identifier
                for spec in production_specs
                if spec.split == "heldout"
            ],
        },
        "heldout_acceptance_criteria": {
            "calibration_slope_relative_error": (
                f"<= {P1_T5R_CALIBRATION_TOLERANCE}"
            ),
            "uncentred_r_squared": f"> {P1_T5R_R2_MINIMUM}",
            "maximum_relative_peak_error": (
                f"<= {P1_T5R_MAXIMUM_RELATIVE_ERROR}"
            ),
            "physical_validity": "nonnegative dynamic slope and predictions",
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


def p1_t5r_record_metric_checks(
    record_path: Path,
    results: dict[str, object],
) -> dict[str, bool]:
    expected = preregistered_cell_specs()
    with np.load(record_path, allow_pickle=False) as record:
        metadata_match = bool(
            np.array_equal(
                record["cell_identifier"],
                np.array([spec.identifier for spec in expected]),
            )
            and np.array_equal(
                record["cell_split"],
                np.array([spec.split for spec in expected]),
            )
            and np.array_equal(
                record["cell_physics_step_s"],
                np.array([spec.physics_step_s for spec in expected]),
            )
        )
        recomputed = analyse_p1_t5r_record(record)
    return {
        "metadata_matches_preregistration": metadata_match,
        "fits_and_heldout_metrics_match_raw_channels": _metric_subset_matches(
            recomputed, results["study"]
        ),
    }


def _candidate_rows(study: dict[str, object]) -> str:
    rows = []
    for name in ("A", "B", "C"):
        candidate = study["candidates"][name]
        metrics = candidate["heldout_metrics"]
        rows.append(
            f"| {name} | {metrics['calibration_slope_through_origin']:.9f} | "
            f"{metrics['uncentred_r_squared']:.9f} | "
            f"{metrics['maximum_relative_peak_error']:.6f} | "
            f"{metrics['mae_n']:.3f} | {metrics['rmse_n']:.3f} | "
            f"{candidate['status']} |"
        )
    return "\n".join(rows)


def _per_pretension_rows(study: dict[str, object]) -> str:
    rows = []
    for model_name in ("A", "B", "C"):
        metrics = study["candidates"][model_name]["heldout_metrics"]
        for pretension, values in metrics["per_pretension"].items():
            rows.append(
                f"| {model_name} | {float(pretension):.1f} | {values['count']} | "
                f"{values['calibration_slope_through_origin']:.9f} | "
                f"{values['uncentred_r_squared']:.9f} | "
                f"{values['maximum_relative_peak_error']:.6f} | "
                f"{values['mae_n']:.3f} | {values['rmse_n']:.3f} |"
            )
    return "\n".join(rows)


def _baseline_rows(study: dict[str, object]) -> str:
    return "\n".join(
        f"| {1000.0 * row['physics_step_s']:.2f} | "
        f"{row['pretension_n']:.1f} | {row['repeat_count']} | "
        f"{row['mean_peak_tension_n']:.6f} | {row['peak_range_n']:.3e} | "
        f"{row['maximum_relative_deviation']:.3e} |"
        for row in study["baseline_repeatability"]
    )


def _format_report(results: dict[str, object]) -> str:
    study = results["study"]
    domain = study["achieved_domain"]
    selected = study["selected_model"] or "none"
    decision = (
        f"PASS: preregistered model {selected} satisfies every held-out criterion."
        if study["status"] == "PASS"
        else "FAIL: no qualification-eligible preregistered model satisfies every held-out criterion."
    )
    training = ", ".join(study["split"]["training_cell_identifiers"])
    held_out = ", ".join(study["split"]["heldout_cell_identifiers"])
    return f"""# P1-T5R Low-Speed Full-Formation Impact Model

## Decision

**{decision}** The original through-origin model A remains a failed baseline and is not retroactively eligible for qualification. P1-T5R is deterministic and separate from the Phase 1R and P1-T9R records. It does not constitute an overall Phase 1 GO.

## Preregistration and method

The study uses the existing five-vessel Drake SAP full plant and the unchanged excursion/reengagement path. The production physics step is {1000.0 * P1_T5R_PRODUCTION_STEP:.2f} ms, while event tracking and raw-channel logging remain at {1000.0 * P1_T5R_RECORDING_STEP:.1f} ms. A matched {1000.0 * P1_T5R_PHYSICS_STEPS[0]:.2f} ms grid provides the step-sensitivity diagnostic. The first sampled target-cable tension maximum after the first geometric reengagement is the peak definition; this definition was fixed before held-out scoring.

The nominal return-speed grid is {P1_T5R_NOMINAL_SPEEDS.tolist()} m/s. Deterministic endpoint probes established the pre-fit forcing grid {P1_T5R_FORCING_SPEEDS.tolist()} m/s. The production cells achieved entry speeds {domain['measured_entry_speed_range_mps']} m/s and return speeds {domain['measured_return_speed_range_mps']} m/s. The held-out cells achieved entry speeds {domain['heldout_measured_entry_speed_range_mps']} m/s and return speeds {domain['heldout_measured_return_speed_range_mps']} m/s.

The preregistered candidates are: A, `{MODEL_FORMULAS['A']}`; B, `{MODEL_FORMULAS['B']}`; and C, `{MODEL_FORMULAS['C']}`. B is primary because it is the simplest candidate that represents the measured nonzero baseline. C was included before scoring because {MODEL_C_JUSTIFICATION.lower()}

## Archived split

Training cells: {training}

Held-out cells: {held_out}

Every fit uses only the listed training cells. The checkerboard assignment tests speed interpolation and transfer across all three pretensions.

## Held-out metrics

| Model | Calibration slope | Uncentred R2 | Maximum relative error | MAE [N] | RMSE [N] | Status |
|:---:|---:|---:|---:|---:|---:|:---:|
{_candidate_rows(study)}

Qualification requires calibration-slope error at most 6%, uncentred R2 greater than 0.99, maximum relative peak error at most 6%, and nonnegative dynamic slope and predictions.

## Metrics by pretension

| Model | Pretension [N] | Cells | Calibration slope | Uncentred R2 | Maximum relative error | MAE [N] | RMSE [N] |
|:---:|---:|---:|---:|---:|---:|---:|---:|
{_per_pretension_rows(study)}

## Diagnostics

| Physics step [ms] | Pretension [N] | Repeats | Mean zero-speed peak [N] | Peak range [N] | Maximum relative deviation |
|---:|---:|---:|---:|---:|---:|
{_baseline_rows(study)}

Across all matched 0.50 and 0.25 ms cells, the maximum return-speed difference is {study['physics_step_sensitivity']['maximum_return_speed_absolute_difference_mps']:.9e} m/s. The maximum peak-tension difference is {study['physics_step_sensitivity']['maximum_peak_tension_absolute_difference_n']:.9e} N, or {study['physics_step_sensitivity']['maximum_peak_tension_relative_difference']:.9e} relative.

## Provenance and scope

- Source revision: `{results['source']['revision']}`; worktree status: `{results['source']['status']}`.
- Deterministic full-plant cells: {results['execution']['cell_count']}; stochastic production: `NOT_RUN`.
- The NPZ archives full state, all cable tensions, elongations, elongation rates, relative loads, split membership, and timing metadata for every cell.
- Replay regenerates into temporary paths, derives targets and fits from raw channels, verifies manifest hashes, and does not overwrite authoritative artifacts.
- All protected Phase 0, Phase 1R, and P1-T9R hashes remained unchanged during generation: `{results['artifact_preservation']['unchanged_during_campaign']}`.

## Remaining blocker recommendation

P1-T4 remains `UNDER-POWERED` and was not rerun here. Retain `NO-GO` for the overall Phase 1 decision. Address P1-T4 in a separate preregistered full-plant study with stronger or longer deterministic excitation at the qualified 0.25 ms physics step and unchanged 1 ms event cadence; do not infer its outcome from P1-T5R. No stochastic production campaign was launched.
"""


def _manifest(
    configuration: dict[str, object],
    source: dict[str, object],
    artifacts: dict[str, dict[str, str]],
    protected_hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "driver": "python -m tether.campaign.phase1_t5r run",
        "source": source,
        "runtime": {"python_version": platform.python_version()},
        "configuration": configuration,
        "configuration_sha256": _configuration_sha256(configuration),
        "acceptance_criteria": configuration["heldout_acceptance_criteria"],
        "artifacts": artifacts,
        "record_sha256": artifacts["record"]["sha256"],
        "protected_artifact_hashes": protected_hashes,
        "probe": {
            "command": "python -m tether.campaign.phase1_t5r replay",
            "temporary_regeneration": True,
            "raw_target_and_fit_recomputation": True,
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
    """Run P1-T5R without modifying any existing authoritative artifact."""
    start = time.perf_counter()
    protected_before = protected_artifact_hashes()
    source = _source_state()
    if cells is None:
        cells = run_p1_t5r_grid()
    arrays = p1_t5r_record_arrays(cells)
    record_bytes = _npz_bytes(arrays)
    with np.load(io.BytesIO(record_bytes), allow_pickle=False) as record:
        study = analyse_p1_t5r_record(record)
    configuration = _configuration()
    for path in (results_path, record_path, manifest_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_bytes(record_bytes)
    if protected_before != protected_artifact_hashes():
        raise RuntimeError("P1-T5R modified a protected authoritative artifact")
    results = {
        "schema_version": 1,
        "scope": "P1-T5R deterministic low-speed impact-model study only",
        "gate": study["status"],
        "phase1_gate": "NO-GO",
        "selected_model": study["selected_model"],
        "configuration": configuration,
        "configuration_sha256": _configuration_sha256(configuration),
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "source": source,
        "execution": {
            "cell_count": len(cells),
            "production_cell_count": sum(
                np.isclose(cell.spec.physics_step_s, P1_T5R_PRODUCTION_STEP)
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
            "P1-T4": (
                "UNDER-POWERED; requires a separate preregistered stronger or "
                "longer deterministic full-plant excitation study"
            ),
            "overall_phase1": "NO-GO retained; P1-T5R cannot establish overall GO",
        },
        "stochastic_production": {
            "status": "NOT_RUN",
            "production_grid_launched": False,
        },
    }
    results_path.write_bytes(_json_bytes(results))
    report_path.write_text(_format_report(results), encoding="ascii")
    artifacts = {
        "record": {
            "path": _recorded_path(record_path),
            "sha256": _sha256_file(record_path),
        },
        "results": {
            "path": _recorded_path(results_path),
            "sha256": _sha256_file(results_path),
        },
        "report": {
            "path": _recorded_path(report_path),
            "sha256": _sha256_file(report_path),
        },
    }
    manifest_path.write_bytes(
        _json_bytes(_manifest(configuration, source, artifacts, protected_before))
    )
    if protected_before != protected_artifact_hashes():
        raise RuntimeError("P1-T5R modified a protected authoritative artifact")
    print(
        f"P1-T5R: {study['status']} "
        f"(selected model: {study['selected_model'] or 'none'}; Phase 1 NO-GO)"
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
    """Regenerate P1-T5R in temporary paths and verify raw-derived results."""
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
        raise RuntimeError("current P1-T5R configuration does not match the manifest")
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
    raw_checks = p1_t5r_record_metric_checks(record_path, committed_results)
    with tempfile.TemporaryDirectory(prefix="phase1-t5r-replay-") as directory:
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
        replay_comparable = json.loads(_json_bytes(replay_results))
        committed_comparable = json.loads(
            json.dumps(committed_results, sort_keys=True)
        )
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
        "p1_t5r_artifacts_unchanged": before == after,
        "protected_artifacts_unchanged": protected_before == protected_after,
        "passed": passed,
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("P1-T5R replay probe failed")
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