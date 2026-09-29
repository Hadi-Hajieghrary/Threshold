"""Phase 1 mechanics campaign artifact tests."""

import json
import hashlib

import numpy as np

from tether.analysis.phase1_figures import generate_figures
from tether.campaign.phase1 import (
    _derive_phase1r_decision,
    _phase1r_record_metric_checks,
    run_deterministic_campaign,
    run_mechanics_campaign,
    replay_phase1r,
    run_phase1r_campaign,
)


def test_phase1r_gate_fields_are_derived_from_blocking_statuses():
    test_ids = (
        "P1-T1",
        "P1-T2",
        "P1-T2b",
        "P1-T3",
        "P1-T4",
        "P1-T5",
        "P1-T6",
        "P1-T7",
        "P1-T9",
    )
    passing = {test_id: {"status": "PASS"} for test_id in test_ids}
    assert _derive_phase1r_decision(passing) == ("GO", "PASS", None, [])

    failing = {**passing, "P1-T6": {"status": "FAIL"}}
    assert _derive_phase1r_decision(failing) == (
        "NO-GO",
        "FAIL",
        "blocking criteria not satisfied: P1-T6 (FAIL)",
        ["P1-T6"],
    )


def test_phase1_mechanics_json_is_deterministic_and_scope_is_explicit(tmp_path):
    results_path = tmp_path / "phase1_mechanics.json"
    report_path = tmp_path / "phase1_mechanics_report.md"

    first = run_mechanics_campaign(results_path, report_path)
    first_bytes = results_path.read_bytes()
    second = run_mechanics_campaign(results_path, report_path)

    assert first == second
    assert first_bytes == results_path.read_bytes()
    assert json.loads(first_bytes)["phase1_gate"] == "NOT_EVALUATED"
    assert "no full Phase 1 GO is claimed" in report_path.read_text(encoding="ascii")


def test_phase1_deterministic_records_are_reproducible_and_verdicts_are_honest(
    tmp_path,
):
    results_path = tmp_path / "phase1_deterministic.json"
    record_path = tmp_path / "phase1_deterministic.npz"
    report_path = tmp_path / "phase1_deterministic_report.md"

    first = run_deterministic_campaign(results_path, record_path, report_path)
    first_json = results_path.read_bytes()
    first_record = record_path.read_bytes()
    second = run_deterministic_campaign(results_path, record_path, report_path)

    assert first == second
    assert first_json == results_path.read_bytes()
    assert first_record == record_path.read_bytes()
    assert first["phase1_gate"] == "NO-GO"
    assert first["deterministic_verdict"] == "PIVOT"
    assert first["tests"]["P1-T4"]["verdict"] == "UNDER-POWERED"
    assert first["tests"]["P1-T6"]["literal_verdict"] == "FAIL"
    assert first["tests"]["P1-T6"]["verdict"] == "PIVOT"
    assert first["tests"]["P1-T6"]["decisive_no_go"]
    assert "no full Phase 1 GO is claimed" in report_path.read_text(encoding="ascii")

    with np.load(record_path, allow_pickle=False) as record:
        assert record["excursion_complete"].shape == (120,)
        assert np.allclose(np.diff(record["full_transit_time_s"]), 1.0e-3)
        assert record["full_transit_tension_n"].shape[1] == 5
        assert np.allclose(np.diff(record["full_impact_time_s"]), 1.0e-3)
        assert np.any(record["full_impact_engagement_event_code"][:, 2] == 1)

    f1_path, f2_path = generate_figures(
        record_path,
        tmp_path / "F1.png",
        tmp_path / "F2.png",
    )
    assert f1_path.stat().st_size > 0
    assert f2_path.stat().st_size > 0


def test_phase1r_campaign_replay_manifest_and_figures(
    tmp_path,
    phase1r_campaign_inputs,
):
    paths = {
        "results_path": tmp_path / "phase1r_results.json",
        "record_path": tmp_path / "phase1r_records.npz",
        "manifest_path": tmp_path / "phase1r_manifest.json",
        "report_path": tmp_path / "phase1r_report.md",
        "figure_f1_path": tmp_path / "phase1r_F1.png",
        "figure_f2_path": tmp_path / "phase1r_F2.png",
    }
    results = run_phase1r_campaign(**paths, **phase1r_campaign_inputs)

    assert results["phase1_gate"] == "NO-GO"
    assert results["tests"]["P1-T5"]["status"] == "MODEL_INVALID_LOW_SPEED"
    assert results["tests"]["P1-T5"]["full_plant_energy_balance"]["status"] == "PASS"
    assert not results["tests"]["P1-T5"]["physical_plant_failure"]
    assert results["tests"]["P1-T9"]["status"] == "FAIL"
    assert results["independent_no_go_basis"] == ["P1-T9"]
    assert all(path.exists() for path in paths.values())
    with np.load(paths["record_path"], allow_pickle=False) as record:
        required = {
            "p1t1_bilateral_truth",
            "p1t1_unilateral_truth",
            "p1t2b_impact_truth",
            "mechanics_impact_tension_n",
            "t9_production_discrete_1ms_load_force_n",
            "trace_truth",
            "trace_kinetic_energy_j",
            "trace_spring_potential_j",
            "trace_gust_power_w",
            "trace_thrust_power_w",
            "trace_linear_drag_power_w",
            "trace_cable_damping_dissipation_w",
            "t6_probe_acceleration_mps2",
        }
        assert required <= set(record.files)
    manifest = json.loads(paths["manifest_path"].read_text(encoding="ascii"))
    assert manifest["runtime"]["python_version"]
    assert manifest["source"]["status"] in {"clean", "dirty", "unavailable"}
    assert manifest["probe"]["numeric_absolute_tolerance"] == 1.0e-6
    assert manifest["probe"]["full_plant_energy_fields_compared"] == 16
    assert manifest["configuration"]["p1_t6_probe"]["fit_windows_s"] == [
        0.002,
        0.005,
        0.01,
        0.02,
    ]
    assert set(manifest["acceptance_criteria"]) == {
        "P1-T1",
        "P1-T2",
        "P1-T2b",
        "P1-T3",
        "P1-T4",
        "P1-T5_full_plant_energy",
        "P1-T5_impact",
        "P1-T6",
        "P1-T7",
        "P1-T8",
        "P1-T9",
        "P1-T10",
    }
    for artifact in manifest["artifacts"].values():
        artifact_path = next(
            path for path in paths.values() if str(path) == artifact["path"]
        )
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact["sha256"]

    assert all(
        _phase1r_record_metric_checks(paths["record_path"], results).values()
    )
    report = paths["report_path"].read_text(encoding="ascii")
    assert "NO-GO is independently sustained by P1-T9 alone" in report
    assert "not a physical plant failure" in report
    assert replay_phase1r(**paths, campaign_inputs=phase1r_campaign_inputs)
