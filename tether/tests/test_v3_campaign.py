"""Plan v3 WP0: cells, seeds, jobs, and the FleetRunSpec extension (Drake-free)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from tether.campaign.fleet_run import FleetRunSpec
from tether.campaign.phase2 import CACHE_PATH, PILOT_SEEDS, STAT_SEEDS
from tether.campaign.v3 import cells
from tether.physics import constants as C


def test_stiffness_field_none_keeps_v1_hash():
    spec = FleetRunSpec(pretension=600.0, heading_gain=250.0, weather_scale=0.5, duration=600.0, warmup=20.0)
    assert spec.config_hash().startswith("ca9fd4636081")  # records/phase2/cache job T600_k250_I0.5
    assert spec.stiffness is None and spec.damping is None
    changed = cells.replace(spec, stiffness=2.0 * C.CABLE_STIFFNESS)
    assert changed.config_hash() != spec.config_hash()
    assert changed.cable_parameters().stiffness == 2.0 * C.CABLE_STIFFNESS
    assert spec.cable_parameters().stiffness == C.CABLE_STIFFNESS


def test_spec_pickled_before_v3_reads_as_plant_constants():
    spec = FleetRunSpec(pretension=600.0)
    state = dict(spec.__dict__)
    del state["stiffness"], state["damping"]
    old = object.__new__(FleetRunSpec)
    old.__dict__.update(state)
    assert old.stiffness is None and old.damping is None
    assert old.config_hash() == spec.config_hash()
    with pytest.raises(AttributeError):
        _ = old.no_such_field


def test_stage_a_cells_are_declared_as_planned():
    specs = cells.cell_specs("A")
    assert len(specs) == 16
    assert set(cells.REUSED_V1_CELLS) <= set(specs)
    assert all(specs[n].formation == "fan" and specs[n].arc_half_angle == 0.55 for n, _, _ in cells.FAN_CELLS)
    assert specs["T600_k500_I0.5_kx0.25"].stiffness == 0.25 * C.CABLE_STIFFNESS
    assert specs["T600_k500_I0.5_kx4"].stiffness == 4.0 * C.CABLE_STIFFNESS
    assert all(s.cable_mode == "recording" and s.weather_direction == "local" and s.duration == 600.0 for s in specs.values())
    assert set(cells.cell_specs("B")) == {"T600_k500_I0.35", "T600_k500_I0.75", "T600_k500_I1.25", "T600_k500_I1.5"}
    assert set(cells.cell_specs("B_ext")) == {"T600_k500_I1.75", "T600_k500_I2.0"}
    with pytest.raises(ValueError):
        cells.cell_specs("C")


def test_parse_cell_round_trips_every_stage():
    for stage in cells.STAGES:
        for name, spec in cells.cell_specs(stage).items():
            parsed = cells.parse_cell(name)
            assert parsed["formation"] == spec.formation
            assert parsed["pretension"] == spec.pretension and parsed["heading_gain"] == spec.heading_gain
            assert parsed["intensity"] == spec.weather_scale
            factor = 1.0 if spec.stiffness is None else spec.stiffness / C.CABLE_STIFFNESS
            assert parsed["stiffness_factor"] == factor


def test_sweep_cells_are_in_intensity_order():
    names = cells.sweep_cells()
    assert names[0] == "T600_k500_I0.35" and names[-1] == "T600_k500_I2.0" and len(names) == 8


@pytest.mark.skipif(not CACHE_PATH.exists(), reason="v1 cascade-grid cache not present")
def test_reused_cells_hash_equal_to_v1_cache_jobs_and_jobs_carry_their_marks():
    with CACHE_PATH.open("rb") as handle:
        payload = pickle.load(handle)
    v1 = {(job.cell, job.seed): job.spec.config_hash() for job in payload["jobs"]}
    jobs = cells.all_jobs("A")
    assert len(jobs) == 16 * (len(PILOT_SEEDS) + len(STAT_SEEDS))
    reused = [job for job in jobs if job.v1_cell is not None]
    assert len(reused) == 10 * 22
    for job in reused:
        assert v1[(job.v1_cell, job.seed)] == job.spec.config_hash()
        assert job.expected_t_up is not None and job.expected_T_peak is not None
        assert job.expected_t_up.shape == job.expected_T_peak.shape
    assert all(job.expected_t_up is None for job in jobs if job.v1_cell is None)
    assert sum(job.pilot for job in jobs) == 16 * len(PILOT_SEEDS)


def test_response_spec_maps_the_run_spec():
    spec = cells.cell_specs("A")["T600_k500_I0.5_kx4"]
    rs = cells.response_spec(spec)
    assert rs.stiffness == 4.0 * C.CABLE_STIFFNESS and rs.damping == C.CABLE_DAMPING
    assert rs.inp.pretension == 600.0 and rs.inp.heading_gain == 500.0 and rs.inp.formation == "parallel"
    fan = cells.response_spec(cells.cell_specs("A")["FAN_T600_I1.0"])
    assert fan.inp.formation == "fan" and fan.inp.arc_half_angle == 0.55 and fan.stiffness == C.CABLE_STIFFNESS
    assert np.isfinite(rs.stiffness)


def test_every_declared_test_has_branches_for_its_verdicts():
    from tether.campaign.v3 import campaign

    for name, spec in campaign.TESTS.items():
        assert spec["blocking"] in (True, False) and spec["work_package"] in (1, 2, 3)
        assert "threshold" in spec and "statement" in spec and spec["branches"], name
        if spec["blocking"]:
            assert {"PASS", "FAIL"} <= set(spec["branches"]), name


def test_gate_record_lists_blocking_tests(tmp_path, monkeypatch):
    from tether.campaign.v3 import campaign

    monkeypatch.setattr(campaign, "RECORD_DIR", tmp_path)
    monkeypatch.setattr(campaign, "DECLARATIONS_PATH", tmp_path / "d.json")
    (tmp_path / "d.json").write_text("{}")
    tests = {"T1.0": {"verdict": "PASS"}, "T1.1": {"verdict": "FAIL"}, "T1.2": {"verdict": "PASS"}, "T1.3": {"verdict": "PASS"}, "T1.4": {"verdict": "PASS-exact"}}
    campaign.write_gate(1, "OPEN", tests, ["ok"], {"cache": "x"})
    import json as _json
    gate = _json.loads((tmp_path / "gate_wp1.json").read_text())
    assert gate["failing_blocking_tests"] == ["T1.1"] and "T1.4" not in gate["blocking_in_scope"]


slow = pytest.mark.skipif(__import__("os").environ.get("TETHER_SLOW") != "1", reason="full 620 s mission; set TETHER_SLOW=1")


@slow
def test_v3_slow_reused_cell_seed_2003_reproduces_v1_marks():
    from tether.campaign.v3.compute import run_v3_job
    from tether.campaign.v3.intervention import validate

    job = next(j for j in cells.all_jobs("A") if j.cell == cells.CALIBRATION_CELL and j.seed == 2003)
    out = run_v3_job(job)
    assert out["reproduction"]["checked"] and out["reproduction"]["passed"]
    assert validate(out["intervention"]["rows"])["passed"]


def test_body_fields_none_keep_hash_and_plant_constants():
    from tether.physics.fleet import BodyParameters
    spec = FleetRunSpec(pretension=600.0, heading_gain=250.0, weather_scale=0.5, duration=600.0, warmup=20.0)
    assert spec.config_hash().startswith("ca9fd4636081")  # unchanged by the three new fields
    assert spec.vessel_mass is None and spec.load_drag is None and spec.vessel_drag is None
    assert spec.body_parameters() == BodyParameters()
    heavy = cells.replace(spec, vessel_mass=1250.0)
    assert heavy.config_hash() != spec.config_hash()
    assert heavy.body_parameters().vessel_mass == 1250.0
    assert heavy.body_parameters().load_mass == C.LOAD_MASS
    fixed_gamma = cells.replace(spec, stiffness=4.0 * C.CABLE_STIFFNESS, damping=2.0 * C.CABLE_DAMPING,
                                load_drag=2.0 * C.LOAD_LINEAR_DRAG, vessel_drag=2.0 * C.VESSEL_LINEAR_DRAG)
    b = fixed_gamma.body_parameters()
    assert b.load_linear_drag == 2.0 * C.LOAD_LINEAR_DRAG and b.vessel_linear_drag == 2.0 * C.VESSEL_LINEAR_DRAG
    assert fixed_gamma.config_hash() != heavy.config_hash()
