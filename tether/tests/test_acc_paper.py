"""Regression tests for the planar forecast behind the ACC 2027 paper (IEEE_ACC_2027).

Covered: the committed forecast (tether/theory/transmission.py against the declared forecast table) and
its Jacobian-step robustness (Sec. V-A), and the SHA-256 integrity of every file the v3 declaration pins
and of the addenda. The tests that compare with the campaign outputs skip when those outputs are absent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
REC = ROOT / "records" / "v3"

# tether/campaign/fleet_run.py after the FleetRunSpec overrides of addenda 6-7 (vessel_mass,
# load_drag, vessel_drag). The v3 declaration pins the earlier file; this constant pins the current one.
FLEET_RUN_SHA256_AFTER_ADDENDA_6_7 = "0b3aaf377e2cfe6edb3aa56a396b5a4b30f780b148cb9e04b311c6ce3f95581a"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _predictions() -> dict:
    path = REC / "cascade_predictions.json"
    if not path.exists():
        pytest.skip("declared forecast table not present")
    return json.loads(path.read_text())["cells"]


# ------------------------------------------------------------------------- the committed forecast
@pytest.mark.parametrize("cell", ["T600_k500_I0.5", "T600_k500_I0.5_kx0.25", "T600_k500_I0.5_kx4", "FAN_T600_I0.5"])
def test_committed_forecast_is_reproduced_by_the_pinned_theory(cell):
    """Sec. V-A: the cell forecast (median R_ij/g_ij) recomputed from tether/theory/transmission.py."""
    from tether.campaign.v3 import cells
    from tether.theory import transmission as TR

    declared = _predictions()[cell]["transmission"]
    table = TR.predicted_pair_table(cells.response_spec(cells.cell_specs("A")[cell]), cell)
    assert table["band_statistic_median_offdiag"] == pytest.approx(declared["band_statistic_median_offdiag"], abs=1e-12)
    assert np.allclose(np.array(table["geometry_factor"], float), np.array(declared["geometry_factor"], float), atol=1e-12)


def test_forecast_is_insensitive_to_the_jacobian_step(monkeypatch):
    """Sec. V-A: steps of 1e-4 and 1e-6 change every R_ij by less than 1e-4 percent."""
    from tether.campaign.v3 import cells
    from tether.theory import reduced_lti as R
    from tether.theory import transmission as TR

    cell = "T600_k500_I0.5"
    spec = cells.response_spec(cells.cell_specs("A")[cell])

    def response():
        rows = TR.predicted_pair_table(spec, cell)["response_M1_half_sine"]
        return np.array([[np.nan if v is None else v for v in row] for row in rows], float)

    base = response()
    for step in (1e-4, 1e-6):
        monkeypatch.setattr(R, "STATE_STEP", step)
        rel = np.nanmax(np.abs(response() / base - 1.0))
        assert rel < 1e-6, (step, rel)


# ------------------------------------------------------------ pre-registration integrity (v3)
def test_pinned_files_match_the_declaration():
    """Every file hashed by the v3 declaration still has its declared SHA-256.

    Two documented exceptions: tether/campaign/v3/reducer.py carries the new hash recorded in addendum 1,
    and tether/campaign/fleet_run.py carries the FleetRunSpec overrides described in addenda 6 and 7.
    Pinned files absent from the checkout are skipped.
    """
    decl_path = REC / "cascade_declarations.json"
    if not decl_path.exists():
        pytest.skip("v3 declaration not present")
    from tether.campaign.v3 import campaign

    decl = json.loads(decl_path.read_text())
    add1 = json.loads((REC / "cascade_addendum_1.json").read_text())["change"]
    expected = {campaign.PLAN_PATH: decl["plan_sha256"], campaign.THEORY_PATH: decl["theory_sha256"],
                campaign.PREDICTIONS_PATH: decl["predictions_sha256"], campaign.V1_RECORD: decl["input_sha256"]["v1_record"],
                campaign.V1_CACHE: decl["input_sha256"]["v1_cache"], campaign.V2_PREDICTIONS: decl["input_sha256"]["v2_predictions"]}
    for name, path in campaign.MODULE_PATHS.items():
        expected[path] = decl["module_sha256"][name]
    expected[campaign.MODULE_PATHS["campaign/v3/reducer"]] = add1["new_sha256"]
    expected[campaign.MODULE_PATHS["campaign/fleet_run"]] = FLEET_RUN_SHA256_AFTER_ADDENDA_6_7
    checked = 0
    for path, sha in expected.items():
        if not Path(path).exists():
            continue
        assert _sha256(Path(path)) == sha, path
        checked += 1
    assert checked >= len(campaign.MODULE_PATHS)


def test_addenda_are_unedited():
    """Every addendum hash recorded by a v3 result file still matches the addendum on disk."""
    names = ("wp1_results.json", "wp1_sensitivity.json", "wp2_results.json", "wp3_results.json", "wp3_sensitivity.json")
    seen = 0
    for name in names:
        path = REC / name
        if not path.exists():
            continue
        for addendum, sha in json.loads(path.read_text()).get("addenda_sha256", {}).items():
            assert _sha256(REC / addendum) == sha, (name, addendum)
            seen += 1
    if seen == 0:
        pytest.skip("no v3 result files present")
