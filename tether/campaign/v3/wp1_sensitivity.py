"""WP1 sensitivity (plan v3, addendum 2): the no-snap floor of the drop statistic and the interventional swing.

    python -m tether.campaign.v3.wp1_sensitivity --workers 16      # records/v3/wp1_sensitivity.{npz,json}

Why.  The declared WP1 statistic is the neighbour's tension drop within one engagement period after a
parent re-engagement, over T_peak.  A neighbour's tension also drops within any 0.335 s window with no
snap at all (weather and vessel motion), so the declared statistic carries a pair-independent floor.
This module measures that floor and a floor-free estimator, on the pilot seeds (2001, 2002) of every
stage-A cell (cell-level medians) and on all 22 seeds of four cells (``FULL_CELLS``: the calibration cell,
its intensity-1.0 neighbour, one fan cell and the stiff k x 4 cell) for the pair matrices, and reports
both beside the declared results.  Nothing here re-scores a declared test.

Two statistics per eligible (parent, neighbour) row, eligibility exactly as WP1 (parent role, x in
[1, 4], every other cable taut at t_up):

* the time-shifted NULL: the same window statistic, drop / T_peak, at t_up + s for s in 7, 11, ..., 67 s,
  keeping shifts whose window starts with every cable taut and holds no re-engagement of any cable in
  [-1 s, +P]; the row's null is the median over its shifts; the floor-corrected drop is
  drop_over_Tpeak - null;
* the INTERVENTIONAL swing: the factual and counterfactual branches of the campaign's own intervention
  integrator (``integrate_state.integrate_with_state``, a verbatim copy of the pinned
  ``intervention.integrate`` whose outputs are asserted equal to it on every parent), from the row at or
  before t_up - 0.5 s to t_up + P; swing_ij = max over [t_up, t_up + P] of (T_i^cf - T_i^f) / T_peak on
  the integrator's 10 ms grid, with T = max(0, k e + c e_dot) while alive.  The factual branch's own drop
  (T_i^f(t_up) - min T_i^f) is recorded beside the record's drop as a grid check.

Records read: records/v3/cache/v3_stageA.pkl (the cell's cached marks: the re-simulation must reproduce
them bit-exactly), records/v3/cascade_predictions.json (R_ij, operator, for the pair comparison),
records/v3/cascade_declarations.json (hash into the output).  Written: records/v3/wp1_sensitivity.npz
(one row per (cell, seed, parent, neighbour)) and records/v3/wp1_sensitivity.json (per-cell medians with
seed intervals over the two pilot seeds, pooled pair matrices of the interventional swing and their rank
correlation with R and with the operator, provenance).
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time as wallclock

import numpy as np
from scipy.stats import spearmanr

from tether.campaign.common import DEFAULT_WORKERS, json_bytes, npz_bytes, run_pool, sha256_file, source_state, write_bytes
from tether.campaign.v3 import campaign, cells, compute
from tether.campaign.v3 import integrate_state as IS
from tether.campaign.v3 import intervention as IV
from tether.theory import transmission as TR

SHIFTS = np.arange(7.0, 70.0, 4.0)
PILOTS = (2001, 2002)
# cell-level medians on the pilot seeds of every stage-A cell; pair matrices need every seed of these four
FULL_CELLS = ("T600_k500_I0.5", "T600_k500_I1.0", "FAN_T600_I0.5", "T600_k500_I0.5_kx4")
LEAD_S = 0.5
OUT_NPZ = campaign.RECORD_DIR / "wp1_sensitivity.npz"
OUT_JSON = campaign.RECORD_DIR / "wp1_sensitivity.json"
FIELDS = ("parent_mark", "cable_j", "neighbour_i", "t_up", "T_peak", "x", "drop_over_Tpeak", "null_over_Tpeak", "n_shifts",
          "factual_drop_over_Tpeak", "swing_over_Tpeak", "swing_lag_s", "R_ij", "operator_ij")


def _tension_1ms(record: dict) -> np.ndarray:
    k, c = float(record["stiffness"]), float(record["damping"])
    e, r, alive = record["elongation"], record["rate"], np.asarray(record["alive"], bool)
    return np.where(alive & (e > 0.0), np.maximum(k * e + c * r, 0.0), 0.0)


def sensitivity_run(job: cells.V3Job) -> dict:
    """One pilot run: re-simulate, check the marks against the cache, compute both statistics per row."""
    from tether.campaign import summaries as v1
    from tether.campaign.fleet_run import build_run, run_to_end
    from tether.campaign.v2 import events as ev
    from tether.campaign.v2 import summaries as v2
    from tether.campaign.v3 import reducer

    started = wallclock.perf_counter()
    run = run_to_end(build_run(job.spec, job.seed))
    marks = v1.summarize_run(run, job.spec.warmup, job.spec.pretension)["marks"]
    record = v2.run_record(run)
    same = (np.array_equal(np.asarray(marks["t_up"]), job.expected_t_up) and np.array_equal(np.asarray(marks["T_peak"]), job.expected_T_peak)) if job.expected_t_up is not None else None
    k, c = float(record["stiffness"]), float(record["damping"])
    n = record["elongation"].shape[1]
    P = TR.frequencies(k, c, n)["engagement_period"]
    W = int(round(P / 1e-3))
    T = _tension_1ms(record)
    t = np.asarray(record["event_time"], float)
    t_up = np.asarray(marks["t_up"], float); cable = np.asarray(marks["cable"], int); peak = np.asarray(marks["T_peak"], float)
    dec = ev.decluster_marks(t_up, cable, rule="anchored")
    x = peak / job.spec.pretension
    end = float(record["sim_end"]) if record["closure"] is None else min(float(record["sim_end"]), float(record["closure"][1]))
    _, downs = ev.crossings(record["event_time"], record["elongation"], np.asarray(record["alive"], bool))
    table = reducer.offspring_table(record, marks, dec, job.spec.warmup, end, downs=downs)
    events = table["events"]
    thrusts = np.asarray(run.fleet.operating.thrusts, float)
    spec = IV.InterventionSpec(horizon=P + 0.05, window=P)
    pred = json.loads(campaign.PREDICTIONS_PATH.read_text())["cells"][job.cell]["transmission"]
    R = np.array([[np.nan if v is None else v for v in row] for row in pred["response_M1_half_sine"]], float)
    OP = np.array([[np.nan if v is None else v for v in row] for row in pred["operator"]], float)
    rows = {f: [] for f in FIELDS}
    copy_ok, n_parents = True, 0
    for p in np.flatnonzero((dec.role == 0) & (x >= 1.0) & (x <= 4.0)):
        k0 = int(np.searchsorted(t, t_up[p]))
        if k0 + W + 1 > t.size:
            continue
        j = int(cable[p])
        if any(T[k0, i] <= 0.0 for i in range(n) if i != j):
            continue
        # the event index of this parent mark in the offspring table (events are heads = parent marks)
        e_idx = int(np.flatnonzero(np.abs(np.asarray(events["t_up"]) - t_up[p]) < 1e-9)[0]) if np.any(np.abs(np.asarray(events["t_up"]) - t_up[p]) < 1e-9) else None
        xw = IV.parent_window(record, events, e_idx, thrusts, spec, job.spec.pretension) if e_idx is not None else None
        swing = np.full(n, np.nan); lag = np.full(n, np.nan); fdrop = np.full(n, np.nan)
        if xw is not None:
            f_ref = IV.integrate(xw, None, "recorded", window=P, peak_window=1.0)
            c_ref = IV.integrate(xw, j, "recorded", window=P, peak_window=1.0)
            f = IS.integrate_with_state(xw, None, window=P, peak_window=1.0)
            g = IS.integrate_with_state(xw, j, window=P, peak_window=1.0)
            copy_ok = copy_ok and IS.same_branch(f_ref, f) and IS.same_branch(c_ref, g)
            Tf = IS.tension(f["e_10ms"], f["rates"], f["alive"], k, c)
            Tc = IS.tension(g["e_10ms"], g["rates"], g["alive"], k, c)
            sel = (f["t_10ms"] >= t_up[p] - 1e-9) & (f["t_10ms"] <= t_up[p] + P + 1e-9)
            k_up = int(np.flatnonzero(sel)[0]) if sel.any() else None
            if k_up is not None:
                diff = Tc[sel] - Tf[sel]
                swing = diff.max(axis=0) / peak[p]
                lag = f["t_10ms"][sel][diff.argmax(axis=0)] - t_up[p]
                fdrop = (Tf[k_up] - Tf[sel].min(axis=0)) / peak[p]
            n_parents += 1
        for i in range(n):
            if i == j:
                continue
            drop = (T[k0, i] - T[k0:k0 + W + 1, i].min()) / peak[p]
            nulls = []
            for s in SHIFTS:
                ks = k0 + int(round(s / 1e-3))
                if ks + W + 1 > t.size or np.any(T[ks, :] <= 0.0) or np.any((t_up > t[ks] - 1.0) & (t_up < t[ks] + P)):
                    continue
                nulls.append((T[ks, i] - T[ks:ks + W + 1, i].min()) / peak[p])
            values = (int(p), j, i, float(t_up[p]), float(peak[p]), float(x[p]), float(drop), float(np.median(nulls)) if nulls else math.nan, len(nulls),
                      float(fdrop[i]), float(swing[i]), float(lag[i]), float(R[i, j]), float(OP[i, j]))
            for fld, v in zip(FIELDS, values):
                rows[fld].append(v)
    return {"cell": job.cell, "seed": int(job.seed), "marks_reproduced": same, "copy_equals_integrator": bool(copy_ok), "n_parents": n_parents,
            "period": P, "rows": {f: np.asarray(v) for f, v in rows.items()}, "wall_seconds": wallclock.perf_counter() - started}


def _pooled_pairs(rows: dict, mask: np.ndarray, key: str, min_parents: int = 5):
    """Mirror-pooled ((i, j) ~ (4 - i, 4 - j)) medians of ``key`` per ordered pair with at least ``min_parents`` rows."""
    out = {}
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            a = (i, j) if (i, j) <= (4 - i, 4 - j) else (4 - i, 4 - j)
            sel = mask & (((rows["neighbour_i"] == i) & (rows["cable_j"] == j)) | ((rows["neighbour_i"] == 4 - i) & (rows["cable_j"] == 4 - j)))
            if a not in out:
                out[a] = sel
    pairs, med, R, OP = [], [], [], []
    for (i, j), sel in sorted(out.items()):
        v = rows[key][sel]
        v = v[np.isfinite(v)]
        if v.size >= min_parents:
            pairs.append([i, j]); med.append(float(np.median(v)))
            R.append(float(np.nanmedian(rows["R_ij"][sel]))); OP.append(float(np.nanmedian(rows["operator_ij"][sel])))
    return pairs, med, R, OP


def analyse(results: list[dict]) -> dict:
    cells_out, pooled = {}, {f: [] for f in FIELDS}
    pooled["cell_index"], pooled["seed"] = [], []
    names = sorted({r["cell"] for r in results})
    for name in names:
        runs = [r for r in results if r["cell"] == name]
        rows = {f: np.concatenate([r["rows"][f] for r in runs]) for f in FIELDS}
        seeds = np.concatenate([np.full(r["rows"]["parent_mark"].size, r["seed"]) for r in runs])
        for f in FIELDS:
            pooled[f].append(rows[f])
        pooled["cell_index"].append(np.full(seeds.size, names.index(name))); pooled["seed"].append(seeds)
        ok = np.isfinite(rows["null_over_Tpeak"])
        okw = np.isfinite(rows["swing_over_Tpeak"])
        per_seed = {}
        for r in runs:
            rr = r["rows"]
            m = np.isfinite(rr["null_over_Tpeak"]); mw = np.isfinite(rr["swing_over_Tpeak"])
            per_seed[str(r["seed"])] = {"drop": float(np.median(rr["drop_over_Tpeak"])) if rr["drop_over_Tpeak"].size else math.nan,
                                        "null": float(np.median(rr["null_over_Tpeak"][m])) if m.any() else math.nan,
                                        "corrected": float(np.median(rr["drop_over_Tpeak"][m] - rr["null_over_Tpeak"][m])) if m.any() else math.nan,
                                        "swing": float(np.median(rr["swing_over_Tpeak"][mw])) if mw.any() else math.nan, "rows": int(rr["parent_mark"].size)}
        pairs, med, Rp, OPp = _pooled_pairs(rows, okw, "swing_over_Tpeak")
        rho_R = float(spearmanr(med, Rp).statistic) if len(pairs) >= 3 else math.nan
        rho_OP = float(spearmanr(med, OPp).statistic) if len(pairs) >= 3 else math.nan
        pairs_d, med_d, Rp_d, _ = _pooled_pairs(rows, np.ones(rows["parent_mark"].size, bool), "drop_over_Tpeak")
        cells_out[name] = {
            "rows": int(rows["parent_mark"].size), "parents": int(sum(r["n_parents"] for r in runs)), "seeds": [r["seed"] for r in runs],
            "marks_reproduced": all(r["marks_reproduced"] for r in runs), "copy_equals_integrator": all(r["copy_equals_integrator"] for r in runs),
            "median_drop_over_Tpeak": float(np.median(rows["drop_over_Tpeak"])) if rows["drop_over_Tpeak"].size else math.nan,
            "median_null_over_Tpeak": float(np.median(rows["null_over_Tpeak"][ok])) if ok.any() else math.nan,
            "median_corrected_drop": float(np.median(rows["drop_over_Tpeak"][ok] - rows["null_over_Tpeak"][ok])) if ok.any() else math.nan,
            "share_rows_above_null": float(np.mean(rows["drop_over_Tpeak"][ok] > rows["null_over_Tpeak"][ok])) if ok.any() else math.nan,
            "median_swing_over_Tpeak": float(np.median(rows["swing_over_Tpeak"][okw])) if okw.any() else math.nan,
            "median_swing_lag_s": float(np.median(rows["swing_lag_s"][okw])) if okw.any() else math.nan,
            "median_factual_drop_over_Tpeak": float(np.median(rows["factual_drop_over_Tpeak"][okw])) if okw.any() else math.nan,
            "median_R_ij_of_rows": float(np.nanmedian(rows["R_ij"])) if rows["R_ij"].size else math.nan,
            "swing_over_R_median": float(np.median(rows["swing_over_Tpeak"][okw] / rows["R_ij"][okw])) if okw.any() else math.nan,
            "per_seed": per_seed,
            "swing_pairs": {"pairs": pairs, "median_swing": med, "R_pooled": Rp, "operator_pooled": OPp, "spearman_vs_R": rho_R, "spearman_vs_operator": rho_OP},
            "drop_pairs": {"pairs": pairs_d, "median_drop": med_d, "R_pooled": Rp_d, "spearman_vs_R": float(spearmanr(med_d, Rp_d).statistic) if len(pairs_d) >= 3 else math.nan},
        }
    npz = {f: np.concatenate(v) for f, v in pooled.items()}
    npz["cell_names"] = np.array(names)
    summary = {
        "schema": "v3-wp1-sensitivity-1", "status": "SENSITIVITY (addendum 2): reported beside the declared WP1 tests, never scored",
        "definitions": {"null": "median over shifts s in 7..67 s step 4 of the drop statistic at t_up + s (window starts with every cable taut; no re-engagement in [-1 s, +P])",
                        "corrected": "drop_over_Tpeak - null_over_Tpeak per row", "swing": "max over [t_up, t_up + P] of (T_i^counterfactual - T_i^factual) / T_peak on the integrator's 10 ms grid",
                        "eligibility": "WP1's: parent role, x in [1, 4], every other cable taut at t_up", "pilot_seeds": list(PILOTS), "full_cells": list(FULL_CELLS)},
        "cells": cells_out,
        "pooled": {"median_drop": float(np.median(npz["drop_over_Tpeak"])), "median_null": float(np.nanmedian(npz["null_over_Tpeak"])),
                   "median_corrected": float(np.nanmedian(npz["drop_over_Tpeak"] - npz["null_over_Tpeak"])), "median_swing": float(np.nanmedian(npz["swing_over_Tpeak"])),
                   "median_swing_over_R": float(np.nanmedian(npz["swing_over_Tpeak"] / npz["R_ij"])), "rows": int(npz["parent_mark"].size)},
        "checks": {"marks_reproduced_all_runs": all(r["marks_reproduced"] for r in results), "copy_equals_integrator_all_runs": all(r["copy_equals_integrator"] for r in results)},
        "compute": {"runs": len(results), "core_hours": sum(r["wall_seconds"] for r in results) / 3600.0},
        "declarations_sha256": sha256_file(campaign.DECLARATIONS_PATH), "predictions_sha256": sha256_file(campaign.PREDICTIONS_PATH),
        "addenda_sha256": {p.name: sha256_file(p) for p in sorted(campaign.RECORD_DIR.glob("cascade_addendum_*.json"))},
        "stage_a_cache_sha256": sha256_file(compute.STAGE_CACHE["A"]), "source": source_state(),
    }
    return {"npz": npz, "json": summary}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--cells", nargs="*", help="development: restrict to these cells")
    args = ap.parse_args(argv)
    campaign.verify_declared()
    if not campaign.addendum_path(2).exists():
        raise SystemExit("addendum 2 (records/v3/cascade_addendum_2.json) must be written before this sensitivity runs")
    jobs = [j for j in cells.all_jobs("A") if (j.seed in PILOTS or j.cell in FULL_CELLS) and (not args.cells or j.cell in args.cells)]
    # expected marks come from the stage-A cache (every reused and new cell alike)
    light, summaries = compute.load_stage("A")
    cached = {(l["cell"], int(l["seed"])): s for l, s in zip(light, summaries)}
    fixed = []
    for j in jobs:
        s = cached[(j.cell, j.seed)]
        fixed.append(cells.V3Job(**{**j.__dict__, "expected_t_up": np.asarray(s["marks"]["t_up"]), "expected_T_peak": np.asarray(s["marks"]["T_peak"])}))
    results = run_pool(sensitivity_run, fixed, args.workers)
    out = analyse(results)
    write_bytes(OUT_NPZ, npz_bytes(out["npz"]))
    write_bytes(OUT_JSON, json_bytes(out["json"]))
    print(json.dumps({"pooled": out["json"]["pooled"], "checks": out["json"]["checks"], "compute": out["json"]["compute"]}))


if __name__ == "__main__":
    main()
