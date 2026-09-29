"""Phase 3 analysis: P3-T1..T9 from the committed shock table, predictions and cells."""

from __future__ import annotations

import json
import math

import numpy as np

from tether.campaign.common import config_sha256, json_bytes, npz_bytes, relative, sha256_file, source_state, write_bytes
from tether.campaign.phase3 import (
    ALPHA,
    CACHE_PATH,
    CLASSES,
    DRAGS,
    MANIFEST_PATH,
    POWER_MIN,
    PREDICTIONS_PATH,
    RECORD_PATH,
    REFERENCE,
    RESULTS_PATH,
    STAT_SEEDS,
    cell_name,
    cell_specs,
)
from tether.campaign.stationary import StationaryJob, load_cache, run_stationary_job
from tether.physics import constants
from tether.physics.fleet import formation_geometry
from tether.physics.weather import stationary_weather_forces
from tether.theory import gust as gusttheory

TAU_W = constants.WEATHER_TIME_CONSTANT
HILL_FRACTION = 0.001

DECLARATIONS = {
    "cells": "the plan lists weather class x drag x direction = 8 cells (its count of 16 cannot be derived from the listed factors); the 8 listed cells run at the Phase 2 reference cell (T0 = 1 kN, k_h = 500, intensity 1.0), 20 statistics seeds + 2 pilot seeds, 600 s",
    "broadside": "common broadside = a common-mode gust front pushing along -y (from port), one scalar t3 or Gaussian innovation per tick, area-scaled per body, total variance matched to the isotropic classes",
    "t1": "Hill estimator on the upper tail of W_rel computed from the cell's own weather (statistics seeds, 10 ms samples, equilibrium chord directions), top 0.1% of samples pooled over seeds per cable; asymptotic normal interval alpha (1 +/- 1.96/sqrt(k)), which ignores the serial dependence of the AR(1) samples and is therefore too narrow",
    "t2": "GPD by maximum likelihood on declustered taut q cluster maxima above the threshold chosen by evt.gpd.select_threshold; profile-likelihood interval",
    "t3": "tail index of the snap marks' T_peak: GPD shape xi above select_threshold (index 1/xi, primary), Hill at the same threshold and the log-log slope of the rate curve over its powered grid as cross-checks",
    "t4": "per slackening (direction, cable): slope of log first-return speed on log R over R in {4, 8, 16, 32}; spectral mass per body proportional to sigma_body^alpha (the plan says equal mass; the physical weights are used and equal-mass results reported beside them)",
    "t5": "crossover = threshold where the measured snap and taut rate curves cross on the level grid; prediction from the Theorem 7 quadrature",
    "t6": "measured fleet virtual rate (snap marks + declustered taut upcrossings) at powered thresholds against the Theorem 7 single-big-jump quadrature",
    "t8": "the plan gives no operational definition of the controller-invariant lead; reported as not evaluable",
    "t9": "the plant has no thruster saturation, so the saturation radius is infinite; contamination at T_b = share of snap exceedances from excursions whose dwell >= tau_w/4",
}


def _pool(summaries: list[dict]) -> dict:
    marks = {key: np.concatenate([s["marks"][key] for s in summaries]) for key in summaries[0]["marks"]}
    marks["seed"] = np.concatenate([np.full(s["marks"]["t_up"].size, s["meta"]["seed"]) for s in summaries])
    return {
        "marks": marks,
        "level_counts": np.sum([s["level_counts"] for s in summaries], axis=0),
        "exposures": np.array([s["meta"]["exposure"] for s in summaries]),
        "exposure": float(sum(s["meta"]["exposure"] for s in summaries)),
        "seeds": np.array([s["meta"]["seed"] for s in summaries]),
        "taut_peaks": {key: np.concatenate([s["taut_peaks"][key] for s in summaries]) for key in summaries[0]["taut_peaks"]},
        "closures": [s["meta"]["closure"] for s in summaries],
        "wall_seconds": float(sum(s["meta"]["wall_seconds"] for s in summaries)),
        "sim_seconds": float(sum(s["meta"]["sim_seconds"] for s in summaries)),
    }


def _gpd_index(values: np.ndarray) -> dict:
    from tether.evt.gpd import fit_gpd, profile_shape_interval, select_threshold

    values = np.asarray(values, dtype=float)
    if values.size < 60:
        return {"n": int(values.size), "status": "UNDER-POWERED"}
    threshold = select_threshold(values)
    fit = fit_gpd(values, threshold)
    lo, hi = profile_shape_interval(values, threshold)
    return {"n": int(values.size), "threshold": float(threshold), "exceedances": int(fit.n_exceedances), "shape": float(fit.shape),
            "shape_lo": float(lo), "shape_hi": float(hi), "index": 1.0 / fit.shape if fit.shape > 0 else float("inf"),
            "index_lo": 1.0 / hi if hi > 0 else float("inf"), "index_hi": 1.0 / lo if lo > 0 else float("inf")}


def _hill(values: np.ndarray, threshold: float | None, fraction: float | None = None) -> dict:
    from tether.evt.hill import hill_estimator

    values = np.sort(np.asarray(values, dtype=float))
    values = values[values > 0]
    if threshold is not None:
        k = int(np.sum(values > threshold))
    else:
        k = int(max(20, round(fraction * values.size)))
    if k < 20 or k >= values.size:
        return {"k": k, "status": "UNDER-POWERED"}
    alpha, lo, hi = hill_estimator(values, k)
    return {"k": k, "alpha": float(alpha), "lo": float(lo), "hi": float(hi)}


def _rate_curve(cell: dict, levels: np.ndarray) -> dict:
    marks = cell["marks"]
    snap = np.array([np.sum(marks["T_peak"] > level) for level in levels])
    from tether.campaign.summaries import LEVEL_GRID

    taut = np.array([cell["level_counts"][int(np.argmin(np.abs(LEVEL_GRID - level)))].sum() for level in levels])
    return {"levels": levels, "snap_count": snap, "taut_count": taut, "snap_rate": snap / cell["exposure"], "taut_rate": taut / cell["exposure"]}


def analyse() -> dict:
    predictions = json.loads(PREDICTIONS_PATH.read_text())
    jobs, summaries = load_cache(CACHE_PATH)
    stat: dict[str, list] = {}
    for job, summary in zip(jobs, summaries):
        if not job.pilot:
            stat.setdefault(job.cell, []).append(summary)
    cells = {name: _pool(items) for name, items in stat.items()}
    geometry = formation_geometry("parallel")
    directions = np.stack([np.cos(geometry.cable_angles), np.sin(geometry.cable_angles)], axis=1)
    specs = cell_specs()
    tests: dict = {}
    rows: dict = {}
    from tether.campaign.summaries import LEVEL_GRID

    for name, cell in cells.items():
        spec = specs[name]
        row = {"excursions": int(cell["marks"]["t_up"].size), "exposure": cell["exposure"], "closures": int(sum(c is not None for c in cell["closures"]))}
        row["snap_tail"] = _gpd_index(cell["marks"]["T_peak"])
        if "threshold" in row["snap_tail"]:
            row["snap_hill"] = _hill(cell["marks"]["T_peak"], row["snap_tail"]["threshold"])
        row["taut_tail"] = _gpd_index(cell["taut_peaks"]["peak"])
        curve = _rate_curve(cell, LEVEL_GRID)
        powered = curve["snap_count"] >= POWER_MIN
        if powered.sum() >= 3:
            x = np.log(LEVEL_GRID[powered])
            slope = float(np.polyfit(x, np.log(curve["snap_rate"][powered]), 1)[0])
            row["snap_rate_log_slope"] = slope
        crossing = None
        both = (curve["snap_count"] >= POWER_MIN) & (curve["taut_count"] >= POWER_MIN)
        diff = np.log(np.maximum(curve["snap_rate"], 1e-300)) - np.log(np.maximum(curve["taut_rate"], 1e-300))
        for i in range(LEVEL_GRID.size - 1):
            if (curve["snap_count"][i] > 0 and curve["taut_count"][i] > 0 and curve["snap_count"][i + 1] > 0 and curve["taut_count"][i + 1] > 0
                    and diff[i] > 0 >= diff[i + 1]):
                crossing = float(np.sqrt(LEVEL_GRID[i] * LEVEL_GRID[i + 1]))
                break
        row["crossover"] = crossing
        row["crossover_powered"] = bool(crossing is not None and both[int(np.argmin(np.abs(LEVEL_GRID - crossing)))])
        row["curve"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in curve.items()}
        if spec.weather_distribution == "student_t3":
            wrel = []
            for seed in STAT_SEEDS:
                forces = stationary_weather_forces(seed, DURATION_WEATHER, distribution=spec.weather_distribution,
                                                   direction=spec.weather_direction, scale=spec.weather_scale, front_angle=spec.weather_front_angle)
                wrel.append(gusttheory.relative_load_series(forces, directions))
            wrel = np.concatenate(wrel)
            row["wrel_hill"] = [_hill(wrel[:, c], None, HILL_FRACTION) for c in range(wrel.shape[1])]
        rows[name] = row

    t3_local = [rows[cell_name("student_t3", d, "local")] for d in DRAGS]
    hills = [h for h in rows[cell_name("student_t3", "linear", "local")].get("wrel_hill", []) if "alpha" in h]
    t1_pass = bool(hills) and all(2.4 <= h["alpha"] <= 3.6 for h in hills)
    tests["P3-T1"] = {"statement": "Hill index of W_rel under t3 in [2.4, 3.6]", "hill_per_cable_local_linear": hills,
                      "hill_front": rows[cell_name("student_t3", "linear", "front_broadside")].get("wrel_hill"),
                      "verdict": "PASS" if t1_pass else "FAIL"}
    taut = rows[cell_name("student_t3", "linear", "local")]["taut_tail"]
    t2_pass = "shape" in taut and 0.25 <= taut["shape"] <= 0.42
    tests["P3-T2"] = {"statement": "GPD shape of taut q-peaks under t3 in [0.25, 0.42] (linear drag); quadratic reported",
                      "linear": taut, "quadratic": rows[cell_name("student_t3", "quadratic", "local")]["taut_tail"],
                      "gaussian_linear_reference": rows[cell_name("gaussian", "linear", "local")]["taut_tail"],
                      "verdict": ("UNDER-POWERED" if "shape" not in taut else ("PASS" if t2_pass else "FAIL"))}
    t3_rows = []
    for drag, row in zip(DRAGS, t3_local):
        tail = row["snap_tail"]
        taut_index = row["taut_tail"].get("index")
        entry = {"drag": drag, "snap_tail": tail, "snap_hill": row.get("snap_hill"), "rate_log_slope": row.get("snap_rate_log_slope"), "taut_index": taut_index}
        if "index" in tail:
            entry["two_alpha_in_range"] = 4.5 <= tail["index"] <= 7.5
            entry["excludes_taut_index"] = taut_index is not None and not (tail["index_lo"] <= taut_index <= tail["index_hi"])
        t3_rows.append(entry)
    t3_ok = all(e.get("two_alpha_in_range") and e.get("excludes_taut_index") for e in t3_rows)
    tests["P3-T3"] = {"statement": "snap-rate tail index 2 alpha in [4.5, 7.5] under both drag laws, interval excluding alpha", "rows": t3_rows,
                      "verdict": ("UNDER-POWERED" if any("index" not in e["snap_tail"] for e in t3_rows) else ("PASS" if t3_ok else "FAIL"))}
    t4_rows = []
    for drag in DRAGS:
        scaling = predictions["drags"][drag]["shock_scaling"]
        total = sum(r["mass"] for r in scaling)
        ok_mass = sum(r["mass"] for r in scaling if r["beta"] is not None and 0.4 <= r["beta"] <= 0.6 and r["beta_8_16"] is not None and r["beta_16_32"] is not None and abs(r["beta_8_16"] - r["beta_16_32"]) < 0.1)
        equal_total = len(scaling)
        equal_ok = sum(1 for r in scaling if r["beta"] is not None and 0.4 <= r["beta"] <= 0.6 and r["beta_8_16"] is not None and r["beta_16_32"] is not None and abs(r["beta_8_16"] - r["beta_16_32"]) < 0.1)
        t4_rows.append({"drag": drag, "slackening_pairs": len(scaling), "mass_fraction_consistent": ok_mass / total if total else None,
                        "equal_mass_fraction_consistent": equal_ok / equal_total if equal_total else None,
                        "beta_median": float(np.median([r["beta"] for r in scaling if r["beta"] is not None])) if scaling else None})
    t4_pass = all(r["mass_fraction_consistent"] is not None and r["mass_fraction_consistent"] >= 0.8 for r in t4_rows)
    tests["P3-T4"] = {"statement": "shock-response slope beta in [0.4, 0.6] with local slopes within 0.1, for >= 80% of the slackening mass", "rows": t4_rows,
                      "verdict": ("UNDER-POWERED" if any(r["slackening_pairs"] == 0 for r in t4_rows) else ("PASS" if t4_pass else "FAIL"))}
    reference = rows[cell_name("student_t3", "linear", "local")]
    predicted_cross = predictions["drags"]["linear"]["predicted_crossover"]
    measured_cross = reference["crossover"]
    tests["P3-T5"] = {"statement": "snap and taut rate curves cross; crossover within 30% of prediction", "measured": measured_cross, "measured_powered": reference["crossover_powered"],
                      "predicted": predicted_cross,
                      "verdict": ("FAIL (no crossing)" if measured_cross is None else ("UNDER-POWERED" if predicted_cross is None else ("PASS" if abs(measured_cross / predicted_cross - 1) <= 0.3 else "FAIL")))}
    thresholds = np.array(predictions["thresholds"])
    fleet_pred = np.array(predictions["drags"]["linear"]["fleet_rate"])
    ratios = []
    curve = reference["curve"]
    for level, snap_count, taut_count in zip(curve["levels"], curve["snap_count"], curve["taut_count"]):
        count = snap_count + taut_count
        if count < POWER_MIN:
            continue
        predicted = float(np.interp(np.log(level), np.log(thresholds), fleet_pred, left=np.nan, right=np.nan))
        if predicted and predicted > 0 and np.isfinite(predicted):
            ratios.append({"threshold": level, "count": int(count), "measured": count / reference["exposure"], "predicted": predicted, "ratio": (count / reference["exposure"]) / predicted})
    t6_pass = bool(ratios) and all(0.5 <= r["ratio"] <= 2.0 for r in ratios)
    tests["P3-T6"] = {"statement": "measured virtual rate at powered T_b against the severance-radius quadrature within [1/2, 2]", "rows": ratios,
                      "verdict": "UNDER-POWERED" if not ratios else ("PASS" if t6_pass else "FAIL")}
    lin, quad = t3_local
    overlap = None
    if "index" in lin["snap_tail"] and "index" in quad["snap_tail"]:
        overlap = not (lin["snap_tail"]["index_hi"] < quad["snap_tail"]["index_lo"] or quad["snap_tail"]["index_hi"] < lin["snap_tail"]["index_lo"])
    tests["P3-T7"] = {"statement": "exponent intervals overlap between drag laws; coefficients differ", "exponents_overlap": overlap,
                      "linear_rate_at_8kN": float(np.interp(8000.0, lin["curve"]["levels"], lin["curve"]["snap_rate"])),
                      "quadratic_rate_at_8kN": float(np.interp(8000.0, quad["curve"]["levels"], quad["curve"]["snap_rate"])),
                      "verdict": "UNDER-POWERED" if overlap is None else ("PASS" if overlap else "FAIL")}
    tests["P3-T8"] = {"statement": "leverage certificate", "verdict": "NOT EVALUABLE", "reason": DECLARATIONS["t8"]}
    contamination = []
    for level in (4000.0, 8000.0, 12000.0, 16000.0, 25000.0):
        marks = reference_marks = cells[cell_name("student_t3", "linear", "local")]["marks"]
        hits = marks["T_peak"] > level
        share = float(np.mean(reference_marks["dwell"][hits] >= TAU_W / 4.0)) if hits.any() else None
        contamination.append({"threshold": level, "exceedances": int(hits.sum()), "contamination": share, "usable_downstream": share is not None and share < 0.2})
    tests["P3-T9"] = {"statement": "validity certificate: thruster saturation radius and contamination per T_b", "saturation_radius": "infinite (no thruster limit in the plant)",
                      "contamination": contamination, "verdict": "REPORTED"}
    gate_ok = tests["P3-T1"]["verdict"] == "PASS"
    gate = {"decision": "GO" if gate_ok else "NO-GO", "rule": "GO requires T1 (and T2 if the taut channel is active); NO-GO only on T1"}
    results = {"schema_version": 1, "source": source_state(), "declarations": DECLARATIONS, "predictions_sha256": sha256_file(PREDICTIONS_PATH),
               "configuration_sha256": config_sha256({k: v.__dict__ for k, v in specs.items()}), "rows": rows, "tests": tests, "gate": gate,
               "execution": {name: {"wall_seconds": c["wall_seconds"], "sim_seconds": c["sim_seconds"]} for name, c in cells.items()}}
    names = sorted(cells)
    arrays = {"cell_names": np.array(names), "marks_cell_index": np.concatenate([np.full(cells[n]["marks"]["t_up"].size, i) for i, n in enumerate(names)])}
    for key in cells[names[0]]["marks"]:
        arrays[f"marks_{key}"] = np.concatenate([cells[n]["marks"][key] for n in names])
    arrays["level_counts"] = np.stack([cells[n]["level_counts"] for n in names])
    results["record_sha256"] = write_bytes(RECORD_PATH, npz_bytes(arrays))
    probe_job, probe_summary = jobs[0], summaries[0]
    results["probe"] = {"cell": probe_job.cell, "seed": probe_job.seed, "marks_t_up": probe_summary["marks"]["t_up"].tolist()}
    write_bytes(RESULTS_PATH, json_bytes(results))
    write_bytes(MANIFEST_PATH, json_bytes({"schema_version": 1, "driver": "python -m tether.campaign.phase3 shocks|predict|compute|analyse",
                                           "predictions_sha256": results["predictions_sha256"], "record_sha256": results["record_sha256"],
                                           "committed_record_path": relative(RECORD_PATH), "probe": results["probe"]}))
    return results


DURATION_WEATHER = 600.0


def replay() -> bool:
    results = json.loads(RESULTS_PATH.read_text())
    probe = results["probe"]
    summary = run_stationary_job(StationaryJob(probe["cell"], cell_specs()[probe["cell"]], probe["seed"]))
    passed = summary["marks"]["t_up"].tolist() == probe["marks_t_up"]
    print(json.dumps({"marks_exact": passed}))
    return passed
