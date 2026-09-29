"""Plan v3 WP2 analysis: offspring kernel, isotonic transfer, intervention, and the kernel path.

Scores T2.1-T2.5 from the stage-A cache exactly as declared (records/v3/cascade_declarations.json):

* events, onsets and offspring come from the reducer (anchored 2 s events, 3 s any-cable parents);
* m_x is fitted isotonically on the calibration cell's PILOT seeds only (the one fitted curve);
* the margin law is the pilot seeds' clean-set tension quantiles per cell (the kernel's only
  fitted input); the kernel's peak average runs over the statistics-seed events;
* the time-shift null of tether/analysis/v2/paperfig/fig_attribution.py (200 circular shifts,
  rng 20260914) is applied to the sampled parents for T2.4;
* the kernel path follows the declared rule: R (T1.3 PASS) -> R_hat (T1.2 PASS) -> empirical.
"""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.optimize import isotonic_regression
from scipy.stats import spearmanr

from tether.campaign.common import json_bytes, npz_bytes, sha256_file, source_state, write_bytes
from tether.campaign.v3 import campaign
from tether.campaign.v3.analyse import addenda_sha256 as ADDENDA, cells, compute
from tether.campaign.v3 import analyse as A
from tether.campaign.v3 import intervention as IV
from tether.theory import branching as B
from tether.theory import transmission as TR

N_SHIFTS, NULL_SEED, SPAN_START = 200, 20260914, 20.0
BINS = campaign.COUPLED_PEAK_BINS


# ----------------------------------------------------------------------------- per-cell tables


def _events_of(runs: list[dict], pilot: bool) -> dict[str, np.ndarray]:
    keys = ("cable_j", "t_up", "T_peak", "n_cross", "offspring_per_cable", "primary", "first_lag", "n_marks")
    picked = [r for r in runs if r["meta"]["pilot"] == pilot]
    out: dict[str, list] = {k: [] for k in keys}
    out["seed"] = []
    for r in picked:
        e = r["events"]
        n = int(np.asarray(e["cable_j"]).size)
        for k in keys:
            out[k].append(np.asarray(e[k]))
        out["seed"].append(np.full(n, int(r["meta"]["seed"]), dtype=np.int64))
    return {k: (np.concatenate(v) if v else np.zeros((0, 5) if k == "offspring_per_cable" else 0)) for k, v in out.items()}


def binned_mx(x: np.ndarray, n_cross: np.ndarray, bins=BINS, min_events: int = 10) -> dict:
    edges = (-np.inf,) + tuple(bins) + (np.inf,)
    labels, means, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        labels.append(f"[{lo}, {hi})")
        counts.append(int(mask.sum()))
        means.append(float(n_cross[mask].mean()) if mask.sum() >= min_events else math.nan)
    order = [k for k, m in enumerate(means) if np.isfinite(m)]
    rho = float(spearmanr(order, [means[k] for k in order]).statistic) if len(order) >= 3 and np.std([means[k] for k in order]) > 0 else math.nan
    first = next((m for m in means if np.isfinite(m)), math.nan)
    return {"labels": labels, "counts": counts, "means": means, "spearman": rho, "first_bin_mean": first}


def fit_isotonic(x: np.ndarray, n_cross: np.ndarray) -> dict:
    order = np.argsort(x, kind="stable")
    xs, ys = np.asarray(x, dtype=float)[order], np.asarray(n_cross, dtype=float)[order]
    fit = isotonic_regression(ys, increasing=True)
    return {"x": xs.tolist(), "y": fit.x.tolist(), "n": int(xs.size)}


def predict_isotonic(fit: dict, x: np.ndarray) -> np.ndarray:
    xs, ys = np.asarray(fit["x"]), np.asarray(fit["y"])
    if xs.size == 0:
        return np.zeros(np.asarray(x).shape)
    return np.interp(np.asarray(x, dtype=float), xs, ys, left=float(ys[0]), right=float(ys[-1]))


def predict_isotonic_step(fit: dict, x: np.ndarray) -> np.ndarray:
    """Right-continuous step reading of the isotonic fit (audit round 2, M4: the interpolation convention was undeclared;
    the declared predictor is ``predict_isotonic``'s linear interpolation, this one is reported as a sensitivity)."""
    xs, ys = np.asarray(fit["x"]), np.asarray(fit["y"])
    if xs.size == 0:
        return np.zeros(np.asarray(x).shape)
    k = np.searchsorted(xs, np.asarray(x, dtype=float), side="right") - 1
    return ys[np.clip(k, 0, xs.size - 1)]


def taut_neighbour_target(runs: list[dict], ratio: np.ndarray, quantiles: np.ndarray) -> dict:
    """Audit round 2, M3: the count the mechanistic kernel models -- distinct neighbours that were taut at t_up and went
    slack within 3 s of a parent-role mark (statistics seeds) -- against the kernel summed over the same rows."""
    r = np.nan_to_num(np.asarray(ratio, dtype=float), nan=0.0)
    cdfs = [B.margin_cdf_from_quantiles(quantiles[i]) for i in range(5)]
    target = predicted = 0.0
    n_rows = 0
    for run in runs:
        if run["meta"]["pilot"]:
            continue
        t = run["transmission"]
        if t["parent_mark"].size == 0:
            continue
        rows = (t["role"] == 0) & (t["T_i_at_t_up"] > 0.0)          # event heads x neighbours taut at t_up
        i, j = t["neighbour_i"][rows].astype(int), t["cable_j"][rows].astype(int)
        e = t["event_id"][rows].astype(int)
        per_cable = np.asarray(run["events"]["offspring_per_cable"])   # attributed by the declared latest-parent rule
        target += float(np.sum(per_cable[e, i] >= 1))                 # distinct (event, neighbour) pairs unloaded
        predicted += float(sum(cdfs[ii](r[ii, jj] * tp) for ii, jj, tp in zip(i, j, t["T_peak"][rows])))
        n_rows += int(rows.sum())
    ratio_value = predicted / target if target > 0 else math.nan
    return {"rows": n_rows, "target": target, "predicted": predicted, "ratio": ratio_value,
            "in_band": bool(1.0 / campaign.FACTOR <= ratio_value <= campaign.FACTOR) if np.isfinite(ratio_value) else None}


def pooled_margin_quantiles(runs: list[dict], pilot: bool = True, levels: int = 101) -> np.ndarray:
    """Merge the per-run quantile vectors (each of equal mass) into one per-cable quantile vector."""
    picked = [r["margin"]["quantiles"] for r in runs if r["meta"]["pilot"] == pilot]
    picked = [q for q in picked if np.all(np.isfinite(q))]
    if not picked:
        return np.full((5, levels), np.nan)
    grid = np.linspace(0.0, 100.0, levels)
    return np.stack([np.percentile(np.concatenate([q[i] for q in picked]), grid) for i in range(5)])


def kernel_prediction(ratio: np.ndarray, quantiles: np.ndarray, events: dict, pulse_factor: float = 1.0) -> dict:
    """Per-event expected cross-cable offspring under K_ij(T) and the peak-averaged K."""
    r = np.nan_to_num(np.asarray(ratio, dtype=float), nan=0.0)
    peaks, parents = np.asarray(events["T_peak"], dtype=float), np.asarray(events["cable_j"], dtype=np.int64)
    cdfs = [B.margin_cdf_from_quantiles(quantiles[i]) for i in range(5)]
    per_event = np.zeros((peaks.size, 5))
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            rows = parents == j
            if rows.any():
                per_event[rows, i] = B.transmission_probability(r[i, j], cdfs[i], peaks[rows], pulse_factor)
    kernel = np.nan_to_num(B.kernel_from_margin_law(r, quantiles, peaks, parents, pulse_factor), nan=0.0)
    return {"per_event": per_event, "kernel": kernel}


def time_shift_excess(runs: list[dict], sampled: dict[int, np.ndarray], window: float = 3.0, n_shifts: int = N_SHIFTS) -> dict:
    """Measured minus null cross-cable onsets after each sampled parent (circular shift of the parent time)."""
    rng = np.random.default_rng(NULL_SEED)
    measured = null = 0.0
    n = 0
    for r in runs:
        seed = int(r["meta"]["seed"])
        if seed not in sampled:
            continue
        onsets_t, onsets_c = r["onsets"]["time"], r["onsets"]["cable"]
        span_end = float(r["meta"]["end"])
        e = r["events"]
        for index in sampled[seed]:
            t_up, cable = float(e["t_up"][index]), int(e["cable_j"][index])
            other = onsets_t[onsets_c != cable]
            measured += float(np.sum((other > t_up) & (other <= t_up + window)))
            shifts = rng.uniform(0.0, span_end - SPAN_START, size=n_shifts)
            starts = SPAN_START + (t_up - SPAN_START + shifts) % (span_end - SPAN_START)
            counts = [np.sum((other > s) & (other <= s + window)) + np.sum((other > s - (span_end - SPAN_START)) & (other <= s + window - (span_end - SPAN_START))) for s in starts]
            null += float(np.mean(counts))
            n += 1
    return {"parents": n, "measured": measured, "null": null, "excess": measured - null}


# ----------------------------------------------------------------------------- the analysis


def wp2() -> dict:
    jobs, summaries = A._load("A")
    predictions = json.loads(campaign.PREDICTIONS_PATH.read_text())
    wp1 = json.loads((campaign.RECORD_DIR / "wp1_results.json").read_text())
    by_cell = A._by_cell(jobs, summaries)
    factor, share = campaign.FACTOR, campaign.SHARE
    # ---- T2.1 and the isotonic fit (calibration pilots)
    calib_pilots = _events_of(by_cell[cells.CALIBRATION_CELL], pilot=True)
    t0_calib = float(cells.parse_cell(cells.CALIBRATION_CELL)["pretension"])
    fit = fit_isotonic(calib_pilots["T_peak"] / t0_calib, calib_pilots["n_cross"])
    per_cell: dict[str, dict] = {}
    t21, t22, t25 = {}, {}, {}
    kernel_paths = {}
    intervention_rows = []
    for name, runs in by_cell.items():
        pretension = float(cells.parse_cell(name)["pretension"])
        stat = _events_of(runs, pilot=False)
        x = stat["T_peak"] / pretension
        measured = float(stat["n_cross"].sum())
        scored = measured >= campaign.MIN_OFFSPRING
        b = binned_mx(x, stat["n_cross"])
        t21[name] = ("PASS" if b["spearman"] >= 0.8 and b["first_bin_mean"] <= 0.1 else "FAIL") if scored and np.isfinite(b["spearman"]) else "UNSCORED"
        predicted_iso = float(predict_isotonic(fit, x).sum())
        ratio_iso = predicted_iso / measured if measured > 0 else math.nan
        t22[name] = ("PASS" if 1.0 / factor <= ratio_iso <= factor else "FAIL") if scored and np.isfinite(ratio_iso) else "UNSCORED"
        predicted_step = float(predict_isotonic_step(fit, x).sum())
        ratio_step = predicted_step / measured if measured > 0 else math.nan
        t22_step = ("PASS" if 1.0 / factor <= ratio_step <= factor else "FAIL") if scored and np.isfinite(ratio_step) else "UNSCORED"
        # ---- kernels: R (exact), R_hat (measured), empirical; margin law from the cell's pilots
        quantiles = pooled_margin_quantiles(runs, pilot=True)
        r_exact = A._matrix(predictions["cells"][name]["transmission"]["response_M1_half_sine"])
        r_hat = _measured_ratio_matrix(wp1["cells"].get(name, {}))
        pilots = _events_of(runs, pilot=True)
        k_emp = np.nan_to_num(B.empirical_kernel(pilots["cable_j"], pilots["offspring_per_cable"]), nan=0.0) if pilots["cable_j"].size else np.zeros((5, 5))
        kernels = {}
        if np.all(np.isfinite(quantiles)) and stat["T_peak"].size:
            for label, ratio in (("R", r_exact), ("R_hat", r_hat)):
                if ratio is None:
                    continue
                for phi_label, phi in (("half_sine", 1.0), ("rectangular", math.pi / 2)):
                    kp = kernel_prediction(ratio, quantiles, stat, phi)
                    pred = float(kp["per_event"].sum())
                    kernels[f"{label}:{phi_label}"] = {"predicted": pred, "ratio": pred / measured if measured > 0 else math.nan, "K": kp["kernel"].tolist(),
                                                       "rho": B.spectral_radius(kp["kernel"])}
        taut_target = taut_neighbour_target(runs, r_exact, quantiles) if np.all(np.isfinite(quantiles)) else None
        pred_emp = float(sum(k_emp[:, j].sum() for j in stat["cable_j"])) if stat["cable_j"].size else 0.0
        kernels["empirical"] = {"predicted": pred_emp, "ratio": pred_emp / measured if measured > 0 else math.nan, "K": k_emp.tolist(), "rho": B.spectral_radius(k_emp)}
        t25[name] = ("PASS" if 1.0 / factor <= kernels["R:half_sine"]["ratio"] <= factor else "FAIL") if scored and "R:half_sine" in kernels and np.isfinite(kernels["R:half_sine"]["ratio"]) else "UNSCORED"
        rows = []
        for r in runs:
            for row in r.get("intervention", {}).get("rows", []):
                row["cell"], row["seed"] = name, int(r["meta"]["seed"])
                rows.append(row)
        intervention_rows.extend(rows)
        sampled = {}
        for r in runs:
            ids = [row["event_id"] for row in r.get("intervention", {}).get("rows", [])]
            if ids:
                sampled[int(r["meta"]["seed"])] = np.array(ids)
        shift = time_shift_excess(runs, sampled)
        causal = float(sum(int(np.sum(row["causal"])) for row in rows))
        causal_held = float(sum(int(np.sum(row.get("causal_held", 0))) for row in rows))
        net = prevented = measured_cross_sampled = 0.0
        for row in rows:
            f, cf = np.asarray(row["factual_onsets"], dtype=float), np.asarray(row["counterfactual_onsets"], dtype=float)
            other = np.arange(f.size) != int(row["cable"])
            net += float(np.sum((f - cf)[other]))
            prevented += float(np.sum(np.maximum(0.0, cf - f)[other]))
            measured_cross_sampled += float(np.sum(row["measured_cross"]))
        per_cell[name] = {
            "n_events_statistics": int(stat["T_peak"].size), "n_events_pilot": int(pilots["T_peak"].size), "measured_cross_offspring": measured, "scored": scored,
            "T2.1": {**b, "verdict": t21[name]},
            "T2.2": {"predicted": predicted_iso, "ratio": ratio_iso, "verdict": t22[name], "formation": cells.parse_cell(name)["formation"],
                     # addendum 3: the interpolation convention of the isotonic predictor was undeclared; declared = linear (above)
                     "sensitivity_step_convention": {"predicted": predicted_step, "ratio": ratio_step, "verdict": t22_step}},
            "margin_quantiles_pilot": None if not np.all(np.isfinite(quantiles)) else quantiles.tolist(),  # unrounded: figures re-derive K from them
            "kernels": kernels,
            "T2.5": {"verdict": t25[name], "ratio": kernels.get("R:half_sine", {}).get("ratio"),
                     # addendum 3: the count the kernel models (distinct taut neighbours unloaded within 3 s) beside the declared target
                     "sensitivity_taut_neighbour_target": taut_target},
            "intervention": {"parents": len(rows), "causal": causal, "causal_held": causal_held, "time_shift": shift,
                             "causal_over_excess": causal / shift["excess"] if shift["excess"] > 0 else math.nan,
                             # addendum 3: 'causal' is the positive part per (parent, cable) of factual - counterfactual (declared at
                             # module level); the signed net, the prevented onsets and the declustered offspring of the same parents
                             "net_causal": net, "prevented_onsets": prevented, "measured_cross_sampled": measured_cross_sampled,
                             "causal_over_measured_cross": causal / measured_cross_sampled if measured_cross_sampled > 0 else math.nan,
                             "net_over_measured_cross": net / measured_cross_sampled if measured_cross_sampled > 0 else math.nan},
        }
    # ---- T2.3 and T2.4
    validation = IV.validate(intervention_rows) if intervention_rows else {"passed": False, "n": 0}
    t23 = {"verdict": "PASS" if validation.get("passed") else ("FAIL" if intervention_rows else "UNDER-POWERED"), **validation}
    if intervention_rows:   # addendum 3: the same rule on parents with x >= 1, and without the k x 4 cell
        t23["sensitivity_x_ge_1"] = IV.validate([r for r in intervention_rows if r["x"] >= 1.0])
        t23["sensitivity_without_kx4"] = IV.validate([r for r in intervention_rows if not r["cell"].endswith("_kx4")])
    t24_cells = {}
    for name, c in per_cell.items():
        iv = c["intervention"]
        if not validation.get("passed"):
            t24_cells[name] = "UNSCORED"
        elif iv["causal"] < campaign.MIN_OFFSPRING or not np.isfinite(iv["causal_over_excess"]):
            t24_cells[name] = "UNSCORED"
        else:
            t24_cells[name] = "PASS" if 1.0 / factor <= iv["causal_over_excess"] <= factor else "FAIL"
    tests = {
        "T2.1": A._share_verdict(t21), "T2.2": A._share_verdict(t22), "T2.3": t23,
        "T2.4": A._share_verdict(t24_cells) if validation.get("passed") else {"verdict": "UNSCORED", "reason": "T2.3 failed", "per_cell": t24_cells},
        "T2.5": A._share_verdict(t25),
    }
    fan = {n: v for n, v in t22.items() if n.startswith("FAN_")}
    tests["T2.2"]["fan_subgroup"] = A._share_verdict(fan, min_scored=1) if fan else None
    # ---- kernel path (declared rule)
    wp1_tests = wp1["tests"]
    if wp1_tests["T1.3"]["verdict"] == "PASS":
        first, label = "R", "R (exact response)"
    elif wp1_tests["T1.2"]["verdict"] == "PASS":
        first, label = "R_hat", "R_hat (measured)"
    else:
        first, label = "empirical", "empirical"
    order = [k for k in (first, "R", "R_hat", "empirical") if k in (first, "R", "R_hat", "empirical")]
    chosen = None
    path_tests = {}
    for candidate in dict.fromkeys(order):
        key = "empirical" if candidate == "empirical" else f"{candidate}:half_sine"
        verdicts = {n: (("PASS" if 1.0 / factor <= c["kernels"][key]["ratio"] <= factor else "FAIL") if c["scored"] and key in c["kernels"] and np.isfinite(c["kernels"][key]["ratio"]) else "UNSCORED") for n, c in per_cell.items()}
        path_tests[candidate] = A._share_verdict(verdicts)
        if path_tests[candidate]["verdict"] == "PASS" and chosen is None:
            chosen = candidate
    results = {
        "schema": "v3-wp2-results-1", "tests": tests, "cells": per_cell,
        "isotonic_fit": {"cell": cells.CALIBRATION_CELL, "seeds": list(campaign.phase2.PILOT_SEEDS), **fit},
        "kernel_path": {"first_by_wp1": label, "tested": path_tests, "chosen": chosen},
        "K_path_chosen": chosen,
        "declarations_sha256": sha256_file(campaign.DECLARATIONS_PATH), "predictions_sha256": sha256_file(campaign.PREDICTIONS_PATH), "addenda_sha256": ADDENDA(),
        "wp1_results_sha256": sha256_file(campaign.RECORD_DIR / "wp1_results.json"), "source": source_state(),
    }
    write_bytes(campaign.RECORD_DIR / "wp2_results.json", json_bytes(_jsonable(results)))
    write_bytes(campaign.RECORD_DIR / "wp2_offspring.npz", npz_bytes(_pool_events(jobs, summaries)))
    write_bytes(campaign.RECORD_DIR / "wp2_intervention.npz", npz_bytes(_pool_intervention(intervention_rows)))
    if chosen is None:
        write_bytes(campaign.RECORD_DIR / "wp3_not_launched.json", json_bytes({
            "schema": "v3-not-launched-1", "work_package": 3, "rule": campaign.KERNEL_PATH_RULE,
            "reason": "no kernel passed its factor-1.5 test in WP2", "kernel_path": _jsonable(path_tests), "source": source_state()}))
    return results


def _measured_ratio_matrix(cell_wp1: dict) -> np.ndarray | None:
    """R_hat_ij = median drop / T_peak per pooled ordered pair from WP1 (mirror-filled), nan elsewhere."""
    t12 = cell_wp1.get("T1.2")
    if not t12 or not t12.get("pairs"):
        return None
    out = np.full((5, 5), np.nan)
    for (i, j), value in zip(t12["pairs"], t12["median_drop_over_Tpeak"]):
        out[i, j] = value
        out[4 - i, 4 - j] = value
    return out


def _pool_events(jobs, summaries) -> dict[str, np.ndarray]:
    names = sorted({light["cell"] for light in jobs})
    index = {name: k for k, name in enumerate(names)}
    pooled: dict[str, list] = {}
    for light, summary in zip(jobs, summaries):
        e = summary["events"]
        n = int(np.asarray(e["cable_j"]).size)
        if n == 0:
            continue
        for key in ("cable_j", "t_up", "T_peak", "n_cross", "offspring_per_cable", "primary", "first_lag", "n_marks"):
            pooled.setdefault("event_" + key, []).append(np.asarray(e[key]))
        pooled.setdefault("event_cell_index", []).append(np.full(n, index[light["cell"]], dtype=np.int64))
        pooled.setdefault("event_seed", []).append(np.full(n, int(light["seed"]), dtype=np.int64))
        pooled.setdefault("event_pilot", []).append(np.full(n, bool(light["pilot"])))
        o = summary["onsets"]
        m = int(np.asarray(o["cable"]).size)
        for key in ("cable", "time", "primary", "parent_event", "parent_cable", "lag", "cross"):
            pooled.setdefault("onset_" + key, []).append(np.asarray(o[key]))
        pooled.setdefault("onset_cell_index", []).append(np.full(m, index[light["cell"]], dtype=np.int64))
        pooled.setdefault("onset_seed", []).append(np.full(m, int(light["seed"]), dtype=np.int64))
    out = {k: np.concatenate(v) for k, v in pooled.items()}
    out["cell_names"] = np.array(names)
    return out


def _pool_intervention(rows: list[dict]) -> dict[str, np.ndarray]:
    if not rows:
        return {"n": np.zeros(0)}
    names = sorted({row["cell"] for row in rows})
    index = {name: k for k, name in enumerate(names)}
    out = {
        "cell_index": np.array([index[r["cell"]] for r in rows]), "seed": np.array([r["seed"] for r in rows]), "event_id": np.array([r["event_id"] for r in rows]),
        "cable": np.array([r["cable"] for r in rows]), "t_up": np.array([r["t_up"] for r in rows]), "x": np.array([r["x"] for r in rows]),
        "T_peak": np.array([r["T_peak"] for r in rows]), "v_return": np.array([r["v_return"] for r in rows]),
        "t_cf": np.array([np.nan if r["factual_parent"]["t_cf"] is None else r["factual_parent"]["t_cf"] for r in rows]),
        "v_cf": np.array([np.nan if r["factual_parent"]["v_cf"] is None else r["factual_parent"]["v_cf"] for r in rows]),
        "T_peak_cf": np.array([np.nan if r["factual_parent"]["T_peak_cf"] is None else r["factual_parent"]["T_peak_cf"] for r in rows]),
        "recorded_onsets": np.array([r["recorded_onsets"] for r in rows]), "factual_onsets": np.array([r["factual_onsets"] for r in rows]),
        "counterfactual_onsets": np.array([r["counterfactual_onsets"] for r in rows]), "causal": np.array([r["causal"] for r in rows]),
        "measured_cross": np.array([r["measured_cross"] for r in rows]), "onset_set_reproduced": np.array([r["onset_set_reproduced"] for r in rows]),
        "cell_names": np.array(names),
    }
    if all("causal_held" in r for r in rows):
        out["causal_held"] = np.array([r["causal_held"] for r in rows])
    return out


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if not math.isfinite(v) else v
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


if __name__ == "__main__":
    print(json.dumps({t: v["verdict"] for t, v in wp2()["tests"].items()}))
