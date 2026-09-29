"""Plan v3 analysis: the declared tests, scored from the stage caches, and the gate records.

    python -m tether.campaign.v3.analyse wp1        # records/v3/wp1_results.json (+ wp1_transmission.npz)
    python -m tether.campaign.v3.analyse wp2        # records/v3/wp2_results.json (+ offspring/intervention npz)
    python -m tether.campaign.v3.analyse wp3        # records/v3/wp3_results.json (+ cluster sizes npz)
    python -m tether.campaign.v3.analyse gate N     # records/v3/gate_wpN.json, after the audit round

Every threshold, population and branch is the declared one (records/v3/cascade_declarations.json,
tether/campaign/v3/campaign.py); pilot seeds are never scored; intervals resample seeds with
rng = default_rng(20260917).  Results carry the declarations' and predictions' sha256.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.stats import spearmanr

from tether.campaign.common import json_bytes, npz_bytes, sha256_file, source_state, write_bytes
from tether.campaign.v3 import campaign, cells, compute
from tether.evt._common import percentile_interval


def addenda_sha256() -> dict:
    """Every dated addendum on disk, by file name, so a result names the declarations *and* their amendments."""
    return {p.name: sha256_file(p) for p in sorted(campaign.RECORD_DIR.glob("cascade_addendum_*.json"))}


ADDENDA = addenda_sha256

RNG_SEED = campaign.BOOTSTRAP_SEED
N_BOOT = 2000


# ----------------------------------------------------------------------------- shared


def _load(stage: str = "A") -> tuple[list[dict], list[dict]]:
    campaign.verify_declared()
    return compute.load_stage(stage)


def _by_cell(jobs: list[dict], summaries: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for light, summary in zip(jobs, summaries):
        out.setdefault(light["cell"], []).append(summary)
    return out


def _seed_bootstrap(per_seed_arrays: list[np.ndarray], statistic, n_boot: int = N_BOOT, rng=None) -> tuple[float, float, float]:
    """``statistic`` of the concatenation, with a seed-resampling percentile interval (nan-safe)."""
    items = [np.asarray(a, dtype=float) for a in per_seed_arrays]
    if not items:
        return math.nan, math.nan, math.nan
    generator = np.random.default_rng(RNG_SEED) if rng is None else rng
    indices = generator.integers(0, len(items), size=(n_boot, len(items)))
    value = statistic(np.concatenate(items))
    replicates = np.array([statistic(np.concatenate([items[k] for k in row])) for row in indices], dtype=float)
    lower, upper = percentile_interval(replicates, 0.95)
    return float(value), float(lower), float(upper)


def _nanmedian(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else math.nan


def _share_verdict(cell_verdicts: dict[str, str], min_scored: int = campaign.MIN_SCORED_CELLS, share: float = campaign.SHARE) -> dict:
    scored = {c: v for c, v in cell_verdicts.items() if v in ("PASS", "FAIL")}
    n_pass = sum(v == "PASS" for v in scored.values())
    if len(scored) < min_scored:
        verdict = "UNDER-POWERED"
    else:
        verdict = "PASS" if n_pass >= share * len(scored) - 1e-9 else "FAIL"
    return {"verdict": verdict, "scored_cells": len(scored), "passing_cells": n_pass, "share_required": share, "per_cell": cell_verdicts}


def _pooled_pair(i: int, j: int) -> tuple[int, int]:
    """Mirror pairs (i, j) ~ (4 - i, 4 - j) share one label (the lexicographically smaller)."""
    a, b = (i, j), (4 - i, 4 - j)
    return min(a, b)


def _matrix(entries: list) -> np.ndarray:
    return np.array([[np.nan if v is None else float(v) for v in row] for row in entries])


# ----------------------------------------------------------------------------- WP1


def _wp1_cell(name: str, runs: list[dict], prediction: dict, declared: dict) -> dict:
    """T1.1-T1.3 statistics of one cell from its statistics-seed runs (pilots excluded)."""
    stat = [r for r in runs if not r["meta"]["pilot"]]
    x_lo, x_hi = campaign.X_RANGE_PATTERN
    e_lo, e_hi = campaign.X_RANGE_EXACT
    r_m1 = _matrix(prediction["transmission"]["response_M1_half_sine"])
    g_design = _matrix(prediction["transmission"]["geometry_factor"])   # the forecast's normaliser (design geometry)
    per_seed_norm, per_seed_cos, per_seed_half, per_seed_bounce, per_seed_exact, per_seed_design = [], [], [], [], [], []
    g_nonpositive = g_small = 0
    pair_drops: dict[tuple[int, int], list] = {}
    pair_parents: dict[tuple[int, int], int] = {}
    n_eligible = 0
    for r in stat:
        t = r["transmission"]
        if t["parent_mark"].size == 0:
            continue
        parent = t["role"] == 0
        base = t["eligible_i"] & t["all_neighbours_taut"]
        band = base & parent & (t["x"] >= x_lo) & (t["x"] <= x_hi)
        exact = base & parent & (t["x"] >= e_lo) & (t["x"] <= e_hi)
        n_eligible += int(band.sum())
        per_seed_norm.append(t["drop_norm"][band])
        per_seed_cos.append(t["drop_norm_cos"][band])
        i_b, j_b = t["neighbour_i"][band].astype(int), t["cable_j"][band].astype(int)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_seed_design.append(t["drop"][band] / (t["T_peak"][band] * g_design[i_b, j_b]))
        g_nonpositive += int(np.sum(t["geometry_factor_ij"][band] <= 0.0))
        g_small += int(np.sum(t["geometry_factor_ij"][band] < 0.25))
        per_seed_bounce.append(t["drop_norm"][base & (t["x"] >= x_lo) & (t["x"] <= x_hi)])
        half = r["transmission_half_window"]
        half_norm = half["drop_norm"] if "drop_norm" in half else np.full(band.size, np.nan)
        per_seed_half.append(np.asarray(half_norm)[band] if np.asarray(half_norm).size == band.size else np.zeros(0))
        i, j = t["neighbour_i"].astype(int), t["cable_j"].astype(int)
        rij = r_m1[i, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = t["drop"] / (t["T_peak"] * rij)
        per_seed_exact.append(ratio[exact & np.isfinite(rij)])
        for k in np.flatnonzero(band):
            key = _pooled_pair(int(i[k]), int(j[k]))
            pair_drops.setdefault(key, []).append(float(t["drop"][k] / t["T_peak"][k]))
            pair_parents[key] = pair_parents.get(key, 0) + 1
    scored = n_eligible >= campaign.MIN_PAIRS
    band_value, band_lo, band_hi = _seed_bootstrap(per_seed_norm, _nanmedian)
    cos_value = _nanmedian(np.concatenate(per_seed_cos)) if per_seed_cos else math.nan
    half_value = _nanmedian(np.concatenate(per_seed_half)) if per_seed_half else math.nan
    bounce_value = _nanmedian(np.concatenate(per_seed_bounce)) if per_seed_bounce else math.nan
    exact_value, exact_lo, exact_hi = _seed_bootstrap(per_seed_exact, _nanmedian)
    design_value, design_lo, design_hi = _seed_bootstrap(per_seed_design, _nanmedian)
    n_exact = int(sum(a.size for a in per_seed_exact))
    # T1.2: pooled ordered pairs with enough parents, against R (mirror-averaged) and against the operator
    pairs = sorted(k for k, n in pair_parents.items() if n >= campaign.MIN_PER_PAIR)
    measured = [float(np.median(pair_drops[k])) for k in pairs]
    r_pooled = [float(np.nanmean([r_m1[k[0], k[1]], r_m1[4 - k[0], 4 - k[1]]])) for k in pairs]
    operator = _matrix(prediction["transmission"]["operator"])
    t_pooled = [float(np.mean([operator[k[0], k[1]], operator[4 - k[0], 4 - k[1]]])) for k in pairs]
    if len(pairs) >= 3 and np.std(r_pooled) > 0 and np.std(measured) > 0:
        rho_r = float(spearmanr(measured, r_pooled).statistic)
        rho_t = float(spearmanr(measured, t_pooled).statistic)
    else:
        rho_r = rho_t = math.nan
    lo, hi = campaign.BAND
    return {
        "cell": name, "n_statistics_runs": len(stat), "n_eligible_pairs": n_eligible, "scored": scored,
        "T1.1": {"median_drop_norm": band_value, "ci95": [band_lo, band_hi], "band": [lo, hi],
                 "verdict": ("PASS" if lo <= band_value <= hi else "FAIL") if scored and np.isfinite(band_value) else "UNSCORED",
                 "committed_prediction": prediction["transmission"]["band_statistic_median_offdiag"]},
        "T1.2": {"pairs": [list(k) for k in pairs], "parents_per_pair": [pair_parents[k] for k in pairs], "median_drop_over_Tpeak": measured,
                 "R_pooled": r_pooled, "operator_pooled": t_pooled, "spearman_vs_R": rho_r, "spearman_vs_operator": rho_t,
                 "verdict": ("PASS" if rho_r >= 0.7 else "FAIL") if scored and np.isfinite(rho_r) else "UNSCORED"},
        "T1.3": {"n": n_exact, "median_drop_over_Tpeak_R": exact_value, "ci95": [exact_lo, exact_hi],
                 "verdict": ("PASS" if 1.0 / campaign.FACTOR <= exact_value <= campaign.FACTOR else "FAIL")
                 if (n_exact >= campaign.MIN_PAIRS and np.isfinite(exact_value)) else "UNSCORED"},
        "sensitivities": {"cos_normalised_median": cos_value, "half_window_median": half_value, "parents_including_bounces_median": bounce_value,
                          # audit round 1, M1: the forecast is normalised by the DESIGN geometry factor, the declared statistic by the
                          # factor at the MEASURED geometry (which is <= 0 for some rows); the like-for-like reading is reported here
                          "design_geometry_median": design_value, "design_geometry_ci95": [design_lo, design_hi],
                          "design_geometry_in_band": bool(lo <= design_value <= hi) if np.isfinite(design_value) else None,
                          "share_rows_measured_factor_nonpositive": g_nonpositive / n_eligible if n_eligible else math.nan,
                          "share_rows_measured_factor_below_0.25": g_small / n_eligible if n_eligible else math.nan},
    }


def wp1() -> dict:
    jobs, summaries = _load("A")
    predictions = json.loads(campaign.PREDICTIONS_PATH.read_text())
    by_cell = _by_cell(jobs, summaries)
    # T1.0
    checked = [s for s in summaries if s["reproduction"]["checked"]]
    failed = [s for s in checked if not s["reproduction"]["passed"]]
    empty = [s for s in checked if np.asarray(s["marks"]["t_up"]).size == 0]
    t10 = {"checked_runs": len(checked), "failed_runs": len(failed), "empty_vs_empty_runs": len(empty),
           "first_failure": None if not failed else {"cell": failed[0]["meta"]["cell"], "seed": failed[0]["meta"]["seed"], **failed[0]["reproduction"]["first_difference"]},
           "verdict": "PASS" if checked and not failed else ("FAIL" if failed else "UNDER-POWERED")}
    per_cell = {name: _wp1_cell(name, runs, predictions["cells"][name], campaign.DECLARATIONS) for name, runs in by_cell.items()}
    t11 = _share_verdict({c: v["T1.1"]["verdict"] for c, v in per_cell.items()})
    t12 = _share_verdict({c: v["T1.2"]["verdict"] for c, v in per_cell.items()})
    t13 = _share_verdict({c: v["T1.3"]["verdict"] for c, v in per_cell.items()})
    # T1.4: paired by seed between the two stiffness cells
    lo_name, hi_name = predictions["stiffness_pair"]["low"], predictions["stiffness_pair"]["high"]
    t14 = _stiffness_test(by_cell.get(lo_name, []), by_cell.get(hi_name, []),
                          _matrix(predictions["cells"][lo_name]["transmission"]["response_over_geometry_factor"]),
                          _matrix(predictions["cells"][hi_name]["transmission"]["response_over_geometry_factor"]),
                          predictions["stiffness_pair"]["delta_exact_uniform_pairs"])
    closures = {name: {"closures": sum(r["meta"]["closure"] is not None for r in runs), "runs": len(runs)} for name, runs in by_cell.items()}
    results = {
        "schema": "v3-wp1-results-1", "tests": {"T1.0": t10, "T1.1": t11, "T1.2": t12, "T1.3": t13, "T1.4": t14},
        "cells": per_cell, "closures": closures,
        "compute": {"runs": len(summaries), "core_hours_run": sum(s["meta"]["wall_seconds_run"] for s in summaries) / 3600.0,
                    "core_hours_reduce": sum(s["meta"]["wall_seconds_reduce"] for s in summaries) / 3600.0},
        "declarations_sha256": sha256_file(campaign.DECLARATIONS_PATH), "predictions_sha256": sha256_file(campaign.PREDICTIONS_PATH), "addenda_sha256": ADDENDA(),
        "source": source_state(),
    }
    write_bytes(campaign.RECORD_DIR / "wp1_results.json", json_bytes(results))
    write_bytes(campaign.RECORD_DIR / "wp1_transmission.npz", npz_bytes(_pool_rows(jobs, summaries)))
    return results


def _stiffness_test(low_runs: list[dict], high_runs: list[dict], norm_low: np.ndarray, norm_high: np.ndarray, delta_uniform: float) -> dict:
    """T1.4: Delta_meas with its interval, against Delta_exact over the SAME rows' pair mix (declared)."""
    x_lo, x_hi = campaign.X_RANGE_PATTERN

    def per_seed(runs, predicted_matrix):
        out, pred = {}, {}
        for r in runs:
            if r["meta"]["pilot"]:
                continue
            t = r["transmission"]
            if t["parent_mark"].size == 0:
                out[r["meta"]["seed"]] = np.zeros(0)
                pred[r["meta"]["seed"]] = np.zeros(0)
                continue
            mask = t["eligible_i"] & t["all_neighbours_taut"] & (t["role"] == 0) & (t["x"] >= x_lo) & (t["x"] <= x_hi)
            out[r["meta"]["seed"]] = t["drop_norm"][mask]
            pred[r["meta"]["seed"]] = predicted_matrix[t["neighbour_i"][mask].astype(int), t["cable_j"][mask].astype(int)]
        return out, pred

    low, pred_low = per_seed(low_runs, norm_low)
    high, pred_high = per_seed(high_runs, norm_high)
    # seeds with eligible rows on both sides (stricter than T1.1/T1.3, which skip zero-mark runs only); the reporting
    # fields below use the same list, so a dropped seed's rows appear nowhere in this test
    seeds = sorted(s for s in set(low) & set(high) if low[s].size and high[s].size)
    if not seeds:
        return {"verdict": "UNSCORED", "reason": "no common statistics seeds"}
    generator = np.random.default_rng(RNG_SEED)
    indices = generator.integers(0, len(seeds), size=(N_BOOT, len(seeds)))

    def delta(rows):
        a = np.concatenate([high[seeds[k]] for k in rows])
        b = np.concatenate([low[seeds[k]] for k in rows])
        return _nanmedian(a) - _nanmedian(b)

    value = delta(range(len(seeds)))
    replicates = np.array([delta(row) for row in indices], dtype=float)
    lower, upper = percentile_interval(replicates, 0.95)
    delta_exact = _nanmedian(np.concatenate([pred_high[s] for s in seeds])) - _nanmedian(np.concatenate([pred_low[s] for s in seeds]))
    n_low, n_high = int(sum(low[s].size for s in seeds)), int(sum(high[s].size for s in seeds))
    if n_low < campaign.MIN_PAIRS or n_high < campaign.MIN_PAIRS or not np.isfinite(value):
        verdict = "UNSCORED"
    else:
        has_exact = lower <= delta_exact <= upper
        has_zero = lower <= 0.0 <= upper
        verdict = "PASS-exact" if has_exact and not has_zero else "PASS-closed" if has_zero and not has_exact else "UNDER-POWERED" if has_exact and has_zero else "FAIL-both"
    return {"delta_measured": value, "ci95": [lower, upper], "delta_exact": delta_exact, "delta_exact_uniform_pairs": delta_uniform, "delta_closed_form": 0.0,
            "n_pairs": {"low": n_low, "high": n_high}, "seeds": seeds, "verdict": verdict,
            "median_low": _nanmedian(np.concatenate([low[s] for s in seeds])), "median_high": _nanmedian(np.concatenate([high[s] for s in seeds]))}


def _pool_rows(jobs: list[dict], summaries: list[dict]) -> dict[str, np.ndarray]:
    names = sorted({light["cell"] for light in jobs})
    index = {name: k for k, name in enumerate(names)}
    pooled: dict[str, list] = {"cell_index": [], "seed": [], "pilot": []}
    for light, summary in zip(jobs, summaries):
        t = summary["transmission"]
        n = int(t["parent_mark"].size)
        if n == 0:
            continue
        for key, value in t.items():
            pooled.setdefault(key, []).append(np.asarray(value))
        pooled["cell_index"].append(np.full(n, index[light["cell"]], dtype=np.int64))
        pooled["seed"].append(np.full(n, int(light["seed"]), dtype=np.int64))
        pooled["pilot"].append(np.full(n, bool(light["pilot"])))
    out = {key: np.concatenate(value) for key, value in pooled.items() if value}
    out["cell_names"] = np.array(names)
    return out


# ----------------------------------------------------------------------------- gates


def gate(work_package: int) -> str:
    results = json.loads((campaign.RECORD_DIR / f"wp{work_package}_results.json").read_text())
    tests = results["tests"]
    blocking = [t for t, spec in campaign.TESTS.items() if spec["blocking"] and spec["work_package"] == work_package]
    if work_package == 1:
        verdict = "OPEN"  # WP2 always runs; the kernel path is chosen by the WP1 verdicts
        consequences = [campaign.TESTS[t]["branches"].get(tests[t]["verdict"], campaign.TESTS[t]["branches"].get("FAIL")) for t in blocking]
    elif work_package == 2:
        verdict = "OPEN" if results.get("K_path_chosen") else "NOT-LAUNCHED"
        consequences = [f"kernel path: {results.get('K_path_chosen')}"] + [campaign.TESTS[t]["branches"].get(tests[t]["verdict"], "") for t in blocking]
    else:
        verdict = "CLOSED"
        consequences = [campaign.TESTS[t]["branches"].get(tests[t]["verdict"], "") for t in blocking] + [f"T3.4 branch: {tests.get('T3.4', {}).get('branch')}"]
    provenance = {"results": f"records/v3/wp{work_package}_results.json", "results_sha256": sha256_file(campaign.RECORD_DIR / f"wp{work_package}_results.json"),
                  "audit": f"reports/v3/corrections_log.md round {work_package}"}
    return campaign.write_gate(work_package, verdict, {t: {"verdict": tests[t].get("verdict"), "what": campaign.TESTS[t]["statement"]} for t in campaign.TESTS if campaign.TESTS[t]["work_package"] == work_package and t in tests},
                               consequences, provenance)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["wp1", "wp2", "wp3", "gate"])
    parser.add_argument("args", nargs="*")
    args = parser.parse_args(argv)
    if args.command == "wp1":
        results = wp1()
        print(json.dumps({t: v["verdict"] for t, v in results["tests"].items()}))
    elif args.command == "gate":
        print("wrote gate", gate(int(args.args[0])))
    elif args.command == "wp2":
        from tether.campaign.v3.analyse_wp2 import wp2

        print(json.dumps({t: v["verdict"] for t, v in wp2()["tests"].items()}))
    else:
        from tether.campaign.v3.analyse_wp3 import wp3

        print(json.dumps({t: v.get("verdict", v.get("branch")) for t, v in wp3()["tests"].items()}))


if __name__ == "__main__":
    main()
