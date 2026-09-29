"""Plan v3 WP3 analysis: rho(K), theta, the cluster-corrected rate law, the sweep and the cluster-size law.

Scores T3.1-T3.5 from the stage caches (A, B and, if run, B_ext) as declared.  The kernel is the
one chosen in WP2 (records/v3/wp2_results.json, ``K_path_chosen``): its margin law comes from the
cell's pilot seeds, its peak average from the statistics-seed events; nu_primary and the primary
type distribution are measured on the statistics seeds.  Intervals resample statistics seeds with
rng = default_rng(20260917).  Branch (c) of T3.4 authorises stage B_ext through
records/v3/b_ext_authorised.json when the declared rule allows it.
"""

from __future__ import annotations

import json
import math

import numpy as np

from tether.campaign.common import json_bytes, npz_bytes, sha256_file, source_state, write_bytes
from tether.campaign.v3 import analyse as A
from tether.campaign.v3 import analyse_wp2 as W2
from tether.campaign.v3 import campaign
from tether.campaign.v3.analyse import addenda_sha256 as ADDENDA, cells, compute
from tether.evt import extremal as E
from tether.evt._common import percentile_interval
from tether.theory import branching as B
from tether.theory import excursion as EX
from tether.theory import reduced_lti as R

N_BOOT = 2000


def _ratio_for(name: str, wp2: dict, predictions: dict, wp1: dict) -> np.ndarray | None:
    path = wp2["K_path_chosen"]
    if path == "R":
        return A._matrix(predictions["cells"][name]["transmission"]["response_M1_half_sine"])
    if path == "R_hat":
        return W2._measured_ratio_matrix(wp1["cells"].get(name, {}))
    return None


def _kernel_from_events(path: str, ratio, quantiles, events: dict, pilots: dict) -> np.ndarray:
    if path == "empirical":
        return np.nan_to_num(B.empirical_kernel(pilots["cable_j"], pilots["offspring_per_cable"]), nan=0.0)
    return W2.kernel_prediction(ratio, quantiles, events)["kernel"]


def _cell(name: str, runs: list[dict], wp2: dict, predictions: dict, wp1: dict) -> dict:
    path = wp2["K_path_chosen"]
    stat = W2._events_of(runs, pilot=False)
    pilots = W2._events_of(runs, pilot=True)
    quantiles = W2.pooled_margin_quantiles(runs, pilot=True)
    ratio = _ratio_for(name, wp2, predictions, wp1)
    stats_runs = [r for r in runs if not r["meta"]["pilot"]]
    seeds = np.array([int(r["meta"]["seed"]) for r in stats_runs])
    n_events = int(stat["T_peak"].size)
    measured_cross = float(stat["n_cross"].sum())
    scored = measured_cross >= campaign.MIN_OFFSPRING and n_events > 0
    out = {"cell": name, "n_events": n_events, "measured_cross_offspring": measured_cross, "scored": scored,
           "closures": int(sum(r["meta"]["closure"] is not None for r in runs)), "runs": len(runs)}
    out["operable"] = out["closures"] <= campaign.OPERABLE_CLOSURE_FRACTION * len(runs)
    if not scored or (path != "empirical" and (ratio is None or not np.all(np.isfinite(quantiles)))):
        out["verdicts"] = {"T3.1": "UNSCORED", "T3.2": "UNSCORED"}
        return out
    kernel = _kernel_from_events(path, ratio, quantiles, stat, pilots)
    rho = B.spectral_radius(kernel)
    # seed bootstrap of rho (statistics seeds; margin law fixed)
    generator = np.random.default_rng(campaign.BOOTSTRAP_SEED)
    per_seed = {int(s): {k: v[stat["seed"] == s] for k, v in stat.items()} for s in np.unique(stat["seed"])}
    seed_list = sorted(per_seed)
    replicates = []
    for row in generator.integers(0, len(seed_list), size=(N_BOOT, len(seed_list))):
        merged = {k: np.concatenate([per_seed[seed_list[i]][k] for i in row]) for k in ("T_peak", "cable_j", "offspring_per_cable")}
        try:
            replicates.append(B.spectral_radius(_kernel_from_events(path, ratio, quantiles, merged, pilots)))
        except ValueError:
            replicates.append(math.nan)
    rho_lo, rho_hi = percentile_interval(np.array(replicates), 0.95)
    # primaries, exposures, rates
    primary = stat["primary"]
    pi = np.array([np.sum(stat["cable_j"][primary] == c) for c in range(5)], dtype=float)
    n_primary = float(pi.sum())
    pi = pi / n_primary if n_primary > 0 else np.full(5, 0.2)
    exposure = float(sum(r["meta"]["exposure_s"] for r in stats_runs))
    primary_exposure = float(sum(r["meta"]["primary_exposure_s"] for r in stats_runs))
    nu_primary = np.array([np.sum(stat["cable_j"][primary] == c) for c in range(5)]) / max(primary_exposure, 1e-9)
    nu_total_measured = np.array([np.sum(stat["cable_j"] == c) for c in range(5)]) / max(exposure, 1e-9)
    subcritical = rho < 1.0
    nu_total_pred = B.total_rate(kernel, nu_primary) if subcritical else np.full(5, np.nan)
    ratio_pooled = float(nu_total_pred.sum() / nu_total_measured.sum()) if subcritical and nu_total_measured.sum() > 0 else math.nan
    theta_pred = B.extremal_index(kernel, pi) if subcritical else math.nan
    theta_runs = n_primary / n_events
    per_seed_theta = [(float(np.sum(per_seed[s]["primary"])), float(per_seed[s]["primary"].size)) for s in seed_list]
    reps = []
    for row in generator.integers(0, len(seed_list), size=(N_BOOT, len(seed_list))):
        p = sum(per_seed_theta[i][0] for i in row)
        n = sum(per_seed_theta[i][1] for i in row)
        reps.append(p / n if n else math.nan)
    theta_lo, theta_hi = percentile_interval(np.array(reps), 0.95)
    # Ferro-Segers on the fleet onset indicator (1 ms), per seed then pooled
    onset_indices = [np.unique(np.round(r["onsets"]["time"] / 1.0e-3).astype(np.int64)) for r in stats_runs]
    theta_fs, fs_lo, fs_hi = E.bootstrap_theta(onset_indices, "intervals", n_boot=500, rng=campaign.BOOTSTRAP_SEED)
    # Rice (report only): LTI onset rate of the taut tow
    rice = _rice_rate(name)
    factor = campaign.FACTOR
    t31 = ("PASS" if 1.0 / factor <= ratio_pooled <= factor else "FAIL") if np.isfinite(ratio_pooled) else "FAIL"
    t32 = ("PASS" if abs(theta_pred - theta_runs) <= 0.10 else "FAIL") if np.isfinite(theta_pred) else "FAIL"
    # cluster sizes from the event forest
    sizes = np.concatenate([E.cluster_sizes_from_forest(np.asarray(r["events"]["parent_event"])) for r in stats_runs if np.asarray(r["events"]["cable_j"]).size]) if stats_runs else np.zeros(0, dtype=np.int64)
    law = E.empirical_cluster_law(sizes)
    bt = None
    if subcritical and sizes.size:
        support = np.arange(1, int(sizes.max()) + 1)
        empirical_cdf = np.cumsum(np.array([law["pmf"].get(int(n), 0.0) for n in support]))
        m = B.mean_offspring(kernel, pi)
        bt_cdf = np.cumsum(B.borel_tanner_pmf(min(m, 0.999), support))
        mc = B.simulate_cluster_sizes(kernel, pi, 20000, rng=campaign.BOOTSTRAP_SEED)
        mc_cdf = np.cumsum(np.bincount(np.minimum(mc, support.max()), minlength=support.max() + 1)[1:] / mc.size)
        bt = {"m_bar": m, "ks_borel_tanner": float(np.max(np.abs(empirical_cdf - bt_cdf))), "ks_multitype_mc": float(np.max(np.abs(empirical_cdf - mc_cdf))),
              "mc_mean": float(mc.mean()), "support_max": int(support.max())}
    out.update({
        "kernel_path": path, "K": kernel.tolist(), "rho": rho, "rho_ci95": [float(rho_lo), float(rho_hi)],
        "pi": pi.tolist(), "n_primary": n_primary, "exposure_s": exposure, "primary_exposure_s": primary_exposure,
        "nu_primary_per_cable_s": nu_primary.tolist(), "nu_total_measured_per_cable_s": nu_total_measured.tolist(),
        "nu_total_predicted_per_cable_s": None if not subcritical else nu_total_pred.tolist(), "rate_ratio_pooled": ratio_pooled,
        "theta_pred": theta_pred, "theta_runs": theta_runs, "theta_runs_ci95": [float(theta_lo), float(theta_hi)],
        "theta_ferro_segers": theta_fs, "theta_ferro_segers_ci95": [fs_lo, fs_hi], "m_bar": B.mean_offspring(kernel, pi),
        "identity_check": {"nu_total_over_primary_measured": float(nu_total_measured.sum() / nu_primary.sum()) if nu_primary.sum() > 0 else None,
                           "one_over_one_minus_m_bar": None if B.mean_offspring(kernel, pi) >= 1 else 1.0 / (1.0 - B.mean_offspring(kernel, pi))},
        "rice": rice, "cluster_law": {k: v for k, v in law.items() if k != "pmf"}, "cluster_pmf": law["pmf"], "borel_tanner": bt,
        "verdicts": {"T3.1": t31, "T3.2": t32},
    })
    return out


def _rice_rate(name: str) -> dict:
    """T3.3 (report only): Rice onset rate of the taut LTI tow, per cable, and the implied theta_primary."""
    parsed = cells.parse_cell(name)
    spec = cells.cell_specs("A").get(name) or cells.cell_specs("B").get(name) or cells.cell_specs("B_ext").get(name)
    try:
        result = R.reduced_model(cells.response_spec(spec).inp)
        rate = np.array([EX.onset_rate(float(result.mu_e[i]) if hasattr(result, "mu_e") else parsed["pretension"] / 1.7e5,
                                        float(result.sigma_e[i]), float(result.sigma_edot[i])) for i in range(5)])
        return {"rice_onset_rate_per_cable_s": rate.tolist(), "model": "tether.theory.reduced_lti.reduced_model + excursion.onset_rate"}
    except Exception as error:  # noqa: BLE001 - report only
        return {"error": str(error)}


def _sweep_branch(sweep: dict) -> dict:
    """T3.4: (a) transition / (b) margin / (c) under-powered, from the sweep cells' rho intervals."""
    rows = sorted((cells.parse_cell(n)["intensity"], n, c) for n, c in sweep.items() if "rho" in c)
    operable = [(i, n, c) for i, n, c in rows if c["operable"]]
    below = [(i, n, c) for i, n, c in operable if c["rho_ci95"][1] < 1.0]
    above = [(i, n, c) for i, n, c in operable if c["rho_ci95"][0] > 1.0]
    table = {n: {"intensity": i, "rho": c["rho"], "ci95": c["rho_ci95"], "closures": c["closures"], "operable": c["operable"]} for i, n, c in rows}
    if above and below and min(i for i, _, _ in above) > max(i for i, _, _ in below):
        lo = max(below, key=lambda r: r[0])
        hi = min(above, key=lambda r: r[0])
        i_c = math.exp(np.interp(0.0, [math.log(lo[2]["rho"]), math.log(hi[2]["rho"])], [math.log(lo[0]), math.log(hi[0])]))
        return {"branch": "(a)", "critical_intensity": float(i_c), "between": [lo[1], hi[1]], "table": table}
    if operable and not above:
        top = max(operable, key=lambda r: r[0])
        if top[2]["rho_ci95"][1] < 1.0:
            return {"branch": "(b)", "highest_operable": top[1], "rho": top[2]["rho"], "margin": 1.0 - top[2]["rho"], "table": table}
    return {"branch": "(c)", "table": table}


def wp3() -> dict:
    predictions = json.loads(campaign.PREDICTIONS_PATH.read_text())
    wp1 = json.loads((campaign.RECORD_DIR / "wp1_results.json").read_text())
    wp2 = json.loads((campaign.RECORD_DIR / "wp2_results.json").read_text())
    if not wp2.get("K_path_chosen"):
        raise SystemExit("WP3 is not launched: no kernel passed in WP2 (records/v3/wp3_not_launched.json)")
    jobs, summaries = A._load("A")
    by_cell = A._by_cell(jobs, summaries)
    stages = ["A"]
    for stage in ("B", "B_ext"):
        if compute.STAGE_CACHE[stage].exists():
            j, s = compute.load_stage(stage)
            for name, runs in A._by_cell(j, s).items():
                by_cell.setdefault(name, []).extend(runs)
            stages.append(stage)
    per_cell = {name: _cell(name, runs, wp2, predictions, wp1) for name, runs in by_cell.items()}
    t31 = A._share_verdict({n: c["verdicts"]["T3.1"] for n, c in per_cell.items()})
    t32 = A._share_verdict({n: c["verdicts"]["T3.2"] for n, c in per_cell.items()})
    sweep = {n: per_cell[n] for n in cells.sweep_cells() if n in per_cell}
    t34 = _sweep_branch(sweep)
    b_ext_rule = False
    top15 = per_cell.get("T600_k500_I1.5")
    if t34["branch"] == "(c)" and top15 is not None and top15["closures"] <= campaign.OPERABLE_CLOSURE_FRACTION * top15["runs"] and "B_ext" not in stages:
        b_ext_rule = True
        write_bytes(campaign.RECORD_DIR / "b_ext_authorised.json", json_bytes({"rule": campaign.DECLARATIONS["stages"]["B_ext"], "closures_at_1.5": top15["closures"],
                                                                                "runs": top15["runs"], "branch_before": "(c)", "source": source_state()}))
    results = {
        "schema": "v3-wp3-results-1", "stages": stages, "kernel_path": wp2["K_path_chosen"],
        "tests": {"T3.1": t31, "T3.2": t32, "T3.3": {"verdict": "REPORT"}, "T3.4": t34, "T3.5": {"verdict": "REPORT"}},
        "cells": per_cell, "b_ext_authorised_now": b_ext_rule,
        "declarations_sha256": sha256_file(campaign.DECLARATIONS_PATH), "predictions_sha256": sha256_file(campaign.PREDICTIONS_PATH), "addenda_sha256": ADDENDA(),
        "wp2_results_sha256": sha256_file(campaign.RECORD_DIR / "wp2_results.json"), "source": source_state(),
    }
    write_bytes(campaign.RECORD_DIR / "wp3_results.json", json_bytes(W2._jsonable(results)))
    sizes = {}
    for name, runs in by_cell.items():
        stats_runs = [r for r in runs if not r["meta"]["pilot"] and np.asarray(r["events"]["cable_j"]).size]
        if stats_runs:
            sizes[name] = np.concatenate([E.cluster_sizes_from_forest(np.asarray(r["events"]["parent_event"])) for r in stats_runs])
    write_bytes(campaign.RECORD_DIR / "wp3_cluster_sizes.npz", npz_bytes({f"sizes__{n}": v for n, v in sizes.items()} | {"cell_names": np.array(sorted(sizes))}))
    return results


if __name__ == "__main__":
    out = wp3()
    print(json.dumps({t: v.get("verdict", v.get("branch")) for t, v in out["tests"].items()}))
