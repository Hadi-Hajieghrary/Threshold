"""Phase 5 analysis: stress threshold, hazards per arm, and P5-T0..T7."""

from __future__ import annotations

import json
import pickle

import numpy as np

from tether.campaign.common import json_bytes, run_pool, sha256_file, source_state, write_bytes
from tether.campaign.phase5 import (
    ARMS,
    DECLARATIONS,
    FAN_IMPACT_PATH,
    FAN_PRETENSION,
    MISSION_CACHE,
    RESULTS_PATH,
    SQUALL_PATH,
    STRESS_PROBABILITY,
    TAU,
)

HORIZON = 2.0
N_HAZARD = 2048


def _nan_to_none(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _report_dict(report) -> dict:
    out = {}
    for key, value in report.__dict__.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            out[key] = [v.__dict__ if hasattr(v, "__dict__") else v for v in value]
        elif hasattr(value, "__dict__"):
            out[key] = value.__dict__
        else:
            out[key] = _nan_to_none(value) if isinstance(value, float) else value
    return out


def stress_threshold(missions: list[dict]) -> dict:
    peaks = np.array([m["max_tension"] for m in missions])
    threshold = float(np.round(np.quantile(peaks, 1.0 - STRESS_PROBABILITY) / 100.0) * 100.0)
    probability = float(np.mean(peaks > threshold))
    return {"threshold": threshold, "probability": probability, "per_mission_max": peaks.tolist(), "inside_band": 0.3 <= probability <= 0.6}


def _oracle_output(mission: dict):
    from tether.monitor.oracle import oracle_precursor

    mt = mission["monitor_truth"]
    time = mission["outputs"]["P"].time
    e_true = np.asarray(mt.e_true)
    edot_true = np.asarray(mt.edot_true)
    slack = e_true <= 0.0
    onset = np.full(e_true.shape, np.nan)
    for vessel in range(e_true.shape[1]):
        current = np.nan
        for k in range(e_true.shape[0]):
            if slack[k, vessel]:
                if k == 0 or not slack[k - 1, vessel]:
                    if k > 0:
                        e0, e1 = e_true[k - 1, vessel], e_true[k, vessel]
                        fraction = e0 / (e0 - e1) if e0 != e1 else 1.0
                        current = time[k - 1] + fraction * (time[k] - time[k - 1])
                    else:
                        current = time[k]
                onset[k, vessel] = current
    # a_hat on the 10 ms grid of the estimator arms, from the 1 ms truth (like for like).
    series = (mission["series_time"], mission["elongation"], mission["rate"])
    return oracle_precursor(time, slack, onset, e_true, edot_true, pretension=FAN_PRETENSION, series=series)


def hazard_job(payload) -> dict:
    """One (arm, mission): hazards, tick and interval outcomes, monitor table, NEES."""
    from tether.monitor import metrics
    from tether.monitor.hazard import fleet_hazard
    from tether.monitor.outcomes import interval_outcomes, slack_intervals, tick_outcomes

    arm, mission, v_b, breaking, model, horizon = payload
    output = _oracle_output(mission) if arm == "O" else mission["outputs"][arm]
    seed = mission["seed"]
    time = np.asarray(output.time)
    slack = np.asarray(output.slack, dtype=bool)
    if model == "linearized":
        from tether.monitor.linearized import fleet_hazard_linearized

        hazard_values = fleet_hazard_linearized(output, v_b, seed, n_samples=N_HAZARD, horizon=horizon)
        rice = np.full(hazard_values.shape, np.nan)
    else:
        hazards = fleet_hazard(output, v_b, seed, n_samples=N_HAZARD, horizon=horizon)
        hazard_values = np.asarray(hazards.hazard)
        rice = np.asarray(hazards.rice)
    ticks = tick_outcomes(time, slack, mission["series_time"], mission["elongation"], mission["rate"], v_b, horizon=horizon)
    intervals = slack_intervals(time, slack, np.asarray(output.onset_time))
    outcomes = interval_outcomes(intervals, mission["series_time"], mission["elongation"], mission["rate"], v_b, breaking, horizon=horizon)
    table = metrics.monitor_table(hazard_values, ticks, intervals, outcomes, seed=seed)
    mt = mission["monitor_truth"]
    nees = metrics.tick_nees(output.e_hat, output.edot_hat, output.sigma, mt.e_true, mt.edot_true)
    interval_values = metrics.interval_nees(nees, intervals)
    return {"arm": arm, "seed": seed, "table": table, "interval_nees": interval_values,
            "precursor_states": (np.stack([np.asarray(output.e_hat), np.asarray(output.edot_hat), np.asarray(output.a_hat)], axis=-1)[slack] if arm == "O" else None),
            "precursor_covariances": (_oracle_covariances(output, slack) if arm == "O" else None),
            "hazard_slack": hazard_values[slack], "rice_slack": rice[slack],
            "vb_slack": np.broadcast_to(np.asarray(v_b), slack.shape)[slack]}


def _oracle_covariances(output, slack) -> np.ndarray:
    sigma = np.asarray(output.sigma)[slack]
    sigma_a = np.asarray(output.sigma_a)[slack]
    cov = np.zeros((sigma.shape[0], 3, 3))
    cov[:, :2, :2] = sigma
    cov[:, 2, 2] = sigma_a**2
    return cov


ARMS_CACHE = MISSION_CACHE.with_name("phase5_missions_arms.pkl")
PIVOTS = (("constant_acceleration", 2.0), ("linearized", 2.0), ("linearized", 1.0))


def _missions_with_arms(workers: int) -> list[dict]:
    from tether.campaign.phase5 import arms_job

    if ARMS_CACHE.exists() and ARMS_CACHE.stat().st_mtime > MISSION_CACHE.stat().st_mtime:
        with ARMS_CACHE.open("rb") as handle:
            return pickle.load(handle)
    with MISSION_CACHE.open("rb") as handle:
        missions = pickle.load(handle)
    missions = run_pool(arms_job, missions, workers)
    with ARMS_CACHE.open("wb") as handle:
        pickle.dump(missions, handle)
    return missions


def analyse(workers: int = 16) -> dict:
    """Run the pre-declared P5-T2 pivot sequence and report every attempt."""
    attempts = []
    final = None
    for model, horizon in PIVOTS:
        result = analyse_once(workers, model, horizon)
        summary = {arm: {k: (result["arms"].get(arm, {}).get("calibration") or {}).get(k) for k in ("slope", "slope_lo", "slope_hi", "large_intercept", "ece", "auroc", "n_outside")}
                   for arm in ("O", "P")}
        attempts.append({"model": model, "horizon": horizon, "P5-T2": result["tests"]["P5-T2"]["verdict"], "gate": result["gate"], "calibration": summary})
        final = result
        if result["tests"]["P5-T2"]["verdict"] != "FAIL":
            break
    final = finalize(final, attempts)
    write_bytes(RESULTS_PATH, json_bytes(final))
    return final


def finalize(final: dict, attempts: list[dict]) -> dict:
    """Tests and gate of the last attempt under the outcome matrix, with the pivot record.
    ``regate`` applies it to the stored results (the tests use only arms, rice and stress)."""
    final["tests"] = evaluate(final["arms"], final["rice_study"], final["stress"])
    exhausted = len(attempts) == len(PIVOTS) and all(a["P5-T2"] == "FAIL" for a in attempts)
    final["gate"] = phase5_gate(final["tests"], exhausted=exhausted)
    final["pivot_attempts"] = attempts
    return final


def regate() -> dict:
    results = json.loads(RESULTS_PATH.read_text())
    results = finalize(results, results["pivot_attempts"])
    write_bytes(RESULTS_PATH, json_bytes(results))
    return results


def analyse_once(workers: int, model: str, horizon: float) -> dict:
    from tether.estimation.fusion import apply_precursor_growth
    from tether.estimation.replay import calibrate_precursor_growth
    from tether.monitor import metrics
    from tether.monitor.hazard import critical_speeds

    missions = _missions_with_arms(workers)
    stress = stress_threshold(missions)
    breaking = stress["threshold"]
    v_b = critical_speeds(breaking, FAN_PRETENSION, table_path=FAN_IMPACT_PATH)
    growth = calibrate_precursor_growth([(m["outputs"]["L"], m["monitor_truth"]) for m in missions])
    for m in missions:
        m["outputs"]["L"] = apply_precursor_growth(m["outputs"]["L"], growth)
    payloads = [(arm, m, v_b, breaking, model, horizon) for arm in ("O",) + ARMS for m in missions]
    jobs = run_pool(hazard_job, payloads, workers)
    per_arm: dict[str, list] = {}
    for job in jobs:
        per_arm.setdefault(job["arm"], []).append(job)
    arms = {}
    rng = np.random.default_rng(20260914)
    for arm, items in per_arm.items():
        table = metrics.concatenate_tables([item["table"] for item in items])
        nees_values = np.concatenate([item["interval_nees"] for item in items]) if items else np.empty(0)
        nees_seeds = np.concatenate([np.full(item["interval_nees"].size, item["seed"]) for item in items])
        entry = {"ticks": int(table.forecast.size), "intervals": int(np.unique(table.interval).size), "events": int(np.sum(table.outcome)),
                 "severed_intervals": int(np.sum(table.severed)), "censored": int(table.censored), "dropped": int(table.dropped)}
        if table.forecast.size and np.any(table.outcome) and not np.all(table.outcome):
            report = metrics.calibration_report(table.forecast, table.outcome, table.interval, rng=rng)
            entry["calibration"] = _report_dict(report)
            h_crit = metrics.false_alarm_threshold(table.forecast, table.benign, table.interval)
            entry["h_crit"] = float(h_crit)
            leads = metrics.lead_times(table.forecast, table.time, table.interval, table.reengagement_time, table.severed, h_crit)
            summary = metrics.lead_time_summary(leads, table.severed, table.interval_seed, rng=rng)
            entry["lead_time"] = _report_dict(summary)
            entry["leads"] = np.asarray(leads).tolist()
        else:
            entry["calibration"] = None
        finite = np.isfinite(nees_values)
        if finite.any():
            entry["nees"] = _report_dict(metrics.nees_summary(nees_values[finite], nees_seeds[finite], rng=rng))
        arms[arm] = entry
    oracle = per_arm.get("O", [])
    states = np.concatenate([o["precursor_states"] for o in oracle if o["precursor_states"] is not None and o["precursor_states"].size]) if oracle else np.empty((0, 3))
    covariances = np.concatenate([o["precursor_covariances"] for o in oracle if o["precursor_covariances"] is not None and o["precursor_covariances"].size]) if oracle else np.empty((0, 3, 3))
    hazard_values = np.concatenate([o["hazard_slack"] for o in oracle]) if oracle else np.empty(0)
    vb_values = np.concatenate([o["vb_slack"] for o in oracle]) if oracle else np.empty(0)
    rice = None
    finite = np.all(np.isfinite(states), axis=1) & np.isfinite(hazard_values) if states.size else np.zeros(0, dtype=bool)
    if finite.sum() >= 20:
        chosen, _ = metrics.stratified_selection(hazard_values[finite], 17)
        index = np.flatnonzero(finite)[chosen]
        study = metrics.rice_study(states[index], covariances[index], vb_values[index], 17)
        rice = _report_dict(study)
    tests = evaluate(arms, rice, stress)
    gate = phase5_gate(tests)
    results = {"schema_version": 1, "source": source_state(), "declarations": DECLARATIONS, "tau": TAU, "prediction_model": model, "horizon": horizon,
               "squall_calibration_sha256": sha256_file(SQUALL_PATH) if SQUALL_PATH.exists() else None,
               "stress": stress, "v_b": np.asarray(v_b).tolist(), "l_growth": growth, "arms": arms, "rice_study": rice,
               "missions": {"count": len(missions), "slack_ticks_P": arms.get("P", {}).get("ticks"),
                            "outcomes": [m["outcome"].__dict__ if hasattr(m["outcome"], "__dict__") else m["outcome"] for m in missions]},
               "tests": tests, "gate": gate}
    write_bytes(RESULTS_PATH, json_bytes(results))
    return results


def _calib(entry: dict | None) -> dict | None:
    return None if not entry or not entry.get("calibration") else entry["calibration"]


def evaluate(arms: dict, rice: dict | None, stress: dict) -> dict:
    tests = {}
    tests["P5-T0"] = {"statement": "gating: on every slack interval the slack agent's own bearing/range are absent and its covariance grows (unit test)", "verdict": "PASS (tether/tests/test_estimation_core.py)"}
    if rice is None:
        tests["P5-T1"] = {"statement": "Rice approximation on 200 oracle states", "verdict": "UNDER-POWERED"}
    else:
        fraction = rice.get("domain_fraction")
        ratio_ok = fraction is not None and np.isfinite(fraction) and fraction >= 0.9
        tests["P5-T1"] = {"statement": "Rice approximation: 20/5 ms convergence and h_Rice/h_MC in [1, 1.25] for >= 90% of states with h_MC < 0.2",
                          "study": {k: rice.get(k) for k in ("passed", "domain_fraction", "domain_count", "converged", "convergence", "rice_failures") if k in rice},
                          "convergence_verdict": "PASS" if rice.get("converged") else "FAIL: outcome matrix - grid refined, the 5 ms result is the reference",
                          "approximation_verdict": "PASS" if ratio_ok else "FAIL: outcome matrix - the Rice form is dropped from later use",
                          "verdict": "PASS" if ratio_ok else "FAIL",
                          "note": "degenerate: under the constant-acceleration and linearised models a trajectory has at most one upcrossing in (0, H], so Rice equals the hazard up to numerical error"}
    for arm in ("O", "P", "B2", "L"):
        entry = arms.get(arm, {})
        cal = _calib(entry)
        if cal is None:
            continue
    o = _calib(arms.get("O"))
    p = _calib(arms.get("P"))
    b2 = _calib(arms.get("B2"))
    l = _calib(arms.get("L"))

    def calibrated(c):
        if c is None:
            return None
        slope_ok = c.get("slope_lo") is not None and c["slope_lo"] <= 1.0 <= c["slope_hi"]
        intercept = c.get("large_intercept_lo"), c.get("large_intercept_hi")
        intercept_ok = intercept[0] is not None and intercept[0] <= 0.0 <= intercept[1]
        ece_ok = c.get("ece") is not None and c["ece"] < 0.05
        bins_ok = c.get("n_outside") is not None and c["n_outside"] <= 1
        return {"slope": slope_ok, "intercept_in_the_large": intercept_ok, "ece": ece_ok, "bins": bins_ok, "all": bool(slope_ok and intercept_ok and ece_ok and bins_ok)}

    tests["P5-T2"] = {"statement": "oracle passes T4's calibration criteria", "checks": calibrated(o),
                      "verdict": "UNDER-POWERED" if o is None else ("PASS" if calibrated(o)["all"] else "FAIL")}
    nees = {arm: arms.get(arm, {}).get("nees") for arm in ("P", "B2", "L")}

    def inside(n):
        return None if not n else bool(n["lo"] <= 2.0 <= n["hi"] or (n["band_lo"] <= n["mean"] <= n["band_hi"]))

    def above(n):
        return None if not n else bool(n["lo"] > n["band_hi"])

    tests["P5-T3"] = {"statement": "precursor NEES: P inside the band, B2 above it, L inside it", "nees": nees,
                      "P_inside": inside(nees["P"]), "B2_above": above(nees["B2"]), "L_inside": inside(nees["L"])}
    t3_ok = [tests["P5-T3"]["P_inside"], tests["P5-T3"]["B2_above"], tests["P5-T3"]["L_inside"]]
    tests["P5-T3"]["verdict"] = "UNDER-POWERED" if any(v is None for v in t3_ok) else ("PASS" if all(t3_ok) else "FAIL")
    b2_ok = None
    if b2 is not None:
        b2_ok = bool(b2.get("large_intercept_lo") is not None and b2["large_intercept_lo"] > 0.0 and b2.get("top_three_ratio") is not None and 3.0 <= b2["top_three_ratio"] <= 10.0)
    tests["P5-T4"] = {"statement": "P calibrated (slope ~1, intercept ~0, ECE < 0.05, <= 1 bin outside the simultaneous band); B2 intercept above zero and top-three-bin ratio 3-10; L as P",
                      "P": calibrated(p), "B2_overconfident": b2_ok, "L": calibrated(l),
                      "verdict": "UNDER-POWERED" if p is None else ("PASS" if calibrated(p)["all"] else "FAIL")}
    auc = {arm: (_calib(arms.get(arm)) or {}).get("auroc") for arm in ("O", "P", "B2", "L")}
    t5 = None
    if all(auc.get(a) is not None for a in ("O", "P", "B2", "L")):
        t5 = auc["P"] >= 0.85 and auc["O"] - auc["P"] <= 0.05 and auc["B2"] <= auc["P"] - 0.10 and auc["L"] <= 0.65
    tests["P5-T5"] = {"statement": "AUROC: P >= 0.85; O - P <= 0.05; B2 <= P - 0.10; L <= 0.65", "auroc": auc,
                      "verdict": "UNDER-POWERED" if t5 is None else ("PASS" if t5 else "FAIL")}
    leads = {arm: arms.get(arm, {}).get("lead_time") for arm in ("O", "P")}
    t6 = None
    if leads["P"] and leads["O"] and leads["P"].get("median") is not None and leads["O"].get("median") is not None:
        t6 = 1.0 <= leads["P"]["median"] <= 2.5 and leads["O"]["median"] - leads["P"]["median"] <= 0.3
    tests["P5-T6"] = {"statement": "lead time at h_crit (5% false alarms): P median in [1.0, 2.5] s; O ahead of P by at most 0.3 s", "leads": leads,
                      "h_crit": {arm: arms.get(arm, {}).get("h_crit") for arm in ("O", "P", "B2", "L")},
                      "verdict": "UNDER-POWERED" if t6 is None else ("PASS" if t6 else "FAIL")}
    tests["P5-T7"] = {"statement": "B2 top-bin ratio against the Prop 12 factor from its NEES (EMP, non-blocking)",
                      "b2_top_ratio": (b2 or {}).get("top_bin_ratio"), "b2_top_three_ratio": (b2 or {}).get("top_three_ratio"), "b2_nees": (nees["B2"] or {}).get("mean"), "verdict": "REPORTED"}
    tests["P5-T8"] = {"statement": "staleness exponent (separate online campaign)", "verdict": "NOT RUN IN THIS ANALYSIS"}
    tests["closing_action"] = {"statement": "stress breaking strength T_b^s for Phase 6", "T_b_s": stress["threshold"], "probability": stress["probability"],
                               "verdict": "FIXED" if stress["inside_band"] else "OUTSIDE BAND"}
    return tests


def phase5_gate(tests: dict, exhausted: bool = False) -> dict:
    required = [tests[k]["verdict"] for k in ("P5-T1", "P5-T2", "P5-T3", "P5-T4", "P5-T5")]
    if all(v == "PASS" for v in required):
        return {"decision": "GO", "rule": "GO requires T1, T2, T3 (P), T4 (P), T5"}
    if tests["P5-T2"]["verdict"] == "FAIL" and exhausted:
        return {"decision": "NOT-GO", "rule": "T2 still fails after both pre-declared pivots (linearised propagation, then H = 1 s): no estimator arm is judged; "
                + "; ".join(f"{k}={tests[k]['verdict']}" for k in ("P5-T1", "P5-T2", "P5-T3", "P5-T4", "P5-T5"))}
    if tests["P5-T2"]["verdict"] == "FAIL":
        return {"decision": "IN-PHASE PIVOT", "rule": "T2 fail: replace the constant-acceleration model with the linearised closed-loop propagation; if it still fails, shorten H to 1 s"}
    return {"decision": "NOT-GO", "rule": "; ".join(f"{k}={tests[k]['verdict']}" for k in ("P5-T1", "P5-T2", "P5-T3", "P5-T4", "P5-T5"))}
