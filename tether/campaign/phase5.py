"""Phase 5: estimation, hazard, and staleness (plan Part V, Phase 5).

Commands (in order): ``impact-fan`` measures the fan formation's impact table (the Phase 1
pivot v_b = Z^-1(T_b) for the Phase 4-6 geometry); ``squall`` calibrates the squall level
empirically; ``missions`` runs the calibration campaign in recording mode and caches every
seed's sensor streams and truth; ``analyse`` runs the arms offline, evaluates hazards and
P5-T0..T7 and fixes the stress threshold; ``staleness`` runs the online staleness campaign.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace

import numpy as np

from tether.campaign.common import RECORDS, json_bytes, run_pool, source_state, write_bytes

RECORD_DIR = RECORDS / "phase5"
FAN_IMPACT_PATH = RECORD_DIR / "impact_table_fan.json"
SQUALL_PATH = RECORD_DIR / "squall_calibration.json"
FAN_ARC = 0.55
FAN_PRETENSION = 1000.0
SQUALL_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
SQUALL_WINDOW = (50.0, 75.0)
PLAN_LEVEL = 1.5

SQUALL_DECLARATION = (
    "lambda is expressed in units of the deep-excursion criterion measured in the plant: lambda = 1 is the smallest squall level "
    "(in the mission module's quasi-static units) at which a cable goes slack during the squall in the deterministic mission with "
    "background weather off; the calibration campaign runs at lambda = 1.5 of that level. The quasi-static linear calibration "
    "does not make any cable slack, because a broadside push reorients the formation instead of unloading the windward cable "
    "(the restatement with the measured effective restoring load that pre-mortem 1 prescribes)."
)


def impact_fan(workers: int) -> dict:
    from tether.campaign.phase1_close import IMPACT_SPEEDS, ImpactJob, run_impact_job

    jobs = [ImpactJob(FAN_PRETENSION, cable, speed, formation="fan", arc_half_angle=FAN_ARC) for cable in (0, 1, 2) for speed in IMPACT_SPEEDS]
    points = run_pool(run_impact_job, jobs, workers)
    table = {}
    for cable in (0, 1, 2):
        rows = sorted((p["speed"], p["peak"]) for p in points if p["cable"] == cable)
        speeds = np.array([r[0] for r in rows])
        peaks = np.array([r[1] for r in rows])
        high = speeds >= 1.0
        slope = float(np.dot(speeds[high], peaks[high]) / np.dot(speeds[high], speeds[high]))
        residual = peaks[high] - slope * speeds[high]
        table[f"T{int(FAN_PRETENSION)}_c{cable}"] = {
            "pretension": FAN_PRETENSION, "cable": cable, "formation": "fan", "arc_half_angle": FAN_ARC,
            "speeds": speeds.tolist(), "peaks": peaks.tolist(), "asymptotic_impedance": slope,
            "asymptotic_r2": float(1.0 - np.dot(residual, residual) / np.dot(peaks[high], peaks[high])),
            "zero_speed_peak": float(peaks[speeds == 0.0][0]), "monotone": bool(np.all(np.diff(peaks) > 0.0)),
        }
    write_bytes(FAN_IMPACT_PATH, json_bytes(table))
    return table


def _squall_probe(level: float) -> dict:
    from tether.campaign.mission import MissionSpec, run_mission

    spec = replace(MissionSpec(), squall_level=level, weather_scale=0.0, duration=SQUALL_WINDOW[1])
    run = run_mission(spec, 1)
    log = run.fleet.cables.log
    time = log.event_time[: log.count]
    window = (time >= SQUALL_WINDOW[0]) & (time <= SQUALL_WINDOW[1])
    elongation = log.elongation[: log.count][window]
    slack = bool(np.any(elongation <= 0.0))
    return {"level": level, "slack": slack, "min_elongation": float(np.min(elongation)) if elongation.size else None,
            "slack_cables": sorted({int(c) for c in np.flatnonzero(np.any(elongation <= 0.0, axis=0))}),
            "marks": [(m.cable, m.t_up, m.depth, m.T_peak) for m in run.fleet.cables.reengagements],
            "closure": run.fleet.cables.closure, "end_time": float(run.simulator.get_context().get_time())}


def calibrate_squall(workers: int) -> dict:
    """Coarse grid, then bisection, for the smallest level that slackens a cable."""
    grid = run_pool(_squall_probe, list(SQUALL_GRID), workers)
    slack_levels = [g["level"] for g in grid if g["slack"]]
    result = {"declaration": SQUALL_DECLARATION, "grid": grid, "source": source_state()}
    if not slack_levels:
        result.update({"threshold_level": None, "campaign_level": None})
    else:
        high = min(slack_levels)
        lows = [g["level"] for g in grid if not g["slack"] and g["level"] < high]
        low = max(lows) if lows else 0.0
        bisection = []
        for _ in range(5):
            middle = 0.5 * (low + high)
            probe = _squall_probe(middle)
            bisection.append(probe)
            if probe["slack"]:
                high = middle
            else:
                low = middle
        result.update({"bisection": bisection, "threshold_level": high, "campaign_level": PLAN_LEVEL * high})
    write_bytes(SQUALL_PATH, json_bytes(result))
    return result


# ----------------------------------------------------------------------------- calibration campaign

CAL_SEEDS = tuple(range(5001, 5041))
TAU = 0.4
ARMS = ("P", "B2", "L")
MISSION_CACHE = RECORD_DIR / "cache" / "phase5_missions.pkl"
RESULTS_PATH = RECORD_DIR / "phase5_results.json"
STRESS_PROBABILITY = 0.45
DECLARATIONS = {
    "mission": "Squall Passage per records/phase5/phase5_spec.md with the mission module's quasi-static squall calibration at lambda = 1.5 (4.6 kN per vessel), common-mode t3 background weather at intensity 1.0, fan formation 0.55 rad, T0 = 1 kN, k_h = 500; no retuning (see the spec addendum)",
    "arms": "P, B2, L run offline on each mission's logged sensor streams (exactly equivalent in recording mode without a supervisor), tau = 0.4 s on the cycle graph; O from plant truth",
    "oracle_slack": "the oracle's slack ticks are the ticks where the true elongation is <= 0; its onset is the true downcrossing time",
    "stress_threshold": "T_b^s = the 55th percentile over the 40 missions of the per-mission maximum engagement or taut tension, rounded to 100 N, so the unsupervised per-mission virtual first-severance probability is 0.45 (inside [0.3, 0.6])",
    "critical_speed": "v_b = Z^-1(T_b^s) per cable position from the fan-formation impact table (records/phase5/impact_table_fan.json)",
    "l_growth": "arm L's precursor growth q_L calibrated so its mean per-interval precursor NEES over the 40 missions equals 2 (declared calibration of IV.8)",
    "calibration": "per arm, slack ticks pooled over vessels and seeds; clusters = slack intervals; judged on the calibration-in-the-large intercept and the joint slope, ECE, the Bonferroni reliability band; B2's 'intercept above zero' on the in-the-large intercept",
}


def mission_job(payload) -> dict:
    from tether.campaign.mission import MissionSpec, mission_outcome, run_mission
    from tether.estimation.replay import monitor_truth, replay_bundle, run_arms
    from tether.physics import constants

    seed, centre = payload
    spec = MissionSpec()
    run = run_mission(spec, seed)
    log, truth = replay_bundle(run)
    series_time = truth.event_time
    e = truth.elongation
    edot = truth.rate
    q = np.where(e > 0.0, constants.CABLE_STIFFNESS * e + constants.CABLE_DAMPING * edot, 0.0)
    outcome = mission_outcome(run, spec, np.asarray(centre))
    del monitor_truth, run_arms
    return {
        "seed": seed,
        "sensor_log": log,
        "truth": truth,
        "series_time": series_time,
        "elongation": e,
        "rate": edot,
        "max_tension": float(np.max(q)) if q.size else 0.0,
        "max_tension_time": float(series_time[int(np.unravel_index(np.argmax(q), q.shape)[0])]) if q.size else None,
        "marks": [(m.cable, m.t_up, m.depth, m.v_return, m.T_peak, m.dwell) for m in run.fleet.cables.reengagements],
        "outcome": outcome,
    }


def arms_job(mission: dict) -> dict:
    """Run the estimator arms offline on one cached mission (recording mode)."""
    from dataclasses import replace as dataclass_replace

    from tether.campaign.mission import MissionSpec, initial_state
    from tether.estimation.replay import monitor_truth, run_arms

    # Surveyed poses are the exact t = 0 state (the online adapter's convention); the
    # 10 ms state log's first entry is already one physics step in.
    poses = initial_state(MissionSpec())[: 3 * 6].reshape(-1, 3)
    log = dataclass_replace(mission["sensor_log"], initial_poses=poses)
    outputs = run_arms(log, arms=ARMS, tau=TAU, master_seed=mission["seed"])
    result = dict(mission)
    result["outputs"] = outputs
    result["monitor_truth"] = monitor_truth(mission["truth"], outputs["P"].time)
    result.pop("sensor_log", None)
    result.pop("truth", None)
    return result


def run_missions(workers: int) -> None:
    import pickle

    from tether.campaign.mission import MissionSpec, docking_reference

    centre = docking_reference(MissionSpec()).centre.tolist()
    results = run_pool(mission_job, [(seed, centre) for seed in CAL_SEEDS], workers)
    MISSION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with MISSION_CACHE.open("wb") as handle:
        pickle.dump(results, handle)


# ----------------------------------------------------------------------------- staleness campaign

STALE_TAUS = (0.1, 0.2, 0.4, 0.8, 1.6)
STALE_ARMS = ("P", "B2")
STALE_SEEDS = tuple(range(5101, 5121))
STALE_CACHE = RECORD_DIR / "cache" / "phase5_staleness.pkl"
STALE_DURATION = 600.0
STALE_WARMUP = 20.0


def staleness_spec():
    from tether.campaign.fleet_run import FleetRunSpec

    return FleetRunSpec(pretension=1000.0, heading_gain=500.0, weather_scale=1.0, duration=STALE_DURATION, warmup=STALE_WARMUP)


def estimated_trim_hook(arm: str, tau: float, seed: int):
    """pre_build hook: the controller's cable trim reads the arm's estimated bearing."""

    def attach(builder, fleet, sensors, controller):
        from tether.estimation.adapter import EstimatorAdapter
        from tether.estimation.replay import estimator_geometry
        from tether.physics.fleet import equilibrium_state

        n = fleet.geometry.vessel_count
        poses = equilibrium_state(fleet.geometry, fleet.operating)[: 3 * (n + 1)].reshape(-1, 3)
        adapter = builder.AddSystem(EstimatorAdapter(arm, estimator_geometry(fleet.geometry), poses, tau, seed, np.zeros(n), compute_hazard=False))
        adapter.set_name(f"estimator_adapter_{arm}")
        for i, suite in enumerate(sensors):
            builder.Connect(suite.GetOutputPort("odometry"), adapter.get_input_port(i))
            builder.Connect(suite.GetOutputPort("cable_bearing"), adapter.get_input_port(n + i))
            builder.Connect(suite.GetOutputPort("tension"), adapter.get_input_port(2 * n + i))
            builder.Disconnect(suite.GetOutputPort("cable_bearing"), controller.get_input_port(n + i))
            builder.Connect(adapter.GetOutputPort(f"estimated_bearing_{i}"), controller.get_input_port(n + i))
        builder.Connect(sensors[0].GetOutputPort("beacon"), adapter.get_input_port(3 * n))
        return {"adapter": adapter}

    return attach


def staleness_job(payload) -> dict:
    from tether.campaign.fleet_run import build_run, run_to_end
    from tether.campaign.summaries import summarize_run

    arm, tau, seed = payload
    spec = staleness_spec()
    try:
        run = run_to_end(build_run(spec, seed, pre_build=estimated_trim_hook(arm, tau, seed)))
    except Exception as error:  # one failed run is recorded and excluded, not fatal to the campaign
        return {"job": {"arm": arm, "tau": tau, "seed": seed}, "error": f"{type(error).__name__}: {error}"}
    summary = summarize_run(run, spec.warmup, spec.pretension)
    summary["job"] = {"arm": arm, "tau": tau, "seed": seed}
    summary.pop("samples", None)
    return summary


def run_staleness(workers: int) -> None:
    import pickle

    payloads = [(arm, tau, seed) for arm in STALE_ARMS for tau in STALE_TAUS for seed in STALE_SEEDS]
    results = run_pool(staleness_job, payloads, workers, progress=lambda done, total: print(f"staleness: {done}/{total} runs", flush=True) if done % 20 == 0 or done == total else None)
    STALE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with STALE_CACHE.open("wb") as handle:
        pickle.dump(results, handle)


def analyse_staleness(common_threshold: float) -> dict:
    """P5-T8: log Lambda_snap = a + b tau^p per arm, with a seed bootstrap for p."""
    import pickle

    from scipy.optimize import curve_fit

    with STALE_CACHE.open("rb") as handle:
        results = pickle.load(handle)
    rows = {}
    rng = np.random.default_rng(20260915)
    for arm in STALE_ARMS:
        counts = {tau: [] for tau in STALE_TAUS}
        exposures = {tau: [] for tau in STALE_TAUS}
        for item in results:
            job = item["job"]
            if job["arm"] != arm or "error" in item:
                continue
            counts[job["tau"]].append(int(np.sum(item["marks"]["T_peak"] > common_threshold)))
            exposures[job["tau"]].append(item["meta"]["exposure"])
        taus = np.array(STALE_TAUS)
        rates = np.array([sum(counts[t]) / max(sum(exposures[t]), 1e-9) for t in STALE_TAUS])

        def model(tau, a, b, p):
            return a + b * tau**p

        fit = None
        if np.all(rates > 0):
            try:
                params, _ = curve_fit(model, taus, np.log(rates), p0=(np.log(rates[0]), 0.1, 1.0), maxfev=20000)
                boot = []
                for _ in range(400):
                    draw = [rng.integers(0, len(counts[t]), len(counts[t])) for t in STALE_TAUS]
                    boot_rates = np.array([np.sum(np.array(counts[t])[d]) / np.sum(np.array(exposures[t])[d]) for t, d in zip(STALE_TAUS, draw)])
                    if np.all(boot_rates > 0):
                        try:
                            boot.append(curve_fit(model, taus, np.log(boot_rates), p0=params, maxfev=20000)[0][2])
                        except RuntimeError:
                            pass
                lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) >= 50 else (np.nan, np.nan))
                fit = {"a": float(params[0]), "b": float(params[1]), "p": float(params[2]), "p_lo": float(lo), "p_hi": float(hi),
                       "half_width": float((hi - lo) / 2) if np.isfinite(lo) else None, "bootstraps": len(boot)}
            except RuntimeError:
                fit = None
        rows[arm] = {"taus": list(STALE_TAUS), "rates": rates.tolist(), "counts": {str(t): sum(counts[t]) for t in STALE_TAUS}, "fit": fit}
    ok = [r["fit"] is not None and r["fit"]["half_width"] is not None and r["fit"]["half_width"] < 0.2 and r["fit"]["b"] > 0 for r in rows.values()]
    failed = [dict(item["job"], error=item["error"]) for item in results if "error" in item]
    return {"statement": "log Lambda_snap = a + b tau^p: half-width of p below 0.2 and b > 0 (non-blocking)", "threshold": common_threshold, "rows": rows, "failed_runs": failed,
            "verdict": "PASS" if all(ok) else ("UNDER-POWERED" if any(r["fit"] is None for r in rows.values()) else "FAIL")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("impact-fan", "squall", "missions", "analyse", "regate", "staleness", "staleness-analyse"))
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.command == "impact-fan":
        for key, value in impact_fan(arguments.workers).items():
            print(key, [round(p) for p in value["peaks"]], round(value["asymptotic_impedance"]), round(value["asymptotic_r2"], 5), value["monotone"])
    elif arguments.command == "squall":
        result = calibrate_squall(arguments.workers)
        for g in result["grid"]:
            print(g["level"], g["slack"], g["slack_cables"], g["min_elongation"], g["closure"])
        print("threshold", result["threshold_level"], "campaign", result["campaign_level"])
    elif arguments.command == "missions":
        run_missions(arguments.workers)
    elif arguments.command == "analyse":
        from tether.campaign.phase5_analysis import analyse

        results = analyse()
        for test_id, test in results["tests"].items():
            print(test_id, test["verdict"])
        print("GATE", results["gate"])
    elif arguments.command == "regate":
        from tether.campaign.phase5_analysis import regate

        results = regate()
        for test_id, test in results["tests"].items():
            print(test_id, test["verdict"])
        print("GATE", results["gate"])
    elif arguments.command == "staleness":
        run_staleness(arguments.workers)
    elif arguments.command == "staleness-analyse":
        predictions = json.loads((RECORDS / "phase2" / "phase2_predictions.json").read_text())
        result = analyse_staleness(float(predictions["constants"]["common_threshold"]))
        write_bytes(RECORD_DIR / "phase5_staleness_results.json", json_bytes(result))
        print(result["verdict"], json.dumps(result["rows"], default=str)[:2000])


if __name__ == "__main__":
    main()
