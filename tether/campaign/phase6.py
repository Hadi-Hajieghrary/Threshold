"""Phase 6: the capstone (plan Part V, Phase 6).

Five arms {N, L, B2, P, O} on the Squall Passage, 60 seeds, common random numbers, with
h_crit and the stress threshold T_b^s fixed in Phase 5.  ``compute`` runs the paired
recording campaign and the live campaign; ``analyse`` evaluates P6-T1..T9.
"""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

from tether.campaign.common import RECORDS, json_bytes, run_pool, source_state, write_bytes

RECORD_DIR = RECORDS / "phase6"
CACHE_PATH = RECORD_DIR / "cache" / "phase6_compute.pkl"
RESULTS_PATH = RECORD_DIR / "phase6_results.json"
PHASE5_RESULTS = RECORDS / "phase5" / "phase5_results.json"
FAN_IMPACT_PATH = RECORDS / "phase5" / "impact_table_fan.json"
ARMS = ("N", "L", "B2", "P", "O")
SEEDS = tuple(range(6001, 6061))
MODES = ("recording", "live")
TAU = 0.4
HORIZON = 2.0

DECLARATIONS = {
    "arms": "N: no monitor, no supervisor; L, B2, P: the online EstimatorAdapter (same core as Phase 5) feeding the hazard supervisor; O: the OracleAdapter (declared lint exemption)",
    "supervisor": "F_T,i <- F_T,i (1 - 0.6 sat(h/h_crit)) with the fleet term on the largest hazard heard over the cycle (one tau per hop), h_crit per arm fixed in Phase 5",
    "virtual_severance": "recording mode: the first time any cable's tension (k e + c edot while taut, 1 ms samples) exceeds T_b^s",
    "live": "cable mode live with break threshold T_b^s; first severance from the cable bank's rupture events",
    "cost": "mission time and docking error from the mission module, paired against arm N on the recording campaign",
    "false_alarm": "an action episode (thrust scale < 1 on a vessel) is false when no re-engagement faster than 0.5 v_b follows within H of its start; rate over squall-time action episodes",
    "h_crit_floor": "h_crit is floored at 1/2048, the hazard's Monte Carlo resolution: a Phase 5 h_crit of 0 (fewer than 5% of benign intervals ever show a positive hazard) means alarm at any positive hazard, so the supervisor eases fully at the first non-zero hazard",
    "closure": "a run that ends in formation closure (a chord below 1 m) before any severance counts as no severance in the plan's indicator; closures and the share of missions ending in severance or closure are reported beside it, and the hero seed must complete its P mission",
    "alarm_rate": "fraction of supervisor ticks (10 Hz) on which any vessel's thrust scale is below 1, up to the first live severance, averaged over the arm's missions (the alternative L criterion of P6-T1)",
    "powered_thresholds": "per-mission virtual first-severance probability at every level of POWERED_LEVELS from each run's maximum cable tension",
    "extrapolation": "P6-T6: per arm, the pooled re-engagement peaks (marks) and declustered taut q-peaks (Phase 2 definition) are GPD-fitted above their 90th percentile and the exceedance rate extrapolated to 25 kN with the profile-likelihood interval; the extrapolation is withheld unless the fitted rate lies inside the Garwood 95% interval at every powered level (>= 5 exceedances) above the threshold; exposure is the summed simulated mission time",
    "post_severance": "P6-T8: on live runs with a severance, the estimated load positions of the vessels; the severed vessel's offset from the survivors' mean (the component the reduced graph no longer ties to a cable channel), the survivors' internal spread, and the survivors' mean error against truth, medians over [t_s - 5, t_s) and [t_s + 2, t_s + 12] s",
}
H_CRIT_FLOOR = 1.0 / 2048.0
SQUALL_WINDOW = (50.0, 70.0)
POWERED_LEVELS = (4000.0, 5000.0, 6000.0, 8000.0, 10000.0, 12000.0, 16000.0, 20000.0, 25000.0)
OPERATIONAL_LEVEL = 25000.0
HERO_PATH = RECORD_DIR / "phase6_hero.npz"


def _phase5() -> dict:
    return json.loads(PHASE5_RESULTS.read_text())


def _h_crit(phase5: dict, arm: str) -> float:
    value = (phase5["arms"].get(arm) or {}).get("h_crit")
    if value is None:
        raise SystemExit(f"Phase 5 fixed no h_crit for arm {arm}")
    return max(float(value), H_CRIT_FLOOR)


def supervisor_hook(arm: str, seed: int, h_crit: float, critical_speeds: np.ndarray, monitor_config: dict):
    """``monitor_config``: the hazard model, horizon and arm-L growth q_L fixed in Phase 5."""

    def attach(builder, fleet, sensors, controller):
        from dataclasses import replace

        from tether.control.supervisor import Supervisor
        from tether.estimation.adapter import EstimatorAdapter, OracleAdapter
        from tether.estimation.core import EstimatorParameters
        from tether.estimation.replay import estimator_geometry

        n = fleet.geometry.vessel_count
        model, horizon = monitor_config["model"], monitor_config["horizon"]
        if arm == "O":
            monitor = builder.AddSystem(OracleAdapter(fleet.geometry, fleet.cables.cable.rest_length, 1000.0, seed, critical_speeds,
                                                      horizon=horizon, hazard_model=model))
            builder.Connect(fleet.plant.get_state_output_port(), monitor.get_input_port(0))
        else:
            from tether.campaign.mission import MissionSpec, initial_state

            poses = initial_state(MissionSpec())[: 3 * (n + 1)].reshape(-1, 3)
            parameters = replace(EstimatorParameters(), precursor_growth=monitor_config["l_growth"]) if arm == "L" else None
            monitor = builder.AddSystem(EstimatorAdapter(arm, estimator_geometry(fleet.geometry), poses, TAU, seed, critical_speeds,
                                                         parameters=parameters, horizon=horizon, hazard_model=model))
            monitor.set_name(f"estimator_adapter_{arm}")
            for i, suite in enumerate(sensors):
                builder.Connect(suite.GetOutputPort("odometry"), monitor.get_input_port(i))
                builder.Connect(suite.GetOutputPort("cable_bearing"), monitor.get_input_port(n + i))
                builder.Connect(suite.GetOutputPort("tension"), monitor.get_input_port(2 * n + i))
            builder.Connect(sensors[0].GetOutputPort("beacon"), monitor.get_input_port(3 * n))
        supervisor = builder.AddSystem(Supervisor(n, h_crit, TAU))
        supervisor.set_name(f"supervisor_{arm}")
        builder.Connect(monitor.GetOutputPort("hazard"), supervisor.get_input_port(0))
        builder.Connect(supervisor.get_output_port(0), controller.GetInputPort("thrust_scale"))
        return {"monitor": monitor, "supervisor": supervisor}

    return attach


def capstone_job(payload) -> dict:
    from tether.campaign.fleet_run import run_to_end
    from tether.campaign.mission import MissionSpec, build_mission, mission_outcome
    from tether.physics import constants

    arm, seed, mode, breaking, h_crit, critical, centre, monitor_config = payload
    spec = MissionSpec(cable_mode=mode, break_threshold=breaking if mode == "live" else None)
    hook = None if arm == "N" else supervisor_hook(arm, seed, h_crit, critical, monitor_config)
    run = build_mission(spec, seed, hook, supervised=arm != "N", exemptions=frozenset({"oracle_adapter"}) if arm == "O" else frozenset())
    run_to_end(run)
    log = run.fleet.cables.log
    time = log.event_time[: log.count]
    e = log.elongation[: log.count]
    edot = log.rate[: log.count]
    alive = log.alive[: log.count]
    q = np.where((e > 0.0) & alive, constants.CABLE_STIFFNESS * e + constants.CABLE_DAMPING * edot, 0.0)
    peaks = _taut_peaks(time, e, q, alive, run.fleet.cables.reengagements)
    over = np.argwhere(q > breaking)
    virtual = (float(time[over[0, 0]]), int(over[0, 1])) if over.size else None
    severances = list(run.fleet.cables.severances)
    first_live = min(severances, key=lambda s: s[1]) if severances else None
    outcome = mission_outcome(run, spec, np.asarray(centre))
    extras = run.fleet.extras
    supervisor_log = [(t, h.tolist(), s.tolist()) for t, h, s in extras["supervisor"].log[::10]] if "supervisor" in extras else []
    return {
        "arm": arm, "seed": seed, "mode": mode,
        "virtual_first": virtual, "live_first": (float(first_live[1]), int(first_live[0])) if first_live else None,
        "outcome": outcome.__dict__ if hasattr(outcome, "__dict__") else outcome,
        "marks": [(m.cable, m.t_up, m.depth, m.v_return, m.T_peak, m.dwell, m.u_entry) for m in run.fleet.cables.reengagements],
        "supervisor_log": supervisor_log,
        "max_tension": float(q.max()) if q.size else 0.0,
        "taut_peaks": peaks,
        "estimates": _estimates(run) if mode == "live" and first_live and arm in ("L", "B2", "P") else None,
        "sim_seconds": float(run.simulator.get_context().get_time()),
        "wall_seconds": run.wall_seconds,
    }


def _taut_peaks(time, e, q, alive, reengagements) -> np.ndarray:
    """Declustered taut q-peaks above 1.2 T0, outside the 1 s after each re-engagement (the
    Phase 2 taut-peak definition), pooled over the cables."""
    from tether.campaign.summaries import ENGAGEMENT_PERIOD, SNAP_WINDOW, TAUT_PEAK_FLOOR_RATIO, cluster_maxima
    from tether.physics.fleet import EVENT_PERIOD

    clean = (e > 0.0) & alive
    for mark in reengagements:
        lo, hi = np.searchsorted(time, [mark.t_up, mark.t_up + SNAP_WINDOW])
        clean[lo:hi, mark.cable] = False
    run_samples = int(round(ENGAGEMENT_PERIOD / EVENT_PERIOD))
    peaks = [cluster_maxima(time, q[:, cable], clean[:, cable], TAUT_PEAK_FLOOR_RATIO * 1000.0, run_samples)[1] for cable in range(q.shape[1])]
    return np.concatenate(peaks).astype(np.float32) if peaks else np.empty(0, dtype=np.float32)


def _estimates(run) -> dict:
    """P6-T8 record: every vessel's estimated load pose and the true load pose at 10 Hz."""
    rows = run.fleet.extras["monitor"].estimator._rows
    time = np.array([t for t, _ in rows])
    log = run.fleet.cables.log
    state_time = log.state_time[: log.state_count]
    index = np.clip(np.searchsorted(state_time, time - 1.0e-9), 0, max(log.state_count - 1, 0))
    return {"time": time.astype(np.float32), "load_mean": np.stack([row["load_mean"][:, :3] for _, row in rows]).astype(np.float32),
            "load_true": log.state[index, :3].astype(np.float32) if log.state_count else None}


def compute(workers: int) -> None:
    from tether.monitor.hazard import critical_speeds

    phase5 = _phase5()
    breaking = float(phase5["stress"]["threshold"])
    critical = critical_speeds(breaking, 1000.0, table_path=FAN_IMPACT_PATH)
    h_crit = {arm: _h_crit(phase5, arm) for arm in ("L", "B2", "P", "O")}
    from tether.campaign.mission import MissionSpec, docking_reference

    centre = docking_reference(MissionSpec()).centre.tolist()
    monitor_config = {"model": phase5.get("prediction_model", "constant_acceleration"), "horizon": float(phase5.get("horizon", HORIZON)),
                      "l_growth": float(phase5.get("l_growth") or 0.0)}
    payloads = [(arm, seed, mode, breaking, h_crit.get(arm, 1.0), critical, centre, monitor_config) for mode in MODES for arm in ARMS for seed in SEEDS]
    results = run_pool(capstone_job, payloads, workers, progress=_progress)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("wb") as handle:
        pickle.dump({"breaking": breaking, "h_crit": h_crit, "critical": np.asarray(critical).tolist(), "monitor": monitor_config,
                     "results": results}, handle)


def _progress(done: int, total: int) -> None:
    if done % 25 == 0 or done == total:
        print(f"phase6: {done}/{total} missions", flush=True)


def _paired(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float]:
    from tether.evt.bootstrap import paired_bootstrap_difference

    return paired_bootstrap_difference(a, b, n_boot=4000, rng=rng)


def _alarm_rate(item: dict) -> float:
    """Fraction of supervisor ticks with any vessel eased, up to the first live severance (a
    severed cable stays slack, so its monitor alarms for the rest of the mission)."""
    end = item["live_first"][0] if item.get("live_first") else np.inf
    log = [scale for t, _, scale in item["supervisor_log"] if t < end]
    return float(np.mean([min(scale) < 1.0 - 1e-9 for scale in log])) if log else 0.0


def tail_extrapolation(values, exposure: float, level: float = OPERATIONAL_LEVEL) -> dict:
    """P6-T6 for one population: GPD above the 90th percentile, rate at ``level`` with its
    profile interval, withheld unless the fit sits inside the Poisson intervals at every
    powered level above the threshold."""
    from tether.evt.gpd import fit_gpd, profile_tail_rate_interval, tail_rate
    from tether.evt.poisson import rate_interval

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 50 or not exposure > 0.0:
        return {"n": int(values.size), "withheld": True, "reason": "fewer than 50 peaks"}
    threshold = float(np.quantile(values, 0.9))
    rate_u = float(np.sum(values > threshold)) / exposure
    fit = fit_gpd(values, threshold)
    checks = []
    for x in np.unique(np.quantile(values[values > threshold], [0.0, 0.25, 0.5, 0.75, 0.9])):
        count = int(np.sum(values > x))
        if count < 5:
            continue
        rate, lo, hi = rate_interval(count, exposure)
        fitted = float(tail_rate(fit, rate_u, x))
        checks.append({"level": float(x), "count": count, "rate": rate, "lo": lo, "hi": hi, "fitted": fitted, "inside": bool(lo <= fitted <= hi)})
    inside = bool(checks) and all(c["inside"] for c in checks)
    estimate = float(tail_rate(fit, rate_u, level))
    lo, hi = profile_tail_rate_interval(values, threshold, rate_u, level)
    return {"n": int(values.size), "threshold": threshold, "exceedances": int(fit.n_exceedances), "shape": float(fit.shape), "scale": float(fit.scale),
            "max_observed": float(values.max()), "rate_at_level": estimate if inside else None, "interval": [float(lo), float(hi)] if inside else None,
            "fitted_rate_unchecked": estimate, "withheld": not inside, "checks": checks}


def false_alarm_episodes(log: list, marks: list, critical: np.ndarray, horizon: float, start: float, stop: float) -> tuple[int, int]:
    """Action episodes (a vessel's thrust scale dropping below 1) starting in [start, stop),
    and those with no re-engagement of that cable above 0.5 v_b within ``horizon``."""
    episodes = false = 0
    for vessel in range(len(critical)):
        active = False
        for t, _hazard, scale in log:
            acting = scale[vessel] < 1.0 - 1e-9
            if acting and not active and start <= t < stop:
                episodes += 1
                fast = [m for m in marks if m[0] == vessel and t <= m[1] <= t + horizon and m[3] > 0.5 * critical[vessel]]
                false += int(not fast)
            active = acting
    return episodes, false


def mechanism_attribution(items: list[dict], min_depth: float = 0.05) -> dict | None:
    """P6-T7 for one arm: deep excursions (depth > 5 cm), their return-leg mean acceleration
    a_bar = (v_up^2 - u_entry^2) / 2 d, depth, and the vessel's mean commanded thrust scale
    over the slack interval (1 for an unsupervised arm)."""
    depth, a_bar, scale = [], [], []
    for item in items:
        log = item["supervisor_log"]
        times = np.array([t for t, _, _ in log]) if log else np.empty(0)
        scales = np.array([sc for _, _, sc in log]) if log else np.empty((0, 5))
        for cable, t_up, d, v_up, _peak, dwell, u_entry in item["marks"]:
            if d <= min_depth:
                continue
            depth.append(d)
            a_bar.append((v_up**2 - u_entry**2) / (2.0 * d))
            window = (times >= t_up - dwell) & (times <= t_up)
            scale.append(float(scales[window, int(cable)].mean()) if window.any() else 1.0)
    if not depth:
        return None
    return {"deep_excursions": len(depth), "median_depth": float(np.median(depth)), "median_a_bar": float(np.median(a_bar)),
            "mean_thrust_scale_during_slack": float(np.mean(scale)), "median_thrust_scale_during_slack": float(np.median(scale))}


def post_severance(item: dict) -> dict | None:
    """P6-T8 statistics of one live run with a severance."""
    estimates = item.get("estimates")
    if not estimates:
        return None
    t_sev, cable = item["live_first"]
    time = np.asarray(estimates["time"], dtype=float)
    position = np.asarray(estimates["load_mean"], dtype=float)[:, :, :2]
    survivors = [i for i in range(position.shape[1]) if i != cable]
    centre = position[:, survivors].mean(axis=1)
    offset = np.linalg.norm(position[:, cable] - centre, axis=1)
    spread = np.max(np.linalg.norm(position[:, survivors] - centre[:, None], axis=2), axis=1)
    truth = estimates.get("load_true")
    error = np.linalg.norm(centre - np.asarray(truth, dtype=float)[:, :2], axis=1) if truth is not None else np.full(time.size, np.nan)
    before = (time >= t_sev - 5.0) & (time < t_sev)
    after = (time >= t_sev + 2.0) & (time <= t_sev + 12.0)

    def median(values, mask):
        return float(np.median(values[mask])) if mask.any() else None

    return {"severed_offset": [median(offset, before), median(offset, after)], "survivor_spread": [median(spread, before), median(spread, after)],
            "survivor_error": [median(error, before), median(error, after)]}


def analyse() -> dict:
    with CACHE_PATH.open("rb") as handle:
        cache = pickle.load(handle)
    results = cache["results"]
    rng = np.random.default_rng(20260916)
    table: dict = {}
    for item in results:
        table[(item["mode"], item["arm"], item["seed"])] = item
    indicator = {mode: {arm: np.array([table[(mode, arm, s)]["virtual_first" if mode == "recording" else "live_first"] is not None for s in SEEDS], dtype=float)
                        for arm in ARMS} for mode in MODES}
    probability = {mode: {arm: float(indicator[mode][arm].mean()) for arm in ARMS} for mode in MODES}
    tests = {}
    rec = indicator["recording"]
    diffs = {name: _paired(rec[a], rec[b], rng) for name, (a, b) in {"N-P": ("N", "P"), "L-P": ("L", "P"), "B2-P": ("B2", "P"), "P-O": ("P", "O")}.items()}
    alarm = {mode: {arm: float(np.mean([_alarm_rate(table[(mode, arm, s)]) for s in SEEDS])) for arm in ARMS} for mode in MODES}
    maxima = {arm: np.array([table[("recording", arm, s)]["max_tension"] for s in SEEDS]) for arm in ARMS}
    powered = {arm: [float(np.mean(maxima[arm] > level)) for level in POWERED_LEVELS] for arm in ARMS}
    t1 = diffs["N-P"][1] > 0.15 and (diffs["L-P"][0] >= 0.15 or alarm["recording"]["L"] >= 0.30) and diffs["B2-P"][0] >= 0.15 and diffs["P-O"][2] <= 0.05
    tests["P6-T1"] = {"statement": "paired virtual: p_N - p_P lower bound > 0.15; p_L - p_P >= 0.15 or L's alarm rate >= 30%; p_B2 - p_P >= 0.15; p_P - p_O <= 0.05 on the upper bound",
                      "probabilities": probability["recording"], "differences": diffs, "alarm_rate": alarm["recording"],
                      "powered_levels": list(POWERED_LEVELS), "probability_by_level": powered, "closures": closure_counts(table, "recording"),
                      "verdict": "PASS" if t1 else "FAIL"}
    live = indicator["live"]
    live_diffs = {name: _paired(live[a], live[b], rng) for name, (a, b) in {"N-P": ("N", "P"), "L-P": ("L", "P"), "B2-P": ("B2", "P"), "P-O": ("P", "O")}.items()}
    t2 = live_diffs["N-P"][1] > 0.15 and (live_diffs["L-P"][0] >= 0.15 or alarm["live"]["L"] >= 0.30) and live_diffs["B2-P"][0] >= 0.15 and live_diffs["P-O"][2] <= 0.05
    tests["P6-T2"] = {"statement": "the same margins on live first-severance events", "probabilities": probability["live"], "differences": live_diffs,
                      "alarm_rate": alarm["live"], "closures": closure_counts(table, "live"), "verdict": "PASS" if t2 else "FAIL"}
    agreement = []
    for arm in ARMS:
        for seed in SEEDS:
            virtual = table[("recording", arm, seed)]["virtual_first"]
            live_first = table[("live", arm, seed)]["live_first"]
            same = (virtual is None) == (live_first is None)
            timing = None if not (virtual and live_first) else abs(virtual[0] - live_first[0])
            agreement.append({"arm": arm, "seed": seed, "indicator_agree": same, "time_difference": timing})
    agree = all(a["indicator_agree"] for a in agreement)
    timing_ok = all(a["time_difference"] is None or a["time_difference"] <= 0.002 for a in agreement)
    tests["P6-T3"] = {"statement": "recording-live identity: indicator agreement 100%, |t_live - t_virt| <= 2 ms", "agreement_fraction": float(np.mean([a["indicator_agree"] for a in agreement])),
                      "max_time_difference": max((a["time_difference"] for a in agreement if a["time_difference"] is not None), default=None),
                      "disagreements": [a for a in agreement if not a["indicator_agree"] or (a["time_difference"] is not None and a["time_difference"] > 0.002)][:20],
                      "verdict": "PASS" if agree and timing_ok else "FAIL"}
    cost = {}
    for arm in ARMS:
        if arm == "N":
            continue
        mission_time = np.array([table[("recording", arm, s)]["outcome"]["mission_time"] - table[("recording", "N", s)]["outcome"]["mission_time"] for s in SEEDS])
        docking = np.array([table[("recording", arm, s)]["outcome"]["docking_error"] - table[("recording", "N", s)]["outcome"]["docking_error"] for s in SEEDS])
        base_time = np.array([table[("recording", "N", s)]["outcome"]["mission_time"] for s in SEEDS])
        from tether.evt.bootstrap import seed_bootstrap_statistic

        penalty = seed_bootstrap_statistic(list(range(len(SEEDS))), lambda idx: float(np.mean(mission_time[idx]) / np.mean(base_time[idx])), n_boot=4000, rng=rng)
        dock = seed_bootstrap_statistic(list(range(len(SEEDS))), lambda idx: float(np.mean(docking[idx])), n_boot=4000, rng=rng)
        cost[arm] = {"mission_time_penalty": penalty, "docking_error_penalty_m": dock}
    t4 = cost.get("P") and cost["P"]["mission_time_penalty"][2] < 0.05 and cost["P"]["docking_error_penalty_m"][2] < 0.1
    tests["P6-T4"] = {"statement": "paired cost: mission-time penalty < 5% and docking-error penalty < 0.1 m on the upper bounds", "cost": cost, "verdict": "PASS" if t4 else "FAIL"}
    critical = np.asarray(cache["critical"])
    horizon = float(cache.get("monitor", {}).get("horizon", HORIZON))
    counts = {"squall": [0, 0], "mission": [0, 0]}
    for seed in SEEDS:
        item = table[("recording", "P", seed)]
        marks = item["marks"]
        for window, (start, stop) in (("squall", SQUALL_WINDOW), ("mission", (0.0, np.inf))):
            episodes, false = false_alarm_episodes(item["supervisor_log"], marks, critical, horizon, start, stop)
            counts[window][0] += episodes
            counts[window][1] += false
    episodes, false = counts["squall"]
    rate = false / episodes if episodes else None
    tests["P6-T5"] = {"statement": "false alarms: supervisor action episodes (starting in the squall, 50-70 s) with no re-engagement above 0.5 v_b within H: P below 10%",
                      "episodes": episodes, "false": false, "rate": rate,
                      "mission_episodes": counts["mission"][0], "mission_false": counts["mission"][1],
                      "mission_rate": counts["mission"][1] / counts["mission"][0] if counts["mission"][0] else None,
                      "verdict": "UNDER-POWERED" if not episodes else ("PASS" if rate < 0.10 else "FAIL")}
    extrapolation = {}
    for arm in ARMS:
        items = [table[("recording", arm, s)] for s in SEEDS]
        exposure = float(sum(item["sim_seconds"] for item in items))
        extrapolation[arm] = {"exposure": exposure,
                              "marks": tail_extrapolation([m[4] for item in items for m in item["marks"]], exposure),
                              "q_peaks": tail_extrapolation(np.concatenate([np.asarray(item["taut_peaks"], dtype=float) for item in items]), exposure)}
    tests["P6-T6"] = {"statement": "operational extrapolation to 25 kN (reported, never gated): per-arm mark and q-peak tail fits with intervals, withheld unless inside the Poisson intervals at every powered point",
                      "max_tension_by_arm": {arm: float(maxima[arm].max()) for arm in ARMS}, "extrapolation": extrapolation, "verdict": "REPORTED"}
    attribution = {arm: mechanism_attribution([table[("recording", arm, s)] for s in SEEDS]) for arm in ("N", "P")}
    if attribution["N"] and attribution["P"]:
        attribution["P_over_N"] = {"a_bar": attribution["P"]["median_a_bar"] / attribution["N"]["median_a_bar"],
                                   "depth": attribution["P"]["median_depth"] / attribution["N"]["median_depth"]}
    tests["P6-T7"] = {"statement": "mechanism attribution: the supervised runs' excursions show reduced a_bar on the return leg and reduced depth, against the commanded thrust reduction (reported)",
                      "attribution": attribution, "verdict": "REPORTED"}
    after = {}
    for arm in ("L", "B2", "P"):
        rows = [row for row in (post_severance(table[("live", arm, s)]) for s in SEEDS) if row]
        after[arm] = {"runs": len(rows)}
        for key in ("severed_offset", "survivor_spread", "survivor_error"):
            for position, window in enumerate(("before", "after")):
                values = [row[key][position] for row in rows if row[key][position] is not None]
                after[arm][f"{key}_{window}"] = float(np.median(values)) if values else None
    tests["P6-T8"] = {"statement": "estimation after severance: fleet disagreement and the severed vessel's component against the reduced graph (EMP)", "verdict": "REPORTED",
                      "live_severances": {arm: int(live[arm].sum()) for arm in ARMS}, "post_severance": after}
    hero = hero_candidates(table)
    tests["P6-T9"] = {"statement": "hero replay: seed-matched N-against-P timeline at T_b^s (reports/figures/phase6_F1_hero_replay.png)", "candidates": hero[:5],
                      "verdict": "AVAILABLE" if hero else "NO CANDIDATE"}
    passed = all(tests[k]["verdict"] == "PASS" for k in ("P6-T1", "P6-T2", "P6-T3", "P6-T4")) and tests["P6-T5"]["verdict"] == "PASS"
    gate = {"decision": "PASS" if passed else "FAIL", "rule": "PASS if T1, T2, T3, T4 pass and T5 holds"}
    output = {"schema_version": 1, "source": source_state(), "declarations": DECLARATIONS, "breaking_strength": cache["breaking"], "h_crit": cache["h_crit"],
              "critical_speeds": cache["critical"], "probabilities": probability, "tests": tests, "gate": gate,
              "wall_seconds": float(sum(r["wall_seconds"] for r in results))}
    write_bytes(RESULTS_PATH, json_bytes(output))
    return output


def hero_candidates(table: dict) -> list[int]:
    """Seeds where N severs virtually at T_b^s and P completes the mission without a virtual
    severance (a P run ended early by formation closure does not count), in seed order."""
    return [s for s in SEEDS if table[("recording", "N", s)]["virtual_first"] is not None and table[("recording", "P", s)]["virtual_first"] is None
            and table[("recording", "P", s)]["outcome"]["closure"] is None]


def closure_counts(table: dict, mode: str) -> dict:
    """Per arm: formation closures, closures before any severance, and missions ending in
    severance or closure (a closure ends the run, so it censors the severance count)."""
    key = "virtual_first" if mode == "recording" else "live_first"
    out = {}
    for arm in ARMS:
        items = [table[(mode, arm, s)] for s in SEEDS]
        closure = [item["outcome"]["closure"] for item in items]
        first = [item[key] for item in items]
        out[arm] = {"closures": sum(c is not None for c in closure),
                    "closure_before_severance": sum(1 for c, f in zip(closure, first) if c is not None and (f is None or c[1] < f[0])),
                    "severance_or_closure": float(np.mean([c is not None or f is not None for c, f in zip(closure, first)]))}
    return out


def hero_job(payload) -> dict:
    """Full-resolution record of one recording-mode mission for the P6-T9 timeline."""
    from tether.campaign.fleet_run import run_to_end
    from tether.campaign.mission import MissionSpec, build_mission
    from tether.physics import constants

    arm, seed, breaking, h_crit, critical, monitor_config = payload
    spec = MissionSpec(cable_mode="recording", break_threshold=None)
    hook = None if arm == "N" else supervisor_hook(arm, seed, h_crit, critical, monitor_config)
    run = build_mission(spec, seed, hook, supervised=arm != "N", exemptions=frozenset({"oracle_adapter"}) if arm == "O" else frozenset())
    run_to_end(run)
    log = run.fleet.cables.log
    e = log.elongation[: log.count]
    edot = log.rate[: log.count]
    q = np.where(e > 0.0, constants.CABLE_STIFFNESS * e + constants.CABLE_DAMPING * edot, 0.0)
    step = 10
    record = {"time": log.event_time[: log.count : step], "q": q[::step], "e": e[::step], "edot": edot[::step],
              "marks": np.array([(m.cable, m.t_up, m.depth, m.v_return, m.T_peak) for m in run.fleet.cables.reengagements]).reshape(-1, 5)}
    over = np.argwhere(q > breaking)
    record["virtual_first"] = np.array([log.event_time[over[0, 0]], over[0, 1]]) if over.size else np.array([np.nan, np.nan])
    supervisor = run.fleet.extras.get("supervisor")
    if supervisor is not None:
        record["supervisor_time"] = np.array([t for t, _, _ in supervisor.log])
        record["hazard"] = np.array([h for _, h, _ in supervisor.log])
        record["scale"] = np.array([sc for _, _, sc in supervisor.log])
    return {"arm": arm, "record": record}


def hero(workers: int) -> None:
    """Re-run the first hero candidate (else the seed with the largest N - P peak gap) for N and P."""
    from tether.campaign.common import npz_bytes

    with CACHE_PATH.open("rb") as handle:
        cache = pickle.load(handle)
    table = {(item["mode"], item["arm"], item["seed"]): item for item in cache["results"]}
    candidates = hero_candidates(table)
    if candidates:
        seed, rule = candidates[0], "first seed on which N severs virtually and P does not"
    else:
        gap = {s: table[("recording", "N", s)]["max_tension"] - table[("recording", "P", s)]["max_tension"] for s in SEEDS
               if table[("recording", "P", s)]["outcome"]["closure"] is None and table[("recording", "N", s)]["outcome"]["closure"] is None}
        seed, rule = max(gap, key=gap.get), ("no seed on which N severs and P completes without severing; the seed with the largest N - P peak-tension gap "
                                             "among seeds where both missions complete")
    monitor = cache.get("monitor", {"model": "constant_acceleration", "horizon": HORIZON, "l_growth": 0.0})
    payloads = [(arm, seed, cache["breaking"], cache["h_crit"].get(arm, 1.0), np.asarray(cache["critical"]), monitor) for arm in ("N", "P")]
    runs = {item["arm"]: item["record"] for item in run_pool(hero_job, payloads, workers)}
    arrays = {f"{arm}_{key}": value for arm, record in runs.items() for key, value in record.items()}
    arrays.update(seed=np.array(seed), breaking=np.array(cache["breaking"]), critical=np.asarray(cache["critical"]), h_crit=np.array(cache["h_crit"].get("P", 1.0)))
    write_bytes(HERO_PATH, npz_bytes(arrays))
    write_bytes(HERO_PATH.with_suffix(".json"), json_bytes({"seed": int(seed), "rule": rule, "candidates": candidates[:10], "source": source_state()}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("compute", "analyse", "hero"))
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    if arguments.command == "compute":
        compute(arguments.workers)
    elif arguments.command == "hero":
        hero(arguments.workers)
    else:
        results = analyse()
        for test_id, test in results["tests"].items():
            print(test_id, test["verdict"])
        print("GATE", results["gate"])


if __name__ == "__main__":
    main()
