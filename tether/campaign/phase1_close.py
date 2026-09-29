"""Phase 1 closure on the production plant.

Executes the Phase 1 items that were never run - the stochastic excursion cells (d),
P1-T5 on stochastic excursions, P1-T8 and P1-T10 - and the two production-plant
measurements every later phase needs: the tabulated impact curve (the pre-declared
"v_b = Z^-1(T_b)" pivot of the Phase 1 outcome matrix) and a regression of the P1-T6
acceleration threshold.  The gate decision combines these with the committed Phase 1R,
P1-T9R, P1-T5R and P1-T4R records.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass

import numpy as np

from tether.campaign.common import (
    FIGURES,
    RECORDS,
    REPORTS,
    config_sha256,
    json_bytes,
    npz_bytes,
    relative,
    run_pool,
    sha256_file,
    source_state,
    write_bytes,
)
from tether.campaign.fleet_run import FleetRunSpec, build_run, run_to_end, set_state
from tether.campaign.summaries import summarize_run
from tether.physics import constants
from tether.physics.fleet import (
    PAIR_REDUCED_MASS,
    equilibrium_state,
    fleet_effective_mass,
    formation_geometry,
    operating_point,
    two_body_effective_mass,
)

RECORD_DIR = RECORDS / "phase1"
RESULTS_PATH = RECORD_DIR / "phase1_close_results.json"
RECORD_PATH = RECORD_DIR / "phase1_close_records.npz"
MANIFEST_PATH = RECORD_DIR / "phase1_close_manifest.json"
IMPACT_TABLE_PATH = RECORD_DIR / "impact_table.json"
REPORT_PATH = REPORTS / "phase1_report.md"

PRETENSIONS = (600.0, 1000.0, 1400.0, 1800.0)
INTENSITIES = (0.5, 1.0)
SEEDS = tuple(range(1001, 1021))
DURATION = 300.0
WARMUP = 20.0
TAU_W = constants.WEATHER_TIME_CONSTANT
IMPACT_SPEEDS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
IMPACT_CABLES = (0, 1, 2)
IMPACT_DURATION = 0.4
T6_RATIOS = (0.9, 0.95, 1.0, 1.05, 1.1)
T6_WINDOW = 0.005
IMPACT_FACTOR = 0.880


@dataclass(frozen=True)
class StochasticJob:
    cell: str
    pretension: float
    intensity: float
    seed: int
    cost: float = 1.0

    def spec(self) -> FleetRunSpec:
        return FleetRunSpec(pretension=self.pretension, weather_scale=self.intensity, duration=DURATION, warmup=WARMUP)


def stochastic_jobs() -> list[StochasticJob]:
    return [
        StochasticJob(f"T{int(p)}_I{i}", p, i, seed)
        for p in PRETENSIONS
        for i in INTENSITIES
        for seed in SEEDS
    ]


def run_stochastic_job(job: StochasticJob) -> dict:
    spec = job.spec()
    run = run_to_end(build_run(spec, job.seed))
    summary = summarize_run(run, spec.warmup, spec.pretension)
    summary["job"] = asdict(job)
    return summary


@dataclass(frozen=True)
class ImpactJob:
    pretension: float
    cable: int
    speed: float
    cost: float = 0.2
    formation: str = "parallel"
    arc_half_angle: float | None = None


def run_impact_job(job: ImpactJob) -> dict:
    """Target cable at zero elongation closing at ``speed``; the others at pretension."""
    geometry = formation_geometry(job.formation, arc_half_angle=job.arc_half_angle)
    point = operating_point(geometry, job.pretension)
    spec = FleetRunSpec(formation=job.formation, arc_half_angle=job.arc_half_angle, pretension=job.pretension, weather_scale=0.0,
                        duration=IMPACT_DURATION, warmup=0.0, closed_loop=False, log_state=False)
    run = build_run(spec, 1)
    state = equilibrium_state(geometry, point)
    direction = np.array([math.cos(geometry.cable_angles[job.cable]), math.sin(geometry.cable_angles[job.cable])])
    body = job.cable + 1
    state[3 * body : 3 * body + 2] -= direction * job.pretension / constants.CABLE_STIFFNESS
    total = constants.LOAD_MASS + geometry.vessel_count * constants.VESSEL_MASS
    base = 3 * (geometry.vessel_count + 1)
    state[base::3] += -constants.VESSEL_MASS / total * job.speed * direction[0]
    state[base + 1 :: 3] += -constants.VESSEL_MASS / total * job.speed * direction[1]
    state[base + 3 * body : base + 3 * body + 2] += direction * job.speed
    set_state(run.fleet, run.simulator.get_mutable_context(), state)
    run_to_end(run)
    log = run.fleet.cables.log
    e = log.elongation[: log.count, job.cable]
    rate = log.rate[: log.count, job.cable]
    q = np.where(e > 0.0, constants.CABLE_STIFFNESS * e + constants.CABLE_DAMPING * rate, 0.0)
    peak = float("nan")
    for index in range(1, q.size):
        if q[index] < q[index - 1] and q[index - 1] > 0.0:
            peak = float(q[index - 1])
            break
    return {"pretension": job.pretension, "cable": job.cable, "speed": job.speed, "peak": peak}


def run_t6_probe(ratio: float) -> dict:
    """Corrected P1-T6 probe: centre cable slack at -1 mm, zero relative speed, gust ratio."""
    geometry = formation_geometry("parallel")
    pretension = 990.0
    point = operating_point(geometry, pretension)
    target = 2
    gust = ratio * pretension
    forces = np.zeros((geometry.vessel_count + 1, 2))
    forces[0, 0] = gust
    forces[target + 1, 0] = -gust
    spec = FleetRunSpec(pretension=pretension, weather_scale=0.0, duration=0.02, warmup=0.0, closed_loop=False, log_state=False)
    run = build_run(spec, 1, scripted_force=lambda t: forces)
    state = equilibrium_state(geometry, point)
    state[3 * (target + 1)] -= pretension / constants.CABLE_STIFFNESS + 1.0e-3
    set_state(run.fleet, run.simulator.get_mutable_context(), state)
    run_to_end(run)
    log = run.fleet.cables.log
    time = log.event_time[: log.count]
    rate = log.rate[: log.count, target]
    window = time <= T6_WINDOW + 1.0e-12
    slope = float(np.polyfit(time[window], rate[window], 1)[0])
    return {"ratio": ratio, "acceleration": slope, "predicted_two_body": (1.0 - ratio) * pretension / PAIR_REDUCED_MASS}


def _through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope = float(np.dot(x, y) / np.dot(x, x))
    residual = y - slope * x
    return slope, float(1.0 - np.dot(residual, residual) / np.dot(y, y))


def build_impact_table(points: list[dict]) -> dict:
    table = {}
    for pretension in PRETENSIONS:
        for cable in IMPACT_CABLES:
            rows = sorted((p["speed"], p["peak"]) for p in points if p["pretension"] == pretension and p["cable"] == cable)
            speeds = np.array([row[0] for row in rows])
            peaks = np.array([row[1] for row in rows])
            high = speeds >= 1.0
            slope, r2 = _through_origin(speeds[high], peaks[high])
            table[f"T{int(pretension)}_c{cable}"] = {
                "pretension": pretension,
                "cable": cable,
                "speeds": speeds.tolist(),
                "peaks": peaks.tolist(),
                "asymptotic_impedance": slope,
                "asymptotic_r2": r2,
                "zero_speed_peak": float(peaks[speeds == 0.0][0]),
                "monotone": bool(np.all(np.diff(peaks) > 0.0)),
            }
    return table


def tabulated_peak(table: dict, pretension: float, cable: int, speed: float) -> float:
    """Measured impact peak at ``speed`` for one cable position (mirror-symmetric).

    Monotone PCHIP through the measured points; beyond the fastest point, linear with
    the asymptotic (through-origin, >= 1 m/s) impedance.
    """
    from scipy.interpolate import PchipInterpolator

    entry = table[f"T{int(pretension)}_c{min(cable, 4 - cable)}"]
    speeds = np.asarray(entry["speeds"])
    peaks = np.asarray(entry["peaks"])
    if speed <= speeds[-1]:
        return float(PchipInterpolator(speeds, peaks)(speed))
    return float(peaks[-1] + entry["asymptotic_impedance"] * (speed - speeds[-1]))


def pool_cell(summaries: list[dict]) -> dict:
    """Concatenate per-seed marks and sum per-seed counts within one cell."""
    marks = {}
    for key in summaries[0]["marks"]:
        marks[key] = np.concatenate([s["marks"][key] for s in summaries])
    marks["seed"] = np.concatenate([np.full(s["marks"]["t_up"].size, s["meta"]["seed"]) for s in summaries])
    return {
        "marks": marks,
        "exposure": float(sum(s["meta"]["exposure"] for s in summaries)),
        "exposures": np.array([s["meta"]["exposure"] for s in summaries]),
        "closures": [s["meta"]["closure"] for s in summaries],
        "onsets": np.sum([s["onsets"] for s in summaries], axis=0),
        "level_counts": np.sum([s["level_counts"] for s in summaries], axis=0),
        "slack_samples": np.sum([s["slack_samples"] for s in summaries], axis=0),
        "window_samples": int(sum(s["meta"]["window_samples"] for s in summaries)),
        "wall_seconds": float(sum(s["meta"]["wall_seconds"] for s in summaries)),
        "sim_seconds": float(sum(s["meta"]["sim_seconds"] for s in summaries)),
        "tracker_dropped": int(sum(s["meta"]["tracker_dropped"] for s in summaries)),
        "cable_angle_std": np.mean([s["geometry_stats"]["cable_angle_std"] for s in summaries], axis=0),
        "load_yaw_std": float(np.mean([s["geometry_stats"]["load_yaw_std"] for s in summaries])),
    }


def analyse(cells: dict[str, dict], impact_table: dict, t6: list[dict]) -> dict:
    geometry = formation_geometry("parallel")
    tests: dict[str, dict] = {}

    ratios = np.array([p["ratio"] for p in t6])
    accelerations = np.array([p["acceleration"] for p in t6])
    fit = np.polyfit(ratios, accelerations, 1)
    lambda_c = float(-fit[1] / fit[0])
    tests["P1-T6-production"] = {
        "statement": "acceleration-sign threshold of the corrected P1-T6 probe, production plant",
        "lambda_c": lambda_c,
        "probes": t6,
        "criterion": "|lambda_c - 1| <= 0.05",
        "verdict": "PASS" if abs(lambda_c - 1.0) <= 0.05 else "FAIL",
    }

    centre = impact_table["T1000_c2"]
    tests["P1-T2b-production"] = {
        "statement": "fleet impedance of the centre cable on the production plant, through-origin fit over speeds >= 1 m/s",
        "impedance": centre["asymptotic_impedance"],
        "r2": centre["asymptotic_r2"],
        "phase1_pinned": 8304.485029550166,
        "relative_difference": centre["asymptotic_impedance"] / 8304.485029550166 - 1.0,
        "criterion": "R2 > 0.99 (linearity above 1 m/s); value recorded",
        "verdict": "PASS" if centre["asymptotic_r2"] > 0.99 else "FAIL",
    }

    energy_rows = []
    tpeak_rows = []
    duration_rows = []
    all_x_two, all_x_fleet, all_y, all_h4 = [], [], [], []
    max_depth = {"depth": 0.0}
    for name, cell in cells.items():
        marks = cell["marks"]
        pretension = float(name.split("_")[0][1:])
        complete = marks["depth"] > 0.0
        depth = marks["depth"][complete]
        u = marks["u_entry"][complete]
        v = marks["v_return"][complete]
        cable = marks["cable"][complete]
        m_two = np.array([two_body_effective_mass(geometry, int(c)) for c in cable])
        m_fleet = np.array([fleet_effective_mass(geometry, int(c)) for c in cable])
        x_two = np.sqrt(u**2 + 2.0 * pretension / m_two * depth)
        x_fleet = np.sqrt(u**2 + 2.0 * pretension / m_fleet * depth)
        h4 = (marks["dwell"][complete] < TAU_W / 4.0) & (marks["n_maxima"][complete] <= 1)
        all_x_two.append(x_two)
        all_x_fleet.append(x_fleet)
        all_y.append(v)
        all_h4.append(h4)
        dwell = marks["dwell"]
        p95 = float(np.percentile(dwell, 95)) if dwell.size else float("nan")
        duration_rows.append({
            "cell": name,
            "excursions": int(dwell.size),
            "dwell_p95": p95,
            "dwell_p50": float(np.percentile(dwell, 50)) if dwell.size else float("nan"),
            "below_tau_w_over_4": bool(p95 < TAU_W / 4.0) if dwell.size else None,
            "multi_max_fraction": float(np.mean(marks["n_maxima"] > 1)) if dwell.size else float("nan"),
            "slack_duty": float(np.sum(cell["slack_samples"]) / (cell["window_samples"] * geometry.vessel_count)),
            "closures": int(sum(c is not None for c in cell["closures"])),
            "snap_max": float(np.max(marks["T_peak"])) if dwell.size else 0.0,
            "cable_angle_std_deg": np.degrees(cell["cable_angle_std"]).tolist(),
            "load_yaw_std_deg": math.degrees(cell["load_yaw_std"]),
        })
        if depth.size and depth.max() > max_depth["depth"]:
            index = int(np.argmax(depth))
            max_depth = {"depth": float(depth[index]), "cell": name, "pretension": pretension, "T_peak": float(marks["T_peak"][complete][index])}
        for label, x, mask in (("two_body_a0", x_two, np.ones_like(h4)), ("two_body_a0_H4H5", x_two, h4), ("fleet_a0_H4H5", x_fleet, h4)):
            if mask.sum() >= 3:
                slope, r2 = _through_origin(x[mask], v[mask])
                energy_rows.append({"cell": name, "selection": label, "n": int(mask.sum()), "slope": slope, "r2": r2})
        key = f"T{int(pretension)}_c"
        big = v >= 1.0
        if big.sum():
            predicted = np.array([tabulated_peak(impact_table, pretension, int(c), float(s)) for c, s in zip(cable[big], v[big])])
            linear = np.array([impact_table[f"{key}{min(int(c), 4 - int(c))}"]["asymptotic_impedance"] * float(s) for c, s in zip(cable[big], v[big])])
            measured = marks["T_peak"][complete][big]
            tpeak_rows.append({
                "cell": name,
                "n": int(big.sum()),
                "tabulated_max_rel_error": float(np.max(np.abs(measured / predicted - 1.0))),
                "tabulated_median_rel_error": float(np.median(np.abs(measured / predicted - 1.0))),
                "linear_max_rel_error": float(np.max(np.abs(measured / linear - 1.0))),
                "linear_median_rel_error": float(np.median(np.abs(measured / linear - 1.0))),
            })

    x_two = np.concatenate(all_x_two)
    x_fleet = np.concatenate(all_x_fleet)
    y = np.concatenate(all_y)
    h4 = np.concatenate(all_h4)
    pooled = {}
    for label, x, mask in (("all_two_body_a0", x_two, np.ones_like(h4)), ("H4H5_two_body_a0", x_two, h4), ("H4H5_fleet_a0", x_fleet, h4), ("not_H4_two_body_a0", x_two, ~h4)):
        slope, r2 = _through_origin(x[mask], y[mask]) if mask.sum() >= 3 else (float("nan"), float("nan"))
        pooled[label] = {"n": int(mask.sum()), "slope": slope, "r2": r2}
    primary = pooled["H4H5_two_body_a0"]
    t5_pass = abs(primary["slope"] - 1.0) <= 0.05 and primary["r2"] > 0.98
    tests["P1-T5-stochastic"] = {
        "statement": "V_up against sqrt(u^2 + 2 a0 Delta) on stochastic excursions; F1 restricts the identity to excursions satisfying (H4),(H5)",
        "pooled": pooled,
        "per_cell": energy_rows,
        "criterion": "slope within 5% of one and uncentred R2 > 0.98 on (H4),(H5) excursions (a0 = T0/m_eff two-body, the plan's definition)",
        "verdict": "PASS" if t5_pass else "FAIL",
        "tpeak_vs_impact_model": tpeak_rows,
    }
    tests["P1-T8"] = {
        "statement": "largest depth attainable before the formation closes (stochastic cells)",
        "observed_max_complete_depth": max_depth,
        "implied_max_snap_f_sqrt_2kT0Delta": IMPACT_FACTOR * math.sqrt(2.0 * constants.CABLE_STIFFNESS * max_depth.get("pretension", 990.0) * max_depth["depth"]),
        "closure_events": {row["cell"]: row["closures"] for row in duration_rows},
        "closure_chord_length_m": 1.0,
        "verdict": "MEASURED",
    }
    tests["P1-T10"] = {
        "statement": "95th-percentile excursion duration against tau_w/4 = 2 s in the stochastic cells",
        "per_cell": duration_rows,
        "criterion": "p95 dwell < tau_w/4 (reported otherwise; non-blocking)",
        "verdict": "PASS" if all(row["below_tau_w_over_4"] in (True, None) for row in duration_rows) else "FAIL (reported)",
    }
    return tests


def phase1_record_arrays(cells: dict[str, dict], impact_points: list[dict]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    names = sorted(cells)
    marks_keys = list(cells[names[0]]["marks"].keys())
    cell_index = np.concatenate([np.full(cells[n]["marks"]["t_up"].size, i) for i, n in enumerate(names)])
    arrays["marks_cell_index"] = cell_index
    for key in marks_keys:
        arrays[f"marks_{key}"] = np.concatenate([cells[n]["marks"][key] for n in names])
    arrays["cell_names"] = np.array(names)
    arrays["cell_exposure"] = np.array([cells[n]["exposure"] for n in names])
    arrays["cell_level_counts"] = np.stack([cells[n]["level_counts"] for n in names])
    from tether.campaign.summaries import LEVEL_GRID

    arrays["level_grid"] = LEVEL_GRID
    arrays["impact_pretension"] = np.array([p["pretension"] for p in impact_points])
    arrays["impact_cable"] = np.array([p["cable"] for p in impact_points])
    arrays["impact_speed"] = np.array([p["speed"] for p in impact_points])
    arrays["impact_peak"] = np.array([p["peak"] for p in impact_points])
    return arrays


def configuration() -> dict:
    return {
        "plant": "production fleet plant (tether.physics.fleet), parallel formation, SAP 0.25 ms physics, 1 ms events",
        "pretensions_n": PRETENSIONS,
        "weather_intensities": INTENSITIES,
        "weather": "gaussian local AR(1), tau_w 8 s, 40 s pre-roll, intensity x pinned 3.5/0.7 kN",
        "heading_gain": 500.0,
        "seeds": SEEDS,
        "duration_s": DURATION,
        "warmup_s": WARMUP,
        "impact_speeds": IMPACT_SPEEDS,
        "impact_cables": IMPACT_CABLES,
        "t6_ratios": T6_RATIOS,
    }


CACHE_PATH = RECORD_DIR / "cache" / "phase1_close_compute.pkl"


def compute(workers: int) -> None:
    """Run every simulation and cache the per-job summaries (the expensive half)."""
    import pickle

    impact_jobs = [ImpactJob(p, c, s) for p in PRETENSIONS for c in IMPACT_CABLES for s in IMPACT_SPEEDS]
    impact_points = run_pool(run_impact_job, impact_jobs, workers)
    t6 = [run_t6_probe(r) for r in T6_RATIOS]
    jobs = stochastic_jobs()

    def progress(count, total):
        if count % 16 == 0 or count == total:
            print(f"stochastic jobs {count}/{total}", flush=True)

    summaries = run_pool(run_stochastic_job, jobs, workers, progress)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("wb") as handle:
        pickle.dump({"impact_points": impact_points, "t6": t6, "jobs": jobs, "summaries": summaries}, handle)


def run_campaign(workers: int | None = None) -> dict:
    """Analyse the cached computation into records, results, and the manifest."""
    import pickle

    with CACHE_PATH.open("rb") as handle:
        cache = pickle.load(handle)
    impact_points, t6, jobs, summaries = cache["impact_points"], cache["t6"], cache["jobs"], cache["summaries"]
    impact_table = build_impact_table(impact_points)
    write_bytes(IMPACT_TABLE_PATH, json_bytes(impact_table))
    by_cell: dict[str, list] = {}
    for job, summary in zip(jobs, summaries):
        by_cell.setdefault(job.cell, []).append(summary)
    cells = {name: pool_cell(items) for name, items in by_cell.items()}
    tests = analyse(cells, impact_table, t6)
    arrays = phase1_record_arrays(cells, impact_points)
    record_sha = write_bytes(RECORD_PATH, npz_bytes(arrays))
    probe_summary = summaries[0]
    probe = {
        "job": probe_summary["job"],
        "marks_t_up": probe_summary["marks"]["t_up"].tolist(),
        "marks_T_peak": probe_summary["marks"]["T_peak"].tolist(),
        "moments_q": probe_summary["moments"]["q"].tolist(),
    }
    execution = {
        "cells": {name: {"seeds": len(by_cell[name]), "wall_seconds": cells[name]["wall_seconds"], "sim_seconds": cells[name]["sim_seconds"], "tracker_dropped": cells[name]["tracker_dropped"]} for name in sorted(cells)},
        "total_wall_core_seconds": float(sum(c["wall_seconds"] for c in cells.values())),
        "total_sim_seconds": float(sum(c["sim_seconds"] for c in cells.values())),
    }
    results = {
        "schema_version": 1,
        "configuration": configuration(),
        "configuration_sha256": config_sha256(configuration()),
        "source": source_state(),
        "execution": execution,
        "tests": tests,
        "impact_table_path": relative(IMPACT_TABLE_PATH),
        "record_sha256": record_sha,
        "probe": probe,
    }
    write_bytes(RESULTS_PATH, json_bytes(results))
    manifest = {
        "schema_version": 1,
        "driver": "python -m tether.campaign.phase1_close run",
        "configuration_sha256": results["configuration_sha256"],
        "seeds": list(SEEDS),
        "committed_record_path": relative(RECORD_PATH),
        "record_sha256": record_sha,
        "impact_table_sha256": sha256_file(IMPACT_TABLE_PATH),
        "probe": {"command": "python -m tether.campaign.phase1_close replay", "job": probe_summary["job"], "marks": "exact", "moments_absolute_tolerance": 1.0e-6},
    }
    write_bytes(MANIFEST_PATH, json_bytes(manifest))
    return results


def replay() -> bool:
    results = json.loads(RESULTS_PATH.read_text())
    probe = results["probe"]
    job = StochasticJob(**probe["job"])
    summary = run_stochastic_job(job)
    marks_equal = summary["marks"]["t_up"].tolist() == probe["marks_t_up"] and summary["marks"]["T_peak"].tolist() == probe["marks_T_peak"]
    moments_close = bool(np.allclose(summary["moments"]["q"], np.array(probe["moments_q"]), rtol=0.0, atol=1.0e-6 * max(1.0, float(np.max(np.abs(probe["moments_q"]))))))
    impact_ok = sha256_file(IMPACT_TABLE_PATH) == json.loads(MANIFEST_PATH.read_text())["impact_table_sha256"]
    passed = marks_equal and moments_close and impact_ok
    print(json.dumps({"marks_exact": marks_equal, "moments_within_tolerance": moments_close, "impact_table_digest": impact_ok, "passed": passed}, indent=2))
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("compute", "analyse", "run", "replay"))
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    if arguments.command in ("compute", "run"):
        compute(arguments.workers)
    if arguments.command in ("analyse", "run"):
        results = run_campaign()
        for test_id, test in results["tests"].items():
            print(test_id, test["verdict"])
    if arguments.command == "replay":
        sys.exit(0 if replay() else 1)


if __name__ == "__main__":
    main()
