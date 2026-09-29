"""Phase 2: the Gaussian threshold laws (plan Part V, Phase 2).

Commands: ``predict`` commits every theory prediction before any Phase 2 plant run;
``compute`` runs the cells; ``analyse`` evaluates P2-T0..T10 and writes the records;
``replay`` re-executes the probe.  Operational choices the plan leaves open are declared
in ``DECLARATIONS`` and copied verbatim into the predictions file and the report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict

import numpy as np

from tether.campaign.common import (
    RECORDS,
    REPORTS,
    config_sha256,
    json_bytes,
    npz_bytes,
    relative,
    sha256_file,
    source_state,
    write_bytes,
)
from tether.campaign.fleet_run import FleetRunSpec
from tether.campaign.stationary import StationaryJob, compute_jobs, load_cache, run_stationary_job
from tether.campaign.summaries import LEVEL_GRID
from tether.physics import constants
from tether.physics.fleet import formation_geometry
from tether.physics.weather import stationary_weather_forces
from tether.theory import gust as gusttheory

RECORD_DIR = RECORDS / "phase2"
PREDICTIONS_PATH = RECORD_DIR / "phase2_predictions.json"
CACHE_PATH = RECORD_DIR / "cache" / "phase2_compute.pkl"
RESULTS_PATH = RECORD_DIR / "phase2_results.json"
RECORD_PATH = RECORD_DIR / "phase2_records.npz"
MANIFEST_PATH = RECORD_DIR / "phase2_manifest.json"
REPORT_PATH = REPORTS / "phase2_report.md"

PRETENSIONS = (600.0, 1000.0, 1400.0, 1800.0)
HEADING_GAINS = (250.0, 500.0, 1000.0)
INTENSITIES = (0.5, 1.0)
NULL_PRETENSION = 1000.0
NULL_HEADING_GAIN = 500.0
NULL_INTENSITY = 0.2
PILOT_SEEDS = (2001, 2002)
STAT_SEEDS = tuple(range(2003, 2023))
WEATHER_SEEDS = tuple(range(9001, 9041))
DURATION = 600.0
WARMUP = 20.0
IMPACT_FACTOR = 0.880
QUANTILES = (50, 80, 95, 99)
POWER_MIN = 20
TAU_W = constants.WEATHER_TIME_CONSTANT
COMMON_THRESHOLD = 8000.0
OPTIMUM_GRID = tuple(float(x) for x in np.arange(400.0, 2201.0, 50.0))
REFERENCE_CELL = "T1000_k500_I1.0"
DELTA_MAX_P1 = None  # filled from the Phase 1 closure record at predict time

DECLARATIONS = {
    "pretension_setting": "T0 set by thrust alone in the parallel formation; with linear drag the tow speed does not enter the perturbation dynamics, so speed is dynamically inert and 'separated' from pretension without a fan (the fan angle is reserved for Phase 4)",
    "weather": "gaussian local AR(1), tau_w = 8 s, intensities multiply the pinned per-component stds 3.5 kN (load) and 0.7 kN (vessels)",
    "seeds": "pilot seeds 2001-2002 set each cell's threshold grid and are excluded from every statistic; statistics use seeds 2003-2022; all cells share the seed list (common random numbers)",
    "grid": "T_b(q) = f sqrt(2 k T0 Delta_q), q in {50,80,95,99}, Delta_q the pilot depth quantiles pooled over cables; a threshold is powered when the statistics seeds give >= 20 exceedances",
    "snap_rate": "Lambda_snap(T_b) = #(re-engagement marks with T_peak > T_b) / recorded time, pooled over cables and seeds; seed-bootstrap interval",
    "taut_rate": "Lambda_taut(T_b) = declustered upcrossings of q on taut samples outside the 1 s snap windows / recorded time; taut grid = pilot taut cluster-maximum quantiles {50,80,95,99} snapped to the level grid",
    "theorem6": "per cable: nu_ep = rate of W_rel > T0 episodes (runs merged across gaps < 1 s) and t_g = their mean duration, from 40 weather-only seeds x 600 s; mu_W, sigma_W the stationary moments of W_rel; Lambda_i(T_b) = nu_ep S((T0 + kappa T_b^2 - mu_W)/sigma_W)/S((T0 - mu_W)/sigma_W), kappa = m_eff/(f^2 k T0 t_g^2), m_eff two-body 483.9 kg, f = 0.880; the cell prediction sums the five cables",
    "theorem6_slope": "primary: least-squares slope of log(measured rate) against T_b^4 over the powered thresholds, compared with the literal -kappa^2/(2 sigma_W^2) at the cable-mean kappa and sigma_W; secondary: compared with the secant slope of the predicted curve over the same thresholds",
    "theorem6_secondary": "secondary predictions committed before analysis after an independent derivation found the plan's Corollary 5 drops the post-gust closing: (i) Theorem 6 with the exact square-gust severance level W* = (T0 + sqrt(T0^2 + 4 kappa T0 T_b^2))/2 in place of T0 + kappa T_b^2; (ii) the exact per-episode snap f t sqrt(k W (W - T0)/m_eff), W = T0 + delta, over the weather episodes. The plan's literal Theorem 6 remains the primary test",
    "taut_prediction": "Rice upcrossing rate with (mu_q = T0, sigma_q, sigma_qdot) from the reduced LTI model (primary) and from measured taut moments (secondary)",
    "domain": "in domain when p95 dwell < tau_w/4, multi-maximum fraction < 0.10 and slack duty < 0.10",
    "null": "the null cell runs at the largest intensity in {0.25, 0.2} whose weather-only W_rel/T0 maximum over all its 22 seeds and five cables is < 1 (0.25 gives 1.018, 0.2 gives 0.814), T0 = 1 kN; its thresholds are the reference cell's powered grid",
    "optimum": "total rate Lambda_snap + Lambda_taut at the common threshold against T0, per (intensity, heading gain); the common threshold is the largest whole kN at which every Phase 1(d) cell at intensity 1.0 has >= 20 snap + taut exceedances (fixed from Phase 1 data); predicted optimum from the Theorem 6 snap branch plus the LTI taut branch on a 50 N grid",
    "t7": "per snap above the cell's q50 threshold with a gust exceedance during slack: T_peak / (f t sqrt(k T0 delta/m_eff)), t = time with W_rel > T0 during the slack interval, delta = mean excess; pass when the median |ratio - 1| <= 0.25",
    "t9": "largest observed snap against f sqrt(2 k T0 Delta_max), Delta_max = the Phase 1 closure's largest observed complete depth",
}


def cell_name(pretension: float, heading_gain: float, intensity: float) -> str:
    return f"T{int(pretension)}_k{int(heading_gain)}_I{intensity}"


def null_cell_name() -> str:
    return "NULL_" + cell_name(NULL_PRETENSION, NULL_HEADING_GAIN, NULL_INTENSITY)


def cell_specs() -> dict[str, FleetRunSpec]:
    specs = {}
    for intensity in INTENSITIES:
        for heading_gain in HEADING_GAINS:
            for pretension in PRETENSIONS:
                specs[cell_name(pretension, heading_gain, intensity)] = FleetRunSpec(
                    pretension=pretension, heading_gain=heading_gain, weather_scale=intensity, duration=DURATION, warmup=WARMUP
                )
    specs[null_cell_name()] = FleetRunSpec(
        pretension=NULL_PRETENSION, heading_gain=NULL_HEADING_GAIN, weather_scale=NULL_INTENSITY, duration=DURATION, warmup=WARMUP
    )
    return specs


def parse_cell(name: str) -> tuple[float, float, float]:
    parts = name.replace("NULL_", "").split("_")
    return float(parts[0][1:]), float(parts[1][1:]), float(parts[2][1:])


def all_jobs() -> list[StationaryJob]:
    jobs = []
    for name, spec in cell_specs().items():
        for seed in PILOT_SEEDS:
            jobs.append(StationaryJob(name, spec, seed, pilot=True))
        for seed in STAT_SEEDS:
            jobs.append(StationaryJob(name, spec, seed, pilot=False))
    return jobs


# ----------------------------------------------------------------------------- predict


def _weather_relative_loads(intensity: float, seeds, directions: np.ndarray) -> list[np.ndarray]:
    return [
        gusttheory.relative_load_series(
            stationary_weather_forces(seed, DURATION, vessel_count=directions.shape[0], scale=intensity), directions
        )
        for seed in seeds
    ]


def _gust_summary(series_list: list[np.ndarray], pretension: float) -> dict:
    period = constants.WEATHER_PERIOD
    n = series_list[0].shape[1]
    exposure = sum(s.shape[0] for s in series_list) * period
    stacked = np.concatenate(series_list)
    per_cable = []
    for cable in range(n):
        durations, excesses = [], []
        for series in series_list:
            found = gusttheory.episodes(series[:, cable], pretension, period)
            durations.append(found["duration"])
            excesses.append(found["mean_excess"])
        durations = np.concatenate(durations)
        excesses = np.concatenate(excesses)
        per_cable.append({
            "episode_rate": durations.size / exposure,
            "mean_duration": float(np.mean(durations)) if durations.size else float("nan"),
            "episode_count": int(durations.size),
            "mu_w": float(np.mean(stacked[:, cable])),
            "sigma_w": float(np.std(stacked[:, cable])),
            "durations": durations.tolist(),
            "mean_excesses": excesses.tolist(),
        })
    return {"exposure": exposure, "cables": per_cable}


def theorem6_cell_rate(levels: np.ndarray, pretension: float, gust: dict, impact_factor: float = IMPACT_FACTOR) -> np.ndarray:
    total = np.zeros(np.asarray(levels, dtype=float).shape)
    for cable in gust["cables"]:
        if cable["episode_count"] == 0:
            continue
        total += gusttheory.theorem6_snap_rate(
            levels, pretension, cable["episode_rate"], cable["mean_duration"], cable["mu_w"], cable["sigma_w"], impact_factor
        )
    return total


def corollary5_cell_rate(levels: np.ndarray, pretension: float, gust: dict, impact_factor: float = IMPACT_FACTOR) -> np.ndarray:
    total = np.zeros(np.asarray(levels, dtype=float).shape)
    for cable in gust["cables"]:
        total += gusttheory.corollary5_functional_rate(
            levels, pretension, np.asarray(cable["durations"]), np.asarray(cable["mean_excesses"]), gust["exposure"], impact_factor
        )
    return total


def theorem6_exact_cell_rate(levels: np.ndarray, pretension: float, gust: dict, impact_factor: float = IMPACT_FACTOR) -> np.ndarray:
    total = np.zeros(np.asarray(levels, dtype=float).shape)
    for cable in gust["cables"]:
        if cable["episode_count"] == 0:
            continue
        total += gusttheory.theorem6_exact_snap_rate(
            levels, pretension, cable["episode_rate"], cable["mean_duration"], cable["mu_w"], cable["sigma_w"], impact_factor
        )
    return total


def corollary5_exact_cell_rate(levels: np.ndarray, pretension: float, gust: dict, impact_factor: float = IMPACT_FACTOR) -> np.ndarray:
    total = np.zeros(np.asarray(levels, dtype=float).shape)
    for cable in gust["cables"]:
        total += gusttheory.corollary5_exact_functional_rate(
            levels, pretension, np.asarray(cable["durations"]), np.asarray(cable["mean_excesses"]), gust["exposure"], impact_factor
        )
    return total


def theorem6_cell_slope(pretension: float, gust: dict, impact_factor: float = IMPACT_FACTOR) -> float | None:
    usable = [c for c in gust["cables"] if c["episode_count"] > 0]
    if not usable:
        return None
    t_g = float(np.mean([c["mean_duration"] for c in usable]))
    sigma = float(np.mean([c["sigma_w"] for c in usable]))
    return gusttheory.theorem6_log_slope(pretension, t_g, sigma, impact_factor)


def _lti(pretension: float, heading_gain: float, intensity: float):
    from tether.theory.reduced_lti import ReducedModelInput, reduced_model

    return reduced_model(ReducedModelInput(pretension=pretension, heading_gain=heading_gain, weather_scale=intensity))


def _lti_summary(result) -> dict:
    return {
        "mu_e": result.mu_e.tolist(),
        "sigma_e": result.sigma_e.tolist(),
        "sigma_edot": result.sigma_edot.tolist(),
        "sigma_eddot": result.sigma_eddot.tolist(),
        "mu_q": result.mu_q.tolist(),
        "sigma_q": result.sigma_q.tolist(),
        "sigma_qdot": result.sigma_qdot.tolist(),
        "mu_w": result.mu_w.tolist(),
        "sigma_w": result.sigma_w.tolist(),
        "sigma_chord_angle": result.sigma_chord_angle.tolist(),
        "decorrelation_lag_s": np.asarray(result.decorrelation_lag).tolist(),
        "integral_time_scale_s": np.asarray(result.integral_time_scale).tolist(),
        "corr_q": result.corr_q.tolist(),
        "spectral_radius": float(result.spectral_radius),
        "unstable_eigenvalues": len(np.atleast_1d(getattr(result, "unstable_eigenvalues", []))),
    }


def common_threshold_rule() -> dict:
    """Largest whole-kN threshold powered (>= 20 snap + taut exceedances) in every
    Phase 1(d) cell at intensity 1.0 - fixed from Phase 1 data before Phase 2 runs."""
    with np.load(RECORDS / "phase1" / "phase1_close_records.npz", allow_pickle=False) as data:
        names = [str(n) for n in data["cell_names"]]
        cell_index = data["marks_cell_index"]
        peaks = data["marks_T_peak"]
        level_counts = data["cell_level_counts"]
        grid = data["level_grid"]
    candidates = []
    for threshold in np.arange(1000.0, 40001.0, 1000.0):
        level = int(np.argmin(np.abs(grid - threshold)))
        counts = {}
        for index, name in enumerate(names):
            if not name.endswith("_I1.0"):
                continue
            snaps = int(np.sum(peaks[cell_index == index] > threshold))
            taut = int(level_counts[index][level].sum())
            counts[name] = snaps + taut
        if counts and min(counts.values()) >= POWER_MIN:
            candidates.append((float(threshold), counts))
    if not candidates:
        return {"threshold": None, "counts": {}}
    threshold, counts = candidates[-1]
    return {"threshold": threshold, "counts": counts}


def predict() -> dict:
    """Compute and commit every Phase 2 prediction before any plant run."""
    global COMMON_THRESHOLD
    rule = common_threshold_rule()
    if rule["threshold"] is not None:
        COMMON_THRESHOLD = rule["threshold"]
    geometry = formation_geometry("parallel")
    directions = np.stack([np.cos(geometry.cable_angles), np.sin(geometry.cable_angles)], axis=1)
    phase1 = json.loads((RECORDS / "phase1" / "phase1_close_results.json").read_text())
    delta_max = phase1["tests"]["P1-T8"]["observed_max_complete_depth"]["depth"]
    gust = {}
    optimum_gust = {}
    for intensity in INTENSITIES + (NULL_INTENSITY,):
        series = _weather_relative_loads(intensity, WEATHER_SEEDS, directions)
        levels = PRETENSIONS if intensity != NULL_INTENSITY else (NULL_PRETENSION,)
        for pretension in levels:
            gust[f"T{int(pretension)}_I{intensity}"] = _gust_summary(series, pretension)
        if intensity in INTENSITIES:
            for pretension in OPTIMUM_GRID:
                summary = _gust_summary(series, pretension)
                optimum_gust[f"T{int(pretension)}_I{intensity}"] = {
                    "rate": float(theorem6_cell_rate(np.array([COMMON_THRESHOLD]), pretension, summary)[0]),
                    "functional_rate": float(corollary5_cell_rate(np.array([COMMON_THRESHOLD]), pretension, summary)[0]),
                }
    lti = {}
    for name in cell_specs():
        pretension, heading_gain, intensity = parse_cell(name)
        lti[name] = _lti_summary(_lti(pretension, heading_gain, intensity))
    optimum = {}
    unit = {(p, h): _lti(p, h, 1.0) for p in OPTIMUM_GRID for h in HEADING_GAINS}
    for intensity in INTENSITIES:
        for heading_gain in HEADING_GAINS:
            totals = []
            for pretension in OPTIMUM_GRID:
                result = unit[(pretension, heading_gain)]
                taut = float(np.sum(gusttheory.taut_rate(np.array([COMMON_THRESHOLD]), pretension, intensity * result.sigma_q, intensity * result.sigma_qdot)))
                snap = optimum_gust[f"T{int(pretension)}_I{intensity}"]["rate"]
                totals.append((pretension, snap, taut, snap + taut))
            best = min(totals, key=lambda row: row[3])
            optimum[f"k{int(heading_gain)}_I{intensity}"] = {
                "grid": [row[0] for row in totals],
                "snap": [row[1] for row in totals],
                "taut": [row[2] for row in totals],
                "total": [row[3] for row in totals],
                "argmin": best[0],
                "all_zero": bool(all(row[3] == 0.0 for row in totals)),
            }
    null_series = _weather_relative_loads(NULL_INTENSITY, PILOT_SEEDS + STAT_SEEDS, directions)
    null_max = float(max(np.max(s) for s in null_series)) / NULL_PRETENSION
    predictions = {
        "schema_version": 1,
        "source": source_state(),
        "declarations": DECLARATIONS,
        "constants": {
            "impact_factor": IMPACT_FACTOR,
            "stiffness": constants.CABLE_STIFFNESS,
            "reduced_mass": gusttheory.PAIR_REDUCED_MASS,
            "common_threshold": COMMON_THRESHOLD,
            "common_threshold_rule": rule,
            "delta_max_phase1": delta_max,
            "quantiles": QUANTILES,
            "power_min": POWER_MIN,
        },
        "gust": gust,
        "lti": lti,
        "optimum": optimum,
        "null_weather_max_ratio": null_max,
        "null_ratio_below_one": null_max < 1.0,
    }
    write_bytes(PREDICTIONS_PATH, json_bytes(predictions))
    return predictions


# ----------------------------------------------------------------------------- analyse


def _pool(summaries: list[dict]) -> dict:
    marks = {key: np.concatenate([s["marks"][key] for s in summaries]) for key in summaries[0]["marks"]}
    marks["seed"] = np.concatenate([np.full(s["marks"]["t_up"].size, s["meta"]["seed"]) for s in summaries])
    moments = {key: np.sum([s["moments"][key] for s in summaries], axis=0) for key in summaries[0]["moments"]}
    return {
        "marks": marks,
        "moments": moments,
        "level_counts": np.sum([s["level_counts"] for s in summaries], axis=0),
        "level_counts_by_seed": np.stack([s["level_counts"].sum(axis=1) for s in summaries]),
        "exposures": np.array([s["meta"]["exposure"] for s in summaries]),
        "seeds": np.array([s["meta"]["seed"] for s in summaries]),
        "exposure": float(sum(s["meta"]["exposure"] for s in summaries)),
        "onsets": np.sum([s["onsets"] for s in summaries], axis=0),
        "slack_samples": np.sum([s["slack_samples"] for s in summaries], axis=0),
        "window_samples": int(sum(s["meta"]["window_samples"] for s in summaries)),
        "samples": {key: np.concatenate([s["samples"][key] for s in summaries]) for key in summaries[0]["samples"]},
        "samples_seed": np.concatenate([np.full(s["samples"]["e"].shape[0], s["meta"]["seed"]) for s in summaries]),
        "taut_peaks": {key: np.concatenate([s["taut_peaks"][key] for s in summaries]) for key in summaries[0]["taut_peaks"]},
        "closures": [s["meta"]["closure"] for s in summaries],
        "wall_seconds": float(sum(s["meta"]["wall_seconds"] for s in summaries)),
        "sim_seconds": float(sum(s["meta"]["sim_seconds"] for s in summaries)),
        "tracker_dropped": int(sum(s["meta"]["tracker_dropped"] for s in summaries)),
        "cable_angle_std": np.nanmean([s["geometry_stats"]["cable_angle_std"] for s in summaries], axis=0),
        "load_yaw_std": float(np.nanmean([s["geometry_stats"]["load_yaw_std"] for s in summaries])),
    }


def _moment_stats(moments: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    count, total, square = moments[name]
    mean = total / np.maximum(count, 1.0)
    variance = square / np.maximum(count, 1.0) - mean**2
    return mean, np.sqrt(np.maximum(variance, 0.0))


def _grid_from_pilot(pilot: dict, pretension: float) -> list[float] | None:
    depth = pilot["marks"]["depth"]
    depth = depth[depth > 0.0]
    if depth.size < 10:
        return None
    return [IMPACT_FACTOR * math.sqrt(2.0 * constants.CABLE_STIFFNESS * pretension * float(np.percentile(depth, q))) for q in QUANTILES]


def _taut_grid_from_pilot(pilot: dict) -> list[float] | None:
    peaks = pilot["taut_peaks"]["peak"]
    if peaks.size < 10:
        return None
    grid = []
    for q in QUANTILES:
        value = float(np.percentile(peaks, q))
        grid.append(float(LEVEL_GRID[int(np.argmin(np.abs(LEVEL_GRID - value)))]))
    return sorted(set(grid))


def _seed_counts(marks: dict, seeds: np.ndarray, threshold: float) -> np.ndarray:
    hits = marks["T_peak"] > threshold
    return np.array([int(np.sum(hits & (marks["seed"] == seed))) for seed in seeds])


def _log_slope(x: np.ndarray, rates: np.ndarray) -> float | None:
    mask = rates > 0.0
    if mask.sum() < 2:
        return None
    return float(np.polyfit(x[mask], np.log(rates[mask]), 1)[0])


def analyse_cells(cells: dict[str, dict], pilots: dict[str, dict], predictions: dict) -> dict:
    from tether.evt.bootstrap import seed_bootstrap_rate
    from tether.evt.gof import anderson_darling_normal, ks_parametric_bootstrap

    rng = np.random.default_rng(20260912)
    geometry = formation_geometry("parallel")
    rows = {}
    for name, cell in cells.items():
        if name.startswith("NULL_"):
            continue
        pretension, heading_gain, intensity = parse_cell(name)
        marks = cell["marks"]
        gust = predictions["gust"][f"T{int(pretension)}_I{intensity}"]
        lti = predictions["lti"][name]
        row: dict = {"pretension": pretension, "heading_gain": heading_gain, "intensity": intensity}
        dwell = marks["dwell"]
        row["excursions"] = int(dwell.size)
        row["exposure"] = cell["exposure"]
        row["slack_duty"] = float(np.sum(cell["slack_samples"]) / (cell["window_samples"] * geometry.vessel_count))
        row["dwell_p95"] = float(np.percentile(dwell, 95)) if dwell.size else None
        row["multi_max_fraction"] = float(np.mean(marks["n_maxima"] > 1)) if dwell.size else None
        row["closures"] = int(sum(c is not None for c in cell["closures"]))
        row["cable_angle_std_deg"] = float(np.degrees(np.mean(cell["cable_angle_std"])))
        row["load_yaw_std_deg"] = math.degrees(cell["load_yaw_std"])
        row["in_domain"] = bool(
            dwell.size > 0 and row["dwell_p95"] < TAU_W / 4.0 and row["multi_max_fraction"] < 0.10 and row["slack_duty"] < 0.10
        )
        # snap channel
        grid = _grid_from_pilot(pilots[name], pretension)
        row["grid"] = grid
        snap = []
        if grid is not None:
            predicted = theorem6_cell_rate(np.array(grid), pretension, gust)
            functional = corollary5_cell_rate(np.array(grid), pretension, gust)
            exact = theorem6_exact_cell_rate(np.array(grid), pretension, gust)
            exact_functional = corollary5_exact_cell_rate(np.array(grid), pretension, gust)
            for threshold, pred, func, ex, exf in zip(grid, predicted, functional, exact, exact_functional):
                counts = _seed_counts(marks, cell["seeds"], threshold)
                rate, lo, hi = seed_bootstrap_rate(counts, cell["exposures"], n_boot=2000, rng=rng)
                long_share = float(np.mean(dwell[marks["T_peak"] > threshold] >= TAU_W / 4.0)) if counts.sum() else None
                snap.append({
                    "threshold": threshold,
                    "count": int(counts.sum()),
                    "powered": bool(counts.sum() >= POWER_MIN),
                    "rate": rate,
                    "rate_lo": lo,
                    "rate_hi": hi,
                    "predicted_theorem6": float(pred),
                    "predicted_corollary5": float(func),
                    "predicted_theorem6_exact": float(ex),
                    "predicted_corollary5_exact": float(exf),
                    "ratio_exact": rate / ex if ex > 0 else None,
                    "ratio": rate / pred if pred > 0 else None,
                    "ratio_lo": lo / pred if pred > 0 else None,
                    "ratio_hi": hi / pred if pred > 0 else None,
                    "long_excursion_share": long_share,
                })
        row["snap"] = snap
        powered = [s for s in snap if s["powered"]]
        row["powered"] = bool(powered)
        if powered:
            ratios_ok = all(s["ratio"] is not None and 0.5 <= s["ratio"] <= 2.0 for s in powered)
            x = np.array([s["threshold"] ** 4 for s in powered])
            measured_slope = _log_slope(x, np.array([s["rate"] for s in powered]))
            literal = theorem6_cell_slope(pretension, gust)
            secant = _log_slope(x, np.array([s["predicted_theorem6"] for s in powered]))
            row["t5"] = {
                "ratios_within": ratios_ok,
                "ratios_within_exact": all(s["ratio_exact"] is not None and 0.5 <= s["ratio_exact"] <= 2.0 for s in powered),
                "measured_slope": measured_slope,
                "literal_slope": literal,
                "secant_slope": secant,
                "slope_within_literal": bool(measured_slope is not None and literal not in (None, 0.0) and abs(measured_slope / literal - 1.0) <= 0.2),
                "slope_within_secant": bool(measured_slope is not None and secant not in (None, 0.0) and abs(measured_slope / secant - 1.0) <= 0.2),
            }
        # taut channel (P2-T6)
        taut_grid = _taut_grid_from_pilot(pilots[name])
        taut_rows = []
        if taut_grid is not None:
            mean_q, sigma_q_meas = _moment_stats(cell["moments"], "q")
            _, sigma_qdot_meas = _moment_stats(cell["moments"], "qdot")
            for level in taut_grid:
                index = int(np.argmin(np.abs(LEVEL_GRID - level)))
                count = int(cell["level_counts"][index].sum())
                by_seed = cell["level_counts_by_seed"][:, index]
                rate, lo, hi = seed_bootstrap_rate(by_seed, cell["exposures"], n_boot=2000, rng=rng)
                pred_lti = float(np.sum(gusttheory.taut_rate(np.array([level]), pretension, np.array(lti["sigma_q"]), np.array(lti["sigma_qdot"]))))
                pred_meas = float(np.sum(gusttheory.taut_rate(np.array([level]), mean_q, sigma_q_meas, sigma_qdot_meas)))
                taut_rows.append({
                    "threshold": level, "count": count, "powered": count >= POWER_MIN, "rate": rate, "rate_lo": lo, "rate_hi": hi,
                    "predicted_lti": pred_lti, "predicted_measured_moments": pred_meas,
                    "ratio_lti": rate / pred_lti if pred_lti > 0 else None,
                    "ratio_measured": rate / pred_meas if pred_meas > 0 else None,
                })
            tp = [r for r in taut_rows if r["powered"]]
            if tp:
                x = np.array([r["threshold"] ** 2 for r in tp])
                slope = _log_slope(x, np.array([r["rate"] for r in tp]))
                literal = -1.0 / (2.0 * float(np.mean(lti["sigma_q"])) ** 2)
                row["t6"] = {
                    "ratios_within_lti": all(r["ratio_lti"] is not None and 0.5 <= r["ratio_lti"] <= 2.0 for r in tp),
                    "ratios_within_measured": all(r["ratio_measured"] is not None and 0.5 <= r["ratio_measured"] <= 2.0 for r in tp),
                    "measured_slope": slope,
                    "literal_slope_lti": literal,
                    "slope_within": bool(slope is not None and abs(slope / literal - 1.0) <= 0.2),
                }
        row["taut"] = taut_rows
        # reduced model (P2-T0)
        mean_e, sigma_e = _moment_stats(cell["moments"], "e")
        _, sigma_edot = _moment_stats(cell["moments"], "edot")
        _, sigma_eddot = _moment_stats(cell["moments"], "eddot")
        mean_q, sigma_q = _moment_stats(cell["moments"], "q")
        _, sigma_qdot = _moment_stats(cell["moments"], "qdot")
        mean_w, sigma_w = _moment_stats(cell["moments"], "wrel")
        measured = {"mu_e": mean_e, "sigma_e": sigma_e, "sigma_edot": sigma_edot, "sigma_q": sigma_q, "sigma_qdot": sigma_qdot, "sigma_w": sigma_w}
        ratios = {}
        for key, values in measured.items():
            predicted = np.asarray(lti[key])
            ratios[key] = float(np.median(predicted / np.maximum(values, 1.0e-12)))
        row["t0"] = {
            "median_predicted_over_measured": ratios,
            "within_factor_1_5": all(1.0 / 1.5 <= value <= 1.5 for value in ratios.values()),
            "measured": {key: values.tolist() for key, values in measured.items()},
        }
        # Gaussian taut state (P2-T1)
        lag = max(1.0, float(np.max(lti["decorrelation_lag_s"])))
        spacing = int(math.ceil(lag))
        samples = cell["samples"]
        taut = samples["taut"][::spacing]
        ad = {}
        for key in ("e", "edot", "q"):
            values = samples[key][::spacing][:, 2]
            values = values[taut[:, 2]]
            ad[key] = anderson_darling_normal(values)[1] if values.size >= 8 else None
        identity_lhs = sigma_qdot**2
        identity_rhs = constants.CABLE_STIFFNESS**2 * sigma_edot**2 + constants.CABLE_DAMPING**2 * sigma_eddot**2
        row["t1"] = {
            "decimation_s": spacing,
            "ad_p_centre_cable": ad,
            "ad_all_above_0_05": all(p is not None and p > 0.05 for p in ad.values()),
            "identity_max_rel_error": float(np.max(np.abs(identity_lhs / identity_rhs - 1.0))),
        }
        # onset rate and entry law (P2-T2)
        rice = (sigma_edot / (2.0 * math.pi * np.maximum(sigma_e, 1.0e-12))) * np.exp(-(mean_e**2) / (2.0 * np.maximum(sigma_e, 1.0e-12) ** 2))
        onset_rate = cell["onsets"] / cell["exposure"]
        entry = marks["u_entry"][marks["u_entry"] > 0.0]
        ks = ks_parametric_bootstrap(entry, "rayleigh", n_boot=999, rng=rng) if entry.size >= 10 else (None, None)
        row["t2"] = {
            "onset_rate": onset_rate.tolist(),
            "rice_rate": rice.tolist(),
            "ratio": (onset_rate / np.maximum(rice, 1.0e-300)).tolist(),
            "ratio_within": bool(np.sum(cell["onsets"]) > 0 and np.all((onset_rate / np.maximum(rice, 1.0e-300) >= 0.7) & (onset_rate / np.maximum(rice, 1.0e-300) <= 1.4))),
            "onsets": int(np.sum(cell["onsets"])),
            "ks_p": ks[1],
            "entry_count": int(entry.size),
        }
        # depth tail (P2-T3)
        depth_rows = []
        if grid is not None:
            pilot_depth = pilots[name]["marks"]["depth"]
            pilot_depth = pilot_depth[pilot_depth > 0.0]
            for q in QUANTILES:
                delta = float(np.percentile(pilot_depth, q))
                count = int(np.sum(marks["depth"] > delta))
                predicted = 0.0
                for cable in gust["cables"]:
                    if cable["episode_count"] == 0:
                        continue
                    level = pretension + 2.0 * gusttheory.PAIR_REDUCED_MASS * delta / cable["mean_duration"] ** 2
                    from scipy.stats import norm

                    predicted += cable["episode_rate"] * norm.sf((level - cable["mu_w"]) / cable["sigma_w"]) / norm.sf((pretension - cable["mu_w"]) / cable["sigma_w"])
                rate = count / cell["exposure"]
                depth_rows.append({"depth": delta, "count": count, "powered": count >= POWER_MIN, "rate": rate, "predicted": predicted, "ratio": rate / predicted if predicted > 0 else None})
        row["t3"] = depth_rows
        # closed-form snap law (P2-T7)
        if grid is not None:
            select = (marks["T_peak"] > grid[0]) & (marks["exceed_duration"] > 0.0) & (marks["exceed_mean"] > 0.0)
            prediction = IMPACT_FACTOR * marks["exceed_duration"][select] * np.sqrt(
                constants.CABLE_STIFFNESS * pretension * marks["exceed_mean"][select] / gusttheory.PAIR_REDUCED_MASS
            )
            ratio = marks["T_peak"][select] / prediction
            exact = IMPACT_FACTOR * marks["exceed_duration"][select] * np.sqrt(
                constants.CABLE_STIFFNESS * (pretension + marks["exceed_mean"][select]) * marks["exceed_mean"][select] / gusttheory.PAIR_REDUCED_MASS
            )
            ratio_exact = marks["T_peak"][select] / exact
            row["t7"] = {
                "n": int(select.sum()),
                "median_ratio_exact_form": float(np.median(ratio_exact)) if ratio_exact.size else None,
                "median_ratio": float(np.median(ratio)) if ratio.size else None,
                "fraction_within_25pct": float(np.mean(np.abs(ratio - 1.0) <= 0.25)) if ratio.size else None,
                "pass": bool(ratio.size >= POWER_MIN and np.median(np.abs(ratio - 1.0)) <= 0.25),
            }
        # saturation (P2-T9)
        bound = IMPACT_FACTOR * math.sqrt(2.0 * constants.CABLE_STIFFNESS * pretension * predictions["constants"]["delta_max_phase1"])
        largest = float(np.max(marks["T_peak"])) if marks["T_peak"].size else 0.0
        row["t9"] = {"largest_snap": largest, "bound": bound, "below_bound": largest <= bound, "bound_within_30pct": bool(largest > 0 and abs(bound / largest - 1.0) <= 0.3)}
        # common threshold rate (P2-T8)
        counts = _seed_counts(marks, cell["seeds"], COMMON_THRESHOLD)
        index = int(np.argmin(np.abs(LEVEL_GRID - COMMON_THRESHOLD)))
        taut_count = int(cell["level_counts"][index].sum())
        row["common"] = {"snap_count": int(counts.sum()), "taut_count": taut_count, "total_rate": (counts.sum() + taut_count) / cell["exposure"]}
        rows[name] = row
    return rows


def evaluate_tests(rows: dict, cells: dict, predictions: dict) -> dict:
    tests = {}
    in_domain = [r for r in rows.values() if r["in_domain"]]
    powered = [r for r in rows.values() if r["powered"]]
    powered_in = [r for r in in_domain if r["powered"]]

    def fraction(items, key):
        values = [bool(item) for item in items]
        return (float(np.mean(values)) if values else None, len(values))

    t0_frac, t0_n = fraction([r["t0"]["within_factor_1_5"] for r in in_domain], "t0")
    tests["P2-T0"] = {"statement": "reduced-model moments within a factor 1.5 in >= 80% of in-domain cells (non-blocking)", "fraction": t0_frac, "cells": t0_n,
                      "fraction_all_cells": fraction([r["t0"]["within_factor_1_5"] for r in rows.values()], "")[0],
                      "verdict": "UNDER-POWERED" if t0_n == 0 else ("PASS" if t0_frac >= 0.8 else "FAIL")}
    t1_frac, t1_n = fraction([r["t1"]["ad_all_above_0_05"] for r in rows.values()], "")
    identity = max(r["t1"]["identity_max_rel_error"] for r in rows.values())
    tests["P2-T1"] = {"statement": "AD p > 0.05 on e, edot, q in >= 80% of cells; spectral-moment identity within 10%", "fraction_ad": t1_frac, "cells": t1_n,
                      "identity_max_rel_error": identity, "verdict": "PASS" if (t1_frac or 0) >= 0.8 and identity <= 0.10 else "FAIL"}
    with_onsets = [r for r in rows.values() if r["t2"]["onsets"] > 0]
    ks_cells = [r for r in with_onsets if r["t2"]["ks_p"] is not None]
    rice_frac = float(np.mean([r["t2"]["ratio_within"] for r in with_onsets])) if with_onsets else None
    ks_frac = float(np.mean([r["t2"]["ks_p"] > 0.05 for r in ks_cells])) if ks_cells else None
    tests["P2-T2"] = {"statement": "onset rate within [0.7, 1.4] of Rice; Rayleigh entry speeds (parametric-bootstrap KS p > 0.05) in >= 80% of cells",
                      "cells_with_onsets": len(with_onsets), "fraction_rice_within": rice_frac, "fraction_ks_pass": ks_frac,
                      "verdict": "UNDER-POWERED" if not with_onsets else ("PASS" if (rice_frac or 0) >= 0.8 and (ks_frac or 0) >= 0.8 else "FAIL")}
    depth_ok = []
    for r in rows.values():
        powered_depth = [d for d in r["t3"] if d["powered"]]
        if powered_depth:
            depth_ok.append(all(d["ratio"] is not None and 0.5 <= d["ratio"] <= 2.0 for d in powered_depth))
    tests["P2-T3"] = {"statement": "depth tail rate within a factor 2 of the Gaussian prediction over the powered range", "cells": len(depth_ok),
                      "fraction_pass": float(np.mean(depth_ok)) if depth_ok else None,
                      "verdict": "UNDER-POWERED" if not depth_ok else ("PASS" if np.mean(depth_ok) >= 0.8 else "FAIL")}
    null = cells[null_cell_name()]
    reference_grid = rows[REFERENCE_CELL]["grid"]
    null_counts = None
    if reference_grid is not None:
        null_counts = [int(np.sum((null["marks"]["T_peak"] > reference_grid[0]) & (null["marks"]["seed"] == seed))) for seed in null["seeds"]]
    tests["P2-T4"] = {"statement": "zero snaps in every seed of the null cell at the powered grid", "null_cell": null_cell_name(),
                      "weather_max_ratio": predictions["null_weather_max_ratio"], "excursions": int(null["marks"]["t_up"].size),
                      "threshold": reference_grid[0] if reference_grid else None, "snaps_per_seed": null_counts,
                      "verdict": "UNDER-POWERED" if null_counts is None else ("PASS" if sum(null_counts) == 0 else "FAIL")}
    t5_cells = powered_in
    ratio_frac = float(np.mean([r["t5"]["ratios_within"] for r in t5_cells])) if t5_cells else None
    slope_frac = float(np.mean([r["t5"]["slope_within_literal"] for r in t5_cells])) if t5_cells else None
    all_ratio_frac = float(np.mean([r["t5"]["ratios_within"] for r in powered])) if powered else None
    all_slope_frac = float(np.mean([r["t5"]["slope_within_literal"] for r in powered])) if powered else None
    all_secant_frac = float(np.mean([r["t5"]["slope_within_secant"] for r in powered])) if powered else None
    all_exact_frac = float(np.mean([r["t5"]["ratios_within_exact"] for r in powered])) if powered else None
    if not t5_cells:
        t5_verdict = "UNDER-POWERED (no powered in-domain cell)"
    else:
        t5_verdict = "PASS" if ratio_frac >= 0.8 and slope_frac >= 0.8 else "FAIL"
    tests["P2-T5"] = {"statement": "snap-rate ratio within [1/2, 2] at every powered threshold in >= 80% of in-domain cells; slope within 20% of -kappa^2/(2 sigma_W^2)",
                      "in_domain_powered_cells": len(t5_cells), "fraction_ratio_in_domain": ratio_frac, "fraction_slope_in_domain": slope_frac,
                      "powered_cells_all": len(powered), "fraction_ratio_all_powered": all_ratio_frac, "fraction_slope_literal_all_powered": all_slope_frac,
                      "fraction_slope_secant_all_powered": all_secant_frac, "fraction_ratio_exact_all_powered": all_exact_frac, "verdict": t5_verdict}
    taut_cells = [r for r in rows.values() if "t6" in r]
    tests["P2-T6"] = {"statement": "taut-rate ratio within [1/2, 2]; slope within 20% of -1/(2 sigma_q^2)", "cells": len(taut_cells),
                      "fraction_ratio_lti": float(np.mean([r["t6"]["ratios_within_lti"] for r in taut_cells])) if taut_cells else None,
                      "fraction_ratio_measured": float(np.mean([r["t6"]["ratios_within_measured"] for r in taut_cells])) if taut_cells else None,
                      "fraction_slope": float(np.mean([r["t6"]["slope_within"] for r in taut_cells])) if taut_cells else None}
    tests["P2-T6"]["verdict"] = "UNDER-POWERED" if not taut_cells else ("PASS" if (tests["P2-T6"]["fraction_ratio_lti"] >= 0.8 and tests["P2-T6"]["fraction_slope"] >= 0.8) else "FAIL")
    t7_cells = [r for r in rows.values() if "t7" in r and r["t7"]["n"] >= POWER_MIN]
    tests["P2-T7"] = {"statement": "measured T_snap within 25% of the closed-form snap law over the powered range", "cells": len(t7_cells),
                      "fraction_pass": float(np.mean([r["t7"]["pass"] for r in t7_cells])) if t7_cells else None,
                      "median_ratios": {k: r["t7"]["median_ratio"] for k, r in rows.items() if "t7" in r}}
    tests["P2-T7"]["verdict"] = "UNDER-POWERED" if not t7_cells else ("PASS" if tests["P2-T7"]["fraction_pass"] >= 0.8 else "FAIL")
    optimum_rows = []
    for intensity in INTENSITIES:
        for heading_gain in HEADING_GAINS:
            group = [(p, rows[cell_name(p, heading_gain, intensity)]["common"]) for p in PRETENSIONS]
            totals = [(p, c["total_rate"], c["snap_count"] + c["taut_count"]) for p, c in group]
            measured_argmin = min(totals, key=lambda row: row[1])[0]
            predicted = predictions["optimum"][f"k{int(heading_gain)}_I{intensity}"]
            span = PRETENSIONS[-1] - PRETENSIONS[0]
            powered_group = sum(t[2] for t in totals) >= POWER_MIN
            optimum_rows.append({"intensity": intensity, "heading_gain": heading_gain, "totals": totals, "measured_argmin": measured_argmin,
                                 "predicted_argmin": predicted["argmin"], "predicted_all_zero": predicted["all_zero"], "powered": powered_group,
                                 "within_20pct_span": abs(measured_argmin - predicted["argmin"]) <= 0.2 * span})
    powered_groups = [r for r in optimum_rows if r["powered"] and not r["predicted_all_zero"]]
    tests["P2-T8"] = {"statement": "argmin over T0 of the total rate at the common threshold within 20% of the span of the predicted optimum",
                      "common_threshold": COMMON_THRESHOLD, "groups": optimum_rows,
                      "verdict": "UNDER-POWERED" if not powered_groups else ("PASS" if all(r["within_20pct_span"] for r in powered_groups) else "FAIL")}
    t9_rows = [r["t9"] for r in rows.values() if r["t9"]["largest_snap"] > 0]
    tests["P2-T9"] = {"statement": "largest observed snap below f sqrt(2 k T0 Delta_max) and the bound within 30% of it", "cells": len(t9_rows),
                      "all_below": all(t["below_bound"] for t in t9_rows) if t9_rows else None,
                      "fraction_within_30pct": float(np.mean([t["bound_within_30pct"] for t in t9_rows])) if t9_rows else None}
    tests["P2-T9"]["verdict"] = "UNDER-POWERED" if not t9_rows else ("PASS" if tests["P2-T9"]["all_below"] and tests["P2-T9"]["fraction_within_30pct"] >= 0.8 else "FAIL")
    region = [{"cell": k, "slack_duty": r["slack_duty"], "dwell_p95": r["dwell_p95"], "long_share_top": (r["snap"][-1]["long_excursion_share"] if r["snap"] else None),
               "t5_cell_pass": bool(r.get("t5", {}).get("ratios_within") and r.get("t5", {}).get("slope_within_literal"))} for k, r in rows.items() if r["powered"]]
    passing = [item for item in region if item["t5_cell_pass"]]
    tests["P2-T10"] = {"statement": "largest connected region of (duty, duration, depth-share) space in which T5 passes", "powered_cells": region,
                       "passing_cells": [p["cell"] for p in passing],
                       "bounding_box": ({"slack_duty": [min(p["slack_duty"] for p in passing), max(p["slack_duty"] for p in passing)],
                                         "dwell_p95": [min(p["dwell_p95"] for p in passing), max(p["dwell_p95"] for p in passing)]} if passing else None),
                       "verdict": "NON-EMPTY" if passing else "EMPTY"}
    return tests


def gate(tests: dict) -> dict:
    """Phase 2 outcome matrix, applied mechanically."""
    t5 = tests["P2-T5"]["verdict"]
    t2 = tests["P2-T2"]["verdict"]
    t4 = tests["P2-T4"]["verdict"]
    t6 = tests["P2-T6"]["verdict"]
    region = tests["P2-T10"]["verdict"]
    if t4 == "FAIL":
        return {"decision": "STOP-AND-INVESTIGATE", "rule": "T4 fails: a snap under nominal weather means some mechanism outside Part II drives excursions"}
    if t5 == "FAIL":
        return {"decision": "NO-GO", "rule": "T5 fails inside the domain at powered thresholds: the physics-structured programme is dead"}
    if t5 == "PASS" and t2 == "PASS" and t4 == "PASS" and region == "NON-EMPTY":
        decision = "GO"
        if t6 == "FAIL":
            decision = "PIVOT-snap"
        return {"decision": decision, "rule": "GO requires T2, T4, T5 and a non-empty T10 region" + ("; T6 failed with T5 passing: taut channel non-Gaussian" if t6 == "FAIL" else "")}
    return {"decision": "NOT-GO (conditions for GO unmet, NO-GO rule not triggered)", "rule": f"T2={t2}, T4={t4}, T5={t5}, T10={region}"}


def analyse() -> dict:
    global COMMON_THRESHOLD
    predictions = json.loads(PREDICTIONS_PATH.read_text())
    COMMON_THRESHOLD = float(predictions["constants"]["common_threshold"])
    jobs, summaries = load_cache(CACHE_PATH)
    stat: dict[str, list] = {}
    pilot: dict[str, list] = {}
    for job, summary in zip(jobs, summaries):
        (pilot if job.pilot else stat).setdefault(job.cell, []).append(summary)
    cells = {name: _pool(items) for name, items in stat.items()}
    pilots = {name: _pool(items) for name, items in pilot.items()}
    rows = analyse_cells(cells, pilots, predictions)
    tests = evaluate_tests(rows, cells, predictions)
    decision = gate(tests)
    names = sorted(cells)
    arrays = {"cell_names": np.array(names), "cell_exposure": np.array([cells[n]["exposure"] for n in names])}
    index = np.concatenate([np.full(cells[n]["marks"]["t_up"].size, i) for i, n in enumerate(names)])
    arrays["marks_cell_index"] = index
    for key in cells[names[0]]["marks"]:
        arrays[f"marks_{key}"] = np.concatenate([cells[n]["marks"][key] for n in names])
    arrays["level_grid"] = LEVEL_GRID
    arrays["level_counts"] = np.stack([cells[n]["level_counts"] for n in names])
    arrays["taut_peak_cell_index"] = np.concatenate([np.full(cells[n]["taut_peaks"]["peak"].size, i) for i, n in enumerate(names)])
    for key in ("time", "peak", "cable"):
        arrays[f"taut_peaks_{key}"] = np.concatenate([cells[n]["taut_peaks"][key] for n in names])
    record_sha = write_bytes(RECORD_PATH, npz_bytes(arrays))
    execution = {name: {"seeds": len(stat[name]), "pilot_seeds": len(pilot.get(name, [])), "wall_seconds": cells[name]["wall_seconds"],
                        "sim_seconds": cells[name]["sim_seconds"], "tracker_dropped": cells[name]["tracker_dropped"]} for name in names}
    probe_job = jobs[0]
    probe_summary = summaries[0]
    results = {
        "schema_version": 1,
        "source": source_state(),
        "configuration": {"cells": {k: asdict(v) for k, v in cell_specs().items()}, "pilot_seeds": PILOT_SEEDS, "stat_seeds": STAT_SEEDS},
        "configuration_sha256": config_sha256({k: asdict(v) for k, v in cell_specs().items()}),
        "predictions_sha256": sha256_file(PREDICTIONS_PATH),
        "execution": execution,
        "rows": rows,
        "tests": tests,
        "gate": decision,
        "record_sha256": record_sha,
        "probe": {"cell": probe_job.cell, "seed": probe_job.seed, "marks_t_up": probe_summary["marks"]["t_up"].tolist(),
                  "marks_T_peak": probe_summary["marks"]["T_peak"].tolist(), "moments_q": probe_summary["moments"]["q"].tolist()},
    }
    write_bytes(RESULTS_PATH, json_bytes(results))
    write_bytes(MANIFEST_PATH, json_bytes({
        "schema_version": 1, "driver": "python -m tether.campaign.phase2 compute && python -m tether.campaign.phase2 analyse",
        "configuration_sha256": results["configuration_sha256"], "seeds": {"pilot": PILOT_SEEDS, "stat": STAT_SEEDS},
        "predictions_path": relative(PREDICTIONS_PATH), "predictions_sha256": results["predictions_sha256"],
        "committed_record_path": relative(RECORD_PATH), "record_sha256": record_sha,
        "probe": {"command": "python -m tether.campaign.phase2 replay", "cell": probe_job.cell, "seed": probe_job.seed, "marks": "exact", "moments_absolute_tolerance": 1.0e-6},
    }))
    return results


def replay() -> bool:
    results = json.loads(RESULTS_PATH.read_text())
    probe = results["probe"]
    spec = cell_specs()[probe["cell"]]
    summary = run_stationary_job(StationaryJob(probe["cell"], spec, probe["seed"], pilot=probe["seed"] in PILOT_SEEDS))
    marks_equal = summary["marks"]["t_up"].tolist() == probe["marks_t_up"] and summary["marks"]["T_peak"].tolist() == probe["marks_T_peak"]
    reference = np.array(probe["moments_q"])
    moments_close = bool(np.allclose(summary["moments"]["q"], reference, rtol=0.0, atol=1.0e-6 * max(1.0, float(np.max(np.abs(reference))))))
    passed = marks_equal and moments_close
    print(json.dumps({"marks_exact": marks_equal, "moments_within_tolerance": moments_close, "passed": passed}, indent=2))
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("predict", "compute", "analyse", "replay"))
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    if arguments.command == "predict":
        predictions = predict()
        print(json.dumps({k: predictions[k] for k in ("null_weather_max_ratio", "null_ratio_below_one")}, indent=2))
        print(json.dumps({k: v["argmin"] for k, v in predictions["optimum"].items()}, indent=2))
    elif arguments.command == "compute":
        if not PREDICTIONS_PATH.exists():
            raise SystemExit("predictions must be committed before the campaign runs")
        compute_jobs(all_jobs(), CACHE_PATH, arguments.workers, "phase2")
    elif arguments.command == "analyse":
        results = analyse()
        for test_id, test in results["tests"].items():
            print(test_id, test["verdict"])
        print("GATE", results["gate"])
    else:
        sys.exit(0 if replay() else 1)


if __name__ == "__main__":
    main()
