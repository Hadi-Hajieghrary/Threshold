"""Phase 3: tail class and the index doubling (plan Part V, Phase 3).

Commands: ``shocks`` runs the deterministic shock-response cell; ``predict`` commits the
Theorem 7 predictions built from it; ``compute`` runs the stochastic cells; ``analyse``
evaluates P3-T1..T9; ``replay`` re-executes the probe.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import asdict, dataclass

import numpy as np

from tether.campaign.common import RECORDS, config_sha256, json_bytes, npz_bytes, relative, run_pool, sha256_file, source_state, write_bytes
from tether.campaign.fleet_run import FleetRunSpec, build_run, run_to_end
from tether.campaign.stationary import StationaryJob, compute_jobs, load_cache, run_stationary_job
from tether.physics import constants
from tether.physics.fleet import formation_geometry
from tether.physics.weather import ar1_coefficient, stationary_weather_forces
from tether.theory import gust as gusttheory

RECORD_DIR = RECORDS / "phase3"
SHOCK_CACHE = RECORD_DIR / "cache" / "phase3_shocks.pkl"
SHOCK_PATH = RECORD_DIR / "phase3_shock_response.json"
PREDICTIONS_PATH = RECORD_DIR / "phase3_predictions.json"
CACHE_PATH = RECORD_DIR / "cache" / "phase3_compute.pkl"
RESULTS_PATH = RECORD_DIR / "phase3_results.json"
RECORD_PATH = RECORD_DIR / "phase3_records.npz"
MANIFEST_PATH = RECORD_DIR / "phase3_manifest.json"

REFERENCE = {"pretension": 1000.0, "heading_gain": 500.0, "intensity": 1.0}
CLASSES = ("gaussian", "student_t3")
DRAGS = ("linear", "quadratic")
DIRECTIONS = ("local", "front_broadside")
BROADSIDE_ANGLE = -math.pi / 2.0
PILOT_SEEDS = (3001, 3002)
STAT_SEEDS = tuple(range(3003, 3023))
DURATION = 600.0
WARMUP = 20.0
ALPHA = 3.0
SHOCK_AMPLITUDES = (2.0, 4.0, 8.0, 16.0, 32.0)
SHOCK_ANGLES = tuple(np.deg2rad(30.0 * np.arange(12)))
SHOCK_START = 1.0
SHOCK_DURATION = 40.0
WEATHER_SEEDS = tuple(range(9101, 9121))
IMPACT_FACTOR = 0.880
POWER_MIN = 20


def cell_name(distribution: str, drag: str, direction: str) -> str:
    return f"{'G' if distribution == 'gaussian' else 't3'}_{drag}_{direction}"


def cell_specs() -> dict[str, FleetRunSpec]:
    specs = {}
    for distribution in CLASSES:
        for drag in DRAGS:
            for direction in DIRECTIONS:
                specs[cell_name(distribution, drag, direction)] = FleetRunSpec(
                    pretension=REFERENCE["pretension"],
                    heading_gain=REFERENCE["heading_gain"],
                    drag_law=drag,
                    weather_distribution=distribution,
                    weather_direction="front" if direction == "front_broadside" else "local",
                    weather_front_angle=BROADSIDE_ANGLE if direction == "front_broadside" else None,
                    weather_scale=REFERENCE["intensity"],
                    duration=DURATION,
                    warmup=WARMUP,
                )
    return specs


def all_jobs() -> list[StationaryJob]:
    return [
        StationaryJob(name, spec, seed, pilot=seed in PILOT_SEEDS)
        for name, spec in cell_specs().items()
        for seed in PILOT_SEEDS + STAT_SEEDS
    ]


# ----------------------------------------------------------------------------- shock-response cell


@dataclass(frozen=True)
class ShockJob:
    body: int
    angle: float
    amplitude: float
    drag: str = "linear"
    cost: float = SHOCK_DURATION


def innovation_scale(body: int) -> float:
    """Standard deviation of one weather innovation on ``body`` at the reference intensity."""
    phi = ar1_coefficient()
    stationary = constants.LOAD_WEATHER_STD if body == 0 else constants.VESSEL_WEATHER_STD
    return REFERENCE["intensity"] * stationary * math.sqrt(1.0 - phi**2)


def shock_weather(job: ShockJob, duration: float) -> np.ndarray:
    samples = int(math.ceil(duration / constants.WEATHER_PERIOD)) + 2
    weather = np.zeros((samples, constants.VESSEL_COUNT + 1, 2))
    start = int(round(SHOCK_START / constants.WEATHER_PERIOD))
    decay = ar1_coefficient() ** np.arange(samples - start)
    magnitude = job.amplitude * innovation_scale(job.body)
    weather[start:, job.body, 0] = magnitude * math.cos(job.angle) * decay
    weather[start:, job.body, 1] = magnitude * math.sin(job.angle) * decay
    return weather


def run_shock(job: ShockJob) -> dict:
    spec = FleetRunSpec(
        pretension=REFERENCE["pretension"], heading_gain=REFERENCE["heading_gain"], drag_law=job.drag,
        weather_scale=0.0, duration=SHOCK_DURATION, warmup=0.0, log_state=False,
    )
    weather = shock_weather(job, SHOCK_DURATION)
    run = run_to_end(build_run(spec, 1, weather=weather))
    log = run.fleet.cables.log
    count = log.count
    e = log.elongation[:count]
    rate = log.rate[:count]
    q = np.where(e > 0.0, constants.CABLE_STIFFNESS * e + constants.CABLE_DAMPING * rate, 0.0)
    wrel = log.relative_load[:count]
    marks = run.fleet.cables.reengagements
    per_cable = []
    for cable in range(constants.VESSEL_COUNT):
        cable_marks = [m for m in marks if m.cable == cable]
        first = min(cable_marks, key=lambda m: m.t_up) if cable_marks else None
        slack = bool(np.any(e[:, cable] <= 0.0))
        taut_mask = e[:, cable] > 0.0
        per_cable.append({
            "slackened": slack,
            "peak_tension": float(np.max(q[:, cable])),
            "taut_peak_tension": float(np.max(np.where(taut_mask, q[:, cable], 0.0))) if not slack else float(np.max(q[: int(np.argmax(~taut_mask)), cable], initial=0.0)),
            "peak_wrel": float(np.max(wrel[:, cable])),
            "first_return_speed": first.v_return if first else None,
            "first_depth": first.depth if first else None,
            "first_peak": first.T_peak if first else None,
            "max_snap": max((m.T_peak for m in cable_marks), default=0.0),
        })
    return {"job": asdict(job), "closure": run.fleet.cables.closure, "cables": per_cable}


def shock_jobs() -> list[ShockJob]:
    return [
        ShockJob(body, float(angle), amplitude, drag)
        for drag in DRAGS
        for body in range(constants.VESSEL_COUNT + 1)
        for angle in SHOCK_ANGLES
        for amplitude in SHOCK_AMPLITUDES
    ]


def run_shocks(workers: int) -> None:
    jobs = shock_jobs()
    results = run_pool(run_shock, jobs, workers)
    SHOCK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with SHOCK_CACHE.open("wb") as handle:
        pickle.dump(results, handle)
    write_bytes(SHOCK_PATH, json_bytes({"source": source_state(), "amplitudes": SHOCK_AMPLITUDES, "angles_deg": np.degrees(SHOCK_ANGLES).tolist(), "results": results}))


def _log_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return None
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def shock_table(results: list[dict], drag: str = "linear") -> dict:
    """Per (body, angle) direction: responses by amplitude for every cable."""
    table: dict = {}
    for item in results:
        job = item["job"]
        if job["drag"] != drag:
            continue
        key = (job["body"], round(job["angle"], 9))
        table.setdefault(key, {})[job["amplitude"]] = item
    return table


def direction_weights(alpha: float = ALPHA) -> dict[int, float]:
    """Spectral mass per body circle for the local class.

    Independent per-body innovations with scales sigma_b make the stacked vector regularly
    varying with mass proportional to sigma_b^alpha on body b's circle (the plan states
    equal mass; the physical weights are used and the difference is reported).
    """
    scales = np.array([innovation_scale(b) for b in range(constants.VESSEL_COUNT + 1)])
    mass = scales**alpha / np.sum(scales**alpha)
    return {b: float(mass[b]) for b in range(constants.VESSEL_COUNT + 1)}


def t3_radial_survival(r: np.ndarray, rng_seed: int = 7, samples: int = 4_000_000) -> np.ndarray:
    """P(|eps| > r) for one standardized planar t3 innovation (common radial factor)."""
    rng = np.random.default_rng(rng_seed)
    gaussian = rng.normal(size=(samples, 2))
    radial = np.sqrt(1.0 / rng.chisquare(3.0, size=samples))
    magnitude = np.linalg.norm(gaussian, axis=1) * radial
    magnitude.sort()
    return 1.0 - np.searchsorted(magnitude, np.asarray(r, dtype=float), side="right") / samples


def severance_radius(peaks_by_amplitude: dict[float, float], threshold: float) -> float:
    """Smallest amplitude R at which the peak reaches ``threshold``, log-log interpolated;
    beyond the largest amplitude, extrapolated with the last local log-log slope."""
    amplitudes = np.array(sorted(peaks_by_amplitude))
    peaks = np.array([peaks_by_amplitude[a] for a in amplitudes])
    if np.any(peaks >= threshold):
        index = int(np.argmax(peaks >= threshold))
        if index == 0:
            return float(amplitudes[0] * threshold / max(peaks[0], 1e-9))
        x0, x1 = np.log(amplitudes[index - 1]), np.log(amplitudes[index])
        y0, y1 = np.log(max(peaks[index - 1], 1e-9)), np.log(peaks[index])
        return float(np.exp(x0 + (np.log(threshold) - y0) * (x1 - x0) / (y1 - y0)))
    slope = _log_slope(amplitudes[-2:], peaks[-2:])
    if slope is None or slope <= 0:
        return float("inf")
    return float(amplitudes[-1] * (threshold / peaks[-1]) ** (1.0 / slope))


def quadrature_rates(table: dict, thresholds: np.ndarray, weights: dict[int, float]) -> dict[str, np.ndarray]:
    """Theorem 7 single-big-jump rates per unit time for the snap and taut channels.

    Lambda(T_b) = (1/dt) sum_b sum_angles w_b/12 P(|eps| > R_sev(direction, T_b)), with the
    severance radius the minimum over cables; channel rates use the channel's own radius.
    """
    dt = constants.WEATHER_PERIOD
    rates = {"fleet": [], "snap": [], "taut": []}
    for threshold in thresholds:
        radii = {"fleet": [], "snap": [], "taut": []}
        masses = []
        for (body, angle), by_amplitude in table.items():
            snap_radius = min(
                severance_radius({a: r["cables"][c]["max_snap"] for a, r in by_amplitude.items()}, threshold) for c in range(constants.VESSEL_COUNT)
            )
            taut_radius = min(
                severance_radius({a: r["cables"][c]["taut_peak_tension"] for a, r in by_amplitude.items()}, threshold) for c in range(constants.VESSEL_COUNT)
            )
            radii["snap"].append(snap_radius)
            radii["taut"].append(taut_radius)
            radii["fleet"].append(min(snap_radius, taut_radius))
            masses.append(weights[body] / len(SHOCK_ANGLES))
        masses = np.array(masses)
        for key in rates:
            radius = np.array(radii[key])
            finite = np.isfinite(radius)
            survival = np.zeros_like(radius)
            if finite.any():
                survival[finite] = t3_radial_survival(radius[finite])
            rates[key].append(float(np.sum(masses * survival) / dt))
    return {key: np.array(value) for key, value in rates.items()}


def shock_scaling(table: dict) -> list[dict]:
    """P3-T4: slope of log closing speed against log R per slackening (direction, cable)."""
    weights = direction_weights()
    rows = []
    for (body, angle), by_amplitude in table.items():
        for cable in range(constants.VESSEL_COUNT):
            points = {a: by_amplitude[a]["cables"][cable]["first_return_speed"] for a in (4.0, 8.0, 16.0, 32.0) if a in by_amplitude}
            if any(v is None or v <= 0 for v in points.values()) or len(points) < 4:
                continue
            amplitudes = np.array(sorted(points))
            speeds = np.array([points[a] for a in amplitudes])
            rows.append({
                "body": body, "angle_deg": math.degrees(angle), "cable": cable,
                "beta": _log_slope(amplitudes, speeds),
                "beta_8_16": _log_slope(amplitudes[1:3], speeds[1:3]),
                "beta_16_32": _log_slope(amplitudes[2:4], speeds[2:4]),
                "mass": weights[body] / len(SHOCK_ANGLES),
            })
    return rows


# ----------------------------------------------------------------------------- predictions


def predict() -> dict:
    with SHOCK_CACHE.open("rb") as handle:
        shocks = pickle.load(handle)
    thresholds = np.geomspace(2000.0, 60000.0, 60)
    weights = direction_weights()
    predictions = {"schema_version": 1, "source": source_state(), "alpha": ALPHA, "weights": weights,
                   "plan_equal_mass_weights": {b: 1.0 / (constants.VESSEL_COUNT + 1) for b in range(constants.VESSEL_COUNT + 1)},
                   "thresholds": thresholds.tolist(), "drags": {}}
    for drag in DRAGS:
        table = shock_table(shocks, drag)
        rates = quadrature_rates(table, thresholds, weights)
        snap = rates["snap"]
        taut = rates["taut"]
        crossover = None
        diff = np.log(np.maximum(snap, 1e-300)) - np.log(np.maximum(taut, 1e-300))
        valid = (snap > 0) & (taut > 0)
        for i in range(len(thresholds) - 1):
            if valid[i] and valid[i + 1] and diff[i] > 0 >= diff[i + 1]:
                t = diff[i] / (diff[i] - diff[i + 1])
                crossover = float(np.exp(np.log(thresholds[i]) + t * (np.log(thresholds[i + 1]) - np.log(thresholds[i]))))
                break
        predictions["drags"][drag] = {
            "fleet_rate": rates["fleet"].tolist(), "snap_rate": snap.tolist(), "taut_rate": taut.tolist(),
            "predicted_crossover": crossover, "shock_scaling": shock_scaling(table),
        }
    predictions["index_predictions"] = {"taut_alpha": ALPHA, "snap_two_alpha": 2 * ALPHA, "taut_gpd_shape": 1.0 / ALPHA}
    write_bytes(PREDICTIONS_PATH, json_bytes(predictions))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("shocks", "predict", "compute", "analyse", "replay"))
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    if arguments.command == "shocks":
        run_shocks(arguments.workers)
    elif arguments.command == "predict":
        predictions = predict()
        print({d: predictions["drags"][d]["predicted_crossover"] for d in DRAGS})
    elif arguments.command == "compute":
        if not PREDICTIONS_PATH.exists():
            raise SystemExit("predictions must be committed before the campaign runs")
        compute_jobs(all_jobs(), CACHE_PATH, arguments.workers, "phase3")
    elif arguments.command == "analyse":
        from tether.campaign.phase3_analysis import analyse

        results = analyse()
        for test_id, test in results["tests"].items():
            print(test_id, test["verdict"])
    else:
        from tether.campaign.phase3_analysis import replay

        sys.exit(0 if replay() else 1)


if __name__ == "__main__":
    main()
