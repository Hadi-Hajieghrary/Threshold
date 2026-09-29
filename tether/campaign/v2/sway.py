"""The sway loop and its declared design rule (plan v2 IV.7; II.2 (H1); Appendix D).

Stages, in the plan's order (every choice is fixed in
``records/v2/phase0/sway_declarations.json``, written before any result):

``model``   the reduced model's commitments, written before any plant run: the settling
            prediction of the chord-angle step, the k_sigma selection table (ii), the psi
            and chord-angle stds (iii), the k_h = 500 stability check, and mu_q2;
``steps``   (i) the deterministic chord-angle step on the plant, zero weather, perturbed
            and reference runs on common random numbers;
``pilot``   (iv) 5 seeds x 300 s after a 20 s warm-up at T0 = 1 kN, k_h = 763, Gaussian
            local weather at intensities 0.5 and 0.35, every pilot k_sigma;
``analyse`` validation verdicts, the rule's selection, the branch gain, the psi branch.

Run: ``python -m tether.campaign.v2.sway all --workers 4``.

Addendum 1 (saturated correction, owner ruling; Phase 1(e) and P1-T11):
``addendum-declare`` writes records/v2/phase0/sway_addendum_1.json and
records/v2/phase1/p1e_declarations.json (never rewritten); ``revalidate`` re-runs step (i)
at k_sigma in {0, 3} and the no-parking test; ``p1e`` runs the pilot; ``p1e-analyse``
gives the P1-T11 verdict.  ``python -m tether.campaign.v2.sway addendum-all --workers 8``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from tether.campaign.common import (
    RECORDS,
    json_bytes,
    npz_bytes,
    relative,
    run_pool,
    sha256_bytes,
    sha256_file,
    source_state,
    write_bytes,
)
from tether.physics import constants

PHASE_DIR = RECORDS / "v2" / "phase0"
DECLARATIONS_PATH = PHASE_DIR / "sway_declarations.json"
MODEL_PATH = PHASE_DIR / "sway_model.json"
STEPS_PATH = PHASE_DIR / "sway_steps.npz"
STEPS_SUMMARY_PATH = PHASE_DIR / "sway_steps.json"
PILOT_PATH = PHASE_DIR / "sway_pilot_runs.json"
RESULTS_PATH = PHASE_DIR / "sway_results.json"

PRETENSION = 1000.0
TRIM_GAIN = 100.0
DESIGN_HEADING_GAIN = 1000.0
PILOT_HEADING_GAIN = 763.0
STEP_HEADING_GAINS = (1000.0, 763.0)  # the first gates, the second is secondary
STEP_K_SIGMA = (0.0, 1.0, 2.0, 3.0, 5.0)
CANDIDATES = (1.0, 2.0, 3.0, 5.0)
STEP_SEED = 7000
STEP_VESSEL = 2
STEP_DEG = 2.0
SETTLING_FRACTION = 0.10
TOLERANCE = 0.30
STATE_PERIOD = 0.01
MODEL_HORIZON_MAX = 20_000.0
PLANT_HORIZON_MIN = 60.0
PLANT_HORIZON_MAX = 3600.0
PLANT_HORIZON_PAD = 30.0
DESIGN_INTENSITY = 0.5
CHORD_LIMIT_DEG = 10.0
SHAPE_LIMIT_DEG = 15.0
TENSION_FRACTION = 0.1
REPORT_INTENSITIES = (0.35, 0.5, 1.0)
STABILITY_HEADING_GAIN = 500.0
STABILITY_K_SIGMA = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0)
BISECTION_TOLERANCE = 1.0e-4
PILOT_INTENSITIES = (0.5, 0.35)
PILOT_K_SIGMA = (0.0, 1.0, 2.0, 3.0, 5.0)
PILOT_SEEDS = (7001, 7002, 7003, 7004, 7005)
PILOT_WARMUP = 20.0
PILOT_DURATION = 300.0
PILOT_DECIMATION = 0.1
FALLBACK_K_SIGMA = 5.0
BRANCH_INTENSITY = 0.5
FALLBACK_INTENSITY = 0.35


def _key(value: float) -> str:
    return f"{value:g}"


# --------------------------------------------------------------------------- geometry


def chord_and_misalignment(states: np.ndarray, geometry) -> tuple[np.ndarray, np.ndarray]:
    """World chord angle sigma_i and psi_i = wrap(theta_i - sigma_i) for rows of ``[q; v]``.

    The chord runs from the load attachment to the stern attachment, as in
    ``fleet.cable_kinematics``.
    """
    states = np.atleast_2d(states)
    n = geometry.vessel_count
    positions = states[:, : 3 * (n + 1)].reshape(-1, n + 1, 3)
    load_heading = positions[:, 0, 2][:, None]
    headings = positions[:, 1:, 2]
    ox, oy = geometry.load_offsets[:, 0][None, :], geometry.load_offsets[:, 1][None, :]
    sx, sy = geometry.vessel_offsets[:, 0][None, :], geometry.vessel_offsets[:, 1][None, :]
    load_x = positions[:, 0, 0][:, None] + np.cos(load_heading) * ox - np.sin(load_heading) * oy
    load_y = positions[:, 0, 1][:, None] + np.sin(load_heading) * ox + np.cos(load_heading) * oy
    stern_x = positions[:, 1:, 0] + np.cos(headings) * sx - np.sin(headings) * sy
    stern_y = positions[:, 1:, 1] + np.sin(headings) * sx + np.cos(headings) * sy
    sigma = np.arctan2(stern_y - load_y, stern_x - load_x)
    psi = np.angle(np.exp(1j * (headings - sigma)))
    return sigma, psi


def rotated_start(geometry, operating, cable, vessel: int = STEP_VESSEL, angle_deg: float = STEP_DEG) -> np.ndarray:
    """Equilibrium with ``vessel`` rotated rigidly about its load attachment point.

    Chord length and hull-chord misalignment are unchanged; velocities stay at the
    steady tow.  The load sits at the origin with zero heading in the equilibrium.
    """
    from tether.physics.fleet import equilibrium_state

    state = equilibrium_state(geometry, operating, cable).copy()
    angle = math.radians(angle_deg)
    pivot = geometry.load_offsets[vessel]
    base = 3 * (vessel + 1)
    offset = state[base : base + 2] - pivot
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    state[base : base + 2] = pivot + rotation @ offset
    state[base + 2] += angle
    return state


# --------------------------------------------------------------------------- model


def model_input(heading_gain: float, intensity: float, k_sigma: float):
    from tether.theory.reduced_lti import ReducedModelInput

    return ReducedModelInput(
        formation="parallel",
        pretension=PRETENSION,
        heading_gain=heading_gain,
        trim_gain=TRIM_GAIN,
        drag_law="linear",
        weather_direction="local",
        weather_scale=intensity,
        sway_gain=k_sigma,
    )


def model_step_prediction(heading_gain: float, k_sigma: float) -> dict:
    """Settling of cable STEP_VESSEL's chord angle after the declared step, linear model."""
    from tether.physics.fleet import CableParameters, formation_geometry, operating_point
    from tether.theory.reduced_lti import reduced_model, to_reduced

    result = reduced_model(model_input(heading_gain, 1.0, k_sigma))
    n = result.geometry.vessel_count
    size = result.state_matrix.shape[0]
    geometry = formation_geometry("parallel")
    start = rotated_start(geometry, operating_point(geometry, PRETENSION), CableParameters())
    x0 = to_reduced(start, n) - result.equilibrium
    output = result.outputs["chord_angle"][STEP_VESSEL, :size]
    band = SETTLING_FRACTION * math.radians(STEP_DEG)
    eigenvalues, vectors = np.linalg.eig(result.state_matrix)
    residues = (output @ vectors) * np.linalg.solve(vectors, x0)
    slowest = float(np.max(eigenvalues.real))
    observed = np.abs(residues) > 1.0e-9 * band
    if np.any(observed & (eigenvalues.real >= -1.0e-9)):
        return {"settling_s": math.inf, "reason": "a non-decaying mode is observed", "slowest_eigenvalue_real": slowest,
                "initial_deg": math.degrees(float(output @ x0))}
    # Modal bound: once it drops below half the band the output stays inside for good.
    grid = np.arange(0.0, MODEL_HORIZON_MAX + STATE_PERIOD, 1.0)
    bound = np.abs(residues)[None, :] * np.exp(np.outer(grid, eigenvalues.real))
    inside = np.flatnonzero(bound.sum(axis=1) < 0.5 * band)
    horizon = float(grid[inside[0]]) if inside.size else MODEL_HORIZON_MAX
    steps = int(round(horizon / STATE_PERIOD))
    transition = expm(result.state_matrix * STATE_PERIOD)
    values = np.empty(steps + 1)
    x = x0.copy()
    for index in range(steps + 1):
        values[index] = output @ x
        x = transition @ x
    above = np.flatnonzero(np.abs(values) > band)
    settling = float(above[-1] * STATE_PERIOD) if above.size else 0.0
    if above.size and above[-1] == steps:
        settling = math.inf
    peak = int(np.argmax(np.abs(values)))
    return {
        "settling_s": settling,
        "model_horizon_s": horizon,
        "initial_deg": math.degrees(float(values[0])),
        "peak_abs_deg": math.degrees(float(abs(values[peak]))),
        "peak_time_s": peak * STATE_PERIOD,
        "slowest_eigenvalue_real": slowest,
        "trace_1s_deg": np.degrees(values[:: int(round(1.0 / STATE_PERIOD))]).tolist(),
    }


def plant_horizon(model_settling: float) -> float:
    if not math.isfinite(model_settling):
        return PLANT_HORIZON_MAX
    return float(min(max(PLANT_HORIZON_MIN, math.ceil(1.5 * model_settling) + PLANT_HORIZON_PAD), PLANT_HORIZON_MAX))


def model_statistics(heading_gain: float, intensity: float, k_sigma: float) -> dict:
    from tether.theory.reduced_lti import _drag_force

    result = reduced_model_cached(heading_gain, intensity, k_sigma)
    chord = result.sigma_chord_angle
    psi = result.sigma_psi
    tension = result.mu_q
    drag = _drag_force(result.operating.speed, constants.VESSEL_LINEAR_DRAG, "linear")
    thrust = result.operating.thrusts
    variance = (tension + drag) ** 2 * chord**4 / 2.0 + thrust**2 * psi**4 / 2.0
    ratio = variance / (TENSION_FRACTION * result.sigma_q) ** 2
    return {
        "chord_std_deg": np.degrees(chord),
        "psi_std_deg": np.degrees(psi),
        "sigma_q_N": result.sigma_q,
        "second_order_variance_N2": variance,
        "second_order_std_N": np.sqrt(variance),
        "sigma_term_std_N": np.sqrt((tension + drag) ** 2 * chord**4 / 2.0),
        "psi_term_std_N": np.sqrt(thrust**2 * psi**4 / 2.0),
        "variance_ratio": ratio,
        "mu_q_N": tension,
        "mu_q2_N": result.mu_q2,
        "mu_q2_sigma_term_N": result.mu_q2_terms["sigma"],
        "mu_q2_psi_term_N": result.mu_q2_terms["psi"],
        "T0_plus_cAv_N": tension + drag,
        "F_T_N": thrust,
        "unstable_eigenvalues": [[float(v.real), float(v.imag)] for v in result.unstable_eigenvalues],
        "max_real_eigenvalue": float(np.max(np.linalg.eigvals(result.state_matrix).real)),
        "chord_max_deg": float(np.max(np.degrees(chord))),
        "psi_max_deg": float(np.max(np.degrees(psi))),
        "psi_mean_deg": float(np.mean(np.degrees(psi))),
        "chord_mean_deg": float(np.mean(np.degrees(chord))),
        "variance_ratio_max": float(np.max(ratio)),
    }


_MODEL_CACHE: dict = {}


def reduced_model_cached(heading_gain: float, intensity: float, k_sigma: float):
    from tether.theory.reduced_lti import reduced_model

    key = (heading_gain, intensity, k_sigma)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = reduced_model(model_input(heading_gain, intensity, k_sigma))
    return _MODEL_CACHE[key]


def max_real_eigenvalue(heading_gain: float, k_sigma: float) -> float:
    result = reduced_model_cached(heading_gain, 1.0, k_sigma)
    return float(np.max(np.linalg.eigvals(result.state_matrix).real))


def stability_check() -> dict:
    values = {_key(k): max_real_eigenvalue(STABILITY_HEADING_GAIN, k) for k in STABILITY_K_SIGMA}
    unstable_without = values["0"] > 0.0
    stable_from_half = all(values[_key(k)] < 0.0 for k in STABILITY_K_SIGMA if k >= 0.5)
    boundary = None
    if unstable_without and values["0.5"] < 0.0:
        low, high = 0.0, 0.5
        while high - low > BISECTION_TOLERANCE:
            middle = 0.5 * (low + high)
            if max_real_eigenvalue(STABILITY_HEADING_GAIN, middle) < 0.0:
                high = middle
            else:
                low = middle
        boundary = [low, high]
    return {
        "heading_gain": STABILITY_HEADING_GAIN,
        "pretension_N": PRETENSION,
        "max_real_eigenvalue": values,
        "plan_claim_k0": 8.7e-4,
        "unstable_without_loop": unstable_without,
        "stable_for_every_k_sigma_ge_0_5": stable_from_half,
        "plan_claim_holds": bool(unstable_without and stable_from_half),
        "k_sigma_boundary_bracket": boundary,
    }


def selection_table(stats: dict) -> dict:
    rows = {}
    for k in CANDIDATES:
        row = stats[_key(DESIGN_HEADING_GAIN)][_key(DESIGN_INTENSITY)][_key(k)]
        clause_a = row["chord_max_deg"] <= CHORD_LIMIT_DEG
        clause_b = row["variance_ratio_max"] < 1.0
        rows[_key(k)] = {"clause_A_chord": bool(clause_a), "clause_B_variance": bool(clause_b), "both": bool(clause_a and clause_b),
                         "chord_max_deg": row["chord_max_deg"], "variance_ratio_max": row["variance_ratio_max"],
                         "psi_max_deg": row["psi_max_deg"]}
    meeting = [k for k in CANDIDATES if rows[_key(k)]["both"]]
    chord_only = [k for k in CANDIDATES if rows[_key(k)]["clause_A_chord"]]
    return {"rows": rows, "model_selection": meeting[0] if meeting else None,
            "smallest_meeting_clause_A_only": chord_only[0] if chord_only else None}


def compute_model() -> dict:
    declarations_sha = _require_declarations()
    predictions = {}
    for heading_gain in STEP_HEADING_GAINS:
        predictions[_key(heading_gain)] = {}
        for k in STEP_K_SIGMA:
            prediction = model_step_prediction(heading_gain, k)
            prediction["plant_horizon_s"] = plant_horizon(prediction["settling_s"])
            predictions[_key(heading_gain)][_key(k)] = prediction
    stats = {}
    for heading_gain in (DESIGN_HEADING_GAIN, PILOT_HEADING_GAIN):
        stats[_key(heading_gain)] = {}
        for intensity in REPORT_INTENSITIES:
            stats[_key(heading_gain)][_key(intensity)] = {_key(k): model_statistics(heading_gain, intensity, k) for k in STEP_K_SIGMA}
    model = {
        "declarations_sha256": declarations_sha,
        "source": source_state(),
        "step_predictions": predictions,
        "statistics": stats,
        "selection": selection_table(stats),
        "stability_check": stability_check(),
        "note": "written before any plant run of this test; the plant horizons of the step validation are read from here",
    }
    payload = json_bytes(model)
    if MODEL_PATH.exists() and MODEL_PATH.read_bytes() != payload:
        raise SystemExit(f"{relative(MODEL_PATH)} exists with different content; committed predictions are never rewritten")
    write_bytes(MODEL_PATH, payload)
    return model


# --------------------------------------------------------------------------- plant


@dataclass(frozen=True)
class StepJob:
    heading_gain: float
    k_sigma: float
    perturbed: bool
    horizon: float
    seed: int = STEP_SEED
    # Addendum 1: the clamp on the sway correction (None = the unsaturated law) and the
    # initial rigid rotation of the step vessel (the no-parking test's large offsets).
    sway_limit: float | None = None
    angle_deg: float = STEP_DEG

    @property
    def cost(self) -> float:
        return self.horizon


def run_step_job(job: StepJob) -> dict:
    from tether.campaign.fleet_run import FleetRunSpec, build_run, run_to_end
    from tether.physics.fleet import CableParameters, formation_geometry, operating_point

    spec = FleetRunSpec(formation="parallel", pretension=PRETENSION, heading_gain=job.heading_gain, trim_gain=TRIM_GAIN,
                        weather_scale=0.0, duration=job.horizon, warmup=0.0, k_sigma=job.k_sigma, sway_limit=job.sway_limit)
    geometry = formation_geometry("parallel")
    start = (rotated_start(geometry, operating_point(geometry, PRETENSION), CableParameters(), angle_deg=job.angle_deg)
             if job.perturbed else None)
    run = run_to_end(build_run(spec, job.seed, initial_state=start))
    log = run.fleet.cables.log
    times = log.state_time[: log.state_count].copy()
    sigma, psi = chord_and_misalignment(log.state[: log.state_count], geometry)
    closure = run.fleet.cables.closure
    return {"time": times, "sigma": sigma, "psi": psi, "closure": None if closure is None else [int(closure[0]), float(closure[1])],
            "load_heading": log.state[: log.state_count, 2].copy(), "wall_seconds": run.wall_seconds}


def settling_time(times: np.ndarray, response: np.ndarray, band: float) -> tuple[float, bool]:
    above = np.flatnonzero(np.abs(response) > band)
    if above.size == 0:
        return 0.0, False
    if above[-1] == response.size - 1:
        return float(times[-1]), True
    return float(times[above[-1]]), False


def compute_steps(workers: int) -> dict:
    declarations_sha = _require_declarations()
    model = json.loads(MODEL_PATH.read_text())
    jobs = []
    for heading_gain in STEP_HEADING_GAINS:
        for k in STEP_K_SIGMA:
            horizon = model["step_predictions"][_key(heading_gain)][_key(k)]["plant_horizon_s"]
            for perturbed in (True, False):
                jobs.append(StepJob(heading_gain, k, perturbed, horizon))
    results = run_pool(run_step_job, jobs, workers)
    band = SETTLING_FRACTION * math.radians(STEP_DEG)
    arrays, summary = {}, {"declarations_sha256": declarations_sha, "model_sha256": sha256_file(MODEL_PATH), "cases": {}}
    for heading_gain in STEP_HEADING_GAINS:
        for k in STEP_K_SIGMA:
            pair = {job.perturbed: result for job, result in zip(jobs, results) if job.heading_gain == heading_gain and job.k_sigma == k}
            perturbed, reference = pair[True], pair[False]
            count = min(perturbed["time"].size, reference["time"].size)
            times = perturbed["time"][:count]
            response = perturbed["sigma"][:count, STEP_VESSEL] - reference["sigma"][:count, STEP_VESSEL]
            settling, censored = settling_time(times, response, band)
            name = f"kh{_key(heading_gain)}_ks{_key(k)}"
            arrays[f"{name}_time"] = times
            arrays[f"{name}_dsigma"] = response
            arrays[f"{name}_reference_sigma"] = reference["sigma"][:count]
            arrays[f"{name}_perturbed_psi"] = perturbed["psi"][:count]
            peak = int(np.argmax(np.abs(response)))
            quiet = reference["sigma"][:count]
            summary["cases"][name] = {
                "heading_gain": heading_gain, "k_sigma": k, "horizon_s": float(times[-1]) if count else 0.0,
                "settling_s": settling, "censored": censored,
                "peak_abs_deg": math.degrees(float(abs(response[peak]))), "peak_time_s": float(times[peak]),
                "final_abs_deg": math.degrees(float(abs(response[-1]))),
                "reference_run_chord_std_deg": np.degrees(np.std(quiet, axis=0)).tolist(),
                "closure_perturbed": perturbed["closure"], "closure_reference": reference["closure"],
                "wall_seconds": perturbed["wall_seconds"] + reference["wall_seconds"],
            }
    write_bytes(STEPS_PATH, npz_bytes(arrays))
    summary["steps_npz_sha256"] = sha256_file(STEPS_PATH)
    write_bytes(STEPS_SUMMARY_PATH, json_bytes(summary))
    return summary


@dataclass(frozen=True)
class PilotJob:
    k_sigma: float
    intensity: float
    seed: int

    @property
    def cost(self) -> float:
        return PILOT_WARMUP + PILOT_DURATION


def run_pilot_job(job: PilotJob) -> dict:
    from tether.campaign.fleet_run import FleetRunSpec, build_run, run_to_end
    from tether.campaign.summaries import summarize_run

    spec = FleetRunSpec(formation="parallel", pretension=PRETENSION, heading_gain=PILOT_HEADING_GAIN, trim_gain=TRIM_GAIN,
                        drag_law="linear", weather_distribution="gaussian", weather_direction="local", weather_scale=job.intensity,
                        duration=PILOT_DURATION, warmup=PILOT_WARMUP, cable_mode="recording", k_sigma=job.k_sigma)
    run = run_to_end(build_run(spec, job.seed))
    summary = summarize_run(run, PILOT_WARMUP, PRETENSION)
    log = run.fleet.cables.log
    times = log.state_time[: log.state_count]
    end = float(summary["meta"]["end_time"])
    window = np.flatnonzero((times >= PILOT_WARMUP - 1.0e-9) & (times <= end + 1.0e-9))
    picked = window[:: int(round(PILOT_DECIMATION / STATE_PERIOD))]
    sigma, psi = chord_and_misalignment(log.state[picked], run.fleet.geometry)
    q_moments = summary["moments"]["q"]
    return {
        "k_sigma": job.k_sigma, "intensity": job.intensity, "seed": job.seed,
        "samples": int(picked.size),
        "sigma_cos_sum": np.cos(sigma).sum(axis=0), "sigma_sin_sum": np.sin(sigma).sum(axis=0),
        "psi_cos_sum": np.cos(psi).sum(axis=0), "psi_sin_sum": np.sin(psi).sum(axis=0),
        "chord_std_deg": np.degrees(_circular_std_from_sums(np.cos(sigma).sum(axis=0), np.sin(sigma).sum(axis=0), picked.size))
        if picked.size else np.full(sigma.shape[1], np.nan),
        "psi_std_deg": np.degrees(_circular_std_from_sums(np.cos(psi).sum(axis=0), np.sin(psi).sum(axis=0), picked.size))
        if picked.size else np.full(psi.shape[1], np.nan),
        "geometry_stats_chord_std_deg": np.degrees(summary["geometry_stats"]["cable_angle_std"]),
        "closure": summary["meta"]["closure"],
        "end_time": end,
        "exposure_s": float(summary["meta"]["exposure"]),
        "onsets": summary["onsets"],
        "marks": int(summary["marks"]["t_up"].size),
        "clean_q_count": q_moments[0], "clean_q_sum": q_moments[1], "clean_q_square": q_moments[2],
        "wall_seconds": run.wall_seconds,
    }


def _circular_std_from_sums(cos_sum, sin_sum, count) -> np.ndarray:
    resultant = np.hypot(np.asarray(cos_sum, dtype=float), np.asarray(sin_sum, dtype=float)) / max(float(count), 1.0)
    return np.sqrt(-2.0 * np.log(np.clip(resultant, 1.0e-12, 1.0)))


def compute_pilot(workers: int) -> list[dict]:
    declarations_sha = _require_declarations()
    jobs = [PilotJob(k, intensity, seed) for intensity in PILOT_INTENSITIES for k in PILOT_K_SIGMA for seed in PILOT_SEEDS]
    runs = run_pool(run_pilot_job, jobs, workers)
    write_bytes(PILOT_PATH, json_bytes({"declarations_sha256": declarations_sha, "runs": runs}))
    return runs


# --------------------------------------------------------------------------- analysis


def pilot_cells(runs: list[dict]) -> dict:
    cells = {}
    for intensity in PILOT_INTENSITIES:
        cells[_key(intensity)] = {}
        for k in PILOT_K_SIGMA:
            chosen = [r for r in runs if r["intensity"] == intensity and r["k_sigma"] == k]
            count = sum(r["samples"] for r in chosen)
            chord = _circular_std_from_sums(sum(np.asarray(r["sigma_cos_sum"]) for r in chosen),
                                            sum(np.asarray(r["sigma_sin_sum"]) for r in chosen), count)
            psi = _circular_std_from_sums(sum(np.asarray(r["psi_cos_sum"]) for r in chosen),
                                          sum(np.asarray(r["psi_sin_sum"]) for r in chosen), count)
            q_count = sum(np.asarray(r["clean_q_count"]) for r in chosen)
            q_mean = sum(np.asarray(r["clean_q_sum"]) for r in chosen) / np.maximum(q_count, 1.0)
            closures = [[r["seed"], r["closure"]] for r in chosen if r["closure"] is not None]
            cells[_key(intensity)][_key(k)] = {
                "chord_std_deg_per_vessel": np.degrees(chord),
                "psi_std_deg_per_vessel": np.degrees(psi),
                "chord_stat_deg": float(np.degrees(np.max(chord))),
                "psi_stat_deg": float(np.degrees(np.max(psi))),
                "chord_mean_over_vessels_deg": float(np.degrees(np.mean(chord))),
                "psi_mean_over_vessels_deg": float(np.degrees(np.mean(psi))),
                # A run that closes inside the warm-up has no recorded window (None).
                "per_seed_chord_max_deg": [float(np.max(r["chord_std_deg"])) if r["samples"] else None for r in chosen],
                "per_seed_psi_max_deg": [float(np.max(r["psi_std_deg"])) if r["samples"] else None for r in chosen],
                "per_seed_geometry_stats_chord_max_deg": [float(np.nanmax(r["geometry_stats_chord_std_deg"])) if r["samples"] else None
                                                          for r in chosen],
                "per_seed_samples": [r["samples"] for r in chosen],
                "per_seed_exposure_s": [r["exposure_s"] for r in chosen],
                "runs_with_no_recorded_window": sum(1 for r in chosen if not r["samples"]),
                "chord_meets_15": bool(np.degrees(np.max(chord)) <= SHAPE_LIMIT_DEG),
                "psi_meets_15": bool(np.degrees(np.max(psi)) <= SHAPE_LIMIT_DEG),
                "closures": len(closures), "closure_runs": closures,
                "exposure_s": float(sum(r["exposure_s"] for r in chosen)),
                "onsets": int(sum(int(np.sum(r["onsets"])) for r in chosen)),
                "marks": int(sum(r["marks"] for r in chosen)),
                "clean_mean_q_N_per_cable": q_mean,
                "seeds": [r["seed"] for r in chosen],
            }
    return cells


def analyse() -> dict:
    declarations_sha = _require_declarations()
    model = json.loads(MODEL_PATH.read_text())
    steps = json.loads(STEPS_SUMMARY_PATH.read_text())
    pilot = json.loads(PILOT_PATH.read_text())
    validation = {}
    for heading_gain in STEP_HEADING_GAINS:
        for k in STEP_K_SIGMA:
            name = f"kh{_key(heading_gain)}_ks{_key(k)}"
            case = steps["cases"][name]
            predicted = model["step_predictions"][_key(heading_gain)][_key(k)]["settling_s"]
            measured = case["settling_s"]
            testable = math.isfinite(predicted) and predicted > 0.0 and not case["censored"]
            error = (measured - predicted) / predicted if testable else None
            validation[name] = {
                "heading_gain": heading_gain, "k_sigma": k, "gating": heading_gain == DESIGN_HEADING_GAIN,
                "model_settling_s": predicted, "plant_settling_s": measured, "plant_censored": case["censored"],
                "relative_error": error, "pass": bool(testable and abs(error) <= TOLERANCE),
                "plant_peak_abs_deg": case["peak_abs_deg"], "plant_peak_time_s": case["peak_time_s"],
                "model_peak_abs_deg": model["step_predictions"][_key(heading_gain)][_key(k)].get("peak_abs_deg"),
                "model_peak_time_s": model["step_predictions"][_key(heading_gain)][_key(k)].get("peak_time_s"),
                "model_slowest_eigenvalue_real": model["step_predictions"][_key(heading_gain)][_key(k)]["slowest_eigenvalue_real"],
            }
    gate0 = validation[f"kh{_key(DESIGN_HEADING_GAIN)}_ks0"]["pass"]
    selection = model["selection"]
    candidate = selection["model_selection"]
    if not gate0:
        rule, reason = None, "step validation (i) failed at k_sigma = 0: the rule cannot select k_sigma from the model"
    elif candidate is None:
        rule, reason = None, "no candidate meets both clauses of (ii) on the validated model"
    elif not validation[f"kh{_key(DESIGN_HEADING_GAIN)}_ks{_key(candidate)}"]["pass"]:
        rule, reason = None, f"the model selects k_sigma = {candidate:g} but step validation (i) fails at that gain"
    else:
        rule, reason = candidate, "selected by the declared rule on the validated model"
    branch_gain = rule if rule is not None else FALLBACK_K_SIGMA
    cells = pilot_cells(pilot["runs"])
    at_branch = cells[_key(BRANCH_INTENSITY)][_key(branch_gain)]
    fires = at_branch["psi_stat_deg"] > SHAPE_LIMIT_DEG
    branch = {
        "gain_read": branch_gain,
        "gain_label": "RULE SELECTION" if rule is not None else "FALLBACK (plan prior k_sigma = 5, not a rule selection)",
        "psi_stat_deg_at_0.5": at_branch["psi_stat_deg"],
        "fires": bool(fires),
        "outcome": ("psi std > 15 deg at intensity 0.5: the Phase 2 radial-law cells drop to intensity 0.35 and Part III's "
                    "committed table is re-committed at 0.35 before any statistics seed" if fires else
                    "psi std <= 15 deg at intensity 0.5: the graded grid keeps its intensity-0.5 cells"),
        "shape_at_branch_gain": {_key(i): {"chord_stat_deg": cells[_key(i)][_key(branch_gain)]["chord_stat_deg"],
                                           "psi_stat_deg": cells[_key(i)][_key(branch_gain)]["psi_stat_deg"],
                                           "chord_meets_15": cells[_key(i)][_key(branch_gain)]["chord_meets_15"],
                                           "psi_meets_15": cells[_key(i)][_key(branch_gain)]["psi_meets_15"],
                                           "closures": cells[_key(i)][_key(branch_gain)]["closures"]}
                                 for i in PILOT_INTENSITIES},
        "fires_at_every_pilot_gain_for_transparency": {_key(k): bool(cells[_key(BRANCH_INTENSITY)][_key(k)]["psi_stat_deg"] > SHAPE_LIMIT_DEG)
                                                       for k in PILOT_K_SIGMA},
    }
    predicted_pilot = {_key(i): {_key(k): {"chord_max_deg": model["statistics"][_key(PILOT_HEADING_GAIN)][_key(i)][_key(k)]["chord_max_deg"],
                                           "psi_max_deg": model["statistics"][_key(PILOT_HEADING_GAIN)][_key(i)][_key(k)]["psi_max_deg"],
                                           "mu_q2_N": model["statistics"][_key(PILOT_HEADING_GAIN)][_key(i)][_key(k)]["mu_q2_N"]}
                                 for k in PILOT_K_SIGMA} for i in PILOT_INTENSITIES}
    results = {
        "declarations_sha256": declarations_sha,
        "model_sha256": sha256_file(MODEL_PATH),
        "steps_summary_sha256": sha256_file(STEPS_SUMMARY_PATH),
        "steps_npz_sha256": sha256_file(STEPS_PATH),
        "pilot_runs_sha256": sha256_file(PILOT_PATH),
        "source": source_state(),
        "step_validation_i": {"cases": validation, "gate_k_sigma_0": bool(gate0),
                              "gate_selected": None if candidate is None else validation[f"kh{_key(DESIGN_HEADING_GAIN)}_ks{_key(candidate)}"]["pass"]},
        "selection_ii": {"table": selection["rows"], "model_selection": candidate,
                         "smallest_meeting_clause_A_only": selection["smallest_meeting_clause_A_only"],
                         "rule_selection": rule, "reason": reason},
        "model_reports_iii": {
            "psi_and_chord_std": {hg: {i: {k: {"chord_max_deg": v["chord_max_deg"], "chord_mean_deg": v["chord_mean_deg"],
                                                "psi_max_deg": v["psi_max_deg"], "psi_mean_deg": v["psi_mean_deg"],
                                                "sigma_q_N": v["sigma_q_N"], "mu_q2_N": v["mu_q2_N"],
                                                "mu_q2_sigma_term_N": v["mu_q2_sigma_term_N"], "mu_q2_psi_term_N": v["mu_q2_psi_term_N"],
                                                "max_real_eigenvalue": v["max_real_eigenvalue"]}
                                           for k, v in row.items()} for i, row in table.items()}
                                  for hg, table in model["statistics"].items()},
            "stability_check": model["stability_check"],
        },
        "pilot_iv": {"cells": cells, "model_predictions_kh763": predicted_pilot,
                     "plant_over_model_non_gating": {
                         _key(i): {_key(k): {"chord": cells[_key(i)][_key(k)]["chord_stat_deg"] / predicted_pilot[_key(i)][_key(k)]["chord_max_deg"],
                                             "psi": cells[_key(i)][_key(k)]["psi_stat_deg"] / predicted_pilot[_key(i)][_key(k)]["psi_max_deg"]}
                                   for k in PILOT_K_SIGMA} for i in PILOT_INTENSITIES}},
        "branch": branch,
    }
    write_bytes(RESULTS_PATH, json_bytes(results))
    return results


def _require_declarations() -> str:
    if not DECLARATIONS_PATH.exists():
        raise SystemExit("declarations must be written before any result")
    return sha256_bytes(DECLARATIONS_PATH.read_bytes())


# =========================================================================== addendum 1
# Saturated sway correction (owner ruling), the clause-A gain fallback, re-validation of
# the saturated law (step (i) and the no-parking test), and Phase 1(e), the shape-condition
# pilot with its P1-T11 verdict.  Declarations: records/v2/phase0/sway_addendum_1.json and
# records/v2/phase1/p1e_declarations.json, both written before any run below.

ADDENDUM_PATH = PHASE_DIR / "sway_addendum_1.json"
ADDENDUM_RESULTS_PATH = PHASE_DIR / "sway_addendum_1_results.json"
P1_DIR = RECORDS / "v2" / "phase1"
P1E_DECLARATIONS_PATH = P1_DIR / "p1e_declarations.json"
P1E_RUNS_PATH = P1_DIR / "p1e_runs.json"
P1E_RESULTS_PATH = P1_DIR / "p1e_results.json"

SWAY_LIMIT_V2 = 0.349  # rad, 20.0 deg
SELECTED_K_SIGMA = 3.0
SELECTION_INTENSITY = 0.35
RESTEP_K_SIGMA = (0.0, 3.0)
PARK_ANGLES_DEG = (40.0, 60.0, 90.0, 120.0)
PARK_K_SIGMA = (2.0, 3.0, 5.0)
PARK_HEADING_GAIN = 763.0
PARK_BAND_DEG = 5.0
PARK_DEADLINE = 120.0
PARK_HORIZON = 150.0
PARK_TRACE_TIMES = (1.0, 10.0, 30.0, 60.0, 90.0, 120.0, 149.0)
P1E_K_SIGMA = (1.0, 2.0, 3.0, 5.0)
P1E_COMPARISON_K_SIGMA = 0.0
P1E_INTENSITIES = (0.35, 0.5)
OFFSET_FLAG_DEG = 10.0
ADDENDUM_WORKERS = 8

PARKING_FINDING = {
    "source": "verifier scratch runs (verify_park.py, verify_step.py), unsaturated law, zero weather, k_h = 763, 60 s",
    "cases": [
        {"k_sigma": 5.0, "start_deg": [40.0, 60.0], "parked_chord_deg": 65.5},
        {"k_sigma": 3.0, "start_deg": [90.0], "parked_chord_deg": 100.0},
        {"k_sigma": 2.0, "start_deg": [120.0], "parked_chord_deg": 131.0},
    ],
    "mechanism": ("the correction k_sigma wrap(sigma_hat - phi) is unsaturated inside the heading error's outer wrap, "
                  "so once |k_sigma wrap(sigma_hat - phi)| passes pi the reference aliases and the hull is steered to "
                  "line up with the swung chord instead of back to the design angle: a spurious equilibrium near "
                  "(1 + k_sigma) |sigma - phi| ~ 2 pi plus the thrust-lateral balance (6 x 65.5 = 393, 4 x 100 = 400, "
                  "3 x 131 = 393 deg)"),
    "consistent_prior_pilot_evidence_non_gating": ("sway_results.json pilot, unsaturated k_sigma = 5 at intensity 0.35: per-seed "
                                                   "max chord std 27.9 / 7.3 / 19.3 / 18.3 / 33.7 deg against 10.1-12.3 deg at "
                                                   "k_sigma = 3"),
}


def wrap_array(angles):
    return np.angle(np.exp(1j * np.asarray(angles, dtype=float)))


def load_relative_chord(states: np.ndarray, sigma: np.ndarray, geometry) -> np.ndarray:
    """Chord angle measured from the load's attachment-face normal, ``wrap(sigma_i - theta_L - phi_i)``.

    Every parallel-formation attachment lies on the load's forward face (body x = const), whose
    normal is the load's body x-axis, and ``phi_i`` (the design chord angle, 0 for the parallel
    formation) is taken in the load frame, which equals the world frame at the equilibrium.
    """
    load_heading = np.atleast_2d(states)[:, 2][:, None]
    return wrap_array(np.atleast_2d(sigma) - load_heading - geometry.cable_angles[None, :])


def saturated_equivalent_prediction(heading_gain: float, intensity: float, k_sigma: float, limit: float = SWAY_LIMIT_V2) -> dict:
    """Statistical (Booton) linearization of the clamped correction on the reduced model.

    For a zero-mean Gaussian chord deviation of std s, ``clip(k x, -c, c)`` has equivalent gain
    ``k erf(c / (k s sqrt 2))``; the fixed point s = s_model(k_eq(s)) is the prediction.  Sensor
    noise, the heading estimator and non-Gaussian excursions are ignored; declared non-gating.
    """
    from scipy.optimize import brentq
    from scipy.special import erf
    from tether.theory.reduced_lti import reduced_model

    def chord(k: float) -> float:
        return float(np.max(reduced_model(model_input(heading_gain, intensity, k)).sigma_chord_angle))

    def k_eq(s: float) -> float:
        return k_sigma * float(erf(limit / (k_sigma * s * math.sqrt(2.0))))

    low, high = chord(k_sigma), chord(0.0)
    fixed = brentq(lambda s: chord(k_eq(s)) - s, low, high, xtol=1.0e-6)
    result = reduced_model(model_input(heading_gain, intensity, k_eq(fixed)))
    linear_range = limit / k_sigma
    return {
        "k_sigma": k_sigma, "heading_gain": heading_gain, "intensity": intensity,
        "equivalent_gain": k_eq(fixed),
        "chord_max_deg": math.degrees(fixed),
        "psi_max_deg": float(np.max(np.degrees(result.sigma_psi))),
        "linear_model_chord_max_deg": math.degrees(low),
        "linear_range_deg": math.degrees(linear_range),
        "clamp_active_probability": float(1.0 - erf(linear_range / (fixed * math.sqrt(2.0)))),
    }


def _write_once(path, payload: dict) -> str:
    if path.exists():
        raise SystemExit(f"{relative(path)} exists; declarations are never rewritten (write a dated addendum)")
    return write_bytes(path, json_bytes(payload))


def build_addendum() -> dict:
    model = json.loads(MODEL_PATH.read_text())
    stats = model["statistics"]
    ratios = {hg: {i: {_key(k): {"variance_ratio_max": stats[hg][i][_key(k)]["variance_ratio_max"],
                                 "psi_term_only_ratio_max": float(np.max(np.asarray(stats[hg][i][_key(k)]["psi_term_std_N"]) ** 2
                                                                         / (TENSION_FRACTION * np.asarray(stats[hg][i][_key(k)]["sigma_q_N"])) ** 2)),
                                 "chord_max_deg": stats[hg][i][_key(k)]["chord_max_deg"],
                                 "psi_max_deg": stats[hg][i][_key(k)]["psi_max_deg"]}
                       for k in CANDIDATES} for i in ("0.35", "0.5")} for hg in ("1000", "763")}
    flat = [(row["variance_ratio_max"], hg, i, k) for hg, table in ratios.items() for i, cells in table.items() for k, row in cells.items()]
    psi_only = [row["psi_term_only_ratio_max"] for table in ratios.values() for cells in table.values() for row in cells.values()]
    clause_a = {hg: {_key(k): {"chord_max_deg": stats[hg][_key(SELECTION_INTENSITY)][_key(k)]["chord_max_deg"],
                               "meets_10_deg": bool(stats[hg][_key(SELECTION_INTENSITY)][_key(k)]["chord_max_deg"] <= CHORD_LIMIT_DEG)}
                     for k in CANDIDATES} for hg in ("1000", "763")}
    selected = [k for k in CANDIDATES if clause_a[_key(DESIGN_HEADING_GAIN)][_key(k)]["meets_10_deg"]][0]
    if selected != SELECTED_K_SIGMA:
        raise SystemExit("the clause-A fallback does not reproduce the owner's ruling; refusing to write the addendum")
    return {
        "title": "Sway addendum 1: saturated sway correction, clause-A gain fallback, re-validation, and the Phase 1(e) shape-condition pilot",
        "date": "2026-09-13",
        "declared_before_any_result": True,
        "amends": {"declarations": relative(DECLARATIONS_PATH), "declarations_sha256": sha256_file(DECLARATIONS_PATH),
                   "results": relative(RESULTS_PATH), "results_sha256": sha256_file(RESULTS_PATH),
                   "model": relative(MODEL_PATH), "model_sha256": sha256_file(MODEL_PATH),
                   "rule": "the original declarations are not edited; this addendum carries every change and its reason"},
        "reason": {"parking_finding": PARKING_FINDING,
                   "clause_B": "clause B of IV.7's rule is unsatisfiable on the validated model (below)"},
        "owner_rulings": [
            "Sway law: clamp the correction at +-20 deg (0.349 rad), then re-run the step validation and the 5-seed pilot before Phase 1 picks a gain.",
            "Gain rule: clause B is unsatisfiable at every candidate gain at both intensities; the declared fallback applies clause A alone "
            "(model chord std <= 10 deg) at the declared branch intensity 0.35, which selects k_sigma = 3; a model-only selection, no plant "
            "pilot outcome enters it.",
            "Phase 2's graded grid: T0 in {0.6, 0.8, 1.0, 1.2, 1.4} kN at intensity 0.35, parallel formation, k_h = 1.5x the stability "
            "boundary (477/620/763/906/1050 N m/rad), k_sigma = 3 (saturated); intensity-0.5 cells only as out-of-(H1) comparisons.",
        ],
        "law": {
            "statement": "yaw_i = k_h wrap(theta_ref,i - theta_hat_i) - k_c sin(bearing_i); "
                         "theta_ref,i = theta_0,i - clip(k_sigma wrap(sigma_hat_i - phi_i), -0.349, +0.349)",
            "sigma_hat": "sigma_hat_i = theta_hat_i - bearing_i (the sign convention of sway_declarations.json, unchanged)",
            "limit_rad": SWAY_LIMIT_V2, "limit_deg": math.degrees(SWAY_LIMIT_V2),
            "linear_range_deg": {_key(k): math.degrees(SWAY_LIMIT_V2 / k) for k in CANDIDATES},
            "implementation": "ControllerBank(sway_limit=0.349) via FleetRunSpec.sway_limit; default None = the unsaturated law "
                              "bit-exactly, and FleetRunSpec.config_hash drops sway_limit when None (earlier hashes unchanged); "
                              "the clamp is inert at k_sigma = 0 (the term is not evaluated)",
            "no_spurious_equilibrium_argument": "with |correction| <= 20 deg the heading reference stays within 20 deg of theta_0, so it "
                                                "can no longer alias onto a chord swung far from phi - the parking mechanism is removed; "
                                                "whether any other equilibrium remains is checked on the plant by the no-parking test",
            "reduced_model": "unchanged (linear, unsaturated): the clamp is outside its linear range whenever |sigma - phi| > 20 deg / k_sigma",
        },
        "gain_rule": {
            "clause_B_unsatisfiable": {
                "statement": "V_i / (0.1 sigma_q,i)^2 < 1 fails at every candidate k_sigma in {1, 2, 3, 5}, at k_h in {1000, 763} and "
                             "intensity in {0.35, 0.5}; the psi term alone exceeds the bound everywhere (the term the loop cannot remove)",
                "minimum_ratio": {"value": min(flat)[0], "at": {"heading_gain": min(flat)[1], "intensity": min(flat)[2], "k_sigma": min(flat)[3]}},
                "minimum_ratio_at_rule_heading_gain_1000": min(r for r, hg, _, _ in flat if hg == "1000"),
                "minimum_psi_term_only_ratio": min(psi_only),
                "table": ratios,
                "source": "records/v2/phase0/sway_model.json statistics (unchanged)",
            },
            "fallback": {
                "rule": "k_sigma = the smallest candidate in {1, 2, 3, 5} whose model chord-angle std (max over cables) is <= 10 deg at "
                        "T0 = 1 kN, k_h = 1000 N m/rad (the rule's cell), at the declared branch intensity 0.35 (clause A alone)",
                "table_intensity_0.35": clause_a,
                "selected_k_sigma": selected,
                "numbers": "k_sigma = 3: 7.9 deg at k_h = 1000 and 9.3 deg at k_h = 763; k_sigma = 2 fails at 10.4 / 12.3 deg",
                "model_only": "no plant pilot outcome enters the selection; the Phase 1(e) cells are never used to re-select",
                "caveat_declared_now": "the model is linear and unsaturated; at k_sigma = 3 the clamp acts beyond 6.67 deg, inside one model "
                                       "chord std (7.9-9.3 deg), so the model's numbers do not describe the saturated loop - the pilot measures "
                                       "it, and a statistical-linearization prediction is declared in records/v2/phase1/p1e_declarations.json",
                "selection_stands_only_if": "step validation (i) passes at k_h = 1000 at k_sigma = 0 and 3 with the saturated law, and the "
                                            "no-parking test passes",
            },
        },
        "revalidation_plan": {
            "step_validation_i": {
                "cases": {"heading_gain": [1000.0, 763.0], "gating_heading_gain": 1000.0, "k_sigma": list(RESTEP_K_SIGMA),
                          "sway_limit_rad": SWAY_LIMIT_V2},
                "protocol": "identical to sway_declarations.json step_validation_i: seed 7000, +2 deg rigid rotation of vessel 2 about its load "
                            "attachment, zero weather, sensor noise present, perturbed and reference runs on common random numbers, plant "
                            "horizons and model settling times read from sway_model.json, PASS iff |T_s,plant - T_s,model| <= 0.30 T_s,model",
                "prediction": "small steps are unaffected: the clamp needs |3 (sigma_hat - phi)| > 20 deg, i.e. a 6.7 deg chord deviation, "
                              "against a 2.35 deg peak response plus 1 deg bearing noise, so the saturated traces should be bit-identical "
                              "to the unsaturated ones in sway_steps.npz; reported as the max |difference| of d_sigma",
                "clamp_proxy_reported": "max over the run of |k_sigma wrap(sigma_true - phi)| (true state; the controller sees sigma_hat)",
            },
            "no_parking_test": {
                "cases": {"start_offsets_deg": list(PARK_ANGLES_DEG), "k_sigma": list(PARK_K_SIGMA), "heading_gain": PARK_HEADING_GAIN,
                          "sway_limit_rad": SWAY_LIMIT_V2, "vessel": STEP_VESSEL, "seed": STEP_SEED, "horizon_s": PARK_HORIZON},
                "start": "the steady-tow equilibrium with vessel 2 rotated rigidly about its load attachment by the offset (to port): chord "
                         "length and psi unchanged, heading estimate initialised to the rotated true heading (rotated_start)",
                "weather": "none; sensor noise present",
                "criterion": "T_5 = the last 10 ms grid time in (0, 150 s] at which |wrap(sigma_2 - phi_2)| > 5 deg (world chord); a case "
                             "PASSES iff T_5 <= 120 s, the last sample is inside the band, and the run does not close",
                "test_passes_iff": "all 12 saturated cases pass",
                "control_non_gating": "the same 12 cases with the unsaturated law, reported to show the test discriminates",
                "reported_per_case": "T_5, chord at 1/10/30/60/90/120/149 s, peak |chord|, final psi, final load heading, closure, 1 s trace",
            },
            "if_revalidation_fails": "the saturated law is not validated: k_sigma = 3 does not stand, the Phase 1(e) pilot is still run and "
                                     "reported, P1-T11 is reported NOT DECIDABLE ON A VALIDATED LAW, and Phase 2 does not launch",
        },
        "phase1e_pilot": {
            "declarations": relative(P1E_DECLARATIONS_PATH),
            "cells": "5 seeds (7001-7005) x 300 s after a 20 s warm-up, parallel formation, T0 = 1 kN, k_h = 763, k_c = 100, linear drag, "
                     "Gaussian local weather, recording mode, k_sigma in {1, 2, 3, 5} saturated at intensities 0.35 and 0.5; k_sigma = 0 "
                     "(the v1 law at k_h = 763, clamp inert) at both intensities as a non-gating comparison",
        },
        "p1_t11_rule": {
            "cell": {"intensity": SELECTION_INTENSITY, "k_sigma": SELECTED_K_SIGMA, "sway_limit_rad": SWAY_LIMIT_V2, "heading_gain": PILOT_HEADING_GAIN},
            "per_vessel_statistic": "circular std sqrt(-2 ln R) of the samples pooled over the 5 seeds (10 ms state log decimated to 0.1 s, "
                                    "window [20 s, end], end = 320 s or the closure time), as sway_declarations.json pilot_iv",
            "cell_statistic": "max over the 5 vessels",
            "passes_iff": "world-frame chord statistic <= 15 deg AND psi statistic <= 15 deg (exactly 15.0 passes)",
            "frames": {
                "gating": "world frame: sigma_i = atan2 of the world chord direction, the plan's literal statistic inherited from v1 "
                          "(summaries.geometry_stats.cable_angle_std is world-frame); the plan text names no frame and cites v1's "
                          "world-frame numbers (38-92 deg)",
                "reported_non_gating": "load-relative chord wrap(sigma_i - theta_L - phi_i), chord angle minus the load's attachment-face "
                                       "normal (the load's body x-axis for the parallel formation)",
                "psi": "psi_i = wrap(theta_i - sigma_i), frame-free",
            },
            "transparency": "every other cell is reported with the same statistics and is never used to re-select k_sigma or the intensity",
            "outcomes": {
                "pass": "the shape condition of (H1) holds at intensity 0.35 with k_sigma = 3 (saturated): Phase 2 can launch at 0.35, "
                        "subject to the population gate (T10'), which is not decided here",
                "fail": "15 deg is unreachable with the declared loop at intensity 0.35: Phase 2 NO-LAUNCH, return to the formation and "
                        "weather design with the measured numbers (plan Phase 1 'T11 is the NO-LAUNCH gate'); nothing is falsified",
            },
        },
        "workers": ADDENDUM_WORKERS,
    }


def build_p1e_declarations() -> dict:
    if not ADDENDUM_PATH.exists():
        raise SystemExit("the addendum must be written first")
    model = json.loads(MODEL_PATH.read_text())
    stats = model["statistics"][_key(PILOT_HEADING_GAIN)]
    linear = {_key(i): {_key(k): {"chord_max_deg": stats[_key(i)][_key(k)]["chord_max_deg"],
                                  "psi_max_deg": stats[_key(i)][_key(k)]["psi_max_deg"],
                                  "mu_q2_N_max": float(np.max(stats[_key(i)][_key(k)]["mu_q2_N"]))}
                        for k in (P1E_COMPARISON_K_SIGMA,) + P1E_K_SIGMA} for i in P1E_INTENSITIES}
    quasi = {_key(i): {_key(k): saturated_equivalent_prediction(PILOT_HEADING_GAIN, i, k) for k in P1E_K_SIGMA} for i in P1E_INTENSITIES}
    return {
        "title": "Phase 1(e): shape-condition pilot on the saturated sway loop, and P1-T11 (plan v2 Phase 1, IV.7, II.2 (H1))",
        "date": "2026-09-13",
        "declared_before_any_result": True,
        "addendum": relative(ADDENDUM_PATH), "addendum_sha256": sha256_file(ADDENDUM_PATH),
        "plan": {"path": "ref/tail_of_the_tether_plan_v2.md", "sections": ["Phase 1 Cells (e)", "P1-T11", "Phase 1 outcome matrix",
                                                                           "IV.7", "II.2 (H1)", "III.3 P1-T11"]},
        "spec": {"formation": "parallel", "pretension_N": PRETENSION, "heading_gain": PILOT_HEADING_GAIN, "trim_gain": TRIM_GAIN,
                 "drag_law": "linear", "weather_distribution": "gaussian", "weather_direction": "local", "cable_mode": "recording",
                 "warmup_s": PILOT_WARMUP, "duration_s": PILOT_DURATION, "seeds": list(PILOT_SEEDS)},
        "cells": {"k_sigma": list(P1E_K_SIGMA), "sway_limit_rad": SWAY_LIMIT_V2, "intensities": list(P1E_INTENSITIES),
                  "comparison_non_gating": {"k_sigma": P1E_COMPARISON_K_SIGMA, "sway_limit_rad": None,
                                            "note": "the v1 law at k_h = 763; the clamp is inert at k_sigma = 0 so the unsaturated spec "
                                                    "(v1 hash) is run"},
                  "runs": len(P1E_INTENSITIES) * (len(P1E_K_SIGMA) + 1) * len(PILOT_SEEDS),
                  "simulated_seconds": len(P1E_INTENSITIES) * (len(P1E_K_SIGMA) + 1) * len(PILOT_SEEDS) * (PILOT_WARMUP + PILOT_DURATION)},
        "observables": {
            "sampling": "10 ms plant state log decimated to 0.1 s over [20 s, end]",
            "chord_world": "per-vessel circular std pooled over seeds; cell statistic max over vessels (GATES)",
            "chord_load_relative": "wrap(sigma_i - theta_L - phi_i); same statistics (reported, non-gating)",
            "psi": "wrap(theta_i - sigma_i); same statistics (GATES)",
            "load_yaw": "circular std of the load heading pooled over seeds",
            "closures": "runs ending in formation closure (chord < 1 m), with cable and time; the window ends at closure",
            "parked_vessel_diagnostics": f"per seed and vessel: circular mean of the world and load-relative chord (deg), per-seed "
                                         f"circular std, peak |wrap(sigma - phi)|; a vessel-seed is flagged OFFSET when |circular mean| > "
                                         f"{OFFSET_FLAG_DEG:g} deg (diagnostic, non-gating)",
            "clamp_duty": "fraction of samples with |k_sigma wrap(sigma_true - phi)| > 0.349 (true-state proxy for the clamp)",
            "slack_duty": "slack samples / window samples per cable at 1 ms (summaries.slack_samples), pooled over seeds",
            "onsets": "taut-to-slack transitions per cable in the window (summaries.onsets) and re-engagement marks, pooled",
            "clean_mean_q": "clean-set mean tension per cable against the model's mu_q2 (non-gating)",
        },
        "predictions_non_gating": {
            "linear_unsaturated_model_kh763": linear,
            "statistical_linearization_saturated_kh763": quasi,
            "qualitative": "the clamp lowers the loop's effective gain once the chord deviation exceeds 20/k_sigma deg, so the saturated "
                           "chord std is expected above the linear model's at every k_sigma, the more so the larger k_sigma (linear range "
                           "20/k_sigma deg); no parked vessels (per-seed chord means within 10 deg) are expected at any gain",
            "p1_t11_predicted_by_statistical_linearization": {
                "chord_world_deg": quasi[_key(SELECTION_INTENSITY)][_key(SELECTED_K_SIGMA)]["chord_max_deg"],
                "psi_deg": quasi[_key(SELECTION_INTENSITY)][_key(SELECTED_K_SIGMA)]["psi_max_deg"],
                "predicted_verdict": ("PASS" if quasi[_key(SELECTION_INTENSITY)][_key(SELECTED_K_SIGMA)]["chord_max_deg"] <= SHAPE_LIMIT_DEG
                                      and quasi[_key(SELECTION_INTENSITY)][_key(SELECTED_K_SIGMA)]["psi_max_deg"] <= SHAPE_LIMIT_DEG
                                      else "FAIL"),
                "status": "a non-gating prediction; the verdict is the plant measurement",
            },
            "prior_unsaturated_pilot_same_seeds": "sway_results.json pilot_iv (k_sigma = 3, 0.35: chord 10.0 deg, psi 14.6 deg) - the "
                                                  "k_sigma = 0 cells here must reproduce it exactly (same spec, same hash)",
        },
        "verdict_rule": "records/v2/phase0/sway_addendum_1.json p1_t11_rule (cell intensity 0.35, k_sigma = 3 saturated, k_h = 763; "
                        "world chord and psi statistics both <= 15 deg), conditional on the addendum's re-validation",
        "outputs": {"runs": relative(P1E_RUNS_PATH), "results": relative(P1E_RESULTS_PATH),
                    "revalidation": relative(ADDENDUM_RESULTS_PATH)},
        "workers": ADDENDUM_WORKERS,
    }


def write_addendum_declarations() -> None:
    _write_once(ADDENDUM_PATH, build_addendum())
    _write_once(P1E_DECLARATIONS_PATH, build_p1e_declarations())


def _require(path) -> str:
    if not path.exists():
        raise SystemExit(f"{relative(path)} must be written before any result")
    return sha256_file(path)


def park_settling(times: np.ndarray, sigma: np.ndarray, band: float = math.radians(PARK_BAND_DEG)) -> tuple[float, bool]:
    return settling_time(times, wrap_array(sigma), band)


def compute_revalidation(workers: int) -> dict:
    addendum_sha = _require(ADDENDUM_PATH)
    model = json.loads(MODEL_PATH.read_text())
    step_jobs = [StepJob(hg, k, perturbed, model["step_predictions"][_key(hg)][_key(k)]["plant_horizon_s"], sway_limit=SWAY_LIMIT_V2)
                 for hg in STEP_HEADING_GAINS for k in RESTEP_K_SIGMA for perturbed in (True, False)]
    park_jobs = [StepJob(PARK_HEADING_GAIN, k, True, PARK_HORIZON, sway_limit=limit, angle_deg=angle)
                 for limit in (SWAY_LIMIT_V2, None) for k in PARK_K_SIGMA for angle in PARK_ANGLES_DEG]
    results = run_pool(run_step_job, step_jobs + park_jobs, workers)
    step_results, park_results = results[: len(step_jobs)], results[len(step_jobs):]
    old = np.load(STEPS_PATH)
    band = SETTLING_FRACTION * math.radians(STEP_DEG)
    steps = {}
    for hg in STEP_HEADING_GAINS:
        for k in RESTEP_K_SIGMA:
            pair = {job.perturbed: result for job, result in zip(step_jobs, step_results) if job.heading_gain == hg and job.k_sigma == k}
            perturbed, reference = pair[True], pair[False]
            count = min(perturbed["time"].size, reference["time"].size)
            times = perturbed["time"][:count]
            response = perturbed["sigma"][:count, STEP_VESSEL] - reference["sigma"][:count, STEP_VESSEL]
            settling, censored = settling_time(times, response, band)
            predicted = model["step_predictions"][_key(hg)][_key(k)]["settling_s"]
            testable = math.isfinite(predicted) and predicted > 0.0 and not censored
            error = (settling - predicted) / predicted if testable else None
            name = f"kh{_key(hg)}_ks{_key(k)}"
            previous = old[f"{name}_dsigma"]
            shared = min(previous.size, response.size)
            proxy = max(float(np.max(np.abs(k * wrap_array(run["sigma"][:count])))) for run in (perturbed, reference))
            peak = int(np.argmax(np.abs(response)))
            steps[name] = {
                "heading_gain": hg, "k_sigma": k, "gating": hg == DESIGN_HEADING_GAIN, "sway_limit_rad": SWAY_LIMIT_V2,
                "model_settling_s": predicted, "plant_settling_s": settling, "plant_censored": censored,
                "relative_error": error, "pass": bool(testable and abs(error) <= TOLERANCE),
                "plant_peak_abs_deg": math.degrees(float(abs(response[peak]))), "plant_peak_time_s": float(times[peak]),
                "max_abs_difference_from_unsaturated_deg": math.degrees(float(np.max(np.abs(response[:shared] - previous[:shared])))),
                "samples_compared": int(shared), "unsaturated_samples": int(previous.size),
                "bit_identical_to_unsaturated": bool(shared == previous.size == response.size and np.array_equal(response, previous)),
                "clamp_proxy_max_deg": math.degrees(proxy), "clamp_limit_deg": math.degrees(SWAY_LIMIT_V2),
                "closure": [perturbed["closure"], reference["closure"]],
                "wall_seconds": perturbed["wall_seconds"] + reference["wall_seconds"],
            }
    parking = {}
    for job, run in zip(park_jobs, park_results):
        times = run["time"]
        sigma = wrap_array(run["sigma"][:, STEP_VESSEL])
        settle, censored = park_settling(times, sigma)
        closed = run["closure"] is not None
        passed = bool(not closed and not censored and settle <= PARK_DEADLINE)
        at = [int(min(np.searchsorted(times, t), times.size - 1)) for t in PARK_TRACE_TIMES]
        stride = int(round(1.0 / STATE_PERIOD))
        law = "saturated" if job.sway_limit is not None else "unsaturated_control"
        name = f"{law}_ks{_key(job.k_sigma)}_start{_key(job.angle_deg)}"
        parking[name] = {
            "law": law, "gating": job.sway_limit is not None, "k_sigma": job.k_sigma, "start_deg": job.angle_deg,
            "heading_gain": job.heading_gain, "sway_limit_rad": job.sway_limit,
            "T5_s": settle, "censored": censored, "closure": run["closure"], "pass": passed,
            "end_time_s": float(times[-1]) if times.size else 0.0,
            "chord_deg_at": {f"{times[i]:.2f}": math.degrees(float(sigma[i])) for i in at},
            "peak_abs_chord_deg": math.degrees(float(np.max(np.abs(sigma)))),
            "final_chord_deg": math.degrees(float(sigma[-1])),
            "final_load_relative_chord_deg": math.degrees(float(wrap_array(sigma[-1] - run["load_heading"][-1]))),
            "final_psi_deg": math.degrees(float(run["psi"][-1, STEP_VESSEL])),
            "final_load_heading_deg": math.degrees(float(run["load_heading"][-1])),
            "chord_trace_1s_deg": np.degrees(sigma[::stride]).tolist(),
            "wall_seconds": run["wall_seconds"],
        }
    gating_steps = [steps[f"kh{_key(DESIGN_HEADING_GAIN)}_ks{_key(k)}"]["pass"] for k in RESTEP_K_SIGMA]
    gating_park = [case["pass"] for case in parking.values() if case["gating"]]
    summary = {
        "addendum_sha256": addendum_sha,
        "model_sha256": sha256_file(MODEL_PATH),
        "unsaturated_steps_npz_sha256": sha256_file(STEPS_PATH),
        "source": source_state(),
        "step_validation_i": {"cases": steps, "gate_k_sigma_0": bool(gating_steps[0]), "gate_k_sigma_3": bool(gating_steps[1])},
        "no_parking_test": {"cases": parking, "saturated_cases": len(gating_park), "saturated_passed": int(sum(gating_park)),
                            "pass": bool(all(gating_park)),
                            "unsaturated_control_passed": int(sum(case["pass"] for case in parking.values() if not case["gating"]))},
        "revalidation_passes": bool(all(gating_steps) and all(gating_park)),
        "selection": {"k_sigma": SELECTED_K_SIGMA, "stands": bool(all(gating_steps) and all(gating_park)),
                      "label": "clause-A fallback at intensity 0.35 (model-only), owner ruling"},
    }
    write_bytes(ADDENDUM_RESULTS_PATH, json_bytes(summary))
    return summary


@dataclass(frozen=True)
class P1eJob:
    k_sigma: float
    intensity: float
    seed: int
    sway_limit: float | None

    @property
    def cost(self) -> float:
        return PILOT_WARMUP + PILOT_DURATION


def p1e_jobs() -> list[P1eJob]:
    jobs = []
    for intensity in P1E_INTENSITIES:
        for k in (P1E_COMPARISON_K_SIGMA,) + P1E_K_SIGMA:
            limit = None if k == P1E_COMPARISON_K_SIGMA else SWAY_LIMIT_V2
            jobs.extend(P1eJob(k, intensity, seed, limit) for seed in PILOT_SEEDS)
    return jobs


def _circular_mean(angles: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * np.asarray(angles)).mean(axis=0)) if len(angles) else np.full(np.shape(angles)[1:], np.nan)


def run_p1e_job(job: P1eJob) -> dict:
    from tether.campaign.fleet_run import FleetRunSpec, build_run, run_to_end
    from tether.campaign.summaries import summarize_run

    spec = FleetRunSpec(formation="parallel", pretension=PRETENSION, heading_gain=PILOT_HEADING_GAIN, trim_gain=TRIM_GAIN,
                        drag_law="linear", weather_distribution="gaussian", weather_direction="local", weather_scale=job.intensity,
                        duration=PILOT_DURATION, warmup=PILOT_WARMUP, cable_mode="recording", k_sigma=job.k_sigma,
                        sway_limit=job.sway_limit)
    run = run_to_end(build_run(spec, job.seed))
    summary = summarize_run(run, PILOT_WARMUP, PRETENSION)
    geometry = run.fleet.geometry
    log = run.fleet.cables.log
    times = log.state_time[: log.state_count]
    end = float(summary["meta"]["end_time"])
    window = np.flatnonzero((times >= PILOT_WARMUP - 1.0e-9) & (times <= end + 1.0e-9))
    picked = window[:: int(round(PILOT_DECIMATION / STATE_PERIOD))]
    states = log.state[picked]
    n = geometry.vessel_count
    sigma, psi = chord_and_misalignment(states, geometry) if picked.size else (np.empty((0, n)), np.empty((0, n)))
    relative_chord = load_relative_chord(states, sigma, geometry) if picked.size else np.empty((0, n))
    deviation = wrap_array(sigma - geometry.cable_angles[None, :]) if picked.size else np.empty((0, n))
    load = states[:, 2:3] if picked.size else np.empty((0, 1))
    count = int(picked.size)

    def sums(angles):
        return {"cos": np.cos(angles).sum(axis=0), "sin": np.sin(angles).sum(axis=0)}

    def std_deg(angles):
        return np.degrees(_circular_std_from_sums(np.cos(angles).sum(axis=0), np.sin(angles).sum(axis=0), count)) if count else \
            np.full(angles.shape[1], np.nan)

    q_moments = summary["moments"]["q"]
    clamp = (np.abs(job.k_sigma * deviation) > SWAY_LIMIT_V2).sum(axis=0) if (count and job.k_sigma != 0.0) else np.zeros(n, dtype=int)
    return {
        "k_sigma": job.k_sigma, "intensity": job.intensity, "seed": job.seed, "sway_limit": job.sway_limit,
        "config_hash": spec.config_hash(), "samples": count,
        "chord_sums": sums(sigma), "relative_chord_sums": sums(relative_chord), "psi_sums": sums(psi), "load_sums": sums(load),
        "chord_std_deg": std_deg(sigma), "relative_chord_std_deg": std_deg(relative_chord), "psi_std_deg": std_deg(psi),
        "load_yaw_std_deg": std_deg(load),
        "chord_mean_deg": np.degrees(_circular_mean(sigma)), "relative_chord_mean_deg": np.degrees(_circular_mean(relative_chord)),
        "psi_mean_deg": np.degrees(_circular_mean(psi)),
        "peak_abs_deviation_deg": np.degrees(np.max(np.abs(deviation), axis=0)) if count else np.full(n, np.nan),
        "clamp_active_samples": clamp,
        "geometry_stats_chord_std_deg": np.degrees(summary["geometry_stats"]["cable_angle_std"]),
        "geometry_stats_load_yaw_std_deg": math.degrees(summary["geometry_stats"]["load_yaw_std"]),
        "closure": summary["meta"]["closure"], "end_time": end, "exposure_s": float(summary["meta"]["exposure"]),
        "onsets": summary["onsets"], "slack_samples": summary["slack_samples"], "window_samples": summary["meta"]["window_samples"],
        "marks": int(summary["marks"]["t_up"].size),
        "clean_q_count": q_moments[0], "clean_q_sum": q_moments[1], "clean_q_square": q_moments[2],
        "tracker_dropped": summary["meta"]["tracker_dropped"],
        "wall_seconds": run.wall_seconds,
    }


def compute_p1e(workers: int) -> list[dict]:
    addendum_sha = _require(ADDENDUM_PATH)
    declarations_sha = _require(P1E_DECLARATIONS_PATH)
    runs = run_pool(run_p1e_job, p1e_jobs(), workers)
    write_bytes(P1E_RUNS_PATH, json_bytes({"addendum_sha256": addendum_sha, "declarations_sha256": declarations_sha, "runs": runs}))
    return runs


def _pooled_std_deg(chosen: list[dict], key: str, count: int) -> np.ndarray:
    return np.degrees(_circular_std_from_sums(sum(np.asarray(r[key]["cos"]) for r in chosen),
                                              sum(np.asarray(r[key]["sin"]) for r in chosen), count))


def p1e_cell(chosen: list[dict]) -> dict:
    count = sum(r["samples"] for r in chosen)
    chord = _pooled_std_deg(chosen, "chord_sums", count)
    relative_chord = _pooled_std_deg(chosen, "relative_chord_sums", count)
    psi = _pooled_std_deg(chosen, "psi_sums", count)
    load = _pooled_std_deg(chosen, "load_sums", count)
    q_count = sum(np.asarray(r["clean_q_count"]) for r in chosen)
    window = sum(int(r["window_samples"]) for r in chosen)
    slack = sum(np.asarray(r["slack_samples"]) for r in chosen)
    onsets = sum(np.asarray(r["onsets"]) for r in chosen)
    exposure = float(sum(r["exposure_s"] for r in chosen))
    closures = [[r["seed"], r["closure"]] for r in chosen if r["closure"] is not None]
    offsets = [[r["seed"], vessel, float(r["chord_mean_deg"][vessel])] for r in chosen if r["samples"]
               for vessel in range(len(r["chord_mean_deg"])) if abs(r["chord_mean_deg"][vessel]) > OFFSET_FLAG_DEG]
    return {
        "seeds": [r["seed"] for r in chosen], "config_hashes": sorted({r["config_hash"] for r in chosen}),
        "sway_limit": chosen[0]["sway_limit"],
        "chord_world_stat_deg": float(np.max(chord)), "chord_world_std_deg_per_vessel": chord,
        "chord_world_mean_over_vessels_deg": float(np.mean(chord)),
        "chord_load_relative_stat_deg": float(np.max(relative_chord)), "chord_load_relative_std_deg_per_vessel": relative_chord,
        "psi_stat_deg": float(np.max(psi)), "psi_std_deg_per_vessel": psi, "psi_mean_over_vessels_deg": float(np.mean(psi)),
        "load_yaw_std_deg": float(load[0]),
        "chord_world_meets_15": bool(np.max(chord) <= SHAPE_LIMIT_DEG),
        "chord_load_relative_meets_15": bool(np.max(relative_chord) <= SHAPE_LIMIT_DEG),
        "psi_meets_15": bool(np.max(psi) <= SHAPE_LIMIT_DEG),
        "shape_condition_world_frame": bool(np.max(chord) <= SHAPE_LIMIT_DEG and np.max(psi) <= SHAPE_LIMIT_DEG),
        "per_seed": {
            "chord_world_std_deg": [r["chord_std_deg"] for r in chosen],
            "chord_load_relative_std_deg": [r["relative_chord_std_deg"] for r in chosen],
            "psi_std_deg": [r["psi_std_deg"] for r in chosen],
            "load_yaw_std_deg": [r["load_yaw_std_deg"] for r in chosen],
            "chord_world_mean_deg": [r["chord_mean_deg"] for r in chosen],
            "chord_load_relative_mean_deg": [r["relative_chord_mean_deg"] for r in chosen],
            "peak_abs_deviation_deg": [r["peak_abs_deviation_deg"] for r in chosen],
            "samples": [r["samples"] for r in chosen], "exposure_s": [r["exposure_s"] for r in chosen],
        },
        "offset_vessel_seeds": offsets, "offset_flag_deg": OFFSET_FLAG_DEG,
        "clamp_duty_per_vessel": sum(np.asarray(r["clamp_active_samples"], dtype=float) for r in chosen) / max(count, 1),
        "closures": len(closures), "closure_runs": closures, "closure_fraction": len(closures) / len(chosen),
        "exposure_s": exposure,
        "slack_duty_per_cable": slack / max(window, 1), "slack_duty": float(np.sum(slack) / max(window * slack.size, 1)),
        "onsets_per_cable": onsets, "onsets": int(np.sum(onsets)),
        "onset_rate_per_cable_s": float(np.sum(onsets)) / max(exposure * onsets.size, 1.0e-12),
        "marks": int(sum(r["marks"] for r in chosen)),
        "clean_mean_q_N_per_cable": sum(np.asarray(r["clean_q_sum"]) for r in chosen) / np.maximum(q_count, 1.0),
        "cross_check_geometry_stats_chord_max_deg_per_seed": [float(np.nanmax(r["geometry_stats_chord_std_deg"])) for r in chosen],
        "tracker_dropped": int(sum(r["tracker_dropped"] for r in chosen)),
        "wall_seconds": float(sum(r["wall_seconds"] for r in chosen)),
    }


def analyse_p1e() -> dict:
    addendum_sha = _require(ADDENDUM_PATH)
    declarations_sha = _require(P1E_DECLARATIONS_PATH)
    revalidation_sha = _require(ADDENDUM_RESULTS_PATH)
    declarations = json.loads(P1E_DECLARATIONS_PATH.read_text())
    revalidation = json.loads(ADDENDUM_RESULTS_PATH.read_text())
    runs = json.loads(P1E_RUNS_PATH.read_text())["runs"]
    cells = {}
    for intensity in P1E_INTENSITIES:
        cells[_key(intensity)] = {}
        for k in (P1E_COMPARISON_K_SIGMA,) + P1E_K_SIGMA:
            cell = p1e_cell([r for r in runs if r["intensity"] == intensity and r["k_sigma"] == k])
            linear = declarations["predictions_non_gating"]["linear_unsaturated_model_kh763"][_key(intensity)][_key(k)]
            quasi = declarations["predictions_non_gating"]["statistical_linearization_saturated_kh763"][_key(intensity)].get(_key(k))
            cell["model_non_gating"] = {"linear_unsaturated": linear, "statistical_linearization_saturated": quasi,
                                        "plant_over_linear_chord": cell["chord_world_stat_deg"] / linear["chord_max_deg"],
                                        "plant_over_linear_psi": cell["psi_stat_deg"] / linear["psi_max_deg"]}
            cells[_key(intensity)][_key(k)] = cell
    prior = json.loads(RESULTS_PATH.read_text())["pilot_iv"]["cells"]
    comparison_reproduces = {_key(i): bool(np.allclose(cells[_key(i)]["0"]["chord_world_std_deg_per_vessel"],
                                                       prior[_key(i)]["0"]["chord_std_deg_per_vessel"], rtol=0.0, atol=1.0e-9)
                                           and np.allclose(cells[_key(i)]["0"]["psi_std_deg_per_vessel"],
                                                           prior[_key(i)]["0"]["psi_std_deg_per_vessel"], rtol=0.0, atol=1.0e-9))
                             for i in P1E_INTENSITIES}
    gate = cells[_key(SELECTION_INTENSITY)][_key(SELECTED_K_SIGMA)]
    validated = bool(revalidation["revalidation_passes"])
    shape = gate["shape_condition_world_frame"]
    if not validated:
        verdict, outcome = "NOT DECIDABLE ON A VALIDATED LAW", ("the saturated law failed re-validation: k_sigma = 3 does not stand; "
                                                               "Phase 2 does not launch")
    elif shape:
        verdict, outcome = "PASS", ("the shape condition of (H1) holds at intensity 0.35 with k_sigma = 3 (saturated): Phase 2 can "
                                    "launch at 0.35, subject to the population gate (T10'), not decided here")
    else:
        verdict, outcome = "FAIL", ("15 deg is unreachable with the declared loop at intensity 0.35: Phase 2 NO-LAUNCH; the plan "
                                    "returns to the formation and weather design with the measured numbers")
    results = {
        "addendum_sha256": addendum_sha, "declarations_sha256": declarations_sha, "revalidation_sha256": revalidation_sha,
        "runs_sha256": sha256_file(P1E_RUNS_PATH), "source": source_state(),
        "p1_t11": {
            "cell": {"intensity": SELECTION_INTENSITY, "k_sigma": SELECTED_K_SIGMA, "sway_limit_rad": SWAY_LIMIT_V2,
                     "heading_gain": PILOT_HEADING_GAIN},
            "chord_world_stat_deg": gate["chord_world_stat_deg"], "psi_stat_deg": gate["psi_stat_deg"],
            "chord_load_relative_stat_deg_non_gating": gate["chord_load_relative_stat_deg"],
            "chord_world_meets_15": gate["chord_world_meets_15"], "psi_meets_15": gate["psi_meets_15"],
            "revalidation_passes": validated, "verdict": verdict, "outcome": outcome,
            "closures": gate["closures"],
        },
        "transparency_every_cell": {i: {k: {"chord_world_stat_deg": c["chord_world_stat_deg"],
                                                        "chord_load_relative_stat_deg": c["chord_load_relative_stat_deg"],
                                                        "psi_stat_deg": c["psi_stat_deg"], "load_yaw_std_deg": c["load_yaw_std_deg"],
                                                        "shape_condition_world_frame": c["shape_condition_world_frame"],
                                                        "closures": c["closures"], "offset_vessel_seeds": len(c["offset_vessel_seeds"]),
                                                        "slack_duty": c["slack_duty"], "onsets": c["onsets"]}
                                              for k, c in row.items()} for i, row in cells.items()},
        "transparency_note": "reported for transparency only; never used to re-select k_sigma or the intensity",
        "comparison_k0_reproduces_prior_pilot": comparison_reproduces,
        "cells": cells,
    }
    write_bytes(P1E_RESULTS_PATH, json_bytes(results))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("model", "steps", "pilot", "analyse", "all",
                                            "addendum-declare", "revalidate", "p1e", "p1e-analyse", "addendum-all"))
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.command == "addendum-declare":
        write_addendum_declarations()
        return
    if arguments.command in ("revalidate", "addendum-all"):
        summary = compute_revalidation(arguments.workers)
        print(json.dumps({"steps": {n: [c["plant_settling_s"], c["model_settling_s"], c["pass"], c["bit_identical_to_unsaturated"]]
                                    for n, c in summary["step_validation_i"]["cases"].items()},
                          "parking": {n: [c["T5_s"], c["pass"]] for n, c in summary["no_parking_test"]["cases"].items()},
                          "revalidation_passes": summary["revalidation_passes"]}, indent=1))
    if arguments.command in ("p1e", "addendum-all"):
        compute_p1e(arguments.workers)
    if arguments.command in ("p1e-analyse", "addendum-all"):
        results = analyse_p1e()
        print(json.dumps({"p1_t11": results["p1_t11"], "cells": results["transparency_every_cell"]}, indent=1, default=str))
    if arguments.command in ("addendum-declare", "revalidate", "p1e", "p1e-analyse", "addendum-all"):
        return
    if arguments.command in ("model", "all"):
        model = compute_model()
        print(json.dumps({"selection": model["selection"]["model_selection"],
                          "horizons": {hg: {k: v["plant_horizon_s"] for k, v in row.items()} for hg, row in model["step_predictions"].items()}}))
    if arguments.command in ("steps", "all"):
        summary = compute_steps(arguments.workers)
        print(json.dumps({name: [case["settling_s"], case["censored"]] for name, case in summary["cases"].items()}))
    if arguments.command in ("pilot", "all"):
        compute_pilot(arguments.workers)
    if arguments.command in ("analyse", "all"):
        results = analyse()
        print(json.dumps({"selection": results["selection_ii"], "branch": results["branch"]}, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
