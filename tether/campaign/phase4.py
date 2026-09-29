"""Phase 4: fleet-level dependence (plan Part V, Phase 4).

Commands: ``shocks`` (fan shock cell for the Prop. 9 ordering and the max-functional),
``predict``, ``compute``, ``analyse``, ``replay``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle
import sys
from dataclasses import asdict, dataclass

import numpy as np

from tether.campaign.common import RECORDS, json_bytes, npz_bytes, relative, run_pool, sha256_file, source_state, write_bytes
from tether.campaign.fleet_run import FleetRunSpec, build_run, run_to_end
from tether.campaign.stationary import StationaryJob, compute_jobs, load_cache, run_stationary_job
from tether.physics import constants
from tether.physics.weather import ar1_coefficient

RECORD_DIR = RECORDS / "phase4"
SHOCK_CACHE = RECORD_DIR / "cache" / "phase4_shocks.pkl"
PREDICTIONS_PATH = RECORD_DIR / "phase4_predictions.json"
CACHE_PATH = RECORD_DIR / "cache" / "phase4_compute.pkl"
RESULTS_PATH = RECORD_DIR / "phase4_results.json"
RECORD_PATH = RECORD_DIR / "phase4_records.npz"
MANIFEST_PATH = RECORD_DIR / "phase4_manifest.json"

ARC = 0.55
PRETENSION = 1000.0
HEADING_GAIN = 500.0
INTENSITY = 1.0
CLASSES = ("gaussian", "student_t3")
FRONTS = {"broadside_port": -math.pi / 2.0, "bow_quartering": -3.0 * math.pi / 4.0, "broadside_starboard": math.pi / 2.0}
PILOT_SEEDS = (4001, 4002)
STAT_SEEDS = tuple(range(4003, 4023))
DURATION = 600.0
WARMUP = 20.0
U_LEVELS = (0.90, 0.95, 0.98, 0.99)
SAMPLE_SPACING = 2.0
BLOCK_BOOTSTRAP = 30.0
EPISODE_WINDOW = 5.0
QUANTILES = (50, 80, 95, 99)
POWER_MIN = 20
SHOCK_AMPLITUDES = (4.0, 8.0, 16.0, 32.0)
SHOCK_DURATION = 40.0
ALPHA = 3.0
IMPACT_FACTOR = 0.880
PIVOT_NU_GRID = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0)
PIVOT_U = (0.90, 0.95, 0.98, 0.99, 0.995)
PIVOT_DRAWS = 400_000
PIVOT_BOOT = 400

DECLARATIONS = {
    "cells": "fan formation, arc half-angle 0.55 rad (cables at 0, +/-15.8, +/-31.5 deg), T0 = 1 kN, k_h = 500, intensity 1.0; weather class {gaussian, t3} x direction {local, common front at broadside port (-90 deg), bow quartering (-135 deg), broadside starboard (+90 deg)} = 8 cells (the plan's count of 12 cannot be derived from the listed factors); front angles are the world direction in which the gust pushes",
    "instantaneous": "(q_i, q_j) sampled every 2 s from the 1 s decimated records, only where both cables are taut; margins rank-transformed; chi and chibar intervals by moving-block bootstrap with 30 s blocks (samples 2 s apart are correlated: the q integral time scale is about 8 s); Gaussian-copula curve at rho = 2 sin(pi rho_S / 6)",
    "block_maxima": "5 s block maxima of the applied tension per cable (all samples), Gaussian copula at the block-max rank correlation as an empirical null (EMP)",
    "episodes": "virtual severance events at T_b: snap marks with T_peak > T_b (time t_up) and taut cluster maxima above T_b (cluster peak time); an episode is a 5 s window opened by the first event; coincidence = share of episodes with >= 2 distinct cables",
    "ordering": "predicted rank of chi_ij(0.99) under common t3 from the fan shock cell: pair joint severance radius r_ij = min over the two front directions of max(R_i, R_j) at the cell's q95 threshold; smaller radius = stronger dependence",
    "max_functional": "coincidence under common t3 predicted as sum_s w_s R_(2)(s)^-alpha / sum_s w_s R_(1)(s)^-alpha over the front's two directions, R_(1) <= R_(2) the two smallest cable severance radii",
    "windward": "under a broadside front pushing toward -y the windward cables are the port cables (positive chord angle, indices 3 and 4) and the leeward cables the starboard ones (0 and 1); mirrored for the starboard front",
    "pivot": "T1-fail pivot (plan outcome matrix; operational form fixed after T1 failed and before the pivot copula was fitted): a five-cable Student-t copula fitted to the rank-transformed instantaneous state (2 s samples with all five cables taut) - correlation from pairwise Kendall tau, sin(pi tau / 2), nearest positive definite; nu by profile pseudo-likelihood on a grid. It reproduces the Gaussian half of T4 when, in every Gaussian cell with a powered T4 trend, its fleet coincidence C(u) = P(>= 2 cables above their u-quantile | >= 1) over u in {0.90, 0.95, 0.98, 0.99, 0.995} (the state-level counterpart of the event coincidence) decreases (Spearman < -0.5, top below bottom) and lies inside the seed-bootstrap 95% interval of the empirical C(u) at every u. The fitted copula's pairwise chi and chibar are also re-checked against the T1 bands (reported)",
}


def cell_name(distribution: str, direction: str) -> str:
    return f"{'G' if distribution == 'gaussian' else 't3'}_{direction}"


def cell_specs() -> dict[str, FleetRunSpec]:
    specs = {}
    for distribution in CLASSES:
        for direction in ("local",) + tuple(FRONTS):
            specs[cell_name(distribution, direction)] = FleetRunSpec(
                formation="fan", arc_half_angle=ARC, pretension=PRETENSION, heading_gain=HEADING_GAIN,
                weather_distribution=distribution, weather_direction="local" if direction == "local" else "front",
                weather_front_angle=None if direction == "local" else FRONTS[direction], weather_scale=INTENSITY,
                duration=DURATION, warmup=WARMUP,
            )
    return specs


def all_jobs() -> list[StationaryJob]:
    return [StationaryJob(n, s, seed, pilot=seed in PILOT_SEEDS) for n, s in cell_specs().items() for seed in PILOT_SEEDS + STAT_SEEDS]


@dataclass(frozen=True)
class FrontShock:
    front: str
    sign: float
    amplitude: float
    cost: float = SHOCK_DURATION


def run_front_shock(job: FrontShock) -> dict:
    angle = FRONTS[job.front] + (0.0 if job.sign > 0 else math.pi)
    samples = int(math.ceil(SHOCK_DURATION / constants.WEATHER_PERIOD)) + 2
    phi = ar1_coefficient()
    decay = phi ** np.arange(samples)
    scale = math.sqrt(2.0) * INTENSITY * math.sqrt(1.0 - phi**2)
    weather = np.zeros((samples, constants.VESSEL_COUNT + 1, 2))
    stds = np.array([constants.LOAD_WEATHER_STD] + [constants.VESSEL_WEATHER_STD] * constants.VESSEL_COUNT)
    start = int(round(1.0 / constants.WEATHER_PERIOD))
    for body in range(constants.VESSEL_COUNT + 1):
        magnitude = job.amplitude * scale * stds[body]
        weather[start:, body, 0] = magnitude * math.cos(angle) * decay[: samples - start]
        weather[start:, body, 1] = magnitude * math.sin(angle) * decay[: samples - start]
    spec = FleetRunSpec(formation="fan", arc_half_angle=ARC, pretension=PRETENSION, heading_gain=HEADING_GAIN, weather_scale=0.0,
                        duration=SHOCK_DURATION, warmup=0.0, log_state=False)
    run = run_to_end(build_run(spec, 1, weather=weather))
    log = run.fleet.cables.log
    e = log.elongation[: log.count]
    rate = log.rate[: log.count]
    q = np.where(e > 0, constants.CABLE_STIFFNESS * e + constants.CABLE_DAMPING * rate, 0.0)
    marks = run.fleet.cables.reengagements
    peaks = [float(max([q[:, c].max()] + [m.T_peak for m in marks if m.cable == c])) for c in range(constants.VESSEL_COUNT)]
    return {"job": asdict(job), "peaks": peaks, "closure": run.fleet.cables.closure}


def front_shock_jobs() -> list[FrontShock]:
    return [FrontShock(front, sign, amplitude) for front in FRONTS for sign in (1.0, -1.0) for amplitude in SHOCK_AMPLITUDES]


def _radius(amplitudes, peaks, threshold) -> float:
    from tether.campaign.phase3 import severance_radius

    return severance_radius(dict(zip(amplitudes, peaks)), threshold)


def front_radii(shocks: list[dict], front: str, threshold: float) -> np.ndarray:
    """Severance radius per (direction sign, cable) for one front at ``threshold``."""
    radii = np.full((2, constants.VESSEL_COUNT), np.inf)
    for s_index, sign in enumerate((1.0, -1.0)):
        rows = sorted((r for r in shocks if r["job"]["front"] == front and r["job"]["sign"] == sign), key=lambda r: r["job"]["amplitude"])
        amplitudes = [r["job"]["amplitude"] for r in rows]
        for cable in range(constants.VESSEL_COUNT):
            radii[s_index, cable] = _radius(amplitudes, [r["peaks"][cable] for r in rows], threshold)
    return radii


def predict(threshold: float = 8000.0) -> dict:
    with SHOCK_CACHE.open("rb") as handle:
        shocks = pickle.load(handle)
    predictions = {"schema_version": 1, "source": source_state(), "declarations": DECLARATIONS, "threshold": threshold, "fronts": {}}
    for front in FRONTS:
        radii = front_radii(shocks, front, threshold)
        pairs = {}
        for i, j in itertools.combinations(range(constants.VESSEL_COUNT), 2):
            joint = float(np.min(np.maximum(radii[:, i], radii[:, j])))
            pairs[f"{i}{j}"] = joint
        sorted_radii = np.sort(radii, axis=1)
        first = sorted_radii[:, 0]
        second = sorted_radii[:, 1]
        numerator = float(np.sum(np.where(np.isfinite(second), second ** -ALPHA, 0.0)))
        denominator = float(np.sum(np.where(np.isfinite(first), first ** -ALPHA, 0.0)))
        predictions["fronts"][front] = {
            "radii": radii.tolist(),
            "pair_joint_radius": pairs,
            "max_functional_coincidence": numerator / denominator if denominator > 0 else None,
        }
    write_bytes(PREDICTIONS_PATH, json_bytes(predictions))
    return predictions


# ----------------------------------------------------------------------------- analysis


def _pool(summaries: list[dict]) -> dict:
    marks = {key: np.concatenate([s["marks"][key] for s in summaries]) for key in summaries[0]["marks"]}
    marks["seed"] = np.concatenate([np.full(s["marks"]["t_up"].size, s["meta"]["seed"]) for s in summaries])
    peaks = {key: np.concatenate([s["taut_peaks"][key] for s in summaries]) for key in summaries[0]["taut_peaks"]}
    peaks["seed"] = np.concatenate([np.full(s["taut_peaks"]["peak"].size, s["meta"]["seed"]) for s in summaries])
    return {
        "marks": marks,
        "taut_peaks": peaks,
        "q_samples": [s["samples"]["q"] for s in summaries],
        "taut_samples": [s["samples"]["taut"] for s in summaries],
        "block_maxima": [s["block_maxima"] for s in summaries],
        "seeds": [s["meta"]["seed"] for s in summaries],
        "exposure": float(sum(s["meta"]["exposure"] for s in summaries)),
        "wall_seconds": float(sum(s["meta"]["wall_seconds"] for s in summaries)),
        "sim_seconds": float(sum(s["meta"]["sim_seconds"] for s in summaries)),
        "closures": [s["meta"]["closure"] for s in summaries],
    }


def _pair_series(cell: dict, i: int, j: int, spacing: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, seeds = [], [], []
    for q, taut, seed in zip(cell["q_samples"], cell["taut_samples"], cell["seeds"]):
        q = q[::spacing]
        taut = taut[::spacing]
        both = taut[:, i] & taut[:, j]
        xs.append(q[both, i])
        ys.append(q[both, j])
        seeds.append(np.full(int(both.sum()), seed))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(seeds)


def _events(cell: dict, threshold: float) -> dict[int, list[tuple[float, int]]]:
    events: dict[int, list[tuple[float, int]]] = {}
    marks = cell["marks"]
    hits = marks["T_peak"] > threshold
    for t, cable, seed in zip(marks["t_up"][hits], marks["cable"][hits], marks["seed"][hits]):
        events.setdefault(int(seed), []).append((float(t), int(cable)))
    peaks = cell["taut_peaks"]
    hits = peaks["peak"] > threshold
    for t, cable, seed in zip(peaks["time"][hits], peaks["cable"][hits], peaks["seed"][hits]):
        events.setdefault(int(seed), []).append((float(t), int(cable)))
    return events


def coincidence(cell: dict, threshold: float) -> tuple[int, int]:
    episodes = 0
    multi = 0
    for seed, items in _events(cell, threshold).items():
        items.sort()
        index = 0
        while index < len(items):
            start = items[index][0]
            cables = set()
            while index < len(items) and items[index][0] <= start + EPISODE_WINDOW:
                cables.add(items[index][1])
                index += 1
            episodes += 1
            multi += int(len(cables) >= 2)
    return episodes, multi


def _five_vectors(cell: dict, spacing: int) -> tuple[np.ndarray, np.ndarray]:
    rows, seeds = [], []
    for q, taut, seed in zip(cell["q_samples"], cell["taut_samples"], cell["seeds"]):
        q, taut = q[::spacing], taut[::spacing]
        keep = taut.all(axis=1)
        rows.append(q[keep])
        seeds.append(np.full(int(keep.sum()), seed))
    return np.concatenate(rows), np.concatenate(seeds)


def fit_t_copula(u: np.ndarray) -> dict:
    """Student-t copula by maximum pseudo-likelihood: correlation from pairwise Kendall tau
    (sin(pi tau / 2), nearest positive definite), nu by profile over ``PIVOT_NU_GRID``; the
    Gaussian copula's log-likelihood at the same correlation for comparison."""
    from scipy import stats

    d = u.shape[1]
    tau = np.eye(d)
    for i, j in itertools.combinations(range(d), 2):
        tau[i, j] = tau[j, i] = stats.kendalltau(u[:, i], u[:, j]).statistic
    values, vectors = np.linalg.eigh(np.sin(0.5 * math.pi * tau))
    corr = vectors @ np.diag(np.clip(values, 1.0e-6, None)) @ vectors.T
    scale = 1.0 / np.sqrt(np.diag(corr))
    corr = corr * scale[:, None] * scale[None, :]

    def loglik(nu: float) -> float:
        x = stats.t.ppf(u, nu)
        return float(np.sum(stats.multivariate_t(loc=np.zeros(d), shape=corr, df=nu).logpdf(x)) - np.sum(stats.t.logpdf(x, nu)))

    profile = [loglik(nu) for nu in PIVOT_NU_GRID]
    z = stats.norm.ppf(u)
    gaussian = float(np.sum(stats.multivariate_normal(mean=np.zeros(d), cov=corr).logpdf(z)) - np.sum(stats.norm.logpdf(z)))
    best = int(np.argmax(profile))
    return {"corr": corr, "nu": PIVOT_NU_GRID[best], "loglik": profile[best], "gaussian_loglik": gaussian, "profile": list(zip(PIVOT_NU_GRID, profile))}


def t_copula_draws(corr: np.ndarray, nu: float, n: int, rng: np.random.Generator) -> np.ndarray:
    from scipy import stats

    z = rng.standard_normal((n, corr.shape[0])) @ np.linalg.cholesky(corr).T
    return stats.t.cdf(z / np.sqrt(rng.chisquare(nu, size=n) / nu)[:, None], nu)


def fleet_coincidence(u: np.ndarray, levels) -> np.ndarray:
    """C(u) = P(>= 2 components above u | >= 1), per level (nan without an exceedance)."""
    out = []
    for level in levels:
        count = np.sum(u > level, axis=1)
        any_ = np.count_nonzero(count >= 1)
        out.append(np.count_nonzero(count >= 2) / any_ if any_ else np.nan)
    return np.asarray(out)


def copula_pivot(cells: dict, rows: dict, t4: dict, rng: np.random.Generator) -> dict:
    """The T1-fail pivot: t copula per Gaussian cell, its T1 re-check, and the T4 trend."""
    from scipy.stats import spearmanr

    from tether.evt.dependence import pseudo_observations

    spacing = int(round(SAMPLE_SPACING))
    out = {}
    for name, cell in cells.items():
        if not name.startswith("G_"):
            continue
        x, seeds = _five_vectors(cell, spacing)
        u = np.column_stack([pseudo_observations(x[:, j]) for j in range(x.shape[1])])
        fit = fit_t_copula(u)
        draws = t_copula_draws(fit["corr"], fit["nu"], PIVOT_DRAWS, rng)
        fitted = fleet_coincidence(draws, PIVOT_U)
        empirical = fleet_coincidence(u, PIVOT_U)
        unique = np.unique(seeds)
        boot = []
        for _ in range(PIVOT_BOOT):
            chosen = rng.choice(unique, size=unique.size, replace=True)
            index = np.concatenate([np.flatnonzero(seeds == s) for s in chosen])
            boot.append(fleet_coincidence(u[index], PIVOT_U))
        lo, hi = np.nanpercentile(np.asarray(boot), [2.5, 97.5], axis=0)
        inside_c = bool(np.all((fitted >= lo) & (fitted <= hi)))
        trend = float(spearmanr(PIVOT_U, fitted).correlation)
        decreasing = bool(trend < -0.5 and fitted[-1] < fitted[0])
        recheck = []
        for key, pair in rows[name]["pairs"].items():
            if "chi" not in pair:
                continue
            i, j = int(key[0]), int(key[1])
            joint = np.array([np.mean((draws[:, i] > level) & (draws[:, j] > level)) for level in U_LEVELS])
            chi = joint / (1.0 - np.asarray(U_LEVELS))
            with np.errstate(divide="ignore"):
                chibar = 2.0 * np.log(1.0 - np.asarray(U_LEVELS)) / np.log(joint) - 1.0
            ok_chi = np.all((chi >= np.asarray(pair["chi_lo"])) & (chi <= np.asarray(pair["chi_hi"])))
            ok_chibar = np.all((chibar >= np.asarray(pair["chibar_lo"])) & (chibar <= np.asarray(pair["chibar_hi"])))
            recheck.append({"pair": key, "chi_99": float(chi[-1]), "chibar_99": float(chibar[-1]), "inside": bool(ok_chi and ok_chibar)})
        out[name] = {"samples": int(u.shape[0]), "nu": fit["nu"], "loglik": fit["loglik"], "gaussian_loglik": fit["gaussian_loglik"],
                     "profile": fit["profile"], "levels": list(PIVOT_U), "fitted_coincidence": fitted.tolist(), "empirical_coincidence": empirical.tolist(),
                     "empirical_lo": lo.tolist(), "empirical_hi": hi.tolist(), "inside": inside_c, "spearman": trend, "decreasing": decreasing,
                     "t4_powered": name in t4, "pair_recheck": recheck,
                     "pairs_inside_fraction": float(np.mean([r["inside"] for r in recheck])) if recheck else None}
    judged = [v for v in out.values() if v["t4_powered"]]
    reproduces = bool(judged) and all(v["decreasing"] and v["inside"] for v in judged)
    return {"cells": out, "judged_cells": [n for n, v in out.items() if v["t4_powered"]], "reproduces_t4_gaussian_trend": reproduces,
            "pairs_inside_fraction": float(np.mean([r["inside"] for v in out.values() for r in v["pair_recheck"]])) if out else None}


def analyse() -> dict:
    from scipy.stats import spearmanr

    from tether.evt.dependence import (
        dependence_bootstrap,
        eta_ledford_tawn,
        gaussian_copula_chi,
        gaussian_copula_chibar,
        pseudo_observations,
        spearman_to_pearson,
    )
    from tether.evt.proportions import wilson_interval

    predictions = json.loads(PREDICTIONS_PATH.read_text())
    jobs, summaries = load_cache(CACHE_PATH)
    stat: dict[str, list] = {}
    pilot: dict[str, list] = {}
    for job, summary in zip(jobs, summaries):
        (pilot if job.pilot else stat).setdefault(job.cell, []).append(summary)
    cells = {n: _pool(v) for n, v in stat.items()}
    pilots = {n: _pool(v) for n, v in pilot.items()}
    rng = np.random.default_rng(20260913)
    spacing = int(round(SAMPLE_SPACING))
    block = int(round(BLOCK_BOOTSTRAP / SAMPLE_SPACING))
    rows: dict = {}
    for name, cell in cells.items():
        pairs = {}
        for i, j in itertools.combinations(range(constants.VESSEL_COUNT), 2):
            x, y, _ = _pair_series(cell, i, j, spacing)
            if x.size < 200:
                pairs[f"{i}{j}"] = {"n": int(x.size), "status": "UNDER-POWERED"}
                continue
            u, v = pseudo_observations(x), pseudo_observations(y)
            rho_s = float(spearmanr(x, y).correlation)
            rho = spearman_to_pearson(rho_s)
            boot = dependence_bootstrap(u, v, np.array(U_LEVELS), block, n_boot=300, rng=rng)
            gauss_chi = np.array([gaussian_copula_chi(level, rho) for level in U_LEVELS])
            gauss_chibar = np.array([gaussian_copula_chibar(level, rho) for level in U_LEVELS])
            inside_chi = (gauss_chi >= np.asarray(boot["chi_lo"])) & (gauss_chi <= np.asarray(boot["chi_hi"]))
            inside_chibar = (gauss_chibar >= np.asarray(boot["chibar_lo"])) & (gauss_chibar <= np.asarray(boot["chibar_hi"]))
            eta, eta_lo, eta_hi = eta_ledford_tawn(u, v)
            pairs[f"{i}{j}"] = {
                "n": int(x.size), "rho": rho, "rho_spearman": rho_s,
                "chi": np.asarray(boot["chi"]).tolist(), "chi_lo": np.asarray(boot["chi_lo"]).tolist(), "chi_hi": np.asarray(boot["chi_hi"]).tolist(),
                "chibar": np.asarray(boot["chibar"]).tolist(), "chibar_lo": np.asarray(boot["chibar_lo"]).tolist(), "chibar_hi": np.asarray(boot["chibar_hi"]).tolist(),
                "gauss_chi": gauss_chi.tolist(), "gauss_chibar": gauss_chibar.tolist(),
                "inside": bool(np.all(inside_chi) and np.all(inside_chibar)),
                "eta": eta, "eta_lo": eta_lo, "eta_hi": eta_hi,
                "eta_contains_gaussian": bool(eta_lo <= (1 + rho) / 2 <= eta_hi), "eta_excludes_one": bool(eta_hi < 1.0),
            }
        block_pairs = {}
        for i, j in itertools.combinations(range(constants.VESSEL_COUNT), 2):
            bx = np.concatenate([b[:, i] for b in cell["block_maxima"]])
            by = np.concatenate([b[:, j] for b in cell["block_maxima"]])
            keep = (bx > 0) & (by > 0)
            if keep.sum() < 200:
                continue
            u, v = pseudo_observations(bx[keep]), pseudo_observations(by[keep])
            rho = spearman_to_pearson(float(spearmanr(bx[keep], by[keep]).correlation))
            joint = np.array([np.mean((u > level) & (v > level)) / (1 - level) for level in U_LEVELS])
            block_pairs[f"{i}{j}"] = {"rho": rho, "chi": joint.tolist(), "gauss_chi": [gaussian_copula_chi(level, rho) for level in U_LEVELS]}
        pilot_marks = pilots[name]["marks"]["T_peak"]
        grid = [float(np.percentile(pilot_marks, q)) for q in QUANTILES] if pilot_marks.size >= 10 else []
        coincidence_rows = []
        for threshold in grid:
            episodes, multi = coincidence(cell, threshold)
            fraction, lo, hi = wilson_interval(multi, episodes) if episodes else (None, None, None)
            coincidence_rows.append({"threshold": threshold, "episodes": episodes, "multi": multi, "fraction": fraction, "lo": lo, "hi": hi, "powered": episodes >= POWER_MIN})
        side = None
        if name.endswith("broadside_port") or name.endswith("broadside_starboard"):
            windward = (3, 4) if name.endswith("broadside_port") else (0, 1)
            leeward = (0, 1) if name.endswith("broadside_port") else (3, 4)
            threshold = grid[1] if len(grid) > 1 else None
            if threshold is not None:
                marks = cell["marks"]
                peaks = cell["taut_peaks"]

                def snap_fraction(cables):
                    snaps = int(np.sum(np.isin(marks["cable"], cables) & (marks["T_peak"] > threshold)))
                    tauts = int(np.sum(np.isin(peaks["cable"], cables) & (peaks["peak"] > threshold)))
                    return (snaps / (snaps + tauts) if snaps + tauts else None), snaps + tauts

                w, nw = snap_fraction(windward)
                l, nl = snap_fraction(leeward)
                side = {"threshold": threshold, "windward_snap_fraction": w, "windward_events": nw, "leeward_snap_fraction": l, "leeward_events": nl,
                        "difference": (abs(w - l) if w is not None and l is not None else None)}
        rows[name] = {"pairs": pairs, "block_maxima": block_pairs, "grid": grid, "coincidence": coincidence_rows, "side": side,
                      "excursions": int(cell["marks"]["t_up"].size), "closures": int(sum(c is not None for c in cell["closures"]))}
    tests = {}
    gaussian_cells = [rows[n] for n in rows if n.startswith("G_")]
    pair_rows = [p for r in gaussian_cells for p in r["pairs"].values() if "inside" in p]
    inside_fraction = float(np.mean([p["inside"] for p in pair_rows])) if pair_rows else None
    eta_ok = float(np.mean([p["eta_contains_gaussian"] and p["eta_excludes_one"] for p in pair_rows])) if pair_rows else None
    tests["P4-T1"] = {"statement": "instantaneous chi, chibar inside the band of the Gaussian copula at rho for >= 80% of pairs (Gaussian cells); eta interval containing (1+rho)/2 and excluding 1",
                      "pairs": len(pair_rows), "fraction_inside": inside_fraction, "fraction_eta_ok": eta_ok,
                      "verdict": "UNDER-POWERED" if not pair_rows else ("PASS" if inside_fraction >= 0.8 and eta_ok >= 0.8 else "FAIL")}
    tests["P4-T1b"] = {"statement": "block-maxima copula against the Gaussian curve at the block-max correlation (EMP, non-blocking)",
                       "cells": {n: r["block_maxima"] for n, r in rows.items() if n.startswith("G_")}, "verdict": "REPORTED"}
    t2_rows = []
    for front in FRONTS:
        row = rows[cell_name("student_t3", front)]
        good = [k for k, p in row["pairs"].items() if "inside" in p and p["chibar_hi"][-1] >= 1.0 and (p["chi"][-1] - p["gauss_chi"][-1]) > (p["chi_hi"][-1] - p["chi_lo"][-1])]
        t2_rows.append({"front": front, "pairs_meeting": good})
    tests["P4-T2"] = {"statement": "common t3: chibar(0.99) interval containing 1 and chi(0.99) above the Gaussian value by more than the band width, for >= 1 pair per front",
                      "rows": t2_rows, "verdict": "PASS" if all(r["pairs_meeting"] for r in t2_rows) else "FAIL"}
    t3_rows = []
    for front in FRONTS:
        row = rows[cell_name("student_t3", front)]
        keys = [k for k, p in row["pairs"].items() if "chi" in p]
        predicted = predictions["fronts"][front]["pair_joint_radius"]
        if len(keys) >= 4:
            measured = [row["pairs"][k]["chi"][-1] for k in keys]
            radius = [-predicted[k] if np.isfinite(predicted[k]) else -1e9 for k in keys]
            t3_rows.append({"front": front, "spearman": float(spearmanr(measured, radius).correlation), "pairs": keys})
    tests["P4-T3"] = {"statement": "Spearman of chi(0.99) rank against the predicted severance-radius overlap rank above 0.7", "rows": t3_rows,
                      "verdict": "UNDER-POWERED" if not t3_rows else ("PASS" if all(r["spearman"] > 0.7 for r in t3_rows) else "FAIL")}
    t4 = {}
    for name, row in rows.items():
        powered = [c for c in row["coincidence"] if c["powered"]]
        if len(powered) >= 2:
            rho = spearmanr([c["threshold"] for c in powered], [c["fraction"] for c in powered]).correlation
            t4[name] = {"points": powered, "spearman": float(rho) if np.isfinite(rho) else None,
                        "top_below_bottom": bool(powered[-1]["hi"] < powered[0]["fraction"])}
    gaussian_ok = [v["spearman"] is not None and v["spearman"] < -0.5 and v["top_below_bottom"] for n, v in t4.items() if n.startswith("G_")]
    tests["P4-T4"] = {"statement": "Gaussian: coincidence decreasing with threshold (Spearman < -0.5, top interval below bottom estimate); common t3: at or increasing, top within factor 2 of the max-functional",
                      "cells": t4, "max_functional": {f: predictions["fronts"][f]["max_functional_coincidence"] for f in FRONTS},
                      "gaussian_verdict": "UNDER-POWERED" if not gaussian_ok else ("PASS" if all(gaussian_ok) else "FAIL")}
    t3_front_ok = []
    for front in FRONTS:
        v = t4.get(cell_name("student_t3", front))
        mf = predictions["fronts"][front]["max_functional_coincidence"]
        if v and v["spearman"] is not None and mf:
            t3_front_ok.append(v["spearman"] >= 0 and 0.5 <= v["points"][-1]["fraction"] / mf <= 2.0 if v["points"][-1]["fraction"] else False)
    tests["P4-T4"]["common_t3_verdict"] = "UNDER-POWERED" if not t3_front_ok else ("PASS" if all(t3_front_ok) else "FAIL")
    tests["P4-T4"]["verdict"] = tests["P4-T4"]["gaussian_verdict"]
    sides = {n: r["side"] for n, r in rows.items() if r["side"]}
    tests["P4-T5"] = {"statement": "under a broadside front the snap fraction on windward and leeward cables differs by > 0.3", "cells": sides,
                      "verdict": "UNDER-POWERED" if not sides else ("PASS" if all(s["difference"] is not None and s["difference"] > 0.3 for s in sides.values()) else "FAIL")}
    t1 = tests["P4-T1"]["verdict"]
    rule = ("GO requires T1 and the Gaussian half of T4. T1 fail: PIVOT to an empirically fitted copula, which must reproduce the Gaussian-half trend of T4; "
            "if it does, GO with the Phase-4 Gaussian claim retagged EMP; if not, NO-GO pending a diagnosis on the yaw channel")
    if t1 == "PASS" and tests["P4-T4"]["gaussian_verdict"] == "PASS":
        gate = {"decision": "GO", "rule": rule}
    elif t1 == "FAIL":
        pivot = copula_pivot(cells, rows, t4, rng)
        tests["P4-pivot"] = {"statement": "T1-fail pivot: fitted t copula reproduces the Gaussian-half trend of T4 (declarations: pivot)", **pivot,
                             "verdict": "PASS" if pivot["reproduces_t4_gaussian_trend"] else "FAIL"}
        gate = {"decision": "GO (PIVOT: Gaussian claim retagged EMP)" if pivot["reproduces_t4_gaussian_trend"] else "NO-GO (pending a diagnosis on the yaw channel)", "rule": rule}
    else:
        gate = {"decision": "NOT-GO", "rule": rule}
    results = {"schema_version": 1, "source": source_state(), "declarations": DECLARATIONS, "predictions_sha256": sha256_file(PREDICTIONS_PATH),
               "rows": rows, "tests": tests, "gate": gate,
               "execution": {n: {"wall_seconds": c["wall_seconds"], "sim_seconds": c["sim_seconds"]} for n, c in cells.items()}}
    names = sorted(cells)
    arrays = {"cell_names": np.array(names), "marks_cell_index": np.concatenate([np.full(cells[n]["marks"]["t_up"].size, i) for i, n in enumerate(names)])}
    for key in cells[names[0]]["marks"]:
        arrays[f"marks_{key}"] = np.concatenate([cells[n]["marks"][key] for n in names])
    results["record_sha256"] = write_bytes(RECORD_PATH, npz_bytes(arrays))
    results["probe"] = {"cell": jobs[0].cell, "seed": jobs[0].seed, "marks_t_up": summaries[0]["marks"]["t_up"].tolist()}
    write_bytes(RESULTS_PATH, json_bytes(results))
    write_bytes(MANIFEST_PATH, json_bytes({"schema_version": 1, "driver": "python -m tether.campaign.phase4 shocks|predict|compute|analyse", "record_sha256": results["record_sha256"],
                                           "committed_record_path": relative(RECORD_PATH), "predictions_sha256": results["predictions_sha256"], "probe": results["probe"]}))
    return results


def replay() -> bool:
    results = json.loads(RESULTS_PATH.read_text())
    probe = results["probe"]
    summary = run_stationary_job(StationaryJob(probe["cell"], cell_specs()[probe["cell"]], probe["seed"]))
    ok = summary["marks"]["t_up"].tolist() == probe["marks_t_up"]
    print(json.dumps({"marks_exact": ok}))
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("shocks", "predict", "compute", "analyse", "replay"))
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    if arguments.command == "shocks":
        results = run_pool(run_front_shock, front_shock_jobs(), arguments.workers)
        SHOCK_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with SHOCK_CACHE.open("wb") as handle:
            pickle.dump(results, handle)
    elif arguments.command == "predict":
        print(json.dumps({f: v["max_functional_coincidence"] for f, v in predict()["fronts"].items()}))
    elif arguments.command == "compute":
        if not PREDICTIONS_PATH.exists():
            raise SystemExit("predictions must be committed before the campaign runs")
        compute_jobs(all_jobs(), CACHE_PATH, arguments.workers, "phase4")
    elif arguments.command == "analyse":
        results = analyse()
        for test_id, test in results["tests"].items():
            print(test_id, test["verdict"])
        print("GATE", results["gate"])
    else:
        sys.exit(0 if replay() else 1)


if __name__ == "__main__":
    main()
