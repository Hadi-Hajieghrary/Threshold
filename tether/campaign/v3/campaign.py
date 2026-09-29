"""Plan v3 pre-registration: declarations, committed predictions, addenda, gate records.

Order of use (Makefile ``v3-declare``):

    python -m tether.campaign.v3.campaign predict    # records/v3/cascade_predictions.json
    python -m tether.campaign.v3.campaign declare    # records/v3/cascade_declarations.json

``predict`` computes every number the campaign commits BEFORE any plant run, from the theory
modules and from inputs that already exist (the v1 cascade-grid record, the reduced LTI
model).  Every such number is labelled FORECAST.  ``declare`` hashes the plan document, the
theory document, the predictions, the inputs and the code, and refuses to overwrite; later
changes are dated addenda (``addendum``), never edits.  ``verify_declared`` is called by the
compute and analysis steps, which refuse to run against drifted declarations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tether.campaign import phase2
from tether.campaign.common import RECORDS, ROOT, json_bytes, sha256_bytes, sha256_file, source_state, write_bytes
from tether.campaign.v2 import events as ev
from tether.campaign.v3 import cells
from tether.physics import constants
from tether.theory import branching as B
from tether.theory import reduced_lti as R
from tether.theory import transmission as TR

PLAN_PATH = ROOT / "ref" / "cascade_plan_v3.md"
THEORY_PATH = ROOT / "reports" / "v3" / "theory_v3.md"
RECORD_DIR = RECORDS / "v3"
PREDICTIONS_PATH = RECORD_DIR / "cascade_predictions.json"
DECLARATIONS_PATH = RECORD_DIR / "cascade_declarations.json"
V1_RECORD = RECORDS / "phase2" / "phase2_records.npz"
V1_CACHE = phase2.CACHE_PATH
V2_PREDICTIONS = RECORDS / "v2" / "phase2" / "phase2_predictions_i035.json"
MODULE_PATHS = {
    "theory/transmission": ROOT / "tether" / "theory" / "transmission.py",
    "theory/branching": ROOT / "tether" / "theory" / "branching.py",
    "theory/reduced_lti": ROOT / "tether" / "theory" / "reduced_lti.py",
    "evt/extremal": ROOT / "tether" / "evt" / "extremal.py",
    "campaign/v2/events": ROOT / "tether" / "campaign" / "v2" / "events.py",
    "campaign/v3/cells": ROOT / "tether" / "campaign" / "v3" / "cells.py",
    "campaign/v3/campaign": ROOT / "tether" / "campaign" / "v3" / "campaign.py",
    "campaign/v3/reducer": ROOT / "tether" / "campaign" / "v3" / "reducer.py",
    "campaign/v3/intervention": ROOT / "tether" / "campaign" / "v3" / "intervention.py",
    "campaign/v3/compute": ROOT / "tether" / "campaign" / "v3" / "compute.py",
    "campaign/fleet_run": ROOT / "tether" / "campaign" / "fleet_run.py",
}
SCHEMA = "v3-declarations-1"
FACTOR = 1.5  # the factor-of-1.5 agreement band used by every transfer test
SHARE = 0.8  # "in at least 80 % of scored cells"
MIN_PAIRS, MIN_OFFSPRING, MIN_PER_PAIR, MIN_SCORED_CELLS = 20, 20, 5, 5
BAND = (0.2, 0.9)  # the paper's declared band (plan v2 Prop. 13 clause iv), tested as the paper's claim
X_RANGE_PATTERN, X_RANGE_EXACT = (1.0, 4.0), (1.0, 2.3)
COUPLED_PEAK_BINS = (1.0, 1.4, 2.0, 3.0, 5.0, 8.0, 15.0)
OPERABLE_CLOSURE_FRACTION = 0.25
BOOTSTRAP_SEED = 20260917

# Where each forecast cell borrows its parent-peak law and primary counts from (v1 record cells).
FORECAST_SOURCE = {
    "FAN_T1000_I0.5": "T1000_k500_I0.5", "FAN_T1000_I1.0": "T1000_k500_I1.0",
    "FAN_T600_I0.5": "T600_k500_I0.5", "FAN_T600_I1.0": "T600_k500_I1.0",
    "T600_k500_I0.5_kx0.25": "T600_k500_I0.5", "T600_k500_I0.5_kx4": "T600_k500_I0.5",
    "T600_k500_I0.5_kx0.5": "T600_k500_I0.5", "T600_k500_I0.5_kx2": "T600_k500_I0.5",
    "T600_k500_I0.35": "T600_k500_I0.5", "T600_k500_I0.75": "T600_k500_I1.0",
    "T600_k500_I1.25": "T600_k500_I1.0", "T600_k500_I1.5": "T600_k500_I1.0",
    "T600_k500_I1.75": "T600_k500_I1.0", "T600_k500_I2.0": "T600_k500_I1.0",
}


def _branch(pass_to: str, fail_to: str) -> dict:
    return {"PASS": pass_to, "FAIL": fail_to, "UNDER-POWERED": "the FAIL branch (blocking tests) or reported as under-powered"}


TESTS = {
    "T1.0": {
        "blocking": True, "work_package": 1,
        "statement": "the ten reused v1 cells reproduce the v1 cascade-grid marks under common random numbers",
        "threshold": "t_up and T_peak arrays bit-equal in all 220 reused runs",
        "branches": _branch("v1 counts are comparable with v3's", "v3 re-baselines on its own marks; the drift is reported with the first differing run; no comparison with v1 counts in the paper"),
    },
    "T1.1": {
        "blocking": True, "work_package": 1,
        "statement": "the paper's transmission band: the median normalised neighbour drop lies in [0.2, 0.9]",
        "definition": "drop_norm = drop / (T_peak * geometry_factor_ij); parents = event heads with x = T_peak/T0 in [1, 4] and every neighbour taut at t_up; cell scored with >= 20 eligible pairs; mirror pairs pooled",
        "threshold": f"median drop_norm in [{BAND[0]}, {BAND[1]}] in >= {int(SHARE * 100)} % of scored cells",
        "committed_prediction": "FAIL: the exact linear response predicts 0.14-0.18 (cascade_predictions.json per cell), below the band's lower edge; the closed form 0.44 rests on an impulsive pulse and pinned vessel ends, neither of which holds on this plant",
        "branches": _branch("the paper's closed-form band stands", "the paper's quantitative claim is corrected to the exact-response value; the mechanism is judged by T1.2 and T1.3, not by this band"),
    },
    "T1.2": {
        "blocking": True, "work_package": 1,
        "statement": "pair pattern: the measured median drop per ordered pair follows the exact-response matrix R",
        "threshold": f"Spearman(median drop per ordered pair, R_ij) >= 0.7 in >= {int(SHARE * 100)} % of scored cells (>= {MIN_PER_PAIR} eligible parents per pooled pair); the closed-form operator reported beside it",
        "committed_prediction": "PASS: the pair pattern of R follows the Delassus column, cos(sigma_i - sigma_j) in the fan formation and the lever-arm (yaw) terms in the parallel formation, which the plant's geometry fixes",
        "branches": _branch("the geometry factor survives; parallel -> fan transfer is meaningful", "geometry falsified; WP2 uses the measured per-cell R_hat; the transfer claim is dropped"),
    },
    "T1.3": {
        "blocking": True, "work_package": 1,
        "statement": "exact-response accuracy for small-peak parents (x in [1, 2.3], all neighbours taut)",
        "threshold": f"median drop / (T_peak * R_ij) in [1/{FACTOR}, {FACTOR}] in >= {int(SHARE * 100)} % of scored cells",
        "committed_prediction": "PASS at the linear level; the plant's unilateral nonlinearity (neighbours that go slack) makes the measured drop saturate at the neighbour's margin, which the eligibility rule removes",
        "branches": _branch("the exact response is the quantitative law; WP2's mechanistic kernel uses R", "the exact response is descriptive only; WP2's kernel uses R_hat (if T1.2 passed) or the empirical kernel"),
    },
    "T1.4": {
        "blocking": False, "work_package": 1,
        "statement": "stiffness dependence of the normalised drop between k x 0.25 and k x 4 at the calibration cell",
        "threshold": "Delta_meas = median(k x 4) - median(k x 0.25) of drop_norm over the eligible rows with a seed-bootstrap 95 % interval; Delta_exact is the same statistic applied to the predicted per-row value R_ij / geometry_factor_ij over the same rows (the pair mix of the scored rows; the uniform-pair value 0.045 and the per-pair range are committed in cascade_predictions.json); PASS-exact if the interval contains Delta_exact and excludes 0; PASS-closed if it contains 0 and excludes Delta_exact; UNDER-POWERED if it contains both; FAIL-both if neither",
        "committed_prediction": "the closed form predicts Delta = 0 (stiffness-free); the exact response predicts a positive Delta_exact, +0.045 for a uniform pair mix and 0.044-0.071 pair by pair (cascade_predictions.json), through the damping ratio and the drag",
        "branches": {"PASS-exact": "the paper credits the exact response", "PASS-closed": "the paper credits the closed form's stiffness independence", "UNDER-POWERED": "reported as not resolved", "FAIL-both": "reported as a failure of both"},
    },
    "T2.1": {
        "blocking": True, "work_package": 2,
        "statement": "m_x is monotone in the parent peak and vanishes for small peaks",
        "threshold": f"binned m_x over the peak bins {list(COUPLED_PEAK_BINS)} T0 (>= 10 events per bin) has Spearman >= 0.8 against bin order in >= {int(SHARE * 100)} % of scored cells, and the fitted value in the first bin is <= 0.1",
        "branches": _branch("T2.2 is scored", "Claim B not established; the kernel is empirical per pair"),
    },
    "T2.2": {
        "blocking": True, "work_package": 2,
        "statement": "transfer of the isotonic m_x fitted on the calibration cell's pilot seeds",
        "threshold": f"predicted / measured cross-cable offspring in [1/{FACTOR}, {FACTOR}] in >= {int(SHARE * 100)} % of cells with >= {MIN_OFFSPRING} offspring; fan cells reported as a subgroup",
        "branches": _branch("Claim B established", "fan fails and parallel passes: the projection is falsified and the impulse survives; otherwise Claim B is withdrawn"),
    },
    "T2.3": {
        "blocking": True, "work_package": 2,
        "statement": "intervention validation: the un-intervened counterfactual reproduces the record",
        "threshold": "for every sampled parent |t_cf - t_up| <= 5 ms, |v_cf - v_return| p95 <= 0.01 and max <= 0.03 m/s, T_peak within 2 % (p95); the factual other-cable onsets in the window reproduced (same cables, times within 20 ms) in >= 95 % of parents",
        "branches": _branch("causal offspring counts are trusted", "causal counts are reported as untrusted; T2.4 is not scored; the paper uses the time-shift excess only"),
    },
    "T2.4": {
        "blocking": True, "conditional_on": "T2.3", "work_package": 2,
        "statement": "causal (interventional) offspring against the observational time-shift excess",
        "threshold": f"both sides are counts of other-cable onsets within 3 s of the sampled parents: causal = factual minus counterfactual, observational excess = measured minus the circular time-shift null of the parent time (200 shifts, rng 20260914, span from 20 s to the run's end); causal / excess in [1/{FACTOR}, {FACTOR}] in >= {int(SHARE * 100)} % of cells with >= {MIN_OFFSPRING} causal offspring",
        "branches": _branch("observational attribution is unbiased", "the kernel is calibrated on causal offspring where available, and the paper says so"),
    },
    "T2.5": {
        "blocking": True, "work_package": 2,
        "statement": "the parameter-free kernel K_ij(T) = P(margin_i < R_ij * phi * T) predicts the offspring counts",
        "threshold": f"predicted sum over the statistics-seed events E and neighbours i of P(margin_i < R_ij * phi * T_E) / measured cross-cable offspring in [1/{FACTOR}, {FACTOR}] in >= {int(SHARE * 100)} % of cells with >= {MIN_OFFSPRING} offspring; phi = 1 (half sine), pi/2 reported; margin law = the pilot seeds' clean-set tension quantiles (the only fitted input)",
        "branches": _branch("WP3 uses the mechanistic kernel", "fallback 1: R_hat measured in WP1; fallback 2: the empirical kernel; if none passes, records/v3/wp3_not_launched.json"),
    },
    "T3.1": {
        "blocking": True, "work_package": 3,
        "statement": "the cluster-corrected rate law nu_total = (I - K)^-1 nu_primary predicts the measured total onset rate",
        "threshold": f"K with the pilot-seed margin law and the statistics-seed peak average, nu_primary measured on the statistics seeds (the primary events per cable over the primary exposure); pooled-over-cables ratio predicted/measured in [1/{FACTOR}, {FACTOR}] in >= {int(SHARE * 100)} % of cells with >= {MIN_OFFSPRING} offspring; per cable reported",
        "branches": _branch("the branching description is quantitative", "descriptive only"),
    },
    "T3.2": {
        "blocking": True, "work_package": 3,
        "statement": "theta predicted from K equals the measured primary share theta_runs",
        "threshold": f"theta_pred = 1 / (1^T (I - K)^-1 pi) with K as in T3.1 and pi the statistics-seed primary type distribution; |theta_pred - theta_runs| <= 0.10 in >= {int(SHARE * 100)} % of scored cells; the Ferro-Segers theta reported with its interval",
        "branches": _branch("quantitative", "descriptive only"),
    },
    "T3.3": {
        "blocking": False, "work_package": 3,
        "statement": "primary onset rate against theta x Rice from the reduced LTI model (report only)",
        "threshold": "none; measured theta_primary per cell; v2's committed theta shown as prior (different cells)",
        "branches": {"REPORT": "reported"},
    },
    "T3.4": {
        "blocking": False, "work_package": 3,
        "statement": "location of rho(K) = 1 along the T0 = 0.6 kN intensity sweep",
        "threshold": "with CI the seed-bootstrap 95 % interval of rho: (a) an operable cell whose CI lies entirely above 1 and a lower-intensity operable cell whose CI lies entirely below 1: transition observed, I_c by log-linear interpolation of rho between them; (b) no such cell above and the CI at the highest operable intensity lies entirely below 1: margin result, margin = 1 - rho there; (c) otherwise (a straddling interval, or no operable cell): UNDER-POWERED, B_ext fires once if its rule allows, then (a)/(b)/(c) final",
        "branches": {"(a)": "the paper reports the critical intensity", "(b)": "the paper reports the margin 1 - rho", "(c)": "reported as unresolved after the extension"},
    },
    "T3.5": {
        "blocking": False, "work_package": 3,
        "statement": "cluster-size law against Borel-Tanner(rho) and the multitype Monte Carlo (report only)",
        "threshold": "none; KS distance and tail exponent reported per sweep cell",
        "branches": {"REPORT": "reported"},
    },
}

KERNEL_PATH_RULE = [
    "T1.3 PASS -> K from the exact-response matrix R and the measured margin law (mechanistic, no fitted parameter)",
    "else T1.2 PASS -> K from the measured per-cell R_hat and the measured margin law",
    "else -> the empirical kernel (mean offspring per ordered pair on pilot seeds)",
    "the first kernel on this list that passes its factor-1.5 test (T2.5 or its fallback) is carried into WP3; if none passes, WP3 is not launched",
]

DEFINITIONS = {
    "mark": "one slack excursion of a cable from the onset of slack to the re-engagement that ends it, with that re-engagement's peak tension (tether.physics.cable.CableEventTracker)",
    "event": "marks of one cable grouped by the anchored rule: a mark joins the cable's open event while t_up - t_event <= 2 s (tether.campaign.v2.events.decluster_marks, rule='anchored'); the parent mark is the event's first re-engagement; its peak is the parent's T_peak (cluster maximum as sensitivity)",
    "onset": "a geometric down-crossing of a live cable (tether.campaign.v2.events.crossings)",
    "offspring": "an onset whose parent re-engagement (the latest re-engagement on any cable in (t - 3 s, t], tether.campaign.v2.events.primary_onsets) belongs to event E on a different cable. The tree is the cross-cable tree: an event is primary iff its own onset has no parent re-engagement on a different cable; an event whose only parent is a re-engagement of its own cable (2-3 s earlier, outside the 2 s burst) counts as primary for theta and the rate law, and the share of such events is reported",
    "m_x": "number of cross-cable offspring of an event as a function of x = T_peak / T0",
    "neighbour swing": "drop_i = T_i(t_up) - min over [t_up, t_up + P] of T_i(t), P the cell's engagement period 2 pi sqrt(mu / k); tension recomputed as max(0, k e + c edot) where e > 0 and alive",
    "eligible neighbour": "taut (e_i > 0) and alive at the first 1 ms sample at or after t_up",
    "geometry_factor": "T_ij / 0.440 with T the closed-form operator (2k/(omega_L omega_e)) W^T M_L^-1 W at the measured chord angles and load yaw",
    "R": "the exact-response matrix: drop of neighbour i per unit peak on j from the closed-loop taut tow linearised about the steady tow, cable j's spring removed, half-sine pulse of duration pi/omega_e (tether.theory.transmission.response_matrix)",
    "margin": "a cable's tension at a random taut instant of the B.3 clean set (pilot seeds), i.e. how much it can be unloaded before going slack",
    "rho": "spectral radius of the peak-averaged mean offspring matrix K on the type space of cables: the margin law (the only fitted input) from the pilot seeds' clean set, the average over the parent peaks of the statistics-seed events; seed-bootstrap interval over the statistics seeds. Averaging over peaks assumes the mean-offspring hypothesis (B4) of reports/v3/theory_v3.md, i.e. that the peak of an offspring is independent of its parent's peak",
    "theta_runs": "primary events / all events on statistics seeds (cross-cable tree); theta_pred = 1 / (1^T (I - K)^-1 pi) with rows = children; theta_FS = Ferro-Segers intervals estimator on the 1 ms fleet onset indicator",
    "operable cell": f"formation closures in <= {OPERABLE_CLOSURE_FRACTION:.0%} of its 22 runs",
    "scored cell": f"a cell holding >= {MIN_PAIRS} eligible pairs (WP1) or >= {MIN_OFFSPRING} cross-cable offspring (WP2/WP3); fewer than {MIN_SCORED_CELLS} scored cells makes a test UNDER-POWERED",
    "identity": "nu_total = nu_primary / (1 - m_bar) holds by construction on any record and is printed, never scored",
}

CONTINGENCY_RULE = (
    "the compute budget is 60 core-hours; stages A and B use about 40; the remainder may add up to 10 extra "
    "statistics seeds (2023-2032) to at most 3 cells whose blocking test is UNDER-POWERED, declared by addendum before the runs"
)

DECLARATIONS = {
    "schema": SCHEMA,
    "plan": "ref/cascade_plan_v3.md",
    "theory": "reports/v3/theory_v3.md",
    "rule": "plan v2 III.3 honoured: v3 re-opens no v2 gate and re-scores no v2 test; v2's forecasts are cited by hash as prior predictions from a reduced LTI model at intensity 0.35, never as results",
    "calibration_cell": {
        "cell": cells.CALIBRATION_CELL, "fit_seeds": list(phase2.PILOT_SEEDS),
        "ruling": "plan v2 II.7(ii) names T0 = 0.6 kN, intensity 0.5, parallel, fitted on pilot seeds (the cell holds 2729 marks on its 20 statistics seeds and 240 on its two pilots in records/phase2, 40 % above 2.3 T0); P4-T6 names T0 = 1 kN on statistics seeds (89 marks on 13 seeds, no isotonic fit possible on two pilots, and it would break the pilot/statistics separation); v3 adopts II.7(ii) and records the P4-T6 wording as an inconsistency",
    },
    "definitions": DEFINITIONS,
    "seeds": {"pilot": list(phase2.PILOT_SEEDS), "statistics": list(phase2.STAT_SEEDS), "common_random_numbers": True},
    "run": {"duration_s": phase2.DURATION, "warmup_s": phase2.WARMUP, "weather": "gaussian local AR(1), 8 s memory", "cable_mode": "recording", "physics_step_s": 5.0e-4},
    "stages": {
        "A": "16 cells (WP1, WP2): ten reused v1 cells, four fan cells, the stiffness pair",
        "B": "4 cells (WP3 sweep at T0 = 0.6 kN, k_h = 500): intensities 0.35, 0.75, 1.25, 1.5; launched only with gate_wp2 OPEN",
        "B_ext": "2 cells (1.75, 2.0): launched only if the closure fraction at 1.5 is <= 0.25 and T3.4 has fired no branch",
    },
    "stiffness_pair": {"factors": dict(cells.STIFFNESS_FACTORS), "damping": "unchanged (the damping ratio changes; declared)",
                       "why_4x": "Delta_exact between k x 0.5 and k x 2 is 0.023 and between k x 0.25 and k x 4 is 0.045 (predicted); the wider pair is chosen for power; the physics step resolves the 37 rad/s cable mode at k x 4 with 335 steps per period"},
    "eligibility": {"x_range_pattern_and_band": list(X_RANGE_PATTERN), "x_range_exact": list(X_RANGE_EXACT), "min_pairs": MIN_PAIRS, "min_per_pair": MIN_PER_PAIR, "min_offspring": MIN_OFFSPRING, "min_scored_cells": MIN_SCORED_CELLS, "mirror_pairs_pooled": "(i, j) ~ (4 - i, 4 - j)"},
    "intervention": {"max_parents_per_run": 40, "selection": "every event with x > 2.3 by descending x, then a systematic sample of the rest, up to the cap", "horizon_s": 4.0, "window_s": 3.0, "headings": "prescribed from the record (p6_t0 addendum-2 rule); in the disabled branch the parent vessel's heading is held from t_up on; 'held' (all headings frozen at t_up) is the declared sensitivity", "counterfactual": "cable j applies no force from t_up on (alive = 0)"},
    "tests": TESTS,
    "kernel_path_rule": KERNEL_PATH_RULE,
    "operable_rule": f"closure fraction <= {OPERABLE_CLOSURE_FRACTION}",
    "contingency_rule": CONTINGENCY_RULE,
    "bootstrap": {"seed": BOOTSTRAP_SEED, "n_boot": 2000, "resampling": "seeds"},
    "forecast_sources": FORECAST_SOURCE,
    "audit": "each work package: unit tests, run, independent adversarial re-derivation from the records by an agent that reads the producing code, fixes logged as a dated round in reports/v3/corrections_log.md with a scoped re-check, then the gate record",
}


# ----------------------------------------------------------------------------- forecast inputs


def _v1_margin_quantiles(cache_path: Path = V1_CACHE, levels: int = 101, min_samples: int = 50) -> dict[str, np.ndarray]:
    """Empirical taut-tension quantiles per cable and v1 cell from the cache's 1 Hz samples (statistics seeds).

    The v1 summaries hold ``samples["q"]`` (600 x N, one tension sample per second) and
    ``samples["taut"]``; the margin law is the tension at taut instants.  Cells whose cable has
    fewer than ``min_samples`` taut samples get nan rows (the forecast then falls back to the LTI law).
    """
    import pickle

    with Path(cache_path).open("rb") as handle:
        payload = pickle.load(handle)
    pooled: dict[str, list] = {}
    for job, summary in zip(payload["jobs"], payload["summaries"]):
        if job.pilot:
            continue
        q = np.asarray(summary["samples"]["q"], dtype=float)
        taut = np.asarray(summary["samples"]["taut"], dtype=bool)
        pooled.setdefault(job.cell, []).append(np.where(taut, q, np.nan))
    out = {}
    grid = np.linspace(0.0, 100.0, levels)
    for cell, rows in pooled.items():
        stacked = np.concatenate(rows, axis=0)
        quantiles = np.full((stacked.shape[1], levels), np.nan)
        for i in range(stacked.shape[1]):
            column = stacked[:, i][np.isfinite(stacked[:, i])]
            if column.size >= min_samples:
                quantiles[i] = np.percentile(column, grid)
        out[cell] = quantiles
    return out


def _v1_cell_inputs(record_path: Path = V1_RECORD) -> dict[str, dict]:
    """Event heads (parents), primaries per cable and exposure of every v1 cell on statistics seeds."""
    d = np.load(record_path, allow_pickle=True)
    names = list(d["cell_names"])
    margins = _v1_margin_quantiles()
    out = {}
    for index, name in enumerate(names):
        m = (d["marks_cell_index"] == index) & (d["marks_seed"] >= min(phase2.STAT_SEEDS))
        if not m.any():
            continue
        t_up, t_down = d["marks_t_up"][m], d["marks_t_down"][m]
        cable, peak, seed = d["marks_cable"][m], d["marks_T_peak"][m], d["marks_seed"][m]
        parent_peak, parent_cable, primaries = [], [], np.zeros(5)
        events = 0
        for s in np.unique(seed):
            k = seed == s
            dec = ev.decluster_marks(t_up[k], cable[k], rule="anchored")
            heads = dec.role == ev.ROLE_CODES[ev.ROLE_PARENT]
            events += int(heads.sum())
            parent_peak.extend(peak[k][heads].tolist())
            parent_cable.extend(cable[k][heads].tolist())
            po = ev.primary_onsets(t_down[k][heads], cable[k][heads], t_up[k], cable[k], window=ev.EXCLUSION_WINDOW)
            for c in cable[k][heads][po.primary]:
                primaries[int(c)] += 1
        out[name] = {
            "parent_peak": np.asarray(parent_peak, dtype=float), "parent_cable": np.asarray(parent_cable, dtype=np.int64),
            "primaries": primaries, "events": events, "exposure_s": float(d["cell_exposure"][index]),
            "seeds": int(np.unique(seed).size),
            "margin_quantiles": margins.get(name),
        }
    return out


def _margin_quantiles_lti(inp: R.ReducedModelInput, pretension: float) -> tuple[np.ndarray, np.ndarray]:
    """FORECAST margin law: Gaussian N(T0, sigma_q) per cable from the reduced LTI model."""
    from scipy.stats import norm

    result = R.reduced_model(inp)
    levels = np.linspace(0.0, 1.0, 101)
    sigma = np.asarray(result.sigma_q, dtype=float)
    quantiles = np.stack([pretension + s * norm.ppf(np.clip(levels, 1e-6, 1 - 1e-6)) for s in sigma])
    return np.maximum(quantiles, 0.0), sigma


def forecast_cell(name: str, spec, v1_inputs: dict) -> dict:
    """Everything committed for one cell: the transmission tables and the FORECAST branching numbers."""
    rs = cells.response_spec(spec)
    table = TR.predicted_pair_table(rs, name)
    source = name if name in v1_inputs else FORECAST_SOURCE.get(name)
    out = {"transmission": table, "forecast_source_cell": source}
    if source is None or source not in v1_inputs:
        out["branching"] = {"status": "no forecast: no v1 source cell", "source": source}
        return out
    src = v1_inputs[source]
    r = np.array([[0.0 if v is None else v for v in row] for row in table["response_M1_half_sine"]])
    lti_quantiles, sigma = _margin_quantiles_lti(rs.inp, spec.pretension)
    empirical = src.get("margin_quantiles")
    use_empirical = empirical is not None and np.all(np.isfinite(empirical))
    quantiles = empirical if use_empirical else lti_quantiles
    peaks, parents = src["parent_peak"], src["parent_cable"]
    kernel = np.nan_to_num(B.kernel_from_margin_law(r, quantiles, peaks, parents), nan=0.0)
    kernel_lti = np.nan_to_num(B.kernel_from_margin_law(r, lti_quantiles, peaks, parents), nan=0.0)
    pi = src["primaries"] / max(src["primaries"].sum(), 1.0)
    rho = B.spectral_radius(kernel)
    sub = rho < 1.0
    per_event = np.zeros(peaks.size)
    cdfs = [B.margin_cdf_from_quantiles(quantiles[i]) for i in range(5)]
    for e, (peak, j) in enumerate(zip(peaks, parents)):
        per_event[e] = sum(float(B.transmission_probability(r[i, j], cdfs[i], np.array([peak]))[0]) for i in range(5) if i != j)
    primary_rate = src["primaries"] / src["exposure_s"]
    out["branching"] = {
        "status": "FORECAST (kernel from the exact-response R, the v1 cache's empirical taut-tension margin law, and v1 parent peaks; not a measurement)",
        "margin_law": {
            "model": ("empirical proxy: taut-tension quantiles (e > 0 samples, unclipped k e + c edot, no B.3 exclusion) of the source cell's statistics seeds from the v1 cache's 1 Hz samples; the declared margin law of WP2 is the B.3 clean set of the v3 pilots"
                      if use_empirical else "Gaussian N(T0, sigma_q) per cable from tether.theory.reduced_lti.reduced_model (empirical law unavailable)"),
            "empirical_quantiles_N": None if not use_empirical else np.round(empirical, 1).tolist(),
            "sigma_q_lti_N": sigma.tolist(),
            "rho_under_lti_gaussian_law": B.spectral_radius(kernel_lti),
        },
        "parents": {"n_events": int(peaks.size), "source_seeds": src["seeds"], "peak_over_T0_quantiles": {str(q): float(np.percentile(peaks, q) / spec.pretension) for q in (50, 80, 95, 99)}},
        "kernel_K": kernel.tolist(),
        "rho": rho,
        "primary_type_distribution_pi": pi.tolist(),
        "primary_rate_per_cable_s": primary_rate.tolist(),
        "m_bar": B.mean_offspring(kernel, pi),
        "theta_pred": B.extremal_index(kernel, pi) if sub else None,
        "expected_cluster_size": B.expected_cluster_size(kernel, pi) if sub else None,
        "total_rate_forecast_per_cable_s": B.total_rate(kernel, primary_rate).tolist() if sub else None,
        "offspring_count_forecast": float(per_event.sum()),
        "offspring_per_event_forecast": float(per_event.mean()),
        "extrapolated": name not in v1_inputs,
    }
    return out


def _sweep_forecast(committed: dict) -> str:
    """The T3.4 branch the forecast kernel implies along the T0 = 0.6 kN sweep."""
    rows = [(cells.parse_cell(name)["intensity"], committed[name]["branching"].get("rho"))
            for name in cells.sweep_cells() if name in committed and "rho" in committed[name]["branching"]]
    rows.sort()
    below = [i for i, r in rows if r is not None and r < 1.0]
    above = [i for i, r in rows if r is not None and r >= 1.0]
    table = ", ".join(f"I={i}: rho={r:.2f}" for i, r in rows)
    if below and above and min(above) > max(below):
        return f"(a) transition forecast between I = {max(below)} and I = {min(above)} ({table}); whether those cells are operable is not forecast"
    if not above:
        return f"(b) margin result forecast: rho below 1 over the sweep ({table})"
    return f"unordered forecast ({table}); the measured kernel decides"


def _v2_priors() -> dict:
    payload = json.loads(V2_PREDICTIONS.read_text())
    ratio = payload["III_1"]["9_cascade_transmission_ratio"]
    m_bar = payload["III_1"]["10_transmitted_onsets_m_bar"]
    theta = {f"T{int(row['T0'])}_kh{int(row['k_h'])}": row["theta_committed"] for row in payload["prop1prime_theta"].values()}
    return {
        "file": "records/v2/phase2/phase2_predictions_i035.json", "sha256": sha256_file(V2_PREDICTIONS),
        "status": "PRIOR FORECASTS of plan v2 (reduced LTI model, intensity 0.35, k_sigma = 3), phase never launched, never scored, never a result",
        "transmission_ratio": ratio, "m_bar": m_bar, "theta_committed": theta,
    }


def predict() -> str:
    if PREDICTIONS_PATH.exists():
        raise SystemExit(f"{PREDICTIONS_PATH} exists; predictions are never edited - write a dated addendum")
    v1 = _v1_cell_inputs()
    committed = {}
    for stage in cells.STAGES:
        for name, spec in cells.cell_specs(stage).items():
            committed[name] = {"stage": stage, "config_hash": spec.config_hash(), **forecast_cell(name, spec, v1)}
    pair = cells.STIFFNESS_FACTORS
    lo, hi = (f"{cells.CALIBRATION_CELL}_{s}" for s, _ in sorted(pair.items(), key=lambda kv: kv[1]))
    delta_exact = committed[hi]["transmission"]["band_statistic_median_offdiag"] - committed[lo]["transmission"]["band_statistic_median_offdiag"]
    norm_hi = np.array(committed[hi]["transmission"]["response_over_geometry_factor"], dtype=float)
    norm_lo = np.array(committed[lo]["transmission"]["response_over_geometry_factor"], dtype=float)
    per_pair = (norm_hi - norm_lo)[~np.eye(norm_hi.shape[0], dtype=bool)]
    per_pair = per_pair[np.isfinite(per_pair)]
    payload = {
        "schema": "v3-predictions-1",
        "what": "Plan v3 committed predictions, computed before any plant run; every number is a FORECAST",
        "closed_form": {"ratio": TR.closed_form_ratio(), "ratio_rectangular": TR.closed_form_ratio() * TR.pulse_factor("rectangular"),
                        "band_paper": list(BAND), "threshold_T0": TR.unloading_threshold(TR.closed_form_ratio()), "frequencies": TR.frequencies()},
        "exact_response_summary": {name: committed[name]["transmission"]["band_statistic_median_offdiag"] for name in committed},
        "stiffness_pair": {"low": lo, "high": hi, "delta_exact_uniform_pairs": delta_exact, "delta_exact_per_pair_range": [float(per_pair.min()), float(per_pair.max())],
                           "scoring_rule": "Delta_exact at scoring time = the same median statistic applied to R_ij / geometry_factor_ij over the scored rows' pair mix (T1.4)",
                           "delta_closed_form": 0.0},
        "verdict_forecasts": {
            "T1.1": "FAIL (band): predicted medians below 0.2 in every cell",
            "T1.2": "PASS", "T1.3": "PASS", "T1.4": "PASS-exact if the seed-bootstrap interval resolves 0.045",
            "T2.1": "PASS", "T2.2": "PASS for parallel cells; the fan subgroup is the genuine test of the projection",
            "T2.3": "PASS (the p6_t0 fleet-mode integrator passed the same tolerances)", "T2.4": "PASS",
            "T2.5": "PASS if the plant's margin law at t_up resembles the v1 taut-tension law; the forecast kernel ignores the neighbour's own motion during the pulse and any nonlinearity, so it is expected to overpredict",
            "T3.1": "PASS", "T3.2": "PASS",
            "T3.4": _sweep_forecast(committed),
        },
        "cells": committed,
        "v2_priors": _v2_priors(),
        "inputs_sha256": {"v1_record": sha256_file(V1_RECORD), "v1_cache": sha256_file(V1_CACHE)},
        "source": source_state(),
    }
    return write_bytes(PREDICTIONS_PATH, json_bytes(payload))


# ----------------------------------------------------------------------------- declarations, addenda, gates


def declaration_payload() -> dict:
    payload = dict(DECLARATIONS)
    payload["plan_sha256"] = sha256_file(PLAN_PATH)
    payload["theory_sha256"] = sha256_file(THEORY_PATH)
    payload["predictions_sha256"] = sha256_file(PREDICTIONS_PATH)
    payload["input_sha256"] = {"v1_record": sha256_file(V1_RECORD), "v1_cache": sha256_file(V1_CACHE), "v2_predictions": sha256_file(V2_PREDICTIONS)}
    payload["module_sha256"] = {name: sha256_file(path) for name, path in MODULE_PATHS.items()}
    payload["cells"] = {stage: {name: {"config_hash": spec.config_hash(), **cells.parse_cell(name)} for name, spec in cells.cell_specs(stage).items()} for stage in cells.STAGES}
    payload["source"] = source_state()
    return payload


def declare() -> str:
    if DECLARATIONS_PATH.exists():
        raise SystemExit(f"{DECLARATIONS_PATH} exists; declarations are never edited - write a dated addendum")
    for path in (PLAN_PATH, THEORY_PATH, PREDICTIONS_PATH):
        if not path.exists():
            raise SystemExit(f"{path} is missing; declare needs the plan, the theory document and the predictions")
    return write_bytes(DECLARATIONS_PATH, json_bytes(declaration_payload()))


def verify_declared() -> dict:
    """The committed declarations, after checking that the static part still matches this module."""
    stored = json.loads(DECLARATIONS_PATH.read_text())
    for key, value in DECLARATIONS.items():
        if json_bytes(stored.get(key)) != json_bytes(value):
            raise SystemExit(f"declaration '{key}' differs from the declared file; write an addendum")
    if stored["predictions_sha256"] != sha256_file(PREDICTIONS_PATH):
        raise SystemExit("the predictions file has changed since it was declared")
    return stored


def addendum_path(number: int) -> Path:
    return RECORD_DIR / f"cascade_addendum_{number}.json"


def write_addendum(number: int, date: str, why: str, change: dict, seen_before_the_change: str) -> str:
    path = addendum_path(number)
    if path.exists():
        raise SystemExit(f"{path} exists; addenda are never edited")
    payload = {"addendum": number, "date": date, "amends": {"path": "records/v3/cascade_declarations.json", "sha256": sha256_file(DECLARATIONS_PATH)},
               "why": why, "change": change, "seen_before_the_change": seen_before_the_change, "source": source_state()}
    return write_bytes(path, json_bytes(payload))


def failing_blocking(tests: dict) -> list[str]:
    """Blocking tests not passed; a conditional test whose prerequisite failed is not scored, not failing."""
    out = []
    for t, v in tests.items():
        spec = TESTS[t]
        if not spec["blocking"] or v.get("verdict") == "PASS":
            continue
        prerequisite = spec.get("conditional_on")
        if prerequisite and tests.get(prerequisite, {}).get("verdict") != "PASS":
            continue
        out.append(t)
    return out


def write_gate(work_package: int, verdict: str, tests: dict, consequences: list[str], provenance: dict) -> str:
    """records/v3/gate_wp{N}.json in the shape of records/v2/phase1/phase1_gate.json."""
    payload = {"schema": "v3-gate-record-1", "work_package": work_package, "verdict": verdict,
               "rule": DECLARATIONS["kernel_path_rule"] if work_package == 2 else f"the blocking tests of WP{work_package} in records/v3/cascade_declarations.json",
               "blocking_in_scope": [t for t, spec in TESTS.items() if spec["blocking"] and spec["work_package"] == work_package],
               "tests": tests, "failing_blocking_tests": failing_blocking(tests),
               "consequences": consequences, "provenance": provenance, "declarations_sha256": sha256_file(DECLARATIONS_PATH), "source": source_state()}
    return write_bytes(RECORD_DIR / f"gate_wp{work_package}.json", json_bytes(payload))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["predict", "declare", "verify"])
    args = parser.parse_args(argv)
    if args.command == "predict":
        print("wrote", PREDICTIONS_PATH, predict())
    elif args.command == "declare":
        print("wrote", DECLARATIONS_PATH, declare())
    else:
        verify_declared()
        print("declarations verified:", sha256_bytes(DECLARATIONS_PATH.read_bytes()))


if __name__ == "__main__":
    main()
