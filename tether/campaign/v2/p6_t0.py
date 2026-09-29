"""P6-T0: the catch admissibility test (plan v2 II.10 Prop. 14, IV.7, Phase 6 row P6-T0, App. D).

Offline counterfactual on the 40 v1 Phase 5 calibration missions (arm N, recording mode):
for every dangerous re-engagement, the slack vessel is re-integrated from the measured
turnaround to re-engagement under the candidate catch law (``tether.control.catch``), with
its own weather regenerated exactly, the recorded headings, and the load either moving on
its recorded trajectory (``literal``) or integrated with the rest of the fleet
(``fleet``).  See ``DECLARATIONS`` (written to ``records/v2/phase6/p6_t0_declarations.json``
by ``declare`` before ``run`` computes anything).

Commands: ``declare`` writes the declarations (refuses to overwrite); ``run`` computes and
writes ``records/v2/phase6/p6_t0_results.json``.
"""

from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from tether.campaign.common import RECORDS, json_bytes, run_pool, sha256_bytes, sha256_file, source_state, write_bytes
from tether.control.catch import (
    LANDING_DECELERATION,
    SOFT_SPEED_FRACTION,
    VARIANTS,
    VELOCITY_GAIN,
    catch_thrust,
)
from tether.physics import constants

RECORD_DIR = RECORDS / "v2" / "phase6"
DECLARATIONS_PATH = RECORD_DIR / "p6_t0_declarations.json"
RESULTS_PATH = RECORD_DIR / "p6_t0_results.json"
MISSIONS_PATH = RECORDS / "phase5" / "cache" / "phase5_missions.pkl"
ARMS_PATH = RECORDS / "phase5" / "cache" / "phase5_missions_arms.pkl"
V1_PHASE6_PATH = RECORDS / "phase6" / "cache" / "phase6_compute.pkl"
FAN_IMPACT_PATH = RECORDS / "phase5" / "impact_table_fan.json"
PLAN_PATH = RECORDS.parent / "ref" / "tail_of_the_tether_plan_v2.md"

STRESS_THRESHOLD = 4500.0
PRETENSION = 1000.0
BURST_WINDOW = 2.0
LONG_DWELL = 2.0
PLATEAU = (55.0, 65.0)
ADMISSIBLE_FRACTION = 0.80
SNAP_WINDOW = 0.05
SNAP_WINDOW_SENSITIVITY = 0.335
STEP = 5.0e-4
STATE_PERIOD = 0.01
EVENT_PERIOD = 1.0e-3
CONTROL_PERIOD = 0.02
MONITOR_PERIOD = 0.1
HORIZON = 20.0
CLOSURE_CHORD = 1.0
PEAK_WINDOW = 1.0
ARMS = ("P", "B2", "L")
CONTROLLERS = ("true10", "true50") + ARMS
MODES = ("fleet", "literal")
PRIMARY = {"mode": "fleet", "controller": "true10", "population": "first_severance"}
VALIDATION_P95 = 0.01
VALIDATION_MAX = 0.03
VALIDATION_TIME = 0.005
WORKERS = 4
STEPS_PER_ROW = int(round(STATE_PERIOD / STEP))

DECLARATIONS = {
    "test": "P6-T0 - blocking pre-campaign admissibility of the velocity-matching catch (plan v2 II.10 Prop. 14; IV.7; Part V Phase 6 test table row P6-T0; Appendix D 'Before Phase 6 is launched')",
    "declared": "2026-09-13, before any counterfactual was integrated on the records; the only look at the Phase 5 records before this file was a plant-timing check (below) and the cache layout; the integrator and the law were exercised on synthetic calm-water excursions only (unit tests)",
    "data": {
        "missions": "records/phase5/cache/phase5_missions.pkl - the 40 v1 Phase 5 calibration missions (seeds 5001-5040), MissionSpec() defaults (v1 squall centred at 60 s, plateau 55-65 s), arm N (no supervisor), recording mode; per mission the 10 ms truth.state, the 1 ms e/edot log and the re-engagement marks",
        "arms": "records/phase5/cache/phase5_missions_arms.pkl - per mission outputs {P, B2, L} EstimatorOutput (10 Hz: time, slack, e_hat, edot_hat per vessel) and monitor_truth (e_true, edot_true at the same ticks)",
        "impact_table": "records/phase5/impact_table_fan.json; v_b per cable = tether.monitor.hazard.critical_speeds(4500, 1000, table_path=impact_table_fan.json)",
        "v1_phase6": "records/phase6/cache/phase6_compute.pkl - the v1 Phase 6 campaign; its 60 arm-N recording missions are used only for the taut-overload share",
    },
    "constants": {
        "T_b_s_N": STRESS_THRESHOLD,
        "pretension_N": PRETENSION,
        "v_soft": f"{SOFT_SPEED_FRACTION} v_b (per cable)",
        "a_c_m_s2": LANDING_DECELERATION,
        "K_v_N_s_m": VELOCITY_GAIN,
        "s_i": "1 (no easing feed-forward in the counterfactual)",
        "F_T": "the scheduled thrust of the recorded mission: operating_point thrust of vessel i times mission.thrust_envelope at the governing 50 Hz control tick (constant F_T,i on [8, 110] s)",
        "F_min_variants": {"hold": "F_min = 0", "reverse": "F_min = -F_T"},
        "vessel": {"mass_kg": constants.VESSEL_MASS, "linear_drag_N_s_m": constants.VESSEL_LINEAR_DRAG},
        "load": {"mass_kg": constants.LOAD_MASS, "linear_drag_N_s_m": constants.LOAD_LINEAR_DRAG, "yaw_inertia_kg_m2": constants.LOAD_YAW_INERTIA, "angular_drag_N_m_s": constants.LOAD_ANGULAR_DRAG},
        "cable": {"rest_length_m": constants.CABLE_REST_LENGTH, "stiffness_N_m": constants.CABLE_STIFFNESS, "damping_N_s_m": constants.CABLE_DAMPING},
        "physics_step_s": STEP,
    },
    "law": "Prop. 14 exactly as IV.7 prints it, implemented in tether.control.catch.catch_thrust: while the slack flag is set and edot_hat > edot_ref(e_hat) = sqrt(v_soft^2 + 2 a_c (-e_hat)), F = clip(F_T s + K_v (edot_ref - edot_hat), F_min, F_T); otherwise F = F_T s. For e_hat >= 0 the profile is v_soft (argument max(-e_hat, 0)); a non-finite estimate means no catch. Thrust acts along the vessel's heading (negative = astern).",
    "sign_convention": "from tether.physics.fleet.cable_kinematics: chord = stern point - load attachment, e = |chord| - L, edot = (v_stern - v_load) . d > 0 while lengthening; a slack cable closing toward e = 0 has edot > 0 and its re-engagement speed is edot at the upcrossing of e = 0; thrust (along the hull, away from the load) raises edot",
    "event_rule": "per cable, marks in t_up order; an event is the first re-engagement of a burst and every later mark on the same cable with t_up - t_event <= 2 s belongs to it (anchored at the event's first mark, B.1 read literally); the next mark more than 2 s after the event's first mark starts a new event",
    "dangerous": "an event is dangerous if any mark of its burst has T_peak > 4500 N (strict); its dangerous re-engagement is the first mark of the burst with T_peak > 4500 N (normally the event's first mark; the count of events where it is not is reported), and the counterfactual integrates that mark's own slack excursion",
    "populations": {
        "first_severance": "VERDICT population: per mission, the dangerous event whose dangerous re-engagement has the earliest t_up (the first declustered event whose re-engagement produces T_peak > 4500 N); one per mission that has one",
        "first_severance_plan_literal": "reported sensitivity: per mission, the first 1 ms sample with q = k e + c edot (e > 0) above 4500 N; if a mark on that cable has |t_up - t_cross| <= 50 ms (the closest such mark) its excursion is the first-severance excursion (a snap), else the first severance is a taut overload and the mission contributes no excursion",
        "all_dangerous": "every dangerous event of the 40 missions",
        "long_dwell": "dangerous events whose dangerous re-engagement has dwell >= 2 s (a mark with dwell >= 2 s cannot be a bounce inside a 2 s burst), split by the recorded t_up of the dangerous re-engagement: pre_plateau t_up < 55 s, plateau 55 <= t_up < 65 s (v1 squall plateau), post_plateau t_up >= 65 s ('post-squall')",
    },
    "turnaround": "the deepest point of the dangerous re-engagement's slack excursion: argmin of the 1 ms truth elongation over labels in [t_up - dwell, t_up]; the counterfactual starts from the recorded full plant state at the latest 10 ms state label at or before it",
    "plant_timing": "the 10 ms truth.state and the 1 ms e/edot logs at label t hold the plant state after the SAP step taken at t, i.e. the state at t + 0.5 ms (Drake's simultaneous discrete updates). Established before this declaration by a timing check on free-flight (slack) windows of mission 5001 with the vessel equations below: 10 ms velocity residual 1e-8 m/s with the offset against 4e-5 m/s without. The counterfactual runs on the record's label axis; the step from label L to L + 0.5 ms applies forces at time L + 0.5 ms; semi-implicit Euler (v += h a(x_n); x += h v_new), as the discrete plant does",
    "vessel_dynamics": "m dv/dt = -c v + F (cos theta, sin theta) + W_i(t) at the body origin (the plant's HullForces; a slack cable applies no force; tension max(k e + c edot, 0) only where e > 0)",
    "weather": "regenerated exactly with tether.campaign.mission.mission_weather(MissionSpec(), seed) (background t3 common-mode weather with the 8 s ramp plus the squall), zero-order hold on the 10 ms grid, index floor((t + 1e-9) / 0.01); body slot 0 is the load, slot 1 + i vessel i",
    "thrust_timing": "the surge command is held for 20 ms: a command computed at control tick T (multiple of 0.02 s) acts on the steps at times in (T, T + 0.02]; other vessels (and vessel i in the replay) use the recorded arm-N command, i.e. the schedule at the governing tick",
    "heading": "every vessel's heading and yaw rate are prescribed from the record (linear interpolation between 10 ms rows); in every law run vessel i's heading is held at its recorded value at the recorded re-engagement label t_up (yaw rate 0) from t_up on, because the record after t_up contains the recorded snap's yaw kick, which the counterfactual (cable still slack) does not have; in the replay (where the cable does re-engage at t_up) the recorded heading is used throughout. Thrust acts along that heading",
    "modes": {
        "fleet": "PRIMARY. The load and all five vessels' translation are integrated with the plant's cable, drag, thrust and weather laws from the recorded full state at the turnaround (vessel headings prescribed as above; load yaw integrated). Up to the recorded re-engagement this reproduces the record exactly (the counterfactual vessel's cable is slack, so nothing else changes); after it, the load moves without the recorded snap. Why primary: the plan's 'exact up to re-engagement' holds only up to the RECORDED re-engagement; a braking law re-engages later, and after the recorded t_up the recorded load trajectory contains the recorded snap (load yanked toward the vessel by an impulse of order 1e3 N s, 0.1-0.3 m/s of load velocity against v_b = 0.46 m/s), which a slack counterfactual cable never applies",
        "literal": "REPORTED. The task's and plan's literal reading: vessel i's translation only, the load attachment point on its recorded trajectory throughout (cubic Hermite in pose using the recorded twist; twist linearly interpolated); after the recorded t_up this includes the recorded snap and is expected to flatter the law",
    },
    "controllers": {
        "replay": "validation: the law replaced by the recorded thrust command",
        "true10": "VERDICT table ('true chord state'): e_hat, edot_hat = the counterfactual true e, edot of cable i at the latest 10 Hz monitor tick strictly before the control tick (the monitor chain of the capstone: estimates at 0.1 s ticks, adapter offset 5 ms, controller at the next 50 Hz tick); slack flag = counterfactual e <= 0",
        "true50": "reported: the counterfactual true e, edot at the control tick itself (50 Hz, no monitor latency)",
        "arms": "P, B2, L ('each arm's recorded estimates'): at the same 10 Hz ticks, e_hat = e_cf + (e_hat_rec - e_true_rec) and edot_hat = edot_cf + (edot_hat_rec - edot_true_rec) - the recorded estimation error at that tick added to the counterfactual true chord state, because the estimate after the turnaround is itself an estimate of the counterfactual motion; held between ticks. Before the recorded t_up the recorded slack flag is used (flag clear or non-finite estimate: no catch at that tick). From the recorded t_up on (recorded estimates are nan while taut) the flag is taken as set - the counterfactual cable is still slack - and the error is held at its last finite value on a slack tick in [t_up - dwell, t_up); if there is none the catch never engages there (counted)",
    },
    "stop": "the first step after which cable i's counterfactual elongation crosses 0 upward: re-engagement time and speed by linear interpolation in e between the two steps (the event tracker's rule); else formation closure (fleet: any chord below 1 m; literal: cable i's chord below 1 m); else the horizon min(turnaround + 20 s, last recorded state label - 10 ms)",
    "peak": "fleet mode only: after the counterfactual re-engagement the fleet is integrated on until the event tracker's first tension peak of cable i completes (first decrease of k e + c edot, or a down-crossing), at most 1 s; reported as the counterfactual T_peak (the recorded-thrust replay reproduces the recorded T_peak as a second check)",
    "excursion_outcome": "an excursion is caught (success) iff the counterfactual re-engages before closure and the horizon with v_return < v_b of its cable; closure and no re-engagement within the horizon count as failures (reported separately, and the fraction excluding them is reported beside)",
    "admissibility_rule": "a variant is admissible iff its caught fraction on the first_severance population, in the fleet mode with the true10 controller, is >= 0.80 (point fraction, as the plan prints it). The literal mode, true50 and the arm tables are published beside and gate nothing. F_min for the capstone: if both variants are admissible, hold (F_min = 0) - the smaller intervention, with no astern thrust and so less closure risk and mission-time cost; if one, that one; if neither, the plan's outcome: NO-LAUNCH of the catch arm (slack-phase authority insufficient at the stress threshold, Claim C withdrawn on this plant)",
    "interval": "Clopper-Pearson exact binomial 95% interval on each fraction (C.3); the normal-approximation half-width 1.96 sqrt(p(1-p)/n) at p = 0.8 is stated beside (0.175 at n = 20)",
    "validation": "before any admissibility number is trusted: the replay (recorded thrust) must reproduce each dangerous excursion's recorded v_return and t_up in both modes: 95th percentile of |v_cf - v_rec| <= 0.01 m/s, max <= 0.03 m/s, max |t_cf - t_up| <= 5 ms; the fleet-mode T_peak error is reported. If it fails, the admissibility numbers are reported as untrusted with the diagnosis (heading, weather slot, attachment geometry)",
    "taut_overload": "first severance of a recording mission = first 1 ms sample with k e + c edot (e > 0) above 4500 N on any cable; a snap if a mark on that cable lies within 50 ms of it (|t_up - t_cross| <= 0.05 s, Phase 6 Observables), else a taut overload. Counted on (a) the 40 Phase 5 missions and (b) the v1 Phase 6 arm-N recording missions (60; the plan's 5/60 = 0.083). Floor: p_arm >= taut-overload count / missions; ceiling on p_N - p_arm = p_N - that floor (the snap-first share). Sensitivity: the window widened to a mark within one engagement period (0.335 s) before the crossing, because a mild snap crosses 4500 N near its peak, about a quarter period (~84 ms) after t_up",
    "estimation_error_at_closure": "reported per arm on each population: the held edot error at the recorded t_up (Prop. 14's V_up = |edot_hat - edot| + r)",
    "workers": WORKERS,
    "predictions": {
        "plan_prior": "plan II.10 / Phase 6 outcome matrix (open-loop counterfactual on the recorded Phase 5 wind): hold clears v_b on 48% of dangerous long-dwell excursions (32% plateau / 71% after), reverse on 81% (68% / 100%); hold misses 80% on first severances widely, reverse clears it by one or two events; the plan expects reverse admissible and does not claim it",
        "plan_arms": "P re-engages near 1.5 v_b (its chord-rate error at closure 0.70 m/s median, p90 1.6 m/s, against v_b = 0.46 m/s); B2 (1.14 m/s) and L (4.1 m/s) above v_b",
        "this_test": "committed here: the fleet-mode caught fractions are no higher than the literal-mode ones for either variant (the literal load trajectory carries the recorded snap after the recorded t_up); the replay validation passes to better than 1 mm/s in median",
        "proportional_offset": "committed here from the synthetic calm-water unit test (tether/tests/test_v2_catch.py, no record involved): the law as printed is a proportional velocity loop; where it does not saturate at F_min it settles ABOVE the landing profile by about (F_T - F_needed)/K_v, F_needed the thrust that holds the vessel's speed relative to the load (in calm water F_T - F_needed ~ T0 = 1 kN, an offset of ~0.5 m/s), and it landed at 0.68-0.71 m/s (about 1.5 v_b) from a 1.5 m calm-water excursion under both variants. Prediction: on the records the re-engagement speed of unsaturated excursions sits near v_soft + 0.3-0.5 m/s, i.e. near or above v_b, hold and reverse differ only on excursions that saturate, and both caught fractions fall well below the plan's prior (hold 48% / reverse 81% on long-dwell); Prop. 14's premise 'the authority suffices to hold edot_hat on the profile' fails through the gain, not only through F_min",
    },
}


ADDENDUM_PATH = RECORD_DIR / "p6_t0_addendum_1.json"
ADDENDUM = {
    "addendum": 1,
    "date": "2026-09-13",
    "amends": "records/v2/phase6/p6_t0_declarations.json (sha256 973b6b64e65741780531296f614bd8a920facc20bf055490306ba84fc461cc99), key 'heading' (and the literal mode's load interpolation)",
    "why": "the declared replay validation FAILED on the first run (records/v2/phase6/p6_t0_results_run1_failed_validation.json, kept): |v_cf - v_rec| p95 0.073 m/s / max 0.090 m/s (fleet) and 0.085 / 0.099 m/s (literal), always low by about 1% of the speed, while the crossing times agreed to 0.06 ms and the fleet T_peak to 0.6%. Diagnosis on seed 5011 cable 0: the counterfactual state matches the record to 1e-6 m and 1e-6 m/s from the turnaround until the last 10 ms row before t_up, then its edot drops by 0.07-0.08 m/s within 6 ms. The linear interpolation between the last pre-contact row and the first post-contact row pulls the recorded snap's yaw kick on vessel i (and, in the literal mode, the snap's impulse on the load twist) back into the few milliseconds before contact. The heading and weather-slot treatments and the attachment geometry are otherwise exact (1e-6 agreement over 2.3 s)",
    "change": "inside the 10 ms row interval that contains the recorded t_up only (labels in [row at or before t_up, t_up)): vessel i's heading is extrapolated from that row with its recorded yaw rate (theta_k + omega_k (L - L_k), yaw rate omega_k), and in the literal mode the load pose and twist are extrapolated from that row with its recorded twist; in law runs vessel i's held heading from t_up on is that extrapolated value at t_up. Everything else is unchanged (the replay after t_up still uses the recorded heading, the literal load after t_up still the recorded trajectory)",
    "seen_before_the_change": "the failed validation above and, in the same file, the untrusted admissibility table (fleet/true10 first-severance 1 of 16 caught for both variants); the change is dictated by the validation (a pre-contact interpolation defect confined to the last <= 10 ms before the recorded contact) and touches no threshold, population, law constant or decision rule",
}


ADDENDUM_2_PATH = RECORD_DIR / "p6_t0_addendum_2.json"
ADDENDUM_2 = {
    "addendum": 2,
    "date": "2026-09-13",
    "amends": "addendum 1 (records/v2/phase6/p6_t0_addendum_1.json): the extrapolation interval",
    "why": "a replay-only diagnostic run after addendum 1 (no law run, no results file) still showed the 5011/0 excursion 0.07 m/s low: the step that brackets the crossing (the first label >= t_up) still used the interpolation toward the post-contact row, and the crossing rate is read from that step",
    "change": "the extrapolation of addendum 1 (vessel i's heading, and the literal mode's load pose and twist) covers labels in [row at or before t_up, first label >= t_up] inclusive, i.e. through the step that brackets the crossing; law runs hold vessel i's heading from that first label on, at the extrapolated value at t_up (unchanged)",
    "seen": "replay-only diagnostic with the change: fleet |dv| median 4.8e-4, p95 7.7e-3, max 9.4e-3 m/s (passes the declared validation); literal median 3.3e-3, p95 0.039, max 0.067 m/s (fails): its residual sits on excursions inside multi-cable cascades (5003 cables 1-4 within 0.2 s; 5013), where the recorded 10 ms load rows cannot resolve the other cables' millisecond-scale snap impulses on the load - a limit of the literal method, which the fleet mode avoids by integrating the load. No threshold, population, law constant or decision rule is changed; the literal table stays published and is flagged by its failed validation",
}


# ----------------------------------------------------------------------------- events and populations


def declustered_events(marks: list[tuple], window: float = BURST_WINDOW, threshold: float = STRESS_THRESHOLD) -> list[dict]:
    """Events of B.1 (anchored bursts per cable) from mark tuples (cable, t_up, depth, v_return, T_peak, dwell)."""
    events: list[dict] = []
    by_cable: dict[int, list[tuple]] = {}
    for mark in marks:
        by_cable.setdefault(int(mark[0]), []).append(mark)
    for cable, rows in sorted(by_cable.items()):
        rows = sorted(rows, key=lambda row: row[1])
        burst: list[tuple] = []
        for row in rows:
            if burst and row[1] - burst[0][1] > window:
                events.append(_event(cable, burst, threshold))
                burst = []
            burst.append(row)
        if burst:
            events.append(_event(cable, burst, threshold))
    return sorted(events, key=lambda event: (event["t_event"], event["cable"]))


def _event(cable: int, burst: list[tuple], threshold: float) -> dict:
    dangerous = [index for index, row in enumerate(burst) if row[4] > threshold]
    first = dangerous[0] if dangerous else None
    mark = burst[first] if first is not None else None
    return {
        "cable": cable,
        "t_event": float(burst[0][1]),
        "marks": len(burst),
        "cluster_max": float(max(row[4] for row in burst)),
        "dangerous": first is not None,
        "dangerous_index": first,
        "t_up": None if mark is None else float(mark[1]),
        "depth": None if mark is None else float(mark[2]),
        "v_return": None if mark is None else float(mark[3]),
        "T_peak": None if mark is None else float(mark[4]),
        "dwell": None if mark is None else float(mark[5]),
    }


def first_crossing(event_time: np.ndarray, elongation: np.ndarray, rate: np.ndarray, level: float = STRESS_THRESHOLD):
    """(t_cross, cable) of the first 1 ms sample with k e + c edot (e > 0) above ``level``, else None."""
    q = np.where(elongation > 0.0, constants.CABLE_STIFFNESS * elongation + constants.CABLE_DAMPING * rate, 0.0)
    over = np.argwhere(q > level)
    if over.size == 0:
        return None
    return float(event_time[over[0, 0]]), int(over[0, 1])


def severance_type(crossing, marks: list[tuple], window: float = SNAP_WINDOW, before_only: bool = False):
    """('none'|'snap'|'taut_overload', closest mark or None) of a first crossing."""
    if crossing is None:
        return "none", None
    time, cable = crossing
    if before_only:
        near = [m for m in marks if int(m[0]) == cable and 0.0 <= time - m[1] <= window]
    else:
        near = [m for m in marks if int(m[0]) == cable and abs(m[1] - time) <= window]
    if not near:
        return "taut_overload", None
    return "snap", min(near, key=lambda m: abs(m[1] - time))


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float | None, float | None]:
    if trials == 0:
        return None, None
    alpha = 1.0 - confidence
    low = 0.0 if successes == 0 else float(stats.beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return low, high


# ----------------------------------------------------------------------------- the counterfactual


@dataclass
class Excursion:
    """Everything one counterfactual needs, sliced from the records (labels on the record's axis)."""

    seed: int
    cable: int
    t_up: float
    v_return: float
    T_peak: float
    dwell: float
    depth: float
    critical_speed: float
    thrusts: np.ndarray  # (N,) operating thrusts
    load_offsets: np.ndarray  # (N, 2)
    vessel_offsets: np.ndarray  # (N, 2)
    start_label: float  # turnaround state label
    state: np.ndarray  # (K, 6 (N + 1)) recorded rows from the turnaround row
    weather: np.ndarray  # (W, N + 1, 2) rows from weather_start
    weather_start: int
    log_start: int  # 1 ms index of e_log[0]
    e_log: np.ndarray  # recorded 1 ms e of cable i
    edot_log: np.ndarray
    estimates: dict = field(default_factory=dict)  # arm -> dict(tick0, slack, e_err, edot_err, held)
    tags: dict = field(default_factory=dict)
    envelope: object = None  # callable t -> thrust envelope (array ok)


def _rotate(angle, vectors):
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * vectors[..., 0] - s * vectors[..., 1], s * vectors[..., 0] + c * vectors[..., 1]], axis=-1)


def _hermite(p0, p1, v0, v1, dt, s):
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * p0 + h10 * dt * v0 + h01 * p1 + h11 * dt * v1


def _control_tick(actual: np.ndarray) -> np.ndarray:
    """Control tick governing a step at time ``actual``: the last 0.02 s tick strictly before it."""
    return CONTROL_PERIOD * np.ceil(actual / CONTROL_PERIOD - 1.0e-9) - CONTROL_PERIOD


def _monitor_tick(control: float) -> float:
    """Latest 10 Hz monitor tick strictly before the control tick."""
    return MONITOR_PERIOD * math.ceil(control / MONITOR_PERIOD - 1.0e-9) - MONITOR_PERIOD


def integrate(x: Excursion, mode: str, controller: str, min_fraction: float | None, horizon: float = HORIZON, trace: bool = False) -> dict:
    """One counterfactual from the turnaround; ``controller`` 'replay' uses the recorded thrust.

    ``trace`` adds the final body positions/velocities (fleet mode) for tests.
    """
    n_vessels = x.thrusts.size
    base = 3 * (n_vessels + 1)
    i = x.cable
    L0 = x.start_label
    rows = x.state.shape[0]
    replay = controller == "replay"
    literal = mode == "literal"
    steps = int(min(round(horizon / STEP), (rows - 2) * STEPS_PER_ROW))
    n_idx = np.arange(steps + 1)
    labels = L0 + n_idx * STEP
    actual = labels + STEP
    # prescribed vessel headings / yaw rates on the step labels (linear between 10 ms rows)
    row = n_idx // STEPS_PER_ROW
    frac = (n_idx % STEPS_PER_ROW) / STEPS_PER_ROW
    after = np.minimum(row + 1, rows - 1)
    theta_rows = x.state[:, 5:base:3]
    omega_rows = x.state[:, base + 5 :: 3]
    theta = theta_rows[row] + frac[:, None] * (theta_rows[after] - theta_rows[row])
    omega = omega_rows[row] + frac[:, None] * (omega_rows[after] - omega_rows[row])
    # Addendum 1: inside the 10 ms row interval that contains the recorded t_up, the row after
    # t_up already carries the recorded snap; vessel i's heading there is extrapolated from the
    # last row at or before t_up with its recorded yaw rate instead of interpolated toward it.
    k_up = int(math.floor((x.t_up - L0) / STATE_PERIOD + 1.0e-9))
    n_up = int(math.ceil((x.t_up - L0) / STEP - 1.0e-9))  # first step label >= t_up
    contact = None
    if 0 <= k_up < rows - 1 and k_up * STEPS_PER_ROW <= steps:
        # addendum 2: through the step that brackets the crossing (the first label >= t_up)
        lo, hi = k_up * STEPS_PER_ROW, min(n_up + 1, steps + 1)
        contact = (lo, hi, k_up)
        elapsed = (n_idx[lo:hi] - lo) * STEP
        theta[lo:hi, i] = theta_rows[k_up, i] + omega_rows[k_up, i] * elapsed
        omega[lo:hi, i] = omega_rows[k_up, i]
        if not replay and n_up <= steps:
            # law runs: vessel i's heading held at its (extrapolated) value at t_up, yaw rate 0
            theta[n_up:, i] = theta_rows[k_up, i] + omega_rows[k_up, i] * (x.t_up - (L0 + k_up * STATE_PERIOD))
            omega[n_up:, i] = 0.0
    widx = np.floor((actual + 1.0e-9) / STATE_PERIOD).astype(int) - x.weather_start
    weather = x.weather[np.clip(widx, 0, x.weather.shape[0] - 1)]
    ticks = _control_tick(actual)
    schedule = x.thrusts[None, :] * np.asarray(x.envelope(ticks), dtype=float)[:, None]
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    heading_unit = np.stack([cos_t, sin_t], axis=-1)  # (S, N, 2)
    so = x.vessel_offsets
    stern = np.stack([cos_t * so[None, :, 0] - sin_t * so[None, :, 1], sin_t * so[None, :, 0] + cos_t * so[None, :, 1]], axis=-1)
    stern_rate = omega[:, :, None] * np.stack([-stern[..., 1], stern[..., 0]], axis=-1)
    s0 = x.state[0]
    pos = s0[:base].reshape(n_vessels + 1, 3)[:, :2].copy()
    vel = s0[base:].reshape(n_vessels + 1, 3)[:, :2].copy()
    load_theta = float(s0[2])
    load_omega = float(s0[base + 2])
    k_c, c_c, rest = constants.CABLE_STIFFNESS, constants.CABLE_DAMPING, constants.CABLE_REST_LENGTH
    inverse_mass = 1.0 / np.array([constants.LOAD_MASS] + [constants.VESSEL_MASS] * n_vessels)[:, None]
    drag = np.array([constants.LOAD_LINEAR_DRAG] + [constants.VESSEL_LINEAR_DRAG] * n_vessels)[:, None]
    if literal:
        load_pose = _hermite(x.state[row, 0:3], x.state[after, 0:3], x.state[row, base : base + 3], x.state[after, base : base + 3], STATE_PERIOD, frac[:, None])
        load_twist = x.state[row, base : base + 3] + frac[:, None] * (x.state[after, base : base + 3] - x.state[row, base : base + 3])
        if contact is not None:
            lo, hi, k = contact
            elapsed = ((n_idx[lo:hi] - lo) * STEP)[:, None]
            load_pose[lo:hi] = x.state[k, 0:3][None] + elapsed * x.state[k, base : base + 3][None]
            load_twist[lo:hi] = x.state[k, base : base + 3][None]
        arm_i = _rotate(load_pose[:, 2], x.load_offsets[i][None, :])
        load_point = load_pose[:, :2] + arm_i
        load_point_vel = load_twist[:, :2] + load_twist[:, 2:3] * np.stack([-arm_i[:, 1], arm_i[:, 0]], axis=-1)
        pos_i = pos[1 + i].copy()
        vel_i = vel[1 + i].copy()
    e_hist = np.full(steps + 1, np.nan)
    edot_hist = np.full(steps + 1, np.nan)

    def recorded_chord(label):
        index = int(round(label / EVENT_PERIOD)) - x.log_start
        return float(x.e_log[index]), float(x.edot_log[index])

    def chord_at_label(label):
        m = int(round((label - L0) / STEP))
        if m < 0:
            return recorded_chord(label)
        return float(e_hist[m]), float(edot_hist[m])

    def law_inputs(tick):
        if controller == "true50":
            e, edot = chord_at_label(tick)
            return e, edot, e <= 0.0
        tau = _monitor_tick(tick)
        e, edot = chord_at_label(tau)
        if controller == "true10":
            return e, edot, e <= 0.0
        est = x.estimates[controller]
        if tau < x.t_up:
            j = int(round(tau / MONITOR_PERIOD)) - est["tick0"]
            if j < 0 or j >= len(est["slack"]) or not est["slack"][j] or not (np.isfinite(est["e_err"][j]) and np.isfinite(est["edot_err"][j])):
                return np.nan, np.nan, False
            return e + float(est["e_err"][j]), edot + float(est["edot_err"][j]), True
        if est["held"] is None:
            return np.nan, np.nan, False
        return e + est["held"][0], edot + est["held"][1], True

    result = {"outcome": "horizon", "t_cf": None, "v_cf": None, "T_peak_cf": None}
    current_tick = None
    command = 0.0
    engaged_ticks = total_ticks = 0
    crossed_at = None
    peak = 0.0
    prev_e = prev_edot = None
    min_thrust = math.inf
    for n in range(steps + 1):
        if literal:
            dx = pos_i + stern[n, i] - load_point[n]
            length = math.sqrt(float(dx @ dx))
            rate = float((vel_i + stern_rate[n, i] - load_point_vel[n]) @ dx) / length
            shortest = length
        else:
            la = _rotate(load_theta, x.load_offsets)
            dx_all = pos[1:] + stern[n] - pos[0] - la
            rel = vel[1:] + stern_rate[n] - vel[0] - load_omega * np.stack([-la[:, 1], la[:, 0]], axis=-1)
            length_all = np.sqrt(np.einsum("ij,ij->i", dx_all, dx_all))
            u = dx_all / length_all[:, None]
            rate_all = np.einsum("ij,ij->i", rel, u)
            length, rate = float(length_all[i]), float(rate_all[i])
            shortest = float(length_all.min())
        e_now = length - rest
        e_hist[n] = e_now
        edot_hist[n] = rate
        if crossed_at is None and prev_e is not None and prev_e <= 0.0 < e_now:
            fraction = -prev_e / (e_now - prev_e)
            v_cf = prev_edot + fraction * (rate - prev_edot)
            result.update(outcome="reengaged", t_cf=float(labels[n - 1] + fraction * STEP), v_cf=float(max(v_cf, 0.0)))
            crossed_at = n
            peak = max(c_c * v_cf, k_c * e_now + c_c * rate)
            if literal:
                break
        elif crossed_at is not None:
            if e_now <= 0.0:
                result["T_peak_cf"] = float(peak)
                break
            q = k_c * e_now + c_c * rate
            if q < peak or (n - crossed_at) * STEP > PEAK_WINDOW:
                result["T_peak_cf"] = float(peak)
                break
            peak = q
        prev_e, prev_edot = e_now, rate
        if crossed_at is None and shortest < CLOSURE_CHORD:
            result.update(outcome="closure", t_cf=float(labels[n]))
            break
        if n == steps:
            break
        if replay:
            command = float(schedule[n, i])
        else:
            tick = float(ticks[n])
            if tick != current_tick:
                current_tick = tick
                scheduled = float(schedule[n, i])
                if crossed_at is not None:
                    command = scheduled  # released: the tension reads taut
                else:
                    e_hat, edot_hat, slack = law_inputs(tick)
                    command = float(catch_thrust(e_hat, edot_hat, slack, scheduled, x.critical_speed, float(min_fraction)))
                    total_ticks += 1
                    engaged_ticks += int(command < scheduled - 1.0e-9)
        min_thrust = min(min_thrust, command)
        if literal:
            acc = (command * heading_unit[n, i] + weather[n, 1 + i] - constants.VESSEL_LINEAR_DRAG * vel_i) / constants.VESSEL_MASS
            vel_i = vel_i + STEP * acc
            pos_i = pos_i + STEP * vel_i
            continue
        e_all = length_all - rest
        tension = np.where(e_all > 0.0, np.maximum(k_c * e_all + c_c * rate_all, 0.0), 0.0)
        cable_force = tension[:, None] * u
        thrust = schedule[n].copy()
        thrust[i] = command
        force = np.empty_like(pos)
        force[0] = cable_force.sum(axis=0)
        force[1:] = thrust[:, None] * heading_unit[n] - cable_force
        force += weather[n] - drag * vel
        torque = float(np.sum(la[:, 0] * cable_force[:, 1] - la[:, 1] * cable_force[:, 0])) - constants.LOAD_ANGULAR_DRAG * load_omega
        vel = vel + STEP * force * inverse_mass
        load_omega = load_omega + STEP * torque / constants.LOAD_YAW_INERTIA
        pos = pos + STEP * vel
        load_theta = load_theta + STEP * load_omega
    if result["outcome"] == "reengaged" and not literal and result["T_peak_cf"] is None:
        result["T_peak_cf"] = float(peak)
    if trace:
        result["history"] = {"label": labels[: n + 1].copy(), "e": e_hist[: n + 1].copy(), "edot": edot_hist[: n + 1].copy()}
        if not literal:
            result["final"] = {"position": pos.copy(), "velocity": vel.copy(), "load_theta": load_theta, "load_omega": load_omega, "label": float(labels[n])}
    result.update(
        mode=mode,
        controller=controller,
        variant=None,
        delay=None if result["t_cf"] is None else float(result["t_cf"] - x.t_up),
        ratio=None if result["v_cf"] is None else float(result["v_cf"] / x.critical_speed),
        caught=bool(result["outcome"] == "reengaged" and result["v_cf"] < x.critical_speed),
        engaged_fraction=float(engaged_ticks / total_ticks) if total_ticks else 0.0,
        min_thrust=None if not math.isfinite(min_thrust) else float(min_thrust),
    )
    return result


def excursion_job(x: Excursion) -> dict:
    """Replay validation and every (mode, controller, variant) counterfactual of one excursion."""
    runs = []
    for mode in MODES:
        runs.append(integrate(x, mode, "replay", None))
        for controller in CONTROLLERS:
            if controller in ARMS and controller not in x.estimates:
                continue
            for name, fraction in VARIANTS.items():
                row = integrate(x, mode, controller, fraction)
                row["variant"] = name
                runs.append(row)
    return {"tags": x.tags, "runs": runs}


# ----------------------------------------------------------------------------- building the excursions


def _estimate_errors(arms_mission: dict, cable: int, slack_start: float, t_up: float, window: tuple[float, float]) -> dict:
    truth = arms_mission["monitor_truth"]
    out = {}
    for arm in ARMS:
        output = arms_mission["outputs"][arm]
        time = np.asarray(output.time)
        lo = int(max(0, math.floor(window[0] / MONITOR_PERIOD) - 2))
        hi = int(min(time.size, math.ceil(window[1] / MONITOR_PERIOD) + 2))
        slack = np.asarray(output.slack[lo:hi, cable], dtype=bool)
        e_err = np.asarray(output.e_hat[lo:hi, cable], float) - np.asarray(truth.e_true[lo:hi, cable], float)
        edot_err = np.asarray(output.edot_hat[lo:hi, cable], float) - np.asarray(truth.edot_true[lo:hi, cable], float)
        tick_time = time[lo:hi]
        usable = slack & np.isfinite(e_err) & np.isfinite(edot_err) & (tick_time >= slack_start - 1.0e-9) & (tick_time < t_up)
        held = None
        if usable.any():
            last = int(np.flatnonzero(usable)[-1])
            held = (float(e_err[last]), float(edot_err[last]))
        if abs(time[lo] - lo * MONITOR_PERIOD) > 1.0e-6:
            raise ValueError("monitor ticks are not on the 0.1 s grid")
        out[arm] = {"tick0": lo, "slack": slack, "e_err": e_err, "edot_err": edot_err, "held": held}
    return out


def build_excursions(missions: list[dict], arms: dict[int, dict], critical: np.ndarray) -> tuple[list[Excursion], dict]:
    from tether.campaign.mission import MissionSpec, mission_geometry, mission_schedules, mission_weather

    spec = MissionSpec()
    geometry = mission_geometry(spec)
    thrusts = mission_schedules(spec).thrusts.copy()
    envelope = _Envelope()
    excursions: list[Excursion] = []
    catalogue: dict = {"missions": [], "weather_sha256": {}}
    for mission in missions:
        seed = int(mission["seed"])
        truth = mission["truth"]
        marks = list(mission["marks"])
        events = declustered_events(marks)
        dangerous = [event for event in events if event["dangerous"]]
        crossing = first_crossing(truth.event_time, truth.elongation, truth.rate)
        kind, snap_mark = severance_type(crossing, marks)
        kind_wide, _ = severance_type(crossing, marks, SNAP_WINDOW_SENSITIVITY, before_only=True)
        first = min(dangerous, key=lambda event: event["t_up"]) if dangerous else None
        weather = mission_weather(spec, seed)
        catalogue["weather_sha256"][seed] = sha256_bytes(np.ascontiguousarray(weather).tobytes())
        entry = {
            "seed": seed,
            "events": len(events),
            "dangerous_events": len(dangerous),
            "dangerous_not_first_mark": sum(1 for event in dangerous if event["dangerous_index"] != 0),
            "first_crossing": crossing,
            "first_severance_type": kind,
            "first_severance_type_wide": kind_wide,
            "snap_mark_t_up": None if snap_mark is None else float(snap_mark[1]),
            "max_tension": float(mission["max_tension"]),
            "completed": bool(mission["outcome"].completed),
            "closure": mission["outcome"].closure,
        }
        catalogue["missions"].append(entry)
        state_time = np.asarray(truth.state_time)
        event_time = np.asarray(truth.event_time)
        for event in dangerous:
            cable = event["cable"]
            t_up, dwell = event["t_up"], event["dwell"]
            slack_start = t_up - dwell
            lo = int(math.ceil(slack_start / EVENT_PERIOD - 1.0e-9))
            hi = int(math.floor(t_up / EVENT_PERIOD + 1.0e-9))
            window = truth.elongation[lo : hi + 1, cable]
            deepest = lo + int(np.argmin(window))
            t_deep = float(event_time[deepest])
            k0 = int(math.floor(t_deep / STATE_PERIOD + 1.0e-9))
            k_end = min(state_time.size, k0 + int(round(HORIZON / STATE_PERIOD)) + 3)
            w0 = int(math.floor(state_time[k0] / STATE_PERIOD + 1.0e-9))
            log_start = max(0, int(round(state_time[k0] / EVENT_PERIOD)) - 400)
            log_end = min(event_time.size, int(round(state_time[k_end - 1] / EVENT_PERIOD)) + 2)
            is_first = first is not None and event is first
            literal_first = kind == "snap" and snap_mark is not None and int(snap_mark[0]) == cable and abs(float(snap_mark[1]) - t_up) < 1.0e-9
            tags = {
                "seed": seed,
                "cable": cable,
                "t_event": event["t_event"],
                "t_up": t_up,
                "dwell": dwell,
                "depth": event["depth"],
                "v_return": event["v_return"],
                "T_peak": event["T_peak"],
                "cluster_max": event["cluster_max"],
                "burst_marks": event["marks"],
                "dangerous_index": event["dangerous_index"],
                "critical_speed": float(critical[cable]),
                "t_turnaround": t_deep,
                "start_label": float(state_time[k0]),
                "first_severance": bool(is_first),
                "first_severance_plan_literal": bool(literal_first),
                "long_dwell": bool(dwell >= LONG_DWELL),
                "window": "pre_plateau" if t_up < PLATEAU[0] else ("plateau" if t_up < PLATEAU[1] else "post_plateau"),
            }
            excursions.append(
                Excursion(
                    seed=seed,
                    cable=cable,
                    t_up=t_up,
                    v_return=event["v_return"],
                    T_peak=event["T_peak"],
                    dwell=dwell,
                    depth=event["depth"],
                    critical_speed=float(critical[cable]),
                    thrusts=thrusts,
                    load_offsets=np.asarray(geometry.load_offsets, float),
                    vessel_offsets=np.asarray(geometry.vessel_offsets, float),
                    start_label=float(state_time[k0]),
                    state=np.asarray(truth.state[k0:k_end], float).copy(),
                    weather=np.asarray(weather[w0 : w0 + (k_end - k0) + 4], float).copy(),
                    weather_start=w0,
                    log_start=log_start,
                    e_log=np.asarray(truth.elongation[log_start:log_end, cable], float).copy(),
                    edot_log=np.asarray(truth.rate[log_start:log_end, cable], float).copy(),
                    estimates=_estimate_errors(arms[seed], cable, slack_start, t_up, (state_time[k0] - 0.3, state_time[k_end - 1])) if seed in arms else {},
                    tags=tags,
                    envelope=envelope,
                )
            )
    return excursions, catalogue


class _Envelope:
    """Picklable thrust envelope of the default mission."""

    def __call__(self, time):
        from tether.campaign.mission import MissionSpec, thrust_envelope

        return thrust_envelope(MissionSpec(), time)


# ----------------------------------------------------------------------------- tables and verdict


def _population(rows: list[dict], name: str) -> list[dict]:
    if name == "first_severance":
        return [r for r in rows if r["tags"]["first_severance"]]
    if name == "first_severance_plan_literal":
        return [r for r in rows if r["tags"]["first_severance_plan_literal"]]
    if name == "all_dangerous":
        return rows
    if name == "long_dwell":
        return [r for r in rows if r["tags"]["long_dwell"]]
    if name.startswith("long_dwell_"):
        window = name[len("long_dwell_") :]
        return [r for r in rows if r["tags"]["long_dwell"] and r["tags"]["window"] == window]
    raise ValueError(name)


POPULATIONS = ("first_severance", "first_severance_plan_literal", "all_dangerous", "long_dwell", "long_dwell_pre_plateau", "long_dwell_plateau", "long_dwell_post_plateau")


def _run(row: dict, mode: str, controller: str, variant: str | None) -> dict | None:
    for run in row["runs"]:
        if run["mode"] == mode and run["controller"] == controller and (variant is None or run["variant"] == variant):
            return run
    return None


def summarise(rows: list[dict], mode: str, controller: str, variant: str) -> dict:
    runs = [_run(r, mode, controller, variant) for r in rows]
    runs = [r for r in runs if r is not None]
    n = len(runs)
    caught = sum(r["caught"] for r in runs)
    closure = sum(r["outcome"] == "closure" for r in runs)
    horizon = sum(r["outcome"] == "horizon" for r in runs)
    reengaged = n - closure - horizon
    low, high = clopper_pearson(caught, n)
    ratios = [r["ratio"] for r in runs if r["ratio"] is not None]
    delays = [r["delay"] for r in runs if r["outcome"] == "reengaged"]
    return {
        "n": n,
        "caught": caught,
        "fraction": caught / n if n else None,
        "interval95": [low, high],
        "normal_half_width": 1.96 * math.sqrt(max(caught / n * (1 - caught / n), 0.0) / n) if n else None,
        "closure": closure,
        "no_reengagement": horizon,
        "fraction_excluding_closure_and_horizon": caught / reengaged if reengaged else None,
        "median_ratio": float(np.median(ratios)) if ratios else None,
        "p90_ratio": float(np.percentile(ratios, 90)) if ratios else None,
        "median_delay_s": float(np.median(delays)) if delays else None,
        "max_delay_s": float(np.max(delays)) if delays else None,
    }


def validation(rows: list[dict]) -> dict:
    out = {}
    for mode in MODES:
        dv, dt, dpeak, failed = [], [], [], []
        for row in rows:
            run = _run(row, mode, "replay", None)
            tags = row["tags"]
            if run["outcome"] != "reengaged":
                failed.append({"seed": tags["seed"], "cable": tags["cable"], "t_up": tags["t_up"], "outcome": run["outcome"]})
                continue
            dv.append(run["v_cf"] - tags["v_return"])
            dt.append(run["t_cf"] - tags["t_up"])
            if run.get("T_peak_cf") is not None:
                dpeak.append(run["T_peak_cf"] / tags["T_peak"] - 1.0)
        adv, adt = np.abs(dv), np.abs(dt)
        passed = not failed and bool(adv.size) and float(np.percentile(adv, 95)) <= VALIDATION_P95 and float(adv.max()) <= VALIDATION_MAX and float(adt.max()) <= VALIDATION_TIME
        out[mode] = {
            "n": len(rows),
            "not_reengaged": failed,
            "abs_dv_m_s": {"median": float(np.median(adv)), "p95": float(np.percentile(adv, 95)), "max": float(adv.max())} if adv.size else None,
            "dv_m_s_signed_median": float(np.median(dv)) if dv else None,
            "abs_dt_s": {"median": float(np.median(adt)), "max": float(adt.max())} if adt.size else None,
            "T_peak_relative_error": {"median_abs": float(np.median(np.abs(dpeak))), "p95_abs": float(np.percentile(np.abs(dpeak), 95)), "max_abs": float(np.max(np.abs(dpeak)))} if dpeak else None,
            "passed": bool(passed),
        }
    return out


def estimate_error_at_closure(rows: list[dict], excursions: dict) -> dict:
    out = {}
    for population in ("first_severance", "all_dangerous"):
        members = _population(rows, population)
        out[population] = {}
        for arm in ARMS:
            values, above = [], []
            for row in members:
                key = (row["tags"]["seed"], row["tags"]["cable"], row["tags"]["t_up"])
                held = excursions[key]
                if held.get(arm) is not None:
                    values.append(abs(held[arm][1]))
                    above.append(abs(held[arm][1]) > row["tags"]["critical_speed"])
            out[population][arm] = {
                "n": len(values),
                "missing": len(members) - len(values),
                "median_abs_edot_error_m_s": float(np.median(values)) if values else None,
                "p90_abs_edot_error_m_s": float(np.percentile(values, 90)) if values else None,
                "fraction_above_v_b": float(np.mean(above)) if above else None,
            }
    return out


def taut_overload_v1_phase6() -> dict:
    with V1_PHASE6_PATH.open("rb") as handle:
        cache = pickle.load(handle)
    items = [r for r in cache["results"] if r["arm"] == "N" and r["mode"] == "recording"]
    counts = {"snap": 0, "taut_overload": 0, "none": 0}
    wide = {"snap": 0, "taut_overload": 0, "none": 0}
    for item in items:
        crossing = item["virtual_first"]
        crossing = None if crossing is None else (float(crossing[0]), int(crossing[1]))
        marks = [(m[0], m[1]) for m in item["marks"]]
        counts[severance_type(crossing, marks)[0]] += 1
        wide[severance_type(crossing, marks, SNAP_WINDOW_SENSITIVITY, before_only=True)[0]] += 1
    return _overload_summary(len(items), counts, wide, float(cache["breaking"]))


def _overload_summary(missions: int, counts: dict, wide: dict, breaking: float) -> dict:
    severed = counts["snap"] + counts["taut_overload"]
    p_n = severed / missions
    floor = counts["taut_overload"] / missions
    return {
        "missions": missions,
        "breaking_N": breaking,
        "counts_50ms": counts,
        "counts_wide_0335s_before": wide,
        "p_N": p_n,
        "taut_overload_floor_p_arm": floor,
        "taut_overload_floor_interval95": list(clopper_pearson(counts["taut_overload"], missions)),
        "taut_overload_share_of_first_severances": counts["taut_overload"] / severed if severed else None,
        "ceiling_p_N_minus_p_arm": p_n - floor,
        "wide_floor_p_arm": wide["taut_overload"] / missions,
        "wide_ceiling_p_N_minus_p_arm": p_n - wide["taut_overload"] / missions,
    }


def verdict(tables: dict) -> dict:
    primary = tables[PRIMARY["mode"]][PRIMARY["controller"]][PRIMARY["population"]]
    admissible = {name: bool(primary[name]["fraction"] is not None and primary[name]["fraction"] >= ADMISSIBLE_FRACTION) for name in VARIANTS}
    if admissible["hold"]:
        choice, outcome = "hold (F_min = 0)", "ADMISSIBLE"
    elif admissible["reverse"]:
        choice, outcome = "reverse (F_min = -F_T)", "ADMISSIBLE"
    else:
        choice, outcome = None, "NO-LAUNCH"
    return {
        "rule": DECLARATIONS["admissibility_rule"],
        "fractions": {name: primary[name]["fraction"] for name in VARIANTS},
        "intervals95": {name: primary[name]["interval95"] for name in VARIANTS},
        "n": primary["hold"]["n"],
        "admissible": admissible,
        "F_min": choice,
        "outcome": outcome,
    }


# ----------------------------------------------------------------------------- commands


def declare() -> str:
    if DECLARATIONS_PATH.exists():
        raise SystemExit(f"{DECLARATIONS_PATH} exists; declarations are never edited - write a dated addendum")
    payload = dict(DECLARATIONS)
    payload["plan_sha256"] = sha256_file(PLAN_PATH)
    payload["input_sha256"] = {
        "missions": sha256_file(MISSIONS_PATH),
        "arms": sha256_file(ARMS_PATH),
        "impact_table": sha256_file(FAN_IMPACT_PATH),
        "v1_phase6": sha256_file(V1_PHASE6_PATH),
    }
    return write_bytes(DECLARATIONS_PATH, json_bytes(payload))


def write_addendum() -> str:
    for path, payload in ((ADDENDUM_PATH, ADDENDUM), (ADDENDUM_2_PATH, ADDENDUM_2)):
        if not path.exists():
            return write_bytes(path, json_bytes(payload))
    raise SystemExit("every addendum exists; addenda are never edited")


def run(workers: int = WORKERS) -> dict:
    if not DECLARATIONS_PATH.exists():
        raise SystemExit("declare first")
    declared = DECLARATIONS_PATH.read_bytes()
    import json

    if not ADDENDUM_PATH.exists() or json.loads(ADDENDUM_PATH.read_bytes()) != json.loads(json_bytes(ADDENDUM)):
        raise SystemExit("addendum 1 missing or different from the code's")
    addendum = ADDENDUM_PATH.read_bytes()
    if not ADDENDUM_2_PATH.exists() or json.loads(ADDENDUM_2_PATH.read_bytes()) != json.loads(json_bytes(ADDENDUM_2)):
        raise SystemExit("addendum 2 missing or different from the code's")
    addendum_2 = ADDENDUM_2_PATH.read_bytes()

    stored = json.loads(declared)
    for key, value in DECLARATIONS.items():
        if json.loads(json_bytes(value)) != stored.get(key):
            raise SystemExit(f"declaration '{key}' differs from the declared file; write an addendum")
    from tether.monitor.hazard import critical_speeds

    critical = critical_speeds(STRESS_THRESHOLD, PRETENSION, table_path=FAN_IMPACT_PATH)
    with MISSIONS_PATH.open("rb") as handle:
        missions = pickle.load(handle)
    with ARMS_PATH.open("rb") as handle:
        arms_list = pickle.load(handle)
    arms = {int(m["seed"]): m for m in arms_list}
    excursions, catalogue = build_excursions(missions, arms, critical)
    held = {(x.seed, x.cable, x.t_up): {arm: x.estimates[arm]["held"] for arm in x.estimates} for x in excursions}
    rows = run_pool(excursion_job, excursions, workers)
    tables = {
        mode: {
            controller: {population: {variant: summarise(_population(rows, population), mode, controller, variant) for variant in VARIANTS} for population in POPULATIONS}
            for controller in CONTROLLERS
        }
        for mode in MODES
    }
    check = validation(rows)
    decision = verdict(tables)
    counts = {"snap": 0, "taut_overload": 0, "none": 0}
    wide = {"snap": 0, "taut_overload": 0, "none": 0}
    for entry in catalogue["missions"]:
        counts[entry["first_severance_type"]] += 1
        wide[entry["first_severance_type_wide"]] += 1
    overload = {"phase5_40": _overload_summary(len(missions), counts, wide, STRESS_THRESHOLD), "v1_phase6_N_recording": taut_overload_v1_phase6()}
    per_excursion = []
    for row in rows:
        compact = {"tags": row["tags"], "runs": [{k: v for k, v in r.items()} for r in row["runs"]]}
        per_excursion.append(compact)
    results = {
        "schema_version": 1,
        "test": "P6-T0",
        "declarations_sha256": sha256_bytes(declared),
        "addendum_1_sha256": sha256_bytes(addendum),
        "addendum_2_sha256": sha256_bytes(addendum_2),
        "validation_by_mode": {"fleet": "PRIMARY (verdict table)", "literal": "reported; gates nothing"},
        "source": source_state(),
        "critical_speeds": critical.tolist(),
        "validation": check,
        "validation_passed": bool(check["fleet"]["passed"] and check["literal"]["passed"]),
        "tables": tables,
        "verdict": decision,
        "taut_overload": overload,
        "estimation_error_at_closure": estimate_error_at_closure(rows, held),
        "catalogue": catalogue,
        "population_sizes": {population: len(_population(rows, population)) for population in POPULATIONS},
        "excursions": per_excursion,
    }
    write_bytes(RESULTS_PATH, json_bytes(results))
    return results


MECHANISM_PATH = RECORD_DIR / "p6_t0_mechanism.json"


def mechanism() -> dict:
    """Post-hoc diagnostic (not declared, gates nothing): what drives each recorded return leg.

    Per dangerous excursion, over the recorded 10 ms rows from the turnaround to the last row
    before t_up: the hull-to-chord alignment cos(psi) of vessel i, the along-chord component of
    its weather force, the along-chord thrust F_T cos(psi), and the recorded closing speed at the
    last row split into the stern term v_stern . d and the load term -v_load . d (with their
    change since the turnaround).
    """
    from tether.campaign.mission import MissionSpec, mission_geometry
    from tether.monitor.hazard import critical_speeds
    from tether.physics.fleet import cable_kinematics

    geometry = mission_geometry(MissionSpec())
    critical = critical_speeds(STRESS_THRESHOLD, PRETENSION, table_path=FAN_IMPACT_PATH)
    with MISSIONS_PATH.open("rb") as handle:
        missions = pickle.load(handle)
    excursions, _ = build_excursions(missions, {}, critical)
    rows = []
    for x in excursions:
        i, base = x.cable, 3 * (x.thrusts.size + 1)
        last = int(math.floor((x.t_up - x.start_label) / STATE_PERIOD + 1.0e-9))
        cosines, wind, terms = [], [], []
        for k in range(last + 1):
            state = x.state[k]
            kin = cable_kinematics(state, geometry, constants.CABLE_REST_LENGTH)
            d = kin["direction"][i]
            heading = state[3 * (i + 1) + 2]
            cosines.append(math.cos(heading) * d[0] + math.sin(heading) * d[1])
            wind.append(float(x.weather[k, 1 + i] @ d))
            la, va = kin["load_arm"][i], kin["vessel_arm"][i]
            load_velocity = state[base : base + 2] + state[base + 2] * np.array([-la[1], la[0]])
            j = 3 * (i + 1)
            stern_velocity = state[base + j : base + j + 2] + state[base + j + 2] * np.array([-va[1], va[0]])
            terms.append((float(stern_velocity @ d), float(-load_velocity @ d)))
        cosines = np.array(cosines)
        rows.append({
            **{key: x.tags[key] for key in ("seed", "cable", "t_up", "window", "first_severance", "long_dwell")},
            "return_leg_s": float(x.t_up - x.tags["t_turnaround"]),
            "cos_psi_mean": float(cosines.mean()),
            "cos_psi_min": float(cosines.min()),
            "wind_along_chord_N": float(np.mean(wind)),
            "thrust_along_chord_N": float(x.thrusts[i] * cosines.mean()),
            "stern_term_at_contact_m_s": terms[-1][0],
            "load_term_at_contact_m_s": terms[-1][1],
            "stern_term_change_m_s": terms[-1][0] - terms[0][0],
            "load_term_change_m_s": terms[-1][1] - terms[0][1],
        })
    summary = {}
    for name, selector in (("first_severance", lambda r: r["first_severance"]), ("plateau", lambda r: r["window"] == "plateau"), ("post_plateau", lambda r: r["window"] == "post_plateau")):
        chosen = [r for r in rows if selector(r)]
        summary[name] = {"n": len(chosen)} | {
            f"median_{key}": float(np.median([r[key] for r in chosen]))
            for key in ("return_leg_s", "cos_psi_mean", "wind_along_chord_N", "thrust_along_chord_N", "stern_term_at_contact_m_s", "load_term_at_contact_m_s", "stern_term_change_m_s", "load_term_change_m_s")
        }
        summary[name]["hull_pointing_at_load"] = sum(r["cos_psi_mean"] < 0.0 for r in chosen)
    output = {
        "status": "post-hoc diagnostic of the P6-T0 records, written after the results; not declared, gates nothing",
        "squall_vessel_amplitude_N": 4613.494026427559,
        "summary": summary,
        "excursions": rows,
        "source": source_state(),
    }
    write_bytes(MECHANISM_PATH, json_bytes(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("declare", "addendum", "run", "mechanism"))
    parser.add_argument("--workers", type=int, default=WORKERS)
    arguments = parser.parse_args()
    if arguments.command == "declare":
        print("declarations sha256", declare())
    elif arguments.command == "addendum":
        print("addendum sha256", write_addendum())
    elif arguments.command == "mechanism":
        print(mechanism()["summary"])
    else:
        results = run(min(arguments.workers, WORKERS))
        print({"validation_passed": results["validation_passed"], "verdict": results["verdict"]})


if __name__ == "__main__":
    main()
