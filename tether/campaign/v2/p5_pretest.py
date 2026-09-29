"""P5-T2' - the oracle fleet-rollout pre-test (plan v2 II.9, IV.9, Phase 5 order of work (i)).

The fleet rollout (``tether.monitor.fleet_rollout``) is run on the oracle state - the true
plant state of the committed Phase 5 cache - at every oracle slack tick of every interval
that begins an event, with the class's own conditional weather law for the future, and its
forecast is scored against the realised label by pooled logistic recalibration with slack
intervals as clusters.  The single-cable linearised model of v1 is scored on exactly the
same ticks and labels.  Every operational choice is in ``DECLARATIONS`` and is written to
``records/v2/phase5/p5_t2prime_declarations.json`` before any result is computed.

Commands, in order: ``declare``, ``fidelity`` (validation replay), ``run`` (rollouts and
the linearised baseline), ``analyse`` (statistics, verdict, results file).  ``all`` runs
them in sequence.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time as wallclock
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tether.campaign.common import RECORDS, json_bytes, npz_bytes, run_pool, sha256_bytes, sha256_file, source_state, write_bytes

RECORD_DIR = RECORDS / "v2" / "phase5"
DECLARATIONS_PATH = RECORD_DIR / "p5_t2prime_declarations.json"
RESULTS_PATH = RECORD_DIR / "p5_t2prime_results.json"
TICKS_PATH = RECORD_DIR / "p5_t2prime_ticks.npz"
FIDELITY_PATH = RECORD_DIR / "p5_t2prime_fidelity.json"
RUN_CACHE = RECORD_DIR / "cache" / "p5_t2prime_run.pkl"
MISSION_CACHE = RECORDS / "phase5" / "cache" / "phase5_missions.pkl"
ARMS_CACHE = RECORDS / "phase5" / "cache" / "phase5_missions_arms.pkl"
FAN_IMPACT_PATH = RECORDS / "phase5" / "impact_table_fan.json"
V1_RESULTS_PATH = RECORDS / "phase5" / "phase5_results.json"

SEEDS = tuple(range(5001, 5041))
FIT_SEEDS = tuple(range(5001, 5021))
TEST_SEEDS = tuple(range(5021, 5041))
HORIZONS = (1.0, 2.0)
PRIMARY_HORIZON = 1.0
N_ROLLOUT = 2048
ROLLOUT_STEP = 1.0e-3
FUTURE_SAMPLES = 200
STRESS_THRESHOLD = 4500.0
PRETENSION = 1000.0
BURST_WINDOW = 2.0
COUPLING_TENSION = 1000.0
N_BOOT = 2000
FORECAST_CLIP = 0.5 / N_ROLLOUT
FIDELITY_STRIDE = 10
FIDELITY_HORIZON = 1.0
FIDELITY_TOLERANCE = {"e_p95_at_0.5s": 0.03, "edot_p95_at_0.5s": 0.05}
PRIMARY_MODE = "controlled"
SECONDARY_MODE = "held"
BOOT_ENTROPY = 20260913
REPORT_CODES = {
    "rollout_H1": 1, "rollout_H2": 2, "linearized_H1": 3, "linearized_H2": 4, "held_H1": 5,
    "rollout_H1_coupled": 6, "rollout_H1_uncoupled": 7, "linearized_H1_coupled": 8, "linearized_H1_uncoupled": 9,
    "rollout_H2_coupled": 10, "rollout_H2_uncoupled": 11, "recal_rollout_H1_test": 12, "recal_linearized_H1_test": 13,
    "recal_rollout_H2_test": 14,
}

DECLARATIONS = {
    "test": "P5-T2' oracle fleet-rollout pre-test (plan v2 II.9, IV.9, Part V Phase 5 order of work (i) and test table, Appendix D)",
    "declared_on": "2026-09-13",
    "data": {
        "missions": "records/phase5/cache/phase5_missions.pkl (truth: 10 ms plant state [q; v] with state_time[k] = k * 10 ms equal to the physics time, 1 ms elongation and rate, re-engagement marks); seeds 5001-5040, MissionSpec() defaults (Squall Passage, fan 0.55 rad, T0 = 1 kN, common-mode t3 background, 8 s weather ramp, squall level 1.5)",
        "ticks": "the 10 Hz monitor tick grid of the v1 Phase 5 arms (records/phase5/cache/phase5_missions_arms.pkl, outputs['P'].time) and the true elongation at those ticks (monitor_truth.e_true); the same cache supplies the v1 oracle precursor for the linearised baseline",
        "no_new_simulation": "no plant simulation, no estimator plumbing, no mission change (order of work (i))",
    },
    "stress_threshold": {"T_b_s": STRESS_THRESHOLD, "source": "v1 Phase 5 (records/phase5/phase5_results.json stress.threshold = 4500 N)"},
    "critical_speed": "v_b per cable = tether.monitor.hazard.critical_speeds(4500, 1000, table_path=records/phase5/impact_table_fan.json), as tether/campaign/phase5_analysis.py",
    "oracle_slack_ticks": "ticks t of the grid with true e_i(t) <= 0 (phase5_analysis._oracle_output's rule)",
    "slack_intervals": "maximal runs of consecutive oracle slack ticks of one cable, additionally split wherever a true re-engagement (1 ms upcrossing) of that cable lies between two consecutive slack ticks, so that every interval lies inside one true slack excursion; interval ids are unique over seeds and cables and are the calibration clusters",
    "reengagement": "a true upcrossing of the 1 ms elongation series e_k <= 0 < e_(k+1), located by linear interpolation (tether.monitor.outcomes.true_upcrossings). The tracker's marks (used only for the coupled flag, which needs T_peak) are the same crossings except the t = 0 start (every chord exactly at rest length), the ~1.2-1.7 s start-up crossing of a slack interval begun before the tracker's first taut sample, and crossings in the last ~0.1 s of a run whose peak was still pending (checked on the cache: 660 upcrossings, 200 cable-missions differ only in these)",
    "onset": "an interval's onset is the last true 1 ms downcrossing (e_k > 0 >= e_(k+1), linearly interpolated) at or before its first slack tick",
    "event_and_bounce_rule": "an event is the first re-engagement of a burst; marks within 2 s on the same cable belong to it (Appendix B.1). A slack interval is a bounce of an earlier event when a true re-engagement of the same cable lies in [onset - 2 s, onset); its ticks are excluded from calibration and counted. Every other interval begins an event, and its slack ticks form the calibration set (intervals that never re-engage are kept)",
    "label": "y = 1 when the first true upcrossing of e_i = 0 in [t, t + H] closes above v_b,i (tether.monitor.outcomes.tick_outcomes); ticks whose window runs past the truth series with no crossing seen are censored and left out, per horizon (as v1's monitor_table)",
    "horizons": {"primary": 1.0, "secondary": 2.0, "note": "H = 1 s is the horizon of the committed 0.45 baseline; one rollout of 2 s serves both horizons (the H = 1 s hazard is read from the same sample paths, first upcrossing in (0, 1 s])"},
    "rollout_model": {
        "module": "tether/monitor/fleet_rollout.py",
        "bodies": "planar load + five vessels on world planar joints, masses and yaw inertias pinned (tether/physics/constants.py)",
        "cables": "unilateral (k e + c edot)_+ while e > 0, k = 1.7e5 N/m, c = 1.8e3 N s/m, rest length 12 m, fan geometry tether.physics.fleet.formation_geometry('fan', arc_half_angle=0.55) (tether.campaign.mission.mission_geometry)",
        "drag": "linear drag on every body's translation (load 5.5e3, vessel 350 N s/m), angular drag on every yaw (load 2e4, vessel 800 N m s)",
        "thrust": "each vessel's surge force along its hull axis at the mission schedule's value tether.campaign.mission.mission_schedules(MissionSpec()).surge, under the controller's 20 ms zero-order hold (the plant step at an update instant uses the previous update's command)",
        "headings_primary": "controlled: vessel yaw integrated with the angular drag, the cable's stern moment and the IV.7 heading law k_h wrap(theta_ref(t) - theta) - k_c sin(bearing), k_h = 500, k_c = 100, evaluated on the true heading and true cable bearing under the 20 ms hold, theta_ref = the mission's pre-loaded heading schedule (dogleg included); the command in force at the start is the law evaluated on the starting state",
        "headings_secondary": "held: every vessel heading frozen at its true starting value with zero yaw rate for the whole horizon (the task's literal wording); run at H = 1 s on the same ticks with the same draws, reported, not gating",
        "why_primary_is_controlled": "declared before any hazard or label was computed: in the development replay of the recorded future (realised weather, 66 slack ticks of seeds 5001-5012, 1 ms step) held headings missed the recorded e_i trajectory by p95 0.22 m / 0.40 m/s at 0.5 s (max 0.32 m / 2.6 m/s), because a frozen hull removes the stern's rotational compliance (1/m_eff at the stern gains r^2 sin^2 / I_A with r = 1.5 m, I_A = 500 kg m^2, comparable to 1/m_A) and the thrust direction's response; the controlled model met p95 1.3 mm / 5 mm/s. The held model therefore fails the task's fidelity requirement and cannot carry the gate",
        "load_yaw": "integrated",
        "integrator": "symplectic (semi-implicit) Euler, the plant's own discrete scheme (Drake SAP without contact or joint damping), internal step 1 ms (<= 2.5 ms); the fidelity check also runs the plant's 0.5 ms step for comparison",
        "weather_grid": "forces on the 10 ms zero-order-hold grid of HullForces, sample k acting over [k T_w, (k + 1) T_w), index floor((t + 1e-9) / 10 ms)",
        "first_upcrossing": "first step pair of the watched cable with e_prev <= 0 < e_next in (0, H], time and closing speed by linear interpolation (the truth labels' rule on a finer grid)",
        "hazard": "fraction of the N rollouts whose first upcrossing in (0, H] has closing speed > v_b,i",
        "transmission": "re-engagements of other cables within H are simulated by the rollout itself (Prop. 13's transmission); no extra term",
        "closure": "formation closure is not modelled inside the rollout (chord floored at 1e-9 m)",
    },
    "weather_law": {
        "known_now": "the oracle knows the current 10 ms weather sample on every body (index k0 = t / 10 ms at a tick t), regenerated exactly with tether.campaign.mission.mission_weather(MissionSpec(), seed) = ramp(t) * background + squall (checked to 1e-9 N)",
        "background_future": "the mission background is common-mode AR(1) t3 with tau_w = 8 s, phi = exp(-0.01 / 8) = 0.99875: F_b[k] = ramp(t_k) sigma_b x_k with sigma_b = 3500 N (load) / 700 N (vessels) and the common standardized 2-D state x_k = phi x_(k-1) + sqrt(1 - phi^2) eps_k, eps_k = (Gaussian 2-vector) * sqrt(1 / chi^2_3) (tether.physics.weather). Future samples k0 + 1 ... k0 + 200 are drawn from this conditional law given x_k0 = background_load[k0] / 3500 N (every body gives the same x_k0 exactly)",
        "squall": "deterministic and known: tether.campaign.mission.squall_forces at the mission's calibration, zero beyond the weather array",
        "ramp": "the mission's 8 s raised-cosine weather ramp is applied to the sampled background as the generator does",
    },
    "monte_carlo": {
        "N": N_ROLLOUT,
        "generator": "tether.monitor.hazard.hazard_generator(seed, cable) = SeedSequence([seed, 17, cable]), one per (seed, cable), advanced only at the evaluated ticks of that cable, in time order",
        "evaluated_ticks": "every oracle slack tick of every non-bounce interval (the calibration set before per-horizon censoring)",
        "draw_order_per_tick": "Gaussians (N, 200, 2) then chi-square(3) factors (N, 200) (tether.monitor.fleet_rollout.common_ar1_future)",
        "same_draws_for_both_heading_modes": True,
    },
    "baseline": "the v1 single-cable linearised drag-damped model (tether/monitor/linearized.py fleet_hazard_linearized) on the v1 oracle precursor (phase5_analysis._oracle_output: true (e, edot) at the tick, a_hat fitted on the 10 ms grid from the 1 ms truth), N = 2048, generator [seed, 17, cable] advanced at every oracle slack tick exactly as v1, at H = 1 s and H = 2 s, then restricted to the same calibration ticks and labels. v1's own all-tick number (slope 0.456 at H = 1 s) is recomputed as a reproduction check with v1's tick-run intervals and rng default_rng(20260914)",
    "gate": {
        "condition": "the pooled logistic recalibration-slope 95% interval of the fleet rollout (controlled headings, 1 ms step) at H = 1 s, over the calibration ticks of all 40 seeds and 5 cables, contains 1",
        "statistic": "tether.monitor.metrics.calibration_report(forecast, label, cluster = slack interval id, n_bins = 10, n_boot = 2000, clip = 1/(2 N), confidence = 0.95); the interval is its cluster-bootstrap percentile interval (slope_lo, slope_hi)",
        "degenerate": "a separated fit (no finite MLE) or a non-finite interval does not meet the condition; it is reported with its reason",
        "pass": "branch (ii): packets grow, arms L, B2, P run the rollout",
        "fail": "branch (iii): Prop. 10' withdrawn; the hazard is judged on discrimination plus the declared post-hoc recalibration fitted on seeds 5001-5020 and tested on seeds 5021-5040, reported as 'calibrated by recalibration', never 'calibrated'",
        "bootstrap_rng": f"np.random.default_rng(np.random.SeedSequence([{BOOT_ENTROPY}, code])) with one code per report: {REPORT_CODES}",
    },
    "reported_not_gating": {
        "items": ["calibration-in-the-large intercept and interval", "ECE and interval", "number of the ten equal-width bins outside the simultaneous (Bonferroni, Kish-effective Wilson) band", "AUROC and interval", "top-decile and top-three-decile ratios", "coupled/uncoupled split", "top-bin coincidence counts", "H = 2 s", "held-heading secondary", "linearised baseline on the same ticks"],
        "coupled": "a calibration tick (t, i) is coupled when another cable j != i has a true re-engagement mark with T_peak > 1 kN and t_up in [t, t + H]; the split reports each subset's calibration_report and mean |y - h|",
        "top_bin_coincidence": "top reliability bin h in [0.9, 1.0]: ticks, misses (y = 0) and misses that are coupled; top decile by forecast value (h >= its 90th percentile): ticks, misses, coupled misses; the mirror counts for the bottom bin h < 0.1 with y = 1",
    },
    "recalibration_branch_iii": "computed in every case and reported: logit p' = a + b logit(clip(h)) fitted by maximum likelihood (tether.evt.calibration.logistic_recalibration, forecasts clipped to [1/(2N), 1 - 1/(2N)]) on the calibration ticks of seeds 5001-5020, applied unchanged to seeds 5021-5040 and scored there by calibration_report; for the rollout at H = 1 s (primary) and H = 2 s and for the linearised baseline at H = 1 s",
    "fidelity_validation": {
        "when": "before the pre-test run; reported in the results file",
        "ticks": "within each (seed, cable), the calibration ticks at positions 0, 10, 20, ... in time order whose 1 s window lies inside the truth series",
        "replay": "one rollout (N = 1) from the true state with the weather future set to the realised mission weather, 1 s, recording e and edot of every cable every 10 ms; heading modes controlled (1 ms and 0.5 ms steps) and held (1 ms)",
        "metrics": "|e_replay - e_true| and |edot_replay - edot_true| of the watched cable at 0.2, 0.5 and 1.0 s and their maxima over [0, 0.5] and [0, 1] s (median, p95, max over ticks); agreement of the replay's first upcrossing in (0, 1 s] with the truth's (crossed or not), closing-speed error, and label agreement at v_b",
        "tolerance": f"primary (controlled, 1 ms): p95 of |delta e| at 0.5 s <= {FIDELITY_TOLERANCE['e_p95_at_0.5s']} m and p95 of |delta edot| at 0.5 s <= {FIDELITY_TOLERANCE['edot_p95_at_0.5s']} m/s; if missed, the verdict is reported as untrusted pending diagnosis and no branch is taken on it",
    },
    "prediction": "none committed: the plan states the outcome is genuinely uncertain (v1 oracle slope 0.45 all 40 missions, 0.38 on the 12-mission subset; an impulse-coupled prototype moved 0.377 -> 0.416)",
    "workers": 8,
}


# ----------------------------------------------------------------------------- declarations


def write_declarations() -> str:
    """Write the declarations once; refuse to change them after they exist."""
    payload = json_bytes(DECLARATIONS)
    if DECLARATIONS_PATH.exists():
        if DECLARATIONS_PATH.read_bytes() != payload:
            raise RuntimeError("declarations already exist and differ; write a dated addendum instead of editing them")
        return sha256_bytes(payload)
    return write_bytes(DECLARATIONS_PATH, payload)


def declarations_sha256() -> str:
    return sha256_file(DECLARATIONS_PATH)


# ----------------------------------------------------------------------------- data layer


def critical_speeds() -> np.ndarray:
    from tether.monitor.hazard import critical_speeds as speeds

    return speeds(STRESS_THRESHOLD, PRETENSION, table_path=FAN_IMPACT_PATH)


def _downcrossings(time: np.ndarray, elongation: np.ndarray) -> np.ndarray:
    k = np.flatnonzero((elongation[:-1] > 0.0) & (elongation[1:] <= 0.0))
    fraction = elongation[k] / (elongation[k] - elongation[k + 1])
    return time[k] + fraction * (time[k + 1] - time[k])


@dataclass
class CableIntervals:
    """Slack intervals of one cable over the ticks: first/last tick index, onset, bounce."""

    first: np.ndarray
    last: np.ndarray
    onset: np.ndarray
    bounce: np.ndarray
    previous_reengagement: np.ndarray


def cable_intervals(ticks: np.ndarray, slack: np.ndarray, series_time: np.ndarray, elongation: np.ndarray, rate: np.ndarray,
                    burst: float = BURST_WINDOW) -> CableIntervals:
    """Declared intervals and the bounce rule for one cable (ticks (m,), slack (m,), 1 ms series)."""
    from tether.monitor.outcomes import true_upcrossings

    ups = true_upcrossings(series_time, elongation, rate).time
    downs = _downcrossings(series_time, elongation)
    index = np.flatnonzero(slack)
    firsts, lasts = [], []
    if index.size:
        start = index[0]
        for a, b in zip(index[:-1], index[1:]):
            broken = b != a + 1
            if not broken:
                # a true re-engagement between two consecutive slack ticks splits the run
                lo = np.searchsorted(ups, ticks[a], side="right")
                broken = lo < ups.size and ups[lo] <= ticks[b]
            if broken:
                firsts.append(start)
                lasts.append(a)
                start = b
        firsts.append(start)
        lasts.append(index[-1])
    first = np.asarray(firsts, dtype=np.intp)
    last = np.asarray(lasts, dtype=np.intp)
    onset = np.full(first.size, np.nan)
    bounce = np.zeros(first.size, dtype=bool)
    previous = np.full(first.size, np.nan)
    for row, tick in enumerate(first):
        position = np.searchsorted(downs, ticks[tick], side="right") - 1
        onset[row] = downs[position] if position >= 0 else series_time[0]
        before = np.searchsorted(ups, onset[row], side="left") - 1
        if before >= 0:
            previous[row] = ups[before]
            bounce[row] = onset[row] - ups[before] <= burst
    return CableIntervals(first, last, onset, bounce, previous)


def load_missions() -> list[dict]:
    """Per seed: ticks, true e at the ticks, 1 ms series, 10 ms state, marks (main process)."""
    with MISSION_CACHE.open("rb") as handle:
        missions = {m["seed"]: m for m in pickle.load(handle)}
    with ARMS_CACHE.open("rb") as handle:
        arms = {m["seed"]: m for m in pickle.load(handle)}
    rows = []
    for seed in SEEDS:
        truth = missions[seed]["truth"]
        arm = arms[seed]
        ticks = np.asarray(arm["outputs"]["P"].time, dtype=float)
        e_tick = np.asarray(arm["monitor_truth"].e_true, dtype=float)
        if not np.array_equal(np.asarray(arm["elongation"]), np.asarray(truth.elongation)):
            raise RuntimeError("the two Phase 5 caches disagree on the truth")
        rows.append({
            "seed": seed, "ticks": ticks, "e_tick": e_tick, "series_time": np.asarray(truth.event_time), "elongation": np.asarray(truth.elongation),
            "rate": np.asarray(truth.rate), "state_time": np.asarray(truth.state_time), "state": np.asarray(truth.state),
            "marks": np.array([(m.cable, m.t_up, m.T_peak) for m in truth.marks], dtype=float).reshape(-1, 3), "arm": arm,
        })
    return rows


def tick_table(missions: list[dict], v_b: np.ndarray) -> dict:
    """Every oracle slack tick with its interval, bounce flag, per-horizon label/censoring/coupling."""
    from tether.monitor.outcomes import tick_outcomes

    columns = {k: [] for k in ("seed", "cable", "tick", "time", "interval", "bounce", "onset", "state_index")}
    for h in HORIZONS:
        for k in ("label", "censored", "coupled", "crossed", "closing_speed"):
            columns[f"{k}_H{h:g}"] = []
    interval_id = 0
    for mission in missions:
        ticks, e_tick = mission["ticks"], mission["e_tick"]
        slack = e_tick <= 0.0
        outcomes = {h: tick_outcomes(ticks, slack, mission["series_time"], mission["elongation"], mission["rate"], v_b, horizon=h) for h in HORIZONS}
        marks = mission["marks"]
        state_index = np.round(ticks / 0.01).astype(int)
        if np.max(np.abs(mission["state_time"][np.minimum(state_index, mission["state_time"].size - 1)] - ticks)) > 1e-9:
            raise RuntimeError("state log does not cover the ticks")
        for cable in range(slack.shape[1]):
            intervals = cable_intervals(ticks, slack[:, cable], mission["series_time"], mission["elongation"][:, cable], mission["rate"][:, cable])
            other = marks[(marks[:, 0] != cable) & (marks[:, 2] > COUPLING_TENSION), 1]
            other.sort()
            for row in range(intervals.first.size):
                for tick in range(intervals.first[row], intervals.last[row] + 1):
                    t = ticks[tick]
                    columns["seed"].append(mission["seed"])
                    columns["cable"].append(cable)
                    columns["tick"].append(tick)
                    columns["time"].append(t)
                    columns["interval"].append(interval_id)
                    columns["bounce"].append(bool(intervals.bounce[row]))
                    columns["onset"].append(intervals.onset[row])
                    columns["state_index"].append(state_index[tick])
                    for h in HORIZONS:
                        o = outcomes[h]
                        columns[f"label_H{h:g}"].append(bool(o.label[tick, cable]))
                        columns[f"censored_H{h:g}"].append(bool(o.censored[tick, cable]))
                        columns[f"crossed_H{h:g}"].append(bool(o.crossed[tick, cable]))
                        columns[f"closing_speed_H{h:g}"].append(float(o.closing_speed[tick, cable]))
                        lo = np.searchsorted(other, t - 1e-12, side="left")
                        columns[f"coupled_H{h:g}"].append(bool(lo < other.size and other[lo] <= t + h))
                interval_id += 1
    return {k: np.asarray(v) for k, v in columns.items()}


# ----------------------------------------------------------------------------- weather and rollout jobs


def mission_forcing(seed: int):
    """(background (S, 6, 2) pre-ramp, ramp (S,), squall (S, 6, 2), full mission weather)."""
    from tether.campaign.mission import MissionSpec, calibrate_squall, mission_weather, raised_cosine, squall_forces
    from tether.physics import constants
    from tether.physics.weather import stationary_weather_forces

    spec = MissionSpec()
    background = stationary_weather_forces(seed, spec.duration, distribution=spec.weather_distribution, direction=spec.weather_direction, scale=spec.weather_scale)
    ramp = raised_cosine(np.arange(background.shape[0]) * constants.WEATHER_PERIOD / spec.weather_ramp_duration)
    squall = squall_forces(spec, calibrate_squall(spec), background.shape[0])
    weather = mission_weather(spec, seed)
    if np.max(np.abs(weather - (background * ramp[:, None, None] + squall))) > 1e-9:
        raise RuntimeError("mission weather decomposition does not reproduce mission_weather")
    return background, ramp, squall, weather


def body_std() -> np.ndarray:
    from tether.campaign.mission import MissionSpec
    from tether.physics import constants

    return MissionSpec().weather_scale * np.array([constants.LOAD_WEATHER_STD] + [constants.VESSEL_WEATHER_STD] * constants.VESSEL_COUNT)


def _extend(array: np.ndarray, first: int, count: int, fill: float = 0.0) -> np.ndarray:
    """array[first : first + count] padded with ``fill`` beyond the end."""
    part = array[first : first + count]
    if part.shape[0] < count:
        pad = np.full((count - part.shape[0],) + array.shape[1:], fill)
        part = np.concatenate([part, pad])
    return part


@dataclass(frozen=True)
class RolloutJob:
    seed: int
    cable: int
    times: np.ndarray
    states: np.ndarray
    mode: str
    horizon: float
    v_b: float

    @property
    def cost(self) -> float:
        return float(self.times.size * self.horizon)


def rollout_job(job: RolloutJob) -> dict:
    from tether.campaign.mission import MissionSpec, mission_geometry, mission_schedules, raised_cosine
    from tether.monitor.fleet_rollout import RolloutModel, WeatherFuture, common_ar1_future, rollout, thrust_on_steps
    from tether.monitor.hazard import hazard_generator
    from tether.physics import constants
    from tether.physics.weather import ar1_coefficient

    spec = MissionSpec()
    schedules = mission_schedules(spec)
    model = RolloutModel(mission_geometry(spec), step=ROLLOUT_STEP, headings=job.mode)
    background, _, squall, _ = mission_forcing(job.seed)
    std = body_std()
    phi = ar1_coefficient()
    rng = hazard_generator(job.seed, job.cable)
    steps = int(round(job.horizon / ROLLOUT_STEP))
    out = {k: np.full(job.times.size, np.nan) for k in ("h_H1", "h_H2", "p_cross_H1", "p_cross_H2", "steps")}
    started = wallclock.perf_counter()
    for row, (t, state) in enumerate(zip(job.times, job.states)):
        k0 = int(round(t / constants.WEATHER_PERIOD))
        current = background[k0, 0] / std[0]
        common = common_ar1_future(rng, N_ROLLOUT, current, FUTURE_SAMPLES, phi, spec.weather_distribution)
        sample_times = (k0 + np.arange(FUTURE_SAMPLES + 1)) * constants.WEATHER_PERIOD
        scale = raised_cosine(sample_times / spec.weather_ramp_duration)[:, None] * std[None, :]
        weather = WeatherFuture(k0, common, scale, _extend(squall, k0, FUTURE_SAMPLES + 1))
        thrust = thrust_on_steps(schedules.surge, t, steps, ROLLOUT_STEP)
        result = rollout(model, state, t, job.horizon, job.cable, weather, thrust, heading_reference=schedules.heading)
        for h in HORIZONS:
            if h <= job.horizon + 1e-12:
                out[f"h_H{h:g}"][row] = result.hazard(job.v_b, h)
                out[f"p_cross_H{h:g}"][row] = float(np.mean(result.crossed & (result.crossing_time <= h + 1e-9)))
        out["steps"][row] = result.steps_taken
    out.update(seed=job.seed, cable=job.cable, mode=job.mode, wall_seconds=wallclock.perf_counter() - started)
    return out


def linearized_job(payload) -> dict:
    """v1's linearised oracle hazard at every oracle slack tick of one mission, both horizons."""
    from tether.campaign.phase5_analysis import _oracle_output
    from tether.monitor.linearized import fleet_hazard_linearized

    arm, v_b = payload
    output = _oracle_output(arm)
    hazards = {f"H{h:g}": fleet_hazard_linearized(output, v_b, arm["seed"], n_samples=N_ROLLOUT, horizon=h) for h in HORIZONS}
    return {"seed": arm["seed"], "hazards": hazards, "onset_time": np.asarray(output.onset_time), "slack": np.asarray(output.slack)}


# ----------------------------------------------------------------------------- fidelity


@dataclass(frozen=True)
class FidelityJob:
    seed: int
    ticks: tuple  # (cable, time, state_index)
    states: np.ndarray
    series_start: np.ndarray  # 1 ms index of each tick
    e_true: np.ndarray  # (ticks, samples) watched cable, 10 ms instants over 1 s
    edot_true: np.ndarray
    v_b: np.ndarray
    truth_cross: np.ndarray  # (ticks, 2): crossed flag, closing speed of the first true upcrossing in [t, t + 1]

    @property
    def cost(self) -> float:
        return float(len(self.ticks))


FIDELITY_VARIANTS = (("controlled", 1.0e-3), ("controlled", 5.0e-4), ("held", 1.0e-3))


def fidelity_job(job: FidelityJob) -> dict:
    from tether.campaign.mission import MissionSpec, mission_geometry, mission_schedules
    from tether.monitor.fleet_rollout import RolloutModel, fixed_weather, rollout, thrust_on_steps
    from tether.physics import constants

    spec = MissionSpec()
    schedules = mission_schedules(spec)
    _, _, _, weather = mission_forcing(job.seed)
    samples = int(round(FIDELITY_HORIZON / constants.WEATHER_PERIOD))
    rows = {}
    for mode, step in FIDELITY_VARIANTS:
        model = RolloutModel(mission_geometry(spec), step=step, headings=mode)
        errors_e, errors_v, crossings = [], [], []
        steps = int(round(FIDELITY_HORIZON / step))
        per = int(round(constants.WEATHER_PERIOD / step))
        for row, (cable, t, k0) in enumerate(job.ticks):
            future = fixed_weather(k0, _clamped(weather, k0, samples + 2))
            thrust = thrust_on_steps(schedules.surge, t, steps, step)
            result = rollout(model, job.states[row], t, FIDELITY_HORIZON, cable, future, thrust, heading_reference=schedules.heading,
                             record_steps=np.arange(0, steps + 1, per), stop_when_all_crossed=False)
            errors_e.append(np.abs(result.record_elongation[:, 0, cable] - job.e_true[row]))
            errors_v.append(np.abs(result.record_rate[:, 0, cable] - job.edot_true[row]))
            crossings.append((bool(result.crossed[0]), float(result.closing_speed[0])))
        rows[f"{mode}_{step:g}"] = {"e": np.array(errors_e), "edot": np.array(errors_v), "crossing": np.array(crossings, dtype=float)}
    return {"seed": job.seed, "ticks": np.array(job.ticks, dtype=float), "truth_cross": job.truth_cross, "v_b": job.v_b, "rows": rows}


def _clamped(weather: np.ndarray, first: int, count: int) -> np.ndarray:
    """The plant's clamped index beyond the weather array (HullForces)."""
    index = np.minimum(first + np.arange(count), weather.shape[0] - 1)
    return weather[index]


def fidelity_jobs(missions: list[dict], table: dict, v_b: np.ndarray) -> list[FidelityJob]:
    from tether.monitor.outcomes import first_upcrossing, true_upcrossings

    jobs = []
    calibration = ~table["bounce"]
    for mission in missions:
        seed = mission["seed"]
        series_end = mission["series_time"][-1]
        chosen = []
        for cable in range(5):
            rows = np.flatnonzero(calibration & (table["seed"] == seed) & (table["cable"] == cable))
            rows = rows[np.argsort(table["time"][rows], kind="stable")]
            for row in rows[::FIDELITY_STRIDE]:
                if table["time"][row] + FIDELITY_HORIZON <= series_end + 1e-9:
                    chosen.append(row)
        if not chosen:
            continue
        ticks, states, starts, e_true, edot_true, cross = [], [], [], [], [], []
        grid = np.arange(0, int(round(FIDELITY_HORIZON / 0.001)) + 1, 10)
        for row in chosen:
            cable = int(table["cable"][row])
            t = float(table["time"][row])
            k0 = int(table["state_index"][row])
            start = int(round(t / 0.001))
            ticks.append((cable, t, k0))
            states.append(mission["state"][k0])
            starts.append(start)
            e_true.append(mission["elongation"][start + grid, cable])
            edot_true.append(mission["rate"][start + grid, cable])
            ups = true_upcrossings(mission["series_time"], mission["elongation"][:, cable], mission["rate"][:, cable])
            hit, _, speed, _ = first_upcrossing(ups, np.array([t]), np.array([t + FIDELITY_HORIZON]))
            cross.append((float(hit[0]), float(speed[0])))
        jobs.append(FidelityJob(seed, tuple(ticks), np.array(states), np.array(starts), np.array(e_true), np.array(edot_true), v_b, np.array(cross)))
    return jobs


def summarize_fidelity(items: list[dict]) -> dict:
    out = {}
    grid_index = {"0.2": 20, "0.5": 50, "1.0": 100}

    def stats_of(values):
        values = np.asarray(values, dtype=float)
        return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": float(np.max(values)), "count": int(values.size)}

    for variant in items[0]["rows"]:
        e = np.concatenate([item["rows"][variant]["e"] for item in items])
        v = np.concatenate([item["rows"][variant]["edot"] for item in items])
        crossing = np.concatenate([item["rows"][variant]["crossing"] for item in items])
        truth = np.concatenate([item["truth_cross"] for item in items])
        cables = np.concatenate([item["ticks"][:, 0].astype(int) for item in items])
        v_b = items[0]["v_b"][cables]
        entry = {}
        for label, column in grid_index.items():
            entry[f"abs_e_at_{label}s"] = stats_of(e[:, column])
            entry[f"abs_edot_at_{label}s"] = stats_of(v[:, column])
        entry["max_abs_e_over_0.5s"] = stats_of(e[:, : grid_index["0.5"] + 1].max(axis=1))
        entry["max_abs_edot_over_0.5s"] = stats_of(v[:, : grid_index["0.5"] + 1].max(axis=1))
        entry["max_abs_e_over_1.0s"] = stats_of(e.max(axis=1))
        entry["max_abs_edot_over_1.0s"] = stats_of(v.max(axis=1))
        replay_crossed = crossing[:, 0] > 0.5
        truth_crossed = truth[:, 0] > 0.5
        both = replay_crossed & truth_crossed
        entry["crossing_agreement"] = float(np.mean(replay_crossed == truth_crossed))
        entry["crossing_disagreements"] = int(np.sum(replay_crossed != truth_crossed))
        entry["closing_speed_error"] = stats_of(np.abs(crossing[both, 1] - truth[both, 1])) if both.any() else None
        replay_label = replay_crossed & (crossing[:, 1] > v_b)
        truth_label = truth_crossed & (truth[:, 1] > v_b)
        entry["label_agreement"] = float(np.mean(replay_label == truth_label))
        entry["label_disagreements"] = int(np.sum(replay_label != truth_label))
        entry["truth_positive_labels"] = int(np.sum(truth_label))
        out[variant] = entry
    primary = out[f"{PRIMARY_MODE}_{ROLLOUT_STEP:g}"]
    met = primary["abs_e_at_0.5s"]["p95"] <= FIDELITY_TOLERANCE["e_p95_at_0.5s"] and primary["abs_edot_at_0.5s"]["p95"] <= FIDELITY_TOLERANCE["edot_p95_at_0.5s"]
    return {"variants": out, "tolerance": FIDELITY_TOLERANCE, "primary_variant": f"{PRIMARY_MODE}_{ROLLOUT_STEP:g}", "primary_met": bool(met),
            "n_ticks": int(sum(len(item["ticks"]) for item in items))}


# ----------------------------------------------------------------------------- orchestration


def _report_dict(report) -> dict:
    from tether.campaign.phase5_analysis import _report_dict as convert

    return convert(report)


def _boot_rng(name: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([BOOT_ENTROPY, REPORT_CODES[name]]))


def run_fidelity(workers: int) -> dict:
    v_b = critical_speeds()
    missions = load_missions()
    table = tick_table(missions, v_b)
    jobs = fidelity_jobs(missions, table, v_b)
    items = run_pool(fidelity_job, jobs, workers)
    summary = summarize_fidelity(items)
    summary["declarations_sha256"] = declarations_sha256()
    write_bytes(FIDELITY_PATH, json_bytes(summary))
    return summary


def run_compute(workers: int) -> dict:
    v_b = critical_speeds()
    missions = load_missions()
    table = tick_table(missions, v_b)
    evaluated = ~table["bounce"]
    jobs = []
    for mode, horizon in ((PRIMARY_MODE, max(HORIZONS)), (SECONDARY_MODE, PRIMARY_HORIZON)):
        for mission in missions:
            for cable in range(5):
                rows = np.flatnonzero(evaluated & (table["seed"] == mission["seed"]) & (table["cable"] == cable))
                rows = rows[np.argsort(table["time"][rows], kind="stable")]
                if rows.size == 0:
                    continue
                jobs.append(RolloutJob(mission["seed"], cable, table["time"][rows].copy(), mission["state"][table["state_index"][rows]].copy(), mode, horizon, float(v_b[cable])))
    started = wallclock.perf_counter()
    linear = run_pool(linearized_job, [(m["arm"], v_b) for m in missions], workers)
    rollouts = run_pool(rollout_job, jobs, workers, progress=lambda done, total: print(f"rollout jobs {done}/{total} ({wallclock.perf_counter() - started:.0f} s)", flush=True) if done % 10 == 0 or done == total else None)
    payload = {"table": table, "v_b": v_b, "rollouts": rollouts, "linear": linear, "wall_seconds": wallclock.perf_counter() - started,
               "declarations_sha256": declarations_sha256()}
    RUN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with RUN_CACHE.open("wb") as handle:
        pickle.dump(payload, handle)
    return payload


def _assemble(payload: dict) -> dict:
    """Per-tick forecasts aligned with the tick table (NaN where not evaluated)."""
    table = payload["table"]
    size = table["time"].size
    forecasts = {name: np.full(size, np.nan) for name in ("rollout_H1", "rollout_H2", "held_H1", "linearized_H1", "linearized_H2", "p_cross_H1", "p_cross_H2")}
    for job_result in payload["rollouts"]:
        seed, cable, mode = job_result["seed"], job_result["cable"], job_result["mode"]
        rows = np.flatnonzero(~table["bounce"] & (table["seed"] == seed) & (table["cable"] == cable))
        rows = rows[np.argsort(table["time"][rows], kind="stable")]
        if mode == PRIMARY_MODE:
            forecasts["rollout_H1"][rows] = job_result["h_H1"]
            forecasts["rollout_H2"][rows] = job_result["h_H2"]
            forecasts["p_cross_H1"][rows] = job_result["p_cross_H1"]
            forecasts["p_cross_H2"][rows] = job_result["p_cross_H2"]
        else:
            forecasts["held_H1"][rows] = job_result["h_H1"]
    for item in payload["linear"]:
        rows = np.flatnonzero(table["seed"] == item["seed"])
        for h in HORIZONS:
            hazard = item["hazards"][f"H{h:g}"]
            forecasts[f"linearized_H{h:g}"][rows] = hazard[table["tick"][rows], table["cable"][rows]]
    return forecasts


def _calibration(name: str, forecast, label, cluster) -> dict | None:
    from tether.monitor.metrics import calibration_report

    forecast = np.asarray(forecast, dtype=float)
    label = np.asarray(label, dtype=bool)
    if forecast.size == 0 or not label.any() or label.all():
        return None
    report = calibration_report(forecast, label, cluster, n_bins=10, n_boot=N_BOOT, clip=FORECAST_CLIP, confidence=0.95, rng=_boot_rng(name))
    out = _report_dict(report)
    out["mean_abs_residual"] = float(np.mean(np.abs(label.astype(float) - forecast)))
    return out


def _coincidence(forecast: np.ndarray, label: np.ndarray, coupled: np.ndarray) -> dict:
    top = forecast >= 0.9
    decile = forecast >= np.quantile(forecast, 0.9)
    bottom = forecast < 0.1
    return {
        "top_bin_ticks": int(top.sum()), "top_bin_misses": int(np.sum(top & ~label)), "top_bin_misses_coupled": int(np.sum(top & ~label & coupled)),
        "top_decile_threshold": float(np.quantile(forecast, 0.9)), "top_decile_ticks": int(decile.sum()), "top_decile_misses": int(np.sum(decile & ~label)),
        "top_decile_misses_coupled": int(np.sum(decile & ~label & coupled)),
        "bottom_bin_ticks": int(bottom.sum()), "bottom_bin_events": int(np.sum(bottom & label)), "bottom_bin_events_coupled": int(np.sum(bottom & label & coupled)),
    }


ADDENDUM_PATH = RECORD_DIR / "p5_t2prime_addendum_2026-09-13.json"
ADDENDUM = {
    "date": "2026-09-13",
    "written": "after the results existed; the declarations file is unchanged",
    "what": "post-hoc degeneracy diagnostics added to the results file under 'post_hoc_diagnostics' for every scored forecast: the number of calibration ticks with h = 0 and h = 1 exactly and the labels they carry, the ticks and clusters with 0 < h < 1 and their mean forecast and outcome by bin, the tick-level recalibration slope (maximum likelihood, Wald interval) under forecast clips 1/(2N), 1e-3 and 1e-2, and the slope fitted on the 0 < h < 1 ticks alone",
    "why": "the rollout at H = 1 s forecast h = 0 or h = 1 exactly on 9054 of 9227 calibration ticks, all of them correctly, so the pooled slope is identified only by a few intermediate ticks and by the clip convention; the diagnostics show how much of the gating statistic that is",
    "effect_on_gate": "none: the gate, its statistic, its interval and the verdict are those declared and are not recomputed or re-read",
}


def _degeneracy(forecast: np.ndarray, label: np.ndarray, cluster: np.ndarray) -> dict:
    """Post-hoc (addendum 2026-09-13): how much of the forecast is exactly 0 or 1 and what the
    recalibration slope rests on."""
    from tether.evt.calibration import logistic_recalibration

    y = label.astype(float)
    zero, one = forecast == 0.0, forecast == 1.0
    middle = ~zero & ~one
    out = {"ticks": int(forecast.size), "h_zero": int(zero.sum()), "h_zero_events": int(label[zero].sum()), "h_one": int(one.sum()),
           "h_one_nonevents": int((~label[one]).sum()), "intermediate": int(middle.sum()), "intermediate_clusters": int(np.unique(cluster[middle]).size),
           "intermediate_mean_forecast": float(forecast[middle].mean()) if middle.any() else None,
           "intermediate_mean_outcome": float(y[middle].mean()) if middle.any() else None, "intermediate_bins": [], "clip_sensitivity": {}}
    for lo, hi in ((0.0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.0)):
        inside = middle & (forecast >= lo) & (forecast < hi)
        out["intermediate_bins"].append({"lo": lo, "hi": hi, "ticks": int(inside.sum()), "mean_forecast": float(forecast[inside].mean()) if inside.any() else None,
                                         "mean_outcome": float(y[inside].mean()) if inside.any() else None})
    for clip in (FORECAST_CLIP, 1.0e-3, 1.0e-2):
        fit = logistic_recalibration(np.clip(forecast, clip, 1.0 - clip), y)
        out["clip_sensitivity"][f"{clip:g}"] = {k: fit[k] for k in ("slope", "slope_lo", "slope_hi")}
    if middle.any() and 0 < y[middle].sum() < middle.sum():
        fit = logistic_recalibration(np.clip(forecast[middle], FORECAST_CLIP, 1.0 - FORECAST_CLIP), y[middle])
        out["intermediate_only_slope"] = {k: fit[k] for k in ("slope", "slope_lo", "slope_hi")}
    return out


def _recalibration(name: str, forecast, label, cluster, seed) -> dict:
    from scipy import special

    from tether.evt.calibration import logistic_recalibration

    fit = np.isin(seed, FIT_SEEDS)
    test = np.isin(seed, TEST_SEEDS)
    clipped = np.clip(np.asarray(forecast, dtype=float), FORECAST_CLIP, 1.0 - FORECAST_CLIP)
    fitted = logistic_recalibration(clipped[fit], label[fit].astype(float))
    mapped = special.expit(fitted["intercept"] + fitted["slope"] * special.logit(clipped[test]))
    return {"fit_seeds": [FIT_SEEDS[0], FIT_SEEDS[-1]], "test_seeds": [TEST_SEEDS[0], TEST_SEEDS[-1]], "fit": fitted,
            "fit_ticks": int(fit.sum()), "test_ticks": int(test.sum()),
            "test_raw": _calibration(name, forecast[test], label[test], cluster[test]),
            "test_recalibrated": _calibration(name, mapped, label[test], cluster[test])}


def _v1_reproduction(payload: dict, v_b: np.ndarray) -> dict:
    """v1's all-tick oracle calibration at H = 1 s (tick-run intervals, rng 20260914)."""
    from tether.monitor import metrics
    from tether.monitor.outcomes import interval_outcomes, slack_intervals, tick_outcomes

    missions = {m["seed"]: m for m in load_missions()}
    tables = []
    for item in payload["linear"]:
        mission = missions[item["seed"]]
        time = mission["ticks"]
        slack = item["slack"]
        ticks = tick_outcomes(time, slack, mission["series_time"], mission["elongation"], mission["rate"], v_b, horizon=1.0)
        intervals = slack_intervals(time, slack, item["onset_time"])
        outcomes = interval_outcomes(intervals, mission["series_time"], mission["elongation"], mission["rate"], v_b, STRESS_THRESHOLD, horizon=1.0)
        tables.append(metrics.monitor_table(item["hazards"]["H1"], ticks, intervals, outcomes, seed=item["seed"]))
    table = metrics.concatenate_tables(tables)
    report = metrics.calibration_report(table.forecast, table.outcome, table.interval, rng=np.random.default_rng(20260914))
    v1 = json.loads(V1_RESULTS_PATH.read_text())["arms"]["O"]["calibration"]
    return {"ticks": int(table.forecast.size), "clusters": report.n_clusters, "events": report.n_events, "slope": report.slope,
            "slope_lo": report.slope_lo, "slope_hi": report.slope_hi, "auroc": report.auroc,
            "v1_committed": {k: v1[k] for k in ("n_ticks", "n_clusters", "n_events", "slope", "slope_lo", "slope_hi", "auroc")},
            "reproduced": bool(abs(report.slope - v1["slope"]) < 1e-9 and report.n_ticks == v1["n_ticks"])}


def analyse() -> dict:
    with RUN_CACHE.open("rb") as handle:
        payload = pickle.load(handle)
    table = payload["table"]
    v_b = np.asarray(payload["v_b"])
    forecasts = _assemble(payload)
    fidelity = json.loads(FIDELITY_PATH.read_text()) if FIDELITY_PATH.exists() else None
    reports = {}
    post_hoc = {}
    sets = {}
    counts = {"oracle_slack_ticks": int(table["time"].size), "intervals": int(np.unique(table["interval"]).size),
              "bounce_intervals": int(np.unique(table["interval"][table["bounce"]]).size), "bounce_ticks": int(np.sum(table["bounce"])),
              "event_intervals": int(np.unique(table["interval"][~table["bounce"]]).size), "event_ticks": int(np.sum(~table["bounce"]))}
    for h in HORIZONS:
        keep = ~table["bounce"] & ~table[f"censored_H{h:g}"]
        sets[h] = keep
        counts[f"calibration_ticks_H{h:g}"] = int(keep.sum())
        counts[f"censored_event_ticks_H{h:g}"] = int(np.sum(~table["bounce"] & table[f"censored_H{h:g}"]))
        counts[f"calibration_clusters_H{h:g}"] = int(np.unique(table["interval"][keep]).size)
        counts[f"positive_labels_H{h:g}"] = int(np.sum(table[f"label_H{h:g}"][keep]))
        counts[f"coupled_ticks_H{h:g}"] = int(np.sum(table[f"coupled_H{h:g}"][keep]))
        counts[f"bounce_ticks_positive_H{h:g}"] = int(np.sum(table[f"label_H{h:g}"][table["bounce"] & ~table[f"censored_H{h:g}"]]))
    for name, forecast_key, h in (("rollout_H1", "rollout_H1", 1.0), ("rollout_H2", "rollout_H2", 2.0), ("linearized_H1", "linearized_H1", 1.0),
                                  ("linearized_H2", "linearized_H2", 2.0), ("held_H1", "held_H1", 1.0)):
        keep = sets[h]
        forecast = forecasts[forecast_key][keep]
        if not np.all(np.isfinite(forecast)):
            raise RuntimeError(f"{name}: non-finite forecasts on calibration ticks")
        label = table[f"label_H{h:g}"][keep]
        cluster = table["interval"][keep]
        coupled = table[f"coupled_H{h:g}"][keep]
        entry = {"all": _calibration(name, forecast, label, cluster), "coincidence": _coincidence(forecast, label, coupled)}
        post_hoc[name] = _degeneracy(forecast, label, cluster)
        if name in ("rollout_H1", "linearized_H1", "rollout_H2"):
            entry["coupled"] = _calibration(f"{name}_coupled", forecast[coupled], label[coupled], cluster[coupled])
            entry["uncoupled"] = _calibration(f"{name}_uncoupled", forecast[~coupled], label[~coupled], cluster[~coupled])
            entry["mean_abs_residual_coupled"] = float(np.mean(np.abs(label[coupled] - forecast[coupled]))) if coupled.any() else None
            entry["mean_abs_residual_uncoupled"] = float(np.mean(np.abs(label[~coupled] - forecast[~coupled]))) if (~coupled).any() else None
        reports[name] = entry
    recal = {}
    for name, forecast_key, h in (("recal_rollout_H1_test", "rollout_H1", 1.0), ("recal_rollout_H2_test", "rollout_H2", 2.0), ("recal_linearized_H1_test", "linearized_H1", 1.0)):
        keep = sets[h]
        recal[name] = _recalibration(name, forecasts[forecast_key][keep], table[f"label_H{h:g}"][keep], table["interval"][keep], table["seed"][keep])
    gate_report = reports["rollout_H1"]["all"]
    lo, hi = (gate_report or {}).get("slope_lo"), (gate_report or {}).get("slope_hi")
    finite = gate_report is not None and not gate_report.get("separated") and lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi)
    passed = bool(finite and lo <= 1.0 <= hi)
    trusted = bool(fidelity and fidelity.get("primary_met"))
    verdict = "PASS" if passed else "FAIL"
    branch = ("(ii) packets grow; arms L, B2 and P run the fleet rollout" if passed else
              "(iii) Prop. 10' withdrawn; hazard judged on discrimination plus the declared post-hoc recalibration (fit 5001-5020, test 5021-5040), reported as 'calibrated by recalibration', never 'calibrated'; Phase 6's engage rule uses the hazard as a ranking and the capstone tests the estimator's (e_hat, edot_hat) through Prop. 14")
    results = {
        "schema_version": 1, "test": "P5-T2'", "source": source_state(), "declarations_sha256": declarations_sha256(),
        "declarations_path": str(DECLARATIONS_PATH.relative_to(RECORDS.parent)),
        "run_declarations_sha256": payload["declarations_sha256"],
        "inputs_sha256": {"code_fleet_rollout": sha256_file(Path(__file__).resolve().parents[2] / "monitor" / "fleet_rollout.py"),
                          "code_p5_pretest": sha256_file(Path(__file__).resolve()), "impact_table_fan": sha256_file(FAN_IMPACT_PATH)},
        "v_b": v_b.tolist(), "counts": counts, "fidelity": fidelity, "reports": reports, "recalibration_branch_iii": recal,
        "v1_reproduction": _v1_reproduction(payload, v_b),
        "post_hoc_diagnostics": {"addendum": str(ADDENDUM_PATH.relative_to(RECORDS.parent)), "addendum_sha256": write_bytes(ADDENDUM_PATH, json_bytes(ADDENDUM)),
                                 "forecasts": post_hoc},
        "compute": {"wall_seconds": payload["wall_seconds"], "rollout_core_seconds": float(sum(r["wall_seconds"] for r in payload["rollouts"]))},
        "gate": {"condition": DECLARATIONS["gate"]["condition"], "slope": (gate_report or {}).get("slope"), "slope_lo": lo, "slope_hi": hi,
                 "separated": (gate_report or {}).get("separated"), "verdict": verdict, "branch": branch, "fidelity_met": trusted,
                 "note": None if trusted else "fidelity tolerance not met: the verdict is untrusted pending diagnosis"},
    }
    tick_arrays = {k: v for k, v in table.items()}
    tick_arrays.update({f"forecast_{k}": v for k, v in forecasts.items()})
    results["ticks_npz_sha256"] = write_bytes(TICKS_PATH, npz_bytes({k: np.asarray(v) for k, v in tick_arrays.items()}))
    write_bytes(RESULTS_PATH, json_bytes(results))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("declare", "fidelity", "run", "analyse", "all"))
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.workers > 8:
        raise SystemExit("P5-T2' is declared at <= 8 workers")
    if arguments.command in ("declare", "all"):
        print("declarations", write_declarations())
    if arguments.command in ("fidelity", "all"):
        summary = run_fidelity(arguments.workers)
        print(json.dumps({k: {m: summary["variants"][k][m] for m in ("abs_e_at_0.5s", "abs_edot_at_0.5s", "label_agreement")} for k in summary["variants"]}, indent=1))
        print("fidelity met", summary["primary_met"])
    if arguments.command in ("run", "all"):
        if not DECLARATIONS_PATH.exists():
            raise SystemExit("declare first")
        payload = run_compute(arguments.workers)
        print("run wall seconds", payload["wall_seconds"])
    if arguments.command in ("analyse", "all"):
        results = analyse()
        print(json.dumps(results["gate"], indent=1))


if __name__ == "__main__":
    main()
