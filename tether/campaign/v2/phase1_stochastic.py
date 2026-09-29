"""Phase 1(f): the stochastic cells, and the tests they decide.

Plan v2 Phase 1 "Cells (f)", "Observables", P1-T5, P1-T8, P1-T10', P1-T11, P1-T12,
II.2 (H1)/(H4'), Appendix B.1-B.4, B.6, B.7.

Eight cells on the selected controller (k_sigma = 3 clamped at 0.349 rad, the owner's
ruling in ``records/v2/phase0/sway_addendum_1.json``), 22 seeds each (2 pilots that fix
the cell's T_peak-quantile threshold grid and never enter a scored statistic, 20
statistics seeds), 300 s after a 20 s warm-up::

    intensity 0.35, T0 in {0.6, 0.8, 1.0} kN   k_h = 1.5x the lateral stability boundary
    intensity 0.50, T0 in {0.6, 1.0, 1.4} kN   (477 / 620 / 763 / 1050 N m/rad)
    intensity 0.35 and 0.50, T0 = 1.0 kN, k_sigma = 0  (the sway term removed, k_h = 763)

Phase 2 will not launch (P1-T11 failed at the Phase 1(e) pilot cell and the committed
population forecast gives no launch-powered (H4') class at intensity 0.35), so nothing
here gates a Phase 2 cell.  These cells are run because they measure the two halves of
the trade-off that is the reason Phase 2 cannot launch: the shape the loop can hold, and
the events the plant then produces.

Everything measured per cell is listed in ``build_declarations()`` and written, with its
declared reading, to ``records/v2/phase1/stochastic_declarations.json`` before any run.
The literal plan readings are primary throughout: the full merged W^c episode for t_x
(IV.4/B.4) and the anchored 2 s burst rule for events (B.1); the clipped-episode and
chained-burst variants are computed beside them and gate nothing.

CLI::

    python -m tether.campaign.v2.phase1_stochastic declare
    python -m tether.campaign.v2.phase1_stochastic compute --workers 8
    python -m tether.campaign.v2.phase1_stochastic analyse
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tether.campaign.common import RECORDS, json_bytes, relative, sha256_file, source_state, write_bytes
from tether.campaign.v2 import events as ev
from tether.campaign.v2 import regime as rg

# --------------------------------------------------------------------------- pinned constants

M_FLEET = 523.9  # kg, the formation restoring reduced mass (P1-T8: a_0 = T0/m_fleet)
TRIM_GAIN = 100.0
WARMUP = 20.0
DURATION = 300.0
SWAY_LIMIT_V2 = 0.349  # rad, the owner's +-20 deg clamp
SELECTED_K_SIGMA = 3.0
N_CABLES = 5

PILOT_SEEDS = (7101, 7102)
STATISTICS_SEEDS = tuple(range(7103, 7123))  # 20 seeds
ALL_SEEDS = PILOT_SEEDS + STATISTICS_SEEDS

SHAPE_BOUND_DEG = 15.0  # (H1)
GRID_QUANTILES = (50.0, 80.0, 95.0, 99.0)
GRID_STEP_N = 500.0  # the plan's 0.5 kN grid
POWERED_EVENTS = 20  # B.6
POPULATION_LEVELS = 3  # P1-T10': >= 20 events at >= 3 grid levels ...
POPULATION_CELLS = 3  # ... in >= 3 cells
CLOSURE_FRACTION_BOUND = 0.25  # P1-T12
DEPTH_BANDS = ((0.05, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 12.0))

P1_DIR = RECORDS / "v2" / "phase1"
DECLARATIONS_PATH = P1_DIR / "stochastic_declarations.json"
RUNS_PATH = P1_DIR / "stochastic_runs.json"
RESULTS_PATH = P1_DIR / "stochastic_results.json"
ADDENDUM_PATH = P1_DIR / "stochastic_addendum_1.json"
PHASE1_PREDICTIONS_PATH = P1_DIR / "phase1_predictions.json"
PHASE2_PREDICTIONS_PATH = RECORDS / "v2" / "phase2" / "phase2_predictions_i035.json"
P1E_RESULTS_PATH = P1_DIR / "p1e_results.json"
SWAY_ADDENDUM_PATH = RECORDS / "v2" / "phase0" / "sway_addendum_1.json"


@dataclass(frozen=True)
class Cell:
    name: str
    intensity: float
    pretension: float
    heading_gain: float
    k_sigma: float
    sway_limit: float | None


CELLS = (
    Cell("i035_T0600_ks3", 0.35, 600.0, 477.0, SELECTED_K_SIGMA, SWAY_LIMIT_V2),
    Cell("i035_T0800_ks3", 0.35, 800.0, 620.0, SELECTED_K_SIGMA, SWAY_LIMIT_V2),
    Cell("i035_T1000_ks3", 0.35, 1000.0, 763.0, SELECTED_K_SIGMA, SWAY_LIMIT_V2),
    Cell("i050_T0600_ks3", 0.50, 600.0, 477.0, SELECTED_K_SIGMA, SWAY_LIMIT_V2),
    Cell("i050_T1000_ks3", 0.50, 1000.0, 763.0, SELECTED_K_SIGMA, SWAY_LIMIT_V2),
    Cell("i050_T1400_ks3", 0.50, 1400.0, 1050.0, SELECTED_K_SIGMA, SWAY_LIMIT_V2),
    Cell("i035_T1000_ks0", 0.35, 1000.0, 763.0, 0.0, None),
    Cell("i050_T1000_ks0", 0.50, 1000.0, 763.0, 0.0, None),
)
CELL_BY_NAME = {cell.name: cell for cell in CELLS}

# The committed intensity-0.35 primary-onset rates (per cable-second) of
# records/v2/phase2/phase2_predictions_i035.json III_1."2_declustered_primary_onset_rate",
# keyed by this module's cell names; intensity 0.5 carries an analytic Rice rate only
# (no committed theta), from "intensity_0.5_out_of_H1_comparisons".
PREDICTION_KEYS = {
    "i035_T0600_ks3": ("theta_rice", "600@477"),
    "i035_T0800_ks3": ("theta_rice", "800@620"),
    "i035_T1000_ks3": ("theta_rice", "1000@763"),
    "i050_T0600_ks3": ("rice_only", "600"),
    "i050_T1000_ks3": ("rice_only", "1000"),
    "i050_T1400_ks3": ("rice_only", "1400"),
}

# The plan's own executed-campaign medians of a_bar_ret/a_0 by depth band (II.4).
PLAN_A_BAR_RET_OVER_A0 = (0.07, 0.36, 0.53, 0.43)


def spec_for(cell: Cell):
    from tether.campaign.fleet_run import FleetRunSpec

    return FleetRunSpec(
        formation="parallel",
        pretension=cell.pretension,
        heading_gain=cell.heading_gain,
        trim_gain=TRIM_GAIN,
        drag_law="linear",
        weather_distribution="gaussian",
        weather_direction="local",
        weather_scale=cell.intensity,
        duration=DURATION,
        warmup=WARMUP,
        cable_mode="recording",
        log_state=True,
        k_sigma=cell.k_sigma,
        sway_limit=cell.sway_limit,
    )


# --------------------------------------------------------------------------- declarations


def build_declarations() -> dict:
    return {
        "schema": "v2-phase1f-declarations-1",
        "title": "Phase 1(f): stochastic cells, P1-T5, P1-T8, P1-T10', P1-T11, P1-T12, and the shape/population trade-off",
        "date": "2026-09-13",
        "declared_before_any_result": True,
        "plan": {
            "path": "ref/tail_of_the_tether_plan_v2.md",
            "sections": ["Phase 1 Cells (f)", "Phase 1 Observables", "P1-T5", "P1-T8", "P1-T10'", "P1-T11", "P1-T12",
                         "Phase 1 outcome matrix", "II.2 (H1), (H4')", "II.4", "Appendix B.1-B.4, B.6, B.7"],
        },
        "context": {
            "phase2_will_not_launch": ("P1-T11 FAILED at the Phase 1(e) selected cell (chord std 25.40 deg against 15 deg, "
                                       "psi 13.01 deg inside) and the committed population forecast gives no launch-powered "
                                       "(H4') class at intensity 0.35; these cells gate no Phase 2 cell and are run to measure "
                                       "the shape/population trade-off itself"),
            "selected_controller": ("k_sigma = 3, sway_limit 0.349 rad (records/v2/phase0/sway_addendum_1.json: clause A at "
                                    "intensity 0.35 selects k_sigma = 3 from the model alone; clause B is unsatisfiable)"),
            "open_machinery_rulings": ("both unruled; the LITERAL plan readings are primary here and the variants are "
                                       "reported beside them and gate nothing: (a) t_x is the FULL merged W^c episode "
                                       "(IV.4/B.4), with the slack-clipped episode as regime_clip; (b) events use the "
                                       "ANCHORED 2 s burst rule (B.1), with the chained rule as a sensitivity count"),
        },
        "cells": {
            "grid": [{"name": c.name, "intensity": c.intensity, "T0_N": c.pretension, "k_h_N_m_per_rad": c.heading_gain,
                      "k_sigma": c.k_sigma, "sway_limit_rad": c.sway_limit,
                      "config_hash": spec_for(c).config_hash()} for c in CELLS],
            "formation": "parallel", "weather": "Gaussian AR(1) local", "cable_mode": "recording", "drag_law": "linear",
            "trim_gain_N_m_per_rad": TRIM_GAIN, "warmup_s": WARMUP, "duration_s": DURATION,
            "heading_gain_rule": "1.5 x the lateral stability boundary 0.318 (1.5 T0 + k_c), k_c = 100: 477 / 620 / 763 / 1050",
            "pilot_seeds": list(PILOT_SEEDS), "statistics_seeds": list(STATISTICS_SEEDS),
            "seed_use": ("the two pilot seeds fix the cell's T_peak-quantile threshold grid and enter no scored statistic; "
                         "every test is scored on the 20 statistics seeds, and the pilots are reported separately"),
            "comparison_cells": ("k_sigma = 0 at T0 = 1 kN at both intensities, at the SELECTED k_h = 763 rather than the "
                                 "plan's Phase 2 v1-comparison gain k_h = 500: the contrast then changes exactly one thing "
                                 "(the sway term) and reproduces the Phase 1(e) k_sigma = 0 cells bit for bit (same spec "
                                 "hash). Declared as a deliberate departure; the k_h = 500 v1 cell is not run here"),
            "cost": {"simulated_seconds": len(CELLS) * len(ALL_SEEDS) * DURATION,
                     "simulated_seconds_including_warmup": len(CELLS) * len(ALL_SEEDS) * (DURATION + WARMUP),
                     "core_hours_plan_basis_7000_per_core_hour": len(CELLS) * len(ALL_SEEDS) * DURATION / 7000.0,
                     "core_hours_including_warmup": len(CELLS) * len(ALL_SEEDS) * (DURATION + WARMUP) / 7000.0,
                     "note": ("the plan's own Phase 1(f) arithmetic (7.5 core-hours) counts recorded seconds only; both "
                              "figures are given. Phase 1(e) measured 10,713 simulated seconds per core-hour on this "
                              "machine, so the expected cost is about 5.3 core-hours. No cell is cut"),
                     "workers": 8},
        },
        "measurements": {
            "shape_H1": ("state rows in [warmup, end] decimated to 0.1 s: circular std of the world chord angle sigma_i, of "
                         "the load-frame chord angle wrap(sigma_i - theta_0), of psi_i = wrap(theta_i - sigma_i) and of the "
                         "load yaw; circular sums pooled over the 20 statistics seeds, then the cell statistic is the MAX "
                         "over vessels (the Phase 1(e) gating statistic, unchanged). tether/campaign/v2/summaries.py"),
            "slack_duty": "slack (e <= 0) 1 ms samples / window samples, pooled over cables and statistics seeds",
            "onsets": "geometric down-crossings of e = 0 in [warmup, end] on any live cable (summaries_v2 onsets_v2)",
            "primary_onsets": ("B.1: an onset is primary iff no re-engagement on any cable lies in (t - 3 s, t]; the primary "
                               "rate is primary onsets / (5 cables x primary exposure), the exposure being the window less "
                               "the union of the [t_up, t_up + 3 s) windows"),
            "marks_by_class": "(H4') class of every re-engagement mark, from weather-side covariates only (B.2, regime.py)",
            "events": "B.1 anchored events; an event's class is its parent's, its snap statistic the cluster maximum of T_peak",
            "threshold_grid": ("per cell, from its two PILOT seeds only: T_b(q) = the q-th percentile of the pilot T_peak "
                               "distribution pooled over cables, q in {50, 80, 95, 99}, rounded to the nearest 0.5 kN and "
                               "de-duplicated. If the pilots hold no mark the cell has NO grid and reports none; if they "
                               "hold fewer than 4 marks the grid is still taken (percentiles of the few) and flagged"),
            "snap_exceedances": ("per class and grid level, on the statistics seeds: EVENTS whose cluster maximum T_peak >= "
                                 "the level (primary, per B.1 'stress counts are per event'), with the mark-level count "
                                 "reported beside it"),
            "closure_B7": "terminal event: per-cell closure fraction over the 22 runs, the closure times, and exposure lost",
            "clean_set_B3": "taut samples outside (t_up, t_up + 3 s] of every re-engagement on any cable; exposure lost reported",
            "a_bar_ret": ("Prop. 3' from the DEEPEST POINT, two declared readings, both reported: depth-mean (work) "
                          "a = V_up^2/(2 Delta), for which Prop. 3' is an identity and the T5 slope is 1 by construction; "
                          "time-mean a = V_up/(t_up - t_deep), the reading P1-T5 is scored on. The offset estimator "
                          "(V_up^2 - u^2)/(2 Delta) is never read as a definition (plan II.4)"),
            "t_deep": "argmin of e over the mark's slack samples [i_down, i_up) - the same turnaround regime.py uses",
            "driven_fraction": ("share of the return-leg 1 ms samples [t_deep, t_up) whose W^c exceeds T0, i.e. the fraction "
                                "of the return on which the weather is still driving the gap shut; the scripted-cell "
                                "analogue in claim_a.py is the share of the return inside the square gust"),
        },
        "tests": {
            "P1-T5": {"statement": "V_up against sqrt(2 a_bar_ret Delta) from the deepest point, per (H4') class",
                      "reading": ("through-origin slope and uncentred R2 of V_up on sqrt(2 a_bar_ret Delta) with the "
                                  "TIME-MEAN a_bar_ret; the depth-mean reading is reported as the identity it is"),
                      "threshold": "slope within 5% of one and R2 > 0.98, in every class holding >= 20 marks",
                      "marks_entering_the_fit": ("marks with a positive recorded depth; a_bar_ret is undefined at zero "
                                                 "depth (no turnaround to read the law from) and those marks are counted "
                                                 "and excluded from both readings"),
                      "scored_on": ("all statistics-seed marks pooled over the eight cells (the primary scoring set, since "
                                    "per-cell class counts at intensity 0.35 will be far below 20), with the per-cell "
                                    "per-class fit reported wherever that class holds >= 20 marks in the cell",),
                      "outcome_rule": ("plan Phase 1 outcome matrix: T5 failing in a class holding >= 20 marks means Prop. 3' "
                                       "does not govern that class, its measured a_bar_ret becomes the operative form and its "
                                       "severance law is EMP")},
            "P1-T8": {"statement": "a_bar_ret/a_0 by depth band and class, a_0 = T0/m_fleet with m_fleet = 523.9 kg; driven fraction",
                      "reading": "median over marks in each (class, depth band) of both a_bar_ret readings and of the driven fraction",
                      "threshold": "pinned per class, not a pass/fail; compared with the committed scripted prediction and with the plan's own executed medians 0.07 / 0.36 / 0.53 / 0.43"},
            "P1-T10'": {"statement": "regime population per (H4') class at each grid level",
                        "rule": ("the Phase 1 NO-LAUNCH population rule: a class is populated when it holds >= 20 EVENTS at "
                                 ">= 3 grid levels in >= 3 cells; scored over the six k_sigma = 3 cells (the comparison "
                                 "cells are reported and excluded from the count), and the mark-level count is reported beside it")},
            "P1-T11": {"statement": "shape condition of (H1), per cell",
                       "threshold": "world chord-angle circular std and psi circular std, cell statistic = max over vessels, both <= 15 deg",
                       "secondary": "the load-relative chord std is reported and gates nothing"},
            "P1-T12": {"statement": "closure as a terminal event",
                       "non_degeneracy": ("at least one stochastic cell at EACH intensity Phase 2 would use (0.35 and 0.5) "
                                          "keeps its closure fraction below 0.25, with its closure-time distribution and "
                                          "lost exposure reported; a phase in which every stochastic cell closes at or above "
                                          "0.25, or in which closures are so few that the terminal-event rate carries no "
                                          "interval, is degenerate"),
                       "interval": "exact Poisson 95% interval on the closure count over the 22 runs of the cell"},
        },
        "trade_off_table": ("the campaign's central Claim A evidence: per (intensity, T0) cell - chord std (world and "
                            "load-relative), psi std, primary onsets per cable-second, marks and events, whether the cell "
                            "meets (H1), and whether the population rule is met - together with the cell that is least far "
                            "from meeting both, measured by the two gaps (chord std / 15 deg, and events at the cell's own "
                            "3rd grid level against 20)"),
        "predictions_compared": {
            "primary_onset_rate_intensity_0.35": ("records/v2/phase2/phase2_predictions_i035.json III_1."
                                                  "2_declustered_primary_onset_rate model_mean (per cable-second), committed "
                                                  "before any of these runs, at 600@477, 800@620, 1000@763"),
            "primary_onset_rate_intensity_0.5": ("records/v2/phase2/phase2_predictions_i035.json "
                                                 "intensity_0.5_out_of_H1_comparisons rice_model_mean_per_cable_s - an "
                                                 "analytic Rice rate with NO committed theta, so the comparison is reported "
                                                 "as measured/Rice and read against the 0.35 thetas 0.26-0.44; non-gating"),
            "P1-T5_P1-T8": "records/v2/phase1/phase1_predictions.json P1-T5 and P1-T8 (committed on the scripted cells)",
            "shape": "records/v2/phase1/p1e_results.json (the 5-seed pilot at T0 = 1 kN) and the p1e statistical-linearization predictions",
        },
        "outputs": {"runs": relative(RUNS_PATH), "results": relative(RESULTS_PATH)},
        "development_disclosure": ("before this file was written, one development run - the i050_T0600_ks3 spec shortened to "
                                   "40 s, seed 9001, which is not one of the 22 declared seeds and enters no result - was used "
                                   "to check that run_job executes end to end. It showed 10 marks, of which several carry a "
                                   "recorded depth of exactly zero; that is why the P1-T5 reading above declares the "
                                   "positive-depth restriction. No test threshold and no cell was changed on seeing it."),
        "discipline": ("declarations written before any run and their sha256 recorded in the results; no post-hoc edit (a "
                       "dated addendum with its reason instead); v1 behaviour untouched; results through "
                       "tether.campaign.common.json_bytes/write_bytes"),
    }


def declare() -> str:
    if DECLARATIONS_PATH.exists():
        raise SystemExit(f"{DECLARATIONS_PATH} exists; write a dated addendum instead of editing it")
    return write_bytes(DECLARATIONS_PATH, json_bytes(build_declarations()))


# --------------------------------------------------------------------------- the runs


@dataclass(frozen=True)
class Job:
    cell: str
    seed: int
    pilot: bool

    @property
    def cost(self) -> float:
        return WARMUP + DURATION


def jobs() -> list[Job]:
    return [Job(cell.name, seed, seed in PILOT_SEEDS) for cell in CELLS for seed in ALL_SEEDS]


def return_leg_quantities(summary: dict, wc: np.ndarray, elongation: np.ndarray, time: np.ndarray,
                          pretension: float) -> dict[str, np.ndarray]:
    """t_deep, the time-mean a_bar_ret and the driven fraction of every mark's return leg."""
    marks = summary["marks"]
    extra = summary["marks_v2"]
    count = int(np.asarray(marks["t_up"]).size)
    out = {name: np.full(count, np.nan) for name in ("t_deep", "a_bar_ret_time", "driven_fraction", "return_leg_s")}
    for row in range(count):
        lo, hi = int(extra["i_down"][row]), int(extra["i_up"][row])
        cable = int(marks["cable"][row])
        if not (0 < lo < hi <= elongation.shape[0]):
            continue
        turn = lo + int(np.argmin(elongation[lo:hi, cable]))
        t_deep = float(time[turn])
        leg = float(marks["t_up"][row]) - t_deep
        out["t_deep"][row] = t_deep
        out["return_leg_s"][row] = leg
        if hi > turn:
            out["driven_fraction"][row] = float(np.mean(wc[turn:hi, cable] > pretension))
        if leg > 0.0:
            out["a_bar_ret_time"][row] = float(marks["v_return"][row]) / leg
    return out


def summary_without_reengagements(run, warmup: float, pretension: float, weather: np.ndarray | None) -> dict:
    """The v2 summary blocks of a run in which NO cable ever re-engaged.

    ``tether/campaign/v2/summaries.py`` raises ``IndexError`` on such a run (line 205,
    ``ups.cable[np.maximum(primary.parent, 0)]`` with an empty ``ups`` and at least one
    onset in the window).  That file belongs to the stage-2 machinery and is not edited
    here; this function reproduces the blocks for that one degenerate case, from the same
    helpers and the same definitions, and is used only when the run has no up-crossing of
    e = 0 at all.  The reason it is exact rather than an approximation:

    * no re-engagement anywhere => no mark and no B.1 event exists;
    * B.1's exclusion windows are empty, so every onset is primary and the primary
      exposure is the whole window;
    * B.3's clean set excludes nothing, so it coincides with v1's clean set and v1's own
      moments are the B.3 moments (v1 excludes only 1 s windows after marks, of which
      there are none);
    * the shape block is ``summaries._shape_statistics`` itself, called unchanged.

    Declared in ``records/v2/phase1/stochastic_addendum_1.json``.
    """
    from tether.campaign import summaries as v1
    from tether.campaign.v2 import summaries as sm

    summary = v1.summarize_run(run, warmup, pretension)
    record = sm.run_record(run)
    time = record["event_time"]
    elongation = record["elongation"]
    alive = record["alive"]
    end = record["sim_end"]
    closure = record["closure"]
    if closure is not None:
        end = min(end, float(closure[1]))
    ups, downs = ev.crossings(time, elongation, alive)
    if ups.size:
        raise ValueError("summary_without_reengagements called on a run that does carry re-engagements")
    n_cables = elongation.shape[1]
    interpolant = rg.StateInterpolant.from_log(record["state_time"], record["state"], record["load_offsets"],
                                               record["vessel_offsets"])
    onset_rows = np.flatnonzero((downs.time >= warmup) & (downs.time <= end))
    window_exposure = max(0.0, end - warmup)
    empty_float = np.empty(0)
    empty_int = np.empty(0, dtype=np.int64)
    empty_bool = np.empty(0, dtype=bool)
    empty_str = np.empty(0, dtype="<U12")
    marks_v2 = {name: empty_float for name in rg.MARK_COVARIATES}
    marks_v2.update({"i_down": empty_int, "i_up": empty_int, "n_episodes": empty_int,
                     "regime": empty_str, "regime_clip": empty_str, "any_exceedance": empty_bool,
                     "episode_truncated": empty_bool, "event_id": empty_int, "event_role": empty_str,
                     "event_parent": empty_int, "gap_previous": empty_float, "primary": empty_bool,
                     "onset_parent_cable": empty_int, "onset_parent_lag": empty_float,
                     "onset_parent_same_cable": empty_bool, "primary_2s_same_cable": empty_bool,
                     "matched_crossings": empty_bool})
    planned_end = float(record["spec"].warmup + record["spec"].duration)
    closure_block = ({"terminal": False, "time": None, "cable": None, "exposure_lost": 0.0,
                      "planned_end": planned_end, "in_window": None} if closure is None else
                     {"terminal": True, "time": float(closure[1]), "cable": int(closure[0]),
                      "exposure_lost": max(0.0, planned_end - max(float(closure[1]), warmup)),
                      "planned_end": planned_end, "in_window": bool(float(closure[1]) >= warmup)})
    q = np.asarray(summary["moments"]["q"], dtype=float)
    run_weather = record.get("run_weather")
    blocks = {
        "marks_v2": marks_v2,
        "events": {"cluster_max_T_peak": empty_float, "regime": empty_str, "n_marks": empty_int,
                   "n_bounces": empty_int, "primary": empty_bool, "event_id": empty_int},
        "onsets_v2": {"count": int(onset_rows.size), "primary_count": int(onset_rows.size),
                      "primary_2s_same_cable_count": int(onset_rows.size),
                      "exposure": window_exposure, "primary_exposure": window_exposure,
                      "per_cable": np.bincount(downs.cable[onset_rows], minlength=n_cables)},
        "clean": {"moments": summary["moments"], "exposure": window_exposure, "window_exposure": window_exposure,
                  "excluded_fraction": 0.0},
        "closure": closure_block,
        "shape": sm._shape_statistics(record, interpolant, warmup, end, n_cables),
        "regime_summary": {"marks": rg.class_counts(empty_str), "events": rg.class_counts(empty_str)},
        "v2_diagnostics": {
            "weather_bit_equal_to_run": (None if run_weather is None or weather is None else
                                         bool(run_weather.shape == weather.shape and np.array_equal(run_weather, weather))),
            "marks_same_order_as_v1": True, "marks_unmatched_to_crossings": 0,
            "window_marks_whose_event_parent_is_outside_window": 0,
            "events_in_window_chained_rule_sensitivity": 0,
            "wrel_offline_vs_log": {
                "passes_1e-6_relative": None,
                "not_checked_reason": ("the run holds no re-engagement mark, so no weather-side covariate is computed "
                                       "from W_rel or W^c and the machinery's C3 check has nothing to gate"),
            },
            "no_reengagement_in_run": True,
        },
    }
    assert q.shape[0] == 3
    out = dict(summary)
    out.update(blocks)
    return out


MARK_EXPORT_V1 = ("t_up", "cable", "u_entry", "depth", "dwell", "v_return", "T_peak", "n_maxima")
MARK_EXPORT_V2 = ("regime", "regime_clip", "t_x", "t_x_clip", "Wc_max", "v_s", "any_exceedance", "n_episodes",
                  "a_bar_ret", "Wc_return_mean", "Wc_slack_max", "primary", "event_id", "event_role")


def run_job(job: Job) -> dict:
    """One (cell, seed).  A failure is returned, not raised, so one bad run cannot destroy
    the batch; ``analyse`` refuses to score any cell holding a failed run."""
    import traceback

    try:
        return _run_job(job)
    except Exception as error:  # noqa: BLE001 - recorded and surfaced by the analysis
        return {"cell": job.cell, "seed": job.seed, "pilot": bool(job.pilot),
                "error": repr(error), "traceback": traceback.format_exc()}


def _run_job(job: Job) -> dict:
    from tether.campaign.fleet_run import build_run, run_to_end
    from tether.campaign.v2.summaries import summarize_run_v2

    cell = CELL_BY_NAME[job.cell]
    spec = spec_for(cell)
    # The run draws its own weather; the offline copy is regenerated independently so the
    # summariser's bit-equality check is a real check (it is also the series W^c is built from).
    run = run_to_end(build_run(spec, job.seed))
    weather = rg.regenerate_weather(spec, job.seed, N_CABLES)
    log_probe = run.fleet.cables.log
    probe_ups, _ = ev.crossings(log_probe.event_time[: log_probe.count], log_probe.elongation[: log_probe.count],
                                log_probe.alive[: log_probe.count])
    no_reengagement = bool(probe_ups.size == 0)
    if no_reengagement:
        summary = summary_without_reengagements(run, WARMUP, cell.pretension, weather)
    else:
        summary = summarize_run_v2(run, WARMUP, cell.pretension, weather=weather)

    log = run.fleet.cables.log
    n = log.count
    time = log.event_time[:n]
    elongation = log.elongation[:n]
    interpolant = rg.StateInterpolant.from_log(log.state_time[: log.state_count], log.state[: log.state_count],
                                               run.fleet.geometry.load_offsets, run.fleet.geometry.vessel_offsets)
    _, wc = rg.weather_side_series(time, weather, interpolant)
    legs = return_leg_quantities(summary, wc, elongation, time, cell.pretension)

    marks = {name: np.asarray(summary["marks"][name]) for name in MARK_EXPORT_V1}
    marks.update({name: np.asarray(summary["marks_v2"][name]) for name in MARK_EXPORT_V2})
    marks.update(legs)

    events = summary["events"]
    clean = summary["clean"]
    q = np.asarray(clean["moments"]["q"], dtype=float)
    shape = summary["shape"]
    window_samples = int(summary["meta"]["window_samples"])
    return {
        "cell": job.cell, "seed": job.seed, "pilot": bool(job.pilot),
        "no_reengagement_in_run": no_reengagement,
        "config_hash": spec.config_hash(),
        "end_time": float(summary["meta"]["end_time"]),
        "exposure_s": float(summary["meta"]["exposure"]),
        "window_samples": window_samples,
        "tracker_dropped": int(summary["meta"]["tracker_dropped"]),
        "lint_violations": list(summary["meta"]["lint_violations"]),
        "closure": summary["closure"],
        "slack_samples": np.asarray(summary["slack_samples"], dtype=np.int64),
        "onsets_count": int(summary["onsets_v2"]["count"]),
        "primary_onsets": int(summary["onsets_v2"]["primary_count"]),
        "primary_onsets_2s_same_cable": int(summary["onsets_v2"]["primary_2s_same_cable_count"]),
        "primary_exposure_s": float(summary["onsets_v2"]["primary_exposure"]),
        "onsets_per_cable": np.asarray(summary["onsets_v2"]["per_cable"], dtype=np.int64),
        "marks": marks,
        "n_marks": int(np.asarray(marks["t_up"]).size),
        "events": {"cluster_max_T_peak": np.asarray(events["cluster_max_T_peak"], dtype=float),
                   "regime": np.asarray(events["regime"]).astype(str),
                   "n_marks": np.asarray(events["n_marks"], dtype=np.int64),
                   "n_bounces": np.asarray(events["n_bounces"], dtype=np.int64),
                   "primary": np.asarray(events["primary"], dtype=bool)},
        "n_events": int(np.asarray(events["cluster_max_T_peak"]).size),
        "events_chained_sensitivity": int(summary["v2_diagnostics"]["events_in_window_chained_rule_sensitivity"]),
        "regime_summary": summary["regime_summary"],
        "clean": {"exposure_s": float(clean["exposure"]), "window_exposure_s": float(clean["window_exposure"]),
                  "excluded_fraction": clean["excluded_fraction"],
                  "q_count": q[0], "q_sum": q[1], "q_square": q[2]},
        "shape_sums": {name: {"cos": np.asarray(block["cos_sum"], dtype=float),
                              "sin": np.asarray(block["sin_sum"], dtype=float),
                              "count": int(block["count"])} for name, block in shape["sums"].items()},
        "shape_samples": int(shape["samples"]),
        "shape_std_deg": {name: np.degrees(np.atleast_1d(shape[f"{name}_std"])) for name in
                          ("chord_world", "chord_load", "psi", "load_yaw")},
        "diagnostics": {"weather_bit_equal": summary["v2_diagnostics"]["weather_bit_equal_to_run"],
                        "marks_same_order_as_v1": summary["v2_diagnostics"]["marks_same_order_as_v1"],
                        "marks_unmatched": summary["v2_diagnostics"]["marks_unmatched_to_crossings"],
                        "wrel_passes": summary["v2_diagnostics"]["wrel_offline_vs_log"]["passes_1e-6_relative"],
                        "orphan_marks": summary["v2_diagnostics"]["window_marks_whose_event_parent_is_outside_window"]},
        "wall_seconds": float(run.wall_seconds),
    }


def compute(workers: int = 8) -> list[dict]:
    from tether.campaign.common import run_pool

    if not DECLARATIONS_PATH.exists():
        raise SystemExit("declarations must be written before any result")
    declarations_sha = sha256_file(DECLARATIONS_PATH)
    runs = run_pool(run_job, jobs(), workers)
    failures = [{"cell": r["cell"], "seed": r["seed"], "error": r["error"]} for r in runs if "error" in r]
    write_bytes(RUNS_PATH, json_bytes({"schema": "v2-phase1f-runs-1", "declarations_sha256": declarations_sha,
                                       "source": source_state(), "failed_runs": failures, "runs": runs}))
    if failures:
        print(f"WARNING: {len(failures)} run(s) failed; the affected cells carry no verdict")
    return runs


# --------------------------------------------------------------------------- analysis helpers


def circular_std_deg(cos_sum, sin_sum, count) -> np.ndarray:
    resultant = np.hypot(np.asarray(cos_sum, dtype=float), np.asarray(sin_sum, dtype=float)) / max(float(count), 1.0)
    return np.degrees(np.sqrt(-2.0 * np.log(np.clip(resultant, 1.0e-12, 1.0))))


def through_origin(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 2 or not np.dot(x, x) > 0.0:
        return {"n": int(x.size), "slope": None, "uncentred_r2": None}
    slope = float(np.dot(x, y) / np.dot(x, x))
    residual = y - slope * x
    return {"n": int(x.size), "slope": slope,
            "uncentred_r2": float(1.0 - np.dot(residual, residual) / np.dot(y, y)) if np.dot(y, y) > 0 else None}


def poisson_interval(count: int, exposure: float) -> list[float | None]:
    """Exact Poisson 95% interval on a rate (C.1)."""
    from scipy.stats import chi2

    if exposure <= 0:
        return [None, None]
    low = 0.0 if count == 0 else float(chi2.ppf(0.025, 2 * count) / 2.0 / exposure)
    high = float(chi2.ppf(0.975, 2 * count + 2) / 2.0 / exposure)
    return [low, high]


def binomial_interval(count: int, total: int) -> list[float | None]:
    from scipy.stats import beta

    if total <= 0:
        return [None, None]
    low = 0.0 if count == 0 else float(beta.ppf(0.025, count, total - count + 1))
    high = 1.0 if count == total else float(beta.ppf(0.975, count + 1, total - count))
    return [low, high]


def threshold_grid(pilot_peaks: np.ndarray) -> dict:
    """T_b(q) from the pilot T_peak distribution, q in {50, 80, 95, 99}, on the 0.5 kN grid."""
    peaks = np.asarray(pilot_peaks, dtype=float)
    peaks = peaks[np.isfinite(peaks)]
    if peaks.size == 0:
        return {"levels_N": [], "quantiles": list(GRID_QUANTILES), "pilot_marks": 0, "raw_N": [],
                "note": "the pilot seeds hold no mark: the cell has no threshold grid"}
    raw = [float(np.percentile(peaks, q)) for q in GRID_QUANTILES]
    snapped = [float(round(value / GRID_STEP_N) * GRID_STEP_N) for value in raw]
    levels: list[float] = []
    for value in snapped:
        if value > 0.0 and value not in levels:
            levels.append(value)
    return {"levels_N": levels, "quantiles": list(GRID_QUANTILES), "pilot_marks": int(peaks.size), "raw_N": raw,
            "snapped_N": snapped, "few_marks_flag": bool(peaks.size < len(GRID_QUANTILES))}


def _pool(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(values) if values else np.empty(0)


# --------------------------------------------------------------------------- analysis


def cell_block(cell: Cell, runs: list[dict]) -> dict:
    statistics = [r for r in runs if not r["pilot"]]
    pilots = [r for r in runs if r["pilot"]]
    a0 = cell.pretension / M_FLEET

    # ---- shape (H1), pooled over the statistics seeds
    shape = {}
    for name in ("chord_world", "chord_load", "psi", "load_yaw"):
        count = sum(r["shape_sums"][name]["count"] for r in statistics)
        cos = sum(np.asarray(r["shape_sums"][name]["cos"]) for r in statistics)
        sin = sum(np.asarray(r["shape_sums"][name]["sin"]) for r in statistics)
        per_vessel = circular_std_deg(cos, sin, count)
        shape[name] = {"per_vessel_deg": per_vessel, "max_deg": float(np.max(per_vessel)),
                       "mean_deg": float(np.mean(per_vessel)), "samples": int(count)}

    # ---- how much of the pooled shape statistic is seed-to-seed scatter (P1-T11's power)
    per_seed_max = np.array([float(np.max(np.asarray(r["shape_std_deg"]["chord_world"], dtype=float)))
                             for r in statistics])
    seed_cos = np.array([np.asarray(r["shape_sums"]["chord_world"]["cos"], dtype=float) for r in statistics])
    seed_sin = np.array([np.asarray(r["shape_sums"]["chord_world"]["sin"], dtype=float) for r in statistics])
    seed_mean = np.arctan2(seed_sin, seed_cos)
    shape["seed_spread"] = {
        "per_seed_max_over_vessels_deg": {"mean": float(np.mean(per_seed_max)), "sd": float(np.std(per_seed_max, ddof=1)),
                                          "min": float(np.min(per_seed_max)), "max": float(np.max(per_seed_max))},
        "per_seed_chord_mean_sd_deg_per_vessel": list(np.degrees(np.std(seed_mean, axis=0, ddof=1))),
        "per_seed_within_seed_std_mean_deg_per_vessel":
            list(np.mean(np.array([np.asarray(r["shape_std_deg"]["chord_world"], dtype=float) for r in statistics]), axis=0)),
        "reading": ("the pooled per-vessel std is the within-seed std inflated by the scatter of the seed circular means; "
                    "that scatter is a slow mode with no tau_A mixing, so the pooled statistic's standard error is set by "
                    "the number of SEEDS, not by the number of tau_A-independent samples the plan's III.3 power note uses"),
    }

    # ---- exposure, slack duty, onsets
    exposure = float(sum(r["exposure_s"] for r in statistics))
    primary_exposure = float(sum(r["primary_exposure_s"] for r in statistics))
    slack = int(sum(int(np.sum(r["slack_samples"])) for r in statistics))
    window_samples = int(sum(r["window_samples"] for r in statistics)) * N_CABLES
    onsets = int(sum(r["onsets_count"] for r in statistics))
    primary = int(sum(r["primary_onsets"] for r in statistics))
    primary_rate = primary / (N_CABLES * primary_exposure) if primary_exposure > 0 else None

    # ---- marks and events
    mark_fields = MARK_EXPORT_V1 + MARK_EXPORT_V2 + ("t_deep", "a_bar_ret_time", "driven_fraction", "return_leg_s")
    marks = {name: _pool([np.asarray(r["marks"][name]) for r in statistics]) for name in mark_fields}
    pilot_peaks = _pool([np.asarray(r["marks"]["T_peak"], dtype=float) for r in pilots])
    grid = threshold_grid(pilot_peaks)
    events = {name: _pool([np.asarray(r["events"][name]) for r in statistics])
              for name in ("cluster_max_T_peak", "regime", "n_marks", "n_bounces", "primary")}
    n_marks = int(marks["t_up"].size)
    n_events = int(events["cluster_max_T_peak"].size)
    regime_marks = np.asarray(marks["regime"]).astype(str)
    regime_events = np.asarray(events["regime"]).astype(str)

    # ---- snap exceedances per class at the grid
    exceedances = {}
    for level in grid["levels_N"]:
        peaks = np.asarray(events["cluster_max_T_peak"], dtype=float)
        mark_peaks = np.asarray(marks["T_peak"], dtype=float)
        exceedances[f"{level:.0f}"] = {
            "events_total": int(np.sum(peaks >= level)),
            "events_by_class": rg.class_counts(regime_events, peaks >= level) if n_events else rg.class_counts(np.empty(0, dtype=str)),
            "marks_total": int(np.sum(mark_peaks >= level)),
            "marks_by_class": rg.class_counts(regime_marks, mark_peaks >= level) if n_marks else rg.class_counts(np.empty(0, dtype=str)),
        }

    # ---- closure (B.7, P1-T12) over every run of the cell
    closures = [r for r in runs if r["closure"]["terminal"]]
    closure_times = sorted(float(r["closure"]["time"]) for r in closures)
    lost = [float(r["closure"]["exposure_lost"]) for r in closures]
    closure_block = {
        "runs": len(runs), "closures": len(closures),
        "fraction": len(closures) / len(runs) if runs else None,
        "fraction_ci95": binomial_interval(len(closures), len(runs)),
        "rate_per_run_second": len(closures) / float(sum(r["exposure_s"] for r in runs)) if runs else None,
        "rate_ci95_per_run_second": poisson_interval(len(closures), float(sum(r["exposure_s"] for r in runs))),
        "times_s": closure_times,
        "time_quartiles_s": (list(np.percentile(closure_times, [25, 50, 75])) if closure_times else None),
        "exposure_lost_s_total": float(sum(lost)), "exposure_lost_s_each": lost,
        "exposure_lost_fraction_of_planned": (float(sum(lost)) / (len(runs) * DURATION) if runs else None),
        "below_0.25": (len(closures) / len(runs) < CLOSURE_FRACTION_BOUND) if runs else None,
        "statistics_seed_closures": sum(1 for r in statistics if r["closure"]["terminal"]),
    }

    # ---- clean set
    clean_count = sum(np.asarray(r["clean"]["q_count"], dtype=float) for r in statistics)
    clean_sum = sum(np.asarray(r["clean"]["q_sum"], dtype=float) for r in statistics)
    clean_square = sum(np.asarray(r["clean"]["q_square"], dtype=float) for r in statistics)
    total_count = float(np.sum(clean_count))
    q_mean = float(np.sum(clean_sum) / total_count) if total_count > 0 else None
    q_var = (float(np.sum(clean_square) / total_count) - q_mean**2) if total_count > 0 else None
    clean_exposure = float(sum(r["clean"]["exposure_s"] for r in statistics))
    clean_window = float(sum(r["clean"]["window_exposure_s"] for r in statistics))

    return {
        "cell": cell.name, "intensity": cell.intensity, "T0_N": cell.pretension, "k_h": cell.heading_gain,
        "k_sigma": cell.k_sigma, "sway_limit_rad": cell.sway_limit, "a_0_m_s2": a0,
        "config_hash": statistics[0]["config_hash"] if statistics else None,
        "seeds": {"statistics": [r["seed"] for r in statistics], "pilots": [r["seed"] for r in pilots]},
        "exposure_s": exposure, "primary_exposure_s": primary_exposure,
        "cable_seconds": exposure * N_CABLES,
        "shape_deg": shape,
        "P1_T11": {
            "chord_world_stat_deg": shape["chord_world"]["max_deg"],
            "psi_stat_deg": shape["psi"]["max_deg"],
            "chord_load_stat_deg_non_gating": shape["chord_load"]["max_deg"],
            "load_yaw_std_deg": shape["load_yaw"]["max_deg"],
            "chord_world_meets_15": bool(shape["chord_world"]["max_deg"] <= SHAPE_BOUND_DEG),
            "psi_meets_15": bool(shape["psi"]["max_deg"] <= SHAPE_BOUND_DEG),
            "verdict": "PASS" if (shape["chord_world"]["max_deg"] <= SHAPE_BOUND_DEG
                                  and shape["psi"]["max_deg"] <= SHAPE_BOUND_DEG) else "FAIL",
        },
        "slack_duty": slack / window_samples if window_samples else None,
        "onsets": onsets, "primary_onsets": primary,
        "primary_onsets_2s_same_cable": int(sum(r["primary_onsets_2s_same_cable"] for r in statistics)),
        "primary_rate_per_cable_s": primary_rate,
        "primary_rate_ci95_per_cable_s": poisson_interval(primary, N_CABLES * primary_exposure),
        "onset_rate_per_cable_s": onsets / (N_CABLES * exposure) if exposure > 0 else None,
        "primary_fraction": primary / onsets if onsets else None,
        "marks": n_marks, "marks_per_cable_s": n_marks / (N_CABLES * exposure) if exposure > 0 else None,
        "events": n_events,
        "events_chained_sensitivity": int(sum(r["events_chained_sensitivity"] for r in statistics)),
        "marks_by_class": rg.class_counts(regime_marks) if n_marks else rg.class_counts(np.empty(0, dtype=str)),
        "marks_by_class_clip_reading": (rg.class_counts(np.asarray(marks["regime_clip"]).astype(str)) if n_marks
                                        else rg.class_counts(np.empty(0, dtype=str))),
        "events_by_class": rg.class_counts(regime_events) if n_events else rg.class_counts(np.empty(0, dtype=str)),
        "threshold_grid": grid,
        "snap_exceedances": exceedances,
        "closure": closure_block,
        "clean_set": {"exposure_s": clean_exposure, "window_exposure_s": clean_window,
                      "exposure_lost_s": clean_window - clean_exposure,
                      "exposure_lost_fraction": (1.0 - clean_exposure / clean_window) if clean_window > 0 else None,
                      "mu_q_N": q_mean, "sigma_q_N": math.sqrt(q_var) if q_var and q_var > 0 else None,
                      "mu_q_over_sigma_q": (q_mean / math.sqrt(q_var)) if q_var and q_var > 0 else None,
                      "mu_q_over_T0": (q_mean / cell.pretension) if q_mean else None},
        "pilots": {"marks": int(pilot_peaks.size), "T_peak_max_N": float(np.max(pilot_peaks)) if pilot_peaks.size else None,
                   "closures": sum(1 for r in pilots if r["closure"]["terminal"])},
        "wall_seconds": float(sum(r["wall_seconds"] for r in runs)),
        "diagnostics": {
            "weather_bit_equal_all": all(r["diagnostics"]["weather_bit_equal"] for r in runs),
            "marks_same_order_as_v1_all": all(r["diagnostics"]["marks_same_order_as_v1"] for r in runs),
            "marks_unmatched_total": int(sum(r["diagnostics"]["marks_unmatched"] for r in runs)),
            # The machinery's C3 check (offline W_rel against the plant's 1 ms log).  A run with no
            # re-engagement anywhere goes through the addendum-1 shim, which computes no weather-side
            # covariate at all - it holds no mark to classify - so C3 is not applicable there and is
            # reported as not checked rather than folded into a boolean.
            "wrel_checked_runs": sum(1 for r in runs if r["diagnostics"]["wrel_passes"] is not None),
            "wrel_passed_runs": sum(1 for r in runs if r["diagnostics"]["wrel_passes"] is True),
            "wrel_not_checked_runs": sum(1 for r in runs if r["diagnostics"]["wrel_passes"] is None),
            "wrel_passes_every_checked_run": all(r["diagnostics"]["wrel_passes"] for r in runs
                                                 if r["diagnostics"]["wrel_passes"] is not None),
            "orphan_marks_total": int(sum(r["diagnostics"]["orphan_marks"] for r in runs)),
            "tracker_dropped_total": int(sum(r["tracker_dropped"] for r in runs)),
            "lint_violations": sorted({v for r in runs for v in r["lint_violations"]}),
        },
        "_marks": marks,  # kept for the pooled tests, dropped before the record is written
    }


def t5_block(sets: dict[str, dict[str, np.ndarray]]) -> dict:
    """P1-T5 per class over a mark set: V_up on sqrt(2 a_bar_ret Delta) from the deepest point."""
    out = {}
    for name, marks in sets.items():
        depth_all = np.asarray(marks["depth"], dtype=float)
        # a_bar_ret is undefined at zero depth (the mark has no turnaround the law can be
        # read from), so those marks are excluded from both readings and counted.
        keep = np.flatnonzero(depth_all > 0.0)
        depth = depth_all[keep]
        v_up = np.asarray(marks["v_return"], dtype=float)[keep]
        a_time = np.asarray(marks["a_bar_ret_time"], dtype=float)[keep]
        a_depth = np.asarray(marks["a_bar_ret"], dtype=float)[keep]
        with np.errstate(invalid="ignore"):
            x_time = np.sqrt(2.0 * a_time * depth)
            x_depth = np.sqrt(2.0 * a_depth * depth)
        time_fit = through_origin(x_time, v_up)
        depth_fit = through_origin(x_depth, v_up)
        powered = bool(time_fit["n"] >= POWERED_EVENTS)
        passes = None
        if powered and time_fit["slope"] is not None and time_fit["uncentred_r2"] is not None:
            passes = bool(abs(time_fit["slope"] - 1.0) <= 0.05 and time_fit["uncentred_r2"] > 0.98)
        out[name] = {"time_mean_reading": time_fit, "depth_mean_reading": depth_fit,
                     "marks": int(depth_all.size), "marks_with_positive_depth": int(depth.size),
                     "marks_zero_depth_excluded": int(depth_all.size - depth.size),
                     "powered_20_marks": powered,
                     "verdict": ("PASS" if passes else "FAIL") if passes is not None else "UNDER-POWERED"}
    return out


def t8_block(sets: dict[str, dict[str, np.ndarray]], pretensions: dict[str, np.ndarray]) -> dict:
    out = {}
    for name, marks in sets.items():
        depth = np.asarray(marks["depth"], dtype=float)
        a_time = np.asarray(marks["a_bar_ret_time"], dtype=float)
        a_depth = np.asarray(marks["a_bar_ret"], dtype=float)
        driven = np.asarray(marks["driven_fraction"], dtype=float)
        a0 = np.asarray(pretensions[name], dtype=float) / M_FLEET
        bands = {}
        for low, high in DEPTH_BANDS:
            rows = np.flatnonzero((depth >= low) & (depth < high))
            bands[f"{low}-{high}"] = {
                "n": int(rows.size),
                "median_a_time_over_a0": float(np.nanmedian(a_time[rows] / a0[rows])) if rows.size else None,
                "median_a_depth_over_a0": float(np.nanmedian(a_depth[rows] / a0[rows])) if rows.size else None,
                "median_driven_fraction": float(np.nanmedian(driven[rows])) if rows.size else None,
            }
        below = int(np.sum(depth < DEPTH_BANDS[0][0]))
        above = int(np.sum(depth >= DEPTH_BANDS[-1][1]))
        out[name] = {"bands": bands, "marks_below_0.05m": below, "marks_above_12m": above,
                     "median_driven_fraction_all": float(np.nanmedian(driven)) if driven.size else None}
    return out


def _subset(marks: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(values)[rows] for name, values in marks.items()}


def analyse(runs: list[dict]) -> dict:
    failed = [{"cell": r["cell"], "seed": r["seed"], "error": r.get("error")} for r in runs if "error" in r]
    failed_cells = {item["cell"] for item in failed}
    cells = {}
    for cell in CELLS:
        chosen = [r for r in runs if r["cell"] == cell.name and "error" not in r]
        if not chosen:
            continue
        cells[cell.name] = cell_block(cell, chosen)
        if cell.name in failed_cells:
            cells[cell.name]["invalid"] = "a run of this cell failed; the cell carries no verdict"
            cells[cell.name]["P1_T11"]["verdict"] = "INVALID"

    # ---- pooled mark set for P1-T5 / P1-T8 (statistics seeds, every cell)
    pooled: dict[str, list[np.ndarray]] = {}
    pooled_T0: list[np.ndarray] = []
    for name, block in cells.items():
        marks = block["_marks"]
        for key, values in marks.items():
            pooled.setdefault(key, []).append(np.asarray(values))
        pooled_T0.append(np.full(int(np.asarray(marks["t_up"]).size), CELL_BY_NAME[name].pretension))
    all_marks = {key: _pool(values) for key, values in pooled.items()}
    all_T0 = _pool(pooled_T0)
    regime = np.asarray(all_marks["regime"]).astype(str) if all_marks else np.empty(0, dtype=str)

    sets = {"all": all_marks}
    pretensions = {"all": all_T0}
    for name in rg.CLASSES:
        rows = np.flatnonzero(regime == name)
        sets[name] = _subset(all_marks, rows)
        pretensions[name] = all_T0[rows]
    t5 = {"pooled_over_cells": t5_block(sets)}
    t8 = {"pooled_over_cells": t8_block(sets, pretensions)}

    # per cell, where a class holds >= 20 marks
    per_cell_t5 = {}
    per_cell_t8 = {}
    for name, block in cells.items():
        marks = block["_marks"]
        cell_regime = np.asarray(marks["regime"]).astype(str)
        cell_sets = {"all": marks}
        cell_T0 = {"all": np.full(int(np.asarray(marks["t_up"]).size), CELL_BY_NAME[name].pretension)}
        for cls in rg.CLASSES:
            rows = np.flatnonzero(cell_regime == cls)
            if rows.size >= POWERED_EVENTS:
                cell_sets[cls] = _subset(marks, rows)
                cell_T0[cls] = np.full(rows.size, CELL_BY_NAME[name].pretension)
        per_cell_t5[name] = t5_block(cell_sets)
        per_cell_t8[name] = t8_block(cell_sets, cell_T0)
    t5["per_cell"] = per_cell_t5
    t8["per_cell"] = per_cell_t8

    # ---- P1-T10': the population rule over the six selected-controller cells
    gating = [c.name for c in CELLS if c.k_sigma == SELECTED_K_SIGMA]
    population = {}
    for cls in rg.CLASSES:
        per_cell = {}
        for name in gating:
            block = cells[name]
            levels = [level for level, counts in block["snap_exceedances"].items()
                      if counts["events_by_class"].get(cls, 0) >= POWERED_EVENTS]
            per_cell[name] = {"levels_with_20_events": len(levels), "levels": levels,
                              "grid_levels": len(block["threshold_grid"]["levels_N"]),
                              "events_in_class": block["events_by_class"].get(cls, 0),
                              "marks_in_class": block["marks_by_class"].get(cls, 0)}
        qualifying = [name for name, item in per_cell.items() if item["levels_with_20_events"] >= POPULATION_LEVELS]
        population[cls] = {"per_cell": per_cell, "cells_with_3_powered_levels": len(qualifying),
                           "qualifying_cells": qualifying,
                           "meets_population_rule": bool(len(qualifying) >= POPULATION_CELLS)}
    population_comparison_cells = {
        cls: {name: {"events_in_class": cells[name]["events_by_class"].get(cls, 0),
                     "marks_in_class": cells[name]["marks_by_class"].get(cls, 0)}
              for name in cells if CELL_BY_NAME[name].k_sigma != SELECTED_K_SIGMA}
        for cls in rg.CLASSES}

    # ---- P1-T12 non-degeneracy
    t12 = {}
    for intensity in (0.35, 0.5):
        names = [c.name for c in CELLS if c.intensity == intensity and c.k_sigma == SELECTED_K_SIGMA]
        below = [name for name in names if cells[name]["closure"]["below_0.25"]]
        t12[f"{intensity}"] = {"cells": names, "cells_below_0.25": below,
                               "fractions": {name: cells[name]["closure"]["fraction"] for name in names},
                               "closures": {name: cells[name]["closure"]["closures"] for name in names},
                               "satisfied": bool(below)}
    total_closures = sum(cells[name]["closure"]["closures"] for name in cells)
    total_runs = sum(cells[name]["closure"]["runs"] for name in cells)
    total_exposure = float(sum(cells[name]["exposure_s"] for name in cells))
    t12["total_closures_all_cells"] = total_closures
    t12["total_runs"] = total_runs
    # The plan's two degeneracy clauses, scored separately and both reported.
    clause_a = bool(all(t12[f"{i}"]["satisfied"] for i in (0.35, 0.5)))
    clause_b_degenerate = bool(total_closures == 0)
    t12["clause_a_one_cell_below_0.25_at_each_intensity"] = clause_a
    t12["clause_b_rate_carries_no_interval"] = {
        "degenerate": clause_b_degenerate,
        "closures": total_closures,
        "phase_rate_per_run_second": total_closures / total_exposure if total_exposure > 0 else None,
        "phase_rate_ci95_per_run_second": poisson_interval(total_closures, total_exposure),
        "reading": ("with zero closures the exact Poisson interval is one-sided, [0, upper]: the terminal-event rate is "
                    "bounded, not measured, which is the plan's second degeneracy clause"),
    }
    t12["non_degenerate"] = bool(clause_a and not clause_b_degenerate)
    if clause_a and clause_b_degenerate:
        t12["verdict"] = ("DEGENERATE on the plan's SECOND clause only: every scored cell keeps its closure fraction below "
                          "0.25 (clause A satisfied at both intensities), but no closure occurred anywhere in the phase, so "
                          "the terminal-event rate carries no two-sided interval. Consequence 'no radial-rate result may be "
                          "carried out of it' is reported and is moot here, since Phase 2 does not launch and no radial rate "
                          "is carried forward")
    elif clause_a:
        t12["verdict"] = "NON-DEGENERATE"
    else:
        t12["verdict"] = "DEGENERATE: some intensity Phase 2 would use has no cell below a 0.25 closure fraction"

    # ---- the trade-off table
    table = []
    for cell in CELLS:
        block = cells.get(cell.name)
        if block is None:
            continue
        levels = block["threshold_grid"]["levels_N"]
        third = levels[2] if len(levels) >= 3 else None
        third_events = block["snap_exceedances"][f"{third:.0f}"]["events_total"] if third is not None else 0
        populated = any(population[cls]["per_cell"].get(cell.name, {}).get("levels_with_20_events", 0) >= POPULATION_LEVELS
                        for cls in rg.CLASSES)
        table.append({
            "cell": cell.name, "intensity": cell.intensity, "T0_kN": cell.pretension / 1000.0, "k_h": cell.heading_gain,
            "k_sigma": cell.k_sigma,
            "chord_world_std_deg": block["P1_T11"]["chord_world_stat_deg"],
            "chord_load_std_deg": block["P1_T11"]["chord_load_stat_deg_non_gating"],
            "psi_std_deg": block["P1_T11"]["psi_stat_deg"],
            "slack_duty": block["slack_duty"],
            "onsets": block["onsets"], "primary_onsets": block["primary_onsets"],
            "primary_rate_per_cable_s": block["primary_rate_per_cable_s"],
            "marks": block["marks"], "events": block["events"],
            "marks_by_class": block["marks_by_class"],
            "grid_levels_N": levels,
            "events_at_third_level": third_events, "third_level_N": third,
            "meets_H1": block["P1_T11"]["verdict"] == "PASS",
            "meets_population_rule_in_some_class": bool(populated),
            "chord_gap_ratio": block["P1_T11"]["chord_world_stat_deg"] / SHAPE_BOUND_DEG,
            "psi_gap_ratio": block["P1_T11"]["psi_stat_deg"] / SHAPE_BOUND_DEG,
            "event_gap_ratio": (third_events / POWERED_EVENTS) if third is not None else 0.0,
        })

    # ---- which cell is least far from meeting both halves (declared measure: the two gaps)
    closest = []
    for row in table:
        if row["k_sigma"] != SELECTED_K_SIGMA:
            continue
        shape_shortfall = max(row["chord_gap_ratio"], row["psi_gap_ratio"])
        events = row["events_at_third_level"]
        population_shortfall = (POWERED_EVENTS / events) if events > 0 else float("inf")
        closest.append({
            "cell": row["cell"], "intensity": row["intensity"], "T0_kN": row["T0_kN"],
            "shape_shortfall_factor": shape_shortfall,
            "population_shortfall_factor": population_shortfall,
            "events_at_third_level": events, "third_level_N": row["third_level_N"],
            "combined": (max(shape_shortfall, 1.0) * max(population_shortfall, 1.0)),
        })
    closest.sort(key=lambda item: item["combined"])
    closest_block = {
        "measure": ("shape shortfall = max(chord std, psi std)/15 deg; population shortfall = 20 / events at the cell's own "
                    "third grid level; the combined figure is the product of the two, each floored at 1 (a cell meeting a "
                    "half contributes 1 to it). Descriptive, declared with the trade-off table; not a test"),
        "ranked": closest,
        "least_far": closest[0]["cell"] if closest else None,
    }

    predictions = _prediction_comparison(cells)

    for block in cells.values():
        block.pop("_marks", None)

    return {
        "schema": "v2-phase1f-results-1",
        "failed_runs": failed,
        "runs_no_reengagement_anywhere": [{"cell": r["cell"], "seed": r["seed"]} for r in runs
                                          if r.get("no_reengagement_in_run")],
        "cells": cells,
        "P1_T5": t5,
        "P1_T8": t8,
        "P1_T8_plan_executed_medians_a_bar_ret_over_a0": list(PLAN_A_BAR_RET_OVER_A0),
        "P1_T10prime": {"by_class": population, "comparison_cells": population_comparison_cells,
                        "rule": ">= 20 events at >= 3 grid levels in >= 3 cells, scored over the six k_sigma = 3 cells",
                        "any_class_populated": bool(any(population[cls]["meets_population_rule"] for cls in rg.CLASSES))},
        "P1_T11": {name: cells[name]["P1_T11"] for name in cells},
        "P1_T12": t12,
        "trade_off": table,
        "closest_to_both": closest_block,
        "predictions": predictions,
    }


def _prediction_comparison(cells: dict) -> dict:
    out: dict = {"primary_onset_rate": {}}
    if PHASE2_PREDICTIONS_PATH.exists():
        import json

        with PHASE2_PREDICTIONS_PATH.open("rb") as handle:
            committed = json.load(handle)
        out["phase2_predictions_sha256"] = sha256_file(PHASE2_PREDICTIONS_PATH)
        theta_rice = committed["III_1"]["2_declustered_primary_onset_rate"]["value"]
        rice_only = committed["intensity_0.5_out_of_H1_comparisons"]
        for name, (kind, key) in PREDICTION_KEYS.items():
            block = cells.get(name)
            if block is None:
                continue
            measured = block["primary_rate_per_cable_s"]
            if kind == "theta_rice":
                predicted = float(theta_rice[key]["model_mean"])
                out["primary_onset_rate"][name] = {
                    "kind": "committed theta x Rice (intensity 0.35)", "prediction_key": key,
                    "predicted_per_cable_s": predicted, "measured_per_cable_s": measured,
                    "measured_over_predicted": (measured / predicted) if (measured is not None and predicted > 0) else None,
                    "predicted_primary_onsets_in_cell": predicted * N_CABLES * block["primary_exposure_s"],
                    "measured_primary_onsets_in_cell": block["primary_onsets"],
                    "measured_ci95_per_cable_s": block["primary_rate_ci95_per_cable_s"],
                }
            else:
                predicted = float(rice_only[key]["rice_model_mean_per_cable_s"])
                out["primary_onset_rate"][name] = {
                    "kind": "analytic Rice only, no committed theta (intensity 0.5, out-of-(H1) comparison)",
                    "prediction_key": key, "rice_per_cable_s": predicted, "measured_per_cable_s": measured,
                    "measured_over_rice": (measured / predicted) if (measured is not None and predicted > 0) else None,
                    "theta_band_at_0.35_for_reference": [0.26, 0.44],
                    "measured_primary_onsets_in_cell": block["primary_onsets"],
                    "measured_ci95_per_cable_s": block["primary_rate_ci95_per_cable_s"],
                }
    if PHASE1_PREDICTIONS_PATH.exists():
        import json

        with PHASE1_PREDICTIONS_PATH.open("rb") as handle:
            phase1 = json.load(handle)
        out["phase1_predictions_sha256"] = sha256_file(PHASE1_PREDICTIONS_PATH)
        out["P1_T5_committed_scripted"] = phase1["P1-T5"]
        out["P1_T8_committed_scripted"] = phase1["P1-T8"]
    if P1E_RESULTS_PATH.exists():
        import json

        with P1E_RESULTS_PATH.open("rb") as handle:
            p1e = json.load(handle)
        out["p1e_results_sha256"] = sha256_file(P1E_RESULTS_PATH)
        out["p1e_selected_cell"] = p1e["p1_t11"]
        same = cells.get("i035_T1000_ks3")
        if same is not None:
            out["P1_T11_same_cell_as_the_pilot"] = {
                "cell": "i035_T1000_ks3",
                "note": ("the identical spec the Phase 1(e) pilot scored P1-T11 on (T0 = 1 kN, k_h = 763, k_sigma = 3, "
                         "clamp 0.349, intensity 0.35), here on 20 statistics seeds instead of the pilot's 5"),
                "pilot_5_seeds": {"chord_world_deg": p1e["p1_t11"]["chord_world_stat_deg"],
                                  "psi_deg": p1e["p1_t11"]["psi_stat_deg"]},
                "this_phase_20_seeds": {"chord_world_deg": same["P1_T11"]["chord_world_stat_deg"],
                                        "psi_deg": same["P1_T11"]["psi_stat_deg"]},
                "verdict_unchanged": same["P1_T11"]["verdict"] == "FAIL",
            }
    return out


def run_analysis() -> dict:
    import json

    if not RUNS_PATH.exists():
        raise SystemExit("the runs must be computed before the analysis")
    with RUNS_PATH.open("rb") as handle:
        payload = json.load(handle)
    results = analyse(payload["runs"])
    results["declarations_sha256"] = sha256_file(DECLARATIONS_PATH)
    results["addendum_1_sha256"] = sha256_file(ADDENDUM_PATH) if ADDENDUM_PATH.exists() else None
    results["runs_sha256"] = sha256_file(RUNS_PATH)
    results["sway_addendum_sha256"] = sha256_file(SWAY_ADDENDUM_PATH) if SWAY_ADDENDUM_PATH.exists() else None
    results["source"] = source_state()
    write_bytes(RESULTS_PATH, json_bytes(results))
    return results


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("declare", "compute", "analyse"))
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    from tether.campaign.v2 import phase1_stochastic as module

    if arguments.command == "declare":
        print(module.declare())
        return
    if arguments.command == "compute":
        runs = module.compute(arguments.workers)
        print(json.dumps({"runs": len(runs), "marks": sum(r["n_marks"] for r in runs),
                          "wall_seconds": sum(r["wall_seconds"] for r in runs)}, indent=1))
        return
    results = module.run_analysis()
    print(json.dumps(results["trade_off"], indent=1, default=str))


if __name__ == "__main__":
    main()
