"""Phase 1 deterministic cells (a)-(d) on the production plant, and tests P1-T1..T9.

Plan v2 Part V Phase 1 ("Objectives", "Cells", "Observables", "Tests", "Outcome
matrix"), II.4-II.6, Appendix A, Appendix B.  The committed predictions this module is
scored against are ``records/v2/phase1/phase1_predictions.json`` and
``records/v2/phase1/claim_a_declarations.json``; their sha256 are recorded in the
results and neither file is re-derived here.

Cells
-----
(a) **Rigid limit** - the unilateral plant against Phase 0's bilateral transit
    (P1-T1).  Two readings, both scored: v1's isolated harness
    (``phase1_deterministic.deterministic_fleet_acceptance``) at the Phase 0 nominal
    thrust, and the production closed loop under the v2 controller at the three v2
    pretensions against ``plant.run_transit_log`` at the same thrust.
(b) **Impact regression** - ``records/phase1/impact_table.json`` is PINNED.  The 12
    (pretension, cable) pairs x 13 closing speeds are re-run with the harness that
    produced it and regressed against it; Z (isolated two-body) and Z_fleet
    (full-formation) are re-pinned (P1-T2, P1-T3).
(c) **Scripted sustained excursions** - square W^c gusts of amplitude lambda T0,
    lambda in {0.6, 0.9, 1.05, 1.25, 1.5, 2.0}, t_x in {0.2, 0.35, 0.71, 1.5, 3.4, 6,
    10} s, T0 in {0.6, 1.0, 1.4} kN: 126 runs x 40 s.
(d) **Scripted 0.3 s impulses** at A in {2, 3, 5, 8} kN, T0 = 1 kN: 4 runs x 40 s.
(t9) **Energy sanity** - the force-free, drag-free two-body cable cell at c = 0 and a
    0.5 ms step (P1-T9).

Forcing (committed in ``claim_a_declarations.json``, reproduced verbatim here): a
vessel-only body force on the test vessel (cable 2, the centre cable) of magnitude
F = (c_A/c_eff) W^c = 1.06364 W^c along -d_i (world -x, the parallel formation's
equilibrium chord direction), square in time over [10 s, 10 s + t_x), zero on every
other body and on the load.  It is delivered through the plant's *weather* channel -
the array is deterministic and declared, the stochastic generator is never called
(``weather_scale = 0``) - so that the plant's own W_rel log, the offline W^c series and
the (H4') classification of ``campaign/v2/regime.py`` all see the same forcing and the
summariser's bit-equality check applies.  ``HullForces`` adds weather and scripted
forces to the same generalized force with the same 10 ms zero-order hold, so the two
routes are the same physics; every declared t_x is a whole number of 10 ms samples.

Open machinery rulings (stage 2, unruled): this module uses the LITERAL plan readings -
``t_x`` over the full merged W^c episode (IV.4/B.4) and the anchored bounce rule of B.1 -
and reports the clipped-episode and chained-bounce variants beside them as secondary.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tether.campaign.common import (
    RECORDS,
    ROOT,
    json_bytes,
    relative,
    run_pool,
    sha256_bytes,
    sha256_file,
    source_state,
    write_bytes,
)
from tether.campaign.fleet_run import FleetRunSpec, build_run, run_to_end
from tether.campaign.v2 import events as ev
from tether.campaign.v2 import regime as rg
from tether.campaign.v2.summaries import summarize_run_v2
from tether.physics import constants
from tether.physics.fleet import formation_geometry, operating_point
from tether.theory import drag_excursion as dx

# ----------------------------------------------------------------------------- paths

PHASE1_DIR = RECORDS / "v2" / "phase1"
DECLARATIONS_PATH = PHASE1_DIR / "scripted_declarations.json"
RESULTS_PATH = PHASE1_DIR / "scripted_results.json"
RUNS_PATH = PHASE1_DIR / "scripted_runs.json"
PREDICTIONS_PATH = PHASE1_DIR / "phase1_predictions.json"
CLAIM_A_PATH = PHASE1_DIR / "claim_a_declarations.json"
IMPACT_TABLE_PATH = RECORDS / "phase1" / "impact_table.json"
MODULE_PATH = ROOT / "tether" / "campaign" / "v2" / "phase1_scripted.py"

# ----------------------------------------------------------------------------- pinned configuration

TEST_CABLE = 2  # the centre cable, impact-table position c2
GUST_START = 10.0
RUN_SECONDS = 40.0
LAMBDAS = (0.6, 0.9, 1.05, 1.25, 1.5, 2.0)
DURATIONS = (0.2, 0.35, 0.71, 1.5, 3.4, 6.0, 10.0)
PRETENSIONS = (600.0, 1000.0, 1400.0)
HEADING_GAINS = {600.0: 477.0, 1000.0: 763.0, 1400.0: 1050.0}  # 1.5 x the stability boundary
K_SIGMA = 3.0
SWAY_LIMIT = 0.349  # rad (records/v2/phase0/sway_addendum_1.json)
TRIM_GAIN = 100.0
D_AMPLITUDES = (2000.0, 3000.0, 5000.0, 8000.0)
D_PRETENSION = 1000.0
D_DURATION = 0.3
VESSEL_FORCE_PER_WC = dx.VESSEL_FORCE_PER_WC  # c_A / c_eff = 1.06364
M_FLEET = dx.M_FLEET  # 523.9 kg, the Phase 1 pivot
M_EFF = dx.M_EFF
IMPACT_FACTOR = 0.880
DEPTH_BANDS = ((0.05, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 12.0))
CLASSES = ("N", "R1", "transitional", "R2")
LATE_WINDOW = 2.0  # s, the P1-T6 drift window [t_off - 2 s, t_off]

# cell (b), the pinned table's own grid and harness
TABLE_PRETENSIONS = (600.0, 1000.0, 1400.0, 1800.0)
TABLE_CABLES = (0, 1, 2)
TABLE_SPEEDS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
TWO_BODY_SPEEDS = (0.25, 0.5, 1.0, 2.0, 3.0)

# cell (a)
RIGID_DURATION = 30.0
RIGID_SAMPLE = 0.1

# P1-T9
T9_SPEEDS = (0.25, 0.5, 1.0, 2.0, 3.0)
T9_STEP = 5.0e-4
T9_DURATION = 0.5
T9_ENERGY_LIMIT = 0.005
T9_IMPULSE_LIMIT = 1.0e-9

# thresholds (plan v2 Part V Phase 1, "Tests")
T1_DISTANCE = 0.01
T1_TENSION = 0.03
T2_TABLE = 0.01
T2_R2 = 0.99
T3_MASS = 0.10
T4_SYMMETRY = 0.05
T4_DEPTH = 0.10
T5_SLOPE = 0.05
T5_R2 = 0.98
T5_MIN_MARKS = 20
T6_CROSSING = 0.05
T7_RATIO = 0.15
T7_DEPTH_FLOOR = 0.05  # m; a closed-form depth below this carries no ratio test


# ----------------------------------------------------------------------------- declarations


def _declarations() -> dict:
    return {
        "schema_version": 1,
        "task": "Phase 1 deterministic cells (a)(b)(c)(d) on the production plant; tests P1-T1..T9",
        "date": "2026-09-13",
        "declared_before_any_result": True,
        "plan": "ref/tail_of_the_tether_plan_v2.md: Part V Phase 1 (Objectives, Cells, Observables, "
                "Tests, Outcome matrix), II.4-II.6, Appendix A, Appendix B.1-B.4, B.6, B.7",
        "committed_predictions_scored_against": {
            "phase1_predictions": relative(PREDICTIONS_PATH),
            "claim_a_declarations": relative(CLAIM_A_PATH),
            "rule": "where a committed prediction and the plan's Appendix disagree (the three-piece "
                    "map's T_peak: committed 6.18/8.89/13.04/17.15 kN against the Appendix's "
                    "4.6/7.4/11.7/15.9 kN at A = 2/3/5/8 kN), the committed prediction is scored and "
                    "both are reported",
        },
        "open_machinery_rulings": {
            "t_x": "LITERAL (IV.4/B.4): the FULL merged W^c episode, regime.mark_covariates 't_x' / "
                   "'regime'. The slack-clipped variant ('t_x_clip' / 'regime_clip') is reported beside it.",
            "bounces": "LITERAL (B.1): the anchored rule, events.decluster_marks(rule='anchored'). "
                       "The chained variant is reported beside it.",
        },
        "plant": {
            "module": "tether.campaign.fleet_run.build_run (the production closed loop)",
            "formation": "parallel",
            "pretensions_N": list(PRETENSIONS),
            "heading_gain_per_pretension": {str(int(t)): HEADING_GAINS[t] for t in PRETENSIONS},
            "heading_gain_rule": "k_h = 1.5 x the stability boundary at each T0 (477/763/1050 N m/rad "
                                 "at 0.6/1.0/1.4 kN), the owner's graded-grid ruling",
            "trim_gain": TRIM_GAIN,
            "k_sigma": K_SIGMA,
            "sway_limit_rad": SWAY_LIMIT,
            "sway_law": "theta_ref = theta_0 - clamp(k_sigma wrap(sigma_hat - phi), +-20 deg) "
                        "(records/v2/phase0/sway_addendum_1.json); clause A at intensity 0.35 selects k_sigma = 3",
            "drag_law": "linear",
            "cable_mode": "recording (no severance; every re-engagement is recorded)",
            "physics_step_s": 5.0e-4,
            "stochastic_weather": "NONE. weather_scale = 0; the stochastic generator is never called.",
            "warmup_s": 0.0,
            "duration_s": RUN_SECONDS,
            "initial_state": "fleet_run's steady-tow equilibrium (all cables taut at T0)",
        },
        "forcing_COMMITTED": {
            "pattern": "vessel-only body force on the test vessel (cable 2, centre, impact-table "
                       "position c2), magnitude F = (c_A/c_eff) W^c = 1.06364 W^c, applied at the "
                       "vessel's centre of mass along -d_i (world -x), square in time on "
                       "[10 s, 10 s + t_x), zero on every other body and on the load",
            "delivery": "through the plant's weather array (deterministic, declared, built here), not "
                        "through scripted_force: HullForces adds both to the same generalized force "
                        "with the same 10 ms zero-order hold, and the weather route makes the plant's "
                        "W_rel log, the offline W^c series and the (H4') classification see the same "
                        "forcing (summariser bit-equality check applies)",
            "W_c_identity": "W^c = c_eff (W_L/c_L - W_A/c_A).d = c_eff F/c_A = W^c exactly when d = +x; "
                            "the measured W^c carries the chord's actual rotation",
            "source": "records/v2/phase1/claim_a_declarations.json, phase1_scripted.body_force_pattern_COMMITTED",
        },
        "cells": {
            "a": {
                "what": "rigid-limit regression against Phase 0 (P1-T1)",
                "reading_1_v1_harness": "tether.physics.phase1_deterministic.deterministic_fleet_acceptance"
                                        "(duration = 0.25 s) at the Phase 0 nominal thrust 1305 N: the "
                                        "unilateral full formation against plant.run_steady_transit",
                "reading_2_production_plant": "build_run(parallel, T0, k_h, k_sigma = 3, sway_limit = 0.349, "
                                              "weather_scale = 0, warmup = 0, duration = 30 s) at T0 in "
                                              "{600, 1000, 1400} N against plant.run_transit_log at the same "
                                              "thrust (1.31818 T0) for the same duration, sampled at 0.1 s",
                "observables": "mean vessel distance travelled over the run, mean cable tension over the "
                               "run, minimum elongation. The production 10 ms state log ends one sample "
                               "short of the planned end, so the bilateral reference distance is linearly "
                               "interpolated to the production log's own elapsed time.",
                "scoring": "P1-T1 passes iff BOTH readings pass: |distance/reference - 1| <= 1%, "
                           "|mean tension/reference - 1| <= 3%, and min elongation > 0 (no slack). "
                           "Reading 2 must pass at every pretension.",
            },
            "b": {
                "what": "impact regression against the PINNED table (P1-T2, P1-T3)",
                "pinned_table": relative(IMPACT_TABLE_PATH),
                "harness": "tether.campaign.phase1_close.run_impact_job, the harness that produced the "
                           "pinned table, at its own grid: 4 pretensions x 3 cable positions x 13 speeds "
                           "(the plan's '12 (pretension, cable) pairs x 13 closing speeds')",
                "regression": "|peak/pinned_peak - 1| <= 1% at every one of the 156 points",
                "R2": "through-origin fit of peak on speed over speeds >= 1 m/s, per (T0, cable), exactly "
                      "as build_impact_table defines the pinned 'asymptotic_impedance'; uncentred R2 > 0.99 "
                      "required at every pair. The baseline-subtracted slope is reported beside it.",
                "Z_repin": "isolated two-body impedance: phase1_mechanics.run_two_body_impact(speed, "
                           "damping = 1800, 1 ms, continuous) at speeds (0.25, 0.5, 1, 2, 3), through-origin "
                           "slope of the first tension peak on speed (v1's pin 7993.513068475832)",
                "Z_fleet_repin": "full-formation impedance: phase1_deterministic.deterministic_fleet_acceptance "
                                 "P1-T2b, the same through-origin slope on the same speeds with the other four "
                                 "cables taut at pretension (v1's pin 8304.485029550166)",
                "T3": "m_eff_hat = Z^2/(f^2 k) against m_eff = m_A m_L/(m_A + m_L) = 483.87 kg; "
                      "m_fleet_hat = Z_fleet^2/(f^2 k) against [1/m_A + 1/(m_L + 4 m_A)]^-1 = 534.55 kg; "
                      "f = 0.880, k = 1.7e5 N/m; each within 10%, reported side by side",
            },
            "c": {
                "grid": "lambda in {0.6, 0.9, 1.05, 1.25, 1.5, 2.0} x t_x in {0.2, 0.35, 0.71, 1.5, 3.4, "
                        "6, 10} s x T0 in {600, 1000, 1400} N = 126 runs x 40 s",
                "main_mark": "the first re-engagement on the test cable (cable 2) whose onset lies at or "
                             "after the gust start (10 s). Later marks on cable 2 are bounces/members; "
                             "marks on the other four cables are 'collateral' and are reported separately.",
                "H4prime_class": "PRIMARY: the class the v2 machinery assigns from the weather-side "
                                 "covariates of the run (regime.mark_covariates on the declared weather "
                                 "array). DECLARED ANALYTIC class, reported beside it and checked for "
                                 "agreement: N if lambda <= 1, else classify(t_x, lambda T0, T0) - so "
                                 "t_x = 3.4 s (< 2 tau_A = 3.4286 s) is transitional, not R2.",
            },
            "d": {
                "grid": "A in {2, 3, 5, 8} kN, t_g = 0.3 s, T0 = 1 kN (A is the W^c amplitude)",
                "scoring": "the plant's measured T_peak against the committed three-piece-map/full-ODE "
                           "prediction (6178/8886/13039/17155 N, phase1_predictions.json cells_d "
                           "'committed_plant_prediction_T_peak_N') and against the plan's Appendix "
                           "(4600/7400/11700/15900 N); both reported, the committed one scored",
            },
            "t9": {
                "what": "energy sanity (P1-T9)",
                "cell": "phase1_mechanics.run_two_body_impact(speed, damping = 0, duration = 0.5 s, "
                        "integration_step = 0.5 ms, continuous fixed-step RK3): force-free, drag-free, "
                        "gravity-free, c = 0",
                "speeds": list(T9_SPEEDS),
                "thresholds": "max_t |E(t) - E(0)|/E(0) <= 0.005 with E = kinetic + spring potential; and "
                              "the cable's two applied forces opposite: max_t |F_load + F_vessel| <= 1e-9 N "
                              "and |sum_t (F_load + F_vessel) dt| <= 1e-9 N s",
            },
        },
        "observables_per_run": [
            "shape: chord-angle circular std (world and load frames), psi std, load-yaw std (H1 covariate)",
            "closure: terminal flag, time, cable, exposure lost (B.7)",
            "marks: t_up, u_entry, depth, dwell, v_return, T_peak, n_maxima, cable (v1 summariser)",
            "marks_v2: (H4') covariates and class, t_x, W^c_max, v_s, episode count, a_bar_ret (depth-mean), "
            "B.1 event id / role / parent, primary flag",
            "events: B.1 anchored events, cluster maximum of T_peak, bounce and member counts",
            "main mark extras from the 1 ms log: t_onset, t_deep, Delta_g at shut-off, max depth, "
            "post-gust deepening, V_up, a_bar_ret time-mean and depth-mean, driven fraction of the return leg, "
            "own peak tension over one engagement period, the P1-T6 late-window drift function",
        ],
        "tests": {
            "P1-T1": {
                "statement": "rigid-limit regression: distance, mean tension",
                "threshold": "within 1% and 3%; no slack",
                "scored_on": "cell (a), both readings (see cells.a.scoring)",
            },
            "P1-T2": {
                "statement": "impact-table regression; v_b = Z^-1(T_b) primary",
                "threshold": "table within 1%; R2 > 0.99 above 1 m/s; Z, Z_fleet re-pinned",
                "scored_on": "cell (b): all 156 points within 1% of the pinned table, uncentred R2 > 0.99 "
                             "for every (T0, cable) pair above 1 m/s, and Z / Z_fleet re-pinned and reported",
            },
            "P1-T3": {
                "statement": "m_eff from Z, m_fleet from Z_fleet, against their inertial estimates",
                "threshold": "within 10% each, reported side by side",
            },
            "P1-T4": {
                "statement": "ballistic symmetry (Prop. 2), class-N main marks",
                "threshold": "|V_up - u|/u < 0.05; depth within 10% of u^2/(2a), a = (1 - lambda) T0/m_fleet",
                "scored_on": "the class-N main marks of the test cable in cell (c) (lambda <= 1 by "
                             "construction, i.e. no W^c > T0 exceedance). PASS iff every scored mark passes "
                             "both clauses; pass fractions reported. Collateral class-N marks on the other "
                             "four cables are reported as a secondary population with a = T0/m_fleet "
                             "(they carry no W^c), and gate nothing.",
                "committed_prediction": "symmetry error 0.175 on the marks that lie inside the gust and "
                                        "1.03 / 1.49 on the two that do not: P1-T4 predicted FAIL",
            },
            "P1-T5": {
                "statement": "closing speed (Prop. 3'): V_up against sqrt(2 a_bar_ret Delta) from the "
                             "deepest point, per class",
                "threshold": "slope within 5% of one, R2 > 0.98, per class holding >= 20 marks",
                "readings": "PRIMARY (the plan's literal law, II.4: a_bar_ret is the mean acceleration over "
                            "the return leg alone, so that 2 a_bar_ret Delta = V_up^2): the depth-mean "
                            "a_bar_ret = V_up^2/(2 Delta), for which the regression is an identity and the "
                            "slope is 1 by construction - this is reported as a vacuous pass and said to be "
                            "vacuous. SECONDARY (informative): the time-mean a_bar_ret = V_up/(t_up - t_deep), "
                            "committed at slope 0.99 / 0.98 / 0.84 / 0.91 for N / R1 / R2 / transitional.",
                "population": "test-cable main marks (one per run); all marks including collateral reported beside",
            },
            "P1-T6": {
                "statement": "drift criterion: sign of the net radial acceleration against W^c - T0",
                "threshold": "fitted crossing within 5% of lambda = 1",
                "readings": "PRIMARY (drift form, committed at lambda_c = 0.9879): on the t_x = 10 s runs, "
                            "the late-window drift function y(lambda) = mean over [t_off - 2 s, t_off] of "
                            "(T + c_eff,f e') / T0 on the test cable, c_eff,f = (1/c_A + 1/c_Lf)^-1, "
                            "c_Lf = c_L + 4 c_A; the crossing is the zero of the linear fit of y on lambda, "
                            "per T0. SECONDARY: the onset-acceleration form, committed at 1.055 (m_Lf) / "
                            "1.077 (m_fleet), which gives the opposite verdict. Both reported; the drift "
                            "form is scored, as the plan's criterion is about the drift. A t_x = 10 s run "
                            "terminated by formation closure (B.7) before the shut-off holds no late "
                            "window; it is censored out of the fit and its lambda is reported.",
            },
            "P1-T7": {
                "statement": "drag-limited depth (Prop. 4')",
                "threshold": "Delta_g against v_d [t_x - tau_A (1 - exp(-t_x/tau_A))] and post-gust "
                             "deepening against tau_A [v_s - v_T ln(1 + v_s/v_T)], both within 15%; depth "
                             "linear in t_x above tau_A, a quadratic fit not preferred by AIC",
                "scored_on": "the test-cable main marks of the lambda > 1 cells whose closed-form Delta_g "
                             "exceeds 5 cm and whose slack interval is still open at shut-off. A (T0, lambda) "
                             "cell passes when every such point satisfies both 15% clauses; P1-T7 passes when "
                             "every powered cell passes. Points censored by formation closure (B.7) carry no "
                             "ratio test and are counted and reported.",
                "AIC": "per (T0, lambda) with lambda > 1, over the uncensored points with t_x >= tau_A = "
                       "1.714 s (t_x in {3.4, 6, 10}; t_x = 1.5 s is below tau_A and excluded), least-squares "
                       "fits of Delta_g on t_x: AIC = n ln(RSS/n) + 2 k with k = 3 (linear) and k = 4 "
                       "(quadratic); the quadratic is 'preferred' iff AIC_quad < AIC_lin. Reported per cell "
                       "and pooled; a cell with fewer than 4 uncensored points carries no AIC.",
                "outcome": "T7 failing WITH a quadratic depth withdraws Prop. 4' and is NO-GO for Claim A; "
                           "T7 failing while the linear fit is still preferred is a quantitative miss of the "
                           "closed form, not a refutation of linear drag, and is reported as such",
            },
            "P1-T8": {
                "statement": "a_bar_ret/a_0 by depth band and class, a_0 = T0/m_fleet, m_fleet = 523.9 kg; "
                             "driven fraction",
                "threshold": "pinned per class; a covariate, not a pass/fail",
                "bands": [f"{lo}-{hi}" for lo, hi in DEPTH_BANDS],
                "committed_prediction": "time-mean medians 0.83 / 0.71 / 0.60 / 0.37 over all classes in the "
                                        "four bands",
            },
            "P1-T9": {
                "statement": "energy sanity, c = 0, no drag, 0.5 ms",
                "threshold": "energy within 0.5%; impulses opposite to 1e-9",
            },
        },
        "outcome_matrix_applied": {
            "GO_requires": ["P1-T1", "P1-T2", "P1-T4", "P1-T5", "P1-T6", "P1-T7", "P1-T9", "P1-T11",
                            "a non-degenerate P1-T12"],
            "note": "P1-T11 FAILED at the selected cell (chord std 25.4 deg vs 15 deg) and P1-T10'/P1-T12 "
                    "belong to cells (e)(f), which are not this task's. This module reports T1..T9 and the "
                    "Outcome-matrix consequence of each failure; the phase verdict is stated as the "
                    "conjunction of what is in scope.",
            "T5_fail_in_a_class": "Prop. 3' does not govern that class; its measured a_bar_ret becomes the "
                                  "operative form and its severance law EMP",
            "T6_fail": "the criterion is wrong and the mechanism is re-derived before Phase 2; NO-GO",
            "T7_fail_quadratic": "linear drag is not limiting the drift; Prop. 4' withdrawn; NO-GO",
            "T9_fail": "a sign or lever-arm error; NO-GO",
        },
        "closure_censoring": "Formation closure (B.7, any chord below 1 m) is a terminal event: it ends "
                             "the run and censors everything after it. A scripted cell whose drift reaches "
                             "the geometric limit therefore yields no re-engagement and no mark; such cells "
                             "contribute to no test population, are counted and listed, and their P1-T6 and "
                             "P1-T7 points are reported as censored with the reason. The reduced one-cable "
                             "ODE the predictions come from has no geometric limit, so the committed "
                             "prediction exists for cells the plant cannot reach.",
        "statistics": {
            "through_origin": "slope = sum(x y)/sum(x^2); uncentred R2 = 1 - sum(residual^2)/sum(y^2)",
            "medians": "numpy linear-interpolation medians",
            "no_intervals": "cells (a)-(d) are deterministic: one run per cell, no seeds, no bootstrap. "
                            "Every number is a point value and is reported as such.",
        },
        "workers": 8,
        "files": {
            "code": relative(MODULE_PATH),
            "tests": "tether/tests/test_v2_phase1_scripted.py",
            "declarations": relative(DECLARATIONS_PATH),
            "results": relative(RESULTS_PATH),
            "runs": relative(RUNS_PATH),
        },
    }


def declare() -> str:
    digest = write_bytes(DECLARATIONS_PATH, json_bytes(_declarations()))
    print("declarations", relative(DECLARATIONS_PATH), digest)
    return digest


def _require_declarations() -> str:
    if not DECLARATIONS_PATH.exists():
        raise RuntimeError(f"declare first: {relative(DECLARATIONS_PATH)} does not exist")
    return sha256_file(DECLARATIONS_PATH)


def _provenance() -> dict:
    return {
        "source": source_state(),
        "declarations": relative(DECLARATIONS_PATH),
        "declarations_sha256": _require_declarations(),
        "module_sha256": sha256_file(MODULE_PATH),
        "predictions_sha256": sha256_file(PREDICTIONS_PATH),
        "claim_a_declarations_sha256": sha256_file(CLAIM_A_PATH),
        "impact_table_sha256": sha256_file(IMPACT_TABLE_PATH),
        "events_module_sha256": sha256_file(ROOT / "tether" / "campaign" / "v2" / "events.py"),
        "regime_module_sha256": sha256_file(ROOT / "tether" / "campaign" / "v2" / "regime.py"),
        "summaries_module_sha256": sha256_file(ROOT / "tether" / "campaign" / "v2" / "summaries.py"),
        "drag_excursion_sha256": sha256_file(ROOT / "tether" / "theory" / "drag_excursion.py"),
    }


# ----------------------------------------------------------------------------- helpers


def through_origin(x, y) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or not np.dot(x, x) > 0.0:
        return {"n": int(x.size), "slope": None, "uncentred_r2": None}
    slope = float(np.dot(x, y) / np.dot(x, x))
    residual = y - slope * x
    return {"n": int(x.size), "slope": slope,
            "uncentred_r2": float(1.0 - np.dot(residual, residual) / np.dot(y, y))}


def _median(values) -> float | None:
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def analytic_class(lam: float, t_x: float, pretension: float) -> str:
    """The declared analytic (H4') class of a scripted square gust."""
    w_c = lam * pretension
    if w_c <= pretension:
        return "N"
    return rg.classify(t_x, w_c, pretension, True)


def gust_weather(pretension: float, lam: float, t_x: float) -> np.ndarray:
    """The declared forcing as the plant's weather array (rows, N+1, 2)."""
    rows = int(round(RUN_SECONDS / rg.WEATHER_PERIOD)) + 2
    weather = np.zeros((rows, constants.VESSEL_COUNT + 1, 2))
    start = int(round(GUST_START / rg.WEATHER_PERIOD))
    length = int(round(t_x / rg.WEATHER_PERIOD))
    weather[start:start + length, TEST_CABLE + 1, 0] = -VESSEL_FORCE_PER_WC * lam * pretension
    return weather


def aic(x: np.ndarray, y: np.ndarray, degree: int) -> float | None:
    n = x.size
    k = degree + 2  # coefficients plus the noise scale
    if n <= k:
        return None
    coefficients = np.polyfit(x, y, degree)
    residual = y - np.polyval(coefficients, x)
    rss = float(np.dot(residual, residual))
    if not rss > 0.0:
        rss = 1.0e-300
    return float(n * math.log(rss / n) + 2 * k)


# ----------------------------------------------------------------------------- cell (c)/(d) jobs


@dataclass(frozen=True)
class GustJob:
    cell: str
    pretension: float
    lam: float
    t_x: float
    cost: float = 1.0

    def spec(self) -> FleetRunSpec:
        return FleetRunSpec(
            formation="parallel",
            pretension=self.pretension,
            heading_gain=HEADING_GAINS[self.pretension],
            trim_gain=TRIM_GAIN,
            drag_law="linear",
            weather_scale=0.0,
            duration=RUN_SECONDS,
            warmup=0.0,
            cable_mode="recording",
            k_sigma=K_SIGMA,
            sway_limit=SWAY_LIMIT,
        )


def gust_jobs() -> list[GustJob]:
    jobs = [GustJob("c", t0, lam, t_x) for t0 in PRETENSIONS for lam in LAMBDAS for t_x in DURATIONS]
    jobs += [GustJob("d", D_PRETENSION, a / D_PRETENSION, D_DURATION) for a in D_AMPLITUDES]
    return jobs


def _main_mark_extras(run, summary, job: GustJob) -> dict:
    """The 1 ms-log observables the v1 mark record does not carry."""
    log = run.fleet.cables.log
    n = log.count
    time = log.event_time[:n]
    elongation = log.elongation[:n, TEST_CABLE]
    rate = log.rate[:n, TEST_CABLE]
    tension = np.where(elongation > 0.0,
                       np.maximum(run.fleet.cables.cable.stiffness * elongation
                                  + run.fleet.cables.cable.damping * rate, 0.0), 0.0)
    t_off = GUST_START + job.t_x
    out: dict = {"t_off": t_off, "min_elongation_test_cable": float(np.min(elongation))}

    # P1-T6 drift function, on the longest gust only (the plan's late window)
    if job.cell == "c" and math.isclose(job.t_x, max(DURATIONS)):
        late = (time >= t_off - LATE_WINDOW) & (time <= t_off)
        c_eff_f = 1.0 / (1.0 / rg.C_A + 1.0 / (rg.C_L + 4.0 * rg.C_A))
        out["late_window"] = {
            "samples": int(np.sum(late)),
            "mean_rate": float(np.mean(rate[late])),
            "mean_tension": float(np.mean(tension[late])),
            "slack_fraction": float(np.mean(elongation[late] <= 0.0)),
            "drift_function_over_T0": float(np.mean(tension[late] + c_eff_f * rate[late]) / job.pretension),
        }

    marks = summary["marks"]
    rows = np.flatnonzero((marks["cable"] == TEST_CABLE)
                          & (marks["t_up"] - marks["dwell"] >= GUST_START - 1.0e-9))
    out["test_cable_marks_after_gust_start"] = int(rows.size)
    closure = summary["closure"]
    if rows.size == 0:
        # no re-engagement: either the cable never emptied or the run ended slack (closure/censor)
        open_slack = bool(np.any(elongation[time >= GUST_START] <= 0.0))
        out["slack"] = open_slack
        out["censored_open_interval_at_end"] = bool(open_slack)
        out["closure_terminal"] = bool(closure["terminal"])
        return out

    position = int(rows[0])
    t_up = float(marks["t_up"][position])
    t_on = t_up - float(marks["dwell"][position])
    depth = float(marks["depth"][position])
    v_up = float(marks["v_return"][position])
    window = (time >= t_on - 1.0e-9) & (time <= t_up + 1.0e-9)
    indices = np.flatnonzero(window)
    deep = int(indices[int(np.argmin(elongation[indices]))])
    t_deep = float(time[deep])
    k_off = int(np.argmin(np.abs(time - t_off)))
    e_off = float(elongation[k_off])
    open_at_shutoff = bool(t_on <= t_off <= t_up and e_off < 0.0)
    delta_g_obs = -e_off if open_at_shutoff else 0.0
    return_leg = t_up - t_deep
    a_time = v_up / return_leg if return_leg > 0.0 else float("nan")
    a_depth = v_up * v_up / (2.0 * depth) if depth > 0.0 else float("nan")
    a_0 = job.pretension / M_FLEET
    driven = (max(0.0, min(t_up, t_off) - max(t_deep, GUST_START)) / return_leg
              if return_leg > 0.0 else float("nan"))
    peak_window = (time >= t_up) & (time <= t_up + ev.ENGAGEMENT_PERIOD)
    out.update({
        "slack": True,
        "mark_position": position,
        "t_onset_after_gust_start": t_on - GUST_START,
        "t_deep_after_gust_start": t_deep - GUST_START,
        "t_up_after_gust_start": t_up - GUST_START,
        "u_entry": float(marks["u_entry"][position]),
        "dwell": float(marks["dwell"][position]),
        "V_up": v_up,
        "max_depth": depth,
        "T_peak": float(marks["T_peak"][position]),
        "n_maxima": float(marks["n_maxima"][position]),
        "open_at_shutoff": open_at_shutoff,
        "returned_during_gust": bool(t_up < t_off),
        "Delta_g": delta_g_obs,
        "post_gust_deepening": (depth - delta_g_obs) if open_at_shutoff else None,
        "a_bar_ret_time_mean": a_time,
        "a_bar_ret_depth_mean": a_depth,
        "a_bar_ret_time_over_a0": a_time / a_0,
        "a_bar_ret_depth_over_a0": a_depth / a_0,
        "driven_fraction_of_return": driven,
        "inside_gust": bool(t_on >= GUST_START and t_up <= t_off),
        "plant_peak_tension_one_period": float(np.max(tension[peak_window])) if peak_window.any() else None,
        "regime": str(summary["marks_v2"]["regime"][position]),
        "regime_clip": str(summary["marks_v2"]["regime_clip"][position]),
        "t_x_measured": float(summary["marks_v2"]["t_x"][position]),
        "t_x_clip_measured": float(summary["marks_v2"]["t_x_clip"][position]),
        "Wc_max_measured": float(summary["marks_v2"]["Wc_max"][position]),
        "Wc_slack_max": float(summary["marks_v2"]["Wc_slack_max"][position]),
        "n_episodes": int(summary["marks_v2"]["n_episodes"][position]),
        "event_role": str(summary["marks_v2"]["event_role"][position]),
        "primary": bool(summary["marks_v2"]["primary"][position]),
        "closure_terminal": bool(closure["terminal"]),
    })
    return out


def _summarize(run, spec, weather) -> tuple[dict, str | None]:
    """``summarize_run_v2`` with a declared fallback for runs that end slack.

    ``campaign/v2/summaries.summarize_record`` evaluates ``ups.cable[...]`` eagerly inside a
    ``np.where``; when a run holds slack onsets but no re-engagement at all - which happens
    exactly when formation closure (B.7) terminates the run while the test cable is still
    empty - ``ups`` is empty and the indexing raises.  Such a run carries no marks, so the
    v2 mark, event and onset-parent blocks are empty by definition; this fallback builds
    them directly and records that it fired.  The shared module is not modified.
    """
    from tether.campaign import summaries as v1_summaries
    from tether.campaign.v2 import summaries as v2_summaries

    try:
        return summarize_run_v2(run, spec.warmup, spec.pretension, weather=weather), None
    except IndexError:
        record = v2_summaries.run_record(run)
        if record["all_marks"]["t_up"].size:
            raise
        summary = dict(v1_summaries.summarize_run(run, spec.warmup, spec.pretension))
        n_cables = record["elongation"].shape[1]
        end = record["sim_end"]
        closure = record["closure"]
        if closure is not None:
            end = min(end, float(closure[1]))
        interpolant = rg.StateInterpolant.from_log(record["state_time"], record["state"],
                                                   record["load_offsets"], record["vessel_offsets"])
        shape = v2_summaries._shape_statistics(record, interpolant, spec.warmup, end, n_cables)
        ups, downs = ev.crossings(record["event_time"], record["elongation"], record["alive"])
        planned_end = float(spec.warmup + spec.duration)
        if closure is None:
            closure_block = {"terminal": False, "time": None, "cable": None, "exposure_lost": 0.0,
                             "planned_end": planned_end, "in_window": None}
        else:
            closure_block = {"terminal": True, "time": float(closure[1]), "cable": int(closure[0]),
                             "exposure_lost": max(0.0, planned_end - max(float(closure[1]), spec.warmup)),
                             "planned_end": planned_end, "in_window": bool(float(closure[1]) >= spec.warmup)}
        empty = np.empty(0)
        summary.update({
            "marks_v2": {name: empty for name in rg.MARK_COVARIATES},
            "events": {"event_id": np.empty(0, dtype=np.int64)},
            "onsets_v2": {"count": int(downs.size), "primary_count": int(downs.size)},
            "clean": {},
            "closure": closure_block,
            "shape": shape,
            "regime_summary": {},
            "v2_diagnostics": {
                "weather_bit_equal_to_run": True,
                "wrel_offline_vs_log": {"passes_1e-6_relative": None},
                "marks_same_order_as_v1": True,
                "marks_unmatched_to_crossings": 0,
                "events_in_window_chained_rule_sensitivity": 0,
            },
        })
        return summary, ("summarize_run_v2 raised IndexError on a run with onsets but no "
                         "re-engagement (formation closure while slack); the declared fallback "
                         "supplied the empty v2 blocks")


def run_gust_job(job: GustJob) -> dict:
    weather = gust_weather(job.pretension, job.lam, job.t_x)
    spec = job.spec()
    run = run_to_end(build_run(spec, 1, weather=weather))
    summary, fallback = _summarize(run, spec, weather)
    marks = summary["marks"]
    marks_v2 = summary["marks_v2"]
    keep = marks["t_up"] - marks["dwell"] >= GUST_START - 1.0e-9
    collateral = keep & (marks["cable"] != TEST_CABLE)
    record = {
        "job": asdict(job),
        "W_c_N": job.lam * job.pretension,
        "force_on_vessel_N": VESSEL_FORCE_PER_WC * job.lam * job.pretension,
        "analytic_class": analytic_class(job.lam, job.t_x, job.pretension),
        "main": _main_mark_extras(run, summary, job),
        "shape": {
            "chord_world_std_max_deg": float(summary["shape"]["chord_world_std_max_deg"]),
            "chord_load_std_max_deg": float(summary["shape"]["chord_load_std_max_deg"]),
            "psi_std_max_deg": float(summary["shape"]["psi_std_max_deg"]),
            "load_yaw_std_deg": float(summary["shape"]["load_yaw_std_deg"]),
        },
        "closure": summary["closure"],
        "counts": {
            "marks_total": int(marks["t_up"].size),
            "marks_after_gust_start": int(np.sum(keep)),
            "collateral_marks": int(np.sum(collateral)),
            "events_anchored": int(summary["events"]["event_id"].size),
            "events_chained_in_window": int(
                summary["v2_diagnostics"]["events_in_window_chained_rule_sensitivity"]),
            "onsets": int(summary["onsets_v2"]["count"]),
            "primary_onsets": int(summary["onsets_v2"]["primary_count"]),
        },
        "collateral": {
            "cable": marks["cable"][collateral].astype(int).tolist(),
            "t_up": marks["t_up"][collateral].tolist(),
            "u_entry": marks["u_entry"][collateral].tolist(),
            "depth": marks["depth"][collateral].tolist(),
            "v_return": marks["v_return"][collateral].tolist(),
            "T_peak": marks["T_peak"][collateral].tolist(),
            "regime": [str(v) for v in marks_v2["regime"][collateral]],
        },
        "diagnostics": {
            "weather_bit_equal_to_run": summary["v2_diagnostics"]["weather_bit_equal_to_run"],
            "wrel_offline_passes_1e-6": summary["v2_diagnostics"]["wrel_offline_vs_log"]["passes_1e-6_relative"],
            "marks_same_order_as_v1": summary["v2_diagnostics"]["marks_same_order_as_v1"],
            "marks_unmatched_to_crossings": summary["v2_diagnostics"]["marks_unmatched_to_crossings"],
        },
        "summariser_fallback": fallback,
        "wall_seconds": float(run.wall_seconds),
        "end_time": float(run.simulator.get_context().get_time()),
    }
    return record


# ----------------------------------------------------------------------------- cell (a)


def _production_rigid(pretension: float) -> dict:
    from tether.physics.plant import run_transit_log

    geometry = formation_geometry("parallel")
    point = operating_point(geometry, pretension)
    thrust = constants.VESSEL_LINEAR_DRAG * point.speed + pretension
    spec = FleetRunSpec(
        formation="parallel", pretension=pretension, heading_gain=HEADING_GAINS[pretension],
        trim_gain=TRIM_GAIN, weather_scale=0.0, duration=RIGID_DURATION, warmup=0.0,
        k_sigma=K_SIGMA, sway_limit=SWAY_LIMIT,
    )
    run = run_to_end(build_run(spec, 1))
    log = run.fleet.cables.log
    n, m = log.count, log.state_count
    states = log.state[:m]
    base = 3 * (constants.VESSEL_COUNT + 1)
    start = run.fleet.extras["initial_state"]
    vessel_x = states[:, 3:base:3]
    distance = float(np.mean(vessel_x[-1] - start[3:base:3]))
    elongation = log.elongation[:n]
    rate = log.rate[:n]
    tension = np.where(elongation > 0.0,
                       np.maximum(run.fleet.cables.cable.stiffness * elongation
                                  + run.fleet.cables.cable.damping * rate, 0.0), 0.0)
    mean_tension = float(np.mean(tension))
    minimum_elongation = float(np.min(elongation))
    elapsed = float(log.state_time[m - 1] - 0.0)

    bilateral = run_transit_log(thrust, RIGID_DURATION, RIGID_SAMPLE)
    positions = bilateral.truth.shape[1] // 2
    reference_track = np.mean(bilateral.truth[:, 3:positions:3] - bilateral.truth[0, 3:positions:3], axis=1)
    # The production 10 ms state log ends one sample short of the planned end; the reference
    # distance is interpolated to the production log's own elapsed time (declared).
    reference_distance = float(np.interp(elapsed, bilateral.time, reference_track))
    reference_tension = float(np.mean(bilateral.tension))
    distance_error = abs(distance / reference_distance - 1.0)
    tension_error = abs(mean_tension / reference_tension - 1.0)
    passed = bool(distance_error <= T1_DISTANCE and tension_error <= T1_TENSION and minimum_elongation > 0.0)
    return {
        "pretension": pretension, "heading_gain": HEADING_GAINS[pretension], "thrust_N": thrust,
        "duration_s": RIGID_DURATION, "logged_elapsed_s": elapsed,
        "analytic_speed_mps": point.speed,
        "production_distance_m": distance, "phase0_distance_m": reference_distance,
        "distance_relative_error": distance_error,
        "production_mean_tension_N": mean_tension, "phase0_mean_tension_N": reference_tension,
        "tension_relative_error": tension_error,
        "minimum_elongation_m": minimum_elongation, "no_cable_slack": bool(minimum_elongation > 0.0),
        "passed": passed,
    }


def run_cell_a(workers: int = 8) -> dict:
    from tether.physics.phase1_deterministic import deterministic_fleet_acceptance

    v1 = deterministic_fleet_acceptance()
    v1_t1 = dict(v1["P1-T1"])
    production = run_pool(_production_rigid, list(PRETENSIONS), workers=min(workers, len(PRETENSIONS)))
    all_pass = bool(v1_t1["passed"] and all(item["passed"] for item in production))
    return {
        "reading_1_v1_harness": v1_t1,
        "reading_2_production_plant": production,
        "P1-T2b_from_the_same_harness": v1["P1-T2b"],
        "passed": all_pass,
        "verdict": "PASS" if all_pass else "FAIL",
    }


# ----------------------------------------------------------------------------- cell (b)


def _table_key(pretension: float, cable: int) -> str:
    return f"T{int(pretension)}_c{cable}"


def run_cell_b(workers: int = 8) -> dict:
    from tether.campaign.phase1_close import ImpactJob, run_impact_job
    from tether.physics.phase1_deterministic import deterministic_fleet_acceptance
    from tether.physics.phase1_mechanics import PINNED_IMPEDANCE, run_two_body_impact

    pinned = json.loads(IMPACT_TABLE_PATH.read_text())
    jobs = [ImpactJob(t0, cable, speed)
            for t0 in TABLE_PRETENSIONS for cable in TABLE_CABLES for speed in TABLE_SPEEDS]
    points = run_pool(run_impact_job, jobs, workers=workers)

    pairs = {}
    worst = 0.0
    worst_point = None
    for t0 in TABLE_PRETENSIONS:
        for cable in TABLE_CABLES:
            key = _table_key(t0, cable)
            entry = pinned[key]
            speeds = np.asarray(entry["speeds"], dtype=float)
            pinned_peaks = np.asarray(entry["peaks"], dtype=float)
            measured = np.array([p["peak"] for p in points
                                 if p["pretension"] == t0 and p["cable"] == cable])
            order = np.argsort([p["speed"] for p in points
                                if p["pretension"] == t0 and p["cable"] == cable])
            measured = measured[order]
            errors = np.abs(measured / pinned_peaks - 1.0)
            high = speeds >= 1.0
            fit = through_origin(speeds[high], measured[high])
            baseline = through_origin(speeds[high], measured[high] - measured[speeds == 0.0][0])
            if float(np.max(errors)) > worst:
                worst = float(np.max(errors))
                worst_point = {"pair": key, "speed": float(speeds[int(np.argmax(errors))])}
            pairs[key] = {
                "pretension": t0, "cable": cable, "speeds": speeds.tolist(),
                "measured_peaks_N": measured.tolist(), "pinned_peaks_N": pinned_peaks.tolist(),
                "max_relative_error": float(np.max(errors)),
                "within_1pct": bool(np.max(errors) <= T2_TABLE),
                "asymptotic_impedance_measured": fit["slope"],
                "asymptotic_impedance_pinned": entry["asymptotic_impedance"],
                "uncentred_r2_above_1mps": fit["uncentred_r2"],
                "r2_above_threshold": bool(fit["uncentred_r2"] is not None and fit["uncentred_r2"] > T2_R2),
                "baseline_subtracted_impedance": baseline["slope"],
                "zero_speed_peak_measured_N": float(measured[speeds == 0.0][0]),
                "zero_speed_peak_over_T0": float(measured[speeds == 0.0][0] / t0),
                "monotone": bool(np.all(np.diff(measured) > 0.0)),
            }

    two_body = np.array([run_two_body_impact(float(v), constants.CABLE_DAMPING, duration=0.25,
                                             integration_step=constants.TIME_STEP).first_peak_tension
                         for v in TWO_BODY_SPEEDS])
    z_fit = through_origin(np.asarray(TWO_BODY_SPEEDS, dtype=float), two_body)
    fleet = deterministic_fleet_acceptance()["P1-T2b"]
    z_value = float(z_fit["slope"])
    z_fleet = float(fleet["impedance_n_s_per_m"])

    table_pass = all(item["within_1pct"] for item in pairs.values())
    r2_pass = all(item["r2_above_threshold"] for item in pairs.values())
    t2_pass = bool(table_pass and r2_pass)

    stiffness = constants.CABLE_STIFFNESS
    m_eff_hat = z_value**2 / (IMPACT_FACTOR**2 * stiffness)
    m_fleet_hat = z_fleet**2 / (IMPACT_FACTOR**2 * stiffness)
    m_eff_inertial = constants.VESSEL_MASS * constants.LOAD_MASS / (constants.VESSEL_MASS + constants.LOAD_MASS)
    m_fleet_inertial = 1.0 / (1.0 / constants.VESSEL_MASS
                              + 1.0 / (constants.LOAD_MASS + 4.0 * constants.VESSEL_MASS))
    m_eff_error = abs(m_eff_hat / m_eff_inertial - 1.0)
    m_fleet_error = abs(m_fleet_hat / m_fleet_inertial - 1.0)
    t3_pass = bool(m_eff_error <= T3_MASS and m_fleet_error <= T3_MASS)

    return {
        "pairs": pairs,
        "points": len(points),
        "max_relative_error": worst,
        "worst_point": worst_point,
        "table_within_1pct": table_pass,
        "all_pairs_r2_above_0.99": r2_pass,
        "Z_repin": {
            "speeds": list(TWO_BODY_SPEEDS), "first_peaks_N": two_body.tolist(),
            "Z_N_s_per_m": z_value, "uncentred_r2": z_fit["uncentred_r2"],
            "v1_pin": PINNED_IMPEDANCE, "relative_difference_from_v1_pin": abs(z_value / PINNED_IMPEDANCE - 1.0),
            "plan_value": 7993.5,
        },
        "Z_fleet_repin": {
            "speeds": fleet["speeds_mps"], "first_peaks_N": fleet["first_peaks_n"],
            "Z_fleet_N_s_per_m": z_fleet, "uncentred_r2": fleet["uncentred_r_squared"],
            "v1_pin": fleet["pinned_impedance_n_s_per_m"],
            "relative_difference_from_v1_pin": fleet["pin_relative_error"],
            "plan_value": 8304.5,
            "baseline_subtracted_impedance": fleet["baseline_subtracted_impedance_n_s_per_m"],
        },
        "P1-T2": {"passed": t2_pass, "verdict": "PASS" if t2_pass else "FAIL"},
        "P1-T3": {
            "m_eff_from_Z_kg": m_eff_hat, "m_eff_inertial_kg": m_eff_inertial,
            "m_eff_relative_error": m_eff_error,
            "m_fleet_from_Z_fleet_kg": m_fleet_hat, "m_fleet_inertial_kg": m_fleet_inertial,
            "m_fleet_relative_error": m_fleet_error,
            "m_fleet_plan_pivot_kg": M_FLEET,
            "passed": t3_pass, "verdict": "PASS" if t3_pass else "FAIL",
        },
    }


# ----------------------------------------------------------------------------- P1-T9


def run_t9() -> dict:
    from tether.physics.phase1_mechanics import run_two_body_impact

    rows = []
    for speed in T9_SPEEDS:
        trajectory = run_two_body_impact(float(speed), 0.0, duration=T9_DURATION,
                                         integration_step=T9_STEP)
        energy = trajectory.mechanical_energy
        drift = float(np.max(np.abs(energy - energy[0])) / energy[0])
        force_sum = trajectory.load_force + trajectory.vessel_force
        impulse = float(abs(np.sum(force_sum) * T9_STEP))
        rows.append({
            "speed_mps": float(speed), "step_s": T9_STEP, "duration_s": T9_DURATION,
            "initial_energy_J": float(energy[0]),
            "max_relative_energy_drift": drift,
            "max_abs_force_sum_N": float(np.max(np.abs(force_sum))),
            "abs_impulse_sum_Ns": impulse,
            "first_peak_tension_N": float(trajectory.first_peak_tension),
            "energy_pass": bool(drift <= T9_ENERGY_LIMIT),
            "impulse_pass": bool(np.max(np.abs(force_sum)) <= T9_IMPULSE_LIMIT and impulse <= T9_IMPULSE_LIMIT),
        })
    passed = all(row["energy_pass"] and row["impulse_pass"] for row in rows)
    return {
        "cell": "force-free, drag-free, gravity-free two-body cable cell, c = 0, 0.5 ms fixed-step RK3",
        "rows": rows,
        "max_energy_drift": max(row["max_relative_energy_drift"] for row in rows),
        "max_abs_force_sum_N": max(row["max_abs_force_sum_N"] for row in rows),
        "zero_damping_impedance_check": {
            "sqrt_k_m_eff": math.sqrt(constants.CABLE_STIFFNESS * M_EFF),
            "measured_peak_over_speed": [row["first_peak_tension_N"] / row["speed_mps"] for row in rows],
        },
        "passed": passed, "verdict": "PASS" if passed else "FAIL",
    }


# ----------------------------------------------------------------------------- scoring


def _predictions() -> dict:
    return json.loads(PREDICTIONS_PATH.read_text())


def _main_rows(records: list[dict], cell: str) -> list[dict]:
    return [r for r in records if r["job"]["cell"] == cell]


def score_t4(records: list[dict], predictions: dict) -> dict:
    marks = []
    for record in _main_rows(records, "c"):
        main = record["main"]
        if not main.get("slack") or "V_up" not in main:
            continue
        if main["regime"] != "N":
            continue
        job = record["job"]
        u = main["u_entry"]
        lam = job["lam"]
        a_plan = (1.0 - lam) * job["pretension"] / M_FLEET
        ballistic = u * u / (2.0 * a_plan) if a_plan > 0.0 else None
        marks.append({
            "T0": job["pretension"], "lambda": lam, "t_x": job["t_x"],
            "u": u, "V_up": main["V_up"], "depth": main["max_depth"],
            "inside_gust": main["inside_gust"],
            "symmetry_error": abs(main["V_up"] - u) / u if u > 0.0 else None,
            "depth_over_plan_ballistic": main["max_depth"] / ballistic if ballistic else None,
            "a_plan": a_plan,
            "symmetry_pass": bool(u > 0.0 and abs(main["V_up"] - u) / u < T4_SYMMETRY),
            "depth_pass": bool(ballistic and abs(main["max_depth"] / ballistic - 1.0) <= T4_DEPTH),
        })
    collateral = []
    for record in records:
        job = record["job"]
        a_plan = job["pretension"] / M_FLEET
        for index, regime in enumerate(record["collateral"]["regime"]):
            if regime != "N":
                continue
            u = record["collateral"]["u_entry"][index]
            v_up = record["collateral"]["v_return"][index]
            depth = record["collateral"]["depth"][index]
            if not u > 0.0 or not depth > 0.0:
                continue
            collateral.append({
                "T0": job["pretension"], "lambda": job["lam"], "t_x": job["t_x"],
                "cable": record["collateral"]["cable"][index],
                "symmetry_error": abs(v_up - u) / u,
                "depth_over_ballistic": depth / (u * u / (2.0 * a_plan)),
            })
    passed = bool(marks) and all(m["symmetry_pass"] and m["depth_pass"] for m in marks)
    predicted = predictions["P1-T4"]["summary"]
    return {
        "marks": marks, "n": len(marks),
        "symmetry_pass_fraction": float(np.mean([m["symmetry_pass"] for m in marks])) if marks else None,
        "depth_pass_fraction": float(np.mean([m["depth_pass"] for m in marks])) if marks else None,
        "median_symmetry_error": _median([m["symmetry_error"] for m in marks]),
        "median_symmetry_error_inside_gust": _median([m["symmetry_error"] for m in marks if m["inside_gust"]]),
        "collateral_secondary": {
            "n": len(collateral),
            "median_symmetry_error": _median([m["symmetry_error"] for m in collateral]),
            "median_depth_over_ballistic": _median([m["depth_over_ballistic"] for m in collateral]),
            "marks": collateral,
        },
        "committed_prediction": predicted,
        "passed": passed, "verdict": "PASS" if passed else "FAIL",
    }


def _t5_population(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        main = record["main"]
        if not main.get("slack") or "V_up" not in main or not main["max_depth"] > 0.0:
            continue
        rows.append({"regime": main["regime"], "V_up": main["V_up"], "depth": main["max_depth"],
                     "a_time": main["a_bar_ret_time_mean"], "a_depth": main["a_bar_ret_depth_mean"],
                     "driven": main["driven_fraction_of_return"],
                     "T0": record["job"]["pretension"], "lambda": record["job"]["lam"],
                     "t_x": record["job"]["t_x"]})
    return rows


def score_t5(records: list[dict], predictions: dict) -> dict:
    rows = _t5_population(records)
    out = {}
    for name in CLASSES + ("all",):
        members = [r for r in rows if name == "all" or r["regime"] == name]
        y = np.array([r["V_up"] for r in members])
        x_time = np.array([math.sqrt(2.0 * r["a_time"] * r["depth"]) for r in members])
        x_depth = np.array([math.sqrt(2.0 * r["a_depth"] * r["depth"]) for r in members])
        time_fit = through_origin(x_time, y)
        depth_fit = through_origin(x_depth, y)
        powered = len(members) >= T5_MIN_MARKS
        def verdict(fit):
            if not powered:
                return None
            if fit["slope"] is None:
                return None
            return bool(abs(fit["slope"] - 1.0) <= T5_SLOPE and fit["uncentred_r2"] > T5_R2)
        out[name] = {
            "n": len(members), "powered": powered,
            "depth_mean_reading": {**depth_fit, "passed": verdict(depth_fit),
                                   "note": "identity: 2 a_bar_ret Delta = V_up^2 by construction"},
            "time_mean_reading": {**time_fit, "passed": verdict(time_fit)},
            "committed_prediction": predictions["P1-T5"].get(name),
        }
    powered_classes = [name for name in CLASSES if out[name]["powered"]]
    primary_pass = all(out[name]["depth_mean_reading"]["passed"] for name in powered_classes) if powered_classes else None
    secondary_fail = [name for name in powered_classes if out[name]["time_mean_reading"]["passed"] is False]
    return {
        "per_class": out,
        "powered_classes": powered_classes,
        "primary_reading": "depth-mean (the plan's literal a_bar_ret); the regression is an identity, so a "
                           "pass here is vacuous and is reported as vacuous",
        "classes_failing_the_time_mean_reading": secondary_fail,
        "passed": primary_pass,
        "verdict": "PASS (vacuous)" if primary_pass else ("FAIL" if primary_pass is False else "UNDER-POWERED"),
    }


def score_t6(records: list[dict], predictions: dict) -> dict:
    per_t0 = {}
    for pretension in PRETENSIONS:
        members = sorted((r for r in _main_rows(records, "c")
                          if r["job"]["pretension"] == pretension and "late_window" in r["main"]),
                         key=lambda r: r["job"]["lam"])
        # runs terminated by formation closure before the shut-off hold no late window and are
        # censored out of the fit (declared)
        usable = [r for r in members if r["main"]["late_window"]["samples"] > 0]
        censored = [r["job"]["lam"] for r in members if r["main"]["late_window"]["samples"] == 0]
        lam = np.array([r["job"]["lam"] for r in usable])
        y = np.array([r["main"]["late_window"]["drift_function_over_T0"] for r in usable])
        fit = np.polyfit(lam, y, 1) if lam.size >= 2 else None
        crossing = float(-fit[1] / fit[0]) if fit is not None else None
        per_t0[str(int(pretension))] = {
            "lambda": lam.tolist(), "drift_function_over_T0": y.tolist(),
            "lambda_censored_by_closure": censored,
            "fitted_crossing": crossing,
            "within_5pct_of_1": bool(crossing is not None and abs(crossing - 1.0) <= T6_CROSSING),
            "slack_fraction_late_window": [r["main"]["late_window"]["slack_fraction"] for r in usable],
        }
    crossings = [entry["fitted_crossing"] for entry in per_t0.values() if entry["fitted_crossing"] is not None]
    passed = bool(crossings) and all(abs(c - 1.0) <= T6_CROSSING for c in crossings)
    return {
        "primary_drift_form": per_t0,
        "pooled_crossing": float(np.mean(crossings)) if crossings else None,
        "committed_prediction": predictions["P1-T6"],
        "secondary_onset_acceleration_form": {
            "committed": {k: v for k, v in predictions["P1-T6"]["analytic"].items()
                          if "onset_acceleration" in k},
            "note": "the onset-acceleration form is committed at 1.055-1.077 and gives the opposite "
                    "verdict; the drift form is the scored one",
        },
        "passed": passed, "verdict": "PASS" if passed else "FAIL",
    }


def score_t7(records: list[dict], predictions: dict) -> dict:
    cells = {}
    for pretension in PRETENSIONS:
        for lam in LAMBDAS:
            if lam <= 1.0:
                continue
            members = sorted((r for r in _main_rows(records, "c")
                              if r["job"]["pretension"] == pretension and r["job"]["lam"] == lam),
                             key=lambda r: r["job"]["t_x"])
            v_d = float(dx.drift_speed(lam * pretension, pretension))
            rows = []
            for record in members:
                job, main = record["job"], record["main"]
                t_x = job["t_x"]
                closed_depth = float(dx.shutoff_depth(v_d, t_x))
                v_s = float(dx.shutoff_speed(v_d, t_x))
                closed_deepening = float(dx.post_gust_deepening(v_s, pretension))
                censored = not main.get("open_at_shutoff", False)
                row = {
                    "t_x": t_x, "v_d": v_d,
                    "Delta_g_closed": closed_depth, "post_gust_deepening_closed": closed_deepening,
                    "Delta_g_plant": main.get("Delta_g"),
                    "post_gust_deepening_plant": main.get("post_gust_deepening"),
                    "max_depth_plant": main.get("max_depth"),
                    "censored": bool(censored),
                    "censor_reason": (None if not censored else
                                      ("closure" if main.get("closure_terminal") else
                                       ("no slack" if not main.get("slack") else
                                        ("returned during the gust" if main.get("returned_during_gust")
                                         else "slack interval not open at shut-off")))),
                    "scored": bool(not censored and closed_depth > T7_DEPTH_FLOOR),
                }
                if row["scored"]:
                    row["depth_ratio"] = main["Delta_g"] / closed_depth
                    row["deepening_ratio"] = (main["post_gust_deepening"] / closed_deepening
                                              if closed_deepening > 0.0 else None)
                    row["depth_within_15pct"] = bool(abs(row["depth_ratio"] - 1.0) <= T7_RATIO)
                    row["deepening_within_15pct"] = bool(row["deepening_ratio"] is not None
                                                         and abs(row["deepening_ratio"] - 1.0) <= T7_RATIO)
                rows.append(row)
            scored = [r for r in rows if r["scored"]]
            linear_points = [r for r in rows if not r["censored"] and r["t_x"] >= dx.TAU_A]
            x = np.array([r["t_x"] for r in linear_points])
            y = np.array([r["Delta_g_plant"] for r in linear_points])
            aic_linear = aic(x, y, 1) if x.size else None
            aic_quadratic = aic(x, y, 2) if x.size else None
            slope = float(np.polyfit(x, y, 1)[0]) if x.size >= 2 else None
            cells[f"{int(pretension)}@{lam}"] = {
                "rows": rows,
                "scored_points": len(scored),
                "censored_points": sum(1 for r in rows if r["censored"]),
                "depth_within_15pct": f"{sum(1 for r in scored if r['depth_within_15pct'])}/{len(scored)}",
                "deepening_within_15pct": f"{sum(1 for r in scored if r['deepening_within_15pct'])}/{len(scored)}",
                "aic_linear": aic_linear, "aic_quadratic": aic_quadratic,
                "quadratic_preferred": (bool(aic_quadratic < aic_linear)
                                        if (aic_linear is not None and aic_quadratic is not None) else None),
                "plant_depth_slope_over_t_x_ge_tau_A": slope,
                "slope_over_v_d": slope / v_d if slope is not None else None,
                "linear_fit_points": len(linear_points),
                "passed": (all(r["depth_within_15pct"] and r["deepening_within_15pct"] for r in scored)
                           if scored else None),
                "committed_prediction": predictions["P1-T7"].get(f"{int(pretension)}@{lam}"),
            }
    powered = [key for key, cell in cells.items() if cell["passed"] is not None]
    passed = all(cells[key]["passed"] for key in powered) if powered else None
    quadratic = [key for key in cells if cells[key]["quadratic_preferred"]]
    return {
        "cells": cells,
        "powered_cells": powered,
        "cells_with_a_preferred_quadratic": quadratic,
        "quadratic_anywhere": bool(quadratic),
        "passed": passed,
        "verdict": "PASS" if passed else ("FAIL" if passed is False else "UNDER-POWERED"),
        "outcome": ("Prop. 4' withdrawn, NO-GO (the failure is with a quadratic depth)" if quadratic
                    else "a quantitative miss of the closed form with the linear depth still preferred; "
                         "Prop. 4''s linearity in t_x is not refuted"),
    }


def score_t8(records: list[dict], predictions: dict) -> dict:
    rows = _t5_population(records)
    out = {}
    for name in CLASSES + ("all",):
        members = [r for r in rows if name == "all" or r["regime"] == name]
        bands = {}
        for low, high in DEPTH_BANDS:
            band = [r for r in members if low <= r["depth"] < high]
            a_0 = [r["T0"] / M_FLEET for r in band]
            bands[f"{low}-{high}"] = {
                "n": len(band),
                "median_a_time_over_a0": _median([r["a_time"] / a for r, a in zip(band, a_0)]),
                "median_a_depth_over_a0": _median([r["a_depth"] / a for r, a in zip(band, a_0)]),
                "median_driven_fraction": _median([r["driven"] for r in band]),
                "committed_prediction": (predictions["P1-T8"].get(name, {}) or {}).get(f"{low}-{high}"),
            }
        out[name] = bands
    return {"per_class": out, "a_0_definition": "a_0 = T0/m_fleet, m_fleet = 523.9 kg",
            "note": "reported covariate; P1-T8 carries no pass/fail"}


def score_d(records: list[dict], predictions: dict) -> dict:
    rows = []
    for record in _main_rows(records, "d"):
        amplitude = record["W_c_N"]
        committed = next(item for item in predictions["cells_d"] if abs(item["A_N"] - amplitude) < 1.0)
        main = record["main"]
        measured = main.get("T_peak")
        rows.append({
            "A_N": amplitude,
            "class_H4prime_machinery": main.get("regime"),
            "class_H4prime_analytic": record["analytic_class"],
            "committed_class": committed["class_H4prime"],
            "plant_T_peak_N": measured,
            "plant_V_up_mps": main.get("V_up"),
            "plant_max_depth_m": main.get("max_depth"),
            "plant_Delta_g_m": main.get("Delta_g"),
            "plant_u_entry_mps": main.get("u_entry"),
            "committed_full_ODE_T_peak_N": committed["committed_plant_prediction_T_peak_N"],
            "committed_three_piece_map_T_peak_N": committed["committed_map"]["T_peak_table"],
            "plan_appendix_T_peak_N": committed["plan_T_peak_N"],
            "plant_over_committed_full_ODE": (measured / committed["committed_plant_prediction_T_peak_N"]
                                              if measured else None),
            "plant_over_committed_map": (measured / committed["committed_map"]["T_peak_table"]
                                         if measured else None),
            "plant_over_plan_appendix": (measured / committed["plan_T_peak_N"] if measured else None),
            "committed_ODE_V_up": committed["ode"]["V_up"],
            "committed_ODE_max_depth": committed["ode"]["max_depth"],
            "cor5_free_body_T_peak_N": committed["cor5_free_body"]["T_peak_table"],
        })
    return {"rows": rows,
            "note": "the committed prediction (full ODE) and the plan's Appendix disagree; both are "
                    "reported and the committed one is the scored comparison"}


def score_population(records: list[dict]) -> dict:
    """(H4') class agreement between the machinery and the declared analytic class, and shape."""
    agree = 0
    disagree = []
    classes = {name: 0 for name in CLASSES}
    for record in _main_rows(records, "c") + _main_rows(records, "d"):
        main = record["main"]
        if not main.get("slack") or "regime" not in main:
            continue
        classes[main["regime"]] = classes.get(main["regime"], 0) + 1
        if main["regime"] == record["analytic_class"]:
            agree += 1
        else:
            disagree.append({"T0": record["job"]["pretension"], "lambda": record["job"]["lam"],
                             "t_x": record["job"]["t_x"], "machinery": main["regime"],
                             "analytic": record["analytic_class"],
                             "t_x_measured": main.get("t_x_measured"),
                             "Wc_max_measured": main.get("Wc_max_measured")})
    clip = {name: 0 for name in CLASSES}
    for record in _main_rows(records, "c") + _main_rows(records, "d"):
        main = record["main"]
        if main.get("slack") and "regime_clip" in main:
            clip[main["regime_clip"]] = clip.get(main["regime_clip"], 0) + 1
    shapes = [r["shape"]["chord_world_std_max_deg"] for r in records]
    psis = [r["shape"]["psi_std_max_deg"] for r in records]
    closures = [r for r in records if r["closure"]["terminal"]]
    return {
        "main_mark_class_counts_literal_full_episode": classes,
        "main_mark_class_counts_secondary_clipped_episode": clip,
        "machinery_vs_declared_analytic_class": {"agree": agree, "disagree": len(disagree),
                                                 "disagreements": disagree},
        "shape_H1": {
            "max_chord_world_std_deg": float(np.max(shapes)), "median_chord_world_std_deg": _median(shapes),
            "max_psi_std_deg": float(np.max(psis)),
            "note": "scripted cells carry no stochastic weather, so these are not the (H1) shape "
                    "measurement of P1-T11 (that is cell (e)/(f)); they show the formation holds shape "
                    "under the scripted gust alone",
        },
        "closures": {
            "n": len(closures),
            "cells": [{"T0": r["job"]["pretension"], "lambda": r["job"]["lam"], "t_x": r["job"]["t_x"],
                       "time": r["closure"]["time"], "cable": r["closure"]["cable"],
                       "exposure_lost": r["closure"]["exposure_lost"]} for r in closures],
        },
        "collateral_marks_total": int(sum(r["counts"]["collateral_marks"] for r in records)),
        "events": {
            "anchored_total": int(sum(r["counts"]["events_anchored"] for r in records)),
            "chained_total_secondary": int(sum(r["counts"]["events_chained_in_window"] for r in records)),
        },
        "machinery_defect_found": {
            "runs_using_the_declared_fallback": [
                {"T0": r["job"]["pretension"], "lambda": r["job"]["lam"], "t_x": r["job"]["t_x"]}
                for r in records if r.get("summariser_fallback")],
            "note": "campaign/v2/summaries.summarize_record raises IndexError when a run holds slack "
                    "onsets but no re-engagement at all (ups empty, indexed eagerly inside np.where at "
                    "the onset-parent line). This is exactly the formation-closure-while-slack case of "
                    "B.7. The shared module was not modified; the declared fallback in phase1_scripted "
                    "supplies the empty v2 blocks. Reported for the owner.",
        },
        "diagnostics": {
            "all_weather_bit_equal": all(r["diagnostics"]["weather_bit_equal_to_run"] for r in records),
            "all_wrel_offline_pass": all(r["diagnostics"]["wrel_offline_passes_1e-6"] for r in records),
            "all_marks_same_order_as_v1": all(r["diagnostics"]["marks_same_order_as_v1"] for r in records),
            "marks_unmatched_to_crossings": int(sum(r["diagnostics"]["marks_unmatched_to_crossings"]
                                                    for r in records)),
        },
    }


# ----------------------------------------------------------------------------- driver


def run(workers: int = 8) -> dict:
    provenance = _provenance()
    predictions = _predictions()

    cell_a = run_cell_a(workers)
    cell_b = run_cell_b(workers)
    t9 = run_t9()

    jobs = gust_jobs()
    records = run_pool(run_gust_job, jobs, workers=workers,
                       progress=lambda done, total: print(f"  gust {done}/{total}", flush=True))
    write_bytes(RUNS_PATH, json_bytes({"schema_version": 1, **provenance, "runs": records}))

    t4 = score_t4(records, predictions)
    t5 = score_t5(records, predictions)
    t6 = score_t6(records, predictions)
    t7 = score_t7(records, predictions)
    t8 = score_t8(records, predictions)
    cell_d = score_d(records, predictions)
    population = score_population(records)

    tests = {
        "P1-T1": {"verdict": cell_a["verdict"], "passed": cell_a["passed"]},
        "P1-T2": {"verdict": cell_b["P1-T2"]["verdict"], "passed": cell_b["P1-T2"]["passed"]},
        "P1-T3": {"verdict": cell_b["P1-T3"]["verdict"], "passed": cell_b["P1-T3"]["passed"]},
        "P1-T4": {"verdict": t4["verdict"], "passed": t4["passed"]},
        "P1-T5": {"verdict": t5["verdict"], "passed": t5["passed"]},
        "P1-T6": {"verdict": t6["verdict"], "passed": t6["passed"]},
        "P1-T7": {"verdict": t7["verdict"], "passed": t7["passed"]},
        "P1-T8": {"verdict": "REPORTED", "passed": None},
        "P1-T9": {"verdict": t9["verdict"], "passed": t9["passed"]},
    }
    blocking = ("P1-T1", "P1-T2", "P1-T4", "P1-T5", "P1-T6", "P1-T7", "P1-T9")
    failing = [name for name in blocking if tests[name]["passed"] is False]
    results = {
        "schema_version": 1, **provenance,
        "what": "Phase 1 deterministic cells (a)-(d) on the production plant; P1-T1..T9 scored against "
                "the committed predictions",
        "workers": workers,
        "cell_a_rigid_limit": cell_a,
        "cell_b_impact": cell_b,
        "cell_c_population_and_diagnostics": population,
        "cell_d_impulses": cell_d,
        "P1-T4": t4, "P1-T5": t5, "P1-T6": t6, "P1-T7": t7, "P1-T8": t8, "P1-T9": t9,
        "tests": tests,
        "blocking_in_scope": list(blocking),
        "failing_blocking_tests": failing,
        "phase1_deterministic_verdict": "GO (in scope)" if not failing else "NO-GO",
        "outcome_matrix_consequences": _consequences(tests, t5, t7),
        "wall_seconds_total": float(sum(r["wall_seconds"] for r in records)),
    }
    digest = write_bytes(RESULTS_PATH, json_bytes(results))
    print("results", relative(RESULTS_PATH), digest)
    return results


def _consequences(tests: dict, t5: dict, t7: dict) -> list[str]:
    out = []
    if tests["P1-T1"]["passed"] is False:
        out.append("P1-T1 FAILED: the unilateral plant does not reproduce Phase 0's rigid transit; "
                   "GO is impossible and every later radial result is on a plant that fails its own "
                   "steady-state identity.")
    if tests["P1-T2"]["passed"] is False:
        out.append("P1-T2 FAILED: the impact law v_b = Z^-1(T_b) is not reproduced, so every severance "
                   "level in the plan (Cor. 5', Thm 6', A.1, A.3) is computed from a table the plant "
                   "does not obey; GO is impossible.")
    if tests["P1-T3"]["passed"] is False:
        out.append("P1-T3 FAILED: the impedances do not return their inertial masses; m_eff and m_fleet "
                   "are reported as measured and the three-mass bookkeeping of II.4/Cor. 5' is EMP.")
    if tests["P1-T4"]["passed"] is False:
        out.append("P1-T4 FAILED: ballistic symmetry (Prop. 2) does not hold at the scored marks, so the "
                   "R1 no-gust leg of Claim A is not the plant's; GO requires T4, so the phase is NO-GO "
                   "on this clause.")
    if tests["P1-T5"]["passed"] is False:
        out.append("P1-T5 FAILED in the classes " + ", ".join(t5["classes_failing_the_time_mean_reading"])
                   + ": Prop. 3' does not govern them, their measured a_bar_ret becomes the operative "
                     "form and their severance law is EMP; Claim A rests on the classes that pass.")
    elif t5["classes_failing_the_time_mean_reading"]:
        out.append("P1-T5 passes only on the plan's literal (depth-mean) a_bar_ret, where the regression "
                   "is an identity. On the informative time-mean reading the classes "
                   + ", ".join(t5["classes_failing_the_time_mean_reading"])
                   + " miss the 5% slope band, which is the Outcome-matrix condition for their severance "
                     "law to be EMP; the owner must rule which reading the matrix means.")
    if tests["P1-T6"]["passed"] is False:
        out.append("P1-T6 FAILED: the drift criterion is wrong and the mechanism is re-derived before "
                   "Phase 2; NO-GO.")
    if tests["P1-T7"]["passed"] is False:
        out.append("P1-T7 FAILED. " + t7["outcome"])
    if tests["P1-T9"]["passed"] is False:
        out.append("P1-T9 FAILED: a sign or lever-arm error in the plant; NO-GO.")
    if not out:
        out.append("Every blocking test in scope passed.")
    return out


ADDENDUM_1_PATH = PHASE1_DIR / "scripted_addendum_1.json"


def addendum_1() -> dict:
    """Dated addendum, 2026-09-13: the declared AIC clause of P1-T7 is not evaluable.

    Written after the results and beside them; ``scripted_results.json`` is not edited.
    """
    results = json.loads(RESULTS_PATH.read_text())
    cells = results["P1-T7"]["cells"]
    rows = []
    for key, cell in cells.items():
        by_t_x = {row["t_x"]: row for row in cell["rows"]}
        long = [t for t in (3.4, 6.0, 10.0) if t in by_t_x and by_t_x[t].get("scored")]
        ratios = [by_t_x[t]["depth_ratio"] for t in long]
        rows.append({
            "cell": key, "t_x_above_tau_A_scored": long,
            "depth_ratio_plant_over_closed_form": ratios,
            "ratio_range": (max(ratios) - min(ratios)) if ratios else None,
            "aic_linear": cell["aic_linear"], "aic_quadratic": cell["aic_quadratic"],
            "linear_fit_points": cell["linear_fit_points"],
            "plant_depth_slope_over_v_d": cell["slope_over_v_d"],
        })
    payload = {
        "schema_version": 1,
        "date": "2026-09-13",
        "addendum_to": relative(RESULTS_PATH),
        "results_sha256": sha256_file(RESULTS_PATH),
        "declarations_sha256": sha256_file(DECLARATIONS_PATH),
        "source": source_state(),
        "module_sha256_recorded_in_the_results": results["module_sha256"],
        "module_sha256_now": sha256_file(MODULE_PATH),
        "module_change_since_the_results": "this ``addendum_1`` function and its --addendum flag were "
                                           "appended after the campaign ran; no declaration, cell, "
                                           "observable, threshold or scoring function was touched, and "
                                           "scripted_results.json is unmodified (its sha256 is above)",
        "reason": "The declared AIC rule (scripted_declarations.json, tests.P1-T7.AIC) uses "
                  "AIC = n ln(RSS/n) + 2 k with k = 3 (linear) and k = 4 (quadratic) and returns no "
                  "value when n <= k. The plan's grid puts only three durations above tau_A = 1.714 s "
                  "(t_x = 3.4, 6, 10 s), so n = 3 at every cell and NEITHER AIC is defined: the "
                  "'a quadratic fit not preferred by AIC' clause of P1-T7 is NOT EVALUABLE on the "
                  "plan's own cell (c) grid. This is a property of the grid, not of the plant, and it "
                  "was fixed by the declaration before the runs; nothing is re-scored here.",
        "what_is_reported_instead": "Descriptive evidence on the same declared observables, not a test: "
                                    "the closed form of Prop. 4' is linear in t_x above tau_A, so a "
                                    "quadratic depth would make the plant/closed-form depth ratio grow "
                                    "with t_x. It does not - the ratio is flat or mildly decreasing in "
                                    "every cell - and the plant's fitted depth slope over t_x >= tau_A "
                                    "divided by v_d is 0.94-1.21. The depth grows linearly; P1-T7's "
                                    "failure is a level miss of the closed form at small excess, not a "
                                    "quadratic depth.",
        "outcome_matrix_reading": "The Outcome matrix withdraws Prop. 4' and returns NO-GO only for "
                                  "'T7 failing WITH a quadratic depth'. That trigger does not fire here. "
                                  "P1-T7 still fails its 15% clause and GO requires T7, so the phase is "
                                  "NO-GO on the GO list without withdrawing Prop. 4'.",
        "cells": rows,
        "recommendation_for_the_owner": "If the AIC clause is to be decidable, cell (c) needs at least "
                                        "four durations above tau_A (e.g. adding t_x = 2.5 and 8 s), or "
                                        "the clause must be declared on a different discriminant.",
    }
    digest = write_bytes(ADDENDUM_1_PATH, json_bytes(payload))
    print("addendum", relative(ADDENDUM_1_PATH), digest)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--addendum", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.declare:
        declare()
    if arguments.run:
        run(min(arguments.workers, 8))
    if arguments.addendum:
        addendum_1()


if __name__ == "__main__":
    main()
