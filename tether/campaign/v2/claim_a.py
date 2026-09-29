"""Claim A's predictions, committed before any Phase 1 plant run (plan v2 II.3-II.6, Part III,
Phase 1 cells (c)(d), P0-W2, Appendix A), under the owner rulings of 2026-09-13
(k_sigma = 3 saturated at +-20 deg; Phase 2's graded grid at intensity 0.35).

Stages (every operational choice is in ``DECLARATIONS``, written to
``records/v2/phase1/claim_a_declarations.json`` by ``declare`` before any result; every
later stage refuses to run unless that file equals ``DECLARATIONS`` byte for byte and
records its sha256):

``phase1``   the scripted cells (c) and (d) on the reduced drag-inclusive one-cable ODE
             and the closed forms of II.4-II.6 -> records/v2/phase1/phase1_predictions.json;
``theta``    Prop. 1' by Monte Carlo of the reduced Gaussian LTI (tether.theory.reduced_lti)
             at every Phase 2 cell -> cache;
``thm6``     the Thm 6' functional: the one-cable ODE driven by the declared weather
             generator -> cache;
``assemble`` records/v2/phase2/phase2_predictions_i035.json (Part III re-committed at 0.35)
             and records/v2/phase0/p0_w2_completion.json (P0-W2's admissibility).

Usage::

    python -m tether.campaign.v2.claim_a declare
    python -m tether.campaign.v2.claim_a all --workers 4 --cache DIR
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time as wallclock
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from tether.campaign.common import RECORDS, ROOT, json_bytes, relative, run_pool, sha256_bytes, sha256_file, source_state, write_bytes
from tether.campaign.v2 import events as ev
from tether.campaign.v2.phase0_weather import classify_wc_episode, merged_episodes, poisson_interval
from tether.monitor.hazard import impact_law
from tether.physics import constants
from tether.physics.weather import stationary_weather_forces
from tether.theory import drag_excursion as dx

PHASE0_DIR = RECORDS / "v2" / "phase0"
PHASE1_DIR = RECORDS / "v2" / "phase1"
PHASE2_DIR = RECORDS / "v2" / "phase2"
DECLARATIONS_PATH = PHASE1_DIR / "claim_a_declarations.json"
PHASE1_PATH = PHASE1_DIR / "phase1_predictions.json"
PHASE2_PATH = PHASE2_DIR / "phase2_predictions_i035.json"
P0W2_PATH = PHASE0_DIR / "p0_w2_completion.json"
P0W_RESULTS_PATH = PHASE0_DIR / "p0_w_results.json"
IMPACT_TABLE_PATH = RECORDS / "phase1" / "impact_table.json"
EVENTS_MODULE_PATH = ROOT / "tether" / "campaign" / "v2" / "events.py"
REDUCED_LTI_PATH = ROOT / "tether" / "theory" / "reduced_lti.py"
SWAY_RESULTS_PATH = PHASE0_DIR / "sway_results.json"

ENTROPY = 20260913
INTENSITY = 0.35
K_SIGMA = 3.0
TRIM_GAIN = 100.0
CLAMP_DEG = 20.0
PHASE2_CELLS = ((600.0, 477.0), (800.0, 620.0), (1000.0, 763.0), (1200.0, 906.0), (1400.0, 1050.0))
CONTRAST_CELLS = ((600.0, 954.0), (1000.0, 1527.0), (1400.0, 2100.0))
LTI_CELLS = PHASE2_CELLS + CONTRAST_CELLS
N_CABLES = constants.VESSEL_COUNT
TABLE_PRETENSIONS = (600.0, 1000.0, 1400.0, 1800.0)
PAIR_IMPEDANCE = dx.IMPEDANCE

# Phase 1 scripted cells
P1_LAMBDAS = (0.6, 0.9, 1.05, 1.25, 1.5, 2.0)
P1_DURATIONS = (0.2, 0.35, 0.71, 1.5, 3.4, 6.0, 10.0)
P1_PRETENSIONS = (600.0, 1000.0, 1400.0)
P1_GUST_START = 10.0
P1_RUN = 40.0
P1_TEST_CABLE = 2
P1D_AMPLITUDES = (2000.0, 3000.0, 5000.0, 8000.0)
P1D_DURATION = 0.3
P1D_PRETENSION = 1000.0
PLAN_P1D_PEAKS = (4600.0, 7400.0, 11700.0, 15900.0)
PLAN_P1D_FREE_BODY = (6900.0, 12000.0, 22000.0, 36900.0)
DEPTH_BANDS = ((0.05, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 12.0))
LATE_WINDOW = 2.0

# theta Monte Carlo
THETA_STREAM = 71
THETA_PATHS = 100
THETA_DURATION = 4000.0
THETA_HISTORY = 3.0
THETA_TARGET_PRIMARIES = 400
THETA_GUESS = 0.3
THETA_MAX_BATCHES = 20
THETA_MIN_COUNT = 100
THETA_Z_GRID = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
THETA_CHUNK = 1000
FLAG_SIGMAS = 8.0
SPEED_GRID = tuple(np.round(np.arange(0.0, 5.01, 0.1), 2).tolist())
SPEED_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
BOOTSTRAP_REPLICATES = 2000

# Thm 6' Monte Carlo
THM6_BLOCKS = 20
THM6_SEEDS_PER_BLOCK = 60
THM6_SEED_BASE = 7001
THM6_WARMUP = 20.0
THM6_DURATION = 600.0
TB_GRID_KN = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0)
GRID_QUANTILES = (50, 80, 95, 99)
CLASSES = ("N", "R1", "transitional", "R2")
PLANNED_SEEDS = 20
PLANNED_SECONDS = 600.0
LONG_SECONDS = 1200.0
ONSET_RULE = 100
ADMISSIBLE_COUNT = 20
ADMISSIBLE_LEVELS = 3
ADMISSIBLE_CELLS = 3
DEPTH_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
M_FLEET = dx.M_FLEET


DECLARATIONS: dict = {
    "schema_version": 1,
    "date": "2026-09-13",
    "task": "Claim A's predictions, committed before any Phase 1 plant run",
    "plan": "ref/tail_of_the_tether_plan_v2.md: II.1, II.3-II.6, Part III (III.1, III.2), Part V Phase 1 cells (c)(d) and tests P1-T4..T8, P0-W2, IV.13, Appendix A, B.1-B.6, C.1",
    "declared_before_any_result": True,
    "owner_rulings_in_force": {
        "sway_law": "theta_ref = theta_0 - clamp(k_sigma wrap(sigma_hat - phi), +-20 deg), k_sigma = 3 (model-only selection by clause A of IV.7 at intensity 0.35)",
        "phase2_graded_grid": "T0 in {0.6, 0.8, 1.0, 1.2, 1.4} kN at intensity 0.35, parallel formation, Gaussian local weather, k_h = 1.5 x boundary = 477/620/763/906/1050 N m/rad, k_sigma = 3 (saturated); intensity 0.5 cells are out-of-(H1) comparisons only",
        "stage1_outcomes": "P0-W1 FAILED (RV programme withdrawn: Thm 7', Cor. 7.1', Prop. 9', prediction 7 and 13 lapse; Phase 3 not run); P6-T0 FAILED (Claim C withdrawn on this plant: predictions 16, 17 lapse as gates); the psi branch fired (radial-law cells at intensity 0.35)",
    },
    "files": {
        "declarations": "records/v2/phase1/claim_a_declarations.json (this file)",
        "phase1": "records/v2/phase1/phase1_predictions.json",
        "phase2": "records/v2/phase2/phase2_predictions_i035.json",
        "p0_w2": "records/v2/phase0/p0_w2_completion.json",
        "code": "tether/theory/drag_excursion.py, tether/campaign/v2/claim_a.py; declustering rules imported unchanged from tether/campaign/v2/events.py (sha256 recorded in every result)",
    },
    "constants": {
        "m_A_kg": dx.M_A, "c_A_Ns_per_m": dx.C_A, "m_L_kg": dx.M_L, "c_L_Ns_per_m": dx.C_L,
        "m_Lf_kg": "m_L + 4 m_A = 4900 (fleet-side load: the load held by the four other cables)",
        "c_Lf_Ns_per_m": "c_L + 4 c_A = 6900",
        "tau_A_s": "m_A/c_A = 1.7143", "tau_Lf_s": "m_Lf/c_Lf = 0.7101",
        "c_eff_Ns_per_m": "(1/c_A + 1/c_L)^-1 = 329.06 (defines W^c, v_d)",
        "m_eff_kg": "483.87 (defines W_rel, Z, kappa)", "m_fleet_kg": "523.9 (Phase 1 pivot; a_0 = T0/m_fleet; b = 1 - m_eff/m_fleet = 0.0764)",
        "k_N_per_m": constants.CABLE_STIFFNESS, "c_Ns_per_m": constants.CABLE_DAMPING,
        "Z_Ns_per_m": "7993.5 (reference law T_b = T0 + Z v_b); f = 0.880 in kappa",
        "impact_law": "PRIMARY: records/phase1/impact_table.json, PCHIP through the knots with the asymptotic-impedance tail (tether.monitor.hazard.impact_law), per pretension and mirror-symmetric cable position (0 outer, 1 inner, 2 centre). At T0 = 800 and 1200 N (not tabulated) the peak at speed V is the linear interpolation in T0 of the two neighbouring tables' peaks at the same V and position",
    },
    "one_cable_ode": {
        "states": "x = (e, dv_A, dv_L); e' = dv_A - dv_L; dv_A' = (T0 - T + W_A.d)/m_A - dv_A/tau_A; dv_L' = (T - T0 + W_L.d)/m_Lf - dv_L/tau_Lf; T = (k e + c e')_+ 1[e > 0] (plan II.1 three-state system, Thm 6')",
        "integration": "tether.theory.drag_excursion.simulate: exact zero-order-hold map of each linear mode (taut: T = k e + c e'; slack: T = 0) over a 0.5 ms physics step (the plant's production step), the mode chosen from the state at the start of each step; forcing held over 10 ms weather samples",
        "marks": "on the 1 ms sample grid: onset = geometric down-crossing of e = 0 (previous e > 0 >= current), re-engagement = the next up-crossing (previous e <= 0 < current); crossing time and speed linearly interpolated with the elongation crossing fraction (tether.physics.events.linear_crossing_time / linear_crossing_speed arithmetic); u_entry = max(0, -e') at onset; depth = -min e over the interval's 1 ms samples; t_deep = time of that sample",
        "declared_simplifications": "one cable: the four other cables are a rigid fleet side (m_Lf, c_Lf), so the cable mode is sqrt(k (1/m_A + 1/m_Lf)) = 17.8 rad/s rather than the plant's 18.7; no chord swing, no psi, no geometric (II.3) mean correction (the ODE's mean tension is T0); weather on the four other vessels does not enter; the ODE's own engagement peak is reported only as a diagnostic, T_peak always comes from the tabulated impact law at V_up",
    },
    "closed_forms": {
        "R2_drift": "v_d = (W^c - T0)/c_eff; v_s = v_d (1 - exp(-t_x/tau_A)); Delta_g = v_d [t_x - tau_A (1 - exp(-t_x/tau_A))]",
        "post_gust": "delta Delta = tau_A [v_s - v_T ln(1 + v_s/v_T)], t_0 = tau_A ln(1 + v_s/v_T), v_T = T0/c_A",
        "undriven_return": "V_up = v_T (1 - x), v_T tau_A [-ln x - (1 - x)] = Delta; reference cap Z v_T",
        "three_piece_map_R1": "COMMITTED FORM (the plan's): impulsive leg m_A v_s = (W^c_max - T0) t_x with its own depth neglected, drag-limited coast delta Delta(v_s), undriven return from the coast depth. Reported beside it: the same with the drag-free leg depth (A - T0) t_x^2/(2 m_A) added, and with the exact vessel leg (drag included in the leg). I*(T_b): r - ln(1 + r) = -ln(1 - rho) - rho, rho = v_b/v_T, I* = m_A v_T r; A*(T_b, t_x) = T0 + I*/t_x",
        "cor5_square_gust": "V_up = (t_g/m_eff) sqrt((W - T0)(W - b T0)), b = 0.0764; W*(T_b) = ((1 + b) T0 + sqrt((1 - b)^2 T0^2 + 4 kappa T0 T_b^2))/2, kappa = m_eff/(f^2 k T0 t_g^2) (T_b = Z V_up, as the plan's 6.9/12.0/22.0/36.9 kN are Z V_up)",
        "cor5_drag_inclusive": "W^c > T0 + c_eff Delta_b(T_b)/(t_x - tau_A), Delta_b = v_T tau_A [-ln(1 - v_b/v_T) - v_b/v_T], v_b from the impact table; the exact-depth inversion Delta_g(W^c, t_x) = Delta_b reported beside it",
        "prop2": "return time 2u/a, V_up = u, depth u^2/(2a)",
        "prop3prime": "V_up = sqrt(2 a_bar_ret Delta) from the deepest point; a_bar_ret read two ways and both reported: depth-mean (work) acceleration V_up^2/(2 Delta), for which the law is an identity, and time-mean V_up/(t_up - t_deep)",
    },
    "phase1_scripted": {
        "cells_c": "square W^c gusts of amplitude lambda T0, lambda in {0.6, 0.9, 1.05, 1.25, 1.5, 2.0}, t_x in {0.2, 0.35, 0.71, 1.5, 3.4, 6, 10} s, T0 in {0.6, 1.0, 1.4} kN (126 runs)",
        "cells_d": "square impulses of W^c amplitude A in {2, 3, 5, 8} kN, t_g = 0.3 s, T0 = 1 kN (the R1 map's A is W^c_max)",
        "body_force_pattern_COMMITTED": "a vessel-only body force on the test vessel (cable 2, the centre cable, impact-table position c2) of magnitude F = (c_A/c_eff) W^c = 1.06364 W^c, applied at the vessel's centre of mass along -d_i (world-fixed: opposite the tow direction, the equilibrium chord direction of the parallel formation), square in time, zero on every other body. Reason: W^c = c_eff (W_L/c_L - W_A/c_A).d; the vessel-only pattern is the only one that applies no force to the load, so the other four cables are disturbed only through the test cable's own tension; any load component would need c_L/c_eff = 16.7 N per N of W^c on the load and would slacken the others. The same pattern carries W_rel = (m_eff/m_A) F = 0.858 W^c",
        "timing": "zero weather; the fleet starts in steady tow (all cables taut at T0); the force is on over [10 s, 10 s + t_x) (t_x a whole number of 10 ms samples, as every declared t_x is); each run lasts 40 s",
        "ode_realisation": "W_A.d = -F on [10, 10 + t_x), W_L.d = 0, from the taut equilibrium",
        "main_mark": "the first slack interval whose onset lies at or after 10 s; later marks of the run are counted as bounces/members",
        "observables": "u_entry, depth at shut-off Delta_g_obs = -e(10 + t_x) if the main interval is open at shut-off (else 0, flagged 'returned during the gust'), maximum depth, post-gust deepening = max depth - Delta_g_obs, V_up, dwell, T_peak (impact table at T0, position c2; and T0 + Z V_up), the ODE's own peak tension over [t_up, t_up + 0.335 s], a_bar_ret (depth-mean and time-mean), a_bar_ret/a_0 with a_0 = T0/m_fleet, the driven fraction (share of [t_deep, t_up] with the force on)",
        "H4prime_class_of_a_scripted_cell": "N if lambda <= 1 (no W^c > T0 exceedance); else classify_wc_episode(t_x, lambda T0, T0) of tether.campaign.v2.phase0_weather (R1 iff t_x < tau_Lf and v_s < 0.3 v_T; R2 iff t_x > 2 tau_A = 3.4286 s; transitional otherwise). The declared t_x = 3.4 s lies below 2 tau_A, so those cells are transitional, not R2",
        "P1-T4": "lambda < 1 main marks: |V_up - u|/u and depth / (u^2/(2a)) with the plan's a = (1 - lambda) T0/m_fleet, and beside it the ODE's actual constant slack acceleration a_ode = T0 (1/m_A + 1/m_Lf) - F/m_A; each mark flagged whether its slack interval lies inside the gust",
        "P1-T5": "per (H4') class: through-origin slope and uncentred R2 of V_up on sqrt(2 a_bar_ret Delta), a_bar_ret time-mean (the depth-mean reading is an identity, slope 1)",
        "P1-T6": "PRIMARY (drift form): the lambda at which the terminal relative velocity changes sign under the committed forcing, lambda_c = (1 + c_A/c_Lf)/(c_A/c_eff); checked on the t_x = 10 s ODE runs by the drift function y = T_late + c_eff,f e'_late (late window [t_off - 2 s, t_off], c_eff,f = (1/c_A + 1/c_Lf)^-1), which is T0 (1 - lambda/lambda_c) on both sides of the crossing, fitted by least squares over the six lambda per T0. REPORTED: the instantaneous (mass-form) crossing of the slack relative acceleration at zero relative speed, (1 + m_A/m_Lf)/(c_A/c_eff), and its m_fleet variant (m_A/m_fleet)/(c_A/c_eff)",
        "P1-T7": "Delta_g and delta Delta from the ODE against the closed forms per cell (ratio), and the ODE depth's slope in t_x over t_x in {3.4, 6, 10} s against v_d",
        "P1-T8": "median a_bar_ret/a_0 (time-mean and depth-mean) by depth band {0.05-0.3, 0.3-1.0, 1.0-3.0, 3.0-12.0} m and class over the scripted main marks; driven fraction",
        "d_conversions": "T_peak at the ODE's V_up by the impact table (primary), T0 + Z V_up and Z V_up; the committed map's V_up and the plan's 4.6/7.4/11.7/15.9 kN compared by ratio",
    },
    "theta_monte_carlo": {
        "cells": "the five Phase 2 graded cells (T0, k_h) = (600, 477), (800, 620), (1000, 763), (1200, 906), (1400, 1050) and the three 3x-boundary contrast cells (600, 954), (1000, 1527), (1400, 2100), which the psi fallback moves to intensity 0.35 with the graded grid",
        "model": "tether.theory.reduced_lti.reduced_model(ReducedModelInput(formation='parallel', pretension=T0, heading_gain=k_h, trim_gain=100, drag_law='linear', weather_direction='local', weather_scale=0.35, sway_gain=3.0)); the linearisation cannot represent the +-20 deg clamp (it saturates when |sigma - phi| > 6.7 deg at k_sigma = 3): declared limitation, with the (H1)-bound sensitivity of the mean reported",
        "simulation": "the augmented 10 ms transition z_{n+1} = F z_n + (0, eta_n), eta ~ N(0, (1 - phi^2) W), z_0 ~ N(0, P_stationary); e and e' at 1 ms by the exact sub-sample maps of the held weather (the plant's event grid); crossings sought only in 10 ms steps where min(s_n, s_{n+1}) <= 8 x the largest std of the within-step deviation from linear interpolation (missed-crossing probability < 1e-14; equivalence to brute force is a unit test)",
        "paths": "batches of 100 paths x 4000 s; batch count per cell n = clip(ceil(400 / (0.3 x sum_i nu_Rice,i(model mean) x 100 x 3997 s)), 1, 20) (at least the plan's 40 x 4000 s); generator SeedSequence([20260913, 71, cell_index, batch])",
        "levels": "PRIMARY: e = 0 with the mean elongation at the model mean mu_q2/k (II.3 with k_sigma = 3, per cable); SECONDARY: T0/k; CURVE: dev = -z sigma_e,i for z in {2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}",
        "declustering": "B.1 exactly as tether.campaign.v2.events: a down-crossing at t on cable i is primary iff no up-crossing of the same level on any cable lies in (t - 3 s, t] (events.in_exclusion(closed_left=True)); primary exposure = [3 s, 4000 s] less the union of [t_up, t_up + 3 s) (events.excluded_exposure); the first 3 s of each path are history only",
        "theta": "theta = sum primaries / sum_paths (exposure_p x sum_i nu_Rice,i), nu_Rice,i = sigma_edot,i/(2 pi sigma_e,i) exp(-mu_i^2/(2 sigma_e,i^2)) from the model's Lyapunov covariance; interval: path bootstrap, 2000 replicates, percentile 95%, generator SeedSequence([20260913, 72, cell_index]); reported beside it: all down-crossings / Rice (the sampling check), mean cluster size, primary rate per cable-second",
        "unresolved_rule": "if the primary count at the primary level is < 100 at the cap, the committed theta is the same cell's theta at the largest z-grid level with >= 100 primaries, labelled 'level-extrapolated'; the direct estimate and its interval are reported beside it",
        "entry_speed_law": "u/sigma_edot,i of primary down-crossings at the primary level: quantiles {0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99}, mean, the empirical CDF on u/sigma_edot in {0, 0.1, ..., 5.0}, Rayleigh(1) beside it; also in m/s",
        "rates": "declustered primary onset rate per cable-second at the model mean (primary) and at T0/k (secondary) = theta x Rice at that mean (the direct Monte Carlo rate); predicted primary onsets per 20-seed cell = rate x 60,000 cable-s",
    },
    "thm6_monte_carlo": {
        "weather": "tether.physics.weather.stationary_weather_forces(seed, 620 s, direction='local', scale=0.35) (the declared generator; the plant's Phase 2 runs call it with the same arguments), seeds 7001 + 60 b + s, s = 0..59, blocks b = 0..19 (1200 seeds; block 0 is P0-W2's block 0); d = (1, 0) for every cable (parallel formation); cable i reads W_A.d = F[:, i+1, 0], W_L.d = F[:, 0, 0]",
        "cells": "T0 in {600, 800, 1000, 1200, 1400} N on the same weather (common random numbers); the ODE has no heading controller, so the functional is the same for a graded cell and its 3x contrast cell",
        "window": "marks with t_up in [20 s, 620 s) (the plant's 20 s warm-up and 600 s record); exposure 600 s per cable per seed = 3.6e6 cable-s per cell; slack intervals open at 620 s are censored and counted",
        "regime_class_of_a_mark": "(H4') from weather-side covariates only: W^c_i = c_eff (F_L,x/c_L - F_A,i,x/c_A) on the 10 ms grid; the merged W^c > T0 episodes of the whole 620 s record (phase0_weather.merged_episodes, gaps < 1 s merged); the samples whose hold interval [n 10 ms, (n+1) 10 ms) meets [t_on, t_up] and exceed T0; none -> N; else among the merged episodes containing such a sample the one with the largest W^c_max (ties: earliest), t_x its duration, classified by phase0_weather.classify_wc_episode(t_x, W^c_max, T0)",
        "events": "B.1 anchored rule, tether.campaign.v2.events.decluster_marks(t_up, path) per path; event severity = cluster maximum of T_peak (impact table at V_up, the cell's pretension and the cable's position); event class = class of its parent mark; the chained rule is reported as a sensitivity",
        "functional": "Lambda_snap(T_b | class) = events of that class with severity > T_b / exposure, on the declared T_b grid {1, 1.5, ..., 30} kN, per class and for all classes pooled",
        "intervals": "Garwood exact Poisson 95% on the event count; seed bootstrap (2000 replicates, percentile 95%, generator SeedSequence([20260913, 73, cell_index])) over the 1200 seeds",
        "also": "(H4') shares of marks, events and snap exceedances at every level; depth quantiles of event parents per class (P2-T3's committed ODE depth law); onset and mark rates; V_up quantiles; the W^c episode statistics of the same records",
        "r1_closed_form": "log Lambda_R1(T_b; t_x) = log nu_R1 - (A*(T_b, t_x)^2 - T0^2)/(2 sigma_A^2), sigma_A = sigma(W^c) = 0.35 x 690.6 N (analytic, parallel formation), nu_R1 = rate of R1-classified W^c episodes starting in [20, 620) s over the same records, A* from the committed three-piece map with v_b = Z^-1(T_b); tabulated at t_x in {0.1, 0.2, 0.3, 0.5, 0.71} s and at the cell's median R1 t_x",
        "taut_branch": "reported only as the analytic, non-declustered Rice upcrossing rate of q at mu_q2 (sigma_q, sigma_qdot from the LTI); the cluster-maximum analogue of theta is NOT computed here",
    },
    "p0_w2_completion": {
        "planned_exposure": "Phase 2: 20 statistics seeds x 600 s x 5 cables = 60,000 cable-s; a cell whose predicted primary-onset count (theta x Rice at the model mean x 60,000) is below 100 runs at 1200 s (120,000 cable-s)",
        "grid": "PRIMARY: the proxy of the pilot rule, T_b(q) for q in {50, 80, 95, 99} = quantiles (numpy linear) of the ODE's event severities pooled over classes and cables, snapped to the nearest 0.5 kN; no grid if the cell's Monte Carlo holds fewer than 20 events. REPORTED: the declared fixed T_b grid",
        "admissible": "(cell, class) admissible iff Lambda_snap(T_b | class) x planned exposure >= 20 at >= 3 of the 4 grid levels (IV.13); the launch-powered population condition per class: admissible in >= 3 cells (Phase 1 NO-LAUNCH clause, B.6)",
        "cross_check": "the W^c episode counts and class counts of this Monte Carlo's block 0 on each record's first 600 s (60,002 samples) must equal P0-W2's (records/v2/phase0/p0_w_results.json) exactly",
    },
    "part_III_recommit": {
        "III.2_at_0.35": "sigma(W^c) = 0.35 x 690.6 = 241.7 N (parallel formation, analytic); z = T0/sigma; mean excess sigma [phi(z)/Phibar(z) - z]; v_d = excess/c_eff; Delta_g at t_x = 3.4 s = 1.9216 v_d; V_up the undriven return from Delta_g (the plan's arithmetic, post-gust deepening excluded) and, beside it, from Delta_g + delta Delta; T0 + Z V_up and the impact table",
        "table_B_at_0.35": "per cell: k_h, theta (Monte Carlo), z at the model mean and at T0/k, the declustered rate at each, onsets per 20-seed cell, the planned record length (600 or 1200 s)",
        "rows": "III.1 rows 1-6 and 8 recomputed at 0.35; rows 7 and 13 lapsed with P0-W1; rows 9-10 carried (intensity-independent); rows 11-12 from P0-W3's 0.35 tabulation; rows 14-18 carried with their stage-1 status",
    },
    "declarer_expectations_written_before_running": {
        "theta": "0.27-0.33 at the lower pretensions (the plan's band); at 1.2 and 1.4 kN the direct count is expected to be too small (z ~ 5-6 at 0.35), so the level-extrapolated rule is expected to fire there",
        "onsets": "predicted primary onsets per 20-seed cell expected to fall below 100 at T0 >= 1.0 kN at intensity 0.35, so those cells are expected to run at 1200 s and still be sparse",
        "population": "R1 and R2 are expected to be unpowered at 0.35 in every cell except possibly T0 = 0.6 kN (P0-W2: 1156 / 128 / 10 / 0 / 0 W^c episodes per 180,000 cable-s), so the launch-powered condition (>= 3 cells) is expected to fail for R1, R2 and transitional; class N is the only class expected to be populated beyond 0.6 kN, and in the one-cable ODE (no chord swing) class N events are taut-fluctuation onsets only",
        "p1d": "the committed map (impulse-leg depth neglected) is expected within ~5% of the plan's 4.6 kN at A = 2 kN; the full ODE with the leg depth is expected to exceed the committed map by 10-30% at the larger amplitudes",
        "p1_t6": "drift form lambda_c = 0.988 (inside 5% of 1); the instantaneous acceleration form 1.055-1.077 (outside 5% of 1)",
    },
}


# ----------------------------------------------------------------------------- discipline


def _declarations_payload() -> bytes:
    return json_bytes(DECLARATIONS)


def declare() -> None:
    payload = _declarations_payload()
    if DECLARATIONS_PATH.exists():
        if DECLARATIONS_PATH.read_bytes() != payload:
            raise SystemExit("declarations already exist and differ; write a dated addendum instead of editing")
        print("declarations unchanged:", sha256_bytes(payload))
        return
    for path in (PHASE1_PATH, PHASE2_PATH, P0W2_PATH):
        if path.exists():
            raise SystemExit(f"results exist ({relative(path)}); declarations cannot be (re)written after results")
    digest = write_bytes(DECLARATIONS_PATH, payload)
    print("declared", relative(DECLARATIONS_PATH), digest)


def _require_declarations() -> str:
    payload = _declarations_payload()
    if not DECLARATIONS_PATH.exists() or DECLARATIONS_PATH.read_bytes() != payload:
        raise SystemExit("declarations missing or differ from the module; run 'declare' first (never edit after results)")
    return sha256_bytes(payload)


def _provenance() -> dict:
    return {
        "declarations": relative(DECLARATIONS_PATH),
        "declarations_sha256": _require_declarations(),
        "source": source_state(),
        "events_module_sha256": sha256_file(EVENTS_MODULE_PATH),
        "reduced_lti_sha256": sha256_file(REDUCED_LTI_PATH),
        "impact_table_sha256": sha256_file(IMPACT_TABLE_PATH),
        "drag_excursion_sha256": sha256_file(ROOT / "tether" / "theory" / "drag_excursion.py"),
        "claim_a_sha256": sha256_file(Path(__file__)),
    }


# ----------------------------------------------------------------------------- impact law


def cable_position(cable: int) -> int:
    return min(int(cable), N_CABLES - 1 - int(cable))


def impact_peak(speed, pretension: float, position: int) -> np.ndarray:
    """Tabulated engagement peak at closing speed(s) ``speed``; linear in T0 between tables."""
    values = np.maximum(np.asarray(speed, dtype=float), 0.0)
    if pretension in TABLE_PRETENSIONS:
        return np.asarray(impact_law(pretension, position).peak(values), dtype=float)
    lower = max(p for p in TABLE_PRETENSIONS if p < pretension)
    upper = min(p for p in TABLE_PRETENSIONS if p > pretension)
    weight = (pretension - lower) / (upper - lower)
    return (1.0 - weight) * np.asarray(impact_law(lower, position).peak(values)) + weight * np.asarray(impact_law(upper, position).peak(values))


@lru_cache(maxsize=4096)
def critical_speed(level: float, pretension: float, position: int) -> float:
    """v_b = Z^-1(T_b) of the (interpolated) table: bisection on the monotone peak."""
    if impact_peak(0.0, pretension, position) >= level:
        return 0.0
    low, high = 0.0, 1.0
    while impact_peak(high, pretension, position) < level:
        high *= 2.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if impact_peak(mid, pretension, position) < level:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def reference_peak(speed, pretension: float) -> np.ndarray:
    return pretension + PAIR_IMPEDANCE * np.asarray(speed, dtype=float)


# ----------------------------------------------------------------------------- Phase 1


def _scripted_runs() -> list[dict]:
    runs = []
    for pretension in P1_PRETENSIONS:
        for lam in P1_LAMBDAS:
            for t_x in P1_DURATIONS:
                runs.append({"cell": "c", "T0": pretension, "lambda": lam, "W_c": lam * pretension, "t_x": t_x})
    for amplitude in P1D_AMPLITUDES:
        runs.append({"cell": "d", "T0": P1D_PRETENSION, "lambda": amplitude / P1D_PRETENSION, "W_c": amplitude, "t_x": P1D_DURATION})
    return runs


def _class_of_square(w_c: float, t_x: float, pretension: float) -> str:
    return "N" if w_c <= pretension else classify_wc_episode(t_x, w_c, pretension)


def _closed_forms(run: dict) -> dict:
    pretension, w_c, t_x = run["T0"], run["W_c"], run["t_x"]
    out: dict = {"v_T": pretension / dx.C_A}
    if w_c <= pretension:
        out.update({"applicable": False, "note": "W^c <= T0: no drift (II.5); Prop. 2 governs"})
        return out
    v_d = float(dx.drift_speed(w_c, pretension))
    v_s = float(dx.shutoff_speed(v_d, t_x))
    delta_g = float(dx.shutoff_depth(v_d, t_x))
    deepening = float(dx.post_gust_deepening(v_s, pretension))
    total = delta_g + deepening
    v_plan = float(dx.undriven_return_speed(delta_g, pretension))
    v_total = float(dx.undriven_return_speed(total, pretension))
    out.update({
        "applicable": True, "v_d": v_d, "v_s": v_s, "Delta_g": delta_g, "post_gust_deepening": deepening,
        "t_0": float(dx.post_gust_stop_time(v_s, pretension)), "total_depth": total,
        "V_up_undriven_from_total": v_total, "V_up_undriven_from_Delta_g": v_plan,
        "T_peak_table_from_total": float(impact_peak(v_total, pretension, P1_TEST_CABLE)),
        "T_peak_reference_from_total": float(reference_peak(v_total, pretension)),
    })
    return out


def _run_observables(run: dict, index: int, result: dx.SimulationResult, marks: dict) -> dict:
    pretension, t_x = run["T0"], run["t_x"]
    force = dx.VESSEL_FORCE_PER_WC * run["W_c"]
    t_off = P1_GUST_START + t_x
    rows = np.flatnonzero((marks["path"] == index) & (marks["t_on"] >= P1_GUST_START - 1e-9))
    out: dict = {"force_on_vessel_N": force, "W_rel_carried_N": dx.M_EFF / dx.M_A * force}
    censored = [t for p, t in result.marks.censored_onsets if p == index and t >= P1_GUST_START - 1e-9]
    out["censored_open_interval_at_end"] = bool(censored)
    if run["cell"] == "c" and t_x == max(P1_DURATIONS):
        late = (result.trace_time >= t_off - LATE_WINDOW) & (result.trace_time <= t_off)
        state = result.trace_state[late, :, index]
        rate = state[:, 1] - state[:, 2]
        tension = result.trace_tension[late, index]
        c_eff_f = 1.0 / (1.0 / dx.C_A + 1.0 / dx.C_LF)
        out["late_window"] = {"mean_rate": float(np.mean(rate)), "mean_tension": float(np.mean(tension)),
                              "slack_fraction": float(np.mean(state[:, 0] <= 0.0)),
                              "drift_function_over_T0": float(np.mean(tension + c_eff_f * rate) / pretension)}
    if rows.size == 0:
        out["slack"] = bool(censored)
        out["marks"] = 0
        return out
    main = rows[0]
    t_on, t_up = marks["t_on"][main], marks["t_up"][main]
    v_up, depth, t_deep, u = marks["v_up"][main], marks["depth"][main], marks["t_deep"][main], marks["u_entry"][main]
    k_off = int(round(t_off / dx.SAMPLE_PERIOD))
    e_off = float(result.trace_state[k_off, 0, index])
    open_at_shutoff = t_on <= t_off <= t_up and e_off < 0.0
    delta_g_obs = -e_off if open_at_shutoff else 0.0
    window = (result.trace_time >= t_up) & (result.trace_time <= t_up + ev.ENGAGEMENT_PERIOD)
    own_peak = float(np.max(result.trace_tension[window, index])) if window.any() else float("nan")
    return_leg = t_up - t_deep
    a_time = v_up / return_leg if return_leg > 0 else float("nan")
    a_depth = v_up**2 / (2.0 * depth) if depth > 0 else float("nan")
    a0 = pretension / M_FLEET
    driven = max(0.0, min(t_up, t_off) - max(t_deep, P1_GUST_START)) / return_leg if return_leg > 0 else float("nan")
    out.update({
        "slack": True, "marks": int(rows.size), "t_onset_after_gust_start": t_on - P1_GUST_START, "u_entry": u,
        "open_at_shutoff": bool(open_at_shutoff), "returned_during_gust": bool(t_up < t_off),
        "Delta_g": delta_g_obs, "max_depth": depth, "post_gust_deepening": depth - delta_g_obs if open_at_shutoff else None,
        "t_deep_after_gust_start": t_deep - P1_GUST_START, "t_up_after_gust_start": t_up - P1_GUST_START,
        "dwell": t_up - t_on, "V_up": v_up,
        "T_peak_table": float(impact_peak(v_up, pretension, P1_TEST_CABLE)),
        "T_peak_reference": float(reference_peak(v_up, pretension)), "Z_V_up": PAIR_IMPEDANCE * v_up,
        "ode_own_peak_tension": own_peak,
        "a_bar_ret_time_mean": a_time, "a_bar_ret_depth_mean": a_depth,
        "a_bar_ret_time_over_a0": a_time / a0, "a_bar_ret_depth_over_a0": a_depth / a0,
        "driven_fraction_of_return": driven,
        "inside_gust": bool(t_on >= P1_GUST_START and t_up <= t_off),
    })
    return out


def _through_origin(x: np.ndarray, y: np.ndarray) -> dict:
    if x.size < 2:
        return {"n": int(x.size), "slope": None, "uncentred_r2": None}
    slope = float(np.dot(x, y) / np.dot(x, x))
    residual = y - slope * x
    return {"n": int(x.size), "slope": slope, "uncentred_r2": float(1.0 - np.dot(residual, residual) / np.dot(y, y))}


def _appendix_a() -> dict:
    out: dict = {}
    out["A2_table"] = [{"T0": t, "v_T": t / dx.C_A, "v_T_tau_A": t / dx.C_A * dx.TAU_A, "Z_v_T_kN": float(dx.undriven_ceiling(t)) / 1e3,
                        "plan": {600.0: [1.714, 2.94, 13.7], 1000.0: [2.857, 4.90, 22.8], 1400.0: [4.000, 6.86, 32.0], 1800.0: [5.143, 8.82, 41.1]}[t]}
                       for t in (600.0, 1000.0, 1400.0, 1800.0)]
    out["A2_example_T0_1kN_excess_500N"] = {"v_d": float(dx.drift_speed(1500.0, 1000.0)),
                                           "Delta_g": {str(t): float(dx.shutoff_depth(dx.drift_speed(1500.0, 1000.0), t)) for t in (1.0, 2.0, 4.0, 8.0)},
                                           "plan": {"v_d": 1.52, "Delta_g": [0.37, 1.25, 3.73, 9.58]}}
    levels = (4000.0, 4500.0, 6000.0, 8000.0, 10000.0)
    v_b = [critical_speed(level, 1000.0, P1_TEST_CABLE) for level in levels]
    out["A1_v_b_centre_T0_1kN"] = {"T_b_N": levels, "v_b_table": v_b, "v_b_reference": [(level - 1000.0) / PAIR_IMPEDANCE for level in levels],
                                   "plan_v_b": [0.39, 0.46, 0.65, 0.90, 1.16]}
    depth_b = [float(dx.critical_depth(v, 1000.0)) for v in v_b]
    depth_b_ref = [float(dx.critical_depth(v, 1000.0)) for v in [0.39, 0.46, 0.65, 0.90, 1.16]]
    out["A3_drag_inclusive_T0_1kN"] = {
        "T_b_N": levels, "Delta_b_table_v_b": depth_b, "Delta_b_at_plan_v_b": depth_b_ref,
        "required_excess_at_t_x_6s_N": [dx.C_EFF * d / (6.0 - dx.TAU_A) for d in depth_b],
        "required_excess_exact_depth_at_t_x_6s_N": [float(dx.drag_inclusive_severance_level(v, 1000.0, 6.0, exact_depth=True)) - 1000.0 for v in v_b],
        "plan": {"Delta_b": [0.050, 0.071, 0.150, 0.310, 0.563], "excess_at_6s": [4, 5, 12, 24, 43]},
    }
    table = []
    for t_g in (0.5, 1.0, 2.0):
        row = {"t_g": t_g, "kappa": float(dx.square_gust_kappa(1000.0, t_g))}
        for level in (6000.0, 12000.0, 25000.0):
            row[f"W*_b0_at_{level/1e3:.0f}kN"] = float(dx.square_gust_severance_level(level, 1000.0, t_g, b=0.0))
            row[f"W*_b0076_at_{level/1e3:.0f}kN"] = float(dx.square_gust_severance_level(level, 1000.0, t_g))
        table.append(row)
    out["A3_square_gust_T0_1kN"] = {"rows": table, "plan_b0": {"0.5": [1380, 2040, 3570], "1.0": [1120, 1380, 2100], "2.0": [1030, 1120, 1410]}}
    return out


def run_phase1() -> dict:
    provenance = _provenance()
    runs = _scripted_runs()
    samples = int(round(P1_RUN / dx.WEATHER_PERIOD))
    start = int(round(P1_GUST_START / dx.WEATHER_PERIOD))
    w_a = np.zeros((samples, len(runs)))
    for index, run in enumerate(runs):
        length = int(round(run["t_x"] / dx.WEATHER_PERIOD))
        w_a[start:start + length, index] = -dx.VESSEL_FORCE_PER_WC * run["W_c"]
    ode = dx.OneCableODE()
    pretensions = np.array([run["T0"] for run in runs])
    result = dx.simulate(ode, pretensions, w_a, np.zeros((samples, 1)), columns_l=np.zeros(len(runs), dtype=np.int64), trace=True)
    marks = result.marks.arrays()
    rows = []
    for index, run in enumerate(runs):
        record = dict(run)
        record["class_H4prime"] = _class_of_square(run["W_c"], run["t_x"], run["T0"])
        record["closed_form"] = _closed_forms(run)
        record["ode"] = _run_observables(run, index, result, marks)
        cf, od = record["closed_form"], record["ode"]
        if cf.get("applicable") and od.get("slack"):
            record["ratios_ode_over_closed_form"] = {
                "Delta_g": od["Delta_g"] / cf["Delta_g"] if cf["Delta_g"] > 0 else None,
                "post_gust_deepening": (od["post_gust_deepening"] / cf["post_gust_deepening"]) if od.get("post_gust_deepening") is not None and cf["post_gust_deepening"] > 0 else None,
                "V_up": od["V_up"] / cf["V_up_undriven_from_total"],
            }
        rows.append(record)
    cells_c = [r for r in rows if r["cell"] == "c"]
    cells_d = [r for r in rows if r["cell"] == "d"]

    # P1-T6
    t6 = {"analytic": {
        "drift_crossing_three_state_ode": ode.drift_crossing_vessel_only,
        "drift_crossing_plan_free_load_c_eff": dx.OneCableODE(m_l=dx.M_L, c_l=dx.C_L).drift_crossing_vessel_only,
        "onset_acceleration_crossing_m_Lf": ode.onset_acceleration_crossing_vessel_only,
        "onset_acceleration_crossing_m_fleet": (dx.M_A / M_FLEET) / dx.VESSEL_FORCE_PER_WC,
    }, "ode_fit_per_T0": {}}
    for pretension in P1_PRETENSIONS:
        points = [(r["lambda"], r["ode"]["late_window"]["drift_function_over_T0"]) for r in cells_c
                  if r["T0"] == pretension and r["t_x"] == max(P1_DURATIONS) and "late_window" in r["ode"]]
        lam = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        fit = np.polyfit(lam, y, 1) if lam.size >= 2 else None
        t6["ode_fit_per_T0"][str(int(pretension))] = {
            "lambda": lam.tolist(), "drift_function_over_T0": y.tolist(),
            "fitted_crossing": float(-fit[1] / fit[0]) if fit is not None else None,
            "within_5pct_of_1": bool(abs(-fit[1] / fit[0] - 1.0) <= 0.05) if fit is not None else None,
        }
    t6["prediction"] = ("drift form: crossing at lambda = %.4f (inside 5%% of 1: P1-T6 predicted PASS when scored on the terminal drift / quasi-static tension); "
                        "instantaneous acceleration at zero relative speed: %.4f (m_Lf) / %.4f (m_fleet), outside 5%% of 1: P1-T6 predicted FAIL if scored on the onset acceleration"
                        % (t6["analytic"]["drift_crossing_three_state_ode"], t6["analytic"]["onset_acceleration_crossing_m_Lf"], t6["analytic"]["onset_acceleration_crossing_m_fleet"]))

    # P1-T4
    t4 = []
    for r in cells_c:
        if r["lambda"] >= 1.0 or not r["ode"].get("slack") or "V_up" not in r["ode"]:
            continue
        od = r["ode"]
        u = od["u_entry"]
        force = dx.VESSEL_FORCE_PER_WC * r["W_c"]
        a_plan = (1.0 - r["lambda"]) * r["T0"] / M_FLEET
        a_ode_in_gust = r["T0"] * (1.0 / dx.M_A + 1.0 / dx.M_LF) - force / dx.M_A
        a_ode_free = r["T0"] * (1.0 / dx.M_A + 1.0 / dx.M_LF)
        a_ode = a_ode_in_gust if od["inside_gust"] else float("nan")
        t4.append({"T0": r["T0"], "lambda": r["lambda"], "t_x": r["t_x"], "u": u, "V_up": od["V_up"], "depth": od["max_depth"],
                   "inside_gust": od["inside_gust"],
                   "symmetry_error": abs(od["V_up"] - u) / u if u > 0 else None,
                   "depth_over_plan_ballistic": od["max_depth"] / (u * u / (2 * a_plan)) if u > 0 else None,
                   "depth_over_ode_ballistic": (od["max_depth"] / (u * u / (2 * a_ode))) if (u > 0 and od["inside_gust"]) else None,
                   "a_plan": a_plan, "a_ode_in_gust": a_ode_in_gust, "a_ode_after_gust": a_ode_free})
    t4_summary = {
        "marks": len(t4),
        "symmetry_pass_fraction": float(np.mean([m["symmetry_error"] < 0.05 for m in t4 if m["symmetry_error"] is not None])) if t4 else None,
        "depth_plan_pass_fraction": float(np.mean([abs(m["depth_over_plan_ballistic"] - 1) <= 0.10 for m in t4 if m["depth_over_plan_ballistic"] is not None])) if t4 else None,
        "depth_ode_pass_fraction_inside_gust": float(np.mean([abs(m["depth_over_ode_ballistic"] - 1) <= 0.10 for m in t4 if m["depth_over_ode_ballistic"] is not None])) if any(m["depth_over_ode_ballistic"] is not None for m in t4) else None,
    }

    # P1-T5 and P1-T8
    t5 = {}
    t8 = {}
    slack_rows = [r for r in cells_c if r["ode"].get("slack") and "V_up" in r["ode"] and r["ode"]["max_depth"] > 0]
    for cls in CLASSES + ("all",):
        members = [r for r in slack_rows if cls == "all" or r["class_H4prime"] == cls]
        x_time = np.array([math.sqrt(2 * r["ode"]["a_bar_ret_time_mean"] * r["ode"]["max_depth"]) for r in members])
        x_depth = np.array([math.sqrt(2 * r["ode"]["a_bar_ret_depth_mean"] * r["ode"]["max_depth"]) for r in members])
        y = np.array([r["ode"]["V_up"] for r in members])
        t5[cls] = {"time_mean_reading": _through_origin(x_time, y), "depth_mean_reading": _through_origin(x_depth, y)}
        bands = {}
        for low, high in DEPTH_BANDS:
            band = [r for r in members if low <= r["ode"]["max_depth"] < high]
            bands[f"{low}-{high}"] = {
                "n": len(band),
                "median_a_time_over_a0": float(np.median([r["ode"]["a_bar_ret_time_over_a0"] for r in band])) if band else None,
                "median_a_depth_over_a0": float(np.median([r["ode"]["a_bar_ret_depth_over_a0"] for r in band])) if band else None,
                "median_driven_fraction": float(np.median([r["ode"]["driven_fraction_of_return"] for r in band])) if band else None,
            }
        t8[cls] = bands

    # P1-T7
    t7 = {}
    for pretension in P1_PRETENSIONS:
        for lam in P1_LAMBDAS:
            if lam <= 1.0:
                continue
            members = [r for r in cells_c if r["T0"] == pretension and r["lambda"] == lam]
            long = [r for r in members if r["t_x"] >= 3.4 and r["ode"].get("open_at_shutoff")]
            entry = {"t_x": [r["t_x"] for r in members],
                     "Delta_g_ode": [r["ode"].get("Delta_g") for r in members],
                     "Delta_g_closed": [r["closed_form"]["Delta_g"] for r in members],
                     "ratio": [r.get("ratios_ode_over_closed_form", {}).get("Delta_g") for r in members],
                     "deepening_ratio": [r.get("ratios_ode_over_closed_form", {}).get("post_gust_deepening") for r in members],
                     "v_d_closed": members[0]["closed_form"]["v_d"]}
            if len(long) >= 2:
                slope = np.polyfit([r["t_x"] for r in long], [r["ode"]["Delta_g"] for r in long], 1)[0]
                entry["ode_depth_slope_over_t_x_ge_3.4"] = float(slope)
                entry["slope_over_v_d"] = float(slope / members[0]["closed_form"]["v_d"])
            ratios = [x for x, closed in zip(entry["ratio"], entry["Delta_g_closed"]) if x is not None and closed > 0.05]
            entry["cells_within_15pct_depth_gt_5cm"] = f"{sum(abs(x - 1) <= 0.15 for x in ratios)}/{len(ratios)}"
            t7[f"{int(pretension)}@{lam}"] = entry

    # (d)
    d_rows = []
    for plan_peak, plan_free, r in zip(PLAN_P1D_PEAKS, PLAN_P1D_FREE_BODY, cells_d):
        amplitude, pretension = r["W_c"], r["T0"]
        entry = {"A_N": amplitude, "t_g": P1D_DURATION, "T0": pretension, "plan_T_peak_N": plan_peak, "plan_free_body_N": plan_free,
                 "class_H4prime": r["class_H4prime"], "ode": r["ode"]}
        maps = {
            "committed_map": dx.three_piece_map(amplitude, pretension, P1D_DURATION),
            "with_drag_free_leg_depth": dx.three_piece_map(amplitude, pretension, P1D_DURATION, include_impulse_depth=True),
            "with_exact_vessel_leg": dx.three_piece_map(amplitude, pretension, P1D_DURATION, include_impulse_depth=True, drag_in_impulse=True),
        }
        for name, mapped in maps.items():
            v = mapped.return_speed
            entry[name] = {"v_s": mapped.shutoff_speed, "depth": mapped.total_depth, "V_up": v,
                           "T_peak_table": float(impact_peak(v, pretension, P1_TEST_CABLE)),
                           "T_peak_reference": float(reference_peak(v, pretension)), "Z_V_up": PAIR_IMPEDANCE * v,
                           "first_order_ratio": mapped.first_order_ratio}
            for form in ("T_peak_table", "T_peak_reference", "Z_V_up"):
                entry[name][f"{form}_over_plan"] = entry[name][form] / plan_peak
        if r["ode"].get("slack"):
            for form in ("T_peak_table", "T_peak_reference", "Z_V_up"):
                entry["ode"][f"{form}_over_plan"] = r["ode"][form] / plan_peak
        free = float(dx.square_gust_return_speed(amplitude, pretension, P1D_DURATION))
        entry["cor5_free_body"] = {"V_up": free, "Z_V_up": PAIR_IMPEDANCE * free, "Z_V_up_over_plan": PAIR_IMPEDANCE * free / plan_free,
                                   "T_peak_table": float(impact_peak(free, pretension, P1_TEST_CABLE))}
        entry["committed_plant_prediction_T_peak_N"] = r["ode"].get("T_peak_table")
        d_rows.append(entry)

    # prediction 6: R1 surface at T0 = 1 kN
    surface = []
    for level, plan_impulse in zip((4500.0, 6000.0, 10000.0, 12000.0), (300.0, 450.0, 930.0, 1230.0)):
        v_b = critical_speed(level, 1000.0, P1_TEST_CABLE)
        v_ref = (level - 1000.0) / PAIR_IMPEDANCE
        impulse = float(dx.required_impulse(v_b, 1000.0))
        surface.append({"T_b_N": level, "v_b_table": v_b, "I_star_Ns": impulse, "I_star_reference_law_Ns": float(dx.required_impulse(v_ref, 1000.0)),
                        "plan_I_star_Ns": plan_impulse, "A_star_at_0.3s": 1000.0 + impulse / 0.3, "A_star_at_0.71s": 1000.0 + impulse / 0.71})

    results = {
        "schema_version": 1, **provenance,
        "what": "Phase 1 cells (c)(d) predictions on the reduced drag-inclusive one-cable ODE, closed forms, P1-T4..T8 model forms, Appendix A recomputed",
        "committed_forcing": DECLARATIONS["phase1_scripted"]["body_force_pattern_COMMITTED"],
        "cells_c": cells_c, "cells_d": d_rows,
        "P1-T4": {"marks": t4, "summary": t4_summary},
        "P1-T5": t5, "P1-T6": t6, "P1-T7": t7, "P1-T8": t8,
        "prediction_6_R1_surface_T0_1kN": surface,
        "appendix_A": _appendix_a(),
    }
    digest = write_bytes(PHASE1_PATH, json_bytes(results))
    print("phase1", relative(PHASE1_PATH), digest)
    return results


# ----------------------------------------------------------------------------- theta Monte Carlo


@lru_cache(maxsize=16)
def lti_setup(pretension: float, heading_gain: float) -> dict:
    from scipy.linalg import expm

    from tether.theory.reduced_lti import AR1_COEFFICIENT, SAMPLE_PERIOD, ReducedModelInput, reduced_model

    model = reduced_model(ReducedModelInput(formation="parallel", pretension=pretension, heading_gain=heading_gain, trim_gain=TRIM_GAIN,
                                            drag_law="linear", weather_direction="local", weather_scale=INTENSITY, sway_gain=K_SIGMA))
    size = model.transition.shape[0]
    state_size = model.state_matrix.shape[0]
    sub = 10
    augmented = np.zeros((size, size))
    augmented[:state_size, :state_size] = model.state_matrix
    augmented[:state_size, state_size:] = model.input_matrix
    maps = [expm(augmented * SAMPLE_PERIOD * j / sub) for j in range(sub)]
    held_next = model.transition.copy()
    held_next[state_size:, :] = 0.0  # x part of the next state; w part irrelevant to e, e'
    maps.append(held_next)
    c_e, c_edot = model.outputs["e"], model.outputs["edot"]
    d_e = np.stack([c_e @ m for m in maps])  # (11, 5, size)
    d_edot = np.stack([c_edot @ m for m in maps])
    covariance = model.covariance
    deviation = [d_e[j] - (1 - j / sub) * d_e[0] - (j / sub) * d_e[sub] for j in range(1, sub)]
    dev_std = np.max(np.stack([np.sqrt(np.maximum(np.diag(d @ covariance @ d.T), 0.0)) for d in deviation]), axis=0)
    eigenvalues, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    root = vectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    noise_std = np.sqrt(np.maximum(np.diag(model.innovation_covariance)[state_size:], 0.0))
    k = constants.CABLE_STIFFNESS
    mu_primary = np.asarray(model.mu_q2) / k
    mu_secondary = np.asarray(model.mu_e)
    sigma_e, sigma_edot = np.asarray(model.sigma_e), np.asarray(model.sigma_edot)
    levels = {"primary": mu_primary, "secondary": mu_secondary}
    for z in THETA_Z_GRID:
        levels[f"z{z}"] = z * sigma_e
    return {
        "model": model, "size": size, "state_size": state_size, "transition": model.transition, "root": root, "noise_std": noise_std,
        "d_e": d_e, "d_edot": d_edot, "flag_margin": FLAG_SIGMAS * dev_std + 1e-15, "levels": levels,
        "sigma_e": sigma_e, "sigma_edot": sigma_edot, "unstable": model.unstable_eigenvalues.size,
        "leak_e": model.leakage["e"], "leak_edot": model.leakage["edot"],
    }


def rice_rates(setup: dict, mean: np.ndarray) -> np.ndarray:
    sigma_e, sigma_edot = setup["sigma_e"], setup["sigma_edot"]
    return sigma_edot / (2 * math.pi * sigma_e) * np.exp(-0.5 * (np.asarray(mean) / sigma_e) ** 2)


def theta_batches(setup: dict) -> int:
    rate = float(np.sum(rice_rates(setup, setup["levels"]["primary"])))
    expected = THETA_GUESS * rate * THETA_PATHS * (THETA_DURATION - THETA_HISTORY)
    return int(min(THETA_MAX_BATCHES, max(1, math.ceil(THETA_TARGET_PRIMARIES / expected)))) if expected > 0 else THETA_MAX_BATCHES


@dataclass(frozen=True)
class ThetaJob:
    cell_index: int
    pretension: float
    heading_gain: float
    batch: int
    paths: int = THETA_PATHS
    duration: float = THETA_DURATION
    cost: float = 1.0


def lti_crossings(setup: dict, rng: np.random.Generator, paths: int, steps: int, level_means: dict, chunk: int = THETA_CHUNK,
                  brute_force: bool = False) -> dict:
    """Simulate ``paths`` LTI paths for ``steps`` 10 ms steps; return crossings per level.

    Crossings of e_abs = dev + mu (per level) on the 1 ms grid, found only inside flagged
    steps (or in every step with ``brute_force``).  Returns, per level, arrays of
    (path, cable, time, rate, direction) with direction +1 up, -1 down.
    """
    size, state_size = setup["size"], setup["state_size"]
    transition = setup["transition"]
    z = setup["root"] @ rng.standard_normal((size, paths))
    noise_std = setup["noise_std"]
    d_e = setup["d_e"].reshape(-1, size)
    d_edot = setup["d_edot"].reshape(-1, size)
    c_e = setup["d_e"][0]
    margin = setup["flag_margin"]
    names = list(level_means)
    means = np.stack([np.asarray(level_means[n]) for n in names])  # (L, 5)
    found = {n: {"path": [], "cable": [], "time": [], "rate": [], "direction": []} for n in names}
    done = 0
    while done < steps:
        m_count = min(chunk, steps - done)
        store = np.empty((m_count + 1, size, paths))
        store[0] = z
        noise = rng.standard_normal((m_count, size - state_size, paths)) * noise_std[:, None]
        for m in range(m_count):
            nxt = transition @ store[m]
            nxt[state_size:] += noise[m]
            store[m + 1] = nxt
        z = store[m_count]
        dev = np.einsum("ck,mkp->mcp", c_e, store)  # (M+1, 5, P)
        if brute_force:
            flagged = np.ones((m_count, paths), dtype=bool)
        else:
            flagged = np.zeros((m_count, paths), dtype=bool)
            for level in means:
                s = dev + level[None, :, None]
                low = np.minimum(s[:-1], s[1:]) <= margin[None, :, None]
                flagged |= low.any(axis=1)
        m_idx, p_idx = np.nonzero(flagged)
        if m_idx.size:
            gathered = store[m_idx, :, p_idx]  # (F, size)
            sub_e = (gathered @ d_e.T).reshape(-1, 11, N_CABLES)
            sub_r = (gathered @ d_edot.T).reshape(-1, 11, N_CABLES)
            base = (done + m_idx).astype(float)
            for name, level in zip(names, means):
                s = sub_e + level[None, None, :]
                prev, cur = s[:, :-1, :], s[:, 1:, :]
                down = (prev > 0.0) & (cur <= 0.0)
                up = (prev <= 0.0) & (cur > 0.0)
                for direction, mask in ((-1, down), (1, up)):
                    f, j, c = np.nonzero(mask)
                    if f.size == 0:
                        continue
                    first, second = prev[f, j, c], cur[f, j, c]
                    fraction = -first / (second - first)
                    t = (base[f] + (j + fraction) / 10.0) * 0.01
                    rate = sub_r[f, j, c] + fraction * (sub_r[f, j + 1, c] - sub_r[f, j, c])
                    rec = found[name]
                    rec["path"].append(p_idx[f])
                    rec["cable"].append(c)
                    rec["time"].append(t)
                    rec["rate"].append(rate)
                    rec["direction"].append(np.full(f.size, direction))
        done += m_count
    out = {}
    for name in names:
        rec = found[name]
        if rec["path"]:
            arrays = {k: np.concatenate(v) for k, v in rec.items()}
        else:
            arrays = {"path": np.zeros(0, int), "cable": np.zeros(0, int), "time": np.zeros(0), "rate": np.zeros(0), "direction": np.zeros(0, int)}
        order = np.lexsort((arrays["cable"], arrays["time"], arrays["path"]))
        out[name] = {k: v[order] for k, v in arrays.items()}
    return out


def decluster_path(times: np.ndarray, cables: np.ndarray, directions: np.ndarray, low: float, high: float) -> dict:
    """B.1 primaries and exposure of one path's crossings of one level."""
    ups = times[directions > 0]
    down_rows = np.flatnonzero((directions < 0) & (times >= low) & (times <= high))
    primary = ~ev.in_exclusion(times[down_rows], ups, ev.EXCLUSION_WINDOW, closed_left=True)
    return {"down_rows": down_rows, "primary": primary, "exposure": ev.excluded_exposure(ups, low, high, ev.EXCLUSION_WINDOW)}


def theta_job(job: ThetaJob) -> dict:
    setup = lti_setup(job.pretension, job.heading_gain)
    rng = np.random.default_rng(np.random.SeedSequence([ENTROPY, THETA_STREAM, job.cell_index, job.batch]))
    steps = int(round(job.duration / 0.01))
    crossings = lti_crossings(setup, rng, job.paths, steps, setup["levels"])
    out = {"cell_index": job.cell_index, "batch": job.batch, "levels": {}}
    for name, rec in crossings.items():
        n_down = np.zeros((job.paths, N_CABLES), dtype=np.int64)
        n_primary = np.zeros((job.paths, N_CABLES), dtype=np.int64)
        n_up = np.zeros((job.paths, N_CABLES), dtype=np.int64)
        exposure = np.zeros(job.paths)
        speeds_primary, speeds_all, cables_primary = [], [], []
        for p in range(job.paths):
            rows = np.flatnonzero(rec["path"] == p)
            t, c, r, d = rec["time"][rows], rec["cable"][rows], rec["rate"][rows], rec["direction"][rows]
            res = decluster_path(t, c, d, THETA_HISTORY, job.duration)
            exposure[p] = res["exposure"]
            dr = res["down_rows"]
            np.add.at(n_down[p], c[dr], 1)
            np.add.at(n_primary[p], c[dr][res["primary"]], 1)
            ups = (d > 0) & (t >= THETA_HISTORY)
            np.add.at(n_up[p], c[ups], 1)
            if name in ("primary", "secondary"):
                speeds_primary.append(np.maximum(0.0, -r[dr][res["primary"]]))
                cables_primary.append(c[dr][res["primary"]])
                speeds_all.append(np.maximum(0.0, -r[dr]))
        entry = {"n_down": n_down, "n_primary": n_primary, "n_up": n_up, "exposure": exposure}
        if name in ("primary", "secondary"):
            entry["u_primary"] = np.concatenate(speeds_primary) if speeds_primary else np.zeros(0)
            entry["u_primary_cable"] = np.concatenate(cables_primary) if cables_primary else np.zeros(0, int)
            entry["u_all"] = np.concatenate(speeds_all) if speeds_all else np.zeros(0)
        out["levels"][name] = entry
    return out


def run_theta(workers: int, cache: Path) -> dict:
    _require_declarations()
    jobs = []
    plan = {}
    for index, (pretension, gain) in enumerate(LTI_CELLS):
        setup = lti_setup(pretension, gain)
        batches = theta_batches(setup)
        plan[index] = batches
        for batch in range(batches):
            jobs.append(ThetaJob(index, pretension, gain, batch, cost=float(pretension)))
    print("theta batches per cell:", plan, "jobs:", len(jobs), flush=True)
    started = wallclock.perf_counter()
    rows = run_pool(theta_job, jobs, workers, progress=lambda d, n: print(f"  theta {d}/{n}  {wallclock.perf_counter() - started:.0f} s", flush=True) if d % 5 == 0 or d == n else None)
    cells = {}
    for index, (pretension, gain) in enumerate(LTI_CELLS):
        mine = [r for r in rows if r["cell_index"] == index]
        mine.sort(key=lambda r: r["batch"])
        cells[index] = {"pretension": pretension, "heading_gain": gain, "batches": len(mine), "levels": {}}
        for name in mine[0]["levels"]:
            merged = {}
            for key in mine[0]["levels"][name]:
                parts = [r["levels"][name][key] for r in mine]
                merged[key] = np.concatenate(parts) if parts[0].ndim else np.array(parts)
            cells[index]["levels"][name] = merged
    path = cache / "theta_cache.npz"
    flat = {}
    for index, cell in cells.items():
        flat[f"{index}/meta"] = np.array([cell["pretension"], cell["heading_gain"], cell["batches"]])
        for name, merged in cell["levels"].items():
            for key, value in merged.items():
                flat[f"{index}/{name}/{key}"] = value
    np.savez_compressed(path, **flat)
    print("theta cache", path, f"{wallclock.perf_counter() - started:.0f} s", flush=True)
    return cells


def load_theta(cache: Path) -> dict:
    data = np.load(cache / "theta_cache.npz")
    cells: dict = {}
    for key in data.files:
        parts = key.split("/")
        index = int(parts[0])
        cell = cells.setdefault(index, {"levels": {}})
        if parts[1] == "meta":
            cell["pretension"], cell["heading_gain"], cell["batches"] = float(data[key][0]), float(data[key][1]), int(data[key][2])
        else:
            cell["levels"].setdefault(parts[1], {})[parts[2]] = data[key]
    return cells


def _bootstrap(numerator: np.ndarray, denominator: np.ndarray, stream: int, cell_index: int) -> list:
    rng = np.random.default_rng(np.random.SeedSequence([ENTROPY, stream, cell_index]))
    n = numerator.size
    if n == 0 or np.sum(denominator) <= 0:
        return [None, None]
    draws = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, n))
    ratio = numerator[draws].sum(axis=1) / denominator[draws].sum(axis=1)
    return [float(np.quantile(ratio, 0.025)), float(np.quantile(ratio, 0.975))]


def summarize_theta(cells: dict) -> dict:
    out = {}
    for index, cell in sorted(cells.items()):
        setup = lti_setup(cell["pretension"], cell["heading_gain"])
        model = setup["model"]
        span = THETA_DURATION - THETA_HISTORY
        entry = {"T0": cell["pretension"], "k_h": cell["heading_gain"], "batches": cell["batches"],
                 "paths": int(cell["levels"]["primary"]["exposure"].size), "path_seconds": float(cell["levels"]["primary"]["exposure"].size * THETA_DURATION),
                 "levels": {}}
        for name, lev in cell["levels"].items():
            mean = setup["levels"][name] if name in ("primary", "secondary") else None
            if name in ("primary", "secondary"):
                rice = rice_rates(setup, mean)
            else:
                rice = rice_rates(setup, setup["levels"][name])
            exposure = lev["exposure"]
            primaries = lev["n_primary"].sum(axis=1)
            downs = lev["n_down"].sum(axis=1)
            weight = exposure * rice.sum()
            total_primary = int(primaries.sum())
            theta = total_primary / weight.sum() if weight.sum() > 0 else float("nan")
            record = {
                "rice_per_cable": rice.tolist(), "rice_mean_per_cable": float(rice.mean()),
                "primaries": total_primary, "down_crossings": int(downs.sum()), "up_crossings": int(lev["n_up"].sum()),
                "primary_exposure_s": float(exposure.sum()), "run_seconds": float(exposure.size * span),
                "theta": theta, "theta_ci95_path_bootstrap": _bootstrap(primaries.astype(float), weight, 72, index),
                "theta_poisson95": [x / (weight.sum()) for x in (poisson_interval(total_primary, 1.0))],
                "all_over_rice": float(downs.sum() / (exposure.size * span * rice.sum())) if rice.sum() > 0 else None,
                "mean_cluster_size": float(downs.sum() / total_primary) if total_primary else None,
                "primary_rate_per_cable_s": total_primary / (N_CABLES * exposure.sum()),
                "primary_rate_poisson95": poisson_interval(total_primary, N_CABLES * exposure.sum()),
            }
            if mean is not None:
                record["mean_elongation_per_cable_m"] = np.asarray(mean).tolist()
                record["z_per_cable"] = (np.asarray(mean) / setup["sigma_e"]).tolist()
                u = lev["u_primary"]
                cab = lev["u_primary_cable"].astype(int)
                if u.size:
                    normalized = u / setup["sigma_edot"][cab]
                    record["entry_speed_law"] = {
                        "n": int(u.size),
                        "quantiles_u_over_sigma_edot": {str(q): float(np.quantile(normalized, q)) for q in SPEED_QUANTILES},
                        "mean_u_over_sigma_edot": float(np.mean(normalized)),
                        "quantiles_u_m_per_s": {str(q): float(np.quantile(u, q)) for q in SPEED_QUANTILES},
                        "ecdf_grid_u_over_sigma_edot": list(SPEED_GRID),
                        "ecdf": [float(np.mean(normalized <= g)) for g in SPEED_GRID],
                        "rayleigh_cdf": [float(1 - math.exp(-0.5 * g * g)) for g in SPEED_GRID],
                        "rayleigh_median": math.sqrt(2 * math.log(2)),
                        "max_u_over_sigma_edot": float(np.max(normalized)),
                    }
                else:
                    record["entry_speed_law"] = {"n": 0}
            else:
                record["z"] = float(name[1:])
            entry["levels"][name] = record
        primary = entry["levels"]["primary"]
        if primary["primaries"] >= THETA_MIN_COUNT:
            entry["theta_committed"] = {"value": primary["theta"], "ci95": primary["theta_ci95_path_bootstrap"], "form": "direct"}
        else:
            resolved = [(float(n[1:]), rec) for n, rec in entry["levels"].items() if n.startswith("z") and rec["primaries"] >= THETA_MIN_COUNT]
            if resolved:
                z, rec = max(resolved, key=lambda item: item[0])
                entry["theta_committed"] = {"value": rec["theta"], "ci95": rec["theta_ci95_path_bootstrap"], "form": f"level-extrapolated from z = {z}",
                                            "direct_value": primary["theta"], "direct_ci95": primary["theta_ci95_path_bootstrap"], "direct_primaries": primary["primaries"]}
            else:
                entry["theta_committed"] = {"value": None, "form": "unresolved at every level", "direct_value": primary["theta"]}
        theta = entry["theta_committed"]["value"]
        for name in ("primary", "secondary"):
            rec = entry["levels"][name]
            rate = theta * rec["rice_mean_per_cable"] if theta is not None else None
            rec["declustered_rate_theta_x_rice"] = rate
        entry["model"] = {
            "unstable_eigenvalues": int(setup["unstable"]), "leakage_e": setup["leak_e"], "leakage_edot": setup["leak_edot"],
            "sigma_e_m": setup["sigma_e"].tolist(), "sigma_edot_m_per_s": setup["sigma_edot"].tolist(),
            "sigma_edot_over_sigma_e": (setup["sigma_edot"] / setup["sigma_e"]).tolist(),
            "sigma_eddot_over_sigma_edot": (np.asarray(model.sigma_eddot) / setup["sigma_edot"]).tolist(),
            "sigma_q_N": np.asarray(model.sigma_q).tolist(), "sigma_qdot_N_per_s": np.asarray(model.sigma_qdot).tolist(),
            "mu_q2_N": np.asarray(model.mu_q2).tolist(), "mu_q2_sigma_term_N": np.asarray(model.mu_q2_terms["sigma"]).tolist(),
            "mu_q2_psi_term_N": np.asarray(model.mu_q2_terms["psi"]).tolist(),
            "chord_std_deg": np.degrees(np.asarray(model.sigma_chord_angle)).tolist(), "psi_std_deg": np.degrees(np.asarray(model.sigma_psi)).tolist(),
            "clamp_saturation_fraction_at_model_chord_std": float(2 * _normal_sf(math.radians(CLAMP_DEG) / K_SIGMA / float(np.max(model.sigma_chord_angle)))),
            "speed_m_per_s": float(model.operating.speed), "thrust_N": float(np.mean(model.operating.thrusts)),
            "flag_margin_m": setup["flag_margin"].tolist(),
        }
        out[str(index)] = entry
    return out


def _normal_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


# ----------------------------------------------------------------------------- Thm 6' Monte Carlo


@dataclass(frozen=True)
class Thm6Job:
    block: int
    cost: float = 1.0


THM6_PRETENSIONS = tuple(cell[0] for cell in PHASE2_CELLS)


def thm6_seeds(block: int) -> list[int]:
    return [THM6_SEED_BASE + THM6_SEEDS_PER_BLOCK * block + s for s in range(THM6_SEEDS_PER_BLOCK)]


def classify_mark(t_on: float, t_up: float, w_c: np.ndarray, episodes: tuple, pretension: float) -> tuple[str, float, float]:
    """(class, t_x, W^c_max) of one slack interval from the weather side (declared rule)."""
    period = dx.WEATHER_PERIOD
    first = max(0, int(math.floor(t_on / period)))
    last = min(w_c.size - 1, int(math.floor(t_up / period)))
    if last < first or not np.any(w_c[first:last + 1] > pretension):
        return "N", float("nan"), float("nan")
    starts, stops, peaks = episodes
    best = None
    for a, b, peak in zip(starts, stops, peaks):
        lo, hi = max(a, first), min(b, last + 1)
        if lo < hi and np.any(w_c[lo:hi] > pretension):
            if best is None or peak > best[2]:
                best = (a, b, peak)
    if best is None:  # pragma: no cover - an exceeding sample always lies in some episode
        return "N", float("nan"), float("nan")
    duration = (best[1] - best[0]) * period
    return classify_wc_episode(duration, best[2], pretension), duration, float(best[2])


def thm6_job(job: Thm6Job) -> dict:
    seeds = thm6_seeds(job.block)
    n_seeds = len(seeds)
    samples = int(np.ceil((THM6_WARMUP + THM6_DURATION) / dx.WEATHER_PERIOD)) + 2
    w_a = np.empty((samples, n_seeds * N_CABLES))
    w_l = np.empty((samples, n_seeds))
    w_c = np.empty((samples, n_seeds * N_CABLES))
    for s, seed in enumerate(seeds):
        forces = stationary_weather_forces(seed, THM6_WARMUP + THM6_DURATION, direction="local", scale=INTENSITY)
        w_a[:, N_CABLES * s:N_CABLES * (s + 1)] = forces[:, 1:, 0]
        w_l[:, s] = forces[:, 0, 0]
        w_c[:, N_CABLES * s:N_CABLES * (s + 1)] = dx.C_EFF * (forces[:, 0:1, 0] / dx.C_L - forces[:, 1:, 0] / dx.C_A)
    n_cells = len(THM6_PRETENSIONS)
    per_cell = n_seeds * N_CABLES
    columns_a = np.tile(np.arange(per_cell), n_cells)
    columns_l = np.tile(np.repeat(np.arange(n_seeds), N_CABLES), n_cells)
    pretensions = np.repeat(np.array(THM6_PRETENSIONS), per_cell)
    started = wallclock.perf_counter()
    result = dx.simulate(dx.OneCableODE(), pretensions, w_a, w_l, columns_a=columns_a, columns_l=columns_l)
    elapsed = wallclock.perf_counter() - started
    marks = result.marks.arrays()
    censored = result.marks.censored_onsets
    out = {"block": job.block, "seeds": seeds, "wall_s": elapsed, "cells": {}}
    end = THM6_WARMUP + THM6_DURATION
    for c, pretension in enumerate(THM6_PRETENSIONS):
        rows = np.flatnonzero((marks["path"] // per_cell) == c)
        local = marks["path"][rows] % per_cell
        keep = (marks["t_up"][rows] >= THM6_WARMUP) & (marks["t_up"][rows] < end)
        onsets_window = int(np.sum((marks["t_on"][rows] >= THM6_WARMUP) & (marks["t_on"][rows] < end)))
        censored_window = sum(1 for p, t in censored if p // per_cell == c and t >= THM6_WARMUP)
        rows, local = rows[keep], local[keep]
        episodes = {}
        cls = []
        t_x = []
        wc_max = []
        for row, column in zip(rows, local):
            if column not in episodes:
                episodes[column] = merged_episodes(w_c[:, column], pretension)
            label, duration, peak = classify_mark(marks["t_on"][row], marks["t_up"][row], w_c[:, column], episodes[column], pretension)
            cls.append(label)
            t_x.append(duration)
            wc_max.append(peak)
        cls = np.array(cls, dtype="<U12")
        cable = local % N_CABLES
        seed_local = local // N_CABLES
        v_up = marks["v_up"][rows]
        peak = np.zeros(rows.size)
        for position in range(3):
            sel = np.array([cable_position(x) == position for x in cable], dtype=bool)
            if sel.any():
                peak[sel] = impact_peak(v_up[sel], pretension, position)
        decl = ev.decluster_marks(marks["t_up"][rows], local) if rows.size else None
        chained = ev.decluster_marks(marks["t_up"][rows], local, rule="chained") if rows.size else None
        cell = {
            "mark_seed": seed_local, "mark_cable": cable, "mark_class": cls, "mark_t_on": marks["t_on"][rows], "mark_t_up": marks["t_up"][rows],
            "mark_u": marks["u_entry"][rows], "mark_v_up": v_up, "mark_depth": marks["depth"][rows], "mark_peak": peak,
            "mark_t_x": np.array(t_x), "mark_wc_max": np.array(wc_max),
            "onsets_in_window": onsets_window, "censored_open_intervals": censored_window,
        }
        if decl is not None:
            severity, _ = ev.event_cluster_maxima(peak, decl)
            cell["event_parent"] = decl.event_parent
            cell["event_severity"] = severity
            cell["event_marks"] = decl.event_marks
            sev_chained, _ = ev.event_cluster_maxima(peak, chained)
            cell["chained_parent"] = chained.event_parent
            cell["chained_severity"] = sev_chained
        else:
            for key in ("event_parent", "event_severity", "event_marks", "chained_parent", "chained_severity"):
                cell[key] = np.zeros(0)
        # weather-side episode statistics
        ep_class_window, ep_tx_window, ep_seed_window = [], [], []
        check = {"R1": 0, "transitional": 0, "R2": 0}
        check_total = 0
        first_600 = int(np.ceil(THM6_DURATION / dx.WEATHER_PERIOD)) + 2
        for column in range(per_cell):
            starts, stops, peaks = episodes[column] if column in episodes else merged_episodes(w_c[:, column], pretension)
            for a, b, p in zip(starts, stops, peaks):
                if THM6_WARMUP <= a * dx.WEATHER_PERIOD < end:
                    ep_class_window.append(classify_wc_episode((b - a) * dx.WEATHER_PERIOD, p, pretension))
                    ep_tx_window.append((b - a) * dx.WEATHER_PERIOD)
                    ep_seed_window.append(column // N_CABLES)
            if job.block == 0:
                s2, e2, p2 = merged_episodes(w_c[:first_600, column], pretension)
                check_total += int(s2.size)
                for a, b, p in zip(s2, e2, p2):
                    check[classify_wc_episode((b - a) * dx.WEATHER_PERIOD, p, pretension)] += 1
        cell["episode_class"] = np.array(ep_class_window, dtype="<U12")
        cell["episode_t_x"] = np.array(ep_tx_window)
        cell["episode_seed"] = np.array(ep_seed_window, dtype=np.int64)
        if job.block == 0:
            cell["p0w2_check"] = {"episodes": check_total, **check}
        out["cells"][str(int(pretension))] = cell
    return out


def run_thm6(workers: int, cache: Path) -> list:
    _require_declarations()
    jobs = [Thm6Job(block) for block in range(THM6_BLOCKS)]
    started = wallclock.perf_counter()
    rows = run_pool(thm6_job, jobs, workers, progress=lambda d, n: print(f"  thm6 {d}/{n}  {wallclock.perf_counter() - started:.0f} s", flush=True))
    import pickle

    with open(cache / "thm6_cache.pkl", "wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("thm6 cache", cache / "thm6_cache.pkl", f"{wallclock.perf_counter() - started:.0f} s", flush=True)
    return rows


def load_thm6(cache: Path) -> list:
    import pickle

    with open(cache / "thm6_cache.pkl", "rb") as handle:
        return pickle.load(handle)


def _snap(value: float) -> float:
    return float(math.floor(value / 500.0 + 0.5) * 500.0)


def summarize_thm6(rows: list) -> dict:
    rows = sorted(rows, key=lambda r: r["block"])
    out = {}
    n_seeds_total = sum(len(r["seeds"]) for r in rows)
    exposure = n_seeds_total * N_CABLES * THM6_DURATION
    levels = np.array(TB_GRID_KN) * 1e3
    for c, pretension in enumerate(THM6_PRETENSIONS):
        key = str(int(pretension))
        severity, ev_class, ev_seed, ev_depth, ev_vup, ev_tx, ev_u = [], [], [], [], [], [], []
        ch_severity, ch_class, ch_seed = [], [], []
        mark_class, onsets, censored = [], 0, 0
        ep_class, ep_tx, ep_seed = [], [], []
        check = None
        for position, r in enumerate(rows):
            cell = r["cells"][key]
            offset = position * len(r["seeds"])
            mark_class.append(cell["mark_class"])
            onsets += cell["onsets_in_window"]
            censored += cell["censored_open_intervals"]
            parent = cell["event_parent"].astype(int)
            if parent.size:
                severity.append(cell["event_severity"])
                ev_class.append(cell["mark_class"][parent])
                ev_seed.append(cell["mark_seed"][parent] + offset)
                ev_depth.append(cell["mark_depth"][parent])
                ev_vup.append(cell["mark_v_up"][parent])
                ev_tx.append(cell["mark_t_x"][parent])
                ev_u.append(cell["mark_u"][parent])
                cp = cell["chained_parent"].astype(int)
                ch_severity.append(cell["chained_severity"])
                ch_class.append(cell["mark_class"][cp])
                ch_seed.append(cell["mark_seed"][cp] + offset)
            ep_class.append(cell["episode_class"])
            ep_tx.append(cell["episode_t_x"])
            ep_seed.append(cell["episode_seed"] + offset)
            if "p0w2_check" in cell:
                check = cell["p0w2_check"]

        def cat(parts, dtype=float):
            return np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)

        severity, ev_class, ev_seed = cat(severity), cat(ev_class, "<U12"), cat(ev_seed, int).astype(int)
        ev_depth, ev_vup, ev_tx, ev_u = cat(ev_depth), cat(ev_vup), cat(ev_tx), cat(ev_u)
        ch_severity, ch_class, ch_seed = cat(ch_severity), cat(ch_class, "<U12"), cat(ch_seed, int).astype(int)
        mark_class = cat(mark_class, "<U12")
        ep_class, ep_tx, ep_seed = cat(ep_class, "<U12"), cat(ep_tx), cat(ep_seed, int).astype(int)
        rng = np.random.default_rng(np.random.SeedSequence([ENTROPY, 73, c]))
        draws = rng.integers(0, n_seeds_total, size=(BOOTSTRAP_REPLICATES, n_seeds_total))
        multiplicity = np.zeros((BOOTSTRAP_REPLICATES, n_seeds_total), dtype=np.int32)
        for b in range(BOOTSTRAP_REPLICATES):
            multiplicity[b] = np.bincount(draws[b], minlength=n_seeds_total)

        def curve(mask_events: np.ndarray, sev: np.ndarray, seeds: np.ndarray) -> dict:
            counts, rates, garwood, boot = [], [], [], []
            for level in levels:
                sel = mask_events & (sev > level)
                count = int(sel.sum())
                counts.append(count)
                rates.append(count / exposure)
                garwood.append(poisson_interval(count, exposure))
                per_seed = np.bincount(seeds[sel], minlength=n_seeds_total).astype(float)
                replicate = multiplicity @ per_seed / exposure
                boot.append([float(np.quantile(replicate, 0.025)), float(np.quantile(replicate, 0.975))])
            return {"counts": counts, "rate_per_cable_s": rates, "garwood95": garwood, "seed_bootstrap95": boot}

        functional = {}
        for cls in CLASSES + ("all",):
            mask = np.ones(severity.size, dtype=bool) if cls == "all" else (ev_class == cls)
            functional[cls] = curve(mask, severity, ev_seed)
        chained_functional = {cls: curve(np.ones(ch_severity.size, bool) if cls == "all" else ch_class == cls, ch_severity, ch_seed)["counts"] for cls in CLASSES + ("all",)}
        shares = {"marks": {cls: float(np.mean(mark_class == cls)) if mark_class.size else None for cls in CLASSES},
                  "events": {cls: float(np.mean(ev_class == cls)) if ev_class.size else None for cls in CLASSES},
                  "snap_exceedances_by_level": {cls: [(float(np.mean(ev_class[severity > level] == cls)) if np.any(severity > level) else None) for level in levels] for cls in CLASSES}}
        depth = {}
        for cls in CLASSES + ("all",):
            sel = np.ones(ev_depth.size, bool) if cls == "all" else ev_class == cls
            depth[cls] = {"n": int(sel.sum()), "quantiles_m": {str(q): float(np.quantile(ev_depth[sel], q)) for q in DEPTH_QUANTILES} if sel.any() else None,
                          "mean_m": float(np.mean(ev_depth[sel])) if sel.any() else None,
                          "V_up_quantiles": {str(q): float(np.quantile(ev_vup[sel], q)) for q in DEPTH_QUANTILES} if sel.any() else None}
        grid = None
        if severity.size >= ADMISSIBLE_COUNT:
            grid = {"quantiles_N": {str(q): float(np.quantile(severity, q / 100.0)) for q in GRID_QUANTILES}}
            grid["snapped_N"] = {str(q): _snap(v) for q, v in grid["quantiles_N"].items()}
            proxy = {}
            for cls in CLASSES + ("all",):
                mask = np.ones(severity.size, bool) if cls == "all" else ev_class == cls
                proxy[cls] = [float(np.sum(mask & (severity > level)) / exposure) for level in grid["snapped_N"].values()]
            grid["rate_per_cable_s"] = proxy
        r1 = ep_class == "R1"
        out[key] = {
            "T0": pretension, "seeds": n_seeds_total, "exposure_cable_s": exposure,
            "onsets_in_window": onsets, "onset_rate_per_cable_s": onsets / exposure, "censored_open_intervals": censored,
            "marks": int(mark_class.size), "events": int(severity.size), "events_chained_rule": int(ch_severity.size),
            "T_b_grid_N": levels.tolist(), "functional": functional, "functional_chained_counts": chained_functional,
            "shares": shares, "event_depth_law": depth, "grid_proxy": grid,
            "severity_max_N": float(np.max(severity)) if severity.size else None,
            "weather_episodes": {
                "count_window": int(ep_class.size), "rate_per_cable_s": ep_class.size / exposure,
                "class_counts": {cls: int(np.sum(ep_class == cls)) for cls in ("R1", "transitional", "R2")},
                "R1_rate_per_cable_s": float(np.sum(r1) / exposure),
                "R1_median_t_x": float(np.median(ep_tx[r1])) if r1.any() else None,
            },
            "p0w2_block0_check": check,
            "event_parent_t_x_median_by_class": {cls: (float(np.nanmedian(ev_tx[ev_class == cls])) if np.any(ev_class == cls) and cls != "N" else None) for cls in CLASSES},
            "event_parent_u_median": float(np.median(ev_u)) if ev_u.size else None,
        }
    return out


# ----------------------------------------------------------------------------- assembly


def _analytic_rice_05(pretension: float, gain: float) -> dict:
    from tether.theory.reduced_lti import ReducedModelInput, reduced_model

    model = reduced_model(ReducedModelInput(formation="parallel", pretension=pretension, heading_gain=gain, trim_gain=TRIM_GAIN,
                                            drag_law="linear", weather_direction="local", weather_scale=0.5, sway_gain=K_SIGMA))
    sigma_e, sigma_edot = np.asarray(model.sigma_e), np.asarray(model.sigma_edot)
    mu = np.asarray(model.mu_q2) / constants.CABLE_STIFFNESS
    rice = sigma_edot / (2 * math.pi * sigma_e) * np.exp(-0.5 * (mu / sigma_e) ** 2)
    return {"z_model_mean": float(np.mean(mu / sigma_e)), "z_T0_over_k": float(np.mean(np.asarray(model.mu_e) / sigma_e)),
            "rice_model_mean_per_cable_s": float(np.mean(rice)), "chord_std_deg": float(np.degrees(np.max(model.sigma_chord_angle))),
            "psi_std_deg": float(np.degrees(np.max(model.sigma_psi))), "status": "out-of-(H1) comparison, analytic Rice only, no theta"}


def _iii2_row(pretension: float, sigma: float) -> dict:
    from scipy.stats import norm

    z = pretension / sigma
    hazard = float(norm.pdf(z) / norm.sf(z))
    excess = sigma * (hazard - z)
    v_d = excess / dx.C_EFF
    delta_g = float(dx.shutoff_depth(v_d, 3.4))
    v_plan = float(dx.undriven_return_speed(delta_g, pretension))
    v_s = float(dx.shutoff_speed(v_d, 3.4))
    total = delta_g + float(dx.post_gust_deepening(v_s, pretension))
    v_total = float(dx.undriven_return_speed(total, pretension))
    return {"T0": pretension, "z": z, "phi_over_Phibar": hazard, "mean_excess_N": excess, "v_d": v_d, "Delta_g_3.4s": delta_g,
            "V_up": v_plan, "T0_plus_Z_V_up_kN": (pretension + PAIR_IMPEDANCE * v_plan) / 1e3,
            "T_peak_table_centre_kN": float(impact_peak(v_plan, pretension, 2)) / 1e3,
            "with_post_gust_deepening": {"total_depth": total, "V_up": v_total, "T0_plus_Z_V_up_kN": (pretension + PAIR_IMPEDANCE * v_total) / 1e3,
                                         "T_peak_table_centre_kN": float(impact_peak(v_total, pretension, 2)) / 1e3}}


def _p0w3_at(p0w3: dict, what: str) -> dict:
    out = {}
    for name, entry in p0w3.get("table", {}).items():
        row = {}
        for load in ("W_rel", "W_c"):
            if load not in entry:
                continue
            if what == "std" and "analytic_std_N_intensity_1" in entry[load]:
                row[f"{load}_std_N"] = [INTENSITY * x for x in entry[load]["analytic_std_N_intensity_1"]]
            if what == "factor" and "factor_at_1kN" in entry[load]:
                row[f"{load}_exp(-T0^2/2sigma^2)_at_1kN"] = entry[load]["factor_at_1kN"].get(str(INTENSITY))
        out[name] = row
    return out


def assemble(cache: Path) -> None:
    provenance = _provenance()
    theta = summarize_theta(load_theta(cache))
    thm6 = summarize_thm6(load_thm6(cache))
    p0w = json.loads(P0W_RESULTS_PATH.read_text())
    sigma_wc = INTENSITY * dx.C_EFF * math.sqrt((constants.LOAD_WEATHER_STD / dx.C_L) ** 2 + (constants.VESSEL_WEATHER_STD / dx.C_A) ** 2)
    planned = PLANNED_SEEDS * PLANNED_SECONDS * N_CABLES

    # Table B at 0.35 and exposure decisions
    table_b = []
    exposure_by_cell = {}
    for index, (pretension, gain) in enumerate(LTI_CELLS):
        entry = theta[str(index)]
        primary, secondary = entry["levels"]["primary"], entry["levels"]["secondary"]
        rate_model = primary["declustered_rate_theta_x_rice"]
        rate_t0k = secondary["declustered_rate_theta_x_rice"]
        onsets = rate_model * planned if rate_model is not None else None
        long = onsets is None or onsets < ONSET_RULE
        exposure_by_cell[index] = LONG_SECONDS if long else PLANNED_SECONDS
        table_b.append({
            "cell_index": index, "T0": pretension, "k_h": gain, "role": "graded" if index < len(PHASE2_CELLS) else "3x-gain contrast",
            "k_h_boundary": 0.318 * (1.5 * pretension + TRIM_GAIN),
            "theta": entry["theta_committed"], "z_model_mean": float(np.mean(primary["z_per_cable"])), "z_T0_over_k": float(np.mean(secondary["z_per_cable"])),
            "rate_model_mean_per_cable_s": rate_model, "rate_T0_over_k_per_cable_s": rate_t0k,
            "direct_primary_rate_model_mean": primary["primary_rate_per_cable_s"], "direct_primary_rate_model_mean_poisson95": primary["primary_rate_poisson95"],
            "onsets_per_20_seed_cell_600s_model_mean": onsets, "onsets_per_20_seed_cell_600s_T0_over_k": rate_t0k * planned if rate_t0k is not None else None,
            "record_length_s": exposure_by_cell[index], "planned_cable_s": planned * exposure_by_cell[index] / PLANNED_SECONDS,
            "predicted_primary_onsets_at_planned_length": rate_model * planned * exposure_by_cell[index] / PLANNED_SECONDS if rate_model is not None else None,
            "H1_model": {"chord_std_deg": max(entry["model"]["chord_std_deg"]), "psi_std_deg": max(entry["model"]["psi_std_deg"]),
                         "meets_15deg": bool(max(entry["model"]["chord_std_deg"]) <= 15 and max(entry["model"]["psi_std_deg"]) <= 15),
                         "clamp_saturation_fraction": entry["model"]["clamp_saturation_fraction_at_model_chord_std"]},
        })

    # (H1)-bound sensitivity of the mean (reported, not a prediction)
    sensitivity = []
    for row in table_b:
        entry = theta[str(row["cell_index"])]
        setup = lti_setup(row["T0"], row["k_h"])
        model = setup["model"]
        speed = float(model.operating.speed)
        mean15 = row["T0"] + 0.5 * (row["T0"] + dx.C_A * speed) * math.radians(15.0) ** 2 + np.asarray(model.mu_q2_terms["psi"])
        rice15 = rice_rates(setup, mean15 / constants.CABLE_STIFFNESS)
        rice_model = rice_rates(setup, setup["levels"]["primary"])
        sensitivity.append({"T0": row["T0"], "k_h": row["k_h"], "mu_q2_at_chord_std_15deg_N": float(np.mean(mean15)),
                            "rice_factor_vs_model_mean": float(np.sum(rice15) / np.sum(rice_model))})

    # III.2 at 0.35
    iii2 = [_iii2_row(t, sigma_wc) for t in THM6_PRETENSIONS]

    # Thm 6' per cell (+ contrast cells share the graded cell's functional)
    r1_curves = {}
    for key, cell in thm6.items():
        pretension = cell["T0"]
        nu = cell["weather_episodes"]["R1_rate_per_cable_s"]
        median_tx = cell["weather_episodes"]["R1_median_t_x"]
        curves = {}
        for t_x in (0.1, 0.2, 0.3, 0.5, 0.71) + ((median_tx,) if median_tx else ()):
            amps, rates = [], []
            for level in cell["T_b_grid_N"]:
                v_b = critical_speed(level, pretension, 2)
                amp = float(dx.r1_required_amplitude(v_b, pretension, t_x))
                amps.append(amp)
                rates.append(float(dx.r1_rate_curve(amp, pretension, sigma_wc, nu)) if math.isfinite(amp) else 0.0)
            curves[f"{t_x:.3f}"] = {"A_star_N": amps, "rate_per_cable_s": rates}
        r1_curves[key] = {"sigma_A_N": sigma_wc, "nu_R1_per_cable_s": nu, "median_R1_t_x": median_tx, "curves": curves,
                          "T_b_grid_N": cell["T_b_grid_N"], "v_b_centre": [critical_speed(level, pretension, 2) for level in cell["T_b_grid_N"]]}

    # taut branch (analytic Rice of q, non-declustered)
    taut = {}
    for index, (pretension, gain) in enumerate(LTI_CELLS):
        model = theta[str(index)]["model"]
        mu, sq, sqd = np.array(model["mu_q2_N"]), np.array(model["sigma_q_N"]), np.array(model["sigma_qdot_N_per_s"])
        levels = np.array(TB_GRID_KN) * 1e3
        rates = [(sqd / (2 * math.pi * sq) * np.exp(-0.5 * ((level - mu) / sq) ** 2)).mean() for level in levels]
        taut[str(index)] = {"T0": pretension, "k_h": gain, "T_b_grid_N": levels.tolist(), "rice_upcrossing_rate_per_cable_s": [float(r) for r in rates],
                            "note": "analytic, not declustered; the cluster-maximum analogue of theta is not computed here"}

    # P0-W2 completion
    admissibility = {}
    powered_cells = {cls: [] for cls in CLASSES}
    cross_check = {}
    for index, (pretension, gain) in enumerate(LTI_CELLS):
        cell = thm6[str(int(pretension))]
        length = exposure_by_cell[index]
        cable_s = planned * length / PLANNED_SECONDS
        p0 = p0w["P0-W2"]["cells"].get(f"{int(pretension)}@{INTENSITY}")
        entry = {"T0": pretension, "k_h": gain, "role": "graded" if index < len(PHASE2_CELLS) else "3x-gain contrast",
                 "record_length_s": length, "planned_cable_s": cable_s,
                 "predicted_primary_onsets": table_b[index]["predicted_primary_onsets_at_planned_length"],
                 "P0-W2_episode_statistics": {"W_c": p0["W_c"], "W_rel": p0["W_rel"], "H4prime_forecast": p0["H4prime_forecast"]} if p0 else None,
                 "model_onsets_expected": cell["onset_rate_per_cable_s"] * cable_s, "model_events_expected": cell["events"] / cell["exposure_cable_s"] * cable_s,
                 "classes": {}}
        grid = cell["grid_proxy"]
        entry["grid_proxy"] = grid["snapped_N"] if grid else None
        for cls in CLASSES:
            fixed_counts = [r * cable_s for r in cell["functional"][cls]["rate_per_cable_s"]]
            fixed_levels_ok = int(sum(c >= ADMISSIBLE_COUNT for c in fixed_counts))
            if grid:
                counts = [r * cable_s for r in grid["rate_per_cable_s"][cls]]
                ok = int(sum(c >= ADMISSIBLE_COUNT for c in counts))
                verdict = ok >= ADMISSIBLE_LEVELS
            else:
                counts, ok, verdict = None, 0, False
            entry["classes"][cls] = {"predicted_counts_at_proxy_grid": counts, "levels_with_20": ok, "admissible": bool(verdict),
                                     "predicted_counts_fixed_grid": fixed_counts, "fixed_grid_levels_with_20": fixed_levels_ok,
                                     "fixed_grid_admissible_secondary": bool(fixed_levels_ok >= ADMISSIBLE_LEVELS)}
            if verdict and index < len(PHASE2_CELLS):
                powered_cells[cls].append(pretension)
        admissibility[str(index)] = entry
        if index < len(PHASE2_CELLS) and p0 is not None and cell["p0w2_block0_check"] is not None:
            chk = cell["p0w2_block0_check"]
            ours = {"episodes": chk["episodes"], "R1": chk["R1"], "transitional": chk["transitional"], "R2": chk["R2"]}
            theirs = {"episodes": p0["W_c"]["episodes"], **p0["H4prime_forecast"]["W_c_episode_class_counts"]}
            cross_check[str(int(pretension))] = {"this_monte_carlo_block0": ours, "P0-W2": theirs, "equal": ours == theirs}
    launch = {cls: {"admissible_graded_cells": powered_cells[cls], "launch_powered_ge_3_cells": len(powered_cells[cls]) >= ADMISSIBLE_CELLS} for cls in CLASSES}
    regimes_powered = [cls for cls in CLASSES if launch[cls]["launch_powered_ge_3_cells"]]
    p0w2 = {
        "schema_version": 1, **provenance,
        "p0_w_results_sha256": sha256_file(P0W_RESULTS_PATH),
        "test": "P0-W2 (regime population forecast, blocking): the reduced-model snap rate per (H4') class x planned exposure, admissible iff >= 20 at >= 3 grid levels",
        "intensity": INTENSITY, "cells": admissibility, "cross_check_with_P0-W2": cross_check,
        "all_cross_checks_equal": bool(cross_check) and all(v["equal"] for v in cross_check.values()),
        "population_half_of_phase1_to_2_gate": launch,
        "verdict": {
            "regimes_launch_powered_at_0.35": regimes_powered,
            "regimes_not_launch_powered_at_0.35": [cls for cls in CLASSES if cls not in regimes_powered],
            "statement": ("At intensity 0.35 the regime classes " + (", ".join(regimes_powered) if regimes_powered else "(none)") +
                          " are launch-powered (admissible in >= 3 graded cells); " +
                          ", ".join(f"{cls} in {launch[cls]['admissible_graded_cells'] or 'no cell'}" for cls in CLASSES) +
                          ". A Phase 2 regime law whose class is not launch-powered is not tested (P0-W2 / Phase 1 NO-LAUNCH clause)."),
        },
    }
    write_bytes(P0W2_PATH, json_bytes(p0w2))

    # Part III re-committed
    p0w3 = p0w.get("P0-W3", {})
    graded = [row for row in table_b if row["role"] == "graded"]
    rows_iii1 = {
        "1_theta": {"value": {str(int(r["T0"])): r["theta"] for r in table_b}, "status": "APPROX", "from": "reduced-LTI Monte Carlo at 0.35, k_sigma = 3, 3 s any-cable rule"},
        "2_declustered_primary_onset_rate": {"value": {str(int(r["T0"])) + "@" + str(int(r["k_h"])): {"model_mean": r["rate_model_mean_per_cable_s"], "T0_over_k": r["rate_T0_over_k_per_cable_s"],
                                                                                                    "onsets_per_20_seed_cell_600s": r["onsets_per_20_seed_cell_600s_model_mean"], "record_length_s": r["record_length_s"]} for r in table_b},
                                             "status": "PROV, theta APPROX"},
        "3_terminal_drift_shutoff_depth": {"value": iii2, "status": "PROV given linear drag"},
        "4_snap_severity_mean_excess_3.4s_episode_kN": {"value": {str(int(r["T0"])): {"T0_plus_Z_V_up": r["T0_plus_Z_V_up_kN"], "table": r["T_peak_table_centre_kN"]} for r in iii2},
                                                        "ratio_across_T0": iii2[-1]["T0_plus_Z_V_up_kN"] / iii2[0]["T0_plus_Z_V_up_kN"], "status": "CONJ"},
        "5_direction_in_T0": {"statement": "non-increasing over the powered pretensions, one strict decrease; taut branch rising; no interior optimum",
                              "model_total_event_rate_per_cable_s_at_T_b": {str(level): [thm6[str(int(t))]["functional"]["all"]["rate_per_cable_s"][i] for t in THM6_PRETENSIONS]
                                                                           for i, level in enumerate(TB_GRID_KN) if level in (2.0, 4.0, 6.0, 8.0)},
                              "primary_onset_rate_model_mean": [r["rate_model_mean_per_cable_s"] for r in graded], "status": "CONJ"},
        "6_R1_surface": {"value": "A*(T_b, t_x) = T0 + I*(T_b)/t_x (records/v2/phase1/phase1_predictions.json prediction_6_R1_surface_T0_1kN); rate curves per cell below", "status": "CONJ"},
        "7_RV_local_index": {"status": "LAPSED: P0-W1 FAILED, Thm 7' withdrawn"},
        "8_snap_tail": {"value": {str(int(t)): thm6[str(int(t))]["severity_max_N"] for t in THM6_PRETENSIONS}, "statement": "finite endpoint; the ODE's largest event per cell over 3.6e6 cable-s", "status": "EMP"},
        "9_cascade_transmission_ratio": {"value": 0.44, "band": [0.2, 0.9], "status": "CONJ (intensity-independent, carried)"},
        "10_transmitted_onsets_m_bar": {"value": [0.2, 0.5], "status": "CONJ (carried; measured per cell by P2-T11)"},
        "11_front_differential_std_at_0.35": {"value": _p0w3_at(p0w3, "std"), "status": "PROV given the model (P0-W3, stds scale linearly with intensity)"},
        "12_slack_population_new_cells": {"value": _p0w3_at(p0w3, "factor"), "verdict_P0-W3_at_0.35": p0w3.get("verdict_by_intensity", {}).get("0.35", {}).get("verdict"), "status": "APPROX"},
        "13_class_equivalence": {"status": "LAPSED: P0-W1 FAILED"},
        "14_squall_amplitude": {"status": "carried (mission; not intensity-dependent)"},
        "15_fleet_rollout_coupled_ticks": {"status": "carried (see records/v2/phase5 P5-T2')"},
        "16_prop14_reengagement_speeds": {"status": "LAPSED as a gate: P6-T0 FAILED, Claim C withdrawn on this plant"},
        "17_catch_authority": {"status": "measured by P6-T0 (FAILED)"},
        "18_taut_overload_floor": {"status": "carried"},
    }
    phase2 = {
        "schema_version": 1, **provenance,
        "what": "Part III's predictions re-committed at intensity 0.35 (psi branch), before any Phase 2 statistics seed",
        "owner_rulings": DECLARATIONS["owner_rulings_in_force"],
        "cells": [{"T0": t, "k_h": g, "k_sigma": K_SIGMA, "intensity": INTENSITY, "formation": "parallel", "weather": "Gaussian local AR(1)"} for t, g in LTI_CELLS],
        "table_B": table_b, "III_2": iii2, "sigma_Wc_N": sigma_wc,
        "prop1prime_theta": theta,
        "thm6prime_functional": thm6, "r1_closed_form_curves": r1_curves, "taut_branch_analytic": taut,
        "mean_sensitivity_chord_at_H1_bound": sensitivity,
        "intensity_0.5_out_of_H1_comparisons": {str(int(t)): _analytic_rice_05(t, g) for t, g in PHASE2_CELLS},
        "III_1": rows_iii1,
        "p0_w2_completion": relative(P0W2_PATH),
        "phase1_predictions": relative(PHASE1_PATH),
    }
    digest = write_bytes(PHASE2_PATH, json_bytes(phase2))
    print("phase2", relative(PHASE2_PATH), digest)
    print("p0w2", relative(P0W2_PATH), sha256_file(P0W2_PATH))


# ----------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("declare", "phase1", "theta", "thm6", "assemble", "all"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "claim_a_cache")
    arguments = parser.parse_args()
    workers = min(arguments.workers, 4)
    arguments.cache.mkdir(parents=True, exist_ok=True)
    if arguments.command == "declare":
        declare()
    elif arguments.command == "phase1":
        _require_declarations()
        run_phase1()
    elif arguments.command == "theta":
        run_theta(workers, arguments.cache)
    elif arguments.command == "thm6":
        run_thm6(workers, arguments.cache)
    elif arguments.command == "assemble":
        _require_declarations()
        assemble(arguments.cache)
    else:
        _require_declarations()
        run_phase1()
        run_theta(workers, arguments.cache)
        run_thm6(workers, arguments.cache)
        assemble(arguments.cache)


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
