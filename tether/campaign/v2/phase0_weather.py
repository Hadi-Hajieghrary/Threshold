"""Plan v2 Phase 0 weather preconditions, weather-only (no plant run).

P0-T8 (gust-class clause), P0-W1 (tail index at gust scale, blocking), the
episode-statistics half of P0-W2 (regime-population forecast), and P0-W3 (front lag,
coherence, mixed(rho); blocking for Phase 4 cells).

Discipline: every operational choice is in ``DECLARATIONS`` and is written to
``records/v2/phase0/p0_w_declarations.json`` by ``declare`` before any result exists;
``run`` refuses to compute unless the file on disk equals ``DECLARATIONS`` and records
the file's sha256 in the results.

Usage::

    python -m tether.campaign.v2.phase0_weather declare
    python -m tether.campaign.v2.phase0_weather run --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import time as wallclock
from dataclasses import dataclass

import numpy as np

from tether.campaign.common import RECORDS, json_bytes, relative, run_pool, sha256_bytes, source_state, write_bytes
from tether.physics import constants
from tether.physics.weather import (
    DEFAULT_PREROLL_SECONDS,
    GUST_DURATION_LOG_STD,
    GUST_DURATION_MEDIAN,
    GUST_RATE,
    GUST_SCALE_FACTOR,
    GUST_TAIL_INDEX,
    RAISED_COSINE_ENERGY,
    ar1_coefficient,
    front_arrival_delays,
    gust_background_factor,
    gust_class_weather,
    gust_second_moments,
    hill_tail_index,
    lagged_front_weather_forces,
    pinned_stationary_std,
    stationary_weather_forces,
)
from tether.theory.gust import PAIR_REDUCED_MASS, relative_load_series

OUT_DIR = RECORDS / "v2" / "phase0"
DECLARATIONS_PATH = OUT_DIR / "p0_w_declarations.json"
RESULTS_PATH = OUT_DIR / "p0_w_results.json"

PERIOD = constants.WEATHER_PERIOD
N_CABLES = constants.VESSEL_COUNT
M_L, M_A = constants.LOAD_MASS, constants.VESSEL_MASS
C_L, C_A = constants.LOAD_LINEAR_DRAG, constants.VESSEL_LINEAR_DRAG
DRAG_REDUCED = 1.0 / (1.0 / C_A + 1.0 / C_L)  # c_eff = 329.06 N s/m
TAU_A = M_A / C_A  # 1.714 s
TAU_LF = (M_L + 4.0 * M_A) / (C_L + 4.0 * C_A)  # 0.7101 s
R2_BOUNDARY = 2.0 * TAU_A  # 3.4286 s

RECORD_SECONDS = 600.0
SEEDS_PER_BLOCK = 60
SEED_BASE = 7001
MERGE_GAP_SAMPLES = 100  # a gap of fewer than 100 non-exceeding 10 ms samples (< 1 s) is merged
LEVEL_GRID = np.arange(0.0, 12000.0 + 2.5, 5.0)  # N, intensity 1.0
MAX_BLOCKS = 12
FAN_ARC = 0.55
FAN_PRETENSION = 1000.0
BOW_QUARTERING = -3.0 * math.pi / 4.0
BROADSIDE = -math.pi / 2.0
DAVENPORT_C = 10.0


def block_seeds(block: int) -> list[int]:
    return [SEED_BASE + SEEDS_PER_BLOCK * block + s for s in range(SEEDS_PER_BLOCK)]


DECLARATIONS: dict = {
    "schema_version": 1,
    "date": "2026-09-13",
    "plan": "ref/tail_of_the_tether_plan_v2.md, sections IV.5, Part V Phase 0 (P0-T8 gust clause, P0-W1, P0-W2 episode half, P0-W3), Appendix B, Appendix D",
    "declared_before_any_result": True,
    "tests": ["P0-T8 (gust-class clause)", "P0-W1", "P0-W2 (episode-statistics half)", "P0-W3"],
    "constants": {
        "weather_period_s": PERIOD,
        "tau_w_s": constants.WEATHER_TIME_CONSTANT,
        "phi": "exp(-0.01/8) = 0.998750781",
        "stationary_std_per_component_N_at_intensity_1": {"load": constants.LOAD_WEATHER_STD, "vessel": constants.VESSEL_WEATHER_STD},
        "m_L_kg": M_L, "m_A_kg": M_A, "c_L_Ns_per_m": C_L, "c_A_Ns_per_m": C_A,
        "m_eff_kg": "m_A m_L/(m_A + m_L) = 483.87 (tether.theory.gust.PAIR_REDUCED_MASS, identical to tether.physics.fleet.PAIR_REDUCED_MASS)",
        "c_eff_Ns_per_m": "(1/c_A + 1/c_L)^-1 = 329.06",
        "tau_A_s": "m_A/c_A = 1.7143", "tau_Lf_s": "(m_L + 4 m_A)/(c_L + 4 c_A) = 0.7101", "R2_boundary_s": "2 tau_A = 3.4286",
    },
    "conjugate_loads": {
        "W_rel": "m_eff (W_L/m_L - W_A,i/m_A) . d_i, computed by tether.theory.gust.relative_load_series (same arithmetic as tether.physics.fleet.relative_gust_load, checked sample-by-sample in a unit test)",
        "W_c": "c_eff (W_L/c_L - W_A,i/c_A) . d_i",
        "d_i": "the formation's equilibrium chord direction (FleetGeometry.cable_angles): (1, 0) for every cable of the parallel formation; angles linspace(-0.55, 0.55, 5) for the fan",
        "sign": "positive when the weather closes the gap (II.1)",
    },
    "episodes": {
        "definition": "B.4: maximal runs of samples with series > level on the 10 ms grid, runs separated by a gap of fewer than 100 non-exceeding samples (gap < 1 s) merged; a gap of exactly 1.00 s is not merged (v1 theory/gust.py merged gaps <= 1 s; the difference is immaterial and declared)",
        "duration": "(last exceeding sample - first exceeding sample + 1) x 10 ms of the merged episode (zero-order-hold semantics of the plant's weather input), no interpolation",
        "maximum": "largest sample of the series within the merged episode (the declustered maximum)",
        "record_edges": "episodes in progress at either edge of a 600 s record are kept with their truncated duration; their count is reported",
        "rate": "episodes per cable-second: count / (seeds x 600 s) per cable, pooled rates divide by 5 x that",
        "level_grid_counting": "on a level grid an episode starts at sample j iff series[j] > u and max(series[j-100:j]) <= u (window truncated at the record start); identical to the explicit merge (unit test)",
    },
    "seeds": {
        "records": "600 s kept per seed after a 40 s AR(1) preroll (v1 stationary_weather_forces default); the kept 10 ms grid is that of stationary_weather_forces (ceil(600/0.01) + 2 = 60002 samples, of which all are used)",
        "block": "block b = master seeds 7001 + 60 b + s, s = 0..59 (60 seeds x 600 s = 36,000 s per block, the plan's Phase 0 weather cell); block 0 serves every class and intensity, so cells are paired by common random numbers",
        "streams": "Gaussian background [master, 11, body] (v1); mixed local parts [master, 11, 100 + body] (v1); gusts [master, 19, body] (local) or [master, 19, 0] (fleet-wide)",
    },
    "gust_class": {
        "formula": "W_j(t) = G_j(t) + sum_n a0_j A_n s_n E((t - t_n)/t_g,n)  (tether.physics.weather.gust_class_weather)",
        "arrivals": "Poisson, lambda_g = 0.1 /s per body (local); fleet-wide variants: one process at 0.1 /s for the fleet",
        "window": "arrivals drawn on [-40 s, end of the kept grid) so the term is stationary from the first kept sample",
        "amplitudes": "A_n/a0_j = (1 - U)^(-1/3): Pareto, P(A > a) = (a/a0_j)^-3; a0_j = 1.5 x intensity x pinned std = 5.25 kN (load), 1.05 kN (vessel) at intensity 1",
        "direction": "local: uniform angle on [0, 2 pi) per gust; common/front variants: the declared front angle, sign +1 (pushing along the front); fleet-wide gusts keep their non-zero mean lambda E[A] E[t_g]/2 = 0.157 a0_j along the front, not removed, reported",
        "envelope": "E(x) = (1 - cos 2 pi x)/2 on [0, 1], zero outside (the mission's raised-cosine squall envelope with zero plateau), peak 1 at mid-gust; sampled at t_k = k x 10 ms on the kept clock",
        "duration": "t_g = 2 exp(0.3 Z), Z standard normal (lognormal, median 2 s, log-std 0.3); E[t_g] = 2.0921 s",
        "draw_order": "per stream: count ~ Poisson(lambda x window); arrival times uniform then sorted; U for amplitudes; U for angles (always drawn); Z for durations",
        "front_lag_of_gusts": "front variant: body b receives each gust delays[b] samples late (the same integer delays as the front(c) background)",
        "variance_preservation": {
            "rule": "background per-body std multiplied by f_B = sqrt(1 - g), g = gust variance per component / pinned variance",
            "derivation": "Campbell: Var per component = lambda E[A^2] E[s_x^2] E[t_g] int_0^1 E^2 = 0.1 x 3 a0^2 x 1/2 x 2.0921 s x 3/8 = 0.117678 a0^2 (Pareto(3): E[A^2] = 3 a0^2, finite); a0 = 1.5 sigma => g = 0.264776, f_B = 0.857452; the same fraction for every body, so W_rel and W_c keep their pinned Gaussian std exactly (W_rel 881.9 N, W_c 690.6 N per cable at intensity 1)",
            "fleet_wide_variants": "the same f_B preserves the total (trace) per-body variance (gust variance 0.235 a0^2 along the front, 0 across)",
            "numerical_checks": [
                "construction check (gating the construction, not a plan test): pooled second moment of the gust term per component over block 0 divided by the realized-mark prediction sum_n (a0 A_n)^2 s_n,c^2 t_g,n 3/8 / T (marks with t_n in [0, T)) lies in [0.97, 1.03]",
                "reported: pooled gust-term variance / 0.117678 a0^2 with a seed-bootstrap 95% interval (Pareto(3) variance converges slowly, ~n^-1/3)",
                "reported: W_rel std of the gust class pooled over cables and block 0 against 881.9 N (plan: about 0.88 kN; 0.99 kN without preservation)",
            ],
        },
        "effective_RV_scale": "a_i = E|cos theta|^3^(1/3) (m_eff/m_j) a0_j = 0.7515 (m_eff/m_j) a0_j: 762 N (load gusts), 636 N (vessel gusts) (plan: 0.76 / 0.63 kN)",
    },
    "P0-T8_gust_clause": {
        "sample": "gust amplitudes (marks) A_n/a0_j of every body, block 0, arrivals t_n in [0, 600) s, pooled over bodies and seeds",
        "estimator_primary": "tether.physics.weather.hill_tail_index(marks, upper_order_count = floor(n/10))",
        "estimator_secondary": "known-scale MLE n / sum log(A_n/a0_j); per-body primary estimator",
        "pass": "primary estimate in [2.7, 3.3]",
        "branch": "fail => not NO-GO but blocks P0-W1: the class is re-specified and T8 re-run before W1 is scored (plan outcome matrix); W1 is then reported but not scored",
    },
    "P0-W1": {
        "class": "gust class, local structure, variance-preserving, intensity 1.0",
        "formation": "parallel (Phase 3's cell is the Phase 2 reference cell); d_i = (1, 0) for all five cables. Under local isotropic weather the per-cable marginal law of W_rel is rotation-invariant, so the result is the same for the fan; the Hill index and u_i/intensity are exactly invariant to intensity (all forces scale linearly with it), so intensity 1.0 stands for every Phase 3 intensity with u_i scaled by the intensity",
        "series": "|W_rel,i| (two-sided) per cable",
        "background_primary": "the gust class's own background G_j (std f_B x pinned, the same draws), i.e. 'the Gaussian background' of the gust record; its W_rel std is 756.2 N",
        "background_secondary": "the full-variance Gaussian local class (pinned std, same draws) - reported with its own u_i and the gust-class Hill at that level, non-gating",
        "level_grid": "u in {0, 5, 10, ..., 12000} N",
        "u_i": "smallest grid level u such that r_B(u') <= 0.1 r_G(u') at every grid level u' >= u (B.5 'at most 0.1'), r = merged-episode rate of |W_rel,i| on the cumulative exposure; per cable i; recomputed on the cumulative exposure after each block",
        "exposure_rule": "start with block 0; add blocks 1, 2, ... until min_i k_i >= 200 (k_i = gust-class episodes of cable i above u_i) and the pooled background episode count above the u_i is >= 20; cap 12 blocks (432,000 s); at the cap the test is scored as UNDER-POWERED",
        "estimator": "Hill at a fixed threshold on declustered maxima: alpha_hat = k / sum_e log(M_e/u_i) over the k merged episodes above u_i (maxima M_e of |W_rel,i|, episodes merged at level u_i itself)",
        "primary_statistic": "pooled over the five cables with per-cable thresholds: alpha_hat = sum_i k_i / sum_i sum_e log(M_e/u_i); k = sum_i k_i episodes; interval alpha_hat (1 +- 1.96/sqrt(k))",
        "secondary_statistics": ["per-cable alpha_hat_i with alpha_hat_i (1 +- 1.96/sqrt(k_i))", "seed-level bootstrap (2000 replicates, rng seed 20260913) of the pooled alpha_hat, which keeps the cross-cable dependence through the shared load gusts that the plan's interval ignores", "gust-class Hill at the full-variance-null threshold"],
        "pass": "pooled alpha_hat in [2.4, 3.6] AND pooled background alpha_hat_B (same estimator, background episodes above the same u_i) outside [2.4, 3.6]; per-cable disagreement reported, not gating",
        "branch_on_fail": "NO-LAUNCH of the RV programme only: Thm 7', Cor. 7.1', Prop. 9' withdrawn with this reason; Phase 3 not run; Phase 4 runs its declared Gaussian-only six-cell design {local, mixed(0.5), mixed(0.8), front(10 m/s) bow-quartering, front(5 m/s) bow-quartering, exact common}; the rest of the campaign continues",
        "branch_on_pass": "the RV programme launches (Phase 3 after the Phase 2 gate; Phase 4's gust half)",
        "committed_prior_context": "plan IV.5: with the background included, a weather-only replay gave Hill on episode maxima 4.13-4.86 at u = 2 a0 and 3.6-3.9 at 3 a0 (k = 59-91) against 3.02-3.27 on gust marks alone; the outcome is declared open",
    },
    "P0-W2": {
        "cells": "Phase 2 graded grid T0 in {0.6, 0.8, 1.0, 1.2, 1.4} kN x intensity {0.35, 0.5}; also the other planned Phase 2 weather cells (T0 = 1.0 kN at intensity 1.0, the out-of-(H1) reference, and at 0.2, the null cell). Weather statistics do not depend on the gain, so the gain-contrast and v1-controller cells share their (T0, intensity) rows",
        "class": "Gaussian local (v1 stationary_weather_forces, direction local), parallel formation, block 0 (60 seeds x 600 s = 36,000 s per cable, 180,000 cable-s pooled)",
        "series": "W_c,i and W_rel,i, one-sided, episodes of series > T0 merged across gaps < 1 s",
        "quantities": "per cell and series: episode count, rate per cable-second (pooled over cables) with exact Poisson 95% interval, duration median/p90/p95 (numpy linear quantiles), edge-truncated count; expected episodes over Phase 2's planned exposure (20 statistics seeds x 600 s x 5 cables = 60,000 cable-s) as rate x 60,000",
        "classes_H4prime": "classified per W_c > T0 episode from weather-side covariates only: R1 iff t_x < tau_Lf = 0.7101 s AND v_s = (W_c,max - T0) t_x/m_A < 0.3 v_T with v_T = T0/c_A; R2 iff t_x > 2 tau_A = 3.4286 s; transitional otherwise",
        "class_N_proxy": "class N (no W_c > T0 exceedance during the slack interval) is a property of plant marks and cannot be forecast weather-only; the declared weather-side proxy is the share of W_rel > T0 episodes during which W_c,i never exceeds T0 (no sample of the W_rel episode has W_c,i > T0); reported beside the three W_c shares, labelled a proxy",
        "reduced_model_field": "reduced_model_snap_rate_per_class left null per cell for the reduced-model agent; the admissibility clause (predicted rate x planned exposure >= 20 at >= 3 grid levels) needs it and is not scored here",
        "under_20": "cells with fewer than 20 weather episodes in the pooled exposure are flagged; exposure is not raised",
    },
    "P0-W3": {
        "formation": "fan, formation_geometry('fan', arc_half_angle=0.55), cables at 0, +-15.76, +-31.51 deg, cable 0 starboard (-31.5 deg) ... cable 4 port (+31.5 deg)",
        "positions": "p_b = body-origin (CoM) positions of equilibrium_state(geometry, operating_point(geometry, 1000 N)) - the steady-tow state the production runs start from (load at the origin, vessels at the operating headings); a hull-aligned-with-cable variant (vessel CoM 1.5 m beyond the stern attachment along the chord) is reported as a diagnostic only",
        "front_direction": "d = (cos a, sin a), a the world direction in which the front pushes and travels (v1 phase4 convention): bow-quartering a = -135 deg, broadside a = -90 deg (port cables windward)",
        "front_implementation": "tether.physics.weather.lagged_front_weather_forces: v1 front stream (first component of [master, 11, 0], x sqrt 2), integer delays rint(lag/10 ms), preroll 40 s + max delay; c = inf is exact common and is bit-identical to v1 front weather",
        "configurations": "local; mixed(rho), rho in {0.5, 0.64, 0.8} (v1 mixed_standardized_innovations, isotropic common part); front(c) at c in {5, 10, inf} m/s for bow-quartering and broadside",
        "quantities": "per cable: std of W_rel,i and W_c,i, analytic (closed form from the AR(1) autocorrelation phi^|n| at the integer delays and the per-body weights) and numerical (block 0, 60 seeds x 600 s at intensity 1.0); stds scale exactly linearly with intensity",
        "intensities": "tabulated at intensity 1.0 (where the plan's committed values are written), 0.5 (the intensity of Phase 4's declared calibration cell, P4-T6) and 0.35 (the Phase 2 psi fallback). Phase 4 runs at 'the Phase 2 in-domain reference' intensity, not yet fixed; the binding verdict is the one at that intensity, and all three are reported",
        "exceedance_factor": "exp(-T0^2/(2 sigma_i^2)) at T0 = 1 kN, sigma_i the analytic per-cable std",
        "primary_sigma": "W_rel (the plan's committed values 160 N / 634 N / 74-546 N and the controls' 1e-17 / 0.05 are W_rel values); W_c factors reported beside it with their own verdict, non-gating",
        "scored_cells": "planned Phase 4 direction cells other than the controls: local, mixed(0.5), front(10 m/s) bow-quartering, front(5 m/s) bow-quartering; a cell holds iff max_i factor_i > 0.05",
        "controls": "exact common bow-quartering (c = inf) and mixed(0.8): exempt, reported",
        "reported_only": "broadside fronts (P4-T5 symmetry check, not a Phase 4 cell), mixed(0.64), exact common broadside",
        "consistency_check": "numerical std within 5% of analytic per cable (absolute 5 N where the analytic std is below 100 N)",
        "davenport": "coherence exp(-C f dx/U) with C = 10 (the value implied by the plan's 0.70 over 18 m at 10 m/s and 0.02 Hz), f = the AR(1) corner frequency 1/(2 pi tau_w) = 0.01989 Hz, dx = the fan span along the front, max_b p_b.d - min_b p_b.d, U = c in {5, 10} m/s, for both front angles",
        "lag_table": "for a unit-projection cable, std of W_rel under a pure load-vessel lag Delta in {0.9, 1.8, 3.6} s and its factor at 1 kN (plan: 0.07 / 0.22 / 0.42, local 0.53)",
        "plan_committed_values_to_reproduce": {
            "front10_bow_quartering_W_rel_N": [74, 204, 343, 464, 546],
            "front10_broadside_W_rel_N": [227, 91, 0, 91, 227],
            "front5_max_W_rel_N": 722,
            "exact_common_W_rel_N": 160,
            "local_W_rel_N": [863, 882],
            "mixed05_W_rel_N": 634,
            "mixed_factors_at_1kN": {"0.5": 0.28, "0.64": 0.18, "0.8": 0.05},
            "exact_common_factor": 1e-17,
            "fan_span_m": {"broadside": 18.1, "bow_quartering": 16.8},
            "davenport_18m_10mps_0.02Hz": 0.70,
            "intensity_of_committed_values": 1.0,
        },
    },
    "declarer_expectations_written_before_running": {
        "P0-W3_mixed05": "with v1's isotropic common part the analytic W_rel std at rho = 0.5 is sqrt(0.5 x 881.9^2 + 0.5 x 112.9^2) = 628.7 N (factor 0.28); the plan's 634 N is reproduced only with the front-projected common value 160 N, so a ~1% disagreement is expected",
        "P0-W3_exact_common": "exact common W_rel = 159.7 N x |cos(front - chord)| per cable (160 N is the unit-projection value); the plan's 1e-17 factor corresponds to the isotropic 112.9 N, not 160 N (which gives ~3e-9)",
        "P0-W3_intensity": "the plan's committed stds are at intensity 1.0; stds scale linearly, so at intensity 0.5 mixed(0.5) falls to ~314 N (factor ~0.006) and front(10) bow-quartering to ~273 N max, i.e. the non-control fronts and mixed(0.5) are expected to fail the 0.05 rule at 0.5 and 0.35 and to hold at 1.0; the W_c factors are expected to be much larger than the W_rel ones under common and front forcing (exact common W_c ~ 635 N x |projection|)",
        "P0-W3_positions": "the plan's spans 18.1 / 16.8 m correspond to vessel CoMs 1.5 m beyond the stern attachment along the chord; the declared equilibrium positions put the hulls at the operating headings (~24.5 deg outer), so the broadside span is expected ~0.3 m shorter and the per-cable values to differ by a few percent",
        "P0-W1": "open: the Gaussian convolution raises the local tail index of the sum above a Pareto(3) by about 12 sigma_B^2/u^2 (asymptotically), ~0.8 at u = 3 sigma_B, so an alpha_hat of roughly 3.3-4.0 is expected at u_i and failure of the [2.4, 3.6] band is plausible",
        "gust_W_rel_std": "881.9 N analytic under variance preservation (plan ~0.88 kN); 992 N without it (plan 0.99 kN)",
    },
    "workers": 4,
}


# ----------------------------------------------------------------------------- geometry


def fan_positions(pretension: float = FAN_PRETENSION, arc: float = FAN_ARC, hull_aligned: bool = False):
    """Body-origin positions (N + 1, 2), load first, and chord unit vectors (N, 2) of the fan."""
    from tether.physics.fleet import equilibrium_state, formation_geometry, operating_point  # plant-side module, geometry only

    geometry = formation_geometry("fan", arc_half_angle=arc)
    operating = operating_point(geometry, pretension)
    state = equilibrium_state(geometry, operating)
    positions = state[: 3 * (geometry.vessel_count + 1)].reshape(-1, 3)[:, :2].copy()
    units = np.stack([np.cos(geometry.cable_angles), np.sin(geometry.cable_angles)], axis=1)
    if hull_aligned:
        chord = constants.CABLE_REST_LENGTH + pretension / constants.CABLE_STIFFNESS
        attach = geometry.load_offsets + chord * units
        positions[1:] = attach + 0.5 * 3.0 * units
    return positions, units


def parallel_units() -> np.ndarray:
    return np.tile(np.array([1.0, 0.0]), (N_CABLES, 1))


# ----------------------------------------------------------------------------- loads


def conjugate_loads(forces: np.ndarray, units: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(W_rel, W_c), each (samples, N), from forces (samples, N + 1, 2)."""
    w_rel = relative_load_series(forces, units)
    drag = forces[:, 0, :][:, None, :] / C_L - forces[:, 1:, :] / C_A
    w_c = DRAG_REDUCED * np.einsum("snk,nk->sn", drag, units)
    return w_rel, w_c


# ----------------------------------------------------------------------------- episodes


def merged_episodes(series: np.ndarray, level: float, gap_samples: int = MERGE_GAP_SAMPLES):
    """Merged runs of series > level: (starts, stops exclusive, maxima)."""
    above = np.asarray(series) > level
    padded = np.concatenate([[False], above, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = changes[0::2], changes[1::2]
    if starts.size == 0:
        return starts, stops, np.zeros(0)
    keep = np.concatenate([[True], (starts[1:] - stops[:-1]) >= gap_samples])
    group = np.cumsum(keep) - 1
    merged_stops = np.zeros(int(group[-1]) + 1, dtype=np.int64)
    np.maximum.at(merged_stops, group, stops)
    starts = starts[keep]
    maxima = np.array([float(np.max(series[a:b])) for a, b in zip(starts, merged_stops)])
    return starts, merged_stops, maxima


def episode_counts_on_grid(series: np.ndarray, levels: np.ndarray, gap_samples: int = MERGE_GAP_SAMPLES) -> np.ndarray:
    """Number of merged episodes of series > u at every level u (sorted ascending)."""
    values = np.asarray(series, dtype=float)
    padded = np.concatenate([np.full(gap_samples, -np.inf), values])
    previous = np.lib.stride_tricks.sliding_window_view(padded, gap_samples)[: values.size].max(axis=1)
    starting = values > previous
    low = np.sort(previous[starting])
    high = np.sort(values[starting])
    return np.searchsorted(low, levels, side="right") - np.searchsorted(high, levels, side="right")


def smallest_negligible_level(levels, gust_counts, background_counts, ratio: float = 0.1):
    """Smallest grid level from which r_B <= ratio r_G holds at every higher grid level."""
    ok = np.asarray(background_counts, dtype=float) <= ratio * np.asarray(gust_counts, dtype=float)
    if not ok[-1]:
        return None
    bad = np.flatnonzero(~ok)
    index = 0 if bad.size == 0 else int(bad[-1]) + 1
    return float(levels[index])


def hill_at_threshold(maxima: np.ndarray, level: float) -> float:
    maxima = np.asarray(maxima, dtype=float)
    maxima = maxima[maxima > level]
    if maxima.size == 0:
        return float("nan")
    return float(maxima.size / np.sum(np.log(maxima / level)))


def poisson_interval(count: int, exposure: float) -> list[float]:
    from scipy.stats import chi2

    lower = 0.0 if count == 0 else chi2.ppf(0.025, 2 * count) / 2.0
    upper = chi2.ppf(0.975, 2 * count + 2) / 2.0
    return [float(lower / exposure), float(upper / exposure)]


# ----------------------------------------------------------------------------- P0-W1 / T8 workers


@dataclass(frozen=True)
class W1Job:
    seed: int
    stage: str  # "grid" or "maxima"
    thresholds: tuple = ()  # per-cable (u_i, u_full_i) for stage "maxima"
    cost: float = 1.0


def _gust_record(seed: int):
    forces, background, gusts, marks = gust_class_weather(seed, RECORD_SECONDS, return_components=True)
    units = parallel_units()
    w_gust, _ = conjugate_loads(forces, units)
    w_back, _ = conjugate_loads(background, units)
    w_full = w_back / gust_background_factor()
    return forces, background, gusts, marks, w_gust, w_back, w_full


def w1_worker(job: W1Job) -> dict:
    forces, background, gusts, marks, w_gust, w_back, w_full = _gust_record(job.seed)
    if job.stage == "grid":
        out = {"seed": job.seed, "samples": int(w_gust.shape[0])}
        out["gust_counts"] = np.stack([episode_counts_on_grid(np.abs(w_gust[:, i]), LEVEL_GRID) for i in range(N_CABLES)])
        out["back_counts"] = np.stack([episode_counts_on_grid(np.abs(w_back[:, i]), LEVEL_GRID) for i in range(N_CABLES)])
        out["full_counts"] = np.stack([episode_counts_on_grid(np.abs(w_full[:, i]), LEVEL_GRID) for i in range(N_CABLES)])
        out["w_rel_sum"] = [w_gust.sum(axis=0), w_back.sum(axis=0), w_full.sum(axis=0)]
        out["w_rel_sq"] = [(w_gust**2).sum(axis=0), (w_back**2).sum(axis=0), (w_full**2).sum(axis=0)]
        a0 = GUST_SCALE_FACTOR * pinned_stationary_std()
        period_total = gusts.shape[0] * PERIOD
        out["gust_sq"] = (gusts**2).sum(axis=0)  # (bodies, 2)
        out["gust_sum"] = gusts.sum(axis=0)
        predicted = np.zeros((N_CABLES + 1, 2))
        mark_rows = []
        for body, m in enumerate(marks):
            inside = (m.times >= 0.0) & (m.times < RECORD_SECONDS)
            mark_rows.append(m.amplitudes[inside])
            inside_all = (m.times >= 0.0) & (m.times < period_total)
            weight = (a0[body] * m.amplitudes[inside_all]) ** 2 * m.durations[inside_all] * RAISED_COSINE_ENERGY / PERIOD
            predicted[body, 0] = np.sum(weight * np.cos(m.angles[inside_all]) ** 2)
            predicted[body, 1] = np.sum(weight * np.sin(m.angles[inside_all]) ** 2)
        out["gust_sq_predicted"] = predicted
        out["marks"] = mark_rows
        return out
    result = {"seed": job.seed}
    for name, series in (("gust", w_gust), ("back", w_back), ("full", w_full)):
        for which, column in (("u", 0), ("ufull", 1)):
            maxima = []
            for i in range(N_CABLES):
                level = job.thresholds[i][column]
                if level is None:
                    maxima.append(np.zeros(0))
                    continue
                _, _, peaks = merged_episodes(np.abs(series[:, i]), level)
                maxima.append(peaks)
            result[f"{name}_{which}"] = maxima
    return result


# ----------------------------------------------------------------------------- P0-W2 worker

W2_T0 = (600.0, 800.0, 1000.0, 1200.0, 1400.0)
W2_INTENSITIES = (0.35, 0.5)
W2_EXTRA_CELLS = ((1000.0, 1.0), (1000.0, 0.2))


def w2_cells() -> list[tuple[float, float]]:
    return [(t0, s) for s in W2_INTENSITIES for t0 in W2_T0] + list(W2_EXTRA_CELLS)


@dataclass(frozen=True)
class W2Job:
    seed: int
    cost: float = 1.0


def classify_wc_episode(duration: float, peak: float, pretension: float) -> str:
    v_terminal = pretension / C_A
    if duration < TAU_LF and (peak - pretension) * duration / M_A < 0.3 * v_terminal:
        return "R1"
    if duration > R2_BOUNDARY:
        return "R2"
    return "transitional"


def w2_worker(job: W2Job) -> dict:
    units = parallel_units()
    out = {"seed": job.seed, "samples": 0, "cells": {}}
    by_intensity = {}
    for t0, s in w2_cells():
        if s not in by_intensity:
            forces = stationary_weather_forces(job.seed, RECORD_SECONDS, direction="local", scale=s)
            by_intensity[s] = conjugate_loads(forces, units)
            out["samples"] = int(forces.shape[0])
        w_rel, w_c = by_intensity[s]
        cell = {"wc_duration": [], "wc_peak": [], "wc_edge": 0, "wrel_duration": [], "wrel_no_wc": [], "wrel_edge": 0}
        n = w_rel.shape[0]
        for i in range(N_CABLES):
            a, b, peaks = merged_episodes(w_c[:, i], t0)
            cell["wc_duration"].extend(((b - a) * PERIOD).tolist())
            cell["wc_peak"].extend(peaks.tolist())
            cell["wc_edge"] += int(np.sum((a == 0) | (b == n)))
            a, b, _ = merged_episodes(w_rel[:, i], t0)
            cell["wrel_duration"].extend(((b - a) * PERIOD).tolist())
            above_c = w_c[:, i] > t0
            cell["wrel_no_wc"].extend([bool(not above_c[x:y].any()) for x, y in zip(a, b)])
            cell["wrel_edge"] += int(np.sum((a == 0) | (b == n)))
        out["cells"][f"{t0:.0f}@{s}"] = cell
    return out


# ----------------------------------------------------------------------------- P0-W3


def w3_configurations() -> list[dict]:
    rows = [{"name": "local", "kind": "local"}]
    for rho in (0.5, 0.64, 0.8):
        rows.append({"name": f"mixed({rho})", "kind": "mixed", "rho": rho})
    for label, angle in (("bow_quartering", BOW_QUARTERING), ("broadside", BROADSIDE)):
        for speed in (5.0, 10.0, math.inf):
            name = f"front({'inf' if math.isinf(speed) else f'{speed:g}'} m/s) {label}"
            if math.isinf(speed):
                name = f"exact common {label}"
            rows.append({"name": name, "kind": "front", "angle": angle, "speed": speed, "label": label})
    return rows


def body_weights(units: np.ndarray, conj: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-cable weights (w_load, w_vessel) so that load_i = w_L F_L . d_i - w_A F_A,i . d_i."""
    if conj == "rel":
        return np.full(N_CABLES, PAIR_REDUCED_MASS / M_L), np.full(N_CABLES, PAIR_REDUCED_MASS / M_A)
    return np.full(N_CABLES, DRAG_REDUCED / C_L), np.full(N_CABLES, DRAG_REDUCED / C_A)


def analytic_std(config: dict, units: np.ndarray, positions: np.ndarray, conj: str, intensity: float = 1.0) -> np.ndarray:
    """Closed-form stationary std per cable of W_rel (conj='rel') or W_c (conj='c')."""
    sigma = pinned_stationary_std(scale=intensity)
    s_l, s_a = sigma[0], sigma[1]
    w_l, w_a = body_weights(units, conj)
    a = w_l * s_l
    b = w_a * s_a
    kind = config["kind"]
    if kind == "local":
        return np.sqrt(a**2 + b**2)
    if kind == "mixed":
        rho = config["rho"]
        return np.sqrt(rho * (a - b) ** 2 + (1.0 - rho) * (a**2 + b**2))
    phi = ar1_coefficient()
    delays = front_arrival_delays(positions, config["angle"], config["speed"])
    direction = np.array([math.cos(config["angle"]), math.sin(config["angle"])])
    projection = units @ direction
    lag = np.abs(delays[1:] - delays[0])
    correlation = phi ** lag
    variance = 2.0 * (a**2 + b**2 - 2.0 * a * b * correlation) * projection**2
    return np.sqrt(np.maximum(variance, 0.0))


@dataclass(frozen=True)
class W3Job:
    seed: int
    positions: tuple
    cost: float = 1.0


def w3_forces(config: dict, seed: int, positions: np.ndarray) -> np.ndarray:
    if config["kind"] == "local":
        return stationary_weather_forces(seed, RECORD_SECONDS, direction="local")
    if config["kind"] == "mixed":
        return stationary_weather_forces(seed, RECORD_SECONDS, direction="mixed", rho=config["rho"])
    return lagged_front_weather_forces(seed, RECORD_SECONDS, positions=positions, front_angle=config["angle"], speed=config["speed"])


def w3_worker(job: W3Job) -> dict:
    positions = np.asarray(job.positions, dtype=float)
    _, units = fan_positions()
    out = {"seed": job.seed, "configs": {}}
    for config in w3_configurations():
        forces = w3_forces(config, job.seed, positions)
        w_rel, w_c = conjugate_loads(forces, units)
        out["configs"][config["name"]] = {
            "n": int(w_rel.shape[0]),
            "rel_sum": w_rel.sum(axis=0), "rel_sq": (w_rel**2).sum(axis=0),
            "c_sum": w_c.sum(axis=0), "c_sq": (w_c**2).sum(axis=0),
        }
    return out


def factor(sigma, pretension: float = 1000.0):
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", over="ignore", under="ignore"):
        return np.where(sigma > 0.0, np.exp(-(pretension**2) / (2.0 * np.maximum(sigma, 1e-300) ** 2)), 0.0)


# ----------------------------------------------------------------------------- driver


def _stack(rows, key):
    return np.stack([np.asarray(r[key]) for r in rows])


def run_t8_w1(workers: int) -> dict:
    grid_rows: list[dict] = []
    history = []
    blocks = 0
    while True:
        jobs = [W1Job(seed, "grid") for seed in block_seeds(blocks)]
        grid_rows.extend(run_pool(w1_worker, jobs, workers))
        blocks += 1
        gust = _stack(grid_rows, "gust_counts").sum(axis=0)  # (cables, levels)
        back = _stack(grid_rows, "back_counts").sum(axis=0)
        full = _stack(grid_rows, "full_counts").sum(axis=0)
        u = [smallest_negligible_level(LEVEL_GRID, gust[i], back[i]) for i in range(N_CABLES)]
        u_full = [smallest_negligible_level(LEVEL_GRID, gust[i], full[i]) for i in range(N_CABLES)]
        k = [int(gust[i][np.searchsorted(LEVEL_GRID, u[i])]) if u[i] is not None else 0 for i in range(N_CABLES)]
        k_b = [int(back[i][np.searchsorted(LEVEL_GRID, u[i])]) if u[i] is not None else 0 for i in range(N_CABLES)]
        history.append({"blocks": blocks, "u_i": u, "k_i": k, "k_background_i": k_b})
        print(f"[W1] blocks={blocks} u={u} k={k} kB={k_b}", flush=True)
        if (min(k) >= 200 and sum(k_b) >= 20) or blocks >= MAX_BLOCKS:
            break
    powered = min(k) >= 200 and sum(k_b) >= 20
    seeds = [r["seed"] for r in grid_rows]
    exposure = len(seeds) * RECORD_SECONDS  # per cable
    thresholds = tuple((u[i], u_full[i]) for i in range(N_CABLES))
    maxima_rows = run_pool(w1_worker, [W1Job(seed, "maxima", thresholds) for seed in seeds], workers)

    def pooled(name: str, which: str, levels):
        per_seed_k = np.zeros((len(seeds), N_CABLES))
        per_seed_log = np.zeros((len(seeds), N_CABLES))
        for s, row in enumerate(maxima_rows):
            for i in range(N_CABLES):
                if levels[i] is None:
                    continue
                m = np.asarray(row[f"{name}_{which}"][i])
                per_seed_k[s, i] = m.size
                per_seed_log[s, i] = np.sum(np.log(m / levels[i])) if m.size else 0.0
        k_i = per_seed_k.sum(axis=0)
        l_i = per_seed_log.sum(axis=0)
        k_total = float(k_i.sum())
        alpha = float(k_total / l_i.sum()) if l_i.sum() > 0 else float("nan")
        per_cable = []
        for i in range(N_CABLES):
            a_i = float(k_i[i] / l_i[i]) if l_i[i] > 0 else float("nan")
            half = 1.96 / math.sqrt(k_i[i]) if k_i[i] > 0 else float("nan")
            per_cable.append({"k": int(k_i[i]), "alpha_hat": a_i, "interval": [a_i * (1 - half), a_i * (1 + half)]})
        rng = np.random.default_rng(20260913)
        boots = []
        for _ in range(2000):
            pick = rng.integers(0, len(seeds), len(seeds))
            kk = per_seed_k[pick].sum()
            ll = per_seed_log[pick].sum()
            if ll > 0:
                boots.append(kk / ll)
        half = 1.96 / math.sqrt(k_total) if k_total > 0 else float("nan")
        return {
            "k": int(k_total),
            "alpha_hat": alpha,
            "interval_plan": [alpha * (1 - half), alpha * (1 + half)],
            "seed_bootstrap_95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))] if boots else None,
            "per_cable": per_cable,
        }

    gust_at_u = pooled("gust", "u", u)
    back_at_u = pooled("back", "u", u)
    gust_at_ufull = pooled("gust", "ufull", u_full)
    full_at_ufull = pooled("full", "ufull", u_full)
    # consistency: explicit maxima counts equal the grid counts
    consistent = all(gust_at_u["per_cable"][i]["k"] == k[i] for i in range(N_CABLES)) and all(
        back_at_u["per_cable"][i]["k"] == k_b[i] for i in range(N_CABLES)
    )

    def in_band(x):
        return bool(2.4 <= x <= 3.6) if np.isfinite(x) else False

    background_outside = (not in_band(back_at_u["alpha_hat"])) if np.isfinite(back_at_u["alpha_hat"]) else None
    if not powered:
        verdict = "UNDER-POWERED"
    else:
        verdict = "PASS" if (in_band(gust_at_u["alpha_hat"]) and background_outside) else "FAIL"
    per_cable_verdicts = [
        bool(in_band(gust_at_u["per_cable"][i]["alpha_hat"]) and not in_band(back_at_u["per_cable"][i]["alpha_hat"]))
        for i in range(N_CABLES)
    ]

    # rates at u_i
    rates = []
    for i in range(N_CABLES):
        j = int(np.searchsorted(LEVEL_GRID, u[i]))
        rates.append({"gust_rate_per_s": float(gust[i][j] / exposure), "background_rate_per_s": float(back[i][j] / exposure),
                      "ratio": float(back[i][j] / gust[i][j]) if gust[i][j] else None})
    # curves (pooled over cables) on a coarse subgrid for the record
    sub = np.arange(0, LEVEL_GRID.size, 20)
    curves = {
        "levels_N": LEVEL_GRID[sub],
        "gust_rate_pooled": gust[:, sub].sum(axis=0) / (N_CABLES * exposure),
        "background_rate_pooled": back[:, sub].sum(axis=0) / (N_CABLES * exposure),
        "full_gaussian_rate_pooled": full[:, sub].sum(axis=0) / (N_CABLES * exposure),
    }

    # ---------- construction checks and T8 (block 0 only)
    block0 = [r for r in grid_rows if r["seed"] in set(block_seeds(0))]
    gust_sq = _stack(block0, "gust_sq").sum(axis=0)
    gust_pred = _stack(block0, "gust_sq_predicted").sum(axis=0)
    n0 = sum(r["samples"] for r in block0)
    a0 = GUST_SCALE_FACTOR * pinned_stationary_std()
    analytic_var = gust_second_moments()["isotropic_component_variance"] * a0**2
    per_seed_ratio = np.stack([np.asarray(r["gust_sq"]) / r["samples"] / analytic_var[:, None] for r in block0])  # (seeds, bodies, 2)
    rng = np.random.default_rng(20260913)
    boot = [float(per_seed_ratio[rng.integers(0, len(block0), len(block0))].mean()) for _ in range(2000)]
    w_sum = np.stack([np.asarray(r["w_rel_sum"]) for r in block0]).sum(axis=0)  # (3, cables)
    w_sq = np.stack([np.asarray(r["w_rel_sq"]) for r in block0]).sum(axis=0)
    w_std = np.sqrt(w_sq / n0 - (w_sum / n0) ** 2)
    analytic_wrel = float(PAIR_REDUCED_MASS * math.hypot(constants.LOAD_WEATHER_STD / M_L, constants.VESSEL_WEATHER_STD / M_A))
    realized_ratio = gust_sq / gust_pred
    construction = {
        "background_factor_f_B": gust_background_factor(),
        "gust_fraction_g": 1.0 - gust_background_factor() ** 2,
        "gust_variance_per_component_analytic_N2": {"load": float(analytic_var[0]), "vessel": float(analytic_var[1])},
        "realized_mark_ratio_per_body_component": realized_ratio,
        "realized_mark_ratio_pooled": float(gust_sq.sum() / gust_pred.sum()),
        "realized_mark_check_pass": bool(0.97 <= gust_sq.sum() / gust_pred.sum() <= 1.03),
        "gust_variance_over_analytic_pooled": float(per_seed_ratio.mean()),
        "gust_variance_over_analytic_seed_bootstrap_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "gust_mean_per_component_N": (_stack(block0, "gust_sum").sum(axis=0) / n0),
        "w_rel_std_gust_class_per_cable_N": w_std[0],
        "w_rel_std_gust_class_pooled_N": float(np.sqrt(np.mean(w_std[0] ** 2))),
        "w_rel_std_background_per_cable_N": w_std[1],
        "w_rel_std_full_gaussian_per_cable_N": w_std[2],
        "w_rel_std_analytic_N": analytic_wrel,
        "w_rel_std_without_preservation_analytic_N": analytic_wrel * math.sqrt(1.0 + (1.0 - gust_background_factor() ** 2)),
        "exposure_s": n0 * PERIOD,
    }
    marks_all = [np.concatenate([np.asarray(r["marks"][b]) for r in block0]) for b in range(N_CABLES + 1)]
    pooled_marks = np.concatenate(marks_all)
    k_hill = pooled_marks.size // 10
    t8_primary = hill_tail_index(pooled_marks, k_hill)
    t8 = {
        "marks": int(pooled_marks.size),
        "hill_primary_top10pct": t8_primary,
        "upper_order_count": int(k_hill),
        "hill_interval_1.96_over_sqrt_k": [t8_primary * (1 - 1.96 / math.sqrt(k_hill)), t8_primary * (1 + 1.96 / math.sqrt(k_hill))],
        "known_scale_mle": float(pooled_marks.size / np.sum(np.log(pooled_marks))),
        "per_body_hill_top10pct": [hill_tail_index(m, max(m.size // 10, 1)) for m in marks_all],
        "per_body_marks": [int(m.size) for m in marks_all],
        "pass_band": [2.7, 3.3],
        "verdict": "PASS" if 2.7 <= t8_primary <= 3.3 else "FAIL",
    }
    m_eff = PAIR_REDUCED_MASS
    e_cos3 = 4.0 / (3.0 * math.pi)
    rv_scale = {
        "E_abs_cos_cubed": e_cos3,
        "factor_E_abs_cos_cubed_pow_1_3": e_cos3 ** (1.0 / 3.0),
        "load_gust_a_i_N": e_cos3 ** (1 / 3) * m_eff / M_L * a0[0],
        "vessel_gust_a_i_N": e_cos3 ** (1 / 3) * m_eff / M_A * a0[1],
        "load_gust_projection_of_a0_N": m_eff / M_L * a0[0],
        "plan": {"load": 760.0, "vessel": 630.0, "load_projection": 1020.0},
        "gust_only_peak_tail_rate_per_cable": "0.1 (a/762)^-3 + 0.1 (a/636)^-3 per s for a above both scales (a single gust's peak |W_rel| contribution, ignoring the background)",
    }
    w1 = {
        "blocks": blocks,
        "seeds": [seeds[0], seeds[-1]],
        "exposure_per_cable_s": exposure,
        "exposure_history": history,
        "powered": powered,
        "u_i_N": u,
        "u_i_over_background_W_rel_std": [x / float(np.mean(w_std[1])) if x is not None else None for x in u],
        "u_i_over_a0_vessel": [x / a0[1] if x is not None else None for x in u],
        "rates_at_u_i": rates,
        "gust_class_at_u_i": gust_at_u,
        "background_at_u_i": back_at_u,
        "background_outside_band": background_outside,
        "grid_and_explicit_counts_agree": consistent,
        "verdict": verdict if t8["verdict"] == "PASS" else "NOT-SCORED (P0-T8 gust clause failed)",
        "per_cable_verdicts_secondary": per_cable_verdicts,
        "secondary_full_variance_null": {
            "u_full_i_N": u_full,
            "gust_class_at_u_full_i": gust_at_ufull,
            "full_gaussian_at_u_full_i": full_at_ufull,
        },
        "rate_curves": curves,
        "branch": None,
    }
    if w1["verdict"] == "PASS":
        w1["branch"] = "RV programme launches (Phase 3 after the Phase 2 gate; Phase 4 gust half)"
    elif w1["verdict"] in ("FAIL", "UNDER-POWERED"):
        w1["branch"] = ("NO-LAUNCH of the RV programme: Thm 7', Cor. 7.1', Prop. 9' withdrawn; Phase 3 not run; Phase 4 runs the "
                        "Gaussian-only six-cell design {local, mixed(0.5), mixed(0.8), front(10) bq, front(5) bq, exact common}; the rest continues")
    return {"P0-T8_gust_clause": t8, "P0-W1": w1, "gust_class_construction": construction, "effective_RV_scale": rv_scale}


def run_w2(workers: int) -> dict:
    rows = run_pool(w2_worker, [W2Job(seed) for seed in block_seeds(0)], workers)
    exposure = sum(r["samples"] for r in rows) * PERIOD  # per cable
    planned = 20 * 600.0 * N_CABLES
    cells = {}
    for t0, s in w2_cells():
        key = f"{t0:.0f}@{s}"
        merged = {name: [] for name in ("wc_duration", "wc_peak", "wrel_duration", "wrel_no_wc")}
        edges = {"wc_edge": 0, "wrel_edge": 0}
        for r in rows:
            for name in merged:
                merged[name].extend(r["cells"][key][name])
            for name in edges:
                edges[name] += r["cells"][key][name]
        entry = {"T0_N": t0, "intensity": s, "graded_grid": s in W2_INTENSITIES}
        for series, dur_key, edge_key in (("W_c", "wc_duration", "wc_edge"), ("W_rel", "wrel_duration", "wrel_edge")):
            d = np.asarray(merged[dur_key])
            count = int(d.size)
            pooled_exposure = N_CABLES * exposure
            q = (lambda p: float(np.quantile(d, p)) if count else None)
            entry[series] = {
                "episodes": count,
                "rate_per_cable_s": count / pooled_exposure,
                "rate_poisson_95": poisson_interval(count, pooled_exposure),
                "duration_median_s": q(0.5), "duration_p90_s": q(0.9), "duration_p95_s": q(0.95),
                "edge_truncated": edges[edge_key],
                "expected_over_phase2_planned_exposure": count / pooled_exposure * planned,
                "under_20_flag": count < 20,
            }
        d = np.asarray(merged["wc_duration"])
        p = np.asarray(merged["wc_peak"])
        labels = [classify_wc_episode(float(x), float(y), t0) for x, y in zip(d, p)]
        counts = {c: labels.count(c) for c in ("R1", "transitional", "R2")}
        total = max(len(labels), 1)
        no_wc = np.asarray(merged["wrel_no_wc"], dtype=bool)
        entry["H4prime_forecast"] = {
            "W_c_episode_class_counts": counts,
            "W_c_episode_class_shares": {c: (counts[c] / total if labels else None) for c in counts},
            "class_N_proxy_share_of_W_rel_episodes_without_W_c_exceedance": float(no_wc.mean()) if no_wc.size else None,
            "class_N_proxy_count": int(no_wc.sum()),
        }
        entry["reduced_model_snap_rate_per_class"] = {"N": None, "R1": None, "R2": None, "transitional": None,
                                                     "note": "to be filled by the reduced-model agent; P0-W2 admissibility needs it"}
        cells[key] = entry
    return {"exposure_per_cable_s": exposure, "seeds": [block_seeds(0)[0], block_seeds(0)[-1]], "phase2_planned_exposure_cable_s": planned,
            "cells": cells, "admissibility_verdict": "NOT SCORED HERE (needs the reduced-model snap rates per class)"}


def run_w3(workers: int) -> dict:
    positions, units = fan_positions()
    aligned, _ = fan_positions(hull_aligned=True)
    rows = run_pool(w3_worker, [W3Job(seed, tuple(map(tuple, positions))) for seed in block_seeds(0)], workers)
    configs = w3_configurations()
    table = {}
    checks = []
    for config in configs:
        name = config["name"]
        n = sum(r["configs"][name]["n"] for r in rows)
        entry = {"config": {k: (None if (isinstance(v, float) and math.isinf(v)) else v) for k, v in config.items()}}
        for conj, key in (("rel", "rel"), ("c", "c")):
            s = np.stack([np.asarray(r["configs"][name][f"{key}_sum"]) for r in rows]).sum(axis=0)
            q = np.stack([np.asarray(r["configs"][name][f"{key}_sq"]) for r in rows]).sum(axis=0)
            numeric = np.sqrt(q / n - (s / n) ** 2)
            ana = analytic_std(config, units, positions, conj)
            label = "W_rel" if conj == "rel" else "W_c"
            ok = bool(np.all(np.abs(numeric - ana) <= np.where(ana < 100.0, 5.0, 0.05 * ana)))
            checks.append(ok)
            entry[label] = {
                "analytic_std_N_intensity_1": ana,
                "numeric_std_N_intensity_1": numeric,
                "numeric_within_tolerance": ok,
                "factor_at_1kN": {f"{s_:g}": factor(ana * s_) for s_ in (1.0, 0.5, 0.35)},
                "max_factor_at_1kN": {f"{s_:g}": float(np.max(factor(ana * s_))) for s_ in (1.0, 0.5, 0.35)},
            }
            if config["kind"] == "front":
                entry["hull_aligned_diagnostic_" + label + "_std_N"] = analytic_std(config, units, aligned, conj)
        if config["kind"] == "front":
            entry["delays_samples"] = front_arrival_delays(positions, config["angle"], config["speed"])
        table[name] = entry
    scored = ["local", "mixed(0.5)", "front(10 m/s) bow_quartering", "front(5 m/s) bow_quartering"]
    controls = ["exact common bow_quartering", "mixed(0.8)"]
    verdicts = {}
    for s_ in ("1", "0.5", "0.35"):
        rows_v = {}
        for name in scored:
            rows_v[name] = {
                "max_factor_W_rel": table[name]["W_rel"]["max_factor_at_1kN"][s_],
                "holds_W_rel": table[name]["W_rel"]["max_factor_at_1kN"][s_] > 0.05,
                "max_factor_W_c_secondary": table[name]["W_c"]["max_factor_at_1kN"][s_],
                "holds_W_c_secondary": table[name]["W_c"]["max_factor_at_1kN"][s_] > 0.05,
            }
        rows_c = {name: {"max_factor_W_rel": table[name]["W_rel"]["max_factor_at_1kN"][s_],
                         "max_factor_W_c": table[name]["W_c"]["max_factor_at_1kN"][s_], "exempt": True} for name in controls}
        failing = [name for name in scored if not rows_v[name]["holds_W_rel"]]
        verdicts[s_] = {"scored": rows_v, "controls": rows_c,
                        "cells_to_replace_before_launch": failing,
                        "verdict": "PASS" if not failing else "FAIL (cells to replace: " + ", ".join(failing) + ")"}
    # spans and Davenport
    spans = {}
    f_c = 1.0 / (2.0 * math.pi * constants.WEATHER_TIME_CONSTANT)
    for label, angle in (("bow_quartering", BOW_QUARTERING), ("broadside", BROADSIDE)):
        d = np.array([math.cos(angle), math.sin(angle)])
        along = positions @ d
        along_aligned = aligned @ d
        span = float(along.max() - along.min())
        spans[label] = {
            "span_along_front_m": span,
            "span_hull_aligned_diagnostic_m": float(along_aligned.max() - along_aligned.min()),
            "max_lag_s": {"10": span / 10.0, "5": span / 5.0},
            "davenport_coherence": {"10": math.exp(-DAVENPORT_C * f_c * span / 10.0), "5": math.exp(-DAVENPORT_C * f_c * span / 5.0)},
        }
    # lag table for a unit-projection cable
    phi = ar1_coefficient()
    a = PAIR_REDUCED_MASS / M_L * constants.LOAD_WEATHER_STD
    b = PAIR_REDUCED_MASS / M_A * constants.VESSEL_WEATHER_STD
    lag_table = {}
    for lag in (0.9, 1.8, 3.6):
        sd = math.sqrt(2.0 * (a * a + b * b - 2 * a * b * phi ** round(lag / PERIOD)))
        lag_table[f"{lag}"] = {"std_N": sd, "factor_at_1kN": float(factor(sd))}
    local_sd = math.hypot(a, b)
    lag_table["local"] = {"std_N": local_sd, "factor_at_1kN": float(factor(local_sd))}
    lag_table["exact_common_unit_projection"] = {"std_N": math.sqrt(2.0) * abs(a - b), "factor_at_1kN": float(factor(math.sqrt(2.0) * abs(a - b)))}
    lag_table["isotropic_common_per_component"] = {"std_N": abs(a - b), "factor_at_1kN": float(factor(abs(a - b)))}
    return {
        "positions_m": positions, "hull_aligned_positions_diagnostic_m": aligned, "cable_units": units,
        "table": table, "numeric_consistency_all": bool(all(checks)),
        "verdict_by_intensity": verdicts,
        "binding_note": "Phase 4 runs at the Phase 2 in-domain reference intensity (0.5, or 0.35 under the psi fallback); the verdict at that intensity binds",
        "spans_and_davenport": spans, "davenport_C": DAVENPORT_C, "davenport_f_Hz": f_c,
        "lag_table": lag_table,
        "exposure_per_config_s": sum(r["configs"]["local"]["n"] for r in rows) * PERIOD,
    }


# ----------------------------------------------------------------------------- entry points


def declare() -> None:
    payload = json_bytes(DECLARATIONS)
    if DECLARATIONS_PATH.exists():
        if DECLARATIONS_PATH.read_bytes() != payload:
            raise SystemExit("declarations already exist and differ; write a dated addendum instead of editing")
        print("declarations unchanged:", sha256_bytes(payload))
        return
    if RESULTS_PATH.exists():
        raise SystemExit("results exist; declarations cannot be (re)written after results")
    digest = write_bytes(DECLARATIONS_PATH, payload)
    print("declared", relative(DECLARATIONS_PATH), digest)


def run(workers: int) -> None:
    payload = json_bytes(DECLARATIONS)
    if not DECLARATIONS_PATH.exists() or DECLARATIONS_PATH.read_bytes() != payload:
        raise SystemExit("declarations missing or differ from the module; run 'declare' first (never edit after results)")
    started = wallclock.perf_counter()
    results = {
        "schema_version": 1,
        "declarations": relative(DECLARATIONS_PATH),
        "declarations_sha256": sha256_bytes(DECLARATIONS_PATH.read_bytes()),
        "source": source_state(),
    }
    results.update(run_t8_w1(workers))
    print("[W1] done", flush=True)
    results["P0-W2"] = run_w2(workers)
    print("[W2] done", flush=True)
    results["P0-W3"] = run_w3(workers)
    print("[W3] done", flush=True)
    print(f"wall seconds {wallclock.perf_counter() - started:.1f}", flush=True)
    digest = write_bytes(RESULTS_PATH, json_bytes(results))
    print("results", relative(RESULTS_PATH), digest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("declare", "run"))
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.command == "declare":
        declare()
    else:
        run(min(arguments.workers, 4))


if __name__ == "__main__":
    main()
