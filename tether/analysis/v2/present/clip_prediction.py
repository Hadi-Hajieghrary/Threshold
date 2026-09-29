"""Clip C4 ``prediction`` -- per-line forecast vs fleet rollout on one recorded slack interval.

Presentation/STORYBOARD.md section C4.  Run:  python3 -m tether.analysis.v2.present.clip_prediction
(add ``--replay`` to recompute the replay cache, ``--stills`` to write inspection stills only).

Records read (read-only) and the keys used
-----------------------------------------
records/v2/phase5/p5_t2prime_ticks.npz        per oracle slack tick: seed, cable, tick, time, interval, bounce, onset,
                                              state_index, censored_H1, label_H1, crossed_H1, closing_speed_H1,
                                              coupled_H1, forecast_linearized_H1, forecast_rollout_H1,
                                              forecast_rollout_H2, forecast_p_cross_H1, forecast_p_cross_H2
records/v2/phase5/p5_t2prime_results.json     v_b; counts.calibration_ticks_H1; ticks_npz_sha256;
                                              inputs_sha256.{code_fleet_rollout, code_p5_pretest, impact_table_fan};
                                              reports.{linearized_H1, rollout_H1}.all.{slope, slope_lo, slope_hi, auroc};
                                              reports.{linearized_H1, rollout_H1}.coincidence.{top_bin_ticks,
                                              top_bin_misses, top_bin_misses_coupled}; gate.{slope, verdict};
                                              post_hoc_diagnostics.forecasts.{linearized_H1, rollout_H1}.{h_zero,
                                              h_one, h_zero_events, h_one_nonevents};
                                              post_hoc_diagnostics.forecasts.linearized_H1.clip_sensitivity["0.01"].slope;
                                              recalibration_branch_iii.recal_rollout_H1_test.test_recalibrated (caveat only)
records/v2/phase5/p5_t2prime_declarations.json  horizons.primary; stress_threshold.T_b_s; monte_carlo.N (and, as quoted
                                              rules: reported_not_gating.coupled, label, top_bin_coincidence)
records/v2/phase6/p6_t0_results.json          critical_speeds (cross-checked equal to the P5-T2' v_b)
records/phase5/cache/phase5_missions.pkl      seed 5008 "truth": state (10 ms plant state [q; v]), event_time,
                                              elongation, rate (1 ms cable log), marks (cable, t_up, T_peak, v_return);
                                              every mission's truth.state / state_time (hull-overlap caveat only)
records/phase5/cache/phase5_missions_arms.pkl seed 5008: outputs['P'].time (the 10 Hz tick grid), monitor_truth.e_true /
                                              edot_true and the 1 ms series (the v1 oracle precursor, phase5_analysis._oracle_output)
records/phase5/impact_table_fan.json          via tether.monitor.hazard.critical_speeds (v_b) and impact_law(...).peak (the
                                              4.5 kN threshold line is the fan impact law's peak at v_b), and its sha256
Paper/Sections/Section_IV_Prediction.tex      quoted, not parsed: the clip-0.01 per-line slope 0.88 [0.79, 1.02] (l. 96-99),
                                              which this module recomputes (below) and asserts equal to two decimals

Selection rule (STORYBOARD C4), re-derived and asserted here
-----------------------------------------------------------
Scored ticks = ~bounce & ~censored_H1 (9227).  Per-line top-bin false alarms = scored ticks with
forecast_linearized_H1 >= 0.9 and label_H1 = 0 (25); coupled ones have coupled_H1 (24); of those, the
ones at 70 <= time <= 110 s (18).  The mission holding the most of these 18 is shown: seed 5008 (6, all
on cable 4 at 75.6-76.1 s, one slack interval; next best 5), and the clip shows EVERY tick of that
slack interval (71.1-77.2 s, 62 ticks) for both models, not only the false alarms.

What is asserted before any frame is drawn (Manifest.check)
----------------------------------------------------------
* the selection above, and at the six ticks: linearized h = 1, rollout h = 0, label 0, coupled;
* v_b = critical_speeds(4500, 1000, impact_table_fan) equals the results file's v_b and P6-T0's;
* the fan impact law's peak at v_b,4 is the declared 4.5 kN stress threshold (the tension panel's line);
* the labels of all 62 ticks re-derived from the 1 ms log (first true upcrossing of e_4 = 0 in
  [t, t + 1 s] closing faster than v_b,4), crossing flags and closing speeds equal to the record; every
  label-1 tick refers to the same crossing (77.288 s), so the per-line h = 0 count is a count of ticks
  for one snap, and is worded so on screen;
* the recorded hulls overlap (separating-axis test on plant-size hulls) at every 10 ms sample of the
  shown window; the first overlap in the mission and the 76 s / 45 s counts over all missions are
  computed for the caveat;
* the coupled flags re-derived from truth.marks (another cable, T_peak > 1 kN, t_up in [t, t + 1 s]);
* the interval is a maximal run of oracle slack ticks (e_true <= 0) and all ticks lie in it;
* the 10 ms state's chord geometry reproduces the 1 ms elongation and rate of cable 4;
* REPLAY 1 (per-line): tether.monitor.linearized.fleet_hazard_linearized on the v1 oracle precursor
  (tether.campaign.phase5_analysis._oracle_output) reproduces forecast_linearized_H1 exactly at all
  439 slack ticks of seed 5008; the sample paths shown are redrawn from the same generator state;
* REPLAY 2 (rollout): tether.monitor.fleet_rollout.rollout, set up exactly as
  tether.campaign.v2.p5_pretest.rollout_job (controlled headings, 1 ms step, 2 s, N = 2048, generator
  SeedSequence([5008, 17, 4]) advanced through every earlier evaluated tick), reproduces
  forecast_rollout_H1, forecast_rollout_H2, forecast_p_cross_H1 and forecast_p_cross_H2 exactly at
  all 62 ticks; the code hashes equal the results file's inputs_sha256.
* REPLAY 3 (per-line slope at forecast clip 0.01, post hoc): the declared statistic
  (tether.monitor.metrics.calibration_report, cluster = slack interval, 2000 replicates, rng
  p5_pretest._boot_rng('linearized_H1')) first reproduces the record's declared-clip slope and interval
  exactly, then at clip 0.01 gives the cluster-bootstrap interval the paper quotes (0.88 [0.79, 1.02]).
Layout, asserted while drawing: every text and legend inside the frame with a 16 px gutter (first and
last frame of each shot, each hold, slide and the end); no body outline under the fleet view's ticker
or note (every frame, separating-axis test in pixels); nothing the gap panel ever draws under its
in-panel note or clipped by its axis range; no tension above the threshold under the threshold label;
captions <= 2 lines, >= 4 s and >= tokens / 2.5 s (a caption continuing on a pause counts both); population
slide and title card <= 5 tokens/s (SLIDE_WPS).  Captions are complete sentences; h and v_b are defined in words
(caption s2) before any other caption uses them.
Replays are cached in Presentation/cache/prediction_replay.npz (its sha256 is a manifest source). A
cached replay is re-checked against the record on every run (h_H1, h_H2, p_cross at all 62 ticks, code
hashes), and two ticks, chosen by rule (the first per-line false alarm and the first per-line h = 0
tick whose horizon holds the snap), are re-simulated from scratch and must equal the cache exactly,
bands included.
"""
from __future__ import annotations

import tempfile
import os
import argparse
import json
import math
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tether.analysis.v2.present import common as C

NAME = "prediction"
R_TICKS = "records/v2/phase5/p5_t2prime_ticks.npz"
R_RESULTS = "records/v2/phase5/p5_t2prime_results.json"
R_DECL = "records/v2/phase5/p5_t2prime_declarations.json"
R_P6T0 = "records/v2/phase6/p6_t0_results.json"
R_MISSIONS = "records/phase5/cache/phase5_missions.pkl"
R_ARMS = "records/phase5/cache/phase5_missions_arms.pkl"
R_IMPACT = "records/phase5/impact_table_fan.json"
CODE_ROLLOUT = "tether/monitor/fleet_rollout.py"
CODE_PRETEST = "tether/campaign/v2/p5_pretest.py"
CODE_LINEAR = "tether/monitor/linearized.py"
PAPER_IV = "Paper/Sections/Section_IV_Prediction.tex"
CACHE_FILE = C.CACHE / "prediction_replay.npz"
CLIP = C.CLIPS / "prediction.mp4"
STILLS = Path(os.environ.get("TETHER_DEV_DIR", Path(tempfile.gettempdir()) / "tether_dev")) / "present" / "prediction"

H = 1.0                  # primary horizon (declarations horizons.primary)
TOP = 0.9                # top reliability bin h >= 0.9
WINDOW = (70.0, 110.0)   # after the squall, before the deceleration (paper Sec. IV)
SHOWN = (70.60, 77.84)   # mission clock of the first and last animated frame (the episode)
COUPLING_N = 1000.0      # declarations reported_not_gating.coupled: T_peak > 1 kN
N_SAMPLES = 2048
PROJ_DT = 0.01           # grid on which the projections are stored (s)
WORKERS = 2              # shared 16-core box: at most two worker processes


def _rel(path) -> Path:
    return C.REPO / path


# =====================================================================================  data + checks

def load_ticks() -> dict:
    z = np.load(_rel(R_TICKS))
    return {k: z[k] for k in z.files}


def select(d: dict, man: C.Manifest) -> dict:
    """Evaluate the storyboard's selection rule on the tick record and assert its outcome."""
    keep = ~d["bounce"] & ~d["censored_H1"]
    lin, rol, y, cp, t = (d["forecast_linearized_H1"], d["forecast_rollout_H1"], d["label_H1"],
                          d["coupled_H1"], d["time"])
    fa = keep & (lin >= TOP) & ~y
    cfa = fa & cp
    win = cfa & (t >= WINDOW[0]) & (t <= WINDOW[1])
    seeds, counts = np.unique(d["seed"][win], return_counts=True)
    order = np.argsort(-counts, kind="stable")
    seed = int(seeds[order[0]])
    man.check("selection: 9227 scored ticks, 25 per-line top-bin false alarms, 24 coupled, 18 at 70-110 s",
              int(keep.sum()) == 9227 and int(fa.sum()) == 25 and int(cfa.sum()) == 24 and int(win.sum()) == 18,
              f"scored {int(keep.sum())}, FA {int(fa.sum())}, coupled FA {int(cfa.sum())}, at 70-110 s {int(win.sum())}")
    man.check("selection: seed 5008 holds the most (6), uniquely",
              seed == 5008 and counts[order[0]] == 6 and counts[order[1]] < 6,
              "per seed: " + ", ".join(f"{s}:{c}" for s, c in zip(seeds.tolist(), counts.tolist())))
    rows = np.flatnonzero(win & (d["seed"] == seed))
    cable = int(np.unique(d["cable"][rows])[0])
    interval = int(np.unique(d["interval"][rows])[0])
    man.check("selection: the six are cable 4, one slack interval, 75.6-76.1 s, lin h = 1, rollout h = 0, label 0, coupled",
              cable == 4 and np.unique(d["cable"][rows]).size == 1 and np.unique(d["interval"][rows]).size == 1
              and np.allclose(t[rows], np.arange(75.6, 76.15, 0.1)) and np.all(lin[rows] == 1.0)
              and np.all(rol[rows] == 0.0) and not y[rows].any() and cp[rows].all(),
              f"times {np.round(t[rows], 2).tolist()}, lin {lin[rows].tolist()}, rollout {rol[rows].tolist()}")
    iv = np.flatnonzero(d["interval"] == interval)
    iv = iv[np.argsort(t[iv], kind="stable")]
    man.check("interval: 62 ticks 71.1-77.2 s, not a bounce, none censored",
              iv.size == 62 and abs(t[iv[0]] - 71.1) < 1e-9 and abs(t[iv[-1]] - 77.2) < 1e-9
              and not d["bounce"][iv].any() and not d["censored_H1"][iv].any(),
              f"{iv.size} ticks, {t[iv[0]]:.1f}-{t[iv[-1]]:.1f} s")
    # the storyboard's population statement derived from the same ticks
    top = keep & (lin >= TOP)
    top_r = keep & (rol >= TOP)
    return {"seed": seed, "cable": cable, "interval": interval, "fa_rows": rows, "rows": iv,
            "n_fa_window": int(win.sum()), "n_cfa": int(cfa.sum()), "n_fa": int(fa.sum()),
            "per_seed": dict(zip(seeds.tolist(), counts.tolist())),
            "top_coupled": int((top & cp).sum()), "top_uncoupled": int((top & ~cp).sum()),
            "fa_uncoupled": int((fa & ~cp).sum()),
            "rol_split": [[int((top_r & ~y & cp).sum()), int((top_r & cp).sum())], [int((top_r & ~y & ~cp).sum()), int((top_r & ~cp).sum())]]}


def _separated(P: np.ndarray, Q: np.ndarray) -> bool:
    """Separating-axis test for two convex polygons (True if they do not overlap)."""
    for poly in (P, Q):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            a, b = P @ axis, Q @ axis
            if a.max() < b.min() or b.max() < a.min():
                return True
    return False


def overlapping_hulls(row) -> int:
    """Number of vessel pairs whose plant-size hulls (common.hull, convex) overlap in a state row."""
    _, vp = C.unpack_state(row)
    polys = [C.body_polygon(v, C.hull()) for v in vp]
    return sum(not _separated(polys[i], polys[j]) for i in range(len(polys)) for j in range(i + 1, len(polys)))


def load_mission(seed: int) -> dict:
    with open(_rel(R_MISSIONS), "rb") as fh:
        missions = pickle.load(fh)
    m = next(x for x in missions if x["seed"] == seed)
    tr = m["truth"]
    out = {"state": np.asarray(tr.state), "state_time": np.asarray(tr.state_time),
           "series_time": np.asarray(tr.event_time), "elongation": np.asarray(tr.elongation),
           "rate": np.asarray(tr.rate),
           "marks": np.array([(r.cable, r.t_up, r.T_peak, r.v_return) for r in tr.marks], float)}
    assert np.array_equal(out["elongation"], np.asarray(m["elongation"]))
    # hull overlap over all missions (the plant models no contact): for the fleet-view caveat
    reach76 = over76 = over45 = 0
    for x in missions:
        S, T = np.asarray(x["truth"].state), np.asarray(x["truth"].state_time)
        over45 += overlapping_hulls(S[C.nearest(T, 45.0)]) > 0
        if T[-1] >= 76.0:
            reach76 += 1
            over76 += overlapping_hulls(S[C.nearest(T, 76.0)]) > 0
    out["overlap_all"] = {"missions": len(missions), "reach76": reach76, "over76": over76, "over45": over45}
    del missions
    return out


def load_arm(seed: int) -> dict:
    with open(_rel(R_ARMS), "rb") as fh:
        arms = pickle.load(fh)
    arm = next(a for a in arms if a["seed"] == seed)
    del arms
    return arm


def verify_truth(d: dict, sel: dict, mis: dict, arm: dict, man: C.Manifest) -> dict:
    """Labels, coupling, slack-ness and geometry of the selected interval, from the logs."""
    from tether.campaign.v2 import p5_pretest as P
    from tether.monitor.outcomes import first_upcrossing, true_upcrossings

    res = json.loads(_rel(R_RESULTS).read_text())
    p6 = json.loads(_rel(R_P6T0).read_text())
    decl = json.loads(_rel(R_DECL).read_text())
    v_b = np.asarray(P.critical_speeds())
    man.check("v_b = critical_speeds(4500 N, 1 kN, fan impact table) equals the P5-T2' record and P6-T0",
              np.array_equal(v_b, np.asarray(res["v_b"])) and np.allclose(v_b, p6["critical_speeds"], rtol=0, atol=1e-15)
              and decl["stress_threshold"]["T_b_s"] == 4500.0 and decl["horizons"]["primary"] == H,
              f"v_b = {np.round(v_b, 4).tolist()} m/s")
    cable, rows = sel["cable"], sel["rows"]
    from tether.monitor.hazard import impact_law
    law_peak = float(impact_law(1000.0, cable, table_path=_rel(R_IMPACT)).peak(float(v_b[cable])))
    man.check("the fan impact law's peak at v_b,4 is the declared 4.5 kN stress threshold (on-screen threshold line)",
              abs(law_peak - decl["stress_threshold"]["T_b_s"]) < 1e-6, f"impact_law(1 kN, cable 4).peak(v_b,4) = {law_peak:.6f} N")
    t = d["time"][rows]
    st, e, r = mis["series_time"], mis["elongation"][:, cable], mis["rate"][:, cable]
    ups = true_upcrossings(st, e, r)
    hit, when, speed, _ = first_upcrossing(ups, t, t + H)
    label = hit & (speed > v_b[cable])
    man.check("labels of all 62 ticks re-derived from the 1 ms log (first upcrossing in [t, t+1 s] faster than v_b,4)",
              np.array_equal(label, d["label_H1"][rows]) and np.array_equal(hit, d["crossed_H1"][rows])
              and np.allclose(speed, d["closing_speed_H1"][rows], equal_nan=True, rtol=0, atol=1e-12),
              f"label 1 at {int(label.sum())} ticks ({t[label].min():.1f}-{t[label].max():.1f} s), 0 at {int((~label).sum())}")
    man.check("every label-1 tick of the interval refers to one and the same crossing of cable 4 (one snap)",
              np.unique(when[label]).size == 1 and np.all(hit[label]),
              f"crossing at {np.unique(when[label]).tolist()} s for the {int(label.sum())} label-1 ticks")
    fa = np.isin(rows, sel["fa_rows"])
    man.check("the six false-alarm ticks have no upcrossing of e_4 in their window (label 0 from the log)",
              not hit[fa].any(), "first upcrossing after the onset: %.4f s" % ups.time[np.searchsorted(ups.time, t[0])])
    first_up = float(ups.time[np.searchsorted(ups.time, t[0])])
    first_speed = float(ups.speed[np.searchsorted(ups.time, t[0])])
    marks = mis["marks"]
    other = np.sort(marks[(marks[:, 0] != cable) & (marks[:, 2] > COUPLING_N), 1])
    lo = np.searchsorted(other, t - 1e-12, side="left")
    coupled = (lo < other.size) & (other[np.minimum(lo, other.size - 1)] <= t + H)
    man.check("coupled flags of all 62 ticks re-derived from truth.marks (other cable, T_peak > 1 kN, t_up in [t, t+1 s])",
              np.array_equal(coupled, d["coupled_H1"][rows]), f"coupled at {int(coupled.sum())} ticks")
    # the tick grid and the oracle slack rule (e_true <= 0) around the interval
    grid = np.asarray(arm["outputs"]["P"].time, float)
    e_true = np.asarray(arm["monitor_truth"].e_true, float)[:, cable]
    k = d["tick"][rows]
    man.check("interval = maximal run of oracle slack ticks (e_true <= 0) of cable 4",
              np.allclose(grid[k], t) and np.all(e_true[k] <= 0) and e_true[k[0] - 1] > 0 and e_true[k[-1] + 1] > 0
              and np.all(np.diff(k) == 1), f"e_true before/after: {e_true[k[0]-1]:.2e} / {e_true[k[-1]+1]:.2e} m")
    # the other cables' first re-engagement after the common slack onset, with the record's peaks
    firsts = []
    for j in range(5):
        mj = marks[(marks[:, 0] == j) & (marks[:, 1] > t[0])]
        firsts.append(mj[np.argmin(mj[:, 1])])
    firsts = np.array(firsts)
    m4 = firsts[cable]
    man.check("cable 4's tracker mark equals the 1 ms upcrossing (time, closing speed)",
              abs(m4[1] - first_up) < 1e-9 and abs(m4[3] - first_speed) < 1e-9,
              f"t_up {m4[1]:.4f} s, v_return {m4[3]:.4f} m/s, T_peak {m4[2]:.1f} N")
    # which other cables re-engage inside each false alarm's horizon (the coupled definition)
    fa_t = t[fa]
    inside = {int(j): [bool(np.any((marks[:, 0] == j) & (marks[:, 2] > COUPLING_N) & (marks[:, 1] >= tt) & (marks[:, 1] <= tt + H)))
                       for tt in fa_t] for j in range(5) if j != cable}
    man.check("cables 1, 2, 0 and 3 each re-engage (> 1 kN) inside the horizon of every false-alarm tick",
              all(all(v) for v in inside.values()), json.dumps(inside))
    # the plant's tension law on the 1 ms log reproduces each first mark's T_peak (the ticker / tension labels)
    peaks = []
    for j in range(5):
        seg = (st >= firsts[j][1]) & (st <= firsts[j][1] + 0.25)
        peaks.append(float(C.tension(mis["elongation"][seg, j], mis["rate"][seg, j], True).max()))
    man.check("plant tension law on the 1 ms log reproduces the recorded T_peak of each cable's first mark (1 N)",
              all(abs(peaks[j] - firsts[j][2]) < 1.0 for j in range(5)),
              "log " + ", ".join(f"{p:.1f}" for p in peaks) + " N vs marks " + ", ".join(f"{f[2]:.1f}" for f in firsts) + " N")
    # 10 ms state geometry reproduces the 1 ms log for cable 4; payload/stern terms (post hoc)
    geo = C.mission_geometry()
    terms = {}
    for tt in (76.10, 76.11, 76.20):
        terms[tt] = chord_terms(mis["state"][int(round(tt / 0.01))], geo, cable)
        i1 = int(round(tt / 0.001))
        assert abs(terms[tt][0] - e[i1]) < 1e-6 and abs(terms[tt][3] - r[i1]) < 1e-6, (tt, terms[tt], e[i1], r[i1])
    man.check("10 ms state chord geometry reproduces the 1 ms elongation and rate of cable 4 (1e-6)", True,
              "at 76.10, 76.11, 76.20 s")
    quiet = (st >= 76.10) & (st <= 76.20)
    no_other = all(float(C.tension(mis["elongation"][quiet, j], mis["rate"][quiet, j], True).max()) == 0.0 for j in (0, 3, 4))
    d_load = terms[76.20][2] - terms[76.11][2]
    d_stern = terms[76.20][1] - terms[76.11][1]
    man.check("post hoc: 76.11-76.20 s only cables 1, 2 carry tension; cable 4's payload term drops, its stern term does not",
              no_other and d_load < -1.5 and d_stern > 0.0,
              f"payload term {terms[76.11][2]:+.3f} -> {terms[76.20][2]:+.3f} m/s, stern term {terms[76.11][1]:+.3f} -> {terms[76.20][1]:+.3f} m/s")
    # the next cable-4 re-engagement after the snap (outside the interval; drawn on the traces, not labelled)
    later = marks[(marks[:, 0] == cable) & (marks[:, 1] > first_up + 1e-9) & (marks[:, 1] <= SHOWN[1])]
    # hull overlap (no contact is modelled): every 10 ms sample of the shown window, and where it starts
    ks = np.arange(int(math.floor(SHOWN[0] / 0.01 + 1e-9)), int(math.floor(SHOWN[1] / 0.01 + 1e-9)) + 1)
    pairs = np.array([overlapping_hulls(mis["state"][k]) for k in ks])
    first_overlap = next(k for k in range(len(mis["state"])) if overlapping_hulls(mis["state"][k]) > 0) * 0.01
    ov = mis["overlap_all"]
    man.check("recorded hulls overlap at every 10 ms sample of the shown window (the on-screen 'hulls may overlap' note)",
              bool(np.all(pairs > 0)),
              f"{ks.size} samples {SHOWN[0]:.2f}-{SHOWN[1]:.2f} s, overlapping pairs {pairs.min()}-{pairs.max()}; first overlap in "
              f"seed {sel['seed']} at {first_overlap:.2f} s; all missions: {ov['over76']} of the {ov['reach76']} that reach 76 s overlap "
              f"at 76 s, {ov['over45']} of {ov['missions']} at 45 s")
    return {"v_b": v_b, "label": label, "crossed": hit, "cross_time": when, "coupled": coupled,
            "first_up": first_up, "first_speed": first_speed, "firsts": firsts, "terms": terms,
            "rate_drop": (float(r[int(round(76.10 / 0.001))]), float(r[int(round(76.20 / 0.001))])),
            "later4": later, "first_overlap": first_overlap, "overlap_pairs": (int(pairs.min()), int(pairs.max())),
            "law_peak": law_peak, "results": res}


def chord_terms(row, geo, i: int):
    """(e, stern term u.v_stern, payload term -u.v_attach, total rate) of cable i from a state row."""
    q, v = np.asarray(row[:18]).reshape(6, 3), np.asarray(row[18:]).reshape(6, 3)
    th = q[0, 2]
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    ra = R @ geo.load_offsets[i]
    a, va = q[0, :2] + ra, v[0, :2] + v[0, 2] * np.array([-ra[1], ra[0]])
    tv = q[1 + i, 2]
    Rv = np.array([[math.cos(tv), -math.sin(tv)], [math.sin(tv), math.cos(tv)]])
    rs = Rv @ geo.vessel_offsets[i]
    s, vs = q[1 + i, :2] + rs, v[1 + i, :2] + v[1 + i, 2] * np.array([-rs[1], rs[0]])
    dvec = s - a
    L = float(np.linalg.norm(dvec))
    u = dvec / L
    return L - C.K.CABLE_REST_LENGTH, float(u @ vs), float(-u @ va), float(u @ (vs - va))


# =====================================================================================  replays

def _code_hashes() -> dict:
    return {"fleet_rollout": C.sha256(_rel(CODE_ROLLOUT)), "p5_pretest": C.sha256(_rel(CODE_PRETEST)),
            "linearized": C.sha256(_rel(CODE_LINEAR)), "ticks": C.sha256(_rel(R_TICKS))}


def replay_linearized(arm: dict, d: dict, sel: dict, v_b: np.ndarray, man: C.Manifest) -> dict:
    """Per-line model: the v1 oracle precursor + fleet_hazard_linearized, and its sample paths."""
    from tether.campaign.phase5_analysis import _oracle_output
    from tether.monitor.hazard import hazard_generator, precursor_state
    from tether.monitor.linearized import fleet_hazard_linearized, linearized_hazard_mc, linearized_paths

    seed, cable = sel["seed"], sel["cable"]
    out = _oracle_output(arm)
    hz = fleet_hazard_linearized(out, v_b, seed, n_samples=N_SAMPLES, horizon=H)
    m = d["seed"] == seed
    rec, rep = d["forecast_linearized_H1"][m], hz[d["tick"][m], d["cable"][m]]
    man.check("REPLAY per-line: fleet_hazard_linearized on the oracle precursor reproduces forecast_linearized_H1 "
              "exactly at all slack ticks of seed 5008", np.array_equal(rec, rep), f"{int(m.sum())} ticks, max |diff| {np.max(np.abs(rec - rep)):.1e}")
    ticks = d["tick"][sel["rows"]]
    want = set(ticks.tolist())
    slack = np.asarray(out.slack, bool)
    rng = hazard_generator(seed, cable)
    grid = np.arange(1, int(round(H / 0.005)) + 1) * 0.005          # the model's own 5 ms grid
    keep_t = np.arange(0, int(round(H / PROJ_DT)) + 1) * PROJ_DT
    pct = np.full((ticks.size, 3, keep_t.size), np.nan)
    first = np.full((ticks.size, 3), np.nan)
    hrep = np.full(ticks.size, np.nan)
    pos = {k: i for i, k in enumerate(ticks.tolist())}
    for k in np.flatnonzero(slack[:, cable]):
        mean, cov = precursor_state(out.e_hat[k, cable], out.edot_hat[k, cable], out.sigma[k, cable],
                                    out.a_hat[k, cable], out.sigma_a[k, cable])
        if k in want:
            state = rng.bit_generator.state
            h = linearized_hazard_mc(mean, cov, H, float(v_b[cable]), N_SAMPLES, rng)
            again = np.random.default_rng()
            again.bit_generator.state = state
            s = again.multivariate_normal(mean, cov, size=N_SAMPLES, method="cholesky")
            e0, v0, a = s[:, 0], s[:, 1], s[:, 2]
            ee, vv = linearized_paths(e0, v0, a + (C.K.VESSEL_LINEAR_DRAG / C.K.VESSEL_MASS) * v0, grid)
            prev = np.concatenate([e0[:, None], ee[:, :-1]], axis=1)
            cr = (prev <= 0) & (ee > 0)
            has = cr.any(axis=1)
            j = np.argmax(cr, axis=1)
            eb = np.where(j > 0, ee[np.arange(N_SAMPLES), j - 1], e0)
            vb_ = np.where(j > 0, vv[np.arange(N_SAMPLES), j - 1], v0)
            fr = -eb / (ee[np.arange(N_SAMPLES), j] - eb)
            spd = vb_ + fr * (vv[np.arange(N_SAMPLES), j] - vb_)
            tcr = (j + fr) * 0.005
            h2 = float(np.sum(has & (spd > v_b[cable])) / N_SAMPLES)
            assert h2 == h, (k, h, h2)
            full = np.concatenate([e0[:, None], ee], axis=1)             # includes t = 0
            tt = np.concatenate([[0.0], grid])
            idx = np.searchsorted(tt, keep_t - 1e-12)
            i = pos[k]
            pct[i] = np.percentile(full[:, idx], [5, 50, 95], axis=0)
            hrep[i] = h
            if has.any():
                first[i] = [float(np.mean(has)), float(np.median(tcr[has])), float(np.median(spd[has]))]
            else:
                first[i] = [0.0, np.nan, np.nan]
        else:
            linearized_hazard_mc(mean, cov, H, float(v_b[cable]), N_SAMPLES, rng)
    man.check("REPLAY per-line: the redrawn sample paths give the record's h at all 62 ticks",
              np.array_equal(hrep, d["forecast_linearized_H1"][sel["rows"]]), "")
    return {"lin_h": hrep, "lin_pct": pct, "lin_first": first, "proj_t": keep_t}


def _rollout_chunk(args):
    """Worker: advance the (seed, cable) generator through every evaluated tick in time order and
    run the rollout (as p5_pretest.rollout_job) at the assigned ticks only."""
    seed, cable, times, states, assigned, v_b = args
    from tether.campaign.mission import MissionSpec, mission_geometry, mission_schedules, raised_cosine
    from tether.campaign.v2 import p5_pretest as P
    from tether.monitor.fleet_rollout import RolloutModel, WeatherFuture, common_ar1_future, rollout, thrust_on_steps
    from tether.monitor.hazard import hazard_generator
    from tether.physics import constants
    from tether.physics.weather import ar1_coefficient

    spec = MissionSpec()
    schedules = mission_schedules(spec)
    model = RolloutModel(mission_geometry(spec), step=P.ROLLOUT_STEP, headings=P.PRIMARY_MODE)
    background, _, squall, _ = P.mission_forcing(seed)
    std = P.body_std()
    phi = ar1_coefficient()
    rng = hazard_generator(seed, cable)
    horizon = max(P.HORIZONS)
    steps = int(round(horizon / P.ROLLOUT_STEP))
    per = int(round(PROJ_DT / P.ROLLOUT_STEP))
    out = {}
    for row, (t, state) in enumerate(zip(times, states)):
        k0 = int(round(t / constants.WEATHER_PERIOD))
        current = background[k0, 0] / std[0]
        common = common_ar1_future(rng, P.N_ROLLOUT, current, P.FUTURE_SAMPLES, phi, spec.weather_distribution)
        if row not in assigned:
            continue
        sample_times = (k0 + np.arange(P.FUTURE_SAMPLES + 1)) * constants.WEATHER_PERIOD
        scale = raised_cosine(sample_times / spec.weather_ramp_duration)[:, None] * std[None, :]
        weather = WeatherFuture(k0, common, scale, P._extend(squall, k0, P.FUTURE_SAMPLES + 1))
        thrust = thrust_on_steps(schedules.surge, t, steps, P.ROLLOUT_STEP)
        res = rollout(model, state, t, horizon, cable, weather, thrust, heading_reference=schedules.heading,
                      record_steps=np.arange(0, steps + 1, per))
        e4 = res.record_elongation[:, :, cable]
        e_all = res.record_elongation
        # median first upcrossing (10 ms grid) of every cable across the samples, within 2 s
        firsts = np.full(5, np.nan)
        for j in range(5):
            up = (e_all[:-1, :, j] <= 0) & (e_all[1:, :, j] > 0)
            has = up.any(axis=0)
            if has.mean() >= 0.5:      # median over the samples that re-engage, sample time of first e > 0
                firsts[j] = t + float(np.median((np.argmax(up[:, has], axis=0) + 1) * PROJ_DT))
        hit1 = res.crossed & (res.crossing_time <= 1.0 + 1e-9)
        out[row] = {"h1": res.hazard(v_b, 1.0), "h2": res.hazard(v_b, 2.0),
                    "p1": float(np.mean(hit1)), "p2": float(np.mean(res.crossed & (res.crossing_time <= 2.0 + 1e-9))),
                    "pct": np.percentile(e4, [5, 50, 95], axis=1),
                    "cross": [float(np.mean(res.crossed)), float(np.nanmedian(res.crossing_time)) if res.crossed.any() else np.nan,
                              float(np.nanmedian(res.closing_speed)) if res.crossed.any() else np.nan],
                    "firsts": firsts}
    return out


def _rollout_inputs(mis: dict, d: dict, sel: dict):
    """Every evaluated (non-bounce) tick of the watched cable in time order -- the generator is
    advanced through all of them, as in p5_pretest.rollout_job -- and the positions of the shown ticks."""
    ev = np.flatnonzero((d["seed"] == sel["seed"]) & (d["cable"] == sel["cable"]) & ~d["bounce"])
    ev = ev[np.argsort(d["time"][ev], kind="stable")]
    times = d["time"][ev].copy()
    states = mis["state"][d["state_index"][ev]].copy()
    target = [int(np.flatnonzero(ev == r)[0]) for r in sel["rows"]]
    return times, states, target


def _run_chunks(sel: dict, times, states, chunks, v_b_cable: float) -> dict:
    args = [(sel["seed"], sel["cable"], times, states, ch, v_b_cable) for ch in chunks]
    merged = {}
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(chunks))) as pool:
        for part in pool.map(_rollout_chunk, args):
            merged.update(part)
    return merged


def replay_rollout(mis: dict, d: dict, sel: dict, v_b: np.ndarray) -> dict:
    cable = sel["cable"]
    times, states, target = _rollout_inputs(mis, d, sel)
    merged = _run_chunks(sel, times, states, [set(target[0::2]), set(target[1::2])], float(v_b[cable]))
    n = len(target)
    steps2 = int(round(2.0 / PROJ_DT)) + 1
    rol = {"rol_h1": np.zeros(n), "rol_h2": np.zeros(n), "rol_p1": np.zeros(n), "rol_p2": np.zeros(n),
           "rol_pct": np.zeros((n, 3, steps2)), "rol_cross": np.zeros((n, 3)), "rol_firsts": np.zeros((n, 5))}
    for i, row in enumerate(target):
        o = merged[row]
        rol["rol_h1"][i], rol["rol_h2"][i], rol["rol_p1"][i], rol["rol_p2"][i] = o["h1"], o["h2"], o["p1"], o["p2"]
        rol["rol_pct"][i] = o["pct"]
        rol["rol_cross"][i] = o["cross"]
        rol["rol_firsts"][i] = o["firsts"]
    return rol


def replays(d, sel, mis, arm, v_b, man: C.Manifest, force: bool = False) -> dict:
    hashes = _code_hashes()
    res = json.loads(_rel(R_RESULTS).read_text())
    ins = res["inputs_sha256"]
    man.check("code of the record = code replayed (fleet_rollout.py, p5_pretest.py, fan impact table sha256)",
              ins["code_fleet_rollout"] == hashes["fleet_rollout"] and ins["code_p5_pretest"] == hashes["p5_pretest"]
              and ins["impact_table_fan"] == C.sha256(_rel(R_IMPACT)) and res["ticks_npz_sha256"] == hashes["ticks"],
              "sha256 equal to results.inputs_sha256 and ticks_npz_sha256")
    cached = None
    if CACHE_FILE.exists() and not force:
        z = np.load(CACHE_FILE, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta.get("hashes") == hashes and meta.get("rows") == sel["rows"].tolist():
            cached = {k: z[k] for k in z.files if k != "meta"}
            print("replay cache: loaded", CACHE_FILE.relative_to(C.REPO))
    if cached is None:
        lin = replay_linearized(arm, d, sel, v_b, man)
        rol = replay_rollout(mis, d, sel, v_b)
        cached = {**lin, **rol}
        meta = {"hashes": hashes, "rows": sel["rows"].tolist(), "seed": sel["seed"], "cable": sel["cable"],
                "what": "per-line (linearized) and fleet-rollout replays of P5-T2' at every tick of the selected interval"}
        np.savez_compressed(CACHE_FILE, meta=np.array(json.dumps(meta)), **cached)
        print("replay cache: wrote", CACHE_FILE.relative_to(C.REPO))
    else:
        # the per-line replay is cheap: re-run it so its reproduction is re-asserted every time
        lin = replay_linearized(arm, d, sel, v_b, man)
        assert np.array_equal(lin["lin_h"], cached["lin_h"]) and np.allclose(lin["lin_pct"], cached["lin_pct"], equal_nan=True)
        # the rollout is not cheap: re-simulate two ticks chosen by rule and require the cache exactly
        rows = sel["rows"]
        y, lin_h = d["label_H1"][rows], d["forecast_linearized_H1"][rows]
        picks = [int(np.flatnonzero(np.isin(rows, sel["fa_rows"]))[0]), int(np.flatnonzero(y & (lin_h == 0.0))[0])]
        times, states, target = _rollout_inputs(mis, d, sel)
        fresh = _run_chunks(sel, times, states, [{target[i]} for i in picks], float(v_b[sel["cable"]]))
        same = all(fresh[target[i]]["h1"] == cached["rol_h1"][i] and fresh[target[i]]["h2"] == cached["rol_h2"][i]
                   and fresh[target[i]]["p1"] == cached["rol_p1"][i] and fresh[target[i]]["p2"] == cached["rol_p2"][i]
                   and np.array_equal(fresh[target[i]]["pct"], cached["rol_pct"][i])
                   and np.array_equal(np.asarray(fresh[target[i]]["cross"]), cached["rol_cross"][i], equal_nan=True)
                   and np.array_equal(fresh[target[i]]["firsts"], cached["rol_firsts"][i], equal_nan=True) for i in picks)
        man.check("REPLAY rollout cache: two ticks (first per-line false alarm, first per-line h = 0 tick holding the snap) "
                  "re-simulated from scratch equal the cache exactly, projection bands included", same,
                  f"ticks {[round(float(d['time'][rows][i]), 1) for i in picks]} s; h_H1 {[fresh[target[i]]['h1'] for i in picks]}")
    rows = sel["rows"]
    man.check("REPLAY rollout: forecast_rollout_H1 reproduced exactly at all 62 ticks",
              np.array_equal(cached["rol_h1"], d["forecast_rollout_H1"][rows]),
              f"max |diff| {np.max(np.abs(cached['rol_h1'] - d['forecast_rollout_H1'][rows])):.1e}")
    man.check("REPLAY rollout: forecast_rollout_H2, forecast_p_cross_H1, forecast_p_cross_H2 reproduced exactly",
              np.array_equal(cached["rol_h2"], d["forecast_rollout_H2"][rows])
              and np.array_equal(cached["rol_p1"], d["forecast_p_cross_H1"][rows])
              and np.array_equal(cached["rol_p2"], d["forecast_p_cross_H2"][rows]), "")
    man.check("REPLAY per-line: cached h equals forecast_linearized_H1 at all 62 ticks",
              np.array_equal(cached["lin_h"], d["forecast_linearized_H1"][rows]), "")
    man.source(CACHE_FILE)                                     # sha256 of the replay cache the frames are drawn from
    return cached


PAPER_CLIP01 = (0.88, 0.79, 1.02)     # Paper Sec. IV l. 96-99 (and closeout, fifth round: 0.884 [0.79, 1.02])


def clip_sensitivity(d: dict, man: C.Manifest) -> dict:
    """Per-line slope with its cluster-bootstrap interval at forecast clip 0.01 (post hoc), by the
    declared statistic; the results file stores only a tick-level interval at that clip."""
    from tether.campaign.v2 import p5_pretest as P
    from tether.monitor.metrics import calibration_report

    res = json.loads(_rel(R_RESULTS).read_text())
    keep = ~d["bounce"] & ~d["censored_H1"]
    f, y, c = d["forecast_linearized_H1"][keep], d["label_H1"][keep], d["interval"][keep]

    def fit(clip):
        r = calibration_report(f, y, c, n_bins=10, n_boot=P.N_BOOT, clip=clip, confidence=0.95, rng=P._boot_rng("linearized_H1"))
        return float(r.slope), float(r.slope_lo), float(r.slope_hi)

    declared = fit(P.FORECAST_CLIP)
    L = res["reports"]["linearized_H1"]["all"]
    man.check("REPLAY clip: the declared statistic reproduces the record's per-line slope and cluster-bootstrap interval exactly",
              declared == (L["slope"], L["slope_lo"], L["slope_hi"]), f"{declared} at clip 1/(2N) = {P.FORECAST_CLIP:g}")
    import re
    tex = " ".join(_rel(PAPER_IV).read_text().split())
    quoted = re.search(r"\$(\d\.\d\d)\$ at the coarsest, \$0\.01\$, where its cluster-bootstrap interval, "
                       r"\$\[(\d\.\d\d), (\d\.\d\d)\]\$, reaches one", tex)
    man.check("the paper (Sec. IV) quotes the clip-0.01 per-line slope as 0.88 [0.79, 1.02]",
              quoted is not None and tuple(float(g) for g in quoted.groups()) == PAPER_CLIP01, PAPER_IV)
    at01 = fit(0.01)
    tick = res["post_hoc_diagnostics"]["forecasts"]["linearized_H1"]["clip_sensitivity"]["0.01"]["slope"]
    man.check("REPLAY clip: at clip 0.01 the slope equals the record's post-hoc value and the interval rounds to the paper's",
              at01[0] == tick and tuple(round(v, 2) for v in at01) == PAPER_CLIP01,
              f"{at01[0]:.4f} [{at01[1]:.4f}, {at01[2]:.4f}] vs paper {PAPER_CLIP01}")
    return {"clip01": at01}


def prepare(man: C.Manifest, force: bool = False) -> dict:
    """Everything the clip draws, with every reproduction and consistency check asserted."""
    for p in (R_TICKS, R_RESULTS, R_DECL, R_P6T0, R_MISSIONS, R_ARMS, R_IMPACT, CODE_ROLLOUT, CODE_PRETEST, CODE_LINEAR, PAPER_IV):
        man.source(p)
    d = load_ticks()
    sel = select(d, man)
    mis = load_mission(sel["seed"])
    arm = load_arm(sel["seed"])
    tru = verify_truth(d, sel, mis, arm, man)
    rep = replays(d, sel, mis, arm, tru["v_b"], man, force=force)
    del arm
    clip = clip_sensitivity(d, man)
    return {"d": d, "sel": sel, "mis": mis, "tru": tru, "rep": rep, "clip": clip}


# =====================================================================================  numbers

class Register:
    """Registers every on-screen number with the manifest and keeps its value for the drawing code."""

    def __init__(self, man: C.Manifest):
        self.man, self.v = man, {}

    def __call__(self, key, label, value, unit="", source="", note=""):
        self.man.value(label, value, unit, source, note)
        self.v[key] = value
        return value

    def __getitem__(self, key):
        return self.v[key]


def numbers(x: dict, man: C.Manifest) -> Register:
    """Every number the clip puts on screen, read from a record or from a checked replay."""
    from tether.campaign.mission import MissionSpec, squall_envelope

    d, sel, mis, tru, rep = x["d"], x["sel"], x["mis"], x["tru"], x["rep"]
    res = tru["results"]
    decl = json.loads(_rel(R_DECL).read_text())
    rows, cable = sel["rows"], sel["cable"]
    t = d["time"][rows]
    lin, rol, y = d["forecast_linearized_H1"][rows], d["forecast_rollout_H1"][rows], d["label_H1"][rows]
    fa = np.isin(rows, sel["fa_rows"])
    n = Register(man)
    # ---- set-up and declared constants
    spec = MissionSpec()
    n("T0", "pretension T0", spec.pretension / 1e3, "kN", "tether/campaign/mission.py MissionSpec.pretension (as the declarations' data.missions)")
    n("missions", "missions in the P5-T2' set", len(range(5001, 5041)), "", f"{R_DECL} data.missions (seeds 5001-5040)")
    n("H", "forecast horizon H", decl["horizons"]["primary"], "s", f"{R_DECL} horizons.primary")
    n("tick_hz", "monitor tick rate", 10, "Hz", f"{R_DECL} data.ticks (10 Hz grid of the v1 Phase 5 arms); tick spacing in {R_TICKS} time")
    man.check("tick spacing 0.1 s in the interval", np.allclose(np.diff(t), 0.1), "")
    n("top", "top reliability bin", TOP, "", f"{R_DECL} reported_not_gating.top_bin_coincidence (h in [0.9, 1.0])")
    n("N_roll", "rollout futures per tick", decl["monte_carlo"]["N"], "", f"{R_DECL} monte_carlo.N")
    n("couple_kN", "coupling threshold on another cable's T_peak", COUPLING_N / 1e3, "kN", f"{R_DECL} reported_not_gating.coupled")
    tt = np.arange(60.0, 80.0, 0.01)
    squall_end = float(tt[np.flatnonzero(squall_envelope(spec, tt) <= 0.0)[0]])
    n("squall_end", "squall ends (envelope reaches 0)", round(squall_end, 2), "s", "tether.campaign.mission.squall_envelope(MissionSpec())")
    n("window", "window of the selection rule", list(WINDOW), "s", "STORYBOARD C4 / paper Sec. IV (after the squall, before the deceleration)")
    n("state_dt", "fleet view: plant state sample period", round(float(np.median(np.diff(mis["state_time"]))) * 1e3, 3), "ms",
      f"{R_MISSIONS} truth.state_time")
    n("log_dt", "cable log sample period", round(float(np.median(np.diff(mis["series_time"]))) * 1e3, 3), "ms", f"{R_MISSIONS} truth.event_time")
    man.check("state every 10 ms, cable log every 1 ms", n["state_dt"] == 10.0 and n["log_dt"] == 1.0, "")
    n("scale_bar", "fleet view scale bar", 5.0, "m", "tether/analysis/v2/present/common.py draw_fleet (drawing scale, not data)")
    n("tick_times", "tick times shown (every tick of the interval; each tick's 1 s horizon is shaded in turn)",
      [round(float(v), 1) for v in t], "s", f"{R_TICKS} time over the interval")
    wl = rep["lin_pct"][:, 2, -1] - rep["lin_pct"][:, 0, -1]
    wr = rep["rol_pct"][:, 2, rep["proj_t"].size - 1] - rep["rol_pct"][:, 0, rep["proj_t"].size - 1]
    early = t < 76.2 - 1e-9
    n("band", "projection bands: 5th-95th percentile of the sampled paths", [5, 95], "%",
      f"replay:{CACHE_FILE.relative_to(C.REPO)} lin_pct / rol_pct",
      f"display choice; the median is the dashed line. Band width at +1 s: per-line <= {wl[early].max() * 100:.1f} cm and rollout <= "
      f"{wr[early].max() * 100:.1f} cm at the ticks before 76.2 s, rollout <= {wr.max() * 100:.1f} cm at every tick, per-line up to "
      f"{wl.max():.2f} m after 76.2 s: most bands are thinner than the dashed line (on screen: 'band often thinner than the line')")
    n.v["band_w"] = (float(wl[early].max()), float(wr.max()), float(wl.max()))
    n("conf", "interval level of the slopes", 95, "%", f"{R_DECL} gate.statistic (confidence = 0.95, cluster bootstrap)")
    # ---- selection
    n("seed", "seed shown", sel["seed"], "", f"{R_TICKS} seed (selection rule)")
    n("cable", "watched cable", cable, "", f"{R_TICKS} cable")
    n("n_win", "coupled per-line top-bin false alarms at 70-110 s", sel["n_fa_window"], "ticks",
      f"{R_TICKS} forecast_linearized_H1>=0.9 & ~label_H1 & coupled_H1 & ~bounce & ~censored_H1 & 70<=time<=110", "computed")
    n("n_sel", "of those in seed 5008", sel["per_seed"][sel["seed"]], "ticks", f"{R_TICKS} computed")
    n("n_next", "next-best seed's count", sorted(sel["per_seed"].values())[-2], "ticks", f"{R_TICKS} computed")
    n("fa_t", "false-alarm ticks, first and last", [round(float(t[fa][0]), 1), round(float(t[fa][-1]), 1)], "s", f"{R_TICKS} time")
    n("n_ticks", "ticks in the slack interval", int(rows.size), "", f"{R_TICKS} interval == {sel['interval']}")
    onsets = [float(d["onset"][(d["seed"] == sel["seed"]) & (d["cable"] == j) & (d["time"] > 71.0) & (d["time"] < 71.3)][0]) for j in range(5)]
    man.check("all five cables' slack onsets lie in [71.0, 71.2] s", min(onsets) >= 71.0 and max(onsets) <= 71.2,
              ", ".join(f"{o:.3f}" for o in onsets))
    n("onsets", "slack onsets of the five cables (shown as '71 s')", [round(o, 3) for o in onsets], "s", f"{R_TICKS} onset")
    # ---- the episode
    n("v_b", "v_b of cable 4", float(tru["v_b"][cable]), "m/s", f"{R_RESULTS} v_b[4] (= critical_speeds(4500, 1000, fan table))", "shown as 0.47")
    n("T_b", "stress threshold behind v_b (tension panel line: the fan impact law's peak at v_b)", decl["stress_threshold"]["T_b_s"] / 1e3,
      "kN", f"{R_DECL} stress_threshold.T_b_s; = tether.monitor.hazard.impact_law(1000 N, cable 4, {R_IMPACT}).peak(v_b,4) "
      f"= {tru['law_peak']:.3f} N (asserted)", "v_b = Z^-1(T_b) (tether/monitor/hazard.py critical_speed)")
    edot = mis["rate"][np.round(t[fa] / 0.001).astype(int), cable]
    n("close", "cable 4 closing speed at the false-alarm ticks (min, max)", [float(edot.min()), float(edot.max())],
      "m/s", f"{R_MISSIONS} seed 5008 truth.rate[:, 4] at the ticks", "shown as 2.0-3.3")
    n("h_fa", "per-line h / rollout h at the six false-alarm ticks", [sorted(set(lin[fa].tolist())), sorted(set(rol[fa].tolist()))], "",
      f"{R_TICKS} forecast_linearized_H1 / forecast_rollout_H1 (both replayed exactly)")
    f = tru["firsts"]
    n("snaps", "first re-engagement since the slack onset per cable: t_up, T_peak",
      {int(j): [float(f[j, 1]), float(f[j, 2]) / 1e3] for j in range(5)}, "s, kN", f"{R_MISSIONS} seed 5008 truth.marks (cable, t_up, T_peak)",
      "t_up shown to 1 ms, T_peak to 0.1 kN")
    r0, r1 = tru["rate_drop"]
    n("drop", "cable 4 closing speed at 76.10 s and 76.20 s", [r0, r1], "m/s", f"{R_MISSIONS} truth.rate[:, 4] at 76.100 / 76.200 s", "shown as 3.3 -> 1.4")
    man.value("payload term of cable 4's closing speed, 76.11 -> 76.20 s (not shown as a number)",
              [tru["terms"][76.11][2], tru["terms"][76.20][2]], "m/s", f"{R_MISSIONS} truth.state rows 7611 / 7620",
              "derived, post hoc: -u.v_attach with u from the payload attachment to vessel 4's stern (negative reduces closing "
              "speed); the stern term rises +4.08 -> +4.30; only cables 1 and 2 carry tension in 76.10-76.20 s. Supports the caption's "
              "'the snaps jerk the payload toward vessel 4'")
    n("up4", "cable 4 re-engages: time, closing speed", [tru["first_up"], tru["first_speed"]], "s, m/s",
      f"{R_MISSIONS} 1 ms upcrossing of truth.elongation[:, 4] = truth.marks t_up/v_return = {R_TICKS} closing_speed_H1",
      "shown as 77.29 s (77.288 in the ticker), 1.96 m/s")
    miss = (lin < 0.1) & y
    n("lin_fa_iv", "per-line top-bin false alarms in this interval", int(((lin >= TOP) & ~y).sum()), "ticks", f"{R_TICKS} over the interval")
    n("lin_miss_iv", "per-line forecast h = 0 at ticks whose 1 s horizon held cable 4's snap (label 1): a count of TICKS, one snap",
      int(miss.sum()), "ticks", f"{R_TICKS} forecast_linearized_H1 == 0 & label_H1 over the interval",
      f"at {np.round(t[miss], 1).tolist()} s; every label-1 tick of the interval refers to the one crossing at "
      f"{tru['first_up']:.3f} s (asserted), so on screen: 'forecast h = 0 at 4 ticks although the snap came within the "
      f"next second'")
    man.check("per-line misses in the interval are exactly h = 0", np.all(lin[miss] == 0.0), "")
    err = float(np.max(np.abs(y - rol)))
    man.check("rollout max |y - h| in the interval < 0.002", err < 0.002, f"{err:.4f}")
    n("rol_err", "rollout bound on |y - h| in this interval", 0.002, "", f"{R_TICKS} forecast_rollout_H1, label_H1 over the interval",
      f"max |y - h| = {err:.4f}, asserted < 0.002")
    # ---- population (read from the results file, never retyped)
    L, R = res["reports"]["linearized_H1"], res["reports"]["rollout_H1"]
    ph = res["post_hoc_diagnostics"]["forecasts"]
    n("N", "scored ticks, H = 1 s", res["counts"]["calibration_ticks_H1"], "ticks", f"{R_RESULTS} counts.calibration_ticks_H1")
    n("lin_slope", "per-line recalibration slope [95% cluster bootstrap]", [L["all"]["slope"], L["all"]["slope_lo"], L["all"]["slope_hi"]], "",
      f"{R_RESULTS} reports.linearized_H1.all.slope/slope_lo/slope_hi")
    n("rol_slope", "rollout recalibration slope [95% cluster bootstrap]", [R["all"]["slope"], R["all"]["slope_lo"], R["all"]["slope_hi"]], "",
      f"{R_RESULTS} reports.rollout_H1.all.slope/slope_lo/slope_hi (= gate.slope)")
    man.check("gate statistic = rollout_H1 slope; verdict FAIL (interval excludes 1)",
              res["gate"]["slope"] == R["all"]["slope"] and res["gate"]["verdict"] == "FAIL" and R["all"]["slope_lo"] > 1.0, "")
    n("auroc", "AUROC per-line / rollout", [L["all"]["auroc"], R["all"]["auroc"]], "", f"{R_RESULTS} reports.*.all.auroc",
      "rollout 0.99999 shown as 1.000, as the paper's table")
    Lc, Rc = L["coincidence"], R["coincidence"]
    n("top_lin", "per-line top-bin false alarms / top-bin ticks / coupled", [Lc["top_bin_misses"], Lc["top_bin_ticks"], Lc["top_bin_misses_coupled"]],
      "ticks", f"{R_RESULTS} reports.linearized_H1.coincidence.top_bin_misses/top_bin_ticks/top_bin_misses_coupled",
      "'misses' in the record's key = non-events forecast >= 0.9, i.e. false alarms")
    n("top_rol", "rollout top-bin false alarms / top-bin ticks / coupled", [Rc["top_bin_misses"], Rc["top_bin_ticks"], Rc["top_bin_misses_coupled"]],
      "ticks", f"{R_RESULTS} reports.rollout_H1.coincidence.top_bin_misses/top_bin_ticks/top_bin_misses_coupled")
    man.check("tick record reproduces the results' top-bin counts",
              sel["n_fa"] == Lc["top_bin_misses"] and sel["n_cfa"] == Lc["top_bin_misses_coupled"]
              and sel["top_coupled"] + sel["top_uncoupled"] == Lc["top_bin_ticks"], "")
    # why the table says 'no DANGEROUS snap within 1 s': a false alarm (label 0) may hold a slower re-engagement
    keep_all = ~d["bounce"] & ~d["censored_H1"]
    fa_all = keep_all & (d["forecast_linearized_H1"] >= TOP) & ~d["label_H1"]
    slow = fa_all & d["crossed_H1"]
    vb_row = np.asarray(tru["v_b"])[d["cable"][slow].astype(int)]
    man.check("per-line top-bin false alarms holding a re-engagement within 1 s: every one closes slower than its v_b",
              bool(np.all(d["closing_speed_H1"][slow] < vb_row)) and not d["crossed_H1"][sel["fa_rows"]].any(),
              f"{int(slow.sum())} of {int(fa_all.sum())}; closing speeds "
              f"{sorted(set(np.round(d['closing_speed_H1'][slow], 3).tolist()))} m/s against v_b "
              f"{sorted(set(np.round(vb_row, 3).tolist()))} m/s; none among the six shown")
    man.value("per-line top-bin false alarms holding a slower (below v_b) re-engagement within 1 s (not shown as a number)",
              [int(slow.sum()), int(fa_all.sum())], "ticks", f"{R_TICKS} crossed_H1 & label_H1 = 0 over the top-bin false alarms",
              "the reason the table row reads 'no dangerous snap within 1 s' rather than 'no snap'")
    n.v["slow_fa"] = (int(slow.sum()), int(fa_all.sum()),
                      sorted(set(np.round(d["closing_speed_H1"][slow], 2).tolist())))
    n("top_split", "per-line top bin: [false alarms, ticks] coupled; uncoupled",
      [[sel["n_cfa"], sel["top_coupled"]], [sel["fa_uncoupled"], sel["top_uncoupled"]]], "ticks",
      f"{R_TICKS} computed: scored ticks with forecast_linearized_H1>=0.9 split by coupled_H1 (paper Sec. IV: 24 of 275, 1 of 164)")
    rs_ = sel["rol_split"]
    man.check("rollout top-bin split from the tick record adds up to the results' counts",
              rs_[0][1] + rs_[1][1] == Rc["top_bin_ticks"] and rs_[0][0] + rs_[1][0] == Rc["top_bin_misses"]
              and rs_[0][0] == Rc["top_bin_misses_coupled"], json.dumps(rs_))
    n("top_split_rol", "rollout top bin: [false alarms, ticks] coupled; uncoupled", rs_, "ticks",
      f"{R_TICKS} computed: scored ticks with forecast_rollout_H1>=0.9 split by coupled_H1 (paper Sec. IV: 'wrong on none of either')")
    n("cw_lin", "per-line forecasts exactly 0 or 1 and wrong (events at 0, non-events at 1)",
      [ph["linearized_H1"]["h_zero_events"], ph["linearized_H1"]["h_one_nonevents"]], "ticks",
      f"{R_RESULTS} post_hoc_diagnostics.forecasts.linearized_H1.h_zero_events/h_one_nonevents", "post hoc (addendum 2026-09-13)")
    n("cw_rol", "rollout forecasts exactly 0 or 1 and wrong", [ph["rollout_H1"]["h_zero_events"], ph["rollout_H1"]["h_one_nonevents"]], "ticks",
      f"{R_RESULTS} post_hoc_diagnostics.forecasts.rollout_H1.h_zero_events/h_one_nonevents", "post hoc")
    n("degenerate", "rollout forecasts exactly 0 or 1", ph["rollout_H1"]["h_zero"] + ph["rollout_H1"]["h_one"], "ticks",
      f"{R_RESULTS} post_hoc_diagnostics.forecasts.rollout_H1.h_zero + h_one", "post hoc")
    c01 = x["clip"]["clip01"]
    n("clip01", "per-line recalibration slope at forecast clip 0.01 [95 % cluster bootstrap] (post hoc)", list(c01), "",
      f"slope: {R_RESULTS} post_hoc_diagnostics.forecasts.linearized_H1.clip_sensitivity.0.01.slope; interval: replay: "
      f"tether.monitor.metrics.calibration_report(forecast_linearized_H1, label_H1, interval, n_boot 2000, clip 0.01, "
      f"rng p5_pretest._boot_rng('linearized_H1')) on the {R_TICKS} scored ticks (the same call reproduces the record's "
      f"declared-clip interval exactly); = {PAPER_IV} l. 96-99 '0.88 [0.79, 1.02]' and reports/v2/campaign_closeout.md fifth round",
      "post hoc; shown as 0.88 [0.79, 1.02]; the record's own clip_sensitivity interval is tick-level (Wald) and is not used")
    return n


def fmt_slope(s) -> str:
    return f"{s[0]:.3f} [{s[1]:.3f}, {s[2]:.3f}]"


SPELLED = {4: "four", 5: "five", 6: "six"}     # small counts written as words in running prose


def texts(n: Register) -> dict:
    """Captions and labels, built from the registered values (nothing retyped).  Every caption is one
    or two complete sentences for a viewer who has not read the paper: what is on screen and what it
    means.  h is defined in words in caption s2a and v_b, with the 'dangerous snap' that h forecasts, in
    caption s2b, before any other caption uses them; 'tick' is defined in caption s3 before s5 uses it."""
    snaps = n["snaps"]
    lin_h, rol_h = n["h_fa"]
    assert lin_h == [1.0] and rol_h == [0.0]
    tl, tr_ = n["top_lin"], n["top_rol"]
    fa0, fa1 = n["fa_t"]
    assert n["n_sel"] == n["lin_fa_iv"] == 6 and len(n["onsets"]) == 5
    assert n["H"] == 1.0                       # caption s9 says 'within the next second' for the 1 s horizon
    six, five = SPELLED[n["n_sel"]], SPELLED[len(n["onsets"])]
    return {
        # title card: the question this clip answers, and what to watch for
        "t_question": "Can one cable's snap be forecast from that cable alone, or does it take the whole fleet?",
        "t_watch": (f"Watch the per-line model, which follows cable {n['cable']} alone, call a snap certain\n"
                    "while other cables pull tight first."),
        "s1": (f"In recorded mission {n['seed']}, all {five} cables go slack at {min(n['onsets']):.0f} s, "
               "one second after the squall ends."),
        "s2a": f"Each model forecasts h, the probability of a dangerous snap within {n['H']:.0f} s.",
        "s2b": (f"A snap is dangerous if cable {n['cable']} pulls tight faster than the severing speed v_b = {n['v_b']:.2f} m/s,\n"
                f"at which its estimated peak reaches {n['T_b']:.1f} kN, where severance is scored."),
        "s3": (f"Every {1 / n['tick_hz']:.1f} s, one tick, both models get the exact state;\n"
               "only the rollout simulates all six bodies, with the known forcing."),
        "s4": (f"From {fa0:.1f} to {fa1:.1f} s, cable {n['cable']}'s gap closes at {n['close'][0]:.1f} to {n['close'][1]:.1f} m/s,\n"
               f"and the per-line model forecasts h = {lin_h[0]:.2f}, a certain dangerous snap."),
        "s5": f"From the same state, the rollout says the opposite: h = {rol_h[0]:.2f} at all {six} ticks.",
        "s6": (f"At {snaps[1][0]:.2f} s, cables 1 and 2 re-engage first, peaking at {snaps[1][1]:.1f} and {snaps[2][1]:.1f} kN;\n"
               "these recording runs cut no cable."),
        "s7": (f"Post hoc, the snaps jerk the payload toward vessel {n['cable']};\n"
               f"cable {n['cable']}'s closing speed falls from {n['drop'][0]:.1f} to {n['drop'][1]:.1f} m/s."),
        "s8": (f"Cable {n['cable']} re-engages only at {n['up4'][0]:.2f} s, at {n['up4'][1]:.2f} m/s:\n"
               f"the {six} certain forecasts were false alarms."),
        "s9": (f"The per-line model gave {n['lin_fa_iv']} false alarms, then forecast h = 0 at {n['lin_miss_iv']} ticks "
               "although the snap came\n"
               f"within the next second. The rollout stayed within {n['rol_err']:.3f} of the truth throughout."),
        # population slide, three captions in turn
        "p1": (f"Over all {n['missions']} missions and {n['N']} scored ticks, the per-line model gave {tl[1]} forecasts\n"
               f"of {n['top']} or more; {tl[0]} were false alarms."),
        "p2": (f"Of those {tl[0]}, {tl[2]} fell on coupled ticks, where another cable re-engaged within {n['H']:.0f} s;\n"
               f"the rollout was wrong on {tr_[0]} of its {tr_[1]}."),
        "p3": ("Both models had the exact state, and the rollout also had the known forcing;\n"
               "neither is a deployable monitor."),
        # population slide: table footnotes (on screen from stage 0) and caveat bullets (stage shown from, highlighted)
        "foot": [
            f"Coupled tick: another cable re-engages (peak above {n['couple_kN']:.0f} kN) within {n['H']:.0f} s.",
            (f"* Declared clip. Post hoc, at clip 0.01 the per-line slope is {n['clip01'][0]:.2f} "
             f"[{n['clip01'][1]:.2f}, {n['clip01'][2]:.2f}]; its {sum(n['cw_lin'])} certain-and-wrong forecasts "
             f"and end-bin failures do not depend on the clip."),
        ],
        "cav": [
            ("Both models had the exact plant state (no estimator).", 1, False),
            ("The rollout also had the known forcing: current weather, squall, thrust and heading schedules, "
             "the background weather's true law.", 1, False),
            ("Neither is a deployable monitor.", 1, False),
            ("The missions come from v1's exploratory continuation; the gate committed no prediction.", 1, False),
            (f"The rollout fails its own pre-declared slope gate, by near-separation: {n['degenerate']} of its "
             f"{n['N']} forecasts are exactly 0 or 1.", 2, True),
        ],
        "end": ("The per-line model is not noisy but incomplete: almost all its confident false alarms\n"
                "come when another cable re-engages."),
        "selection": (f"Selection rule: of the {n['n_win']} coupled per-line top-bin false alarms at {n['window'][0]:.0f}–{n['window'][1]:.0f} s, "
                      f"the mission holding most: seed {n['seed']} ({n['n_sel']}; next best {n['n_next']}). The extreme case, not a "
                      f"typical one. All {n['n_ticks']} ticks of the interval shown."),
        "band_note": "dashed: median; band: 5–95 %,\noften thinner than the line",
        "h_title": (f"Forecast h = P(dangerous snap within {n['H']:.0f} s): cable {n['cable']} re-engages faster than "
                    f"v_b = {n['v_b']:.2f} m/s"),
        "threshold": f"{n['T_b']:.1f} kN stress threshold = the impact law's peak at v_b",
        "miss_note": f"per-line h = 0 at {n['lin_miss_iv']} ticks,\nthough the snap came within {n['H']:.0f} s",
        "subtitle": (f"Seed {n['seed']} · squall-passage mission, fan formation, T0 = {n['T0']:.0f} kN · watched: cable {n['cable']} · "
                     f"{n['tick_hz']} Hz ticks · H = {n['H']:.0f} s"),
        # end card: the key number, one sentence on one line (KEY_FS)
        "key": (f"In all {n['missions']} missions, the per-line model had {tl[0]} false alarms in {tl[1]} near-certain "
                f"forecasts ({tl[2]} coupled); the rollout had {tr_[0]} in {tr_[1]}."),
    }


# =====================================================================================  rendering

X_RANGE = (70.5, 78.25)          # the last tick (77.2 s) keeps its whole 1 s horizon on the axis
GAP_YLIM = (-4.5, 1.5)           # every drawn projection lies in [-4.22, 0.59] m (asserted); the horizon label sits above
MISS_Y = 0.27                    # height of the h = 0 note and its comb on the forecast panel
FLEET_MARGIN = 3.5               # m around the fleet's extent; keeps every body clear of the ticker and the note (asserted per frame)
CAPTION_Y = 0.082
GREY_Y1 = "#d9dde3"          # truth y = 1 column
PALE_RED = "#f6d5d2"         # per-line false-alarm column
TENSION_COLORS = {0: "#9ca3af", 1: "#1f2328", 2: "#4b5563", 3: "#b8bec7", 4: C.ACCENT}
FOOTER = ("records: p5_t2prime_ticks.npz · phase5_missions.pkl (seed 5008: 10 ms state, 1 ms log, marks) · "
          "projections: replays reproducing the record's h at all 62 ticks (cache/prediction_replay.npz)")
FOOTER_POP = ("records/v2/phase5/p5_t2prime_results.json (reports, post_hoc_diagnostics, gate) · p5_t2prime_ticks.npz "
              "(coupled split; clip-0.01 interval by the declared statistic) · Paper Sec. IV")


def shots(tx: dict) -> list:
    """(kind, sim start, sim end, speed or hold seconds, speed label, caption).  run: sim a -> b at the speed;
    hold: the frame at sim time a, paused.  The same simulation instants as before the captions were rewritten
    as sentences; for reading time the drift before the false alarms plays slower (x1/5, x1/6 instead of x1/4)
    and a caption may continue on a pause that follows its run (s2b at 74.10 s, s3 at 75.40 s, s4 at 76.10 s,
    s6 at 76.25 s).  The x1/5 run 72.10 -> 74.10 s is split at 73.30 s between captions s2a and s2b (same speed,
    same instants)."""
    return [
        ("run", SHOWN[0], 72.10, 1 / 5, "×1/5 slow motion", tx["s1"]),
        ("run", 72.10, 73.30, 1 / 5, "×1/5 slow motion", tx["s2a"]),
        ("run", 73.30, 74.10, 1 / 5, "×1/5 slow motion", tx["s2b"]),
        ("hold", 74.10, None, 8.0, "paused", tx["s2b"]),
        ("run", 74.10, 75.40, 1 / 6, "×1/6 slow motion", tx["s3"]),
        ("hold", 75.40, None, 1.0, "paused", tx["s3"]),
        ("run", 75.40, 76.10, 1 / 10, "×1/10 slow motion", tx["s4"]),
        ("hold", 76.10, None, 3.4, "paused", tx["s4"]),
        ("hold", 76.10, None, 6.6, "paused", tx["s5"]),
        ("run", 76.10, 76.25, 1 / 40, "×1/40 slow motion", tx["s6"]),
        ("hold", 76.25, None, 2.6, "paused", tx["s6"]),
        ("run", 76.25, 76.45, 1 / 40, "×1/40 slow motion", tx["s7"]),
        ("run", 76.45, SHOWN[1], 1 / 5, "×1/5 slow motion", tx["s8"]),
        ("hold", SHOWN[1], None, 12.8, "paused", tx["s9"]),
    ]


TITLE_SECONDS = 9.0              # title, the question and what to watch for, read at <= SLIDE_WPS (asserted)
POP_SECONDS = (14.1, 13.5, 12.6)  # the table with its footnotes, under three captions; the caveats follow on the
                                 # presentation's slide 'Prediction: what it does and does not show' (slides.py
                                 # s05_prediction), after the result slide, not in a box beside the table
END_SECONDS = 9.0                # the end caption and the key-number sentence together, <= SLIDE_WPS (asserted)
KEY_FS = 15                      # end card's key-number sentence: one line inside the frame's 16 px gutter (asserted)
SLIDE_WPS = 5.0                  # bound for slide text that stays on screen, in tokens/s: numbers and symbols count, so
                                 # this is about 3.5-4 words/s of prose (captions: 2.5 tokens/s)


def pop_captions(tx: dict) -> tuple:
    """The population slide's captions, one per entry of POP_SECONDS."""
    caps = (tx["p1"], tx["p2"], tx["p3"])
    assert len(caps) == len(POP_SECONDS)
    return caps


def _line(color, marker, ms, label, hollow=False, lw=1.8):
    from matplotlib.lines import Line2D
    if marker is None:
        return Line2D([], [], color=color, lw=lw, label=label)
    if hollow:
        return Line2D([], [], ls="none", marker=marker, ms=ms, mfc="none", mec=color, mew=1.8, label=label)
    return Line2D([], [], ls="none", marker=marker, ms=ms, color=color, label=label)


def _patch(color, label, alpha=1.0):
    from matplotlib.patches import Patch
    return Patch(facecolor=color, alpha=alpha, edgecolor="none", label=label)


class Episode:
    """The animated frame: fleet (left) and three synchronized panels (right)."""

    def __init__(self, fig, x: dict, n: Register, tx: dict):
        self.fig, self.x, self.n, self.tx = fig, x, n, tx
        d, sel, mis, tru, rep = x["d"], x["sel"], x["mis"], x["tru"], x["rep"]
        self.cable = sel["cable"]
        rows = sel["rows"]
        self.t = d["time"][rows]
        self.lin = d["forecast_linearized_H1"][rows]
        self.rol = d["forecast_rollout_H1"][rows]
        self.y = d["label_H1"][rows]
        self.fa = np.isin(rows, sel["fa_rows"])
        # when each tick's label becomes known: its crossing, if inside the window, else t + H
        self.known = np.where(tru["crossed"], tru["cross_time"], self.t + H)
        self.rep, self.mis = rep, mis
        self.geo = C.mission_geometry()
        st = mis["series_time"]
        lo, hi = int(round(X_RANGE[0] / 0.001)), int(round(X_RANGE[1] / 0.001))
        self.ts = st[lo:hi + 1]
        self.e4 = mis["elongation"][lo:hi + 1, self.cable]
        self.T = np.stack([C.tension(mis["elongation"][lo:hi + 1, j], mis["rate"][lo:hi + 1, j], True) for j in range(5)], axis=1)
        self.snaps = n["snaps"]
        self.peak_t = {}                                        # where each first peak sits in the 1 ms log (label placement)
        for j, (tj, _) in self.snaps.items():
            seg = (self.ts >= tj) & (self.ts <= tj + 0.25)
            self.peak_t[j] = float(self.ts[seg][np.argmax(self.T[seg, j])])
            assert abs(self.T[seg, j].max() / 1e3 - self.snaps[j][1]) < 1e-3, (j, self.T[seg, j].max(), self.snaps[j])
        self.first_up, self.first_speed = n["up4"]
        self.v_b = n["v_b"]
        # layout guarantees: nothing drawn on the gap panel is clipped or runs into the horizon label
        pt, lp_all, rp_all = rep["proj_t"], rep["lin_pct"], rep["rol_pct"][:, :, : rep["proj_t"].size]
        drawn = [self.e4]
        for i in range(len(self.t)):
            over = np.flatnonzero(lp_all[i, 1] > 0.35)
            drawn += [lp_all[i, :, : (int(over[0]) + 1 if over.size else pt.size)].ravel(), rp_all[i].ravel()]
        lo_, hi_ = min(float(np.min(v)) for v in drawn), max(float(np.max(v)) for v in drawn)
        assert GAP_YLIM[0] + 0.1 < lo_ and hi_ < GAP_YLIM[1] - 0.85, (lo_, hi_, GAP_YLIM)   # label box spans ~0.65 m
        # the threshold label on the tension panel sits where every cable carries less than T_b
        assert float(self.T[self.ts <= 74.5].max()) < 1e3 * n["T_b"], "tension above the threshold under its label"
        self._curves = [(self.ts, self.e4)]                     # everything the gap panel ever draws, as (x, y) point sets
        for i, t0 in enumerate(self.t):
            over = np.flatnonzero(lp_all[i, 1] > 0.35)
            end = int(over[0]) + 1 if over.size else pt.size
            for row in lp_all[i, :, :end]:
                self._curves.append((t0 + pt[:end], row))
            for row in rp_all[i]:
                self._curves.append((t0 + pt, row))
        self._note_clear = False
        k0, k1 = int(round(X_RANGE[0] / 0.01)), int(round(X_RANGE[1] / 0.01))
        self.k0 = k0
        cen, half = C.fleet_bounds(mis["state"][k0:k1 + 1], self.geo, margin=FLEET_MARGIN)
        self.cam = C.smooth_camera(cen, window=61)
        self.half0 = half
        self.dyn = []
        self.cap = self.cap_text = None
        self.build()

    def _assert_clear(self, box, pad=(0.05, 0.1)):
        """No point of any curve the gap panel draws, at any frame, falls inside ``box`` (data coords)."""
        (x0, y0), (x1, y1) = box
        x0, x1, y0, y1 = x0 - pad[0], x1 + pad[0], y0 - pad[1], y1 + pad[1]
        for x, y in self._curves:
            inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
            assert not inside.any(), f"gap-panel note would hide data at t = {x[inside][0]:.3f} s, e = {y[inside][0]:.3f} m"
        self._note_clear = True

    def build(self):
        """Static furniture (after a fig.clf())."""
        f = self.fig
        self.ax_f = f.add_axes([0.03, 0.19, 0.37, 0.635])
        box = self.ax_f.get_position()
        self.half = C.fit_aspect(self.half0, (box.width * C.W) / (box.height * C.H))
        self.ax_e = f.add_axes([0.47, 0.615, 0.505, 0.176])
        self.ax_T = f.add_axes([0.47, 0.455, 0.505, 0.115])
        self.ax_h = f.add_axes([0.47, 0.235, 0.505, 0.175])
        C.title(f, "Per-line forecast vs fleet rollout")
        self.sub = f.text(0.03, 0.905, self.tx["subtitle"], fontsize=C.FS_SUB, color=C.MUTED, ha="left", va="top")
        f.text(0.03, 0.874, self.tx["selection"], fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="top")
        f.text(0.03, 0.832, "Fleet, top view (10 ms plant state)", fontsize=C.FS_SMALL, color=C.INK, ha="left", va="bottom")
        f.text(0.245, 0.832, f"▬▬ halo: cable {self.cable} (watched)", fontsize=C.FS_SMALL, color=C.ACCENT, ha="left",
               va="bottom", weight="bold")
        self.gap_title = f.text(0.47, 0.826, f"Cable {self.cable} gap e = chord − rest length (m); e ≤ 0: slack (1 ms log)",
                                fontsize=C.FS_SMALL, color=C.INK, ha="left", va="bottom")
        # the gap panel's key sits on its own strip above the axes, so it can never hide the trace
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        band = lambda col: (Patch(facecolor=col, alpha=0.28, edgecolor="none"), Line2D([], [], color=col, lw=2.0, ls=(0, (5, 2))))
        self.gap_key = f.legend([Line2D([], [], color=C.INK, lw=1.8), band(C.SLACK), band(C.TAUT)],
                                [f"cable {self.cable}, truth", "per-line projection", f"rollout projection ({self.n['N_roll']} futures)"],
                                loc="lower left", bbox_to_anchor=(0.47, 0.793), ncol=3, fontsize=C.FS_TINY, frameon=False,
                                handlelength=2.4, columnspacing=1.1, handletextpad=0.5, borderpad=0.0, borderaxespad=0.0)
        f.text(0.47, 0.576, "Cable tension (kN), 1 ms log", fontsize=C.FS_SMALL, color=C.INK, ha="left", va="bottom")
        f.text(0.47, 0.416, self.tx["h_title"], fontsize=C.FS_SMALL, color=C.INK, ha="left", va="bottom")
        C.cable_key(f, y=0.145, x=0.215)
        handles = [_line(C.SLACK, "s", 8, "per-line h"), _line(C.TAUT, "o", 11, "rollout h", hollow=True),
                   _patch(GREY_Y1, "truth: snap within 1 s (once known)"), _patch(PALE_RED, "per-line false alarm")]
        f.legend(handles=handles, loc="center", ncol=4, bbox_to_anchor=(0.722, 0.145), fontsize=C.FS_SMALL,
                 handlelength=1.3, columnspacing=1.0, handletextpad=0.4)
        C.footer(f, FOOTER)
        self.dyn, self.cap, self.cap_text = [], None, None

    # ---------------------------------------------------------------- per frame
    def draw(self, now: float, speed: str, caption: str | None, key_box: bool = False):
        for a in self.dyn:
            a.remove()
        self.dyn = [C.clock(self.fig, now, speed)]
        if caption != self.cap_text:
            if self.cap is not None:
                self.cap.remove()
            self.cap = C.caption(self.fig, caption, y=CAPTION_Y) if caption else None
            self.cap_text = caption
        self._fleet(now)
        self._gap(now)
        self._tension(now)
        self._forecast(now)
        if key_box:                                            # the key number, in place of the subtitle
            self.sub.set_visible(False)
            self.dyn.append(self.fig.text(0.034, 0.909, self.tx["key"], fontsize=KEY_FS, color=C.INK, ha="left", va="top",
                                          weight="bold", zorder=20,
                                          bbox=dict(boxstyle="round,pad=0.22", fc="#fbf7ef", ec=C.ACCENT, lw=1.4)))
        else:
            self.sub.set_visible(True)

    def _fleet(self, now):
        ax = self.ax_f
        k = int(math.floor(now / 0.01 + 1e-9))                 # latest 10 ms state sample (no interpolation)
        row = self.mis["state"][k]
        i1 = k * 10                                            # the same instant in the 1 ms log
        T_row = C.tension(self.mis["elongation"][i1], self.mis["rate"][i1], np.ones(5, bool))
        centre = self.cam[k - self.k0]
        C.draw_fleet(ax, row, self.geo, T_row, centre=centre, half=self.half, labels=False)
        lp, vp = C.unpack_state(row)
        a, b = C.attachment_points(lp, vp, self.geo)
        c = self.cable
        ax.plot(*np.stack([a[c], b[c]]).T, color=C.ACCENT, lw=13, alpha=0.30, solid_capstyle="round", zorder=2.5)
        from matplotlib.patches import Polygon
        ax.add_patch(Polygon(C.body_polygon(vp[c], C.hull()), closed=True, facecolor="none", edgecolor=C.ACCENT,
                             lw=2.4, zorder=4.5))                    # the watched vessel's hull, outlined
        self._vessel_labels(ax, vp)
        lines = []
        for j in sorted(self.snaps, key=lambda j: self.snaps[j][0]):
            tj, pk = self.snaps[j]
            if now >= tj:
                lines.append(f"{tj:7.3f} s  cable {j}  {pk:4.1f} kN")
                if now < tj + 0.15:
                    ax.plot([a[j][0]], [a[j][1]], marker="*", ms=24, color=C.ACCENT, mec=C.INK, mew=1.0, zorder=8)
        lo, hi = centre - self.half, centre + self.half
        body = "\n".join(lines) if lines else "  none yet"
        ticker = ax.text(lo[0] + 0.02 * (hi[0] - lo[0]), hi[1] - 0.02 * (hi[1] - lo[1]),
                         "Re-engagements since the 71 s slack\n(first per cable, record's peak)\n" + body,
                         fontsize=C.FS_TINY, family="DejaVu Sans Mono", color=C.INK, ha="left", va="top", zorder=9,
                         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C.FAINT, alpha=0.92))
        note = ax.text(hi[0] - 0.02 * (hi[0] - lo[0]), lo[1] + 0.03 * (hi[1] - lo[1]),
                       "no contact is modelled: recorded hulls may overlap", fontsize=C.FS_TINY, color=C.MUTED,
                       ha="right", va="bottom", zorder=9)
        # no body may pass under the ticker or the note (pixel test on the plant-size outlines)
        r = self.fig.canvas.get_renderer()
        from tether.physics import fleet as F
        polys = [ax.transData.transform(C.body_polygon(lp, F.pentagon_vertices()))] + \
                [ax.transData.transform(C.body_polygon(v, C.hull())) for v in vp]
        pad = 0.4 * C.FS_TINY * C.DPI / 72.0                        # the ticker's bbox pad (0.4 font sizes), in px
        for label, art, p in (("ticker", ticker, pad), ("note", note, 2.0)):
            e = art.get_window_extent(r).padded(p)
            rect = np.array([[e.x0, e.y0], [e.x1, e.y0], [e.x1, e.y1], [e.x0, e.y1]])
            assert all(_separated(q, rect) for q in polys), f"a body passes under the fleet-view {label} at t = {now:.3f} s"

    def _vessel_labels(self, ax, vp):
        """Vessel numbers just ahead of each bow, pushed apart where the recorded hulls bunch up
        (a thin leader line joins a displaced number to its bow)."""
        bows = np.array([C.body_polygon(v, np.array([[0.5 * C.L_HULL + 0.9, 0.0]]))[0] for v in vp])
        tips = np.array([C.body_polygon(v, np.array([[0.5 * C.L_HULL, 0.0]]))[0] for v in vp])
        lab = bows.copy()
        dmin = 1.35
        for _ in range(300):
            moved = False
            for i in range(len(lab)):
                for j in range(i + 1, len(lab)):
                    dv = lab[j] - lab[i]
                    dist = float(np.linalg.norm(dv))
                    if dist < dmin:
                        dv = dv / dist if dist > 1e-9 else np.array([0.0, 1.0])
                        push = 0.5 * (dmin - dist) * dv
                        lab[i] -= push
                        lab[j] += push
                        moved = True
            if not moved:
                break
        for i, (p, q) in enumerate(zip(bows, lab)):
            watched = i == self.cable
            if watched or np.linalg.norm(q - p) > 0.45:            # the watched vessel's number is always tied to its bow
                ax.plot([tips[i][0], q[0]], [tips[i][1], q[1]], color=C.ACCENT if watched else C.MUTED,
                        lw=1.2 if watched else 0.7, zorder=6)
            ax.text(q[0], q[1], str(i), fontsize=C.FS_SMALL, color=C.ACCENT if i == self.cable else C.INK,
                    weight="bold" if i == self.cable else "normal", ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))

    def _tick_now(self, now):
        i = np.flatnonzero(self.t <= now + 1e-9)
        if i.size == 0 or now >= self.t[-1] + 0.1 - 1e-9:
            return None
        return int(i[-1])

    def _gap(self, now):
        ax = self.ax_e
        ax.cla()
        ax.set_xlim(*X_RANGE)
        ax.set_ylim(*GAP_YLIM)
        ax.set_yticks([-4, -3, -2, -1, 0, 1])
        ax.axhline(0.0, color=C.MUTED, lw=0.9, zorder=1)
        ax.set_ylabel("e (m)", fontsize=C.FS_SMALL)
        ax.tick_params(labelbottom=False)
        ax.grid(True, axis="y")
        i = self._tick_now(now)
        if i is not None:
            t0 = self.t[i]
            ax.axvspan(t0, t0 + H, color=C.ACCENT, alpha=0.10, lw=0, zorder=0)
            ax.text(min(t0 + 0.5 * H, X_RANGE[1] - 1.2), GAP_YLIM[1] - 0.06, f"1 s horizon of the {t0:.1f} s tick", fontsize=C.FS_TINY,
                    color=C.ACCENT, ha="center", va="top", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            pt = self.rep["proj_t"]
            lp = self.rep["lin_pct"][i]
            over = np.flatnonzero(lp[1] > 0.35)                 # stop the per-line path just past e = 0
            end = int(over[0]) + 1 if over.size else pt.size
            ax.fill_between(t0 + pt[:end], lp[0][:end], lp[2][:end], color=C.SLACK, alpha=0.28, lw=0, zorder=2)
            ax.plot(t0 + pt[:end], lp[1][:end], color=C.SLACK, lw=2.0, ls=(0, (5, 2)), zorder=3)
            rp = self.rep["rol_pct"][i][:, : pt.size]
            ax.fill_between(t0 + pt, rp[0], rp[2], color=C.TAUT, alpha=0.28, lw=0, zorder=2)
            ax.plot(t0 + pt, rp[1], color=C.TAUT, lw=2.0, ls=(0, (5, 2)), zorder=3)
        m = self.ts <= now
        ax.plot(self.ts[m], self.e4[m], color=C.INK, lw=1.8, zorder=4)
        note = ax.text(X_RANGE[0] + 0.06, GAP_YLIM[0] + 0.08, self.tx["band_note"], fontsize=C.FS_TINY, color=C.MUTED,
                       ha="left", va="bottom", linespacing=1.2, zorder=6)
        if not self._note_clear:                              # once: nothing the panel ever draws may pass under the note
            self._assert_clear(ax.transData.inverted().transform(note.get_window_extent(self.fig.canvas.get_renderer())))
        if now >= self.first_up:
            ax.plot([self.first_up], [0.0], marker="o", ms=9, color=C.ACCENT, mec=C.INK, zorder=6)
            ax.annotate(f"cable {self.cable} re-engages {self.first_up:.2f} s\nat {self.first_speed:.2f} m/s > v_b", (self.first_up, 0.0),
                        xytext=(X_RANGE[1] - 0.04, -2.3), textcoords="data", ha="right", va="top", fontsize=C.FS_TINY, color=C.INK,
                        arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.8))
        ax.axvline(now, color=C.INK, lw=0.8, alpha=0.5, zorder=5)

    def _tension(self, now):
        ax = self.ax_T
        ax.cla()
        ax.set_xlim(*X_RANGE)
        ax.set_ylim(0, 21.5)
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.set_ylabel("T (kN)", fontsize=C.FS_SMALL)
        ax.tick_params(labelbottom=False)
        ax.grid(True, axis="y")
        ax.axhline(self.n["T_b"], color=C.MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
        ax.text(X_RANGE[0] + 0.05, self.n["T_b"] + 0.5, self.tx["threshold"], fontsize=C.FS_TINY, color=C.MUTED,
                ha="left", va="bottom", zorder=2)
        m = self.ts <= now
        for j in (0, 3, 2, 1, 4):
            ax.plot(self.ts[m], self.T[m, j] / 1e3, color=TENSION_COLORS[j], lw=1.8 if j == 4 else 1.3, zorder=4 if j == 4 else 3)
        s, pk = self.snaps, self.peak_t
        done = [j for j in sorted(s, key=lambda j: s[j][0]) if now >= pk[j]]
        if done:
            ax.text(X_RANGE[0] + 0.05, 20.8, "first peaks (kN)   " + "  ·  ".join(f"{j}: {s[j][1]:.1f}" for j in done),
                    fontsize=C.FS_TINY, color=C.INK, ha="left", va="top")
        for j in done:
            if j == 2:
                continue                                        # drawn with cable 1 (0.6 ms apart, 0.4 kN lower)
            label = "1, 2" if j == 1 else str(j)
            ax.text(pk[j], s[j][1] + 0.4, label, fontsize=C.FS_TINY, color=C.ACCENT if j == self.cable else C.INK,
                    weight="bold" if j == self.cable else "normal", ha="center", va="bottom")
        ax.axvline(now, color=C.INK, lw=0.8, alpha=0.5, zorder=5)

    def _forecast(self, now):
        ax = self.ax_h
        ax.cla()
        ax.set_xlim(*X_RANGE)
        ax.set_ylim(-0.1, 1.12)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "0.5", "1"])
        ax.set_xlabel("simulation time t (s)", fontsize=C.FS_SMALL, labelpad=2)
        ax.set_ylabel("h", fontsize=C.FS_SMALL)
        ax.axhline(TOP, color=C.MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(X_RANGE[0] + 0.05, TOP + 0.02, f"top bin h ≥ {self.n['top']}", fontsize=C.FS_TINY, color=C.MUTED,
                ha="left", va="bottom")
        shown = self.t <= now + 1e-9
        known = self.known <= now + 1e-9
        for i in np.flatnonzero(shown & known):
            if self.y[i]:
                ax.axvspan(self.t[i] - 0.05, self.t[i] + 0.05, color=GREY_Y1, lw=0, zorder=0)
            elif self.lin[i] >= TOP:
                ax.axvspan(self.t[i] - 0.05, self.t[i] + 0.05, color=PALE_RED, lw=0, zorder=0)
        ax.plot(self.t[shown], self.lin[shown], ls="none", marker="s", ms=6.5, color=C.SLACK, zorder=4)
        ax.plot(self.t[shown], self.rol[shown], ls="none", marker="o", ms=10.5, mfc="none", mec=C.TAUT, mew=1.8, zorder=5)
        fa_known = self.fa & known
        if fa_known.any():
            k = int(fa_known.sum())
            ax.annotate(f"{k} per-line false alarm{'s' if k > 1 else ''}:\nh = 1.00, no snap within 1 s",
                        (self.t[self.fa][0], 1.0), xytext=(74.0, 0.62), fontsize=C.FS_TINY, color=C.SLACK,
                        ha="center", va="center", arrowprops=dict(arrowstyle="->", color=C.SLACK, lw=1.0, shrinkA=2, shrinkB=6))
        miss = (self.lin < 0.1) & self.y & known
        if miss.any():
            # one snap, several ticks: a comb from the note down onto each h = 0 square (inside its grey column)
            idx = np.flatnonzero(miss)
            note = ax.text(74.0, MISS_Y, self.tx["miss_note"].replace(f"{self.n['lin_miss_iv']} ticks", f"{idx.size} ticks"),
                           fontsize=C.FS_TINY, color=C.SLACK, ha="center", va="center")
            xr_ = ax.transData.inverted().transform(note.get_window_extent(self.fig.canvas.get_renderer()))[1, 0]
            ax.plot([xr_ + 0.05, self.t[idx].max()], [MISS_Y, MISS_Y], color=C.SLACK, lw=1.0, zorder=3)
            for tt in self.t[idx]:
                ax.annotate("", (tt, 0.0), xytext=(tt, MISS_Y), arrowprops=dict(arrowstyle="->", color=C.SLACK, lw=1.0,
                                                                                  shrinkA=0, shrinkB=6), zorder=3)
        ax.axvline(now, color=C.INK, lw=0.8, alpha=0.5, zorder=6)


TITLE = "Prediction: a per-line forecast vs a fleet rollout"


def title_card(fig, n: Register, tx: dict):
    """The clip title, the question the clip answers, and what to watch for (two complete sentences)."""
    fig.clf()
    fig.text(0.5, 0.62, TITLE, fontsize=34, weight="bold", color=C.INK, ha="center", va="center")
    fig.text(0.5, 0.505, tx["t_question"], fontsize=C.FS_SUB + 4, color=C.INK, ha="center", va="center")
    fig.text(0.5, 0.39, tx["t_watch"], fontsize=C.FS_SUB + 1, color=C.INK, ha="center", va="center", linespacing=1.35)
    C.footer(fig, "Paper Sec. IV · records/v2/phase5/p5_t2prime_ticks.npz · p5_t2prime_results.json · "
                  "records/phase5/cache/phase5_missions.pkl")


POP_TITLE = "Both from the exact plant state · P5-T2′ oracle pre-test"
POP_SEPARATORS = (0.766, 0.590, 0.505, 0.410, 0.340)


def _pop_title(n: Register) -> str:
    return f"All {n['missions']} missions: the same {n['N']} scored ticks, H = {n['H']:.0f} s"


def _table_rows(n: Register) -> list:
    """(y, label, per-line value, rollout value, font size) of the population table."""
    tl, tr_ = n["top_lin"], n["top_rol"]
    (lc, lct), (lu, lut) = n["top_split"]
    (rc, rct), (ru, rut) = n["top_split_rol"]
    assert lc == tl[2] and lc + lu == tl[0] and rc == tr_[2] and rc + ru == tr_[0]
    cl, cr = n["cw_lin"], n["cw_rol"]
    ls, rs = n["lin_slope"], n["rol_slope"]
    return [
        (0.715, f"Top-bin false alarms\n(h ≥ {n['top']}, no dangerous snap within 1 s)", f"{tl[0]} / {tl[1]}", f"{tr_[0]} / {tr_[1]}",
         C.FS_TITLE),
        (0.635, "   on coupled ticks\n   on uncoupled ticks", f"{lc} of {lct}\n{lu} of {lut}", f"{rc} of {rct}\n{ru} of {rut}", C.FS_BODY),
        (0.548, "Certain and wrong\n(h exactly 0 or 1; post hoc)", f"{cl[0] + cl[1]}", f"{cr[0] + cr[1]}", C.FS_BODY),
        (0.458, "Recalibration slope *\n(calibrated = 1; 95 % interval)", f"{ls[0]:.3f}\n[{ls[1]:.3f}, {ls[2]:.3f}]",
         f"{rs[0]:.3f}\n[{rs[1]:.3f}, {rs[2]:.3f}]", C.FS_BODY),
        (0.376, "AUROC (perfect ranking = 1)", f"{n['auroc'][0]:.3f}", f"{n['auroc'][1]:.3f}", C.FS_BODY),
    ]


def population_slide(fig, n: Register, tx: dict, stage: int, caption: str):
    """All-mission numbers (left, with their footnotes, from stage 0) and the caveats box (right): the
    first four caveats from stage 1, the rollout's failed gate added, highlighted, at stage 2."""
    import textwrap
    from matplotlib.patches import FancyBboxPatch, Rectangle

    fig.clf()
    C.title(fig, _pop_title(n), POP_TITLE)
    x0, xl, xr, x_end = 0.04, 0.405, 0.525, 0.585
    fig.text(xl, 0.795, "Per-line", fontsize=C.FS_BODY, weight="bold", color=C.SLACK, ha="center", va="center")
    fig.text(xr, 0.795, "Fleet rollout", fontsize=C.FS_BODY, weight="bold", color=C.TAUT, ha="center", va="center")
    for y, label, va_, vb_, fs in _table_rows(n):
        fig.text(x0, y, label, fontsize=C.FS_BODY, color=C.INK, ha="left", va="center", linespacing=1.15)
        bold = "bold" if fs == C.FS_TITLE else "normal"
        fig.text(xl, y, va_, fontsize=fs, color=C.SLACK, ha="center", va="center", weight=bold, linespacing=1.1)
        fig.text(xr, y, vb_, fontsize=fs, color=C.TAUT, ha="center", va="center", weight=bold, linespacing=1.1)
    for y in POP_SEPARATORS:
        fig.add_artist(Rectangle((x0, y), x_end - x0, 0.0012, transform=fig.transFigure, color=C.FAINT, lw=0))
    yy = 0.312
    for note in tx["foot"]:
        wrapped = textwrap.fill(note, 100)
        fig.text(x0, yy, wrapped, fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="top", linespacing=1.35)
        yy -= 0.0225 * (wrapped.count("\n") + 1) + 0.012
    if False:  # caveats box moved to the presentation's s05_prediction slide (reading load; see caption_budget)
        bx0, by0, bw, bht = 0.625, 0.17, 0.345, 0.63
        fig.add_artist(FancyBboxPatch((bx0, by0), bw, bht, boxstyle="round,pad=0.008", transform=fig.transFigure,
                                      fc="#fbf7ef", ec=C.ACCENT, lw=1.4))
        fig.text(bx0 + 0.02, by0 + bht - 0.028, "Caveats", fontsize=C.FS_BODY, weight="bold", color=C.INK, ha="left", va="top")
        yy = by0 + bht - 0.082
        for text, first, hi in tx["cav"]:
            if stage < first:
                continue
            wrapped = textwrap.fill(text, 44 if hi else 48)
            fig.text(bx0 + 0.02, yy, "•", fontsize=C.FS_SMALL, color=C.INK, ha="left", va="top")
            fig.text(bx0 + 0.035, yy, wrapped, fontsize=C.FS_SMALL, color=C.SLACK if hi else C.INK, ha="left", va="top",
                     linespacing=1.3, weight="bold" if hi else "normal")
            yy -= 0.0275 * (wrapped.count("\n") + 1) + 0.026
    C.caption(fig, caption, y=CAPTION_Y)
    C.footer(fig, FOOTER_POP)


# =====================================================================================  main

STORY = ("One recorded slack interval of cable 4 (seed 5008, fan formation, T0 = 1 kN), forecast every 0.1 s by two "
         "models given the exact plant state: the per-line first-passage model (red) and the six-body fleet rollout "
         "(blue), which is also given the known forcing; each forecasts h, the probability of a dangerous snap within 1 s "
         "(cable 4 pulling tight faster than the severing speed v_b = 0.47 m/s). At 75.6-76.1 s the per-line model is "
         "certain of a dangerous snap of cable 4 within 1 s; the rollout says no. Cables 1 and 2 re-engage first "
         "(17.0/16.6 kN); post hoc, the payload is jerked toward vessel 4 and cable 4's closing speed falls; it re-engages "
         "only at 77.29 s. The per-line model gives six coupled top-bin false alarms, then h = 0 at four ticks whose 1 s "
         "horizon held that one snap; the rollout is within 0.002 of the truth throughout. Then the population numbers over "
         "all 9227 ticks; the caveats (exact state; the rollout also knew the forcing; neither is a deployable monitor; the "
         "rollout fails its own slope gate by near-separation; post hoc, the per-line slope depends on the forecast clip) "
         "follow on the presentation's 'Prediction: what it does and does not show' slide, after the result slide.")
CAVEATS = [
    "Both models were given the exact plant state (oracle precursor); the comparison excludes estimation error and says "
    "nothing about a deployable monitor. Fed by the estimator, the v1 per-line arm scored a better, still failing slope "
    "(0.71 vs 0.46, v1 event definition; paper Sec. IV) - not shown.",
    "The rollout was also given the known forcing (current weather sample on every body, the full squall, the vessels' "
    "thrust and heading schedules, 2048 futures from the true background law). The comparison does not separate fleet "
    "dynamics from forcing information (paper Sec. IV); the clip does not claim it does.",
    "The rollout fails its own pre-declared slope gate by near-separation ({gate}); the declared held-out "
    "recalibration also fails ({recal}: records/v2/phase5/p5_t2prime_results.json "
    "recalibration_branch_iii.recal_rollout_H1_test.test_recalibrated) - not on screen. It is not called calibrated.",
    "The per-line slope rests on the declared clip 1/(2N): post hoc, at clip 0.01 it is {clip01} (cluster bootstrap by the "
    "declared statistic, recomputed here; Paper Sec. IV: 0.88 [0.79, 1.02]), on screen as a footnote; its end-bin failures "
    "and the 63 certain-and-wrong forecasts do not rest on the clip.",
    "The episode is selected by rule (the mission holding most of the 18 coupled top-bin false alarms at 70-110 s): the "
    "extreme case, not a typical one (said on screen). 'The snaps jerk the payload toward vessel 4' is a post-hoc reading "
    "of the 10 ms state, labelled 'Post hoc' in its caption (the payload term of cable 4's closing speed drops from -0.78 "
    "to -2.88 m/s while only cables 1 and 2 carry tension); the paper does not analyse this episode.",
    "'Forecast h = 0 at 4 ticks although the snap came within the next second' (caption s9) counts ticks (76.3, 76.4, "
    "76.5, 76.8 s), all for the one re-engagement of cable 4 at 77.288 s, which lies inside each of their 1 s horizons "
    "(asserted from the labels). Cable 4 re-engages again at {later4} after a short dwell, outside the interval: it "
    "is drawn on the traces but not labelled.",
    "Recording runs: the forecast event, called a 'dangerous snap' (captions s2a-s2b, the h-panel title and the table), "
    "is cable 4's first re-engagement within 1 s closing faster than v_b, the speed at which the fan impact law peaks at "
    "the 4.5 kN stress threshold (on screen also the tension panel's threshold line); no cable is cut. Caption s2b glosses "
    "this as 'its estimated peak reaches 4.5 kN, where severance is scored' (the impact law is the per-line estimate of the "
    "peak; paper Sec. II scores severance at 4.5 kN). Elsewhere 'snap' keeps the presentation's broad meaning, any "
    "re-engagement jolt (caption s7's 'the snaps' are cables 1 and 2's re-engagements). The qualifier matters in the "
    "table: {slow_fa} of the {n_fa} per-line top-bin false alarms hold a slower re-engagement within 1 s ({slow_v} m/s, "
    "below v_b; none of the six shown, asserted). The captions call v_b 'the severing speed', as paper Sec. IV does, and "
    "say 'these recording runs cut no cable' when the 17.0 and 16.6 kN peaks appear. "
    "The missions come from v1's exploratory continuation; the P5-T2' gate committed no prediction.",
    "The projections drawn on the gap panel are model outputs of replays that reproduce the record's h exactly (per-line "
    "at all 439 ticks of seed 5008, rollout at all 62 ticks of the interval); the record itself stores only h. Their "
    "5-95 % bands are mostly thinner than the dashed median line (at +1 s: per-line <= {w_lin} cm before 76.2 s, rollout "
    "<= {w_rol} cm at every tick; per-line up to {w_lin_max} m after 76.2 s); the key says 'band often thinner than the line'.",
    "Fleet view: latest 10 ms state sample, no interpolation; traces: 1 ms log. The plant models no contact: in seed 5008 "
    "the recorded vessel hulls overlap at every 10 ms sample of the shown window ({pairs} overlapping pairs; the first "
    "overlap of the mission is at {first_overlap} s), as is typical of these missions after the squall ({over76} of the "
    "{reach76} missions that reach 76 s have overlapping hulls at 76 s, {over45} of {missions} at 45 s). Drawn as recorded "
    "(noted on screen). Vessel numbers are pushed apart for legibility, with a leader line to the bow; vessel 4's hull is "
    "outlined and its number is always tied to its bow.",
    "Clock: the record's mission time (these missions have no warm-up: MissionSpec.run_spec warmup = 0).",
    "Population slide: all its text and its {n_pop} captions, read once, need {table_wps} tokens/s over its {pop_s} s (numbers "
    "and symbols count as tokens; bound 5.0/s, about 3.5-4 words/s of prose). The caveats box that used to sit beside the "
    "table was moved to the presentation's slide 'Prediction: what it does and does not show' (slides.py s05_prediction), "
    "which follows the result slide (s05a_prediction_result); caption p3 still states the two main caveats (exact state; "
    "the rollout also had the known forcing; neither is a deployable monitor). End card: the end caption and the "
    "key-number sentence need {end_wps} tokens/s over its {end_s} s; the sentence is set at {key_fs} pt (the subtitle's "
    "18 pt would not fit one line).",
    "Captions are complete sentences (revision 2026-09-15; the numbers, instants and drawings are unchanged). For reading "
    "time the drift before the false alarms plays at x1/5 and x1/6 instead of x1/4, the re-engagement of cable 4 at x1/5 "
    "instead of x1/4, and pauses were added ({pauses}); the clock shows each speed and 'paused'. The second revision "
    "(same day, after review) splits the definition of h into captions s2a and s2b, defines v_b and the 'dangerous snap' "
    "there, defines 'tick' in caption s3 and states caption s9 and the end card's key number as full sentences; the x1/5 "
    "run 72.10-74.10 s is split at 73.30 s between s2a and s2b, and the pauses at 74.10 s and 75.40 s were added and those "
    "at 76.10 s (s4) and 77.84 s (s9) lengthened for their longer captions. 'The per-line model "
    "follows cable 4 alone' (title card) and 'only the rollout simulates all six bodies' (caption s3) paraphrase paper "
    "Sec. IV's protocol (the per-line model propagates one cable's radial coordinate; the rollout integrates the payload "
    "and five vessels). 'The six certain forecasts were false alarms' refers to the six h = 1.00 ticks named in the two "
    "captions before it; the per-line h is also 1.00 at {late_certain}, ticks whose horizon holds the snap, where it is "
    "right (asserted).",
]


def _speed_ok(label: str, speed: float) -> bool:
    return label == f"×1/{round(1 / speed)} slow motion"


def assert_in_frame(fig, where: str, margin: int = 16) -> None:
    """Every visible text and legend lies inside the 1920 x 1080 frame with a side gutter of ``margin`` px."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    arts = list(fig.texts) + list(fig.legends) + [a for ax in fig.axes for a in ax.texts]
    for a in arts:
        if not a.get_visible() or (hasattr(a, "get_text") and not a.get_text()):
            continue
        e = a.get_window_extent(r)
        assert e.x0 >= margin - 0.5 and e.x1 <= C.W - margin + 0.5 and e.y0 >= -0.5 and e.y1 <= C.H + 0.5, \
            f"{where}: text outside the frame: {getattr(a, 'get_text', lambda: a)()!r} x {e.x0:.0f}-{e.x1:.0f}, y {e.y0:.0f}-{e.y1:.0f}"


def render(x: dict, n: Register, tx: dict, man: C.Manifest, stills: list | None = None) -> int:
    """Stream every frame to ffmpeg (or, with ``stills``, write only those instants as PNG)."""
    fig = C.new_frame()
    if stills is not None:
        STILLS.mkdir(parents=True, exist_ok=True)
        title_card(fig, n, tx)
        assert_in_frame(fig, "title")
        fig.savefig(STILLS / "still_title.png")
        fig.clf()
        ep = Episode(fig, x, n, tx)
        for now, label, cap in stills:
            ep.draw(now, label, cap)
            assert_in_frame(fig, f"episode {now:.3f} s")
            fig.savefig(STILLS / f"still_{now:.3f}.png")
        for stage, cap in enumerate(pop_captions(tx)):
            population_slide(fig, n, tx, stage, cap)
            assert_in_frame(fig, f"population stage {stage}")
            fig.savefig(STILLS / f"still_pop{stage}.png")
        fig.clf()
        ep.build()
        ep.draw(SHOWN[1], "paused", tx["end"], key_box=True)
        assert_in_frame(fig, "end")
        fig.savefig(STILLS / "still_end.png")
        return 0
    w = C.writer()
    frames = 0
    with w.saving(fig, str(CLIP), dpi=C.DPI):
        title_card(fig, n, tx)
        assert_in_frame(fig, "title")
        frames += C.hold(w, TITLE_SECONDS)
        fig.clf()
        ep = Episode(fig, x, n, tx)
        for kind, a, b, sp, label, cap in shots(tx):
            if kind == "run":
                assert _speed_ok(label, sp), (label, sp)
                count = int(round((b - a) / sp * C.FPS))
                for i in range(count):
                    ep.draw(a + (i / C.FPS) * sp, label, cap)
                    if i in (0, count - 1):
                        assert_in_frame(fig, f"shot {a:.2f}-{b:.2f} s, frame {i}")
                    w.grab_frame()
                    frames += 1
            else:
                ep.draw(a, label, cap)
                assert_in_frame(fig, f"hold at {a:.2f} s")
                frames += C.hold(w, sp)
        for stage, (secs, cap) in enumerate(zip(POP_SECONDS, pop_captions(tx))):
            population_slide(fig, n, tx, stage, cap)
            assert_in_frame(fig, f"population stage {stage}")
            frames += C.hold(w, secs)
        fig.clf()
        ep.build()
        ep.draw(SHOWN[1], "paused", tx["end"], key_box=True)
        assert_in_frame(fig, "end")
        frames += C.hold(w, END_SECONDS)
    return frames


def _words(s: str) -> int:
    return len(s.replace("\n", " ").split())


def caption_budget(tx: dict, n: Register, man: C.Manifest) -> dict:
    """Each caption: at most two lines, on screen >= 4 s and >= words / 2.5 s (every token counts as a word);
    a caption that continues over consecutive shots (a run and the pause after it) counts its whole time.
    Population slide: its table + footnotes (on screen for all stages) and its captions are readable at
    SLIDE_WPS words/s as one reading load.  Title card: title, question and what to watch for, at SLIDE_WPS."""
    items = []
    for kind, a, b, sp, _, cap in shots(tx):
        secs = ((b - a) / sp) if kind == "run" else sp
        if items and items[-1][0] == cap:
            items[-1][1] += secs
        else:
            items.append([cap, secs])
    items += [list(p) for p in zip(pop_captions(tx), POP_SECONDS)] + [[tx["end"], END_SECONDS]]
    caps = [c for c, _ in items]
    assert len(set(caps)) == len(caps), "a caption returns after another one: count its time per appearance"
    for cap, secs in items:
        words = _words(cap)
        assert cap.count("\n") <= 1 and secs >= 4.0 - 1e-9 and secs >= words / 2.5 - 1e-9, (cap, secs, words)
    man.check("every caption: <= 2 lines, >= 4 s on screen, <= 2.5 tokens/s (numbers and symbols count as words)", True,
              "; ".join(f"{_words(c)} tokens / {s:.2f} s" for c, s in items))
    table = [_pop_title(n), POP_TITLE, "Per-line Fleet rollout"] + [" ".join(r[1:4]) for r in _table_rows(n)] + tx["foot"]
    table_words, table_s = sum(_words(s) for s in table), sum(POP_SECONDS)
    load = table_words + sum(_words(c) for c in pop_captions(tx))      # everything on the slide, read once
    man.check(f"population slide readable at <= {SLIDE_WPS:g} tokens/s as ONE reading load (all slide text and all "
              f"{len(POP_SECONDS)} captions over the slide's total time)", load / table_s <= SLIDE_WPS,
              f"{load} tokens (table, title, footnotes {table_words}; captions {load - table_words}) over {table_s:g} s "
              f"= {load / table_s:.2f}/s")
    title_load = _words(TITLE) + _words(tx["t_question"]) + _words(tx["t_watch"])
    man.check(f"title card readable at <= {SLIDE_WPS:g} tokens/s (title, question, what to watch for)",
              title_load / TITLE_SECONDS <= SLIDE_WPS and tx["t_watch"].count("\n") <= 1,
              f"{title_load} tokens over {TITLE_SECONDS:g} s = {title_load / TITLE_SECONDS:.2f}/s")
    end_load = _words(tx["end"]) + _words(tx["key"])
    man.check(f"end card readable at <= {SLIDE_WPS:g} tokens/s (end caption and the key-number sentence, read once)",
              end_load / END_SECONDS <= SLIDE_WPS and "\n" not in tx["key"],
              f"{end_load} tokens over {END_SECONDS:g} s = {end_load / END_SECONDS:.2f}/s")
    return {"pop_s": f"{table_s:g}", "cav_wps": "n/a (caveats on the slide after the result slide)",
            "table_wps": f"{load / table_s:.2f}", "n_pop": str(len(POP_SECONDS)),
            "end_s": f"{END_SECONDS:g}", "end_wps": f"{end_load / END_SECONDS:.2f}", "key_fs": str(KEY_FS)}


def ffprobe_frames(path: Path) -> int:
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replay", action="store_true", help="recompute the replay cache")
    ap.add_argument("--stills", action="store_true", help="write inspection stills only (no video)")
    args = ap.parse_args()
    C.ensure_dirs()
    man = C.Manifest(NAME, TITLE, STORY)
    x = prepare(man, force=args.replay)
    n = numbers(x, man)
    tx = texts(n)
    budget = caption_budget(tx, n, man)
    # caption s8 says 'the six certain forecasts were false alarms': the per-line h = 1 ticks of the interval are
    # the six false alarms and, later, only ticks whose horizon holds the snap (label 1), where h = 1 is right
    d_, rows_ = x["d"], x["sel"]["rows"]
    t_, lin_, y_ = d_["time"][rows_], d_["forecast_linearized_H1"][rows_], d_["label_H1"][rows_]
    certain = lin_ == 1.0
    fa_ = np.isin(rows_, x["sel"]["fa_rows"])
    man.check("per-line h = 1.00 in the interval: the six false alarms (75.6-76.1 s, label 0), then only label-1 ticks",
              np.array_equal(certain & ~y_, fa_) and int((certain & ~y_).sum()) == n["n_sel"]
              and bool(np.all(y_[certain & (t_ > n["fa_t"][1] + 1e-9)])),
              f"h = 1.00 at {np.round(t_[certain], 1).tolist()} s; label 1 at {np.round(t_[certain & y_], 1).tolist()} s")
    late = t_[certain & y_]
    sh = shots(tx)
    added = [(a, sp) for prev, (kind, a, _, sp, _, cap) in zip(sh, sh[1:]) if kind == "hold" and cap == prev[5]]
    budget["pauses"] = ", ".join(f"{sp:g} s at {a:.2f} s" for a, sp in added)
    budget["late_certain"] = f"{late.min():.1f}-{late.max():.1f} s"
    man.selection = tx["selection"] + (f" The six are on cable {n['cable']} at {n['fa_t'][0]:.1f}-{n['fa_t'][1]:.1f} s, in one slack "
                                       f"interval of {n['n_ticks']} ticks (71.1-77.2 s)."
                                       " Rule evaluated in code on records/v2/phase5/p5_t2prime_ticks.npz: scored ticks = "
                                       "~bounce & ~censored_H1; top bin forecast_linearized_H1 >= 0.9; false alarm label_H1 = 0; "
                                       "coupled_H1; 70 <= time <= 110 s; per-seed counts " + json.dumps(x["sel"]["per_seed"]) + ".")
    rr = x["tru"]["results"]["recalibration_branch_iii"]["recal_rollout_H1_test"]["test_recalibrated"]
    tru, ov, bw = x["tru"], x["mis"]["overlap_all"], n["band_w"]
    later = tru["later4"]
    fill = {"gate": fmt_slope(n["rol_slope"]), "recal": fmt_slope((rr["slope"], rr["slope_lo"], rr["slope_hi"])),
            "clip01": fmt_slope(n["clip01"]),
            "later4": "; ".join(f"{m[1]:.3f} s ({m[3]:.2f} m/s, {m[2] / 1e3:.1f} kN)" for m in later) if len(later) else "no time in the window",
            "w_lin": f"{bw[0] * 100:.1f}", "w_rol": f"{bw[1] * 100:.1f}", "w_lin_max": f"{bw[2]:.2f}",
            "pairs": "{}-{}".format(*tru["overlap_pairs"]), "first_overlap": f"{tru['first_overlap']:.2f}", **ov, **budget,
            "slow_fa": n.v["slow_fa"][0], "n_fa": n.v["slow_fa"][1], "slow_v": ", ".join(f"{v:.2f}" for v in n.v["slow_fa"][2])}
    man.caveats = [c.format(**fill) for c in CAVEATS]
    if args.stills:
        picks = [(71.60, "×1/5 slow motion", tx["s1"]), (72.70, "×1/5 slow motion", tx["s2a"]), (74.10, "paused", tx["s2b"]),
                 (75.40, "paused", tx["s3"]), (75.85, "×1/10 slow motion", tx["s4"]), (76.10, "paused", tx["s5"]),
                 (76.15, "×1/40 slow motion", tx["s6"]), (76.35, "×1/40 slow motion", tx["s7"]), (77.40, "×1/5 slow motion", tx["s8"]),
                 (77.84, "paused", tx["s9"])]
        render(x, n, tx, man, stills=picks)
        print("stills in", STILLS)
        return
    frames = render(x, n, tx, man)
    man.frames = frames
    man.check("frames written = frames in the file (ffprobe)", ffprobe_frames(CLIP) == frames, f"{frames} frames")
    man.write(CLIP)


if __name__ == "__main__":
    main()
