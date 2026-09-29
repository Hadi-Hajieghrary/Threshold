"""Clip C1 ``setting`` -- the operation: one recorded v1 Squall Passage mission, whole mission at x4.

Storyboard: Presentation/STORYBOARD.md, section C1.  Output: Presentation/clips/setting.mp4,
manifest Presentation/manifests/setting.json, extract cache Presentation/cache/setting_5008.npz.
Run:  python3 -m tether.analysis.v2.present.clip_setting   [--stills DIR  (inspection PNGs only)]

Every frame is plant state read from the campaign record; nothing is re-simulated.  The cache is
an extract of the record (the rows drawn), not a replay.

Records and keys read
---------------------
records/phase5/cache/phase5_missions.pkl   pickle list of 40 dicts written by
    tether/campaign/phase5.py::mission_job / run_missions.  For seed 5008:
      'truth'   PlantTruth: state_time (13000,), state (13000, 36) at 10 ms (18 positions then 18
                velocities; load then vessels 0..4, each x, y, theta); event_time (130000,),
                elongation and rate (130000, 5) at 1 ms; marks (ReengagementRecord tuple)
      'marks'   list of tuples (cable, t_up, depth, v_return, T_peak, dwell) -- field order from
                phase5.mission_job, checked field by field against truth.marks
      'max_tension', 'max_tension_time'
      'outcome' MissionOutcome (severances, completed, closure, mark_count)
records/phase5/phase5_results.json          stress.threshold, stress.per_mission_max (seed order
    5001..5040), missions.outcomes[*] (seed, mark_count, severances), declarations.mission
records/phase5/phase5_spec.md               addendum 2026-09-12 (squall-only runs: the formation
    "rotates about 90 deg and drifts sideways"), cited for caption 5's "in the squall the payload swings"
records/v2/phase5/p5_t2prime_ticks.npz      seed, time, bounce, censored_H1, forecast_linearized_H1,
    label_H1, coupled_H1  (selection rule, shared with clip C4)
records/v2/phase5/p5_t2prime_results.json   reports.linearized_H1.coincidence.top_bin_misses,
    top_bin_misses_coupled  (cross-check of the selection counts)
records/v2/phase6/p6_t0_results.json        excursions[*].tags for seed 5008 with first_severance
    (cable, t_up, T_peak): the scored first severance of this mission
records/v2/phase1/scripted_results.json     cell_b_impact.Z_repin.Z_N_s_per_m (Z: a record check; not on screen,
    since the relation and Z are given on the next slide, s03_perline)
Code read (not modified): tether/campaign/mission.py MissionSpec (incl. weather_distribution,
    weather_direction, weather_scale), thrust_envelope, dogleg_offset, squall_envelope,
    calibrate_squall, initial_state, raised_cosine; tether/physics/constants.py;
    common.WIDTH_FULL_N (the cable width cap, stated in this clip's own legend).

Selection rule (shown on screen, written to the manifest)
---------------------------------------------------------
The mission of clip C4: among the per-line model's coupled top-bin false alarms (scored ticks,
i.e. not bounce and not censored at H = 1 s; forecast_linearized_H1 >= 0.9; label_H1 false;
coupled_H1 true) that fall between the end of the squall and the start of the deceleration
(70-110 s, read from MissionSpec), the seed holding the most of them.  Asserted: 25 false alarms,
24 coupled, 18 in the window, seed 5008 holds 6 (unique maximum).  Not chosen for typicality.
Three pauses, each by rule: (1) the peak of the snap holding the first 1 ms sample above 4.5 kN
(the recorded first scored severance); (3) the payload's heading extreme during the squall
(50-70 s); (2) the mission's maximum tension.  The end hold returns to pause 2's key frame and carries
the end card (the clip's takeaway).  Two reading holds (clock "paused"), for reading time and not for any
event, repeat the first and the last 10 ms row.

Narration: every caption, the title card's question and watch-for line, and the end card are complete
sentences; captions define their terms in words and use no symbols.

Asserted before any frame is drawn (Manifest.check)
---------------------------------------------------
* the pickle holds the 40 seeds 5001-5040; seed 5008's outcome: no severance (recording mode, so
  alive = True throughout), completed, no closure, 30 marks, equal to phase5_results.json;
* the marks' tuple order matches truth.marks field by field;
* log grids: 10 ms state from t = 0, 1 ms cable log, state_time == event_time[::10];
* geometry: state[0] equals mission.initial_state(MissionSpec()); the chord drawn between the
  attachment points of common.mission_geometry() equals rest length + the logged elongation at
  every 10 ms row (so the drawn cables are the plant's chords);
* the plant tension recomputed from the 1 ms log (common.tension) reproduces max_tension,
  max_tension_time and stress.per_mission_max for seed 5008; every mark's T_peak is reproduced
  by the first local maximum of the recomputed tension after t_up (gives each peak's time);
* the 4.5 kN level: stress.threshold == 55th percentile of the 40 per-mission maxima rounded to
  100 N; 18 of 40 missions exceed it;
* the first 1 ms sample above 4.5 kN belongs to the snap P6-T0 records as seed 5008's first
  severance (cable 4, t_up 58.743 s, 13.42 kN);
* caption 5: the payload's heading swings about 120 deg to starboard (squall onset to its extreme
  in the squall), the payload moves 89 m along the squall direction over 50-70 s, and the fan
  collapses for good (closest tug centres < 1 m from 62 s in > 99 % of rows; spread < 5 m from
  65 s against 17.8 m in the design fan); the spec addendum sentence is present;
* caption 6: all five slack from 71 s; the first re-engagement of each cable spans 76.1-77.3 s;
* caption 7: every mark's v_return > 0 (the chord lengthens at re-engagement: the line's ends move apart);
* caption 1: the pretension is the steady towing pull (fleet.operating_point: every cable at T0; the mission
  schedules' thrusts are that operating point's; at t = 0 no cable carries tension);
* schedule numbers, squall amplitude and background weather against MissionSpec /
  calibrate_squall and the Phase 5 declaration; Z against the P1 record;
* caption timing (end card included): <= 2 lines, >= 4 s on screen, <= 2.5 words/s; the title card
  stays up for its two sentences' reading time; the reading holds repeat only the first and last row.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon

from tether.analysis.v2.present import common as C
from tether.campaign.mission import (CALIBRATION_SEEDS, MissionSpec, calibrate_squall, dogleg_offset,
                                     initial_state, raised_cosine, squall_envelope, thrust_envelope)
from tether.physics import constants as K

NAME = "setting"
SEED = 5008
SPEED = 4.0                                   # playback: simulated seconds per video second
PKL = "records/phase5/cache/phase5_missions.pkl"
RES5 = "records/phase5/phase5_results.json"
SPEC_MD = "records/phase5/phase5_spec.md"
TICKS = "records/v2/phase5/p5_t2prime_ticks.npz"
RES_T2 = "records/v2/phase5/p5_t2prime_results.json"
REC_T0 = "records/v2/phase6/p6_t0_results.json"
REC_Z = "records/v2/phase1/scripted_results.json"
CACHE_FILE = C.CACHE / f"setting_{SEED}.npz"
MARK_ORDER = ("cable", "t_up", "depth", "v_return", "T_peak", "dwell")   # phase5.mission_job

# seconds on screen.  TITLE_S is the title card's minimum (it stays up for its sentences' reading time);
# HOLD0_S / HOLDZ_S are reading holds on the first and the last 10 ms row (clock "paused"); the pauses
# are sized so each caption that spans them meets the reading rate below; END_S carries the end card.
TITLE_S, HOLD0_S, PAUSE1_S, PAUSE3_S, PAUSE2_S, HOLDZ_S, END_S = 3.5, 5.4, 5.7, 6.25, 5.8, 2.4, 6.5
WORDS_PER_S = 2.5
TITLE_QUESTION = "What happens to towing cables in a squall?"
TITLE_WATCH = "Watch slack cables snap back taut."
END_CARD = "Slack cables snapping back taut are the failure mode\nstudied here; this snap reached {T:.1f} kN."
TICKER_ROWS = 6
SPARK_S = 8.0                                 # pause 1/2 inset: seconds of tension history shown
HEAD_S = 12.0                                 # pause 3 inset: seconds of payload heading shown
CLOSE_M = 1.0                                 # "bunched": two tug centres closer than this
CAPTION_Y = 0.078                             # caption band centre (clears the footer with a math caption)
ADDENDUM = "the formation rotates about 90 deg and drifts sideways"


# ============================================================================ record, checks

def _json(path):
    return json.loads((C.REPO / path).read_text())


def tug_distances(S):
    """Closest and farthest pair of tug centres per 10 ms row (m)."""
    V = S[:, 3:18].reshape(-1, 5, 3)[:, :, :2]
    dd = np.stack([np.hypot(*(V[:, i] - V[:, j]).T) for i, j in itertools.combinations(range(5), 2)], axis=1)
    return dd.min(axis=1), dd.max(axis=1)


def load_and_check(man: C.Manifest) -> dict:
    """Read the record, assert everything the clip relies on, register the fixed numbers."""
    spec = MissionSpec()
    for p in (PKL, RES5, SPEC_MD, TICKS, RES_T2, REC_T0, REC_Z):
        man.source(p)
    res5, rest2, rec_t0, rec_z = _json(RES5), _json(RES_T2), _json(REC_T0), _json(REC_Z)

    # ---- selection rule (clip C4's episode)
    tk = np.load(C.REPO / TICKS)
    sq0 = spec.squall_centre - 0.5 * spec.squall_plateau - spec.squall_rise
    squall_end = spec.squall_centre + 0.5 * spec.squall_plateau + spec.squall_rise
    keep = ~tk["bounce"] & ~tk["censored_H1"]
    fa = keep & (tk["forecast_linearized_H1"] >= 0.9) & ~tk["label_H1"]
    cfa = fa & tk["coupled_H1"]
    win = cfa & (tk["time"] >= squall_end) & (tk["time"] <= spec.deceleration_start)
    seeds, counts = np.unique(tk["seed"][win], return_counts=True)
    order = np.argsort(counts)[::-1]
    coin = rest2["reports"]["linearized_H1"]["coincidence"]
    man.check("selection counts match the P5-T2' record", int(fa.sum()) == coin["top_bin_misses"] == 25
              and int(cfa.sum()) == coin["top_bin_misses_coupled"] == 24,
              f"false alarms {int(fa.sum())}, coupled {int(cfa.sum())}")
    man.check("selection rule picks seed 5008 (unique maximum)", int(win.sum()) == 18 and
              int(seeds[order[0]]) == SEED and counts[order[0]] > counts[order[1]],
              f"window {squall_end:g}-{spec.deceleration_start:g} s: " +
              ", ".join(f"{s}:{c}" for s, c in zip(seeds[order], counts[order])))
    n_sel, n_win = int(counts[order[0]]), int(win.sum())

    # ---- the mission record
    with open(C.REPO / PKL, "rb") as fh:
        missions = pickle.load(fh)
    man.check("record holds the 40 calibration seeds 5001-5040",
              [m["seed"] for m in missions] == list(CALIBRATION_SEEDS))
    m = next(x for x in missions if x["seed"] == SEED)
    tr, out, marks = m["truth"], m["outcome"], m["marks"]
    # post hoc (manifest only, not on screen): how common bunching is across the 40 missions
    bunch = [float(np.mean(tug_distances(x["truth"].state)[0] < CLOSE_M)) for x in missions]
    n_bunch, med_bunch = int(sum(b > 0.0 for b in bunch)), float(np.median(bunch))
    man.check("recording mission: no cable severed, alive throughout", tuple(out.severances) == ()
              and spec.cable_mode == "recording", f"severances {out.severances}")
    man.check("mission completed without formation closure", out.completed and out.closure is None
              and abs(out.end_time - spec.duration) < 1e-9)
    rec_out = next(o for o in res5["missions"]["outcomes"] if o["seed"] == SEED)
    man.check("outcome agrees with phase5_results.json", rec_out["mark_count"] == out.mark_count ==
              len(marks) == len(tr.marks) and rec_out["severances"] == [], f"{len(marks)} marks")
    ok = all(abs(getattr(r, f) - mk[j]) == 0.0 for mk, r in zip(marks, tr.marks)
             for j, f in enumerate(MARK_ORDER))
    man.check("mark tuple order (cable, t_up, depth, v_return, T_peak, dwell) matches truth.marks", ok)

    st, S, et = tr.state_time, tr.state, tr.event_time
    e, rate = tr.elongation, tr.rate
    man.check("log grids: 10 ms state from 0, 1 ms cable log, shared clock",
              S.shape == (13000, 36) and e.shape == (130000, 5) and st[0] == 0.0
              and np.allclose(np.diff(st), 0.01) and np.allclose(np.diff(et), 1e-3)
              and np.array_equal(et[::10], st))
    geometry = C.mission_geometry()
    man.check("state[0] is the design fan at rest (mission.initial_state)",
              np.array_equal(S[0, :18], initial_state(spec)[:18]))
    chord = np.empty((S.shape[0], 5))
    for i, row in enumerate(S):
        lp, vp = C.unpack_state(row)
        a, b = C.attachment_points(lp, vp, geometry)
        chord[i] = np.hypot(*(b - a).T)
    dev = float(np.max(np.abs(chord - K.CABLE_REST_LENGTH - e[::10])))
    man.check("drawn chords = rest length + logged elongation at every 10 ms row", dev < 1e-9,
              f"max |chord - L - e| = {dev:.2e} m")

    alive = np.ones_like(e, dtype=bool)
    T = C.tension(e, rate, alive)
    k_max = np.unravel_index(np.argmax(T), T.shape)
    t_max, T_max, c_max = float(et[k_max[0]]), float(T[k_max]), int(k_max[1])
    per_mission = res5["stress"]["per_mission_max"]
    man.check("recomputed plant tension reproduces max_tension and its time",
              T_max == m["max_tension"] and t_max == m["max_tension_time"]
              and per_mission[list(CALIBRATION_SEEDS).index(SEED)] == T_max,
              f"{T_max:.3f} N at {t_max:.3f} s on cable {c_max}")

    # per-mark peaks and their times (first local maximum after t_up, as the plant's tracker)
    rows = []
    for cab, t_up, depth, v_ret, T_pk, dwell in marks:
        j = int(np.searchsorted(et, t_up, side="right"))
        while T[j + 1, cab] >= T[j, cab]:
            j += 1
        if not abs(T[j, cab] - T_pk) <= 1e-9 * max(1.0, T_pk):
            man.check(f"mark peak reproduced (cable {cab}, t_up {t_up:.4f})", False, f"{T[j, cab]} vs {T_pk}")
        rows.append((float(et[j]), int(cab), float(t_up), float(T_pk), float(depth), float(dwell), float(v_ret)))
    man.check("every mark's T_peak = first local maximum of the recomputed tension after t_up", True,
              f"{len(rows)} marks")
    mk = np.array(sorted(rows), dtype=float)   # columns: t_peak, cable, t_up, T_peak, depth, dwell, v_return
    man.check("caption 7: at every re-engagement the chord lengthens (record v_return > 0), so the line's ends "
              "move apart as it comes taut", bool(np.all(mk[:, 6] > 0.0)),
              f"v_return {mk[:, 6].min():.4f}-{mk[:, 6].max():.3f} m/s over {len(mk)} marks")

    # the 4.5 kN level (a scoring threshold) and the scored first severance
    level = float(res5["stress"]["threshold"])
    p55 = float(np.percentile(per_mission, 55))
    n_above = int(np.sum(np.array(per_mission) > level))
    man.check("4.5 kN = 55th percentile of the 40 per-mission maxima, rounded to 100 N",
              level == 4500.0 and round(p55 / 100.0) * 100.0 == level and n_above == 18,
              f"p55 {p55:.1f} N; {n_above}/40 above")
    k_first = int(np.argmax(T.max(axis=1) > level))
    c_first = int(np.argmax(T[k_first] > level))
    fs = [x["tags"] for x in rec_t0["excursions"] if x["tags"]["seed"] == SEED and x["tags"]["first_severance"]]
    first = mk[(mk[:, 1] == c_first) & (mk[:, 2] <= et[k_first]) & (mk[:, 0] >= et[k_first])]
    man.check("first 1 ms sample above 4.5 kN lies in the snap P6-T0 records as the first severance",
              len(fs) == 1 and len(first) == 1 and fs[0]["cable"] == c_first
              and fs[0]["t_up"] == first[0, 2] and fs[0]["T_peak"] == first[0, 3],
              f"cable {c_first}, first sample {et[k_first]:.3f} s, t_up {first[0, 2]:.4f} s, peak {first[0, 3]:.1f} N")
    first = first[0]

    # ---- caption 5: what the fleet does in the squall (swing, drift, collapse)
    r0, r1 = int(round(sq0 / 0.01)), int(round(squall_end / 0.01))
    assert abs(st[r0] - sq0) < 1e-9 and abs(st[r1] - squall_end) < 1e-9
    head = np.degrees(S[:, 2])
    p3 = r0 + int(np.argmin(head[r0: r1 + 1]))           # the payload's heading extreme in the squall
    swing = float(head[p3] - head[r0])                    # < 0: clockwise = to starboard
    head_range = float(head[r0: r1 + 1].max() - head[p3])
    tug_swing = -(np.degrees(S[p3, 5:18:3]) - np.degrees(S[r0, 5:18:3]))   # > 0: to starboard
    man.check("caption 5: the payload swings to starboard, about 120 deg (squall onset -> extreme in the squall)",
              head[p3] < 0.0 and swing < 0.0 and round(-swing, -1) == 120.0 and round(head_range, -1) == 120.0,
              f"heading {head[r0]:+.2f} deg at {st[r0]:.2f} s -> {head[p3]:+.2f} deg at {st[p3]:.2f} s "
              f"(change {swing:+.1f} deg; range over the squall {head_range:.1f} deg); tugs' headings over the same "
              f"interval turn " + ", ".join(f"{x:.1f}" for x in tug_swing) + " deg to starboard")
    u_sq = np.array([math.cos(spec.squall_direction), math.sin(spec.squall_direction)])
    drift_vec = S[r1, :2] - S[r0, :2]
    drift = float(drift_vec @ u_sq)
    man.check("caption 5: the payload moves 89 m along the squall direction over the squall",
              round(drift) == 89 and abs(float(drift_vec @ np.array([-u_sq[1], u_sq[0]]))) < 0.1 * drift,
              f"displacement {sq0:.0f}-{squall_end:.0f} s = ({drift_vec[0]:.2f}, {drift_vec[1]:.2f}) m; "
              f"{drift:.2f} m downwind")
    closest, spread = tug_distances(S)
    k_col = int(np.argmax(closest < CLOSE_M))
    share = float(np.mean(closest[k_col:] < CLOSE_M))
    k65 = int(round(65.0 / 0.01))
    spread_after, design_spread = float(spread[k65:].max()), float(spread[0])
    spread_pre, closest_pre = float(spread[:r0].min()), float(closest[:r0].min())
    man.check("caption 5: the fan collapses in the squall and does not re-form (tugs bunched from 62 s to the end)",
              61.5 <= st[k_col] < 62.5 and st[k_col] < st[p3] and share > 0.99 and spread_after < 5.0
              and design_spread > 17.0 and spread_pre > 2.0 * spread_after and closest_pre > 2.0
              and np.all(closest[:k_col] >= CLOSE_M),
              f"before the squall (0-{sq0:.0f} s) spread >= {spread_pre:.2f} m, closest pair >= {closest_pre:.2f} m; "
              f"first closest pair < {CLOSE_M:g} m at {st[k_col]:.2f} s, then in {100 * share:.2f} % of rows; "
              f"spread <= {spread_after:.2f} m from 65 s; {design_spread:.2f} m at t = 0 (design fan)")
    addendum = " ".join((C.REPO / SPEC_MD).read_text().split())
    man.check("phase5_spec.md addendum records the squall-only rotation (cited for 'the squall swings')",
              ADDENDUM in addendum and "Addendum (2026-09-12" in addendum)

    # ---- caption 6: the all-slack episode and the re-engagements that end it
    allslack = np.all(e <= 0.0, axis=1)
    idx = np.flatnonzero(allslack & (et > 60.0))
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    longest = max(runs, key=len)
    t_s0, t_s1 = float(et[longest[0]]), float(et[longest[-1]])
    after = mk[mk[:, 2] > t_s1 - 0.01]
    first_re = {int(c): float(after[after[:, 1] == c, 2].min()) for c in range(5)}
    re_lo, re_hi = min(first_re.values()), max(first_re.values())
    man.check("caption 6: all five slack from 71 s; each cable's first re-engagement spans 76.1-77.3 s",
              71.0 <= t_s0 < 71.5 and 76.0 <= t_s1 < 76.2 and f"{re_lo:.1f}" == "76.1" and f"{re_hi:.1f}" == "77.3",
              f"all slack {t_s0:.3f}-{t_s1:.3f} s; first re-engagements " +
              ", ".join(f"cable {c} {v:.3f} s" for c, v in sorted(first_re.items(), key=lambda x: x[1])))

    # schedule, squall, weather, Z
    cal = calibrate_squall(spec)
    decl = res5["declarations"]["mission"]
    man.check("MissionSpec agrees with the Phase 5 declaration", spec.arc_half_angle == 0.55
              and spec.pretension == 1000.0 and "0.55 rad" in decl and "T0 = 1 kN" in decl
              and f"{cal.vessel_amplitude / 1e3:.1f} kN per vessel" in decl,
              f"squall {cal.vessel_amplitude:.1f} N per vessel, {cal.load_amplitude:.1f} N on the payload")
    from tether.campaign.mission import mission_geometry, mission_schedules
    from tether.physics.fleet import operating_point
    op = operating_point(mission_geometry(spec), spec.pretension, spec.drag_law)
    man.check("caption 1: the pretension is the steady towing pull (the schedules' thrusts are the steady tow with "
              "every cable at T0; the mission starts at rest, chords at rest length)",
              bool(np.all(op.tensions == spec.pretension)) and np.array_equal(mission_schedules(spec).thrusts, op.thrusts)
              and np.all(C.tension(e[0], rate[0], np.ones(5, bool)) == 0.0),
              f"operating point: {op.speed:.3f} m/s, cable tensions {sorted(set(op.tensions.tolist()))} N")
    man.check("background weather: MissionSpec student_t3, common-mode, intensity 1.0, as declared",
              spec.weather_distribution == "student_t3" and spec.weather_direction == "common"
              and spec.weather_scale == 1.0 and "common-mode t3 background weather at intensity 1.0" in decl)
    Z = float(rec_z["cell_b_impact"]["Z_repin"]["Z_N_s_per_m"])
    man.check("Z = 7.99 kN s/m (P1 pair re-pin)", f"{Z / 1e3:.2f}" == "7.99", f"{Z:.2f} N s/m")

    maxima = np.sort(per_mission)[::-1]
    rank = int(np.flatnonzero(maxima == T_max)[0]) + 1

    return dict(spec=spec, geometry=geometry, st=st, S=S, et=et, e=e, T=T, marks=mk, level=level,
                T_max=T_max, t_max=t_max, c_max=c_max, first=first, t_slack=(t_s0, t_s1), cal=cal, Z=Z,
                rank=rank, n_above=n_above, n_sel=n_sel, n_win=n_win, median_max=float(np.median(per_mission)),
                sq0=sq0, squall_end=squall_end, per_mission=per_mission, head=head, p3=p3, swing=swing,
                tug_swing=tug_swing,
                head_range=head_range, drift=drift, drift_vec=drift_vec, spread=spread, closest=closest,
                t_col=float(st[k_col]), share=share, spread_after=spread_after, design_spread=design_spread,
                spread_pre=spread_pre, closest_pre=closest_pre, n_bunch=n_bunch, med_bunch=med_bunch,
                bunch=bunch,
                first_re=first_re, re_lo=re_lo, re_hi=re_hi)


# ============================================================================ frame plan

def frame_plan(d: dict):
    """(row, mode) per frame: a reading hold on the first row, x4 over the 10 ms rows, three pauses,
    a reading hold on the last row, then the end hold on pause 2's key frame (the end card)."""
    n_rows = d["S"].shape[0]
    step = SPEED / C.FPS / 0.01
    regular = [int(round(k * step)) for k in range(int(math.floor((n_rows - 1) / step)) + 1)]
    if regular[-1] != n_rows - 1:
        regular.append(n_rows - 1)
    p1 = int(round(d["first"][0] / 0.01))                 # row nearest the first scored snap's peak
    p2 = int(round(d["t_max"] / 0.01))                    # the mission's maximum (on the 10 ms grid)
    p3 = d["p3"]                                          # the payload's heading extreme in the squall
    assert abs(d["st"][p2] - d["t_max"]) < 1e-9 and p1 < p3 < p2
    rows = sorted(set(regular) | {p1, p2, p3})
    pauses = {p1: ("pause1", PAUSE1_S), p3: ("pause3", PAUSE3_S), p2: ("pause2", PAUSE2_S)}
    plan = [(rows[0], "hold")] * int(round(HOLD0_S * C.FPS))     # reading hold on t = 0 (same row, held)
    for r in rows:
        plan.append((r, "play"))
        if r in pauses:
            mode, secs = pauses[r]
            plan += [(r, mode)] * int(round(secs * C.FPS))
    plan += [(rows[-1], "hold")] * int(round(HOLDZ_S * C.FPS))   # reading hold on the mission's last row
    plan += [(p2, "end")] * int(round(END_S * C.FPS))
    return plan, p1, p2, p3


def captions(d: dict):
    """(sim-time start, text) -- the narration; start times are simulation seconds.  The last entry is
    the end card (the clip's takeaway): it has no start time and is shown only on the end hold."""
    spec, first = d["spec"], d["first"]
    return [
        # pretension = the tension every cable carries in the design steady tow (fleet.operating_point; the
        # mission starts at rest with the chords at rest length, so caption 1 says "steady towing pull")
        (0.0, f"Five tugs tow one payload on {K.CABLE_REST_LENGTH:.0f} m cables\n"
              f"whose pretension, the steady towing pull, is {spec.pretension / 1e3:.0f} kN."),
        (7.5, "Cables pull only when stretched,\nand random weather pushes the payload and tugs."),
        (28.5, f"The plan commands a {math.degrees(spec.dogleg_angle):.0f}° port turn during "
               f"{spec.dogleg_start:.0f}–{spec.dogleg_end:.0f} s;\n"
               f"a squall blows from port during {d['sq0']:.0f}–{d['squall_end']:.0f} s."),
        (58.0, f"Cable {int(first[1])} snaps back taut at {first[3] / 1e3:.1f} kN,\n"
               f"above the {d['level'] / 1e3:.1f} kN level where severance is scored."),
        (float(d["st"][d["p3"]]) - 1e-6,
         f"In the squall the payload swings about {abs(round(d['swing'], -1)):.0f}° to starboard\n"
         f"and drifts {d['drift']:.0f} m downwind; the tugs' fan collapses for good."),
        (72.0, f"All five cables are slack from {math.floor(d['t_slack'][0]):.0f} s, then snap back\n"
               f"at {d['re_lo']:.1f}–{d['re_hi']:.1f} s, peaking at {d['T_max'] / 1e3:.1f} kN."),
        # the closing speed is defined in words where it first appears: at re-engagement the chord rate is
        # positive (the record's v_return), so the line's ends move apart; the relation T0 + Z V_up and
        # Z = 7.99 kN s/m are left to the next slide (s03_perline), which gives them with every symbol defined
        (78.0, "Per-line practice predicts a snap's peak from the line's closing speed,\n"
               "how fast its ends move apart as it comes taut."),
        (112.0, "This study asks whether the closing speed still belongs\n"
                "to one line when lines share one payload."),
        (math.inf, END_CARD.format(T=d["T_max"] / 1e3)),
    ]


def caption_schedule(plan, d):
    """Caption index per frame: by the drawn row's time (the last row's hold keeps the caption running
    at the mission's end); the end hold shows the end card, the last caption."""
    caps = captions(d)
    starts = [c[0] for c in caps]
    assert all(a < b for a, b in zip(starts, starts[1:])) and starts[-1] == math.inf
    idx = [len(caps) - 1 if mode == "end" else int(np.searchsorted(starts, d["st"][r], side="right")) - 1
           for r, mode in plan]
    return caps, idx


def title_words() -> int:
    """Words of the title card's two sentences (the reading-time rule applies to them)."""
    return _words(TITLE_QUESTION) + _words(TITLE_WATCH)


def title_frames() -> int:
    """Title-card frames: at least TITLE_S, and the reading time of its question and watch-for sentences."""
    return int(math.ceil(max(TITLE_S, title_words() / WORDS_PER_S) * C.FPS - 1e-9))


def _words(text: str) -> int:
    plain = text.replace("$", " ").replace("\\approx", "~").replace("\\,", " ")
    return len([w for w in plain.split() if any(ch.isalnum() for ch in w)])


# ============================================================================ drawing helpers

def cable_key(fig, y: float, x: float):
    """This clip's cable legend (common.cable_key says 'width ∝ tension'; the width is an offset plus
    a term that saturates at common.WIDTH_FULL_N, so the wording here says so)."""
    handles = [Line2D([], [], color=C.TAUT, lw=4,
                      label=f"taut (wider with tension, up to {C.WIDTH_FULL_N / 1e3:.0f} kN)"),
               Line2D([], [], color=C.SLACK, lw=2, ls=(0, (3.0, 2.2)), label="carrying no tension")]
    fig.legend(handles=handles, loc="center", ncol=2, bbox_to_anchor=(x, y), fontsize=C.FS_SMALL,
               handlelength=2.6, columnspacing=1.6)


def highlight_cable(ax, lp, vp, geometry, cab):
    """Pause highlight: an orange halo under the cable's chord and the tug's hull outlined in orange."""
    a, b = C.attachment_points(lp, vp, geometry)
    ax.plot([a[cab][0], b[cab][0]], [a[cab][1], b[cab][1]], color=C.ACCENT, lw=12, alpha=0.4,
            solid_capstyle="round", zorder=2.5)
    ax.add_patch(Polygon(C.body_polygon(vp[cab], C.hull()), closed=True, fill=False, edgecolor=C.ACCENT,
                         lw=2.8, zorder=4.6))


def _inside(P, lo, hi, inset):
    return bool(np.all(P >= lo + inset) and np.all(P <= hi - inset))


def _clear(P, lo, hi, keepout):
    """True when no point of P falls in any keep-out rectangle (axes fractions x0, x1, y0, y1)."""
    F = (P - lo) / (hi - lo)
    for x0, x1, y0, y1 in keepout:
        if np.any((F[:, 0] >= x0) & (F[:, 0] <= x1) & (F[:, 1] >= y0) & (F[:, 1] <= y1)):
            return False
    return True


_PERMS = list(itertools.permutations(range(5)))
_LABEL_SIDE = {"side": None}          # the side used on the previous frame (keeps labels from jumping)


def _label_row(anchors, hulls, u, gap, clear):
    """Five label slots in a row across ``u``, just beyond the cluster, assigned to the tugs (anchor =
    hull centre) by the minimum total distance (a minimum-length matching has no crossing segments)."""
    n = len(anchors)
    cen = 0.5 * (hulls.min(0) + hulls.max(0))
    w = np.array([-u[1], u[0]])
    c_row = cen + (float(np.max((hulls - cen) @ u)) + clear) * u
    slots = c_row + ((np.arange(n) - 0.5 * (n - 1)) * gap)[:, None] * w
    dist = np.hypot(*(anchors[:, None, :] - slots[None, :, :]).transpose(2, 0, 1))
    best = min(_PERMS, key=lambda pm: sum(dist[i, pm[i]] for i in range(n)))
    return slots[list(best)], float(sum(dist[i, best[i]] for i in range(n)))


def _open_layout(vp, bows, heads, lo, hi, keepout, gap, ahead, inset):
    """Open formation: each number next to its own hull -- ahead of the bow, else beside the hull,
    else astern -- inside the view, clear of the view's text, ``gap`` m from the other numbers and
    clear of the other hulls.  None when some tug has no such place (then the row layout is used)."""
    n = len(vp)
    centres = np.array([v[:2] for v in vp])
    if min(float(np.hypot(*(centres[i] - centres[j]))) for i, j in itertools.combinations(range(n), 2)) < 2.2:
        return None                                        # bunched: use the row layout with leaders
    pos = []
    for i in range(n):
        h = heads[i]
        p = np.array([-h[1], h[0]])
        cands = [bows[i] + ahead * h, centres[i] + 1.7 * p, centres[i] - 1.7 * p,
                 centres[i] - (0.5 * C.L_HULL + ahead + 0.3) * h]
        for q in cands:
            if (_inside(q[None], lo, hi, inset) and _clear(q[None], lo, hi, keepout)
                    and all(np.hypot(*(q - r)) >= gap for r in pos)
                    and all(np.hypot(*(q - centres[j])) >= 1.4 for j in range(n) if j != i)):
                pos.append(q)
                break
        else:
            return None
    return np.array(pos)


def vessel_labels(ax, vp, lo, hi, keepout=(), highlight=None, gap=2.0, ahead=1.2, inset=1.3, clear=1.8):
    """Vessel numbers.  In an open formation each sits next to its own hull (``_open_layout``).  When
    the hulls bunch and that fails, the numbers are laid out in a row outside the cluster, with dark
    leader lines that do not cross, on a side (ahead, port, starboard, astern) where the row fits in
    the view clear of the view's text: the previous frame's side if it still fits, else the fitting
    side with the shortest leaders."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = len(vp)
    bows = np.array([C.body_polygon(v, np.array([[0.5 * C.L_HULL, 0.0]]))[0] for v in vp])
    heads = np.array([[math.cos(v[2]), math.sin(v[2])] for v in vp])
    home = bows + ahead * heads
    pos, leaders = _open_layout(vp, bows, heads, lo, hi, keepout, gap, ahead, inset), False
    if pos is not None:
        _LABEL_SIDE["side"] = None
    else:
        polys = [C.body_polygon(v, C.hull()) for v in vp]
        hulls = np.vstack(polys)
        h = heads.sum(0)
        h = h / np.hypot(*h) if np.hypot(*h) > 1e-6 else np.array([1.0, 0.0])
        p = np.array([-h[1], h[0]])
        sides = {"ahead": h, "port": p, "starboard": -p, "astern": -h}
        fits, inside_only = {}, {}
        for name, u in sides.items():
            cand, length = _label_row(np.array([v[:2] for v in vp]), hulls, u, gap, clear)
            if _inside(cand, lo, hi, inset):
                inside_only[name] = (length, cand)
                if _clear(cand, lo, hi, keepout):
                    fits[name] = (length, cand)
        prev = _LABEL_SIDE["side"]
        pool = fits or inside_only
        if prev in pool:
            name = prev
        elif pool:
            name = min(pool, key=lambda k: pool[k][0])
        else:
            name = None
        if name is None:
            pos = np.clip(home, lo + inset, hi - inset)
        else:
            pos = pool[name][1]
        _LABEL_SIDE["side"] = name
        leaders = True
    for i in range(n):
        if leaders:
            P = C.body_polygon(vp[i], C.hull())
            a = P[int(np.argmin(np.hypot(*(P - pos[i]).T)))]     # hull vertex nearest the label
            ax.plot([a[0], pos[i][0]], [a[1], pos[i][1]], color=C.INK, lw=1.0, alpha=0.8, zorder=5)
        hl = highlight == i
        ax.text(*pos[i], str(i), fontsize=C.FS_SMALL, color=C.INK, weight="bold" if hl else "normal",
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=C.ACCENT if hl else C.FAINT,
                          lw=2.2 if hl else 0.8, alpha=0.95))
    return pos


# ============================================================================ camera and keep-outs

FLEET_AXES = [0.025, 0.425, 0.50, 0.43]
FLEET_BOX = (FLEET_AXES[2] * C.W) / (FLEET_AXES[3] * C.H)
# keep-out rectangles in fleet-axes fractions (x0, x1, y0, y1): text and the inset drawn over the view
KO_NOTE = (0.705, 1.0, 0.56, 1.0)      # corner note (view, tug spread, contact caveat / collapse)
KO_PAUSE = (0.0, 0.62, 0.925, 1.0)     # "paused at ..." line
KO_SQUALL = (0.0, 0.26, 0.45, 0.93)    # squall arrow and its label
KO_INSET = (0.70, 1.0, 0.0, 0.47)      # pause inset with its title and tick labels
KO_SCALE = (0.0, 0.20, 0.0, 0.15)      # 5 m scale bar


def camera(d, rows, box=FLEET_BOX):
    """Common half-span over every drawn row, fitted to the box; smoothed centre per row."""
    uniq = np.array(sorted(set(rows)))
    centres, half = C.fleet_bounds(d["S"][uniq], d["geometry"], margin=3.0)
    cam = C.smooth_camera(centres, window=31)
    return {int(r): cam[k] for k, r in enumerate(uniq)}, C.fit_aspect(half, box)


def keepouts(d, t: float, mode: str):
    ko = [KO_NOTE, KO_SCALE]
    if mode not in ("play", "hold"):                   # a reading hold draws the play frame, held
        ko += [KO_PAUSE, KO_INSET]
    if float(squall_envelope(d["spec"], t)) > 1e-3:
        ko.append(KO_SQUALL)
    return ko


def bodies_clear(d, cam, half, row: int, mode: str) -> bool:
    """No body vertex under the fleet view's text or the pause inset."""
    from tether.physics import fleet as F
    lp, vp = C.unpack_state(d["S"][row])
    P = np.vstack([C.body_polygon(lp, F.pentagon_vertices())] + [C.body_polygon(v, C.hull()) for v in vp])
    return _clear(P, cam[row] - half, cam[row] + half, keepouts(d, float(d["st"][row]), mode))


# ============================================================================ the scene

class Scene:
    """Static layout drawn once; ``draw`` updates the time-varying artists for one frame."""

    FLEET = FLEET_AXES
    TRACK = [0.548, 0.60, 0.125, 0.235]
    SCHED = [0.135, 0.318, 0.84, 0.078]
    TENS = [0.135, 0.158, 0.84, 0.122]
    INSET = [0.405, 0.475, 0.108, 0.115]

    def __init__(self, fig, d, man, p1, p2, p3):
        self.fig, self.d, self.man, self.p1, self.p2, self.p3 = fig, d, man, p1, p2, p3
        _LABEL_SIDE["side"] = None
        self.label_side = {}
        spec = d["spec"]
        self.t_all = d["st"]
        C.title(fig, "The operation: one squall-passage mission",
                f"Fan formation (design geometry), pretension T0 = {spec.pretension / 1e3:.0f} kN · recording run: "
                f"severance is scored, no cable is cut · seed {SEED}")
        fig.text(0.025, 0.866,
                 f"Seed {SEED} was chosen by rule: it holds the most ({d['n_sel']} of {d['n_win']}) of the per-line model's "
                 f"confident coupled false alarms at {d['squall_end']:.0f}–{spec.deceleration_start:.0f} s (see the "
                 f"prediction movie). Its peak ranks {self._ordinal(d['rank'])} of 40, so it is not typical.",
                 fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="center")
        C.footer(fig, f"Plant state: {PKL}, seed {SEED} (10 ms state, 1 ms cable log, record marks) · "
                      "v1 Phase 5 missions, exploratory continuation · clock from mission start")
        cable_key(fig, y=0.405, x=0.275)

        self.ax_f = fig.add_axes(self.FLEET)
        self.box = (self.FLEET[2] * C.W) / (self.FLEET[3] * C.H)

        # ---- payload track
        ax = self.ax_t = fig.add_axes(self.TRACK)
        S = d["S"]
        ax.plot(S[:, 0], S[:, 1], color=C.FAINT, lw=1.5, zorder=1)
        self.track_line, = ax.plot([], [], color=C.INK, lw=1.8, zorder=2)
        self.track_dot, = ax.plot([], [], "o", color=C.ACCENT, ms=7, mec=C.INK, mew=0.8, zorder=3)
        ax.plot([S[0, 0]], [S[0, 1]], "s", color=C.INK, ms=4, zorder=3)
        ax.text(S[0, 0] + 4, S[0, 1] + 3, "start", fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="bottom")
        lo, hi = S[:, :2].min(0) - 8.0, S[:, :2].max(0) + 8.0
        mid, hs = 0.5 * (lo + hi), C.fit_aspect(0.5 * (hi - lo), (self.TRACK[2] * C.W) / (self.TRACK[3] * C.H))
        ax.set_xlim(mid[0] - hs[0], mid[0] + hs[0]); ax.set_ylim(mid[1] - hs[1], mid[1] + hs[1]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(C.FAINT)
        x0, y0 = ax.get_xlim()[0] + 4.0, ax.get_ylim()[0] + 5.0
        ax.plot([x0, x0 + 20.0], [y0, y0], color=C.INK, lw=2.2, solid_capstyle="butt")
        ax.text(x0 + 10.0, y0 + 2.0, "20 m", fontsize=C.FS_TINY, ha="center", va="bottom")
        ax.set_title("payload track (world frame)", fontsize=C.FS_TINY, color=C.MUTED, pad=4)

        # ---- cable status panel
        xs = 0.695
        fig.text(xs, 0.835, "Cables at this instant", fontsize=C.FS_SMALL, weight="bold", color=C.INK,
                 ha="left", va="center")
        self.status = []
        for c in range(5):
            y = 0.795 - 0.038 * c
            fig.text(xs, y, f"cable {c}", fontsize=C.FS_SMALL, color=C.INK, ha="left", va="center")
            self.status.append(fig.text(xs + 0.07, y, "", fontsize=C.FS_SMALL, family="DejaVu Sans Mono",
                                        color=C.INK, ha="left", va="center"))
        fig.text(xs, 0.60, "slack = chord at or below rest length (12 m)", fontsize=C.FS_TINY, color=C.MUTED,
                 ha="left", va="center")

        # ---- mark ticker
        xt = 0.548
        fig.text(xt, 0.568, "Re-engagement marks from the record (latest first)", fontsize=C.FS_SMALL,
                 weight="bold", color=C.INK, ha="left", va="center")
        hdr = dict(fontsize=C.FS_TINY, color=C.MUTED, va="center")
        cols = (xt + 0.062, xt + 0.098, xt + 0.19)             # right edge, centre, right edge
        fig.text(cols[0], 0.540, "peak time", ha="right", **hdr)
        fig.text(cols[1], 0.540, "cable", ha="center", **hdr)
        fig.text(cols[2], 0.540, "peak tension", ha="right", **hdr)
        self.tick_rows = []
        mono = dict(fontsize=C.FS_TINY, family="DejaVu Sans Mono", va="center")
        for j in range(TICKER_ROWS):
            y = 0.515 - 0.0205 * j
            self.tick_rows.append((fig.text(cols[0], y, "", ha="right", **mono),
                                   fig.text(cols[1], y, "", ha="center", **mono),
                                   fig.text(cols[2], y, "", ha="right", **mono)))
        self.tick_sum = fig.text(xt + 0.225, 0.515, "", fontsize=C.FS_TINY, color=C.INK, ha="left", va="top",
                                 linespacing=1.5)

        # ---- schedule strip
        ax = self.ax_s = fig.add_axes(self.SCHED)
        tt = np.linspace(0.0, spec.duration, 2601)
        sq0 = d["sq0"]
        sq1, sq2 = sq0 + spec.squall_rise, sq0 + spec.squall_rise + spec.squall_plateau
        lanes = [
            ("squall Ψ", squall_envelope(spec, tt),
             [(1.0, f"plateau {d['cal'].vessel_amplitude / 1e3:.1f} kN per tug, {d['cal'].load_amplitude / 1e3:.0f} kN "
                    f"on the payload, toward world −y"),
              (d["squall_end"] + 1.5, f"rise {sq0:.0f}–{sq1:.0f} s · plateau {sq1:.0f}–{sq2:.0f} s · "
                                      f"fall {sq2:.0f}–{d['squall_end']:.0f} s")], C.SLACK),
            ("commanded turn", dogleg_offset(spec, tt) / spec.dogleg_angle,
             [(1.0, f"heading offset 0 → {math.degrees(spec.dogleg_angle):.0f}° to port, "
                    f"{spec.dogleg_start:.0f}–{spec.dogleg_end:.0f} s (dogleg)")], C.OK),
            ("background weather", raised_cosine(tt / spec.weather_ramp_duration),
             [(10.0, f"random (t3, common-mode, {K.WEATHER_TIME_CONSTANT:.0f} s memory), "
                     f"ramped in 0–{spec.weather_ramp_duration:.0f} s")], C.MUTED),
            ("thrust", thrust_envelope(spec, tt),
             [(10.0, f"ramp 0–{spec.ramp_duration:.0f} s; deceleration {spec.deceleration_start:.0f}–{spec.duration:.0f} s")],
             C.TAUT),
        ]
        for j, (lab, env, notes, col) in enumerate(lanes):
            base = j + 0.12
            ax.fill_between(tt, base, base + 0.76 * env, color=col, alpha=0.18, lw=0)
            ax.plot(tt, base + 0.76 * env, color=col, lw=1.2)
            ax.text(-0.006, (j + 0.5) / len(lanes), lab, transform=ax.transAxes, ha="right", va="center",
                    fontsize=C.FS_TINY, color=C.INK)
            for xn, note in notes:
                ax.text(xn, j + 0.5, note, ha="left", va="center", fontsize=C.FS_TINY - 1, color=C.INK)
        ax.set_xlim(0.0, spec.duration); ax.set_ylim(0.0, len(lanes))
        ax.set_yticks([]); ax.set_xticks(np.arange(0, spec.duration + 1, 10)); ax.tick_params(labelbottom=False, length=2)
        for s in ("left", "right", "top"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(C.FAINT)
        ax.text(1.0, 1.03, "schedule: MissionSpec", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=C.FS_TINY, color=C.MUTED)
        self.cur_s = ax.axvline(0.0, color=C.ACCENT, lw=1.5, alpha=0.7, zorder=2)

        # ---- running max-tension strip
        ax = self.ax_T = fig.add_axes(self.TENS)
        T = d["T"]
        Tm = T.max(axis=1)
        # 10 ms bins ending at each state row (max-pooled, so no peak is lost when drawn)
        pooled = np.empty(S.shape[0])
        pooled[0] = Tm[0]
        pooled[1:] = Tm[1: 10 * (S.shape[0] - 1) + 1].reshape(S.shape[0] - 1, 10).max(axis=1)
        assert np.max(pooled) == d["T_max"]
        self.pooled = pooled / 1e3
        self.Tm = Tm
        self.trace, = ax.plot([], [], color=C.TAUT, lw=1.3, zorder=3)
        self.dots = ax.scatter([], [], s=22, color=C.ACCENT, edgecolor=C.INK, linewidth=0.5, zorder=4)
        ax.axhline(d["level"] / 1e3, color=C.SLACK, ls=(0, (5, 3)), lw=1.3, zorder=2)
        ax.text(spec.duration - 0.5, d["level"] / 1e3 + 0.35,
                f"{d['level'] / 1e3:.1f} kN scored severance level (a scoring threshold; no cable is cut)",
                ha="right", va="bottom", fontsize=C.FS_TINY, color=C.SLACK)
        ax.axhline(spec.pretension / 1e3, color=C.MUTED, ls=":", lw=1.0, zorder=2)
        ax.text(0.5, spec.pretension / 1e3 + 0.3, f"T0 = {spec.pretension / 1e3:.0f} kN", ha="left", va="bottom",
                fontsize=C.FS_TINY, color=C.MUTED)
        ax.set_xlim(0.0, spec.duration); ax.set_ylim(0.0, 18.5)
        ax.set_xticks(np.arange(0, spec.duration + 1, 10)); ax.set_yticks([0, 5, 10, 15])
        ax.set_ylabel("kN", fontsize=C.FS_TINY, labelpad=2)
        ax.text(1.0, 1.03, "tension, max over the five cables (plant law on the 1 ms log, 10 ms max-pooled); "
                "dots = record marks", transform=ax.transAxes, ha="right", va="bottom", fontsize=C.FS_TINY, color=C.MUTED)
        ax.text(-0.006, -0.1, "time (s)", transform=ax.transAxes, ha="right", va="top", fontsize=C.FS_TINY,
                color=C.MUTED)
        ax.grid(axis="y", lw=0.5)
        self.cur_T = ax.axvline(0.0, color=C.ACCENT, lw=1.5, alpha=0.7, zorder=2)
        self.readout = ax.text(0.006, 0.97, "", transform=ax.transAxes, ha="left", va="top", fontsize=C.FS_TINY,
                               color=C.INK, zorder=6, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        # pause inset, inside the fleet view's lower right (bodies are checked to stay clear of it)
        self.ax_sp = fig.add_axes(self.INSET)
        self.ax_sp.set_zorder(10); self.ax_sp.set_visible(False)
        self.callout = None
        self.clock_txt = None
        self.cap_txt = None
        self.cap_idx = None

    @staticmethod
    def _ordinal(n: int) -> str:
        return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

    def set_camera(self, rows):
        self.cam, self.half = camera(self.d, rows, self.box)

    def keepouts(self, t: float, mode: str):
        return keepouts(self.d, t, mode)

    # ------------------------------------------------------------------ per frame
    def draw(self, row: int, mode: str, cap_text: str, cap_idx: int):
        d, fig = self.d, self.fig
        t = float(d["st"][row])
        k1 = 10 * row                                          # 1 ms sample at the same instant
        T_row, e_row = d["T"][k1], d["e"][k1]
        paused = mode != "play"
        centre = self.cam[row]
        ax = self.ax_f
        C.draw_fleet(ax, d["S"][row], d["geometry"], T_row, centre=centre, half=self.half, labels=False)
        lp, vp = C.unpack_state(d["S"][row])
        lo, hi = centre - self.half, centre + self.half
        W, Hh = hi[0] - lo[0], hi[1] - lo[1]
        cab = None
        if mode == "pause1":
            cab = int(d["first"][1])
        elif mode in ("pause2", "end"):
            cab = d["c_max"]
        if cab is not None:
            highlight_cable(ax, lp, vp, d["geometry"], cab)
        if mode == "end":                                  # the key frame repeats pause 2's label layout
            _LABEL_SIDE["side"] = self.label_side.get((row, "pause2"))
        vessel_labels(ax, vp, lo, hi, keepout=self.keepouts(t, mode), highlight=cab)
        self.label_side[(row, mode)] = _LABEL_SIDE["side"]
        for x in np.arange(math.ceil(lo[0] / 10.0) * 10.0, hi[0], 10.0):
            ax.axvline(x, color=C.FAINT, lw=0.6, zorder=0)
        for y in np.arange(math.ceil(lo[1] / 10.0) * 10.0, hi[1], 10.0):
            ax.axhline(y, color=C.FAINT, lw=0.6, zorder=0)
        # corner note (right of every body; asserted): view, live tug spread against the design fan,
        # and the no-contact caveat, which turns into the collapse note once the tugs bunch
        note = ("top-down view\nworld frame, grid 10 m\n"
                f"tug spread {d['spread'][row]:.1f} m\n(design fan {d['design_spread']:.1f} m)")
        ax.text(0.995, 0.985, note, transform=ax.transAxes, ha="right", va="top", fontsize=C.FS_TINY,
                color=C.INK, linespacing=1.3, zorder=7)
        if t >= d["t_col"]:
            sub, col = (f"tugs bunch from {d['t_col']:.0f} s\n(closest pair < {CLOSE_M:.0f} m apart);\n"
                        "no contact model:\nhulls overlap"), C.SLACK
        else:
            sub, col = "the plant models no\ncontact between bodies:\nhulls may overlap", C.MUTED
        ax.text(0.995, 0.755, sub, transform=ax.transAxes, ha="right", va="top", fontsize=C.FS_TINY,
                color=col, linespacing=1.3, zorder=7)
        # scheduled squall force (MissionSpec envelope x calibrated amplitude) -- toward world -y
        psi = float(squall_envelope(d["spec"], t))
        if psi > 1e-3:
            x0, y0 = lo[0] + 0.035 * W, hi[1] - 0.10 * Hh
            ax.add_patch(FancyArrowPatch((x0, y0), (x0, y0 - 0.42 * Hh * psi), arrowstyle="-|>", mutation_scale=18,
                                         lw=2.2, color=C.SLACK, zorder=6))
            ax.text(x0 + 0.015 * W, y0, f"squall\n{psi * d['cal'].vessel_amplitude / 1e3:.1f} kN per tug",
                    ha="left", va="top", fontsize=C.FS_TINY, color=C.SLACK, zorder=6)
        pause_rule = {"pause1": f"paused: peak of the first snap above {d['level'] / 1e3:.1f} kN",
                      "pause3": "paused: payload's heading extreme in the squall",
                      "pause2": "paused: the mission's maximum tension",
                      "end": "key frame, shown again: the mission's maximum"}.get(mode)
        if pause_rule:
            ax.text(0.008, 0.985, pause_rule, transform=ax.transAxes, ha="left", va="top", fontsize=C.FS_SMALL,
                    color=C.ACCENT, weight="bold", zorder=7)

        sp = self.ax_sp
        if mode in ("pause1", "pause2", "end"):
            # inset: the highlighted cable over the last SPARK_S seconds of the 1 ms log (past only)
            k0 = max(0, k1 - int(round(SPARK_S / 1e-3)))
            tt = d["et"][k0: k1 + 1] - t
            sp.cla(); sp.set_visible(True)
            slack = d["e"][k0: k1 + 1, cab] <= 0.0
            sp.fill_between(tt, 0.0, 18.5, where=slack, color=C.SLACK, alpha=0.12, lw=0, step="mid")
            sp.plot(tt, d["T"][k0: k1 + 1, cab] / 1e3, color=C.TAUT, lw=1.3)
            sp.axhline(d["level"] / 1e3, color=C.SLACK, ls=(0, (4, 2)), lw=1.0)
            if slack.any():
                ts = tt[slack]
                sp.text(0.5 * (ts.min() + ts.max()), 9.0, "slack", ha="center", va="center", fontsize=C.FS_TINY,
                        color=C.SLACK)
            sp.set_xlim(-SPARK_S, 0.3); sp.set_ylim(0.0, 18.5)
            sp.set_xticks([-8, -4, 0]); sp.set_yticks([0, 5, 10, 15])
            sp.tick_params(labelsize=C.FS_TINY - 2, length=2, pad=1)
            sp.set_title(f"cable {cab}, last {SPARK_S:.0f} s (kN)", fontsize=C.FS_TINY, color=C.INK, pad=3)
            sp.set_xlabel("s before now", fontsize=C.FS_TINY - 2, labelpad=1)
        elif mode == "pause3":
            # inset: the payload's heading over the last HEAD_S seconds of the 10 ms state (past only)
            r0 = max(0, row - int(round(HEAD_S / 0.01)))
            tt = d["st"][r0: row + 1] - t
            sp.cla(); sp.set_visible(True)
            sp.axhline(0.0, color=C.FAINT, lw=0.8)
            sp.plot(tt, d["head"][r0: row + 1], color=C.INK, lw=1.4)
            sp.set_xlim(-HEAD_S, 0.4); sp.set_ylim(-120.0, 60.0)
            sp.set_xticks([-12, -6, 0]); sp.set_yticks([-90, -45, 0, 45])
            sp.tick_params(labelsize=C.FS_TINY - 2, length=2, pad=1)
            sp.set_title(f"payload heading (deg), last {HEAD_S:.0f} s", fontsize=C.FS_TINY, color=C.INK, pad=3)
            sp.set_xlabel("s before now", fontsize=C.FS_TINY - 2, labelpad=1)
        else:
            sp.set_visible(False)

        # track
        self.track_line.set_data(d["S"][: row + 1, 0], d["S"][: row + 1, 1])
        self.track_dot.set_data([d["S"][row, 0]], [d["S"][row, 1]])

        # status panel
        for c in range(5):
            if T_row[c] > 0.0:
                s, col = f"taut  {T_row[c] / 1e3:5.2f} kN", C.TAUT
            elif e_row[c] > 0.0:
                s, col = f"no tension (chord {e_row[c] * 1e3:.0f} mm long)", C.SLACK
            elif -e_row[c] < 5e-4:
                s, col = "slack (chord at rest length)", C.SLACK
            elif -e_row[c] < 0.1:
                s, col = f"slack (chord {-e_row[c] * 1e3:.0f} mm short)", C.SLACK
            else:
                s, col = f"slack (chord {-e_row[c]:.2f} m short)", C.SLACK
            self.status[c].set_text(s); self.status[c].set_color(col)

        # ticker: marks whose peak time has passed
        mk = d["marks"]
        seen = mk[mk[:, 0] <= t + 1e-9]
        latest = seen[::-1][:TICKER_ROWS]
        for j, (a, b, cc) in enumerate(self.tick_rows):
            if j < len(latest):
                tp, cb, _, Tp = latest[j][:4]
                col = C.SLACK if Tp > d["level"] else C.INK
                a.set_text(f"{tp:.3f} s"); b.set_text(f"{int(cb)}"); cc.set_text(f"{Tp / 1e3:.2f} kN")
                for x in (a, b, cc):
                    x.set_color(col); x.set_weight("bold" if Tp > d["level"] else "normal")
            else:
                for x in (a, b, cc):
                    x.set_text("")
        n_seen, n_hi = len(seen), int(np.sum(seen[:, 3] > d["level"])) if len(seen) else 0
        self.tick_sum.set_text(f"marks so far: {n_seen}\nof which above\n{d['level'] / 1e3:.1f} kN (red): {n_hi}")

        # strips
        for cur in (self.cur_s, self.cur_T):
            cur.set_xdata([t, t])
        self.trace.set_data(self.t_all[: row + 1], self.pooled[: row + 1])
        self.dots.set_offsets(np.c_[seen[:, 0], seen[:, 3] / 1e3] if len(seen) else np.empty((0, 2)))
        k_run = int(np.argmax(self.Tm[: k1 + 1]))
        c_run = int(np.argmax(d["T"][k_run]))
        if self.Tm[k_run] > 0.0:
            self.readout.set_text(f"running max so far: {self.Tm[k_run] / 1e3:.2f} kN "
                                  f"(cable {c_run}, t = {d['et'][k_run]:.3f} s)")
        if self.callout is not None:
            self.callout.remove(); self.callout = None
        if mode == "pause1":
            f = d["first"]
            self.callout = self.ax_T.annotate(
                f"cable {int(f[1])}: {f[3] / 1e3:.1f} kN — first crossing of {d['level'] / 1e3:.1f} kN",
                xy=(f[0], f[3] / 1e3), xytext=(f[0] - 6.0, 10.0), fontsize=C.FS_SMALL, color=C.INK, ha="right",
                va="center", arrowprops=dict(arrowstyle="->", color=C.INK, lw=1.0), zorder=7)
        elif mode in ("pause2", "end"):
            self.callout = self.ax_T.annotate(
                f"maximum {d['T_max'] / 1e3:.1f} kN = {d['T_max'] / d['spec'].pretension:.0f} T0 "
                f"(cable {d['c_max']}, {d['t_max']:.3f} s)",
                xy=(d["t_max"], d["T_max"] / 1e3), xytext=(d["t_max"] + 8.0, 15.5), fontsize=C.FS_SMALL,
                color=C.INK, weight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=C.INK, lw=1.2), zorder=7)

        # clock and caption
        if self.clock_txt is not None:
            self.clock_txt.remove()
        speed = {"play": f"×{SPEED:.0f}", "end": "paused · key frame"}.get(mode, "paused")   # holds: "paused"
        self.clock_txt = C.clock(fig, t, speed)
        if cap_idx != self.cap_idx:
            if self.cap_txt is not None:
                self.cap_txt.remove()
            self.cap_txt = C.caption(fig, cap_text, y=CAPTION_Y)
            self.cap_idx = cap_idx


def title_note() -> str:
    return (f"This is seed {SEED} of the {len(CALIBRATION_SEEDS)} recorded v1 squall missions (seeds "
            f"{CALIBRATION_SEEDS[0]}–{CALIBRATION_SEEDS[-1]}), and every frame is plant state from the record.")


def title_card(fig):
    fig.clf()
    fig.text(0.5, 0.62, "The operation", fontsize=C.FS_TITLE + 14, weight="bold", color=C.INK, ha="center", va="center")
    fig.text(0.5, 0.50, TITLE_QUESTION, fontsize=C.FS_SUB + 4, color=C.INK, ha="center", va="center")
    fig.text(0.5, 0.43, TITLE_WATCH, fontsize=C.FS_SUB + 4, color=C.MUTED, ha="center", va="center")
    fig.text(0.5, 0.33, title_note(), fontsize=C.FS_SMALL, color=C.MUTED, ha="center", va="center")
    C.footer(fig, f"Presentation clip C1 · source {PKL}")


# ============================================================================ manifest values

def register(man: C.Manifest, d: dict, caps):
    spec, cal = d["spec"], d["cal"]
    ms = "tether/campaign/mission.py::MissionSpec"
    man.value("tugs / cables", K.VESSEL_COUNT, "", "tether/physics/constants.py::VESSEL_COUNT")
    man.value("cable rest length", K.CABLE_REST_LENGTH, "m", "tether/physics/constants.py::CABLE_REST_LENGTH")
    man.value("pretension T0", spec.pretension, "N", f"{ms}.pretension", "declared 'T0 = 1 kN' in phase5_results.json "
              "declarations.mission; the tension of every cable in the design steady tow (tether/physics/fleet.py::"
              "operating_point, whose thrusts the mission schedules use; asserted); caption 1 'whose pretension, the "
              "steady towing pull, is 1 kN', subtitle 'pretension T0 = 1 kN'")
    man.value("seed", SEED, "", PKL + " [i].seed")
    man.value("mission set", [CALIBRATION_SEEDS[0], CALIBRATION_SEEDS[-1], len(CALIBRATION_SEEDS)], "",
              "tether/campaign/mission.py::CALIBRATION_SEEDS; " + PKL, "40 missions, seeds 5001-5040")
    man.value("selection: C4 false alarms held by seed 5008 / in the 70-110 s window", [d["n_sel"], d["n_win"]], "ticks",
              TICKS + " seed,time,bounce,censored_H1,forecast_linearized_H1,label_H1,coupled_H1",
              "computed by the stated rule; totals cross-checked with " + RES_T2)
    man.value("rank of this mission's maximum among the 40", d["rank"], "",
              RES5 + " stress.per_mission_max", "computed: descending order")
    man.value("commanded dogleg (heading offset of every tug's reference)", [math.degrees(spec.dogleg_angle),
              spec.dogleg_start, spec.dogleg_end], "deg, s, s", f"{ms}.dogleg_angle/dogleg_start/dogleg_end",
              "caption 3 and the schedule strip state it as commanded; what the fleet did is caption 5")
    sq0 = d["sq0"]
    man.value("squall rise / plateau / fall", [sq0, sq0 + spec.squall_rise, sq0 + spec.squall_rise + spec.squall_plateau,
                                              d["squall_end"]], "s", f"{ms}.squall_centre/squall_rise/squall_plateau")
    man.value("squall plateau force per tug / on the payload", [cal.vessel_amplitude, cal.load_amplitude], "N",
              "tether/campaign/mission.py::calibrate_squall(MissionSpec())",
              "matches '4.6 kN per vessel' in phase5_results.json declarations.mission; shown x squall_envelope(t) on the arrow")
    man.value("thrust ramp / deceleration", [spec.ramp_duration, spec.deceleration_start, spec.duration], "s",
              f"{ms}.ramp_duration/deceleration_start/duration")
    man.value("background weather ramp, memory", [spec.weather_ramp_duration, K.WEATHER_TIME_CONSTANT], "s",
              f"{ms}.weather_ramp_duration; tether/physics/constants.py::WEATHER_TIME_CONSTANT")
    man.value("background weather law ('t3, common-mode')", [spec.weather_distribution, spec.weather_direction,
              spec.weather_scale], "", f"{ms}.weather_distribution/weather_direction/weather_scale",
              "declared 'common-mode t3 background weather at intensity 1.0' in " + RES5 + " declarations.mission (asserted)")
    man.value("scored severance level (stress threshold)", d["level"], "N", RES5 + " stress.threshold",
              "55th percentile of the 40 per-mission maxima rounded to 100 N (asserted); scoring threshold, no cut")
    f = d["first"]
    man.value("first scored severance: cable, peak time, peak", [int(f[1]), f[0], f[3]], "-, s, N",
              REC_T0 + " excursions[seed 5008, first_severance].tags (cable, t_up, T_peak); peak time from the 1 ms log",
              "pause 1 frame shows the 10 ms row nearest that peak (58.780 s, where the 1 ms sample is 13.41 kN: the "
              "status panel, headed 'at this instant'); the ticker gives the recorded peak 13.42 kN at 58.779 s")
    p3 = d["p3"]
    man.value("payload heading: squall onset, extreme in the squall (pause 3), change", [float(d["head"][int(round(sq0 / 0.01))]),
              float(d["head"][p3]), float(d["st"][p3]), d["swing"]], "deg, deg, s, deg",
              PKL + " truth.state[:, 2] (load theta) at 10 ms",
              f"computed; caption 5 'about {abs(round(d['swing'], -1)):.0f}° to starboard' (negative = clockwise = starboard, "
              f"MissionSpec convention 'positive = turning to port'); the payload's heading range over the squall is "
              f"{d['head_range']:.1f} deg; pause 3 instant = argmin of the heading over {sq0:.0f}-{d['squall_end']:.0f} s; "
              f"the tugs' headings turn " + ", ".join(f"{x:.1f}" for x in d["tug_swing"]) + " deg to starboard over "
              "the same interval (manifest only)")
    man.value("payload displacement along the squall direction over the squall", [d["drift"]] + d["drift_vec"].tolist(),
              "m (downwind; x, y)", PKL + " truth.state[:, 0:2] at 50.00 and 70.00 s",
              "computed; squall direction -y (MissionSpec.squall_direction); caption 5 '89 m downwind'")
    man.value("fan collapse: first closest tug pair < 1 m, share of later rows < 1 m, max spread from 65 s, design "
              "spread, min spread and min closest pair before the squall",
              [d["t_col"], d["share"], d["spread_after"], d["design_spread"], d["spread_pre"], d["closest_pre"]],
              "s, -, m, m, m, m", PKL + " truth.state[:, 3:18] (tug centres) at 10 ms",
              "computed; caption 5 'the fan collapses for good'; fleet-view note 'tugs bunch from 62 s (closest pair "
              "< 1 m apart)'; design spread = largest tug-centre distance at t = 0 (mission.initial_state)")
    man.value("post hoc: missions with two tug centres < 1 m at some row / median share of rows", [d["n_bunch"],
              d["med_bunch"]], "missions, -", PKL + " truth.state[:, 3:18] of all 40 missions",
              "post hoc, manifest only (caveats); per-mission shares in seed order: " +
              ", ".join(f"{b:.3f}" for b in d["bunch"]))
    man.value("tug spread (dynamic)", "largest distance between two tug centres at the drawn 10 ms row", "m",
              PKL + " truth.state[:, 3:18]", "fleet-view note, against the design fan's 17.8 m at t = 0")
    man.value("all five cables slack (elongation <= 0)", list(d["t_slack"]), "s", PKL + " truth.elongation",
              "computed: longest all-slack run after 60 s; caption 6 'from 71 s'")
    man.value("first re-engagement of each cable after the all-slack run", d["first_re"], "s",
              PKL + " marks (t_up)", f"computed; caption 6 '{d['re_lo']:.1f}–{d['re_hi']:.1f} s'")
    man.value("mission maximum tension, time, cable", [d["T_max"], d["t_max"], d["c_max"]], "N, s, -",
              PKL + " max_tension, max_tension_time; recomputed with common.tension from truth.elongation/rate",
              f"also {d['T_max'] / spec.pretension:.1f} T0 (derived); pause 2 and the end hold show this row")
    man.value("Z (impact impedance, one tug-payload pair)", d["Z"], "N s/m", REC_Z + " cell_b_impact.Z_repin.Z_N_s_per_m",
              "record check only, not on screen in this clip: caption 7 states the per-line practice in words (the snap's "
              "peak predicted from the line's closing speed); the relation T_peak ~ T0 + Z V_up (paper eq. 1) and "
              "Z = 7.99 kN s/m for a single vessel-payload pair (Sec. II) are given on the next slide, s03_perline")
    man.value("re-engagement marks (peak time, cable, t_up, T_peak, depth, dwell, v_return)", d["marks"].tolist(),
              "s, -, s, N, m, s, m/s",
              PKL + " marks; peak time = first local max of recomputed tension after t_up",
              "ticker shows the latest 6 whose peak time has passed (peak time, cable, T_peak); red/bold when "
              "T_peak > 4.5 kN; strip dots at (peak time, T_peak)")
    man.value("marks total / above 4.5 kN", [len(d["marks"]), int(np.sum(d["marks"][:, 3] > d["level"]))], "",
              PKL + " marks", "computed")
    man.value("clock (every animated frame)", "state_time of the drawn 10 ms row", "s", PKL + " truth.state_time",
              "the end hold repeats the pause-2 row (76.150 s) and says 'paused · key frame'; the reading holds on the "
              "first and the last row say 'paused'")
    man.value("playback speed", SPEED, "x", "frame plan: every 13-14th 10 ms row at 30 fps (nominal x4)")
    man.value("per-cable status (dynamic)", "tension of the 1 ms sample at the drawn row; chord shortfall -e when e <= 0",
              "kN, m, mm", PKL + " truth.elongation/rate -> common.tension")
    man.value("running max so far (dynamic)", "max of recomputed tension over the 1 ms log up to the drawn instant",
              "kN, s", PKL + " truth.elongation/rate -> common.tension")
    man.value("max-tension strip (dynamic)", "max over the 5 cables, 10 ms bins max-pooled from the 1 ms log", "kN",
              PKL + " truth.elongation/rate -> common.tension")
    man.value("cable line width cap (legend)", C.WIDTH_FULL_N, "N", "tether/analysis/v2/present/common.py::WIDTH_FULL_N",
              "draw_fleet width = 1.4 + 5.0 min(T / 12 kN, 1) pt; legend 'wider with tension, up to 12 kN'")
    man.value("squall direction", spec.squall_direction, "rad", f"{ms}.squall_direction",
              "-pi/2: force toward world -y (wind from port); drawn as the arrow and in the schedule note")
    man.value("selection window", [d["squall_end"], spec.deceleration_start], "s",
              f"{ms}: squall end (squall_centre + plateau/2 + rise) and deceleration_start")
    man.value("median of the 40 per-mission maxima", d["median_max"], "N", RES5 + " stress.per_mission_max",
              "computed; manifest only (selection, caveats)")
    man.value("axis scales", {"time": [0, spec.duration], "tension_kN": [0, 18.5], "fleet_grid_m": 10,
                              "fleet_scale_bar_m": 5, "track_scale_bar_m": 20, "heading_inset_deg": [-120, 60]},
              "", "drawing scales")
    man.value("pause inset history", {"tension_s": SPARK_S, "heading_s": HEAD_S}, "s",
              "drawing choice: pauses 1/2 and the end hold show the highlighted cable's recomputed tension over the "
              "last 8 s of the 1 ms log (past only), slack (e <= 0) shaded; pause 3 shows the payload heading "
              "(truth.state[:, 2]) over the last 12 s of the 10 ms state")
    man.value("pause and hold durations", {"title": title_frames() / C.FPS, "hold_first_row": HOLD0_S,
                                           "pause1": PAUSE1_S, "pause3": PAUSE3_S, "pause2": PAUSE2_S,
                                           "hold_last_row": HOLDZ_S, "end_hold": END_S}, "s",
              "frame plan; the title card stays up for the reading time of its two sentences (2.5 words/s); the "
              "reading holds repeat the first and the last 10 ms row with the clock reading 'paused'")
    man.value("title card", ["The operation", TITLE_QUESTION, TITLE_WATCH, title_note()], "",
              "text: title, the question the clip answers, what to watch for; numbers: seed and mission set as "
              "registered above")
    man.value("caption texts", [c[1] for c in caps], "", "numbers as registered above; the last is the end card "
              "(the clip's takeaway), shown on the end hold only")


# ============================================================================ main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stills", type=Path, default=None,
                    help="write inspection PNGs of a few frames to this directory instead of the video")
    args = ap.parse_args(argv)
    C.ensure_dirs()
    man = C.Manifest(NAME, "The operation: one squall-passage mission",
                     "One recorded v1 Squall Passage mission (seed 5008; fan formation by design, T0 = 1 kN, recording "
                     "mode) from start to end at x4: five tugs tow one payload on five unilateral 12 m cables under a "
                     "scheduled thrust ramp, a commanded 60 deg dogleg to port (40-70 s), a squall from port (50-70 s) "
                     "and deceleration (110-130 s), with random background weather. In the squall the payload swings "
                     "about 120 deg to starboard, not to port, drifts 89 m downwind, and the fan collapses for the rest "
                     "of the mission. Slack cables re-engage with snaps; the record's marks appear in a ticker and on a "
                     "running max-tension strip against the 4.5 kN scored severance level. The clip then says that per-line "
                     "practice predicts a snap's peak from the line's closing speed (how fast its ends move apart as it comes "
                     "taut) and asks the study's question: does the closing speed still belong to one line when lines share "
                     "one payload? Its end card, on the key frame of the mission's maximum, "
                     "gives the takeaway: slack cables snapping back taut are the failure mode studied here.")
    d = load_and_check(man)
    plan, p1, p2, p3 = frame_plan(d)
    caps, cap_of = caption_schedule(plan, d)

    # caption timing: <= 2 lines, >= 4 s, <= 2.5 words/s (checked on the frame plan; the last caption is the
    # end card, on the end hold); each caption shown as one contiguous run of frames
    for i, (t0, text) in enumerate(caps):
        run = [k for k, c in enumerate(cap_of) if c == i]
        dur = len(run) / C.FPS
        words = _words(text)
        man.check(f"caption {i + 1}{' (end card)' if i == len(caps) - 1 else ''} timing",
                  text.count("\n") <= 1 and dur >= 4.0 and words / dur <= WORDS_PER_S + 1e-9
                  and run == list(range(run[0], run[0] + len(run))),
                  f"{words} words over {dur:.2f} s ({words / dur:.2f} w/s)")
    tw, tf = title_words(), title_frames()
    man.check("title card: >= 3 s and its two sentences at <= 2.5 words/s; end card on the key frame (end hold >= 3 s)",
              tf / C.FPS >= 3.0 and tw / (tf / C.FPS) <= WORDS_PER_S + 1e-9 and END_S >= 3.0
              and all(r == p2 for r, m in plan if m == "end") and all(c == len(caps) - 1 for (r, m), c in zip(plan, cap_of)
                                                                   if m == "end"),
              f"title {tw} words over {tf / C.FPS:.2f} s; end hold {END_S:.2f} s")
    n_rows = d["S"].shape[0]
    man.check("reading holds repeat the first and the last 10 ms row only",
              {r for r, m in plan if m == "hold"} == {0, n_rows - 1})
    register(man, d, caps)
    man.selection = (f"Seed {SEED}: the mission of clip C4 -- among the per-line model's coupled top-bin false alarms "
                     f"(scored ticks: not bounce, not censored at H = 1 s; forecast_linearized_H1 >= 0.9; label_H1 false; "
                     f"coupled_H1 true) at {d['squall_end']:.0f}-{d['spec'].deceleration_start:.0f} s, the seed holding the most "
                     f"({d['n_sel']} of {d['n_win']}; unique maximum). Chosen to match C4, not for typicality: its maximum "
                     f"tension {d['T_max'] / 1e3:.2f} kN is the {Scene._ordinal(d['rank'])} largest of the 40 missions' maxima "
                     f"(median {d['median_max'] / 1e3:.2f} kN). Pause instants by rule, each named on screen: (1) the peak of "
                     f"the snap holding the first 1 ms sample above 4.5 kN (the recorded first scored severance, "
                     f"{d['first'][0]:.3f} s); (3) the payload's heading extreme in the squall "
                     f"({d['st'][p3]:.2f} s, argmin over {d['sq0']:.0f}-{d['squall_end']:.0f} s); (2) the mission's maximum "
                     f"tension ({d['t_max']:.3f} s). The end hold returns to pause 2's key frame and carries the end card. "
                     f"Two reading holds, chosen for reading time and not for any event, repeat the first row "
                     f"(t = {d['st'][0]:.3f} s) and the mission's last row (t = {d['st'][-1]:.3f} s); the clock reads "
                     f"'paused' on them.")
    mk = d["marks"]
    ratio = mk[:, 3] / (d["spec"].pretension + d["Z"] * mk[:, 6])
    big = mk[:, 3] > 10e3
    above_t0 = mk[:, 3] > d["spec"].pretension
    man.caveats = [
        "Recording run: severance is scored at the 4.5 kN stress threshold (55th percentile of the 40 per-mission maxima) "
        "and no cable is cut; the clip never says 'severed'.",
        f"One mission, chosen to match clip C4, not typical: its maximum ({d['T_max'] / 1e3:.2f} kN) is the "
        f"{Scene._ordinal(d['rank'])} largest of the 40 missions' maxima (median {d['median_max'] / 1e3:.2f} kN); "
        f"{d['n_above']} of the 40 exceed {d['level'] / 1e3:.1f} kN.",
        "The v1 Phase 5 missions ran in the exploratory continuation after the protocol's stop condition (Paper Sec. II).",
        "The fan is the design geometry. In this mission the fleet does not hold it: during the squall the payload's heading "
        f"swings {abs(d['swing']):.0f} deg to starboard against the commanded 60 deg turn to port, the payload moves "
        f"{d['drift']:.0f} m downwind, and from {d['t_col']:.2f} s the tugs bunch (closest centres < 1 m in "
        f"{100 * d['share']:.1f} % of later rows; spread <= {d['spread_after']:.1f} m from 65 s against {d['design_spread']:.1f} m "
        "at t = 0). The plant models no contact, so hulls overlap. Post hoc (not on screen; computed here from the same "
        f"pickle): {d['n_bunch']} of the 40 missions bring two tug centres within 1 m, for a median "
        f"{100 * d['med_bunch']:.0f} % of the 10 ms rows of a mission. The corrected paper does not state that the fan "
        "collapses in these missions.",
        "Caption 5 places the swing in the squall ('In the squall the payload swings ...'); that the squall drives it "
        "rests on this mission's timing (the swing runs from about 55 s to 62.85 s, inside "
        "the squall) and on records/phase5/phase5_spec.md, addendum 2026-09-12 (deterministic squall-only runs: 'the "
        "formation rotates about 90 deg and drifts sideways'). The background weather also acts; its share is not "
        "separated. Caption 5's numbers are the payload's (heading, displacement); over the same interval the tugs' "
        f"headings turn {d['tug_swing'].min():.0f} to {d['tug_swing'].max():.0f} deg to starboard. The collapse and the "
        "all-slack episode are stated as timing only.",
        "Cables are drawn as straight chords between the plant's attachment points (lumped Kelvin-Voigt elements; no cable "
        "shape is modelled). Dashed red = carrying no tension, which includes but is not limited to slack (e <= 0); the "
        "status panel separates the two.",
        "The squall arrow is the scheduled squall force (MissionSpec envelope x calibrate_squall amplitude), drawn on the "
        "view, not a measured wind; the background weather is not drawn.",
        "Tension in the fleet view and status panel ('at this instant') is the 1 ms sample at the drawn instant, so peaks "
        "between frames are not drawn there (pause 1: 13.41 kN at 58.780 s against the recorded peak 13.42 kN at 58.779 s); "
        "the strip is max-pooled over 10 ms bins of the 1 ms log (no peak lost) and the ticker gives each mark's recorded "
        "peak. Line width grows with tension only up to 12 kN, so the 13.4 kN and 17.0 kN snaps are drawn equally wide.",
        "x4 is nominal: frames show every 13th or 14th 10 ms state row (0.13-0.14 s steps); the clock shows each drawn "
        "row's own time. The clock starts at the mission start (these missions have no warm-up). The end hold repeats the "
        "76.150 s key frame after the mission's last frame, so the clock steps back there; it reads 'paused · key frame'. "
        "Reading holds repeat the first and the last 10 ms row (clock 'paused') so the captions can be read; they show "
        "no instant that the x4 playback does not.",
        "Caption 7 states the per-line practice in words only: a snap's peak is predicted from the line's closing speed, "
        "the chord rate at re-engagement (positive for every mark here: the ends move apart as the line comes taut). The "
        "relation T_peak ~ T0 + Z V_up (paper eq. 1) and Z = 7.99 kN s/m (single-pair re-pin) follow on the next slide "
        "(s03_perline); the clip does not apply the relation to this mission's marks, and V_up is not shown. Post hoc, for this "
        f"mission's {int(big.sum())} snaps above 10 kN, T0 + Z v_return exceeds the recorded peak (recorded / law "
        f"{ratio[big].min():.2f}-{ratio[big].max():.2f}; the {int(above_t0.sum())} marks above T0 "
        f"{ratio[above_t0].min():.2f}-{ratio[above_t0].max():.2f}; post hoc, not shown).",
    ]

    scene_rows = [r for r, _ in plan]
    cam, half = camera(d, scene_rows)
    clear_bad = sorted({(float(d["st"][r]), m) for r, m in plan if not bodies_clear(d, cam, half, r, m)})
    man.check("no body lies under the fleet view's text or pause inset on any frame (checked before drawing)",
              not clear_bad, f"{len(clear_bad)} (row, mode) pairs" +
              (f", first at t = {clear_bad[0][0]:.2f} s ({clear_bad[0][1]})" if clear_bad else ""))

    fig = C.new_frame()
    sc = None
    if args.stills is not None:
        args.stills.mkdir(parents=True, exist_ok=True)
        title_card(fig); fig.savefig(args.stills / "still_title.png", dpi=C.DPI)
        fig.clf()
        sc = Scene(fig, d, man, p1, p2, p3); sc.set_camera(scene_rows)
        first_k = lambda pred: next(k for k, (r, m) in enumerate(plan) if pred(r, m))
        picks = {"t000": 0, "t010": first_k(lambda r, m: d["st"][r] >= 10.0),
                 "t030": first_k(lambda r, m: d["st"][r] >= 30.0),
                 "t058": first_k(lambda r, m: d["st"][r] >= 58.0),
                 "pause1": first_k(lambda r, m: m == "pause1"),
                 "t062": first_k(lambda r, m: d["st"][r] >= 62.0),
                 "pause3": first_k(lambda r, m: m == "pause3"),
                 "t066": first_k(lambda r, m: d["st"][r] >= 65.87),
                 "t071": first_k(lambda r, m: d["st"][r] >= 71.07),
                 "pause2": first_k(lambda r, m: m == "pause2"),
                 "t077": first_k(lambda r, m: d["st"][r] >= 77.33),
                 "t095": first_k(lambda r, m: d["st"][r] >= 95.0),
                 "t106": first_k(lambda r, m: d["st"][r] >= 106.5),
                 "t125": first_k(lambda r, m: d["st"][r] >= 125.0),
                 "hold_last": first_k(lambda r, m: m == "hold" and r > 0),
                 "end": len(plan) - 1}
        for tag, k in picks.items():
            r, mode = plan[k]
            sc.draw(r, mode, caps[cap_of[k]][1], cap_of[k])
            fig.savefig(args.stills / f"still_{tag}.png", dpi=C.DPI)
        print("stills written to", args.stills)
        return

    clip = C.CLIPS / f"{NAME}.mp4"
    w = C.writer()
    frames = 0
    with w.saving(fig, str(clip), dpi=C.DPI):
        title_card(fig)
        for _ in range(title_frames()):
            w.grab_frame()
            frames += 1
        fig.clf()
        sc = Scene(fig, d, man, p1, p2, p3)
        sc.set_camera(scene_rows)
        for k, (r, mode) in enumerate(plan):
            if k == 0 or plan[k - 1] != (r, mode) or mode == "play" or cap_of[k] != cap_of[k - 1]:
                sc.draw(r, mode, caps[cap_of[k]][1], cap_of[k])
            w.grab_frame()
            frames += 1
    man.frames = frames
    man.check("frames written = title + frame plan", frames == title_frames() + len(plan),
              f"{frames} frames")
    np.savez_compressed(CACHE_FILE, seed=SEED, source=PKL, source_sha256=man.sources[PKL],
                        rows=np.array(scene_rows), state_time=d["st"][scene_rows], state=d["S"][scene_rows],
                        tension=d["T"][10 * np.array(scene_rows)], elongation=d["e"][10 * np.array(scene_rows)],
                        max_tension_10ms_pooled_kN=sc.pooled, marks=d["marks"],
                        marks_columns=np.array(["t_peak", "cable", "t_up", "T_peak", "depth", "dwell", "v_return"]),
                        payload_heading_deg=d["head"][scene_rows], tug_spread_m=d["spread"][scene_rows])
    man.value("extract cache", str(CACHE_FILE.relative_to(C.REPO)), "", PKL, "rows drawn, not a replay")
    man.write(clip)


if __name__ == "__main__":
    main()
