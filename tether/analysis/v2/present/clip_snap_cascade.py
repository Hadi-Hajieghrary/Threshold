"""C2 ``snap_cascade`` -- the mechanism: anatomy of one recorded snap and cascade
(Presentation/STORYBOARD.md, section C2).

What the clip shows
-------------------
One recorded mission of the Phase 1(f) stochastic cell ``i035_T0600_ks3`` (parallel formation,
T0 = 600 N, weather intensity 0.35, k_h = 477 N m/rad, k_sigma = 3.0, sway limit 0.349 rad, linear
drag, recording cables: severance is scored, no cable is ever cut), seed 7105, around the
re-engagement of cable 2 at t0 = 91.486 s.  The simulation clock includes the 20 s warm-up.

  (a) 6 s of real-time approach (85.086-91.086 s): cable 2 slack, its chord gap e2 = chord - 12 m,
      and the tensions of the other four cables; holds on its first and last frames;
  (b) slow motion x1/20 over t0 - 0.4 s .. t0 + 0.6 s: the top-down fleet, the five tensions from
      the 1 ms log, and the slack raster (e <= 0); a pause on cable 2's peak at t0 + 66 ms and a hold
      on the window's last frame (+600 ms), where both follow-on peaks are marked;
  (c) the transmission law of paper Section III as a slide, its gain computed here from the plant
      constants and labelled derived, not measured, with the paper's two qualifiers: treating the
      contact as an impulse is an approximation (the 0.17 s contact is comparable to the payload's
      0.19 s half-period), and the projection onto each neighbour's chord reduces the swing (so
      "up to about 0.44", less for cables at an angle to cable 2); the slide ends by saying that
      one recorded event cannot measure the gain and that the campaign measured its consequence;
  then a hold on the key frame (t0 + 66 ms) with its key number and one sentence saying what it is.

The title card frames the event as the paper does (Section III: "one such event from the inside"),
not as proof of the mechanism; it does not repeat the question, which the bridge slide just before the
clip asks (slides.py p1_mechanism).  The fleet panel names the formation: parallel = cables commanded
along the tow axis (``fleet.formation_geometry``); at T0 = 0.6 kN they spread (the cell's
chord-angle std, 35.9 deg).  The key states the line-width rule as ``common.draw_fleet`` applies it
(grows with tension, capped at 12 kN); common.py itself is not modified.

Body poses are interpolated linearly between the 10 ms state rows (the footer says so); tensions
are the plant's law on the 1 ms log, never interpolated.  Every frame of (a)/(b) is drawn at a
1 ms log sample, and the clock shows that sample's time.

Records read (path -> keys)
---------------------------
* ``records/v2/figures/cascade_run.npz`` -- the replay of (i035_T0600_ks3, 7105) made by
  ``tether/analysis/v2/paperfig/cascade_run.py`` (reused, not re-simulated): ``seed``,
  ``time`` / ``elongation`` / ``rate`` / ``alive`` (1 ms), ``state_time`` / ``state`` (10 ms rows:
  18 positions then 18 velocities; load then vessels 0..4, each x, y, theta), ``geometry_load``,
  ``geometry_vessel``, ``cascade_events``.
* ``records/v2/phase1/stochastic_runs.json`` -- ``runs[cell == i035_T0600_ks3, seed == 7105]``:
  ``config_hash``, ``pilot``, ``end_time``, ``n_marks``, ``marks.{cable, t_up, T_peak, dwell,
  depth, t_deep, v_return}``; ``runs[cell == i035_T0600_ks3, pilot == false].marks.T_peak`` (the
  cell's scored marks: count, median, rank of this one); ``runs[cell, seed in 7101..7106]
  .marks.T_peak`` (the selection scan).
* ``records/v2/phase1/stochastic_declarations.json`` -- ``cells.grid[name == cell]``: ``T0_N``,
  ``intensity``, ``k_h_N_m_per_rad``, ``k_sigma``, ``sway_limit_rad``, ``config_hash``;
  ``cells.formation``.
* ``records/v2/phase1/stochastic_results.json`` -- ``trade_off[cell == cell].marks`` (510) and
  ``.chord_world_std_deg`` (35.86: world chord-angle circular std, maximum over vessels).
* ``tether/physics/constants.py`` (CABLE_STIFFNESS, LOAD_MASS, VESSEL_MASS, CABLE_REST_LENGTH),
  ``tether/physics/fleet.py`` PAIR_REDUCED_MASS, ``tether/campaign/v2/events.py``
  ENGAGEMENT_PERIOD -- the plant parameters of the transmission law;
  ``tether/analysis/v2/present/common.py`` WIDTH_FULL_N (the 12 kN line-width cap in the key).
* ``Paper/Sections/Section_III_Coupling.tex``, ``Section_II_Setting.tex`` and
  ``Section_VI_Domain.tex`` -- the law's printed values (0.44, 0.69, [0.2, 0.9], 2.3 T0, 1.4 T0,
  omega_L = sqrt(4k/m_L), the 0.19 s half-period and "is an approximation", the projection
  cos(sigma_i - sigma_j), "one such event from the inside"; Table I omega_e 18.7, omega_L 16.5
  rad/s; the 35.9 deg chord-angle std), string-checked (whitespace-normalised) so the slides and
  captions cannot drift from the corrected paper.

Written: ``Presentation/cache/snap_cascade_window.npz`` (the drawn window of the record, the
tracker-replayed marks and the frame schedule), ``Presentation/clips/snap_cascade.mp4``,
``Presentation/manifests/snap_cascade.json``.

Selection rule
--------------
The event: the largest-peak cascade (a re-engagement of >= 4.5 kN followed within 3 s by a slack
onset on another cable) in ``cascade_run.main()``'s scan of seeds 7101-7106 of this cell.  Its
peak, 13.5 kN, is the 2nd-largest of the cell's 510 scored marks (median 0.94 kN): chosen for
clarity, not typicality.  The windows follow from the event, not from the look of the frames: the
storyboard's 6 s approach ending 0.4 s before t0, slow motion t0 - 0.4 s .. t0 + 0.6 s, a pause
at cable 2's peak (the maximum of its tension in [t_up, t_up + 0.4 s], the campaign's T_peak
window); the key frame is that peak.

What is asserted before any frame is drawn (``Manifest.check``)
----------------------------------------------------------------
* the npz is seed 7105; ``cascade_run.SPEC`` equals the declared cell field by field, and its
  ``FleetRunSpec.config_hash()`` equals the declared hash and the run record's; the seed is a
  scored (non-pilot) seed; the logs sit on exact 1 ms / 10 ms grids; every cable stays alive;
  the geometry is the plant's parallel formation; this module's tension law equals the plant's;
* the run record holds exactly one cable-2 mark at t_up = 91.486 s with T_peak = 13502.4 N, and
  the npz's cable-2 maximum in [t_up, t_up + 0.4 s] equals it within 1 N;
* the plant's own ``CableEventTracker``, replayed over the npz's 1 ms samples and filtered as the
  v2 summary filters marks, reproduces every one of the record's 29 marks (count, cable, t_up to
  1e-9 s, T_peak to 1e-9 relative, dwell, depth, v_return);
* the event facts put on screen: the cable-2 zero crossing lies between the 91.486 and 91.487 s
  samples; the peak is at +66 ms and 22.5 T0; cables 0, 4, 3 first reach e <= 0 at +30, +52,
  +70 ms (the paper's Fig. 3 caption); cables 0, 3, 4 carry no tension at the peak while cable 3's
  chord is still longer than rest length there; cable 1 carries tension throughout (b); cable 2 is
  slack through (a); the follow-on marks of cables 3 and 4 inside (b) are the record's; the load
  moves toward +x (the tow-direction arrow); headings do not wrap (safe to interpolate);
* the selection: the cell has 510 scored marks (= stochastic_results trade_off), 13502.4 N is
  the 2nd largest, the median is 0.94 kN; among seeds 7101-7106 seed 7105 holds the largest mark;
  ``cascade_run.find_cascades`` on the npz returns this event first, with cables 0, 3 and 4 among
  its children;
* the manifest caveats' description of the event: the payload's x-velocity rises from t0 to the
  peak while cables 0, 3, 4 shorten (their chord rates turn negative); cable 1's chord lies about
  95 deg from cable 2's at t0 and its tension moves < 0.2 kN within t0 + 170 ms;
* the formation note: the cell's chord_world_std_deg rounds to 35.9 and the paper prints 35.9 deg;
* the law: omega_e = sqrt(k / mu_pair) equals 2 pi / ENGAGEMENT_PERIOD and Table I's 18.7 rad/s;
  omega_L = sqrt(4k/m_L) is 16.5 rad/s; the contact pi/omega_e and the payload's half-period
  pi/omega_L round to 0.17 and 0.19 s; 2k/(m_L omega_L omega_e) rounds to 0.44, the rectangular
  pulse's k pi/(m_L omega_L omega_e) to 0.69, their thresholds to 2.3 and 1.4 T0; the paper
  prints each of these.
Drawing preconditions, also asserted before the first frame: one fixed camera holds every body
over the whole window (``common.draw_fleet`` re-asserts it per frame); each vessel number keeps
>= 0.5 m from every other hull and the payload in every drawn frame (positions chosen with
hysteresis by ``label_plan``, so a number never names the wrong vessel); the key-frame banner
covers no body and no cable at the key frame, and the formation note none in any drawn frame
(both also measured from the rendered text); the e2 panel's slack label stays above the e2 trace;
the t0 label sits above the top of the t0 and playhead lines (which stop at AXP_LINE_TOP; the peak
annotation is white-backed so the playhead passes behind it); the slide's left column ends left of
its boxes and the DERIVED, NOT MEASURED text fits its box (``check_layout`` renders sample frames to
the figure only); every caption stays on screen >= 4 s and >= words / 2.5 s, and renders in at most
two lines inside the frame, clear of every other text, legend, axes and slide box (``check_captions``);
the playback speeds are 1.000 (real time) and 1/20 within 0.5 %.  The manifest's frame count is the
number of frames written.

Narration: the captions, the title card, the slide's notes, the key-frame banner and the closing caption
are complete sentences, and each symbol is defined in words before a caption uses it (T0, the pretension,
on the title card and again in the peak caption; t0, the re-engagement time, in the slow-motion caption;
the chord, the distance between the cable's ends, in the approach caption, so that the slow-motion
caption's "its chord lengthening at 1.78 m/s" reads the right way: the record's v_return is the rate at
which the chord grows through 12 m).  Captions longer than
ONE_LINE_MAX characters are drawn as two balanced lines (``two_lines``); the manifest logs the drawn text.

Run: ``nice -n 10 python3 -m tether.analysis.v2.present.clip_snap_cascade``
(``--check`` verifies and stops; ``--stills`` also writes a few PNGs to the development directory ($TETHER_DEV_DIR)).
"""
from __future__ import annotations

import tempfile
import os
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tether.analysis.v2.present import common as C

NAME = "snap_cascade"
TITLE = "Anatomy of a snap and a cascade"
SEED, CELL, PARENT = 7105, "i035_T0600_ks3", 2
WARMUP = 20.0
T_UP_REC = 91.486            # the record's t_up of the cable-2 mark, on the 1 ms grid
T_PEAK_REC = 13502.4         # N, the record's T_peak of that mark
T0_MS = 91486                # t0 as an index into the 1 ms log (time[i] = i ms)
PEAK_WINDOW = 0.4            # s, the campaign's T_peak window [t_up, t_up + 0.4 s]
APPROACH = (-6400, -400)     # ms from t0: (a), real time
SLOW = (-400, 600)           # ms from t0: (b), slow motion
SLOW_FACTOR = 20
EXPECT_SLACK_MS = {0: 30, 4: 52, 3: 70}     # paper Fig. 3 caption / storyboard
SCAN = tuple(range(7101, 7107))
BAND = (0.2, 0.9)            # declared band on the transmission ratio (paper Sec. III)

NPZ = Path("records/v2/figures/cascade_run.npz")
RUNS = Path("records/v2/phase1/stochastic_runs.json")
DECL = Path("records/v2/phase1/stochastic_declarations.json")
RESULTS = Path("records/v2/phase1/stochastic_results.json")
CONSTANTS = Path("tether/physics/constants.py")
SEC2 = Path("Paper/Sections/Section_II_Setting.tex")
SEC3 = Path("Paper/Sections/Section_III_Coupling.tex")
SEC6 = Path("Paper/Sections/Section_VI_Domain.tex")
CASCADE_SRC = Path("tether/analysis/v2/paperfig/cascade_run.py")
COMMON_SRC = Path("tether/analysis/v2/present/common.py")
CACHE = C.CACHE / "snap_cascade_window.npz"
CLIP = C.CLIPS / f"{NAME}.mp4"
STILLS = Path(os.environ.get("TETHER_DEV_DIR", Path(tempfile.gettempdir()) / "tether_dev")) / "present" / "snap_cascade"

# per-cable identity colours for the traces and the bow labels; deliberately neither the
# fleet view's taut blue nor its no-tension red, which encode cable state, not identity
CAB = {0: "#1b9e77", 1: "#6b7280", 2: "#d98c00", 3: "#7b3294", 4: "#8c510a"}

# layout (figure fractions)
FLEET_AX = [0.020, 0.250, 0.560, 0.615]
AX_P = [0.650, 0.665, 0.325, 0.195]     # (b) cable 2 tension
AX_N = [0.650, 0.430, 0.325, 0.175]     # (b) other cables' tension
AX_R = [0.650, 0.265, 0.325, 0.115]     # (b) slack raster
AX_E = [0.650, 0.530, 0.325, 0.330]     # (a) cable 2 chord gap
AX_O = [0.650, 0.265, 0.325, 0.190]     # (a) other cables' tension
CAPTION_Y = 0.135
KEY_Y = 0.212

SELECTION_TEXT = ("Selection: the largest-peak cascade in a scan of seeds 7101–7106; 2nd-largest of the "
                  "cell's 510 scored marks (median 0.94 kN) — chosen for clarity, not typicality.")
FOOTER_DATA = ("Plant state: records/v2/figures/cascade_run.npz, the replay of seed 7105 whose 29 marks equal "
               "records/v2/phase1/stochastic_runs.json · recording run: no cable is cut")
FOOTER_INTERP = ("Bodies interpolated linearly between 10 ms state rows; tensions from the 1 ms log · times include "
                 "the 20 s warm-up · no contact is modelled: lines and hulls may overlap")
SUB = "Seed 7105 · cell i035_T0600_ks3 · parallel formation · T0 = 600 N · intensity 0.35"
# the title card: the title, then (in the caption) the paper's framing -- one event seen from the inside (paper
# Sec. III: "one such event from the inside"), not proof -- and what to watch for (the storyboard's title card).
# The question the clip answers is asked by the bridge slide just before it (slides.py p1_mechanism: "Does a snap
# on one line affect the others?"), so the title card does not ask it again.
TITLE_DETAILS = ("The event comes from seed {seed} of cell {cell}, towing in parallel formation at weather intensity "
                 "{intensity:.2f}.",
                 "Each cable's pretension is T0 = {T0:.0f} N, and each tug's heading gain is k_h = {k_h:.0f} N·m/rad.",
                 "The animated frames replay recorded plant state; in this recording run, severance is scored but no "
                 "cable is cut.")

# narration: complete sentences, each symbol defined in words before it is used (T0 on the title card and in
# CAP_P1, t0 in CAP_B); every caption is asserted to stay >= 4 s and >= words / 2.5 s on screen, and to render
# in at most two lines inside its band (check_layout)
CAP_TITLE = ("We look inside one recorded event. "
             "Watch cable 2 snap taut and three other cables go slack moments later.")
CAP_A1 = "Five tugs tow one payload on 12 m cables that can only pull."
CAP_A2 = "Cable 2 has been slack since 82.2 s because its chord, the distance between its ends, is under 12 m."
# the chord (CAP_A2: the distance between the cable's ends) lengthens through 12 m at re-engagement: the record's
# v_return is the crossing rate of e2 = chord - 12 m, +1.78 m/s (e2 -0.645 mm -> +1.132 mm across t0, checked)
CAP_B = ("The movie now runs 20 times slower. "
         "Cable 2 re-engages at time t0 = 91.486 s, its chord lengthening at 1.78 m/s.")
CAP_P1 = ("Cable 2 peaks 66 ms later at 13.5 kN, 22.5 times the pretension T0, "
          "while cables 0, 3 and 4 carry no tension.")
CAP_B3 = "The bottom panel shows cables 0, 4 and 3 going slack 30, 52 and 70 ms after t0."
CAP_B4 = "Cables 3 and 4 then re-engage in turn, with smaller peaks of 3.5 and 2.8 kN."
CAP_C1 = "The derivation approximates the snap as an impulse J, a sudden push on the shared payload."
CAP_C2 = ("The push swings the other cables' tension by up to about 0.44 of the peak, "
          "less for cables at an angle to cable 2.")
CAP_C3 = ("So another cable can go slack once a peak exceeds about 2.3 times the pretension T0; "
          "this peak was 22.5 T0.")
# paper Sec. III: "(transmission) is an analytic prediction that this campaign did not measure"; it "is sensitive to
# the assumed pulse shape ... We therefore carry [0.2, 0.9] as the declared uncertainty band on the transmission ratio"
CAP_C4 = ("The gain of 0.44 is derived, never measured; because it depends on the assumed pulse shape, "
          "its uncertainty band is 0.2 to 0.9.")
CAP_C5 = ("One recorded event cannot measure the gain either; the campaign measured its predicted consequence, "
          "how often slack follows another cable's re-engagement.")
CAP_END = ("Because the cables share one payload, a snap on one cable can unload other cables, "
           "which a one-cable model cannot represent.")      # paper Sec. III: "has no representation of it at all"
CAPTIONS = (CAP_TITLE, CAP_A1, CAP_A2, CAP_B, CAP_P1, CAP_B3, CAP_B4, CAP_C1, CAP_C2, CAP_C3, CAP_C4, CAP_C5,
            CAP_END)
ONE_LINE_MAX = 96            # characters: a longer caption is broken into two balanced lines (two_lines)
CAPTION_Y_LAW = 0.115        # the slide's caption sits a little lower, clear of the threshold strip's axis label

# the fleet panel's formation note (static; asserted clear of every body and cable in every drawn frame)
FORMATION_NOTE = ("parallel formation: cables commanded along the tow axis;\n"      # fleet.formation_geometry
                  "at T0 = 0.6 kN they spread (cell chord-angle std {std:.1f}°)")
NOTE_XY = (0.985, 0.03)                  # axes fraction, anchor bottom-right
NOTE_BOX = (0.355, 0.015, 0.995, 0.15)   # axes-fraction box the note must stay inside (x0, y0, x1, y1)
KEY_TAUT = "taut: width grows with tension, capped at {cap:.0f} kN"
E2_LABEL = "e₂ ≤ 0: cable 2 slack"           # the storyboard's definition of slack (rule 6)
T0_LABEL = "t0 = 91.486 s: cable 2 re-engages"
AXP_LINE_TOP = 0.87                           # axes fraction where the t0 and playhead lines stop (below T0_LABEL)
# the key-frame banner: the key number, then one sentence saying what it is (rule 8: end cards in sentences);
# every fact is asserted in verify (peak at +66 ms; cables 0, 3, 4 carry no tension at the peak)
BANNER_SENTENCE = ("Cable 2 reaches this peak\n"
                   "{ms} ms after re-engaging, while\n"
                   "cables 0, 3 and 4 carry no tension.")
# the slide's DERIVED, NOT MEASURED box (paper Sec. III; closeout: P2-T11 lived in Phase 2, not launched);
# on screen 17.6 s under CAP_C4 and CAP_C5, which narrate it
BOX_TEXT = ("No test here measured a neighbour's swing:\n"
            "that test sat in a phase not launched.\n"
            "One recorded event cannot measure it either.")


def _json(path: Path) -> dict:
    return json.loads((C.REPO / path).read_text())


class _Geom:
    def __init__(self, load, vessel):
        self.load_offsets, self.vessel_offsets = np.asarray(load, float), np.asarray(vessel, float)


# ----------------------------------------------------------------------------- verification

def tracker_marks(d: dict, warmup: float, end: float) -> dict:
    """The plant's own CableEventTracker replayed over the logged 1 ms samples.

    The plant feeds each tracker, every 1 ms, the (time, e, edot, k e + c edot) it writes to the log
    (tether/physics/fleet.py FleetCables._update) with ``_discard_event`` installed, so this replay
    reproduces its marks; the relative load (not logged) only feeds the W_rel fields, which are not
    compared.  The filter ``t_up - dwell >= warmup and t_up <= end`` is the v2 summary's.
    """
    from tether.physics.cable import CableEventTracker, CableMode
    from tether.physics.fleet import CableParameters, _discard_event
    cab = CableParameters(mode="recording")
    time, e, r = d["time"], d["elongation"], d["rate"]
    taut = cab.stiffness * e + cab.damping * r
    out = []
    for i in range(e.shape[1]):
        tr = CableEventTracker(stiffness=cab.stiffness, damping=cab.damping, cable=i, capacity=4096,
                               mode=CableMode.RECORDING, break_threshold=None)
        tr._append_event = _discard_event.__get__(tr)
        ti, ei, ri, qi = time.tolist(), e[:, i].tolist(), r[:, i].tolist(), taut[:, i].tolist()
        for k in range(len(ti)):
            tr.sample(ti[k], ei[k], ri[k], qi[k], 0.0)
            got, _, _ = tr.drain_if_due(ti[k])
            out.extend(got)
        out.extend(tr.state.reengagement_ring.drain())
    kept = sorted((m for m in out if m.t_up - m.dwell >= warmup and m.t_up <= end), key=lambda m: (m.t_up, m.cable))
    return {k: np.array([getattr(m, k) for m in kept]) for k in ("t_up", "T_peak", "cable", "dwell", "depth", "v_return")}


def verify(man: C.Manifest, d: dict) -> dict:
    """Every assertion the clip rests on; returns the facts the frames and captions use."""
    from tether.analysis.v2.paperfig import cascade_run as CR
    from tether.campaign.fleet_run import FleetRunSpec
    from tether.physics import fleet as F

    runs_all = _json(RUNS)["runs"]
    decl, res = _json(DECL), _json(RESULTS)
    f: dict = {}

    # ---- identity of the replay
    man.check("cascade_run.npz is seed 7105", int(d["seed"]) == SEED, f"seed {int(d['seed'])}")
    grid = {g["name"]: g for g in decl["cells"]["grid"]}[CELL]
    spec = FleetRunSpec(**CR.SPEC)
    fields = {"pretension": (spec.pretension, grid["T0_N"]), "weather_scale": (spec.weather_scale, grid["intensity"]),
              "heading_gain": (spec.heading_gain, grid["k_h_N_m_per_rad"]), "k_sigma": (spec.k_sigma, grid["k_sigma"]),
              "sway_limit": (spec.sway_limit, grid["sway_limit_rad"]),
              "formation": (spec.formation, decl["cells"]["formation"]),
              "cable_mode": (spec.cable_mode, "recording")}
    bad = {k: v for k, v in fields.items() if v[0] != v[1]}
    man.check("cascade_run.SPEC equals the declared cell i035_T0600_ks3 field by field", not bad,
              f"{len(fields)} fields checked; mismatches {bad}")
    run = [r for r in runs_all if r["cell"] == CELL and r["seed"] == SEED]
    man.check("the run record holds exactly one (i035_T0600_ks3, 7105) run", len(run) == 1)
    run = run[0]
    h = spec.config_hash()
    man.check("FleetRunSpec(cascade_run.SPEC).config_hash() equals the declared hash and the run record's",
              h == grid["config_hash"] == run["config_hash"], h)
    man.check("seed 7105 is a scored statistics seed of the cell (pilot == false)", run["pilot"] is False)
    f.update(T0=float(grid["T0_N"]), intensity=float(grid["intensity"]), k_h=float(grid["k_h_N_m_per_rad"]))

    t, e, r, alive = d["time"], d["elongation"], d["rate"], d["alive"]
    n = len(t)
    man.check("1 ms log on an exact grid (time[i] = i ms)",
              float(np.max(np.abs(t - np.arange(n) * 1e-3))) < 1e-9, f"{n} samples, last {t[-1]:.3f} s")
    st = d["state_time"]
    man.check("10 ms state rows on an exact grid", float(np.max(np.abs(st - np.arange(len(st)) * 1e-2))) < 1e-9,
              f"{len(st)} rows")
    man.check("recording run: every cable alive throughout", bool(alive.all()))
    g = C.parallel_geometry()
    man.check("npz geometry equals the plant's parallel formation",
              np.allclose(d["geometry_load"], g.load_offsets) and np.allclose(d["geometry_vessel"], g.vessel_offsets))
    T = C.tension(e, r, alive)
    man.check("common.tension equals the plant law cascade_run.tension on the whole log",
              np.array_equal(T, CR.tension(e, r, alive)))
    f["T"] = T

    # ---- the record's mark and its reproduction
    mk = run["marks"]
    idx = [j for j, (c, tu) in enumerate(zip(mk["cable"], mk["t_up"])) if c == PARENT and abs(tu - T_UP_REC) < 5e-4]
    man.check("the record holds exactly one cable-2 mark at t_up = 91.486 s", len(idx) == 1, f"indices {idx}")
    j = idx[0]
    t_up, T_pk = float(mk["t_up"][j]), float(mk["T_peak"][j])
    man.check("that mark's T_peak is 13502.4 N", abs(T_pk - T_PEAK_REC) < 0.05, f"record T_peak {T_pk:.4f} N, t_up {t_up:.6f} s")
    w = (t >= t_up) & (t <= t_up + PEAK_WINDOW)
    ipk = int(np.flatnonzero(w)[np.argmax(T[w, PARENT])])
    man.check("npz: cable-2 peak in [t_up, t_up + 0.4 s] equals 13502.4 N within 1 N (and the record's T_peak)",
              abs(T[ipk, PARENT] - T_PEAK_REC) < 1.0 and abs(T[ipk, PARENT] - T_pk) < 1.0,
              f"recomputed {T[ipk, PARENT]:.4f} N at {t[ipk]:.3f} s")
    ev = d["cascade_events"]
    man.check("npz cascade_events[0] is (91.486 s, cable 2, 13502.4 N)",
              abs(ev[0, 0] - T_UP_REC) < 1e-9 and int(ev[0, 1]) == PARENT and abs(ev[0, 2] - T_PEAK_REC) < 0.05,
              f"{ev[0].tolist()}")

    tm = tracker_marks(d, WARMUP, float(run["end_time"]))
    o = np.lexsort((np.asarray(mk["cable"]), np.asarray(mk["t_up"], float)))
    rec = {k: np.asarray(mk[k], float)[o] for k in ("t_up", "T_peak", "cable", "dwell", "depth", "v_return")}
    nm = len(tm["t_up"])
    man.check("plant CableEventTracker over the npz: mark count equals the record", nm == run["n_marks"] == len(o),
              f"replay {nm}, record {run['n_marks']}")
    same = (np.array_equal(tm["cable"].astype(int), rec["cable"].astype(int))
            and np.max(np.abs(tm["t_up"] - rec["t_up"])) < 1e-9
            and np.max(np.abs(tm["T_peak"] - rec["T_peak"]) / rec["T_peak"]) < 1e-9
            and np.max(np.abs(tm["dwell"] - rec["dwell"])) < 1e-9
            and np.max(np.abs(tm["depth"] - rec["depth"])) < 1e-9
            and np.max(np.abs(tm["v_return"] - rec["v_return"])) < 1e-9)
    man.check("plant CableEventTracker over the npz reproduces every record mark (cable, t_up, T_peak, dwell, depth, v_return)",
              bool(same), f"{nm} marks; max |dt_up| {np.max(np.abs(tm['t_up'] - rec['t_up'])):.1e} s, "
                          f"max rel |dT_peak| {np.max(np.abs(tm['T_peak'] - rec['T_peak']) / rec['T_peak']):.1e}")
    f["tracker"] = tm

    # ---- event facts on the 1 ms grid (t0 = 91.486 s is sample T0_MS)
    i0 = T0_MS
    man.check("t0 sample: t[91486] = 91.486 s; cable 2's zero crossing (record t_up) lies between it and the next",
              abs(t[i0] - T_UP_REC) < 1e-9 and e[i0, PARENT] <= 0.0 < e[i0 + 1, PARENT] and t[i0] <= t_up < t[i0 + 1],
              f"e2 {e[i0, PARENT]*1e3:.3f} mm -> {e[i0+1, PARENT]*1e3:.3f} mm; record t_up {t_up:.6f} s")
    peak_ms = ipk - i0
    man.check("cable-2 peak at t0 + 66 ms, 22.5 T0", peak_ms == 66 and round(T[ipk, PARENT] / f["T0"], 1) == 22.5,
              f"+{peak_ms} ms, {T[ipk, PARENT] / f['T0']:.3f} T0")
    after = np.arange(i0 + 1, n)
    slack_ms = {c: int(after[np.argmax(e[i0 + 1:, c] <= 0.0)] - i0) for c in EXPECT_SLACK_MS}
    zero_ms = {c: int(after[np.argmax(T[i0 + 1:, c] <= 0.0)] - i0) for c in EXPECT_SLACK_MS}
    man.check("cables 0, 4, 3 first reach e <= 0 at +30, +52, +70 ms (paper Fig. 3 caption)",
              slack_ms == EXPECT_SLACK_MS, f"{slack_ms}")
    man.check("cables 0, 4, 3 first carry no tension before their chords reach rest length",
              all(0 < zero_ms[c] < slack_ms[c] for c in zero_ms), f"zero tension at {zero_ms} ms")
    man.check("at the peak cables 0, 3, 4 carry no tension; cable 3's chord is still longer than rest length; cable 1 is loaded",
              all(T[ipk, c] == 0.0 for c in (0, 3, 4)) and e[ipk, 3] > 0.0 and T[ipk, 1] > 0.0,
              f"T {np.round(T[ipk], 1).tolist()} N; e3 {e[ipk, 3]*1e3:.3f} mm")
    s_lo, s_hi = i0 + SLOW[0], i0 + SLOW[1]
    a_lo, a_hi = i0 + APPROACH[0], i0 + APPROACH[1]
    man.check("cable 1 carries tension throughout the slow-motion window", bool((T[s_lo:s_hi + 1, 1] > 0).all()),
              f"min {T[s_lo:s_hi + 1, 1].min():.0f} N")
    man.check("cable 2 is slack (e <= 0) from the approach start to t0; cables 0, 1, 3, 4 are not",
              bool((e[a_lo:i0 + 1, PARENT] <= 0).all()) and bool((e[a_lo:i0 + 1, [0, 1, 3, 4]] > 0).all()))
    f.update(ipk=ipk, peak_ms=peak_ms, peak_N=float(T[ipk, PARENT]), slack_ms=slack_ms, zero_ms=zero_ms,
             t_up_exact=t_up, T_at_t0=T[i0].copy())

    # the excursion (approach) from the record, checked against the log
    dwell, depth, t_deep, v_ret = (float(mk[k][j]) for k in ("dwell", "depth", "t_deep", "v_return"))
    on = np.flatnonzero((e[:-1, PARENT] > 0) & (e[1:, PARENT] <= 0)) + 1
    on = int(on[on <= i0][-1])
    kd = on + int(np.argmin(e[on:i0 + 1, PARENT]))
    man.check("record excursion: slack onset t_up - dwell = 82.2 s matches the log's last e <= 0 onset before t0",
              abs((t_up - dwell) - t[on]) < 1e-3 and round(t_up - dwell, 1) == 82.2,
              f"t_up - dwell {t_up - dwell:.4f} s, log onset sample {t[on]:.3f} s")
    man.check("record depth 3.82 m = the log's deepest cable-2 gap, at the record's t_deep",
              abs(depth + e[kd, PARENT]) < 1e-9 and abs(t[kd] - t_deep) < 1e-9 and round(depth, 2) == 3.82,
              f"depth {depth:.6f} m at {t_deep:.3f} s")
    man.check("record v_return 1.78 m/s agrees with the log's rate at t0 within 0.01 m/s",
              abs(v_ret - r[i0, PARENT]) < 0.01 and round(v_ret, 2) == 1.78, f"v_return {v_ret:.4f}, rate {r[i0, PARENT]:.4f}")
    f.update(onset_t=t_up - dwell, depth=depth, t_deep=t_deep, v_return=v_ret, dwell=dwell)

    # follow-on marks inside (b): the first re-engagements of cables 3 and 4 after t0
    fol = {}
    for c in (3, 4):
        js = [k for k in range(len(mk["cable"])) if mk["cable"][k] == c and t_up < mk["t_up"][k] < t_up + SLOW[1] / 1e3]
        js.sort(key=lambda k: mk["t_up"][k])
        man.check(f"record: cable {c} re-engages inside the slow-motion window", len(js) >= 1, f"{len(js)} marks")
        k = js[0]
        tu_c = float(mk["t_up"][k])
        wc = (t >= tu_c) & (t <= tu_c + PEAK_WINDOW)
        ic = int(np.flatnonzero(wc)[np.argmax(T[wc, c])])
        man.check(f"cable {c}'s follow-on peak is reached inside the window and equals the record's T_peak",
                  ic <= s_hi and abs(T[ic, c] - mk["T_peak"][k]) < 1.0,
                  f"t_up {tu_c:.3f} s (+{1e3*(tu_c - T_UP_REC):.0f} ms), peak {T[ic, c]:.1f} N at +{ic - i0} ms")
        fol[c] = dict(t_up=tu_c, T_peak=float(mk["T_peak"][k]), i_peak=ic)
    man.check("follow-on peaks read 3.5 kN (cable 3) and 2.8 kN (cable 4)",
              round(fol[3]["T_peak"] / 1e3, 1) == 3.5 and round(fol[4]["T_peak"] / 1e3, 1) == 2.8)
    f["follow"] = fol

    # drawing preconditions
    rows = slice((i0 + APPROACH[0]) // 10, (i0 + SLOW[1]) // 10 + 2)
    q = d["state"][rows, :18]
    th = q[:, 2::3]
    man.check("headings do not wrap between 10 ms rows in the drawn window (linear interpolation is safe)",
              float(np.max(np.abs(np.diff(th, axis=0)))) < 0.05, f"max step {np.max(np.abs(np.diff(th, axis=0))):.2e} rad")
    vload = d["state"][rows, 18]
    man.check("the load moves toward +x through the window (tow-direction arrow)", bool((vload > 0).all()),
              f"load vx {vload.min():.2f}-{vload.max():.2f} m/s")
    f["vessel_ahead"] = bool((q[:, 3::3] > q[:, [0]]).all())
    man.check("every vessel is ahead (+x) of the payload through the window", f["vessel_ahead"])
    # what the manifest caveat says of the event (not put on screen): the payload surges along +x after the
    # snap while the neighbours' chords shorten
    vx0, vxp = float(interp_state(d, i0)[18]), float(interp_state(d, ipk)[18])
    man.check("from t0 to the peak the payload's x-velocity rises and cables 0, 3, 4 shorten (chord rates < 0 at the peak)",
              vxp > vx0 and bool((r[ipk, [0, 3, 4]] < 0).all()),
              f"load vx {vx0:.3f} -> {vxp:.3f} m/s; chord rates at the peak {np.round(r[ipk, [0, 3, 4]], 3).tolist()} m/s")
    f.update(vx0=vx0, vxp=vxp)
    lp0, vp0 = C.unpack_state(interp_state(d, i0))
    la, lb = C.attachment_points(lp0, vp0, _Geom(d["geometry_load"], d["geometry_vessel"]))
    ang = np.degrees(np.arctan2(*(lb - la)[:, ::-1].T))          # chord angles, load attachment -> stern
    d12 = float(abs(ang[1] - ang[PARENT]))
    dT1 = float(np.max(np.abs(T[i0:ipk + 105, 1] - T[i0, 1])))
    man.check("caveat: at t0 cable 1's chord is ~95 deg from cable 2's (|cos| < 0.2) and its tension moves < 0.2 kN "
              "within t0 + 170 ms", abs(math.cos(math.radians(d12))) < 0.2 and dT1 < 200.0,
              f"chord angles {np.round(ang, 1).tolist()} deg; |sigma_1 - sigma_2| {d12:.1f} deg; max |dT1| {dT1:.0f} N")
    f["ang12"] = d12

    # ---- selection
    cell = [x for x in runs_all if x["cell"] == CELL]
    scored = [x for x in cell if not x["pilot"]]
    pk_all = np.concatenate([np.asarray(x["marks"]["T_peak"], float) for x in scored])
    to = [x for x in res["trade_off"] if x["cell"] == CELL][0]
    rank = int((pk_all > T_pk + 1e-6).sum()) + 1
    med = float(np.median(pk_all))
    man.check("the cell's scored marks number 510 (= stochastic_results trade_off.marks); this is the 2nd largest; median 0.94 kN",
              len(pk_all) == 510 == to["marks"] and rank == 2 and round(med / 1e3, 2) == 0.94,
              f"{len(scored)} scored seeds, {len(pk_all)} marks, rank {rank}, median {med:.1f} N, largest {pk_all.max():.1f} N")
    scan = {x["seed"]: max(x["marks"]["T_peak"]) for x in cell if x["seed"] in SCAN}
    man.check("record: among seeds 7101-7106 of the cell, seed 7105 holds the largest mark (cascade_run.main's pick)",
              sorted(scan) == list(SCAN) and max(scan, key=scan.get) == SEED,
              "; ".join(f"{s}: {v:.0f} N" for s, v in sorted(scan.items())))
    casc = CR.find_cascades(d)
    kids = {c for c, _ in casc[0][3]}
    man.check("cascade_run.find_cascades on the npz: this event first, with cables 0, 3, 4 among its children",
              abs(casc[0][0] - T_UP_REC) < 1e-9 and casc[0][1] == PARENT and {0, 3, 4} <= kids,
              f"{len(casc)} cascades; first {casc[0][0]:.3f} s cable {casc[0][1]} {casc[0][2]:.1f} N children {casc[0][3]}")
    f.update(n_scored=len(pk_all), rank=rank, median=med, n_marks=nm)

    # ---- the formation note: the cell's chord-angle spread
    std = float(to["chord_world_std_deg"])
    s6 = " ".join((C.REPO / SEC6).read_text().split())
    man.check("formation note: the cell's chord_world_std_deg rounds to 35.9 deg and the paper (Sec. VI) prints 35.9 deg",
              round(std, 1) == 35.9 and r"from $35.9^\circ$" in s6, f"{std:.4f} deg")
    f["chord_std"] = std
    return f


def derive_law(man: C.Manifest) -> dict:
    """The transmission law of paper Sec. III at the plant's constants (derived, not measured)."""
    from tether.physics import constants as K
    from tether.physics.fleet import PAIR_REDUCED_MASS
    from tether.campaign.v2.events import ENGAGEMENT_PERIOD
    k, mL, mA = K.CABLE_STIFFNESS, K.LOAD_MASS, K.VESSEL_MASS
    mu = mA * mL / (mA + mL)
    w_e, w_L = math.sqrt(k / mu), math.sqrt(4.0 * k / mL)
    gain = 2.0 * k / (mL * w_L * w_e)
    rect = math.pi * k / (mL * w_L * w_e)
    L = dict(k=k, mL=mL, mA=mA, mu=mu, w_e=w_e, w_L=w_L, gain=gain, rect=rect, thr=1.0 / gain,
             thr_rect=1.0 / rect, thr_band=(1.0 / BAND[1], 1.0 / BAND[0]), contact=math.pi / w_e,
             half_L=math.pi / w_L)
    man.check("omega_e = sqrt(k/mu_pair) with the plant's PAIR_REDUCED_MASS; 2 pi/omega_e = campaign ENGAGEMENT_PERIOD",
              abs(mu - PAIR_REDUCED_MASS) < 1e-9 and abs(2 * math.pi / w_e - ENGAGEMENT_PERIOD) < 1e-12,
              f"mu {mu:.3f} kg, omega_e {w_e:.4f} rad/s, period {ENGAGEMENT_PERIOD:.4f} s")
    man.check("omega_e and omega_L round to Table I's 18.7 and 16.5 rad/s", round(w_e, 1) == 18.7 and round(w_L, 1) == 16.5,
              f"{w_e:.4f}, {w_L:.4f}")
    man.check("2k/(m_L omega_L omega_e) rounds to 0.44 and threshold 1/gain to 2.3 T0 (paper eqs. transmission, threshold)",
              round(gain, 2) == 0.44 and round(L["thr"], 1) == 2.3, f"gain {gain:.5f}, threshold {L['thr']:.4f} T0")
    man.check("rectangular pulse of the same duration: k pi/(m_L omega_L omega_e) rounds to 0.69, threshold 1.4 T0",
              round(rect, 2) == 0.69 and round(L["thr_rect"], 1) == 1.4, f"{rect:.5f}, {L['thr_rect']:.4f} T0")
    man.check("the band's thresholds: T0/0.9 = 1.1 T0, T0/0.2 = 5.0 T0; contact pi/omega_e rounds to 0.17 s",
              round(L["thr_band"][0], 1) == 1.1 and round(L["thr_band"][1], 1) == 5.0 and round(L["contact"], 2) == 0.17)
    man.check("the payload's half-period pi/omega_L rounds to 0.19 s (paper Sec. III: the impulse is an approximation)",
              round(L["half_L"], 2) == 0.19, f"{L['half_L']:.4f} s against the contact's {L['contact']:.4f} s")
    s3 = " ".join((C.REPO / SEC3).read_text().split())       # whitespace-normalised: phrases span tex lines
    s2 = " ".join((C.REPO / SEC2).read_text().split())
    need3 = [r"0.44\,T_{\text{peak}}", r"0.69\,T_{\text{peak}}", r"$[0.2, 0.9]$", r"2.3\,T_0", r"1.4\,T_0",
             r"\omega_L = \sqrt{4k/m_L}", r"\frac{2k}{m_L\,\omega_L\,\omega_e}", "analytic prediction that",
             "this campaign did not measure",
             "is sensitive to the assumed pulse shape",                     # CAP_C4: why the band
             "declared uncertainty band on the transmission ratio", "whose error the declared band below covers",
             r"half-period of $0.19$\,s", "treating the contact as an impulse delivered to the payload is an approximation",
             r"the projection $\cos(\sigma_i - \sigma_j)$ onto each neighbour's chord reducing it further",
             r"downstream \emph{consequence} the mechanism predicts", "which we did measure",
             "shows one such event from the inside"]
    need2 = [r"$18.7$, $16.5$", r"$1.7\times10^{5}$", r"$2500$\,kg", r"$600$\,kg", r"$12$\,m"]
    miss = [s for s in need3 if s not in s3] + [s for s in need2 if s not in s2]
    man.check("the corrected paper prints the slide's law, band, thresholds, qualifiers (impulse approximation, "
              "projection, measured consequence), the title card's framing and Table I values", not miss,
              f"{len(need3) + len(need2)} phrases; missing {miss}")
    return L


# ----------------------------------------------------------------------------- timeline

@dataclass(frozen=True)
class Spec:
    kind: str                 # title | a | b | law | end
    t_ms: int | None = None   # 1 ms log sample drawn (and shown on the clock)
    speed: str | None = None
    caption: str = ""
    step: int = 0


def secs(s: float) -> int:
    return int(round(s * C.FPS))


_UNITS = {"m", "s", "ms", "kN", "N", "m/s", "T0", "="}
_WEAK = {"a", "an", "the", "of", "to", "at", "on", "by", "and", "cable", "cables", "its", "=", "t0", "times", "than",
         "about", "up", "with", "for", "from", "after", "in", "is", "was", "can"}


def two_lines(text: str) -> str:
    """The caption as drawn: one line if short, else two balanced lines broken at a word boundary, preferring a
    sentence or clause boundary and avoiding a break between a number and its unit, next to a number, or after a
    weak word.  Only whitespace changes, so the words are the caption's words (the manifest logs the drawn text)."""
    if len(text) <= ONE_LINE_MAX:
        return text
    w = text.split(" ")
    best = None
    for i in range(1, len(w)):
        a, b = " ".join(w[:i]), " ".join(w[i:])
        prev, nxt = w[i - 1], w[i].strip(".,;:")
        score = abs(len(a) - len(b))
        score += 100 * (nxt in _UNITS or prev == "=")
        score += 30 * (nxt[:1].isdigit() or prev[:1].isdigit() or prev.lower() in _WEAK)
        score -= 40 * prev.endswith((".", ";", ":", "?")) + 12 * prev.endswith(",")
        if best is None or score < best[0]:
            best = (score, a + "\n" + b)
    return best[1]


def timeline(f: dict) -> list[Spec]:
    """The frame schedule.  The moving frames (real time and x1/20) are fixed by the event; reading time for the
    captions comes only from holds on drawn instants: the title card, the first approach frame, the approach's
    last frame, the peak, the window's last frame (+600 ms, where both follow-on peaks are marked), the slide
    steps and the closing key frame."""
    fr: list[Spec] = []
    add = lambda spec, n: fr.extend([spec] * n)
    add(Spec("title", caption=CAP_TITLE), secs(8.0))
    a0, a1 = T0_MS + APPROACH[0], T0_MS + APPROACH[1]
    add(Spec("a", a0, "paused", CAP_A1), secs(5.2))
    n_rt = int(round((a1 - a0) / 1000 * C.FPS))                       # real time: 30 frames per s
    for tt in np.linspace(a0, a1, n_rt + 1)[:-1]:
        fr.append(Spec("a", int(round(tt)), "real time", CAP_A2))
    add(Spec("a", fr[-1].t_ms, "paused", CAP_A2), secs(2.0))         # hold on the approach's last frame
    # slow motion to the peak, a pause there, then on to the window's end and a hold on its last frame
    pk = T0_MS + f["peak_ms"]
    n1 = int(round((pk - (T0_MS + SLOW[0])) / 1000 * SLOW_FACTOR * C.FPS))
    seq1 = np.round(np.linspace(T0_MS + SLOW[0], pk, n1 + 1)).astype(int)
    for tt in seq1:
        fr.append(Spec("b", int(tt), "×1/20 slow motion", CAP_B))
    add(Spec("b", pk, "paused", CAP_P1), secs(9.4))
    n2 = int(round((SLOW[1] - f["peak_ms"]) / 1000 * SLOW_FACTOR * C.FPS))
    seq2 = np.round(np.linspace(pk, T0_MS + SLOW[1], n2 + 1)).astype(int)[1:]
    for k, tt in enumerate(seq2):
        fr.append(Spec("b", int(tt), "×1/20 slow motion", CAP_B3 if k < secs(7.2) else CAP_B4))
    add(Spec("b", int(seq2[-1]), "paused", CAP_B4), secs(3.0))
    add(Spec("law", caption=CAP_C1, step=1), secs(6.6))
    add(Spec("law", caption=CAP_C2, step=2), secs(9.6))
    add(Spec("law", caption=CAP_C3, step=3), secs(8.4))
    # the DERIVED, NOT MEASURED box stays up under two captions (16.8 s), the second carrying its last sentence
    add(Spec("law", caption=CAP_C4, step=4), secs(9.2))
    add(Spec("law", caption=CAP_C5, step=4), secs(8.4))
    # the key frame carries the closing caption and the banner's sentence (a recap of CAP_P1)
    add(Spec("end", pk, "paused on the key frame", CAP_END), secs(10.0))
    return fr


def check_timeline(man: C.Manifest, fr: list[Spec], f: dict) -> dict:
    runs, cur, n = [], None, 0
    for s in fr:
        if s.caption != cur:
            if cur is not None:
                runs.append((cur, n))
            cur, n = s.caption, 0
        n += 1
    runs.append((cur, n))
    short = [(c, n / C.FPS) for c, n in runs if n / C.FPS < max(4.0, len(c.split()) / 2.5) - 1e-9]
    man.check("every caption stays on screen >= 4 s and >= words / 2.5 s", not short,
              "; ".join(f"{len(c.split())} w / {n / C.FPS:.2f} s" for c, n in runs) + f"; too short: {short}")
    man.check("every caption fits in two lines", all(len(c) <= 2 * 80 for c, _ in runs))
    # playback speeds: real time = 1000/30 ms per frame; slow motion = 1000/600 ms per frame on average
    a = [s.t_ms for s in fr if s.kind == "a" and s.speed == "real time"]
    b = [s.t_ms for s in fr if s.kind == "b" and s.speed.startswith("×")]
    b1 = [t for t in b if t <= T0_MS + f["peak_ms"]]
    b2 = [t for t in b if t > T0_MS + f["peak_ms"]]
    sp_a = (a[-1] - a[0]) / (len(a) - 1) * C.FPS / 1000
    sp_b1 = (b1[-1] - b1[0]) / (len(b1) - 1) * C.FPS / 1000
    sp_b2 = (b2[-1] - b2[0] + (b2[0] - b1[-1])) / len(b2) * C.FPS / 1000
    man.check("playback speeds: real time 1.000; slow motion 1/20 within 0.5 %",
              abs(sp_a - 1) < 1e-3 and abs(sp_b1 * 20 - 1) < 5e-3 and abs(sp_b2 * 20 - 1) < 5e-3,
              f"(a) {sp_a:.4f}; (b) to the peak 1/{1 / sp_b1:.2f}, after it 1/{1 / sp_b2:.2f}")
    man.check("slow-motion frames advance monotonically on the 1 ms grid", all(np.diff(b1) > 0) and all(np.diff(b2) > 0)
              and b2[0] > b1[-1])
    return dict(speed_a=sp_a, speed_b1=sp_b1, speed_b2=sp_b2)


# ----------------------------------------------------------------------------- drawing

def interp_state(d: dict, t_ms: int) -> np.ndarray:
    """Body poses at t_ms, linear between the 10 ms rows (exact on a row)."""
    j, rem = divmod(int(t_ms), 10)
    a = d["state"][j]
    return a.copy() if rem == 0 else (1.0 - rem / 10.0) * a + (rem / 10.0) * d["state"][j + 1]


def camera(d: dict, lo_ms: int, hi_ms: int):
    """One fixed camera over every body in [lo_ms, hi_ms] (a moving fleet, a still frame)."""
    from tether.physics import fleet as F
    P = []
    for row in d["state"][lo_ms // 10: hi_ms // 10 + 2]:
        lp, vp = C.unpack_state(row)
        P.append(C.body_polygon(lp, F.pentagon_vertices()))
        P += [C.body_polygon(v, C.hull()) for v in vp]
    P = np.vstack(P)
    lo, hi = P.min(0), P.max(0)
    box_aspect = (FLEET_AX[2] * C.W) / (FLEET_AX[3] * C.H)
    return 0.5 * (lo + hi), C.fit_aspect(0.5 * (hi - lo) + np.array([2.5, 2.5]), box_aspect)


def _dist_seg(p, a, b) -> np.ndarray:
    """Distance from point p to each segment a[i]-b[i]."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    ab = b - a
    s = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(a + s[:, None] * ab - p, axis=1)


def _dist_poly(p, P) -> float:
    """Distance from point p to polygon P (0 inside)."""
    from matplotlib.path import Path as MPath
    if MPath(P).contains_point(p):
        return 0.0
    return float(np.min(_dist_seg(p, P, np.roll(P, -1, axis=0))))


LABEL_CANDIDATES = np.array([[0.5 * C.L_HULL + 1.1, 0.0], [0.0, 1.3], [0.0, -1.3], [0.5 * C.L_HULL + 0.9, 1.0],
                             [0.5 * C.L_HULL + 0.9, -1.0]])   # bow, port, starboard, bow-port, bow-starboard


def _label_clearances(d: dict, geom, t_ms: int):
    """Per vessel and candidate: distance to the nearest other hull or the payload, and to the nearest chord."""
    from tether.physics import fleet as F
    lp, vp = C.unpack_state(interp_state(d, t_ms))
    hulls = [C.body_polygon(v, C.hull()) for v in vp]
    pay = C.body_polygon(lp, F.pentagon_vertices())
    a, b = C.attachment_points(lp, vp, geom)
    hc, cc = np.zeros((5, len(LABEL_CANDIDATES))), np.zeros((5, len(LABEL_CANDIDATES)))
    for i in range(5):
        for k, c in enumerate(LABEL_CANDIDATES):
            p = C.body_polygon(vp[i], c[None, :])[0]
            hc[i, k] = min([_dist_poly(p, h) for j, h in enumerate(hulls) if j != i] + [_dist_poly(p, pay)])
            cc[i, k] = float(np.min(_dist_seg(p, a, b)))
    return hc, cc


def label_plan(d: dict, frames: list) -> dict:
    """Vessel-number label position (body frame) for every drawn instant, in playback order, with hysteresis:
    a label keeps its position while it stays LABEL_HULL_CLEAR from every other hull and the payload, and only
    then moves to the candidate with the most clearance (cable clearance, then the bow, break ties).  A number
    sitting on another hull would name the wrong vessel; one crossing a cable line stays readable.
    Returns {t_ms: (offsets (5, 2), hull clearances (5,))}."""
    geom = _Geom(d["geometry_load"], d["geometry_vessel"])
    plan, cur = {}, [0] * 5
    for t in [s.t_ms for s in frames if s.t_ms is not None]:
        if t in plan:
            continue
        hc, cc = _label_clearances(d, geom, t)
        for i in range(5):
            if cur[i] != 0 and hc[i, 0] >= 1.6 and cc[i, 0] >= 0.5:       # back to the bow once it is well clear
                cur[i] = 0
            elif hc[i, cur[i]] < LABEL_HULL_CLEAR:
                cur[i] = max(range(len(LABEL_CANDIDATES)),
                             key=lambda k: (min(hc[i, k], 1.5), min(cc[i, k], 0.8), k == 0))
        plan[t] = (LABEL_CANDIDATES[cur].copy(), np.array([hc[i, cur[i]] for i in range(5)]))
    return plan


LABEL_HULL_CLEAR = 0.7


def cable_key(fig, y: float, x: float):
    """The fleet view's key: common.cable_key's two handles, with the width rule stated as common.draw_fleet
    applies it (lw = 1.4 + 5.0 min(T / WIDTH_FULL_N, 1): grows with tension, capped at 12 kN)."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C.TAUT, lw=4, label=KEY_TAUT.format(cap=C.WIDTH_FULL_N / 1e3)),
               Line2D([], [], color=C.SLACK, lw=2, ls=(0, (3.0, 2.2)), label="carrying no tension")]
    fig.legend(handles=handles, loc="center", ncol=len(handles), bbox_to_anchor=(x, y),
               fontsize=C.FS_SMALL, handlelength=2.6, columnspacing=2.0)


def furniture(fig, *, clock_t=None, speed=None, tau_ms=None, caption="", key=True):
    C.title(fig, TITLE, SUB)
    if clock_t is not None:
        C.clock(fig, clock_t, speed)
    if tau_ms is not None:
        fig.text(0.97, 0.905, f"t − t0 = {tau_ms:+5d} ms", fontsize=C.FS_SUB, family="DejaVu Sans Mono",
                 color=C.INK, ha="right", va="top")
    if caption:
        C.caption(fig, two_lines(caption), y=CAPTION_Y)
    if key:
        cable_key(fig, y=KEY_Y, x=FLEET_AX[0] + FLEET_AX[2] / 2)
    fig.text(0.015, 0.056, SELECTION_TEXT, fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="bottom")
    fig.text(0.015, 0.034, FOOTER_INTERP, fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="bottom")
    fig.text(0.015, 0.012, FOOTER_DATA, fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="bottom")


def draw_fleet_panel(fig, d, geom, cam, t_ms, f, labels, banner=False):
    ax = fig.add_axes(FLEET_AX)
    row = interp_state(d, t_ms)
    T_row = f["T"][t_ms]
    C.draw_fleet(ax, row, geom, T_row, d["alive"][t_ms], centre=cam[0], half=cam[1], labels=False)
    lp, vp = C.unpack_state(row)
    for i in range(len(vp)):
        p = C.body_polygon(vp[i], labels[t_ms][0][i][None, :])[0]
        ax.text(*p, str(i), fontsize=C.FS_SUB, weight="bold", color=CAB[i], ha="center", va="center", zorder=7)
    lo, hi = cam[0] - cam[1], cam[0] + cam[1]
    y = hi[1] - 0.07 * (hi[1] - lo[1])
    x1 = hi[0] - 0.04 * (hi[0] - lo[0])
    ax.annotate("", xy=(x1, y), xytext=(x1 - 4.0, y), arrowprops=dict(arrowstyle="-|>", lw=1.8, color=C.MUTED))
    ax.text(x1 - 4.4, y, "tow direction", fontsize=C.FS_SMALL, color=C.MUTED, ha="right", va="center")
    ax.text(lo[0] + 0.02 * (hi[0] - lo[0]), y, "top-down view, to scale", fontsize=C.FS_SMALL, color=C.MUTED,
            ha="left", va="center")
    # what 'parallel' means, and why this fleet does not look parallel (asserted clear of bodies: check_free_box)
    ax.text(*NOTE_XY, FORMATION_NOTE.format(std=f["chord_std"]), transform=ax.transAxes, fontsize=C.FS_SMALL,
            color=C.MUTED, ha="right", va="bottom", linespacing=1.35, zorder=8)
    if banner:      # in the region the fleet leaves empty at the key frame (asserted free by check_banner)
        ax.text(BANNER_XY[0], BANNER_XY[1], f"{f['peak_N'] / 1e3:.1f} kN = {f['peak_N'] / f['T0']:.1f} T0",
                transform=ax.transAxes, fontsize=32, weight="bold", color=CAB[PARENT], ha="left", va="center", zorder=8)
        ax.text(BANNER_XY[0], BANNER_XY[1] - 0.07, BANNER_SENTENCE.format(ms=f["peak_ms"]), transform=ax.transAxes,
                fontsize=C.FS_BODY, color=C.INK, ha="left", va="top", zorder=8, linespacing=1.35)
    return ax


BANNER_XY = (0.465, 0.83)          # axes fraction of the key-number banner (top-right of the fleet panel)
BANNER_BOX = (0.455, 0.57, 0.99, 0.89)   # axes-fraction box it occupies (x0, y0, x1, y1)


def _scene_points(d: dict, t_ms: int) -> np.ndarray:
    """Points along every body outline (edges sampled) and every cable at one instant (world metres)."""
    from tether.physics import fleet as F
    lp, vp = C.unpack_state(interp_state(d, t_ms))
    a, b = C.attachment_points(lp, vp, _Geom(d["geometry_load"], d["geometry_vessel"]))
    polys = [C.body_polygon(lp, F.pentagon_vertices())] + [C.body_polygon(v, C.hull()) for v in vp]
    pts = [np.linspace(P[k], P[(k + 1) % len(P)], 12) for P in polys for k in range(len(P))]
    pts += [np.linspace(a[i], b[i], 60) for i in range(5)]
    return np.vstack(pts)


def check_free_box(man: C.Manifest, d: dict, cam, box, instants, what: str) -> None:
    """No body outline or cable enters ``box`` (axes fraction of the fleet panel) at any of ``instants``."""
    lo, hi = cam[0] - cam[1], cam[0] + cam[1]
    x0, y0 = lo + np.array(box[:2]) * (hi - lo)
    x1, y1 = lo + np.array(box[2:]) * (hi - lo)
    worst = 0
    for t in sorted(set(instants)):
        P = _scene_points(d, t)
        worst = max(worst, int(((P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1)).sum()))
    man.check(f"{what} covers no body and no cable", worst == 0,
              f"{len(set(instants))} instants; most points inside at one instant: {worst}")


def check_banner(man: C.Manifest, d: dict, cam, f: dict) -> None:
    """The key-number banner must not cover any body or cable at the key frame."""
    check_free_box(man, d, cam, BANNER_BOX, [T0_MS + f["peak_ms"]], "the key-frame banner (at the key frame)")


def _text_box(fig, text: str, frame: str = "axes"):
    """The rendered extent of the first text artist whose string is ``text``: (x0, y0, x1, y1) in its axes'
    fraction (frame='axes') or data coordinates (frame='data'), or in figure fraction for a figure text."""
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text() == text:
                bb = t.get_window_extent(r)
                tr = (ax.transAxes if frame == "axes" else ax.transData).inverted()
                (a, b), (c, e) = tr.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
                return (a, b, c, e), ax
    for t in fig.texts:
        if t.get_text() == text:
            bb = t.get_window_extent(r)
            (a, b), (c, e) = fig.transFigure.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            return (a, b, c, e), None
    raise KeyError(text)


def check_layout(man: C.Manifest, fig, fr: list, d, geom, cam, f, L, labels) -> None:
    """Render sample frames (to the figure only; nothing is written) and measure the texts whose clearance the
    free-box checks assume: the formation note and the banner inside their boxes, the e2 panel's slack label
    above the e2 trace, the slide's left-column notes left of its right-hand boxes."""
    def inside(bx, box):
        return bx[0] >= box[0] and bx[1] >= box[1] and bx[2] <= box[2] and bx[3] <= box[3]
    last_a = max(i for i, s in enumerate(fr) if s.kind == "a")
    draw(fig, fr[last_a], d, geom, cam, f, L, labels)
    fig.canvas.draw()
    nb, _ = _text_box(fig, FORMATION_NOTE.format(std=f["chord_std"]))
    man.check("the formation note renders inside its asserted-free box", inside(nb, NOTE_BOX),
              f"note {np.round(nb, 3).tolist()} in box {list(NOTE_BOX)}")
    eb, _ = _text_box(fig, E2_LABEL, frame="data")
    tt = np.arange(T0_MS + APPROACH[0], T0_MS + APPROACH[1] + 1)
    under = (tt / 1e3 >= eb[0] - 0.05) & (tt / 1e3 <= eb[2] + 0.05)
    gap = float(eb[1] - d["elongation"][tt[under], PARENT].max()) if under.any() else float("inf")
    man.check("the e2 panel's slack label stays above the e2 trace over the whole approach", gap > 0.3,
              f"label {np.round(eb, 3).tolist()} (s, m); smallest gap to the trace {gap:.2f} m")
    end = next(i for i, s in enumerate(fr) if s.kind == "end")
    draw(fig, fr[end], d, geom, cam, f, L, labels)
    fig.canvas.draw()
    tb, _ = _text_box(fig, T0_LABEL)
    man.check("the t0 label sits above the top of the t0 and playhead lines in the cable-2 panel",
              tb[1] > AXP_LINE_TOP + 0.005 and tb[3] <= 1.0,
              f"label spans y {tb[1]:.3f}-{tb[3]:.3f} (axes fraction); lines stop at {AXP_LINE_TOP}")
    bb1, _ = _text_box(fig, f"{f['peak_N'] / 1e3:.1f} kN = {f['peak_N'] / f['T0']:.1f} T0")
    bb2, _ = _text_box(fig, BANNER_SENTENCE.format(ms=f["peak_ms"]))
    man.check("the key-frame banner renders inside its asserted-free box", inside(bb1, BANNER_BOX) and inside(bb2, BANNER_BOX),
              f"banner {np.round(bb1, 3).tolist()}, {np.round(bb2, 3).tolist()} in box {list(BANNER_BOX)}")
    last_law = max(i for i, s in enumerate(fr) if s.kind == "law")
    draw(fig, fr[last_law], d, geom, cam, f, L, labels)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    right = max(fig.transFigure.inverted().transform([[t.get_window_extent(r).x1, 0]])[0][0]
                for t in fig.texts if t.get_position()[0] < 0.1 and 0.3 < t.get_position()[1] < 0.9)
    man.check("the slide's equations and notes end left of its right-hand boxes (x < 0.605)", right < 0.605,
              f"rightmost left-column text ends at {right:.3f}")
    xb, _ = _text_box(fig, BOX_TEXT)
    man.check("the DERIVED, NOT MEASURED text renders inside its red box", inside(xb, (0.615, 0.225, 0.975, 0.43)),
              f"text {np.round(xb, 3).tolist()} in box [0.615, 0.225, 0.975, 0.43]")
    check_captions(man, fig, fr, d, geom, cam, f, L, labels)


def check_captions(man: C.Manifest, fig, fr: list, d, geom, cam, f, L, labels) -> None:
    """Every caption as drawn (two_lines) has at most two lines, and its band -- the rounded box -- lies inside the
    frame and touches no other text, legend, axes (with its labels) or slide box, on the first and the last frame
    of its run."""
    bad, gaps, n_lines = [], [], []
    for cap in CAPTIONS:
        ks = [i for i, s in enumerate(fr) if s.caption == cap]
        n_lines.append(two_lines(cap).count("\n") + 1)
        for i in sorted({ks[0], ks[-1]}):
            draw(fig, fr[i], d, geom, cam, f, L, labels)
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            ct = [t for t in fig.texts if t.get_text() == two_lines(cap)]
            if len(ct) != 1:
                bad.append((cap[:30], i, "caption text not found once"))
                continue
            cb = ct[0].get_bbox_patch().get_window_extent(r)
            if cb.x0 < 0.01 * C.W or cb.x1 > 0.99 * C.W or cb.y0 < 0 or cb.y1 > C.H:
                bad.append((cap[:30], i, "outside the frame"))
            others = ([t.get_window_extent(r) for t in fig.texts if t is not ct[0] and t.get_text()]
                      + [lg.get_window_extent(r) for lg in fig.legends]
                      + [ax.get_tightbbox(r) for ax in fig.axes]
                      + [p.get_window_extent(r) for p in fig.patches])
            hit = [o for o in others if cb.overlaps(o)]
            if hit:
                bad.append((cap[:30], i, f"overlaps {len(hit)} artist(s)"))
            above = [o.y0 - cb.y1 for o in others if o.y0 >= cb.y1 and o.x1 > cb.x0 and o.x0 < cb.x1]
            below = [cb.y0 - o.y1 for o in others if o.y1 <= cb.y0 and o.x1 > cb.x0 and o.x0 < cb.x1]
            gaps.append(min(above + below + [C.H]))
    man.check("every caption renders in at most two lines, inside the frame, clear of every other text, legend, "
              "axes and slide box", not bad and max(n_lines) <= 2,
              f"{len(CAPTIONS)} captions, lines {n_lines}; smallest clearance to a neighbour {min(gaps):.0f} px; "
              f"problems {bad}")


def _style(ax):
    ax.grid(True, alpha=0.25)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C.MUTED)


def draw_traces_a(fig, f, t_ms, d):
    a0 = T0_MS + APPROACH[0]
    ii = np.arange(a0, t_ms + 1)
    tt = ii / 1000.0
    e2 = d["elongation"][ii, PARENT]
    ax = fig.add_axes(AX_E)
    _style(ax)
    ax.axhline(0.0, color=C.INK, lw=1.2)
    ax.text(85.05, 0.12, "rest length: chord = 12 m (cable re-engages)", fontsize=C.FS_SMALL, color=C.INK,
            ha="left", va="bottom")
    ax.plot(tt, e2, color=CAB[PARENT], lw=2.6)
    ax.plot([tt[-1]], [e2[-1]], "o", color=CAB[PARENT], ms=7)
    if t_ms >= int(round(f["t_deep"] * 1000)):
        ax.plot([f["t_deep"]], [-f["depth"]], "^", color=C.INK, ms=9)
        ax.text(f["t_deep"] + 0.1, -f["depth"] - 0.32, f"deepest: {f['depth']:.2f} m short, t = {f['t_deep']:.2f} s",
                fontsize=C.FS_SMALL, color=C.INK, ha="center", va="center")
    ax.set_xlim(85.0, 91.2); ax.set_ylim(-4.35, 0.75)
    ax.set_ylabel("cable 2: chord − 12 m, e₂ [m]")
    # just under the rest-length line, where the trace never comes (asserted in check_layout)
    ax.text(85.05, -0.42, E2_LABEL, fontsize=C.FS_SMALL, color=CAB[PARENT], ha="left", va="center")
    ax.tick_params(labelbottom=False)
    ax2 = fig.add_axes(AX_O, sharex=ax)
    _style(ax2)
    for c in (0, 1, 3, 4):
        ax2.plot(tt, f["T"][ii, c] / 1e3, color=CAB[c], lw=1.8, label=f"cable {c}")
        ax2.plot([tt[-1]], [f["T"][ii[-1], c] / 1e3], "o", color=CAB[c], ms=5)     # the current sample
    ax2.axhline(f["T0"] / 1e3, color=C.MUTED, ls=":", lw=1.3)
    ax2.text(1.008, f["T0"] / 1e3, "T0", transform=ax2.get_yaxis_transform(), fontsize=C.FS_SMALL, color=C.MUTED,
             ha="left", va="center")
    ax2.set_ylim(0.0, 1.25)
    ax2.set_ylabel("tension [kN]")
    ax2.set_xlabel("simulation time t [s]")
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, fontsize=C.FS_SMALL, handlelength=1.6,
               columnspacing=1.2, borderaxespad=0.2)


def draw_traces_b(fig, f, t_ms, d):
    lo = T0_MS + SLOW[0]
    ii = np.arange(lo, t_ms + 1)
    tau = ii - T0_MS
    T, e = f["T"], d["elongation"]
    now = t_ms - T0_MS
    axp = fig.add_axes(AX_P)
    axn = fig.add_axes(AX_N, sharex=axp)
    axr = fig.add_axes(AX_R, sharex=axp)
    for ax in (axp, axn):
        _style(ax)
        ax.tick_params(labelbottom=False)
    for ax in (axp, axn, axr):
        top = AXP_LINE_TOP if ax is axp else 1.0      # in the cable-2 panel both lines stop below the t0 label
        ax.axvline(0.0, color=C.INK, ls="--", lw=1.0, zorder=1, ymax=top)
        ax.axvline(now, color=C.MUTED, lw=1.0, alpha=0.6, zorder=1, ymax=top)
    axp.plot(tau, T[ii, PARENT] / 1e3, color=CAB[PARENT], lw=2.8, zorder=3)
    axp.set_xlim(*SLOW); axp.set_ylim(-0.4, 16.0)
    axp.set_ylabel("tension [kN]")
    axp.text(0.012, 0.93, "cable 2", transform=axp.transAxes, fontsize=C.FS_SUB, weight="bold", color=CAB[PARENT],
             ha="left", va="top")
    axp.text(8, 15.2, T0_LABEL, fontsize=C.FS_TINY, color=C.INK, ha="left", va="center")
    if now >= f["peak_ms"]:
        axp.plot([f["peak_ms"]], [f["peak_N"] / 1e3], "o", color=CAB[PARENT], ms=8, zorder=4)
        axp.annotate(f"{f['peak_N'] / 1e3:.1f} kN = {f['peak_N'] / f['T0']:.1f} T0\nat t0 + {f['peak_ms']} ms",
                     xy=(f["peak_ms"], f["peak_N"] / 1e3), xytext=(190, 12.6), fontsize=C.FS_SMALL, weight="bold",
                     color=CAB[PARENT], ha="left", va="top", linespacing=1.25, zorder=5,
                     bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"),     # the playhead passes behind
                     arrowprops=dict(arrowstyle="-", lw=1.0, color=CAB[PARENT], shrinkA=2, shrinkB=5))
    for c in (0, 1, 3, 4):
        axn.plot(tau, T[ii, c] / 1e3, color=CAB[c], lw=2.0, label=f"cable {c}", zorder=3)
    axn.axhline(f["T0"] / 1e3, color=C.MUTED, ls=":", lw=1.3)
    axn.text(1.008, f["T0"] / 1e3, "T0", transform=axn.get_yaxis_transform(), fontsize=C.FS_SMALL, color=C.MUTED,
             ha="left", va="center")
    axn.set_ylim(-0.12, 4.3)
    axn.set_ylabel("tension [kN]")
    axn.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, fontsize=C.FS_SMALL, handlelength=1.6,
               columnspacing=1.2, borderaxespad=0.2)
    for c, dy in ((3, 0.25), (4, 0.25)):
        fo = f["follow"][c]
        if t_ms >= fo["i_peak"]:
            x = fo["i_peak"] - T0_MS
            axn.plot([x], [fo["T_peak"] / 1e3], "o", color=CAB[c], ms=6, zorder=4)
            axn.text(x, fo["T_peak"] / 1e3 + dy, f"{fo['T_peak'] / 1e3:.1f} kN", fontsize=C.FS_SMALL, color=CAB[c],
                     weight="bold", ha="center", va="bottom")
    # slack raster: e <= 0
    for c in range(5):
        s = (e[ii, c] <= 0.0).astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, s, 0]))
        spans = [(tau[a], (tau[b - 1] - tau[a]) + 1) for a, b in zip(edges[::2], edges[1::2])]
        if spans:
            axr.broken_barh(spans, (c - 0.36, 0.72), facecolors=CAB[c], alpha=0.85)
    for c, ms in f["slack_ms"].items():       # left of the onset: that stretch of the row is empty
        if now >= ms:
            axr.text(ms - 8, c, f"+{ms} ms", fontsize=C.FS_TINY, weight="bold", color=CAB[c], ha="right", va="center",
                     bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"), zorder=5)
    axr.set_ylim(4.6, -0.6)
    axr.set_yticks(range(5)); axr.set_yticklabels([f"cable {c}" for c in range(5)])
    for lab, c in zip(axr.get_yticklabels(), range(5)):
        lab.set_color(CAB[c])
    axr.set_xticks(range(SLOW[0], SLOW[1] + 1, 200))
    axr.set_xlabel("time from cable 2's re-engagement, t − t0 [ms]")
    axr.set_title("slack: chord at or below rest length (e ≤ 0)", fontsize=C.FS_SMALL, loc="left", pad=4)
    for sp in ("left", "bottom"):
        axr.spines[sp].set_color(C.MUTED)


def draw_title_card(fig, spec, f):
    fig.text(0.5, 0.63, TITLE, fontsize=40, weight="bold", color=C.INK, ha="center", va="center")
    C.caption(fig, two_lines(spec.caption), y=0.50)
    for k, line in enumerate(TITLE_DETAILS):      # the numbers are the verified cell's (verify)
        fig.text(0.5, 0.375 - 0.042 * k, line.format(seed=SEED, cell=CELL, intensity=f["intensity"], T0=f["T0"],
                                                   k_h=f["k_h"]),
                 fontsize=C.FS_BODY, color=C.INK, ha="center", va="center")
    fig.text(0.015, 0.034, SELECTION_TEXT, fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="bottom")
    fig.text(0.015, 0.012, FOOTER_DATA, fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="bottom")


def draw_law(fig, spec, L, f):
    """The transmission-law slide, built up in four steps (one per caption)."""
    from matplotlib.patches import FancyBboxPatch
    C.title(fig, "The transmission law: derived, not measured",
            "Paper §III: rigid-body algebra at the testbed's parameters (Table I). A slide, not simulation output.")
    x0, fs_eq, fs_note = 0.045, 25, C.FS_SMALL + 1
    rows = [
        (0.818, r"$J \;=\; \int_0^{\pi/\omega_e} T_{\mathrm{peak}}\,\sin(\omega_e t)\,\mathrm{d}t \;=\; "
                r"\frac{2\,T_{\mathrm{peak}}}{\omega_e}$",
         r"The tension pulse is approximated by a half sine lasting $\pi/\omega_e \approx$ " + f"{L['contact']:.2f} s."),
        (0.683, r"$\Delta v_L \;=\; J\,/\,m_L$",
         "The shared payload takes the impulse. This is an approximation: the contact is\n"
         r"comparable to the payload's half-period $\pi/\omega_L \approx$ " + f"{L['half_L']:.2f} s"
         + "; the uncertainty band covers it."),
        (0.528, r"$\Delta T_{\mathrm{neighbour}} \;\simeq\; \frac{k\,\Delta v_L}{\omega_L} \;=\; "
                r"\frac{2k}{m_L\,\omega_L\,\omega_e}\,T_{\mathrm{peak}} \;=\; 0.44\;T_{\mathrm{peak}}$",
         r"The other cables hold the payload, which responds at $\omega_L = \sqrt{4k/m_L}$; the projection" + "\n"
         r"$\cos(\sigma_i - \sigma_j)$ onto each neighbour's chord ($\sigma$: chord angle) reduces the swing further."),
        (0.382, r"$T_{\mathrm{peak}} \;\gtrsim\; T_0\,/\,0.44 \;\approx\; 2.3\;T_0$",
         "When a neighbour's swing exceeds its pretension, that neighbour can go slack."),
    ]
    shown = {1: 2, 2: 3, 3: 4, 4: 4}[spec.step]
    for y, eq, note in rows[:shown]:
        fig.text(x0, y, eq, fontsize=fs_eq, color=C.INK, ha="left", va="center")
        fig.text(x0 + 0.005, y - 0.043, note, fontsize=fs_note, color=C.MUTED, ha="left", va="top", linespacing=1.35)
    if spec.step >= 2:          # the plant's values, computed in derive_law()
        fig.patches.append(FancyBboxPatch((0.615, 0.475), 0.36, 0.37, boxstyle="round,pad=0.006,rounding_size=0.012",
                                          transform=fig.transFigure, fc="#f7f7f8", ec=C.FAINT, lw=1.0, zorder=0))
        lines = [("At the plant's constants (tether/physics/constants.py)", C.MUTED),
                 (r"$k = 1.7\times10^{5}$ N/m,   $m_L = 2500$ kg,   $m_A = 600$ kg", C.INK),
                 (r"$\omega_e = \sqrt{k/\mu}$,  $\mu = m_A m_L/(m_A+m_L)$:   " + f"{L['w_e']:.1f} rad/s", C.INK),
                 (r"$\omega_L = \sqrt{4k/m_L}$:   " + f"{L['w_L']:.1f} rad/s", C.INK),
                 (r"$2k\,/\,(m_L\,\omega_L\,\omega_e)$ = " + f"{L['gain']:.3f}   (half-sine pulse)", C.INK),
                 (f"rectangular pulse: {L['rect']:.2f}   (threshold {L['thr_rect']:.1f} " + r"$T_0$)", C.INK),
                 (f"declared uncertainty band on the gain: [{BAND[0]}, {BAND[1]}]", C.INK)]
        for k, (txt, col) in enumerate(lines):
            fig.text(0.63, 0.815 - 0.049 * k, txt, fontsize=C.FS_SMALL + 1, color=col, ha="left", va="center")
    if spec.step >= 3:          # where this event sits against the derived threshold
        ax = fig.add_axes([0.07, 0.235, 0.50, 0.042])
        ax.set_xscale("log"); ax.set_xlim(0.9, 32); ax.set_ylim(0, 1)
        ax.axvspan(*L["thr_band"], color=C.ACCENT, alpha=0.20, lw=0)
        ax.axvline(L["thr"], color=C.INK, lw=1.8); ax.axvline(L["thr_rect"], color=C.INK, lw=1.2, ls="--")
        ax.plot([f["peak_N"] / f["T0"]], [0.5], "o", color=CAB[PARENT], ms=13, zorder=4)
        ax.text(f["peak_N"] / f["T0"] / 1.25, 1.2, f"this event (measured): {f['peak_N'] / f['T0']:.1f} " + r"$T_0$",
                fontsize=C.FS_SMALL, weight="bold", color=CAB[PARENT], ha="center", va="bottom")
        ax.text(L["thr"] * 1.04, 1.2, f"{L['thr']:.1f} (half sine)", fontsize=C.FS_SMALL, color=C.INK, ha="left", va="bottom")
        ax.text(L["thr_rect"] * 0.97, 1.2, f"{L['thr_rect']:.1f} (rect.)", fontsize=C.FS_SMALL, color=C.INK,
                ha="right", va="bottom")
        ax.set_yticks([])
        ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.xaxis.set_minor_formatter(plt_null_formatter())
        ax.tick_params(axis="x", labelsize=C.FS_SMALL, pad=3)
        ax.set_xlabel(r"$T_{\mathrm{peak}}\,/\,T_0$ (log scale); shaded: derived threshold for gains 0.9 to 0.2, "
                      f"{L['thr_band'][0]:.1f}–{L['thr_band'][1]:.1f} " + r"$T_0$", fontsize=C.FS_SMALL, labelpad=3,
                      color=C.INK)
        for s in ("left", "right", "top"):
            ax.spines[s].set_visible(False)
    if spec.step >= 4:
        fig.patches.append(FancyBboxPatch((0.615, 0.225), 0.36, 0.205, boxstyle="round,pad=0.006,rounding_size=0.012",
                                          transform=fig.transFigure, fc="#fbeceb", ec=C.SLACK, lw=1.6, zorder=0))
        fig.text(0.63, 0.40, "DERIVED, NOT MEASURED", fontsize=C.FS_SUB + 2, weight="bold", color=C.SLACK,
                 ha="left", va="center")
        fig.text(0.63, 0.36, BOX_TEXT, fontsize=C.FS_SMALL + 2, color=C.INK, ha="left", va="top", linespacing=1.45)
    fig.text(0.015, 0.012, "Sources: Paper/Sections/Section_III_Coupling.tex (impulse, transmission and threshold "
             "equations; declared band); tether/physics/constants.py; Table I in Section_II_Setting.tex",
             fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="bottom")
    C.caption(fig, two_lines(spec.caption), y=CAPTION_Y_LAW)


def plt_null_formatter():
    from matplotlib.ticker import NullFormatter
    return NullFormatter()


def draw(fig, spec: Spec, d, geom, cam, f, L, labels):
    fig.clf()
    if spec.kind == "title":
        draw_title_card(fig, spec, f)
    elif spec.kind == "law":
        draw_law(fig, spec, L, f)
    else:
        b = spec.kind in ("b", "end")
        furniture(fig, clock_t=spec.t_ms / 1000.0, speed=spec.speed, tau_ms=(spec.t_ms - T0_MS) if b else None,
                  caption=spec.caption)
        draw_fleet_panel(fig, d, geom, cam, spec.t_ms, f, labels, banner=spec.kind == "end")
        (draw_traces_b if b else draw_traces_a)(fig, f, spec.t_ms, d)


# ----------------------------------------------------------------------------- manifest values

def register(man: C.Manifest, f: dict, L: dict, sp: dict) -> None:
    npz, runs, decl = str(NPZ), str(RUNS), str(DECL)
    run_key = f"{runs}: runs[cell == {CELL}, seed == {SEED}]"
    mark = f"{run_key}.marks[cable == 2, t_up == 91.486 s]"
    V = man.value
    V("seed", SEED, "", f"{run_key}.seed; {npz}:seed", "subtitle, title card")
    V("cell name i035_T0600_ks3 (intensity 0.35, T0 0600 N, k_sigma 3)", CELL, "", f"{decl}: cells.grid[name]", "subtitle")
    V("T0 = 600 N (0.6 kN on the T0 reference lines and in the formation note)", f["T0"], "N",
      f"{decl}: cells.grid[{CELL}].T0_N", "")
    V("formation note: chord-angle std 35.9°", round(f["chord_std"], 4), "deg",
      f"{RESULTS}: trade_off[cell == {CELL}].chord_world_std_deg",
      "the cell's world chord-angle circular std, maximum over vessels (paper Sec. VI, Fig. window caption); "
      "= the 35.9 deg of Sec. VI; a cell statistic, not this run's")
    V("key: line width capped at 12 kN", C.WIDTH_FULL_N, "N", f"{COMMON_SRC}: WIDTH_FULL_N",
      "common.draw_fleet draws lw = 1.4 + 5.0 min(T / WIDTH_FULL_N, 1)")
    V("weather intensity 0.35", f["intensity"], "", f"{decl}: cells.grid[{CELL}].intensity", "subtitle")
    V("k_h = 477 N·m/rad", f["k_h"], "N m/rad", f"{decl}: cells.grid[{CELL}].k_h_N_m_per_rad", "title card")
    V("cable rest length 12 m", 12.0, "m", f"{CONSTANTS}: CABLE_REST_LENGTH", "captions, e2 panel")
    V("cable 2 slack since 82.2 s", round(f["onset_t"], 3), "s", f"{mark}: t_up - dwell",
      "log's first e <= 0 sample of the excursion 82.247 s (checked)")
    V("deepest: 3.82 m short at 87.96 s", [round(f["depth"], 4), round(f["t_deep"], 3)], "m, s", f"{mark}: depth, t_deep",
      "= min e2 of the 1 ms log (checked)")
    V("closing speed at re-engagement 1.78 m/s", round(f["v_return"], 4), "m/s", f"{mark}: v_return",
      "interpolated crossing rate of e2 = chord - 12 m, so the chord lengthens at this rate as the slack closes "
      "(caption: 'its chord lengthening at 1.78 m/s'); log rate at 91.486 s is 1.779 m/s")
    V("t0 = 91.486 s", round(f["t_up_exact"], 6), "s", f"{mark}: t_up",
      "record t_up 91.48636 s (interpolated crossing); shown to the 1 ms grid; all ms offsets are from 91.486 s")
    V("cable-2 peak 13.5 kN", round(f["peak_N"], 1), "N", f"{npz}: time/elongation/rate/alive -> k e + c edot (e > 0), "
      f"max over [t_up, t_up + 0.4 s]; = {mark}.T_peak", "")
    V("peak at t0 + 66 ms", f["peak_ms"], "ms", f"{npz}: time of that maximum on the 1 ms grid", "")
    V("peak = 22.5 T0", round(f["peak_N"] / f["T0"], 3), "T0", "peak / declared T0", "derived ratio of two registered numbers")
    V("cables 0, 4, 3 reach rest length (e <= 0) at +30, +52, +70 ms", f["slack_ms"], "ms",
      f"{npz}: first 1 ms sample after 91.486 s with elongation <= 0", "raster labels and caption")
    V("cables 0, 4, 3 first carry no tension at +21, +43, +60 ms", f["zero_ms"], "ms",
      f"{npz}: first 1 ms sample after 91.486 s with tension == 0", "visible as the dashed-red switch in the fleet view; not printed")
    for c in (3, 4):
        fo = f["follow"][c]
        V(f"cable {c} follow-on peak {fo['T_peak'] / 1e3:.1f} kN", round(fo["T_peak"], 1), "N",
          f"{run_key}.marks[cable == {c}, t_up == {fo['t_up']:.3f} s].T_peak",
          f"= max of the npz tension over [t_up, t_up + 0.4 s], reached at +{fo['i_peak'] - T0_MS} ms")
    V("simulation clock (every animated frame)", [(T0_MS + APPROACH[0]) / 1e3, (T0_MS + SLOW[1]) / 1e3], "s",
      f"{npz}: time (the 1 ms sample drawn)", "includes the 20 s warm-up")
    V("t − t0 clock", [SLOW[0], SLOW[1]], "ms", "frame's 1 ms sample minus 91.486 s", "")
    V("playback: real time", round(sp["speed_a"], 4), "x", "frame schedule: 1000/30 ms of simulation per frame",
      "Presentation/cache/snap_cascade_window.npz: frame_t_ms")
    V("playback: ×1/20 slow motion", [round(1 / sp["speed_b1"], 2), round(1 / sp["speed_b2"], 2)], "1/x",
      "frame schedule: linspace between the key instants, rounded to the 1 ms log",
      "nominal 1.667 ms per frame; 1.664 before the peak, 1.669 after")
    V("scale bar 5 m", 5.0, "m", "common.draw_fleet", "plant-size hulls 3.0 x 1.0 m")
    V("selection: seeds 7101–7106 scanned", list(SCAN), "", f"{CASCADE_SRC}: main(seeds=...); {runs}: runs[seed in scan]",
      "seed 7105 holds the largest mark among them (checked)")
    V("selection: 2nd-largest of the cell's 510 scored marks", [f["rank"], f["n_scored"]], "",
      f"{runs}: runs[cell == {CELL}, pilot == false].marks.T_peak; {RESULTS}: trade_off[{CELL}].marks", "")
    V("selection: median 0.94 kN", round(f["median"], 1), "N", f"{runs}: median of those 510 T_peak", "")
    V("29 marks in this run (footer)", f["n_marks"], "", f"{run_key}.n_marks", "all reproduced by the tracker replay")
    V("10 ms state rows, 1 ms log (footer)", [10, 1], "ms", f"{npz}: state_time, time (exact grids asserted)", "")
    V("20 s warm-up (footer)", WARMUP, "s", f"{CASCADE_SRC}: FleetRunSpec default warmup; {run_key}.end_time 320 = 20 + 300", "")
    V("'three others go slack' (title card)", sorted(f["slack_ms"]), "cables",
      f"{npz}: cables reaching e <= 0 within +70 ms of 91.486 s", "cables 0, 3, 4")
    V("contact pulse π/ω_e ≈ 0.17 s", round(L["contact"], 4), "s", f"derived: pi / sqrt(k / mu), {CONSTANTS}", "DERIVED")
    V("payload half-period π/ω_L ≈ 0.19 s", round(L["half_L"], 4), "s", f"derived: pi / sqrt(4k / m_L), {CONSTANTS}",
      f"DERIVED; {SEC3} prints 'a half-period of 0.19 s' and calls the impulse treatment an approximation")
    V("k = 1.7e5 N/m, m_L = 2500 kg, m_A = 600 kg", [L["k"], L["mL"], L["mA"]], "N/m, kg",
      f"{CONSTANTS}: CABLE_STIFFNESS, LOAD_MASS, VESSEL_MASS", "= paper Table I")
    V("ω_e = 18.7 rad/s", round(L["w_e"], 4), "rad/s", f"derived: sqrt(k/mu), mu = PAIR_REDUCED_MASS (tether/physics/fleet.py)",
      "DERIVED; = 2 pi / ENGAGEMENT_PERIOD (tether/campaign/v2/events.py); Table I 18.7")
    V("ω_L = 16.5 rad/s", round(L["w_L"], 4), "rad/s", "derived: sqrt(4k/m_L) (paper Sec. III)", "DERIVED; Table I 16.5")
    V("transmission gain 2k/(m_L ω_L ω_e) = 0.44 (0.440 on the slide; 'up to about 0.44' in the caption)",
      round(L["gain"], 5), "", f"derived from {CONSTANTS}; printed in {SEC3} eq. transmission",
      "DERIVED, NOT MEASURED (paper Sec. III; closeout 'claimed a measured transmission ratio'); 'up to': the "
      "paper's projection cos(sigma_i - sigma_j) onto each neighbour's chord reduces it further")
    V("threshold ≈ 2.3 T0", round(L["thr"], 4), "T0", f"derived: 1/gain; {SEC3} eq. threshold", "DERIVED")
    V("rectangular pulse 0.69, threshold 1.4 T0", [round(L["rect"], 5), round(L["thr_rect"], 4)], ", T0",
      f"derived: k pi/(m_L w_L w_e); {SEC3}", "DERIVED")
    V("declared band [0.2, 0.9]", list(BAND), "", f"{SEC3}; tether/campaign/v2/claim_a.py 9_cascade_transmission_ratio (CONJ)",
      "declared uncertainty band, not a measured interval")
    V("derived threshold across the band 1.1–5.0 T0", [round(x, 4) for x in L["thr_band"]], "T0", "derived: 1/0.9, 1/0.2", "DERIVED")
    V("this event 22.5 T0 on the threshold strip", round(f["peak_N"] / f["T0"], 3), "T0", "as 'peak = 22.5 T0'", "measured")


# ----------------------------------------------------------------------------- main

def still_indices(fr: list[Spec]) -> list[int]:
    want, out = [], []
    kinds = [(s.kind, s.caption) for s in fr]
    for cap in CAPTIONS:
        ks = [i for i, (_, c) in enumerate(kinds) if c == cap]
        want.append(ks[len(ks) // 2])
    for i in want:
        if i not in out:
            out.append(i)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="verify the records and stop")
    ap.add_argument("--stills", action="store_true", help="also write inspection PNGs to the development directory ($TETHER_DEV_DIR)")
    ap.add_argument("--no-video", action="store_true", help="skip the video (with --stills)")
    args = ap.parse_args(argv)
    C.ensure_dirs()
    man = C.Manifest(
        name=NAME, title=TITLE,
        story=("One recorded cascade from the inside. In seed 7105 of the 0.6 kN cell, cable 2 has been slack for "
               "9.2 s; it re-engages with its chord lengthening at 1.78 m/s and peaks 66 ms later at 13.5 kN, 22.5 "
               "times its pretension. Within "
               "70 ms of the re-engagement three other cables have gone slack, and at the peak those three carry no "
               "tension. The paper's transmission law says why this is possible - the snap's impulse (an "
               "approximation: the contact is comparable to the payload's half-period) moves the shared payload, "
               "swinging the other cables' tension by up to about 0.44 of the peak, less for cables at an angle to "
               "cable 2 - but that gain is derived, not measured, and one event does not measure it; the campaign "
               "measured its consequence (the attribution statistics)."))
    for p in (NPZ, RUNS, DECL, RESULTS, CONSTANTS, SEC2, SEC3, SEC6, CASCADE_SRC, COMMON_SRC):
        man.source(p)
    print("verifying the record ...", flush=True)
    d = dict(np.load(C.REPO / NPZ))
    f = verify(man, d)
    L = derive_law(man)
    fr = timeline(f)
    sp = check_timeline(man, fr, f)
    man.selection = (SELECTION_TEXT.replace("Selection: ", "") +
                     " Seed 7105, cell i035_T0600_ks3; the cable-2 mark at t_up = 91.486 s. Windows fixed by the event: "
                     "the storyboard's 6 s real-time approach ending 0.4 s before t0, slow motion ×1/20 from t0 - 0.4 s "
                     "to t0 + 0.6 s, a pause and the closing key frame at cable 2's peak (max over [t_up, t_up + 0.4 s], "
                     "the campaign's T_peak window).")
    man.caveats = [
        "One event, chosen for clarity, not typicality: the 2nd-largest of the cell's 510 scored marks (median 0.94 kN).",
        "Recording run: severance is scored, no cable is cut; 'snap' is the re-engagement's tension pulse.",
        "The 0.44 transmission gain is derived, not measured; the campaign never measured a neighbour's swing (the "
        "test was in Phase 2, not launched) and this one event does not measure it (on screen: the slide's red box "
        f"for {sum(s.kind == 'law' and s.step == 4 for s in fr) / C.FPS:.1f} s, and the captions 'The gain of 0.44 "
        "is derived, never measured; ...' and 'One recorded event cannot measure the gain either ...'). The slide "
        "computes the gain; the clip does not compare it with the event's tensions. The captions call the paper's "
        "declared uncertainty band on the transmission ratio, [0.2, 0.9], 'its uncertainty band', and give the "
        "paper's stated reason for it, the pulse-shape sensitivity (the band also covers the impulse approximation, "
        "said in the slide's note).",
        f"One event, seen from the inside (the paper's framing of its Fig. 3), not evidence for the mechanism. The "
        f"plant state shows the sequence the law describes - from t0 to the peak the payload's x-velocity rises "
        f"({f['vx0']:.2f} -> {f['vxp']:.2f} m/s, not shown on screen) while cables 0, 3, 4 shorten and unload - but "
        f"one event neither measures the gain nor shows that such sequences exceed chance. That evidence is the "
        f"post-hoc attribution statistic (clip cascade_statistics), whose time-shift baseline also removes co-timing "
        f"by shared weather, so it bounds the transmission effect rather than isolating it (paper Sec. III).",
        "The closing caption states the paper's general point (Sec. III: the lines are coupled through the shared "
        "payload, and a model that contains only one cable has no representation of that coupling); its 'can "
        "unload' is the derived law's possibility, not something this one event measures.",
        "On screen the gain is 'up to about 0.44' of the peak, less for cables at an angle to cable 2 (the paper's "
        f"projection cos(sigma_i - sigma_j)); cable 1, whose chord is {f['ang12']:.0f} deg from cable 2's here, "
        f"barely changes (checked), consistent with that qualifier but not a measurement of it.",
        "The formation is named parallel (cables commanded along the tow axis); at T0 = 0.6 kN they spread (the "
        "cell's chord-angle std 35.9 deg, a cell statistic, not this run's) and, with no contact modelled, lines "
        "and hulls overlap.",
        "The fleet view's key follows common.draw_fleet: line width grows with tension and saturates at 12 kN, so "
        "cable 2's 13.5 kN peak is drawn at the cap.",
        "Body poses are interpolated linearly between 10 ms state rows (sub-centimetre motion in slow motion is "
        "invisible at this scale); tensions are the plant's law on the 1 ms log; ms offsets are on the 1 ms grid "
        "from 91.486 s (the record's t_up is 91.48636 s).",
        "Carrying no tension (tension clamped to 0) precedes slack (e <= 0) by 9-10 ms here: at the peak cable 3 "
        "carries no tension while its chord is still longer than rest length.",
        "A single planar testbed, parallel formation, one cell (T0 = 600 N, intensity 0.35); times include the 20 s warm-up.",
    ]
    # the drawn window, the tracker marks and the frame schedule, for audit
    lo, hi = T0_MS + APPROACH[0], T0_MS + SLOW[1]
    np.savez_compressed(
        CACHE, source=str(NPZ), source_sha256=man.sources[str(NPZ)], t0=T_UP_REC,
        time=d["time"][lo:hi + 1], elongation=d["elongation"][lo:hi + 1], rate=d["rate"][lo:hi + 1],
        alive=d["alive"][lo:hi + 1], tension=f["T"][lo:hi + 1],
        state_time=d["state_time"][lo // 10: hi // 10 + 2], state=d["state"][lo // 10: hi // 10 + 2],
        tracker_t_up=f["tracker"]["t_up"], tracker_T_peak=f["tracker"]["T_peak"], tracker_cable=f["tracker"]["cable"],
        frame_kind=np.array([s.kind for s in fr]), frame_t_ms=np.array([-1 if s.t_ms is None else s.t_ms for s in fr]))
    man.source(CACHE)
    register(man, f, L, sp)
    # drawing preconditions, still before any frame: one fixed camera, fixed label positions, a free banner area
    geom = _Geom(d["geometry_load"], d["geometry_vessel"])
    cam = camera(d, T0_MS + APPROACH[0], T0_MS + SLOW[1])
    labels = label_plan(d, fr)
    worst = min(float(c.min()) for _, c in labels.values())
    moves = sum(int(np.any(labels[t1][0] != labels[t0][0])) for t0, t1 in zip(list(labels)[:-1], list(labels)[1:]))
    man.check("vessel number labels keep >= 0.5 m from every other hull and the payload in every drawn frame",
              worst >= 0.5, f"{len(labels)} instants; smallest clearance {worst:.2f} m; label position changes {moves}")
    check_banner(man, d, cam, f)
    check_free_box(man, d, cam, NOTE_BOX, [s.t_ms for s in fr if s.t_ms is not None],
                   "the formation note (every drawn instant)")
    fig = C.new_frame()
    check_layout(man, fig, fr, d, geom, cam, f, L, labels)     # renders to the figure only; writes nothing
    print(f"{len(man.checks)} checks passed; {len(fr)} frames ({len(fr) / C.FPS:.1f} s) scheduled", flush=True)
    if args.check:
        for c in man.checks:
            print(" -", c["check"], "|", c["detail"])
        return

    if args.stills:
        STILLS.mkdir(parents=True, exist_ok=True)
        for i in still_indices(fr):
            draw(fig, fr[i], d, geom, cam, f, L, labels)
            fig.savefig(STILLS / f"still_{i:04d}_{fr[i].kind}.png", dpi=C.DPI)
        print(f"stills -> {STILLS}")
        if args.no_video:
            return
    w = C.writer()
    w.extra_args = list(w.extra_args) + ["-threads", "2"]      # shared 16-core host
    n, prev = 0, None
    with w.saving(fig, str(CLIP), C.DPI):
        for spec in fr:
            if spec != prev:
                draw(fig, spec, d, geom, cam, f, L, labels)
                prev = spec
            w.grab_frame()
            n += 1
            if n % 300 == 0:
                print(f"  {n}/{len(fr)} frames", flush=True)
    man.check("frames written equal the schedule", n == len(fr), f"{n}")
    man.frames = n
    man.write(CLIP)
    print(f"wrote {CLIP.relative_to(C.REPO)}")


if __name__ == "__main__":
    main()
