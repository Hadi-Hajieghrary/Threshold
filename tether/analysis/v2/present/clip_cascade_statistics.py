"""Clip C3 ``cascade_statistics`` -- is it chance?  (Presentation/STORYBOARD.md, section C3)

A data animation drawn only from a campaign record.  No fleet motion is shown: the record holds
the slack marks, not the plant state, and nothing is re-simulated.  Every number on screen is the
paper figure's own statistic (tether/analysis/v2/paperfig/fig_attribution.py) recomputed from the
record, or a per-mission piece of that same computation, and is registered in the manifest.

Record read (the only campaign file this module opens)
------------------------------------------------------
records/phase2/phase2_records.npz -- the v1 Phase 2 grid (the paper's "cascade grid"): parallel
formation, recording runs (cables are scored, never cut), 25 cells x seeds 2003-2022, 600 s scored
after a 20 s warm-up; the record's clock includes the warm-up (tether/campaign/phase2.py).
    marks_t_down      slack onset of each mark: the chord falls to rest length       [s]
    marks_t_up        re-engagement of each mark: the chord rises above rest length   [s]
    marks_cable       cable 0..4                                                       [-]
    marks_seed        mission seed                                                     [-]
    marks_cell_index  index into cell_names                                            [-]
    cell_names        "T<T0 [N]>_k<k_h [N m/rad]>_I<intensity>", plus the NULL cell
    cell_exposure     recorded seconds per cell -- read only to confirm that the densest cell by
                      events is also the densest per recorded second, and that the chosen cell's
                      missions all ran their full 600 s
records/phase2/phase2_manifest.json -- read only for its key record_sha256, which the record's
sha256 must equal (the committed Phase 2 record, not a copy).
Code reused, not modified: fig_attribution.attribution_by_cell, _burst_heads, _n_cross, _pretty and
its constants WINDOW (3 s), BURST_GAP (2 s), MIN_MARKS (20), SPAN_START (20 s), N_SHIFTS (200),
SEED (20260914).  Paper text read for one consistency check: Paper/Sections/Section_III_Coupling.tex.
Nothing is re-simulated: every per-mission number is computed from the record with fig_attribution's
own functions; the cache below only stores those computed values (it is not a physics replay).

The statistic (the module's, restated)
--------------------------------------
Per (cell, seed) mission: events are burst heads (a mark whose onset lies within 2 s of the previous
re-engagement on the SAME cable is a bounce, absorbed into its event); an event with onset t is
cross-attributed when any re-engagement (every mark's t_up) on a DIFFERENT cable lies in [t-3 s, t).
Share = cross-attributed events / events, pooled over the cell's 20 missions; cells with < 20 marks
are not scored.  Chance baseline (post hoc): each mission's whole re-engagement train is circularly
shifted by one uniform offset modulo [20 s, the mission's last mark], onsets fixed; 200 offsets per
mission from numpy.random.default_rng(20260914), drawn in (cell index, ascending seed) order.

Selection rule (on screen and in the manifest; chosen for clarity, not typicality)
---------------------------------------------------------------------------------
cell     the lower median of the 18 scored cells ranked by measured share (9th of 18, ascending)
mission  that cell's seed with the most events (ties -> lowest seed)
window   the 90 s window [s, s + 90) with s on whole seconds in [20 s, last mark - 90 s] holding the
         most event onsets (ties -> earliest)
shift    the first of the 200 offsets the module's rng draws for that mission

Asserted before any frame is drawn (Manifest.check)
----------------------------------------------------
* the record is the one fig_attribution reads, and its sha256 equals phase2_manifest.json
  record_sha256; 25 cells; only the statistics seeds 2003-2022;
* attribution_by_cell scores 18 of the 25 cells (7 dropped, the NULL cell among them);
* re-running the module's loop here, keeping each mission's offsets, reproduces attribution_by_cell
  exactly (events, measured share, chance mean and 97.5th percentile of every cell) -- so the
  offsets used for the animated shift are the module's own;
* measured share 26-68 % (rounded); above the chance 97.5th percentile in 18/18 cells; measured /
  chance mean from 1.4 to 21 (rounded); the densest cell (most events, and most events per recorded
  second) has the smallest factor and a chance mean of 49 % (rounded);
* the corrected paper (Section III) states the same numbers and calls the analysis post hoc;
* the chosen cell's missions all ran their full 600 s (cell_exposure = 20 x 600 s), so the
  whole-mission strip and counter span t = 20-620 s;
* the selected mission's per-event flags sum to fig_attribution._n_cross (measured) and its shift
  span is [20 s, last mark]; for every one of its 200 offsets the per-event flags sum to _n_cross
  and to the count the module itself added for that offset (so the window's 200-shift mean shown on
  screen is taken over the module's own shifts).
At draw time further asserts guard the animation: the sweep passes every onset in the window, the
slide ends exactly on the module's shifted train, and at the end of each section every visible text
lies inside the 1920x1080 canvas.  Captions are checked after rendering:
<= 2 lines, each on screen >= max(4 s, words / 2.5 s).  Every caption, the title card and the end card are
written as complete sentences; the holds (lead-in, end pause, shift hold, chart hold, final hold) are sized
from the captions' reading times, so no simulation instant or drawn element changed when the narration was
rewritten.  Total duration: at least 75 % of the storyboard's 45 s and at most 130 % of the 49.27 s
(1478 frames) the clip ran before its captions were rewritten as sentences.

Time labelling (clock slot, top right)
--------------------------------------
sweep     the simulation clock of the playhead, t = 395 s + i x 8/30 s, "(x8)"; "(paused)" on the
          lead-in and end pause
shift     the frames show the whole 395-485 s window with a counterfactual (shifted) train, no
          instant: the slot reads "window t = 395-485 s | baseline shift: counterfactual, no playback"
chart     pooled over whole missions, no instant: "pooled over whole missions | no playback"

Outputs: Presentation/clips/cascade_statistics.mp4, Presentation/manifests/cascade_statistics.json,
Presentation/cache/cascade_statistics_attribution.npz (values computed from the record, not a
replay: the 18 rows, the selected mission's marks and flags, its 200 offsets with their
whole-mission and window counts).  Run:  python3 -m tether.analysis.v2.present.clip_cascade_statistics
(--preview writes a few PNG stills to the development directory ($TETHER_DEV_DIR) instead of encoding).
"""
from __future__ import annotations

import tempfile
import os
import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory

from tether.analysis.v2.present import common as C
from tether.analysis.v2.paperfig import fig_attribution as FA

NAME = "cascade_statistics"
RECORD_REL = "records/phase2/phase2_records.npz"
FA_REL = "tether/analysis/v2/paperfig/fig_attribution.py"
PAPER_REL = "Paper/Sections/Section_III_Coupling.tex"
P2_MANIFEST_REL = "records/phase2/phase2_manifest.json"
CLIP = C.CLIPS / f"{NAME}.mp4"
CACHE_FILE = C.CACHE / f"{NAME}_attribution.npz"
STILLS = Path(os.environ.get("TETHER_DEV_DIR", Path(tempfile.gettempdir()) / "tether_dev")) / "present" / "cascade_statistics"

N_GRID_CELLS = 25
STAT_SEEDS = tuple(range(2003, 2023))
SCORED_S = 600.0            # scored seconds per mission (phase2.DURATION)
WIN_LEN = 90.0              # s, timeline window
SPEED = 8.0                 # playback during the sweep: x8
SLIDE_S = 3.6               # s of video for the animated shift
SHIFT_MOTION = (0.6, SLIDE_S, 0.5)   # s of video: fade, slide, re-score (the hold follows, sized by captions)
CHART_BUILD = (0.5, 2.6, 2.6, 1.0, 1.0)   # s of video: axes, chance rows, measured rows, summary, notes
TITLE_S = 5.0               # title card
LEAD_S, PAUSE_MIN_S = 1.2, 2.0   # sweep: paused lead-in at the window start; shortest end pause
TARGET_S = 45.0
PRE_SENTENCES_FRAMES = 1478      # the clip's length before its captions were rewritten as sentences
MAX_GROWTH = 1.30                # the rewrite may lengthen the clip by at most 30 %

AMBER = C.ACCENT
CAP_Y = 0.075
X0, X1 = 0.105, 0.965       # timeline axes, figure fractions

FOOTER = ("record: records/phase2/phase2_records.npz (v1 cascade grid, parallel formation; recording runs, "
          "no cable is cut)  |  statistic: tether/analysis/v2/paperfig/fig_attribution.py")
POST_HOC = "post hoc: the grid was recorded before the mechanism was derived"

# ------------------------------------------------------------------ narration (<= 2 lines each)
# Complete sentences, each saying what is on screen and what it means.  Terms are defined in words before
# they are used: a slack spell (the red bar), a re-engagement (the blue tick: the cable snaps taut again),
# a slack event and its amber colour (a re-engagement on another cable in the 3 s before), a bounce, the
# chance baseline (the slide) and its mean and 97.5th percentile.  Numbers are formatted from the record.
TITLE_TXT = "Is it chance?"
QUESTION_TXT = "Do slack events follow snaps on other cables more often than chance would produce?"
WATCH_TXT = "Watch for slack events turning amber: another cable snapped taut in the {w:.0f} s before."
TITLE_NOTE = ("The data are recording runs of the v1 cascade grid (parallel formation); "
              "this analysis is post hoc.")
CAP = {
    "A1": "Red bars show slack spells; blue ticks show re-engagements,\nwhen a cable snaps taut.",
    "A2": "A slack event turns amber if another cable re-engaged in the {w:.0f} s before;\n"
          "going slack again within {g:.0f} s counts as a bounce.",
    "B1": "For a chance baseline, all re-engagements slide together in time,\n"
          "keeping the rates but destroying causal timing.",
    "B2": "This shift lowers the mission's amber share from {meas:.0f} % to {shift:.0f} %;\n"
          "its {n} shifts average {mean:.0f} %.",
    "C1": "In all {n} cells with at least {m} slack spells, measured shares\n"
          "exceed the post hoc baseline's 97.5th percentile.",
    "C2": "Measured shares run from {fmin:.1f} times the chance mean in the densest cell,\n"
          "where chance gives {dense:.0f} %, to {fmax:.0f} times.",
    "C3": "The shift also removes weather-driven co-timing, so the excess\n"
          "bounds the cable-to-cable effect rather than isolating it.",
    # end card: the takeaway, as the corrected paper states it (Section III, fig:attribution)
    "END": "In every scored cell, slack events follow other cables' re-engagements\n"
           "more often than the chance baseline.",
}


def words(text: str) -> int:
    return len(text.split())


def read_frames(text: str) -> int:
    """Frames a caption needs on screen: max(4 s, words / 2.5 per s), rounded up to whole frames."""
    return int(math.ceil(round(max(4.0, words(text) / 2.5) * C.FPS, 6)))


def narration(D: dict) -> dict:
    """The captions with the record's numbers filled in (each field is one word, as in the template)."""
    rows = D["rows"]
    return {
        "A1": CAP["A1"],
        "A2": CAP["A2"].format(w=FA.WINDOW, g=FA.BURST_GAP),
        "B1": CAP["B1"],
        "B2": CAP["B2"].format(meas=100 * D["mis_share"], shift=100 * D["mis_share_s"],
                               mean=100 * D["mis_null_mean"], n=FA.N_SHIFTS),
        "C1": CAP["C1"].format(n=len(rows), m=FA.MIN_MARKS),
        "C2": CAP["C2"].format(fmin=D["ratio"].min(), dense=100 * D["nmean"][D["i_dense"]],
                               fmax=D["ratio"].max()),
        "C3": CAP["C3"],
        "END": CAP["END"],
    }


def plan(D: dict) -> dict:
    """Frame budget of every section, sized so each caption meets its reading time.

    Only holds grow: the sweep's paused end pause, the shift's hold, the chart's hold and the final hold.
    The sweep itself (x8), the slide, the re-score and the chart build keep their frames."""
    T = narration(D)
    rf = {k: read_frames(v) for k, v in T.items()}
    lead = int(round(LEAD_S * C.FPS))
    n_sweep = int(np.floor(WIN_LEN * C.FPS / SPEED)) + 1
    a1 = max(rf["A1"], lead)
    pause = max(int(round(PAUSE_MIN_S * C.FPS)), a1 + rf["A2"] - lead - n_sweep)
    motion = sum(int(round(s * C.FPS)) for s in SHIFT_MOTION)
    b1 = max(rf["B1"], motion)
    b_hold = b1 + rf["B2"] - motion
    build = int(round(sum(CHART_BUILD) * C.FPS))
    c1 = max(rf["C1"], build)
    return dict(text=T, lead=lead, n_sweep=n_sweep, a1=a1, pause=pause, motion=motion, b1=b1,
                b_hold=b_hold, b_total=motion + b_hold, c1=c1, c2=rf["C2"], c3=rf["C3"],
                chart=c1 + rf["C2"] + rf["C3"], end=rf["END"])


def fmt_factor(r: float) -> str:
    """Measured / chance mean as shown in the chart: one decimal below 10, whole numbers above."""
    return f"{r:.1f}" if r < 10 else f"{r:.0f}"


# ================================================================== data (asserted before drawing)

def cross_flags(ev_t, ev_c, up_t, up_c):
    """Per event: is there a re-engagement on a different cable in [t - WINDOW, t)?  (as _n_cross)"""
    flags = np.zeros(ev_t.size, bool)
    hits = []
    for i, (t, c) in enumerate(zip(ev_t, ev_c)):
        q = np.flatnonzero((up_t >= t - FA.WINDOW) & (up_t < t) & (up_c != c))
        flags[i] = q.size > 0
        hits.append(q)
    return flags, hits


def shifted(up_t, off, span):
    """The module's circular shift of a re-engagement train."""
    return FA.SPAN_START + np.mod(up_t - FA.SPAN_START + off, span)


def compute(man: C.Manifest) -> dict:
    rec = C.REPO / RECORD_REL
    man.check("fig_attribution reads this record", Path(FA.RECORD).resolve() == rec.resolve(), str(FA.RECORD))
    for p in (RECORD_REL, FA_REL, PAPER_REL, P2_MANIFEST_REL):
        man.source(p)
    committed = json.loads((C.REPO / P2_MANIFEST_REL).read_text())["record_sha256"]
    man.check("record sha256 equals records/phase2/phase2_manifest.json record_sha256",
              man.sources[RECORD_REL] == committed, committed)
    d = np.load(rec, allow_pickle=True)
    t_down, t_up, cable = d["marks_t_down"], d["marks_t_up"], d["marks_cable"]
    seed, cell = d["marks_seed"], d["marks_cell_index"]
    names = [str(x) for x in d["cell_names"]]
    exposure = np.asarray(d["cell_exposure"], float)
    man.check("record holds the 25 grid cells and only the statistics seeds 2003-2022",
              len(names) == N_GRID_CELLS and set(np.unique(seed).tolist()) == set(STAT_SEEDS),
              f"{len(names)} cells, seeds {seed.min()}-{seed.max()}")

    # -- the paper figure's own computation
    rows = FA.attribution_by_cell(rec)
    dropped = [n for i, n in enumerate(names) if int(np.sum(cell == i)) < FA.MIN_MARKS]
    man.check("fig_attribution scores 18 of 25 cells (>= 20 marks); 7 dropped incl. the NULL cell",
              len(rows) == 18 and len(dropped) == 7 and any(n.startswith("NULL") for n in dropped),
              f"scored {len(rows)}, dropped {dropped}")

    # -- the same loop, keeping each mission's offsets and shift counts
    rng = np.random.default_rng(FA.SEED)
    missions, rep = {}, []
    for ci, name in enumerate(names):
        m = cell == ci
        if int(m.sum()) < FA.MIN_MARKS:
            continue
        n_ev = n_cr = 0
        null = np.zeros(FA.N_SHIFTS)
        for s in np.unique(seed[m]):
            k = m & (seed == s)
            td, tu, cb = t_down[k], t_up[k], cable[k]
            head = FA._burst_heads(td, tu, cb)
            ev_t, ev_c = td[head], cb[head]
            ncr = FA._n_cross(ev_t, ev_c, tu, cb)
            n_ev += ev_t.size
            n_cr += ncr
            span = max(td.max(), tu.max()) - FA.SPAN_START
            offs = rng.uniform(0.0, span, FA.N_SHIFTS)
            counts = np.array([FA._n_cross(ev_t, ev_c, shifted(tu, off, span), cb) for off in offs])
            null += counts
            missions[(name, int(s))] = dict(span=span, offsets=offs, counts=counts,
                                            events=int(ev_t.size), cross=int(ncr))
        null /= n_ev
        rep.append(dict(cell=name, marks=int(m.sum()), events=n_ev, measured=n_cr / n_ev,
                        null_mean=float(null.mean()), null_p975=float(np.percentile(null, 97.5))))
    rep.sort(key=lambda r: r["measured"])
    same = len(rep) == len(rows) and all(
        a["cell"] == b["cell"] and a["marks"] == b["marks"] and a["events"] == b["events"]
        and a["measured"] == b["measured"] and a["null_mean"] == b["null_mean"]
        and a["null_p975"] == b["null_p975"] for a, b in zip(rows, rep))
    man.check("re-running the module's shift loop reproduces attribution_by_cell exactly "
              "(so the per-mission offsets used on screen are the module's)", same,
              "events, measured, null mean, null 97.5th pct identical in all 18 cells")

    meas = np.array([r["measured"] for r in rows])
    nmean = np.array([r["null_mean"] for r in rows])
    np975 = np.array([r["null_p975"] for r in rows])
    events = np.array([r["events"] for r in rows])
    ratio = meas / nmean
    man.check("measured share 26-68 % across the 18 cells",
              round(100 * meas.min()) == 26 and round(100 * meas.max()) == 68,
              f"{100 * meas.min():.2f}-{100 * meas.max():.2f} %")
    man.check("measured share above the chance 97.5th percentile in all 18 cells",
              bool(np.all(meas > np975)), f"{int(np.sum(meas > np975))}/18; min margin "
              f"{100 * np.min(meas - np975):.2f} points")
    man.check("measured / chance mean from 1.4x to 21x",
              round(ratio.min(), 1) == 1.4 and round(ratio.max()) == 21,
              f"{ratio.min():.3f}-{ratio.max():.3f}")
    exp_of = {n: exposure[i] for i, n in enumerate(names)}
    i_dense = int(np.argmax(events))
    per_s = np.array([r["events"] / exp_of[r["cell"]] for r in rows])
    man.check("densest cell (most events and most events per recorded second) has the smallest "
              "factor and a chance mean of 49 %",
              i_dense == int(np.argmax(per_s)) == int(np.argmin(ratio)) and round(100 * nmean[i_dense]) == 49,
              f"{rows[i_dense]['cell']}: {events[i_dense]} events, {per_s[i_dense]:.3f}/s, "
              f"chance {100 * nmean[i_dense]:.2f} %, factor {ratio[i_dense]:.3f}")
    i_max = int(np.argmax(ratio))

    paper = " ".join((C.REPO / PAPER_REL).read_text().split())
    phrases = ["between $26$ and $68$ percent of slack events", "This analysis is post hoc",
               "the grid was recorded before the mechanism above was derived",
               "$97.5$th percentile in all $18$ cells", "ranging from $1.4$, in the densest cell",
               "chance alone reaches $49$ percent, to $21$", "bounds the transmission effect rather than isolating it"]
    man.check("the corrected paper (Section III) states these numbers and labels the analysis post hoc",
              all(p in paper for p in phrases), "; ".join(phrases))

    # -- selection
    i_sel = (len(rows) - 1) // 2
    sel_cell = rows[i_sel]["cell"]
    ci = names.index(sel_cell)
    seeds_c = sorted(s for (n, s) in missions if n == sel_cell)
    n_events = [missions[(sel_cell, s)]["events"] for s in seeds_c]
    sel_seed = seeds_c[int(np.argmax(n_events))]          # argmax -> first (lowest seed) on ties
    mis = missions[(sel_cell, sel_seed)]
    man.check("chosen cell's missions all ran their full 600 s (cell_exposure = 20 x 600 s)",
              abs(exp_of[sel_cell] - len(STAT_SEEDS) * SCORED_S) < 1e-6, f"{exp_of[sel_cell]:.3f} s")

    k = (cell == ci) & (seed == sel_seed)
    order = np.argsort(t_down[k], kind="stable")
    td, tu, cb = t_down[k][order], t_up[k][order], cable[k][order]
    head = FA._burst_heads(td, tu, cb)
    ev_idx = np.flatnonzero(head)
    ev_t, ev_c = td[head], cb[head]
    flags, hits = cross_flags(ev_t, ev_c, tu, cb)
    man.check("per-event flags of the chosen mission sum to fig_attribution._n_cross",
              int(flags.sum()) == FA._n_cross(ev_t, ev_c, tu, cb) == mis["cross"]
              and ev_t.size == mis["events"], f"{int(flags.sum())} of {ev_t.size}")
    last = max(td.max(), tu.max())
    span = mis["span"]
    man.check("mission span is [20 s, last mark]", abs(span - (last - FA.SPAN_START)) < 1e-12,
              f"last mark {last:.3f} s")

    starts = np.arange(FA.SPAN_START, np.floor(last - WIN_LEN) + 0.5, 1.0)
    counts_w = np.array([np.sum((ev_t >= s) & (ev_t < s + WIN_LEN)) for s in starts])
    w_lo = float(starts[int(np.argmax(counts_w))])
    w_hi = w_lo + WIN_LEN
    in_w = (ev_t >= w_lo) & (ev_t < w_hi)

    off = float(mis["offsets"][0])
    tu_s = shifted(tu, off, span)
    flags_s, hits_s = cross_flags(ev_t, ev_c, tu_s, cb)
    man.check("the chosen shift's per-event flags sum to _n_cross and to the count the module added",
              int(flags_s.sum()) == FA._n_cross(ev_t, ev_c, tu_s, cb) == int(mis["counts"][0]),
              f"offset {off:.3f} s -> {int(flags_s.sum())} of {ev_t.size}")
    # every one of the mission's 200 shifts, scored per event: whole mission and window
    all_mis, all_win = [], []
    for o in mis["offsets"]:
        f_o, _ = cross_flags(ev_t, ev_c, shifted(tu, o, span), cb)
        all_mis.append(int(f_o.sum()))
        all_win.append(int(f_o[in_w].sum()))
    all_mis, all_win = np.array(all_mis), np.array(all_win)
    man.check("for all 200 offsets of the chosen mission the per-event flags sum to the counts the "
              "module added (so the window's 200-shift mean is over the module's own shifts)",
              bool(np.array_equal(all_mis, mis["counts"])),
              f"window: shown shift {int(all_win[0])} of {int(in_w.sum())}, mean {all_win.mean():.3f}, "
              f"{100 * np.mean(all_win <= all_win[0]):.0f} % of shifts <= the shown one")
    # same shift over the mission's first 90 s of scored time -- disclosed as a caveat
    first = (ev_t >= FA.SPAN_START) & (ev_t < FA.SPAN_START + WIN_LEN)

    D = dict(rows=rows, meas=meas, nmean=nmean, np975=np975, events=events, ratio=ratio,
             i_dense=i_dense, i_max=i_max, i_sel=i_sel, sel_cell=sel_cell, sel_seed=sel_seed,
             n_events_by_seed=dict(zip(seeds_c, n_events)),
             td=td, tu=tu, cb=cb, head=head, ev_idx=ev_idx, ev_t=ev_t, ev_c=ev_c,
             flags=flags, hits=hits, flags_s=flags_s, hits_s=hits_s, tu_s=tu_s,
             span=span, last=last, off=off, offsets=mis["offsets"], counts=mis["counts"],
             w_lo=w_lo, w_hi=w_hi, in_w=in_w, n_w=int(in_w.sum()), c_w=int(flags[in_w].sum()),
             c_w_s=int(flags_s[in_w].sum()), n_first=int(first.sum()), c_first=int(flags[first].sum()),
             c_first_s=int(flags_s[first].sum()), dropped=dropped,
             exposure_sel=exp_of[sel_cell], win_counts=all_win, win_mean=float(all_win.mean()),
             win_le_shown=float(np.mean(all_win <= all_win[0])),
             # absorbed (non-head) slack spells longer than the bounce gap, onset in the window
             long_absorbed=[(int(cb[j]), float(td[j]), float(tu[j] - td[j])) for j in np.flatnonzero(~head)
                            if w_lo <= td[j] < w_hi and tu[j] - td[j] > FA.BURST_GAP])
    D["mis_share"] = flags.mean()
    D["mis_share_s"] = flags_s.mean()
    D["mis_null_mean"] = float(np.mean(mis["counts"]) / ev_t.size)
    D["mis_null_max"] = int(np.max(mis["counts"]))
    return D


def save_cache(D: dict) -> None:
    C.CACHE.mkdir(parents=True, exist_ok=True)
    rows = D["rows"]
    np.savez_compressed(
        CACHE_FILE,
        cell=np.array([r["cell"] for r in rows]), marks=np.array([r["marks"] for r in rows]),
        events=D["events"], measured=D["meas"], null_mean=D["nmean"], null_p975=D["np975"],
        sel_cell=D["sel_cell"], sel_seed=D["sel_seed"], t_down=D["td"], t_up=D["tu"], cable=D["cb"],
        head=D["head"], event_t=D["ev_t"], event_cable=D["ev_c"], cross=D["flags"],
        cross_shifted=D["flags_s"], t_up_shifted=D["tu_s"], offsets=D["offsets"], shift_counts=D["counts"],
        window_shift_counts=D["win_counts"], span=D["span"], window=np.array([D["w_lo"], D["w_hi"]]),
        offset_shown=D["off"])


def register(man: C.Manifest, D: dict) -> None:
    rec = RECORD_REL
    keys = "marks_t_down, marks_t_up, marks_cable, marks_seed, marks_cell_index, cell_names"
    via = f"{rec} [{keys}] via {FA_REL}::attribution_by_cell"
    cache_rel = CACHE_FILE.relative_to(C.REPO)
    # per-mission values: computed from the record with the module's functions (no replay exists)
    mission = (f"{rec} [marks_t_down, marks_t_up, marks_cable, marks_seed, marks_cell_index] via "
               f"{FA_REL}::_burst_heads/_n_cross, cell {D['sel_cell']} seed {D['sel_seed']} "
               f"(cached in {cache_rel})")
    shifts = (f"{mission}; re-engagements circularly shifted within [20 s, last mark] by the offsets "
              f"numpy.random.default_rng({FA.SEED}) draws for this mission in {FA_REL}'s order")
    rows = D["rows"]
    t0, kh, inten = FA._pretty(D["sel_cell"]).split(" / ")
    man.value("chosen cell T0 / k_h / I", [float(t0), float(kh), float(inten)], "N / N m/rad / -",
              f"{rec} cell_names[{D['sel_cell']}]", "lower median of the 18 scored cells by measured share")
    man.value("chosen cell measured share", round(100 * D["meas"][D["i_sel"]], 1), "%", via,
              "derived; post hoc; on screen as 45 % in the whole-mission counter and as the highlighted chart row")
    man.value("chosen mission seed", D["sel_seed"], "", f"{rec} marks_seed", "most events in the chosen cell")
    man.value("timeline window", [D["w_lo"], D["w_hi"]], "s (simulation clock, incl. 20 s warm-up)",
              f"{rec} marks_t_down (event onsets)",
              "derived (selection rule): busiest 90 s of the mission by event onsets; also the clock slot "
              "during the shift ('window t = 395–485 s')")
    man.value("playback speed of the sweep", SPEED, "x", "presentation choice")
    P = D["_plan"]
    n_sw = P["n_sweep"]
    man.value("sweep clock (animated)", [D["w_lo"], round(D["w_lo"] + (n_sw - 1) * SPEED / C.FPS, 3)],
              "s (simulation clock)", f"{rec} clock (marks_t_down / marks_t_up, incl. 20 s warm-up)",
              f"frame i of {n_sw} shows the playhead t = {D['w_lo']:.0f} s + i x {SPEED:.0f}/{C.FPS} s; "
              f"'(paused)' on the {P['lead'] / C.FPS:.1f} s lead-in at {D['w_lo']:.0f} s and the "
              f"{P['pause'] / C.FPS:.1f} s end pause at the last value")
    man.value("window length", WIN_LEN, "s", "presentation choice (selection rule)")
    man.value("baseline percentile shown", 97.5, "%", f"{FA_REL} (np.percentile(null, 97.5))")
    man.value("chart row labels (T0 N / k_h N m/rad / I)", [FA._pretty(r["cell"]) for r in rows], "",
              f"{rec} cell_names via {FA_REL}::_pretty", "rows ordered by measured share")
    man.value("cells not scored (< 20 marks)", D["dropped"], "", via, "on screen as 'the null cell and 6 sparse cells'")
    man.value("warm-up excluded from scoring", FA.SPAN_START, "s", f"{FA_REL} SPAN_START; tether/campaign/phase2.py WARMUP")
    man.value("look-back window", FA.WINDOW, "s", f"{FA_REL} WINDOW")
    man.value("bounce (burst) gap", FA.BURST_GAP, "s", f"{FA_REL} BURST_GAP")
    man.value("window: events / cross-attributed (measured)", [D["n_w"], D["c_w"]], "events", mission,
              "derived: burst heads with onset in the window / those with a re-engagement on a different cable "
              "in [t-3 s, t); the running counter counts up to these (see 'running window counter')")
    in_w = np.flatnonzero(D["in_w"])
    steps = [[round(float(D["ev_t"][e]), 3), k + 1, int(np.sum(D["flags"][in_w[:k + 1]]))]
             for k, e in enumerate(in_w)]
    man.value("running window counter during the sweep (animated): [onset t, n so far, m so far]", steps,
              "s / events / events", mission,
              "derived; 'm of n so far' counts the events with onset in [395 s, playhead] (n) and those "
              "cross-attributed (m); it reads 0 of 0 before the first onset, steps at each listed onset and "
              "ends at 16 of 21; the same per-event flags colour each onset and look-back tail amber (m) or grey")
    man.value("mission: events / cross-attributed (measured)", [int(D["ev_t"].size), int(D["flags"].sum())],
              "events", mission, "derived: fig_attribution._burst_heads / _n_cross on the chosen mission")
    man.value("mission measured share", round(100 * D["mis_share"], 1), "%", mission,
              "derived; post hoc; on screen as 62 %")
    man.value("shift shown (offset)", round(D["off"], 1), "s", shifts,
              "derived; post-hoc baseline: the first of the 200 offsets drawn for this mission")
    ns = int(round(SLIDE_S * C.FPS))
    man.value("shift label during the slide (animated), per frame k = 1..%d" % ns,
              [float(f"{D['off'] * ease(k / ns):.1f}") for k in range(1, ns + 1)], "s", shifts,
              f"derived; post-hoc baseline: the k-th frame of the slide shows 'shift +x s' (one decimal) with "
              f"x = {D['off']:.3f} s x ease(k/{ns}), ease(u) = (1 - cos(pi u))/2 (the list is every label shown, "
              f"in order); each intermediate frame draws a valid circular shift of the train but is not scored; "
              f"only the final offset is scored")
    man.value("shift wraps within", [FA.SPAN_START, round(D["last"], 1)], "s", f"{rec} marks_t_down/marks_t_up",
              "derived: [20 s, the mission's last mark]")
    man.value("window: cross-attributed after the shift", [D["n_w"], D["c_w_s"]], "events", shifts,
              "derived; post-hoc baseline; the shown (first) offset")
    man.value("window: mean cross-attributed count over the mission's 200 shifts", round(D["win_mean"], 3),
              "events (of 21)", shifts,
              f"derived; post-hoc baseline; on screen as 4.1 of 21; {100 * D['win_le_shown']:.0f} % of the 200 "
              f"shifts give {D['c_w_s']} or fewer here")
    man.value("mission: cross-attributed after the shift", [int(D["ev_t"].size), int(D["flags_s"].sum())],
              "events", shifts, "derived; post-hoc baseline; equals the count fig_attribution adds for this offset")
    man.value("mission share after the shift", round(100 * D["mis_share_s"], 1), "%", shifts,
              "derived; post-hoc baseline; on screen as 15 %")
    man.value("mean share over the mission's 200 shifts", round(100 * D["mis_null_mean"], 1), "%", shifts,
              "derived; post-hoc baseline; on screen as 18 % (whole-mission counter and caption B2, "
              "'its 200 shifts average 18 %')")
    man.value("number of shifts per mission", FA.N_SHIFTS, "", f"{FA_REL} N_SHIFTS")
    man.value("rng seed of the baseline", FA.SEED, "", f"{FA_REL} SEED")
    man.value("scored cells / grid cells", [len(rows), N_GRID_CELLS], "cells", via, ">= 20 marks")
    man.value("minimum marks per scored cell", FA.MIN_MARKS, "marks", f"{FA_REL} MIN_MARKS")
    man.value("measured share range", [round(100 * D["meas"].min(), 1), round(100 * D["meas"].max(), 1)], "%",
              via, "on screen as 26-68 %; derived; post hoc")
    man.value("cells above the chance 97.5th percentile", [int(np.sum(D["meas"] > D["np975"])), len(rows)],
              "cells", via, "derived; post-hoc baseline")
    man.value("measured / chance mean range", [round(float(D["ratio"].min()), 2), round(float(D["ratio"].max()), 2)],
              "x", via, "derived; post-hoc baseline; on screen as 1.4x to 21x")
    man.value("densest cell chance mean", round(100 * D["nmean"][D["i_dense"]], 1), "%", via,
              f"derived; post-hoc baseline; {rows[D['i_dense']]['cell']}; on screen as 49 %")
    for r, q in zip(rows, D["ratio"]):
        man.value(f"cell {r['cell']}: events / measured / chance mean / chance 97.5th pct / factor",
                  [r["events"], round(100 * r["measured"], 2), round(100 * r["null_mean"], 2),
                   round(100 * r["null_p975"], 2), fmt_factor(q)], "events / % / % / % / x", via,
                  "row of the chart (dot, square, bar end, events and factor columns); derived; post hoc")
    man.value("seeds per cell", len(STAT_SEEDS), "missions", f"{rec} marks_seed", "2003-2022")
    man.value("mission scored span", [FA.SPAN_START, FA.SPAN_START + SCORED_S], "s",
              f"tether/campaign/phase2.py WARMUP, DURATION; {rec} cell_exposure = 20 x 600 s for the chosen cell",
              "whole-mission counter and strip axis")


# ================================================================== frame sink / furniture

class Sink:
    """Writes frames (or preview stills), counts them, and times every caption."""

    def __init__(self, fig, w=None, stills: dict | None = None):
        self.fig, self.w, self.stills = fig, w, stills or {}
        self.frames = 0
        self.caps: list[dict] = []
        self._art = None
        self._text = None
        self.saved: list[str] = []

    def caption(self, text: str | None) -> None:
        if text == self._text and self._art is not None:
            return
        if self._art is not None:
            try:
                self._art.remove()
            except Exception:
                pass
        self._art = C.caption(self.fig, text, y=CAP_Y) if text else None
        if text != self._text:
            if self.caps and self.caps[-1]["end"] is None:
                self.caps[-1]["end"] = self.frames
            if text:
                self.caps.append({"text": text, "start": self.frames, "end": None})
            self._text = text

    def cleared(self) -> None:
        """Call after fig.clf(): redraw the running caption."""
        self._art = C.caption(self.fig, self._text, y=CAP_Y) if self._text else None

    def _still(self, lo: int, hi: int) -> None:
        for f, name in self.stills.items():
            if lo <= f < hi:
                STILLS.mkdir(parents=True, exist_ok=True)
                p = STILLS / name
                self.fig.savefig(p, dpi=C.DPI)
                self.saved.append(str(p))

    def grab(self) -> None:
        if self.w is not None:
            self.w.grab_frame()
        else:
            self._still(self.frames, self.frames + 1)
        self.frames += 1

    def hold(self, seconds: float) -> None:
        if self.w is not None:
            n = C.hold(self.w, seconds)
        else:
            n = int(round(seconds * C.FPS))
            self._still(self.frames, self.frames + n)
        self.frames += n

    def close(self) -> list[dict]:
        if self.caps and self.caps[-1]["end"] is None:
            self.caps[-1]["end"] = self.frames
        return self.caps


class Clock:
    def __init__(self, fig):
        self.fig, self.art = fig, None

    def set(self, t: float, speed: str) -> None:
        self._drop()
        self.art = C.clock(self.fig, t, speed)

    def label(self, text: str) -> None:
        """For frames that depict no instant (the pooled chart): say so in the clock's place."""
        self._drop()
        self.art = self.fig.text(0.97, 0.955, text, fontsize=C.FS_SMALL, family="DejaVu Sans Mono",
                                 color=C.INK, ha="right", va="top")

    def _drop(self):
        if self.art is not None:
            try:
                self.art.remove()
            except Exception:
                pass
        self.art = None


def furniture(fig, title: str, sub: str) -> None:
    C.title(fig, title, sub)
    fig.text(0.97, 0.905, POST_HOC, ha="right", va="top", fontsize=C.FS_SMALL, color=C.INK,
             bbox=dict(boxstyle="round,pad=0.35", fc="#fff6e5", ec=AMBER, lw=1.2))
    C.footer(fig, FOOTER)


def assert_texts_inside(fig, where: str) -> None:
    """Every visible, non-empty text of the current frame lies inside the 1920x1080 canvas."""
    from matplotlib.text import Text
    fig.canvas.draw()                 # lay out the frame as written (drawing changes no state)
    r = fig.canvas.get_renderer()
    bad = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > C.W + 0.5 or bb.y1 > C.H + 0.5:
            bad.append((t.get_text()[:40], [round(v) for v in (bb.x0, bb.y0, bb.x1, bb.y1)]))
    assert not bad, f"{where}: text outside the frame: {bad}"


def ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


# ================================================================== section 0: title card

def title_card(sink: Sink, fig) -> None:
    """The clip title, the question it answers, what to watch for, and what the data are."""
    fig.clf()
    fig.text(0.5, 0.64, TITLE_TXT, fontsize=C.FS_TITLE * 1.7, weight="bold", color=C.INK,
             ha="center", va="center")
    fig.text(0.5, 0.525, QUESTION_TXT, fontsize=C.FS_SUB * 1.25, color=C.INK, ha="center", va="center")
    fig.text(0.5, 0.435, WATCH_TXT.format(w=FA.WINDOW), fontsize=C.FS_SUB, color=C.INK,
             ha="center", va="center")
    fig.text(0.5, 0.355, TITLE_NOTE, fontsize=C.FS_SMALL, color=C.MUTED, ha="center", va="center")
    C.footer(fig, FOOTER)
    assert_texts_inside(fig, "title card")
    sink.cleared()
    sink.hold(TITLE_S)


# ================================================================== sections A + B: one mission

class Timeline:
    """One mission's marks in a 90 s window, plus the whole-mission strip above it."""

    ROW_BAR = 0.20
    ROW_TICK = 0.36
    TAIL_Y = 0.36

    def __init__(self, fig, D):
        self.fig, self.D = fig, D
        lo, hi = D["w_lo"], D["w_hi"]
        self.lo, self.hi = lo, hi
        t0, kh, inten = FA._pretty(D["sel_cell"]).split(" / ")
        furniture(fig, "Is it chance?  One mission's record",
                  f"cell $T_0$ = {t0} N, $k_h$ = {kh} N m/rad, $I$ = {inten};  seed {D['sel_seed']}")
        # ---------------------------------------------------------------- whole-mission strip
        ov = fig.add_axes([X0, 0.757, X1 - X0, 0.088])     # tall enough for the rotated 'warm-up'
        self.ov = ov
        end = FA.SPAN_START + SCORED_S
        ov.set_xlim(0.0, end)
        ov.set_ylim(-0.55, 1.55)
        ov.set_yticks([1.0, 0.0], ["event onsets", "re-engagements"], fontsize=C.FS_TINY)
        ov.tick_params(axis="y", length=0)
        ov.set_xticks(np.arange(0, 601, 100))
        ov.tick_params(axis="x", labelsize=C.FS_TINY, pad=2)
        for s in ("left",):
            ov.spines[s].set_visible(False)
        ov.axvspan(0.0, FA.SPAN_START, color=C.FAINT, alpha=0.6, lw=0)
        ov.text(FA.SPAN_START / 2, 0.5, "warm-up", rotation=90, ha="center", va="center",
                fontsize=C.FS_TINY, color=C.MUTED)
        ov.add_patch(Rectangle((lo, -0.55), hi - lo, 2.1, fc=AMBER, alpha=0.15, ec=AMBER, lw=1.2))
        ov.add_collection(LineCollection([[(t, 0.62), (t, 1.38)] for t in D["ev_t"]], colors=C.SLACK, lw=1.3))
        self.ov_up = LineCollection(self._ov_segs(D["tu"]), colors=C.TAUT, lw=1.3)
        ov.add_collection(self.ov_up)
        self.ov_head = ov.axvline(lo, color=C.INK, lw=1.4)
        fig.text(X0, 0.852, "whole mission (the shaded box is the window below)", fontsize=C.FS_TINY,
                 color=C.MUTED, ha="left", va="bottom")
        fig.text(X0 - 0.006, 0.752, "simulation time [s]", fontsize=C.FS_TINY, color=C.MUTED, ha="right",
                 va="top")
        # ---------------------------------------------------------------- window
        ax = fig.add_axes([X0, 0.305, X1 - X0, 0.345])
        self.ax = ax
        ax.set_xlim(lo, hi)
        ax.set_ylim(4.55, -0.55)
        ax.set_yticks(range(5), [f"cable {i}" for i in range(5)], fontsize=C.FS_SMALL)
        ax.tick_params(axis="y", length=0)
        ax.set_xticks(np.arange(np.ceil(lo / 10) * 10, hi + 0.1, 10))
        ax.tick_params(axis="x", labelsize=C.FS_TINY)
        ax.set_xlabel("simulation time [s]  (the record's clock includes the 20 s warm-up)",
                      fontsize=C.FS_SMALL, color=C.INK, labelpad=4)
        for i in range(6):
            ax.axhline(i - 0.5, color=C.FAINT, lw=0.8, zorder=0)
        # marks that can appear in the window (bars and ticks), plus the ticks that can attribute
        td, tu, cb, head = D["td"], D["tu"], D["cb"], D["head"]
        self.bars = []
        for j in range(td.size):
            if tu[j] < lo - FA.WINDOW or td[j] > hi:
                continue
            rect = Rectangle((td[j], cb[j] - self.ROW_BAR), 0.0, 2 * self.ROW_BAR,
                             fc=C.SLACK, ec=C.SLACK, lw=0.8, alpha=0.9 if head[j] else 0.32, zorder=2)
            rect.set_visible(False)
            ax.add_patch(rect)
            self.bars.append((j, rect, rect.get_alpha()))
        self.ticks = LineCollection([], lw=2.2, zorder=4)
        ax.add_collection(self.ticks)
        self.tick_amber = np.zeros(tu.size, bool)
        # events
        self.ev = []
        for e, (t, c) in enumerate(zip(D["ev_t"], D["ev_c"])):
            if t < lo - FA.WINDOW or t >= hi:
                continue
            mk, = ax.plot([t], [c], ls="none", marker="o", ms=11, mfc="white", mec=C.SLACK, mew=2.0, zorder=6)
            tail, = ax.plot([t - FA.WINDOW, t], [c + self.TAIL_Y] * 2, color=C.MUTED, lw=2.0,
                            solid_capstyle="butt", zorder=5)
            cap, = ax.plot([t - FA.WINDOW] * 2, [c + self.TAIL_Y - 0.1, c + self.TAIL_Y + 0.1],
                           color=C.MUTED, lw=2.0, zorder=5)
            flash = ax.axvspan(t - FA.WINDOW, t, color=AMBER, alpha=0.0, lw=0, zorder=1)
            for a in (mk, tail, cap):
                a.set_visible(False)
            self.ev.append(dict(e=e, t=t, c=c, mk=mk, tail=tail, cap=cap, flash=flash))
        self.play = ax.axvline(lo, color=C.INK, lw=1.3, zorder=7)
        # ---------------------------------------------------------------- legend + selection
        hand = [Patch(fc=C.SLACK, ec=C.SLACK, alpha=0.9, label="slack spell (chord at or below rest length)"),
                Patch(fc=C.SLACK, ec=C.SLACK, alpha=0.32, label="starts within 2 s of the same cable's re-engagement: absorbed"),
                Line2D([], [], color=C.TAUT, lw=2.5, label="re-engagement (chord back above rest length)"),
                Line2D([], [], ls="none", marker="o", ms=10, mfc="white", mec=C.SLACK, mew=2, label="slack event onset"),
                Line2D([], [], color=AMBER, lw=2.5, marker="o", ms=10, mfc=AMBER, mec=AMBER,
                       label="re-engagement on a different cable in the 3 s before"),
                Line2D([], [], color=C.MUTED, lw=2.0, label="3 s look-back, none found")]
        fig.legend(handles=hand, loc="center", ncol=3, bbox_to_anchor=(0.5, 0.198), fontsize=C.FS_TINY,
                   handlelength=2.2, columnspacing=1.6, labelspacing=0.5)
        fig.text(X0, 0.137,
                 f"Selection, by rule: the cell with the lower-median share of the 18 scored cells; its mission with "
                 f"the most events (seed {D['sel_seed']}: {int(D['ev_t'].size)} events);\nthat mission's busiest "
                 f"90 s (most event onsets). Chosen for clarity, not typicality.",
                 fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="center", linespacing=1.25)
        # counters: the window (running) above, the whole mission below it
        self.cnt1 = fig.text(X0, 0.703, "", fontsize=C.FS_SMALL, color=C.INK, ha="left", va="center")
        self.cnt2 = fig.text(X0, 0.672, "", fontsize=C.FS_SMALL, color=C.INK, ha="left", va="center")
        self.shift_txt = None

    # ---------------------------------------------------------------- helpers
    def _ov_segs(self, up):
        return [[(t, -0.38), (t, 0.38)] for t in up]

    def set_ticks(self, up, visible_to=None, amber=None):
        cb = self.D["cb"]
        segs, cols, lws = [], [], []
        for j, t in enumerate(up):
            if visible_to is not None and t > visible_to:
                continue
            if t < self.lo - FA.WINDOW - 1 or t > self.hi + 1:
                continue
            segs.append([(t, cb[j] - self.ROW_TICK), (t, cb[j] + self.ROW_TICK)])
            a = amber is not None and amber[j]
            cols.append(AMBER if a else C.TAUT)
            lws.append(3.4 if a else 2.2)
        self.ticks.set_segments(segs)
        self.ticks.set_colors(cols if cols else [C.TAUT])
        self.ticks.set_linewidths(lws if lws else [2.2])

    def show_event(self, ev, hit: bool, flash_alpha: float) -> None:
        ev["mk"].set_visible(True)
        ev["tail"].set_visible(True)
        ev["cap"].set_visible(True)
        if hit:
            ev["mk"].set_markerfacecolor(AMBER)
            ev["mk"].set_markeredgecolor(AMBER)
            ev["tail"].set_color(AMBER)
            ev["cap"].set_color(AMBER)
            ev["tail"].set_linewidth(3.0)
            ev["cap"].set_linewidth(3.0)
        else:
            ev["mk"].set_markerfacecolor("white")
            ev["mk"].set_markeredgecolor(C.SLACK)
            ev["tail"].set_color(C.MUTED)
            ev["cap"].set_color(C.MUTED)
            ev["tail"].set_linewidth(2.0)
            ev["cap"].set_linewidth(2.0)
        ev["flash"].set_alpha(flash_alpha)
        ev["flash"].set_color(AMBER if hit else C.MUTED)

    def hide_event(self, ev) -> None:
        for a in ("mk", "tail", "cap"):
            ev[a].set_visible(False)
        ev["flash"].set_alpha(0.0)


def sweep(sink: Sink, clock: Clock, fig, D) -> None:
    """Section A: play the window at x8; classify each onset as the playhead passes it."""
    fig.clf()
    sink.cleared()
    T = Timeline(fig, D)
    td, tu = D["td"], D["tu"]
    flags, hits = D["flags"], D["hits"]
    lo, hi = D["w_lo"], D["w_hi"]
    P = D["_plan"]
    n = P["n_sweep"]
    flash_fr = int(round(0.5 * C.FPS))         # a flash fades over 0.5 s of video
    a1_fr = P["a1"]                            # section frame at which caption A2 replaces A1
    speed = f"×{SPEED:.0f}"

    def cap() -> None:
        sink.caption(P["text"]["A1"] if fr < a1_fr else P["text"]["A2"])
    revealed = {}                              # event -> frame at which the playhead reached it

    def draw(p: float) -> None:
        for j, rect, a0 in T.bars:
            if td[j] <= p:
                rect.set_visible(True)
                rect.set_width(min(p, tu[j]) - td[j])
            else:
                rect.set_visible(False)
        amber = np.zeros(tu.size, bool)
        n_seen = n_hit = 0
        for ev in T.ev:
            if ev["t"] <= p:
                hit = bool(flags[ev["e"]])
                age = fr - revealed.setdefault(ev["e"], fr)
                T.show_event(ev, hit, 0.28 * max(0.0, 1.0 - age / flash_fr))
                if hit:
                    amber[hits[ev["e"]]] = True
                if lo <= ev["t"] < hi:
                    n_seen += 1
                    n_hit += int(hit)
            else:
                T.hide_event(ev)
        T.set_ticks(tu, visible_to=p, amber=amber)
        T.play.set_xdata([p, p])
        T.ov_head.set_xdata([p, p])
        T.cnt1.set_text(f"This window so far: {n_hit} of {n_seen} slack events follow a re-engagement "
                        f"on a different cable within 3 s")

    fr = 0
    for i in range(P["lead"]):
        cap()
        draw(lo)
        clock.set(lo, "paused")
        sink.grab()
        fr += 1
    for i in range(n):
        p = lo + i * SPEED / C.FPS
        cap()
        draw(p)
        clock.set(p, speed)
        sink.grab()
        fr += 1
    p_end = lo + (n - 1) * SPEED / C.FPS
    assert p_end <= hi and p_end >= max(D["ev_t"][D["in_w"]]), "sweep must pass every onset in the window"
    T.cnt2.set_text(f"Whole mission (t = 20-620 s): {int(D['flags'].sum())} of {int(D['ev_t'].size)} "
                    f"events ({100 * D['mis_share']:.0f} %)   |   its cell, all 20 missions pooled: "
                    f"{100 * D['meas'][D['i_sel']]:.0f} %")
    clock.set(p_end, "paused")
    for i in range(P["pause"]):
        cap()
        draw(p_end)
        sink.grab()
        fr += 1
    assert_texts_inside(fig, "sweep, end pause")
    D["_timeline"] = T
    D["_p_end"] = p_end


def shift(sink: Sink, clock: Clock, fig, D) -> None:
    """Section B: slide the whole re-engagement train by the chosen offset and re-score."""
    T: Timeline = D["_timeline"]
    tu, span, off = D["tu"], D["span"], D["off"]
    flags_s, hits_s = D["flags_s"], D["hits_s"]
    P = D["_plan"]
    fade_s, slide_s, rescore_s = SHIFT_MOTION
    fr = 0
    T.play.set_visible(False)
    T.ov_head.set_visible(False)
    T.cnt1.set_text("")
    T.cnt2.set_text("")
    shift_txt = T.fig.text(X1, 0.852, "", ha="right", va="bottom", fontsize=C.FS_SMALL, color=C.INK,
                           weight="bold")
    # these frames show the whole window with a counterfactual train: no instant, no playback
    clock.label(f"window t = {D['w_lo']:.0f}–{D['w_hi']:.0f} s  |  baseline shift: counterfactual, no playback")

    def cap():
        sink.caption(P["text"]["B1"] if fr < P["b1"] else P["text"]["B2"])

    # fade the record's classification; onsets stay
    nf = int(round(fade_s * C.FPS))
    for i in range(nf):
        u = (i + 1) / nf
        for j, rect, a0 in T.bars:
            rect.set_alpha(a0 * (1 - 0.8 * u))
        for ev in T.ev:
            T.show_event(ev, False, 0.0)
            ev["tail"].set_alpha(1 - u)
            ev["cap"].set_alpha(1 - u)
        T.set_ticks(tu)
        cap()
        sink.grab()
        fr += 1
    for ev in T.ev:
        ev["tail"].set_visible(False)
        ev["cap"].set_visible(False)
        ev["tail"].set_alpha(1.0)
        ev["cap"].set_alpha(1.0)
    T.cnt1.set_text("Onsets fixed; the whole re-engagement train slides, wrapping within "
                    f"[20 s, last mark {D['last']:.1f} s]")
    ns = int(round(slide_s * C.FPS))
    for i in range(ns):
        o = off * ease((i + 1) / ns)
        up = shifted(tu, o, span)
        T.set_ticks(up)
        T.ov_up.set_segments(T._ov_segs(up))
        shift_txt.set_text(f"shift +{o:.1f} s")
        cap()
        sink.grab()
        fr += 1
    up = D["tu_s"]
    assert np.allclose(shifted(tu, off * ease(1.0), span), up)
    T.set_ticks(up)
    T.ov_up.set_segments(T._ov_segs(up))
    shift_txt.set_text(f"shift +{off:.1f} s  (the first of the mission's 200 baseline offsets)")
    # re-score with the shifted train
    nr = int(round(rescore_s * C.FPS))
    amber = np.zeros(tu.size, bool)
    for ev in T.ev:
        if flags_s[ev["e"]]:
            amber[hits_s[ev["e"]]] = True
    for i in range(nr):
        a = 0.28 * (1 - i / nr)
        for ev in T.ev:
            T.show_event(ev, bool(flags_s[ev["e"]]), a)
        T.set_ticks(up, amber=amber)
        cap()
        sink.grab()
        fr += 1
    for ev in T.ev:
        T.show_event(ev, bool(flags_s[ev["e"]]), 0.0)
    T.cnt1.set_text(f"Shifted: {D['c_w_s']} of {D['n_w']} events in this window follow a re-engagement on a "
                    f"different cable within 3 s  |  measured {D['c_w']} of {D['n_w']}  |  mean of the "
                    f"200 shifts {D['win_mean']:.1f} of {D['n_w']}")
    T.cnt2.set_text(f"Whole mission (t = 20-620 s): {int(D['flags_s'].sum())} of {int(D['ev_t'].size)} "
                    f"({100 * D['mis_share_s']:.0f} %) shifted  |  measured {int(D['flags'].sum())} of "
                    f"{int(D['ev_t'].size)} ({100 * D['mis_share']:.0f} %)  |  mean of the 200 shifts "
                    f"{100 * D['mis_null_mean']:.0f} %  |  its cell, measured: {100 * D['meas'][D['i_sel']]:.0f} %")
    assert fr == P["motion"], (fr, P["motion"])
    for i in range(P["b_hold"]):
        cap()
        sink.grab()
        fr += 1
    assert_texts_inside(fig, "shift, re-scored hold")


# ================================================================== section C: the 18 cells

def chart(sink: Sink, clock: Clock, fig, D, man: C.Manifest) -> None:
    rows = D["rows"]
    meas, nmean, np975 = 100 * D["meas"], 100 * D["nmean"], 100 * D["np975"]
    ratio, events = D["ratio"], D["events"]
    n = len(rows)
    y = np.arange(n, dtype=float)
    fig.clf()
    sink.cleared()
    furniture(fig, "Is it chance?  All 18 scored cells",
              "measured share vs time-shift chance; 20 missions per cell")
    clock.label("pooled over whole missions  |  no playback")
    ax = fig.add_axes([0.215, 0.255, 0.395, 0.555])
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.tick_params(axis="x", labelsize=C.FS_TINY)
    ax.set_xlabel("slack events preceded within 3 s by a re-engagement\non a different cable [% of events]",
                  fontsize=C.FS_SMALL, labelpad=5)
    ax.set_yticks(y, [FA._pretty(r["cell"]) for r in rows], fontsize=C.FS_SMALL)
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel(r"cell:  $T_0$ [N] / $k_h$ [N m rad$^{-1}$] / $I$ [-]", fontsize=C.FS_SMALL, labelpad=8)
    ax.xaxis.grid(True, color=C.FAINT, lw=0.8)
    ax.set_axisbelow(True)
    for yi in y:
        ax.axhline(yi, color=C.FAINT, lw=0.5, alpha=0.6, zorder=0)
    i_sel, i_dense, i_max = D["i_sel"], D["i_dense"], D["i_max"]
    band = ax.axhspan(i_sel - 0.45, i_sel + 0.45, color=AMBER, alpha=0.13, lw=0, zorder=0)
    band_t = ax.text(79, i_sel, "cell of the mission shown", ha="right", va="center", fontsize=C.FS_TINY,
                     color="#8a5a00")
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.10, n - 0.1, "events", transform=tr, ha="right", va="bottom", fontsize=C.FS_TINY, color=C.MUTED)
    ax.text(1.30, n - 0.1, "÷ chance\nmean", transform=tr, ha="right", va="bottom", fontsize=C.FS_TINY,
            color=C.MUTED, linespacing=1.05)
    chance_art, meas_art, col_art = [], [], []
    for i in range(n):
        h = ax.hlines(y[i], nmean[i], np975[i], color=C.MUTED, lw=2.0, zorder=2)
        cap_, = ax.plot([np975[i]], [y[i]], ls="none", marker="|", ms=13, mew=2.0, color=C.MUTED, zorder=2)
        sq, = ax.plot([nmean[i]], [y[i]], ls="none", marker="s", ms=9, mfc="white", mec=C.MUTED, mew=1.6, zorder=3)
        dot, = ax.plot([meas[i]], [y[i]], ls="none", marker="o", ms=11, mfc=C.TAUT, mec=C.TAUT, zorder=4)
        ev_t = ax.text(1.10, y[i], f"{events[i]:d}", transform=tr, ha="right", va="center",
                       fontsize=C.FS_TINY, color=C.MUTED)
        bold = i in (i_dense, i_max)
        fx = ax.text(1.30, y[i], f"{fmt_factor(ratio[i])}×", transform=tr, ha="right", va="center",
                     fontsize=C.FS_TINY, color=C.INK if bold else C.MUTED, weight="bold" if bold else "normal")
        chance_art.append((h, cap_, sq, ev_t))
        meas_art.append((dot, fx))
        for a in (h, cap_, sq, ev_t, dot, fx):
            a.set_visible(False)
    for a in (band, band_t):
        a.set_visible(False)

    # right panel: legend, key numbers, notes
    xr = 0.745
    h_meas = Line2D([], [], ls="none", marker="o", ms=11, mfc=C.TAUT, mec=C.TAUT)
    h_ch = (Line2D([], [], color=C.MUTED, lw=2.0), Line2D([], [], ls="none", marker="s", ms=9, mfc="white",
                                                         mec=C.MUTED, mew=1.6))
    fig.legend([h_meas, h_ch], ["measured share\n(events; same-cable repeats\nwithin 2 s absorbed)",
                                "time-shift chance (post hoc):\nmean, bar to 97.5th percentile"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)}, loc="upper left",
               bbox_to_anchor=(xr - 0.008, 0.855), fontsize=C.FS_SMALL, handlelength=2.0, labelspacing=0.9)
    k1 = fig.text(xr, 0.64, f"{meas.min():.0f}–{meas.max():.0f} %", fontsize=C.FS_TITLE, weight="bold",
                  color=C.TAUT, ha="left", va="center")
    k1b = fig.text(xr, 0.595, "measured share across the 18 cells", fontsize=C.FS_SMALL, color=C.INK,
                   ha="left", va="center")
    n_above = int(np.sum(D["meas"] > D["np975"]))
    k2 = fig.text(xr, 0.525, f"{n_above} / {n}", fontsize=C.FS_TITLE * 1.25, weight="bold", color=C.INK,
                  ha="left", va="center")
    k2b = fig.text(xr, 0.472, "cells above the baseline's\n97.5th percentile", fontsize=C.FS_SMALL,
                   color=C.INK, ha="left", va="center", linespacing=1.15)
    k3 = fig.text(xr, 0.395, f"{ratio.min():.1f}× to {ratio.max():.0f}× chance mean",
                  fontsize=C.FS_SUB, weight="bold", color=C.INK, ha="left", va="center")
    k3b = fig.text(xr, 0.345, f"{ratio.min():.1f}× in the densest cell, where\nchance alone gives "
                              f"{nmean[i_dense]:.0f} %", fontsize=C.FS_SMALL, color=C.INK, ha="left",
                   va="center", linespacing=1.15)
    note = fig.text(xr, 0.240, "Baseline: each mission's re-engagements\nshifted by one uniform offset, "
                               "wrapping\nwithin [20 s, last mark]; 200 shifts.\nIt also removes co-timing "
                               "from shared\nweather, so the excess bounds the\ntransmission effect rather "
                               "than\nisolating it.", fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="center",
                    linespacing=1.2)
    fig.text(0.215, 0.140, f"Scored: the {n} of {N_GRID_CELLS} grid cells with at least {FA.MIN_MARKS} marks "
                           f"(the null cell and 6 sparse cells are not scored); seeds 2003-2022; rows ordered by "
                           f"measured share.\nEvent rule: same-cable repeats within 2 s of a re-engagement are "
                           f"absorbed, a chained, onset-based variant of the declared rule (paper Sec. III).",
             fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="center", linespacing=1.25)
    keys = [k1, k1b, k2, k2b, k3, k3b]
    for a in keys:
        a.set_visible(False)
    k2box = None

    ax_s, ch_s, me_s, sum_s, ann_s = CHART_BUILD      # the build; the hold after it is sized by captions
    P = D["_plan"]
    f_c2, f_c3 = P["c1"], P["c1"] + P["c2"]            # frames at which captions C2 and C3 start
    for f in range(P["chart"]):
        t = f / C.FPS
        sink.caption(P["text"]["C1"] if f < f_c2 else (P["text"]["C2"] if f < f_c3 else P["text"]["C3"]))
        tc = t - ax_s
        for i in range(n):
            vis = tc >= ch_s * i / n
            for a in chance_art[i]:
                a.set_visible(vis)
        tm = t - ax_s - ch_s
        for i in range(n):
            vis = tm >= me_s * i / n
            for a in meas_art[i]:
                a.set_visible(vis)
        ts = t - ax_s - ch_s - me_s
        if ts >= 0:
            for a in (band, band_t, k1, k1b):
                a.set_visible(True)
        if ts >= 0.5:
            for a in (k2, k2b):
                a.set_visible(True)
        if ts >= sum_s:
            for a in (k3, k3b):
                a.set_visible(True)
        sink.grab()
    assert_texts_inside(fig, "chart, built")
    D["_chart_keys"] = (k2, k2b)


def final_hold(sink: Sink, fig, D) -> None:
    """End card: the key number boxed, and the clip's takeaway as one sentence."""
    k2, k2b = D["_chart_keys"]
    fig.add_artist(Rectangle((0.737, 0.44), 0.225, 0.125, transform=fig.transFigure, fill=False,
                             ec=C.TAUT, lw=2.0))
    P = D["_plan"]
    sink.caption(P["text"]["END"])
    assert_texts_inside(fig, "end card")
    sink.hold(P["end"] / C.FPS)


# ================================================================== main

def check_captions(caps: list[dict]) -> list[dict]:
    out = []
    for c in caps:
        dur = (c["end"] - c["start"]) / C.FPS
        need = max(4.0, words(c["text"]) / 2.5)
        lines = c["text"].split("\n")
        assert len(lines) <= 2, f"caption has more than 2 lines: {c['text']!r}"
        assert all(len(l) <= 100 for l in lines), f"caption line too long: {c['text']!r}"
        assert dur + 1e-9 >= need, f"caption on screen {dur:.2f} s < {need:.2f} s: {c['text']!r}"
        out.append({"t": round(c["start"] / C.FPS, 3), "text": c["text"].replace("\n", " "),
                    "duration_s": round(dur, 3), "words": words(c["text"])})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="write a few stills to the development directory ($TETHER_DEV_DIR), no video")
    args = ap.parse_args(argv)
    C.ensure_dirs()
    man = C.Manifest(
        name=NAME, title="Is it chance? Cascades against a time-shift baseline",
        story="")
    D = compute(man)
    for c in man.checks:
        print("PASS", c["check"], "|", c["detail"])
    ne = int(D["ev_t"].size)
    man.story = (
        f"Is the cross-cable cascade more than coincidence? One grid mission's record shows the statistic event "
        f"by event: in its busiest 90 s, {D['c_w']} of {D['n_w']} slack events follow a re-engagement on a "
        f"different cable within 3 s (same-cable repeats within 2 s absorbed); over the whole mission "
        f"{int(D['flags'].sum())} of {ne}. Sliding the mission's whole re-engagement train in time -- which keeps "
        f"both rates but destroys causal timing -- leaves {D['c_w_s']} of {D['n_w']} in the window and "
        f"{int(D['flags_s'].sum())} of {ne} in the mission for the shift shown (the mission's 200 shifts average "
        f"{100 * D['mis_null_mean']:.0f} %). Across all 18 scored cells of the cascade grid the measured share "
        f"({100 * D['meas'].min():.0f}-{100 * D['meas'].max():.0f} %) exceeds the baseline's 97.5th percentile "
        f"in every cell, by {D['ratio'].min():.1f}x in the densest cell (where chance alone gives "
        f"{100 * D['nmean'][D['i_dense']]:.0f} %) up to {D['ratio'].max():.0f}x. The analysis is post hoc, and "
        f"because the shift also removes co-timing from shared weather the excess bounds the transmission effect "
        f"rather than isolating it.")
    save_cache(D)
    D["_plan"] = P = plan(D)
    register(man, D)
    man.selection = (
        f"Cell: the lower median of the 18 scored cells ranked by measured share (9th of 18 ascending): "
        f"{D['sel_cell']} ({100 * D['meas'][D['i_sel']]:.1f} %). Mission: that cell's seed with the most events "
        f"(ties -> lowest seed): {D['sel_seed']} ({int(D['ev_t'].size)} events; per seed "
        f"{D['n_events_by_seed']}). Window: the 90 s window [s, s+90) with s on whole seconds in "
        f"[20 s, last mark - 90 s] holding the most event onsets (ties -> earliest): "
        f"{D['w_lo']:.0f}-{D['w_hi']:.0f} s. Shift: the first of the 200 offsets that "
        f"numpy.random.default_rng(20260914) draws for this mission in fig_attribution's order: "
        f"{D['off']:.3f} s. Chosen for clarity (density), not typicality. The chart shows all 18 scored cells.")
    man.caveats = [
        "Post hoc: the attribution statistic and its time-shift baseline were not pre-registered; the grid "
        "(v1 Phase 2) was recorded before the transmission mechanism was derived.",
        "The shift also destroys any co-timing produced by shared weather, so the excess over the baseline "
        "bounds the transmission effect rather than isolating it (paper Section III).",
        "The event rule (a mark starting within 2 s of the same cable's previous re-engagement is absorbed) is a "
        "chained, onset-based variant of the campaign's declared anchored rule; the anchored-rule share was not "
        "recomputed (paper Section III; closeout, fifth round). It absorbs long same-cable slack spells too, not "
        "only short bounces: in the window shown the absorbed spells longer than 2 s are "
        + "; ".join(f"cable {c} from {t:.1f} s ({d:.1f} s)" for c, t, d in D["long_absorbed"])
        + ". The timeline legend states the rule literally; the chart legend says 'same-cable repeats within "
        "2 s absorbed' and its footnote says the rule is the chained variant.",
        f"Selection for clarity, not typicality: the chosen mission's share ({100 * D['mis_share']:.1f} %) exceeds "
        f"its cell's ({100 * D['meas'][D['i_sel']]:.1f} %, on screen in the whole-mission counter), and the "
        f"window's ({D['c_w']}/{D['n_w']}) exceeds the mission's. Window counts fluctuate: over the mission's "
        f"first 90 s of scored time (20-110 s) the same shift gives {D['c_first_s']} of {D['n_first']} events, "
        f"while the record gives {D['c_first']} of {D['n_first']} there; only the pooled cell shares carry the "
        f"claim.",
        f"One shift is shown (the first drawn), and it sits low for the window: {D['c_w_s']} of {D['n_w']} "
        f"against a mean of {D['win_mean']:.2f} of {D['n_w']} over the mission's 200 shifts "
        f"({100 * D['win_le_shown']:.0f} % of shifts give {D['c_w_s']} or fewer; the mean is on screen). At "
        f"mission level it is typical: {int(D['flags_s'].sum())} of {int(D['ev_t'].size)} against a mean of "
        f"{100 * D['mis_null_mean']:.1f} % ({np.mean(D['counts']):.2f} of {int(D['ev_t'].size)}), with a maximum "
        f"of {D['mis_null_max']} of {int(D['ev_t'].size)} (offsets near 0 or the span nearly reproduce the "
        f"record); the chart's baseline pools 20 missions per cell.",
        "No fleet motion is drawn: records/phase2/phase2_records.npz holds slack marks, not plant state, and "
        "nothing was re-simulated. Recording runs: no cable is cut. The per-mission values are computed from "
        "the record with fig_attribution's functions (their sources name the record keys) and only cached in "
        "Presentation/cache/cascade_statistics_attribution.npz; that file is not a physics replay.",
        f"During the baseline shift (the {P['b_total'] / C.FPS:.1f} s after the sweep's end pause) the frames show the whole "
        f"{D['w_lo']:.0f}-{D['w_hi']:.0f} s window with a counterfactual (shifted) re-engagement train, not an "
        f"instant: the clock slot reads 'window t = {D['w_lo']:.0f}–{D['w_hi']:.0f} s | baseline shift: "
        f"counterfactual, no playback' instead of a running clock. The 'shift +x s' label counts up over the "
        f"slide; its intermediate values are unscored circular shifts (registered as a rule).",
        "The chart frames depict no single instant; the clock slot says 'pooled over whole missions | no playback' "
        "instead of a simulation clock.",
        "The 0.44 transmission gain is a derivation and is neither used nor shown here.",
    ]

    fig = C.new_frame()
    assert plt.rcParams["savefig.bbox"] in (None, "standard"), "frames must keep the full 1920x1080 canvas"
    stills = {}
    if args.preview:
        stills = {int(t * C.FPS): f"preview_{t:05.1f}s.png" for t in
                  (2.5, 7.5, 12.0, 18.5, 21.0, 24.5, 30.0, 36.0, 41.0, 46.0, 54.0, 60.0)}
        sink = Sink(fig, None, stills)
        run(sink, fig, D, man)
    else:
        w = C.writer()
        with w.saving(fig, str(CLIP), dpi=C.DPI):
            sink = Sink(fig, w)
            run(sink, fig, D, man)
    caps = check_captions(sink.close())
    man.frames = sink.frames
    dur = sink.frames / C.FPS
    print(f"frames {sink.frames} ({dur:.2f} s); captions:")
    for c in caps:
        print(f"  {c['t']:6.2f} s  {c['duration_s']:5.2f} s  {c['words']:2d} w  {c['text']}")
    expect = (int(round(TITLE_S * C.FPS)) + P["lead"] + P["n_sweep"] + P["pause"] + P["b_total"]
              + P["chart"] + P["end"])
    assert sink.frames == expect, f"frames {sink.frames} != planned {expect}"
    assert 0.75 * TARGET_S <= dur and sink.frames <= MAX_GROWTH * PRE_SENTENCES_FRAMES, \
        f"duration {dur:.2f} s: below 75 % of {TARGET_S} s or above {MAX_GROWTH:.2f} x {PRE_SENTENCES_FRAMES} frames"
    if args.preview:
        print("stills:", *sink.saved, sep="\n  ")
        return
    out = man.write(CLIP)
    info = json.loads(out.read_text())
    info["captions"] = caps
    info["target_duration_s"] = TARGET_S
    out.write_text(json.dumps(info, indent=1, default=float))


def run(sink: Sink, fig, D, man) -> None:
    clock = Clock(fig)
    title_card(sink, fig)
    sweep(sink, clock, fig, D)
    shift(sink, clock, fig, D)
    chart(sink, clock, fig, D, man)
    final_hold(sink, fig, D)


if __name__ == "__main__":
    main()
