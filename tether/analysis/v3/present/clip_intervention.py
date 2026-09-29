"""``intervention`` -- the counterfactual: the same recorded moment with and without the snap
(plan v3 WP4; Presentation/STORYBOARD.md rules; the complete-sentences rule for on-screen text).

What the clip shows
-------------------
One parent re-engagement of the calibration cell (parallel formation, T0 = 600 N, intensity 0.5),
chosen by the declared rule below, replayed twice from the same recorded plant state under the same
recorded weather and headings with the campaign's own intervention integrator
(``tether/campaign/v3/intervention.py``):

  * left, FACTUAL: every cable active -- this branch must reproduce the record (test T2.3);
  * right, COUNTERFACTUAL: the snapping cable applies no force from its re-engagement on, so the
    payload never receives the snap ("snap removed").

  (a) a title card; (b) slow motion x1/10 over t_up - 0.2 s .. t_up + 0.8 s, two top-down fleet
  panels side by side with the five tensions of each branch under them and every other-cable slack
  onset flagged as it happens; (c) real time to t_up + 3 s (the declared attribution window);
  (d) a hold on the tally: onsets on the other cables within 3 s in each branch, and their
  difference, which is the record's causal offspring count for this parent;
  (e) if ``records/v3/wp3_results.json`` exists, a second segment sweeps the criticality map:
  the spectral radius rho of the kernel along the T0 = 0.6 kN intensity sweep with its seed
  intervals against the rho = 1 level, ending on the sweep's declared branch.

Records read (path -> keys)
---------------------------
* ``records/v3/wp2_intervention.npz`` -- ``cell_names``, ``cell_index``, ``seed``, ``event_id``,
  ``cable``, ``t_up``, ``x``, ``T_peak``, ``v_return``, ``t_cf``, ``v_cf``, ``T_peak_cf``,
  ``recorded_onsets``, ``factual_onsets``, ``counterfactual_onsets``, ``causal``,
  ``onset_set_reproduced`` (the campaign's intervention rows, computed in the stage-A workers).
* ``records/v3/wp2_results.json`` -- ``tests.T2.3`` (the validation verdict and its thresholds),
  ``cells[cell].intervention``; ``records/v3/cascade_declarations.json`` -- the declared
  intervention spec, window and validation thresholds.
* ``records/v3/cache/v3_stageA.pkl`` -- the run's cached marks (``t_up``, ``T_peak``), which the
  re-simulation must reproduce bit-exactly before a frame is drawn.
* ``records/v3/wp3_results.json`` (segment (e) only) -- ``cells[name].{rho, rho_ci95, operable,
  n_events}``, ``tests.T3.4.{branch, critical_intensity?, margin?}``, ``kernel_path``.

Selection rule (declared in the plan, WP4)
------------------------------------------
Among the calibration cell's sampled parents in ``wp2_intervention.npz`` the one with the largest
causal offspring count (sum over the other cables); ties broken by the larger T_peak.

What is asserted before any frame is drawn (``Manifest.check``)
----------------------------------------------------------------
* the re-simulated run reproduces the cached marks bit-exactly (``t_up`` and ``T_peak``);
* the two branches re-integrated here with ``intervention.integrate`` reproduce the record's row
  for this parent exactly (onset counts per cable in both branches, t_cf, v_cf, T_peak_cf, causal
  counts), and the state-capturing copy of the integrator used for drawing returns the same
  elongations, onsets and parent facts bit-exactly (``_integrate_with_state`` is a verbatim copy of
  ``intervention.integrate`` that additionally records poses; the equality is asserted, so the
  drawn branch is the declared integrator's);
* the factual branch's replay residuals lie within the declared T2.3 tolerances for this parent
  (|t_cf - t_up| <= 5 ms, |v_cf - v_return| <= 0.03 m/s, T_peak within 2 %) and its other-cable onset
  set matches the record (the row's ``onset_set_reproduced``);
* the counterfactual removes exactly the recorded causal count: factual - counterfactual onset
  counts per other cable equal the row's ``causal`` vector;
* both branches coincide before t_up (max pose difference < 1e-9 m) and the counterfactual's
  parent cable carries no tension after t_up;
* every caption stays on screen >= 4 s and >= words / 2.5 s, renders in at most two lines inside
  the frame and clear of every other artist; the tally banner covers no body or cable.

Written: ``Presentation/cache/intervention_window.npz`` (the replayed window: record slices and both
branches, so a re-render needs no re-simulation), ``Presentation/clips/intervention.mp4``,
``Presentation/manifests/intervention.json``.

Run: ``python3 -m tether.analysis.v3.present.clip_intervention`` (``--check`` verifies and stops;
``--stills`` writes inspection PNGs to the development directory ($TETHER_DEV_DIR); ``--dry-run CELL SEED EVENT_ID`` builds from a
reduced run pickle in the development directory ($TETHER_DEV_DIR) instead of the campaign records, writing everything to the
development directory -- development only, never the presentation).
"""

from __future__ import annotations

import tempfile
import os
import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tether.analysis.v2.present import common as C
from tether.campaign.v3 import campaign, cells, compute
from tether.campaign.v3 import intervention as IV
from tether.physics import constants

NAME = "intervention"
TITLE = "The same moment, with and without the snap"
WINDOW = 3.0                     # s, the declared attribution window (InterventionSpec.window)
SLOW = (-200, 800)               # ms from t_up: slow motion
SLOW_FACTOR = 10
REAL = (800, 3000)               # ms from t_up: real time
STATE_MS = 10                    # the integrator's decimated output grid
CLIP = C.CLIPS / f"{NAME}.mp4"
CACHE = C.CACHE / "intervention_window.npz"
STILLS = Path(os.environ.get("TETHER_DEV_DIR", Path(tempfile.gettempdir()) / "tether_dev")) / "present" / "intervention"
CAB = {0: "#1b9e77", 1: "#6b7280", 2: "#d98c00", 3: "#7b3294", 4: "#8c510a"}   # identity colours (as clip_snap_cascade)

FLEET_L = [0.020, 0.430, 0.470, 0.440]
FLEET_R = [0.510, 0.430, 0.470, 0.440]
FLEET_L_TALLY = [0.020, 0.430, 0.470, 0.355]     # the tally card: shorter panels, the counts in the strip above them
FLEET_R_TALLY = [0.510, 0.430, 0.470, 0.355]
TALLY_Y = 0.845
TRACE_L = [0.060, 0.265, 0.420, 0.125]
TRACE_R = [0.550, 0.265, 0.420, 0.125]
KEY_Y = 0.188                    # the cable key, below the traces' x labels and above the caption band
CAPTION_Y = 0.124                # the caption band clears the three footer lines below it and the key above it
ONE_LINE_MAX = 96

SUB = "Cell {cell} · seed {seed} · cable {cable} · T_peak = {peak:.1f} kN = {x:.1f} T0"
CAP_TITLE = ("The next movie replays one recorded snap twice from the same state: "
             "once as recorded, and once with the snapping cable's force removed.")
CAP_SLOW = ("The movie runs 10 times slower. On the left cable {cable} re-engages at t_up = {t_up:.3f} s; "
            "on the right the same cable applies no force from that instant on.")
CAP_PEAK = ("On the left cable {cable} peaks at {peak:.1f} kN, {x:.1f} times the pretension T0; "
            "on the right the payload never receives that pull.")
CAP_ONSET = "A dashed cable carries no tension; each flag marks the moment another cable goes slack."
CAP_REAL = "The movie now runs in real time to t_up + 3 s, the window in which the campaign attributes slack onsets to a snap."
CAP_TALLY = ("Within 3 s, {n_f} onsets on the other cables followed the recorded snap and {n_c} occurred without it; "
             "the difference, {n_causal}, is the number of onsets this snap caused in the replay.")
VALIDATION = {"text": ""}         # the campaign's T2.3 facts for the footer, filled from the record in main
CAP_RHO = ("Counting offspring across many snaps gives a branching matrix whose spectral radius ρ says "
           "whether cascades die out (below one) or grow.")
# the matrix is NOT built from the interventional counts shown above: it is the empirical fallback, fitted on each
# cell's two fitting runs by the observational attribution rule, and the next caption says so on screen
CAP_RHO_FIT = ("Here each ρ is fitted on one cell's two fitting runs, not on the replay above, so the zigzag "
               "across intensity is a two-run artefact rather than a trend.")
CAP_RHO_END = "{sentence}"
FOOTER = ("Plant state re-simulated from the declared run (its marks equal records/v3/cache/v3_stageA.pkl); both branches by "
          "tether/campaign/v3/intervention.py, the campaign's integrator; recording run, no cable is cut")
FOOTER_2 = ("Bodies from the integrator's 10 ms poses, linear in between; tensions from the plant's law on the same grid; "
            "times include the 20 s warm-up; no contact is modelled, so lines and hulls may overlap")


# ----------------------------------------------------------------------------- selection and replay


def _rows_for_cell(npz: dict, cell: str) -> np.ndarray:
    names = list(npz["cell_names"])
    if cell not in names:
        raise SystemExit(f"{cell} has no intervention rows in wp2_intervention.npz")
    return np.flatnonzero(npz["cell_index"] == names.index(cell))


def select_parent(npz: dict, cell: str) -> int:
    """The declared rule: largest causal count, ties by the larger T_peak."""
    idx = _rows_for_cell(npz, cell)
    causal = npz["causal"][idx].sum(axis=1)
    order = sorted(idx, key=lambda k: (int(npz["causal"][k].sum()), float(npz["T_peak"][k])), reverse=True)
    assert int(npz["causal"][order[0]].sum()) == int(causal.max())
    return int(order[0])


from tether.campaign.v3.integrate_state import integrate_with_state as _integrate_with_state, same_branch as _same_branch, tension as _tension  # noqa: E402


def replay(man: C.Manifest, cell: str, seed: int, event_id: int, row: dict, spec: IV.InterventionSpec, expected_marks: dict | None) -> dict:
    """Re-simulate the run, re-integrate both branches, assert every reproduction, and return the drawn window."""
    from tether.campaign import summaries as v1
    from tether.campaign.fleet_run import build_run, run_to_end
    from tether.campaign.v2 import events as ev
    from tether.campaign.v2 import summaries as v2
    from tether.campaign.v3 import reducer

    job = next(j for j in cells.all_jobs("A", expected=False) if j.cell == cell and j.seed == seed)
    print(f"re-simulating {cell} seed {seed} ...", flush=True)
    run = run_to_end(build_run(job.spec, job.seed))
    summary = v1.summarize_run(run, job.spec.warmup, job.spec.pretension)
    marks = summary["marks"]
    if expected_marks is not None:
        same = np.array_equal(np.asarray(marks["t_up"]), expected_marks["t_up"]) and np.array_equal(np.asarray(marks["T_peak"]), expected_marks["T_peak"])
        man.check("the re-simulated run reproduces the cached marks bit-exactly (t_up, T_peak)", same,
                  f"{np.asarray(marks['t_up']).size} marks")
    record = v2.run_record(run)
    end = float(record["sim_end"]) if record["closure"] is None else min(float(record["sim_end"]), float(record["closure"][1]))
    t_up_all = np.asarray(marks["t_up"], dtype=float)
    cable_all = np.asarray(marks["cable"], dtype=np.int64)
    declustering = ev.decluster_marks(t_up_all, cable_all, rule="anchored")
    _, downs = ev.crossings(record["event_time"], record["elongation"], np.asarray(record["alive"], dtype=bool))
    table = reducer.offspring_table(record, marks, declustering, job.spec.warmup, end, downs=downs)
    events = table["events"]
    thrusts = np.asarray(run.fleet.operating.thrusts, dtype=float)
    x = IV.parent_window(record, events, event_id, thrusts, spec, job.spec.pretension)
    assert x is not None, "the parent's window does not fit the record"
    man.check("the parent is the record's: cable, t_up, T_peak equal the intervention row",
              x.cable == int(row["cable"]) and abs(x.t_up - float(row["t_up"])) < 1e-9 and abs(x.T_peak - float(row["T_peak"])) < 1e-6,
              f"cable {x.cable}, t_up {x.t_up:.6f} s, T_peak {x.T_peak:.1f} N")
    factual = IV.integrate(x, None, "recorded", window=spec.window, peak_window=spec.peak_window)
    counter = IV.integrate(x, x.cable, "recorded", window=spec.window, peak_window=spec.peak_window)
    f_state = _integrate_with_state(x, None, spec.window, spec.peak_window)
    c_state = _integrate_with_state(x, x.cable, spec.window, spec.peak_window)
    man.check("the state-capturing copy of the integrator returns the declared integrator's outputs bit-exactly (both branches)",
              _same_branch(factual, f_state) and _same_branch(counter, c_state))
    recorded = IV.recorded_onsets(record, downs, x.t_up, spec.window, x.cable)
    causal = IV.causal_offspring(factual, counter, x.cable, spec.window)
    man.check("the re-integrated branches reproduce the campaign's intervention row for this parent (onset counts, t_cf, v_cf, T_peak_cf, causal)",
              [int(v.size) for v in factual["onsets"]] == [int(v) for v in row["factual_onsets"]]
              and [int(v.size) for v in counter["onsets"]] == [int(v) for v in row["counterfactual_onsets"]]
              and np.array_equal(causal, np.asarray(row["causal"], dtype=np.int64))
              and abs(factual["parent"]["t_cf"] - float(row["t_cf"])) < 1e-9 and abs(factual["parent"]["v_cf"] - float(row["v_cf"])) < 1e-9
              and abs(factual["parent"]["T_peak_cf"] - float(row["T_peak_cf"])) < 1e-6,
              f"factual {[int(v.size) for v in factual['onsets']]}, counterfactual {[int(v.size) for v in counter['onsets']]}, causal {causal.tolist()}")
    p = factual["parent"]
    man.check("the factual branch's replay residuals lie within the declared T2.3 tolerances for this parent",
              p["outcome"] == "reengaged" and abs(p["t_cf"] - x.t_up) <= 0.005 and abs(p["v_cf"] - x.v_return) <= 0.03
              and abs(p["T_peak_cf"] / x.T_peak - 1.0) <= 0.02,
              f"dt {1e3 * (p['t_cf'] - x.t_up):+.3f} ms, dv {p['v_cf'] - x.v_return:+.4f} m/s, peak {100 * (p['T_peak_cf'] / x.T_peak - 1):+.2f} %")
    reproduced = all(IV._match(recorded[c], factual["onsets"][c], 0.02) for c in range(len(recorded)) if c != x.cable)
    man.check("the factual branch reproduces the record's other-cable onset set (same cables, times within 20 ms)",
              reproduced and bool(row["onset_set_reproduced"]),
              f"recorded {[int(v.size) for v in recorded]}")
    man.check("the counterfactual removes exactly the recorded causal count (factual - counterfactual onsets per other cable)",
              np.array_equal(np.maximum(0, np.array([len(a) for a in factual["onsets"]]) - np.array([len(b) for b in counter["onsets"]])) * (np.arange(5) != x.cable), causal))
    before = f_state["t_10ms"] < x.t_up
    man.check("both branches coincide before t_up (max pose difference < 1e-9 m)",
              float(np.abs(f_state["poses"][before] - c_state["poses"][before]).max()) < 1e-9)
    k, c_ = x.stiffness, x.damping
    T_f = _tension(f_state["e_10ms"], f_state["rates"], f_state["alive"], k, c_)
    T_c = _tension(c_state["e_10ms"], c_state["rates"], c_state["alive"], k, c_)
    after = c_state["t_10ms"] >= x.t_up
    man.check("the counterfactual's parent cable carries no tension after t_up", float(T_c[after, x.cable].max()) == 0.0)
    peak_k = int(np.argmax(T_f[:, x.cable]))
    return {
        "cell": cell, "seed": seed, "event_id": event_id, "cable": x.cable, "t_up": x.t_up, "T_peak": x.T_peak, "x": x.x,
        "v_return": x.v_return, "pretension": job.spec.pretension, "stiffness": k, "damping": c_,
        "t": f_state["t_10ms"], "poses_f": f_state["poses"], "poses_c": c_state["poses"], "T_f": T_f, "T_c": T_c,
        "alive_c": c_state["alive"], "onsets_f": factual["onsets"], "onsets_c": counter["onsets"], "causal": causal,
        "load_offsets": x.load_offsets, "vessel_offsets": x.vessel_offsets,
        "peak_t": float(f_state["t_10ms"][peak_k]), "peak_T": float(T_f[peak_k, x.cable]),
        "t_cf": p["t_cf"], "v_cf": p["v_cf"], "T_peak_cf": p["T_peak_cf"],
        "n_marks": int(t_up_all.size),
    }


# ----------------------------------------------------------------------------- the replay cache


def save_cache(cache: Path, d: dict, man: C.Manifest) -> None:
    """The replayed window plus the checks that passed at creation and the stage cache's hash, so a re-render
    needs no re-simulation and the manifest can say which checks are inherited from the cached replay."""
    stage_sha = C.sha256(compute.STAGE_CACHE["A"]) if compute.STAGE_CACHE["A"].exists() else ""
    np.savez_compressed(cache, **{k_: v for k_, v in d.items() if isinstance(v, (np.ndarray, float, int, str))},
                        onsets_f=np.array(d["onsets_f"], dtype=object), onsets_c=np.array(d["onsets_c"], dtype=object),
                        checks_at_creation=json.dumps(man.checks), stage_cache_sha256=stage_sha)


def load_cache(man: C.Manifest, cache: Path, cell: str, seed: int, event_id: int, row: dict) -> dict:
    """Re-render from the cached replay: re-assert what the cached arrays allow, inherit the rest by hash."""
    z = np.load(cache, allow_pickle=True)
    d = {k_: (z[k_].item() if z[k_].ndim == 0 else z[k_]) for k_ in z.files if k_ not in ("checks_at_creation", "stage_cache_sha256")}
    d["onsets_f"], d["onsets_c"] = list(z["onsets_f"]), list(z["onsets_c"])
    man.check("the cached replay is this parent's (cell, seed, event, cable, t_up)",
              d["cell"] == cell and int(d["seed"]) == seed and int(d["event_id"]) == event_id and int(d["cable"]) == int(row["cable"])
              and abs(float(d["t_up"]) - float(row["t_up"])) < 1e-9, str(cache))
    stored = str(z["stage_cache_sha256"]) if "stage_cache_sha256" in z.files else ""
    if stored:
        man.check("the cached replay was made against the current stage-A cache (sha256 equal)",
                  stored == C.sha256(compute.STAGE_CACHE["A"]))
    else:
        man.caveats.append("DRY RUN: the cached replay predates the stage-A cache; no hash check possible.")
    inherited = json.loads(str(z["checks_at_creation"])) if "checks_at_creation" in z.files else []
    for chk in inherited:
        man.checks.append({"check": chk["check"] + " [asserted when the cached replay was made]", "passed": True, "detail": chk.get("detail", "")})
    man.check("the cached branches reproduce the campaign's intervention row (onset counts, causal)",
              [int(v.size) for v in d["onsets_f"]] == [int(v) for v in row["factual_onsets"]]
              and [int(v.size) for v in d["onsets_c"]] == [int(v) for v in row["counterfactual_onsets"]]
              and np.array_equal(np.asarray(d["causal"]), np.asarray(row["causal"], dtype=np.int64)))
    return d


# ----------------------------------------------------------------------------- frames


@dataclass
class Spec:
    kind: str                    # title | run | tally | rho | rho_end
    t_ms: int | None = None      # ms from t_up (run frames)
    speed: str | None = None
    caption: str = ""
    step: int = 0                # rho: number of sweep points revealed


def secs(s: float) -> int:
    return int(round(s * C.FPS))


def two_lines(text: str) -> str:
    if len(text) <= ONE_LINE_MAX:
        return text
    w = text.split(" ")
    best = None
    for i in range(1, len(w)):
        a, b = " ".join(w[:i]), " ".join(w[i:])
        score = abs(len(a) - len(b)) - 40 * w[i - 1].endswith((".", ";", ":")) - 12 * w[i - 1].endswith(",")
        if w[i].strip(".,")[:1].isdigit() or w[i - 1][:1].isdigit():
            score += 30
        if best is None or score < best[0]:
            best = (score, a + "\n" + b)
    return best[1]


def timeline(d: dict, rho: dict | None) -> list[Spec]:
    cap = dict(cable=d["cable"], t_up=d["t_up"], peak=d["T_peak"] / 1e3, x=d["x"])
    fr = [Spec("title", caption=CAP_TITLE)] * secs(max(7.0, len(CAP_TITLE.split()) / 2.5 + 1.5))
    peak_ms = int(round(1e3 * (d["peak_t"] - d["t_up"])))
    first_onset = min([float(v.min()) for v in d["onsets_f"] if v.size] or [np.inf])
    onset_ms = int(round(1e3 * (first_onset - d["t_up"]))) if np.isfinite(first_onset) else None
    # slow motion: one frame per STATE_MS / SLOW_FACTOR ms of simulated time is not an integer; draw every 10 ms
    # row SLOW_FACTOR * FPS * 0.01 = 3 frames, so 1 s of simulated time lasts 10 s
    per_row = int(round(SLOW_FACTOR * C.FPS * STATE_MS / 1e3))
    for t in range(SLOW[0], SLOW[1] + 1, STATE_MS):
        if onset_ms is not None and t >= onset_ms + 200 and t < REAL[0]:
            text = CAP_ONSET
        elif t >= peak_ms:
            text = CAP_PEAK.format(**cap)
        else:
            text = CAP_SLOW.format(**cap)
        fr += [Spec("run", t, f"×1/{SLOW_FACTOR}", text)] * per_row
    # real time: 30 frames per second of simulated time = one row per 33.3 ms; draw every 3rd/4th row so the
    # speed is exact on average (asserted in check_timeline)
    n_real = secs((REAL[1] - REAL[0]) / 1e3)
    for k in range(n_real):
        t = REAL[0] + int(round((k + 1) * (REAL[1] - REAL[0]) / n_real / STATE_MS)) * STATE_MS
        fr.append(Spec("run", min(t, REAL[1]), "×1", CAP_REAL))
    tally = CAP_TALLY.format(n_f=int(sum(len(v) for c, v in enumerate(d["onsets_f"]) if c != d["cable"])),
                             n_c=int(sum(len(v) for c, v in enumerate(d["onsets_c"]) if c != d["cable"])), n_causal=int(d["causal"].sum()))
    fr += [Spec("tally", REAL[1], "paused", tally)] * secs(9.0)
    if rho is not None:
        n = len(rho["sweep"])
        fr += [Spec("rho", caption=CAP_RHO, step=0)] * secs(3.0)
        for s in range(1, n + 1):
            fr += [Spec("rho", caption=CAP_RHO if s <= n // 2 else CAP_RHO_FIT, step=s)] * secs(1.2)
        fr += [Spec("rho_end", caption=rho["sentence"], step=n)] * secs(max(6.0, len(rho["sentence"].split()) / 2.5 + 1.0))
    return pace(fr, hold_at_start={CAP_SLOW.format(**cap)})


def reading_time(text: str) -> float:
    return max(4.0, len(text.split()) / 2.5)


def pace(fr: list[Spec], hold_at_start: set) -> list[Spec]:
    """Every caption stays on screen for its reading time: a run that is too short is padded by holding one of
    its frames (the first for captions in ``hold_at_start``, else the last) with the clock marked "paused"."""
    out, i = [], 0
    while i < len(fr):
        j = i
        while j < len(fr) and fr[j].caption == fr[i].caption:
            j += 1
        run = fr[i:j]
        short = secs(reading_time(fr[i].caption)) - len(run) if fr[i].caption else 0
        if short > 0:
            anchor = run[0] if fr[i].caption in hold_at_start else run[-1]
            held = Spec(anchor.kind, anchor.t_ms, "paused" if anchor.kind == "run" else anchor.speed, anchor.caption, anchor.step)
            run = ([held] * short + run) if fr[i].caption in hold_at_start else (run + [held] * short)
        out += run
        i = j
    return out


def check_timeline(man: C.Manifest, fr: list[Spec], d: dict) -> None:
    caps = {}
    for i, s in enumerate(fr):
        if s.caption:
            caps.setdefault(s.caption, []).append(i)
    for text, ks in caps.items():
        on = len(ks) / C.FPS
        need = max(4.0, len(text.split()) / 2.5)
        man.check(f"caption on screen >= {need:.1f} s: '{text[:40]}...'", on >= need - 1e-9, f"{on:.1f} s")
    # speeds are measured over the moving frames only; held frames are labelled "paused" on the clock
    slow = [s for s in fr if s.kind == "run" and s.speed == f"×1/{SLOW_FACTOR}"]
    sim = (slow[-1].t_ms - slow[0].t_ms) / 1e3
    per_row = int(round(SLOW_FACTOR * C.FPS * STATE_MS / 1e3))     # frames per drawn row; n rows span n - 1 intervals
    speed = (len(slow) - per_row) / C.FPS / sim
    man.check("slow-motion speed is 1/10 within 0.5 % over the moving frames", abs(speed - SLOW_FACTOR) / SLOW_FACTOR < 0.005,
              f"{speed:.3f}; {sum(s.speed == 'paused' for s in fr) / C.FPS:.1f} s of held frames in the run segments")
    real = [s for s in fr if s.kind == "run" and s.speed == "×1"]
    sim = (real[-1].t_ms - real[0].t_ms) / 1e3
    man.check("real-time speed is 1 within 2 % over the moving frames", abs(len(real) / C.FPS / sim - 1.0) < 0.02, f"{len(real) / C.FPS / sim:.3f}")


def _interp(poses: np.ndarray, t: np.ndarray, t_abs: float) -> np.ndarray:
    k = int(np.searchsorted(t, t_abs, side="right") - 1)
    k = max(0, min(k, len(t) - 2))
    a = (t_abs - t[k]) / (t[k + 1] - t[k])
    return (1.0 - a) * poses[k] + a * poses[k + 1]


def camera(d: dict, box):
    from tether.physics import fleet as F
    P = []
    lo_t, hi_t = d["t_up"] + SLOW[0] / 1e3, d["t_up"] + REAL[1] / 1e3
    sel = (d["t"] >= lo_t - 0.05) & (d["t"] <= hi_t + 0.05)
    for poses in (d["poses_f"][sel], d["poses_c"][sel]):
        for row in poses:
            lp, vp = C.unpack_state(np.concatenate([row, np.zeros(18)]))
            P.append(C.body_polygon(lp, F.pentagon_vertices()))
            P += [C.body_polygon(v, C.hull()) for v in vp]
    P = np.vstack(P)
    lo, hi = P.min(0), P.max(0)
    box_aspect = (box[2] * C.W) / (box[3] * C.H)
    return 0.5 * (lo + hi), C.fit_aspect(0.5 * (hi - lo) + np.array([2.5, 2.5]), box_aspect)


class _Geom:
    def __init__(self, load, vessel):
        self.load_offsets, self.vessel_offsets = np.asarray(load, float), np.asarray(vessel, float)


def draw_fleet(fig, box, d, branch: str, cam, t_abs: float, geom):
    ax = fig.add_axes(box)
    poses = d["poses_f"] if branch == "factual" else d["poses_c"]
    T = d["T_f"] if branch == "factual" else d["T_c"]
    row = _interp(poses, d["t"], t_abs)
    k = int(np.searchsorted(d["t"], t_abs, side="right") - 1)
    state = np.concatenate([row, np.zeros(18)])
    C.draw_fleet(ax, state, geom, T[k], centre=cam[0], half=cam[1], labels=False, scale_bar=(branch == "factual"))
    lp, vp = C.unpack_state(state)
    for i in range(len(vp)):
        bow = C.body_polygon(vp[i], np.array([[0.5 * C.L_HULL + 1.0, 0.0]]))[0]
        ax.text(*bow, str(i), fontsize=C.FS_SUB, weight="bold", color=CAB[i], ha="center", va="center", zorder=7)
    head = "as recorded (factual)" if branch == "factual" else "snap removed (counterfactual)"
    ax.text(0.02, 0.97, head, transform=ax.transAxes, fontsize=C.FS_SUB, weight="bold",
            color=C.INK if branch == "factual" else C.SLACK, ha="left", va="top", zorder=8)
    if branch != "factual" and t_abs >= d["t_up"]:
        from matplotlib.transforms import offset_copy
        # a fixed 30 px below the header, whatever the panel's height
        ax.text(0.02, 0.97, f"cable {d['cable']} applies no force from t_up on", fontsize=C.FS_SMALL, color=C.SLACK, ha="left", va="top",
                zorder=8, transform=offset_copy(ax.transAxes, fig=fig, x=0, y=-30, units="dots"))
    return ax


def draw_traces(fig, box, d, branch: str, t_abs: float, flags: bool = True):
    ax = fig.add_axes(box)
    T = d["T_f"] if branch == "factual" else d["T_c"]
    onsets = d["onsets_f"] if branch == "factual" else d["onsets_c"]
    tau = (d["t"] - d["t_up"]) * 1e3
    lo, hi = SLOW[0], REAL[1]
    sel = (tau >= lo) & (tau <= hi)
    ymax = max(1.05 * float(d["T_f"].max()) / 1e3, 1.0)
    for i in range(T.shape[1]):
        vis = sel & (d["t"] <= t_abs + 1e-9)
        ax.plot(tau[vis], T[vis, i] / 1e3, color=CAB[i], lw=2.0 if i == d["cable"] else 1.4, alpha=1.0 if i == d["cable"] else 0.85)
    ax.axvline(0.0, color=C.MUTED, lw=0.8, ls=(0, (3, 2)))
    if flags:
        for c, v in enumerate(onsets):
            if c == d["cable"]:
                continue
            for t_on in v:
                ms = (t_on - d["t_up"]) * 1e3
                if ms <= (t_abs - d["t_up"]) * 1e3:
                    ax.plot([ms], [0.0], marker="v", ms=13, color=CAB[c], mec="white", mew=1.0, clip_on=False, zorder=6)
    ax.set_xlim(lo, hi); ax.set_ylim(0.0, ymax)
    ax.set_xlabel("time from t_up (ms)")
    ax.set_ylabel("tension (kN)")
    ax.grid(True)
    ax.set_title("five cable tensions" + (" · ▼ another cable goes slack" if flags else ""), fontsize=C.FS_SMALL, loc="left", color=C.MUTED)
    return ax


def furniture(fig, d, *, clock_t=None, speed=None, caption=""):
    C.title(fig, TITLE, SUB.format(cell=d["cell"], seed=d["seed"], cable=d["cable"], t_up=d["t_up"], peak=d["T_peak"] / 1e3, x=d["x"]))
    if clock_t is not None:
        C.clock(fig, clock_t, speed)
        fig.text(0.97, 0.912, f"t − t_up = {int(round(1e3 * (clock_t - d['t_up']))):+5d} ms", fontsize=C.FS_SUB, family="DejaVu Sans Mono",
                 color=C.INK, ha="right", va="top")
    if caption:
        C.caption(fig, two_lines(caption), y=CAPTION_Y)
    fig.text(0.015, 0.034, FOOTER_2, fontsize=C.FS_TINY - 1, color=C.MUTED, ha="left", va="bottom")
    fig.text(0.015, 0.012, FOOTER, fontsize=C.FS_TINY - 1, color=C.MUTED, ha="left", va="bottom")
    if VALIDATION["text"]:      # third footer line: the campaign's validation verdict and this parent's own residuals
        fig.text(0.015, 0.056, VALIDATION["text"], fontsize=C.FS_TINY - 1, color=C.MUTED, ha="left", va="bottom")


def draw_title(fig, d):
    fig.clf()
    furniture(fig, d, caption=CAP_TITLE)
    fig.text(0.5, 0.70, "Does removing the snap remove the follow-on slack?", fontsize=34, weight="bold", color=C.INK, ha="center", va="center")
    lines = [f"The event is cable {d['cable']}'s re-engagement at t_up = {d['t_up']:.3f} s in seed {d['seed']} of cell {d['cell']}; "
             f"it peaks at {d['T_peak'] / 1e3:.1f} kN, {d['x']:.1f} times the pretension T0 = {d['pretension']:.0f} N.",
             "Both movies start from the same recorded plant state under the same recorded weather and headings, integrated "
             "by the campaign's own intervention integrator, whose factual branch reproduces the record.",
             "In the right-hand movie the snapping cable applies no force from t_up on, so the payload never receives the snap; "
             "every other cable is unchanged."]
    y = 0.58
    for line in lines:
        fig.text(0.5, y, _wrap(line, 100), fontsize=C.FS_BODY + 2, color=C.INK, ha="center", va="top", linespacing=1.4)
        y -= 0.13


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width))


def draw_run(fig, spec: Spec, d, cam, geom, tally: bool = False):
    fig.clf()
    t_abs = d["t_up"] + spec.t_ms / 1e3
    furniture(fig, d, clock_t=t_abs, speed=spec.speed, caption=spec.caption)
    if tally:      # its own card: shorter panels (own camera), the counts outside the axes
        cam = camera(d, FLEET_L_TALLY)
        draw_fleet(fig, FLEET_L_TALLY, d, "factual", cam, t_abs, geom)
        draw_fleet(fig, FLEET_R_TALLY, d, "counterfactual", cam, t_abs, geom)
    else:
        draw_fleet(fig, FLEET_L, d, "factual", cam, t_abs, geom)
        draw_fleet(fig, FLEET_R, d, "counterfactual", cam, t_abs, geom)
    draw_traces(fig, TRACE_L, d, "factual", t_abs)
    draw_traces(fig, TRACE_R, d, "counterfactual", t_abs)
    if not tally:
        C.cable_key(fig, y=KEY_Y, x=0.5)
    if tally:
        n_f = int(sum(len(v) for c, v in enumerate(d["onsets_f"]) if c != d["cable"]))
        n_c = int(sum(len(v) for c, v in enumerate(d["onsets_c"]) if c != d["cable"]))
        fig.text(0.255, TALLY_Y, f"{n_f} slack onsets on other cables within 3 s", fontsize=21, weight="bold", color=C.INK, ha="center", va="center", zorder=9)
        fig.text(0.745, TALLY_Y, f"{n_c} slack onsets on other cables within 3 s", fontsize=21, weight="bold", color=C.SLACK, ha="center", va="center", zorder=9)
        fig.text(0.5, KEY_Y, f"{n_f} − {n_c} = {n_f - n_c} onsets caused by this snap in the replay", fontsize=22, weight="bold", color=C.ACCENT,
                 ha="center", va="center", zorder=9)


def draw_rho(fig, spec: Spec, d, rho: dict):
    fig.clf()
    C.title(fig, "What the fitted branching matrix says", "Spectral radius ρ along the T0 = 0.6 kN sweep · empirical fallback matrix, no mechanism")
    C.caption(fig, two_lines(spec.caption), y=CAPTION_Y)
    fig.text(0.015, 0.012, "records/v3/wp3_results.json (cells[name].rho, operable; tests.T3.4) · each ρ is a point estimate from two fitting "
             "runs; the record's intervals are degenerate and are not drawn (addendum 4)", fontsize=C.FS_TINY - 1, color=C.MUTED, ha="left", va="bottom")
    ax = fig.add_axes([0.10, 0.25, 0.80, 0.58])
    ax.axhline(1.0, color=C.INK, lw=1.2)
    ax.text(0.01, 1.0, "ρ = 1: critical", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=C.FS_SMALL, color=C.INK)
    sweep = rho["sweep"][: spec.step]
    if len(sweep) > 1:
        ax.plot([s["intensity"] for s in sweep], [s["rho"] for s in sweep], color=C.FAINT, lw=1.2, ls=(0, (4, 3)), zorder=2)
    for s in sweep:
        ax.plot(s["intensity"], s["rho"], "o", ms=12, color=C.TAUT if s["operable"] else "white", mec=C.TAUT, mew=2.0, zorder=5)
        ax.text(s["intensity"], s["rho"] + 0.04, f"ρ = {s['rho']:.2f}", ha="center", va="bottom", fontsize=C.FS_SMALL, color=C.INK)
    if spec.kind == "rho_end" and rho.get("critical_intensity") is not None:
        ax.axvline(rho["critical_intensity"], color=C.SLACK, lw=1.5, ls="--")
        ax.text(rho["critical_intensity"], 0.05, f"  I_c ≈ {rho['critical_intensity']:.2f}", fontsize=C.FS_BODY, color=C.SLACK)
    xs = [s["intensity"] for s in rho["sweep"]]
    ax.set_xlim(min(xs) - 0.1, max(xs) + 0.1)
    ax.set_ylim(0.0, max(1.3, 1.15 * max(s["rho"] for s in rho["sweep"])))
    ax.set_xlabel("weather intensity"); ax.set_ylabel("spectral radius ρ(K)")
    ax.grid(True)
    ax.text(0.01, 0.97, "filled: operable cell (closures ≤ 25 %); hollow: not operable", transform=ax.transAxes, fontsize=C.FS_SMALL,
            color=C.MUTED, ha="left", va="top")


def draw(fig, spec: Spec, d, cam, geom, rho):
    if spec.kind == "title":
        draw_title(fig, d)
    elif spec.kind == "run":
        draw_run(fig, spec, d, cam, geom)
    elif spec.kind == "tally":
        draw_run(fig, spec, d, cam, geom, tally=True)
    else:
        draw_rho(fig, spec, d, rho)


# ----------------------------------------------------------------------------- layout checks


def _bboxes(fig):
    r = fig.canvas.get_renderer()
    return r, ([t for t in fig.texts if t.get_text()] + [t for ax in fig.axes for t in ax.texts if t.get_text()])


def check_captions(man: C.Manifest, fig, fr: list[Spec], d, cam, geom, rho) -> None:
    caps, bad, n_lines = {}, [], []
    for i, s in enumerate(fr):
        if s.caption:
            caps.setdefault(s.caption, []).append(i)
    for cap, ks in caps.items():
        n_lines.append(two_lines(cap).count("\n") + 1)
        for i in sorted({ks[0], ks[-1]}):
            draw(fig, fr[i], d, cam, geom, rho)
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            ct = [t for t in fig.texts if t.get_text() == two_lines(cap)]
            if len(ct) != 1:
                bad.append((cap[:30], i, "caption not found once")); continue
            cb = ct[0].get_bbox_patch().get_window_extent(r)
            if cb.x0 < 0.01 * C.W or cb.x1 > 0.99 * C.W or cb.y0 < 0 or cb.y1 > C.H:
                bad.append((cap[:30], i, "outside the frame"))
            others = ([t.get_window_extent(r) for t in fig.texts if t is not ct[0] and t.get_text()]
                      + [lg.get_window_extent(r) for lg in fig.legends] + [ax.get_tightbbox(r) for ax in fig.axes])
            if any(cb.overlaps(o) for o in others):
                bad.append((cap[:30], i, "overlaps another artist"))
    man.check("every caption renders in at most two lines, inside the frame, clear of every other text, legend and axes",
              not bad and max(n_lines) <= 2, f"{len(caps)} captions, lines {n_lines}; problems {bad}")


def check_text_layout(man: C.Manifest, fig, fr: list[Spec], d, cam, geom, rho, indices) -> None:
    """On the inspected frames no two text artists overlap and every text lies inside the frame (the vessel numbers,
    which sit on the fleet by design, are excluded from the pairwise test but not from the frame test)."""
    problems = []
    for i in indices:
        draw(fig, fr[i], d, cam, geom, rho)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        items = [(t.get_text(), t.get_window_extent(r)) for t in fig.texts if t.get_text()]
        items += [(t.get_text(), t.get_window_extent(r)) for ax in fig.axes for t in ax.texts if t.get_text()]
        items += [(f"legend", lg.get_window_extent(r)) for lg in fig.legends]
        # figure-level texts must also stay clear of every axes (its ticks and labels included)
        axes_boxes = [ax.get_tightbbox(r) for ax in fig.axes]
        for t in fig.texts:
            if t.get_text() and any(t.get_window_extent(r).overlaps(b) for b in axes_boxes):
                problems.append((i, t.get_text()[:30], "overlaps an axes"))
        for label, bb in items:
            if bb.x0 < 0 or bb.x1 > C.W or bb.y0 < 0 or bb.y1 > C.H:
                problems.append((i, label[:30], "outside the frame"))
        prose = [(l, b) for l, b in items if not (len(l) == 1 and l.isdigit())]
        for a in range(len(prose)):
            for b in range(a + 1, len(prose)):
                if prose[a][1].overlaps(prose[b][1]):
                    problems.append((i, prose[a][0][:30], prose[b][0][:30]))
    man.check("on the inspected frames no two texts overlap and every text lies inside the frame", not problems, str(problems[:6]))


def check_tally_banner(man: C.Manifest, fig, fr: list[Spec], d, cam, geom, rho) -> None:
    """The tally boxes cover no body and no cable of either fleet panel at the tally frame."""
    from tether.physics import fleet as F
    spec = next(s for s in fr if s.kind == "tally")
    draw(fig, spec, d, cam, geom, rho)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in fig.texts if "slack onsets on other cables" in t.get_text() or "onsets caused by this snap in the replay" in t.get_text()]
    t_abs = d["t_up"] + spec.t_ms / 1e3
    clear = True
    for ax, poses in zip([a for a in fig.axes if a.get_position().height > 0.3], (d["poses_f"], d["poses_c"])):
        row = _interp(poses, d["t"], t_abs)
        lp, vp = C.unpack_state(np.concatenate([row, np.zeros(18)]))
        pts = [C.body_polygon(lp, F.pentagon_vertices())] + [C.body_polygon(v, C.hull()) for v in vp]
        a, b = C.attachment_points(lp, vp, _Geom(d["load_offsets"], d["vessel_offsets"]))
        for i in range(len(vp)):
            pts.append(np.linspace(a[i], b[i], 40))
        P = ax.transData.transform(np.vstack(pts))
        for bx in boxes:
            if np.any((P[:, 0] >= bx.x0) & (P[:, 0] <= bx.x1) & (P[:, 1] >= bx.y0) & (P[:, 1] <= bx.y1)):
                clear = False
    man.check("the tally banners cover no body and no cable at the tally frame", clear)


# ----------------------------------------------------------------------------- the rho segment


def rho_segment(man: C.Manifest) -> dict | None:
    path = campaign.RECORD_DIR / "wp3_results.json"
    if not path.exists():
        return None
    man.source(path)
    wp3 = json.loads(path.read_text())
    sweep = []
    for name in cells.sweep_cells():
        c = wp3["cells"].get(name, {})
        if "rho" not in c:
            continue
        sweep.append({"cell": name, "intensity": cells.parse_cell(name)["intensity"], "rho": float(c["rho"]), "ci": [float(v) for v in c["rho_ci95"]],
                      "operable": bool(c["operable"])})
    sweep.sort(key=lambda s: s["intensity"])
    t34 = wp3["tests"]["T3.4"]
    branch = t34["branch"]
    if branch == "(a)":
        sentence = (f"Along the sweep the spectral radius crosses one: the cascade turns critical near intensity {t34['critical_intensity']:.2f}, "
                    f"where a snap's offspring, on average, replace it.")
    elif branch == "(b)":
        # the branch is about the highest OPERABLE cell, which the record names; the sweep's highest intensity is
        # usually a non-operable cell and quoting it would put a number on screen the branch is not about
        top = next((s for s in sweep if s["cell"] == t34.get("highest_operable")), None)
        where = f"weather intensity {top['intensity']:g}" if top else "the highest operable intensity"
        # addenda 4 and 5: the margin is a point estimate from a kernel fitted on two runs, with no supported interval,
        # and subcriticality in the operable range is consistent with the data rather than measured with a margin
        sentence = (f"At {where}, the harshest the formation survives, ρ is {t34['rho']:.2f}: a two-run point estimate "
                    f"with no interval, so staying below one is consistent with the data, not measured.")
    else:
        sentence = "The sweep's intervals straddle ρ = 1, so the declared rule calls the transition under-powered rather than observed."
    for s in sweep:
        man.value(f"rho {s['cell']}", s["rho"], "", f"wp3_results.json cells[{s['cell']}].rho",
                  "point estimate from the cell's two fitting runs; the record's rho_ci95 is degenerate and is not drawn (addendum 4)")
    man.value("T3.4 branch", branch, "", "wp3_results.json tests.T3.4.branch")
    return {"sweep": sweep, "branch": branch, "critical_intensity": t34.get("critical_intensity"), "sentence": sentence,
            "kernel_path": wp3.get("kernel_path")}


# ----------------------------------------------------------------------------- main


def _load_row(npz: dict, k: int) -> dict:
    return {key: npz[key][k] for key in npz if key not in ("cell_names", "n")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="verify and stop")
    ap.add_argument("--stills", action="store_true", help="write inspection PNGs to the development directory ($TETHER_DEV_DIR)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--dry-run", nargs=3, metavar=("CELL", "SEED", "EVENT_ID"), help="development only: build from a reduced-run pickle in the development directory ($TETHER_DEV_DIR)")
    ap.add_argument("--dry-pickle", default=str(Path(os.environ.get("TETHER_DEV_DIR", Path(tempfile.gettempdir()) / "tether_dev")) / "v3_repro_2003.pkl"))
    args = ap.parse_args(argv)
    C.ensure_dirs()
    spec = compute.INTERVENTION
    man = C.Manifest(name=NAME, title=TITLE,
                     story=("One recorded snap replayed twice from the same plant state: as recorded, and with the snapping "
                            "cable's force removed. The onsets that disappear are the snap's causal offspring, the quantity the "
                            "branching kernel is built from; the second segment shows the kernel's spectral radius along the "
                            "intensity sweep against the critical level rho = 1."))
    clip, cache = CLIP, CACHE
    if args.dry_run:
        cell, seed, event_id = args.dry_run[0], int(args.dry_run[1]), int(args.dry_run[2])
        reduced = pickle.load(open(args.dry_pickle, "rb"))
        assert reduced["meta"]["cell"] == cell and reduced["meta"]["seed"] == seed
        raw = next(r for r in reduced["intervention"]["rows"] if r["event_id"] == event_id)
        row = {**raw, "t_cf": raw["factual_parent"]["t_cf"], "v_cf": raw["factual_parent"]["v_cf"], "T_peak_cf": raw["factual_parent"]["T_peak_cf"]}
        expected = {"t_up": np.asarray(reduced["marks"]["t_up"]), "T_peak": np.asarray(reduced["marks"]["T_peak"])}
        man.selection = f"DRY RUN (development): parent event {event_id} of {cell} seed {seed} from {args.dry_pickle}; not the declared selection"
        STILLS.mkdir(parents=True, exist_ok=True)
        clip, cache = STILLS / f"{NAME}.mp4", STILLS / "intervention_window.npz"
        C.MANIFESTS = STILLS
    else:
        npz_path = campaign.RECORD_DIR / "wp2_intervention.npz"
        for p in (npz_path, campaign.RECORD_DIR / "wp2_results.json", campaign.DECLARATIONS_PATH, compute.STAGE_CACHE["A"], Path(IV.__file__)):
            man.source(p)
        npz = dict(np.load(npz_path, allow_pickle=False))
        cell = cells.CALIBRATION_CELL
        k = select_parent(npz, cell)
        row = _load_row(npz, k)
        seed, event_id = int(row["seed"]), int(row["event_id"])
        wp2 = json.loads((campaign.RECORD_DIR / "wp2_results.json").read_text())
        t23 = wp2["tests"]["T2.3"]
        # the declared verdict is registered, never asserted: the clip shows one replay whose own residuals are checked below
        man.value("T2.3 (intervention validation) verdict", t23["verdict"], "", "wp2_results.json tests.T2.3",
                  f"n {t23['n']}; onset sets reproduced {t23['onset_set_reproduced_share']:.4f} vs 0.95 declared; |dt| max {1e3 * t23['abs_dt_s']['max']:.2f} ms vs 5; "
                  f"|dv| p95 {t23['abs_dv_m_s']['p95']:.4f} m/s vs 0.01; peak p95 {100 * t23['T_peak_relative_error']['p95_abs']:.2f} % vs 2")
        VALIDATION["text"] = (f"Campaign validation T2.3: {t23['verdict']} ({100 * t23['onset_set_reproduced_share']:.1f} % of {t23['n']} parents reproduced, "
                              f"declared 95 %); this parent reproduces the record within every declared tolerance (checked before drawing)")
        if t23["verdict"] != "PASS":
            man.caveats.append(f"The campaign's intervention validation T2.3 is {t23['verdict']} (onset sets reproduced in {100 * t23['onset_set_reproduced_share']:.2f} % of "
                               f"{t23['n']} parents against the declared 95 %; |dt| max {1e3 * t23['abs_dt_s']['max']:.2f} ms against 5 ms); by the declared branch the "
                               "campaign reports causal counts as untrusted. The clip shows one replay whose own residuals are within every tolerance and calls "
                               "its difference the onsets the snap caused in this replay, not a campaign-level causal count.")
        jobs, summaries = compute.load_stage("A")
        cached = next(s for j, s in zip(jobs, summaries) if j["cell"] == cell and int(j["seed"]) == seed)
        expected = {"t_up": np.asarray(cached["marks"]["t_up"]), "T_peak": np.asarray(cached["marks"]["T_peak"])}
        idx = _rows_for_cell(npz, cell)
        man.selection = (f"The calibration cell's sampled parent with the largest causal offspring count ({int(row['causal'].sum())} of "
                         f"{idx.size} sampled parents; ties by T_peak): seed {seed}, event {event_id}, cable {int(row['cable'])} at "
                         f"t_up = {float(row['t_up']):.3f} s, T_peak = {float(row['T_peak']) / 1e3:.2f} kN.")
        man.value("causal offspring (declared rule maximum)", int(row["causal"].sum()), "", f"wp2_intervention.npz causal[{k}]")
    if cache.exists():
        d = load_cache(man, cache, cell, seed, event_id, row)
    else:
        d = replay(man, cell, seed, event_id, row, spec, expected)
        save_cache(cache, d, man)
    rho = rho_segment(man)
    fr = timeline(d, rho)
    check_timeline(man, fr, d)
    geom = _Geom(d["load_offsets"], d["vessel_offsets"])
    cam = camera(d, FLEET_L)
    for label, value, unit, src in (("T_peak", d["T_peak"], "N", "record mark"), ("x = T_peak/T0", d["x"], "", "record"),
                                    ("t_up", d["t_up"], "s", "record mark"), ("peak time after t_up", 1e3 * (d["peak_t"] - d["t_up"]), "ms", "factual branch (10 ms grid)"),
                                    ("factual other-cable onsets within 3 s", int(sum(len(v) for c, v in enumerate(d["onsets_f"]) if c != d["cable"])), "", "integrate(x, None)"),
                                    ("counterfactual other-cable onsets within 3 s", int(sum(len(v) for c, v in enumerate(d["onsets_c"]) if c != d["cable"])), "", "integrate(x, cable)"),
                                    ("causal offspring", int(d["causal"].sum()), "", "causal_offspring (= record row)"),
                                    ("replay dt", 1e3 * (d["t_cf"] - d["t_up"]), "ms", "factual parent"), ("replay dv", d["v_cf"] - d["v_return"], "m/s", "factual parent"),
                                    ("replay peak error", 100 * (d["T_peak_cf"] / d["T_peak"] - 1), "%", "factual parent")):
        man.value(label, value, unit, src)
    man.caveats = [
        "One parent chosen for the largest causal count (clarity, not typicality); the campaign's T2.4 compares causal and observational counts over all sampled parents.",
        "The counterfactual removes the snapping cable's force from t_up on and holds that vessel's heading at its t_up value; every other input (weather, other headings, thrusts) is the record's.",
        "Fleet poses are the integrator's 10 ms output, drawn linearly in between; tensions are the plant's law on the integrator's elongations and rates at the same grid, so sub-10 ms detail is not shown.",
        "The factual branch is a re-integration, not the record itself; its residuals against the record are in the values above and within the declared T2.3 tolerances.",
    ]
    if rho is None:
        man.caveats.append("records/v3/wp3_results.json was absent when this clip was built: the criticality segment is omitted.")
    fig = C.new_frame()
    picks = sorted({0, next(i for i, s in enumerate(fr) if s.kind == "run"), next(i for i, s in enumerate(fr) if s.kind == "run" and s.t_ms >= 100),
                    next(i for i, s in enumerate(fr) if s.kind == "run" and s.t_ms >= 500),
                    next(i for i, s in enumerate(fr) if s.kind == "run" and s.speed == "×1"), next(i for i, s in enumerate(fr) if s.kind == "tally")}
                   | ({next(i for i, s in enumerate(fr) if s.kind == "rho_end")} if rho else set()))
    check_captions(man, fig, fr, d, cam, geom, rho)
    check_tally_banner(man, fig, fr, d, cam, geom, rho)
    check_text_layout(man, fig, fr, d, cam, geom, rho, picks)
    if args.check:
        print("checks passed:", len(man.checks)); return
    if args.stills:
        STILLS.mkdir(parents=True, exist_ok=True)
        for i in picks:
            draw(fig, fr[i], d, cam, geom, rho)
            fig.savefig(STILLS / f"frame_{i:05d}.png", dpi=C.DPI)
        print("stills:", STILLS)
    if args.no_video:
        return
    w = C.writer()
    with w.saving(fig, str(clip), dpi=C.DPI):
        last = None
        for s in fr:
            key = (s.kind, s.t_ms, s.speed, s.caption, s.step)
            if key != last:
                draw(fig, s, d, cam, geom, rho)
                last = key
            w.grab_frame()
    man.frames = C._FRAMES[0]
    if args.dry_run:      # the development clip is outside the repository: write the manifest by hand, same layout
        info = {"name": man.name, "title": man.title, "story": man.story, "clip": str(clip), "clip_sha256": C.sha256(clip), "fps": C.FPS,
                "frames": man.frames, "duration_s": round(man.frames / C.FPS, 3), "resolution": [C.W, C.H], "selection": man.selection,
                "sources": man.sources, "values": man.values, "checks": man.checks, "caveats": man.caveats,
                "captions": [{"t": round(f / C.FPS, 2), "text": t} for f, t in C.CAPTION_LOG]}
        (STILLS / f"{NAME}.json").write_text(json.dumps(info, indent=1, default=float))
    else:
        man.write(clip)
    print(f"wrote {clip}")


if __name__ == "__main__":
    main()
