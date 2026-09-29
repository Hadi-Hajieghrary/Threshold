"""C8 ``slack_criterion`` - the one genuinely new piece of theory the campaign confirms (paper VI):
drag decides which gusts hold a line slack.

Storyboard: Presentation/STORYBOARD.md, section C8.  Output: Presentation/clips/slack_criterion.mp4,
manifest Presentation/manifests/slack_criterion.json, replay caches
Presentation/cache/slack_criterion_lam{0p90,1p05,1p50}.npz.

What is simulated
-----------------
Three scripted gust runs of Phase 1 cell (c), re-simulated with the campaign's own code path:
``tether.campaign.v2.phase1_scripted.run_gust_job(GustJob('c', 1000.0, lam, 10.0))`` for
lam in (0.9, 1.05, 1.5) - i.e. ``build_run(job.spec(), 1, weather=gust_weather(1000, lam, 10))``,
``run_to_end``, then the campaign's summariser (``_summarize``) and main-mark extractor
(``_main_mark_extras``).  Parallel formation, T0 = 1 kN, k_h = 763 N m/rad, k_sigma = 3, sway limit
0.349 rad, no stochastic weather (weather_scale = 0), warm-up 0, 40 s runs, recording cables (no
severance).  The forcing is the declared square body force 1.06364 W^c on vessel 2 (test cable 2)
along world -x over [10 s, 20 s), W^c = lam T0.  To keep the plant logs for drawing, the module
attribute ``phase1_scripted.run_to_end`` is wrapped *in this process only* so that the run object
``run_gust_job`` builds is captured; nothing in tether/campaign is modified.

Records read (every on-screen number is registered in the manifest with one of these as source)
------------------------------------------------------------------------------------------------
* records/v2/phase1/scripted_runs.json, key ``runs[*]``: the run whose ``job`` is
  {cell 'c', pretension 1000.0, lam, t_x 10.0}.  Every key is compared with the replay; the keys
  that reach the screen or a targeted check are ``W_c_N``, ``job.{lam, pretension, t_x}``,
  ``main.{max_depth, V_up, T_peak, plant_peak_tension_one_period, t_up_after_gust_start,
  t_onset_after_gust_start, t_deep_after_gust_start, t_off, min_elongation_test_cable,
  test_cable_marks_after_gust_start}``, ``main.late_window.{drift_function_over_T0, samples}``,
  ``counts.collateral_marks``, ``closure.{terminal, time, cable}``, ``end_time``.
* records/v2/phase1/scripted_results.json, key ``P1-T6``: ``primary_drift_form[T0].{lambda,
  drift_function_over_T0, fitted_crossing, lambda_censored_by_closure}``, ``pooled_crossing``,
  ``committed_prediction.analytic.{drift_crossing_plan_free_load_c_eff,
  onset_acceleration_crossing_m_Lf, onset_acceleration_crossing_m_fleet}``, ``verdict``.
* records/v2/phase1/phase1_gate.json, key ``tests.P1-T6``: ``verdict``, ``predicted``,
  ``per_T0_measured_crossing``, ``measured_crossing_pooled``.
* records/v2/phase1/scripted_declarations.json, key ``tests.P1-T6.{readings, threshold}``: the
  drift form is the scored reading (late window [t_off - 2 s, t_off]); the onset-acceleration form
  is committed at 1.055 (m_Lf) / 1.077 (m_fleet); threshold "fitted crossing within 5% of 1".

Selection rule
--------------
Storyboard C8: at T0 = 1 kN and t_x = 10 s (the P1-T6 duration) the sweep holds lam in
{0.6, 0.9, 1.05, 1.25, 1.5, 2.0}; the clip plays lam = 0.9 and 1.05 (the grid values either side of
the predicted crossing lam = 1) and lam = 1.5 (well above it; the largest lam at this T0 still in
the scored fit - lam = 2.0 is censored).  Chosen for illustration; every scored point of all three
pretensions appears in the closing chart.

What is asserted before any frame is drawn (Manifest.check)
-----------------------------------------------------------
1. Each replay's full run record (every key of the scripted_runs.json entry except
   ``wall_seconds``) equals the campaign record: every leaf of the same Python type (int is not
   float, bool is not int), floats exactly or within 1e-9 relative.
   Singled out: main-mark t_up, depth, V_up (v_return) and T_peak for lam = 0.9 and 1.05; the
   formation-closure time and cable, end time and minimum gap for lam = 1.5 (which has no
   re-engagement: the record's closure is terminal at 18.883 s, before the gust ends).
2. The cached plant logs reproduce the record: minimum test-cable gap, late-window drift function
   (T + c_eff,f e')/T0 recomputed from the 1 ms log, re-engagement peak tension over one engagement
   period, the up-crossing time of e = 0, and the closure time (first 1 ms sample with a chord
   below 1 m).
3. The P1-T6 chart is recomputed from scripted_results.json: each per-T0 linear fit's zero equals
   the recorded ``fitted_crossing``, their mean equals ``pooled_crossing``, and both agree with
   phase1_gate.json; the three replayed drift values equal the recorded 1 kN points.  The chart's
   sign labels are asserted from ``primary_drift_form[T0].slack_fraction_late_window``: every
   plotted y > 0 point is taut through its window (fraction 0), every y < 0 point slack through it
   (fraction 1, so T = 0 there and y = c_eff,f <e'>/T0 < 0: the gap is still shrinking).
   Censoring departure (on screen, post hoc): the declaration (``tests.P1-T6.readings``,
   ``closure_censoring``) censors every t_x = 10 s run ended by formation closure before the gust's
   end, but ``score_t6`` keeps any run with late-window samples > 0, so the recorded 1 kN fit keeps
   lam = 1.5 (closure 18.883 s, 884 of 2001 samples).  Asserted: that run is the only
   closure-terminated run in any recorded fit.  Post hoc (in no record): re-fitting 1 kN without it
   gives 0.989, pooled 0.987, every crossing still within 5 % of 1 (asserted).

4. The campaign constants this module leans on: phase1_scripted.GUST_START = 10 s, TEST_CABLE = 2,
   RUN_SECONDS = 40 s, LATE_WINDOW = 2 s; fleet.CLOSURE_CHORD_LENGTH = 1 m.  Every drawn body of
   every played state row lies inside the common camera box (also asserted per frame by
   common.draw_fleet).  Each caption is one or two complete sentences, on screen >= 4 s, <= 2 lines,
   <= 2.5 words/s (numbers count as words), and the captions naming the closure and the snap are on
   screen when those instants play (and through the hold on each).  The two holds are asserted to
   sit on the first frame after the lam = 1.5 closure and on the lam = 1.05 peak-tension row.  The
   caption artist is driven by set_text, so write_manifest fills common.CAPTION_LOG from the schedule
   (mathtext stripped) and asserts the manifest's ``captions`` match it (assemble.py reads them).

Schedule (30 fps): title card 6.2 s; t = 5.5-10 s at real time; t = 10-24.5 s at x1/2 slow motion
with two paused holds (1.6 s on the first frame after the lam = 1.5 closure, t = 18.900 s; 2.1 s on
the lam = 1.05 re-engagement peak, t = 21.720 s), the clock reading "paused"; then the P1-T6 chart
in three stages (11.6 s, 6.8 s, 7.6 s final hold).  69.4 s in all.

Cache (Presentation/cache/slack_criterion_lam*.npz): the 1 ms cable log (event_time, elongation,
rate, alive) and the 10 ms state log up to t = 25 s, the declared gust force on vessel 2, the cable
constants, the full-run minimum gap, and in ``meta`` the replay's run record (JSON), the spec's
config hash and the sha256 of phase1_scripted.py at replay time.  Loading a cache re-runs checks
1-2 against the records, so a stale cache cannot be drawn.

Run: ``nice -n 10 python3 -m tether.analysis.v2.present.clip_slack_criterion [--replay]``
(``--replay`` re-simulates even if the caches exist; about 12-27 s of wall time per run).
``--stills 5,27 --stills-dir DIR`` saves PNG stills at those video seconds instead of the clip.
"""
from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tether.analysis.v2.present import common as C
from matplotlib import patheffects
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator

NAME = "slack_criterion"
T0 = 1000.0
T_X = 10.0
LAMS = (0.9, 1.05, 1.5)
TEST = 2
GUST = 10.0                 # gust start, s (= phase1_scripted.GUST_START, asserted)

RUNS_PATH = C.REPO / "records/v2/phase1/scripted_runs.json"
RESULTS_PATH = C.REPO / "records/v2/phase1/scripted_results.json"
GATE_PATH = C.REPO / "records/v2/phase1/phase1_gate.json"
DECL_PATH = C.REPO / "records/v2/phase1/scripted_declarations.json"
CAMPAIGN_MODULE = C.REPO / "tether/campaign/v2/phase1_scripted.py"

CACHE_UNTIL = 25.0          # s of plant log kept in the cache (everything drawn lies before this)
REL_TOL = 1.0e-9


def rel(path: Path) -> str:
    return str(path.relative_to(C.REPO))


def cache_path(lam: float) -> Path:
    return C.CACHE / (f"{NAME}_lam{lam:.2f}".replace(".", "p") + ".npz")


# ------------------------------------------------------------------------------------ records

def load_json(path: Path):
    return json.loads(path.read_text())


def campaign_record(lam: float) -> dict:
    runs = load_json(RUNS_PATH)["runs"]
    hits = [r for r in runs if r["job"]["cell"] == "c" and r["job"]["pretension"] == T0
            and r["job"]["lam"] == lam and r["job"]["t_x"] == T_X]
    assert len(hits) == 1, f"expected one scripted run for lam={lam}, found {len(hits)}"
    return hits[0]


def _normalise(value):
    """The campaign's own serialisation (json_bytes) read back, so types match the record."""
    from tether.campaign.common import json_bytes
    return json.loads(json_bytes(value))


def compare_records(replayed, recorded, skip=("wall_seconds",)):
    """Leaf-by-leaf comparison -> (n_leaves, n_exact, max_rel_dev, mismatches)."""
    stats = {"n": 0, "exact": 0, "max_rel": 0.0}
    bad: list[str] = []

    def walk(a, b, path):
        if isinstance(b, dict):
            if not isinstance(a, dict):
                bad.append(f"{path}: type {type(a).__name__} vs dict"); return
            keys = set(a) | set(b)
            for k in sorted(keys):
                if k in skip:
                    continue
                if k not in a or k not in b:
                    bad.append(f"{path}.{k}: missing in {'replay' if k not in a else 'record'}"); continue
                walk(a[k], b[k], f"{path}.{k}")
        elif isinstance(b, list):
            if not isinstance(a, list) or len(a) != len(b):
                bad.append(f"{path}: list length {len(a) if isinstance(a, list) else '?'} vs {len(b)}"); return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        else:
            stats["n"] += 1
            if type(a) is not type(b):                    # type-strict: 1 != 1.0, True != 1
                bad.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}"); return
            if isinstance(b, float):
                if (math.isnan(a) and math.isnan(b)) or a == b:
                    stats["exact"] += 1; return
                d = abs(a - b) / max(abs(b), 1e-300)
                stats["max_rel"] = max(stats["max_rel"], d)
                if d > REL_TOL:
                    bad.append(f"{path}: {a!r} vs {b!r}")
            elif a == b:
                stats["exact"] += 1
            else:
                bad.append(f"{path}: {a!r} vs {b!r}")

    walk(replayed, recorded, "")
    return stats["n"], stats["exact"], stats["max_rel"], bad


# ------------------------------------------------------------------------------------ replay

def simulate(lam: float):
    """Run the campaign's run_gust_job and keep the plant run it builds (see module docstring)."""
    from tether.campaign.v2 import phase1_scripted as ps
    job = ps.GustJob("c", T0, lam, T_X)
    captured = {}
    original = ps.run_to_end

    def capture(run):
        captured["run"] = original(run)
        return captured["run"]

    ps.run_to_end = capture
    try:
        record = ps.run_gust_job(job)
    finally:
        ps.run_to_end = original
    run = captured["run"]
    log = run.fleet.cables.log
    n, m = log.count, log.state_count
    keep = log.event_time[:n] <= CACHE_UNTIL + 1e-9
    keep_s = log.state_time[:m] <= CACHE_UNTIL + 1e-9
    weather = ps.gust_weather(job.pretension, job.lam, job.t_x)
    arrays = {
        "event_time": log.event_time[:n][keep],
        "elongation": log.elongation[:n][keep],
        "rate": log.rate[:n][keep],
        "alive": log.alive[:n][keep],
        "state_time": log.state_time[:m][keep_s],
        "state": log.state[:m][keep_s],
        "gust_force_x_vessel2": weather[:, TEST + 1, 0],
        "weather_period": np.array(0.01),
        "stiffness": np.array(run.fleet.cables.cable.stiffness),
        "damping": np.array(run.fleet.cables.cable.damping),
        "rest_length": np.array(run.fleet.cables.cable.rest_length),
        "full_log_end": np.array(log.event_time[n - 1]),
        "full_log_min_e_test": np.array(np.min(log.elongation[:n, TEST])),
        "initial_state": np.asarray(run.fleet.extras["initial_state"]),
    }
    meta = {
        "job": asdict(job),
        "config_hash": job.spec().config_hash(),
        "record_json": json.dumps(_normalise(record), sort_keys=True, allow_nan=True),
        "campaign_module_sha256": C.sha256(CAMPAIGN_MODULE),
        "replayed_with": "tether.campaign.v2.phase1_scripted.run_gust_job (run_to_end wrapped to keep the run)",
    }
    return arrays, meta


def save_cache(lam: float, arrays: dict, meta: dict) -> Path:
    C.ensure_dirs()
    path = cache_path(lam)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
    return path


def load_cache(lam: float):
    path = cache_path(lam)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    return arrays, meta


# ------------------------------------------------------------------------------------ checks

def check_replay(man: C.Manifest, lam: float, arrays: dict, meta: dict) -> dict:
    """Assert the replay reproduces the campaign record; return the record for the clip."""
    rec = campaign_record(lam)
    replayed = json.loads(meta["record_json"])
    n, exact, max_rel, bad = compare_records(replayed, rec)
    man.check(f"lam={lam}: replayed run record equals records/v2/phase1/scripted_runs.json entry "
              f"(all keys except wall_seconds; every leaf of the same type)", not bad,
              f"{n} leaves compared, {exact} bit-identical, max relative deviation {max_rel:.3g}"
              + (f"; mismatches: {bad[:5]}" if bad else ""))
    main, rmain = replayed["main"], rec["main"]
    if rec["closure"]["terminal"]:
        for key in ("time", "cable", "terminal"):
            man.check(f"lam={lam}: closure.{key} reproduced", replayed["closure"][key] == rec["closure"][key],
                      f"replay {replayed['closure'][key]} vs record {rec['closure'][key]}")
        man.check(f"lam={lam}: end_time reproduced", replayed["end_time"] == rec["end_time"],
                  f"{replayed['end_time']} s")
        man.check(f"lam={lam}: min test-cable gap reproduced",
                  main["min_elongation_test_cable"] == rmain["min_elongation_test_cable"],
                  f"{main['min_elongation_test_cable']} m")
        man.check(f"lam={lam}: no re-engagement on the test cable after the gust start (record: 0 marks)",
                  main["test_cable_marks_after_gust_start"] == 0 == rmain["test_cable_marks_after_gust_start"])
    else:
        for key in ("t_up_after_gust_start", "max_depth", "V_up", "T_peak"):
            man.check(f"lam={lam}: main mark {key} reproduced", main[key] == rmain[key],
                      f"replay {main[key]!r} vs record {rmain[key]!r}")

    # the cached plant log against the record
    t = arrays["event_time"]
    e = arrays["elongation"][:, TEST]
    rate = arrays["rate"][:, TEST]
    k_c, c_c = float(arrays["stiffness"]), float(arrays["damping"])
    tens = np.where(e > 0.0, np.maximum(k_c * e + c_c * rate, 0.0), 0.0)
    man.check(f"lam={lam}: min test-cable gap in the cached log equals the record",
              float(np.min(e)) == rmain["min_elongation_test_cable"]
              == float(arrays["full_log_min_e_test"]), f"{float(np.min(e))!r} m")
    t_off = rmain["t_off"]
    late = (t >= t_off - 2.0) & (t <= t_off)
    drift = float(np.mean(tens[late] + C_EFF_F * rate[late]) / T0)
    rlate = rmain["late_window"]
    man.check(f"lam={lam}: late-window drift function recomputed from the cached log equals the record",
              int(late.sum()) == rlate["samples"] and abs(drift - rlate["drift_function_over_T0"]) <= 1e-12,
              f"{int(late.sum())} samples, y = {drift!r} vs {rlate['drift_function_over_T0']!r}")
    out = {"record": rec, "drift": drift, "t": t, "e": e, "tension": tens}
    if not rec["closure"]["terminal"]:
        t_up = GUST + rmain["t_up_after_gust_start"]
        t_on = GUST + rmain["t_onset_after_gust_start"]
        from tether.campaign.v2 import events as ev
        win = (t >= t_up) & (t <= t_up + ev.ENGAGEMENT_PERIOD)
        peak = float(np.max(tens[win]))
        man.check(f"lam={lam}: re-engagement peak over one engagement period (cached log) equals record T_peak",
                  peak == rmain["plant_peak_tension_one_period"] == rmain["T_peak"], f"{peak!r} N")
        k_up = int(np.flatnonzero((t > t_on) & (e > 0.0))[0])
        man.check(f"lam={lam}: first e > 0 sample after onset lies within 1 ms of the record's t_up",
                  abs(t[k_up] - t_up) <= 1.0e-3 + 1e-9, f"sample {t[k_up]:.4f} s vs t_up {t_up:.6f} s")
        k_peak = int(np.flatnonzero(win)[np.argmax(tens[win])])
        out.update(t_up=t_up, t_onset=t_on, t_peak_time=float(t[k_peak]))
    else:
        rl = float(arrays["rest_length"])
        chord_min = np.min(arrays["elongation"], axis=1) + rl
        k_cl = int(np.flatnonzero(chord_min < 1.0)[0])
        man.check(f"lam={lam}: first 1 ms sample with a chord below 1 m is the record's closure time",
                  abs(t[k_cl] - rec["closure"]["time"]) <= 1e-9
                  and int(np.argmin(arrays["elongation"][k_cl])) == rec["closure"]["cable"],
                  f"t = {t[k_cl]:.4f} s, cable {int(np.argmin(arrays['elongation'][k_cl]))}")
        out.update(t_closure=rec["closure"]["time"])
    return out


def _c_eff_f() -> float:
    """c_eff,f = (1/c_A + 1/c_Lf)^-1, c_Lf = c_L + 4 c_A: the drift function's weight, exactly as
    phase1_scripted._main_mark_extras computes it."""
    from tether.campaign.v2 import regime as rg
    return 1.0 / (1.0 / rg.C_A + 1.0 / (rg.C_L + 4.0 * rg.C_A))


C_EFF_F = _c_eff_f()


def get_replays(man: C.Manifest, force: bool = False) -> dict:
    from tether.campaign.v2 import phase1_scripted as ps
    from tether.physics import fleet as F
    man.check("campaign constants used here: GUST_START 10 s, TEST_CABLE 2, RUN_SECONDS 40 s, "
              "LATE_WINDOW 2 s, CLOSURE_CHORD_LENGTH 1 m",
              ps.GUST_START == GUST == 10.0 and ps.TEST_CABLE == TEST and ps.RUN_SECONDS == 40.0
              and ps.LATE_WINDOW == 2.0 and F.CLOSURE_CHORD_LENGTH == 1.0)
    data = {}
    for lam in LAMS:
        path = cache_path(lam)
        if force or not path.exists():
            print(f"replaying lam = {lam} ...", flush=True)
            arrays, meta = simulate(lam)
            save_cache(lam, arrays, meta)
        arrays, meta = load_cache(lam)
        man.source(path)
        out = check_replay(man, lam, arrays, meta)
        out.update(arrays=arrays, meta=meta, cache=rel(path))
        data[lam] = out
    return data


# ------------------------------------------------------------------------------------ P1-T6 records

def t6_data(man: C.Manifest, replays: dict) -> dict:
    """The P1-T6 chart, recomputed from the records and cross-checked (see module docstring)."""
    res = load_json(RESULTS_PATH)["P1-T6"]
    gate = load_json(GATE_PATH)["tests"]["P1-T6"]
    decl = load_json(DECL_PATH)["tests"]["P1-T6"]
    per = {}
    for key in ("600", "1000", "1400"):
        blk = res["primary_drift_form"][key]
        lam = np.asarray(blk["lambda"], float)
        y = np.asarray(blk["drift_function_over_T0"], float)
        a, b = np.polyfit(lam, y, 1)
        cross = float(-b / a)
        man.check(f"P1-T6 T0 = {key} N: zero of the linear fit of the recorded (lambda, y) points equals "
                  f"scripted_results.json fitted_crossing", abs(cross - blk["fitted_crossing"]) <= 1e-12,
                  f"{cross!r} vs {blk['fitted_crossing']!r}")
        g = gate["per_T0_measured_crossing"][key]
        man.check(f"P1-T6 T0 = {key} N: crossing agrees with phase1_gate.json per_T0_measured_crossing "
                  f"(gate rounds 600/1400 to 5 decimals)", abs(cross - g) <= 5e-6, f"{cross:.6f} vs {g}")
        frac = np.asarray(blk["slack_fraction_late_window"], float)
        man.check(f"P1-T6 T0 = {key} N: chart sign labels hold on every plotted point (y > 0: taut through the "
                  f"scored window, slack fraction 0; y < 0: slack through it, fraction 1, so T = 0 there and "
                  f"y = c_eff,f <e'>/T0 < 0, the gap still shrinking)",
                  len(frac) == len(y) and bool(np.all(np.where(y > 0, frac == 0.0, frac == 1.0)))
                  and bool(np.all(y != 0.0)),
                  f"scripted_results.json P1-T6.primary_drift_form.{key}.slack_fraction_late_window {frac.tolist()}")
        per[key] = {"lam": lam, "y": y, "fit": (float(a), float(b)), "crossing": float(blk["fitted_crossing"]),
                    "censored": list(blk["lambda_censored_by_closure"])}
    pooled = float(np.mean([per[k]["crossing"] for k in per]))
    man.check("P1-T6 pooled crossing = mean of the three per-T0 crossings = scripted_results.json pooled_crossing "
              "= phase1_gate.json measured_crossing_pooled",
              abs(pooled - res["pooled_crossing"]) <= 1e-15 and abs(pooled - gate["measured_crossing_pooled"]) <= 1e-15,
              f"{pooled!r}")
    analytic = res["committed_prediction"]["analytic"]
    predicted = float(analytic["drift_crossing_plan_free_load_c_eff"])
    man.check("P1-T6 predicted crossing: scripted_results.json drift_crossing_plan_free_load_c_eff = "
              "phase1_gate.json predicted = 1", predicted == float(gate["predicted"]) == 1.0, f"{predicted}")
    in_lf = float(analytic["onset_acceleration_crossing_m_Lf"])
    in_fl = float(analytic["onset_acceleration_crossing_m_fleet"])
    man.check("inertial (onset-acceleration) crossings as committed in scripted_declarations.json P1-T6 readings",
              f"committed at {in_lf:.3f} (m_Lf) / {in_fl:.3f} (m_fleet)" in decl["readings"],
              f"{in_lf!r}, {in_fl!r}")
    man.check("the scored P1-T6 reading is the drift form over the late window [t_off - 2 s, t_off]",
              "PRIMARY (drift form" in decl["readings"] and "[t_off - 2 s, t_off]" in decl["readings"]
              and "the drift form is scored" in decl["readings"], "scripted_declarations.json tests.P1-T6.readings")
    man.check("P1-T6 declared threshold is 'fitted crossing within 5% of lambda = 1'",
              decl["threshold"] == "fitted crossing within 5% of lambda = 1", decl["threshold"])
    man.check("P1-T6 verdict PASS in scripted_results.json and phase1_gate.json",
              res["verdict"] == "PASS" == gate["verdict"])
    blk = per["1000"]
    for lam, rp in replays.items():
        i = int(np.flatnonzero(blk["lam"] == lam)[0])
        man.check(f"lam={lam}: the replayed late-window drift value is the chart's 1 kN point",
                  rp["drift"] == blk["y"][i], f"{rp['drift']!r}")

    # censoring: declared rule vs what the scoring code kept (shown on screen, with a post-hoc re-fit)
    man.check("declared P1-T6 censoring: a t_x = 10 s run terminated by formation closure before the shut-off "
              "is censored out of the fit (scripted_declarations.json tests.P1-T6.readings; closure_censoring: "
              "such cells contribute to no test population)",
              "terminated by formation closure (B.7) before the shut-off holds no late window; it is censored "
              "out of the fit" in decl["readings"]
              and "contribute to no test population" in load_json(DECL_PATH)["closure_censoring"])
    runs = load_json(RUNS_PATH)["runs"]
    kept_closed = []
    for key in per:
        for lam in per[key]["lam"]:
            r = [x for x in runs if x["job"]["cell"] == "c" and x["job"]["pretension"] == float(key)
                 and x["job"]["lam"] == lam and x["job"]["t_x"] == T_X]
            assert len(r) == 1
            if r[0]["closure"]["terminal"]:
                kept_closed.append((key, float(lam), r[0]["closure"]["time"], r[0]["main"]["t_off"],
                                    r[0]["main"]["late_window"]["samples"]))
    man.check("the only closure-terminated run inside any recorded P1-T6 fit is lambda = 1.5 at T0 = 1 kN, "
              "closed before the gust's end with a partial late window (scoring code keeps samples > 0)",
              len(kept_closed) == 1 and kept_closed[0][:2] == ("1000", 1.5) and kept_closed[0][2] < kept_closed[0][3]
              and 0 < kept_closed[0][4] < 2001, f"{kept_closed}")
    keep = blk["lam"] != 1.5
    a_ph, b_ph = np.polyfit(blk["lam"][keep], blk["y"][keep], 1)
    ph1000 = float(-b_ph / a_ph)
    ph_pooled = float(np.mean([per["600"]["crossing"], ph1000, per["1400"]["crossing"]]))
    from tether.campaign.v2 import phase1_scripted as ps
    man.check("post hoc: with that run censored as declared, every per-T0 crossing and the pooled crossing stay "
              "within the declared 5 % of 1 (phase1_scripted.T6_CROSSING; verdict unchanged)",
              ps.T6_CROSSING == 0.05 and all(abs(c - 1.0) <= ps.T6_CROSSING for c in
                                             (per["600"]["crossing"], ph1000, per["1400"]["crossing"], ph_pooled)),
              f"1 kN {ph1000!r}, pooled {ph_pooled!r}")
    return {"per": per, "pooled": pooled, "predicted": predicted, "inertial": (in_lf, in_fl),
            "threshold": decl["threshold"], "verdict": res["verdict"],
            "posthoc": {"1000": ph1000, "pooled": ph_pooled}}


# ------------------------------------------------------------------------------------ registration

class Reg:
    """Every on-screen number passes through here: registered once in the manifest, returned formatted."""

    def __init__(self, man: C.Manifest):
        self.man = man
        self.seen: dict[str, str] = {}

    def __call__(self, label, value, fmt="{}", unit="", source="", note="") -> str:
        text = fmt.format(value)
        if label in self.seen:
            assert self.seen[label] == text, f"{label} shown two ways: {self.seen[label]} / {text}"
        else:
            self.man.value(label, value, unit, source, (note + "; " if note else "") + f"shown as '{text}'")
            self.seen[label] = text
        return text


def run_src(lam: float, key: str) -> str:
    return f"{rel(RUNS_PATH)} runs[job = (c, T0 1000, lam {lam}, t_x 10)].{key}"


def minus(s: str) -> str:
    return s.replace("-", "−")


def plain(s: str) -> str:
    """Caption text for the speaker script: the on-screen mathtext (W$^c$, T$_0$) as plain text."""
    out = s.replace("W$^c$", "W^c").replace("T$_0$", "T0")
    assert "$" not in out, f"unconverted mathtext in caption: {s!r}"
    return out


GUST_ALPHA, SCORED_ALPHA = 0.08, 0.20
_A = np.array(to_rgb(C.ACCENT))
BG_GUST = tuple((1 - GUST_ALPHA) * np.ones(3) + GUST_ALPHA * _A)          # white under the gust band
BG_SCORED = tuple((1 - SCORED_ALPHA) * np.array(BG_GUST) + SCORED_ALPHA * _A)  # both bands


def halo(bg=C.BG):
    """Outline text in its local background colour so the moving cursor never shows through it."""
    return [patheffects.withStroke(linewidth=3.0, foreground=bg)]


# ------------------------------------------------------------------------------------ schedule

FPS = C.FPS
T_PRE0, T_GUST0, T_GUST_END = 5.5, 10.0, 24.5
N_TITLE, N_PRE, N_GUST = 186, 135, 870          # 6.2 s title card, 4.5 s at x1, 29 s of motion at x1/2
# Two paused holds inside the slow-motion span give the captions that name those instants their
# reading time; which gust frame each holds is asserted from the replays in Clip._holds.
K_HOLD_CLOSURE, N_HOLD_CLOSURE = 534, 48        # t = 18.900 s, first frame after the lam = 1.5 closure: 1.6 s
K_HOLD_PEAK, N_HOLD_PEAK = 703, 63              # t = 21.720 s, the lam = 1.05 peak-tension row: 2.1 s
N_CHART1, N_CHART2, N_FINAL = 348, 204, 228     # 11.6 s, 6.8 s, 7.6 s final hold (key frame)
SPEED_PRE, SPEED_GUST, SPEED_HOLD = "real time", "×1/2 slow motion", "paused"


def sim_times():
    pre = T_GUST0 - (N_PRE - np.arange(N_PRE)) / FPS           # 5.5 ... 9.967 at x1
    gust = T_GUST0 + np.arange(N_GUST) / (2.0 * FPS)            # 10.0 ... 24.483 at x1/2
    return pre, gust


def video_of_sim(t: float) -> float:
    """Video second at which simulation time t (>= 10 s) is on screen; a hold delays everything
    after the frame it holds."""
    f = 2.0 * FPS * (t - T_GUST0)                               # position in the x1/2 gust frames
    held = (N_HOLD_CLOSURE if f > K_HOLD_CLOSURE + 1e-9 else 0) + (N_HOLD_PEAK if f > K_HOLD_PEAK + 1e-9 else 0)
    return (N_TITLE + N_PRE + f + held) / FPS


def hold_span(k: int) -> tuple[float, float]:
    """Video interval [start, end) of the paused frames that follow gust frame k (a hold)."""
    n = {K_HOLD_CLOSURE: N_HOLD_CLOSURE, K_HOLD_PEAK: N_HOLD_PEAK}[k]
    before = N_TITLE + N_PRE + k + 1 + (N_HOLD_CLOSURE if k > K_HOLD_CLOSURE else 0)
    return before / FPS, (before + n) / FPS


# ------------------------------------------------------------------------------------ layout checks

def _extent(artist, renderer):
    """Text box (its bbox patch if it has one); for an annotation, the text only, not its leader line."""
    from matplotlib.text import Text
    patch = artist.get_bbox_patch() if hasattr(artist, "get_bbox_patch") else None
    if patch is not None:
        return patch.get_window_extent(renderer)
    return Text.get_window_extent(artist, renderer)


def check_layout(fig, where: str, views=()) -> None:
    """No two pieces of text overlap, no text leaves the frame or its axes, axes do not overlap, and in
    the fleet views no text sits on a hull or the payload (except labels marked gid='inside-ok')."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    fig_items = [(t.get_text()[:40], _extent(t, r)) for t in fig.texts if t.get_visible() and t.get_text().strip()]
    fig_items += [("legend", lg.get_window_extent(r)) for lg in fig.legends]
    axes_items = [(f"axes{i}", ax.get_tightbbox(r)) for i, ax in enumerate(fig.axes) if ax.get_visible()]
    problems = []
    for name, bb in fig_items + axes_items:
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
            problems.append(f"{name!r} leaves the frame {bb}")
    items = fig_items + axes_items
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][1].overlaps(items[j][1]):
                problems.append(f"{items[i][0]!r} overlaps {items[j][0]!r}")
    for k, ax in enumerate(fig.axes):
        box = ax.get_window_extent(r)
        inner = [(t.get_text()[:40], _extent(t, r), t.get_gid()) for t in ax.texts
                 if t.get_visible() and t.get_text().strip()]
        for name, bb, gid in inner:
            if gid == "tick-row":   # the axis unit sits on the tick-label row, outside the box by design
                continue
            if bb.x0 < box.x0 - 1 or bb.x1 > box.x1 + 1 or bb.y0 < box.y0 - 1 or bb.y1 > box.y1 + 1:
                problems.append(f"axes{k}: {name!r} clipped by its axes")
        for i in range(len(inner)):
            for j in range(i + 1, len(inner)):
                if inner[i][1].overlaps(inner[j][1]):
                    problems.append(f"axes{k}: {inner[i][0]!r} overlaps {inner[j][0]!r}")
    for k, ax in enumerate(views):
        bodies = [p.get_window_extent(r) for p in ax.patches]
        for t in ax.texts:
            if not t.get_visible() or not t.get_text().strip() or t.get_gid() == "inside-ok":
                continue
            bb = _extent(t, r)
            if any(bb.overlaps(pb) for pb in bodies):
                problems.append(f"view {k}: {t.get_text()[:40]!r} sits on a body")
    assert not problems, f"layout at {where}: " + "; ".join(problems)


# ------------------------------------------------------------------------------------ the clip

class Clip:
    ROW_TOP, HEAD_H, AX_H, GAP = 0.848, 0.034, 0.170, 0.016
    VIEW_X, VIEW_W, E_X, E_W = 0.03, 0.40, 0.505, 0.47
    KEY_Y = 0.151               # selection note + cable key row: clear of the tick labels and of a 2-line caption

    def __init__(self, man: C.Manifest, replays: dict, t6: dict):
        self.man, self.rp, self.t6 = man, replays, t6
        self.reg = Reg(man)
        self.geometry = C.parallel_geometry()
        self.fig = C.new_frame()
        self.segment = None
        self.caption_artist = None
        self._numbers()
        self._holds()
        self._captions()
        self._camera()

    # ---------------------------------------------------------------- numbers (all registered)
    def _numbers(self):
        R, rp = self.reg, self.rp
        caches = ", ".join(rp[l]["cache"] for l in LAMS)
        self.src_cache = caches
        self.n = {}
        for lam in LAMS:
            rec = rp[lam]["record"]
            m = rec["main"]
            assert rec["job"]["lam"] == lam and rec["job"]["pretension"] == T0 and rec["job"]["t_x"] == T_X
            self.n[lam] = {
                "lam": R(f"lambda (run {lam})", lam, "{:g}", "-", run_src(lam, "job.lam"),
                         "gust size in units of the predicted crossing (W^c/T0)"),
                "Wc": R(f"W^c (run {lam})", rec["W_c_N"], "{:.0f}", "N", run_src(lam, "W_c_N")),
            }
        m9, m105 = rp[0.9]["record"]["main"], rp[1.05]["record"]["main"]
        rec15 = rp[1.5]["record"]
        self.n["T0"] = R("T0 (the three replays)", T0 / 1e3, "{:g}", "kN", run_src(1.05, "job.pretension"), "/1000")
        self.n["one"] = R("predicted crossing (short form in row headers)", self.t6["predicted"], "{:g}", "-",
                          f"{rel(GATE_PATH)} tests.P1-T6.predicted")
        self.n["vessel"] = R("test vessel / cable index", TEST, "{:d}", "-",
                             f"{rel(DECL_PATH)} forcing_COMMITTED.pattern (cable 2, centre cable)")
        self.n["t_gust"] = R("gust start", 10.0, "{:.0f}", "s", f"{rel(DECL_PATH)} forcing_COMMITTED.pattern; "
                             "phase1_scripted.GUST_START", "square gust on [10 s, 10 s + t_x)")
        self.n["t_x"] = R("gust duration t_x", rp[1.05]["record"]["job"]["t_x"], "{:.0f}", "s", run_src(1.05, "job.t_x"))
        self.n["t_off"] = R("gust end t_off", m105["t_off"], "{:.0f}", "s", run_src(1.05, "main.t_off"))
        self.n["dep9_mm"] = R("lam=0.9 deepest slack (main mark depth)", m9["max_depth"] * 1e3, "{:.1f}", "mm",
                              run_src(0.9, "main.max_depth"), "x 1000, shown negative as e")
        self.n["t_deep9"] = GUST + m9["t_deep_after_gust_start"]
        self.n["dep105"] = R("lam=1.05 deepest slack (main mark depth)", m105["max_depth"], "{:.2f}", "m",
                             run_src(1.05, "main.max_depth"), "shown negative as e")
        self.n["t_deep105"] = GUST + m105["t_deep_after_gust_start"]
        self.n["tup105"] = R("lam=1.05 re-engagement time t_up", rp[1.05]["t_up"], "{:.3f}", "s",
                             run_src(1.05, "main.t_up_after_gust_start") + " + 10 s")
        self.n["vup105"] = R("lam=1.05 closing speed V_up", m105["V_up"], "{:.2f}", "m/s", run_src(1.05, "main.V_up"))
        self.n["Tp105"] = R("lam=1.05 re-engagement peak tension", m105["T_peak"] / 1e3, "{:.1f}", "kN",
                            run_src(1.05, "main.T_peak"), "/1000")
        self.n["coll105"] = R("lam=1.05 collateral marks on cables 0,1,3,4", rp[1.05]["record"]["counts"]["collateral_marks"],
                              "{:d}", "marks", run_src(1.05, "counts.collateral_marks"),
                              "shown from the first sample after the snap where another cable carries no tension")
        oth = np.delete(C.tension(rp[1.05]["arrays"]["elongation"], rp[1.05]["arrays"]["rate"],
                                  rp[1.05]["arrays"]["alive"]), TEST, axis=1)
        t105 = rp[1.05]["arrays"]["event_time"]
        k = int(np.flatnonzero((t105 > rp[1.05]["t_up"]) & np.any(oth <= 0.0, axis=1))[0])
        self.n["t_coll105"] = float(t105[k])
        self.man.check("lam=1.05: the other four cables carry no tension shortly after the snap (drawn dashed)",
                       0.0 < self.n["t_coll105"] - rp[1.05]["t_up"] < 0.2,
                       f"first such 1 ms sample {self.n['t_coll105']:.3f} s (t_up {rp[1.05]['t_up']:.3f} s)")
        self.n["tcl15"] = R("lam=1.5 formation-closure time", rec15["closure"]["time"], "{:.3f}", "s",
                            run_src(1.5, "closure.time"), "run terminated (B.7): a chord below 1 m")
        self.n["chord_cl"] = R("formation-closure chord threshold", 1.0, "{:.0f}", "m",
                               f"{rel(DECL_PATH)} closure_censoring; tether.physics.fleet.CLOSURE_CHORD_LENGTH")
        from tether.physics import fleet as F
        assert F.CLOSURE_CHORD_LENGTH == 1.0
        self.n["emin15"] = R("lam=1.5 gap at closure (minimum)", rec15["main"]["min_elongation_test_cable"], "{:.1f}",
                             "m", run_src(1.5, "main.min_elongation_test_cable"))
        # late window drift values (replay = record, asserted)
        res1000 = self.t6["per"]["1000"]
        for lam in LAMS:
            self.n[lam]["y"] = R(f"lam={lam} late-window drift function y", self.rp[lam]["drift"], "{:+.3f}", "-",
                                 run_src(lam, "main.late_window.drift_function_over_T0"),
                                 "= scripted_results.json P1-T6 primary_drift_form.1000 point; recomputed from the replay")
        lw15 = rec15["main"]["late_window"]
        self.n["samples15"] = R("lam=1.5 late-window samples", lw15["samples"], "{:d}", "1 ms samples",
                                run_src(1.5, "main.late_window.samples"),
                                "partial window: run ended at 18.883 s; kept in the recorded 1 kN fit")
        self.n["samples_full"] = R("full late-window samples", self.rp[0.9]["record"]["main"]["late_window"]["samples"],
                                   "{:d}", "1 ms samples", run_src(0.9, "main.late_window.samples"))
        self.n["late_lo"] = R("late window start", m105["t_off"] - 2.0, "{:.0f}", "s",
                              f"{rel(DECL_PATH)} tests.P1-T6.readings: [t_off - 2 s, t_off]")
        self.n["late_len"] = R("late window length", 2.0, "{:.0f}", "s", f"{rel(DECL_PATH)} tests.P1-T6.readings")
        # P1-T6
        per = self.t6["per"]
        self.n["cross"] = {k: R(f"P1-T6 crossing T0={k} N", per[k]["crossing"], "{:.3f}", "-",
                                f"{rel(RESULTS_PATH)} P1-T6.primary_drift_form.{k}.fitted_crossing",
                                "measured (plant); = phase1_gate.json per_T0_measured_crossing") for k in per}
        self.n["pooled"] = R("P1-T6 pooled crossing", self.t6["pooled"], "{:.3f}", "-",
                             f"{rel(GATE_PATH)} tests.P1-T6.measured_crossing_pooled",
                             "mean of the three per-T0 crossings (measured)")
        self.n["pred"] = R("P1-T6 predicted crossing", self.t6["predicted"], "{:.3f}", "-",
                           f"{rel(GATE_PATH)} tests.P1-T6.predicted",
                           "prediction (the criterion W^c = T0); lambda is in these units")
        self.n["band"] = R("P1-T6 declared band", 5, "{:d}", "%", f"{rel(DECL_PATH)} tests.P1-T6.threshold")
        self.n["in_lf"] = R("inertial onset-acceleration crossing (m_Lf)", self.t6["inertial"][0], "{:.3f}", "-",
                            f"{rel(RESULTS_PATH)} P1-T6.committed_prediction.analytic.onset_acceleration_crossing_m_Lf",
                            "committed prediction, not measured; also in scripted_declarations.json P1-T6 readings")
        self.n["in_fl"] = R("inertial onset-acceleration crossing (m_fleet)", self.t6["inertial"][1], "{:.3f}", "-",
                            f"{rel(RESULTS_PATH)} P1-T6.committed_prediction.analytic.onset_acceleration_crossing_m_fleet",
                            "committed prediction, not measured")
        self.n["T0s"] = {k: R(f"pretension {k} N", float(k) / 1e3, "{:.1f}", "kN",
                              f"{rel(RESULTS_PATH)} P1-T6.primary_drift_form keys") for k in per}
        cens = sorted({l for k in per for l in per[k]["censored"]})
        assert cens == [1.5, 2.0] and per["1400"]["censored"] == [1.5, 2.0] \
            and per["600"]["censored"] == [2.0] == per["1000"]["censored"]
        self.n["cens20"] = R("lambda censored at every T0", 2.0, "{:.1f}", "-",
                             f"{rel(RESULTS_PATH)} P1-T6.primary_drift_form.*.lambda_censored_by_closure")
        self.n["cens15"] = R("lambda censored at T0 = 1.4 kN", 1.5, "{:.1f}", "-",
                             f"{rel(RESULTS_PATH)} P1-T6.primary_drift_form.1400.lambda_censored_by_closure")
        ph_src = (f"{rel(RESULTS_PATH)} P1-T6.primary_drift_form.1000.{{lambda, drift_function_over_T0}} "
                  f"without lambda = 1.5 (closure 18.883 s, partial window)")
        ph_note = ("POST HOC, in no record: applies the declared censoring (scripted_declarations.json "
                   "tests.P1-T6.readings, closure_censoring) that score_t6 did not")
        self.n["ph1000"] = R("post hoc: 1 kN crossing with the lambda = 1.5 closure-ended run censored",
                             self.t6["posthoc"]["1000"], "{:.3f}", "-",
                             ph_src + "; zero of the linear fit (this module, t6_data)", ph_note)
        self.n["phpooled"] = R("post hoc: pooled crossing with that run censored", self.t6["posthoc"]["pooled"],
                               "{:.3f}", "-", ph_src + "; mean with the recorded 600 N and 1400 N fitted_crossing "
                               "(this module, t6_data)", ph_note)
        self.n["scale"] = R("scale bar", 5.0, "{:.0f}", "m", "this module, Clip._scale_bar (drawing, plant metres)",
                            "drawn top-right of each fleet view; common.draw_fleet's own bar is switched off because "
                            "it meets the payload in this framing")
        self.man.value("clock", "time of the drawn 10 ms state row", "s", f"replay:{caches} state_time",
                       "every playback frame; the 1 ms gap trace is drawn up to the same instant")

    # ---------------------------------------------------------------- captions (the narration)
    def _captions(self):
        """The narration: one or two complete sentences per caption, each symbol defined in words
        before it is used (W^c in the first caption, lambda in the second)."""
        n = self.n
        l9, l105, l15, one = n[0.9]["lam"], n[1.05]["lam"], n[1.5]["lam"], n["one"]
        # '1 cm' bound: the lam = 0.9 depth is below it
        assert self.rp[0.9]["record"]["main"]["max_depth"] < 0.01
        cm = self.reg("lam=0.9 slack depth bound", 1, "{:d}", "cm", run_src(0.9, "main.max_depth"),
                      "the recorded depth 9.1 mm is below 1 cm")
        play_end = N_TITLE + N_PRE + N_GUST + N_HOLD_CLOSURE + N_HOLD_PEAK
        stage2, stage3 = play_end + N_CHART1, play_end + N_CHART1 + N_CHART2
        end = stage3 + N_FINAL
        # boundaries in frames: at 2.5 words/s a word needs 12 frames, so each caption holds 12 x words
        cuts = [
            (N_TITLE, 414, "The gust's drag-conjugate load W$^c$ is the drift-speed difference it causes\n"
                           f"between vessel {n['vessel']} and the payload, times drag."),
            (414, 618, "The gust size λ is W$^c$ over the pretension;\n"
                       f"the theory predicts deepening slack above λ = {one}."),
            (618, 846, f"At λ = {l9} the slack stays under {cm} cm and soon ends,\n"
                       f"but above λ = {one} it keeps deepening."),
            (846, 1062, f"At λ = {l15}, cable {n['vessel']}'s two ends come within {n['chord_cl']} m,\n"
                        "and this formation closure ends the run."),
            (1062, play_end, f"After the gust, the λ = {l105} line snaps taut, closing at {n['vup105']} m/s,\n"
                             f"and its tension peaks at {n['Tp105']} kN."),
            (play_end, play_end + 180, f"Each point is one run, scored over the gust's last {n['late_len']} s, "
                                       "not at onset."),
            (play_end + 180, stage2, f"The fitted lines cross zero at λ = {n['pooled']} on average, "
                                     f"against the predicted {n['pred']}."),
            (stage2, stage3, "An inertial criterion, based on the acceleration at onset,\n"
                             f"had predicted the crossing at {n['in_lf']} or {n['in_fl']}."),
            (stage3, end, "Drag, not inertia, decides which gusts hold a line slack,\n"
                          "and per-line practice can adopt this one-cable criterion unchanged."),
        ]
        caps = [(a / FPS, b / FPS, text) for a, b, text in cuts]
        total = (N_TITLE + N_PRE + N_GUST + N_HOLD_CLOSURE + N_HOLD_PEAK + N_CHART1 + N_CHART2 + N_FINAL) / FPS
        assert abs(caps[-1][1] - total) < 1e-9 and caps[0][0] == N_TITLE / FPS
        for (a, b, text), nxt in zip(caps, caps[1:] + [(total, None, None)]):
            words = len([w for w in text.replace("\n", " ").split() if any(ch.isalnum() for ch in w)])
            self.man.check(f"caption [{a:.1f}, {b:.1f}) s: sentence form (capital to full stop), >= 4 s on screen, "
                           f"<= 2 lines, <= 2.5 words/s (numbers count as words), contiguous",
                           b - a >= 4.0 - 1e-9 and text.count("\n") <= 1 and words <= 2.5 * (b - a) + 1e-9
                           and abs(b - nxt[0]) < 1e-9 and text[0].isupper() and text.endswith("."),
                           f"{words} words, {words / (b - a):.2f} words/s")
        self.captions = caps
        # the narration must meet the events it names
        v_cl, v_up = video_of_sim(self.rp[1.5]["t_closure"]), video_of_sim(self.rp[1.05]["t_up"])
        (a_cl, b_cl, _), (a_up, b_up, _) = caps[3], caps[4]
        h_cl, h_pk = hold_span(K_HOLD_CLOSURE), hold_span(K_HOLD_PEAK)
        self.man.check("the closure caption starts within 0.5 s of the lam = 1.5 closure instant on screen and is up "
                       "through the whole hold on the first frame after it",
                       abs(v_cl - a_cl) <= 0.5 and a_cl <= h_cl[0] and h_cl[1] <= b_cl,
                       f"closure plays at video {v_cl:.2f} s, hold {h_cl[0]:.2f}-{h_cl[1]:.2f} s; "
                       f"caption {a_cl:.2f}-{b_cl:.2f} s")
        self.man.check("the snap caption is on screen when the lam = 1.05 re-engagement plays and through the whole "
                       "hold on its peak", a_up <= v_up < b_up and a_up <= h_pk[0] and h_pk[1] <= b_up,
                       f"re-engagement plays at video {v_up:.2f} s, hold {h_pk[0]:.2f}-{h_pk[1]:.2f} s; "
                       f"caption {a_up:.2f}-{b_up:.2f} s")
        self.man.check("chart captions sit inside their chart stage: two on the fits (stage 1), the inertial one on "
                       "stage 2 (dashed lines drawn), the takeaway on the final hold (stage 3)",
                       cuts[5][0] == play_end and cuts[6][1] == stage2 and cuts[7][:2] == (stage2, stage3)
                       and cuts[8][:2] == (stage3, end), f"stages from {play_end / FPS:.1f}, {stage2 / FPS:.1f}, "
                       f"{stage3 / FPS:.1f} s; end {end / FPS:.1f} s")

    def _holds(self):
        """The holds sit where the schedule says: K_HOLD_CLOSURE is the first gust frame drawn after the
        lam = 1.5 run's last state row (the closure notes appear there), K_HOLD_PEAK the gust frame that
        draws the lam = 1.05 10 ms row of cable 2's largest tension within one engagement period of t_up."""
        from tether.campaign.v2 import events as ev
        _, gust = sim_times()
        st0 = self.rp[LAMS[0]]["arrays"]["state_time"]
        drawn = np.array([st0[C.nearest(st0, t)] for t in gust])      # the clock of every gust frame
        k_cl = int(np.flatnonzero(drawn > self.rp[1.5]["arrays"]["state_time"][-1] + 1e-9)[0])
        a, t_up = self.rp[1.05]["arrays"], self.rp[1.05]["t_up"]
        rows = a["state_time"][(a["state_time"] >= t_up) & (a["state_time"] <= t_up + ev.ENGAGEMENT_PERIOD)]
        j = np.searchsorted(a["event_time"], rows - 1e-9)
        t2 = C.tension(a["elongation"][j], a["rate"][j], a["alive"][j])[:, TEST]
        row_pk = float(rows[int(np.argmax(t2))])
        k_pk = int(np.flatnonzero(np.abs(drawn - row_pk) < 1e-9)[0])
        self.man.check("playback holds (clock reads 'paused'; the held frame is redrawn unchanged): the first frame "
                       "after the lam = 1.5 closure, and the lam = 1.05 row of cable 2's largest tension after "
                       "re-engagement, which is the 1 ms peak instant",
                       k_cl == K_HOLD_CLOSURE and k_pk == K_HOLD_PEAK and drawn[k_cl] >= self.rp[1.5]["t_closure"]
                       and abs(row_pk - self.rp[1.05]["t_peak_time"]) < 1e-9
                       and abs(float(np.max(t2)) - self.rp[1.05]["record"]["main"]["T_peak"])
                       <= REL_TOL * self.rp[1.05]["record"]["main"]["T_peak"],
                       f"hold 1 on gust frame {k_cl} (t = {drawn[k_cl]:.3f} s) for {N_HOLD_CLOSURE / FPS:.1f} s; "
                       f"hold 2 on gust frame {k_pk} (t = {row_pk:.3f} s, cable {TEST} at {np.max(t2):.1f} N) "
                       f"for {N_HOLD_PEAK / FPS:.1f} s")

    def caption_at(self, v: float) -> str:
        for a, b, text in self.captions:
            if a - 1e-9 <= v < b - 1e-9:
                return text
        return ""

    # ---------------------------------------------------------------- camera
    def _camera(self):
        """Camera follows the payload (smoothed); one scale for all three runs."""
        from tether.physics import fleet as F
        box = (self.VIEW_W * C.W) / (self.AX_H * C.H)
        lo = np.array([np.inf, np.inf]); hi = -lo
        for lam in LAMS:
            a = self.rp[lam]["arrays"]
            st, S = a["state_time"], a["state"]
            load = C.smooth_camera(S[:, :2], window=31)
            self.rp[lam]["cam_load"] = load
            sel = np.flatnonzero((st >= T_PRE0 - 0.02) & (st <= T_GUST_END + 0.02))
            for k in sel:
                lp, vp = C.unpack_state(S[k])
                P = np.vstack([C.body_polygon(lp, F.pentagon_vertices())] + [C.body_polygon(v, C.hull()) for v in vp])
                rel_ = P - load[k]
                lo = np.minimum(lo, rel_.min(0)); hi = np.maximum(hi, rel_.max(0))
        margin = 1.2
        hy = max(abs(lo[1]), abs(hi[1])) + margin
        hx = hy * box
        self.half = np.array([hx, hy])
        self.dx = lo[0] - margin + hx      # payload's back edge at the left margin; room ahead for labels
        inside = (lo[0] >= self.dx - hx and hi[0] <= self.dx + hx and lo[1] >= -hy and hi[1] <= hy)
        self.man.check("camera: every body of every played state row lies inside the common view box",
                       bool(inside), f"extents relative to the smoothed payload x [{lo[0]:.2f}, {hi[0]:.2f}] m, "
                       f"y [{lo[1]:.2f}, {hi[1]:.2f}] m; box x [{self.dx - hx:.2f}, {self.dx + hx:.2f}] m, "
                       f"y [{-hy:.2f}, {hy:.2f}] m (one scale for all three runs)")

    # ---------------------------------------------------------------- common furniture
    def _furniture(self, title, sub, footer):
        C.title(self.fig, title, sub)
        C.footer(self.fig, footer)
        self.caption_artist = C.caption(self.fig, "", y=0.078)
        self.caption_artist.set_wrap(False)

    def set_caption(self, v):
        text = self.caption_at(v)
        self.caption_artist.set_text(text)
        self.caption_artist.set_visible(bool(text))

    # ---------------------------------------------------------------- title card
    def setup_title(self):
        fig = self.fig
        fig.clf()
        fig.text(0.5, 0.63, "The slack criterion: drag decides which gusts hold a line slack",
                 fontsize=C.FS_TITLE + 4, weight="bold", color=C.INK, ha="center", va="center")
        fig.text(0.5, 0.52, "Which gusts hold a towing line slack, and does drag or inertia decide?",
                 fontsize=C.FS_SUB + 2, color=C.INK, ha="center", va="center")
        fig.text(0.5, 0.455, "Watch the slack die out just below the predicted threshold "
                 "and keep deepening just above it.", fontsize=C.FS_SUB, color=C.INK, ha="center", va="center")
        fig.text(0.5, 0.37, "The three runs are scripted gust runs from the campaign's first phase, re-simulated "
                 "here and checked field by field against its record.", fontsize=C.FS_SMALL, color=C.MUTED,
                 ha="center", va="center")
        C.footer(fig, f"Replays of tether/campaign/v2/phase1_scripted.py run_gust_job · record: "
                      f"{rel(RUNS_PATH)} · corrected paper §VI")
        self.segment = "title"

    # ---------------------------------------------------------------- playback
    def setup_play(self):
        fig = self.fig
        fig.clf()
        n = self.n
        self._furniture("Three gust sizes, one fleet",
                        f"Parallel formation · T$_0$ = {n['T0']} kN · no background weather",
                        f"Replays: phase1_scripted.run_gust_job(GustJob('c', 1000, λ, 10)), identical to "
                        f"{rel(RUNS_PATH)} · recording runs (no cable severs) · camera follows the payload")
        self.clock_artist = C.clock(fig, T_PRE0, SPEED_PRE)
        fig.text(0.97, 0.905, r"$W^c = c_{\mathrm{eff}}\,(W_L/c_L - W_A/c_A)$,   $\lambda = W^c/T_0$",
                 fontsize=C.FS_SUB, color=C.INK, ha="right", va="top")
        fig.text(0.03, self.KEY_Y, f"Selection: λ = {n[0.9]['lam']} and {n[1.05]['lam']} are the sweep's "
                                   f"grid values either side of the predicted crossing;\nλ = {n[1.5]['lam']} is "
                                   f"well above it. Illustrative; every scored run is in the closing chart.",
                 fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="center", linespacing=1.3)
        C.cable_key(fig, y=self.KEY_Y, x=0.80)
        heads = {0.9: "just below", 1.05: "just above", 1.5: "well above"}
        self.views, self.eaxes, self.dyn = {}, {}, {}
        pitch = self.HEAD_H + self.AX_H + self.GAP
        for i, lam in enumerate(LAMS):
            top = self.ROW_TOP - i * pitch
            bottom = top - self.HEAD_H - self.AX_H
            fig.text(self.VIEW_X, top, f"λ = {n[lam]['lam']}   (W$^c$ = {n[lam]['Wc']} N, {heads[lam]} "
                                       f"λ = {n['one']})", fontsize=C.FS_BODY, color=C.INK, ha="left",
                     va="top", weight="bold")
            if i == 0:
                fig.text(self.E_X, top, f"gap e of cable {n['vessel']} = chord − rest length   "
                                        f"(shaded: e ≤ 0, slack)", fontsize=C.FS_SMALL, color=C.INK,
                         ha="left", va="top")
            av = fig.add_axes([self.VIEW_X, bottom, self.VIEW_W, self.AX_H])
            ae = fig.add_axes([self.E_X, bottom, self.E_W, self.AX_H])
            self.views[lam], self.eaxes[lam] = av, ae
            self._setup_e_axes(ae, lam, first=(i == 0), last=(i == len(LAMS) - 1))
        self.segment = "play"

    def _e_scale(self, lam):
        return (1e3, "mm", (-12.5, 10.5)) if lam == 0.9 else (1.0, "m", (-2.05, 0.45) if lam == 1.05 else (-12.5, 1.5))

    def _setup_e_axes(self, ae, lam, first, last):
        n = self.n
        s, unit, ylim = self._e_scale(lam)
        ae.set_xlim(T_PRE0, T_GUST_END)
        ae.set_ylim(*ylim)
        ae.axvspan(10.0, 20.0, color=C.ACCENT, alpha=GUST_ALPHA, lw=0, zorder=0)
        ae.axvspan(18.0, 20.0, color=C.ACCENT, alpha=SCORED_ALPHA, lw=0, zorder=0)
        ae.axhline(0.0, color=C.MUTED, lw=0.9, zorder=1)
        ae.set_ylabel(f"e [{unit}]", fontsize=C.FS_SMALL)
        ae.tick_params(labelsize=C.FS_TINY)
        ae.set_xticks(np.arange(6, 23, 2))
        ae.yaxis.set_major_locator(MaxNLocator(4))
        ae.text(1.0, -0.035, "t [s]", transform=ae.transAxes, fontsize=C.FS_TINY, color=C.MUTED,
                ha="right", va="top", gid="tick-row")
        if first:
            ae.text(10.15, ylim[1] - 0.06 * (ylim[1] - ylim[0]),
                    f"gust on ({n['t_gust']}–{n['t_off']} s)", fontsize=C.FS_TINY, color=C.INK,
                    ha="left", va="top", path_effects=halo(BG_GUST))
            ae.text(19.0, ylim[1] - 0.06 * (ylim[1] - ylim[0]), f"scored\nlast {n['late_len']} s",
                    fontsize=C.FS_TINY, color=C.INK, ha="center", va="top", path_effects=halo(BG_SCORED))
        a = self.rp[lam]["arrays"]
        t = a["event_time"]
        sel = t >= T_PRE0 - 1e-9
        self.rp[lam]["t_draw"] = t[sel]
        self.rp[lam]["e_draw"] = a["elongation"][sel, TEST] * s
        line, = ae.plot([], [], color=C.INK, lw=1.5, zorder=3)
        cursor = ae.axvline(T_PRE0, color=C.MUTED, lw=0.9, zorder=2)
        dot, = ae.plot([], [], "o", ms=5, color=C.INK, zorder=4)
        self.dyn[lam] = {"line": line, "cursor": cursor, "dot": dot, "fill": None, "notes": []}

    def _e_notes(self, lam, ts):
        """Event annotations, each shown once its instant has been drawn."""
        ae, d, n, rp = self.eaxes[lam], self.dyn[lam], self.n, self.rp[lam]
        for artist in d["notes"]:
            artist.remove()
        d["notes"] = []
        s, unit, ylim = self._e_scale(lam)
        if lam == 0.9 and ts >= n["t_deep9"]:
            y = -rp["record"]["main"]["max_depth"] * s
            d["notes"].append(ae.annotate(minus(f"deepest -{n['dep9_mm']} mm"), xy=(n["t_deep9"], y),
                                          xytext=(12.2, y), fontsize=C.FS_TINY, color=C.INK, va="center",
                                          path_effects=halo(BG_GUST),
                                          arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.8)))
        if lam == 1.05 and ts >= n["t_deep105"]:
            y = -rp["record"]["main"]["max_depth"]
            d["notes"].append(ae.annotate(minus(f"deepest -{n['dep105']} m"), xy=(n["t_deep105"], y),
                                          xytext=(20.7, y - 0.12), fontsize=C.FS_TINY, color=C.INK, va="top",
                                          path_effects=halo(),
                                          arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.8)))
        if lam == 1.05 and ts >= rp["t_up"]:
            d["notes"].append(ae.annotate(f"re-engages at t = {n['tup105']} s, {n['vup105']} m/s: "
                                          f"peak {n['Tp105']} kN", xy=(rp["t_up"], 0.0), xytext=(21.35, 0.25),
                                          fontsize=C.FS_TINY, color=C.INK, ha="right", va="center",
                                          arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.8)))
        if lam == 1.5 and ts >= rp["t_closure"] - 1e-9:
            y = rp["record"]["main"]["min_elongation_test_cable"]
            d["notes"].append(ae.annotate(f"run ended at\nt = {n['tcl15']} s:\nformation closure",
                                          xy=(rp["t_closure"], y),
                                          xytext=(24.35, -6.0), fontsize=C.FS_TINY, color=C.INK,
                                          ha="right", va="center", path_effects=halo(),
                                          arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.8)))

    def draw_play(self, t_frame: float, speed: str, v: float):
        """One playback frame: the shared clock is the time of the drawn 10 ms state row."""
        st0 = self.rp[LAMS[0]]["arrays"]["state_time"]
        ts = float(st0[C.nearest(st0, t_frame)])
        for lam in LAMS:
            rp, a = self.rp[lam], self.rp[lam]["arrays"]
            st, et = a["state_time"], a["event_time"]
            ended = ts > st[-1] + 1e-9            # only lam = 1.5 (closure at 18.883 s) runs out
            if ended:
                assert lam == 1.5 and ts >= rp["t_closure"]
                k = len(st) - 1
            else:
                k = C.nearest(st, ts)
                assert abs(st[k] - ts) < 1e-9, "the three replays share the 10 ms state grid"
            j = int(np.searchsorted(et, st[k] - 1e-9))
            assert abs(et[j] - st[k]) < 1e-9, "state row and 1 ms sample out of step"
            T_row = C.tension(a["elongation"][j], a["rate"][j], a["alive"][j])
            av = self.views[lam]
            centre = rp["cam_load"][k] + np.array([self.dx, 0.0])
            C.draw_fleet(av, a["state"][k], self.geometry, T_row, a["alive"][j], centre=centre, half=self.half,
                         labels=False, scale_bar=False, check_inside=True)
            self._scale_bar(av, centre)
            self._view_overlays(av, lam, a, k, ts, ended)
            # gap trace up to the drawn instant (1 ms log); an ended run is shown to its last sample
            j_trace = len(et) - 1 if ended else j
            d = self.dyn[lam]
            upto = rp["t_draw"] <= et[j_trace] + 1e-9
            tt, ee = rp["t_draw"][upto], rp["e_draw"][upto]
            d["line"].set_data(tt, ee)
            d["cursor"].set_xdata([ts, ts])
            d["cursor"].set_visible(not ended)
            d["dot"].set_data([tt[-1]], [ee[-1]])
            if d["fill"] is not None:
                d["fill"].remove()
            d["fill"] = self.eaxes[lam].fill_between(tt, ee, 0.0, where=ee <= 0.0, color=C.SLACK, alpha=0.20,
                                                     lw=0, zorder=2)
            self._e_notes(lam, float(tt[-1]))
        self.clock_artist.set_text(f"t = {ts:7.3f} s   ({speed})")
        self.set_caption(v)
        return ts

    def _scale_bar(self, av, centre):
        """5 m bar in the view's top-right corner (common's bottom-left bar meets the payload here)."""
        hx, hy = self.half
        x1 = centre[0] + hx - 0.03 * 2 * hx
        y = centre[1] + hy - 0.9
        av.plot([x1 - 5.0, x1], [y, y], color=C.INK, lw=2.5, solid_capstyle="butt", zorder=6)
        av.text(x1 - 2.5, y - 0.25, f"{self.n['scale']} m", ha="center", va="top", fontsize=C.FS_TINY,
                color=C.INK, zorder=6)

    def _view_overlays(self, av, lam, a, k, clock, ended):
        """Labels and the gust arrow.  ``clock`` is the shared playback clock; for a run that has not
        ended it equals the drawn state row's time (asserted in draw_play).  The gust arrow shows the
        declared force at the clock time, so the frozen last state of an ended run keeps its arrow
        only while the gust would still be on (10-20 s) and loses it once the clock passes 20 s."""
        n = self.n
        lp, vp = C.unpack_state(a["state"][k])
        # payload label
        av.text(lp[0] - 0.4, lp[1], "payload", fontsize=C.FS_TINY, color=C.MUTED, ha="center", va="center", zorder=6,
                gid="inside-ok")
        bow = C.body_polygon(vp[TEST], np.array([[0.5 * C.L_HULL, 0.0]]))[0]
        # labels start right of every other bow, so they never sit on a hull when vessel 2 drifts back
        clear_x = max(C.body_polygon(vp[i], C.hull())[:, 0].max() for i in range(len(vp)) if i != TEST) + 0.8
        row = int(round(clock / float(a["weather_period"])))
        force = float(a["gust_force_x_vessel2"][min(row, len(a["gust_force_x_vessel2"]) - 1)])
        if force != 0.0:
            length = abs(force) / 1000.0 * 4.0
            tail = bow + np.array([0.5 + length, 0.0])
            av.annotate("", xy=bow + np.array([0.4, 0.0]), xytext=tail,
                        arrowprops=dict(arrowstyle="-|>", color=C.ACCENT, lw=3.0, mutation_scale=18), zorder=7)
            if not ended:
                av.text(max(tail[0] + 0.5, clear_x), tail[1], f"gust on vessel {n['vessel']}",
                        fontsize=C.FS_SMALL, color=C.INK, ha="left", va="center", zorder=7)
        elif not ended:
            av.text(max(bow[0] + 0.8, clear_x), bow[1], f"vessel {n['vessel']}", fontsize=C.FS_SMALL,
                    color=C.INK, ha="left", va="center", zorder=7)
        if lam == LAMS[0]:
            av.text(0.99, 0.04, "hulls drawn to scale; the plant models no hull contact", transform=av.transAxes,
                    fontsize=C.FS_TINY, color=C.MUTED, ha="right", va="bottom", zorder=7)
        if lam == 1.05 and clock >= n["t_coll105"]:
            av.text(0.99, 0.04, f"after the snap, the other\nfour lines briefly slack\n({n['coll105']} marks "
                                f"in the record)", transform=av.transAxes, fontsize=C.FS_TINY, color=C.INK,
                    ha="right", va="bottom", zorder=7, linespacing=1.15)
        if ended:
            av.text(0.985, 0.5, f"run ended at t = {n['tcl15']} s:\nformation closure, chord\n"
                                f"under {n['chord_cl']} m (last state shown)",
                    transform=av.transAxes, fontsize=C.FS_SMALL, color=C.INK, ha="right", va="center",
                    zorder=10, linespacing=1.25, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C.FAINT))

    # ---------------------------------------------------------------- the P1-T6 chart
    def setup_chart(self, stage: int):
        fig = self.fig
        fig.clf()
        n, per = self.n, self.t6["per"]
        self._furniture("Where sustained drift begins (test P1-T6)",
                        f"Every scored t$_x$ = {n['t_x']} s run at three pretensions · drift function over the "
                        f"gust's last {n['late_len']} s",
                        f"Sources: {rel(RESULTS_PATH)} P1-T6 · {rel(GATE_PATH)} · "
                        f"{rel(DECL_PATH)} · left: the three replays")
        fig.text(0.97, 0.955, "record summary (no playback)", fontsize=C.FS_SUB, color=C.MUTED,
                 ha="right", va="top")
        # left: the three replayed gaps with their scored window
        for i, lam in enumerate(LAMS):
            rp = self.rp[lam]
            y0 = 0.655 - i * 0.205
            ax = fig.add_axes([0.045, y0, 0.28, 0.135])
            s, unit, ylim = self._e_scale(lam)
            t, e = rp["t_draw"], rp["e_draw"]
            ax.axvspan(10.0, 20.0, color=C.ACCENT, alpha=0.08, lw=0)
            ax.axvspan(18.0, 20.0, color=C.ACCENT, alpha=0.20, lw=0)
            ax.axhline(0.0, color=C.MUTED, lw=0.8)
            ax.fill_between(t, e, 0.0, where=e <= 0.0, color=C.SLACK, alpha=0.20, lw=0)
            ax.plot(t, e, color=C.INK, lw=1.2)
            ax.set_xlim(T_PRE0, T_GUST_END); ax.set_ylim(*ylim)
            ax.set_ylabel(f"e [{unit}]", fontsize=C.FS_TINY)
            ax.tick_params(labelsize=C.FS_TINY)
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.set_xticks(np.arange(6, 25, 4))
            if i == len(LAMS) - 1:
                ax.set_xlabel("t [s]", fontsize=C.FS_TINY, labelpad=1)
            extra = (f"  (partial: {n['samples15']} of {n['samples_full']} samples)" if lam == 1.5 else "")
            fig.text(0.045, y0 + 0.135 + 0.008, minus(f"λ = {n[lam]['lam']}:  y = {n[lam]['y']}") + extra,
                     fontsize=C.FS_SMALL, color=C.INK, ha="left", va="bottom")
        # right: y(lambda) with the three linear fits
        ax = fig.add_axes([0.43, 0.255, 0.54, 0.535])
        self.chart_ax = ax
        # the three pretensions' points nearly coincide at every lambda (y depends on lambda, hardly on
        # T0), so they are drawn concentric at their true positions: large open, medium filled, small
        # filled on top; no horizontal offset.
        styles = {"600": ("#4f9ac4", "v", dict(ms=15, mfc="none", mec="#4f9ac4", mew=2.0), 3.6),
                  "1000": (C.TAUT, "o", dict(ms=9.5, mfc=C.TAUT, mec=C.INK, mew=0.7), 4.0),
                  "1400": ("#1b2f45", "^", dict(ms=5.5, mfc="#1b2f45", mec="white", mew=0.8), 4.4)}
        ax.axhline(0.0, color=C.MUTED, lw=0.9, zorder=1)
        ax.axvline(self.t6["predicted"], color=C.INK, lw=1.4, zorder=1)
        ax.text(self.t6["predicted"] - 0.008, 0.455, f"predicted\n{n['pred']}", fontsize=C.FS_SMALL,
                color=C.INK, ha="right", va="top")
        for key in ("600", "1000", "1400"):
            col, mk, mstyle, z = styles[key]
            p = per[key]
            xs = np.linspace(p["lam"].min() - 0.03, p["lam"].max() + 0.03, 50)
            ax.plot(xs, np.polyval(p["fit"], xs), color=col, lw=1.3, zorder=2)
            ax.plot(p["lam"], p["y"], mk, ls="none", zorder=z, label=f"T$_0$ = {n['T0s'][key]} kN", **mstyle)
            ax.plot([p["crossing"]], [0.0], "|", ms=16, mew=2.2, color=col, zorder=5)
            xe = p["lam"].max() + 0.035
            ax.text(xe, np.polyval(p["fit"], xe), f"{n['T0s'][key]} kN", fontsize=C.FS_TINY, color=C.INK,
                    ha="left", va="center")
        rl = np.array(LAMS)
        ry = np.array([self.rp[l]["drift"] for l in LAMS])
        ax.plot(rl, ry, "o", ms=17, mfc="none", mec=C.ACCENT, mew=2.0, zorder=3, label="replayed in this clip")
        ax.annotate(f"λ = {n[1.5]['lam']}: partial window\n(run ended at {n['tcl15']} s)",
                    xy=(1.5, self.rp[1.5]["drift"]), xytext=(1.58, -0.17), fontsize=C.FS_TINY, color=C.INK,
                    ha="right", va="center", arrowprops=dict(arrowstyle="-", color=C.MUTED, lw=0.8))
        ax.set_xlim(0.5, 1.62)
        ax.set_ylim(-0.62, 0.47)
        ax.set_xlabel(r"$\lambda = W^c/T_0$  (units of the predicted crossing)", fontsize=C.FS_SMALL)
        ax.set_ylabel(r"drift function  $y = \langle T + c_{\mathrm{eff},f}\,\dot e\,\rangle / T_0$  [–]",
                      fontsize=C.FS_SMALL)
        ax.tick_params(labelsize=C.FS_TINY)
        ax.grid(True, lw=0.5)
        ax.text(0.515, 0.02, "y > 0: line taut (every run shown)", fontsize=C.FS_TINY, color=C.MUTED,
                ha="left", va="bottom")
        ax.text(1.60, -0.025, "y < 0: slack, still deepening (every run shown)", fontsize=C.FS_TINY,
                color=C.MUTED, ha="right", va="top")
        box = (f"fitted crossing (zero of the linear fit)\n"
               f"T$_0$ = {n['T0s']['600']} kN:  {n['cross']['600']}\n"
               f"T$_0$ = {n['T0s']['1000']} kN:  {n['cross']['1000']}\n"
               f"T$_0$ = {n['T0s']['1400']} kN:  {n['cross']['1400']}\n"
               f"pooled (mean):  {n['pooled']}   predicted:  {n['pred']}\n"
               f"declared band: within {n['band']} % of {n['one']}  →  {self.t6['verdict']}")
        ax.text(0.985, 0.975, box, transform=ax.transAxes, fontsize=C.FS_SMALL, color=C.INK, ha="right",
                va="top", linespacing=1.35, bbox=dict(boxstyle="round,pad=0.5", fc="#f3f4f6", ec="none"))
        ax.legend(loc="lower left", fontsize=C.FS_TINY, handletextpad=0.4, borderaxespad=0.6)
        fig.text(0.045, 0.162, minus(
            f"Declared: a run ended by formation closure before the gust's end is censored "
            f"(λ = {n['cens20']} at every T$_0$; λ = {n['cens15']} at {n['T0s']['1400']} kN).\n"
            f"The recorded 1 kN fit still keeps λ = {n[1.5]['lam']} (closure at {n['tcl15']} s, {n['samples15']} "
            f"of {n['samples_full']} samples). Post hoc, without it: {n['ph1000']} at 1 kN, pooled "
            f"{n['phpooled']}, still within {n['band']} %.\n"
            f"Measured crossings are plant results; "
            + (f"{n['pred']} and the inertial values are predictions." if stage >= 2
               else f"{n['pred']} is the prediction.")),
                 fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="center", linespacing=1.25)
        if stage >= 2:
            for x in self.t6["inertial"]:
                ax.axvline(x, color=C.MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
            ax.text(self.t6["inertial"][1] + 0.01, -0.60,
                    f"inertial (onset-acceleration)\nform, committed:\n{n['in_lf']} (m$_{{Lf}}$) / "
                    f"{n['in_fl']} (m$_{{fleet}}$)", fontsize=C.FS_TINY, color=C.INK, ha="left", va="bottom")
        if stage >= 3:
            fig.text(0.70, 0.835, f"pooled measured crossing λ = {n['pooled']}   vs   predicted {n['pred']}",
                     fontsize=C.FS_SUB, weight="bold", color=C.INK, ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=C.ACCENT, lw=2.0))
        self.segment = f"chart{stage}"

    # ---------------------------------------------------------------- driver
    def frames(self):
        """Yield (segment, video second, payload) for every frame, in order."""
        pre, gust = sim_times()
        i = 0
        for _ in range(N_TITLE):
            yield "title", i / FPS, None; i += 1
        for t in pre:
            yield "play", i / FPS, (float(t), SPEED_PRE); i += 1
        holds = {K_HOLD_CLOSURE: N_HOLD_CLOSURE, K_HOLD_PEAK: N_HOLD_PEAK}
        for k, t in enumerate(gust):
            yield "play", i / FPS, (float(t), SPEED_GUST); i += 1
            for _ in range(holds.get(k, 0)):                      # the same instant, redrawn, clock 'paused'
                yield "play", i / FPS, (float(t), SPEED_HOLD); i += 1
        for _ in range(N_CHART1):
            yield "chart1", i / FPS, None; i += 1
        for _ in range(N_CHART2):
            yield "chart2", i / FPS, None; i += 1
        for _ in range(N_FINAL):
            yield "chart3", i / FPS, None; i += 1

    def view_list(self):
        return list(self.views.values()) if self.segment == "play" else []

    def render_frame(self, seg, v, payload):
        if seg == "title":
            if self.segment != "title":
                self.setup_title()
            return None
        if seg == "play":
            if self.segment != "play":
                self.setup_play()
            return self.draw_play(payload[0], payload[1], v)
        stage = int(seg[-1])
        if self.segment != seg:
            self.setup_chart(stage)
        self.set_caption(v)
        return None


# ------------------------------------------------------------------------------------ main

STORY = ("The drag-conjugate slack criterion is the one genuinely new piece of theory the campaign confirms "
         "(paper Section VI); it involves one cable only, so per-line practice can adopt it unchanged. "
         "Three replayed scripted gusts on one vessel of the parallel fleet (T0 = 1 kN, 10 s square gust) show "
         "what it separates: just below the predicted crossing (lambda = 0.9) the line's slack stays under 1 cm; "
         "just above it (1.05) the slack keeps deepening for as long as the gust acts and then closes with a "
         "14.9 kN snap; well above it (1.5) vessel 2 drifts back until its chord is under 1 m and formation closure "
         "ends the run before the gust does. "
         "Scored on the sustained drift over the gust's last 2 s (not on onset), the measured crossing pooled over "
         "three pretensions is 0.989 against the predicted 1.000; the inertial (onset-acceleration) form had been "
         "committed at 1.055 / 1.077. The recorded 1 kN fit keeps one run (lambda = 1.5, partial window) that the "
         "declared censoring rule excludes; post hoc, without it the pooled crossing is 0.987, still inside the "
         "declared 5 % band.")

CAVEATS = [
    "Deterministic scripted runs (one run per cell, no seeds): every number is a point value, as in the record.",
    "The gust is a declared square body force on vessel 2 only (F = 1.06364 W^c along world -x), delivered through "
    "the weather channel; there is no background weather. Real gusts act on every body.",
    "All three lines go slack at gust onset, including lambda = 0.9 (a 9.1 mm, 0.56 s excursion, then two short "
    "bounces, all over by 11.6 s in the replay); P1-T6 scores sustained drift over [t_off - 2 s, t_off], not onset.",
    "lambda = 1.5 at T0 = 1 kN has no re-engagement: formation closure (a chord under 1 m, B.7) terminated the run at "
    "18.883 s, before the gust ended. Its late window is partial (884 of 2001 1 ms samples, 18.0-18.883 s). The "
    "declaration censors such runs (scripted_declarations.json tests.P1-T6.readings: 'A t_x = 10 s run terminated by "
    "formation closure (B.7) before the shut-off holds no late window; it is censored out of the fit'; "
    "closure_censoring: such cells 'contribute to no test population'), but score_t6 keeps any run with late-window "
    "samples > 0, so the recorded 1 kN fit (0.996) and the pooled crossing (0.989) include it. It is the only such "
    "run in any fit. The chart keeps the recorded fits and says this on screen, with a POST-HOC re-fit that is in no "
    "record: without the run, 1 kN 0.989 and pooled 0.987, every crossing still within 5 % of 1, so the verdict is "
    "unchanged. The paper's Section VI quotes 0.996 and 'each inside the pre-declared 5% band' from the recorded fit.",
    "The inertial crossings 1.055 (m_Lf) / 1.077 (m_fleet) are committed predictions; no onset-acceleration crossing "
    "was measured. 1.000 is the prediction; the per-T0 crossings 0.989 / 0.996 / 0.984 (T0 = 0.6 / 1.0 / 1.4 kN) and "
    "their mean 0.989 are measured plant crossings.",
    "Hull outlines overlap slightly in the drawing: the parallel formation's vessels sit 0.89-1.09 m apart laterally "
    "at t = 5.5 s (0.65-1.16 m over the played window, all three runs) and the plant-size hull is 1.0 m wide; the plant models no hull contact (tether/physics/fleet.py: 'No contact is "
    "modelled'), so this is the plant's geometry, to scale. The top fleet view says so on screen.",
    "The frozen lambda = 1.5 view (after its 18.883 s closure) shows the last state; its gust arrow follows the "
    "declared forcing at the playback clock, so it disappears when the clock passes the 20 s gust end.",
    "The runs are recording runs: no cable severs, so the 14.9 kN re-engagement peak is scored, not a severance.",
    "The crossing that is identical across T0 in scripted_results.json (0.98865) is the committed ODE fit, not a "
    "plant result, and is not shown.",
    "The playback clock is the time of the drawn 10 ms state row; the gap traces use the 1 ms cable log up to the "
    "same instant. Each gap panel has its own vertical scale (mm for lambda = 0.9, m otherwise).",
    "The slow-motion playback pauses twice for reading time, with the clock reading 'paused': 1.6 s on the first "
    "frame after the lambda = 1.5 closure (t = 18.900 s) and 2.1 s on the lambda = 1.05 row of peak tension "
    "(t = 21.720 s). A held frame is the same plant state redrawn; no instant is added or skipped.",
    "After the lambda = 1.05 snap the other four cables go briefly slack (12 collateral marks in the record); the clip "
    "notes it but makes no claim about the transmission mechanism here.",
    "tether/campaign/v2/phase1_scripted.py's current sha256 differs from the module_sha256 stored in "
    "scripted_runs.json; reproduction is established on the outputs (bit-identical run records), not on the file hash.",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replay", action="store_true", help="re-simulate even if the caches exist")
    ap.add_argument("--stills", type=str, default="", help="comma-separated video seconds: save PNG stills "
                    "to --stills-dir instead of rendering the clip")
    ap.add_argument("--stills-dir", type=str, default="")
    ap.add_argument("--keep-going", action="store_true", help="stills: report layout problems, save anyway")
    ap.add_argument("--dry-check", type=int, default=0, help="run the layout checks on every N-th frame; "
                    "no video, no files")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    man = C.Manifest(NAME, "The slack criterion: drag decides which gusts hold a line slack", STORY)
    for p in (RUNS_PATH, RESULTS_PATH, GATE_PATH, DECL_PATH, CAMPAIGN_MODULE):
        man.source(p)
    replays = get_replays(man, force=args.replay)
    t6 = t6_data(man, replays)
    man.selection = ("Storyboard C8: T0 = 1 kN, t_x = 10 s (the P1-T6 duration); lambda = 0.9 and 1.05, the sweep's "
                     "grid values either side of the predicted crossing lambda = 1, and lambda = 1.5, well above it "
                     "(the largest lambda at 1 kN still in the scored fit; 2.0 is censored). Illustrative, not "
                     "typical-by-construction; the closing chart shows every scored point at all three pretensions.")
    man.caveats = CAVEATS
    clip = Clip(man, replays, t6)
    man.check("the three replays share one plant spec (one config hash; the gust enters through the weather "
              "array, not the spec). scripted_runs.json stores no config hash, so this is not a comparison with "
              "the record; reproduction rests on the run-record checks above",
              len({replays[l]["meta"]["config_hash"] for l in LAMS}) == 1, replays[0.9]["meta"]["config_hash"])

    if args.dry_check:
        n_checked = 0
        for i, (seg, v, payload) in enumerate(clip.frames()):
            if i % args.dry_check == 0:
                clip.render_frame(seg, v, payload)
                check_layout(clip.fig, f"frame {i} (v = {v:.2f} s, {seg})", clip.view_list())
                n_checked += 1
        print(f"layout clean on {n_checked} frames (every {args.dry_check}th)")
        return 0

    if args.stills:
        out = Path(args.stills_dir) if args.stills_dir else None
        assert out is not None, "--stills-dir is required with --stills"
        out.mkdir(parents=True, exist_ok=True)
        wanted = sorted(float(s) for s in args.stills.split(","))
        todo = list(wanted)
        for i, (seg, v, payload) in enumerate(clip.frames()):
            if todo and v >= todo[0] - 1e-9:
                clip.render_frame(seg, v, payload)
                try:
                    check_layout(clip.fig, f"v = {v:.2f} s ({seg})", clip.view_list())
                except AssertionError as exc:
                    if not args.keep_going:
                        raise
                    print("LAYOUT:", exc)
                clip.fig.savefig(out / f"still_{v:05.2f}.png", dpi=C.DPI)
                print("still", f"{v:.2f}", seg, flush=True)
                todo.pop(0)
            if not todo:
                break
        return 0

    path = C.CLIPS / f"{NAME}.mp4"
    w = C.writer()
    frames = 0
    key_times = {round(video_of_sim(x) * FPS) for x in (replays[1.5]["t_closure"], replays[1.05]["t_up"],
                                                        replays[1.05]["t_up"] + 0.1)}
    key_times |= {round(s * FPS) for k in (K_HOLD_CLOSURE, K_HOLD_PEAK) for s in hold_span(k)}
    with w.saving(clip.fig, str(path), dpi=C.DPI):
        last_seg = None
        for i, (seg, v, payload) in enumerate(clip.frames()):
            clip.render_frame(seg, v, payload)
            if seg != last_seg or i % (15 if seg == "play" else 45) == 0 or i in key_times:
                check_layout(clip.fig, f"frame {i} (v = {v:.2f} s, {seg})", clip.view_list())
            last_seg = seg
            w.grab_frame()
            frames += 1
            if i % 150 == 0:
                print(f"frame {i}  v = {v:5.2f} s  {seg}", flush=True)
    expected = N_TITLE + N_PRE + N_GUST + N_HOLD_CLOSURE + N_HOLD_PEAK + N_CHART1 + N_CHART2 + N_FINAL
    man.check("frames written equal the schedule", frames == expected, f"{frames} frames = {frames / FPS:.2f} s")
    man.frames = frames
    write_manifest(man, clip, path)
    return 0


def write_manifest(man: C.Manifest, clip: "Clip", path: Path) -> None:
    """The captions are drawn by set_text on one artist (common.caption is created empty, so
    common.CAPTION_LOG stays empty during the render); log the schedule here so the manifest's
    ``captions`` (read by assemble.py for SCRIPT.md) carries the narration, start frame by start frame."""
    C.CAPTION_LOG[:] = [(int(round(a * FPS)), plain(text)) for a, b, text in clip.captions]
    man.check("manifest captions = the on-screen caption schedule (one entry per caption, start time, text "
              "without mathtext)", len(C.CAPTION_LOG) == len(clip.captions) > 0
              and all(abs(f / FPS - a) < 1e-9 for (f, _), (a, _b, _t) in zip(C.CAPTION_LOG, clip.captions)),
              f"{len(C.CAPTION_LOG)} captions")
    man.value("caption schedule", [[a, b, plain(t)] for a, b, t in clip.captions], "s (video)", "this module",
              "narration: start, end, text")
    man.write(path)


if __name__ == "__main__":
    sys.exit(main())
