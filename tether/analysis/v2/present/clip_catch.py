"""C6 ``catch`` -- mitigation 2: the velocity-matching catch (the P6-T0 counterfactual).

Run:  python3 -m tether.analysis.v2.present.clip_catch
      (``--stills 12.5,30 --stills-dir DIR`` renders only those instants, in seconds of clip
      time, as PNG stills for layout inspection; nothing but those stills and the replay caches is
      written in that mode.)

Outputs: Presentation/clips/catch.mp4, Presentation/manifests/catch.json and the replay caches
Presentation/cache/catch_5020_4.npz and Presentation/cache/catch_5028_0.npz.

What the viewer should learn
----------------------------
A catch that brakes the slack tug so that it meets the payload slowly was tested as a
counterfactual on the recorded squall missions (fan formation).  From the recorded turnaround of
a dangerous slack excursion the campaign's own integrator re-runs the whole fleet three ways: the
recorded thrust replayed (no catch; it reproduces the record), and the catch in its thrust-hold
(F >= 0) and thrust-reverse (F >= -F_T) variants, fed the TRUE chord state at 10 Hz.  On the
typical failure all three re-engage above the severing speed v_b, and the catch's first tension
peak there is higher than the replay's, not lower (the record's T_peak_cf, reported, gates
nothing); on the one excursion it saves the tug lands at 0.89 v_b.  Over the 16 first-severance
excursions it catches 1 of 16 in both variants against a pre-declared bar of 0.80; why it fails
is not established.  Post hoc readings are labelled post hoc on screen.

Records read (their sha256 go into the manifest's ``sources``)
--------------------------------------------------------------
records/v2/phase6/p6_t0_results.json
    critical_speeds                                  v_b per cable (asserted = recomputed v_b)
    population_sizes.first_severance, verdict.n      the verdict population size (16)
    excursions[j].tags                               seed, cable, t_up, first_severance,
                                                     critical_speed, v_return, T_peak, depth,
                                                     t_turnaround, start_label, window
                                                     (asserted equal to the rebuilt tags)
    excursions[j].runs[mode='fleet', controller='replay', variant=None]
    excursions[j].runs[mode='fleet', controller='true10', variant in {'hold', 'reverse'}]
        .outcome, .t_cf, .v_cf, .ratio, .caught, .T_peak_cf, .delay, .engaged_fraction,
        .min_thrust                                  (asserted against the replays; ratios,
                                                     caught flags, T_peak_cf and min_thrust of
                                                     all 16 feed the tally)
    tables.fleet.true10.first_severance.{hold,reverse}.{n, caught, interval95, median_ratio}
    tables.fleet.{P,B2,L}.first_severance.{hold,reverse}.caught   estimator-fed arms (1-4 of 16)
    verdict.{fractions, intervals95, n, admissible, outcome}
    validation.fleet.{passed, abs_dv_m_s.max}
records/v2/phase6/p6_t0_declarations.json
    admissibility_rule (the 0.80 bar), constants.T_b_s_N (4500 N), constants.K_v_N_s_m,
    constants.a_c_m_s2, constants.v_soft, constants.F_min_variants, constants.cable.rest_length_m,
    controllers.true10 (asserted to say 10 Hz)
records/phase5/cache/phase5_missions.pkl
    the mission dicts of seeds 5020 and 5028 (seed, marks, truth = PlantTruth(state_time, state,
    event_time, elongation, rate), max_tension, outcome incl. outcome.severances), passed to the
    campaign's own excursion builder ``tether.campaign.v2.p6_t0.build_excursions``; truth.state
    (10 ms) and truth.elongation / truth.rate (1 ms) at the turnaround label for the top-down frame
records/phase5/impact_table_fan.json
    v_b = tether.monitor.hazard.critical_speeds(4500, 1000, table_path=...), as p6_t0.run does

Code called (not modified): tether.campaign.v2.p6_t0 (build_excursions, integrate, _control_tick,
_monitor_tick), tether.control.catch (catch_thrust, VARIANTS), tether.monitor.hazard.critical_speeds,
tether.campaign.mission.mission_schedules, and the shared renderer common.py.

Selection rule (stated on screen and in the manifest)
-----------------------------------------------------
Population: the 16 first-severance excursions of P6-T0 (each mission's first re-engagement above
4.5 kN), fleet mode, controller true10 (the verdict table).
  * the caught excursion: the only one caught under both variants -- seed 5028, cable 0;
  * the typical failure: the lower median of the 16 thrust-hold ratios v/v_b (8th of 16 sorted),
    seed 5020 cable 4 (2.575 v_b).  The median 2.863 is the midpoint of 2.575 and 3.150, so the
    two middle excursions are equidistant from it; the lower one is taken.

Asserted before any frame is drawn (each one a Manifest.check)
-------------------------------------------------------------
* population size 16 in the excursions, population_sizes and verdict; the selection rule picks
  (5028, 0) and (5020, 4); the thrust-hold median equals the table's median_ratio;
* the recomputed v_b equal the record's critical_speeds; the rebuilt excursion tags equal the
  record's tags;
* for each of the two excursions and each of (replay), (true10, hold), (true10, reverse):
  ``p6_t0.integrate(x, 'fleet', controller, VARIANTS[variant], trace=True)`` reproduces the
  record's outcome and caught flag exactly and t_cf, v_cf, ratio, T_peak_cf, delay,
  engaged_fraction and min_thrust to 1e-9 relative;
* the replay's traced e(t), edot(t) equal the recorded 1 ms log of the mission up to the recorded
  t_up (tolerances 1e-4 m, 1e-3 m/s); each law run coincides with the replay before its first
  braking tick;
* the thrust command drawn is reconstructed from the traced e, edot with the campaign's own tick
  functions (p6_t0._control_tick, _monitor_tick) and law (tether.control.catch.catch_thrust); its
  engaged-tick fraction and minimum equal the record's engaged_fraction and min_thrust;
* the top-down frame's state row is the excursion's own start row, the watched cable is slack
  there, and the run is a recording run (no cable cut); which drawn bodies overlap (a
  separating-axis test; the plant models no body contact) -- the on-screen note appears exactly
  where they do (5028/0) and not where they do not (5020/4);
* 5020/4: the catch's first tension peak T_peak_cf exceeds the replay's in both variants (the
  caption that says so);
* tally: the caught counts and fractions equal the verdict and tables, the Clopper-Pearson upper
  end, and the estimator-fed arms span 1-4 of 16; under each variant exactly one excursion
  (5011/2) closes faster than with no catch (the paper's Fig. catch caption);
* post hoc readings, labelled post hoc on screen: thrust-reverse reaches full astern
  (min_thrust = -F_T) on exactly 6 runs, its 6 fastest closers, and its other 9 failures never do
  (paper Sec. V); the catch's T_peak_cf exceeds the replay's in 9 of 16 (hold) and 8 of 16
  (reverse) (counted here; gates nothing);
* every caption is on screen >= max(4 s, words / 2.5 s) with <= 2 lines, and every text drawn on
  a rendered frame lies inside the 1920 x 1080 frame.

Narration: every caption is one or two complete sentences (at most 24 words) saying what is on
screen and what it means; the severing speed v_b is defined in words before its symbol is used.
The narration states the law in words (it lowers thrust while the slack closes faster than a
landing profile that tapers to a soft landing at contact) and what separates the two variants
(thrust-hold may cut thrust to zero; thrust-reverse may also reverse it, down to full astern).
The title card states the question and what to watch for and stays up for their reading time.
Each excursion opens on captions held on its first frame (the typical failure's second one states
the law), then plays at the fastest speed that shows its last playing caption, and at least 4 s of
the one before, while the run is playing; reading time that caption still lacks is a paused hold
on the run's first frame (no simulation instant is added or skipped).
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

from tether.analysis.v2.present import common as C
from tether.campaign.mission import MissionSpec, mission_schedules
from tether.campaign.v2 import p6_t0 as P
from tether.control import catch as CATCH
from tether.monitor.hazard import critical_speeds
from tether.physics import constants as K
from tether.physics import fleet as FLEET

NAME = "catch"
CLIP = C.CLIPS / f"{NAME}.mp4"
RESULTS = P.RESULTS_PATH
DECLARATIONS = P.DECLARATIONS_PATH
MISSIONS = P.MISSIONS_PATH
IMPACT = P.FAN_IMPACT_PATH
REL = 1.0e-9

CAUGHT = (5028, 0)
TYPICAL = (5020, 4)
FASTER = (5011, 2)          # the paper's Fig. catch caption: closes faster than with no catch
# key -> (controller, variant)
RUNS = {"replay": ("replay", None), "hold": ("true10", "hold"), "reverse": ("true10", "reverse")}
RUN_KEYS = ("outcome", "caught", "t_cf", "v_cf", "ratio", "T_peak_cf", "delay", "engaged_fraction", "min_thrust")

VB = "$v_b$"                 # typeset symbols (matplotlib mathtext)
FT = "$F_T$"
C_RUN = {"replay": "#4b5563", "hold": C.ACCENT, "reverse": "#7b52ab"}
LS_RUN = {"replay": "-", "hold": "-", "reverse": (0, (5.0, 3.0))}
RUN_LABEL = {"replay": "recorded thrust (no catch)", "hold": "catch: thrust-hold (F ≥ 0)",
             "reverse": f"catch: thrust-reverse (F ≥ −{FT})"}
RUN_SHORT = {"replay": "recorded thrust", "hold": "thrust-hold", "reverse": "thrust-reverse"}
LANE_Y = {"replay": 2.1, "hold": 1.05, "reverse": 0.0}
SPEEDS = (0.5, 1.0 / 3.0, 0.25, 0.2, 1.0 / 6.0, 0.125)
READING_WPS = 2.5
CAPTION_Y = 0.077            # common.caption's default 0.068 lets a 2-line box graze the footer
MIN_CAPTION_S = 4.0
TITLE_S = 3.5                # shortest title card; it stays up for its two sentences' reading time if longer
END_HOLD_S = 4.0
# The title card: the question this clip answers and what to watch for (read at READING_WPS), plus a
# small provenance sentence that, like the footer, is not narration and is not timed.
TITLE_QUESTION = "Can braking a slack tug keep the snap below the severing tension?"
TITLE_WATCH = "Watch whether the cable pulls taut slower than the severing speed."
TITLE_PROVENANCE = ("The catch is tested as a counterfactual on the recorded squall missions "
                    "of the fan formation (campaign test P6-T0).")


def _rel_same(a, b, rel: float = REL) -> bool:
    if a is None or b is None or isinstance(a, (bool, str, np.bool_)) or isinstance(b, (bool, str, np.bool_)):
        return a == b
    a, b = float(a), float(b)
    return abs(a - b) <= rel * max(abs(a), abs(b)) or abs(a - b) <= 1.0e-12


def _run_of(excursion: dict, controller: str, variant) -> dict:
    runs = [r for r in excursion["runs"]
            if r["mode"] == "fleet" and r["controller"] == controller and r["variant"] == variant]
    assert len(runs) == 1, (excursion["tags"]["seed"], excursion["tags"]["cable"], controller, variant)
    return runs[0]


def _convex_overlap(P1: np.ndarray, P2: np.ndarray) -> bool:
    """Separating-axis test for two convex polygons given as vertex arrays (touching = apart)."""
    for poly in (P1, P2):
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            n = np.array([a[1] - b[1], b[0] - a[0]])
            p, q = P1 @ n, P2 @ n
            if p.max() <= q.min() or q.max() <= p.min():
                return False
    return True


def body_overlaps(row) -> list[tuple[str, str]]:
    """Pairs of drawn bodies (payload pentagon and the five hulls) whose outlines overlap."""
    lp, vp = C.unpack_state(row)
    bodies = [("payload", C.body_polygon(lp, FLEET.pentagon_vertices()))]
    bodies += [(f"vessel {i}", C.body_polygon(v, C.hull())) for i, v in enumerate(vp)]
    return [(bodies[a][0], bodies[b][0]) for a in range(len(bodies)) for b in range(a + 1, len(bodies))
            if _convex_overlap(bodies[a][1], bodies[b][1])]


# ---------------------------------------------------------------- on-screen numbers

class Board:
    """Every on-screen number goes through ``num``, which registers it with the manifest."""

    def __init__(self, man: C.Manifest):
        self.man = man
        self.seen: dict = {}

    def num(self, label, value, fmt, unit="", source="", note="") -> str:
        if label in self.seen:
            assert self.seen[label] == value, label
        else:
            self.man.value(label, value, unit, source, note)
            self.seen[label] = value
        return "" if fmt is None else format(value, fmt)


# ---------------------------------------------------------------- the thrust command (reconstructed)

def thrust_command(x: P.Excursion, history: dict, controller: str, min_fraction):
    """Surge command of vessel i on each traced step, rebuilt exactly as ``p6_t0.integrate`` forms it.

    Uses the traced e/edot (the law reads them at the latest 10 Hz monitor tick strictly before
    each 50 Hz control tick), the campaign's tick functions and ``catch_thrust``.  Returns the
    per-step command (length len(label) - 1: the last traced step applies none), the engaged-tick
    fraction and the minimum, which the caller asserts against the record.
    """
    labels, e, edot = history["label"], history["e"], history["edot"]
    i, L0 = x.cable, x.start_label
    ticks = P._control_tick(labels + P.STEP)
    schedule = x.thrusts[i] * np.asarray(x.envelope(ticks), dtype=float)
    n_cmd = labels.size - 1
    command = np.full(n_cmd, np.nan)

    def chord(label):
        m = int(round((label - L0) / P.STEP))
        if m < 0:
            k = int(round(label / P.EVENT_PERIOD)) - x.log_start
            return float(x.e_log[k]), float(x.edot_log[k])
        return float(e[m]), float(edot[m])

    crossed, current, value, engaged, total = None, None, 0.0, 0, 0
    for n in range(n_cmd):
        if crossed is None and n > 0 and e[n - 1] <= 0.0 < e[n]:
            crossed = n
        if controller == "replay":
            value = float(schedule[n])
        else:
            tick = float(ticks[n])
            if tick != current:
                current = tick
                scheduled = float(schedule[n])
                if crossed is not None:
                    value = scheduled
                else:
                    e_hat, edot_hat = chord(P._monitor_tick(tick))
                    value = float(CATCH.catch_thrust(e_hat, edot_hat, e_hat <= 0.0, scheduled,
                                                     x.critical_speed, float(min_fraction)))
                    total += 1
                    engaged += int(value < scheduled - 1.0e-9)
        command[n] = value
    return command, (engaged / total if total else 0.0), float(command.min()), schedule


# ---------------------------------------------------------------- data (all assertions happen here)

def prepare(man: C.Manifest) -> dict:
    import tether.campaign.mission as mission_module
    import tether.monitor.hazard as hazard_module
    for path in (RESULTS, DECLARATIONS, MISSIONS, IMPACT, Path(P.__file__), Path(CATCH.__file__),
                 Path(mission_module.__file__), Path(hazard_module.__file__), Path(FLEET.__file__),
                 Path(C.__file__), Path(__file__)):
        man.source(path)
    res = json.loads(RESULTS.read_text())
    decl = json.loads(DECLARATIONS.read_text())
    rel_res = str(RESULTS.relative_to(C.REPO))

    # ---- population and selection, from the record
    fs = [e for e in res["excursions"] if e["tags"]["first_severance"]]
    n = len(fs)
    man.check("first-severance population has 16 excursions (excursions, population_sizes, verdict.n)",
              n == 16 == int(res["population_sizes"]["first_severance"]) == int(res["verdict"]["n"]), f"n = {n}")
    both = [(int(e["tags"]["seed"]), int(e["tags"]["cable"])) for e in fs
            if _run_of(e, "true10", "hold")["caught"] and _run_of(e, "true10", "reverse")["caught"]]
    man.check("the only excursion caught under both variants is seed 5028 cable 0", both == [CAUGHT], str(both))
    hold_sorted = sorted((float(_run_of(e, "true10", "hold")["ratio"]), int(e["tags"]["seed"]), int(e["tags"]["cable"]))
                         for e in fs)
    lower = hold_sorted[n // 2 - 1]
    upper = hold_sorted[n // 2]
    median_hold = 0.5 * (lower[0] + upper[0])
    table = res["tables"]["fleet"]["true10"]["first_severance"]
    man.check("lower median of the 16 thrust-hold ratios is seed 5020 cable 4",
              (lower[1], lower[2]) == TYPICAL, f"8th of 16 = {lower}, 9th = {upper}")
    man.check("thrust-hold median = midpoint of the two middle ratios = tables...hold.median_ratio",
              _rel_same(median_hold, table["hold"]["median_ratio"]), f"{median_hold} vs {table['hold']['median_ratio']}")

    # ---- replays
    crit = critical_speeds(P.STRESS_THRESHOLD, P.PRETENSION, table_path=IMPACT)
    man.check("recomputed v_b per cable equal record critical_speeds",
              bool(np.allclose(crit, res["critical_speeds"], rtol=0.0, atol=1e-12)), str(list(crit)))
    with MISSIONS.open("rb") as handle:
        missions = pickle.load(handle)
    chosen = {int(m["seed"]): m for m in missions if int(m["seed"]) in (TYPICAL[0], CAUGHT[0])}
    del missions
    excursions, _ = P.build_excursions([chosen[TYPICAL[0]], chosen[CAUGHT[0]]], {}, crit)
    out = {"res": res, "decl": decl, "crit": crit, "ex": {}}
    for seed, cable in (TYPICAL, CAUGHT):
        xs = [x for x in excursions if x.seed == seed and x.cable == cable and x.tags["first_severance"]]
        assert len(xs) == 1
        x = xs[0]
        recs = [e for e in fs if int(e["tags"]["seed"]) == seed and int(e["tags"]["cable"]) == cable]
        assert len(recs) == 1
        rec = recs[0]
        man.check(f"{seed}/{cable}: excursion tags rebuilt by p6_t0.build_excursions equal the record's",
                  json.loads(json.dumps(x.tags)) == rec["tags"], f"t_up {x.t_up}")
        runs = {}
        for key, (controller, variant) in RUNS.items():
            fraction = None if variant is None else CATCH.VARIANTS[variant]
            result = P.integrate(x, "fleet", controller, fraction, trace=True)
            recorded = _run_of(rec, controller, variant)
            bad = [k for k in RUN_KEYS if not _rel_same(result[k], recorded[k])]
            man.check(f"{seed}/{cable} {key}: integrate(fleet, {controller}, {variant}) reproduces the record",
                      not bad, "; ".join(f"{k} {result[k]!r} = {recorded[k]!r}" for k in RUN_KEYS) + (f"; MISMATCH {bad}" if bad else ""))
            history = result["history"]
            command, engaged, minimum, schedule = thrust_command(x, history, controller, fraction)
            man.check(f"{seed}/{cable} {key}: reconstructed thrust command matches record engaged_fraction and min_thrust",
                      _rel_same(engaged, recorded["engaged_fraction"]) and _rel_same(minimum, recorded["min_thrust"]),
                      f"engaged {engaged} vs {recorded['engaged_fraction']}; min {minimum} vs {recorded['min_thrust']}")
            runs[key] = {"label": history["label"], "e": history["e"], "edot": history["edot"], "cmd": command,
                         "schedule": schedule, **{k: result[k] for k in RUN_KEYS}}
        truth = chosen[seed]["truth"]
        # replay vs the recorded 1 ms log, up to the recorded contact
        lab = runs["replay"]["label"]
        on_ms = np.abs(lab / P.EVENT_PERIOD - np.round(lab / P.EVENT_PERIOD)) < 1.0e-6
        pre = on_ms & (lab <= x.t_up)
        idx = np.round(lab[pre] / P.EVENT_PERIOD).astype(int)
        de = float(np.max(np.abs(runs["replay"]["e"][pre] - truth.elongation[idx, cable])))
        dr = float(np.max(np.abs(runs["replay"]["edot"][pre] - truth.rate[idx, cable])))
        man.check(f"{seed}/{cable}: replay e(t), edot(t) equal the recorded 1 ms log up to t_up",
                  de < 1.0e-4 and dr < 1.0e-3, f"max |de| {de:.2e} m, max |dedot| {dr:.2e} m/s over {int(pre.sum())} samples")
        # law runs coincide with the replay until they first brake (and before t_up)
        for key in ("hold", "reverse"):
            braking = np.flatnonzero(runs[key]["cmd"] < runs[key]["schedule"][:-1] - 1.0e-9)
            first = int(braking[0])
            n_up = int(math.ceil((x.t_up - x.start_label) / P.STEP - 1.0e-9))
            m = min(first, n_up) + 1
            same = (np.array_equal(runs[key]["e"][:m], runs["replay"]["e"][:m])
                    and np.array_equal(runs[key]["edot"][:m], runs["replay"]["edot"][:m]))
            man.check(f"{seed}/{cable} {key}: identical to the replay until its first braking tick",
                      same, f"first braking step label {runs[key]['label'][first]:.3f} s, {m} steps identical")
            runs[key]["first_brake"] = float(runs[key]["label"][first])
        # top-down frame: the recorded state at the turnaround row
        k0 = int(round(x.start_label / P.STATE_PERIOD))
        j0 = int(round(x.start_label / P.EVENT_PERIOD))
        row = np.asarray(truth.state[k0], float)
        e_row = np.asarray(truth.elongation[j0], float)
        r_row = np.asarray(truth.rate[j0], float)
        outcome = chosen[seed]["outcome"]
        man.check(f"{seed}/{cable}: top-down row is the excursion's start row, watched cable slack, recording run",
                  abs(truth.state_time[k0] - x.start_label) < 1e-9 and abs(truth.event_time[j0] - x.start_label) < 1e-9
                  and np.array_equal(row, x.state[0]) and e_row[cable] < 0.0 and len(outcome.severances) == 0,
                  f"label {x.start_label}, e_i {e_row[cable]:.4f} m, severances {outcome.severances}")
        overlaps = body_overlaps(row)
        man.check(f"{seed}/{cable}: overlapping bodies in the drawn top-down row (separating-axis test; the plant "
                  "models no body contact, tether/physics/fleet.py); the on-screen note is shown iff there are any",
                  bool(overlaps) == ((seed, cable) == CAUGHT), f"overlapping pairs: {overlaps or 'none'}")
        alive = np.ones(5, bool)
        cache = C.CACHE / f"catch_{seed}_{cable}.npz"
        arrays = {"state_row": row, "e_row": e_row, "rate_row": r_row}
        for key, r in runs.items():
            for field in ("label", "e", "edot", "cmd"):
                arrays[f"{key}_{field}"] = np.asarray(r[field], float)
        np.savez_compressed(cache, **arrays)
        out["ex"][(seed, cable)] = {
            "x": x, "tags": rec["tags"], "runs": runs, "row": row, "e_row": e_row,
            "T_row": C.tension(e_row, r_row, alive), "cache": str(cache.relative_to(C.REPO)),
            "F_T": float(x.thrusts[cable]), "overlaps": overlaps,
        }
    # thrust schedule: F_T constant on [8, 110] s
    thrusts = mission_schedules(MissionSpec()).thrusts
    man.check("scheduled thrust F_T of cables 0 and 4 is the operating thrust (envelope 1 on the excursions)",
              all(np.all(r["schedule"] == thrusts[c]) for (s, c), ex in out["ex"].items() for r in ex["runs"].values()),
              f"F_T = {list(thrusts)}")
    ra = out["ex"][TYPICAL]["runs"]
    man.check("5020/4: the catch's first tension peak T_peak_cf exceeds the replay's under both variants (caption A4)",
              ra["hold"]["T_peak_cf"] > ra["replay"]["T_peak_cf"] and ra["reverse"]["T_peak_cf"] > ra["replay"]["T_peak_cf"],
              f"replay {ra['replay']['T_peak_cf']:.1f} N, hold {ra['hold']['T_peak_cf']:.1f} N, "
              f"reverse {ra['reverse']['T_peak_cf']:.1f} N")

    # ---- tally, from the record
    tally = []
    for e in fs:
        t = e["tags"]
        cable = int(t["cable"])
        reverse = _run_of(e, "true10", "reverse")
        tally.append({
            "id": (int(t["seed"]), cable),
            "ratio": {k: float(_run_of(e, c, v)["ratio"]) for k, (c, v) in RUNS.items()},
            "caught": {k: bool(_run_of(e, c, v)["caught"]) for k, (c, v) in RUNS.items()},
            "outcome": {k: _run_of(e, c, v)["outcome"] for k, (c, v) in RUNS.items()},
            "T_peak": {k: float(_run_of(e, c, v)["T_peak_cf"]) for k, (c, v) in RUNS.items()},
            "full_astern": bool(reverse["min_thrust"] <= -thrusts[cable] * (1.0 - 1e-9)),
        })
    man.check("all 16 runs of each controller re-engage (no closure, no horizon)",
              all(o == "reengaged" for r in tally for o in r["outcome"].values()), "")
    med = {k: float(np.median([r["ratio"][k] for r in tally])) for k in RUNS}
    for v in ("hold", "reverse"):
        caught = sum(r["caught"][v] for r in tally)
        man.check(f"{v}: medians, caught count and fraction equal tables.fleet.true10.first_severance and verdict",
                  _rel_same(med[v], table[v]["median_ratio"]) and caught == table[v]["caught"] == 1
                  and _rel_same(caught / n, res["verdict"]["fractions"][v]) and table[v]["n"] == n
                  and np.allclose(table[v]["interval95"], res["verdict"]["intervals95"][v]),
                  f"median {med[v]}, caught {caught}/{n}, CI {res['verdict']['intervals95'][v]}")
    faster = {v: [r["id"] for r in tally if r["ratio"][v] > r["ratio"]["replay"]] for v in ("hold", "reverse")}
    man.check("under each variant exactly one excursion closes faster than with no catch, 5011/2 "
              "(paper Sec. V, Fig. catch caption)", faster == {"hold": [FASTER], "reverse": [FASTER]}, str(faster))
    rev_sorted = sorted(tally, key=lambda r: -r["ratio"]["reverse"])
    sat = [r for r in tally if r["full_astern"]]
    man.check("post hoc (paper Sec. V): thrust-reverse reaches full astern (min_thrust = -F_T) on exactly 6 runs, "
              "the 6 fastest closers, all failures",
              len(sat) == 6 and {r["id"] for r in sat} == {r["id"] for r in rev_sorted[:6]}
              and not any(r["caught"]["reverse"] for r in sat),
              f"{sorted(r['id'] for r in sat)}; ratios {min(r['ratio']['reverse'] for r in sat):.3f}-{max(r['ratio']['reverse'] for r in sat):.3f}")
    others = [r for r in tally if not r["full_astern"] and not r["caught"]["reverse"]]
    man.check("post hoc (paper Sec. V, 'its other nine failures'): the other 9 thrust-reverse failures never command "
              "full astern", len(others) == 9 and len(others) + len(sat) + sum(r["caught"]["reverse"] for r in tally) == n,
              str(sorted(r["id"] for r in others)))
    peak_up = {v: sum(r["T_peak"][v] > r["T_peak"]["replay"] for r in tally) for v in ("hold", "reverse")}
    lower_v = {v: sum(r["ratio"][v] < r["ratio"]["replay"] for r in tally) for v in ("hold", "reverse")}
    man.check("post hoc (counted here, gates nothing): the catch's first tension peak T_peak_cf exceeds the replay's "
              "in 9 of 16 (hold) and 8 of 16 (reverse), while its closing speed is lower in 15 of 16 under each",
              peak_up == {"hold": 9, "reverse": 8} and lower_v == {"hold": 15, "reverse": 15},
              f"peak above replay {peak_up}; closing speed below replay {lower_v}")
    arms = {(a, v): int(res["tables"]["fleet"][a]["first_severance"][v]["caught"]) for a in ("P", "B2", "L") for v in ("hold", "reverse")}
    man.check("estimator-fed arms (P, B2, L) catch 1 to 4 of 16", min(arms.values()) == 1 and max(arms.values()) == 4, str(arms))
    bar_text = decl["admissibility_rule"]
    man.check("pre-declared admissibility bar is 0.80 and the verdict is NO-LAUNCH, neither variant admissible",
              ">= 0.80" in bar_text and P.ADMISSIBLE_FRACTION == 0.80 and res["verdict"]["outcome"] == "NO-LAUNCH"
              and not any(res["verdict"]["admissible"].values()), res["verdict"]["outcome"])
    man.check("fleet-mode replay validation passed (the verdict mode)", bool(res["validation"]["fleet"]["passed"]),
              f"max |dv| {res['validation']['fleet']['abs_dv_m_s']['max']:.4f} m/s over 43")
    out.update(tally=tally, med=med, sat=sat, others=others, peak_up=peak_up, lower_v=lower_v, arms=arms, n=n,
               table=table, rel_res=rel_res, hold_middle=(lower, upper), median_hold=median_hold)
    return out


# ---------------------------------------------------------------- population numbers (registered once)

def population_numbers(board: Board, data: dict) -> dict:
    """Register every population-level number once; returns the on-screen strings."""
    num, src = board.num, data["rel_res"]
    decl = data["decl"]
    verdict = data["res"]["verdict"]
    table = data["table"]
    sat = data["sat"]
    arms = data["arms"]
    s = {}
    s["n"] = num("first-severance population size", data["n"], "d", "excursions",
                 f"{src} population_sizes.first_severance = verdict.n = len(excursions[tags.first_severance])")
    s["bar"] = num("pre-declared admissibility bar (caught fraction)", P.ADMISSIBLE_FRACTION, ".2f", "",
                   "records/v2/phase6/p6_t0_declarations.json admissibility_rule ('>= 0.80')")
    assert "10 Hz" in decl["controllers"]["true10"] and round(1.0 / P.MONITOR_PERIOD) == 10
    assert decl["constants"]["F_min_variants"] == {"hold": "F_min = 0", "reverse": "F_min = -F_T"}
    s["hz"] = num("monitor rate of the verdict controller (true chord state)", round(1.0 / P.MONITOR_PERIOD), "d", "Hz",
                  "records/v2/phase6/p6_t0_declarations.json controllers.true10; p6_t0.MONITOR_PERIOD")
    for v in ("hold", "reverse"):
        s[f"caught_{v}"] = num(f"{v}: caught count", int(table[v]["caught"]), "d", f"of {data['n']}",
                               f"{src} tables.fleet.true10.first_severance.{v}.caught")
        num(f"{v}: caught fraction", float(verdict["fractions"][v]), None, "", f"{src} verdict.fractions.{v}",
            "drawn as a point in panel (b)")
        num(f"{v}: Clopper-Pearson 95% lower end", float(verdict["intervals95"][v][0]), None, "",
            f"{src} verdict.intervals95.{v}[0]", "drawn as the error bar's lower end")
        s[f"hi_{v}"] = num(f"{v}: Clopper-Pearson 95% upper end", float(verdict["intervals95"][v][1]), ".2f", "",
                           f"{src} verdict.intervals95.{v}[1]")
    assert s["caught_hold"] == s["caught_reverse"]
    s["caught"] = s["caught_hold"]
    for k in RUNS:
        source = (f"{src} tables.fleet.true10.first_severance.{k}.median_ratio" if k != "replay" else
                  f"{src} excursions[first_severance].runs[fleet, replay].ratio (median of the 16, computed here)")
        s[f"med_{k}"] = num(f"median v/v_b over the 16, {k}", data["med"][k], ".2f", "v_b", source,
                            "asserted equal to the median of the 16 per-run ratios" if k != "replay"
                            else "derived; gates nothing")
    s["n_sat"] = num("post hoc: thrust-reverse runs reaching full astern (min_thrust = -F_T)", len(sat), "d", "of 16",
                     f"{src} excursions[first_severance].runs[fleet, true10, reverse].min_thrust vs -F_T",
                     "post hoc (paper Sec. V: not part of the pre-registered test); counted here from the record; "
                     "asserted to be the 6 fastest reverse closers")
    s["lo_sat"] = num("post hoc: slowest full-astern reverse run", min(r["ratio"]["reverse"] for r in sat), ".1f", "v_b",
                      f"{src} excursions[first_severance].runs[fleet, true10, reverse].ratio", "post hoc (paper Sec. V)")
    s["hi_sat"] = num("post hoc: fastest full-astern reverse run", max(r["ratio"]["reverse"] for r in sat), ".1f", "v_b",
                      f"{src} excursions[first_severance].runs[fleet, true10, reverse].ratio", "post hoc (paper Sec. V)")
    s["n_other"] = num("post hoc: other thrust-reverse failures, never commanding full astern", len(data["others"]), "d",
                       "of 16", f"{src} excursions[first_severance].runs[fleet, true10, reverse].{{min_thrust, caught}}",
                       "post hoc (paper Sec. V: 'its other nine failures'); counted here from the record")
    for v in ("hold", "reverse"):
        s[f"peak_up_{v}"] = num(f"post hoc: {v} runs whose first tension peak T_peak_cf exceeds the replay's",
                                data["peak_up"][v], "d", "of 16",
                                f"{src} excursions[first_severance].runs[fleet, true10, {v}].T_peak_cf vs "
                                "runs[fleet, replay].T_peak_cf",
                                "post hoc, counted here from the record; T_peak_cf is reported by P6-T0 and gates nothing; "
                                "not stated in the paper")
    fx = [r for r in data["tally"] if r["id"] == FASTER][0]
    s["faster"] = num("excursion closing faster than with no catch under each variant", f"{FASTER[0]}/{FASTER[1]}", "s", "",
                      f"{src} excursions[first_severance].runs[fleet].ratio (asserted the only one per variant)",
                      "identifier; paper Sec. V Fig. catch caption: 'Under each variant one excursion closes faster'")
    for k in RUNS:
        s[f"faster_{k}"] = num(f"{FASTER[0]}/{FASTER[1]} {k}: v/v_b (printed)", fx["ratio"][k], ".2f", "v_b",
                               f"{src} excursions[{FASTER[0]}/{FASTER[1]}].runs[fleet, {RUNS[k][0]}, {RUNS[k][1]}].ratio")
    s["arms_lo"] = num("estimator-fed arms: fewest caught", min(arms.values()), "d", "of 16",
                       f"{src} tables.fleet.{{P,B2,L}}.first_severance.{{hold,reverse}}.caught")
    s["arms_hi"] = num("estimator-fed arms: most caught", max(arms.values()), "d", "of 16",
                       f"{src} tables.fleet.P.first_severance.hold.caught")
    s["outcome"] = num("P6-T0 verdict", verdict["outcome"], "s", "", f"{src} verdict.outcome")
    s["sel_lo"] = num("selection: lower-median thrust-hold ratio (5020/4)", data["hold_middle"][0][0], ".3f", "v_b",
                      f"{src} excursions[5020/4].runs[fleet, true10, hold].ratio")
    s["sel_hi"] = num("selection: upper-median thrust-hold ratio (5015/2)", data["hold_middle"][1][0], ".3f", "v_b",
                      f"{src} excursions[5015/2].runs[fleet, true10, hold].ratio")
    s["sel_med"] = num("median v/v_b over the 16, hold", data["med"]["hold"], ".3f", "v_b")
    s["tb"] = num("stress threshold T_b^s (severance scoring threshold)", float(decl["constants"]["T_b_s_N"]) / 1000.0,
                  ".1f", "kN", "records/v2/phase6/p6_t0_declarations.json constants.T_b_s_N",
                  "a scoring threshold (paper Sec. II: the 55th percentile of the per-mission maximum tension), "
                  "not a break strength")
    s["k_v"] = num("catch gain K_v", float(decl["constants"]["K_v_N_s_m"]), ".0f", "N s/m",
                   "records/v2/phase6/p6_t0_declarations.json constants.K_v_N_s_m")
    s["a_c"] = num("landing deceleration a_c", float(decl["constants"]["a_c_m_s2"]), ".1f", "m/s^2",
                   "records/v2/phase6/p6_t0_declarations.json constants.a_c_m_s2")
    assert decl["constants"]["v_soft"].startswith(str(CATCH.SOFT_SPEED_FRACTION))
    s["soft"] = num("soft landing speed v_soft / v_b", CATCH.SOFT_SPEED_FRACTION, ".1f", "",
                    "records/v2/phase6/p6_t0_declarations.json constants.v_soft; tether/control/catch.py")
    s["ordinal"] = num("clip ordinal (mitigation 2)", 2, "d", "", "Presentation/STORYBOARD.md C6", "ordinal, not data")
    assert CATCH.VARIANTS == {"hold": 0.0, "reverse": -1.0}
    num("F_min of the two variants (F_min / F_T)", dict(CATCH.VARIANTS), None, "F_T",
        "records/v2/phase6/p6_t0_declarations.json constants.F_min_variants; tether/control/catch.py VARIANTS",
        "shown as 'F ≥ 0' (thrust-hold) and 'F ≥ −F_T' (thrust-reverse)")
    for (seed, cable) in (TYPICAL, CAUGHT):
        num(f"shown excursion seed {seed}", seed, "d", "", f"{src} excursions[.].tags.seed", "identifier (selection rule)")
        num(f"shown excursion {seed}: cable", cable, "d", "", f"{src} excursions[.].tags.cable", "identifier")
    return s


def contact_numbers(board: Board, data: dict, key) -> dict:
    """Closing speed, v/v_b and first tension peak at contact of the three runs of one excursion (registered)."""
    seed, cable = key
    src = data["rel_res"]
    out = {}
    for k, r in data["ex"][key]["runs"].items():
        base = f"{src} excursions[{seed}/{cable}].runs[fleet, {RUNS[k][0]}, {RUNS[k][1]}]"
        v = board.num(f"{seed}/{cable} {k}: closing speed at contact v_cf", float(r["v_cf"]), ".2f", "m/s",
                      base + ".v_cf", "asserted equal to the replay (1e-9 relative)")
        q = board.num(f"{seed}/{cable} {k}: v_cf / v_b", float(r["ratio"]), ".2f", "v_b",
                      base + ".ratio", "asserted equal to the replay (1e-9 relative)")
        p = board.num(f"{seed}/{cable} {k}: first tension peak after contact T_peak_cf", float(r["T_peak_cf"]) / 1000.0,
                      ".1f", "kN", base + ".T_peak_cf",
                      "reported by P6-T0 (declarations 'peak': first peak of cable i after the counterfactual contact, "
                      "at most 1 s); gates nothing (the verdict is v_cf < v_b); asserted equal to the replay (1e-9 "
                      "relative); only the replay's has a recorded counterpart (tags.T_peak)")
        out[k] = (float(r["t_cf"]), float(r["v_cf"]), v, q, bool(r["caught"]), p)
    return out


# ---------------------------------------------------------------- drawing helpers

def speed_label(speed: float) -> str:
    if abs(speed - 1.0) < 1e-12:
        return "real time"
    return f"×1/{round(1.0 / speed):d} slow motion"


def _hull_xy(stern_x: float, y: float, half_height: float = 0.26) -> np.ndarray:
    """Plant hull outline laid along the chord axis: stern at ``stern_x``, bow toward +x (length to scale)."""
    h = C.hull()
    return np.column_stack([h[:, 0] + 0.5 * C.L_HULL + stern_x, y + h[:, 1] / C.B_HULL * 2.0 * half_height])


def _register_axis(board: Board, label: str, ax, which: str, unit: str) -> None:
    """Register an axis' limits and the tick values printed on it."""
    lim = ax.get_xlim() if which == "x" else ax.get_ylim()
    lo, hi = min(lim), max(lim)
    ticks = ax.get_xticks() if which == "x" else ax.get_yticks()
    shown = [round(float(t), 6) for t in ticks if lo - 1e-9 <= t <= hi + 1e-9]
    board.num(label, [round(float(lim[0]), 4), round(float(lim[1]), 4)], None, unit, "drawing scale", "axis limits")
    board.num(label.replace("limits", "tick values"), shown, None, unit, "drawing scale",
              "tick values printed on this axis (matplotlib ticks inside the limits)")


def texts_inside(fig) -> list[str]:
    """Visible texts whose rendered extent leaves the 1920 x 1080 frame (should be none)."""
    renderer = fig.canvas.get_renderer()
    bad = []
    artists = list(fig.texts) + [t for ax in fig.axes if ax.get_visible() for t in ax.texts]
    artists += [lg for ax in fig.axes if ax.get_visible() for lg in [ax.get_legend()] if lg is not None]
    artists += list(fig.legends)
    for a in artists:
        if not a.get_visible() or (hasattr(a, "get_text") and not a.get_text()):
            continue
        bb = a.get_window_extent(renderer)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > C.W + 0.5 or bb.y1 > C.H + 0.5:
            bad.append(f"{getattr(a, 'get_text', lambda: 'legend')()!r} {bb}")
    return bad


class ExcursionScene:
    """Top-down recorded frame + chord view + e(t), edot(t), F(t) for one excursion."""

    def __init__(self, fig, data: dict, key, board: Board, info: dict, pop: dict):
        self.fig, self.board, self.info = fig, board, info
        ex = data["ex"][key]
        self.ex, self.runs, self.x = ex, ex["runs"], ex["x"]
        seed, cable = key
        self.seed, self.cable = seed, cable
        tags = ex["tags"]
        vb = float(tags["critical_speed"])
        self.vb = vb
        rest = float(data["decl"]["constants"]["cable"]["rest_length_m"])
        assert rest == K.CABLE_REST_LENGTH
        self.rest = rest
        self.t0 = float(self.x.start_label)
        self.t_end = max(float(r["label"][-1]) for r in self.runs.values())
        src_rec = data["rel_res"]
        cache = ex["cache"]
        num = board.num

        fig.clf()
        C.title(fig, info["title"], info["sub"])
        fig.text(0.03, 0.862, info["selection"], fontsize=C.FS_SMALL, color=C.MUTED, ha="left", va="top")
        C.footer(fig, "P6-T0 counterfactual (p6_t0.integrate, fleet mode) asserted = records/v2/phase6/p6_t0_results.json"
                      "  ·  recorded state: records/phase5/cache/phase5_missions.pkl")

        # ---- top-down: the recorded fleet at the turnaround (static)
        t_turn = num(f"{seed}/{cable} turnaround state label", self.t0, ".2f", "s",
                     "records/phase5/cache/phase5_missions.pkl truth.state_time[k0] = "
                     f"{src_rec} excursions[{seed}/{cable}].tags.start_label",
                     "the 10 ms state row at or before the deepest point of the excursion")
        e_turn = num(f"{seed}/{cable} e of cable {cable} at the turnaround", float(ex["e_row"][cable]), ".2f", "m",
                     "records/phase5/cache/phase5_missions.pkl truth.elongation[start_label / 1 ms, cable]")
        fig.text(0.03, 0.81, f"Recorded fleet, t = {t_turn} s (turnaround, static)", fontsize=C.FS_SMALL,
                 weight="bold", color=C.INK, ha="left", va="bottom")
        ax_td = fig.add_axes([0.03, 0.50, 0.31, 0.305])
        geometry = C.mission_geometry()
        centres, half = C.fleet_bounds(ex["row"][None, :], geometry, margin=3.0)
        box = ax_td.get_position()
        half = C.fit_aspect(half, (box.width * C.W) / (box.height * C.H))
        C.draw_fleet(ax_td, ex["row"], geometry, ex["T_row"], None, centre=centres[0], half=half,
                     labels=info["vessel_labels"])
        num("top-down scale bar", 5.0, ".0f", "m", "common.draw_fleet", "drawing scale bar")
        if info["vessel_labels"]:
            for i in range(5):
                num(f"top-down vessel label {i}", i, "d", "", "vessel index (common.draw_fleet)", "identifier")
        lp, vp = C.unpack_state(ex["row"])
        a, b = C.attachment_points(lp, vp, geometry)
        seg = np.stack([a[cable], b[cable]])
        ax_td.plot(*seg.T, color=C.ACCENT, lw=11, alpha=0.30, solid_capstyle="round", zorder=2.5)
        ax_td.annotate(f"cable {cable}: slack, e = {e_turn.replace('-', '−')} m", xy=seg.mean(axis=0), xycoords="data",
                       xytext=info["label_at"], textcoords="axes fraction", fontsize=C.FS_SMALL, color=C.INK,
                       ha="center", va="center", zorder=7,
                       bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C.ACCENT, lw=1.0),
                       arrowprops=dict(arrowstyle="-|>", color=C.ACCENT, lw=1.6, shrinkA=2, shrinkB=4))
        if ex["overlaps"]:
            ax_td.text(0.985, 0.03, "no body contact in the plant: hulls may overlap", transform=ax_td.transAxes,
                       fontsize=C.FS_TINY, color=C.MUTED, ha="right", va="bottom")
        handles = [Line2D([], [], color=C.TAUT, lw=4, label="taut (width ∝ tension)"),
                   Line2D([], [], color=C.SLACK, lw=2, ls=(0, (3.0, 2.2)), label="carrying no tension"),
                   Line2D([], [], color=C.ACCENT, lw=9, alpha=0.3, solid_capstyle="butt", label="watched cable")]
        fig.legend(handles=handles, loc="upper left", ncol=3, bbox_to_anchor=(0.022, 0.494), fontsize=C.FS_TINY,
                   handlelength=2.0, handletextpad=0.6, columnspacing=1.0, frameon=False)

        # ---- the law, as declared
        law = ("The catch (declared law): while the cable is slack\n"
               "and $\\dot e > \\dot e_{ref}(e)$, the slack tug's thrust is\n"
               "   $F = \\mathrm{clip}\\,(F_T + K_v(\\dot e_{ref} - \\dot e),\\ F_{min},\\ F_T)$\n"
               f"   landing profile: $\\dot e_{{ref}}(e) = \\sqrt{{v_{{soft}}^2 + 2a_c(-e)}}$,  $v_{{soft}}$ = {pop['soft']} $v_b$\n"
               f"   $a_c$ = {pop['a_c']} m/s²,  $K_v$ = {pop['k_v']} N·s/m\n"
               "   thrust-hold: $F_{min}$ = 0;  thrust-reverse: $F_{min} = -F_T$\n"
               f"Fed the true chord state e, ė at {pop['hz']} Hz (not an\n"
               "estimator). The whole fleet is integrated from the\n"
               "recorded turnaround in the recorded weather; headings\n"
               "from the record (slack tug's held from recorded contact).")
        fig.text(0.03, 0.44, law, fontsize=C.FS_TINY + 1, color=C.INK, ha="left", va="top", linespacing=1.4)

        # ---- chord view
        fig.text(0.37, 0.81, "Chord view: attachment-to-stern distance to scale (hull and payload block are markers)",
                 fontsize=C.FS_SMALL, weight="bold", color=C.INK, ha="left", va="bottom")
        ax = fig.add_axes([0.37, 0.645, 0.605, 0.16])
        self.ax_ch = ax
        ax.set_xlim(-2.6, 30.0)
        ax.set_ylim(-0.5, 2.95)
        for s in ("left", "right", "top"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_bounds(0.0, 15.0)
        ax.set_yticks([])
        rest_txt = num("cable rest length L", rest, ".0f", "m",
                       "records/v2/phase6/p6_t0_declarations.json constants.cable.rest_length_m")
        ax.set_xticks([0, 3, 6, 9, 12, 15])
        ax.set_xticklabels(["0 m", "3", "6", "9", f"L = {rest_txt}", "15"])
        for v in (0, 3, 6, 9, 15):
            num(f"chord-view axis tick {v} m", v, "d", "m", "drawing scale", "axis tick")
        ax.tick_params(axis="x", labelsize=C.FS_TINY, pad=2)
        ax.axvline(rest, color=C.MUTED, ls=":", lw=1.3, zorder=1)
        ax.text(rest - 0.15, 2.93, "contact (e = 0)", fontsize=C.FS_TINY, color=C.MUTED, ha="right", va="top")
        self.lane = {}
        for key_run in RUNS:
            y = LANE_Y[key_run]
            col = C_RUN[key_run]
            ax.add_patch(Rectangle((-2.2, y - 0.28), 2.2, 0.56, fc=C.PAYLOAD_FACE, ec=C.INK, lw=1.0, zorder=2))
            ax.text(-1.1, y, "payload", fontsize=C.FS_TINY - 1, color=C.INK, ha="center", va="center", zorder=3)
            ax.plot([0.0], [y], marker="o", ms=5, color=C.INK, zorder=4)
            chord, = ax.plot([], [], lw=2.0, zorder=3)
            hull = Polygon(_hull_xy(rest, y), closed=True, fc="white", ec=col, lw=1.8, zorder=4)
            ax.add_patch(hull)
            stern, = ax.plot([], [], marker="o", ms=6, color=col, zorder=5)
            ax.text(15.6, y + 0.05, RUN_LABEL[key_run], fontsize=C.FS_SMALL, color=col, ha="left", va="bottom",
                    weight="bold")
            status = ax.text(15.6, y - 0.03, "", fontsize=C.FS_SMALL, color=col, ha="left", va="top")
            self.lane[key_run] = (chord, hull, stern, status, y)
        fig.text(0.975, 0.632, "peak: first tension peak after contact (gates nothing)",
                 fontsize=C.FS_TINY, color=C.MUTED, ha="right", va="center")

        # ---- time series
        tlo = self.t0 - 0.03 * (self.t_end - self.t0)
        thi = self.t_end + 0.06 * (self.t_end - self.t0)
        emin = min(float(np.min(r["e"])) for r in self.runs.values())
        emax = max(float(np.max(r["e"])) for r in self.runs.values())
        rmin = min(float(np.min(r["edot"])) for r in self.runs.values())
        rmax = max(float(np.max(r["edot"])) for r in self.runs.values())
        F_T = ex["F_T"]
        ax_e = fig.add_axes([0.44, 0.495, 0.535, 0.11])
        ax_r = fig.add_axes([0.44, 0.325, 0.535, 0.14], sharex=ax_e)
        ax_f = fig.add_axes([0.44, 0.19, 0.535, 0.105], sharex=ax_e)
        ax_e.set_xlim(tlo, thi)
        ax_e.set_ylim(emin - 0.12 * (emax - emin), emax + 0.6 * (emax - emin))
        span_r = max(rmax, vb) - min(rmin, 0.0)
        ax_r.set_ylim(min(rmin, 0.0) - 0.1 * span_r, max(rmax, vb) + 1.0 * span_r)
        ax_f.set_ylim(-1.3 * F_T, 1.9 * F_T)
        for axis in (ax_e, ax_r, ax_f):
            axis.grid(True, axis="both", ls=":")
            axis.tick_params(labelsize=C.FS_TINY)
        for axis in (ax_e, ax_r):
            axis.tick_params(labelbottom=False)
        ax_e.set_ylabel("e (m)", fontsize=C.FS_SMALL)
        ax_r.set_ylabel("ė (m/s)", fontsize=C.FS_SMALL)
        ax_f.set_ylabel("F (N)", fontsize=C.FS_SMALL)
        note = dict(fontsize=C.FS_TINY, color=C.MUTED, va="top", zorder=7)
        ax_e.text(0.995, 0.96, "e = chord − L  (≤ 0: slack)", transform=ax_e.transAxes, ha="right", **note)
        ax_f.set_xlabel("simulation clock t (s)", fontsize=C.FS_TINY + 1, labelpad=2)
        ax_e.axhline(0.0, color=C.MUTED, ls=":", lw=1.3)
        vb_txt = num(f"severing speed v_b of cable {cable}", vb, ".3f", "m/s",
                     f"{src_rec} excursions[{seed}/{cable}].tags.critical_speed = critical_speeds[{cable}]",
                     "closing speed whose snap reaches the 4.5 kN scoring threshold in the fan impact table")
        vb_line = ax_r.axhline(vb, color=C.OK, ls=(0, (5, 3)), lw=1.8, zorder=2,
                               label=f"severing speed {VB} = {vb_txt} m/s")
        ax_r.legend(handles=[vb_line], loc="upper left", fontsize=C.FS_TINY + 1, handlelength=2.6, borderaxespad=0.3,
                    labelcolor=C.OK)
        ax_r.text(0.012, 0.74, f"the closing speed (ė at e = 0) whose snap reaches the {pop['tb']} kN scoring "
                               "threshold (fan impact table)", transform=ax_r.transAxes, ha="left", **note)
        ft = num(f"scheduled thrust F_T of vessel {cable}", F_T, ".0f", "N",
                 "tether.campaign.mission.mission_schedules(MissionSpec()).thrusts (envelope 1 on [8, 110] s, asserted); "
                 "p6_t0 declarations constants.F_T")
        ax_f.set_yticks([-F_T, 0.0, F_T])
        ax_f.set_yticklabels([f"−{ft}", "0", ft])
        for level in (-F_T, 0.0, F_T):
            ax_f.axhline(level, color=C.FAINT, lw=1.0, zorder=1)
        ax_f.text(0.005, 0.96, f"thrust command of the slack tug   ({FT} = {ft} N scheduled;  −{FT} = full astern)",
                  transform=ax_f.transAxes, ha="left", **note)
        _register_axis(board, f"{seed}/{cable} time axis limits", ax_e, "x", "s")
        _register_axis(board, f"{seed}/{cable} e axis limits", ax_e, "y", "m")
        _register_axis(board, f"{seed}/{cable} edot axis limits", ax_r, "y", "m/s")
        _register_axis(board, f"{seed}/{cable} F axis limits", ax_f, "y", "N")
        self.lines = {}
        for key_run in RUNS:
            col, ls = C_RUN[key_run], LS_RUN[key_run]
            le, = ax_e.plot([], [], color=col, ls=ls, lw=2.2, label=RUN_SHORT[key_run])
            lr, = ax_r.plot([], [], color=col, ls=ls, lw=2.2)
            lf, = ax_f.plot([], [], color=col, ls=ls, lw=2.0, drawstyle="steps-post")
            me, = ax_e.plot([], [], ls="none", marker="o", ms=8, mfc=col, mec="white", mew=1.2, zorder=6)
            mr, = ax_r.plot([], [], ls="none", marker="o", ms=9, mfc=col, mec="white", mew=1.2, zorder=6)
            self.lines[key_run] = (le, lr, lf, me, mr)
        ax_e.legend(loc="upper left", ncol=3, fontsize=C.FS_TINY + 1, handlelength=2.6, columnspacing=1.4,
                    borderaxespad=0.3, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
        e0 = float(self.runs["replay"]["e"][0])
        assert all(float(r["e"][0]) == e0 for r in self.runs.values())
        ax_e.plot([self.t0], [e0], ls="none", marker="o", ms=6, color=C.INK, zorder=7)
        ax_e.annotate("turnaround", xy=(self.t0, e0), xytext=(8, 4), textcoords="offset points", fontsize=C.FS_TINY,
                      color=C.INK, ha="left", va="bottom")
        # least command of the law runs, marked once the animation reaches it (only where it stays above 0 N)
        self.least = None
        hold = self.runs["hold"]
        if float(hold["min_thrust"]) > 0.0:
            j = int(np.argmin(hold["cmd"]))
            t_least = float(hold["label"][j])
            f_least = float(hold["cmd"][j])
            assert _rel_same(f_least, hold["min_thrust"])
            least_txt = num(f"{seed}/{cable} hold and reverse: least thrust commanded", float(hold["min_thrust"]), ".0f", "N",
                            f"{src_rec} excursions[{seed}/{cable}].runs[fleet, true10, hold|reverse].min_thrust",
                            "identical in both variants; asserted against the reconstructed command")
            num(f"{seed}/{cable} time of the least command", t_least, None, "s",
                f"replay:{cache} hold_label[argmin hold_cmd]", "marker position, not printed")
            mk, = ax_f.plot([], [], ls="none", marker="o", ms=7, mfc="white", mec=C_RUN["hold"], mew=2.0, zorder=8)
            tx = ax_f.annotate(f"least {least_txt} N", xy=(t_least, f_least), xytext=(-10, -3), textcoords="offset points",
                               fontsize=C.FS_TINY, color=C.INK, ha="right", va="top", zorder=8)
            tx.set_visible(False)
            self.least = (t_least, f_least, mk, tx)
        self.contact = contact_numbers(board, data, key)
        num(f"{seed}/{cable} simulation clock range", [self.t0, self.t_end], None, "s",
            f"replay:{cache} replay_label, hold_label, reverse_label (record label clock; plant state at label + 0.5 ms)",
            "clock shown top-right on every animated frame, 3 decimals")
        self.update(self.t0)

    def update(self, t: float) -> None:
        t = min(max(t, self.t0), self.t_end)
        for key_run, r in self.runs.items():
            lab = r["label"]
            k = int(np.searchsorted(lab, t + 1e-12, side="right")) - 1
            k = min(max(k, 0), lab.size - 1)
            le, lr, lf, me, mr = self.lines[key_run]
            le.set_data(lab[: k + 1], r["e"][: k + 1])
            lr.set_data(lab[: k + 1], r["edot"][: k + 1])
            kc = min(k, r["cmd"].size - 1)
            lf.set_data(lab[: kc + 1], r["cmd"][: kc + 1])
            chord, hull, stern, status, y = self.lane[key_run]
            e_now = float(r["e"][k])
            edot_now = float(r["edot"][k])
            xs = self.rest + e_now
            chord.set_data([0.0, xs], [y, y])
            if e_now > 0.0:
                chord.set_color(C.TAUT)
                chord.set_linestyle("-")
            else:
                chord.set_color(C.SLACK)
                chord.set_linestyle((0, (3.0, 2.2)))
            hull.set_xy(_hull_xy(xs, y))
            stern.set_data([xs], [y])
            t_cf, v_cf, v_txt, q_txt, caught, p_txt = self.contact[key_run]
            if t >= t_cf:
                me.set_data([t_cf], [0.0])
                mr.set_data([t_cf], [v_cf])
                head = "caught" if caught else "contact"
                status.set_text(f"{head}: {v_txt} m/s = {q_txt} {VB}, peak {p_txt} kN")
                status.set_color(C.OK if caught else C_RUN[key_run])
                status.set_weight("bold" if caught else "normal")
            else:
                me.set_data([], [])
                mr.set_data([], [])
                # e <= 0 here (e > 0 only after the first crossing, t >= t_cf)
                status.set_text("slack, closing" if edot_now > 0.0 else "slack, not yet closing (ė ≤ 0)")
                status.set_color(C.MUTED)
                status.set_weight("normal")
        if self.least is not None:
            t_least, f_least, mk, tx = self.least
            shown = t >= t_least
            mk.set_data([t_least] if shown else [], [f_least] if shown else [])
            tx.set_visible(shown)


class TallyScene:
    """The 16 first-severance excursions: v/v_b per controller and the caught fraction vs the bar."""

    def __init__(self, fig, data: dict, board: Board, pop: dict):
        num = board.num
        tally, n = data["tally"], data["n"]
        fig.clf()
        C.title(fig, f"All {pop['n']} first-severance excursions",
                f"closing speed at contact ÷ severing speed {VB}; the catch fed the true chord state at {pop['hz']} Hz")
        C.footer(fig, "records/v2/phase6/p6_t0_results.json: excursions[first_severance].runs[fleet], "
                      "tables.fleet.true10.first_severance, verdict  ·  bar: p6_t0_declarations.json")
        # ---- panel (a)
        ax = fig.add_axes([0.19, 0.49, 0.47, 0.32])
        rows = list(RUNS)
        ypos = {k: i for i, k in enumerate(rows)}
        jit = np.linspace(-0.24, 0.24, n)[np.random.default_rng(3).permutation(n)]
        xmax = 8.4
        ax.axvspan(0.0, 1.0, color=C.OK, alpha=0.12, lw=0, zorder=0)
        ax.axvline(1.0, color=C.OK, lw=1.8, ls=(0, (5, 3)), zorder=3)
        ax.text(0.5, -0.66, f"caught\n($v < v_b$)", ha="center", va="top", fontsize=C.FS_TINY, color=C.OK)
        for j, r in enumerate(tally):
            ax.plot([r["ratio"][k] for k in rows], [ypos[k] + jit[j] for k in rows], color=C.FAINT, lw=1.0, zorder=1)
        for j, r in enumerate(tally):
            for k in rows:
                num(f"tally point {r['id'][0]}/{r['id'][1]} {k}: v/v_b", r["ratio"][k], None, "v_b",
                    f"{data['rel_res']} excursions[{r['id'][0]}/{r['id'][1]}].runs[fleet, {RUNS[k][0]}, {RUNS[k][1]}].ratio",
                    "drawn as a point, not printed")
        for k in rows:
            vals = np.array([r["ratio"][k] for r in tally])
            ys = ypos[k] + jit
            col = C_RUN[k]
            if k == "reverse":
                full = np.array([r["full_astern"] for r in tally])
                ax.plot(vals[full], ys[full], ls="none", marker="s", ms=9, mfc=col, mec=col, zorder=4)
                ax.plot(vals[~full], ys[~full], ls="none", marker="s", ms=9, mfc="white", mec=col, mew=1.8, zorder=4)
            elif k == "hold":
                ax.plot(vals, ys, ls="none", marker="o", ms=9, mfc=col, mec=col, zorder=4)
            else:
                ax.plot(vals, ys, ls="none", marker="D", ms=8, mfc="white", mec=col, mew=1.8, zorder=4)
            m = data["med"][k]
            ax.plot([m, m], [ypos[k] - 0.38, ypos[k] + 0.38], color=col, lw=3.2, zorder=5, solid_capstyle="butt")
            ax.text(xmax + 0.12, ypos[k], f"median\n{pop['med_' + k]} {VB}", ha="left", va="center", fontsize=C.FS_SMALL,
                    color=col, weight="bold", linespacing=1.1)
        for ident in (CAUGHT, TYPICAL):
            tag = f"{ident[0]}/{ident[1]}"
            j = [i for i, r in enumerate(tally) if r["id"] == ident][0]
            col = C.OK if ident == CAUGHT else C.INK
            for k in rows:
                ax.plot(tally[j]["ratio"][k], ypos[k] + jit[j], ls="none", marker="o", ms=17, mfc="none",
                        mec=col, mew=1.6, zorder=6)
            left = ident == CAUGHT
            ax.annotate(tag, xy=(tally[j]["ratio"]["hold"], ypos["hold"] + jit[j]),
                        xytext=(-14, 0) if left else (-10, 10), textcoords="offset points",
                        ha="right", va="center" if left else "bottom", fontsize=C.FS_TINY,
                        color=col, zorder=7, bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.8))
        jf = [i for i, r in enumerate(tally) if r["id"] == FASTER][0]
        ax.annotate(pop["faster"], xy=(tally[jf]["ratio"]["reverse"], ypos["reverse"] + jit[jf]), xytext=(9, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=C.FS_TINY, color=C.INK, zorder=7)
        ax.set_yticks([ypos[k] for k in rows])
        ax.set_yticklabels(["recorded thrust\n(replay, no catch)", "catch:\nthrust-hold", "catch:\nthrust-reverse"],
                           fontsize=C.FS_SMALL)
        for lab, k in zip(ax.get_yticklabels(), rows):
            lab.set_color(C_RUN[k])
        ax.tick_params(axis="y", length=0, pad=8)
        ax.set_ylim(2.62, -0.8)
        ax.set_xlim(0.0, xmax)
        ax.set_xticks(list(range(9)))
        _register_axis(board, "tally v/v_b axis limits", ax, "x", "v_b")
        ax.tick_params(axis="x", labelsize=C.FS_TINY)
        ax.set_xlabel(f"closing speed at re-engagement ÷ severing speed ($v$ / {VB})", fontsize=C.FS_SMALL)
        ax.grid(axis="x", ls=":")
        handles = [Line2D([], [], ls="none", marker="s", ms=9, mfc=C_RUN["reverse"], mec=C_RUN["reverse"],
                          label=f"reverse reached full astern ({pop['n_sat']} runs; post hoc)"),
                   Line2D([], [], ls="none", marker="s", ms=9, mfc="white", mec=C_RUN["reverse"], mew=1.8,
                          label="never full astern"),
                   Line2D([], [], color=C.FAINT, lw=1.5, label="same excursion"),
                   Line2D([], [], ls="none", marker="o", ms=13, mfc="none", mec=C.INK, mew=1.4, label="shown: 5020/4"),
                   Line2D([], [], ls="none", marker="o", ms=13, mfc="none", mec=C.OK, mew=1.4,
                          label="shown: 5028/0 (caught)")]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.16, 1.0), ncol=5, fontsize=C.FS_TINY + 1,
                  handletextpad=0.4, columnspacing=1.1, borderaxespad=0.2)
        # ---- panel (b)
        axb = fig.add_axes([0.775, 0.49, 0.2, 0.32])
        axb.axvspan(P.ADMISSIBLE_FRACTION, 1.0, color=C.OK, alpha=0.12, lw=0)
        axb.axvline(P.ADMISSIBLE_FRACTION, color=C.ACCENT, lw=2.0, ls=(0, (5, 3)))
        axb.text(P.ADMISSIBLE_FRACTION - 0.03, -0.78, f"pre-declared\nbar {pop['bar']}", ha="right", va="center",
                 fontsize=C.FS_TINY + 1, color=C.ACCENT)
        verdict = data["res"]["verdict"]
        for y, v in enumerate(("hold", "reverse")):
            frac = float(verdict["fractions"][v])
            lo, hi = map(float, verdict["intervals95"][v])
            axb.errorbar(frac, y, xerr=[[frac - lo], [hi - frac]], fmt="o" if v == "hold" else "s", ms=9,
                         color=C_RUN[v], elinewidth=2.0, capsize=5, capthick=2.0)
            axb.text(0.0, y - 0.14, f"{RUN_SHORT[v]}: {pop['caught_' + v]}/{pop['n']}\n95% CI to {pop['hi_' + v]}",
                     ha="left", va="bottom", fontsize=C.FS_TINY + 1, color=C_RUN[v], linespacing=1.15)
        axb.set_yticks([])
        axb.spines["left"].set_visible(False)
        axb.set_ylim(1.6, -1.25)
        axb.set_xlim(-0.03, 1.0)
        axb.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        _register_axis(board, "caught-fraction axis limits", axb, "x", "")
        axb.tick_params(axis="x", labelsize=C.FS_TINY)
        axb.set_xlabel("caught fraction", fontsize=C.FS_SMALL)
        axb.grid(axis="x", ls=":")
        self.panel_b = [axb]
        # ---- notes under panel (a), the key number and the notes under it
        self.note_a = fig.text(
            0.5, 0.393,
            f"Under each variant one excursion ({pop['faster']}, labelled) closes faster than with no catch: "
            f"{pop['faster_replay']} → {pop['faster_hold']} (hold), {pop['faster_reverse']} (reverse) {VB}.\n"
            f"Post hoc, gates nothing: most closing speeds fall, yet the catch's first tension peak (record T_peak_cf) "
            f"exceeds the replay's in {pop['peak_up_hold']} of {pop['n']} (hold), {pop['peak_up_reverse']} of {pop['n']} (reverse).",
            fontsize=C.FS_TINY + 1, color=C.INK, ha="center", va="center", linespacing=1.5)
        self.key = fig.text(0.5, 0.325, f"Both variants caught {pop['caught']} of {pop['n']}, "
                                        f"against a pre-declared bar of {pop['bar']}.",
                            fontsize=C.FS_TITLE, weight="bold", color=C.SLACK, ha="center", va="center")
        self.verdict = fig.text(0.5, 0.275, f"Neither variant reaches the bar, so the campaign did not take the catch further "
                                            f"({pop['outcome']}).",
                                fontsize=C.FS_SUB, color=C.INK, ha="center", va="center")
        self.note1 = fig.text(0.5, 0.232, f"The law was given the true chord state at {pop['hz']} Hz, not an estimator; "
                                          f"fed the true state plus each estimator's recorded error, it catches "
                                          f"{pop['arms_lo']} to {pop['arms_hi']} of {pop['n']}.",
                              fontsize=C.FS_SMALL, color=C.INK, ha="center", va="center")
        self.note2 = fig.text(0.5, 0.18, f"Post hoc, not part of the pre-registered test: thrust-reverse reaches full astern "
                                         f"only on its {pop['n_sat']} fastest closers ({pop['lo_sat']} to {pop['hi_sat']} {VB});\n"
                                         f"its other {pop['n_other']} failures never do. Why the catch fails is not established.",
                              fontsize=C.FS_SMALL, color=C.INK, ha="center", va="center", linespacing=1.4)
        self.level(1)

    def level(self, k: int) -> None:
        for a in self.panel_b:
            a.set_visible(k >= 2)
        self.note_a.set_visible(k >= 1)
        self.key.set_visible(k >= 2)
        self.note1.set_visible(k >= 3)
        self.note2.set_visible(k >= 4)
        self.verdict.set_visible(k >= 5)


def title_card(fig, pop: dict) -> None:
    fig.clf()
    fig.text(0.5, 0.60, f"Mitigation {pop['ordinal']}: a velocity-matching catch", fontsize=C.FS_TITLE + 10,
             weight="bold", color=C.INK, ha="center", va="center")
    fig.text(0.5, 0.495, TITLE_QUESTION, fontsize=C.FS_SUB + 4, color=C.INK, ha="center", va="center")
    fig.text(0.5, 0.435, TITLE_WATCH, fontsize=C.FS_SUB + 2, color=C.MUTED, ha="center", va="center")
    fig.text(0.5, 0.355, TITLE_PROVENANCE, fontsize=C.FS_SMALL, color=C.MUTED, ha="center", va="center")


def title_frames() -> int:
    """Title-card frames: at least TITLE_S, and the reading time of its question and watch-for sentences."""
    words = len(TITLE_QUESTION.split()) + len(TITLE_WATCH.split())
    return int(math.ceil(max(TITLE_S, words / READING_WPS) * C.FPS))


# ---------------------------------------------------------------- timeline

@dataclass
class Seg:
    scene: str
    frames: int
    caption: str | None
    t_start: float | None = None      # sim time of the segment's first frame
    step: float = 0.0                 # sim seconds per frame (0: paused)
    speed: str | None = None
    level: int = 0


def need_frames(text: str) -> int:
    words = len(text.split())
    return int(math.ceil(max(MIN_CAPTION_S, words / READING_WPS) * C.FPS))


def build_timeline(data: dict, board: Board, pop: dict) -> tuple[list[Seg], dict]:
    num = board.num
    src = data["rel_res"]
    A, B = data["ex"][TYPICAL], data["ex"][CAUGHT]
    rb = B["runs"]
    ca = contact_numbers(board, data, TYPICAL)
    cb = contact_numbers(board, data, CAUGHT)
    qa = {k: v[3] for k, v in ca.items()}
    pa = {k: v[5] for k, v in ca.items()}
    qb = {k: v[3] for k, v in cb.items()}
    delay = num("5028/0 hold: contact delayed vs the recording (t_cf - t_up)", float(rb["hold"]["delay"]), ".1f", "s",
                f"{src} excursions[5028/0].runs[fleet, true10, hold].delay", "asserted equal to the replay (1e-9)")
    fmin = num("5028/0 hold and reverse: least thrust commanded", float(rb["hold"]["min_thrust"]), ".0f", "N",
               f"{src} excursions[5028/0].runs[fleet, true10, hold|reverse].min_thrust",
               "identical in both variants; asserted against the reconstructed command")
    board.man.check("5028/0: thrust-hold and thrust-reverse runs are identical (the law never commanded below 0 N)",
                    float(rb["hold"]["min_thrust"]) > 0.0 and rb["hold"]["min_thrust"] == rb["reverse"]["min_thrust"]
                    and np.array_equal(rb["hold"]["e"], rb["reverse"]["e"]),
                    f"least command {rb['hold']['min_thrust']:.1f} N in both")
    # Narration: complete sentences, each saying what is on screen and what it means.  The severing
    # speed v_b is defined in words (A3) before the symbol is used anywhere else.
    # AL states the law in words (held on the run's first frame, beside the law box); A2 says how the
    # two variants differ while the animation shows their thrust commands falling.
    captions = {
        "A0": "In seed 5020, cable 4 is slack in the squall;\nthis is the turnaround, where the slack is deepest.",
        "AL": "The catch lowers thrust while the slack closes faster than a landing profile,\n"
              "a reference speed that tapers to a soft landing at contact.",
        "A1": "From here the whole fleet is re-run\nwith the recorded thrust and with two catch variants.",
        "A2": "Thrust-hold may cut thrust to zero;\nthrust-reverse may also reverse it, down to full astern.",
        "A3": f"All three runs pull the cable taut above the severing speed {VB},\n"
              f"where the snap reaches {pop['tb']} kN: {qa['replay']}, {qa['hold']} and {qa['reverse']} {VB}.",
        "A4": "Braking lowered the closing speed, yet the tension peak\n"
              f"rose from {pa['replay']} to {pa['hold']} and {pa['reverse']} kN.",
        "B0": "In seed 5028, cable 0 is slack after the squall;\nthis is the one excursion the catch saves.",
        "B1": f"Here the catch never commands less than {fmin} N,\nso its two variants run identically.",
        "B2": f"Braking delays the snap by {delay} s\nand lowers the closing speed.",
        "B3": f"The catch brings this cable taut at {qb['hold']} {VB}, below the severing speed;\n"
              f"the recorded thrust gave {qb['replay']} {VB}.",
        "T1": f"Over all {pop['n']} first-severance excursions, the catch lowers the median closing speed\n"
              f"from {pop['med_replay']} {VB} to {pop['med_hold']} (thrust-hold) and {pop['med_reverse']} (thrust-reverse).",
        "T2": f"Yet both variants catch only {pop['caught']} of the {pop['n']};\n"
              f"the pre-declared bar required {pop['bar']} of them.",
        "T3": f"The law was fed the true chord state at {pop['hz']} Hz,\nso this tests the law, not an estimator.",
        "T4": "Why the catch fails is not established; post hoc, thrust-reverse\n"
              f"reaches full astern only on its {pop['n_sat']} fastest closers.",
        "T5": "The catch lowered most closing speeds, yet even given the true state\n"
              f"it saved only {pop['caught']} excursion in {pop['n']}.",
    }
    for key, text in captions.items():
        assert text.count("\n") <= 1 and all(len(line) <= 92 for line in text.split("\n")), key
        assert len(text.split()) <= 24, key               # two lines of narration at most
    assert all(VB not in captions[k] for k in ("A0", "AL", "A1", "A2")), "v_b used before it is defined (A3)"
    segs: list[Seg] = [Seg("title", title_frames(), None)]
    anim = {}
    for tag, key, ex, lead, after in (("A", TYPICAL, A, ("AL",), ("A3", "A4")), ("B", CAUGHT, B, (), ("B3",))):
        t0 = float(ex["x"].start_label)
        t_end = max(float(r["label"][-1]) for r in ex["runs"].values())
        c0, c1, c2 = (captions[f"{tag}{i}"] for i in range(3))
        # Playback speed: the fastest at which the animation shows c2 for its whole reading time and c1
        # for at least MIN_CAPTION_S; whatever reading time c1 still lacks is a paused hold on the
        # animation's first frame (the same instant as c0), before the run starts playing.
        needed = need_frames(c2) + int(round(MIN_CAPTION_S * C.FPS))
        speed = next((s for s in SPEEDS if (t_end - t0) / s * C.FPS + 1 >= needed), SPEEDS[-1])
        n_anim = int(math.ceil((t_end - t0) / speed * C.FPS)) + 1
        assert n_anim >= needed
        step = speed / C.FPS
        label = speed_label(speed)
        n1 = max(n_anim - need_frames(c2), 0)
        n2 = n_anim - n1
        pre = max(need_frames(c1) - n1, 0)
        scene = f"ex{tag}"
        segs.append(Seg(scene, need_frames(c0), c0, t0, 0.0, "paused"))
        segs += [Seg(scene, need_frames(captions[c]), captions[c], t0, 0.0, "paused") for c in lead]
        if pre:
            segs.append(Seg(scene, pre, c1, t0, 0.0, "paused"))
        segs += [Seg(scene, n1, c1, t0, step, label),
                 Seg(scene, n2, c2, t0 + n1 * step, step, label)]
        segs += [Seg(scene, need_frames(captions[c]), captions[c], t_end, 0.0, "paused") for c in after]
        anim[tag] = {"speed": speed, "label": label, "t0": t0, "t_end": t_end, "frames": n_anim}
        num(f"playback speed, excursion {key[0]}/{key[1]}", label, "s", "", "chosen so each caption meets reading time",
            f"{speed:.4f} x real time")
    for i, key in enumerate(("T1", "T2", "T3", "T4"), start=1):
        segs.append(Seg("tally", need_frames(captions[key]), captions[key], level=i))
    segs.append(Seg("tally", max(need_frames(captions["T5"]), int(round(END_HOLD_S * C.FPS))), captions["T5"], level=5))
    # caption timing (consecutive segments with the same caption count together)
    shown: dict = {}
    for s in segs:
        if s.caption:
            shown[s.caption] = shown.get(s.caption, 0) + s.frames
    short = {c: f for c, f in shown.items() if f < need_frames(c)}
    board.man.check("every caption stays on screen >= max(4 s, words / 2.5 s) and has <= 2 lines", not short,
                    f"{len(shown)} captions; shortest margin "
                    f"{min(f - need_frames(c) for c, f in shown.items()) / C.FPS:.2f} s")
    return segs, {"captions": captions, "anim": anim}


# ---------------------------------------------------------------- render

def render(data: dict, man: C.Manifest, board: Board, stills: list[float] | None = None, stills_dir: Path | None = None):
    fig = C.new_frame()
    pop = population_numbers(board, data)
    segs, info = build_timeline(data, board, pop)
    scene_info = {
        "exA": dict(title="The catch: a typical failure",
                    sub=f"Seed 5020, cable 4, squall plateau (fan formation)  ·  recording run: severance scored at "
                        f"{pop['tb']} kN, no cable cut",
                    selection=f"Selection: the lower median of the {pop['n']} thrust-hold ratios v/{VB} ({pop['sel_lo']}; "
                              f"the median {pop['sel_med']} is the midpoint of {pop['sel_lo']} and {pop['sel_hi']})",
                    label_at=(0.33, 0.92), vessel_labels=True),
        "exB": dict(title="The catch: the one excursion it saves",
                    sub=f"Seed 5028, cable 0, after the squall (fan formation)  ·  recording run: severance scored at "
                        f"{pop['tb']} kN, no cable cut",
                    selection=f"Selection: the only one of the {pop['n']} first-severance excursions caught under "
                              "both variants",
                    label_at=(0.62, 0.92), vessel_labels=False),
    }
    total = sum(s.frames for s in segs)
    wanted = None if stills is None else sorted({min(total - 1, int(round(t * C.FPS))) for t in stills})
    w = None
    if stills is None:
        w = C.writer()
        w.extra_args = list(w.extra_args) + ["-threads", "2"]   # shared 16-core box
    state = {"scene": None, "obj": None, "cap": None, "cap_text": None, "clk": None}
    frame = 0
    checked = 0

    def ensure(scene: str):
        if scene == state["scene"]:
            return state["obj"]
        if scene == "title":
            title_card(fig, pop)
            obj = None
        elif scene == "tally":
            obj = TallyScene(fig, data, board, pop)
        else:
            obj = ExcursionScene(fig, data, TYPICAL if scene == "exA" else CAUGHT, board, scene_info[scene], pop)
        state.update(scene=scene, obj=obj, cap=None, cap_text=None, clk=None)
        return obj

    ctx = w.saving(fig, str(CLIP), C.DPI) if w is not None else None
    if ctx is not None:
        ctx.__enter__()
    try:
        for s in segs:
            obj = ensure(s.scene)
            if s.caption != state["cap_text"]:
                if state["cap"] is not None:
                    state["cap"].remove()
                state["cap"] = C.caption(fig, s.caption, y=CAPTION_Y) if s.caption else None
                state["cap_text"] = s.caption
            if s.scene == "tally":
                obj.level(s.level)
            for k in range(s.frames):
                if wanted is not None and frame not in wanted:
                    frame += 1
                    continue
                if isinstance(obj, ExcursionScene):
                    t = min(max(s.t_start + k * s.step, obj.t0), obj.t_end)
                    obj.update(t)
                    if state["clk"] is not None:
                        state["clk"].remove()
                    state["clk"] = C.clock(fig, t, s.speed)
                if k in (0, s.frames - 1):          # layout guard on each segment's first and last frame
                    bad = texts_inside(fig)
                    assert not bad, f"text outside the frame at frame {frame}: {bad}"
                    checked += 1
                if w is not None:
                    w.grab_frame()
                else:
                    stills_dir.mkdir(parents=True, exist_ok=True)
                    fig.savefig(stills_dir / f"still_{frame / C.FPS:06.2f}s.png", dpi=C.DPI)
                frame += 1
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    assert frame == total
    if stills is None:
        man.check("every text drawn on each segment's first and last frame lies inside the 1920 x 1080 frame",
                  True, f"{checked} frames checked")
    return total, info


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stills", type=str, default=None, help="comma-separated clip times (s): render only these")
    parser.add_argument("--stills-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    C.ensure_dirs()
    man = C.Manifest(
        name=NAME,
        title="Mitigation 2: the velocity-matching catch",
        story=("The velocity-matching catch, the paper's second mitigation law, evaluated as the P6-T0 counterfactual: from "
               "the recorded turnaround of a dangerous slack excursion the campaign's integrator re-runs the whole fleet "
               "with the recorded thrust (no catch; reproduces the record) and with the catch in its thrust-hold and "
               "thrust-reverse variants, fed the true chord state at 10 Hz. On the typical failure (seed 5020, cable 4) "
               "all three re-engage above the severing speed, and the catch's first tension peak there is higher than the "
               "replay's, not lower (reported, gates nothing); on the one excursion it saves (5028/0) the tug lands at "
               "0.89 v_b. Over the 16 first-severance excursions it catches 1 of 16 in both variants against a "
               "pre-declared 0.80 bar; the cause is not established."),
    )
    man.selection = ("Population: the 16 first-severance excursions of P6-T0 (each mission's first re-engagement above "
                     "4.5 kN; fleet mode; controller true10 = the verdict table). Shown: (1) the typical failure = the "
                     "lower median of the 16 thrust-hold ratios v/v_b, seed 5020 cable 4 (2.575; the median 2.863 is the "
                     "midpoint of 2.575 and 3.150 (5015/2), both equidistant, the lower taken); (2) the only excursion "
                     "caught under both variants, seed 5028 cable 0. The top-down frames are the recorded state at each "
                     "excursion's turnaround row (the counterfactual's start). Both selections asserted from the record.")
    man.caveats = [
        "Counterfactual, not a Drake re-simulation: tether/campaign/v2/p6_t0.py::integrate (semi-implicit Euler of the "
        "plant's cable, drag, thrust and weather laws at 0.5 ms, fleet mode). Its recorded-thrust replay reproduces the "
        "record (campaign validation, fleet mode: max |dv| 9.4 mm/s over 43 re-engagements); the declared two-mode "
        "validation formally failed in the literal mode, which gates nothing (paper Sec. V).",
        "Vessel headings are prescribed from the record; in the law runs the slack tug's heading is held from the "
        "recorded contact on (declared). The fleet motion of the law runs is not drawn: only cable i's e(t), edot(t) "
        "are traced; the top-down frame is the recorded state at the turnaround, static.",
        "The chord view is a 1-D schematic: only the stern-to-attachment distance L + e is plant state (to scale); the "
        "hull is drawn along the chord, not at its heading, its drawn height is not to scale, and the payload block is "
        "a marker for the attachment point, not the payload's shape (on screen: 'hull and payload block are markers').",
        "The thrust command is not logged by the campaign; it is reconstructed from the traced e, edot with the "
        "campaign's tick functions and catch_thrust, and asserted against the record's min_thrust and engaged_fraction.",
        "The verdict law was fed the TRUE chord state at 10 Hz (controller true10), not an estimator; estimator-fed "
        "arms catch 1-4 of 16. Missions are recording runs: severance is scored at the 4.5 kN stress threshold (a "
        "scoring threshold, the 55th percentile of the per-mission maximum tension, paper Sec. II), no cable is cut; "
        "stated on screen in each excursion's subtitle.",
        "The peak shown is the record's T_peak_cf: the first tension peak of cable i after the counterfactual contact "
        "(at most 1 s), reported by P6-T0 and gating nothing (the verdict is v_cf < v_b). Only the replay's has a "
        "recorded counterpart (tags.T_peak; fleet-mode error at most 0.57 %, paper Sec. V). It is shown because a lower "
        "closing speed did not mean a lower peak: on 5020/4 it is 8.6 kN (replay) vs 10.4 kN (hold) and 9.9 kN "
        "(reverse), and over the 16 the catch's exceeds the replay's in 9 (hold) and 8 (reverse), a post hoc count made "
        "here that the paper does not state. No mechanism is argued.",
        "Why the catch fails is not established (paper Sec. V). The clip shows, labelled post hoc as in the paper, that "
        "thrust-reverse reaches full astern only on its 6 fastest closers (5.1-7.2 v_b) and that its other 9 failures "
        "never do; it does not argue a mechanism. Under each variant one excursion (5011/2) closes faster than with no "
        "catch (paper Fig. catch caption).",
        "On 5028/0 thrust-hold and thrust-reverse are identical (least command 795 N > 0), so the reverse lane repeats hold.",
        "The 5028/0 top-down row is the recorded state; the plant models no body contact (tether/physics/fleet.py; paper "
        "Fig. 1 caption), so hulls overlap there (vessels 1-4; the pairs are listed in the checks). Noted on screen.",
        "Clock = the record's label clock (the plant state at label t is the state at t + 0.5 ms, declared plant timing); "
        "mission time, no warm-up.",
        "The median of the recorded-thrust replay ratios (4.39) is computed here from the per-run ratios (derived, gates "
        "nothing); the catch medians are the record's table values.",
    ]
    data = prepare(man)
    board = Board(man)
    stills = None if args.stills is None else [float(s) for s in args.stills.split(",")]
    if stills is not None:
        assert args.stills_dir is not None, "--stills needs --stills-dir"
    frames, info = render(data, man, board, stills, args.stills_dir)
    man.frames = frames
    print(f"frames {frames} ({frames / C.FPS:.2f} s); playback {info['anim']}")
    if stills is None:
        man.write(CLIP)


if __name__ == "__main__":
    main()
