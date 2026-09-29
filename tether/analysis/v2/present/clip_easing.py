"""Clip C5 ``easing`` -- mitigation 1: ease the fleet's thrust (storyboard section C5).

What it shows
-------------
Six re-simulated v1 Phase 6 **live** missions (the plant cuts a cable whose tension reaches the
T_b^s = 4.5 kN stress threshold), arm N (no supervisor) and arm P (the hazard supervisor fed by the
proposed estimator), side by side on the same seed and weather, followed by the campaign's tally.

Records read (nothing else is put on screen)
--------------------------------------------
* ``records/phase6/cache/phase6_compute.pkl`` -- top-level keys ``breaking`` (T_b^s),
  ``h_crit`` (``["P"]``), ``critical`` (per-vessel severing speeds), ``monitor`` (hazard model,
  horizon, q_L) and ``results``: one dict per (mode, arm, seed) with ``live_first``
  ((time, cable) or None), ``marks`` ((cable, t_up, depth, v_return, T_peak, dwell, u_entry)),
  ``max_tension``, ``supervisor_log`` (every 10th supervisor tick: t, hazard, thrust scale) and
  ``outcome`` (``closure``, ``severances``, ``end_time``, ``docking_error``, ``mark_count``).
  The paired 2x2 table (live mode) is computed from ``results[*]["live_first"]``.
* ``records/phase6/phase6_results.json`` -- ``tests["P6-T2"]["probabilities"]`` (live severance
  per mission), ``tests["P6-T2"]["differences"]["N-P"]`` (paired difference, 95 % interval) and
  ``["L-P"]`` (caveat only), ``tests["P6-T2"]["closures"][arm]`` (``severance_or_closure``,
  ``closure_before_severance``), ``tests["P6-T2"]["verdict"]``, ``tests["P6-T1"]["verdict"]`` and
  ``["probabilities"]`` (recording mode, the same values),
  ``tests["P6-T4"]["cost"]["P"]["docking_error_penalty_m"]`` and ``["mission_time_penalty"]``,
  ``tests["P6-T4"]["verdict"]`` and ``["statement"]`` (the bars), ``declarations["supervisor"]``,
  ``declarations["closure"]``.
* ``tether/campaign/mission.py::MissionSpec`` (T0, squall schedule), ``tether/control/supervisor.py``
  (``THRUST_EASING``, the per-vessel hop-delayed fleet term), ``tether/campaign/phase6.py`` (``TAU``,
  the per-hop latency), ``tether/physics/fleet.py`` (``CLOSURE_CHORD_LENGTH`` = 1 m; "No contact is
  modelled") for labels and on-screen notes.

Replay (mirrors ``tether/campaign/phase6.py::capstone_job``)
-------------------------------------------------------------
The payload is rebuilt exactly as ``phase6.compute`` builds it: ``breaking``, ``h_crit["P"]``,
``critical`` and ``monitor`` from the pkl; ``centre = docking_reference(MissionSpec()).centre``
(a quiet reference mission, re-run here); ``MissionSpec(cable_mode="live",
break_threshold=breaking)``, ``build_mission(spec, seed, hook, supervised=arm != "N")`` with
``hook = phase6.supervisor_hook("P", ...)`` for P and None for N, ``run_to_end``.  The runner
returns what ``capstone_job`` discards: the 10 ms body state, the 1 ms cable log (elongation,
rate, alive), the severances, the marks and the full supervisor log.  Caches:
``Presentation/cache/easing_{N,P}_{6001,6009,6023}.npz`` (and ``easing_centre.npz``).

Asserted before any frame is drawn (``Manifest.check``), for each of the six runs, against the
pkl record with the same (mode="live", arm, seed):
  * ``live_first``: same cable, time within 1 ms (both None when the record has none);
  * number of marks, and every mark's cable, t_up (1 ms) and T_peak (relative 1e-6);
  * ``max_tension`` recomputed from the cached 1 ms log, relative 1e-6;
  * outcome ``closure``, ``severances``, ``end_time``; ``docking_error`` relative 1e-6 (this
    also checks the recomputed docking centre);
  * P only: the cached supervisor log subsampled every 10th tick equals ``supervisor_log``
    (times, hazards and thrust scales to 1e-9), so the thrust-scale bars are the record's;
  * N and P of one seed ran the same MissionSpec (config hash) and the same pre-rolled weather
    (sha256 of ``mission_weather``), and share the 10 ms state clock (one clock for both panels);
  * each first severance's severing sample carries tension k e + c edot >= T_b^s;
  * the selection rule, applied to the record, returns seeds 6001, 6009, 6023.
Also asserted: the tally's counts from the pkl equal P6-T2's probabilities x 60; the 2x2 is
23/2/2/33; every N-only seed is a P run stopped by formation closure before any severance
(the dagger); the declared supervisor (0.6, fleet term, one tau per hop) and closure rules; the
P6-T4 bars; the slack onsets the narration quotes (6009: cables 3 and 4, 6023: all five cables),
equal in N and P and, for 6009, continuous until the event; every caption >= 4 s on screen,
<= 2 lines, <= 2.5 words/s, and every caption that states an event starts after the event is on
screen; every vessel label, in every frame, clear of the other hulls and the payload, nearer its
own hull than any other (0.4 m and 1.35x), and inside the frame; and, on sample frames of every
caption and status change, that no text, legend or axes box overlaps another (measured from the
renderer).  If any check fails the clip stops (AssertionError) and nothing is rendered.

Playback windows end after each run's first severance (or stop) and before any later,
unscored severance; the screen says so and the manifest's caveats say what that leaves out.
Each seed opens on its first frame held (clock: "paused") while the opening caption is read; the
held frames are copies of the first played frame (camera and vessel numbers included).  Captions
are complete sentences; the last tally caption is the clip's end card (its takeaway), held on the
key frame with the key number.

Selection rule
--------------
From the record's paired live table (both sever 23, N only {6009, 6012}, P only {6023, 6042},
neither 33): the first seed where both arms sever (6001) and the first discordant seed of each
kind (6009: only N severs; 6023: only P severs).  Chosen to show each kind of outcome, not
for typicality.

Run:  nice -n 10 python3 -m tether.analysis.v2.present.clip_easing
      (--stills <dir> seg:t ...  writes inspection PNGs; --check runs every check without rendering;
       --replay-only runs the replays)
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from tether.analysis.v2.present import common as C

NAME = "easing"
RECORD_PKL = C.REPO / "records" / "phase6" / "cache" / "phase6_compute.pkl"
RESULTS_JSON = C.REPO / "records" / "phase6" / "phase6_results.json"
PHASE6_CODE = C.REPO / "tether" / "campaign" / "phase6.py"
ARMS = ("N", "P")
MODE = "live"
EXPECTED_SEEDS = (6001, 6009, 6023)
WORKERS = 2
REC_PKL_REL = "records/phase6/cache/phase6_compute.pkl"
REC_JSON_REL = "records/phase6/phase6_results.json"


def cache_path(arm: str, seed: int) -> Path:
    return C.CACHE / f"easing_{arm}_{seed}.npz"


def centre_path() -> Path:
    return C.CACHE / "easing_centre.npz"


# ---------------------------------------------------------------- record


def load_record() -> dict:
    with RECORD_PKL.open("rb") as fh:
        return pickle.load(fh)


def record_table(record: dict) -> dict:
    return {(r["mode"], r["arm"], r["seed"]): r for r in record["results"]}


def paired_table(table: dict, mode: str = MODE) -> dict:
    """Paired first-severance indicator over the record's 60 seeds (N against P)."""
    from tether.campaign.phase6 import SEEDS

    key = "live_first" if mode == "live" else "virtual_first"
    out = {"both": [], "N only": [], "P only": [], "neither": []}
    for s in SEEDS:
        n = table[(mode, "N", s)][key] is not None
        p = table[(mode, "P", s)][key] is not None
        out["both" if n and p else "N only" if n else "P only" if p else "neither"].append(s)
    return out


def select_seeds(pairs: dict) -> tuple[int, int, int]:
    """The stated rule: first seed where both sever, first N-only seed, first P-only seed."""
    return pairs["both"][0], pairs["N only"][0], pairs["P only"][0]


# ---------------------------------------------------------------- replay


def build_payload(arm: str, seed: int, record: dict, centre) -> tuple:
    """``phase6.compute``'s payload for (arm, seed, live) from the record's own constants."""
    breaking = float(record["breaking"])
    h_crit = float(record["h_crit"].get(arm, 1.0))
    critical = np.asarray(record["critical"], dtype=float)
    return (arm, seed, MODE, breaking, h_crit, critical, list(map(float, centre)), dict(record["monitor"]))


def replay_job(payload) -> dict:
    """``capstone_job`` with the state kept: same spec, hook, build and run; returns logs."""
    from tether.campaign.fleet_run import run_to_end
    from tether.campaign.mission import MissionSpec, build_mission, mission_outcome, mission_weather
    from tether.campaign.phase6 import supervisor_hook
    from tether.physics import constants

    arm, seed, mode, breaking, h_crit, critical, centre, monitor_config = payload
    spec = MissionSpec(cable_mode=mode, break_threshold=breaking if mode == "live" else None)
    hook = None if arm == "N" else supervisor_hook(arm, seed, h_crit, critical, monitor_config)
    run = build_mission(spec, seed, hook, supervised=arm != "N",
                        exemptions=frozenset({"oracle_adapter"}) if arm == "O" else frozenset())
    run_to_end(run)
    log = run.fleet.cables.log
    e = log.elongation[: log.count].copy()
    edot = log.rate[: log.count].copy()
    alive = log.alive[: log.count].copy()
    q = np.where((e > 0.0) & alive, constants.CABLE_STIFFNESS * e + constants.CABLE_DAMPING * edot, 0.0)
    severances = list(run.fleet.cables.severances)
    outcome = mission_outcome(run, spec, np.asarray(centre))
    marks = np.array([(m.cable, m.t_up, m.depth, m.v_return, m.T_peak, m.dwell, m.u_entry)
                      for m in run.fleet.cables.reengagements], dtype=float).reshape(-1, 7)
    weather = mission_weather(spec, seed)
    out = {
        "arm": np.array(arm), "seed": np.array(seed), "mode": np.array(mode),
        "config_hash": np.array(spec.config_hash()),
        "weather_sha256": np.array(hashlib.sha256(np.ascontiguousarray(weather).tobytes()).hexdigest()),
        "breaking": np.array(breaking), "h_crit": np.array(h_crit), "critical": np.asarray(critical, float),
        "centre": np.asarray(centre, float), "monitor_config": np.array(json.dumps(monitor_config, sort_keys=True)),
        "state_time": log.state_time[: log.state_count].copy(), "state": log.state[: log.state_count].copy(),
        "event_time": log.event_time[: log.count].copy(), "elongation": e, "rate": edot, "alive": alive,
        "severances": np.array(severances, dtype=float).reshape(-1, 2),       # (cable, time)
        "marks": marks,
        "max_tension": np.array(float(q.max()) if q.size else 0.0),
        "end_time": np.array(outcome.end_time),
        "closure": np.array(outcome.closure if outcome.closure is not None else (np.nan, np.nan), dtype=float),
        "docking_error": np.array(outcome.docking_error),
        "mission_time": np.array(outcome.mission_time),
        "sim_seconds": np.array(float(run.simulator.get_context().get_time())),
        "wall_seconds": np.array(run.wall_seconds),
    }
    sup = run.fleet.extras.get("supervisor")
    if sup is not None:
        out["sup_t"] = np.array([t for t, _, _ in sup.log], dtype=float)
        out["sup_h"] = np.array([h for _, h, _ in sup.log], dtype=float)
        out["sup_s"] = np.array([s for _, _, s in sup.log], dtype=float)
    np.savez_compressed(cache_path(arm, seed), **out)
    return {"arm": arm, "seed": seed, "wall": run.wall_seconds}


def docking_centre() -> np.ndarray:
    """``docking_reference(MissionSpec()).centre`` as ``phase6.compute`` derives it (cached)."""
    p = centre_path()
    if p.exists():
        return np.load(p)["centre"]
    from tether.campaign.mission import MissionSpec, docking_reference

    ref = docking_reference(MissionSpec())
    np.savez(p, centre=ref.centre, seed=ref.seed, config_hash=np.array(ref.config_hash))
    return ref.centre


def ensure_replays(record: dict, seeds) -> None:
    from tether.campaign.common import run_pool

    C.ensure_dirs()
    todo = [(a, s) for s in seeds for a in ARMS if not cache_path(a, s).exists()]
    if not todo:
        return
    # import by the module's real name so the spawned workers can unpickle the job
    from tether.analysis.v2.present.clip_easing import replay_job as job

    centre = docking_centre()
    payloads = [build_payload(a, s, record, centre) for a, s in todo]
    print(f"easing: replaying {len(payloads)} live missions on {WORKERS} workers", flush=True)
    for r in run_pool(job, payloads, WORKERS,
                      progress=lambda d, t: print(f"easing: {d}/{t} replays", flush=True)):
        print(f"  {r['arm']} {r['seed']}: {r['wall']:.0f} wall-s", flush=True)


def load_replay(arm: str, seed: int) -> dict:
    with np.load(cache_path(arm, seed), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def replay_tension(rp: dict) -> np.ndarray:
    """Plant tension law on the cached 1 ms log (the same q as ``capstone_job``)."""
    return C.tension(rp["elongation"], rp["rate"], rp["alive"])


def replay_first(rp: dict):
    sev = rp["severances"]
    if not len(sev):
        return None
    i = int(np.argmin(sev[:, 1]))
    return float(sev[i, 1]), int(sev[i, 0])


# ---------------------------------------------------------------- reproduction checks


def check_replays(man: C.Manifest, record: dict, table: dict, seeds) -> dict:
    """Assert every replay reproduces its campaign record; returns the loaded replays."""
    payload_ref = build_payload("P", seeds[0], record, np.zeros(2))
    replays = {}
    for seed in seeds:
        for arm in ARMS:
            rp = load_replay(arm, seed)
            rec = table[(MODE, arm, seed)]
            tag = f"{arm} {seed} live"
            src = f"replay:{cache_path(arm, seed).relative_to(C.REPO)} vs {REC_PKL_REL} results[(live,{arm},{seed})]"
            # payload constants are the record's own
            ok = (float(rp["breaking"]) == float(record["breaking"]) and str(rp["mode"]) == MODE
                  and np.array_equal(rp["critical"], np.asarray(record["critical"], float))
                  and json.loads(str(rp["monitor_config"])) == dict(record["monitor"])
                  and (arm == "N" or float(rp["h_crit"]) == float(record["h_crit"]["P"])))
            man.check(f"{tag}: payload = record constants (breaking, h_crit, critical, monitor)", ok,
                      f"breaking {float(rp['breaking'])}, h_crit {float(rp['h_crit'])}, monitor {str(rp['monitor_config'])}")
            # first live severance
            got, want = replay_first(rp), rec["live_first"]
            if want is None:
                ok, det = got is None, f"record none, replay {got}"
            else:
                ok = got is not None and got[1] == int(want[1]) and abs(got[0] - float(want[0])) <= 1.0e-3 + 1e-9
                det = f"record ({float(want[0]):.3f} s, cable {int(want[1])}), replay {got}"
            man.check(f"{tag}: live_first reproduced (same cable, time within 1 ms)", ok, det + f"; {src}")
            # marks
            mk, rm = rp["marks"], rec["marks"]
            man.check(f"{tag}: number of marks reproduced", len(mk) == len(rm), f"record {len(rm)}, replay {len(mk)}")
            same = all(int(a[0]) == int(b[0]) and abs(a[1] - b[1]) <= 1e-3 and abs(a[4] - b[4]) <= 1e-6 * max(1.0, abs(b[4]))
                       for a, b in zip(mk, rm))
            man.check(f"{tag}: every mark's cable, t_up (1 ms) and T_peak (rel 1e-6) reproduced", same, f"{len(rm)} marks")
            # max tension from the cached 1 ms log
            qmax = float(replay_tension(rp).max())
            rel = abs(qmax - rec["max_tension"]) / rec["max_tension"]
            man.check(f"{tag}: max_tension reproduced (rel 1e-6)", rel <= 1e-6,
                      f"record {rec['max_tension']:.4f} N, replay {qmax:.4f} N (from cached log), rel {rel:.2e}")
            # outcome
            out = rec["outcome"]
            cl = rp["closure"]
            got_cl = None if np.isnan(cl[0]) else (int(cl[0]), float(cl[1]))
            want_cl = out["closure"]
            ok = (got_cl is None and want_cl is None) or (got_cl is not None and want_cl is not None and got_cl[0] == want_cl[0]
                                                         and abs(got_cl[1] - want_cl[1]) <= 1e-9)
            man.check(f"{tag}: outcome closure reproduced", ok, f"record {want_cl}, replay {got_cl}")
            sev_rec = [(int(c), float(t)) for c, t in out["severances"]]
            sev_rep = [(int(c), float(t)) for c, t in rp["severances"]]
            ok = len(sev_rec) == len(sev_rep) and all(a[0] == b[0] and abs(a[1] - b[1]) <= 1e-9 for a, b in zip(sev_rec, sev_rep))
            man.check(f"{tag}: every severance (cable, time) reproduced", ok, f"record {sev_rec}")
            man.check(f"{tag}: end time reproduced", abs(float(rp["end_time"]) - out["end_time"]) <= 1e-9,
                      f"record {out['end_time']} s, replay {float(rp['end_time'])} s")
            rel = abs(float(rp["docking_error"]) - out["docking_error"]) / out["docking_error"]
            man.check(f"{tag}: docking error reproduced (checks the docking centre)", rel <= 1e-6,
                      f"record {out['docking_error']:.6f} m, replay {float(rp['docking_error']):.6f} m")
            # supervisor log (P): the bars drawn are the record's thrust scales
            if arm == "P":
                slog = rec["supervisor_log"]
                t_r = np.array([x[0] for x in slog]); h_r = np.array([x[1] for x in slog]); s_r = np.array([x[2] for x in slog])
                t_p, h_p, s_p = rp["sup_t"][::10], rp["sup_h"][::10], rp["sup_s"][::10]
                ok = (len(t_r) == len(t_p) and np.allclose(t_r, t_p, atol=1e-9, rtol=0)
                      and np.allclose(h_r, h_p, atol=1e-9, rtol=0) and np.allclose(s_r, s_p, atol=1e-9, rtol=0))
                man.check(f"{tag}: supervisor log (t, hazard, thrust scale) every 10th tick = record supervisor_log", ok,
                          f"{len(t_r)} ticks; min scale {s_r.min():.3f}")
            else:
                man.check(f"{tag}: no supervisor (record has an empty supervisor_log)", len(rec["supervisor_log"]) == 0 and "sup_t" not in rp, "")
            replays[(arm, seed)] = rp
        a, b = replays[("N", seed)], replays[("P", seed)]
        k = min(len(a["state_time"]), len(b["state_time"]))
        man.check(f"seed {seed}: N and P share the 10 ms state clock (one clock drawn for both panels)",
                  np.array_equal(a["state_time"][:k], b["state_time"][:k]), f"{k} common rows")
        man.check(f"seed {seed}: N and P share the MissionSpec (config hash) and the pre-rolled weather (sha256)",
                  str(a["config_hash"]) == str(b["config_hash"]) and str(a["weather_sha256"]) == str(b["weather_sha256"]),
                  f"config {str(a['config_hash'])[:12]}, weather {str(a['weather_sha256'])[:12]}")
    del payload_ref
    return replays


# ---------------------------------------------------------------- per-run facts (registered)


def run_facts(man: C.Manifest, replays: dict, table: dict, seeds) -> dict:
    """First severance / closure of every run, and how long before its cut the severed cable
    was last slack (chord at or below rest length, e <= 0), from the asserted replays."""
    facts = {}
    for seed in seeds:
        for arm in ARMS:
            rp, rec = replays[(arm, seed)], table[(MODE, arm, seed)]
            f = {"first": replay_first(rp), "closure": None, "slack_lag_ms": None, "sev": []}
            cl = rp["closure"]
            if not np.isnan(cl[0]):
                f["closure"] = (float(cl[1]), int(cl[0]))
            et, e = rp["event_time"], rp["elongation"]
            for c, t in rp["severances"]:
                f["sev"].append((float(t), int(c)))
            if f["first"] is not None:
                t, c = f["first"]
                i = int(np.searchsorted(et, t - 1e-9))
                man.check(f"{arm} {seed}: the severing sample's tension (k e + c edot) reaches T_b^s",
                          C.K.CABLE_STIFFNESS * e[i, c] + C.K.CABLE_DAMPING * rp["rate"][i, c] >= float(rp["breaking"]) - 1e-6,
                          f"cable {c} at {t:.3f} s")
                slack = np.flatnonzero(e[:i, c] <= 0.0)
                f["slack_lag_ms"] = round(1e3 * (t - et[slack[-1]])) if slack.size else None
                src = f"{REC_PKL_REL} results[(live,{arm},{seed})].live_first; replay:{cache_path(arm, seed).relative_to(C.REPO)} severances"
                man.value(f"seed {seed} arm {arm}: first severance time", round(t, 3), "s", src, f"cable {c}")
                man.value(f"seed {seed} arm {arm}: first severed cable", c, "", src)
                man.value(f"seed {seed} arm {arm}: severed cable last slack before its cut", f["slack_lag_ms"], "ms",
                          f"replay:{cache_path(arm, seed).relative_to(C.REPO)} elongation (1 ms log)",
                          "derived: time from the last sample with e <= 0 on the severed cable to the cut (a snap on re-engagement)")
            else:
                man.value(f"seed {seed} arm {arm}: no severance", "none", "",
                          f"{REC_PKL_REL} results[(live,{arm},{seed})].live_first = None")
            if f["closure"] is not None:
                t, c = f["closure"]
                man.value(f"seed {seed} arm {arm}: run stopped by formation closure at", round(t, 3), "s",
                          f"{REC_PKL_REL} results[(live,{arm},{seed})].outcome.closure = {rec['outcome']['closure']}",
                          f"cable {c}; closure = a chord below CLOSURE_CHORD_LENGTH = 1.0 m (tether/physics/fleet.py)")
            facts[(arm, seed)] = f
    return facts



# ---------------------------------------------------------------- narration facts (registered)


def slack_onset(rp: dict, cable: int, t_from: float, t_to: float):
    """First 1 ms sample in [t_from, t_to] at which the chord is at or below rest length (e <= 0)
    after a sample with e > 0; None if the cable is not slack then."""
    et, e = rp["event_time"], rp["elongation"][:, cable]
    lo, hi = int(np.searchsorted(et, t_from - 1e-9)), int(np.searchsorted(et, t_to + 1e-9))
    for i in range(max(lo, 1), hi):
        if e[i] <= 0.0 < e[i - 1]:
            return float(et[i]), i
    return None


def slack_through(rp: dict, cable: int, i0: int, t_end: float) -> bool:
    """The cable stays slack (e <= 0 on every 1 ms sample) from sample i0 through t_end."""
    et, e = rp["event_time"], rp["elongation"][:, cable]
    hi = int(np.searchsorted(et, t_end + 1e-9))
    return bool(np.all(e[i0:hi] <= 0.0))


def narration_facts(man: C.Manifest, replays: dict, facts: dict) -> dict:
    """The slack onsets the pre-event captions quote, asserted equal in N and P (same weather)."""
    out = {}
    for seed, seg in SEGMENTS.items():
        if not seg.get("slack_cables"):
            continue
        k_cap = seg["slack_caption"]
        # onsets searched up to the instant on screen when the caption quoting them starts
        cables, t_from, t_cap = seg["slack_cables"], seg["t_a"], sim_at(seg, seg["captions"][k_cap][0])
        on = {}
        for a in ARMS:
            rp = replays[(a, seed)]
            for c in cables:
                o = slack_onset(rp, c, t_from, t_cap)
                man.check(f"seed {seed} {a}: cable {c} goes slack (e <= 0) between {t_from} s and {t_cap} s", o is not None,
                          f"onset {o}")
                on[(a, c)] = o
        for c in cables:
            dn, dp = on[("N", c)][0], on[("P", c)][0]
            man.check(f"seed {seed}: cable {c}'s slack onset is the same in N and P (within 1 ms)", abs(dn - dp) <= 1e-3 + 1e-9,
                      f"N {dn:.3f} s, P {dp:.3f} s")
        latest = max(on[(a, c)][0] for a in ARMS for c in cables)
        man.check(f"seed {seed}: the caption quoting the slack onsets starts after every onset it quotes is on screen",
                  t_cap >= latest - 1e-9, f"caption starts at simulation time {t_cap:.3f} s; latest onset {latest:.3f} s")
        if seg["slack_claim"] == "since":
            # the caption says 'slack since': continuous from the onset until the caption ends, or until the
            # cable re-engages just before its cut (that instant is narrated by the next caption)
            t_end_sim = sim_at(seg, seg["captions"][k_cap + 1][0])
            for a in ARMS:
                rp, f = replays[(a, seed)], facts[(a, seed)]
                for c in cables:
                    end = min(t_end_sim, float(rp["event_time"][-1]))
                    if f["first"] is not None and f["first"][1] == c:
                        end = min(end, f["first"][0] - 2e-3)
                    man.check(f"seed {seed} {a}: cable {c} stays slack from its onset through {end:.3f} s", slack_through(rp, c, on[(a, c)][1], end),
                              "every 1 ms sample has e <= 0")
        vals = {c: on[("N", c)][0] for c in cables}
        src = f"replay:{cache_path('N', seed).relative_to(C.REPO)} and {cache_path('P', seed).relative_to(C.REPO)} elongation (1 ms log)"
        for c, v in vals.items():
            man.value(f"seed {seed}: cable {c} slack onset (both arms)", round(v, 3), "s", src,
                      "derived: first sample with e <= 0 after e > 0; asserted equal in N and P within 1 ms")
        out[seed] = vals
    man.value("seed 6009 caption: cables 3 and 4 slack since", [round(out[6009][3], 1), round(out[6009][4], 1)], "s",
              "the onsets above, rounded to 0.1 s", "derived; continuity until the event asserted")
    man.value("seed 6023 caption: all five cables slack by", round(max(out[6023].values()), 1), "s",
              "the latest of the five onsets above (50.685 s), rounded to 0.1 s", "derived")
    rp = replays[("P", 6023)]
    t_cut = facts[("P", 6023)]["first"][0]
    s_idx = max(int(np.searchsorted(rp["sup_t"], t_cut + 1e-9)) - 1, 0)
    sc = rp["sup_s"][s_idx]
    man.check("seed 6023 P: every vessel's thrust scale is below 1 at the cut ('eased by its supervisor')", bool(np.all(sc < 1.0)),
              f"scales {np.round(sc, 3).tolist()} at {float(rp['sup_t'][s_idx]):.2f} s")
    for (a, seed), f in facts.items():
        t_c = SEGMENTS[seed]["t_c"]
        later = [t for t, _ in f["sev"] if f["first"] is not None and t > f["first"][0] + 1e-9]
        man.check(f"{a} {seed}: the window ends before any later (unscored) severance", all(t > t_c for t in later),
                  f"window ends {t_c} s; later severances {[round(t, 3) for t in later]}")
    return out


def severing_samples(replays: dict, facts: dict) -> dict:
    """Tension at the severing sample of every severance (logged after the cut; not drawn)."""
    inside, outside = [], []
    for (a, seed), f in facts.items():
        rp = replays[(a, seed)]
        for t, c in f["sev"]:
            i = int(np.searchsorted(rp["event_time"], t - 1e-9))
            T = float(C.K.CABLE_STIFFNESS * rp["elongation"][i, c] + C.K.CABLE_DAMPING * rp["rate"][i, c])
            (inside if SEGMENTS[seed]["t_a"] <= t <= SEGMENTS[seed]["t_c"] else outside).append((T, a, seed, c, t))
    return {"inside": inside, "outside": outside}


def _clip_convex(subject, clipper) -> list:
    """Sutherland-Hodgman: the part of convex polygon ``subject`` inside convex CCW ``clipper``."""
    out = [np.asarray(p, float) for p in subject]
    for k in range(len(clipper)):
        a, b = np.asarray(clipper[k], float), np.asarray(clipper[(k + 1) % len(clipper)], float)
        inp, out = out, []
        if not inp:
            break
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        s = inp[-1]
        for e in inp:
            if side(e) >= 0:
                if side(s) < 0:
                    out.append(s + (e - s) * side(s) / (side(s) - side(e)))
                out.append(e)
            elif side(s) >= 0:
                out.append(s + (e - s) * side(s) / (side(s) - side(e)))
            s = e
    return out


def hull_overlaps(replays: dict) -> dict:
    """Largest hull-hull overlap area inside each playback window (no contact is modelled)."""
    out = {}
    for (a, seed), rp in replays.items():
        seg = SEGMENTS[seed]
        st = rp["state_time"]
        worst, span = (0.0, None, None), []
        for i in np.flatnonzero((st >= seg["t_a"] - 1e-9) & (st <= seg["t_c"] + 1e-9)):
            _, vp = C.unpack_state(rp["state"][i])
            polys = [C.body_polygon(v, C.hull()) for v in vp]
            for p in range(5):
                for q in range(p + 1, 5):
                    if np.linalg.norm(vp[p][:2] - vp[q][:2]) > C.L_HULL + 0.2:
                        continue
                    ar = abs(_signed_area(_clip_convex(polys[p], polys[q])))
                    if ar > 1e-3:
                        span.append(float(st[i]))
                        if ar > worst[0]:
                            worst = (ar, float(st[i]), (p, q))
        out[(a, seed)] = {"worst_m2": worst[0], "at": worst[1], "pair": worst[2],
                          "from": min(span) if span else None, "to": max(span) if span else None}
    return out


# ---------------------------------------------------------------- geometry helpers (drawing only)


def _seg_dists(P, A, B) -> np.ndarray:
    """Distances from points P (m, 2) to the segments A[k]-B[k] (k, 2) -> (m, k)."""
    P, A, B = (np.atleast_2d(np.asarray(x, float)) for x in (P, A, B))
    AB = B - A
    AP = P[:, None, :] - A[None, :, :]
    den = np.maximum(np.einsum("kd,kd->k", AB, AB), 1e-12)
    t = np.clip(np.einsum("mkd,kd->mk", AP, AB) / den, 0.0, 1.0)
    D = AP - t[..., None] * AB[None, :, :]
    return np.sqrt(np.einsum("mkd,mkd->mk", D, D))


def _poly_dist(P, poly) -> np.ndarray:
    """Distance from points P (m, 2) to a convex counter-clockwise polygon (0 inside) -> (m,)."""
    P = np.atleast_2d(np.asarray(P, float))
    A = np.asarray(poly, float)
    B = np.roll(A, -1, axis=0)
    d = _seg_dists(P, A, B).min(axis=1)
    cross = ((B[:, 0] - A[:, 0])[None, :] * (P[:, 1:2] - A[None, :, 1])
             - (B[:, 1] - A[:, 1])[None, :] * (P[:, 0:1] - A[None, :, 0]))
    return np.where(np.all(cross >= 0.0, axis=1), 0.0, d)


def _signed_area(poly) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = np.asarray(poly, float).T
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ---------------------------------------------------------------- layout (figure fractions)

SPEEDS = {1.0: "real time", 0.5: "×0.5 slow motion"}
COLS = {"N": 0.035, "P": 0.520}
FLEET = (0.000, 0.437, 0.315, 0.350)        # dx, bottom, width, height of each fleet view
Y_STATUS = 0.426                             # event status box, below each fleet view
BARS = (0.350, 0.548, 0.090, 0.190)
STRIP = (0.030, 0.214, 0.410, 0.106)
Y_SEL, Y_NOTE = 0.868, 0.847                 # selection rule and disclosure lines (small text)
KEY_Y = 0.142                                # the fleet key and the cable-colour key
CAPTION_Y = 0.077                            # common.caption's default 0.068 lets a 2-line box graze the footer
CAM_MARGIN = 2.5                             # m around the bodies' bounding box (room for vessel numbers)
ARM_TITLE = {"N": "N · no supervisor", "P": "P · supervisor fed by the proposed estimator"}
BAR_COLOR = {"N": C.MUTED, "P": C.TAUT}      # the arms' colours, as in the tally
KEY_TAUT = "taut (wider = more tension, up to {cap:.0f} kN)"       # the pretension clip's wording

# per-seed playback: the first frame (t_a) held for ``hold`` s so the opening caption can be read, real time
# from t_a, ×0.5 from t_b to t_c, then a pause; captions keyed to video seconds (the hold included), each with
# the events it states (it must start after they are on screen).  ``slack_caption`` indexes the caption that
# quotes the slack onsets: the onsets are searched between t_a and the simulation time at which that caption
# starts, and (for 'since') the slack is asserted continuous until the next caption starts.
SEGMENTS = {
    6001: {"title": "Seed 6001: both arms sever", "t_a": 46.0, "t_b": 55.0, "t_c": 58.6, "hold": 13.2, "pause": 7.8,
           "captions": [(0.0, "In these live runs, the simulator cuts any cable whose tension\n"
                              "reaches the {Tb} kN stress threshold.", ()),
                        (6.8, "Both panels replay the same seed and weather:\n"
                              "arm N (left) has no supervisor; arm P (right) has one.", ()),
                        (14.4, "Arm P's supervisor eases each vessel's thrust by up to {ease} %\n"
                               "when any vessel forecasts a cable-cutting snap.", ()),
                        (22.0, "Each vessel forecasts with the proposed estimator,\n"
                               "which fuses its neighbours' delayed estimates of the payload.", ()),
                        (28.8, "Both arms still lose cable {c_N} as it snaps taut,\n"
                               "arm N at {t_N} s and arm P at {t_P} s.", ("first:N", "first:P"))]},
    6009: {"title": "Seed 6009: only N severs, as scored", "t_a": 50.0, "t_b": 56.0, "t_c": 58.8, "hold": 2.3, "pause": 12.9,
           "slack_cables": (3, 4), "slack_claim": "since", "slack_caption": 1,
           "captions": [(0.0, "Seed 6009 is the first seed where only arm N loses a cable;\nwatch arm P's fleet.", ()),
                        (6.8, "In both arms, cables 3 and 4 have been slack\nsince {s6009_3} and {s6009_4} s.", ()),
                        (12.8, "Arm N loses cable {c_N} at {t_N} s,\nbut arm P's run had stopped at {closure_P} s.",
                         ("first:N", "stop:P")),
                        (19.6, "It stopped when cable {closure_cable_P}'s two ends came within 1 m,\n"
                               "a formation closure scored as no severance.", ("stop:P",))]},
    6023: {"title": "Seed 6023: only P severs", "t_a": 46.0, "t_b": 51.0, "t_c": 54.0, "hold": 0.5, "pause": 9.3,
           "slack_cables": (0, 1, 2, 3, 4), "slack_claim": None, "slack_caption": 1,
           "captions": [(0.0, "Seed 6023 is the first seed where only arm P loses a cable.", ()),
                        (5.3, "In both arms, all five cables have gone slack by {s6023_all} s.", ()),
                        (10.4, "Arm P, eased by its supervisor, loses cable {c_P} at {t_P} s\nas it snaps taut.", ("first:P",)),
                        (16.8, "Arm N, in the same weather, never loses a cable.", ("nosev:N",))]},
}
TITLE_CARD_S = 8.0
MIN_CAPTION_S = 4.0


def sim_at(seg: dict, t_video: float) -> float:
    """Simulation time of the frame on screen at video time ``t_video`` of one seed's segment."""
    frames = seed_frames(seg)
    return frames[min(int(round(t_video * C.FPS)), len(frames) - 1)][0]


def seed_frames(seg: dict) -> list:
    """(simulation time, speed label, video time) for every frame of one seed's segment."""
    out = [(seg["t_a"], "paused")] * int(round(seg.get("hold", 0.0) * C.FPS))
    n1 = int(round((seg["t_b"] - seg["t_a"]) * C.FPS))
    out += [(seg["t_a"] + k / C.FPS, SPEEDS[1.0]) for k in range(n1)]
    n2 = int(round((seg["t_c"] - seg["t_b"]) / 0.5 * C.FPS))
    out += [(seg["t_b"] + k * 0.5 / C.FPS, SPEEDS[0.5]) for k in range(n2 + 1)]
    out += [(seg["t_c"], "paused")] * int(round(seg["pause"] * C.FPS))
    return [(t, s, k / C.FPS) for k, (t, s) in enumerate(out)]


def caption_at(captions: list, t_video: float) -> str:
    text = captions[0][1]
    for start, txt, *_ in captions:
        if t_video + 1e-9 >= start:
            text = txt
    return text


def check_captions(man: C.Manifest, name: str, captions: list, total: float) -> None:
    for (s0, txt, *_), nxt in zip(captions, captions[1:] + [(total, None)]):
        dur = nxt[0] - s0
        words = len(txt.replace("\n", " ").split())
        lines = txt.count("\n") + 1
        man.check(f"caption timing ({name} @ {s0:.1f} s): >= 4 s on screen, <= 2 lines, <= 2.5 words/s",
                  dur >= MIN_CAPTION_S - 1e-9 and lines <= 2 and words / dur <= 2.5 + 1e-9,
                  f"{dur:.2f} s, {words} words, {lines} lines")


def row_index(rp: dict, t: float):
    """(10 ms state row, stopped?) for simulation time t; a run that stopped holds its last row."""
    st = rp["state_time"]
    stopped = (not np.isnan(rp["closure"][0])) and t >= float(rp["end_time"]) - 1e-9
    return min(C.nearest(st, t), len(st) - 1), stopped


def in_end_pause(t: float, speed: str, t_c: float) -> bool:
    """The frame belongs to the pause after playback (not to the held first frame)."""
    return speed == "paused" and t >= t_c - 1e-9


def event_video_times(frames: list, rps: dict, facts: dict) -> dict:
    """Video time of the first frame on which each run's first severance / stop / 'no severance' note is drawn."""
    out = {}
    t_c = frames[-1][0]
    for a in ARMS:
        rp, f = rps[a], facts[a]
        for k, (t, speed, tv) in enumerate(frames):
            i, stopped = row_index(rp, t)
            ts = float(rp["state_time"][i])
            if f["first"] is not None and f"first:{a}" not in out and f["first"][0] <= ts + 1e-9:
                out[f"first:{a}"] = tv
            if f["closure"] is not None and f"stop:{a}" not in out and stopped:
                out[f"stop:{a}"] = tv
            if f["first"] is None and f["closure"] is None and f"nosev:{a}" not in out and in_end_pause(t, speed, t_c):
                out[f"nosev:{a}"] = tv
    return out


def squall_schedule():
    """Squall envelope breakpoints from MissionSpec (rise, plateau, fall) in seconds."""
    from tether.campaign.mission import MissionSpec

    s = MissionSpec()
    plateau_end = s.squall_centre + 0.5 * s.squall_plateau
    onset = plateau_end - s.squall_plateau - s.squall_rise
    return onset, onset + s.squall_rise, plateau_end, plateau_end + s.squall_rise


# ---------------------------------------------------------------- camera and vessel numbers (own helpers)


def seed_camera(rps: dict, frames: list, geo, n_hold: int = 0) -> dict:
    """One scale for both panels, each centred on its own fleet (moving average of its bounding box).
    The held first frames (``n_hold``) are copies of the first played frame, camera included, so the
    played frames are framed exactly as without the hold."""
    centres, halves = {}, []
    for a in ARMS:
        rows = [row_index(rps[a], t)[0] for t, _, _ in frames[n_hold:]]
        c, h = C.fleet_bounds(rps[a]["state"][rows], geo, margin=CAM_MARGIN)
        cs = C.smooth_camera(c, window=31)
        centres[a] = np.concatenate([np.repeat(cs[:1], n_hold, axis=0), cs], axis=0)
        halves.append(h)
    fig_w, fig_h = C.FIGSIZE
    half = C.fit_aspect(np.max(np.array(halves), axis=0), (FLEET[2] * fig_w) / (FLEET[3] * fig_h))
    return {"centres": centres, "half": half, "px_per_m": FLEET[2] * C.W / (2.0 * half[0])}


# common.draw_fleet puts each number 1 m ahead of the bow, where it can land on a neighbour's hull when
# vessels bunch (6009, 6023).  This plans every number for every frame under hard constraints: clear of
# every other hull and the payload, nearer its own hull than any other (or joined to it by a leader line),
# clear of the other numbers and the scale bar, inside the frame; a number keeps its spot while that spot
# stays admissible and nearly as good (no flicker).  Positions only; the plant state is untouched.
LABEL_RHO_PX = 12.0                      # radius (px) of a circle holding one vessel number at FS_SMALL
LABEL_DIRS = np.deg2rad(np.arange(0.0, 360.0, 22.5))
LABEL_GAPS = (0.15, 0.6, 1.1)            # m from the number's circle to the hull's bounding ellipse
LEADER_GAPS = (1.8, 2.6)                 # farther spots, joined to their hull by a thin leader line
LABEL_ASSOC_M, LABEL_ASSOC_RATIO = 0.4, 1.35   # any other hull at least this much farther than the own hull
LABEL_CLEAR_M = 0.25                     # gap from the number's circle to any other hull or the payload
LEADER_CLEAR_M = 0.15                    # gap from a leader line to any other hull or the payload


def _label_offsets(rho: float):
    offs, gi, lead, start = [], [], [], []
    for g_i, (g, is_leader) in enumerate([(g, False) for g in LABEL_GAPS] + [(g, True) for g in LEADER_GAPS]):
        for th in LABEL_DIRS:
            offs.append([(0.5 * C.L_HULL + rho + g) * np.cos(th), (0.5 * C.B_HULL + rho + g) * np.sin(th)])
            start.append([(0.5 * C.L_HULL + 0.05) * np.cos(th), (0.5 * C.B_HULL + 0.05) * np.sin(th)])
            gi.append(g_i)
            lead.append(is_leader)
    return np.array(offs), np.array(gi, float), np.array(lead), np.array(start)


def scale_bar_box(lo, hi, px_per_m: float) -> np.ndarray:
    """The region common.draw_fleet uses for its 5 m scale bar and its label (CCW rectangle)."""
    span = hi - lo
    x0, y0 = lo[0] + 0.04 * span[0], lo[1] + 0.05 * span[1]
    top = y0 + 0.02 * span[1] + 18.0 / px_per_m
    return np.array([[x0 - 0.3, y0 - 0.4], [x0 + 5.3, y0 - 0.4], [x0 + 5.3, top], [x0 - 0.3, top]])


def label_plan(rp: dict, frames: list, cam: dict, arm: str, geo) -> dict:
    from tether.physics import fleet as F

    rho = LABEL_RHO_PX / cam["px_per_m"]
    offs, gi, lead, start = _label_offsets(rho)
    base_cost = gi + 3.0 * lead
    n = len(frames)
    plan = {"pos": np.zeros((n, 5, 2)), "leader": np.zeros((n, 5), bool), "lead_from": np.zeros((n, 5, 2)),
            "lead_to": np.zeros((n, 5, 2)), "ok": np.ones((n, 5), bool), "d_own": np.zeros((n, 5)),
            "d_other": np.zeros((n, 5)), "clear": np.zeros((n, 5)), "rho": rho, "moves": 0}
    prev = [None] * 5
    half = cam["half"]
    cache = {}
    for k, (t, _, _) in enumerate(frames):
        i_row, _ = row_index(rp, t)
        lo, hi = cam["centres"][arm][k] - half, cam["centres"][arm][k] + half
        key = (i_row, tuple(np.round(lo, 9)))
        if key in cache:
            for name in ("pos", "leader", "lead_from", "lead_to", "ok", "d_own", "d_other", "clear"):
                plan[name][k] = plan[name][cache[key]]
            continue
        cache[key] = k
        lp, vp = C.unpack_state(rp["state"][i_row])
        hulls = [C.body_polygon(v, C.hull()) for v in vp]
        pay = C.body_polygon(lp, F.pentagon_vertices())
        a, b = C.attachment_points(lp, vp, geo)
        sb = scale_bar_box(lo, hi, cam["px_per_m"])
        cand = []
        for i in range(5):
            Q, S = C.body_polygon(vp[i], offs), C.body_polygon(vp[i], start)
            d_own = _poly_dist(Q, hulls[i])
            d_oth = np.min([_poly_dist(Q, hulls[j]) for j in range(5) if j != i], axis=0)
            d_pay = _poly_dist(Q, pay)
            d_cab = _seg_dists(Q, a, b).min(axis=1)
            inside = np.all((Q >= lo + rho + 0.1) & (Q <= hi - rho - 0.1), axis=1)
            assoc = (d_oth >= d_own + LABEL_ASSOC_M) & (d_oth >= LABEL_ASSOC_RATIO * d_own)
            L = np.maximum(np.linalg.norm(Q - S, axis=1), 1e-9)
            E = S + (Q - S) * (1.0 - 0.9 * rho / L)[:, None]
            path = np.ones(len(Q), bool)
            for f in np.linspace(0.0, 1.0, 9):
                P = S + f * (E - S)
                dp = np.min([_poly_dist(P, hulls[j]) for j in range(5) if j != i] + [_poly_dist(P, pay)], axis=0)
                path &= dp >= LEADER_CLEAR_M
            ok = (inside & (d_own >= rho + 0.05) & (d_oth >= rho + LABEL_CLEAR_M) & (d_pay >= rho + LABEL_CLEAR_M)
                  & (d_cab >= 0.6 * rho) & (_poly_dist(Q, sb) >= 0.8 * rho) & np.where(lead, path, assoc))
            cost = base_cost + 1.5 * np.clip(rho + 0.4 - d_cab, 0.0, None)
            cand.append((Q, S, E, ok, cost, d_own, d_oth, np.minimum(d_oth, d_pay) - rho))
        placed = {}
        for i in sorted(range(5), key=lambda j: int(cand[j][3].sum())):          # most constrained first
            Q, S, E, ok, cost, d_own, d_oth, clear = cand[i]
            ok = ok.copy()
            for qj in placed.values():
                ok &= np.linalg.norm(Q - qj, axis=1) >= 2.0 * rho + 0.1
            if ok.any():
                c = np.where(ok, cost, np.inf)
                best = int(np.argmin(c))
                if prev[i] is not None and ok[prev[i]] and c[prev[i]] <= c[best] + 0.8:
                    best = prev[i]
            else:                                                                 # flagged; the check fails
                best = int(np.argmax(np.minimum(d_oth - d_own, clear)))
                plan["ok"][k, i] = False
            if prev[i] is not None and best != prev[i]:
                plan["moves"] += 1
            prev[i] = best
            placed[i] = Q[best]
            plan["pos"][k, i], plan["leader"][k, i] = Q[best], lead[best]
            plan["lead_from"][k, i], plan["lead_to"][k, i] = S[best], E[best]
            plan["d_own"][k, i], plan["d_other"][k, i], plan["clear"][k, i] = d_own[best], d_oth[best], clear[best]
    return plan


def check_labels(man: C.Manifest, seed: int, arm: str, plan: dict, cam: dict) -> None:
    n = plan["ok"].size
    plain = ~plan["leader"]
    margin = (plan["d_other"] - plan["d_own"])[plain]
    ratio = (plan["d_other"] / np.maximum(plan["d_own"], 1e-9))[plain]
    if margin.size == 0:
        margin, ratio = np.array([np.inf]), np.array([np.inf])
    man.check(f"seed {seed} {arm}: every vessel number, in every frame, is clear of the other hulls and the payload "
              f"(>= {LABEL_CLEAR_M} m), nearer its own hull than any other (>= {LABEL_ASSOC_M} m and "
              f"{LABEL_ASSOC_RATIO}x) or joined to it by a leader, and inside the frame",
              bool(plan["ok"].all()) and margin.min() >= LABEL_ASSOC_M - 1e-9 and ratio.min() >= LABEL_ASSOC_RATIO - 1e-9
              and plan["clear"].min() >= LABEL_CLEAR_M - 1e-9,
              f"{n} labels over {plan['ok'].shape[0]} frames; failures {int((~plan['ok']).sum())}; "
              f"smallest margin own/other hull {margin.min():.2f} m, ratio {ratio.min():.2f}; smallest clearance "
              f"{plan['clear'].min():.2f} m; leaders {int(plan['leader'].sum())}; spot changes {plan['moves']}; "
              f"number radius {plan['rho']:.2f} m at {cam['px_per_m']:.1f} px/m")


def draw_labels(ax, plan: dict, k: int) -> np.ndarray:
    for i in range(5):
        if plan["leader"][k, i]:
            ax.plot(*np.stack([plan["lead_from"][k, i], plan["lead_to"][k, i]]).T, color=C.MUTED, lw=0.9, zorder=3.5)
        ax.text(*plan["pos"][k, i], str(i), fontsize=C.FS_SMALL, color=C.INK, ha="center", va="center", zorder=6)
    return plan["pos"][k]


def mark_severed(ax, state_row, geo, alive, label_xy, rho: float, prev_u: dict) -> dict:
    """Replace draw_fleet's severed-cable cross (at the chord midpoint, where it can sit on a neighbouring
    line) by the legend's glyph: a dotted grey chord between the stubs, the cross at its clearest point."""
    from tether.physics import fleet as F

    for ln in [ln for ln in ax.lines if ln.get_marker() == "x"]:
        ln.remove()
    lp, vp = C.unpack_state(state_row)
    a, b = C.attachment_points(lp, vp, geo)
    bodies = [C.body_polygon(v, C.hull()) for v in vp] + [C.body_polygon(lp, F.pentagon_vertices())]
    U = np.linspace(0.24, 0.76, 14)
    out = {}
    for c in np.flatnonzero(~np.asarray(alive, bool)):
        d = b[c] - a[c]
        ax.plot(*np.stack([a[c] + 0.18 * d, a[c] + 0.82 * d]).T, color=C.SEVERED, lw=1.5, ls=(0, (1.0, 1.8)), zorder=3)
        P = a[c] + U[:, None] * d
        clr = np.full(len(U), np.inf)
        others = [j for j in range(5) if j != c and alive[j]]
        if others:
            clr = np.minimum(clr, _seg_dists(P, a[others], b[others]).min(axis=1))
        for body in bodies:
            clr = np.minimum(clr, _poly_dist(P, body))
        clr = np.minimum(clr, np.linalg.norm(P[:, None, :] - np.asarray(label_xy)[None], axis=2).min(axis=1) - rho)
        best = int(np.argmax(clr))
        if c in prev_u:
            j = int(np.argmin(np.abs(U - prev_u[c])))
            if clr[j] >= clr[best] - 0.15:
                best = j
        out[c] = float(U[best])
        m = P[best]
        ax.plot([m[0]], [m[1]], marker="x", ms=10, mew=2.6, color=C.SLACK, zorder=5)
    return out


def fleet_key(fig, y: float, x: float):
    """The fleet view's key: common.cable_key's handles, the width rule worded as common.draw_fleet applies
    it (lw = 1.4 + 5.0 min(T / WIDTH_FULL_N, 1)), and the severed glyph as this clip draws it."""
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=C.TAUT, lw=4, label=KEY_TAUT.format(cap=C.WIDTH_FULL_N / 1e3)),
               Line2D([], [], color=C.SLACK, lw=2, ls=(0, (3.0, 2.2)), label="carrying no tension"),
               Line2D([], [], color=C.SEVERED, lw=1.6, ls=(0, (1.0, 1.8)), marker="x", mec=C.SLACK, ms=10, mew=2.4,
                      label="severed")]
    fig.legend(handles=handles, loc="center left", ncol=3, bbox_to_anchor=(x, y), fontsize=C.FS_TINY + 1,
               handlelength=2.2, columnspacing=1.2, borderaxespad=0.0, borderpad=0.2)


# ---------------------------------------------------------------- the seed scene


class SeedScene:
    """One seed: N | P fleet views, thrust-scale bars and tension strips on one figure."""

    def __init__(self, fig, seed, seg, replays, facts, geometry, footer_text, selection_text, note_text, cam, labels):
        from matplotlib.lines import Line2D

        self.fig, self.seed, self.seg, self.geo = fig, seed, seg, geometry
        self.rp = {a: replays[(a, seed)] for a in ARMS}
        self.facts = {a: facts[(a, seed)] for a in ARMS}
        self.frames = seed_frames(seg)
        self.q = {a: replay_tension(self.rp[a]) for a in ARMS}
        self.cam, self.labels = cam, labels
        fig.clf()
        C.title(fig, seg["title"], "v1 Phase 6 live missions, fan formation, T0 = 1 kN, squall 50–70 s;"
                " both panels: the same seed and weather")
        fig.text(0.03, Y_SEL, selection_text, fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="top")
        fig.text(0.03, Y_NOTE, note_text, fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="top")
        C.footer(fig, footer_text)
        fleet_key(fig, y=KEY_Y, x=0.020)
        fig.legend(handles=[Line2D([], [], color=C.CABLE_COLORS[c], lw=2.6, label=f"cable {c}") for c in range(5)]
                   + [Line2D([], [], color=C.INK, lw=0, marker="x", ms=9, mew=2.4, label="cut (strips)")],
                   loc="center right", ncol=6, bbox_to_anchor=(0.985, KEY_Y), fontsize=C.FS_TINY + 1, handlelength=1.4,
                   columnspacing=1.0, frameon=False, borderaxespad=0.0, borderpad=0.2)
        self.ax_fleet, self.ax_bars, self.ax_strip, self.bars, self.lines, self.cursor = {}, {}, {}, {}, {}, {}
        self.cut_marks = {}
        on, rise_end, plateau_end, off = squall_schedule()
        t_a, t_c = seg["t_a"], seg["t_c"]
        for arm in ARMS:
            x0 = COLS[arm]
            self.ax_fleet[arm] = fig.add_axes([x0 + FLEET[0], FLEET[1], FLEET[2], FLEET[3]])
            ab = fig.add_axes([x0 + BARS[0], BARS[1], BARS[2], BARS[3]])
            ab.set_ylim(0.0, 1.08); ab.set_xlim(-0.6, 4.6)
            ab.set_yticks([0.0, 0.4, 1.0]); ab.set_yticklabels(["0", "0.4", "1"], fontsize=C.FS_TINY)
            ab.set_xticks(range(5)); ab.set_xticklabels([str(i) for i in range(5)], fontsize=C.FS_TINY)
            ab.set_xlabel("vessel", fontsize=C.FS_TINY, labelpad=1)
            ab.set_title("thrust scale", fontsize=C.FS_SMALL, loc="left", pad=6)
            if arm == "P":
                ab.axhline(0.4, color=C.INK, lw=1.0, ls=(0, (2, 2)), zorder=3)
            self.bars[arm] = ab.bar(range(5), np.ones(5), width=0.7, color=BAR_COLOR[arm], edgecolor="none")
            note = ("no supervisor:\nscale 1 on every vessel" if arm == "N" else
                    "dashed: floor 0.4 (eased 60 %)\neach vessel: largest hazard\nit has heard, 0.4 s per hop")
            ab.text(0.5, -0.29, note, transform=ab.transAxes, ha="center", va="top", fontsize=C.FS_TINY, color=C.MUTED)
            self.ax_bars[arm] = ab
            ax = fig.add_axes([x0 + STRIP[0], STRIP[1], STRIP[2], STRIP[3]])
            ax.set_xlim(t_a, t_c); ax.set_ylim(0.0, 6.2)
            ax.set_yticks([0, 2, 4]); ax.tick_params(labelsize=C.FS_TINY)
            ax.set_ylabel("kN", fontsize=C.FS_TINY, labelpad=2)
            ax.set_xlabel("simulation time t (s)", fontsize=C.FS_TINY, labelpad=1)
            ax.axvspan(max(on, t_a), min(rise_end, t_c), color="#eef1f4", lw=0, zorder=0)
            ax.axvspan(max(rise_end, t_a), min(plateau_end, t_c), color="#e1e6eb", lw=0, zorder=0)
            ax.axhline(4.5, color=C.INK, lw=1.0, ls=(0, (4, 2)), alpha=0.85, zorder=1)
            ax.text(t_a + 0.01 * (t_c - t_a), 4.62, "4.5 kN stress threshold: the plant cuts the cable", fontsize=C.FS_TINY,
                    color=C.INK, ha="left", va="bottom")
            ax.set_title("cable tension (1 ms log)", fontsize=C.FS_SMALL, loc="left", pad=4)
            ax.set_title("shaded: squall rising (50–55 s), plateau (55–65 s)", fontsize=C.FS_TINY, loc="right",
                         color=C.MUTED, pad=4)
            self.lines[arm] = [ax.plot([], [], color=C.CABLE_COLORS[c], lw=1.3, zorder=3)[0] for c in range(5)]
            self.cursor[arm] = ax.axvline(t_a, color=C.INK, lw=0.8, alpha=0.5, zorder=4)
            self.cut_marks[arm] = []
            for t, c in self.facts[arm]["sev"]:
                if t_a <= t <= t_c:
                    m, = ax.plot([t], [4.5], marker="x", ms=11, mew=2.6, color=C.INK, zorder=6, visible=False)
                    self.cut_marks[arm].append((t, m))
            self.ax_strip[arm] = ax
        self.clock_txt = None
        self.stop_marked = {a: False for a in ARMS}
        self.status = {a: None for a in ARMS}
        self.cross_u = {a: {} for a in ARMS}

    def draw(self, k):
        from tether.analysis.v2.present.common import draw_fleet

        t, speed, tv = self.frames[k]
        t_clock = float(self.rp["N"]["state_time"][row_index(self.rp["N"], t)[0]])
        for arm in ARMS:
            rp, f = self.rp[arm], self.facts[arm]
            i, stopped = row_index(rp, t)
            ts = float(rp["state_time"][i])
            j = min(int(np.searchsorted(rp["event_time"], ts - 1e-9)), len(rp["event_time"]) - 1)
            ax = self.ax_fleet[arm]
            draw_fleet(ax, rp["state"][i], self.geo, self.q[arm][j], rp["alive"][j],
                       centre=self.cam["centres"][arm][k], half=self.cam["half"], labels=False)
            xy = draw_labels(ax, self.labels[arm], k)
            self.cross_u[arm] = mark_severed(ax, rp["state"][i], self.geo, rp["alive"][j], xy, self.labels[arm]["rho"],
                                             self.cross_u[arm])
            ax.set_title(ARM_TITLE[arm], fontsize=C.FS_SUB, loc="left", pad=6, color=C.INK)
            if arm == "P":        # thrust scale at this instant: the supervisor log's most recent tick
                s_idx = max(int(np.searchsorted(rp["sup_t"], ts + 1e-9)) - 1, 0)
                scale = rp["sup_s"][s_idx]
            else:                 # no supervisor: 1 by construction
                scale = np.ones(5)
            for bar, v in zip(self.bars[arm], scale):
                bar.set_height(float(v))
            et = rp["event_time"]
            lo = int(np.searchsorted(et, self.seg["t_a"] - 1e-9))
            hi = int(np.searchsorted(et, ts + 1e-9))
            for c, ln in enumerate(self.lines[arm]):
                ln.set_data(et[lo:hi], self.q[arm][lo:hi, c] / 1e3)
            self.cursor[arm].set_xdata([ts, ts])
            for tm, m in self.cut_marks[arm]:
                m.set_visible(tm <= ts + 1e-9)
            note = None
            if f["first"] is not None and f["first"][0] <= ts + 1e-9:
                tt, cc = f["first"]
                note = (f"first severance: cable {cc} at t = {tt:.3f} s\n({f['slack_lag_ms']} ms after its chord reached rest length)", C.SLACK)
            if f["closure"] is not None and stopped:
                tt, cc = f["closure"]
                note = (f"run stopped at t = {tt:.3f} s: formation closure\n(cable {cc}'s chord below 1 m), scored as no severance",
                        C.ACCENT)
            if f["first"] is None and f["closure"] is None and in_end_pause(t, speed, self.seg["t_c"]):
                note = (f"no severance in this mission (run to {float(rp['end_time']):.0f} s)", C.OK)
            if self.status[arm] is not None:
                self.status[arm].remove()
                self.status[arm] = None
            if note:
                self.status[arm] = self.fig.text(COLS[arm] + 0.004, Y_STATUS, note[0], ha="left", va="top", fontsize=C.FS_SMALL,
                                                 color=note[1], linespacing=1.25,
                                                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=note[1], lw=1.0))
            if stopped and not self.stop_marked[arm]:
                t_stop = float(rp["state_time"][-1])
                self.ax_strip[arm].axvline(t_stop, color=C.ACCENT, lw=1.2, ls=(0, (3, 2)))
                self.ax_strip[arm].text(t_stop - 0.02, 5.6, "run stopped ", color=C.ACCENT, fontsize=C.FS_TINY, ha="right", va="center")
                self.stop_marked[arm] = True
        if self.clock_txt is not None:
            self.clock_txt.remove()
        self.clock_txt = C.clock(self.fig, t_clock, speed)
        return tv


class Captioner:
    def __init__(self, fig):
        self.fig, self.txt, self.current = fig, None, None

    def set(self, text):
        if text == self.current:
            return
        if self.txt is not None:
            self.txt.remove()
        self.txt = C.caption(self.fig, text, y=CAPTION_Y) if text else None
        self.current = text


# ---------------------------------------------------------------- layout check (measured from the renderer)


def _layout_items(fig) -> list:
    """(name, window box, owner axes index or None, role) for every figure text (its box if boxed), legend,
    and, per axes, its frame, titles, axis labels, tick labels and texts (role 'inner' if anchored inside the
    frame: such a text may overlap its own frame, e.g. a bar's value label, but nothing else)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            p = t.get_bbox_patch()
            items.append((f"text '{t.get_text().splitlines()[0][:34]}'", (p if p is not None else t).get_window_extent(r), None, "fig"))
    for n, lg in enumerate(fig.legends):
        items.append((f"legend {n}", lg.get_window_extent(r), None, "fig"))
    for ai, ax in enumerate(fig.axes):
        fr = ax.get_window_extent(r)
        name = (ax.get_title(loc="left") or ax.get_title() or f"axes {ai}")[:22]
        items.append((f"[{name}] frame", fr, ai, "frame"))
        for t in (ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label):
            if t.get_visible() and t.get_text().strip():
                items.append((f"[{name}] '{t.get_text()[:26]}'", t.get_window_extent(r), ai, "out"))
        (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
        ticks = ([t for t in ax.get_xticklabels() if x0 - 1e-9 <= t.get_position()[0] <= x1 + 1e-9]
                 + [t for t in ax.get_yticklabels() if y0 - 1e-9 <= t.get_position()[1] <= y1 + 1e-9])
        for t in ticks:                        # only ticks inside the view limits are drawn
            if t.get_visible() and t.get_text().strip():
                items.append((f"[{name}] tick '{t.get_text()}'", t.get_window_extent(r), ai, "tick"))
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip():
                bb = t.get_window_extent(r)
                ax_, ay_ = t.get_transform().transform(t.get_position())   # anchored inside its own frame?
                inner = fr.x0 - 0.5 <= ax_ <= fr.x1 + 0.5 and fr.y0 - 0.5 <= ay_ <= fr.y1 + 0.5
                items.append((f"[{name}] '{t.get_text().splitlines()[0][:26]}'", bb, ai, "inner" if inner else "out"))
    return items


def check_layout(man: C.Manifest, fig, tag: str) -> None:
    """No figure text, legend, axes frame, axes title/label/tick label or axes text overlaps another, and all
    lie inside the frame (measured from the renderer).  Exempt: an axes' own texts inside its frame, and its
    tick labels against its own frame (they abut it)."""
    items = _layout_items(fig)
    bad = []
    for n, (na, a, oa, ra) in enumerate(items):
        if a.x0 < -0.5 or a.y0 < -0.5 or a.x1 > C.W + 0.5 or a.y1 > C.H + 0.5:
            bad.append(f"{na} leaves the frame")
        for nb, b, ob, rb in items[n + 1:]:
            if oa is not None and oa == ob and {ra, rb} & {"frame"} and {ra, rb} & {"inner", "tick"}:
                continue
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 1.0 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1.0:
                bad.append(f"{na} overlaps {nb}")
    man.check(f"layout ({tag}): no text, legend, axes frame or axes text overlaps another or leaves the frame", not bad,
              "; ".join(bad) if bad else f"{len(items)} boxes, measured")


# ---------------------------------------------------------------- tally (record numbers)


def tally_numbers(man: C.Manifest, results: dict, pairs: dict, table_ref: dict) -> dict:
    t1, t2, t4 = (results["tests"][k] for k in ("P6-T1", "P6-T2", "P6-T4"))
    n = {"p": t2["probabilities"], "diff": t2["differences"]["N-P"], "closures": t2["closures"],
         "dock": t4["cost"]["P"]["docking_error_penalty_m"], "time": t4["cost"]["P"]["mission_time_penalty"],
         "verdicts": {"P6-T1": t1["verdict"], "P6-T2": t2["verdict"], "P6-T4": t4["verdict"]},
         "lp": t2["differences"]["L-P"], "pairs": pairs, "n_seeds": sum(len(v) for v in pairs.values())}
    J = REC_JSON_REL
    man.check("record: P6-T1 (recording) and P6-T2 (live) probabilities are identical for N and P",
              all(t1["probabilities"][a] == t2["probabilities"][a] for a in ("N", "P")), str(t2["probabilities"]))
    counts = {a: sum(1 for k in ("both", "N only" if a == "N" else "P only") for _ in pairs[k]) for a in ARMS}
    for a in ARMS:
        man.check(f"record: {a}'s severance count from the pkl pairs matches P6-T2's probability",
                  abs(counts[a] / n["n_seeds"] - n["p"][a]) < 1e-12, f"{counts[a]}/{n['n_seeds']} vs {n['p'][a]}")
    n["counts"] = counts
    for sd in pairs["N only"]:
        rec_p = table_ref[(MODE, "P", sd)]
        man.check(f"record: N-only seed {sd} (dagger): P's run stopped by a formation closure before any severance",
                  rec_p["outcome"]["closure"] is not None and rec_p["live_first"] is None and not rec_p["outcome"]["severances"],
                  f"closure {rec_p['outcome']['closure']}, end {rec_p['outcome']['end_time']} s")
    man.value("N-only seeds whose P run stopped by formation closure (dagger)", list(pairs["N only"]), "",
              f"{REC_PKL_REL} results[(live,P,seed)].outcome.closure", "asserted for every N-only seed")
    man.check("record: 2x2 of the paired live table is 23 / 2 / 2 / 33",
              [len(pairs[k]) for k in ("both", "N only", "P only", "neither")] == [23, 2, 2, 33],
              str({k: len(v) for k, v in pairs.items()}))
    man.value("paired seeds", n["n_seeds"], "", f"{REC_PKL_REL} results (live, N and P)", "count of seeds 6001-6060")
    for k in ("both", "N only", "P only", "neither"):
        man.value(f"paired live table: {k}", len(pairs[k]), "seeds", f"{REC_PKL_REL} results[*].live_first (N vs P)",
                  "computed from the record: " + ", ".join(map(str, pairs[k])) if k in ("N only", "P only") else "computed from the record")
    for a, label in (("N", "no supervisor"), ("P", "supervisor, proposed estimator"), ("O", "supervisor, true-state monitor"),
                     ("L", "supervisor, local monitor without fusion")):
        man.value(f"severance per mission, arm {a} ({label})", n["p"][a], "", f"{J} tests.P6-T2.probabilities.{a}",
                  "live; identical in tests.P6-T1.probabilities (recording)")
    for a in ARMS:
        man.value(f"severed missions, arm {a}", f"{counts[a]}/{n['n_seeds']}", "", f"{REC_PKL_REL} results[*].live_first",
                  "computed count; equals P6-T2 probability x 60")
    man.value("paired difference N - P [mean, lo95, hi95]", n["diff"], "", f"{J} tests.P6-T2.differences.N-P")
    man.check("record: L - P excludes zero, so 'fed by the proposed estimator' must qualify 'only resolved effects are costs'",
              n["lp"][1] > 0.0, f"L - P = {n['lp']}")
    man.value("P6-T2 declared margin: lower bound of N - P must exceed", 0.15, "", f"{J} tests.P6-T2.statement (same margins as P6-T1)")
    for a in ARMS:
        man.value(f"severance or formation closure, arm {a}", n["closures"][a]["severance_or_closure"], "",
                  f"{J} tests.P6-T2.closures.{a}.severance_or_closure")
        man.value(f"closures before any severance, arm {a}", n["closures"][a]["closure_before_severance"], "missions",
                  f"{J} tests.P6-T2.closures.{a}.closure_before_severance")
    man.value("docking-error penalty, P vs N [mean, lo95, hi95]", n["dock"], "m", f"{J} tests.P6-T4.cost.P.docking_error_penalty_m",
              "recording campaign, paired against N")
    man.value("mission-time penalty, P vs N [mean, lo95, hi95]", n["time"], "fraction", f"{J} tests.P6-T4.cost.P.mission_time_penalty",
              "shown as +0.2 % [0.05, 0.44] %")
    man.value("P6-T4 declared bar: docking-error penalty upper bound below", 0.1, "m", f"{J} tests.P6-T4.statement")
    man.value("P6-T4 declared bar: mission-time penalty upper bound below", 5, "%", f"{J} tests.P6-T4.statement")
    man.check("record: P6-T4 statement holds the 5 % and 0.1 m bars", "< 5%" in t4["statement"] and "< 0.1 m" in t4["statement"],
              t4["statement"])
    for k, v in n["verdicts"].items():
        man.value(f"{k} verdict", v, "", f"{J} tests.{k}.verdict")
    return n


def draw_tally(fig, n: dict, stage: int, footer_text: str):
    """The campaign tally, built up in stages 0..5 (each stage adds to the previous).

    Left: the paired 2x2 table (stage >= 1), its closure footnote (>= 2), the key number (5).
    Right: severance per mission (N, P; O and L from stage 3), the paired difference with the
    declared margin (>= 1), costs and verdicts (>= 4).  All positions are figure fractions.
    """
    from matplotlib.patches import FancyBboxPatch

    fig.clf()
    C.title(fig, "Over 60 paired seeds, severance does not move",
            f"Live missions, cut at the 4.5 kN stress threshold: arm N against arm P on the same {n['n_seeds']} seeds, "
            f"{min(min(v) for v in n['pairs'].values())}–{max(max(v) for v in n['pairs'].values())}")
    fig.text(0.97, 0.955, "campaign record (not a playback)", fontsize=C.FS_SMALL, color=C.MUTED, ha="right", va="top")
    C.footer(fig, footer_text)
    p, counts, pr = n["p"], n["counts"], n["pairs"]
    fig.text(0.50, 0.83, "severance per mission (first live severance)", fontsize=C.FS_SUB, ha="left", va="bottom")
    ax = fig.add_axes([0.755, 0.585, 0.15, 0.225])
    slots = [("N", "N · no supervisor", 3), ("P", "P · supervisor, proposed estimator", 2),
             ("O", "O · supervisor, true-state monitor", 1), ("L", "L · supervisor, local monitor (no fusion)", 0)]
    for a_, lab, y in slots:
        if a_ in ("O", "L") and stage < 3:
            continue
        col = C.MUTED if a_ == "N" else C.TAUT if a_ == "P" else "#c9ced6"
        ax.barh(y, p[a_], height=0.62, color=col)
        txt = f"{p[a_]:.3f}" + (f"  ({counts[a_]}/{n['n_seeds']})" if a_ in counts else "")
        ax.text(p[a_] + 0.015, y, txt, va="center", ha="left", fontsize=C.FS_BODY, color=C.INK,
                weight="bold" if a_ in ("N", "P") else "normal")
        ax.text(-0.02, y, lab, va="center", ha="right", fontsize=C.FS_SMALL, color=C.INK)
    ax.set_xlim(0, 0.8); ax.set_ylim(-0.5, 3.5)
    ax.set_yticks([]); ax.set_xticks([0, 0.4, 0.8]); ax.tick_params(labelsize=C.FS_TINY)
    ax.spines["left"].set_visible(False)
    if stage >= 1:
        fig.text(0.04, 0.83, "paired table (same seed and weather)", fontsize=C.FS_SUB, ha="left", va="bottom")
        cx = {"P severs": 0.255, "P does not": 0.395}
        cy = {"N severs": 0.695, "N does not": 0.505}
        for lab, x in cx.items():
            fig.text(x, 0.792, lab, ha="center", va="bottom", fontsize=C.FS_SMALL, color=C.MUTED)
        for lab, y in cy.items():
            fig.text(0.180, y, lab, ha="right", va="center", fontsize=C.FS_SMALL, color=C.MUTED)
        cells = {"both": ("P severs", "N severs"), "N only": ("P does not", "N severs"),
                 "P only": ("P severs", "N does not"), "neither": ("P does not", "N does not")}
        shown = {"both": pr["both"][0], "N only": pr["N only"][0], "P only": pr["P only"][0]}
        for k, (col, row) in cells.items():
            x, y = cx[col], cy[row]
            disc = k in ("N only", "P only")
            fig.patches.append(FancyBboxPatch((x - 0.064, y - 0.088), 0.128, 0.176, transform=fig.transFigure,
                                              boxstyle="round,pad=0,rounding_size=0.008", fc="#fff7e6" if disc else "#f3f4f6",
                                              ec=C.ACCENT if disc else "none", lw=1.6))
            fig.text(x, y + 0.038, str(len(pr[k])), ha="center", va="center", fontsize=C.FS_TITLE, weight="bold")
            if disc:
                mark = "†" if (k == "N only" and stage >= 2) else ""
                desc = f"{k}: " + ", ".join(f"{s}{mark}" for s in pr[k])
            else:
                desc = k
            fig.text(x, y - 0.022, desc, ha="center", va="center", fontsize=C.FS_SMALL, color=C.INK)
            if k in shown:
                fig.text(x, y - 0.058, f"shown: {shown[k]}", ha="center", va="center", fontsize=C.FS_TINY, color=C.MUTED)
        if stage >= 2:
            cl = n["closures"]
            fig.text(0.04, 0.385, "† P's run stopped at a formation closure (a chord below 1 m)\n"
                     "   before any severance; the declared rule scores it as no severance.\n"
                     f"   Severance or closure: N {cl['N']['severance_or_closure']:.3f}, P {cl['P']['severance_or_closure']:.3f}\n"
                     f"   (closures before any severance: N {cl['N']['closure_before_severance']}, P {cl['P']['closure_before_severance']})",
                     ha="left", va="top", fontsize=C.FS_SMALL, color=C.ACCENT if stage == 2 else C.INK, linespacing=1.35)
        d, lo, hi = n["diff"]
        fig.text(0.50, 0.515, f"paired difference N − P = {d:.2f}   [{lo:.3f}, {hi:.3f}]  (95 %)".replace("-0.", "−0."),
                 fontsize=C.FS_SMALL, ha="left", va="bottom")
        ax3 = fig.add_axes([0.53, 0.445, 0.25, 0.055])
        ax3.errorbar([d], [0], xerr=[[d - lo], [hi - d]], fmt="o", color=C.TAUT, ms=8, capsize=6, lw=2.2)
        ax3.axvline(0, color=C.MUTED, lw=0.8)
        ax3.axvline(0.15, color=C.SLACK, lw=1.2, ls=(0, (4, 2)))
        ax3.set_xlim(-0.2, 0.3); ax3.set_ylim(-1, 1); ax3.set_yticks([])
        ax3.tick_params(labelsize=C.FS_TINY); ax3.spines["left"].set_visible(False)
        fig.text(0.79, 0.475, "declared margin (P6-T1/T2):\nlower bound must exceed 0.15", fontsize=C.FS_TINY, color=C.SLACK,
                 ha="left", va="center")
    if stage >= 4:
        dm, dlo, dhi = n["dock"]
        tm, tlo, thi = n["time"]
        fig.text(0.52, 0.375, "costs of P against N (recording runs, P6-T4):", fontsize=C.FS_SMALL, color=C.INK, ha="left", va="top")
        fig.text(0.54, 0.338, f"docking error +{dm:.2f} m  [{dlo:.2f}, {dhi:.2f}]   (bar: upper bound < 0.1 m)", fontsize=C.FS_SMALL,
                 color=C.INK, ha="left", va="top")
        fig.text(0.54, 0.303, f"mission time +{100 * tm:.1f} %  [{100 * tlo:.2f}, {100 * thi:.2f}] %   (bar: upper bound < 5 %)", fontsize=C.FS_SMALL,
                 color=C.INK, ha="left", va="top")
        v = n["verdicts"]
        fig.text(0.52, 0.255, f"declared tests:  P6-T1 (virtual) {v['P6-T1']}   P6-T2 (live) {v['P6-T2']}   P6-T4 (cost) {v['P6-T4']}",
                 fontsize=C.FS_SMALL, color=C.SLACK, ha="left", va="top", weight="bold")
    if stage >= 5:
        fig.text(0.04, 0.205, f"Severance per mission was {p['N']:.3f}\n"
                              f"without the supervisor and {p['P']:.3f} with it,\n"
                              "fed by the proposed estimator.",
                 fontsize=C.FS_SUB, weight="bold", color=C.INK, ha="left", va="center", linespacing=1.3,
                 bbox=dict(boxstyle="round,pad=0.55", fc="white", ec=C.TAUT, lw=1.6))
        fig.text(0.52, 0.205, "The campaign ran after the protocol's stop condition, so it is\n"
                              "exploratory: its declared tests carry no gate authority.",
                 fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="center", linespacing=1.3)


TALLY = [  # (stage, seconds, caption); the last entry is the end card: the takeaway, held on the key frame
    (0, 4.8, "Over {n} paired seeds, each arm loses a cable\nin {nN} missions."),
    (1, 8.4, "Seed by seed, the arms differ by {d} (95 % interval {lo} to {hi})\nand disagree on {kN} seeds each way."),
    (2, 8.0, "In both seeds where only arm N severs, a formation closure\nhad stopped arm P's run, scored as no severance."),
    (3, 11.2, "Fed by the true state, the same supervisor gives a per-mission severance probability\n"
              "of {pO}; fed by a local monitor that fuses no neighbour estimates, it gives {pL}."),
    (4, 9.2, "Fed by the proposed estimator, easing's only statistically resolved effects are costs:\n"
             "{dock} m more docking error and {mt} % more mission time."),
    (5, 7.6, "This campaign ran after the pre-registered protocol's stop condition,\n"
             "so it is exploratory and its tests are not confirmatory."),
    (5, 6.8, "Easing the fleet's thrust left severance unchanged:\n{nN} of {n} missions lost a cable in each arm."),
]
TALLY_FOOTER = ("numbers: records/phase6/phase6_results.json tests P6-T1, P6-T2, P6-T4; "
                "paired table: records/phase6/cache/phase6_compute.pkl results (live)")


TITLE_QUESTION = "Does easing the fleet's thrust when a snap is forecast make cable cuts rarer?"
TITLE_WATCH = "Watch whether the eased fleet still loses a cable in the same weather."
TITLE_NOTE = "The runs are re-simulated v1 Phase 6 live missions: arm N has no supervisor, and arm P has one."


def title_card(fig):
    fig.clf()
    fig.text(0.5, 0.62, "Mitigation 1: ease the fleet's thrust", fontsize=34, weight="bold", ha="center", va="center", color=C.INK)
    fig.text(0.5, 0.50, TITLE_QUESTION, fontsize=C.FS_SUB + 2, ha="center", va="center", color=C.INK)
    fig.text(0.5, 0.43, TITLE_WATCH, fontsize=C.FS_SUB + 2, ha="center", va="center", color=C.MUTED)
    fig.text(0.5, 0.33, TITLE_NOTE, fontsize=C.FS_SMALL, ha="center", va="center", color=C.MUTED)
    C.footer(fig, "records/phase6/cache/phase6_compute.pkl, records/phase6/phase6_results.json; "
                  "an exploratory continuation after the protocol's stop condition (Paper §II, §V)")


# ---------------------------------------------------------------- main


def prepare() -> dict:
    """Every check, registration, label plan and caption, before anything is drawn."""
    C.ensure_dirs()
    C.video_style()
    man = C.Manifest(
        name=NAME, title="Mitigation 1: ease the fleet's thrust",
        story=("Arm P's supervisor eases each vessel's thrust by up to 60 % on the largest hazard that vessel has heard "
               "across the fleet (0.4 s per hop). On the same seed and weather, live missions with and without it lose "
               "cables alike: one seed where both sever, one where only N severs (P's run was stopped by a formation "
               "closure, scored as no severance), one where only P severs. Over 60 paired seeds the first-severance "
               "probability is 0.417 in both arms (paired difference 0.00 [-0.067, 0.067]). Fed by the proposed "
               "estimator, in which each vessel fuses its neighbours' delayed estimates of the payload, easing's only "
               "statistically resolved effects are costs (+0.52 m docking error, +0.2 % mission time); fed by a local "
               "monitor that fuses no neighbour estimates, the same supervisor gives a severance probability per mission "
               "of 0.550. The campaign ran after the pre-registered protocol's stop condition, so it is exploratory and "
               "its tests are not confirmatory."))
    for p in (REC_PKL_REL, REC_JSON_REL, "tether/campaign/phase6.py", "tether/control/supervisor.py",
              "tether/campaign/mission.py", "tether/physics/fleet.py", "tether/analysis/v2/present/clip_easing.py"):
        man.source(p)
    record = load_record()
    table = record_table(record)
    results = json.loads(RESULTS_JSON.read_text())
    pairs = paired_table(table)
    seeds = select_seeds(pairs)
    man.check("selection rule applied to the record returns 6001 (both), 6009 (N only), 6023 (P only)",
              tuple(seeds) == EXPECTED_SEEDS, f"{seeds}")
    man.selection = ("From the record's paired live table (records/phase6/cache/phase6_compute.pkl, arms N and P, "
                     "seeds 6001-6060: both sever 23, N only {6009, 6012}, P only {6023, 6042}, neither 33): the first "
                     "seed where both arms sever (6001) and the first discordant seed of each kind (6009: only N severs; "
                     "6023: only P severs). Chosen to show each kind of paired outcome, not for typicality. Playback "
                     "windows: 46-58.6 s (6001), 50-58.8 s (6009), 46-54 s (6023), each ending after the run's first severance or "
                     "stop and before any later (unscored) severance.")
    ensure_replays(record, seeds)
    replays = check_replays(man, record, table, seeds)            # asserts; stops before any drawing
    facts = run_facts(man, replays, table, seeds)
    nfacts = narration_facts(man, replays, facts)
    tn = tally_numbers(man, results, pairs, table)
    decl = results["declarations"]
    man.check("record: the declared supervisor law eases by 0.6 on the fleet term, one tau per hop",
              "0.6" in decl["supervisor"] and "fleet term" in decl["supervisor"] and "one tau per hop" in decl["supervisor"],
              decl["supervisor"])
    man.check("record: the declared closure rule scores a closure before any severance as no severance",
              "formation closure" in decl["closure"] and "counts as no severance" in decl["closure"], decl["closure"])
    man.value("closure before any severance is scored as no severance", "rule", "", f"{REC_JSON_REL} declarations.closure",
              "the words 'scored (by the protocol) as no severance' on screen")
    from tether.campaign import phase6 as P6
    from tether.control import supervisor as SUP
    code6 = PHASE6_CODE.read_text()
    man.check("code: THRUST_EASING = 0.6; phase6.supervisor_hook builds Supervisor(n, h_crit, TAU) with the default "
              "fleet_term=True; TAU = 0.4 s", SUP.THRUST_EASING == 0.6 and "Supervisor(n, h_crit, TAU)" in code6 and P6.TAU == 0.4
              and SUP.Supervisor.__init__.__defaults__[0] is True, f"THRUST_EASING {SUP.THRUST_EASING}, TAU {P6.TAU}")
    man.value("maximum easing", 60, "%", f"{REC_JSON_REL} declarations.supervisor; tether/control/supervisor.py THRUST_EASING",
              "F_T,i <- F_T,i (1 - 0.6 sat(h/h_crit)), fleet term on")
    man.value("hazard delay per hop on the communication ring ('0.4 s per hop')", P6.TAU, "s",
              "tether/campaign/phase6.py TAU; tether/control/supervisor.py (vessel i uses max(h_i(t), max_j h_j(t - d_ij tau)))",
              f"{REC_JSON_REL} declarations.supervisor: 'one tau per hop'")
    man.value("thrust-scale floor (bars' dashed line)", 0.4, "", "1 - THRUST_EASING (tether/control/supervisor.py)")
    man.value("stress threshold T_b^s (plant cuts the cable)", float(record["breaking"]) / 1e3, "kN", f"{REC_PKL_REL} breaking",
              "Paper §II: a stress threshold, the 55th percentile of the per-mission maximum tension")
    from tether.campaign.mission import MissionSpec
    ms = MissionSpec()
    man.value("pretension T0", ms.pretension / 1e3, "kN", "tether/campaign/mission.py MissionSpec.pretension")
    on, rise_end, plateau_end, off = squall_schedule()
    man.value("squall rise / plateau / end", [on, rise_end, plateau_end, off], "s",
              "tether/campaign/mission.py MissionSpec squall_centre/rise/plateau (squall_envelope)")
    man.value("closure chord length", 1.0, "m", "tether/physics/fleet.py CLOSURE_CHORD_LENGTH")
    from tether.physics import fleet as F
    man.check("code: the plant models no contact between bodies (fleet.py), so the on-screen 'hulls may overlap' holds",
              "No contact is modelled" in (C.REPO / "tether/physics/fleet.py").read_text() and F.CLOSURE_CHORD_LENGTH == 1.0, "")
    man.value("seeds shown (selection rule)", list(seeds), "", f"{REC_PKL_REL} results (live): first of 'both', 'N only', 'P only'")
    man.value("seed range of the paired campaign", [min(min(v) for v in pairs.values()), max(max(v) for v in pairs.values())], "",
              "tether/campaign/phase6.py SEEDS (6001-6060)")
    man.value("mission length ('run to 130 s')", ms.duration, "s", "tether/campaign/mission.py MissionSpec.duration; "
              f"{REC_PKL_REL} results[(live,N,6023)].outcome.end_time = 130.0")
    man.value("interval level of the paired difference and cost intervals", 95, "%",
              "tether/campaign/phase6.py analyse (_paired, seed_bootstrap_statistic, n_boot=4000); "
              "tether/evt/bootstrap.py default confidence=0.95 (percentile interval over seeds)")
    man.value("thrust scale shown for arm N", 1.0, "", "tether/campaign/phase6.py capstone_job: hook None, supervised False for N")
    man.value("thrust scale bars, arm P (animated)", "series", "",
              "replay:Presentation/cache/easing_P_<seed>.npz sup_s (10 ms supervisor log; every 10th tick asserted = record supervisor_log)")
    man.value("cable tension strips and cable widths (animated)", "series", "N",
              "replay:Presentation/cache/easing_<arm>_<seed>.npz elongation/rate/alive (1 ms log), plant law common.tension")
    man.value("simulation clock", "series", "s", "replay state_time (10 ms log; the mission clock starts at 0, no warm-up)")
    man.value("scale bar", 5, "m", "common.draw_fleet")
    man.value("cable width cap in the key", C.WIDTH_FULL_N / 1e3, "kN", "common.WIDTH_FULL_N (draw_fleet's width law)")
    man.value("playback speeds", ["real time", "x0.5 slow motion", "paused"], "", "this module SEGMENTS")
    sev = severing_samples(replays, facts)
    ov = hull_overlaps(replays)
    ov_runs = sorted((k for k, v in ov.items() if v["worst_m2"] > 0), key=lambda k: (k[1], k[0]))
    w = max(ov.items(), key=lambda kv: kv[1]["worst_m2"])
    lp_ = tn["lp"]
    man.caveats = [
        "Six missions of 60, chosen by the stated rule to show each kind of paired outcome; they are not typical and "
        "show no effect of easing either way. The verdict is the 60-seed tally.",
        "T_b^s = 4.5 kN is the campaign's stress threshold (55th percentile of the per-mission maximum tension), not a "
        "certified break force; the screen calls it a stress threshold.",
        "In seed 6009 (and 6012) P's 'no severance' is a formation closure (a chord below 1 m) that stopped the run; the "
        "declared rule scores it as no severance. Over 60 seeds P has 4 closures before severance, N 2; severance or "
        "closure 0.483 vs 0.450. That 0.450 (this 60-seed set, severance or closure) is not the 0.45 unsupervised severance "
        "rate of the 40-mission set quoted in Paper §V and possibly by the catch clip; the 0.417 here is this set's severance.",
        f"The supervisor eases every vessel: each on the largest hazard it has heard, its own at once and another vessel's "
        f"delayed one tau = {P6.TAU} s per hop on the communication ring (fleet term on); it is not a single-vessel law, "
        "and the bars differ between vessels for that reason. The hazards themselves are not shown.",
        "Only the first severance per mission is scored; the screen says each window ends before any later one. In 6001 N "
        "loses cables 0, 1, 2 at 58.658-58.735 s and cable 4 at 58.993 s, P loses cable 4 at 59.099 s (outcome.severances); "
        "6009 N loses cable 4 at 60.427 s. The 6001 5-vs-2 contrast is atypical and untested: over the 60 live seeds the "
        "record holds 101 severances for N and 100 for P (post-hoc count, not shown on screen).",
        "Costs (+0.52 m docking error, +0.2 % mission time) are from the recording campaign, paired against N (P6-T4). They "
        f"are easing's only resolved effects when it is fed by the proposed estimator; fed by the local monitor, severance "
        f"rises: L - P = {lp_[0]:+.3f} [{lp_[1]:.3f}, {lp_[2]:.3f}] ({REC_JSON_REL} tests.P6-T2.differences.L-P; not on screen).",
        "The easing campaign is an exploratory continuation after the protocol's stop condition; its tests carry no gate "
        "authority (Paper §II).",
        "The tension strips plot the logged, alive-masked tension; a cut is marked by a cross on the 4.5 kN line, since the "
        "severing sample is logged after the cut. The first severances shown carry "
        f"{min(s[0] for s in sev['inside']) / 1e3:.2f}-{max(s[0] for s in sev['inside']) / 1e3:.2f} kN at that sample; "
        f"later severances outside the windows reach {max(s[0] for s in sev['outside']) / 1e3:.1f} kN. In 6009 N cable 3 jumps "
        "from slack to over 4.5 kN within one 1 ms sample, so its strip shows only the cross.",
        "The plant models no contact between bodies (tether/physics/fleet.py), so hulls may overlap; the screen says so. "
        f"Overlaps occur inside the windows of {len(ov_runs)} of the 6 runs "
        f"({', '.join(f'{a} {s}' for a, s in ov_runs)}); the largest is {w[1]['worst_m2']:.2f} m2 of a "
        f"{abs(_signed_area(C.hull())):.3f} m2 hull (vessels {w[1]['pair'][0]} and {w[1]['pair'][1]}, {w[0][0]} {w[0][1]}, "
        f"t = {w[1]['at']:.2f} s; derived from the replays' 10 ms state).",
        "Vessel numbers are placed by this module (not common.draw_fleet) under asserted clearance rules; the severed "
        "cable is drawn as grey stubs joined by a dotted chord with the cross at its clearest point (the key's glyph). "
        "Cable colours in the strips follow common.CABLE_COLORS (shared across clips), so cable 3 is the same red as "
        "'carrying no tension'; the threshold line and cut marks are drawn in ink to keep them apart.",
    ]
    man.check("hull-hull overlap measured from the replays (caveat numbers)", w[1]["worst_m2"] > 0.0,
              f"{len(ov_runs)} runs; largest {w[1]['worst_m2']:.2f} m2 in {w[0]}")
    footer_seed = ("plant state: replay of records/phase6 live missions (capstone_job construction), asserted to reproduce "
                   "phase6_compute.pkl: first severance, marks, max tension, supervisor log")
    selection_text = ("Selection (stated rule, from the record's paired table): first seed where both arms sever (6001), "
                      "first where only N severs (6009), first where only P severs (6023); not chosen for typicality.")
    note_text = ("Only each mission's first severance is scored; every window ends before any later severance. "
                 "The plant models no contact between bodies, so hulls may overlap.")
    geo = C.mission_geometry()
    captions, cams, labels, events = {}, {}, {}, {}
    for s_ in seeds:
        seg = SEGMENTS[s_]
        frames = seed_frames(seg)
        fill = {"Tb": f"{float(record['breaking']) / 1e3:.1f}", "ease": f"{100 * SUP.THRUST_EASING:.0f}",
                "s6009_3": f"{nfacts[6009][3]:.1f}", "s6009_4": f"{nfacts[6009][4]:.1f}",
                "s6023_all": f"{max(nfacts[6023].values()):.1f}"}
        for a in ARMS:
            f = facts[(a, s_)]
            if f["first"]:
                fill[f"t_{a}"], fill[f"c_{a}"] = f"{f['first'][0]:.3f}", f"{f['first'][1]}"
            if f["closure"]:
                fill[f"closure_{a}"], fill[f"closure_cable_{a}"] = f"{f['closure'][0]:.3f}", f"{f['closure'][1]}"
        if s_ == 6001:
            man.check("seed 6001: both arms lose the same cable first (the caption names one cable)",
                      facts[("N", s_)]["first"][1] == facts[("P", s_)]["first"][1], "")
        caps = [(st, txt.format(**fill), ev) for st, txt, ev in seg["captions"]]
        total = len(frames) / C.FPS
        check_captions(man, f"seed {s_}", caps, total)
        ev_t = event_video_times(frames, {a: replays[(a, s_)] for a in ARMS}, {a: facts[(a, s_)] for a in ARMS})
        for st, txt, ev in caps:
            if ev:
                man.check(f"seed {s_}: the caption stating {', '.join(ev)} starts after those events are on screen",
                          all(e in ev_t and st >= ev_t[e] - 1e-9 for e in ev),
                          f"caption at {st:.2f} s; events at " + ", ".join(f"{e} {ev_t.get(e, float('nan')):.2f} s" for e in ev))
        captions[s_], events[s_] = caps, ev_t
        cams[s_] = seed_camera({a: replays[(a, s_)] for a in ARMS}, frames, geo, n_hold=int(round(seg.get("hold", 0.0) * C.FPS)))
        for a in ARMS:
            labels[(a, s_)] = label_plan(replays[(a, s_)], frames, cams[s_], a, geo)
            check_labels(man, s_, a, labels[(a, s_)], cams[s_])
    lags = []
    for s_ in seeds:
        for st, _, ev in captions[s_]:
            if ev:
                lags.append(f"{s_}: " + ", ".join(f"{e} +{st - events[s_][e]:.2f} s" for e in ev))
    man.caveats.append("Captions that state an event start after the event is on screen (asserted); narration lag behind "
                       "each event, in video seconds: " + "; ".join(lags) + ".")
    man.caveats.append("Each seed opens on its first frame held for reading time (" + ", ".join(
        f"{s_}: {SEGMENTS[s_]['hold']:.1f} s at t = {SEGMENTS[s_]['t_a']:.0f} s" for s_ in seeds)
        + "; the clock reads 'paused'); the held frames are copies of the first played frame.")
    d, lo, hi = tn["diff"]
    tally_caps, t0 = [], 0.0
    for stage, dur, txt in TALLY:
        txt = txt.format(n=tn["n_seeds"], nN=tn["counts"]["N"], nP=tn["counts"]["P"], d=f"{d:.2f}",
                         lo=f"{lo:.3f}".replace("-", "−"), hi=f"{hi:.3f}", kN=len(pairs["N only"]), kP=len(pairs["P only"]),
                         pO=f"{tn['p']['O']:.3f}", pL=f"{tn['p']['L']:.3f}", dock=f"{tn['dock'][0]:.2f}",
                         mt=f"{100 * tn['time'][0]:.1f}")
        tally_caps.append((t0, txt, stage, dur))
        t0 += dur
    check_captions(man, "tally", [(a, b) for a, b, _, _ in tally_caps], t0)
    man.check("tally wording: 'disagree on k seeds each way' needs as many N-only as P-only seeds",
              len(pairs["N only"]) == len(pairs["P only"]), f"N only {len(pairs['N only'])}, P only {len(pairs['P only'])}")
    man.check("tally and end-card wording ('each arm loses a cable in k missions') needs equal counts in N and P",
              tn["counts"]["N"] == tn["counts"]["P"], f"N {tn['counts']['N']}, P {tn['counts']['P']}")
    man.value("mission-time penalty shown", round(100 * tn["time"][0], 1), "%", f"{REC_JSON_REL} tests.P6-T4.cost.P.mission_time_penalty[0]")
    return {"man": man, "seeds": seeds, "replays": replays, "facts": facts, "tn": tn, "captions": captions,
            "tally_caps": tally_caps, "footer_seed": footer_seed, "selection_text": selection_text, "note_text": note_text,
            "cams": cams, "labels": labels, "events": events, "geo": geo}


def make_scene(fig, ctx: dict, s: int) -> SeedScene:
    return SeedScene(fig, s, SEGMENTS[s], ctx["replays"], ctx["facts"], ctx["geo"], ctx["footer_seed"],
                     ctx["selection_text"], ctx["note_text"], ctx["cams"][s], {a: ctx["labels"][(a, s)] for a in ARMS})


def preflight_layout(ctx: dict) -> None:
    """Draw (to the figure only; nothing is written) the frames where the layout changes -- each segment's
    first frame, every caption start, every status change, the last frame, each tally stage, the title card --
    and assert that no text, legend or axes box overlaps another."""
    import matplotlib.pyplot as plt

    man = ctx["man"]
    fig = C.new_frame()
    title_card(fig)
    check_layout(man, fig, "title card")
    for s in ctx["seeds"]:
        scene = make_scene(fig, ctx, s)
        cap = Captioner(fig)
        n = len(scene.frames)
        picks = {0, n - 1} | {min(int(round(st * C.FPS)), n - 1) for st, _, _ in ctx["captions"][s]}
        picks |= {min(int(round(tv * C.FPS)), n - 1) for tv in ctx["events"][s].values()}
        for k in sorted(picks):                # the layout at a frame depends only on that frame (and on the stop
            scene.draw(k)                      # line, drawn once a stopped frame has been drawn)
            cap.set(caption_at(ctx["captions"][s], scene.frames[k][2]))
            check_layout(man, fig, f"seed {s}, video {scene.frames[k][2]:.2f} s")
        del scene
    for t_start, txt, stage, dur in ctx["tally_caps"]:
        draw_tally(fig, ctx["tn"], stage, TALLY_FOOTER)
        Captioner(fig).set(txt)
        check_layout(man, fig, f"tally stage {stage}")
    plt.close(fig)
    C.CAPTION_LOG.clear()


def render(ctx: dict) -> None:
    """Stream every frame to ffmpeg (no frame images on disk); runs only after prepare() and the layout preflight."""
    import matplotlib.pyplot as plt

    man = ctx["man"]
    fig = C.new_frame()
    out = C.CLIPS / f"{NAME}.mp4"
    w = C.writer()
    frames = 0
    with w.saving(fig, str(out), dpi=C.DPI):
        title_card(fig)
        frames += C.hold(w, TITLE_CARD_S)
        for s in ctx["seeds"]:
            scene = make_scene(fig, ctx, s)
            cap = Captioner(fig)
            for k in range(len(scene.frames)):
                tv = scene.draw(k)
                cap.set(caption_at(ctx["captions"][s], tv))
                w.grab_frame()
                frames += 1
            print(f"easing: seed {s} rendered ({len(scene.frames)} frames)", flush=True)
        for t_start, txt, stage, dur in ctx["tally_caps"]:
            draw_tally(fig, ctx["tn"], stage, TALLY_FOOTER)
            Captioner(fig).set(txt)
            frames += C.hold(w, dur)
        print("easing: tally rendered", flush=True)
    plt.close(fig)
    man.frames = frames
    man.check("frames written = frames counted by common's writer", frames == C._FRAMES[0], f"{frames} vs {C._FRAMES[0]}")
    man.write(out)


def stills(ctx: dict, outdir: Path, picks: list) -> None:
    """Inspection stills (a handful of PNGs, scratch only): picks = [(segment, video seconds)]."""
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    fig = C.new_frame()
    for seg_key, tv in picks:
        if seg_key == "title":
            title_card(fig)
        elif seg_key == "tally":
            row = [r for r in ctx["tally_caps"] if r[0] <= tv][-1]
            draw_tally(fig, ctx["tn"], row[2], TALLY_FOOTER)
            Captioner(fig).set(row[1])
        else:
            s = int(seg_key)
            scene = make_scene(fig, ctx, s)
            k = min(int(round(tv * C.FPS)), len(scene.frames) - 1)
            scene.draw(k)
            Captioner(fig).set(caption_at(ctx["captions"][s], scene.frames[k][2]))
        fig.savefig(outdir / f"still_{seg_key}_{tv:05.1f}.png", dpi=C.DPI)
    plt.close(fig)


def main() -> None:
    ctx = prepare()
    preflight_layout(ctx)
    render(ctx)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--replay-only":
        ensure_replays(load_record(), EXPECTED_SEEDS)
    elif len(sys.argv) > 1 and sys.argv[1] == "--check":
        _ctx = prepare()
        preflight_layout(_ctx)
        print(f"easing: {len(_ctx['man'].checks)} checks passed; {len(_ctx['man'].values)} values registered")
    elif len(sys.argv) > 2 and sys.argv[1] == "--stills":
        # e.g. --stills /tmp/.../easing 6001:14.9 tally:30 title:1
        stills(prepare(), Path(sys.argv[2]), [(a.split(":")[0], float(a.split(":")[1])) for a in sys.argv[3:]])
    else:
        main()
