"""C7 ``pretension`` -- design time: one knob, two requirements (Presentation/STORYBOARD.md, C7).

What the clip shows
-------------------
Seed 7105 of the Phase 1(f) stochastic cells, run in two cells that differ only in pretension
and, by the declared schedule, heading gain -- the same weather realisation, the same sensor
noise, the same parallel formation:

  * ``i035_T0600_ks3``: T0 = 600 N,  k_h = 477 N m/rad (left panel);
  * ``i035_T1000_ks3``: T0 = 1000 N, k_h = 763 N m/rad (right panel).

Both at weather intensity 0.35, k_sigma = 3, sway limit 0.349 rad, linear drag, recording cables
(severance is scored, the plant never cuts), 20 s warm-up + 300 s.  A window of the simulation
clock is played side by side with running counters, then the cell-level numbers of the record.

Records read (path -> keys)
---------------------------
* ``records/v2/phase1/stochastic_declarations.json``
    ``cells.grid[name == cell]``: T0_N, intensity, k_h_N_m_per_rad, k_sigma, sway_limit_rad,
    config_hash; ``cells``: formation, drag_law, cable_mode, duration_s, warmup_s,
    trim_gain_N_m_per_rad, weather; ``cells.heading_gain_rule``; ``tests.P1-T11.threshold``.
* ``records/v2/phase1/stochastic_runs.json``
    ``runs[cell == c, seed == 7105]``: config_hash, end_time, closure.terminal, n_marks,
    marks.{t_up, T_peak, cable}, onsets_count, onsets_per_cable, slack_samples,
    shape_samples, shape_sums.chord_world.{cos, sin, count}, shape_std_deg.chord_world,
    clean.{q_count, q_sum, q_square, exposure_s}, diagnostics.weather_bit_equal, pilot.
* ``records/v2/phase1/stochastic_results.json``
    ``P1_T11[cell].{chord_world_stat_deg, psi_stat_deg, chord_world_meets_15}`` for all eight
    cells (35.9 / 20.9 deg; no cell meets the band; psi below chord in every cell); ``trade_off[cell == c].{marks,
    chord_world_std_deg, meets_population_rule_in_some_class}`` (510 / 4);
    ``closest_to_both.least_far`` and ``closest_to_both.ranked[0].{cell, combined,
    shape_shortfall_factor, population_shortfall_factor}`` (2.39).
* ``records/v2/phase1/phase1_gate.json``
    ``tests.P1-T11.{cells_failing, cells_total, best_cell}``,
    ``central_finding.{best_combined_shortfall, best_cell, monotone_trade}``.
* ``records/v2/figures/cascade_run.npz`` -- the cached replay of (i035_T0600_ks3, 7105) made by
  ``tether/analysis/v2/paperfig/cascade_run.py`` (keys time, elongation, rate, alive,
  state_time, state, geometry_load, geometry_vessel, seed).  Reused, not re-simulated.
* ``Presentation/cache/pretension_T1000_s7105.npz`` -- this clip's replay of
  (i035_T1000_ks3, 7105), produced here with ``tether.campaign.fleet_run`` exactly as
  ``cascade_run.simulate`` does (same keys).

Selection rule
--------------
Seed 7105 is the seed of the storyboard's C2 cascade (the largest-peak cascade in a scan of seeds
7101-7106 of the 0.6 kN cell, chosen for clarity, not typicality); it is a statistics seed of
both cells.  The played window is 60-175 s of the simulation clock (which includes the 20 s
warm-up), chosen to contain that cascade; it is played at x4, slowed to real time over
86-95 s around the cascade (cable 2 slack from 82.2 s, snap at 91.486 s).  Counters run from
the end of the warm-up (t = 20 s).  The cell-level numbers involve no selection.

Holds (reading time only): the animation holds two of its own instants on screen before playing
them, with the clock reading "(paused)" -- the first frame (t = 60.00 s, 3.6 s) and the frame just
after the snap (t = 91.57 s, 3.2 s: cable 2 near its peak, cables 0, 3, 4 already slack).  A held
frame repeats a played instant; no instant is added or dropped (``HOLDS``, ``timeline``).

What is asserted before any frame is drawn (``Manifest.check``)
---------------------------------------------------------------
For each cell: every spec field against the declarations; ``FleetRunSpec.config_hash()``
equals the declared hash and the run record's hash; the spec equals
``phase1_stochastic.spec_for(cell)``; the replay's weather equals the other cell's bit for bit
(same seed, same forcing); the replay's re-engagement marks -- the plant's own
``CableEventTracker`` replayed over the logged 1 ms samples, filtered as ``summarize_run``
filters them -- equal the record's in count, cable, t_up and T_peak; slack onsets per cable,
slack samples per cable, the end time, the absence of closure, the B.3 clean-set moments of the
plant's tension law on the 1 ms log (``clean.q_count/q_sum/q_square``: q = k e + c edot on taut
in-window samples outside 3 s after any re-engagement) and its exposure, and the circular sums of
the world chord angle (the P1-T11 statistic on this run) equal the record's.  For the 1.0 kN run
simulated here, the in-process tracker's marks also equal the offline replay's.  Also asserted:
the cell-level numbers' internal consistency across stochastic_results.json and the gate
record, the heading-gain schedule, the P1-T11 threshold text (chord angle and psi, 15 deg) and psi
below chord in all 8 cells, every number a caption states (snap cable/peak/time, the three onsets
within 70 ms, no slack sample at 1.0 kN anywhere in the replay t = 0-320 s, the whole-run spreads
38.1 / 23.5 deg above 15, the drawn spread staying above 15 deg in the window), every caption's
length (<= 2 lines, >= 4 s, <= 2.5 words/s), that captions C2, C3 and C5 are on screen exactly when what
they describe is (cable 2's onset and the switch to real time; the snap, the other three onsets and the
peak in the tile by the hold; the smaller 1.0 kN spread), that held frames only repeat played instants
and read "(paused)", at every caption change and card step that no
figure-level text is clipped or overlaps another text or panel, and in every animated frame that
no vessel label sits on a hull, the payload, a panel corner note or another label and that no body
touches a corner note (``label_layout``).

The chord-angle spread on screen is the campaign's P1-T11 statistic computed on one run: the
circular standard deviation of each cable's world chord angle, state rows every 0.1 s from
t = 20 s up to the current time, the largest of the five cables.  At t = 320 s it equals the
run record's ``shape_std_deg.chord_world`` maximum (asserted); the cell statistic pools the
20 statistics seeds.

Re-engagement tiles count a mark when the plant's tracker completes it (at its peak, 35-134 ms
after t_up in the 0.6 kN run), so a tile never shows a peak before the plant has reached it.

Wording held to the corrected paper: at cell level higher pretension *suppresses* the snaps
(510 -> 4 marks, -99.2 %; Sec. I/VI "destroys 99.2% of the marks", Sec. VIII "suppresses the
events"), never "removes"; the shape change is attributed to T0 *with* its scheduled k_h (Sec. VI).
The cable key is this module's (``cable_key``): the drawn width grows with tension up to 12 kN.

Run: ``nice -n 10 python3 -m tether.analysis.v2.present.clip_pretension``
(``--data-only`` builds/validates the replays and stops; ``--preview T ...`` saves stills at
video times T to ``--preview-dir`` instead of writing the clip).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.colors
import numpy as np

from tether.analysis.v2.present import common as C

NAME = "pretension"
SEED = 7105
WARMUP = 20.0
CELL_LO = "i035_T0600_ks3"
CELL_HI = "i035_T1000_ks3"
CELLS = (CELL_LO, CELL_HI)

DECL = Path("records/v2/phase1/stochastic_declarations.json")
RUNS = Path("records/v2/phase1/stochastic_runs.json")
RESULTS = Path("records/v2/phase1/stochastic_results.json")
GATE = Path("records/v2/phase1/phase1_gate.json")
CASCADE_NPZ = Path("records/v2/figures/cascade_run.npz")
CACHE_HI = C.CACHE / "pretension_T1000_s7105.npz"
CASCADE_SPEC_SRC = Path("tether/analysis/v2/paperfig/cascade_run.py")
STILLS = Path("/tmp/present_stills/pretension")          # --preview default; pass --preview-dir to change

# playback: (sim start, sim end, speed factor); speed = sim seconds per video second
SEGMENTS = ((60.0, 86.0, 4.0), (86.0, 95.0, 1.0), (95.0, 175.0, 4.0))
ANIM_SECONDS = sum((t1 - t0) / s for t0, t1, s in SEGMENTS)     # 35.5 s of video, played
# holds for reading time: (state-row instant s, seconds held); the instant is shown for the hold with the clock
# reading "(paused)", just before it is played.  60.00 s is the first frame; 91.57 s is the frame after the snap
# (cable 2 near its 13.5 kN peak, cables 0, 3, 4 already slack), on the real-time grid 86 + 167/30 s.
HOLDS = ((60.0, 3.6), (91.57, 3.2))
HOLD_FRAMES = sum(int(round(s * C.FPS)) for _, s in HOLDS)
ANIM_FRAMES = int(round(ANIM_SECONDS * C.FPS)) + HOLD_FRAMES    # 1065 played + 204 held = 42.3 s of video
TITLE_SECONDS = 8.5                       # title card: the clip title, the question it answers, what to watch
SHAPE_BAND_DEG = 15.0


def _load_json(path: Path) -> dict:
    return json.loads((C.REPO / path).read_text())


# ----------------------------------------------------------------------------- specs

def cell_spec(cell: str):
    """The run spec of ``cell``: cascade_run.SPEC for 0.6 kN; the same with T0 and k_h moved for 1.0 kN."""
    from tether.analysis.v2.paperfig.cascade_run import SPEC
    from tether.campaign.fleet_run import FleetRunSpec
    if cell == CELL_LO:
        return FleetRunSpec(**SPEC)
    if cell == CELL_HI:
        return FleetRunSpec(**{**SPEC, "pretension": 1000.0, "heading_gain": 763.0})
    raise KeyError(cell)


def check_specs(man: C.Manifest, decl: dict, runs: dict[str, dict]) -> dict:
    from tether.campaign.v2.phase1_stochastic import CELL_BY_NAME, spec_for
    block = decl["cells"]
    grid = {g["name"]: g for g in block["grid"]}
    specs = {}
    for cell in CELLS:
        spec, g = cell_spec(cell), grid[cell]
        fields = {
            "pretension": (spec.pretension, g["T0_N"]),
            "weather_scale": (spec.weather_scale, g["intensity"]),
            "heading_gain": (spec.heading_gain, g["k_h_N_m_per_rad"]),
            "k_sigma": (spec.k_sigma, g["k_sigma"]),
            "sway_limit": (spec.sway_limit, g["sway_limit_rad"]),
            "formation": (spec.formation, block["formation"]),
            "drag_law": (spec.drag_law, block["drag_law"]),
            "cable_mode": (spec.cable_mode, block["cable_mode"]),
            "duration": (spec.duration, block["duration_s"]),
            "warmup": (spec.warmup, block["warmup_s"]),
            "trim_gain": (spec.trim_gain, block["trim_gain_N_m_per_rad"]),
            "weather_distribution": (spec.weather_distribution, "gaussian"),
            "weather_direction": (spec.weather_direction, "local"),
        }
        bad = {k: v for k, v in fields.items() if v[0] != v[1]}
        man.check(f"{cell}: every spec field equals the declarations", not bad,
                  f"{len(fields)} fields; weather declared '{block['weather']}'; mismatches {bad}")
        man.check(f"{cell}: spec equals phase1_stochastic.spec_for(cell)", spec == spec_for(CELL_BY_NAME[cell]))
        h = spec.config_hash()
        man.check(f"{cell}: config_hash equals the declared hash", h == g["config_hash"], h)
        man.check(f"{cell}: config_hash equals the seed-{SEED} run record's", h == runs[cell]["config_hash"], h)
        specs[cell] = spec
    man.check("the two specs differ only in pretension and heading gain",
              {k for k, v in asdict(specs[CELL_LO]).items() if asdict(specs[CELL_HI])[k] != v}
              == {"pretension", "heading_gain"})
    return specs


# ----------------------------------------------------------------------------- replays

def simulate(spec, seed: int) -> dict:
    """One run through tether.campaign.fleet_run, as cascade_run.simulate does; plus its marks."""
    from tether.campaign import fleet_run as FR
    run = FR.build_run(spec, seed)
    FR.run_to_end(run)
    log = run.fleet.cables.log
    n, m = log.count, log.state_count
    marks = list(run.fleet.cables.reengagements)
    return dict(
        seed=seed,
        time=log.event_time[:n].copy(), elongation=log.elongation[:n].copy(),
        rate=log.rate[:n].copy(), alive=log.alive[:n].copy(),
        state_time=log.state_time[:m].copy(), state=log.state[:m].copy(),
        geometry_load=run.fleet.geometry.load_offsets.copy(),
        geometry_vessel=run.fleet.geometry.vessel_offsets.copy(),
        tensions0=run.fleet.operating.tensions.copy(),
        sim_end=float(run.simulator.get_context().get_time()),
        closure=np.array([] if run.fleet.cables.closure is None else run.fleet.cables.closure, float),
        tracker_t_up=np.array([k.t_up for k in marks]), tracker_T_peak=np.array([k.T_peak for k in marks]),
        tracker_cable=np.array([k.cable for k in marks], dtype=np.int64),
        tracker_dwell=np.array([k.dwell for k in marks]),
        wall_seconds=float(run.wall_seconds),
    )


def load_replays() -> dict[str, dict]:
    lo = dict(np.load(C.REPO / CASCADE_NPZ))
    if not CACHE_HI.exists():
        print(f"simulating {CELL_HI} seed {SEED} (300 s + warm-up) ...", flush=True)
        rec = simulate(cell_spec(CELL_HI), SEED)
        C.ensure_dirs()
        np.savez_compressed(CACHE_HI, **rec)
        print(f"cached -> {CACHE_HI.relative_to(C.REPO)}  ({rec['wall_seconds']:.0f} wall-s)", flush=True)
    hi = dict(np.load(CACHE_HI))
    return {CELL_LO: lo, CELL_HI: hi}


def offline_marks(rec: dict, warmup: float, end: float) -> dict[str, np.ndarray]:
    """The plant's own CableEventTracker replayed over the logged 1 ms samples.

    The plant feeds each tracker, every 1 ms, the same (time, e, edot, k e + c edot) it writes to
    the log (tether/physics/fleet.py FleetCables._update), so this replay reproduces its marks;
    the filter ``t_up - dwell >= warmup and t_up <= end`` is summarize_run's.
    """
    from tether.physics.cable import CableEventTracker, CableMode
    from tether.physics.fleet import CableParameters
    cable = CableParameters(mode="recording")
    time, e, r = rec["time"], rec["elongation"], rec["rate"]
    taut = cable.stiffness * e + cable.damping * r
    out = []                                             # (record, time the tracker completed it)
    for i in range(e.shape[1]):
        tr = CableEventTracker(stiffness=cable.stiffness, damping=cable.damping, cable=i, capacity=4096,
                               mode=CableMode.RECORDING, break_threshold=None)
        tr._append_event = lambda *a, **k: None          # as the plant does (_discard_event)
        ring = tr.state.reengagement_ring
        ti, ei, ri, qi = time.tolist(), e[:, i].tolist(), r[:, i].tolist(), taut[:, i].tolist()
        done = []                                        # completion times, in ring order
        for k in range(len(ti)):
            before = ring.size
            tr.sample(ti[k], ei[k], ri[k], qi[k], 0.0)
            done.extend([ti[k]] * (ring.size - before))
            got, _, _ = tr.drain_if_due(ti[k])
            out.extend(zip(got, done[:len(got)]))
            done = done[len(got):]
        out.extend(zip(ring.drain(), done))
    kept = [(m, td) for m, td in out if m.t_up - m.dwell >= warmup and m.t_up <= end]
    kept.sort(key=lambda p: (p[0].t_up, p[0].cable))
    return {"t_up": np.array([m.t_up for m, _ in kept]), "T_peak": np.array([m.T_peak for m, _ in kept]),
            "cable": np.array([m.cable for m, _ in kept], dtype=np.int64),
            "t_done": np.array([td for _, td in kept])}


def sorted_record_marks(run: dict) -> dict[str, np.ndarray]:
    m = run["marks"]
    t, T, c = np.asarray(m["t_up"], float), np.asarray(m["T_peak"], float), np.asarray(m["cable"], np.int64)
    o = np.lexsort((c, t))
    return {"t_up": t[o], "T_peak": T[o], "cable": c[o]}


def slack_onsets(rec: dict, warmup: float, end: float):
    """Geometric down-crossings of e = 0 in [warmup, end] (summaries_v2 onsets_v2)."""
    from tether.campaign.v2 import events as ev
    _, downs = ev.crossings(rec["time"], rec["elongation"], rec["alive"])
    keep = (downs.time >= warmup) & (downs.time <= end)
    return downs.time[keep], downs.cable[keep]


def chord_angles(rec: dict, warmup: float, end: float):
    """World chord angle of every cable on the P1-T11 rows (state rows in [warmup, end], every 0.1 s)."""
    from tether.campaign.v2 import regime as rg
    from tether.campaign.v2.summaries import SHAPE_DECIMATION, STATE_PERIOD
    st = rec["state_time"]
    rows = np.flatnonzero((st >= warmup - 1.0e-9) & (st <= end + 1.0e-9))
    rows = rows[:: int(round(SHAPE_DECIMATION / STATE_PERIOD))]
    d = rg.unit(rg.chord_vectors(rec["state"][rows], rec["geometry_load"], rec["geometry_vessel"]))
    return st[rows], np.arctan2(d[..., 1], d[..., 0])


def running_spread_deg(sigma: np.ndarray) -> np.ndarray:
    """Cumulative circular std (deg) per cable, row by row, then the max over cables."""
    cs, ss = np.cumsum(np.cos(sigma), axis=0), np.cumsum(np.sin(sigma), axis=0)
    n = np.arange(1, sigma.shape[0] + 1)[:, None]
    R = np.clip(np.hypot(cs, ss) / n, 1.0e-12, 1.0)
    return np.degrees(np.sqrt(-2.0 * np.log(R))).max(axis=1)


def check_replays(man: C.Manifest, specs: dict, recs: dict, runs: dict) -> dict:
    """Assert every replay reproduces its campaign record; return the per-cell derived series."""
    from tether.campaign.v2 import regime as rg
    w_lo = rg.regenerate_weather(specs[CELL_LO], SEED, 5)
    w_hi = rg.regenerate_weather(specs[CELL_HI], SEED, 5)
    man.check("same weather: the two cells' seed-7105 forcing is bit-equal", np.array_equal(w_lo, w_hi),
              f"shape {w_lo.shape}, regenerated with regime.regenerate_weather")
    out = {}
    for cell in CELLS:
        rec, run = recs[cell], runs[cell]
        man.check(f"{cell}: replay seed is {SEED}", int(rec["seed"]) == SEED)
        man.check(f"{cell}: run record is a statistics seed (not a pilot)", run["pilot"] is False)
        man.check(f"{cell}: the campaign's own weather bit-equality check passed for this run",
                  run["diagnostics"]["weather_bit_equal"] is True)
        end = float(run["end_time"])
        man.check(f"{cell}: record end time 320 s and no closure", end == 320.0 and not run["closure"]["terminal"],
                  f"end_time {end}")
        man.check(f"{cell}: replay log reaches the record's end", abs(rec["time"][-1] + 1e-3 - end) < 1e-9
                  and abs(rec["state_time"][-1] + 1e-2 - end) < 1e-9,
                  f"last 1 ms sample {rec['time'][-1]}, last state row {rec['state_time'][-1]}")
        man.check(f"{cell}: recording cables stay alive", bool(rec["alive"].all()))
        # marks
        mine, theirs = offline_marks(rec, WARMUP, end), sorted_record_marks(run)
        n = len(mine["t_up"])
        man.check(f"{cell}: mark count equals the record", n == run["n_marks"] == len(theirs["t_up"]),
                  f"replay {n}, record {run['n_marks']}")
        if n:
            dt = float(np.max(np.abs(mine["t_up"] - theirs["t_up"])))
            dT = float(np.max(np.abs(mine["T_peak"] - theirs["T_peak"]) / theirs["T_peak"]))
            man.check(f"{cell}: marks' cable, t_up and T_peak equal the record",
                      np.array_equal(mine["cable"], theirs["cable"]) and dt < 1e-9 and dT < 1e-9,
                      f"{n} marks; max |dt_up| {dt:.2e} s, max rel |dT_peak| {dT:.2e}")
        if n:
            lag = mine["t_done"] - mine["t_up"]
            man.check(f"{cell}: each mark completes (its peak is registered) within 0.5 s of t_up",
                      bool((lag >= 0).all() and (lag < 0.5).all()), f"max lag {lag.max() * 1e3:.0f} ms")
        if "tracker_t_up" in rec:
            tr = rec["tracker_t_up"]
            keep = (tr - rec["tracker_dwell"] >= WARMUP) & (tr <= end)
            man.check(f"{cell}: in-process tracker marks equal the offline replay's",
                      int(keep.sum()) == n and np.allclose(np.sort(tr[keep]), mine["t_up"], atol=1e-12, rtol=0),
                      f"{int(keep.sum())} in-window tracker marks")
        # slack onsets and slack samples
        ot, oc = slack_onsets(rec, WARMUP, end)
        per = np.bincount(oc, minlength=5)
        man.check(f"{cell}: slack onsets per cable equal the record", per.tolist() == list(run["onsets_per_cable"])
                  and len(ot) == run["onsets_count"], f"replay {per.tolist()}, record {run['onsets_per_cable']}")
        win = (rec["time"] >= WARMUP - 1e-9) & (rec["time"] <= end + 1e-9)
        slack = (~(rec["elongation"][win] > 0.0)).sum(axis=0)
        man.check(f"{cell}: slack samples per cable equal the record", slack.tolist() == list(run["slack_samples"]),
                  f"replay {slack.tolist()}")
        # B.3 clean-set moments of the plant's tension law on the 1 ms log (summaries._clean_statistics):
        # taut samples in [warm-up, end] outside (t_up, t_up + 3 s] of every up-crossing on any cable
        from tether.campaign.v2 import events as ev
        from tether.physics.fleet import CableParameters
        cab = CableParameters(mode="recording")
        ups, _ = ev.crossings(rec["time"], rec["elongation"], rec["alive"])
        clean = ev.clean_mask(rec["time"], rec["elongation"], ups.time, win, alive=rec["alive"])
        q = cab.stiffness * rec["elongation"] + cab.damping * rec["rate"]
        mine_q = (clean.sum(axis=0).astype(float), np.where(clean, q, 0.0).sum(axis=0),
                  np.where(clean, q * q, 0.0).sum(axis=0))
        rc = run["clean"]
        theirs_q = tuple(np.asarray(rc[k], float) for k in ("q_count", "q_sum", "q_square"))
        rel = max(float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0))) for a, b in zip(mine_q, theirs_q))
        man.check(f"{cell}: B.3 clean-set tension moments per cable (count, sum q, sum q^2; q = k e + c edot) "
                  "equal the record", np.array_equal(mine_q[0], theirs_q[0]) and rel < 1e-9,
                  f"counts {mine_q[0].astype(int).tolist()}; max relative deviation {rel:.1e}")
        expo = ev.excluded_exposure(ups.time, WARMUP, end)
        man.check(f"{cell}: B.3 clean exposure equals the record", abs(expo - rc["exposure_s"]) < 1e-9,
                  f"{expo:.6f} s vs {rc['exposure_s']:.6f} s")
        # chord-angle statistic
        ts, sigma = chord_angles(rec, WARMUP, end)
        sums = run["shape_sums"]["chord_world"]
        cs, ss = np.cos(sigma).sum(0), np.sin(sigma).sum(0)
        dev = float(max(np.max(np.abs(cs - sums["cos"])), np.max(np.abs(ss - sums["sin"]))))
        man.check(f"{cell}: world chord-angle circular sums equal the record (P1-T11 rows)",
                  len(ts) == sums["count"] == run["shape_samples"] and dev < 1e-6,
                  f"{len(ts)} rows; max |sum deviation| {dev:.2e}")
        spread = running_spread_deg(sigma)
        rec_max = float(np.max(run["shape_std_deg"]["chord_world"]))
        man.check(f"{cell}: running spread at t = {end:.0f} s equals the record's per-run max",
                  abs(spread[-1] - rec_max) < 1e-6, f"{spread[-1]:.6f} vs {rec_max:.6f} deg")
        T = C.tension(rec["elongation"], rec["rate"], rec["alive"])
        o = np.argsort(ot, kind="stable")
        out[cell] = dict(marks=mine, onset_t=ot[o], onset_c=oc[o], shape_t=ts, spread=spread, T=T, end=end,
                         rec_spread_end=rec_max)
    return out


# ----------------------------------------------------------------------------- main

def load_records():
    decl, runs_all, results, gate = (_load_json(p) for p in (DECL, RUNS, RESULTS, GATE))
    runs = {}
    for cell in CELLS:
        hit = [r for r in runs_all["runs"] if r["cell"] == cell and r["seed"] == SEED]
        assert len(hit) == 1, (cell, len(hit))
        runs[cell] = hit[0]
    return decl, runs, results, gate


def build_data(man: C.Manifest):
    decl, runs, results, gate = load_records()
    for p in (DECL, RUNS, RESULTS, GATE, CASCADE_NPZ, CASCADE_SPEC_SRC):
        man.source(p)
    specs = check_specs(man, decl, runs)
    recs = load_replays()
    man.source(CACHE_HI.relative_to(C.REPO))
    series = check_replays(man, specs, recs, runs)
    return decl, runs, results, gate, specs, recs, series


class StillWriter:
    """Layout preview: saves the frames at the requested video times as PNG, draws nothing else.

    Only for development (``--preview``); the clip itself always streams through common.writer().
    """

    def __init__(self, fig, times, outdir: Path):
        self.fig, self.count, self.outdir = fig, 0, Path(outdir)
        self.want = {int(round(t * C.FPS)): t for t in times}
        self.outdir.mkdir(parents=True, exist_ok=True)

    def wants(self, index: int) -> bool:
        return index in self.want

    def grab_frame(self):
        if self.count in self.want:
            p = self.outdir / f"preview_{self.want[self.count]:05.1f}s.png"
            self.fig.savefig(p, dpi=C.DPI)
            print("still", p)
        self.count += 1

    def saving(self, fig, path, dpi):
        import contextlib
        return contextlib.nullcontext()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-only", action="store_true")
    ap.add_argument("--preview", type=float, nargs="*", help="video times (s): save stills instead of the clip")
    ap.add_argument("--preview-dir", type=Path, default=STILLS)
    args = ap.parse_args(argv)
    C.ensure_dirs()
    man = C.Manifest(name=NAME, title="Pretension: one knob, two requirements", story="")
    data = build_data(man)
    for c in man.checks:
        print("PASS", c["check"], "|", c["detail"])
    if args.data_only:
        return
    render(man, *data, preview=args.preview, preview_dir=args.preview_dir)


# ----------------------------------------------------------------------------- rendering

C_LO, C_HI = C.ACCENT, "#5b4b8a"          # cell identity (not cable state: cables keep blue/red)
PANEL_Y, PANEL_H, PANEL_W = 0.44, 0.365, 0.465
PANEL_X = (0.025, 0.51)
TILE_Y0, TILE_H = 0.318, 0.110            # stat-tile boxes under each panel
STRIP = [0.075, 0.178, 0.855, 0.100]
CAPTION_Y = 0.078                         # common.caption's band, raised clear of the footer
FS_NUM = 24
SUBTITLE_ANIM = "Seed 7105 · same weather (intensity 0.35) in both runs"
SELECTION_TEXT = ("Selection: seed 7105 holds the cascade shown earlier (largest-peak cascade in a scan of seeds "
                  "7101–7106 of the 0.6 kN cell, chosen for clarity, not typicality); the 60–175 s window contains it.")
FOOTER_ANIM = ("replays asserted against records/v2/phase1/stochastic_runs.json · 0.6 kN: "
               "records/v2/figures/cascade_run.npz · 1.0 kN: Presentation/cache/pretension_T1000_s7105.npz")
FOOTER_CELL = ("cell level: records/v2/phase1/stochastic_results.json (P1_T11, trade_off, closest_to_both) · "
               "records/v2/phase1/phase1_gate.json (P1-T11, central_finding)")


KEY_TAUT = "taut (wider = more tension, up to 12 kN)"


def cable_key(fig, y: float, x: float):
    """The cable legend of common.cable_key, worded for the width draw_fleet actually draws:
    lw = 1.4 + 5.0 min(T / WIDTH_FULL_N, 1), i.e. wider with tension, saturating at 12 kN."""
    from matplotlib.lines import Line2D
    assert C.WIDTH_FULL_N == 12_000.0, "the key's '12 kN' must match common.WIDTH_FULL_N"
    handles = [Line2D([], [], color=C.TAUT, lw=4, label=KEY_TAUT),
               Line2D([], [], color=C.SLACK, lw=2, ls=(0, (3.0, 2.2)), label="carrying no tension")]
    fig.legend(handles=handles, loc="center right", ncol=2, bbox_to_anchor=(x, y), fontsize=C.FS_SMALL,
               handlelength=2.6, columnspacing=2.0, borderaxespad=0.0)


def _speed_label(s: float) -> str:
    return "real time" if s == 1.0 else f"×{s:g}"


def frame_plan(state_time: np.ndarray):
    """(state-row index, speed) of every animated frame: SEGMENTS snapped to the 10 ms state rows."""
    plan = []
    for t0, t1, s in SEGMENTS:
        n = int(round((t1 - t0) * C.FPS / s))
        for k in range(n):
            plan.append((C.nearest(state_time, t0 + k * s / C.FPS), s))
    return plan


def timeline(state_time: np.ndarray):
    """(state row, clock label, held) of every animation frame: frame_plan's played frames, each HOLDS instant
    shown for its hold (clock "(paused)") just before it is played.  Holds repeat played instants only."""
    out, hit = [], {h: 0 for h, _ in HOLDS}
    for row, s in frame_plan(state_time):
        for h, secs in HOLDS:
            if abs(float(state_time[row]) - h) < 1.0e-6:
                out += [(row, "paused", True)] * int(round(secs * C.FPS))
                hit[h] += 1
        out.append((row, _speed_label(s), False))
    assert all(v == 1 for v in hit.values()), f"each hold instant must be one played frame: {hit}"
    assert len(out) == ANIM_FRAMES, (len(out), ANIM_FRAMES)
    return out


def video_time_at(tl: list, state_time: np.ndarray, t_sim: float) -> float:
    """Video time (s, animation clock) of the first played frame at or after simulation time ``t_sim``."""
    for f, (row, _, held) in enumerate(tl):
        if not held and float(state_time[row]) >= t_sim - 1.0e-9:
            return f / C.FPS
    raise ValueError(t_sim)


def camera(rec: dict, geom, t_lo: float, t_hi: float, margin: float = 2.5):
    """Per-state-row camera centre (4 s moving average) and the half-span over [t_lo, t_hi]."""
    st = rec["state_time"]
    rows = np.flatnonzero((st >= t_lo - 5.0) & (st <= t_hi + 5.0))
    centres, half = C.fleet_bounds(rec["state"][rows], geom, margin=margin)
    smooth = C.smooth_camera(centres, window=401)
    return rows, smooth, half, centres


LABEL_R_PX = 13.0                         # clearance radius of a vessel label (digit + its white pad), px
LABEL_DIST_M = (1.0, 1.9, 2.8, 3.8, 5.0)  # candidate label distances from the bow tip, m
LABEL_ANG_DEG = (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180)   # relative to the heading


def _seg_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distance (px) from points P (k, 2) to the segment AB."""
    d = B - A
    L2 = float(d @ d)
    t = np.clip(((P - A) @ d) / L2, 0.0, 1.0) if L2 > 0.0 else np.zeros(len(P))
    return np.hypot(*(P - (A + t[:, None] * d)).T)


def _poly_dist(P: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance (px) from points P to a closed polygon: 0 inside, else to the nearest edge."""
    from matplotlib.path import Path as MPath
    edge = np.min([_seg_dist(P, poly[k], poly[(k + 1) % len(poly)]) for k in range(len(poly))], axis=0)
    return np.where(MPath(poly).contains_points(P), 0.0, edge)


def _box_dist(P: np.ndarray, box) -> np.ndarray:
    """Distance (px) from points P to a display-space Bbox (0 inside)."""
    dx = np.maximum.reduce([box.x0 - P[:, 0], np.zeros(len(P)), P[:, 0] - box.x1])
    dy = np.maximum.reduce([box.y0 - P[:, 1], np.zeros(len(P)), P[:, 1] - box.y1])
    return np.hypot(dx, dy)


def _poly_box_dist(poly: np.ndarray, box) -> float:
    """Distance (px) from a closed polygon to a Bbox, to within 0.5 px: edges densified to <= 1 px
    steps; 0 if a box corner lies inside the polygon."""
    from matplotlib.path import Path as MPath
    corners = np.array([[box.x0, box.y0], [box.x1, box.y0], [box.x1, box.y1], [box.x0, box.y1]])
    if MPath(poly).contains_points(corners).any():
        return 0.0
    pts = [np.linspace(poly[k], poly[(k + 1) % len(poly)],
                       max(2, int(np.ceil(np.hypot(*(poly[(k + 1) % len(poly)] - poly[k])))) + 1))
           for k in range(len(poly))]
    return float(_box_dist(np.vstack(pts), box).min())


def label_layout(ax, state_row, geometry, T_row, avoid=()) -> dict:
    """Place the vessel (= cable) numbers so that none sits on a hull, the payload or another label.

    Each label is searched over candidate points around its own bow tip (``LABEL_DIST_M`` x
    ``LABEL_ANG_DEG``), in screen pixels, at the least cost: intrusion into any hull or the payload
    (heavy), into an already placed label (heavy), onto a drawn cable (light), then distance from the
    bow and angle off the heading.  Most constrained vessels are placed first, then every label is
    re-placed twice against the others.  ``avoid`` are display-space boxes (the panel's corner notes)
    that no label may touch.  Returns data-space label points, bow tips, and the smallest clearance (px)
    of any label from any body, any other label and any avoided box, and of any body from any avoided
    box, for the per-frame assertion.
    """
    from tether.physics import fleet as F
    ax.apply_aspect()
    tf = ax.transData.transform
    lp, vp = C.unpack_state(state_row)
    n = len(vp)
    bows = np.array([C.body_polygon(v, np.array([[0.5 * C.L_HULL, 0.0]]))[0] for v in vp])
    ahead = np.array([C.body_polygon(v, np.array([[0.5 * C.L_HULL + 1.0, 0.0]]))[0] for v in vp])
    Bp, Ap = tf(bows), tf(ahead)
    px_per_m = float(np.hypot(*(Ap[0] - Bp[0])))                # 1 m along the heading, in px
    bodies = [tf(C.body_polygon(v, C.hull())) for v in vp] + [tf(C.body_polygon(lp, F.pentagon_vertices()))]
    a, b = C.attachment_points(lp, vp, geometry)
    cables = [(tf(a[i][None])[0], tf(b[i][None])[0]) for i in range(n)]
    x0, y0, x1, y1 = ax.get_window_extent().extents
    R = LABEL_R_PX
    cand = []
    for i in range(n):
        u = (Ap[i] - Bp[i]) / px_per_m
        v = np.array([-u[1], u[0]])
        pts, pref = [], []
        for d in LABEL_DIST_M:
            for g in LABEL_ANG_DEG:
                r = np.radians(g)
                pts.append(Bp[i] + d * px_per_m * (np.cos(r) * u + np.sin(r) * v))
                pref.append(0.6 * d * px_per_m + 0.04 * abs(g))
        P = np.array(pts)
        own = _poly_dist(P, bodies[i])                          # may touch its own bow (draw_fleet's spot)
        other = np.min([_poly_dist(P, poly) for k, poly in enumerate(bodies) if k != i], axis=0)
        wire = np.min([_seg_dist(P, A, B) for A, B in cables], axis=0)
        inside = (P[:, 0] > x0 + R + 4) & (P[:, 0] < x1 - R - 4) & (P[:, 1] > y0 + R + 4) & (P[:, 1] < y1 - R - 4)
        note = np.min([_box_dist(P, bx) for bx in avoid], axis=0) if len(avoid) else np.full(len(P), np.inf)
        hit = (other < R + 3.0) | (own < R) | (note < R + 3.0)  # any intrusion is a hard penalty
        base = (100.0 * np.maximum(0.0, R + 3.0 - other) + 100.0 * np.maximum(0.0, R - own) + 1.0e4 * hit
                + 20.0 * np.maximum(0.0, R - wire) + np.array(pref) + np.where(inside, 0.0, 1.0e6))
        cand.append((P, base, np.minimum(np.minimum(other, note) - 3.0, own)))

    def best(i, placed):
        P, base, _ = cand[i]
        cost = base.copy()
        for j, q in placed.items():
            if j != i:
                gap = 2.0 * R + 2.0 - np.hypot(*(P - q).T)
                cost += 100.0 * np.maximum(0.0, gap) + 1.0e4 * (gap > 0.0)
        return int(np.argmin(cost))

    crowd = [sum(np.hypot(*(Bp[j] - Bp[i])) < 4.0 * R for j in range(n) if j != i) for i in range(n)]
    order = sorted(range(n), key=lambda i: -crowd[i])
    placed, pick = {}, {}
    for _ in range(3):                                          # greedy pass, then two refinements
        for i in order:
            pick[i] = best(i, placed)
            placed[i] = cand[i][0][pick[i]]
    L = np.array([placed[i] for i in range(n)])
    body_clear = min(float(cand[i][2][pick[i]]) for i in range(n)) - R
    pair = min(float(np.hypot(*(L[i] - L[j]))) for i in range(n) for j in range(i + 1, n)) - 2.0 * R
    body_note = min((_poly_box_dist(poly, bx) for poly in bodies for bx in avoid), default=np.inf)
    inv = ax.transData.inverted()
    return {"label": inv.transform(L), "bow": bows, "label_px": L, "bow_px": Bp,
            "body_clearance_px": body_clear, "label_clearance_px": pair, "body_note_px": float(body_note),
            "moved": np.hypot(*(L - Ap).T)}                     # px from the default spot, bow + 1 m


def vessel_labels(ax, state_row, geometry, T_row, avoid=()) -> dict:
    """Draw the vessel (= cable) numbers placed by ``label_layout``; a label away from its default spot
    (bow + 1 m along the heading, where common.draw_fleet puts it) gets a dark leader to its own bow tip."""
    lay = label_layout(ax, state_row, geometry, T_row, avoid)
    for i, (q, bow) in enumerate(zip(lay["label"], lay["bow"])):
        if lay["moved"][i] > 3.0:
            Lp, Bp = lay["label_px"][i], lay["bow_px"][i]
            e = Lp - (LABEL_R_PX - 2.0) * (Lp - Bp) / max(float(np.hypot(*(Lp - Bp))), 1e-9)
            ee = ax.transData.inverted().transform(e)
            ax.plot([bow[0], ee[0]], [bow[1], ee[1]], color=C.INK, lw=1.1, alpha=0.9, zorder=5,
                    solid_capstyle="round")
        ax.text(q[0], q[1], str(i), fontsize=C.FS_SMALL, color=C.INK, ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    return lay


class Tiles:
    """Three stat tiles under a fleet panel: slack onsets, re-engagement marks (largest peak), spread."""

    def __init__(self, fig, x0: float, w: float, colour: str):
        from matplotlib.patches import FancyBboxPatch
        self.num, self.lab = [], []
        for j in range(3):
            cx = x0 + (j + 0.5) * w / 3
            fig.patches.append(FancyBboxPatch((cx - 0.155 * w, TILE_Y0), 0.31 * w, TILE_H,
                                              boxstyle="round,pad=0.003", transform=fig.transFigure,
                                              fc="#f6f7f9", ec=C.FAINT, lw=0.8, zorder=0))
            self.num.append(fig.text(cx, TILE_Y0 + 0.078, "", fontsize=FS_NUM, weight="bold", color=colour,
                                     ha="center", va="center"))
            self.lab.append(fig.text(cx, TILE_Y0 + 0.050, "", fontsize=C.FS_SMALL, color=C.MUTED, ha="center",
                                     va="top", linespacing=1.15))

    def set(self, onsets: int, marks: int, peak_kN, spread: float):
        self.num[0].set_text(f"{onsets}")
        self.lab[0].set_text("slack onsets\nsince t = 20 s")
        self.num[1].set_text(f"{marks}")
        self.lab[1].set_text("re-engagements\n" + (f"largest peak {peak_kN:.1f} kN" if peak_kN is not None
                                                   else "largest peak: none"))
        self.num[2].set_text(f"{spread:.1f}°")
        self.lab[2].set_text("chord-angle spread\nsince t = 20 s")

    def texts(self):
        return self.num + self.lab


def _extent(artist, renderer):
    patch = artist.get_bbox_patch() if hasattr(artist, "get_bbox_patch") else None
    return (patch.get_window_extent(renderer) if patch is not None else artist.get_window_extent(renderer))


def layout_check(fig, man: C.Manifest, where: str, texts, boxes=()) -> None:
    """Assert that no figure-level text is clipped by the frame or overlaps another text or a panel."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = [(t.get_text().replace("\n", " ")[:40], _extent(t, r)) for t in texts if t.get_text()]
    items += [(name, b) for name, b in boxes]
    bad = []
    for name, b in items[: len(items) - len(boxes)]:
        if b.x0 < 0 or b.y0 < 0 or b.x1 > C.W or b.y1 > C.H:
            bad.append(f"clipped: {name!r}")
    for a in range(len(items)):
        for b_ in range(a + 1, len(items)):
            (na, ba), (nb, bb) = items[a], items[b_]
            if ba.overlaps(bb):
                bad.append(f"overlap: {na!r} / {nb!r}")
    man.check(f"layout ({where}): no text clipped or overlapping", not bad, "; ".join(bad) or f"{len(items)} boxes")


TITLE_QUESTION = ("Can one pretension keep the formation in shape and still produce the slack events "
                  "a design law needs?")
TITLE_WATCH = "Watch how far the cables swing and whether any goes slack."
TITLE_CONTEXT = ("Both runs are simulation replays: five tugs tow one payload in parallel formation "
                 "(paper Section VI).")


def title_card(fig, w, man: C.Manifest) -> int:
    fig.clf()
    ts = [fig.text(0.5, 0.62, "Pretension: one knob, two requirements", fontsize=34, weight="bold", color=C.INK,
                   ha="center", va="center"),
          fig.text(0.5, 0.52, TITLE_QUESTION, fontsize=C.FS_SUB, color=C.INK, ha="center", va="center"),
          fig.text(0.5, 0.46, TITLE_WATCH, fontsize=C.FS_SUB, color=C.INK, ha="center", va="center"),
          fig.text(0.5, 0.39, TITLE_CONTEXT, fontsize=C.FS_SMALL, color=C.MUTED, ha="center", va="center")]
    C.footer(fig, "Tail of the Tether campaign · records/v2/phase1 (Phase 1(f) stochastic cells)")
    layout_check(fig, man, "title card", fig.texts)
    w.grab_frame()
    return 1 + C.hold(w, TITLE_SECONDS - 1.0 / C.FPS)


def animate(fig, w, man, recs, series, decl) -> int:
    """The side-by-side window.  Returns frames written."""
    geom = C.parallel_geometry()
    lo, hi = recs[CELL_LO], recs[CELL_HI]
    man.check("both replays share the state clock and the parallel geometry",
              np.array_equal(lo["state_time"], hi["state_time"])
              and all(np.allclose(r["geometry_load"], geom.load_offsets)
                      and np.allclose(r["geometry_vessel"], geom.vessel_offsets) for r in (lo, hi)))
    st = lo["state_time"]
    plan = frame_plan(st)
    t_lo, t_hi = st[plan[0][0]], st[plan[-1][0]]
    cams = {c: camera(recs[c], geom, t_lo, t_hi) for c in CELLS}
    half = np.maximum(cams[CELL_LO][2], cams[CELL_HI][2])        # one scale for both panels
    box_aspect = (PANEL_W * C.W) / (PANEL_H * C.H)
    half = C.fit_aspect(half, box_aspect)
    fleet_half_x = max(float(np.max(np.abs(cams[c][3][:, 0] - cams[c][1][:, 0]))) + cams[c][2][0] for c in CELLS)
    free_px = (half[0] - fleet_half_x) / (2 * half[0]) * PANEL_W * C.W
    man.check("panel corners beyond the fleet's reach hold the corner notes",
              free_px > 200.0, f"{free_px:.0f} px free at each side of every panel")
    for c in CELLS:
        smooth = cams[c][1]
        man.check(f"{c}: the fleet tows toward +x over the window (label 'tow direction')",
                  smooth[-1, 0] - smooth[0, 0] > 10.0, f"camera x {smooth[0, 0]:.1f} -> {smooth[-1, 0]:.1f} m")
    man.value("panel half-span (both panels, same scale)", [round(float(half[0]), 2), round(float(half[1]), 2)], "m",
              "replay: fleet_bounds over both runs in the window + 2.5 m margin, fit to the panel aspect",
              "scale bar 5 m drawn in each panel")

    fig.clf()
    C.title(fig, "Pretension: one knob, two requirements", SUBTITLE_ANIM)
    cable_key(fig, y=0.892, x=0.972)
    fig.text(0.03, 0.862, SELECTION_TEXT, fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="top")
    C.footer(fig, FOOTER_ANIM)
    heads = {CELL_LO: ("T0 = 0.6 kN", "heading gain k_h = 477 N·m/rad"),
             CELL_HI: ("T0 = 1.0 kN", "k_h = 763 N·m/rad (declared schedule: rises with T0)")}
    axes = {}
    for c, x0, col in ((CELL_LO, PANEL_X[0], C_LO), (CELL_HI, PANEL_X[1], C_HI)):
        axes[c] = fig.add_axes([x0, PANEL_Y, PANEL_W, PANEL_H])
        a, b = heads[c]
        t1 = fig.text(x0 + 0.004, PANEL_Y + PANEL_H + 0.008, a, fontsize=C.FS_SUB, weight="bold", color=col,
                      ha="left", va="bottom")
        fig.canvas.draw()
        bb = t1.get_window_extent().transformed(fig.transFigure.inverted())
        fig.text(bb.x1 + 0.008, PANEL_Y + PANEL_H + 0.010, b, fontsize=C.FS_SMALL, color=C.INK, ha="left",
                 va="bottom")
    tiles = {CELL_LO: Tiles(fig, PANEL_X[0], PANEL_W, C_LO), CELL_HI: Tiles(fig, PANEL_X[1], PANEL_W, C_HI)}

    axS = fig.add_axes(STRIP)
    axS.set_xlim(60.0, 175.0); axS.set_ylim(0.0, 50.0)
    axS.axhspan(0.0, SHAPE_BAND_DEG, color=C.OK, alpha=0.12, lw=0)
    axS.axhline(SHAPE_BAND_DEG, color=C.OK, lw=1.2, ls=(0, (4, 3)))
    band_rgb = tuple(float(v) for v in 1.0 - 0.12 * (1.0 - np.array(matplotlib.colors.to_rgb(C.OK))))  # opaque band tint
    # the band label and the legend sit on opaque band-tinted boxes so the playhead never crosses their text
    axS.text(61.0, 7.5, "P1-T11 shape band (chord half): spread ≤ 15°", fontsize=C.FS_TINY, color=C.OK,
             ha="left", va="center", zorder=6, bbox=dict(boxstyle="square,pad=0.2", fc=band_rgb, ec="none"))
    axS.set_yticks([0, 15, 30, 45]); axS.set_xticks(np.arange(60, 176, 20))
    axS.set_ylabel("spread (°)", fontsize=C.FS_SMALL)
    axS.text(1.004, -0.04, "t (s)", transform=axS.transAxes, fontsize=C.FS_TINY, color=C.MUTED, va="top")
    axS.grid(True, axis="y")
    fig.text(STRIP[0], STRIP[1] + STRIP[3] + 0.016,
             "Chord-angle spread since t = 20 s (end of warm-up): circular std of the world chord angle, "
             "largest of 5 cables (P1-T11, one run)",
             fontsize=C.FS_SMALL, color=C.INK, ha="left", va="bottom")
    lines = {c: axS.plot([], [], color=col, lw=2.2, label=lab)[0]
             for c, col, lab in ((CELL_LO, C_LO, "T0 = 0.6 kN"), (CELL_HI, C_HI, "T0 = 1.0 kN"))}
    head = axS.axvline(t_lo, color=C.INK, lw=0.8, alpha=0.6)
    # inside the band, bottom right: no drawn line enters the band in the window (asserted in build_captions)
    leg = axS.legend(handles=list(lines.values()), loc="lower right", ncol=2, fontsize=C.FS_TINY, handlelength=2.0,
                     borderaxespad=0.25, frameon=True, fancybox=False, framealpha=1.0, facecolor=band_rgb,
                     edgecolor="none", borderpad=0.25)
    leg.set_zorder(6)

    caps = ANIM_CAPTIONS                       # (video start s, video end s, text), animation clock
    tl = timeline(st)                          # the played frames (plan) plus the HOLDS frames
    n_frames = len(tl)
    cap_txt, cap_now, clk = None, None, None
    frames = 0
    clear = {c: [np.inf, np.inf, 0, 0, np.inf] for c in CELLS}   # label clearance from bodies+notes / labels (px),
    #                                                              leaders, labels, body clearance from the notes (px)

    def draw_row(row, t):
        """Panels, tiles, strip line and playhead at state row ``row`` (simulation time ``t``)."""
        for c in CELLS:
            rec, S = recs[c], series[c]
            rows, smooth, _, _ = cams[c]
            k = int(np.searchsorted(rows, row))
            assert rows[k] == row
            i1 = int(np.searchsorted(rec["time"], t - 1e-7))
            assert abs(rec["time"][i1] - t) < 1e-9, "state row and cable sample must be the same instant"
            ax = axes[c]
            C.draw_fleet(ax, rec["state"][row], geom, S["T"][i1], rec["alive"][i1], centre=smooth[k],
                         half=half, labels=False, scale_bar=True, check_inside=True)
            notes = [ax.text(0.985, 0.965, "parallel formation\ntow direction →", transform=ax.transAxes,
                             fontsize=C.FS_TINY, color=C.MUTED, ha="right", va="top", linespacing=1.15),
                     ax.text(0.985, 0.035, "bodies do not contact:\nhulls and lines may overlap",
                             transform=ax.transAxes, fontsize=C.FS_TINY, color=C.MUTED, ha="right", va="bottom",
                             linespacing=1.15)]
            rend = fig.canvas.get_renderer()
            lay = vessel_labels(ax, rec["state"][row], geom, S["T"][i1],
                                avoid=[n.get_window_extent(rend) for n in notes])
            clear[c][0] = min(clear[c][0], lay["body_clearance_px"])
            clear[c][1] = min(clear[c][1], lay["label_clearance_px"])
            clear[c][2] += int((lay["moved"] > 3.0).sum()); clear[c][3] += len(lay["moved"])
            clear[c][4] = min(clear[c][4], lay["body_note_px"])
            m = S["marks"]
            done = m["t_done"] <= t + 1e-12
            onsets = int(np.sum(S["onset_t"] <= t + 1e-12))
            peak = float(m["T_peak"][done].max()) / 1e3 if done.any() else None
            j = int(np.searchsorted(S["shape_t"], t + 1e-9)) - 1
            tiles[c].set(onsets, int(done.sum()), peak, float(S["spread"][j]))
            sel = (S["shape_t"] <= t + 1e-9) & (S["shape_t"] >= SEGMENTS[0][0] - 1e-9)
            lines[c].set_data(S["shape_t"][sel], S["spread"][sel])
        head.set_xdata([t, t])

    drawn = None                               # state row the panels, tiles and strip currently show
    for f, (row, speed_label, held) in enumerate(tl):
        t = float(st[row])
        tv = f / C.FPS
        if hasattr(w, "wants") and not w.wants(w.count):        # preview only: skip undrawn frames
            w.grab_frame(); frames += 1
            continue
        if row != drawn:                       # a held frame repeats the row already drawn; clock/caption may change
            draw_row(row, t)
            drawn = row
        want = next((txt for a, b, txt in caps if a <= tv < b), None)
        if clk is not None:
            clk.remove()
        clk = C.clock(fig, t, speed_label)
        if want != cap_now:
            if cap_txt is not None:
                cap_txt.remove()
            cap_txt = C.caption(fig, want, y=CAPTION_Y) if want else None
            cap_now = want
            layout_check(fig, man, f"animation, caption {caps.index(next(x for x in caps if x[2] == want)) + 1}",
                         fig.texts, boxes=[(f"panel {c}", axes[c].get_window_extent()) for c in CELLS]
                         + [("strip", axS.get_tightbbox()), ("cable key", fig.legends[0].get_window_extent())])
        w.grab_frame()
        frames += 1
    assert frames == n_frames == ANIM_FRAMES
    held = [(r, lab) for r, lab, h in tl if h]
    man.check("held frames repeat played instants and read '(paused)'; the played frames are exactly the plan's "
              "(no instant added or dropped)",
              len(held) == HOLD_FRAMES and all(lab == "paused" for _, lab in held)
              and {r for r, _ in held} <= {r for r, _ in plan} and [r for r, _, h in tl if not h] == [r for r, _ in plan],
              "; ".join(f"t = {float(st[r]):.3f} s held {sum(1 for q, _ in held if q == r) / C.FPS:.1f} s"
                        for r in sorted({r for r, _ in held})) + f"; {len(plan)} played + {len(held)} held frames")
    if not hasattr(w, "wants"):                                  # the clip itself (every frame drawn)
        for c in CELLS:
            b, l_, moved, total, bn = clear[c]
            man.check(f"{c}: vessel labels, every animated frame: no label on a hull, the payload, a corner note "
                      "or another label", b >= 0.0 and l_ >= 0.0,
                      f"min clearance {b:.2f} px beyond a {LABEL_R_PX:.0f} px label radius (other bodies and "
                      f"corner notes +3 px), {l_:.2f} px between labels; {moved} of {total} labels moved off "
                      "bow + 1 m, each with a leader to its own bow tip")
            man.check(f"{c}: every animated frame: no hull or payload touches a panel corner note",
                      bn > 5.0, f"smallest body-to-note distance {bn:.0f} px")
        man.value("cable width saturates at (cable key)", 12, "kN", "tether/analysis/v2/present/common.py "
                  "WIDTH_FULL_N (drawing constant, not data)", "key: 'taut (wider = more tension, up to 12 kN)'")
    man.value("simulation clock shown (first, last animated frame)", [round(float(t_lo), 3), round(float(t_hi), 3)],
              "s", "replay state_time rows (10 ms grid; includes the 20 s warm-up)",
              f"{len(plan)} frames; every frame is one logged instant (state row and 1 ms cable sample)")
    # the numbers the tiles start on, and every peak value a tile shows
    for c in CELLS:
        S, m = series[c], series[c]["marks"]
        done0 = m["t_done"] <= t_lo
        j0 = int(np.searchsorted(S["shape_t"], t_lo + 1e-9)) - 1
        man.value(f"{c}: tiles at start of window (onsets, re-engagements, spread deg)",
                  [int(np.sum(S["onset_t"] <= t_lo)), int(done0.sum()), round(float(S["spread"][j0]), 1)], "",
                  "replay (same definitions as the end-of-window values)")
        shown = sorted({round(float(m["T_peak"][(m["t_done"] <= t)].max()) / 1e3, 1)
                        for t in st[[r for r, _ in plan]] if (m["t_done"] <= t).any()})
        if shown:
            man.value(f"{c}: 'largest peak' values shown in the tile", shown, "kN",
                      f"{RUNS} runs[{c}, 7105].marks.T_peak (= replay tracker), running maximum")
    # the numbers the animation ended on
    for c in CELLS:
        S = series[c]
        t = float(st[plan[-1][0]])
        m = S["marks"]; done = m["t_done"] <= t
        man.value(f"{c}: slack onsets at end of window (t = {t:.2f} s)", int(np.sum(S['onset_t'] <= t)), "",
                  "replay: tether.campaign.v2.events.crossings down-crossings of e = 0 in [20 s, t]",
                  "running counter; full-run total asserted equal to stochastic_runs.json onsets_per_cable")
        man.value(f"{c}: re-engagements at end of window", int(done.sum()), "",
                  "replay: plant CableEventTracker over the 1 ms log, counted when the tracker completes the mark",
                  "running counter; full-run marks asserted equal to stochastic_runs.json marks")
        if done.any():
            man.value(f"{c}: largest re-engagement peak in tile at end of window", round(float(m["T_peak"][done].max()) / 1e3, 1),
                      "kN", f"{RUNS} runs[{c}, 7105].marks.T_peak (= replay tracker)", "tile sub-label")
        j = int(np.searchsorted(S["shape_t"], t + 1e-9)) - 1
        man.value(f"{c}: chord-angle spread at end of window", round(float(S["spread"][j]), 1), "deg",
                  "replay: circular std of world chord angle, rows [20 s, t] every 0.1 s, max over cables",
                  "running statistic; at t = 320 s it equals stochastic_runs.json shape_std_deg.chord_world max")
    return frames


ANIM_CAPTIONS: list = []


def build_captions(man: C.Manifest, series: dict, runs: dict, decl: dict, recs: dict) -> None:
    """Narration for the animation, with every number in it registered."""
    lo = series[CELL_LO]
    m = lo["marks"]
    big = int(np.argmax(m["T_peak"]))
    t_snap, T_snap, cab = float(m["t_up"][big]), float(m["T_peak"][big]), int(m["cable"][big])
    man.check("the window's large snap is cable 2, 13 502.4 N at t_up 91.486 s (the C2 cascade mark)",
              cab == 2 and abs(T_snap - 13502.404156406299) < 1e-6 and abs(t_snap - 91.486) < 5e-4,
              f"cable {cab}, {T_snap:.1f} N at {t_snap:.4f} s")
    # the slack onset that precedes it on the same cable
    ot, oc = lo["onset_t"], lo["onset_c"]
    prev = ot[(oc == cab) & (ot < t_snap)].max()
    after = [(float(t - t_snap), int(c)) for t, c in zip(ot, oc) if 0 < t - t_snap <= 0.1 and c != cab]
    man.check("three other cables go slack within 70 ms of the snap (cables 0, 4, 3)",
              [c for _, c in after] == [0, 4, 3] and max(d for d, _ in after) < 0.071,
              f"{[(round(d * 1e3, 1), c) for d, c in after]} ms after t_up")
    from tether.campaign.v2 import events as ev
    ups, _ = ev.crossings(recs[CELL_LO]["time"], recs[CELL_LO]["elongation"], recs[CELL_LO]["alive"])
    between = ups.time[(ups.cable == cab) & (ups.time > prev) & (ups.time < t_snap - 1e-9)]
    man.check("cable 2 stays slack from its onset to the snap (no up-crossing in between)", between.size == 0,
              f"onset {prev:.3f} s, re-engagement {t_snap:.3f} s")
    for c in CELLS:
        S = series[c]
        win = (S["shape_t"] >= SEGMENTS[0][0] - 1e-9) & (S["shape_t"] <= SEGMENTS[-1][1] + 1e-9)
        man.check(f"{c}: drawn spread stays above the 15 deg band over the played window (no line enters the "
                  "band, where the strip legend sits)",
                  float(S["spread"][win].min()) > SHAPE_BAND_DEG, f"min {S['spread'][win].min():.1f} deg")
    man.value("slack onset of cable 2 before the snap", round(float(prev), 1), "s",
              "replay: events.crossings down-crossing on cable 2, cascade_run.npz", "caption C2")
    man.value("snap: cable 2 peak", round(T_snap / 1e3, 1), "kN",
              "records/v2/phase1/stochastic_runs.json runs[i035_T0600_ks3, 7105].marks.T_peak (= replay)",
              "caption C3; 22.5 T0")
    man.value("snap: re-engagement instant", round(t_snap, 3), "s",
              f"{RUNS} runs[{CELL_LO}, 7105].marks.t_up (= replay)",
              "timing of caption C3")
    man.value("other cables slack within", 70, "ms",
              "replay: onsets on cables 0, 4, 3 at +29/+52/+70 ms after t_up (cascade_run.npz)", "caption C3")
    hi_run = runs[CELL_HI]
    man.check("1.0 kN run: no slack sample and no onset in the record's window (t = 20-320 s)",
              sum(hi_run["slack_samples"]) == 0 and hi_run["onsets_count"] == 0,
              f"slack_samples {hi_run['slack_samples']}")
    e_hi = recs[CELL_HI]["elongation"]
    man.check("1.0 kN run: no slack sample (e <= 0) at any 1 ms sample of the whole replay, t = 0-320 s "
              "(caption C4 'at any time in the run'; the warm-up part is the asserted replay's)",
              bool((e_hi > 0.0).all()) and recs[CELL_HI]["time"][0] == 0.0,
              f"{e_hi.shape[0]} samples x {e_hi.shape[1]} cables; min e {e_hi.min() * 1e3:.2f} mm")
    T_hi = series[CELL_HI]["T"]
    man.check("1.0 kN run: every cable carries tension at every 1 ms sample of the replay (no red dashes)",
              bool((T_hi > 0).all()), f"min {T_hi.min():.0f} N")
    man.value("1.0 kN run: slack samples, whole run t = 0-320 s", 0, "",
              f"{RUNS} runs[{CELL_HI}, 7105].slack_samples (t = 20-320 s, = replay) + "
              f"replay:{CACHE_HI.name} elongation > 0 at every sample of t = 0-20 s",
              "caption C4 'no cable goes slack at any time in the run'")
    whole = {c: float(np.max(runs[c]["shape_std_deg"]["chord_world"])) for c in CELLS}
    man.check("seed 7105 whole-run spread (record = replay at t = 320 s) is above 15 deg in both runs (caption C6)",
              all(v > SHAPE_BAND_DEG and abs(v - series[c]["rec_spread_end"]) < 1e-12 for c, v in whole.items()),
              f"{whole[CELL_LO]:.2f} and {whole[CELL_HI]:.2f} deg")
    for c in CELLS:
        man.value(f"{c}: seed 7105 whole-run chord-angle spread (caption C6)", round(whole[c], 1), "deg",
                  f"{RUNS} runs[{c}, 7105].shape_std_deg.chord_world max (= replay running spread at t = 320 s)",
                  "caption C6 'whole-run spread'")
    texts = [
        "Both fleets meet the same gusts; the right is pretensioned to 1.0 kN, not 0.6 kN,\n"
        "with its heading gain raised by schedule.",
        f"On the left, cable 2 goes slack at {prev:.1f} s,\nand playback slows to real time.",
        f"Cable 2 re-engages, snapping taut at {T_snap / 1e3:.1f} kN,\nand within 70 ms three other cables go slack.",
        "At 1.0 kN, no cable goes slack at any time in the run,\nso nothing snaps.",
        "The strip tracks the chord-angle spread, how widely cable directions wander;\n"
        "more pretension and heading gain keep it smaller.",
        f"Yet both runs miss the 15° shape band:\n"
        f"their whole-run spreads are {whole[CELL_LO]:.1f}° and {whole[CELL_HI]:.1f}°.",
    ]
    # timing on the animation clock (played + held frames): C2 from the first frame showing cable 2 slack, C3 from
    # the first frame after its re-engagement (through the hold after the snap); C3-C5 then stay for their reading
    # time (<= 2.5 words/s) and C6 takes the rest.
    st = recs[CELL_LO]["state_time"]
    tl = timeline(st)
    need = [int(np.ceil((len(x.split()) / 2.5 - 0.05) * C.FPS - 1e-9)) for x in texts]
    f1 = int(round(video_time_at(tl, st, prev) * C.FPS))
    f2 = int(round(video_time_at(tl, st, t_snap) * C.FPS))
    f3 = f2 + need[2]
    f4 = f3 + need[3]
    f5 = f4 + need[4]
    edges = [0, f1, f2, f3, f4, f5, len(tl)]
    assert all(b - a >= n for a, b, n in zip(edges, edges[1:], need)), (edges, need)
    ANIM_CAPTIONS[:] = [(a / C.FPS, b / C.FPS, x) for a, b, x in zip(edges, edges[1:], texts)]
    shown = lambda k: [(f, *tl[f]) for f in range(edges[k], edges[k + 1])]          # frames of caption k (0-based)
    # C2 is on screen when cable 2 goes slack and when playback slows to real time
    c2 = shown(1)
    first_rt = next(f for f, (r, lab, h) in enumerate(tl) if lab == "real time")
    man.check("caption C2 appears on the first frame at or after cable 2's slack onset and is on screen when "
              "playback slows to real time",
              float(st[c2[0][1]]) >= prev and float(st[tl[f1 - 1][0]]) < prev and edges[1] <= first_rt < edges[2],
              f"C2 from t = {float(st[c2[0][1]]):.3f} s (onset {prev:.3f} s) to t = {float(st[c2[-1][1]]):.3f} s; "
              f"real time from t = {float(st[tl[first_rt][0]]):.3f} s")
    # C3 is on screen from the first frame after t_up and through the hold after the snap, where the three other
    # cables have gone slack and the tile shows the 13.5 kN peak
    c3 = shown(2)
    held3 = [float(st[r]) for _, r, _, h in c3 if h]
    t_hold = held3[0] if held3 else float("nan")
    lag = [d for d, _ in after]
    man.check("caption C3 appears on the first frame after the re-engagement and stays through the hold after the "
              "snap, by which the three other cables have gone slack and the mark's 13.5 kN peak is in the tile",
              float(st[c3[0][1]]) >= t_snap and float(st[tl[f2 - 1][0]]) < t_snap and len(held3) > 0
              and t_snap + max(lag) <= t_hold and float(m["t_done"][big]) <= t_hold,
              f"C3 from t = {float(st[c3[0][1]]):.3f} s (t_up {t_snap:.3f} s), hold at t = {t_hold:.3f} s for "
              f"{len(held3) / C.FPS:.1f} s; onsets +{', +'.join(f'{d * 1e3:.0f}' for d in lag)} ms, peak registered "
              f"at t = {float(m['t_done'][big]):.3f} s")
    # caption C5 ('keep it smaller' at 1.0 kN): true on every drawn row while it is on screen
    c5 = shown(4)
    s0, s1 = float(st[c5[0][1]]), float(st[c5[-1][1]])
    lo_s, hi_s = series[CELL_LO], series[CELL_HI]
    rows5 = (lo_s["shape_t"] >= s0 - 1e-9) & (lo_s["shape_t"] <= s1 + 1e-9)
    gap = lo_s["spread"][rows5] - hi_s["spread"][rows5]
    man.check("caption C5: the 1.0 kN running spread is below the 0.6 kN one on every row while C5 is shown",
              np.array_equal(lo_s["shape_t"], hi_s["shape_t"]) and rows5.sum() > 0 and bool((gap > 0).all()),
              f"t = {s0:.1f}-{s1:.1f} s, {int(rows5.sum())} rows, smallest gap {gap.min():.1f} deg")


def cell_numbers(man: C.Manifest, decl, runs, results, gate) -> dict:
    """Read and register the cell-level numbers; assert their internal consistency."""
    t11, trade = results["P1_T11"], {r["cell"]: r for r in results["trade_off"]}
    ranked = results["closest_to_both"]["ranked"][0]
    g11 = gate["tests"]["P1-T11"]
    cf = gate["central_finding"]
    d = {
        "std": {c: t11[c]["chord_world_stat_deg"] for c in CELLS},
        "marks": {c: trade[c]["marks"] for c in CELLS},
        "all_std": {c: t11[c]["chord_world_stat_deg"] for c in t11},
        "least": results["closest_to_both"]["least_far"],
        "combined": ranked["combined"], "shape_factor": ranked["shape_shortfall_factor"],
        "pop_factor": ranked["population_shortfall_factor"],
    }
    man.check("stochastic_results trade_off chord std equals P1_T11 for both cells",
              all(abs(trade[c]["chord_world_std_deg"] - d["std"][c]) < 1e-9 for c in CELLS))
    man.check("no cell meets the 15 deg band (P1_T11 every chord_world_meets_15 false; gate 8 of 8 failing)",
              not any(v["chord_world_meets_15"] for v in t11.values()) and g11["cells_failing"] == g11["cells_total"] == 8
              and len(t11) == 8 and min(d["all_std"].values()) > SHAPE_BAND_DEG)
    man.check("closest cell to both halves is the 0.6 kN cell of this clip, 2.39 = 35.9/15, entirely shape",
              d["least"] == CELL_LO == ranked["cell"] == cf["best_cell"] and abs(d["combined"] - cf["best_combined_shortfall"]) < 1e-12
              and abs(d["combined"] - d["std"][CELL_LO] / SHAPE_BAND_DEG) < 1e-9 and d["pop_factor"] < 1.0,
              f"combined {d['combined']:.4f}, population factor {d['pop_factor']:.4f} (floored to 1)")
    man.check("per-class population rule not met by the least-far cell (qualifier on screen)",
              trade[CELL_LO]["meets_population_rule_in_some_class"] is False)
    man.check("psi spread below the chord-angle spread in all 8 cells (card: 'psi, the band's other half, has "
              "the smaller spread in all 8 cells'), so the band's verdict is the chord half's",
              len(t11) == 8 and all(v["psi_stat_deg"] < v["chord_world_stat_deg"] for v in t11.values()),
              ", ".join(f"{c}: psi {v['psi_stat_deg']:.1f} < chord {v['chord_world_stat_deg']:.1f}"
                        for c, v in sorted(t11.items())))
    man.check("P1-T11 declared threshold bounds both the world chord angle and psi at 15 deg",
              "psi" in decl["tests"]["P1-T11"]["threshold"] and "15 deg" in decl["tests"]["P1-T11"]["threshold"],
              decl["tests"]["P1-T11"]["threshold"])
    man.check("gate monotone_trade text states the same four numbers",
              all(s in cf["monotone_trade"] for s in ("35.9", "20.9", "510", "4")), cf["monotone_trade"])
    man.check("best shape cell of the gate is i035_T1000_ks3 at 20.9 deg", g11["best_cell"]["cell"] == CELL_HI
              and abs(g11["best_cell"]["chord_world_stat_deg"] - d["std"][CELL_HI]) < 1e-9)
    drop = 1.0 - d["marks"][CELL_HI] / d["marks"][CELL_LO]
    d["drop"] = drop
    # heading-gain schedule (declared rule, checked, shown as a note)
    for c, T0, kh in ((CELL_LO, 600.0, 477.0), (CELL_HI, 1000.0, 763.0)):
        man.check(f"{c}: k_h follows the declared schedule 1.5 x 0.318 (1.5 T0 + 100)",
                  round(1.5 * 0.318 * (1.5 * T0 + 100.0)) == kh, decl["cells"]["heading_gain_rule"])
    src_r, src_d, src_g = str(RESULTS), str(DECL), str(GATE)
    man.value("T0, left / right", [600, 1000], "N", f"{src_d} cells.grid[].T0_N", "panel headers, captions (0.6 / 1.0 kN)")
    man.value("heading gain k_h, left / right", [477, 763], "N m/rad", f"{src_d} cells.grid[].k_h_N_m_per_rad",
              "declared schedule: cells.heading_gain_rule")
    man.value("weather intensity", 0.35, "", f"{src_d} cells.grid[].intensity", "animation subtitle "
              "'same weather (intensity 0.35)'; also in the cell names i035_... on the card")
    man.value("seed", SEED, "", f"{RUNS} runs[].seed", "subtitle, selection")
    man.value("warm-up", WARMUP, "s", f"{src_d} cells.warmup_s", "counters run from t = 20 s")
    man.value("shape band", SHAPE_BAND_DEG, "deg", f"{src_d} tests.P1-T11.threshold", "strip (labelled as its "
              "chord half) and card")
    man.value("cells where psi's spread is below the chord-angle spread", [8, 8], "",
              f"{src_r} P1_T11[*].psi_stat_deg < chord_world_stat_deg", "card note under the shape axis")
    man.value("played window", [60, 175], "s", "this module SEGMENTS (selection)", "x4, real time over 86-95 s")
    man.value("seed scan of the C2 selection", [7101, 7106], "",
              "tether/analysis/v2/paperfig/cascade_run.py main(seeds=7101..7106)", "selection text")
    for c in CELLS:
        man.value(f"{c}: chord-angle std, cell statistic", round(d["std"][c], 1), "deg",
                  f"{src_r} P1_T11[{c}].chord_world_stat_deg", "20 statistics seeds pooled, max over vessels")
        man.value(f"{c}: re-engagement marks, 20 statistics seeds", d["marks"][c], "", f"{src_r} trade_off[{c}].marks",
                  "card caption 2 'suppresses the snaps' (4 marks remain at 1.0 kN; paper: 'destroys 99.2% of the marks')")
    man.value("marks lost raising T0 0.6 -> 1.0 kN", round(100 * drop, 1), "%",
              f"derived: 1 - trade_off[{CELL_HI}].marks / trade_off[{CELL_LO}].marks", "derived")
    man.value("cells on the grid / meeting the shape band", [8, 0], "", f"{src_g} tests.P1-T11.cells_total, cells_failing")
    for c, v in sorted(d["all_std"].items()):
        man.value(f"all-cells strip: {c} chord-angle std", round(v, 1), "deg", f"{src_r} P1_T11[{c}].chord_world_stat_deg",
                  "grey dots on the card")
    man.value("closest to both: shortfall factor", round(d["combined"], 2), "x",
              f"{src_r} closest_to_both.ranked[0].combined (= {src_g} central_finding.best_combined_shortfall)",
              "entirely shape: population factor 0.45 floored to 1 under the pooled measure")
    for c in CELLS:
        r = runs[c]
        man.value(f"{c}: seed 7105 whole run, marks", r["n_marks"], "", f"{RUNS} runs[{c}, 7105].n_marks (= replay)")
        man.value(f"{c}: seed 7105 whole run, chord-angle spread", round(float(np.max(r["shape_std_deg"]["chord_world"])), 1),
                  "deg", f"{RUNS} runs[{c}, 7105].shape_std_deg.chord_world max (= replay)")
    man.value("statistics seeds per cell", 20, "", f"{src_d} cells.statistics_seeds (7103-7122)")
    man.value("run duration after warm-up", 300, "s", f"{DECL} cells.duration_s",
              "card subtitle '20 statistics seeds x 300 s' and 'whole 300 s run'")
    return d


def summary(fig, w, man: C.Manifest, d: dict, runs: dict) -> int:
    """The cell-level card, revealed in stages; returns frames written."""
    frames = 0
    for n, (stage, secs, text) in enumerate(summary_stages(d), 1):
        boxes = draw_card(fig, d, runs, stage)
        C.caption(fig, text, y=CAPTION_Y)
        layout_check(fig, man, f"cell card, step {n}", fig.texts, boxes=boxes)
        w.grab_frame()
        frames += 1 + C.hold(w, secs - 1.0 / C.FPS)
    return frames


def summary_stages(d: dict) -> list:
    """(card stage, seconds on screen, caption); the last is the end hold on the key frame (the takeaway)."""
    return [
        (1, 6.4, f"Over 20 runs at each pretension, raising it narrows the chord-angle spread\n"
                 f"from {d['std'][CELL_LO]:.1f}° to {d['std'][CELL_HI]:.1f}°."),
        (2, 6.8, f"It also suppresses the snaps: re-engagement marks fall from {d['marks'][CELL_LO]} to "
                 f"{d['marks'][CELL_HI]},\na drop of {100 * d['drop']:.1f} %."),
        (3, 8.0, f"None of the eight settings meets the 15° band; the closest to both requirements\n"
                 f"misses by a factor of {d['combined']:.2f}."),
        (3, 8.8, "More pretension steadies the shape but suppresses the slack events a design law needs,\n"
                 "so no setting on this grid gives both."),
    ]


CARD_X, CARD_W = 0.135, 0.455            # left block (the two measured halves)
CARD_RIGHT = 0.645                       # right block (the window)


def draw_card(fig, d: dict, runs: dict, stage: int) -> list:
    """Draw card ``stage`` (1: shape, 2: + slack events, 3: + all cells and the window); returns the
    panel boxes for the layout check."""
    from matplotlib.patches import FancyArrowPatch
    fig.clf()
    C.title(fig, "Pretension: one knob, two requirements",
            "Cell level: i035_T0600_ks3 and i035_T1000_ks3 · 20 statistics seeds × 300 s each")
    C.footer(fig, FOOTER_CELL)
    boxes = []
    # row A: chord-angle std (cell statistic)
    axA = fig.add_axes([CARD_X, 0.585, CARD_W, 0.17])
    axA.set_xlim(0, 60); axA.set_ylim(-1, 1); axA.set_yticks([])
    axA.spines["left"].set_visible(False)
    axA.axvspan(0, SHAPE_BAND_DEG, color=C.OK, alpha=0.12, lw=0)
    axA.axvline(SHAPE_BAND_DEG, color=C.OK, lw=1.2, ls=(0, (4, 3)))
    axA.text(7.5, 0.60, "shape band\n≤ 15°", fontsize=C.FS_SMALL, color=C.OK, ha="center", va="center")
    axA.set_xlabel("chord-angle spread, cell statistic (°)", fontsize=C.FS_SMALL)
    axA.tick_params(labelsize=C.FS_SMALL)
    fig.text(CARD_X, 0.775, "Formation shape", fontsize=C.FS_SUB, weight="bold", color=C.INK, ha="left", va="bottom")
    fig.text(CARD_X, 0.525, "circular std of each cable's world chord angle, largest of the five; 20 seeds pooled\n"
             "ψ (heading misalignment), the band's other half, has the smaller spread in all 8 cells",
             fontsize=C.FS_TINY, color=C.MUTED, ha="left", va="top", linespacing=1.3)
    if stage >= 3:
        others = [v for c, v in d["all_std"].items() if c not in CELLS]
        axA.plot(others, np.zeros(len(others)), "o", ms=9, color=C.SEVERED, zorder=2)
        axA.text(59.5, -0.75, "grey: the other six cells", fontsize=C.FS_SMALL, color=C.MUTED, ha="right",
                 va="center")
    for c, col, lab in ((CELL_LO, C_LO, "0.6 kN"), (CELL_HI, C_HI, "1.0 kN")):
        v = d["std"][c]
        axA.plot([v], [0], "o", ms=15, color=col, zorder=4)
        axA.text(v, 0.56, f"{lab}\n{v:.1f}°", fontsize=C.FS_BODY, weight="bold", color=col, ha="center",
                 va="center", linespacing=1.1)
    axA.add_patch(FancyArrowPatch((d["std"][CELL_LO] - 1.2, -0.45), (d["std"][CELL_HI] + 1.2, -0.45),
                                  arrowstyle="-|>", mutation_scale=18, color=C.INK, lw=1.4))
    axA.text(0.5 * (d["std"][CELL_LO] + d["std"][CELL_HI]), -0.75, "raise T0 (k_h by schedule)",
             fontsize=C.FS_SMALL, color=C.INK, ha="center", va="center")
    boxes.append(("shape axes", axA.get_tightbbox()))
    # row B: re-engagement marks
    if stage >= 2:
        axB = fig.add_axes([CARD_X, 0.245, CARD_W, 0.16])
        fig.text(CARD_X, 0.425, "Re-engagements (marks)", fontsize=C.FS_SUB, weight="bold", color=C.INK, ha="left",
                 va="bottom")
        vals = [d["marks"][CELL_LO], d["marks"][CELL_HI]]
        axB.barh([1, 0], vals, color=[C_LO, C_HI], height=0.55)
        axB.set_yticks([1, 0]); axB.set_yticklabels(["T0 = 0.6 kN", "T0 = 1.0 kN"], fontsize=C.FS_SMALL)
        axB.set_xlim(0, 600); axB.set_ylim(-0.6, 1.6)
        axB.tick_params(axis="x", labelsize=C.FS_SMALL)
        axB.set_xlabel("re-engagement marks, 20 seeds × 300 s", fontsize=C.FS_SMALL)
        for y, v, col in ((1, vals[0], C_LO), (0, vals[1], C_HI)):
            axB.text(v + 8, y, f"{v}", fontsize=C.FS_BODY, weight="bold", color=col, va="center")
        axB.text(595, 0.0, f"−{100 * d['drop']:.1f} %", fontsize=C.FS_TITLE, weight="bold", color=C.INK,
                 ha="right", va="center")
        boxes.append(("marks axes", axB.get_tightbbox()))
    # right block: the window
    x = CARD_RIGHT
    if stage >= 3:
        fig.text(x, 0.775, "The operating window", fontsize=C.FS_SUB, weight="bold", color=C.INK, ha="left",
                 va="bottom")
        fig.text(x, 0.745, "A design-time rate law must be fitted\n"
                           "where the formation holds its shape\n"
                           "(spread ≤ 15°) and where slack events\n"
                           "are plentiful enough to fit it.",
                 fontsize=C.FS_BODY, color=C.INK, ha="left", va="top", linespacing=1.35)
        fig.text(x, 0.54, "0 of 8 cells meet the shape band.", fontsize=C.FS_BODY, weight="bold", color=C.INK,
                 ha="left", va="top")
        fig.text(x, 0.485, f"The cell closest to both, at 0.6 kN, is\n"
                           f"{d['std'][CELL_LO]:.1f}° / 15° = {d['combined']:.2f}× outside the band.",
                 fontsize=C.FS_BODY, weight="bold", color=C.INK, ha="left", va="top", linespacing=1.35)
        fig.text(x, 0.395, "Its event supply meets the declared pooled measure,\n"
                           "not the per-class population rule (P1-T10′):\n"
                           "under the pooled measure the shortfall is all shape.",
                 fontsize=C.FS_SMALL, color=C.MUTED, ha="left", va="top", linespacing=1.3)
        lo, hi = runs[CELL_LO], runs[CELL_HI]
        fig.text(x, 0.285, f"In seed 7105 alone, over the whole 300 s run, the marks\n"
                           f"are {lo['n_marks']} vs {hi['n_marks']} and the spread is "
                           f"{np.max(lo['shape_std_deg']['chord_world']):.1f}° vs "
                           f"{np.max(hi['shape_std_deg']['chord_world']):.1f}°.",
                 fontsize=C.FS_SMALL, color=C.MUTED, ha="left", va="top", linespacing=1.3)
    return boxes


def caveats(series: dict, runs: dict, results: dict) -> list:
    """What the clip does not show or claim; every number here computed from the replay/record."""
    t0, t1 = SEGMENTS[0][0], SEGMENTS[-1][1]
    lo, hi = series[CELL_LO], series[CELL_HI]
    m = lo["marks"]
    in_win = (m["t_up"] >= t0) & (m["t_up"] <= t1)
    burst = in_win & (m["t_up"] >= 91.0) & (m["t_up"] <= 94.0)
    w = (lo["shape_t"] >= t0 - 1e-9) & (lo["shape_t"] <= t1 + 1e-9)
    assert np.array_equal(lo["shape_t"], hi["shape_t"])
    ahead = lo["shape_t"][w][hi["spread"][w] > lo["spread"][w]]
    run_lo = float(np.max(runs[CELL_LO]["shape_std_deg"]["chord_world"]))
    run_hi = float(np.max(runs[CELL_HI]["shape_std_deg"]["chord_world"]))
    k0 = int(np.flatnonzero(w)[0])
    lag = m["t_done"] - m["t_up"]
    first15 = {c: float(series[c]["shape_t"][np.flatnonzero(series[c]["spread"] > SHAPE_BAND_DEG)[0]]) for c in CELLS}
    psi_lo = float(results["P1_T11"][CELL_LO]["psi_stat_deg"])
    return [
        f"One seed is shown; its per-run spread differs from the cell statistic (seed 7105 whole run: {run_lo:.1f} vs "
        f"{run_hi:.1f} deg; cells: 35.9 vs 20.9 deg). The window was chosen to contain this seed's cascade: the 0.6 kN "
        f"side shows {int(in_win.sum())} of the run's {len(m['t_up'])} marks, {int(burst.sum())} of them in one 2 s "
        "episode; it is not a typical two minutes.",
        "The two cells differ in pretension AND heading gain (k_h raised with T0 by the declared schedule, 1.5 x the "
        "lateral stability boundary); the comparison is not pretension alone.",
        f"The strip draws the running statistic over the played window only; its value counts from t = 20 s (end of "
        f"warm-up, named in the strip title). At t = {t0:.0f} s the runs are close ({lo['spread'][k0]:.1f} vs "
        f"{hi['spread'][k0]:.1f} deg); the 1.0 kN run's running spread is the larger over {0.1 * ahead.size:.1f} s of "
        "the window" + (f" (between {ahead.min():.1f} and {ahead.max():.1f} s)" if ahead.size else "")
        + "; the gap opens after the cascade. Before the window the running statistic starts at 0 and first exceeds "
        f"15 deg at t = {first15[CELL_LO]:.1f} s (0.6 kN) and {first15[CELL_HI]:.1f} s (1.0 kN); that start-up is "
        "not drawn, and caption C6 quotes the whole-run values (the P1-T11 per-run statistic).",
        "The strip, tiles and captions show the P1-T11 world-chord statistic (max over vessels), the chord half of the "
        "(H1) shape band. The band also bounds the heading misalignment psi, whose values the clip does not show; the "
        "card states that psi's spread is below the chord spread in all 8 cells (asserted), so the band's verdict is "
        f"the chord half's. psi itself is above 15 deg in the 0.6 kN cell ({psi_lo:.1f} deg).",
        "'Operating window' means a cell where a design-time rate law could be fitted: shape within 15 deg and enough "
        "events. The closest cell's population half is met only under the declared pooled measure, not the per-class "
        "rule P1-T10'. The factor 2.39 is a property of these parameters (paper Sec. VI-VII), not a general constant.",
        "Recording cables: severance is scored, never simulated; no cable is cut in either run.",
        "Red dashes mean 'carrying no tension' (plant law clamped at 0), which can precede slack by a few ms; the "
        "tiles count slack onsets (e <= 0) and count a re-engagement when the plant's tracker completes the mark (its "
        f"peak, {1e3 * lag.min():.0f}-{1e3 * lag.max():.0f} ms after t_up in this run), so a tile never shows a peak "
        "before the plant reaches it.",
        "Bodies do not contact in the plant (no contact model), so hulls and lines may overlap on screen. Vessel "
        "labels are placed per frame by a search around each bow tip that keeps every digit off every hull, the "
        "payload, the panel corner notes and the other labels, and no body touches a corner note (both asserted in "
        "every animated frame); a label moved off its default spot "
        "(bow + 1 m) gets a dark leader to its own bow tip. Where two hulls overlap almost exactly, their leaders end "
        "close together and the pairing rests on those leaders.",
        "Cable line width grows with tension and saturates at 12 kN (common.draw_fleet), so the 13.5 kN snap is "
        "drawn at the cap; the clip's key says so (the shared common.cable_key wording 'width ∝ tension' is not "
        "used here).",
        "For reading time the animation holds two of its own instants before playing them, the clock reading "
        "'(paused)': the first frame (" + ", ".join(f"t = {h:.2f} s for {s:.1f} s" for h, s in HOLDS[:1]) + ") and "
        "the frame just after the snap (" + ", ".join(f"t = {h:.2f} s for {s:.1f} s" for h, s in HOLDS[1:]) + "). "
        "A held frame repeats a played instant; the played instants and speeds are unchanged.",
    ]


def render(man: C.Manifest, decl, runs, results, gate, specs, recs, series, preview=None, preview_dir=STILLS) -> None:
    man.story = ("Pretension is the fleet's one cheap lever over both formation shape and slack. Seed 7105 in two "
                 "cells (0.6 kN and 1.0 kN, heading gain by the declared schedule), same weather: in this seed the "
                 "1.0 kN run has no slack and no snap and a smaller chord-angle spread, but neither run meets the "
                 "15 deg shape band. At cell level (20 seeds) the spread falls 35.9 -> 20.9 deg while marks fall "
                 "510 -> 4 (-99.2 %: suppressed, not removed); no cell of eight meets the band, and the closest to "
                 "both halves misses it by 2.39x: no operating window on the grid.")
    man.selection = SELECTION_TEXT + (" Both cells use seed 7105, the same weather realisation (asserted bit-equal). "
                                      "Counters start at t = 20 s (end of warm-up). Cell-level numbers are the "
                                      "record's own (20 statistics seeds, no selection).")
    man.caveats = caveats(series, runs, results)
    d = cell_numbers(man, decl, runs, results, gate)
    build_captions(man, series, runs, decl, recs)
    timed = [(b - a, txt) for a, b, txt in ANIM_CAPTIONS] + [(s, txt) for _, s, txt in summary_stages(d)]
    for secs, txt in timed:
        words = len(txt.split())
        man.check(f"caption readable: <= 2 lines, >= 4 s, <= 2.5 words/s ({words} words, {secs:.1f} s)",
                  txt.count("\n") <= 1 and secs >= 4.0 and words / 2.5 <= secs + 0.05, txt.replace("\n", " "))
    man.check("animation captions tile the animation (played and held frames)",
              ANIM_CAPTIONS[0][0] == 0.0 and ANIM_CAPTIONS[-1][1] == ANIM_FRAMES / C.FPS
              and all(p[1] == q[0] for p, q in zip(ANIM_CAPTIONS, ANIM_CAPTIONS[1:])),
              f"{ANIM_FRAMES / C.FPS:.2f} s = {ANIM_SECONDS:.1f} s played + {HOLD_FRAMES / C.FPS:.1f} s held")
    fig = C.new_frame()
    out = C.CLIPS / f"{NAME}.mp4"
    w = C.writer() if preview is None else StillWriter(fig, preview, preview_dir)
    frames = 0
    with w.saving(fig, str(out), dpi=C.DPI):
        frames += title_card(fig, w, man)
        frames += animate(fig, w, man, recs, series, decl)
        frames += summary(fig, w, man, d, runs)
    if preview is not None:
        print(f"preview: {frames} frames ({frames / C.FPS:.2f} s); no clip or manifest written")
        return
    man.frames = frames
    man.value("playback speeds", ["x4", "real time", "x4"], "", "this module SEGMENTS", "clock label on every animated frame")
    man.value("axis scales (not data): strip t, strip spread, card spread, card marks",
              [[60, 175], [0, 50], [0, 60], [0, 600]], "s, deg, deg, count", "this module (fixed axis limits)",
              "tick labels only; the 15 deg band line is the P1-T11 threshold registered above")
    man.write(out)
    print(f"frames {frames} ({frames / C.FPS:.2f} s) -> {out.relative_to(C.REPO)}")


if __name__ == "__main__":
    main()
