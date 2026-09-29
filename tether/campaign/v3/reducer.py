"""In-worker reduction of one plan v3 run (WP1 transmission rows, WP2 offspring and margins).

Runs inside the worker on the finished ``FleetRun``; the 1 ms logs are never persisted.  The
output is a plain dict of small NumPy arrays (pickled into records/v3/cache/).  Every rule below
is the one declared in records/v3/cascade_declarations.json (``definitions``); the code is the law.

* ``tension_series``: max(0, k e + c edot) where e > 0 and alive, exactly the plant's law
  (tether/physics/fleet.py::scalar_cable_step), with the run's own k and c.
* ``transmission_rows``: one row per (mark j, neighbour i != j): the neighbour's tension at t_up,
  its minimum and maximum within one engagement period after t_up, the drop normalised by
  T_peak and by the geometry factor T_ij / 0.44 at the measured chord angles and load yaw.
* ``offspring_table``: events by the anchored 2 s rule, onsets = geometric down-crossings of
  live cables, parents by the 3 s any-cable rule; an onset is a cross-cable offspring of event E
  iff its parent re-engagement belongs to E and E's cable differs.
* ``margin_samples``: tension at B.3 clean taut samples, decimated, as per-cable quantiles.
"""

from __future__ import annotations

import math
import time as wallclock

import numpy as np

from tether.campaign import summaries as v1
from tether.campaign.v2 import events as ev
from tether.campaign.v2 import regime as rg
from tether.campaign.v2 import summaries as v2
from tether.theory import transmission as TR

WINDOW_3S = ev.EXCLUSION_WINDOW  # 3 s: offspring attribution and onset-within-window flags
MARGIN_DECIMATION_S = 1.0
MARGIN_LEVELS = 101

ROW_FIELDS = (
    "parent_mark", "event_id", "role", "cable_j", "t_up", "T_peak", "x", "all_neighbours_taut",
    "neighbour_i", "eligible_i", "T_i_at_t_up", "T_i_min", "T_i_max", "lag_of_min", "drop", "rise",
    "drop_over_Tpeak", "cos_dsigma", "geometry_factor_ij", "drop_norm", "drop_norm_cos",
    "margin_over_T0", "onset_within_3s", "onset_lag",
)


def tension_series(elongation: np.ndarray, rate: np.ndarray, alive: np.ndarray, stiffness: float, damping: float) -> np.ndarray:
    e = np.asarray(elongation, dtype=float)
    q = stiffness * e + damping * np.asarray(rate, dtype=float)
    return np.where((e > 0.0) & np.asarray(alive, dtype=bool), np.maximum(q, 0.0), 0.0)


def _first_index_at_or_after(time: np.ndarray, value: float) -> int:
    return int(np.searchsorted(time, value - 1.0e-9, side="left"))


def transmission_rows(
    record: dict,
    marks: dict,
    declustering: ev.Declustering,
    interpolant: rg.StateInterpolant,
    pretension: float,
    stiffness: float,
    damping: float,
    window: float,
    downs: ev.Crossings | None = None,
) -> dict[str, np.ndarray]:
    time = record["event_time"]
    elongation = record["elongation"]
    alive = np.asarray(record["alive"], dtype=bool)
    tension = tension_series(elongation, record["rate"], alive, stiffness, damping)
    n_samples, n = elongation.shape
    if downs is None:
        _, downs = ev.crossings(time, elongation, alive)
    down_by_cable = [np.sort(downs.time[downs.cable == c]) for c in range(n)]
    w = int(round(window / v1.EVENT_PERIOD))
    t_up = np.asarray(marks["t_up"], dtype=float)
    peak = np.asarray(marks["T_peak"], dtype=float)
    cable = np.asarray(marks["cable"], dtype=np.int64)
    m = t_up.size
    rows = {name: [] for name in ROW_FIELDS}
    if m == 0:
        return {name: np.zeros(0) for name in ROW_FIELDS}
    angles = interpolant.angles(t_up)
    directions = interpolant.direction(t_up)
    ratio = TR.closed_form_ratio(n)
    for p in range(m):
        k0 = _first_index_at_or_after(time, t_up[p])
        if k0 >= n_samples:
            continue
        k1 = min(k0 + w + 1, n_samples)
        j = int(cable[p])
        operator = TR.closed_form_operator(record["load_offsets"], float(angles["load_yaw"][p]), directions[p], stiffness)
        factor = operator / ratio
        sigma = angles["sigma"][p]
        eligible = (elongation[k0] > 0.0) & alive[k0]
        eligible[j] = False
        all_taut = bool(np.all(np.delete(eligible, j)))
        for i in range(n):
            if i == j:
                continue
            seg = tension[k0:k1, i]
            at = float(tension[k0, i])
            arg = int(np.argmin(seg))
            t_min, t_max = float(seg[arg]), float(seg.max())
            drop, rise = at - t_min, t_max - at
            cos_ds = float(math.cos(sigma[i] - sigma[j]))
            g = float(factor[i, j])
            after = down_by_cable[i][np.searchsorted(down_by_cable[i], t_up[p], side="right"):]
            lag = float(after[0] - t_up[p]) if after.size and after[0] - t_up[p] <= WINDOW_3S else math.nan
            values = (
                p, int(declustering.event_id[p]), int(declustering.role[p]), j, t_up[p], peak[p], peak[p] / pretension,
                all_taut, i, bool(eligible[i]), at, t_min, t_max, arg * v1.EVENT_PERIOD, drop, rise,
                drop / peak[p] if peak[p] > 0.0 else math.nan, cos_ds, g,
                drop / (peak[p] * g) if peak[p] > 0.0 and g != 0.0 else math.nan,
                drop / (peak[p] * cos_ds) if peak[p] > 0.0 and cos_ds != 0.0 else math.nan,
                at / pretension, bool(np.isfinite(lag)), lag,
            )
            for name, value in zip(ROW_FIELDS, values):
                rows[name].append(value)
    return {name: np.asarray(value) for name, value in rows.items()}


def offspring_table(
    record: dict,
    marks: dict,
    declustering: ev.Declustering,
    warmup: float,
    end: float,
    window: float = WINDOW_3S,
    downs: ev.Crossings | None = None,
) -> dict:
    """Events (heads), onsets and the parent/offspring relation of one run."""
    time = record["event_time"]
    n = record["elongation"].shape[1]
    if downs is None:
        _, downs = ev.crossings(time, record["elongation"], np.asarray(record["alive"], dtype=bool))
    inside = (downs.time >= warmup - 1.0e-9) & (downs.time <= end + 1.0e-9)
    onset_t, onset_c = downs.time[inside], downs.cable[inside]
    t_up = np.asarray(marks["t_up"], dtype=float)
    cable = np.asarray(marks["cable"], dtype=np.int64)
    peak = np.asarray(marks["T_peak"], dtype=float)
    po = ev.primary_onsets(onset_t, onset_c, t_up, cable, window=window)
    # masked assignment: a run with onsets but no marks has an empty event_id (addendum 1)
    has_parent = po.parent >= 0
    parent_event = np.full(po.parent.shape, -1, dtype=np.int64)
    parent_cable = np.full(po.parent.shape, -1, dtype=np.int64)
    parent_event[has_parent] = declustering.event_id[po.parent[has_parent]]
    parent_cable[has_parent] = cable[po.parent[has_parent]]
    cross = (po.parent >= 0) & (parent_cable != onset_c)
    n_events = declustering.event_count
    per_cable = np.zeros((n_events, n), dtype=np.int64)
    first_lag = np.full(n_events, np.nan)
    for k in np.flatnonzero(cross):
        e = int(parent_event[k])
        per_cable[e, int(onset_c[k])] += 1
        first_lag[e] = po.parent_lag[k] if np.isnan(first_lag[e]) else min(first_lag[e], po.parent_lag[k])
    head = declustering.event_parent
    return {
        "events": {
            "cable_j": declustering.event_cable.astype(np.int64),
            "t_up": declustering.event_time.astype(float),
            "T_peak": peak[head],
            "cluster_max_T_peak": ev.event_cluster_maxima(peak, declustering)[0],
            "n_cross": per_cable.sum(axis=1),
            "offspring_per_cable": per_cable,
            "first_lag": first_lag,
            "n_marks": declustering.event_marks.astype(np.int64),
            "primary": np.zeros(n_events, dtype=bool),
        },
        "onsets": {
            "cable": onset_c.astype(np.int64), "time": onset_t.astype(float), "primary": po.primary,
            "parent_event": parent_event.astype(np.int64), "parent_cable": parent_cable.astype(np.int64),
            "lag": po.parent_lag, "cross": cross,
        },
    }


def _events_primary_flag(table: dict, marks: dict, declustering: ev.Declustering) -> None:
    """The cross-cable tree: an event is primary iff its own onset (the parent mark's t_down) has no
    parent re-engagement on a DIFFERENT cable in the 3 s before it.

    A same-cable parent (a re-engagement of the event's own cable 2-3 s earlier, outside the 2 s
    burst window) is not transmission and is what the kernel (K_jj = 0) cannot produce; such an
    event counts as primary for theta and the rate law, and its share is reported separately.
    """
    t_down = np.asarray(marks["t_down"], dtype=float)[declustering.event_parent]
    cable = np.asarray(marks["cable"], dtype=np.int64)
    po = ev.primary_onsets(t_down, declustering.event_cable, np.asarray(marks["t_up"], dtype=float), cable, window=WINDOW_3S)
    cross_parent = (po.parent >= 0) & ~po.parent_same_cable
    table["events"]["primary"] = ~cross_parent
    table["events"]["same_cable_parent"] = (po.parent >= 0) & po.parent_same_cable
    table["events"]["parent_lag"] = np.where(cross_parent, po.parent_lag, np.nan)
    table["events"]["t_down"] = t_down
    # the event forest on the cross-cable tree: each event's parent event (-1 for a primary)
    parent_event = np.where(cross_parent, declustering.event_id[np.maximum(po.parent, 0)], -1).astype(np.int64)
    parent_event[parent_event == np.arange(parent_event.size)] = -1
    table["events"]["parent_event"] = parent_event


def margin_samples(record: dict, marks: dict, warmup: float, end: float, stiffness: float, damping: float,
                   decimation_s: float = MARGIN_DECIMATION_S, levels: int = MARGIN_LEVELS) -> dict:
    """Per-cable quantiles of the tension at B.3 clean taut samples (outside every (t_up, t_up + 3 s])."""
    time = record["event_time"]
    alive = np.asarray(record["alive"], dtype=bool)
    window_mask = (time >= warmup - 1.0e-9) & (time <= end + 1.0e-9)
    mask = ev.clean_mask(time, record["elongation"], np.asarray(marks["t_up"], dtype=float), window_mask, alive=alive)
    tension = tension_series(record["elongation"], record["rate"], alive, stiffness, damping)
    step = max(int(round(decimation_s / v1.EVENT_PERIOD)), 1)
    grid = np.linspace(0.0, 100.0, levels)
    n = tension.shape[1]
    quantiles = np.full((n, levels), np.nan)
    counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        column = tension[::step, i][mask[::step, i]]
        counts[i] = column.size
        if column.size >= 2:
            quantiles[i] = np.percentile(column, grid)
    return {"quantiles": quantiles, "counts": counts, "decimation_s": decimation_s}


def reduce_run(run, job, intervention_spec=None) -> dict:
    """The whole reduction of one finished run (v1 marks exactly as v1, then the v3 blocks)."""
    started = wallclock.perf_counter()
    spec = job.spec
    summary_v1 = v1.summarize_run(run, spec.warmup, spec.pretension)
    marks = summary_v1["marks"]
    record = v2.run_record(run)
    stiffness, damping = float(record["stiffness"]), float(record["damping"])
    end = float(record["sim_end"])
    closure = record["closure"]
    if closure is not None:
        end = min(end, float(closure[1]))
    reproduction = {"checked": False, "passed": None, "first_difference": None}
    if job.expected_t_up is not None:
        same = np.array_equal(np.asarray(marks["t_up"]), job.expected_t_up) and np.array_equal(np.asarray(marks["T_peak"]), job.expected_T_peak)
        reproduction = {"checked": True, "passed": bool(same), "first_difference": None}
        if not same:
            a, b = np.asarray(marks["t_up"]), job.expected_t_up
            k = int(np.argmax(a[: min(a.size, b.size)] != b[: min(a.size, b.size)])) if min(a.size, b.size) else 0
            reproduction["first_difference"] = {"n_marks": [int(a.size), int(b.size)], "index": k,
                                                "t_up": [float(a[k]) if k < a.size else None, float(b[k]) if k < b.size else None]}
    t_up = np.asarray(marks["t_up"], dtype=float)
    cable = np.asarray(marks["cable"], dtype=np.int64)
    declustering = ev.decluster_marks(t_up, cable, rule="anchored")
    interpolant = rg.StateInterpolant.from_log(record["state_time"], record["state"], record["load_offsets"], record["vessel_offsets"])
    period = TR.frequencies(stiffness, damping, int(record["elongation"].shape[1]))["engagement_period"]
    _, downs = ev.crossings(record["event_time"], record["elongation"], np.asarray(record["alive"], dtype=bool))
    rows = transmission_rows(record, marks, declustering, interpolant, spec.pretension, stiffness, damping, period, downs)
    rows_half = transmission_rows(record, marks, declustering, interpolant, spec.pretension, stiffness, damping, 0.5 * period, downs)
    table = offspring_table(record, marks, declustering, spec.warmup, end, downs=downs)
    _events_primary_flag(table, marks, declustering)
    margins = margin_samples(record, marks, spec.warmup, end, stiffness, damping)
    out = {
        "meta": {
            "cell": job.cell, "seed": int(job.seed), "pilot": bool(job.pilot), "stage": job.stage, "v1_cell": job.v1_cell,
            "config_hash": spec.config_hash(), "stiffness": stiffness, "damping": damping, "pretension": spec.pretension,
            "engagement_period": period, "sim_end": float(record["sim_end"]), "end": end,
            "closure": None if closure is None else [int(closure[0]), float(closure[1])],
            "exposure_s": max(0.0, end - spec.warmup),
            "primary_exposure_s": ev.excluded_exposure(t_up, spec.warmup, end),
            "wall_seconds_run": float(getattr(run, "wall_seconds", 0.0)),
        },
        "reproduction": reproduction,
        "marks": marks,
        "events": table["events"],
        "onsets": table["onsets"],
        "transmission": rows,
        "transmission_half_window": {k: rows_half[k] for k in ("parent_mark", "neighbour_i", "drop", "drop_norm")},
        "margin": margins,
        "declustering": {"event_id": declustering.event_id, "role": declustering.role},
    }
    if intervention_spec is not None:
        from tether.campaign.v3 import intervention as iv

        out["intervention"] = iv.run_interventions(run, record, marks, declustering, table, out["meta"], intervention_spec)
    out["meta"]["wall_seconds_reduce"] = wallclock.perf_counter() - started
    return out
