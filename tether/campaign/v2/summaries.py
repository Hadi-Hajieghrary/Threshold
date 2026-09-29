"""The v2 run summariser: v1's ``summarize_run`` fields, unchanged, plus the v2 machinery.

``summarize_run_v2`` calls ``tether.campaign.summaries.summarize_run`` (never modified) for
v1's fields and returns them as they are - the same objects under the same keys - with
these v2 blocks added (plan v2 IV.4, IV.9, Appendix B, Phase 1 "Observables"):

* ``marks_v2``: per v1 mark (same order, same window filter), the (H4') covariates and
  class (``regime.mark_covariates``), the B.1 event membership (``event_id``,
  ``event_role`` parent / bounce / member, ``event_parent`` as an index into the v1
  marks or -1 when the event's parent lies outside the recorded window,
  ``gap_previous``), the B.1 primary flag of the mark's onset and IV.4's ``parent``
  (the latest re-engagement on any cable in (t_onset - 3 s, t_onset]:
  ``onset_parent_cable``, ``onset_parent_lag``, ``onset_parent_same_cable``).
* ``events``: the B.1 events whose parent is a v1 mark, with member and bounce counts,
  cluster maximum of T_peak and the parent's class and primary flag.  An event's class
  is its parent's class (declared); bounces are absorbed, never dropped.
* ``onsets_v2``: every geometric slack onset in the recorded window with its primary
  flag and parent, and the primary exposure (window less the union of the
  [t_up, t_up + 3 s) windows of every re-engagement on any cable).
* ``clean``: the B.3 fleet-wide clean set - moments, level counts and taut cluster
  maxima computed exactly as v1 computes them but on taut samples outside
  (t_up, t_up + 3 s] of every re-engagement on any cable, and its exposure.
* ``closure``: formation closure as a terminal event (time, cable, exposure lost).
* ``shape``: chord-angle circular std in the world frame and in the load frame, psi
  std, load-yaw std, per vessel, with the circular sums for pooling across seeds.
* ``v2_diagnostics``: weather bit-equality, the offline W_rel against the plant log,
  mark-to-crossing matching.

The v1 fields are computed first and the v2 blocks never write into them.
"""

from __future__ import annotations

import math

import numpy as np

from tether.campaign import summaries as v1
from tether.campaign.v2 import events as ev
from tether.campaign.v2 import regime as rg

SHAPE_DECIMATION = 0.1  # s; the sway pilot's (H1) sampling (campaign/v2/sway.py)
STATE_PERIOD = 0.01
SHORT_DWELL = 2.0  # s; "short marks" for the plan's class-N evidence (v1's tau_w/4 dwell bound)
SNAP_LEVEL_REPORT = 6000.0  # N; the plan's "snaps above 6 kN"
SCHEMA = "v2-summary-1"


def circular_sums(angles: np.ndarray) -> dict[str, np.ndarray]:
    angles = np.asarray(angles, dtype=float)
    return {"cos_sum": np.cos(angles).sum(axis=0), "sin_sum": np.sin(angles).sum(axis=0), "count": int(angles.shape[0])}


def circular_std_from_sums(cos_sum, sin_sum, count) -> np.ndarray:
    resultant = np.hypot(np.asarray(cos_sum, dtype=float), np.asarray(sin_sum, dtype=float)) / max(float(count), 1.0)
    return np.sqrt(-2.0 * np.log(np.clip(resultant, 1.0e-12, 1.0)))


# ----------------------------------------------------------------------------- record extraction


def run_record(run) -> dict:
    """The arrays a v2 summary needs, detached from the Drake objects."""
    cables = run.fleet.cables
    log = cables.log
    n = log.count
    m = log.state_count
    marks = list(cables.reengagements)
    return {
        "event_time": log.event_time[:n],
        "elongation": log.elongation[:n],
        "rate": log.rate[:n],
        "relative_load": log.relative_load[:n],
        "alive": log.alive[:n],
        "state_time": log.state_time[:m],
        "state": log.state[:m],
        "all_marks": {name: np.array([getattr(mark, name) for mark in marks], dtype=float) for name in v1.MARK_FIELDS},
        "closure": cables.closure,
        "sim_end": float(run.simulator.get_context().get_time()),
        "load_offsets": run.fleet.geometry.load_offsets,
        "vessel_offsets": run.fleet.geometry.vessel_offsets,
        "stiffness": cables.cable.stiffness,
        "damping": cables.cable.damping,
        "spec": run.spec,
        "master_seed": run.master_seed,
        "run_weather": getattr(cables, "_weather", None),
        "scripted_force": getattr(cables, "_scripted_force", None) is not None,
    }


# ----------------------------------------------------------------------------- the summariser


def summarize_run_v2(run, warmup: float, pretension: float, *, weather: np.ndarray | None = None) -> dict:
    """v1's ``summarize_run`` output plus the v2 blocks (module docstring)."""
    summary_v1 = v1.summarize_run(run, warmup, pretension)
    blocks = summarize_record(run_record(run), warmup, pretension, weather=weather, v1_marks=summary_v1["marks"])
    out = dict(summary_v1)
    out.update(blocks)
    return out


def summarize_record(record: dict, warmup: float, pretension: float, *, weather: np.ndarray | None = None,
                     v1_marks: dict | None = None) -> dict:
    """The v2 blocks from a ``run_record``."""
    time = record["event_time"]
    elongation = record["elongation"]
    rate = record["rate"]
    relative_load = record["relative_load"]
    alive = record["alive"]
    n_samples, n_cables = elongation.shape
    end = record["sim_end"]
    closure = record["closure"]
    if closure is not None:
        end = min(end, float(closure[1]))
    window = (time >= warmup - 1.0e-9) & (time <= end + 1.0e-9)
    if record["state_time"].size < 2:
        raise ValueError("the v2 summary needs the 10 ms state log (spec.log_state)")

    # ---- weather and the conjugate loads
    spec = record["spec"]
    regenerated = weather is None
    if weather is None:
        weather = rg.regenerate_weather(spec, record["master_seed"], n_cables)
    run_weather = record.get("run_weather")
    weather_equal = None if run_weather is None or weather is None else bool(
        run_weather.shape == weather.shape and np.array_equal(run_weather, weather))
    interpolant = rg.StateInterpolant.from_log(record["state_time"], record["state"], record["load_offsets"],
                                               record["vessel_offsets"])
    if weather is None:
        weather = np.zeros((int(math.ceil((spec.warmup + spec.duration) / rg.WEATHER_PERIOD)) + 2, n_cables + 1, 2))
    wrel_offline, wc = rg.weather_side_series(time, weather, interpolant)
    tick_index = np.rint(record["state_time"] / rg.EVENT_PERIOD).astype(np.int64)
    tick_index = tick_index[tick_index < n_samples]
    tick_ok = np.allclose(time[tick_index], record["state_time"][: tick_index.size], rtol=0.0, atol=1.0e-12)
    tick_diff = wrel_offline[tick_index] - relative_load[tick_index]
    all_diff = wrel_offline - relative_load
    log_abs = np.abs(relative_load[tick_index])
    big = log_abs >= 1.0
    rms_log = float(np.sqrt(np.mean(relative_load[tick_index] ** 2))) if tick_index.size else float("nan")
    wrel_check = {
        "ticks_compared": int(tick_index.size * n_cables),
        "tick_times_aligned": bool(tick_ok),
        "max_abs_diff_N": float(np.max(np.abs(tick_diff))) if tick_diff.size else None,
        "max_rel_diff_where_abs_log_ge_1N": float(np.max(np.abs(tick_diff)[big] / log_abs[big])) if big.any() else None,
        "rms_log_N": rms_log,
        "passes_1e-6_relative": bool(tick_diff.size and np.max(np.abs(tick_diff)) <= 1.0e-6 * max(rms_log, 1.0)
                                     and (not big.any() or np.max(np.abs(tick_diff)[big] / log_abs[big]) <= 1.0e-6)),
        "interpolated_1ms_max_abs_diff_N": float(np.max(np.abs(all_diff))),
        "interpolated_1ms_rms_diff_N": float(np.sqrt(np.mean(all_diff**2))),
    }

    # ---- crossings, marks, onsets
    ups, downs = ev.crossings(time, elongation, alive)
    all_marks = record["all_marks"]
    keep = (all_marks["t_up"] - all_marks["dwell"] >= warmup) & (all_marks["t_up"] <= end)
    window_rows = np.flatnonzero(keep)  # v1's filter, in v1's order
    all_t_up = all_marks["t_up"]
    all_cable = all_marks["cable"].astype(np.int64)
    marks = {name: all_marks[name][window_rows] for name in v1.MARK_FIELDS}
    marks["cable"] = marks["cable"].astype(np.int64)
    if v1_marks is not None:
        same_order = bool(np.array_equal(v1_marks["t_up"], marks["t_up"]) and np.array_equal(v1_marks["cable"], marks["cable"]))
    else:
        same_order = None
    up_match = ev.match_marks(marks["t_up"], marks["cable"], ups)
    down_k = np.full(window_rows.size, -1, dtype=np.int64)
    for position in range(window_rows.size):
        candidates = np.flatnonzero((downs.cable == marks["cable"][position]) & (downs.time < marks["t_up"][position]))
        if candidates.size:
            down_k[position] = candidates[-1]
    matched = (up_match >= 0) & (down_k >= 0)
    up_index = np.where(up_match >= 0, ups.index[np.maximum(up_match, 0)], -1)
    down_index = np.where(down_k >= 0, downs.index[np.maximum(down_k, 0)], -1)
    down_time = np.where(down_k >= 0, downs.time[np.maximum(down_k, 0)], np.nan)
    onset_residual = np.abs(down_time - (marks["t_up"] - marks["dwell"]))

    covariates = rg.mark_covariates(
        time=time, elongation=elongation, relative_load=relative_load, wc=wc, marks=marks,
        up_index=up_index, down_index=down_index, down_time=down_time, interpolant=interpolant,
        pretension=pretension,
    )

    # ---- B.1 primary onsets (every onset in the window) and IV.4's parent
    in_window_onset = (downs.time >= warmup) & (downs.time <= end)
    onset_rows = np.flatnonzero(in_window_onset)
    primary = ev.primary_onsets(downs.time[onset_rows], downs.cable[onset_rows], ups.time, ups.cable)
    primary_2s_same = ev.primary_onsets(downs.time[onset_rows], downs.cable[onset_rows], ups.time, ups.cable,
                                        window=ev.BURST_WINDOW, same_cable_only=True)
    primary_exposure = ev.excluded_exposure(ups.time, warmup, end)
    onset_of_row = {int(row): k for k, row in enumerate(onset_rows)}
    onset_mark = np.full(onset_rows.size, -1, dtype=np.int64)
    mark_onset = np.array([onset_of_row.get(int(k), -1) for k in down_k], dtype=np.int64)
    for position, k in enumerate(mark_onset):
        if k >= 0:
            onset_mark[k] = position

    def per_mark(values, fill):
        values = np.asarray(values)
        out = np.full(mark_onset.size, fill)
        valid = mark_onset >= 0
        out[valid] = values[mark_onset[valid]]
        return out

    parent_cable = np.where(primary.parent >= 0, ups.cable[np.maximum(primary.parent, 0)], -1)

    # ---- B.1 events over every mark of the run
    declustering = ev.decluster_marks(all_t_up, all_cable)
    chained = ev.decluster_marks(all_t_up, all_cable, rule="chained")
    window_position = np.full(all_t_up.size, -1, dtype=np.int64)
    window_position[window_rows] = np.arange(window_rows.size)
    event_parent_all = declustering.parent_index[window_rows]
    event_parent = window_position[event_parent_all]
    role_names = np.array(declustering.role_names(), dtype=str) if all_t_up.size else np.empty(0, dtype=str)

    marks_v2 = dict(covariates)
    marks_v2.update({
        "event_id": declustering.event_id[window_rows],
        "event_role": role_names[window_rows] if window_rows.size else np.empty(0, dtype=str),
        "event_parent": event_parent,
        "gap_previous": declustering.gap_previous[window_rows],
        "primary": per_mark(primary.primary, False),
        "onset_parent_cable": per_mark(parent_cable, -1),
        "onset_parent_lag": per_mark(primary.parent_lag, np.nan),
        "onset_parent_same_cable": per_mark(primary.parent_same_cable, False),
        "primary_2s_same_cable": per_mark(primary_2s_same.primary, False),
        "matched_crossings": matched,
    })

    # events whose parent is a v1 (window) mark
    parents_in_window = window_position[declustering.event_parent] >= 0
    event_rows = np.flatnonzero(parents_in_window)
    maxima, argmax = ev.event_cluster_maxima(all_marks["T_peak"], declustering)
    parent_position = window_position[declustering.event_parent[event_rows]]
    events = {
        "event_id": event_rows,
        "cable": declustering.event_cable[event_rows],
        "t_event": declustering.event_time[event_rows],
        "t_last": declustering.event_last[event_rows],
        "parent_mark": parent_position,
        "n_marks": declustering.event_marks[event_rows],
        "n_bounces": declustering.event_bounces[event_rows],
        "n_members_beyond_bounce": declustering.event_marks[event_rows] - 1 - declustering.event_bounces[event_rows],
        "cluster_max_T_peak": maxima[event_rows],
        "cluster_max_mark": window_position[argmax[event_rows]] if event_rows.size else np.empty(0, dtype=np.int64),
        "parent_T_peak": marks["T_peak"][parent_position] if event_rows.size else np.empty(0),
        "parent_u_entry": marks["u_entry"][parent_position] if event_rows.size else np.empty(0),
        "regime": covariates["regime"][parent_position] if event_rows.size else np.empty(0, dtype=str),
        "primary": marks_v2["primary"][parent_position] if event_rows.size else np.empty(0, dtype=bool),
    }
    orphans = int(np.sum(event_parent < 0))
    chained_in_window = int(np.sum(window_position[chained.event_parent] >= 0))

    onsets_v2 = {
        "time": downs.time[onset_rows],
        "cable": downs.cable[onset_rows],
        "primary": primary.primary,
        "parent_cable": parent_cable,
        "parent_lag": primary.parent_lag,
        "parent_same_cable": primary.parent_same_cable,
        "primary_2s_same_cable": primary_2s_same.primary,
        "mark": onset_mark,
        "count": int(onset_rows.size),
        "primary_count": int(np.sum(primary.primary)),
        "primary_2s_same_cable_count": int(np.sum(primary_2s_same.primary)),
        "exposure": max(0.0, end - warmup),
        "primary_exposure": primary_exposure,
        "per_cable": np.bincount(downs.cable[onset_rows], minlength=n_cables),
        "per_cable_primary": np.bincount(downs.cable[onset_rows][primary.primary], minlength=n_cables),
    }

    # ---- B.3 clean set, v1's statistics on it
    clean = ev.clean_mask(time, elongation, ups.time, window, alive=alive)
    clean_block = _clean_statistics(time, elongation, rate, clean, record["stiffness"], record["damping"], pretension)
    clean_block["exposure"] = ev.excluded_exposure(ups.time, warmup, end)
    clean_block["window_exposure"] = max(0.0, end - warmup)
    clean_block["excluded_fraction"] = (1.0 - clean_block["exposure"] / clean_block["window_exposure"]
                                        if clean_block["window_exposure"] > 0 else None)
    clean_block["taut_samples_in_window"] = ((elongation > 0.0) & window[:, None]).sum(axis=0)

    # ---- closure (B.7), a terminal event
    planned_end = float(spec.warmup + spec.duration)
    if closure is None:
        closure_block = {"terminal": False, "time": None, "cable": None, "exposure_lost": 0.0, "planned_end": planned_end,
                         "in_window": None}
    else:
        closure_time = float(closure[1])
        closure_block = {"terminal": True, "time": closure_time, "cable": int(closure[0]),
                         "exposure_lost": max(0.0, planned_end - max(closure_time, warmup)), "planned_end": planned_end,
                         "in_window": bool(closure_time >= warmup)}

    # ---- shape statistics (H1)
    shape = _shape_statistics(record, interpolant, warmup, end, n_cables)

    # ---- per-run regime summary
    regime_summary = _regime_summary(marks, covariates, events, marks_v2)

    diagnostics = {
        "weather_regenerated": regenerated,
        "weather_bit_equal_to_run": weather_equal,
        "scripted_force_present": bool(record.get("scripted_force")),
        "wrel_offline_vs_log": wrel_check,
        "marks_same_order_as_v1": same_order,
        "marks_unmatched_to_crossings": int(np.sum(~matched)),
        "onset_time_max_residual_s": float(np.nanmax(onset_residual)) if onset_residual.size else None,
        "up_crossings": ups.size,
        "down_crossings": downs.size,
        "all_marks": int(all_t_up.size),
        "window_marks": int(window_rows.size),
        "window_marks_whose_event_parent_is_outside_window": orphans,
        "events_in_window_chained_rule_sensitivity": chained_in_window,
    }
    return {
        "marks_v2": marks_v2,
        "events": events,
        "onsets_v2": onsets_v2,
        "clean": clean_block,
        "closure": closure_block,
        "shape": shape,
        "regime_summary": regime_summary,
        "v2_diagnostics": diagnostics,
        "v2_meta": {"schema": SCHEMA, "pretension": pretension, "v_T": pretension / rg.C_A, "constants": rg.constants_table(),
                    "burst_window": ev.BURST_WINDOW, "exclusion_window": ev.EXCLUSION_WINDOW,
                    "engagement_period": ev.ENGAGEMENT_PERIOD, "shape_decimation": SHAPE_DECIMATION},
    }


def _clean_statistics(time, elongation, rate, clean, stiffness, damping, pretension) -> dict:
    """v1's moments, declustered level counts and taut cluster maxima, on the B.3 mask."""
    n_samples, n_cables = elongation.shape
    q = stiffness * elongation + damping * rate
    acceleration = np.zeros_like(rate)
    acceleration[1:-1] = (rate[2:] - rate[:-2]) / (2.0 * v1.EVENT_PERIOD)
    interior = np.zeros(n_samples, dtype=bool)
    interior[1:-1] = True
    qdot = stiffness * rate + damping * acceleration
    moments = {}
    for name, values, mask in (
        ("e", elongation, clean),
        ("edot", rate, clean),
        ("eddot", acceleration, clean & interior[:, None]),
        ("q", q, clean),
        ("qdot", qdot, clean & interior[:, None]),
    ):
        count = mask.sum(axis=0).astype(np.int64)
        total = np.where(mask, values, 0.0).sum(axis=0)
        square = np.where(mask, values * values, 0.0).sum(axis=0)
        moments[name] = np.stack([count.astype(float), total, square])
    run_samples = int(round(v1.ENGAGEMENT_PERIOD / v1.EVENT_PERIOD))
    level_counts = np.zeros((v1.LEVEL_GRID.size, n_cables), dtype=np.int64)
    peak_times, peak_values, peak_cables = [], [], []
    floor = v1.TAUT_PEAK_FLOOR_RATIO * pretension
    for cable in range(n_cables):
        level_counts[:, cable] = v1.declustered_upcrossing_counts(q[:, cable], clean[:, cable], v1.LEVEL_GRID, run_samples)
        times, peaks = v1.cluster_maxima(time, q[:, cable], clean[:, cable], floor, run_samples)
        peak_times.append(times)
        peak_values.append(peaks)
        peak_cables.append(np.full(peaks.size, cable, dtype=np.int64))
    return {
        "moments": moments,
        "level_counts": level_counts,
        "taut_peaks": {"time": np.concatenate(peak_times), "peak": np.concatenate(peak_values),
                       "cable": np.concatenate(peak_cables)},
        "samples": clean.sum(axis=0),
    }


def _shape_statistics(record, interpolant, warmup, end, n_cables) -> dict:
    state_time = record["state_time"]
    states = record["state"]
    rows = np.flatnonzero((state_time >= warmup - 1.0e-9) & (state_time <= end + 1.0e-9))
    rows = rows[:: int(round(SHAPE_DECIMATION / STATE_PERIOD))]
    if rows.size == 0:
        nan = np.full(n_cables, np.nan)
        return {"samples": 0, "chord_world_std": nan, "chord_load_std": nan, "psi_std": nan, "load_yaw_std": float("nan")}
    direction = rg.unit(interpolant.chord[rows])
    sigma = np.arctan2(direction[..., 1], direction[..., 0])
    load_yaw = states[rows, 2]
    headings = states[rows, 5 : 3 * (n_cables + 1) : 3]
    relative = np.angle(np.exp(1j * (sigma - load_yaw[:, None])))
    psi = np.angle(np.exp(1j * (headings - sigma)))
    sums = {"chord_world": circular_sums(sigma), "chord_load": circular_sums(relative), "psi": circular_sums(psi),
            "load_yaw": circular_sums(load_yaw[:, None])}
    out = {"samples": int(rows.size), "decimation": SHAPE_DECIMATION, "sums": sums}
    for name, block in sums.items():
        std = circular_std_from_sums(block["cos_sum"], block["sin_sum"], block["count"])
        out[f"{name}_std"] = float(std[0]) if name == "load_yaw" else std
    out["chord_world_std_max_deg"] = float(np.degrees(np.max(out["chord_world_std"])))
    out["chord_load_std_max_deg"] = float(np.degrees(np.max(out["chord_load_std"])))
    out["psi_std_max_deg"] = float(np.degrees(np.max(out["psi_std"])))
    out["load_yaw_std_deg"] = float(np.degrees(out["load_yaw_std"]))
    return out


def _regime_summary(marks, covariates, events, marks_v2) -> dict:
    regime = covariates["regime"]
    short = np.asarray(marks["dwell"]) < SHORT_DWELL
    snaps = np.asarray(marks["T_peak"]) > SNAP_LEVEL_REPORT
    return {
        "marks": rg.class_counts(regime),
        "marks_clip_reading": rg.class_counts(covariates["regime_clip"]),
        "short_marks": rg.class_counts(regime, short),
        "short_marks_above_6kN": rg.class_counts(regime, short & snaps),
        "marks_above_6kN": rg.class_counts(regime, snaps),
        "events": rg.class_counts(events["regime"]),
        "primary_marks": rg.class_counts(regime, marks_v2["primary"]),
        "roles": {name: int(np.sum(marks_v2["event_role"] == name)) for name in (ev.ROLE_PARENT, ev.ROLE_BOUNCE, ev.ROLE_MEMBER)},
    }


# =========================================================================== validation on v1 records
#
# ``python -m tether.campaign.v2.summaries declare`` writes the declarations (refuses to
# overwrite); ``validate --workers 2`` regenerates v1 Phase 2 runs and writes
# ``records/v2/machinery/machinery_validation.json``.

VALIDATION_DIR_NAME = "machinery"
VALIDATION_CELLS = ("T1000_k500_I1.0", "T1000_k500_I0.5")
VALIDATION_SEEDS = (2003, 2004, 2005)
V1_CACHE_RELATIVE = ("phase2", "cache", "phase2_compute.pkl")


def _paths():
    from tether.campaign.common import RECORDS

    directory = RECORDS / "v2" / VALIDATION_DIR_NAME
    return directory / "machinery_declarations.json", directory / "machinery_validation.json"


VALIDATION_DECLARATIONS = {
    "schema": "v2-machinery-declarations-1",
    "purpose": ("Validate the v2 event and regime machinery (tether/campaign/v2/events.py, regime.py, summaries.py) on "
                "regenerated v1 Phase 2 runs before any v2 phase uses it. Plan v2 IV.4, IV.9 'Event definition', II.2 (H4'), "
                "Appendix B.1-B.4, B.7, Phase 1 'Observables'. Nothing here is a phase verdict; the checks below are the "
                "machinery's acceptance and the class shares are reported against the plan's v1 evidence."),
    "cells": {"primary": "T1000_k500_I1.0 (v1 Phase 2 FleetRunSpec from tether.campaign.phase2.cell_specs(): parallel, "
                         "T0 = 1000 N, k_h = 500, k_sigma = 0 (v1 controller), Gaussian local weather at intensity 1.0, "
                         "600 s after a 20 s warm-up, recording mode)",
              "secondary": "T1000_k500_I0.5 (same, intensity 0.5), reported for its class shares at the lower intensity",
              "seeds": list(VALIDATION_SEEDS), "workers": 2},
    "definitions": {
        "event_B1": ("anchored: a mark of cable i joins the open event of cable i while t_up - t_event <= 2 s (inclusive), "
                     "t_event the event's first t_up; the first mark beyond opens a new event. The literal 'marks within 2 s on "
                     "the same cable belong to it', 'it' being the event. The chained reading (within 2 s of the previous "
                     "member) is computed as a sensitivity count only. Same rule as campaign/v2/p6_t0.py"),
        "bounce": ("absorbed, never dropped: every mark belongs to exactly one event; role 'parent' (the event's first mark), "
                   "'bounce' (later member whose t_up gap to the preceding mark of the same cable is <= one engagement "
                   "period 2 pi sqrt(m_eff/k) = 0.3354 s), 'member' (later member beyond one engagement period)"),
        "event_class": "an event's (H4') class is its parent mark's class; its snap statistic is the cluster maximum of T_peak over its members",
        "event_window": ("events are formed on every mark of the run (warm-up included); an event is counted for the window iff "
                         "its parent is one of v1's window marks (t_up - dwell >= warmup and t_up <= end); window marks whose "
                         "parent precedes the window keep their membership and are counted as orphans"),
        "reengagement_set": ("every geometric up-crossing of e = 0 of a live cable in the 1 ms log, interpolated as the plant's "
                             "tracker interpolates t_up (a superset of the recorded marks)"),
        "primary_onset_B1": ("onsets = geometric down-crossings of e = 0 (tracker interpolation) with time in [warmup, end]; "
                             "primary iff no re-engagement on any cable has t_up in (t - 3 s, t]; primary exposure = "
                             "[warmup, end] less the union of [t_up, t_up + 3 s); IV.4's 'parent' = the latest such "
                             "re-engagement, with its cable and lag. The 2 s same-cable variant (Phase 2 T2's declared "
                             "re-scoring) is computed beside it"),
        "clean_set_B3": ("taut (e > 0) samples in [warmup, end] outside (t_up, t_up + 3 s] of every re-engagement on any cable; "
                         "exposure = [warmup, end] less the union of those windows; v1's moments, run-declustered level "
                         "counts and taut cluster maxima (engagement-period runs, floor 1.2 T0) recomputed on it"),
        "W_rel": ("the plant's 1 ms relative_load log (m_eff (W_L/m_L - W_A/m_A) . d_i) is used; an offline recomputation from "
                  "the regenerated weather and the 10 ms state log is checked against it at the 10 ms rows"),
        "W_c": ("c_eff (W_L/c_L - W_A/c_A) . d_i, c_eff = (1/c_A + 1/c_L)^-1 = 329.1 N s/m, on the 1 ms grid: weather by "
                "the plant's zero-order-hold index floor((t + 1e-9)/0.01), d_i the chord vector linearly interpolated between "
                "the bracketing 10 ms state rows and normalised"),
        "weather": "regenerated from (spec, seed) exactly as fleet_run.build_run calls stationary_weather_forces; checked bit-equal to the run's own array",
        "slack_interval": "a mark's slack samples: from its onset (the cable's last down-crossing before t_up) to its up-crossing, i.e. its e <= 0 samples",
        "episode_B4": ("maximal runs of W^c > T0 on the cable's whole-run 1 ms series after merging runs separated by gaps "
                       "shorter than 1 s (< 1000 samples); duration = last - first exceeding sample + 1 ms; peak = max W^c"),
        "t_x_primary": ("IV.4/B.4 literal: the full duration and peak of the merged episode containing the mark's in-slack "
                        "exceedances; if several do, the one holding most in-slack exceedance samples (earliest on a tie)"),
        "t_x_secondary": "the same episode clipped to the slack interval (t_x_clip, Wc_max_clip, regime_clip); reported, gates nothing",
        "classes_H4prime": ("N: no W^c > T0 sample in the slack interval; R1: t_x < tau_Lf = 0.7101 s and v_s = (W^c_max - T0) "
                            "t_x/m_A < 0.3 v_T, v_T = T0/c_A; R2: t_x > 2 tau_A = 3.4286 s; transitional otherwise; T0 the "
                            "cell's nominal pretension; dwell reported, never used"),
        "return_leg": "samples from argmin e over the slack interval to re-engagement; mean W^c and mean W_rel reported",
        "a_bar_ret": "Prop. 3' identity V_up^2/(2 Delta) from the deepest point (v_return, depth of the mark); never the offset estimator",
        "closure_B7": "terminal event: time, cable, exposure lost = planned end - max(closure time, warmup)",
        "shape": ("state rows in [warmup, end] every 0.1 s (the sway pilot's (H1) sampling): circular std of the world chord "
                  "angle sigma_i, of the load-frame chord angle wrap(sigma_i - theta_0), of psi_i = wrap(theta_i - sigma_i), "
                  "and of the load yaw theta_0; per vessel, with circular sums for pooling. v1's 1 s world-frame "
                  "cable_angle_std stays in v1's geometry_stats"),
        "short_marks": ("the plan's 'short marks' is not defined in the plan; read as dwell < 2 s (= tau_w/4, v1's (H4) dwell "
                        "bound), declared here; shares over all marks are reported beside it"),
    },
    "acceptance_checks": {
        "C1_v1_fields_bit_for_bit": ("on the same run object, every v1 field of summarize_run_v2's output is byte-identical "
                                     "(dtype, shape, bytes; scalars and lists by equality) to a separate summarize_run call "
                                     "made before it"),
        "C2_weather_bit_equal": "the regenerated weather array equals the run's own array exactly",
        "C3_W_rel_matches_log": "offline W_rel at every 10 ms state row within 1e-6 relative of the plant's relative_load",
        "C4_marks_matched": "every window mark's t_up is an up-crossing (1e-9 s) and its t_up - dwell its onset (1e-9 s)",
        "C5_onsets_consistent": ("window onset count equals v1's per-cable 'onsets' count, differences allowed only for a "
                                 "crossing straddling a window edge (reported)"),
        "C6_hand_checks": ("one mark per class present: W^c recomputed at the 10 ms rows of its slack interval with "
                           "fleet.cable_kinematics (an independent code path) equals the machinery's value at those rows "
                           "within 1e-9 relative; the class re-derived by a scalar scan of the 10 ms W^c series (episodes, "
                           "1 s merge, t_x to 10 ms) agrees unless t_x lies within 20 ms of a class boundary"),
        "C7_reproducibility": ("reported, not a machinery gate: v1 fields against the v1 cached summaries "
                               "(records/phase2/cache/phase2_compute.pkl) for the same (cell, seed), meta.wall_seconds excluded"),
    },
    "plan_evidence_compared": {
        "class_N_share_of_short_marks": "65-84% (plan II.5, Phase 1)",
        "R2_by_exceedance_share_of_marks": "0-4% (Phase 1)",
        "class_N_share_of_short_mark_snaps_above_6kN": "about 95% (plan II.5)",
        "bounces_IV9": "41-69% of attributed offspring are restitution bounces; median gap 0.19 s; 86% inside 0.335 s",
        "clean_set_IV4": "negligible where slack is rare (in-window fraction <= 0.09, moments within 1%); large at intensity 1.0",
        "note": ("the plan does not say which t_x reading produced its 0-4%; both readings are reported. No definition is "
                 "changed on seeing these numbers"),
    },
    "development_disclosure": ("before this file was written, one development run (T1000_k500_I1.0 spec at 80 s after the "
                               "20 s warm-up, seed 2004; not part of this validation) was used to debug the code. It showed "
                               "the full-episode (IV.4) and clipped readings differ materially for R2 (5 vs 0 of 25 marks). "
                               "The primary reading had been fixed from IV.4/B.4's text before that run and is unchanged."),
}


def _bytes_equal(a, b, path="") -> list[str]:
    """Paths at which two v1 summaries differ (byte-level for arrays)."""
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return [path or "/"]
        out = []
        for key in a:
            out += _bytes_equal(a[key], b[key], f"{path}/{key}")
        return out
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a_arr, b_arr = np.asarray(a), np.asarray(b)
        same = a_arr.dtype == b_arr.dtype and a_arr.shape == b_arr.shape and a_arr.tobytes() == b_arr.tobytes()
        return [] if same else [path]
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return [path]
        out = []
        for k, (x, y) in enumerate(zip(a, b)):
            out += _bytes_equal(x, y, f"{path}[{k}]")
        return out
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if (type(a) is type(b) and a == b) else [path]


def _hand_check(run, summary, position: int, pretension: float) -> dict:
    """Independent recomputation of one mark's W^c and class on the 10 ms rows."""
    from tether.physics.fleet import cable_kinematics

    cables = run.fleet.cables
    log = cables.log
    weather = cables._weather
    geometry = run.fleet.geometry
    rest = cables.cable.rest_length
    marks = summary["marks"]
    extra = summary["marks_v2"]
    cable = int(marks["cable"][position])
    t_up = float(marks["t_up"][position])
    t_down = float(extra["t_down"][position])
    state_time = log.state_time[: log.state_count]
    states = log.state[: log.state_count]

    def wc_at_row(k: int) -> float:
        kin = cable_kinematics(states[k], geometry, rest)
        index = min(int(math.floor((state_time[k] + 1.0e-9) / rg.WEATHER_PERIOD)), weather.shape[0] - 1)
        w = weather[index]
        d = kin["direction"][cable]
        return float(rg.C_EFF * ((w[0, 0] / rg.C_L - w[cable + 1, 0] / rg.C_A) * d[0]
                                 + (w[0, 1] / rg.C_L - w[cable + 1, 1] / rg.C_A) * d[1]))

    # the whole-run 10 ms W^c series of this cable, by a scalar loop
    series = np.array([wc_at_row(k) for k in range(state_time.size)])
    inside = np.flatnonzero((state_time >= t_down) & (state_time <= t_up))
    machine_rows = np.rint(state_time[inside] / rg.EVENT_PERIOD).astype(np.int64)
    # the machinery's W^c at the same rows, recomputed through the module
    wrel_unused, wc_machine = rg.weather_side_series(log.event_time[machine_rows], weather,
                                                     rg.StateInterpolant.from_log(state_time, states, geometry.load_offsets,
                                                                                  geometry.vessel_offsets))
    wc_machine = wc_machine[:, cable]
    rel = float(np.max(np.abs(series[inside] - wc_machine) / np.maximum(np.abs(series[inside]), 1.0))) if inside.size else None
    # scalar episode scan on the 10 ms rows
    exceed = series > pretension
    episode = None
    in_slack = [k for k in inside if exceed[k]]
    if in_slack:
        k0 = in_slack[0]
        start = k0
        while True:  # walk back through exceedances and gaps < 1 s
            j = start - 1
            while j >= 0 and not exceed[j]:
                j -= 1
            if j >= 0 and (start - j - 1) * 0.01 < 1.0 - 1.0e-9:
                start = j
                while start - 1 >= 0 and exceed[start - 1]:
                    start -= 1
            else:
                break
        stop = k0
        while True:
            j = stop + 1
            while j < series.size and not exceed[j]:
                j += 1
            if j < series.size and (j - stop - 1) * 0.01 < 1.0 - 1.0e-9:
                stop = j
                while stop + 1 < series.size and exceed[stop + 1]:
                    stop += 1
            else:
                break
        t_x = (stop - start + 1) * 0.01
        peak = float(np.max(series[start : stop + 1]))
        episode = {"start": float(state_time[start]), "end": float(state_time[stop]), "t_x_10ms": t_x, "Wc_max_10ms": peak,
                   "v_s_10ms": (peak - pretension) * t_x / rg.M_A}
        by_hand = rg.classify(t_x, peak, pretension, True)
    else:
        by_hand = "N"
    machine = str(extra["regime"][position])
    t_x_machine = float(extra["t_x"][position])
    near_boundary = any(abs(t_x_machine - b) <= 0.02 for b in (rg.R1_DURATION, rg.R2_DURATION)) if machine != "N" else False
    return {
        "mark": position, "cable": cable, "t_down": t_down, "t_up": t_up, "dwell": float(marks["dwell"][position]),
        "T_peak": float(marks["T_peak"][position]),
        "rows_in_slack": int(inside.size),
        "Wc_rows_in_slack_by_hand": series[inside].tolist()[:40],
        "Wc_rows_max_rel_diff_vs_machinery": rel,
        "episode_by_hand": episode,
        "class_by_hand": by_hand,
        "class_machinery": machine,
        "machinery": {"t_x": t_x_machine, "Wc_max": float(extra["Wc_max"][position]), "v_s": float(extra["v_s"][position]),
                      "n_episodes": int(extra["n_episodes"][position]), "t_x_clip": float(extra["t_x_clip"][position]),
                      "regime_clip": str(extra["regime_clip"][position]), "Wc_slack_max": float(extra["Wc_slack_max"][position]),
                      "Wc_return_mean": float(extra["Wc_return_mean"][position]),
                      "v_T": pretension / rg.C_A, "R1_v_s_bound": rg.R1_SPEED_FRACTION * pretension / rg.C_A},
        "agree": by_hand == machine,
        "near_boundary": near_boundary,
        "passes": bool((rel is None or rel <= 1.0e-9) and (by_hand == machine or near_boundary)),
    }


def _validation_job(job) -> dict:
    import copy

    from tether.campaign.fleet_run import build_run, run_to_end

    cell, spec, seed = job
    run = run_to_end(build_run(spec, seed))
    reference = copy.deepcopy(v1.summarize_run(run, spec.warmup, spec.pretension))
    summary = summarize_run_v2(run, spec.warmup, spec.pretension)
    v1_keys = list(reference)
    diff_same_run = _bytes_equal({k: reference[k] for k in v1_keys}, {k: summary[k] for k in v1_keys})
    hand = {}
    regime = summary["marks_v2"]["regime"]
    for name in rg.CLASSES:
        rows = np.flatnonzero(regime == name)
        if rows.size:
            hand[name] = _hand_check(run, summary, int(rows[0]), spec.pretension)
    light = {key: value for key, value in summary.items() if key not in ("samples", "block_maxima")}
    return {"cell": cell, "seed": seed, "summary": light, "v1_reference": reference, "v1_diff_same_run": diff_same_run,
            "hand_checks": hand, "wall_seconds": run.wall_seconds}


class _Job(tuple):
    @property
    def cost(self) -> float:
        return float(self[1].duration + self[1].warmup)


def declare() -> str:
    from tether.campaign.common import json_bytes, write_bytes

    declarations_path, results_path = _paths()
    if declarations_path.exists():
        raise SystemExit(f"{declarations_path} exists; write a dated addendum instead of editing it")
    return write_bytes(declarations_path, json_bytes(VALIDATION_DECLARATIONS))


def validate(workers: int = 2) -> dict:
    import pickle

    from tether.campaign.common import RECORDS, json_bytes, run_pool, sha256_file, source_state, write_bytes
    from tether.campaign.phase2 import cell_specs

    declarations_path, results_path = _paths()
    if not declarations_path.exists():
        raise SystemExit("declarations must be written before any result")
    declarations_sha = sha256_file(declarations_path)
    specs = cell_specs()
    jobs = [_Job((cell, specs[cell], seed)) for cell in VALIDATION_CELLS for seed in VALIDATION_SEEDS]
    runs = run_pool(_validation_job, jobs, workers)
    scratch = results_path.parent / "cache"
    scratch.mkdir(parents=True, exist_ok=True)
    with (scratch / "machinery_validation_runs.pkl").open("wb") as handle:
        pickle.dump(runs, handle)
    cache_path = RECORDS.joinpath(*V1_CACHE_RELATIVE)
    cached = {}
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        for job, summary in zip(payload["jobs"], payload["summaries"]):
            if job.cell in VALIDATION_CELLS and job.seed in VALIDATION_SEEDS:
                cached[(job.cell, job.seed)] = summary
    results = analyse_validation(runs, cached)
    results["declarations_sha256"] = declarations_sha
    results["source"] = source_state()
    write_bytes(results_path, json_bytes(results))
    return results


def _share(counts: dict, name: str):
    total = sum(v for k, v in counts.items())
    return None if total == 0 else counts.get(name, 0) / total


def _add(total: dict, counts: dict) -> dict:
    for key, value in counts.items():
        total[key] = total.get(key, 0) + int(value)
    return total


def analyse_validation(runs: list[dict], cached: dict) -> dict:
    per_run = []
    cells: dict[str, dict] = {}
    for item in runs:
        s = item["summary"]
        d = s["v2_diagnostics"]
        onsets = s["onsets_v2"]
        v1_onsets = np.asarray(item["v1_reference"]["onsets"])
        repro = None
        key = (item["cell"], item["seed"])
        if key in cached:
            ref = {k: v for k, v in cached[key].items() if k != "job"}
            ref_meta = {k: v for k, v in ref["meta"].items() if k != "wall_seconds"}
            now = {k: item["v1_reference"][k] for k in ref}
            now_meta = {k: v for k, v in now["meta"].items() if k != "wall_seconds"}
            ref["meta"], now["meta"] = ref_meta, now_meta
            repro = _bytes_equal(ref, now)
        marks_v2 = s["marks_v2"]
        roles = marks_v2["event_role"]
        bounce_gaps = marks_v2["gap_previous"][roles == ev.ROLE_BOUNCE]
        later = roles != ev.ROLE_PARENT
        row = {
            "cell": item["cell"], "seed": item["seed"], "wall_seconds": item["wall_seconds"],
            "C1_v1_fields_bit_for_bit": {"pass": not item["v1_diff_same_run"], "differing_paths": item["v1_diff_same_run"]},
            "C2_weather_bit_equal": d["weather_bit_equal_to_run"],
            "C3_W_rel": d["wrel_offline_vs_log"],
            "C4_marks_matched": {"unmatched": d["marks_unmatched_to_crossings"], "onset_max_residual_s": d["onset_time_max_residual_s"],
                                 "pass": d["marks_unmatched_to_crossings"] == 0 and (d["onset_time_max_residual_s"] or 0.0) <= 1.0e-9},
            "C5_onsets": {"v2_per_cable": onsets["per_cable"], "v1_per_cable": v1_onsets,
                          "pass": bool(np.array_equal(onsets["per_cable"], v1_onsets))},
            "C6_hand_checks": item["hand_checks"],
            "C7_reproducibility_vs_v1_cache": None if repro is None else {"identical": not repro, "differing_paths": repro},
            "closure": s["closure"],
            "marks": d["window_marks"],
            "events_in_window": int(s["events"]["event_id"].size),
            "events_chained_sensitivity": d["events_in_window_chained_rule_sensitivity"],
            "orphan_marks": d["window_marks_whose_event_parent_is_outside_window"],
            "roles": s["regime_summary"]["roles"],
            "bounce_gap_median_s": float(np.median(bounce_gaps)) if bounce_gaps.size else None,
            "later_members_that_are_bounces": (float(np.mean(roles[later] == ev.ROLE_BOUNCE)) if later.any() else None),
            "onsets": onsets["count"], "primary_onsets": onsets["primary_count"],
            "primary_onsets_2s_same_cable": onsets["primary_2s_same_cable_count"],
            "exposure_s": onsets["exposure"], "primary_exposure_s": onsets["primary_exposure"],
            "clean_exposure_s": s["clean"]["exposure"], "clean_excluded_fraction": s["clean"]["excluded_fraction"],
            "regime": s["regime_summary"],
            "shape_deg": {"chord_world": np.degrees(s["shape"]["chord_world_std"]), "chord_load": np.degrees(s["shape"]["chord_load_std"]),
                          "psi": np.degrees(s["shape"]["psi_std"]), "load_yaw": s["shape"]["load_yaw_std_deg"],
                          "v1_chord_world_1s": np.degrees(item["v1_reference"]["geometry_stats"]["cable_angle_std"])},
        }
        # clean-set moments against v1's 1 s same-cable snap window
        v1q = item["v1_reference"]["moments"]["q"]
        v2q = s["clean"]["moments"]["q"]
        def mean_std(m):
            count, total, square = m.sum(axis=1) if m.ndim == 2 else m
            mean = total / count
            return mean, math.sqrt(max(square / count - mean * mean, 0.0))
        row["q_mean_std_v1_clean"] = mean_std(v1q)
        row["q_mean_std_B3_clean"] = mean_std(v2q)
        per_run.append(row)
        cell = cells.setdefault(item["cell"], {"marks": {}, "marks_clip": {}, "short": {}, "short_clip_dummy": {}, "short6": {},
                                                "all6": {}, "events": {}, "roles": {}, "onsets": 0, "primary": 0,
                                                "primary_2s": 0, "exposure": 0.0, "primary_exposure": 0.0, "n_marks": 0,
                                                "n_events": 0, "n_events_chained": 0, "bounce_gaps": [], "closures": 0,
                                                "shape": [], "later": 0, "later_bounces": 0})
        rs = s["regime_summary"]
        _add(cell["marks"], rs["marks"])
        _add(cell["marks_clip"], rs["marks_clip_reading"])
        _add(cell["short"], rs["short_marks"])
        _add(cell["short6"], rs["short_marks_above_6kN"])
        _add(cell["all6"], rs["marks_above_6kN"])
        _add(cell["events"], rs["events"])
        _add(cell["roles"], rs["roles"])
        clip_short = rg.class_counts(marks_v2["regime_clip"], np.asarray(s["marks"]["dwell"]) < SHORT_DWELL)
        _add(cell["short_clip_dummy"], clip_short)
        cell["onsets"] += onsets["count"]
        cell["primary"] += onsets["primary_count"]
        cell["primary_2s"] += onsets["primary_2s_same_cable_count"]
        cell["exposure"] += onsets["exposure"]
        cell["primary_exposure"] += onsets["primary_exposure"]
        cell["n_marks"] += d["window_marks"]
        cell["n_events"] += int(s["events"]["event_id"].size)
        cell["n_events_chained"] += d["events_in_window_chained_rule_sensitivity"]
        cell["bounce_gaps"] += bounce_gaps.tolist()
        cell["closures"] += int(bool(s["closure"]["terminal"]))
        cell["later"] += int(later.sum())
        cell["later_bounces"] += int(np.sum(roles[later] == ev.ROLE_BOUNCE))
    table = {}
    for name, c in cells.items():
        table[name] = {
            "marks": c["n_marks"], "events": c["n_events"], "events_chained_sensitivity": c["n_events_chained"],
            "roles": c["roles"],
            "marks_absorbed_fraction": None if not c["n_marks"] else 1.0 - c["roles"].get("parent", 0) / c["n_marks"],
            "later_members_that_are_bounces": None if not c["later"] else c["later_bounces"] / c["later"],
            "bounce_gap_median_s": float(np.median(c["bounce_gaps"])) if c["bounce_gaps"] else None,
            "onsets": c["onsets"], "primary_onsets": c["primary"], "primary_fraction": None if not c["onsets"] else c["primary"] / c["onsets"],
            "primary_onsets_2s_same_cable": c["primary_2s"],
            "exposure_s": c["exposure"], "primary_exposure_s": c["primary_exposure"],
            "primary_rate_per_cable_s": None if not c["primary_exposure"] else c["primary"] / (5.0 * c["primary_exposure"]),
            "closures": c["closures"],
            "class_counts_marks": c["marks"], "class_counts_marks_clip_reading": c["marks_clip"],
            "class_counts_short_marks": c["short"], "class_counts_short_marks_clip_reading": c["short_clip_dummy"],
            "class_counts_short_marks_above_6kN": c["short6"], "class_counts_marks_above_6kN": c["all6"],
            "class_counts_events": c["events"],
            "shares": {
                "N_of_short_marks": _share(c["short"], "N"),
                "N_of_all_marks": _share(c["marks"], "N"),
                "R2_of_marks": _share(c["marks"], "R2"),
                "R2_of_marks_clip_reading": _share(c["marks_clip"], "R2"),
                "R1_of_marks": _share(c["marks"], "R1"),
                "transitional_of_marks": _share(c["marks"], "transitional"),
                "N_of_short_mark_snaps_above_6kN": _share(c["short6"], "N"),
            },
            "plan_evidence": {
                "N_of_short_marks_in_65_84pct": None if _share(c["short"], "N") is None else bool(0.65 <= _share(c["short"], "N") <= 0.84),
                "R2_of_marks_in_0_4pct": None if _share(c["marks"], "R2") is None else bool(_share(c["marks"], "R2") <= 0.04),
                "R2_of_marks_clip_reading_in_0_4pct": None if _share(c["marks_clip"], "R2") is None else bool(_share(c["marks_clip"], "R2") <= 0.04),
            },
        }
    checks = {
        "C1": all(r["C1_v1_fields_bit_for_bit"]["pass"] for r in per_run),
        "C2": all(r["C2_weather_bit_equal"] for r in per_run),
        "C3": all(r["C3_W_rel"]["passes_1e-6_relative"] for r in per_run),
        "C4": all(r["C4_marks_matched"]["pass"] for r in per_run),
        "C5": all(r["C5_onsets"]["pass"] for r in per_run),
        "C6": all(h["passes"] for r in per_run for h in r["C6_hand_checks"].values()),
        "C7_reported": {f"{r['cell']}/{r['seed']}": (None if r["C7_reproducibility_vs_v1_cache"] is None
                                                      else r["C7_reproducibility_vs_v1_cache"]["identical"]) for r in per_run},
    }
    checks["machinery_accepted"] = all(checks[k] for k in ("C1", "C2", "C3", "C4", "C5", "C6"))
    return {"schema": "v2-machinery-validation-1", "checks": checks, "cells": table, "runs": per_run}


C6B_CELL = "T1000_k500_I1.0"
C6B_SEED = 2003
C6B_DURATION = 70.0
C6B_MARKS = {"N": 5, "R1": 0, "R2": 1, "transitional": 12}
C6B_READ_OFFSET = 2.5e-4


def supplementary_c6b() -> dict:
    """Addendum 2026-09-13: class the C6 marks of one run from the true 1 ms plant state."""
    import pickle
    from dataclasses import replace

    from tether.campaign.common import json_bytes, sha256_file, source_state, write_bytes
    from tether.campaign.fleet_run import build_run
    from tether.campaign.phase2 import cell_specs
    from tether.physics.fleet import cable_kinematics, plant_state

    declarations_path, results_path = _paths()
    addendum = results_path.parent / "machinery_addendum_2026-09-13.json"
    if not addendum.exists():
        raise SystemExit("the addendum must be written before the supplementary check")
    with (results_path.parent / "cache" / "machinery_validation_runs.pkl").open("rb") as handle:
        runs = pickle.load(handle)
    reference = next(r for r in runs if r["cell"] == C6B_CELL and r["seed"] == C6B_SEED)
    full_spec = cell_specs()[C6B_CELL]
    spec = replace(full_spec, duration=C6B_DURATION)
    run = build_run(spec, C6B_SEED)
    weather = run.fleet.cables._weather
    full_weather = rg.regenerate_weather(full_spec, C6B_SEED)
    prefix_equal = bool(np.array_equal(weather, full_weather[: weather.shape[0]]))
    geometry = run.fleet.geometry
    rest = run.fleet.cables.cable.rest_length
    total = spec.warmup + spec.duration
    ticks = int(round(total / rg.EVENT_PERIOD))
    simulator = run.simulator
    simulator.Initialize()
    states = np.empty((ticks + 1, 6 * (geometry.vessel_count + 1)))
    times = np.empty(ticks + 1)
    # The cable bank's update at tick t reads the plant state *after* the plant's own
    # discrete update at t (state-log row 0 equals the context at t = 0.25 ms bit for bit,
    # cf. the note in fleet_run.build_run), so the state is read inside (t, t + 0.5 ms).
    for k in range(ticks + 1):
        t = k * rg.EVENT_PERIOD
        simulator.AdvanceTo(t + C6B_READ_OFFSET)
        times[k] = t
        states[k] = plant_state(run.fleet, simulator.get_context())
    run.fleet.cables.flush()
    log = run.fleet.cables.log
    n = log.count
    # (i) the trajectory is the validation run's
    ref_marks = reference["summary"]["marks"]
    cutoff = total - 1.0  # marks near the shortened record's end may still be pending
    marks_now = sorted((float(m.t_up), int(m.cable), float(m.T_peak)) for m in run.fleet.cables.reengagements
                       if m.t_up - m.dwell >= spec.warmup and m.t_up <= cutoff)
    ref_list = sorted((float(t), int(c), float(p)) for t, c, p in zip(ref_marks["t_up"], ref_marks["cable"], ref_marks["T_peak"])
                      if t <= cutoff)
    marks_equal = bool(marks_now == ref_list and len(ref_list) > 0)
    # W^c and W_rel from the true 1 ms state, scalar loop through cable_kinematics
    n_cables = geometry.vessel_count
    wc_true = np.empty((ticks + 1, n_cables))
    wrel_true = np.empty((ticks + 1, n_cables))
    for k in range(ticks + 1):
        kin = cable_kinematics(states[k], geometry, rest)
        index = min(int(math.floor((times[k] + 1.0e-9) / rg.WEATHER_PERIOD)), weather.shape[0] - 1)
        w = weather[index]
        for i in range(n_cables):
            d = kin["direction"][i]
            wc_true[k, i] = rg.C_EFF * ((w[0, 0] / rg.C_L - w[i + 1, 0] / rg.C_A) * d[0] + (w[0, 1] / rg.C_L - w[i + 1, 1] / rg.C_A) * d[1])
            wrel_true[k, i] = rg.M_EFF * ((w[0, 0] / rg.M_L - w[i + 1, 0] / rg.M_A) * d[0] + (w[0, 1] / rg.M_L - w[i + 1, 1] / rg.M_A) * d[1])
    log_time = log.event_time[:n]
    rows = np.rint(log_time / rg.EVENT_PERIOD).astype(np.int64)
    keep_rows = rows <= ticks
    log_time, rows, n = log_time[keep_rows], rows[keep_rows], int(np.sum(keep_rows))
    time_aligned = bool(np.allclose(times[rows], log_time, rtol=0.0, atol=1.0e-12))
    diff = np.abs(wrel_true[rows] - log.relative_load[:n])
    scale = np.maximum(np.abs(log.relative_load[:n]), 1.0)
    wrel_rel = float(np.max(diff / scale))
    # the machinery's interpolated series on the same run
    interpolant = rg.StateInterpolant.from_log(log.state_time[: log.state_count], log.state[: log.state_count],
                                               geometry.load_offsets, geometry.vessel_offsets)
    _, wc_machine = rg.weather_side_series(log_time, weather, interpolant)
    pretension = spec.pretension
    extra = reference["summary"]["marks_v2"]
    cases = {}
    for name, position in C6B_MARKS.items():
        cable = int(ref_marks["cable"][position])
        lo, hi = int(extra["i_down"][position]), int(extra["i_up"][position])
        series = wc_true[rows, cable]
        exceed = [k for k in range(lo, hi) if series[k] > pretension]
        if exceed:
            above = series > pretension
            start = stop = exceed[0]
            while True:  # extend left over exceedances and gaps shorter than 1000 samples
                j = start - 1
                while j >= 0 and not above[j]:
                    j -= 1
                if j >= 0 and start - j - 1 < 1000:
                    start = j
                    while start - 1 >= 0 and above[start - 1]:
                        start -= 1
                else:
                    break
            while True:
                j = stop + 1
                while j < series.size and not above[j]:
                    j += 1
                if j < series.size and j - stop - 1 < 1000:
                    stop = j
                    while stop + 1 < series.size and above[stop + 1]:
                        stop += 1
                else:
                    break
            t_x = (stop - start + 1) * rg.EVENT_PERIOD
            peak = float(np.max(series[start : stop + 1]))
            label = rg.classify(t_x, peak, pretension, True)
            episode = {"start": float(log_time[start]), "end": float(log_time[stop]), "t_x": t_x, "Wc_max": peak,
                       "v_s": (peak - pretension) * t_x / rg.M_A, "in_slack_exceedance_samples": len(exceed),
                       "touches_record_edge": bool(start == 0 or stop == series.size - 1)}
        else:
            label, episode = "N", None
        cases[name] = {
            "mark": position, "cable": cable, "t_down": float(extra["t_down"][position]), "t_up": float(ref_marks["t_up"][position]),
            "class_true_1ms": label, "class_machinery": str(extra["regime"][position]), "agree": label == str(extra["regime"][position]),
            "episode_true_1ms": episode,
            "machinery": {"t_x": float(extra["t_x"][position]), "Wc_max": float(extra["Wc_max"][position]),
                          "Wc_slack_max": float(extra["Wc_slack_max"][position])},
            "Wc_true_slack_max": float(np.max(series[lo:hi])),
            "max_abs_Wc_true_minus_machinery_in_slack_N": float(np.max(np.abs(series[lo:hi] - wc_machine[lo:hi, cable]))),
            "first_slack_samples_true": series[lo : min(hi, lo + 12)].tolist(),
            "first_slack_samples_machinery": wc_machine[lo : min(hi, lo + 12), cable].tolist(),
            "row_before_t_down_Wc": float(series[(lo // 10) * 10]),
        }
    passes = bool(prefix_equal and marks_equal and time_aligned and wrel_rel <= 1.0e-6 and all(c["agree"] for c in cases.values()))
    results = {
        "schema": "v2-machinery-addendum-results-1",
        "addendum_sha256": sha256_file(addendum),
        "declarations_sha256": sha256_file(declarations_path),
        "weather_prefix_equal_to_full_run": prefix_equal,
        "marks_equal_to_validation_run": marks_equal,
        "one_ms_times_aligned_with_log": time_aligned,
        "W_rel_true_1ms_vs_log_max_rel": wrel_rel,
        "cases": cases,
        "C6b_pass": passes,
        "state_read_instant": ("the state at tick t is read at t + 0.25 ms, i.e. after the plant's discrete update at t, which is "
                               "the state the cable bank reads at t. A first execution (kept as "
                               "machinery_addendum_results_run1_wrong_instant.json) read the context at t itself - the state "
                               "before that update - and failed clause (i) at 0.14 relative on W_rel while agreeing on all four "
                               "classes; only the read instant was changed"),
        "source": source_state(),
    }
    write_bytes(results_path.parent / "machinery_addendum_results.json", json_bytes(results))
    return results


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("declare", "validate", "c6b"))
    parser.add_argument("--workers", type=int, default=2)
    arguments = parser.parse_args()
    # Call through the package module so the pool pickles package-qualified functions.
    from tether.campaign.v2 import summaries as module

    if arguments.command == "declare":
        print(module.declare())
        return
    if arguments.command == "c6b":
        out = module.supplementary_c6b()
        print(json.dumps({k: v for k, v in out.items() if k != "cases"} | {"cases": {k: [c["class_true_1ms"], c["class_machinery"],
              c["max_abs_Wc_true_minus_machinery_in_slack_N"]] for k, c in out["cases"].items()}}, indent=1, default=str))
        return
    results = module.validate(arguments.workers)
    print(json.dumps({"checks": results["checks"], "cells": {k: v["shares"] for k, v in results["cells"].items()}}, indent=1, default=str))


if __name__ == "__main__":
    main()
