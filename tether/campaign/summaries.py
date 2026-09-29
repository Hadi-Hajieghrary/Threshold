"""Reduce one stationary production run to the records every later phase needs.

Definitions (declared here once, used by Phases 1(d)-4):

* Recorded window: ``[warmup, end]`` where ``end`` is the run's end or its closure time.
* Snap window: ``[t_up, t_up + SNAP_WINDOW]`` after each re-engagement.  Tension
  exceedances starting inside it belong to the snap channel (the engagement and its
  ringing), all others to the taut channel.
* Declustering: runs declustering with run length one engagement period
  ``2 pi sqrt(m_eff / k)`` (Appendix C), on 1 ms samples.
* Gust episodes: runs of ``W_rel > T0`` on each cable, merged across gaps shorter than
  ``GUST_MERGE`` (the weather is an Ornstein-Uhlenbeck-like process whose sample paths
  chatter around any level).
"""

from __future__ import annotations

import math

import numpy as np

from tether.physics import constants
from tether.physics.fleet import EVENT_PERIOD, PAIR_REDUCED_MASS, cable_kinematics

ENGAGEMENT_PERIOD = 2.0 * math.pi * math.sqrt(PAIR_REDUCED_MASS / constants.CABLE_STIFFNESS)
SNAP_WINDOW = 1.0
GUST_MERGE = 1.0
DECIMATION = 1.0
BLOCK_LENGTH = 5.0
LEVEL_GRID = np.geomspace(500.0, 80_000.0, 121)
TAUT_PEAK_FLOOR_RATIO = 1.2
MARK_FIELDS = (
    "t_up",
    "u_entry",
    "depth",
    "dwell",
    "v_return",
    "T_peak",
    "n_maxima",
    "cable",
    "W_rel_at_onset",
    "W_rel_max",
)


def circular_std(angles: np.ndarray) -> np.ndarray:
    """Circular standard deviation sqrt(-2 ln R) along axis 0."""
    resultant = np.abs(np.mean(np.exp(1j * angles), axis=0))
    return np.sqrt(-2.0 * np.log(np.clip(resultant, 1.0e-12, 1.0)))


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and stop (exclusive) indices of runs of True."""
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return changes[0::2], changes[1::2]


def merge_runs(starts: np.ndarray, stops: np.ndarray, gap: int) -> tuple[np.ndarray, np.ndarray]:
    if starts.size == 0:
        return starts, stops
    keep = np.concatenate([[True], starts[1:] - stops[:-1] > gap])
    group = np.cumsum(keep) - 1
    merged_starts = starts[keep]
    merged_stops = np.zeros_like(merged_starts)
    np.maximum.at(merged_stops, group, stops)
    return merged_starts, merged_stops


def declustered_exceedances(values: np.ndarray, valid: np.ndarray, level: float, run_samples: int) -> np.ndarray:
    """Start indices of exceedance clusters of ``level`` (runs declustering)."""
    indices = np.flatnonzero(valid & (values >= level))
    if indices.size == 0:
        return indices
    starts = np.concatenate([[True], np.diff(indices) > run_samples])
    return indices[starts]


def declustered_upcrossing_counts(
    values: np.ndarray, valid: np.ndarray, levels: np.ndarray, run_samples: int
) -> np.ndarray:
    return np.array(
        [declustered_exceedances(values, valid, float(level), run_samples).size for level in levels],
        dtype=np.int64,
    )


def cluster_maxima(
    times: np.ndarray, values: np.ndarray, valid: np.ndarray, level: float, run_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(valid & (values >= level))
    if indices.size == 0:
        return np.empty(0), np.empty(0)
    new_cluster = np.concatenate([[True], np.diff(indices) > run_samples])
    group = np.cumsum(new_cluster) - 1
    peaks = np.full(group[-1] + 1, -np.inf)
    np.maximum.at(peaks, group, values[indices])
    peak_index = np.zeros(group[-1] + 1, dtype=np.int64)
    is_peak = values[indices] == peaks[group]
    first = np.flatnonzero(is_peak)
    seen = np.zeros(group[-1] + 1, dtype=bool)
    for position in first:
        cluster = group[position]
        if not seen[cluster]:
            seen[cluster] = True
            peak_index[cluster] = indices[position]
    return times[peak_index], peaks


def summarize_run(run, warmup: float, pretension: float) -> dict[str, object]:
    """Reduce a finished stationary run to marks, moments, counts, and samples."""
    cables = run.fleet.cables
    log = cables.log
    n_samples = log.count
    n_cables = run.fleet.geometry.vessel_count
    stiffness = cables.cable.stiffness
    damping = cables.cable.damping
    time = log.event_time[:n_samples]
    elongation = log.elongation[:n_samples]
    rate = log.rate[:n_samples]
    relative_load = log.relative_load[:n_samples]
    end = float(run.simulator.get_context().get_time())
    if cables.closure is not None:
        end = min(end, float(cables.closure[1]))
    window = (time >= warmup - 1.0e-9) & (time <= end + 1.0e-9)
    exposure = max(0.0, end - warmup)
    run_samples = int(round(ENGAGEMENT_PERIOD / EVENT_PERIOD))

    raw_marks = [
        mark
        for mark in cables.reengagements
        if mark.t_up - mark.dwell >= warmup and mark.t_up <= end
    ]
    marks = {name: np.array([getattr(mark, name) for mark in raw_marks], dtype=float) for name in MARK_FIELDS}
    marks["cable"] = marks["cable"].astype(np.int64)
    marks["n_maxima"] = marks["n_maxima"].astype(np.int64)
    extra = {
        name: np.full(len(raw_marks), np.nan)
        for name in ("t_down", "t_turn", "wrel_return_mean", "wrel_slack_mean", "exceed_duration", "exceed_mean", "a_bar_return")
    }
    for position, mark in enumerate(raw_marks):
        cable = mark.cable
        t_down = mark.t_up - mark.dwell
        lo = int(np.searchsorted(time, t_down))
        hi = int(np.searchsorted(time, mark.t_up, side="right"))
        extra["t_down"][position] = t_down
        if hi > lo:
            segment = elongation[lo:hi, cable]
            turn = lo + int(np.argmin(segment))
            extra["t_turn"][position] = time[turn]
            load = relative_load[lo:hi, cable]
            extra["wrel_slack_mean"][position] = float(np.mean(load))
            extra["wrel_return_mean"][position] = float(np.mean(relative_load[turn:hi, cable]))
            exceed = load > pretension
            extra["exceed_duration"][position] = float(np.sum(exceed)) * EVENT_PERIOD
            extra["exceed_mean"][position] = float(np.mean(load[exceed] - pretension)) if exceed.any() else 0.0
        if mark.depth > 0.0:
            extra["a_bar_return"][position] = (mark.v_return**2 - mark.u_entry**2) / (2.0 * mark.depth)
    marks.update(extra)

    snap_mask = np.zeros_like(elongation, dtype=bool)
    for mark in cables.reengagements:
        lo = int(np.searchsorted(time, mark.t_up))
        hi = int(np.searchsorted(time, mark.t_up + SNAP_WINDOW, side="right"))
        snap_mask[lo:hi, mark.cable] = True

    q = stiffness * elongation + damping * rate
    taut = elongation > 0.0
    clean = taut & ~snap_mask & window[:, None]
    acceleration = np.zeros_like(rate)
    acceleration[1:-1] = (rate[2:] - rate[:-2]) / (2.0 * EVENT_PERIOD)
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
        ("wrel", relative_load, np.broadcast_to(window[:, None], relative_load.shape)),
    ):
        count = mask.sum(axis=0).astype(np.int64)
        total = np.where(mask, values, 0.0).sum(axis=0)
        square = np.where(mask, values * values, 0.0).sum(axis=0)
        moments[name] = np.stack([count.astype(float), total, square])

    level_counts = np.zeros((LEVEL_GRID.size, n_cables), dtype=np.int64)
    peak_times, peak_values, peak_cables = [], [], []
    floor = TAUT_PEAK_FLOOR_RATIO * pretension
    for cable in range(n_cables):
        level_counts[:, cable] = declustered_upcrossing_counts(q[:, cable], clean[:, cable], LEVEL_GRID, run_samples)
        times, peaks = cluster_maxima(time, q[:, cable], clean[:, cable], floor, run_samples)
        peak_times.append(times)
        peak_values.append(peaks)
        peak_cables.append(np.full(peaks.size, cable, dtype=np.int64))

    gust = {"start": [], "duration": [], "max_excess": [], "mean_excess": [], "cable": []}
    merge = int(round(GUST_MERGE / EVENT_PERIOD))
    for cable in range(n_cables):
        above = window & (relative_load[:, cable] > pretension)
        starts, stops = merge_runs(*_runs(above), merge)
        for start, stop in zip(starts, stops):
            excess = relative_load[start:stop, cable] - pretension
            gust["start"].append(time[start])
            gust["duration"].append((stop - start) * EVENT_PERIOD)
            gust["max_excess"].append(float(np.max(excess)))
            gust["mean_excess"].append(float(np.mean(np.maximum(excess, 0.0))))
            gust["cable"].append(cable)
    gust = {key: np.asarray(value, dtype=np.int64 if key == "cable" else float) for key, value in gust.items()}

    onsets = np.zeros(n_cables, dtype=np.int64)
    slack_samples = np.zeros(n_cables, dtype=np.int64)
    for cable in range(n_cables):
        series = taut[window, cable]
        onsets[cable] = int(np.sum(series[:-1] & ~series[1:]))
        slack_samples[cable] = int(np.sum(~series))
    window_count = int(window.sum())

    step = int(round(DECIMATION / EVENT_PERIOD))
    first = int(np.searchsorted(time, warmup - 1.0e-9))
    decimated = slice(first, n_samples, step)
    applied = np.where(taut, np.maximum(q, 0.0), 0.0)
    block = int(round(BLOCK_LENGTH / EVENT_PERIOD))
    usable = (int(window.sum()) // block) * block
    block_maxima = applied[first : first + usable].reshape(-1, block, n_cables).max(axis=1) if usable else np.empty((0, n_cables))

    state_time = log.state_time[: log.state_count]
    states = log.state[: log.state_count]
    state_window = (state_time >= warmup) & (state_time <= end)
    angle_std = np.full(n_cables, np.nan)
    load_yaw_std = float("nan")
    heading_std = np.full(n_cables, np.nan)
    if state_window.any():
        picked = states[state_window][:: int(round(DECIMATION / 0.01))]
        angles = []
        for row in picked:
            kin = cable_kinematics(row, run.fleet.geometry, cables.cable.rest_length)
            angles.append(np.arctan2(kin["direction"][:, 1], kin["direction"][:, 0]))
        angles = np.asarray(angles)
        angle_std = circular_std(angles)
        load_yaw_std = float(circular_std(picked[:, 2:3])[0])
        heading_std = circular_std(picked[:, 5 : 3 * (n_cables + 1) : 3])

    tracker_dropped = int(sum(t.state.reengagement_ring.dropped + t.state.taut_ring.dropped for t in cables.trackers))
    return {
        "meta": {
            "seed": run.master_seed,
            "exposure": exposure,
            "end_time": end,
            "closure": None if cables.closure is None else [int(cables.closure[0]), float(cables.closure[1])],
            "wall_seconds": run.wall_seconds,
            "sim_seconds": float(run.simulator.get_context().get_time()),
            "tracker_dropped": tracker_dropped,
            "window_samples": window_count,
            "lint_violations": list(run.lint_violations),
        },
        "marks": marks,
        "moments": moments,
        "level_counts": level_counts,
        "taut_peaks": {
            "time": np.concatenate(peak_times) if peak_times else np.empty(0),
            "peak": np.concatenate(peak_values) if peak_values else np.empty(0),
            "cable": np.concatenate(peak_cables) if peak_cables else np.empty(0, dtype=np.int64),
        },
        "gusts": gust,
        "onsets": onsets,
        "slack_samples": slack_samples,
        "samples": {
            "e": elongation[decimated],
            "edot": rate[decimated],
            "q": q[decimated],
            "taut": taut[decimated],
            "wrel": relative_load[decimated],
        },
        "block_maxima": block_maxima,
        "geometry_stats": {
            "cable_angle_std": angle_std,
            "load_yaw_std": load_yaw_std,
            "vessel_heading_std": heading_std,
        },
    }
