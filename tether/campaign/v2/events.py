"""Declustered events, primary onsets and the fleet-wide clean set (plan v2 IV.9, Appendix B.1, B.3).

Drake-free: every function works on plain arrays, so the reduced-model Monte Carlo that
computes theta applies exactly the rule the plant records are scored with (B.1: "the
identical rule is applied to the reduced model when theta is computed").

Declared rules (the plan's text, and the reading taken where it leaves a choice):

* **Event (B.1).** "An event is the first re-engagement of a burst; marks within 2 s on
  the same cable belong to it."  Read literally as **anchored**: a mark of cable i
  belongs to the open event of cable i when its ``t_up`` lies within ``BURST_WINDOW`` of
  that event's *first* ``t_up`` (inclusive, ``t_up - t_event <= 2 s``); the first mark
  beyond it opens a new event.  "Within 2 s" is measured from the event ("belong to
  it"), not from the previous member, so a burst cannot grow without bound.  The
  chained reading (within 2 s of the previous member) is implemented as
  ``rule="chained"`` for sensitivity reports only; it gates nothing.  The anchored rule
  is the one ``campaign/v2/p6_t0.py`` already applies.
* **Bounces are absorbed.**  Every mark belongs to exactly one event.  Its role is
  ``parent`` (the event's first re-engagement), ``bounce`` (a later member whose gap to
  the preceding mark of the same cable is at most one engagement period,
  ``ENGAGEMENT_PERIOD`` = 2 pi sqrt(m_eff/k) = 0.335 s: B.1's restitution bounce) or
  ``member`` (a later member beyond one engagement period).  Nothing is dropped: an
  event carries its member count, bounce count and cluster maximum.
* **Primary onset (B.1).** An onset at time t is primary iff no re-engagement on any
  cable lies in (t - 3 s, t].  Equivalently t lies outside [t_up, t_up + 3 s) for every
  re-engagement; the primary exposure is the record length less the union of those
  windows (the measure is the same for the half-open and closed windows).
* **Clean set (B.3).** Taut samples whose time lies outside (t_up, t_up + 3 s] for every
  re-engagement on any cable of the run; exposure is the recorded window less the union
  of those windows.  No all-cables-taut clause, no 1 s variant.
* **Re-engagement set.**  "Every re-engagement on any cable" is taken as every geometric
  up-crossing of e = 0 of a live cable in the 1 ms log, interpolated exactly as the
  plant's ``CableEventTracker`` interpolates ``t_up``.  This is a superset of the
  recorded marks (it includes up-crossings whose mark is censored at the record's
  boundaries); ``match_marks`` checks that every mark's ``t_up`` is one of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tether.physics import constants

BURST_WINDOW = 2.0
EXCLUSION_WINDOW = 3.0
PAIR_REDUCED_MASS = constants.VESSEL_MASS * constants.LOAD_MASS / (constants.VESSEL_MASS + constants.LOAD_MASS)
ENGAGEMENT_PERIOD = 2.0 * math.pi * math.sqrt(PAIR_REDUCED_MASS / constants.CABLE_STIFFNESS)
ROLE_PARENT = "parent"
ROLE_BOUNCE = "bounce"
ROLE_MEMBER = "member"
ROLE_CODES = {ROLE_PARENT: 0, ROLE_BOUNCE: 1, ROLE_MEMBER: 2}


# ----------------------------------------------------------------------------- crossings


@dataclass(frozen=True)
class Crossings:
    """Geometric crossings of e = 0 on the 1 ms log, in time order."""

    time: np.ndarray  # interpolated crossing time
    cable: np.ndarray
    index: np.ndarray  # index of the first sample after the crossing

    @property
    def size(self) -> int:
        return int(self.time.size)


def _interpolated(time: np.ndarray, values: np.ndarray, before: np.ndarray, cable: int) -> np.ndarray:
    # Same arithmetic as tether.physics.events.linear_crossing_time.
    first = values[before, cable]
    second = values[before + 1, cable]
    fraction = -first / (second - first)
    return time[before] + fraction * (time[before + 1] - time[before])


def crossings(time: np.ndarray, elongation: np.ndarray, alive: np.ndarray | None = None) -> tuple[Crossings, Crossings]:
    """(up, down) crossings of every cable, sorted by time then cable.

    Up: previous sample e <= 0 < current (the tracker's ``crossed_up``); down: previous
    e > 0 >= current.  With ``alive`` given, a transition counts only when the cable is
    alive at both samples (the tracker stops sampling a severed cable).
    """
    time = np.asarray(time, dtype=float)
    elongation = np.asarray(elongation, dtype=float)
    if elongation.ndim == 1:
        elongation = elongation[:, None]
    ups_t, ups_c, ups_i, downs_t, downs_c, downs_i = [], [], [], [], [], []
    for cable in range(elongation.shape[1]):
        series = elongation[:, cable]
        previous = series[:-1]
        current = series[1:]
        live = np.ones(previous.size, dtype=bool)
        if alive is not None:
            live = alive[:-1, cable] & alive[1:, cable]
        up = np.flatnonzero((previous <= 0.0) & (current > 0.0) & live)
        down = np.flatnonzero((previous > 0.0) & (current <= 0.0) & live)
        ups_t.append(_interpolated(time, elongation, up, cable))
        ups_c.append(np.full(up.size, cable, dtype=np.int64))
        ups_i.append(up + 1)
        downs_t.append(_interpolated(time, elongation, down, cable))
        downs_c.append(np.full(down.size, cable, dtype=np.int64))
        downs_i.append(down + 1)

    def pack(t, c, i) -> Crossings:
        t = np.concatenate(t) if t else np.empty(0)
        c = np.concatenate(c) if c else np.empty(0, dtype=np.int64)
        i = np.concatenate(i) if i else np.empty(0, dtype=np.int64)
        order = np.lexsort((c, t))
        return Crossings(t[order], c[order], i[order])

    return pack(ups_t, ups_c, ups_i), pack(downs_t, downs_c, downs_i)


def match_marks(t_up: np.ndarray, cable: np.ndarray, ups: Crossings, tolerance: float = 1.0e-9) -> np.ndarray:
    """Index into ``ups`` of each mark's up-crossing (-1 when none within ``tolerance``)."""
    out = np.full(np.asarray(t_up).size, -1, dtype=np.int64)
    for position, (time, which) in enumerate(zip(np.asarray(t_up, dtype=float), np.asarray(cable))):
        candidates = np.flatnonzero(ups.cable == which)
        if candidates.size == 0:
            continue
        nearest = candidates[int(np.argmin(np.abs(ups.time[candidates] - time)))]
        if abs(ups.time[nearest] - time) <= tolerance:
            out[position] = nearest
    return out


# ----------------------------------------------------------------------------- events (B.1)


@dataclass(frozen=True)
class Declustering:
    """B.1 membership of each mark (input order) and the event table."""

    event_id: np.ndarray  # (marks,) event of each mark
    role: np.ndarray  # (marks,) 0 parent, 1 bounce, 2 member (ROLE_CODES)
    parent_index: np.ndarray  # (marks,) input index of the event's parent mark
    gap_previous: np.ndarray  # (marks,) t_up gap to the preceding mark of the same cable (nan if none)
    event_parent: np.ndarray  # (events,) input index of the parent mark, events in (t_event, cable) order
    event_cable: np.ndarray
    event_time: np.ndarray
    event_marks: np.ndarray  # member count including the parent
    event_bounces: np.ndarray
    event_last: np.ndarray  # t_up of the last member

    @property
    def event_count(self) -> int:
        return int(self.event_parent.size)

    def role_names(self) -> list[str]:
        names = {code: name for name, code in ROLE_CODES.items()}
        return [names[int(code)] for code in self.role]


def decluster_marks(
    t_up: np.ndarray,
    cable: np.ndarray,
    window: float = BURST_WINDOW,
    bounce_period: float = ENGAGEMENT_PERIOD,
    rule: str = "anchored",
) -> Declustering:
    """Group re-engagement marks into B.1 events, absorbing bounces.

    ``rule="anchored"`` (declared): a mark joins the cable's open event while
    ``t_up - t_event <= window``.  ``rule="chained"`` (sensitivity only): while
    ``t_up - t_up_previous <= window``.
    """
    if rule not in ("anchored", "chained"):
        raise ValueError(f"unknown declustering rule: {rule}")
    t_up = np.asarray(t_up, dtype=float)
    cable = np.asarray(cable, dtype=np.int64)
    n = t_up.size
    event_id = np.full(n, -1, dtype=np.int64)
    role = np.full(n, -1, dtype=np.int64)
    parent_index = np.full(n, -1, dtype=np.int64)
    gap_previous = np.full(n, np.nan)
    provisional: list[tuple[float, int, int]] = []  # (t_event, cable, parent input index)
    members: list[list[int]] = []
    for which in np.unique(cable):
        rows = np.flatnonzero(cable == which)
        rows = rows[np.argsort(t_up[rows], kind="stable")]
        open_event = -1
        anchor = -math.inf
        previous = -math.inf
        for row in rows:
            if np.isfinite(previous):
                gap_previous[row] = t_up[row] - previous
            reference = anchor if rule == "anchored" else previous
            if open_event >= 0 and t_up[row] - reference <= window:
                members[open_event].append(int(row))
                role[row] = ROLE_CODES[ROLE_BOUNCE] if gap_previous[row] <= bounce_period else ROLE_CODES[ROLE_MEMBER]
            else:
                provisional.append((float(t_up[row]), int(which), int(row)))
                members.append([int(row)])
                open_event = len(members) - 1
                anchor = t_up[row]
                role[row] = ROLE_CODES[ROLE_PARENT]
            previous = t_up[row]
    order = sorted(range(len(provisional)), key=lambda k: (provisional[k][0], provisional[k][1]))
    event_parent = np.empty(len(order), dtype=np.int64)
    event_cable = np.empty(len(order), dtype=np.int64)
    event_time = np.empty(len(order))
    event_marks = np.empty(len(order), dtype=np.int64)
    event_bounces = np.empty(len(order), dtype=np.int64)
    event_last = np.empty(len(order))
    for new_id, old in enumerate(order):
        rows = members[old]
        event_parent[new_id] = provisional[old][2]
        event_cable[new_id] = provisional[old][1]
        event_time[new_id] = provisional[old][0]
        event_marks[new_id] = len(rows)
        event_bounces[new_id] = int(np.sum(role[rows] == ROLE_CODES[ROLE_BOUNCE]))
        event_last[new_id] = float(np.max(t_up[rows]))
        event_id[rows] = new_id
        parent_index[rows] = provisional[old][2]
    return Declustering(event_id, role, parent_index, gap_previous, event_parent, event_cable, event_time,
                        event_marks, event_bounces, event_last)


def event_cluster_maxima(values: np.ndarray, declustering: Declustering) -> tuple[np.ndarray, np.ndarray]:
    """(cluster maximum, input index of the member attaining it) of ``values`` per event."""
    values = np.asarray(values, dtype=float)
    maxima = np.full(declustering.event_count, -np.inf)
    np.maximum.at(maxima, declustering.event_id, values)
    argmax = np.full(declustering.event_count, -1, dtype=np.int64)
    for row in np.argsort(declustering.event_id, kind="stable"):
        event = declustering.event_id[row]
        if argmax[event] < 0 and values[row] == maxima[event]:
            argmax[event] = row
    return maxima, argmax


# ----------------------------------------------------------------------------- windows and exposure


def union_measure(starts: np.ndarray, length: float, low: float, high: float) -> float:
    """Measure of (union of [s, s + length)) intersected with [low, high]."""
    if high <= low:
        return 0.0
    starts = np.sort(np.asarray(starts, dtype=float))
    lo = np.clip(starts, low, high)
    hi = np.clip(starts + length, low, high)
    total = 0.0
    current_lo = current_hi = None
    for a, b in zip(lo, hi):
        if b <= a:
            continue
        if current_hi is None or a > current_hi:
            if current_hi is not None:
                total += current_hi - current_lo
            current_lo, current_hi = a, b
        else:
            current_hi = max(current_hi, b)
    if current_hi is not None:
        total += current_hi - current_lo
    return float(total)


def excluded_exposure(reengagement_times: np.ndarray, low: float, high: float, window: float = EXCLUSION_WINDOW) -> float:
    """[low, high] less the union of the (t_up, t_up + window] windows (B.1 and B.3)."""
    return max(0.0, (high - low) - union_measure(reengagement_times, window, low, high))


def in_exclusion(times: np.ndarray, reengagement_times: np.ndarray, window: float = EXCLUSION_WINDOW,
                 closed_left: bool = False) -> np.ndarray:
    """Whether each time lies in some (t_up, t_up + window] (``closed_left``: [t_up, t_up + window)).

    B.3 uses the default (the sample at t_up itself stays clean); B.1's primary rule is
    ``closed_left=True``: a re-engagement at or before t, within 3 s, removes onset t.
    """
    times = np.asarray(times, dtype=float)
    ups = np.sort(np.asarray(reengagement_times, dtype=float))
    if ups.size == 0:
        return np.zeros(times.shape, dtype=bool)
    if closed_left:
        # latest t_up <= t; excluded when t - t_up < window
        k = np.searchsorted(ups, times, side="right") - 1
        valid = k >= 0
        out = np.zeros(times.shape, dtype=bool)
        out[valid] = times[valid] - ups[k[valid]] < window
        return out
    # latest t_up < t; excluded when t - t_up <= window
    k = np.searchsorted(ups, times, side="left") - 1
    valid = k >= 0
    out = np.zeros(times.shape, dtype=bool)
    out[valid] = times[valid] - ups[k[valid]] <= window
    return out


@dataclass(frozen=True)
class PrimaryOnsets:
    primary: np.ndarray  # (onsets,) bool
    parent: np.ndarray  # (onsets,) index into the re-engagement arrays of the latest one in (t - window, t], -1 if none
    parent_lag: np.ndarray  # t - t_up(parent), nan if none
    parent_same_cable: np.ndarray  # bool (False if none)


def primary_onsets(
    onset_times: np.ndarray,
    onset_cables: np.ndarray,
    reengagement_times: np.ndarray,
    reengagement_cables: np.ndarray,
    window: float = EXCLUSION_WINDOW,
    same_cable_only: bool = False,
) -> PrimaryOnsets:
    """B.1's primary flag and IV.4's ``parent`` of each onset.

    The parent is the nearest (latest) re-engagement in (t - window, t] on any cable
    (``same_cable_only``: on the onset's own cable, the Phase 2 T2 fallback's 2 s
    same-cable re-scoring).  An onset is primary iff it has no parent.
    """
    onset_times = np.asarray(onset_times, dtype=float)
    onset_cables = np.asarray(onset_cables, dtype=np.int64)
    up_t = np.asarray(reengagement_times, dtype=float)
    up_c = np.asarray(reengagement_cables, dtype=np.int64)
    parent = np.full(onset_times.size, -1, dtype=np.int64)
    lag = np.full(onset_times.size, np.nan)
    same = np.zeros(onset_times.size, dtype=bool)
    for position, (time, which) in enumerate(zip(onset_times, onset_cables)):
        candidates = (up_t <= time) & (up_t > time - window)
        if same_cable_only:
            candidates &= up_c == which
        rows = np.flatnonzero(candidates)
        if rows.size:
            best = rows[np.argmax(up_t[rows])]
            # ties in time: prefer the same cable, then the lower cable index
            tied = rows[up_t[rows] == up_t[best]]
            if tied.size > 1:
                own = tied[up_c[tied] == which]
                best = own[0] if own.size else tied[np.argmin(up_c[tied])]
            parent[position] = best
            lag[position] = time - up_t[best]
            same[position] = up_c[best] == which
    return PrimaryOnsets(parent < 0, parent, lag, same)


def clean_mask(time: np.ndarray, elongation: np.ndarray, reengagement_times: np.ndarray, window_mask: np.ndarray,
               window: float = EXCLUSION_WINDOW, alive: np.ndarray | None = None) -> np.ndarray:
    """B.3 clean set: taut (e > 0) samples in the recorded window outside every (t_up, t_up + window]."""
    elongation = np.asarray(elongation, dtype=float)
    excluded = in_exclusion(time, reengagement_times, window)
    mask = (elongation > 0.0) & (window_mask & ~excluded)[:, None]
    if alive is not None:
        mask &= alive
    return mask
