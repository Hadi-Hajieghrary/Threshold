"""Outcome labels from plant truth (evaluation only, never a monitor input).

Upcrossings of e = 0 are found in the 1 ms truth with the cable tracker's rule
(e_k <= 0 < e_k+1) and located by linear interpolation of e, the closing speed by linear
interpolation of e' at the same fraction. A tick's label is 1 when the first upcrossing in
[t, t + H] exists and its closing speed exceeds v_b; it is censored (unknown) when the window
runs past the truth series and no upcrossing lies in the covered part. Slack intervals are the
runs of the vessel's own slack flag at the monitor ticks, split where the onset time changes;
an interval's re-engagements are the upcrossings from its onset through its last slack tick
(and the next one if the cable is still slack there), and its virtual severance compares
their engagement peaks (the tracker's T_peak rule) with T_b.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tether.physics import constants

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.intp]

DEFAULT_HORIZON = 2.0
BENIGN_FRACTION = 0.5
_PEAK_CHUNK = 512
_SPAN_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Upcrossings:
    """Sorted upcrossings of one series; ``index`` is the first sample with e > 0."""

    time: FloatArray
    speed: FloatArray
    index: IntArray


def true_upcrossings(time: ArrayLike, elongation: ArrayLike, rate: ArrayLike) -> Upcrossings:
    """Every sample pair with e_k <= 0 < e_k+1, located by linear interpolation."""
    grid = np.asarray(time, dtype=float)
    values = np.asarray(elongation, dtype=float)
    rates = np.asarray(rate, dtype=float)
    if grid.ndim != 1 or values.shape != grid.shape or rates.shape != grid.shape:
        raise ValueError("time, elongation and rate must be 1-D of equal length")
    k = np.flatnonzero((values[:-1] <= 0.0) & (values[1:] > 0.0))
    fraction = -values[k] / (values[k + 1] - values[k])
    return Upcrossings(
        time=grid[k] + fraction * (grid[k + 1] - grid[k]),
        speed=rates[k] + fraction * (rates[k + 1] - rates[k]),
        index=k + 1,
    )


def first_upcrossing(
    crossings: Upcrossings, start: ArrayLike, stop: ArrayLike
) -> tuple[BoolArray, FloatArray, FloatArray, IntArray]:
    """First crossing with start <= time <= stop, vectorised over windows.

    Returns (crossed, time, speed, position in ``crossings`` or -1).
    """
    lower = np.asarray(start, dtype=float)
    upper = np.broadcast_to(np.asarray(stop, dtype=float), lower.shape)
    if crossings.time.size == 0:
        empty = np.full(lower.shape, np.nan)
        return np.zeros(lower.shape, dtype=bool), empty, empty.copy(), np.full(lower.shape, -1)
    position = np.searchsorted(crossings.time, lower, side="left")
    safe = np.minimum(position, crossings.time.size - 1)
    crossed = (position < crossings.time.size) & (crossings.time[safe] <= upper)
    return (
        crossed,
        np.where(crossed, crossings.time[safe], np.nan),
        np.where(crossed, crossings.speed[safe], np.nan),
        np.where(crossed, safe, -1).astype(np.intp),
    )


def _columns(values: ArrayLike, rows: int, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    array = array[:, None] if array.ndim == 1 else array
    if array.ndim != 2 or array.shape[0] != rows:
        raise ValueError(f"{name} must be (n,) or (n, N) on the series grid")
    return array


@dataclass(frozen=True)
class TickOutcomes:
    """First true upcrossing in [t, t + H] at every tick, arrays (m, N).

    ``observed`` marks ticks whose whole window lies inside the truth series; a tick that is
    not observed and has no crossing in the covered part has an unknown label (``censored``).
    """

    time: FloatArray
    slack: BoolArray
    crossed: BoolArray
    crossing_time: FloatArray
    closing_speed: FloatArray
    v_b: FloatArray
    horizon: float
    observed: BoolArray

    @property
    def label(self) -> BoolArray:
        """y: slack tick whose first upcrossing within H closes faster than v_b."""
        return self.slack & self.crossed & (self.closing_speed > self.v_b)

    @property
    def censored(self) -> BoolArray:
        """The window runs past the truth series and no crossing was seen: y is unknown."""
        return ~self.observed & ~self.crossed

    def benign(self, fraction: float = BENIGN_FRACTION) -> BoolArray:
        """The first upcrossing within H, if any, closes no faster than ``fraction`` v_b."""
        return ~(self.crossed & (self.closing_speed > fraction * self.v_b))


def tick_outcomes(
    time: ArrayLike,
    slack: ArrayLike,
    series_time: ArrayLike,
    elongation: ArrayLike,
    rate: ArrayLike,
    v_b: ArrayLike,
    horizon: float = DEFAULT_HORIZON,
) -> TickOutcomes:
    """Label every monitor tick from the 1 ms e / e' truth (arrays (n,) or (n, N))."""
    ticks = np.asarray(time, dtype=float)
    grid = np.asarray(series_time, dtype=float)
    values = _columns(elongation, grid.size, "elongation")
    rates = _columns(rate, grid.size, "rate")
    flags = np.asarray(slack, dtype=bool).reshape(ticks.size, values.shape[1])
    speeds = np.broadcast_to(np.asarray(v_b, dtype=float), (values.shape[1],)).copy()
    shape = flags.shape
    crossed = np.zeros(shape, dtype=bool)
    crossing_time = np.full(shape, np.nan)
    closing_speed = np.full(shape, np.nan)
    for vessel in range(shape[1]):
        crossings = true_upcrossings(grid, values[:, vessel], rates[:, vessel])
        hit, when, speed, _ = first_upcrossing(crossings, ticks, ticks + horizon)
        crossed[:, vessel], crossing_time[:, vessel], closing_speed[:, vessel] = hit, when, speed
    covered = _covered(grid, ticks, ticks + horizon)
    return TickOutcomes(
        time=ticks,
        slack=flags,
        crossed=crossed,
        crossing_time=crossing_time,
        closing_speed=closing_speed,
        v_b=speeds,
        horizon=float(horizon),
        observed=np.broadcast_to(covered[:, None], shape).copy(),
    )


def _covered(grid: FloatArray, start: FloatArray, stop: FloatArray) -> BoolArray:
    """Windows [start, stop] inside the series span (to rounding)."""
    if grid.size == 0:
        return np.zeros(np.shape(start), dtype=bool)
    slack = _SPAN_TOLERANCE * max(1.0, abs(float(grid[-1])))
    return (start >= grid[0] - slack) & (stop <= grid[-1] + slack)


@dataclass(frozen=True)
class SlackIntervals:
    """Slack intervals over the ticks, ordered by vessel then time; ``last`` is inclusive."""

    vessel: IntArray
    first: IntArray
    last: IntArray
    start_time: FloatArray
    end_time: FloatArray
    onset_time: FloatArray
    shape: tuple[int, int]

    @property
    def count(self) -> int:
        return int(self.vessel.size)

    def tick_index(self) -> IntArray:
        """Interval id of every tick, (m, N), -1 while taut."""
        index = np.full(self.shape, -1, dtype=np.intp)
        for interval, (vessel, first, last) in enumerate(
            zip(self.vessel, self.first, self.last)
        ):
            index[first : last + 1, vessel] = interval
        return index


def slack_intervals(
    time: ArrayLike, slack: ArrayLike, onset_time: ArrayLike | None = None
) -> SlackIntervals:
    """Runs of the slack flag per vessel, split where a finite onset time changes."""
    ticks = np.asarray(time, dtype=float)
    flags = np.asarray(slack, dtype=bool)
    flags = flags[:, None] if flags.ndim == 1 else flags
    if flags.shape[0] != ticks.size:
        raise ValueError("slack must be (m,) or (m, N) over the ticks")
    previous = np.vstack([np.zeros((1, flags.shape[1]), dtype=bool), flags[:-1]])
    begins = flags & ~previous
    onsets = np.full(flags.shape, np.nan)
    if onset_time is not None:
        onsets = np.asarray(onset_time, dtype=float).reshape(flags.shape)
        prior = np.vstack([np.full((1, flags.shape[1]), np.nan), onsets[:-1]])
        changed = np.isfinite(onsets) & np.isfinite(prior) & (onsets != prior)
        begins |= flags & previous & changed
    following = np.vstack([begins[1:], np.ones((1, flags.shape[1]), dtype=bool)])
    closing = np.vstack([flags[1:], np.zeros((1, flags.shape[1]), dtype=bool)])
    ends = flags & (~closing | following)
    vessels, firsts, lasts = [], [], []
    for vessel in range(flags.shape[1]):
        starts = np.flatnonzero(begins[:, vessel])
        stops = np.flatnonzero(ends[:, vessel])
        vessels.append(np.full(starts.size, vessel, dtype=np.intp))
        firsts.append(starts)
        lasts.append(stops)
    vessel = np.concatenate(vessels).astype(np.intp)
    first = np.concatenate(firsts).astype(np.intp)
    last = np.concatenate(lasts).astype(np.intp)
    return SlackIntervals(
        vessel=vessel,
        first=first,
        last=last,
        start_time=ticks[first],
        end_time=ticks[last],
        onset_time=onsets[first, vessel] if first.size else np.zeros(0),
        shape=(int(flags.shape[0]), int(flags.shape[1])),
    )


def engagement_peak(
    elongation: FloatArray,
    tension: FloatArray,
    index: int,
    crossing_speed: float,
    damping: float = constants.CABLE_DAMPING,
) -> float:
    """The tracker's T_peak: from max(c V, q) at the first taut sample, the running maximum of
    q = k e + c e' until q first falls below it or e returns to <= 0 (then max with that q).
    Censored at the end of the series."""
    peak = max(damping * crossing_speed, float(tension[index]))
    start = index + 1
    while start < elongation.size:
        stop = min(elongation.size, start + _PEAK_CHUNK)
        values, loads = elongation[start:stop], tension[start:stop]
        running = np.maximum.accumulate(np.concatenate(([peak], loads)))[:-1]
        down = values <= 0.0
        hits = np.flatnonzero(down | (loads < running))
        if hits.size:
            hit = hits[0]
            return float(max(running[hit], loads[hit]) if down[hit] else running[hit])
        peak = max(peak, float(loads.max()))
        start = stop
    return float(peak)


@dataclass(frozen=True)
class IntervalOutcomes:
    """True re-engagement of every slack interval (NaN where none is attributed);
    ``censored`` marks intervals without one, still slack at their last tick, whose window
    runs past the truth series; ``count`` is the number of re-engagements attributed."""

    reengaged: BoolArray
    reengagement_time: FloatArray
    reengagement_speed: FloatArray
    peak_tension: FloatArray
    v_b: FloatArray
    breaking_strength: float
    censored: BoolArray
    count: IntArray

    @property
    def dangerous(self) -> BoolArray:
        return self.reengaged & (self.reengagement_speed > self.v_b)

    @property
    def severed(self) -> BoolArray:
        """Virtual severance: the engagement peak exceeds T_b."""
        return self.reengaged & (self.peak_tension > self.breaking_strength)

    def benign(self, fraction: float = BENIGN_FRACTION) -> BoolArray:
        return ~(self.reengaged & (self.reengagement_speed > fraction * self.v_b))


def interval_outcomes(
    intervals: SlackIntervals,
    series_time: ArrayLike,
    elongation: ArrayLike,
    rate: ArrayLike,
    v_b: ArrayLike,
    breaking_strength: float,
    horizon: float = DEFAULT_HORIZON,
    stiffness: float = constants.CABLE_STIFFNESS,
    damping: float = constants.CABLE_DAMPING,
) -> IntervalOutcomes:
    """True re-engagements of each interval from the 1 ms truth.

    An interval's re-engagements are the true upcrossings from its onset (its first slack
    tick when the onset is not finite or later) through its last slack tick, and, when the
    cable is still truly slack at that tick, the first upcrossing after it within H and
    before the next interval of the vessel begins. A debounced flag stays set for a few
    samples after the true re-engagement, so a short excursion can end before its first
    slack tick; a later excursion's upcrossing is never attributed to it. The reported
    re-engagement is the first whose engagement peak exceeds T_b (``severed``), else the
    last one, which ends the interval.
    """
    grid = np.asarray(series_time, dtype=float)
    values = _columns(elongation, grid.size, "elongation")
    rates = _columns(rate, grid.size, "rate")
    speeds = np.broadcast_to(np.asarray(v_b, dtype=float), (values.shape[1],))
    count = intervals.count
    level = float(breaking_strength)
    reengaged = np.zeros(count, dtype=bool)
    when = np.full(count, np.nan)
    speed = np.full(count, np.nan)
    peak = np.full(count, np.nan)
    attributed = np.zeros(count, dtype=np.intp)
    open_end = np.ones(count, dtype=bool)
    onset = np.asarray(intervals.onset_time, dtype=float)
    early = np.isfinite(onset) & (onset < intervals.start_time)
    begin = np.where(early, onset, intervals.start_time)
    for vessel in np.unique(intervals.vessel) if grid.size else ():
        rows = np.flatnonzero(intervals.vessel == vessel)
        rows = rows[np.argsort(intervals.start_time[rows], kind="stable")]
        crossings = true_upcrossings(grid, values[:, vessel], rates[:, vessel])
        tension = stiffness * values[:, vessel] + damping * rates[:, vessel]
        following = np.append(begin[rows][1:], np.inf)
        stop = intervals.end_time[rows]
        still_slack = np.interp(stop, grid, values[:, vessel]) <= 0.0
        open_end[rows] = still_slack
        low = np.searchsorted(crossings.time, begin[rows], side="left")
        high = np.searchsorted(crossings.time, np.minimum(stop, following), side="right")
        for row, first, last, slack, cap, end in zip(rows, low, high, still_slack, following, stop):
            members = list(range(first, max(first, last)))
            after = max(first, last)
            if slack and after < crossings.time.size:
                moment = crossings.time[after]
                if moment <= end + horizon and moment < cap:
                    members.append(after)
            if not members:
                continue
            peaks = [
                engagement_peak(
                    values[:, vessel],
                    tension,
                    int(crossings.index[which]),
                    float(crossings.speed[which]),
                    damping,
                )
                for which in members
            ]
            severing = [which for which, value in zip(members, peaks) if value > level]
            chosen = severing[0] if severing else members[-1]
            reengaged[row] = True
            attributed[row] = len(members)
            when[row] = crossings.time[chosen]
            speed[row] = crossings.speed[chosen]
            peak[row] = peaks[members.index(chosen)]
    covered = _covered(grid, begin, intervals.end_time + horizon)
    return IntervalOutcomes(
        reengaged=reengaged,
        reengagement_time=when,
        reengagement_speed=speed,
        peak_tension=peak,
        v_b=speeds[intervals.vessel].astype(float) if count else np.zeros(0),
        breaking_strength=level,
        censored=~reengaged & open_end & ~covered,
        count=attributed,
    )
