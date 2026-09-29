"""Runs declustering of threshold exceedances and declustered level upcrossings."""

from __future__ import annotations

import numpy as np


def _series(times, values) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    if times.shape != values.shape:
        raise ValueError("times and values must have equal length")
    if times.size > 1 and np.any(np.diff(times) < 0.0):
        raise ValueError("times must be non-decreasing")
    return times, values


def runs_decluster(
    times, values, threshold: float, run_length: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster maxima of the exceedances ``values > threshold``.

    Consecutive exceedances belong to one cluster unless the time between them exceeds
    ``run_length``.  Returns ``(peak_times, peaks)``, one entry per cluster; ties within a
    cluster resolve to the earliest sample.
    """
    times, values = _series(times, values)
    exceedances = np.flatnonzero(values > threshold)
    if exceedances.size == 0:
        return np.empty(0), np.empty(0)
    exceedance_times = times[exceedances]
    exceedance_values = values[exceedances]
    starts = np.concatenate(
        ([0], np.flatnonzero(np.diff(exceedance_times) > run_length) + 1)
    )
    sizes = np.diff(np.append(starts, exceedances.size))
    cluster = np.repeat(np.arange(starts.size), sizes)
    order = np.lexsort((-exceedance_values, cluster))
    peak_positions = order[starts]
    return exceedance_times[peak_positions], exceedance_values[peak_positions]


def upcrossing_counts(
    times, values, levels, run_length: float, valid=None
) -> np.ndarray:
    """Declustered upcrossing count of each level.

    An upcrossing of ``level`` at sample ``i`` is ``values[i-1] < level <= values[i]`` with
    both samples valid.  It is counted only if no valid sample at or above the level occurred
    within ``run_length`` before it, i.e. once the series has stayed below the level for
    longer than ``run_length``; otherwise it is merged into the running cluster.  Invalid
    (or non-finite) samples neither cross nor extend a cluster.
    """
    times, values = _series(times, values)
    levels = np.asarray(levels, dtype=float).ravel()
    usable = np.isfinite(values)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool).ravel()
        if valid.shape != values.shape:
            raise ValueError("valid must match values")
        usable &= valid
    previous_usable = np.concatenate(([False], usable[:-1]))
    previous_values = np.concatenate(([np.nan], values[:-1]))
    kept = np.flatnonzero(usable)
    kept_times = times[kept]
    kept_values = values[kept]
    kept_previous_usable = previous_usable[kept]
    kept_previous_values = previous_values[kept]
    counts = np.zeros(levels.size, dtype=np.int64)
    for index, level in enumerate(levels):
        above = np.flatnonzero(kept_values >= level)
        if above.size == 0:
            continue
        crossing = kept_previous_usable[above] & (kept_previous_values[above] < level)
        separated = np.empty(above.size, dtype=bool)
        separated[0] = True
        separated[1:] = np.diff(kept_times[above]) > run_length
        counts[index] = np.count_nonzero(crossing & separated)
    return counts
