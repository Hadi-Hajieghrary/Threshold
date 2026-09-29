"""Per-mark (H4') regime covariates and classes, computed offline (plan v2 II.1, II.2, B.2, B.4).

Everything here is weather-side: the covariates come from the run's weather array and
the 10 ms body-state log (for the chord direction d_i), never from the mark's outcome
(depth, dwell, V_up, T_peak).  Dwell is reported beside the class, never used for it.

Declared definitions (the plan's text, and the reading taken where it leaves a choice):

* Weather: regenerated from ``(spec, master_seed)`` exactly as ``fleet_run.build_run``
  draws it (``regenerate_weather``); the summariser checks bit-equality against the
  run's own array when that is available.
* Conjugate loads on cable i, both positive when the weather closes the gap:
  ``W_rel,i = m_eff (W_L/m_L - W_A,i/m_A) . d_i`` (the plant already logs it at 1 ms as
  ``relative_load``; that log is used, and the offline form is checked against it) and
  ``W^c_i = c_eff (W_L/c_L - W_A,i/c_A) . d_i``, ``c_eff = (1/c_A + 1/c_L)^-1``.
* Time base: the plant's 1 ms event grid.  The weather is the plant's 10 ms zero-order
  hold (index ``floor((t + 1e-9)/0.01)``, as ``UnilateralCableBank.exogenous_force``);
  d_i at a 1 ms sample is the chord vector linearly interpolated between the bracketing
  10 ms state rows and normalised.  At the 10 ms rows this is the plant's own d_i.
* Slack interval of a mark: the samples from its onset (the cable's last geometric
  down-crossing before t_up) to its re-engagement, i.e. exactly its e <= 0 samples.
* Episodes (B.4): on the cable's whole-run W^c series, maximal runs of W^c > T0 after
  merging runs separated by gaps shorter than 1 s (gap < 1000 samples); duration =
  (last exceeding sample - first exceeding sample + 1 ms), peak = max W^c over it.
* t_x, W^c_max (IV.4: "the duration of the W^c > T0 episode overlapping the slack
  interval, gaps below 1 s merged"): the *full* duration and peak of the episode that
  contains the mark's in-slack exceedances; when two or more episodes do, the one
  holding the most in-slack exceedance samples (earliest on a tie) governs, and
  ``n_episodes`` reports the count.  The episode clipped to the slack interval is
  reported as a declared secondary reading (``t_x_clip``, ``Wc_max_clip``,
  ``regime_clip``), gating nothing.
* Classes (H4'): **N** no W^c > T0 sample in the slack interval; **R1** t_x < tau_Lf and
  v_s = (W^c_max - T0) t_x / m_A < 0.3 v_T, v_T = T0/c_A; **R2** t_x > 2 tau_A;
  **transitional** otherwise.  T0 is the cell's nominal pretension.
* Return leg: the samples from the deepest point (argmin e over the slack interval, as
  v1's summariser) to re-engagement; its mean W^c is reported.
* a_bar_ret: the Prop. 3' identity V_up^2/(2 Delta) from the deepest point (Delta the
  recorded depth), never the offset estimator (V_up^2 - u^2)/(2 Delta) (plan II.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tether.physics import constants
from tether.physics.weather import stationary_weather_forces

EVENT_PERIOD = 1.0e-3
WEATHER_PERIOD = constants.WEATHER_PERIOD
M_A = constants.VESSEL_MASS
M_L = constants.LOAD_MASS
C_A = constants.VESSEL_LINEAR_DRAG
C_L = constants.LOAD_LINEAR_DRAG
M_EFF = M_A * M_L / (M_A + M_L)
C_EFF = 1.0 / (1.0 / C_A + 1.0 / C_L)
TAU_A = M_A / C_A  # 1.714 s
TAU_LF = (M_L + 4.0 * M_A) / (C_L + 4.0 * C_A)  # 0.710 s
R1_DURATION = TAU_LF
R2_DURATION = 2.0 * TAU_A  # 3.43 s
R1_SPEED_FRACTION = 0.3
EPISODE_MERGE_GAP = 1.0
CLASSES = ("N", "R1", "R2", "transitional")
CLASS_CODES = {name: code for code, name in enumerate(CLASSES)}


# ----------------------------------------------------------------------------- weather


def regenerate_weather(spec, master_seed: int, vessel_count: int = constants.VESSEL_COUNT) -> np.ndarray | None:
    """The run's weather array, drawn exactly as ``fleet_run.build_run`` draws it."""
    if not spec.weather_scale > 0.0:
        return None
    return stationary_weather_forces(
        master_seed,
        spec.warmup + spec.duration,
        vessel_count=vessel_count,
        distribution=spec.weather_distribution,
        direction=spec.weather_direction,
        scale=spec.weather_scale,
        rho=spec.weather_rho,
        front_angle=spec.weather_front_angle,
    )


def weather_index(times: np.ndarray, samples: int) -> np.ndarray:
    """The plant's zero-order-hold index of each time (``exogenous_force``)."""
    index = np.floor((np.asarray(times, dtype=float) + 1.0e-9) / WEATHER_PERIOD).astype(np.int64)
    return np.minimum(index, samples - 1)


# ----------------------------------------------------------------------------- geometry


def chord_vectors(states: np.ndarray, load_offsets: np.ndarray, vessel_offsets: np.ndarray) -> np.ndarray:
    """Unnormalised chords (rows, N, 2), load attachment -> stern, as ``fleet.fast_cable_terms``."""
    states = np.atleast_2d(np.asarray(states, dtype=float))
    n = load_offsets.shape[0]
    base = 3 * (n + 1)
    cosine0 = np.cos(states[:, 2])[:, None]
    sine0 = np.sin(states[:, 2])[:, None]
    ox, oy = load_offsets[:, 0][None, :], load_offsets[:, 1][None, :]
    lax = cosine0 * ox - sine0 * oy
    lay = sine0 * ox + cosine0 * oy
    heading = states[:, 5:base:3]
    cosine = np.cos(heading)
    sine = np.sin(heading)
    sx, sy = vessel_offsets[:, 0][None, :], vessel_offsets[:, 1][None, :]
    vax = cosine * sx - sine * sy
    vay = sine * sx + cosine * sy
    dx = states[:, 3:base:3] + vax - states[:, 0:1] - lax
    dy = states[:, 4:base:3] + vay - states[:, 1:2] - lay
    return np.stack([dx, dy], axis=-1)


def unit(vectors: np.ndarray) -> np.ndarray:
    length = np.sqrt(vectors[..., 0] * vectors[..., 0] + vectors[..., 1] * vectors[..., 1])
    return vectors / length[..., None]


def conjugate_loads(weather_rows: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(W_rel, W^c) of every cable for weather rows (rows, N+1, 2) and unit chords (rows, N, 2)."""
    load = weather_rows[:, 0:1, :]
    vessel = weather_rows[:, 1:, :]
    w_rel = M_EFF * np.sum((load / M_L - vessel / M_A) * direction, axis=-1)
    w_c = C_EFF * np.sum((load / C_L - vessel / C_A) * direction, axis=-1)
    return w_rel, w_c


@dataclass(frozen=True)
class StateInterpolant:
    """Chord vectors and headings on the 10 ms state rows, interpolated to any time."""

    time: np.ndarray  # (m,)
    chord: np.ndarray  # (m, N, 2) unnormalised
    load_heading: np.ndarray  # (m,) unwrapped
    headings: np.ndarray  # (m, N) unwrapped

    @classmethod
    def from_log(cls, state_time, states, load_offsets, vessel_offsets) -> "StateInterpolant":
        states = np.asarray(states, dtype=float)
        n = load_offsets.shape[0]
        return cls(
            np.asarray(state_time, dtype=float),
            chord_vectors(states, load_offsets, vessel_offsets),
            np.unwrap(states[:, 2]),
            np.unwrap(states[:, 5 : 3 * (n + 1) : 3], axis=0),
        )

    def _bracket(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        times = np.asarray(times, dtype=float)
        last = self.time.size - 1
        k = np.clip(np.searchsorted(self.time, times, side="right") - 1, 0, max(last - 1, 0))
        if last == 0:
            return k, np.zeros(times.shape)
        fraction = np.clip((times - self.time[k]) / (self.time[k + 1] - self.time[k]), 0.0, 1.0)
        return k, fraction

    def direction(self, times: np.ndarray) -> np.ndarray:
        k, fraction = self._bracket(times)
        if self.time.size == 1:
            return unit(self.chord[k])
        chord = self.chord[k] + fraction[:, None, None] * (self.chord[k + 1] - self.chord[k])
        return unit(chord)

    def angles(self, times: np.ndarray) -> dict[str, np.ndarray]:
        """World chord angle sigma, hull-chord misalignment psi, load yaw theta_0 at ``times``."""
        k, fraction = self._bracket(times)
        direction = self.direction(times)
        sigma = np.arctan2(direction[..., 1], direction[..., 0])
        if self.time.size == 1:
            headings, load = self.headings[k], self.load_heading[k]
        else:
            headings = self.headings[k] + fraction[:, None] * (self.headings[k + 1] - self.headings[k])
            load = self.load_heading[k] + fraction * (self.load_heading[k + 1] - self.load_heading[k])
        psi = np.angle(np.exp(1j * (headings - sigma)))
        return {"sigma": sigma, "psi": psi, "load_yaw": load}


def weather_side_series(event_time: np.ndarray, weather: np.ndarray, interpolant: StateInterpolant) -> tuple[np.ndarray, np.ndarray]:
    """(W_rel, W^c) on the 1 ms event grid, shape (samples, N) each."""
    rows = weather[weather_index(event_time, weather.shape[0])]
    return conjugate_loads(rows, interpolant.direction(event_time))


# ----------------------------------------------------------------------------- episodes (B.4)


def exceedance_episodes(values: np.ndarray, level: float, merge_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """(start, stop) sample indices (stop exclusive) of merged ``values > level`` episodes.

    Runs separated by fewer than ``merge_samples`` non-exceeding samples are merged
    (B.4: gaps shorter than 1 s).
    """
    mask = np.asarray(values) > level
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = changes[0::2], changes[1::2]
    if starts.size == 0:
        return starts, stops
    keep = np.concatenate([[True], starts[1:] - stops[:-1] >= merge_samples])
    group = np.cumsum(keep) - 1
    merged_starts = starts[keep]
    merged_stops = np.zeros_like(merged_starts)
    np.maximum.at(merged_stops, group, stops)
    return merged_starts, merged_stops


# ----------------------------------------------------------------------------- classification


def classify(t_x: float, wc_max: float, pretension: float, any_exceedance: bool) -> str:
    """(H4') class from weather-side covariates only."""
    if not any_exceedance:
        return "N"
    if t_x > R2_DURATION:
        return "R2"
    v_s = (wc_max - pretension) * t_x / M_A
    if t_x < R1_DURATION and v_s < R1_SPEED_FRACTION * (pretension / C_A):
        return "R1"
    return "transitional"


MARK_COVARIATES = (
    "t_down",
    "i_down",
    "i_up",
    "Wc_at_onset",
    "Wc_slack_max",
    "Wc_slack_mean",
    "Wc_return_mean",
    "Wrel_slack_max",
    "Wrel_return_mean",
    "any_exceedance",
    "n_episodes",
    "t_x",
    "Wc_max",
    "v_s",
    "episode_start",
    "episode_truncated",
    "t_x_clip",
    "Wc_max_clip",
    "v_s_clip",
    "sigma_onset",
    "sigma_up",
    "psi_onset",
    "psi_up",
    "dwell",
    "a_bar_ret",
    "regime",
    "regime_clip",
)


def mark_covariates(
    *,
    time: np.ndarray,
    elongation: np.ndarray,
    relative_load: np.ndarray,
    wc: np.ndarray,
    marks: dict[str, np.ndarray],
    up_index: np.ndarray,
    down_index: np.ndarray,
    down_time: np.ndarray,
    interpolant: StateInterpolant,
    pretension: float,
    merge_gap: float = EPISODE_MERGE_GAP,
) -> dict[str, np.ndarray]:
    """Per-mark covariates and (H4') class.

    ``up_index``/``down_index``: first sample after each mark's up-/down-crossing (the
    slack samples are ``[down_index, up_index)``); ``down_time``: the onset crossing time.
    """
    n_marks = int(np.asarray(marks["t_up"]).size)
    n_samples, n_cables = wc.shape
    merge = int(round(merge_gap / EVENT_PERIOD))
    episodes = [exceedance_episodes(wc[:, cable], pretension, merge) for cable in range(n_cables)]
    out: dict[str, np.ndarray] = {name: np.full(n_marks, np.nan) for name in MARK_COVARIATES}
    for name in ("i_down", "i_up", "n_episodes"):
        out[name] = np.full(n_marks, -1, dtype=np.int64)
    out["any_exceedance"] = np.zeros(n_marks, dtype=bool)
    out["episode_truncated"] = np.zeros(n_marks, dtype=bool)
    regime = []
    regime_clip = []
    for row in range(n_marks):
        cable = int(marks["cable"][row])
        lo, hi = int(down_index[row]), int(up_index[row])
        out["t_down"][row] = down_time[row]
        out["i_down"][row] = lo
        out["i_up"][row] = hi
        out["dwell"][row] = marks["dwell"][row]
        depth = float(marks["depth"][row])
        if depth > 0.0:
            out["a_bar_ret"][row] = float(marks["v_return"][row]) ** 2 / (2.0 * depth)
        if not (0 < lo < hi <= n_samples):
            regime.append("unassigned")
            regime_clip.append("unassigned")
            continue
        segment = wc[lo:hi, cable]
        # onset value interpolated at the crossing fraction, as the tracker does for W_rel
        e0, e1 = elongation[lo - 1, cable], elongation[lo, cable]
        fraction = -e0 / (e1 - e0)
        out["Wc_at_onset"][row] = wc[lo - 1, cable] + fraction * (wc[lo, cable] - wc[lo - 1, cable])
        out["Wc_slack_max"][row] = float(np.max(segment))
        out["Wc_slack_mean"][row] = float(np.mean(segment))
        turn = lo + int(np.argmin(elongation[lo:hi, cable]))
        out["Wc_return_mean"][row] = float(np.mean(wc[turn:hi, cable]))
        out["Wrel_slack_max"][row] = float(np.max(relative_load[lo:hi, cable]))
        out["Wrel_return_mean"][row] = float(np.mean(relative_load[turn:hi, cable]))
        exceed = np.flatnonzero(segment > pretension) + lo
        if exceed.size == 0:
            out["n_episodes"][row] = 0
            out["t_x"][row] = 0.0
            out["t_x_clip"][row] = 0.0
            regime.append("N")
            regime_clip.append("N")
            continue
        out["any_exceedance"][row] = True
        starts, stops = episodes[cable]
        which = np.searchsorted(starts, exceed, side="right") - 1
        ids, counts = np.unique(which, return_counts=True)
        out["n_episodes"][row] = ids.size
        chosen = int(ids[int(np.argmax(counts))])  # most in-slack samples; np.argmax takes the earliest on a tie
        start, stop = int(starts[chosen]), int(stops[chosen])
        t_x = (stop - start) * EVENT_PERIOD
        wc_max = float(np.max(wc[start:stop, cable]))
        clip_lo, clip_hi = max(start, lo), min(stop, hi)
        t_x_clip = (clip_hi - clip_lo) * EVENT_PERIOD
        wc_max_clip = float(np.max(wc[clip_lo:clip_hi, cable]))
        out["t_x"][row] = t_x
        out["Wc_max"][row] = wc_max
        out["v_s"][row] = (wc_max - pretension) * t_x / M_A
        out["episode_start"][row] = time[start]
        out["episode_truncated"][row] = bool(start == 0 or stop == n_samples)
        out["t_x_clip"][row] = t_x_clip
        out["Wc_max_clip"][row] = wc_max_clip
        out["v_s_clip"][row] = (wc_max_clip - pretension) * t_x_clip / M_A
        regime.append(classify(t_x, wc_max, pretension, True))
        regime_clip.append(classify(t_x_clip, wc_max_clip, pretension, True))
    if n_marks:
        onset = interpolant.angles(out["t_down"])
        up = interpolant.angles(np.asarray(marks["t_up"], dtype=float))
        cables = np.asarray(marks["cable"], dtype=np.int64)
        rows = np.arange(n_marks)
        out["sigma_onset"] = onset["sigma"][rows, cables]
        out["psi_onset"] = onset["psi"][rows, cables]
        out["sigma_up"] = up["sigma"][rows, cables]
        out["psi_up"] = up["psi"][rows, cables]
    out["regime"] = np.asarray(regime, dtype=object).astype(str) if n_marks else np.empty(0, dtype=str)
    out["regime_clip"] = np.asarray(regime_clip, dtype=object).astype(str) if n_marks else np.empty(0, dtype=str)
    return out


def class_counts(regime: np.ndarray, mask: np.ndarray | None = None) -> dict[str, int]:
    regime = np.asarray(regime).astype(str)
    if mask is not None:
        regime = regime[np.asarray(mask, dtype=bool)]
    return {name: int(np.sum(regime == name)) for name in CLASSES + ("unassigned",)}


def class_shares(regime: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | None]:
    counts = class_counts(regime, mask)
    total = sum(counts.values())
    return {name: (count / total if total else None) for name, count in counts.items()}


def constants_table() -> dict[str, float]:
    return {
        "m_A": M_A, "m_L": M_L, "c_A": C_A, "c_L": C_L, "m_eff": M_EFF, "c_eff": C_EFF,
        "tau_A": TAU_A, "tau_Lf": TAU_LF, "R1_duration": R1_DURATION, "R2_duration": R2_DURATION,
        "R1_speed_fraction": R1_SPEED_FRACTION, "episode_merge_gap": EPISODE_MERGE_GAP,
        "weather_period": WEATHER_PERIOD, "event_period": EVENT_PERIOD,
        "c_eff_check_329": math.isclose(C_EFF, 329.0, rel_tol=0.01),
    }
