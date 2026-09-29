"""Interventional counterfactuals for cascade attribution (plan v3 WP2, test T2.3/T2.4).

For a parent re-engagement of cable j at t_up, the fleet is re-integrated from the recorded state
twice under the recorded weather and the recorded headings: the FACTUAL branch (every cable
active; it must reproduce the record, which is the validation T2.3) and the COUNTERFACTUAL branch
in which cable j applies no force (alive = 0), so the payload never receives the snap.  Other
cables' onsets in the window that occur in the factual branch but not in the counterfactual one
are causal offspring of the snap.

The integrator generalises tether/campaign/v2/p6_t0.py::integrate (fleet mode, validated there):
semi-implicit Euler at 0.5 ms, labels on the record's 10 ms axis, vessel headings and yaw rates
prescribed from the record (linear between rows; the addendum-2 extrapolation through the row
that brackets t_up for the parent's vessel), constant operating thrusts, the plant's cable law with
the run's k and c, linear drag, the payload's yaw integrated.  In the counterfactual branch the
parent vessel's heading is held at its value at t_up (its recorded heading after t_up carries the
snap's yaw kick); ``heading_mode = "held"`` freezes every heading at t_up in both branches
(declared sensitivity).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from tether.campaign.v2 import events as ev
from tether.campaign.v2 import regime as rg
from tether.physics import constants

STEP = 5.0e-4
STATE_PERIOD = 0.01
STEPS_PER_ROW = int(round(STATE_PERIOD / STEP))
CLOSURE_CHORD = 1.0
PEAK_WINDOW = 1.0
LEAD_S = 0.5  # window starts at the row at or before t_up - LEAD_S


@dataclass(frozen=True)
class InterventionSpec:
    max_parents: int = 40
    horizon: float = 4.0
    window: float = 3.0
    peak_window: float = PEAK_WINDOW
    heading_mode: str = "recorded"  # or "held"
    x_threshold: float = 2.3  # every event above this is sampled first
    sensitivity_held: bool = True  # also run the "held" heading mode


@dataclass(frozen=True)
class ParentWindow:
    cable: int
    t_up: float
    v_return: float
    T_peak: float
    event_id: int
    x: float
    thrusts: np.ndarray
    load_offsets: np.ndarray
    vessel_offsets: np.ndarray
    start_label: float
    state: np.ndarray  # (K, 6 (N+1)) rows from the start row
    weather: np.ndarray  # (W, N+1, 2) rows from weather_start
    weather_start: int
    stiffness: float
    damping: float
    rest_length: float = constants.CABLE_REST_LENGTH
    alive: np.ndarray = field(default_factory=lambda: np.ones(5, dtype=bool))


def select_parents(events: dict, pretension: float, spec: InterventionSpec, rng=None) -> np.ndarray:
    """Every event with x > x_threshold by descending x, then a systematic sample of the rest, up to the cap."""
    x = np.asarray(events["T_peak"], dtype=float) / pretension
    order = np.argsort(-x, kind="stable")
    big = [int(k) for k in order if x[k] > spec.x_threshold]
    rest = [int(k) for k in order if x[k] <= spec.x_threshold]
    chosen = big[: spec.max_parents]
    room = spec.max_parents - len(chosen)
    if room > 0 and rest:
        stride = max(len(rest) / room, 1.0)
        chosen.extend(rest[int(round(k * stride))] for k in range(min(room, len(rest))) if int(round(k * stride)) < len(rest))
    return np.array(sorted(set(chosen)), dtype=np.int64)


def parent_window(record: dict, events: dict, index: int, thrusts: np.ndarray, spec: InterventionSpec, pretension: float) -> ParentWindow | None:
    state_time = np.asarray(record["state_time"], dtype=float)
    t_up = float(events["t_up"][index])
    k0 = int(np.searchsorted(state_time, t_up - LEAD_S, side="right") - 1)
    k1 = int(np.searchsorted(state_time, t_up + spec.horizon, side="right")) + 2
    if k0 < 0 or k1 > state_time.size:
        return None
    weather = record.get("run_weather")
    if weather is None:
        weather = rg.regenerate_weather(record["spec"], record["master_seed"], record["elongation"].shape[1])
    w0 = int(math.floor((state_time[k0] + STEP + 1.0e-9) / STATE_PERIOD))
    w1 = int(math.floor((state_time[k1 - 1] + STEP + 1.0e-9) / STATE_PERIOD)) + 2
    if w0 < 0 or w1 > weather.shape[0]:
        return None
    marks_t_up = np.asarray(record["all_marks"]["t_up"], dtype=float)
    marks_cable = np.asarray(record["all_marks"]["cable"], dtype=np.int64)
    marks_v = np.asarray(record["all_marks"]["v_return"], dtype=float)
    hit = np.flatnonzero((marks_cable == int(events["cable_j"][index])) & (np.abs(marks_t_up - t_up) < 1.0e-9))
    v_return = float(marks_v[hit[0]]) if hit.size else math.nan
    return ParentWindow(
        cable=int(events["cable_j"][index]), t_up=t_up, v_return=v_return, T_peak=float(events["T_peak"][index]),
        event_id=int(index), x=float(events["T_peak"][index]) / pretension, thrusts=np.asarray(thrusts, dtype=float),
        load_offsets=np.asarray(record["load_offsets"], dtype=float), vessel_offsets=np.asarray(record["vessel_offsets"], dtype=float),
        start_label=float(state_time[k0]), state=np.asarray(record["state"][k0:k1], dtype=float),
        weather=np.asarray(weather[w0:w1], dtype=float), weather_start=w0,
        stiffness=float(record["stiffness"]), damping=float(record["damping"]),
    )


def _rotate(angle, vectors):
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * vectors[..., 0] - s * vectors[..., 1], s * vectors[..., 0] + c * vectors[..., 1]], axis=-1)


def integrate(x: ParentWindow, disable_cable: int | None, heading_mode: str = "recorded", horizon: float | None = None,
              window: float = 3.0, peak_window: float = PEAK_WINDOW) -> dict:
    """One branch from the window's first row: onsets and up-crossings of every cable in (t_up, t_up + window]."""
    n_vessels = x.thrusts.size
    base = 3 * (n_vessels + 1)
    rows = x.state.shape[0]
    horizon_s = (x.t_up - x.start_label) + (window if horizon is None else horizon)
    steps = int(min(round(horizon_s / STEP), (rows - 2) * STEPS_PER_ROW))
    n_idx = np.arange(steps + 1)
    labels = x.start_label + n_idx * STEP
    actual = labels + STEP
    row = n_idx // STEPS_PER_ROW
    frac = (n_idx % STEPS_PER_ROW) / STEPS_PER_ROW
    after = np.minimum(row + 1, rows - 1)
    theta_rows = x.state[:, 5:base:3]
    omega_rows = x.state[:, base + 5 :: 3]
    theta = theta_rows[row] + frac[:, None] * (theta_rows[after] - theta_rows[row])
    omega = omega_rows[row] + frac[:, None] * (omega_rows[after] - omega_rows[row])
    i = x.cable
    k_up = int(math.floor((x.t_up - x.start_label) / STATE_PERIOD + 1.0e-9))
    n_up = int(math.ceil((x.t_up - x.start_label) / STEP - 1.0e-9))
    if 0 <= k_up < rows - 1 and k_up * STEPS_PER_ROW <= steps:
        lo, hi = k_up * STEPS_PER_ROW, min(n_up + 1, steps + 1)
        elapsed = (n_idx[lo:hi] - lo) * STEP
        theta[lo:hi, i] = theta_rows[k_up, i] + omega_rows[k_up, i] * elapsed
        omega[lo:hi, i] = omega_rows[k_up, i]
        if disable_cable is not None and n_up <= steps:
            theta[n_up:, i] = theta_rows[k_up, i] + omega_rows[k_up, i] * (x.t_up - (x.start_label + k_up * STATE_PERIOD))
            omega[n_up:, i] = 0.0
    if heading_mode == "held" and n_up <= steps:
        theta[n_up:] = theta[n_up]
        omega[n_up:] = 0.0
    elif heading_mode != "recorded":
        raise ValueError(f"unknown heading mode: {heading_mode}")
    widx = np.floor((actual + 1.0e-9) / STATE_PERIOD).astype(int) - x.weather_start
    weather = x.weather[np.clip(widx, 0, x.weather.shape[0] - 1)]
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    heading_unit = np.stack([cos_t, sin_t], axis=-1)
    so = x.vessel_offsets
    stern = np.stack([cos_t * so[None, :, 0] - sin_t * so[None, :, 1], sin_t * so[None, :, 0] + cos_t * so[None, :, 1]], axis=-1)
    stern_rate = omega[:, :, None] * np.stack([-stern[..., 1], stern[..., 0]], axis=-1)
    s0 = x.state[0]
    pos = s0[:base].reshape(n_vessels + 1, 3)[:, :2].copy()
    vel = s0[base:].reshape(n_vessels + 1, 3)[:, :2].copy()
    load_theta = float(s0[2])
    load_omega = float(s0[base + 2])
    k_c, c_c, rest = x.stiffness, x.damping, x.rest_length
    alive = x.alive.astype(float).copy()
    # the counterfactual removes cable j's force from t_up on (declared); before t_up both branches
    # carry the record's forces, so they coincide there whatever the length of j's excursion
    inverse_mass = 1.0 / np.array([constants.LOAD_MASS] + [constants.VESSEL_MASS] * n_vessels)[:, None]
    drag = np.array([constants.LOAD_LINEAR_DRAG] + [constants.VESSEL_LINEAR_DRAG] * n_vessels)[:, None]
    thrust = x.thrusts
    prev_e = None
    onsets = [[] for _ in range(n_vessels)]
    ups = [[] for _ in range(n_vessels)]
    parent = {"outcome": "horizon", "t_cf": None, "v_cf": None, "T_peak_cf": None}
    crossed_at = None
    peak = 0.0
    closure = None
    e_decimated, t_decimated = [], []
    for n in range(steps + 1):
        la = _rotate(load_theta, x.load_offsets)
        dx_all = pos[1:] + stern[n] - pos[0] - la
        rel = vel[1:] + stern_rate[n] - vel[0] - load_omega * np.stack([-la[:, 1], la[:, 0]], axis=-1)
        length_all = np.sqrt(np.einsum("ij,ij->i", dx_all, dx_all))
        u = dx_all / length_all[:, None]
        rate_all = np.einsum("ij,ij->i", rel, u)
        e_all = length_all - rest
        label = float(labels[n])
        if n % STEPS_PER_ROW == 0:
            e_decimated.append(e_all.copy())
            t_decimated.append(label)
        if disable_cable is not None and label >= x.t_up:
            alive[disable_cable] = 0.0
        if prev_e is not None and label > x.t_up:
            for c in range(n_vessels):
                if alive[c] < 0.5 or c == disable_cable:
                    continue
                if prev_e[c] > 0.0 >= e_all[c] and label - x.t_up <= window:
                    fraction = prev_e[c] / (prev_e[c] - e_all[c])
                    onsets[c].append(float(labels[n - 1] + fraction * STEP))
                if prev_e[c] <= 0.0 < e_all[c] and label - x.t_up <= window:
                    ups[c].append(float(labels[n - 1] + (-prev_e[c] / (e_all[c] - prev_e[c])) * STEP))
        if disable_cable is None and prev_e is not None:
            if crossed_at is None and prev_e[i] <= 0.0 < e_all[i] and label >= x.t_up - STATE_PERIOD:
                fraction = -prev_e[i] / (e_all[i] - prev_e[i])
                v_cf = prev_rate[i] + fraction * (rate_all[i] - prev_rate[i])
                parent.update(outcome="reengaged", t_cf=float(labels[n - 1] + fraction * STEP), v_cf=float(max(v_cf, 0.0)))
                crossed_at = n
                peak = max(c_c * v_cf, k_c * e_all[i] + c_c * rate_all[i])
            elif crossed_at is not None and parent["T_peak_cf"] is None:
                if e_all[i] <= 0.0:
                    parent["T_peak_cf"] = float(peak)
                else:
                    q = k_c * e_all[i] + c_c * rate_all[i]
                    if q < peak or (n - crossed_at) * STEP > peak_window:
                        parent["T_peak_cf"] = float(peak)
                    else:
                        peak = q
        prev_e, prev_rate = e_all.copy(), rate_all.copy()
        shortest = float(length_all.min())
        if closure is None and shortest < CLOSURE_CHORD:
            closure = label
            break
        if n == steps:
            break
        tension = np.where((e_all > 0.0) & (alive > 0.5), np.maximum(k_c * e_all + c_c * rate_all, 0.0), 0.0)
        cable_force = tension[:, None] * u
        force = np.empty_like(pos)
        force[0] = cable_force.sum(axis=0)
        force[1:] = thrust[:, None] * heading_unit[n] - cable_force
        force += weather[n] - drag * vel
        torque = float(np.sum(la[:, 0] * cable_force[:, 1] - la[:, 1] * cable_force[:, 0])) - constants.LOAD_ANGULAR_DRAG * load_omega
        vel = vel + STEP * force * inverse_mass
        load_omega = load_omega + STEP * torque / constants.LOAD_YAW_INERTIA
        pos = pos + STEP * vel
        load_theta = load_theta + STEP * load_omega
    if parent["outcome"] == "reengaged" and parent["T_peak_cf"] is None:
        parent["T_peak_cf"] = float(peak)
    return {
        "onsets": [np.asarray(v, dtype=float) for v in onsets],
        "ups": [np.asarray(v, dtype=float) for v in ups],
        "parent": parent,
        "closure": closure,
        "steps": int(n),
        "e_10ms": np.asarray(e_decimated, dtype=float),
        "t_10ms": np.asarray(t_decimated, dtype=float),
    }


def causal_offspring(factual: dict, counterfactual: dict, parent_cable: int, window: float) -> np.ndarray:
    n = len(factual["onsets"])
    out = np.zeros(n, dtype=np.int64)
    for c in range(n):
        if c == parent_cable:
            continue
        out[c] = max(0, int(factual["onsets"][c].size) - int(counterfactual["onsets"][c].size))
    return out


def recorded_onsets(record: dict, downs: ev.Crossings, t_up: float, window: float, parent_cable: int) -> list[np.ndarray]:
    n = record["elongation"].shape[1]
    inside = (downs.time > t_up) & (downs.time <= t_up + window)
    return [np.sort(downs.time[inside & (downs.cable == c)]) if c != parent_cable else np.zeros(0) for c in range(n)]


def _match(a: np.ndarray, b: np.ndarray, tolerance: float) -> bool:
    if a.size != b.size:
        return False
    return bool(np.all(np.abs(np.sort(a) - np.sort(b)) <= tolerance))


def run_interventions(run, record: dict, marks: dict, declustering: ev.Declustering, table: dict, meta: dict,
                      spec: InterventionSpec) -> dict:
    """Factual and counterfactual branches for the selected parents of one run (in-worker)."""
    events = table["events"]
    pretension = float(meta["pretension"])
    thrusts = np.asarray(run.fleet.operating.thrusts, dtype=float)
    _, downs = ev.crossings(record["event_time"], record["elongation"], np.asarray(record["alive"], dtype=bool))
    rows = []
    for index in select_parents(events, pretension, spec):
        x = parent_window(record, events, int(index), thrusts, spec, pretension)
        if x is None:
            continue
        factual = integrate(x, None, "recorded", window=spec.window, peak_window=spec.peak_window)
        counter = integrate(x, x.cable, "recorded", window=spec.window, peak_window=spec.peak_window)
        recorded = recorded_onsets(record, downs, x.t_up, spec.window, x.cable)
        row = {
            "event_id": int(index), "cable": x.cable, "t_up": x.t_up, "x": x.x, "T_peak": x.T_peak, "v_return": x.v_return,
            "factual_parent": factual["parent"], "factual_closure": factual["closure"], "counterfactual_closure": counter["closure"],
            "recorded_onsets": [int(v.size) for v in recorded],
            "factual_onsets": [int(v.size) for v in factual["onsets"]],
            "counterfactual_onsets": [int(v.size) for v in counter["onsets"]],
            "onset_set_reproduced": all(_match(recorded[c], factual["onsets"][c], 0.02) for c in range(len(recorded)) if c != x.cable),
            "causal": causal_offspring(factual, counter, x.cable, spec.window),
            "measured_cross": np.asarray(events["offspring_per_cable"][int(index)], dtype=np.int64),
        }
        if spec.sensitivity_held:
            held_f = integrate(x, None, "held", window=spec.window, peak_window=spec.peak_window)
            held_c = integrate(x, x.cable, "held", window=spec.window, peak_window=spec.peak_window)
            row["causal_held"] = causal_offspring(held_f, held_c, x.cable, spec.window)
            row["held_parent"] = held_f["parent"]
        rows.append(row)
    return {"spec": {"max_parents": spec.max_parents, "horizon": spec.horizon, "window": spec.window, "x_threshold": spec.x_threshold},
            "rows": rows}


def validate(rows: list[dict], dt_max: float = 0.005, dv_p95: float = 0.01, dv_max: float = 0.03, dpeak_p95: float = 0.02,
             onset_share: float = 0.95) -> dict:
    """T2.3: does the factual branch reproduce the record?  Thresholds are the declared ones."""
    dv, dt, dpeak, failed, reproduced = [], [], [], [], []
    for row in rows:
        p = row["factual_parent"]
        if p["outcome"] != "reengaged":
            failed.append({"cable": row["cable"], "t_up": row["t_up"], "outcome": p["outcome"]})
            continue
        dt.append(p["t_cf"] - row["t_up"])
        if np.isfinite(row["v_return"]):
            dv.append(p["v_cf"] - row["v_return"])
        if p["T_peak_cf"] is not None and row["T_peak"] > 0:
            dpeak.append(p["T_peak_cf"] / row["T_peak"] - 1.0)
        reproduced.append(bool(row["onset_set_reproduced"]))
    adv, adt, adp = np.abs(dv), np.abs(dt), np.abs(dpeak)
    share = float(np.mean(reproduced)) if reproduced else math.nan
    passed = (not failed and adv.size > 0 and float(np.percentile(adv, 95)) <= dv_p95 and float(adv.max()) <= dv_max
              and float(adt.max()) <= dt_max and (adp.size == 0 or float(np.percentile(adp, 95)) <= dpeak_p95) and share >= onset_share)
    return {
        "n": len(rows), "not_reengaged": failed,
        "abs_dv_m_s": None if not adv.size else {"median": float(np.median(adv)), "p95": float(np.percentile(adv, 95)), "max": float(adv.max())},
        "abs_dt_s": None if not adt.size else {"median": float(np.median(adt)), "max": float(adt.max())},
        "T_peak_relative_error": None if not adp.size else {"median_abs": float(np.median(adp)), "p95_abs": float(np.percentile(adp, 95)), "max_abs": float(adp.max())},
        "onset_set_reproduced_share": share,
        "thresholds": {"dt_max": dt_max, "dv_p95": dv_p95, "dv_max": dv_max, "dpeak_p95": dpeak_p95, "onset_share": onset_share},
        "passed": bool(passed),
    }
