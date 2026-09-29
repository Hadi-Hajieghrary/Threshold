"""A state-capturing copy of the declared intervention integrator (plan v3, declared in addendum 2).

``integrate_with_state`` is ``tether/campaign/v3/intervention.py::integrate`` (heading mode "recorded") verbatim,
plus the poses, chord elongations, chord rates and alive flags at every 10 ms label.  The pinned module is never
edited; every user of this copy asserts, on the data it draws or scores, that the copy's onsets, ups, parent facts,
closure, steps and 10 ms elongations equal the declared integrator's bit-exactly (``same_branch``), and
``tether/tests/test_v3_present.py`` asserts it on a synthetic tow on both branches.  Consumers: the WP4 clip
(``tether/analysis/v3/present/clip_intervention.py``) and the WP1 sensitivity (``tether/campaign/v3/wp1_sensitivity.py``).
"""

from __future__ import annotations

import math

import numpy as np

from tether.campaign.v3 import intervention as IV
from tether.physics import constants


def integrate_with_state(x: IV.ParentWindow, disable_cable: int | None, window: float, peak_window: float) -> dict:
    """``intervention.integrate`` (heading_mode "recorded", verbatim) plus the poses, elongations and rates at
    every 10 ms label; callers assert its outputs equal ``intervention.integrate``'s bit-exactly."""
    STEP, STATE_PERIOD, STEPS_PER_ROW, CLOSURE_CHORD = IV.STEP, IV.STATE_PERIOD, IV.STEPS_PER_ROW, IV.CLOSURE_CHORD
    n_vessels = x.thrusts.size
    base = 3 * (n_vessels + 1)
    rows = x.state.shape[0]
    horizon_s = (x.t_up - x.start_label) + window
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
    poses, rates, alive_rows = [], [], []          # the additions: state at every 10 ms label
    for n in range(steps + 1):
        la = IV._rotate(load_theta, x.load_offsets)
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
            poses.append(np.concatenate([[pos[0, 0], pos[0, 1], load_theta], np.column_stack([pos[1:], theta[n]]).ravel()]))
            rates.append(rate_all.copy())
        if disable_cable is not None and label >= x.t_up:
            alive[disable_cable] = 0.0
        if n % STEPS_PER_ROW == 0:
            alive_rows.append(alive.copy())
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
        "parent": parent, "closure": closure, "steps": int(n),
        "e_10ms": np.asarray(e_decimated, dtype=float), "t_10ms": np.asarray(t_decimated, dtype=float),
        "poses": np.asarray(poses, dtype=float), "rates": np.asarray(rates, dtype=float), "alive": np.asarray(alive_rows, dtype=float),
    }



def same_branch(a: dict, b: dict) -> bool:
    if len(a["onsets"]) != len(b["onsets"]) or a["steps"] != b["steps"] or a["closure"] != b["closure"]:
        return False
    if not all(np.array_equal(u, v) for u, v in zip(a["onsets"], b["onsets"])) or not all(np.array_equal(u, v) for u, v in zip(a["ups"], b["ups"])):
        return False
    if not (np.array_equal(a["e_10ms"], b["e_10ms"]) and np.array_equal(a["t_10ms"], b["t_10ms"])):
        return False
    return all(a["parent"][k] == b["parent"][k] for k in ("outcome", "t_cf", "v_cf", "T_peak_cf"))



def tension(e: np.ndarray, rate: np.ndarray, alive: np.ndarray, k: float, c: float) -> np.ndarray:
    return np.where((e > 0.0) & (alive > 0.5), np.maximum(k * e + c * rate, 0.0), 0.0)


