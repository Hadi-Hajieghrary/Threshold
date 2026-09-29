"""Snap transmission through the payload: closed form, operator with yaw, exact linear response.

Plan v3 WP0 (ref/cascade_plan_v3.md).  Everything in this module is DERIVED; nothing is a
measurement.  The measurement lives in tether/campaign/v3/reducer.py (WP1).

Three levels of description of the same mechanism, each a hypothesis weaker than the last:

* ``closed_form_ratio`` -- the paper's 2k/(m_L omega_L omega_e) under the impulse and
  taut-neighbour hypotheses; it equals 2 sqrt(m_A / ((N-1)(m_A + m_L))), so the cable
  stiffness cancels (Theorem 1 of reports/v3/theory_v3.md).
* ``closed_form_operator`` -- the N x N matrix (2k/(omega_L omega_e)) W^T M_L^-1 W, the
  Delassus operator of the cable set weighted by the two modes; it carries the payload's yaw
  inertia and the attachment offsets and reduces to 0.44 cos(sigma_i - sigma_j) when the lever
  arms vanish (fan formation) (Theorem 2).
* ``exact_response`` -- the response of every cable's tension to the half-sine tension pulse
  of a re-engagement, integrated through the closed-loop taut tow linearised about the steady
  tow (drag, heading loop, yaw and offsets all included, cable j's own spring removed while it
  is slack); this makes no impulse approximation, which matters because on this plant
  epsilon = omega_L/omega_e = 0.88 is not small (Theorem 3).

Conventions: a pulse of peak ``T_peak`` on cable j; the NEIGHBOUR SWING of cable i is the drop
of its tension below the pre-pulse value, ``drop_i = max_t (-(q_i(t)))``, over one engagement
period 2 pi/omega_e after the pulse starts, normalised by ``T_peak``; a positive drop means the
neighbour is unloaded (the payload is yanked toward vessel j, which also lies on the side of
every neighbour whose chord direction has a positive projection on d_j).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm

from tether.physics import constants
from tether.theory import reduced_lti as R

PULSE_SHAPES = ("half_sine", "rectangular")


# ----------------------------------------------------------------------------- closed form


def pair_reduced_mass(load_mass: float = constants.LOAD_MASS, vessel_mass: float = constants.VESSEL_MASS) -> float:
    return vessel_mass * load_mass / (vessel_mass + load_mass)


def frequencies(
    stiffness: float = constants.CABLE_STIFFNESS,
    damping: float = constants.CABLE_DAMPING,
    vessel_count: int = constants.VESSEL_COUNT,
    load_mass: float = constants.LOAD_MASS,
    vessel_mass: float = constants.VESSEL_MASS,
) -> dict:
    """The two modes of the transmission argument and the ratios built from them.

    ``omega_e`` is the cable mode of one vessel-payload pair (reduced mass ``mu``); its period
    is the campaign's ENGAGEMENT_PERIOD.  ``omega_L`` is the payload against the other N - 1
    taut cables, translation only.  ``epsilon = omega_L / omega_e`` measures how far the
    contact is from an impulse (0 = impulse).
    """
    mu = pair_reduced_mass(load_mass, vessel_mass)
    omega_e = math.sqrt(stiffness / mu)
    omega_l = math.sqrt((vessel_count - 1) * stiffness / load_mass)
    return {
        "mu": mu,
        "omega_e": omega_e,
        "omega_L": omega_l,
        "epsilon": omega_l / omega_e,
        "engagement_period": 2.0 * math.pi / omega_e,
        "contact_duration": math.pi / omega_e,
        "load_half_period": math.pi / omega_l,
        "zeta": damping / (2.0 * math.sqrt(stiffness * mu)),
    }


def closed_form_ratio(
    vessel_count: int = constants.VESSEL_COUNT,
    load_mass: float = constants.LOAD_MASS,
    vessel_mass: float = constants.VESSEL_MASS,
) -> float:
    """Neighbour swing over T_peak for a half-sine pulse: 2 sqrt(m_A / ((N-1)(m_A + m_L)))."""
    return 2.0 * math.sqrt(vessel_mass / ((vessel_count - 1) * (vessel_mass + load_mass)))


def closed_form_ratio_from_modes(
    stiffness: float = constants.CABLE_STIFFNESS,
    vessel_count: int = constants.VESSEL_COUNT,
    load_mass: float = constants.LOAD_MASS,
    vessel_mass: float = constants.VESSEL_MASS,
) -> float:
    """The paper's form 2k/(m_L omega_L omega_e); equals ``closed_form_ratio`` identically."""
    f = frequencies(stiffness, vessel_count=vessel_count, load_mass=load_mass, vessel_mass=vessel_mass)
    return 2.0 * stiffness / (load_mass * f["omega_L"] * f["omega_e"])


def pulse_factor(shape: str) -> float:
    """Impulse of a pulse of the same peak and duration relative to the half sine."""
    if shape == "half_sine":
        return 1.0
    if shape == "rectangular":
        return math.pi / 2.0
    raise ValueError(f"unknown pulse shape: {shape}")


def unloading_threshold(ratio: float) -> float:
    """T_peak / T0 above which a neighbour on the same chord direction is unloaded to slack."""
    return 1.0 / ratio


# ----------------------------------------------------------------------------- operator with yaw


def _rotate(angle: float, vectors: np.ndarray) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    vectors = np.asarray(vectors, dtype=float)
    return np.stack([c * vectors[:, 0] - s * vectors[:, 1], s * vectors[:, 0] + c * vectors[:, 1]], axis=1)


def lever_matrix(load_offsets: np.ndarray, load_yaw: float, chord_directions: np.ndarray) -> np.ndarray:
    """Wrench directions ``w_i = (d_i, r_i x d_i)`` as the columns of a (3, N) matrix.

    ``load_offsets`` are the attachment points in the payload frame, ``chord_directions`` the
    world-frame unit chord vectors (from the attachment toward the vessel).
    """
    arms = _rotate(load_yaw, load_offsets)
    d = np.asarray(chord_directions, dtype=float)
    lever = arms[:, 0] * d[:, 1] - arms[:, 1] * d[:, 0]
    return np.vstack([d.T, lever[None, :]])


def closed_form_operator(
    load_offsets: np.ndarray,
    load_yaw: float,
    chord_directions: np.ndarray,
    stiffness: float = constants.CABLE_STIFFNESS,
    load_mass: float = constants.LOAD_MASS,
    load_inertia: float = constants.LOAD_YAW_INERTIA,
    vessel_mass: float = constants.VESSEL_MASS,
) -> np.ndarray:
    """T_ij = (2k/(omega_L omega_e)) (W^T M_L^-1 W)_ij: neighbour i's swing per unit peak on j.

    Row i is the neighbour, column j the snapping cable.  With zero lever arms the entry is
    ``closed_form_ratio() * cos(sigma_i - sigma_j)``; the diagonal is the self term.
    """
    w = lever_matrix(load_offsets, load_yaw, chord_directions)
    n = w.shape[1]
    minv = np.diag([1.0 / load_mass, 1.0 / load_mass, 1.0 / load_inertia])
    f = frequencies(stiffness, vessel_count=n, load_mass=load_mass, vessel_mass=vessel_mass)
    return (2.0 * stiffness / (f["omega_L"] * f["omega_e"])) * (w.T @ minv @ w)


def geometry_factor(
    operator: np.ndarray,
    load_mass: float = constants.LOAD_MASS,
    vessel_mass: float = constants.VESSEL_MASS,
) -> np.ndarray:
    """The operator divided by the closed-form ratio: cos(sigma_i - sigma_j) plus lever terms.

    This is the v3 normaliser of the band statistic (WP1): a measured swing divided by
    ``T_peak * geometry_factor[i, j]`` is compared with the declared band [0.2, 0.9].
    """
    n = operator.shape[0]
    return np.asarray(operator, dtype=float) / closed_form_ratio(n, load_mass, vessel_mass)


def formation_operator(
    formation: str,
    arc_half_angle: float | None = None,
    stiffness: float = constants.CABLE_STIFFNESS,
) -> np.ndarray:
    """The operator at the design geometry of a formation (load yaw 0, chords at design angles)."""
    geometry = R.tow_geometry(formation, arc_half_angle=arc_half_angle)
    directions = np.stack([np.cos(geometry.cable_angles), np.sin(geometry.cable_angles)], axis=1)
    return closed_form_operator(geometry.load_offsets, 0.0, directions, stiffness)


# ----------------------------------------------------------------------------- exact linear response


@dataclass(frozen=True)
class ResponseSpec:
    """One cell of the taut tow and the pulse model to drive it with."""

    inp: R.ReducedModelInput
    stiffness: float = constants.CABLE_STIFFNESS
    damping: float = constants.CABLE_DAMPING
    pulse: str = "half_sine"
    parent_removed: bool = True  # M1: the snapping cable is slack except for the pulse
    horizon: float | None = None  # seconds after the pulse start; None = one engagement period
    step: float = 1.0e-4
    inertia: tuple | None = None  # optional (load_mass, load_inertia, vessel_mass, vessel_inertia)
    # Hypothesis switches for Theorem 3's recovery of the closed form (tests only):
    pulse_duration: float | None = None  # None = pi/omega_e; shorter pulses keep the same impulse
    drag_scale: float = 1.0  # 0 removes every linear drag term from the linearised field


@dataclass(frozen=True)
class LinearPlant:
    """``x' = A x + B_force u``, ``q = C_q x``, ``e = C_e x`` about the steady tow (reduced coords)."""

    A: np.ndarray
    B_force: np.ndarray  # (state, N): unit tension along each cable's chord
    C_q: np.ndarray  # (N, state): tension deviation k e + c edot
    C_e: np.ndarray  # (N, state): elongation deviation
    x0: np.ndarray  # reduced equilibrium
    directions: np.ndarray  # (N, 2) equilibrium chord directions
    load_offsets: np.ndarray
    removed_cable: int | None


@dataclass(frozen=True)
class ResponseResult:
    times: np.ndarray  # (T,)
    tension: np.ndarray  # (T, N) tension deviation per unit peak; column parent = nan under M1
    drop: np.ndarray  # (N,) max over the window of -tension / T_peak
    rise: np.ndarray  # (N,) max over the window of +tension / T_peak
    lag_of_min: np.ndarray  # (N,) seconds after the pulse start at which the drop is attained
    parent: int


def _inertia_vector(vessel_count: int, inertia: tuple | None) -> np.ndarray:
    if inertia is None:
        return R.generalized_inertia(vessel_count)
    load_mass, load_inertia, vessel_mass, vessel_inertia = inertia
    return np.array([load_mass, load_mass, load_inertia] + [vessel_mass, vessel_mass, vessel_inertia] * vessel_count)


def _plant(spec: ResponseSpec) -> R.TowPlant:
    plant = R.tow_plant(spec.inp)
    plant = replace(plant, stiffness=float(spec.stiffness), damping=float(spec.damping))
    return replace(plant, heading_reference=R.tow_heading_reference(plant))


def _cable_generalized_force(plant: R.TowPlant, full_state: np.ndarray, cable: int, tension: float | None = None) -> np.ndarray:
    """(N+1, 3) generalized force of one cable at ``tension`` (None: its own k e + c edot)."""
    n = plant.geometry.vessel_count
    terms = R.cable_terms(plant, full_state)
    if tension is None:
        tension = plant.stiffness * terms["elongation"][cable] + plant.damping * terms["rate"][cable]
    direction = terms["direction"][cable]
    force = tension * direction
    out = np.zeros((n + 1, 3))
    out[0, :2] = force
    out[0, 2] = terms["load_arm"][cable, 0] * force[1] - terms["load_arm"][cable, 1] * force[0]
    out[1 + cable, :2] = -force
    out[1 + cable, 2] = -(terms["vessel_arm"][cable, 0] * force[1] - terms["vessel_arm"][cable, 1] * force[0])
    return out


def linearised_plant(spec: ResponseSpec, removed_cable: int | None) -> LinearPlant:
    """Linearise the closed-loop taut tow about its steady state (weather zero).

    With ``removed_cable = j`` the spring-damper of cable j is subtracted from the vector field
    (the cable is slack during the parent excursion); the pulse then enters only through
    ``B_force[:, j]``.  Finite-difference Jacobians, central, step 1e-5 (reduced_lti's
    STATE_STEP).
    """
    plant = _plant(spec)
    n = plant.geometry.vessel_count
    x0 = R.tow_equilibrium(plant)
    inertia = _inertia_vector(n, spec.inertia)
    pose_size = 1 + 3 * n
    zero_weather = np.zeros(2 * (n + 1))

    linear_drag = np.array([constants.LOAD_LINEAR_DRAG] + [constants.VESSEL_LINEAR_DRAG] * n)
    angular_drag = np.array([constants.LOAD_ANGULAR_DRAG] + [constants.VESSEL_ANGULAR_DRAG] * n)

    def field(x: np.ndarray) -> np.ndarray:
        full = R.to_full(x, n)
        force = R.generalized_forces(plant, full, zero_weather).reshape(n + 1, 3)
        if removed_cable is not None:
            force = force - _cable_generalized_force(plant, full, removed_cable)
        velocities = x[pose_size:].reshape(n + 1, 3)
        if spec.drag_scale != 1.0:
            if plant.drag_law != "linear":
                raise ValueError("drag_scale is defined for the linear drag law only")
            # generalized_forces subtracted the full linear drag; add back the unwanted share.
            force[:, :2] += (1.0 - spec.drag_scale) * linear_drag[:, None] * velocities[:, :2]
            force[:, 2] += (1.0 - spec.drag_scale) * angular_drag * velocities[:, 2]
        kinematic = velocities[1:].copy()
        kinematic[:, :2] -= velocities[0, :2]
        return np.concatenate([velocities[0, 2:3], kinematic.ravel(), force.ravel() / inertia])

    a = R._jacobian(field, x0, R.STATE_STEP)
    full0 = R.to_full(x0, n)
    b = np.zeros((x0.size, n))
    for cable in range(n):
        unit = _cable_generalized_force(plant, full0, cable, tension=1.0)
        b[pose_size:, cable] = unit.ravel() / inertia
    c_e = R._jacobian(lambda x: R.cable_terms(plant, R.to_full(x, n))["elongation"], x0, R.STATE_STEP)
    c_rate = R._jacobian(lambda x: R.cable_terms(plant, R.to_full(x, n))["rate"], x0, R.STATE_STEP)
    c_q = plant.stiffness * c_e + plant.damping * c_rate
    terms = R.cable_terms(plant, full0)
    return LinearPlant(a, b, c_q, c_e, x0, terms["direction"], plant.geometry.load_offsets, removed_cable)


def zoh_response(a: np.ndarray, b: np.ndarray, c: np.ndarray, inputs: np.ndarray, dt: float) -> np.ndarray:
    """``y_k = C x_k`` for ``x' = A x + B u``, ``x_0 = 0``, ``u`` held over each step (ZOH, expm)."""
    inputs = np.asarray(inputs, dtype=float)
    if inputs.ndim == 1:
        inputs = inputs[:, None]
    state_size, input_size = b.shape
    augmented = np.zeros((state_size + input_size, state_size + input_size))
    augmented[:state_size, :state_size] = a * dt
    augmented[:state_size, state_size:] = b * dt
    phi = expm(augmented)
    ad, bd = phi[:state_size, :state_size], phi[:state_size, state_size:]
    x = np.zeros(state_size)
    out = np.zeros((inputs.shape[0] + 1, c.shape[0]))
    for k in range(inputs.shape[0]):
        x = ad @ x + bd @ inputs[k]
        out[k + 1] = c @ x
    return out


def pulse_samples(shape: str, omega_e: float, times: np.ndarray, duration: float | None = None) -> np.ndarray:
    """Tension pulse of the re-engagement sampled at ``times``, unit peak at the nominal duration.

    The nominal duration is the contact ``pi/omega_e``.  A shorter ``duration`` (Theorem 3's
    impulse limit) keeps the nominal pulse's impulse by scaling the amplitude up by
    ``nominal / duration``, so the response can be compared with the closed form directly.
    """
    nominal = math.pi / omega_e
    duration = nominal if duration is None else float(duration)
    amplitude = nominal / duration
    inside = (times >= 0.0) & (times < duration)
    if shape == "half_sine":
        phase = math.pi * np.clip(times, 0.0, duration) / duration
        return np.where(inside, amplitude * np.sin(phase), 0.0)
    if shape == "rectangular":
        return amplitude * inside.astype(float)
    raise ValueError(f"unknown pulse shape: {shape}")


def exact_response(spec: ResponseSpec, parent: int, plant: LinearPlant | None = None) -> ResponseResult:
    """Tension response of every cable to a unit-peak pulse on ``parent`` (M1 unless told otherwise)."""
    n = R.tow_geometry(spec.inp.formation, arc_half_angle=spec.inp.arc_half_angle).vessel_count
    # The pulse is the plant's snap: its duration pi/omega_e and impulse 2 T_peak/omega_e use the
    # plant's pair reduced mass whatever ``inertia`` override the response dynamics carry.
    f = frequencies(spec.stiffness, spec.damping, n)
    horizon = f["engagement_period"] if spec.horizon is None else float(spec.horizon)
    if plant is None:
        plant = linearised_plant(spec, parent if spec.parent_removed else None)
    steps = int(math.ceil(horizon / spec.step))
    sample_times = np.arange(steps) * spec.step
    u = np.zeros((steps, n))
    u[:, parent] = pulse_samples(spec.pulse, f["omega_e"], sample_times, spec.pulse_duration)
    tension = zoh_response(plant.A, plant.B_force, plant.C_q, u, spec.step)
    times = np.arange(steps + 1) * spec.step
    drop = np.max(-tension, axis=0)
    rise = np.max(tension, axis=0)
    lag = times[np.argmax(-tension, axis=0)]
    if spec.parent_removed:
        tension[:, parent] = np.nan
        drop[parent] = rise[parent] = lag[parent] = np.nan
    return ResponseResult(times, tension, drop, rise, lag, parent)


def response_matrix(spec: ResponseSpec) -> np.ndarray:
    """R[i, j] = drop of neighbour i per unit peak on j; the diagonal is nan under M1."""
    n = R.tow_geometry(spec.inp.formation, arc_half_angle=spec.inp.arc_half_angle).vessel_count
    out = np.full((n, n), np.nan)
    shared = None if spec.parent_removed else linearised_plant(spec, None)
    for j in range(n):
        out[:, j] = exact_response(spec, j, shared).drop
    return out


def predicted_pair_table(spec: ResponseSpec, cell_name: str) -> dict:
    """Everything WP0 commits for one cell, JSON-ready; all entries are derived quantities."""
    plant = _plant(spec)
    n = plant.geometry.vessel_count
    directions = np.stack([np.cos(plant.geometry.cable_angles), np.sin(plant.geometry.cable_angles)], axis=1)
    operator = closed_form_operator(plant.geometry.load_offsets, 0.0, directions, spec.stiffness)
    factor = geometry_factor(operator)
    r_m1 = response_matrix(spec)
    r_m2 = response_matrix(replace(spec, parent_removed=False))
    r_rect = response_matrix(replace(spec, pulse="rectangular"))
    off = ~np.eye(n, dtype=bool)
    normalised = np.where(off, r_m1 / np.where(factor == 0.0, np.nan, factor), np.nan)
    f = frequencies(spec.stiffness, spec.damping, n)
    return {
        "cell": cell_name,
        "status": "FORECAST (linearised closed-loop tow + pulse model; not a measurement)",
        "formation": plant.geometry.formation,
        "arc_half_angle": plant.geometry.arc_half_angle,
        "pretension": spec.inp.pretension,
        "heading_gain": spec.inp.heading_gain,
        "stiffness": spec.stiffness,
        "damping": spec.damping,
        "frequencies": f,
        "closed_form_ratio": closed_form_ratio(n),
        "closed_form_ratio_rectangular": closed_form_ratio(n) * pulse_factor("rectangular"),
        "operator": operator.tolist(),
        "geometry_factor": factor.tolist(),
        "response_M1_half_sine": _nan_to_none(r_m1),
        "response_M2_parent_taut": _nan_to_none(r_m2),
        "response_M1_rectangular": _nan_to_none(r_rect),
        "response_over_geometry_factor": _nan_to_none(normalised),
        "band_statistic_median_offdiag": float(np.nanmedian(normalised[off])),
        "window_seconds": f["engagement_period"],
    }


def _nan_to_none(array: np.ndarray) -> list:
    return [[None if not np.isfinite(v) else float(v) for v in row] for row in np.asarray(array, dtype=float)]
