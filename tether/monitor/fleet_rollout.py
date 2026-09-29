"""Vectorised planar six-body fleet rollout: the prediction model of plan v2 II.9 / IV.9.

The rollout integrates ``N`` sampled futures of the load and the five vessels from one
starting state, jointly, so that a re-engagement of any other cable within the horizon
acts on the slack cable through the load (Prop. 13's transmission) without any model
term of its own.  It is the plant's own physics, re-implemented on arrays:

* bodies on world planar joints, generalized coordinates ``(x, y, theta)`` per body and
  world-frame twists ``(vx, vy, omega)`` (no Coriolis term: every body frame is at its
  centre of mass, ``tether.physics.plant._spatial_inertia``);
* unilateral Kelvin-Voigt cables ``T = (k e + c edot)_+`` while ``e > 0``, applied +T d at
  the load attachment and -T d at the vessel's stern (``tether.physics.fleet``);
* linear drag on every body's translation, angular drag on every body's yaw;
* each vessel's surge force along its hull axis (per-step thrust array supplied by the
  caller, e.g. the mission's surge schedule under the controller's zero-order hold);
* weather forces on the 10 ms zero-order-hold grid of ``HullForces``: a per-sample
  common-mode standardized state times a per-sample-instant, per-body scale, plus a
  deterministic per-body part (the squall), exactly the structure of
  ``tether.physics.weather`` for the common-mode classes;
* the integrator is the plant's own discrete scheme: symplectic (semi-implicit) Euler,
  ``v+ = v + h M^-1 f(q, v, t)``, ``q+ = q + h v+``, which is what Drake's SAP step reduces
  to without contact and without joint damping (all forces enter through the applied
  generalized-force port and are evaluated at the start of the step).

Vessel headings: ``"held"`` freezes every vessel's heading at its starting value with a
zero yaw rate for the whole horizon (the heading controller is taken to hold the hull);
``"controlled"`` integrates the vessel yaw with the angular drag, the cable's stern
moment and a model of the IV.7 heading law evaluated on the true heading and true bearing
(``k_h wrap(ref - theta) - k_c sin(bearing)``) under the controller's 20 ms hold.  The
load's yaw is always integrated.

The monitor output of a rollout is, per sample, the first upcrossing of ``e_i = 0`` of the
watched cable on the step grid (``e_prev <= 0 < e_next``), located and given a closing
speed by linear interpolation exactly as ``tether.monitor.outcomes.true_upcrossings``
does on the 1 ms truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from tether.physics import constants
from tether.physics.fleet import FleetGeometry

FloatArray = NDArray[np.float64]

ROLLOUT_STEP = 5.0e-4
WEATHER_PERIOD = constants.WEATHER_PERIOD
CONTROL_PERIOD = 0.02
HEADING_MODES = ("held", "controlled")
_TIME_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class RolloutModel:
    """Pinned plant constants and the formation geometry of one rollout."""

    geometry: FleetGeometry
    rest_length: float = constants.CABLE_REST_LENGTH
    stiffness: float = constants.CABLE_STIFFNESS
    damping: float = constants.CABLE_DAMPING
    load_mass: float = constants.LOAD_MASS
    load_yaw_inertia: float = constants.LOAD_YAW_INERTIA
    vessel_mass: float = constants.VESSEL_MASS
    vessel_yaw_inertia: float = constants.VESSEL_YAW_INERTIA
    load_linear_drag: float = constants.LOAD_LINEAR_DRAG
    vessel_linear_drag: float = constants.VESSEL_LINEAR_DRAG
    load_angular_drag: float = constants.LOAD_ANGULAR_DRAG
    vessel_angular_drag: float = constants.VESSEL_ANGULAR_DRAG
    step: float = ROLLOUT_STEP
    headings: str = "held"
    heading_gain: float = 500.0
    trim_gain: float = 100.0

    def __post_init__(self) -> None:
        if self.headings not in HEADING_MODES:
            raise ValueError(f"heading mode must be one of {HEADING_MODES}")
        if not self.step > 0.0:
            raise ValueError("the rollout step must be positive")
        ratio = WEATHER_PERIOD / self.step
        if abs(ratio - round(ratio)) > 1.0e-9:
            raise ValueError("the rollout step must divide the 10 ms weather period")

    @property
    def vessel_count(self) -> int:
        return self.geometry.vessel_count


@dataclass(frozen=True)
class WeatherFuture:
    """Weather force on every body at the 10 ms samples ``k0, k0 + 1, ...`` of the rollout.

    ``force[n, j, b] = scale[j, b] * common[n, j] + fixed[j, b]`` (2-vectors), where ``common``
    (samples, K, 2) is the per-sample common-mode state, ``scale`` (K, bodies) the per-body
    multiplier at sample ``j`` (stationary std times the ramp) and ``fixed`` (K, bodies, 2) the
    deterministic part (squall, or the whole realised weather).  Sample ``j`` acts over
    ``[(k0 + j) T_w, (k0 + j + 1) T_w)``.
    """

    first_index: int
    common: FloatArray
    scale: FloatArray
    fixed: FloatArray

    @property
    def length(self) -> int:
        return int(self.fixed.shape[0])

    def forces(self, j: int) -> tuple[FloatArray, FloatArray]:
        """(fx, fy), each (samples, bodies), of sample ``j``."""
        common = self.common[:, j, :]
        scale = self.scale[j]
        fixed = self.fixed[j]
        fx = common[:, 0:1] * scale[None, :] + fixed[None, :, 0]
        fy = common[:, 1:2] * scale[None, :] + fixed[None, :, 1]
        return fx, fy


def fixed_weather(first_index: int, forces: FloatArray, samples: int = 1) -> WeatherFuture:
    """A known future: ``forces`` (K, bodies, 2) acting on every sample."""
    forces = np.asarray(forces, dtype=float)
    return WeatherFuture(
        first_index=int(first_index),
        common=np.zeros((samples, forces.shape[0], 2)),
        scale=np.zeros(forces.shape[:2]),
        fixed=forces,
    )


@dataclass
class RolloutResult:
    """First upcrossing of the watched cable per sample (NaN where none within the horizon),
    plus optional recorded (e, edot) of every cable at the requested step indices."""

    crossed: np.ndarray
    crossing_time: FloatArray
    closing_speed: FloatArray
    steps_taken: int
    record_steps: np.ndarray | None = None
    record_elongation: FloatArray | None = None
    record_rate: FloatArray | None = None
    record_state: FloatArray | None = None
    other_reengaged: np.ndarray | None = None
    extras: dict = field(default_factory=dict)

    def hazard(self, v_b: float, horizon: float | None = None) -> float:
        """Fraction of samples whose first upcrossing within ``horizon`` closes above v_b."""
        hit = self.crossed.copy()
        if horizon is not None:
            hit &= self.crossing_time <= horizon + _TIME_TOLERANCE
        return float(np.count_nonzero(hit & (self.closing_speed > v_b)) / self.crossed.size)


def controller_hold_time(time: float) -> float:
    """Time of the controller update whose command acts on the plant step at ``time``.

    The 50 Hz controller and the plant's step are simultaneous discrete updates, so the
    step at an update instant uses the previous update's command.
    """
    index = math.ceil(time / CONTROL_PERIOD - _TIME_TOLERANCE) - 1
    return max(index, 0) * CONTROL_PERIOD if index >= 0 else 0.0


def thrust_on_steps(surge: Callable[[float], np.ndarray], start: float, steps: int, step: float = ROLLOUT_STEP) -> FloatArray:
    """(steps, N) surge command acting on each plant step from ``start``, under the
    controller's 20 ms hold."""
    cache: dict[float, np.ndarray] = {}
    rows = []
    for n in range(steps):
        held = round(controller_hold_time(start + n * step), 9)
        if held not in cache:
            cache[held] = np.asarray(surge(held), dtype=float)
        rows.append(cache[held])
    return np.array(rows)


def _is_update_instant(time: float) -> bool:
    ratio = time / CONTROL_PERIOD
    return abs(ratio - round(ratio)) < 1.0e-6


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rollout(
    model: RolloutModel,
    state: np.ndarray,
    start_time: float,
    horizon: float,
    watched: int,
    weather: WeatherFuture,
    thrust: FloatArray,
    heading_reference: Callable[[float], np.ndarray] | None = None,
    record_steps: np.ndarray | None = None,
    stop_when_all_crossed: bool = True,
    other_threshold: float | None = None,
) -> RolloutResult:
    """Integrate every weather sample from the plant state ``state`` = [q; v] at ``start_time``.

    ``thrust`` is (steps, N) with at least ``horizon / step`` rows.  Returns the first
    upcrossing of cable ``watched`` in (0, horizon] per sample.  With ``record_steps``
    (sorted step indices) the (e, edot) of every cable and the full state are recorded
    there (sample 0 only is kept for the state).  ``other_threshold`` marks samples in
    which a cable other than ``watched`` re-engages within the horizon with a tension above
    it on the first taut step (reported, not used by the hazard).
    """
    n = model.vessel_count
    bodies = n + 1
    samples = weather.common.shape[0]
    h = model.step
    total = int(round(horizon / h))
    if thrust.shape[0] < total:
        raise ValueError("thrust must cover every step of the horizon")
    state = np.asarray(state, dtype=float)
    q = state[: 3 * bodies].reshape(bodies, 3)
    v = state[3 * bodies :].reshape(bodies, 3)
    ones = np.ones(samples)
    px0, py0, th0 = q[0, 0] * ones, q[0, 1] * ones, q[0, 2] * ones
    vx0, vy0, om0 = v[0, 0] * ones, v[0, 1] * ones, v[0, 2] * ones
    px = np.tile(q[1:, 0], (samples, 1))
    py = np.tile(q[1:, 1], (samples, 1))
    vx = np.tile(v[1:, 0], (samples, 1))
    vy = np.tile(v[1:, 1], (samples, 1))
    controlled = model.headings == "controlled"
    # Held headings: one (1, N) heading row broadcast over the samples, zero yaw rate.
    th = np.tile(q[1:, 2], (samples, 1)) if controlled else q[1:, 2][None, :].copy()
    om = np.tile(v[1:, 2], (samples, 1)) if controlled else np.zeros((1, n))
    ox = model.geometry.load_offsets[:, 0][None, :]
    oy = model.geometry.load_offsets[:, 1][None, :]
    sx = model.geometry.vessel_offsets[:, 0][None, :]
    sy = model.geometry.vessel_offsets[:, 1][None, :]
    cos_v, sin_v = np.cos(th), np.sin(th)
    vax = cos_v * sx - sin_v * sy
    vay = sin_v * sx + cos_v * sy
    k, c, rest = model.stiffness, model.damping, model.rest_length
    inv_ml, inv_il = 1.0 / model.load_mass, 1.0 / model.load_yaw_inertia
    inv_mv, inv_iv = 1.0 / model.vessel_mass, 1.0 / model.vessel_yaw_inertia
    cl, cal = model.load_linear_drag, model.load_angular_drag
    cv, cav = model.vessel_linear_drag, model.vessel_angular_drag
    crossed = np.zeros(samples, dtype=bool)
    crossing_time = np.full(samples, np.nan)
    closing_speed = np.full(samples, np.nan)
    other = np.zeros(samples, dtype=bool) if other_threshold is not None else None
    previous_e = None
    previous_edot = None
    previous_all_e = None
    record = None
    if record_steps is not None:
        record_steps = np.asarray(record_steps, dtype=int)
        record = {"e": np.full((record_steps.size, samples, n), np.nan), "edot": np.full((record_steps.size, samples, n), np.nan),
                  "state": np.full((record_steps.size, 6 * bodies), np.nan)}
        record_position = {int(s): i for i, s in enumerate(record_steps)}
    yaw = np.zeros((samples, n))
    last_step = total
    wx = wy = None
    weather_index = None
    for step in range(total + 1):
        time = start_time + step * h
        cos0, sin0 = np.cos(th0), np.sin(th0)
        lax = cos0[:, None] * ox - sin0[:, None] * oy
        lay = sin0[:, None] * ox + cos0[:, None] * oy
        if controlled:
            cos_v, sin_v = np.cos(th), np.sin(th)
            vax = cos_v * sx - sin_v * sy
            vay = sin_v * sx + cos_v * sy
        dx = px + vax - px0[:, None] - lax
        dy = py + vay - py0[:, None] - lay
        rvx = vx - om * vay - vx0[:, None] + om0[:, None] * lay
        rvy = vy + om * vax - vy0[:, None] - om0[:, None] * lax
        length = np.sqrt(dx * dx + dy * dy)
        np.maximum(length, 1.0e-9, out=length)
        ux = dx / length
        uy = dy / length
        e = length - rest
        edot = rvx * ux + rvy * uy
        if record is not None and step in record_position:
            row = record_position[step]
            record["e"][row] = e
            record["edot"][row] = edot
            record["state"][row] = _pack(px0, py0, th0, vx0, vy0, om0, px, py, th, vx, vy, om)
        if step > 0:
            ew, ew_prev = e[:, watched], previous_e
            up = (~crossed) & (ew_prev <= 0.0) & (ew > 0.0)
            if up.any():
                fraction = -ew_prev[up] / (ew[up] - ew_prev[up])
                crossing_time[up] = (step - 1 + fraction) * h
                closing_speed[up] = previous_edot[up] + fraction * (edot[up, watched] - previous_edot[up])
                crossed |= up
            if other is not None:
                rising = (previous_all_e <= 0.0) & (e > 0.0)
                rising[:, watched] = False
                if rising.any():
                    first_tension = np.maximum(k * e + c * edot, 0.0)
                    other |= np.any(rising & (first_tension > other_threshold), axis=1)
            if stop_when_all_crossed and crossed.all() and record is None:
                last_step = step
                break
        if step == total:
            break
        previous_e = e[:, watched].copy()
        previous_edot = edot[:, watched].copy()
        if other is not None:
            previous_all_e = e.copy()
        tension = k * e + c * edot
        np.maximum(tension, 0.0, out=tension)
        tension[e <= 0.0] = 0.0
        fx = tension * ux
        fy = tension * uy
        index = int(math.floor((time + _TIME_TOLERANCE) / WEATHER_PERIOD)) - weather.first_index
        if index != weather_index:
            if not 0 <= index < weather.length:
                raise ValueError("the weather future does not cover the horizon")
            wx, wy = weather.forces(index)
            weather_index = index
        # load
        ax0 = (fx.sum(axis=1) - cl * vx0 + wx[:, 0]) * inv_ml
        ay0 = (fy.sum(axis=1) - cl * vy0 + wy[:, 0]) * inv_ml
        al0 = (np.sum(lax * fy - lay * fx, axis=1) - cal * om0) * inv_il
        surge = thrust[step]
        ax = (-fx - cv * vx + surge[None, :] * cos_v + wx[:, 1:]) * inv_mv
        ay = (-fy - cv * vy + surge[None, :] * sin_v + wy[:, 1:]) * inv_mv
        vx0 = vx0 + h * ax0
        vy0 = vy0 + h * ay0
        om0 = om0 + h * al0
        vx = vx + h * ax
        vy = vy + h * ay
        if controlled:
            if step == 0:
                # The command in force at the start: the law on the starting state.
                yaw = _heading_law(model, heading_reference, controller_hold_time(time), th, px, py, px0, py0, lax, lay, vax, vay)
            moment = -(vax * fy - vay * fx) - cav * om + yaw
            if _is_update_instant(time):
                # The 50 Hz update at this instant acts from the next plant step on.
                yaw = _heading_law(model, heading_reference, time, th, px, py, px0, py0, lax, lay, vax, vay)
            om = om + h * moment * inv_iv
            th = th + h * om
        px0 = px0 + h * vx0
        py0 = py0 + h * vy0
        th0 = th0 + h * om0
        px = px + h * vx
        py = py + h * vy
    result = RolloutResult(crossed=crossed, crossing_time=crossing_time, closing_speed=closing_speed, steps_taken=int(last_step),
                           other_reengaged=other)
    if record is not None:
        result.record_steps = record_steps
        result.record_elongation = record["e"]
        result.record_rate = record["edot"]
        result.record_state = record["state"]
    return result


def _heading_law(model, reference, time, th, px, py, px0, py0, lax, lay, vax, vay):
    """IV.7 yaw command on the true heading and the true cable bearing (``controlled`` mode)."""
    if reference is None:
        raise ValueError("controlled headings need a heading reference")
    ref = np.asarray(reference(time), dtype=float)[None, :]
    tx = px0[:, None] + lax - px - vax
    ty = py0[:, None] + lay - py - vay
    cos_v, sin_v = np.cos(th), np.sin(th)
    body_x = cos_v * tx + sin_v * ty
    body_y = -sin_v * tx + cos_v * ty
    bearing = np.arctan2(body_y, -body_x)
    return model.heading_gain * _wrap(ref - th) - model.trim_gain * np.sin(bearing)


def _pack(px0, py0, th0, vx0, vy0, om0, px, py, th, vx, vy, om) -> FloatArray:
    q = [px0[0], py0[0], th0[0]]
    v = [vx0[0], vy0[0], om0[0]]
    for i in range(px.shape[1]):
        q += [px[0, i], py[0, i], th[0, i]]
        v += [vx[0, i], vy[0, i], om[0, i]]
    return np.array(q + v)


# ----------------------------------------------------------------------------- weather law


def common_ar1_future(
    rng: np.random.Generator,
    samples: int,
    current: np.ndarray,
    count: int,
    phi: float,
    distribution: str = "student_t3",
) -> FloatArray:
    """(samples, count + 1, 2) standardized common-mode AR(1) state: index 0 is the known
    current state ``current`` (2,), indices 1..count follow ``x_j = phi x_(j-1) +
    sqrt(1 - phi^2) eps_j`` with ``eps_j`` the generator's standardized innovation law
    (``tether.physics.weather._draw_standardized``: a 2-D Gaussian times one common
    ``sqrt(1 / chi^2_3)`` radial factor for t3).  Draw order per call: all Gaussians
    (samples, count, 2), then all chi-square factors (samples, count)."""
    gaussian = rng.standard_normal((samples, count, 2))
    if distribution == "student_t3":
        radial = np.sqrt(1.0 / rng.chisquare(3.0, size=(samples, count)))
        innovations = gaussian * radial[:, :, None]
    elif distribution == "gaussian":
        innovations = gaussian
    else:
        raise ValueError(f"unknown innovation distribution: {distribution}")
    scale = math.sqrt(1.0 - phi * phi)
    out = np.empty((samples, count + 1, 2))
    out[:, 0, :] = np.asarray(current, dtype=float)[None, :]
    for j in range(1, count + 1):
        out[:, j, :] = phi * out[:, j - 1, :] + scale * innovations[:, j - 1, :]
    return out
