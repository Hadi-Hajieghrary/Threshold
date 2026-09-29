"""Reduced linear model of the taut tow (plan IV.10), independent of Drake.

The closed loop of the production fleet plant is re-derived here in NumPy/SciPy from the
same force laws: planar rigid bodies (load and ``N`` vessels) with world-frame
generalized coordinates ``(x, y, theta)`` and forces ``(f_x, f_y, tau_z)`` about the body
origin; ``N`` Kelvin-Voigt cables held taut and linear (``T = k e + c edot``, no
clipping); linear or quadratic hull drag with linear yaw damping; constant-magnitude
thrust along each vessel's true heading; the heading controller
``yaw = k_h wrap(theta_ref - theta) - k_c sin(bearing)`` with the production run's
pre-loaded reference ``theta_ref`` (the operating heading in the parallel formation); and
the weather force at each body origin.

v2 (plan IV.7, opt-in through ``ReducedModelInput.sway_gain``; 0 reproduces v1): the
controller's sway term, ``theta_ref - k_sigma wrap(sigma - phi)`` with
``sigma = theta - bearing`` the world chord angle and ``phi`` the design cable angle; a
hull-chord misalignment output ``psi = wrap(theta - sigma)``; and the second-order
mean-tension prediction ``mu_q2 = T0 + (T0 + c_A v)/2 E[sigma^2] - (F_T/2) E[psi^2]``
(plan II.3), returned beside the v1 ``mu_q``, which is unchanged.

Declared simplifications: the controller reads the true heading and the true cable
bearing, so sensor noise, gyro drift, the 50 Hz command hold, the 20 Hz bearing sampling,
and the one-tick sensor delay are neglected; the cables never slacken.

The load's absolute position, on which nothing depends, is dropped and each vessel's
position is taken relative to the load's, so the state deviation is measured in a frame
translating with the steady tow.  The weather is a zero-order-hold input driven by its
own AR(1) on the 10 ms grid; the augmented system is discretized exactly over one weather
period and the stationary covariance solves the discrete Lyapunov equation.

Removing the translation symmetry does not make every cell strictly stable: the vessels'
lateral swing about the load (heading following the chord through the stern moment and
the trim term, thrust turning outward) is weakly unstable in part of the Phase 2 grid,
including the nominal cell.  The covariance is therefore computed on the stable
invariant subspace of the discrete transition; outputs that observe an unstable mode
have no stationary variance and are returned as ``inf``.  In the parallel formation the
unstable modes are the lateral swings with zero net force on the load, which the radial
outputs (``e``, ``edot``, ``q``, ``W_rel``) do not observe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm, schur, solve_discrete_lyapunov, solve_sylvester

from tether.physics import constants

PAIR_REDUCED_MASS = (
    constants.VESSEL_MASS * constants.LOAD_MASS / (constants.VESSEL_MASS + constants.LOAD_MASS)
)
STERN_OFFSET = np.array([-1.5, 0.0])
QUADRATIC_DRAG_REFERENCE_SPEED = 0.9
PENTAGON_VERTEX_ANGLES = np.deg2rad(36.0 + 72.0 * np.arange(5))
SAMPLE_PERIOD = constants.WEATHER_PERIOD
AR1_COEFFICIENT = math.exp(-constants.WEATHER_PERIOD / constants.WEATHER_TIME_CONSTANT)
STATE_STEP = 1.0e-5
CHECK_STEP = 1.0e-4
WEATHER_STEP = 1.0
EQUILIBRIUM_TOLERANCE = 1.0e-11
LEAKAGE_TOLERANCE = 1.0e-7
DEGENERATE_RATIO = 1.0e-8
DECORRELATION_LEVEL = 0.05
MAX_LAG_SAMPLES = 100_000


@dataclass(frozen=True)
class ReducedModelInput:
    """One operating cell of the taut tow, in the vocabulary of ``FleetRunSpec``."""

    formation: str = "parallel"
    arc_half_angle: float | None = None
    pretension: float = 990.0
    heading_gain: float = 500.0
    trim_gain: float = 100.0
    drag_law: str = "linear"
    weather_direction: str = "local"
    weather_scale: float = 1.0
    weather_rho: float | None = None
    weather_front_angle: float | None = None
    sway_gain: float = 0.0


@dataclass(frozen=True)
class TowGeometry:
    """Attachment points and equilibrium chord directions of one formation."""

    formation: str
    load_offsets: np.ndarray
    vessel_offsets: np.ndarray
    cable_angles: np.ndarray
    arc_half_angle: float

    @property
    def vessel_count(self) -> int:
        return int(self.load_offsets.shape[0])


@dataclass(frozen=True)
class TowOperatingPoint:
    """Steady tow: speed, per-vessel thrust and heading, per-cable tension."""

    speed: float
    thrusts: np.ndarray
    headings: np.ndarray
    tensions: np.ndarray


@dataclass(frozen=True)
class TowPlant:
    """Force-law parameters of the closed loop about one operating point."""

    geometry: TowGeometry
    operating: TowOperatingPoint
    heading_gain: float
    trim_gain: float
    drag_law: str
    heading_reference: np.ndarray
    rest_length: float = constants.CABLE_REST_LENGTH
    stiffness: float = constants.CABLE_STIFFNESS
    damping: float = constants.CABLE_DAMPING
    sway_gain: float = 0.0
    design_chord_angles: np.ndarray | None = None


@dataclass(frozen=True)
class ReducedModelResult:
    """Stationary statistics of the taut tow under the reduced linear model.

    Per-cable arrays have shape ``(N,)``.  ``outputs[name]`` maps the augmented state
    ``[x; w]`` on the 10 ms grid to the per-cable output ``name`` (``e``, ``edot``,
    ``eddot``, ``q``, ``qdot``, ``w`` for ``W_rel``, ``chord_angle``, ``psi``); ``w`` is the
    weather sample held over the following 10 ms.  ``e`` and ``q`` are deviations from
    ``mu_e`` and ``mu_q``.  ``covariance`` is the stationary covariance of ``[x; w]`` on
    the stable invariant subspace (the full stationary covariance when
    ``unstable_eigenvalues`` is empty); ``leakage[name]`` measures how much output
    ``name`` observes the unstable subspace, and outputs above ``LEAKAGE_TOLERANCE``
    carry ``inf`` standard deviations.  ``degenerate[name]`` flags the cables whose output
    the weather cannot reach (a symmetric formation under a directional front): their
    standard deviation is 0 and their correlations and time scales are ``nan``.
    ``sigma_psi`` is the std of the hull-chord misalignment and ``mu_q2`` the plan's
    second-order mean tension (v2); ``mu_q2_terms`` holds its chord (``sigma``) and
    misalignment (``psi``) terms.
    """

    inp: ReducedModelInput
    geometry: TowGeometry
    operating: TowOperatingPoint
    equilibrium: np.ndarray
    chord_directions: np.ndarray
    mu_e: np.ndarray
    sigma_e: np.ndarray
    sigma_edot: np.ndarray
    sigma_eddot: np.ndarray
    mu_q: np.ndarray
    sigma_q: np.ndarray
    sigma_qdot: np.ndarray
    mu_w: np.ndarray
    sigma_w: np.ndarray
    sigma_chord_angle: np.ndarray
    corr_q: np.ndarray
    corr_w: np.ndarray
    integral_time_scale: np.ndarray
    decorrelation_lag: np.ndarray
    covariance: np.ndarray
    state_matrix: np.ndarray
    input_matrix: np.ndarray
    transition: np.ndarray
    innovation_covariance: np.ndarray
    outputs: dict[str, np.ndarray]
    leakage: dict[str, float]
    degenerate: dict[str, np.ndarray]
    state_labels: tuple[str, ...]
    unstable_eigenvalues: np.ndarray
    equilibrium_residual: float
    linearization_discrepancy: float
    lyapunov_residual: float
    spectral_radius: float
    sigma_psi: np.ndarray | None = None
    mu_q2: np.ndarray | None = None
    mu_q2_terms: dict[str, np.ndarray] | None = None

    def stationary(self, name: str) -> bool:
        return self.leakage[name] <= LEAKAGE_TOLERANCE

    def output_covariance(self, name: str) -> np.ndarray:
        if not self.stationary(name):
            return np.full((self.mu_e.size, self.mu_e.size), np.inf)
        output = self.outputs[name]
        return output @ self.covariance @ output.T

    def correlation(self, name: str) -> np.ndarray:
        """Correlation matrix of output ``name`` across cables."""
        return _correlation(self.output_covariance(name), self.degenerate[name])

    def phase_averaged_covariance(self, name: str, phases: int = 10) -> np.ndarray:
        """Output covariance averaged over ``phases`` equally spaced instants of the hold.

        This is what a log sampled uniformly in time (1 ms for ``phases=10``) estimates;
        the grid-instant values differ from it materially only for the outputs that read
        ``w`` directly (``eddot``, ``qdot``).
        """
        if not self.stationary(name):
            return np.full((self.mu_e.size, self.mu_e.size), np.inf)
        state_size = self.state_matrix.shape[0]
        size = self.covariance.shape[0]
        augmented = np.zeros((size, size))
        augmented[:state_size, :state_size] = self.state_matrix
        augmented[:state_size, state_size:] = self.input_matrix
        step = expm(augmented * SAMPLE_PERIOD / phases)
        output = self.outputs[name]
        total = np.zeros((output.shape[0], output.shape[0]))
        moved = output
        for _ in range(phases):
            total += moved @ self.covariance @ moved.T
            moved = moved @ step
        return total / phases

    def autocorrelation(self, name: str, max_lag: int) -> np.ndarray:
        """Autocorrelation of each cable's output at lags ``0..max_lag`` (10 ms units)."""
        output = self.outputs[name]
        values = np.full((max_lag + 1, output.shape[0]), np.nan)
        if not self.stationary(name):
            return values
        propagated = self.covariance @ output.T
        variance = np.sum(output * propagated.T, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            for lag in range(max_lag + 1):
                values[lag] = np.sum(output * propagated.T, axis=1) / variance
                propagated = self.transition @ propagated
        values[:, self.degenerate[name]] = np.nan
        return values


def pentagon_vertices(circumradius: float = constants.LOAD_RADIUS) -> np.ndarray:
    """Load outline, flat face forward (vertices at 36 + 72k degrees)."""
    return circumradius * np.stack(
        [np.cos(PENTAGON_VERTEX_ANGLES), np.sin(PENTAGON_VERTEX_ANGLES)], axis=1
    )


def ray_polygon_intersection(angle: float, vertices: np.ndarray) -> np.ndarray:
    """Point where a ray from the origin at ``angle`` leaves the polygon."""
    direction = np.array([math.cos(angle), math.sin(angle)])
    best = None
    for index in range(vertices.shape[0]):
        start = vertices[index]
        edge = vertices[(index + 1) % vertices.shape[0]] - start
        matrix = np.column_stack([direction, -edge])
        if abs(np.linalg.det(matrix)) < 1.0e-12:
            continue
        ray_length, edge_fraction = np.linalg.solve(matrix, start)
        if ray_length > 0.0 and -1.0e-12 <= edge_fraction <= 1.0 + 1.0e-12:
            if best is None or ray_length < best:
                best = ray_length
    if best is None:
        raise ValueError("ray does not intersect the polygon")
    return best * direction


def tow_geometry(
    formation: str = "parallel",
    vessel_count: int = constants.VESSEL_COUNT,
    arc_half_angle: float | None = None,
) -> TowGeometry:
    """Attachments on the load's forward boundary by ray-polygon intersection."""
    vertices = pentagon_vertices()
    face_x = constants.LOAD_RADIUS * math.cos(math.radians(36.0))
    if arc_half_angle is None:
        arc_half_angle = math.atan2(2.0, face_x)
    angles = np.linspace(-arc_half_angle, arc_half_angle, vessel_count)
    load_offsets = np.stack([ray_polygon_intersection(a, vertices) for a in angles])
    vessel_offsets = np.tile(STERN_OFFSET, (vessel_count, 1))
    if formation == "parallel":
        cable_angles = np.zeros(vessel_count)
    elif formation == "fan":
        cable_angles = angles.copy()
    else:
        raise ValueError(f"unknown formation kind: {formation}")
    return TowGeometry(formation, load_offsets, vessel_offsets, cable_angles, float(arc_half_angle))


def _drag_force(speed: float, linear_coefficient: float, drag_law: str) -> float:
    if drag_law == "linear":
        return linear_coefficient * speed
    if drag_law == "quadratic":
        return linear_coefficient / QUADRATIC_DRAG_REFERENCE_SPEED * speed * abs(speed)
    raise ValueError(f"unknown drag law: {drag_law}")


def tow_operating_point(
    geometry: TowGeometry, pretension: float, drag_law: str = "linear"
) -> TowOperatingPoint:
    """Steady tow with equal tension ``pretension`` in every cable."""
    cosines = np.cos(geometry.cable_angles)
    sines = np.sin(geometry.cable_angles)
    load_drag = pretension * float(np.sum(cosines))
    if abs(pretension * float(np.sum(sines))) > 1.0e-9 * max(1.0, pretension):
        raise ValueError("formation is not laterally balanced at equal tension")
    if drag_law == "linear":
        speed = load_drag / constants.LOAD_LINEAR_DRAG
    else:
        speed = math.sqrt(load_drag * QUADRATIC_DRAG_REFERENCE_SPEED / constants.LOAD_LINEAR_DRAG)
    vessel_drag = _drag_force(speed, constants.VESSEL_LINEAR_DRAG, drag_law)
    thrust_x = vessel_drag + pretension * cosines
    thrust_y = pretension * sines
    return TowOperatingPoint(
        speed=speed,
        thrusts=np.hypot(thrust_x, thrust_y),
        headings=np.arctan2(thrust_y, thrust_x),
        tensions=np.full(geometry.vessel_count, pretension),
    )


def _rotate(angles: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    cosines = np.cos(angles)
    sines = np.sin(angles)
    return np.stack(
        [cosines * vectors[:, 0] - sines * vectors[:, 1], sines * vectors[:, 0] + cosines * vectors[:, 1]],
        axis=1,
    )


def _perpendicular(vectors: np.ndarray) -> np.ndarray:
    return np.stack([-vectors[:, 1], vectors[:, 0]], axis=1)


def _cross(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def steady_tow_state(
    geometry: TowGeometry,
    operating: TowOperatingPoint,
    rest_length: float = constants.CABLE_REST_LENGTH,
    stiffness: float = constants.CABLE_STIFFNESS,
    load_heading: float = 0.0,
) -> np.ndarray:
    """Plant-layout design state ``[q; v]``: load at the origin, cables at T0/k."""
    n = geometry.vessel_count
    positions = np.zeros((n + 1, 3))
    velocities = np.zeros((n + 1, 3))
    positions[0, 2] = load_heading
    velocities[0, :2] = operating.speed * np.array([math.cos(load_heading), math.sin(load_heading)])
    load_arm = _rotate(np.full(n, load_heading), geometry.load_offsets)
    cable_angles = geometry.cable_angles + load_heading
    chord_length = rest_length + operating.tensions / stiffness
    chord = chord_length[:, None] * np.stack([np.cos(cable_angles), np.sin(cable_angles)], axis=1)
    headings = operating.headings + load_heading
    vessel_arm = _rotate(headings, geometry.vessel_offsets)
    positions[1:, :2] = load_arm + chord - vessel_arm
    positions[1:, 2] = headings
    velocities[1:, :2] = velocities[0, :2]
    return np.concatenate([positions.ravel(), velocities.ravel()])


def tow_plant(inp: ReducedModelInput) -> TowPlant:
    """Geometry, design operating point, gains, and heading reference of one cell."""
    if inp.drag_law not in ("linear", "quadratic"):
        raise ValueError(f"unknown drag law: {inp.drag_law}")
    geometry = tow_geometry(inp.formation, arc_half_angle=inp.arc_half_angle)
    operating = tow_operating_point(geometry, inp.pretension, inp.drag_law)
    plant = TowPlant(geometry, operating, inp.heading_gain, inp.trim_gain, inp.drag_law, operating.headings)
    if inp.sway_gain != 0.0:
        plant = replace(plant, sway_gain=float(inp.sway_gain), design_chord_angles=geometry.cable_angles.copy())
    return replace(plant, heading_reference=tow_heading_reference(plant))


def tow_heading_reference(plant: TowPlant) -> np.ndarray:
    """Production pre-loaded reference: the design tow is a closed-loop equilibrium.

    At the design state ``k_h (theta_ref - theta) - k_c sin(bearing) + M_cable = 0`` with
    ``M_cable`` the stern cable's yaw moment on the vessel; for the parallel formation
    ``M_cable = bearing = 0`` and ``theta_ref`` is the operating heading.
    """
    geometry = plant.geometry
    operating = plant.operating
    if plant.heading_gain == 0.0:
        return operating.headings.copy()
    angle = geometry.cable_angles
    force = -operating.tensions[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    moments = _cross(_rotate(operating.headings, geometry.vessel_offsets), force)
    design = steady_tow_state(geometry, operating, plant.rest_length, plant.stiffness)
    bearings = cable_terms(plant, design)["bearing"]
    return operating.headings + (plant.trim_gain * np.sin(bearings) - moments) / plant.heading_gain


def cable_terms(plant: TowPlant, state: np.ndarray) -> dict[str, np.ndarray]:
    """Arms, chord, unit direction, elongation, rate, and bearing from ``[q; v]``."""
    geometry = plant.geometry
    n = geometry.vessel_count
    positions = state[: 3 * (n + 1)].reshape(n + 1, 3)
    velocities = state[3 * (n + 1) :].reshape(n + 1, 3)
    load_arm = _rotate(np.full(n, positions[0, 2]), geometry.load_offsets)
    vessel_arm = _rotate(positions[1:, 2], geometry.vessel_offsets)
    chord = positions[1:, :2] + vessel_arm - positions[0, :2] - load_arm
    length = np.sqrt(np.sum(chord * chord, axis=1))
    direction = chord / length[:, None]
    load_point_velocity = velocities[0, :2] + velocities[0, 2] * _perpendicular(load_arm)
    vessel_point_velocity = velocities[1:, :2] + velocities[1:, 2:3] * _perpendicular(vessel_arm)
    rate = np.sum((vessel_point_velocity - load_point_velocity) * direction, axis=1)
    headings = positions[1:, 2]
    to_load = -chord
    body_x = np.cos(headings) * to_load[:, 0] + np.sin(headings) * to_load[:, 1]
    body_y = -np.sin(headings) * to_load[:, 0] + np.cos(headings) * to_load[:, 1]
    return {
        "load_arm": load_arm,
        "vessel_arm": vessel_arm,
        "chord": chord,
        "direction": direction,
        "elongation": length - plant.rest_length,
        "rate": rate,
        "bearing": np.arctan2(body_y, -body_x),
        "chord_angle": np.arctan2(chord[:, 1], chord[:, 0]),
    }


def generalized_forces(plant: TowPlant, state: np.ndarray, weather: np.ndarray) -> np.ndarray:
    """World-frame ``(f_x, f_y, tau_z)`` of every body at ``[q; v]``, cables always taut."""
    n = plant.geometry.vessel_count
    velocities = state[3 * (n + 1) :].reshape(n + 1, 3)
    headings = state[5 : 3 * (n + 1) : 3]
    terms = cable_terms(plant, state)
    tension = plant.stiffness * terms["elongation"] + plant.damping * terms["rate"]
    force = tension[:, None] * terms["direction"]
    generalized = np.zeros((n + 1, 3))
    generalized[0, :2] = np.sum(force, axis=0)
    generalized[0, 2] = float(np.sum(_cross(terms["load_arm"], force)))
    generalized[1:, :2] = -force
    generalized[1:, 2] = -_cross(terms["vessel_arm"], force)
    linear = np.array([constants.LOAD_LINEAR_DRAG] + [constants.VESSEL_LINEAR_DRAG] * n)
    angular = np.array([constants.LOAD_ANGULAR_DRAG] + [constants.VESSEL_ANGULAR_DRAG] * n)
    if plant.drag_law == "quadratic":
        coefficient = linear / QUADRATIC_DRAG_REFERENCE_SPEED * np.hypot(velocities[:, 0], velocities[:, 1])
    else:
        coefficient = linear
    generalized[:, :2] -= coefficient[:, None] * velocities[:, :2]
    generalized[:, 2] -= angular * velocities[:, 2]
    thrust = plant.operating.thrusts
    generalized[1:, 0] += thrust * np.cos(headings)
    generalized[1:, 1] += thrust * np.sin(headings)
    reference = plant.heading_reference
    if plant.sway_gain != 0.0:
        # Sway term: sigma = theta - bearing is the world chord angle (see the controller).
        reference = reference - plant.sway_gain * wrap_angle(headings - terms["bearing"] - plant.design_chord_angles)
    generalized[1:, 2] += plant.heading_gain * wrap_angle(
        reference - headings
    ) - plant.trim_gain * np.sin(terms["bearing"])
    generalized[:, :2] += np.asarray(weather, dtype=float).reshape(n + 1, 2)
    return generalized.ravel()


def to_reduced(state: np.ndarray, vessel_count: int) -> np.ndarray:
    """``[q; v]`` to ``[theta_0, (x_i - x_0, y_i - y_0, theta_i)_i, v]``."""
    positions = state[: 3 * (vessel_count + 1)].reshape(vessel_count + 1, 3)
    relative = positions[1:].copy()
    relative[:, :2] -= positions[0, :2]
    return np.concatenate([positions[0, 2:3], relative.ravel(), state[3 * (vessel_count + 1) :]])


def to_full(reduced: np.ndarray, vessel_count: int, load_position=(0.0, 0.0)) -> np.ndarray:
    """Inverse of ``to_reduced`` with the load placed at ``load_position``."""
    load = np.asarray(load_position, dtype=float)
    relative = reduced[1 : 1 + 3 * vessel_count].reshape(vessel_count, 3).copy()
    relative[:, :2] += load
    positions = np.concatenate([[load[0], load[1], reduced[0]], relative.ravel()])
    return np.concatenate([positions, reduced[1 + 3 * vessel_count :]])


def generalized_inertia(vessel_count: int) -> np.ndarray:
    load = [constants.LOAD_MASS, constants.LOAD_MASS, constants.LOAD_YAW_INERTIA]
    vessel = [constants.VESSEL_MASS, constants.VESSEL_MASS, constants.VESSEL_YAW_INERTIA]
    return np.array(load + vessel * vessel_count)


def reduced_vector_field(plant: TowPlant, reduced: np.ndarray, weather: np.ndarray) -> np.ndarray:
    """Time derivative of the reduced state under weather ``w`` (N + 1 planar forces)."""
    n = plant.geometry.vessel_count
    force = generalized_forces(plant, to_full(reduced, n), weather)
    velocities = reduced[1 + 3 * n :].reshape(n + 1, 3)
    kinematic = velocities[1:].copy()
    kinematic[:, :2] -= velocities[0, :2]
    return np.concatenate([velocities[0, 2:3], kinematic.ravel(), force / generalized_inertia(n)])


def _jacobian(function, point: np.ndarray, step: float) -> np.ndarray:
    columns = []
    for index in range(point.size):
        delta = np.zeros(point.size)
        delta[index] = step
        columns.append((function(point + delta) - function(point - delta)) / (2.0 * step))
    return np.column_stack(columns)


def _newton(residual, unknown: np.ndarray, iterations: int = 25) -> np.ndarray | None:
    for _ in range(iterations):
        value = residual(unknown)
        if not np.all(np.isfinite(value)):
            return None
        if np.max(np.abs(value)) < EQUILIBRIUM_TOLERANCE:
            return unknown
        unknown = unknown - np.linalg.solve(_jacobian(residual, unknown, STATE_STEP), value)
    return None


def tow_equilibrium(plant: TowPlant) -> np.ndarray:
    """Reduced state of the exact closed-loop steady tow.

    Unknowns are the load heading, the vessel poses relative to the load, and the common
    tow velocity; every body translates at that velocity without rotating.  With the
    production heading reference the design state of ``steady_tow_state`` is exact; for
    any residual left there the root is followed by Newton homotopy from the design
    state, ``r(u) - (1 - s) r(u_design)``, s: 0 -> 1.
    """
    n = plant.geometry.vessel_count
    design = steady_tow_state(plant.geometry, plant.operating, plant.rest_length, plant.stiffness)
    guess = to_reduced(design, n)
    pose_size = 1 + 3 * n
    zero_weather = np.zeros(2 * (n + 1))

    def assemble(unknown: np.ndarray) -> np.ndarray:
        velocities = np.zeros((n + 1, 3))
        velocities[:, :2] = unknown[pose_size:]
        return np.concatenate([unknown[:pose_size], velocities.ravel()])

    def residual(unknown: np.ndarray) -> np.ndarray:
        return reduced_vector_field(plant, assemble(unknown), zero_weather)[pose_size:]

    unknown = np.concatenate([guess[:pose_size], guess[pose_size : pose_size + 2]])
    offset = residual(unknown)
    progress, increment = 0.0, 1.0
    while progress < 1.0:
        target = min(1.0, progress + increment)
        solved = _newton(lambda u, s=target: residual(u) - (1.0 - s) * offset, unknown)
        if solved is None:
            increment *= 0.5
            if increment < 1.0e-4:
                raise RuntimeError(f"steady-tow branch lost at homotopy parameter {progress:.4f}")
            continue
        unknown, progress = solved, target
        increment = min(2.0 * increment, 1.0)
    state = assemble(unknown)
    reference = plant.operating.headings
    state[3:pose_size:3] = reference + wrap_angle(state[3:pose_size:3] - reference)
    terms = cable_terms(plant, to_full(state, n))
    tension = plant.stiffness * terms["elongation"] + plant.damping * terms["rate"]
    ahead = np.cos(terms["chord_angle"] - plant.geometry.cable_angles)
    if np.any(tension <= 0.0) or np.any(ahead <= 0.0):
        raise RuntimeError("the closed-loop steady tow on the homotopy branch is not a taut tow")
    return state


def weather_covariance(
    vessel_count: int,
    direction: str,
    scale: float,
    rho: float | None = None,
    front_angle: float | None = None,
) -> np.ndarray:
    """Stationary covariance of the weather vector ``(w_0x, w_0y, w_1x, ...)``.

    ``front`` is the common gust along ``front_angle`` with the variance of the isotropic
    classes, all of it on that axis.
    """
    std = scale * np.array([constants.LOAD_WEATHER_STD] + [constants.VESSEL_WEATHER_STD] * vessel_count)
    if direction == "front":
        if front_angle is None:
            raise ValueError("front weather requires front_angle")
        axis = np.array([math.cos(front_angle), math.sin(front_angle)])
        return np.kron(np.outer(std, std), 2.0 * np.outer(axis, axis))
    if direction == "local":
        share = 0.0
    elif direction == "common":
        share = 1.0
    elif direction == "mixed":
        if rho is None or not 0.0 <= rho <= 1.0:
            raise ValueError("mixed weather requires rho in [0, 1]")
        share = float(rho)
    else:
        raise ValueError(f"unknown direction structure: {direction}")
    body = np.outer(std, std) * (share + (1.0 - share) * np.eye(vessel_count + 1))
    return np.kron(body, np.eye(2))


def relative_load_map(directions: np.ndarray) -> np.ndarray:
    """Rows mapping the weather vector to ``W_rel,i`` along fixed chord directions."""
    n = directions.shape[0]
    rows = np.zeros((n, 2 * (n + 1)))
    for i in range(n):
        rows[i, 0:2] = PAIR_REDUCED_MASS * directions[i] / constants.LOAD_MASS
        rows[i, 2 * (i + 1) : 2 * (i + 2)] = -PAIR_REDUCED_MASS * directions[i] / constants.VESSEL_MASS
    return rows


def zoh_transition(state_matrix: np.ndarray, input_matrix: np.ndarray) -> np.ndarray:
    """Exact 10 ms transition of ``[x; w]`` with ``w`` held, then ``w <- phi w``."""
    state_size = state_matrix.shape[0]
    size = state_size + input_matrix.shape[1]
    augmented = np.zeros((size, size))
    augmented[:state_size, :state_size] = state_matrix
    augmented[:state_size, state_size:] = input_matrix
    transition = expm(augmented * SAMPLE_PERIOD)
    transition[state_size:, :state_size] = 0.0
    transition[state_size:, state_size:] = AR1_COEFFICIENT * np.eye(size - state_size)
    return transition


def stable_covariance(
    transition: np.ndarray, innovation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Stationary covariance on the stable invariant subspace of ``transition``.

    Returns the covariance, a unit-column basis of the unstable invariant subspace, and
    the residual of the projected Lyapunov equation in correlation units.
    """
    schur_form, basis, stable = schur(transition, output="real", sort="iuc")
    block = schur_form[:stable, :stable]
    coupling = solve_sylvester(block, -schur_form[stable:, stable:], -schur_form[:stable, stable:])
    left = basis[:, :stable].T - coupling @ basis[:, stable:].T
    right = basis[:, :stable]
    unstable = basis[:, :stable] @ coupling + basis[:, stable:]
    if unstable.shape[1]:
        unstable = unstable / np.linalg.norm(unstable, axis=0)
    projected = left @ innovation @ left.T
    reduced = solve_discrete_lyapunov(block, 0.5 * (projected + projected.T))
    covariance = right @ (0.5 * (reduced + reduced.T)) @ right.T
    projector = right @ left
    residual = transition @ covariance @ transition.T + projector @ innovation @ projector.T - covariance
    return covariance, unstable, correlation_residual(residual, covariance)


def correlation_residual(residual: np.ndarray, covariance: np.ndarray) -> float:
    """Largest ``|R_ij| / sqrt(P_ii P_jj)``, so that no block's scale hides its error."""
    variance = np.diag(covariance)
    floor = np.finfo(float).eps * max(float(np.max(variance)), np.finfo(float).tiny)
    scale = np.sqrt(np.maximum(variance, floor))
    return float(np.max(np.abs(residual) / np.outer(scale, scale)))


def _correlation(covariance: np.ndarray, degenerate: np.ndarray | None = None) -> np.ndarray:
    std = np.sqrt(np.diag(covariance))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / np.outer(std, std)
    if degenerate is not None:
        correlation[degenerate, :] = np.nan
        correlation[:, degenerate] = np.nan
    return correlation


def _state_labels(vessel_count: int) -> tuple[str, ...]:
    labels = ["theta_0"]
    for i in range(1, vessel_count + 1):
        labels += [f"x_{i}-x_0", f"y_{i}-y_0", f"theta_{i}"]
    for j in range(vessel_count + 1):
        labels += [f"vx_{j}", f"vy_{j}", f"omega_{j}"]
    for j in range(vessel_count + 1):
        labels += [f"wx_{j}", f"wy_{j}"]
    return tuple(labels)


def _leakage(output: np.ndarray, unstable: np.ndarray) -> float:
    if unstable.shape[1] == 0:
        return 0.0
    norms = np.linalg.norm(output, axis=1)
    return float(np.max(np.linalg.norm(output @ unstable, axis=1) / np.where(norms > 0.0, norms, 1.0)))


def reduced_model(inp: ReducedModelInput) -> ReducedModelResult:
    """Linearize the taut tow, discretize with the ZOH weather, solve the Lyapunov equation."""
    if not inp.weather_scale > 0.0:
        raise ValueError("weather_scale must be positive")
    plant = tow_plant(inp)
    n = plant.geometry.vessel_count
    state_size = 6 * n + 4
    weather_size = 2 * (n + 1)
    zero_weather = np.zeros(weather_size)
    equilibrium = tow_equilibrium(plant)
    residual = float(np.max(np.abs(reduced_vector_field(plant, equilibrium, zero_weather))))

    def field(reduced: np.ndarray) -> np.ndarray:
        return reduced_vector_field(plant, reduced, zero_weather)

    state_matrix = _jacobian(field, equilibrium, STATE_STEP)
    check = _jacobian(field, equilibrium, CHECK_STEP)
    discrepancy = float(np.max(np.abs(state_matrix - check)) / np.max(np.abs(state_matrix)))
    input_matrix = _jacobian(lambda w: reduced_vector_field(plant, equilibrium, w), zero_weather, WEATHER_STEP)

    def terms(name: str):
        return lambda reduced: cable_terms(plant, to_full(reduced, n))[name]

    psi_at_rest = wrap_angle(equilibrium[3 : 1 + 3 * n : 3] - cable_terms(plant, to_full(equilibrium, n))["chord_angle"])

    def misalignment(reduced: np.ndarray) -> np.ndarray:
        # psi = theta - sigma, unwrapped about the equilibrium value.
        full_terms = cable_terms(plant, to_full(reduced, n))
        return psi_at_rest + wrap_angle(reduced[3 : 1 + 3 * n : 3] - full_terms["chord_angle"] - psi_at_rest)

    at_rest = cable_terms(plant, to_full(equilibrium, n))
    directions = at_rest["direction"]
    tensions = plant.stiffness * at_rest["elongation"] + plant.damping * at_rest["rate"]
    output_edot = _jacobian(terms("rate"), equilibrium, STATE_STEP)
    pad = np.zeros((n, weather_size))
    outputs = {
        "e": np.hstack([_jacobian(terms("elongation"), equilibrium, STATE_STEP), pad]),
        "edot": np.hstack([output_edot, pad]),
        "eddot": np.hstack([output_edot @ state_matrix, output_edot @ input_matrix]),
        "w": np.hstack([np.zeros((n, state_size)), relative_load_map(directions)]),
        "chord_angle": np.hstack([_jacobian(terms("chord_angle"), equilibrium, STATE_STEP), pad]),
        "psi": np.hstack([_jacobian(misalignment, equilibrium, STATE_STEP), pad]),
    }
    outputs["q"] = plant.stiffness * outputs["e"] + plant.damping * outputs["edot"]
    outputs["qdot"] = plant.stiffness * outputs["edot"] + plant.damping * outputs["eddot"]

    transition = zoh_transition(state_matrix, input_matrix)
    weather = weather_covariance(
        n, inp.weather_direction, inp.weather_scale, inp.weather_rho, inp.weather_front_angle
    )
    innovation = np.zeros((state_size + weather_size,) * 2)
    innovation[state_size:, state_size:] = (1.0 - AR1_COEFFICIENT**2) * weather
    covariance, unstable, lyapunov_residual = stable_covariance(transition, innovation)
    leakage = {name: _leakage(output, unstable) for name, output in outputs.items()}
    reference = covariance
    if inp.weather_direction != "local":
        isotropic = np.zeros_like(innovation)
        isotropic[state_size:, state_size:] = (1.0 - AR1_COEFFICIENT**2) * weather_covariance(
            n, "local", inp.weather_scale
        )
        reference = stable_covariance(transition, isotropic)[0]
    degenerate = {
        name: np.diag(output @ covariance @ output.T)
        <= DEGENERATE_RATIO**2 * np.diag(output @ reference @ output.T)
        for name, output in outputs.items()
    }

    def sigma(name: str) -> np.ndarray:
        if leakage[name] > LEAKAGE_TOLERANCE:
            return np.full(n, np.inf)
        variance = np.diag(outputs[name] @ covariance @ outputs[name].T)
        return np.where(degenerate[name], 0.0, np.sqrt(np.maximum(variance, 0.0)))

    output_q = outputs["q"]
    integral_time_scale = np.full(n, np.inf)
    decorrelation_lag = np.full(n, np.inf)
    corr_q = np.full((n, n), np.nan)
    if leakage["q"] <= LEAKAGE_TOLERANCE:
        live = ~degenerate["q"]
        q_covariance = output_q @ covariance @ output_q.T
        q_variance = np.where(live, np.diag(q_covariance), 1.0)
        corr_q = _correlation(q_covariance, degenerate["q"])
        summed = output_q @ np.linalg.solve(np.eye(transition.shape[0]) - transition, covariance @ output_q.T)
        integral_time_scale = np.where(live, SAMPLE_PERIOD * (np.diag(summed) / q_variance - 0.5), np.nan)
        decorrelation_lag[~live] = np.nan
        propagated = covariance @ output_q.T
        for lag in range(1, MAX_LAG_SAMPLES + 1):
            if not np.any(np.isinf(decorrelation_lag)):
                break
            propagated = transition @ propagated
            below = np.sum(output_q * propagated.T, axis=1) / q_variance < DECORRELATION_LEVEL
            decorrelation_lag[below & np.isinf(decorrelation_lag)] = lag * SAMPLE_PERIOD

    sigma_chord = sigma("chord_angle")
    sigma_psi = sigma("psi")
    vessel_drag = _drag_force(plant.operating.speed, constants.VESSEL_LINEAR_DRAG, plant.drag_law)
    mu_q2_terms = {
        "sigma": 0.5 * (tensions + vessel_drag) * sigma_chord**2,
        "psi": -0.5 * plant.operating.thrusts * sigma_psi**2,
    }
    with np.errstate(invalid="ignore"):
        mu_q2 = tensions + mu_q2_terms["sigma"] + mu_q2_terms["psi"]

    eigenvalues = np.linalg.eigvals(state_matrix)
    moduli = np.sort(np.abs(np.linalg.eigvals(transition)))
    return ReducedModelResult(
        inp=inp,
        geometry=plant.geometry,
        operating=plant.operating,
        equilibrium=equilibrium,
        chord_directions=directions,
        mu_e=tensions / plant.stiffness,
        sigma_e=sigma("e"),
        sigma_edot=sigma("edot"),
        sigma_eddot=sigma("eddot"),
        mu_q=tensions,
        sigma_q=sigma("q"),
        sigma_qdot=sigma("qdot"),
        mu_w=np.zeros(n),
        sigma_w=sigma("w"),
        sigma_chord_angle=sigma_chord,
        corr_q=corr_q,
        corr_w=_correlation(outputs["w"] @ covariance @ outputs["w"].T, degenerate["w"]),
        integral_time_scale=integral_time_scale,
        decorrelation_lag=decorrelation_lag,
        covariance=covariance,
        state_matrix=state_matrix,
        input_matrix=input_matrix,
        transition=transition,
        innovation_covariance=innovation,
        outputs=outputs,
        leakage=leakage,
        degenerate=degenerate,
        state_labels=_state_labels(n),
        unstable_eigenvalues=np.sort_complex(eigenvalues[eigenvalues.real > 0.0]),
        equilibrium_residual=residual,
        linearization_discrepancy=discrepancy,
        lyapunov_residual=lyapunov_residual,
        spectral_radius=float(moduli[moduli.size - unstable.shape[1] - 1]),
        sigma_psi=sigma_psi,
        mu_q2=mu_q2,
        mu_q2_terms=mu_q2_terms,
    )
