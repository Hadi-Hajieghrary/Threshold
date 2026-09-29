"""Production fleet plant for Phase 1(d) onward.

One discrete Drake ``MultibodyPlant`` (SAP, 0.5 ms step; see ``PHYSICS_STEP`` below for
the P1-T9R/P1-T5R basis) carries
the load and ``N`` vessels on world ``PlanarJoint``s.  All forces enter through the
plant's applied *generalized* force port: for a planar joint whose parent is the world,
the generalized force of a body is exactly its world-frame ``(f_x, f_y, tau_z)`` about the
body origin, so this is the same force path as a list of spatial forces at the body
origin, without constructing Python force objects on every physics step.

Two plant-side leaf systems produce the forces and are summed by a Drake ``Adder``:

* ``UnilateralCableBank`` owns all ``N`` unilateral Kelvin-Voigt cables (plan IV.4), their
  alive flags, the 1 ms event trackers, and the 1 ms logs.
* ``HullForces`` applies drag (linear or quadratic), each vessel's commanded surge force
  along its true hull axis and commanded yaw moment (the thrusters of IV.7), and the
  10 ms zero-order-hold weather forces (IV.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from pydrake.multibody.plant import DiscreteContactApproximation, MultibodyPlant
from pydrake.multibody.tree import PlanarJoint
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.systems.primitives import Adder

from tether.physics import constants
from tether.physics.cable import CableEventTracker, CableMode
from tether.physics.plant import _spatial_inertia

# P1-T9R: the 0.5 ms SAP step meets the unchanged 0.5% physical-energy criterion
# (0.471%, a deterministic bound identical at every impact speed) and changes engagement
# peaks by 0.08% relative to 0.25 ms (P1-T5R); it is the production step from the
# Phase 1 closure on, for the compute budget of IV.14.
PHYSICS_STEP = 5.0e-4
EVENT_PERIOD = 1.0e-3
WEATHER_PERIOD = constants.WEATHER_PERIOD
PAIR_REDUCED_MASS = (
    constants.VESSEL_MASS * constants.LOAD_MASS / (constants.VESSEL_MASS + constants.LOAD_MASS)
)
VESSEL_LENGTH = 3.0
VESSEL_BEAM = 1.0
STERN_OFFSET = np.array([-0.5 * VESSEL_LENGTH, 0.0])
QUADRATIC_DRAG_REFERENCE_SPEED = 0.9
# Attachment separation at which the formation has closed: the vessel's stern is within
# one hull beam of the load attachment.  No contact is modelled, so runs stop there.
CLOSURE_CHORD_LENGTH = 1.0
PENTAGON_VERTEX_ANGLES = np.deg2rad(36.0 + 72.0 * np.arange(5))


def pentagon_vertices(circumradius: float = constants.LOAD_RADIUS) -> np.ndarray:
    """Return the load outline, flat face forward (vertices at 36 + 72k degrees)."""
    return circumradius * np.stack(
        [np.cos(PENTAGON_VERTEX_ANGLES), np.sin(PENTAGON_VERTEX_ANGLES)], axis=1
    )


def ray_polygon_intersection(angle: float, vertices: np.ndarray) -> np.ndarray:
    """Return the point where a ray from the origin at ``angle`` leaves the polygon."""
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


@dataclass(frozen=True)
class FleetGeometry:
    """Attachment points and equilibrium cable directions of one formation."""

    name: str
    load_offsets: np.ndarray  # (N, 2), load frame
    vessel_offsets: np.ndarray  # (N, 2), vessel frame (stern points)
    cable_angles: np.ndarray  # (N,), equilibrium chord directions in the world frame
    arc_half_angle: float

    @property
    def vessel_count(self) -> int:
        return int(self.load_offsets.shape[0])


def formation_geometry(
    kind: str = "parallel",
    vessel_count: int = constants.VESSEL_COUNT,
    arc_half_angle: float | None = None,
) -> FleetGeometry:
    """Place attachments on the load's forward boundary by ray-polygon intersection.

    ``parallel``: cables along the tow axis (Phases 2-3).  The default half-angle puts
    the outer attachments at y = +/-2 m, matching the Phase 0/1 lateral span.
    ``fan``: each cable continues along its attachment ray (Phase 4).
    """
    vertices = pentagon_vertices()
    face_x = constants.LOAD_RADIUS * math.cos(math.radians(36.0))
    if arc_half_angle is None:
        arc_half_angle = math.atan2(2.0, face_x)
    angles = np.linspace(-arc_half_angle, arc_half_angle, vessel_count)
    load_offsets = np.stack([ray_polygon_intersection(a, vertices) for a in angles])
    vessel_offsets = np.tile(STERN_OFFSET, (vessel_count, 1))
    if kind == "parallel":
        cable_angles = np.zeros(vessel_count)
    elif kind == "fan":
        cable_angles = angles.copy()
    else:
        raise ValueError(f"unknown formation kind: {kind}")
    return FleetGeometry(kind, load_offsets, vessel_offsets, cable_angles, float(arc_half_angle))


@dataclass(frozen=True)
class OperatingPoint:
    """Steady tow: speed, per-vessel thrust and heading, per-cable tension."""

    speed: float
    thrusts: np.ndarray
    headings: np.ndarray
    tensions: np.ndarray


def drag_force(speed: float, linear_coefficient: float, drag_law: str) -> float:
    if drag_law == "linear":
        return linear_coefficient * speed
    if drag_law == "quadratic":
        return linear_coefficient / QUADRATIC_DRAG_REFERENCE_SPEED * speed * abs(speed)
    raise ValueError(f"unknown drag law: {drag_law}")


def operating_point(
    geometry: FleetGeometry, pretension: float, drag_law: str = "linear",
    body: "BodyParameters | None" = None,
) -> OperatingPoint:
    """Steady tow with equal tension ``pretension`` in every cable.

    Load balance along the tow axis fixes the speed; each vessel's thrust vector then
    balances its drag and its cable, which fixes its magnitude and hull heading.
    """
    body = body or BodyParameters()
    cosines = np.cos(geometry.cable_angles)
    sines = np.sin(geometry.cable_angles)
    load_drag = pretension * float(np.sum(cosines))
    if abs(pretension * float(np.sum(sines))) > 1.0e-9 * max(1.0, pretension):
        raise ValueError("formation is not laterally balanced at equal tension")
    if drag_law == "linear":
        speed = load_drag / body.load_linear_drag
    else:
        speed = math.sqrt(
            load_drag * QUADRATIC_DRAG_REFERENCE_SPEED / body.load_linear_drag
        )
    vessel_drag = drag_force(speed, body.vessel_linear_drag, drag_law)
    thrust_x = vessel_drag + pretension * cosines
    thrust_y = pretension * sines
    return OperatingPoint(
        speed=speed,
        thrusts=np.hypot(thrust_x, thrust_y),
        headings=np.arctan2(thrust_y, thrust_x),
        tensions=np.full(geometry.vessel_count, pretension),
    )


@dataclass(frozen=True)
class CableParameters:
    rest_length: float = constants.CABLE_REST_LENGTH
    stiffness: float = constants.CABLE_STIFFNESS
    damping: float = constants.CABLE_DAMPING
    mode: str = "recording"
    break_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("recording", "live"):
            raise ValueError("cable mode must be recording or live")
        if self.mode == "live" and self.break_threshold is None:
            raise ValueError("live mode requires a break threshold")


@dataclass(frozen=True)
class BodyParameters:
    """Masses and linear drag of the bodies; the plant constants unless a run overrides them.

    Added for the ACC 2027 confirmatory runs (the fixed-gamma stiffness pair and the
    1250 kg vessel-mass test).  Yaw inertias stay the plant constants.  A ``None`` field on
    ``FleetRunSpec`` maps to the constant here, so every earlier run keeps its meaning and
    its configuration hash.
    """

    load_mass: float = constants.LOAD_MASS
    vessel_mass: float = constants.VESSEL_MASS
    load_linear_drag: float = constants.LOAD_LINEAR_DRAG
    vessel_linear_drag: float = constants.VESSEL_LINEAR_DRAG

    @property
    def pair_reduced_mass(self) -> float:
        return self.vessel_mass * self.load_mass / (self.vessel_mass + self.load_mass)


def _rotate(angles: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    cosines = np.cos(angles)
    sines = np.sin(angles)
    return np.stack(
        [cosines * vectors[:, 0] - sines * vectors[:, 1], sines * vectors[:, 0] + cosines * vectors[:, 1]],
        axis=1,
    )


def cable_kinematics(
    state: np.ndarray, geometry: FleetGeometry, rest_length: float
) -> dict[str, np.ndarray]:
    """Attachment kinematics of every cable from the plant state ``[q; v]``.

    Attachment-point velocities come from the body twists (v + omega x R r), never from
    differencing positions (Appendix B).
    """
    body_count = geometry.vessel_count + 1
    positions = state[: 3 * body_count].reshape(body_count, 3)
    velocities = state[3 * body_count :].reshape(body_count, 3)
    load_arm = _rotate(np.full(geometry.vessel_count, positions[0, 2]), geometry.load_offsets)
    vessel_arm = _rotate(positions[1:, 2], geometry.vessel_offsets)
    load_point = positions[0, :2] + load_arm
    vessel_point = positions[1:, :2] + vessel_arm
    load_point_velocity = velocities[0, :2] + velocities[0, 2] * np.stack(
        [-load_arm[:, 1], load_arm[:, 0]], axis=1
    )
    vessel_point_velocity = velocities[1:, :2] + velocities[1:, 2:3] * np.stack(
        [-vessel_arm[:, 1], vessel_arm[:, 0]], axis=1
    )
    chord = vessel_point - load_point
    length = np.sqrt(np.sum(chord * chord, axis=1))
    direction = chord / length[:, None]
    rate = np.sum((vessel_point_velocity - load_point_velocity) * direction, axis=1)
    return {
        "load_arm": load_arm,
        "vessel_arm": vessel_arm,
        "load_point": load_point,
        "vessel_point": vessel_point,
        "direction": direction,
        "length": length,
        "elongation": length - rest_length,
        "rate": rate,
    }


def fast_cable_terms(state: np.ndarray, geometry: FleetGeometry, rest_length: float):
    """Hot-path form of ``cable_kinematics``: arms, unit chord, elongation, rate.

    Returns ``(lax, lay, vax, vay, ux, uy, elongation, rate)``; identical arithmetic to
    ``cable_kinematics`` without building intermediate stacked arrays.
    """
    n = geometry.vessel_count
    base = 3 * (n + 1)
    cosine0 = math.cos(state[2])
    sine0 = math.sin(state[2])
    ox = geometry.load_offsets[:, 0]
    oy = geometry.load_offsets[:, 1]
    lax = cosine0 * ox - sine0 * oy
    lay = sine0 * ox + cosine0 * oy
    heading = state[5:base:3]
    cosine = np.cos(heading)
    sine = np.sin(heading)
    sx = geometry.vessel_offsets[:, 0]
    sy = geometry.vessel_offsets[:, 1]
    vax = cosine * sx - sine * sy
    vay = sine * sx + cosine * sy
    dx = state[3:base:3] + vax - state[0] - lax
    dy = state[4:base:3] + vay - state[1] - lay
    omega = state[base + 5 :: 3]
    omega0 = state[base + 2]
    rvx = state[base + 3 :: 3] - omega * vay - state[base] + omega0 * lay
    rvy = state[base + 4 :: 3] + omega * vax - state[base + 1] - omega0 * lax
    length = np.sqrt(dx * dx + dy * dy)
    ux = dx / length
    uy = dy / length
    return lax, lay, vax, vay, ux, uy, length - rest_length, rvx * ux + rvy * uy


def fast_cable_forces(terms, tensions: np.ndarray, vessel_count: int) -> np.ndarray:
    lax, lay, vax, vay, ux, uy = terms[:6]
    fx = tensions * ux
    fy = tensions * uy
    generalized = np.zeros(3 * (vessel_count + 1))
    generalized[0] = fx.sum()
    generalized[1] = fy.sum()
    generalized[2] = float(np.dot(lax, fy) - np.dot(lay, fx))
    generalized[3::3] = -fx
    generalized[4::3] = -fy
    generalized[5::3] = -(vax * fy - vay * fx)
    return generalized


def scalar_cable_step(
    state: list[float],
    geometry_lists: tuple[list[float], list[float], list[float], list[float]],
    rest_length: float,
    stiffness: float,
    damping: float,
    alive: list[float],
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Per-physics-step cable forces in scalar Python (the hot path).

    Same arithmetic as ``fast_cable_terms`` + ``fast_cable_forces``; scalar code avoids
    NumPy call overhead on five-element arrays.  Returns the generalized force list and
    the per-cable elongation, rate, and unit chord components.
    """
    ox, oy, sx, sy = geometry_lists
    n = len(ox)
    base = 3 * (n + 1)
    theta0 = state[2]
    cosine0 = math.cos(theta0)
    sine0 = math.sin(theta0)
    px0 = state[0]
    py0 = state[1]
    vx0 = state[base]
    vy0 = state[base + 1]
    omega0 = state[base + 2]
    generalized = [0.0] * base
    elongations = [0.0] * n
    rates = [0.0] * n
    uxs = [0.0] * n
    uys = [0.0] * n
    gx = gy = gt = 0.0
    for i in range(n):
        lax = cosine0 * ox[i] - sine0 * oy[i]
        lay = sine0 * ox[i] + cosine0 * oy[i]
        j = 3 * (i + 1)
        theta = state[j + 2]
        cosine = math.cos(theta)
        sine = math.sin(theta)
        vax = cosine * sx[i] - sine * sy[i]
        vay = sine * sx[i] + cosine * sy[i]
        dx = state[j] + vax - px0 - lax
        dy = state[j + 1] + vay - py0 - lay
        omega = state[base + j + 2]
        rvx = state[base + j] - omega * vay - vx0 + omega0 * lay
        rvy = state[base + j + 1] + omega * vax - vy0 - omega0 * lax
        length = math.sqrt(dx * dx + dy * dy)
        ux = dx / length
        uy = dy / length
        elongation = length - rest_length
        rate = rvx * ux + rvy * uy
        elongations[i] = elongation
        rates[i] = rate
        uxs[i] = ux
        uys[i] = uy
        tension = 0.0
        if elongation > 0.0 and alive[i] > 0.5:
            tension = stiffness * elongation + damping * rate
            if tension < 0.0:
                tension = 0.0
        fx = tension * ux
        fy = tension * uy
        gx += fx
        gy += fy
        gt += lax * fy - lay * fx
        generalized[j] = -fx
        generalized[j + 1] = -fy
        generalized[j + 2] = -(vax * fy - vay * fx)
    generalized[0] = gx
    generalized[1] = gy
    generalized[2] = gt
    return generalized, elongations, rates, uxs, uys


def cable_generalized_forces(
    kinematics: dict[str, np.ndarray], tensions: np.ndarray, vessel_count: int
) -> np.ndarray:
    """Generalized forces of tensions ``T`` applied +T d on the load and -T d on vessels."""
    force = tensions[:, None] * kinematics["direction"]
    generalized = np.zeros((vessel_count + 1, 3))
    load_arm = kinematics["load_arm"]
    vessel_arm = kinematics["vessel_arm"]
    generalized[0, :2] = np.sum(force, axis=0)
    generalized[0, 2] = float(np.sum(load_arm[:, 0] * force[:, 1] - load_arm[:, 1] * force[:, 0]))
    generalized[1:, :2] = -force
    generalized[1:, 2] = -(vessel_arm[:, 0] * force[:, 1] - vessel_arm[:, 1] * force[:, 0])
    return generalized.ravel()


def relative_gust_load(
    weather: np.ndarray, direction: np.ndarray, body: "BodyParameters | None" = None
) -> np.ndarray:
    """Differential radial gust load W_rel (II.1), positive when it closes the gap."""
    body = body or BodyParameters()
    load_acceleration = weather[0] / body.load_mass
    vessel_acceleration = weather[1:] / body.vessel_mass
    return body.pair_reduced_mass * np.sum((load_acceleration[None, :] - vessel_acceleration) * direction, axis=1)


@dataclass
class FleetLog:
    """Per-run 1 ms cable logs and 10 ms body-state logs held on the cable bank."""

    event_time: np.ndarray
    elongation: np.ndarray
    rate: np.ndarray
    relative_load: np.ndarray
    alive: np.ndarray
    state_time: np.ndarray
    state: np.ndarray
    count: int = 0
    state_count: int = 0


class UnilateralCableBank(LeafSystem):
    """All N unilateral cables: explicit spring-damper forces, alive flags, events.

    Plant-side.  The 1 ms periodic discrete update samples every cable, advances its
    ``CableEventTracker`` (held on the system and mutated in place, so no per-update
    deep copy), severs cables in live mode, and appends to the logs.
    """

    PLANT_SIDE = True

    def __init__(
        self,
        geometry: FleetGeometry,
        cable: CableParameters,
        weather: np.ndarray | None,
        scripted_force: Callable[[float], np.ndarray] | None,
        log_duration: float,
        log_state: bool = True,
        body: "BodyParameters | None" = None,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.cable = cable
        self.body = body or BodyParameters()
        self._weather = weather
        self._scripted_force = scripted_force
        n = geometry.vessel_count
        self._state_port = self.DeclareVectorInputPort("plant_state", 6 * (n + 1))
        self._alive_index = self.DeclareDiscreteState(np.ones(n))
        self.DeclareVectorOutputPort(
            "generalized_force", 3 * (n + 1), self._calc_force, prerequisites_of_calc={self.all_input_ports_ticket(), self.xd_ticket()}
        )
        self.DeclareVectorOutputPort(
            "tension", n, self._calc_tension, prerequisites_of_calc={self.all_input_ports_ticket(), self.xd_ticket()}
        )
        self.DeclarePeriodicDiscreteUpdateEvent(EVENT_PERIOD, 0.0, self._update)
        mode = CableMode.LIVE if cable.mode == "live" else CableMode.RECORDING
        self.trackers = [
            CableEventTracker(
                stiffness=cable.stiffness,
                damping=cable.damping,
                cable=index,
                capacity=4096,
                mode=mode,
                break_threshold=cable.break_threshold,
            )
            for index in range(n)
        ]
        for tracker in self.trackers:
            tracker._append_event = _discard_event.__get__(tracker)
        self._geometry_lists = (
            geometry.load_offsets[:, 0].tolist(),
            geometry.load_offsets[:, 1].tolist(),
            geometry.vessel_offsets[:, 0].tolist(),
            geometry.vessel_offsets[:, 1].tolist(),
        )
        self.reengagements: list = []
        self.taut_intervals: list = []
        self.severances: list[tuple[int, float]] = []
        self.closure: tuple[int, float] | None = None
        samples = int(round(log_duration / EVENT_PERIOD)) + 2
        state_samples = int(round(log_duration / WEATHER_PERIOD)) + 2 if log_state else 0
        self.log = FleetLog(
            event_time=np.empty(samples),
            elongation=np.empty((samples, n)),
            rate=np.empty((samples, n)),
            relative_load=np.empty((samples, n)),
            alive=np.empty((samples, n), dtype=bool),
            state_time=np.empty(state_samples),
            state=np.empty((state_samples, 6 * (n + 1))),
        )
        self._log_state = log_state

    def exogenous_force(self, time: float) -> np.ndarray:
        forces = np.zeros((self.geometry.vessel_count + 1, 2))
        if self._weather is not None:
            index = min(int(math.floor((time + 1.0e-9) / WEATHER_PERIOD)), self._weather.shape[0] - 1)
            forces += self._weather[index]
        if self._scripted_force is not None:
            forces += self._scripted_force(time)
        return forces

    def _tensions(self, elongation: np.ndarray, rate: np.ndarray, alive: np.ndarray) -> np.ndarray:
        taut = self.cable.stiffness * elongation + self.cable.damping * rate
        return np.where((elongation > 0.0) & (alive > 0.5), np.maximum(taut, 0.0), 0.0)

    def _calc_force(self, context, output) -> None:
        state = self._state_port.Eval(context).tolist()
        alive = context.get_discrete_state(self._alive_index).get_value().tolist()
        generalized = scalar_cable_step(
            state,
            self._geometry_lists,
            self.cable.rest_length,
            self.cable.stiffness,
            self.cable.damping,
            alive,
        )[0]
        output.SetFromVector(generalized)

    def _calc_tension(self, context, output) -> None:
        state = self._state_port.Eval(context)
        terms = fast_cable_terms(state, self.geometry, self.cable.rest_length)
        alive = context.get_discrete_state(self._alive_index).get_value()
        output.SetFromVector(self._tensions(terms[6], terms[7], alive))

    def _update(self, context, discrete_state) -> None:
        time = context.get_time()
        state = self._state_port.Eval(context)
        terms = fast_cable_terms(state, self.geometry, self.cable.rest_length)
        alive = context.get_discrete_state(self._alive_index).get_value().copy()
        elongation = terms[6]
        rate = terms[7]
        if self.closure is None:
            chord = elongation + self.cable.rest_length
            closest = int(np.argmin(chord))
            if chord[closest] < CLOSURE_CHORD_LENGTH:
                self.closure = (closest, time)
        taut = self.cable.stiffness * elongation + self.cable.damping * rate
        direction = np.column_stack([terms[4], terms[5]])
        relative_load = relative_gust_load(self.exogenous_force(time), direction, self.body)
        for index, tracker in enumerate(self.trackers):
            if alive[index] < 0.5:
                continue
            sever = tracker.sample(time, float(elongation[index]), float(rate[index]), float(taut[index]), float(relative_load[index]))
            if sever and self.cable.mode == "live":
                alive[index] = 0.0
                self.severances.append((index, time))
            reengagement, taut_records, _ = tracker.drain_if_due(time)
            self.reengagements.extend(reengagement)
            self.taut_intervals.extend(taut_records)
        log = self.log
        if log.count < log.event_time.size:
            log.event_time[log.count] = time
            log.elongation[log.count] = elongation
            log.rate[log.count] = rate
            log.relative_load[log.count] = relative_load
            log.alive[log.count] = alive > 0.5
            log.count += 1
        if self._log_state and log.state_count < log.state_time.size:
            tick = int(round(time / EVENT_PERIOD))
            if tick % int(round(WEATHER_PERIOD / EVENT_PERIOD)) == 0:
                log.state_time[log.state_count] = time
                log.state[log.state_count] = state
                log.state_count += 1
        discrete_state.get_mutable_vector(self._alive_index).SetFromVector(alive)

    def flush(self) -> None:
        """Move records still held in the trackers' rings onto the system lists."""
        for tracker in self.trackers:
            self.reengagements.extend(tracker.state.reengagement_ring.drain())
            self.taut_intervals.extend(tracker.state.taut_ring.drain())


def _discard_event(self, *args, **kwargs) -> None:
    """Production trackers keep marks and taut intervals, not diagnostic transitions."""


class HullForces(LeafSystem):
    """Drag, thrusters (surge along the true hull axis, yaw moment), and weather.

    Plant-side.  The command input carries ``[surge_1..N, yaw_1..N]`` from the
    controllers; reading the pose to express the body-fixed thrust in world coordinates
    is physics, and the command itself carries no truth (IV.7).
    """

    PLANT_SIDE = True

    def __init__(
        self,
        vessel_count: int,
        drag_law: str,
        weather: np.ndarray | None,
        scripted_force: Callable[[float], np.ndarray] | None,
        body: "BodyParameters | None" = None,
    ) -> None:
        super().__init__()
        body = body or BodyParameters()
        if drag_law not in ("linear", "quadratic"):
            raise ValueError(f"unknown drag law: {drag_law}")
        self._n = vessel_count
        self._drag_law = drag_law
        self._weather = weather
        self._scripted_force = scripted_force
        self._state_port = self.DeclareVectorInputPort("plant_state", 6 * (vessel_count + 1))
        self._command_port = self.DeclareVectorInputPort("thrust_command", 2 * vessel_count)
        self.DeclareVectorOutputPort("generalized_force", 3 * (vessel_count + 1), self._calc)
        linear = np.array([body.load_linear_drag] + [body.vessel_linear_drag] * vessel_count)
        self._linear_list = linear.tolist()
        self._quadratic_list = (linear / QUADRATIC_DRAG_REFERENCE_SPEED).tolist()
        self._angular_list = [constants.LOAD_ANGULAR_DRAG] + [constants.VESSEL_ANGULAR_DRAG] * vessel_count
        self._weather_lists = None
        if weather is not None and weather.shape[0] <= 200_000:
            self._weather_lists = weather.reshape(weather.shape[0], -1).tolist()

    def _calc(self, context, output) -> None:
        n = self._n
        base = 3 * (n + 1)
        state = self._state_port.Eval(context).tolist()
        command = self._command_port.Eval(context).tolist()
        linear = self._linear_list
        angular = self._angular_list
        quadratic = self._drag_law == "quadratic"
        generalized = [0.0] * base
        for body in range(n + 1):
            j = 3 * body
            vx = state[base + j]
            vy = state[base + j + 1]
            if quadratic:
                coefficient = self._quadratic_list[body] * math.sqrt(vx * vx + vy * vy)
            else:
                coefficient = linear[body]
            generalized[j] = -coefficient * vx
            generalized[j + 1] = -coefficient * vy
            generalized[j + 2] = -angular[body] * state[base + j + 2]
        for i in range(n):
            j = 3 * (i + 1)
            heading = state[j + 2]
            generalized[j] += command[i] * math.cos(heading)
            generalized[j + 1] += command[i] * math.sin(heading)
            generalized[j + 2] += command[n + i]
        time = context.get_time()
        if self._weather is not None:
            index = min(int(math.floor((time + 1.0e-9) / WEATHER_PERIOD)), self._weather.shape[0] - 1)
            sample = self._weather_lists[index] if self._weather_lists is not None else self._weather[index].ravel().tolist()
            for body in range(n + 1):
                generalized[3 * body] += sample[2 * body]
                generalized[3 * body + 1] += sample[2 * body + 1]
        if self._scripted_force is not None:
            scripted = self._scripted_force(time)
            for body in range(n + 1):
                generalized[3 * body] += float(scripted[body, 0])
                generalized[3 * body + 1] += float(scripted[body, 1])
        output.SetFromVector(generalized)


class ConstantCommand(LeafSystem):
    """Open-loop thrust command, for deterministic mechanics cells without sensors."""

    PLANT_SIDE = False

    def __init__(self, command: np.ndarray) -> None:
        super().__init__()
        self._command = np.asarray(command, dtype=float).copy()
        self.DeclareVectorOutputPort("thrust_command", self._command.size, self._calc)

    def _calc(self, context, output) -> None:
        output.SetFromVector(self._command)


@dataclass
class FleetPlant:
    builder: DiagramBuilder
    plant: object
    cables: UnilateralCableBank
    hull: HullForces
    geometry: FleetGeometry
    operating: OperatingPoint
    diagram: object = None
    extras: dict = field(default_factory=dict)
    body: BodyParameters = field(default_factory=BodyParameters)


def add_fleet_plant(
    builder: DiagramBuilder,
    geometry: FleetGeometry,
    pretension: float,
    *,
    drag_law: str = "linear",
    cable: CableParameters | None = None,
    weather: np.ndarray | None = None,
    scripted_force: Callable[[float], np.ndarray] | None = None,
    log_duration: float = 0.0,
    log_state: bool = True,
    time_step: float = PHYSICS_STEP,
    body: BodyParameters | None = None,
) -> FleetPlant:
    """Add the plant, the cable bank, and the hull forces; leave the command input open."""
    cable = cable or CableParameters()
    body = body or BodyParameters()
    n = geometry.vessel_count
    plant = builder.AddSystem(MultibodyPlant(time_step))
    plant.set_name("plant")
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
    plant.mutable_gravity_field().set_gravity_vector(np.zeros(3))
    load = plant.AddRigidBody("load", _spatial_inertia(body.load_mass, constants.LOAD_YAW_INERTIA))
    plant.AddJoint(PlanarJoint("load_planar", plant.world_frame(), load.body_frame()))
    for index in range(n):
        vessel = plant.AddRigidBody(
            f"vessel_{index}", _spatial_inertia(body.vessel_mass, constants.VESSEL_YAW_INERTIA)
        )
        plant.AddJoint(PlanarJoint(f"vessel_{index}_planar", plant.world_frame(), vessel.body_frame()))
    plant.Finalize()
    if plant.num_positions() != 3 * (n + 1):
        raise RuntimeError("unexpected generalized coordinate layout")
    cables = builder.AddSystem(
        UnilateralCableBank(geometry, cable, weather, scripted_force, log_duration, log_state, body)
    )
    cables.set_name("unilateral_cable_bank")
    hull = builder.AddSystem(HullForces(n, drag_law, weather, scripted_force, body))
    hull.set_name("hull_forces")
    adder = builder.AddSystem(Adder(2, 3 * (n + 1)))
    adder.set_name("force_adder")
    builder.Connect(plant.get_state_output_port(), cables.get_input_port(0))
    builder.Connect(plant.get_state_output_port(), hull.get_input_port(0))
    builder.Connect(cables.get_output_port(0), adder.get_input_port(0))
    builder.Connect(hull.get_output_port(0), adder.get_input_port(1))
    builder.Connect(adder.get_output_port(0), plant.get_applied_generalized_force_input_port())
    return FleetPlant(
        builder=builder,
        plant=plant,
        cables=cables,
        hull=hull,
        geometry=geometry,
        operating=operating_point(geometry, pretension, drag_law, body),
        body=body,
    )


def equilibrium_state(
    geometry: FleetGeometry,
    operating: OperatingPoint,
    cable: CableParameters | None = None,
    load_heading: float = 0.0,
) -> np.ndarray:
    """Exact steady-tow state ``[q; v]``: load at the origin, cables at T0/k elongation."""
    cable = cable or CableParameters()
    n = geometry.vessel_count
    positions = np.zeros((n + 1, 3))
    velocities = np.zeros((n + 1, 3))
    positions[0, 2] = load_heading
    velocities[0, :2] = operating.speed * np.array([math.cos(load_heading), math.sin(load_heading)])
    load_arm = _rotate(np.full(n, load_heading), geometry.load_offsets)
    cable_angles = geometry.cable_angles + load_heading
    chord_length = cable.rest_length + operating.tensions / cable.stiffness
    chord = chord_length[:, None] * np.stack([np.cos(cable_angles), np.sin(cable_angles)], axis=1)
    headings = operating.headings + load_heading
    vessel_arm = _rotate(headings, geometry.vessel_offsets)
    positions[1:, :2] = load_arm + chord - vessel_arm
    positions[1:, 2] = headings
    velocities[1:, :2] = velocities[0, :2]
    return np.concatenate([positions.ravel(), velocities.ravel()])


def hull_moment_balance(geometry: FleetGeometry, operating: OperatingPoint) -> np.ndarray:
    """Yaw moment of each cable about its vessel at equilibrium (zero when aligned)."""
    angle = geometry.cable_angles
    force = -operating.tensions[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    arm = _rotate(operating.headings, geometry.vessel_offsets)
    return arm[:, 0] * force[:, 1] - arm[:, 1] * force[:, 0]


def set_state(fleet: FleetPlant, root_context, state: np.ndarray) -> None:
    plant_context = fleet.plant.GetMyMutableContextFromRoot(root_context)
    count = fleet.plant.num_positions()
    fleet.plant.SetPositions(plant_context, state[:count])
    fleet.plant.SetVelocities(plant_context, state[count:])


def plant_state(fleet: FleetPlant, root_context) -> np.ndarray:
    plant_context = fleet.plant.GetMyContextFromRoot(root_context)
    return np.concatenate(
        [fleet.plant.GetPositions(plant_context), fleet.plant.GetVelocities(plant_context)]
    )


def fleet_effective_mass(
    geometry: FleetGeometry, cable_index: int, body: "BodyParameters | None" = None
) -> float:
    """Radial reduced mass of one cable with the other N-1 vessels rigidly on the load.

    This is the stiff-cable limit of the formation: the other vessels move with the
    load through their taut cables, so the load side carries m_L + (N-1) m_A, plus the
    load's rotational term along the cable's lever arm.
    """
    n = geometry.vessel_count
    angle = geometry.cable_angles[cable_index]
    direction = np.array([math.cos(angle), math.sin(angle)])
    arm = geometry.load_offsets[cable_index]
    lever = arm[0] * direction[1] - arm[1] * direction[0]
    body = body or BodyParameters()
    load_side_mass = body.load_mass + (n - 1) * body.vessel_mass
    inverse = 1.0 / body.vessel_mass + 1.0 / load_side_mass + lever**2 / constants.LOAD_YAW_INERTIA
    return 1.0 / inverse


def two_body_effective_mass(
    geometry: FleetGeometry, cable_index: int, body: "BodyParameters | None" = None
) -> float:
    """Two-body radial reduced mass including the load's rotational term (II.1)."""
    angle = geometry.cable_angles[cable_index]
    direction = np.array([math.cos(angle), math.sin(angle)])
    arm = geometry.load_offsets[cable_index]
    lever = arm[0] * direction[1] - arm[1] * direction[0]
    body = body or BodyParameters()
    inverse = 1.0 / body.vessel_mass + 1.0 / body.load_mass + lever**2 / constants.LOAD_YAW_INERTIA
    return 1.0 / inverse


def weather_samples_needed(duration: float) -> int:
    return int(math.ceil(duration / WEATHER_PERIOD)) + 2


def as_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)
