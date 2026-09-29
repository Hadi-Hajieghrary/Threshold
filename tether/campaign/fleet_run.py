"""Assemble and run the production closed loop for one (cell, seed)."""

from __future__ import annotations

import hashlib
import json
import math
import time as wallclock
from dataclasses import asdict, dataclass, field

import numpy as np
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, EventStatus

from tether.control.controller import CABLE_TRIM_GAIN, NOMINAL_HEADING_GAIN, SWAY_GAIN, SWAY_LIMIT, ControllerBank
from tether.estimation.truth_isolation import lint_diagram
from tether.physics.fleet import (
    PHYSICS_STEP,
    BodyParameters,
    CableParameters,
    ConstantCommand,
    FleetPlant,
    add_fleet_plant,
    equilibrium_state,
    formation_geometry,
    set_state,
)
from tether.physics.sensors import SensorSuite
from tether.physics.weather import stationary_weather_forces


@dataclass(frozen=True)
class FleetRunSpec:
    """Everything that determines one production run apart from the master seed."""

    formation: str = "parallel"
    arc_half_angle: float | None = None
    pretension: float = 990.0
    heading_gain: float = NOMINAL_HEADING_GAIN
    trim_gain: float = CABLE_TRIM_GAIN
    drag_law: str = "linear"
    weather_distribution: str = "gaussian"
    weather_direction: str = "local"
    weather_scale: float = 1.0
    weather_rho: float | None = None
    weather_front_angle: float | None = None
    duration: float = 300.0
    warmup: float = 20.0
    cable_mode: str = "recording"
    break_threshold: float | None = None
    time_step: float = PHYSICS_STEP
    closed_loop: bool = True
    log_state: bool = True
    # v2 sway (chord-angle) gain, plan v2 IV.7; 0 is the v1 controller.
    k_sigma: float = SWAY_GAIN
    # Clamp on the sway correction (records/v2/phase0/sway_addendum_1.json); None is the
    # unsaturated law, the v2 value is 0.349 rad.  Inert when k_sigma = 0.
    sway_limit: float | None = SWAY_LIMIT
    # Plan v3 stiffness pair (WP1 T1.4): None means the plant constants, and keeps the hash.
    stiffness: float | None = None
    damping: float | None = None
    # ACC 2027 confirmatory runs: vessel mass and the two linear drag coefficients.  None
    # means the plant constants and keeps the hash, exactly as for the cable pair above.
    vessel_mass: float | None = None
    load_drag: float | None = None
    vessel_drag: float | None = None

    _OVERRIDE_FIELDS = ("stiffness", "damping", "vessel_mass", "load_drag", "vessel_drag")

    def __getattr__(self, name: str):
        # Specs pickled before these fields existed (records/*/cache/*.pkl) lack them; they
        # mean the plant constants.  Only reached when normal lookup fails.
        if name in ("stiffness", "damping", "vessel_mass", "load_drag", "vessel_drag"):
            return None
        raise AttributeError(name)

    def body_parameters(self) -> BodyParameters:
        overrides = {}
        if self.vessel_mass is not None:
            overrides["vessel_mass"] = float(self.vessel_mass)
        if self.load_drag is not None:
            overrides["load_linear_drag"] = float(self.load_drag)
        if self.vessel_drag is not None:
            overrides["vessel_linear_drag"] = float(self.vessel_drag)
        return BodyParameters(**overrides)

    def cable_parameters(self) -> CableParameters:
        overrides = {k: v for k, v in (("stiffness", self.stiffness), ("damping", self.damping)) if v is not None}
        return CableParameters(mode=self.cable_mode, break_threshold=self.break_threshold, **overrides)

    def config_hash(self) -> str:
        payload = asdict(self)
        if payload["k_sigma"] == SWAY_GAIN:
            # v1 specs keep their v1 hash.
            del payload["k_sigma"]
        if payload["sway_limit"] is SWAY_LIMIT:
            # Unsaturated specs keep their earlier hash.
            del payload["sway_limit"]
        for name in ("stiffness", "damping", "vessel_mass", "load_drag", "vessel_drag"):
            if payload.get(name) is None:
                # Plant-constant specs keep their earlier hash.
                payload.pop(name, None)
        payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass
class FleetRun:
    spec: FleetRunSpec
    master_seed: int
    fleet: FleetPlant
    simulator: Simulator
    controller: ControllerBank | None
    sensors: list = field(default_factory=list)
    lint_violations: list = field(default_factory=list)
    wall_seconds: float = 0.0


def fan_heading_reference(geometry, operating, cable, heading_gain: float, trim_gain: float) -> np.ndarray:
    """Pre-loaded heading reference that makes the designed tow an equilibrium.

    At the operating point the controller must cancel the stern cable's yaw moment:
    ``k_h (theta_ref - theta_eq) - k_c sin(bearing_eq) + M_cable = 0``.  For the parallel
    formation ``M_cable = bearing_eq = 0`` and the reference is the operating heading.
    """
    from tether.physics.fleet import hull_moment_balance
    from tether.physics.sensors import cable_bearing

    state = equilibrium_state(geometry, operating, cable)
    moments = hull_moment_balance(geometry, operating)
    bearings = np.array([cable_bearing(state, geometry, i, cable.rest_length) for i in range(geometry.vessel_count)])
    return operating.headings + (trim_gain * np.sin(bearings) - moments) / heading_gain


def build_run(
    spec: FleetRunSpec,
    master_seed: int,
    *,
    weather: np.ndarray | None = None,
    scripted_force=None,
    exemptions: frozenset[str] = frozenset(),
    heading_reference=None,
    surge_schedule=None,
    initial_state: np.ndarray | None = None,
    beacon_on: float | None = None,
    supervised: bool = False,
    pre_build=None,
) -> FleetRun:
    """Build the diagram, lint it, and set the initial state.

    By default the controllers hold the steady tow (constant thrust, pre-loaded fan
    reference) from the steady-tow equilibrium; missions pass their own schedules
    ``t -> (N,)`` and initial state.
    """
    geometry = formation_geometry(spec.formation, arc_half_angle=spec.arc_half_angle)
    total = spec.warmup + spec.duration
    if weather is None and spec.weather_scale > 0.0:
        weather = stationary_weather_forces(
            master_seed,
            total,
            vessel_count=geometry.vessel_count,
            distribution=spec.weather_distribution,
            direction=spec.weather_direction,
            scale=spec.weather_scale,
            rho=spec.weather_rho,
            front_angle=spec.weather_front_angle,
        )
    cable = spec.cable_parameters()
    builder = DiagramBuilder()
    fleet = add_fleet_plant(
        builder,
        geometry,
        spec.pretension,
        drag_law=spec.drag_law,
        cable=cable,
        weather=weather,
        scripted_force=scripted_force,
        log_duration=total,
        log_state=spec.log_state,
        time_step=spec.time_step,
        body=spec.body_parameters(),
    )
    operating = fleet.operating
    n = geometry.vessel_count
    controller = None
    sensors = []
    if spec.closed_loop:
        for agent in range(n):
            suite = builder.AddSystem(
                SensorSuite(geometry, agent, master_seed, cable.rest_length, **({} if beacon_on is None else {"beacon_on": beacon_on}))
            )
            suite.set_name(f"sensor_suite_{agent}")
            builder.Connect(fleet.plant.get_state_output_port(), suite.get_input_port(0))
            builder.Connect(fleet.cables.get_output_port(1), suite.get_input_port(1))
            sensors.append(suite)
        thrusts = operating.thrusts.copy()
        headings = operating.headings.copy()
        reference = fan_heading_reference(geometry, operating, cable, spec.heading_gain, spec.trim_gain)
        initial_headings = headings if initial_state is None else np.asarray(initial_state)[5 : 3 * (n + 1) : 3]
        if spec.k_sigma != SWAY_GAIN and heading_reference is not None:
            # The design chord angle phi would have to follow the mission's course.
            raise NotImplementedError("k_sigma != 0 with a custom heading reference is not supported")
        sway = {} if spec.k_sigma == SWAY_GAIN else {"sway_gain": spec.k_sigma, "design_chord_angles": geometry.cable_angles}
        if sway and spec.sway_limit is not SWAY_LIMIT:
            sway["sway_limit"] = spec.sway_limit
        controller = builder.AddSystem(
            ControllerBank(
                n,
                heading_reference=heading_reference or (lambda t, h=reference: h),
                surge_schedule=surge_schedule or (lambda t, f=thrusts: f),
                initial_headings=initial_headings,
                heading_gain=spec.heading_gain,
                trim_gain=spec.trim_gain,
                supervised=supervised,
                **sway,
            )
        )
        controller.set_name("controller_bank")
        for agent, suite in enumerate(sensors):
            builder.Connect(suite.GetOutputPort("odometry"), controller.get_input_port(agent))
            builder.Connect(suite.GetOutputPort("cable_bearing"), controller.get_input_port(n + agent))
        builder.Connect(controller.get_output_port(0), fleet.hull.get_input_port(1))
    else:
        command = builder.AddSystem(
            ConstantCommand(np.concatenate([operating.thrusts, np.zeros(n)]))
        )
        command.set_name("constant_command")
        builder.Connect(command.get_output_port(0), fleet.hull.get_input_port(1))
    if pre_build is not None:
        # Hook for monitors, supervisors and estimator adapters; they are linted with
        # everything else below.
        fleet.extras.update(pre_build(builder, fleet, sensors, controller) or {})
    diagram = builder.Build()
    fleet.diagram = diagram
    violations = lint_diagram(diagram, exemptions)
    if violations:
        raise RuntimeError("truth-isolation lint failed: " + "; ".join(violations))
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    start = equilibrium_state(geometry, operating, cable) if initial_state is None else np.asarray(initial_state, dtype=float)
    set_state(fleet, context, start)
    # The exact t = 0 state: the 10 ms state log's first entry is taken after the plant's
    # first step, so surveyed initial poses must come from here.
    fleet.extras["initial_state"] = start.copy()
    cables = fleet.cables

    def monitor(root_context):
        if cables.closure is not None:
            return EventStatus.ReachedTermination(cables, "formation closure")
        return EventStatus.Succeeded()

    simulator.set_monitor(monitor)
    return FleetRun(spec, master_seed, fleet, simulator, controller, sensors, violations)


def advance(run: FleetRun, until: float) -> None:
    started = wallclock.perf_counter()
    if not run.simulator.get_context().get_time() > 0.0:
        run.simulator.Initialize()
    run.simulator.AdvanceTo(until)
    run.wall_seconds += wallclock.perf_counter() - started


def run_to_end(run: FleetRun) -> FleetRun:
    advance(run, run.spec.warmup + run.spec.duration)
    run.fleet.cables.flush()
    return run


def end_time(run: FleetRun) -> float:
    return float(run.simulator.get_context().get_time())


def wall_seconds_per_sim_second(run: FleetRun) -> float:
    return run.wall_seconds / max(run.simulator.get_context().get_time(), 1.0e-12)


def seconds_to_samples(seconds: float, period: float) -> int:
    return int(math.floor(seconds / period + 1.0e-9))
