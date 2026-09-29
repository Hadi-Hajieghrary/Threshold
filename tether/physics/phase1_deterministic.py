"""Deterministic full-formation and excursion mechanics for Phase 1."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.multibody.math import SpatialForce, SpatialVelocity
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    DiscreteContactApproximation,
    ExternallyAppliedSpatialForce,
    ExternallyAppliedSpatialForceMultiplexer,
)
from pydrake.multibody.tree import BodyIndex, PlanarJoint
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem

from tether.physics import constants
from tether.physics.cable import CableConfig, CableMode, UnilateralCable
from tether.physics.phase1_mechanics import IMPACT_FACTOR, PINNED_IMPEDANCE
from tether.physics.plant import (
    _spatial_inertia,
    run_steady_transit,
    run_transit_log,
    steady_state_values,
)

TARGET_CABLE = constants.VESSEL_COUNT // 2
FLEET_IMPACT_SPEEDS = np.array([0.25, 0.5, 1.0, 2.0, 3.0])
FLEET_IMPEDANCE_REFERENCE = 9_000.0
PINNED_FLEET_IMPEDANCE = 8_304.485029550166
PRETENSION_LEVELS = np.array([742.5, 990.0, 1237.5])
GUST_RATIOS = np.array([0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 2.0])
GUST_DURATIONS = np.array([0.5, 1.0, 2.0, 4.0])
PRIMARY_ENTRY_SPEED = 0.10
SENSITIVITY_ENTRY_SPEEDS = (0.05, 0.20)
REDUCED_MASS = (
    constants.VESSEL_MASS
    * constants.LOAD_MASS
    / (constants.VESSEL_MASS + constants.LOAD_MASS)
)
GEOMETRIC_DEPTH_LIMIT = 0.99 * constants.CABLE_REST_LENGTH
PHASE1R_GUST_RATIOS = np.array([0.8, 0.95, 1.0, 1.05, 1.2])
PHASE1R_GUST_DURATIONS = np.array([0.05, 0.15])
PHASE1R_ENTRY_SPEEDS = np.array([0.10, 0.20])
PHASE1R_HORIZON = 0.8
PHASE1R_INITIAL_EXTENSION = 0.002
PHASE1R_WORKERS = 4
PHASE1R_ENERGY_BALANCE_TOLERANCE = 0.05
PHASE1R_IMPACT_MIN_SPEED = 0.25
PHASE1R_IMPACT_MAX_SPEED = 3.0
PHASE1R_IMPACT_MIN_MARKS = 3
PHASE1R_T6_FIT_WINDOWS = (0.002, 0.005, 0.010, 0.020)
PHASE1R_T6_PROBE_STEPS = (0.001, 0.0005)
PHASE1R_T6_PRIMARY_STEP = 0.001
PHASE1R_T6_PRIMARY_WINDOW = 0.005
PHASE1R_T6_SLACK_EXTENSION = -1.0e-3
PINNED_FLEET_EFFECTIVE_MASS = (
    PINNED_FLEET_IMPEDANCE / IMPACT_FACTOR
) ** 2 / constants.CABLE_STIFFNESS


@dataclass(frozen=True)
class FullFormationConfig:
    """Configuration for the collinear five-vessel unilateral plant."""

    thrusts: float | Sequence[float] = constants.NOMINAL_THRUST
    pretension: float = 990.0
    weather_forces: np.ndarray | None = None
    constant_body_forces: np.ndarray | None = None
    target_cable: int = TARGET_CABLE
    differential_gust_n: float = 0.0
    gust_start_s: float = 0.0
    gust_duration_s: float = 0.0
    closure_depth_m: float = GEOMETRIC_DEPTH_LIMIT
    time_step_s: float = constants.TIME_STEP
    recording_step_s: float | None = None
    thrust_enabled: bool = True
    drag_enabled: bool = True
    weather_enabled: bool = True
    gust_enabled: bool = True
    cable_damping_n_s_per_m: float = constants.CABLE_DAMPING

    def __post_init__(self) -> None:
        thrusts = np.asarray(self.thrusts, dtype=float)
        if thrusts.ndim == 0:
            thrusts = np.full(constants.VESSEL_COUNT, float(thrusts))
        if thrusts.shape != (constants.VESSEL_COUNT,):
            raise ValueError("thrusts must be scalar or contain one value per vessel")
        if self.pretension < 0.0:
            raise ValueError("pretension cannot be negative")
        if not 0 <= self.target_cable < constants.VESSEL_COUNT:
            raise ValueError("target_cable is outside the formation")
        if self.gust_start_s < 0.0 or self.gust_duration_s < 0.0:
            raise ValueError("gust timing cannot be negative")
        if not 0.0 < self.closure_depth_m < constants.CABLE_REST_LENGTH:
            raise ValueError("closure depth must lie inside the cable rest length")
        if self.time_step_s <= 0.0:
            raise ValueError("time step must be positive")
        if self.cable_damping_n_s_per_m < 0.0:
            raise ValueError("cable damping cannot be negative")
        recording_step = (
            self.time_step_s
            if self.recording_step_s is None
            else self.recording_step_s
        )
        step_ratio = recording_step / self.time_step_s
        if recording_step <= 0.0 or not np.isclose(step_ratio, round(step_ratio)):
            raise ValueError(
                "recording step must be a positive integer multiple of the physics step"
            )
        object.__setattr__(self, "recording_step_s", float(recording_step))
        object.__setattr__(self, "thrusts", thrusts.copy())
        if self.weather_forces is not None:
            weather = np.asarray(self.weather_forces, dtype=float)
            if weather.ndim != 3 or weather.shape[1:] != (
                constants.VESSEL_COUNT + 1,
                2,
            ):
                raise ValueError("weather_forces must have shape (samples, 6, 2)")
            object.__setattr__(self, "weather_forces", weather.copy())
        if self.constant_body_forces is not None:
            constant_forces = np.asarray(self.constant_body_forces, dtype=float)
            if constant_forces.shape != (constants.VESSEL_COUNT + 1, 2):
                raise ValueError("constant_body_forces must have shape (6, 2)")
            object.__setattr__(self, "constant_body_forces", constant_forces.copy())


class FormationExternalForceSystem(LeafSystem):
    """Apply drag, thrust, weather, and a square differential gust."""

    def __init__(
        self,
        load_body_index: int,
        vessel_body_indices: Sequence[int],
        config: FullFormationConfig,
    ) -> None:
        super().__init__()
        self._load_body_index = load_body_index
        self._vessel_body_indices = tuple(vessel_body_indices)
        self._config = config
        self._velocities_port = self.DeclareAbstractInputPort(
            "body_spatial_velocities", AbstractValue.Make([SpatialVelocity()])
        )
        self.DeclareAbstractOutputPort(
            "applied_spatial_forces",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self._calc_forces,
        )
        for cable_number in range(constants.VESSEL_COUNT):
            self.DeclareVectorOutputPort(
                f"relative_load_{cable_number}",
                1,
                lambda context, output, index=cable_number: self._calc_relative_load(
                    context, output, index
                ),
            )

    @staticmethod
    def _make_force(
        body_index: int,
        force_world: np.ndarray,
        torque_world: np.ndarray,
    ) -> ExternallyAppliedSpatialForce:
        applied = ExternallyAppliedSpatialForce()
        applied.body_index = BodyIndex(body_index)
        applied.p_BoBq_B = np.zeros(3)
        applied.F_Bq_W = SpatialForce(tau=torque_world, f=force_world)
        return applied

    def _weather_at(self, time: float) -> np.ndarray:
        weather = self._config.weather_forces
        if weather is None or not self._config.weather_enabled:
            return np.zeros((constants.VESSEL_COUNT + 1, 2))
        sample = min(
            int(np.floor((time + 1.0e-12) / constants.WEATHER_PERIOD)),
            weather.shape[0] - 1,
        )
        return weather[sample]

    def _gust_at(self, time: float) -> float:
        config = self._config
        active = (
            config.gust_enabled
            and
            config.gust_duration_s > 0.0
            and config.gust_start_s <= time
            and time < config.gust_start_s + config.gust_duration_s
        )
        return config.differential_gust_n if active else 0.0

    def _calc_relative_load(self, context, output, cable_number: int) -> None:
        gust = self._gust_at(context.get_time())
        pair_reduced_mass = (
            constants.LOAD_MASS
            * constants.VESSEL_MASS
            / (constants.LOAD_MASS + constants.VESSEL_MASS)
        )
        relative_load = pair_reduced_mass * gust / constants.LOAD_MASS
        if cable_number == self._config.target_cable:
            relative_load += pair_reduced_mass * gust / constants.VESSEL_MASS
        output.SetAtIndex(0, relative_load)

    def _calc_forces(self, context, output) -> None:
        velocities = self._velocities_port.Eval(context)
        weather = self._weather_at(context.get_time())
        gust = self._gust_at(context.get_time())
        load_velocity = velocities[self._load_body_index]
        load_force = np.zeros(3)
        if self._config.drag_enabled:
            load_force[:2] -= (
                constants.LOAD_LINEAR_DRAG * load_velocity.translational()[:2]
            )
        load_force[:2] += weather[0]
        if self._config.constant_body_forces is not None:
            load_force[:2] += self._config.constant_body_forces[0]
        load_force[0] += gust
        load_torque = np.array(
            [
                0.0,
                0.0,
                (
                    -constants.LOAD_ANGULAR_DRAG
                    * load_velocity.rotational()[2]
                    if self._config.drag_enabled
                    else 0.0
                ),
            ]
        )
        forces = [
            self._make_force(self._load_body_index, load_force, load_torque)
        ]

        for vessel_number, body_index in enumerate(self._vessel_body_indices):
            velocity = velocities[body_index]
            vessel_force = np.zeros(3)
            if self._config.thrust_enabled:
                vessel_force[0] += self._config.thrusts[vessel_number]
            if self._config.drag_enabled:
                vessel_force[:2] -= (
                    constants.VESSEL_LINEAR_DRAG * velocity.translational()[:2]
                )
            vessel_force[:2] += weather[vessel_number + 1]
            if self._config.constant_body_forces is not None:
                vessel_force[:2] += self._config.constant_body_forces[
                    vessel_number + 1
                ]
            if vessel_number == self._config.target_cable:
                vessel_force[0] -= gust
            vessel_torque = np.array(
                [
                    0.0,
                    0.0,
                    (
                        -constants.VESSEL_ANGULAR_DRAG
                        * velocity.rotational()[2]
                        if self._config.drag_enabled
                        else 0.0
                    ),
                ]
            )
            forces.append(self._make_force(body_index, vessel_force, vessel_torque))
        output.set_value(forces)


@dataclass(frozen=True)
class FullFormationPlant:
    diagram: object
    plant: object
    external_forces: FormationExternalForceSystem
    cables: tuple[UnilateralCable, ...]
    load_body_index: int
    vessel_body_indices: tuple[int, ...]
    load_attachment_offsets: tuple[np.ndarray, ...]
    config: FullFormationConfig


@dataclass(frozen=True)
class FullFormationTrajectory:
    time: np.ndarray
    truth: np.ndarray
    tension: np.ndarray
    elongation: np.ndarray
    elongation_rate: np.ndarray
    geometrically_engaged: np.ndarray
    force_positive: np.ndarray
    relative_load: np.ndarray
    kinetic_energy_j: np.ndarray
    spring_potential_j: np.ndarray
    gust_power_w: np.ndarray
    thrust_power_w: np.ndarray
    linear_drag_power_w: np.ndarray
    angular_drag_power_w: np.ndarray
    weather_power_w: np.ndarray
    constant_body_force_power_w: np.ndarray
    cable_damping_dissipation_w: np.ndarray
    cable_clipping_dissipation_w: np.ndarray
    reengagement_records: tuple[object, ...]
    taut_records: tuple[object, ...]
    event_records: tuple[object, ...]
    closure_reached: bool
    censor_reason: str | None
    target_cable: int
    pretension: float

    @property
    def mean_distance(self) -> float:
        position_count = self.truth.shape[1] // 2
        positions = self.truth[:, :position_count]
        return float(np.mean(positions[-1, 0::3] - positions[0, 0::3]))

    @property
    def first_target_peak(self) -> float:
        values = self.tension[:, self.target_cable]
        for index in range(1, values.size):
            if values[index] < values[index - 1]:
                return float(values[index - 1])
        raise RuntimeError("trajectory ended before the target cable's first peak")


def build_full_formation_plant(
    config: FullFormationConfig | None = None,
) -> FullFormationPlant:
    """Build the Phase 0 geometry with five recording-mode unilateral cables."""
    if config is None:
        config = FullFormationConfig()
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=config.time_step_s)
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
    plant.mutable_gravity_field().set_gravity_vector(np.zeros(3))

    load = plant.AddRigidBody(
        "load", _spatial_inertia(constants.LOAD_MASS, constants.LOAD_YAW_INERTIA)
    )
    plant.AddJoint(PlanarJoint("load_planar", plant.world_frame(), load.body_frame()))
    vessels = []
    for vessel_number in range(constants.VESSEL_COUNT):
        vessel = plant.AddRigidBody(
            f"vessel_{vessel_number}",
            _spatial_inertia(constants.VESSEL_MASS, constants.VESSEL_YAW_INERTIA),
        )
        plant.AddJoint(
            PlanarJoint(
                f"vessel_{vessel_number}_planar",
                plant.world_frame(),
                vessel.body_frame(),
            )
        )
        vessels.append(vessel)
    plant.Finalize()

    load_offsets = tuple(
        np.array([0.0, lateral_offset, 0.0])
        for lateral_offset in np.linspace(-2.0, 2.0, constants.VESSEL_COUNT)
    )
    cables = []
    for cable_number, (vessel, load_offset) in enumerate(zip(vessels, load_offsets)):
        cable = builder.AddSystem(
            UnilateralCable(
                CableConfig(
                    load_body_index=int(load.index()),
                    vessel_body_index=int(vessel.index()),
                    load_offset=load_offset,
                    vessel_offset=np.zeros(3),
                    damping=config.cable_damping_n_s_per_m,
                    mode=CableMode.RECORDING,
                    cable=cable_number,
                )
            )
        )
        cable.set_name(f"unilateral_cable_{cable_number}")
        builder.Connect(plant.get_body_poses_output_port(), cable.get_input_port(0))
        builder.Connect(
            plant.get_body_spatial_velocities_output_port(), cable.get_input_port(1)
        )
        cables.append(cable)

    external_forces = builder.AddSystem(
        FormationExternalForceSystem(
            int(load.index()),
            [int(vessel.index()) for vessel in vessels],
            config,
        )
    )
    builder.Connect(
        plant.get_body_spatial_velocities_output_port(),
        external_forces.get_input_port(0),
    )
    force_mux = builder.AddSystem(
        ExternallyAppliedSpatialForceMultiplexer(constants.VESSEL_COUNT + 1)
    )
    builder.Connect(external_forces.get_output_port(0), force_mux.get_input_port(0))
    for cable_number, cable in enumerate(cables):
        builder.Connect(cable.get_output_port(), force_mux.get_input_port(cable_number + 1))
        builder.Connect(
            external_forces.get_output_port(cable_number + 1),
            cable.get_input_port(2),
        )
    builder.Connect(force_mux.get_output_port(), plant.get_applied_spatial_force_input_port())

    return FullFormationPlant(
        diagram=builder.Build(),
        plant=plant,
        external_forces=external_forces,
        cables=tuple(cables),
        load_body_index=int(load.index()),
        vessel_body_indices=tuple(int(vessel.index()) for vessel in vessels),
        load_attachment_offsets=load_offsets,
        config=config,
    )


def _initialize_full_formation(
    model: FullFormationPlant,
    root_context,
    common_speed: float,
    target_relative_speed: float = 0.0,
    target_elongation: float | None = None,
    initial_truth: np.ndarray | None = None,
) -> None:
    plant_context = model.plant.GetMyMutableContextFromRoot(root_context)
    if initial_truth is not None:
        truth = np.asarray(initial_truth, dtype=float)
        expected_size = model.plant.num_positions() + model.plant.num_velocities()
        if truth.shape != (expected_size,):
            raise ValueError(f"initial truth must have shape ({expected_size},)")
        position_count = model.plant.num_positions()
        model.plant.SetPositions(plant_context, truth[:position_count])
        model.plant.SetVelocities(plant_context, truth[position_count:])
        return
    truth = full_formation_initial_truth(
        model.config,
        common_speed=common_speed,
        target_relative_speed=target_relative_speed,
        target_elongation=target_elongation,
    )
    position_count = model.plant.num_positions()
    model.plant.SetPositions(plant_context, truth[:position_count])
    model.plant.SetVelocities(plant_context, truth[position_count:])


def full_formation_initial_truth(
    config: FullFormationConfig,
    *,
    common_speed: float,
    target_relative_speed: float = 0.0,
    target_elongation: float | None = None,
) -> np.ndarray:
    """Construct the exact six-body state used by the full-formation initializer."""
    extension = config.pretension / constants.CABLE_STIFFNESS
    state_size = 3 * (constants.VESSEL_COUNT + 1)
    positions = np.zeros(state_size)
    velocities = np.zeros(state_size)
    velocities[0] = common_speed
    for vessel_number, lateral_offset in enumerate(
        np.linspace(-2.0, 2.0, constants.VESSEL_COUNT)
    ):
        position_start = 3 * (vessel_number + 1)
        cable_extension = (
            target_elongation
            if vessel_number == config.target_cable
            and target_elongation is not None
            else extension
        )
        positions[position_start : position_start + 3] = (
            constants.CABLE_REST_LENGTH + cable_extension,
            lateral_offset,
            0.0,
        )
        velocities[position_start] = common_speed

    if target_relative_speed:
        total_mass = constants.LOAD_MASS + constants.VESSEL_COUNT * constants.VESSEL_MASS
        target_delta = (total_mass - constants.VESSEL_MASS) / total_mass
        rest_delta = -constants.VESSEL_MASS / total_mass
        velocities[0] += rest_delta * target_relative_speed
        for vessel_number in range(constants.VESSEL_COUNT):
            velocity_index = 3 * (vessel_number + 1)
            velocity_scale = (
                target_delta if vessel_number == config.target_cable else rest_delta
            )
            velocities[velocity_index] += velocity_scale * target_relative_speed
    return np.concatenate((positions, velocities))


def run_full_formation(
    config: FullFormationConfig,
    duration: float,
    *,
    common_speed: float,
    target_relative_speed: float = 0.0,
    target_elongation: float | None = None,
    initial_truth: np.ndarray | None = None,
) -> FullFormationTrajectory:
    """Advance the full plant and record all cable states at the configured cadence."""
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    model = build_full_formation_plant(config)
    simulator = Simulator(model.diagram)
    root_context = simulator.get_mutable_context()
    plant_context = model.plant.GetMyMutableContextFromRoot(root_context)
    _initialize_full_formation(
        model,
        root_context,
        common_speed,
        target_relative_speed,
        target_elongation,
        initial_truth,
    )
    simulator.Initialize()

    times = np.arange(
        0.0,
        duration + 0.5 * config.recording_step_s,
        config.recording_step_s,
    )
    truth = np.empty(
        (times.size, model.plant.num_positions() + model.plant.num_velocities())
    )
    tension = np.empty((times.size, constants.VESSEL_COUNT))
    elongation = np.empty_like(tension)
    elongation_rate = np.empty_like(tension)
    geometrically_engaged = np.empty_like(tension, dtype=bool)
    force_positive = np.empty_like(tension, dtype=bool)
    relative_load = np.empty_like(tension)
    closure_reached = False
    final_sample_count = times.size
    for sample_index, sample_time in enumerate(times):
        if sample_index:
            simulator.AdvanceTo(float(sample_time))
        truth[sample_index] = np.concatenate(
            (
                model.plant.GetPositions(plant_context),
                model.plant.GetVelocities(plant_context),
            )
        )
        for cable_number, cable in enumerate(model.cables):
            cable_context = cable.GetMyContextFromRoot(root_context)
            sample = cable.sample(cable_context)
            tension[sample_index, cable_number] = sample.applied_tension
            elongation[sample_index, cable_number] = sample.elongation
            elongation_rate[sample_index, cable_number] = sample.elongation_rate
            geometrically_engaged[sample_index, cable_number] = (
                sample.geometrically_engaged
            )
            force_positive[sample_index, cable_number] = sample.force_positive
            relative_load[sample_index, cable_number] = cable.get_input_port(2).Eval(
                cable_context
            )[0]
        if (
            -elongation[sample_index, config.target_cable]
            >= config.closure_depth_m
        ):
            closure_reached = True
            final_sample_count = sample_index + 1
            break
    times = times[:final_sample_count]
    truth = truth[:final_sample_count]
    tension = tension[:final_sample_count]
    elongation = elongation[:final_sample_count]
    elongation_rate = elongation_rate[:final_sample_count]
    geometrically_engaged = geometrically_engaged[:final_sample_count]
    force_positive = force_positive[:final_sample_count]
    relative_load = relative_load[:final_sample_count]
    position_count = model.plant.num_positions()
    body_velocities = truth[:, position_count:].reshape(
        -1, constants.VESSEL_COUNT + 1, 3
    )
    translational_speed_squared = np.sum(body_velocities[:, :, :2] ** 2, axis=2)
    kinetic_energy = (
        0.5 * constants.LOAD_MASS * translational_speed_squared[:, 0]
        + 0.5
        * constants.VESSEL_MASS
        * np.sum(translational_speed_squared[:, 1:], axis=1)
        + 0.5 * constants.LOAD_YAW_INERTIA * body_velocities[:, 0, 2] ** 2
        + 0.5
        * constants.VESSEL_YAW_INERTIA
        * np.sum(body_velocities[:, 1:, 2] ** 2, axis=1)
    )
    spring_potential = 0.5 * constants.CABLE_STIFFNESS * np.sum(
        np.maximum(elongation, 0.0) ** 2, axis=1
    )
    gust_active = (
        (times >= config.gust_start_s)
        & (times < config.gust_start_s + config.gust_duration_s)
    )
    gust_force = config.differential_gust_n * gust_active * config.gust_enabled
    target_velocity = body_velocities[:, config.target_cable + 1, 0]
    gust_power = gust_force * (body_velocities[:, 0, 0] - target_velocity)
    thrust_power = np.zeros(times.size)
    if config.thrust_enabled:
        thrust_power = np.sum(
            np.asarray(config.thrusts)[None, :] * body_velocities[:, 1:, 0], axis=1
        )
    linear_drag_power = np.zeros(times.size)
    angular_drag_power = np.zeros(times.size)
    if config.drag_enabled:
        linear_drag_power = (
            -constants.LOAD_LINEAR_DRAG * translational_speed_squared[:, 0]
            - constants.VESSEL_LINEAR_DRAG
            * np.sum(translational_speed_squared[:, 1:], axis=1)
        )
        angular_drag_power = (
            -constants.LOAD_ANGULAR_DRAG * body_velocities[:, 0, 2] ** 2
            - constants.VESSEL_ANGULAR_DRAG
            * np.sum(body_velocities[:, 1:, 2] ** 2, axis=1)
        )
    weather_power = np.zeros(times.size)
    if config.weather_forces is not None and config.weather_enabled:
        weather_indices = np.minimum(
            np.floor((times + 1.0e-12) / constants.WEATHER_PERIOD).astype(int),
            config.weather_forces.shape[0] - 1,
        )
        weather_power = np.sum(
            config.weather_forces[weather_indices]
            * body_velocities[:, :, :2],
            axis=(1, 2),
        )
    constant_body_force_power = np.zeros(times.size)
    if config.constant_body_forces is not None:
        constant_body_force_power = np.sum(
            config.constant_body_forces[None, :, :]
            * body_velocities[:, :, :2],
            axis=(1, 2),
        )
    damping_active = force_positive & geometrically_engaged
    cable_damping_dissipation = config.cable_damping_n_s_per_m * np.sum(
        np.where(damping_active, elongation_rate**2, 0.0), axis=1
    )
    clipped = geometrically_engaged & ~force_positive
    cable_clipping_dissipation = np.sum(
        np.where(
            clipped,
            np.maximum(
                0.0,
                -constants.CABLE_STIFFNESS * elongation * elongation_rate,
            ),
            0.0,
        ),
        axis=1,
    )
    reengagement_records = []
    taut_records = []
    event_records = []
    for cable in model.cables:
        cable_context = cable.GetMyContextFromRoot(root_context)
        event_state = cable.event_state(cable_context)
        reengagement_records.extend(cable.drained_reengagement_records)
        reengagement_records.extend(event_state.reengagement_ring.snapshot())
        taut_records.extend(cable.drained_taut_records)
        taut_records.extend(event_state.taut_ring.snapshot())
        event_records.extend(cable.drained_event_records)
        event_records.extend(event_state.event_ring.snapshot())
    return FullFormationTrajectory(
        time=times,
        truth=truth,
        tension=tension,
        elongation=elongation,
        elongation_rate=elongation_rate,
        geometrically_engaged=geometrically_engaged,
        force_positive=force_positive,
        relative_load=relative_load,
        kinetic_energy_j=kinetic_energy,
        spring_potential_j=spring_potential,
        gust_power_w=gust_power,
        thrust_power_w=thrust_power,
        linear_drag_power_w=linear_drag_power,
        angular_drag_power_w=angular_drag_power,
        weather_power_w=weather_power,
        constant_body_force_power_w=constant_body_force_power,
        cable_damping_dissipation_w=cable_damping_dissipation,
        cable_clipping_dissipation_w=cable_clipping_dissipation,
        reengagement_records=tuple(reengagement_records),
        taut_records=tuple(taut_records),
        event_records=tuple(event_records),
        closure_reached=closure_reached,
        censor_reason="minimum_attachment_separation" if closure_reached else None,
        target_cable=config.target_cable,
        pretension=config.pretension,
    )


def deterministic_fleet_acceptance(
    duration: float = 0.25,
    *,
    include_evidence: bool = False,
) -> dict[str, object] | tuple[dict[str, object], dict[str, np.ndarray]]:
    """Evaluate full-formation P1-T1 and P1-T2b from Drake trajectories."""
    speed, pretension = steady_state_values(constants.NOMINAL_THRUST)
    config = FullFormationConfig(pretension=pretension)
    unilateral = run_full_formation(config, duration, common_speed=speed)
    bilateral = run_steady_transit(constants.NOMINAL_THRUST, duration=duration)
    reference_distance = bilateral.speed * duration
    distance_error = abs(unilateral.mean_distance / reference_distance - 1.0)
    mean_tension = float(np.mean(unilateral.tension))
    tension_error = abs(mean_tension / bilateral.mean_tension - 1.0)
    no_slack = bool(np.all(unilateral.elongation > 0.0))

    baseline_trajectory = run_full_formation(
        config,
        duration,
        common_speed=speed,
        target_relative_speed=0.0,
        target_elongation=0.0,
    )
    zero_speed_baseline = float(
        np.max(baseline_trajectory.tension[:, config.target_cable])
    )
    trajectories = [
        run_full_formation(
            config,
            duration,
            common_speed=speed,
            target_relative_speed=float(relative_speed),
            target_elongation=0.0,
        )
        for relative_speed in FLEET_IMPACT_SPEEDS
    ]
    peaks = np.array([trajectory.first_target_peak for trajectory in trajectories])
    impedance = float(
        np.dot(FLEET_IMPACT_SPEEDS, peaks)
        / np.dot(FLEET_IMPACT_SPEEDS, FLEET_IMPACT_SPEEDS)
    )
    residual = peaks - impedance * FLEET_IMPACT_SPEEDS
    uncentred_r_squared = float(
        1.0 - np.dot(residual, residual) / np.dot(peaks, peaks)
    )
    baseline_subtracted_peaks = peaks - zero_speed_baseline
    baseline_subtracted_impedance, baseline_subtracted_r_squared = (
        _through_origin_fit(FLEET_IMPACT_SPEEDS, baseline_subtracted_peaks)
    )
    affine_slope, affine_intercept = np.polyfit(FLEET_IMPACT_SPEEDS, peaks, 1)
    affine_residual = peaks - (
        affine_intercept + affine_slope * FLEET_IMPACT_SPEEDS
    )
    affine_r_squared = float(
        1.0
        - np.dot(affine_residual, affine_residual)
        / np.dot(peaks - np.mean(peaks), peaks - np.mean(peaks))
    )
    other_initial_tensions = np.array(
        [
            np.delete(trajectory.tension[0], config.target_cable)
            for trajectory in trajectories
        ]
    )
    other_taut_at_pretension = bool(
        np.allclose(other_initial_tensions, pretension, rtol=0.0, atol=1.0e-8)
    )
    tests = {
        "P1-T1": {
            "phase0_distance_m": reference_distance,
            "unilateral_distance_m": unilateral.mean_distance,
            "distance_relative_error": distance_error,
            "phase0_mean_tension_n": bilateral.mean_tension,
            "unilateral_mean_tension_n": mean_tension,
            "tension_relative_error": tension_error,
            "minimum_elongation_m": float(np.min(unilateral.elongation)),
            "no_cable_slack": no_slack,
            "rotational_excess_kg": 0.0,
            "geometry": "collinear Phase 0 specialization",
            "passed": distance_error <= 0.01 and tension_error <= 0.03 and no_slack,
        },
        "P1-T2b": {
            "target_cable": config.target_cable,
            "speeds_mps": FLEET_IMPACT_SPEEDS.tolist(),
            "first_peaks_n": peaks.tolist(),
            "target_initial_tension_n": 0.0,
            "impedance_n_s_per_m": impedance,
            "pinned_impedance_n_s_per_m": PINNED_FLEET_IMPEDANCE,
            "pin_relative_error": abs(impedance / PINNED_FLEET_IMPEDANCE - 1.0),
            "reference_impedance_n_s_per_m": FLEET_IMPEDANCE_REFERENCE,
            "reference_relative_difference": abs(
                impedance / FLEET_IMPEDANCE_REFERENCE - 1.0
            ),
            "uncentred_r_squared": uncentred_r_squared,
            "zero_speed_baseline_n": zero_speed_baseline,
            "baseline_subtracted_peaks_n": baseline_subtracted_peaks.tolist(),
            "baseline_subtracted_impedance_n_s_per_m": baseline_subtracted_impedance,
            "baseline_subtracted_uncentred_r_squared": baseline_subtracted_r_squared,
            "affine_slope_n_s_per_m": float(affine_slope),
            "affine_intercept_n": float(affine_intercept),
            "affine_r_squared": affine_r_squared,
            "low_speed_residuals_n": {
                "required_through_origin": float(residual[0]),
                "baseline_subtracted": float(
                    baseline_subtracted_peaks[0]
                    - baseline_subtracted_impedance * FLEET_IMPACT_SPEEDS[0]
                ),
                "affine": float(affine_residual[0]),
            },
            "other_cables_taut_at_pretension": other_taut_at_pretension,
            "perturbation": "momentum-neutral target relative velocity",
            "passed": uncentred_r_squared > 0.99 and other_taut_at_pretension,
        },
    }
    if not include_evidence:
        return tests
    bilateral_log = run_transit_log(
        constants.NOMINAL_THRUST,
        duration,
        constants.TIME_STEP,
    )
    evidence = {
        "p1t1_bilateral_time_s": bilateral_log.time,
        "p1t1_bilateral_truth": bilateral_log.truth,
        "p1t1_bilateral_tension_n": bilateral_log.tension,
        "p1t1_unilateral_time_s": unilateral.time,
        "p1t1_unilateral_truth": unilateral.truth,
        "p1t1_unilateral_tension_n": unilateral.tension,
        "p1t1_unilateral_elongation_m": unilateral.elongation,
        "p1t2b_impact_speeds_mps": FLEET_IMPACT_SPEEDS,
        "p1t2b_first_peaks_n": peaks,
        "p1t2b_baseline_time_s": baseline_trajectory.time,
        "p1t2b_baseline_tension_n": baseline_trajectory.tension,
        "p1t2b_impact_time_s": trajectories[0].time,
        "p1t2b_impact_truth": np.stack(
            [trajectory.truth for trajectory in trajectories]
        ),
        "p1t2b_impact_tension_n": np.stack(
            [trajectory.tension for trajectory in trajectories]
        ),
        "p1t2b_impact_elongation_m": np.stack(
            [trajectory.elongation for trajectory in trajectories]
        ),
        "p1t2b_impact_elongation_rate_mps": np.stack(
            [trajectory.elongation_rate for trajectory in trajectories]
        ),
    }
    return tests, evidence


@dataclass(frozen=True)
class ScriptedExcursion:
    pretension_n: float
    gust_ratio: float
    gust_duration_s: float
    entry_speed_mps: float
    reduced_mass_kg: float
    restoring_acceleration_mps2: float
    gust_acceleration_mps2: float
    slack_acceleration_mps2: float
    time: np.ndarray
    elongation: np.ndarray
    elongation_rate: np.ndarray
    complete: bool
    censored: bool
    censor_reason: str | None
    reengagement_time_s: float | None
    maximum_depth_m: float
    depth_at_gust_shutoff_m: float
    rate_at_gust_shutoff_mps: float | None
    slack_at_gust_shutoff: bool
    gust_active_entire_slack_interval: bool
    gust_ceased_entire_slack_interval: bool
    return_speed_mps: float | None
    peak_tension_n: float | None

    @property
    def ballistic_depth_m(self) -> float:
        return self.entry_speed_mps**2 / (2.0 * self.restoring_acceleration_mps2)

    @property
    def corrected_constant_gust_depth_m(self) -> float | None:
        if self.gust_ratio >= 1.0:
            return None
        return self.entry_speed_mps**2 / (
            2.0 * self.restoring_acceleration_mps2 * (1.0 - self.gust_ratio)
        )

    @property
    def literal_shutoff_depth_prediction_m(self) -> float:
        return (
            self.entry_speed_mps * self.gust_duration_s
            + 0.5
            * self.restoring_acceleration_mps2
            * (self.gust_ratio - 1.0)
            * self.gust_duration_s**2
        )

    @property
    def amended_maximum_depth_prediction_m(self) -> float:
        gust_acceleration = self.slack_acceleration_mps2
        if gust_acceleration > 0.0:
            turning_time = self.entry_speed_mps / gust_acceleration
        else:
            turning_time = np.inf
        if turning_time < self.gust_duration_s:
            before_shutoff = self.entry_speed_mps**2 / (2.0 * gust_acceleration)
        else:
            before_shutoff = max(0.0, self.literal_shutoff_depth_prediction_m)
        shutoff_rate = (
            -self.entry_speed_mps
            + gust_acceleration * self.gust_duration_s
        )
        post_gust = max(0.0, self.literal_shutoff_depth_prediction_m)
        if shutoff_rate < 0.0:
            post_gust += shutoff_rate**2 / (
                2.0 * self.restoring_acceleration_mps2
            )
        return max(before_shutoff, post_gust)


def _crossing_time(
    position: float,
    velocity: float,
    acceleration: float,
    maximum_step: float,
) -> float:
    if abs(acceleration) < np.finfo(float).eps:
        return -position / velocity
    discriminant = velocity**2 - 2.0 * acceleration * position
    roots = (
        (-velocity - np.sqrt(max(0.0, discriminant))) / acceleration,
        (-velocity + np.sqrt(max(0.0, discriminant))) / acceleration,
    )
    valid = [root for root in roots if -1.0e-12 <= root <= maximum_step + 1.0e-12]
    if not valid:
        raise RuntimeError("unable to locate a bracketed excursion crossing")
    return float(min(root for root in valid if root >= -1.0e-12))


def run_scripted_excursion(
    pretension_n: float,
    gust_ratio: float,
    gust_duration_s: float,
    entry_speed_mps: float = PRIMARY_ENTRY_SPEED,
    *,
    time_step: float = constants.TIME_STEP,
    geometric_depth_limit: float = GEOMETRIC_DEPTH_LIMIT,
    impact_impedance: float = PINNED_IMPEDANCE,
) -> ScriptedExcursion:
    """Integrate the plan's reduced slack equation with exact constant-force steps."""
    if pretension_n <= 0.0 or gust_ratio < 0.0 or gust_duration_s <= 0.0:
        raise ValueError("pretension and duration must be positive; gust ratio cannot be negative")
    if entry_speed_mps <= 0.0 or time_step <= 0.0:
        raise ValueError("entry speed and time step must be positive")
    if not 0.0 < geometric_depth_limit < constants.CABLE_REST_LENGTH:
        raise ValueError("geometric depth limit must lie inside the cable length")

    restoring_acceleration = pretension_n / REDUCED_MASS
    gust_acceleration = gust_ratio * pretension_n / REDUCED_MASS
    slack_acceleration = restoring_acceleration - gust_acceleration
    maximum_time = gust_duration_s + 20.0
    time_values = [0.0]
    elongation_values = [0.0]
    rate_values = [-entry_speed_mps]
    time = 0.0
    elongation = 0.0
    rate = -entry_speed_mps
    maximum_depth = 0.0
    depth_at_shutoff = 0.0
    rate_at_shutoff = None
    slack_at_shutoff = False
    complete = False
    censored = False
    censor_reason = None
    reengagement_time = None
    return_speed = None

    while time < maximum_time - 1.0e-12:
        step = min(time_step, maximum_time - time)
        if time < gust_duration_s < time + step:
            step = gust_duration_s - time
        acceleration = (
            slack_acceleration if time < gust_duration_s - 1.0e-12 else restoring_acceleration
        )
        next_elongation = elongation + rate * step + 0.5 * acceleration * step**2
        next_rate = rate + acceleration * step
        next_time = time + step

        if acceleration > 0.0 and rate < 0.0 <= next_rate:
            turn_time = -rate / acceleration
            turn_elongation = (
                elongation + rate * turn_time + 0.5 * acceleration * turn_time**2
            )
            maximum_depth = max(maximum_depth, -turn_elongation)
        maximum_depth = max(maximum_depth, -next_elongation)

        if abs(next_time - gust_duration_s) <= 1.0e-10:
            depth_at_shutoff = max(0.0, -next_elongation)
            rate_at_shutoff = next_rate
            slack_at_shutoff = next_elongation < 0.0

        if next_elongation <= -geometric_depth_limit:
            crossing = _crossing_time(
                elongation + geometric_depth_limit,
                rate,
                acceleration,
                step,
            )
            time = time + crossing
            elongation = -geometric_depth_limit
            rate = rate + acceleration * crossing
            time_values.append(time)
            elongation_values.append(elongation)
            rate_values.append(rate)
            maximum_depth = geometric_depth_limit
            censored = True
            censor_reason = "attachment_coincidence_guard"
            break

        if time > 0.0 and elongation < 0.0 <= next_elongation and next_rate > 0.0:
            crossing = _crossing_time(elongation, rate, acceleration, step)
            reengagement_time = time + crossing
            return_speed = rate + acceleration * crossing
            time_values.append(reengagement_time)
            elongation_values.append(0.0)
            rate_values.append(return_speed)
            complete = True
            break

        time = next_time
        elongation = next_elongation
        rate = next_rate
        time_values.append(time)
        elongation_values.append(elongation)
        rate_values.append(rate)

    if not complete and not censored:
        censored = True
        censor_reason = "time_horizon"
    if complete and reengagement_time is not None and reengagement_time < gust_duration_s:
        depth_at_shutoff = 0.0
        rate_at_shutoff = None
        slack_at_shutoff = False
    peak_tension = impact_impedance * return_speed if return_speed is not None else None
    return ScriptedExcursion(
        pretension_n=pretension_n,
        gust_ratio=gust_ratio,
        gust_duration_s=gust_duration_s,
        entry_speed_mps=entry_speed_mps,
        reduced_mass_kg=REDUCED_MASS,
        restoring_acceleration_mps2=restoring_acceleration,
        gust_acceleration_mps2=gust_acceleration,
        slack_acceleration_mps2=slack_acceleration,
        time=np.asarray(time_values),
        elongation=np.asarray(elongation_values),
        elongation_rate=np.asarray(rate_values),
        complete=complete,
        censored=censored,
        censor_reason=censor_reason,
        reengagement_time_s=reengagement_time,
        maximum_depth_m=maximum_depth,
        depth_at_gust_shutoff_m=depth_at_shutoff,
        rate_at_gust_shutoff_mps=rate_at_shutoff,
        slack_at_gust_shutoff=slack_at_shutoff,
        gust_active_entire_slack_interval=bool(
            complete
            and reengagement_time is not None
            and reengagement_time <= gust_duration_s + 1.0e-12
        ),
        gust_ceased_entire_slack_interval=False,
        return_speed_mps=return_speed,
        peak_tension_n=peak_tension,
    )


def run_scripted_grid(entry_speed_mps: float) -> tuple[ScriptedExcursion, ...]:
    """Run the complete deterministic pretension, gust, and duration grid."""
    return tuple(
        run_scripted_excursion(
            float(pretension),
            float(gust_ratio),
            float(gust_duration),
            entry_speed_mps,
        )
        for pretension in PRETENSION_LEVELS
        for gust_ratio in GUST_RATIOS
        for gust_duration in GUST_DURATIONS
    )


def _thrust_for_pretension(pretension_n: float) -> tuple[float, float]:
    speed = constants.VESSEL_COUNT * pretension_n / constants.LOAD_LINEAR_DRAG
    thrust = speed * (
        constants.LOAD_LINEAR_DRAG
        + constants.VESSEL_COUNT * constants.VESSEL_LINEAR_DRAG
    ) / constants.VESSEL_COUNT
    return thrust, speed


def full_formation_slack_acceleration(
    pretension_n: float,
    gust_ratio: float,
    entry_speed_mps: float = PRIMARY_ENTRY_SPEED,
) -> float:
    """Measure target radial acceleration just after slack onset in the full plant."""
    thrust, speed = _thrust_for_pretension(pretension_n)
    config = FullFormationConfig(
        thrusts=thrust,
        pretension=pretension_n,
        differential_gust_n=gust_ratio * pretension_n,
        gust_duration_s=0.02,
    )
    trajectory = run_full_formation(
        config,
        0.006,
        common_speed=speed,
        target_relative_speed=-entry_speed_mps,
        target_elongation=0.0,
    )
    sample_count = min(5, trajectory.time.size)
    return float(
        np.polyfit(
            trajectory.time[:sample_count],
            trajectory.elongation_rate[:sample_count, config.target_cable],
            1,
        )[0]
    )


def _through_origin_fit(predictor: np.ndarray, response: np.ndarray) -> tuple[float, float]:
    slope = float(np.dot(predictor, response) / np.dot(predictor, predictor))
    residual = response - slope * predictor
    r_squared = float(1.0 - np.dot(residual, residual) / np.dot(response, response))
    return slope, r_squared


def _linear_threshold(predictor: np.ndarray, response: np.ndarray) -> dict[str, float]:
    beta, alpha = np.polyfit(predictor, response, 1)
    threshold = -alpha / beta
    return {"alpha": float(alpha), "beta": float(beta), "lambda_c": float(threshold)}


def scripted_acceptance() -> tuple[dict[str, object], tuple[ScriptedExcursion, ...]]:
    """Evaluate P1-T4 through P1-T8 with literal and amended diagnostics."""
    primary = run_scripted_grid(PRIMARY_ENTRY_SPEED)
    complete = tuple(outcome for outcome in primary if outcome.complete)
    literal_t4 = tuple(
        outcome
        for outcome in complete
        if outcome.gust_ratio < 1.0 and outcome.gust_ceased_entire_slack_interval
    )
    corrected_t4 = tuple(
        outcome
        for outcome in complete
        if outcome.gust_ratio < 1.0 and outcome.gust_active_entire_slack_interval
    )
    corrected_speed_errors = np.array(
        [
            abs(outcome.return_speed_mps / outcome.entry_speed_mps - 1.0)
            for outcome in corrected_t4
        ]
    )
    corrected_depth_errors = np.array(
        [
            abs(
                outcome.maximum_depth_m
                / outcome.corrected_constant_gust_depth_m
                - 1.0
            )
            for outcome in corrected_t4
        ]
    )
    corrected_t4_passed = bool(
        corrected_t4
        and np.max(corrected_speed_errors) < 0.05
        and np.max(corrected_depth_errors) < 0.05
    )

    energy_predictor = np.array(
        [
            np.sqrt(
                outcome.entry_speed_mps**2
                + 2.0
                * outcome.restoring_acceleration_mps2
                * outcome.maximum_depth_m
            )
            for outcome in complete
        ]
    )
    return_speeds = np.array([outcome.return_speed_mps for outcome in complete])
    energy_slope, energy_r_squared = _through_origin_fit(
        energy_predictor, return_speeds
    )
    peak_tensions = np.array([outcome.peak_tension_n for outcome in complete])
    tension_predictions = PINNED_FLEET_IMPEDANCE * return_speeds
    tension_relative_errors = np.abs(peak_tensions / tension_predictions - 1.0)
    per_pretension = {}
    for pretension in PRETENSION_LEVELS:
        selected = [
            index
            for index, outcome in enumerate(complete)
            if outcome.pretension_n == pretension
        ]
        slope, r_squared = _through_origin_fit(
            energy_predictor[selected], return_speeds[selected]
        )
        per_pretension[str(pretension)] = {
            "complete_count": len(selected),
            "slope": slope,
            "uncentred_r_squared": r_squared,
        }
    t5_passed = bool(
        abs(energy_slope - 1.0) <= 0.05
        and energy_r_squared > 0.98
        and np.max(tension_relative_errors) <= 0.06
    )

    low_lambda = tuple(
        outcome for outcome in complete if outcome.gust_ratio <= 1.0
    )
    high_lambda = tuple(
        outcome for outcome in complete if outcome.gust_ratio >= 1.05
    )
    literal_low_passed = bool(
        low_lambda
        and all(
            outcome.maximum_depth_m < outcome.ballistic_depth_m
            for outcome in low_lambda
        )
    )
    literal_high_passed = bool(
        high_lambda
        and all(
            outcome.maximum_depth_m > outcome.ballistic_depth_m
            for outcome in high_lambda
        )
    )
    threshold_samples = tuple(
        outcome
        for outcome in complete
        if outcome.maximum_depth_m < 1.0
    )
    threshold_x = np.array([outcome.gust_ratio for outcome in threshold_samples])
    threshold_y = np.array(
        [
            outcome.maximum_depth_m / outcome.ballistic_depth_m - 1.0
            for outcome in threshold_samples
        ]
    )
    literal_threshold = _linear_threshold(threshold_x, threshold_y)
    literal_threshold_passed = abs(literal_threshold["lambda_c"] - 1.0) <= 0.05

    acceleration_x = np.array([outcome.gust_ratio for outcome in primary])
    acceleration_y = np.array(
        [
            outcome.slack_acceleration_mps2
            / outcome.restoring_acceleration_mps2
            for outcome in primary
        ]
    )
    acceleration_fit = _linear_threshold(acceleration_x, acceleration_y)
    acceleration_fit_passed = abs(acceleration_fit["lambda_c"] - 1.0) <= 0.05
    full_accelerations = np.array(
        [
            full_formation_slack_acceleration(990.0, float(gust_ratio))
            for gust_ratio in GUST_RATIOS
        ]
    )
    full_acceleration_fit = _linear_threshold(
        GUST_RATIOS, full_accelerations / (990.0 / REDUCED_MASS)
    )
    full_acceleration_fit_passed = (
        abs(full_acceleration_fit["lambda_c"] - 1.0) <= 0.05
    )

    shutoff_eligible = tuple(
        outcome
        for outcome in complete
        if outcome.slack_at_gust_shutoff
        and outcome.depth_at_gust_shutoff_m < 1.0
    )
    literal_depth_errors = np.array(
        [
            abs(
                outcome.depth_at_gust_shutoff_m
                / outcome.literal_shutoff_depth_prediction_m
                - 1.0
            )
            for outcome in shutoff_eligible
        ]
    )
    amended_eligible = tuple(
        outcome
        for outcome in shutoff_eligible
        if outcome.maximum_depth_m < 1.0
    )
    amended_depth_errors = np.array(
        [
            abs(
                outcome.maximum_depth_m
                / outcome.amended_maximum_depth_prediction_m
                - 1.0
            )
            for outcome in amended_eligible
        ]
    )
    literal_t7_passed = bool(
        shutoff_eligible and np.max(literal_depth_errors) <= 0.15
    )
    amended_t7_passed = bool(
        amended_eligible and np.max(amended_depth_errors) <= 0.15
    )

    deepest = max(complete, key=lambda outcome: outcome.maximum_depth_m)
    implied_peak = PINNED_FLEET_IMPEDANCE * deepest.return_speed_mps
    sensitivity = []
    for entry_speed in (PRIMARY_ENTRY_SPEED, *SENSITIVITY_ENTRY_SPEEDS):
        outcomes = primary if entry_speed == PRIMARY_ENTRY_SPEED else run_scripted_grid(entry_speed)
        completed = [outcome for outcome in outcomes if outcome.complete]
        sensitivity.append(
            {
                "entry_speed_mps": entry_speed,
                "complete_count": len(completed),
                "censored_count": len(outcomes) - len(completed),
                "observed_maximum_complete_depth_m": max(
                    outcome.maximum_depth_m for outcome in completed
                ),
            }
        )

    literal_t6_passed = (
        literal_low_passed and literal_high_passed and literal_threshold_passed
    )
    tests = {
        "P1-T4": {
            "literal_eligible_count": len(literal_t4),
            "literal_verdict": "UNDER-POWERED" if not literal_t4 else "PASS",
            "corrected_constant_gust_eligible_count": len(corrected_t4),
            "corrected_maximum_speed_relative_error": float(
                np.max(corrected_speed_errors)
            ),
            "corrected_maximum_depth_relative_error": float(
                np.max(corrected_depth_errors)
            ),
            "corrected_verdict": "PASS" if corrected_t4_passed else "FAIL",
            "verdict": "UNDER-POWERED" if not literal_t4 else "PASS",
        },
        "P1-T5": {
            "complete_count": len(complete),
            "censored_count": len(primary) - len(complete),
            "energy_slope": energy_slope,
            "energy_uncentred_r_squared": energy_r_squared,
            "maximum_tension_relative_error": float(
                np.max(tension_relative_errors)
            ),
            "pinned_fleet_impedance_n_s_per_m": PINNED_FLEET_IMPEDANCE,
            "per_pretension": per_pretension,
            "verdict": "PASS" if t5_passed else "FAIL",
        },
        "P1-T6": {
            "literal_low_lambda_count": len(low_lambda),
            "literal_high_lambda_count": len(high_lambda),
            "literal_low_lambda_passed": literal_low_passed,
            "literal_high_lambda_passed": literal_high_passed,
            "literal_threshold_fit": literal_threshold,
            "literal_threshold_passed": literal_threshold_passed,
            "literal_verdict": "PASS" if literal_t6_passed else "FAIL",
            "force_consistent_acceleration_fit": acceleration_fit,
            "force_consistent_verdict": "PASS" if acceleration_fit_passed else "FAIL",
            "full_plant_accelerations_mps2": full_accelerations.tolist(),
            "full_plant_acceleration_fit": full_acceleration_fit,
            "full_plant_force_consistent_verdict": (
                "PASS" if full_acceleration_fit_passed else "FAIL"
            ),
            "verdict": "PASS" if literal_t6_passed else "PIVOT",
            "decisive_no_go": not literal_t6_passed,
        },
        "P1-T7": {
            "literal_eligible_count": len(shutoff_eligible),
            "literal_maximum_relative_error": float(np.max(literal_depth_errors)),
            "literal_verdict": "PASS" if literal_t7_passed else "FAIL",
            "amended_eligible_count": len(amended_eligible),
            "amended_maximum_relative_error": float(np.max(amended_depth_errors)),
            "amended_verdict": "PASS" if amended_t7_passed else "FAIL",
            "verdict": "PASS" if literal_t7_passed and amended_t7_passed else "FAIL",
        },
        "P1-T8": {
            "observed_not_global": True,
            "observed_maximum_complete_depth_m": deepest.maximum_depth_m,
            "implied_maximum_snap_tension_n": implied_peak,
            "deepest_case": {
                "pretension_n": deepest.pretension_n,
                "gust_ratio": deepest.gust_ratio,
                "gust_duration_s": deepest.gust_duration_s,
                "entry_speed_mps": deepest.entry_speed_mps,
                "return_speed_mps": deepest.return_speed_mps,
            },
            "complete_count": len(complete),
            "censored_count": len(primary) - len(complete),
            "geometric_depth_guard_m": GEOMETRIC_DEPTH_LIMIT,
            "sensitivity": sensitivity,
            "verdict": "PASS",
        },
    }
    return tests, primary


@dataclass(frozen=True)
class FullPlantExcursion:
    pretension_n: float
    gust_ratio: float
    gust_duration_s: float
    requested_entry_speed_mps: float
    status: str
    censor_reason: str | None
    geometric_down_time_s: float | None
    geometric_up_time_s: float | None
    measured_entry_speed_mps: float | None
    measured_depth_at_gust_shutoff_m: float
    measured_maximum_depth_m: float
    measured_return_speed_mps: float | None
    measured_peak_tension_n: float | None
    measured_gust_work_j: float
    measured_slack_acceleration_mps2: float | None
    reduced_shutoff_depth_prediction_m: float | None
    amended_maximum_depth_prediction_m: float | None
    slack_at_gust_shutoff: bool
    impedance_extrapolation: bool
    energy_balance: dict[str, float] | None
    trajectory: FullFormationTrajectory


def _event_time(trajectory: FullFormationTrajectory, kind: str, after: float = -np.inf):
    matches = [
        record.time
        for record in trajectory.event_records
        if record.cable == trajectory.target_cable
        and record.kind == kind
        and record.time > after
    ]
    return min(matches) if matches else None


def _interpolate_trace(
    trajectory: FullFormationTrajectory,
    values: np.ndarray,
    sample_time: float,
) -> np.ndarray | float:
    if values.ndim == 1:
        return float(np.interp(sample_time, trajectory.time, values))
    return np.array(
        [
            np.interp(sample_time, trajectory.time, values[:, column])
            for column in range(values.shape[1])
        ]
    )


def _integrate_power(
    trajectory: FullFormationTrajectory,
    power: np.ndarray,
    start_time: float,
    end_time: float,
) -> float:
    if end_time <= start_time:
        return 0.0
    interior = (trajectory.time > start_time) & (trajectory.time < end_time)
    times = np.concatenate(([start_time], trajectory.time[interior], [end_time]))
    values = np.concatenate(
        (
            [float(_interpolate_trace(trajectory, power, start_time))],
            power[interior],
            [float(_interpolate_trace(trajectory, power, end_time))],
        )
    )
    return float(np.trapezoid(values, times))


def _square_gust_work(
    trajectory: FullFormationTrajectory,
    config: FullFormationConfig,
    start_time: float,
    end_time: float,
) -> float:
    active_start = max(start_time, config.gust_start_s)
    active_end = min(end_time, config.gust_start_s + config.gust_duration_s)
    if active_end <= active_start:
        return 0.0
    position_count = trajectory.truth.shape[1] // 2
    positions = trajectory.truth[:, :position_count]
    start_positions = _interpolate_trace(trajectory, positions, active_start)
    end_positions = _interpolate_trace(trajectory, positions, active_end)
    target_x = 3 * (config.target_cable + 1)
    relative_displacement = (
        end_positions[0]
        - start_positions[0]
        - end_positions[target_x]
        + start_positions[target_x]
    )
    return float(config.differential_gust_n * relative_displacement)


def _full_plant_energy_balance(
    trajectory: FullFormationTrajectory,
    config: FullFormationConfig,
    start_time: float,
    end_time: float,
) -> dict[str, float]:
    position_count = trajectory.truth.shape[1] // 2
    positions = trajectory.truth[:, :position_count]
    start_positions = _interpolate_trace(trajectory, positions, start_time)
    end_positions = _interpolate_trace(trajectory, positions, end_time)
    delta_kinetic = float(
        _interpolate_trace(trajectory, trajectory.kinetic_energy_j, end_time)
        - _interpolate_trace(trajectory, trajectory.kinetic_energy_j, start_time)
    )
    delta_spring = float(
        _interpolate_trace(trajectory, trajectory.spring_potential_j, end_time)
        - _interpolate_trace(trajectory, trajectory.spring_potential_j, start_time)
    )
    thrust_work = float(
        np.dot(
            np.asarray(config.thrusts),
            end_positions[3::3] - start_positions[3::3],
        )
    )
    gust_work = _square_gust_work(trajectory, config, start_time, end_time)
    linear_drag_work = _integrate_power(
        trajectory, trajectory.linear_drag_power_w, start_time, end_time
    )
    angular_drag_work = _integrate_power(
        trajectory, trajectory.angular_drag_power_w, start_time, end_time
    )
    weather_work = _integrate_power(
        trajectory, trajectory.weather_power_w, start_time, end_time
    )
    damping_dissipation = _integrate_power(
        trajectory,
        trajectory.cable_damping_dissipation_w,
        start_time,
        end_time,
    )
    clipping_dissipation = _integrate_power(
        trajectory,
        trajectory.cable_clipping_dissipation_w,
        start_time,
        end_time,
    )
    delta_mechanical = delta_kinetic + delta_spring
    signed_work = (
        gust_work
        + thrust_work
        + linear_drag_work
        + angular_drag_work
        + weather_work
        - damping_dissipation
        - clipping_dissipation
    )
    residual = delta_mechanical - signed_work
    scale = max(
        abs(delta_mechanical),
        abs(gust_work)
        + abs(thrust_work)
        + abs(linear_drag_work)
        + abs(angular_drag_work)
        + abs(weather_work)
        + abs(damping_dissipation)
        + abs(clipping_dissipation),
        np.finfo(float).tiny,
    )
    return {
        "window_start_s": float(start_time),
        "window_end_s": float(end_time),
        "delta_kinetic_j": delta_kinetic,
        "delta_spring_potential_j": delta_spring,
        "delta_mechanical_energy_j": delta_mechanical,
        "gust_work_j": gust_work,
        "thrust_work_j": thrust_work,
        "linear_drag_work_j": linear_drag_work,
        "angular_drag_work_j": angular_drag_work,
        "weather_work_j": weather_work,
        "cable_damping_dissipation_j": damping_dissipation,
        "cable_clipping_dissipation_j": clipping_dissipation,
        "signed_nonconservative_work_j": signed_work,
        "residual_j": residual,
        "normalization_j": scale,
        "normalized_residual": abs(residual) / scale,
    }


def run_phase1r_acceleration_probes(
    *,
    pretensions: Sequence[float] = PRETENSION_LEVELS,
    gust_ratios: Sequence[float] = PHASE1R_GUST_RATIOS,
    time_steps: Sequence[float] = PHASE1R_T6_PROBE_STEPS,
    fit_windows: Sequence[float] = PHASE1R_T6_FIT_WINDOWS,
) -> tuple[dict[str, float], ...]:
    """Measure slack acceleration from force-consistent, initially slack states."""
    maximum_window = max(fit_windows)
    measurements = []
    for time_step in time_steps:
        for pretension in pretensions:
            thrust, common_speed = _thrust_for_pretension(float(pretension))
            for gust_ratio in gust_ratios:
                trajectory = run_full_formation(
                    FullFormationConfig(
                        thrusts=thrust,
                        pretension=float(pretension),
                        differential_gust_n=float(gust_ratio * pretension),
                        gust_duration_s=maximum_window + float(time_step),
                        time_step_s=float(time_step),
                    ),
                    maximum_window,
                    common_speed=common_speed,
                    target_elongation=PHASE1R_T6_SLACK_EXTENSION,
                )
                cable = trajectory.target_cable
                if not trajectory.elongation[0, cable] < 0.0:
                    raise RuntimeError("P1-T6 probe did not start geometrically slack")
                if trajectory.tension[0, cable] != 0.0:
                    raise RuntimeError("P1-T6 probe started with a clipped cable force")
                for fit_window in fit_windows:
                    fit_mask = trajectory.time <= fit_window + 1.0e-12
                    if np.count_nonzero(fit_mask) < 3:
                        continue
                    acceleration = float(
                        np.polyfit(
                            trajectory.time[fit_mask],
                            trajectory.elongation_rate[fit_mask, cable],
                            1,
                        )[0]
                    )
                    measurements.append(
                        {
                            "pretension_n": float(pretension),
                            "gust_ratio": float(gust_ratio),
                            "time_step_s": float(time_step),
                            "fit_window_s": float(fit_window),
                            "initial_elongation_m": float(
                                trajectory.elongation[0, cable]
                            ),
                            "initial_tension_n": float(trajectory.tension[0, cable]),
                            "maximum_elongation_m": float(
                                np.max(trajectory.elongation[fit_mask, cable])
                            ),
                            "maximum_tension_n": float(
                                np.max(trajectory.tension[fit_mask, cable])
                            ),
                            "acceleration_mps2": acceleration,
                        }
                    )
    return tuple(measurements)


def _reduced_depth_predictions(
    entry_speed: float,
    slack_duration: float,
    restoring_acceleration: float,
    gust_ratio: float,
) -> tuple[float, float]:
    slack_acceleration = restoring_acceleration * (1.0 - gust_ratio)
    depth_at_shutoff = max(
        0.0,
        entry_speed * slack_duration
        - 0.5 * slack_acceleration * slack_duration**2,
    )
    inward_speed_at_shutoff = entry_speed - slack_acceleration * slack_duration
    if slack_acceleration > 0.0:
        turning_time = entry_speed / slack_acceleration
    else:
        turning_time = np.inf
    if turning_time <= slack_duration:
        maximum_depth = entry_speed**2 / (2.0 * slack_acceleration)
    else:
        maximum_depth = depth_at_shutoff
        if inward_speed_at_shutoff > 0.0:
            maximum_depth += inward_speed_at_shutoff**2 / (
                2.0 * restoring_acceleration
            )
    return depth_at_shutoff, maximum_depth


def run_phase1r_excursion(
    pretension_n: float,
    gust_ratio: float,
    gust_duration_s: float,
    entry_speed_mps: float,
    *,
    time_step_s: float = constants.TIME_STEP,
    recording_step_s: float | None = None,
    horizon_s: float = PHASE1R_HORIZON,
) -> FullPlantExcursion:
    """Run and measure one corrected full-plant Phase 1R excursion cell."""
    thrust, common_speed = _thrust_for_pretension(pretension_n)
    config = FullFormationConfig(
        thrusts=thrust,
        pretension=pretension_n,
        differential_gust_n=gust_ratio * pretension_n,
        gust_duration_s=gust_duration_s,
        closure_depth_m=GEOMETRIC_DEPTH_LIMIT,
        time_step_s=time_step_s,
        recording_step_s=recording_step_s,
    )
    trajectory = run_full_formation(
        config,
        horizon_s,
        common_speed=common_speed,
        target_relative_speed=-entry_speed_mps,
        target_elongation=PHASE1R_INITIAL_EXTENSION,
    )
    down_time = _event_time(trajectory, "geometric_down")
    up_time = (
        _event_time(trajectory, "geometric_up", down_time)
        if down_time is not None
        else None
    )
    records = sorted(
        (
            record
            for record in trajectory.reengagement_records
            if record.cable == trajectory.target_cable
            and (down_time is None or record.t_up > down_time)
        ),
        key=lambda record: record.t_up,
    )
    record = records[0] if records else None
    target_elongation = trajectory.elongation[:, trajectory.target_cable]
    depth_at_shutoff = max(
        0.0,
        -float(
            np.interp(
                gust_duration_s,
                trajectory.time,
                target_elongation,
            )
        ),
    )
    maximum_depth = float(max(0.0, -np.min(target_elongation)))
    slack_at_shutoff = bool(
        np.interp(gust_duration_s, trajectory.time, target_elongation) < 0.0
    )
    measured_acceleration = None
    if down_time is not None:
        fit_end = min(
            gust_duration_s,
            down_time + 0.010,
            up_time if up_time is not None else trajectory.time[-1],
        )
        fit_mask = (trajectory.time >= down_time) & (trajectory.time <= fit_end)
        if np.count_nonzero(fit_mask) >= 3:
            measured_acceleration = float(
                np.polyfit(
                    trajectory.time[fit_mask],
                    trajectory.elongation_rate[
                        fit_mask, trajectory.target_cable
                    ],
                    1,
                )[0]
            )
    work_end = min(
        gust_duration_s,
        up_time if up_time is not None else trajectory.time[-1],
    )
    gust_work = (
        _square_gust_work(trajectory, config, down_time, work_end)
        if down_time is not None
        else 0.0
    )
    energy_balance = (
        _full_plant_energy_balance(trajectory, config, down_time, up_time)
        if down_time is not None and up_time is not None
        else None
    )
    measured_entry_speed = record.u_entry if record is not None else None
    restoring_acceleration = pretension_n / PINNED_FLEET_EFFECTIVE_MASS
    reduced_shutoff = None
    amended_maximum = None
    if measured_entry_speed is not None and down_time is not None:
        reduced_shutoff, amended_maximum = _reduced_depth_predictions(
            measured_entry_speed,
            max(0.0, gust_duration_s - down_time),
            restoring_acceleration,
            gust_ratio,
        )
    if record is not None:
        status = "MEASURED"
        censor_reason = None
    else:
        status = "RIGHT_CENSORED"
        censor_reason = trajectory.censor_reason or "time_horizon"
    return FullPlantExcursion(
        pretension_n=pretension_n,
        gust_ratio=gust_ratio,
        gust_duration_s=gust_duration_s,
        requested_entry_speed_mps=entry_speed_mps,
        status=status,
        censor_reason=censor_reason,
        geometric_down_time_s=down_time,
        geometric_up_time_s=up_time,
        measured_entry_speed_mps=measured_entry_speed,
        measured_depth_at_gust_shutoff_m=depth_at_shutoff,
        measured_maximum_depth_m=maximum_depth,
        measured_return_speed_mps=(record.v_return if record is not None else None),
        measured_peak_tension_n=(record.T_peak if record is not None else None),
        measured_gust_work_j=gust_work,
        measured_slack_acceleration_mps2=measured_acceleration,
        reduced_shutoff_depth_prediction_m=reduced_shutoff,
        amended_maximum_depth_prediction_m=amended_maximum,
        slack_at_gust_shutoff=slack_at_shutoff,
        impedance_extrapolation=bool(
            record is not None
            and not PHASE1R_IMPACT_MIN_SPEED
            <= record.v_return
            <= PHASE1R_IMPACT_MAX_SPEED
        ),
        energy_balance=energy_balance,
        trajectory=trajectory,
    )


def _run_phase1r_cell(parameters) -> FullPlantExcursion:
    return run_phase1r_excursion(*parameters)


def run_phase1r_grid(
    max_workers: int = PHASE1R_WORKERS,
) -> tuple[FullPlantExcursion, ...]:
    """Run the preregistered compact 60-cell full-plant Phase 1R grid."""
    parameters = tuple(
        (
            float(pretension),
            float(gust_ratio),
            float(gust_duration),
            float(entry_speed),
        )
        for pretension in PRETENSION_LEVELS
        for gust_ratio in PHASE1R_GUST_RATIOS
        for gust_duration in PHASE1R_GUST_DURATIONS
        for entry_speed in PHASE1R_ENTRY_SPEEDS
    )
    if max_workers == 1:
        return tuple(_run_phase1r_cell(cell) for cell in parameters)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return tuple(executor.map(_run_phase1r_cell, parameters))


def phase1r_excursion_acceptance(
    outcomes: tuple[FullPlantExcursion, ...] | None = None,
    acceleration_probes: tuple[dict[str, float], ...] | None = None,
) -> tuple[dict[str, object], tuple[FullPlantExcursion, ...]]:
    """Evaluate corrected P1-T5 through P1-T8 from full-plant measurements."""
    if outcomes is None:
        outcomes = run_phase1r_grid()
    if acceleration_probes is None:
        acceleration_probes = run_phase1r_acceleration_probes()
    complete = tuple(outcome for outcome in outcomes if outcome.status == "MEASURED")

    gust_work = np.array([outcome.measured_gust_work_j for outcome in complete])
    kinetic_change = np.array(
        [
            0.5
            * PINNED_FLEET_EFFECTIVE_MASS
            * (
                outcome.measured_return_speed_mps**2
                - outcome.measured_entry_speed_mps**2
            )
            for outcome in complete
        ]
    )
    energy_slope, energy_r_squared = _through_origin_fit(gust_work, kinetic_change)
    return_speeds = np.array(
        [outcome.measured_return_speed_mps for outcome in complete]
    )
    predicted_peaks = PINNED_FLEET_IMPEDANCE * return_speeds
    measured_peaks = np.array(
        [outcome.measured_peak_tension_n for outcome in complete]
    )
    all_mark_impact_slope, all_mark_impact_r_squared = _through_origin_fit(
        predicted_peaks, measured_peaks
    )
    in_domain = (
        (return_speeds >= PHASE1R_IMPACT_MIN_SPEED)
        & (return_speeds <= PHASE1R_IMPACT_MAX_SPEED)
    )
    in_domain_count = int(np.count_nonzero(in_domain))
    if in_domain_count >= PHASE1R_IMPACT_MIN_MARKS:
        impact_slope, impact_r_squared = _through_origin_fit(
            predicted_peaks[in_domain], measured_peaks[in_domain]
        )
        affine_slope, affine_intercept = np.polyfit(
            return_speeds[in_domain], measured_peaks[in_domain], 1
        )
        affine_residual = measured_peaks[in_domain] - (
            affine_intercept + affine_slope * return_speeds[in_domain]
        )
        centred_peaks = measured_peaks[in_domain] - np.mean(
            measured_peaks[in_domain]
        )
        affine_r_squared = float(
            1.0
            - np.dot(affine_residual, affine_residual)
            / np.dot(centred_peaks, centred_peaks)
        )
        affine_intercept_fraction = float(
            affine_intercept
            / (PINNED_FLEET_IMPEDANCE * np.min(return_speeds[in_domain]))
        )
        impact_status = (
            "PASS"
            if abs(impact_slope - 1.0) <= 0.06
            else "MODEL_INVALID_LOW_SPEED"
        )
    else:
        impact_slope = None
        impact_r_squared = None
        affine_slope = None
        affine_intercept = None
        affine_r_squared = None
        affine_intercept_fraction = None
        impact_status = "UNDER-POWERED"
    energy_balances = tuple(
        outcome.energy_balance
        for outcome in complete
        if outcome.energy_balance is not None
    )
    maximum_energy_residual = max(
        (balance["normalized_residual"] for balance in energy_balances),
        default=np.inf,
    )
    energy_balance_status = (
        "PASS"
        if len(energy_balances) == len(complete)
        and maximum_energy_residual <= PHASE1R_ENERGY_BALANCE_TOLERANCE
        else "FAIL"
    )
    if energy_balance_status == "FAIL":
        t5_status = "FAIL"
    elif impact_status == "MODEL_INVALID_LOW_SPEED":
        t5_status = "MODEL_INVALID_LOW_SPEED"
    elif impact_status == "UNDER-POWERED":
        t5_status = "UNDER-POWERED"
    else:
        t5_status = "PASS"

    legacy_acceleration_samples = tuple(
        outcome
        for outcome in outcomes
        if outcome.measured_slack_acceleration_mps2 is not None
    )
    legacy_acceleration_x = np.array(
        [outcome.gust_ratio for outcome in legacy_acceleration_samples]
    )
    legacy_acceleration_y = np.array(
        [
            outcome.measured_slack_acceleration_mps2
            / (outcome.pretension_n / PINNED_FLEET_EFFECTIVE_MASS)
            for outcome in legacy_acceleration_samples
        ]
    )
    legacy_acceleration_fit = _linear_threshold(
        legacy_acceleration_x, legacy_acceleration_y
    )
    probe_conditions = []
    condition_keys = sorted(
        {
            (sample["time_step_s"], sample["fit_window_s"])
            for sample in acceleration_probes
        },
        reverse=True,
    )
    for time_step, fit_window in condition_keys:
        condition_samples = tuple(
            sample
            for sample in acceleration_probes
            if sample["time_step_s"] == time_step
            and sample["fit_window_s"] == fit_window
        )
        acceleration_x = np.array(
            [sample["gust_ratio"] for sample in condition_samples]
        )
        acceleration_y = np.array(
            [
                sample["acceleration_mps2"]
                / (sample["pretension_n"] / PINNED_FLEET_EFFECTIVE_MASS)
                for sample in condition_samples
            ]
        )
        acceleration_fit = _linear_threshold(acceleration_x, acceleration_y)
        lower_sign = any(
            sample["gust_ratio"] < 1.0
            and sample["acceleration_mps2"] > 0.0
            for sample in condition_samples
        )
        upper_sign = any(
            sample["gust_ratio"] > 1.0
            and sample["acceleration_mps2"] < 0.0
            for sample in condition_samples
        )
        bracketed = lower_sign and upper_sign
        condition_passed = bool(
            bracketed and abs(acceleration_fit["lambda_c"] - 1.0) <= 0.05
        )
        probe_conditions.append(
            {
                "time_step_s": time_step,
                "fit_window_s": fit_window,
                "sample_count": len(condition_samples),
                "bracketed_around_one": bracketed,
                **acceleration_fit,
                "status": "PASS" if condition_passed else "FAIL",
            }
        )
    primary_condition = next(
        (
            condition
            for condition in probe_conditions
            if condition["time_step_s"] == PHASE1R_T6_PRIMARY_STEP
            and condition["fit_window_s"] == PHASE1R_T6_PRIMARY_WINDOW
        ),
        None,
    )
    t6_passed = bool(primary_condition and primary_condition["status"] == "PASS")
    sensitivity_passed = bool(
        probe_conditions
        and all(condition["status"] == "PASS" for condition in probe_conditions)
    )
    sensitivity_status = (
        "ROBUST_WITHIN_5_PERCENT"
        if sensitivity_passed
        else "WINDOW_OR_STEP_SENSITIVE"
        if t6_passed
        else "PRIMARY_FAIL"
    )

    t7_eligible = tuple(
        outcome
        for outcome in complete
        if outcome.slack_at_gust_shutoff
        and outcome.measured_depth_at_gust_shutoff_m < 1.0
        and outcome.reduced_shutoff_depth_prediction_m is not None
        and outcome.amended_maximum_depth_prediction_m is not None
    )
    shutoff_errors = np.array(
        [
            abs(
                outcome.measured_depth_at_gust_shutoff_m
                / outcome.reduced_shutoff_depth_prediction_m
                - 1.0
            )
            for outcome in t7_eligible
        ]
    )
    maximum_depth_errors = np.array(
        [
            abs(
                outcome.measured_maximum_depth_m
                / outcome.amended_maximum_depth_prediction_m
                - 1.0
            )
            for outcome in t7_eligible
        ]
    )
    t7_passed = bool(
        t7_eligible
        and np.max(shutoff_errors) <= 0.15
        and np.max(maximum_depth_errors) <= 0.15
    )

    deepest = max(outcomes, key=lambda outcome: outcome.measured_maximum_depth_m)
    censored = tuple(
        outcome for outcome in outcomes if outcome.status == "RIGHT_CENSORED"
    )
    tests = {
        "P1-T5": {
            "status": t5_status,
            "nonpassing_basis": (
                "impact_model_specification"
                if impact_status == "MODEL_INVALID_LOW_SPEED"
                else "full_plant_energy_balance"
                if energy_balance_status == "FAIL"
                else None
            ),
            "physical_plant_failure": energy_balance_status == "FAIL",
            "complete_count": len(complete),
            "right_censored_count": len(outcomes) - len(complete),
            "full_plant_energy_balance": {
                "status": energy_balance_status,
                "evaluated_count": len(energy_balances),
                "maximum_normalized_residual": maximum_energy_residual,
                "relative_tolerance": PHASE1R_ENERGY_BALANCE_TOLERANCE,
                "sign_convention": (
                    "Delta(K+U) - (W_gust + W_thrust + W_linear_drag + "
                    "W_angular_drag + W_weather - D_cable - D_clipping)"
                ),
            },
            "reduced_effective_mass_surrogate": {
                "status": (
                    "PASS"
                    if abs(energy_slope - 1.0) <= 0.05
                    and energy_r_squared > 0.98
                    else "FAIL"
                ),
                "effective_mass_kg": PINNED_FLEET_EFFECTIVE_MASS,
                "through_origin_slope": energy_slope,
                "uncentred_r_squared": energy_r_squared,
                "criterion": "diagnostic only; slope within 5% and R2 > 0.98",
            },
            "impact_calibration": {
                "status": impact_status,
                "physical_plant_failure": False,
                "validated_speed_range_mps": [
                    PHASE1R_IMPACT_MIN_SPEED,
                    PHASE1R_IMPACT_MAX_SPEED,
                ],
                "minimum_mark_count": PHASE1R_IMPACT_MIN_MARKS,
                "return_speed_range_mps": [
                    float(np.min(return_speeds)),
                    float(np.max(return_speeds)),
                ],
                "below_range_count": int(
                    np.count_nonzero(return_speeds < PHASE1R_IMPACT_MIN_SPEED)
                ),
                "in_range_count": in_domain_count,
                "above_range_count": int(
                    np.count_nonzero(return_speeds > PHASE1R_IMPACT_MAX_SPEED)
                ),
                "in_range_through_origin_slope": impact_slope,
                "in_range_uncentred_r_squared": impact_r_squared,
                "in_range_affine_slope_n_s_per_m": affine_slope,
                "in_range_affine_intercept_n": affine_intercept,
                "in_range_affine_r_squared": affine_r_squared,
                "affine_intercept_fraction_at_min_speed": (
                    affine_intercept_fraction
                ),
                "classification_basis": (
                    "through-origin impedance is invalid for the observed "
                    "low-speed range when its slope criterion fails; the "
                    "affine fit exposes the nonzero intercept"
                ),
                "all_mark_diagnostic": {
                    "through_origin_slope": all_mark_impact_slope,
                    "uncentred_r_squared": all_mark_impact_r_squared,
                },
                "pinned_fleet_impedance_n_s_per_m": PINNED_FLEET_IMPEDANCE,
            },
            "measurement_source": "complete full-plant energy and work traces",
        },
        "P1-T6": {
            "status": "PASS" if t6_passed else "FAIL",
            "primary_estimator": primary_condition,
            "bracketed_around_one": bool(
                primary_condition
                and primary_condition["bracketed_around_one"]
            ),
            "lambda_values": PHASE1R_GUST_RATIOS.tolist(),
            "fit_windows_s": list(PHASE1R_T6_FIT_WINDOWS),
            "probe_time_steps_s": list(PHASE1R_T6_PROBE_STEPS),
            "lambda_c_interval": [
                min(condition["lambda_c"] for condition in probe_conditions),
                max(condition["lambda_c"] for condition in probe_conditions),
            ],
            "sensitivity_status": sensitivity_status,
            "sensitivity": probe_conditions,
            "probe_measurements": list(acceleration_probes),
            "initial_state": {
                "target_elongation_m": PHASE1R_T6_SLACK_EXTENSION,
                "target_tension_n": 0.0,
                "target_relative_speed_mps": 0.0,
                "classification": "true_slack_event_local",
            },
            "legacy_transient_diagnostic": {
                "primary": False,
                "sample_count": len(legacy_acceleration_samples),
                "acceleration_fit": legacy_acceleration_fit,
                "reason": "starts taut and crosses a clipped-force transition",
            },
            "withdrawn_original_depth_comparator": {
                "status": "WITHDRAWN_SPECIFICATION",
                "physical_verdict": None,
                "threshold_estimate": None,
            },
        },
        "P1-T7": {
            "status": "PASS" if t7_passed else "FAIL",
            "eligible_count": len(t7_eligible),
            "maximum_shutoff_depth_relative_error": float(
                np.max(shutoff_errors)
            ),
            "maximum_amended_depth_relative_error": float(
                np.max(maximum_depth_errors)
            ),
            "comparator": "reduced piecewise formula only",
            "measurement_source": "full plant",
        },
        "P1-T8": {
            "status": "RIGHT_CENSORED" if censored else "OBSERVED_NOT_GLOBAL",
            "case_status_counts": {
                "MEASURED": len(complete),
                "RIGHT_CENSORED": len(censored),
            },
            "observed_maximum_depth_m": deepest.measured_maximum_depth_m,
            "observed_maximum_case": {
                "pretension_n": deepest.pretension_n,
                "gust_ratio": deepest.gust_ratio,
                "gust_duration_s": deepest.gust_duration_s,
                "entry_speed_mps": deepest.requested_entry_speed_mps,
                "status": deepest.status,
            },
            "closure_depth_m": GEOMETRIC_DEPTH_LIMIT,
            "global_maximum_established": False,
            "impedance_extrapolation_count": sum(
                outcome.impedance_extrapolation for outcome in complete
            ),
        },
    }
    return tests, outcomes