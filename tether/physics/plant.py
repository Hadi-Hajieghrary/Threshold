"""Drake Phase 0 bilateral tether plant."""

from __future__ import annotations

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
)
from pydrake.multibody.tree import BodyIndex, PlanarJoint, SpatialInertia, UnitInertia
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem

from tether.physics import constants
from tether.physics.geometry import attachment_length_rate


@dataclass(frozen=True)
class CableGeometry:
    load_body_index: int
    vessel_body_index: int
    load_offset: np.ndarray
    vessel_offset: np.ndarray


@dataclass(frozen=True)
class TransitResult:
    speed: float
    mean_tension: float
    tensions: np.ndarray
    simulated_seconds: float


@dataclass(frozen=True)
class TransitLog:
    time: np.ndarray
    truth: np.ndarray
    tension: np.ndarray
    marks: np.ndarray


def steady_state_values(thrust: float, vessel_count: int = constants.VESSEL_COUNT) -> tuple[float, float]:
    """Return analytic speed and per-cable pretension for the collinear convoy."""
    speed = vessel_count * thrust / (
        constants.LOAD_LINEAR_DRAG + vessel_count * constants.VESSEL_LINEAR_DRAG
    )
    tension = constants.LOAD_LINEAR_DRAG * speed / vessel_count
    return speed, tension


def _spatial_inertia(mass: float, yaw_inertia: float) -> SpatialInertia:
    transverse_inertia = 0.51 * yaw_inertia
    return SpatialInertia(
        mass=mass,
        p_PScm_E=np.zeros(3),
        G_SP_E=UnitInertia(
            transverse_inertia / mass,
            transverse_inertia / mass,
            yaw_inertia / mass,
        ),
    )


class TetherForceSystem(LeafSystem):
    """Produces Drake externally applied forces for the Phase 0 plant."""

    def __init__(
        self,
        load_body_index: int,
        vessel_body_indices: Sequence[int],
        cables: Sequence[CableGeometry],
        thrust: float,
        weather_forces: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self._load_body_index = load_body_index
        self._vessel_body_indices = tuple(vessel_body_indices)
        self._cables = tuple(cables)
        self._thrust = thrust
        self._weather_forces = weather_forces
        self._poses_port = self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self._velocities_port = self.DeclareAbstractInputPort(
            "body_spatial_velocities", AbstractValue.Make([SpatialVelocity()])
        )
        self.DeclareAbstractOutputPort(
            "applied_spatial_forces",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self._calc_forces,
        )

    @staticmethod
    def _point_kinematics(
        pose: RigidTransform,
        velocity: SpatialVelocity,
        offset_body: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        offset_world = pose.rotation().multiply(offset_body)
        position = pose.translation() + offset_world
        point_velocity = velocity.translational() + np.cross(velocity.rotational(), offset_world)
        return position, point_velocity

    @staticmethod
    def _make_force(
        body_index: int,
        offset_body: np.ndarray,
        force_world: np.ndarray,
        torque_world: np.ndarray | None = None,
    ):
        applied = ExternallyAppliedSpatialForce()
        applied.body_index = BodyIndex(body_index)
        applied.p_BoBq_B = offset_body
        if torque_world is None:
            torque_world = np.zeros(3)
        applied.F_Bq_W = SpatialForce(tau=torque_world, f=force_world)
        return applied

    def _weather_at(self, time: float) -> np.ndarray:
        if self._weather_forces is None:
            return np.zeros((len(self._vessel_body_indices) + 1, 2))
        sample = min(
            int(np.floor((time + 1.0e-12) / constants.WEATHER_PERIOD)),
            self._weather_forces.shape[0] - 1,
        )
        return self._weather_forces[sample]

    def _cable_state(self, poses, velocities, cable: CableGeometry):
        load_position, load_velocity = self._point_kinematics(
            poses[cable.load_body_index],
            velocities[cable.load_body_index],
            cable.load_offset,
        )
        vessel_position, vessel_velocity = self._point_kinematics(
            poses[cable.vessel_body_index],
            velocities[cable.vessel_body_index],
            cable.vessel_offset,
        )
        displacement = vessel_position - load_position
        length = np.linalg.norm(displacement)
        direction = displacement / length
        rate = attachment_length_rate(
            load_position, load_velocity, vessel_position, vessel_velocity
        )
        tension = constants.CABLE_STIFFNESS * (length - constants.CABLE_REST_LENGTH)
        tension += constants.CABLE_DAMPING * rate
        return direction, tension

    def tensions(self, context) -> np.ndarray:
        poses = self._poses_port.Eval(context)
        velocities = self._velocities_port.Eval(context)
        return np.array(
            [self._cable_state(poses, velocities, cable)[1] for cable in self._cables]
        )

    def _calc_forces(self, context, output) -> None:
        poses = self._poses_port.Eval(context)
        velocities = self._velocities_port.Eval(context)
        weather = self._weather_at(context.get_time())
        forces = []

        load_velocity = velocities[self._load_body_index]
        load_drag = np.array(
            [
                -constants.LOAD_LINEAR_DRAG * load_velocity.translational()[0],
                -constants.LOAD_LINEAR_DRAG * load_velocity.translational()[1],
                0.0,
            ]
        )
        load_drag[:2] += weather[0]
        load_torque = np.array(
            [0.0, 0.0, -constants.LOAD_ANGULAR_DRAG * load_velocity.rotational()[2]]
        )
        forces.append(
            self._make_force(
                self._load_body_index, np.zeros(3), load_drag, torque_world=load_torque
            )
        )

        for vessel_number, body_index in enumerate(self._vessel_body_indices):
            velocity = velocities[body_index]
            vessel_force = np.array(
                [
                    self._thrust - constants.VESSEL_LINEAR_DRAG * velocity.translational()[0],
                    -constants.VESSEL_LINEAR_DRAG * velocity.translational()[1],
                    0.0,
                ]
            )
            vessel_force[:2] += weather[vessel_number + 1]
            vessel_torque = np.array(
                [0.0, 0.0, -constants.VESSEL_ANGULAR_DRAG * velocity.rotational()[2]]
            )
            forces.append(
                self._make_force(
                    body_index, np.zeros(3), vessel_force, torque_world=vessel_torque
                )
            )

        for cable in self._cables:
            direction, tension = self._cable_state(poses, velocities, cable)
            force = tension * direction
            forces.append(self._make_force(cable.load_body_index, cable.load_offset, force))
            forces.append(self._make_force(cable.vessel_body_index, cable.vessel_offset, -force))

        output.set_value(forces)


@dataclass
class Phase0Plant:
    diagram: object
    plant: object
    force_system: TetherForceSystem
    load_body_index: int
    vessel_body_indices: tuple[int, ...]
    load_attachment_offsets: tuple[np.ndarray, ...]


def build_phase0_plant(
    thrust: float = constants.NOMINAL_THRUST,
    vessel_count: int = constants.VESSEL_COUNT,
    weather_forces: np.ndarray | None = None,
) -> Phase0Plant:
    """Build the collinear Phase 0 specialization with bilateral cables."""
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=constants.TIME_STEP)
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)

    load = plant.AddRigidBody(
        "load", _spatial_inertia(constants.LOAD_MASS, constants.LOAD_YAW_INERTIA)
    )
    plant.AddJoint(PlanarJoint("load_planar", plant.world_frame(), load.body_frame()))

    vessels = []
    for vessel_number in range(vessel_count):
        vessel = plant.AddRigidBody(
            f"vessel_{vessel_number}",
            _spatial_inertia(constants.VESSEL_MASS, constants.VESSEL_YAW_INERTIA),
        )
        plant.AddJoint(
            PlanarJoint(
                f"vessel_{vessel_number}_planar", plant.world_frame(), vessel.body_frame()
            )
        )
        vessels.append(vessel)

    plant.Finalize()
    lateral_offsets = np.linspace(-2.0, 2.0, vessel_count)
    cables = tuple(
        CableGeometry(
            load_body_index=int(load.index()),
            vessel_body_index=int(vessel.index()),
            load_offset=np.array([0.0, lateral_offset, 0.0]),
            vessel_offset=np.zeros(3),
        )
        for vessel, lateral_offset in zip(vessels, lateral_offsets)
    )
    force_system = builder.AddSystem(
        TetherForceSystem(
            int(load.index()),
            [int(vessel.index()) for vessel in vessels],
            cables,
            thrust,
            weather_forces=weather_forces,
        )
    )
    builder.Connect(plant.get_body_poses_output_port(), force_system.get_input_port(0))
    builder.Connect(
        plant.get_body_spatial_velocities_output_port(), force_system.get_input_port(1)
    )
    builder.Connect(force_system.get_output_port(), plant.get_applied_spatial_force_input_port())

    return Phase0Plant(
        diagram=builder.Build(),
        plant=plant,
        force_system=force_system,
        load_body_index=int(load.index()),
        vessel_body_indices=tuple(int(vessel.index()) for vessel in vessels),
        load_attachment_offsets=tuple(cable.load_offset for cable in cables),
    )


def _initialize_steady_state(model: Phase0Plant, root_context, thrust: float) -> None:
    plant_context = model.plant.GetMyMutableContextFromRoot(root_context)
    vessel_count = len(model.vessel_body_indices)
    speed, pretension = steady_state_values(thrust, vessel_count)
    extension = pretension / constants.CABLE_STIFFNESS
    positions = np.zeros(model.plant.num_positions())
    velocities = np.zeros(model.plant.num_velocities())
    velocities[0] = speed
    for vessel_number, lateral_offset in enumerate(np.linspace(-2.0, 2.0, vessel_count)):
        position_start = 3 * (vessel_number + 1)
        positions[position_start : position_start + 3] = (
            constants.CABLE_REST_LENGTH + extension,
            lateral_offset,
            0.0,
        )
        velocities[position_start] = speed
    model.plant.SetPositions(plant_context, positions)
    model.plant.SetVelocities(plant_context, velocities)


def run_steady_transit(
    thrust: float,
    duration: float = 0.25,
    vessel_count: int = constants.VESSEL_COUNT,
) -> TransitResult:
    """Advance a Drake Simulator from the analytic pretension equilibrium."""
    model = build_phase0_plant(thrust=thrust, vessel_count=vessel_count)
    simulator = Simulator(model.diagram)
    root_context = simulator.get_mutable_context()
    plant_context = model.plant.GetMyMutableContextFromRoot(root_context)
    _initialize_steady_state(model, root_context, thrust)

    simulator.Initialize()
    simulator.AdvanceTo(duration)
    final_velocities = model.plant.GetVelocities(plant_context)
    force_context = model.force_system.GetMyContextFromRoot(root_context)
    tensions = model.force_system.tensions(force_context)
    body_speeds = final_velocities[0::3]
    return TransitResult(
        speed=float(np.mean(body_speeds)),
        mean_tension=float(np.mean(tensions)),
        tensions=tensions,
        simulated_seconds=duration,
    )


def run_transit_log(
    thrust: float,
    duration: float,
    sample_period: float,
    weather_forces: np.ndarray | None = None,
    vessel_count: int = constants.VESSEL_COUNT,
) -> TransitLog:
    """Run Drake and sample deterministic truth, cable tension, and Phase 0 marks."""
    model = build_phase0_plant(
        thrust=thrust,
        vessel_count=vessel_count,
        weather_forces=weather_forces,
    )
    simulator = Simulator(model.diagram)
    root_context = simulator.get_mutable_context()
    plant_context = model.plant.GetMyMutableContextFromRoot(root_context)
    _initialize_steady_state(model, root_context, thrust)
    simulator.Initialize()

    times = np.arange(0.0, duration + 0.5 * sample_period, sample_period)
    truth = np.empty((times.size, model.plant.num_positions() + model.plant.num_velocities()))
    tensions = np.empty((times.size, vessel_count))
    for sample_index, sample_time in enumerate(times):
        simulator.AdvanceTo(float(sample_time))
        truth[sample_index] = np.concatenate(
            (
                model.plant.GetPositions(plant_context),
                model.plant.GetVelocities(plant_context),
            )
        )
        force_context = model.force_system.GetMyContextFromRoot(root_context)
        tensions[sample_index] = model.force_system.tensions(force_context)

    return TransitLog(
        time=times,
        truth=truth,
        tension=tensions,
        marks=np.empty((0, 4), dtype=np.float64),
    )