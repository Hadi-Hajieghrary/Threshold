"""Isolated Drake mechanics cells for Phase 1 acceptance tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    DiscreteContactApproximation,
)
from pydrake.multibody.tree import PrismaticJoint, SpatialInertia, UnitInertia
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

from tether.physics import constants
from tether.physics.cable import CableConfig, CableMode, UnilateralCable

IMPACT_SPEEDS = np.array([0.25, 0.5, 1.0, 2.0, 3.0])
REFINEMENT_STEPS = (1.0e-3, 5.0e-4, 2.5e-4)
P1_T9R_CANDIDATE_STEP = 2.5e-4
P1_T9R_ENERGY_LIMIT = 0.005
P1_T9R_MOMENTUM_LIMIT = 1.0e-9
P1_T9R_ORDER_BOUNDS = (0.8, 1.2)
ANALYTIC_IMPEDANCE = 7_990.0
IMPACT_FACTOR = 0.880
PINNED_IMPEDANCE = 7_993.513068475832


@dataclass(frozen=True)
class TwoBodyCableCell:
    diagram: object
    plant: object
    cable: UnilateralCable
    load_joint: PrismaticJoint
    vessel_joint: PrismaticJoint
    load_body_index: int
    vessel_body_index: int


@dataclass(frozen=True)
class ImpactTrajectory:
    time: np.ndarray
    load_position: np.ndarray
    vessel_position: np.ndarray
    load_velocity: np.ndarray
    vessel_velocity: np.ndarray
    elongation: np.ndarray
    elongation_rate: np.ndarray
    tension: np.ndarray
    load_force: np.ndarray
    vessel_force: np.ndarray
    kinetic_energy: np.ndarray
    spring_potential: np.ndarray
    alive: np.ndarray
    taut_records: tuple[object, ...]
    reengagement_records: tuple[object, ...]
    event_records: tuple[object, ...]
    drained_taut_record_count: int
    resident_taut_record_count: int
    integration_step_s: float
    plant_time_step_s: float

    @property
    def mechanical_energy(self) -> np.ndarray:
        return self.kinetic_energy + self.spring_potential

    @property
    def first_peak_tension(self) -> float:
        positive = np.flatnonzero(self.tension > 0.0)
        if positive.size == 0:
            raise RuntimeError("trajectory contains no positive cable force")
        for index in range(positive[0] + 1, self.tension.size):
            if self.tension[index] < self.tension[index - 1]:
                return float(self.tension[index - 1])
        raise RuntimeError("trajectory ended before the first tension peak")


def _body_inertia(mass: float) -> SpatialInertia:
    return SpatialInertia(
        mass=mass,
        p_PScm_E=np.zeros(3),
        G_SP_E=UnitInertia(1.0, 1.0, 1.0),
    )


def build_two_body_cable_cell(
    damping: float,
    *,
    mode: CableMode = CableMode.RECORDING,
    break_threshold: float | None = None,
    plant_time_step: float = 0.0,
) -> TwoBodyCableCell:
    """Build the collinear, force-free two-body cable identification cell."""
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=plant_time_step)
    if plant_time_step > 0.0:
        plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
    plant.mutable_gravity_field().set_gravity_vector(np.zeros(3))

    load = plant.AddRigidBody("load", _body_inertia(constants.LOAD_MASS))
    load_joint = plant.AddJoint(
        PrismaticJoint(
            "load_translation",
            plant.world_frame(),
            load.body_frame(),
            np.array([1.0, 0.0, 0.0]),
        )
    )
    vessel = plant.AddRigidBody("vessel", _body_inertia(constants.VESSEL_MASS))
    vessel_joint = plant.AddJoint(
        PrismaticJoint(
            "vessel_translation",
            plant.world_frame(),
            vessel.body_frame(),
            np.array([1.0, 0.0, 0.0]),
        )
    )
    plant.Finalize()

    cable = builder.AddSystem(
        UnilateralCable(
            CableConfig(
                load_body_index=int(load.index()),
                vessel_body_index=int(vessel.index()),
                load_offset=np.zeros(3),
                vessel_offset=np.zeros(3),
                damping=damping,
                mode=mode,
                break_threshold=break_threshold,
            )
        )
    )
    builder.Connect(plant.get_body_poses_output_port(), cable.get_input_port(0))
    builder.Connect(
        plant.get_body_spatial_velocities_output_port(), cable.get_input_port(1)
    )
    builder.Connect(cable.get_output_port(), plant.get_applied_spatial_force_input_port())
    return TwoBodyCableCell(
        diagram=builder.Build(),
        plant=plant,
        cable=cable,
        load_joint=load_joint,
        vessel_joint=vessel_joint,
        load_body_index=int(load.index()),
        vessel_body_index=int(vessel.index()),
    )


def run_two_body_impact(
    relative_speed: float,
    damping: float,
    duration: float = 0.25,
    *,
    mode: CableMode = CableMode.RECORDING,
    break_threshold: float | None = None,
    integration_step: float = constants.TIME_STEP,
    discrete: bool = False,
    _legacy_advance_to_zero: bool = False,
) -> ImpactTrajectory:
    """Advance Drake at 1 ms from zero extension and prescribed relative speed."""
    if relative_speed <= 0.0:
        raise ValueError("relative speed must be positive")
    if integration_step <= 0.0:
        raise ValueError("integration step must be positive")
    model = build_two_body_cable_cell(
        damping,
        mode=mode,
        break_threshold=break_threshold,
        plant_time_step=integration_step if discrete else 0.0,
    )
    simulator = Simulator(model.diagram)
    root_context = simulator.get_mutable_context()
    plant_context = model.plant.GetMyMutableContextFromRoot(root_context)
    model.load_joint.set_translation(plant_context, 0.0)
    model.vessel_joint.set_translation(plant_context, constants.CABLE_REST_LENGTH)
    total_mass = constants.LOAD_MASS + constants.VESSEL_MASS
    model.load_joint.set_translation_rate(
        plant_context, -constants.VESSEL_MASS * relative_speed / total_mass
    )
    model.vessel_joint.set_translation_rate(
        plant_context, constants.LOAD_MASS * relative_speed / total_mass
    )
    if not discrete:
        integrator = simulator.get_mutable_integrator()
        integrator.set_maximum_step_size(integration_step)
        integrator.set_fixed_step_mode(True)
    simulator.Initialize()

    times = np.arange(0.0, duration + 0.5 * integration_step, integration_step)
    sample_count = times.size
    arrays = {
        name: np.empty(sample_count)
        for name in (
            "load_position",
            "vessel_position",
            "load_velocity",
            "vessel_velocity",
            "elongation",
            "elongation_rate",
            "tension",
            "load_force",
            "vessel_force",
            "kinetic_energy",
            "spring_potential",
            "alive",
        )
    }
    for sample_index, sample_time in enumerate(times):
        if sample_index or _legacy_advance_to_zero:
            simulator.AdvanceTo(float(sample_time))
        cable_context = model.cable.GetMyContextFromRoot(root_context)
        cable_sample = model.cable.sample(cable_context)
        applied = model.cable.get_output_port().Eval(cable_context)
        forces_by_body = {
            int(item.body_index): float(item.F_Bq_W.translational()[0])
            for item in applied
        }
        load_position = model.load_joint.get_translation(plant_context)
        vessel_position = model.vessel_joint.get_translation(plant_context)
        load_velocity = model.load_joint.get_translation_rate(plant_context)
        vessel_velocity = model.vessel_joint.get_translation_rate(plant_context)
        arrays["load_position"][sample_index] = load_position
        arrays["vessel_position"][sample_index] = vessel_position
        arrays["load_velocity"][sample_index] = load_velocity
        arrays["vessel_velocity"][sample_index] = vessel_velocity
        arrays["elongation"][sample_index] = cable_sample.elongation
        arrays["elongation_rate"][sample_index] = cable_sample.elongation_rate
        arrays["tension"][sample_index] = cable_sample.applied_tension
        arrays["load_force"][sample_index] = forces_by_body[model.load_body_index]
        arrays["vessel_force"][sample_index] = forces_by_body[model.vessel_body_index]
        arrays["kinetic_energy"][sample_index] = (
            0.5 * constants.LOAD_MASS * load_velocity**2
            + 0.5 * constants.VESSEL_MASS * vessel_velocity**2
        )
        arrays["spring_potential"][sample_index] = (
            0.5 * constants.CABLE_STIFFNESS * max(0.0, cable_sample.elongation) ** 2
        )
        arrays["alive"][sample_index] = float(cable_sample.alive)
    event_state = model.cable.event_state(cable_context)
    taut_records = (
        model.cable.drained_taut_records + event_state.taut_ring.snapshot()
    )
    reengagement_records = (
        model.cable.drained_reengagement_records
        + event_state.reengagement_ring.snapshot()
    )
    event_records = model.cable.drained_event_records + event_state.event_ring.snapshot()
    return ImpactTrajectory(
        time=times,
        taut_records=taut_records,
        reengagement_records=reengagement_records,
        event_records=event_records,
        drained_taut_record_count=len(model.cable.drained_taut_records),
        resident_taut_record_count=event_state.taut_ring.size,
        integration_step_s=integration_step,
        plant_time_step_s=model.plant.time_step(),
        **arrays,
    )


def _conservation_metrics(trajectory: ImpactTrajectory) -> dict[str, float]:
    load_delta_p = constants.LOAD_MASS * (
        trajectory.load_velocity[-1] - trajectory.load_velocity[0]
    )
    vessel_delta_p = constants.VESSEL_MASS * (
        trajectory.vessel_velocity[-1] - trajectory.vessel_velocity[0]
    )
    if trajectory.plant_time_step_s > 0.0:
        step = trajectory.plant_time_step_s
        load_interval_delta_p = constants.LOAD_MASS * np.diff(
            trajectory.load_velocity
        )
        vessel_interval_delta_p = constants.VESSEL_MASS * np.diff(
            trajectory.vessel_velocity
        )
        left_residual = np.concatenate(
            (
                load_interval_delta_p - step * trajectory.load_force[:-1],
                vessel_interval_delta_p - step * trajectory.vessel_force[:-1],
            )
        )
        right_residual = np.concatenate(
            (
                load_interval_delta_p - step * trajectory.load_force[1:],
                vessel_interval_delta_p - step * trajectory.vessel_force[1:],
            )
        )
        interval_scale = max(
            float(np.max(np.abs(load_interval_delta_p))),
            float(np.max(np.abs(vessel_interval_delta_p))),
            np.finfo(float).tiny,
        )
        left_normalized_residual = float(
            np.max(np.abs(left_residual)) / interval_scale
        )
        right_normalized_residual = float(
            np.max(np.abs(right_residual)) / interval_scale
        )
        if (
            left_normalized_residual <= 1.0e-9
            and left_normalized_residual < right_normalized_residual
        ):
            selected_per_step_residual = left_normalized_residual
            load_impulse = float(np.sum(trajectory.load_force[:-1]) * step)
            vessel_impulse = float(np.sum(trajectory.vessel_force[:-1]) * step)
            impulse_evaluation = (
                "left_endpoint_discrete_force_verified_from_state_update"
            )
        elif (
            right_normalized_residual <= 1.0e-9
            and right_normalized_residual < left_normalized_residual
        ):
            selected_per_step_residual = right_normalized_residual
            load_impulse = float(np.sum(trajectory.load_force[1:]) * step)
            vessel_impulse = float(np.sum(trajectory.vessel_force[1:]) * step)
            impulse_evaluation = (
                "right_endpoint_discrete_force_verified_from_state_update"
            )
        else:
            raise AssertionError(
                "neither endpoint force closes the discrete momentum update"
            )
    else:
        left_normalized_residual = float("nan")
        right_normalized_residual = float("nan")
        selected_per_step_residual = float("nan")
        load_impulse = float(np.trapezoid(trajectory.load_force, trajectory.time))
        vessel_impulse = float(np.trapezoid(trajectory.vessel_force, trajectory.time))
        impulse_evaluation = "trapezoidal_continuous_force"
    load_residual = float(load_delta_p - load_impulse)
    vessel_residual = float(vessel_delta_p - vessel_impulse)
    momentum = (
        constants.LOAD_MASS * trajectory.load_velocity
        + constants.VESSEL_MASS * trajectory.vessel_velocity
    )
    momentum_scale = max(
        np.max(np.abs(constants.LOAD_MASS * trajectory.load_velocity)),
        np.max(np.abs(constants.VESSEL_MASS * trajectory.vessel_velocity)),
        np.finfo(float).tiny,
    )
    impulse_scale = max(
        abs(load_delta_p),
        abs(vessel_delta_p),
        abs(load_impulse),
        abs(vessel_impulse),
        np.finfo(float).tiny,
    )
    energy = trajectory.mechanical_energy
    physical_energy_envelope = float(np.max(np.abs(energy / energy[0] - 1.0)))
    positive_force = np.flatnonzero(trajectory.tension > 0.0)
    release_candidates = (
        np.flatnonzero(trajectory.tension[positive_force[0] + 1 :] <= 0.0)
        if positive_force.size
        else np.array([], dtype=int)
    )
    release_index = (
        int(positive_force[0] + 1 + release_candidates[0])
        if release_candidates.size
        else None
    )
    metrics = {
        "maximum_physical_energy_relative_envelope": physical_energy_envelope,
        "maximum_energy_relative_drift": physical_energy_envelope,
        "post_release_endpoint_relative_energy_defect": float(
            abs(energy[-1] / energy[0] - 1.0)
        ),
        "first_release_time_s": (
            None if release_index is None else float(trajectory.time[release_index])
        ),
        "endpoint_is_post_release": bool(
            release_index is not None and release_index < trajectory.time.size - 1
        ),
        "load_delta_momentum_kg_mps": float(load_delta_p),
        "load_impulse_n_s": load_impulse,
        "load_momentum_impulse_residual_kg_mps": load_residual,
        "vessel_delta_momentum_kg_mps": float(vessel_delta_p),
        "vessel_impulse_n_s": vessel_impulse,
        "vessel_momentum_impulse_residual_kg_mps": vessel_residual,
        "body_impulse_evaluation": impulse_evaluation,
        "equal_opposite_integrated_impulse_mismatch": float(
            abs(load_impulse + vessel_impulse) / impulse_scale
        ),
        "maximum_normalized_body_residual": float(
            max(abs(load_residual), abs(vessel_residual)) / impulse_scale
        ),
        "maximum_total_momentum_drift_kg_mps": float(
            np.max(np.abs(momentum - momentum[0]))
        ),
        "normalized_total_momentum_drift": float(
            np.max(np.abs(momentum - momentum[0])) / momentum_scale
        ),
    }
    if trajectory.plant_time_step_s > 0.0:
        metrics.update(
            {
                "left_endpoint_per_step_normalized_momentum_residual": (
                    left_normalized_residual
                ),
                "right_endpoint_per_step_normalized_momentum_residual": (
                    right_normalized_residual
                ),
                "maximum_per_step_normalized_body_momentum_residual": (
                    selected_per_step_residual
                ),
            }
        )
        shadow_energy = energy - (
            0.5
            * trajectory.plant_time_step_s
            * constants.CABLE_STIFFNESS
            * np.maximum(trajectory.elongation, 0.0)
            * trajectory.elongation_rate
        )
        metrics["maximum_shadow_energy_relative_drift"] = float(
            np.max(np.abs(shadow_energy / energy[0] - 1.0))
        )
    return metrics


def _legacy_phase1r_conservation_metrics(
    trajectory: ImpactTrajectory,
) -> dict[str, float]:
    """Reproduce the immutable Phase 1R metric schema and sampling convention."""
    load_delta_p = constants.LOAD_MASS * (
        trajectory.load_velocity[-1] - trajectory.load_velocity[0]
    )
    vessel_delta_p = constants.VESSEL_MASS * (
        trajectory.vessel_velocity[-1] - trajectory.vessel_velocity[0]
    )
    if trajectory.plant_time_step_s > 0.0:
        load_impulse = float(
            np.sum(trajectory.load_force[1:]) * trajectory.plant_time_step_s
        )
        vessel_impulse = float(
            np.sum(trajectory.vessel_force[1:]) * trajectory.plant_time_step_s
        )
        impulse_evaluation = "right_endpoint_discrete_force"
    else:
        load_impulse = float(np.trapezoid(trajectory.load_force, trajectory.time))
        vessel_impulse = float(
            np.trapezoid(trajectory.vessel_force, trajectory.time)
        )
        impulse_evaluation = "trapezoidal_continuous_force"
    load_residual = float(load_delta_p - load_impulse)
    vessel_residual = float(vessel_delta_p - vessel_impulse)
    momentum = (
        constants.LOAD_MASS * trajectory.load_velocity
        + constants.VESSEL_MASS * trajectory.vessel_velocity
    )
    momentum_scale = max(
        np.max(np.abs(constants.LOAD_MASS * trajectory.load_velocity)),
        np.max(np.abs(constants.VESSEL_MASS * trajectory.vessel_velocity)),
        np.finfo(float).tiny,
    )
    impulse_scale = max(
        abs(load_delta_p),
        abs(vessel_delta_p),
        abs(load_impulse),
        abs(vessel_impulse),
        np.finfo(float).tiny,
    )
    energy = trajectory.mechanical_energy
    return {
        "maximum_energy_relative_drift": float(
            np.max(np.abs(energy / energy[0] - 1.0))
        ),
        "load_delta_momentum_kg_mps": float(load_delta_p),
        "load_impulse_n_s": load_impulse,
        "load_momentum_impulse_residual_kg_mps": load_residual,
        "vessel_delta_momentum_kg_mps": float(vessel_delta_p),
        "vessel_impulse_n_s": vessel_impulse,
        "vessel_momentum_impulse_residual_kg_mps": vessel_residual,
        "body_impulse_evaluation": impulse_evaluation,
        "maximum_normalized_body_residual": float(
            max(abs(load_residual), abs(vessel_residual)) / impulse_scale
        ),
        "maximum_total_momentum_drift_kg_mps": float(
            np.max(np.abs(momentum - momentum[0]))
        ),
        "normalized_total_momentum_drift": float(
            np.max(np.abs(momentum - momentum[0])) / momentum_scale
        ),
    }


def _initial_state_is_exact(
    trajectory: ImpactTrajectory, relative_speed: float
) -> bool:
    total_mass = constants.LOAD_MASS + constants.VESSEL_MASS
    return bool(
        trajectory.time[0] == 0.0
        and trajectory.load_position[0] == 0.0
        and trajectory.vessel_position[0] == constants.CABLE_REST_LENGTH
        and trajectory.load_velocity[0]
        == -constants.VESSEL_MASS * relative_speed / total_mass
        and trajectory.vessel_velocity[0]
        == constants.LOAD_MASS * relative_speed / total_mass
        and trajectory.elongation[0] == 0.0
        and trajectory.tension[0] == 0.0
    )


def _step_halving_order(coarse: float, fine: float) -> float | None:
    if not np.isfinite(coarse) or not np.isfinite(fine) or coarse <= 0.0 or fine <= 0.0:
        return None
    return float(np.log2(coarse / fine))


def _trajectory_evidence(prefix: str, trajectory: ImpactTrajectory) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_time_s": trajectory.time,
        f"{prefix}_load_position_m": trajectory.load_position,
        f"{prefix}_vessel_position_m": trajectory.vessel_position,
        f"{prefix}_load_velocity_mps": trajectory.load_velocity,
        f"{prefix}_vessel_velocity_mps": trajectory.vessel_velocity,
        f"{prefix}_elongation_m": trajectory.elongation,
        f"{prefix}_elongation_rate_mps": trajectory.elongation_rate,
        f"{prefix}_tension_n": trajectory.tension,
        f"{prefix}_load_force_n": trajectory.load_force,
        f"{prefix}_vessel_force_n": trajectory.vessel_force,
        f"{prefix}_kinetic_energy_j": trajectory.kinetic_energy,
        f"{prefix}_spring_potential_j": trajectory.spring_potential,
    }


def mechanics_acceptance(
    *,
    include_evidence: bool = False,
    _legacy_phase1r_sampling: bool = False,
) -> dict[str, object] | tuple[dict[str, object], dict[str, np.ndarray]]:
    """Measure the P1-T2, P1-T3, and P1-T9 acceptance quantities."""
    impact_trajectories = tuple(
        run_two_body_impact(speed, constants.CABLE_DAMPING)
        for speed in IMPACT_SPEEDS
    )
    peaks = np.array(
        [trajectory.first_peak_tension for trajectory in impact_trajectories]
    )
    impedance = float(np.dot(IMPACT_SPEEDS, peaks) / np.dot(IMPACT_SPEEDS, IMPACT_SPEEDS))
    residual = peaks - impedance * IMPACT_SPEEDS
    uncentred_r_squared = float(1.0 - np.dot(residual, residual) / np.dot(peaks, peaks))
    reduced_mass_estimate = impedance**2 / (
        constants.CABLE_STIFFNESS * IMPACT_FACTOR**2
    )
    reduced_mass = (
        constants.VESSEL_MASS
        * constants.LOAD_MASS
        / (constants.VESSEL_MASS + constants.LOAD_MASS)
    )

    refinement = []
    refinement_trajectories = []
    metric_function = (
        _legacy_phase1r_conservation_metrics
        if _legacy_phase1r_sampling
        else _conservation_metrics
    )
    for step in REFINEMENT_STEPS:
        trajectory = run_two_body_impact(1.0, 0.0, integration_step=step)
        refinement_trajectories.append(trajectory)
        metrics = metric_function(trajectory)
        metrics["step_s"] = step
        metrics["integrator"] = "continuous_fixed_step_rk3"
        refinement.append(metrics)
    production_trajectory = run_two_body_impact(
        1.0,
        0.0,
        integration_step=constants.TIME_STEP,
        discrete=True,
        _legacy_advance_to_zero=_legacy_phase1r_sampling,
    )
    production = metric_function(production_trajectory)
    production["step_s"] = constants.TIME_STEP
    production["integrator"] = "discrete_sap"
    production["passed"] = bool(
        production["maximum_energy_relative_drift"] <= 0.005
        and production["maximum_normalized_body_residual"] <= 0.005
        and production["normalized_total_momentum_drift"] <= 0.005
    )
    refinement_passed = all(
        result["maximum_energy_relative_drift"] <= 0.005
        and result["maximum_normalized_body_residual"] <= 0.005
        and result["normalized_total_momentum_drift"] <= 0.005
        for result in refinement
    )

    tests = {
        "P1-T2": {
            "speeds_mps": IMPACT_SPEEDS.tolist(),
            "first_peaks_n": peaks.tolist(),
            "impedance_n_s_per_m": impedance,
            "pinned_impedance_n_s_per_m": PINNED_IMPEDANCE,
            "analytic_impedance_n_s_per_m": ANALYTIC_IMPEDANCE,
            "relative_error": abs(impedance / ANALYTIC_IMPEDANCE - 1.0),
            "uncentred_r_squared": uncentred_r_squared,
            "passed": uncentred_r_squared > 0.99
            and abs(impedance / ANALYTIC_IMPEDANCE - 1.0) <= 0.15,
        },
        "P1-T3": {
            "estimated_reduced_mass_kg": reduced_mass_estimate,
            "analytic_reduced_mass_kg": reduced_mass,
            "relative_error": abs(reduced_mass_estimate / reduced_mass - 1.0),
            "estimation_residual_kg": reduced_mass_estimate - reduced_mass,
            "rotational_excess_kg": 0.0,
            "passed": abs(reduced_mass_estimate / reduced_mass - 1.0) <= 0.10,
        },
        "P1-T9": {
            "continuous_refinement": refinement,
            "continuous_refinement_passed": refinement_passed,
            "production_discrete_1ms": production,
            "maximum_energy_relative_drift": production[
                "maximum_energy_relative_drift"
            ],
            "normalized_impulse_mismatch": production[
                "maximum_normalized_body_residual"
            ],
            "blocking_failure_basis": (
                "discrete_1ms_energy"
                if production["maximum_energy_relative_drift"] > 0.005
                else None
            ),
            "passed": bool(refinement_passed and production["passed"]),
        },
    }
    if not include_evidence:
        return tests
    evidence = {
        "mechanics_impact_speeds_mps": IMPACT_SPEEDS,
        "mechanics_impact_tension_n": np.stack(
            [trajectory.tension for trajectory in impact_trajectories]
        ),
        "mechanics_impact_elongation_m": np.stack(
            [trajectory.elongation for trajectory in impact_trajectories]
        ),
        "mechanics_impact_time_s": impact_trajectories[0].time,
    }
    for index, trajectory in enumerate(refinement_trajectories):
        evidence.update(_trajectory_evidence(f"t9_refinement_{index}", trajectory))
    evidence.update(_trajectory_evidence("t9_production_discrete_1ms", production_trajectory))
    return tests, evidence


def p1_t9r_acceptance(
    *,
    include_evidence: bool = False,
) -> dict[str, object] | tuple[dict[str, object], dict[str, np.ndarray]]:
    """Evaluate the deterministic P1-T9R SAP physics-step remediation gate."""
    step_results = []
    discrete_trajectories: dict[tuple[int, int], ImpactTrajectory] = {}
    for step_index, step in enumerate(REFINEMENT_STEPS):
        speed_results = []
        for speed_index, speed in enumerate(IMPACT_SPEEDS):
            trajectory = run_two_body_impact(
                float(speed),
                0.0,
                integration_step=step,
                discrete=True,
            )
            discrete_trajectories[(step_index, speed_index)] = trajectory
            metrics = _conservation_metrics(trajectory)
            metrics.update(
                {
                    "impact_speed_mps": float(speed),
                    "initial_state_exact": _initial_state_is_exact(
                        trajectory, float(speed)
                    ),
                    "step_s": step,
                }
            )
            speed_results.append(metrics)
        maximum_envelope = max(
            row["maximum_physical_energy_relative_envelope"]
            for row in speed_results
        )
        maximum_per_step_residual = max(
            row["maximum_per_step_normalized_body_momentum_residual"]
            for row in speed_results
        )
        maximum_impulse_mismatch = max(
            row["equal_opposite_integrated_impulse_mismatch"]
            for row in speed_results
        )
        maximum_momentum_drift = max(
            row["normalized_total_momentum_drift"] for row in speed_results
        )
        passes_numerically = bool(
            all(row["initial_state_exact"] for row in speed_results)
            and maximum_envelope <= P1_T9R_ENERGY_LIMIT
            and maximum_per_step_residual <= P1_T9R_MOMENTUM_LIMIT
            and maximum_impulse_mismatch <= P1_T9R_MOMENTUM_LIMIT
            and maximum_momentum_drift <= P1_T9R_MOMENTUM_LIMIT
        )
        step_results.append(
            {
                "step_s": step,
                "speed_results": speed_results,
                "maximum_physical_energy_relative_envelope": maximum_envelope,
                "energy_margin_to_limit": P1_T9R_ENERGY_LIMIT - maximum_envelope,
                "maximum_per_step_normalized_body_momentum_residual": (
                    maximum_per_step_residual
                ),
                "maximum_equal_opposite_integrated_impulse_mismatch": (
                    maximum_impulse_mismatch
                ),
                "maximum_normalized_total_momentum_drift": maximum_momentum_drift,
                "passes_numerically": passes_numerically,
            }
        )

    convergence = []
    for speed_index, speed in enumerate(IMPACT_SPEEDS):
        envelopes = [
            result["speed_results"][speed_index][
                "maximum_physical_energy_relative_envelope"
            ]
            for result in step_results
        ]
        orders = [
            _step_halving_order(envelopes[0], envelopes[1]),
            _step_halving_order(envelopes[1], envelopes[2]),
        ]
        orders_defined = all(order is not None for order in orders)
        order_bounds_pass = bool(
            orders_defined
            and all(
                P1_T9R_ORDER_BOUNDS[0] <= order <= P1_T9R_ORDER_BOUNDS[1]
                for order in orders
                if order is not None
            )
        )
        convergence.append(
            {
                "impact_speed_mps": float(speed),
                "physical_energy_envelopes": envelopes,
                "monotone_decrease": bool(envelopes[0] > envelopes[1] > envelopes[2]),
                "coarse_to_medium_order": orders[0],
                "medium_to_fine_order": orders[1],
                "orders_defined": orders_defined,
                "orders_within_bounds": order_bounds_pass,
            }
        )

    continuous_controls = []
    continuous_trajectories = []
    for step in REFINEMENT_STEPS:
        trajectory = run_two_body_impact(1.0, 0.0, integration_step=step)
        continuous_trajectories.append(trajectory)
        metrics = _conservation_metrics(trajectory)
        metrics.update({"step_s": step, "integrator": "continuous_fixed_step_rk3"})
        continuous_controls.append(metrics)
    continuous_controls_passed = all(
        row["maximum_physical_energy_relative_envelope"] <= P1_T9R_ENERGY_LIMIT
        and row["maximum_normalized_body_residual"] <= P1_T9R_ENERGY_LIMIT
        and row["normalized_total_momentum_drift"] <= P1_T9R_ENERGY_LIMIT
        for row in continuous_controls
    )

    candidate = step_results[REFINEMENT_STEPS.index(P1_T9R_CANDIDATE_STEP)]
    half_millisecond = step_results[REFINEMENT_STEPS.index(5.0e-4)]
    monotone_passed = all(row["monotone_decrease"] for row in convergence)
    convergence_passed = all(row["orders_within_bounds"] for row in convergence)
    passed = bool(
        candidate["passes_numerically"]
        and monotone_passed
        and convergence_passed
        and continuous_controls_passed
    )
    results = {
        "test_id": "P1-T9R",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "candidate_physics_step_s": P1_T9R_CANDIDATE_STEP,
        "production_event_logging_step_s": constants.TIME_STEP,
        "physical_energy_relative_limit": P1_T9R_ENERGY_LIMIT,
        "momentum_relative_limit": P1_T9R_MOMENTUM_LIMIT,
        "convergence_order_bounds": list(P1_T9R_ORDER_BOUNDS),
        "step_results": step_results,
        "convergence": convergence,
        "monotone_decrease_passed": monotone_passed,
        "convergence_orders_passed": convergence_passed,
        "continuous_rk3_controls": continuous_controls,
        "continuous_rk3_controls_passed": continuous_controls_passed,
        "half_millisecond_passes_numerically": half_millisecond[
            "passes_numerically"
        ],
        "half_millisecond_selected": False,
        "half_millisecond_selection_reason": "inadequate numerical margin",
        "candidate_passes_numerically": candidate["passes_numerically"],
        "gate_energy_quantity": "maximum_physical_energy_relative_envelope",
        "excluded_gate_energy_quantities": [
            "post_release_endpoint_relative_energy_defect",
            "maximum_shadow_energy_relative_drift",
            "continuous_rk3_energy",
        ],
    }
    if not include_evidence:
        return results
    evidence = {
        "p1_t9r_physics_steps_s": np.asarray(REFINEMENT_STEPS),
        "p1_t9r_impact_speeds_mps": IMPACT_SPEEDS,
    }
    for (step_index, speed_index), trajectory in discrete_trajectories.items():
        evidence.update(
            _trajectory_evidence(
                f"p1_t9r_discrete_s{step_index}_v{speed_index}", trajectory
            )
        )
    for step_index, trajectory in enumerate(continuous_trajectories):
        evidence.update(
            _trajectory_evidence(f"p1_t9r_continuous_s{step_index}", trajectory)
        )
    return results, evidence