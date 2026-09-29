"""Regression tests for the production fleet plant (Phase 1(d) onward)."""

import math

import numpy as np
import pytest
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.tree import MultibodyForces
from pydrake.systems.framework import DiagramBuilder, LeafSystem

from tether.campaign.fleet_run import FleetRunSpec, build_run, run_to_end
from tether.control.controller import ControllerBank
from tether.estimation.truth_isolation import lint_diagram
from tether.physics import constants
from tether.physics.fleet import (
    ConstantCommand,
    add_fleet_plant,
    cable_kinematics,
    equilibrium_state,
    fast_cable_forces,
    fast_cable_terms,
    formation_geometry,
    operating_point,
    plant_state,
    scalar_cable_step,
)
from tether.physics.sensors import SensorSuite
from tether.physics.weather import stationary_weather_forces


def _random_state(rng, geometry):
    state = rng.normal(size=6 * (geometry.vessel_count + 1))
    state[3 : 3 * (geometry.vessel_count + 1) : 3] += 15.0
    return state


@pytest.mark.parametrize("pretension", [742.5, 990.0, 1237.5])
@pytest.mark.parametrize("closed_loop", [False, True])
def test_steady_state_identity(pretension, closed_loop):
    spec = FleetRunSpec(pretension=pretension, weather_scale=0.0, duration=2.0, warmup=0.0, closed_loop=closed_loop)
    run = run_to_end(build_run(spec, 3))
    state = plant_state(run.fleet, run.simulator.get_context())
    kinematics = cable_kinematics(state, run.fleet.geometry, constants.CABLE_REST_LENGTH)
    tensions = constants.CABLE_STIFFNESS * kinematics["elongation"]
    speed = pretension * constants.VESSEL_COUNT / constants.LOAD_LINEAR_DRAG
    assert np.allclose(state[18::3], speed, rtol=2.0e-4)
    assert np.allclose(tensions, pretension, rtol=2.0e-3)


def test_operating_point_matches_plan_relations():
    point = operating_point(formation_geometry("parallel"), 990.0)
    assert point.speed == pytest.approx(0.9)
    assert np.allclose(point.thrusts, 1305.0)


def test_attachment_rate_matches_central_difference():
    rng = np.random.default_rng(4)
    for kind in ("parallel", "fan"):
        geometry = formation_geometry(kind, arc_half_angle=0.5 if kind == "fan" else None)
        state = _random_state(rng, geometry)
        count = 3 * (geometry.vessel_count + 1)
        step = 1.0e-6
        forward = state.copy()
        backward = state.copy()
        forward[:count] += step * state[count:]
        backward[:count] -= step * state[count:]
        numeric = (
            cable_kinematics(forward, geometry, 12.0)["length"] - cable_kinematics(backward, geometry, 12.0)["length"]
        ) / (2.0 * step)
        assert np.allclose(cable_kinematics(state, geometry, 12.0)["rate"], numeric, atol=1.0e-6)


def test_scalar_hot_path_matches_vectorized_reference():
    rng = np.random.default_rng(5)
    geometry = formation_geometry("fan", arc_half_angle=0.6)
    lists = tuple(array.tolist() for array in (geometry.load_offsets[:, 0], geometry.load_offsets[:, 1], geometry.vessel_offsets[:, 0], geometry.vessel_offsets[:, 1]))
    for _ in range(100):
        state = _random_state(rng, geometry)
        alive = (rng.uniform(size=5) > 0.2).astype(float)
        terms = fast_cable_terms(state, geometry, 12.0)
        tensions = np.where((terms[6] > 0) & (alive > 0.5), np.maximum(1.7e5 * terms[6] + 1.8e3 * terms[7], 0.0), 0.0)
        generalized = scalar_cable_step(state.tolist(), lists, 12.0, 1.7e5, 1.8e3, alive.tolist())[0]
        assert np.allclose(generalized, fast_cable_forces(terms, tensions, 5), atol=1.0e-7)


def test_generalized_forces_match_drake_spatial_forces():
    """The generalized-force path equals Drake's own map of point forces."""
    rng = np.random.default_rng(6)
    geometry = formation_geometry("fan", arc_half_angle=0.6)
    builder = DiagramBuilder()
    fleet = add_fleet_plant(builder, geometry, 990.0)
    command = builder.AddSystem(ConstantCommand(np.zeros(10)))
    builder.Connect(command.get_output_port(0), fleet.hull.get_input_port(1))
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant = fleet.plant
    plant_context = plant.GetMyMutableContextFromRoot(context)
    state = _random_state(rng, geometry)
    plant.SetPositions(plant_context, state[:18])
    plant.SetVelocities(plant_context, state[18:])
    kinematics = cable_kinematics(state, geometry, 12.0)
    tensions = rng.uniform(100.0, 5000.0, size=5)
    forces = MultibodyForces(plant)
    world = plant.world_frame()
    load = plant.GetBodyByName("load")
    for i in range(5):
        vessel = plant.GetBodyByName(f"vessel_{i}")
        force = tensions[i] * kinematics["direction"][i]
        load_arm = np.append(kinematics["load_arm"][i], 0.0)
        vessel_arm = np.append(kinematics["vessel_arm"][i], 0.0)
        load.AddInForce(plant_context, load_arm, SpatialForce(np.zeros(3), np.append(force, 0.0)), world, forces)
        vessel.AddInForce(plant_context, vessel_arm, SpatialForce(np.zeros(3), np.append(-force, 0.0)), world, forces)
    drake_generalized = plant.CalcGeneralizedForces(plant_context, forces)
    terms = fast_cable_terms(state, geometry, 12.0)
    assert np.allclose(fast_cable_forces(terms, tensions, 5), drake_generalized, atol=1.0e-8)


def test_determinism_same_seed():
    spec = FleetRunSpec(pretension=990.0, weather_scale=1.0, duration=3.0, warmup=0.0)
    first = run_to_end(build_run(spec, 21))
    second = run_to_end(build_run(spec, 21))
    count = first.fleet.cables.log.count
    assert count == second.fleet.cables.log.count
    assert np.array_equal(first.fleet.cables.log.elongation[:count], second.fleet.cables.log.elongation[:count])
    assert np.array_equal(first.fleet.cables.log.rate[:count], second.fleet.cables.log.rate[:count])
    assert first.fleet.cables.reengagements == second.fleet.cables.reengagements
    third = run_to_end(build_run(spec, 22))
    assert not np.array_equal(first.fleet.cables.log.elongation[:count], third.fleet.cables.log.elongation[:count])


class _LeakyOdometry(LeafSystem):
    """Non-plant-side system that reads plant truth: the lint must flag it."""

    def __init__(self):
        super().__init__()
        self.DeclareVectorInputPort("plant_state", 36)
        self.DeclareVectorOutputPort("odometry", 4, lambda context, output: output.SetFromVector(np.zeros(4)))


def test_diagram_lint_passes_production_and_flags_truth_leak():
    run = build_run(FleetRunSpec(weather_scale=0.0, duration=1.0, warmup=0.0), 1)
    assert lint_diagram(run.fleet.diagram) == []
    geometry = formation_geometry("parallel")
    builder = DiagramBuilder()
    fleet = add_fleet_plant(builder, geometry, 990.0)
    point = fleet.operating
    controller = builder.AddSystem(
        ControllerBank(5, lambda t: point.headings, lambda t: point.thrusts, point.headings)
    )
    leak = builder.AddSystem(_LeakyOdometry())
    leak.set_name("leaky_odometry")
    builder.Connect(fleet.plant.get_state_output_port(), leak.get_input_port(0))
    for i in range(5):
        suite = builder.AddSystem(SensorSuite(geometry, i, 1, 12.0))
        builder.Connect(fleet.plant.get_state_output_port(), suite.get_input_port(0))
        builder.Connect(fleet.cables.get_output_port(1), suite.get_input_port(1))
        builder.Connect(suite.GetOutputPort("cable_bearing"), controller.get_input_port(5 + i))
        builder.Connect(leak.get_output_port(0), controller.get_input_port(i))
    builder.Connect(controller.get_output_port(0), fleet.hull.get_input_port(1))
    diagram = builder.Build()
    assert lint_diagram(diagram) == ["leaky_odometry[plant_state] <- plant[state]"]
    assert lint_diagram(diagram, exemptions={"leaky_odometry"}) == []


def test_fleet_impedance_regression_against_phase1_pin():
    """Axial impact on the centre cable reproduces the Phase 1 pinned fleet impedance."""
    geometry = formation_geometry("parallel")
    point = operating_point(geometry, 990.0)
    peaks = []
    speeds = np.array([1.0, 2.0, 3.0])
    for speed in speeds:
        spec = FleetRunSpec(pretension=990.0, weather_scale=0.0, duration=0.25, warmup=0.0, closed_loop=False)
        run = build_run(spec, 1)
        state = equilibrium_state(geometry, point)
        state[3 * 3] -= 990.0 / constants.CABLE_STIFFNESS
        total = constants.LOAD_MASS + 5 * constants.VESSEL_MASS
        base = 18
        state[base::3] += -constants.VESSEL_MASS / total * speed
        state[base + 9] += speed
        from tether.physics.fleet import set_state

        set_state(run.fleet, run.simulator.get_mutable_context(), state)
        run_to_end(run)
        log = run.fleet.cables.log
        q = 1.7e5 * log.elongation[: log.count, 2] + 1.8e3 * log.rate[: log.count, 2]
        q = np.where(log.elongation[: log.count, 2] > 0, q, 0.0)
        first_peak = next(q[i - 1] for i in range(1, q.size) if q[i] < q[i - 1] and q[i - 1] > 1000.0)
        peaks.append(first_peak)
    slope = float(np.dot(speeds, peaks) / np.dot(speeds, speeds))
    assert slope == pytest.approx(8_304.485, rel=0.03)


def test_stationary_weather_preroll_variance():
    samples = np.array([stationary_weather_forces(seed, 0.05, preroll=40.0)[0] for seed in range(400)])
    assert np.std(samples[:, 0, 0]) == pytest.approx(3500.0, rel=0.12)
    assert np.std(samples[:, 1:, :]) == pytest.approx(700.0, rel=0.06)


def test_sensor_streams_are_distinct_and_seeded():
    run = run_to_end(build_run(FleetRunSpec(weather_scale=0.0, duration=1.0, warmup=0.0), 9))
    context = run.simulator.get_context()
    readings = [suite.GetOutputPort("odometry").Eval(suite.GetMyContextFromRoot(context)) for suite in run.sensors]
    surges = [reading[0] for reading in readings]
    assert len(set(np.round(surges, 12))) == 5
    assert all(abs(surge - 0.9) < 0.2 for surge in surges)
    assert all(math.isclose(reading[3], 1.0, abs_tol=0.021) for reading in readings)
