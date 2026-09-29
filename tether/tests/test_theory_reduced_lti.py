"""Reduced linear model of the taut tow (plan IV.10; P2-T0 predictions)."""

from __future__ import annotations

import math
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydrake.systems.framework import DiagramBuilder
from scipy.linalg import solve_discrete_lyapunov

from tether.campaign.fleet_run import fan_heading_reference
from tether.control.controller import wrap_angle as controller_wrap_angle
from tether.physics import constants
from tether.physics import fleet
from tether.physics.sensors import cable_bearing
from tether.physics.weather import stationary_weather_forces
from tether.theory.reduced_lti import (
    AR1_COEFFICIENT,
    PAIR_REDUCED_MASS,
    ReducedModelInput,
    cable_terms,
    correlation_residual,
    generalized_forces,
    generalized_inertia,
    reduced_model,
    relative_load_map,
    steady_tow_state,
    to_full,
    to_reduced,
    tow_geometry,
    tow_operating_point,
    tow_plant,
    weather_covariance,
)

slow = pytest.mark.skipif(
    os.environ.get("TETHER_SLOW") != "1", reason="Drake validation runs; set TETHER_SLOW=1"
)

VALIDATION_RUNS = ((0.15, 1, 600.0), (0.15, 2, 600.0), (1.0, 1, 300.0))
VALIDATION_WARMUP = 20.0
COMPARED = ("sigma_e", "sigma_edot", "sigma_q", "sigma_w")
P2_T0_FACTOR = 1.5


@pytest.fixture(scope="module")
def nominal():
    return reduced_model(ReducedModelInput())


@pytest.fixture(scope="module")
def stiff_heading():
    return reduced_model(ReducedModelInput(heading_gain=1000.0))


def test_module_imports_no_drake():
    code = (
        "import sys, tether.theory.reduced_lti; "
        "sys.exit(any(name.split('.')[0] == 'pydrake' for name in sys.modules))"
    )
    root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])))
    completed = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, check=False)
    assert completed.returncode == 0


@pytest.mark.parametrize(
    "formation, arc_half_angle", [("parallel", None), ("fan", None), ("parallel", 0.3), ("fan", 0.45)]
)
def test_geometry_and_operating_point_equal_production_fleet(formation, arc_half_angle):
    ours = tow_geometry(formation, arc_half_angle=arc_half_angle)
    theirs = fleet.formation_geometry(formation, arc_half_angle=arc_half_angle)
    np.testing.assert_array_equal(ours.load_offsets, theirs.load_offsets)
    np.testing.assert_array_equal(ours.vessel_offsets, theirs.vessel_offsets)
    np.testing.assert_array_equal(ours.cable_angles, theirs.cable_angles)
    assert ours.arc_half_angle == theirs.arc_half_angle
    assert PAIR_REDUCED_MASS == fleet.PAIR_REDUCED_MASS
    for drag_law in ("linear", "quadratic"):
        for pretension in (600.0, 990.0, 1800.0):
            mine = tow_operating_point(ours, pretension, drag_law)
            reference = fleet.operating_point(theirs, pretension, drag_law)
            assert mine.speed == reference.speed
            np.testing.assert_array_equal(mine.thrusts, reference.thrusts)
            np.testing.assert_array_equal(mine.headings, reference.headings)
            np.testing.assert_array_equal(mine.tensions, reference.tensions)
            for heading in (0.0, 0.2):
                np.testing.assert_array_equal(
                    steady_tow_state(ours, mine, load_heading=heading),
                    fleet.equilibrium_state(theirs, reference, load_heading=heading),
                )


@pytest.mark.parametrize(
    "formation, drag_law, load_heading",
    [("parallel", "linear", 0.0), ("parallel", "quadratic", 0.1), ("fan", "linear", -0.05)],
)
def test_force_laws_equal_production_plant(formation, drag_law, load_heading):
    inp = ReducedModelInput(formation=formation, drag_law=drag_law)
    plant = tow_plant(inp)
    geometry = fleet.formation_geometry(formation)
    operating = fleet.operating_point(geometry, inp.pretension, drag_law)
    n = geometry.vessel_count
    size = 3 * (n + 1)
    generator = np.random.default_rng(np.random.SeedSequence([2026, 912, n]))
    state = fleet.equilibrium_state(geometry, operating, load_heading=load_heading)
    state[:size] += generator.uniform(-1.0e-3, 1.0e-3, size)
    state[size:] += generator.uniform(-0.03, 0.03, size)
    state[:2] += (40.0, -7.0)
    weather = generator.normal(0.0, 1000.0, size=(1, n + 1, 2))

    cables = fleet.UnilateralCableBank(geometry, fleet.CableParameters(), None, None, 0.0, log_state=False)
    cable_context = cables.CreateDefaultContext()
    cables.get_input_port(0).FixValue(cable_context, state)
    assert np.all(cables.get_output_port(1).Eval(cable_context) > 0.0)
    bearings = np.array([cable_bearing(state, geometry, i, constants.CABLE_REST_LENGTH) for i in range(n)])
    reference = fan_heading_reference(geometry, operating, fleet.CableParameters(), inp.heading_gain, inp.trim_gain)
    yaw = np.array(
        [
            inp.heading_gain * controller_wrap_angle(reference[i] - state[3 * (i + 1) + 2])
            - inp.trim_gain * math.sin(bearings[i])
            for i in range(n)
        ]
    )
    hull = fleet.HullForces(n, drag_law, weather, None)
    hull_context = hull.CreateDefaultContext()
    hull.get_input_port(0).FixValue(hull_context, state)
    hull.get_input_port(1).FixValue(hull_context, np.concatenate([operating.thrusts, yaw]))
    expected = cables.get_output_port(0).Eval(cable_context) + hull.get_output_port(0).Eval(hull_context)
    np.testing.assert_allclose(
        generalized_forces(plant, state, weather[0]), expected, rtol=0.0, atol=1.0e-9 * np.max(np.abs(expected))
    )

    terms = cable_terms(plant, state)
    kinematics = fleet.cable_kinematics(state, geometry, constants.CABLE_REST_LENGTH)
    np.testing.assert_allclose(terms["elongation"], kinematics["elongation"], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(terms["rate"], kinematics["rate"], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(terms["bearing"], bearings, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        relative_load_map(terms["direction"]) @ weather[0].ravel(),
        fleet.relative_gust_load(weather[0], terms["direction"]),
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(to_full(to_reduced(state, n), n, state[:2]), state, rtol=0.0, atol=1.0e-12)


def test_generalized_inertia_equals_drake_mass_matrix():
    geometry = fleet.formation_geometry("parallel")
    builder = DiagramBuilder()
    production = fleet.add_fleet_plant(builder, geometry, 990.0)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = production.plant.GetMyMutableContextFromRoot(context)
    state = fleet.equilibrium_state(geometry, production.operating, load_heading=0.4)
    production.plant.SetPositions(plant_context, state[: production.plant.num_positions()])
    mass_matrix = production.plant.CalcMassMatrix(plant_context)
    np.testing.assert_allclose(mass_matrix, np.diag(generalized_inertia(geometry.vessel_count)), atol=1.0e-9)


def test_equilibrium_is_exact_and_matches_design_state(nominal):
    geometry = fleet.formation_geometry("parallel")
    operating = fleet.operating_point(geometry, 990.0)
    design = to_reduced(fleet.equilibrium_state(geometry, operating), geometry.vessel_count)
    assert nominal.equilibrium_residual < 1.0e-10
    np.testing.assert_allclose(nominal.equilibrium, design, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(nominal.mu_e, 990.0 / constants.CABLE_STIFFNESS, rtol=1.0e-9)
    np.testing.assert_allclose(nominal.mu_q, 990.0, rtol=1.0e-9)
    np.testing.assert_allclose(nominal.chord_directions, np.tile([1.0, 0.0], (5, 1)), atol=1.0e-12)


@pytest.mark.parametrize(
    "formation, heading_gain, pretension, drag_law",
    [("parallel", 500.0, 990.0, "linear"), ("fan", 500.0, 990.0, "linear"), ("fan", 1000.0, 1400.0, "quadratic")],
)
def test_heading_reference_is_the_production_run_reference(formation, heading_gain, pretension, drag_law):
    inp = ReducedModelInput(formation=formation, heading_gain=heading_gain, pretension=pretension, drag_law=drag_law)
    geometry = fleet.formation_geometry(formation)
    operating = fleet.operating_point(geometry, pretension, drag_law)
    production = fan_heading_reference(geometry, operating, fleet.CableParameters(), heading_gain, inp.trim_gain)
    np.testing.assert_allclose(tow_plant(inp).heading_reference, production, rtol=0.0, atol=1.0e-14)


def test_fan_design_state_is_the_closed_loop_equilibrium():
    geometry = fleet.formation_geometry("fan")
    operating = fleet.operating_point(geometry, 990.0)
    design = to_reduced(fleet.equilibrium_state(geometry, operating), geometry.vessel_count)
    for heading_gain in (500.0, 1000.0):
        result = reduced_model(ReducedModelInput(formation="fan", heading_gain=heading_gain))
        assert result.equilibrium_residual < 1.0e-10
        np.testing.assert_allclose(result.equilibrium, design, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(result.mu_q, 990.0, rtol=1.0e-9)
        np.testing.assert_allclose(
            np.arctan2(result.chord_directions[:, 1], result.chord_directions[:, 0]), geometry.cable_angles, atol=1.0e-12
        )
        assert result.unstable_eigenvalues.size == 0
        for name in ("sigma_e", "sigma_q", "sigma_w"):
            values = getattr(result, name)
            assert np.all(np.isfinite(values) & (values > 0.0))
            np.testing.assert_allclose(values, values[::-1], rtol=1.0e-7)


def test_linearization_is_step_size_consistent(nominal):
    state_size = nominal.state_matrix.shape[0]
    assert nominal.linearization_discrepancy < 1.0e-6
    edot = nominal.outputs["edot"][:, :state_size]
    e = nominal.outputs["e"][:, :state_size]
    np.testing.assert_allclose(edot, e @ nominal.state_matrix, rtol=0.0, atol=1.0e-8 * np.max(np.abs(edot)))
    acceleration_rows = nominal.input_matrix[state_size - 18 :]
    np.testing.assert_allclose(
        np.abs(acceleration_rows).sum(axis=1),
        np.where(np.arange(18) % 3 == 2, 0.0, 1.0 / generalized_inertia(5)),
        rtol=1.0e-9,
    )


def test_lyapunov_residual_and_weather_block(nominal, stiff_heading):
    for result in (nominal, stiff_heading):
        assert result.lyapunov_residual < 1.0e-10
        assert result.spectral_radius < 1.0
        state_size = result.state_matrix.shape[0]
        np.testing.assert_allclose(
            result.covariance[state_size:, state_size:],
            weather_covariance(5, "local", 1.0),
            rtol=1.0e-8,
            atol=1.0e-8 * constants.LOAD_WEATHER_STD**2,
        )
    direct = solve_discrete_lyapunov(stiff_heading.transition, stiff_heading.innovation_covariance)
    scale = np.sqrt(np.diag(direct))
    np.testing.assert_allclose(stiff_heading.covariance / np.outer(scale, scale), direct / np.outer(scale, scale), atol=1.0e-9)
    assert AR1_COEFFICIENT == pytest.approx(0.998751, abs=1.0e-6)


def test_lyapunov_residual_sees_every_block(stiff_heading):
    transition = stiff_heading.transition
    innovation = stiff_heading.innovation_covariance
    covariance = stiff_heading.covariance
    residual = transition @ covariance @ transition.T + innovation - covariance
    assert stiff_heading.lyapunov_residual == pytest.approx(correlation_residual(residual, covariance), rel=1.0e-6, abs=1.0e-15)
    state_size = stiff_heading.state_matrix.shape[0]
    corrupted = covariance.copy()
    corrupted[:state_size, :state_size] *= 1.001
    residual = transition @ corrupted @ transition.T + innovation - corrupted
    assert np.max(np.abs(residual)) / np.max(np.abs(corrupted)) < 1.0e-10
    assert correlation_residual(residual, corrupted) > 1.0e-5 > 1.0e6 * stiff_heading.lyapunov_residual


def test_stability_structure(nominal, stiff_heading):
    assert stiff_heading.unstable_eigenvalues.size == 0
    assert np.max(np.linalg.eigvals(stiff_heading.state_matrix).real) < 0.0
    assert all(np.all(np.isfinite(getattr(stiff_heading, name))) for name in COMPARED + ("sigma_chord_angle",))

    unstable = nominal.unstable_eigenvalues
    assert unstable.size == 4
    np.testing.assert_allclose(unstable.imag, 0.0, atol=1.0e-9)
    assert np.all((unstable.real > 0.0) & (unstable.real < 1.0e-3))
    for name in ("e", "edot", "eddot", "q", "qdot", "w"):
        assert nominal.leakage[name] < 1.0e-10
    assert nominal.leakage["chord_angle"] > 1.0e-3
    assert np.all(np.isinf(nominal.sigma_chord_angle))
    direct = solve_discrete_lyapunov(nominal.transition, nominal.innovation_covariance)
    for name in ("e", "edot", "eddot", "q", "qdot"):
        output = nominal.outputs[name]
        np.testing.assert_allclose(
            np.diag(output @ direct @ output.T), np.diag(nominal.output_covariance(name)), rtol=1.0e-6
        )
    np.testing.assert_allclose(nominal.sigma_e, stiff_heading.sigma_e, rtol=1.0e-3)


def test_parallel_formation_is_left_right_symmetric(nominal):
    for name in ("sigma_e", "sigma_edot", "sigma_eddot", "sigma_q", "sigma_qdot", "sigma_w"):
        values = getattr(nominal, name)
        np.testing.assert_allclose(values, values[::-1], rtol=1.0e-8)
    np.testing.assert_allclose(nominal.corr_q, nominal.corr_q[::-1, ::-1], atol=1.0e-8)
    np.testing.assert_allclose(nominal.corr_w, nominal.corr_w[::-1, ::-1], atol=1.0e-12)


def test_variance_scales_with_weather_scale_squared(nominal):
    half = reduced_model(ReducedModelInput(weather_scale=0.5))
    np.testing.assert_allclose(half.covariance, 0.25 * nominal.covariance, rtol=0.0, atol=1.0e-9 * np.max(np.abs(nominal.covariance)))
    for name in ("sigma_e", "sigma_edot", "sigma_eddot", "sigma_q", "sigma_qdot", "sigma_w"):
        np.testing.assert_allclose(getattr(half, name), 0.5 * getattr(nominal, name), rtol=1.0e-9)
    np.testing.assert_allclose(half.corr_q, nominal.corr_q, atol=1.0e-9)
    np.testing.assert_allclose(half.integral_time_scale, nominal.integral_time_scale, rtol=1.0e-9)


def test_relative_load_statistics_are_exact(nominal):
    reduced = PAIR_REDUCED_MASS * math.hypot(
        constants.LOAD_WEATHER_STD / constants.LOAD_MASS, constants.VESSEL_WEATHER_STD / constants.VESSEL_MASS
    )
    np.testing.assert_allclose(nominal.sigma_w, reduced, rtol=1.0e-12)
    shared = PAIR_REDUCED_MASS**2 * (constants.LOAD_WEATHER_STD / constants.LOAD_MASS) ** 2 / reduced**2
    off_diagonal = nominal.corr_w[~np.eye(5, dtype=bool)]
    np.testing.assert_allclose(off_diagonal, shared, rtol=1.0e-12)
    common = reduced_model(ReducedModelInput(weather_direction="common"))
    np.testing.assert_allclose(common.corr_w, 1.0, rtol=1.0e-12)
    mixed = reduced_model(ReducedModelInput(weather_direction="mixed", weather_rho=0.0))
    np.testing.assert_allclose(mixed.sigma_q, nominal.sigma_q, rtol=1.0e-9)
    np.testing.assert_allclose(nominal.mu_w, 0.0)


@pytest.mark.parametrize(
    "direction, rho, front_angle",
    [("local", None, None), ("common", None, None), ("mixed", 0.3, None), ("front", None, -math.pi / 2), ("front", None, 0.7)],
)
def test_weather_covariance_matches_production_generator(direction, rho, front_angle):
    forces = stationary_weather_forces(4242, 1200.0, direction=direction, scale=0.8, rho=rho, front_angle=front_angle)
    flat = forces.reshape(forces.shape[0], -1)
    innovations = flat[1:] - AR1_COEFFICIENT * flat[:-1]
    measured = innovations.T @ innovations / innovations.shape[0] / (1.0 - AR1_COEFFICIENT**2)
    scale = np.sqrt(np.diag(weather_covariance(5, "local", 0.8)))
    normalizer = np.outer(scale, scale)
    np.testing.assert_allclose(
        measured / normalizer, weather_covariance(5, direction, 0.8, rho, front_angle) / normalizer, atol=0.04
    )


def test_directional_front_statistics_and_degenerate_outputs():
    head_on = reduced_model(ReducedModelInput(weather_direction="front", weather_front_angle=0.0))
    load_share = constants.LOAD_WEATHER_STD / constants.LOAD_MASS - constants.VESSEL_WEATHER_STD / constants.VESSEL_MASS
    np.testing.assert_allclose(head_on.sigma_w, PAIR_REDUCED_MASS * math.sqrt(2.0) * abs(load_share), rtol=1.0e-12)
    np.testing.assert_allclose(head_on.corr_w, 1.0, rtol=1.0e-12)
    for name in ("e", "edot", "q", "qdot", "w"):
        assert not np.any(head_on.degenerate[name])

    broadside = reduced_model(ReducedModelInput(weather_direction="front", weather_front_angle=-math.pi / 2))
    assert np.all(broadside.degenerate["w"]) and np.all(broadside.sigma_w == 0.0)
    assert np.all(np.isnan(broadside.corr_w))
    assert broadside.degenerate["q"].tolist() == [False, False, True, False, False]
    assert broadside.sigma_q[2] == 0.0 and broadside.sigma_e[2] == 0.0
    assert np.isnan(broadside.integral_time_scale[2]) and np.isnan(broadside.decorrelation_lag[2])
    assert np.all(np.isnan(broadside.corr_q[2])) and np.all(np.isnan(broadside.corr_q[:, 2]))
    assert np.all(np.isnan(broadside.autocorrelation("q", 3)[:, 2]))
    outer = [0, 1, 3, 4]
    assert np.all(broadside.sigma_q[outer] > 1.0) and np.all(np.isfinite(broadside.decorrelation_lag[outer]))
    np.testing.assert_allclose(broadside.sigma_q, broadside.sigma_q[::-1], rtol=1.0e-6)
    np.testing.assert_allclose(broadside.corr_q[0, 4], -1.0, atol=1.0e-6)

    fan = reduced_model(ReducedModelInput(formation="fan", weather_direction="front", weather_front_angle=math.pi / 2))
    assert fan.degenerate["q"].tolist() == [False, False, True, False, False]
    assert fan.degenerate["w"].tolist() == [False, False, True, False, False]
    assert np.all(fan.sigma_q[outer] > 1.0) and np.all(np.isfinite(fan.corr_q[np.ix_(outer, outer)]))


def test_rates_and_time_scales(nominal):
    covariance = nominal.covariance
    e = nominal.outputs["e"]
    edot = nominal.outputs["edot"]
    cross = np.diag(e @ covariance @ edot.T) / (nominal.sigma_e * nominal.sigma_edot)
    assert np.all(np.abs(cross) < 1.0e-2)
    identity = constants.CABLE_STIFFNESS**2 * nominal.sigma_edot**2 + constants.CABLE_DAMPING**2 * nominal.sigma_eddot**2
    np.testing.assert_allclose(nominal.sigma_qdot**2, identity, rtol=0.05)
    assert np.all((nominal.integral_time_scale > 0.0) & (nominal.integral_time_scale < 2.0 * constants.WEATHER_TIME_CONSTANT))
    assert np.all(nominal.decorrelation_lag > nominal.integral_time_scale)
    lags = np.rint(nominal.decorrelation_lag / constants.WEATHER_PERIOD).astype(int)
    autocorrelation = nominal.autocorrelation("q", int(lags.max()))
    np.testing.assert_allclose(autocorrelation[0], 1.0, rtol=1.0e-12)
    for cable, lag in enumerate(lags):
        assert autocorrelation[lag, cable] < 0.05 <= autocorrelation[lag - 1, cable]
    averaged = np.sqrt(np.diag(nominal.phase_averaged_covariance("e")))
    np.testing.assert_allclose(averaged, nominal.sigma_e, rtol=1.0e-6)


def _measure_drake_run(job: tuple[float, int, float]) -> dict:
    from tether.campaign.fleet_run import FleetRunSpec, build_run, end_time, run_to_end

    scale, seed, duration = job
    spec = FleetRunSpec(pretension=990.0, weather_scale=scale, duration=duration, warmup=VALIDATION_WARMUP)
    run = run_to_end(build_run(spec, seed))
    log = run.fleet.cables.log
    keep = log.event_time[: log.count] >= VALIDATION_WARMUP
    elongation = log.elongation[: log.count][keep]
    rate = log.rate[: log.count][keep]
    acceleration = np.gradient(rate, fleet.EVENT_PERIOD, axis=0)
    tension = constants.CABLE_STIFFNESS * elongation + constants.CABLE_DAMPING * rate
    tension_rate = constants.CABLE_STIFFNESS * rate + constants.CABLE_DAMPING * acceleration
    plant = tow_plant(ReducedModelInput())
    states = log.state[: log.state_count][log.state_time[: log.state_count] >= VALIDATION_WARMUP]
    chord = np.degrees(np.array([cable_terms(plant, state)["chord_angle"] for state in states]))
    return {
        "scale": scale,
        "seed": seed,
        "end_time": end_time(run),
        "requested_end": spec.warmup + spec.duration,
        "slack_fraction": np.mean(elongation <= 0.0, axis=0),
        "mu_e": elongation.mean(axis=0),
        "sigma_e": elongation.std(axis=0),
        "sigma_edot": rate.std(axis=0),
        "sigma_eddot": acceleration.std(axis=0),
        "sigma_q": tension.std(axis=0),
        "sigma_qdot": tension_rate.std(axis=0),
        "sigma_w": log.relative_load[: log.count][keep].std(axis=0),
        "chord_std_deg": chord.std(axis=0),
        "chord_max_deg": np.abs(chord).max(axis=0),
    }


def validate_against_drake(runs=VALIDATION_RUNS, processes: int | None = None) -> list[dict]:
    """Run the production plant and pair each run with the reduced model at its scale."""
    with multiprocessing.get_context("spawn").Pool(processes or len(runs)) as pool:
        measured = pool.map(_measure_drake_run, runs)
    rows = []
    for record in measured:
        model = reduced_model(ReducedModelInput(weather_scale=record["scale"]))
        row = dict(record)
        for name in ("mu_e",) + COMPARED + ("sigma_eddot", "sigma_qdot"):
            row[f"model_{name}"] = getattr(model, name)
            row[f"ratio_{name}"] = getattr(model, name) / record[name]
        rows.append(row)
    return rows


def format_validation(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        lines.append(
            f"S={row['scale']} seed={row['seed']} end={row['end_time']:.1f}/{row['requested_end']:.0f} s "
            f"slack={np.array2string(row['slack_fraction'], precision=4)} "
            f"chord std deg={np.array2string(row['chord_std_deg'], precision=1)}"
        )
        for name in ("mu_e",) + COMPARED + ("sigma_eddot", "sigma_qdot"):
            lines.append(
                f"  {name:12s} measured={np.array2string(row[name], precision=5)} "
                f"model={np.array2string(row['model_' + name], precision=5)} "
                f"model/measured={np.array2string(row['ratio_' + name], precision=3)}"
            )
    return "\n".join(lines)


@slow
def test_reduced_model_against_drake():
    rows = validate_against_drake()
    print(format_validation(rows))
    for row in rows:
        if row["scale"] > 0.5:
            continue
        assert row["end_time"] == pytest.approx(row["requested_end"])
        assert np.all(row["slack_fraction"] == 0.0)
        for name in COMPARED:
            ratio = row[f"ratio_{name}"]
            assert np.all((ratio > 1.0 / P2_T0_FACTOR) & (ratio < P2_T0_FACTOR)), name


if __name__ == "__main__":
    print(format_validation(validate_against_drake()))
