"""Estimation core: SE(2), filter Jacobians, gating (P5-T0), comms, fusion, synthetic NEES."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from tether.estimation import comms, core, fusion, se2
from tether.estimation.core import (
    EstimatorGeometry,
    EstimatorParameters,
    NominalState,
    SensorLog,
    VesselFilter,
    retract,
)

RNG_ENTROPY = 20260912
STEP = 1.0e-3
REST_LENGTH = 12.0
STIFFNESS = 1.7e5
DAMPING = 1.8e3
SYNTHETIC_PSD = (0.01, 0.01, 1.0e-4)
SYNTHETIC_DURATION = 50.0
SYNTHETIC_BEACON_ON = 10.0
SYNTHETIC_SLACK = ((2, 12.0), (4, 16.0), (1, 21.0), (3, 25.0), (2, 29.0), (4, 33.0), (1, 38.0), (3, 42.0))
SLACK_VESSELS = (1, 2, 3, 4)
NEES_CORRELATION_TIME = 2.5
BAND_PROBABILITY = 0.99


def fan_geometry(vessel_count: int = 5, half_angle: float = 0.55) -> tuple[EstimatorGeometry, np.ndarray]:
    angles = np.linspace(-half_angle, half_angle, vessel_count)
    offsets = 3.5 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    stern = np.tile([-1.5, 0.0], (vessel_count, 1))
    return EstimatorGeometry(offsets, stern, REST_LENGTH, STIFFNESS, DAMPING), angles


def synthetic_scenario(
    seed: int,
    duration: float = SYNTHETIC_DURATION,
    slack: tuple = SYNTHETIC_SLACK,
    beacon_on: float = SYNTHETIC_BEACON_ON,
    psd: tuple = SYNTHETIC_PSD,
    dip_depth: float = 0.15,
    dip_width: float = 2.0,
) -> tuple[SensorLog, dict]:
    """Planar tow with exact kinematics: a load on an Ornstein-Uhlenbeck twist (30 s reversion,
    increments of the random-acceleration model with ``psd``), five vessels on 12 m chords
    whose angles, hull offsets, and lengths wander smoothly, scripted slack dips, and
    sensors drawn with the sensors.py noise model."""
    random = np.random.default_rng(np.random.SeedSequence([RNG_ENTROPY, seed, 1]))
    geometry, angles = fan_geometry()
    n = geometry.vessel_count
    steps = int(round(duration / STEP)) + 1
    t = np.arange(steps) * STEP
    reversion = 30.0
    phi = math.exp(-STEP / reversion)
    mean_twist = np.array([0.9, 0.0, 0.0])
    innovation = np.sqrt(np.asarray(psd) * reversion / 2.0 * (1.0 - phi**2))
    noise = random.standard_normal((steps, 3)) * innovation
    twist = np.empty((steps, 3))
    twist[0] = mean_twist
    for k in range(steps - 1):
        twist[k + 1] = mean_twist + (twist[k] - mean_twist) * phi + noise[k]
    pose = np.empty((steps, 3))
    pose[0] = 0.0
    for k in range(steps - 1):
        pose[k + 1] = se2.compose(pose[k], se2.exp(twist[k] * STEP))
    mu = 1000.0 / STIFFNESS
    e = np.empty((steps, n))
    edot = np.empty((steps, n))
    vessels = np.empty((steps, n, 3))
    vessel_twist = np.empty((steps, n, 3))
    cos0, sin0 = np.cos(pose[:, 2]), np.sin(pose[:, 2])
    world_velocity = np.stack([cos0 * twist[:, 0] - sin0 * twist[:, 1], sin0 * twist[:, 0] + cos0 * twist[:, 1]], axis=1)
    for i in range(n):
        period_chord, period_hull, period_length = 15.0 + 2.0 * i, 11.0 + i, 7.0 + i
        gamma = angles[i] + 0.03 * (1.0 - np.cos(2 * np.pi * t / period_chord))
        gamma_dot = 0.03 * 2 * np.pi / period_chord * np.sin(2 * np.pi * t / period_chord)
        psi = 0.05 * (1.0 - np.cos(2 * np.pi * t / period_hull))
        psi_dot = 0.05 * 2 * np.pi / period_hull * np.sin(2 * np.pi * t / period_hull)
        e[:, i] = mu + 1.0e-3 * (1.0 - np.cos(2 * np.pi * t / period_length))
        edot[:, i] = 1.0e-3 * 2 * np.pi / period_length * np.sin(2 * np.pi * t / period_length)
        for vessel, start in slack:
            if vessel != i:
                continue
            s = t - start
            inside = (s >= 0.0) & (s <= dip_width)
            e[inside, i] -= dip_depth * np.sin(np.pi * s[inside] / dip_width) ** 2
            edot[inside, i] -= dip_depth * np.pi / dip_width * np.sin(2 * np.pi * s[inside] / dip_width)
        r = geometry.load_offsets[i]
        arm = np.stack([cos0 * r[0] - sin0 * r[1], sin0 * r[0] + cos0 * r[1]], axis=1)
        load_point = pose[:, :2] + arm
        load_point_velocity = world_velocity + twist[:, 2:3] * np.stack([-arm[:, 1], arm[:, 0]], axis=1)
        direction = pose[:, 2] + gamma
        unit = np.stack([np.cos(direction), np.sin(direction)], axis=1)
        normal = np.stack([-unit[:, 1], unit[:, 0]], axis=1)
        length = REST_LENGTH + e[:, i]
        stern = load_point + length[:, None] * unit
        stern_velocity = (
            load_point_velocity + edot[:, i : i + 1] * unit + (length * (twist[:, 2] + gamma_dot))[:, None] * normal
        )
        heading = pose[:, 2] + gamma + psi
        omega = twist[:, 2] + gamma_dot + psi_dot
        ch, sh = np.cos(heading), np.sin(heading)
        s_off = geometry.stern_offsets[i]
        stern_arm = np.stack([ch * s_off[0] - sh * s_off[1], sh * s_off[0] + ch * s_off[1]], axis=1)
        position = stern - stern_arm
        velocity = stern_velocity - omega[:, None] * np.stack([-stern_arm[:, 1], stern_arm[:, 0]], axis=1)
        vessels[:, i, :2] = position
        vessels[:, i, 2] = heading
        vessel_twist[:, i] = np.column_stack([ch * velocity[:, 0] + sh * velocity[:, 1], -sh * velocity[:, 0] + ch * velocity[:, 1], omega])
    odometry, bearing, tension, surge_scales = [], [], [], []
    for i in range(n):
        sensor = np.random.default_rng(np.random.SeedSequence([RNG_ENTROPY, seed, i, 7]))
        scale = sensor.normal(0.0, core.SURGE_SCALE_STD)
        surge_scales.append(scale)
        index = np.arange(0, steps, 20)
        surge_true = vessel_twist[index, i, 0]
        surge = surge_true * (1.0 + scale) + sensor.standard_normal(index.size) * (
            core.SURGE_NOISE_BASE + core.SURGE_NOISE_SLOPE * np.abs(surge_true)
        )
        sway = vessel_twist[index, i, 1] + sensor.normal(0.0, core.SWAY_NOISE_STD, index.size)
        gyro = (
            vessel_twist[index, i, 2]
            + sensor.normal(0.0, core.GYRO_ARW_DENSITY / math.sqrt(core.ODOMETRY_PERIOD), index.size)
            + sensor.normal(0.0, core.GYRO_NOISE_STD, index.size)
        )
        odometry.append(np.column_stack([t[index], surge, sway, gyro]))
        index = np.arange(0, steps, 50)
        relative = vessels[index, i, 2] - pose[index, 2] - (angles[i] + 0.03 * (1.0 - np.cos(2 * np.pi * t[index] / (15.0 + 2.0 * i))))
        bearing.append(np.column_stack([t[index], relative + sensor.normal(0.0, core.BEARING_NOISE_STD, index.size)]))
        index = np.arange(0, steps, 20)
        taut = np.maximum(STIFFNESS * e[index, i] + DAMPING * edot[index, i], 0.0)
        measured = np.where(e[index, i] > 0.0, taut, 0.0) + sensor.normal(0.0, core.TENSION_NOISE_STD, index.size)
        tension.append(np.column_stack([t[index], measured, (measured < core.SLACK_FLAG_THRESHOLD).astype(float)]))
    beacon_sensor = np.random.default_rng(np.random.SeedSequence([RNG_ENTROPY, seed, 0, 8]))
    index = np.arange(int(round(beacon_on / STEP)), steps, 200)
    beacon = np.column_stack(
        [
            t[index],
            pose[index, 0] + beacon_sensor.normal(0.0, core.BEACON_POSITION_STD, index.size),
            pose[index, 1] + beacon_sensor.normal(0.0, core.BEACON_POSITION_STD, index.size),
            pose[index, 2] + beacon_sensor.normal(0.0, core.BEACON_ANGLE_STD, index.size),
        ]
    )
    log = SensorLog(tuple(odometry), tuple(bearing), tuple(tension), beacon, np.vstack([pose[0], vessels[0]]), geometry)
    truth = {"pose": pose, "twist": twist, "e": e, "edot": edot, "vessels": vessels, "surge_scale": np.array(surge_scales)}
    return log, truth


def truth_at(truth: dict, times: np.ndarray) -> dict:
    index = np.round(np.asarray(times) / STEP).astype(int)
    return {key: truth[key][index] for key in ("pose", "twist", "e", "edot")}


def pose_nees(output, truth: dict) -> np.ndarray:
    sampled = truth_at(truth, output.time)
    errors = sampled["pose"][:, None, :] - output.load_mean[..., :3]
    errors[..., 2] = (errors[..., 2] + math.pi) % (2.0 * math.pi) - math.pi
    return np.einsum("...i,...i->...", errors, np.linalg.solve(output.load_cov[..., :3, :3], errors[..., None])[..., 0])


def precursor_errors(output, truth: dict) -> tuple[np.ndarray, np.ndarray]:
    sampled = truth_at(truth, output.time)
    live = output.slack & np.isfinite(output.e_hat)
    errors = np.stack([output.e_hat - sampled["e"], output.edot_hat - sampled["edot"]], axis=-1)
    return errors, live


def precursor_nees(output, truth: dict) -> np.ndarray:
    errors, live = precursor_errors(output, truth)
    values = np.full(live.shape, np.nan)
    values[live] = np.einsum("ki,ki->k", errors[live], np.linalg.solve(output.sigma[live], errors[live][..., None])[..., 0])
    return values


def interval_means(output, values: np.ndarray) -> np.ndarray:
    means = []
    for vessel in range(output.slack.shape[1]):
        onsets = output.onset_time[:, vessel]
        for onset in np.unique(onsets[np.isfinite(onsets)]):
            selected = values[onsets == onset, vessel]
            means.append(float(np.nanmean(selected)))
    return np.asarray(means)


def mean_band(dof: int, samples: int) -> tuple[float, float]:
    """Two-sided band of the mean of ``samples`` independent chi-square(dof) variables."""
    tail = 0.5 * (1.0 - BAND_PROBABILITY)
    total = dof * samples
    return stats.chi2.ppf(tail, total) / samples, stats.chi2.ppf(1.0 - tail, total) / samples


def synthetic_parameters() -> EstimatorParameters:
    """The filter given the synthetic scenario's own random-acceleration model."""
    return EstimatorParameters(load_translation_psd=SYNTHETIC_PSD[0], load_yaw_psd=SYNTHETIC_PSD[2])


@pytest.fixture(scope="module")
def synthetic_run():
    """One synthetic log through P, B2, L; the slack vessels' chord variance per tick."""
    log, truth = synthetic_scenario(1)
    parameters = synthetic_parameters()
    estimators = [fusion.FleetEstimator(arm, log.geometry, log.initial_poses, 0.4, 1, parameters) for arm in fusion.ESTIMATOR_ARMS]
    history = {arm: {i: [] for i in SLACK_VESSELS} for arm in fusion.ESTIMATOR_ARMS}
    for tick, samples, beacon in fusion.log_ticks(log):
        for estimator in estimators:
            estimator.step(tick, samples, beacon)
            for i in SLACK_VESSELS:
                vessel = estimator.filters[i]
                vessel.propagate_to(tick * core.TICK)
                history[estimator.arm][i].append(
                    (tick * core.TICK, vessel.slack, vessel.chord_length_variance(), float(np.trace(vessel.covariance)))
                )
            if tick % fusion.MONITOR_PERIOD_TICKS == 0:
                estimator.record(tick * core.TICK)
    outputs = {estimator.arm: estimator.output() for estimator in estimators}
    return {"log": log, "truth": truth, "estimators": {e.arm: e for e in estimators}, "outputs": outputs, "history": history}


# SE(2)


def random_twists(count: int, scale: float, seed: int) -> np.ndarray:
    twists = np.random.default_rng(np.random.SeedSequence([RNG_ENTROPY, seed])).normal(size=(count, 3)) * scale
    twists[:, 2] = np.clip(twists[:, 2], -3.0, 3.0)
    return twists


@pytest.mark.parametrize("scale", [1.0e-8, 0.3, 1.5])
def test_se2_exp_log_round_trip(scale):
    for xi in random_twists(20, scale, 2):
        np.testing.assert_allclose(se2.log(se2.exp(xi)), xi, atol=1e-12)


def test_se2_compose_inverse_and_act():
    for g, h in zip(random_twists(20, 2.0, 3), random_twists(20, 2.0, 4)):
        np.testing.assert_allclose(se2.compose(g, se2.inverse(g)), np.zeros(3), atol=1e-12)
        point = h[:2]
        np.testing.assert_allclose(se2.act(se2.compose(g, h), point), se2.act(g, se2.act(h, point)), atol=1e-12)


def test_se2_adjoint_conjugates_exponential():
    for g, xi in zip(random_twists(20, 2.0, 5), random_twists(20, 0.7, 6)):
        conjugated = se2.compose(se2.compose(g, se2.exp(xi)), se2.inverse(g))
        np.testing.assert_allclose(se2.log(conjugated), se2.adjoint(g) @ xi, atol=1e-10)


@pytest.mark.parametrize("scale", [1.0e-9, 0.5, 1.2])
def test_se2_right_jacobian_matches_finite_differences(scale):
    epsilon = 1.0e-7
    for xi in random_twists(10, scale, 7):
        numeric = np.column_stack(
            [se2.log(se2.compose(se2.inverse(se2.exp(xi)), se2.exp(xi + epsilon * unit))) / epsilon for unit in np.eye(3)]
        )
        np.testing.assert_allclose(se2.right_jacobian(xi), numeric, atol=2e-6)
        np.testing.assert_allclose(se2.right_jacobian_inverse(xi) @ se2.right_jacobian(xi), np.eye(3), atol=1e-12)


def test_left_invariant_error_recovers_body_perturbation():
    for estimate, xi in zip(random_twists(20, 3.0, 8), random_twists(20, 0.5, 9)):
        truth = se2.compose(estimate, se2.exp(xi))
        np.testing.assert_allclose(se2.left_invariant_error(estimate, truth), xi, atol=1e-12)


@pytest.mark.parametrize("phi", [1.01e-6, 3.0e-6, 1.0e-5, 1.0e-4])
def test_se2_small_angle_precision(phi):
    """Just above the series switch ``1 - cos`` cancels (1e-4 relative at 1e-6 rad); the
    series to O(phi^4) is exact here, so the closed forms must match it on a 10 m arm."""
    rho = np.array([10.0, -7.0])
    half_cot = 1.0 - phi**2 / 12.0 - phi**4 / 720.0
    expected_log = np.array([half_cot * rho[0] + 0.5 * phi * rho[1], -0.5 * phi * rho[0] + half_cot * rho[1], phi])
    np.testing.assert_allclose(se2.log(np.array([*rho, phi])), expected_log, rtol=0.0, atol=1e-12)
    sinc = 1.0 - phi**2 / 6.0 + phi**4 / 120.0
    cosc = 0.5 * phi - phi**3 / 24.0 + phi**5 / 720.0
    expected_exp = np.array([sinc * rho[0] - cosc * rho[1], cosc * rho[0] + sinc * rho[1], phi])
    np.testing.assert_allclose(se2.exp(np.array([*rho, phi])), expected_exp, rtol=0.0, atol=1e-12)
    residual = phi / 6.0 - phi**3 / 120.0
    versine = 0.5 - phi**2 / 24.0 + phi**4 / 720.0
    expected_jacobian = np.array(
        [[sinc, cosc, residual * rho[0] - versine * rho[1]], [-cosc, sinc, versine * rho[0] + residual * rho[1]], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(se2.right_jacobian(np.array([*rho, phi])), expected_jacobian, rtol=0.0, atol=1e-8)


# Filter models and Jacobians


def nominal_states(count: int) -> list[tuple[NominalState, int]]:
    geometry, angles = fan_geometry()
    random = np.random.default_rng(np.random.SeedSequence([RNG_ENTROPY, 10]))
    states = []
    for k in range(count):
        agent = k % geometry.vessel_count
        load = np.array([random.normal(0, 20), random.normal(0, 20), random.uniform(-math.pi, math.pi)])
        chord = angles[agent] + load[2] + random.normal(0, 0.05)
        attachment = se2.act(load, geometry.load_offsets[agent])
        stern = attachment + (REST_LENGTH + random.normal(0, 0.05)) * np.array([math.cos(chord), math.sin(chord)])
        heading = chord + random.normal(0, 0.1)
        own = np.array([*(stern - se2.rotation(heading) @ geometry.stern_offsets[agent]), heading])
        twist = np.array([random.normal(0.9, 0.3), random.normal(0, 0.3), random.normal(0, 0.05)])
        states.append((NominalState(load, twist, own, random.normal(0, 0.01), random.normal(0, 1e-3)), agent))
    return states


def numeric_jacobian(function, state: NominalState, epsilon: float = 1.0e-7, wrap: bool = False) -> np.ndarray:
    columns = []
    for unit in np.eye(core.STATE_SIZE):
        plus = np.atleast_1d(function(retract(state, epsilon * unit)))
        minus = np.atleast_1d(function(retract(state, -epsilon * unit)))
        difference = plus - minus
        if wrap:
            difference = (difference + math.pi) % (2.0 * math.pi) - math.pi
        columns.append(difference / (2.0 * epsilon))
    return np.column_stack(columns)


def test_bearing_jacobian_matches_finite_differences():
    geometry, _ = fan_geometry()
    for state, agent in nominal_states(15):
        r, s = geometry.load_offsets[agent], geometry.stern_offsets[agent]
        _, analytic = core.bearing_model(state, r, s)
        numeric = numeric_jacobian(lambda x: core.bearing_model(x, r, s)[0], state, wrap=True)
        np.testing.assert_allclose(analytic, numeric[0], atol=1e-7)


def test_range_jacobian_matches_finite_differences():
    geometry, _ = fan_geometry()
    for state, agent in nominal_states(15):
        r, s = geometry.load_offsets[agent], geometry.stern_offsets[agent]
        _, analytic = core.range_model(state, r, s)
        numeric = numeric_jacobian(lambda x: core.range_model(x, r, s)[0], state)
        np.testing.assert_allclose(analytic, numeric[0], atol=1e-7)


def test_beacon_jacobian_matches_finite_differences():
    for state, _ in nominal_states(10):
        _, analytic = core.beacon_model(state)
        numeric = np.vstack(
            [numeric_jacobian(lambda x, k=k: core.beacon_model(x)[0][k], state, wrap=(k == 2))[0] for k in range(3)]
        )
        np.testing.assert_allclose(analytic, numeric, atol=1e-7)


def test_chord_rate_jacobians_match_finite_differences():
    geometry, _ = fan_geometry()
    random = np.random.default_rng(np.random.SeedSequence([RNG_ENTROPY, 11]))
    epsilon = 1.0e-7
    for state, agent in nominal_states(15):
        r, s = geometry.load_offsets[agent], geometry.stern_offsets[agent]
        odometry = np.array([random.normal(0.9, 0.2), random.normal(0, 0.1), random.normal(0, 0.05)])
        _, analytic, analytic_noise = core.chord_rate_model(state, odometry, r, s)
        numeric = numeric_jacobian(lambda x: core.chord_rate_model(x, odometry, r, s)[0], state)
        np.testing.assert_allclose(analytic, numeric[0], atol=1e-7)
        # The true twist is the measured one less the sample noise.
        noise_numeric = [
            (core.chord_rate_model(state, odometry - epsilon * unit, r, s)[0] - core.chord_rate_model(state, odometry + epsilon * unit, r, s)[0])
            / (2.0 * epsilon)
            for unit in np.eye(3)
        ]
        np.testing.assert_allclose(analytic_noise, noise_numeric, atol=1e-7)


def test_tension_chord_jacobians_match_finite_differences():
    geometry, _ = fan_geometry()
    relaxation = DAMPING / STIFFNESS
    odometry = np.array([0.85, -0.05, 0.02])
    epsilon = 1.0e-7
    for state, agent in nominal_states(10):
        r, s = geometry.load_offsets[agent], geometry.stern_offsets[agent]
        value, analytic, analytic_noise = core.tension_chord_model(state, odometry, r, s, relaxation)
        length, _ = core.range_model(state, r, s)
        rate, _, _ = core.chord_rate_model(state, odometry, r, s)
        assert value == pytest.approx(length + relaxation * rate)
        numeric = numeric_jacobian(lambda x: core.tension_chord_model(x, odometry, r, s, relaxation)[0], state)
        np.testing.assert_allclose(analytic, numeric[0], atol=1e-7)
        noise_numeric = [
            (
                core.tension_chord_model(state, odometry - epsilon * unit, r, s, relaxation)[0]
                - core.tension_chord_model(state, odometry + epsilon * unit, r, s, relaxation)[0]
            )
            / (2.0 * epsilon)
            for unit in np.eye(3)
        ]
        np.testing.assert_allclose(analytic_noise, noise_numeric, atol=5e-8)


def state_difference(truth: NominalState, estimate: NominalState) -> np.ndarray:
    return np.concatenate(
        [
            se2.left_invariant_error(estimate.load_pose, truth.load_pose),
            truth.load_twist - estimate.load_twist,
            se2.left_invariant_error(estimate.own_pose, truth.own_pose),
            [truth.surge_scale - estimate.surge_scale, truth.gyro_bias - estimate.gyro_bias],
        ]
    )


def held_filter(state: NominalState, agent: int, odometry: np.ndarray) -> VesselFilter:
    geometry, _ = fan_geometry()
    vessel = VesselFilter(agent, geometry, EstimatorParameters(), state.load_pose, state.own_pose)
    vessel.state = state
    vessel.odometry = odometry.copy()
    vessel.odometry_reference = odometry.copy()
    return vessel


def test_propagation_transition_matches_finite_differences():
    epsilon = 1.0e-6
    dt = 0.02
    odometry = np.array([0.8, 0.1, 0.03])
    for state, agent in nominal_states(6):
        nominal = held_filter(state, agent, odometry)
        transition, _ = nominal.propagation_matrices(dt)
        nominal.propagate_to(dt)
        numeric = []
        for unit in np.eye(core.STATE_SIZE):
            moved = []
            for sign in (1.0, -1.0):
                perturbed = held_filter(retract(state, sign * epsilon * unit), agent, odometry)
                perturbed.propagate_to(dt)
                moved.append(state_difference(perturbed.state, nominal.state))
            numeric.append((moved[0] - moved[1]) / (2.0 * epsilon))
        np.testing.assert_allclose(transition, np.column_stack(numeric), atol=2e-6)


@pytest.mark.parametrize("age", [0.1, 0.4, 1.6])
def test_fast_forward_transition_matches_finite_differences(age):
    """Load-block error transition over a packet age (up to the 1.6 s latency cell)."""
    epsilon = 1.0e-6
    random = np.random.default_rng(np.random.SeedSequence([RNG_ENTROPY, 15]))
    for _ in range(10):
        pose = random.normal(0.0, [20.0, 20.0, 2.0])
        twist = random.normal([0.9, 0.0, 0.0], [0.5, 0.3, 0.3])
        nominal = se2.compose(pose, se2.exp(twist * age))
        numeric = []
        for unit in np.eye(core.LOAD_SIZE):
            moved = []
            for sign in (1.0, -1.0):
                perturbed_pose = se2.compose(pose, se2.exp(sign * epsilon * unit[:3]))
                perturbed_twist = twist + sign * epsilon * unit[3:]
                moved_pose = se2.compose(perturbed_pose, se2.exp(perturbed_twist * age))
                moved.append(np.concatenate([se2.left_invariant_error(nominal, moved_pose), perturbed_twist - twist]))
            numeric.append((moved[0] - moved[1]) / (2.0 * epsilon))
        np.testing.assert_allclose(core.constant_twist_transition(twist, age), np.column_stack(numeric), rtol=0.0, atol=1e-7)


def test_sensor_constants_mirror_sensors_module():
    sensors = pytest.importorskip("tether.physics.sensors")
    for name in (
        "SURGE_SCALE_STD",
        "SURGE_NOISE_BASE",
        "SURGE_NOISE_SLOPE",
        "SWAY_NOISE_STD",
        "GYRO_NOISE_STD",
        "GYRO_ARW_DENSITY",
        "BEARING_NOISE_STD",
        "TENSION_NOISE_STD",
        "SLACK_FLAG_THRESHOLD",
        "BEACON_POSITION_STD",
        "BEACON_ANGLE_STD",
    ):
        assert getattr(core, name) == getattr(sensors, name), name
    assert core.ODOMETRY_PERIOD == sensors.SENSOR_PERIOD * sensors.ODOMETRY_DIVISOR
    assert core.BEARING_PERIOD == pytest.approx(sensors.SENSOR_PERIOD * sensors.BEARING_DIVISOR)
    assert core.TENSION_PERIOD == sensors.SENSOR_PERIOD * sensors.TENSION_DIVISOR
    assert core.BEACON_PERIOD == pytest.approx(sensors.SENSOR_PERIOD * sensors.BEACON_DIVISOR)


def test_core_modules_are_drake_free():
    import ast
    from pathlib import Path

    package = Path(core.__file__).parent
    for name in ("se2", "core", "comms", "fusion"):
        tree = ast.parse((package / f"{name}.py").read_text())
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        assert not any(module.startswith("pydrake") or module == "tether.physics.fleet" for module in imported), name


def test_process_noise_declarations():
    assert core.LOAD_TRANSLATION_PSD == pytest.approx(0.1407, rel=1e-3)
    assert core.formation_velocity_increment_rate(0.4) == pytest.approx(0.0466, rel=1e-3)
    parameters = EstimatorParameters(weather_scale=0.5)
    np.testing.assert_allclose(parameters.load_process_psd(), 0.25 * np.array([core.LOAD_TRANSLATION_PSD, core.LOAD_TRANSLATION_PSD, core.LOAD_YAW_PSD]))
    noise = core.random_acceleration_noise(np.array([1.0, 2.0, 3.0]), 0.4)
    assert noise[3, 3] == pytest.approx(0.4) and noise[0, 3] == pytest.approx(0.08) and noise[0, 0] == pytest.approx(0.064 / 3.0)


# Gating (P5-T0)


def debounced_intervals(history: list) -> list[tuple[int, int]]:
    intervals, start = [], None
    for index, (_, slack, _, _) in enumerate(history):
        if slack and start is None:
            start = index
        if not slack and start is not None:
            intervals.append((start, index))
            start = None
    return intervals


def test_slack_gating_removes_own_updates_and_covariance_grows(synthetic_run):
    for arm in fusion.ESTIMATOR_ARMS:
        estimator = synthetic_run["estimators"][arm]
        for vessel in SLACK_VESSELS:
            history = synthetic_run["history"][arm][vessel]
            intervals = debounced_intervals(history)
            scripted = sum(1 for index, _ in SYNTHETIC_SLACK if index == vessel)
            assert len(intervals) == scripted, (arm, vessel, intervals)
            log = estimator.filters[vessel].update_log
            for first, last in intervals:
                onset, release = history[first][0], history[last][0]
                own = [kind for time, kind in log if onset - 1e-9 <= time < release - 1e-9 and kind in ("bearing", "range")]
                assert not own, (arm, vessel, onset, own[:3])
                assert history[last - 1][2] > 4.0 * history[first][2], (arm, vessel, onset)
                if arm == "L":
                    traces = np.array([row[3] for row in history[first:last]])
                    assert np.all(np.diff(traces) >= -1e-15 * traces[:-1])
            taut_updates = [kind for time, kind in log if kind in ("bearing", "range")]
            assert len(taut_updates) > 0.9 * (70.0 * SYNTHETIC_DURATION - 70.0 * 2.2 * scripted)


def test_slack_state_debounces_spurious_clear_samples():
    geometry, _ = fan_geometry()
    vessel = VesselFilter(0, geometry, EstimatorParameters(), np.zeros(3), np.array([15.0, 0.0, 0.0]))
    vessel.process_tension(0.00, 0.0, 1.0)
    assert vessel.slack and vessel.slack_onset == 0.0
    for time, clear in ((0.02, 0.0), (0.04, 0.0), (0.06, 1.0), (0.08, 0.0), (0.10, 0.0)):
        vessel.process_tension(time, 60.0, clear)
    assert vessel.slack and not vessel.update_log
    vessel.process_tension(0.12, 900.0, 0.0)
    assert not vessel.slack and math.isnan(vessel.slack_onset) and vessel.update_log[-1][1] == "range"


# Comms


def test_cycle_and_complete_graphs():
    assert comms.graph_edges("cycle", 5) == ((0, 1), (0, 4), (1, 2), (1, 0), (2, 3), (2, 1), (3, 4), (3, 2), (4, 0), (4, 3))
    assert len(comms.graph_edges("complete", 5)) == 20
    assert comms.delay_ticks(0.4) == 40 and comms.delay_ticks(1.6) == 160


@pytest.mark.parametrize("delay", [0, 7, 40, 160])
def test_delay_ring_delivers_after_exact_tick_delay(delay):
    fabric = comms.PacketFabric(5, delay, master_seed=3)
    delivered = []
    for tick in range(400):
        if fabric.is_send_tick(tick):
            for source in range(5):
                fabric.send(tick, source, (tick, source))
        delivered += [(tick, source, destination, payload) for source, destination, payload in fabric.deliver(tick)]
    assert delivered
    for tick, source, destination, (sent, origin) in delivered:
        assert tick - sent == delay and origin == source and (source, destination) in fabric.edges
    expected = 10 * sum(1 for tick in range(400) if tick % 10 == 0 and tick + delay < 400)
    assert len(delivered) == expected and fabric.dropped == 0


def test_drop_gates_follow_their_seed_sequences():
    fabric = comms.PacketFabric(5, 4, master_seed=11, drop_probability=0.5)
    gates = {edge: comms.comms_generator(11, *edge) for edge in fabric.edges}
    kept = []
    for tick in range(0, 200, 10):
        for source in range(5):
            fabric.send(tick, source, (tick, source))
        kept += [(payload[0], source, destination) for source, destination, payload in fabric.deliver(tick + 4)]
    expected = [
        (tick, edge[0], edge[1])
        for tick in range(0, 200, 10)
        for source in range(5)
        for edge in fabric.edges
        if edge[0] == source and gates[edge].random() >= 0.5
    ]
    assert sorted(kept) == sorted(expected) and 0 < fabric.dropped < 200


# Fusion


def fusion_vessel(covariance_scale: float = 1.0) -> VesselFilter:
    geometry, _ = fan_geometry()
    vessel = VesselFilter(2, geometry, EstimatorParameters(), np.array([1.0, 2.0, 0.3]), np.array([16.0, 2.0, 0.3]), np.array([0.9, 0.0, 0.01]))
    random = np.random.default_rng(np.random.SeedSequence([RNG_ENTROPY, 12]))
    factor = random.normal(size=(core.STATE_SIZE, core.STATE_SIZE)) * 0.05
    vessel.covariance = covariance_scale * (factor @ factor.T + 1e-4 * np.eye(core.STATE_SIZE))
    return vessel


def test_fusing_an_identical_estimate():
    """CI of an estimate with itself changes nothing; information addition halves it."""
    for rule, expected in (("P", 1.0), ("B2", 0.5)):
        vessel = fusion_vessel()
        before = vessel.covariance.copy()
        pose, twist, load = vessel.load_packet()
        fusion.fusion_rule(rule, EstimatorParameters()).fuse(vessel, fusion.Packet(0.0, 1, pose, twist, load), 0.0)
        np.testing.assert_allclose(vessel.covariance[:6, :6], expected * before[:6, :6], rtol=2e-3, atol=1e-12)
        np.testing.assert_allclose(vessel.state.load_pose, pose, atol=1e-12)


def test_covariance_intersection_ignores_useless_and_adopts_precise_neighbours():
    vessel = fusion_vessel()
    before = vessel.covariance.copy()
    pose, twist, load = vessel.load_packet()
    rule = fusion.CovarianceIntersection(EstimatorParameters())
    rule.fuse(vessel, fusion.Packet(0.0, 1, pose + 0.5, twist, 1.0e4 * np.eye(6)), 0.0)
    assert rule.weights[-1] == 1.0
    np.testing.assert_array_equal(vessel.covariance, before)
    rule.fuse(vessel, fusion.Packet(0.0, 1, pose, twist, 1.0e-8 * np.eye(6)), 0.0)
    assert rule.weights[-1] < 0.01 and np.trace(vessel.covariance[:6, :6]) < 1e-5


def test_age_inflation_is_the_random_acceleration_model_over_the_age():
    parameters = EstimatorParameters(weather_scale=0.7)
    rule = fusion.CovarianceIntersection(parameters)
    for age in (0.1, 0.4, 1.6):
        inflation = rule.age_inflation(age)
        np.testing.assert_allclose(inflation, core.random_acceleration_noise(parameters.load_process_psd(), age))
        np.testing.assert_allclose(inflation[3:, 3:], age * np.diag(parameters.load_process_psd()))


def test_covariance_intersection_is_consistent_under_unknown_correlation():
    """Monte Carlo: the neighbour's load error is 0.8 x the receiver's plus an independent
    part; CI bounds the fused error covariance, the naive rule does not."""
    random = np.random.default_rng(np.random.SeedSequence([RNG_ENTROPY, 13]))
    base = fusion_vessel()
    own_covariance = base.covariance[:6, :6]
    independent = 0.1 * np.diag(np.diag(own_covariance))
    neighbour_covariance = 0.64 * own_covariance + independent
    own_factor = np.linalg.cholesky(own_covariance)
    independent_factor = np.linalg.cholesky(independent)
    truth_pose, truth_twist = base.state.load_pose.copy(), base.state.load_twist.copy()
    errors = {"P": [], "B2": []}
    claimed = {}
    for _ in range(500):
        own_error = own_factor @ random.standard_normal(6)
        neighbour_error = 0.8 * own_error + independent_factor @ random.standard_normal(6)
        for rule_name in errors:
            vessel = fusion_vessel()
            vessel.state = replace(
                vessel.state,
                load_pose=se2.compose(truth_pose, se2.exp(-own_error[:3])),
                load_twist=truth_twist - own_error[3:],
            )
            packet = fusion.Packet(
                0.0, 1, se2.compose(truth_pose, se2.exp(-neighbour_error[:3])), truth_twist - neighbour_error[3:], neighbour_covariance
            )
            fusion.fusion_rule(rule_name, EstimatorParameters()).fuse(vessel, packet, 0.0)
            errors[rule_name].append(
                np.concatenate([se2.left_invariant_error(vessel.state.load_pose, truth_pose), truth_twist - vessel.state.load_twist])
            )
            claimed[rule_name] = vessel.covariance[:6, :6]
    nees = {}
    for rule_name, values in errors.items():
        values = np.asarray(values)
        nees[rule_name] = float(np.mean(np.einsum("ki,ki->k", values, np.linalg.solve(claimed[rule_name], values.T).T)))
    assert nees["P"] < 6.0 * 1.2
    assert nees["B2"] > 6.0 * 1.5


def test_covariance_intersection_matches_the_ci_formula():
    """P's load block after fusion is ``(w P^-1 + (1 - w) R^-1)^-1`` with the mean by the same
    weights; the other states follow their conditional given the load block."""
    random = np.random.default_rng(np.random.SeedSequence([RNG_ENTROPY, 16]))
    for _ in range(5):
        vessel = fusion_vessel()
        before = vessel.covariance.copy()
        pose, twist, _ = vessel.load_packet()
        factor = random.normal(size=(core.LOAD_SIZE, core.LOAD_SIZE)) * 0.03
        neighbour_pose = se2.compose(pose, se2.exp(random.normal(0.0, [0.05, 0.05, 0.01])))
        neighbour_twist = twist + random.normal(0.0, 0.02, 3)
        packet = fusion.Packet(0.0, 1, neighbour_pose, neighbour_twist, factor @ factor.T + 1e-4 * np.eye(core.LOAD_SIZE))
        residual, _, noise = fusion.neighbour_measurement(vessel, neighbour_pose, neighbour_twist, packet.covariance)
        rule = fusion.CovarianceIntersection(EstimatorParameters())
        rule.fuse(vessel, packet, 0.0)
        weight = rule.weights[-1]
        assert 0.0 < weight < 1.0
        load = before[:6, :6]
        fused = np.linalg.inv(weight * np.linalg.inv(load) + (1.0 - weight) * np.linalg.inv(noise))
        np.testing.assert_allclose(vessel.covariance[:6, :6], fused, rtol=1e-9, atol=1e-15)
        moved = np.concatenate([se2.left_invariant_error(pose, vessel.state.load_pose), vessel.state.load_twist - twist])
        np.testing.assert_allclose(moved, fused @ ((1.0 - weight) * np.linalg.solve(noise, residual)), rtol=1e-9, atol=1e-13)
        lift = before[:, :6] @ np.linalg.inv(load)
        np.testing.assert_allclose(vessel.covariance, before - lift @ load @ lift.T + lift @ fused @ lift.T, rtol=1e-9, atol=1e-15)


def test_neighbour_measurement_noise_map_matches_finite_differences():
    """``xi_i = Log(g_i^-1 g_nb Exp(xi_nb))``: the neighbour's error enters through J_r(z)^-1."""
    epsilon = 1.0e-6
    vessel = fusion_vessel()
    pose, twist, _ = vessel.load_packet()
    neighbour = se2.compose(pose, se2.exp(np.array([0.4, -0.3, 0.25])))
    residual, jacobian, _ = fusion.neighbour_measurement(vessel, neighbour, twist + 0.1, np.eye(core.LOAD_SIZE))
    np.testing.assert_allclose(residual, np.concatenate([se2.left_invariant_error(pose, neighbour), np.full(3, 0.1)]), atol=1e-12)
    np.testing.assert_array_equal(jacobian, np.hstack([np.eye(6), np.zeros((6, 5))]))
    for k in range(core.LOAD_SIZE):
        unit = np.eye(core.LOAD_SIZE)[k]
        numeric = np.zeros(core.LOAD_SIZE)
        if k < 3:
            plus = se2.left_invariant_error(pose, se2.compose(neighbour, se2.exp(epsilon * unit[:3])))
            minus = se2.left_invariant_error(pose, se2.compose(neighbour, se2.exp(-epsilon * unit[:3])))
            numeric[:3] = (plus - minus) / (2.0 * epsilon)
        else:
            numeric[k] = 1.0
        _, _, noise = fusion.neighbour_measurement(vessel, neighbour, twist + 0.1, np.outer(unit, unit))
        np.testing.assert_allclose(noise, np.outer(numeric, numeric), atol=1e-8)


def test_intersection_weight_minimises_the_trace():
    vessel = fusion_vessel()
    full = vessel.covariance
    load = full[:6, :6]
    lift = np.linalg.solve(load, full[:6, :]).T
    noise = 0.5 * load + 0.01 * np.eye(6)
    weight = fusion.intersection_weight(load, noise, lift)
    trace = fusion.intersection_trace(load, noise, lift)
    grid = np.linspace(0.01, 1.0, 400)
    assert trace(weight) <= min(trace(w) for w in grid) + 1e-9
    fused = np.linalg.inv(weight * np.linalg.inv(load) + (1.0 - weight) * np.linalg.inv(noise))
    assert trace(weight) == pytest.approx(np.trace(lift @ fused @ lift.T), rel=1e-9)


# Precursor


def test_precursor_slope_window_fallback_and_floor():
    geometry, _ = fan_geometry()
    precursor = core.Precursor(geometry, EstimatorParameters(pretension=1000.0))
    precursor._times, precursor._rates = [0.0, 0.01], [-0.3, -0.29]
    assert precursor._slope(0.01) == (pytest.approx(1000.0 / core.EFFECTIVE_MASS), core.SLOPE_STD_FLOOR)
    times = list(np.arange(0.0, 0.5, 0.01))
    precursor._times, precursor._rates = times, [-0.4 + 1.5 * t for t in times]
    slope, error = precursor._slope(times[-1])
    assert slope == pytest.approx(1.5) and error == core.SLOPE_STD_FLOOR
    noisy = np.random.default_rng(np.random.SeedSequence([RNG_ENTROPY, 14])).normal(0.0, 0.2, len(times))
    precursor._rates = list(-0.4 + 1.5 * np.asarray(times) + noisy)
    window = np.asarray(times) >= times[-1] - core.SLOPE_WINDOW - 1e-9
    fit = stats.linregress(np.asarray(times)[window], np.asarray(precursor._rates)[window])
    slope, error = precursor._slope(times[-1])
    assert slope == pytest.approx(fit.slope) and error == pytest.approx(max(fit.stderr, core.SLOPE_STD_FLOOR))


def test_precursor_growth_gramian_and_offline_application(synthetic_run):
    output = synthetic_run["outputs"]["L"]
    grown = fusion.apply_precursor_growth(output, 0.3)
    live = output.slack
    age = output.time[:, None] - output.onset_time
    np.testing.assert_allclose((grown.sigma - output.sigma)[live][:, 1, 1], 0.3 * age[live])
    np.testing.assert_allclose((grown.sigma - output.sigma)[live][:, 0, 0], 0.3 * age[live] ** 3 / 3.0)
    back = fusion.apply_precursor_growth(grown, 0.0)
    np.testing.assert_allclose(back.sigma[live], output.sigma[live], atol=1e-15)


# Synthetic consistency


def test_output_layout(synthetic_run):
    output = synthetic_run["outputs"]["P"]
    m = int(round(SYNTHETIC_DURATION / 0.1)) + 1
    assert output.time.shape == (m,) and np.allclose(np.diff(output.time), 0.1)
    for name, shape in (
        ("slack", (m, 5)), ("onset_time", (m, 5)), ("e_hat", (m, 5)), ("edot_hat", (m, 5)), ("sigma", (m, 5, 2, 2)),
        ("a_hat", (m, 5)), ("sigma_a", (m, 5)), ("load_mean", (m, 5, 6)), ("load_cov", (m, 5, 6, 6)),
    ):
        assert getattr(output, name).shape == shape, name
    assert output.slack.dtype == bool
    assert np.all(np.isnan(output.onset_time[~output.slack])) and np.all(np.isfinite(output.onset_time[output.slack]))
    assert np.all(np.isfinite(output.sigma[output.slack])) and np.all(output.sigma_a[output.slack] >= core.SLOPE_STD_FLOOR)


@pytest.fixture(scope="module")
def no_slack_run():
    """The seed-1 scenario without scripted slack, arm P only."""
    log, truth = synthetic_scenario(1, slack=())
    return fusion.run_log(log, ("P",), 0.4, 1, synthetic_parameters())["P"], truth


@pytest.fixture(scope="module")
def held_out_local_run():
    """An independent seed with the scripted slack, arm L only (out-of-sample q_L check)."""
    log, truth = synthetic_scenario(3)
    return fusion.run_log(log, ("L",), 0.4, 1, synthetic_parameters())["L"], truth


def test_arm_p_load_pose_nees_is_consistent(synthetic_run):
    values = pose_nees(synthetic_run["outputs"]["P"], synthetic_run["truth"])
    low, high = mean_band(3, int(SYNTHETIC_DURATION / NEES_CORRELATION_TIME))
    assert low < float(values.mean()) < high, (values.mean(), low, high)


def test_arm_p_load_pose_nees_is_consistent_without_slack(no_slack_run):
    output, truth = no_slack_run
    assert not output.slack.any()
    values = pose_nees(output, truth)
    low, high = mean_band(3, int(SYNTHETIC_DURATION / NEES_CORRELATION_TIME))
    assert low < float(values.mean()) < high, (values.mean(), low, high)


def test_arm_p_precursor_nees_is_consistent(synthetic_run):
    output = synthetic_run["outputs"]["P"]
    means = interval_means(output, precursor_nees(output, synthetic_run["truth"]))
    assert means.size == len(SYNTHETIC_SLACK)
    low, high = mean_band(2, means.size)
    assert low < float(means.mean()) < high, (means, low, high)


def test_arm_b2_is_overconfident(synthetic_run):
    truth = synthetic_run["truth"]
    pose_p = pose_nees(synthetic_run["outputs"]["P"], truth).mean()
    pose_b2 = pose_nees(synthetic_run["outputs"]["B2"], truth).mean()
    assert pose_b2 > 100.0 * pose_p
    output = synthetic_run["outputs"]["B2"]
    means = interval_means(output, precursor_nees(output, truth))
    _, high = mean_band(2, means.size)
    p_means = interval_means(synthetic_run["outputs"]["P"], precursor_nees(synthetic_run["outputs"]["P"], truth))
    assert means.mean() > high and means.mean() > 3.0 * p_means.mean()


def calibrated_growth(output, truth: dict, target: float = 2.0) -> float:
    """q_L making the mean per-interval precursor NEES equal ``target`` (0 if already below)."""

    def mean_nees(growth: float) -> float:
        grown = fusion.apply_precursor_growth(output, growth)
        return float(interval_means(grown, precursor_nees(grown, truth)).mean())

    if mean_nees(0.0) <= target:
        return 0.0
    lower, upper = 0.0, 1.0
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        lower, upper = (middle, upper) if mean_nees(middle) > target else (lower, middle)
    return 0.5 * (lower + upper)


def test_arm_l_precursor_is_uninformative_but_consistent_after_growth(synthetic_run):
    truth = synthetic_run["truth"]
    proposed = synthetic_run["outputs"]["P"]
    local = synthetic_run["outputs"]["L"]
    local = fusion.apply_precursor_growth(local, calibrated_growth(local, truth))
    means = interval_means(local, precursor_nees(local, truth))
    low, high = mean_band(2, means.size)
    assert low < means.mean() < high
    spread, error = {}, {}
    for name, output in (("L", local), ("P", proposed)):
        errors, live = precursor_errors(output, truth)
        spread[name] = float(np.sqrt(np.mean(output.sigma[live][:, 1, 1])))
        error[name] = float(np.sqrt(np.mean(errors[live][:, 1] ** 2)))
    assert spread["L"] > 1.2 * spread["P"]
    assert error["L"] > error["P"]


def test_arm_l_growth_calibrated_on_one_seed_is_consistent_on_another(synthetic_run, held_out_local_run):
    """The in-sample check above is 2 by construction; this one is out of sample."""
    growth = calibrated_growth(synthetic_run["outputs"]["L"], synthetic_run["truth"])
    assert growth > 0.0
    output, truth = held_out_local_run
    grown = fusion.apply_precursor_growth(output, growth)
    means = interval_means(grown, precursor_nees(grown, truth))
    assert means.size == len(SYNTHETIC_SLACK)
    low, high = mean_band(2, means.size)
    assert low < means.mean() < high, (means, low, high)


def test_step_accepts_numpy_sample_rows():
    """``FleetEstimator.step`` takes rows as sequences or arrays (SensorLog rows are arrays)."""
    log, _ = synthetic_scenario(2, duration=1.0, slack=((2, 0.2),), beacon_on=0.5, dip_width=0.5)
    as_lists = fusion.FleetEstimator("P", log.geometry, log.initial_poses, 0.1, 5, EstimatorParameters())
    as_arrays = fusion.FleetEstimator("P", log.geometry, log.initial_poses, 0.1, 5, EstimatorParameters())
    for tick, samples, beacon in fusion.log_ticks(log):
        as_lists.step(tick, samples, beacon)
        rows = [tuple(None if row is None else np.asarray(row) for row in vessel) for vessel in samples]
        as_arrays.step(tick, rows, None if beacon is None else np.asarray(beacon))
        if tick % fusion.MONITOR_PERIOD_TICKS == 0:
            as_lists.record(tick * core.TICK)
            as_arrays.record(tick * core.TICK)
    first, second = as_lists.output(), as_arrays.output()
    assert first.slack.any()
    for name in ("load_mean", "load_cov", "e_hat", "sigma"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_replay_is_deterministic():
    log, _ = synthetic_scenario(2, duration=4.0, slack=((2, 1.0),), beacon_on=2.0)
    first = fusion.run_log(log, ("P", "B2", "L"), 0.4, 5)
    second = fusion.run_log(log, ("P", "B2", "L"), 0.4, 5)
    for arm in first:
        for name in ("load_mean", "load_cov", "e_hat", "sigma"):
            np.testing.assert_array_equal(getattr(first[arm], name), getattr(second[arm], name))


def test_update_with_singular_innovation_uses_minimum_norm_gain():
    geometry, _ = fan_geometry()
    vessel = VesselFilter(0, geometry, EstimatorParameters(), np.zeros(3), np.array([15.0, 0.0, 0.0]))
    vessel.covariance = np.zeros_like(vessel.covariance)
    vessel.covariance[0, 0] = 1.0e-2
    jacobian = np.zeros((2, vessel.covariance.shape[0]))
    jacobian[0, 0] = 1.0
    jacobian[1, 1] = 1.0  # a direction with zero prior variance and zero noise: singular innovation
    before = vessel.covariance.copy()
    vessel.apply_update(np.array([0.1, 0.2]), jacobian, np.zeros((2, 2)), "neighbour")
    assert np.all(np.isfinite(vessel.covariance))
    assert vessel.covariance[0, 0] < before[0, 0]
    assert vessel.covariance[1, 1] == 0.0
