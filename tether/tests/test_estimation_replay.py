"""Offline replay: sensor-log and truth conversion, NEES tools, q_L calibration, production run."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from tether.estimation import fusion, replay
from tether.estimation.core import EstimatorOutput, EstimatorParameters
from tether.physics.fleet import CableParameters, formation_geometry
from tether.tests.test_estimation_core import synthetic_scenario

SLOW = os.environ.get("TETHER_SLOW") == "1"
RNG_ENTROPY = 20260913


def fake_samples(vessel_count: int, duration: float) -> list[dict]:
    samples = []
    for agent in range(vessel_count):
        suite = {"odometry": [], "bearing": [], "tension": [], "beacon": []}
        for tick in range(int(round(duration / 0.01)) + 1):
            t = tick * 0.01
            if tick % 2 == 0:
                suite["odometry"].append((t, 0.9 + agent, 0.01, 0.002))
                suite["tension"].append((t, 1000.0 + agent, 0.0))
            if tick % 5 == 0:
                suite["bearing"].append((t, 0.01 * agent))
            if agent == 0 and t >= 0.5 - 1e-9 and tick % 20 == 0:
                suite["beacon"].append((t, 1.0, 2.0, 0.1))
        samples.append(suite)
    return samples


def test_sensor_log_from_samples_keeps_the_spec_layout():
    geometry = replay.estimator_geometry(formation_geometry("fan", arc_half_angle=0.55), CableParameters())
    state = np.arange(36, dtype=float)
    log = replay.sensor_log_from_samples(fake_samples(5, 1.0), state, geometry)
    assert log.vessel_count == 5 and log.end_time == pytest.approx(1.0)
    assert all(stream.shape == (51, 4) for stream in log.odometry)
    assert all(stream.shape == (21, 2) for stream in log.bearing)
    assert all(stream.shape == (51, 3) for stream in log.tension)
    assert log.beacon.shape == (3, 4) and np.all(log.beacon[:, 0] >= 0.5)
    np.testing.assert_array_equal(log.initial_poses, state[:18].reshape(6, 3))
    np.testing.assert_array_equal(log.odometry[3][:, 1], 3.9)
    assert log.geometry.rest_length == 12.0 and log.geometry.stiffness == 1.7e5


def test_plant_truth_and_monitor_ticks():
    n = 5
    state_time = np.arange(0, 201) * 0.01
    state = np.zeros((state_time.size, 6 * (n + 1)))
    state[:, 2] = 0.5 * state_time
    state[:, 18] = 1.0
    state[:, 19] = 2.0
    state[:, 20] = 0.5
    event_time = np.arange(0, 2001) * 0.001
    elongation = np.outer(np.sin(event_time), np.arange(1, n + 1))
    rate = np.outer(np.cos(event_time), np.arange(1, n + 1))
    truth = replay.plant_truth(state_time, state, event_time, elongation, rate, [])
    theta = state[:, 2]
    np.testing.assert_allclose(truth.load_state[:, 3], np.cos(theta) + 2.0 * np.sin(theta))
    np.testing.assert_allclose(truth.load_state[:, 4], -np.sin(theta) + 2.0 * np.cos(theta))
    ticks = np.arange(0, 21) * 0.1
    sampled = replay.monitor_truth(truth, ticks)
    np.testing.assert_allclose(sampled.e_true, np.outer(np.sin(ticks), np.arange(1, n + 1)), atol=1e-12)
    np.testing.assert_allclose(sampled.edot_true[:, 2], 3.0 * np.cos(ticks), atol=1e-12)
    np.testing.assert_allclose(sampled.load_true[:, 2], 0.5 * ticks, atol=1e-12)
    with pytest.raises(ValueError):
        replay.monitor_truth(truth, np.array([5.0]))


def fabricated_output(growth: float, seed: int, intervals: int = 400) -> tuple[EstimatorOutput, replay.MonitorTruth]:
    """One vessel, slack intervals of 1 s at 10 Hz; errors drawn with covariance
    base + growth G(s); the output reports only the base."""
    random = np.random.default_rng(np.random.SeedSequence([RNG_ENTROPY, seed]))
    per_interval = 11
    m = intervals * (per_interval + 5)
    time = np.arange(m) * 0.1
    slack = np.zeros((m, 1), dtype=bool)
    onset = np.full((m, 1), math.nan)
    sigma = np.full((m, 1, 2, 2), math.nan)
    e_hat = np.full((m, 1), math.nan)
    edot_hat = np.full((m, 1), math.nan)
    base = np.array([[4.0e-4, 1.0e-4], [1.0e-4, 1.0e-3]])
    for k in range(intervals):
        rows = np.arange(k * (per_interval + 5), k * (per_interval + 5) + per_interval)
        slack[rows, 0] = True
        onset[rows, 0] = time[rows[0]]
        age = time[rows] - time[rows[0]]
        for row, s in zip(rows, age):
            actual = base + growth * fusion.growth_gramian(np.array(s))
            e_hat[row, 0], edot_hat[row, 0] = np.linalg.cholesky(actual) @ random.standard_normal(2)
            sigma[row, 0] = base
    output = EstimatorOutput(
        arm="L",
        time=time,
        slack=slack,
        onset_time=onset,
        e_hat=e_hat,
        edot_hat=edot_hat,
        sigma=sigma,
        a_hat=np.zeros((m, 1)),
        sigma_a=np.zeros((m, 1)),
        load_mean=np.zeros((m, 1, 6)),
        load_cov=np.tile(np.eye(6), (m, 1, 1, 1)),
    )
    truth = replay.MonitorTruth(
        time=time,
        e_true=np.zeros((m, 1)),
        edot_true=np.zeros((m, 1)),
        load_true=np.zeros((m, 6)),
        event_time=time,
        e=np.zeros((m, 1)),
        edot=np.zeros((m, 1)),
    )
    return output, truth


def test_precursor_nees_and_interval_means():
    output, truth = fabricated_output(0.0, 1)
    values = replay.precursor_nees(output, truth)
    assert np.all(np.isnan(values[~output.slack])) and np.all(np.isfinite(values[output.slack]))
    means = replay.slack_interval_means(output, values)
    assert means.size == 400
    assert means.mean() == pytest.approx(2.0, abs=0.15)


def test_precursor_growth_calibration_recovers_the_growth():
    growth = 0.05
    output, truth = fabricated_output(growth, 2)
    calibrated = replay.calibrate_precursor_growth([(output, truth)])
    grown = fusion.apply_precursor_growth(output, calibrated)
    assert replay.slack_interval_means(grown, replay.precursor_nees(grown, truth)).mean() == pytest.approx(2.0, abs=1e-4)
    assert calibrated == pytest.approx(growth, rel=0.3)
    consistent, consistent_truth = fabricated_output(0.0, 3)
    consistent = EstimatorOutput(**{**consistent.__dict__, "sigma": 2.0 * consistent.sigma})
    assert replay.calibrate_precursor_growth([(consistent, consistent_truth)]) == 0.0


def test_run_arms_is_the_core_runner_and_rejects_the_oracle():
    log, _ = synthetic_scenario(3, duration=3.0, slack=((1, 0.8),), beacon_on=1.0)
    outputs = replay.run_arms(log, None, ("P", "L"), 0.2, 9)
    expected = fusion.run_log(log, ("P", "L"), 0.2, 9)
    for arm in ("P", "L"):
        assert outputs[arm].arm == arm
        np.testing.assert_array_equal(outputs[arm].load_mean, expected[arm].load_mean)
        np.testing.assert_array_equal(outputs[arm].sigma, expected[arm].sigma)
    with pytest.raises(ValueError):
        replay.run_arms(log, None, ("O",), 0.2, 9)


@pytest.mark.skipif(not SLOW, reason="production-plant replay; set TETHER_SLOW=1")
def test_production_run_replayed_through_all_arms():
    """60 s fan tow (common-mode weather at intensity 0.5) replayed through P, B2, L."""
    from tether.campaign.fleet_run import FleetRunSpec, build_run, run_to_end

    spec = FleetRunSpec(
        formation="fan",
        arc_half_angle=0.55,
        pretension=1000.0,
        weather_scale=0.5,
        weather_direction="common",
        duration=60.0,
        warmup=0.0,
    )
    master_seed = 5
    run = run_to_end(build_run(spec, master_seed))
    assert not run.lint_violations
    sensor_log, truth = replay.replay_bundle(run)
    for streams, period in ((sensor_log.odometry, 0.02), (sensor_log.bearing, 0.05), (sensor_log.tension, 0.02)):
        for stream in streams:
            assert stream[0, 0] == 0.0 and stream[-1, 0] > 59.9
            np.testing.assert_allclose(np.diff(stream[:, 0]), period, atol=1e-9)
    assert sensor_log.beacon[0, 0] == pytest.approx(30.0) and np.allclose(np.diff(sensor_log.beacon[:, 0]), 0.2)
    parameters = EstimatorParameters(weather_scale=spec.weather_scale)
    outputs = replay.run_arms(sensor_log, sensor_log.geometry, ("P", "B2", "L"), 0.4, master_seed, parameters)
    report = {}
    for arm, output in outputs.items():
        sampled = replay.monitor_truth(truth, output.time)
        pose = replay.load_pose_nees(output, sampled)
        precursor = replay.slack_interval_means(output, replay.precursor_nees(output, sampled))
        report[arm] = {
            "pose_nees": float(pose.mean()),
            "pose_nees_before_beacon": float(pose[output.time < 30.0].mean()),
            "pose_nees_after_beacon": float(pose[output.time >= 30.0].mean()),
            "slack_intervals": int(precursor.size),
            "precursor_nees": float(precursor.mean()) if precursor.size else math.nan,
        }
        assert output.time[0] == 0.0 and output.time[-1] > 59.8 and np.all(np.isfinite(output.load_cov))
    print("production replay NEES:", report)
    assert report["P"]["pose_nees"] < 10.0
    assert report["B2"]["pose_nees"] > 100.0 * report["P"]["pose_nees"]
