"""Offline replay of the estimator arms on a recorded production run (spec: Architecture).

In recording mode without a supervisor the plant does not depend on the estimators, so
a run's sensor samples are converted to a ``SensorLog`` and the arms run afterwards,
exactly as they would online.  The truth bundle (plant state at 10 ms, cable e/edot at
1 ms, re-engagement marks, true load state) is for evaluation only and never reaches an
estimator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tether.estimation.core import (
    REFERENCE_AGE,
    EstimatorGeometry,
    EstimatorOutput,
    EstimatorParameters,
    SensorLog,
)
from tether.estimation.fusion import ESTIMATOR_ARMS, apply_precursor_growth, run_log
from tether.physics.fleet import EVENT_PERIOD, WEATHER_PERIOD, CableParameters, FleetGeometry

PRECURSOR_TARGET_NEES = 2.0


@dataclass(frozen=True)
class PlantTruth:
    """Evaluation-only plant logs of one run."""

    state_time: np.ndarray
    state: np.ndarray
    event_time: np.ndarray
    elongation: np.ndarray
    rate: np.ndarray
    marks: tuple
    load_state: np.ndarray


@dataclass(frozen=True)
class MonitorTruth:
    """Truth at the monitor ticks, plus the 1 ms cable series."""

    time: np.ndarray
    e_true: np.ndarray
    edot_true: np.ndarray
    load_true: np.ndarray
    event_time: np.ndarray
    e: np.ndarray
    edot: np.ndarray


def estimator_geometry(geometry: FleetGeometry, cable: CableParameters | None = None) -> EstimatorGeometry:
    cable = cable or CableParameters()
    return EstimatorGeometry(
        load_offsets=np.asarray(geometry.load_offsets, dtype=float).copy(),
        stern_offsets=np.asarray(geometry.vessel_offsets, dtype=float).copy(),
        rest_length=cable.rest_length,
        stiffness=cable.stiffness,
        damping=cable.damping,
    )


def _rows(values: list, width: int) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1, width)


def sensor_log_from_samples(
    samples: list[dict], initial_state: np.ndarray, geometry: EstimatorGeometry
) -> SensorLog:
    """``SensorSuite.samples`` of every vessel (vessel order) to a ``SensorLog``.

    ``initial_state`` is the plant state ``[q; v]`` at t = 0, whose poses are the declared
    surveyed formation.
    """
    n = len(samples)
    poses = np.asarray(initial_state, dtype=float)[: 3 * (n + 1)].reshape(n + 1, 3).copy()
    beacon = [row for suite in samples for row in suite["beacon"]]
    return SensorLog(
        odometry=tuple(_rows(suite["odometry"], 4) for suite in samples),
        bearing=tuple(_rows(suite["bearing"], 2) for suite in samples),
        tension=tuple(_rows(suite["tension"], 3) for suite in samples),
        beacon=_rows(sorted(beacon), 4),
        initial_poses=poses,
        geometry=geometry,
    )


def load_body_state(state: np.ndarray, vessel_count: int) -> np.ndarray:
    """``[x, y, theta, u, v, omega]`` of the load (body-frame twist) from ``[q; v]`` rows."""
    base = 3 * (vessel_count + 1)
    theta = state[:, 2]
    vx, vy = state[:, base], state[:, base + 1]
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.column_stack([state[:, 0], state[:, 1], theta, cosine * vx + sine * vy, -sine * vx + cosine * vy, state[:, base + 2]])


def plant_truth(
    state_time: np.ndarray,
    state: np.ndarray,
    event_time: np.ndarray,
    elongation: np.ndarray,
    rate: np.ndarray,
    marks,
) -> PlantTruth:
    vessel_count = elongation.shape[1]
    return PlantTruth(
        state_time=np.asarray(state_time, dtype=float),
        state=np.asarray(state, dtype=float),
        event_time=np.asarray(event_time, dtype=float),
        elongation=np.asarray(elongation, dtype=float),
        rate=np.asarray(rate, dtype=float),
        marks=tuple(marks),
        load_state=load_body_state(np.asarray(state, dtype=float), vessel_count),
    )


def replay_bundle(run) -> tuple[SensorLog, PlantTruth]:
    """A finished ``FleetRun`` to its ``SensorLog`` and ``PlantTruth``."""
    cables = run.fleet.cables
    log = cables.log
    if log.state_count == 0 or abs(log.state_time[0]) > 1.0e-9:
        raise ValueError("replay needs the 10 ms state log from t = 0")
    truth = plant_truth(
        log.state_time[: log.state_count].copy(),
        log.state[: log.state_count].copy(),
        log.event_time[: log.count].copy(),
        log.elongation[: log.count].copy(),
        log.rate[: log.count].copy(),
        cables.reengagements,
    )
    geometry = estimator_geometry(run.fleet.geometry, cables.cable)
    # Surveyed poses are the exact t = 0 state recorded by build_run; the state log's
    # first entry is taken after the plant's first physics step.
    initial = run.fleet.extras.get("initial_state", truth.state[0])
    sensor_log = sensor_log_from_samples([suite.samples for suite in run.sensors], initial, geometry)
    return sensor_log, truth


def run_arms(
    sensor_log: SensorLog,
    geometry: EstimatorGeometry | None = None,
    arms: tuple[str, ...] = ESTIMATOR_ARMS,
    tau: float = REFERENCE_AGE,
    master_seed: int = 0,
    parameters: EstimatorParameters | None = None,
) -> dict[str, EstimatorOutput]:
    """Run the estimator arms over the log (10 ms grid, samples in time order, 10 Hz output)."""
    return run_log(sensor_log, tuple(arms), tau, master_seed, parameters, geometry)


def monitor_truth(truth: PlantTruth, times: np.ndarray) -> MonitorTruth:
    times = np.asarray(times, dtype=float)
    event = np.clip(np.round(times / EVENT_PERIOD).astype(int), 0, truth.event_time.size - 1)
    state = np.clip(np.round(times / WEATHER_PERIOD).astype(int), 0, truth.state_time.size - 1)
    if np.max(np.abs(truth.event_time[event] - times)) > 0.5 * EVENT_PERIOD:
        raise ValueError("cable log does not cover the monitor ticks")
    if np.max(np.abs(truth.state_time[state] - times)) > 0.5 * WEATHER_PERIOD:
        raise ValueError("state log does not cover the monitor ticks")
    return MonitorTruth(
        time=times,
        e_true=truth.elongation[event],
        edot_true=truth.rate[event],
        load_true=truth.load_state[state],
        event_time=truth.event_time,
        e=truth.elongation,
        edot=truth.rate,
    )


def load_errors(output: EstimatorOutput, truth: MonitorTruth) -> np.ndarray:
    """Truth minus estimate (m, N, 6) in the coordinates of ``load_mean``."""
    errors = truth.load_true[:, None, :] - output.load_mean
    errors[..., 2] = (errors[..., 2] + math.pi) % (2.0 * math.pi) - math.pi
    return errors


def _quadratic(errors: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    return np.einsum("...i,...i->...", errors, np.linalg.solve(covariance, errors[..., None])[..., 0])


def load_pose_nees(output: EstimatorOutput, truth: MonitorTruth) -> np.ndarray:
    """3-dof NEES of the load pose (m, N)."""
    return _quadratic(load_errors(output, truth)[..., :3], output.load_cov[..., :3, :3])


def load_state_nees(output: EstimatorOutput, truth: MonitorTruth) -> np.ndarray:
    """6-dof NEES of the load pose and twist (m, N)."""
    return _quadratic(load_errors(output, truth), output.load_cov)


def precursor_nees(output: EstimatorOutput, truth: MonitorTruth) -> np.ndarray:
    """2-dof NEES of ``(e_hat, edot_hat)`` (m, N), nan while taut."""
    errors = np.stack([output.e_hat - truth.e_true, output.edot_hat - truth.edot_true], axis=-1)
    live = output.slack & np.isfinite(output.e_hat)
    values = np.full(output.e_hat.shape, math.nan)
    if np.any(live):
        values[live] = _quadratic(errors[live], output.sigma[live])
    return values


def slack_interval_means(output: EstimatorOutput, values: np.ndarray) -> np.ndarray:
    """One time-averaged value per slack interval (vessel, onset), in interval order."""
    means = []
    for vessel in range(output.slack.shape[1]):
        onsets = output.onset_time[:, vessel]
        for onset in np.unique(onsets[np.isfinite(onsets)]):
            selected = values[onsets == onset, vessel]
            selected = selected[np.isfinite(selected)]
            if selected.size:
                means.append(float(selected.mean()))
    return np.asarray(means)


def calibrate_precursor_growth(
    pairs: list[tuple[EstimatorOutput, MonitorTruth]], target: float = PRECURSOR_TARGET_NEES
) -> float:
    """q_L for arm L: the growth making the mean per-interval precursor NEES equal ``target``.

    Returns 0 when the arm is already at or below the target without growth.
    """

    def mean_nees(growth: float) -> float:
        means = [
            slack_interval_means(grown, precursor_nees(grown, truth))
            for grown, truth in ((apply_precursor_growth(output, growth), truth) for output, truth in pairs)
        ]
        joined = np.concatenate(means) if means else np.array([])
        return float(joined.mean()) if joined.size else math.nan

    if not mean_nees(0.0) > target:
        return 0.0
    lower, upper = 0.0, 1.0e-6
    while mean_nees(upper) > target:
        lower, upper = upper, upper * 10.0
        if upper > 1.0e6:
            raise RuntimeError("precursor growth calibration did not bracket the target")
    for _ in range(100):
        middle = math.sqrt(lower * upper) if lower > 0.0 else 0.5 * upper
        if mean_nees(middle) > target:
            lower = middle
        else:
            upper = middle
        if upper - lower <= 1.0e-6 * upper:
            break
    return 0.5 * (lower + upper)
