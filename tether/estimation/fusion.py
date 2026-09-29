"""Fusion rules (P, B2, L, O) and the per-arm fleet estimator on the 10 ms tick grid.

On receipt of a packet of age ``D = t - stamp`` the neighbour's load block is
fast-forwarded by the constant-twist model and read as a measurement of the receiver's
load block (6 of the 11 error coordinates):

* P: age inflation ``D^2 Q_tau``, with ``Q_tau = Q(D) / D^2`` the random-acceleration
  noise of ``Q_L`` over ``D``; covariance intersection on the load block,
  ``P_LL^-1 <- w P_LL^-1 + (1 - w) P_nb,ff^-1``, embedded in the full state by carrying the
  other states on their conditional given the load block (equivalently a Kalman update
  with prior ``P + (1/w - 1) P[:, L] P_LL^-1 P[L, :]`` and noise ``R / (1 - w)``); ``w``
  minimises the full ``trace(P)`` by golden section.  Declared: the literal full-state
  form ``P^-1 <- w P^-1 + (1 - w) H' R^-1 H`` also inflates the own pose, surge scale, and
  gyro bias by ``1/w`` at every reception (20 per second) though packets carry nothing on
  them; in the synthetic test the surge-scale std grew from 0.010 to 0.050 in 11 s.
* B2: no age inflation, naive Kalman update (information addition).
* L: no fusion; its precursor covariance grows by ``q_L G(s)`` (``G`` the random
  acceleration Gramian over the slack age ``s``), q_L calibrated offline.
* O: pass-through (the monitor substitutes plant truth through the declared exemption).

Within a tick: own samples (odometry, tension, bearing, beacon), then packets are sent
(every 100 ms), then due packets are fused, then the precursors step.  Drake-free.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np

from tether.estimation import se2
from tether.estimation.comms import PACKET_PERIOD_TICKS, PacketFabric, delay_ticks
from tether.estimation.core import (
    LOAD_BLOCK,
    LOAD_SIZE,
    REFERENCE_AGE,
    STATE_SIZE,
    TICK,
    EstimatorGeometry,
    EstimatorOutput,
    EstimatorParameters,
    Precursor,
    SensorLog,
    VesselFilter,
    constant_twist_transition,
    random_acceleration_noise,
)

ESTIMATOR_ARMS = ("P", "B2", "L")
MONITOR_PERIOD_TICKS = 10
GOLDEN_RATIO = (math.sqrt(5.0) - 1.0) / 2.0
WEIGHT_TOLERANCE = 1.0e-4
FULL_WEIGHT = 1.0 - 1.0e-9


@dataclass(frozen=True)
class Packet:
    """``(stamp, load mean (6), load covariance (6 x 6))`` in the sender's error coordinates."""

    stamp: float
    source: int
    pose: np.ndarray
    twist: np.ndarray
    covariance: np.ndarray


def fast_forward(packet: Packet, age: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Neighbour load estimate carried ``age`` seconds forward at constant twist, no inflation."""
    transition = constant_twist_transition(packet.twist, age)
    pose = se2.compose(packet.pose, se2.exp(packet.twist * age))
    return pose, packet.twist.copy(), transition @ packet.covariance @ transition.T


def neighbour_measurement(
    vessel: VesselFilter, pose: np.ndarray, twist: np.ndarray, covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residual, Jacobian (6 x 11), and noise of a neighbour load estimate.

    ``g = g_i Exp(xi_i) = g_nb Exp(xi_nb)`` gives ``xi_i = z + J_r(z)^-1 xi_nb`` with
    ``z = Log(g_i^-1 g_nb)``; the twist enters additively.
    """
    offset = se2.log(se2.compose(se2.inverse(vessel.state.load_pose), pose))
    residual = np.concatenate([offset, twist - vessel.state.load_twist])
    frame = np.eye(LOAD_SIZE)
    frame[0:3, 0:3] = se2.right_jacobian_inverse(offset)
    jacobian = np.zeros((LOAD_SIZE, STATE_SIZE))
    jacobian[:, LOAD_BLOCK] = np.eye(LOAD_SIZE)
    return residual, jacobian, frame @ covariance @ frame.T


def golden_section(function, lower: float, upper: float, tolerance: float = WEIGHT_TOLERANCE) -> float:
    a, b = lower, upper
    c = b - GOLDEN_RATIO * (b - a)
    d = a + GOLDEN_RATIO * (b - a)
    fc, fd = function(c), function(d)
    while b - a > tolerance:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - GOLDEN_RATIO * (b - a)
            fc = function(c)
        else:
            a, c, fc = c, d, fd
            d = a + GOLDEN_RATIO * (b - a)
            fd = function(d)
    return 0.5 * (a + b)


def intersection_trace(load_covariance: np.ndarray, noise: np.ndarray, lift: np.ndarray):
    """``w -> trace(T P_LL(w) T')`` with ``P_LL(w) = (w P_LL^-1 + (1 - w) R^-1)^-1``.

    ``T = P[:, L] P_LL^-1`` lifts the load block into the full state; the rest of the
    full trace does not depend on ``w``.  O(n) per call after one eigensolve.
    """
    factor = np.linalg.cholesky(load_covariance)
    eigenvalues, vectors = np.linalg.eigh(factor.T @ np.linalg.solve(noise, factor))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    weights = np.sum((lift @ factor @ vectors) ** 2, axis=0)

    def trace(weight: float) -> float:
        with np.errstate(divide="ignore"):
            return float(np.sum(weights / (weight + (1.0 - weight) * eigenvalues)))

    return trace


def intersection_weight(load_covariance: np.ndarray, noise: np.ndarray, lift: np.ndarray) -> float:
    """CI weight in [0, 1] minimising the trace of the fused full covariance (golden section)."""
    trace = intersection_trace(load_covariance, noise, lift)
    weight = golden_section(trace, 0.0, 1.0)
    return 1.0 if trace(1.0) <= trace(weight) else weight


class CovarianceIntersection:
    """Arm P: fast-forward, age inflation, covariance intersection."""

    name = "P"
    exchanges = True

    def __init__(self, parameters: EstimatorParameters) -> None:
        self._psd = parameters.load_process_psd()
        self.weights: list[float] = []

    def age_inflation(self, age: float) -> np.ndarray:
        return random_acceleration_noise(self._psd, age)

    def fuse(self, vessel: VesselFilter, packet: Packet, time: float) -> None:
        age = time - packet.stamp
        pose, twist, covariance = fast_forward(packet, age)
        residual, jacobian, noise = neighbour_measurement(vessel, pose, twist, covariance + self.age_inflation(age))
        full = vessel.covariance
        load = full[LOAD_BLOCK, LOAD_BLOCK]
        lift = np.linalg.solve(load, full[LOAD_BLOCK, :]).T
        weight = intersection_weight(load, noise, lift)
        self.weights.append(weight)
        if weight >= FULL_WEIGHT:
            return
        vessel.covariance = full + (1.0 / weight - 1.0) * (lift @ full[LOAD_BLOCK, :])
        vessel.apply_update(residual, jacobian, noise / (1.0 - weight), "neighbour")


class NaiveFusion:
    """Arm B2: fast-forward, no age inflation, independent-information Kalman update."""

    name = "B2"
    exchanges = True

    def __init__(self, parameters: EstimatorParameters) -> None:
        del parameters

    def fuse(self, vessel: VesselFilter, packet: Packet, time: float) -> None:
        pose, twist, covariance = fast_forward(packet, time - packet.stamp)
        residual, jacobian, noise = neighbour_measurement(vessel, pose, twist, covariance)
        vessel.apply_update(residual, jacobian, noise, "neighbour")


class NoFusion:
    """Arm L: no packets; the load twist is frozen through slack by the constant-twist model."""

    name = "L"
    exchanges = False

    def __init__(self, parameters: EstimatorParameters) -> None:
        del parameters

    def fuse(self, vessel: VesselFilter, packet: Packet, time: float) -> None:
        del vessel, packet, time


class PassThrough(NoFusion):
    """Arm O: nothing to fuse; the monitor substitutes plant truth (declared exemption)."""

    name = "O"


FUSION_RULES = {"P": CovarianceIntersection, "B2": NaiveFusion, "L": NoFusion, "O": PassThrough}


def fusion_rule(arm: str, parameters: EstimatorParameters):
    if arm not in FUSION_RULES:
        raise ValueError(f"unknown arm: {arm}")
    return FUSION_RULES[arm](parameters)


def growth_gramian(age: np.ndarray) -> np.ndarray:
    """``G(s)`` (..., 2, 2) of a unit random acceleration on ``(e, edot)`` over age ``s``."""
    s = np.asarray(age, dtype=float)
    gramian = np.empty(s.shape + (2, 2))
    gramian[..., 0, 0] = s**3 / 3.0
    gramian[..., 0, 1] = s**2 / 2.0
    gramian[..., 1, 0] = s**2 / 2.0
    gramian[..., 1, 1] = s
    return gramian


def apply_precursor_growth(output: EstimatorOutput, precursor_growth: float) -> EstimatorOutput:
    """Re-express an output's ``sigma`` with growth ``q_L`` (offline scalar calibration)."""
    age = np.where(output.slack, output.time[:, None] - output.onset_time, 0.0)
    change = (precursor_growth - output.precursor_growth) * growth_gramian(np.nan_to_num(age))
    return replace(output, sigma=output.sigma + change, precursor_growth=float(precursor_growth))


class FleetEstimator:
    """One arm: N vessel filters, the packet fabric, the fusion rule, the precursors."""

    def __init__(
        self,
        arm: str,
        geometry: EstimatorGeometry,
        initial_poses: np.ndarray,
        tau: float,
        master_seed: int,
        parameters: EstimatorParameters,
    ) -> None:
        if arm == "O":
            raise ValueError("arm O is served by the monitor from plant truth, not by an estimator")
        self.arm = arm
        self.parameters = parameters
        self.rule = fusion_rule(arm, parameters)
        n = geometry.vessel_count
        poses = np.asarray(initial_poses, dtype=float)
        self.filters = [VesselFilter(i, geometry, parameters, poses[0], poses[i + 1]) for i in range(n)]
        self.precursors = [Precursor(geometry, parameters) for _ in range(n)]
        self.fabric = (
            PacketFabric(n, delay_ticks(tau), master_seed, parameters.graph, parameters.drop_probability)
            if self.rule.exchanges
            else None
        )
        self.precursor_growth = float(parameters.precursor_growth) if arm == "L" else 0.0
        self._started = [False] * n
        self._rows: list[tuple] = []

    def step(self, tick: int, samples: list[tuple], beacon: tuple | None) -> None:
        """Process tick ``tick``; ``samples[i] = (odometry, tension, bearing)`` rows or None."""
        time = tick * TICK
        send = self.fabric is not None and tick % PACKET_PERIOD_TICKS == 0
        busy = send or tick % MONITOR_PERIOD_TICKS == 0
        for index, vessel in enumerate(self.filters):
            odometry, tension, bearing = samples[index]
            own_beacon = beacon if index == 0 else None
            fresh = any(row is not None for row in (odometry, tension, bearing, own_beacon))
            if not (busy or vessel.slack or fresh):
                continue
            vessel.propagate_to(time)
            if odometry is not None:
                if not self._started[index]:
                    vessel.initialize_twist(np.asarray(odometry[1:4]))
                    self._started[index] = True
                vessel.propagate(time, odometry[1], odometry[2], odometry[3])
            if tension is not None:
                vessel.process_tension(time, tension[1], tension[2])
            if bearing is not None:
                vessel.process_bearing(bearing[1])
            if own_beacon is not None:
                vessel.update_beacon(own_beacon[1], own_beacon[2], own_beacon[3])
        if send:
            for index, vessel in enumerate(self.filters):
                pose, twist, covariance = vessel.load_packet()
                self.fabric.send(tick, index, Packet(time, index, pose, twist, covariance))
        if self.fabric is not None:
            for _, destination, packet in self.fabric.deliver(tick):
                receiver = self.filters[destination]
                receiver.propagate_to(time)
                self.rule.fuse(receiver, packet, time)
        for vessel, precursor in zip(self.filters, self.precursors):
            if vessel.slack or precursor.active:
                precursor.step(time, vessel)

    def record(self, time: float) -> None:
        """Append the monitor row for ``time``."""
        n = len(self.filters)
        for vessel in self.filters:
            vessel.propagate_to(time)
        row = {
            "slack": np.zeros(n, dtype=bool),
            "onset_time": np.full(n, math.nan),
            "e_hat": np.full(n, math.nan),
            "edot_hat": np.full(n, math.nan),
            "sigma": np.full((n, 2, 2), math.nan),
            "a_hat": np.full(n, math.nan),
            "sigma_a": np.full(n, math.nan),
            "load_mean": np.empty((n, LOAD_SIZE)),
            "load_cov": np.empty((n, LOAD_SIZE, LOAD_SIZE)),
        }
        for index, (vessel, precursor) in enumerate(zip(self.filters, self.precursors)):
            row["load_mean"][index], row["load_cov"][index] = vessel.load_output()
            row["slack"][index] = vessel.slack
            if precursor.active:
                row["onset_time"][index] = precursor.onset_time
                row["e_hat"][index] = precursor.e_hat
                row["edot_hat"][index] = precursor.edot_hat
                row["sigma"][index] = precursor.sigma + self.precursor_growth * growth_gramian(
                    time - precursor.onset_time
                )
                row["a_hat"][index] = precursor.a_hat
                row["sigma_a"][index] = precursor.sigma_a
        self._rows.append((time, row))

    def output(self) -> EstimatorOutput:
        times = np.array([time for time, _ in self._rows])
        stacked = {key: np.stack([row[key] for _, row in self._rows]) for key in self._rows[0][1]}
        return EstimatorOutput(arm=self.arm, time=times, precursor_growth=self.precursor_growth, **stacked)


def _tick_index(stream: np.ndarray, ticks: int) -> list[int]:
    index = [-1] * ticks
    for row, time in enumerate(stream[:, 0].tolist()):
        tick = int(round(time / TICK))
        if 0 <= tick < ticks:
            index[tick] = row
    return index


def log_ticks(log: SensorLog) -> Iterator[tuple[int, list[tuple], list | None]]:
    """Yield ``(tick, samples, beacon)`` on the 10 ms grid, samples as ``FleetEstimator.step`` takes them."""
    ticks = int(round(log.end_time / TICK)) + 1
    streams = [
        [(_tick_index(stream, ticks), stream.tolist()) for stream in (log.odometry[i], log.tension[i], log.bearing[i])]
        for i in range(log.vessel_count)
    ]
    beacon_index, beacon_rows = _tick_index(log.beacon, ticks), log.beacon.tolist()
    for tick in range(ticks):
        samples = [
            tuple(rows[index[tick]] if index[tick] >= 0 else None for index, rows in vessel) for vessel in streams
        ]
        yield tick, samples, beacon_rows[beacon_index[tick]] if beacon_index[tick] >= 0 else None


def run_estimators(
    log: SensorLog,
    arms: tuple[str, ...] = ESTIMATOR_ARMS,
    tau: float = REFERENCE_AGE,
    master_seed: int = 0,
    parameters: EstimatorParameters | None = None,
    geometry: EstimatorGeometry | None = None,
) -> list[FleetEstimator]:
    """Run the arms over a sensor log on the 10 ms grid, recording monitor rows at 10 Hz."""
    parameters = parameters or EstimatorParameters()
    geometry = geometry or log.geometry
    estimators = [FleetEstimator(arm, geometry, log.initial_poses, tau, master_seed, parameters) for arm in arms]
    for tick, samples, beacon in log_ticks(log):
        for estimator in estimators:
            estimator.step(tick, samples, beacon)
            if tick % MONITOR_PERIOD_TICKS == 0:
                estimator.record(tick * TICK)
    return estimators


def run_log(
    log: SensorLog,
    arms: tuple[str, ...] = ESTIMATOR_ARMS,
    tau: float = REFERENCE_AGE,
    master_seed: int = 0,
    parameters: EstimatorParameters | None = None,
    geometry: EstimatorGeometry | None = None,
) -> dict[str, EstimatorOutput]:
    """``EstimatorOutput`` per arm (see ``run_estimators``)."""
    estimators = run_estimators(log, arms, tau, master_seed, parameters, geometry)
    return {estimator.arm: estimator.output() for estimator in estimators}
