"""Drake adapters that run the estimation core and the monitor online.

``EstimatorAdapter`` (not plant-side) reads only the sensor suites' sanctioned outputs,
drives one arm's ``FleetEstimator`` on the 10 ms tick grid, and outputs each vessel's
estimated cable bearing ``[bearing, stamp]`` (staleness campaign) and its hazard
(Phase 6 supervisor).  Its periodic update runs at a 5 ms offset from the sensors so
that a sample stamped ``t`` reaches the estimator at tick ``round(t / 10 ms)``, exactly as
in the offline replay.

``OracleAdapter`` computes the oracle arm's precursor and hazard from plant truth; it is
the single declared truth-isolation exemption.
"""

from __future__ import annotations

import math

import numpy as np
from pydrake.systems.framework import LeafSystem

from tether.estimation.core import TICK, EstimatorGeometry, EstimatorParameters, bearing_model
from tether.monitor.oracle import ORACLE_SIGMA_E, ORACLE_SIGMA_EDOT, SIGMA_A_FLOOR, SLOPE_WINDOW
from tether.estimation.fusion import MONITOR_PERIOD_TICKS, FleetEstimator
from tether.monitor.hazard import hazard_generator
from tether.monitor.linearized import model_tick_hazard
from tether.physics.sensors import BEACON_SIZE, BEARING_SIZE, ODOMETRY_SIZE, TENSION_SIZE

ADAPTER_OFFSET = 0.005
ORACLE_EXEMPTION = "oracle_adapter"


class EstimatorAdapter(LeafSystem):
    PLANT_SIDE = False

    def __init__(
        self,
        arm: str,
        geometry: EstimatorGeometry,
        initial_poses: np.ndarray,
        tau: float,
        master_seed: int,
        critical_speeds: np.ndarray,
        parameters: EstimatorParameters | None = None,
        hazard_samples: int = 2048,
        horizon: float = 2.0,
        compute_hazard: bool = True,
        hazard_model: str = "constant_acceleration",
    ) -> None:
        super().__init__()
        n = geometry.vessel_count
        self._n = n
        self.estimator = FleetEstimator(arm, geometry, initial_poses, tau, master_seed, parameters or EstimatorParameters())
        self._critical = np.asarray(critical_speeds, dtype=float)
        self._generators = [hazard_generator(master_seed, i) for i in range(n)]
        self._samples = hazard_samples
        self._horizon = horizon
        self._compute_hazard = compute_hazard
        self._model = hazard_model
        self._odometry = [self.DeclareVectorInputPort(f"odometry_{i}", ODOMETRY_SIZE) for i in range(n)]
        self._bearing = [self.DeclareVectorInputPort(f"cable_bearing_{i}", BEARING_SIZE) for i in range(n)]
        self._tension = [self.DeclareVectorInputPort(f"tension_{i}", TENSION_SIZE) for i in range(n)]
        self._beacon = self.DeclareVectorInputPort("beacon", BEACON_SIZE)
        self._last = {"odometry": [-1.0] * n, "bearing": [-1.0] * n, "tension": [-1.0] * n, "beacon": -1.0}
        self._output_index = self.DeclareDiscreteState(np.concatenate([np.zeros(2 * n), np.zeros(n)]))
        for i in range(n):
            self.DeclareVectorOutputPort(f"estimated_bearing_{i}", 2, lambda context, output, i=i: output.SetFromVector(
                context.get_discrete_state(self._output_index).get_value()[2 * i : 2 * i + 2]))
        self.DeclareVectorOutputPort("hazard", n, lambda context, output: output.SetFromVector(
            context.get_discrete_state(self._output_index).get_value()[2 * n :]))
        self.DeclarePeriodicDiscreteUpdateEvent(TICK, ADAPTER_OFFSET, self._update)
        self.hazard_log: list[tuple[float, np.ndarray]] = []

    def _fresh(self, kind: str, index: int, row: np.ndarray, stamp_index: int):
        stamp = float(row[stamp_index])
        if stamp > self._last[kind][index] + 1.0e-9:
            self._last[kind][index] = stamp
            return stamp
        return None

    def _update(self, context, discrete_state) -> None:
        tick = int(round((context.get_time() - ADAPTER_OFFSET) / TICK))
        samples = []
        for i in range(self._n):
            odometry = np.asarray(self._odometry[i].Eval(context))
            bearing = np.asarray(self._bearing[i].Eval(context))
            tension = np.asarray(self._tension[i].Eval(context))
            row_odometry = row_bearing = row_tension = None
            stamp = self._fresh("odometry", i, odometry, 3)
            if stamp is not None and int(round(stamp / TICK)) == tick:
                row_odometry = (stamp, float(odometry[0]), float(odometry[1]), float(odometry[2]))
            stamp = self._fresh("tension", i, tension, 2)
            if stamp is not None and int(round(stamp / TICK)) == tick:
                row_tension = (stamp, float(tension[0]), float(tension[1]))
            stamp = self._fresh("bearing", i, bearing, 1)
            if stamp is not None and int(round(stamp / TICK)) == tick:
                row_bearing = (stamp, float(bearing[0]))
            samples.append((row_odometry, row_tension, row_bearing))
        beacon = np.asarray(self._beacon.Eval(context))
        row_beacon = None
        if beacon[3] > 0.5 and float(beacon[4]) > self._last["beacon"] + 1.0e-9:
            self._last["beacon"] = float(beacon[4])
            if int(round(beacon[4] / TICK)) == tick:
                row_beacon = (float(beacon[4]), float(beacon[0]), float(beacon[1]), float(beacon[2]))
        estimator = self.estimator
        estimator.step(tick, samples, row_beacon)
        values = context.get_discrete_state(self._output_index).get_value().copy()
        time = tick * TICK
        for i, vessel in enumerate(estimator.filters):
            values[2 * i] = bearing_model(vessel.state, vessel.load_offset, vessel.stern_offset)[0]
            values[2 * i + 1] = time
        if tick % MONITOR_PERIOD_TICKS == 0:
            estimator.record(time)
            if self._compute_hazard:
                row = estimator._rows[-1][1]
                hazard = np.zeros(self._n)
                for i in range(self._n):
                    if row["slack"][i] and np.isfinite(row["e_hat"][i]):
                        hazard[i] = model_tick_hazard(self._model, row["e_hat"][i], row["edot_hat"][i], row["sigma"][i], row["a_hat"][i],
                                                      row["sigma_a"][i], self._critical[i], self._generators[i],
                                                      n_samples=self._samples, horizon=self._horizon)
                values[2 * self._n :] = np.nan_to_num(hazard, nan=0.0)
                self.hazard_log.append((time, hazard))
        discrete_state.get_mutable_vector(self._output_index).SetFromVector(values)


class OracleAdapter(LeafSystem):
    """Oracle arm: precursor and hazard from plant truth (declared exemption).

    The true (e, e') are read every 10 ms, the estimator arms' tick; a_hat is the
    least-squares slope over the 10 ms samples of the current slack interval within the last
    0.3 s (a0 = T0 / m_eff, sigma_a at the floor, before three samples exist), as in the
    estimator core, and the hazard is evaluated at the 10 Hz monitor ticks.
    """

    PLANT_SIDE = False

    def __init__(self, fleet_geometry, rest_length: float, pretension: float, master_seed: int, critical_speeds: np.ndarray,
                 hazard_samples: int = 2048, horizon: float = 2.0, hazard_model: str = "constant_acceleration") -> None:
        super().__init__()
        from tether.physics import constants

        self.set_name(ORACLE_EXEMPTION)
        self._geometry = fleet_geometry
        self._rest = rest_length
        n = fleet_geometry.vessel_count
        self._n = n
        self._a0 = pretension * (1.0 / constants.VESSEL_MASS + 1.0 / constants.LOAD_MASS)
        self._critical = np.asarray(critical_speeds, dtype=float)
        self._generators = [hazard_generator(master_seed, i) for i in range(n)]
        self._samples = hazard_samples
        self._horizon = horizon
        self._model = hazard_model
        self._state = self.DeclareVectorInputPort("plant_state", 6 * (n + 1))
        self._history: list[list[tuple[float, float]]] = [[] for _ in range(n)]
        self._hazard_index = self.DeclareDiscreteState(np.zeros(n))
        self.DeclareStateOutputPort("hazard", self._hazard_index)
        self.DeclarePeriodicDiscreteUpdateEvent(TICK, 0.0, self._update)
        self.hazard_log: list[tuple[float, np.ndarray]] = []

    def _slope(self, samples: list[tuple[float, float]]) -> tuple[float, float]:
        if len(samples) < 3:
            return self._a0, SIGMA_A_FLOOR
        t = np.array([sample[0] for sample in samples])
        v = np.array([sample[1] for sample in samples])
        centred = t - t.mean()
        spread = float(centred @ centred)
        slope = float(centred @ (v - v.mean())) / spread
        residual = v - v.mean() - slope * centred
        error = math.sqrt(float(residual @ residual) / (t.size - 2) / spread)
        return slope, max(error, SIGMA_A_FLOOR)

    def _update(self, context, discrete_state) -> None:
        from tether.physics.fleet import cable_kinematics

        tick = int(round(context.get_time() / TICK))
        time = tick * TICK
        kinematics = cable_kinematics(np.asarray(self._state.Eval(context)), self._geometry, self._rest)
        monitor = tick % MONITOR_PERIOD_TICKS == 0
        hazard = np.zeros(self._n)
        for i in range(self._n):
            e = float(kinematics["elongation"][i])
            edot = float(kinematics["rate"][i])
            if e > 0.0:
                self._history[i] = []
                continue
            history = self._history[i]
            history.append((time, edot))
            while history[0][0] < time - SLOPE_WINDOW - 1.0e-9:
                history.pop(0)
            if monitor:
                a_hat, sigma_a = self._slope(history)
                sigma = np.diag([ORACLE_SIGMA_E**2, ORACLE_SIGMA_EDOT**2])
                hazard[i] = model_tick_hazard(self._model, e, edot, sigma, a_hat, sigma_a, self._critical[i], self._generators[i],
                                              n_samples=self._samples, horizon=self._horizon)
        if monitor:
            self.hazard_log.append((time, hazard))
            discrete_state.get_mutable_vector(self._hazard_index).SetFromVector(np.nan_to_num(hazard, nan=0.0))
