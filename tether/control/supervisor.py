"""Hazard supervisor (plan IV.7, Phase 6).

``F_T,i <- F_T,i (1 - kappa_s sat(h / h_crit))`` with ``kappa_s = 0.6``.  The fleet term
throttles every vessel on the largest hazard it has heard of: vessel i uses
``max(h_i(t), max_j h_j(t - d_ij tau))`` where ``d_ij`` is the hop distance on the
communication cycle (declared: hazards travel the same ring buffers as the estimator's
packets, one ``tau`` per hop).  Not plant-side; its input is the monitor's hazard output.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from pydrake.systems.framework import LeafSystem

SUPERVISOR_PERIOD = 0.01
THRUST_EASING = 0.6


def cycle_hops(vessel_count: int) -> np.ndarray:
    index = np.arange(vessel_count)
    distance = np.abs(index[:, None] - index[None, :])
    return np.minimum(distance, vessel_count - distance)


def thrust_scale(hazard: np.ndarray, critical: float, easing: float = THRUST_EASING) -> np.ndarray:
    return 1.0 - easing * np.clip(np.asarray(hazard, dtype=float) / critical, 0.0, 1.0)


class Supervisor(LeafSystem):
    PLANT_SIDE = False

    def __init__(self, vessel_count: int, critical_hazard: float, latency: float, fleet_term: bool = True, easing: float = THRUST_EASING) -> None:
        super().__init__()
        self._n = vessel_count
        self._critical = critical_hazard
        self._easing = easing
        self._fleet = fleet_term
        self._hops = cycle_hops(vessel_count)
        self._delay_ticks = int(round(latency / SUPERVISOR_PERIOD))
        depth = self._delay_ticks * int(self._hops.max()) + 1
        self._history: deque = deque(maxlen=max(depth, 1))
        self._hazard_port = self.DeclareVectorInputPort("hazard", vessel_count)
        self._scale_index = self.DeclareDiscreteState(np.ones(vessel_count))
        self.DeclareStateOutputPort("thrust_scale", self._scale_index)
        self.DeclarePeriodicDiscreteUpdateEvent(SUPERVISOR_PERIOD, 0.0, self._update)
        self.log: list[tuple[float, np.ndarray, np.ndarray]] = []

    def _update(self, context, discrete_state) -> None:
        hazard = np.asarray(self._hazard_port.Eval(context), dtype=float).copy()
        self._history.appendleft(hazard)
        effective = hazard.copy()
        if self._fleet:
            for i in range(self._n):
                for j in range(self._n):
                    if i == j:
                        continue
                    lag = self._hops[i, j] * self._delay_ticks
                    if lag < len(self._history):
                        effective[i] = max(effective[i], self._history[lag][j])
        scale = thrust_scale(effective, self._critical, self._easing)
        self.log.append((context.get_time(), hazard, scale))
        discrete_state.get_mutable_vector(self._scale_index).SetFromVector(scale)
