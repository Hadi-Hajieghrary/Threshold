"""Sensor-driven per-vessel controllers (plan IV.7).

``yaw = k_h wrap(theta_ref(t) - theta_hat) - k_c sin(bearing)`` where ``theta_hat`` is
the gyro-integrated heading and ``bearing`` the measured cable bearing (0 dead astern,
positive to port).  The plan writes ``+ k_c sin(sigma)``; with this bearing convention
the minus sign is the one that turns the hull toward its cable, which is the trim the
term exists for (the same sense as the stern attachment's weathervane moment).

v2 sway (chord-angle) term, opt-in (plan v2 IV.7): the reference becomes
``theta_ref(t) - k_sigma wrap(sigma_hat - phi)`` with ``sigma_hat = theta_hat - bearing``
the world chord angle and ``phi`` the design cable angle.  The plan writes
``theta_hat + bearing``; with this bearing convention the measured bearing equals
``theta - sigma`` exactly, so the minus sign is the one that gives the world chord angle,
and with it the term turns the hull away from the side the vessel has swung to, which
corrects a lateral offset (tether/tests/test_v2_sway.py).  ``sway_gain = 0`` (default)
skips the term and reproduces v1 bit-exactly.

Saturation, opt-in (records/v2/phase0/sway_addendum_1.json, owner ruling): with
``sway_limit`` set, the correction ``k_sigma wrap(sigma_hat - phi)`` is clipped to
``[-sway_limit, +sway_limit]`` before it enters the reference.  The unsaturated law, whose
correction is itself wrapped by the heading error's outer wrap, has spurious "parked"
equilibria roughly 2 pi / (1 + k_sigma) away from the design angle, where the wrapped
reference lines the hull up with the swung chord; the clamp removes them.
``sway_limit = None`` (default) is the unsaturated law, bit-exactly; v2 uses 0.349 rad.

The surge force follows a pre-loaded schedule, optionally scaled by a supervisor factor
(Phase 6).  The controller is not plant-side: its only inputs are sensor outputs and
non-plant-side signals, which the diagram lint verifies.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from pydrake.systems.framework import LeafSystem

from tether.physics.sensors import BEARING_SIZE, ODOMETRY_SIZE

CONTROL_PERIOD = 0.02
NOMINAL_HEADING_GAIN = 500.0
CABLE_TRIM_GAIN = 100.0
SWAY_GAIN = 0.0  # v1: no sway term
SWAY_LIMIT = None  # unsaturated sway correction (the plan's literal law)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class ControllerBank(LeafSystem):
    """One heading/surge controller per vessel, sharing a single 50 Hz update.

    Each vessel's controller reads only its own odometry and cable-bearing ports.
    Output: ``[surge_1..N, yaw_1..N]``.
    """

    PLANT_SIDE = False

    def __init__(
        self,
        vessel_count: int,
        heading_reference: Callable[[float], np.ndarray],
        surge_schedule: Callable[[float], np.ndarray],
        initial_headings: np.ndarray,
        heading_gain: float = NOMINAL_HEADING_GAIN,
        trim_gain: float = CABLE_TRIM_GAIN,
        supervised: bool = False,
        sway_gain: float = SWAY_GAIN,
        design_chord_angles: np.ndarray | None = None,
        sway_limit: float | None = SWAY_LIMIT,
    ) -> None:
        super().__init__()
        self._n = vessel_count
        self._heading_reference = heading_reference
        self._surge_schedule = surge_schedule
        self._heading_gain = heading_gain
        self._trim_gain = trim_gain
        self._sway_gain = float(sway_gain)
        if sway_limit is not None and not float(sway_limit) > 0.0:
            raise ValueError("sway_limit must be positive (or None for the unsaturated law)")
        self._sway_limit = None if sway_limit is None else float(sway_limit)
        self._design_chord_angles = (
            np.zeros(vessel_count) if design_chord_angles is None else np.asarray(design_chord_angles, dtype=float).copy()
        )
        if self._design_chord_angles.shape != (vessel_count,):
            raise ValueError("design_chord_angles must have one entry per vessel")
        self._odometry_ports = [
            self.DeclareVectorInputPort(f"odometry_{i}", ODOMETRY_SIZE) for i in range(vessel_count)
        ]
        self._bearing_ports = [
            self.DeclareVectorInputPort(f"cable_bearing_{i}", BEARING_SIZE) for i in range(vessel_count)
        ]
        self._scale_port = (
            self.DeclareVectorInputPort("thrust_scale", vessel_count) if supervised else None
        )
        # State: heading estimates, last odometry stamps, then the held command.
        initial = np.concatenate(
            [
                np.asarray(initial_headings, dtype=float),
                np.full(vessel_count, -1.0),
                np.asarray(surge_schedule(0.0), dtype=float),
                np.zeros(vessel_count),
            ]
        )
        self._state_index = self.DeclareDiscreteState(initial)
        self.DeclareVectorOutputPort("thrust_command", 2 * vessel_count, self._calc_command)
        self.DeclarePeriodicDiscreteUpdateEvent(CONTROL_PERIOD, 0.0, self._update)

    def _calc_command(self, context, output) -> None:
        state = context.get_discrete_state(self._state_index).get_value()
        output.SetFromVector(state[2 * self._n :])

    def heading_estimates(self, context) -> np.ndarray:
        return context.get_discrete_state(self._state_index).get_value()[: self._n].copy()

    def _update(self, context, discrete_state) -> None:
        n = self._n
        time = context.get_time()
        state = context.get_discrete_state(self._state_index).get_value().copy()
        headings = state[:n]
        stamps = state[n : 2 * n]
        reference = np.asarray(self._heading_reference(time), dtype=float)
        surge = np.asarray(self._surge_schedule(time), dtype=float).copy()
        if self._scale_port is not None:
            surge *= np.asarray(self._scale_port.Eval(context), dtype=float)
        yaw = np.zeros(n)
        for i in range(n):
            odometry = self._odometry_ports[i].Eval(context)
            stamp = float(odometry[3])
            if stamp > stamps[i] + 1.0e-9:
                if stamps[i] >= 0.0:
                    headings[i] += float(odometry[2]) * (stamp - stamps[i])
                stamps[i] = stamp
            bearing = float(self._bearing_ports[i].Eval(context)[0])
            target = reference[i]
            if self._sway_gain != 0.0:
                # sigma_hat = theta_hat - bearing: world chord angle from sensor outputs only.
                correction = self._sway_gain * wrap_angle(headings[i] - bearing - self._design_chord_angles[i])
                if self._sway_limit is not None:
                    correction = min(max(correction, -self._sway_limit), self._sway_limit)
                target = target - correction
            yaw[i] = self._heading_gain * wrap_angle(target - headings[i]) - self._trim_gain * math.sin(bearing)
        state[:n] = headings
        state[n : 2 * n] = stamps
        state[2 * n : 3 * n] = surge
        state[3 * n :] = yaw
        discrete_state.get_mutable_vector(self._state_index).SetFromVector(state)
