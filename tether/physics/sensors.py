"""Per-vessel sensor suites (plan IV.6).

One ``SensorSuite`` per vessel runs a single 100 Hz discrete update with internal
dispatch: odometry and tension at 50 Hz, cable bearing at 20 Hz, and the load beacon at
5 Hz on vessel 0 after ``beacon_on``.  Each suite holds its own generator seeded by
``SeedSequence([master, agent, 7])`` and draws only inside that update.  Sensor suites
are plant-side, and their outputs are the sanctioned measurements that non-plant-side
systems may read.

Gyro interpretation (declared deviation): the plan's "b a random walk of 0.01 deg/sqrt(s)"
carries angle-random-walk units, so ``b`` is drawn as white rate noise of density
0.01 deg/sqrt(s) in addition to the 0.2 deg/s per-sample noise.  Read instead as a
rate-bias random walk (deg/s/sqrt(s)), it would drift the gyro-integrated heading by
roughly 85 deg (1 sigma) over a 600 s cell.
"""

from __future__ import annotations

import math

import numpy as np
from pydrake.systems.framework import LeafSystem

from tether.physics.fleet import FleetGeometry, cable_kinematics

SENSOR_STREAM = 7
SENSOR_PERIOD = 0.01
ODOMETRY_DIVISOR = 2
BEARING_DIVISOR = 5
TENSION_DIVISOR = 2
BEACON_DIVISOR = 20
SURGE_SCALE_STD = 0.01
SURGE_NOISE_BASE = 0.01
SURGE_NOISE_SLOPE = 0.02
SWAY_NOISE_STD = 0.01
GYRO_NOISE_STD = math.radians(0.2)
GYRO_ARW_DENSITY = math.radians(0.01)
BEARING_NOISE_STD = math.radians(1.0)
TENSION_NOISE_STD = 25.0
SLACK_FLAG_THRESHOLD = 50.0
BEACON_POSITION_STD = 0.05
BEACON_ANGLE_STD = math.radians(0.5)
DEFAULT_BEACON_ON = 30.0

ODOMETRY_SIZE = 4  # surge, sway, gyro, stamp
BEARING_SIZE = 2  # bearing, stamp
TENSION_SIZE = 3  # tension, slack flag, stamp
BEACON_SIZE = 5  # x, y, theta, valid, stamp


def sensor_generator(master_seed: int, agent: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([master_seed, agent, SENSOR_STREAM]))


def cable_bearing(state: np.ndarray, geometry: FleetGeometry, agent: int, rest_length: float) -> float:
    """Bearing of the load attachment seen from the stern, vessel frame; 0 dead astern.

    Positive when the attachment lies to port of astern.
    """
    kinematics = cable_kinematics(state, geometry, rest_length)
    n = geometry.vessel_count
    heading = state[3 * (agent + 1) + 2]
    to_load = kinematics["load_point"][agent] - kinematics["vessel_point"][agent]
    cosine, sine = math.cos(heading), math.sin(heading)
    body_x = cosine * to_load[0] + sine * to_load[1]
    body_y = -sine * to_load[0] + cosine * to_load[1]
    del n
    return math.atan2(body_y, -body_x)


class SensorSuite(LeafSystem):
    """Noisy odometry, gyro, cable bearing, tension, and (vessel 0) beacon."""

    PLANT_SIDE = True
    SANCTIONED_OUTPUTS = True

    def __init__(
        self,
        geometry: FleetGeometry,
        agent: int,
        master_seed: int,
        rest_length: float,
        beacon_on: float | None = DEFAULT_BEACON_ON,
    ) -> None:
        super().__init__()
        self._geometry = geometry
        self._agent = agent
        self._rest_length = rest_length
        self._beacon_on = beacon_on if agent == 0 else None
        self._generator = sensor_generator(master_seed, agent)
        self._surge_scale: float | None = None
        # Every emitted sample, for offline replay of the estimators (recording mode).
        self.samples: dict[str, list] = {"odometry": [], "bearing": [], "tension": [], "beacon": []}
        n = geometry.vessel_count
        self._state_port = self.DeclareVectorInputPort("plant_state", 6 * (n + 1))
        self._tension_port = self.DeclareVectorInputPort("cable_tension", n)
        self._odometry = self.DeclareDiscreteState(ODOMETRY_SIZE)
        self._bearing = self.DeclareDiscreteState(BEARING_SIZE)
        self._tension = self.DeclareDiscreteState(TENSION_SIZE)
        self._beacon = self.DeclareDiscreteState(BEACON_SIZE)
        self.DeclareStateOutputPort("odometry", self._odometry)
        self.DeclareStateOutputPort("cable_bearing", self._bearing)
        self.DeclareStateOutputPort("tension", self._tension)
        self.DeclareStateOutputPort("beacon", self._beacon)
        self.DeclarePeriodicDiscreteUpdateEvent(SENSOR_PERIOD, 0.0, self._update)

    def _update(self, context, discrete_state) -> None:
        time = context.get_time()
        tick = int(round(time / SENSOR_PERIOD))
        generator = self._generator
        if self._surge_scale is None:
            self._surge_scale = float(generator.normal(0.0, SURGE_SCALE_STD))
        state = self._state_port.Eval(context)
        n = self._geometry.vessel_count
        agent = self._agent
        base = 3 * (agent + 1)
        heading = state[base + 2]
        velocity = state[3 * (n + 1) + base : 3 * (n + 1) + base + 3]
        cosine, sine = math.cos(heading), math.sin(heading)
        surge_true = cosine * velocity[0] + sine * velocity[1]
        sway_true = -sine * velocity[0] + cosine * velocity[1]
        if tick % ODOMETRY_DIVISOR == 0:
            dt = SENSOR_PERIOD * ODOMETRY_DIVISOR
            surge = surge_true * (1.0 + self._surge_scale) + generator.normal(
                0.0, SURGE_NOISE_BASE + SURGE_NOISE_SLOPE * abs(surge_true)
            )
            sway = sway_true + generator.normal(0.0, SWAY_NOISE_STD)
            gyro = (
                velocity[2]
                + generator.normal(0.0, GYRO_ARW_DENSITY / math.sqrt(dt))
                + generator.normal(0.0, GYRO_NOISE_STD)
            )
            discrete_state.get_mutable_vector(self._odometry).SetFromVector([surge, sway, gyro, time])
            self.samples["odometry"].append((time, surge, sway, gyro))
        if tick % BEARING_DIVISOR == 0:
            bearing = cable_bearing(state, self._geometry, agent, self._rest_length)
            bearing += generator.normal(0.0, BEARING_NOISE_STD)
            discrete_state.get_mutable_vector(self._bearing).SetFromVector([bearing, time])
            self.samples["bearing"].append((time, bearing))
        if tick % TENSION_DIVISOR == 0:
            tension = float(self._tension_port.Eval(context)[agent]) + generator.normal(0.0, TENSION_NOISE_STD)
            slack = 1.0 if tension < SLACK_FLAG_THRESHOLD else 0.0
            discrete_state.get_mutable_vector(self._tension).SetFromVector([tension, slack, time])
            self.samples["tension"].append((time, tension, slack))
        if (
            self._beacon_on is not None
            and time >= self._beacon_on - 1.0e-9
            and tick % BEACON_DIVISOR == 0
        ):
            load = state[0:3]
            beacon = [
                load[0] + generator.normal(0.0, BEACON_POSITION_STD),
                load[1] + generator.normal(0.0, BEACON_POSITION_STD),
                load[2] + generator.normal(0.0, BEACON_ANGLE_STD),
                1.0,
                time,
            ]
            discrete_state.get_mutable_vector(self._beacon).SetFromVector(beacon)
            self.samples["beacon"].append((time, beacon[0], beacon[1], beacon[2]))
