"""Per-vessel error-state filter, slack gating, and slack precursor (Phase 5 spec).

State per vessel, world frame W: load pose with a left-invariant SE(2) error, load body
twist ``(u, v, omega)``, own pose with a left-invariant SE(2) error, surge scale ``beta``,
gyro bias ``b``.  The error state is ``[xi_L, d_zeta, xi_A, d_beta, d_b]`` (11) with
``truth = estimate (+) error``: ``g = g_hat Exp(xi)`` on the poses, addition elsewhere.

Drake-free: sensor constants mirror ``tether/physics/sensors.py`` (checked by a test).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tether.estimation import se2
from tether.physics import constants

# Sensor model of tether/physics/sensors.py.
TICK = 0.01
ODOMETRY_PERIOD = 0.02
BEARING_PERIOD = 0.05
TENSION_PERIOD = 0.02
BEACON_PERIOD = 0.2
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

LOAD_POSE = slice(0, 3)
LOAD_TWIST = slice(3, 6)
LOAD_BLOCK = slice(0, 6)
OWN_POSE = slice(6, 9)
SURGE_SCALE_INDEX = 9
GYRO_BIAS_INDEX = 10
STATE_SIZE = 11
LOAD_SIZE = 6

# Spec: chord length l = L + T/k at 50 Hz with sigma 0.5 mm.  Declared model: the plant's
# cable law T = k e + c edot gives l = |b - a| + (c/k) edot, edot from the chord-rate map
# (its odometry noise added to sigma); the geometric |b - a| alone leaves the post-impact
# ringing (c/k edot up to 9.5 mm, 20 sigma) in the residual for ~0.5 s per re-engagement.
CHORD_NOISE_STD = 0.5e-3
# Spec: T0 = 1 kN design pretension of the Squall Passage, used for the a0 fallback.
DESIGN_PRETENSION = 1000.0
# m_eff = m_A m_L / (m_A + m_L) (IV.12), for a0 = T0 / m_eff.
EFFECTIVE_MASS = constants.VESSEL_MASS * constants.LOAD_MASS / (constants.VESSEL_MASS + constants.LOAD_MASS)
# Default latency tau (IV.12).
REFERENCE_AGE = 0.4
# Spec: slope window, floor on its standard error.
SLOPE_WINDOW = 0.3
SLOPE_STD_FLOOR = 0.05
# Declared: the slack state starts on the first flagged tension sample and ends after
# three consecutive clear samples; during a true slack T = 0 and a sample reads clear
# with probability P(N(0, 25^2) >= 50) = 2.3 %, so a false release needs 1.2e-5.
SLACK_RELEASE_SAMPLES = 3
# Declared survey accuracy of the t = 0 poses (the mission starts from a surveyed formation).
INITIAL_POSITION_STD = 0.01
INITIAL_HEADING_STD = math.radians(0.1)
# Declared prior on the rigid-formation initial load twist: odometry noise (0.028 m/s,
# 0.2 deg/s per sample) with margin for a start that is not exactly rigid.
INITIAL_SPEED_STD = 0.05
INITIAL_YAW_RATE_STD = 0.005
# Declared: the own-pose transition (and the surge noise level) is linearized at a 1 s
# exponential average of the odometry, the mean still integrates the raw sample.  At the
# raw sample the Jacobian carries the sample noise, the gains correlate with the residuals
# that noise drives, and the unobservable states drift: the surge scale by +0.04 in 20 s
# with surge noise alone in the synthetic test.
ODOMETRY_REFERENCE_TIME = 1.0
# The plant's surge scale is fixed per run (IV.6); the walk only keeps it adaptable.
SURGE_SCALE_WALK = 1.0e-5
# The plant's gyro has no rate bias (sensors.py declared deviation: white rate noise
# only), so the spec's bias state is kept nearly pinned: 0.001 deg/s allows 0.13 deg over
# a 130 s mission, below the 0.03 deg/sqrt(s) heading walk of the white gyro noise.  With
# a loose prior (0.05 deg/s) the bias absorbs spurious information on the unobservable
# global rotation (+0.26 deg/s in 20 s in the synthetic test) and drags the own heading.
GYRO_BIAS_STD = math.radians(0.001)
GYRO_BIAS_WALK = math.radians(1.0e-4)
# Q_L is the smallest white body acceleration whose velocity increments bound the
# physical ones at every horizon: q = sup over lags D of D_v(D) / D, reached at D ~ 4 s
# (the 8 s weather memory).  Matching at the 0.4 s packet age instead (0.047 / 9.0e-5)
# leaves the random walk ~6x too stiff for the formation's slow yaw swings: in the 60 s
# production run the yaw-rate estimate stayed near 0 while the truth swung +-0.05 rad/s.
# Load yaw: sup of D_omega(D) / D of the reduced LTI model (theory/reduced_lti.py, fan
# 0.55 rad, T0 = 1 kN, common-mode weather at intensity 1), at D = 4 s; the 60 s
# production run at intensity 0.5 measures 1.27e-4, i.e. 5.1e-4 at intensity 1.
LOAD_YAW_PSD = 5.6e-4


def formation_velocity_increment_rate(lag: float, weather_scale: float = 1.0) -> float:
    """``D_v(lag) / lag`` of the rigid formation under common-mode AR(1) weather.

    Mass ``m_L + N m_A``, drag ``c_L + N c_A``, per-axis force std ``sigma_L + N sigma_A``
    (common mode adds coherently), memory ``tau_w``: ``v`` is the AR(1) force through a
    first-order lag, whose velocity structure function is closed form.
    """
    n = constants.VESSEL_COUNT
    mass = constants.LOAD_MASS + n * constants.VESSEL_MASS
    drag = constants.LOAD_LINEAR_DRAG + n * constants.VESSEL_LINEAR_DRAG
    force = weather_scale * (constants.LOAD_WEATHER_STD + n * constants.VESSEL_WEATHER_STD)
    tau_w = constants.WEATHER_TIME_CONSTANT
    tau_m = mass / drag
    structure = (
        2.0
        * (force / drag) ** 2
        * tau_w
        / (tau_w**2 - tau_m**2)
        * (tau_w * (1.0 - math.exp(-lag / tau_w)) - tau_m * (1.0 - math.exp(-lag / tau_m)))
    )
    return structure / lag


# Load translation: sup of the rigid-formation structure function, 0.141 m^2/s^3 at
# D = 4.0 s, intensity 1 (reduced LTI fan model 0.131 / 0.079 along / across; the 60 s
# production run at intensity 0.5 measures 0.036 / 0.020, i.e. 0.143 / 0.082).
LOAD_TRANSLATION_PSD = max(formation_velocity_increment_rate(lag) for lag in np.geomspace(0.1, 60.0, 400))


@dataclass(frozen=True)
class EstimatorGeometry:
    """Load attachment offsets r_i (load frame), stern offsets s_i (vessel frame), cable."""

    load_offsets: np.ndarray
    stern_offsets: np.ndarray
    rest_length: float = constants.CABLE_REST_LENGTH
    stiffness: float = constants.CABLE_STIFFNESS
    damping: float = constants.CABLE_DAMPING

    @property
    def vessel_count(self) -> int:
        return int(np.asarray(self.load_offsets).shape[0])


@dataclass(frozen=True)
class EstimatorParameters:
    """Declared filter, fusion, and precursor settings; defaults are the Phase 5 values."""

    weather_scale: float = 1.0
    load_translation_psd: float = LOAD_TRANSLATION_PSD
    load_yaw_psd: float = LOAD_YAW_PSD
    chord_noise_std: float = CHORD_NOISE_STD
    pretension: float = DESIGN_PRETENSION
    precursor_growth: float = 0.0
    slack_release_samples: int = SLACK_RELEASE_SAMPLES
    graph: str = "cycle"
    drop_probability: float = 0.0
    initial_position_std: float = INITIAL_POSITION_STD
    initial_heading_std: float = INITIAL_HEADING_STD
    initial_speed_std: float = INITIAL_SPEED_STD
    initial_yaw_rate_std: float = INITIAL_YAW_RATE_STD

    def load_process_psd(self) -> np.ndarray:
        """Q_L: white body-frame acceleration PSD of the load twist, scaled by intensity^2."""
        scale = self.weather_scale**2
        return scale * np.array([self.load_translation_psd, self.load_translation_psd, self.load_yaw_psd])


@dataclass(frozen=True)
class SensorLog:
    """Every sensor sample of one run (spec: Sensor log format)."""

    odometry: tuple[np.ndarray, ...]
    bearing: tuple[np.ndarray, ...]
    tension: tuple[np.ndarray, ...]
    beacon: np.ndarray
    initial_poses: np.ndarray
    geometry: EstimatorGeometry

    @property
    def vessel_count(self) -> int:
        return len(self.odometry)

    @property
    def end_time(self) -> float:
        streams = [*self.odometry, *self.bearing, *self.tension, self.beacon]
        return max(float(stream[-1, 0]) for stream in streams if stream.shape[0])


@dataclass(frozen=True)
class EstimatorOutput:
    """One arm's monitor outputs at 10 Hz (spec: Interfaces).

    ``load_cov`` is expressed in the coordinates of ``load_mean``: world ``(dx, dy)``,
    ``d theta``, body ``(du, dv, d omega)`` (the first-order image of the invariant error).
    Precursor fields are nan while the vessel is taut.  ``precursor_growth`` is the q_L
    already included in ``sigma`` (arm L).
    """

    arm: str
    time: np.ndarray
    slack: np.ndarray
    onset_time: np.ndarray
    e_hat: np.ndarray
    edot_hat: np.ndarray
    sigma: np.ndarray
    a_hat: np.ndarray
    sigma_a: np.ndarray
    load_mean: np.ndarray
    load_cov: np.ndarray
    precursor_growth: float = 0.0


@dataclass(frozen=True)
class NominalState:
    load_pose: np.ndarray
    load_twist: np.ndarray
    own_pose: np.ndarray
    surge_scale: float
    gyro_bias: float


def retract(state: NominalState, delta: np.ndarray) -> NominalState:
    """``estimate (+) delta`` on the 11-dimensional error state."""
    return NominalState(
        load_pose=se2.compose(state.load_pose, se2.exp(delta[LOAD_POSE])),
        load_twist=state.load_twist + delta[LOAD_TWIST],
        own_pose=se2.compose(state.own_pose, se2.exp(delta[OWN_POSE])),
        surge_scale=state.surge_scale + float(delta[SURGE_SCALE_INDEX]),
        gyro_bias=state.gyro_bias + float(delta[GYRO_BIAS_INDEX]),
    )


_PERP = np.array([[0.0, -1.0], [1.0, 0.0]])


def _attachments(state: NominalState, load_offset: np.ndarray, stern_offset: np.ndarray):
    """Load point a, stern point b, rotations, and c = R_A^T (a - b) in the vessel frame."""
    load_rotation = se2.rotation(state.load_pose[2])
    own_rotation = se2.rotation(state.own_pose[2])
    a = state.load_pose[:2] + load_rotation @ load_offset
    b = state.own_pose[:2] + own_rotation @ stern_offset
    c = own_rotation.T @ (a - b)
    return a, b, c, load_rotation, own_rotation


def _chord_vector_jacobian(state, load_offset, stern_offset, c, load_rotation, own_rotation) -> np.ndarray:
    """d c / d error (2 x 11)."""
    jacobian = np.zeros((2, STATE_SIZE))
    jacobian[:, 0:2] = own_rotation.T @ load_rotation
    jacobian[:, 2] = own_rotation.T @ load_rotation @ (_PERP @ load_offset)
    jacobian[:, 6:8] = -np.eye(2)
    jacobian[:, 8] = -_PERP @ (c + stern_offset)
    return jacobian


def bearing_model(state: NominalState, load_offset: np.ndarray, stern_offset: np.ndarray) -> tuple[float, np.ndarray]:
    """Cable bearing (0 dead astern, + to port) and its Jacobian (11,)."""
    _, _, c, load_rotation, own_rotation = _attachments(state, load_offset, stern_offset)
    value = math.atan2(c[1], -c[0])
    gradient = np.array([c[1], -c[0]]) / float(c @ c)
    return value, gradient @ _chord_vector_jacobian(state, load_offset, stern_offset, c, load_rotation, own_rotation)


def range_model(state: NominalState, load_offset: np.ndarray, stern_offset: np.ndarray) -> tuple[float, np.ndarray]:
    """Chord length |b - a| and its Jacobian (11,)."""
    _, _, c, load_rotation, own_rotation = _attachments(state, load_offset, stern_offset)
    length = math.sqrt(float(c @ c))
    return length, (c / length) @ _chord_vector_jacobian(state, load_offset, stern_offset, c, load_rotation, own_rotation)


def beacon_model(state: NominalState) -> tuple[np.ndarray, np.ndarray]:
    """Load pose ``(x, y, theta)`` and its Jacobian (3 x 11)."""
    jacobian = np.zeros((3, STATE_SIZE))
    jacobian[0:2, 0:2] = se2.rotation(state.load_pose[2])
    jacobian[2, 2] = 1.0
    return state.load_pose.copy(), jacobian


def corrected_twist(state: NominalState, odometry: np.ndarray) -> np.ndarray:
    """Bias-corrected own body twist ``(surge / (1 + beta), sway, gyro - b)``."""
    return np.array([odometry[0] / (1.0 + state.surge_scale), odometry[1], odometry[2] - state.gyro_bias])


def odometry_noise_covariance(surge_level: float) -> np.ndarray:
    """Per-sample odometry noise of sensors.py (surge, sway, gyro incl. the ARW term)."""
    surge = SURGE_NOISE_BASE + SURGE_NOISE_SLOPE * abs(float(surge_level))
    gyro = GYRO_NOISE_STD**2 + GYRO_ARW_DENSITY**2 / ODOMETRY_PERIOD
    return np.diag([surge * surge, SWAY_NOISE_STD**2, gyro])


def _twist_noise_map(state: NominalState, surge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """d (true own twist) / d (beta, b, sample noise) at measured surge ``surge``."""
    scale = 1.0 + state.surge_scale
    beta_column = np.array([-surge / scale**2, 0.0, 0.0])
    bias_column = np.array([0.0, 0.0, -1.0])
    noise = np.diag([-1.0 / scale, -1.0, -1.0])
    return beta_column, bias_column, noise


def chord_rate_model(
    state: NominalState, odometry: np.ndarray, load_offset: np.ndarray, stern_offset: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """``edot = (v_b - v_a) . d_hat`` with Jacobians on the error (11,) and on the sample noise (3,)."""
    a, b, _, load_rotation, own_rotation = _attachments(state, load_offset, stern_offset)
    chord = b - a
    length = math.sqrt(float(chord @ chord))
    direction = chord / length
    twist = state.load_twist
    own = corrected_twist(state, odometry)
    load_body = twist[:2] + twist[2] * (_PERP @ load_offset)
    own_body = own[:2] + own[2] * (_PERP @ stern_offset)
    load_velocity = load_rotation @ load_body
    own_velocity = own_rotation @ own_body
    relative = own_velocity - load_velocity
    value = float(relative @ direction)
    projector = (np.eye(2) - np.outer(direction, direction)) / length
    lever = relative @ projector
    chord_jacobian = np.zeros((2, STATE_SIZE))
    chord_jacobian[:, 0:2] = -load_rotation
    chord_jacobian[:, 2] = -load_rotation @ (_PERP @ load_offset)
    chord_jacobian[:, 6:8] = own_rotation
    chord_jacobian[:, 8] = own_rotation @ (_PERP @ stern_offset)
    velocity_jacobian = np.zeros((2, STATE_SIZE))
    velocity_jacobian[:, 2] = -load_rotation @ (_PERP @ load_body)
    velocity_jacobian[:, 3:5] = -load_rotation
    velocity_jacobian[:, 5] = -load_rotation @ (_PERP @ load_offset)
    velocity_jacobian[:, 8] = own_rotation @ (_PERP @ own_body)
    beta_column, bias_column, noise = _twist_noise_map(state, float(odometry[0]))
    stern_map = np.column_stack([np.array([1.0, 0.0]), np.array([0.0, 1.0]), _PERP @ stern_offset])
    velocity_jacobian[:, SURGE_SCALE_INDEX] = own_rotation @ (stern_map @ beta_column)
    velocity_jacobian[:, GYRO_BIAS_INDEX] = own_rotation @ (stern_map @ bias_column)
    jacobian = direction @ velocity_jacobian + lever @ chord_jacobian
    noise_jacobian = direction @ (own_rotation @ (stern_map @ noise))
    return value, jacobian, noise_jacobian


def tension_chord_model(
    state: NominalState, odometry: np.ndarray, load_offset: np.ndarray, stern_offset: np.ndarray, relaxation: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """``L + T / k = l + (c / k) edot`` of a taut Kelvin-Voigt cable, with Jacobians on the
    error (11,) and on the odometry sample noise (3,); ``relaxation = c / k``."""
    length, length_jacobian = range_model(state, load_offset, stern_offset)
    rate, rate_jacobian, noise_jacobian = chord_rate_model(state, odometry, load_offset, stern_offset)
    return length + relaxation * rate, length_jacobian + relaxation * rate_jacobian, relaxation * noise_jacobian


def random_acceleration_noise(psd: np.ndarray, duration: float) -> np.ndarray:
    """Pose/twist covariance (6 x 6) of a white body acceleration of PSD ``psd`` over ``duration``."""
    diagonal = np.diag(psd)
    noise = np.zeros((LOAD_SIZE, LOAD_SIZE))
    noise[0:3, 0:3] = diagonal * duration**3 / 3.0
    noise[0:3, 3:6] = diagonal * duration**2 / 2.0
    noise[3:6, 0:3] = diagonal * duration**2 / 2.0
    noise[3:6, 3:6] = diagonal * duration
    return noise


def constant_twist_transition(twist: np.ndarray, duration: float) -> np.ndarray:
    """Error transition (6 x 6) of the load block over ``duration`` at constant ``twist``."""
    step = twist * duration
    transition = np.eye(LOAD_SIZE)
    transition[0:3, 0:3] = se2.adjoint(se2.exp(-step))
    transition[0:3, 3:6] = se2.right_jacobian(step) * duration
    return transition


def initial_covariance(parameters: EstimatorParameters) -> np.ndarray:
    position = parameters.initial_position_std**2
    heading = parameters.initial_heading_std**2
    speed = parameters.initial_speed_std**2
    yaw_rate = parameters.initial_yaw_rate_std**2
    return np.diag(
        [position, position, heading, speed, speed, yaw_rate, position, position, heading, SURGE_SCALE_STD**2, GYRO_BIAS_STD**2]
    )


class VesselFilter:
    """Vessel ``agent``'s 11-state error-state filter with constraint gating."""

    def __init__(
        self,
        agent: int,
        geometry: EstimatorGeometry,
        parameters: EstimatorParameters,
        load_pose: np.ndarray,
        own_pose: np.ndarray,
        load_twist: np.ndarray | None = None,
        time: float = 0.0,
    ) -> None:
        self.agent = agent
        self.geometry = geometry
        self.parameters = parameters
        self.load_offset = np.asarray(geometry.load_offsets[agent], dtype=float)
        self.stern_offset = np.asarray(geometry.stern_offsets[agent], dtype=float)
        self.state = NominalState(
            np.asarray(load_pose, dtype=float).copy(),
            np.zeros(3) if load_twist is None else np.asarray(load_twist, dtype=float).copy(),
            np.asarray(own_pose, dtype=float).copy(),
            0.0,
            0.0,
        )
        self.covariance = initial_covariance(parameters)
        self.time = float(time)
        self.odometry: np.ndarray | None = None
        self.odometry_reference: np.ndarray | None = None
        self.slack = False
        self.slack_onset = math.nan
        self._clear_run = 0
        self._load_psd = parameters.load_process_psd()
        self.update_log: list[tuple[float, str]] = []

    def initialize_twist(self, odometry: np.ndarray) -> None:
        """Rigid-formation initial load twist from the first odometry sample.

        The load turns at the vessel's gyro rate and its attachment moves with the stern
        less the rigid rotation of the chord: ``v_a = v_b - omega J (b - a)``.
        """
        own = corrected_twist(self.state, odometry)
        a, b, _, load_rotation, own_rotation = _attachments(self.state, self.load_offset, self.stern_offset)
        stern_velocity = own_rotation @ (own[:2] + own[2] * (_PERP @ self.stern_offset))
        attachment_velocity = stern_velocity - own[2] * (_PERP @ (b - a))
        body = load_rotation.T @ attachment_velocity - own[2] * (_PERP @ self.load_offset)
        self.state = NominalState(
            self.state.load_pose, np.array([body[0], body[1], own[2]]), self.state.own_pose, self.state.surge_scale, self.state.gyro_bias
        )

    def own_twist(self) -> np.ndarray:
        return np.zeros(3) if self.odometry is None else corrected_twist(self.state, self.odometry)

    def propagation_matrices(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Error transition and process noise over ``dt`` from the current estimate."""
        state = self.state
        transition = np.eye(STATE_SIZE)
        noise = np.zeros((STATE_SIZE, STATE_SIZE))
        transition[LOAD_BLOCK, LOAD_BLOCK] = constant_twist_transition(state.load_twist, dt)
        noise[LOAD_BLOCK, LOAD_BLOCK] = random_acceleration_noise(self._load_psd, dt)
        if self.odometry is not None:
            reference = corrected_twist(state, self.odometry_reference) * dt
            transition[OWN_POSE, OWN_POSE] = se2.adjoint(se2.exp(-reference))
            coupling = se2.right_jacobian(reference) * dt
            beta_column, bias_column, sample_noise = _twist_noise_map(state, float(self.odometry_reference[0]))
            transition[OWN_POSE, SURGE_SCALE_INDEX] = coupling @ beta_column
            transition[OWN_POSE, GYRO_BIAS_INDEX] = coupling @ bias_column
            mapped = coupling @ sample_noise
            hold = max(dt, ODOMETRY_PERIOD)
            sample_covariance = odometry_noise_covariance(float(self.odometry_reference[0]))
            noise[OWN_POSE, OWN_POSE] = mapped @ sample_covariance @ mapped.T * (hold / dt)
        noise[SURGE_SCALE_INDEX, SURGE_SCALE_INDEX] = SURGE_SCALE_WALK**2 * dt
        noise[GYRO_BIAS_INDEX, GYRO_BIAS_INDEX] = GYRO_BIAS_WALK**2 * dt
        return transition, noise

    def propagate_to(self, time: float) -> None:
        """Constant load twist, own pose on the held bias-corrected odometry sample."""
        dt = time - self.time
        if dt <= 1.0e-12:
            return
        state = self.state
        transition, noise = self.propagation_matrices(dt)
        step = self.own_twist() * dt
        self.state = NominalState(
            se2.compose(state.load_pose, se2.exp(state.load_twist * dt)),
            state.load_twist,
            se2.compose(state.own_pose, se2.exp(step)),
            state.surge_scale,
            state.gyro_bias,
        )
        covariance = transition @ self.covariance @ transition.T + noise
        self.covariance = 0.5 * (covariance + covariance.T)
        self.time = time

    def propagate(self, time: float, surge: float, sway: float, gyro: float) -> None:
        """Odometry sample: propagate to its stamp on the held sample, then hold it."""
        self.propagate_to(time)
        self.odometry = np.array([surge, sway, gyro], dtype=float)
        if self.odometry_reference is None:
            self.odometry_reference = self.odometry.copy()
        else:
            self.odometry_reference += ODOMETRY_PERIOD / ODOMETRY_REFERENCE_TIME * (self.odometry - self.odometry_reference)

    def apply_update(self, residual: np.ndarray, jacobian: np.ndarray, noise: np.ndarray, kind: str) -> None:
        """Kalman update of the error state (Joseph form) and injection into the estimate."""
        covariance = self.covariance
        projected = covariance @ jacobian.T
        innovation = jacobian @ projected + noise
        try:
            gain = np.linalg.solve(innovation, projected.T).T
        except np.linalg.LinAlgError:
            # Numerical guard: naive fusion (B2) can drive the covariance to machine precision
            # on long runs, making the innovation exactly singular; the minimum-norm gain
            # leaves the directions that carry no information unchanged.
            gain = projected @ np.linalg.pinv(innovation)
        delta = gain @ residual
        reduction = np.eye(STATE_SIZE) - gain @ jacobian
        covariance = reduction @ covariance @ reduction.T + gain @ noise @ gain.T
        self.covariance = 0.5 * (covariance + covariance.T)
        self.state = retract(self.state, delta)
        self.update_log.append((self.time, kind))

    def update_bearing(self, bearing: float) -> None:
        predicted, jacobian = bearing_model(self.state, self.load_offset, self.stern_offset)
        residual = np.array([se2.wrap(bearing - predicted)])
        self.apply_update(residual, jacobian[None, :], np.array([[BEARING_NOISE_STD**2]]), "bearing")

    def update_range(self, tension: float) -> None:
        """Chord length l = L + T / k from the measured tension (Kelvin-Voigt model)."""
        odometry = np.zeros(3) if self.odometry is None else self.odometry
        geometry = self.geometry
        predicted, jacobian, noise_jacobian = tension_chord_model(
            self.state, odometry, self.load_offset, self.stern_offset, geometry.damping / geometry.stiffness
        )
        level = 0.0 if self.odometry_reference is None else float(self.odometry_reference[0])
        variance = self.parameters.chord_noise_std**2 + float(noise_jacobian @ odometry_noise_covariance(level) @ noise_jacobian)
        measured = geometry.rest_length + tension / geometry.stiffness
        self.apply_update(np.array([measured - predicted]), jacobian[None, :], np.array([[variance]]), "range")

    def update_beacon(self, x: float, y: float, theta: float) -> None:
        predicted, jacobian = beacon_model(self.state)
        residual = np.array([x - predicted[0], y - predicted[1], se2.wrap(theta - predicted[2])])
        noise = np.diag([BEACON_POSITION_STD**2, BEACON_POSITION_STD**2, BEACON_ANGLE_STD**2])
        self.apply_update(residual, jacobian, noise, "beacon")

    def process_tension(self, time: float, tension: float, flag: float) -> None:
        """Debounced slack state; the range update only while taut."""
        if flag >= 0.5:
            if not self.slack:
                self.slack = True
                self.slack_onset = float(time)
            self._clear_run = 0
            return
        if self.slack:
            self._clear_run += 1
            if self._clear_run < self.parameters.slack_release_samples:
                return
            self.slack = False
            self.slack_onset = math.nan
            self._clear_run = 0
        self.update_range(tension)

    def process_bearing(self, bearing: float) -> None:
        if not self.slack:
            self.update_bearing(bearing)

    def chord_rate(self) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """``(edot_hat, J, J_noise, R_noise)`` from the held odometry sample."""
        odometry = np.zeros(3) if self.odometry is None else self.odometry
        value, jacobian, noise_jacobian = chord_rate_model(self.state, odometry, self.load_offset, self.stern_offset)
        level = 0.0 if self.odometry_reference is None else float(self.odometry_reference[0])
        return value, jacobian, noise_jacobian, odometry_noise_covariance(level)

    def chord_length_variance(self) -> float:
        _, jacobian = range_model(self.state, self.load_offset, self.stern_offset)
        return float(jacobian @ self.covariance @ jacobian)

    def load_packet(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(pose, twist, covariance)`` of the load block in this filter's error coordinates."""
        return self.state.load_pose.copy(), self.state.load_twist.copy(), self.covariance[LOAD_BLOCK, LOAD_BLOCK].copy()

    def load_output(self) -> tuple[np.ndarray, np.ndarray]:
        """``load_mean`` and ``load_cov`` in the coordinates of the mean (see EstimatorOutput)."""
        mean = np.concatenate([self.state.load_pose, self.state.load_twist])
        frame = np.eye(LOAD_SIZE)
        frame[0:2, 0:2] = se2.rotation(self.state.load_pose[2])
        return mean, frame @ self.covariance[LOAD_BLOCK, LOAD_BLOCK] @ frame.T


class Precursor:
    """Slack precursor ``(e_hat, edot_hat, Sigma, a_hat, sigma_a)`` of one vessel.

    ``edot_hat`` is the geometric chord rate; ``e_hat`` integrates it on the tick grid
    from 0 at the onset.  ``Sigma``: ``Var(edot) = J P J' + J_n R J_n'`` (filter covariance
    and held odometry sample through the Jacobian of the geometric map).  ``Var(e)`` follows
    ``e_{k+1} = e_k + edot_k dt`` with the edot variance as input noise, declared as a 2 x 2
    recursion on ``(e, edot)``: the state part of the edot error persists from tick to tick
    (transition ``[[1, dt], [0, 1]]``) and its variance is reset to ``J P J'`` at each tick,
    a decrease scaling the cross term by the ratio of standard deviations; the white sample
    part enters ``Var(e)`` as ``dt^2`` per tick (twice for the 20 ms hold).  The result is
    positive semidefinite and ``Var(e)`` stays below ``(sum dt sigma_edot)^2`` whatever the
    fusion rule's correlations.  The onset value is uncertain by one tension period of
    travel plus the damping offset ``c edot / k`` and the flag threshold (declared uniform).
    """

    def __init__(self, geometry: EstimatorGeometry, parameters: EstimatorParameters) -> None:
        self._geometry = geometry
        self._fallback = parameters.pretension / EFFECTIVE_MASS
        self._window_ticks = int(round(SLOPE_WINDOW / TICK))
        self.active = False
        self.onset_time = math.nan
        self._reset_outputs()

    def _reset_outputs(self) -> None:
        self.e_hat = math.nan
        self.edot_hat = math.nan
        self.sigma = np.full((2, 2), math.nan)
        self.a_hat = math.nan
        self.sigma_a = math.nan
        self._e_next = math.nan
        self._state_covariance = np.full((2, 2), math.nan)
        self._times: list[float] = []
        self._rates: list[float] = []

    def onset_variance(self, rate: float) -> float:
        geometry = self._geometry
        width = abs(rate) * (TENSION_PERIOD + geometry.damping / geometry.stiffness) + SLACK_FLAG_THRESHOLD / geometry.stiffness
        return width * width / 3.0

    def step(self, time: float, vessel: VesselFilter) -> None:
        """Evaluate at this tick (after its updates), then advance ``e`` to the next tick."""
        if not vessel.slack:
            if self.active:
                self.active = False
                self.onset_time = math.nan
                self._reset_outputs()
            return
        rate, jacobian, noise_jacobian, noise = vessel.chord_rate()
        state_variance = float(jacobian @ vessel.covariance @ jacobian)
        sample_variance = float(noise_jacobian @ noise @ noise_jacobian)
        covariance = self._state_covariance
        if not self.active:
            self.active = True
            self.onset_time = vessel.slack_onset
            self._e_next = 0.0
            covariance = np.array([[self.onset_variance(rate), 0.0], [0.0, state_variance]])
        else:
            if state_variance < covariance[1, 1]:
                covariance[0, 1] = covariance[1, 0] = covariance[0, 1] * math.sqrt(state_variance / covariance[1, 1])
            covariance[1, 1] = state_variance
        self.edot_hat = rate
        self.sigma = covariance + np.array([[0.0, 0.0], [0.0, sample_variance]])
        self._times.append(time)
        self._rates.append(rate)
        del self._times[: -self._window_ticks - 1]
        del self._rates[: -self._window_ticks - 1]
        self.a_hat, self.sigma_a = self._slope(time)
        # Outputs describe this tick: e_hat is the value at ``time``, like edot_hat and
        # Sigma; the Euler step to the next tick is carried separately.
        self.e_hat = self._e_next
        self._e_next = self.e_hat + TICK * rate
        transition = np.array([[1.0, TICK], [0.0, 1.0]])
        covariance = transition @ covariance @ transition.T
        covariance[0, 0] += 2.0 * TICK * TICK * sample_variance
        self._state_covariance = covariance

    def _slope(self, time: float) -> tuple[float, float]:
        times = np.asarray(self._times)
        keep = times >= time - SLOPE_WINDOW - 1.0e-9
        times = times[keep]
        if times.size < 3:
            return self._fallback, SLOPE_STD_FLOOR
        rates = np.asarray(self._rates)[keep]
        centred = times - times.mean()
        spread = float(centred @ centred)
        slope = float(centred @ (rates - rates.mean())) / spread
        residual = rates - rates.mean() - slope * centred
        error = math.sqrt(float(residual @ residual) / (times.size - 2) / spread)
        return slope, max(error, SLOPE_STD_FLOOR)
