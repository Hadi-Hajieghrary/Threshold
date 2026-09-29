"""Deterministic Phase 0 weather innovations and AR(1) state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tether.physics import constants

WEATHER_STREAM = 11
STUDENT_T_DOF = 3


@dataclass(frozen=True)
class WeatherRecord:
    standardized_innovations: np.ndarray
    innovations: np.ndarray
    states: np.ndarray


def ar1_coefficient(
    period: float = constants.WEATHER_PERIOD,
    time_constant: float = constants.WEATHER_TIME_CONSTANT,
) -> float:
    return float(np.exp(-period / time_constant))


def weather_generator(master_seed: int, body: int) -> np.random.Generator:
    """Create the plan-mandated independent [master, 11, body] stream."""
    return np.random.default_rng(np.random.SeedSequence([master_seed, WEATHER_STREAM, body]))


def _draw_standardized(
    generator: np.random.Generator,
    sample_count: int,
    distribution: str,
) -> np.ndarray:
    gaussian = generator.normal(size=(sample_count, 2))
    if distribution == "gaussian":
        return gaussian
    if distribution == "student_t3":
        radial_scale = np.sqrt(
            (STUDENT_T_DOF - 2.0)
            / generator.chisquare(STUDENT_T_DOF, size=(sample_count, 1))
        )
        return gaussian * radial_scale
    raise ValueError(f"unknown innovation distribution: {distribution}")


def standardized_innovations(
    master_seed: int,
    sample_count: int,
    body_count: int,
    distribution: str = "gaussian",
    direction: str = "local",
) -> np.ndarray:
    """Draw standardized local or exactly shared common innovations."""
    if direction == "common":
        common = _draw_standardized(weather_generator(master_seed, 0), sample_count, distribution)
        return np.repeat(common[:, np.newaxis, :], body_count, axis=1)
    if direction == "local":
        streams = [
            _draw_standardized(weather_generator(master_seed, body), sample_count, distribution)
            for body in range(body_count)
        ]
        return np.stack(streams, axis=1)
    raise ValueError(f"unknown direction structure: {direction}")


def apply_ar1(innovations: np.ndarray, phi: float | None = None) -> np.ndarray:
    """Apply the weather AR(1) recursion from a deterministic zero initial state."""
    if phi is None:
        phi = ar1_coefficient()
    states = np.empty_like(innovations)
    previous = np.zeros(innovations.shape[1:])
    for sample_index in range(innovations.shape[0]):
        previous = phi * previous + innovations[sample_index]
        states[sample_index] = previous
    return states


def generate_weather(
    master_seed: int,
    sample_count: int,
    vessel_count: int = constants.VESSEL_COUNT,
    distribution: str = "gaussian",
    direction: str = "local",
) -> WeatherRecord:
    """Generate variance-matched force innovations and their AR(1) states."""
    body_count = vessel_count + 1
    standardized = standardized_innovations(
        master_seed, sample_count, body_count, distribution, direction
    )
    stationary_std = np.array(
        [constants.LOAD_WEATHER_STD] + [constants.VESSEL_WEATHER_STD] * vessel_count
    )
    innovation_scale = stationary_std * np.sqrt(1.0 - ar1_coefficient() ** 2)
    innovations = standardized * innovation_scale[np.newaxis, :, np.newaxis]
    return WeatherRecord(
        standardized_innovations=standardized,
        innovations=innovations,
        states=apply_ar1(innovations),
    )


MIXED_LOCAL_STREAM_OFFSET = 100
DEFAULT_PREROLL_SECONDS = 40.0


def mixed_standardized_innovations(
    master_seed: int,
    sample_count: int,
    body_count: int,
    distribution: str,
    rho: float,
) -> np.ndarray:
    """``mixed(rho)``: sqrt(rho) common + sqrt(1 - rho) local, unit total variance.

    The common part uses the ``[master, 11, 0]`` stream exactly as ``common`` does; the
    local parts use ``[master, 11, 100 + body]`` so they never alias the common stream.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    common = _draw_standardized(weather_generator(master_seed, 0), sample_count, distribution)
    local = np.stack(
        [
            _draw_standardized(
                weather_generator(master_seed, MIXED_LOCAL_STREAM_OFFSET + body),
                sample_count,
                distribution,
            )
            for body in range(body_count)
        ],
        axis=1,
    )
    return np.sqrt(rho) * common[:, np.newaxis, :] + np.sqrt(1.0 - rho) * local


def front_standardized_innovations(
    master_seed: int,
    sample_count: int,
    body_count: int,
    distribution: str,
    front_angle: float,
) -> np.ndarray:
    """Common-mode gust front: one scalar innovation per tick along ``front_angle``.

    The scalar stream is the first component of the ``[master, 11, 0]`` stream (with the
    common radial factor for t3), scaled by sqrt(2) so that the force vector's total
    variance equals that of the isotropic classes (variance matching).  ``front_angle``
    is the world direction in which the gust pushes.
    """
    scalar = _draw_standardized(weather_generator(master_seed, 0), sample_count, distribution)[:, 0]
    direction = np.array([np.cos(front_angle), np.sin(front_angle)])
    innovations = np.sqrt(2.0) * scalar[:, np.newaxis] * direction[np.newaxis, :]
    return np.repeat(innovations[:, np.newaxis, :], body_count, axis=1)


def stationary_weather_forces(
    master_seed: int,
    duration: float,
    *,
    vessel_count: int = constants.VESSEL_COUNT,
    distribution: str = "gaussian",
    direction: str = "local",
    scale: float = 1.0,
    rho: float | None = None,
    preroll: float = DEFAULT_PREROLL_SECONDS,
    front_angle: float | None = None,
) -> np.ndarray:
    """Weather force samples on the 10 ms grid, stationary from the first kept sample.

    The AR(1) recursion starts from zero ``preroll`` seconds before the kept window
    (5 memory times by default, leaving a variance deficit of exp(-10)).  ``scale``
    multiplies the pinned per-body stationary standard deviations of IV.12.
    Returns an array of shape (samples, N + 1, 2).
    """
    body_count = vessel_count + 1
    preroll_samples = int(round(preroll / constants.WEATHER_PERIOD))
    kept = int(np.ceil(duration / constants.WEATHER_PERIOD)) + 2
    total = preroll_samples + kept
    if direction == "mixed":
        if rho is None:
            raise ValueError("mixed weather requires rho")
        standardized = mixed_standardized_innovations(master_seed, total, body_count, distribution, rho)
    elif direction == "front":
        if front_angle is None:
            raise ValueError("front weather requires front_angle")
        standardized = front_standardized_innovations(master_seed, total, body_count, distribution, front_angle)
    else:
        standardized = standardized_innovations(master_seed, total, body_count, distribution, direction)
    stationary_std = scale * np.array(
        [constants.LOAD_WEATHER_STD] + [constants.VESSEL_WEATHER_STD] * vessel_count
    )
    innovation_scale = stationary_std * np.sqrt(1.0 - ar1_coefficient() ** 2)
    states = apply_ar1(standardized * innovation_scale[np.newaxis, :, np.newaxis])
    return states[preroll_samples:]


def hill_tail_index(samples: np.ndarray, upper_order_count: int) -> float:
    """Estimate a two-sided regularly varying tail index with Hill's estimator."""
    magnitudes = np.sort(np.abs(np.asarray(samples, dtype=float)))
    if not 1 <= upper_order_count < magnitudes.size:
        raise ValueError("upper_order_count must be between 1 and sample_count - 1")
    threshold = magnitudes[-upper_order_count - 1]
    return float(
        upper_order_count
        / np.sum(np.log(magnitudes[-upper_order_count:] / threshold))
    )


def estimate_ar1_coefficient(states: np.ndarray) -> float:
    previous = np.asarray(states[:-1], dtype=float).reshape(-1)
    following = np.asarray(states[1:], dtype=float).reshape(-1)
    return float(previous @ following / (previous @ previous))


def ar1_95_interval(sample_count: int, phi: float | None = None) -> tuple[float, float]:
    """Return the asymptotic 95% interval centered on the specified AR coefficient."""
    if phi is None:
        phi = ar1_coefficient()
    half_width = 1.96 * np.sqrt((1.0 - phi**2) / sample_count)
    return float(phi - half_width), float(phi + half_width)

# --------------------------------------------------------------------------------------
# v2 additions (plan v2, IV.5).  Everything below is new and opt-in; nothing above this
# line changes, so every v1 record replays unchanged.
# --------------------------------------------------------------------------------------

GUST_STREAM = 19
GUST_RATE = 0.1  # arrivals per second per body (local) or per fleet (fleet-wide)
GUST_TAIL_INDEX = 3.0  # Pareto: P(A > a) = (a / a0)^-3
GUST_SCALE_FACTOR = 1.5  # a0_j = 1.5 x the body's stationary std
GUST_DURATION_MEDIAN = 2.0  # t_g lognormal, median 2 s
GUST_DURATION_LOG_STD = 0.3  # ... and log-std 0.3
# int_0^1 E(x)^2 dx for the raised-cosine pulse E(x) = (1 - cos 2 pi x)/2.
RAISED_COSINE_ENERGY = 3.0 / 8.0


def pinned_stationary_std(vessel_count: int = constants.VESSEL_COUNT, scale: float = 1.0) -> np.ndarray:
    """Per-body stationary std per component, load first (IV.12), times ``scale``."""
    return scale * np.array([constants.LOAD_WEATHER_STD] + [constants.VESSEL_WEATHER_STD] * vessel_count)


def front_arrival_delays(
    positions: np.ndarray,
    front_angle: float,
    speed: float,
    period: float = constants.WEATHER_PERIOD,
) -> np.ndarray:
    """Integer per-body sample delays of a frozen front (Taylor), IV.5 front(c).

    ``d_b = (p_b . d - min_b' p_b' . d) / c`` with ``d`` the unit vector of ``front_angle``
    (the world direction in which the front pushes and travels), rounded to the nearest
    10 ms sample.  ``speed = inf`` is exact common forcing (all delays zero).
    """
    positions = np.asarray(positions, dtype=float)
    if not speed > 0.0:
        raise ValueError("front speed must be positive")
    if np.isinf(speed):
        return np.zeros(positions.shape[0], dtype=np.int64)
    direction = np.array([np.cos(front_angle), np.sin(front_angle)])
    along = positions @ direction
    lag = (along - along.min()) / speed
    return np.rint(lag / period).astype(np.int64)


def lagged_front_weather_forces(
    master_seed: int,
    duration: float,
    *,
    positions: np.ndarray,
    front_angle: float,
    speed: float,
    vessel_count: int = constants.VESSEL_COUNT,
    distribution: str = "gaussian",
    scale: float = 1.0,
    preroll: float = DEFAULT_PREROLL_SECONDS,
) -> np.ndarray:
    """front(c): one scalar AR(1) stream along ``front_angle`` delivered with arrival lags.

    The scalar stream is v1's front stream (first component of ``[master, 11, 0]``, times
    sqrt(2) for variance matching).  Body b receives the innovation sequence delayed by
    ``d_b`` samples (leading zeros), so its AR(1) state is the shared state delayed by
    exactly ``d_b`` samples.  The preroll is ``preroll`` plus the maximum delay, so every
    body's recursion has run at least ``preroll`` seconds before the kept window.  With
    ``speed = inf`` the result is bit-identical to
    ``stationary_weather_forces(direction="front", front_angle=...)``.
    Returns (samples, N + 1, 2).
    """
    body_count = vessel_count + 1
    delays = front_arrival_delays(positions, front_angle, speed)
    if delays.shape[0] != body_count:
        raise ValueError("positions must hold one row per body, load first")
    max_delay = int(delays.max())
    preroll_samples = int(round(preroll / constants.WEATHER_PERIOD)) + max_delay
    kept = int(np.ceil(duration / constants.WEATHER_PERIOD)) + 2
    total = preroll_samples + kept
    scalar = _draw_standardized(weather_generator(master_seed, 0), total, distribution)[:, 0]
    direction = np.array([np.cos(front_angle), np.sin(front_angle)])
    standardized = np.zeros((total, body_count, 2))
    for body in range(body_count):
        delay = int(delays[body])
        shifted = np.zeros(total)
        shifted[delay:] = scalar[: total - delay]
        standardized[:, body, :] = np.sqrt(2.0) * shifted[:, np.newaxis] * direction[np.newaxis, :]
    stationary_std = pinned_stationary_std(vessel_count, scale)
    innovation_scale = stationary_std * np.sqrt(1.0 - ar1_coefficient() ** 2)
    states = apply_ar1(standardized * innovation_scale[np.newaxis, :, np.newaxis])
    return states[preroll_samples:]


def raised_cosine_pulse(fraction) -> np.ndarray:
    """Gust envelope E(x) = (1 - cos 2 pi x)/2 on [0, 1], zero outside; peak 1 at x = 1/2.

    This is the mission's squall envelope (raised-cosine rise and fall) with zero
    plateau and total duration 1.
    """
    x = np.asarray(fraction, dtype=float)
    inside = (x > 0.0) & (x < 1.0)
    return np.where(inside, 0.5 * (1.0 - np.cos(2.0 * np.pi * x)), 0.0)


def gust_second_moments(
    rate: float = GUST_RATE,
    tail_index: float = GUST_TAIL_INDEX,
    duration_median: float = GUST_DURATION_MEDIAN,
    duration_log_std: float = GUST_DURATION_LOG_STD,
) -> dict[str, float]:
    """Compound-Poisson moments of the gust term per unit a0^2 (Campbell's theorem).

    For shot noise sum_n A_n s_n E((t - t_n)/t_g,n) with Poisson rate lambda, the
    stationary covariance is lambda E[A^2] E[s s^T] E[t_g] int E^2.  With Pareto(alpha)
    amplitudes of scale a0, E[A^2] = alpha a0^2/(alpha - 2) (finite for alpha = 3), and
    E[t_g] = median exp(s^2/2) for the lognormal.  Returns the variance per component for
    an isotropic direction (E[s_x^2] = 1/2) and along the front for a fixed direction
    (E[s_x^2] = 1), plus the mean along a fixed direction (lambda E[A] E[t_g] int E).
    """
    if tail_index <= 2.0:
        raise ValueError("the gust variance is finite only for tail index > 2")
    second = tail_index / (tail_index - 2.0)
    first = tail_index / (tail_index - 1.0)
    mean_duration = duration_median * np.exp(0.5 * duration_log_std**2)
    along = rate * second * mean_duration * RAISED_COSINE_ENERGY
    return {
        "isotropic_component_variance": 0.5 * along,
        "fixed_direction_along_variance": along,
        "fixed_direction_mean": rate * first * mean_duration * 0.5,
        "mean_duration": float(mean_duration),
        "amplitude_second_moment": second,
    }


def gust_background_factor(scale_factor: float = GUST_SCALE_FACTOR, **moments) -> float:
    """Variance preservation: background std multiplier so the total per-component
    variance of background + gusts equals the pinned stationary variance.

    With a0_j = scale_factor x sigma_j the isotropic gust variance per component is
    g sigma_j^2 with g = scale_factor^2 x (isotropic variance per unit a0^2), the same
    fraction for every body, so the background std becomes sqrt(1 - g) sigma_j.  For the
    fixed-direction (fleet-wide) variants the same factor preserves the total (trace)
    variance.
    """
    fraction = scale_factor**2 * gust_second_moments(**moments)["isotropic_component_variance"]
    if fraction >= 1.0:
        raise ValueError("gust variance exceeds the pinned total variance")
    return float(np.sqrt(1.0 - fraction))


def gust_generator(master_seed: int, body: int) -> np.random.Generator:
    """The plan's own gust stream ``[master, 19, body]``."""
    return np.random.default_rng(np.random.SeedSequence([master_seed, GUST_STREAM, body]))


@dataclass(frozen=True)
class GustMarks:
    """Gust marks of one stream: arrival times [s], standardized amplitudes A/a0 (>= 1),
    direction angles [rad] and durations t_g [s]."""

    times: np.ndarray
    amplitudes: np.ndarray
    angles: np.ndarray
    durations: np.ndarray


def draw_gust_marks(
    master_seed: int,
    body: int,
    start: float,
    stop: float,
    rate: float = GUST_RATE,
    tail_index: float = GUST_TAIL_INDEX,
    duration_median: float = GUST_DURATION_MEDIAN,
    duration_log_std: float = GUST_DURATION_LOG_STD,
) -> GustMarks:
    """Poisson arrivals on [start, stop) from stream ``[master, 19, body]``.

    Draw order (fixed): count ~ Poisson(rate (stop - start)); arrival times uniform,
    sorted; amplitudes A/a0 = (1 - U)^(-1/alpha); angles uniform on [0, 2 pi);
    durations median exp(log_std Z).
    """
    generator = gust_generator(master_seed, body)
    count = int(generator.poisson(rate * (stop - start)))
    times = np.sort(start + (stop - start) * generator.random(count))
    amplitudes = (1.0 - generator.random(count)) ** (-1.0 / tail_index)
    angles = 2.0 * np.pi * generator.random(count)
    durations = duration_median * np.exp(duration_log_std * generator.standard_normal(count))
    return GustMarks(times, amplitudes, angles, durations)


def _add_pulses(target: np.ndarray, marks: GustMarks, amplitude_scale: float, directions: np.ndarray,
                offset: float, period: float) -> None:
    """Add sum_n a0 A_n s_n E((t_k - t_n)/t_g,n) into ``target`` (samples, 2), t_k = k period + offset."""
    samples = target.shape[0]
    for time, amplitude, direction, duration in zip(marks.times, marks.amplitudes, directions, marks.durations):
        first = max(int(np.floor((time - offset) / period)), 0)
        last = min(int(np.ceil((time + duration - offset) / period)), samples - 1)
        if last < first:
            continue
        index = np.arange(first, last + 1)
        envelope = raised_cosine_pulse((index * period + offset - time) / duration)
        target[first : last + 1] += (amplitude_scale * amplitude) * envelope[:, np.newaxis] * direction[np.newaxis, :]


def gust_forces(
    master_seed: int,
    duration: float,
    *,
    vessel_count: int = constants.VESSEL_COUNT,
    scale: float = 1.0,
    structure: str = "local",
    front_angle: float | None = None,
    delays: np.ndarray | None = None,
    preroll: float = DEFAULT_PREROLL_SECONDS,
    rate: float = GUST_RATE,
    scale_factor: float = GUST_SCALE_FACTOR,
) -> tuple[np.ndarray, list[GustMarks]]:
    """The gust term of the gust class on the kept 10 ms grid, and its marks.

    ``structure``: ``local`` - independent arrivals at ``rate`` per body, each from
    ``[master, 19, body]``, direction uniform on the circle; ``common`` - one fleet-wide
    arrival process at ``rate`` from ``[master, 19, 0]``, every gust on every body at once
    along ``front_angle``, body amplitude a0_j A; ``front`` - as ``common`` but body b
    receives each gust ``delays[b]`` samples late.  a0_j = scale_factor x scale x pinned
    std.  Arrivals are drawn on [-preroll, kept end) so the term is stationary from the
    first kept sample (the kept grid matches ``stationary_weather_forces``).
    Returns the (samples, N + 1, 2) force array and the marks (one entry per body for
    ``local``, a single entry for the fleet-wide structures); times are on the kept clock.
    """
    body_count = vessel_count + 1
    period = constants.WEATHER_PERIOD
    kept = int(np.ceil(duration / period)) + 2
    a0 = scale_factor * pinned_stationary_std(vessel_count, scale)
    forces = np.zeros((kept, body_count, 2))
    stop = kept * period
    if structure == "local":
        marks = []
        for body in range(body_count):
            body_marks = draw_gust_marks(master_seed, body, -preroll, stop, rate=rate)
            directions = np.stack([np.cos(body_marks.angles), np.sin(body_marks.angles)], axis=1)
            _add_pulses(forces[:, body, :], body_marks, float(a0[body]), directions, 0.0, period)
            marks.append(body_marks)
        return forces, marks
    if structure not in ("common", "front"):
        raise ValueError(f"unknown gust structure: {structure}")
    if front_angle is None:
        raise ValueError("fleet-wide gusts require front_angle")
    if structure == "front":
        if delays is None:
            raise ValueError("front gusts require the per-body sample delays")
        delays = np.asarray(delays, dtype=np.int64)
    else:
        delays = np.zeros(body_count, dtype=np.int64)
    fleet_marks = draw_gust_marks(master_seed, 0, -preroll, stop, rate=rate)
    unit = np.array([np.cos(front_angle), np.sin(front_angle)])
    directions = np.tile(unit, (fleet_marks.times.size, 1))
    for body in range(body_count):
        _add_pulses(forces[:, body, :], fleet_marks, float(a0[body]), directions, -float(delays[body]) * period, period)
    return forces, [fleet_marks]


def gust_class_weather(
    master_seed: int,
    duration: float,
    *,
    vessel_count: int = constants.VESSEL_COUNT,
    scale: float = 1.0,
    structure: str = "local",
    front_angle: float | None = None,
    positions: np.ndarray | None = None,
    front_speed: float | None = None,
    variance_preserving: bool = True,
    preroll: float = DEFAULT_PREROLL_SECONDS,
    return_components: bool = False,
):
    """Gust class W_j = G_j + sum_n A_n s_n E((t - t_n)/t_g,n) on the 10 ms grid (IV.5).

    The background G is the Gaussian AR(1) of the matching direction class drawn from the
    same ``[master, 11, .]`` streams as the Gaussian cell (local for ``local``, v1's exact
    common front for ``common``, front(c) for ``front``), with its std multiplied by
    ``gust_background_factor()`` when ``variance_preserving`` (the declared rule).
    Returns forces (samples, N + 1, 2), or (forces, background, gusts, marks) when
    ``return_components``.
    """
    factor = gust_background_factor() if variance_preserving else 1.0
    delays = None
    if structure == "local":
        background = stationary_weather_forces(
            master_seed, duration, vessel_count=vessel_count, direction="local", scale=scale * factor, preroll=preroll
        )
    elif structure == "common":
        background = stationary_weather_forces(
            master_seed, duration, vessel_count=vessel_count, direction="front", scale=scale * factor,
            preroll=preroll, front_angle=front_angle,
        )
    elif structure == "front":
        if positions is None or front_speed is None:
            raise ValueError("front gusts require body positions and the front speed")
        delays = front_arrival_delays(positions, front_angle, front_speed)
        background = lagged_front_weather_forces(
            master_seed, duration, positions=positions, front_angle=front_angle, speed=front_speed,
            vessel_count=vessel_count, scale=scale * factor, preroll=preroll,
        )
    else:
        raise ValueError(f"unknown gust structure: {structure}")
    gusts, marks = gust_forces(
        master_seed, duration, vessel_count=vessel_count, scale=scale, structure=structure,
        front_angle=front_angle, delays=delays, preroll=preroll,
    )
    forces = background + gusts
    if return_components:
        return forces, background, gusts, marks
    return forces
