"""Part II excursion, impact, threshold-law, and hazard predictors (NumPy/SciPy only).

Propositions 1-4, Corollary 5, Theorems 6-7, Corollaries 7.1-7.2, Propositions 11-12, and
Appendix A.1, A.2, A.4 of the plan, plus the constant-acceleration hazard model of IV.9.
Array arguments broadcast; scalar inputs return ``float``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, optimize, special

from tether.physics import constants

FloatArray = NDArray[np.float64]

EFFECTIVE_MASS = (
    constants.VESSEL_MASS * constants.LOAD_MASS / (constants.VESSEL_MASS + constants.LOAD_MASS)
)
NOMINAL_TOW_SPEED = (
    constants.VESSEL_COUNT
    * constants.NOMINAL_THRUST
    / (constants.LOAD_LINEAR_DRAG + constants.VESSEL_COUNT * constants.VESSEL_LINEAR_DRAG)
)
NOMINAL_PRETENSION = constants.LOAD_LINEAR_DRAG * NOMINAL_TOW_SPEED / constants.VESSEL_COUNT
RESTORING_ACCELERATION = NOMINAL_PRETENSION / EFFECTIVE_MASS
PINNED_IMPACT_FACTOR = 0.880
DEFAULT_RICE_GRID = 2001
RICE_TOLERANCE = 1e-12
_RICE_MAX_PASSES = 80
_RICE_MAX_INTERVALS = 1_000_000
_BOOLE_NODES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_GRID_BLOCK_ELEMENTS = 250_000
_SQRT_TWO_PI = np.sqrt(2.0 * np.pi)


def _result(value: ArrayLike) -> float | FloatArray:
    array = np.asarray(value, dtype=float)
    return float(array) if array.ndim == 0 else array


def damping_ratio(
    k: float = constants.CABLE_STIFFNESS,
    c: float = constants.CABLE_DAMPING,
    meff: float = EFFECTIVE_MASS,
) -> float:
    """Engagement damping ratio zeta = c / (2 sqrt(k meff))."""
    return float(c / (2.0 * np.sqrt(k * meff)))


def impact_factor(zeta: ArrayLike) -> float | FloatArray:
    """Appendix A.1 damped-impact factor f(zeta) = max_tau T(tau) / (sqrt(k meff) V).

    For meff e'' + c e' + k e = 0 from e = 0, e' = V the normalised tension is
    exp(-zeta tau) sin(w tau + phi) / w with w = sqrt(1 - zeta^2), phi = 2 arcsin(zeta). Its
    first stationary point gives f = exp(-zeta (pi/2 - 3 arcsin zeta) / w) for zeta < 1/2.
    From zeta = 1/2 on, overdamped included, the tension only falls from the initial damper
    jump c V, so f = 2 zeta.
    """
    ratio = np.asarray(zeta, dtype=float)
    if np.any(ratio < 0.0) or not np.all(np.isfinite(ratio)):
        raise ValueError("damping ratio must be finite and nonnegative")
    below = np.minimum(ratio, 0.5)
    interior = np.exp(-below * (0.5 * np.pi - 3.0 * np.arcsin(below)) / np.sqrt(1.0 - below**2))
    return _result(np.where(ratio < 0.5, interior, 2.0 * ratio))


def impedance(
    k: float = constants.CABLE_STIFFNESS,
    c: float = constants.CABLE_DAMPING,
    meff: float = EFFECTIVE_MASS,
) -> float:
    """Engagement impedance Z = f(zeta) sqrt(k meff), peak tension per unit closing speed."""
    return float(impact_factor(damping_ratio(k, c, meff)) * np.sqrt(k * meff))


DAMPING_RATIO = damping_ratio()
IMPACT_FACTOR = float(impact_factor(DAMPING_RATIO))
IMPEDANCE = impedance()


def onset_rate(mu: ArrayLike, sigma_e: ArrayLike, sigma_edot: ArrayLike) -> float | FloatArray:
    """Prop 1 / A.2: Rice rate of downcrossings of e = 0 by stationary Gaussian e of mean mu."""
    mean = np.asarray(mu, dtype=float)
    return _result(
        np.asarray(sigma_edot, dtype=float)
        / (2.0 * np.pi * np.asarray(sigma_e, dtype=float))
        * np.exp(-0.5 * mean**2 / np.asarray(sigma_e, dtype=float) ** 2)
    )


def rayleigh_pdf(u: ArrayLike, sigma: ArrayLike) -> float | FloatArray:
    """Prop 1 entry-speed density (u / sigma^2) exp(-u^2 / 2 sigma^2), zero for u <= 0."""
    speed = np.asarray(u, dtype=float)
    scale = np.asarray(sigma, dtype=float)
    density = speed / scale**2 * np.exp(-0.5 * (speed / scale) ** 2)
    return _result(np.where(speed > 0.0, density, 0.0))


def rayleigh_cdf(u: ArrayLike, sigma: ArrayLike) -> float | FloatArray:
    """Prop 1 entry-speed distribution 1 - exp(-u^2 / 2 sigma^2), zero for u <= 0."""
    speed = np.asarray(u, dtype=float)
    scale = np.asarray(sigma, dtype=float)
    return _result(np.where(speed > 0.0, -np.expm1(-0.5 * (speed / scale) ** 2), 0.0))


def ballistic_return(u: ArrayLike, a: ArrayLike) -> dict[str, float | FloatArray]:
    """Prop 2: from onset speed u under constant net acceleration a > 0 the excursion returns
    at time 2u/a with speed u, after reaching depth u^2 / 2a."""
    speed = np.asarray(u, dtype=float)
    acceleration = np.asarray(a, dtype=float)
    if np.any(acceleration <= 0.0):
        raise ValueError("ballistic return requires a positive net acceleration")
    if np.any(speed < 0.0):
        raise ValueError("onset speed must be nonnegative")
    return {
        "return_time": _result(2.0 * speed / acceleration),
        "depth": _result(speed**2 / (2.0 * acceleration)),
        "return_speed": _result(speed * np.ones_like(acceleration)),
    }


def ballistic_snap_rate(
    Tb: ArrayLike,
    mu: float,
    sigma_e: float,
    sigma_edot: float,
    Z: float = IMPEDANCE,
) -> float | FloatArray:
    """Cor 2.1: snap rate of ballistic excursions, nu_on exp(-Tb^2 / 2 Z^2 sigma_edot^2)."""
    level = np.asarray(Tb, dtype=float) / Z
    return _result(onset_rate(mu, sigma_e, sigma_edot) * np.exp(-0.5 * (level / sigma_edot) ** 2))


def return_speed(u: ArrayLike, a_bar: ArrayLike, depth: ArrayLike) -> float | FloatArray:
    """Prop 3 work-energy identity V = sqrt(u^2 + 2 a_bar depth); NaN where the leg cannot close.

    Exact for a leg entered at speed u that covers ``depth`` against the depth-averaged
    acceleration a_bar. Measured from the vertex (maximum depth, e' = 0) the entry speed is
    0 and a_bar is the return-leg mean, so V = sqrt(2 a_bar depth). Pairing the onset speed
    with the maximum depth, as Prop 3 is printed, counts the entry energy twice: for the
    ballistic excursion of Prop 2 it gives sqrt(2) u instead of u.
    """
    radicand = np.asarray(u, dtype=float) ** 2 + 2.0 * np.asarray(a_bar, dtype=float) * np.asarray(
        depth, dtype=float
    )
    return _result(np.where(radicand >= 0.0, np.sqrt(np.maximum(radicand, 0.0)), np.nan))


def snap_tension_from_depth(
    depth: ArrayLike,
    T0: ArrayLike,
    k: float = constants.CABLE_STIFFNESS,
    f: float = IMPACT_FACTOR,
    meff: float = EFFECTIVE_MASS,
    u: ArrayLike = 0.0,
) -> float | FloatArray:
    """Prop 3 with a_bar = a0 = T0/meff and Z = f sqrt(k meff): f sqrt(2 k T0 depth + k meff u^2).

    With u = 0 (depth measured to the vertex) the reduced mass cancels and the law is exact
    for an undriven return leg; see ``return_speed`` for the meaning of a nonzero u.
    """
    radicand = 2.0 * k * np.asarray(T0, dtype=float) * np.asarray(depth, dtype=float) + k * meff * (
        np.asarray(u, dtype=float) ** 2
    )
    return _result(f * np.sqrt(radicand))


def deep_excursion_depth(
    delta_w: ArrayLike,
    t_g: ArrayLike,
    u: ArrayLike,
    meff: float = EFFECTIVE_MASS,
    T0: float = NOMINAL_PRETENSION,
) -> float | FloatArray:
    """Prop 4 depth, by branch.

    delta_w = W_rel - T0 > 0: delta_w t_g^2 / (2 meff) + u t_g, the depth accumulated while the
    excess acts, i.e. the depth at gust shutoff. The bodies are still closing then, so the
    maximum depth is larger by (u + delta_w t_g / meff)^2 / (2 a0); see ``two_phase_excursion``.
    delta_w <= 0: the ballistic depth u^2 / (2 a0), a0 = T0 / meff, exact when the gust is absent
    during the excursion; a sub-threshold gust that persists deepens it to
    u^2 meff / (2 (T0 - W_rel)), which is still independent of t_g.
    """
    excess = np.asarray(delta_w, dtype=float)
    duration = np.asarray(t_g, dtype=float)
    speed = np.asarray(u, dtype=float)
    driven = excess * duration**2 / (2.0 * meff) + speed * duration
    ballistic = speed**2 / (2.0 * T0 / meff)
    return _result(np.where(excess > 0.0, driven, ballistic))


@dataclass(frozen=True)
class ExcursionOutcome:
    depth_at_shutoff: float
    max_depth: float
    turning_time: float
    return_time: float
    return_speed: float
    returned_during_gust: bool


def two_phase_excursion(
    delta_w: float,
    t_g: float,
    u: float = 0.0,
    T0: float = NOMINAL_PRETENSION,
    meff: float = EFFECTIVE_MASS,
) -> ExcursionOutcome:
    """Exact slack excursion under a square gust: e'' = -delta_w / meff for t < t_g, then a0.

    Onset at t = 0 with e = 0, e' = -u, eta = 0 (H4). For delta_w > 0 and u = 0 the maximum
    depth is (W_rel / T0) times the shutoff depth of Prop 4 and the return speed is
    t_g sqrt(W_rel delta_w) / meff.
    """
    restoring = T0 / meff
    if restoring <= 0.0 or t_g < 0.0 or u < 0.0:
        raise ValueError("requires T0 > 0, t_g >= 0, and u >= 0")
    gust = -delta_w / meff
    if gust > 0.0 and 2.0 * u <= gust * t_g:
        return ExcursionOutcome(
            depth_at_shutoff=0.0,
            max_depth=u * u / (2.0 * gust),
            turning_time=u / gust,
            return_time=2.0 * u / gust,
            return_speed=float(u),
            returned_during_gust=True,
        )
    elongation = -u * t_g + 0.5 * gust * t_g**2
    rate = -u + gust * t_g
    if gust > 0.0 and u <= gust * t_g:
        max_depth, turning_time = u * u / (2.0 * gust), u / gust
    else:
        closing = max(-rate, 0.0)
        max_depth = -elongation + closing**2 / (2.0 * restoring)
        turning_time = t_g + closing / restoring
    speed = float(np.sqrt(max(rate**2 - 2.0 * restoring * elongation, 0.0)))
    return ExcursionOutcome(
        depth_at_shutoff=max(-elongation, 0.0),
        max_depth=float(max_depth),
        turning_time=float(turning_time),
        return_time=float(t_g + (speed - rate) / restoring),
        return_speed=speed,
        returned_during_gust=False,
    )


def closed_form_snap(
    w_rel: ArrayLike,
    T0: ArrayLike,
    t_g: ArrayLike,
    k: float = constants.CABLE_STIFFNESS,
    f: float = IMPACT_FACTOR,
    meff: float = EFFECTIVE_MASS,
) -> float | FloatArray:
    """Cor 5: f t_g sqrt(k T0 (W_rel - T0) / meff) for W_rel > T0, else 0.

    Prop 3 (u = 0) applied to the shutoff depth of Prop 4; ``two_phase_snap`` is the same
    square gust with the post-gust closing included.
    """
    load = np.asarray(w_rel, dtype=float)
    tension = np.asarray(T0, dtype=float)
    excess = np.maximum(load - tension, 0.0)
    return _result(f * np.asarray(t_g, dtype=float) * np.sqrt(k * tension * excess / meff))


def two_phase_snap(
    w_rel: ArrayLike,
    T0: ArrayLike,
    t_g: ArrayLike,
    k: float = constants.CABLE_STIFFNESS,
    f: float = IMPACT_FACTOR,
    meff: float = EFFECTIVE_MASS,
) -> float | FloatArray:
    """Snap of the exact square-gust excursion (u = 0): f t_g sqrt(k W_rel (W_rel - T0) / meff).

    Equals ``closed_form_snap`` times sqrt(W_rel / T0): the gust leaves the bodies closing at
    delta_w t_g / meff, and that speed deepens the excursion before a0 turns it.
    """
    load = np.asarray(w_rel, dtype=float)
    excess = np.maximum(load - np.asarray(T0, dtype=float), 0.0)
    return _result(f * np.asarray(t_g, dtype=float) * np.sqrt(k * load * excess / meff))


def two_phase_severance_load(
    Tb: ArrayLike,
    T0: ArrayLike,
    t_g: ArrayLike,
    k: float = constants.CABLE_STIFFNESS,
    f: float = IMPACT_FACTOR,
    meff: float = EFFECTIVE_MASS,
) -> float | FloatArray:
    """W_rel at which ``two_phase_snap`` reaches Tb: (T0 + sqrt(T0^2 + 4 kappa T0 Tb^2)) / 2.

    For kappa Tb^2 << T0 it expands to the II.6 level T0 + kappa Tb^2; for kappa Tb^2 >> T0 it
    grows as Tb sqrt(kappa T0), so a tail of index alpha in W_rel gives the snap rate a local
    index alpha (1 - (1 + 4 kappa Tb^2 / T0)^(-1/2)), below alpha rather than 2 alpha.
    """
    tension = np.asarray(T0, dtype=float)
    duration = np.asarray(t_g, dtype=float)
    momentum = meff * np.asarray(Tb, dtype=float) ** 2 / (f**2 * k * duration**2)
    return _result(0.5 * (tension + np.sqrt(tension**2 + 4.0 * momentum)))


def pretension_optimum(w_rel: ArrayLike) -> float | FloatArray:
    """Cor 5: the pretension maximising the snap severity at fixed weather, W_rel / 2."""
    return _result(0.5 * np.asarray(w_rel, dtype=float))


def kappa(
    T0: ArrayLike,
    t_g: ArrayLike,
    k: float = constants.CABLE_STIFFNESS,
    f: float = IMPACT_FACTOR,
    meff: float = EFFECTIVE_MASS,
) -> float | FloatArray:
    """Severance coefficient meff / (f^2 k T0 t_g^2): snap iff W_rel > T0 + kappa Tb^2."""
    return _result(
        meff / (f**2 * k * np.asarray(T0, dtype=float) * np.asarray(t_g, dtype=float) ** 2)
    )


def critical_depth(
    Tb: ArrayLike,
    T0: ArrayLike,
    k: float = constants.CABLE_STIFFNESS,
    f: float = IMPACT_FACTOR,
) -> float | FloatArray:
    """Appendix A.4 critical depth Tb^2 / (2 f^2 k T0)."""
    level = np.asarray(Tb, dtype=float)
    return _result(level**2 / (2.0 * f**2 * k * np.asarray(T0, dtype=float)))


def critical_closing_speed(Tb: ArrayLike, Z: float = IMPEDANCE) -> float | FloatArray:
    """II.8 critical closing speed v_b = Tb / Z under impact linearity (H3)."""
    return _result(np.asarray(Tb, dtype=float) / Z)


def saturation_snap(
    T0: ArrayLike,
    depth_max: ArrayLike,
    k: float = constants.CABLE_STIFFNESS,
    f: float = IMPACT_FACTOR,
) -> float | FloatArray:
    """Cor 7.2: largest snap f sqrt(2 k T0 depth_max) under a geometric depth bound."""
    return _result(
        f * np.sqrt(2.0 * k * np.asarray(T0, dtype=float) * np.asarray(depth_max, dtype=float))
    )


def _snap_standard_score(
    Tb: ArrayLike, T0: float, kappa: float, mu_w: float, sigma_w: float
) -> FloatArray:
    return (np.asarray(T0, dtype=float) + kappa * np.asarray(Tb, dtype=float) ** 2 - mu_w) / sigma_w


def gaussian_snap_rate(
    Tb: ArrayLike, nu_g: float, T0: float, kappa: float, mu_w: float, sigma_w: float
) -> float | FloatArray:
    """Thm 6 snap rate nu_g S((T0 + kappa Tb^2 - mu_W) / sigma_W), S the normal survival."""
    return _result(nu_g * special.ndtr(-_snap_standard_score(Tb, T0, kappa, mu_w, sigma_w)))


def gaussian_snap_log_rate(
    Tb: ArrayLike, nu_g: float, T0: float, kappa: float, mu_w: float, sigma_w: float
) -> float | FloatArray:
    """Natural log of ``gaussian_snap_rate``, finite far into the tail."""
    score = _snap_standard_score(Tb, T0, kappa, mu_w, sigma_w)
    return _result(np.log(nu_g) + special.log_ndtr(-score))


def gaussian_snap_log_slope(kappa: float, sigma_w: float) -> float:
    """Thm 6: coefficient -kappa^2 / (2 sigma_W^2) of Tb^4 in the asymptotic log snap rate.

    The expansion of the log rate also carries -(T0 - mu_W) kappa Tb^2 / sigma_W^2, which
    dominates until kappa Tb^2 >> 2 (T0 - mu_W).
    """
    return float(-(kappa**2) / (2.0 * sigma_w**2))


def gaussian_taut_rate(
    Tb: ArrayLike, k_mu: float, sigma_q: float, sigma_qdot: float
) -> float | FloatArray:
    """Thm 6 taut rate (1 / 2 pi)(sigma_qdot / sigma_q) exp(-(Tb - k mu)^2 / 2 sigma_q^2)."""
    offset = np.asarray(Tb, dtype=float) - k_mu
    return _result(sigma_qdot / (2.0 * np.pi * sigma_q) * np.exp(-0.5 * (offset / sigma_q) ** 2))


def gaussian_taut_log_rate(
    Tb: ArrayLike, k_mu: float, sigma_q: float, sigma_qdot: float
) -> float | FloatArray:
    """Natural log of ``gaussian_taut_rate``."""
    offset = np.asarray(Tb, dtype=float) - k_mu
    return _result(np.log(sigma_qdot / (2.0 * np.pi * sigma_q)) - 0.5 * (offset / sigma_q) ** 2)


def gaussian_taut_log_slope(sigma_q: float) -> float:
    """Thm 6: coefficient -1 / (2 sigma_q^2) of Tb^2 in the asymptotic log taut rate."""
    return float(-1.0 / (2.0 * sigma_q**2))


def rv_snap_rate(
    Tb: ArrayLike, nu_g: float, C_w: float, kappa: float, alpha: float, T0: float = 0.0
) -> float | FloatArray:
    """Thm 7 snap rate nu_g C_W (T0 + kappa Tb^2)^(-alpha); T0 = 0 is the printed asymptote."""
    level = T0 + kappa * np.asarray(Tb, dtype=float) ** 2
    return _result(nu_g * C_w * level ** (-alpha))


def rv_snap_local_index(Tb: ArrayLike, T0: float, kappa: float, alpha: float) -> float | FloatArray:
    """-d log Lambda_snap / d log Tb = 2 alpha kappa Tb^2 / (T0 + kappa Tb^2); 2 alpha only
    once kappa Tb^2 >> T0."""
    level = kappa * np.asarray(Tb, dtype=float) ** 2
    return _result(2.0 * alpha * level / (T0 + level))


def rv_taut_rate(Tb: ArrayLike, C_q: float, alpha: float) -> float | FloatArray:
    """Thm 7 taut rate C_q Tb^(-alpha)."""
    return _result(C_q * np.asarray(Tb, dtype=float) ** (-alpha))


def severance_radius(
    Tb: float,
    taut_peak_response: ArrayLike,
    relative_load_response: ArrayLike,
    T0: ArrayLike,
    kappa: ArrayLike,
) -> FloatArray:
    """Thm 7 radius R_sev^(i)(s) = min(Tb / p_i(s), (T0_i + kappa_i Tb^2) / w_i(s)), (n_dir, N).

    A nonpositive response means that channel cannot sever cable i in direction s; its
    radius is infinite. T0 and kappa are scalars or per-cable (N,) arrays.
    """
    if Tb <= 0.0:
        raise ValueError("breaking strength must be positive")
    peak = np.atleast_2d(np.asarray(taut_peak_response, dtype=float))
    load = np.atleast_2d(np.asarray(relative_load_response, dtype=float))
    if peak.shape != load.shape:
        raise ValueError("taut and relative-load responses must share shape (n_dir, N)")
    snap_level = np.asarray(T0, dtype=float) + np.asarray(kappa, dtype=float) * Tb**2
    taut = np.where(peak > 0.0, Tb / np.where(peak > 0.0, peak, 1.0), np.inf)
    snap = np.where(load > 0.0, snap_level / np.where(load > 0.0, load, 1.0), np.inf)
    return np.minimum(taut, snap)


def _direction_weights(directions_weights: ArrayLike, n_dir: int) -> FloatArray:
    weights = np.asarray(directions_weights, dtype=float)
    if weights.shape != (n_dir,) or np.any(weights < 0.0):
        raise ValueError("direction weights must be nonnegative with shape (n_dir,)")
    return weights


def cable_rv_rates(
    Tb: float,
    prob_R_gt_1: float,
    directions_weights: ArrayLike,
    taut_peak_response: ArrayLike,
    relative_load_response: ArrayLike,
    T0: ArrayLike,
    kappa: ArrayLike,
    alpha: float,
) -> FloatArray:
    """Per-cable Thm 7 rates Pr(R > 1) sum_s Gamma_s R_sev^(i)(s)^(-alpha), shape (N,)."""
    radius = severance_radius(Tb, taut_peak_response, relative_load_response, T0, kappa)
    weights = _direction_weights(directions_weights, radius.shape[0])
    return prob_R_gt_1 * np.sum(weights[:, None] * radius ** (-alpha), axis=0)


def fleet_rv_rate(
    Tb: ArrayLike,
    prob_R_gt_1: float,
    directions_weights: ArrayLike,
    taut_peak_response: ArrayLike,
    relative_load_response: ArrayLike,
    T0: ArrayLike,
    kappa: ArrayLike,
    alpha: float,
) -> float | FloatArray:
    """Thm 7 fleet rate Pr(R > 1) sum_s Gamma_s [min_i R_sev^(i)(s)]^(-alpha).

    Assumptions: a shock of radius R in direction s produces taut peak R p_i(s) and
    differential load R w_i(s) (homogeneous response, P3-T4), with static offsets neglected
    in the taut channel; the snap channel severs through W_rel > T0 + kappa Tb^2; the radial
    tail is exactly Pareto, Pr(R > r) = Pr(R > 1) r^(-alpha), which is meaningful for radii
    beyond 1; ``directions_weights`` are the quadrature weights of Gamma (unit total mass for
    a probability measure); any one cable severing counts as a fleet event. The result has
    the units of ``prob_R_gt_1`` (per episode, or per second if it is a rate).
    """
    levels = np.asarray(Tb, dtype=float)
    rates = []
    for level in levels.ravel():
        radius = severance_radius(
            float(level), taut_peak_response, relative_load_response, T0, kappa
        ).min(axis=1)
        weights = _direction_weights(directions_weights, radius.shape[0])
        rates.append(prob_R_gt_1 * float(np.sum(weights * radius ** (-alpha))))
    return _result(np.asarray(rates).reshape(levels.shape))


def _bisect(function: Callable[[float], float], low: float, high: float, low_sign: float) -> float:
    for _ in range(200):
        middle = 0.5 * (low + high)
        value = function(middle)
        if value == 0.0:
            return middle
        if np.isnan(value):
            break
        if np.sign(value) == low_sign:
            low = middle
        else:
            high = middle
        if high - low <= 1e-12 * max(abs(low), abs(high), 1.0):
            break
    return 0.5 * (low + high)


def crossover_threshold(
    rate_snap_fn: Callable[[float], float],
    rate_taut_fn: Callable[[float], float],
    bracket: tuple[float, float],
    *,
    log_rates: bool = False,
    n_scan: int = 257,
) -> float | None:
    """Cor 7.1: smallest Tb in ``bracket`` where log Lambda_snap = log Lambda_taut, else None.

    The log-rate gap is scanned on ``n_scan`` points, NaN points are skipped, and the first
    sign change is refined. Set ``log_rates`` when the functions already return log rates.
    """
    low, high = (float(value) for value in bracket)
    if not low < high:
        raise ValueError("bracket must be increasing")

    def gap(level: float) -> float:
        snap, taut = rate_snap_fn(level), rate_taut_fn(level)
        if not log_rates:
            with np.errstate(divide="ignore", invalid="ignore"):
                snap, taut = np.log(snap), np.log(taut)
        with np.errstate(invalid="ignore"):
            return float(snap - taut)

    grid = np.linspace(low, high, n_scan)
    values = np.array([gap(level) for level in grid])
    valid = np.flatnonzero(~np.isnan(values))
    for first, second in zip(valid[:-1], valid[1:]):
        if values[first] == 0.0:
            return float(grid[first])
        if np.sign(values[first]) != np.sign(values[second]):
            if np.isfinite(values[first]) and np.isfinite(values[second]):
                return float(
                    optimize.brentq(gap, grid[first], grid[second], xtol=1e-10, rtol=1e-12)
                )
            return float(_bisect(gap, grid[first], grid[second], np.sign(values[first])))
    if valid.size and values[valid[-1]] == 0.0:
        return float(grid[valid[-1]])
    return None


def overconfidence_factor(a: ArrayLike, v_b: ArrayLike, sigma: ArrayLike) -> float | FloatArray:
    """Prop 12: a posterior variance sigma^2 / a understates Pr(e' > v_b) by
    exp((a - 1) v_b^2 / 2 sigma^2), to leading order, mean error ignored."""
    return _result(
        np.exp(
            (np.asarray(a, dtype=float) - 1.0)
            * np.asarray(v_b, dtype=float) ** 2
            / (2.0 * np.asarray(sigma, dtype=float) ** 2)
        )
    )


def overconfidence_factor_exact(
    a: ArrayLike, v_b: ArrayLike, sigma: ArrayLike, mean: ArrayLike = 0.0
) -> float | FloatArray:
    """Exact Gaussian ratio S((v_b - m) / sigma) / S(sqrt(a) (v_b - m) / sigma) behind Prop 12;
    it tends to sqrt(a) times the leading-order factor as v_b / sigma grows."""
    score = (np.asarray(v_b, dtype=float) - np.asarray(mean, dtype=float)) / np.asarray(
        sigma, dtype=float
    )
    inflated = np.sqrt(np.asarray(a, dtype=float)) * score
    return _result(np.exp(special.log_ndtr(-score) - special.log_ndtr(-inflated)))


def _dangerous_upcrossing_intensity(
    mean_e: FloatArray,
    mean_edot: FloatArray,
    var_e: FloatArray,
    var_edot: FloatArray,
    cov_e_edot: FloatArray,
    level: float,
) -> FloatArray:
    positive = var_e > 0.0
    if np.any(~positive & (mean_e == 0.0)):
        raise ValueError("e(t) is degenerate at the crossing level")
    safe_var = np.where(positive, var_e, 1.0)
    density = np.where(
        positive, np.exp(-0.5 * mean_e**2 / safe_var) / (_SQRT_TWO_PI * np.sqrt(safe_var)), 0.0
    )
    conditional_mean = mean_edot - cov_e_edot * mean_e / safe_var
    conditional_sd = np.sqrt(np.maximum(var_edot - cov_e_edot**2 / safe_var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        score = (level - conditional_mean) / conditional_sd
        smooth = conditional_mean * special.ndtr(-score) + conditional_sd * np.exp(
            -0.5 * score**2
        ) / _SQRT_TWO_PI
    sharp = np.where(conditional_mean > level, conditional_mean, 0.0)
    return density * np.where(conditional_sd > 0.0, smooth, sharp)


def rice_hazard_bound(
    times: ArrayLike,
    mean_e: ArrayLike,
    mean_edot: ArrayLike,
    var_e: ArrayLike,
    var_edot: ArrayLike,
    cov_e_edot: ArrayLike,
    v_b: float,
) -> float:
    """Prop 11 h_Rice: expected number of upcrossings of e = 0 with e' > v_b over ``times``.

    Integrates E[e' 1{e' > max(v_b, 0)} | e(t) = 0] p_e(t)(0) for a Gaussian (e(t), e'(t))
    with the given pointwise moments (Simpson's rule on the supplied grid, which must resolve
    the crossing-time density). Since Pr(N >= 1) <= E[N] it bounds the hazard from above; it
    is not clipped at 1.
    """
    grid = np.asarray(times, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("times must be a strictly increasing 1-D grid")
    moments = [
        np.broadcast_to(np.asarray(value, dtype=float), grid.shape)
        for value in (mean_e, mean_edot, var_e, var_edot, cov_e_edot)
    ]
    if np.any(moments[2] < 0.0) or np.any(moments[3] < 0.0):
        raise ValueError("variances must be nonnegative")
    integrand = _dangerous_upcrossing_intensity(*moments, max(float(v_b), 0.0))
    return float(integrate.simpson(integrand, x=grid))


def _mean3(mean: ArrayLike) -> FloatArray:
    vector = np.asarray(mean, dtype=float)
    if vector.shape != (3,):
        raise ValueError("mean must have shape (3,) for (e, e_dot, a)")
    return vector


def _cov3(cov: ArrayLike) -> FloatArray:
    matrix = np.asarray(cov, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("covariance must have shape (3, 3) for (e, e_dot, a)")
    scale = max(float(np.max(np.abs(matrix))), 1e-300)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12 * scale):
        raise ValueError("covariance must be symmetric")
    if np.min(np.linalg.eigvalsh(matrix)) < -1e-12 * scale:
        raise ValueError("covariance must be positive semidefinite")
    return 0.5 * (matrix + matrix.T)


def constant_acceleration_moments(
    mean: ArrayLike, cov: ArrayLike, times: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Means, variances and cross-covariance of (e(t), e'(t)) for e = e0 + v0 t + a t^2 / 2."""
    return _moments(_mean3(mean), _cov3(cov), np.asarray(times, dtype=float))


def _moments(
    vector: FloatArray, matrix: FloatArray, grid: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    position = np.stack([np.ones_like(grid), grid, 0.5 * grid**2], axis=-1)
    rate = np.stack([np.zeros_like(grid), np.ones_like(grid), grid], axis=-1)
    return (
        position @ vector,
        rate @ vector,
        np.einsum("ni,ij,nj->n", position, matrix, position),
        np.einsum("ni,ij,nj->n", rate, matrix, rate),
        np.einsum("ni,ij,nj->n", position, matrix, rate),
    )


def _mean_path_landmarks(vector: FloatArray, horizon: float) -> FloatArray:
    """Zeros and vertex of the mean path e0 + v0 t + a t^2 / 2 inside (0, horizon)."""
    e0, v0, a = (float(value) for value in vector)
    candidates = []
    if a != 0.0:
        candidates.append(-v0 / a)
        discriminant = v0 * v0 - 2.0 * a * e0
        if discriminant >= 0.0:
            q = -(v0 + np.copysign(np.sqrt(discriminant), v0))
            if q != 0.0:
                candidates.extend([q / a, 2.0 * e0 / q])
    elif v0 != 0.0:
        candidates.append(-e0 / v0)
    points = np.asarray(candidates, dtype=float)
    return points[(points > 0.0) & (points < horizon)]


def _constant_acceleration_intensity(
    vector: FloatArray, matrix: FloatArray, times: FloatArray, level: float
) -> FloatArray:
    """Dangerous-upcrossing intensity; an onset exactly at e = 0 contributes nothing at t = 0."""
    moments = _moments(vector, matrix, times)
    pinned = (times == 0.0) & (moments[0] == 0.0) & (moments[2] == 0.0)
    if not np.any(pinned):
        return _dangerous_upcrossing_intensity(*moments, level)
    intensity = np.zeros_like(times)
    free = ~pinned
    intensity[free] = _dangerous_upcrossing_intensity(*(value[free] for value in moments), level)
    return intensity


def _adaptive_simpson(
    function: Callable[[FloatArray], FloatArray], edges: FloatArray, tolerance: float
) -> float:
    """Vectorised adaptive Simpson over the partition ``edges``; accepted panels add Boole."""
    low, high = edges[:-1], edges[1:]
    total = 0.0
    for _ in range(_RICE_MAX_PASSES):
        width = high - low
        values = function((low[:, None] + width[:, None] * _BOOLE_NODES).ravel()).reshape(-1, 5)
        coarse = width / 6.0 * (values[:, 0] + 4.0 * values[:, 2] + values[:, 4])
        fine = width / 12.0 * (values[:, 0] + 4.0 * values[:, 1] + 2.0 * values[:, 2])
        fine += width / 12.0 * (4.0 * values[:, 3] + values[:, 4])
        done = np.abs(fine - coarse) <= 15.0 * tolerance
        total += float(np.sum(fine[done] + (fine[done] - coarse[done]) / 15.0))
        if np.all(done):
            return total
        low, high = low[~done], high[~done]
        middle = 0.5 * (low + high)
        quarter = 0.5 * (low + middle)
        if np.any((quarter <= low) | (quarter >= middle)):
            raise ValueError("Rice integrand narrower than the time resolution; use Monte Carlo")
        low, high = np.concatenate([low, middle]), np.concatenate([middle, high])
        if low.size > _RICE_MAX_INTERVALS:
            break
    raise ValueError("Rice quadrature did not converge; use Monte Carlo")


def rice_dangerous_upcrossings_joint(
    mean: ArrayLike,
    cov: ArrayLike,
    horizon: float,
    v_b: float,
    n_grid: int = DEFAULT_RICE_GRID,
) -> float:
    """Prop 11 for the IV.9 constant-acceleration model with jointly Gaussian (e0, v0, a).

    A quadratic path has at most one upcrossing, so for this model the integral is the
    hazard itself, not merely a bound. The uniform ``n_grid`` partition of [0, horizon],
    seeded with the zeros and vertex of the mean path (where a tight posterior concentrates
    its crossing-time density), is refined adaptively until each panel's Simpson error is
    below ``RICE_TOLERANCE``. A zero covariance gives the deterministic count.
    """
    vector, matrix = _mean3(mean), _cov3(cov)
    if not horizon > 0.0 or int(n_grid) < 2:
        raise ValueError("requires horizon > 0 and n_grid >= 2")
    if not np.any(matrix):
        return float(first_upcrossings_exact(vector[None, :], horizon).dangerous(v_b)[0])
    edges = np.union1d(
        np.linspace(0.0, horizon, int(n_grid)), _mean_path_landmarks(vector, horizon)
    )
    level = max(float(v_b), 0.0)
    return _adaptive_simpson(
        lambda times: _constant_acceleration_intensity(vector, matrix, times, level),
        edges,
        RICE_TOLERANCE,
    )


def rice_dangerous_upcrossings(
    mean_e: float,
    mean_edot: float,
    cov: ArrayLike,
    a_hat: float,
    a_sigma: float,
    horizon: float,
    v_b: float,
    n_grid: int = DEFAULT_RICE_GRID,
) -> float:
    """Prop 11 for the IV.9 model: (e0, v0) ~ N((mean_e, mean_edot), cov) and an independent
    acceleration a ~ N(a_hat, a_sigma^2)."""
    block = np.asarray(cov, dtype=float)
    if block.shape != (2, 2):
        raise ValueError("cov must have shape (2, 2) for (e, e_dot)")
    joint = np.zeros((3, 3))
    joint[:2, :2] = block
    joint[2, 2] = a_sigma**2
    return rice_dangerous_upcrossings_joint(
        np.array([mean_e, mean_edot, a_hat], dtype=float), joint, horizon, v_b, n_grid
    )


@dataclass(frozen=True)
class FirstUpcrossings:
    crossed: NDArray[np.bool_]
    time: FloatArray
    speed: FloatArray

    def dangerous(self, v_b: float) -> NDArray[np.bool_]:
        """Trajectories whose first upcrossing has closing speed above v_b."""
        return self.crossed & (np.where(self.crossed, self.speed, -np.inf) > v_b)


def sample_constant_acceleration(
    mean: ArrayLike, cov: ArrayLike, n_samples: int, rng: np.random.Generator
) -> FloatArray:
    """Draw (n_samples, 3) initial states (e0, v0, a) of the IV.9 prediction model."""
    vector, matrix = _mean3(mean), _cov3(cov)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    return vector + rng.standard_normal((int(n_samples), 3)) @ factor.T


def first_upcrossings_from_paths(
    times: ArrayLike, elongation: ArrayLike, elongation_rate: ArrayLike
) -> FirstUpcrossings:
    """First sample pair with e_k < 0 <= e_k+1 per row; time and speed by linear interpolation."""
    grid = np.asarray(times, dtype=float)
    values = np.atleast_2d(np.asarray(elongation, dtype=float))
    rates = np.atleast_2d(np.asarray(elongation_rate, dtype=float))
    if values.shape != rates.shape or values.shape[1] != grid.size:
        raise ValueError("paths must have shape (n, len(times))")
    crossed, rows, index, fraction = _first_upward_brackets(values)
    return _interpolated_upcrossings(
        grid, crossed, rows, index, fraction, rates[rows, index], rates[rows, index + 1]
    )


def _first_upward_brackets(
    values: FloatArray,
) -> tuple[NDArray[np.bool_], NDArray[np.intp], NDArray[np.intp], FloatArray]:
    negative = values < 0.0
    upward = negative[:, :-1] & ~negative[:, 1:]
    crossed = upward.any(axis=1)
    rows = np.flatnonzero(crossed)
    index = upward[rows].argmax(axis=1)
    before, after = values[rows, index], values[rows, index + 1]
    return crossed, rows, index, -before / (after - before)


def _interpolated_upcrossings(
    grid: FloatArray,
    crossed: NDArray[np.bool_],
    rows: NDArray[np.intp],
    index: NDArray[np.intp],
    fraction: FloatArray,
    rate_before: FloatArray,
    rate_after: FloatArray,
) -> FirstUpcrossings:
    time = np.full(crossed.size, np.nan)
    speed = np.full(crossed.size, np.nan)
    time[rows] = grid[index] + fraction * (grid[index + 1] - grid[index])
    speed[rows] = rate_before + fraction * (rate_after - rate_before)
    return FirstUpcrossings(crossed=crossed, time=time, speed=speed)


def first_upcrossings_on_grid(samples: ArrayLike, horizon: float, dt: float) -> FirstUpcrossings:
    """First upcrossing of e0 + v0 t + a t^2 / 2 on a grid of step <= dt over [0, horizon]."""
    states = np.atleast_2d(np.asarray(samples, dtype=float))
    if states.shape[1] != 3 or horizon <= 0.0 or dt <= 0.0:
        raise ValueError("samples must be (n, 3); horizon and dt must be positive")
    steps = max(1, int(np.ceil(horizon / dt - 1e-9)))
    grid = np.linspace(0.0, horizon, steps + 1)
    block = max(1, _GRID_BLOCK_ELEMENTS // grid.size)
    parts = []
    for start in range(0, states.shape[0], block):
        e0, v0, a = states[start : start + block].T
        values = e0[:, None] + grid * (v0[:, None] + (0.5 * grid) * a[:, None])
        crossed, rows, index, fraction = _first_upward_brackets(values)
        parts.append(
            _interpolated_upcrossings(
                grid,
                crossed,
                rows,
                index,
                fraction,
                v0[rows] + a[rows] * grid[index],
                v0[rows] + a[rows] * grid[index + 1],
            )
        )
    if not parts:
        return FirstUpcrossings(np.zeros(0, dtype=bool), np.zeros(0), np.zeros(0))
    return FirstUpcrossings(
        crossed=np.concatenate([part.crossed for part in parts]),
        time=np.concatenate([part.time for part in parts]),
        speed=np.concatenate([part.speed for part in parts]),
    )


def first_upcrossings_exact(samples: ArrayLike, horizon: float) -> FirstUpcrossings:
    """Exact first upcrossing in (0, horizon] of e0 + v0 t + a t^2 / 2 (the dt -> 0 reference).

    The upcrossing root is the one with slope +sqrt(v0^2 - 2 a e0), t = -2 e0 / (v0 + sqrt(D));
    tangencies (D = 0) are not crossings.
    """
    states = np.atleast_2d(np.asarray(samples, dtype=float))
    e0, v0, a = states.T
    discriminant = v0**2 - 2.0 * a * e0
    root = np.sqrt(np.maximum(discriminant, 0.0))
    denominator = v0 + root
    with np.errstate(divide="ignore", invalid="ignore"):
        time = np.where(
            denominator > 0.0,
            -2.0 * e0 / np.where(denominator > 0.0, denominator, 1.0),
            np.where(a != 0.0, (root - v0) / np.where(a != 0.0, a, 1.0), np.nan),
        )
    crossed = (discriminant > 0.0) & np.isfinite(time) & (time > 0.0) & (time <= horizon)
    return FirstUpcrossings(
        crossed=crossed,
        time=np.where(crossed, time, np.nan),
        speed=np.where(crossed, root, np.nan),
    )


def constant_acceleration_hazard_mc(
    mean: ArrayLike,
    cov: ArrayLike,
    horizon: float,
    v_b: float,
    n_samples: int,
    rng: np.random.Generator,
    dt: float,
) -> float:
    """IV.9 Monte Carlo hazard: fraction of sampled (e0, v0, a) trajectories whose first
    upcrossing of e = 0 in [0, horizon], detected on a grid of step dt, exceeds v_b."""
    if int(n_samples) < 1:
        raise ValueError("n_samples must be positive")
    samples = sample_constant_acceleration(mean, cov, n_samples, rng)
    return float(np.mean(first_upcrossings_on_grid(samples, horizon, dt).dangerous(v_b)))
