"""Drag-inclusive one-cable slack excursion (plan v2 II.1, II.4-II.6, Appendix A.2-A.3).

Drake-free.  Two layers:

* **The reduced drag-inclusive one-cable ODE** of II.1 / Thm 6', states
  ``x = (e, dv_A, dv_L)`` (elongation, vessel and load-side radial velocity perturbations
  from steady tow), inputs ``u = (T0, W_A . d, W_L . d)``::

      e'    = dv_A - dv_L
      dv_A' = (T0 - T + W_A . d)/m_A  - dv_A / tau_A          tau_A  = m_A/c_A  = 1.714 s
      dv_L' = (T - T0 + W_L . d)/m_Lf - dv_L / tau_Lf         tau_Lf = m_Lf/c_Lf = 0.710 s
      T     = (k e + c e')_+ 1[e > 0]

  with the fleet-side load ``m_Lf = m_L + 4 m_A``, ``c_Lf = c_L + 4 c_A`` (the load is held
  by the four other cables).  The system is linear in each of its two modes (taut:
  ``T = k e + c e'``; slack: ``T = 0``), so ``OneCableODE`` discretizes each mode exactly
  under a zero-order-hold input and ``simulate`` steps many paths at once, choosing the
  mode from the state at the start of each step (as the plant's discrete cable does) and
  recording marks on a 1 ms sample grid with the plant tracker's linear interpolation.

* **Closed forms** of the three-piece map (impulse, drag-limited coast, undriven return),
  the R2 drift, Cor. 5' (square-gust severance level with the ``m_fleet`` return), the
  drag-inclusive severance level, Prop. 2 and Prop. 3', in the vessel-dominant reduction
  of Prop. 4' (load pinned, vessel alone, lag ``tau_A``).  ``slack_solution`` is the exact
  two-lag solution of the ODE's slack mode under constant forcing.

Default parameters are the pinned constants (``tether.physics.constants``) and the
Phase 1 pivots: ``Z = 7993.5 N s/m``, ``f = 0.880``, ``m_fleet = 523.9 kg``.
Array arguments broadcast; scalar inputs return ``float``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm
from scipy.optimize import brentq

from tether.physics import constants

FloatArray = NDArray[np.float64]

M_A = constants.VESSEL_MASS
C_A = constants.VESSEL_LINEAR_DRAG
M_L = constants.LOAD_MASS
C_L = constants.LOAD_LINEAR_DRAG
OTHER_VESSELS = constants.VESSEL_COUNT - 1
M_LF = M_L + OTHER_VESSELS * M_A
C_LF = C_L + OTHER_VESSELS * C_A
TAU_A = M_A / C_A
TAU_LF = M_LF / C_LF
TAU_L = M_L / C_L
C_EFF = 1.0 / (1.0 / C_A + 1.0 / C_L)
M_EFF = M_A * M_L / (M_A + M_L)
M_FLEET = 523.9
B_FLEET = 1.0 - M_EFF / M_FLEET
STIFFNESS = constants.CABLE_STIFFNESS
DAMPING = constants.CABLE_DAMPING
IMPEDANCE = 7993.5
IMPACT_FACTOR = 0.880
R2_BOUNDARY = 2.0 * TAU_A
# Vessel-only body force (along -d) that realises a W^c of 1 N: W^c = c_eff (0/c_L + F/c_A).
VESSEL_FORCE_PER_WC = C_A / C_EFF
# W_rel carried by that force: m_eff F / m_A.
WREL_PER_WC_VESSEL_ONLY = M_EFF / M_A * VESSEL_FORCE_PER_WC

WEATHER_PERIOD = constants.WEATHER_PERIOD
PHYSICS_STEP = 5.0e-4
SAMPLE_PERIOD = 1.0e-3


def _result(value: ArrayLike) -> float | FloatArray:
    array = np.asarray(value, dtype=float)
    return float(array) if array.ndim == 0 else array


def terminal_speed(pretension: ArrayLike, c_a: float = C_A) -> float | FloatArray:
    """v_T = T0/c_A: the vessel's terminal speed against the released pretension."""
    return _result(np.asarray(pretension, dtype=float) / c_a)


# ----------------------------------------------------------------------------- R2 drift (Prop. 4')


def drift_speed(w_c: ArrayLike, pretension: ArrayLike, c_eff: float = C_EFF) -> float | FloatArray:
    """v_d = (W^c - T0)/c_eff (negative below the criterion)."""
    return _result((np.asarray(w_c, dtype=float) - np.asarray(pretension, dtype=float)) / c_eff)


def shutoff_speed(v_d: ArrayLike, t_x: ArrayLike, tau: float = TAU_A) -> float | FloatArray:
    """v_s = v_d (1 - exp(-t_x/tau_A)): closing speed at gust shut-off from rest."""
    return _result(np.asarray(v_d, dtype=float) * -np.expm1(-np.asarray(t_x, dtype=float) / tau))


def shutoff_depth(v_d: ArrayLike, t_x: ArrayLike, tau: float = TAU_A) -> float | FloatArray:
    """Delta_g = v_d [t_x - tau_A (1 - exp(-t_x/tau_A))]: depth at shut-off, linear in t_x >> tau_A."""
    duration = np.asarray(t_x, dtype=float)
    return _result(np.asarray(v_d, dtype=float) * (duration + tau * np.expm1(-duration / tau)))


def post_gust_deepening(v_s: ArrayLike, pretension: ArrayLike, tau: float = TAU_A, c_a: float = C_A) -> float | FloatArray:
    """delta Delta = tau_A [v_s - v_T ln(1 + v_s/v_T)]: the drag-limited coast after shut-off."""
    speed = np.maximum(np.asarray(v_s, dtype=float), 0.0)
    v_t = np.asarray(pretension, dtype=float) / c_a
    return _result(tau * (speed - v_t * np.log1p(speed / v_t)))


def post_gust_stop_time(v_s: ArrayLike, pretension: ArrayLike, tau: float = TAU_A, c_a: float = C_A) -> float | FloatArray:
    """t_0 = tau_A ln(1 + v_s/v_T): time from shut-off to the deepest point."""
    speed = np.maximum(np.asarray(v_s, dtype=float), 0.0)
    return _result(tau * np.log1p(speed * c_a / np.asarray(pretension, dtype=float)))


def _return_fraction(depth_ratio: float) -> float:
    """x in (0, 1] with -ln x - (1 - x) = depth_ratio."""
    if depth_ratio <= 0.0:
        return 1.0
    if depth_ratio < 1.0e-8:
        return 1.0 - math.sqrt(2.0 * depth_ratio)
    upper = 1.0 - 1.0e-15
    lower = min(0.5, math.exp(-depth_ratio - 1.0))
    while -math.log(lower) - (1.0 - lower) < depth_ratio:
        lower *= 0.5
    return brentq(lambda x: -math.log(x) - (1.0 - x) - depth_ratio, lower, upper, xtol=1e-15, rtol=4.0 * np.finfo(float).eps)


def undriven_return_speed(depth: ArrayLike, pretension: ArrayLike, tau: float = TAU_A, c_a: float = C_A) -> float | FloatArray:
    """V_up = v_T (1 - x) with v_T tau_A [-ln x - (1 - x)] = Delta (vessel-dominant, undriven).

    Reduces to sqrt(2 T0 Delta/m_A) for Delta << v_T tau_A/2 and saturates at v_T.
    """
    depth, pretension = np.broadcast_arrays(np.asarray(depth, dtype=float), np.asarray(pretension, dtype=float))
    v_t = pretension / c_a
    ratio = np.maximum(depth, 0.0) / (v_t * tau)
    x = np.vectorize(_return_fraction, otypes=[float])(ratio)
    return _result(v_t * (1.0 - x))


def undriven_return_time(depth: ArrayLike, pretension: ArrayLike, tau: float = TAU_A, c_a: float = C_A) -> float | FloatArray:
    """Duration of the undriven return leg, -tau_A ln x."""
    depth, pretension = np.broadcast_arrays(np.asarray(depth, dtype=float), np.asarray(pretension, dtype=float))
    v_t = pretension / c_a
    x = np.vectorize(_return_fraction, otypes=[float])(np.maximum(depth, 0.0) / (v_t * tau))
    return _result(-tau * np.log(x))


def undriven_ceiling(pretension: ArrayLike, impedance: float = IMPEDANCE, c_a: float = C_A) -> float | FloatArray:
    """Z v_T = Z T0/c_A: the largest reference-law peak an undriven return reaches (a reference, never a bound)."""
    return _result(impedance * np.asarray(pretension, dtype=float) / c_a)


def critical_depth(v_b: ArrayLike, pretension: ArrayLike, tau: float = TAU_A, c_a: float = C_A) -> float | FloatArray:
    """Delta_b = v_T tau_A [-ln(1 - v_b/v_T) - v_b/v_T]: undriven-return depth that closes at v_b (inf at v_b >= v_T)."""
    speed, pretension = np.broadcast_arrays(np.asarray(v_b, dtype=float), np.asarray(pretension, dtype=float))
    v_t = pretension / c_a
    rho = speed / v_t
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(rho < 1.0, v_t * tau * (-np.log1p(-np.minimum(rho, 1.0 - 1e-300)) - rho), np.inf)
    return _result(np.where(speed <= 0.0, 0.0, depth))


def drag_inclusive_severance_level(
    v_b: ArrayLike, pretension: ArrayLike, t_x: ArrayLike, exact_depth: bool = False, c_eff: float = C_EFF, tau: float = TAU_A
) -> float | FloatArray:
    """Cor. 5' drag-inclusive form: W^c > T0 + c_eff Delta_b/(t_x - tau_A) (primary for R2).

    ``exact_depth`` inverts the full shut-off depth, Delta_g(W^c, t_x) = Delta_b, instead
    of its large-t_x asymptote; both ignore the post-gust deepening (conservative).
    """
    depth = np.asarray(critical_depth(v_b, pretension), dtype=float)
    duration = np.asarray(t_x, dtype=float)
    if exact_depth:
        lever = duration + tau * np.expm1(-duration / tau)
    else:
        lever = np.where(duration > tau, duration - tau, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _result(np.asarray(pretension, dtype=float) + c_eff * depth / lever)


# ----------------------------------------------------------------------------- Cor. 5' square gust


def square_gust_return_speed(
    w: ArrayLike, pretension: ArrayLike, t_g: ArrayLike, m_eff: float = M_EFF, b: float = B_FLEET
) -> float | FloatArray:
    """Cor. 5': V_up = (t_g/m_eff) sqrt((W - T0)(W - b T0)), free-body square gust from u = 0 (0 below T0)."""
    load = np.asarray(w, dtype=float)
    tension = np.asarray(pretension, dtype=float)
    excess = np.maximum(load - tension, 0.0)
    return _result(np.asarray(t_g, dtype=float) / m_eff * np.sqrt(excess * np.maximum(load - b * tension, 0.0)))


def square_gust_kappa(pretension: ArrayLike, t_g: ArrayLike, m_eff: float = M_EFF, f: float = IMPACT_FACTOR, k: float = STIFFNESS) -> float | FloatArray:
    return _result(m_eff / (f * f * k * np.asarray(pretension, dtype=float) * np.asarray(t_g, dtype=float) ** 2))


def square_gust_severance_level(
    t_b: ArrayLike, pretension: ArrayLike, t_g: ArrayLike, b: float = B_FLEET, m_eff: float = M_EFF, f: float = IMPACT_FACTOR, k: float = STIFFNESS
) -> float | FloatArray:
    """Cor. 5': W*(T_b) = ((1 + b) T0 + sqrt((1 - b)^2 T0^2 + 4 kappa T0 T_b^2))/2, with T_b = Z V_up."""
    tension = np.asarray(pretension, dtype=float)
    kappa = np.asarray(square_gust_kappa(tension, t_g, m_eff, f, k), dtype=float)
    level = np.asarray(t_b, dtype=float)
    return _result(0.5 * ((1.0 + b) * tension + np.sqrt((1.0 - b) ** 2 * tension**2 + 4.0 * kappa * tension * level**2)))


# ----------------------------------------------------------------------------- Props. 2 and 3'


def ballistic_excursion(u: ArrayLike, a: ArrayLike) -> dict[str, float | FloatArray]:
    """Prop. 2: from onset speed u under constant restoring acceleration a > 0: time 2u/a, speed u, depth u^2/2a."""
    speed = np.asarray(u, dtype=float)
    acceleration = np.asarray(a, dtype=float)
    return {
        "return_time": _result(2.0 * speed / acceleration),
        "return_speed": _result(speed + 0.0 * acceleration),
        "depth": _result(speed**2 / (2.0 * acceleration)),
    }


def closing_speed(a_bar_ret: ArrayLike, depth: ArrayLike) -> float | FloatArray:
    """Prop. 3': V_up = sqrt(2 a_bar_ret Delta), both measured from the deepest point."""
    return _result(np.sqrt(np.maximum(2.0 * np.asarray(a_bar_ret, dtype=float) * np.asarray(depth, dtype=float), 0.0)))


# ----------------------------------------------------------------------------- R1 three-piece map


@dataclass(frozen=True)
class ThreePieceMap:
    """The R1 three-piece map of II.4 for one square W^c impulse."""

    shutoff_speed: float
    impulse_depth: float
    coast_depth: float
    total_depth: float
    return_speed: float
    ballistic_speed: float
    first_order_ratio: float


def three_piece_map(
    amplitude: float, pretension: float, t_x: float, include_impulse_depth: bool = False, drag_in_impulse: bool = False
) -> ThreePieceMap:
    """Impulse leg, drag-limited coast, undriven return (vessel-dominant reduction, u = 0).

    Impulse: ``m_A v_s = (W^c_max - T0) t_x`` (``drag_in_impulse``: the exact vessel leg
    ``v_s = (A - T0)/c_A (1 - exp(-t_x/tau_A))``).  The plan's committed map treats the leg
    as impulsive and neglects its depth; ``include_impulse_depth`` adds it (drag-free
    ``(A - T0) t_x^2/(2 m_A)`` or the exact vessel leg's depth).  Coast:
    ``tau_A [v_s - v_T ln(1 + v_s/v_T)]``.  Return: undriven from the total depth.
    """
    excess = max(amplitude - pretension, 0.0)
    if drag_in_impulse:
        v_inf = excess / C_A
        v_s = v_inf * -math.expm1(-t_x / TAU_A)
        leg = v_inf * (t_x + TAU_A * math.expm1(-t_x / TAU_A))
    else:
        v_s = excess * t_x / M_A
        leg = 0.5 * excess * t_x**2 / M_A
    impulse_depth = leg if include_impulse_depth else 0.0
    coast = float(post_gust_deepening(v_s, pretension))
    total = impulse_depth + coast
    v_up = float(undriven_return_speed(total, pretension))
    v_t = pretension / C_A
    return ThreePieceMap(v_s, impulse_depth, coast, total, v_up, v_s, 1.0 - 2.0 * v_s / (3.0 * v_t))


def required_impulse(v_b: ArrayLike, pretension: ArrayLike, m_a: float = M_A, c_a: float = C_A) -> float | FloatArray:
    """I*: the impulse m_A v_s whose drag-limited coast and undriven return close at v_b.

    Solves r - ln(1 + r) = -ln(1 - rho) - rho, rho = v_b/v_T, r = v_s/v_T (the impulse
    leg's own depth neglected, as in the plan's committed map).  inf for v_b >= v_T.
    """
    speed, pretension = np.broadcast_arrays(np.asarray(v_b, dtype=float), np.asarray(pretension, dtype=float))
    v_t = pretension / c_a

    def solve(rho: float) -> float:
        if rho <= 0.0:
            return 0.0
        if rho >= 1.0:
            return math.inf
        target = -math.log1p(-rho) - rho
        upper = 1.0
        while upper - math.log1p(upper) < target:
            upper *= 2.0
        return brentq(lambda r: r - math.log1p(r) - target, 0.0, upper, xtol=1e-14, rtol=4.0 * np.finfo(float).eps)

    ratio = np.vectorize(solve, otypes=[float])(speed / v_t)
    return _result(m_a * v_t * ratio)


def r1_required_amplitude(v_b: ArrayLike, pretension: ArrayLike, t_x: ArrayLike) -> float | FloatArray:
    """A*(T_b, t_x) = T0 + I*(T_b)/t_x: the R1 amplitude surface of prediction 6."""
    return _result(np.asarray(pretension, dtype=float) + np.asarray(required_impulse(v_b, pretension), dtype=float) / np.asarray(t_x, dtype=float))


def r1_rate_curve(amplitude: ArrayLike, pretension: float, sigma_a: float, episode_rate: float) -> float | FloatArray:
    """Thm 6' R1 closed form: Lambda = nu_R1 exp(-(A*^2 - T0^2)/(2 sigma_A^2)) (Lambda = nu_R1 at A* = T0)."""
    level = np.asarray(amplitude, dtype=float)
    return _result(episode_rate * np.exp(-(level**2 - pretension**2) / (2.0 * sigma_a**2)))


# ----------------------------------------------------------------------------- the ODE


@dataclass(frozen=True)
class OneCableODE:
    """Parameters of the reduced drag-inclusive one-cable ODE (inputs u = (T0, W_A.d, W_L.d))."""

    m_a: float = M_A
    c_a: float = C_A
    m_l: float = M_LF
    c_l: float = C_LF
    stiffness: float = STIFFNESS
    damping: float = DAMPING
    pinned_load: bool = False

    def matrices(self, taut: bool) -> tuple[np.ndarray, np.ndarray]:
        k = self.stiffness if taut else 0.0
        c = self.damping if taut else 0.0
        a = np.array(
            [
                [0.0, 1.0, -1.0],
                [-k / self.m_a, -(c + self.c_a) / self.m_a, c / self.m_a],
                [k / self.m_l, c / self.m_l, -(c + self.c_l) / self.m_l],
            ]
        )
        b = np.array([[0.0, 0.0, 0.0], [1.0 / self.m_a, 1.0 / self.m_a, 0.0], [-1.0 / self.m_l, 0.0, 1.0 / self.m_l]])
        if self.pinned_load:
            a[2, :] = 0.0
            b[2, :] = 0.0
        return a, b

    def discretized(self, taut: bool, step: float) -> tuple[np.ndarray, np.ndarray]:
        """Exact zero-order-hold map: x(t + h) = Phi x(t) + Gamma u."""
        a, b = self.matrices(taut)
        augmented = np.zeros((6, 6))
        augmented[:3, :3] = a
        augmented[:3, 3:] = b
        exponential = expm(augmented * step)
        return exponential[:3, :3], exponential[:3, 3:]

    def equilibrium(self, pretension: float) -> np.ndarray:
        return np.array([pretension / self.stiffness, 0.0, 0.0])

    def tension(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        e, rate = x[0], x[1] - x[2]
        return np.where(e > 0.0, np.maximum(self.stiffness * e + self.damping * rate, 0.0), 0.0)

    @property
    def relative_mass(self) -> float:
        """Reduced mass of the relative coordinate (m_A for a pinned load)."""
        if self.pinned_load:
            return self.m_a
        return self.m_a * self.m_l / (self.m_a + self.m_l)

    @property
    def drift_crossing_vessel_only(self) -> float:
        """lambda at which a vessel-only W^c = lambda T0 gives zero terminal relative speed.

        Vessel force F = lambda T0 c_A/c_eff: T0/c_A + T0/c_l - F/c_A = 0.  Exactly 1 when
        the load side carries the free-load drag c_L (the plan's c_eff); 0.988 with the
        fleet-side c_Lf.
        """
        load = 0.0 if self.pinned_load else self.c_a / self.c_l
        return (1.0 + load) / VESSEL_FORCE_PER_WC

    @property
    def onset_acceleration_crossing_vessel_only(self) -> float:
        """lambda at which the slack relative acceleration at zero relative speed changes sign (mass form)."""
        load = 0.0 if self.pinned_load else self.m_a / self.m_l
        return (1.0 + load) / VESSEL_FORCE_PER_WC


def slack_solution(
    ode: OneCableODE, x0: ArrayLike, pretension: float, w_a: float, w_l: float, t: ArrayLike
) -> np.ndarray:
    """Exact slack-mode state (T = 0) under constant forcing: two first-order lags. Shape (3,) + t.shape."""
    time = np.asarray(t, dtype=float)
    e0, va0, vl0 = (float(v) for v in np.asarray(x0, dtype=float))

    def lag(v0: float, force: float, mass: float, drag: float) -> tuple[np.ndarray, np.ndarray]:
        if drag == 0.0:  # drag-free limit: constant acceleration
            acceleration = force / mass
            return v0 + acceleration * time, v0 * time + 0.5 * acceleration * time**2
        tau = mass / drag
        v_inf = force / drag
        return v_inf + (v0 - v_inf) * np.exp(-time / tau), v_inf * time - (v0 - v_inf) * tau * np.expm1(-time / tau)

    va, ea = lag(va0, pretension + w_a, ode.m_a, ode.c_a)
    if ode.pinned_load:
        vl = np.zeros_like(time)
        el = np.zeros_like(time)
    else:
        vl, el = lag(vl0, -pretension + w_l, ode.m_l, ode.c_l)
    return np.stack([e0 + ea - el, va, vl])


# ----------------------------------------------------------------------------- vectorized simulation


@dataclass
class Marks:
    """Re-engagement marks of ``simulate`` (one row per completed slack interval).

    Onset: geometric down-crossing of e = 0 on the 1 ms sample grid; re-engagement: the
    next up-crossing; times and speeds linearly interpolated between samples exactly as
    the plant's ``CableEventTracker`` does; depth is -min e over the interval's samples.
    ``censored_onsets`` are slack intervals still open at the end of the record.
    """

    path: list = field(default_factory=list)
    t_on: list = field(default_factory=list)
    u_entry: list = field(default_factory=list)
    t_up: list = field(default_factory=list)
    v_up: list = field(default_factory=list)
    depth: list = field(default_factory=list)
    t_deep: list = field(default_factory=list)
    censored_onsets: list = field(default_factory=list)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "path": np.asarray(self.path, dtype=np.int64),
            "t_on": np.asarray(self.t_on, dtype=float),
            "u_entry": np.asarray(self.u_entry, dtype=float),
            "t_up": np.asarray(self.t_up, dtype=float),
            "v_up": np.asarray(self.v_up, dtype=float),
            "depth": np.asarray(self.depth, dtype=float),
            "t_deep": np.asarray(self.t_deep, dtype=float),
        }


@dataclass
class SimulationResult:
    marks: Marks
    trace_time: np.ndarray | None = None
    trace_state: np.ndarray | None = None  # (samples, 3, paths) on the 1 ms grid
    trace_tension: np.ndarray | None = None  # (samples, paths) applied tension on the 1 ms grid


def simulate(
    ode: OneCableODE,
    pretension: ArrayLike,
    w_a: np.ndarray,
    w_l: np.ndarray,
    *,
    period: float = WEATHER_PERIOD,
    step: float = PHYSICS_STEP,
    sample: float = SAMPLE_PERIOD,
    x0: np.ndarray | None = None,
    trace: bool = False,
    columns_a: np.ndarray | None = None,
    columns_l: np.ndarray | None = None,
) -> SimulationResult:
    """Integrate many independent paths driven by ZOH forcing samples ``w_a, w_l`` (samples, columns).

    Each forcing sample holds over ``period``; the state advances by exact per-mode maps of
    length ``step``, the mode (taut iff e > 0 and k e + c e' > 0) taken from the state at
    the start of each step.  Marks are detected on the ``sample`` grid (1 ms) starting at
    t = 0.  ``x0`` defaults to the taut equilibrium of each path.  Path p reads column
    ``columns_a[p]`` of ``w_a`` and ``columns_l[p]`` of ``w_l`` (default: column p), so
    paths that share a forcing series need not duplicate it.
    """
    w_a = np.asarray(w_a, dtype=float)
    w_l = np.asarray(w_l, dtype=float)
    if w_a.ndim == 1:
        w_a = w_a[:, None]
        w_l = w_l[:, None]
    samples = w_a.shape[0]
    if w_l.shape[0] != samples:
        raise ValueError("w_a and w_l must have the same number of samples")
    if columns_a is None:
        columns_a = np.arange(w_a.shape[1])
    if columns_l is None:
        columns_l = np.arange(w_l.shape[1])
    columns_a = np.asarray(columns_a, dtype=np.int64)
    columns_l = np.asarray(columns_l, dtype=np.int64)
    if columns_a.shape != columns_l.shape:
        raise ValueError("columns_a and columns_l must have one entry per path")
    paths = columns_a.size
    tension0 = np.broadcast_to(np.asarray(pretension, dtype=float), (paths,)).copy()
    substeps = int(round(period / step))
    per_sample = int(round(sample / step))
    if abs(substeps * step - period) > 1e-12 or abs(per_sample * step - sample) > 1e-12 or substeps % per_sample:
        raise ValueError("period must be a multiple of sample, which must be a multiple of step")
    phi_t, gam_t = ode.discretized(True, step)
    phi_s, gam_s = ode.discretized(False, step)
    if x0 is None:
        state = np.stack([tension0 / ode.stiffness, np.zeros(paths), np.zeros(paths)])
    else:
        state = np.array(x0, dtype=float).reshape(3, paths).copy()
    k, c = ode.stiffness, ode.damping
    marks = Marks()
    prev_e = state[0].copy()
    prev_rate = state[1] - state[2]
    prev_t = 0.0
    open_onset = np.full(paths, np.nan)
    open_speed = np.full(paths, np.nan)
    running_min = np.full(paths, np.inf)
    running_min_t = np.full(paths, np.nan)
    slack_now = prev_e <= 0.0
    open_onset[slack_now] = 0.0
    open_speed[slack_now] = np.maximum(0.0, -prev_rate[slack_now])
    running_min[slack_now] = prev_e[slack_now]
    running_min_t[slack_now] = 0.0
    total_samples = samples * (substeps // per_sample)
    if trace:
        trace_time = np.arange(total_samples + 1) * sample
        trace_state = np.empty((total_samples + 1, 3, paths))
        trace_tension = np.empty((total_samples + 1, paths))
        trace_state[0] = state
        trace_tension[0] = ode.tension(state)
    counter = 0
    inputs = np.empty((3, paths))
    inputs[0] = tension0
    prev_pos = prev_e > 0.0
    for n in range(samples):
        inputs[1] = w_a[n, columns_a]
        inputs[2] = w_l[n, columns_l]
        forced_t = gam_t @ inputs
        forced_s = gam_s @ inputs
        for j in range(substeps):
            e = state[0]
            taut = (e > 0.0) & (k * e + c * (state[1] - state[2]) > 0.0)
            advanced = phi_t @ state + forced_t
            if not taut.all():
                slack = phi_s @ state + forced_s
                advanced = np.where(taut, advanced, slack)
            state = advanced
            if (j + 1) % per_sample:
                continue
            counter += 1
            time = counter * sample
            e = state[0]
            pos = e > 0.0
            if trace:
                trace_state[counter] = state
                trace_tension[counter] = ode.tension(state)
            rate = state[1] - state[2]
            if not pos.all():
                deeper = (e < running_min) & ~pos
                if deeper.any():
                    running_min = np.where(deeper, e, running_min)
                    running_min_t = np.where(deeper, time, running_min_t)
            changed = pos != prev_pos
            if changed.any():
                for p in np.flatnonzero(changed):
                    fraction = -prev_e[p] / (e[p] - prev_e[p])
                    t_cross = prev_t + fraction * sample
                    v_cross = prev_rate[p] + fraction * (rate[p] - prev_rate[p])
                    if e[p] <= 0.0:  # down-crossing: onset
                        open_onset[p] = t_cross
                        open_speed[p] = max(0.0, -v_cross)
                        running_min[p] = e[p]
                        running_min_t[p] = time
                    elif np.isfinite(open_onset[p]):  # up-crossing closing an open interval
                        marks.path.append(int(p))
                        marks.t_on.append(float(open_onset[p]))
                        marks.u_entry.append(float(open_speed[p]))
                        marks.t_up.append(float(t_cross))
                        marks.v_up.append(float(v_cross))
                        marks.depth.append(float(max(0.0, -running_min[p])))
                        marks.t_deep.append(float(running_min_t[p]))
                        open_onset[p] = np.nan
                        running_min[p] = np.inf
            prev_e = e
            prev_pos = pos
            prev_rate = rate
            prev_t = time
    for p in np.flatnonzero(np.isfinite(open_onset)):
        marks.censored_onsets.append((int(p), float(open_onset[p])))
    if trace:
        return SimulationResult(marks, trace_time, trace_state, trace_tension)
    return SimulationResult(marks)
