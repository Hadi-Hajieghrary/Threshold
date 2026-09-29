"""The velocity-matching catch of plan v2 (II.10, Proposition 14; IV.7 supervisor).

While cable i's slack flag is set and the estimated chord rate ``edot_hat`` exceeds the
landing profile

    edot_ref(e) = sqrt(v_soft^2 + 2 a_c (-e)),   v_soft = 0.3 v_b,   a_c = 0.2 m/s^2,

the vessel's surge force is

    F = clip(F_T s + K_v (edot_ref - edot_hat), F_min, F_T),   K_v = 2000 N s/m,

and the law releases (F = F_T s) when the tension reads taut.  ``F_T`` is the scheduled
thrust, ``s`` the easing feed-forward (1 = none) and ``F_min`` one of {0, -F_T} (the hold
and reverse variants).

Sign convention (from ``tether.physics.fleet.cable_kinematics``): the chord points from
the load attachment to the vessel stern, the elongation is ``e = |chord| - L`` and
``edot = (v_stern - v_load) . d`` is positive while the cable lengthens.  A slack cable
(e < 0) closing back toward e = 0 therefore has ``edot > 0``, and its re-engagement speed
is ``edot`` at the upcrossing of ``e = 0``.  Thrust along the hull pulls the vessel away
from the load, so reducing it lowers ``edot``.

Two operational details the plan does not print are fixed here: for ``e_hat >= 0`` the
profile is ``v_soft`` (the square root's argument is taken with ``max(-e_hat, 0)``), and a
non-finite estimate (no precursor published) means no catch (``F_T s``).

Pure functions only: no Drake, no state.  Scalars or arrays broadcast.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

SOFT_SPEED_FRACTION = 0.3
LANDING_DECELERATION = 0.2
VELOCITY_GAIN = 2000.0
HOLD = 0.0
REVERSE = -1.0
VARIANTS = {"hold": HOLD, "reverse": REVERSE}


def landing_profile(
    e_hat: ArrayLike,
    critical_speed: ArrayLike,
    soft_fraction: float = SOFT_SPEED_FRACTION,
    deceleration: float = LANDING_DECELERATION,
):
    """edot_ref(e) = sqrt(v_soft^2 + 2 a_c max(-e, 0)) with v_soft = soft_fraction * v_b."""
    e_hat = np.asarray(e_hat, dtype=float)
    soft = soft_fraction * np.asarray(critical_speed, dtype=float)
    value = np.sqrt(soft * soft + 2.0 * deceleration * np.maximum(-e_hat, 0.0))
    return float(value) if value.ndim == 0 else value


def catch_engaged(e_hat: ArrayLike, edot_hat: ArrayLike, slack: ArrayLike, critical_speed: ArrayLike, **profile):
    """True where the slack flag is set, the estimate is finite and edot_hat > edot_ref(e_hat)."""
    e_hat = np.asarray(e_hat, dtype=float)
    edot_hat = np.asarray(edot_hat, dtype=float)
    finite = np.isfinite(e_hat) & np.isfinite(edot_hat)
    reference = landing_profile(np.where(finite, e_hat, 0.0), critical_speed, **profile)
    engaged = np.asarray(slack, dtype=bool) & finite & (np.where(finite, edot_hat, -np.inf) > reference)
    return bool(engaged) if engaged.ndim == 0 else engaged


def catch_thrust(
    e_hat: ArrayLike,
    edot_hat: ArrayLike,
    slack: ArrayLike,
    scheduled_thrust: ArrayLike,
    critical_speed: ArrayLike,
    min_fraction: float,
    easing: ArrayLike = 1.0,
    gain: float = VELOCITY_GAIN,
    soft_fraction: float = SOFT_SPEED_FRACTION,
    deceleration: float = LANDING_DECELERATION,
):
    """Surge force of Prop. 14's catch.

    ``min_fraction`` sets F_min = min_fraction * F_T (0 hold, -1 reverse).  Released
    (``F_T * easing``) where the slack flag is clear, the estimate is non-finite, or
    ``edot_hat <= edot_ref(e_hat)``.
    """
    thrust = np.asarray(scheduled_thrust, dtype=float)
    released = thrust * np.asarray(easing, dtype=float)
    e_hat = np.asarray(e_hat, dtype=float)
    edot_hat = np.asarray(edot_hat, dtype=float)
    finite = np.isfinite(e_hat) & np.isfinite(edot_hat)
    safe_e = np.where(finite, e_hat, 0.0)
    safe_rate = np.where(finite, edot_hat, 0.0)
    reference = landing_profile(safe_e, critical_speed, soft_fraction, deceleration)
    engaged = np.asarray(slack, dtype=bool) & finite & (safe_rate > reference)
    low = min_fraction * thrust
    commanded = np.clip(released + gain * (reference - safe_rate), np.minimum(low, thrust), np.maximum(low, thrust))
    value = np.where(engaged, commanded, released)
    return float(value) if value.ndim == 0 else value
