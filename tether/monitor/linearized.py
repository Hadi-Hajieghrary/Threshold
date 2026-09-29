"""Linearised closed-loop prediction model for the hazard (the P5-T2 pivot of plan IV.9).

While a cable is slack its vessel is a free body driven by thrust and damped by its own
drag, so the radial rate relaxes rather than accelerating without bound:

    edot' = a_eff - gamma edot,     gamma = c_A / m_A,

with ``a_eff`` the net forcing (thrust surplus, the load's deceleration, the slowly varying
gust) held constant over the horizon.  ``a_eff`` is identified from the precursor's current
acceleration estimate, ``a_eff = a_hat + gamma edot_0``, so the model differs from the
constant-acceleration model only by the drag relaxation.  Trajectories are analytic, so
the first upcrossing is found on a fine grid exactly as for the default model.
"""

from __future__ import annotations

import numpy as np

from tether.physics import constants

GAMMA = constants.VESSEL_LINEAR_DRAG / constants.VESSEL_MASS


def linearized_paths(e0: np.ndarray, edot0: np.ndarray, a_eff: np.ndarray, times: np.ndarray, gamma: float = GAMMA) -> tuple[np.ndarray, np.ndarray]:
    """(e, edot) at ``times`` for each sample; arrays (samples, len(times))."""
    decay = np.exp(-gamma * times)[None, :]
    terminal = (a_eff / gamma)[:, None]
    edot = terminal + (edot0[:, None] - terminal) * decay
    e = e0[:, None] + terminal * times[None, :] + (edot0[:, None] - terminal) * (1.0 - decay) / gamma
    return e, edot


def fleet_hazard_linearized(output, v_b: np.ndarray, master_seed: int, n_samples: int = 2048, horizon: float = 2.0) -> np.ndarray:
    """Hazard (m, N) under the linearised model, one generator per agent advanced only at
    slack ticks, in tick order (the same streams as the default model)."""
    from tether.monitor.hazard import hazard_generator, precursor_state

    slack = np.asarray(output.slack, dtype=bool)
    hazard = np.zeros(slack.shape)
    for agent in range(slack.shape[1]):
        generator = hazard_generator(master_seed, agent)
        for tick in np.flatnonzero(slack[:, agent]):
            e = output.e_hat[tick, agent]
            if not np.isfinite(e):
                hazard[tick, agent] = np.nan
                continue
            mean, covariance = precursor_state(e, output.edot_hat[tick, agent], output.sigma[tick, agent], output.a_hat[tick, agent], output.sigma_a[tick, agent])
            hazard[tick, agent] = linearized_hazard_mc(mean, covariance, horizon, float(np.asarray(v_b)[agent]), n_samples, generator)
    return hazard


def model_tick_hazard(model: str, e: float, edot: float, sigma, a_hat: float, sigma_a: float, v_b: float, rng: np.random.Generator,
                      n_samples: int = 2048, horizon: float = 2.0) -> float:
    """One tick of the hazard under ``model`` ("constant_acceleration" or "linearized"), drawing
    from ``rng`` exactly as the offline fleet evaluations do."""
    from tether.monitor.hazard import precursor_state, tick_hazard

    if model == "constant_acceleration":
        return tick_hazard(e, edot, sigma, a_hat, sigma_a, v_b, rng, n_samples=n_samples, horizon=horizon)
    if model != "linearized":
        raise ValueError(f"unknown hazard model {model!r}")
    if not np.isfinite(e):
        return float("nan")
    mean, covariance = precursor_state(e, edot, sigma, a_hat, sigma_a)
    return linearized_hazard_mc(mean, covariance, horizon, v_b, n_samples, rng)


def linearized_hazard_mc(mean: np.ndarray, covariance: np.ndarray, horizon: float, v_b: float, n_samples: int, rng: np.random.Generator,
                         dt: float = 0.005, gamma: float = GAMMA) -> float:
    """Fraction of sampled drag-damped trajectories whose first upcrossing of e = 0 in
    (0, H] has closing speed above ``v_b``.  ``mean``/``covariance`` are over
    (e, edot, a_hat) as for the constant-acceleration model."""
    samples = rng.multivariate_normal(np.asarray(mean, dtype=float), np.asarray(covariance, dtype=float), size=n_samples, method="cholesky")
    e0, edot0, a_hat = samples[:, 0], samples[:, 1], samples[:, 2]
    a_eff = a_hat + gamma * edot0
    times = np.arange(1, int(round(horizon / dt)) + 1) * dt
    e, edot = linearized_paths(e0, edot0, a_eff, times, gamma)
    previous = np.concatenate([e0[:, None], e[:, :-1]], axis=1)
    crossing = (previous <= 0.0) & (e > 0.0)
    has = crossing.any(axis=1)
    first = np.argmax(crossing, axis=1)
    rows = np.flatnonzero(has)
    if rows.size == 0:
        return 0.0
    k = first[rows]
    e_before = np.where(k > 0, e[rows, k - 1], e0[rows])
    edot_before = np.where(k > 0, edot[rows, k - 1], edot0[rows])
    fraction = -e_before / (e[rows, k] - e_before)
    speed = edot_before + fraction * (edot[rows, k] - edot_before)
    return float(np.sum(speed > v_b) / n_samples)
