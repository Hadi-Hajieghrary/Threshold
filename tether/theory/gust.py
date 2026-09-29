"""Weather-only gust-episode statistics for the threshold laws (Drake-free).

Everything here is computed from the weather model and the design geometry alone - no
plant simulation - so that the Phase 2 rate predictions are computed, not fitted.

Operational definitions (declared before the Phase 2 campaign):

* ``W_rel,i(t) = m_eff (W_0/m_L - W_i/m_A) . d_i`` with ``d_i`` the equilibrium chord
  direction and ``m_eff`` the two-body reduced mass of II.1.
* A gust episode of cable i is a run of ``W_rel,i > T0`` on the 10 ms weather grid,
  merged across gaps shorter than ``merge`` seconds.
* ``nu_ep``: episodes per unit time; ``t_g``: mean episode duration; ``delta_w``: mean
  excess ``W_rel - T0`` over each episode.
* Theorem 6 is assembled as ``Lambda(Tb) = nu_ep * S((T0 + kappa Tb^2 - mu_W)/sigma_W) /
  S((T0 - mu_W)/sigma_W)`` with ``S`` the normal survival function and
  ``kappa = m_eff/(f^2 k T0 t_g^2)``, so that its ``Tb -> 0`` limit is the measured
  episode rate (``nu_g = nu_ep / S((T0-mu_W)/sigma_W)``).
* The Corollary 5 functional applies ``T_snap = f t sqrt(k T0 delta/m_eff)`` to every
  weather episode of duration ``t`` and mean excess ``delta``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from tether.physics import constants
from tether.physics.weather import stationary_weather_forces

PAIR_REDUCED_MASS = constants.VESSEL_MASS * constants.LOAD_MASS / (constants.VESSEL_MASS + constants.LOAD_MASS)
DEFAULT_MERGE = 1.0


def relative_load_series(forces: np.ndarray, directions: np.ndarray, reduced_mass: float = PAIR_REDUCED_MASS) -> np.ndarray:
    """W_rel for every cable from weather forces (samples, N+1, 2) and chord directions (N, 2)."""
    load = forces[:, 0, :] / constants.LOAD_MASS
    vessels = forces[:, 1:, :] / constants.VESSEL_MASS
    return reduced_mass * np.einsum("snk,nk->sn", load[:, None, :] - vessels, directions)


def episodes(series: np.ndarray, level: float, period: float, merge: float = DEFAULT_MERGE) -> dict[str, np.ndarray]:
    above = series > level
    padded = np.concatenate([[False], above, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = changes[0::2], changes[1::2]
    if starts.size:
        gap = int(round(merge / period))
        keep = np.concatenate([[True], starts[1:] - stops[:-1] > gap])
        group = np.cumsum(keep) - 1
        merged_stops = np.zeros(int(group[-1]) + 1, dtype=np.int64)
        np.maximum.at(merged_stops, group, stops)
        starts = starts[keep]
        stops = merged_stops
    durations = (stops - starts) * period
    mean_excess = np.array([np.mean(np.maximum(series[a:b] - level, 0.0)) for a, b in zip(starts, stops)])
    max_excess = np.array([np.max(series[a:b] - level) for a, b in zip(starts, stops)])
    return {"duration": durations, "mean_excess": mean_excess, "max_excess": max_excess}


@dataclass(frozen=True)
class GustStatistics:
    pretension: float
    intensity: float
    mu_w: np.ndarray
    sigma_w: np.ndarray
    sigma_w_analytic: np.ndarray
    episode_rate: np.ndarray
    mean_duration: np.ndarray
    durations: list[np.ndarray]
    mean_excesses: list[np.ndarray]
    exposure: float

    def as_dict(self) -> dict:
        return {
            "pretension": self.pretension,
            "intensity": self.intensity,
            "mu_w": self.mu_w.tolist(),
            "sigma_w": self.sigma_w.tolist(),
            "sigma_w_analytic": self.sigma_w_analytic.tolist(),
            "episode_rate": self.episode_rate.tolist(),
            "mean_duration": self.mean_duration.tolist(),
            "episode_count": [int(d.size) for d in self.durations],
            "exposure": self.exposure,
        }


def analytic_sigma_w(intensity: float, directions: np.ndarray, direction: str = "local", reduced_mass: float = PAIR_REDUCED_MASS) -> np.ndarray:
    """Stationary std of W_rel for per-component stds (3.5, 0.7) kN x intensity."""
    load = intensity * constants.LOAD_WEATHER_STD / constants.LOAD_MASS
    vessel = intensity * constants.VESSEL_WEATHER_STD / constants.VESSEL_MASS
    norms = np.linalg.norm(directions, axis=1)
    if direction == "local":
        return reduced_mass * np.sqrt(load**2 + vessel**2) * norms
    if direction == "common":
        return reduced_mass * abs(load - vessel) * norms
    raise ValueError(direction)


def gust_statistics(
    pretension: float,
    intensity: float,
    directions: np.ndarray,
    seeds: range | list[int],
    duration: float,
    *,
    distribution: str = "gaussian",
    direction: str = "local",
    merge: float = DEFAULT_MERGE,
) -> GustStatistics:
    period = constants.WEATHER_PERIOD
    n = directions.shape[0]
    durations = [[] for _ in range(n)]
    excesses = [[] for _ in range(n)]
    total = 0.0
    sums = np.zeros(n)
    squares = np.zeros(n)
    count = 0
    for seed in seeds:
        forces = stationary_weather_forces(seed, duration, vessel_count=n, distribution=distribution, direction=direction, scale=intensity)
        series = relative_load_series(forces, directions)
        sums += series.sum(axis=0)
        squares += (series**2).sum(axis=0)
        count += series.shape[0]
        total += series.shape[0] * period
        for cable in range(n):
            found = episodes(series[:, cable], pretension, period, merge)
            durations[cable].append(found["duration"])
            excesses[cable].append(found["mean_excess"])
    durations = [np.concatenate(d) for d in durations]
    excesses = [np.concatenate(e) for e in excesses]
    mean = sums / count
    return GustStatistics(
        pretension=pretension,
        intensity=intensity,
        mu_w=mean,
        sigma_w=np.sqrt(squares / count - mean**2),
        sigma_w_analytic=analytic_sigma_w(intensity, directions, direction),
        episode_rate=np.array([d.size / total for d in durations]),
        mean_duration=np.array([float(np.mean(d)) if d.size else float("nan") for d in durations]),
        durations=durations,
        mean_excesses=excesses,
        exposure=total,
    )


def kappa(pretension: float, t_g: float, impact_factor: float, stiffness: float = constants.CABLE_STIFFNESS, reduced_mass: float = PAIR_REDUCED_MASS) -> float:
    return reduced_mass / (impact_factor**2 * stiffness * pretension * t_g**2)


def theorem6_snap_rate(levels: np.ndarray, pretension: float, episode_rate: float, t_g: float, mu_w: float, sigma_w: float, impact_factor: float, reduced_mass: float = PAIR_REDUCED_MASS) -> np.ndarray:
    coefficient = kappa(pretension, t_g, impact_factor, reduced_mass=reduced_mass)
    base = norm.sf((pretension - mu_w) / sigma_w)
    return episode_rate * norm.sf((pretension + coefficient * np.asarray(levels) ** 2 - mu_w) / sigma_w) / base


def theorem6_log_slope(pretension: float, t_g: float, sigma_w: float, impact_factor: float, reduced_mass: float = PAIR_REDUCED_MASS) -> float:
    """Coefficient of Tb^4 in log Lambda_snap: -kappa^2 / (2 sigma_W^2)."""
    coefficient = kappa(pretension, t_g, impact_factor, reduced_mass=reduced_mass)
    return -(coefficient**2) / (2.0 * sigma_w**2)


def corollary5_functional_rate(levels: np.ndarray, pretension: float, durations: np.ndarray, excesses: np.ndarray, exposure: float, impact_factor: float, stiffness: float = constants.CABLE_STIFFNESS, reduced_mass: float = PAIR_REDUCED_MASS) -> np.ndarray:
    snaps = impact_factor * durations * np.sqrt(stiffness * pretension * np.maximum(excesses, 0.0) / reduced_mass)
    return np.array([np.sum(snaps > level) / exposure for level in np.asarray(levels)])


def exact_severance_level(levels: np.ndarray, pretension: float, coefficient: float) -> np.ndarray:
    """Square-gust severance level including the post-gust closing (secondary form).

    Solving f t_g sqrt(k W (W - T0)/m_eff) = T_b for W gives
    W* = (T0 + sqrt(T0^2 + 4 kappa T0 T_b^2))/2, which equals T0 + kappa T_b^2 only while
    kappa T_b^2 << T0 (see the Phase 1 report).
    """
    levels = np.asarray(levels, dtype=float)
    return 0.5 * (pretension + np.sqrt(pretension**2 + 4.0 * coefficient * pretension * levels**2))


def theorem6_exact_snap_rate(levels: np.ndarray, pretension: float, episode_rate: float, t_g: float, mu_w: float, sigma_w: float, impact_factor: float, reduced_mass: float = PAIR_REDUCED_MASS) -> np.ndarray:
    coefficient = kappa(pretension, t_g, impact_factor, reduced_mass=reduced_mass)
    base = norm.sf((pretension - mu_w) / sigma_w)
    return episode_rate * norm.sf((exact_severance_level(levels, pretension, coefficient) - mu_w) / sigma_w) / base


def corollary5_exact_functional_rate(levels: np.ndarray, pretension: float, durations: np.ndarray, excesses: np.ndarray, exposure: float, impact_factor: float, stiffness: float = constants.CABLE_STIFFNESS, reduced_mass: float = PAIR_REDUCED_MASS) -> np.ndarray:
    load = pretension + np.maximum(excesses, 0.0)
    snaps = impact_factor * durations * np.sqrt(stiffness * load * np.maximum(excesses, 0.0) / reduced_mass)
    return np.array([np.sum(snaps > level) / exposure for level in np.asarray(levels)])


def taut_rate(levels: np.ndarray, mean_tension: float, sigma_q: float, sigma_qdot: float) -> np.ndarray:
    """Rice upcrossing rate of a stationary Gaussian tension (Theorem 6, taut branch)."""
    levels = np.asarray(levels, dtype=float)
    return (sigma_qdot / (2.0 * np.pi * sigma_q)) * np.exp(-((levels - mean_tension) ** 2) / (2.0 * sigma_q**2))
