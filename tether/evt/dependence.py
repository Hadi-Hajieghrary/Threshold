"""Extremal dependence: empirical chi, chibar and eta, and Gaussian-copula predictors."""

from __future__ import annotations

import numpy as np
from scipy import special, stats

from tether.evt._common import (
    as_generator,
    percentile_interval,
    scalar_or_array,
    two_sided_z,
)
from tether.evt.hill import hill_estimator

_BLOCK_CHUNK = 2_000_000
_LAGUERRE_NODES, _LAGUERRE_WEIGHTS = np.polynomial.laguerre.laggauss(64)
_LAGUERRE_MIN_EXPONENT = 2.0


def pseudo_observations(x) -> np.ndarray:
    """Ranks divided by ``n + 1`` (average ranks for ties), column-wise for 2-D input."""
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("pseudo-observations require finite data")
    return stats.rankdata(x, axis=0) / (x.shape[0] + 1.0)


def _pair(U, V) -> tuple[np.ndarray, np.ndarray]:
    U = np.asarray(U, dtype=float).ravel()
    V = np.asarray(V, dtype=float).ravel()
    if U.shape != V.shape or U.size == 0:
        raise ValueError("U and V must be non-empty and of equal length")
    return U, V


def _levels(u_values) -> np.ndarray:
    u = np.atleast_1d(np.asarray(u_values, dtype=float))
    if np.any((u <= 0.0) | (u >= 1.0)):
        raise ValueError("u values must lie strictly between 0 and 1")
    return u


def _joint_fraction(u: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.minimum(U, V))
    return (ordered.size - np.searchsorted(ordered, u, side="right")) / ordered.size


def _chi(u: np.ndarray, joint: np.ndarray) -> np.ndarray:
    return joint / (1.0 - u)


def _chibar(u: np.ndarray, joint: np.ndarray, empty: float = np.nan) -> np.ndarray:
    usable = (joint > 0.0) & (joint < 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = 2.0 * np.log1p(-u) / np.log(np.where(usable, joint, 0.5)) - 1.0
    return np.where(usable, value, np.where(joint == 0.0, empty, np.nan))


def chi_empirical(u_values, U, V) -> np.ndarray:
    """``P(U > u, V > u) / (1 - u)`` at each ``u`` for pseudo-observations ``U, V``."""
    U, V = _pair(U, V)
    u = _levels(u_values)
    return _chi(u, _joint_fraction(u, U, V))


def chibar_empirical(u_values, U, V) -> np.ndarray:
    """``2 log(1 - u) / log P(U > u, V > u) - 1``; NaN where no joint exceedance occurs."""
    U, V = _pair(U, V)
    u = _levels(u_values)
    return _chibar(u, _joint_fraction(u, U, V))


def dependence_bootstrap(
    U,
    V,
    u_values,
    block_length: int,
    n_boot: int = 500,
    rng: np.random.Generator | None = None,
    confidence: float = 0.95,
) -> dict[str, np.ndarray]:
    """Empirical chi and chibar with moving-block bootstrap percentile intervals.

    Each replicate concatenates ``ceil(n / block_length)`` blocks of consecutive time indices
    with uniform starts (the last block truncated), as in ``moving_block_indices``; the same
    blocks serve every level.  Joint exceedance counts come from per-level cumulative sums,
    so the cost per replicate is proportional to the number of blocks.  Margins are not
    re-ranked.  A replicate without joint exceedances has chibar -1, its limit as the joint
    probability vanishes, so sparse levels are not reported with falsely narrow intervals.
    """
    U, V = _pair(U, V)
    u = _levels(u_values)
    n = U.size
    if not 1 <= block_length <= n:
        raise ValueError("block_length must lie in [1, n]")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    block_count = -(-n // block_length)
    tail_length = n - (block_count - 1) * block_length
    rows_per_chunk = max(1, _BLOCK_CHUNK // block_count)
    block_seed = int(as_generator(rng).integers(2**63))
    minimum = np.minimum(U, V)
    joint = np.empty((n_boot, u.size))
    for index, level in enumerate(u):
        cumulative = np.concatenate(([0], np.cumsum(minimum > level)))
        blocks = np.random.default_rng(block_seed)
        for first in range(0, n_boot, rows_per_chunk):
            rows = min(rows_per_chunk, n_boot - first)
            starts = blocks.integers(0, n - block_length + 1, size=(rows, block_count))
            ends = starts + block_length
            ends[:, -1] = starts[:, -1] + tail_length
            joint[first:first + rows, index] = (
                cumulative[ends] - cumulative[starts]
            ).sum(axis=1) / n
    chi_lo, chi_hi = percentile_interval(_chi(u, joint), confidence, axis=0)
    chibar_lo, chibar_hi = percentile_interval(_chibar(u, joint, empty=-1.0), confidence, axis=0)
    observed = _joint_fraction(u, U, V)
    return {
        "u": u,
        "chi": _chi(u, observed),
        "chi_lo": chi_lo,
        "chi_hi": chi_hi,
        "chibar": _chibar(u, observed),
        "chibar_lo": chibar_lo,
        "chibar_hi": chibar_hi,
    }


def eta_ledford_tawn(
    U, V, tail_fraction: float = 0.05, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Ledford-Tawn coefficient of tail dependence with a delta-method interval.

    Margins go to unit Frechet, ``Z = -1 / log U``; ``T = min(Z1, Z2)`` is regularly varying
    with index ``1 / eta``, estimated by Hill on its top ``tail_fraction``; the interval is
    ``eta (1 -/+ z / sqrt(k))``.
    """
    U, V = _pair(U, V)
    if np.any((U <= 0.0) | (U >= 1.0) | (V <= 0.0) | (V >= 1.0)):
        raise ValueError("U and V must lie strictly between 0 and 1")
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("tail_fraction must lie strictly between 0 and 1")
    structure = np.minimum(-1.0 / np.log(U), -1.0 / np.log(V))
    k = int(np.floor(tail_fraction * structure.size))
    alpha = hill_estimator(structure, k, confidence)[0]
    eta = 1.0 / alpha
    half_width = two_sided_z(confidence) / float(np.sqrt(k))
    return eta, eta * (1.0 - half_width), eta * (1.0 + half_width)


def _joint_normal_log_survival(z, rho) -> np.ndarray:
    """``log P(X > z, Y > z)`` for a standard bivariate normal with correlation ``rho``.

    Owen's T gives ``Phi2(h, h; rho) = Phi(h) - 2 T(h, sqrt((1 - rho) / (1 + rho)))`` at
    ``h = -z``.  That difference cancels catastrophically for ``rho <= 0`` in the tail, where
    Plackett's identity is used instead: with ``c = z^2 / (1 + rho) >= 2`` and ``z > 0``,
    ``P = int_{-1}^{rho} phi2(z, z; r) dr
    = z e^{-c} / (2 pi) int_0^inf e^{-w} dw / ((c + w) sqrt(2 (c + w) - z^2))``,
    evaluated by 64-point Gauss-Laguerre quadrature (relative error of order 1e-11).
    """
    z, rho = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(rho, dtype=float))
    if np.any((rho <= -1.0) | (rho > 1.0)):
        raise ValueError("rho must lie in (-1, 1]")
    shape = z.shape
    z, rho = z.ravel(), rho.ravel()
    exponent = z * z / (1.0 + rho)
    ratio = np.sqrt((1.0 - rho) / (1.0 + rho))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_survival = np.log(stats.norm.sf(z) - 2.0 * special.owens_t(z, ratio))
    anticorrelated = (rho <= 0.0) & (z > 0.0) & (exponent >= _LAGUERRE_MIN_EXPONENT)
    if np.any(anticorrelated):
        level, offset = z[anticorrelated], exponent[anticorrelated]
        shifted = offset[:, np.newaxis] + _LAGUERRE_NODES
        integrand = 1.0 / (shifted * np.sqrt(2.0 * shifted - level[:, np.newaxis] ** 2))
        log_survival[anticorrelated] = (
            np.log(level / (2.0 * np.pi)) - offset + np.log(integrand @ _LAGUERRE_WEIGHTS)
        )
    return log_survival.reshape(shape)


def _joint_normal_survival(z, rho) -> np.ndarray:
    """``P(X > z, Y > z)`` for a standard bivariate normal with correlation ``rho``."""
    return np.exp(_joint_normal_log_survival(z, rho))


def gaussian_copula_chi(u, rho) -> np.ndarray | float:
    """``chi(u; rho) = (1 - 2u + Phi2(z_u, z_u; rho)) / (1 - u)`` for the Gaussian copula."""
    u = np.asarray(u, dtype=float)
    return scalar_or_array(_joint_normal_survival(stats.norm.ppf(u), rho) / (1.0 - u))


def gaussian_copula_chibar(u, rho) -> np.ndarray | float:
    """``chibar(u; rho) = 2 log(1 - u) / log P(U > u, V > u) - 1`` for the Gaussian copula."""
    u = np.asarray(u, dtype=float)
    log_survival = _joint_normal_log_survival(stats.norm.ppf(u), rho)
    with np.errstate(divide="ignore"):
        return scalar_or_array(2.0 * np.log1p(-u) / log_survival - 1.0)


def gaussian_eta(rho) -> np.ndarray | float:
    """Ledford-Tawn coefficient of the Gaussian copula, ``(1 + rho) / 2``."""
    return scalar_or_array((1.0 + np.asarray(rho, dtype=float)) / 2.0)


def spearman_to_pearson(rho_s) -> np.ndarray | float:
    """Pearson correlation of the Gaussian copula with Spearman correlation ``rho_s``."""
    return scalar_or_array(2.0 * np.sin(np.pi * np.asarray(rho_s, dtype=float) / 6.0))
