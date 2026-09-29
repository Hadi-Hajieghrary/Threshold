"""The cascade as a multitype branching process (plan v3 WP0, Theorem 4 of reports/v3/theory_v3.md).

Types are cables.  A primary slack onset of cable j whose re-engagement peaks at ``T`` produces,
independently on every other cable i, a Poisson number of cross-cable offspring with mean
``K_ij(T)``; the mean offspring matrix ``K`` (peak-averaged, ``K_jj = 0``) governs everything
below.  Nothing here is fitted: ``kernel_from_margin_law`` builds ``K`` from the transmission
ratio (a derived or measured matrix) and a measured law of the neighbours' tension margins.

Established facts used (multitype Galton-Watson, Athreya & Ney 1972; Borel-Tanner total
progeny; Hawkes-Oakes cluster representation):
* subcritical iff the spectral radius rho(K) < 1; then the mean total progeny of one primary
  event of type j is column j of (I - K)^-1 summed (rows = children throughout this module),
  and the total onset rate is nu_total = (I - K)^-1 nu_primary;
* the extremal index of the onset process under the declustering that assigns every offspring
  to its primary ancestor is 1 / E[cluster size];
* in the single-type case with mean m < 1 the total progeny is Borel-Tanner(m):
  P(n) = exp(-m n) (m n)^(n-1) / n!, mean 1/(1 - m).
"""

from __future__ import annotations

import math

import numpy as np

from tether.evt._common import as_generator, percentile_interval


# ----------------------------------------------------------------------------- kernels


def transmission_probability(ratio_ij: float, margin_cdf, peaks: np.ndarray, pulse_factor: float = 1.0) -> np.ndarray:
    """P(neighbour i is unloaded | a snap of peak T on j) = margin_cdf(ratio_ij * pulse_factor * T).

    ``margin_cdf`` is the cumulative law of the neighbour's tension margin (its tension above
    zero at a random taut instant); a swing larger than the margin unloads it.
    """
    peaks = np.asarray(peaks, dtype=float)
    return np.clip(margin_cdf(ratio_ij * pulse_factor * peaks), 0.0, 1.0)


def margin_cdf_from_quantiles(quantiles: np.ndarray):
    """Empirical CDF from 101 quantiles (0..100 %) of the margin, linearly interpolated."""
    q = np.asarray(quantiles, dtype=float)
    if q.ndim != 1 or q.size < 2:
        raise ValueError("quantiles must be a 1-d array of at least two levels")
    levels = np.linspace(0.0, 1.0, q.size)

    def cdf(x):
        return np.interp(np.asarray(x, dtype=float), q, levels, left=0.0, right=1.0)

    return cdf


def kernel_from_margin_law(
    ratio: np.ndarray,
    margin_quantiles: np.ndarray,
    peaks: np.ndarray,
    parent_cable: np.ndarray,
    pulse_factor: float = 1.0,
) -> np.ndarray:
    """Peak-averaged mean offspring matrix K_ij = mean over parents on j of P(i unloaded | T).

    ``ratio`` (N, N): neighbour i's swing per unit peak on j (row i, column j); its diagonal is
    ignored.  ``margin_quantiles`` (N, L): per-cable quantiles of the tension margin.
    ``peaks``/``parent_cable``: the parents used for the averaging.  Cables with no parent get
    a column of nan.
    """
    ratio = np.asarray(ratio, dtype=float)
    n = ratio.shape[0]
    peaks = np.asarray(peaks, dtype=float)
    parent_cable = np.asarray(parent_cable, dtype=np.int64)
    cdfs = [margin_cdf_from_quantiles(margin_quantiles[i]) for i in range(n)]
    kernel = np.full((n, n), np.nan)
    for j in range(n):
        rows = parent_cable == j
        if not rows.any():
            continue
        for i in range(n):
            if i == j:
                kernel[i, j] = 0.0
                continue
            kernel[i, j] = float(np.mean(transmission_probability(ratio[i, j], cdfs[i], peaks[rows], pulse_factor)))
    return kernel


def empirical_kernel(parent_cable: np.ndarray, offspring_counts: np.ndarray) -> np.ndarray:
    """K_ij = mean number of offspring on cable i per event on cable j (columns with no event: nan)."""
    parent_cable = np.asarray(parent_cable, dtype=np.int64)
    counts = np.asarray(offspring_counts, dtype=float)
    n = counts.shape[1]
    kernel = np.full((n, n), np.nan)
    for j in range(n):
        rows = parent_cable == j
        if rows.any():
            kernel[:, j] = counts[rows].mean(axis=0)
            kernel[j, j] = 0.0
    return kernel


# ----------------------------------------------------------------------------- criticality, progeny, theta


def spectral_radius(kernel: np.ndarray) -> float:
    kernel = np.asarray(kernel, dtype=float)
    if not np.all(np.isfinite(kernel)):
        raise ValueError("the kernel has undefined entries")
    return float(np.max(np.abs(np.linalg.eigvals(kernel))))


def mean_progeny(kernel: np.ndarray) -> np.ndarray:
    """(I - K)^-1: entry (i, j) is the mean number of type-i descendants (self included) of a type-j primary."""
    kernel = np.asarray(kernel, dtype=float)
    if spectral_radius(kernel) >= 1.0:
        raise ValueError("the cascade is critical or supercritical: rho(K) >= 1")
    return np.linalg.inv(np.eye(kernel.shape[0]) - kernel)


def expected_cluster_size(kernel: np.ndarray, primary_weights: np.ndarray) -> float:
    """Mean total progeny (self included) of a primary event of random type pi.

    Convention: ``K[i, j]`` is the mean number of children of type i per parent of type j
    (rows = children), so the mean descendants of a type-j primary are the COLUMN sums of
    (I - K)^-1 and the cluster size is 1^T (I - K)^-1 pi.  reports/v3/theory_v3.md writes the
    same quantity with rows = parents, pi^T (I - K_theory)^-1 1, K_theory = K^T.
    """
    pi = np.asarray(primary_weights, dtype=float)
    pi = pi / pi.sum()
    return float(mean_progeny(kernel).sum(axis=0) @ pi)


def extremal_index(kernel: np.ndarray, primary_weights: np.ndarray) -> float:
    return 1.0 / expected_cluster_size(kernel, primary_weights)


def mean_offspring(kernel: np.ndarray, primary_weights: np.ndarray) -> float:
    """m_bar = mean number of direct cross-cable offspring of a primary event (pi^T K^T 1)."""
    pi = np.asarray(primary_weights, dtype=float)
    pi = pi / pi.sum()
    return float(pi @ np.asarray(kernel, dtype=float).sum(axis=0))


def total_rate(kernel: np.ndarray, primary_rate: np.ndarray) -> np.ndarray:
    """nu_total = (I - K)^-1 nu_primary (per cable)."""
    return mean_progeny(kernel) @ np.asarray(primary_rate, dtype=float)


def borel_tanner_pmf(m: float, sizes: np.ndarray) -> np.ndarray:
    """Total progeny of a single-type Poisson(m) Galton-Watson tree started by one event."""
    sizes = np.asarray(sizes, dtype=float)
    if not 0.0 <= m < 1.0:
        raise ValueError("Borel-Tanner needs 0 <= m < 1")
    out = np.zeros(sizes.shape)
    for k, n in enumerate(sizes):
        n = int(n)
        if n < 1:
            continue
        if m == 0.0:
            out[k] = 1.0 if n == 1 else 0.0
            continue
        out[k] = math.exp(-m * n + (n - 1) * math.log(m * n) - math.lgamma(n + 1))
    return out


def simulate_cluster_sizes(
    kernel: np.ndarray,
    primary_weights: np.ndarray,
    n: int,
    rng=None,
    max_generations: int = 50,
) -> np.ndarray:
    """Total progeny (self included) of ``n`` primary events under the multitype Poisson process."""
    kernel = np.asarray(kernel, dtype=float)
    generator = as_generator(rng)
    pi = np.asarray(primary_weights, dtype=float)
    pi = pi / pi.sum()
    types = kernel.shape[0]
    sizes = np.zeros(n, dtype=np.int64)
    for index in range(n):
        alive = np.zeros(types, dtype=np.int64)
        alive[generator.choice(types, p=pi)] = 1
        total = 1
        for _ in range(max_generations):
            if not alive.any():
                break
            means = kernel @ alive  # expected offspring per type from this generation
            children = generator.poisson(means)
            total += int(children.sum())
            alive = children
        sizes[index] = total
    return sizes


def bootstrap_spectral_radius(kernels: list, n_boot: int = 2000, confidence: float = 0.95, rng=None) -> tuple:
    """rho of the seed-pooled kernel with a seed-resampling percentile interval.

    ``kernels`` holds one (K, weight) pair per seed, the weight being the seed's number of
    parent events; the pooled kernel is the weighted mean.
    """
    generator = as_generator(rng)
    ks = np.array([np.asarray(k, dtype=float) for k, _ in kernels])
    ws = np.array([float(w) for _, w in kernels])

    def pooled(rows):
        w = ws[rows]
        if w.sum() <= 0.0:
            return math.nan
        k = np.tensordot(w, ks[rows], axes=1) / w.sum()
        return spectral_radius(np.nan_to_num(k)) if np.all(np.isfinite(k)) else math.nan

    indices = generator.integers(0, len(kernels), size=(n_boot, len(kernels)))
    replicates = np.array([pooled(row) for row in indices])
    lower, upper = percentile_interval(replicates, confidence)
    return pooled(np.arange(len(kernels))), lower, upper
