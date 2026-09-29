"""Phase 5 monitor metrics on plain arrays (plan P5-T1 to P5-T7, Appendix C).

Declared operational definitions:
- Forecasts are clipped to [1/(2n), 1 - 1/(2n)], n = 2048, before any logit: the Monte Carlo
  cannot resolve a hazard below half its step.
- Ticks within one slack interval are dependent, so every calibration and discrimination
  interval comes from a bootstrap that resamples whole slack intervals (clusters); the
  tick-level Wald and Wilson intervals are reported alongside. The reliability band is the
  exception: a Wilson band at each bin's Kish effective cluster size, since a percentile
  bootstrap collapses to a point when all of a bin's clusters share one outcome.
- Recalibration: logit P(y) = intercept + slope logit(h) jointly, plus the calibration
  intercept "in the large" (slope fixed at 1), the one whose sign says whether an arm
  under-forecasts: the joint intercept is evaluated at h = 1/2, outside the forecast range.
- Top-bin ratios: realized over forecast frequency among the top 10 % and top 30 % of slack
  ticks ranked by forecast; ticks tied at the boundary value share the remaining places
  equally (fractional weight), so pooling order cannot change the ratio.
- Slack ticks whose label is right-censored by the end of the truth series are left out.
- NEES band: chi^2(2K)/K at the family level for the mean of K interval-averaged NEES.
- False alarm: an alarm tick (h > h_crit) with no true upcrossing above 0.5 v_b within H. The
  false-alarm rate is the fraction of slack intervals holding one, over the intervals with at
  least one retained tick; h_crit is the smallest threshold at which it is at most 5 %.
- Lead time: re-engagement time minus the first alarm tick at or before it, over severing
  intervals with an alarm; the median over those, with a seed-bootstrap interval.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special, stats

from tether.evt.calibration import (
    auroc,
    expected_calibration_error,
    logistic_recalibration,
    reliability_bins,
)
from tether.evt.proportions import wilson_interval
from tether.monitor.hazard import DEFAULT_HORIZON, DEFAULT_SAMPLES, HAZARD_STREAM
from tether.monitor.outcomes import (
    BENIGN_FRACTION,
    IntervalOutcomes,
    SlackIntervals,
    TickOutcomes,
)
from tether.theory.excursion import (
    first_upcrossings_exact,
    first_upcrossings_on_grid,
    overconfidence_factor,
    overconfidence_factor_exact,
    rice_dangerous_upcrossings_joint,
    sample_constant_acceleration,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.intp]

FORECAST_CLIP = 0.5 / DEFAULT_SAMPLES
DEFAULT_BOOT = 2000
FALSE_ALARM_RATE = 0.05
TOP_FRACTIONS = (0.1, 0.3)
RICE_STRATA = (0.0, 0.02, 0.05, 0.1, 0.2, 1.0)
RICE_SELECT_STREAM = 1000
RICE_SAMPLE_STREAM = 1001
RICE_GRIDS = (0.02, 0.01, 0.005)
RICE_SAMPLES = 100_000
RICE_BAND = (1.0, 1.25)
RICE_DOMAIN_LIMIT = 0.2
RICE_CONVERGENCE = 0.005
_RECALIBRATION_KEYS = ("intercept", "slope", "intercept_lo", "intercept_hi", "slope_lo", "slope_hi")


def _generator(rng: np.random.Generator | int | Sequence[int] | None) -> np.random.Generator:
    """A Generator from itself or from integer SeedSequence entropy; None means [0]."""
    if isinstance(rng, np.random.Generator):
        return rng
    entropy = [0] if rng is None else [int(rng)] if np.isscalar(rng) else [int(v) for v in rng]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _interval(replicates: ArrayLike, confidence: float) -> tuple[float, float]:
    values = np.asarray(replicates, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    tail = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return float(lower), float(upper)


@dataclass(frozen=True)
class MonitorTable:
    """One arm's slack ticks (n,) and the slack intervals (k,) they index, pooled over seeds."""

    forecast: FloatArray
    outcome: BoolArray
    benign: BoolArray
    interval: IntArray
    time: FloatArray
    interval_seed: IntArray
    interval_vessel: IntArray
    reengaged: BoolArray
    reengagement_time: FloatArray
    reengagement_speed: FloatArray
    severed: BoolArray
    dropped: int
    censored: int

    @property
    def seed(self) -> IntArray:
        return self.interval_seed[self.interval]

    @property
    def n_intervals(self) -> int:
        return int(self.interval_seed.size)


def monitor_table(
    hazard: ArrayLike,
    ticks: TickOutcomes,
    intervals: SlackIntervals,
    outcomes: IntervalOutcomes,
    seed: int = 0,
    benign_fraction: float = BENIGN_FRACTION,
) -> MonitorTable:
    """Slack ticks of one run; ticks with a non-finite forecast (``dropped``) or an unknown,
    right-censored label (``censored``) are left out and counted."""
    forecast = np.asarray(hazard, dtype=float).reshape(intervals.shape)
    index = intervals.tick_index()
    rows, vessels = np.nonzero(index >= 0)
    finite = np.isfinite(forecast[rows, vessels])
    unknown = ticks.censored[rows, vessels]
    keep = finite & ~unknown
    rows, vessels = rows[keep], vessels[keep]
    return MonitorTable(
        forecast=forecast[rows, vessels],
        outcome=ticks.label[rows, vessels],
        benign=ticks.benign(benign_fraction)[rows, vessels],
        interval=index[rows, vessels],
        time=ticks.time[rows],
        interval_seed=np.full(intervals.count, int(seed), dtype=np.intp),
        interval_vessel=intervals.vessel.copy(),
        reengaged=outcomes.reengaged.copy(),
        reengagement_time=outcomes.reengagement_time.copy(),
        reengagement_speed=outcomes.reengagement_speed.copy(),
        severed=outcomes.severed,
        dropped=int(np.count_nonzero(~finite)),
        censored=int(np.count_nonzero(finite & unknown)),
    )


def concatenate_tables(tables: Sequence[MonitorTable]) -> MonitorTable:
    """Pool runs; interval rows are renumbered so they stay unique."""
    offsets = np.cumsum([0] + [table.n_intervals for table in tables[:-1]])
    return MonitorTable(
        forecast=np.concatenate([table.forecast for table in tables]),
        outcome=np.concatenate([table.outcome for table in tables]),
        benign=np.concatenate([table.benign for table in tables]),
        interval=np.concatenate([table.interval + o for table, o in zip(tables, offsets)]),
        time=np.concatenate([table.time for table in tables]),
        interval_seed=np.concatenate([table.interval_seed for table in tables]),
        interval_vessel=np.concatenate([table.interval_vessel for table in tables]),
        reengaged=np.concatenate([table.reengaged for table in tables]),
        reengagement_time=np.concatenate([table.reengagement_time for table in tables]),
        reengagement_speed=np.concatenate([table.reengagement_speed for table in tables]),
        severed=np.concatenate([table.severed for table in tables]),
        dropped=int(sum(table.dropped for table in tables)),
        censored=int(sum(table.censored for table in tables)),
    )


def tick_nees(
    e_hat: ArrayLike,
    edot_hat: ArrayLike,
    sigma: ArrayLike,
    e_true: ArrayLike,
    edot_true: ArrayLike,
) -> FloatArray:
    """Precursor NEES of (e, e') at every tick; NaN where Sigma is not positive definite."""
    first = np.asarray(e_hat, dtype=float) - np.asarray(e_true, dtype=float)
    second = np.asarray(edot_hat, dtype=float) - np.asarray(edot_true, dtype=float)
    matrix = np.asarray(sigma, dtype=float)
    s00, s11 = matrix[..., 0, 0], matrix[..., 1, 1]
    s01 = 0.5 * (matrix[..., 0, 1] + matrix[..., 1, 0])
    determinant = s00 * s11 - s01 * s01
    valid = (determinant > 0.0) & (s00 > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (s11 * first**2 - 2.0 * s01 * first * second + s00 * second**2) / determinant
    return np.where(valid, value, np.nan)


def interval_nees(nees: ArrayLike, intervals: SlackIntervals) -> FloatArray:
    """Time average of the tick NEES over each slack interval (NaN if none is finite)."""
    values = np.asarray(nees, dtype=float).reshape(intervals.shape)
    average = np.full(intervals.count, np.nan)
    spans = zip(intervals.vessel, intervals.first, intervals.last)
    for row, (vessel, first, last) in enumerate(spans):
        segment = values[first : last + 1, vessel]
        segment = segment[np.isfinite(segment)]
        if segment.size:
            average[row] = float(segment.mean())
    return average


@dataclass(frozen=True)
class NeesSummary:
    mean: float
    lo: float
    hi: float
    band_lo: float
    band_hi: float
    n_intervals: int
    n_seeds: int
    dim: int

    @property
    def verdict(self) -> str:
        """'inside', 'above' or 'below' the chi-square band."""
        if self.mean > self.band_hi:
            return "above"
        return "below" if self.mean < self.band_lo else "inside"


def nees_summary(
    values: ArrayLike,
    seeds: ArrayLike,
    dim: int = 2,
    confidence: float = 0.95,
    n_boot: int = DEFAULT_BOOT,
    rng: np.random.Generator | int | Sequence[int] | None = None,
) -> NeesSummary:
    """Mean interval NEES with a seed-bootstrap interval and the chi^2(dim K)/K band."""
    nees = np.asarray(values, dtype=float).ravel()
    labels = np.asarray(seeds).ravel()
    if labels.shape != nees.shape:
        raise ValueError("one seed label per interval")
    keep = np.isfinite(nees)
    nees, labels = nees[keep], labels[keep]
    if nees.size == 0:
        raise ValueError("no finite interval NEES")
    unique, owner = np.unique(labels, return_inverse=True)
    sums = np.bincount(owner, weights=nees, minlength=unique.size)
    counts = np.bincount(owner, minlength=unique.size).astype(float)
    generator = _generator(rng)
    replicates = np.empty(n_boot)
    for index in range(n_boot):
        weight = np.bincount(generator.integers(0, unique.size, unique.size), minlength=unique.size)
        replicates[index] = float(weight @ sums) / float(weight @ counts)
    lower, upper = _interval(replicates, confidence)
    tail = 0.5 * (1.0 - confidence)
    band = stats.chi2.ppf([tail, 1.0 - tail], dim * nees.size) / nees.size
    return NeesSummary(
        mean=float(nees.mean()),
        lo=lower,
        hi=upper,
        band_lo=float(band[0]),
        band_hi=float(band[1]),
        n_intervals=int(nees.size),
        n_seeds=int(unique.size),
        dim=int(dim),
    )


def _weighted_loglik(design: FloatArray, y: FloatArray, w: FloatArray, beta: FloatArray) -> float:
    eta = design @ beta
    return float(np.sum(w * (y * eta - np.logaddexp(0.0, eta))))


def _weighted_logistic(
    logit: FloatArray, y: FloatArray, w: FloatArray, start: FloatArray
) -> FloatArray:
    """Weighted Newton-IRLS for (intercept, slope) with step halving; NaN if separated."""
    if _separated(logit, y, w):
        return np.full(2, np.nan)
    design = np.column_stack([np.ones_like(logit), logit])
    beta = np.asarray(start, dtype=float).copy()
    loglik = _weighted_loglik(design, y, w, beta)
    for _ in range(100):
        mean = special.expit(design @ beta)
        information = design.T @ (design * (w * mean * (1.0 - mean))[:, None])
        try:
            step = np.linalg.solve(information, design.T @ (w * (y - mean)))
        except np.linalg.LinAlgError:
            return np.full(2, np.nan)
        fraction = 1.0
        while True:
            candidate = beta + fraction * step
            value = _weighted_loglik(design, y, w, candidate)
            if value >= loglik - 1e-12 * abs(loglik) or fraction < 1e-10:
                break
            fraction *= 0.5
        beta, loglik = candidate, value
        if np.max(np.abs(fraction * step)) < 1e-10:
            break
    return beta


def _large_intercept(logit: FloatArray, y: FloatArray, w: FloatArray) -> float:
    """Calibration in the large: alpha of logit P(y) = alpha + logit(h), by Newton."""
    if not (np.sum(w * y) > 0.0 and np.sum(w * (1.0 - y)) > 0.0):
        return float("nan")
    design = np.column_stack([np.ones_like(logit), logit])
    alpha = 0.0
    loglik = _weighted_loglik(design, y, w, np.array([alpha, 1.0]))
    for _ in range(100):
        mean = special.expit(alpha + logit)
        step = float(np.sum(w * (y - mean)) / np.sum(w * mean * (1.0 - mean)))
        fraction = 1.0
        while True:
            candidate = alpha + fraction * step
            value = _weighted_loglik(design, y, w, np.array([candidate, 1.0]))
            if value >= loglik - 1e-12 * abs(loglik) or fraction < 1e-10:
                break
            fraction *= 0.5
        alpha, loglik = candidate, value
        if abs(fraction * step) < 1e-12:
            break
    return float(alpha)


class _WeightedAuroc:
    """Mann-Whitney AUROC under tick weights, ties counted 1/2, sorting done once."""

    def __init__(self, scores: FloatArray, y: FloatArray) -> None:
        self.positive = np.flatnonzero(y == 1.0)
        negative = np.flatnonzero(y == 0.0)
        self.negative = negative[np.argsort(scores[negative], kind="stable")]
        ordered = scores[self.negative]
        self.below = np.searchsorted(ordered, scores[self.positive], side="left")
        self.at_or_below = np.searchsorted(ordered, scores[self.positive], side="right")

    def __call__(self, w: FloatArray) -> float:
        cumulative = np.concatenate(([0.0], np.cumsum(w[self.negative])))
        below = cumulative[self.below]
        wins = below + 0.5 * (cumulative[self.at_or_below] - below)
        positive = w[self.positive]
        denominator = float(positive.sum() * cumulative[-1])
        return float(positive @ wins) / denominator if denominator > 0.0 else float("nan")


def _top_weights(forecast: FloatArray, fraction: float) -> FloatArray:
    """Membership of the top ceil(fraction n) forecasts; ties at the boundary value share the
    remaining places equally, so the set does not depend on the tick order."""
    count = int(np.ceil(fraction * forecast.size))
    weights = np.zeros(forecast.size)
    if count == 0:
        return weights
    boundary = np.partition(forecast, forecast.size - count)[forecast.size - count]
    above = forecast > boundary
    tied = forecast == boundary
    weights[above] = 1.0
    weights[tied] = (count - np.count_nonzero(above)) / np.count_nonzero(tied)
    return weights


def _effective_wilson(
    observed: float, effective: float, confidence: float
) -> tuple[float, float]:
    """Wilson score interval of a proportion with a possibly non-integer effective size."""
    z = float(stats.norm.ppf(0.5 + 0.5 * confidence))
    ratio = z * z / effective
    centre = (observed + 0.5 * ratio) / (1.0 + ratio)
    half = z * np.sqrt(observed * (1.0 - observed) / effective + 0.25 * ratio / effective)
    half /= 1.0 + ratio
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _ratio(y: FloatArray, p: FloatArray, w: FloatArray, top: FloatArray) -> float:
    forecast = float(np.sum(w * top * p))
    return float(np.sum(w * top * y)) / forecast if forecast > 0.0 else float("nan")


def _separated(logit: FloatArray, y: FloatArray, w: FloatArray) -> bool:
    """Complete or quasi-complete separation of the weighted classes on logit(h) (a constant
    forecast included): the joint recalibration then has no finite maximum-likelihood fit."""
    active = w > 0.0
    events, others = logit[active & (y == 1.0)], logit[active & (y == 0.0)]
    if events.size == 0 or others.size == 0:
        return True
    return bool(events.min() >= others.max() or events.max() <= others.min())


@dataclass(frozen=True)
class CalibrationReport:
    """Recalibration, ECE, reliability, AUROC and top-bin ratios, with cluster intervals."""

    n_ticks: int
    n_clusters: int
    n_events: int
    base_rate: float
    mean_forecast: float
    intercept: float
    intercept_lo: float
    intercept_hi: float
    slope: float
    slope_lo: float
    slope_hi: float
    wald_intercept_lo: float
    wald_intercept_hi: float
    wald_slope_lo: float
    wald_slope_hi: float
    large_intercept: float
    large_intercept_lo: float
    large_intercept_hi: float
    ece: float
    ece_lo: float
    ece_hi: float
    auroc: float
    auroc_lo: float
    auroc_hi: float
    top_bin_forecast: float
    top_bin_observed: float
    top_bin_ratio: float
    top_bin_ratio_lo: float
    top_bin_ratio_hi: float
    top_three_forecast: float
    top_three_observed: float
    top_three_ratio: float
    top_three_ratio_lo: float
    top_three_ratio_hi: float
    bins: tuple[dict[str, float | int | bool], ...]
    n_outside: int
    n_outside_boot: int
    n_outside_wilson: int
    clip: float
    n_boot: int
    confidence: float
    separated: bool


def calibration_report(
    forecast: ArrayLike,
    outcome: ArrayLike,
    cluster: ArrayLike,
    n_bins: int = 10,
    n_boot: int = DEFAULT_BOOT,
    clip: float = FORECAST_CLIP,
    confidence: float = 0.95,
    family_confidence: float = 0.95,
    top_fractions: tuple[float, float] = TOP_FRACTIONS,
    rng: np.random.Generator | int | Sequence[int] | None = None,
) -> CalibrationReport:
    """Appendix C calibration of forecasts against 0/1 outcomes, resampling whole clusters.

    Reliability uses ``n_bins`` equal-width bins, each judged at level
    1 - (1 - family_confidence) / n_occupied (Bonferroni). The simultaneous band (``lo``,
    ``hi``, ``inside``, ``n_outside``) is the Wilson interval of the bin's observed frequency at
    Kish's effective size n^2 / sum(n_c^2) over the bin's clusters: exact when a cluster's
    ticks share one outcome (a label is shared by every tick within H of one re-engagement)
    and conservative for any non-negative within-cluster correlation. The cluster-bootstrap
    percentile band (``boot_*``), degenerate when all of a bin's clusters agree, and evt's
    tick-level Wilson band (``wilson_*``) are reported beside it. When the outcomes are
    separated on logit(h) (``separated``) the joint fit has no finite estimate and its
    intercept, slope and intervals are NaN; separated replicates are left out likewise.
    """
    p = np.asarray(forecast, dtype=float).ravel()
    y = np.asarray(outcome, dtype=float).ravel()
    groups = np.asarray(cluster).ravel()
    if groups.shape != p.shape:
        raise ValueError("one cluster label per forecast")
    _, owner = np.unique(groups, return_inverse=True)
    clusters = int(owner.max()) + 1 if owner.size else 0
    clipped = np.clip(p, clip, 1.0 - clip)
    logit = special.logit(clipped)
    ones = np.ones_like(p)
    separated = _separated(logit, y, ones)
    if separated:
        joint = dict.fromkeys(_RECALIBRATION_KEYS, float("nan"))
    else:
        joint = logistic_recalibration(clipped, y, confidence)
    point = np.array([joint["intercept"], joint["slope"]])
    wilson = reliability_bins(p, y, n_bins, family_confidence)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    occupied = np.flatnonzero(np.bincount(bins, minlength=n_bins))
    tops = [_top_weights(p, fraction) for fraction in top_fractions]
    area = _WeightedAuroc(p, y)
    generator = _generator(rng)
    replicates = np.full((n_boot, 7 + occupied.size), np.nan)
    for index in range(n_boot):
        weight = np.bincount(generator.integers(0, clusters, clusters), minlength=clusters)[owner]
        weight = weight.astype(float)
        forecast_sum = np.bincount(bins, weights=weight * p, minlength=n_bins)
        outcome_sum = np.bincount(bins, weights=weight * y, minlength=n_bins)
        count = np.bincount(bins, weights=weight, minlength=n_bins)
        replicates[index, :2] = _weighted_logistic(logit, y, weight, point)
        replicates[index, 2] = _large_intercept(logit, y, weight)
        replicates[index, 3] = np.sum(np.abs(forecast_sum - outcome_sum)) / np.sum(weight)
        replicates[index, 4] = area(weight)
        replicates[index, 5] = _ratio(y, p, weight, tops[0])
        replicates[index, 6] = _ratio(y, p, weight, tops[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            replicates[index, 7:] = outcome_sum[occupied] / count[occupied]
    intervals = [_interval(replicates[:, column], confidence) for column in range(7)]
    family = 1.0 - (1.0 - family_confidence) / max(occupied.size, 1)
    entries = []
    for column, entry in enumerate(wilson):
        boot_lo, boot_hi = _interval(replicates[:, 7 + column], family)
        members = owner[bins == occupied[column]]
        sizes = np.bincount(members)
        sizes = sizes[sizes > 0].astype(float)
        effective = float(sizes.sum() ** 2 / np.sum(sizes**2))
        lower, upper = _effective_wilson(entry["observed"], effective, family)
        entries.append(
            {
                "bin_lo": entry["bin_lo"],
                "bin_hi": entry["bin_hi"],
                "count": entry["count"],
                "clusters": int(sizes.size),
                "effective_count": effective,
                "mean_forecast": entry["mean_forecast"],
                "observed": entry["observed"],
                "lo": lower,
                "hi": upper,
                "inside": bool(lower <= entry["mean_forecast"] <= upper),
                "boot_lo": boot_lo,
                "boot_hi": boot_hi,
                "boot_inside": bool(boot_lo <= entry["mean_forecast"] <= boot_hi),
                "wilson_lo": entry["lo"],
                "wilson_hi": entry["hi"],
                "wilson_inside": entry["inside"],
            }
        )
    return CalibrationReport(
        n_ticks=int(p.size),
        n_clusters=clusters,
        n_events=int(y.sum()),
        base_rate=float(y.mean()),
        mean_forecast=float(p.mean()),
        intercept=joint["intercept"],
        intercept_lo=intervals[0][0],
        intercept_hi=intervals[0][1],
        slope=joint["slope"],
        slope_lo=intervals[1][0],
        slope_hi=intervals[1][1],
        wald_intercept_lo=joint["intercept_lo"],
        wald_intercept_hi=joint["intercept_hi"],
        wald_slope_lo=joint["slope_lo"],
        wald_slope_hi=joint["slope_hi"],
        large_intercept=_large_intercept(logit, y, ones),
        large_intercept_lo=intervals[2][0],
        large_intercept_hi=intervals[2][1],
        ece=expected_calibration_error(p, y, n_bins),
        ece_lo=intervals[3][0],
        ece_hi=intervals[3][1],
        auroc=auroc(p, y),
        auroc_lo=intervals[4][0],
        auroc_hi=intervals[4][1],
        top_bin_forecast=float(np.average(p, weights=tops[0])),
        top_bin_observed=float(np.average(y, weights=tops[0])),
        top_bin_ratio=_ratio(y, p, ones, tops[0]),
        top_bin_ratio_lo=intervals[5][0],
        top_bin_ratio_hi=intervals[5][1],
        top_three_forecast=float(np.average(p, weights=tops[1])),
        top_three_observed=float(np.average(y, weights=tops[1])),
        top_three_ratio=_ratio(y, p, ones, tops[1]),
        top_three_ratio_lo=intervals[6][0],
        top_three_ratio_hi=intervals[6][1],
        bins=tuple(entries),
        n_outside=int(sum(not entry["inside"] for entry in entries)),
        n_outside_boot=int(sum(not entry["boot_inside"] for entry in entries)),
        n_outside_wilson=int(sum(not entry["wilson_inside"] for entry in entries)),
        clip=float(clip),
        n_boot=int(n_boot),
        confidence=float(confidence),
        separated=separated,
    )


def closing_speed_moments(
    e: ArrayLike, edot: ArrayLike, sigma: ArrayLike, a_hat: ArrayLike, sigma_a: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Mean-path closing speed sqrt(e'^2 - 2 a e) of the IV.9 model and its reported standard
    deviation to first order in (e0, v0, a); NaN where the mean path has no upcrossing."""
    e0, v0 = np.asarray(e, dtype=float), np.asarray(edot, dtype=float)
    a, spread = np.asarray(a_hat, dtype=float), np.asarray(sigma_a, dtype=float)
    matrix = np.asarray(sigma, dtype=float)
    discriminant = v0**2 - 2.0 * a * e0
    speed = np.sqrt(np.where(discriminant > 0.0, discriminant, np.nan))
    g0, g1, g2 = -a / speed, v0 / speed, -e0 / speed
    s01 = 0.5 * (matrix[..., 0, 1] + matrix[..., 1, 0])
    variance = g0**2 * matrix[..., 0, 0] + 2.0 * g0 * g1 * s01 + g1**2 * matrix[..., 1, 1]
    return speed, np.sqrt(variance + g2**2 * spread**2)


@dataclass(frozen=True)
class OverconfidencePrediction:
    """Prop 12 factor by which a posterior of NEES ``nees`` under-states Pr(e' > v_b)."""

    nees: float
    variance_ratio: float
    leading: float
    exact: float

    def agrees(self, measured_ratio: float, factor: float = 2.0) -> bool:
        """P5-T7: the measured ratio within ``factor`` of the leading-order prediction."""
        return bool(self.leading / factor <= measured_ratio <= self.leading * factor)


def overconfidence_prediction(
    nees: float,
    v_b: ArrayLike,
    reported_sd: ArrayLike,
    mean: ArrayLike = 0.0,
    dim: int = 2,
    weights: ArrayLike | None = None,
) -> OverconfidencePrediction:
    """Prop 12 at the measured NEES: a = NEES / dim; the true spread is sqrt(a) times the
    reported one. Arrays are averaged with ``weights`` (e.g. the forecasts of a bin). The
    leading order is a tail asymptotic: it is NaN when any weighted entry has v_b at or below
    its mean, where overconfidence over-states the exceedance instead (``exact`` < 1)."""
    ratio = float(nees) / dim
    spread = np.sqrt(ratio) * np.asarray(reported_sd, dtype=float)
    distance = np.asarray(v_b, dtype=float) - np.asarray(mean, dtype=float)
    leading = np.asarray(overconfidence_factor(ratio, distance, spread), dtype=float)
    leading = np.where(distance > 0.0, leading, np.nan)
    exact = np.asarray(overconfidence_factor_exact(ratio, distance, spread), dtype=float)
    shape = np.broadcast_shapes(leading.shape, exact.shape)
    weight = np.ones(shape) if weights is None else np.broadcast_to(np.asarray(weights), shape)
    used = weight > 0.0
    return OverconfidencePrediction(
        nees=float(nees),
        variance_ratio=ratio,
        leading=float(np.average(np.broadcast_to(leading, shape)[used], weights=weight[used])),
        exact=float(np.average(np.broadcast_to(exact, shape)[used], weights=weight[used])),
    )


def _interval_worst(
    forecast: FloatArray, interval: IntArray, mask: BoolArray, n_intervals: int
) -> FloatArray:
    worst = np.full(n_intervals, -np.inf)
    np.maximum.at(worst, interval[mask], forecast[mask])
    return worst


def _alarm_inputs(
    forecast: ArrayLike, benign: ArrayLike, interval: ArrayLike, n_intervals: int | None
) -> tuple[FloatArray, BoolArray, IntArray, int]:
    """Without ``n_intervals`` the population is the intervals holding at least one tick, so
    an interval that lost every tick (censored or dropped) never enters the denominator."""
    values = np.asarray(forecast, dtype=float).ravel()
    quiet = np.asarray(benign, dtype=bool).ravel()
    rows = np.asarray(interval, dtype=np.intp).ravel()
    if not values.shape == quiet.shape == rows.shape:
        raise ValueError("forecast, benign and interval must align")
    if n_intervals is not None:
        return values, quiet, rows, int(n_intervals)
    present, compact = np.unique(rows, return_inverse=True)
    return values, quiet, compact.astype(np.intp), int(present.size)


def false_alarm_rate(
    forecast: ArrayLike,
    benign: ArrayLike,
    interval: ArrayLike,
    h_crit: float,
    n_intervals: int | None = None,
) -> float:
    """Fraction of slack intervals holding a false-alarm tick at threshold ``h_crit``."""
    values, quiet, rows, count = _alarm_inputs(forecast, benign, interval, n_intervals)
    return float(np.mean(_interval_worst(values, rows, quiet, count) > h_crit))


def false_alarm_threshold(
    forecast: ArrayLike,
    benign: ArrayLike,
    interval: ArrayLike,
    rate: float = FALSE_ALARM_RATE,
    n_intervals: int | None = None,
) -> float:
    """h_crit: the smallest threshold whose false-alarm rate is at most ``rate``."""
    values, quiet, rows, count = _alarm_inputs(forecast, benign, interval, n_intervals)
    allowed = int(np.floor(rate * count + 1e-9))
    if allowed >= count:
        return 0.0
    worst = np.sort(_interval_worst(values, rows, quiet, count))[::-1]
    return float(max(worst[allowed], 0.0))


def lead_times(
    forecast: ArrayLike,
    time: ArrayLike,
    interval: ArrayLike,
    reengagement_time: ArrayLike,
    positive: ArrayLike,
    h_crit: float,
) -> FloatArray:
    """Per interval: re-engagement time minus the first alarm tick at or before it; NaN for
    intervals that are not ``positive`` or raise no such alarm."""
    values = np.asarray(forecast, dtype=float).ravel()
    ticks = np.asarray(time, dtype=float).ravel()
    rows = np.asarray(interval, dtype=np.intp).ravel()
    arrival = np.asarray(reengagement_time, dtype=float).ravel()
    target = np.asarray(positive, dtype=bool).ravel()
    eligible = (values > h_crit) & (ticks <= arrival[rows])
    first = np.full(arrival.size, np.inf)
    np.minimum.at(first, rows[eligible], ticks[eligible])
    return np.where(target & np.isfinite(first), arrival - first, np.nan)


@dataclass(frozen=True)
class LeadTimeSummary:
    n_positive: int
    n_detected: int
    detection_fraction: float
    median: float
    lo: float
    hi: float


def _seed_weights(seeds: ArrayLike, n_boot: int, rng) -> list[IntArray]:
    """Per-interval multiplicities of ``n_boot`` seed-bootstrap replicates."""
    unique, owner = np.unique(np.asarray(seeds).ravel(), return_inverse=True)
    generator = _generator(rng)
    return [
        np.bincount(generator.integers(0, unique.size, unique.size), minlength=unique.size)[owner]
        for _ in range(n_boot)
    ]


def _median(values: FloatArray, weight: IntArray | None = None) -> float:
    kept = values if weight is None else np.repeat(values, weight)
    return float(np.median(kept)) if kept.size else float("nan")


def lead_time_summary(
    leads: ArrayLike,
    positive: ArrayLike,
    seeds: ArrayLike,
    confidence: float = 0.95,
    n_boot: int = DEFAULT_BOOT,
    rng: np.random.Generator | int | Sequence[int] | None = None,
) -> LeadTimeSummary:
    """Median lead over detected positive intervals, with a seed-bootstrap interval."""
    values = np.asarray(leads, dtype=float).ravel()
    target = np.asarray(positive, dtype=bool).ravel()
    detected = target & np.isfinite(values)
    draws = _seed_weights(seeds, n_boot, rng)
    replicates = [_median(values[detected], weight[detected]) for weight in draws]
    lower, upper = _interval(replicates, confidence)
    count = int(target.sum())
    return LeadTimeSummary(
        n_positive=count,
        n_detected=int(detected.sum()),
        detection_fraction=float(detected.sum() / count) if count else float("nan"),
        median=_median(values[detected]),
        lo=lower,
        hi=upper,
    )


def paired_lead_difference(
    leads_a: ArrayLike,
    leads_b: ArrayLike,
    positive: ArrayLike,
    seeds: ArrayLike,
    confidence: float = 0.95,
    n_boot: int = DEFAULT_BOOT,
    rng: np.random.Generator | int | Sequence[int] | None = None,
) -> tuple[float, float, float]:
    """median(lead a) - median(lead b) on the same intervals, seeds resampled jointly."""
    first = np.asarray(leads_a, dtype=float).ravel()
    second = np.asarray(leads_b, dtype=float).ravel()
    target = np.asarray(positive, dtype=bool).ravel()
    keep_a, keep_b = target & np.isfinite(first), target & np.isfinite(second)
    draws = _seed_weights(seeds, n_boot, rng)
    replicates = [
        _median(first[keep_a], weight[keep_a]) - _median(second[keep_b], weight[keep_b])
        for weight in draws
    ]
    lower, upper = _interval(replicates, confidence)
    return _median(first[keep_a]) - _median(second[keep_b]), lower, upper


def stratified_selection(
    hazard: ArrayLike,
    master_seed: int,
    n_select: int = 200,
    edges: Sequence[float] = RICE_STRATA,
) -> tuple[IntArray, IntArray]:
    """P5-T1 state selection: equal allocation over the (lo, hi] strata of ``hazard``, a
    stratum's shortfall passed to the others; drawn without replacement with
    ``SeedSequence([master, 17, 1000])``. Returns sorted indices and their strata."""
    values = np.asarray(hazard, dtype=float).ravel()
    bounds = np.asarray(edges, dtype=float)
    stratum = np.where(np.isfinite(values), np.searchsorted(bounds, values, side="left") - 1, -1)
    stratum = np.where(stratum < bounds.size - 1, stratum, -1)
    members = [np.flatnonzero(stratum == level) for level in range(bounds.size - 1)]
    capacity = [member.size for member in members]
    allocation = [0] * len(members)
    remaining = min(int(n_select), int(sum(capacity)))
    open_levels = [level for level, size in enumerate(capacity) if size > 0]
    while remaining > 0 and open_levels:
        share, extra = divmod(remaining, len(open_levels))
        for order, level in enumerate(open_levels):
            give = min(capacity[level] - allocation[level], share + (1 if order < extra else 0))
            allocation[level] += give
            remaining -= give
        open_levels = [level for level in open_levels if allocation[level] < capacity[level]]
    generator = np.random.default_rng(
        np.random.SeedSequence([int(master_seed), HAZARD_STREAM, RICE_SELECT_STREAM])
    )
    chosen = [
        generator.choice(member, size=size, replace=False)
        for member, size in zip(members, allocation)
    ]
    index = np.sort(np.concatenate(chosen)).astype(np.intp)
    return index, stratum[index].astype(np.intp)


@dataclass(frozen=True)
class RiceStudy:
    """P5-T1: grid Monte Carlo, exact-root Monte Carlo (same draws) and Rice, per state."""

    state_id: IntArray
    grids: tuple[float, ...]
    h_mc: FloatArray
    h_exact: FloatArray
    h_rice: FloatArray
    mc_lo: FloatArray
    mc_hi: FloatArray
    ratio: FloatArray
    ratio_lo: FloatArray
    ratio_hi: FloatArray
    consistent: BoolArray
    n_samples: int
    convergence: float
    converged: bool
    domain_count: int
    domain_fraction: float
    passed: bool
    rice_failures: int


def rice_study(
    means: ArrayLike,
    covariances: ArrayLike,
    v_b: ArrayLike,
    master_seed: int,
    state_ids: ArrayLike | None = None,
    n_samples: int = RICE_SAMPLES,
    grids: Sequence[float] = RICE_GRIDS,
    horizon: float = DEFAULT_HORIZON,
    band: tuple[float, float] = RICE_BAND,
    domain_limit: float = RICE_DOMAIN_LIMIT,
    tolerance: float = RICE_CONVERGENCE,
    confidence: float = 0.95,
    required_fraction: float = 0.9,
) -> RiceStudy:
    """h_MC on each grid from one set of ``n_samples`` draws per state (stream
    ``[master, 17, 1001, state_id]``), and h_Rice. Convergence is the largest
    |h(coarsest) - h(finest)|; the ratio h_Rice / h_MC(finest) carries the Wilson interval of
    the finest-grid count, and a state is consistent when that interval meets ``band``. The
    domain is 0 < h_MC(finest) < ``domain_limit``; a state whose Rice quadrature fails stays in
    it and counts as inconsistent. ``h_exact`` locates the crossing by the quadratic's root on
    the same draws, the dt -> 0 limit of the grids."""
    centres = np.asarray(means, dtype=float).reshape(-1, 3)
    spreads = np.asarray(covariances, dtype=float).reshape(-1, 3, 3)
    speeds = np.broadcast_to(np.asarray(v_b, dtype=float), (centres.shape[0],))
    ids = np.arange(centres.shape[0]) if state_ids is None else np.asarray(state_ids).ravel()
    steps = tuple(float(step) for step in grids)
    coarsest, finest = int(np.argmax(steps)), int(np.argmin(steps))
    h_mc = np.zeros((centres.shape[0], len(steps)))
    h_exact = np.zeros(centres.shape[0])
    h_rice = np.full(centres.shape[0], np.nan)
    bounds = np.zeros((centres.shape[0], 2))
    for row, (centre, spread, speed, state) in enumerate(zip(centres, spreads, speeds, ids)):
        entropy = [int(master_seed), HAZARD_STREAM, RICE_SAMPLE_STREAM, int(state)]
        generator = np.random.default_rng(np.random.SeedSequence(entropy))
        samples = sample_constant_acceleration(centre, spread, n_samples, generator)
        for column, step in enumerate(steps):
            crossings = first_upcrossings_on_grid(samples, horizon, step)
            h_mc[row, column] = np.mean(crossings.dangerous(speed))
        h_exact[row] = np.mean(first_upcrossings_exact(samples, horizon).dangerous(speed))
        count = int(round(h_mc[row, finest] * n_samples))
        bounds[row] = wilson_interval(count, int(n_samples), confidence)[1:]
        try:
            h_rice[row] = rice_dangerous_upcrossings_joint(centre, spread, horizon, speed)
        except ValueError:
            pass
    reference = h_mc[:, finest]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(reference > 0.0, h_rice / reference, np.nan)
        ratio_lo = np.where(reference > 0.0, h_rice / bounds[:, 1], np.nan)
        ratio_hi = np.where(bounds[:, 0] > 0.0, h_rice / bounds[:, 0], np.inf)
    ratio_hi = np.where(reference > 0.0, ratio_hi, np.nan)
    consistent = (ratio_lo <= band[1]) & (ratio_hi >= band[0])
    domain = (reference > 0.0) & (reference < domain_limit)
    fraction = float(consistent[domain].mean()) if domain.any() else float("nan")
    difference = np.abs(h_mc[:, coarsest] - h_mc[:, finest])
    convergence = float(difference.max()) if difference.size else 0.0
    return RiceStudy(
        state_id=ids.astype(np.intp),
        grids=steps,
        h_mc=h_mc,
        h_exact=h_exact,
        h_rice=h_rice,
        mc_lo=bounds[:, 0],
        mc_hi=bounds[:, 1],
        ratio=ratio,
        ratio_lo=ratio_lo,
        ratio_hi=ratio_hi,
        consistent=consistent,
        n_samples=int(n_samples),
        convergence=convergence,
        converged=convergence < tolerance,
        domain_count=int(domain.sum()),
        domain_fraction=fraction,
        passed=bool(convergence < tolerance and domain.any() and fraction >= required_fraction),
        rice_failures=int(np.count_nonzero(~np.isfinite(h_rice))),
    )
