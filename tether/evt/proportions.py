"""Wilson score intervals for proportions and Newcombe intervals for their differences."""

from __future__ import annotations

import numpy as np

from tether.evt._common import two_sided_z


def _check_binomial(successes, trials) -> tuple[int, int]:
    if int(trials) != trials or trials < 1:
        raise ValueError("trials must be a positive integer")
    if int(successes) != successes or not 0 <= successes <= trials:
        raise ValueError("successes must be an integer in [0, trials]")
    return int(successes), int(trials)


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Proportion with its Wilson score interval, as ``(p, lo, hi)``."""
    successes, trials = _check_binomial(successes, trials)
    z = two_sided_z(confidence)
    p = successes / trials
    z2n = z * z / trials
    denominator = 1.0 + z2n
    center = (p + 0.5 * z2n) / denominator
    half_width = z * np.sqrt(p * (1.0 - p) / trials + 0.25 * z2n / trials) / denominator
    return p, float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def newcombe_difference(
    s1: int, n1: int, s2: int, n2: int, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Difference ``p1 - p2`` with Newcombe's hybrid score interval (method 10, 1998)."""
    p1, lo1, hi1 = wilson_interval(s1, n1, confidence)
    p2, lo2, hi2 = wilson_interval(s2, n2, confidence)
    difference = p1 - p2
    below = np.hypot(p1 - lo1, hi2 - p2)
    above = np.hypot(hi1 - p1, p2 - lo2)
    return difference, float(difference - below), float(difference + above)
