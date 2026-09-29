"""Tabulated engagement law: the v_b = Z^-1(Tb) pivot of the Phase 1 outcome matrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize
from scipy.interpolate import PchipInterpolator

DEFAULT_FIT_SPEED_FLOOR = 1.0


def _result(value: ArrayLike) -> float | NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    return float(array) if array.ndim == 0 else array


@dataclass(frozen=True)
class TabulatedImpact:
    """Peak engagement tension as a function of closing speed.

    Monotone PCHIP through the table knots. Above the largest speed: the line leaving the
    last knot with slope ``tail_slope``. Below the smallest speed: the line toward
    (0, ``zero_speed_peak``), or toward the origin when no zero-speed peak is supplied. The
    knots may include speed 0, in which case ``zero_speed_peak`` must be omitted.
    """

    speeds: tuple[float, ...]
    peaks: tuple[float, ...]
    tail_slope: float
    zero_speed_peak: float | None = None
    fit_speed_floor: float = DEFAULT_FIT_SPEED_FLOOR
    _interpolant: PchipInterpolator | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        speeds = tuple(float(value) for value in self.speeds)
        peaks = tuple(float(value) for value in self.peaks)
        if not speeds or len(speeds) != len(peaks):
            raise ValueError("speeds and peaks must be nonempty and of equal length")
        if not np.all(np.isfinite(speeds + peaks)) or not np.isfinite(self.tail_slope):
            raise ValueError("table entries must be finite")
        if speeds[0] < 0.0 or np.any(np.diff(speeds) <= 0.0):
            raise ValueError("speeds must be nonnegative and strictly increasing")
        if self.zero_speed_peak is not None and speeds[0] == 0.0:
            raise ValueError("a zero-speed knot and zero_speed_peak are mutually exclusive")
        object.__setattr__(self, "speeds", speeds)
        object.__setattr__(self, "peaks", peaks)
        object.__setattr__(self, "tail_slope", float(self.tail_slope))
        object.__setattr__(self, "fit_speed_floor", float(self.fit_speed_floor))
        if self.zero_speed_peak is not None:
            object.__setattr__(self, "zero_speed_peak", float(self.zero_speed_peak))
        interpolant = PchipInterpolator(speeds, peaks) if len(speeds) > 1 else None
        object.__setattr__(self, "_interpolant", interpolant)

    @classmethod
    def from_measurements(
        cls,
        return_speeds: ArrayLike,
        peak_tensions: ArrayLike,
        zero_speed_peak: float | None = None,
        fit_speed_floor: float = DEFAULT_FIT_SPEED_FLOOR,
        tail_slope: float | None = None,
    ) -> TabulatedImpact:
        """Sort the (speed, peak) pairs and average duplicate speeds.

        Unless given, ``tail_slope`` is the through-origin least-squares slope of the measured
        pairs with speed >= ``fit_speed_floor``.
        """
        speeds = np.asarray(return_speeds, dtype=float).ravel()
        peaks = np.asarray(peak_tensions, dtype=float).ravel()
        if speeds.size == 0 or speeds.shape != peaks.shape:
            raise ValueError("need matching, nonempty speed and peak arrays")
        knots, inverse = np.unique(speeds, return_inverse=True)
        averaged = np.bincount(inverse, weights=peaks) / np.bincount(inverse)
        if tail_slope is None:
            fast = speeds >= fit_speed_floor
            if not np.any(fast):
                raise ValueError("no measured speed reaches fit_speed_floor; supply tail_slope")
            tail_slope = float(np.sum(speeds[fast] * peaks[fast]) / np.sum(speeds[fast] ** 2))
        return cls(
            speeds=tuple(knots),
            peaks=tuple(averaged),
            tail_slope=tail_slope,
            zero_speed_peak=zero_speed_peak,
            fit_speed_floor=fit_speed_floor,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TabulatedImpact:
        return cls(
            speeds=tuple(data["speeds"]),
            peaks=tuple(data["peaks"]),
            tail_slope=data["tail_slope"],
            zero_speed_peak=data.get("zero_speed_peak"),
            fit_speed_floor=data.get("fit_speed_floor", DEFAULT_FIT_SPEED_FLOOR),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dictionary; ``from_dict`` inverts it exactly."""
        return {
            "speeds": list(self.speeds),
            "peaks": list(self.peaks),
            "tail_slope": self.tail_slope,
            "zero_speed_peak": self.zero_speed_peak,
            "fit_speed_floor": self.fit_speed_floor,
        }

    @property
    def base_peak(self) -> float:
        """Peak at zero closing speed."""
        if self.speeds[0] == 0.0:
            return self.peaks[0]
        return 0.0 if self.zero_speed_peak is None else self.zero_speed_peak

    @property
    def is_monotone(self) -> bool:
        """Strictly increasing over [0, inf): base below first knot, rising knots, positive tail."""
        rising = bool(np.all(np.diff(self.peaks) > 0.0))
        base_ok = self.speeds[0] == 0.0 or self.base_peak < self.peaks[0]
        return rising and base_ok and self.tail_slope > 0.0

    def peak(self, speed: ArrayLike) -> float | NDArray[np.float64]:
        """Peak tension at closing speed(s) ``speed`` >= 0."""
        values = np.asarray(speed, dtype=float)
        if np.any(values < 0.0) or not np.all(np.isfinite(values)):
            raise ValueError("closing speeds must be finite and nonnegative")
        first_speed, last_speed = self.speeds[0], self.speeds[-1]
        first_peak, last_peak = self.peaks[0], self.peaks[-1]
        inside = np.clip(values, first_speed, last_speed)
        interior = self._interpolant(inside) if self._interpolant is not None else np.full_like(
            values, first_peak
        )
        base = self.base_peak
        below = base + (first_peak - base) * values / first_speed if first_speed > 0.0 else interior
        above = last_peak + self.tail_slope * (values - last_speed)
        return _result(
            np.where(values < first_speed, below, np.where(values > last_speed, above, interior))
        )

    def critical_speed(self, Tb: ArrayLike) -> float | NDArray[np.float64]:
        """Inverse closing speed Z^-1(Tb); 0 where even a zero-speed engagement reaches Tb."""
        if not self.is_monotone:
            raise ValueError("impact table is not monotone; Z^-1(Tb) is undefined")
        levels = np.asarray(Tb, dtype=float)
        if np.any(np.isnan(levels)):
            raise ValueError("breaking strengths must not be NaN")
        speeds = np.array([self._invert(float(level)) for level in levels.ravel()])
        return _result(speeds.reshape(levels.shape))

    def _invert(self, level: float) -> float:
        base = self.base_peak
        first_speed, last_speed = self.speeds[0], self.speeds[-1]
        first_peak, last_peak = self.peaks[0], self.peaks[-1]
        if level <= base:
            return 0.0
        if level < first_peak:
            return first_speed * (level - base) / (first_peak - base)
        if level > last_peak:
            return last_speed + (level - last_peak) / self.tail_slope
        index = int(np.searchsorted(self.peaks, level))
        if self.peaks[index] == level:
            return self.speeds[index]
        return float(
            optimize.brentq(
                lambda value: float(self._interpolant(value)) - level,
                self.speeds[index - 1],
                self.speeds[index],
                xtol=1e-12,
                rtol=4.0 * np.finfo(float).eps,
            )
        )
