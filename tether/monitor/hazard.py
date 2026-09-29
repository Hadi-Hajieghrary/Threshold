"""Run-time hazard of plan IV.9 (Props 10-11): constant-acceleration Monte Carlo and Rice series.

The hazard is evaluated at every 10 Hz monitor tick at which the vessel's tension slack flag
is set and is 0 while taut. Each agent owns one generator, ``SeedSequence([master, 17,
agent])``, advanced only at its slack ticks and in tick order, so the offline series equals
the online monitor tick for tick. The critical closing speed is v_b = Z^-1(T_b) from the
Phase 1 impact table at the cell's pretension and the cable's mirror-symmetric position.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tether.theory.excursion import constant_acceleration_hazard_mc, rice_dangerous_upcrossings
from tether.theory.impact import TabulatedImpact

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

HAZARD_STREAM = 17
DEFAULT_SAMPLES = 2048
DEFAULT_HORIZON = 2.0
DEFAULT_GRID = 0.01
DEFAULT_VESSEL_COUNT = 5
IMPACT_TABLE_PATH = Path(__file__).resolve().parents[2] / "records" / "phase1" / "impact_table.json"
_PSD_TOLERANCE = 1e-12


def field(output: Any, name: str) -> np.ndarray:
    """Array ``name`` of an EstimatorOutput-like object or of a mapping."""
    value = output[name] if isinstance(output, Mapping) else getattr(output, name)
    return np.asarray(value)


def hazard_generator(master_seed: int, agent: int) -> np.random.Generator:
    """The agent's hazard Monte Carlo generator, ``SeedSequence([master, 17, agent])``."""
    return np.random.default_rng(
        np.random.SeedSequence([int(master_seed), HAZARD_STREAM, int(agent)])
    )


def cable_position(agent: int, vessel_count: int = DEFAULT_VESSEL_COUNT) -> int:
    """Impact-table position of a cable: 0 outer, 1 inner, 2 centre for five vessels."""
    if not 0 <= agent < vessel_count:
        raise ValueError("agent index outside the fleet")
    return min(int(agent), vessel_count - 1 - int(agent))


@lru_cache(maxsize=4)
def _impact_table(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def impact_law(
    pretension: float,
    agent: int,
    vessel_count: int = DEFAULT_VESSEL_COUNT,
    table_path: Path | str = IMPACT_TABLE_PATH,
) -> TabulatedImpact:
    """Tabulated engagement law of one cable: PCHIP through the knots, asymptotic-impedance tail."""
    table = _impact_table(str(table_path))
    key = f"T{int(round(pretension))}_c{cable_position(agent, vessel_count)}"
    if key not in table:
        available = sorted({entry["pretension"] for entry in table.values()})
        raise ValueError(f"no impact table at pretension {pretension}; tabulated: {available}")
    entry = table[key]
    return TabulatedImpact(
        speeds=tuple(entry["speeds"]),
        peaks=tuple(entry["peaks"]),
        tail_slope=entry["asymptotic_impedance"],
    )


def critical_speed(
    breaking_strength: float,
    pretension: float,
    agent: int,
    vessel_count: int = DEFAULT_VESSEL_COUNT,
    table_path: Path | str = IMPACT_TABLE_PATH,
) -> float:
    """v_b = Z^-1(T_b) for one cable."""
    law = impact_law(pretension, agent, vessel_count, table_path)
    return float(law.critical_speed(float(breaking_strength)))


def critical_speeds(
    breaking_strength: float,
    pretension: float,
    vessel_count: int = DEFAULT_VESSEL_COUNT,
    table_path: Path | str = IMPACT_TABLE_PATH,
) -> FloatArray:
    """v_b per vessel, shape (vessel_count,)."""
    return np.array(
        [
            critical_speed(breaking_strength, pretension, agent, vessel_count, table_path)
            for agent in range(vessel_count)
        ]
    )


def precursor_state(
    e: float, edot: float, sigma: ArrayLike, a_hat: float, sigma_a: float
) -> tuple[FloatArray, FloatArray]:
    """Mean (e0, v0, a) and covariance blockdiag(Sigma, sigma_a^2) of the IV.9 prediction model."""
    block = np.asarray(sigma, dtype=float).reshape(2, 2)
    covariance = np.zeros((3, 3))
    covariance[:2, :2] = 0.5 * (block + block.T)
    covariance[2, 2] = float(sigma_a) ** 2
    return np.array([e, edot, a_hat], dtype=float), covariance


def _admissible(mean: FloatArray, covariance: FloatArray) -> bool:
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(covariance))):
        return False
    scale = max(float(np.max(np.abs(covariance))), 1e-300)
    return bool(np.min(np.linalg.eigvalsh(covariance)) >= -_PSD_TOLERANCE * scale)


def _check_model(n_samples: int, horizon: float, dt: float) -> None:
    if int(n_samples) < 1 or not horizon > 0.0 or not dt > 0.0:
        raise ValueError("requires n_samples >= 1, horizon > 0 and dt > 0")


def tick_hazard(
    e: float,
    edot: float,
    sigma: ArrayLike,
    a_hat: float,
    sigma_a: float,
    v_b: float,
    rng: np.random.Generator,
    n_samples: int = DEFAULT_SAMPLES,
    horizon: float = DEFAULT_HORIZON,
    dt: float = DEFAULT_GRID,
) -> float:
    """One monitor tick of the IV.9 hazard; NaN, without drawing, for a non-finite or
    indefinite precursor."""
    _check_model(n_samples, horizon, dt)
    mean, covariance = precursor_state(e, edot, sigma, a_hat, sigma_a)
    if not _admissible(mean, covariance):
        return float("nan")
    return constant_acceleration_hazard_mc(mean, covariance, horizon, v_b, n_samples, rng, dt)


def tick_rice(
    e: float,
    edot: float,
    sigma: ArrayLike,
    a_hat: float,
    sigma_a: float,
    v_b: float,
    horizon: float = DEFAULT_HORIZON,
) -> float:
    """Prop 11 count of dangerous upcrossings for one tick; NaN where the quadrature fails."""
    mean, covariance = precursor_state(e, edot, sigma, a_hat, sigma_a)
    if not _admissible(mean, covariance):
        return float("nan")
    try:
        return rice_dangerous_upcrossings(
            mean[0], mean[1], covariance[:2, :2], mean[2], float(sigma_a), horizon, v_b
        )
    except ValueError:
        return float("nan")


@dataclass(frozen=True)
class HazardSeries:
    """One agent's hazard over the monitor ticks; ``rice`` is NaN where not evaluated."""

    agent: int
    v_b: float
    time: FloatArray
    slack: BoolArray
    hazard: FloatArray
    rice: FloatArray


def hazard_series(
    output: Any,
    v_b: float,
    master_seed: int,
    agent_index: int,
    n_samples: int = DEFAULT_SAMPLES,
    horizon: float = DEFAULT_HORIZON,
    dt: float = DEFAULT_GRID,
    rice: bool = True,
) -> HazardSeries:
    """Hazard at every tick of an EstimatorOutput-like ``output`` (object or mapping holding
    ``time``, ``slack``, ``e_hat``, ``edot_hat``, ``sigma``, ``a_hat``, ``sigma_a``)."""
    _check_model(n_samples, horizon, dt)
    time = field(output, "time").astype(float)
    slack = field(output, "slack").astype(bool)[:, agent_index]
    e = field(output, "e_hat")[:, agent_index].astype(float)
    edot = field(output, "edot_hat")[:, agent_index].astype(float)
    sigma = field(output, "sigma")[:, agent_index].astype(float)
    a_hat = field(output, "a_hat")[:, agent_index].astype(float)
    sigma_a = field(output, "sigma_a")[:, agent_index].astype(float)
    if slack.shape != time.shape or sigma.shape != time.shape + (2, 2):
        raise ValueError("output arrays must be (m, N), with sigma (m, N, 2, 2)")
    rng = hazard_generator(master_seed, agent_index)
    hazard = np.zeros(time.size)
    rice_count = np.where(slack, np.nan, 0.0) if rice else np.full(time.size, np.nan)
    for tick in np.flatnonzero(slack):
        state = (e[tick], edot[tick], sigma[tick], a_hat[tick], sigma_a[tick], v_b)
        hazard[tick] = tick_hazard(*state, rng, n_samples, horizon, dt)
        if rice:
            rice_count[tick] = tick_rice(*state, horizon)
    return HazardSeries(
        agent=int(agent_index),
        v_b=float(v_b),
        time=time,
        slack=slack,
        hazard=hazard,
        rice=rice_count,
    )


@dataclass(frozen=True)
class FleetHazard:
    """Hazard of every agent, arrays (m, N)."""

    time: FloatArray
    slack: BoolArray
    hazard: FloatArray
    rice: FloatArray
    v_b: FloatArray


def fleet_hazard(
    output: Any,
    v_b: ArrayLike,
    master_seed: int,
    n_samples: int = DEFAULT_SAMPLES,
    horizon: float = DEFAULT_HORIZON,
    dt: float = DEFAULT_GRID,
    rice: bool = True,
) -> FleetHazard:
    """``hazard_series`` for every agent; ``v_b`` is a scalar or one value per vessel."""
    vessels = field(output, "slack").shape[1]
    speeds = np.broadcast_to(np.asarray(v_b, dtype=float), (vessels,)).copy()
    series = [
        hazard_series(output, speeds[agent], master_seed, agent, n_samples, horizon, dt, rice)
        for agent in range(vessels)
    ]
    return FleetHazard(
        time=series[0].time,
        slack=np.column_stack([item.slack for item in series]),
        hazard=np.column_stack([item.hazard for item in series]),
        rice=np.column_stack([item.rice for item in series]),
        v_b=speeds,
    )
