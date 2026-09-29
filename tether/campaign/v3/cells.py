"""Plan v3 cells, seeds and jobs (ref/cascade_plan_v3.md, WP0).

Stage A (WP1/WP2, 16 cells): ten v1 cascade-grid cells re-simulated with bit-identical
``FleetRunSpec`` (the reducer asserts that their marks reproduce records/phase2), four
fan-formation cells, and the stiffness pair at the calibration cell.  Stage B (WP3, 4 cells)
is the intensity sweep at T0 = 0.6 kN; stage B_ext (2 cells) is launched only under the WP3
rule.  All cells: Gaussian local AR(1) weather, recording cable mode, 600 s after a 20 s
warm-up, pilot seeds 2001-2002 (never scored) and statistics seeds 2003-2022, common random
numbers across cells, exactly as tether/campaign/phase2.py.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from tether.campaign.fleet_run import FleetRunSpec
from tether.campaign.phase2 import CACHE_PATH as V1_CACHE_PATH
from tether.campaign.phase2 import DURATION, PILOT_SEEDS, STAT_SEEDS, WARMUP
from tether.campaign.phase2 import cell_name as v1_cell_name
from tether.campaign.phase2 import cell_specs as v1_cell_specs
from tether.physics import constants
from tether.theory import reduced_lti as R
from tether.theory import transmission as TR

CALIBRATION_CELL = "T600_k500_I0.5"
REUSED_V1_CELLS = (
    "T600_k500_I0.5",
    "T600_k500_I1.0",
    "T600_k1000_I0.5",
    "T600_k1000_I1.0",
    "T1000_k500_I1.0",
    "T1000_k1000_I0.5",
    "T1000_k1000_I1.0",
    "T1400_k500_I1.0",
    "T1400_k1000_I1.0",
    "T1800_k1000_I1.0",
)
FAN_ARC_HALF_ANGLE = 0.55
FAN_HEADING_GAIN = 500.0
FAN_CELLS = (("FAN_T1000_I0.5", 1000.0, 0.5), ("FAN_T1000_I1.0", 1000.0, 1.0),
             ("FAN_T600_I0.5", 600.0, 0.5), ("FAN_T600_I1.0", 600.0, 1.0))
# The stiffness pair spans x16 (not x4): the exact response predicts a difference of 0.045 in the
# normalised drop between these, against 0.023 for k/2 vs 2k (records/v3/cascade_predictions.json).
STIFFNESS_FACTORS = {"kx0.25": 0.25, "kx4": 4.0}
SWEEP_PRETENSION, SWEEP_HEADING_GAIN = 600.0, 500.0
SWEEP_INTENSITIES_B = (0.35, 0.75, 1.25, 1.5)
SWEEP_INTENSITIES_B_EXT = (1.75, 2.0)
SWEEP_INTENSITIES_FROM_A = (0.5, 1.0)
STAGES = ("A", "B", "B_ext")


@dataclass(frozen=True)
class V3Job:
    cell: str
    spec: FleetRunSpec
    seed: int
    pilot: bool
    stage: str
    v1_cell: str | None = None  # the v1 cascade-grid cell whose marks this run must reproduce
    expected_t_up: np.ndarray | None = None
    expected_T_peak: np.ndarray | None = None

    @property
    def cost(self) -> float:
        return self.spec.duration + self.spec.warmup


def _base(pretension: float, heading_gain: float, intensity: float, **changes) -> FleetRunSpec:
    return replace(
        FleetRunSpec(pretension=pretension, heading_gain=heading_gain, weather_scale=intensity,
                     duration=DURATION, warmup=WARMUP),
        **changes,
    )


def cell_specs(stage: str = "A") -> dict[str, FleetRunSpec]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    specs: dict[str, FleetRunSpec] = {}
    if stage == "A":
        v1 = v1_cell_specs()
        for name in REUSED_V1_CELLS:
            specs[name] = v1[name]
        for name, pretension, intensity in FAN_CELLS:
            specs[name] = _base(pretension, FAN_HEADING_GAIN, intensity, formation="fan",
                                arc_half_angle=FAN_ARC_HALF_ANGLE)
        for suffix, factor in STIFFNESS_FACTORS.items():
            specs[f"{CALIBRATION_CELL}_{suffix}"] = replace(v1[CALIBRATION_CELL],
                                                            stiffness=factor * constants.CABLE_STIFFNESS)
        return specs
    intensities = SWEEP_INTENSITIES_B if stage == "B" else SWEEP_INTENSITIES_B_EXT
    for intensity in intensities:
        specs[v1_cell_name(SWEEP_PRETENSION, SWEEP_HEADING_GAIN, intensity)] = _base(
            SWEEP_PRETENSION, SWEEP_HEADING_GAIN, intensity)
    return specs


def sweep_cells() -> tuple[str, ...]:
    """Every cell of the T0 = 0.6 kN intensity sweep, in intensity order (A, B and B_ext)."""
    names = {v1_cell_name(SWEEP_PRETENSION, SWEEP_HEADING_GAIN, i): i
             for i in SWEEP_INTENSITIES_FROM_A + SWEEP_INTENSITIES_B + SWEEP_INTENSITIES_B_EXT}
    return tuple(sorted(names, key=names.get))


def parse_cell(name: str) -> dict:
    """Formation, pretension, heading gain, intensity and stiffness factor from a cell name."""
    base, factor = name, 1.0
    for suffix, value in STIFFNESS_FACTORS.items():
        if name.endswith("_" + suffix):
            base, factor = name[: -len(suffix) - 1], value
    if base.startswith("FAN_"):
        _, t0, intensity = base.split("_")
        return {"formation": "fan", "pretension": float(t0[1:]), "heading_gain": FAN_HEADING_GAIN,
                "intensity": float(intensity[1:]), "stiffness_factor": factor}
    t0, kh, intensity = base.split("_")
    return {"formation": "parallel", "pretension": float(t0[1:]), "heading_gain": float(kh[1:]),
            "intensity": float(intensity[1:]), "stiffness_factor": factor}


def v1_reference(cell: str) -> str | None:
    return cell if cell in REUSED_V1_CELLS else None


@lru_cache(maxsize=1)
def v1_marks_index(cache_path: Path = V1_CACHE_PATH) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    """(cell, seed) -> (t_up, T_peak) of every v1 cascade-grid run, from the committed cache."""
    import pickle

    with Path(cache_path).open("rb") as handle:
        payload = pickle.load(handle)
    index = {}
    for job, summary in zip(payload["jobs"], payload["summaries"]):
        marks = summary["marks"]
        index[(job.cell, int(job.seed))] = (
            np.asarray(marks["t_up"], dtype=float).copy(),
            np.asarray(marks["T_peak"], dtype=float).copy(),
        )
    return index


def all_jobs(stage: str = "A", expected: bool = True) -> list[V3Job]:
    jobs = []
    index = v1_marks_index() if expected and stage == "A" else {}
    for name, spec in cell_specs(stage).items():
        reference = v1_reference(name)
        for seed, pilot in [(s, True) for s in PILOT_SEEDS] + [(s, False) for s in STAT_SEEDS]:
            t_up = t_peak = None
            if reference is not None and (reference, seed) in index:
                t_up, t_peak = index[(reference, seed)]
            jobs.append(V3Job(name, spec, seed, pilot, stage, reference, t_up, t_peak))
    return jobs


def response_spec(spec: FleetRunSpec, **changes) -> TR.ResponseSpec:
    """The transmission model's input for one run spec (the taut tow it is linearised about)."""
    inp = R.ReducedModelInput(
        formation=spec.formation, arc_half_angle=spec.arc_half_angle, pretension=spec.pretension,
        heading_gain=spec.heading_gain, trim_gain=spec.trim_gain, drag_law=spec.drag_law,
        weather_direction=spec.weather_direction, weather_scale=spec.weather_scale,
        weather_rho=spec.weather_rho, weather_front_angle=spec.weather_front_angle, sway_gain=spec.k_sigma,
    )
    stiffness = constants.CABLE_STIFFNESS if spec.stiffness is None else float(spec.stiffness)
    damping = constants.CABLE_DAMPING if spec.damping is None else float(spec.damping)
    return replace(TR.ResponseSpec(inp, stiffness, damping), **changes)
