"""The Squall Passage mission and its campaign runner (plan I.1, IV.5, IV.7; Phase 5 spec).

A 130 s closed-loop mission of the fan formation from rest: thrust ramp, beacon on vessel
0 from 30 s, a 60 deg dogleg, a common-mode broadside squall on top of stationary
common-mode t3 weather, deceleration, and a docking box.  Phase 5 runs it in recording
mode for the calibration campaign; Phase 6 adds the estimator adapter and the supervisor
through ``extra_systems`` with ``supervised`` (the controller's ``thrust_scale`` port).

Declared choices (copied into the Phase 5 report):

* Time origin t = 0, no warm-up.  Start from rest in the design fan geometry: load at the
  origin with heading 0, vessels at the design headings, every chord at the rest length
  (e = 0), every velocity zero.  The controllers' gyro headings start at the true ones.
* The background weather is ramped in with the thrust, by the same raised cosine over
  [0, 8] s (``weather_ramp_duration``); from 8 s on it is the stationary pre-rolled series.
  Without the ramp, full weather acting on the untensioned fleet from t = 0 drifts vessels
  and load apart before thrust builds tension, and the start-up re-engagement reached
  7.5-18.5 kN in 5 of 8 probe seeds (depth to 4.6 m), as large as the squall's snaps; with
  the ramp no cable slackens in [0, 20] s.  Declared 2026-09-13, before the Phase 5
  calibration missions were run (records/phase5/phase5_spec.md addendum).
* Envelopes are raised cosines (1 - cos(pi s)) / 2: thrust 0 -> F_T,i over [0, 8] s and
  F_T,i -> 0 over [110, 130] s; the dogleg offset 0 -> +60 deg over [40, 70] s, common to
  every vessel and added to the pre-loaded fan reference, positive = turning to port;
  the squall envelope Psi rises over [50, 55] s, holds 1 over [55, 65] s, and falls over
  [65, 70] s.
* Squall force: Psi(t) A u on every vessel and Psi(t) (A_L / A_A) A u on the load, with
  A_L / A_A = 3.5 / 0.7 = 5 (the pinned sigma_W ratio) and u = -y in the world frame
  (wind from port); sampled at the 10 ms grid instants and added to the pre-rolled
  weather array, so the cable bank's W_rel includes it.
* Squall amplitude: A = lambda T0 / d*, with d* the largest quasi-static tension deficit
  per newton of A at the pre-dogleg operating point, from the reduced linear model:
  A x = -B w solved on the invariant subspace of the modes faster than the plateau
  (decay rate >= 1 / 10 s), the slower lateral-swing modes (time constants 20-325 s)
  held, and mapped through k e + c edot.  Solving every mode to rest halves d* and is not
  approached within the squall.  ``squall_drake_check`` compares d* with the odd part of
  two Drake step runs (the pattern and its reverse, same seed) at lambda = 0.1 from the
  steady tow, averaged 3-6 s after the step; the even part is the second-order response.
* Docking box: centre = load position at 130 s of the no-weather, no-squall mission at
  ``REFERENCE_SEED``; docking error = load distance from the centre at the end; mission
  time = first 10 ms log instant with the load within 5 m of the centre, else the mission
  duration (130 s, also for a run stopped early by closure) plus the remaining distance
  to the 5 m circle at the nominal tow speed.
* ``extra_systems`` is ``build_run``'s pre-build hook ``(builder, fleet, sensors,
  controller) -> dict | None``; ``supervised`` gives the controller bank its
  ``thrust_scale`` input port (Phase 6).
* Records: one NPZ per seed (sensor samples, 1 ms and 10 ms truth logs, marks, taut
  intervals, surveyed initial poses, outcome) under ``CALIBRATION_DIR``, plus a manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.linalg import expm, schur, solve_sylvester

from tether.campaign.common import DEFAULT_WORKERS, RECORDS, json_bytes, npz_bytes, relative, run_pool, source_state, write_bytes
from tether.campaign.fleet_run import FleetRun, FleetRunSpec, build_run, fan_heading_reference, run_to_end
from tether.campaign.summaries import MARK_FIELDS
from tether.control.controller import CABLE_TRIM_GAIN, NOMINAL_HEADING_GAIN
from tether.physics import constants
from tether.physics.fleet import CableParameters, equilibrium_state, formation_geometry, operating_point, plant_state
from tether.physics.weather import stationary_weather_forces
from tether.theory.reduced_lti import ReducedModelInput, reduced_model

AREA_RATIO = constants.LOAD_WEATHER_STD / constants.VESSEL_WEATHER_STD
REFERENCE_SEED = 0
CHECK_LEVEL = 0.1
CHECK_SEED = 1
CHECK_DURATION = 6.0
CHECK_WINDOW = (3.0, 6.0)
CALIBRATION_SEEDS = tuple(range(5001, 5041))
CALIBRATION_DIR = RECORDS / "phase5" / "cache" / "missions"
TAUT_FIELDS = ("t_start", "t_end", "q_peak", "cable")


@dataclass(frozen=True)
class MissionSpec:
    """Everything that fixes one Squall Passage mission apart from the master seed."""

    arc_half_angle: float = 0.55
    pretension: float = 1000.0
    heading_gain: float = NOMINAL_HEADING_GAIN
    trim_gain: float = CABLE_TRIM_GAIN
    drag_law: str = "linear"
    squall_level: float = 1.5
    squall_centre: float = 60.0
    squall_rise: float = 5.0
    squall_plateau: float = 10.0
    squall_direction: float = -math.pi / 2.0
    weather_distribution: str = "student_t3"
    weather_direction: str = "common"
    weather_scale: float = 1.0
    weather_ramp_duration: float = 8.0
    ramp_duration: float = 8.0
    dogleg_start: float = 40.0
    dogleg_end: float = 70.0
    dogleg_angle: float = math.pi / 3.0
    deceleration_start: float = 110.0
    duration: float = 130.0
    beacon_on: float = 30.0
    docking_radius: float = 5.0
    cable_mode: str = "recording"
    break_threshold: float | None = None

    def run_spec(self) -> FleetRunSpec:
        return FleetRunSpec(
            formation="fan",
            arc_half_angle=self.arc_half_angle,
            pretension=self.pretension,
            heading_gain=self.heading_gain,
            trim_gain=self.trim_gain,
            drag_law=self.drag_law,
            weather_distribution=self.weather_distribution,
            weather_direction=self.weather_direction,
            weather_scale=self.weather_scale,
            duration=self.duration,
            warmup=0.0,
            cable_mode=self.cable_mode,
            break_threshold=self.break_threshold,
        )

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


def raised_cosine(fraction):
    """0 below 0, 1 above 1, (1 - cos(pi s)) / 2 between; scalars or arrays."""
    return 0.5 * (1.0 - np.cos(math.pi * np.clip(fraction, 0.0, 1.0)))


def thrust_envelope(spec: MissionSpec, time):
    envelope = raised_cosine(np.asarray(time) / spec.ramp_duration)
    span = spec.duration - spec.deceleration_start
    if span > 0.0:
        envelope = envelope * (1.0 - raised_cosine((np.asarray(time) - spec.deceleration_start) / span))
    return envelope


def dogleg_offset(spec: MissionSpec, time):
    return spec.dogleg_angle * raised_cosine((np.asarray(time) - spec.dogleg_start) / (spec.dogleg_end - spec.dogleg_start))


def squall_envelope(spec: MissionSpec, time):
    time = np.asarray(time)
    plateau_end = spec.squall_centre + 0.5 * spec.squall_plateau
    onset = plateau_end - spec.squall_plateau - spec.squall_rise
    return raised_cosine((time - onset) / spec.squall_rise) - raised_cosine((time - plateau_end) / spec.squall_rise)


def mission_geometry(spec: MissionSpec):
    return formation_geometry("fan", arc_half_angle=spec.arc_half_angle)


def nominal_speed(spec: MissionSpec) -> float:
    return operating_point(mission_geometry(spec), spec.pretension, spec.drag_law).speed


@dataclass(frozen=True)
class MissionSchedules:
    """Pre-loaded controller schedules ``t -> (N,)``; ``thrust_scale`` is the supervisor hook."""

    spec: MissionSpec
    thrusts: np.ndarray
    reference: np.ndarray
    thrust_scale: Callable[[float], np.ndarray] | None = None

    def surge(self, time: float) -> np.ndarray:
        command = self.thrusts * float(thrust_envelope(self.spec, time))
        if self.thrust_scale is not None:
            command = command * np.asarray(self.thrust_scale(time), dtype=float)
        return command

    def heading(self, time: float) -> np.ndarray:
        return self.reference + float(dogleg_offset(self.spec, time))


def mission_schedules(
    spec: MissionSpec, thrust_scale: Callable[[float], np.ndarray] | None = None
) -> MissionSchedules:
    geometry = mission_geometry(spec)
    operating = operating_point(geometry, spec.pretension, spec.drag_law)
    reference = fan_heading_reference(geometry, operating, CableParameters(), spec.heading_gain, spec.trim_gain)
    return MissionSchedules(spec, operating.thrusts.copy(), reference, thrust_scale)


def initial_state(spec: MissionSpec) -> np.ndarray:
    """Design fan geometry at rest: chords at the rest length, design headings."""
    geometry = mission_geometry(spec)
    operating = operating_point(geometry, spec.pretension, spec.drag_law)
    at_rest = replace(operating, speed=0.0, tensions=np.zeros(geometry.vessel_count))
    return equilibrium_state(geometry, at_rest)


def squall_pattern(direction: float, vessel_count: int = constants.VESSEL_COUNT) -> np.ndarray:
    """(N + 1, 2) squall force per newton of vessel amplitude, area-scaled on the load."""
    unit = np.array([math.cos(direction), math.sin(direction)])
    return np.array([AREA_RATIO] + [1.0] * vessel_count)[:, None] * unit[None, :]


@dataclass(frozen=True)
class SquallResponse:
    """Reduced linear model of the tow driven by the unit squall pattern (per newton of A)."""

    state_matrix: np.ndarray
    forcing: np.ndarray
    tension_output: np.ndarray

    def static_deficit(self) -> np.ndarray:
        return self.tension_output @ np.linalg.solve(self.state_matrix, self.forcing)

    def quasi_static_deficit(self, rate: float) -> tuple[np.ndarray, np.ndarray]:
        """Deficit with the modes decaying slower than ``rate`` held; also their eigenvalues."""
        form, basis, count = schur(self.state_matrix, output="real", sort=lambda re, im: re <= -rate)
        if count == form.shape[0]:
            return self.static_deficit(), np.array([])
        fast = form[:count, :count]
        coupling = solve_sylvester(fast, -form[count:, count:], -form[:count, count:])
        projection = basis[:, :count].T - coupling @ basis[:, count:].T
        state = basis[:, :count] @ np.linalg.solve(fast, projection @ self.forcing)
        return self.tension_output @ state, np.sort_complex(np.linalg.eigvals(form[count:, count:]))

    def step_deficit(self, times: np.ndarray) -> np.ndarray:
        """Deficit at ``times`` after a unit step of the pattern from the operating point."""
        size = self.state_matrix.shape[0]
        augmented = np.zeros((size + 1, size + 1))
        augmented[:size, :size] = self.state_matrix
        augmented[:size, size] = self.forcing
        return np.array([-self.tension_output @ expm(augmented * t)[:size, size] for t in np.asarray(times, dtype=float)])


@lru_cache(maxsize=8)
def _squall_response(arc_half_angle, pretension, heading_gain, trim_gain, drag_law, direction) -> SquallResponse:
    model = reduced_model(
        ReducedModelInput(
            formation="fan",
            arc_half_angle=arc_half_angle,
            pretension=pretension,
            heading_gain=heading_gain,
            trim_gain=trim_gain,
            drag_law=drag_law,
            weather_direction="common",
        )
    )
    size = model.state_matrix.shape[0]
    pattern = squall_pattern(direction, model.geometry.vessel_count)
    return SquallResponse(model.state_matrix, model.input_matrix @ pattern.ravel(), model.outputs["q"][:, :size])


def squall_response(spec: MissionSpec) -> SquallResponse:
    return _squall_response(
        spec.arc_half_angle, spec.pretension, spec.heading_gain, spec.trim_gain, spec.drag_law, spec.squall_direction
    )


@dataclass(frozen=True)
class SquallCalibration:
    """Squall amplitude per body from the quasi-static deficit criterion."""

    level: float
    pretension: float
    unit_deficit: np.ndarray
    static_unit_deficit: np.ndarray
    held_eigenvalues: np.ndarray
    critical_cable: int
    vessel_amplitude: float
    load_amplitude: float


def calibrate_squall(spec: MissionSpec) -> SquallCalibration:
    if spec.squall_level < 0.0:
        raise ValueError("the squall level must be non-negative")
    response = squall_response(spec)
    unit, held = response.quasi_static_deficit(1.0 / spec.squall_plateau)
    critical = int(np.argmax(unit))
    if not unit[critical] > 0.0:
        raise ValueError("the squall direction unloads no cable quasi-statically")
    amplitude = spec.squall_level * spec.pretension / float(unit[critical])
    return SquallCalibration(
        level=spec.squall_level,
        pretension=spec.pretension,
        unit_deficit=unit,
        static_unit_deficit=response.static_deficit(),
        held_eigenvalues=held,
        critical_cable=critical,
        vessel_amplitude=amplitude,
        load_amplitude=AREA_RATIO * amplitude,
    )


def weather_sample_count(duration: float) -> int:
    return int(math.ceil(duration / constants.WEATHER_PERIOD)) + 2


def squall_forces(spec: MissionSpec, calibration: SquallCalibration, samples: int) -> np.ndarray:
    envelope = squall_envelope(spec, np.arange(samples) * constants.WEATHER_PERIOD)
    return calibration.vessel_amplitude * envelope[:, None, None] * squall_pattern(spec.squall_direction)[None]


def mission_weather(spec: MissionSpec, seed: int) -> np.ndarray | None:
    """Pre-rolled background weather plus the squall on the 10 ms grid (None when both are off)."""
    weather = None
    if spec.weather_scale > 0.0:
        weather = stationary_weather_forces(
            seed,
            spec.duration,
            distribution=spec.weather_distribution,
            direction=spec.weather_direction,
            scale=spec.weather_scale,
        )
        if spec.weather_ramp_duration > 0.0:
            time = np.arange(weather.shape[0]) * constants.WEATHER_PERIOD
            weather = weather * raised_cosine(time / spec.weather_ramp_duration)[:, None, None]
    if spec.squall_level != 0.0:
        samples = weather_sample_count(spec.duration) if weather is None else weather.shape[0]
        squall = squall_forces(spec, calibrate_squall(spec), samples)
        weather = squall if weather is None else weather + squall
    return weather


def build_mission(
    spec: MissionSpec,
    seed: int,
    extra_systems: Callable | None = None,
    *,
    thrust_scale: Callable[[float], np.ndarray] | None = None,
    supervised: bool = False,
    exemptions: frozenset[str] = frozenset(),
) -> FleetRun:
    """Assemble the mission's closed loop through ``build_run`` without advancing it.

    ``extra_systems(builder, fleet, sensors, controller)`` is ``build_run``'s pre-build
    hook: it may add and connect systems before the build (returning a dict merged into
    ``fleet.extras``), and they are linted with the rest of the diagram.  ``supervised``
    declares the controller bank's ``thrust_scale`` input port for such a system;
    the callable ``thrust_scale(t) -> (N,)`` multiplies the pre-loaded surge schedule.
    """
    schedules = mission_schedules(spec, thrust_scale)
    return build_run(
        spec.run_spec(),
        seed,
        weather=mission_weather(spec, seed),
        heading_reference=schedules.heading,
        surge_schedule=schedules.surge,
        initial_state=initial_state(spec),
        beacon_on=spec.beacon_on,
        supervised=supervised,
        pre_build=extra_systems,
        exemptions=exemptions,
    )


def run_mission(
    spec: MissionSpec,
    seed: int,
    extra_systems: Callable | None = None,
    *,
    thrust_scale: Callable[[float], np.ndarray] | None = None,
    supervised: bool = False,
    exemptions: frozenset[str] = frozenset(),
) -> FleetRun:
    """Run one mission to its end (or to formation closure); logs stay on the run."""
    return run_to_end(
        build_mission(spec, seed, extra_systems, thrust_scale=thrust_scale, supervised=supervised, exemptions=exemptions)
    )


def quiet_spec(spec: MissionSpec) -> MissionSpec:
    """The reference mission: no weather, no squall, recording mode."""
    return replace(spec, squall_level=0.0, weather_scale=0.0, cable_mode="recording", break_threshold=None)


@dataclass(frozen=True)
class DockingReference:
    centre: np.ndarray
    heading: float
    seed: int
    config_hash: str


@lru_cache(maxsize=4)
def docking_reference(spec: MissionSpec, seed: int = REFERENCE_SEED) -> DockingReference:
    quiet = quiet_spec(spec)
    run = run_mission(quiet, seed)
    context = run.simulator.get_context()
    if run.fleet.cables.closure is not None or context.get_time() < quiet.duration - 1.0e-9:
        raise RuntimeError("the reference mission did not complete")
    state = plant_state(run.fleet, context)
    return DockingReference(state[:2].copy(), float(state[2]), seed, quiet.config_hash())


@dataclass(frozen=True)
class SquallCheck:
    """Drake step runs at a small squall level against the quasi-static prediction.

    ``measured`` is the deficit under the pattern and ``reversed`` under the opposite
    pattern (same seed); their half-difference ``odd`` is the linear response and their
    half-sum ``even`` the second-order one.  ``ratio`` = odd / quasi-static on the
    critical cable; ``one_sided_ratio`` uses ``measured``; ``linear_ratio`` compares with
    the linear model's step response over the same window.
    """

    level: float
    vessel_amplitude: float
    window: tuple[float, float]
    critical_cable: int
    predicted: np.ndarray
    linear: np.ndarray
    measured: np.ndarray
    reversed: np.ndarray
    odd: np.ndarray
    even: np.ndarray
    ratio: float
    one_sided_ratio: float
    linear_ratio: float
    static_ratio: float


def constant_squall_tension(
    spec: MissionSpec, amplitude: float, direction: float, seed: int, duration: float
) -> tuple[np.ndarray, np.ndarray]:
    """1 ms times and true tensions of the closed-loop steady tow under a constant pattern."""
    weather = np.repeat((amplitude * squall_pattern(direction))[None], weather_sample_count(duration), axis=0)
    run = run_to_end(build_run(replace(spec.run_spec(), weather_scale=0.0, duration=duration), seed, weather=weather))
    cables = run.fleet.cables
    log = cables.log
    elongation = log.elongation[: log.count]
    taut = cables.cable.stiffness * elongation + cables.cable.damping * log.rate[: log.count]
    return log.event_time[: log.count].copy(), np.where(elongation > 0.0, np.maximum(taut, 0.0), 0.0)


def squall_drake_check(
    spec: MissionSpec,
    level: float = CHECK_LEVEL,
    seed: int = CHECK_SEED,
    duration: float = CHECK_DURATION,
    window: tuple[float, float] = CHECK_WINDOW,
) -> SquallCheck:
    """Constant squall pattern and its reverse from the steady tow (closed loop, no weather);
    tension deficits averaged over ``window``."""
    calibration = calibrate_squall(replace(spec, squall_level=level))
    amplitude = calibration.vessel_amplitude
    deficits = []
    for direction in (spec.squall_direction, spec.squall_direction + math.pi):
        time, tension = constant_squall_tension(spec, amplitude, direction, seed, duration)
        inside = (time >= window[0] - 1.0e-9) & (time <= window[1] + 1.0e-9)
        deficits.append(spec.pretension - tension[inside].mean(axis=0))
    odd = 0.5 * (deficits[0] - deficits[1])
    linear = amplitude * squall_response(spec).step_deficit(time[inside][::10]).mean(axis=0)
    predicted = amplitude * calibration.unit_deficit
    critical = calibration.critical_cable
    return SquallCheck(
        level=level,
        vessel_amplitude=amplitude,
        window=window,
        critical_cable=critical,
        predicted=predicted,
        linear=linear,
        measured=deficits[0],
        reversed=deficits[1],
        odd=odd,
        even=0.5 * (deficits[0] + deficits[1]),
        ratio=float(odd[critical] / predicted[critical]),
        one_sided_ratio=float(deficits[0][critical] / predicted[critical]),
        linear_ratio=float(odd[critical] / linear[critical]),
        static_ratio=float(odd[critical] / (amplitude * calibration.static_unit_deficit[critical])),
    )


@dataclass(frozen=True)
class MissionOutcome:
    seed: int
    end_time: float
    completed: bool
    closure: tuple[int, float] | None
    severances: tuple[tuple[int, float], ...]
    load_pose: np.ndarray
    docking_error: float
    mission_time: float
    mark_count: int


def mission_outcome(run: FleetRun, spec: MissionSpec, centre: np.ndarray) -> MissionOutcome:
    context = run.simulator.get_context()
    end = float(context.get_time())
    cables = run.fleet.cables
    state = plant_state(run.fleet, context)
    error = float(np.hypot(*(state[:2] - centre)))
    log = cables.log
    distance = np.hypot(*(log.state[: log.state_count, :2] - centre).T)
    inside = np.flatnonzero(distance <= spec.docking_radius)
    if inside.size:
        mission_time = float(log.state_time[inside[0]])
    else:
        mission_time = max(end, spec.duration) + max(0.0, error - spec.docking_radius) / nominal_speed(spec)
    return MissionOutcome(
        seed=run.master_seed,
        end_time=end,
        completed=cables.closure is None and end >= spec.duration - 1.0e-9,
        closure=None if cables.closure is None else (int(cables.closure[0]), float(cables.closure[1])),
        severances=tuple((int(cable), float(time)) for cable, time in cables.severances),
        load_pose=state[:3].copy(),
        docking_error=error,
        mission_time=mission_time,
        mark_count=len(cables.reengagements),
    )


def mission_record(run: FleetRun, spec: MissionSpec, outcome: MissionOutcome) -> dict[str, np.ndarray]:
    """Sensor samples, truth logs, marks, geometry, and outcome of one mission (for replay)."""
    arrays: dict[str, np.ndarray] = {}
    for agent, suite in enumerate(run.sensors):
        for channel, width in (("odometry", 4), ("bearing", 2), ("tension", 3)):
            arrays[f"{channel}_{agent}"] = np.asarray(suite.samples[channel], dtype=float).reshape(-1, width)
    arrays["beacon"] = np.asarray(run.sensors[0].samples["beacon"], dtype=float).reshape(-1, 4)
    cables = run.fleet.cables
    log = cables.log
    arrays.update(
        event_time=log.event_time[: log.count],
        elongation=log.elongation[: log.count],
        rate=log.rate[: log.count],
        relative_load=log.relative_load[: log.count],
        alive=log.alive[: log.count],
        state_time=log.state_time[: log.state_count],
        state=log.state[: log.state_count],
        initial_poses=initial_state(spec)[: 3 * (run.fleet.geometry.vessel_count + 1)].reshape(-1, 3),
        load_offsets=run.fleet.geometry.load_offsets,
        vessel_offsets=run.fleet.geometry.vessel_offsets,
        rest_length=np.array(cables.cable.rest_length),
        marks=np.array([[getattr(m, f) for f in MARK_FIELDS] for m in cables.reengagements], dtype=float).reshape(-1, len(MARK_FIELDS)),
        taut_intervals=np.array([[getattr(r, f) for f in TAUT_FIELDS] for r in cables.taut_intervals], dtype=float).reshape(-1, len(TAUT_FIELDS)),
        severances=np.array(outcome.severances, dtype=float).reshape(-1, 2),
        closure=np.array(outcome.closure if outcome.closure is not None else (np.nan, np.nan), dtype=float),
        end_time=np.array(outcome.end_time),
        completed=np.array(outcome.completed),
        load_pose=outcome.load_pose,
        docking_error=np.array(outcome.docking_error),
        mission_time=np.array(outcome.mission_time),
    )
    return arrays


def outcome_summary(outcome: MissionOutcome) -> dict:
    summary = asdict(outcome)
    summary["load_pose"] = outcome.load_pose.tolist()
    return summary


@dataclass(frozen=True)
class MissionJob:
    spec: MissionSpec
    seed: int
    centre: tuple[float, float]
    path: str

    @property
    def cost(self) -> float:
        return self.spec.duration


def run_mission_job(job: MissionJob) -> dict:
    run = run_mission(job.spec, job.seed)
    outcome = mission_outcome(run, job.spec, np.asarray(job.centre))
    path = Path(job.path)
    digest = write_bytes(path, npz_bytes(mission_record(run, job.spec, outcome)))
    return {"seed": job.seed, "record": relative(path), "sha256": digest, "outcome": outcome_summary(outcome), "wall_seconds": run.wall_seconds}


def run_campaign(
    spec: MissionSpec, seeds: tuple[int, ...], directory: Path, workers: int = DEFAULT_WORKERS
) -> dict:
    """Reference run, then one mission per seed (one process each); writes a manifest."""
    reference = docking_reference(spec)
    calibration = calibrate_squall(spec)
    jobs = [
        MissionJob(spec, seed, (float(reference.centre[0]), float(reference.centre[1])), str(directory / f"mission_{seed}.npz"))
        for seed in seeds
    ]
    results = run_pool(run_mission_job, jobs, workers)
    manifest = {
        "spec": asdict(spec),
        "config_hash": spec.config_hash(),
        "seeds": list(seeds),
        "reference": {"seed": reference.seed, "centre": reference.centre.tolist(), "heading": reference.heading, "config_hash": reference.config_hash},
        "squall": {
            "vessel_amplitude": calibration.vessel_amplitude,
            "load_amplitude": calibration.load_amplitude,
            "critical_cable": calibration.critical_cable,
            "unit_deficit": calibration.unit_deficit.tolist(),
            "static_unit_deficit": calibration.static_unit_deficit.tolist(),
        },
        "results": results,
        "source": source_state(),
    }
    write_bytes(directory / "manifest.json", json_bytes(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("reference", "check", "calibration"))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seeds", type=int, default=len(CALIBRATION_SEEDS))
    arguments = parser.parse_args()
    spec = MissionSpec()
    if arguments.command == "reference":
        reference = docking_reference(spec)
        print({"centre": reference.centre.tolist(), "heading_deg": math.degrees(reference.heading)})
    elif arguments.command == "check":
        check = squall_drake_check(spec)
        print({name: getattr(check, name) for name in ("vessel_amplitude", "ratio", "one_sided_ratio", "linear_ratio", "static_ratio")})
    else:
        manifest = run_campaign(spec, CALIBRATION_SEEDS[: arguments.seeds], CALIBRATION_DIR, arguments.workers)
        print(f"calibration campaign: {len(manifest['results'])} missions, reference centre {manifest['reference']['centre']}")


if __name__ == "__main__":
    main()
