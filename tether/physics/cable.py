"""Unilateral cable forces for Drake multibody plants."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.multibody.math import SpatialForce, SpatialVelocity
from pydrake.multibody.plant import ExternallyAppliedSpatialForce
from pydrake.multibody.tree import BodyIndex
from pydrake.systems.framework import Context, LeafSystem

from tether.physics import constants
from tether.physics.events import linear_crossing_speed, linear_crossing_time

EVENT_PERIOD = 1.0e-3
DRAIN_PERIOD = 1.0
REENGAGEMENT_FIELDS = (
    "t_up",
    "u_entry",
    "depth",
    "dwell",
    "v_return",
    "T_peak",
    "n_maxima",
    "cable",
    "W_rel_at_onset",
    "W_rel_max",
)
TAUT_FIELDS = ("t_start", "t_end", "q_peak", "cable")
EVENT_FIELDS = (
    "kind",
    "time",
    "cable",
    "elongation",
    "elongation_rate",
    "tension",
    "relative_load",
)
Record = TypeVar("Record")


class CableMode(str, Enum):
    RECORDING = "recording"
    LIVE = "live"


class CableEventKind(str, Enum):
    GEOMETRIC_DOWN = "geometric_down"
    GEOMETRIC_UP = "geometric_up"
    FORCE_ONSET = "force_onset"
    FORCE_CESSATION = "force_cessation"
    TURNING = "turning"
    RUPTURE = "rupture"


@dataclass(frozen=True)
class CableConfig:
    load_body_index: int
    vessel_body_index: int
    load_offset: np.ndarray
    vessel_offset: np.ndarray
    rest_length: float = constants.CABLE_REST_LENGTH
    stiffness: float = constants.CABLE_STIFFNESS
    damping: float = constants.CABLE_DAMPING
    mode: CableMode = CableMode.RECORDING
    break_threshold: float | None = None
    cable: int = 0
    ring_capacity: int = 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", CableMode(self.mode))
        object.__setattr__(self, "load_offset", np.asarray(self.load_offset, dtype=float).copy())
        object.__setattr__(
            self, "vessel_offset", np.asarray(self.vessel_offset, dtype=float).copy()
        )
        if self.load_offset.shape != (3,) or self.vessel_offset.shape != (3,):
            raise ValueError("cable attachment offsets must be three-vectors")
        if self.rest_length <= 0.0 or self.stiffness <= 0.0 or self.damping < 0.0:
            raise ValueError("cable length and stiffness must be positive; damping cannot be negative")
        if self.mode is CableMode.LIVE and self.break_threshold is None:
            raise ValueError("live cable mode requires a break threshold")
        if self.break_threshold is not None and self.break_threshold <= 0.0:
            raise ValueError("break threshold must be positive")
        if self.ring_capacity <= 0:
            raise ValueError("ring capacity must be positive")


@dataclass(frozen=True)
class ReengagementRecord:
    t_up: float
    u_entry: float
    depth: float
    dwell: float
    v_return: float
    T_peak: float
    n_maxima: int
    cable: int
    W_rel_at_onset: float
    W_rel_max: float


@dataclass(frozen=True)
class TautRecord:
    t_start: float
    t_end: float
    q_peak: float
    cable: int


@dataclass(frozen=True)
class CableEventRecord:
    kind: str
    time: float
    cable: int
    elongation: float
    elongation_rate: float
    tension: float
    relative_load: float


@dataclass
class FixedCapacityRing(Generic[Record]):
    capacity: int
    values: list[Record | None]
    start: int = 0
    size: int = 0
    dropped: int = 0

    @classmethod
    def create(cls, capacity: int) -> FixedCapacityRing:
        if capacity <= 0:
            raise ValueError("ring capacity must be positive")
        return cls(capacity=capacity, values=[None] * capacity)

    def append(self, record: Record) -> None:
        if self.size < self.capacity:
            index = (self.start + self.size) % self.capacity
            self.size += 1
        else:
            index = self.start
            self.start = (self.start + 1) % self.capacity
            self.dropped += 1
        self.values[index] = record

    def snapshot(self) -> tuple[Record, ...]:
        records = []
        for offset in range(self.size):
            record = self.values[(self.start + offset) % self.capacity]
            if record is not None:
                records.append(record)
        return tuple(records)

    def drain(self) -> tuple[Record, ...]:
        records = self.snapshot()
        self.values = [None] * self.capacity
        self.start = 0
        self.size = 0
        return records


@dataclass(frozen=True)
class _PendingReengagement:
    t_up: float
    u_entry: float
    depth: float
    dwell: float
    v_return: float
    n_maxima: int
    W_rel_at_onset: float
    W_rel_max: float


@dataclass
class CableEventState:
    reengagement_ring: FixedCapacityRing[ReengagementRecord]
    taut_ring: FixedCapacityRing[TautRecord]
    event_ring: FixedCapacityRing[CableEventRecord]
    initialized: bool = False
    previous_time: float = 0.0
    previous_elongation: float = 0.0
    previous_rate: float = 0.0
    previous_taut_tension: float = 0.0
    previous_relative_load: float = 0.0
    taut_start: float | None = None
    taut_peak: float = 0.0
    slack_start: float | None = None
    slack_entry_speed: float = 0.0
    slack_depth: float = 0.0
    slack_maxima: int = 0
    slack_relative_load_onset: float = 0.0
    slack_relative_load_max: float = 0.0
    slack_boundary_censored: bool = False
    pending_reengagement: _PendingReengagement | None = None
    boundary_censored_taut_intervals: int = 0
    boundary_censored_slack_intervals: int = 0
    next_drain_time: float = DRAIN_PERIOD

    @classmethod
    def create(cls, capacity: int) -> CableEventState:
        return cls(
            reengagement_ring=FixedCapacityRing.create(capacity),
            taut_ring=FixedCapacityRing.create(capacity),
            event_ring=FixedCapacityRing.create(capacity),
        )


class CableEventTracker:
    """Deterministic sampled state machine for one cable's event marks."""

    def __init__(
        self,
        *,
        stiffness: float,
        damping: float,
        cable: int,
        capacity: int,
        mode: CableMode,
        break_threshold: float | None,
    ) -> None:
        self.stiffness = stiffness
        self.damping = damping
        self.cable = cable
        self.mode = CableMode(mode)
        self.break_threshold = break_threshold
        self.state = CableEventState.create(capacity)

    @staticmethod
    def _interpolate_value(
        first_value: float,
        second_value: float,
        first_sample: float,
        second_sample: float,
    ) -> float:
        fraction = -first_value / (second_value - first_value)
        return first_sample + fraction * (second_sample - first_sample)

    def _complete_reengagement_peak(self, q_peak: float) -> bool:
        pending = self.state.pending_reengagement
        if pending is None:
            return False
        self.state.reengagement_ring.append(
            ReengagementRecord(
                t_up=pending.t_up,
                u_entry=pending.u_entry,
                depth=pending.depth,
                dwell=pending.dwell,
                v_return=pending.v_return,
                T_peak=q_peak,
                n_maxima=pending.n_maxima,
                cable=self.cable,
                W_rel_at_onset=pending.W_rel_at_onset,
                W_rel_max=pending.W_rel_max,
            )
        )
        self.state.pending_reengagement = None
        return False

    def _append_event(
        self,
        kind: CableEventKind,
        time: float,
        elongation: float,
        elongation_rate: float,
        tension: float,
        relative_load: float,
    ) -> None:
        self.state.event_ring.append(
            CableEventRecord(
                kind=kind.value,
                time=float(time),
                cable=self.cable,
                elongation=float(elongation),
                elongation_rate=float(elongation_rate),
                tension=float(tension),
                relative_load=float(relative_load),
            )
        )

    def _interpolated_sample(
        self,
        time: float,
        elongation: float,
        elongation_rate: float,
        taut_tension: float,
        relative_load: float,
        event_time: float,
    ) -> tuple[float, float, float, float]:
        state = self.state
        fraction = (event_time - state.previous_time) / (time - state.previous_time)
        return (
            state.previous_elongation
            + fraction * (elongation - state.previous_elongation),
            state.previous_rate + fraction * (elongation_rate - state.previous_rate),
            state.previous_taut_tension
            + fraction * (taut_tension - state.previous_taut_tension),
            state.previous_relative_load
            + fraction * (relative_load - state.previous_relative_load),
        )

    def sample(
        self,
        time: float,
        elongation: float,
        elongation_rate: float,
        taut_tension: float,
        relative_load: float = 0.0,
    ) -> bool:
        """Advance one sample and return whether a completed peak severs the cable."""
        state = self.state
        if not state.initialized:
            state.initialized = True
            state.previous_time = time
            state.previous_elongation = elongation
            state.previous_rate = elongation_rate
            state.previous_taut_tension = taut_tension
            state.previous_relative_load = relative_load
            if elongation > 0.0:
                state.taut_start = time
                state.taut_peak = max(0.0, taut_tension)
                state.boundary_censored_taut_intervals += 1
            else:
                state.slack_start = time
                state.slack_entry_speed = max(0.0, -elongation_rate)
                state.slack_depth = max(0.0, -elongation)
                state.slack_relative_load_onset = relative_load
                state.slack_relative_load_max = relative_load
                state.slack_boundary_censored = True
                state.boundary_censored_slack_intervals += 1
            return False

        sever = False
        crossed_up = state.previous_elongation <= 0.0 < elongation
        crossed_down = state.previous_elongation > 0.0 >= elongation
        current_force = max(0.0, taut_tension) if elongation > 0.0 else 0.0
        previous_force = (
            max(0.0, state.previous_taut_tension)
            if state.previous_elongation > 0.0
            else 0.0
        )
        if crossed_up:
            crossing_time = linear_crossing_time(
                state.previous_time,
                state.previous_elongation,
                time,
                elongation,
            )
            crossing_rate = linear_crossing_speed(
                state.previous_elongation,
                elongation,
                state.previous_rate,
                elongation_rate,
            )
            crossing_load = self._interpolate_value(
                state.previous_elongation,
                elongation,
                state.previous_relative_load,
                relative_load,
            )
            self._append_event(
                CableEventKind.GEOMETRIC_UP,
                crossing_time,
                0.0,
                crossing_rate,
                0.0,
                crossing_load,
            )
            if state.slack_start is not None and not state.slack_boundary_censored:
                state.pending_reengagement = _PendingReengagement(
                    t_up=crossing_time,
                    u_entry=state.slack_entry_speed,
                    depth=state.slack_depth,
                    dwell=crossing_time - state.slack_start,
                    v_return=max(0.0, crossing_rate),
                    n_maxima=state.slack_maxima,
                    W_rel_at_onset=state.slack_relative_load_onset,
                    W_rel_max=state.slack_relative_load_max,
                )
            state.taut_start = crossing_time
            state.taut_peak = max(
                self.damping * crossing_rate,
                taut_tension,
            )
            state.slack_start = None
            state.slack_boundary_censored = False
        elif crossed_down:
            crossing_time = linear_crossing_time(
                state.previous_time,
                state.previous_elongation,
                time,
                elongation,
            )
            crossing_rate = linear_crossing_speed(
                state.previous_elongation,
                elongation,
                state.previous_rate,
                elongation_rate,
            )
            crossing_load = self._interpolate_value(
                state.previous_elongation,
                elongation,
                state.previous_relative_load,
                relative_load,
            )
            self._append_event(
                CableEventKind.GEOMETRIC_DOWN,
                crossing_time,
                0.0,
                crossing_rate,
                0.0,
                crossing_load,
            )
            q_peak = max(state.taut_peak, taut_tension)
            if state.taut_start is not None:
                state.taut_ring.append(
                    TautRecord(
                        t_start=state.taut_start,
                        t_end=crossing_time,
                        q_peak=q_peak,
                        cable=self.cable,
                    )
                )
            sever = self._complete_reengagement_peak(q_peak)
            state.taut_start = None
            state.taut_peak = 0.0
            state.slack_start = crossing_time
            state.slack_entry_speed = max(0.0, -crossing_rate)
            state.slack_depth = max(0.0, -elongation)
            state.slack_maxima = 0
            state.slack_relative_load_onset = crossing_load
            state.slack_relative_load_max = max(crossing_load, relative_load)
            state.slack_boundary_censored = False
        elif elongation > 0.0:
            if taut_tension < state.taut_peak:
                sever = self._complete_reengagement_peak(state.taut_peak)
            state.taut_peak = max(state.taut_peak, taut_tension)
        else:
            state.slack_depth = max(state.slack_depth, -elongation)
            state.slack_relative_load_max = max(state.slack_relative_load_max, relative_load)
            if state.previous_rate < 0.0 <= elongation_rate:
                state.slack_maxima += 1

        previous_positive = previous_force > 0.0
        current_positive = current_force > 0.0
        if previous_positive != current_positive:
            boundary_times = []
            if (state.previous_elongation <= 0.0 < elongation) or (
                state.previous_elongation > 0.0 >= elongation
            ):
                boundary_times.append(
                    linear_crossing_time(
                        state.previous_time,
                        state.previous_elongation,
                        time,
                        elongation,
                    )
                )
            if (state.previous_taut_tension <= 0.0 < taut_tension) or (
                state.previous_taut_tension > 0.0 >= taut_tension
            ):
                boundary_times.append(
                    linear_crossing_time(
                        state.previous_time,
                        state.previous_taut_tension,
                        time,
                        taut_tension,
                    )
                )
            event_time = (
                max(boundary_times) if current_positive else min(boundary_times)
            )
            event_sample = self._interpolated_sample(
                time,
                elongation,
                elongation_rate,
                taut_tension,
                relative_load,
                event_time,
            )
            self._append_event(
                CableEventKind.FORCE_ONSET
                if current_positive
                else CableEventKind.FORCE_CESSATION,
                event_time,
                *event_sample,
            )

        crossed_turning = (state.previous_rate < 0.0 <= elongation_rate) or (
            state.previous_rate > 0.0 >= elongation_rate
        )
        if crossed_turning:
            event_time = linear_crossing_time(
                state.previous_time,
                state.previous_rate,
                time,
                elongation_rate,
            )
            event_sample = self._interpolated_sample(
                time,
                elongation,
                elongation_rate,
                taut_tension,
                relative_load,
                event_time,
            )
            self._append_event(
                CableEventKind.TURNING,
                event_time,
                *event_sample,
            )

        if (
            self.mode is CableMode.LIVE
            and self.break_threshold is not None
            and previous_force < self.break_threshold <= current_force
        ):
            event_time = self._interpolate_value(
                previous_force - self.break_threshold,
                current_force - self.break_threshold,
                state.previous_time,
                time,
            )
            event_sample = self._interpolated_sample(
                time,
                elongation,
                elongation_rate,
                taut_tension,
                relative_load,
                event_time,
            )
            self._append_event(
                CableEventKind.RUPTURE,
                event_time,
                event_sample[0],
                event_sample[1],
                self.break_threshold,
                event_sample[3],
            )
            sever = True

        state.previous_time = time
        state.previous_elongation = elongation
        state.previous_rate = elongation_rate
        state.previous_taut_tension = taut_tension
        state.previous_relative_load = relative_load
        return sever

    def drain_if_due(
        self, time: float
    ) -> tuple[
        tuple[ReengagementRecord, ...],
        tuple[TautRecord, ...],
        tuple[CableEventRecord, ...],
    ]:
        if time + 1.0e-12 < self.state.next_drain_time:
            return (), (), ()
        reengagement = self.state.reengagement_ring.drain()
        taut = self.state.taut_ring.drain()
        events = self.state.event_ring.drain()
        while self.state.next_drain_time <= time + 1.0e-12:
            self.state.next_drain_time += DRAIN_PERIOD
        return reengagement, taut, events


@dataclass(frozen=True)
class CableSample:
    elongation: float
    elongation_rate: float
    taut_tension: float
    applied_tension: float
    geometrically_engaged: bool
    force_positive: bool
    alive: bool
    direction: np.ndarray


class UnilateralCable(LeafSystem):
    """Apply an explicit spring-damper cable force at two attachment points."""

    def __init__(self, config: CableConfig) -> None:
        super().__init__()
        self.config = config
        self._poses_port = self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self._velocities_port = self.DeclareAbstractInputPort(
            "body_spatial_velocities", AbstractValue.Make([SpatialVelocity()])
        )
        self._relative_load_port = self.DeclareVectorInputPort("relative_load", 1)
        self.DeclareDiscreteState(np.array([1.0]))
        tracker = CableEventTracker(
            stiffness=config.stiffness,
            damping=config.damping,
            cable=config.cable,
            capacity=config.ring_capacity,
            mode=config.mode,
            break_threshold=config.break_threshold,
        )
        self._event_state_index = self.DeclareAbstractState(AbstractValue.Make(tracker))
        self._drained_reengagement_records: list[ReengagementRecord] = []
        self._drained_taut_records: list[TautRecord] = []
        self._drained_event_records: list[CableEventRecord] = []
        self.DeclarePeriodicUnrestrictedUpdateEvent(
            EVENT_PERIOD, 0.0, self._update_events
        )
        self.DeclareAbstractOutputPort(
            "applied_spatial_forces",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self._calc_forces,
        )

    @staticmethod
    def _point_kinematics(
        pose: RigidTransform,
        velocity: SpatialVelocity,
        offset_body: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        offset_world = pose.rotation().multiply(offset_body)
        position = pose.translation() + offset_world
        point_velocity = velocity.translational() + np.cross(
            velocity.rotational(), offset_world
        )
        return position, point_velocity

    @staticmethod
    def _make_force(
        body_index: int,
        offset_body: np.ndarray,
        force_world: np.ndarray,
    ) -> ExternallyAppliedSpatialForce:
        applied = ExternallyAppliedSpatialForce()
        applied.body_index = BodyIndex(body_index)
        applied.p_BoBq_B = offset_body
        applied.F_Bq_W = SpatialForce(tau=np.zeros(3), f=force_world)
        return applied

    def is_alive(self, context: Context) -> bool:
        return bool(context.get_discrete_state_vector().GetAtIndex(0) > 0.5)

    def set_alive(self, context: Context, alive: bool) -> None:
        if self.config.mode is CableMode.RECORDING and not alive:
            raise ValueError("recording-mode cables cannot be severed")
        context.get_mutable_discrete_state_vector().SetAtIndex(0, float(alive))

    @property
    def drained_reengagement_records(self) -> tuple[ReengagementRecord, ...]:
        return tuple(self._drained_reengagement_records)

    @property
    def drained_taut_records(self) -> tuple[TautRecord, ...]:
        return tuple(self._drained_taut_records)

    @property
    def drained_event_records(self) -> tuple[CableEventRecord, ...]:
        return tuple(self._drained_event_records)

    def event_state(self, context: Context) -> CableEventState:
        tracker = context.get_abstract_state(int(self._event_state_index)).get_value()
        return copy.deepcopy(tracker.state)

    def _relative_load(self, context: Context) -> float:
        if not self._relative_load_port.HasValue(context):
            return 0.0
        return float(self._relative_load_port.Eval(context)[0])

    def _update_events(self, context: Context, state) -> None:
        tracker = copy.deepcopy(
            context.get_abstract_state(int(self._event_state_index)).get_value()
        )
        sample = self.sample(context)
        sever = tracker.sample(
            context.get_time(),
            sample.elongation,
            sample.elongation_rate,
            sample.taut_tension,
            self._relative_load(context),
        )
        reengagement, taut, events = tracker.drain_if_due(context.get_time())
        self._drained_reengagement_records.extend(reengagement)
        self._drained_taut_records.extend(taut)
        self._drained_event_records.extend(events)
        state.get_mutable_abstract_state(int(self._event_state_index)).set_value(tracker)
        if sever:
            state.get_mutable_discrete_state(0).SetAtIndex(0, 0.0)

    def sample(self, context: Context) -> CableSample:
        poses = self._poses_port.Eval(context)
        velocities = self._velocities_port.Eval(context)
        load_position, load_velocity = self._point_kinematics(
            poses[self.config.load_body_index],
            velocities[self.config.load_body_index],
            self.config.load_offset,
        )
        vessel_position, vessel_velocity = self._point_kinematics(
            poses[self.config.vessel_body_index],
            velocities[self.config.vessel_body_index],
            self.config.vessel_offset,
        )
        displacement = vessel_position - load_position
        length = float(np.linalg.norm(displacement))
        if length == 0.0:
            raise RuntimeError("cable attachment points are coincident")
        direction = displacement / length
        elongation_rate = float(np.dot(vessel_velocity - load_velocity, direction))
        elongation = length - self.config.rest_length
        taut_tension = self.config.stiffness * elongation + self.config.damping * elongation_rate
        engaged = elongation > 0.0
        alive = self.is_alive(context)
        applied_tension = max(0.0, taut_tension) if engaged and alive else 0.0
        return CableSample(
            elongation=elongation,
            elongation_rate=elongation_rate,
            taut_tension=taut_tension,
            applied_tension=applied_tension,
            geometrically_engaged=engaged,
            force_positive=applied_tension > 0.0,
            alive=alive,
            direction=direction,
        )

    def _calc_forces(self, context: Context, output: AbstractValue) -> None:
        sample = self.sample(context)
        force = sample.applied_tension * sample.direction
        output.set_value(
            [
                self._make_force(
                    self.config.load_body_index,
                    self.config.load_offset,
                    force,
                ),
                self._make_force(
                    self.config.vessel_body_index,
                    self.config.vessel_offset,
                    -force,
                ),
            ]
        )