"""Focused Phase 1 unilateral-force and event-state tests."""

from dataclasses import fields

import numpy as np
import pytest
from pydrake.math import RigidTransform
from pydrake.multibody.math import SpatialVelocity

from tether.physics.cable import (
    EVENT_FIELDS,
    REENGAGEMENT_FIELDS,
    TAUT_FIELDS,
    CableConfig,
    CableEventRecord,
    CableEventTracker,
    CableMode,
    FixedCapacityRing,
    ReengagementRecord,
    TautRecord,
    UnilateralCable,
)
from tether.physics.phase1_mechanics import run_two_body_impact


def _cable_context(elongation, elongation_rate, *, mode=CableMode.RECORDING):
    cable = UnilateralCable(
        CableConfig(
            load_body_index=0,
            vessel_body_index=1,
            load_offset=np.zeros(3),
            vessel_offset=np.zeros(3),
            mode=mode,
            break_threshold=1.0 if mode is CableMode.LIVE else None,
        )
    )
    context = cable.CreateDefaultContext()
    cable.get_input_port(0).FixValue(
        context,
        [
            RigidTransform(),
            RigidTransform([12.0 + elongation, 0.0, 0.0]),
        ],
    )
    cable.get_input_port(1).FixValue(
        context,
        [
            SpatialVelocity(w=np.zeros(3), v=np.zeros(3)),
            SpatialVelocity(w=np.zeros(3), v=[elongation_rate, 0.0, 0.0]),
        ],
    )
    return cable, context


@pytest.mark.parametrize(
    ("elongation", "rate", "engaged", "force_positive", "expected_tension"),
    [
        (-0.01, 10.0, False, False, 0.0),
        (0.01, -10.0, True, False, 0.0),
        (0.01, 1.0, True, True, 3500.0),
    ],
)
def test_unilateral_force_gating_distinguishes_geometry_and_force(
    elongation, rate, engaged, force_positive, expected_tension
):
    cable, context = _cable_context(elongation, rate)
    sample = cable.sample(context)
    forces = cable.get_output_port().Eval(context)

    assert sample.geometrically_engaged is engaged
    assert sample.force_positive is force_positive
    assert sample.applied_tension == pytest.approx(expected_tension)
    assert forces[0].F_Bq_W.translational()[0] == pytest.approx(expected_tension)
    assert forces[1].F_Bq_W.translational()[0] == pytest.approx(-expected_tension)


def test_live_cable_force_is_zero_when_not_alive():
    cable, context = _cable_context(0.01, 1.0, mode=CableMode.LIVE)
    cable.set_alive(context, False)
    sample = cable.sample(context)
    assert sample.geometrically_engaged
    assert not sample.force_positive
    assert sample.applied_tension == 0.0


def _completed_excursion_tracker(mode=CableMode.RECORDING, threshold=None):
    tracker = CableEventTracker(
        stiffness=100.0,
        damping=10.0,
        cable=4,
        capacity=8,
        mode=mode,
        break_threshold=threshold,
    )
    samples = (
        (0.000, 0.10, -1.0, 0.0, 2.0),
        (0.001, -0.10, -2.0, -30.0, 4.0),
        (0.002, -0.20, -1.0, -30.0, 6.0),
        (0.003, -0.10, 1.0, 0.0, 5.0),
        (0.004, 0.10, 2.0, 80.0, 3.0),
        (0.005, 0.08, -1.0, 100.0, 2.0),
        (0.006, -0.02, -2.0, -22.0, 1.0),
    )
    severed = False
    for sample in samples:
        severed = tracker.sample(*sample) or severed
    return tracker, severed


def test_event_schemas_interpolation_and_state_transitions():
    tracker, severed = _completed_excursion_tracker()
    assert tuple(field.name for field in fields(ReengagementRecord)) == REENGAGEMENT_FIELDS
    assert tuple(field.name for field in fields(TautRecord)) == TAUT_FIELDS
    assert tuple(field.name for field in fields(CableEventRecord)) == EVENT_FIELDS
    assert not severed

    mark = tracker.state.reengagement_ring.snapshot()[0]
    assert mark.t_up == pytest.approx(0.0035)
    assert mark.u_entry == pytest.approx(1.5)
    assert mark.depth == pytest.approx(0.2)
    assert mark.dwell == pytest.approx(0.003)
    assert mark.v_return == pytest.approx(1.5)
    assert mark.T_peak == pytest.approx(100.0)
    assert mark.n_maxima == 1
    assert mark.cable == 4
    assert mark.W_rel_at_onset == pytest.approx(3.0)
    assert mark.W_rel_max == pytest.approx(6.0)

    taut = tracker.state.taut_ring.snapshot()[-1]
    assert taut.t_start == pytest.approx(0.0035)
    assert taut.t_end == pytest.approx(0.0058)
    assert taut.q_peak == pytest.approx(100.0)

    events = tracker.state.event_ring.snapshot()
    assert [event.kind for event in events] == [
        "geometric_down",
        "turning",
        "geometric_up",
        "force_onset",
        "turning",
        "geometric_down",
        "force_cessation",
    ]
    assert events[0].time == pytest.approx(0.0005)
    assert events[2].time == pytest.approx(0.0035)
    assert events[3].time == pytest.approx(0.0035)
    assert events[-1].time == pytest.approx(0.0058)


def test_completed_peak_severs_only_in_live_mode():
    recording, recording_severed = _completed_excursion_tracker(
        CableMode.RECORDING, 90.0
    )
    live, live_severed = _completed_excursion_tracker(CableMode.LIVE, 90.0)
    assert recording.state.taut_ring.snapshot() == live.state.taut_ring.snapshot()
    assert not recording_severed
    assert live_severed
    rupture = [
        event
        for event in live.state.event_ring.snapshot()
        if event.kind == "rupture"
    ]
    assert len(rupture) == 1
    assert rupture[0].time == pytest.approx(0.0045)
    assert rupture[0].tension == pytest.approx(90.0)


def test_fixed_capacity_ring_order_and_drain_are_deterministic():
    first = FixedCapacityRing.create(2)
    second = FixedCapacityRing.create(2)
    for ring in (first, second):
        ring.append("a")
        ring.append("b")
        ring.append("c")
    assert first.snapshot() == second.snapshot() == ("b", "c")
    assert first.dropped == second.dropped == 1
    assert first.drain() == second.drain() == ("b", "c")
    assert first.snapshot() == second.snapshot() == ()


def test_periodic_event_state_drains_at_one_second():
    first = run_two_body_impact(1.0, 1800.0, duration=1.001)
    second = run_two_body_impact(1.0, 1800.0, duration=1.001)
    assert first.taut_records == second.taut_records
    assert len(first.taut_records) == 1
    assert first.taut_records[0].q_peak > 0.0
    assert first.drained_taut_record_count == 1
    assert first.resident_taut_record_count == 0


def test_live_mode_changes_alive_only_after_completed_taut_peak():
    trajectory = run_two_body_impact(
        1.0,
        1800.0,
        mode=CableMode.LIVE,
        break_threshold=1000.0,
    )
    peak_index = int(np.argmax(trajectory.tension))
    assert trajectory.alive[peak_index] == 1.0
    assert trajectory.alive[-1] == 0.0