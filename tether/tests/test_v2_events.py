"""Unit tests for the v2 event machinery (plan v2 IV.9, Appendix B.1, B.3)."""

import math

import numpy as np
import pytest

from tether.campaign.v2 import events as ev
from tether.physics.events import linear_crossing_time


def test_engagement_period_is_the_plans_0335_s():
    assert ev.ENGAGEMENT_PERIOD == pytest.approx(0.335, abs=1.0e-3)
    assert ev.BURST_WINDOW == 2.0 and ev.EXCLUSION_WINDOW == 3.0


def test_crossings_interpolate_like_the_tracker_and_respect_alive():
    time = np.arange(0.0, 0.010, 0.001)
    e = np.array([[0.3, 0.1, -0.2, -0.4, -0.1, 0.2, 0.5, 0.1, -0.3, -0.5]]).T
    e = np.hstack([e, np.full_like(e, 0.1)])
    ups, downs = ev.crossings(time, e)
    assert downs.size == 2 and ups.size == 1
    assert downs.time[0] == linear_crossing_time(time[1], 0.1, time[2], -0.2)
    assert ups.time[0] == linear_crossing_time(time[4], -0.1, time[5], 0.2)
    assert ups.index[0] == 5 and downs.index[0] == 2
    assert np.all(downs.cable == 0)
    alive = np.ones_like(e, dtype=bool)
    alive[6:, 0] = False  # severed: the later down-crossing is not a slack onset
    _, downs_alive = ev.crossings(time, e, alive)
    assert downs_alive.size == 1


def test_zero_is_slack_as_in_the_tracker():
    time = np.arange(4) * 1.0e-3
    e = np.array([[0.1], [0.0], [0.0], [0.1]])
    ups, downs = ev.crossings(time, e)
    assert downs.size == 1 and ups.size == 1
    assert downs.time[0] == pytest.approx(time[1])


def test_anchored_declustering_roles_and_absorption():
    t = np.array([0.0, 0.2, 1.0, 1.9, 2.0, 2.1, 2.3, 5.0])
    c = np.zeros(t.size, dtype=int)
    result = ev.decluster_marks(t, c)
    # anchored at 0.0: 0.2, 1.0, 1.9 and 2.0 (inclusive) belong; 2.1 opens a new event
    assert result.event_count == 3
    assert list(result.event_id) == [0, 0, 0, 0, 0, 1, 1, 2]
    assert result.role_names() == ["parent", "bounce", "member", "member", "bounce", "parent", "bounce", "parent"]
    assert list(result.event_marks) == [5, 2, 1]
    assert list(result.event_bounces) == [2, 1, 0]
    assert result.event_marks.sum() == t.size  # nothing dropped
    assert list(result.parent_index) == [0, 0, 0, 0, 0, 5, 5, 7]
    assert math.isnan(result.gap_previous[0]) and result.gap_previous[1] == pytest.approx(0.2)


def test_chained_rule_is_a_different_reading():
    t = np.array([0.0, 1.5, 3.0, 4.5, 7.0])
    c = np.zeros(t.size, dtype=int)
    assert ev.decluster_marks(t, c).event_count == 3  # {0, 1.5}, {3.0, 4.5}, {7.0}
    assert ev.decluster_marks(t, c, rule="chained").event_count == 2  # {0 ... 4.5}, {7.0}
    with pytest.raises(ValueError):
        ev.decluster_marks(t, c, rule="other")


def test_declustering_is_per_cable_and_input_order_free():
    t = np.array([0.5, 0.0, 0.1, 0.6, 3.0])
    c = np.array([1, 0, 1, 0, 1])
    result = ev.decluster_marks(t, c)
    # cable 0: {0.0, 0.6}; cable 1: {0.1, 0.5}, {3.0}; events ordered by time
    assert result.event_count == 3
    assert list(result.event_time) == [0.0, 0.1, 3.0]
    assert list(result.event_cable) == [0, 1, 1]
    assert result.event_id[1] == result.event_id[3] == 0
    assert result.event_id[2] == result.event_id[0] == 1
    assert result.role[0] == ev.ROLE_CODES["member"]  # gap 0.4 s > one engagement period
    maxima, argmax = ev.event_cluster_maxima(np.array([5.0, 1.0, 2.0, 7.0, 3.0]), result)
    assert list(maxima) == [7.0, 5.0, 3.0] and list(argmax) == [3, 0, 4]


def test_empty_marks():
    result = ev.decluster_marks(np.empty(0), np.empty(0, dtype=int))
    assert result.event_count == 0 and result.event_id.size == 0


def test_union_measure_and_exposure():
    assert ev.union_measure(np.array([0.0, 1.0, 10.0]), 3.0, 0.0, 20.0) == pytest.approx(7.0)
    assert ev.union_measure(np.array([-2.0, 18.5]), 3.0, 0.0, 20.0) == pytest.approx(2.5)
    assert ev.union_measure(np.empty(0), 3.0, 0.0, 20.0) == 0.0
    assert ev.excluded_exposure(np.array([5.0, 6.0]), 0.0, 20.0) == pytest.approx(16.0)
    assert ev.excluded_exposure(np.array([5.0]), 10.0, 10.0) == 0.0


def test_exclusion_windows_open_and_closed_ends():
    ups = np.array([10.0])
    times = np.array([9.999, 10.0, 11.0, 13.0, 13.001])
    # B.3: (t_up, t_up + 3]
    assert list(ev.in_exclusion(times, ups)) == [False, False, True, True, False]
    # B.1 primary rule, t excluded iff t_up in (t - 3, t]: [t_up, t_up + 3)
    assert list(ev.in_exclusion(times, ups, closed_left=True)) == [False, True, True, False, False]


def test_primary_onsets_any_cable_and_parent():
    up_t = np.array([1.0, 4.0, 4.5])
    up_c = np.array([0, 2, 1])
    onset_t = np.array([0.5, 1.0, 3.9, 4.0, 7.49, 7.5, 10.0])
    onset_c = np.array([0, 3, 1, 1, 1, 0, 4])
    result = ev.primary_onsets(onset_t, onset_c, up_t, up_c)
    assert list(result.primary) == [True, False, False, False, False, True, True]
    assert list(result.parent) == [-1, 0, 0, 1, 2, -1, -1]
    assert result.parent_lag[1] == 0.0 and result.parent_lag[4] == pytest.approx(2.99)
    assert list(result.parent_same_cable) == [False, False, False, False, True, False, False]
    # the 2 s same-cable variant: only the onset's own cable counts
    same = ev.primary_onsets(np.array([4.6, 4.6, 6.6]), np.array([1, 2, 1]), up_t, up_c, window=2.0, same_cable_only=True)
    assert list(same.primary) == [False, False, True]
    assert list(same.parent) == [2, 1, -1]
    assert same.parent_same_cable[:2].all()


def test_primary_consistent_with_exclusion_exposure():
    rng = np.random.default_rng(3)
    ups = np.sort(rng.uniform(0.0, 100.0, 40))
    onsets = np.sort(rng.uniform(0.0, 100.0, 400))
    flags = ev.primary_onsets(onsets, np.zeros(onsets.size, dtype=int), ups, np.ones(ups.size, dtype=int)).primary
    assert np.array_equal(flags, ~ev.in_exclusion(onsets, ups, closed_left=True))
    grid = np.linspace(0.0, 100.0, 200_001)
    covered = ev.in_exclusion(grid, ups).mean() * 100.0
    assert ev.excluded_exposure(ups, 0.0, 100.0) == pytest.approx(100.0 - covered, abs=5.0e-2)


def test_clean_mask_is_fleet_wide_with_no_all_taut_clause():
    time = np.arange(0.0, 10.0, 0.5)
    e = np.full((time.size, 3), 0.01)
    e[4, 1] = -0.01  # cable 1 slack at t = 2.0: does not remove the other cables' samples by itself
    window = time >= 1.0
    ups = np.array([2.25])  # one re-engagement, on cable 1
    clean = ev.clean_mask(time, e, ups, window)
    excluded_times = (time > 2.25) & (time <= 5.25)
    for cable in range(3):
        assert not clean[excluded_times, cable].any()  # every cable, not only cable 1
        assert clean[(time >= 1.0) & ~excluded_times & (e[:, cable] > 0), cable].all()
    assert not clean[4, 1]  # slack sample
    assert not clean[0].any()  # outside the recorded window


def test_match_marks():
    ups = ev.Crossings(np.array([1.0, 1.0, 2.0]), np.array([0, 1, 0]), np.array([10, 10, 20]))
    assert list(ev.match_marks(np.array([2.0, 1.0, 1.0, 3.0]), np.array([0, 1, 0, 0]), ups)) == [2, 1, 0, -1]
