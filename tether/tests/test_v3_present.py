"""Plan v3 WP4: the intervention clip's helpers (Drake-free).

The clip draws the fleet from a state-capturing copy of ``intervention.integrate``; the copy must return the
declared integrator's outputs bit-exactly (the clip asserts it on the real parent too, but this test catches a
divergence at edit time, on both branches, with weather driving the tow).
"""

from __future__ import annotations

import numpy as np

from tether.campaign.v3 import intervention as IV
from tether.tests.test_v3_intervention import _window

clip = __import__("tether.analysis.v3.present.clip_intervention", fromlist=["x"])


def test_state_capturing_copy_equals_the_declared_integrator_on_both_branches():
    x = _window(t_up=10.5, weather_scale=250.0)
    for disable in (None, 2):
        a = IV.integrate(x, disable, "recorded", window=1.5, peak_window=1.0)
        b = clip._integrate_with_state(x, disable, window=1.5, peak_window=1.0)
        assert clip._same_branch(a, b)
        assert b["poses"].shape == (b["e_10ms"].shape[0], 18) and b["rates"].shape == b["e_10ms"].shape
        # the poses are the integrator's: the recomputed chord elongation from the captured poses equals e_10ms
        for k in (0, b["poses"].shape[0] // 2, b["poses"].shape[0] - 1):
            row = b["poses"][k]
            load, vessels = row[:3], row[3:].reshape(5, 3)
            la = IV._rotate(load[2], x.load_offsets)
            stern = IV._rotate(vessels[:, 2], x.vessel_offsets)
            length = np.linalg.norm(vessels[:, :2] + stern - load[:2] - la, axis=1)
            assert np.allclose(length - x.rest_length, b["e_10ms"][k], atol=1e-12)


def test_copy_differs_from_integrator_when_a_branch_is_perturbed():
    """The equality check is not vacuous: a different disable choice is detected."""
    x = _window(t_up=10.5, weather_scale=250.0)
    a = IV.integrate(x, None, "recorded", window=1.5, peak_window=1.0)
    b = clip._integrate_with_state(x, 2, window=1.5, peak_window=1.0)
    assert not clip._same_branch(a, b)


def test_select_parent_rule_largest_causal_then_largest_peak():
    npz = {
        "cell_names": np.array(["A", "B"]), "cell_index": np.array([0, 0, 0, 1]),
        "causal": np.array([[0, 1, 0, 0, 0], [0, 2, 1, 0, 0], [0, 3, 0, 0, 0], [0, 9, 9, 9, 9]]),
        "T_peak": np.array([5000.0, 9000.0, 4000.0, 1.0]),
    }
    assert clip.select_parent(npz, "A") == 1          # rows 1 and 2 tie at 3 causal; T_peak 9000 > 4000
    npz["causal"][2] = [0, 4, 0, 0, 0]
    assert clip.select_parent(npz, "A") == 2          # the larger causal count wins regardless of T_peak
    assert clip.select_parent(npz, "B") == 3          # rows of another cell are never chosen


def test_two_lines_breaks_long_captions_once():
    short = "A short caption."
    assert clip.two_lines(short) == short
    long = clip.CAP_REAL
    out = clip.two_lines(long)
    assert out.count("\n") == 1 and out.replace("\n", " ") == long
