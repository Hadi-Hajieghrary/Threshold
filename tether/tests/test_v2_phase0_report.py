"""Unit tests for the v2 Phase 0 report generator.

The generator must read every number from a committed record and must never round a
verdict-carrying quantity into a different verdict.
"""

from __future__ import annotations

from tether.analysis.v2 import phase0_report as report


def test_factor_never_rounds_a_small_exceedance_to_zero():
    # The 0.05 rule lives at the top of a range that spans 250 orders of magnitude.
    assert report._factor(0.0) == "0"
    assert report._factor(1.2193529e-3) == "1.22e-03"
    assert report._factor(3.945e-3) == "3.94e-03"
    assert report._factor(0.076375) == "0.08"
    assert report._factor(0.5257) == "0.53"


def test_fmt_and_interval():
    assert report._fmt(None) == "-"
    assert report._fmt(True) == "yes"
    assert report._fmt(3) == "3"
    assert report._fmt(3.9745675) == "3.975"
    assert report._fmt(2.5e-8) == "2.500e-08"
    assert report._interval([3.8134226, 4.1357125]) == "[3.813, 4.136]"


def test_verdict():
    assert report._verdict(True) == "PASS"
    assert report._verdict(False) == "FAIL"
