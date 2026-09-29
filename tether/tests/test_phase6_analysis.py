"""Phase 6 analysis helpers: the P6-T6 tail extrapolation and the P6-T8 post-severance record."""

from __future__ import annotations

import numpy as np

from tether.campaign.phase6 import post_severance, tail_extrapolation


def test_tail_extrapolation_recovers_exponential_rate():
    rng = np.random.default_rng(5)
    exposure = 10_000.0
    peaks = 1200.0 + rng.exponential(1500.0, size=4000)
    result = tail_extrapolation(peaks, exposure, level=15_000.0)
    truth = peaks.size / exposure * np.exp(-(15_000.0 - 1200.0) / 1500.0)
    assert not result["withheld"]
    assert abs(result["shape"]) < 0.15
    lo, hi = result["interval"]
    assert lo <= truth <= hi
    assert all(check["inside"] for check in result["checks"])


def test_tail_extrapolation_under_powered():
    result = tail_extrapolation(np.linspace(1000.0, 2000.0, 20), 100.0)
    assert result["withheld"] and result["n"] == 20


def test_post_severance_windows():
    time = np.arange(0.0, 40.0, 0.1)
    n = 5
    mean = np.zeros((time.size, n, 3))
    mean[:, :, 0] = np.linspace(0.0, 1.0, n)[None, :] * 0.01
    mean[time >= 22.0, 2, 1] = 3.0  # the severed vessel's estimate drifts off after t_s
    truth = np.zeros((time.size, 3))
    item = {"live_first": (20.0, 2), "estimates": {"time": time, "load_mean": mean, "load_true": truth}}
    row = post_severance(item)
    before, after = row["severed_offset"]
    assert before < 0.01 and abs(after - 3.0) < 0.01
    assert row["survivor_spread"][1] < 0.01
    assert post_severance({"live_first": (1.0, 0), "estimates": None}) is None


def test_mechanism_attribution_uses_return_energy_and_commanded_scale():
    from tether.campaign.phase6 import mechanism_attribution

    log = [(t, [0.0] * 5, [1.0, 0.4, 1.0, 1.0, 1.0]) for t in np.arange(0.0, 20.0, 0.1)]
    marks = [(1, 10.0, 0.5, 2.0, 5000.0, 3.0, 1.0), (2, 12.0, 0.01, 1.0, 1000.0, 1.0, 0.5)]
    row = mechanism_attribution([{"supervisor_log": log, "marks": marks}])
    assert row["deep_excursions"] == 1
    assert row["median_a_bar"] == (2.0**2 - 1.0**2) / (2 * 0.5)
    assert abs(row["mean_thrust_scale_during_slack"] - 0.4) < 1e-12
    assert mechanism_attribution([{"supervisor_log": [], "marks": [(0, 1.0, 0.01, 1.0, 1.0, 1.0, 1.0)]}]) is None


def test_false_alarm_episodes_window_and_outcome():
    from tether.campaign.phase6 import false_alarm_episodes

    scale = lambda t: [0.5 if (52.0 <= t < 53.0 or 20.0 <= t < 21.0) else 1.0] + [1.0] * 4
    log = [(t, [0.0] * 5, scale(t)) for t in np.round(np.arange(0.0, 80.0, 0.1), 1)]
    critical = np.full(5, 0.4)
    fast = [(0, 52.5, 0.5, 0.3, 9000.0, 1.0, 0.1)]  # 0.3 > 0.5 v_b, within H of the 52 s episode
    assert false_alarm_episodes(log, fast, critical, 1.0, 50.0, 70.0) == (1, 0)
    assert false_alarm_episodes(log, [], critical, 1.0, 50.0, 70.0) == (1, 1)
    assert false_alarm_episodes(log, fast, critical, 1.0, 0.0, np.inf) == (2, 1)
