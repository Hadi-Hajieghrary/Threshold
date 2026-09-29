"""Plan v3 WP1-WP3 analysis rules on synthetic inputs (Drake-free)."""

from __future__ import annotations

import numpy as np
import pytest

from tether.campaign.v3 import analyse as A
from tether.campaign.v3 import analyse_wp2 as W2
from tether.campaign.v3 import analyse_wp3 as W3
from tether.campaign.v3 import campaign
from tether.theory import branching as B


def test_share_verdict_rules():
    assert A._share_verdict({f"c{i}": "PASS" for i in range(5)})["verdict"] == "PASS"
    assert A._share_verdict({f"c{i}": ("PASS" if i < 4 else "FAIL") for i in range(5)})["verdict"] == "PASS"  # 4/5 = 80 %
    assert A._share_verdict({f"c{i}": ("PASS" if i < 3 else "FAIL") for i in range(5)})["verdict"] == "FAIL"
    assert A._share_verdict({"a": "PASS", "b": "PASS", "c": "UNSCORED"})["verdict"] == "UNDER-POWERED"
    out = A._share_verdict({f"c{i}": "PASS" for i in range(6)} | {"u": "UNSCORED"})
    assert out["scored_cells"] == 6 and out["passing_cells"] == 6


def test_pooled_pairs_are_mirror_symmetric():
    for i in range(5):
        for j in range(5):
            assert A._pooled_pair(i, j) == A._pooled_pair(4 - i, 4 - j)
    assert A._pooled_pair(4, 2) == (0, 2) and A._pooled_pair(2, 4) == (2, 0)


def _run(seed, pilot, drop_norm, x=None):
    n = drop_norm.size
    t = {
        "parent_mark": np.arange(n), "role": np.zeros(n, dtype=np.int64), "eligible_i": np.ones(n, dtype=bool),
        "all_neighbours_taut": np.ones(n, dtype=bool), "x": np.full(n, 2.0) if x is None else x, "drop_norm": drop_norm,
        "drop_norm_cos": drop_norm, "neighbour_i": np.zeros(n, dtype=np.int64), "cable_j": np.full(n, 2, dtype=np.int64),
        "drop": drop_norm, "T_peak": np.ones(n),
    }
    return {"meta": {"seed": seed, "pilot": pilot, "closure": None}, "transmission": t, "transmission_half_window": {"drop_norm": drop_norm}}


def test_stiffness_test_verdicts():
    rng = np.random.default_rng(0)
    # every synthetic row is the pair (neighbour 0, parent 2); the predicted normalised drops are 0.14 and 0.185
    norm_low, norm_high = np.full((5, 5), np.nan), np.full((5, 5), np.nan)
    norm_low[0, 2], norm_high[0, 2] = 0.14, 0.185
    low = [_run(s, False, 0.14 + 0.01 * rng.standard_normal(30)) for s in range(2003, 2013)]
    high = [_run(s, False, 0.185 + 0.01 * rng.standard_normal(30)) for s in range(2003, 2013)]
    out = A._stiffness_test(low, high, norm_low, norm_high, 0.045)
    assert out["verdict"] == "PASS-exact" and abs(out["delta_measured"] - 0.045) < 0.01
    assert abs(out["delta_exact"] - 0.045) < 1e-12 and out["delta_exact_uniform_pairs"] == 0.045
    flat = [_run(s, False, 0.14 + 0.01 * rng.standard_normal(30)) for s in range(2003, 2013)]
    assert A._stiffness_test(low, flat, norm_low, norm_high, 0.045)["verdict"] == "PASS-closed"
    noisy_low = [_run(s, False, 0.14 + 0.2 * rng.standard_normal(30)) for s in range(2003, 2013)]
    noisy_high = [_run(s, False, 0.16 + 0.2 * rng.standard_normal(30)) for s in range(2003, 2013)]
    assert A._stiffness_test(noisy_low, noisy_high, norm_low, norm_high, 0.045)["verdict"] == "UNDER-POWERED"
    assert A._stiffness_test(low[:1], high[:1], norm_low, norm_high, 0.045)["n_pairs"]["low"] == 30
    pilots = [_run(2001, True, np.full(30, 0.1))]
    assert A._stiffness_test(pilots, pilots, norm_low, norm_high, 0.045)["verdict"] == "UNSCORED"


def test_binned_mx_and_isotonic_clamp():
    x = np.array([0.5, 0.8, 1.2, 1.3, 1.8, 2.5, 4.0, 6.0, 10.0, 20.0] * 12)
    n_cross = np.floor(x / 2.0)
    b = W2.binned_mx(x, n_cross, min_events=10)
    assert b["spearman"] > 0.9 and b["first_bin_mean"] == 0.0 and len(b["counts"]) == len(campaign.COUPLED_PEAK_BINS) + 1
    fit = W2.fit_isotonic(x, n_cross)
    assert np.all(np.diff(fit["y"]) >= -1e-12)
    pred = W2.predict_isotonic(fit, np.array([0.1, 100.0]))
    assert pred[0] == min(fit["y"]) and pred[1] == max(fit["y"])


def test_time_shift_excess_counts_window_onsets_against_a_uniform_null():
    rng = np.random.default_rng(1)
    end = 620.0
    onsets_t = np.sort(rng.uniform(20.0, end, size=600))  # a Poisson-like background on the other cables
    onsets_c = rng.integers(0, 4, size=600) + 1  # never cable 0
    parent_t = 300.0
    onsets_t = np.concatenate([onsets_t, [parent_t + 0.5, parent_t + 1.0, parent_t + 2.0]])
    onsets_c = np.concatenate([onsets_c, [1, 2, 3]])
    run = {"meta": {"seed": 5, "end": end}, "onsets": {"time": onsets_t, "cable": onsets_c},
           "events": {"t_up": np.array([parent_t]), "cable_j": np.array([0])}}
    out = W2.time_shift_excess([run], {5: np.array([0])})
    background = 603 / 600.0 * 3.0  # onsets per 3 s window on average
    assert out["parents"] == 1 and out["measured"] >= 3.0
    assert abs(out["null"] - background) < 0.5 and out["excess"] > 1.0


def test_sweep_branch_rules():
    def cell(i, rho, lo, hi, operable=True):
        return {"rho": rho, "rho_ci95": [lo, hi], "operable": operable, "closures": 0 if operable else 10}

    sweep = {"T600_k500_I0.5": cell(0.5, 0.6, 0.5, 0.7), "T600_k500_I1.0": cell(1.0, 0.9, 0.8, 0.98), "T600_k500_I1.5": cell(1.5, 1.3, 1.1, 1.5)}
    out = W3._sweep_branch(sweep)
    assert out["branch"] == "(a)" and 1.0 < out["critical_intensity"] < 1.5 and out["between"] == ["T600_k500_I1.0", "T600_k500_I1.5"]
    margin = {"T600_k500_I0.5": cell(0.5, 0.6, 0.5, 0.7), "T600_k500_I1.0": cell(1.0, 0.8, 0.7, 0.9)}
    out = W3._sweep_branch(margin)
    assert out["branch"] == "(b)" and abs(out["margin"] - 0.2) < 1e-12
    straddle = {"T600_k500_I0.5": cell(0.5, 0.6, 0.5, 0.7), "T600_k500_I1.0": cell(1.0, 1.0, 0.9, 1.1)}
    assert W3._sweep_branch(straddle)["branch"] == "(c)"
    # a supercritical cell that is not operable cannot carry the transition
    closed = {"T600_k500_I0.5": cell(0.5, 0.6, 0.5, 0.7), "T600_k500_I1.5": cell(1.5, 1.3, 1.1, 1.5, operable=False)}
    assert W3._sweep_branch(closed)["branch"] == "(b)"


def test_rate_law_on_synthetic_branching_process():
    """Simulate primaries and Poisson offspring with a known K; the empirical kernel and rate law recover it."""
    rng = np.random.default_rng(3)
    k_true = np.array([[0.0, 0.3, 0.1, 0.0, 0.0], [0.2, 0.0, 0.3, 0.1, 0.0], [0.1, 0.2, 0.0, 0.2, 0.1], [0.0, 0.1, 0.3, 0.0, 0.2], [0.0, 0.0, 0.1, 0.3, 0.0]])
    pi = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
    exposure = 5000.0
    n_primary = 4000
    cable_j, offspring, primary = [], [], []
    queue = [(int(rng.choice(5, p=pi)), True) for _ in range(n_primary)]
    while queue:
        c, is_primary = queue.pop()
        children = rng.poisson(k_true[:, c])
        cable_j.append(c)
        offspring.append(children)
        primary.append(is_primary)
        for i in range(5):
            queue.extend([(i, False)] * int(children[i]))
    cable_j, offspring, primary = np.array(cable_j), np.array(offspring), np.array(primary)
    k_hat = np.nan_to_num(B.empirical_kernel(cable_j, offspring), nan=0.0)
    assert np.max(np.abs(k_hat - k_true)) < 0.05
    nu_primary = np.array([np.sum(cable_j[primary] == c) for c in range(5)]) / exposure
    nu_total = np.array([np.sum(cable_j == c) for c in range(5)]) / exposure
    predicted = B.total_rate(k_hat, nu_primary)
    assert abs(predicted.sum() / nu_total.sum() - 1.0) < 0.05
    theta = np.sum(primary) / primary.size
    assert abs(B.extremal_index(k_hat, nu_primary / nu_primary.sum()) - theta) < 0.03
    assert abs(B.spectral_radius(k_hat) - B.spectral_radius(k_true)) < 0.05


def test_measured_ratio_matrix_mirror_fills():
    cell = {"T1.2": {"pairs": [[0, 2], [1, 3]], "median_drop_over_Tpeak": [0.16, 0.12]}}
    r = W2._measured_ratio_matrix(cell)
    assert r[0, 2] == 0.16 and r[4, 2] == 0.16 and r[1, 3] == 0.12 and r[3, 1] == 0.12 and np.isnan(r[0, 1])
    assert W2._measured_ratio_matrix({}) is None


def test_jsonable_handles_numpy_and_nan():
    out = W2._jsonable({"a": np.array([1.0, np.nan]), "b": np.int64(3), "c": np.bool_(True), "d": (np.float64(2.5),)})
    assert out == {"a": [1.0, None], "b": 3, "c": True, "d": [2.5]}


# --------------------------------------------------------------------- corrections rounds 1 and 2


def _wp1_run(seed, pilot, drop, T_peak, g_measured, g_design_unused=None, n_i=0, c_j=2):
    """One synthetic WP1 run: every row is the ordered pair (neighbour n_i, parent c_j)."""
    n = np.asarray(drop).size
    t = {
        "parent_mark": np.arange(n), "role": np.zeros(n, dtype=np.int64), "eligible_i": np.ones(n, dtype=bool),
        "all_neighbours_taut": np.ones(n, dtype=bool), "x": np.full(n, 2.0), "drop": np.asarray(drop, float),
        "T_peak": np.full(n, float(T_peak)), "geometry_factor_ij": np.asarray(g_measured, float),
        "neighbour_i": np.full(n, n_i, dtype=np.int64), "cable_j": np.full(n, c_j, dtype=np.int64),
        "drop_norm": np.asarray(drop, float) / (T_peak * np.asarray(g_measured, float)),
        "drop_norm_cos": np.asarray(drop, float) / (T_peak * np.asarray(g_measured, float)),
    }
    return {"meta": {"seed": seed, "pilot": pilot, "closure": None}, "transmission": t,
            "transmission_half_window": {"drop_norm": t["drop_norm"]}}


def _prediction(r_value=0.165, g_design=1.0, operator=0.44):
    m = lambda v: [[None if i == j else v for j in range(5)] for i in range(5)]
    return {"transmission": {"response_M1_half_sine": m(r_value), "operator": m(operator), "geometry_factor": m(g_design),
                             "band_statistic_median_offdiag": r_value}}


def test_wp1_design_geometry_sensitivity_is_the_like_for_like_reading():
    """Audit round 1 M1: rows are normalised by the operator at the MEASURED geometry, the forecast by the DESIGN
    factor, so the declared statistic and its forecast are not comparable; the sensitivity reports the like-for-like
    reading and the share of rows whose measured factor is non-positive."""
    n = 40
    drop = np.full(n, 0.0165)          # 0.165 of a 0.1 N peak
    # half the rows sit at a measured geometry factor of 0.5, a quarter at 2.0, a quarter at -0.1 (past 90 degrees)
    g = np.concatenate([np.full(n // 2, 0.5), np.full(n // 4, 2.0), np.full(n - n // 2 - n // 4, -0.1)])
    runs = [_wp1_run(s, False, drop, 0.1, g) for s in range(2003, 2013)]
    out = A._wp1_cell("synthetic", runs, _prediction(r_value=0.165, g_design=1.0), campaign.DECLARATIONS)
    s = out["sensitivities"]
    # the like-for-like reading is drop / (T_peak * g_design) = 0.165 for every row, whatever the measured factor
    assert abs(s["design_geometry_median"] - 0.165) < 1e-12
    # 0.165 is the exact response, which lies BELOW the paper's band: the like-for-like reading still fails T1.1
    assert s["design_geometry_in_band"] is False and 0.165 < campaign.BAND[0]
    inside = [_wp1_run(s_, False, np.full(n, 0.03), 0.1, g) for s_ in range(2003, 2013)]   # 0.30 of the peak
    s_in = A._wp1_cell("synthetic", inside, _prediction(r_value=0.30, g_design=1.0), campaign.DECLARATIONS)["sensitivities"]
    assert abs(s_in["design_geometry_median"] - 0.30) < 1e-12 and s_in["design_geometry_in_band"] is True
    # the declared statistic is the median over the row-wise {0.33, 0.0825, -1.65} = 0.20625, which sits just
    # INSIDE the band: on identical rows the two normalisers return opposite verdicts, which is the finding
    assert abs(out["T1.1"]["median_drop_norm"] - 0.20625) < 1e-12 and out["T1.1"]["verdict"] == "PASS"
    assert s["design_geometry_in_band"] is not (campaign.BAND[0] <= out["T1.1"]["median_drop_norm"] <= campaign.BAND[1])
    assert abs(s["share_rows_measured_factor_nonpositive"] - 0.25) < 1e-12
    assert abs(s["share_rows_measured_factor_below_0.25"] - 0.25) < 1e-12
    lo, hi = s["design_geometry_ci95"]
    assert lo <= s["design_geometry_median"] <= hi


def test_predict_isotonic_step_is_right_continuous_and_clamped():
    """Audit round 2 M4: the declared predictor interpolates linearly; the step reading is the reported sensitivity."""
    fit = {"x": [1.0, 2.0, 3.0], "y": [0.0, 1.0, 4.0], "n": 3}
    step = W2.predict_isotonic_step(fit, np.array([0.5, 1.0, 1.5, 2.0, 2.999, 3.0, 9.0]))
    assert step.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 4.0, 4.0]          # right-continuous, clamped both ends
    linear = W2.predict_isotonic(fit, np.array([1.5, 2.5]))
    assert linear.tolist() == [0.5, 2.5] and step[2] != linear[0]        # the two conventions really differ
    assert W2.predict_isotonic_step({"x": [], "y": [], "n": 0}, np.array([1.0])).tolist() == [0.0]


def _taut_run(pilot, rows, offspring, T_peak=1000.0):
    """rows = [(event_id, neighbour_i, cable_j, T_i_at_t_up)]; offspring[event_id, cable] = attributed onsets."""
    r = np.array(rows, dtype=float)
    t = {
        "parent_mark": np.arange(len(rows)), "role": np.zeros(len(rows), dtype=np.int64),
        "event_id": r[:, 0].astype(np.int64), "neighbour_i": r[:, 1].astype(np.int64),
        "cable_j": r[:, 2].astype(np.int64), "T_i_at_t_up": r[:, 3], "T_peak": np.full(len(rows), T_peak),
    }
    return {"meta": {"pilot": pilot}, "transmission": t, "events": {"offspring_per_cable": np.asarray(offspring)}}


def test_taut_neighbour_target_counts_distinct_pairs_of_taut_neighbours():
    """Audit round 2 M3: the count the mechanistic kernel models is distinct (event, neighbour) pairs whose
    neighbour was taut at t_up and was attributed an onset - not every cross-cable onset."""
    offspring = np.zeros((2, 5), dtype=np.int64)
    offspring[0, 1] = 3          # event 0 unloaded neighbour 1 three times: one pair, not three
    offspring[0, 3] = 1          # event 0 unloaded neighbour 3 once
    offspring[1, 1] = 1          # event 1 unloaded neighbour 1, but that neighbour was slack at t_up
    rows = [(0, 1, 2, 600.0), (0, 3, 2, 600.0), (0, 4, 2, 600.0), (1, 1, 2, 0.0)]
    runs = [_taut_run(False, rows, offspring), _taut_run(True, rows, offspring)]   # the pilot run is ignored
    quantiles = np.tile(np.linspace(0.0, 1000.0, 101), (5, 1))     # margin uniform on [0, 1000] N for every cable
    ratio = np.full((5, 5), 0.2)
    out = W2.taut_neighbour_target(runs, ratio, quantiles)
    assert out["rows"] == 3                                        # the slack-at-t_up row is excluded
    assert out["target"] == 2.0                                    # pairs (0,1) and (0,3); (0,4) was not unloaded
    # each scored row predicts P(margin < 0.2 * 1000 N) = 0.2 under the uniform margin law
    assert abs(out["predicted"] - 3 * 0.2) < 1e-9
    assert abs(out["ratio"] - 0.3) < 1e-9 and out["in_band"] is False
    assert W2.taut_neighbour_target([], ratio, quantiles)["ratio"] != W2.taut_neighbour_target([], ratio, quantiles)["ratio"]


def test_addenda_sha256_lists_every_addendum_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "RECORD_DIR", tmp_path)
    assert A.addenda_sha256() == {}
    (tmp_path / "cascade_addendum_2.json").write_text('{"addendum": 2}')
    (tmp_path / "cascade_addendum_1.json").write_text('{"addendum": 1}')
    (tmp_path / "cascade_declarations.json").write_text("{}")      # not an addendum
    out = A.addenda_sha256()
    assert list(out) == ["cascade_addendum_1.json", "cascade_addendum_2.json"]   # sorted, addenda only
    assert all(len(v) == 64 for v in out.values()) and out["cascade_addendum_1.json"] != out["cascade_addendum_2.json"]
