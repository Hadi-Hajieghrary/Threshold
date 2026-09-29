"""Reliability diagram: per-line first-passage hazard vs six-body fleet rollout (Phase 5, T2').

Contribution 2 of the paper.  Both forecasts are issued from the same exact true state at
the same ticks and scored against the same realised labels.  Every plotted value is a
MEASUREMENT recomputed here from the tick-level record of a phase that ran, and asserted
equal to the run's own report; no committed prediction, declaration, closed form or
conjectured constant is drawn.

Records read
------------
records/v2/phase5/p5_t2prime_ticks.npz   (measured ticks; npz array names, length 11310)
    forecast_linearized_H1   per-line (v1 single-cable linearised) hazard h, H = 1 s
    forecast_rollout_H1      six-body fleet rollout hazard h, H = 1 s, N = 2048 paths
    label_H1                 realised outcome y (first true upcrossing in [t, t+1 s]
                             closes above v_b)
    censored_H1              label right-censored at H = 1 s          -> excluded
    bounce                   tick lies in a bounce slack interval     -> excluded
    interval                 slack-interval id = bootstrap cluster of the scoring set

records/v2/phase5/p5_t2prime_declarations.json   (pre-registered scoring rule; read, not
                                                   plotted)
    ["event_and_bounce_rule"]  bounce-interval ticks "are excluded from calibration"
    ["label"]                  censored ticks "are left out, per horizon"
    ["gate"]["statistic"]      calibration_report(forecast, label, cluster = slack interval
                               id, n_bins = 10, n_boot = 2000, clip = 1/(2N), 0.95)
    ["gate"]["bootstrap_rng"]  SeedSequence([20260913, code]); code 3 = linearized_H1,
                               code 1 = rollout_H1
    ["reported_not_gating"]["items"]  "number of the ten equal-width bins outside the
                               simultaneous (Bonferroni, Kish-effective Wilson) band" and
                               "top-bin coincidence counts" (both declared, not gating)

records/v2/phase5/p5_t2prime_results.json   (the run's own report; every recomputed number
                                             is asserted equal to it before drawing)
    ["counts"]["calibration_ticks_H1" | "positive_labels_H1" | "calibration_clusters_H1"]
    ["reports"][key]["all"]["bins"][i]["mean_forecast" | "observed" | "count" | "lo" |
        "hi" | "inside"]                               key = linearized_H1 | rollout_H1
    ["reports"][key]["all"]["n_outside" | "slope" | "slope_lo" | "slope_hi"]
    ["reports"][key]["coincidence"]["bottom_bin_ticks" | "bottom_bin_events" |
        "top_bin_ticks" | "top_bin_misses"]
    ["gate"]["verdict" | "slope" | "slope_lo" | "slope_hi"]          (fleet rollout gate)
    ["post_hoc_diagnostics"]["forecasts"][key]["h_zero" | "h_one" | "h_zero_events" |
        "h_one_nonevents"]                             (POST-HOC: not pre-registered)

Scoring set: the declared one, keep = (~bounce) & (~censored_H1): 9227 of 11310 oracle
slack ticks, 480 positives, 376 clusters.

What is drawn
-------------
(a)  Reliability on LOGIT axes, so the two end bins (8756 + 439 per-line ticks, 8725 + 471
     fleet ticks: all but 32 and 31 of the 9227) are not compressed into the corners.
     x = mean forecast in each of the 10 declared equal-width bins, y = observed frequency;
     an observed frequency of exactly 0 or 1 (logit infinite) is drawn in a shaded strip
     beyond an axis break.  Vertical bar = the declared simultaneous 95% band for the bin
     (Wilson at the Kish effective cluster size, Bonferroni over occupied bins); the
     declared test puts a bin OUTSIDE when its mean forecast is not in the band, i.e. the
     bar misses the diagonal.  Outside bins are drawn bold and labelled.  Marker AREA is
     proportional to the bin's tick count (display floor 0.8 pt^2 so the 2-8-tick middle
     bins stay visible as dots).  The legend carries the declared n_outside, the pooled
     logistic recalibration slope with its declared cluster-bootstrap 95% interval, and
     the fleet gate verdict (FAIL: the interval excludes 1).  Slope < 1: forecasts too
     extreme (over-confident); slope > 1: not extreme enough (under-confident).
(b)  Confident errors under the declared end-bin coincidence rule (events at h < 0.1,
     non-events at h >= 0.9), and, as a POST-HOC split, the part of each that sits at a
     forecast of exactly 0 or exactly 1 (solid segment).
"""

from __future__ import annotations

import json
import math

import numpy as np

from tether.analysis.v2.paperfig.style import *  # noqa: F401,F403
from tether.analysis.v2.paperfig.style import (
    COL,
    C_BAND,
    C_FLEET,
    C_MUTED,
    C_NEUTRAL,
    C_PERLINE,
    REPO,
    save,
    use_paper_style,
)
from tether.monitor.metrics import calibration_report

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PHASE5 = REPO / "records" / "v2" / "phase5"
TICKS = PHASE5 / "p5_t2prime_ticks.npz"
RESULTS = PHASE5 / "p5_t2prime_results.json"
DECLARATIONS = PHASE5 / "p5_t2prime_declarations.json"

N_BINS = 10
N_PATHS = 2048
CLIP = 0.5 / N_PATHS          # declared forecast clip 1/(2N)
N_BOOT = 2000
BOOT_CODE = {"linearized_H1": 3, "rollout_H1": 1}   # declarations gate.bootstrap_rng

FS = 7.0                      # every piece of text on this figure, in pt

# Logit display geometry.  Finite logits live in [LO_LIM, HI_LIM]; a probability of
# exactly 0 or 1 is drawn at Y_ZERO / Y_ONE, in a shaded strip beyond an axis break.
X_LIM = (-10.9, 7.7)
LO_LIM, HI_LIM = -11.0, 8.0
Y_ZERO, Y_ONE = -12.7, 9.7
Y_LIM = (-13.5, 10.5)
PROB_TICKS = (1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.999)
PROB_LABELS = ("$10^{-4}$", "", "0.01", "0.1", "0.5", "0.9", "0.99", "")

AREA_PER_TICK = 110.0 / 8756.0   # pt^2 per tick: the largest bin gets 110 pt^2
AREA_FLOOR = 0.8                 # pt^2, display floor for the 2-8-tick middle bins

FORECASTS = (
    # report key, npz array, legend name, colour, marker, face, bar line style
    ("linearized_H1", "forecast_linearized_H1", "per-line hazard", C_PERLINE, "o",
     C_PERLINE, "-"),
    ("rollout_H1", "forecast_rollout_H1", "fleet rollout", C_FLEET, "s", "white",
     (0, (2.4, 1.2))),
)


def logit_y(p):
    """Probability -> display coordinate; exactly 0 and 1 go to the break strips."""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    out = np.empty_like(p)
    zero, one = p <= 0.0, p >= 1.0
    mid = ~(zero | one)
    out[mid] = np.log(p[mid] / (1.0 - p[mid]))
    out[zero], out[one] = Y_ZERO, Y_ONE
    assert np.all((out[mid] > LO_LIM) & (out[mid] < HI_LIM)), "finite logit off-axis"
    return out


def _load():
    ticks = np.load(TICKS)
    results = json.loads(RESULTS.read_text())
    declarations = json.loads(DECLARATIONS.read_text())
    # The declared scoring set (read the rule, then apply it).
    assert "excluded from calibration" in declarations["event_and_bounce_rule"]
    assert "censored and left out" in declarations["label"]
    items = declarations["reported_not_gating"]["items"]
    assert any("outside the simultaneous" in it for it in items)
    assert "top-bin coincidence counts" in items
    keep = (~ticks["bounce"]) & (~ticks["censored_H1"])
    counts = results["counts"]
    assert int(keep.sum()) == counts["calibration_ticks_H1"] == 9227
    assert int(ticks["label_H1"][keep].sum()) == counts["positive_labels_H1"]
    assert np.unique(ticks["interval"][keep]).size == counts["calibration_clusters_H1"]
    return ticks, keep, results


def _score(name, h, y, cluster, recorded, diag):
    """Recompute the declared calibration report, end-bin misses and point-mass split."""
    rng = np.random.default_rng(np.random.SeedSequence([20260913, BOOT_CODE[name]]))
    rep = calibration_report(h, y, cluster, n_bins=N_BINS, n_boot=N_BOOT, clip=CLIP,
                             confidence=0.95, rng=rng)
    ref = recorded["all"]
    assert len(rep.bins) == len(ref["bins"]) == N_BINS, "all ten bins occupied"
    for mine, theirs in zip(rep.bins, ref["bins"]):
        for key in ("mean_forecast", "observed", "lo", "hi"):
            assert abs(mine[key] - theirs[key]) < 1e-9, (name, key)
        assert int(mine["count"]) == int(theirs["count"]), (name, "count")
        assert bool(mine["inside"]) == bool(theirs["inside"]), (name, "inside")
        # The declared test: the bin's mean forecast lies in its simultaneous band.
        assert bool(theirs["inside"]) == (theirs["lo"] <= theirs["mean_forecast"]
                                          <= theirs["hi"])
    assert rep.n_outside == ref["n_outside"], (name, "n_outside")
    for key in ("slope", "slope_lo", "slope_hi"):
        assert abs(getattr(rep, key) - ref[key]) < 1e-9, (name, key)

    # Declared coincidence rule: top bin h in [0.9, 1], bottom bin h < 0.1.
    top, bottom = h >= 0.9, h < 0.1
    rc = recorded["coincidence"]
    miss = {
        "bottom_ticks": int(bottom.sum()),
        "bottom_events": int((bottom & y).sum()),
        "top_ticks": int(top.sum()),
        "top_nonevents": int((top & ~y).sum()),
        # POST-HOC split: the part of each at a forecast of exactly 0 / exactly 1.
        "zero_ticks": int((h == 0.0).sum()),
        "zero_events": int(((h == 0.0) & y).sum()),
        "one_ticks": int((h == 1.0).sum()),
        "one_nonevents": int(((h == 1.0) & ~y).sum()),
    }
    assert miss["bottom_ticks"] == rc["bottom_bin_ticks"]
    assert miss["bottom_events"] == rc["bottom_bin_events"]
    assert miss["top_ticks"] == rc["top_bin_ticks"]
    assert miss["top_nonevents"] == rc["top_bin_misses"]
    assert miss["zero_ticks"] == diag["h_zero"] and miss["one_ticks"] == diag["h_one"]
    assert miss["zero_events"] == diag["h_zero_events"]
    assert miss["one_nonevents"] == diag["h_one_nonevents"]

    return {
        "x": np.array([b["mean_forecast"] for b in rep.bins]),
        "obs": np.array([b["observed"] for b in rep.bins]),
        "lo": np.array([b["lo"] for b in rep.bins]),
        "hi": np.array([b["hi"] for b in rep.bins]),
        "inside": np.array([bool(b["inside"]) for b in rep.bins]),
        "count": np.array([b["count"] for b in rep.bins], dtype=float),
        "n_outside": int(rep.n_outside),
        "slope": rep.slope,
        "slope_lo": rep.slope_lo,
        "slope_hi": rep.slope_hi,
        "miss": miss,
    }


def _break_marks(ax, y_data):
    """Two short slashes across the left spine at data height y_data."""
    trans = ax.get_yaxis_transform()     # x in axes fraction, y in data
    for dy in (-0.35, 0.35):
        ax.plot([-0.018, 0.018], [y_data + dy - 0.28, y_data + dy + 0.28],
                transform=trans, color=C_NEUTRAL, lw=0.6, clip_on=False, zorder=6)
    ax.plot([0.0, 0.0], [y_data - 0.3, y_data + 0.3], transform=trans, color="white",
            lw=1.6, clip_on=False, zorder=5.5)


def main() -> None:
    use_paper_style()
    ticks, keep, results = _load()
    y = ticks["label_H1"][keep].astype(bool)
    cluster = ticks["interval"][keep]

    series = {}
    for name, array, *_ in FORECASTS:
        h = ticks[array][keep].astype(float)
        series[name] = _score(name, h, y, cluster, results["reports"][name],
                              results["post_hoc_diagnostics"]["forecasts"][name])
    pl, fl = series["linearized_H1"], series["rollout_H1"]
    gate = results["gate"]
    for key in ("slope", "slope_lo", "slope_hi"):
        assert abs(gate[key] - fl[key]) < 1e-9, ("gate", key)
    assert (gate["verdict"] == "FAIL") == (not fl["slope_lo"] <= 1.0 <= fl["slope_hi"])
    assert pl["n_outside"] == 2 and fl["n_outside"] == 0

    fig = plt.figure(figsize=(COL, 4.35))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.75, 0.78], hspace=0.56,
                          left=0.185, right=0.972, top=0.85, bottom=0.078)
    ax = fig.add_subplot(gs[0])
    axm = fig.add_subplot(gs[1])

    # --------------------------------------------------------- (a) reliability, logit
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    for yc in (Y_ZERO, Y_ONE):
        ax.axhspan(yc - 0.75, yc + 0.75, color=C_BAND, alpha=0.45, lw=0, zorder=0)
    diag_x = np.array([X_LIM[0], X_LIM[1]])
    ax.plot(diag_x, diag_x, color=C_MUTED, ls=(0, (1.2, 1.4)), lw=0.9, zorder=1)

    for name, _, label, colour, marker, face, ls in FORECASTS:
        s = series[name]
        xs = logit_y(s["x"])
        for k in range(N_BINS):
            out = not s["inside"][k]
            ylo, yhi = logit_y([s["lo"][k], s["hi"][k]])
            ax.plot([xs[k], xs[k]], [ylo, yhi], color=colour, ls=ls,
                    lw=1.5 if out else 0.6, alpha=1.0 if out else 0.7,
                    solid_capstyle="butt", zorder=3 if out else 2)
        area = np.maximum(AREA_PER_TICK * s["count"], AREA_FLOOR)
        ax.scatter(xs, logit_y(s["obs"]), s=area, marker=marker, facecolor=face,
                   edgecolor=colour, linewidths=np.where(area > 3.0, 0.8, 0.45),
                   zorder=5 if name == "linearized_H1" else 4)

    # Direct labels.
    ax.text(-10.55, logit_y(pl["hi"][0])[0] + 0.55, "forecast\noutside band",
            fontsize=FS, color=C_PERLINE, ha="left", va="bottom", linespacing=1.05)
    ax.text(X_LIM[1] - 0.15, logit_y(pl["lo"][-1])[0] - 0.45, "forecast\noutside band",
            fontsize=FS, color=C_PERLINE, ha="right", va="top", linespacing=1.05)
    ax.text(X_LIM[1] - 0.15, -5.6,
            "bar: declared 95%\nsimultaneous band\narea $\\propto$ ticks in bin",
            fontsize=FS, color=C_NEUTRAL, ha="right", va="top", linespacing=1.15)
    # "ideal" along the diagonal, rotated to its on-screen angle.
    p0, p1 = ax.transData.transform([(-7.0, -7.0), (-3.0, -3.0)])
    angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(-5.9, -5.3, "ideal", fontsize=FS, color=C_MUTED, rotation=angle,
            rotation_mode="anchor", ha="center", va="bottom")

    ticks_y = [Y_ZERO] + list(np.log(np.array(PROB_TICKS) / (1 - np.array(PROB_TICKS))))
    ticks_y += [Y_ONE]
    ax.set_yticks(ticks_y)
    ax.set_yticklabels(["0"] + list(PROB_LABELS) + ["1"])
    ax.set_xticks(ticks_y[1:-1])
    ax.set_xticklabels(PROB_LABELS)
    ax.tick_params(labelsize=FS)
    for yc in (0.5 * (Y_ZERO + LO_LIM), 0.5 * (Y_ONE + HI_LIM)):
        _break_marks(ax, yc)
    ax.grid(True, lw=0.35, alpha=0.22)
    ax.set_xlabel("forecast probability $h$ ($H = 1$ s; bin mean, logit scale)", fontsize=FS,
                  labelpad=1.5)
    ax.set_ylabel("observed severance frequency", fontsize=FS, labelpad=1.5)

    handles = []
    for name, _, label, colour, marker, face, ls in FORECASTS:
        s = series[name]
        tone = "over-confident" if s["slope"] < 1.0 else "under-confident"
        verdict = f", gate {gate['verdict']}" if name == "rollout_H1" else ""
        handles.append(Line2D(
            [], [], color=colour, ls=ls, lw=0.9, marker=marker, ms=4.2, mfc=face,
            mec=colour, mew=0.8,
            label=(f"{label}: {s['n_outside']} of {N_BINS} bins outside band\n"
                   f"recal. slope {s['slope']:.2f} [{s['slope_lo']:.2f}, {s['slope_hi']:.2f}]: "
                   f"{tone}{verdict}")))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.215, 1.005),
              ncol=1, handlelength=2.2, borderpad=0.1, labelspacing=0.3,
              handletextpad=0.5, borderaxespad=0.1, fontsize=FS)
    fig.text(0.005, 0.998, "(a)", fontsize=FS, ha="left", va="top", weight="bold")

    # --------------------------------------------------------- (b) confident errors
    rows = (
        # y, series, declared miss, declared ticks, post-hoc point-mass part, symbol
        (3.0, "linearized_H1", "bottom_events", "bottom_ticks", "zero_events", "0"),
        (2.2, "rollout_H1", "bottom_events", "bottom_ticks", "zero_events", "0"),
        (1.0, "linearized_H1", "top_nonevents", "top_ticks", "one_nonevents", "1"),
        (0.2, "rollout_H1", "top_nonevents", "top_ticks", "one_nonevents", "1"),
    )
    style = {n: (c, m, f) for n, _, _, c, m, f, _ in FORECASTS}
    xmax = 80.0
    for yr, name, key, key_n, key_pm, sym in rows:
        colour, marker, face = style[name]
        m = series[name]["miss"]
        total, part, n_bin = m[key], m[key_pm], m[key_n]
        assert part <= total
        if total:
            axm.barh(yr, part, height=0.62, color=colour, lw=0, zorder=3)
            axm.barh(yr, total - part, left=part, height=0.62, facecolor="white",
                     edgecolor=colour, lw=0.6, hatch="//////", zorder=3)
            axm.text(part / 2, yr, f"{part} at $h={sym}$", fontsize=FS, color="white",
                     ha="center", va="center", zorder=4)
        else:
            axm.plot([0], [yr], marker=marker, ms=3.6, mfc=face, mec=colour, mew=0.8,
                     ls="none", clip_on=False, zorder=4)
        axm.text(total + 1.4, yr, f"{total} of {n_bin}", fontsize=FS, color=C_NEUTRAL,
                 ha="left", va="center")
    axm.set_xlim(0, xmax)
    axm.set_ylim(-0.3, 3.5)
    axm.set_yticks([2.6, 0.6])
    axm.set_yticklabels(["events\nat $h<0.1$", "non-events\nat $h\\geq0.9$"],
                        fontsize=FS, linespacing=1.0)
    axm.tick_params(axis="y", length=0)
    axm.tick_params(axis="x", labelsize=FS)
    axm.set_xlabel("confident errors, declared end-bin rule (ticks)", fontsize=FS, labelpad=1.5)
    axm.grid(True, axis="x", lw=0.35, alpha=0.22)
    axm.legend(handles=[Patch(facecolor=C_NEUTRAL, lw=0,
                              label="at $h$ exactly 0 or 1 (post-hoc split)"),
                        Patch(facecolor="white", edgecolor=C_NEUTRAL, lw=0.6,
                              hatch="//////", label="rest of end bin")],
               loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=FS,
               handlelength=1.3, handleheight=0.8, columnspacing=0.9,
               handletextpad=0.4, borderpad=0.1, borderaxespad=0.1)
    bb = axm.get_position()
    fig.text(0.005, bb.y1 + 0.05, "(b)", fontsize=FS, ha="left", va="bottom",
             weight="bold")

    save(fig, "fig_reliability")

    for name in ("linearized_H1", "rollout_H1"):
        s = series[name]
        print(f"{name}: n_outside {s['n_outside']}  slope {s['slope']:.4f} "
              f"[{s['slope_lo']:.4f}, {s['slope_hi']:.4f}]  miss {s['miss']}")
        print("   bins x   ", np.round(s["x"], 5).tolist())
        print("   bins obs ", np.round(s["obs"], 4).tolist())
        print("   bins n   ", s["count"].astype(int).tolist())
        print("   inside   ", s["inside"].tolist())
    print("gate", gate["verdict"])


if __name__ == "__main__":
    main()
