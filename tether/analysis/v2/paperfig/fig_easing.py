"""Why easing thrust does not reduce severance (paper contribution 3, Sec. V-A).

The v1 Phase 6 supervisor eases every vessel on the largest hazard it has heard across the
fleet (tether/control/supervisor.py, fleet_term=True).  Easing thrust is meant to lower the
return-leg mean acceleration a_bar_ret, but it can deepen the excursion Delta.

DEFINITION (corrected 2026-09-14, fourth round).  The paper's a_bar_ret is the depth-mean
radial acceleration from the DEEPEST point, where the chord rate vanishes, so per excursion
``v_up^2 = 2 a_bar_ret Delta`` exactly (the corrected Prop. 3 of reports/phase1_report.md).
The campaign's own P6-T7 statistic instead solved the plan's PRINTED Prop. 3,
a = (v_up^2 - u_entry^2)/(2 Delta), which counts the entry energy twice and is negative for
59/187 (N) and 49/172 (P) deep excursions; its P/N median ratio is 0.565.  This figure plots
the paper's definition, a_bar_ret = v_up^2/(2 Delta) (P/N 0.817, 95% CI 0.58-1.04), and uses
the printed form only to assert that re-pooling the marks reproduces the record.  The figure
shows

  (a) the (a_bar_ret, Delta) plane, as P/N ratios of pooled medians on log axes (the grey
      dotted iso-V_up lines are level sets of sqrt(2 a Delta) for a single excursion -- an
      identity, not data; ratios of medians need not lie on them);
  (b) the MEASURED closing-speed ratio V_up/V_up,N (ratio of median v_up);
  (c) the per-mission severance probability (live runs), and
  (d) the recorded paired (same-seed) differences with their 95 % intervals.

Two excursion populations are drawn in (a) and (b):
  * "all deep": every deep excursion of the recording runs over the full mission -- the
    population /tests/P6-T7 pools (phase6.py::mechanism_attribution).  About 77 % of these
    occur AFTER that run's virtual first severance, i.e. where a live run would already have
    lost the cable.
  * "pre-severance" (POST-HOC, not part of the pre-registered campaign): only the deep
    excursions with t_up earlier than the run's ["virtual_first"][0] (all of them when the run
    has no virtual severance).  Each population is normalised by its own N medians.

RECORDS READ
------------
records/phase6/phase6_results.json
  /tests/P6-T7/attribution/{N,P}/{median_a_bar, median_depth, deep_excursions,
                                  median_thrust_scale_during_slack}
  /tests/P6-T7/attribution/P_over_N/{a_bar, depth}           -> "all deep" operating point
  /tests/P6-T2/probabilities/{N, P, O}                        -> panel (c) (live severance)
  /tests/P6-T2/differences/{N-P, P-O}                         -> panel (d), [mean, lo95, hi95]
  (/probabilities/live and /tests/P6-T1/{probabilities, differences} are read only to assert
   that they equal the P6-T2 values; P6-T3 records 100 % recording/live agreement.)

records/phase6/cache/phase6_compute.pkl                       (per-excursion marks)
  ["results"][i] with keys "mode", "arm", "seed", "virtual_first", "marks"; each mark is
  (cable, t_up, depth, v_up, T_peak, dwell, u_entry).  Deep excursions are marks with
  depth > 0.05 m on the ("recording", arm) runs, a_bar = (v_up^2 - u_entry^2)/(2 depth),
  exactly as phase6.py::mechanism_attribution computes them.  Used for (i) an assertion that
  re-pooling reproduces the JSON medians and counts, (ii) the measured median v_up ratio,
  (iii) the pre-severance split, and (iv) all 95 % intervals: a paired, seed-clustered
  bootstrap (the 60 campaign seeds resampled jointly for both arms, ratio of pooled medians,
  4000 draws, fixed RNG seed).  (ii)-(iv) are computed here, post hoc; they are not stored in
  the JSON.  The pickle is required.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

from tether.analysis.v2.paperfig.style import *  # noqa: F401,F403
from tether.analysis.v2.paperfig.style import (
    REPO, COL, C_PERLINE, C_NEUTRAL, C_MUTED, C_OK, C_ACCENT, use_paper_style, save,
)
from tether.campaign.phase6 import SEEDS  # the campaign's 60 paired seeds (read-only)

RESULTS = REPO / "records" / "phase6" / "phase6_results.json"
CACHE = REPO / "records" / "phase6" / "cache" / "phase6_compute.pkl"

MIN_DEPTH = 0.05   # P6-T7's deep-excursion cut [m]
N_BOOT = 4000
BOOT_SEED = 20260914
FS = 7.0           # every piece of text on the figure (printed at 100 %)

ARM = {  # label, colour, bar hatch
    "N": ("unsupervised", C_NEUTRAL, ""),
    "P": ("eased: monitor", C_PERLINE, "////"),
    "O": ("eased: oracle", C_OK, "...."),
}
POP = {  # population: colour, marker, CI line style, legend text
    "all": (C_PERLINE, "s", "-", "all deep exc."),
    "pre": (C_ACCENT, "D", (0, (2.2, 1.2)), "pre-severance only (post-hoc)"),
}


# ------------------------------------------------------------------ records

def load_summary() -> dict:
    with RESULTS.open() as fh:
        return json.load(fh)


def load_excursions() -> dict:
    """Per-seed arrays of (a_bar_ret, depth, v_up, pre, a_printed) for the deep excursions of
    arms N and P: a_bar_ret = v_up^2/(2 d) (the paper's definition), a_printed =
    (v_up^2 - u_entry^2)/(2 d) (the record's P6-T7 form, used only for the provenance check)."""
    with CACHE.open("rb") as fh:
        results = pickle.load(fh)["results"]
    table = {(r["mode"], r["arm"], r["seed"]): r for r in results}
    out: dict = {}
    for arm in ("N", "P"):
        out[arm] = {}
        for seed in SEEDS:
            run = table[("recording", arm, seed)]
            t_sev = np.inf if run["virtual_first"] is None else float(run["virtual_first"][0])
            rows = [(v_up ** 2 / (2.0 * d), d, v_up, float(t_up < t_sev),
                     (v_up ** 2 - u_entry ** 2) / (2.0 * d))
                    for _c, t_up, d, v_up, _pk, _dw, u_entry in run["marks"] if d > MIN_DEPTH]
            out[arm][seed] = np.asarray(rows, float).reshape(-1, 5)
    return out


def pooled(ex_arm: dict, seeds, pop: str = "all") -> np.ndarray:
    x = np.concatenate([ex_arm[s] for s in seeds])
    return x if pop == "all" else x[x[:, 3] == 1.0]


def ratios(ex: dict, seeds, pop: str) -> np.ndarray:
    """P/N ratios of pooled medians: (a_bar, depth, v_up measured, sqrt(a_bar*depth) derived)."""
    r = np.median(pooled(ex["P"], seeds, pop)[:, :3], axis=0) / \
        np.median(pooled(ex["N"], seeds, pop)[:, :3], axis=0)
    return np.append(r, np.sqrt(r[0] * r[1]))


def paired_interval(ex: dict, pop: str) -> np.ndarray:
    """Paired seed-clustered bootstrap: 2.5/97.5 percentiles of each ratio, shape (2, 4)."""
    rng = np.random.default_rng(BOOT_SEED)
    seeds = np.asarray(SEEDS)
    draws = np.array([ratios(ex, rng.choice(seeds, seeds.size, replace=True), pop)
                      for _ in range(N_BOOT)])
    return np.percentile(draws, [2.5, 97.5], axis=0)


# ------------------------------------------------------------------ panels

def panel_plane(ax, est: dict, ci: dict) -> None:
    xlim, ylim = (0.38, 1.6), (0.5, 3.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for axis, ticks in ((ax.xaxis, [0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5]),
                        (ax.yaxis, [0.5, 0.7, 1.0, 1.5, 2.0, 3.0])):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:g}"))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())

    # iso-V_up lines of the identity sqrt(2 a Delta) (not data): V/V_N = sqrt(x y) -> y = r^2/x,
    # each labelled inside the axes where it leaves the top or the left edge.
    gx = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)
    for r in (0.75, 0.9, 1.0, 1.25):
        solid = r == 1.0
        colour = C_NEUTRAL if solid else C_MUTED
        ax.plot(gx, r ** 2 / gx, color=colour, lw=0.7 if solid else 0.6,
                ls="-" if solid else ":", zorder=1)
        if r > 1.0:    # right edge (the top edge is taken by P's interval)
            ax.text(xlim[1] / 1.02, r ** 2 / xlim[1] * 1.06, f"{r:g}", fontsize=FS,
                    color=colour, ha="right", va="bottom",
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
        else:
            ax.text(xlim[0] * 1.02, r ** 2 / xlim[0] * 1.03, f"{r:g}", fontsize=FS,
                    color=colour, ha="left", va="bottom")
    ax.text(xlim[1] / 1.02, ylim[1] / 1.03, r"iso-$V_\uparrow/V_{\uparrow,N}$ (identity)",
            fontsize=FS, color=C_MUTED, ha="right", va="top")

    # classical design assumption: easing lowers a_bar with Delta held fixed (not measured)
    ra_all = est["all"][0]
    ax.add_patch(FancyArrowPatch((1.0, 1.0), (ra_all, 1.0), arrowstyle="-|>", mutation_scale=6,
                                 lw=0.7, ls=(0, (2, 1.5)), color=C_NEUTRAL,
                                 shrinkA=4.0, shrinkB=3.5, zorder=3))
    ax.plot([ra_all], [1.0], marker="o", ms=4.4, mfc="white", mec=C_NEUTRAL, mew=0.8,
            ls="none", zorder=4)

    # measured: P/N operating points with paired-bootstrap intervals
    for pop in ("pre", "all"):
        colour, marker, ls, _ = POP[pop]
        ra, rd = est[pop][0], est[pop][1]
        lo, hi = ci[pop][0], ci[pop][1]
        eb = ax.errorbar([ra], [rd], xerr=[[ra - lo[0]], [hi[0] - ra]],
                         yerr=[[rd - lo[1]], [hi[1] - rd]], fmt="none", ecolor=colour,
                         elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=5)
        for line in eb[2]:
            line.set_linestyle(ls)
        ax.add_patch(FancyArrowPatch((1.0, 1.0), (ra, rd), arrowstyle="-|>", mutation_scale=7,
                                     lw=1.0, color=colour, shrinkA=4.0, shrinkB=4.0, zorder=6))
        ax.plot([ra], [rd], marker=marker, ms=4.8, mfc=colour, mec="white", mew=0.6,
                ls="none", zorder=7)
    ax.plot([1.0], [1.0], marker="o", ms=5.0, mfc=C_NEUTRAL, mec="white", mew=0.6,
            ls="none", zorder=7)

    box = dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85)
    ax.text(ra_all * 0.97, 1.0 / 1.05, "$\\Delta$ held fixed\n(not measured)", fontsize=FS,
            color=C_NEUTRAL, ha="right", va="top", linespacing=1.1, bbox=box, zorder=8)
    thrust = est["thrust"]
    ax.text(0.985, 0.03, f"P: median thrust $\\times${thrust:.2f}\nwhile slack",
            transform=ax.transAxes, fontsize=FS, color=C_PERLINE, ha="right", va="bottom",
            linespacing=1.1, bbox=box, zorder=8)

    ax.set_xlabel(r"return-leg accel. $\bar a_{\rm ret}$, P/N ratio [$-$]", labelpad=1.5)
    ax.set_ylabel(r"depth $\Delta$, P/N ratio [$-$]", labelpad=1.0)
    ax.grid(False)

    ax.text(1.0, 1.0 / 1.07, "N", fontsize=FS, color=C_NEUTRAL, ha="left", va="top",
            fontweight="bold")
    ax.annotate("P, all deep", xy=(est["all"][0], est["all"][1]), xytext=(6, 7),
                textcoords="offset points", fontsize=FS, color=POP["all"][0], ha="left",
                va="bottom", bbox=box, zorder=8)
    ax.annotate("P, pre-sev.\n(post-hoc)", xy=(est["pre"][0], est["pre"][1]), xytext=(-6, -5),
                textcoords="offset points", fontsize=FS, color=POP["pre"][0], ha="right",
                va="top", linespacing=1.05, bbox=box, zorder=8)
    ax.set_title("recording runs, deep exc. ($\\Delta>0.05$ m); 95% CI", fontsize=FS, pad=3)


def panel_speed(ax, est: dict, ci: dict, n: dict) -> None:
    """Closing-speed ratio: measured median v_up with its paired-bootstrap interval."""
    groups = {"all": 1.0, "pre": 0.0}
    off = {2: 0.0}                     # index 2 = measured v_up
    for pop, y0 in groups.items():
        colour, marker, ls, _ = POP[pop]
        for k, dy in off.items():
            y = y0 + dy
            measured = k == 2
            pt, lo, hi = est[pop][k], ci[pop][0][k], ci[pop][1][k]
            ax.plot([lo, hi], [y, y], color=colour, lw=1.0 if measured else 0.6, ls=ls,
                    solid_capstyle="butt", zorder=3)
            for e in (lo, hi):
                ax.plot([e, e], [y - 0.07, y + 0.07], color=colour, lw=0.7, zorder=3)
            ax.plot([pt], [y], marker=marker, ms=4.4 if measured else 4.0, ls="none", zorder=4,
                    mfc=colour if measured else "white", mec=colour, mew=0.9)
            ax.text(hi + 0.015, y, f"{pt:.3f}", fontsize=FS, color=colour, ha="left",
                    va="center", fontweight="bold" if measured else "normal")
    classical = float(np.sqrt(est["all"][0]))
    ax.axvline(1.0, color=C_NEUTRAL, lw=0.6, zorder=1)
    ax.axvline(classical, color=C_NEUTRAL, lw=0.7, ls=(0, (2, 1.5)), zorder=1)
    ax.text(classical - 0.01, 0.5, f"$\\Delta$ fixed {classical:.2f}", fontsize=FS,
            color=C_NEUTRAL, ha="right", va="center")
    ax.set_yticks(list(groups.values()))
    ax.set_yticklabels([f"all deep\n({n['all'][0]}/{n['all'][1]})",
                        f"pre-sev.\n({n['pre'][0]}/{n['pre'][1]})"], fontsize=FS,
                       linespacing=1.05)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.36, 1.85)
    ax.set_xlim(0.42, 1.6)
    ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_xlabel(r"closing speed $V_\uparrow/V_{\uparrow,N}$ [$-$]", labelpad=1.5)
    handles = [
        Line2D([], [], color=C_NEUTRAL, marker="s", ms=4.0, lw=1.0,
               label=r"measured median $v_\uparrow$, 95% CI"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.01, 1.03), ncol=2,
              fontsize=FS, handlelength=1.6, handletextpad=0.4, columnspacing=0.9,
              borderaxespad=0.0, borderpad=0.2, frameon=True, framealpha=0.9,
              edgecolor="none", facecolor="white")


def panel_probability(ax, probs: dict, n_seeds: int) -> None:
    ypos = {"N": 2.0, "P": 1.0, "O": 0.0}
    for arm, y in ypos.items():
        _label, colour, hatch = ARM[arm]
        p = probs[arm]
        ax.barh(y, p, height=0.6, facecolor="none", edgecolor=colour, linewidth=0.8,
                hatch=hatch, zorder=2)
        ax.barh(y, p, height=0.6, facecolor=colour, alpha=0.22, lw=0, zorder=1)
        ax.text(p + 0.015, y, f"{round(p * n_seeds)}/{n_seeds}", va="center", ha="left",
                fontsize=FS, color=colour)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([ARM[a][0] for a in ypos], fontsize=FS)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlim(0.0, 0.62)
    ax.set_xticks([0.0, 0.2, 0.4])
    ax.set_xlabel("sev. prob./mission [$-$]", labelpad=1.5)
    ax.set_title("live runs", fontsize=FS, pad=2)


def panel_difference(ax, diffs: dict) -> None:
    rows = {"N-P": 1.5, "P-O": 0.5}
    for name, y in rows.items():
        point, lo, hi = diffs[name]
        ax.plot([lo, hi], [y, y], color=C_NEUTRAL, lw=1.0, solid_capstyle="butt", zorder=3)
        for e in (lo, hi):
            ax.plot([e, e], [y - 0.12, y + 0.12], color=C_NEUTRAL, lw=0.8, zorder=3)
        ax.plot([point], [y], marker="D", ms=3.4, color=C_NEUTRAL, zorder=4)
        ax.text(-0.155, y - 0.2, name.replace("-", "$-$"), fontsize=FS, color=C_NEUTRAL,
                ha="left", va="top")
    ax.axvline(0.0, color=C_MUTED, lw=0.6, ls="--", zorder=1)
    ax.set_ylim(-0.3, 2.0)
    ax.set_xlim(-0.16, 0.16)
    ax.set_xticks([-0.1, 0.0, 0.1])
    ax.set_xticklabels(["$-$0.1", "0", "0.1"])
    ax.set_xlabel("paired diff. [$-$]", labelpad=1.5)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines["left"].set_visible(False)
    ax.set_title("95% CI", fontsize=FS, pad=2)


def panel_label(fig, ax, text: str, x: float | None = None) -> None:
    pos = ax.get_position()
    fig.text(0.004 if x is None else x, pos.y1 + 0.004, text, fontsize=7.5, ha="left",
             va="bottom", fontweight="bold")


# ------------------------------------------------------------------ main

def main() -> None:
    use_paper_style()
    record = load_summary()
    attr = record["tests"]["P6-T7"]["attribution"]
    probs = record["tests"]["P6-T2"]["probabilities"]
    diffs = record["tests"]["P6-T2"]["differences"]
    n_seeds = len(SEEDS)

    # consistency of the record with itself
    for arm in ("N", "P", "O"):
        assert probs[arm] == record["probabilities"]["live"][arm]
        assert probs[arm] == record["tests"]["P6-T1"]["probabilities"][arm]
    assert diffs["N-P"] == record["tests"]["P6-T1"]["differences"]["N-P"]
    assert abs(attr["P_over_N"]["a_bar"] - attr["P"]["median_a_bar"] / attr["N"]["median_a_bar"]) < 1e-12
    assert abs(attr["P_over_N"]["depth"] - attr["P"]["median_depth"] / attr["N"]["median_depth"]) < 1e-12

    ex = load_excursions()
    for arm in ("N", "P"):   # re-pooling the marks reproduces /tests/P6-T7/attribution exactly
        allx = pooled(ex[arm], SEEDS)
        assert len(allx) == attr[arm]["deep_excursions"]
        assert abs(np.median(allx[:, 4]) - attr[arm]["median_a_bar"]) < 1e-12
        assert abs(np.median(allx[:, 1]) - attr[arm]["median_depth"]) < 1e-12
    est = {pop: ratios(ex, SEEDS, pop) for pop in ("all", "pre")}
    printed = (np.median(pooled(ex["P"], SEEDS)[:, 4]) / np.median(pooled(ex["N"], SEEDS)[:, 4]))
    assert abs(printed - attr["P_over_N"]["a_bar"]) < 1e-12      # the record's 0.565
    assert abs(est["all"][1] - attr["P_over_N"]["depth"]) < 1e-12
    print(f"record P6-T7 a_bar ratio (printed Prop. 3 form, not plotted): {printed:.3f}")
    est["thrust"] = attr["P"]["median_thrust_scale_during_slack"]
    ci = {pop: paired_interval(ex, pop) for pop in ("all", "pre")}
    n = {pop: tuple(len(pooled(ex[a], SEEDS, pop)) for a in ("N", "P")) for pop in ("all", "pre")}

    for pop in ("all", "pre"):
        print(f"[{pop}] n N/P = {n[pop]}")
        for k, name in enumerate(("a_bar", "depth", "v_up measured", "sqrt(a*d) derived")):
            print(f"    {name:18s} {est[pop][k]:.3f}  95% CI [{ci[pop][0][k]:.3f}, {ci[pop][1][k]:.3f}]")
    for arm in ("N", "P"):
        v = {pop: float(np.median(pooled(ex[arm], SEEDS, pop)[:, 2])) for pop in ("all", "pre")}
        print(f"median v_up {arm}: all {v['all']:.3f}, pre {v['pre']:.3f} m/s")
    print(f"post-severance share of deep excursions: N {1 - n['pre'][0] / n['all'][0]:.3f}, "
          f"P {1 - n['pre'][1] / n['all'][1]:.3f}")

    fig = plt.figure(figsize=(COL, 5.0))
    left, right = 0.222, 0.985
    ax = fig.add_axes([left, 0.555, right - left, 0.405])
    axs = fig.add_axes([left, 0.3, right - left, 0.165])
    axp = fig.add_axes([left, 0.07, 0.40, 0.13])
    axd = fig.add_axes([left + 0.47, 0.07, right - left - 0.47, 0.13])
    panel_plane(ax, est, ci)
    panel_speed(axs, est, ci, n)
    panel_probability(axp, probs, n_seeds)
    panel_difference(axd, diffs)
    for a, t in ((ax, "(a)"), (axs, "(b)"), (axp, "(c)")):
        panel_label(fig, a, t)
    panel_label(fig, axd, "(d)", x=left + 0.47 - 0.05)
    print(f"plotted: p N {probs['N']:.4f} P {probs['P']:.4f} O {probs['O']:.4f}; "
          f"N-P {diffs['N-P']}; P-O {diffs['P-O']}")
    save(fig, "fig_easing")


if __name__ == "__main__":
    main()
