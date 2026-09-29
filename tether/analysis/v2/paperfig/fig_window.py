"""Figure: the operating window does not exist (paper contribution 4).

The Phase 2 design needed one weather/pretension cell that satisfies two requirements
at once: a formation shape the towing loop holds (the (H1) shape band, chord-angle and
heading std <= 15 deg) and an event supply at the cell's own tension grid.  The figure
draws every measured selected-controller cell in the plane of the DECLARED DESCRIPTIVE
gap measure (["closest_to_both"]["measure"], "Descriptive ... not a test"):

  x  shape half       world chord-angle circular std, max over vessels (the gating
                      statistic of (H1)); band x <= 15 deg.  (H1) also bounds the heading
                      std psi at 15 deg, but psi < chord std in every plotted cell (asserted
                      below), so the chord std alone decides this half here.
  y  population half  events at the cell's third tension grid level, ALL (H4') CLASSES
                      POOLED (N + R1 + R2 + transitional); band y >= 20.

The pooled population half is LOOSER than the pre-declared per-class population rule
P1-T10' (>= 20 events of one class at >= 3 grid levels in >= 3 cells).  The figure never
presents the pooled factor as that rule: the least-far cell i035_T0600_ks3 satisfies the
pooled factor (44 events at 2500 N) but, per class, holds only N 19, R1 4, R2 5,
transitional 16 at that level and does NOT meet P1-T10' in any class
(["trade_off"][i]["meets_population_rule_in_some_class"] is false).  The window is
empty regardless of the population definition, because no cell reaches the shape band.

Raising the pretension T0 at fixed weather moves a cell down and to the left.  The
heading gain k_h is co-scheduled with T0 by the declared rule (1.5 x the lateral
stability boundary: 477 / 620 / 763 / 1050 N m/rad at T0 = 0.6 / 0.8 / 1.0 / 1.4 kN), so
each arrow changes T0 and k_h together.

RECORD READ (measurements only; no prediction, forecast or closed form is drawn)
-------------------------------------------------------------------------------
records/v2/phase1/stochastic_results.json

  ["trade_off"][i]                     per-cell measurement rows (8 rows).  Plotted:
                                       the six rows with k_sigma == 3.0 (the selected
                                       controller), the same six cells the declared
                                       ranking ["closest_to_both"]["ranked"] scores.
      ["cell"], ["intensity"], ["T0_kN"], ["k_sigma"], ["k_h"]
                                       identity; intensity groups the lines, T0 labels
                                       the points, k_h is checked to rise with T0
      ["chord_world_std_deg"]          x (deg)
      ["events_at_third_level"]        y (count, classes pooled).  UNDEFINED where the
                                       cell has no threshold grid (["grid_levels_N"] == [],
                                       ["third_level_N"] is null): i035_T1000_ks3 (pilot
                                       seeds held 0 marks; 2 events in all) and
                                       i050_T1400_ks3 (pilot seeds held 1 mark; 5 events
                                       in all).  The record stores 0 there by convention
                                       and the declared measure scores the population
                                       factor as inf.  These two are drawn hollow in a
                                       separate "no grid" strip, not as measured zeros.
      ["psi_std_deg"]                  checked only: < chord std in every plotted cell
      ["marks"]                        callout text only (510 -> 4 slack marks)
      ["meets_population_rule_in_some_class"]
                                       checked only: false for the least-far cell
      ["chord_gap_ratio"]              = chord std / 15 deg   -> recovers the 15 deg bound
      ["event_gap_ratio"]              = third-level events / 20 -> recovers the 20 bound

  ["cells"][cell]["snap_exceedances"][third_level]["events_by_class"]
                                       checked only: per-class counts at the least-far
                                       cell's third level (all < 20)
  ["closest_to_both"]["least_far"]     the least-far cell (i035_T0600_ks3)
  ["closest_to_both"]["ranked"][0]     ["combined"] (2.39), ["shape_shortfall_factor"]
                                       (2.39), ["population_shortfall_factor"] (0.45,
                                       i.e. the POOLED half is met): callout on the figure
  ["P1_T11"][cell]["chord_world_stat_deg"], ["verdict"]
                                       cross-check only: identical to the trade_off x
                                       values; every verdict is "FAIL"

Not plotted: the two k_sigma = 0 comparison rows (sway term removed).  They lie outside
the selected controller, the declared ranking ["closest_to_both"]["ranked"] and the
P1-T10' count (["P1_T10prime"]["comparison_cells"]).  Their y is also undefined: neither
has a threshold grid, for the same empirical reason as i035_T1000_ks3 (their pilot seeds
held no mark), although i050_T1000_ks0 holds 66 events on its statistics seeds.  Their
chord std (41.1 and 52.6 deg) is farther from the shape band than any plotted cell.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

from tether.analysis.v2.paperfig.style import *  # noqa: F401,F403
from tether.analysis.v2.paperfig.style import (
    COL, C_ACCENT, C_BAND, C_FLEET, C_MUTED, C_NEUTRAL, C_OK, REPO,
    save, use_paper_style,
)

RECORD = REPO / "records" / "v2" / "phase1" / "stochastic_results.json"
SELECTED_K_SIGMA = 3.0
FS = 7.0                       # every piece of text on the figure, in points
# text drawn in the accent hue is darkened so it keeps contrast in greyscale print
TEXT_COL = {C_FLEET: C_FLEET, C_ACCENT: "#9a6200"}


# ------------------------------------------------------------------ records

def load() -> dict:
    with RECORD.open() as fh:
        doc = json.load(fh)
    rows = [r for r in doc["trade_off"] if r["k_sigma"] == SELECTED_K_SIGMA]
    closest = doc["closest_to_both"]

    # The two admissible bounds, recovered from the record's own gap ratios rather than
    # typed in: chord_gap_ratio = chord std / 15 deg, event_gap_ratio = events / 20.
    shape_bounds = {r["chord_world_std_deg"] / r["chord_gap_ratio"] for r in rows}
    pop_bounds = {r["events_at_third_level"] / r["event_gap_ratio"]
                  for r in rows if r["event_gap_ratio"] > 0}
    shape_bound = float(np.mean(list(shape_bounds)))
    pop_bound = float(np.mean(list(pop_bounds)))
    assert np.ptp(list(shape_bounds)) < 1e-9 and abs(shape_bound - 15.0) < 1e-9
    assert np.ptp(list(pop_bounds)) < 1e-9 and abs(pop_bound - 20.0) < 1e-9

    ranked_cells = {item["cell"] for item in closest["ranked"]}
    assert ranked_cells == {r["cell"] for r in rows}, "plotted set != declared ranking set"
    for r in rows:
        t11 = doc["P1_T11"][r["cell"]]
        assert abs(t11["chord_world_stat_deg"] - r["chord_world_std_deg"]) < 1e-9
        assert t11["verdict"] == "FAIL"
        assert r["psi_std_deg"] < r["chord_world_std_deg"]   # chord decides the shape half
        assert r["chord_world_std_deg"] > shape_bound          # no cell in the shape band
        # a row without a grid carries a conventional 0, not a measured count
        r["defined"] = r["third_level_N"] is not None
        assert r["defined"] == bool(r["grid_levels_N"])
        if not r["defined"]:
            assert r["events_at_third_level"] == 0

    # k_h is co-scheduled with T0 (declared heading-gain rule): it rises with T0
    kh = sorted({(r["T0_kN"], r["k_h"]) for r in rows})
    assert all(b[1] > a[1] for a, b in zip(kh, kh[1:]))

    # the least-far cell meets the POOLED population factor but not the per-class rule
    least = closest["least_far"]
    lrow = next(r for r in rows if r["cell"] == least)
    third = f"{lrow['third_level_N']:.0f}"
    by_class = doc["cells"][least]["snap_exceedances"][third]["events_by_class"]
    assert sum(by_class.values()) == lrow["events_at_third_level"] >= pop_bound
    assert max(by_class.values()) < pop_bound
    assert lrow["meets_population_rule_in_some_class"] is False
    return {"rows": rows, "closest": closest, "by_class": by_class, "kh": kh,
            "shape_bound": shape_bound, "pop_bound": pop_bound}


# ------------------------------------------------------------------ helpers

def _offset_arrow(ax, p0, p1, offset_pt, frac=(0.22, 0.78), **kw):
    """Arrow parallel to the data segment p0 -> p1, shifted sideways by offset_pt points."""
    fig = ax.figure
    to_disp = ax.transData.transform
    a, b = to_disp(p0), to_disp(p1)
    d = b - a
    n = np.array([-d[1], d[0]]) / np.hypot(*d)
    shift = n * offset_pt * fig.dpi / 72.0
    s = a + frac[0] * d + shift
    e = a + frac[1] * d + shift
    inv = ax.transData.inverted()
    ax.add_patch(FancyArrowPatch(inv.transform(s), inv.transform(e), **kw))
    angle = float(np.degrees(np.arctan2(d[1], d[0])))
    return inv.transform(0.5 * (s + e)), n, angle


# ------------------------------------------------------------------ figure

def main() -> None:
    data = load()
    rows, closest = data["rows"], data["closest"]
    X_BOUND, Y_BOUND = data["shape_bound"], data["pop_bound"]
    by_id = {r["cell"]: r for r in rows}

    use_paper_style()
    fig, ax = plt.subplots(figsize=(COL, 3.0))

    # Log y axis for the measured counts; a separate strip at the bottom holds the two
    # cells whose third-level count is undefined (no threshold grid).
    xlim = (0.0, 60.0)
    Y_LO, STRIP_TOP, Y_UNDEF, Y_HI = 1.0, 2.1, 1.45, 450.0
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(Y_LO, Y_HI)

    # ------------------------------------------------------ the two admissible bands
    shape_edge = "#4f6f8c"
    pop_edge = "#2e7d5b"
    ax.axvspan(xlim[0], X_BOUND, ymin=0, ymax=1, color=C_BAND, alpha=0.75, lw=0, zorder=0)
    ax.axhspan(Y_BOUND, Y_HI, color=C_OK, alpha=0.12, lw=0, zorder=0)
    ax.axvline(X_BOUND, color=shape_edge, lw=0.7, ls=(0, (1.2, 1.2)), zorder=1)
    ax.axhline(Y_BOUND, color=pop_edge, lw=0.7, ls=(0, (1.2, 1.2)), zorder=1)

    # the "no grid" strip: not part of the count axis
    ax.axhspan(Y_LO, STRIP_TOP, color="white", lw=0, zorder=1.5)
    ax.axhspan(Y_LO, STRIP_TOP, facecolor="none", edgecolor=C_MUTED, hatch="....",
               lw=0, alpha=0.35, zorder=1.6)
    ax.axhline(STRIP_TOP, color=C_NEUTRAL, lw=0.6, zorder=1.7)
    ax.text(xlim[1] - 0.8, Y_UNDEF, "count undefined",
            ha="right", va="center", fontsize=FS, color=C_NEUTRAL, zorder=5,
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))

    # their intersection: the operating window the design needed, and it is empty
    plt.rcParams["hatch.color"] = "#9aa9b8"
    plt.rcParams["hatch.linewidth"] = 0.5
    ax.add_patch(Rectangle(
        (xlim[0], Y_BOUND), X_BOUND - xlim[0], Y_HI - Y_BOUND,
        facecolor="none", edgecolor=C_NEUTRAL, lw=1.0, ls=(0, (3.0, 1.6)),
        hatch="////", zorder=2,
    ))
    ax.text(X_BOUND / 2.0, 90.0, "operating\nwindow:\nno cell",
            ha="center", va="center", fontsize=FS, color=C_NEUTRAL, weight="bold",
            linespacing=1.1, zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.92))

    ax.text(X_BOUND / 2.0, 5.2, "shape\nband\n$\\leq$%.0f$\\degree$" % X_BOUND,
            ha="center", va="center", fontsize=FS, color=shape_edge,
            linespacing=1.1, zorder=3)
    ax.text(xlim[1] - 0.8, 250.0,
            "$\\geq$%.0f events, classes pooled\n"
            "(declared descriptive measure,\n"
            "not per-class rule P1-T10$'$)" % Y_BOUND,
            ha="right", va="center", fontsize=FS, color=pop_edge, linespacing=1.1,
            zorder=3)

    # --------------------------------------------------------------- measurements
    series = [
        (0.35, C_FLEET, "o", "-", "intensity 0.35"),
        (0.50, C_ACCENT, "s", (0, (4.0, 1.6)), "intensity 0.50"),
    ]
    # Label placement (offset points, alignment) per cell; purely typographic.
    LABEL = {
        "i035_T0600_ks3": ((3, 3), "left", "bottom"),
        "i035_T0800_ks3": ((-5, 0), "right", "center"),
        "i035_T1000_ks3": ((-5, 0), "right", "center"),
        "i050_T0600_ks3": ((0, -6), "center", "top"),
        "i050_T1000_ks3": ((5, -2), "left", "top"),
        "i050_T1400_ks3": ((5, 0), "left", "center"),
    }
    ypos = lambda r: r["events_at_third_level"] if r["defined"] else Y_UNDEF  # noqa: E731
    groups, handles = {}, []
    for intensity, colour, marker, ls, label in series:
        grp = sorted((r for r in rows if r["intensity"] == intensity),
                     key=lambda r: r["T0_kN"])
        groups[intensity] = grp
        for a, b in zip(grp, grp[1:]):
            seg_ls = ls if (a["defined"] and b["defined"]) else (0, (1.0, 1.4))
            ax.plot([a["chord_world_std_deg"], b["chord_world_std_deg"]],
                    [ypos(a), ypos(b)], ls=seg_ls, color=colour, lw=1.15, zorder=6)
        for r in grp:
            x, y = r["chord_world_std_deg"], ypos(r)
            ax.plot([x], [y], marker=marker, ms=4.4, ls="none", zorder=7,
                    mfc=colour if r["defined"] else "white",
                    mec=colour if not r["defined"] else "white",
                    mew=0.9 if not r["defined"] else 0.6)
            off, ha, va = LABEL[r["cell"]]
            ax.annotate(f"{r['T0_kN']:.1f}", (x, y), textcoords="offset points",
                        xytext=off, ha=ha, va=va, fontsize=FS, color=TEXT_COL[colour],
                        zorder=8)
        handles.append(Line2D([], [], ls=ls, color=colour, lw=1.15, marker=marker,
                              ms=4.4, mfc=colour, mec="white", mew=0.6, label=label))

    # direction of increasing pretension (and co-scheduled k_h), alongside each line's
    # first segment
    for intensity, colour, frac in ((0.35, C_FLEET, (0.30, 0.86)),
                                    (0.50, C_ACCENT, (0.18, 0.72))):
        g = groups[intensity]
        p0 = (g[0]["chord_world_std_deg"], ypos(g[0]))
        p1 = (g[1]["chord_world_std_deg"], ypos(g[1]))
        mid, normal, ang = _offset_arrow(
            ax, p0, p1, offset_pt=-6.0, frac=frac,
            arrowstyle="-|>,head_length=3.2,head_width=1.8", lw=0.8,
            color=colour, zorder=5, shrinkA=0, shrinkB=0,
        )
        rot = ang - 180.0 if ang > 90.0 else (ang + 180.0 if ang < -90.0 else ang)
        ax.annotate("raise $T_0$, $k_h$", mid, textcoords="offset points",
                    xytext=tuple(-5.5 * normal), rotation=rot,
                    rotation_mode="anchor", ha="center",
                    va="top" if (-normal[1]) < 0 else "bottom",
                    fontsize=FS, style="italic", color=TEXT_COL[colour], zorder=5)

    # --------------------------------------------------------------------- callouts
    least = by_id[closest["least_far"]]
    rank0 = closest["ranked"][0]
    assert rank0["cell"] == least["cell"]
    assert rank0["population_shortfall_factor"] < 1.0           # pooled half met
    assert abs(rank0["combined"] - rank0["shape_shortfall_factor"]) < 1e-12
    xl, yl = least["chord_world_std_deg"], least["events_at_third_level"]
    ax.add_patch(FancyArrowPatch(
        (xl, yl), (X_BOUND, yl), arrowstyle="<|-|>,head_length=3.2,head_width=1.8",
        lw=0.8, color=C_NEUTRAL, shrinkA=3.5, shrinkB=0.0, zorder=4))
    ax.text(X_BOUND + 0.8, yl * 1.12,
            f"$\\times${rank0['combined']:.2f}, shape only",
            ha="left", va="bottom", fontsize=FS, color=C_NEUTRAL, zorder=5)

    a, b = by_id["i035_T0600_ks3"], by_id["i035_T1000_ks3"]
    destroyed = 100.0 * (1.0 - b["marks"] / a["marks"])
    ax.text(
        59.4, 5.0,
        f"int. 0.35, $T_0$ 0.6$\\to$1.0 kN:\n"
        f"chord {a['chord_world_std_deg']:.1f}$\\to${b['chord_world_std_deg']:.1f}$\\degree$\n"
        f"marks {a['marks']}$\\to${b['marks']} ($-${destroyed:.1f}%)",
        ha="right", va="center", fontsize=FS, color=C_NEUTRAL,
        linespacing=1.2, zorder=5,
    )

    # -------------------------------------------------------------------- cosmetics
    ax.set_yticks([Y_UNDEF, 10, Y_BOUND, 100])
    ax.set_yticklabels(["no\ngrid", "10", f"{Y_BOUND:.0f}", "100"])
    ax.set_yticks([], minor=True)
    ax.set_xticks([0, 15, 30, 45, 60])
    ax.set_xlabel("world chord-angle circular std,\nmax over vessels (deg)", fontsize=FS)
    ax.set_ylabel("events at third grid level,\nclasses pooled (count)", fontsize=FS)
    ax.tick_params(labelsize=FS)
    ax.grid(True, which="major", axis="both", color=C_MUTED, lw=0.35, alpha=0.25)
    ax.set_axisbelow(True)

    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2,
              handlelength=2.0, columnspacing=0.8, handletextpad=0.4,
              borderaxespad=0.15, fontsize=FS)
    ax.text(1.0, 1.03, "labels: $T_0$ (kN)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=FS, color=C_NEUTRAL)

    save(fig, "fig_window")

    for r in rows:
        y = r["events_at_third_level"] if r["defined"] else "undefined (no grid)"
        print(f"  {r['cell']}: T0={r['T0_kN']:.1f} kN k_h={r['k_h']:.0f}"
              f"  chord={r['chord_world_std_deg']:.2f} deg"
              f"  psi={r['psi_std_deg']:.2f} deg  N3(pooled)={y}  marks={r['marks']}")
    print(f"  least-far {least['cell']}: combined x{rank0['combined']:.3f},"
          f" shape x{rank0['shape_shortfall_factor']:.3f},"
          f" pooled population x{rank0['population_shortfall_factor']:.3f};"
          f" per-class at third level {data['by_class']}")
    print(f"  marks destroyed 0.6->1.0 kN @0.35: {destroyed:.2f}%")


if __name__ == "__main__":
    main()
