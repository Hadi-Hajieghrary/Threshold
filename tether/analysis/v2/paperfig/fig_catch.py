"""fig_catch -- the slack-phase catch does not bring the closing speed under v_b (P6-T0).

Only measured, pre-declared quantities are drawn.  No mechanism is argued: the post-hoc
stern/load decomposition (p6_t0_mechanism.json, "not declared, gates nothing") is NOT read
and no thrust-authority bound is drawn.

Panel (a): per-excursion closing speed at re-engagement relative to the critical speed of its
cable, v/v_b, for the 16 first-severance excursions, in the primary (fleet) mode under three
controllers: the replay of the recorded thrust command (controller "replay", the declared
validation run -- the mission as recorded, no catch), and the catch law with the true chord
state at 10 Hz (controller "true10", the verdict controller) in its two declared variants,
thrust-hold (F_min = 0) and thrust-reverse (F_min = -F_T).  Thin grey lines join the three
values of one excursion.  v/v_b = 1 is the success line: an excursion is caught iff it
re-engages with v < v_b.  The vertical ticks are medians.

Panel (b): the measured caught fraction of both variants with its Clopper-Pearson 95% interval,
against the pre-declared admissibility bar (caught fraction >= 0.80).

Records read -- every plotted number comes from these files, nothing is typed in:

  records/v2/phase6/p6_t0_results.json        [MEASURED]
      /excursions[j] with tags.first_severance == true (16 of 43):
          /tags/{seed, cable, critical_speed, v_return}
          /runs[mode=='fleet', controller=='replay', variant==None]/{ratio, v_cf, caught, outcome}
          /runs[mode=='fleet', controller=='true10', variant in {hold, reverse}]/
              {ratio, v_cf, caught, outcome}                 -- panel (a) dots
          (asserted: ratio == v_cf / critical_speed; caught <=> reengaged with ratio < 1;
           replay v_cf reproduces the recorded v_return)
      /tables/fleet/true10/first_severance/{hold,reverse}/{median_ratio, caught, n,
          interval95}                                         -- medians (asserted equal to the
                                                                 medians of the dots), counts
      /verdict/{fractions, intervals95, n, admissible, rule, outcome}      -- panel (b)
      /critical_speeds                                                     -- v_b per cable

  records/v2/phase6/p6_t0_declarations.json   [PRE-REGISTERED RULE]
      /admissibility_rule                                                  -- the 0.80 bar

No committed_prediction / model curve is drawn: every series is a measurement.  The one
statistic not tabulated in the record is the median of the 16 replay ratios, computed here
from the per-run values (a plain median; it gates nothing).
"""
from __future__ import annotations

import json
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tether.analysis.v2.paperfig.style import (
    COL, C_PERLINE, C_NEUTRAL, C_MUTED, C_ACCENT, C_OK,
    use_paper_style, save, REPO,
)

P6 = REPO / "records" / "v2" / "phase6"
RES = P6 / "p6_t0_results.json"
DECL = P6 / "p6_t0_declarations.json"

POP = "first_severance"   # the gating population of the declared admissibility rule
MODE = "fleet"            # the primary mode (verdict table)
CONDS = (                 # (key, controller, variant, tick label, marker, colour, filled)
    ("replay", "replay", None, "recorded thrust\n(replay, no catch)", "D", C_NEUTRAL, False),
    ("hold", "true10", "hold", "catch:\nthrust-hold", "o", C_PERLINE, True),
    ("reverse", "true10", "reverse", "catch:\nthrust-reverse", "s", C_PERLINE, True),
)


# ------------------------------------------------------------------ data

def load_data() -> dict:
    res = json.loads(RES.read_text())
    decl = json.loads(DECL.read_text())

    exc = [e for e in res["excursions"] if e["tags"][POP]]
    n = len(exc)
    assert n == int(res["verdict"]["n"]) == int(res["population_sizes"][POP]) == 16, n
    tab = res["tables"][MODE]["true10"][POP]

    ratio = {k: [] for k, *_ in CONDS}
    caught = {k: [] for k, *_ in CONDS}
    ids = []
    for e in exc:
        t = e["tags"]
        vb = float(t["critical_speed"])
        assert any(abs(vb - c) < 1e-12 for c in res["critical_speeds"])
        ids.append((int(t["seed"]), int(t["cable"])))
        for k, ctrl, var, *_ in CONDS:
            runs = [r for r in e["runs"]
                    if r["mode"] == MODE and r["controller"] == ctrl and r["variant"] == var]
            assert len(runs) == 1, (t["seed"], t["cable"], k)
            r = runs[0]
            assert r["outcome"] == "reengaged"          # no closure / horizon censoring here
            q = float(r["ratio"])
            assert abs(q - float(r["v_cf"]) / vb) < 1e-9
            assert bool(r["caught"]) == (q < 1.0)
            if k == "replay":                           # validation run reproduces the record
                assert abs(float(r["v_cf"]) - float(t["v_return"])) < 0.01
            ratio[k].append(q)
            caught[k].append(bool(r["caught"]))
    ratio = {k: np.array(v) for k, v in ratio.items()}
    caught = {k: np.array(v) for k, v in caught.items()}

    med = {k: float(np.median(v)) for k, v in ratio.items()}
    for v in ("hold", "reverse"):
        assert abs(med[v] - float(tab[v]["median_ratio"])) < 1e-9, (v, med[v])
        assert int(caught[v].sum()) == int(tab[v]["caught"])

    # panel (b): verdict fractions and intervals, with counts from the gating table
    ver = res["verdict"]
    m = re.search(r">=\s*([0-9.]+)", decl["admissibility_rule"])
    bar = float(m.group(1))
    assert abs(bar - 0.80) < 1e-12 and decl["admissibility_rule"] == ver["rule"]
    cells = {}
    for v in ("hold", "reverse"):
        frac = float(ver["fractions"][v])
        lo, hi = map(float, ver["intervals95"][v])
        c, nn = int(tab[v]["caught"]), int(tab[v]["n"])
        assert abs(c / nn - frac) < 1e-12 and nn == n
        assert np.allclose(tab[v]["interval95"], [lo, hi])
        cells[v] = dict(caught=c, n=nn, fraction=frac, lo=lo, hi=hi,
                        admissible=bool(ver["admissible"][v]))

    caught_ids = {k: [ids[i] for i in np.flatnonzero(caught[k])] for k in caught}
    return dict(ratio=ratio, caught=caught, caught_ids=caught_ids, med=med, n=n,
                cells=cells, bar=bar, outcome=ver["outcome"])


# ------------------------------------------------------------------ panel (a)

def panel_ratio(ax, d):
    ypos = np.arange(len(CONDS), dtype=float)
    n = d["n"]
    # one fixed vertical offset per excursion, the same in every row, so the joining
    # lines stay readable
    jit = np.linspace(-0.22, 0.22, n)[np.random.default_rng(3).permutation(n)]

    ax.axvspan(0.0, 1.0, color=C_OK, alpha=0.14, lw=0, zorder=0)
    ax.axvline(1.0, color=C_OK, lw=1.1, ls=(0, (4, 2)), zorder=3)
    ax.text(0.5, ypos[1], "caught ($v<v_b$)", ha="center", va="center", rotation=90,
            fontsize=7, color=C_OK, zorder=6)

    # grey lines joining the three values of one excursion
    stack = np.vstack([d["ratio"][k] for k, *_ in CONDS])       # (3, n)
    for j in range(n):
        ax.plot(stack[:, j], ypos + jit[j], color=C_MUTED, lw=0.35, alpha=0.7, zorder=1.5)

    for y, (k, _c, _v, _lab, mk, col, filled) in zip(ypos, CONDS):
        vals = d["ratio"][k]
        ax.plot(vals, y + jit, ls="none", marker=mk, ms=2.9 if mk != "D" else 2.6,
                mfc=col if filled else "white", mec=col, mew=0.6, alpha=0.9, zorder=4)
        med = d["med"][k]
        ax.plot([med, med], [y - 0.34, y + 0.34], color=col, lw=1.8, zorder=5,
                solid_capstyle="butt")
        ax.text(9.45, y, f"{med:.2f}", ha="right", va="center", fontsize=7, color=col)
    ax.text(9.45, ypos[0] - 0.62, "median", ha="right", va="center", fontsize=7,
            color=C_MUTED)

    ax.set_yticks(ypos)
    ax.set_yticklabels([c[3] for c in CONDS], fontsize=7, linespacing=0.95)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(ypos[-1] + 0.6, ypos[0] - 0.85)      # replay row at the top
    ax.set_xlim(0.0, 9.5)
    ax.set_xticks([0, 1, 2, 4, 6, 8])
    ax.set_xlabel(r"closing speed at re-engagement, $v/v_b$", labelpad=1.5)
    ax.grid(axis="x", ls=":", zorder=0)
    ax.set_axisbelow(True)

    handles = [
        Line2D([], [], color=C_MUTED, lw=0.6, label=f"same excursion ($n$={n})"),
        Line2D([], [], color=C_NEUTRAL, lw=1.8, label="median"),
        Line2D([], [], color=C_OK, lw=1.1, ls=(0, (4, 2)), label="$v = v_b$"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3,
              fontsize=7, handlelength=1.5, handletextpad=0.4, columnspacing=0.9,
              borderaxespad=0.1)
    return {k: d["med"][k] for k, *_ in CONDS}


# ------------------------------------------------------------------ panel (b)

def panel_verdict(ax, d):
    variants = [("hold", "catch:\nthrust-hold", "o"), ("reverse", "catch:\nthrust-reverse", "s")]
    bar = d["bar"]
    ax.axvspan(bar, 1.0, color=C_OK, alpha=0.14, lw=0, zorder=0)
    ax.axvline(bar, color=C_ACCENT, lw=1.1, ls=(0, (4, 2)), zorder=3)
    ax.text(bar - 0.015, -0.72, f"pre-declared admissibility\nbar: fraction $\\geq$ {bar:.2f}",
            ha="right", va="center", fontsize=7, color=C_ACCENT, linespacing=0.95)

    for y, (v, _lab, mk) in enumerate(variants):
        c = d["cells"][v]
        ax.errorbar(c["fraction"], y, xerr=[[c["fraction"] - c["lo"]], [c["hi"] - c["fraction"]]],
                    fmt=mk, ms=3.6, color=C_PERLINE, mfc=C_PERLINE, mec=C_PERLINE,
                    elinewidth=1.0, capsize=2.2, capthick=1.0, zorder=4)
        ax.text(c["hi"] + 0.025, y, f"{c['caught']}/{c['n']} caught", ha="left",
                va="center", fontsize=7, color=C_NEUTRAL)

    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([v[1] for v in variants], fontsize=7, linespacing=0.95)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(len(variants) - 0.45, -1.2)
    ax.set_xlim(-0.04, 1.0)                       # keeps the 0.0016 cap off the spine
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("caught fraction (Clopper-Pearson 95% CI)", labelpad=1.5)
    ax.grid(axis="x", ls=":", zorder=0)
    ax.set_axisbelow(True)
    return {v: d["cells"][v] for v, *_ in variants}


def main() -> None:
    use_paper_style()
    d = load_data()

    fig = plt.figure(figsize=(COL, 2.95))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.75, 1.0], hspace=0.62,
                          left=0.255, right=0.97, top=0.90, bottom=0.115)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    info_a = panel_ratio(ax_a, d)
    info_b = panel_verdict(ax_b, d)

    fig.text(0.005, ax_a.get_position().y1 + 0.035, "(a)", ha="left", va="bottom",
             fontsize=8, fontweight="bold")
    fig.text(0.005, ax_b.get_position().y1 + 0.005, "(b)", ha="left", va="bottom",
             fontsize=8, fontweight="bold")

    save(fig, "fig_catch")
    print("panel (a) medians v/v_b:", info_a)
    print("panel (a) caught excursions (seed, cable):", d["caught_ids"])
    print("panel (a) ranges:", {k: (float(v.min()), float(v.max())) for k, v in d["ratio"].items()})
    print("panel (b):", info_b, "outcome:", d["outcome"])


if __name__ == "__main__":
    main()
