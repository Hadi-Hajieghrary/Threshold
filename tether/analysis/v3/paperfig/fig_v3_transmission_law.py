"""fig_v3_transmission_law -- the measured neighbour swing against the paper's band and the exact response (T1.1, T1.3).

Panel (a): per cell, the median normalised neighbour drop, drop / (T_peak * geometry_factor_ij),
over the eligible pairs (event heads, x in [1, 4], every neighbour taut), with its seed-bootstrap
95 % interval [MEASURED]; the exact-response prediction of the same statistic [FORECAST, drawn as an
open marker]; the paper's closed form 0.44 [DERIVED] and its declared band [0.2, 0.9] [PRE-REGISTERED].
Panel (b): per cell, the median of drop / (T_peak * R_ij) over the small-peak parents (x in [1, 2.3])
against the declared factor-1.5 band around 1 [MEASURED vs the exact response].

Records read -- every plotted number comes from these files, nothing is typed in:
  records/v3/wp1_results.json          cells[name].T1.1.{median_drop_norm, ci95, verdict, committed_prediction},
                                       cells[name].T1.3.{median_drop_over_Tpeak_R, ci95, n, verdict},
                                       cells[name].{scored, n_eligible_pairs}, tests.T1.1/T1.3.verdict
  records/v3/cascade_predictions.json  closed_form.{ratio, band_paper};
                                       cells[name].transmission.band_statistic_median_offdiag (asserted equal to
                                       the committed_prediction stored in wp1_results)
Result reproduced here: the per-cell medians and verdicts of wp1_results.json; nothing is recomputed.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from tether.analysis.v2.paperfig.style import COL, C_NEUTRAL, save, use_paper_style
from tether.analysis.v3.paperfig.common_v3 import (
    C_CLOSED, C_MEASURED, C_PREDICTED, C_REFERENCE, FACTOR, band, cell_label, cell_order, is_fan, load_json, tidy,
)


def main() -> None:
    use_paper_style()
    wp1 = load_json("wp1_results.json")
    pred = load_json("cascade_predictions.json")
    lo, hi = pred["closed_form"]["band_paper"]
    closed = pred["closed_form"]["ratio"]
    names = [n for n in cell_order(wp1["cells"]) if wp1["cells"][n]["scored"]]
    assert names, "no scored cell"
    fig, axes = plt.subplots(2, 1, figsize=(COL, 4.3), sharex=True, gridspec_kw={"height_ratios": [1.35, 1.0]})
    x = np.arange(len(names))
    # (a)
    ax = axes[0]
    band(ax, lo, hi, label=f"declared band [{lo}, {hi}]")
    ax.axhline(closed, color=C_CLOSED, lw=1.0, ls="-", label=f"closed form {closed:.2f}")
    for k, name in enumerate(names):
        c = wp1["cells"][name]["T1.1"]
        assert abs(c["committed_prediction"] - pred["cells"][name]["transmission"]["band_statistic_median_offdiag"]) < 1e-12
        marker = "s" if is_fan(name) else "o"
        ax.plot([k, k], c["ci95"], color=C_MEASURED, lw=1.2, solid_capstyle="butt")
        ax.plot(k, c["median_drop_norm"], marker=marker, ms=4.5, color=C_MEASURED, mec="white", mew=0.8, ls="none", zorder=4)
        ax.plot(k, c["committed_prediction"], marker=marker, ms=4.5, mfc="none", mec=C_PREDICTED, mew=1.0, ls="none", zorder=3)
    ax.set_ylabel("median normalised drop\n$\\Delta T_i / (T_{\\mathrm{peak}}\\, g_{ij})$")
    ax.set_ylim(0.0, max(hi + 0.05, 1.0))
    ax.plot([], [], marker="o", color=C_MEASURED, ls="none", ms=4.5, label="measured (95 % seed interval)")
    ax.plot([], [], marker="o", mfc="none", mec=C_PREDICTED, ls="none", ms=4.5, label="exact response (committed forecast)")
    ax.legend(loc="upper left", frameon=False, fontsize=6.3, ncol=2)
    t11 = wp1["tests"]["T1.1"]
    ax.set_title(f"(a) the paper's band: {t11['passing_cells']} of {t11['scored_cells']} scored cells inside, verdict {t11['verdict']}", fontsize=7.5, loc="left")
    tidy(ax)
    # (b)
    ax = axes[1]
    band(ax, 1.0 / FACTOR, FACTOR, label="factor 1.5")
    ax.axhline(1.0, color=C_NEUTRAL, lw=0.8)
    for k, name in enumerate(names):
        c = wp1["cells"][name]["T1.3"]
        if c["verdict"] == "UNSCORED" or not np.isfinite(c["median_drop_over_Tpeak_R"]):
            continue
        marker = "s" if is_fan(name) else "o"
        ax.plot([k, k], c["ci95"], color=C_MEASURED, lw=1.2, solid_capstyle="butt")
        ax.plot(k, c["median_drop_over_Tpeak_R"], marker=marker, ms=4.5, color=C_MEASURED, mec="white", mew=0.8, ls="none", zorder=4)
    ax.set_yscale("log")
    ax.set_ylabel("measured / exact response\n$\\Delta T_i / (T_{\\mathrm{peak}}\\, R_{ij})$")
    t13 = wp1["tests"]["T1.3"]
    ax.set_title(f"(b) small-peak parents (x ≤ 2.3): verdict {t13['verdict']}", fontsize=7.5, loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels([cell_label(n) for n in names], rotation=60, ha="right", fontsize=6)
    tidy(ax)
    fig.subplots_adjust(left=0.19, right=0.99, top=0.95, bottom=0.27, hspace=0.35)
    save(fig, "fig_v3_transmission_law")
    print(f"  {len(names)} scored cells; T1.1 {t11['verdict']} ({t11['passing_cells']}/{t11['scored_cells']}), T1.3 {t13['verdict']}")


if __name__ == "__main__":
    main()
