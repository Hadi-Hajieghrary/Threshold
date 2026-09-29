"""fig_v3_mx_transfer -- the offspring curve m_x and its transfer across cells and geometry (T2.1, T2.2).

Panel (a): the cross-cable offspring per event against the parent peak x = T_peak / T0 in the
calibration cell: the binned means on the statistics seeds [MEASURED] and the isotonic fit on the
two pilot seeds [FITTED, the one fitted curve of the programme].  Panel (b): per cell, the offspring
count predicted by transferring that fit unchanged against the measured count, on log axes with the
declared factor-1.5 band; fan cells carry a square marker.

Records read: records/v3/wp2_results.json (isotonic_fit.{x, y, n, cell, seeds}; cells[name].T2.1.{labels,
counts, means, verdict}; cells[name].T2.2.{predicted, ratio, verdict, formation};
cells[name].measured_cross_offspring; tests.T2.1/T2.2 verdicts); records/v3/wp2_offspring.npz
(event_T_peak, event_n_cross, event_cell_index, event_pilot, cell_names, for the calibration cell's
event cloud in panel (a)).  Result reproduced here: predicted / measured of wp2_results equals the
stored ratio (asserted); the binned means are re-derived from the npz and asserted equal.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from tether.analysis.v2.paperfig.style import WIDE, C_NEUTRAL, save, use_paper_style
from tether.analysis.v3.paperfig.common_v3 import (
    C_MEASURED, C_PREDICTED, C_REFERENCE, cell_label, cell_order, factor_band, is_fan, load_json, load_npz, tidy,
)
from tether.campaign.v3 import campaign, cells
from tether.campaign.v3.analyse_wp2 import binned_mx


def main() -> None:
    use_paper_style()
    wp2 = load_json("wp2_results.json")
    npz = load_npz("wp2_offspring.npz")
    calib = cells.CALIBRATION_CELL
    names = list(npz["cell_names"])
    k = names.index(calib)
    t0 = cells.parse_cell(calib)["pretension"]
    sel = (npz["event_cell_index"] == k) & (~npz["event_pilot"])
    x, n_cross = npz["event_T_peak"][sel] / t0, npz["event_n_cross"][sel]
    b = binned_mx(x, n_cross)
    stored = wp2["cells"][calib]["T2.1"]
    for a, c in zip(b["means"], stored["means"]):
        assert (a is None or not np.isfinite(a)) and (c is None) or abs(a - c) < 1e-9, (a, c)
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.5))
    # (a)
    ax = axes[0]
    ax.scatter(x, n_cross + 0.03 * np.random.default_rng(0).standard_normal(x.size), s=5, color=C_MEASURED, alpha=0.25, lw=0, label="events (statistics seeds)")
    edges = (0.5,) + tuple(campaign.COUPLED_PEAK_BINS) + (max(30.0, float(np.nanmax(x)) * 1.1),)
    centres = np.sqrt(np.array(edges[:-1]) * np.array(edges[1:]))
    means = np.array([np.nan if m is None else m for m in stored["means"]], dtype=float)
    ax.plot(centres, means, "o", ms=5, color=C_MEASURED, mec="white", mew=0.8, zorder=4, label="binned mean")
    fit = wp2["isotonic_fit"]
    ax.step(fit["x"], fit["y"], where="post", color=C_PREDICTED, lw=1.6, zorder=3, label=f"isotonic fit, pilots ({fit['n']} events)")
    ax.set_xscale("log")
    ax.set_xlabel("parent peak $x = T_{\\mathrm{peak}} / T_0$")
    ax.set_ylabel("cross-cable offspring per event")
    ax.set_title(f"(a) {cell_label(calib)}: T2.1 {stored['verdict']} (ρ_S = {stored['spearman'] if stored['spearman'] is None else round(stored['spearman'], 2)})", fontsize=7.5, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=6.3)
    tidy(ax)
    # (b)
    ax = axes[1]
    pairs = [(n, wp2["cells"][n]) for n in cell_order(wp2["cells"]) if wp2["cells"][n]["scored"]]
    measured = np.array([c["measured_cross_offspring"] for _, c in pairs])
    predicted = np.array([c["T2.2"]["predicted"] for _, c in pairs])
    for (n, c), m, p in zip(pairs, measured, predicted):
        assert abs(p / m - c["T2.2"]["ratio"]) < 1e-9
    lo, hi = max(1.0, 0.7 * min(measured.min(), predicted.min())), 1.4 * max(measured.max(), predicted.max())
    factor_band(ax, lo, hi)
    for (n, c), m, p in zip(pairs, measured, predicted):
        ax.plot(m, p, "s" if is_fan(n) else "o", ms=5, color=C_MEASURED if c["T2.2"]["verdict"] == "PASS" else C_REFERENCE, mec="white", mew=0.8, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("measured cross-cable offspring per cell")
    ax.set_ylabel("predicted by the transferred fit")
    t22 = wp2["tests"]["T2.2"]
    fan = t22.get("fan_subgroup") or {}
    ax.set_title(f"(b) transfer: {t22['passing_cells']}/{t22['scored_cells']} cells in band, {t22['verdict']}; fan {fan.get('passing_cells', 0)}/{fan.get('scored_cells', 0)}", fontsize=7.5, loc="left")
    ax.plot([], [], "o", color=C_MEASURED, ms=5, label="parallel cell (filled: in band)")
    ax.plot([], [], "s", color=C_MEASURED, ms=5, label="fan cell")
    ax.legend(loc="upper left", frameon=False, fontsize=6.3)
    tidy(ax)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.2, wspace=0.3)
    save(fig, "fig_v3_mx_transfer")
    print(f"  T2.1 {wp2['tests']['T2.1']['verdict']}, T2.2 {t22['verdict']}")


if __name__ == "__main__":
    main()
