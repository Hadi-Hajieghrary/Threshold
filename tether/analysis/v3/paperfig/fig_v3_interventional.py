"""fig_v3_interventional -- the transmission measured by intervention (addendum 2 sensitivity, reported beside T1.1-T1.3).

Panel (a): per cell, the median interventional swing max_[t_up, t_up+P] (T_i^cf - T_i^f) / T_peak [MEASURED,
counterfactual minus factual under the same weather], the exact-response prediction R_ij (median over the
same rows) [FORECAST], the declared drop statistic and its no-snap floor (the same window at time-shifted
instants) [MEASURED]; the closed form 0.44 for reference.  Panel (b): for the four cells run on all 22 seeds,
the mirror-pooled pair matrix of the interventional swing above the exact response R, one shared scale,
Spearman rank correlations in the titles (the declared drop's correlation in brackets).

Records read: records/v3/wp1_sensitivity.json (cells[name].{median_swing_over_Tpeak, median_R_ij_of_rows,
median_drop_over_Tpeak, median_null_over_Tpeak, rows, swing_pairs.{pairs, median_swing, R_pooled,
spearman_vs_R}, drop_pairs.spearman_vs_R}; definitions.full_cells), records/v3/wp1_sensitivity.npz (rows).
Result reproduced here: for every full cell the pooled pair medians and the Spearman vs R are re-derived
from the npz rows and asserted equal to the json; the cell-level medians of panel (a) are re-derived for
every cell and asserted equal.
"""

from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from tether.analysis.v2.paperfig.style import WIDE, C_NEUTRAL, save, use_paper_style
from tether.analysis.v3.paperfig.common_v3 import C_MEASURED, C_PREDICTED, C_REFERENCE, C_CLOSED, cell_label, cell_order, is_fan, load_json, load_npz, tidy
from tether.theory import transmission as TR

CMAP = "Blues"


def _pooled(rows: dict, mask: np.ndarray, key: str, min_parents: int = 5):
    seen, out = {}, []
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            a = (i, j) if (i, j) <= (4 - i, 4 - j) else (4 - i, 4 - j)
            if a in seen:
                continue
            sel = mask & (((rows["neighbour_i"] == i) & (rows["cable_j"] == j)) | ((rows["neighbour_i"] == 4 - i) & (rows["cable_j"] == 4 - j)))
            v = rows[key][sel]; v = v[np.isfinite(v)]
            if v.size >= min_parents:
                out.append((list(a), float(np.median(v)), float(np.nanmedian(rows["R_ij"][sel]))))
            seen[a] = True
    return out


def _matrix(pairs, values) -> np.ndarray:
    m = np.full((5, 5), np.nan)
    for (i, j), v in zip(pairs, values):
        m[i, j] = m[4 - i, 4 - j] = v
    return m


def _draw(ax, m, title, vmax):
    im = ax.imshow(np.where(np.isfinite(m), m, np.nan), cmap=CMAP, vmin=0.0, vmax=vmax, origin="upper")
    for i in range(5):
        for j in range(5):
            if i == j:
                ax.plot(j, i, marker="x", color="#b8b8b8", ms=3.5, mew=0.8)
            elif np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=5.0, color="white" if m[i, j] > 0.6 * vmax else C_NEUTRAL)
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.tick_params(labelsize=5.5)
    ax.set_title(title, fontsize=6.5, loc="left")
    return im


def main() -> None:
    use_paper_style()
    s = load_json("wp1_sensitivity.json")
    z = load_npz("wp1_sensitivity.npz")
    names = [n for n in cell_order(s["cells"]) if s["cells"][n]["rows"] > 0]
    idx = {str(n): k for k, n in enumerate(z["cell_names"])}
    # reproduce the cell-level medians from the rows
    for n in names:
        m = z["cell_index"] == idx[n]
        sw = z["swing_over_Tpeak"][m]; sw = sw[np.isfinite(sw)]
        assert abs(float(np.median(sw)) - s["cells"][n]["median_swing_over_Tpeak"]) < 1e-12, n
        nu = z["null_over_Tpeak"][m]; nu = nu[np.isfinite(nu)]
        assert abs(float(np.median(nu)) - s["cells"][n]["median_null_over_Tpeak"]) < 1e-12, n
    full = [n for n in s["definitions"]["full_cells"] if n in s["cells"]]
    fig = plt.figure(figsize=(WIDE, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.5], hspace=1.05)
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(names))
    ax.axhline(TR.closed_form_ratio(), color=C_CLOSED, lw=1.0, label="closed form 0.44")
    for k, n in enumerate(names):
        c = s["cells"][n]
        mk = "s" if is_fan(n) else "o"
        ax.plot(k, c["median_drop_over_Tpeak"], marker=mk, ms=3.2, color=C_REFERENCE, mec="none", zorder=3)
        ax.plot(k, c["median_null_over_Tpeak"], marker=mk, ms=3.2, mfc="none", mec=C_REFERENCE, mew=0.8, zorder=3)
        ax.plot(k, c["median_R_ij_of_rows"], marker=mk, ms=4.5, mfc="none", mec=C_PREDICTED, mew=1.0, zorder=4)
        ax.plot(k, c["median_swing_over_Tpeak"], marker=mk, ms=4.5, color=C_MEASURED, mec="white", mew=0.6, zorder=5)
    ax.plot([], [], "o", color=C_MEASURED, ms=4.5, label="interventional swing (counterfactual − factual)")
    ax.plot([], [], "o", mfc="none", mec=C_PREDICTED, ms=4.5, label="exact response R (forecast)")
    ax.plot([], [], "o", color=C_REFERENCE, ms=3.2, label="declared drop statistic")
    ax.plot([], [], "o", mfc="none", mec=C_REFERENCE, ms=3.2, label="its no-snap floor (time-shifted)")
    ax.set_xticks(x); ax.set_xticklabels([f"{cell_label(n)} ({s['cells'][n]['rows']})" for n in names], rotation=60, ha="right", fontsize=5.3)
    ax.set_ylabel("swing / $T_{\\mathrm{peak}}$")
    ax.set_ylim(0, 0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=5.6, handletextpad=0.4, columnspacing=1.2)
    ax.set_title("(a) per cell: the transmission by intervention against the exact response (rows in brackets)", fontsize=7.5, loc="left", pad=16)
    tidy(ax)
    sub = gs[1].subgridspec(2, len(full), wspace=0.45, hspace=0.6)
    mats = {}
    for n in full:
        m = z["cell_index"] == idx[n]
        pooled = _pooled({k: z[k][m] for k in ("neighbour_i", "cable_j", "swing_over_Tpeak", "R_ij")}, np.ones(int(m.sum()), bool), "swing_over_Tpeak")
        pairs = [p for p, _, _ in pooled]; med = [v for _, v, _ in pooled]; Rp = [r for _, _, r in pooled]
        sp = s["cells"][n]["swing_pairs"]
        assert pairs == sp["pairs"] and np.allclose(med, sp["median_swing"]) and np.allclose(Rp, sp["R_pooled"]), n
        rho = spearmanr(med, Rp).statistic
        assert abs(rho - sp["spearman_vs_R"]) < 1e-9, (n, rho, sp["spearman_vs_R"])
        mats[n] = (_matrix(pairs, med), _matrix(pairs, Rp), rho, s["cells"][n]["drop_pairs"]["spearman_vs_R"])
    vmax = max(float(np.nanmax(a)) for a, b, _, _ in mats.values() for _ in [0]) if mats else 0.3
    vmax = max(vmax, max(float(np.nanmax(b)) for _, b, _, _ in mats.values()))
    for col, n in enumerate(full):
        a, b, rho, rho_drop = mats[n]
        ax1 = fig.add_subplot(sub[0, col]); ax2 = fig.add_subplot(sub[1, col])
        _draw(ax1, a, f"{cell_label(n)}: swing\nρ_S vs R = {rho:.2f}\n(declared drop: {rho_drop:.2f})", vmax)
        im = _draw(ax2, b, "exact response R (forecast)", vmax)
        if col == 0:
            ax1.set_ylabel("neighbour i", fontsize=6); ax2.set_ylabel("neighbour i", fontsize=6)
        ax2.set_xlabel("snapping cable j", fontsize=6)
    cax = fig.add_axes([0.925, 0.08, 0.012, 0.32])
    cbar = fig.colorbar(im, cax=cax); cbar.set_label("swing / $T_{\\mathrm{peak}}$", fontsize=6.5); cbar.ax.tick_params(labelsize=5.5)
    top_b = fig.axes[1].get_position().y1 if len(fig.axes) > 1 else 0.5
    fig.text(0.06, top_b + 0.075, "(b) pair matrices (mirror pairs pooled) in the four cells run on all 22 seeds", fontsize=7.5, ha="left")
    fig.subplots_adjust(left=0.06, right=0.905, top=0.92, bottom=0.07)
    save(fig, "fig_v3_interventional")
    print(f"  {len(names)} cells; pooled swing/R {s['pooled']['median_swing_over_R']:.2f}; Spearman swing vs R:", {n: round(mats[n][2], 2) for n in full})


if __name__ == "__main__":
    main()
