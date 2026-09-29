"""fig_v3_criticality -- rho(K) across the grid and along the sweep, and theta predicted against measured (T3.2, T3.4).

Panel (a): the spectral radius rho of the chosen kernel per cell, placed by weather intensity and
pretension (marker area ∝ number of events; fan cells square; non-operable cells hollow), with the
T0 = 0.6 kN sweep joined and its seed-bootstrap intervals, and the rho = 1 level [MEASURED kernel
inputs: pilot margin law, statistics-seed peaks].  Panel (b): theta_pred = 1/(1^T (I-K)^-1 pi) against
theta_runs (primary events / all events, cross-cable tree) per cell with intervals and the declared
±0.10 band, the Ferro-Segers estimate as a small grey marker.

Records read: records/v3/wp3_results.json (cells[name].{rho, rho_ci95, theta_pred, theta_runs,
theta_runs_ci95, theta_ferro_segers, operable, n_events, verdicts}; tests.T3.4.{branch, table,
critical_intensity?, margin?}; tests.T3.2.verdict; kernel_path).  Result reproduced here: the T3.4
branch is re-derived from the plotted intervals with tether.campaign.v3.analyse_wp3._sweep_branch and
asserted equal.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from tether.analysis.v2.paperfig.style import WIDE, C_NEUTRAL, save, use_paper_style
from tether.analysis.v3.paperfig.common_v3 import C_MEASURED, C_PREDICTED, C_REFERENCE, band, cell_label, is_fan, load_json, tidy
from tether.campaign.v3 import cells
from tether.campaign.v3.analyse_wp3 import _sweep_branch


def main() -> None:
    use_paper_style()
    wp3 = load_json("wp3_results.json")
    scored = {n: c for n, c in wp3["cells"].items() if "rho" in c}
    sweep = {n: scored[n] for n in cells.sweep_cells() if n in scored}
    branch = _sweep_branch(sweep)
    assert branch["branch"] == wp3["tests"]["T3.4"]["branch"], (branch["branch"], wp3["tests"]["T3.4"]["branch"])
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.6), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    ax.axhline(1.0, color=C_NEUTRAL, lw=0.8)
    ax.text(0.99, 1.02, "ρ = 1 (critical)", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6)
    for name, c in scored.items():
        p = cells.parse_cell(name)
        if p["stiffness_factor"] != 1.0:
            continue
        marker = "s" if is_fan(name) else "o"
        size = 3.0 + 2.5 * np.log10(max(c["n_events"], 10))
        colour = C_MEASURED if p["pretension"] == 600.0 else C_REFERENCE
        ax.plot(p["intensity"], c["rho"], marker, ms=size, color=colour if c["operable"] else "none", mec=colour, mew=0.9, zorder=4)
    xs = [cells.parse_cell(n)["intensity"] for n in sweep]
    if sweep:
        order = np.argsort(xs)
        names = [list(sweep)[k] for k in order]
        ax.plot([cells.parse_cell(n)["intensity"] for n in names], [sweep[n]["rho"] for n in names], color=C_MEASURED, lw=1.4, zorder=3)
        for n in names:
            ax.plot([cells.parse_cell(n)["intensity"]] * 2, sweep[n]["rho_ci95"], color=C_MEASURED, lw=1.2, zorder=3)
    if branch["branch"] == "(a)":
        ax.axvline(branch["critical_intensity"], color=C_PREDICTED, lw=1.0, ls="--")
        ax.text(branch["critical_intensity"], 0.05, f" I_c ≈ {branch['critical_intensity']:.2f}", fontsize=6.3, color=C_PREDICTED)
    ax.set_xlabel("weather intensity")
    ax.set_ylabel("spectral radius ρ(K)")
    ax.set_ylim(0.0, max(1.3, 1.1 * max(c["rho_ci95"][1] for c in scored.values())) if scored else 1.3)
    ax.plot([], [], "o", color=C_MEASURED, ms=5, label="T0 = 0.6 kN sweep (95 % seed interval)")
    ax.plot([], [], "o", color=C_REFERENCE, ms=5, label="other pretensions")
    ax.plot([], [], "s", mfc="none", mec=C_REFERENCE, ms=5, label="fan; hollow = not operable")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=6.0, handletextpad=0.4, columnspacing=1.0)
    ax.set_title(f"(a) T3.4 branch {branch['branch']}; kernel: {wp3['kernel_path']}", fontsize=7.5, loc="left")
    tidy(ax)
    ax = axes[1]
    band(ax, -0.1, 0.1, label="±0.10")
    ax.axhline(0.0, color=C_NEUTRAL, lw=0.8)
    names = [n for n in scored if np.isfinite(scored[n]["theta_pred"] or np.nan)]
    x = np.arange(len(names))
    for k, n in enumerate(names):
        c = scored[n]
        diff = c["theta_pred"] - c["theta_runs"]
        ci = [c["theta_pred"] - c["theta_runs_ci95"][1], c["theta_pred"] - c["theta_runs_ci95"][0]]
        ax.plot([k, k], ci, color=C_MEASURED, lw=1.2)
        ax.plot(k, diff, "s" if is_fan(n) else "o", ms=5, color=C_MEASURED, mec="white", mew=0.8, zorder=4)
        if c.get("theta_ferro_segers") is not None and np.isfinite(c["theta_ferro_segers"]):
            ax.plot(k, c["theta_pred"] - c["theta_ferro_segers"], "_", ms=6, color=C_REFERENCE, mew=1.2, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([cell_label(n) for n in names], fontsize=5.5, rotation=60, ha="right")
    ax.set_ylabel("θ_pred − θ_runs")
    ax.set_title(f"(b) T3.2 {wp3['tests']['T3.2']['verdict']}; grey tick: Ferro–Segers θ", fontsize=7.5, loc="left")
    tidy(ax)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.34, wspace=0.3)
    save(fig, "fig_v3_criticality")
    print(f"  T3.4 {branch['branch']}, T3.2 {wp3['tests']['T3.2']['verdict']}")


if __name__ == "__main__":
    main()
