"""Phase 1 closure figures, regenerated exclusively from committed records."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tether.analysis.captions import hedge, register
from tether.campaign.common import FIGURES
from tether.campaign.phase1_close import IMPACT_TABLE_PATH, RECORD_PATH, TAU_W
from tether.physics import constants
from tether.physics.fleet import formation_geometry, two_body_effective_mass


def _load():
    with np.load(RECORD_PATH, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    return arrays, json.loads(IMPACT_TABLE_PATH.read_text())


def figure_excursion_law(arrays) -> str:
    geometry = formation_geometry("parallel")
    names = arrays["cell_names"]
    pretension = np.array([float(str(names[i]).split("_")[0][1:]) for i in arrays["marks_cell_index"]])
    depth = arrays["marks_depth"]
    complete = depth > 0.0
    u = arrays["marks_u_entry"][complete]
    v = arrays["marks_v_return"][complete]
    cable = arrays["marks_cable"][complete]
    mass = np.array([two_body_effective_mass(geometry, int(c)) for c in cable])
    x = np.sqrt(u**2 + 2.0 * pretension[complete] / mass * depth[complete])
    h4 = (arrays["marks_dwell"][complete] < TAU_W / 4.0) & (arrays["marks_n_maxima"][complete] <= 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(x[~h4], v[~h4], s=6, alpha=0.4, color="#c0504d", label="dwell >= tau_w/4 or multi-maximum")
    ax.scatter(x[h4], v[h4], s=6, alpha=0.6, color="#1f4e79", label="(H4),(H5) satisfied")
    top = float(max(x.max(), v.max())) * 1.05
    ax.plot([0, top], [0, top], color="black", lw=1.0, label="identity (Prop. 3, a_bar = a0)")
    ax.set_xlabel("sqrt(u^2 + 2 a0 Delta)  [m/s],  a0 = T0 / m_eff (two-body)")
    ax.set_ylabel("measured re-engagement speed V_up [m/s]")
    ax.set_title("Phase 1(d): energy conversion on stochastic excursions")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    path = FIGURES / "phase1_F1_excursion_law.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    register("phase1_F1", f"{hedge('Prop. 3')}: measured return speed against the energy-conversion prediction with a_bar = a0, every complete stochastic excursion of the eight Phase 1(d) cells.")
    return str(path)


def figure_depth_threshold(arrays) -> str:
    names = arrays["cell_names"]
    pretension = np.array([float(str(names[i]).split("_")[0][1:]) for i in arrays["marks_cell_index"]])
    ratio = arrays["marks_W_rel_max"] / pretension
    depth = arrays["marks_depth"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(ratio, np.maximum(depth, 1.0e-5), s=6, alpha=0.5, color="#1f4e79")
    ax.axvline(1.0, color="black", lw=1.0, ls="--", label="W_rel = T0 (Prop. 4)")
    ax.set_yscale("log")
    ax.set_xlabel("maximum W_rel during the excursion / T0")
    ax.set_ylabel("excursion depth [m]")
    ax.set_title("Phase 1(d): depth against the deep-excursion criterion")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    path = FIGURES / "phase1_F2_depth_threshold.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    register("phase1_F2", f"{hedge('Prop. 4')}: depth of every stochastic excursion against the largest differential radial gust load it experienced.")
    return str(path)


def figure_impact_table(table) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    colours = {600.0: "#9bbb59", 1000.0: "#1f4e79", 1400.0: "#c0504d", 1800.0: "#8064a2"}
    for key, entry in sorted(table.items()):
        style = {0: ":", 1: "--", 2: "-"}[entry["cable"]]
        ax.plot(entry["speeds"], np.asarray(entry["peaks"]) / 1000.0, style, color=colours[entry["pretension"]], marker="o", ms=3,
                label=f"T0={entry['pretension']:.0f} N, cable {entry['cable']}" if entry["cable"] == 2 else None)
    speeds = np.linspace(0.0, 6.0, 10)
    ax.plot(speeds, 7.99 * speeds, color="black", lw=0.8, label="Z = 7.99 kN s/m (analytic single cable)")
    ax.set_xlabel("closing speed at engagement [m/s]")
    ax.set_ylabel("first peak tension [kN]")
    ax.set_title("Production-plant impact table (pivot v_b = Z^-1(T_b))")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    path = FIGURES / "phase1_F3_impact_table.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    register("phase1_F3", "Measured engagement peak against closing speed on the production plant; line styles: centre (solid), inner (dashed), outer (dotted) cable positions.")
    return str(path)


def main() -> list[str]:
    arrays, table = _load()
    FIGURES.mkdir(parents=True, exist_ok=True)
    return [figure_excursion_law(arrays), figure_depth_threshold(arrays), figure_impact_table(table)]


if __name__ == "__main__":
    for item in main():
        print(item)
