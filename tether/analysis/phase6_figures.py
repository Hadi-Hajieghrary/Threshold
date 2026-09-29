"""Phase 6 figure (the P6-T9 hero replay of plan I.5), regenerated from committed records only."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tether.analysis.captions import hedge, register
from tether.campaign.common import FIGURES
from tether.campaign.phase6 import HERO_PATH

WINDOW = (-20.0, 10.0)
COLOURS = {"N": "#c0504d", "P": "#1f4e79"}


def figure_hero() -> str:
    data = np.load(HERO_PATH)
    meta = json.loads(HERO_PATH.with_suffix(".json").read_text())
    breaking = float(data["breaking"])
    critical = np.asarray(data["critical"])
    first = data["N_virtual_first"]
    if np.isfinite(first[0]):
        centre, cable = float(first[0]), int(first[1])
    else:
        # No virtual severance on N: centre on N's largest tension peak.
        peak = np.unravel_index(np.argmax(data["N_q"]), data["N_q"].shape)
        centre, cable = float(data["N_time"][peak[0]]), int(peak[1])
    lo, hi = centre + WINDOW[0], centre + WINDOW[1]
    fig, axes = plt.subplots(5, 1, figsize=(10.0, 11.0), sharex=True)
    ax_h, ax_q, ax_s, ax_e, ax_v = axes
    if "P_hazard" in data.files:
        t = data["P_supervisor_time"]
        ax_h.plot(t, data["P_hazard"][:, cable], color=COLOURS["P"], lw=0.9, label="hazard heard by the supervisor (P)")
        ax_h.axhline(float(data["h_crit"]), color="#888888", ls="--", lw=0.8, label="h_crit (P)")
        ax_s.plot(t, data["P_scale"][:, cable], color=COLOURS["P"], lw=0.9, label="thrust scale (P)")
    ax_s.axhline(1.0, color=COLOURS["N"], lw=0.9, label="thrust scale (N)")
    for arm in ("N", "P"):
        time = data[f"{arm}_time"]
        ax_q.plot(time, data[f"{arm}_q"][:, cable] / 1000.0, color=COLOURS[arm], lw=0.8, label=f"tension, arm {arm}")
        ax_e.plot(time, data[f"{arm}_e"][:, cable], color=COLOURS[arm], lw=0.8, label=f"elongation, arm {arm}")
        marks = data[f"{arm}_marks"]
        marks = marks[marks[:, 0] == cable] if marks.size else marks
        if marks.size:
            ax_v.plot(marks[:, 1], marks[:, 3], "o", color=COLOURS[arm], ms=4, label=f"closing speed at re-engagement, arm {arm}")
    ax_q.axhline(breaking / 1000.0, color="#444444", ls="--", lw=0.8, label="T_b^s")
    ax_v.axhline(float(critical[cable]), color="#444444", ls="--", lw=0.8, label="v_b")
    ax_e.axhline(0.0, color="#888888", lw=0.5)
    for ax, label in ((ax_h, "hazard"), (ax_q, "tension [kN]"), (ax_s, "thrust scale"), (ax_e, "e [m]"), (ax_v, "closing speed [m/s]")):
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    ax_v.set_xlim(lo, hi)
    ax_v.set_xlabel("mission time [s]")
    ax_h.set_title(f"seed {meta['seed']}, cable {cable}: arm N against arm P at T_b^s = {breaking / 1000:.1f} kN (selection rule in the report)", fontsize=9)
    path = FIGURES / "phase6_F1_hero_replay.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    register("phase6_F1", f"{hedge('Prop. 10')}: seed-matched replay of arms N and P at T_b^s = {breaking:.0f} N (recording mode); "
                          "hazard, tension, thrust, depth and closing speed on the cable that severs first under N; exploratory continuation, no gate authority.")
    return str(path)


def main() -> None:
    print(figure_hero())


if __name__ == "__main__":
    main()
