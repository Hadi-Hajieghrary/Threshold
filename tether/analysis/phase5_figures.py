"""Phase 5 figure (the reliability diagram of plan I.5), regenerated from committed records only."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tether.analysis.captions import hedge, register
from tether.campaign.common import FIGURES
from tether.campaign.phase5 import RESULTS_PATH

ARM_STYLE = {"O": ("#444444", "s", "O (oracle)"), "P": ("#1f4e79", "o", "P (proposed)"), "B2": ("#c0504d", "^", "B2 (naive consensus)"), "L": ("#9bbb59", "D", "L (local)")}


def figure_reliability(results: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    ax, ax_count = axes
    ax.plot([0, 1], [0, 1], color="#888888", lw=0.8, ls="--")
    for arm, (colour, marker, label) in ARM_STYLE.items():
        calibration = (results["arms"].get(arm) or {}).get("calibration")
        if not calibration:
            continue
        bins = [b for b in calibration["bins"] if b["count"] > 0]
        forecast = np.array([b["mean_forecast"] for b in bins])
        observed = np.array([b["observed"] for b in bins])
        error = np.array([[b["observed"] - b["lo"], b["hi"] - b["observed"]] for b in bins]).T
        ax.errorbar(forecast, observed, yerr=np.clip(error, 0.0, None), fmt=marker, color=colour, ms=4, capsize=2, lw=0.8,
                    label=f"{label}: ECE {calibration['ece']:.3f}, AUROC {calibration['auroc']:.2f}")
        ax_count.plot(forecast, [b["count"] for b in bins], marker, color=colour, ms=4, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("forecast hazard (bin mean)")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"reliability, {results['prediction_model']} model, H = {results['horizon']:g} s", fontsize=9)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3)
    ax_count.set_yscale("log")
    ax_count.set_xlabel("forecast hazard (bin mean)")
    ax_count.set_ylabel("slack ticks in bin")
    ax_count.grid(alpha=0.3, which="both")
    ax_count.legend(fontsize=7)
    path = FIGURES / "phase5_F1_reliability.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    register("phase5_F1", f"{hedge('Prop. 10')}: reliability of the 10 Hz hazard per arm on the Phase 5 calibration missions (bins with "
                          "cluster-effective Wilson bands, simultaneous over bins); exploratory continuation, no gate authority.")
    return str(path)


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text())
    print(figure_reliability(results))


if __name__ == "__main__":
    main()
