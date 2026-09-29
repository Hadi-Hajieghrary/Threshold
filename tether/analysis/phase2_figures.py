"""Phase 2 figures (F3, F4, depth tail), regenerated from committed records only."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tether.analysis.captions import hedge, register
from tether.campaign.common import FIGURES
from tether.campaign.phase2 import HEADING_GAINS, INTENSITIES, PREDICTIONS_PATH, PRETENSIONS, RESULTS_PATH, cell_name

COLOURS = {600.0: "#9bbb59", 1000.0: "#1f4e79", 1400.0: "#c0504d", 1800.0: "#8064a2"}


def _load():
    return json.loads(RESULTS_PATH.read_text()), json.loads(PREDICTIONS_PATH.read_text())


def figure_f3(results) -> str:
    rows = results["rows"]
    fig, axes = plt.subplots(len(INTENSITIES), 2, figsize=(11.0, 4.2 * len(INTENSITIES)), squeeze=False)
    for r, intensity in enumerate(INTENSITIES):
        ax_snap, ax_taut = axes[r]
        for pretension in PRETENSIONS:
            row = rows[cell_name(pretension, 500.0, intensity)]
            snap = [s for s in row["snap"] if s["count"] > 0]
            if snap:
                x = np.array([s["threshold"] / 1000.0 for s in snap]) ** 4
                rate = np.array([s["rate"] for s in snap])
                err = np.array([[s["rate"] - s["rate_lo"], s["rate_hi"] - s["rate"]] for s in snap]).T
                ax_snap.errorbar(x, rate, yerr=err, fmt="o", color=COLOURS[pretension], ms=4, capsize=2, label=f"T0 = {pretension:.0f} N")
                pred = np.array([s["predicted_theorem6"] for s in snap])
                ax_snap.plot(x, np.where(pred > 0, pred, np.nan), "-", color=COLOURS[pretension], lw=1.0)
            taut = [t for t in row["taut"] if t["count"] > 0]
            if taut:
                x = np.array([t["threshold"] / 1000.0 for t in taut]) ** 2
                ax_taut.plot(x, [t["rate"] for t in taut], "o", color=COLOURS[pretension], ms=4, label=f"T0 = {pretension:.0f} N")
                ax_taut.plot(x, [t["predicted_lti"] if t["predicted_lti"] > 0 else np.nan for t in taut], "-", color=COLOURS[pretension], lw=1.0)
        for ax in (ax_snap, ax_taut):
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=7)
        ax_snap.set_xlabel("T_b^4 [kN^4]")
        ax_snap.set_ylabel("snap rate [1/s]")
        ax_snap.set_title(f"snap, intensity {intensity}, k_h 500 (lines: Thm 6)", fontsize=9)
        ax_taut.set_xlabel("T_b^2 [kN^2]")
        ax_taut.set_ylabel("taut-overload rate [1/s]")
        ax_taut.set_title(f"taut, intensity {intensity} (lines: Rice + LTI)", fontsize=9)
    path = FIGURES / "phase2_F3_rate_vs_threshold.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    register("phase2_F3", f"{hedge('Thm 6')}: measured virtual severance rates (seed-bootstrap 95% intervals) at each cell's powered grid against the computed, not fitted, Theorem 6 lines.")
    return str(path)


def figure_f4(results, predictions) -> str:
    groups = results["tests"]["P2-T8"]["groups"]
    fig, axes = plt.subplots(1, len(INTENSITIES), figsize=(5.2 * len(INTENSITIES), 4.0), squeeze=False)
    for c, intensity in enumerate(INTENSITIES):
        ax = axes[0, c]
        for heading_gain, style in zip(HEADING_GAINS, (":", "-", "--")):
            group = next(g for g in groups if g["intensity"] == intensity and g["heading_gain"] == heading_gain)
            pretensions = [t[0] for t in group["totals"]]
            rates = [t[1] if t[1] > 0 else np.nan for t in group["totals"]]
            ax.plot(pretensions, rates, "o" + style, color="#1f4e79", label=f"measured, k_h = {heading_gain:.0f}")
            predicted = predictions["optimum"][f"k{int(heading_gain)}_I{intensity}"]
            ax.plot(predicted["grid"], np.where(np.array(predicted["total"]) > 0, predicted["total"], np.nan), style, color="#c0504d", lw=1.0,
                    label=f"predicted, k_h = {heading_gain:.0f}")
        ax.set_yscale("log")
        ax.set_xlabel("pretension T0 [N]")
        ax.set_ylabel(f"total rate at {results['tests']['P2-T8']['common_threshold'] / 1000:.0f} kN [1/s]")
        ax.set_title(f"F4 pretension optimum, intensity {intensity}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    path = FIGURES / "phase2_F4_pretension_optimum.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    register("phase2_F4", f"{hedge('Cor. 5')}: total virtual severance rate at the common threshold against pretension, measured and predicted before the sweep.")
    return str(path)


def main() -> list[str]:
    results, predictions = _load()
    FIGURES.mkdir(parents=True, exist_ok=True)
    return [figure_f3(results), figure_f4(results, predictions)]


if __name__ == "__main__":
    for item in main():
        print(item)
