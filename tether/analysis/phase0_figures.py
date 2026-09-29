"""Generate Phase 0 figures exclusively from the committed record."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tether.physics import constants

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORD = ROOT / "records" / "phase0" / "transit_nominal.npz"
DEFAULT_FIGURE = ROOT / "reports" / "figures" / "phase0_validation.png"


def generate_figure(
    record_path: Path = DEFAULT_RECORD,
    output_path: Path = DEFAULT_FIGURE,
) -> Path:
    """Plot speed and tension from record data without running the simulator."""
    if not record_path.exists():
        raise FileNotFoundError(f"committed Phase 0 record is missing: {record_path}")
    with np.load(record_path, allow_pickle=False) as record:
        times = record["time"]
        truth = record["truth"]
        tensions = record["tension"]

    position_count = truth.shape[1] // 2
    load_speed = truth[:, position_count]
    mean_tension = np.mean(tensions, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    figure.patch.set_facecolor("#f5f2ea")
    for axis in axes:
        axis.set_facecolor("#fbfaf6")
        axis.grid(color="#d6d1c7", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].plot(times, load_speed, color="#176b87", linewidth=1.8)
    axes[0].set_ylabel("Load speed [m/s]")
    axes[1].plot(times, mean_tension, color="#b1442e", linewidth=1.8)
    axes[1].axhline(990.0, color="#3c3c3c", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Mean tension [N]")
    axes[1].set_xlabel("Simulation time [s]")
    figure.suptitle("Phase 0 deterministic weather transit")
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
        metadata={"Software": "tether.analysis.phase0_figures"},
    )
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    print(generate_figure().relative_to(ROOT))