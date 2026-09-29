"""Generate deterministic Phase 1 figures exclusively from recorded arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORD = ROOT / "records" / "phase1" / "phase1_deterministic.npz"
DEFAULT_F1 = ROOT / "reports" / "figures" / "phase1_F1_fleet_impedance.png"
DEFAULT_F2 = ROOT / "reports" / "figures" / "phase1_F2_excursion_depth.png"
DEFAULT_PHASE1R_RECORD = ROOT / "records" / "phase1" / "phase1r_records.npz"
DEFAULT_PHASE1R_F1 = (
    ROOT / "reports" / "figures" / "phase1r_F1_acceleration_threshold.png"
)
DEFAULT_PHASE1R_F2 = ROOT / "reports" / "figures" / "phase1r_F2_depth_comparison.png"


def generate_figures(
    record_path: Path = DEFAULT_RECORD,
    f1_path: Path = DEFAULT_F1,
    f2_path: Path = DEFAULT_F2,
) -> tuple[Path, Path]:
    """Render F1 and F2 from the deterministic NPZ without running physics."""
    if not record_path.exists():
        raise FileNotFoundError(f"deterministic Phase 1 record is missing: {record_path}")
    with np.load(record_path, allow_pickle=False) as record:
        speeds = record["fleet_impact_speeds_mps"]
        peaks = record["fleet_first_peaks_n"]
        pretensions = record["excursion_pretension_n"]
        gust_ratios = record["excursion_gust_ratio"]
        durations = record["excursion_gust_duration_s"]
        depths = record["excursion_maximum_depth_m"]
        complete = record["excursion_complete"]

    f1_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.8, 4.5))
    figure.patch.set_facecolor("#f4f1e8")
    axis.set_facecolor("#fbfaf6")
    axis.grid(color="#d5d0c5", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    fit = float(np.dot(speeds, peaks) / np.dot(speeds, speeds))
    line_speed = np.linspace(0.0, float(np.max(speeds)), 100)
    axis.scatter(speeds, peaks / 1000.0, color="#b13f2d", zorder=3, label="Full plant")
    axis.plot(
        line_speed,
        fit * line_speed / 1000.0,
        color="#176b87",
        linewidth=1.8,
        label=f"Fit: {fit / 1000.0:.3f} kN s/m",
    )
    axis.plot(
        line_speed,
        9_000.0 * line_speed / 1000.0,
        color="#343434",
        linestyle="--",
        linewidth=1.1,
        label="9.000 kN s/m reference",
    )
    axis.set_xlabel("Target relative speed [m/s]")
    axis.set_ylabel("First peak tension [kN]")
    axis.set_title("F1. Full-formation engagement impedance")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        f1_path,
        dpi=160,
        metadata={"Software": "tether.analysis.phase1_figures"},
    )
    plt.close(figure)

    f2_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    figure.patch.set_facecolor("#f4f1e8")
    axis.set_facecolor("#fbfaf6")
    axis.grid(color="#d5d0c5", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    colors = ("#176b87", "#3f7d3c", "#c17d11", "#b13f2d")
    nominal = np.isclose(pretensions, 990.0)
    for color, duration in zip(colors, np.unique(durations[nominal])):
        selected = nominal & np.isclose(durations, duration)
        axis.plot(
            gust_ratios[selected],
            depths[selected],
            color=color,
            marker="o",
            linewidth=1.5,
            label=f"{duration:g} s",
        )
        censored = selected & ~complete
        axis.scatter(
            gust_ratios[censored],
            depths[censored],
            color=color,
            marker="x",
            s=60,
            linewidth=1.8,
            zorder=4,
        )
    axis.axvline(1.0, color="#343434", linestyle="--", linewidth=1.1)
    axis.set_xlabel("Gust ratio lambda")
    axis.set_ylabel("Maximum depth [m]")
    axis.set_title("F2. Scripted excursion depth at 990 N pretension")
    axis.legend(title="Gust duration", frameon=False)
    figure.tight_layout()
    figure.savefig(
        f2_path,
        dpi=160,
        metadata={"Software": "tether.analysis.phase1_figures"},
    )
    plt.close(figure)
    return f1_path, f2_path


def generate_phase1r_figures(
    record_path: Path = DEFAULT_PHASE1R_RECORD,
    f1_path: Path = DEFAULT_PHASE1R_F1,
    f2_path: Path = DEFAULT_PHASE1R_F2,
) -> tuple[Path, Path]:
    """Render corrected Phase 1R threshold and depth figures from records."""
    if not record_path.exists():
        raise FileNotFoundError(f"Phase 1R record is missing: {record_path}")
    with np.load(record_path, allow_pickle=False) as record:
        probe_step = float(record["t6_primary_time_step_s"][0])
        probe_window = float(record["t6_primary_fit_window_s"][0])
        primary_probe = np.isclose(record["t6_probe_time_step_s"], probe_step) & np.isclose(
            record["t6_probe_fit_window_s"], probe_window
        )
        pretension = record["t6_probe_pretension_n"][primary_probe]
        gust_ratio = record["t6_probe_gust_ratio"][primary_probe]
        acceleration = record["t6_probe_acceleration_mps2"][primary_probe]
        measured_depth = record["cell_maximum_depth_m"]
        predicted_depth = record["cell_amended_maximum_depth_prediction_m"]
        depth_gust_ratio = record["cell_gust_ratio"]
        effective_mass = float(record["pinned_fleet_effective_mass_kg"][0])

    normalized_acceleration = acceleration / (pretension / effective_mass)
    valid_acceleration = np.isfinite(normalized_acceleration)
    beta, alpha = np.polyfit(
        gust_ratio[valid_acceleration],
        normalized_acceleration[valid_acceleration],
        1,
    )
    threshold = -alpha / beta

    f1_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.8, 4.5))
    figure.patch.set_facecolor("#f4f1e8")
    axis.set_facecolor("#fbfaf6")
    axis.grid(color="#d5d0c5", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.scatter(
        gust_ratio[valid_acceleration],
        normalized_acceleration[valid_acceleration],
        color="#176b87",
        alpha=0.65,
        label="True-slack probes",
    )
    fit_x = np.linspace(float(np.min(gust_ratio)), float(np.max(gust_ratio)), 100)
    axis.plot(
        fit_x,
        alpha + beta * fit_x,
        color="#b13f2d",
        linewidth=1.8,
        label=f"Fit threshold: {threshold:.3f}",
    )
    axis.axhline(0.0, color="#343434", linewidth=1.0)
    axis.axvline(1.0, color="#343434", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Gust ratio lambda")
    axis.set_ylabel("Measured slack acceleration / a0")
    axis.set_title("Phase 1R acceleration-sign threshold")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        f1_path,
        dpi=160,
        metadata={"Software": "tether.analysis.phase1_figures"},
    )
    plt.close(figure)

    valid_depth = np.isfinite(predicted_depth) & (measured_depth < 1.0)
    limit = 1.05 * max(
        float(np.max(measured_depth[valid_depth])),
        float(np.max(predicted_depth[valid_depth])),
    )
    figure, axis = plt.subplots(figsize=(5.4, 5.0))
    figure.patch.set_facecolor("#f4f1e8")
    axis.set_facecolor("#fbfaf6")
    axis.grid(color="#d5d0c5", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.scatter(
        predicted_depth[valid_depth],
        measured_depth[valid_depth],
        c=depth_gust_ratio[valid_depth],
        cmap="viridis",
        alpha=0.75,
    )
    axis.plot([0.0, limit], [0.0, limit], color="#b13f2d", linewidth=1.2)
    axis.set_xlim(0.0, limit)
    axis.set_ylim(0.0, limit)
    axis.set_xlabel("Amended predictor [m]")
    axis.set_ylabel("Full-plant maximum depth [m]")
    axis.set_title("Phase 1R depth comparator")
    figure.tight_layout()
    figure.savefig(
        f2_path,
        dpi=160,
        metadata={"Software": "tether.analysis.phase1_figures"},
    )
    plt.close(figure)
    return f1_path, f2_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", choices=("legacy", "phase1r"), nargs="?", default="legacy")
    arguments = parser.parse_args()
    paths = (
        generate_phase1r_figures()
        if arguments.campaign == "phase1r"
        else generate_figures()
    )
    for path in paths:
        print(path.relative_to(ROOT))