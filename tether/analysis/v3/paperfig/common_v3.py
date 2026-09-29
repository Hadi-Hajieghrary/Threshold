"""Shared loaders and drawing helpers for the plan v3 figures.

Records come from ``tether.campaign.v3.campaign.RECORD_DIR`` (records/v3/), so a test can redirect
every figure at once.  Colour follows the paper's palette (tether/analysis/v2/paperfig/style.py):
the cool hue is the measurement, the warm hue the theory's prediction, grey the reference.  No
chart carries more than three hues, and identity is always doubled by marker shape or a label.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tether.analysis.v2.paperfig.style import C_ACCENT, C_BAND, C_FLEET, C_MUTED, C_NEUTRAL, C_PERLINE
from tether.campaign.v3 import campaign, cells

C_MEASURED = C_FLEET  # cool: the plant
C_PREDICTED = C_PERLINE  # warm: the theory (exact response / kernel)
C_REFERENCE = C_MUTED  # grey: the paper's closed form and bands
C_CLOSED = C_ACCENT  # the closed form 0.44 line, when it must stand apart from the band

FACTOR = campaign.FACTOR


def record_dir() -> Path:
    return campaign.RECORD_DIR


def load_json(name: str) -> dict:
    # the committed predictions always come from the declared file, whatever RECORD_DIR points at
    path = campaign.PREDICTIONS_PATH if name == "cascade_predictions.json" else record_dir() / name
    if not path.exists():
        raise SystemExit(f"{path} is missing: run the analysis that writes it first")
    return json.loads(path.read_text())


def load_npz(name: str) -> dict:
    path = record_dir() / name
    if not path.exists():
        raise SystemExit(f"{path} is missing: run the analysis that writes it first")
    return dict(np.load(path, allow_pickle=True))


def cell_order(names) -> list[str]:
    """Parallel cells by pretension then intensity, then fan cells, then the stiffness pair."""
    def key(name):
        p = cells.parse_cell(name)
        return (p["stiffness_factor"] != 1.0, p["formation"] == "fan", p["pretension"], p["heading_gain"], p["intensity"])
    return sorted(names, key=key)


def cell_label(name: str) -> str:
    p = cells.parse_cell(name)
    label = f"{p['pretension'] / 1000:.1f} kN / {p['intensity']:g}"
    if p["formation"] == "fan":
        label = "fan " + label
    if p["stiffness_factor"] != 1.0:
        label += f" / k×{p['stiffness_factor']:g}"
    if p["heading_gain"] != 500.0 and p["formation"] == "parallel":
        label += f" / k_h {p['heading_gain']:.0f}"
    return label


def is_fan(name: str) -> bool:
    return cells.parse_cell(name)["formation"] == "fan"


def matrix(entries) -> np.ndarray:
    return np.array([[np.nan if v is None else float(v) for v in row] for row in entries], dtype=float)


def band(ax, low: float, high: float, axis: str = "y", label: str | None = None) -> None:
    if axis == "y":
        ax.axhspan(low, high, color=C_BAND, alpha=0.6, lw=0, zorder=0, label=label)
    else:
        ax.axvspan(low, high, color=C_BAND, alpha=0.6, lw=0, zorder=0, label=label)


def factor_band(ax, lo: float, hi: float) -> None:
    """The declared factor-1.5 agreement band around the diagonal on a log-log axis."""
    x = np.array([lo, hi])
    ax.fill_between(x, x / FACTOR, x * FACTOR, color=C_BAND, alpha=0.6, lw=0, zorder=0)
    ax.plot(x, x, color=C_NEUTRAL, lw=0.8, zorder=1)


def tidy(ax) -> None:
    ax.grid(True, color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
