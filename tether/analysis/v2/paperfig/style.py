"""Shared plotting style and geometry helpers for the paper's figures.

Every figure in Paper/Figures/ is produced by a module in this package and is drawn
from a record under records/ or records/v2/.  Nothing here invents data: a figure that
cannot be traced to a record is a defect, not a decoration.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]
FIGDIR = REPO / "Paper" / "Figures"

# IEEE two-column geometry.
COL = 3.5      # single-column width, inches
WIDE = 7.16    # full-width (two-column) figure

# One palette for the whole paper.  Per-line/single-vessel objects are warm, fleet
# objects are cool, so the paper's central contrast reads at a glance.
C_PERLINE = "#c1443c"
C_FLEET   = "#2f6f9f"
C_NEUTRAL = "#3b3b3b"
C_MUTED   = "#8a8a8a"
C_ACCENT  = "#d98c00"
C_OK      = "#2e7d5b"
C_BAND    = "#cdd9e5"
CABLE_COLORS = ["#2f6f9f", "#4f9ac4", "#7ab5d6", "#c1443c", "#e08a5a"]


def use_paper_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.1,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.30,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        # IEEE PDF eXpress rejects Type 3 fonts; embed TrueType (Type 42) instead.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, name: str) -> Path:
    """Write <name>.pdf (for LaTeX) and <name>.png (for quick viewing)."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / f"{name}.pdf"
    fig.savefig(out)
    fig.savefig(FIGDIR / f"{name}.png")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")
    return out


# ---------------------------------------------------------------- geometry

def unpack_state(row: np.ndarray, n_vessels: int = 5):
    """(36,) planar state -> load pose (3,) and vessel poses (N, 3)."""
    q = np.asarray(row)[: 3 * (n_vessels + 1)]
    return q[:3].copy(), q[3:].reshape(n_vessels, 3).copy()


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def body_polygon(pose: np.ndarray, local: np.ndarray) -> np.ndarray:
    """Transform body-frame vertices (K,2) into the world frame."""
    return (_rot(pose[2]) @ np.asarray(local).T).T + pose[:2]


def attachment_points(load_pose, vessel_poses, geometry):
    """World-frame (load-side, vessel-side) cable endpoints for every cable."""
    load_pts = body_polygon(load_pose, geometry.load_offsets)
    ship_pts = np.stack([
        body_polygon(vessel_poses[i], geometry.vessel_offsets[i][None, :])[0]
        for i in range(len(vessel_poses))
    ])
    return load_pts, ship_pts


def vessel_hull(length: float = 5.0, beam: float = 2.0) -> np.ndarray:
    """A simple planar hull outline in the vessel frame, bow toward +x."""
    return np.array([
        [ 0.60 * length, 0.0],
        [ 0.25 * length,  0.50 * beam],
        [-0.40 * length,  0.50 * beam],
        [-0.40 * length, -0.50 * beam],
        [ 0.25 * length, -0.50 * beam],
    ])
