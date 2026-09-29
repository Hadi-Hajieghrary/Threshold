"""Shared renderer for the presentation clips in Presentation/.

Every clip is drawn from plant state: a campaign record, or a replay whose reproduction of the
campaign record is asserted before a single frame is drawn.  Nothing here invents data.  Every
number a clip puts on screen is registered with a ``Manifest``, which records where the number
came from (a record path and key, or the replay it was computed from) so that it can be audited.

Conventions (plant: tether/physics/fleet.py, constants.py):
  * state rows are 36 wide: 18 positions then 18 velocities; load first, then vessels 0..4, each
    (x, y, theta);
  * cable tension is the plant's law, (k e + c edot) while e > 0 and the cable is alive, clamped at
    0 (``tension``); hulls are the plant's 3.0 x 1.0 m, the stern (cable) point at x = -1.5 m;
  * the 10 ms state and the 1 ms cable logs share the simulation clock, which includes warm-up.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from tether.physics import constants as K
from tether.physics import fleet as F

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "Presentation"
CLIPS = OUT / "clips"
MANIFESTS = OUT / "manifests"
CACHE = OUT / "cache"          # replays (npz) so a clip can be re-rendered without re-simulating

# ---------------------------------------------------------------- video format
W, H, DPI, FPS = 1920, 1080, 120, 30
FIGSIZE = (W / DPI, H / DPI)

# ---------------------------------------------------------------- palette (the paper's)
BG = "#ffffff"
INK = "#1f2328"
MUTED = "#6b7280"
FAINT = "#d1d5db"
TAUT = "#2f6f9f"        # fleet / taut cable (paper C_FLEET)
SLACK = "#c1443c"       # per-line / slack cable (paper C_PERLINE)
SEVERED = "#9ca3af"
ACCENT = "#d98c00"
OK = "#2e7d5b"
PAYLOAD_FACE = "#dfe6ec"
HULL_FACE = "#f4f4f4"
CABLE_COLORS = ["#2f6f9f", "#4f9ac4", "#7ab5d6", "#c1443c", "#e08a5a"]

# point sizes at DPI 120 (1 pt = 1.667 px): legible on a 1080p screen
FS_TITLE, FS_SUB, FS_BODY, FS_CAPTION, FS_SMALL, FS_TINY = 26, 18, 17, 17, 13, 11

L_HULL, B_HULL = F.VESSEL_LENGTH, F.VESSEL_BEAM
WIDTH_FULL_N = 12_000.0     # cable line width saturates at this tension


def video_style() -> None:
    plt.rcParams.update({
        "figure.dpi": DPI, "savefig.dpi": DPI,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": FS_BODY, "axes.titlesize": FS_SUB, "axes.labelsize": FS_SMALL,
        "xtick.labelsize": FS_TINY, "ytick.labelsize": FS_TINY, "legend.fontsize": FS_SMALL,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 2.0, "grid.alpha": 0.25,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    })


# ---------------------------------------------------------------- plant geometry and tension

def tension(elong, rate, alive):
    """The plant's cable law: k e + c edot while e > 0 and alive, clamped at 0 (N)."""
    t = K.CABLE_STIFFNESS * np.asarray(elong) + K.CABLE_DAMPING * np.asarray(rate)
    return np.where(np.asarray(alive, bool) & (np.asarray(elong) > 0.0) & (t > 0.0), t, 0.0)


def hull() -> np.ndarray:
    """Plant-sized hull (3.0 x 1.0 m), bow toward +x, stern face at x = -L/2 (the cable point)."""
    L, B = L_HULL, B_HULL
    return np.array([[0.5 * L, 0.0], [0.15 * L, 0.5 * B], [-0.5 * L, 0.5 * B],
                     [-0.5 * L, -0.5 * B], [0.15 * L, -0.5 * B]])


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def body_polygon(pose, local) -> np.ndarray:
    return (_rot(float(pose[2])) @ np.asarray(local, float).T).T + np.asarray(pose[:2], float)


def unpack_state(row, n_vessels: int = 5):
    q = np.asarray(row)[: 3 * (n_vessels + 1)]
    return q[:3].copy(), q[3:].reshape(n_vessels, 3).copy()


def attachment_points(load_pose, vessel_poses, geometry):
    load_pts = body_polygon(load_pose, geometry.load_offsets)
    ship_pts = np.stack([body_polygon(vessel_poses[i], geometry.vessel_offsets[i][None, :])[0]
                         for i in range(len(vessel_poses))])
    return load_pts, ship_pts


def parallel_geometry():
    return F.formation_geometry("parallel")


def mission_geometry():
    """The v1 mission fleet (MissionSpec defaults: fan formation, arc half-angle 0.55 rad)."""
    from tether.campaign.mission import MissionSpec, mission_geometry as mg
    return mg(MissionSpec())


def nearest(times, t) -> int:
    return int(np.argmin(np.abs(np.asarray(times) - t)))


# ---------------------------------------------------------------- drawing

def fleet_bounds(state_rows, geometry, margin: float = 3.0):
    """Per-row bounding boxes of every body -> (centres (n,2), common half-span (2,))."""
    centres, halves = [], []
    for row in np.atleast_2d(state_rows):
        lp, vp = unpack_state(row)
        P = np.vstack([body_polygon(lp, F.pentagon_vertices())] + [body_polygon(v, hull()) for v in vp])
        lo, hi = P.min(0), P.max(0)
        centres.append(0.5 * (lo + hi)); halves.append(0.5 * (hi - lo))
    return np.array(centres), np.max(np.array(halves), axis=0) + margin


def smooth_camera(centres, window: int = 31):
    """Moving-average camera so the view follows the fleet without jitter."""
    c = np.asarray(centres, float)
    if len(c) < 3:
        return c
    k = max(1, min(window, len(c)) // 2)
    pad = np.pad(c, ((k, k), (0, 0)), mode="edge")
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    return np.stack([np.convolve(pad[:, j], kernel, mode="valid") for j in range(2)], axis=1)


def fit_aspect(half, box_aspect: float):
    """Grow the half-span so its x:y ratio matches the axes box (equal-aspect drawing)."""
    hx, hy = float(half[0]), float(half[1])
    if hx / hy < box_aspect:
        hx = hy * box_aspect
    else:
        hy = hx / box_aspect
    return np.array([hx, hy])


def draw_fleet(ax, state_row, geometry, T_row, alive_row=None, *, centre, half, labels=True,
               scale_bar=True, check_inside=True, show_tension_kN=False):
    """Draw payload, hulls and cables at one instant.

    Taut cables are solid, width proportional to tension (saturating at 12 kN); a cable carrying
    no tension is dashed red; a severed cable is drawn as two grey stubs with a cross.  Returns
    the tension row that was drawn.
    """
    ax.cla()
    lp, vp = unpack_state(state_row)
    a, b = attachment_points(lp, vp, geometry)
    n = len(vp)
    alive = np.ones(n, bool) if alive_row is None else np.asarray(alive_row, bool)
    lo, hi = np.asarray(centre) - half, np.asarray(centre) + half
    if check_inside:
        polys = [body_polygon(lp, F.pentagon_vertices())] + [body_polygon(v, hull()) for v in vp]
        for P in polys:
            assert np.all(P >= lo - 1e-9) and np.all(P <= hi + 1e-9), "a body falls outside the frame"
    ax.add_patch(Polygon(body_polygon(lp, F.pentagon_vertices()), closed=True, facecolor=PAYLOAD_FACE,
                         edgecolor=INK, lw=1.2, zorder=2))
    for i in range(n):
        seg = np.stack([a[i], b[i]])
        if not alive[i]:
            d = seg[1] - seg[0]
            ax.plot(*np.stack([seg[0], seg[0] + 0.18 * d]).T, color=SEVERED, lw=2.0, zorder=3)
            ax.plot(*np.stack([seg[1], seg[1] - 0.18 * d]).T, color=SEVERED, lw=2.0, zorder=3)
            m = seg[0] + 0.5 * d
            ax.plot([m[0]], [m[1]], marker="x", ms=14, mew=3, color=SLACK, zorder=5)
        elif T_row[i] <= 0.0:
            ax.plot(*seg.T, ls=(0, (3.0, 2.2)), lw=2.0, color=SLACK, zorder=3)
        else:
            ax.plot(*seg.T, ls="-", color=TAUT, lw=1.4 + 5.0 * min(T_row[i] / WIDTH_FULL_N, 1.0), zorder=3)
            if show_tension_kN:
                m = 0.5 * (seg[0] + seg[1])
                ax.text(m[0], m[1] + 0.5, f"{T_row[i] / 1e3:.1f} kN", fontsize=FS_TINY, color=TAUT,
                        ha="center", va="bottom", zorder=6)
        ax.add_patch(Polygon(body_polygon(vp[i], hull()), closed=True, facecolor=HULL_FACE,
                             edgecolor=INK, lw=1.0, zorder=4))
        if labels:
            bow = body_polygon(vp[i], np.array([[0.5 * L_HULL + 1.0, 0.0]]))[0]
            ax.text(*bow, str(i), fontsize=FS_SMALL, color=INK, ha="center", va="center", zorder=6)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.8); s.set_color(FAINT)
    if scale_bar:
        x0, y0 = lo[0] + 0.04 * (hi[0] - lo[0]), lo[1] + 0.05 * (hi[1] - lo[1])
        ax.plot([x0, x0 + 5.0], [y0, y0], color=INK, lw=2.5, solid_capstyle="butt")
        ax.text(x0 + 2.5, y0 + 0.02 * (hi[1] - lo[1]), "5 m", ha="center", va="bottom",
                fontsize=FS_TINY, color=INK)
    return T_row


def cable_key(fig, y: float = 0.115, x: float = 0.5, severed: bool = False):
    handles = [Line2D([], [], color=TAUT, lw=4, label="taut (width ∝ tension)"),
               Line2D([], [], color=SLACK, lw=2, ls=(0, (3.0, 2.2)), label="carrying no tension")]
    if severed:
        handles.append(Line2D([], [], color=SEVERED, lw=2, marker="x", mec=SLACK, ms=10, mew=2.5,
                              label="severed"))
    fig.legend(handles=handles, loc="center", ncol=len(handles), bbox_to_anchor=(x, y),
               fontsize=FS_SMALL, handlelength=2.6, columnspacing=2.0)


# ---------------------------------------------------------------- frame furniture

def new_frame():
    video_style()
    return plt.figure(figsize=FIGSIZE, dpi=DPI)


def title(fig, text: str, sub: str | None = None):
    fig.text(0.03, 0.955, text, fontsize=FS_TITLE, weight="bold", color=INK, ha="left", va="top")
    if sub:
        fig.text(0.03, 0.905, sub, fontsize=FS_SUB, color=MUTED, ha="left", va="top")


# Narration log: every caption, with the frame at which it was created (the writer counts frames).
CAPTION_LOG: list = []
_FRAMES = [0]


def caption(fig, text: str, y: float = 0.068):
    """Explanatory caption band at the bottom (the video's narration); logged for the script."""
    if text and (not CAPTION_LOG or CAPTION_LOG[-1][1] != text):
        CAPTION_LOG.append((_FRAMES[0], text))
    return fig.text(0.5, y, text, fontsize=FS_CAPTION, color=INK, ha="center", va="center",
                    wrap=True, bbox=dict(boxstyle="round,pad=0.5", fc="#f3f4f6", ec="none"))


def footer(fig, text: str):
    """Provenance line: what was simulated / which record."""
    fig.text(0.985, 0.012, text, fontsize=FS_TINY, color=MUTED, ha="right", va="bottom")


def clock(fig, t: float, speed: str | None = None, x: float = 0.97, y: float = 0.955):
    s = f"t = {t:7.3f} s"
    if speed:
        s += f"   ({speed})"
    return fig.text(x, y, s, fontsize=FS_SUB, family="DejaVu Sans Mono", color=INK, ha="right", va="top")


class _CountingWriter(FFMpegWriter):
    """FFMpegWriter that counts frames, so captions can be time-stamped for the script."""

    def setup(self, *args, **kwargs):
        _FRAMES[0] = 0
        CAPTION_LOG.clear()
        return super().setup(*args, **kwargs)

    def grab_frame(self, **kwargs):
        _FRAMES[0] += 1
        return super().grab_frame(**kwargs)


def writer(crf: int = 20) -> FFMpegWriter:
    return _CountingWriter(fps=FPS, codec="libx264",
                        extra_args=["-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "medium",
                                    "-movflags", "+faststart"])


def hold(w: FFMpegWriter, seconds: float) -> int:
    """Repeat the current frame for ``seconds`` (for reading time); returns frames written."""
    n = int(round(seconds * FPS))
    for _ in range(n):
        w.grab_frame()
    return n


# ---------------------------------------------------------------- provenance

def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Manifest:
    """What a clip shows and where every on-screen number comes from."""
    name: str
    title: str
    story: str                                   # one paragraph: what the viewer should learn
    sources: dict = field(default_factory=dict)  # path -> sha256
    values: list = field(default_factory=list)   # on-screen numbers with provenance
    checks: list = field(default_factory=list)   # reproduction / consistency checks that passed
    selection: str = ""                          # how seeds / events / instants were chosen
    caveats: list = field(default_factory=list)  # what the clip does not show or claim
    frames: int = 0

    def source(self, path) -> None:
        p = Path(path)
        self.sources[str(p.relative_to(REPO) if p.is_absolute() else p)] = sha256(REPO / p if not p.is_absolute() else p)

    def value(self, label: str, value, unit: str = "", source: str = "", note: str = "") -> str:
        """Register an on-screen number; returns ``label`` for convenience."""
        self.values.append({"label": label, "value": value, "unit": unit, "source": source, "note": note})
        return label

    def check(self, what: str, passed: bool, detail: str = "") -> None:
        assert passed, f"{self.name}: check failed: {what} ({detail})"
        self.checks.append({"check": what, "passed": True, "detail": detail})

    def write(self, clip_path) -> Path:
        MANIFESTS.mkdir(parents=True, exist_ok=True)
        clip_path = Path(clip_path)
        info = {"name": self.name, "title": self.title, "story": self.story,
                "clip": str(clip_path.relative_to(REPO)), "clip_sha256": sha256(clip_path),
                "fps": FPS, "frames": self.frames, "duration_s": round(self.frames / FPS, 3),
                "resolution": [W, H], "selection": self.selection, "sources": self.sources,
                "values": self.values, "checks": self.checks, "caveats": self.caveats,
                "captions": [{"t": round(f / FPS, 2), "text": t} for f, t in CAPTION_LOG]}
        out = MANIFESTS / f"{self.name}.json"
        out.write_text(json.dumps(info, indent=1, default=float))
        print(f"wrote {out.relative_to(REPO)}  ({info['duration_s']} s, {len(self.values)} values, "
              f"{len(self.checks)} checks)")
        return out


def ensure_dirs() -> None:
    for d in (CLIPS, MANIFESTS, CACHE):
        d.mkdir(parents=True, exist_ok=True)
