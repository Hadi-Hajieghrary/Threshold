"""Simulation snapshots of the towed fleet.

Data: records/v2/figures/cascade_run.npz, written by tether/analysis/v2/paperfig/cascade_run.py.
It is one mission of the event-rich Phase 1 cell i035_T0600_ks3 (parallel formation, T0 = 600 N,
weather intensity 0.35, k_h = 477 N m/rad, k_sigma = 3.0, sway_limit = 0.349 rad, linear drag),
seed 7105.  Its FleetRunSpec reproduces the declared config_hash 8b0259429d7e... of that cell
(records/v2/phase1/stochastic_declarations.json), seed 7105 is one of the cell's 20 scored
statistics seeds, and the campaign's own record of the mission (records/v2/phase1/
stochastic_runs.json) holds the same cable-2 mark, T_peak = 13502.4 N at t_up = 91.486 s.

SELECTION (disclosed in the caption).  cascade_run.main() scanned seeds 7101-7106 and kept the one
whose largest cascade peak was highest, so the event is chosen for clarity, not typicality: 13.5 kN
is the second-largest of the cell's 510 scored marks (median about 0.94 kN).  The formation frames
are chosen by rule, not by eye: over the all-taut 0.25 s samples of the scored window, the instants
at the MEDIAN and at the 95TH PERCENTILE of the instantaneous spread of the five chord angles.  That
spread is a selection rule only; the paper's shape metric (35.9 deg for this cell) is a different
statistic, each cable's chord-angle standard deviation over time.

Arrays used: state (10 ms body poses: 18 positions then 18 velocities; load then vessels 0..4, each
x, y, theta), state_time, elongation / rate / alive / time (1 ms per-cable logs), geometry_load,
geometry_vessel.  Hull size, stern offset and the tension law are imported from the plant.
Times are simulation time, which includes the 20 s warm-up (the campaign's t_up convention).

fig_snapshot_formation : the plant at a typical and at a disturbed all-taut instant.
fig_snapshot_cascade   : three frames through the 13.5 kN cascade at t0 = 91.486 s.
fig_snapshot_formation_col : the formation figure drawn at single-column width (Main_clear.tex).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from tether.physics import fleet as F
from tether.analysis.v2.paperfig.cascade_run import tension as plant_tension
from tether.analysis.v2.paperfig.style import (
    use_paper_style, save, WIDE, COL, C_NEUTRAL, C_MUTED, C_PERLINE, C_FLEET,
    unpack_state, body_polygon, attachment_points, REPO,
)

CACHE = REPO / "records" / "v2" / "figures" / "cascade_run.npz"
T0 = 600.0
T_UP = 91.486          # campaign record, records/v2/phase1/stochastic_runs.json
WIDTH_FULL_KN = 12.0   # line width saturates at this tension
L, B = F.VESSEL_LENGTH, F.VESSEL_BEAM


class _Geom:
    def __init__(self, load, vessel):
        self.load_offsets, self.vessel_offsets = load, vessel


def load_run():
    d = np.load(CACHE)
    return d, _Geom(d["geometry_load"], d["geometry_vessel"])


def hull() -> np.ndarray:
    """Plant-sized hull (3.0 x 1.0 m), stern face at x = -L/2 = the cable's stern offset."""
    return np.array([[0.5 * L, 0.0], [0.15 * L, 0.5 * B], [-0.5 * L, 0.5 * B],
                     [-0.5 * L, -0.5 * B], [0.15 * L, -0.5 * B]])


def tensions(d):
    return plant_tension(d["elongation"], d["rate"], d["alive"])


def nearest(arr, t):
    return int(np.argmin(np.abs(arr - t)))


def frame_bodies(d, g, t):
    si = nearest(d["state_time"], t)
    lp, vp = unpack_state(d["state"][si])
    a, b = attachment_points(lp, vp, g)
    return lp, vp, a, b


def extent(d, g, times, margin=3.2):
    """A common SCALE with per-frame CENTRING: each frame's bounding box of every body, then one
    half-span that holds the largest of them.  The fleet travels ~0.5 m/s, so frames far apart in
    time sit far apart in x; a single shared box would shrink each fleet to a speck."""
    boxes = []
    for t in times:
        lp, vp, _, _ = frame_bodies(d, g, t)
        P = np.vstack([body_polygon(lp, F.pentagon_vertices())] +
                      [body_polygon(v, hull()) for v in vp])
        boxes.append((P.min(0), P.max(0)))
    half = np.max([0.5 * (hi - lo) for lo, hi in boxes], axis=0) + margin
    return {t: (0.5 * (lo + hi), half) for t, (lo, hi) in zip(times, boxes)}


def draw(ax, d, g, T, t, box, scale_bar=False):
    lp, vp, a, b = frame_bodies(d, g, t)
    Tn = T[nearest(d["time"], t)]
    c, half = box[t]               # this frame's centre, the figure's common half-span
    lo, hi = c - half, c + half
    inside = lambda P: bool(np.all(P >= lo - 1e-9) and np.all(P <= hi + 1e-9))
    assert inside(body_polygon(lp, F.pentagon_vertices())), "payload outside frame"
    assert all(inside(body_polygon(v, hull())) for v in vp), "a vessel falls outside the frame"

    ax.add_patch(Polygon(body_polygon(lp, F.pentagon_vertices()), closed=True,
                         facecolor="#dfe6ec", edgecolor=C_NEUTRAL, lw=0.7, zorder=2))
    for i in range(5):
        seg = np.stack([a[i], b[i]]).T
        if Tn[i] <= 0.0:
            ax.plot(*seg, ls=(0, (2.0, 1.4)), lw=1.0, color=C_PERLINE, zorder=3)
        else:
            ax.plot(*seg, ls="-", color=C_FLEET, zorder=3,
                    lw=0.7 + 2.3 * min(Tn[i] / (WIDTH_FULL_KN * 1e3), 1.0))
        ax.add_patch(Polygon(body_polygon(vp[i], hull()), closed=True, facecolor="#f4f4f4",
                             edgecolor=C_NEUTRAL, lw=0.6, zorder=4))
        bow = body_polygon(vp[i], np.array([[0.5 * L + 0.9, 0.0]]))[0]
        ax.text(*bow, str(i), fontsize=7, color=C_NEUTRAL, ha="center", va="center", zorder=6)

    ax.set_xlim(c[0] - half[0], c[0] + half[0])
    ax.set_ylim(c[1] - half[1], c[1] + half[1])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.5); s.set_color("#b8b8b8")
    if scale_bar:
        x0, y0 = c[0] - half[0] + 1.2, c[1] - half[1] + 1.0
        ax.plot([x0, x0 + 5.0], [y0, y0], color=C_NEUTRAL, lw=1.3, solid_capstyle="butt")
        ax.text(x0 + 2.5, y0 + 0.45, "5 m", ha="center", va="bottom", fontsize=7, color=C_NEUTRAL)
    return Tn


def key(fig, fontsize=7, handlelength=2.4, columnspacing=1.6):
    handles = [Line2D([], [], color=C_FLEET, lw=2.2, label="taut (width $\\propto$ tension)"),
               Line2D([], [], color=C_PERLINE, lw=1.0, ls=(0, (2.0, 1.4)),
                      label="carrying no tension"),
               Line2D([], [], color="none", marker="$\\rightarrow$", ms=9, mec=C_MUTED,
                      label="tow direction $+x$")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02),
               fontsize=fontsize, handlelength=handlelength, columnspacing=columnspacing)


def chord_spread(d, g, t):
    _, _, a, b = frame_bodies(d, g, t)
    v = b - a
    return float(np.degrees(np.arctan2(v[:, 1], v[:, 0])).std())


def formation_instants(d, g, T):
    ts = d["state_time"]
    cand = []
    for si in range(int(np.searchsorted(ts, 20.0)), len(ts), 25):     # scored window, 0.25 s
        t = float(ts[si])
        if T[nearest(d["time"], t)].min() > 0.0:
            cand.append((chord_spread(d, g, t), t))
    s = np.array([c[0] for c in cand]); tt = np.array([c[1] for c in cand])
    pick = lambda q: float(tt[np.argmin(np.abs(s - np.percentile(s, q)))])
    return pick(50), pick(95), s


def figure_formation(column=False):
    """Full-width figure (Main.tex), or with column=True the single-column layout used by the
    submission (Main_clear.tex): same frames, same selection rule, drawn at COL width so its
    7 pt labels stay 7 pt in print."""
    d, g = load_run(); T = tensions(d)
    t_med, t_p95, s = formation_instants(d, g, T)
    box = extent(d, g, [t_med, t_p95])
    fig, axes = plt.subplots(1, 2, figsize=(COL, 2.05) if column else (WIDE, 2.55))
    draw(axes[0], d, g, T, t_med, box, scale_bar=True)
    draw(axes[1], d, g, T, t_p95, box)
    axes[0].set_title(f"(a) typical: $t = {t_med:.1f}$ s", fontsize=7.5, pad=3)
    axes[1].set_title(f"(b) disturbed: $t = {t_p95:.1f}$ s", fontsize=7.5, pad=3)
    if column:
        key(fig, fontsize=6.3, handlelength=1.6, columnspacing=0.8)
        fig.subplots_adjust(wspace=0.04, bottom=0.15, left=0.01, right=0.99)
        save(fig, "fig_snapshot_formation_col")
        return
    key(fig)
    fig.subplots_adjust(wspace=0.03, bottom=0.12)
    save(fig, "fig_snapshot_formation")
    print(f"  {len(s)} all-taut samples; spread median {np.median(s):.1f} deg, "
          f"p95 {np.percentile(s, 95):.1f} deg; frames at t = {t_med:.2f} s "
          f"({chord_spread(d, g, t_med):.1f} deg) and {t_p95:.2f} s "
          f"({chord_spread(d, g, t_p95):.1f} deg)")


def figure_cascade():
    d, g = load_run(); T = tensions(d)
    parent = 2
    w = (d["time"] >= T_UP) & (d["time"] <= T_UP + 0.4)
    t_pk = float(d["time"][w][T[w, parent].argmax()]); pk = float(T[w, parent].max())
    frames = [T_UP - 0.40, t_pk, T_UP + 0.25]
    titles = ["(a) $t_0 - 0.40$ s", f"(b) $t_0 + {1e3*(t_pk - T_UP):.0f}$ ms: {pk/1e3:.1f} kN",
              "(c) $t_0 + 0.25$ s"]
    box = extent(d, g, frames)
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.15))
    for i, (ax, t, ti) in enumerate(zip(axes, frames, titles)):
        Tn = draw(ax, d, g, T, t, box, scale_bar=(i == 0))
        ax.set_title(ti, fontsize=7.5, pad=3)
        loaded = [j for j in range(5) if Tn[j] > 0.0]
        print(f"  frame {ti[:3]} t = {t:.3f} s: tensions kN = {np.round(Tn/1e3, 2)}, loaded {loaded}")
    key(fig)
    fig.subplots_adjust(wspace=0.03, bottom=0.13)
    save(fig, "fig_snapshot_cascade")
    print(f"  peak {pk:.1f} N = {pk/T0:.2f} T0 at t0 + {1e3*(t_pk - T_UP):.1f} ms")


def main():
    use_paper_style()
    figure_formation()
    figure_formation(column=True)
    figure_cascade()


if __name__ == "__main__":
    main()
