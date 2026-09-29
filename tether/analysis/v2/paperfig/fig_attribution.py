"""Event-level cross-cable attribution against a time-shift chance baseline.

The measured series on this figure is recomputed from the campaign's mark record.
The chance series is a POST-HOC baseline (not part of the pre-registered campaign)
and is labelled as such on the figure.  No prediction, declaration or closed form
is drawn; in particular the 0.44 transmission ratio of Contribution 1 is a
derivation and does NOT appear here.

Record read
-----------
records/phase2/phase2_records.npz  (the only file this module opens)
    marks_t_down     slack-onset time of each mark            [s]
    marks_t_up       re-engagement time of each mark          [s]
    marks_cable      cable index 0..4 of each mark            [-]
    marks_seed       mission seed of each mark                [-]
    marks_cell_index index into cell_names of each mark       [-]
    cell_names       grid-cell labels, "T<T0>_k<k_h>_I<I>"

Statistic (paper's event definition)
------------------------------------
Work within one (cell, seed) mission.  Cells with fewer than 20 marks are dropped
(7 of 25 grid cells, incl. the NULL cell).

* Events = burst heads.  Within a cable, marks are sorted by onset; a mark is a
  burst MEMBER (not a head) when its onset t_down is within 2.0 s of the previous
  mark's re-engagement t_up on the SAME cable.  Restitution bounces are therefore
  merged into their burst and are never scored as events.
* An event with onset t is CROSS-ATTRIBUTED when any re-engagement (t_up of ANY
  mark, heads or members) on a DIFFERENT cable lies in [t - 3 s, t).
* Measured share = cross-attributed events / events, pooled over the cell's seeds.

Chance baseline (POST-HOC)
--------------------------
For each mission, the whole re-engagement train (every t_up, each keeping its
cable) is circularly shifted by one uniform offset modulo the mission span
[20 s, max(t_down, t_up) of that mission]; event onsets stay fixed and the share
is recomputed.  200 shifts; rng = numpy.random.default_rng(20260914), with the
200 offsets of each mission drawn in (cell index, ascending seed) order.  The
pooled null share is reported as its mean and 97.5th percentile over the shifts.

Result reproduced here: measured 25.6-67.6 % across 18 cells, above the null's
97.5th percentile in all 18; measured / null mean = 1.4x in the densest cells
(T600_k1000_I1.0: 67.6 % vs 49.4 %) up to 21x in sparse ones (T1800_k500_I1.0:
28.2 % vs 1.3 %).  Rows are ordered by measured share (largest on top).
"""
from __future__ import annotations

import numpy as np

from tether.analysis.v2.paperfig.style import *  # noqa: F401,F403
from tether.analysis.v2.paperfig.style import (
    REPO, use_paper_style, save, COL,
    C_FLEET, C_MUTED, C_NEUTRAL,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

RECORD = REPO / "records" / "phase2" / "phase2_records.npz"

WINDOW = 3.0        # s, cross-cable attribution window [t - 3 s, t)
BURST_GAP = 2.0     # s, same-cable onset within this of previous t_up -> member
MIN_MARKS = 20      # a cell needs this many marks to be scored
SPAN_START = 20.0   # s, start of the scored mission span
N_SHIFTS = 200
SEED = 20260914


# ------------------------------------------------------------------ statistic

def _burst_heads(t_down, t_up, cable):
    """Boolean mask of burst heads (events) within one mission."""
    head = np.ones(t_down.size, dtype=bool)
    for c in np.unique(cable):
        idx = np.flatnonzero(cable == c)
        idx = idx[np.argsort(t_down[idx], kind="stable")]
        gap = t_down[idx[1:]] - t_up[idx[:-1]]
        head[idx[1:][gap <= BURST_GAP]] = False
    return head


def _n_cross(ev_t, ev_c, up_t, up_c) -> int:
    """Events with a re-engagement on a different cable in [t - WINDOW, t)."""
    order = np.argsort(up_t, kind="stable")
    up_t, up_c = up_t[order], up_c[order]
    lo = np.searchsorted(up_t, ev_t - WINDOW, "left")
    hi = np.searchsorted(up_t, ev_t, "left")          # excludes t_up == t
    n = 0
    for a, b, c in zip(lo, hi, ev_c):
        if b > a and np.any(up_c[a:b] != c):
            n += 1
    return n


def attribution_by_cell(path=RECORD):
    """-> list of dicts: cell, marks, events, measured, null_mean, null_p975."""
    d = np.load(path, allow_pickle=True)
    t_down = d["marks_t_down"]
    t_up = d["marks_t_up"]
    cable = d["marks_cable"]
    seed = d["marks_seed"]
    cell = d["marks_cell_index"]
    names = d["cell_names"]
    rng = np.random.default_rng(SEED)

    rows = []
    for ci, name in enumerate(names):
        m = cell == ci
        n_marks = int(m.sum())
        if n_marks < MIN_MARKS:
            continue
        n_ev = n_cr = 0
        null = np.zeros(N_SHIFTS)
        for s in np.unique(seed[m]):
            k = m & (seed == s)
            td, tu, cb = t_down[k], t_up[k], cable[k]
            head = _burst_heads(td, tu, cb)
            ev_t, ev_c = td[head], cb[head]
            n_ev += ev_t.size
            n_cr += _n_cross(ev_t, ev_c, tu, cb)
            span = max(td.max(), tu.max()) - SPAN_START
            for j, off in enumerate(rng.uniform(0.0, span, N_SHIFTS)):
                tu_shift = SPAN_START + np.mod(tu - SPAN_START + off, span)
                null[j] += _n_cross(ev_t, ev_c, tu_shift, cb)
        null /= n_ev
        rows.append(dict(cell=str(name), marks=n_marks, events=n_ev,
                         measured=n_cr / n_ev, null_mean=float(null.mean()),
                         null_p975=float(np.percentile(null, 97.5))))
    rows.sort(key=lambda r: r["measured"])       # ascending -> largest on top
    return rows


def _pretty(name: str) -> str:
    """'T600_k1000_I1.0' -> '600 / 1000 / 1.0'."""
    t0, kh, inten = name.split("_")
    return f"{t0[1:]} / {kh[1:]} / {inten[1:]}"


# ------------------------------------------------------------------- figure

def main() -> None:
    use_paper_style()
    rows = attribution_by_cell()
    if not rows:
        raise SystemExit(f"no cell in {RECORD} holds >= {MIN_MARKS} marks")

    labels = [_pretty(r["cell"]) for r in rows]
    events = np.array([r["events"] for r in rows])
    meas = 100.0 * np.array([r["measured"] for r in rows])
    nmean = 100.0 * np.array([r["null_mean"] for r in rows])
    np975 = 100.0 * np.array([r["null_p975"] for r in rows])
    y = np.arange(len(rows), dtype=float)

    fig, ax = plt.subplots(figsize=(COL, 0.150 * len(rows) + 1.20))

    for yi in y:                                  # faint row guides
        ax.axhline(yi, color=C_MUTED, lw=0.3, alpha=0.30, zorder=0)

    # Chance: open square at the mean, bar to the 97.5th percentile with a cap.
    ax.hlines(y, nmean, np975, color=C_NEUTRAL, lw=0.9, zorder=2)
    ax.plot(np975, y, ls="none", marker="|", ms=4.0, mew=0.9,
            color=C_NEUTRAL, zorder=2)
    ax.plot(nmean, y, ls="none", marker="s", ms=3.6, mfc="white",
            mec=C_NEUTRAL, mew=0.8, zorder=3)
    # Measured: solid circle.
    ax.plot(meas, y, ls="none", marker="o", ms=4.2, mfc=C_FLEET,
            mec=C_FLEET, mew=0.6, zorder=4)

    xmax = 80.0
    for yi, n in zip(y, events):
        ax.text(xmax + 14.0, yi, f"{n:d}", ha="right", va="center",
                fontsize=7, color=C_NEUTRAL, clip_on=False)
    ax.text(xmax + 14.0, y[-1] + 0.95, "events", ha="right", va="center",
            fontsize=7, color=C_NEUTRAL, clip_on=False)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0.0, xmax)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("cross-attributed share of events [%]")
    ax.set_ylabel("cell:  $T_0$ [N] / $k_h$ [N m rad$^{-1}$] / $I$ [-]")
    ax.xaxis.grid(True, color=C_MUTED, lw=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    h_meas = Line2D([], [], ls="none", marker="o", ms=4.2, mfc=C_FLEET,
                    mec=C_FLEET)
    h_bar = Line2D([], [], color=C_NEUTRAL, lw=0.9)
    h_mean = Line2D([], [], ls="none", marker="s", ms=3.6, mfc="white",
                    mec=C_NEUTRAL, mew=0.8)
    ax.legend([h_meas, (h_bar, h_mean)],
              ["measured share (event level)",
               "time-shift chance (post-hoc):\nmean, bar to 97.5th pct"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
              loc="upper left", bbox_to_anchor=(-0.62, -0.115), ncol=2,
              columnspacing=1.0,
              fontsize=7, handlelength=1.6, handletextpad=0.5,
              borderpad=0.0, borderaxespad=0.0, labelspacing=0.3)

    fig.subplots_adjust(left=0.33, right=0.86, top=0.97, bottom=0.16)
    save(fig, "fig_attribution")

    above = int(np.sum(meas > np975))
    ratio = meas / nmean
    print(f"cells scored: {len(rows)}")
    print(f"measured    : {meas.min():.1f}-{meas.max():.1f} %")
    print(f"above null 97.5th pct: {above}/{len(rows)}")
    print(f"measured / null mean: {ratio.min():.1f}x-{ratio.max():.1f}x")
    for r in rows[::-1]:
        print(f"  {r['cell']:22s} marks={r['marks']:5d} events={r['events']:5d}"
              f"  meas={100*r['measured']:5.1f}%  null={100*r['null_mean']:5.1f}%"
              f"  p97.5={100*r['null_p975']:5.1f}%"
              f"  x{r['measured']/r['null_mean']:5.1f}")


if __name__ == "__main__":
    main()
