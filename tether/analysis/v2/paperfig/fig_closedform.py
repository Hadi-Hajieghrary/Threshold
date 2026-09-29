"""fig_closedform -- the design-time closed form is biased where the events live.

Every plotted value is a MEASURED plant quantity from the scripted deterministic campaign,
test P1-T7 ("drag-limited depth, Prop. 4'"), divided by the closed-form depth that the same
row records beside it.  No committed prediction is plotted: the block
``/P1-T7/cells/<cell>/committed_prediction/`` (the ODE / closed-form FORECAST, including its
``ratio`` and ``deepening_ratio`` arrays) is never read by this module.

Records and key paths read
--------------------------
records/v2/phase1/scripted_results.json
    /P1-T7/cells/<cell>/rows[]/t_x                  gust duration, s
    /P1-T7/cells/<cell>/rows[]/Delta_g_plant        measured plant slack depth, m (None if censored)
    /P1-T7/cells/<cell>/rows[]/Delta_g_closed       closed-form slack depth of Prop. 4', m
    /P1-T7/cells/<cell>/rows[]/censored             True for rows voided by formation closure
    /P1-T7/cells/<cell>/rows[]/censor_reason        asserted "closure" on every censored row
    /P1-T7/cells/<cell>/rows[]/scored               declared scoring rule (closed-form depth > 5 cm)
    /P1-T7/cells/<cell>/rows[]/depth_ratio          recorded ratio on scored rows (cross-check)
    /P1-T7/cells/<cell>/rows[]/depth_within_15pct   per-row depth clause (verdict cross-check)
    /P1-T7/cells/<cell>/rows[]/deepening_within_15pct  per-row deepening clause (cross-check)
    /P1-T7/cells/<cell>/passed                      pre-declared cell verdict (plotted as the strip)
    /P1-T7/verdict                                  test-level verdict (asserted to be FAIL)
    <cell> is "<T0 in N>@<lambda>", e.g. "1000@1.05"; lambda = W^c / T0.
records/v2/phase1/scripted_addendum_1.json
    /cells[]/cell
    /cells[]/t_x_above_tau_A_scored                 t_x >= tau_A of each cell, s
    /cells[]/depth_ratio_plant_over_closed_form     their plant / closed-form ratios
    Used as an independent cross-check of the ratios recomputed from the rows (to 1e-9).
records/v2/phase1/scripted_declarations.json
    /tests/P1-T7/AIC        text source of tau_A (= 1.714 s), parsed, not typed in
    /tests/P1-T7/threshold  text source of the 15% acceptance band, parsed, not typed in
    /tests/P1-T7/scored_on  text source of the 5 cm scoring floor, parsed, not typed in

Plotted quantity: the measured depth ratio Delta_g_plant / Delta_g_closed for every uncensored
row (84 rows over the 12 lambda > 1 cells, 7 censored by closure, 77 plotted) against the
gust strength lambda, one marker shape per pretension (600, 1000, 1400 N, dodged
horizontally), with
  * filled markers      -- scored rows with t_x >= tau_A (the drift regime; the paper's table),
  * half-filled markers -- scored rows with t_x < tau_A (they count toward the cell verdict),
  * small grey rings    -- rows the declared rule does not score (closed-form depth <= 5 cm,
                           the floor parsed from /tests/P1-T7/scored_on and asserted to equal
                           the recorded ``scored`` flag on every uncensored row),
  * a line per pretension through the per-cell MEDIAN of its t_x >= tau_A ratios (for the
    2-point cells the vertex is not itself a plotted row),
  * the pre-declared +/-15% band, a verdict strip (check = pass, x = fail per cell), and a
    strip giving each cell's count of rows censored by formation closure (7 of 84),
  * an on-plot note that the largest ratio (5.13, 600 N @ 1.05, t_x = 0.2 s) is an UNSCORED
    row whose closed-form depth (1.0 mm, from the record) lies below the 5 cm floor.
The module asserts that the recorded cell verdict equals the depth clause alone over the scored
rows, so the figure's depth ratios fully explain every pass and fail shown on it.
"""
from __future__ import annotations

import json
import re

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tether.analysis.v2.paperfig.style import *  # noqa: F401,F403

RESULTS = REPO / "records" / "v2" / "phase1" / "scripted_results.json"
ADDENDUM = REPO / "records" / "v2" / "phase1" / "scripted_addendum_1.json"
DECLARATIONS = REPO / "records" / "v2" / "phase1" / "scripted_declarations.json"

T0S = (600, 1000, 1400)
DX = {600: -0.05, 1000: 0.0, 1400: +0.05}        # horizontal dodge, in lambda units
MARKER = {600: "o", 1000: "s", 1400: "^"}
LINESTYLE = {600: "-", 1000: (0, (3.5, 1.5)), 1400: (0, (1, 1.2))}
Y_VERDICT = 0.705                                 # strip row, below the band and the data
Y_CENSOR = 0.615                                  # censored-row count per cell


# ------------------------------------------------------------------ data

def declared_constants():
    """tau_A [s] and the band half-width, parsed from the pre-registered test text."""
    test = json.loads(DECLARATIONS.read_text())["tests"]["P1-T7"]
    tau_a = float(re.search(r"tau_A\s*=\s*([0-9.]+)\s*s", test["AIC"]).group(1))
    band = float(re.search(r"within\s+([0-9.]+)%", test["threshold"]).group(1)) / 100.0
    return tau_a, band


def load_cells(tau_a: float, band: float):
    """-> {(T0, lambda): dict(rows=[(t_x, ratio, scored)], passed=bool)}, cross-checked."""
    t7 = json.loads(RESULTS.read_text())["P1-T7"]
    assert t7["verdict"] == "FAIL" and t7["passed"] is False
    addendum = {c["cell"]: c for c in json.loads(ADDENDUM.read_text())["cells"]}

    out = {}
    for name, cell in t7["cells"].items():
        t0_s, lam_s = name.split("@")
        pts, closed, n_cens = [], [], 0
        for row in cell["rows"]:
            if row["censored"]:                      # formation closure: no ratio test
                assert row["Delta_g_plant"] is None and not row["scored"], name
                assert row["censor_reason"] == "closure", name
                n_cens += 1
                continue
            ratio = row["Delta_g_plant"] / row["Delta_g_closed"]
            if row["scored"]:
                assert abs(ratio - row["depth_ratio"]) < 1e-12, name
                assert row["depth_within_15pct"] == (abs(ratio - 1.0) <= band), name
            pts.append((row["t_x"], ratio, bool(row["scored"])))
            closed.append(row["Delta_g_closed"])

        # Every t_x >= tau_A row is scored; its ratios match the addendum's own array.
        long = [(t, r, s) for t, r, s in pts if t >= tau_a]
        assert all(s for _, _, s in long), name
        ref = addendum[name]
        assert [t for t, _, _ in long] == ref["t_x_above_tau_A_scored"], name
        for (_, r, _), r_ref in zip(long, ref["depth_ratio_plant_over_closed_form"]):
            assert abs(r - r_ref) < 1e-9, (name, r, r_ref)

        # The recorded verdict is reproduced by both clauses, and by the depth clause alone.
        scored = [row for row in cell["rows"] if row["scored"]]
        both = all(r["depth_within_15pct"] and r["deepening_within_15pct"] for r in scored)
        depth_only = all(abs(r_ - 1.0) <= band for _, r_, s in pts if s)
        assert bool(cell["passed"]) == both == depth_only, name

        out[(int(t0_s), float(lam_s))] = {"rows": pts, "closed": closed,
                                          "n_censored": n_cens, "n_long": len(long),
                                          "passed": bool(cell["passed"])}
    return out


# ------------------------------------------------------------------ figure

def score_floor() -> float:
    """The declared scoring floor on the closed-form depth [m], parsed from the test text."""
    test = json.loads(DECLARATIONS.read_text())["tests"]["P1-T7"]
    return float(re.search(r"exceeds\s+([0-9.]+)\s*cm", test["scored_on"]).group(1)) / 100.0


def main() -> None:
    use_paper_style()
    tau_a, band = declared_constants()
    floor = score_floor()
    data = load_cells(tau_a, band)
    for c in data.values():                       # the scoring rule is exactly the 5 cm floor
        for (_, _, scored), dg in zip(c["rows"], c["closed"]):
            assert scored == (dg > floor)
    lambdas = sorted({lam for _, lam in data})
    n_pass = sum(c["passed"] for c in data.values())
    n_cens = sum(c["n_censored"] for c in data.values())
    n_rows = n_cens + sum(len(c["rows"]) for c in data.values())
    fs = 7.0                                      # every annotation at the 7 pt floor

    fig = plt.figure(figsize=(COL, 3.62))
    ax = fig.add_axes((0.215, 0.33, 0.77, 0.663))

    # Pre-declared acceptance band around a perfect closed form.
    ax.axhspan(1 - band, 1 + band, color=C_BAND, zorder=0, lw=0)
    ax.axhline(1.0, color=C_NEUTRAL, lw=0.55, ls=(0, (4, 2)), zorder=1)

    # Marker encodings: fill says how the declared rule treats the row (greyscale-safe:
    # full / half / thin open), shape says the pretension.
    sty_long = dict(ms=3.6, mfc=C_PERLINE, mec=C_PERLINE, mew=0.5)
    sty_short = dict(ms=3.9, fillstyle="bottom", mfc=C_PERLINE, mfcalt="white",
                     mec=C_PERLINE, mew=0.7)
    sty_free = dict(ms=2.6, mfc="none", mec=C_MUTED, mew=0.45)
    sty_pass = dict(marker=r"$\checkmark$", ms=5.0, mfc=C_OK, mec=C_OK, mew=0.0)
    sty_fail = dict(marker="x", ms=3.2, mec=C_NEUTRAL, mew=0.8)

    for t0 in T0S:
        xs = {"long": [], "short": [], "free": []}
        ys = {"long": [], "short": [], "free": []}
        med_x, med_y = [], []
        for lam in lambdas:
            cell = data[(t0, lam)]
            x = lam + DX[t0]
            long = []
            for t_x, ratio, scored in cell["rows"]:
                kind = "long" if t_x >= tau_a else ("short" if scored else "free")
                xs[kind].append(x)
                ys[kind].append(ratio)
                if kind == "long":
                    long.append(ratio)
            med_x.append(x)
            med_y.append(float(np.median(long)))

            ax.plot(x, Y_VERDICT, ls="none", zorder=5,
                    **(sty_pass if cell["passed"] else sty_fail))
            k = cell["n_censored"]
            ax.text(x, Y_CENSOR, f"{k}", ha="center", va="center", fontsize=fs,
                    color=C_NEUTRAL if k else C_MUTED,
                    fontweight="bold" if k else "normal")

        ax.plot(med_x, med_y, ls=LINESTYLE[t0], color=C_PERLINE, lw=0.8, zorder=3)
        ax.plot(xs["free"], ys["free"], ls="none", marker=MARKER[t0], zorder=3, **sty_free)
        ax.plot(xs["short"], ys["short"], ls="none", marker=MARKER[t0], zorder=4,
                **sty_short)
        ax.plot(xs["long"], ys["long"], ls="none", marker=MARKER[t0], zorder=4.5,
                **sty_long)

    # Name the gust durations once, left of the lambda = 1.05 column where they separate.
    lam0 = lambdas[0]
    col = {t: r for t, r, _ in data[(1000, lam0)]["rows"]}
    x_lab = lam0 + DX[600] - 0.024
    for t_x in sorted(col):
        if t_x >= tau_a:
            continue
        ax.text(x_lab, col[t_x], rf"{t_x:g} s", fontsize=fs, color=C_NEUTRAL,
                va="center", ha="right")
    ax.text(x_lab, 6.3, r"gust $t_x$", fontsize=fs, color=C_NEUTRAL, va="center",
            ha="right", fontstyle="italic")
    long_t = [t for t in col if t >= tau_a]
    long_r = [r for t, r in col.items() if t >= tau_a]
    ax.text(x_lab, float(np.exp(np.mean(np.log(long_r)))),
            rf"{min(long_t):g}$-${max(long_t):g} s", fontsize=fs, color=C_NEUTRAL,
            va="center", ha="right")

    # The largest ratio on the figure is an UNSCORED row: say so where it sits.
    (t0m, lamm), im = max(((key, i) for key, c in data.items()
                           for i in range(len(c["rows"]))),
                          key=lambda ki: data[ki[0]]["rows"][ki[1]][1])
    t_m, r_m, s_m = data[(t0m, lamm)]["rows"][im]
    dg_m = data[(t0m, lamm)]["closed"][im]
    assert not s_m and dg_m <= floor
    ax.text(lamm + DX[1400] + 0.045, r_m,
            f"max {r_m:.2f} ({t0m} N): unscored,\n"
            rf"$\Delta g_{{\mathrm{{closed}}}}$ = {dg_m * 1e3:.1f} mm < {floor * 100:g} cm",
            fontsize=fs, color=C_NEUTRAL, ha="left", va="center", linespacing=1.1)

    # Separate the bookkeeping strips from the ratio axis.
    ax.axhline(0.775, color=C_MUTED, lw=0.4, zorder=1)

    ax.set_yscale("log")
    ax.set_ylim(0.555, 7.0)
    yt = [Y_CENSOR, Y_VERDICT, 1 - band, 1.0, 1 + band, 1.5, 2.0, 3.0, 5.0]
    ax.set_yticks(yt)
    ax.set_yticklabels(["censored", "verdict", f"{1 - band:.2f}", "1.00",
                        f"{1 + band:.2f}", "1.5", "2", "3", "5"])
    ax.set_yticks([], minor=True)
    for i in (0, 1):
        ax.get_yticklabels()[i].set_fontstyle("italic")
        ax.yaxis.get_major_ticks()[i].tick1line.set_visible(False)

    ax.set_xlim(0.74, 2.1)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{lam:.2f}" for lam in lambdas])
    ax.set_xlabel(r"gust strength $\lambda = W^{c}/T_0$ ($-$)")
    ax.set_ylabel(r"depth ratio $\Delta g_{\mathrm{plant}}\,/\,\Delta g_{\mathrm{closed}}$ ($-$)")
    ax.tick_params(axis="both", which="major", length=2.5)
    ax.grid(axis="y", color=C_MUTED, lw=0.35, alpha=0.25)
    ax.set_axisbelow(True)

    blank = Line2D([], [], ls="none", label=" ")
    handles = [
        Line2D([], [], color=C_PERLINE, ls=LINESTYLE[t0], lw=0.8, marker=MARKER[t0],
               label=f"$T_0$ = {t0} N", **sty_long)
        for t0 in T0S
    ] + [
        Patch(facecolor=C_BAND, edgecolor="none",
              label=rf"declared $\pm{band * 100:.0f}\%$ band"),
        Line2D([], [], ls="none", label=f"cell passes ({n_pass} of {len(data)})",
               **sty_pass),
        Line2D([], [], ls="none", marker="o", label=rf"scored, $t_x \geq \tau_A$ = {tau_a:.2f} s",
               **sty_long),
        Line2D([], [], ls="none", marker="o", label=r"scored, $t_x < \tau_A$", **sty_short),
        Line2D([], [], ls="none", marker="o",
               label=rf"unscored ($\Delta g_{{\mathrm{{closed}}}} \leq {floor * 100:g}$ cm)",
               **sty_free),
        blank,
        Line2D([], [], ls="none", label=f"cell fails ({len(data) - n_pass} of {len(data)})",
               **sty_fail),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, fontsize=fs, handlelength=2.1, handletextpad=0.4,
               labelspacing=0.28, columnspacing=0.9, borderaxespad=0.0,
               title=r"lines: per-cell median of the $t_x \geq \tau_A$ ratios",
               title_fontsize=fs)

    # Console summary of the plotted values, for auditing against the record.
    for lam in lambdas:
        for t0 in T0S:
            c = data[(t0, lam)]
            lr = [r for t, r, _ in c["rows"] if t >= tau_a]
            print(f"  {t0:4d} N @ {lam:.2f}: t_x>=tau_A ratios "
                  + ", ".join(f"{r:.4f}" for r in lr)
                  + f" (median {np.median(lr):.4f})"
                  + f" | all-rows max {max(r for _, r, _ in c['rows']):.4f}"
                  + f" | censored {c['n_censored']}"
                  + f" | {'PASS' if c['passed'] else 'fail'}")
    print(f"  max ratio {r_m:.4f} at {t0m} N @ {lamm}, t_x = {t_m} s, scored={s_m},"
          f" Delta_g_closed = {dg_m * 1e3:.3f} mm;"
          f" {n_pass} of {len(data)} cells pass;"
          f" {sum(len(c['rows']) for c in data.values())} uncensored rows plotted,"
          f" {n_cens} of {n_rows} censored")

    save(fig, "fig_closedform")


if __name__ == "__main__":
    main()
