"""Explainer slides for the presentation (Presentation/STORYBOARD.md, "The slides").

Every number on a slide is read from a record here and registered in the ``slides`` manifest; the
few that are not stored as a key (the catch's replay median) are computed from the record and
labelled as computed.  Each slide is rendered to its own MP4 segment with the same encoder settings
as the clips, so the assembler can concatenate them.

Records read:
  records/v2/phase5/p5_t2prime_results.json   reports.{linearized_H1,rollout_H1}.all / .coincidence
  records/phase6/phase6_results.json          tests.P6-T2.{probabilities,differences}, tests.P6-T4.cost
  records/v2/phase6/p6_t0_results.json        verdict, tables.fleet.true10.first_severance,
                                              excursions[*].runs[*] (replay median), taut_overload
  records/v2/phase1/phase1_gate.json          tests.P1-T6, tests.P1-T11
  records/v2/phase1/stochastic_results.json   trade_off, closest_to_both
"""
from __future__ import annotations

import json
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from tether.analysis.v2.present import common as C

R_P5 = "records/v2/phase5/p5_t2prime_results.json"
R_P6 = "records/phase6/phase6_results.json"
R_T0 = "records/v2/phase6/p6_t0_results.json"
R_GATE = "records/v2/phase1/phase1_gate.json"
R_STOCH = "records/v2/phase1/stochastic_results.json"


def _load(path):
    return json.loads((C.REPO / path).read_text())


def numbers(man: C.Manifest) -> dict:
    """Read every slide number from its record and register it."""
    p5, p6, t0, gate, st = (_load(p) for p in (R_P5, R_P6, R_T0, R_GATE, R_STOCH))
    for p in (R_P5, R_P6, R_T0, R_GATE, R_STOCH):
        man.source(p)
    n = {}
    lin, rol = p5["reports"]["linearized_H1"], p5["reports"]["rollout_H1"]
    for tag, rep in (("perline", lin), ("rollout", rol)):
        a = rep["all"]
        n[f"{tag}_slope"] = (a["slope"], a["slope_lo"], a["slope_hi"])
        n[f"{tag}_auroc"] = a["auroc"]
        n[f"{tag}_ece"] = a["ece"]
        n[f"{tag}_ticks"] = a["n_ticks"]
        c = rep["coincidence"]
        n[f"{tag}_top"] = (c["top_bin_misses"], c["top_bin_ticks"], c["top_bin_misses_coupled"])
        n[f"{tag}_bottom"] = (c["bottom_bin_events"], c["bottom_bin_events_coupled"])
        man.value(f"{tag} recalibration slope [95% CI]", n[f"{tag}_slope"], "", f"{R_P5} reports.{'linearized_H1' if tag == 'perline' else 'rollout_H1'}.all.slope/slope_lo/slope_hi")
        man.value(f"{tag} AUROC", a["auroc"], "", f"{R_P5} reports.*.all.auroc")
        man.value(f"{tag} ECE", a["ece"], "", f"{R_P5} reports.*.all.ece")
        man.value(f"{tag} top-bin false alarms / top-bin ticks (coupled)", n[f"{tag}_top"], "ticks",
                  f"{R_P5} reports.*.coincidence.top_bin_misses/top_bin_ticks/top_bin_misses_coupled",
                  "'misses' in the record's key = non-events forecast >= 0.9, i.e. false alarms")
    probs, diffs = p6["tests"]["P6-T2"]["probabilities"], p6["tests"]["P6-T2"]["differences"]
    n["sev"] = {k: probs[k] for k in ("N", "P", "O", "L")}
    n["sev_NP"] = diffs["N-P"]
    n["dock"] = p6["tests"]["P6-T4"]["cost"]["P"]["docking_error_penalty_m"]
    man.value("severance per mission N/P/O/L", n["sev"], "", f"{R_P6} tests.P6-T2.probabilities")
    man.value("paired N-P [mean, lo95, hi95]", n["sev_NP"], "", f"{R_P6} tests.P6-T2.differences.N-P")
    man.value("docking-error penalty, eased arm [mean, lo95, hi95]", n["dock"], "m", f"{R_P6} tests.P6-T4.cost.P.docking_error_penalty_m")
    fs = t0["tables"]["fleet"]["true10"]["first_severance"]
    n["catch"] = {v: (fs[v]["caught"], fs[v]["n"], fs[v]["median_ratio"], fs[v]["interval95"][1]) for v in ("hold", "reverse")}
    man.value("catch caught / n, median v/v_b, CI upper (hold, reverse)", n["catch"], "",
              f"{R_T0} tables.fleet.true10.first_severance.{{hold,reverse}}.{{caught,n,median_ratio,interval95}}")
    rep_ratios = [run["ratio"] for e in t0["excursions"] if e["tags"]["first_severance"]
                  for run in e["runs"] if run["mode"] == "fleet" and run["controller"] == "replay"]
    assert len(rep_ratios) == 16
    n["replay_median"] = float(np.median(rep_ratios))
    man.value("recorded thrust replayed: median v/v_b over the 16", n["replay_median"], "",
              f"{R_T0} excursions[first_severance].runs[fleet,replay].ratio", "computed: median over 16")
    man.value("admissibility bar", 0.80, "", f"{R_T0} verdict.rule")
    to = t0["taut_overload"]["phase5_40"]
    n["floor"] = (to["counts_50ms"]["snap"], to["counts_50ms"]["taut_overload"], to["taut_overload_floor_p_arm"], to["p_N"])
    man.value("first severances snap / taut overload; floor; p_N", n["floor"], "", f"{R_T0} taut_overload.phase5_40")
    t6 = gate["tests"]["P1-T6"]
    n["t6"] = (t6["measured_crossing_pooled"], t6["predicted"])
    man.value("P1-T6 pooled measured crossing vs predicted", n["t6"], "lambda", f"{R_GATE} tests.P1-T6")
    cells = {c["cell"]: c for c in st["trade_off"]}
    a, b = cells["i035_T0600_ks3"], cells["i035_T1000_ks3"]
    n["trade"] = (a["chord_world_std_deg"], b["chord_world_std_deg"], a["marks"], b["marks"])
    man.value("chord-angle std 0.6 kN vs 1.0 kN; marks 0.6 vs 1.0 kN", n["trade"], "deg / marks",
              f"{R_STOCH} trade_off[i035_T0600_ks3, i035_T1000_ks3].chord_world_std_deg/marks")
    n["best_gap"] = st["closest_to_both"]["ranked"][0]["combined"]
    man.value("closest cell's combined shortfall", n["best_gap"], "x", f"{R_STOCH} closest_to_both.ranked[0].combined")
    n["bunch"] = bunching(man)
    n.update(prediction_caveats(man, p5))
    n["easing_pairs"] = easing_pairs(man)
    cs_ = t0["critical_speeds"]
    vb = [float(v) for v in (cs_.values() if isinstance(cs_, dict) else cs_)]
    n["vb"] = (min(vb), max(vb))
    man.value("severing speed v_b range over cable positions", n["vb"], "m/s", f"{R_T0} critical_speeds")
    n["cascade"] = cascade_stats(man)
    return n


def easing_pairs(man: C.Manifest) -> dict:
    """Paired live outcomes of arms N and P over the 60 seeds, and the closures behind the N-only seeds."""
    import pickle
    pkl = C.REPO / "records/phase6/cache/phase6_compute.pkl"
    res = pickle.load(open(pkl, "rb"))["results"]
    tab = {(r["mode"], r["arm"], r["seed"]): r for r in res}
    seeds = sorted({r["seed"] for r in res})
    n_sev = {a: sum(1 for s in seeds if tab[("live", a, s)]["live_first"]) for a in ("N", "P")}
    n_only = [s for s in seeds if tab[("live", "N", s)]["live_first"] and not tab[("live", "P", s)]["live_first"]]
    p_only = [s for s in seeds if tab[("live", "P", s)]["live_first"] and not tab[("live", "N", s)]["live_first"]]
    closure = lambda r: (r["outcome"].get("closure") if isinstance(r["outcome"], dict) else None)
    man.check("every N-only seed's P run ended in a formation closure before any severance",
              all(closure(tab[("live", "P", s)]) for s in n_only), f"N-only seeds {n_only}")
    man.check("every P-only seed's N run finished with neither severance nor closure",
              all(not closure(tab[("live", "N", s)]) for s in p_only), f"P-only seeds {p_only}")
    n_L = sum(1 for s in seeds if tab[("live", "L", s)]["live_first"])
    out = {"n": len(seeds), "N_sev": n_sev["N"], "P_sev": n_sev["P"], "N_only": len(n_only), "P_only": len(p_only),
           "L_sev": n_L}
    man.source("records/phase6/cache/phase6_compute.pkl")
    man.value("easing, live: missions severed N / P of n; N-only / P-only seeds", out, "",
              "records/phase6/cache/phase6_compute.pkl results[mode=live].live_first, outcome.closure")
    return out


def cascade_stats(man: C.Manifest) -> dict:
    """The attribution numbers, from the verified cascade_statistics clip manifest (itself computed from
    records/phase2/phase2_records.npz with fig_attribution's code)."""
    m = json.loads((C.MANIFESTS / "cascade_statistics.json").read_text())
    v = {x["label"]: x["value"] for x in m["values"]}
    out = {"share": v["measured share range"], "above": v["cells above the chance 97.5th percentile"],
           "factor": v["measured / chance mean range"]}
    man.value("measured cross-cable share range (%); cells above chance; factor range", out, "",
              "Presentation/manifests/cascade_statistics.json (computed from records/phase2/phase2_records.npz)",
              "post hoc")
    return out


V3_RECORDS = {"wp1": C.REPO / "records/v3/wp1_results.json", "wp2": C.REPO / "records/v3/wp2_results.json",
              "wp3": C.REPO / "records/v3/wp3_results.json"}
V3_MANIFEST = C.MANIFESTS / "intervention.json"


def v3_numbers(man: C.Manifest) -> dict | None:
    """Plan v3 (cascade criticality): the verdict counts and sweep values the criticality slide quotes, read from
    records/v3/wp{1,2,3}_results.json and the intervention clip's manifest; None while any of them is absent."""
    if not V3_MANIFEST.exists() or not all(p.exists() for p in V3_RECORDS.values()):
        return None
    for p in list(V3_RECORDS.values()) + [V3_MANIFEST]:
        man.source(p)
    w1, w2, w3 = (json.loads(V3_RECORDS[k].read_text()) for k in ("wp1", "wp2", "wp3"))
    clip = json.loads(V3_MANIFEST.read_text())
    vals = {v["label"]: v["value"] for v in clip["values"]}
    from tether.campaign.v3 import cells as v3cells
    sweep = sorted(((v3cells.parse_cell(n)["intensity"], w3["cells"][n]["rho"]) for n in v3cells.sweep_cells() if "rho" in w3["cells"].get(n, {})))
    t = {tid: w[ "tests"][tid] for w, ids in ((w1, ("T1.1", "T1.2", "T1.3", "T1.4")), (w2, ("T2.2", "T2.3", "T2.4", "T2.5")), (w3, ("T3.1", "T3.2", "T3.4"))) for tid in ids}
    out = {
        "n_f": int(vals["factual other-cable onsets within 3 s"]), "n_c": int(vals["counterfactual other-cable onsets within 3 s"]),
        "causal": int(vals["causal offspring"]),
        "t11": (t["T1.1"]["passing_cells"], t["T1.1"]["scored_cells"], t["T1.1"]["verdict"]),
        "t12": (t["T1.2"]["passing_cells"], t["T1.2"]["scored_cells"]), "t13": (t["T1.3"]["passing_cells"], t["T1.3"]["scored_cells"]),
        "t14": t["T1.4"]["verdict"], "t22": (t["T2.2"]["passing_cells"], t["T2.2"]["scored_cells"]),
        "t23": (t["T2.3"]["verdict"], t["T2.3"]["n"], t["T2.3"]["abs_dv_m_s"]["p95"], t["T2.3"]["onset_set_reproduced_share"]),
        "t24": (t["T2.4"]["passing_cells"], t["T2.4"]["scored_cells"], t["T2.4"]["verdict"]),
        "t25": (t["T2.5"]["passing_cells"], t["T2.5"]["scored_cells"]), "kernel_path": w3.get("kernel_path"),
        "t31": (t["T3.1"]["passing_cells"], t["T3.1"]["scored_cells"]), "t32": (t["T3.2"]["passing_cells"], t["T3.2"]["scored_cells"]),
        "t34": t["T3.4"], "sweep": sweep,
    }
    for k, v in out.items():
        man.value(f"v3 {k}", v, "", "records/v3/wp*_results.json / Presentation/manifests/intervention.json")
    return out


def prediction_caveats(man: C.Manifest, p5: dict) -> dict:
    """Numbers for the prediction caveats slide, each from its record (the clip-0.01 interval recomputed post hoc
    with the declared cluster-bootstrap statistic, as the paper's Sec. IV does)."""
    out = {}
    v1 = _load("records/phase5/phase5_results.json")
    man.source("records/phase5/phase5_results.json")
    cP, cO = v1["arms"]["P"]["calibration"], v1["arms"]["O"]["calibration"]
    out["est_slope"] = (cP["slope"], cP["slope_lo"], cP["slope_hi"])
    out["exact_slope_v1"] = cO["slope"]
    man.value("estimator-fed per-line slope (v1 event definition) vs exact-state", (out["est_slope"], cO["slope"]), "",
              "records/phase5/phase5_results.json arms.P.calibration / arms.O.calibration")
    d = p5["post_hoc_diagnostics"]["forecasts"]
    out["degenerate"] = d["rollout_H1"]["h_zero"] + d["rollout_H1"]["h_one"]
    out["certain_wrong"] = d["linearized_H1"]["h_zero_events"] + d["linearized_H1"]["h_one_nonevents"]
    man.value("rollout forecasts exactly 0 or 1; per-line certain-and-wrong", (out["degenerate"], out["certain_wrong"]), "ticks",
              "records/v2/phase5/p5_t2prime_results.json post_hoc_diagnostics.forecasts.*.h_zero/h_one/...")
    t = np.load(C.REPO / "records/v2/phase5/p5_t2prime_ticks.npz")
    man.source("records/v2/phase5/p5_t2prime_ticks.npz")
    keep = (~t["bounce"]) & (~t["censored_H1"])
    fa = keep & (t["forecast_linearized_H1"] >= 0.9) & (~t["label_H1"].astype(bool)) & t["coupled_H1"].astype(bool)
    out["fa_mid"] = (int((fa & (t["time"] >= 70) & (t["time"] < 110)).sum()), int(fa.sum()))
    man.value("coupled false alarms at 70-110 s (after the squall, before deceleration) / all coupled", out["fa_mid"], "ticks",
              "records/v2/phase5/p5_t2prime_ticks.npz (h_linearized >= 0.9, label 0, coupled)", "post hoc")
    cache = C.CACHE / "slides_clip01.json"
    digest = C.sha256(C.REPO / "records/v2/phase5/p5_t2prime_ticks.npz")
    if cache.exists() and json.loads(cache.read_text()).get("sha256") == digest:
        out["clip01"] = tuple(json.loads(cache.read_text())["clip01"])
    else:
        from tether.monitor.metrics import calibration_report
        y, cl, h = t["label_H1"][keep].astype(bool), t["interval"][keep], t["forecast_linearized_H1"][keep].astype(float)
        rep = calibration_report(h, y, cl, n_bins=10, n_boot=2000, clip=0.01, confidence=0.95,
                                 rng=np.random.default_rng(np.random.SeedSequence([20260913, 3])))
        out["clip01"] = (rep.slope, rep.slope_lo, rep.slope_hi)
        cache.write_text(json.dumps({"sha256": digest, "clip01": out["clip01"]}))
    man.value("per-line slope at clip 0.01 [95% CI]", out["clip01"], "",
              "recomputed: tether.monitor.metrics.calibration_report on p5_t2prime_ticks.npz (declared statistic, clip 0.01)",
              "post hoc")
    return out


def bunching(man: C.Manifest):
    """Post hoc: in the 40 squall missions, how often do two tug centres come within 1 m?
    (hulls are 3 m x 1 m and the plant models no contact).  Cached against the pickle's sha256."""
    from itertools import combinations
    pkl = C.REPO / "records/phase5/cache/phase5_missions.pkl"
    digest = C.sha256(pkl)
    cache = C.CACHE / "slides_bunching.json"
    if cache.exists() and json.loads(cache.read_text()).get("sha256") == digest:
        b = json.loads(cache.read_text())
    else:
        import pickle
        missions = pickle.load(open(pkl, "rb"))
        fr = []
        for m in missions:
            pos = m["truth"].state[:, 3:18].reshape(-1, 5, 3)[:, :, :2]
            dmin = np.full(len(pos), np.inf)
            for i, j in combinations(range(5), 2):
                dmin = np.minimum(dmin, np.linalg.norm(pos[:, i] - pos[:, j], axis=1))
            fr.append(float(np.mean(dmin < 1.0)))
        fr = np.array(fr)
        b = {"sha256": digest, "missions": int(fr.size), "with_pair_within_1m": int((fr > 0).sum()),
             "median_fraction": float(np.median(fr))}
        cache.write_text(json.dumps(b, indent=1))
    man.source("records/phase5/cache/phase5_missions.pkl")
    man.value("missions with two tug centres within 1 m; median fraction of mission time",
              (b["with_pair_within_1m"], b["missions"], b["median_fraction"]), "",
              "records/phase5/cache/phase5_missions.pkl truth.state (vessel x, y at 10 ms)",
              "post hoc: min pairwise vessel-centre distance < 1 m")
    return b


# ---------------------------------------------------------------- drawing helpers

def _base(fig, kicker: str, heading: str):
    fig.clf()
    fig.text(0.06, 0.90, kicker.upper(), fontsize=C.FS_SMALL, color=C.TAUT, weight="bold", ha="left")
    fig.text(0.06, 0.845, heading, fontsize=C.FS_TITLE + 4, color=C.INK, weight="bold", ha="left", va="top")
    fig.add_artist(Line2D([0.06, 0.94], [0.765, 0.765], color=C.FAINT, lw=1.5))


def _bullets(fig, items, y0=0.70, dy=0.085, x=0.075, fs=None, color=None, line=None, gap=0.03):
    """Bullets top-down.  With ``line`` (height of one text line, figure fraction) the spacing follows each
    bullet's number of lines; otherwise a fixed step ``dy``."""
    arts, y = [], y0
    for k, s in enumerate(items):
        arts.append(fig.text(x, y, "•  " + s, fontsize=fs or C.FS_BODY + 3, color=color or C.INK,
                             ha="left", va="top", wrap=True))
        y -= (line * (s.count("\n") + 1) + gap) if line else dy
    return arts


def _table(fig, rows, col_x, y0, dy, header_color=C.MUTED, fs=None, colors=None):
    fs = fs or C.FS_BODY + 1
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            col = header_color if r == 0 else (colors[c] if colors else C.INK)
            fig.text(col_x[c], y0 - r * dy, cell, fontsize=fs if r else C.FS_SMALL + 1, color=col,
                     ha="left", va="top", weight="bold" if r == 0 else "normal")
    top, bot = y0 + 0.015, y0 - (len(rows) - 0.35) * dy
    fig.add_artist(Line2D([col_x[0], 0.94], [y0 - 0.55 * dy] * 2, color=C.FAINT, lw=1.2))
    return top, bot


class Slides:
    """Renders each slide as a sequence of beats: (seconds, caption, draw-callback)."""

    def __init__(self):
        C.ensure_dirs()
        (C.CLIPS / "slides").mkdir(parents=True, exist_ok=True)
        self.man = C.Manifest(name="slides", title="Explainer slides",
                              story="Title, framing, summaries and limits between the simulation clips.")
        self.n = numbers(self.man)
        self.segments = []

    def render(self, name: str, beats, footer: str = "", script=None):
        """``script``: the narration of a text slide as [(offset seconds, text)], written to SCRIPT.md."""
        fig = C.new_frame()
        if os.environ.get("SLIDES_DRY"):          # layout check only: draw every beat, write no video
            for seconds, text, draw in beats:
                if draw is not None:
                    draw(fig)
            plt.close(fig)
            print(f"dry {name}: ok")
            return
        w = C.writer()
        path = C.CLIPS / "slides" / f"{name}.mp4"
        frames = 0
        cap = None
        with w.saving(fig, str(path), dpi=C.DPI):
            for seconds, text, draw in beats:
                if draw is not None:
                    draw(fig)
                    if footer:
                        C.footer(fig, footer)
                if cap is not None:
                    cap.remove()
                cap = C.caption(fig, text) if text else None
                frames += C.hold(w, seconds)
        plt.close(fig)
        self.man.frames += frames
        caps = [{"t": round(t, 2), "text": x} for t, x in (script or [])]
        caps += [{"t": None, "text": b[1]} for b in beats if b[1]]
        self.segments.append({"name": name, "path": str(path.relative_to(C.REPO)), "frames": frames,
                              "seconds": frames / C.FPS, "captions": caps})
        print(f"slide {name}: {frames / C.FPS:.1f} s")


# ---------------------------------------------------------------- sentence layout

W_WORDS_PER_S = 3.6          # reading pace for slide prose, read silently (captions in the clips: <= 2.5 words/s)


def _words(text: str) -> int:
    return len(text.replace("\n", " ").split())


def _pace(text: str, extra: float = 1.0, floor: float = 5.0) -> float:
    return max(floor, round(_words(text) / W_WORDS_PER_S + extra, 1))


def _headline(fig, kicker: str, sentence: str, y: float = 0.915):
    """A small orientation label and the slide's point as one bold sentence."""
    fig.clf()
    fig.text(0.07, y, kicker.upper(), fontsize=C.FS_SMALL, color=C.TAUT, weight="bold", ha="left", va="top")
    wrapped = textwrap.fill(sentence, 62)
    fig.text(0.07, y - 0.045, wrapped, fontsize=C.FS_TITLE + 2, color=C.INK, weight="bold", ha="left", va="top",
             linespacing=1.18)
    n_lines = wrapped.count("\n") + 1
    y_rule = y - 0.045 - n_lines * 0.056 - 0.02
    fig.add_artist(Line2D([0.07, 0.93], [y_rule, y_rule], color=C.FAINT, lw=1.5))
    return y_rule - 0.035


def _prose(fig, paras, y0: float, fs: float = None, width: int = 88, gap: float = 0.028, x: float = 0.07):
    """Paragraphs of complete sentences, wrapped at ``width`` characters; a paragraph given as
    (text, "bold") is emphasised.  Returns the y below the last paragraph."""
    fs = fs or C.FS_BODY + 4
    line_h = fs * 1.42 / 72 * C.DPI / C.H
    y = y0
    for para in paras:
        text, style = (para, "") if isinstance(para, str) else para
        wrapped = textwrap.fill(text, width)
        fig.text(x, y, wrapped, fontsize=fs, color=C.INK, ha="left", va="top", linespacing=1.42,
                 weight="bold" if style == "bold" else "normal")
        y -= (wrapped.count("\n") + 1) * line_h + gap
    return y


def _numbered(fig, items, y0: float, fs: float = None, width: int = 96, gap: float = 0.022, start: int = 1):
    """Numbered paragraphs (for the list of contributions, whose order is the paper's)."""
    fs = fs or C.FS_BODY + 2
    line_h = fs * 1.42 / 72 * C.DPI / C.H
    y = y0
    for k, (lead, text) in enumerate(items, start):
        wrapped = textwrap.fill(text, width)
        fig.text(0.07, y, f"{k}", fontsize=fs + 6, color=C.TAUT, weight="bold", ha="left", va="top")
        fig.text(0.105, y, lead, fontsize=fs, color=C.INK, weight="bold", ha="left", va="top")
        fig.text(0.105, y - line_h, wrapped, fontsize=fs, color=C.INK, ha="left", va="top", linespacing=1.42)
        y -= (wrapped.count("\n") + 2) * line_h + gap
    return y


def _fit(kicker, headline, paras, fs, width):
    """Largest type size (from ``fs`` down) at which the full slide fits above the footer; the wrap width
    grows as the type shrinks so lines keep the same physical length."""
    fig = C.new_frame()
    try:
        for f in range(int(fs), int(C.FS_BODY) - 3, -1):
            w = int(round(width * fs / f))
            y = _headline(fig, kicker, headline)
            yb = _prose(fig, paras, y, fs=f, width=w)
            fig.clf()
            if yb > 0.05:
                return f, w
        raise AssertionError(f"{kicker}: the text does not fit even at {C.FS_BODY - 2} pt")
    finally:
        plt.close(fig)


def _reveal(S, name, kicker, headline, paras, footer="", width=88, fs=None, first_extra=2.0):
    """One slide: the headline, then the paragraphs revealed one at a time, each held long enough to
    read at W_WORDS_PER_S.  The type size is fitted once so the full slide never overflows."""
    fs, width = _fit(kicker, headline, paras, fs or C.FS_BODY + 4, width)
    beats = []
    for k in range(1, len(paras) + 1):
        def draw(fig, k=k):
            y = _headline(fig, kicker, headline)
            yb = _prose(fig, paras[:k], y, fs=fs, width=width)
            assert yb > 0.05, f"{name}: text overflows the frame (bottom at {yb:.3f})"
        para_text = paras[k - 1] if isinstance(paras[k - 1], str) else paras[k - 1][0]
        secs = _pace(para_text) + (first_extra + _words(headline) / W_WORDS_PER_S if k == 1 else 0.0)
        beats.append((round(secs, 1), "", draw))
    offsets = [0.0]
    for b in beats[:-1]:
        offsets.append(offsets[-1] + b[0])
    script = [(0.0, headline)] + [(o, p if isinstance(p, str) else p[0]) for o, p in zip(offsets, paras)]
    S.render(name, beats, footer=footer, script=script)


def _bridge(S, name, part, question, text):
    """Before a movie: which part of the story, the question the movie answers, and what it shows."""
    def draw(fig):
        fig.clf()
        fig.text(0.5, 0.70, part.upper(), fontsize=C.FS_SUB, color=C.TAUT, weight="bold", ha="center")
        fig.text(0.5, 0.60, textwrap.fill(question, 48), fontsize=36, color=C.INK, weight="bold", ha="center",
                 va="top", linespacing=1.15)
        fig.text(0.5, 0.40, textwrap.fill(text, 84), fontsize=C.FS_BODY + 4, color=C.INK, ha="center", va="top",
                 linespacing=1.45)
    S.render(name, [(_pace(question + " " + text, extra=2.5, floor=9.0), "", draw)],
             script=[(0.0, f"{part}: {question}"), (0.0, text)])


# ---------------------------------------------------------------- the slides

def build() -> list:
    S = Slides()
    n = S.n
    man = S.man
    ep = n["easing_pairs"]
    cs = n["cascade"]

    # --- title
    def title(fig):
        fig.clf()
        fig.text(0.5, 0.70, "Cable Severance in Cooperative Towing\nis a Fleet Problem", fontsize=46,
                 weight="bold", color=C.INK, ha="center", va="center", linespacing=1.15)
        fig.text(0.5, 0.555, "Limits of per-line prediction and of thrust-based mitigation", fontsize=26,
                 color=C.MUTED, ha="center", va="center")
        fig.text(0.5, 0.43, textwrap.fill(
            "When several tugs tow one object, a snap on one towline changes the loads on all the others. "
            "This presentation uses simulations of five tugs and one payload to show what that means for "
            "predicting, preventing and designing against towline failure.", 90),
            fontsize=C.FS_BODY + 3, color=C.INK, ha="center", va="top", linespacing=1.4)
        fig.text(0.5, 0.24, "Hadi Hajieghrary", fontsize=C.FS_SUB, color=C.INK, ha="center")
        fig.text(0.5, 0.17, "Every movie shows recorded or replayed simulation state, or charts computed from the campaign records; "
                 "every number is traced to those records.",
                 fontsize=C.FS_SMALL + 1, color=C.TAUT, ha="center")
    S.render("s01_title", [(14.0, "", title)], script=[(0.0, "Cable Severance in Cooperative Towing is a Fleet Problem. "
             "When several tugs tow one object, a snap on one towline changes the loads on all the others. This presentation "
             "uses simulations of five tugs and one payload to show what that means for predicting, preventing and designing "
             "against towline failure.")])

    # --- the problem
    _reveal(S, "s01b_problem", "The problem",
            "Towlines are judged one at a time, but in a fleet they all hold the same payload.", [
        "Towing a large floating object, such as a caisson or a barge, often takes several tugs, each connected "
        "to the object by its own towline.",
        "A towline can only pull. In steady towing it carries a steady tension, its pretension T0. When the weather "
        "pushes a tug toward the object, its line goes slack; when the line tightens again (it re-engages), it does so "
        "with a sudden jolt called a snap. The snap's peak tension can be many times the pretension, and it can break "
        "the line.",
        "Today, engineers predict, prevent and design against snaps one line at a time: the danger to a line is "
        "judged from that line's own motion, and the motion of its far end is taken as given, as if the other lines "
        "did not respond.",
        ("This paper asks whether that is valid when several lines hold the same object. If a snap on one line "
         "changes the loads on the others, per-line tools will misjudge the danger exactly when it is greatest.", "bold"),
    ])

    # --- why it matters
    _reveal(S, "s02_why", "Why it matters",
            "Field data suggest that lines holding one structure do not fail independently.", [
        "On the Norwegian continental shelf between 2010 and 2013, the regulator recorded about 88 single-line, "
        "10 double-line and 2 triple-line mooring failures per 10,000 line-years (Kvitrud, 2014). Double and triple "
        "failures are far more frequent than independent lines would allow.",
        "The explanations on offer all act after a first line has already broken. None explains how lines could "
        "affect each other before any line parts, and that is the question this paper takes up.",
    ], footer="Field rates: Kvitrud, OMAE 2014 (cited in the paper, Sec. I)")

    # --- contributions
    contrib = [
        ("The mechanism.",
         "A snap on one line jolts the payload that all the lines share, and the payload's response changes the "
         "tension in the other lines, by up to about 0.44 of the snap's peak according to our derivation (not measured). "
         f"In the simulations, {cs['share'][0]:.0f} to {cs['share'][1]:.0f} percent of slack events follow another line's "
         f"snap within 3 seconds, more often than a chance baseline computed after the fact, in all {cs['above'][0]} "
         "configurations with enough events to score."),
        ("Prediction.",
         "A hazard model that watches one line, even when it is given the exact state, gives almost all its confident "
         "false alarms while another line is snapping. A model that simulates the whole fleet from the same state, and "
         "is also given the known forcing, gives none; it is a simulation benchmark, not a deployable monitor."),
        ("Mitigation.",
         "The two thrust laws we tested, easing every tug's thrust and making the slack tug match the payload's speed, "
         "did not reduce how often lines broke; other laws were not tested."),
        ("Design.",
         "Designers use a formula to turn the expected weather into a rate of line breaks. On the tested grid, no "
         "setting both keeps the formation in shape and produces enough slack events to fit that formula. Its new "
         "ingredient, a drag-based rule for which gusts hold a line slack, holds and can be used line by line."),
    ]
    head_c = ("The paper shows, in simulation, that a snap on one towline changes the loads on the other lines, and tests "
              "what this means for prediction, mitigation and design.")
    foot_c = "All results come from a planar multibody simulation of five tugs and one payload, run under a largely pre-registered protocol."
    for name, part, first in (("s02c_contrib", contrib[:2], 1), ("s02d_contrib", contrib[2:], 3)):
        beats = []
        for k in range(1, len(part) + 1):
            def draw(fig, k=k, part=part, first=first):
                y = _headline(fig, "What this paper contributes" + ("" if first == 1 else " (continued)"), head_c)
                yb = _numbered(fig, part[:k], y, fs=C.FS_BODY + 2, width=92, start=first)
                assert yb > 0.06, "the contributions overflow the frame"
            secs = _pace(part[k - 1][0] + " " + part[k - 1][1]) + (2.0 + _words(head_c) / W_WORDS_PER_S if k == 1 and first == 1 else 0)
            beats.append((round(secs + (2.0 if k == len(part) else 0), 1), "", draw))
        offs = [0.0]
        for bt in beats[:-1]:
            offs.append(offs[-1] + bt[0])
        S.render(name, beats, footer=foot_c,
                 script=([(0.0, head_c)] if first == 1 else []) + [(o, f"{first + i}. {a} {b}") for i, (o, (a, b)) in enumerate(zip(offs, part))])

    # --- how to read the movies
    def legend(fig):
        y = _headline(fig, "How to read the movies", "The fleet movies show the simulated tugs, payload and cables; the others chart the campaign records.")
        items = [(C.TAUT, "-", 6, "A solid blue line is a taut cable; the thicker the line, the higher its tension, up to 12 kN."),
                 (C.SLACK, (0, (3.0, 2.2)), 2.5, "A dashed red line is a cable that carries no tension. A cable is slack when the\n"
                  "distance between its ends is at or below its 12 m rest length."),
                 (C.SEVERED, "-", 2.5, "Two grey stubs and a red cross mark a severed cable. A cable counts as severed at 4.5 kN, a\n"
                  "threshold chosen so that about half the missions reach it, not a real line's breaking strength.\n"
                  "Live runs cut the cable there; recording runs only score it.")]
        for k, (col, ls, lw, text) in enumerate(items):
            yy = y - 0.02 - k * 0.105
            if col == C.SEVERED:      # as drawn by common.draw_fleet: two grey stubs and a red cross
                fig.add_artist(Line2D([0.07, 0.09], [yy, yy], color=col, lw=lw))
                fig.add_artist(Line2D([0.13, 0.15], [yy, yy], color=col, lw=lw))
                fig.add_artist(Line2D([0.11], [yy], color=C.SLACK, marker="x", ms=14, mew=3, ls="none"))
            else:
                fig.add_artist(Line2D([0.07, 0.15], [yy, yy], color=col, ls=ls, lw=lw))
            fig.text(0.17, yy, text, fontsize=C.FS_BODY + 1, color=C.INK, va="center", linespacing=1.3)
        _prose(fig, ["The movies draw on two sets of simulations: squall-passage missions, in which the tugs start in a "
                     "fan at a pretension of 1 kN (used for prediction and mitigation), and runs with the tugs in parallel "
                     "at several pretensions, in random weather or under scripted gusts (used for the mechanism and for design).",
                     "The clock in the top right corner gives the simulation time and the playback speed: x4 is fast "
                     "forward and x1/20 is slow motion. The caption at the bottom narrates the movie, and the grey "
                     "footer names the record or simulation replay behind the picture. The pentagon is the payload, "
                     "the vessels are drawn at the simulated size of 3 m by 1 m, and the scale bar is 5 m."],
               y - 0.39, fs=C.FS_BODY, width=108, gap=0.02)
    S.render("s02b_legend", [(40.0, "", legend)], script=[(0.0, "How to read the movies. A solid blue line is a taut cable, "
             "thicker for higher tension; a dashed red line carries no tension; two grey stubs and a red cross mark a severed "
             "cable, counted at 4.5 kN, a threshold chosen so that about half the missions reach it. The movies use two sets "
             "of simulations: squall-passage missions (fan, 1 kN) for prediction and mitigation, and parallel-formation runs "
             "for the mechanism and design.")])

    # --- today's practice
    def perline(fig):
        y = _headline(fig, "How snaps are treated today", "Today, a snap's peak tension is estimated from the line's own closing speed.")
        fig.text(0.5, y - 0.02, r"$T_{\mathrm{peak}} \;\simeq\; T_0 + Z\,V_{\uparrow}$", fontsize=40, ha="center",
                 va="top", color=C.INK)
        return y - 0.13
    per_paras = [
        "In this relation, T0 is the line's pretension; V↑ is the closing speed, how fast the slack closes, that is, "
        "how fast the distance between the line's two ends grows back to its rest length at the instant the line comes "
        "taut; and Z is the line's impedance. On our simulated system the relation is linear, with Z = 7.99 kN·s/m for "
        "one tug and the payload.",
        "Slack criteria, monitoring thresholds and design formulas are all built on this per-line picture, in which "
        "the motion of the line's far end is simply given.",
        ("The question is whether the closing speed V↑ really belongs to one line when five lines share the payload.", "bold"),
    ]
    man.value("impact impedance Z (pair), Z_fleet", (7.99e3, 8.30e3), "N s/m", "records/v2/phase1/phase1_gate.json tests.P1-T2; Paper Sec. II")
    beats = []
    for k in range(1, len(per_paras) + 1):
        def draw(fig, k=k):
            y = perline(fig)
            _prose(fig, per_paras[:k], y)
        t = per_paras[k - 1] if isinstance(per_paras[k - 1], str) else per_paras[k - 1][0]
        beats.append((round(_pace(t) + (6.0 if k == 1 else 0), 1), "", draw))
    offs = [0.0]
    for bt in beats[:-1]:
        offs.append(offs[-1] + bt[0])
    S.render("s03_perline", beats, footer="Impact law fit: records/v2/phase1/phase1_gate.json (P1-T1, P1-T2)",
             script=[(0.0, "Today, a snap's peak tension is estimated from the line's own closing speed: T_peak ≈ T0 + Z V↑.")]
             + [(o, p if isinstance(p, str) else p[0]) for o, p in zip(offs, per_paras)])

    # --- part 1
    _bridge(S, "p1_mechanism", "Part 1 · The mechanism", "Does a snap on one line affect the others?",
            "When one line snaps taut, the jolt goes into the payload that all the lines share. The next movie shows "
            "one recorded snap in slow motion: within 70 milliseconds, three other lines lose their tension.")
    _bridge(S, "p1b_evidence", "Part 1 · The evidence", "Is it more than a coincidence?",
            f"One event cannot separate cause from coincidence. The next movie counts, over {cs['above'][0]} simulated "
            "configurations, how often a slack event follows a snap on a different line, and compares that count with "
            "what chance alone would give.")
    v3 = v3_numbers(S.man)
    if v3 is None:
        print("v3 slides skipped: records/v3 results or Presentation/manifests/intervention.json absent")
    else:
        _bridge(S, "p1c_intervention", "Part 1 · The intervention", "Does removing the snap remove the follow-on slack?",
                "Counting how often slack follows a snap cannot separate the snap's effect from the weather that all cables "
                "share. The next movie replays one recorded snap twice from the same state, once as recorded and once with "
                "the snapping cable's force removed; the slack onsets that disappear are the snap's causal offspring.")
        t34 = v3["t34"]
        if t34["branch"] == "(a)":
            branch = (f"the spectral radius crosses one near intensity {t34['critical_intensity']:.2f}, where each snap's offspring "
                      "on average replace it, so the cascade turns critical inside the operable range.")
        elif t34["branch"] == "(b)":
            branch = (f"it stays below one up to the highest operable intensity, with a margin of {t34['margin']:.2f} at "
                      f"ρ = {t34['rho']:.2f}, so the cascade is subcritical throughout the operable range.")
        else:
            branch = "its intervals straddle one, so the declared rule calls the transition under-powered rather than observed."
        lo, hi = v3["sweep"][0], v3["sweep"][-1]
        _reveal(S, "s04b_criticality", "The cascade as a branching process",
                "Measured snap transmission, a causal offspring count and a branching kernel turn the cascade into a testable process.", [
            f"The transmission gain was measured this time: the median normalised neighbour swing lies inside the paper's "
            f"band [0.2, 0.9] in {v3['t11'][0]} of {v3['t11'][1]} cells (verdict {v3['t11'][2]}), and the exact linear "
            f"response, which keeps the finite contact time and the tugs' motion, reproduces the pattern across cable pairs "
            f"in {v3['t12'][0]} of {v3['t12'][1]} cells and the swing's size within a factor 1.5 in {v3['t13'][0]} of {v3['t13'][1]}.",
            f"Across {v3['t23'][1]} sampled snaps the factual replay reproduced the record (closing-speed error at the 95th "
            f"percentile {v3['t23'][2]:.3f} m/s; other cables' onset sets reproduced in {v3['t23'][3]:.0%}), so the counterfactual "
            f"counts are trusted (verdict {v3['t23'][0]}); causal and observational offspring counts agree within a factor 1.5 "
            f"in {v3['t24'][0]} of {v3['t24'][1]} cells.",
            f"The branching kernel used is the {v3['kernel_path']} one; along the 0.6 kN sweep the spectral radius ρ runs from "
            f"{lo[1]:.2f} at intensity {lo[0]:g} to {hi[1]:.2f} at {hi[0]:g}, and {branch}",
            f"The predicted extremal index θ = 1/E[cluster size] agrees with the measured one within 0.10 in {v3['t32'][0]} of "
            f"{v3['t32'][1]} cells, and the cluster-corrected rate law ν_total = (I − K)⁻¹ ν_primary predicts the total onset "
            f"rate within a factor 1.5 in {v3['t31'][0]} of {v3['t31'][1]} cells.",
        ], footer="records/v3/wp1_results.json · wp2_results.json · wp3_results.json · Presentation/manifests/intervention.json",
                fs=C.FS_BODY + 2, width=96)

    # --- what the mechanism implies
    _reveal(S, "s04_roadmap", "What the mechanism implies",
            "If snaps couple the lines, three everyday uses of a snap model are affected.", [
        "For prediction, a monitor that watches one line should misjudge that line's danger whenever a neighbour snaps.",
        "For mitigation, thrust laws built on the per-line picture, whether they ease every tug or brake the slack tug, "
        "may fail to prevent the snap.",
        "For design, a per-line formula for how often lines break may have no range in which it can be used.",
        ("Each claim is tested on matched simulations: in the prediction test both models start from the same exact state, "
         "but only the fleet model is told the forcing; the mitigation tests compare runs with and without a law in the same "
         "weather; the design tests sweep gust strength and pretension.", "bold"),
    ])

    # --- part 2: prediction
    _bridge(S, "p2_prediction", "Part 2 · Prediction", "Can one line's danger be forecast from that line alone?",
            "Every 0.1 s (one tick), two models forecast whether a slack line will snap dangerously within the next second, "
            f"that is, close faster than its severing speed v_b, about {n['vb'][0]:.2f} to {n['vb'][1]:.2f} m/s, at which the snap "
            "would reach 4.5 kN. The per-line model sees only the watched line; the fleet model simulates the whole fleet. Both "
            "start from the exact simulated state, and the fleet model is also given the known forcing: the squall, the tugs' "
            "thrust schedules, the current weather and the statistical law of the random weather.")
    tl, tr = n["perline_top"], n["rollout_top"]
    _reveal(S, "s05a_prediction_result", "Prediction: the result",
            "The per-line model's confident false alarms come when another line snaps; the fleet model made none.", [
        f"Over 40 missions and {n['perline_ticks']:,} forecasts, the per-line model predicted a snap with near certainty "
        f"(a probability of 0.9 or more) {tl[1]} times. {tl[0]} of those were false alarms, and {tl[2]} of the {tl[0]} came "
        "while another line was re-engaging within the one-second forecast horizon.",
        f"The fleet model, which was also given the known forcing, made no such error: none of its {tr[1]} near-certain "
        "forecasts was wrong.",
        f"Both models rank dangerous moments well, but the per-line model's probabilities are structurally too extreme: "
        f"its recalibration slope is {n['perline_slope'][0]:.3f}, where a calibrated forecast would score 1.",
        ("Its confident false alarms point to the other lines.", "bold"),
    ], footer=f"{R_P5}")

    e, cw, fm, c1 = n["est_slope"], n["certain_wrong"], n["fa_mid"], n["clip01"]
    b = n["rollout_slope"]
    _reveal(S, "s05_prediction", "Prediction: what it does and does not show",
            "Five qualifications limit what this comparison shows.", [
        "Both models had the exact state, so estimation error plays no part; that does not make either score a best "
        f"case: fed by the tugs' own state estimator, the per-line model scored a better but still failing slope, {e[0]:.2f}, "
        f"where its exact-state slope was {n['exact_slope_v1']:.2f} (under an earlier definition of the event).",
        "The fleet model also had the known forcing: the current weather, the squall, the schedules and the background "
        f"weather's true law. Still, {fm[0]} of the per-line model's {fm[1]} coupled false alarms came at 70 to 110 "
        "seconds, when only the background weather acts, although the fleet model knew that weather's current value and "
        "its law (post hoc).",
        f"The fleet model fails its own slope test ({b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}]) because nearly all its forecasts "
        f"are exactly 0 or 1 ({n['degenerate']} of {n['rollout_ticks']}), so it is not called calibrated.",
        f"After the fact, the per-line slope turns out to depend on how forecasts near 0 and 1 are clipped ({c1[0]:.2f} "
        f"[{c1[1]:.2f}, {c1[2]:.2f}] at a clip of 0.01); its failures in the most confident bins and its {cw} certain-and-"
        "wrong forecasts do not.",
        "The missions were recorded in the first campaign's exploratory continuation, after its protocol had stopped, "
        "and this test committed no prediction in advance.",
    ], footer=f"{R_P5}  ·  records/phase5/phase5_results.json  ·  p5_t2prime_ticks.npz", fs=C.FS_BODY + 2, width=96)

    # --- part 3: mitigation
    _bridge(S, "p3_mitigation", "Part 3 · Mitigation", "Can thrust control prevent the snaps?",
            "Two natural control laws try to prevent snaps by adjusting thrust. The first eases the thrust of every tug "
            "when a hazard is detected anywhere in the fleet. The next movie compares runs with and without it on the "
            "same weather.")
    _bridge(S, "p3b_catch", "Part 3 · The second law", "Can the slack tug catch its line gently?",
            "The second law makes the slack tug brake so that its line tightens at a low closing speed. In each of "
            f"{n['catch']['hold'][1]} squall missions we replay the first slack excursion whose snap would have broken the "
            "line, with and without the law, in two variants: one may cut the tug's thrust to zero, the other may also "
            "reverse it. The law is given the true state, so the result measures the law itself.")
    s_, d_, ch, fl = n["sev"], n["sev_NP"], n["catch"], n["floor"]
    _reveal(S, "s06_mitigation", "Mitigation: the result", "Neither thrust law reduced how often lines broke.", [
        f"Easing the fleet's thrust left severance where it was: {ep['N_sev']} of {ep['n']} missions lost a line without "
        f"the supervisor and {ep['P_sev']} of {ep['n']} with it (paired difference {d_[0]:.2f}, 95 % interval "
        f"{d_[1]:.3f} to {d_[2]:.3f}).",
        f"The paired missions disagreed in only {ep['N_only'] + ep['P_only']} seeds. In the {ep['P_only']} seeds where only "
        "the eased run lost a line, the unsupervised run finished intact; in the "
        f"{ep['N_only']} seeds where only the unsupervised run lost a line, the eased run had already ended in a "
        "formation closure, which the protocol scores as no break.",
        f"Fed by the proposed estimator, easing's only statistically resolved effects were costs: {n['dock'][0]:+.2f} m of "
        f"docking error and +0.2 % of mission time. Fed by a local monitor without fusion, the same supervisor raised "
        f"severance to {ep['L_sev']} of {ep['n']} missions.",
        f"The catch, even when given the true state, avoided a break in only {ch['hold'][0]} of the {ch['hold'][1]} "
        "excursions it was tested on, far short of the 0.80 success rate it had to reach.",
        f"Even a perfect catch could address at most five sixths of first breaks: of the {fl[0] + fl[1]} squall missions "
        f"that lost a line, {fl[1]} lost it to the overload of a line that was already taut, not to a snap.",
    ], footer=f"{R_P6}  ·  {R_T0}  ·  records/phase6/cache/phase6_compute.pkl")

    # --- part 4: design
    _bridge(S, "p4_design", "Part 4 · Design time", "Can a formula tell designers how often lines will break?",
            "Designers choose pretension and line strength from a formula for how often lines break. The next movie "
            "tests one of its ingredients: the rule for when a gust makes a line go slack and stay slack.")
    _bridge(S, "p4b_pretension", "Part 4 · The pretension trade",
            "Can one pretension both hold the formation and produce the events a formula needs?",
            "Raising the pretension, together with the heading gain, steadies the fleet's shape, but it also suppresses "
            "the slack events that a rate formula must be fitted to. The next movie runs the same weather at two pretensions.")
    t6, tr_ = n["t6"], n["trade"]
    _reveal(S, "s07_design", "Design time: the result",
            "The new slack rule survives, but the design formula has no usable range on the grid.", [
        f"The drag-based slack criterion holds: the measured threshold lands at {t6[0]:.3f} times the predicted gust "
        "strength, a 1.1 % error. It involves one line only, so per-line practice can adopt it unchanged; the impact "
        "law also passes its tests.",
        "The formula also needs a law for how deep a slack excursion runs, and its closed form does not hold: for gusts "
        "that last longer than a tug takes to reach its drift speed, the simulated excursions near the slack threshold "
        "run 24 to 28 percent deeper than it predicts, and 8 of 12 test cells miss its 15 percent band.",
        f"Raising the pretension from 0.6 to 1.0 kN, with the heading gain raised by schedule, narrows the spread of the "
        f"cable angles from {tr_[0]:.1f}° to {tr_[1]:.1f}°, but cuts the re-engagement marks from {tr_[2]} to {tr_[3]}. None "
        "of the 8 configurations holds the fleet's shape within 15°, and those nearest the band produce too few events to "
        f"fit a formula; the closest to meeting both misses by a factor of {n['best_gap']:.2f}.",
    ], footer=f"{R_GATE}  ·  {R_STOCH}")

    # --- without vs with
    _reveal(S, "s08_versus", "Without and with the fleet view",
            "A fleet model, given the state and the forcing, avoids the per-line model's confident errors; neither tested "
            "thrust law reduced breaks.", [
        "Without the fleet view, each line is judged alone. With it, a snap on one line is expected to disturb the "
        f"others, and the simulations show the consequence: in all {cs['above'][0]} scored configurations, slack events "
        "follow other lines' snaps more often than a chance baseline computed after the fact, by "
        f"{cs['factor'][0]:.1f} to {cs['factor'][1]:.0f} times.",
        "For prediction, modelling the whole fleet removes the per-line model's confident errors in simulation, "
        "provided the fleet model knows the state and the forcing; a deployable fleet monitor has not yet been built "
        "or tested.",
        "For mitigation, the two thrust laws we tested, easing the whole fleet and catching with the slack tug, did "
        "not reduce breaks; other laws, such as thrust coordinated for the purpose, were not tested.",
        "For design, the drag-based slack criterion survives, but no setting on the tested grid both holds the formation "
        "and produces the events that a formula needs.",
        ("Towline severance in a fleet is a fleet problem: a model that judges one line at a time is confidently wrong "
         "almost only when another line snaps, and monitoring should model the fleet.", "bold"),
    ], fs=C.FS_BODY + 3, width=92)

    # --- limits
    bu = n["bunch"]
    _reveal(S, "s09_limits", "Limits", "These results come with seven limits.", [
        "They come from one planar simulated testbed with one set of parameters, so the numbers belong to that testbed.",
        "The transmission gain of 0.44 is derived, not measured; what was measured is its consequence.",
        "The attribution analysis and the mitigation decompositions were done after the fact (post hoc).",
        "The fleet forecast had the exact state and the known forcing; a fleet monitor fed by a real estimator was "
        "planned but not run.",
        "The easing campaign was run after the first campaign's protocol had stopped, as an exploratory continuation, "
        "and why the catch fails is not established.",
        "A line counts as broken when its tension reaches 4.5 kN, a stress threshold set at the 55th percentile of the "
        "missions' peak tensions so that about half the missions sever; it is not a rated breaking strength.",
        f"The simulation models no contact between bodies: in {bu['with_pair_within_1m']} of the {bu['missions']} squall "
        f"missions behind the prediction and catch results, the centres of two tugs come within 1 m of each other, less "
        f"than a hull's width, for a median {bu['median_fraction']:.0%} of the mission, so their hulls overlap (post hoc).",
    ], fs=C.FS_BODY + 2, width=96)

    # --- end
    def end(fig):
        fig.clf()
        fig.text(0.5, 0.64, "Cable Severance in Cooperative Towing is a Fleet Problem", fontsize=34, weight="bold",
                 color=C.INK, ha="center")
        fig.text(0.5, 0.53, textwrap.fill(
            "The paper, the campaign records and the source of every number in this presentation are in the project "
            "repository: Paper/, records/ and Presentation/manifests/.", 90),
            fontsize=C.FS_BODY + 2, color=C.INK, ha="center", va="top", linespacing=1.4)
        fig.text(0.5, 0.36, "Simulated with a planar Drake multibody plant at a 0.5 ms step; the catch counterfactual uses "
                 "the campaign's own validated integrator.", fontsize=C.FS_BODY, color=C.TAUT, ha="center")
    S.render("s10_end", [(10.0, "", end)], script=[(0.0, "End. The paper, the campaign records and the source of every number "
             "are in the repository: Paper/, records/ and Presentation/manifests/.")])

    man.selection = "not applicable (slides carry record-level numbers only)"
    man.caveats = ["slide text paraphrases the corrected paper (Paper/Sections/Section_*.tex); no claim is stronger"]
    (C.MANIFESTS / "slides_segments.json").write_text(json.dumps(S.segments, indent=1))
    first = C.CLIPS / "slides" / "s01_title.mp4"
    man.write(first)
    return S.segments


if __name__ == "__main__":
    build()
