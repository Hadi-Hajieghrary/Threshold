"""The cascade explorer: a self-contained HTML page over the plan v3 records.

    python -m tether.analysis.v3.explorer            # records/v3/export/cascade_explorer.json + reports/v3/cascade_explorer.html

``export_json`` gathers every number the page shows from the records (with the sha256 of each
source file) into one JSON document; ``build_html`` inlines that document into a page with no
external fetches (inline JSON, vanilla JavaScript, inline SVG), so it also publishes unchanged as
an Artifact.  The page never computes a statistic that the records do not already hold, except the
plain geometric interpolation of the critical intensity that the WP3 analysis also performs.

Panels: the transmission matrices (measured / exact response / closed form) per cell; the band
statistic per cell against the paper's band; the kernel K per cell with rho and theta; the
(T0, intensity) map of rho with the sweep and the rho = 1 level; the cluster-size distributions
with the Borel-Tanner overlay; the verdict board with each test's declared threshold, committed
forecast and measured verdict; a provenance footer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tether.campaign.common import ROOT, json_bytes, sha256_file, source_state, write_bytes
from tether.campaign.v3 import campaign, cells

EXPORT_DIR = campaign.RECORD_DIR / "export"
EXPORT_PATH = EXPORT_DIR / "cascade_explorer.json"
HTML_PATH = ROOT / "reports" / "v3" / "cascade_explorer.html"


def _read(name: str):
    path = campaign.PREDICTIONS_PATH if name == "cascade_predictions.json" else campaign.RECORD_DIR / name
    if not path.exists():
        return None, None
    return json.loads(path.read_text()), sha256_file(path)


def _matrix(entries):
    return [[None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v) for v in row] for row in entries]


def _measured_matrix(cell_wp1: dict):
    out = [[None] * 5 for _ in range(5)]
    t12 = (cell_wp1 or {}).get("T1.2", {})
    for (i, j), value in zip(t12.get("pairs", []), t12.get("median_drop_over_Tpeak", [])):
        out[i][j] = out[4 - i][4 - j] = float(value)
    return out


def export_json() -> str:
    pred, pred_sha = _read("cascade_predictions.json")
    decl, decl_sha = _read("cascade_declarations.json") if (campaign.RECORD_DIR / "cascade_declarations.json").exists() else (None, None)
    wp1, wp1_sha = _read("wp1_results.json")
    wp2, wp2_sha = _read("wp2_results.json")
    wp3, wp3_sha = _read("wp3_results.json")
    if pred is None:
        raise SystemExit("cascade_predictions.json is missing")
    names = sorted(pred["cells"], key=lambda n: (cells.parse_cell(n)["formation"] == "fan", cells.parse_cell(n)["stiffness_factor"] != 1.0,
                                                  cells.parse_cell(n)["pretension"], cells.parse_cell(n)["intensity"]))
    cell_rows = []
    for name in names:
        p = cells.parse_cell(name)
        tr = pred["cells"][name]["transmission"]
        row = {
            "cell": name, "stage": pred["cells"][name]["stage"], **p,
            "forecast": {
                "band_statistic": tr["band_statistic_median_offdiag"], "response_M1": _matrix(tr["response_M1_half_sine"]),
                "operator": _matrix(tr["operator"]), "geometry_factor": _matrix(tr["geometry_factor"]),
                "rho": pred["cells"][name]["branching"].get("rho"), "theta_pred": pred["cells"][name]["branching"].get("theta_pred"),
                "m_bar": pred["cells"][name]["branching"].get("m_bar"), "extrapolated": pred["cells"][name]["branching"].get("extrapolated"),
            },
        }
        if wp1 and name in wp1["cells"]:
            c = wp1["cells"][name]
            row["wp1"] = {"scored": c["scored"], "n_eligible_pairs": c["n_eligible_pairs"], "T1.1": c["T1.1"], "T1.3": c["T1.3"],
                          "spearman_vs_R": c["T1.2"]["spearman_vs_R"], "spearman_vs_operator": c["T1.2"]["spearman_vs_operator"],
                          "measured_matrix": _measured_matrix(c), "sensitivities": c["sensitivities"]}
        if wp2 and name in wp2["cells"]:
            c = wp2["cells"][name]
            row["wp2"] = {"scored": c["scored"], "measured_cross_offspring": c["measured_cross_offspring"], "n_events": c["n_events_statistics"],
                          "T2.1": {k: c["T2.1"][k] for k in ("means", "counts", "spearman", "verdict")}, "T2.2": c["T2.2"], "T2.5": c["T2.5"],
                          "kernels": {k: {"ratio": v["ratio"], "rho": v["rho"], "K": v["K"]} for k, v in c["kernels"].items()},
                          "intervention": c["intervention"]}
        if wp3 and name in wp3["cells"] and "rho" in wp3["cells"][name]:
            c = wp3["cells"][name]
            row["wp3"] = {k: c.get(k) for k in ("rho", "rho_ci95", "theta_pred", "theta_runs", "theta_runs_ci95", "theta_ferro_segers", "m_bar",
                                                 "rate_ratio_pooled", "operable", "closures", "runs", "n_events", "K", "pi", "cluster_pmf",
                                                 "borel_tanner", "verdicts", "nu_total_measured_per_cable_s", "nu_total_predicted_per_cable_s")}
        cell_rows.append(row)
    tests = []
    for tid, spec in campaign.TESTS.items():
        measured = None
        for source in (wp1, wp2, wp3):
            if source and tid in source.get("tests", {}):
                measured = source["tests"][tid]
        tests.append({"id": tid, "blocking": spec["blocking"], "work_package": spec["work_package"], "statement": spec["statement"],
                      "threshold": spec["threshold"], "committed_prediction": spec.get("committed_prediction") or pred["verdict_forecasts"].get(tid),
                      "verdict": None if measured is None else measured.get("verdict", measured.get("branch")),
                      "detail": None if measured is None else {k: v for k, v in measured.items() if k in ("scored_cells", "passing_cells", "delta_measured", "ci95", "delta_exact", "critical_intensity", "margin", "n", "onset_set_reproduced_share")}})
    gates = {}
    for k in (1, 2, 3):
        g, sha = _read(f"gate_wp{k}.json")
        if g:
            gates[f"wp{k}"] = {"verdict": g["verdict"], "failing_blocking_tests": g["failing_blocking_tests"], "sha256": sha}
    payload = {
        "schema": "v3-explorer-export-1",
        "title": "Cascade criticality of cable-towed fleets — plan v3 explorer",
        "closed_form": pred["closed_form"], "stiffness_pair": pred["stiffness_pair"], "sweep_cells": list(cells.sweep_cells()),
        "kernel_path": None if not wp2 else wp2.get("K_path_chosen"),
        "sweep_branch": None if not wp3 else wp3["tests"]["T3.4"],
        "cells": cell_rows, "tests": tests, "gates": gates,
        "sources": {"cascade_predictions.json": pred_sha, "cascade_declarations.json": decl_sha, "wp1_results.json": wp1_sha, "wp2_results.json": wp2_sha, "wp3_results.json": wp3_sha},
        "source": source_state(),
    }
    return write_bytes(EXPORT_PATH, json_bytes(payload))


HTML_TEMPLATE = r"""<title>Cascade Criticality Explorer</title>
<style>
:root{--bg:#f6f4ef;--panel:#ffffff;--ink:#1f2a33;--ink2:#4c5a66;--muted:#8a94a0;--line:#dcd8d0;--cool:#2f6f9f;--warm:#c1443c;--grey:#8a8a8a;--band:#cdd9e5;--accent:#d98c00;--ok:#2e7d5b;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15191d;--panel:#1e2429;--ink:#e8e4dc;--ink2:#b8b2a8;--muted:#7c8690;--line:#333a41;--band:#2c3b48;}}
:root[data-theme="dark"]{--bg:#15191d;--panel:#1e2429;--ink:#e8e4dc;--ink2:#b8b2a8;--muted:#7c8690;--line:#333a41;--band:#2c3b48;}
body{background:var(--bg);color:var(--ink);font:14px/1.5 "IBM Plex Sans","Segoe UI",system-ui,sans-serif;padding-block:24px;padding-inline:clamp(16px,4vw,48px);}
h1{font-family:"IBM Plex Serif",Georgia,serif;font-weight:600;font-size:clamp(22px,3vw,30px);margin:0 0 4px;text-wrap:balance}
h2{font-size:15px;letter-spacing:.04em;text-transform:uppercase;color:var(--ink2);margin:28px 0 10px}
p.lead{max-width:68ch;color:var(--ink2);margin:0 0 18px}
.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:14px 16px;min-width:0}
.panel h3{margin:0 0 6px;font-size:14px;font-weight:600}
.panel .note{color:var(--muted);font-size:12px;margin:0 0 8px}
svg{max-width:100%;height:auto;display:block}
svg text{fill:var(--ink2);font-size:11px}
.axis{stroke:var(--line);stroke-width:1}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:0 0 10px}
select,button{font:inherit;background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:4px;padding:4px 8px}
button:focus-visible,select:focus-visible{outline:2px solid var(--cool);outline-offset:2px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:12.5px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--ink2);font-weight:600}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11.5px;font-weight:600;border:1px solid var(--line)}
.PASS{background:color-mix(in oklab,var(--ok) 18%,transparent);color:var(--ok)}
.FAIL{background:color-mix(in oklab,var(--warm) 18%,transparent);color:var(--warm)}
.none{color:var(--muted)}
.wrap{overflow-x:auto}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--ink2)}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:4px}
footer{margin-top:28px;color:var(--muted);font-size:12px}
footer code{font-size:11px}
@media (prefers-reduced-motion: reduce){*{transition:none!important}}
</style>
<h1>Cascade Criticality Explorer</h1>
<p class="lead" id="lead"></p>
<div class="row"><label for="cell">Cell</label><select id="cell"></select><span id="cellmeta" class="note"></span></div>
<div class="grid">
  <div class="panel"><h3>Transmission matrices</h3><p class="note">Neighbour <em>i</em> (rows) per unit peak on cable <em>j</em> (columns). Hover a tile.</p><div id="matrices"></div></div>
  <div class="panel"><h3>The paper's band, cell by cell</h3><p class="note">Median normalised neighbour drop with its 95 % seed interval (filled), the exact-response forecast (open), the closed form 0.44 (amber) and the declared band [0.2, 0.9].</p><div id="bandchart"></div></div>
  <div class="panel"><h3>Kernel and criticality</h3><p class="note">The chosen kernel K (rows = children), its spectral radius ρ and the extremal index θ, forecast beside measured.</p><div id="kernel"></div></div>
  <div class="panel"><h3>ρ across the grid</h3><p class="note">Spectral radius by weather intensity; the 0.6 kN sweep joined with its interval; ρ = 1 is the critical level.</p><div id="rhomap"></div></div>
  <div class="panel"><h3>Cluster sizes</h3><p class="note">Events per cross-cable tree in the selected cell (filled) against Borel–Tanner at m̄ (line).</p><div id="clusters"></div></div>
</div>
<h2>Verdict board</h2>
<div class="panel wrap"><table id="tests"><thead><tr><th>test</th><th>blocking</th><th>statement</th><th>threshold</th><th>committed forecast</th><th>verdict</th></tr></thead><tbody></tbody></table></div>
<footer id="prov"></footer>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : (Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : Number(v).toFixed(d));
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const cellSel = document.getElementById('cell');
D.cells.forEach(c => { const o = document.createElement('option'); o.value = c.cell; o.textContent = c.cell; cellSel.appendChild(o); });
document.getElementById('lead').textContent = `${D.cells.length} declared cells; kernel path ${D.kernel_path || 'not yet chosen'}; gates ${Object.entries(D.gates).map(([k, g]) => k + ' ' + g.verdict).join(', ') || 'none written yet'}. Every number is read from records/v3; forecasts are labelled.`;
function heat(m, title, vmax, x0) {
  let s = `<text x="${x0}" y="12" font-weight="600">${title}</text>`;
  for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
    const v = m && m[i] ? m[i][j] : null;
    const t = v === null || v === undefined ? 0 : Math.min(1, v / vmax);
    const fill = v === null || v === undefined ? 'var(--line)' : `color-mix(in oklab, var(--cool) ${Math.round(15 + 85 * t)}%, var(--panel))`;
    s += `<rect x="${x0 + j * 24}" y="${18 + i * 24}" width="22" height="22" rx="3" fill="${i === j ? 'var(--line)' : fill}"><title>i=${i} ← j=${j}: ${i === j ? 'self' : fmt(v)}</title></rect>`;
    if (i !== j && v !== null && v !== undefined) s += `<text x="${x0 + j * 24 + 11}" y="${18 + i * 24 + 15}" text-anchor="middle" font-size="9" fill="${t > 0.55 ? '#fff' : 'var(--ink)'}">${v.toFixed(2)}</text>`;
  }
  return s;
}
function renderCell(name) {
  const c = D.cells.find(x => x.cell === name);
  document.getElementById('cellmeta').textContent = `${c.formation}, T0 ${c.pretension} N, k_h ${c.heading_gain}, intensity ${c.intensity}, k×${c.stiffness_factor}, stage ${c.stage}`;
  const vmax = Math.max(0.05, ...[].concat(...(c.wp1 ? c.wp1.measured_matrix : [[0]]), ...c.forecast.response_M1).filter(v => v !== null));
  document.getElementById('matrices').innerHTML = `<svg viewBox="0 0 400 150" role="img" aria-label="transmission matrices">${heat(c.wp1 ? c.wp1.measured_matrix : null, c.wp1 ? `measured (ρ_S vs R ${fmt(c.wp1.spearman_vs_R, 2)})` : 'measured: not yet run', vmax, 4)}${heat(c.forecast.response_M1, 'exact response R (forecast)', vmax, 138)}${heat(c.forecast.operator, 'closed-form operator', Math.max(...[].concat(...c.forecast.operator)), 272)}</svg>`;
  const k = (c.wp3 && c.wp3.K) || (c.wp2 && c.wp2.kernels[Object.keys(c.wp2.kernels)[0]] && c.wp2.kernels[Object.keys(c.wp2.kernels)[0]].K) || null;
  const rows = [['ρ forecast', fmt(c.forecast.rho)], ['ρ measured', c.wp3 ? `${fmt(c.wp3.rho)} [${fmt(c.wp3.rho_ci95[0])}, ${fmt(c.wp3.rho_ci95[1])}]` : '—'], ['θ forecast', fmt(c.forecast.theta_pred)], ['θ_pred / θ_runs', c.wp3 ? `${fmt(c.wp3.theta_pred)} / ${fmt(c.wp3.theta_runs)}` : '—'], ['m̄ forecast / measured', `${fmt(c.forecast.m_bar)} / ${c.wp3 ? fmt(c.wp3.m_bar) : '—'}`], ['rate law ratio', c.wp3 ? fmt(c.wp3.rate_ratio_pooled) : '—'], ['operable', c.wp3 ? `${c.wp3.operable} (${c.wp3.closures}/${c.wp3.runs} closures)` : '—']];
  document.getElementById('kernel').innerHTML = `<svg viewBox="0 0 140 150" role="img" aria-label="kernel K">${heat(k, k ? 'K (measured inputs)' : 'K: not yet run', k ? Math.max(0.05, ...[].concat(...k)) : 1, 4)}</svg><table>${rows.map(r => `<tr><th>${r[0]}</th><td>${r[1]}</td></tr>`).join('')}</table>`;
  const pmf = c.wp3 && c.wp3.cluster_pmf ? c.wp3.cluster_pmf : null;
  if (pmf) {
    const sizes = Object.keys(pmf).map(Number).sort((a, b) => a - b), maxn = Math.max(...sizes, 8);
    const m = c.wp3.borel_tanner ? c.wp3.borel_tanner.m_bar : null;
    const bt = n => m === null ? null : Math.exp(-m * n + (n - 1) * Math.log(m * n) - lgamma(n + 1));
    const X = n => 30 + (n - 1) / (maxn - 1) * 250, Y = p => 120 - Math.max(0, (Math.log10(Math.max(p, 1e-3)) + 3) / 3) * 100;
    let s = `<line class="axis" x1="30" y1="120" x2="290" y2="120"/><line class="axis" x1="30" y1="20" x2="30" y2="120"/>`;
    if (m !== null) { const pts = []; for (let n = 1; n <= maxn; n++) pts.push(`${X(n)},${Y(bt(n))}`); s += `<polyline points="${pts.join(' ')}" fill="none" stroke="var(--warm)" stroke-width="2" stroke-linejoin="round"/>`; }
    sizes.forEach(n => { s += `<circle cx="${X(n)}" cy="${Y(pmf[n])}" r="4.5" fill="var(--cool)" stroke="var(--panel)" stroke-width="2"><title>size ${n}: ${fmt(pmf[n])}</title></circle>`; });
    s += `<text x="160" y="140" text-anchor="middle">cluster size (events)</text><text x="12" y="70" transform="rotate(-90 12 70)" text-anchor="middle">probability (log)</text>`;
    document.getElementById('clusters').innerHTML = `<svg viewBox="0 0 300 150" role="img" aria-label="cluster sizes">${s}</svg><div class="legend"><span><span class="sw" style="background:var(--cool)"></span>measured</span><span><span class="sw" style="background:var(--warm)"></span>Borel–Tanner m̄ ${fmt(m, 2)}</span></div>`;
  } else document.getElementById('clusters').innerHTML = '<p class="note">WP3 not yet run for this cell.</p>';
}
function lgamma(z) { const g = 7, p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]; if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z); z -= 1; let x = p[0]; for (let i = 1; i < g + 2; i++) x += p[i] / (z + i); const t = z + g + 0.5; return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x); }
function bandChart() {
  const rows = D.cells.filter(c => c.wp1 && c.wp1.scored);
  const n = Math.max(rows.length, 1), W = 40 + n * 22 + 10, X = i => 40 + i * 22 + 11, Y = v => 130 - v * 110;
  let s = `<rect x="34" y="${Y(0.9)}" width="${W - 40}" height="${Y(0.2) - Y(0.9)}" fill="var(--band)" opacity=".7"/><line x1="34" x2="${W - 6}" y1="${Y(D.closed_form.ratio)}" y2="${Y(D.closed_form.ratio)}" stroke="var(--accent)" stroke-width="1.5"/><line class="axis" x1="34" y1="130" x2="${W - 6}" y2="130"/><line class="axis" x1="34" y1="20" x2="34" y2="130"/>`;
  [0, 0.2, 0.44, 0.9].forEach(v => { s += `<text x="30" y="${Y(v) + 4}" text-anchor="end" font-size="9">${v}</text>`; });
  rows.forEach((c, i) => { const t = c.wp1['T1.1']; s += `<line x1="${X(i)}" x2="${X(i)}" y1="${Y(t.ci95[0])}" y2="${Y(t.ci95[1])}" stroke="var(--cool)" stroke-width="2"/><circle cx="${X(i)}" cy="${Y(t.median_drop_norm)}" r="4.5" fill="var(--cool)" stroke="var(--panel)" stroke-width="2"><title>${c.cell}: measured ${fmt(t.median_drop_norm)} [${fmt(t.ci95[0])}, ${fmt(t.ci95[1])}]; forecast ${fmt(t.committed_prediction)}; ${t.verdict}</title></circle><circle cx="${X(i)}" cy="${Y(t.committed_prediction)}" r="4" fill="none" stroke="var(--warm)" stroke-width="1.6"/>`; });
  if (!rows.length) s += `<text x="${W / 2}" y="75" text-anchor="middle">WP1 not yet run</text>`;
  document.getElementById('bandchart').innerHTML = `<div class="wrap"><svg viewBox="0 0 ${W} 150" role="img" aria-label="band statistic per cell" style="min-width:${Math.min(W, 640)}px">${s}</svg></div><div class="legend"><span><span class="sw" style="background:var(--cool)"></span>measured</span><span><span class="sw" style="border:2px solid var(--warm)"></span>exact response (forecast)</span><span><span class="sw" style="background:var(--accent)"></span>closed form 0.44</span><span><span class="sw" style="background:var(--band)"></span>declared band</span></div>`;
}
function rhoMap() {
  const rows = D.cells.filter(c => c.wp3 && c.wp3.rho !== null && c.stiffness_factor === 1);
  const W = 320, X = I => 40 + (I - 0.2) / 1.9 * 260, Y = r => 130 - Math.min(r, 1.6) / 1.6 * 110;
  let s = `<line class="axis" x1="34" y1="130" x2="310" y2="130"/><line class="axis" x1="34" y1="20" x2="34" y2="130"/><line x1="34" x2="310" y1="${Y(1)}" y2="${Y(1)}" stroke="var(--ink2)" stroke-width="1"/><text x="308" y="${Y(1) - 4}" text-anchor="end" font-size="9">ρ = 1</text>`;
  [0.5, 1, 1.5, 2].forEach(I => { s += `<text x="${X(I)}" y="144" text-anchor="middle" font-size="9">${I}</text>`; });
  [0, 0.5, 1, 1.5].forEach(r => { s += `<text x="30" y="${Y(r) + 4}" text-anchor="end" font-size="9">${r}</text>`; });
  const sweep = rows.filter(c => D.sweep_cells.includes(c.cell)).sort((a, b) => a.intensity - b.intensity);
  if (sweep.length > 1) s += `<polyline points="${sweep.map(c => `${X(c.intensity)},${Y(c.wp3.rho)}`).join(' ')}" fill="none" stroke="var(--cool)" stroke-width="2"/>`;
  rows.forEach(c => { const sw = D.sweep_cells.includes(c.cell); s += `<line x1="${X(c.intensity)}" x2="${X(c.intensity)}" y1="${Y(c.wp3.rho_ci95[0])}" y2="${Y(c.wp3.rho_ci95[1])}" stroke="${sw ? 'var(--cool)' : 'var(--grey)'}" stroke-width="2"/>`; s += c.formation === 'fan' ? `<rect x="${X(c.intensity) - 4.5}" y="${Y(c.wp3.rho) - 4.5}" width="9" height="9" fill="${c.wp3.operable ? 'var(--grey)' : 'none'}" stroke="var(--grey)" stroke-width="1.5"><title>${c.cell}: ρ ${fmt(c.wp3.rho)}</title></rect>` : `<circle cx="${X(c.intensity)}" cy="${Y(c.wp3.rho)}" r="4.5" fill="${c.wp3.operable ? (sw ? 'var(--cool)' : 'var(--grey)') : 'none'}" stroke="${sw ? 'var(--cool)' : 'var(--grey)'}" stroke-width="1.5"><title>${c.cell}: ρ ${fmt(c.wp3.rho)} [${fmt(c.wp3.rho_ci95[0])}, ${fmt(c.wp3.rho_ci95[1])}]${c.wp3.operable ? '' : ' (not operable)'}</title></circle>`; });
  if (D.sweep_branch && D.sweep_branch.branch === '(a)') { const Ic = D.sweep_branch.critical_intensity; s += `<line x1="${X(Ic)}" x2="${X(Ic)}" y1="20" y2="130" stroke="var(--warm)" stroke-dasharray="4 3"/><text x="${X(Ic) + 4}" y="30" font-size="9">I_c ≈ ${fmt(Ic, 2)}</text>`; }
  if (!rows.length) s += `<text x="170" y="75" text-anchor="middle">WP3 not yet run</text>`;
  s += `<text x="170" y="150" text-anchor="middle" font-size="10">weather intensity</text>`;
  document.getElementById('rhomap').innerHTML = `<svg viewBox="0 0 ${W} 156" role="img" aria-label="rho map">${s}</svg><div class="legend"><span><span class="sw" style="background:var(--cool)"></span>0.6 kN sweep</span><span><span class="sw" style="background:var(--grey)"></span>other cells (square = fan; hollow = not operable)</span>${D.sweep_branch ? `<span>branch ${D.sweep_branch.branch}</span>` : ''}</div>`;
}
function board() {
  const tb = document.querySelector('#tests tbody');
  tb.innerHTML = D.tests.map(t => `<tr><td><b>${t.id}</b></td><td>${t.blocking ? 'yes' : 'no'}</td><td>${t.statement}</td><td>${t.threshold}</td><td>${t.committed_prediction || '—'}</td><td>${t.verdict ? `<span class="pill ${t.verdict.startsWith('PASS') ? 'PASS' : (t.verdict.startsWith('FAIL') ? 'FAIL' : '')}">${t.verdict}</span>` : '<span class="none">not yet run</span>'}${t.detail && t.detail.scored_cells !== undefined ? ` <span class="note">${t.detail.passing_cells}/${t.detail.scored_cells} cells</span>` : ''}</td></tr>`).join('');
}
document.getElementById('prov').innerHTML = `Provenance — ${Object.entries(D.sources).map(([k, v]) => `<code>${k}</code> ${v ? v.slice(0, 12) : 'absent'}`).join(' · ')} · git ${D.source.revision ? D.source.revision.slice(0, 10) : '?'}${D.source.status === 'dirty' ? ' (dirty)' : ''}. Forecasts come from the declared predictions and are never results.`;
cellSel.addEventListener('change', e => renderCell(e.target.value));
renderCell(D.cells[0].cell); bandChart(); rhoMap(); board();
</script>
"""


def build_html() -> str:
    if not EXPORT_PATH.exists():
        export_json()
    payload = EXPORT_PATH.read_text()
    html = HTML_TEMPLATE.replace("__DATA__", payload.replace("</", "<\\/"))
    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(html)
    return str(HTML_PATH)


if __name__ == "__main__":
    print("wrote", EXPORT_PATH, export_json())
    print("wrote", build_html())
