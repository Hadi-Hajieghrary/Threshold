"""Phase 3 report, generated from the committed shock table, predictions and results."""

from __future__ import annotations

import json

from tether.analysis.report_common import EXPLORATORY, artifacts, discussion, execution_table, fmt, table
from tether.campaign.common import RECORDS, REPORTS, write_bytes
from tether.campaign.phase3 import PREDICTIONS_PATH, RESULTS_PATH

REPORT = REPORTS / "phase3_report.md"


def build() -> str:
    results = json.loads(RESULTS_PATH.read_text())
    predictions = json.loads(PREDICTIONS_PATH.read_text())
    tests = results["tests"]
    lines = ["# Phase 3 Report - Tail class and the index doubling", "", "## Gate verdict", "",
             f"**{results['gate']['decision']}.** Rule: {results['gate']['rule']}.", "", EXPLORATORY, ""]
    lines += ["## Declarations (fixed before the campaign)", ""]
    lines += [f"- **{k}**: {v}" for k, v in results["declarations"].items()]
    lines += ["", f"Predictions: `records/phase3/phase3_predictions.json` (sha256 `{results['predictions_sha256'][:16]}...`), computed from the deterministic shock cell before any stochastic Phase 3 run. Physical spectral weights per body: {', '.join(f'{k}: {v:.4f}' for k, v in predictions['weights'].items())} (plan: equal mass).", ""]
    rows = []
    t1 = tests["P3-T1"]
    rows.append(["P3-T1", "Hill alpha of W_rel per cable (t3, local): " + ", ".join(fmt(h["alpha"], 2) for h in t1["hill_per_cable_local_linear"]), "[2.4, 3.6]", t1["verdict"]])
    t2 = tests["P3-T2"]
    rows.append(["P3-T2", f"taut GPD shape (t3, linear) {fmt(t2['linear'].get('shape'))} [{fmt(t2['linear'].get('shape_lo'))}, {fmt(t2['linear'].get('shape_hi'))}]; quadratic {fmt(t2['quadratic'].get('shape'))}; Gaussian reference {fmt(t2['gaussian_linear_reference'].get('shape'))}", "[0.25, 0.42]", t2["verdict"]])
    t3 = tests["P3-T3"]
    rows.append(["P3-T3", "; ".join(f"{r['drag']}: 2alpha = {fmt(r['snap_tail'].get('index'))} [{fmt(r['snap_tail'].get('index_lo'))}, {fmt(r['snap_tail'].get('index_hi'))}], taut alpha {fmt(r['taut_index'])}" for r in t3["rows"]), "[4.5, 7.5], excluding alpha", t3["verdict"]])
    t4 = tests["P3-T4"]
    rows.append(["P3-T4", "; ".join(f"{r['drag']}: {r['slackening_pairs']} slackening pairs, consistent mass fraction {fmt(r['mass_fraction_consistent'])} (equal-mass {fmt(r['equal_mass_fraction_consistent'])}), median beta {fmt(r['beta_median'])}" for r in t4["rows"]), "beta in [0.4, 0.6] for >= 80% of mass", t4["verdict"]])
    t5 = tests["P3-T5"]
    rows.append(["P3-T5", f"measured crossover {fmt(t5['measured'])} N (powered {fmt(t5['measured_powered'])}), predicted {fmt(t5['predicted'])} N", "exists; within 30%", t5["verdict"]])
    t6 = tests["P3-T6"]
    rows.append(["P3-T6", f"{len(t6['rows'])} powered thresholds; ratios " + ", ".join(fmt(r["ratio"], 2) for r in t6["rows"][:8]), "[1/2, 2]", t6["verdict"]])
    t7 = tests["P3-T7"]
    rows.append(["P3-T7", f"exponent intervals overlap {fmt(t7['exponents_overlap'])}; snap rate at 8 kN linear {fmt(t7['linear_rate_at_8kN'])} vs quadratic {fmt(t7['quadratic_rate_at_8kN'])} 1/s", "overlap; coefficients differ", t7["verdict"]])
    rows.append(["P3-T8", tests["P3-T8"]["reason"], "reported", tests["P3-T8"]["verdict"]])
    t9 = tests["P3-T9"]
    rows.append(["P3-T9", "contamination " + ", ".join(f"{c['threshold'] / 1000:.0f} kN: {fmt(c['contamination'], 2)}" for c in t9["contamination"]), "reported; Phases 4-6 use < 0.2", t9["verdict"]])
    lines += ["## Acceptance results", ""] + table(["Test", "Measurement", "Criterion", "Verdict"], rows) + [""]
    cell_rows = []
    for name, row in sorted(results["rows"].items()):
        cell_rows.append([name, row["excursions"], row["closures"], fmt(row["snap_tail"].get("index")), fmt(row["taut_tail"].get("index")), fmt(row.get("snap_rate_log_slope")), fmt(row["crossover"])])
    lines += ["### Cells", ""] + table(["Cell", "Excursions", "Closures", "snap index (GPD)", "taut index (GPD)", "log-log snap slope", "crossover [N]"], cell_rows) + [""]
    lines += execution_table(results["execution"], "3003-3022 (pilots 3001-3002), 600 s")
    lines += artifacts(["records/phase3/phase3_shock_response.json", "records/phase3/phase3_predictions.json", "records/phase3/phase3_results.json",
                        "records/phase3/phase3_records.npz", "records/phase3/phase3_manifest.json"])
    lines += discussion(RECORDS / "phase3" / "phase3_discussion.md")
    return "\n".join(lines)


def main() -> None:
    write_bytes(REPORT, build().encode("utf-8"))
    print(REPORT)


if __name__ == "__main__":
    main()
