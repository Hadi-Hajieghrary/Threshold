"""Phase 4 report, generated from the committed predictions and results."""

from __future__ import annotations

import json

from tether.analysis.report_common import EXPLORATORY, artifacts, discussion, execution_table, fmt, table
from tether.campaign.common import RECORDS, REPORTS, write_bytes
from tether.campaign.phase4 import RESULTS_PATH

REPORT = REPORTS / "phase4_report.md"


def build() -> str:
    results = json.loads(RESULTS_PATH.read_text())
    tests = results["tests"]
    lines = ["# Phase 4 Report - Fleet-level dependence", "", "## Gate verdict", "",
             f"**{results['gate']['decision']}.** Rule: {results['gate']['rule']}.", "", EXPLORATORY, "", "## Declarations (fixed before the campaign)", ""]
    lines += [f"- **{k}**: {v}" for k, v in results["declarations"].items()] + [""]
    rows = [
        ["P4-T1", f"{tests['P4-T1']['pairs']} Gaussian-cell pairs; inside Gaussian band {fmt(tests['P4-T1']['fraction_inside'])}; eta ok {fmt(tests['P4-T1']['fraction_eta_ok'])}", ">= 80% of pairs", tests["P4-T1"]["verdict"]],
        ["P4-T1b", "block maxima against the Gaussian curve (EMP)", "reported", tests["P4-T1b"]["verdict"]],
        ["P4-T2", "; ".join(f"{r['front']}: {r['pairs_meeting'] or 'none'}" for r in tests["P4-T2"]["rows"]), ">= 1 pair per front", tests["P4-T2"]["verdict"]],
        ["P4-T3", "; ".join(f"{r['front']}: Spearman {fmt(r['spearman'])}" for r in tests["P4-T3"]["rows"]) or "-", "> 0.7", tests["P4-T3"]["verdict"]],
        ["P4-T4", f"Gaussian {tests['P4-T4']['gaussian_verdict']}; common t3 {tests['P4-T4']['common_t3_verdict']}; max-functional " + ", ".join(f"{k}: {fmt(v)}" for k, v in tests["P4-T4"]["max_functional"].items()), "decreasing (Gaussian); at or increasing, within x2 (t3)", tests["P4-T4"]["verdict"]],
        ["P4-T5", "; ".join(f"{k}: windward {fmt(v['windward_snap_fraction'])}, leeward {fmt(v['leeward_snap_fraction'])}" for k, v in tests["P4-T5"]["cells"].items()) or "-", "difference > 0.3", tests["P4-T5"]["verdict"]],
    ]
    pivot = tests.get("P4-pivot")
    if pivot:
        rows.append(["P4-pivot", f"t copula; judged cells {', '.join(pivot['judged_cells']) or 'none'}; reproduces the Gaussian-half T4 trend: {pivot['reproduces_t4_gaussian_trend']}",
                     "decreasing C(u) inside the empirical interval", pivot["verdict"]])
    lines += ["## Acceptance results", ""] + table(["Test", "Measurement", "Criterion", "Verdict"], rows) + [""]
    if pivot:
        prow = []
        for name, c in pivot["cells"].items():
            prow.append([name, c["samples"], fmt(c["nu"]), fmt(c["loglik"] - c["gaussian_loglik"], 1), ", ".join(fmt(x) for x in c["fitted_coincidence"]),
                         ", ".join(f"{fmt(x)} [{fmt(lo)}, {fmt(hi)}]" for x, lo, hi in zip(c["empirical_coincidence"], c["empirical_lo"], c["empirical_hi"])),
                         fmt(c["spearman"]), c["inside"], fmt(c["pairs_inside_fraction"])])
        lines += ["### T1-fail pivot: fitted five-cable t copula (Gaussian cells)", "",
                  f"Levels u = {', '.join(str(u) for u in pivot['cells'][next(iter(pivot['cells']))]['levels'])}; nu grid 2-100 (a value at either end is a bound, not an estimate). "
                  f"Pairwise re-check of the fitted copula against the T1 bands: {fmt(pivot['pairs_inside_fraction'])} of Gaussian-cell pairs inside.", ""]
        lines += table(["Cell", "5-vectors", "nu", "log-lik gain over Gaussian", "fitted C(u)", "empirical C(u) [95%]", "Spearman", "inside", "pairs inside"], prow) + [""]
    pair_rows = []
    for name, row in sorted(results["rows"].items()):
        for key, pair in sorted(row["pairs"].items()):
            if "chi" in pair:
                pair_rows.append([name, key, fmt(pair["rho"]), fmt(pair["chi"][-1]), f"[{fmt(pair['chi_lo'][-1])}, {fmt(pair['chi_hi'][-1])}]", fmt(pair["gauss_chi"][-1]), fmt(pair["chibar"][-1]), fmt(pair["eta"]), fmt(pair["inside"])])
    lines += ["### Pair dependence at u = 0.99", ""] + table(["Cell", "Pair", "rho", "chi", "chi 95%", "Gaussian chi", "chibar", "eta", "inside band"], pair_rows) + [""]
    co_rows = []
    for name, row in sorted(results["rows"].items()):
        for c in row["coincidence"]:
            co_rows.append([name, fmt(c["threshold"] / 1000.0, 2), c["episodes"], c["multi"], fmt(c["fraction"]), f"[{fmt(c['lo'])}, {fmt(c['hi'])}]"])
    lines += ["### Coincidence fraction", ""] + table(["Cell", "T_b [kN]", "episodes", ">= 2 cables", "fraction", "Wilson 95%"], co_rows) + [""]
    lines += execution_table(results["execution"], "4003-4022 (pilots 4001-4002), 600 s")
    lines += artifacts(["records/phase4/phase4_predictions.json", "records/phase4/phase4_results.json", "records/phase4/phase4_records.npz", "records/phase4/phase4_manifest.json"])
    lines += discussion(RECORDS / "phase4" / "phase4_discussion.md")
    return "\n".join(lines)


def main() -> None:
    write_bytes(REPORT, build().encode("utf-8"))
    print(REPORT)


if __name__ == "__main__":
    main()
