"""Phase 5 report, generated from the committed results."""

from __future__ import annotations

import json

from tether.analysis.report_common import EXPLORATORY, artifacts, discussion, fmt, table
from tether.campaign.common import RECORDS, REPORTS, write_bytes

REPORT = REPORTS / "phase5_report.md"


def build() -> str:
    results = json.loads((RECORDS / "phase5" / "phase5_results.json").read_text())
    staleness_path = RECORDS / "phase5" / "phase5_staleness_results.json"
    staleness = json.loads(staleness_path.read_text()) if staleness_path.exists() else None
    tests = results["tests"]
    lines = ["# Phase 5 Report - Estimation, hazard, and staleness", "", "## Gate verdict", "",
             f"**{results['gate']['decision']}.** Rule: {results['gate']['rule']}.", "", EXPLORATORY, "", "## Declarations (fixed before the campaign)", ""]
    lines += [f"- **{k}**: {v}" for k, v in results["declarations"].items()] + [""]
    stress = results["stress"]
    lines += [f"Stress breaking strength T_b^s = **{stress['threshold']:.0f} N** (per-mission virtual first-severance probability {stress['probability']:.3f}); critical closing speeds v_b = {', '.join(fmt(v) for v in results['v_b'])} m/s; arm L precursor growth q_L = {fmt(results['l_growth'])}.", ""]
    attempts = results.get("pivot_attempts") or []
    if attempts:
        lines += ["## P5-T2 pivot sequence (pre-declared)", "",
                  "The plan's in-phase pivot: if the oracle fails calibration under the constant-acceleration model, replace it with the linearised closed-loop "
                  "propagation; if that still fails, shorten H to 1 s. Every attempt is reported; the final one is the one analysed below.", ""]
        def oracle(a, key):
            return fmt(((a.get("calibration") or {}).get("O") or {}).get(key))

        lines += table(["Attempt", "prediction model", "H [s]", "P5-T2", "oracle slope", "oracle AUROC"],
                       [[i + 1, a["model"], fmt(a["horizon"]), a["P5-T2"], oracle(a, "slope"), oracle(a, "auroc")] for i, a in enumerate(attempts)]) + [""]
    lines += [f"Final prediction model: **{results['prediction_model']}**, H = {fmt(results['horizon'])} s; {results['missions']['count']} calibration missions.", ""]
    rows = []
    for arm in ("O", "P", "B2", "L"):
        entry = results["arms"].get(arm, {})
        cal = entry.get("calibration") or {}
        nees = entry.get("nees") or {}
        lead = entry.get("lead_time") or {}
        nees_mean, nees_interval = (fmt(nees.get("mean")), f"[{fmt(nees.get('lo'))}, {fmt(nees.get('hi'))}]") if arm != "O" else ("n/a (truth)", "-")
        rows.append([arm, entry.get("ticks"), entry.get("intervals"), entry.get("events"), entry.get("censored"), nees_mean, nees_interval,
                     fmt(cal.get("slope")), fmt(cal.get("large_intercept")), fmt(cal.get("ece")), fmt(cal.get("auroc")), fmt(cal.get("top_three_ratio")),
                     fmt(entry.get("h_crit")), fmt(lead.get("median"))])
    lines += ["## Arms", ""] + table(["Arm", "slack ticks", "intervals", "events", "censored ticks", "precursor NEES", "NEES 95%", "recal. slope", "in-the-large intercept", "ECE", "AUROC", "top-3 ratio", "h_crit", "median lead [s]"], rows) + [""]
    calibration_rows = []
    for arm in ("O", "P", "B2", "L"):
        cal = (results["arms"].get(arm) or {}).get("calibration") or {}
        if cal:
            calibration_rows.append([arm, f"{fmt(cal.get('slope'))} [{fmt(cal.get('slope_lo'))}, {fmt(cal.get('slope_hi'))}]",
                                     f"{fmt(cal.get('large_intercept'))} [{fmt(cal.get('large_intercept_lo'))}, {fmt(cal.get('large_intercept_hi'))}]",
                                     f"{fmt(cal.get('ece'))} [{fmt(cal.get('ece_lo'))}, {fmt(cal.get('ece_hi'))}]",
                                     f"{fmt(cal.get('auroc'))} [{fmt(cal.get('auroc_lo'))}, {fmt(cal.get('auroc_hi'))}]", cal.get("n_outside"), fmt(cal.get("base_rate")), fmt(cal.get("mean_forecast"))])
    if calibration_rows:
        lines += ["## Calibration with cluster intervals", ""] + table(["Arm", "recal. slope [95%]", "in-the-large intercept [95%]", "ECE [95%]", "AUROC [95%]",
                                                                         "bins outside band", "base rate", "mean forecast"], calibration_rows) + [""]
    lines += ["![reliability](figures/phase5_F1_reliability.png)", "", "*Figure F1 (phase5_F1).* Reliability per arm on the calibration missions; bars are the "
              "cluster-effective Wilson bands, simultaneous over bins. Exploratory continuation, no gate authority.", ""]
    rice = results.get("rice_study")
    if rice:
        t1 = tests["P5-T1"]
        lines += ["## Rice approximation study (P5-T1)", "", f"{len(rice.get('state_id', []))} stratified oracle states; largest |h(20 ms) - h(5 ms)| = {fmt(rice.get('convergence'))} "
                  f"(criterion 0.005): {t1.get('convergence_verdict')}. Approximation on the {rice.get('domain_count')} states with 0 < h_MC(5 ms) < 0.2: "
                  f"fraction consistent with [1, 1.25] = {fmt(rice.get('domain_fraction'))}: {t1.get('approximation_verdict')}.", ""]
    test_rows = [[k, v.get("statement", ""), v["verdict"]] for k, v in tests.items() if not (staleness and k == "P5-T8")]
    if staleness:
        test_rows.insert(8, ["P5-T8", staleness["statement"], staleness["verdict"]])
    lines += ["## Acceptance results", ""] + table(["Test", "Statement", "Verdict"], test_rows) + [""]
    if staleness:
        srows = []
        for arm, row in staleness["rows"].items():
            fit = row.get("fit") or {}
            failed = [f for f in staleness.get("failed_runs", []) if f["arm"] == arm]
            by_tau = ", ".join(f"{t}: {sum(1 for f in failed if f['tau'] == t)}" for t in row["taus"])
            srows.append([arm, ", ".join(f"{t}: {fmt(r)}" for t, r in zip(row["taus"], row["rates"])), ", ".join(f"{t}: {row['counts'][str(t)]}" for t in row["taus"]),
                          by_tau, fmt(fit.get("p")), f"[{fmt(fit.get('p_lo'))}, {fmt(fit.get('p_hi'))}]", fmt(fit.get("b"))])
        lines += ["### Staleness (P5-T8)", "", f"Snaps counted above the Phase 2 common threshold ({fmt(staleness['threshold'], 0)} N); runs that failed numerically are excluded and counted.", ""]
        lines += table(["Arm", "snap rate by tau [1/s]", "snaps by tau", "failed runs by tau", "p", "p 95%", "b"], srows) + [""]
    lines += ["## Cells, seeds and cost", "", f"Calibration campaign: the Squall Passage, {results['missions']['count']} missions (seeds 5001-5040, 130 s each, recording mode, "
              "common-mode t3 weather, squall plateau at lambda = 1.5), arms O, P, B2, L replayed offline on each mission's sensor log. "
              "Staleness campaign: parallel formation, Gaussian weather, 600 s x 20 seeds (5101-5120) x tau in {0.1, 0.2, 0.4, 0.8, 1.6} s x arms {P, B2}, online.", ""]
    lines += artifacts(["records/phase5/phase5_spec.md", "records/phase5/impact_table_fan.json", "records/phase5/squall_calibration.json", "records/phase5/phase5_results.json",
                        "records/phase5/phase5_staleness_results.json", "reports/figures/phase5_F1_reliability.png"])
    lines += discussion(RECORDS / "phase5" / "phase5_discussion.md")
    return "\n".join(lines)


def main() -> None:
    write_bytes(REPORT, build().encode("utf-8"))
    print(REPORT)


if __name__ == "__main__":
    main()
