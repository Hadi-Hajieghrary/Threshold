"""Phase 6 report, generated from the committed results."""

from __future__ import annotations

import json

from tether.analysis.report_common import EXPLORATORY, artifacts, discussion, fmt, table
from tether.campaign.common import RECORDS, REPORTS, write_bytes

REPORT = REPORTS / "phase6_report.md"


def build() -> str:
    results = json.loads((RECORDS / "phase6" / "phase6_results.json").read_text())
    tests = results["tests"]
    lines = ["# Phase 6 Report - Capstone", "", "## Verdict", "", f"**{results['gate']['decision']}.** Rule: {results['gate']['rule']}.", "", EXPLORATORY, "",
             "## Declarations", ""]
    lines += [f"- **{k}**: {v}" for k, v in results["declarations"].items()] + [""]
    lines += [f"Stress breaking strength T_b^s = {results['breaking_strength']:.0f} N (fixed in Phase 5); h_crit per arm {results['h_crit']}; 60 seeds, common random numbers, recording and live campaigns.", ""]
    closures = tests["P6-T1"].get("closures") or {}
    rows = [[arm, fmt(results["probabilities"]["recording"][arm]), fmt(results["probabilities"]["live"][arm]),
             (closures.get(arm) or {}).get("closures"), (closures.get(arm) or {}).get("closure_before_severance"), fmt((closures.get(arm) or {}).get("severance_or_closure"))]
            for arm in results["probabilities"]["recording"]]
    lines += ["## First-severance probability per mission", ""] + table(["Arm", "recording (virtual)", "live", "closures (recording)", "closure before severance",
                                                                          "severance or closure (recording)"], rows) + [""]
    diff_rows = []
    for mode, key in (("recording", "P6-T1"), ("live", "P6-T2")):
        for name, (value, lo, hi) in tests[key]["differences"].items():
            diff_rows.append([mode, name, fmt(value), f"[{fmt(lo)}, {fmt(hi)}]"])
    lines += ["## Paired differences", ""] + table(["Campaign", "Difference", "estimate", "95% seed bootstrap"], diff_rows) + [""]
    t1 = tests["P6-T1"]
    if "alarm_rate" in t1:
        lines += ["### Alarm rate (the alternative L criterion of P6-T1)", ""]
        lines += table(["Arm", "recording", "live"], [[arm, fmt(t1["alarm_rate"][arm]), fmt(tests["P6-T2"]["alarm_rate"][arm])] for arm in t1["alarm_rate"]]) + [""]
    if "probability_by_level" in t1:
        levels = t1["powered_levels"]
        lines += ["### Per-mission virtual first-severance probability at every threshold (recording)", ""]
        lines += table(["Arm"] + [f"{level / 1000:g} kN" for level in levels], [[arm] + [fmt(p) for p in probs] for arm, probs in t1["probability_by_level"].items()]) + [""]
    lines += ["## Acceptance results", ""] + table(["Test", "Statement", "Verdict"], [[k, v.get("statement", ""), v["verdict"]] for k, v in tests.items()]) + [""]
    t5 = tests["P6-T5"]
    lines += ["### False alarms (P6-T5)", "", f"Arm P, recording campaign: {t5['episodes']} action episodes starting in the squall, {t5['false']} with no re-engagement above 0.5 v_b within H "
              f"(rate {fmt(t5['rate'])}); over the whole mission {t5.get('mission_episodes')} episodes, {t5.get('mission_false')} false (rate {fmt(t5.get('mission_rate'))}).", ""]
    extrapolation = tests["P6-T6"].get("extrapolation")
    if extrapolation:
        rows = []
        for arm, entry in extrapolation.items():
            for population in ("marks", "q_peaks"):
                fit = entry[population]
                interval = fit.get("interval")
                rows.append([arm, population, fit.get("n"), fmt(fit.get("threshold")), fmt(fit.get("shape")), fmt(fit.get("max_observed")),
                             "withheld" if fit.get("withheld") else f"{fmt(fit.get('rate_at_level'))} [{fmt(interval[0])}, {fmt(interval[1])}]",
                             fmt(fit.get("fitted_rate_unchecked"))])
        lines += ["### Operational extrapolation to 25 kN (P6-T6, reported, never gated)", ""]
        lines += table(["Arm", "population", "peaks", "threshold u [N]", "GPD shape", "largest peak [N]", "rate at 25 kN [1/s] (95% profile)", "fitted rate before the check"], rows) + [""]
    attribution = tests["P6-T7"].get("attribution") or {}
    if attribution.get("N") and attribution.get("P"):
        rows = [[arm, e["deep_excursions"], fmt(e["median_depth"]), fmt(e["median_a_bar"]), fmt(e["mean_thrust_scale_during_slack"])] for arm, e in
                ((a, attribution[a]) for a in ("N", "P"))]
        ratio = attribution.get("P_over_N") or {}
        lines += ["### Mechanism attribution (P6-T7, reported)", ""]
        lines += table(["Arm", "deep excursions", "median depth [m]", "median return-leg a_bar [m/s^2]", "mean commanded thrust scale during slack"], rows)
        lines += ["", f"P over N: a_bar ratio {fmt(ratio.get('a_bar'))}, depth ratio {fmt(ratio.get('depth'))}.", ""]
    after = tests["P6-T8"].get("post_severance")
    if after:
        rows = [[arm, e["runs"], f"{fmt(e.get('severed_offset_before'))} -> {fmt(e.get('severed_offset_after'))}", f"{fmt(e.get('survivor_spread_before'))} -> {fmt(e.get('survivor_spread_after'))}",
                 f"{fmt(e.get('survivor_error_before'))} -> {fmt(e.get('survivor_error_after'))}"] for arm, e in after.items()]
        lines += ["### Estimation after a live severance (P6-T8, EMP)", ""]
        lines += table(["Arm", "runs", "severed vessel offset [m]", "survivor spread [m]", "survivor error [m]"], rows) + [""]
    hero = RECORDS / "phase6" / "phase6_hero.json"
    if hero.exists():
        meta = json.loads(hero.read_text())
        lines += ["### Hero replay (P6-T9)", "", f"Seed {meta['seed']} ({meta['rule']}).", "", "![hero replay](figures/phase6_F1_hero_replay.png)", "",
                  "*Figure F1 (phase6_F1).* Seed-matched arms N and P at T_b^s in recording mode: hazard, tension, thrust scale, elongation and closing speed "
                  "on the cable that severs first under N. Exploratory continuation, no gate authority.", ""]
    cost_rows = []
    for arm, c in tests["P6-T4"]["cost"].items():
        cost_rows.append([arm, f"{fmt(c['mission_time_penalty'][0])} [{fmt(c['mission_time_penalty'][1])}, {fmt(c['mission_time_penalty'][2])}]",
                          f"{fmt(c['docking_error_penalty_m'][0])} [{fmt(c['docking_error_penalty_m'][1])}, {fmt(c['docking_error_penalty_m'][2])}]"])
    lines += ["### Cost against arm N (paired)", ""] + table(["Arm", "mission-time penalty (fraction)", "docking-error penalty [m]"], cost_rows) + [""]
    lines += ["## Cells, seeds and cost", "", f"Two campaigns on the Squall Passage (130 s, common-mode t3 weather, squall plateau at lambda = 1.5), seeds 6001-6060, arms N, L, B2, P, O, "
              f"recording and live: {2 * 5 * 60} missions, {fmt(results['wall_seconds'] / 3600.0, 1)} worker-hours.", ""]
    lines += artifacts(["records/phase6/phase6_results.json", "records/phase6/phase6_hero.npz", "records/phase6/phase6_hero.json", "reports/figures/phase6_F1_hero_replay.png"])
    lines += discussion(RECORDS / "phase6" / "phase6_discussion.md")
    return "\n".join(lines)


def main() -> None:
    write_bytes(REPORT, build().encode("utf-8"))
    print(REPORT)


if __name__ == "__main__":
    main()
