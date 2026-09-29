"""Phase 2 report, generated from the committed predictions and results."""

from __future__ import annotations

import json

import numpy as np

from tether.analysis.captions import CAPTIONS
from tether.campaign.common import RECORDS, REPORTS, write_bytes
from tether.campaign.phase2 import (
    HEADING_GAINS,
    INTENSITIES,
    PREDICTIONS_PATH,
    PRETENSIONS,
    RESULTS_PATH,
    cell_name,
)

REPORT = REPORTS / "phase2_report.md"


def _fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if abs(value) >= 1e4 or (abs(value) < 1e-3 and value != 0):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def build() -> str:
    results = json.loads(RESULTS_PATH.read_text())
    predictions = json.loads(PREDICTIONS_PATH.read_text())
    phase1 = json.loads((RECORDS / "phase1" / "phase1_close_results.json").read_text())
    tests = results["tests"]
    rows = results["rows"]
    gate = results["gate"]
    lines = []
    add = lines.append
    add("# Phase 2 Report - The Gaussian threshold laws")
    add("")
    add("## Gate verdict")
    add("")
    add(f"**{gate['decision']}.** Rule: {gate['rule']}.")
    if gate["decision"] == "NO-GO":
        add("")
        add("The physics-structured programme is dead: the snap-rate law fails inside its own domain at powered thresholds.")
    add("")
    add("## Gate review of Phase 1")
    add("")
    add("Phase 1 closed GO with its pre-declared pivots (`reports/phase1_report.md`): impact linearity above 1 m/s with a tabulated low-speed law (`v_b = Z^-1(T_b)`), ballistic symmetry with the formation's effective mass, the deep-excursion threshold at lambda = 1, and the 0.5 ms production step. Pinned values used here: Z = 7993.5 N s/m, f = 0.880, k = 1.7e5 N/m, m_eff = 483.9 kg (two-body, the plan's definition), "
        f"Delta_max = {predictions['constants']['delta_max_phase1']:.2f} m (P1-T8). Open risk carried in: Phase 1(d) excursions are long relative to tau_w/4 (P1-T10 {phase1['tests']['P1-T10']['verdict']}).")
    add("")
    add("## Objectives and tests, fixed before the campaign")
    add("")
    add(f"All predictions were computed and written to `records/phase2/phase2_predictions.json` (sha256 `{results['predictions_sha256'][:16]}...`) before any Phase 2 plant run. Operational choices the plan leaves open were declared there verbatim:")
    add("")
    for key, text in predictions["declarations"].items():
        add(f"- **{key}**: {text}")
    add(f"- **common threshold**: {predictions['constants']['common_threshold'] / 1000:.0f} kN by the rule above (Phase 1(d) counts {predictions['constants']['common_threshold_rule']['counts']}).")
    add(f"- **null check**: weather-only max W_rel/T0 over the null cell's seeds = {predictions['null_weather_max_ratio']:.3f}.")
    add("")
    add("## Cells, seeds and cost")
    add("")
    execution = results["execution"]
    total_wall = sum(v["wall_seconds"] for v in execution.values())
    total_sim = sum(v["sim_seconds"] for v in execution.values())
    add(f"- {len(execution)} cells (4 pretensions x 3 heading gains x intensities {list(INTENSITIES)} x pinned weather, plus the null cell), 20 statistics seeds each (2003-2022) plus 2 pilot seeds (2001-2002), 600 s after a 20 s warm-up, parallel formation, recording mode.")
    add(f"- Statistics seeds: {total_sim:.0f} simulated seconds in {total_wall / 3600:.1f} core-hours; pilot seeds are extra and excluded from every statistic.")
    add("")
    add("## Acceptance results")
    add("")
    add("| Test | Measurement | Criterion | Verdict |")
    add("|---|---|---|---|")
    t = tests
    add(f"| P2-T0 | in-domain cells {t['P2-T0']['cells']}; fraction within 1.5x {_fmt(t['P2-T0']['fraction'])}; all cells {_fmt(t['P2-T0']['fraction_all_cells'])} | >= 80% of in-domain cells (non-blocking) | {t['P2-T0']['verdict']} |")
    add(f"| P2-T1 | AD p > 0.05 on e, edot, q in fraction {_fmt(t['P2-T1']['fraction_ad'])} of {t['P2-T1']['cells']} cells; identity max error {_fmt(t['P2-T1']['identity_max_rel_error'])} | >= 80%; identity within 10% | {t['P2-T1']['verdict']} |")
    add(f"| P2-T2 | cells with onsets {t['P2-T2']['cells_with_onsets']}; Rice ratio within [0.7,1.4] in {_fmt(t['P2-T2']['fraction_rice_within'])}; Rayleigh KS pass in {_fmt(t['P2-T2']['fraction_ks_pass'])} | >= 80% of cells | {t['P2-T2']['verdict']} |")
    add(f"| P2-T3 | powered cells {t['P2-T3']['cells']}; fraction within factor 2 {_fmt(t['P2-T3']['fraction_pass'])} | within factor 2 | {t['P2-T3']['verdict']} |")
    add(f"| P2-T4 | null cell {t['P2-T4']['null_cell']}: excursions {t['P2-T4']['excursions']}, snaps per seed above {_fmt(t['P2-T4']['threshold'])} N: {t['P2-T4']['snaps_per_seed']} | zero in every seed | {t['P2-T4']['verdict']} |")
    add(f"| P2-T5 | in-domain powered cells {t['P2-T5']['in_domain_powered_cells']}; all powered cells {t['P2-T5']['powered_cells_all']}: ratio-within fraction {_fmt(t['P2-T5']['fraction_ratio_all_powered'])}, literal-slope fraction {_fmt(t['P2-T5']['fraction_slope_literal_all_powered'])}, secant-slope fraction {_fmt(t['P2-T5']['fraction_slope_secant_all_powered'])} | ratio in [1/2, 2] at every powered threshold in >= 80% of in-domain cells; slope within 20% | {t['P2-T5']['verdict']} |")
    add(f"| P2-T6 | cells {t['P2-T6']['cells']}; ratio within (LTI) {_fmt(t['P2-T6']['fraction_ratio_lti'])}, (measured moments) {_fmt(t['P2-T6']['fraction_ratio_measured'])}; slope within {_fmt(t['P2-T6']['fraction_slope'])} | ratio in [1/2, 2]; slope within 20% | {t['P2-T6']['verdict']} |")
    add(f"| P2-T7 | powered cells {t['P2-T7']['cells']}; pass fraction {_fmt(t['P2-T7']['fraction_pass'])} | median abs(ratio-1) <= 0.25 | {t['P2-T7']['verdict']} |")
    add(f"| P2-T8 | common threshold {t['P2-T8']['common_threshold'] / 1000:.0f} kN; groups within 20% of span: {sum(g['within_20pct_span'] for g in t['P2-T8']['groups'])}/{len(t['P2-T8']['groups'])} | within 20% of the span | {t['P2-T8']['verdict']} |")
    add(f"| P2-T9 | cells {t['P2-T9']['cells']}; all below bound {_fmt(t['P2-T9']['all_below'])}; bound within 30% in {_fmt(t['P2-T9']['fraction_within_30pct'])} | below bound; within 30% | {t['P2-T9']['verdict']} |")
    add(f"| P2-T10 | passing cells: {t['P2-T10']['passing_cells'] or 'none'}; bounding box {t['P2-T10']['bounding_box']} | reported | {t['P2-T10']['verdict']} |")
    add("")
    add("### Cell table")
    add("")
    add("| Cell | Excursions | Slack duty | p95 dwell [s] | Multi-max | In domain | Grid T_b [kN] | Powered | Snap ratio (meas/Thm 6) per threshold | Slope meas / literal |")
    add("|---|---:|---:|---:|---:|:---:|---|:---:|---|---|")
    for intensity in INTENSITIES:
        for heading_gain in HEADING_GAINS:
            for pretension in PRETENSIONS:
                name = cell_name(pretension, heading_gain, intensity)
                row = rows[name]
                grid = ", ".join(f"{g / 1000:.1f}" for g in row["grid"]) if row["grid"] else "-"
                ratios = ", ".join(_fmt(s["ratio"], 2) + ("" if s["powered"] else "*") for s in row["snap"]) if row["snap"] else "-"
                slope = f"{_fmt(row['t5']['measured_slope'])} / {_fmt(row['t5']['literal_slope'])}" if "t5" in row else "-"
                add(f"| {name} | {row['excursions']} | {row['slack_duty']:.4f} | {_fmt(row['dwell_p95'], 2)} | {_fmt(row['multi_max_fraction'], 3)} | {_fmt(row['in_domain'])} | {grid} | {_fmt(row['powered'])} | {ratios} | {slope} |")
    add("")
    add("`*` marks thresholds with fewer than 20 exceedances (not powered, no verdict).")
    add("")
    add("### Snap rates at the powered grid (reference heading gain 500 N m/rad)")
    add("")
    add("| Cell | T_b [kN] | count | measured rate [1/s] (95% CI) | Theorem 6 | Cor. 5 functional | long-excursion share |")
    add("|---|---:|---:|---|---:|---:|---:|")
    for intensity in INTENSITIES:
        for pretension in PRETENSIONS:
            name = cell_name(pretension, 500.0, intensity)
            for s in rows[name]["snap"]:
                add(f"| {name} | {s['threshold'] / 1000:.2f} | {s['count']} | {_fmt(s['rate'])} ({_fmt(s['rate_lo'])}, {_fmt(s['rate_hi'])}) | {_fmt(s['predicted_theorem6'])} | {_fmt(s['predicted_corollary5'])} | {_fmt(s['long_excursion_share'], 2)} |")
    add("")
    add("### Pretension optimum (P2-T8)")
    add("")
    add("| Intensity | k_h | measured total rate by T0 | measured argmin | predicted argmin | within 20% |")
    add("|---:|---:|---|---:|---:|:---:|")
    for group in t["P2-T8"]["groups"]:
        totals = ", ".join(f"{p:.0f}: {_fmt(r)}" for p, r, _ in group["totals"])
        add(f"| {group['intensity']} | {group['heading_gain']:.0f} | {totals} | {group['measured_argmin']:.0f} | {group['predicted_argmin']:.0f} | {_fmt(group['within_20pct_span'])} |")
    add("")
    add("## Deviations from the plan")
    add("")
    add("1. Pretension is set by thrust alone (declared above): with linear drag the tow speed is dynamically inert, so the fan angle is not needed to separate speed from pretension.")
    add("2. The domain, grids, gust episodes, and slopes follow the declarations above wherever the plan leaves the operational definition open.")
    add("")
    add("## Artifacts")
    add("")
    for path in ("records/phase2/phase2_predictions.json", "records/phase2/phase2_results.json", "records/phase2/phase2_records.npz", "records/phase2/phase2_manifest.json"):
        add(f"- `{path}`")
    for key, caption in CAPTIONS.items():
        if key.startswith("phase2"):
            add(f"- `reports/figures/{key}_*.png`: {caption}")
    add("")
    discussion = RECORDS / "phase2" / "phase2_discussion.md"
    if discussion.exists():
        add(discussion.read_text().rstrip())
        add("")
    return "\n".join(lines)


def main() -> None:
    from tether.analysis.phase2_figures import main as figures

    figures()
    text = build()
    write_bytes(REPORT, text.encode("utf-8"))
    print(REPORT)


if __name__ == "__main__":
    main()
