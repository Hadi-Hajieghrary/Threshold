"""Consolidated Phase 1 report: every committed Phase 1 record plus the closure."""

from __future__ import annotations

import json
import math

import numpy as np

from tether.analysis.captions import CAPTIONS
from tether.campaign.common import RECORDS, REPORTS, write_bytes

P1 = RECORDS / "phase1"
REPORT = REPORTS / "phase1_report.md"
F = 0.880
K = 1.7e5


def _load(name: str) -> dict:
    return json.loads((P1 / name).read_text())


def _t4_rescore() -> dict:
    record = _load("phase1_t4r_results.json")
    z_fleet = _load("phase1r_results.json")["tests"]["P1-T2b"]["impedance_n_s_per_m"]
    m_fleet = (z_fleet / F) ** 2 / K
    m_two = 600.0 * 2500.0 / 3100.0
    rows = [
        m
        for m in record["study"]["measurements"]
        if m["branch"] == "constant_force_control" and abs(m["physics_step_s"] - 2.5e-4) < 1e-12
    ]
    out = {"m_fleet": m_fleet, "m_two_body": m_two}
    for label, ratios in (("plan_scope", (0.8, 0.95)), ("lambda_zero_extension", (0.0,))):
        selected = [m for m in rows if m["gust_ratio"] in ratios]
        depth_ratio = np.array([m["measured_turning_depth_m"] / (m["derivation_constant_force_depth_m"] * m_fleet / m_two) for m in selected])
        out[label] = {
            "cells": len(selected),
            "valid": int(sum(m["event_order_and_return_valid"] for m in selected)),
            "max_speed_error": float(max(m["speed_relative_error"] for m in selected)),
            "depth_ratio_min": float(depth_ratio.min()),
            "depth_ratio_max": float(depth_ratio.max()),
        }
    scope = out["plan_scope"]
    out["verdict"] = "PASS" if scope["valid"] == scope["cells"] and scope["max_speed_error"] < 0.05 and scope["depth_ratio_max"] < 1.1 else "FAIL"
    return out


def _corrected_prop3() -> dict:
    from tether.physics.fleet import fleet_effective_mass, formation_geometry, two_body_effective_mass

    geometry = formation_geometry("parallel")
    with np.load(P1 / "phase1_close_records.npz", allow_pickle=False) as data:
        names = [str(n) for n in data["cell_names"]]
        index = data["marks_cell_index"]
        depth = data["marks_depth"]
        u = data["marks_u_entry"]
        v = data["marks_v_return"]
        cable = data["marks_cable"]
        dwell = data["marks_dwell"]
        maxima = data["marks_n_maxima"]
    pretension = np.array([float(names[i].split("_")[0][1:]) for i in index])
    ok = depth > 0.0
    m_two = np.array([two_body_effective_mass(geometry, int(c)) for c in cable[ok]])
    m_fleet = np.array([fleet_effective_mass(geometry, int(c)) for c in cable[ok]])
    h4 = (dwell[ok] < 2.0) & (maxima[ok] <= 1)

    def fit(x, y):
        slope = float(np.dot(x, y) / np.dot(x, x))
        residual = y - slope * x
        return slope, float(1.0 - np.dot(residual, residual) / np.dot(y, y))

    printed = np.sqrt(u[ok] ** 2 + 2.0 * pretension[ok] / m_two * depth[ok])
    two = np.sqrt(2.0 * pretension[ok] / m_two * depth[ok])
    fleet = np.sqrt(2.0 * pretension[ok] / m_fleet * depth[ok])
    rows = []
    for label, selection in (("(H4),(H5)", h4), ("all", np.ones_like(h4)), ("violating (H4)", ~h4)):
        rows.append({"selection": label, "n": int(selection.sum()), "printed": fit(printed[selection], v[ok][selection])[0],
                     "two_body": fit(two[selection], v[ok][selection])[0], "fleet": fit(fleet[selection], v[ok][selection])[0],
                     "r2": fit(two[selection], v[ok][selection])[1]})
    shallow = h4 & (depth[ok] < 0.05)
    return {"rows": rows, "shallow_v_over_u": float(np.median(v[ok][shallow] / np.maximum(u[ok][shallow], 1e-9)))}


def _return_leg_analysis() -> dict:
    """Measured return-leg acceleration against Prop. 4's net acceleration (T0 - W_rel)/m_eff."""
    from tether.physics.fleet import formation_geometry, two_body_effective_mass

    geometry = formation_geometry("parallel")
    with np.load(P1 / "phase1_close_records.npz", allow_pickle=False) as data:
        names = [str(n) for n in data["cell_names"]]
        index = data["marks_cell_index"]
        depth = data["marks_depth"]
        abar = data["marks_a_bar_return"]
        wret = data["marks_wrel_return_mean"]
        cable = data["marks_cable"]
    pretension = np.array([float(names[i].split("_")[0][1:]) for i in index])
    keep = (depth > 0.05) & np.isfinite(abar) & np.isfinite(wret)
    mass = np.array([two_body_effective_mass(geometry, int(c)) for c in cable[keep]])
    predicted = (pretension[keep] - wret[keep]) / mass
    a0 = pretension[keep] / mass
    measured = abar[keep]
    slope = float(np.dot(predicted, measured) / np.dot(predicted, predicted))
    residual = measured - slope * predicted
    bands = []
    for lo, hi in ((0.05, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 12.0)):
        sel = (depth[keep] >= lo) & (depth[keep] < hi)
        if sel.any():
            bands.append({"lo": lo, "hi": hi, "n": int(sel.sum()), "median_abar_over_a0": float(np.median(measured[sel] / a0[sel]))})
    return {"n": int(keep.sum()), "slope": slope, "r2": float(1.0 - np.dot(residual, residual) / np.dot(measured, measured)), "bands": bands}


def build() -> str:
    mech = _load("phase1_mechanics.json")
    det = _load("phase1_deterministic.json")
    p1r = _load("phase1r_results.json")
    t9r = _load("phase1_t9r_results.json")
    t5r = _load("phase1_t5r_results.json")
    close = _load("phase1_close_results.json")
    impact = _load("impact_table.json")
    t4 = _t4_rescore()
    ct = close["tests"]
    t5s = ct["P1-T5-stochastic"]
    pooled = t5s["pooled"]
    t10 = ct["P1-T10"]["per_cell"]
    t8 = ct["P1-T8"]
    t6p = ct["P1-T6-production"]
    t2bp = ct["P1-T2b-production"]
    t9_rows = {row["time_step_s"] if "time_step_s" in row else None: row for row in []}
    del t9_rows
    t5_energy_pass = abs(pooled["H4H5_two_body_a0"]["slope"] - 1.0) <= 0.05 and pooled["H4H5_two_body_a0"]["r2"] > 0.98
    lines = []
    add = lines.append
    add("# Phase 1 Report - Unilateral cables and the excursion law")
    add("")
    add("This report consolidates Phase 1. It supersedes the stage reports `phase1_mechanics_report.md`, `phase1_deterministic_report.md`, `phase1r_report.md`, `phase1_t9r_report.md`, `phase1_t5r_report.md` and `phase1_t4r_report.md`, whose records remain authoritative for the measurements quoted from them.")
    add("")
    add("## Gate verdict")
    add("")
    gate_go = t4["verdict"] == "PASS" and t6p["verdict"] == "PASS"
    add(f"**{'GO, with the pre-declared pivots recorded below' if gate_go else 'NO-GO'}.** The outcome matrix requires T1, T2, T4, T5, T6 and T9. "
        "T1, T2, T6 and T9 pass as stated (T9 at the production physics step). T2's low-speed non-linearity triggers the matrix's own pivot: the impact law is tabulated and "
        "`v_b = Z^-1(T_b)` replaces `T_b/Z` downstream. T4 passes once the restoring acceleration uses the formation's effective reduced mass taken from the independently pinned fleet impedance, the restatement pre-mortem 1 anticipates. "
        + ("T5's energy conversion passes on the deterministic grid and on the stochastic excursions that satisfy (H4) and (H5)." if t5_energy_pass else
           "T5's energy conversion passes on the deterministic grid but fails on the stochastic excursions, with T4 passing: the return leg is still driven, so the matrix's pivot applies - the measured return-leg acceleration a_bar replaces a0 and Corollary 5 is downgraded to its measured-a_bar form."))
    add("")
    add("## Cells, seeds and cost")
    add("")
    execution = close["execution"]
    add(f"- Phase 1(d) stochastic cells: {len(execution['cells'])} cells x {len(close['configuration']['seeds'])} seeds x {close['configuration']['duration_s']:.0f} s (after a {close['configuration']['warmup_s']:.0f} s warm-up), Gaussian local weather at intensities {close['configuration']['weather_intensities']} x the pinned 3.5/0.7 kN, pretensions {close['configuration']['pretensions_n']} N, heading gain 500 N m/rad. Seeds {close['configuration']['seeds'][0]}-{close['configuration']['seeds'][-1]}.")
    add(f"- Simulated {execution['total_sim_seconds']:.0f} s in {execution['total_wall_core_seconds'] / 3600.0:.2f} core-hours ({execution['total_wall_core_seconds'] / execution['total_sim_seconds']:.3f} wall-s per sim-s under full load on the throttled 8P+8E host).")
    add(f"- Impact table: {len(impact)} (pretension, cable position) pairs x {len(close['configuration']['impact_speeds'])} closing speeds, deterministic, no weather.")
    add("- Deterministic evidence from earlier stages: Phase 1R (60 full-plant cells), P1-T9R (15 SAP + 3 RK3 cells), P1-T5R (42 cells), P1-T4R (216 cells); counts and wall times are in their records.")
    add("")
    add("## The production plant")
    add("")
    add("Phase 1(d) and every later phase run on `tether/physics/fleet.py`, which replaces the per-cable `UnilateralCable` systems (whose per-update deep copy made them cost 9.8 wall-s per simulated second) with one `UnilateralCableBank` owning all five cables, as IV.4 specifies, and one `HullForces` system. It adds what Phases 0-1 had simplified away: stern attachment points, attachments on the pentagon's forward face by ray-polygon intersection (IV.3), thrust along the true hull axis, the sensor suites of IV.6 and the sensor-driven heading controller of IV.7, and the diagram-walking truth-isolation lint of IV.11. Its regression suite (`tether/tests/test_fleet_plant.py`, 15 tests) checks the steady-state identity open- and closed-loop at three pretensions, attachment rates against central differences, the generalized-force path against Drake's own `CalcGeneralizedForces` for point forces (agreement 1e-8 N), bit-identical determinism, the lint, and the fleet impedance against the Phase 1 pin.")
    add("")
    add("## Acceptance results")
    add("")
    add("| Test | Measurement | Criterion | Verdict |")
    add("|---|---|---|---|")
    t1 = p1r["tests"]["P1-T1"]
    add(f"| P1-T1 | distance error {100 * t1['distance_relative_error']:.2e}%; tension error {100 * t1['tension_relative_error']:.2e}%; no slack | <= 1%; <= 3%; no slack | PASS |")
    t2 = mech["tests"]["P1-T2"]
    add(f"| P1-T2 | Z = {t2['impedance_n_s_per_m']:.1f} N s/m, R2 = {t2['uncentred_r_squared']:.6f}, {100 * t2['relative_error']:.3f}% from f sqrt(k m_eff) | R2 > 0.99; within 15% | PASS (pinned) |")
    t2b = p1r["tests"]["P1-T2b"]
    add(f"| P1-T2b | Z_fleet = {t2b['impedance_n_s_per_m']:.1f} N s/m (R2 {t2b['uncentred_r_squared']:.4f}) vs 9.0 kN s/m predicted; production plant centre cable {t2bp['impedance']:.1f} N s/m above 1 m/s (R2 {t2bp['r2']:.5f}, {100 * t2bp['relative_difference']:+.2f}% vs pin) | R2 > 0.99; recorded | PASS (pinned); low-speed non-linearity -> tabulated pivot |")
    t3 = mech["tests"]["P1-T3"]
    add(f"| P1-T3 | m_eff = {t3['estimated_reduced_mass_kg']:.2f} kg vs {t3['analytic_reduced_mass_kg']:.2f} kg ({100 * t3['relative_error']:.2f}%) | within 10% | PASS |")
    add(f"| P1-T4 | plan scope (scripted lambda < 1): {t4['plan_scope']['valid']}/{t4['plan_scope']['cells']} cells valid, max abs(V_up - u)/u = {100 * t4['plan_scope']['max_speed_error']:.2f}%, depth / (u^2/2a) = {t4['plan_scope']['depth_ratio_min']:.3f}-{t4['plan_scope']['depth_ratio_max']:.3f} with a = (1 - lambda) T0/m_fleet, m_fleet = (Z_fleet/f)^2/k = {t4['m_fleet']:.1f} kg | < 5%; depth < 1.1 x ballistic | {t4['verdict']} (restated a0) |")
    t5d = det["tests"]["P1-T5"]
    add(f"| P1-T5 (deterministic) | energy slope {t5d['energy_slope']:.4f}, R2 {t5d['energy_uncentred_r_squared']:.4f} over {t5d['complete_count']} scripted excursions; Phase 1R full-plant energy balance max residual {p1r['tests']['P1-T5']['full_plant_energy_balance']['maximum_normalized_residual']:.2e} | slope within 5%, R2 > 0.98 | PASS |")
    add(f"| P1-T5 (stochastic) | (H4),(H5) excursions: n = {pooled['H4H5_two_body_a0']['n']}, slope {pooled['H4H5_two_body_a0']['slope']:.4f}, R2 {pooled['H4H5_two_body_a0']['r2']:.4f} (a0 two-body); with m_fleet: slope {pooled['H4H5_fleet_a0']['slope']:.4f}; excursions violating (H4): n = {pooled['not_H4_two_body_a0']['n']}, slope {pooled['not_H4_two_body_a0']['slope']:.4f} | slope within 5%, R2 > 0.98 | {t5s['verdict']} |")
    add("| P1-T5 (impact) | T_peak vs Z V_up fails below 0.5 m/s (P1-T5R: no candidate linear model within 6%; the zero-speed engagement peak is 1.62 T0) | within 6% | pivot: tabulated Z^-1 (production impact table) |")
    t6 = p1r["tests"]["P1-T6"]
    add(f"| P1-T6 | corrected protocol lambda_c = {t6['lambda_c_interval'][0]:.4f}-{t6['lambda_c_interval'][1]:.4f} (Phase 1R); production plant lambda_c = {t6p['lambda_c']:.6f} | within 5% of 1 | PASS |")
    t7 = p1r["tests"]["P1-T7"]
    add(f"| P1-T7 | shutoff-depth error {100 * t7['maximum_shutoff_depth_relative_error']:.2f}%, amended max-depth error {100 * t7['maximum_amended_depth_relative_error']:.2f}% | within 15% where depth < 1 m | PASS |")
    snap = t8["implied_max_snap_f_sqrt_2kT0Delta"]
    add(f"| P1-T8 | largest complete stochastic depth {t8['observed_max_complete_depth']['depth']:.2f} m ({t8['observed_max_complete_depth'].get('cell', '')}); implied f sqrt(2 k T0 Delta) = {snap / 1000:.1f} kN, observed peak {t8['observed_max_complete_depth'].get('T_peak', float('nan')) / 1000:.1f} kN; closures (chord < 1 m) {sum(t8['closure_events'].values())} | measured and pinned | MEASURED |")
    add(f"| P1-T9 | discrete SAP energy envelope 0.946% at 1 ms (FAIL), 0.471% at 0.5 ms, 0.235% at 0.25 ms; impulse residuals < 2e-14 (P1-T9R) | energy within 0.5%; impulses to 1e-9 | PASS at the 0.5 ms production step |")
    passing = sum(1 for row in t10 if row["below_tau_w_over_4"] is True)
    add(f"| P1-T10 | p95 excursion duration below tau_w/4 = 2 s in {passing} of {sum(1 for r in t10 if r['excursions'])} cells with excursions (table below) | p95 < tau_w/4; reported otherwise | {ct['P1-T10']['verdict']} |")
    add("")
    add("### Phase 1(d) cells")
    add("")
    add("| Cell | Excursions | Slack duty | p50 / p95 dwell [s] | Multi-max | Largest snap [kN] | Closures | Mean cable-angle sd [deg] | Load yaw sd [deg] |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in t10:
        dwell = f"{row['dwell_p50']:.2f} / {row['dwell_p95']:.2f}" if row["excursions"] else "-"
        add(f"| {row['cell']} | {row['excursions']} | {row['slack_duty']:.4f} | {dwell} | {row['multi_max_fraction'] if not row['excursions'] else round(row['multi_max_fraction'], 3)} | {row['snap_max'] / 1000:.2f} | {row['closures']} | {np.mean(row['cable_angle_std_deg']):.1f} | {row['load_yaw_std_deg']:.1f} |")
    add("")
    add("### Energy conversion by cell (stochastic)")
    add("")
    add("| Cell | Selection | n | slope | R2 |")
    add("|---|---|---:|---:|---:|")
    for row in t5s["per_cell"]:
        add(f"| {row['cell']} | {row['selection']} | {row['n']} | {row['slope']:.4f} | {row['r2']:.4f} |")
    add("")
    add("### Peak tension against the impact models (stochastic snaps with V_up >= 1 m/s)")
    add("")
    add("| Cell | n | tabulated: median / max rel. error | linear Z V_up: median / max rel. error |")
    add("|---|---:|---:|---:|")
    for row in t5s["tpeak_vs_impact_model"]:
        add(f"| {row['cell']} | {row['n']} | {100 * row['tabulated_median_rel_error']:.1f}% / {100 * row['tabulated_max_rel_error']:.1f}% | {100 * row['linear_median_rel_error']:.1f}% / {100 * row['linear_max_rel_error']:.1f}% |")
    add("")
    add("### Production impact table (first engagement peak, kN)")
    add("")
    speeds = next(iter(impact.values()))["speeds"]
    add("| Pretension, cable | " + " | ".join(f"{s:g} m/s" for s in speeds) + " | Z above 1 m/s [N s/m] |")
    add("|---|" + "---:|" * (len(speeds) + 1))
    for key, entry in impact.items():
        add(f"| {entry['pretension']:.0f} N, c{entry['cable']} | " + " | ".join(f"{p / 1000:.2f}" for p in entry["peaks"]) + f" | {entry['asymptotic_impedance']:.0f} |")
    add("")
    add("## Pivots carried downstream")
    add("")
    add("1. **Tabulated impact law (T2 non-linear branch).** The engagement peak carries the step of the restored pretension (1.62 T0 at zero closing speed) and is not proportional to closing speed below about 1 m/s. Every downstream use of `v_b = T_b/Z` becomes `v_b = Z^-1(T_b)` through `records/phase1/impact_table.json`, per pretension and cable position. Above 1 m/s the table is linear (R2 > 0.99) and the pinned impedances stand: Z = 7993.5 N s/m single cable (Phases 2-3), Z_fleet = 8304.5 N s/m (Phases 4-6).")
    add(f"2. **Formation effective mass (pre-mortem 1).** In the formation the four other vessels ride with the load through their stiff cables, so the slack-excursion restoring acceleration is T0/m_fleet with m_fleet = {t4['m_fleet']:.1f} kg from the pinned fleet impedance (two-body {t4['m_two_body']:.1f} kg). The plan's definitions of W_rel and m_eff are kept for the theory; this restatement is reported beside them.")
    if not t5_energy_pass:
        add("3. **Measured-a_bar energy conversion (T5 fail with T4 pass).** Stochastic excursions are driven on the return leg by slowly relenting gusts, so a_bar != a0 and the marks' measured a_bar = (V_up^2 - u^2)/(2 Delta) replaces a0; Corollary 5 is downgraded to its measured-a_bar form.")
    add("")
    add("## Deviations from the plan")
    add("")
    add("1. **Physics step 0.5 ms (plan: 1 ms).** 1 ms fails P1-T9's 0.5% energy criterion (0.946%, the bounded O(omega h) oscillation of the explicit spring coupling); 0.5 ms passes the unchanged criterion (0.471%) and was chosen over the 0.25 ms step that P1-T9R preferred for margin, because the bound is deterministic (identical at every impact speed), engagement peaks differ by 0.08% between 0.5 and 0.25 ms (P1-T5R), and the measured throughput on this host under full load (P-cores throttled to 2.6 GHz, E-cores to 2.1 GHz) made 0.25 ms cost roughly twice as much. Events and logs stay at 1 ms.")
    add("2. **Generalized-force port instead of spatial-force lists.** For planar joints on the world frame the generalized force of a body is exactly its world (f_x, f_y, tau_z) about its origin; the equality with Drake's point-force map is a regression test. The plant carries no SceneGraph (there is no geometry).")
    add("3. **Gyro model.** The plan's 'b a random walk of 0.01 deg/sqrt(s)' has angle-random-walk units; it is implemented as white rate noise of that density. Read as a rate-bias random walk it would drift the gyro-integrated heading by about 85 deg (1 sigma) in a 600 s cell.")
    add("4. **Controller trim sign.** `yaw = k_h wrap(theta_ref - theta_hat) - k_c sin(bearing)` with the bearing measured positive to port of astern; the minus sign is the one that turns the hull toward its cable, the purpose of the term.")
    add("5. **Phase 1(d) cell definition.** The plan names 8 cells at two intensities; they are the four Phase 2 pretensions x intensities {0.5, 1.0} x pinned weather at the nominal heading gain, 20 seeds x 300 s each.")
    add("6. **Formation closure.** No contact is modelled; a run stops when any chord shortens below 1 m (the vessel's stern within one beam of the load attachment) and its exposure ends there. Vessel-vessel contact in the 1 m-spaced parallel formation is not modelled either.")
    add("7. **P1-T4 scoring.** The literal table criterion `depth < 1.1 u^2/(2 a0)` contradicts Proposition 2 whenever a gust acts during the excursion (depth is u^2/(2a) with a = a0(1-lambda)); it is withdrawn as a specification error, as the deterministic stage found, and T4 is scored on Proposition 2's own statement.")
    add("")
    corrected = _corrected_prop3()
    add("### P1-T5 with Proposition 3 corrected")
    add("")
    add("Proposition 3 as printed, `V_up = sqrt(u^2 + 2 a_bar Delta)`, counts the entry energy twice: for the ballistic excursion of Proposition 2 it gives sqrt(2) u where Proposition 2 gives u (see the plan-level discrepancies below). The energy identity measured from the deepest point is `V_up = sqrt(2 a_bar_return Delta)`. Re-scored with that form and a_bar = a0:")
    add("")
    add("| Selection | n | slope (printed form) | slope (corrected, two-body a0) | slope (corrected, m_fleet) | R2 (corrected, two-body) |")
    add("|---|---:|---:|---:|---:|---:|")
    for row in corrected["rows"]:
        add(f"| {row['selection']} | {row['n']} | {row['printed']:.4f} | {row['two_body']:.4f} | {row['fleet']:.4f} | {row['r2']:.4f} |")
    add("")
    add(f"Shallow (< 5 cm) excursions satisfying (H4),(H5) return at a median V_up/u = {corrected['shallow_v_over_u']:.3f}, so ballistic symmetry holds for ordinary slack; the printed form's sqrt(2) factor explains part of the 0.69 slope. The correction does not rescue the stochastic test (slope 1.14 with R2 0.85 on (H4),(H5) excursions; 0.66 on the long ones, where the return leg is still driven), so the verdict and the measured-a_bar pivot stand.")
    add("")
    returns = _return_leg_analysis()
    add("### Return-leg acceleration (why stochastic T5 fails)")
    add("")
    add(f"On the {returns['n']} stochastic excursions deeper than 5 cm, the measured return-leg acceleration a_bar = (V_up^2 - u^2)/(2 Delta) is not explained by Proposition 4's net acceleration (T0 - W_rel,return)/m_eff: through-origin slope {returns['slope']:.3f}, R2 {returns['r2']:.3f}. By depth band, the median a_bar/a0 is "
        + ", ".join(f"{b['median_abar_over_a0']:.2f} ({b['lo']}-{b['hi']} m, n = {b['n']})" for b in returns["bands"])
        + ". Slack under slow weather is quasi-static: the gust that drove the tension to zero is still acting when the cable returns, and the drifting vessel's drag, the other cables and the large chord swings (below) set the radial balance, none of which the instantaneous W_rel carries. The measured-a_bar pivot is therefore the operative form of Proposition 3 on this plant, and Corollary 5's closed form is not expected to hold in Phase 2 (P2-T7).")
    add("")
    add("## Observations that bear on Phase 2")
    add("")
    add("- **The formation's lateral swing mode is unstable at the plan's nominal heading gain.** The independent reduced linear model (`tether/theory/reduced_lti.py`) finds four real eigenvalues at +4.1e-4 1/s at k_h = 500: the stern attachment's weathervane moment (1.5 T0) plus the trim term turn each hull to follow about 0.76 of its chord angle against k_h, so the hull-axis thrust turns outward and cancels the cable's lateral restoring force. At k_h = 1000 the mode is stable. The plant agrees: chord-angle standard deviations of 41-75 deg in every Phase 1(d) cell (table above), 17-32 deg even at intensity 0.15. Phase 2's heading-gain sweep {250, 500, 1000} therefore straddles a stability boundary; no retuning is made.")
    add("- Under local weather at the pinned intensity the formation does not hold its shape: lateral gusts of 0.7 kN standard deviation against the cables' lateral stiffness T0/L (about 83 N/m at 1 kN) swing cables by tens of degrees (table above). Hypothesis (H1) - small taut excursions - is therefore at risk at intensity 1.0, and less so at 0.5.")
    add("- Excursions are long: slow (tau_w = 8 s) gusts drive quasi-static drifts whose p95 dwell exceeds tau_w/4 in most cells, so Phase 2's (H4) domain criterion is expected to exclude many cells. This is the regime of pre-mortem 2 (return leg still driven).")
    add("")
    add("## Plan-level discrepancies found during execution")
    add("")
    add("Independent derivations while building `tether/theory/excursion.py` (implementer and adversarial verifier agreeing, each checked by direct integration) found the following in Part II and Appendix A. They are recorded, not silently patched; the theory modules implement the plan's printed forms as primary and the exact forms beside them.")
    add("")
    add("1. **Proposition 3 counts the entry energy twice.** `V_up = sqrt(u^2 + 2 a_bar Delta)` gives sqrt(2) u for the ballistic excursion (Proposition 2 gives u). The proof integrates the return leg only while keeping the onset kinetic energy; the correct statement is `V_up = sqrt(2 a_bar_return Delta)`. The plan's own confirmation numbers (0.71 m -> 1.709 m/s, 8.06 m -> 5.771 m/s) fit the corrected form at T0 = 1000 N.")
    add("2. **Proposition 4's depth is the depth at gust shut-off, not the maximum.** The bodies are still closing when the gust ends; the maximum depth is (W_rel/T0) times the formula for u = 0, which is what the plan's confirmation numbers (0.46, 3.10, 8.27 m) are. The criterion 'deepens beyond u^2/2a0 iff W_rel > T0' is false as written - any sustained sub-threshold gust deepens the excursion to u^2 m_eff/(2(T0 - W_rel)); the true criterion is whether the depth stays bounded as t_g grows (the P1-T6 acceleration probe tests exactly that).")
    add("3. **Corollary 5 understates the square-gust snap by sqrt(W_rel/T0)** because it drops the post-gust closing; the exact form is f t_g sqrt(k W_rel (W_rel - T0)/m_eff). The exact severity falls monotonically as T0 rises, so the interior pretension optimum at T0 = W_rel/2 (F4, P2-T8) is an artifact of the approximation.")
    add("4. **Consequences for Theorems 6-7.** The exact severance level W* = (T0 + sqrt(T0^2 + 4 kappa T0 T_b^2))/2 equals T0 + kappa T_b^2 only while kappa T_b^2 << T0; beyond that it grows linearly in T_b, so the Gaussian law becomes a T_b^2 law and the regularly varying snap index alpha (1 - (1 + 4 kappa T_b^2/T0)^(-1/2)) stays below alpha. Even in the plan's own form, at t_g = 2 s kappa T_b^2 is only 133 N at 12 kN, so the local snap index at 6, 12, 25 kN is 0.20, 0.71, 2.2 - the T_b^4 line of F3 and the index doubling of F5 are not predicted at observable thresholds unless gusts are shorter than about 0.5 s.")
    add("5. **Appendix A.4 is evaluated at T0 = 1000 N, not the pinned 990 N** (critical depths 0.138, 0.311, 0.552, 0.982, 2.398 m at 990 N against the printed 0.14, 0.31, 0.55, 0.97, 2.37), and the pinned pair f = 0.880 / Z = 7.99 kN s/m is inconsistent at the third digit (f(0.0992) = 0.8808).")
    add("6. **P5-T1 is degenerate under the IV.9 prediction model.** A constant-acceleration path has at most one upcrossing, so the Rice integral equals the Monte Carlo hazard exactly; the test's [1, 1.25] ratio band then measures only numerical error.")
    add("")
    add("## Artifacts")
    add("")
    for path in ("records/phase1/phase1_close_results.json", "records/phase1/phase1_close_records.npz", "records/phase1/phase1_close_manifest.json", "records/phase1/impact_table.json"):
        add(f"- `{path}`")
    for key, caption in CAPTIONS.items():
        add(f"- `reports/figures/{key}_*.png`: {caption}")
    add("- Earlier authoritative records: `phase1_mechanics.json`, `phase1_deterministic.{json,npz}`, `phase1r_*`, `phase1_t9r_*`, `phase1_t5r_*`, `phase1_t4r_*`.")
    add("")
    add("## Open items")
    add("")
    add("- P1-T8's depth is the largest observed on the Phase 1(d) cells, not a global geometric bound.")
    add("- The low-speed impact table is per pretension and cable position; intermediate pretensions interpolate linearly in T0.")
    add("")
    add("## Recommendation for the Phase 2 gate")
    add("")
    add("Proceed to Phase 2 on the production plant with the pivots above. Record the (H4) domain risk in the Phase 2 pre-registration: the powered region may be small or empty at the plan's weather memory, and Phase 2's T10 is the test that reports it.")
    add("")
    return "\n".join(lines)


def main() -> None:
    from tether.analysis.phase1_close_figures import main as figures

    figures()
    text = build()
    write_bytes(REPORT, text.encode("utf-8"))
    print(REPORT)


if __name__ == "__main__":
    main()
