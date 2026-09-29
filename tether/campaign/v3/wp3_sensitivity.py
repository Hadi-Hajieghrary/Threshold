"""WP3 sensitivities (plan v3, addendum 4): the consistent-unit reading and the honest rho intervals.

    python -m tether.campaign.v3.wp3_sensitivity        # records/v3/wp3_sensitivity.json

Audit round 3 (``reports/v3/audit_round3.md``) found that the declared WP3 statistics mix two point
processes.  ``definitions.offspring`` makes an offspring an ONSET, so the kernel's entries are
"cross-cable offspring onsets per parent event"; but ``definitions.theta_runs`` counts EVENTS
("primary events / all events"), ``tests.T3.1``'s threshold measures ``nu_primary`` in events, and the
cluster forest has events as nodes.  Cross onsets outnumber child events by 2.090 pooled (1.28-3.73 per
cell), because a child event is counted once per triggering onset inside its burst.  A second, separate
defect gives the two sides of T3.1 different time bases: ``nu_primary`` is divided by the primary
exposure (the record less the 3 s post-re-engagement windows) and the measured rate by the full exposure.

Nothing here re-scores a declared test.  T3.1 and T3.2 keep their FAIL verdicts and T3.4 keeps branch
(b): the code implemented the declaration faithfully, and the declaration - not the code - is where the
two populations were mixed.  This module records, as never-scored sensitivities:

* the EVENT-UNIT reading: the same estimator on the same pilot population, counting child events rather
  than offspring onsets, with the rates on a single (full-exposure) time base;
* its POWER, stated explicitly: fitted and scored on the same events the rate law and theta are exact
  algebraic identities, so the out-of-sample reading below tests whether the pilot-fitted branching ratio
  transfers to the held-out statistics seeds - it does not test that the cascade is a branching process.
  Both numbers are emitted (``in_sample_ratio`` next to ``ratio``) so the gap between them is visible;
* the three honest rho intervals, because the published ``rho_ci95`` is degenerate: on the chosen
  (empirical) kernel path the kernel is a function of the pilot seeds alone, while the declared bootstrap
  resamples the statistics seeds, so all 2000 replicates are one number;
* which cells carry NO kernel support (their pilot seeds hold zero cross-cable offspring, so K is
  identically zero, rho = 0 and theta_pred = 1 exactly - an artefact, not a measurement);
* the statistics-seed kernel per cell, which is what the sweep's non-monotonicity turns on.

Records read: records/v3/cache/v3_stage{A,B}.pkl, records/v3/wp3_results.json,
records/v3/cascade_declarations.json.  Written: records/v3/wp3_sensitivity.json.
"""

from __future__ import annotations

import json
import math
import pickle

import numpy as np

from tether.campaign.common import json_bytes, sha256_file, source_state, write_bytes
from tether.campaign.v3 import campaign
from tether.evt._common import percentile_interval
from tether.theory import branching as B

OUT = campaign.RECORD_DIR / "wp3_sensitivity.json"
CACHES = {"A": campaign.RECORD_DIR / "cache" / "v3_stageA.pkl", "B": campaign.RECORD_DIR / "cache" / "v3_stageB.pkl"}
N_BOOT = 2000
FACTOR = campaign.FACTOR
THETA_BAND = 0.10


def _load_runs() -> dict[str, list[dict]]:
    by_cell: dict[str, list[dict]] = {}
    for path in CACHES.values():
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        for light, summary in zip(payload["jobs"], payload["summaries"]):
            by_cell.setdefault(light["cell"], []).append(summary)
    return by_cell


def event_kernel(runs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """K_ev[i, j] = mean child EVENTS on cable i per parent EVENT on cable j, and the parent counts per cable.

    The same estimator as the declared empirical kernel (mean offspring per ordered pair over the fitted
    population), differing only in what an offspring is: a child event of the cross-cable tree
    (``events.parent_event``) rather than each cross-cable onset attributed to the parent.
    """
    num = np.zeros((5, 5))
    den = np.zeros(5)
    for run in runs:
        events = run["events"]
        cable = np.asarray(events["cable_j"], dtype=int)
        parent = np.asarray(events["parent_event"], dtype=int)
        den += np.bincount(cable, minlength=5)
        for child, p in enumerate(parent):
            if p >= 0:
                num[cable[child], cable[p]] += 1.0
    return np.divide(num, den[None, :], out=np.zeros((5, 5)), where=den[None, :] > 0), den


def _event_rates(runs: list[dict]) -> dict:
    """Primary and total EVENT counts per cable and the full exposure of a set of runs."""
    primary = np.zeros(5)
    total = np.zeros(5)
    exposure = 0.0
    primary_exposure = 0.0
    for run in runs:
        events = run["events"]
        cable = np.asarray(events["cable_j"], dtype=int)
        flag = np.asarray(events["primary"], dtype=bool)
        total += np.bincount(cable, minlength=5)
        primary += np.bincount(cable[flag], minlength=5)
        exposure += float(run["meta"]["exposure_s"])
        primary_exposure += float(run["meta"]["primary_exposure_s"])
    return {"primary": primary, "total": total, "exposure": exposure, "primary_exposure": primary_exposure}


def _rho_intervals(pilots: list[dict], stats: list[dict], generator) -> dict:
    """The three honest readings of an interval on rho for the declared (onset-counting) empirical kernel.

    The published rho_ci95 is degenerate because the estimator ignores the resampled population; these are
    the constructions the audit found, each labelled with what it resamples.
    """
    def onset_kernel(runs):
        parents, offspring = [], []
        for run in runs:
            events = run["events"]
            parents.append(np.asarray(events["cable_j"], dtype=int))
            offspring.append(np.asarray(events["offspring_per_cable"]))
        if not parents:
            return np.zeros((5, 5))
        return np.nan_to_num(B.empirical_kernel(np.concatenate(parents), np.concatenate(offspring)), nan=0.0)

    out: dict = {}
    # (a) resample the PILOT seeds, which is the population the kernel is actually fitted on
    if pilots:
        reps = [B.spectral_radius(onset_kernel([pilots[k] for k in row]))
                for row in generator.integers(0, len(pilots), size=(N_BOOT, len(pilots)))]
        lo, hi = percentile_interval(np.array(reps), 0.95)
        out["pilot_seed_bootstrap"] = {"ci95": [float(lo), float(hi)], "resamples": f"{len(pilots)} pilot seeds",
                                       "note": f"only {len(pilots)} resampling units: at most {len(pilots) * (len(pilots) + 1) // 2} distinct replicates"}
    # (b) resample the pilot EVENTS (a finer unit than the declaration names, reported as a sensitivity)
    rows = []
    for run in pilots:
        events = run["events"]
        rows.append((np.asarray(events["cable_j"], dtype=int), np.asarray(events["offspring_per_cable"])))
    if rows:
        cable = np.concatenate([a for a, _ in rows])
        offspring = np.concatenate([b for _, b in rows])
        reps = []
        for row in generator.integers(0, cable.size, size=(N_BOOT, cable.size)):
            reps.append(B.spectral_radius(np.nan_to_num(B.empirical_kernel(cable[row], offspring[row]), nan=0.0)))
        arr = np.array(reps)
        lo, hi = percentile_interval(arr, 0.95)
        out["pilot_event_bootstrap"] = {"ci95": [float(lo), float(hi)], "resamples": f"{int(cable.size)} pilot events",
                                        "share_above_one": float(np.mean(arr >= 1.0))}
    # (c) refit the same estimator on the STATISTICS seeds, with a seed bootstrap over them
    if stats:
        point = B.spectral_radius(onset_kernel(stats))
        reps = [B.spectral_radius(onset_kernel([stats[k] for k in row]))
                for row in generator.integers(0, len(stats), size=(N_BOOT, len(stats)))]
        lo, hi = percentile_interval(np.array(reps), 0.95)
        out["statistics_seed_refit"] = {"rho": float(point), "ci95": [float(lo), float(hi)],
                                        "resamples": f"{len(stats)} statistics seeds"}
    return out


def _cell(name: str, runs: list[dict], recorded: dict) -> dict:
    pilots = [r for r in runs if r["meta"]["pilot"]]
    stats = [r for r in runs if not r["meta"]["pilot"]]
    generator = np.random.default_rng(campaign.BOOTSTRAP_SEED)
    K_ev, parent_counts = event_kernel(pilots)
    rates = _event_rates(stats)
    primary, total = rates["primary"], rates["total"]
    n_primary, n_events = float(primary.sum()), float(total.sum())
    pi = primary / n_primary if n_primary > 0 else np.full(5, 0.2)
    # the pilot seeds' own cross-cable support: with none, K is identically zero and theta_pred is 1 by construction
    pilot_children = sum(int(np.sum(np.asarray(r["events"]["parent_event"]) >= 0)) for r in pilots)
    pilot_cross_onsets = sum(int(np.sum(np.asarray(r["events"]["n_cross"]))) for r in pilots)
    support = pilot_cross_onsets >= campaign.MIN_OFFSPRING
    rho_ev = float(B.spectral_radius(K_ev))
    out = {
        "cell": name,
        "kernel_support": {"pilot_events": int(parent_counts.sum()), "pilot_child_events": pilot_children,
                           "pilot_cross_onsets": pilot_cross_onsets, "has_support": bool(support),
                           "rule": f"pilot_cross_onsets >= {campaign.MIN_OFFSPRING} (the declared minimum-offspring rule applied to the FITTED population, which the declaration applies only to the target)",
                           "note": (None if support else
                                    ("pilot seeds hold NO cross-cable offspring at all: K is identically zero, rho = 0 and "
                                     "theta_pred = 1 exactly; an artefact of the fitted population, not a measurement"
                                     if pilot_cross_onsets == 0 else
                                     f"the pilot seeds hold only {pilot_cross_onsets} cross-cable offspring, below the declared "
                                     f"minimum of {campaign.MIN_OFFSPRING}: K is fitted on too little to carry an interval, but it "
                                     f"is not degenerate"))},
        "_cross_onsets": sum(int(np.sum(np.asarray(r["events"]["n_cross"]))) for r in stats),
        "_child_events": sum(int(np.sum(np.asarray(r["events"]["parent_event"]) >= 0)) for r in stats),
        "onsets_per_child_event": float(sum(int(np.sum(np.asarray(r["events"]["n_cross"]))) for r in stats)
                                        / max(sum(int(np.sum(np.asarray(r["events"]["parent_event"]) >= 0)) for r in stats), 1)),
        "event_units": {"K": K_ev.tolist(), "rho": rho_ev, "m_bar": float(pi @ K_ev.sum(axis=0))},
        "rho_intervals_declared_kernel": _rho_intervals(pilots, stats, generator),
    }
    if rho_ev < 1.0 and n_events > 0:
        nu_primary_full = primary / max(rates["exposure"], 1e-9)
        nu_total_measured = total / max(rates["exposure"], 1e-9)
        predicted = B.total_rate(K_ev, nu_primary_full)
        ratio = float(predicted.sum() / nu_total_measured.sum()) if nu_total_measured.sum() > 0 else math.nan
        theta_pred = float(B.extremal_index(K_ev, pi))
        theta_runs = n_primary / n_events
        # the power statement: the same construction fitted AND scored on the statistics seeds is an identity
        K_in, _ = event_kernel(stats)
        in_ratio = math.nan
        in_theta = math.nan
        if B.spectral_radius(K_in) < 1.0:
            in_ratio = float(B.total_rate(K_in, nu_primary_full).sum() / nu_total_measured.sum())
            in_theta = float(B.extremal_index(K_in, pi))
        onsets = float(sum(np.asarray(r["onsets"]["time"]).size for r in stats)) / max(rates["exposure"], 1e-9)
        out["event_units"].update({
            "ratio_equals_theta_ratio_residual": abs(ratio - theta_runs / theta_pred) if theta_pred else math.nan,
            "null_kernel_ratio": theta_runs,
            "null_kernel_in_band": bool(1.0 / FACTOR <= theta_runs <= FACTOR),
            "measured_total_onset_rate_per_cable_s": onsets / 5.0,
            "nu_primary_per_cable_s_full_exposure": nu_primary_full.tolist(),
            "nu_total_measured_per_cable_s": nu_total_measured.tolist(),
            "nu_total_predicted_per_cable_s": predicted.tolist(),
            "ratio": ratio, "ratio_in_band": bool(1.0 / FACTOR <= ratio <= FACTOR) if np.isfinite(ratio) else None,
            "theta_pred": theta_pred, "theta_runs": theta_runs,
            "theta_within_band": bool(abs(theta_pred - theta_runs) <= THETA_BAND),
            "in_sample_ratio": in_ratio, "in_sample_theta_pred": in_theta,
            "power_note": ("(i) Fitted and scored on the same events the rate law and theta are exact identities "
                           "(in_sample_ratio = 1 and in_sample_theta_pred = theta_runs to machine precision), because the "
                           "parent-count normalisation conserves the child count per cable for any labelled forest; no data "
                           "could violate it. (ii) On a single time base the two readings are ONE statistic, not two: "
                           "ratio = theta_runs / theta_pred exactly (residual above), so the rate-law reading and the theta "
                           "reading are the same comparison at different tolerances and must never be presented as two "
                           "independent confirmations. (iii) The band is lax for this quantity: the strictly null kernel "
                           "K = 0 predicts ratio = theta_runs and lands in band in 4 of 20 cells (null_kernel_in_band), so "
                           "a high in-band count is a weak pass even for the transfer claim. (iv) What the out-of-sample "
                           "numbers therefore test is whether the pilot-fitted branching ratio transfers to the held-out "
                           "seeds - not whether the cascade is a branching process, and not a repaired T3.1."),
        })
    else:
        out["event_units"]["reason_unscored"] = "rho_ev >= 1 or no events"
    # the exposure sensitivity on the DECLARED (onset-counting) kernel, to separate the two defects
    declared_K = np.array(recorded["K"], dtype=float) if "K" in recorded else None
    if declared_K is not None and recorded.get("rho", 1.0) < 1.0:
        nu_p_full = primary / max(rates["exposure"], 1e-9)
        nu_t = total / max(rates["exposure"], 1e-9)
        r_full = float(B.total_rate(declared_K, nu_p_full).sum() / nu_t.sum()) if nu_t.sum() > 0 else math.nan
        onset_rate = float(sum(np.asarray(r["onsets"]["time"]).size for r in stats)) / max(rates["exposure"], 1e-9) / 5.0
        declared_pred = np.array(recorded["nu_total_predicted_per_cable_s"], dtype=float)
        r_onset = float(declared_pred.sum() / (onset_rate * 5.0)) if onset_rate > 0 else math.nan
        out["declared_kernel_full_exposure"] = {
            "ratio": r_full, "ratio_in_band": bool(1.0 / FACTOR <= r_full <= FACTOR) if np.isfinite(r_full) else None,
            "exposure_over_primary_exposure": float(rates["exposure"] / max(rates["primary_exposure"], 1e-9)),
            "note": "the time-base fix alone, with the declared onset-counting kernel: it does not flip T3.1",
        }
        # M11: T3.1's STATEMENT names the measured total onset rate; the code compared against the event rate
        out["declared_kernel_onset_target"] = {
            "measured_total_onset_rate_per_cable_s": onset_rate,
            "ratio": r_onset, "ratio_in_band": bool(1.0 / FACTOR <= r_onset <= FACTOR) if np.isfinite(r_onset) else None,
            "recorded_event_target_ratio": recorded.get("rate_ratio_pooled"),
            "note": "both readings FAIL the factor-1.5 band; the headline ratio differs by about 2.2x between them",
        }
    return out


def build() -> dict:
    recorded = json.loads((campaign.RECORD_DIR / "wp3_results.json").read_text())
    by_cell = _load_runs()
    cells_out = {}
    for name, runs in sorted(by_cell.items()):
        rec = recorded["cells"].get(name, {})
        if "rho" not in rec:
            continue
        cells_out[name] = _cell(name, runs, rec)
    scored = [c for c in cells_out.values() if "ratio" in c.get("event_units", {})]
    with_support = [c for c in scored if c["kernel_support"]["has_support"]]
    pooled_cross = sum(c["kernel_support"]["pilot_cross_onsets"] for c in cells_out.values())
    summary = {
        "schema": "v3-wp3-sensitivity-1",
        "status": ("SENSITIVITY (addendum 4): reported beside the declared WP3 verdicts, never scored. "
                   "T3.1 and T3.2 keep their FAIL verdicts and T3.4 keeps branch (b)."),
        "why": ("audit round 3 (reports/v3/audit_round3.md) found that the declared kernel counts cross-cable "
                "offspring ONSETS per parent EVENT while theta_runs, the cluster law, the event forest and both "
                "rates in T3.1 count EVENTS, and that T3.1's two sides use different time bases"),
        "definitions": {
            "event_units.K": "mean child EVENTS on cable i per parent EVENT on cable j, fitted on the pilot seeds "
                             "(the same population and normalisation as the declared empirical kernel)",
            "event_units.ratio": "sum (I - K_ev)^-1 nu_primary / sum nu_total_measured, both rates per cable per second "
                                 "over the FULL exposure of the statistics seeds (a single time base)",
            "in_sample_ratio": "the same quantity with K_ev refitted on the statistics seeds it is scored against; it is "
                               "an algebraic identity and is emitted to make the out-of-sample reading's power visible",
            "rho_intervals_declared_kernel": "three constructions of an interval on the DECLARED onset-counting kernel's "
                                             "rho, because the published rho_ci95 is degenerate on the empirical path",
        },
        "pooled": {
            "cells": len(cells_out),
            "onsets_per_child_event_mean_of_cells": float(np.mean([c["onsets_per_child_event"] for c in cells_out.values()])),
            "onsets_per_child_event_pooled": float(sum(c["_cross_onsets"] for c in cells_out.values()) / max(sum(c["_child_events"] for c in cells_out.values()), 1)),
            "null_kernel_ratio_in_band": sum(bool(c["event_units"].get("null_kernel_in_band")) for c in cells_out.values()),
            "max_ratio_equals_theta_ratio_residual": float(max([c["event_units"].get("ratio_equals_theta_ratio_residual", 0.0) or 0.0 for c in cells_out.values()])),
            "theta_pred_range_over_supported_cells": [
                float(min(c["event_units"]["theta_pred"] for c in cells_out.values() if c["kernel_support"]["has_support"] and "theta_pred" in c["event_units"])),
                float(max(c["event_units"]["theta_pred"] for c in cells_out.values() if c["kernel_support"]["has_support"] and "theta_pred" in c["event_units"]))],
            "cells_with_kernel_support": len(with_support),
            "event_units_ratio_in_band": sum(bool(c["event_units"].get("ratio_in_band")) for c in scored),
            "event_units_theta_within_band": sum(bool(c["event_units"].get("theta_within_band")) for c in scored),
            "scored_in_event_units": len(scored),
            "pilot_cross_onsets_total": pooled_cross,
        },
        "cells": cells_out,
        "declared_verdicts_unchanged": {k: recorded["tests"][k].get("verdict", recorded["tests"][k].get("branch"))
                                        for k in ("T3.1", "T3.2", "T3.3", "T3.4", "T3.5")},
        "declarations_sha256": sha256_file(campaign.DECLARATIONS_PATH),
        "wp3_results_sha256": sha256_file(campaign.RECORD_DIR / "wp3_results.json"),
        "addenda_sha256": {p.name: sha256_file(p) for p in sorted(campaign.RECORD_DIR.glob("cascade_addendum_*.json"))},
        "source": source_state(),
    }
    return summary


def main() -> None:
    out = build()
    write_bytes(OUT, json_bytes(out))
    print(json.dumps({"pooled": out["pooled"], "declared_verdicts_unchanged": out["declared_verdicts_unchanged"]}))


if __name__ == "__main__":
    main()
