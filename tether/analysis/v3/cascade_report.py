"""Plan v3 report (cascade criticality of cable-towed fleets), rendered from the committed records alone.

    python -m tether.analysis.v3.cascade_report      # renders reports/v3/cascade_report.md

Every number below is read from a record under ``records/v3/`` or hashed from a file on disk;
none is typed into the generator.  A record that does not exist yet is reported as "not yet run",
so the report renders at every stage of the campaign: before stage A finishes it says so, after
WP1 it carries the T1.x results, and so on.  Forecasts (``cascade_predictions.json`` and the
committed predictions inside the declarations) are labelled FORECAST and are never placed in a
measured column.  ``reports/v3/open_items.md`` and ``reports/v3/discussion.md`` are inserted
verbatim when they exist; the discussion file is expected to carry the two subsections
``### Limits of this evidence`` and ``### What would change the verdict``.

Record layouts are those of the producing code (``tether/campaign/v3/campaign.py``,
``analyse.py``, ``analyse_wp2.py``, ``analyse_wp3.py``), which is the law.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from tether.campaign.common import RECORDS, REPORTS, ROOT, sha256_file, write_bytes

RECORD_DIR = RECORDS / "v3"
REPORT_DIR = REPORTS / "v3"

DECLARATIONS = RECORD_DIR / "cascade_declarations.json"
PREDICTIONS = RECORD_DIR / "cascade_predictions.json"
WP1_RESULTS = RECORD_DIR / "wp1_results.json"
WP2_RESULTS = RECORD_DIR / "wp2_results.json"
WP3_RESULTS = RECORD_DIR / "wp3_results.json"
WP1_TRANSMISSION = RECORD_DIR / "wp1_transmission.npz"
WP2_OFFSPRING = RECORD_DIR / "wp2_offspring.npz"
WP2_INTERVENTION = RECORD_DIR / "wp2_intervention.npz"
WP3_CLUSTER_SIZES = RECORD_DIR / "wp3_cluster_sizes.npz"
WP1_SENSITIVITY = RECORD_DIR / "wp1_sensitivity.json"   # addendum 2: reported beside T1.1-T1.3, never scored
GATES = {1: RECORD_DIR / "gate_wp1.json", 2: RECORD_DIR / "gate_wp2.json", 3: RECORD_DIR / "gate_wp3.json"}
WP3_NOT_LAUNCHED = RECORD_DIR / "wp3_not_launched.json"
B_EXT_AUTHORISED = RECORD_DIR / "b_ext_authorised.json"
ADDENDUM_GLOB = "cascade_addendum_*.json"
# tether.campaign.v3.compute.STAGE_CACHE (a stage's ``.partial.pkl`` means it is still running).
STAGE_CACHES = {"A": RECORD_DIR / "cache" / "v3_stageA.pkl", "B": RECORD_DIR / "cache" / "v3_stageB.pkl",
                "B_ext": RECORD_DIR / "cache" / "v3_stageB_ext.pkl"}
FIGURE_DIR, FIGURE_GLOB = ROOT / "Paper" / "Figures", "fig_v3_*.pdf"
CLIP = ROOT / "Presentation" / "clips" / "intervention.mp4"
EXPLORER = REPORT_DIR / "cascade_explorer.html"
OPEN_ITEMS = REPORT_DIR / "open_items.md"
DISCUSSION = REPORT_DIR / "discussion.md"
REPORT = REPORT_DIR / "cascade_report.md"

# The declared test ids in plan order; their content (statement, threshold, prediction) is read
# from the declarations record, this tuple only fixes the order when that record is absent.
TEST_IDS = ("T1.0", "T1.1", "T1.2", "T1.3", "T1.4", "T2.1", "T2.2", "T2.3", "T2.4", "T2.5",
            "T3.1", "T3.2", "T3.3", "T3.4", "T3.5")
WORK_PACKAGE_RESULTS = {"T1": WP1_RESULTS, "T2": WP2_RESULTS, "T3": WP3_RESULTS}
STAGES = ("A", "B", "B_ext")
NOT_RUN = "not yet run"
DISCUSSION_SUBSECTIONS = ("### Limits of this evidence", "### What would change the verdict")


# ----------------------------------------------------------------------------- formatting


def _load(path: Path):
    """The parsed record, or None when it does not exist yet (NaN literals parse to float nan)."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _digest(path: Path, chars: int = 16) -> str:
    return f"`{sha256_file(path)[:chars]}...`" if path.exists() else "absent"


def _prefix(digest, chars: int = 16) -> str:
    return f"`{digest[:chars]}...`" if isinstance(digest, str) and digest else "-"


def _fmt(value, sig: int = 3) -> str:
    """Three significant figures; scientific notation outside [1e-3, 1e5); ints exact; '-' for absent."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _esc(str(value))
    if not math.isfinite(v):
        return "n/a"
    if v == 0.0:
        return "0"
    if abs(v) >= 1e5 or abs(v) < 1e-3:
        return f"{v:.{sig - 1}e}"
    exponent = math.floor(math.log10(abs(v)))
    rounded = round(v, sig - 1 - exponent)
    decimals = max(sig - 1 - exponent, 0)
    return f"{rounded:.{decimals}f}"


def _rate(value, sig: int = 3) -> str:
    """Rates (per cable-second) always in scientific notation."""
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _esc(str(value))
    return "n/a" if not math.isfinite(v) else f"{v:.{sig - 1}e}"


def _count(value) -> str:
    """A count stored as a float is printed exactly."""
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _esc(str(value))
    if not math.isfinite(v):
        return "n/a"
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else _fmt(v)


def _interval(pair, formatter=_fmt) -> str:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return "-"
    return f"[{formatter(pair[0])}, {formatter(pair[1])}]"


def _rates(values) -> str:
    return " / ".join(_rate(v) for v in values) if isinstance(values, list) else "-"


def _bold(text) -> str:
    return f"**{text}**" if text not in (None, "") else "**-**"


def _esc(text) -> str:
    """A table cell: no pipes, no line breaks."""
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _pct(value) -> str:
    try:
        return f"{100.0 * float(value):.0f} %"
    except (TypeError, ValueError):
        return "-"


def _source(record) -> str:
    if not isinstance(record, dict) or not isinstance(record.get("source"), dict):
        return "-"
    src = record["source"]
    revision = src.get("revision")
    return f"`{revision[:12]}` ({src.get('status', '-')})" if revision else f"- ({src.get('status', '-')})"


def _table(add, header: list[str], rows: list[list[str]], align: str | None = None) -> None:
    """A Markdown table; ``align`` is one of '-', 'r', 'c' per column (left by default)."""
    marks = {"-": "---", "r": "---:", "c": ":---:"}
    add("| " + " | ".join(header) + " |")
    add("|" + "|".join(marks[a] for a in (align or "-" * len(header))) + "|")
    for row in rows:
        add("| " + " | ".join(row) + " |")


def _at(values, k: int):
    return values[k] if isinstance(values, list) and k < len(values) else None


def _cell_order(declarations) -> list[str]:
    """Cells in declared order (stage A, B, B_ext); unknown names sort after them."""
    if not declarations:
        return []
    return [name for stage in STAGES for name in (declarations.get("cells") or {}).get(stage, {})]


def _ordered(cells: dict, declarations) -> list[str]:
    order = {name: k for k, name in enumerate(_cell_order(declarations))}
    return sorted(cells, key=lambda n: (order.get(n, len(order)), n))


def _declared_cell(declarations, name: str) -> dict:
    for stage in STAGES:
        block = ((declarations or {}).get("cells") or {}).get(stage, {})
        if name in block:
            return {"stage": stage, **block[name]}
    return {}


def _stage_state(stage: str) -> str:
    cache = STAGE_CACHES[stage]
    if cache.exists():
        return "complete"
    if cache.with_suffix(".partial.pkl").exists():
        return "in progress (partial cache present)"
    return "not started"


def _addenda() -> list[tuple[Path, dict]]:
    out = []
    for path in sorted(RECORD_DIR.glob(ADDENDUM_GLOB)):
        try:
            out.append((path, json.loads(path.read_text())))
        except (OSError, ValueError) as error:  # a half-written file must not stop the report
            out.append((path, {"error": str(error)}))
    return out


def _share_line(test: dict, min_scored) -> str:
    verdict = _bold(test.get("verdict"))
    if "scored_cells" in test:
        line = (f"{verdict}: {_count(test.get('passing_cells'))} of {_count(test.get('scored_cells'))} scored cells pass "
                f"(required share {_pct(test.get('share_required'))}; fewer than {_fmt(min_scored)} scored cells is UNDER-POWERED)")
    else:
        line = verdict
    if test.get("reason"):
        line += f"; {test['reason']}"
    return line


# ----------------------------------------------------------------------------- sections


def _status(add, ctx) -> None:
    decl, pred, wp1, wp2, wp3 = ctx["decl"], ctx["pred"], ctx["wp1"], ctx["wp2"], ctx["wp3"]
    add("## Status")
    add("")
    if wp3 is not None:
        stage = f"WP3 scored on stages {', '.join(wp3.get('stages', []))}; the campaign's records are complete up to the gate."
    elif ctx["not_launched"] is not None:
        stage = "WP2 scored; WP3 was not launched (no kernel passed its factor test)."
    elif wp2 is not None:
        stage = f"WP2 scored; stage B is {_stage_state('B')}."
    elif wp1 is not None:
        stage = "WP1 scored; WP2 not yet analysed."
    elif STAGE_CACHES["A"].exists():
        stage = "Stage A computed; WP1 not yet analysed."
    elif _stage_state("A").startswith("in progress"):
        stage = "Stage A is running; no test has been scored."
    elif decl is not None:
        stage = "Declared and predicted; no plant run has started."
    else:
        stage = "Not yet declared."
    add(f"**{stage}**")
    add("")
    rows = []
    for label, path in (("Declarations", DECLARATIONS), ("Committed predictions", PREDICTIONS),
                        ("WP1 results", WP1_RESULTS), ("WP2 results", WP2_RESULTS), ("WP3 results", WP3_RESULTS),
                        ("Gate WP1", GATES[1]), ("Gate WP2", GATES[2]), ("Gate WP3", GATES[3]),
                        ("WP3 not-launched record", WP3_NOT_LAUNCHED), ("B_ext authorisation", B_EXT_AUTHORISED)):
        rows.append([label, f"`{_rel(path)}`", "present" if path.exists() else NOT_RUN])
    for stage in STAGES:
        rows.append([f"Stage {stage} plant cache", f"`{_rel(STAGE_CACHES[stage])}`", _stage_state(stage)])
    addenda = ctx["addenda"]
    rows.append(["Addenda", f"`records/v3/{ADDENDUM_GLOB}`", f"{len(addenda)} written" if addenda else "none"])
    _table(add, ["Record", "Path", "State"], rows)
    add("")
    if decl is not None and pred is not None:
        declared = decl.get("predictions_sha256")
        on_disk = sha256_file(PREDICTIONS)
        add(f"The declarations pin the predictions at `{(declared or '')[:16]}...`; the file on disk hashes to `{on_disk[:16]}...` "
            f"({'unchanged since declaration' if declared == on_disk else 'DRIFTED: the predictions file was changed after it was declared'}).")
        add("")


def _gates(add, ctx) -> None:
    add("## Gate verdicts")
    add("")
    any_gate = False
    for n in (1, 2, 3):
        gate = ctx["gates"][n]
        if gate is None:
            add(f"- `{_rel(GATES[n])}`: {NOT_RUN}.")
            continue
        any_gate = True
        add(f"- **WP{n}: {gate.get('verdict', '-')}** (`{_rel(GATES[n])}`, schema `{gate.get('schema', '-')}`).")
        add(f"  - Rule: {gate['rule'] if isinstance(gate.get('rule'), str) else '; '.join(gate.get('rule') or []) or '-'}")
        add(f"  - Blocking tests in scope: {', '.join(gate.get('blocking_in_scope') or []) or 'none'}.")
        tests = gate.get("tests") or {}
        add("  - Test verdicts in the gate record: " + (", ".join(f"{t} {_bold(v.get('verdict'))}" for t, v in tests.items()) or "none") + ".")
        failing = gate.get("failing_blocking_tests") or []
        add(f"  - Failing blocking tests: {', '.join(failing) if failing else 'none'}.")
        consequences = gate.get("consequences") or []
        add("  - Consequences:" + ("" if consequences else " none recorded."))
        for item in consequences:
            add(f"    - {item}")
        provenance = gate.get("provenance") or {}
        add(f"  - Provenance: results `{provenance.get('results', '-')}` {_prefix(provenance.get('results_sha256'))}; audit {provenance.get('audit', '-')}; "
            f"declarations {_prefix(gate.get('declarations_sha256'))}; source {_source(gate)}.")
    nl = ctx["not_launched"]
    if nl is not None:
        any_gate = True
        add(f"- **WP3 NOT LAUNCHED** (`{_rel(WP3_NOT_LAUNCHED)}`): {nl.get('reason', '-')}.")
        tested = nl.get("kernel_path") or {}
        if tested:
            add("  - Kernel paths tested: " + ", ".join(f"{k} {_bold(v.get('verdict'))} ({_count(v.get('passing_cells'))}/{_count(v.get('scored_cells'))} scored cells)" for k, v in tested.items()) + ".")
        add(f"  - Source {_source(nl)}.")
    if not any_gate:
        add("")
        add("No gate record has been written; the verdicts above are pending the analysis and audit of each work package.")
    add("")


def _provenance(add, ctx) -> None:
    decl, pred, wp1 = ctx["decl"], ctx["pred"], ctx["wp1"]
    add("## Provenance")
    add("")
    rows = [["Declarations", f"`{_rel(DECLARATIONS)}`", _digest(DECLARATIONS), _source(decl) if decl else NOT_RUN]]
    if decl is not None:
        rows.append(["Committed predictions (as declared)", f"`{_rel(PREDICTIONS)}`", _prefix(decl.get("predictions_sha256")), _source(pred) if pred else "absent"])
        rows.append(["Committed predictions (on disk)", f"`{_rel(PREDICTIONS)}`", _digest(PREDICTIONS), "-"])
        rows.append(["Theory document", f"`{decl.get('theory', '-')}`", _prefix(decl.get("theory_sha256")), "-"])
        rows.append(["Plan document", f"`{decl.get('plan', '-')}`", _prefix(decl.get("plan_sha256")), "-"])
        for name, digest in sorted((decl.get("input_sha256") or {}).items()):
            rows.append([f"Input `{name}`", "-", _prefix(digest), "-"])
        for name, digest in sorted((decl.get("module_sha256") or {}).items()):
            rows.append([f"Module `{name}`", "-", _prefix(digest), "-"])
    else:
        rows.append(["Committed predictions", f"`{_rel(PREDICTIONS)}`", _digest(PREDICTIONS), _source(pred) if pred else NOT_RUN])
    declared_digest = sha256_file(DECLARATIONS) if DECLARATIONS.exists() else None
    for label, path, record in (("WP1 results", WP1_RESULTS, wp1), ("WP2 results", WP2_RESULTS, ctx["wp2"]), ("WP3 results", WP3_RESULTS, ctx["wp3"])):
        if record is None:
            rows.append([label, f"`{_rel(path)}`", NOT_RUN, "-"])
            continue
        scored_against = record.get("declarations_sha256")
        note = "" if declared_digest is None or scored_against is None else (" (scored against the declared file)" if scored_against == declared_digest else " (scored against a DIFFERENT declarations file)")
        rows.append([label + note, f"`{_rel(path)}`", _digest(path), _source(record)])
    for n in (1, 2, 3):
        gate = ctx["gates"][n]
        rows.append([f"Gate WP{n}", f"`{_rel(GATES[n])}`", _digest(GATES[n]) if gate else NOT_RUN, _source(gate) if gate else "-"])
    for path, record in ctx["addenda"]:
        rows.append([f"Addendum {record.get('addendum', '?')}", f"`{_rel(path)}`", _digest(path), _source(record)])
    _table(add, ["Item", "Path", "sha256", "Source revision (status)"], rows)
    add("")
    if wp1 is not None and isinstance(wp1.get("compute"), dict):
        c = wp1["compute"]
        add(f"Stage A compute (`{_rel(WP1_RESULTS)}`, `compute`): {_count(c.get('runs'))} runs, {_fmt(c.get('core_hours_run'))} core-hours of plant time "
            f"and {_fmt(c.get('core_hours_reduce'))} core-hours of reduction ({_fmt((c.get('core_hours_run') or 0.0) + (c.get('core_hours_reduce') or 0.0))} in total).")
    else:
        add(f"Compute: {NOT_RUN} (core-hours are read from `{_rel(WP1_RESULTS)}`).")
    add("")


def _declarations(add, ctx) -> None:
    decl = ctx["decl"]
    add("## Declarations fixed before the campaign")
    add("")
    if decl is None:
        add(f"`{_rel(DECLARATIONS)}`: {NOT_RUN}.")
        add("")
        return
    add(f"Schema `{decl.get('schema', '-')}`. {decl.get('rule', '')}")
    add("")
    cal = decl.get("calibration_cell") or {}
    add(f"- **Calibration cell ruling.** Cell `{cal.get('cell', '-')}`, fitted on seeds {', '.join(str(s) for s in cal.get('fit_seeds', []))}. {cal.get('ruling', '')}")
    seeds = decl.get("seeds") or {}
    stats = seeds.get("statistics") or []
    add(f"- **Seeds.** Pilot {', '.join(str(s) for s in seeds.get('pilot', []))} (never scored); statistics "
        f"{(str(stats[0]) + '-' + str(stats[-1])) if stats else '-'} ({len(stats)} seeds); common random numbers across cells: {_fmt(seeds.get('common_random_numbers'))}.")
    run = decl.get("run") or {}
    add(f"- **Run settings.** {_fmt(run.get('duration_s'))} s after a {_fmt(run.get('warmup_s'))} s warm-up; weather {run.get('weather', '-')}; cable mode {run.get('cable_mode', '-')}; physics step {_fmt(run.get('physics_step_s'))} s.")
    stages = decl.get("stages") or {}
    add("- **Stages.** " + " ".join(f"{k}: {v}." for k, v in stages.items()))
    pair = decl.get("stiffness_pair") or {}
    factors = pair.get("factors") or {}
    add(f"- **Stiffness pair.** Factors {', '.join(f'{k} = {_fmt(v)}' for k, v in factors.items())}; damping {pair.get('damping', '-')}. Why: {pair.get('why_4x', '-')}")
    elig = decl.get("eligibility") or {}
    add("- **Eligibility.** " + "; ".join(f"{k} = {_interval(v) if isinstance(v, list) and len(v) == 2 else _fmt(v)}" for k, v in elig.items()) + ".")
    iv = decl.get("intervention") or {}
    add("- **Intervention block.** " + "; ".join(f"{k}: {_fmt(v) if not isinstance(v, str) else v}" for k, v in iv.items()) + ".")
    add("- **Kernel-path rule.**")
    for line in decl.get("kernel_path_rule") or []:
        add(f"  - {line}")
    add(f"- **Operable rule.** {decl.get('operable_rule', '-')}.")
    add(f"- **Contingency rule.** {decl.get('contingency_rule', '-')}.")
    boot = decl.get("bootstrap") or {}
    add(f"- **Bootstrap.** seed {boot.get('seed', '-')}, {_fmt(boot.get('n_boot'))} replicates, resampling {boot.get('resampling', '-')}.")
    add(f"- **Audit.** {decl.get('audit', '-')}")
    add("")
    add("**Declared cells** (configuration hashes are `FleetRunSpec.config_hash`):")
    add("")
    rows = []
    for stage in STAGES:
        for name, spec in (decl.get("cells") or {}).get(stage, {}).items():
            rows.append([stage, f"`{name}`", str(spec.get("formation", "-")), _fmt(spec.get("pretension")), _fmt(spec.get("heading_gain")),
                         _fmt(spec.get("intensity")), _fmt(spec.get("stiffness_factor")), _prefix(spec.get("config_hash"), 12)])
    _table(add, ["Stage", "Cell", "Formation", "T0 [N]", "k_h [N m/rad]", "Intensity", "Stiffness factor", "Config hash"], rows)
    add("")
    sources = decl.get("forecast_sources") or {}
    if sources:
        add("Forecast source cells (a cell without a v1 record borrows its parent-peak law and primary counts from): "
            + ", ".join(f"`{k}` <- `{v}`" for k, v in sources.items()) + ".")
        add("")
    add("**Definitions** (verbatim from the record):")
    add("")
    for key, value in (decl.get("definitions") or {}).items():
        add(f"- *{key}*: {value}")
    add("")
    add("**Declared tests:**")
    add("")
    rows = []
    for tid, spec in (decl.get("tests") or {}).items():
        extra = f" (conditional on {spec['conditional_on']})" if spec.get("conditional_on") else ""
        rows.append([f"**{tid}**", "yes" if spec.get("blocking") else "no", str(spec.get("work_package", "-")) + extra,
                     _esc(spec.get("statement", "-")), _esc(spec.get("threshold", "-")),
                     ("FORECAST: " + _esc(spec["committed_prediction"])) if spec.get("committed_prediction") else "-"])
    _table(add, ["Test", "Blocking", "WP", "Statement", "Threshold", "Committed prediction (FORECAST)"], rows)
    add("")


def _predictions(add, ctx) -> None:
    pred, decl = ctx["pred"], ctx["decl"]
    add("## Committed predictions (forecasts, never results)")
    add("")
    if pred is None:
        add(f"`{_rel(PREDICTIONS)}`: {NOT_RUN}.")
        add("")
        return
    add(f"Schema `{pred.get('schema', '-')}`: {pred.get('what', '')}. Inputs pinned: "
        + ", ".join(f"{k} {_prefix(v)}" for k, v in (pred.get("inputs_sha256") or {}).items()) + f"; source {_source(pred)}. Every number in this section is a FORECAST.")
    add("")
    cf = pred.get("closed_form") or {}
    add("### Closed form (FORECAST)")
    add("")
    add(f"- Transmission ratio {_fmt(cf.get('ratio'))} (half-sine pulse); {_fmt(cf.get('ratio_rectangular'))} with a rectangular pulse; the paper's band {_interval(cf.get('band_paper'))}; "
        f"unloading threshold {_fmt(cf.get('threshold_T0'))} T0.")
    freq = cf.get("frequencies") or {}
    if freq:
        add("- Frequencies: " + ", ".join(f"{k} = {_fmt(v)}" for k, v in freq.items()) + ".")
    add("")
    add("### Per-cell forecasts (FORECAST)")
    add("")
    add("The exact-response band statistic is the median off-diagonal entry of R_ij / geometry_factor_ij (`transmission.band_statistic_median_offdiag`); "
        "the branching numbers come from the kernel built on R, the v1 taut-tension margin law and v1 parent peaks (`branching.*`). "
        "A cell marked *extrapolated* borrows its parents from another v1 cell (`forecast_source_cell`).")
    add("")
    rows = []
    for name in _ordered(pred.get("cells") or {}, decl):
        cell = pred["cells"][name]
        tr, br = cell.get("transmission") or {}, cell.get("branching") or {}
        extrapolated = br.get("extrapolated")
        rows.append([f"`{name}`", str(cell.get("stage", "-")), str(tr.get("formation", "-")), _fmt(tr.get("band_statistic_median_offdiag")),
                     _fmt(br.get("rho")), _fmt(br.get("theta_pred")), _fmt(br.get("m_bar")), _fmt(br.get("offspring_count_forecast")),
                     _fmt(br.get("offspring_per_event_forecast")), _count((br.get("parents") or {}).get("n_events")),
                     f"`{cell.get('forecast_source_cell') or '-'}`" + (" (extrapolated)" if extrapolated else ""),
                     "no forecast" if "rho" not in br else ("extrapolated" if extrapolated else "own v1 record")])
    _table(add, ["Cell", "Stage", "Formation", "Band statistic (FORECAST)", "rho (FORECAST)", "theta_pred (FORECAST)", "m_bar (FORECAST)",
                 "Offspring count (FORECAST)", "Offspring per event (FORECAST)", "v1 parent events", "Source cell", "Status"], rows)
    add("")
    sp = pred.get("stiffness_pair") or {}
    add("### Stiffness pair (FORECAST)")
    add("")
    add(f"- Low `{sp.get('low', '-')}`, high `{sp.get('high', '-')}`; Delta_exact for a uniform pair mix {_fmt(sp.get('delta_exact_uniform_pairs'))}, "
        f"pair by pair {_interval(sp.get('delta_exact_per_pair_range'))}; the closed form's Delta {_fmt(sp.get('delta_closed_form'))}.")
    add(f"- Scoring rule: {sp.get('scoring_rule', '-')}")
    add("")
    add("### Verdict forecasts (FORECAST)")
    add("")
    _table(add, ["Test", "Forecast"], [[f"**{t}**", "FORECAST: " + _esc(v)] for t, v in (pred.get("verdict_forecasts") or {}).items()])
    add("")
    v2 = pred.get("v2_priors") or {}
    add("### Plan v2 priors (FORECAST, never results)")
    add("")
    add(f"- Status: {v2.get('status', '-')}. File `{v2.get('file', '-')}` {_prefix(v2.get('sha256'))}.")
    ratio, m_bar = v2.get("transmission_ratio"), v2.get("m_bar")
    add(f"- Transmission ratio: {_esc(json.dumps(ratio)) if not isinstance(ratio, (int, float)) else _fmt(ratio)}; m_bar: "
        f"{_esc(json.dumps(m_bar)) if not isinstance(m_bar, (int, float)) else _fmt(m_bar)}.")
    theta = v2.get("theta_committed") or {}
    if theta:
        add("")
        rows = [[f"`{k}`", _fmt(v.get("value")), _interval(v.get("ci95")), str(v.get("form", "-")),
                 _fmt(v.get("direct_value")), _count(v.get("direct_primaries"))] for k, v in theta.items()]
        _table(add, ["v2 cell", "theta (FORECAST)", "95 % interval", "Form", "Direct value", "Direct primaries"], rows)
    add("")


def _test_header(add, ctx, tid: str) -> dict:
    decl, pred = ctx["decl"], ctx["pred"]
    spec = ((decl or {}).get("tests") or {}).get(tid, {})
    kind = "blocking" if spec.get("blocking") else ("report only" if "REPORT" in (spec.get("branches") or {}) else "non-blocking")
    add(f"### {tid} ({kind}): {spec.get('statement', 'statement not declared')}")
    add("")
    add(f"- Declared threshold: {spec.get('threshold', '-') if spec else NOT_RUN}")
    if spec.get("definition"):
        add(f"- Declared definition: {spec['definition']}")
    if spec.get("committed_prediction"):
        add(f"- Committed prediction (FORECAST, declarations): {spec['committed_prediction']}")
    forecast = ((pred or {}).get("verdict_forecasts") or {}).get(tid)
    if forecast:
        add(f"- Verdict forecast (FORECAST, predictions): {forecast}")
    branches = spec.get("branches") or {}
    if branches:
        add("- Declared branches: " + "; ".join(f"{k} -> {v}" for k, v in branches.items()))
    return spec


def _results_for(ctx, tid: str):
    return ctx[{"T1": "wp1", "T2": "wp2", "T3": "wp3"}[tid[:2]]]


def _not_run(add, tid: str) -> None:
    add(f"- Result: **{NOT_RUN}** (`{_rel(WORK_PACKAGE_RESULTS[tid[:2]])}` absent).")
    add("")


def _acceptance(add, ctx) -> None:
    decl = ctx["decl"]
    min_scored = ((decl or {}).get("eligibility") or {}).get("min_scored_cells")
    add("## Acceptance results")
    add("")
    add("Measured values come from `wp1_results.json`, `wp2_results.json` and `wp3_results.json`; forecasts appear only in columns headed FORECAST.")
    add("")
    renderers = {"T1.0": _t10, "T1.1": _t11, "T1.2": _t12, "T1.3": _t13, "T1.4": _t14, "T2.1": _t21, "T2.2": _t22, "T2.3": _t23,
                 "T2.4": _t24, "T2.5": _t25, "T3.1": _t31, "T3.2": _t32, "T3.3": _t33, "T3.4": _t34, "T3.5": _t35}
    for tid in TEST_IDS:
        _test_header(add, ctx, tid)
        results = _results_for(ctx, tid)
        if results is None:
            _not_run(add, tid)
            continue
        test = (results.get("tests") or {}).get(tid)
        if test is None:
            add(f"- Result: **not scored** (`{tid}` absent from the results record).")
            add("")
            continue
        renderers[tid](add, ctx, test, results, min_scored)
        add("")
        if tid == "T1.4":
            _wp1_sensitivity(add, ctx)


def _wp1_sensitivity(add, ctx) -> None:
    """Addendum 2 (records/v3/wp1_sensitivity.json): the no-snap floor of the declared drop statistic and the
    interventional swing, per cell and pooled; a sensitivity, never a scored test."""
    add("### Addendum 2 sensitivity (WP1): the no-snap floor and the interventional swing")
    add("")
    s = _load(WP1_SENSITIVITY)
    if s is None:
        add(f"- Result: **{NOT_RUN}** (`{_rel(WP1_SENSITIVITY)}` absent).")
        add("")
        return
    d = s.get("definitions", {})
    add(f"- Status: {s.get('status', '')}. Null: {d.get('null', '')}. Swing: {d.get('swing', '')}. Eligibility: {d.get('eligibility', '')}.")
    pooled = s.get("pooled", {})
    add(f"- Pooled over {_count(pooled.get('rows'))} rows: declared drop / T_peak {_fmt(pooled.get('median_drop'))}, no-snap floor {_fmt(pooled.get('median_null'))}, "
        f"floor-corrected drop {_fmt(pooled.get('median_corrected'))}, interventional swing {_fmt(pooled.get('median_swing'))}, swing / R {_fmt(pooled.get('median_swing_over_R'))}.")
    checks = s.get("checks", {})
    add(f"- Checks: marks reproduced in every re-simulated run: {checks.get('marks_reproduced_all_runs')}; the integrator copy equals the declared integrator on every parent: {checks.get('copy_equals_integrator_all_runs')}. "
        f"Compute: {_count((s.get('compute') or {}).get('runs'))} runs, {_fmt((s.get('compute') or {}).get('core_hours'))} core-hours.")
    add("")
    rows = []
    for name, c in (s.get("cells") or {}).items():
        sp, dp = c.get("swing_pairs") or {}, c.get("drop_pairs") or {}
        rows.append([f"`{name}`", _count(c.get("rows")), _fmt(c.get("median_drop_over_Tpeak")), _fmt(c.get("median_null_over_Tpeak")), _fmt(c.get("median_corrected_drop")),
                     _fmt(c.get("median_swing_over_Tpeak")), _fmt(c.get("median_R_ij_of_rows")), _fmt(c.get("swing_over_R_median")),
                     _fmt(1e3 * c["median_swing_lag_s"], 3) if c.get("median_swing_lag_s") is not None and c["median_swing_lag_s"] == c["median_swing_lag_s"] else "n/a",
                     _fmt(sp.get("spearman_vs_R")) if sp.get("pairs") else "n/a", _fmt(dp.get("spearman_vs_R")) if dp.get("pairs") else "n/a"])
    _table(add, ["cell", "rows", "declared drop", "floor", "corrected", "interventional swing", "R (rows)", "swing / R", "swing lag (ms)", "Spearman swing vs R", "Spearman declared drop vs R"], rows)
    add("")
    add(f"- Provenance: `{_rel(WP1_SENSITIVITY)}` {_digest(WP1_SENSITIVITY)}; addenda {', '.join(f'`{k}` `{v[:12]}...`' for k, v in (s.get('addenda_sha256') or {}).items())}.")
    add("")


def _wp1_cells(ctx, results) -> list[tuple[str, dict]]:
    cells = results.get("cells") or {}
    return [(name, cells[name]) for name in _ordered(cells, ctx["decl"])]


def _t10(add, ctx, test, results, min_scored) -> None:
    add(f"- Result: {_bold(test.get('verdict'))}; {_count(test.get('checked_runs'))} reused runs checked against the v1 marks, {_count(test.get('failed_runs'))} differ.")
    ff = test.get("first_failure")
    if ff:
        add(f"- First differing run: cell `{ff.get('cell', '-')}`, seed {ff.get('seed', '-')}, mark counts {_esc(json.dumps(ff.get('n_marks')))} (v3, v1), "
            f"first difference at index {_count(ff.get('index'))}, t_up {_esc(json.dumps(ff.get('t_up')))} s.")
    closures = results.get("closures") or {}
    if closures:
        rows = [[f"`{n}`", _count(closures[n].get("closures")), _count(closures[n].get("runs"))] for n in _ordered(closures, ctx["decl"])]
        add("")
        add("Formation closures per stage-A cell (pilots included):")
        add("")
        _table(add, ["Cell", "Closures", "Runs"], rows)


def _t11(add, ctx, test, results, min_scored) -> None:
    add(f"- Result: {_share_line(test, min_scored)}")
    add("")
    rows = []
    for name, cell in _wp1_cells(ctx, results):
        t = cell.get("T1.1") or {}
        rows.append([f"`{name}`", _count(cell.get("n_statistics_runs")), _count(cell.get("n_eligible_pairs")), _fmt(cell.get("scored")),
                     _fmt(t.get("median_drop_norm")), _interval(t.get("ci95")), _interval(t.get("band")), _fmt(t.get("committed_prediction")), _bold(t.get("verdict"))])
    _table(add, ["Cell", "Statistics runs", "Eligible pairs", "Scored", "Median drop_norm", "95 % CI (seeds)", "Band", "Exact response (FORECAST)", "Verdict"], rows)
    add("")
    add("Sensitivities (report only; medians of the same statistic under the declared variants):")
    add("")
    rows = []
    for name, cell in _wp1_cells(ctx, results):
        s = cell.get("sensitivities") or {}
        rows.append([f"`{name}`", _fmt(s.get("cos_normalised_median")), _fmt(s.get("half_window_median")), _fmt(s.get("parents_including_bounces_median"))])
    _table(add, ["Cell", "cos-normalised median", "Half-window median", "Parents incl. bounces median"], rows)


def _t12(add, ctx, test, results, min_scored) -> None:
    add(f"- Result: {_share_line(test, min_scored)}")
    add("")
    rows = []
    for name, cell in _wp1_cells(ctx, results):
        t = cell.get("T1.2") or {}
        parents = t.get("parents_per_pair") or []
        rows.append([f"`{name}`", _fmt(cell.get("scored")), _count(len(t.get("pairs") or [])), _count(sum(parents)) if parents else "0",
                     _fmt(t.get("spearman_vs_R")), _fmt(t.get("spearman_vs_operator")), _bold(t.get("verdict"))])
    _table(add, ["Cell", "Scored", "Pooled pairs (>= min parents)", "Parents in pooled pairs", "Spearman vs R", "Spearman vs closed-form operator", "Verdict"], rows)
    calibration = (((ctx["decl"] or {}).get("calibration_cell") or {}).get("cell"))
    cell = (results.get("cells") or {}).get(calibration)
    if cell and (cell.get("T1.2") or {}).get("pairs"):
        t = cell["T1.2"]
        add("")
        add(f"Per pooled ordered pair at the calibration cell `{calibration}` (mirror pairs (i, j) ~ (4 - i, 4 - j) share a label):")
        add("")
        rows = []
        for k, pair in enumerate(t["pairs"]):
            rows.append([f"({pair[0]}, {pair[1]})", _count(_at(t.get("parents_per_pair"), k)), _fmt(_at(t.get("median_drop_over_Tpeak"), k)),
                         _fmt(_at(t.get("R_pooled"), k)), _fmt(_at(t.get("operator_pooled"), k))])
        _table(add, ["Pair (i, j)", "Parents", "Median drop / T_peak (measured)", "R_ij (FORECAST)", "Closed-form operator (FORECAST)"], rows)


def _t13(add, ctx, test, results, min_scored) -> None:
    add(f"- Result: {_share_line(test, min_scored)}")
    add("")
    rows = []
    for name, cell in _wp1_cells(ctx, results):
        t = cell.get("T1.3") or {}
        rows.append([f"`{name}`", _count(t.get("n")), _fmt(t.get("median_drop_over_Tpeak_R")), _interval(t.get("ci95")), _bold(t.get("verdict"))])
    _table(add, ["Cell", "Small-peak parents (n)", "Median drop / (T_peak R_ij)", "95 % CI (seeds)", "Verdict"], rows)


def _t14(add, ctx, test, results, min_scored) -> None:
    add(f"- Result: {_bold(test.get('verdict'))}" + (f"; {test['reason']}" if test.get("reason") else ""))
    if "delta_measured" in test:
        n = test.get("n_pairs") or {}
        add(f"- Delta_meas = {_fmt(test.get('delta_measured'))}, seed-bootstrap 95 % interval {_interval(test.get('ci95'))} "
            f"(median drop_norm {_fmt(test.get('median_low'))} at the low stiffness on {_count(n.get('low'))} rows, {_fmt(test.get('median_high'))} at the high on {_count(n.get('high'))} rows; "
            f"{_count(len(test.get('seeds') or []))} common statistics seeds).")
        add(f"- Delta_exact over the scored rows' pair mix (FORECAST) = {_fmt(test.get('delta_exact'))}; committed uniform-pair value (FORECAST) = {_fmt(test.get('delta_exact_uniform_pairs'))}; "
            f"closed form (FORECAST) = {_fmt(test.get('delta_closed_form'))}.")


def _wp2_cells(ctx, results) -> list[tuple[str, dict]]:
    cells = results.get("cells") or {}
    return [(name, cells[name]) for name in _ordered(cells, ctx["decl"])]


def _t21(add, ctx, test, results, min_scored) -> None:
    add(f"- Result: {_share_line(test, min_scored)}")
    add("")
    labels = None
    rows = []
    for name, cell in _wp2_cells(ctx, results):
        t = cell.get("T2.1") or {}
        labels = labels or t.get("labels")
        means, counts = t.get("means") or [], t.get("counts") or []
        binned = " / ".join(f"{_fmt(m)} [{_count(c)}]" for m, c in zip(means, counts))
        rows.append([f"`{name}`", _count(cell.get("n_events_statistics")), _count(cell.get("n_events_pilot")), _count(cell.get("measured_cross_offspring")),
                     _fmt(cell.get("scored")), _fmt(t.get("spearman")), _fmt(t.get("first_bin_mean")), binned or "-", _bold(t.get("verdict"))])
    if labels:
        add("Peak bins (x = T_peak / T0): " + ", ".join(labels) + "; each entry is the mean cross-cable offspring [events in the bin].")
        add("")
    _table(add, ["Cell", "Events (statistics)", "Events (pilot)", "Cross-cable offspring", "Scored", "Spearman vs bin order", "First-bin mean", "Binned m_x", "Verdict"], rows)


def _t22(add, ctx, test, results, min_scored) -> None:
    add(f"- Result: {_share_line(test, min_scored)}")
    fan = test.get("fan_subgroup")
    add(f"- Fan subgroup: {_share_line(fan, 1) if fan else 'no fan cell scored'}")
    fit = results.get("isotonic_fit") or {}
    add(f"- Isotonic m_x fitted on `{fit.get('cell', '-')}` pilot seeds {', '.join(str(s) for s in fit.get('seeds', []))}: {_count(fit.get('n'))} events.")
    add("")
    rows = []
    for name, cell in _wp2_cells(ctx, results):
        t = cell.get("T2.2") or {}
        rows.append([f"`{name}`", str(t.get("formation", "-")), _count(cell.get("measured_cross_offspring")), _fmt(t.get("predicted")), _fmt(t.get("ratio")), _bold(t.get("verdict"))])
    _table(add, ["Cell", "Formation", "Measured cross-cable offspring", "Predicted (isotonic transfer)", "Predicted / measured", "Verdict"], rows)


def _t23(add, ctx, test, results, min_scored) -> None:
    add(f"- Result: {_bold(test.get('verdict'))} on {_count(test.get('n'))} sampled parents; passed = {_fmt(test.get('passed'))}.")
    failed = test.get("not_reengaged") or []
    add(f"- Parents whose factual branch did not re-engage: {_count(len(failed))}" + (f" (first: {_esc(json.dumps(failed[0]))})" if failed else "") + ".")
    dv, dt, dp = test.get("abs_dv_m_s") or {}, test.get("abs_dt_s") or {}, test.get("T_peak_relative_error") or {}
    add(f"- |v_cf - v_return| [m/s]: median {_fmt(dv.get('median'))}, p95 {_fmt(dv.get('p95'))}, max {_fmt(dv.get('max'))}.")
    add(f"- |t_cf - t_up| [s]: median {_fmt(dt.get('median'))}, max {_fmt(dt.get('max'))}.")
    add(f"- |T_peak_cf / T_peak - 1|: median {_fmt(dp.get('median_abs'))}, p95 {_fmt(dp.get('p95_abs'))}, max {_fmt(dp.get('max_abs'))}.")
    add(f"- Factual other-cable onset set reproduced in {_pct(test.get('onset_set_reproduced_share'))} of parents.")
    th = test.get("thresholds") or {}
    if th:
        add("- Thresholds applied: " + ", ".join(f"{k} = {_fmt(v)}" for k, v in th.items()) + ".")


def _t24(add, ctx, test, results, min_scored) -> None:
    add(f"- Result: {_share_line(test, min_scored)}")
    add("")
    rows = []
    per_cell = test.get("per_cell") or {}
    for name, cell in _wp2_cells(ctx, results):
        iv = cell.get("intervention") or {}
        ts = iv.get("time_shift") or {}
        rows.append([f"`{name}`", _count(iv.get("parents")), _count(iv.get("causal")), _count(iv.get("causal_held")), _count(ts.get("measured")),
                     _fmt(ts.get("null")), _fmt(ts.get("excess")), _fmt(iv.get("causal_over_excess")), _bold(per_cell.get(name, "-"))])
    _table(add, ["Cell", "Sampled parents", "Causal offspring", "Causal (held-heading sensitivity)", "Measured in window", "Time-shift null", "Excess", "Causal / excess", "Verdict"], rows)


def _t25(add, ctx, test, results, min_scored) -> None:
    add(f"- Result: {_share_line(test, min_scored)}")
    kp = results.get("kernel_path") or {}
    add(f"- Kernel path: first candidate by the WP1 verdicts = {kp.get('first_by_wp1', '-')}; chosen = **{results.get('K_path_chosen') or 'none'}**.")
    for candidate, share in (kp.get("tested") or {}).items():
        add(f"  - {candidate}: {_share_line(share, min_scored)}")
    add("")
    rows = []
    for name, cell in _wp2_cells(ctx, results):
        k = cell.get("kernels") or {}
        r_hs = k.get("R:half_sine") or {}
        rows.append([f"`{name}`", _count(cell.get("measured_cross_offspring")), _fmt(cell.get("scored")), _fmt(r_hs.get("predicted")), _fmt(r_hs.get("ratio")),
                     _fmt((k.get("R:rectangular") or {}).get("ratio")), _fmt((k.get("R_hat:half_sine") or {}).get("ratio")), _fmt((k.get("R_hat:rectangular") or {}).get("ratio")),
                     _fmt((k.get("empirical") or {}).get("ratio")), _fmt(r_hs.get("rho")), _bold((cell.get("T2.5") or {}).get("verdict"))])
    _table(add, ["Cell", "Measured cross-cable offspring", "Scored", "Predicted, R half-sine", "Ratio, R half-sine", "Ratio, R rectangular",
                 "Ratio, R_hat half-sine", "Ratio, R_hat rectangular", "Ratio, empirical", "rho(K), R half-sine", "Verdict"], rows)


def _wp3_cells(ctx, results) -> list[tuple[str, dict]]:
    cells = results.get("cells") or {}
    return [(name, cells[name]) for name in _ordered(cells, ctx["decl"])]


def _t31(add, ctx, test, results, min_scored) -> None:
    add(f"- Result: {_share_line(test, min_scored)}; kernel path `{results.get('kernel_path', '-')}` on stages {', '.join(results.get('stages') or [])}.")
    add("")
    rows = []
    for name, cell in _wp3_cells(ctx, results):
        ident = cell.get("identity_check") or {}
        nu_p, nu_m, nu_t = cell.get("nu_primary_per_cable_s"), cell.get("nu_total_measured_per_cable_s"), cell.get("nu_total_predicted_per_cable_s")
        rows.append([f"`{name}`", _fmt(cell.get("scored")), _fmt(cell.get("operable")), f"{_count(cell.get('closures'))}/{_count(cell.get('runs'))}",
                     f"{_fmt(cell.get('rho'))} {_interval(cell.get('rho_ci95'))}" if "rho" in cell else "-",
                     _rate(sum(nu_p)) if isinstance(nu_p, list) else "-", _rate(sum(nu_m)) if isinstance(nu_m, list) else "-",
                     _rate(sum(nu_t)) if isinstance(nu_t, list) else "-", _fmt(cell.get("rate_ratio_pooled")),
                     _fmt(ident.get("nu_total_over_primary_measured")), _fmt(ident.get("one_over_one_minus_m_bar")), _bold((cell.get("verdicts") or {}).get("T3.1"))])
    _table(add, ["Cell", "Scored", "Operable", "Closures/runs", "rho(K) [95 % CI]", "nu_primary, 5 cables [1/cable-s]", "nu_total measured [1/cable-s]",
                 "nu_total predicted [1/cable-s]", "Predicted / measured", "Identity: measured total/primary", "Identity: 1/(1 - m_bar)", "Verdict"], rows)
    add("")
    add("Per cable (rates in 1/cable-s, cables 0-4):")
    add("")
    rows = []
    for name, cell in _wp3_cells(ctx, results):
        if "rho" not in cell:
            continue
        rows.append([f"`{name}`", _rates(cell.get("nu_primary_per_cable_s")), _rates(cell.get("nu_total_measured_per_cable_s")), _rates(cell.get("nu_total_predicted_per_cable_s")),
                     " / ".join(_fmt(p) for p in cell.get("pi") or []) or "-"])
    _table(add, ["Cell", "nu_primary", "nu_total measured", "nu_total predicted", "pi (primary type distribution)"], rows or [["-", "-", "-", "-", "-"]])


def _t32(add, ctx, test, results, min_scored) -> None:
    add(f"- Result: {_share_line(test, min_scored)}")
    add("")
    rows = []
    for name, cell in _wp3_cells(ctx, results):
        tp, tr = cell.get("theta_pred"), cell.get("theta_runs")
        diff = abs(tp - tr) if isinstance(tp, (int, float)) and isinstance(tr, (int, float)) and math.isfinite(tp) and math.isfinite(tr) else None
        rows.append([f"`{name}`", _fmt(cell.get("scored")), _count(cell.get("n_primary")), _count(cell.get("n_events")), _fmt(tp), f"{_fmt(tr)} {_interval(cell.get('theta_runs_ci95'))}" if tr is not None else "-",
                     _fmt(diff), f"{_fmt(cell.get('theta_ferro_segers'))} {_interval(cell.get('theta_ferro_segers_ci95'))}" if cell.get("theta_ferro_segers") is not None else "-",
                     _fmt(cell.get("m_bar")), _bold((cell.get("verdicts") or {}).get("T3.2"))])
    _table(add, ["Cell", "Scored", "Primary events", "Events", "theta_pred (from K)", "theta_runs [95 % CI]", "abs(theta_pred - theta_runs)", "theta Ferro-Segers [95 % CI]", "m_bar", "Verdict"], rows)


def _t33(add, ctx, test, results, min_scored) -> None:
    add(f"- Result: {_bold(test.get('verdict'))} (no threshold).")
    add("")
    rows = []
    for name, cell in _wp3_cells(ctx, results):
        rice = cell.get("rice") or {}
        rate = rice.get("rice_onset_rate_per_cable_s")
        nu_p = cell.get("nu_primary_per_cable_s")
        pooled = _fmt(sum(nu_p) / sum(rate)) if isinstance(rate, list) and isinstance(nu_p, list) and sum(rate) > 0 else "-"
        rows.append([f"`{name}`", _rates(nu_p) if isinstance(nu_p, list) else "-", _rates(rate) if isinstance(rate, list) else ("error: " + _esc(rice.get("error", "-"))), pooled])
    _table(add, ["Cell", "nu_primary measured [1/cable-s]", "Rice onset rate of the taut LTI tow [1/cable-s] (model)", "Pooled nu_primary / Rice (theta_primary)"], rows)
    v2 = ((ctx["pred"] or {}).get("v2_priors") or {}).get("theta_committed") or {}
    if v2:
        add("")
        add("Plan v2's committed theta (PRIOR FORECAST at intensity 0.35, different cells, never a result): "
            + ", ".join(f"`{k}` {_fmt(v.get('value'))} {_interval(v.get('ci95'))}" for k, v in v2.items()) + ".")


def _t34(add, ctx, test, results, min_scored) -> None:
    branch = test.get("branch")
    add(f"- Result: branch {_bold(branch)}.")
    if branch == "(a)":
        add(f"- Critical intensity I_c = {_fmt(test.get('critical_intensity'))} by log-linear interpolation between `{'` and `'.join(test.get('between') or ['-', '-'])}`.")
    elif branch == "(b)":
        add(f"- Margin result: highest operable cell `{test.get('highest_operable', '-')}` with rho = {_fmt(test.get('rho'))}, margin 1 - rho = {_fmt(test.get('margin'))}.")
    else:
        add("- Unresolved: no operable cell with an interval entirely above 1 and one below it, or a straddling interval.")
    add(f"- B_ext authorised by this analysis: {_fmt(results.get('b_ext_authorised_now'))}"
        + (f"; `{_rel(B_EXT_AUTHORISED)}` present (closures at 1.5: {_count(ctx['b_ext'].get('closures_at_1.5'))} of {_count(ctx['b_ext'].get('runs'))} runs, branch before: {ctx['b_ext'].get('branch_before', '-')})." if ctx["b_ext"] else "."))
    add("")
    table = test.get("table") or {}
    rows = []
    for name, row in sorted(table.items(), key=lambda kv: (kv[1].get("intensity") if isinstance(kv[1].get("intensity"), (int, float)) else math.inf, kv[0])):
        rows.append([f"`{name}`", _fmt(row.get("intensity")), _fmt(row.get("rho")), _interval(row.get("ci95")), _count(row.get("closures")), _fmt(row.get("operable"))])
    _table(add, ["Sweep cell", "Intensity", "rho(K)", "95 % CI (seeds)", "Closures", "Operable"], rows or [["-"] * 6])


def _t35(add, ctx, test, results, min_scored) -> None:
    add(f"- Result: {_bold(test.get('verdict'))} (no threshold).")
    add("")
    rows = []
    for name, cell in _wp3_cells(ctx, results):
        law = cell.get("cluster_law") or {}
        bt = cell.get("borel_tanner") or {}
        rows.append([f"`{name}`", _count(law.get("n")), _fmt(law.get("mean")), _fmt(law.get("theta")), _count(law.get("max")),
                     _fmt(bt.get("m_bar")), _fmt(bt.get("ks_borel_tanner")), _fmt(bt.get("ks_multitype_mc")), _fmt(bt.get("mc_mean"))])
    _table(add, ["Cell", "Clusters", "Mean size", "1 / mean size", "Largest", "m_bar", "KS vs Borel-Tanner(m_bar)", "KS vs multitype Monte Carlo", "Monte Carlo mean size"], rows)
    add("")
    add(f"Cluster-size samples: `{_rel(WP3_CLUSTER_SIZES)}` ({'present' if WP3_CLUSTER_SIZES.exists() else 'absent'}).")


def _deviations(add, ctx) -> None:
    add("## Deviations from the plan, with addenda")
    add("")
    addenda = ctx["addenda"]
    if not addenda:
        add("No addendum has been written; the declarations stand as committed.")
        add("")
        return
    for path, record in addenda:
        if "error" in record and "addendum" not in record:
            add(f"- `{_rel(path)}`: unreadable ({record['error']}).")
            continue
        amends = record.get("amends") or {}
        add(f"- **Addendum {record.get('addendum', '?')}** ({record.get('date', 'undated')}), `{_rel(path)}`, amends `{amends.get('path', '-')}` {_prefix(amends.get('sha256'))}.")
        add(f"  - Why: {record.get('why', '-')}")
        add(f"  - Change: {_esc(json.dumps(record.get('change'), sort_keys=True))}")
        add(f"  - Seen before the change: {record.get('seen_before_the_change', '-')}")
        add(f"  - Source {_source(record)}.")
    add("")


def _artifacts(add, ctx) -> None:
    add("## Artifacts")
    add("")
    add("Records present in `records/v3/`:")
    add("")
    present = sorted(list(RECORD_DIR.glob("*.json")) + list(RECORD_DIR.glob("*.npz"))) if RECORD_DIR.exists() else []
    for path in present:
        add(f"- `{_rel(path)}` {_digest(path, 12)}")
    if not present:
        add("- none")
    add("")
    expected = [DECLARATIONS, PREDICTIONS, WP1_RESULTS, WP1_TRANSMISSION, WP2_RESULTS, WP2_OFFSPRING, WP2_INTERVENTION, WP3_RESULTS, WP3_CLUSTER_SIZES,
                GATES[1], GATES[2], GATES[3]]
    missing = [p for p in expected if not p.exists()]
    add("Expected records not yet written: " + (", ".join(f"`{_rel(p)}`" for p in missing) if missing else "none") + ".")
    add("")
    figures = sorted(FIGURE_DIR.glob(FIGURE_GLOB)) if FIGURE_DIR.exists() else []
    add(f"Figures `{_rel(FIGURE_DIR)}/{FIGURE_GLOB}`: " + (", ".join(f"`{p.name}`" for p in figures) if figures else "none yet") + ".")
    add(f"Presentation clip `{_rel(CLIP)}`: {'present' if CLIP.exists() else 'absent'}.")
    add(f"Explorer `{_rel(EXPLORER)}`: {'present' if EXPLORER.exists() else 'absent'}.")
    add("")


def _open_items(add, ctx) -> None:
    add("## Open items")
    add("")
    if OPEN_ITEMS.exists():
        add(OPEN_ITEMS.read_text().rstrip())
    else:
        add(f"- No open item has been recorded yet; `{_rel(OPEN_ITEMS)}` is inserted here verbatim once it exists.")
        add("- Every test not yet scored above is open by construction.")
    add("")


def _discussion(add, ctx) -> None:
    add("## Discussion")
    add("")
    text = DISCUSSION.read_text().rstrip() if DISCUSSION.exists() else ""
    if text:
        add(text)
        add("")
    for heading in DISCUSSION_SUBSECTIONS:
        if heading not in text:
            add(heading)
            add("")
            add(f"Not yet written (`{_rel(DISCUSSION)}`).")
            add("")


# ----------------------------------------------------------------------------- build


def _context() -> dict:
    return {
        "decl": _load(DECLARATIONS), "pred": _load(PREDICTIONS),
        "wp1": _load(WP1_RESULTS), "wp2": _load(WP2_RESULTS), "wp3": _load(WP3_RESULTS),
        "gates": {n: _load(path) for n, path in GATES.items()},
        "not_launched": _load(WP3_NOT_LAUNCHED), "b_ext": _load(B_EXT_AUTHORISED),
        "addenda": _addenda(),
    }


def build() -> str:
    ctx = _context()
    lines: list[str] = []
    add = lines.append
    add("# Plan v3 Report — Cascade criticality of cable-towed fleets")
    add("")
    add("Generated by `tether/analysis/v3/cascade_report.py` from the committed records in `records/v3/`; every number below is read from a record, none is typed into the generator. "
        "Forecasts are labelled FORECAST and never share a column with a measurement.")
    add("")
    for section in (_status, _gates, _provenance, _declarations, _predictions, _acceptance, _deviations, _artifacts, _open_items, _discussion):
        start = len(lines)
        try:
            section(add, ctx)
        except Exception as error:  # noqa: BLE001 - a layout surprise in one record must not lose the rest of the report
            del lines[start + 1:]  # keep the section heading if it was written
            add("")
            add(f"_(This section could not be rendered from the records: {type(error).__name__}: {error}. The record layout may differ from the producing code this generator follows.)_")
            add("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    text = build()
    write_bytes(REPORT, text.encode("utf-8"))
    print(REPORT)


if __name__ == "__main__":
    main()
