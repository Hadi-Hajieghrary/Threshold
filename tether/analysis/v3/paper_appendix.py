"""The paper's pre-registration appendix, generated from the records.

    python -m tether.analysis.v3.paper_appendix     # Paper/Sections/v3_appendix_prereg.tex

The appendix exists so a reader can check that what the paper reports is what was declared before the
runs.  Every hash, date, verdict and addendum below is read from ``records/v3/``; nothing is typed in.
Regenerate it whenever a record changes, and recompile the paper.
"""

from __future__ import annotations

import json

from tether.campaign.common import ROOT, sha256_file, write_bytes
from tether.campaign.v3 import campaign

OUT = ROOT / "Paper" / "Sections" / "v3_appendix_prereg.tex"
TESTS = ("T1.0", "T1.1", "T1.2", "T1.3", "T1.4", "T2.1", "T2.2", "T2.3", "T2.4", "T2.5",
         "T3.1", "T3.2", "T3.3", "T3.4", "T3.5")
RESULTS = {"T1": "wp1_results.json", "T2": "wp2_results.json", "T3": "wp3_results.json"}

# One clause per test, written for a reader of the paper; the declarations' own statements are internal
# shorthand carrying raw identifiers and are not reproduced.  Verdicts and blocking flags come from the records.
STATEMENTS = {
    "T1.0": "the reused configurations reproduce the earlier campaign's re-engagement marks exactly",
    "T1.1": "the median normalised neighbour swing lies inside the derivation's declared band $[0.2, 0.9]$",
    "T1.2": "the ordering of the swing across cable pairs follows the exact linear response",
    "T1.3": "for snaps below the derived threshold, the swing over its predicted value lies within a factor $1.5$",
    "T1.4": "the swing's dependence on cable stiffness is the exact response's, not the stiffness-free closed form's",
    "T2.1": "the mean cross-cable offspring per event grows with the snap's peak and vanishes below the pretension",
    "T2.2": "the offspring curve fitted on one configuration transfers to the others within a factor $1.5$",
    "T2.3": "the counterfactual integrator reproduces the record each of its branches starts from",
    "T2.4": "interventional and observational offspring counts agree within a factor $1.5$",
    "T2.5": "a kernel with no fitted parameter beyond the margin law predicts the offspring counts",
    "T3.1": "the cluster-corrected rate law predicts the measured total rate within a factor $1.5$",
    "T3.2": "the extremal index predicted from the kernel matches the measured primary share within $0.10$",
    "T3.3": "the primary rate against a Rice up-crossing estimate (reported, not gated)",
    "T3.4": "whether the spectral radius reaches one inside the operable range of the intensity sweep",
    "T3.5": "the cluster-size distribution against Borel--Tanner and a multitype simulation (reported, not gated)",
}

# One sentence per amendment, written for a reader of the paper.  The date and the hash of each are read
# from the record; only this gloss is authored, and each is a faithful precis of that record's own reason.
AMENDMENTS = {
    1: ("a defect in the reduction, found when a run containing slack onsets but no re-engagement at all "
        "reached code that assumed at least one; the repair leaves the output of every run with a "
        "re-engagement bit-identical, and the cached runs already reduced were unaffected."),
    2: ("the discovery that the scored window statistic carries a floor unrelated to any snap, and the "
        "declaration of an interventional estimator free of it, both reported beside the scored verdicts "
        "and never scored."),
    3: ("four under-specifications in the offspring definitions, found by audit and none of them changing a "
        "verdict: the positive-part convention in the interventional counts, the comparator used for the "
        "observational baseline, the interpolation of the fitted offspring curve, and the population the "
        "parameter-free kernel actually models. Each is reported as a labelled sensitivity."),
    4: ("the finding that the declaration counts offspring as onsets while counting the tree, the primary "
        "share and both rates of the rate law in events, and that the rate law's two sides use different "
        "exposures; together with the degeneracy of the declared interval on the spectral radius and the "
        "two configurations whose fitting runs contain no offspring. The consistently counted reading is "
        "recorded as a sensitivity, with explicit constraints on what may be reported from it."),
    5: ("corrections to the text of the fourth amendment, and two facts that bound the consistently counted "
        "reading: on one exposure its two readings are a single statistic, and a kernel set identically to "
        "zero already satisfies the declared band in a fifth of the configurations."),
}


def _load(name: str):
    path = campaign.RECORD_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def _tex(text: str) -> str:
    return (text.replace("\\", "/").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("#", r"\#").replace(">=", r"$\ge$").replace("<=", r"$\le$"))


def build() -> str:
    out: list[str] = []
    add = out.append
    decl = _load("cascade_declarations.json")
    results = {k: _load(v) for k, v in RESULTS.items()}
    add("\\section{Pre-registration of the extension}")
    add("\\label{App:Prereg}")
    add("")
    add("The measurements of \\Cref{Sec:TransmissionMeasured,Sec:CascadeProcess} come from a campaign whose")
    add("tests, thresholds and branches were written down and hashed before any run of it, so that the")
    add("verdicts below could not be chosen after the fact. This appendix records what was declared, what")
    add("was amended and by whom it was checked. The declarations, the committed forecasts, the reduced")
    add("records, the analysis code and the audit reports are in the project repository.")
    add("")
    add("\\subsection{What was declared, and when}")
    add("")
    add("The plan, the theorem statements and a per-configuration forecast of every quantity were committed")
    add("first; the declarations file pins each of them by hash, and the analysis refuses to run unless the")
    add("file on disk still matches the module that wrote it.")
    add("")
    add("\\begin{itemize}")
    if decl:
        add(f"  \\item Declarations \\texttt{{cascade\\_declarations.json}}, SHA-256 \\texttt{{{sha256_file(campaign.DECLARATIONS_PATH)[:16]}\\ldots}}, pinning")
        add(f"  the plan (\\texttt{{{decl['plan_sha256'][:12]}\\ldots}}), the theorem statements (\\texttt{{{decl['theory_sha256'][:12]}\\ldots}})")
        add(f"  and the committed forecasts (\\texttt{{{decl['predictions_sha256'][:12]}\\ldots}}).")
        stages = decl.get("cells", {})
        n_declared = sum(len(v) for v in stages.values()) if isinstance(stages, dict) else len(stages)
        n_run = sum(len(v) for k, v in stages.items() if k != "B_ext") if isinstance(stages, dict) else n_declared
        add(f"  \\item {n_declared} configurations declared, of which {n_run} were run (the remaining stage was")
        add("  conditional on a branch that did not fire), and "
            f"{len(decl.get('tests', {}))} tests, each with its threshold, its three possible")
        add("  verdicts and a declared destination for each verdict.")
    add("  \\item Eleven analysis and plant modules pinned by hash, so that a change to any of them after")
    add("  declaration requires a dated amendment.")
    add("\\end{itemize}")
    add("")
    add("\\subsection{Verdicts as scored}")
    add("")
    add("Each test was scored once, by the rule declared for it. \\textsc{fail} here means the declared")
    add("threshold was not met; it does not always mean the underlying quantity is absent, and where the")
    add("distinction matters the text says so.")
    add("")
    add("\\begin{table}[t]")
    add("  \\centering")
    add("  \\caption{Pre-registered tests of the extension and their verdicts. Blocking tests gate the next")
    add("  work package. \\textsc{unscored} means a declared precondition was not met.}")
    add("  \\label{tab:v3verdicts}")
    add("  \\begin{tabular}{llcp{0.46\\columnwidth}}")
    add("    \\toprule")
    add("    Test & Verdict & Blocking & What it tested \\\\")
    add("    \\midrule")
    for tid in TESTS:
        spec = (decl or {}).get("tests", {}).get(tid, {})
        res = results.get(tid.split(".")[0][:2])
        got = (res or {}).get("tests", {}).get(tid, {}) if res else {}
        verdict = got.get("verdict") or got.get("branch") or "not run"
        blocking = "yes" if spec.get("blocking") else "--"
        add(f"    {_tex(tid)} & \\textsc{{{_tex(str(verdict)).lower()}}} & {blocking} & {STATEMENTS.get(tid, '')} \\\\")
    add("    \\bottomrule")
    add("  \\end{tabular}")
    add("\\end{table}")
    add("")
    add("\\subsection{Gates}")
    add("")
    add("A gate record was written only after the work package's analysis had been independently re-derived.")
    add("")
    add("\\begin{itemize}")
    for k in (1, 2, 3):
        g = _load(f"gate_wp{k}.json")
        if not g:
            continue
        failing = ", ".join(_tex(t) for t in g["failing_blocking_tests"]) or "none"
        add(f"  \\item Work package {k}: \\textsc{{{g['verdict'].lower()}}}, failing blocking tests {failing}.")
    add("\\end{itemize}")
    add("")
    add("\\subsection{Amendments}")
    add("")
    add("Declarations were never edited. Five dated amendments record every change made after the")
    add("declaration, each naming what was seen before it was written.")
    add("")
    add("\\begin{itemize}")
    for n in range(1, 10):
        path = campaign.addendum_path(n)
        if not path.exists():
            continue
        a = json.loads(path.read_text())
        add(f"  \\item Amendment {n} ({a['date']}), SHA-256 \\texttt{{{sha256_file(path)[:12]}\\ldots}}: {AMENDMENTS.get(n, 'see the record.')}")
    add("\\end{itemize}")
    add("")
    add("\\subsection{Independent checking}")
    add("")
    add("Each work package was re-derived from the records by an independent verifier writing fresh code and")
    add("reading the producing code, before its gate was written. Every quantity in all three result files")
    add("reproduced exactly; no arithmetic error was found in any of them. The findings were definitional:")
    add("the mismatch between the forecast's normaliser and the scored statistic's")
    add("(\\Cref{Sec:TransmissionMeasured}); the positive-part convention in the interventional counts and the")
    add("population the parameter-free kernel actually models (\\Cref{Sec:CascadeProcess}); and the mixture of")
    add("offspring onsets with events that made the two criticality tests undecidable")
    add("(\\Cref{Sec:CascadeProcess}). Each was recorded as an amendment, with its corrected reading reported")
    add("beside the verdict it qualifies rather than in place of it. One finding claiming a further defect in")
    add("the sweep's branch rule was itself refuted on re-derivation and is recorded as refuted.")
    add("")
    return "\n".join(out) + "\n"


def main() -> None:
    print(write_bytes(OUT, build().encode()), OUT)


if __name__ == "__main__":
    main()
