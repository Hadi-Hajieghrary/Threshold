# ACC 2027 manuscript

*Snap-Load Transmission Between Cables Sharing a Towed Payload: Why the Impulsive Estimate
Overpredicts*: an IEEE ACC 2027 conference paper (`ieeeconf` class, 8 pages).

## Files

| File | Contents |
|---|---|
| `main.tex` | Preamble, title, authors, abstract; inputs the six sections. |
| `Sec_01_introduction.tex` | I. Introduction and contributions. |
| `Sec_02_related.tex` | II. Related work. |
| `Sec_03_model.tex` | III. Model and problem: transmission ratio (Eq. 1), collinear reduction (Eq. 2), impulsive estimate (Eq. 3); Fig. 1. |
| `Sec_04_theory.tex` | IV. Transmission theory: Proposition 1, Corollary 1, Lemma 1, Theorem 1, Corollary 2, self-consistent contact, dissipation and stiffness; Figs. 2–3, Table I. |
| `Sec_05_sim.tex` | V. Simulation of a multibody fleet: Table II (testbed), forecast and protocol, Table III (evidence status), magnitude, parameter dependence, unloading; Figs. 4–7. |
| `Sec_06_consequences.tex` | VI. Consequences, limitations, conclusion, and the data and pre-registration statement. |
| `refs.bib` | Bibliography. |
| `ieeeconf.cls`, `IEEEtran.bst` | IEEE conference class and bibliography style. |
| `fig_*.pdf` | The seven figures (vector PDF). |
| `main.pdf` | Compiled manuscript. |

## Build

From the repository root:

```bash
make acc-paper     # latexmk -pdf main.tex in this folder
```

or, in this folder, `pdflatex main && bibtex main && pdflatex main && pdflatex main`. Requires a TeX
Live installation with `amsmath`, `amsthm`, `mathtools`, `booktabs`, `cite` and `hyperref`; the class
and bibliography style are included.

## Figures and the code behind them

The figures are included as PDFs; the scripts that drew them are not part of this repository. The
quantities they show come from the following code in `tether/`:

| Figure | Content | Code |
|---|---|---|
| Fig. 1 `fig_fleet_snap.pdf` | A recorded snap on the testbed, to scale | plant `tether/physics/fleet.py`, run `tether/campaign/fleet_run.py` |
| Fig. 2 `fig_shock_spectrum.pdf` | Shock spectrum `S(ε)` (Lemma 1) with the testbed values | closed forms of Sec. IV; frequencies as in `tether/theory/transmission.py` |
| Fig. 3 `fig_massratio.pdf` | Transmission versus mass ratio: collinear model and testbed markers | collinear model of Sec. IV; nominal-cell markers from the v3 campaign (`tether/campaign/v3/`); the markers at mass ratio 0.50 from the declared second-mass test, whose driver (built on the `tether/campaign/v3/` machinery) is not included |
| Fig. 4 `fig_cells.pdf` | Declared statistic, design-geometry reading and interventional ratio in the sixteen cells | `tether/campaign/v3/reducer.py`, `analyse.py`, `wp1_sensitivity.py`; forecast `tether/theory/transmission.py` |
| Fig. 5 `fig_swing_vs_peak.pdf` | Interventional swing versus snap peak | `tether/campaign/v3/wp1_sensitivity.py`, `intervention.py` |
| Fig. 6 `fig_stiffness.pdf` | Stiffness test, fixed-dissipation control, and lags | `tether/campaign/v3/analyse.py` (T1.4), `wp1_sensitivity.py`; collinear model of Sec. IV; the fixed-dissipation control comes from a declared follow-up test whose driver is not included |
| Fig. 7 `fig_threshold_curve.pdf` | Empirical analog of Corollary 1 | `tether/campaign/v3/reducer.py` (transmission rows), `intervention.py` (causal series) |

The forecast of Sec. V-A is `predicted_pair_table` in `tether/theory/transmission.py`, built on the
34-state linearization of `tether/theory/reduced_lti.py`. `tether/tests/test_acc_paper.py` checks that the
forecast is insensitive to the Jacobian step (Sec. V-A); its other tests compare the forecast and the
pinned v3 files with the campaign outputs and skip when those are absent.

## Status

The author block lists names and e-mail addresses without affiliations (TODO: verify affiliations and
any funding acknowledgment before submission).
