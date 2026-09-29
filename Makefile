PYTHON ?= python
WORKERS ?= 16
export PYTHONPATH := $(CURDIR)

.PHONY: test test-fast phase0 phase1 phase2 phase2-predict phase2-compute phase2-analyse figs replay \
	phase3 phase4 phase5 phase5-staleness phase6 \
	phase1-mechanics phase1-deterministic phase1r phase1-t4r phase1-t5r phase1-t9r

# Unit and gate tests (pytest.ini scopes collection to tether/tests).  The legacy Phase 1 stage
# tests build large session fixtures and take long; test-fast runs the plant, theory, statistics,
# v3 and ACC 2027 tests (about 1 minute).  Several other modules without session fixtures
# (test_v2_*, test_estimation_*, test_monitor, ...) also run in seconds each.
test:
	$(PYTHON) -m pytest -q

test-fast:
	$(PYTHON) -m pytest -q tether/tests/test_fleet_plant.py tether/tests/test_evt.py \
		tether/tests/test_theory_excursion.py tether/tests/test_theory_reduced_lti.py \
		tether/tests/test_phase0_environment.py tether/tests/test_phase0_events.py \
		tether/tests/test_phase0_plant.py tether/tests/test_phase0_truth_isolation.py tether/tests/test_phase0_weather.py \
		tether/tests/test_v3_transmission.py tether/tests/test_v3_branching.py tether/tests/test_v3_extremal.py \
		tether/tests/test_v3_campaign.py tether/tests/test_v3_reducer.py tether/tests/test_v3_intervention.py \
		tether/tests/test_v3_analyse.py tether/tests/test_v3_present.py tether/tests/test_acc_paper.py

phase0:
	$(PYTHON) -m tether.campaign.phase0 run

# Phase 1 closure on the production plant (stochastic cells (d), impact table, gate).
phase1:
	$(PYTHON) -m tether.campaign.phase1_close run --workers $(WORKERS)
	$(PYTHON) -m tether.analysis.phase1_report

phase2: phase2-predict phase2-compute phase2-analyse

phase2-predict:
	$(PYTHON) -m tether.campaign.phase2 predict

phase2-compute:
	$(PYTHON) -m tether.campaign.phase2 compute --workers $(WORKERS)

phase2-analyse:
	$(PYTHON) -m tether.campaign.phase2 analyse
	$(PYTHON) -m tether.analysis.phase2_report

# Phases 3-6: exploratory continuation after the Phase 2 NOT-GO (verdicts carry no gate authority).
phase3:
	$(PYTHON) -m tether.campaign.phase3 shocks --workers $(WORKERS)
	$(PYTHON) -m tether.campaign.phase3 predict
	$(PYTHON) -m tether.campaign.phase3 compute --workers $(WORKERS)
	$(PYTHON) -m tether.campaign.phase3 analyse
	$(PYTHON) -m tether.analysis.phase3_report

phase4:
	$(PYTHON) -m tether.campaign.phase4 shocks --workers $(WORKERS)
	$(PYTHON) -m tether.campaign.phase4 predict
	$(PYTHON) -m tether.campaign.phase4 compute --workers $(WORKERS)
	$(PYTHON) -m tether.campaign.phase4 analyse
	$(PYTHON) -m tether.analysis.phase4_report

phase5:
	$(PYTHON) -m tether.campaign.phase5 impact-fan --workers $(WORKERS)
	$(PYTHON) -m tether.campaign.phase5 missions --workers $(WORKERS)
	$(PYTHON) -m tether.campaign.phase5 analyse
	$(PYTHON) -m tether.analysis.phase5_figures
	$(PYTHON) -m tether.analysis.phase5_report

phase5-staleness:
	$(PYTHON) -m tether.campaign.phase5 staleness --workers $(WORKERS)
	$(PYTHON) -m tether.campaign.phase5 staleness-analyse
	$(PYTHON) -m tether.analysis.phase5_report

phase6:
	$(PYTHON) -m tether.campaign.phase6 compute --workers $(WORKERS)
	$(PYTHON) -m tether.campaign.phase6 analyse
	$(PYTHON) -m tether.campaign.phase6 hero --workers 2
	$(PYTHON) -m tether.analysis.phase6_figures
	$(PYTHON) -m tether.analysis.phase6_report

# Every v1 figure, regenerated from the campaign outputs of the phases above.
figs:
	$(PYTHON) -m tether.analysis.phase0_figures
	$(PYTHON) -m tether.analysis.phase1_figures
	$(PYTHON) -m tether.analysis.phase1_figures phase1r
	$(PYTHON) -m tether.analysis.phase1_close_figures
	$(PYTHON) -m tether.analysis.phase2_figures
	$(PYTHON) -m tether.analysis.phase5_figures
	$(PYTHON) -m tether.analysis.phase6_figures

# Re-simulate each phase's recorded probe run and check that it reproduces the output.
replay:
	$(PYTHON) -m tether.campaign.phase0 replay
	$(PYTHON) -m tether.campaign.phase1 replay
	$(PYTHON) -m tether.campaign.phase1_t4r replay
	$(PYTHON) -m tether.campaign.phase1_t5r replay
	$(PYTHON) -m tether.campaign.phase1_t9r replay
	$(PYTHON) -m tether.campaign.phase1_close replay
	$(PYTHON) -m tether.campaign.phase2 replay
	$(PYTHON) -m tether.campaign.phase3 replay
	$(PYTHON) -m tether.campaign.phase4 replay

# Historical Phase 1 stage drivers (superseded by phase1; kept for replay).
phase1-mechanics:
	$(PYTHON) -m tether.campaign.phase1 mechanics

phase1-deterministic:
	$(PYTHON) -m tether.campaign.phase1 deterministic

phase1r:
	$(PYTHON) -m tether.campaign.phase1 phase1r

phase1-t4r:
	$(PYTHON) -m tether.campaign.phase1_t4r run

phase1-t5r:
	$(PYTHON) -m tether.campaign.phase1_t5r run

phase1-t9r:
	$(PYTHON) -m tether.campaign.phase1_t9r run

# ---------------------------------------------------------------------------
# Plan v2. The campaign closed at the Phase 1 -> 2 gate (NO-LAUNCH); Phases 2-6 were never
# launched, so there are no targets for them.
# ---------------------------------------------------------------------------
.PHONY: v2 v2-phase0 v2-sway v2-phase1 v2-phase1-scripted v2-phase1-stochastic v2-p5-pretest v2-claim-a

# Phase 0: weather programme admissibility (P0-W1) and the sway pilot.
v2-phase0:
	$(PYTHON) -m tether.campaign.v2.phase0_weather declare
	$(PYTHON) -m tether.campaign.v2.phase0_weather run
	$(PYTHON) -m tether.analysis.v2.phase0_report

# "all" is the original pilot; "addendum-all" is the +-20 deg clamp revision that the executed
# campaign used.
v2-sway:
	$(PYTHON) -m tether.campaign.v2.sway all
	$(PYTHON) -m tether.campaign.v2.sway addendum-all

# Committed predictions for Claim A, hashed before the Phase 1 runs.
v2-claim-a:
	$(PYTHON) -m tether.campaign.v2.claim_a all

# Phase 1: the gating phase. Deterministic cells then stochastic cells.
v2-phase1-scripted:
	$(PYTHON) -m tether.campaign.v2.phase1_scripted --declare --run --addendum

v2-phase1-stochastic:
	$(PYTHON) -m tether.campaign.v2.phase1_stochastic declare
	$(PYTHON) -m tether.campaign.v2.phase1_stochastic compute
	$(PYTHON) -m tether.campaign.v2.phase1_stochastic analyse

v2-phase1: v2-phase1-scripted v2-phase1-stochastic

# Pre-test P5-T2' (about 13 core-hours; reads the v1 Phase 5 missions).
v2-p5-pretest:
	$(PYTHON) -m tether.campaign.v2.p5_pretest all

# The v2 campaign, in order.
v2: v2-phase0 v2-sway v2-claim-a v2-phase1 v2-p5-pretest

# ---------------------------------------------------------------------------
# ACC 2027 paper "Snap-Load Transmission Between Cables Sharing a Towed Payload" (IEEE_ACC_2027/).
# ---------------------------------------------------------------------------
ACCPAPER = IEEE_ACC_2027
.PHONY: acc-paper

acc-paper:
	cd $(ACCPAPER) && latexmk -pdf -interaction=nonstopmode main.tex
