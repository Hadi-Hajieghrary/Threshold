"""Shared deterministic campaign fixtures.

Imports stay inside the fixtures so that Drake-free packages (evt/, theory/) can be
tested without loading the simulator (plan IV.2).
"""

import pytest


@pytest.fixture(scope="session")
def phase1r_results():
    from tether.physics.phase1_deterministic import phase1r_excursion_acceptance

    return phase1r_excursion_acceptance()


@pytest.fixture(scope="session")
def phase1r_campaign_inputs(phase1r_results):
    from tether.physics.phase1_deterministic import deterministic_fleet_acceptance
    from tether.physics.phase1_mechanics import mechanics_acceptance

    return {
        "mechanics_bundle": mechanics_acceptance(
            include_evidence=True,
            _legacy_phase1r_sampling=True,
        ),
        "fleet_bundle": deterministic_fleet_acceptance(include_evidence=True),
        "excursion_bundle": phase1r_results,
    }

