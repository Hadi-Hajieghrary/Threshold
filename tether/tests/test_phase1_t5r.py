"""P1-T5R deterministic identification and replay tests."""

import numpy as np

from tether.physics.phase1_deterministic import PRETENSION_LEVELS
from tether.physics.phase1_t5r import (
    P1_T5R_NOMINAL_SPEEDS,
    P1_T5R_PRODUCTION_STEP,
    preregistered_cell_specs,
)


def test_preregistered_split_is_disjoint_and_covers_grid():
    production = [
        spec
        for spec in preregistered_cell_specs()
        if np.isclose(spec.physics_step_s, P1_T5R_PRODUCTION_STEP)
    ]
    training = {spec.identifier for spec in production if spec.split == "train"}
    heldout = {spec.identifier for spec in production if spec.split == "heldout"}

    assert training.isdisjoint(heldout)
    assert training | heldout == {spec.identifier for spec in production}
    assert len(training) == 14
    assert len(heldout) == 7
    assert all(
        spec.split == "train"
        for spec in production
        if np.isclose(spec.nominal_speed_mps, 0.0)
    )
    for speed in P1_T5R_NOMINAL_SPEEDS:
        assert any(
            np.isclose(spec.nominal_speed_mps, speed) and spec.split == "train"
            for spec in production
        )
        assert any(
            np.isclose(spec.nominal_speed_mps, speed) and spec.split == "heldout"
            for spec in production
        )
    for pretension in PRETENSION_LEVELS:
        assert any(
            np.isclose(spec.pretension_n, pretension) and spec.split == "train"
            for spec in production
        )
        assert any(
            np.isclose(spec.pretension_n, pretension) and spec.split == "heldout"
            for spec in production
        )
