"""Focused Phase 0 plant and geometry acceptance tests."""

import numpy as np
import pytest
from pydrake.multibody.plant import DiscreteContactApproximation

from tether.physics import constants
from tether.physics.geometry import (
    attachment_length,
    attachment_length_rate,
    attachment_position,
    attachment_velocity,
)
from tether.physics.plant import build_phase0_plant, run_steady_transit, steady_state_values


@pytest.mark.parametrize("thrust", constants.THRUST_LEVELS)
def test_p0_t2_steady_state_identity(thrust):
    expected_speed, expected_tension = steady_state_values(thrust)
    result = run_steady_transit(thrust)

    assert abs(result.speed / expected_speed - 1.0) <= 0.02
    assert abs(result.mean_tension / expected_tension - 1.0) <= 0.02


def test_p0_t2_uses_sap():
    model = build_phase0_plant()
    assert model.plant.get_discrete_contact_approximation() == DiscreteContactApproximation.kSap


def test_p0_t3_attachment_rate_matches_independent_central_difference():
    first_position = np.array([1.2, -0.7])
    second_position = np.array([8.1, 4.3])
    first_angle = 0.37
    second_angle = -0.61
    first_offset = np.array([1.1, -0.4])
    second_offset = np.array([-0.8, 0.9])
    first_velocity = np.array([0.8, -0.3])
    second_velocity = np.array([-0.2, 1.4])
    first_rate = 0.73
    second_rate = -0.46

    first_attachment = attachment_position(first_position, first_angle, first_offset)
    second_attachment = attachment_position(second_position, second_angle, second_offset)
    first_attachment_velocity = attachment_velocity(
        first_velocity, first_rate, first_angle, first_offset
    )
    second_attachment_velocity = attachment_velocity(
        second_velocity, second_rate, second_angle, second_offset
    )
    production_rate = attachment_length_rate(
        first_attachment,
        first_attachment_velocity,
        second_attachment,
        second_attachment_velocity,
    )

    step = 1.0e-5

    def length_at(time):
        first = attachment_position(
            first_position + time * first_velocity,
            first_angle + time * first_rate,
            first_offset,
        )
        second = attachment_position(
            second_position + time * second_velocity,
            second_angle + time * second_rate,
            second_offset,
        )
        return attachment_length(first, second)

    oracle_rate = (length_at(step) - length_at(-step)) / (2.0 * step)
    assert production_rate == pytest.approx(oracle_rate, abs=1.0e-6)