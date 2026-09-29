"""Planar attachment-point geometry, independent of Drake."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def planar_rotation(angle: float) -> NDArray[np.float64]:
    """Return the active body-to-world planar rotation."""
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=float)


def attachment_position(
    body_position: ArrayLike,
    body_angle: float,
    attachment_offset: ArrayLike,
) -> NDArray[np.float64]:
    """Return a body-fixed attachment point expressed in world coordinates."""
    return np.asarray(body_position, dtype=float) + planar_rotation(body_angle) @ np.asarray(
        attachment_offset, dtype=float
    )


def attachment_velocity(
    translational_velocity: ArrayLike,
    angular_rate: float,
    body_angle: float,
    attachment_offset: ArrayLike,
) -> NDArray[np.float64]:
    """Return attachment velocity from planar spatial velocity data."""
    offset_world = planar_rotation(body_angle) @ np.asarray(attachment_offset, dtype=float)
    rotational_velocity = angular_rate * np.array([-offset_world[1], offset_world[0]])
    return np.asarray(translational_velocity, dtype=float) + rotational_velocity


def attachment_length(first_position: ArrayLike, second_position: ArrayLike) -> float:
    """Return distance between two world-frame attachment positions."""
    return float(np.linalg.norm(np.asarray(second_position) - np.asarray(first_position)))


def attachment_length_rate(
    first_position: ArrayLike,
    first_velocity: ArrayLike,
    second_position: ArrayLike,
    second_velocity: ArrayLike,
) -> float:
    """Return the signed rate of distance between two attachment points."""
    displacement = np.asarray(second_position, dtype=float) - np.asarray(first_position, dtype=float)
    length = np.linalg.norm(displacement)
    if length == 0.0:
        raise ValueError("attachment length rate is undefined at zero length")
    relative_velocity = np.asarray(second_velocity, dtype=float) - np.asarray(
        first_velocity, dtype=float
    )
    return float(displacement @ relative_velocity / length)