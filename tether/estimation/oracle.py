"""Declared Phase 0 oracle arm."""

from tether.physics.plant import TransitLog

TRUTH_ISOLATION_EXEMPTIONS = {"tether.physics.plant"}


class OracleArm:
    """Oracle whose plant-side dependency is explicit and linted."""

    name = "oracle"
    observation_type = TransitLog