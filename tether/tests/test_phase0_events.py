"""Phase 0 crossing interpolation acceptance test."""

from tether.physics.events import synthetic_crossing_benchmark


def test_p0_t7_curved_non_grid_aligned_crossing():
    benchmark = synthetic_crossing_benchmark()
    assert benchmark["time_order"] >= 1.8
    assert benchmark["finest_speed_relative_error"] < 0.01