"""The v2 sway (chord-angle) loop: sign, v1 defaults, reduced-model parity (plan v2 IV.7)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tether.campaign.fleet_run import FleetRunSpec, build_run, run_to_end
from tether.campaign.v2.sway import chord_and_misalignment, rotated_start, settling_time
from tether.control.controller import ControllerBank, wrap_angle
from tether.physics import constants, fleet
from tether.physics.sensors import cable_bearing
from tether.theory.reduced_lti import (
    ReducedModelInput,
    cable_terms,
    generalized_forces,
    reduced_model,
    tow_plant,
)

V1_DEFAULT_SPEC_HASH = "ee580e04b96335a81c0a08aee83e9ec64c2b0271d82deed6aa7ec4b08d51d325"


def _perturbed_parallel_state(seed: int = 11) -> tuple[np.ndarray, fleet.FleetGeometry, fleet.OperatingPoint]:
    geometry = fleet.formation_geometry("parallel")
    operating = fleet.operating_point(geometry, 1000.0)
    state = fleet.equilibrium_state(geometry, operating)
    size = 3 * (geometry.vessel_count + 1)
    generator = np.random.default_rng(seed)
    state[:size] += generator.uniform(-0.3, 0.3, size)
    state[2:size:3] += generator.uniform(-0.2, 0.2, geometry.vessel_count + 1)
    return state, geometry, operating


def _controller_yaw(sway_gain: float, heading: float, bearing: float, reference: float = 0.0, **kwargs) -> float:
    controller = ControllerBank(
        1,
        heading_reference=lambda t: np.array([reference]),
        surge_schedule=lambda t: np.array([1300.0]),
        initial_headings=np.array([heading]),
        heading_gain=1000.0,
        trim_gain=100.0,
        sway_gain=sway_gain,
        **kwargs,
    )
    context = controller.CreateDefaultContext()
    controller.get_input_port(0).FixValue(context, np.array([0.9, 0.0, 0.0, 0.0]))
    controller.get_input_port(1).FixValue(context, np.array([bearing, 0.0]))
    values = controller.EvalUniquePeriodicDiscreteUpdate(context).get_vector(0).get_value()
    return float(values[-1])


def test_bearing_is_heading_minus_world_chord_angle():
    """sigma_hat = theta_hat - bearing is the world chord angle; psi = bearing."""
    state, geometry, _ = _perturbed_parallel_state()
    sigma, psi = chord_and_misalignment(state, geometry)
    kinematics = fleet.cable_kinematics(state, geometry, constants.CABLE_REST_LENGTH)
    np.testing.assert_allclose(sigma[0], np.arctan2(kinematics["direction"][:, 1], kinematics["direction"][:, 0]), atol=1e-12)
    headings = state[5 : 3 * (geometry.vessel_count + 1) : 3]
    for i in range(geometry.vessel_count):
        bearing = cable_bearing(state, geometry, i, constants.CABLE_REST_LENGTH)
        assert abs(wrap_angle(headings[i] - bearing - sigma[0, i])) < 1e-12
        assert abs(wrap_angle(psi[0, i] - bearing)) < 1e-12
    terms = cable_terms(tow_plant(ReducedModelInput(pretension=1000.0)), state)
    np.testing.assert_allclose(terms["chord_angle"], sigma[0], atol=1e-12)


def test_default_controller_and_spec_are_v1():
    assert FleetRunSpec().config_hash() == V1_DEFAULT_SPEC_HASH
    assert FleetRunSpec(k_sigma=0.0).config_hash() == V1_DEFAULT_SPEC_HASH
    assert FleetRunSpec(k_sigma=2.0).config_hash() != V1_DEFAULT_SPEC_HASH
    for heading, bearing in ((0.02, -0.05), (-0.3, 0.4)):
        v1 = 1000.0 * wrap_angle(0.0 - heading) - 100.0 * math.sin(bearing)
        assert _controller_yaw(0.0, heading, bearing) == v1


def test_sway_term_turns_the_hull_back_toward_the_design_chord():
    """Vessel swung to port (sigma > 0), hull on the chord: the reference turns starboard."""
    sigma = math.radians(3.0)
    heading = sigma  # hull aligned with its chord, psi = 0
    bearing = heading - sigma
    k_sigma = 2.0
    yaw = _controller_yaw(k_sigma, heading, bearing)
    expected = 1000.0 * wrap_angle(0.0 - k_sigma * sigma - heading) - 100.0 * math.sin(bearing)
    assert yaw == pytest.approx(expected, abs=1e-9)
    assert yaw < _controller_yaw(0.0, heading, bearing) < 0.0
    # The design chord angle is the origin of the term.
    assert _controller_yaw(k_sigma, heading, bearing, design_chord_angles=np.array([sigma])) == pytest.approx(
        _controller_yaw(0.0, heading, bearing), abs=1e-9
    )


@pytest.mark.parametrize("formation", ["parallel", "fan"])
def test_reduced_model_sway_force_law_equals_controller(formation):
    inp = ReducedModelInput(formation=formation, pretension=1000.0, heading_gain=1000.0, sway_gain=3.0)
    plant = tow_plant(inp)
    geometry = fleet.formation_geometry(formation)
    operating = fleet.operating_point(geometry, 1000.0)
    state = fleet.equilibrium_state(geometry, operating)
    size = 3 * (geometry.vessel_count + 1)
    state[:size] += np.random.default_rng(5).uniform(-2e-2, 2e-2, size)
    headings = state[5:size:3]
    bearings = np.array([cable_bearing(state, geometry, i, constants.CABLE_REST_LENGTH) for i in range(geometry.vessel_count)])
    v1_plant = tow_plant(ReducedModelInput(formation=formation, pretension=1000.0, heading_gain=1000.0))
    reference = v1_plant.heading_reference
    target = reference - 3.0 * np.array([wrap_angle(h - b - phi) for h, b, phi in zip(headings, bearings, geometry.cable_angles)])
    yaw = 1000.0 * np.array([wrap_angle(t - h) for t, h in zip(target, headings)]) - 100.0 * np.sin(bearings)
    yaw_v1 = 1000.0 * np.array([wrap_angle(r - h) for r, h in zip(reference, headings)]) - 100.0 * np.sin(bearings)
    zero = np.zeros(2 * (geometry.vessel_count + 1))
    difference = generalized_forces(plant, state, zero) - generalized_forces(v1_plant, state, zero)
    np.testing.assert_allclose(difference[5::3], yaw - yaw_v1, rtol=0.0, atol=1e-9)
    assert np.all(difference[np.setdiff1d(np.arange(difference.size), np.arange(5, difference.size, 3))] == 0.0)


def test_reduced_model_defaults_psi_and_second_order_mean():
    base = reduced_model(ReducedModelInput(pretension=1000.0, heading_gain=1000.0, weather_scale=0.5))
    explicit = reduced_model(ReducedModelInput(pretension=1000.0, heading_gain=1000.0, weather_scale=0.5, sway_gain=0.0))
    for name in ("sigma_e", "sigma_q", "sigma_chord_angle", "mu_q", "state_matrix", "covariance"):
        np.testing.assert_array_equal(getattr(base, name), getattr(explicit, name))
    n = base.geometry.vessel_count
    size = base.state_matrix.shape[0]
    theta_rows = np.zeros((n, base.covariance.shape[0]))
    for i in range(n):
        theta_rows[i, 3 + 3 * i] = 1.0
    np.testing.assert_allclose(base.outputs["psi"], theta_rows - base.outputs["chord_angle"], atol=1e-6)
    np.testing.assert_allclose(base.sigma_psi, np.sqrt(np.diag(base.output_covariance("psi"))), rtol=1e-12)
    drag = constants.VESSEL_LINEAR_DRAG * base.operating.speed
    expected = base.mu_q + 0.5 * (base.mu_q + drag) * base.sigma_chord_angle**2 - 0.5 * base.operating.thrusts * base.sigma_psi**2
    np.testing.assert_allclose(base.mu_q2, expected, rtol=1e-12)
    assert size == 6 * n + 4


def test_sway_gain_moves_the_lateral_mode_the_right_way():
    """At the unstable k_h = 500 the corrective sign stabilizes; the reversed sign destabilizes further."""
    def worst(k_sigma: float) -> float:
        result = reduced_model(ReducedModelInput(pretension=1000.0, heading_gain=500.0, sway_gain=k_sigma))
        return float(np.max(np.linalg.eigvals(result.state_matrix).real))

    assert worst(1.0) < worst(0.0) < worst(-1.0)
    assert worst(1.0) < 0.0


def test_rotated_start_is_a_pure_chord_angle_step():
    geometry = fleet.formation_geometry("parallel")
    operating = fleet.operating_point(geometry, 1000.0)
    design = fleet.equilibrium_state(geometry, operating)
    start = rotated_start(geometry, operating, fleet.CableParameters(), vessel=2, angle_deg=2.0)
    sigma0, psi0 = chord_and_misalignment(design, geometry)
    sigma1, psi1 = chord_and_misalignment(start, geometry)
    np.testing.assert_allclose(sigma1[0] - sigma0[0], [0, 0, math.radians(2.0), 0, 0], atol=1e-12)
    np.testing.assert_allclose(psi1, psi0, atol=1e-12)
    lengths = [fleet.cable_kinematics(s, geometry, constants.CABLE_REST_LENGTH)["length"] for s in (design, start)]
    np.testing.assert_allclose(lengths[0], lengths[1], atol=1e-12)
    np.testing.assert_array_equal(start[18:], design[18:])


def test_settling_time_rule():
    times = np.arange(6) * 1.0
    assert settling_time(times, np.array([1.0, 0.5, 0.2, 0.05, 0.0, 0.01]), 0.1) == (2.0, False)
    assert settling_time(times, np.array([1.0, 0.5, 0.2, 0.05, 0.0, 0.2]), 0.1) == (5.0, True)


def test_custom_reference_with_sway_is_refused():
    spec = FleetRunSpec(pretension=1000.0, weather_scale=0.0, duration=0.1, warmup=0.0, k_sigma=1.0)
    with pytest.raises(NotImplementedError):
        build_run(spec, 1, heading_reference=lambda t: np.zeros(5))


def _step_response(k_sigma: float, horizon: float = 20.0) -> np.ndarray:
    geometry = fleet.formation_geometry("parallel")
    operating = fleet.operating_point(geometry, 1000.0)
    start = rotated_start(geometry, operating, fleet.CableParameters(), vessel=2, angle_deg=2.0)
    spec = FleetRunSpec(pretension=1000.0, heading_gain=1000.0, weather_scale=0.0, duration=horizon, warmup=0.0, k_sigma=k_sigma)
    run = run_to_end(build_run(spec, 99, initial_state=start))
    log = run.fleet.cables.log
    sigma, _ = chord_and_misalignment(log.state[: log.state_count], geometry)
    return sigma[:, 2]


def test_plant_sway_loop_corrects_a_lateral_offset_and_the_reversed_sign_amplifies_it():
    """Deterministic 2 deg chord-angle offset on the plant, zero weather, 20 s.

    Turning the hull swings its stern attachment the other way first, so the corrected
    response may rise briefly before it returns; the test reads the last 5 s.
    """
    step = math.radians(2.0)
    corrected = _step_response(2.0)
    reversed_sign = _step_response(-2.0)
    assert np.max(np.abs(corrected[-500:])) < 0.5 * step
    assert np.max(np.abs(reversed_sign)) > 1.5 * step


def test_pooled_circular_std_matches_summaries():
    from tether.campaign.summaries import circular_std
    from tether.campaign.v2.sway import _circular_std_from_sums

    angles = np.random.default_rng(3).normal(0.1, 0.3, size=(400, 5))
    pooled = _circular_std_from_sums(np.cos(angles).sum(axis=0), np.sin(angles).sum(axis=0), angles.shape[0])
    np.testing.assert_allclose(pooled, circular_std(angles), rtol=1e-12)


# --------------------------------------------------------------------------- addendum 1: the clamp


def test_sway_limit_defaults_to_the_unsaturated_law():
    """sway_limit = None is the unsaturated law bit-exactly, and specs keep their earlier hashes."""
    from tether.control.controller import SWAY_LIMIT

    assert SWAY_LIMIT is None
    assert FleetRunSpec(sway_limit=None).config_hash() == V1_DEFAULT_SPEC_HASH
    unsaturated = FleetRunSpec(k_sigma=3.0).config_hash()
    assert FleetRunSpec(k_sigma=3.0, sway_limit=None).config_hash() == unsaturated
    assert FleetRunSpec(k_sigma=3.0, sway_limit=0.349).config_hash() != unsaturated
    for heading, bearing in ((0.02, -0.05), (0.7, -0.4), (-1.4, 0.3)):
        assert _controller_yaw(3.0, heading, bearing) == _controller_yaw(3.0, heading, bearing, sway_limit=None)


def test_sway_limit_is_inert_inside_its_linear_range_and_clips_outside():
    limit = 0.349
    # 3 deg chord deviation at k_sigma = 3: 9 deg correction, inside the 20 deg clamp.
    sigma = math.radians(3.0)
    assert _controller_yaw(3.0, sigma, 0.0, sway_limit=limit) == _controller_yaw(3.0, sigma, 0.0)
    for sigma_deg in (10.0, 40.0, 90.0, 120.0, -60.0, 170.0):
        sigma = math.radians(sigma_deg)
        heading, bearing = sigma, 0.0  # hull on its chord
        clipped = min(max(3.0 * wrap_angle(sigma), -limit), limit)
        expected = 1000.0 * wrap_angle(0.0 - clipped - heading) - 100.0 * math.sin(bearing)
        assert _controller_yaw(3.0, heading, bearing, sway_limit=limit) == pytest.approx(expected, abs=1e-9)


def test_saturated_reference_never_aliases_onto_a_swung_chord():
    """Unsaturated, a chord at 360/(1 + k_sigma) deg with the hull on it gives zero heading error
    (the parking mechanism); saturated, the reference stays within the limit of theta_0."""
    limit = 0.349
    for k_sigma in (2.0, 3.0, 5.0):
        sigma = 2.0 * math.pi / (1.0 + k_sigma)
        heading, bearing = sigma, 0.0
        assert abs(_controller_yaw(k_sigma, heading, bearing)) < 1e-6
        yaw = _controller_yaw(k_sigma, heading, bearing, sway_limit=limit)
        assert yaw == pytest.approx(1000.0 * wrap_angle(-limit - sigma), abs=1e-9)
        assert yaw < -1000.0 * limit


def test_sway_limit_must_be_positive():
    with pytest.raises(ValueError):
        _controller_yaw(3.0, 0.0, 0.0, sway_limit=0.0)


def test_build_run_threads_the_sway_limit():
    spec = FleetRunSpec(pretension=1000.0, weather_scale=0.0, duration=0.1, warmup=0.0, k_sigma=3.0, sway_limit=0.349)
    assert build_run(spec, 1).controller._sway_limit == 0.349
    assert build_run(FleetRunSpec(pretension=1000.0, weather_scale=0.0, duration=0.1, warmup=0.0, k_sigma=3.0), 1).controller._sway_limit is None


def test_load_relative_chord_removes_a_rigid_formation_yaw():
    from tether.campaign.v2.sway import load_relative_chord

    geometry = fleet.formation_geometry("parallel")
    # Every parallel attachment lies on the load's forward face, whose normal is the body x-axis.
    np.testing.assert_allclose(geometry.load_offsets[:, 0], geometry.load_offsets[0, 0], atol=1e-12)
    operating = fleet.operating_point(geometry, 1000.0)
    state = fleet.equilibrium_state(geometry, operating).copy()
    size = 3 * (geometry.vessel_count + 1)
    yaw = math.radians(12.0)
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    positions = state[:size].reshape(-1, 3)
    positions[:, :2] = positions[:, :2] @ rotation.T
    positions[:, 2] += yaw
    state[:size] = positions.ravel()
    sigma, _ = chord_and_misalignment(state, geometry)
    np.testing.assert_allclose(sigma[0], yaw, atol=1e-12)
    np.testing.assert_allclose(load_relative_chord(state, sigma, geometry)[0], 0.0, atol=1e-12)


def test_statistical_linearization_reduces_to_the_linear_model_for_a_wide_clamp():
    from tether.campaign.v2.sway import saturated_equivalent_prediction

    wide = saturated_equivalent_prediction(1000.0, 0.35, 3.0, limit=50.0)
    assert wide["equivalent_gain"] == pytest.approx(3.0, rel=1e-6)
    assert wide["chord_max_deg"] == pytest.approx(wide["linear_model_chord_max_deg"], rel=1e-4)
    tight = saturated_equivalent_prediction(1000.0, 0.35, 3.0)
    assert tight["equivalent_gain"] < 3.0 and tight["chord_max_deg"] > tight["linear_model_chord_max_deg"]


def _large_offset_chord(sway_limit, k_sigma: float = 3.0, angle_deg: float = 90.0, horizon: float = 40.0) -> np.ndarray:
    geometry = fleet.formation_geometry("parallel")
    operating = fleet.operating_point(geometry, 1000.0)
    start = rotated_start(geometry, operating, fleet.CableParameters(), vessel=2, angle_deg=angle_deg)
    spec = FleetRunSpec(pretension=1000.0, heading_gain=763.0, trim_gain=100.0, weather_scale=0.0, duration=horizon, warmup=0.0,
                        k_sigma=k_sigma, sway_limit=sway_limit)
    run = run_to_end(build_run(spec, 7000, initial_state=start))
    log = run.fleet.cables.log
    sigma, _ = chord_and_misalignment(log.state[: log.state_count], geometry)
    return sigma[:, 2]


def test_saturated_law_returns_from_a_90_deg_offset_where_the_unsaturated_law_parks():
    """Zero weather, k_h = 763, k_sigma = 3, vessel 2 rotated 90 deg about its load attachment.

    The unsaturated law parks near 100 deg (the verifier's finding); the clamped law returns.
    """
    saturated = _large_offset_chord(0.349)
    assert np.max(np.abs(saturated[-500:])) < math.radians(5.0)
    unsaturated = _large_offset_chord(None)
    assert np.min(np.abs(unsaturated[-500:])) > math.radians(60.0)
