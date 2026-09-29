"""Plan v3 WP0: the transmission theory (closed form, operator with yaw, exact linear response)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tether.campaign.v2.events import ENGAGEMENT_PERIOD
from tether.physics import constants as C
from tether.theory import reduced_lti as R
from tether.theory import transmission as TR


def test_closed_form_is_0440_and_stiffness_free():
    assert abs(TR.closed_form_ratio() - 0.43994) < 1e-4
    for factor in (0.25, 0.5, 1.0, 2.0, 4.0):
        assert abs(TR.closed_form_ratio_from_modes(factor * C.CABLE_STIFFNESS) - TR.closed_form_ratio()) < 1e-12


def test_frequencies_pin_omega_e_omega_L_epsilon_088():
    f = TR.frequencies()
    assert round(f["omega_e"], 1) == 18.7 and round(f["omega_L"], 1) == 16.5
    assert abs(f["epsilon"] - 0.88) < 0.005
    assert abs(f["engagement_period"] - ENGAGEMENT_PERIOD) < 1e-12
    assert round(f["contact_duration"], 2) == 0.17 and round(f["load_half_period"], 2) == 0.19
    # epsilon does not depend on the stiffness; the contact duration does
    assert abs(TR.frequencies(2.0 * C.CABLE_STIFFNESS)["epsilon"] - f["epsilon"]) < 1e-12
    assert abs(TR.frequencies(4.0 * C.CABLE_STIFFNESS)["contact_duration"] - f["contact_duration"] / 2.0) < 1e-12


def test_pulse_factors_and_threshold():
    assert TR.pulse_factor("half_sine") == 1.0 and abs(TR.pulse_factor("rectangular") - math.pi / 2) < 1e-12
    assert round(TR.unloading_threshold(TR.closed_form_ratio()), 1) == 2.3
    assert round(TR.unloading_threshold(TR.closed_form_ratio() * TR.pulse_factor("rectangular")), 1) == 1.4
    with pytest.raises(ValueError):
        TR.pulse_factor("triangular")


def test_fan_operator_has_zero_levers_and_cosine_entries():
    geometry = R.tow_geometry("fan", arc_half_angle=0.55)
    directions = np.stack([np.cos(geometry.cable_angles), np.sin(geometry.cable_angles)], axis=1)
    levers = TR.lever_matrix(geometry.load_offsets, 0.0, directions)[2]
    assert np.max(np.abs(levers)) < 1e-9
    operator = TR.formation_operator("fan", 0.55)
    expected = TR.closed_form_ratio() * np.cos(geometry.cable_angles[:, None] - geometry.cable_angles[None, :])
    assert np.allclose(operator, expected, atol=1e-9)
    assert np.allclose(TR.geometry_factor(operator), np.cos(geometry.cable_angles[:, None] - geometry.cable_angles[None, :]))


def test_parallel_operator_outer_self_term_066():
    operator = TR.formation_operator("parallel")
    assert np.allclose(operator, operator.T)
    assert abs(operator[0, 0] - 0.660) < 0.005 and abs(operator[2, 2] - 0.440) < 0.001
    assert abs(operator[0, 4] - 0.220) < 0.005 and abs(operator[0, 1] - 0.541) < 0.005
    # the centre cable has no lever arm, so its column is the translation-only 0.44
    assert np.allclose(operator[:, 2], TR.closed_form_ratio(), atol=1e-9)


def test_zoh_response_matches_an_independent_integrator():
    m, K, dt = 2.0, 50.0, 1e-4
    a = np.array([[0.0, 1.0], [-K / m, 0.0]])
    b = np.array([[0.0], [1.0 / m]])
    c = np.array([[1.0, 0.0]])
    times = np.arange(0.0, 0.5, dt)
    u = np.where(times < 0.1, np.sin(math.pi * times / 0.1), 0.0)
    ours = TR.zoh_response(a, b, c, u, dt)[:, 0]

    def rhs(t, y):
        force = math.sin(math.pi * t / 0.1) if t < 0.1 else 0.0
        return [y[1], (-K * y[0] + force) / m]

    reference = solve_ivp(rhs, (0.0, 0.5), [0.0, 0.0], t_eval=np.arange(0.0, 0.5 + dt / 2, dt), rtol=1e-10, atol=1e-13)
    # the zero-order hold of the input costs O(dt) relative to the continuous pulse
    assert np.max(np.abs(ours[: reference.y.shape[1]] - reference.y[0])) < 1e-3 * np.max(np.abs(reference.y[0]))


def _chain_drop(parent: int = 2, free_vessels: bool = True, drag: bool = True) -> float:
    """Collinear five-cable chain, parent's spring removed, integrated independently of the module."""
    mL, mA, k, c = C.LOAD_MASS, C.VESSEL_MASS, C.CABLE_STIFFNESS, C.CABLE_DAMPING
    f = TR.frequencies()
    tau, horizon, n = math.pi / f["omega_e"], f["engagement_period"], 5

    def rhs(t, y):
        x, v = y[: n + 1], y[n + 1 :]
        a = np.zeros(n + 1)
        pulse = math.sin(f["omega_e"] * t) if 0.0 <= t < tau else 0.0
        a[0] += pulse / mL
        a[1 + parent] -= pulse / mA
        for i in range(n):
            if i == parent:
                continue
            q = k * (x[1 + i] - x[0]) + c * (v[1 + i] - v[0])
            a[0] += q / mL
            if free_vessels:
                a[1 + i] -= q / mA
        if drag:
            a[0] -= C.LOAD_LINEAR_DRAG * v[0] / mL
            a[1:] -= C.VESSEL_LINEAR_DRAG * v[1:] / mA
        if not free_vessels:
            a[1:] = 0.0
        return np.concatenate([v, a])

    sol = solve_ivp(rhs, (0.0, horizon), np.zeros(2 * (n + 1)), max_step=1e-4, rtol=1e-9, atol=1e-12, dense_output=True)
    ts = np.linspace(0.0, horizon, 4000)
    y = sol.sol(ts)
    i = 0 if parent != 0 else 1
    q = k * (y[1 + i] - y[0]) + c * (y[n + 1 + 1 + i] - y[n + 1])
    return float(-q.min())


def test_exact_response_matches_the_independent_chain():
    spec = TR.ResponseSpec(R.ReducedModelInput(pretension=600.0, heading_gain=500.0, weather_scale=0.5))
    module = TR.exact_response(spec, 2).drop
    chain = _chain_drop()
    assert np.isnan(module[2]) and np.allclose(module[[0, 1, 3, 4]], module[0], atol=1e-6)
    assert abs(module[0] - chain) < 2e-3, (module[0], chain)
    assert abs(chain - 0.165) < 0.01  # the plant's transmission is a third of the closed form


def test_exact_response_recovers_the_closed_form_under_its_two_hypotheses():
    """Pinned vessel ends + an impulsive pulse (same impulse) + no drag give the paper's 0.44."""
    spec = TR.ResponseSpec(
        R.ReducedModelInput(pretension=600.0, heading_gain=500.0, weather_scale=0.5),
        damping=0.0,  # the closed form carries no damper term (c edot adds ~1.5 % otherwise)
        inertia=(C.LOAD_MASS, C.LOAD_YAW_INERTIA, 1.0e9, 1.0e9),
        pulse_duration=1.0e-3,
        drag_scale=0.0,
        step=1.0e-5,
    )
    drop = TR.exact_response(spec, 2).drop
    assert np.allclose(drop[[0, 1, 3, 4]], TR.closed_form_ratio(), rtol=0.01), drop
    # relaxing the hypotheses one at a time lowers the response: finite pulse, then free vessels
    finite = TR.exact_response(TR.ResponseSpec(spec.inp, inertia=spec.inertia, drag_scale=0.0), 2).drop[0]
    assert 0.30 < finite < 0.35, finite
    free = TR.exact_response(TR.ResponseSpec(spec.inp, drag_scale=0.0), 2).drop[0]
    assert 0.16 < free < 0.20 and free < finite, free
    assert abs(free - _chain_drop(drag=False)) < 2e-3


def test_response_matrix_and_pair_table_are_consistent():
    spec = TR.ResponseSpec(R.ReducedModelInput(formation="fan", arc_half_angle=0.55, pretension=600.0, heading_gain=500.0))
    table = TR.predicted_pair_table(spec, "FAN_T600_I0.5")
    r = np.array([[np.nan if v is None else v for v in row] for row in table["response_M1_half_sine"]])
    assert np.all(np.isnan(np.diag(r))) and np.all(np.isfinite(r[~np.eye(5, dtype=bool)]))
    assert table["status"].startswith("FORECAST")
    assert 0.1 < table["band_statistic_median_offdiag"] < 0.3
    # rectangular pulses carry pi/2 the impulse and transmit more
    rect = np.array([[np.nan if v is None else v for v in row] for row in table["response_M1_rectangular"]])
    assert np.nanmedian(rect) > np.nanmedian(r)
