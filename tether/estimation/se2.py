"""SE(2) on pose vectors ``(x, y, theta)`` and twists ``(rho_x, rho_y, phi)``.

Poses act on points as ``g . p = R(theta) p + (x, y)``.  The left-invariant error of an
estimate ``g_hat`` against the truth ``g`` is ``Log(g_hat^-1 g)``, so ``g = g_hat Exp(xi)``
(a perturbation in the estimate's body frame).  Drake-free.
"""

from __future__ import annotations

import math

import numpy as np

SMALL_ANGLE = 1.0e-6


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rotation(theta: float) -> np.ndarray:
    cosine, sine = math.cos(theta), math.sin(theta)
    return np.array([[cosine, -sine], [sine, cosine]])


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cosine, sine = math.cos(a[2]), math.sin(a[2])
    return np.array(
        [a[0] + cosine * b[0] - sine * b[1], a[1] + sine * b[0] + cosine * b[1], wrap(a[2] + b[2])]
    )


def inverse(g: np.ndarray) -> np.ndarray:
    cosine, sine = math.cos(g[2]), math.sin(g[2])
    return np.array([-cosine * g[0] - sine * g[1], sine * g[0] - cosine * g[1], wrap(-g[2])])


def act(g: np.ndarray, point: np.ndarray) -> np.ndarray:
    cosine, sine = math.cos(g[2]), math.sin(g[2])
    return np.array([g[0] + cosine * point[0] - sine * point[1], g[1] + sine * point[0] + cosine * point[1]])


def _one_minus_cos(phi: float) -> float:
    """``1 - cos(phi)`` without cancellation (``1 - cos`` loses 1e-4 relative at 1e-6 rad)."""
    return 2.0 * math.sin(0.5 * phi) ** 2


def _sinc_terms(phi: float) -> tuple[float, float]:
    """``(sin(phi)/phi, (1 - cos(phi))/phi)`` with their series near zero."""
    if abs(phi) < SMALL_ANGLE:
        return 1.0 - phi * phi / 6.0, 0.5 * phi - phi**3 / 24.0
    return math.sin(phi) / phi, _one_minus_cos(phi) / phi


def exp(xi: np.ndarray) -> np.ndarray:
    a, b = _sinc_terms(float(xi[2]))
    return np.array([a * xi[0] - b * xi[1], b * xi[0] + a * xi[1], wrap(float(xi[2]))])


def log(g: np.ndarray) -> np.ndarray:
    phi = wrap(float(g[2]))
    half = 0.5 * phi
    if abs(phi) < SMALL_ANGLE:
        half_cot = 1.0 - phi * phi / 12.0
    else:
        half_cot = half * math.cos(half) / math.sin(half)
    return np.array([half_cot * g[0] + half * g[1], -half * g[0] + half_cot * g[1], phi])


def adjoint(g: np.ndarray) -> np.ndarray:
    """``Ad_g`` with ``g Exp(xi) g^-1 = Exp(Ad_g xi)``."""
    cosine, sine = math.cos(g[2]), math.sin(g[2])
    return np.array([[cosine, -sine, g[1]], [sine, cosine, -g[0]], [0.0, 0.0, 1.0]])


def left_invariant_error(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return log(compose(inverse(estimate), truth))


def right_jacobian(xi: np.ndarray) -> np.ndarray:
    """``J_r`` with ``Exp(xi + d) = Exp(xi) Exp(J_r(xi) d + O(d^2))``."""
    rho_x, rho_y, phi = float(xi[0]), float(xi[1]), float(xi[2])
    a, b = _sinc_terms(phi)
    if abs(phi) < SMALL_ANGLE:
        c = phi / 6.0
        top = c * rho_x - 0.5 * rho_y
        middle = 0.5 * rho_x + c * rho_y
    else:
        residual, versine = phi - math.sin(phi), _one_minus_cos(phi)
        top = (rho_x * residual - rho_y * versine) / (phi * phi)
        middle = (rho_x * versine + rho_y * residual) / (phi * phi)
    return np.array([[a, b, top], [-b, a, middle], [0.0, 0.0, 1.0]])


def right_jacobian_inverse(xi: np.ndarray) -> np.ndarray:
    jacobian = right_jacobian(xi)
    a, b = jacobian[0, 0], jacobian[0, 1]
    scale = 1.0 / (a * a + b * b)
    block = scale * np.array([[a, -b], [b, a]])
    result = np.eye(3)
    result[:2, :2] = block
    result[:2, 2] = -block @ jacobian[:2, 2]
    return result
