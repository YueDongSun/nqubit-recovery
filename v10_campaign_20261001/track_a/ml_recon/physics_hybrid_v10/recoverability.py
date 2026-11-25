"""Operational recoverability, same-forward Jacobian, and amplitude/phase Shapley."""

from __future__ import annotations

import hashlib
import inspect
from typing import Callable, Dict

import numpy as np

from .control_solver import solve_control


def _forward_sha(forward: Callable[[np.ndarray], np.ndarray]) -> str:
    try:
        source = inspect.getsource(forward)
    except (OSError, TypeError):
        source = repr(forward)
    identity = f"{forward.__module__}|{getattr(forward, '__qualname__', repr(forward))}|{source}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def same_forward_jacobian(forward: Callable[[np.ndarray], np.ndarray], u0: np.ndarray,
                          epsilon: float | np.ndarray = 1e-6) -> np.ndarray:
    u = np.asarray(u0, dtype=np.float64).reshape(-1)
    base = np.asarray(forward(u), dtype=np.float64).reshape(-1)
    J = np.empty((base.size, u.size), dtype=np.float64)
    steps = np.broadcast_to(np.asarray(epsilon, dtype=np.float64), u.shape)
    if np.any(steps <= 0.0):
        raise ValueError("every finite-difference step must be positive")
    for j in range(u.size):
        plus, minus = u.copy(), u.copy()
        plus[j] += steps[j]
        minus[j] -= steps[j]
        J[:, j] = (np.asarray(forward(plus)).reshape(-1) -
                   np.asarray(forward(minus)).reshape(-1)) / (2.0 * steps[j])
    return J


def same_forward_jacobian_audit(
    forward: Callable[[np.ndarray], np.ndarray],
    u0: np.ndarray,
    epsilon: float | np.ndarray = 1e-6,
    *,
    forward_model_sha: str | None = None,
) -> Dict[str, object]:
    u = np.asarray(u0, dtype=np.float64).reshape(-1)
    steps = np.broadcast_to(np.asarray(epsilon, dtype=np.float64), u.shape).copy()
    return {
        "jacobian": same_forward_jacobian(forward, u, steps),
        "finite_difference_steps": steps,
        "forward_model_sha": forward_model_sha or _forward_sha(forward),
        "same_forward_verified": True,
    }


def total_covariance(*components: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    if not components:
        raise ValueError("at least one covariance component is required")
    result = np.zeros_like(np.asarray(components[0], dtype=np.float64))
    for component in components:
        result = result + np.asarray(component, dtype=np.float64)
    if result.ndim == 1:
        return np.maximum(result, floor)
    return result + floor * np.eye(result.shape[0])


def recoverability(r: np.ndarray, J: np.ndarray, Sigma: np.ndarray, **solver_kwargs: object
                   ) -> Dict[str, object]:
    solved = solve_control(r, J, Sigma, **solver_kwargs)
    residual = np.asarray(r, dtype=np.float64)
    W = np.asarray(solved["W"])
    perpendicular = np.asarray(solved["r_perp"])
    total = float(residual @ W @ residual)
    geometry_perpendicular = np.asarray(solved["geometry_r_perp"])
    geometry_reduction = total - float(geometry_perpendicular @ W @ geometry_perpendicular)
    regularized_reduction = float(solved["objective_reduction"])
    kappa = float(np.clip(regularized_reduction / (total + 1e-15), 0.0, 1.0))
    geometry_kappa = float(np.clip(geometry_reduction / (total + 1e-15), 0.0, 1.0))
    epsilon = float(np.sqrt(max(perpendicular @ W @ perpendicular, 0.0) /
                            (total + 1e-15)))
    return {**solved, "kappa_ctrl": kappa, "epsilon_irr": epsilon,
            "geometry_kappa_ctrl": geometry_kappa,
            "objective_reduction": regularized_reduction,
            "explained_energy": regularized_reduction, "total_energy": total}


def amplitude_phase_shapley(r: np.ndarray, J_amplitude: np.ndarray, J_phase: np.ndarray,
                            Sigma: np.ndarray, **kwargs: object) -> Dict[str, float]:
    def value(J: np.ndarray) -> float:
        return float(recoverability(r, J, Sigma, **kwargs)["objective_reduction"])
    empty = 0.0
    a = value(J_amplitude)
    p = value(J_phase)
    ap = value(np.concatenate((J_amplitude, J_phase), axis=1))
    phi_a = 0.5 * (a - empty) + 0.5 * (ap - p)
    phi_p = 0.5 * (p - empty) + 0.5 * (ap - a)
    total = float(recoverability(r, np.concatenate((J_amplitude, J_phase), axis=1),
                                 Sigma, **kwargs)["total_energy"])
    return {"phi_amp": phi_a, "phi_phase": phi_p,
            "chi_amp": phi_a / (total + 1e-15),
            "chi_phase": phi_p / (total + 1e-15),
            "kappa_ctrl": ap / (total + 1e-15),
            "value_definition": "regularized_objective_reduction"}
