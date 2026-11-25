"""Whitened regularized control QP with explicit geometry diagnostics."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


BOX_COORDINATE_MAX_ITERATIONS = 128
BOX_COORDINATE_TOLERANCE = 1e-10


def solve_control(
    r: np.ndarray,
    J: np.ndarray,
    Sigma: np.ndarray,
    lambda_u: float = 1e-6,
    graph_laplacian: Optional[np.ndarray] = None,
    lambda_graph: float = 0.0,
    action_cost: Optional[np.ndarray] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    inequality_matrix: Optional[np.ndarray] = None,
    inequality_upper: Optional[np.ndarray] = None,
    solver_backend: str = "auto",
) -> Dict[str, np.ndarray | float | str | bool]:
    """Minimize ``(r-Ju)^T W (r-Ju) + u^T R u`` under linear bounds."""
    residual = np.asarray(r, dtype=np.float64).reshape(-1)
    jacobian = np.asarray(J, dtype=np.float64)
    covariance = np.asarray(Sigma, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[0] != residual.size:
        raise ValueError("J must have shape [len(r), action_dimension]")
    if covariance.ndim == 1:
        covariance = np.diag(covariance)
    if covariance.shape != (residual.size, residual.size):
        raise ValueError("Sigma must match the residual dimension")
    weight = np.linalg.pinv(covariance, rcond=1e-10)
    dimension = jacobian.shape[1]
    regularizer = lambda_u * np.eye(dimension)
    if graph_laplacian is not None:
        graph = np.asarray(graph_laplacian, dtype=np.float64)
        if graph.shape != (dimension, dimension):
            raise ValueError("graph_laplacian must match action dimension")
        regularizer += lambda_graph * graph
    if action_cost is not None:
        cost = np.asarray(action_cost, dtype=np.float64)
        if cost.shape != (dimension, dimension):
            raise ValueError("action_cost must match action dimension")
        regularizer += cost
    normal_geometry = jacobian.T @ weight @ jacobian
    rhs = jacobian.T @ weight @ residual
    normal_regularized = normal_geometry + regularizer
    geometry_action = np.linalg.pinv(normal_geometry, rcond=1e-10) @ rhs
    regularized_action = np.linalg.pinv(normal_regularized, rcond=1e-10) @ rhs

    lower = upper = None
    if bounds is not None:
        lower = np.broadcast_to(np.asarray(bounds[0], dtype=np.float64), (dimension,))
        upper = np.broadcast_to(np.asarray(bounds[1], dtype=np.float64), (dimension,))
        if np.any(lower > upper):
            raise ValueError("control lower bounds exceed upper bounds")
    A = None if inequality_matrix is None else np.asarray(inequality_matrix, dtype=np.float64)
    b = None if inequality_upper is None else np.asarray(inequality_upper, dtype=np.float64).reshape(-1)
    if (A is None) != (b is None) or (A is not None and A.shape != (b.size, dimension)):
        raise ValueError("linear inequalities require A[action] <= b with aligned shapes")

    constrained = bounds is not None or A is not None
    action = regularized_action
    solver_status = "closed_form"
    solver_success = True
    if solver_backend not in ("auto", "box_coordinate"):
        raise ValueError("solver_backend must be auto or box_coordinate")
    if constrained and solver_backend == "box_coordinate":
        if A is not None:
            raise ValueError(
                "box_coordinate backend does not support linear inequalities")
        action = np.clip(regularized_action, lower, upper)
        solver_success = False
        for _ in range(BOX_COORDINATE_MAX_ITERATIONS):
            previous = action.copy()
            for coordinate in range(dimension):
                diagonal = normal_regularized[coordinate, coordinate]
                if diagonal <= 0.0:
                    continue
                cross = (normal_regularized[coordinate] @ action
                         - diagonal * action[coordinate])
                action[coordinate] = np.clip(
                    (rhs[coordinate] - cross) / diagonal,
                    lower[coordinate], upper[coordinate])
            if np.max(np.abs(action - previous), initial=0.0) \
                    <= BOX_COORDINATE_TOLERANCE:
                solver_success = True
                break
        solver_status = ("box_coordinate_converged" if solver_success
                         else "box_coordinate_iteration_limit")
    elif constrained:
        try:
            from scipy.optimize import minimize

            def objective(u: np.ndarray) -> float:
                error = residual - jacobian @ u
                return 0.5 * float(error @ weight @ error + u @ regularizer @ u)

            def gradient(u: np.ndarray) -> np.ndarray:
                return normal_regularized @ u - rhs

            constraints = []
            if A is not None:
                constraints.append({"type": "ineq", "fun": lambda u: b - A @ u,
                                    "jac": lambda u: -A})
            scipy_bounds = None if lower is None else tuple(zip(lower, upper))
            fitted = minimize(objective, regularized_action, jac=gradient,
                              method="SLSQP", bounds=scipy_bounds,
                              constraints=constraints,
                              options={"ftol": 1e-12, "maxiter": 1000})
            action = np.asarray(fitted.x, dtype=np.float64)
            solver_status = str(fitted.message)
            solver_success = bool(fitted.success)
        except ImportError:
            if A is not None:
                raise RuntimeError("scipy is required for linearly constrained V10 control")
            action = np.clip(regularized_action, lower, upper)
            solver_status = "box_clip_fallback"

    geometry_parallel = jacobian @ geometry_action
    regularized_parallel = jacobian @ action
    geometry_perpendicular = residual - geometry_parallel
    regularized_perpendicular = residual - regularized_parallel
    baseline_objective = float(residual @ weight @ residual)
    post_residual_energy = float(regularized_perpendicular @ weight @ regularized_perpendicular)
    action_penalty = float(action @ regularizer @ action)
    post_objective = post_residual_energy + action_penalty
    max_inequality_violation = 0.0 if A is None else float(np.maximum(A @ action - b, 0.0).max(initial=0.0))
    max_bound_violation = 0.0
    if lower is not None:
        max_bound_violation = float(max(
            np.maximum(lower - action, 0.0).max(initial=0.0),
            np.maximum(action - upper, 0.0).max(initial=0.0)))
    return {
        "u_star": action,
        "geometry_action": geometry_action,
        "regularized_unconstrained_action": regularized_action,
        "r_parallel": regularized_parallel,
        "r_perp": regularized_perpendicular,
        "geometry_r_parallel": geometry_parallel,
        "geometry_r_perp": geometry_perpendicular,
        "W": weight,
        "geometry_normal_matrix": normal_geometry,
        "normal_matrix": normal_regularized,
        "regularizer": regularizer,
        "baseline_objective": baseline_objective,
        "post_residual_energy": post_residual_energy,
        "action_penalty": action_penalty,
        "post_objective": post_objective,
        "objective_reduction": baseline_objective - post_objective,
        "solver_status": solver_status,
        "solver_success": solver_success,
        "constrained": constrained,
        "stationarity_norm_unconstrained": float(np.linalg.norm(
            normal_regularized @ regularized_action - rhs)),
        "max_inequality_violation": max_inequality_violation,
        "max_bound_violation": max_bound_violation,
    }
