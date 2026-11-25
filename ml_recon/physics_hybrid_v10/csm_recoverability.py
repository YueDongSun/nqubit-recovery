"""CSM-v1 recoverability semantics: geometry versus operation."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .control_solver import solve_control
from .csm_covariance import (
    BlockSupportWhitener,
    MultinomialBlockSupportWhitener,
    SupportWhitener,
    support_whitening,
)
from .csm_contract import SingularSupportContract
from .csm_nuisance import CSMNuisanceProjection


def geometry_only_recoverability(
    residual: np.ndarray,
    whitened_jacobian: np.ndarray,
    *,
    epsilon: float = 1e-12,
    singular_threshold: float | None = None,
    singular_support_threshold: float | None = None,
    singular_support: SingularSupportContract | None = None,
) -> Dict[str, object]:
    """Measure tangent alignment without computing or returning an action."""
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    J = np.asarray(whitened_jacobian, dtype=np.float64)
    if J.ndim != 2 or J.shape[0] != r.size:
        raise ValueError("whitened_jacobian must align with residual")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(J)):
        raise ValueError("geometry recoverability inputs must be finite")
    if singular_threshold is not None and singular_support_threshold is not None:
        raise ValueError("singular threshold aliases must not both be supplied")
    if singular_threshold is not None:
        singular_support_threshold = float(singular_threshold)
    if singular_support_threshold is not None and singular_support_threshold <= 0.0:
        raise ValueError("singular-support threshold must be positive")
    if singular_support is not None:
        singular_support.validate()
    rank = 0
    threshold = float(singular_support_threshold or 1e-10)
    if J.shape[1] == 0:
        parallel = np.zeros_like(r)
    else:
        U, singular, _ = np.linalg.svd(J, full_matrices=False)
        if singular_support is not None:
            threshold = singular_support.effective_threshold(
                float(singular.max(initial=0.0)))
        rank = int(np.sum(singular > threshold))
        parallel = U[:, :rank] @ (U[:, :rank].T @ r) if rank else np.zeros_like(r)
    perpendicular = r - parallel
    energy = float(r @ r)
    parallel_energy = float(parallel @ parallel)
    perpendicular_energy = float(perpendicular @ perpendicular)
    return {
        "kappa_tangent": float(np.clip(parallel_energy / (energy + epsilon), 0.0, 1.0)),
        "epsilon_irr_geom": float(np.sqrt(max(perpendicular_energy, 0.0)
                                       / (energy + epsilon))),
        "geometry_support_rank": int(rank if J.shape[1] else 0),
        "geometry_singular_support_threshold": threshold,
        "geometry_residual_energy": energy,
        "geometry_only": True,
    }


def geometry_recoverability_from_whitened_svd(
    residual: np.ndarray,
    whitened_jacobian: np.ndarray,
    *,
    epsilon: float = 1e-12,
    singular_support_threshold: float | None = None,
    singular_support: SingularSupportContract | None = None,
) -> Dict[str, object]:
    """Use one explicit whitened-SVD support for geometry reference metrics."""
    result = geometry_only_recoverability(
        residual,
        whitened_jacobian,
        epsilon=epsilon,
        singular_support_threshold=singular_support_threshold,
        singular_support=singular_support,
    )
    result["geometry_metric_kind"] = "common_support_whitened_svd"
    return result

def operational_recoverability(
    residual: np.ndarray,
    jacobian: np.ndarray,
    covariance: np.ndarray,
    *,
    solved: Dict[str, object] | None = None,
    epsilon: float = 1e-12,
    **solver_kwargs: object,
) -> Dict[str, object]:
    """Report constrained/regularized recovery separately from geometry."""
    result = solved or solve_control(residual, jacobian, covariance, **solver_kwargs)
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    W = np.asarray(result["W"], dtype=np.float64)
    action = np.asarray(result["u_star"], dtype=np.float64)
    regularizer = np.asarray(result["regularizer"], dtype=np.float64)
    post_residual = r - np.asarray(jacobian, dtype=np.float64) @ action
    E0 = float(r @ W @ r)
    Epost = float(post_residual @ W @ post_residual)
    Eact = float(action @ regularizer @ action)
    solver_success = bool(result.get("solver_success", False))
    constraint_valid = bool(
        np.isfinite(result.get("max_inequality_violation", 0.0))
        and np.isfinite(result.get("max_bound_violation", 0.0))
        and float(result.get("max_inequality_violation", 0.0)) <= 1e-8
        and float(result.get("max_bound_violation", 0.0)) <= 1e-8
    )
    kappa = float(np.clip((E0 - Epost - Eact) / (E0 + epsilon), 0.0, 1.0))
    return {
        "E0": E0,
        "E_post": Epost,
        "E_post_residual": Epost,
        "E_act": Eact,
        "kappa_operational_reg": kappa,
        "recoverability_mask": bool(np.isfinite(E0) and np.isfinite(Epost)
                                     and np.isfinite(Eact)
                                     and solver_success and constraint_valid),
        "solver_status": result.get("solver_status", "unknown"),
        "solver_success": solver_success,
        "constraint_valid": constraint_valid,
        "geometry_only": False,
    }


def csm_recoverability(
    residual: np.ndarray,
    logical_jacobian: np.ndarray,
    covariance: np.ndarray,
    *,
    whitener: (
        SupportWhitener | BlockSupportWhitener | MultinomialBlockSupportWhitener
        | None
    ) = None,
    singular_support_threshold: float | None = None,
    singular_support: SingularSupportContract | None = None,
    nuisance_projection: CSMNuisanceProjection | None = None,
    **solver_kwargs: object,
) -> Dict[str, object]:
    """Return both v2 semantics while keeping action vectors out of the CSM record."""
    if "solved" in solver_kwargs:
        raise ValueError("CSM common-support evaluation cannot accept an unbound precomputed solve")
    if any(np.iscomplexobj(value) for value in (residual, logical_jacobian, covariance)):
        raise ValueError("CSM residual, Jacobian, and covariance must be real")
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    J = np.asarray(logical_jacobian, dtype=np.float64)
    covariance_value = np.asarray(covariance, dtype=np.float64)
    if (J.ndim != 2 or J.shape[0] != r.size
            or covariance_value.shape not in ((r.size,), (r.size, r.size))
            or not all(np.all(np.isfinite(value)) for value in (r, J, covariance_value))):
        raise ValueError("CSM residual, Jacobian, and covariance must have finite aligned dimensions")
    if nuisance_projection is not None:
        if (singular_support is None
                or nuisance_projection.singular_support_sha != singular_support.sha256()):
            raise ValueError("nuisance projection and control singular-support contracts disagree")
    active_whitener = whitener or support_whitening(covariance_value)
    whitened_residual = active_whitener.apply(r)
    whitened_jacobian = active_whitener.apply(J)
    if nuisance_projection is not None:
        whitened_residual = nuisance_projection.project(whitened_residual)
        whitened_jacobian = nuisance_projection.project(whitened_jacobian)
    geometry = geometry_recoverability_from_whitened_svd(
        whitened_residual, whitened_jacobian,
        singular_support_threshold=singular_support_threshold,
        singular_support=singular_support)
    operational = operational_recoverability(
        whitened_residual, whitened_jacobian,
        np.eye(whitened_residual.size), **solver_kwargs)
    return {**geometry, **operational,
            "support_rank": active_whitener.rank,
            "covariance_sha": active_whitener.covariance_sha,
            "operational_metric_kind": "COMMON_PROFILED_WHITENED_SUPPORT",
            "nuisance_projection_sha": (
                nuisance_projection.projection_sha
                if nuisance_projection is not None else ""),
            "nuisance_rank": nuisance_projection.rank if nuisance_projection is not None else 0}


__all__ = [
    "csm_recoverability", "geometry_only_recoverability",
    "geometry_recoverability_from_whitened_svd",
    "operational_recoverability",
]
