"""Fail-closed validators for CSM-v1 invariants, FD audits, and leakage."""

from __future__ import annotations

from typing import Callable, Dict, Mapping

import numpy as np

from .csm_contract import FORBIDDEN_CSM_FIELDS, CSMContractV1
from .csm_geometry import CSMGeometryResult
from .recoverability import same_forward_jacobian


def finite_difference_plateau_audit(
    forward: Callable[[np.ndarray], np.ndarray],
    u0: np.ndarray,
    *,
    h_amplitude: float,
    h_phase: float,
    tolerance: float = 1e-3,
    forward_sha: str = "",
    manifold_sha: str = "",
    epsilon: float = 1e-12,
) -> Dict[str, object]:
    """Audit ``J(2h), J(h), J(h/2)`` with interleaved per-coordinate steps."""
    base = np.asarray(u0, dtype=np.float64).reshape(-1)
    if base.size % 2 != 0 or h_amplitude <= 0.0 or h_phase <= 0.0:
        raise ValueError("interleaved FD inputs require positive h_A/h_phi")
    steps = np.tile(np.asarray((h_amplitude, h_phase), dtype=np.float64), base.size // 2)
    jacobians = {
        "2h": same_forward_jacobian(forward, base, 2.0 * steps),
        "h": same_forward_jacobian(forward, base, steps),
        "h/2": same_forward_jacobian(forward, base, 0.5 * steps),
    }
    def relative(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.linalg.norm(left - right) /
                     (np.linalg.norm(right) + epsilon))
    plateau_h_half = relative(jacobians["h"], jacobians["h/2"])
    plateau_2h = relative(jacobians["2h"], jacobians["h"])
    return {
        "finite_difference_steps": steps,
        "jacobian_2h": jacobians["2h"],
        "jacobian_h": jacobians["h"],
        "jacobian_h_half": jacobians["h/2"],
        "relative_change_h_vs_h_half": plateau_h_half,
        "relative_change_2h_vs_h": plateau_2h,
        "plateau_tolerance": float(tolerance),
        "plateau_pass": bool(plateau_h_half < tolerance and plateau_2h < tolerance),
        "forward_sha": forward_sha,
        "manifold_sha": manifold_sha,
        "bound": bool(len(forward_sha) == 64 and len(manifold_sha) == 64),
    }


def csm_invariant_report(
    geometry: CSMGeometryResult,
    *,
    tolerance: float = 1e-8,
) -> Dict[str, object]:
    """Return numerical CSM-G invariants without hiding measured residuals."""
    eigenvalues = np.linalg.eigvalsh(geometry.gram)
    gauge = max(geometry.gauge_residual_amplitude, geometry.gauge_residual_phase)
    symmetry = float(np.linalg.norm(geometry.gram - geometry.gram.T))
    return {
        "gauge_residual_amplitude": geometry.gauge_residual_amplitude,
        "gauge_residual_phase": geometry.gauge_residual_phase,
        "gauge_invariance_pass": bool(gauge <= tolerance),
        "gram_symmetry_residual": symmetry,
        "gram_symmetry_pass": bool(symmetry <= tolerance),
        "gram_min_eigenvalue": float(eigenvalues.min(initial=0.0)),
        "gram_psd_pass": bool(float(eigenvalues.min(initial=0.0)) >= -tolerance),
        "support_rank": geometry.support_rank,
        "effective_rank_pr_singular": geometry.effective_rank_pr_singular,
        "condition_number_jacobian": (
            float(geometry.condition_number_jacobian)
            if np.isfinite(geometry.condition_number_jacobian) else None),
        "condition_status": (
            "FINITE" if np.isfinite(geometry.condition_number_jacobian)
            else "NO_IDENTIFIABLE_SUPPORT"),
        "all_finite": bool(np.all(np.isfinite(geometry.node_features))
                           and np.all(np.isfinite(geometry.singular_values))),
        "tolerance": tolerance,
    }


def _walk_keys(value: object, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for name, child in value.items():
            path = f"{prefix}.{name}" if prefix else str(name)
            keys.append(path)
            keys.extend(_walk_keys(child, path))
    return keys


def validate_no_action_leakage(artifact: Mapping[str, object]) -> Dict[str, object]:
    """Reject direction-sufficient fields from CSM/structure artifacts."""
    paths = _walk_keys(artifact)
    forbidden = sorted({
        path for path in paths
        if any(path == name or path.endswith("." + name) for name in FORBIDDEN_CSM_FIELDS)
    })
    return {
        "passed": not forbidden,
        "forbidden_fields": forbidden,
        "checked_fields": paths,
    }


def validate_csm_contract(contract: CSMContractV1, *, require_bound: bool = False) -> Dict[str, object]:
    contract.validate(require_bound=require_bound)
    return {
        "passed": True,
        "schema": "V10CSMContractV1",
        "contract_sha": contract.sha256(require_bound=require_bound),
        "forbidden_fields_verified_absent": list(contract.forbidden_fields),
    }


__all__ = [
    "csm_invariant_report", "finite_difference_plateau_audit",
    "validate_csm_contract", "validate_no_action_leakage",
]
