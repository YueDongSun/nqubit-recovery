"""Build one fail-closed recoverability bundle on a common covariance support."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ml_recon.physics_hybrid_v10.contracts import sha256_of
from ml_recon.physics_hybrid_v10.track_a_n8.csm_context_builder import build_csm_model_context
from ml_recon.physics_hybrid_v10.csm_covariance import support_whitening
from ml_recon.physics_hybrid_v10.csm_geometry import compute_csm_geometry
from ml_recon.physics_hybrid_v10.csm_recoverability import csm_recoverability

from ..contracts import RecoverabilityBundleV1, TrackAAuthority

if TYPE_CHECKING:
    from ml_recon.physics_hybrid_v10.csm_geometry import CSMGeometryResult


def _array_sha(value: np.ndarray) -> str:
    array = np.asarray(value)
    return sha256_of({
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "real": np.real(array).tolist(),
        "imag": np.imag(array).tolist() if np.iscomplexobj(array) else None,
    })


def _bound_whitener(
    bound_geometry: "CSMGeometryResult",
    logical_jacobian: np.ndarray,
    covariance: np.ndarray,
    sample_id: str,
):
    bound_geometry.validate_bound()
    if sample_id != bound_geometry.forward_sample_id:
        raise ValueError("recoverability sample does not match the bound CSM instance")
    if (not bound_geometry.forward_instance_sha
            or bound_geometry.forward_instance_sha
            != bound_geometry.contract.forward_instance_sha):
        raise ValueError("bound recoverability requires an exact forward-instance identity")
    if not np.allclose(
            np.asarray(bound_geometry.logical_jacobian, dtype=np.float64),
            logical_jacobian,
            atol=0.0,
            rtol=0.0):
        raise ValueError("logical Jacobian does not match the bound CSM forward instance")
    direct = support_whitening(covariance)
    if direct.covariance_sha != bound_geometry.whitener.covariance_sha:
        raise ValueError("covariance does not match the bound CSM forward instance")
    return bound_geometry.whitener


def build_recoverability_bundle(
    *,
    sample_id: str,
    residual: np.ndarray,
    logical_jacobian: np.ndarray,
    covariance: np.ndarray,
    action_mean: np.ndarray | None = None,
    action_covariance: np.ndarray | None = None,
    direction_stability: float = 0.0,
    bound_geometry: "CSMGeometryResult | None" = None,
) -> RecoverabilityBundleV1:
    """Use one support-restricted whitener for context, SVD, and all metrics.

    Raw arrays remain an engineering-only path.  Passing the result of
    ``build_bound_csm_from_forward`` upgrades provenance only after its bound
    contract, forward instance, Jacobian, and covariance identities agree.
    """

    if any(np.iscomplexobj(value) for value in (residual, logical_jacobian, covariance)):
        raise ValueError("recoverability residual, Jacobian, and covariance must be real")
    residual_value = np.asarray(residual, dtype=np.float64).reshape(-1)
    jacobian = np.asarray(logical_jacobian, dtype=np.float64)
    covariance_value = np.asarray(covariance, dtype=np.float64)
    if (not sample_id or not np.all(np.isfinite(residual_value))
            or not np.all(np.isfinite(jacobian))):
        raise ValueError("recoverability sample, residual, and Jacobian must be finite")
    if jacobian.ndim != 2 or jacobian.shape[0] != residual_value.size:
        raise ValueError("residual and logical Jacobian observation dimensions disagree")
    if covariance_value.shape != (residual_value.size, residual_value.size):
        raise ValueError("covariance and residual observation dimensions disagree")
    logical_dim = jacobian.shape[1]
    if logical_dim % 2 or logical_dim < 4:
        raise ValueError("Track A W recoverability requires logical_dim=2N-2")
    n = logical_dim // 2 + 1

    active_whitener = (
        support_whitening(covariance_value)
        if bound_geometry is None
        else _bound_whitener(bound_geometry, jacobian, covariance_value, sample_id)
    )
    geometry = bound_geometry or compute_csm_geometry(
        jacobian, covariance_value, n, whitener=active_whitener)
    whitened_jacobian = geometry.whiten_and_profile(jacobian)
    provenance_fields = {
        "sample_id": sample_id,
        "covariance_sha": active_whitener.covariance_sha,
        "logical_jacobian_sha": _array_sha(jacobian),
        "csm_contract_sha": geometry.contract.sha256(),
        "forward_instance_sha": (
            bound_geometry.forward_instance_sha if bound_geometry is not None else ""),
    }
    context = build_csm_model_context(
        geometry, n, source_fields=provenance_fields,
        require_bound=bound_geometry is not None)

    _, singular, vh = np.linalg.svd(whitened_jacobian, full_matrices=False)
    threshold = geometry.contract.singular_support.effective_threshold(
        float(singular.max(initial=0.0)))
    rank = int(np.sum(singular > threshold))
    supported_singular = np.asarray(singular[:rank], dtype=np.float64)
    basis = (
        np.asarray(vh[:rank].T, dtype=np.float64)
        if rank else np.zeros((logical_dim, 0), dtype=np.float64)
    )

    metrics = csm_recoverability(
        residual_value,
        jacobian,
        covariance_value,
        whitener=active_whitener,
        singular_support=geometry.contract.singular_support,
        nuisance_projection=geometry.nuisance_projection,
    )
    if not bool(metrics.get("recoverability_mask", False)):
        raise ValueError(
            "recoverability solver or constraint validation failed; no bundle was emitted")
    if int(metrics.get("geometry_support_rank", -1)) != rank:
        raise RuntimeError("recoverability metrics and basis used different singular support")

    action = np.zeros(logical_dim, dtype=np.float64) if action_mean is None else np.asarray(
        action_mean, dtype=np.float64).reshape(-1)
    action_cov = (
        np.eye(logical_dim, dtype=np.float64)
        if action_covariance is None else np.asarray(action_covariance, dtype=np.float64)
    )
    if action.size != logical_dim or action_cov.shape != (logical_dim, logical_dim):
        raise ValueError("action posterior does not match the logical tangent dimension")
    action_energy = float(action @ action)
    uncertainty = float(np.trace(action_cov))
    result = RecoverabilityBundleV1(
        sample_id=sample_id,
        csm_context_sha=str(context["context_sha"]),
        recoverable_basis=basis,
        supported_singular_values=supported_singular,
        rank_fraction=rank / logical_dim,
        condition_number=float(
            supported_singular[0] / supported_singular[-1] if rank else 0.0),
        kappa_geometry=float(metrics["kappa_tangent"]),
        epsilon_irrecoverable=float(metrics["epsilon_irr_geom"]),
        action_mean=action,
        action_covariance=action_cov,
        action_snr=float(np.sqrt(action_energy / (uncertainty + 1e-12))),
        direction_stability=float(np.clip(direction_stability, 0.0, 1.0)),
        reason_code=(
            "RECOVERABLE_SUPPORT_PRESENT" if rank else "RECOVERABILITY_ZERO_SUPPORT"),
        same_forward_bound=bound_geometry is not None,
        forward_instance_sha=(
            bound_geometry.forward_instance_sha if bound_geometry is not None else ""),
        logical_jacobian_sha=_array_sha(jacobian),
        covariance_sha=active_whitener.covariance_sha,
        authority=TrackAAuthority(),
    )
    result.validate()
    return result
