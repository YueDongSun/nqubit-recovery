"""Noise-weighted, residual-direction-free CSM-v1 geometry."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Sequence

import numpy as np

from .contracts import sha256_of
from .control_manifold import (
    INTERLEAVED_AMP_PHASE,
    ControlManifold,
    get_control_manifold,
)
from .csm_contract import CSMContractV1, SingularSupportContract, make_csm_contract
from .csm_covariance import (
    BlockSupportWhitener,
    MultinomialBlockSupportWhitener,
    SupportWhitener,
    support_whitening,
)
from .csm_storage import (
    default_max_persistent_edges,
    estimate_persistent_bytes,
    validate_low_disk_payload,
)
from .csm_nuisance import CSMNuisanceProjection, build_nuisance_projection
from .forward_instance import runtime_value_sha


NODE_FEATURE_NAMES = (
    "G_AA", "G_Aphi", "G_phiphi", "lambda_plus", "lambda_minus",
    "trace", "anisotropy", "rho_Aphi", "theta",
)
EDGE_FEATURE_NAMES = ("G_AA", "G_Aphi", "G_phiA", "G_phiphi", "frobenius", "K")
ORIENTATION_RELATIVE_TOLERANCE = 1e-8
_BOUND_CONSTRUCTION_TOKEN = object()


def _safe_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


@dataclass
class CSMGeometryResult:
    N: int
    manifold_id: str
    coordinate_layout: str
    logical_jacobian: np.ndarray
    mode_jacobian: np.ndarray
    whitened_mode_jacobian: np.ndarray
    gram: np.ndarray
    node_features: np.ndarray
    edge_features: np.ndarray
    edge_coupling: np.ndarray
    singular_values: np.ndarray
    effective_rank_pr_singular: float
    condition_number_jacobian: float
    singular_support_threshold: float
    gauge_residual_amplitude: float
    gauge_residual_phase: float
    support_rank: int
    covariance_sha: str
    contract: CSMContractV1
    whitener: (
        SupportWhitener | BlockSupportWhitener | MultinomialBlockSupportWhitener
    )
    forward_instance_sha: str = ""
    forward_sample_id: str = ""
    finite_difference_audit: Dict[str, object] | None = None
    nuisance_projection: CSMNuisanceProjection | None = None
    profiling_mode: str = "UNPROFILED_POINT_ESTIMATE"
    _bound_construction_token: object | None = field(default=None, repr=False)
    _bound_geometry_sha: str = field(default="", repr=False)

    def whiten_and_profile(self, values: np.ndarray) -> np.ndarray:
        """Apply the same covariance support and nuisance operator to J or r."""
        if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
            raise ValueError("CSM observation-space values must be real and finite")
        whitened = self.whitener.apply(np.asarray(values, dtype=np.float64))
        if self.nuisance_projection is None:
            return whitened
        return self.nuisance_projection.project(whitened)

    def _construction_sha(self) -> str:
        return sha256_of({
            "N": self.N,
            "manifold_id": self.manifold_id,
            "coordinate_layout": self.coordinate_layout,
            "support_rank": self.support_rank,
            "contract_sha": self.contract.sha256(),
            "forward_instance_sha": self.forward_instance_sha,
            "forward_sample_id": self.forward_sample_id,
            "covariance_sha": self.covariance_sha,
            "whitener_runtime_sha": runtime_value_sha(self.whitener),
            "logical_jacobian": self.logical_jacobian.tolist(),
            "mode_jacobian": self.mode_jacobian.tolist(),
            "whitened_mode_jacobian": self.whitened_mode_jacobian.tolist(),
            "gram": self.gram.tolist(),
            "node_features": self.node_features.tolist(),
            "edge_features": self.edge_features.tolist(),
            "edge_coupling": self.edge_coupling.tolist(),
            "singular_values": self.singular_values.tolist(),
            "effective_rank_pr_singular": self.effective_rank_pr_singular,
            "condition_number_jacobian": _safe_float(self.condition_number_jacobian),
            "singular_support_threshold": self.singular_support_threshold,
            "gauge_residual_amplitude": self.gauge_residual_amplitude,
            "gauge_residual_phase": self.gauge_residual_phase,
            "finite_difference_audit": {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in (self.finite_difference_audit or {}).items()
            },
            "profiling_mode": self.profiling_mode,
            "nuisance_projection_sha": (
                self.nuisance_projection.projection_sha
                if self.nuisance_projection is not None else ""),
            "nuisance_projection_basis": (
                self.nuisance_projection.basis.tolist()
                if self.nuisance_projection is not None else None),
        })

    def validate_bound(self) -> None:
        self.contract.validate(require_bound=True)
        if (self._bound_construction_token is not _BOUND_CONSTRUCTION_TOKEN
                or not self._bound_geometry_sha
                or self._bound_geometry_sha != self._construction_sha()):
            raise ValueError("scientific CSM requires unmodified formal same-forward construction")
        if (not self.forward_instance_sha or not self.forward_sample_id
                or self.forward_instance_sha != self.contract.forward_instance_sha
                or not self.finite_difference_audit
                or self.finite_difference_audit.get("plateau_pass") is not True):
            raise ValueError("persistent scientific CSM requires a passed same-forward FD audit")
        if (self.covariance_sha != self.whitener.covariance_sha
                or self.finite_difference_audit.get("forward_sha") != self.contract.forward_sha
                or self.finite_difference_audit.get("manifold_sha")
                != self.contract.manifold_contract_sha):
            raise ValueError("scientific CSM construction dependencies changed")

    @property
    def logical_jacobian_sha(self) -> str:
        value = np.asarray(self.logical_jacobian)
        return sha256_of({"dtype": str(value.dtype), "shape": list(value.shape),
                          "real": value.tolist(), "imag": None})

    def sparse_edges(self, top_k: int | None = None) -> np.ndarray:
        """Return ``[i,j,features...]`` for upper-triangular non-zero edges."""
        pairs = []
        N = self.N
        for i in range(N):
            for j in range(i + 1, N):
                pairs.append((float(self.edge_coupling[i, j]), i, j))
        pairs.sort(key=lambda row: (-row[0], row[1], row[2]))
        if top_k is not None:
            if top_k <= 0:
                return np.zeros((0, 2 + len(EDGE_FEATURE_NAMES)), dtype=np.float64)
            pairs = pairs[: int(top_k)]
        return np.asarray([
            [float(i), float(j), *self.edge_features[i, j].tolist()]
            for _, i, j in pairs
        ], dtype=np.float64)

    def runtime_dict(self) -> Dict[str, object]:
        """Return the complete in-memory diagnostic, including dense matrices."""
        return {
            "schema": "V10CSMRuntimeGeometryV1",
            "N": self.N,
            "control_manifold_id": self.manifold_id,
            "coordinate_layout": self.coordinate_layout,
            "logical_jacobian": self.logical_jacobian,
            "mode_jacobian": self.mode_jacobian,
            "whitened_mode_jacobian": self.whitened_mode_jacobian,
            "gram": self.gram,
            "node_features": self.node_features,
            "edge_features": self.edge_features,
            "edge_coupling": self.edge_coupling,
            "singular_values": self.singular_values,
            "effective_rank_pr_singular": self.effective_rank_pr_singular,
            "condition_number_jacobian": _safe_float(self.condition_number_jacobian),
            "condition_status": (
                "FINITE" if np.isfinite(self.condition_number_jacobian)
                else "NO_IDENTIFIABLE_SUPPORT"),
            "singular_support_threshold": self.singular_support_threshold,
            "gauge_residual_amplitude": self.gauge_residual_amplitude,
            "gauge_residual_phase": self.gauge_residual_phase,
            "support_rank": self.support_rank,
            "covariance_sha": self.covariance_sha,
            "forward_instance_sha": self.forward_instance_sha,
            "forward_sample_id": self.forward_sample_id,
            "finite_difference_audit": self.finite_difference_audit,
            "profiling_mode": self.profiling_mode,
            "nuisance_projection": (
                self.nuisance_projection.to_dict()
                if self.nuisance_projection is not None else None),
            "contract": self.contract.to_dict(),
        }

    def to_dict(
        self, *, top_k_edges: int | None = None, require_bound: bool = True,
    ) -> Dict[str, object]:
        """Return the low-disk persistent representation.

        Dense Jacobians, Gram blocks, projectors, and full SVD matrices are
        intentionally omitted.  ``top_k_edges=None`` means the registered
        O(N) engineering budget, never "persist every edge".
        """
        if require_bound:
            self.validate_bound()
        edge_limit = (default_max_persistent_edges(self.N)
                      if top_k_edges is None else int(top_k_edges))
        sparse = self.sparse_edges(edge_limit)
        value: Dict[str, object] = {
            "schema": "V10CSMContractV1",
            "N": self.N,
            "control_manifold_id": self.manifold_id,
            "logical_dim": self.contract.logical_dim,
            "mode_local_dim": self.contract.mode_local_dim,
            "coordinate_layout": self.coordinate_layout,
            "geometry": {
                "node_features": self.node_features.astype(np.float32).tolist(),
                "edge_storage": "sparse_pairs",
                "edge_features": sparse.astype(np.float32).tolist(),
                "singular_values": self.singular_values.astype(np.float32).tolist(),
                "effective_rank_pr_singular": self.effective_rank_pr_singular,
                "condition_number_jacobian": _safe_float(self.condition_number_jacobian),
                "condition_status": (
                    "FINITE" if np.isfinite(self.condition_number_jacobian)
                    else "NO_IDENTIFIABLE_SUPPORT"),
                "theta_valid_mask": (
                    self.node_features[:, 3] - self.node_features[:, 4]
                    > ORIENTATION_RELATIVE_TOLERANCE
                    * np.maximum(self.node_features[:, 5], 1e-12)).tolist(),
                "singular_support_threshold": self.singular_support_threshold,
                "singular_support_sha": self.contract.singular_support.sha256(),
            },
            "profiling_mode": self.profiling_mode,
            "nuisance_projection": (
                self.nuisance_projection.to_dict()
                if self.nuisance_projection is not None else None),
            "uncertainty_kind": "POINT_ESTIMATE_NOT_POSTERIOR",
            "formal_construction_sha": self._bound_geometry_sha,
            "covariance": {
                "kind": self.contract.covariance_kind,
                "geometry_source": self.contract.geometry_source,
                "support_rank": self.support_rank,
                "support_threshold_sha": sha256_of({
                    "absolute": self.whitener.absolute_threshold,
                    "relative": self.whitener.relative_threshold,
                    "epsilon": self.whitener.epsilon,
                }),
                "support_semantics": self.whitener.support_semantics,
                "whitening_support_kind": self.contract.whitening_support_kind,
                "feedback_observation_contract_sha": (
                    self.contract.feedback_observation_contract_sha),
                "observation_contract_kind": (
                    self.contract.observation_contract_kind),
                "covariance_instance_sha": self.covariance_sha,
                "statistic_transform_sha": self.contract.statistic_transform_sha,
                "csm_contract_sha": self.contract.sha256(),
            },
            "materialization_policy": self.contract.materialization_policy,
            "forbidden_fields_verified_absent": list(self.contract.forbidden_fields),
            "forward_sha": self.contract.forward_sha,
            "jacobian_contract_sha": self.contract.jacobian_contract_sha,
            "manifold_contract_sha": self.contract.manifold_contract_sha,
            "forward_instance_sha": self.forward_instance_sha,
            "forward_sample_id": self.forward_sample_id,
        }
        value["storage_audit"] = {
            "persistent_edge_count": int(sparse.shape[0]),
            "max_persistent_edges": edge_limit,
            "estimated_persistent_bytes": 0,
        }
        for _ in range(3):
            value["storage_audit"]["estimated_persistent_bytes"] = estimate_persistent_bytes(value)
        validate_low_disk_payload(
            value, self.N, persistent=True, max_persistent_edges=edge_limit)
        return value


def _mode_blocks(gram: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node = np.zeros((N, len(NODE_FEATURE_NAMES)), dtype=np.float64)
    edges = np.zeros((N, N, len(EDGE_FEATURE_NAMES)), dtype=np.float64)
    coupling = np.zeros((N, N), dtype=np.float64)
    eps = 1e-12
    for i in range(N):
        block = 0.5 * (gram[2 * i:2 * i + 2, 2 * i:2 * i + 2]
                       + gram[2 * i:2 * i + 2, 2 * i:2 * i + 2].T)
        values = np.maximum(np.linalg.eigvalsh(block)[::-1], 0.0)
        trace = float(np.trace(block))
        node[i] = np.asarray((
            block[0, 0], block[0, 1], block[1, 1], values[0], values[1],
            trace, (values[0] - values[1]) / (trace + eps),
            block[0, 1] / np.sqrt(max(block[0, 0] * block[1, 1] + eps, eps)),
            (0.5 * np.arctan2(2.0 * block[0, 1], block[0, 0] - block[1, 1])
             if values[0] - values[1]
             > ORIENTATION_RELATIVE_TOLERANCE * max(trace, eps) else 0.0),
        ))
    for i in range(N):
        for j in range(i + 1, N):
            block = gram[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            frobenius = float(np.linalg.norm(block, ord="fro"))
            left_norm = float(np.linalg.norm(
                gram[2 * i:2 * i + 2, 2 * i:2 * i + 2], ord="fro"))
            right_norm = float(np.linalg.norm(
                gram[2 * j:2 * j + 2, 2 * j:2 * j + 2], ord="fro"))
            normalized = frobenius / np.sqrt(
                (left_norm + 1e-12) * (right_norm + 1e-12))
            row = np.asarray((block[0, 0], block[0, 1], block[1, 0],
                              block[1, 1], frobenius, normalized))
            edges[i, j] = row
            edges[j, i] = np.asarray((block[0, 0], block[1, 0], block[0, 1],
                                      block[1, 1], frobenius, normalized))
            coupling[i, j] = coupling[j, i] = normalized
    return node, edges, coupling


def compute_csm_geometry(
    logical_jacobian: np.ndarray,
    covariance: np.ndarray | Sequence[np.ndarray] | str,
    N: int,
    *,
    manifold: ControlManifold | None = None,
    contract: CSMContractV1 | None = None,
    whitener: (
        SupportWhitener | BlockSupportWhitener | MultinomialBlockSupportWhitener
        | None
    ) = None,
    absolute_threshold: float | None = None,
    relative_threshold: float | None = None,
    singular_threshold: float | None = None,
    epsilon: float | None = None,
    forward_instance_sha: str = "",
    finite_difference_audit: Dict[str, object] | None = None,
    require_bound: bool = False,
    nuisance_jacobian: np.ndarray | None = None,
) -> CSMGeometryResult:
    """Compute CSM-G from same-forward ``J_q`` and observation covariance."""
    if require_bound:
        raise ValueError("use build_bound_csm_from_forward for scientific construction")
    if np.iscomplexobj(logical_jacobian) or np.iscomplexobj(nuisance_jacobian):
        raise ValueError("CSM control and nuisance Jacobians must be real")
    selected = manifold or get_control_manifold(N=N)
    if selected.N != N:
        raise ValueError("manifold N does not match geometry N")
    J_q = np.asarray(logical_jacobian, dtype=np.float64)
    if J_q.ndim != 2 or J_q.shape[1] != selected.logical_dim:
        raise ValueError("logical_jacobian must have shape [observations, logical_dim]")
    if not np.all(np.isfinite(J_q)):
        raise ValueError("CSM Jacobian must be finite")
    active_contract = contract or make_csm_contract(
        N, manifold_contract_sha=selected.sha256())
    active_contract.validate(require_bound=require_bound)
    if active_contract.N != N or active_contract.control_manifold_id != selected.manifold_id:
        raise ValueError("CSM contract/manifold mismatch")
    abs_threshold = (active_contract.whitening.absolute
                     if absolute_threshold is None else absolute_threshold)
    rel_threshold = (active_contract.whitening.relative
                     if relative_threshold is None else relative_threshold)
    active_epsilon = active_contract.whitening.epsilon if epsilon is None else epsilon
    effective_whitening = replace(active_contract.whitening,
        absolute=abs_threshold, relative=rel_threshold, epsilon=active_epsilon)
    effective_whitening.validate()
    if effective_whitening != active_contract.whitening:
        active_contract = replace(active_contract, whitening=effective_whitening)
    if singular_threshold is not None:
        if not np.isfinite(singular_threshold) or singular_threshold <= 0.0:
            raise ValueError("singular threshold must be finite and positive")
        active_contract = replace(active_contract, singular_support=SingularSupportContract(
            absolute=float(singular_threshold), relative=0.0))
    active_whitener = whitener or support_whitening(
        covariance, absolute_threshold=abs_threshold,
        relative_threshold=rel_threshold, epsilon=active_epsilon)
    if active_whitener.support_semantics != active_contract.whitening_support_kind:
        raise ValueError(
            "supplied whitener support semantics do not match the CSM contract")
    if (active_whitener.absolute_threshold != abs_threshold
            or active_whitener.relative_threshold != rel_threshold
            or active_whitener.epsilon != active_epsilon):
        raise ValueError("supplied whitener thresholds do not match the CSM contract")
    T = selected.mode_to_logical_lift()
    J_mode = J_q @ T
    whitened = active_whitener.whiten_jacobian(J_mode)
    nuisance_projection = None
    if nuisance_jacobian is not None:
        nuisance = np.asarray(nuisance_jacobian, dtype=np.float64)
        if (nuisance.ndim != 2 or nuisance.shape[0] != J_q.shape[0]
                or not np.all(np.isfinite(nuisance))):
            raise ValueError("nuisance Jacobian must have the same finite observation rows")
        nuisance_projection = build_nuisance_projection(
            active_whitener.whiten_jacobian(nuisance),
            singular_support=active_contract.singular_support)
        whitened = nuisance_projection.project(whitened)
    gram = 0.5 * (whitened.T @ whitened + (whitened.T @ whitened).T)
    if not np.all(np.isfinite(gram)):
        raise ValueError("CSM information geometry is not finite")
    node, edges, coupling = _mode_blocks(gram, N)
    singular_values = np.linalg.svd(whitened, compute_uv=False)
    singular_values = np.asarray(singular_values, dtype=np.float64)
    support_threshold = active_contract.singular_support.effective_threshold(
        float(singular_values.max(initial=0.0)))
    positive = singular_values[singular_values > support_threshold]
    sum_s = float(positive.sum())
    sum_sq = float(np.square(positive).sum())
    rank_pr = (sum_s * sum_s) / (sum_sq + active_epsilon) if sum_sq > 0.0 else 0.0
    condition = (float(positive.max() / positive.min())
                 if positive.size else float("inf"))
    gauges = selected.gauge_null_basis()
    gauge_amp = float(np.linalg.norm(J_mode @ gauges[:, 0]))
    gauge_phase = float(np.linalg.norm(J_mode @ gauges[:, 1]))
    return CSMGeometryResult(
        N=N,
        manifold_id=selected.manifold_id,
        coordinate_layout=INTERLEAVED_AMP_PHASE,
        logical_jacobian=J_q,
        mode_jacobian=J_mode,
        whitened_mode_jacobian=whitened,
        gram=gram,
        node_features=node,
        edge_features=edges,
        edge_coupling=coupling,
        singular_values=singular_values,
        effective_rank_pr_singular=float(rank_pr),
        condition_number_jacobian=condition,
        singular_support_threshold=support_threshold,
        gauge_residual_amplitude=gauge_amp,
        gauge_residual_phase=gauge_phase,
        support_rank=active_whitener.rank,
        covariance_sha=active_whitener.covariance_sha,
        contract=active_contract,
        whitener=active_whitener,
        forward_instance_sha=forward_instance_sha,
        finite_difference_audit=finite_difference_audit,
        nuisance_projection=nuisance_projection,
        profiling_mode=("NUISANCE_PROFILED_POINT_ESTIMATE"
                        if nuisance_jacobian is not None else "UNPROFILED_POINT_ESTIMATE"),
    )


def compute_profiled_csm_geometry(
    logical_control_jacobian: np.ndarray,
    nuisance_jacobian: np.ndarray,
    covariance: np.ndarray | Sequence[np.ndarray] | str,
    n: int,
    **kwargs,
) -> CSMGeometryResult:
    """Target-free numerical entry point, not a posterior or bound artifact."""
    return compute_csm_geometry(
        logical_control_jacobian, covariance, n,
        nuisance_jacobian=nuisance_jacobian, **kwargs)


def _seal_bound_csm_geometry(geometry: CSMGeometryResult) -> None:
    """Internal final step of the verified same-forward builder."""
    geometry._bound_construction_token = _BOUND_CONSTRUCTION_TOKEN
    geometry._bound_geometry_sha = geometry._construction_sha()
    geometry.validate_bound()


def csm_geometry(*args, **kwargs) -> CSMGeometryResult:
    """Short compatibility alias for :func:`compute_csm_geometry`."""
    return compute_csm_geometry(*args, **kwargs)


def build_bound_csm_from_forward(*args, **kwargs) -> CSMGeometryResult:
    """Lazy compatibility export for the bound scientific construction path."""
    from .build_bound_csm import build_bound_csm_from_forward as builder
    return builder(*args, **kwargs)


__all__ = [
    "CSMGeometryResult", "EDGE_FEATURE_NAMES", "NODE_FEATURE_NAMES",
    "build_bound_csm_from_forward", "compute_csm_geometry",
    "compute_profiled_csm_geometry", "csm_geometry",
]
