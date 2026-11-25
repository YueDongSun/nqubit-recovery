"""Typed, target-independent mode-local CSM context for the V10 sidecar."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ml_recon.physics_hybrid_v10.contracts import sha256_of
from ml_recon.physics_hybrid_v10.csm_geometry import CSMGeometryResult, ORIENTATION_RELATIVE_TOLERANCE


CSM_NODE_CONTEXT_DIM = 9
CSM_GLOBAL_CONTEXT_DIM = 4
FORBIDDEN_CSM_CONTEXT_KEYS = frozenset({
    "target", "target_sha", "residual", "residual_sha", "oracle_action",
    "action", "action_sha", "u_star", "P_J_r", "JtSigma_r",
    "kappa_tangent", "epsilon_irr_geom", "kappa_operational",
})
CONTEXT_SOURCE_FIELDS = frozenset({
    "sample_id", "case_sha", "current_index", "forward_instance_sha",
    "covariance_sha", "csm_contract_sha", "logical_jacobian_sha",
})
CONTEXT_FIELDS = frozenset({
    "schema", "csm_node_context", "csm_edge_context", "csm_global_context",
    "csm_valid_mask", "csm_theta_valid_mask", "source_fields", "geometry_only",
    "profiling_mode", "nuisance_projection_sha", "shot_normalization",
    "normalization_shots", "global_spectrum_kind", "context_sha",
    "construction_kind",
})


def _signed_log(value: float) -> float:
    return float(np.sign(value) * np.log1p(abs(float(value))))


def _context_sha(context: Mapping[str, Any]) -> str:
    payload = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in context.items() if key != "context_sha"
    }
    return sha256_of(payload)


def build_csm_model_context(
    geometry: CSMGeometryResult,
    N: int | None = None,
    *,
    shots: float | None = None,
    source_fields: dict[str, Any] | None = None,
    require_bound: bool = False,
) -> dict[str, Any]:
    """Consume formal mode-local geometry, never an ambiguous raw logical Gram.

    Covariance already carries its registered shot convention. No further
    normalization is implicit. Supplying shots explicitly requests per-shot
    features and is recorded in the context identity.
    """
    if not isinstance(geometry, CSMGeometryResult):
        raise TypeError("CSM context requires CSMGeometryResult; raw logical Gram is not accepted")
    if N is not None and int(N) != geometry.N:
        raise ValueError("CSM context N differs from its geometry")
    if geometry.N != geometry.contract.N:
        raise ValueError("CSM context geometry and contract N disagree")
    geometry.contract.validate()
    if require_bound:
        geometry.validate_bound()
    metadata = dict(source_fields or {})
    expected_metadata = {
        "forward_instance_sha": geometry.forward_instance_sha,
        "covariance_sha": geometry.covariance_sha,
        "csm_contract_sha": geometry.contract.sha256(),
        "logical_jacobian_sha": geometry.logical_jacobian_sha,
    }
    if require_bound:
        expected_metadata["sample_id"] = geometry.forward_sample_id
    for name, expected_value in expected_metadata.items():
        if name in metadata and metadata[name] != expected_value:
            raise ValueError(f"CSM source field does not match geometry: {name}")
    metadata = {**expected_metadata, **metadata}
    if shots is not None and (not np.isfinite(shots) or shots <= 0.0):
        raise ValueError("explicit per-shot normalization requires finite positive shots")
    n = geometry.N
    value = np.asarray(geometry.gram, dtype=np.float64)
    if value.shape != (2 * n, 2 * n) or not np.all(np.isfinite(value)):
        raise ValueError("CSM context requires a finite mode-local 2N Gram")
    if not np.allclose(value, value.T, atol=1e-10, rtol=1e-8):
        raise ValueError("mode-local Gram must be symmetric")
    scale = 1.0 if shots is None else float(shots)
    symmetric = 0.5 * (value + value.T) / scale
    node = np.zeros((n, CSM_NODE_CONTEXT_DIM), dtype=np.float32)
    theta_valid = np.zeros(n, dtype=np.bool_)
    traces = np.zeros(n, dtype=np.float64)
    for index in range(n):
        block = symmetric[2 * index:2 * index + 2, 2 * index:2 * index + 2]
        aa, ap, pp = float(block[0, 0]), float(block[0, 1]), float(block[1, 1])
        eigenvalues = np.linalg.eigvalsh(block)
        if eigenvalues[0] < -1e-8 * max(abs(float(np.trace(block))), 1.0):
            raise ValueError("mode-local Gram must be positive semidefinite")
        lambda_minus = max(float(eigenvalues[0]), 0.0)
        lambda_plus = max(float(eigenvalues[1]), lambda_minus)
        traces[index] = max(aa + pp, 0.0)
        trace = max(traces[index], 1e-12)
        anisotropy = (lambda_plus - lambda_minus) / trace
        theta_valid[index] = anisotropy > ORIENTATION_RELATIVE_TOLERANCE
        theta = (0.5 * np.arctan2(2.0 * ap, aa - pp) / np.pi
                 if theta_valid[index] else 0.0)
        rho = ap / max(np.sqrt(max(aa, 0.0) * max(pp, 0.0)), 1e-12)
        node[index] = np.asarray((
            np.log1p(max(aa, 0.0)), _signed_log(ap), np.log1p(max(pp, 0.0)),
            lambda_plus / trace, lambda_minus / trace, np.log(trace),
            anisotropy, np.clip(rho, -1.0, 1.0), theta,
        ), dtype=np.float32)
    edge = np.asarray(geometry.edge_coupling, dtype=np.float64)
    if (not np.all(np.isfinite(edge)) or np.any(edge < -1e-7)
            or np.any(edge > 1.0 + 1e-7)):
        raise ValueError("mode-local edge coupling is outside its registered bounds")
    edge = np.clip(edge, 0.0, 1.0)
    logical_dimension = geometry.contract.logical_dim
    condition = geometry.condition_number_jacobian
    global_context = np.asarray((
        geometry.effective_rank_pr_singular / logical_dimension,
        np.clip(np.log10(max(condition, 1.0)), -8.0, 8.0)
        if np.isfinite(condition) else 8.0,
        np.log(max(float(np.trace(symmetric)), 1e-12) / logical_dimension),
        float(np.std(traces) / max(abs(np.mean(traces)), 1e-12)),
    ), dtype=np.float32)
    result = {
        "schema": "V10CSMModelContextV2",
        "csm_node_context": node,
        "csm_edge_context": edge.astype(np.float32),
        "csm_global_context": global_context,
        "csm_valid_mask": traces > 0.0,
        "csm_theta_valid_mask": theta_valid,
        "source_fields": metadata,
        "construction_kind": ("FORMAL_SAME_FORWARD" if require_bound
                              else "UNBOUND_NUMERICAL"),
        "geometry_only": True,
        "profiling_mode": geometry.profiling_mode,
        "nuisance_projection_sha": (
            geometry.nuisance_projection.projection_sha
            if geometry.nuisance_projection is not None else ""),
        "shot_normalization": ("RAW_OBSERVATION_INFORMATION" if shots is None
                               else "EXPLICIT_PER_SHOT_INFORMATION"),
        "normalization_shots": None if shots is None else float(shots),
        "global_spectrum_kind": "JACOBIAN_SINGULAR_VALUES",
    }
    result["context_sha"] = _context_sha(result)
    validate_csm_model_context(result)
    return result


def validate_csm_model_context(context: dict[str, Any]) -> None:
    """Reject recursive target/action leakage, invalid geometry, and tampering."""
    forbidden = {key.lower() for key in FORBIDDEN_CSM_CONTEXT_KEYS}
    pending: list[tuple[str, object]] = [("", context)]
    leaked = []
    while pending:
        prefix, item = pending.pop()
        if isinstance(item, Mapping):
            for key, child in item.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if str(key).lower() in forbidden:
                    leaked.append(path)
                pending.append((path, child))
        elif isinstance(item, (list, tuple)):
            pending.extend((f"{prefix}[{index}]", child)
                           for index, child in enumerate(item))
    if leaked:
        raise ValueError(f"CSM model context contains forbidden fields: {sorted(leaked)}")
    if set(context) != CONTEXT_FIELDS or context.get("schema") != "V10CSMModelContextV2":
        raise ValueError("CSM model context must follow the closed V2 field schema")
    metadata = context["source_fields"]
    if not isinstance(metadata, Mapping) or set(metadata) - CONTEXT_SOURCE_FIELDS:
        raise ValueError("CSM source_fields contains unsupported metadata")
    required_metadata = {"forward_instance_sha", "covariance_sha", "csm_contract_sha",
                         "logical_jacobian_sha"}
    if context["construction_kind"] == "FORMAL_SAME_FORWARD":
        required_metadata.add("sample_id")
        if not metadata.get("sample_id") or not metadata.get("forward_instance_sha"):
            raise ValueError("bound CSM context requires its sample/forward identities")
    elif context["construction_kind"] != "UNBOUND_NUMERICAL":
        raise ValueError("CSM construction kind is not registered")
    if not required_metadata.issubset(metadata):
        raise ValueError("CSM context is missing geometry identities")
    if any(not isinstance(value, (str, int, float, bool, type(None)))
           or isinstance(value, float) and not np.isfinite(value)
           for value in metadata.values()):
        raise ValueError("CSM source metadata must contain finite scalar identities")
    node = np.asarray(context["csm_node_context"], dtype=np.float64)
    if node.ndim != 2 or node.shape[1] != CSM_NODE_CONTEXT_DIM:
        raise ValueError("CSM node context has the wrong shape")
    n = node.shape[0]
    if not 3 <= n <= 256:
        raise ValueError("CSM context requires 3 <= N <= 256")
    edge = np.asarray(context["csm_edge_context"], dtype=np.float64)
    global_context = np.asarray(context["csm_global_context"], dtype=np.float64)
    if edge.shape != (n, n) or global_context.shape != (CSM_GLOBAL_CONTEXT_DIM,):
        raise ValueError("CSM edge/global context has the wrong shape")
    for name in ("csm_valid_mask", "csm_theta_valid_mask"):
        mask = np.asarray(context[name])
        if mask.shape != (n,) or mask.dtype != np.bool_:
            raise ValueError("CSM validity masks must be boolean node vectors")
    if not all(np.all(np.isfinite(value)) for value in (node, edge, global_context)):
        raise ValueError("CSM context is not finite")
    if (not np.allclose(edge, edge.T, atol=1e-7)
            or not np.allclose(np.diag(edge), 0.0, atol=1e-7)
            or np.any(edge < -1e-7) or np.any(edge > 1.0 + 1e-7)):
        raise ValueError("CSM edge context is not a normalized symmetric coupling")
    if context["geometry_only"] is not True:
        raise ValueError("CSM model context must be marked geometry_only")
    if context["global_spectrum_kind"] != "JACOBIAN_SINGULAR_VALUES":
        raise ValueError("CSM global spectrum convention is not registered")
    if context["profiling_mode"] not in (
            "UNPROFILED_POINT_ESTIMATE", "NUISANCE_PROFILED_POINT_ESTIMATE"):
        raise ValueError("CSM profiling mode is not registered")
    projection_sha = context["nuisance_projection_sha"]
    if context["profiling_mode"] == "NUISANCE_PROFILED_POINT_ESTIMATE":
        if (not isinstance(projection_sha, str) or len(projection_sha) != 64
                or any(character not in "0123456789abcdef" for character in projection_sha)):
            raise ValueError("profiled CSM context requires a nuisance projection SHA")
    elif projection_sha != "":
        raise ValueError("unprofiled CSM context cannot claim a nuisance projection")
    if context["shot_normalization"] == "RAW_OBSERVATION_INFORMATION":
        if context["normalization_shots"] is not None:
            raise ValueError("raw information cannot carry per-shot normalization")
    elif context["shot_normalization"] == "EXPLICIT_PER_SHOT_INFORMATION":
        shots = context["normalization_shots"]
        if not isinstance(shots, (int, float)) or not np.isfinite(shots) or shots <= 0.0:
            raise ValueError("per-shot normalization requires finite positive shots")
    else:
        raise ValueError("CSM shot normalization is not registered")
    if context["context_sha"] != _context_sha(context):
        raise ValueError("CSM context SHA does not match its complete payload")


__all__ = [
    "CSM_GLOBAL_CONTEXT_DIM", "CSM_NODE_CONTEXT_DIM",
    "FORBIDDEN_CSM_CONTEXT_KEYS", "build_csm_model_context",
    "validate_csm_model_context",
]
