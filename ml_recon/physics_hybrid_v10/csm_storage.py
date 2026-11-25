"""Low-disk persistence guards for CSM and partition sidecars."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np


FORBIDDEN_PERSISTENT_CSM_FIELDS = frozenset({
    "same_block_probability", "dense_same_block", "gram", "dense_gram",
    "same_block_probability_defined", "edge_coupling", "full_projector",
    "svd_u", "svd_vh", "full_svd",
})
PERSISTENT_EDGE_MULTIPLIER = 4


def default_max_persistent_edges(N: int) -> int:
    if N < 3 or N > 256:
        raise ValueError("edge budget requires 3 <= N <= 256")
    return PERSISTENT_EDGE_MULTIPLIER * int(N)


def estimate_persistent_bytes(payload: object) -> int:
    """Estimate UTF-8 JSON bytes for the serialized low-disk payload."""
    def normalize(value: object) -> object:
        if isinstance(value, np.ndarray):
            return normalize(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {str(key): normalize(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(child) for child in value]
        return value

    encoded = json.dumps(
        normalize(payload), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")
    return len(encoded)

def _looks_like_dense_N_square(value: object, N: int) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim >= 2 and value.shape[0] == N and value.shape[1] == N
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return len(value) == N and len(value[0]) == N
    if isinstance(value, Mapping) and "shape" in value:
        shape = tuple(int(item) for item in value["shape"])
        return len(shape) >= 2 and shape[0] == N and shape[1] == N
    return False


def _leading_rows(value: object) -> int | None:
    if isinstance(value, np.ndarray) and value.ndim >= 1:
        return int(value.shape[0])
    if isinstance(value, (list, tuple)) and value:
        return len(value)
    if isinstance(value, Mapping) and "shape" in value:
        shape = tuple(int(item) for item in value["shape"])
        return int(shape[0]) if shape else None
    return None


def _walk_payload(value: object, prefix: str = "") -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = []
    if isinstance(value, Mapping):
        for name, child in value.items():
            path = f"{prefix}.{name}" if prefix else str(name)
            rows.append((path, child))
            rows.extend(_walk_payload(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            if isinstance(child, Mapping):
                path = f"{prefix}[{index}]"
                rows.extend(_walk_payload(child, path))
    return rows


def validate_low_disk_payload(
    payload: Mapping[str, Any],
    N: int,
    *,
    persistent: bool = True,
    max_persistent_edges: int | None = None,
) -> dict[str, object]:
    """Reject dense N-squared CSM/partition tensors in train persistence."""
    if N < 3 or N > 256:
        raise ValueError("low-disk validation requires 3 <= N <= 256")
    edge_limit = (default_max_persistent_edges(N)
                  if max_persistent_edges is None else int(max_persistent_edges))
    if edge_limit <= 0:
        raise ValueError("max_persistent_edges must be positive")
    forbidden: list[str] = []
    oversized_edges: list[str] = []
    for path, value in _walk_payload(payload):
        lower = path.lower()
        leaf = lower.rsplit(".", 1)[-1]
        if persistent and leaf in FORBIDDEN_PERSISTENT_CSM_FIELDS:
            forbidden.append(path)
        if persistent and _looks_like_dense_N_square(value, N) and (
                "partition" in lower or "csm" in lower or
                leaf in {"gram", "coupling", "edge_features", "edge_coupling"}):
            forbidden.append(path)
        if persistent and "edge" in lower:
            edge_count = _leading_rows(value)
            if edge_count is not None and edge_count > edge_limit:
                oversized_edges.append(f"{path} ({edge_count}>{edge_limit})")
    if forbidden or oversized_edges:
        raise ValueError(
            "large-rung CSM/partition persistence violates low-disk policy: "
            + ", ".join(sorted(set(forbidden + oversized_edges))))
    return {
        "passed": True,
        "N": int(N),
        "persistent": bool(persistent),
        "forbidden_dense_fields": (),
        "max_persistent_edges": edge_limit,
        "edge_budget_checked_recursively": True,
        "estimated_persistent_bytes": estimate_persistent_bytes(payload),
        "allowed_patterns": (
            "candidate_topk_block_ids[K,N]", "candidate_weights[K]",
            "node_features[N,d]", "singular_values[r]", "sparse_edges[E,d]",
        ),
    }


def assert_low_disk_safe(
    payload: Mapping[str, Any], N: int, *, max_persistent_edges: int | None = None,
) -> None:
    validate_low_disk_payload(
        payload, N, persistent=True, max_persistent_edges=max_persistent_edges)


__all__ = [
    "FORBIDDEN_PERSISTENT_CSM_FIELDS", "PERSISTENT_EDGE_MULTIPLIER",
    "assert_low_disk_safe", "default_max_persistent_edges",
    "estimate_persistent_bytes", "validate_low_disk_payload",
]
