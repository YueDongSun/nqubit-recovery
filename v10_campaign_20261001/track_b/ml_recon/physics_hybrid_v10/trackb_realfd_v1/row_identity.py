"""Canonical row-content identities shared by Track B admission and runtime."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import canonical_sha256


ROW_CONTENT_FIELDS = (
    "four_cell_counts", "operation_mask", "projector_descriptor", "raw_rf_descriptor",
    "raw_rf_feature_mask", "prepared_command", "prepared_command_mask", "exposure_s",
    "exposure_known_mask", "global_multi",
    "global_multi_mask", "global_multi_scope", "operation_ids", "operation_families",
    "cycle_indices", "system_type", "prepared_command_semantics", "raw_rf_status",
    "sample_id", "group_id",
)


def _value_digest(value: Any) -> str:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(b"TrackBRowValueV1\x00")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(int(item) for item in array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
        return digest.hexdigest()
    if isinstance(value, Mapping):
        return canonical_sha256({str(key): _value_digest(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))})
    if isinstance(value, (tuple, list)):
        return canonical_sha256([_value_digest(item) for item in value])
    if isinstance(value, (bool, int, float, str)) or value is None:
        return canonical_sha256({"type": type(value).__name__, "value": value})
    return canonical_sha256({"type": type(value).__name__, "value": str(value)})


def source_row_content_sha(values: Mapping[str, Any]) -> str:
    """Hash only the registered row fields, including all physical inputs."""
    return canonical_sha256({
        "schema": "TrackBRealStaticRowContentV1",
        "fields": {name: _value_digest(values.get(name)) for name in ROW_CONTENT_FIELDS},
    })


def row_identity_manifest_sha(
    records: Sequence[tuple[str, str, str, str]],
) -> str:
    normalized = tuple(sorted(
        (str(sample_id), str(group_id), str(content_sha), str(role))
        for sample_id, group_id, content_sha, role in records
    ))
    return canonical_sha256({
        "schema": "TrackBRealStaticRowIdentityManifestV1",
        "records": [list(record) for record in normalized],
    })


__all__ = ["ROW_CONTENT_FIELDS", "row_identity_manifest_sha", "source_row_content_sha"]
