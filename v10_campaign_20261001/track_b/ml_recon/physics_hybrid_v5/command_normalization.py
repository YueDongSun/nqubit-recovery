"""Train-table-only normalization for compact static command features."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .command_contract import PhysicsHybridCommandTableV1
from .command_features import (
    FREQUENCY_CENTER_MHZ,
    FREQUENCY_SCALE_MHZ,
    command_tone_features,
)


def fit_command_table_normalization(
    train_tables: Sequence[PhysicsHybridCommandTableV1],
) -> Dict[str, Any]:
    if not train_tables:
        raise ValueError("at least one train command table is required")
    feature_rows = []
    schema = None
    table_hashes = []
    for table in train_tables:
        features, current_schema = command_tone_features(table)
        if schema is None:
            schema = current_schema
        elif current_schema.ordered_names != schema.ordered_names:
            raise ValueError("train command tables have different feature schemas")
        feature_rows.append(features.reshape(-1, features.shape[-1]))
        table_hashes.append(table.table_hash)
    matrix = np.concatenate(feature_rows, axis=0)
    records = []
    for index, name in enumerate(schema.ordered_names):
        if name == "absolute_frequency_mhz" or name.endswith(".absolute_frequency_mhz"):
            records.append({
                "name": name, "kind": "fixed_physical",
                "center": FREQUENCY_CENTER_MHZ, "scale": FREQUENCY_SCALE_MHZ,
            })
        elif name.endswith("phase_sin") or name.endswith("phase_cos") or name.endswith(".missing"):
            records.append({"name": name, "kind": "identity_preserve_geometry_or_mask", "center": 0.0, "scale": 1.0})
        else:
            mean = float(np.mean(matrix[:, index]))
            std = float(max(np.std(matrix[:, index]), 1e-8))
            records.append({"name": name, "kind": "train_table_standard", "center": mean, "scale": std})
    payload = {
        "schema": "PhysicsHybridCommandNormalizationV1",
        "schema_version": 1,
        "fit_population": "train_command_tables_only",
        "table_hashes": sorted(table_hashes),
        "per_operation_group_minmax": False,
        "features": records,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def validate_command_normalization_contract(
    contract: Mapping[str, Any],
    feature_order: Sequence[str],
    *,
    expected_normalization_sha256: str | None = None,
) -> str:
    """Validate the train-table contract and return its canonical SHA-256."""

    payload = dict(contract)
    recorded_sha256 = str(payload.pop("sha256", "") or "")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    computed_sha256 = hashlib.sha256(canonical).hexdigest()
    if not recorded_sha256 or recorded_sha256 != computed_sha256:
        raise ValueError("command normalization contract SHA-256 mismatch")
    if expected_normalization_sha256 is not None and recorded_sha256 != str(expected_normalization_sha256):
        raise ValueError("command normalization does not match the checkpoint metadata")
    if payload.get("schema") != "PhysicsHybridCommandNormalizationV1" or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported command normalization contract")
    if payload.get("fit_population") != "train_command_tables_only":
        raise ValueError("command normalization must be fit on train command tables only")
    if bool(payload.get("per_operation_group_minmax", True)):
        raise ValueError("per-operation/group min-max command normalization is forbidden")
    records = list(payload.get("features") or [])
    names = [str(record.get("name")) for record in records]
    if names != list(feature_order):
        raise ValueError("command normalization feature order mismatch")
    for record in records:
        scale = float(record.get("scale", 0.0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"invalid command normalization scale for {record.get('name')}")
    return recorded_sha256


def apply_command_table_normalization(
    features: Any,
    feature_order: Sequence[str],
    normalization_contract: Mapping[str, Any],
    *,
    expected_normalization_sha256: str | None = None,
) -> np.ndarray:
    """Apply one immutable train-table normalization to packed features."""

    validate_command_normalization_contract(
        normalization_contract,
        feature_order,
        expected_normalization_sha256=expected_normalization_sha256,
    )
    values = np.asarray(features, dtype=np.float64)
    if values.shape[-1] != len(feature_order):
        raise ValueError("packed command feature dimension does not match normalization")
    output = values.copy()
    for index, record in enumerate(normalization_contract["features"]):
        kind = str(record["kind"])
        if kind == "identity_preserve_geometry_or_mask":
            continue
        center = float(record["center"])
        scale = float(record["scale"])
        output[..., index] = (output[..., index] - center) / scale
    return output


__all__ = [
    "fit_command_table_normalization",
    "apply_command_table_normalization",
    "validate_command_normalization_contract",
]
