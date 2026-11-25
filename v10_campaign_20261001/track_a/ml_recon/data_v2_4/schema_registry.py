"""Schema lookup and canonical schema hash for V2.4."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from .contracts import NQubitDatasetContractV240
from .hashing import sha256_json
from .schema_types import FieldSpec


def schema_registry() -> Dict[str, FieldSpec]:
    contract = NQubitDatasetContractV240.default()
    return {item.field_path: item for item in contract.fields}


def schema_sha256() -> str:
    registry = schema_registry()
    return sha256_json({key: value.__dict__ for key, value in sorted(registry.items())})


def require_field(field_path: str) -> FieldSpec:
    try:
        return schema_registry()[field_path]
    except KeyError as error:
        raise KeyError(f"field is not registered in V2.4: {field_path}") from error


def _dtype_matches(declared: str, actual: np.dtype) -> bool:
    choices = set(declared.split("|"))
    if actual.kind == "S":
        return "bytes" in choices
    canonical = {
        "b": "bool", "i": f"int{actual.itemsize * 8}",
        "u": f"uint{actual.itemsize * 8}", "f": f"float{actual.itemsize * 8}",
    }.get(actual.kind, "")
    return canonical in choices


def _shape_matches(rule: str, sample_shape: tuple[int, ...], N: int) -> bool:
    exact = {
        "[]": (), "[N]": (N,), "[N,N]": (N, N), "[9]": (9,),
        "[7]": (7,), "[4]": (4,), "[3]": (3,),
    }
    if rule in exact:
        return sample_shape == exact[rule]
    if rule.endswith(",4]"):
        return bool(sample_shape) and sample_shape[-1] == 4
    return len(sample_shape) >= 1


def validate_persisted_arrays(arrays: Mapping[str, np.ndarray],
                              metadata: Mapping[str, Any]) -> None:
    """Reject unregistered/mistyped fields before a formal shard is written."""
    store = str(metadata.get("store", ""))
    provenance = str(metadata.get("provenance", ""))
    if store not in ("observation", "truth", "derived", "cell_static"):
        raise ValueError("V2.4 shard metadata must declare a registered store")
    if provenance not in ("real", "synthetic"):
        raise ValueError("V2.4 shard metadata must declare real/synthetic provenance")
    registry = schema_registry()
    field_paths = {f"{store}/{name}" for name in arrays}
    unregistered = sorted(field_paths - set(registry))
    if unregistered:
        raise ValueError(f"unregistered V2.4 persisted fields: {unregistered}")
    applicable = [
        spec for spec in registry.values()
        if spec.store == store and (
            provenance in spec.provenance_applicability
            or "both" in spec.provenance_applicability)
    ]
    missing_required = sorted(
        spec.field_path for spec in applicable
        if spec.required and spec.field_path not in field_paths)
    if missing_required:
        raise ValueError(f"missing required V2.4 fields: {missing_required}")
    N = int(metadata.get("N", 0))
    for name, value in arrays.items():
        spec = registry[f"{store}/{name}"]
        array = np.asarray(value)
        if not _dtype_matches(spec.dtype, array.dtype):
            raise ValueError(
                f"V2.4 dtype mismatch for {spec.field_path}: {array.dtype} vs {spec.dtype}")
        if not _shape_matches(spec.shape_rule, tuple(array.shape[1:]), N):
            raise ValueError(
                f"V2.4 shape mismatch for {spec.field_path}: {array.shape[1:]} "
                f"vs {spec.shape_rule}")
        mask_name = spec.mask_path.split("/", 1)[1]
        if mask_name not in arrays:
            raise ValueError(f"V2.4 mask is missing for {spec.field_path}: {spec.mask_path}")
        if np.asarray(arrays[mask_name]).dtype.kind != "b":
            raise ValueError(f"V2.4 mask must be bool: {spec.mask_path}")
