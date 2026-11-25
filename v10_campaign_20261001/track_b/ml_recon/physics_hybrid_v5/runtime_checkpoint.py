"""Strict, atomic runtime checkpoints for V6 joint research training.

Checkpoint metadata is part of the executable contract.  A load fails closed
when command, normalization, policy, geometry, detector/source, probe, graph,
dataset, calibration, or stage identity is absent or inconsistent.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


CHECKPOINT_SCHEMA = "PhysicsHybridV6JointCheckpoint"
CHECKPOINT_VERSION = 1
REQUIRED_METADATA_FIELDS = (
    "architecture_id",
    "architecture_sha256",
    "system_type",
    "command_schema",
    "command_schema_hash",
    "command_table_hashes",
    "command_normalization_sha256",
    "command_feature_order",
    "policy_schema_hash",
    "action_geometry_schema",
    "action_geometry_version",
    "detector_schema",
    "source_multiplicity_schema",
    "probe_schema",
    "registry_graph_hash",
    "curriculum_stage_index",
    "curriculum_stage_name",
    "dataset_sha256",
    "calibration_sha256",
    "backend",
    "config_sha256",
    "metrics",
    "research_only",
    "promotion_ready",
)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_joint_checkpoint_metadata(
    metadata: Mapping[str, Any],
    *,
    expected: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    value = dict(metadata)
    missing = [name for name in REQUIRED_METADATA_FIELDS if name not in value]
    if missing:
        raise ValueError(f"V6 checkpoint metadata is incomplete: {missing}")
    if value.get("schema") != CHECKPOINT_SCHEMA or int(value.get("schema_version", 0)) != CHECKPOINT_VERSION:
        raise ValueError("unsupported V6 joint checkpoint metadata schema")
    if value["research_only"] is not True:
        raise ValueError("V6 checkpoint must remain research_only=true")
    if value["promotion_ready"] is not False:
        raise ValueError("V6 checkpoint cannot set promotion_ready")
    if not value["command_table_hashes"]:
        raise ValueError("V6 checkpoint must bind at least one command table hash")
    if not value["command_feature_order"]:
        raise ValueError("V6 checkpoint command feature order is empty")
    for name in (
        "architecture_sha256", "command_schema_hash", "command_normalization_sha256",
        "policy_schema_hash", "registry_graph_hash", "dataset_sha256",
        "calibration_sha256", "config_sha256",
    ):
        if len(str(value[name])) != 64:
            raise ValueError(f"V6 checkpoint metadata field {name} is not a SHA-256")
    recorded_hash = str(value.pop("metadata_sha256", "") or "")
    computed_hash = canonical_sha256(value)
    if recorded_hash and recorded_hash != computed_hash:
        raise ValueError("V6 checkpoint metadata SHA-256 mismatch")
    value["metadata_sha256"] = computed_hash
    if expected is not None:
        mismatched = {
            name: {"checkpoint": value.get(name), "expected": expected_value}
            for name, expected_value in expected.items()
            if value.get(name) != expected_value
        }
        if mismatched:
            raise ValueError(f"V6 checkpoint runtime contract mismatch: {mismatched}")
    return value


def build_joint_checkpoint_metadata(
    *,
    architecture_id: str,
    architecture_config: Mapping[str, Any],
    system_type: str,
    command_schema: str,
    command_schema_hash: str,
    command_table_hashes: Sequence[str],
    command_normalization_sha256: str,
    command_feature_order: Sequence[str],
    policy_config: Mapping[str, Any],
    action_geometry_schema: str,
    action_geometry_version: int,
    detector_schema: str,
    source_multiplicity_schema: str,
    probe_schema: str,
    registry_graph_hash: str,
    curriculum_stage_index: int,
    curriculum_stage_name: str,
    dataset_sha256: str,
    calibration_sha256: str,
    backend: str,
    config_sha256: str,
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_VERSION,
        "architecture_id": str(architecture_id),
        "architecture_sha256": canonical_sha256(architecture_config),
        "system_type": str(system_type),
        "command_schema": str(command_schema),
        "command_schema_hash": str(command_schema_hash),
        "command_table_hashes": sorted(str(value) for value in command_table_hashes),
        "command_normalization_sha256": str(command_normalization_sha256),
        "command_feature_order": list(command_feature_order),
        "policy_schema_hash": canonical_sha256(policy_config),
        "action_geometry_schema": str(action_geometry_schema),
        "action_geometry_version": int(action_geometry_version),
        "detector_schema": str(detector_schema),
        "source_multiplicity_schema": str(source_multiplicity_schema),
        "probe_schema": str(probe_schema),
        "registry_graph_hash": str(registry_graph_hash),
        "curriculum_stage_index": int(curriculum_stage_index),
        "curriculum_stage_name": str(curriculum_stage_name),
        "dataset_sha256": str(dataset_sha256),
        "calibration_sha256": str(calibration_sha256),
        "backend": str(backend),
        "config_sha256": str(config_sha256),
        "metrics": dict(metrics),
        "research_only": True,
        "promotion_ready": False,
    }
    metadata["metadata_sha256"] = canonical_sha256(metadata)
    return validate_joint_checkpoint_metadata(metadata)


def save_joint_checkpoint(
    path: str | Path,
    model: Any,
    metadata: Mapping[str, Any],
    *,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    training_state: Optional[Mapping[str, Any]] = None,
) -> str:
    if torch is None:
        raise RuntimeError("save_joint_checkpoint requires torch")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checked_metadata = validate_joint_checkpoint_metadata(metadata)
    payload = {
        "metadata": checked_metadata,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "training_state": dict(training_state or {}),
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    checkpoint_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    receipt = {
        "checkpoint_path": output.name,
        "checkpoint_sha256": checkpoint_sha256,
        "metadata_sha256": checked_metadata["metadata_sha256"],
        "research_only": True,
        "promotion_ready": False,
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    receipt_temporary = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.tmp")
    receipt_temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(receipt_temporary, receipt_path)
    return checkpoint_sha256


def load_joint_checkpoint(
    path: str | Path,
    model: Any,
    *,
    expected_metadata: Optional[Mapping[str, Any]] = None,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    map_location: str | Any = "cpu",
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("load_joint_checkpoint requires torch")
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V6 checkpoint payload must be a mapping")
    metadata = validate_joint_checkpoint_metadata(
        payload.get("metadata") or {}, expected=expected_metadata,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return {
        "metadata": metadata,
        "training_state": dict(payload.get("training_state") or {}),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_VERSION",
    "REQUIRED_METADATA_FIELDS",
    "build_joint_checkpoint_metadata",
    "canonical_sha256",
    "load_joint_checkpoint",
    "save_joint_checkpoint",
    "validate_joint_checkpoint_metadata",
]
