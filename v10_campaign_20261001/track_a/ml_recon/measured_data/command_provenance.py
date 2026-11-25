"""Fail-closed command provenance resolution for historical snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class CommandProvenanceResult:
    provenance: str
    command_table_id: Optional[str]
    command_table_hash: Optional[str]
    operation_registry_hash: Optional[str]
    matrix_snapshot_hash: Optional[str]
    waveform_hash: Optional[str]
    supervision_mask: int
    reason: str


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_payload_hash(payload: Any, expected_hash: str, label: str) -> None:
    actual_hash = _stable_hash(payload)
    if actual_hash != str(expected_hash):
        raise ValueError(f"{label} hash mismatch: expected {expected_hash}, got {actual_hash}")


def resolve_command_provenance(
    snapshot_dir: str | Path,
    snapshot_manifest: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> CommandProvenanceResult:
    """Resolve snapshot-bound commands without consulting current matrices."""
    directory = Path(snapshot_dir)
    command_table_id = snapshot_manifest.get("command_table_id")
    command_table_hash = snapshot_manifest.get("command_table_hash")
    registry_hash = snapshot_manifest.get("operation_registry_hash")
    matrix_hash = snapshot_manifest.get("matrix_snapshot_hash")
    waveform_hash = snapshot_manifest.get("waveform_hash")

    command_payload = snapshot_manifest.get("command_table")
    if command_payload is not None:
        if not command_table_id or not command_table_hash:
            raise ValueError("incomplete exact command table provenance")
        _verify_payload_hash(command_payload, str(command_table_hash), "command table")
        return CommandProvenanceResult(
            "exact_prepared_artifact", str(command_table_id), str(command_table_hash),
            _optional_str(registry_hash), _optional_str(matrix_hash), _optional_str(waveform_hash),
            1, "snapshot embeds an exact hash-verified command table",
        )

    if (command_table_id is None) != (command_table_hash is None):
        raise ValueError("incomplete command table ID/hash pair")

    registry_payload = snapshot_manifest.get("operation_point_registry")
    if registry_payload is not None:
        if not registry_hash:
            raise ValueError("incomplete snapshot registry provenance")
        _verify_payload_hash(registry_payload, str(registry_hash), "operation registry")
        return CommandProvenanceResult(
            "exact_snapshot_registry", None, None, str(registry_hash),
            _optional_str(matrix_hash), _optional_str(waveform_hash), 1,
            "snapshot embeds a hash-verified operation registry",
        )

    artifact = snapshot_manifest.get("prepared_sequence_artifact")
    if artifact is not None:
        artifact_hash = artifact.get("artifact_hash") if isinstance(artifact, Mapping) else None
        artifact_payload = artifact.get("payload") if isinstance(artifact, Mapping) else None
        if not artifact_hash or artifact_payload is None:
            raise ValueError("incomplete prepared artifact provenance")
        _verify_payload_hash(artifact_payload, str(artifact_hash), "prepared artifact")
        return CommandProvenanceResult(
            "exact_prepared_artifact", _optional_str(command_table_id),
            _optional_str(command_table_hash), _optional_str(registry_hash),
            _optional_str(matrix_hash), _optional_str(waveform_hash), 1,
            "snapshot embeds a hash-verified prepared artifact",
        )

    reconstruction = snapshot_manifest.get("historical_command_reconstruction")
    if reconstruction is not None:
        if not isinstance(reconstruction, Mapping):
            raise ValueError("historical command reconstruction must be a mapping")
        expected_matrix = reconstruction.get("matrix_snapshot_hash")
        expected_waveform = reconstruction.get("waveform_hash")
        payload = reconstruction.get("payload")
        if payload is None or not expected_matrix or not expected_waveform:
            raise ValueError("historical reconstruction lacks matrix/waveform hashes")
        if not matrix_hash or not waveform_hash:
            raise ValueError("historical reconstruction is not bound to snapshot hashes")
        if str(matrix_hash) != str(expected_matrix):
            raise ValueError("historical reconstruction matrix hash mismatch")
        if str(waveform_hash) != str(expected_waveform):
            raise ValueError("historical reconstruction waveform hash mismatch")
        return CommandProvenanceResult(
            "verified_historical_reconstruction", _optional_str(command_table_id),
            _optional_str(command_table_hash), _optional_str(registry_hash),
            str(expected_matrix), str(expected_waveform), 1,
            "deterministic reconstruction is bound to historical hashes",
        )

    target = snapshot_manifest.get("target_state_params") or (
        metadata.get("feedback_context", {}) if isinstance(metadata.get("feedback_context"), Mapping) else {}
    ).get("target_state_params")
    if isinstance(target, Mapping) and target.get("amplitudes") and (
        target.get("phases_deg") or target.get("phases_rad")
    ):
        return CommandProvenanceResult(
            "target_only_approximation", None, None, _optional_str(registry_hash),
            _optional_str(matrix_hash), _optional_str(waveform_hash), 0,
            "only nominal target controls are recoverable",
        )

    # The directory argument is deliberately unused for fallback reads.  It is
    # accepted only so callers can include it in diagnostics.
    _ = directory
    return CommandProvenanceResult(
        "missing", None, None, _optional_str(registry_hash), _optional_str(matrix_hash),
        _optional_str(waveform_hash), 0, "no snapshot-bound command facts",
    )


def _optional_str(value: Any) -> Optional[str]:
    return None if value in (None, "") else str(value)


__all__ = ["CommandProvenanceResult", "resolve_command_provenance"]
