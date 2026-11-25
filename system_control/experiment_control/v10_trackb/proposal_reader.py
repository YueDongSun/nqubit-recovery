"""Fail-closed reader for immutable Track B research proposals.

This module is intentionally independent of model loading and inference. It
accepts only canonical JSON bytes produced by the research package, binds each
proposal to the current engineering context, and preserves exact bytes and the
digest for later review. It does not generate a command or contact hardware.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REQUIRED_AUTHORITY_FLAGS = {
    "research_only": True,
    "conclusion_only": True,
    "selection_evidence": False,
    "promotion_ready": False,
    "hardware_action_authorized": False,
    "external_execution_required": True,
    "causal_gain_available": False,
    "real_transition_supervision": False,
}
_PROPOSAL_FIELDS = frozenset({
    "schema",
    "proposal_id",
    "system_type",
    "source_commit",
    "source_tree",
    "model_checkpoint_sha",
    "model_artifact_kind",
    "model_provider_type",
    "model_output_bundle_sha",
    "data_manifest_sha",
    "desired_target_sha",
    "action_sha",
    "current_control_snapshot_sha",
    "parent_command_bundle_sha",
    "bounds_policy_sha",
    "rf_coordinate_layout_sha",
    "rf_command_basis_sha",
    "logical_coordinate_layout_sha",
    "hardware_path_id",
    "operating_point_id",
    "operation_registry_sha",
    "command_table_sha",
    "selected_candidate_sha",
    "eligibility_policy_sha",
    "eligibility_receipt_sha",
    "posterior_receipt_sha",
    "applied_evaluation_sha",
    "rf_to_awg_mapping_sha",
    "awg_recipe_sha",
    "expires_at_utc",
    "logical_action",
    *_REQUIRED_AUTHORITY_FLAGS,
})
_LOGICAL_ACTION_FIELDS = frozenset({
    "schema",
    "system_type",
    "amplitude_coordinates",
    "phase_coordinates",
    "coordinate_layout_sha",
})
_MODEL_ARTIFACT_PROVIDERS = {
    "FILE_BACKED_CHECKPOINT": {
        "TRACKB_REALFD_V1",
        "FROZEN_V24",
        "TRACK_A_FROZEN_V10",
    },
    "ANALYTIC_PROVIDER_BUNDLE": {"ANALYTIC_V24", "TRACKB_LOCAL_ANALYTIC"},
}


class ProposalValidationError(ValueError):
    """Raised when an external proposal is not an exact safe input."""


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the one JSON representation accepted at this ABI boundary."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a lowercase SHA-256 digest for immutable bytes."""

    return hashlib.sha256(value).hexdigest()


def validate_sha256(value: object, field_name: str) -> str:
    """Require a lowercase, full SHA-256 digest rather than a placeholder."""

    text = str(value)
    if not _SHA256_RE.fullmatch(text):
        raise ProposalValidationError(
            f"{field_name} must be a lowercase 64-hex SHA-256 digest"
        )
    return text


def validate_git_object_id(value: object, field_name: str) -> str:
    """Require a lowercase SHA-1/SHA-256 Git object identity."""

    text = str(value)
    if not _GIT_OBJECT_RE.fullmatch(text):
        raise ProposalValidationError(
            f"{field_name} must be a lowercase 40- or 64-hex Git object ID"
        )
    return text


def parse_utc_timestamp(value: object, field_name: str = "expires_at_utc") -> datetime:
    """Parse a timezone-aware ISO-8601 timestamp into UTC."""

    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ProposalValidationError(f"{field_name} must be ISO-8601 UTC time") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ProposalValidationError(f"{field_name} must use a UTC timezone")
    return parsed.astimezone(timezone.utc)


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProposalValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def thaw_json(value: Any) -> Any:
    """Return a mutable JSON-shaped copy of a frozen proposal value."""

    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ProposalBindingsV1:
    """Current engineering identities that a proposal must match exactly."""

    source_commit: str
    source_tree: str
    model_checkpoint_sha: str
    model_artifact_kind: str
    model_provider_type: str
    model_output_bundle_sha: str
    data_manifest_sha: str
    desired_target_sha: str
    action_sha: str
    current_control_snapshot_sha: str
    parent_command_bundle_sha: str
    bounds_policy_sha: str
    rf_coordinate_layout_sha: str
    rf_command_basis_sha: str
    logical_coordinate_layout_sha: str
    hardware_path_id: str
    operating_point_id: str
    operation_registry_sha: str
    command_table_sha: str
    selected_candidate_sha: str
    eligibility_policy_sha: str
    eligibility_receipt_sha: str
    posterior_receipt_sha: str
    applied_evaluation_sha: str
    rf_to_awg_mapping_sha: str
    awg_recipe_sha: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_commit", validate_git_object_id(
            self.source_commit, "source_commit"))
        object.__setattr__(self, "source_tree", validate_git_object_id(
            self.source_tree, "source_tree"))
        for field_name in (
            "model_checkpoint_sha",
            "model_output_bundle_sha",
            "data_manifest_sha",
            "desired_target_sha",
            "action_sha",
            "current_control_snapshot_sha",
            "parent_command_bundle_sha",
            "bounds_policy_sha",
            "rf_coordinate_layout_sha",
            "rf_command_basis_sha",
            "logical_coordinate_layout_sha",
            "operation_registry_sha",
            "command_table_sha",
            "selected_candidate_sha",
            "eligibility_policy_sha",
            "eligibility_receipt_sha",
            "posterior_receipt_sha",
            "applied_evaluation_sha",
            "rf_to_awg_mapping_sha",
            "awg_recipe_sha",
        ):
            object.__setattr__(
                self, field_name, validate_sha256(getattr(self, field_name), field_name)
            )
        _validate_model_identity(self.model_artifact_kind, self.model_provider_type)
        if not str(self.hardware_path_id).strip() or not str(self.operating_point_id).strip():
            raise ProposalValidationError(
                "hardware_path_id and operating_point_id must be explicit"
            )

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "model_checkpoint_sha": self.model_checkpoint_sha,
            "model_artifact_kind": self.model_artifact_kind,
            "model_provider_type": self.model_provider_type,
            "model_output_bundle_sha": self.model_output_bundle_sha,
            "data_manifest_sha": self.data_manifest_sha,
            "desired_target_sha": self.desired_target_sha,
            "action_sha": self.action_sha,
            "current_control_snapshot_sha": self.current_control_snapshot_sha,
            "parent_command_bundle_sha": self.parent_command_bundle_sha,
            "bounds_policy_sha": self.bounds_policy_sha,
            "rf_coordinate_layout_sha": self.rf_coordinate_layout_sha,
            "rf_command_basis_sha": self.rf_command_basis_sha,
            "logical_coordinate_layout_sha": self.logical_coordinate_layout_sha,
            "hardware_path_id": self.hardware_path_id,
            "operating_point_id": self.operating_point_id,
            "operation_registry_sha": self.operation_registry_sha,
            "command_table_sha": self.command_table_sha,
            "selected_candidate_sha": self.selected_candidate_sha,
            "eligibility_policy_sha": self.eligibility_policy_sha,
            "eligibility_receipt_sha": self.eligibility_receipt_sha,
            "posterior_receipt_sha": self.posterior_receipt_sha,
            "applied_evaluation_sha": self.applied_evaluation_sha,
            "rf_to_awg_mapping_sha": self.rf_to_awg_mapping_sha,
            "awg_recipe_sha": self.awg_recipe_sha,
        }


@dataclass(frozen=True)
class ResearchProposalDocumentV1:
    """Exact proposal bytes plus validated, immutable decoded JSON."""

    raw_bytes: bytes
    proposal_sha: str
    payload: Mapping[str, Any]
    expires_at_utc: datetime

    @property
    def system_type(self) -> str:
        return str(self.payload["system_type"])

    @property
    def action_sha(self) -> str:
        return str(self.payload["action_sha"])

    @property
    def logical_action(self) -> Mapping[str, Any]:
        value = self.payload["logical_action"]
        if not isinstance(value, Mapping):
            raise ProposalValidationError("logical_action must be a JSON object")
        return value

    @property
    def logical_action_vector(self) -> Tuple[float, ...]:
        action = self.logical_action
        amplitude = tuple(float(value) for value in action["amplitude_coordinates"])
        phase = tuple(float(value) for value in action["phase_coordinates"])
        return amplitude + phase

    def action_materializer_payload(self) -> Dict[str, Any]:
        """Adapt the explicit grouped ABI to the existing bounded materializer."""
        from .coordinate_adapter import trackb_to_materializer_action_payload

        return trackb_to_materializer_action_payload(
            self.system_type, self.logical_action_vector,
        )

    def assert_bindings(self, expected: ProposalBindingsV1) -> None:
        for field_name, expected_value in expected.to_json_dict().items():
            actual = self.payload.get(field_name)
            if actual != expected_value:
                raise ProposalValidationError(
                    f"proposal {field_name} does not match the current bound value"
                )

    def assert_not_expired(self, now: Optional[datetime] = None) -> None:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if self.expires_at_utc <= current:
            raise ProposalValidationError("proposal has expired")


def read_research_proposal(
    path: str | Path,
    *,
    expected_bindings: Optional[ProposalBindingsV1] = None,
    now: Optional[datetime] = None,
) -> ResearchProposalDocumentV1:
    """Read proposal bytes exactly once, then validate canonical bytes and bindings."""

    raw_bytes = Path(path).read_bytes()
    return parse_research_proposal_bytes(
        raw_bytes, expected_bindings=expected_bindings, now=now,
    )


def parse_research_proposal_bytes(
    raw_bytes: bytes,
    *,
    expected_bindings: Optional[ProposalBindingsV1] = None,
    now: Optional[datetime] = None,
) -> ResearchProposalDocumentV1:
    """Validate one in-memory byte string without opening a proposal a second time."""

    if not isinstance(raw_bytes, bytes) or not raw_bytes:
        raise ProposalValidationError("proposal must contain non-empty bytes")
    try:
        decoded = raw_bytes.decode("utf-8")
        payload = json.loads(decoded, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProposalValidationError("proposal is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ProposalValidationError("proposal root must be a JSON object")
    canonical = canonical_json_bytes(payload)
    if raw_bytes != canonical:
        raise ProposalValidationError("proposal bytes are not canonical JSON")
    _validate_payload(payload)
    document = ResearchProposalDocumentV1(
        raw_bytes=raw_bytes,
        proposal_sha=sha256_bytes(raw_bytes),
        payload=_freeze_json(payload),
        expires_at_utc=parse_utc_timestamp(payload["expires_at_utc"]),
    )
    if expected_bindings is not None:
        document.assert_bindings(expected_bindings)
    document.assert_not_expired(now)
    return document


def _validate_payload(payload: Mapping[str, Any]) -> None:
    if set(payload) != _PROPOSAL_FIELDS:
        raise ProposalValidationError(
            "proposal schema must contain exactly the registered V1 fields"
        )
    if payload.get("schema") != "ResearchRFActionProposalV1":
        raise ProposalValidationError("unsupported research proposal schema")
    if not isinstance(payload.get("proposal_id"), str) or not str(payload["proposal_id"]).strip():
        raise ProposalValidationError("proposal_id is required")
    system_type = str(payload.get("system_type", ""))
    if system_type not in {"3qubit", "4qubit"}:
        raise ProposalValidationError("proposal system_type must be 3qubit or 4qubit")
    for field_name in ("source_commit", "source_tree"):
        validate_git_object_id(payload.get(field_name), field_name)
    for field_name in (
        "model_checkpoint_sha",
        "model_output_bundle_sha",
        "data_manifest_sha",
        "desired_target_sha",
        "action_sha",
        "current_control_snapshot_sha",
        "parent_command_bundle_sha",
        "bounds_policy_sha",
        "rf_coordinate_layout_sha",
        "rf_command_basis_sha",
        "logical_coordinate_layout_sha",
        "operation_registry_sha",
        "command_table_sha",
        "selected_candidate_sha",
        "eligibility_policy_sha",
        "eligibility_receipt_sha",
        "posterior_receipt_sha",
        "applied_evaluation_sha",
        "rf_to_awg_mapping_sha",
        "awg_recipe_sha",
    ):
        validate_sha256(payload.get(field_name), field_name)
    _validate_model_identity(
        str(payload.get("model_artifact_kind", "")),
        str(payload.get("model_provider_type", "")),
    )
    if not str(payload.get("hardware_path_id", "")).strip() or not str(
        payload.get("operating_point_id", "")
    ).strip():
        raise ProposalValidationError(
            "proposal hardware_path_id and operating_point_id are required"
        )
    for field_name, expected_value in _REQUIRED_AUTHORITY_FLAGS.items():
        if payload.get(field_name) is not expected_value:
            raise ProposalValidationError(
                f"proposal {field_name} must be exactly {expected_value!r}"
            )
    parse_utc_timestamp(payload.get("expires_at_utc"))
    action = payload.get("logical_action")
    if not isinstance(action, Mapping):
        raise ProposalValidationError("logical_action must be a JSON object")
    if set(action) != _LOGICAL_ACTION_FIELDS:
        raise ProposalValidationError(
            "logical_action schema must contain exactly the registered V1 fields"
        )
    if action.get("schema") != "TrackBLogicalActionPayloadV1":
        raise ProposalValidationError("unsupported logical_action schema")
    if str(action.get("system_type", "")) != system_type:
        raise ProposalValidationError("logical_action system_type mismatch")
    if str(action.get("coordinate_layout_sha", "")) != str(
        payload["logical_coordinate_layout_sha"]
    ):
        raise ProposalValidationError("logical_action coordinate layout mismatch")
    coordinate_count = 2 if system_type == "3qubit" else 3
    for field_name in ("amplitude_coordinates", "phase_coordinates"):
        values = action.get(field_name)
        if not isinstance(values, list) or len(values) != coordinate_count:
            raise ProposalValidationError(
                f"logical_action {field_name} must have {coordinate_count} values"
            )
        if any(type(value) is bool or type(value) not in (int, float) for value in values):
            raise ProposalValidationError(
                f"logical_action {field_name} must contain JSON numeric values, never booleans or strings"
            )
        try:
            numeric = tuple(float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ProposalValidationError(
                f"logical_action {field_name} must contain floats"
            ) from exc
        if not all(math.isfinite(value) for value in numeric):
            raise ProposalValidationError(
                f"logical_action {field_name} contains non-finite values"
            )
    if sha256_bytes(canonical_json_bytes(action)) != str(payload["action_sha"]):
        raise ProposalValidationError("action_sha does not bind the canonical logical_action")


def _validate_model_identity(model_artifact_kind: str, model_provider_type: str) -> None:
    providers = _MODEL_ARTIFACT_PROVIDERS.get(str(model_artifact_kind))
    if providers is None or str(model_provider_type) not in providers:
        raise ProposalValidationError("proposal model artifact kind/provider type is not registered")


__all__ = [
    "ProposalBindingsV1",
    "ProposalValidationError",
    "ResearchProposalDocumentV1",
    "canonical_json_bytes",
    "parse_research_proposal_bytes",
    "parse_utc_timestamp",
    "read_research_proposal",
    "sha256_bytes",
    "thaw_json",
    "validate_git_object_id",
    "validate_sha256",
]
