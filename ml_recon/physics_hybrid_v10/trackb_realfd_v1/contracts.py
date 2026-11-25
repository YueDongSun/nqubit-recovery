"""Research-only contracts for the V10 Track B RealFD/RF bridge.

This module deliberately describes conclusions and proposals without possessing
any execution authority.  It has no dependency on ``system_control``, AWG
drivers, FPGA code, or mutable command state.  A consumer must revalidate the
canonical bytes before it can consider a proposal for an external transaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
import re
from typing import Any, Mapping, Sequence


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
MODEL_ARTIFACT_KINDS = frozenset({"FILE_BACKED_CHECKPOINT", "ANALYTIC_PROVIDER_BUNDLE"})
MODEL_PROVIDER_TYPES = frozenset({
    "TRACKB_REALFD_V1", "FROZEN_V24", "TRACK_A_FROZEN_V10",
    "ANALYTIC_V24", "TRACKB_LOCAL_ANALYTIC",
})
V24_ORACLE_HISTORY_VIEW = "v24_oracle_history_v1"
LEARNER_VISITED_HISTORY_VIEW = "learner_visited_history_v1"
LEARNER_VISITED_ROW_SCHEMA = "TrackBLearnerVisitedHistoryRowV1"


def canonical_json_bytes(value: object) -> bytes:
    """Return the one canonical JSON encoding used across the Track B seam."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON (or immutable bytes) without a platform dependency."""
    payload = value if isinstance(value, bytes) else canonical_json_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def parse_canonical_json_object(
    payload: bytes, *, schema: str, exact_fields: frozenset[str],
) -> dict[str, object]:
    """Parse one canonical JSON object with duplicate and unknown-key rejection."""
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("canonical JSON object cannot contain duplicate keys")
            result[key] = value
        return result

    try:
        decoded = json.loads(payload.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("payload is not valid canonical UTF-8 JSON") from error
    if type(decoded) is not dict or decoded.get("schema") != schema or set(decoded) != exact_fields:
        raise ValueError("canonical JSON schema or exact field set is invalid")
    if canonical_json_bytes(decoded) != payload:
        raise ValueError("JSON bytes are valid but not canonical")
    return decoded


def require_exact_canonical_json_round_trip(payload: bytes, value: object) -> None:
    """Reject a parser result that would normalize a canonical input artifact.

    ``parse_canonical_json_object`` establishes JSON lexical canonicality, but
    a typed ``from_dict`` implementation can still coerce an otherwise
    canonical value (for example a string number) into a different in-memory
    object.  File-backed Track B artifacts are an exact transport contract, so
    every external reader must prove that reserializing the parsed value
    produces the original bytes.
    """
    serializer = getattr(value, "canonical_json_bytes", None)
    if not callable(serializer):
        raise TypeError("canonical parser result must expose canonical_json_bytes")
    if serializer() != payload:
        raise ValueError("canonical JSON payload changes under typed parsing")


def require_sha256(value: object, name: str) -> str:
    """Require a lowercase SHA-256 identity; placeholders cannot pass."""
    if type(value) is not str or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def require_git_oid(value: object, name: str) -> str:
    """Require a lowercase 40/64 hexadecimal Git object identity."""
    if type(value) is not str or not _GIT_OID_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 40/64-hex Git object id")
    return value


def _finite_tuple(values: Sequence[object], name: str) -> tuple[float, ...]:
    if any(type(value) is bool or not isinstance(value, Real) for value in values):
        raise ValueError(f"{name} must contain JSON numeric values, never booleans or strings")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_utc_timestamp(value: str) -> None:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError("expires_at_utc must be ISO-8601 UTC") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("expires_at_utc must use a UTC offset")


def _system_mode_count(system_type: str) -> int:
    values = {"3qubit": 3, "4qubit": 4}
    try:
        return values[system_type]
    except KeyError as error:
        raise ValueError("Track B supports only 3qubit or 4qubit") from error


@dataclass(frozen=True)
class TrackBResearchAuthorityV1:
    """Non-overridable authority statement carried by every public output."""

    research_only: bool = True
    conclusion_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False
    external_execution_required: bool = True
    causal_gain_available: bool = False
    real_transition_supervision: bool = False

    def validate(self) -> None:
        required_true = ("research_only", "conclusion_only", "external_execution_required")
        required_false = (
            "selection_evidence", "promotion_ready", "hardware_action_authorized",
            "causal_gain_available", "real_transition_supervision",
        )
        if any(type(getattr(self, name)) is not bool for name in required_true + required_false):
            raise TypeError("Track B authority fields must be exact booleans")
        if any(not getattr(self, name) for name in required_true):
            raise ValueError("Track B outputs must remain research/conclusion-only")
        if any(getattr(self, name) for name in required_false):
            raise ValueError("Track B model outputs cannot assert selection, causal, or hardware authority")

    def to_dict(self) -> dict[str, bool]:
        self.validate()
        return {
            "research_only": self.research_only,
            "conclusion_only": self.conclusion_only,
            "selection_evidence": self.selection_evidence,
            "promotion_ready": self.promotion_ready,
            "hardware_action_authorized": self.hardware_action_authorized,
            "external_execution_required": self.external_execution_required,
            "causal_gain_available": self.causal_gain_available,
            "real_transition_supervision": self.real_transition_supervision,
        }


@dataclass(frozen=True)
class TrackBRealV5CompatibilityManifestV1:
    """Read-only compatibility record; it never mutates ``real_v5`` semantics."""

    base_commit: str
    base_tree: str
    real_v5_contract_sha: str
    data_v24_schema_sha: str
    field_aliases: tuple[tuple[str, str], ...]
    deprecated_fields: tuple[str, ...]
    new_optional_fields: tuple[str, ...]
    coordinate_adapter_sha: str
    backward_compatibility_tests: tuple[str, ...]
    schema: str = field(default="TrackBRealV5CompatibilityManifestV1", init=False)

    def validate(self) -> None:
        require_git_oid(self.base_commit, "base_commit")
        require_git_oid(self.base_tree, "base_tree")
        for name in ("real_v5_contract_sha", "data_v24_schema_sha", "coordinate_adapter_sha"):
            require_sha256(getattr(self, name), name)
        if any(
            type(left) is not str or not left or type(right) is not str or not right
            for left, right in self.field_aliases
        ):
            raise ValueError("field aliases must be nonempty string pairs")
        for values, name in (
            (self.deprecated_fields, "deprecated_fields"),
            (self.new_optional_fields, "new_optional_fields"),
            (self.backward_compatibility_tests, "backward_compatibility_tests"),
        ):
            if not values or any(type(value) is not str or not value for value in values):
                raise ValueError(f"{name} must name at least one explicit item")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "base_commit": self.base_commit,
            "base_tree": self.base_tree,
            "real_v5_contract_sha": self.real_v5_contract_sha,
            "data_v24_schema_sha": self.data_v24_schema_sha,
            "field_aliases": [list(value) for value in self.field_aliases],
            "deprecated_fields": list(self.deprecated_fields),
            "new_optional_fields": list(self.new_optional_fields),
            "coordinate_adapter_sha": self.coordinate_adapter_sha,
            "backward_compatibility_tests": list(self.backward_compatibility_tests),
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

REAL_STATIC_FORBIDDEN_LOSSES = frozenset({
    "action", "action_mse", "gain", "gain_pinball", "jacobian", "executed_action",
    "after_state", "hardware_response", "causal_policy", "transition",
})


def validate_real_static_loss_terms(loss_names: Sequence[str]) -> None:
    """Reject causal/control supervision before a real-static optimizer can run."""
    values = tuple(loss_names)
    if any(type(value) is not str for value in values):
        raise TypeError("loss names must be strings")
    lowered = {value.lower() for value in values}
    if REAL_STATIC_FORBIDDEN_LOSSES & lowered:
        raise ValueError("real_static cannot receive action, gain, Jacobian, or causal supervision")
    # Preserve the historical V5 rule as an additional read-only compatibility gate.
    from ..real_v5.contracts import validate_static_loss_contract
    validate_static_loss_contract(values)


@dataclass(frozen=True)
class RealStaticSupervisionV1:
    """Explicitly records the absent causal labels on a real-static row."""

    dataset_family: str = "real_static"
    executed_action_available: bool = False
    after_action_state_available: bool = False
    real_gain_available: bool = False
    actuator_jacobian_available: bool = False
    causal_policy_label_available: bool = False

    def validate(self) -> None:
        if self.dataset_family != "real_static":
            raise ValueError("this supervision guard is only valid for real_static rows")
        fields = (
            "executed_action_available", "after_action_state_available", "real_gain_available",
            "actuator_jacobian_available", "causal_policy_label_available",
        )
        if any(type(getattr(self, name)) is not bool for name in fields):
            raise TypeError("real-static availability fields must be exact booleans")
        if any(getattr(self, name) for name in fields):
            raise ValueError("real_static rows cannot claim causal action/response labels")


@dataclass(frozen=True)
class TrackBProposalEligibilityReceiptV1:
    """Immutable cross-object receipt required before an actionful proposal."""

    selection_sha: str
    candidate_library_sha: str
    registered_candidate_sha: str
    applied_candidate_sha: str
    posterior_sha: str
    forward_draw_set_sha: str
    csm_sha: str
    rsm_sha: str
    envelope_sha: str
    eligibility_policy_sha: str
    rf_mapping_core_sha: str
    rf_preview_sha: str
    rf_command_delta_sha: str
    evaluation_set_sha: str
    posterior_receipt_sha: str
    applied_evaluation_sha: str
    logical_action_payload_sha: str
    rf_to_awg_mapping_sha: str
    awg_recipe_sha: str
    schema: str = field(default="TrackBProposalEligibilityReceiptV1", init=False)

    def validate(self) -> None:
        for name in (
            "selection_sha", "candidate_library_sha", "registered_candidate_sha",
            "applied_candidate_sha", "posterior_sha", "forward_draw_set_sha", "csm_sha",
            "rsm_sha", "envelope_sha", "eligibility_policy_sha", "rf_mapping_core_sha",
            "rf_preview_sha", "rf_command_delta_sha", "evaluation_set_sha",
            "posterior_receipt_sha", "applied_evaluation_sha", "logical_action_payload_sha",
            "rf_to_awg_mapping_sha", "awg_recipe_sha",
        ):
            require_sha256(getattr(self, name), name)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "selection_sha": self.selection_sha,
            "candidate_library_sha": self.candidate_library_sha,
            "registered_candidate_sha": self.registered_candidate_sha,
            "applied_candidate_sha": self.applied_candidate_sha,
            "posterior_sha": self.posterior_sha,
            "forward_draw_set_sha": self.forward_draw_set_sha,
            "csm_sha": self.csm_sha,
            "rsm_sha": self.rsm_sha,
            "envelope_sha": self.envelope_sha,
            "eligibility_policy_sha": self.eligibility_policy_sha,
            "rf_mapping_core_sha": self.rf_mapping_core_sha,
            "rf_preview_sha": self.rf_preview_sha,
            "rf_command_delta_sha": self.rf_command_delta_sha,
            "evaluation_set_sha": self.evaluation_set_sha,
            "posterior_receipt_sha": self.posterior_receipt_sha,
            "applied_evaluation_sha": self.applied_evaluation_sha,
            "logical_action_payload_sha": self.logical_action_payload_sha,
            "rf_to_awg_mapping_sha": self.rf_to_awg_mapping_sha,
            "awg_recipe_sha": self.awg_recipe_sha,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrackBProposalEligibilityReceiptV1":
        expected = {
            "schema", "selection_sha", "candidate_library_sha", "registered_candidate_sha",
            "applied_candidate_sha", "posterior_sha", "forward_draw_set_sha", "csm_sha",
            "rsm_sha", "envelope_sha", "eligibility_policy_sha", "rf_mapping_core_sha",
            "rf_preview_sha", "rf_command_delta_sha", "evaluation_set_sha",
            "posterior_receipt_sha", "applied_evaluation_sha", "logical_action_payload_sha",
            "rf_to_awg_mapping_sha", "awg_recipe_sha",
        }
        if set(payload) != expected or payload.get("schema") != "TrackBProposalEligibilityReceiptV1":
            raise ValueError("proposal eligibility receipt schema is invalid")
        receipt = cls(
            selection_sha=str(payload["selection_sha"]),
            candidate_library_sha=str(payload["candidate_library_sha"]),
            registered_candidate_sha=str(payload["registered_candidate_sha"]),
            applied_candidate_sha=str(payload["applied_candidate_sha"]),
            posterior_sha=str(payload["posterior_sha"]),
            forward_draw_set_sha=str(payload["forward_draw_set_sha"]),
            csm_sha=str(payload["csm_sha"]),
            rsm_sha=str(payload["rsm_sha"]),
            envelope_sha=str(payload["envelope_sha"]),
            eligibility_policy_sha=str(payload["eligibility_policy_sha"]),
            rf_mapping_core_sha=str(payload["rf_mapping_core_sha"]),
            rf_preview_sha=str(payload["rf_preview_sha"]),
            rf_command_delta_sha=str(payload["rf_command_delta_sha"]),
            evaluation_set_sha=str(payload["evaluation_set_sha"]),
            posterior_receipt_sha=str(payload["posterior_receipt_sha"]),
            applied_evaluation_sha=str(payload["applied_evaluation_sha"]),
            logical_action_payload_sha=str(payload["logical_action_payload_sha"]),
            rf_to_awg_mapping_sha=str(payload["rf_to_awg_mapping_sha"]),
            awg_recipe_sha=str(payload["awg_recipe_sha"]),
        )
        receipt.validate()
        return receipt

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "TrackBProposalEligibilityReceiptV1":
        decoded = parse_canonical_json_object(
            payload,
            schema="TrackBProposalEligibilityReceiptV1",
            exact_fields=frozenset({
                "schema", "selection_sha", "candidate_library_sha", "registered_candidate_sha",
                "applied_candidate_sha", "posterior_sha", "forward_draw_set_sha", "csm_sha",
                "rsm_sha", "envelope_sha", "eligibility_policy_sha", "rf_mapping_core_sha",
                "rf_preview_sha", "rf_command_delta_sha", "evaluation_set_sha",
                "posterior_receipt_sha", "applied_evaluation_sha", "logical_action_payload_sha",
                "rf_to_awg_mapping_sha", "awg_recipe_sha",
            }),
        )
        receipt = cls.from_dict(decoded)
        require_exact_canonical_json_round_trip(payload, receipt)
        return receipt


@dataclass(frozen=True)
class ResearchRFActionProposalV1:
    """Immutable research proposal for an external, independently-authorized consumer.

    The top-level ``action_sha`` is *only* the SHA-256 of the canonical nested
    ``logical_action`` object.  This allows a consumer to validate action bytes
    without importing this package or relying on dataclass field order.
    """

    proposal_id: str
    system_type: str
    source_commit: str
    source_tree: str
    # This is always a SHA.  Analytic providers bind their immutable provider
    # bundle SHA here and declare that fact through ``model_artifact_kind``;
    # they never serialize a fake checkpoint or an unbound null identity.
    model_checkpoint_sha: str
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
    expires_at_utc: str
    amplitude_coordinates: tuple[float, ...]
    phase_coordinates: tuple[float, ...]
    model_artifact_kind: str = "FILE_BACKED_CHECKPOINT"
    model_provider_type: str = "TRACKB_REALFD_V1"
    authority: TrackBResearchAuthorityV1 = field(default_factory=TrackBResearchAuthorityV1)
    schema: str = field(default="ResearchRFActionProposalV1", init=False)

    @property
    def logical_action(self) -> dict[str, object]:
        return {
            "schema": "TrackBLogicalActionPayloadV1",
            "system_type": self.system_type,
            "amplitude_coordinates": list(self.amplitude_coordinates),
            "phase_coordinates": list(self.phase_coordinates),
            "coordinate_layout_sha": self.logical_coordinate_layout_sha,
        }

    def validate(self) -> None:
        if type(self.proposal_id) is not str or not self.proposal_id:
            raise ValueError("proposal_id must be a nonempty string")
        mode_count = _system_mode_count(self.system_type)
        expected = mode_count - 1
        amplitude = _finite_tuple(self.amplitude_coordinates, "amplitude_coordinates")
        phase = _finite_tuple(self.phase_coordinates, "phase_coordinates")
        if len(amplitude) != expected or len(phase) != expected:
            raise ValueError("proposal logical action does not match the system mode count")
        require_git_oid(self.source_commit, "source_commit")
        require_git_oid(self.source_tree, "source_tree")
        for name in (
            "model_output_bundle_sha", "data_manifest_sha", "desired_target_sha", "action_sha",
            "current_control_snapshot_sha", "parent_command_bundle_sha", "bounds_policy_sha",
            "rf_coordinate_layout_sha", "rf_command_basis_sha", "logical_coordinate_layout_sha",
            "operation_registry_sha", "command_table_sha", "selected_candidate_sha",
            "eligibility_policy_sha",
            "eligibility_receipt_sha",
            "posterior_receipt_sha", "applied_evaluation_sha", "rf_to_awg_mapping_sha",
            "awg_recipe_sha",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.hardware_path_id) is not str or not self.hardware_path_id:
            raise ValueError("hardware_path_id must be source-bound and nonempty")
        if type(self.operating_point_id) is not str or not self.operating_point_id:
            raise ValueError("operating_point_id must be source-bound and nonempty")
        require_sha256(self.model_checkpoint_sha, "model_checkpoint_sha")
        if self.model_artifact_kind not in MODEL_ARTIFACT_KINDS:
            raise ValueError("model_artifact_kind is not registered")
        if self.model_provider_type not in MODEL_PROVIDER_TYPES:
            raise ValueError("model_provider_type is not registered")
        if self.model_artifact_kind == "FILE_BACKED_CHECKPOINT" and self.model_provider_type not in (
            "TRACKB_REALFD_V1", "FROZEN_V24", "TRACK_A_FROZEN_V10",
        ):
            raise ValueError("file-backed checkpoint identity requires a frozen/model provider type")
        if self.model_artifact_kind == "ANALYTIC_PROVIDER_BUNDLE" and self.model_provider_type not in (
            "ANALYTIC_V24", "TRACKB_LOCAL_ANALYTIC",
        ):
            raise ValueError("analytic provider bundle identity requires an analytic provider type")
        if type(self.expires_at_utc) is not str:
            raise ValueError("expires_at_utc must be an explicit UTC timestamp")
        _validate_utc_timestamp(self.expires_at_utc)
        if self.action_sha != canonical_sha256(self.logical_action):
            raise ValueError("action_sha must bind the canonical logical_action payload")
        self.authority.validate()
    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "proposal_id": self.proposal_id,
            "system_type": self.system_type,
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
            "expires_at_utc": self.expires_at_utc,
            "logical_action": self.logical_action,
            **self.authority.to_dict(),
        }

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def sha256(self) -> str:
        return canonical_sha256(self.canonical_json_bytes())

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "ResearchRFActionProposalV1":
        """Parse exactly one canonical JSON representation and immediately revalidate."""
        def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("canonical proposal JSON cannot contain duplicate keys")
                result[key] = value
            return result

        try:
            decoded = json.loads(payload.decode("utf-8"), object_pairs_hook=no_duplicates)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("proposal bytes are not valid canonical UTF-8 JSON") from error
        if type(decoded) is not dict or decoded.get("schema") != "ResearchRFActionProposalV1":
            raise ValueError("proposal schema is invalid")
        expected_fields = {
            "schema", "proposal_id", "system_type", "source_commit", "source_tree",
            "model_checkpoint_sha", "model_artifact_kind", "model_provider_type",
            "model_output_bundle_sha", "data_manifest_sha", "desired_target_sha", "action_sha",
            "current_control_snapshot_sha", "parent_command_bundle_sha", "bounds_policy_sha",
            "rf_coordinate_layout_sha", "rf_command_basis_sha", "logical_coordinate_layout_sha",
            "hardware_path_id", "operating_point_id", "operation_registry_sha", "command_table_sha",
            "selected_candidate_sha", "eligibility_policy_sha", "eligibility_receipt_sha",
            "posterior_receipt_sha", "applied_evaluation_sha", "rf_to_awg_mapping_sha",
            "awg_recipe_sha", "expires_at_utc", "logical_action", "research_only",
            "conclusion_only", "selection_evidence", "promotion_ready",
            "hardware_action_authorized", "external_execution_required", "causal_gain_available",
            "real_transition_supervision",
        }
        if set(decoded) != expected_fields:
            raise ValueError("proposal canonical JSON has unknown or missing fields")
        logical = decoded.get("logical_action")
        if type(logical) is not dict or logical.get("schema") != "TrackBLogicalActionPayloadV1":
            raise ValueError("proposal logical_action is invalid")
        authority = TrackBResearchAuthorityV1(
            research_only=decoded.get("research_only"),
            conclusion_only=decoded.get("conclusion_only"),
            selection_evidence=decoded.get("selection_evidence"),
            promotion_ready=decoded.get("promotion_ready"),
            hardware_action_authorized=decoded.get("hardware_action_authorized"),
            external_execution_required=decoded.get("external_execution_required"),
            causal_gain_available=decoded.get("causal_gain_available"),
            real_transition_supervision=decoded.get("real_transition_supervision"),
        )
        instance = cls(
            proposal_id=decoded.get("proposal_id"),
            system_type=decoded.get("system_type"),
            source_commit=decoded.get("source_commit"),
            source_tree=decoded.get("source_tree"),
            model_checkpoint_sha=decoded.get("model_checkpoint_sha"),
            model_output_bundle_sha=decoded.get("model_output_bundle_sha"),
            data_manifest_sha=decoded.get("data_manifest_sha"),
            desired_target_sha=decoded.get("desired_target_sha"),
            action_sha=decoded.get("action_sha"),
            current_control_snapshot_sha=decoded.get("current_control_snapshot_sha"),
            parent_command_bundle_sha=decoded.get("parent_command_bundle_sha"),
            bounds_policy_sha=decoded.get("bounds_policy_sha"),
            rf_coordinate_layout_sha=decoded.get("rf_coordinate_layout_sha"),
            rf_command_basis_sha=decoded.get("rf_command_basis_sha"),
            logical_coordinate_layout_sha=decoded.get("logical_coordinate_layout_sha"),
            hardware_path_id=decoded.get("hardware_path_id"),
            operating_point_id=decoded.get("operating_point_id"),
            operation_registry_sha=decoded.get("operation_registry_sha"),
            command_table_sha=decoded.get("command_table_sha"),
            selected_candidate_sha=decoded.get("selected_candidate_sha"),
            eligibility_policy_sha=decoded.get("eligibility_policy_sha"),
            eligibility_receipt_sha=decoded.get("eligibility_receipt_sha"),
            posterior_receipt_sha=decoded.get("posterior_receipt_sha"),
            applied_evaluation_sha=decoded.get("applied_evaluation_sha"),
            rf_to_awg_mapping_sha=decoded.get("rf_to_awg_mapping_sha"),
            awg_recipe_sha=decoded.get("awg_recipe_sha"),
            expires_at_utc=decoded.get("expires_at_utc"),
            amplitude_coordinates=tuple(logical.get("amplitude_coordinates", ())),
            phase_coordinates=tuple(logical.get("phase_coordinates", ())),
            model_artifact_kind=decoded.get("model_artifact_kind", "FILE_BACKED_CHECKPOINT"),
            model_provider_type=decoded.get("model_provider_type", "TRACKB_REALFD_V1"),
            authority=authority,
        )
        if logical.get("system_type") != instance.system_type or (
            logical.get("coordinate_layout_sha") != instance.logical_coordinate_layout_sha
        ):
            raise ValueError("logical_action duplicates do not match the top-level proposal")
        if instance.canonical_json_bytes() != payload:
            raise ValueError("proposal bytes are valid JSON but not canonical JSON")
        return instance


@dataclass(frozen=True)
class TrackBFDResearchConclusionV1:
    """Pure research conclusion; no causal or hardware claim is representable."""

    source_commit: str
    source_tree: str
    data_manifest_sha: str
    selected_candidate_sha: str
    envelope_sha: str
    reason_codes: tuple[str, ...]
    eligibility_policy_sha: str
    proposal_sha: str | None = None
    authority: TrackBResearchAuthorityV1 = field(default_factory=TrackBResearchAuthorityV1)
    schema: str = field(default="TrackBFDResearchConclusionV1", init=False)

    def validate(self) -> None:
        require_git_oid(self.source_commit, "source_commit")
        require_git_oid(self.source_tree, "source_tree")
        for name in (
            "data_manifest_sha", "selected_candidate_sha", "envelope_sha",
            "eligibility_policy_sha",
        ):
            require_sha256(getattr(self, name), name)
        if self.proposal_sha is not None:
            require_sha256(self.proposal_sha, "proposal_sha")
        if not self.reason_codes or any(type(code) is not str or not code for code in self.reason_codes):
            raise ValueError("research conclusion requires explicit reason codes")
        self.authority.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "data_manifest_sha": self.data_manifest_sha,
            "selected_candidate_sha": self.selected_candidate_sha,
            "envelope_sha": self.envelope_sha,
            "proposal_sha": self.proposal_sha,
            "reason_codes": list(self.reason_codes),
            "eligibility_policy_sha": self.eligibility_policy_sha,
            **self.authority.to_dict(),
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


__all__ = [
    "REAL_STATIC_FORBIDDEN_LOSSES", "RealStaticSupervisionV1",
    "ResearchRFActionProposalV1", "TrackBFDResearchConclusionV1",
    "TrackBProposalEligibilityReceiptV1",
    "TrackBRealV5CompatibilityManifestV1", "TrackBResearchAuthorityV1",
    "canonical_json_bytes", "canonical_sha256", "parse_canonical_json_object",
    "require_git_oid", "require_sha256",
    "validate_real_static_loss_terms", "MODEL_ARTIFACT_KINDS", "MODEL_PROVIDER_TYPES",
    "V24_ORACLE_HISTORY_VIEW", "LEARNER_VISITED_HISTORY_VIEW", "LEARNER_VISITED_ROW_SCHEMA",
]
