"""Immutable, non-uploading Track B binding to the existing command-bundle store."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from system_control.experiment_control.feedback_command_bundle import (
    CommandBundleManifestV1,
    FeedbackCommandBundleStore,
)
from system_control.experiment_control.sequence_service import PreparedSequenceArtifactV1
from system_control.experiment_control.models import ControlStateSpecV1, TargetStateSpecV1

from .materialization_preview import RFMaterializationPreviewV1, preview_rf_materialization
from .proposal_reader import (
    ProposalBindingsV1,
    ProposalValidationError,
    ResearchProposalDocumentV1,
    canonical_json_bytes,
    sha256_bytes,
    validate_sha256,
)
from .rf_inverse import NominalRFInverseContractV1


@dataclass(frozen=True)
class TrackBStagedCommandBundleV1:
    """Immutable link between proposal bytes, preview, parent, and staged bundle."""

    proposal_sha: str
    action_sha: str
    desired_target_sha: str
    coordinate_adapter_sha: str
    preview_sha: str
    requested_action_hash: str
    projected_action_hash: str
    actual_action_hash: str
    control_hash: str
    bundle_id: str
    bundle_sha: str
    parent_bundle_id: str
    parent_bundle_sha: str
    prepared_artifact_id: str
    system_type: str
    preview_only: bool = True
    hardware_action_authorized: bool = False

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": "TrackBStagedCommandBundleV1",
            "proposal_sha": self.proposal_sha,
            "action_sha": self.action_sha,
            "desired_target_sha": self.desired_target_sha,
            "coordinate_adapter_sha": self.coordinate_adapter_sha,
            "preview_sha": self.preview_sha,
            "requested_action_hash": self.requested_action_hash,
            "projected_action_hash": self.projected_action_hash,
            "actual_action_hash": self.actual_action_hash,
            "control_hash": self.control_hash,
            "bundle_id": self.bundle_id,
            "bundle_sha": self.bundle_sha,
            "parent_bundle_id": self.parent_bundle_id,
            "parent_bundle_sha": self.parent_bundle_sha,
            "prepared_artifact_id": self.prepared_artifact_id,
            "system_type": self.system_type,
            "preview_only": True,
            "hardware_action_authorized": False,
        }

    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_json_dict()))


def stage_trackb_command_bundle(
    store: FeedbackCommandBundleStore,
    proposal: ResearchProposalDocumentV1,
    bindings: ProposalBindingsV1,
    artifact: PreparedSequenceArtifactV1,
    *,
    inverse_contract: NominalRFInverseContractV1,
    target_state: TargetStateSpecV1,
    control_state: ControlStateSpecV1,
    relative_bundle_dir: str,
    parent_bundle_id: str,
) -> TrackBStagedCommandBundleV1:
    """Recompute preview then stage an already prepared artifact.

    A caller may show a previous preview to an operator, but staging never
    trusts a caller-provided diagnostic object: it recomputes the inverse,
    bounded materialization and post-materialization audit from the proposal
    bytes and current bound control context immediately before writing state.
    """

    proposal.assert_bindings(bindings)
    preview = preview_rf_materialization(
        proposal,
        bindings,
        inverse_contract,
        target_state=target_state,
        control_state=control_state,
    )
    proposal.assert_not_expired()
    if not preview.preview_only or preview.hardware_action_authorized:
        raise ProposalValidationError("only a dry-run preview may be staged")
    if not preview.accepted:
        raise ProposalValidationError("rejected materialization preview cannot be staged")
    if preview.proposal_sha != proposal.proposal_sha or preview.action_sha != proposal.action_sha:
        raise ProposalValidationError("materialization preview does not bind proposal bytes/action")
    expected_artifact_hashes = {
        "requested_action_hash": preview.requested_action_hash,
        "projected_action_hash": preview.projected_action_hash,
        "actual_action_hash": preview.actual_action_hash,
        "control_hash": preview.control_hash,
    }
    for field_name, expected_value in expected_artifact_hashes.items():
        if str(getattr(artifact, field_name)) != expected_value:
            raise ProposalValidationError(
                f"prepared artifact {field_name} does not bind the accepted preview"
            )
    if artifact.system_type != proposal.system_type:
        raise ProposalValidationError("prepared artifact system_type does not match proposal")
    if artifact.target_hash != bindings.desired_target_sha:
        raise ProposalValidationError("prepared artifact target does not match proposal binding")
    if artifact.upload_receipt is not None:
        raise ProposalValidationError("Track B can stage only a pre-upload artifact")
    parent = store.load_manifest(parent_bundle_id)
    if parent.status != "accepted":
        raise ProposalValidationError("Track B parent must be an accepted bundle")
    if parent.bundle_hash != bindings.parent_command_bundle_sha:
        raise ProposalValidationError("parent command bundle SHA does not match proposal binding")
    if parent.system_type != proposal.system_type:
        raise ProposalValidationError("parent command bundle system_type mismatch")
    if artifact.parent_control_hash != parent.control_hash:
        raise ProposalValidationError("prepared artifact does not bind the parent control state")
    if artifact.registry_hash != bindings.operation_registry_sha:
        raise ProposalValidationError("prepared artifact operation registry does not match proposal")
    if artifact.command_table_hash != bindings.command_table_sha:
        raise ProposalValidationError("prepared artifact command table does not match proposal")
    manifest = store.stage_prepared_artifact(
        artifact, relative_bundle_dir, parent_bundle_id=parent_bundle_id,
    )
    return _write_staged_binding(
        store,
        TrackBStagedCommandBundleV1(
            proposal_sha=proposal.proposal_sha,
            action_sha=proposal.action_sha,
            desired_target_sha=bindings.desired_target_sha,
            coordinate_adapter_sha=preview.coordinate_adapter_sha,
            preview_sha=preview.sha256(),
            requested_action_hash=preview.requested_action_hash,
            projected_action_hash=preview.projected_action_hash,
            actual_action_hash=preview.actual_action_hash,
            control_hash=preview.control_hash,
            bundle_id=manifest.bundle_id,
            bundle_sha=manifest.bundle_hash,
            parent_bundle_id=parent.bundle_id,
            parent_bundle_sha=parent.bundle_hash,
            prepared_artifact_id=artifact.artifact_id,
            system_type=proposal.system_type,
        ),
    )


def load_trackb_staged_binding(
    store: FeedbackCommandBundleStore, bundle_id: str,
) -> TrackBStagedCommandBundleV1:
    """Load and re-hash the immutable companion record for a staged bundle."""

    path = _binding_path(store, bundle_id)
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProposalValidationError("staged binding is not valid UTF-8 JSON") from exc
    if raw != canonical_json_bytes(payload):
        raise ProposalValidationError("staged binding bytes are not canonical")
    if payload.get("schema") != "TrackBStagedCommandBundleV1":
        raise ProposalValidationError("unsupported staged binding schema")
    values = {name: payload.get(name) for name in (
        "proposal_sha", "action_sha", "desired_target_sha", "coordinate_adapter_sha", "preview_sha",
        "requested_action_hash", "projected_action_hash", "actual_action_hash",
        "control_hash", "bundle_id", "bundle_sha",
        "parent_bundle_id", "parent_bundle_sha", "prepared_artifact_id", "system_type",
    )}
    if not all(isinstance(value, str) and value for value in values.values()):
        raise ProposalValidationError("staged binding is incomplete")
    if payload.get("preview_only") is not True or payload.get("hardware_action_authorized") is not False:
        raise ProposalValidationError("staged binding authority flags are invalid")
    for field_name in (
        "proposal_sha", "action_sha", "desired_target_sha", "coordinate_adapter_sha", "preview_sha",
        "requested_action_hash", "projected_action_hash", "actual_action_hash",
        "control_hash", "bundle_sha", "parent_bundle_sha",
    ):
        validate_sha256(values[field_name], field_name)
    binding = TrackBStagedCommandBundleV1(**values)
    manifest = store.load_manifest(binding.bundle_id)
    if manifest.bundle_hash != binding.bundle_sha:
        raise ProposalValidationError("staged binding bundle SHA does not match immutable manifest")
    if manifest.parent_bundle_id != binding.parent_bundle_id:
        raise ProposalValidationError("staged binding parent ID does not match immutable manifest")
    if manifest.system_type != binding.system_type:
        raise ProposalValidationError("staged binding system_type does not match immutable manifest")
    if manifest.target_hash != binding.desired_target_sha:
        raise ProposalValidationError("staged binding target does not match immutable manifest")
    if manifest.control_hash != binding.control_hash:
        raise ProposalValidationError("staged binding control hash does not match immutable manifest")
    for name, expected in (
        ("requested", binding.requested_action_hash),
        ("projected", binding.projected_action_hash),
        ("actual", binding.actual_action_hash),
    ):
        if manifest.action_hashes.get(name) != expected:
            raise ProposalValidationError(
                f"staged binding {name} action hash does not match immutable manifest"
            )
    parent = store.load_manifest(binding.parent_bundle_id)
    if parent.bundle_hash != binding.parent_bundle_sha:
        raise ProposalValidationError("staged binding parent SHA does not match immutable manifest")
    return binding


def _write_staged_binding(
    store: FeedbackCommandBundleStore,
    binding: TrackBStagedCommandBundleV1,
) -> TrackBStagedCommandBundleV1:
    path = _binding_path(store, binding.bundle_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(binding.to_json_dict())
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise ProposalValidationError("immutable staged binding already differs")
    return binding


def _binding_path(store: FeedbackCommandBundleStore, bundle_id: str) -> Path:
    return store.session_dir / "trackb_staged_bindings" / f"{bundle_id}.json"


__all__ = [
    "TrackBStagedCommandBundleV1",
    "load_trackb_staged_binding",
    "stage_trackb_command_bundle",
]
