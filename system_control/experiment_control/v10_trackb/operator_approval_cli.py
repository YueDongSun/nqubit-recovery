"""Explicit operator approval wrapper for a dry-run Track B staged bundle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from system_control.experiment_control.feedback_command_bundle import FeedbackCommandBundleStore

from .execution_authority import ExternalRFExecutionAuthorityV1
from .proposal_reader import ProposalBindingsV1, ProposalValidationError, ResearchProposalDocumentV1
from .stage_command_bundle import TrackBStagedCommandBundleV1, load_trackb_staged_binding


@dataclass(frozen=True)
class TrackBOperatorApprovalV1:
    """Approval result retaining the separate external-authority digest."""

    bundle_id: str
    bundle_sha: str
    proposal_sha: str
    external_authority_sha: str
    status: str
    preview_only: bool = True
    hardware_action_authorized: bool = False


def approve_trackb_staged_bundle(
    store: FeedbackCommandBundleStore,
    proposal: ResearchProposalDocumentV1,
    bindings: ProposalBindingsV1,
    authority: ExternalRFExecutionAuthorityV1,
    *,
    bundle_id: str,
    now: Optional[datetime] = None,
) -> TrackBOperatorApprovalV1:
    """Record operator approval without activating, uploading, or repeating a trial."""

    binding = load_trackb_staged_binding(store, bundle_id)
    _validate_staged_authority(proposal, bindings, authority, binding, now=now)
    manifest = store.load_manifest(bundle_id)
    if manifest.status != "staged":
        raise ProposalValidationError("only a staged Track B bundle may be approved")
    approved = store.approve(
        bundle_id,
        operator=authority.operator_id,
        approval_note=f"trackb_external_authority_sha={authority.sha256()}",
    )
    return TrackBOperatorApprovalV1(
        bundle_id=approved.bundle_id,
        bundle_sha=approved.bundle_hash,
        proposal_sha=proposal.proposal_sha,
        external_authority_sha=authority.sha256(),
        status=approved.status,
    )


def _validate_staged_authority(
    proposal: ResearchProposalDocumentV1,
    bindings: ProposalBindingsV1,
    authority: ExternalRFExecutionAuthorityV1,
    binding: TrackBStagedCommandBundleV1,
    *,
    now: Optional[datetime],
) -> None:
    if binding.proposal_sha != proposal.proposal_sha:
        raise ProposalValidationError("staged binding does not match proposal bytes")
    if binding.action_sha != proposal.action_sha:
        raise ProposalValidationError("staged binding does not match proposal action")
    if binding.parent_bundle_sha != bindings.parent_command_bundle_sha:
        raise ProposalValidationError("staged binding parent does not match current proposal binding")
    authority.assert_matches(
        proposal, bindings, command_bundle_sha=binding.bundle_sha, now=now,
    )


__all__ = ["TrackBOperatorApprovalV1", "approve_trackb_staged_bundle"]
