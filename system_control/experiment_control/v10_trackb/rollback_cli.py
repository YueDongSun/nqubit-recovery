"""Fail-closed rollback review entry point for Track B command bundles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from system_control.experiment_control.feedback_command_bundle import FeedbackCommandBundleStore

from .execution_authority import ExternalRFExecutionAuthorityV1
from .operator_approval_cli import _validate_staged_authority
from .proposal_reader import ProposalBindingsV1, ProposalValidationError, ResearchProposalDocumentV1
from .stage_command_bundle import load_trackb_staged_binding


@dataclass(frozen=True)
class TrackBRollbackPreviewV1:
    """Exact parent-restoration eligibility without a device-facing rollback."""

    candidate_bundle_id: str
    candidate_bundle_sha: str
    parent_bundle_id: str
    parent_bundle_sha: str
    eligible_for_future_exact_rollback: bool
    executed: bool = False
    preview_only: bool = True
    hardware_action_authorized: bool = False


def rollback_trackb_trial(
    store: FeedbackCommandBundleStore,
    proposal: ResearchProposalDocumentV1,
    bindings: ProposalBindingsV1,
    authority: ExternalRFExecutionAuthorityV1,
    *,
    bundle_id: str,
    execute: bool = False,
    now: Optional[datetime] = None,
) -> TrackBRollbackPreviewV1:
    """Check exact-parent rollback gating; execution is intentionally disabled."""

    binding = load_trackb_staged_binding(store, bundle_id)
    _validate_staged_authority(proposal, bindings, authority, binding, now=now)
    candidate = store.load_manifest(bundle_id)
    parent = store.load_manifest(binding.parent_bundle_id)
    if parent.status != "accepted" or parent.bundle_hash != binding.parent_bundle_sha:
        raise ProposalValidationError("rollback parent is not the exact accepted bound bundle")
    active = store.active_pointer()
    eligible = bool(
        active is not None
        and active.get("bundle_id") == candidate.bundle_id
        and candidate.status in {"activated_for_trial", "measurement_completed"}
    )
    preview = TrackBRollbackPreviewV1(
        candidate_bundle_id=candidate.bundle_id,
        candidate_bundle_sha=candidate.bundle_hash,
        parent_bundle_id=parent.bundle_id,
        parent_bundle_sha=parent.bundle_hash,
        eligible_for_future_exact_rollback=eligible,
    )
    if execute:
        raise PermissionError(
            "live Track B rollback is intentionally unavailable in this dry-run-only bridge"
        )
    return preview


__all__ = ["TrackBRollbackPreviewV1", "rollback_trackb_trial"]
