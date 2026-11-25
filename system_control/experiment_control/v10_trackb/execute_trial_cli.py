"""Fail-closed Track B trial execution entry point.

The callable exposes only a reviewable preview. Passing ``execute=True`` is
intentionally rejected before any state-machine activation is attempted.
"""

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
class TrackBTrialActivationPreviewV1:
    """A non-executing statement of what a future separately authorized path needs."""

    bundle_id: str
    bundle_sha: str
    proposal_sha: str
    status: str
    eligible_for_future_external_trial: bool
    executed: bool = False
    preview_only: bool = True
    hardware_action_authorized: bool = False


def execute_trackb_trial(
    store: FeedbackCommandBundleStore,
    proposal: ResearchProposalDocumentV1,
    bindings: ProposalBindingsV1,
    authority: ExternalRFExecutionAuthorityV1,
    *,
    bundle_id: str,
    execute: bool = False,
    now: Optional[datetime] = None,
) -> TrackBTrialActivationPreviewV1:
    """Validate approval and return a preview; live activation is unavailable."""

    binding = load_trackb_staged_binding(store, bundle_id)
    _validate_staged_authority(proposal, bindings, authority, binding, now=now)
    manifest = store.load_manifest(bundle_id)
    if manifest.status != "operator_approved":
        raise ProposalValidationError("Track B trial activation requires operator_approved bundle")
    current = store.active_pointer()
    if current is None or current.get("bundle_id") != binding.parent_bundle_id:
        raise ProposalValidationError("active command bundle is not the bound accepted parent")
    preview = TrackBTrialActivationPreviewV1(
        bundle_id=manifest.bundle_id,
        bundle_sha=manifest.bundle_hash,
        proposal_sha=proposal.proposal_sha,
        status=manifest.status,
        eligible_for_future_external_trial=True,
    )
    if execute:
        raise PermissionError(
            "live Track B trial execution is intentionally unavailable in this dry-run-only bridge"
        )
    return preview


__all__ = ["TrackBTrialActivationPreviewV1", "execute_trackb_trial"]
