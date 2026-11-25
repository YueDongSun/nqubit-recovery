"""Separate, dry-run-only operator authority for a Track B proposal.

The research proposal never grants execution permission. This small contract
records independent operator-side identifiers that a future transaction would
need. In this branch it can validate and reserve a nonce, but it cannot
activate a trial or invoke a device-facing interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Optional, Tuple

from .proposal_reader import (
    ProposalBindingsV1,
    ProposalValidationError,
    ResearchProposalDocumentV1,
    canonical_json_bytes,
    parse_utc_timestamp,
    sha256_bytes,
    validate_sha256,
)


@dataclass(frozen=True)
class ExternalRFExecutionAuthorityV1:
    """A non-model, one-trial authority record with no live grant in Track B."""

    operator_id: str
    session_id: str
    approval_receipt_sha: str
    proposal_sha: str
    current_control_snapshot_sha: str
    parent_command_bundle_sha: str
    command_bundle_sha: str
    bounds_policy_sha: str
    one_shot_nonce: str
    expires_at_utc: str
    trial_only: bool = True
    automatic_repeated_execution: bool = False
    operator_approval_required: bool = True
    simulate_only: bool = True
    hardware_action_authorized: bool = False

    def __post_init__(self) -> None:
        if not str(self.operator_id).strip():
            raise ProposalValidationError("operator_id is required")
        if not str(self.session_id).strip():
            raise ProposalValidationError("session_id is required")
        if not str(self.one_shot_nonce).strip():
            raise ProposalValidationError("one_shot_nonce is required")
        for field_name in (
            "approval_receipt_sha",
            "proposal_sha",
            "current_control_snapshot_sha",
            "parent_command_bundle_sha",
            "command_bundle_sha",
            "bounds_policy_sha",
        ):
            object.__setattr__(
                self, field_name, validate_sha256(getattr(self, field_name), field_name)
            )
        parse_utc_timestamp(self.expires_at_utc)
        if self.trial_only is not True:
            raise ProposalValidationError("external authority must be trial_only")
        if self.automatic_repeated_execution is not False:
            raise ProposalValidationError("automatic repeated execution is forbidden")
        if self.operator_approval_required is not True:
            raise ProposalValidationError("operator approval cannot be disabled")
        if self.simulate_only is not True or self.hardware_action_authorized is not False:
            raise ProposalValidationError(
                "Track B external authority is dry-run-only and cannot grant hardware authority"
            )

    @property
    def expires_at(self) -> datetime:
        return parse_utc_timestamp(self.expires_at_utc)

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes({
            "schema": "ExternalRFExecutionAuthorityV1",
            "operator_id": self.operator_id,
            "session_id": self.session_id,
            "approval_receipt_sha": self.approval_receipt_sha,
            "proposal_sha": self.proposal_sha,
            "current_control_snapshot_sha": self.current_control_snapshot_sha,
            "parent_command_bundle_sha": self.parent_command_bundle_sha,
            "command_bundle_sha": self.command_bundle_sha,
            "bounds_policy_sha": self.bounds_policy_sha,
            "one_shot_nonce": self.one_shot_nonce,
            "expires_at_utc": self.expires_at_utc,
            "trial_only": self.trial_only,
            "automatic_repeated_execution": self.automatic_repeated_execution,
            "operator_approval_required": self.operator_approval_required,
            "simulate_only": self.simulate_only,
            "hardware_action_authorized": self.hardware_action_authorized,
        })

    def sha256(self) -> str:
        return sha256_bytes(self.canonical_json_bytes())

    def assert_matches(
        self,
        proposal: ResearchProposalDocumentV1,
        bindings: ProposalBindingsV1,
        *,
        command_bundle_sha: str,
        now: Optional[datetime] = None,
    ) -> None:
        proposal.assert_bindings(bindings)
        proposal.assert_not_expired(now)
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if self.expires_at <= current:
            raise ProposalValidationError("external execution authority has expired")
        expected_values = {
            "proposal_sha": proposal.proposal_sha,
            "current_control_snapshot_sha": bindings.current_control_snapshot_sha,
            "parent_command_bundle_sha": bindings.parent_command_bundle_sha,
            "bounds_policy_sha": bindings.bounds_policy_sha,
            "command_bundle_sha": validate_sha256(command_bundle_sha, "command_bundle_sha"),
        }
        for field_name, expected_value in expected_values.items():
            if getattr(self, field_name) != expected_value:
                raise ProposalValidationError(
                    f"external authority {field_name} does not match the bound transaction"
                )


class OneShotNonceRegistry:
    """In-memory replay guard for a future explicitly enabled executor.

    The current dry-run executor never claims a nonce. Keeping the registry
    separate makes replay protection testable without manufacturing a trial.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._claimed: Dict[Tuple[str, str], str] = {}

    def claim(self, authority: ExternalRFExecutionAuthorityV1) -> None:
        key = (authority.session_id, authority.one_shot_nonce)
        with self._lock:
            if key in self._claimed:
                raise ProposalValidationError("one_shot_nonce replay rejected")
            self._claimed[key] = authority.sha256()

    def is_claimed(self, authority: ExternalRFExecutionAuthorityV1) -> bool:
        key = (authority.session_id, authority.one_shot_nonce)
        with self._lock:
            return key in self._claimed


__all__ = [
    "ExternalRFExecutionAuthorityV1",
    "OneShotNonceRegistry",
]
