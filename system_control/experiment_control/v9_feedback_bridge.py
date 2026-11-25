"""V9 feedback bridge (charter sections 23.3, 38).

Research-only boundary between the V9 model package and system-control. The bridge
accepts a ``V9OperationPointActionProposalV1``, verifies it against the frozen identity
set (target/registry/command/checkpoint hashes), and produces an operator-approval
request record. It never imports AWG uploaders, FPGA programmers, or the ExperimentRunner,
and it grants no upload authority.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ml_recon.physics_hybrid_v9.contracts import V9OperationPointActionProposalV1

from .v9_operation_action_contract import bind_proposal_to_contract


@dataclass(frozen=True)
class V9OperatorApprovalRequestV1:
    """An operator-approval request produced by the bridge (never self-authorized)."""

    request_id: str
    proposal: Dict[str, Any]
    binding: Dict[str, Any]
    frozen_identity: Dict[str, str]
    status: str = "PENDING_OPERATOR_APPROVAL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "proposal_id": self.proposal.get("proposal_id"),
            "system_type": self.proposal.get("system_type"),
            "action_stage": self.proposal.get("action_stage"),
            "no_op_eligible": self.proposal.get("no_op_eligible"),
            "binding": self.binding,
            "frozen_identity": self.frozen_identity,
            "status": self.status,
            "hardware_action_authorized": False,
            "operator_approval_required": True,
        }


class V9FeedbackBridge:
    """Research-only bridge: proposal -> verified operator-approval request.

    Model package has no upload authority. Approval, materialization, upload, verify,
    measure, accept/rollback all remain in system-control under explicit operator action.
    """

    def __init__(
        self,
        *,
        expected_target_hash: str,
        expected_registry_hash: str,
        expected_command_table_hash: str,
        expected_checkpoint_hash: str,
        request_counter: int = 0,
    ):
        self.expected_target_hash = expected_target_hash
        self.expected_registry_hash = expected_registry_hash
        self.expected_command_table_hash = expected_command_table_hash
        self.expected_checkpoint_hash = expected_checkpoint_hash
        self._counter = request_counter

    def submit_proposal(
        self,
        proposal: V9OperationPointActionProposalV1,
        contract,
    ) -> V9OperatorApprovalRequestV1:
        """Verify the proposal against the frozen identity set and emit an approval request."""
        binding = bind_proposal_to_contract(
            proposal, contract,
            expected_target_hash=self.expected_target_hash,
            expected_registry_hash=self.expected_registry_hash,
            expected_command_table_hash=self.expected_command_table_hash,
            expected_checkpoint_hash=self.expected_checkpoint_hash,
        )
        self._counter += 1
        return V9OperatorApprovalRequestV1(
            request_id=f"v9_req_{self._counter:04d}",
            proposal=proposal.to_dict(),
            binding=binding,
            frozen_identity={
                "target_hash": self.expected_target_hash,
                "registry_hash": self.expected_registry_hash,
                "command_table_hash": self.expected_command_table_hash,
                "checkpoint_hash": self.expected_checkpoint_hash,
            },
            status="PENDING_OPERATOR_APPROVAL",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_target_hash": self.expected_target_hash,
            "expected_registry_hash": self.expected_registry_hash,
            "expected_command_table_hash": self.expected_command_table_hash,
            "expected_checkpoint_hash": self.expected_checkpoint_hash,
            "requests_submitted": self._counter,
        }
