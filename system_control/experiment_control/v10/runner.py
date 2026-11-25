"""Simulation-only V10 bridge runner; live execution fails closed."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .gate import evaluate_proposal_gate
from .materializer import materialize_action
from .receipt_verifier import verify_receipt
from .schemas import (
    V10ActionProposal,
    V10CampaignAuthority,
    V10ModelPackage,
    V10UploadReceipt,
)


class V10FeedbackRunner:
    def __init__(self, authority: V10CampaignAuthority,
                 model_package: V10ModelPackage):
        authority.validate()
        model_package.validate()
        self.authority = authority
        self.model_package = model_package

    def simulate(
        self,
        *,
        proposal: V10ActionProposal,
        expected_target_hash: str,
        expected_command_table_hash: str,
        cumulative_log_amp: np.ndarray,
        cumulative_phase: np.ndarray,
        previous_artifact_sha: str = "0" * 64,
    ) -> Dict[str, object]:
        if not self.authority.simulate_only:
            raise PermissionError("simulate() requires simulation authority")
        gate = evaluate_proposal_gate(
            proposal=proposal, model_package=self.model_package,
            authority=self.authority, expected_target_hash=expected_target_hash,
            expected_command_table_hash=expected_command_table_hash)
        artifact = materialize_action(
            proposal=proposal, authority=self.authority, gate_result=gate,
            cumulative_log_amp=cumulative_log_amp,
            cumulative_phase=cumulative_phase,
            previous_artifact_sha=previous_artifact_sha)
        receipt = V10UploadReceipt(
            receipt_id=f"sim_receipt_{artifact.artifact_id}",
            artifact_sha=artifact.sha256(),
            campaign_id=self.authority.campaign_id,
            uploaded_waveform_hashes={},
            instrument_identity="SIMULATION_NO_INSTRUMENT",
            simulate_only=True,
            verified=True,
        )
        verification = verify_receipt(artifact, receipt)
        if not verification["passed"]:
            raise RuntimeError("internal simulation receipt verification failed")
        return {"gate": gate, "artifact": artifact, "receipt": receipt,
                "receipt_verification": verification,
                "hardware_touched": False}

    def execute_live(self, *args, **kwargs):
        raise PermissionError(
            "live V10 execution is fail-closed; no validated E0-E5 implementation is installed")
