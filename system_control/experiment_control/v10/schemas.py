"""Hash-bound schemas for the one-way V10 engineering bridge."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple


def stable_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class V10ModelPackage:
    model_id: str
    checkpoint_sha: str
    source_commit: str
    operation_registry_hash: str
    measurement_contract_sha: str
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False
    research_only: bool = True

    def validate(self) -> None:
        if self.model_id not in ("F_3Q_HW_V10", "F_4Q_HW_V10"):
            raise ValueError("system-control accepts only a formal V10 hardware cell")
        if self.selection_evidence or self.promotion_ready or self.hardware_action_authorized:
            raise ValueError("V10 model packages cannot carry promotion or hardware authority")
        if not self.research_only:
            raise ValueError("V10 package must remain research_only")
        if any(len(value) not in (40, 64) for value in (
                self.checkpoint_sha, self.source_commit, self.operation_registry_hash,
                self.measurement_contract_sha)):
            raise ValueError("model-package identities must be SHA digests")


@dataclass(frozen=True)
class V10CampaignAuthority:
    campaign_id: str
    simulate_only: bool
    operator_approval: bool
    live_gate_results: Dict[str, bool]
    five_state_protocol_eligible: bool
    hardware_action_authorized: bool
    authority_issuer: str = "system_control.experiment_control.v10"

    def validate(self) -> None:
        expected_gates = {f"E{index}" for index in range(6)}
        if set(self.live_gate_results) != expected_gates:
            raise ValueError("campaign authority must bind live gates E0-E5")
        if self.hardware_action_authorized:
            if self.simulate_only or not self.operator_approval \
                    or not all(self.live_gate_results.values()):
                raise ValueError("hardware authority requires live E0-E5 and operator approval")
        if self.five_state_protocol_eligible and not all(self.live_gate_results.values()):
            raise ValueError("five-state eligibility requires complete live-gate evidence")
        if self.authority_issuer != "system_control.experiment_control.v10":
            raise ValueError("campaign authority can only be issued by the V10 system-control package")


@dataclass(frozen=True)
class V10ActionProposal:
    proposal_id: str
    system_type: str
    target_hash: str
    current_state_hash: str
    operation_registry_hash: str
    command_table_hash: str
    forward_model_sha: str
    model_checkpoint_sha: str
    delta_log_amp: Tuple[float, ...]
    delta_phase: Tuple[float, ...]
    predicted_gain_quantiles: Tuple[float, float, float]
    harmful_probability: float
    abstain_probability: float
    history_hash: str
    source: str = "V10_research_proposal"
    materialization_authorized: bool = False
    upload_authorized: bool = False

    def validate(self) -> None:
        N = 3 if self.system_type == "3qubit" else (4 if self.system_type == "4qubit" else 0)
        if N == 0 or len(self.delta_log_amp) != N or len(self.delta_phase) != N:
            raise ValueError("V10 hardware proposal must be a global 3Q/4Q W action")
        if abs(sum(self.delta_log_amp)) > 1e-9 or abs(sum(self.delta_phase)) > 1e-9:
            raise ValueError("V10 physical action vectors must be zero mean")
        if self.materialization_authorized or self.upload_authorized:
            raise ValueError("research proposals cannot self-authorize materialization/upload")
        if not (0.0 <= self.harmful_probability <= 1.0
                and 0.0 <= self.abstain_probability <= 1.0):
            raise ValueError("proposal risk probabilities must lie in [0,1]")
        if sorted(self.predicted_gain_quantiles) != list(self.predicted_gain_quantiles):
            raise ValueError("predicted gain quantiles must be monotone")

    def sha256(self) -> str:
        self.validate()
        return stable_sha(asdict(self))


@dataclass(frozen=True)
class V10MaterializedArtifact:
    artifact_id: str
    proposal_sha: str
    campaign_id: str
    previous_artifact_sha: str
    requested_action_sha: str
    actual_action_sha: str
    guard_results: Dict[str, bool]
    discrepancy: Dict[str, float]
    simulate_only: bool
    waveform_hashes: Dict[str, str] = field(default_factory=dict)

    def sha256(self) -> str:
        return stable_sha(asdict(self))


@dataclass(frozen=True)
class V10UploadReceipt:
    receipt_id: str
    artifact_sha: str
    campaign_id: str
    uploaded_waveform_hashes: Dict[str, str]
    instrument_identity: str
    simulate_only: bool
    verified: bool

    def sha256(self) -> str:
        return stable_sha(asdict(self))


@dataclass(frozen=True)
class V10RollbackRecord:
    rollback_id: str
    campaign_id: str
    harmful_artifact_sha: str
    restored_artifact_sha: str
    restored_waveform_hashes: Dict[str, str]
    receipt_sha: str
    simulate_only: bool
    verified: bool
