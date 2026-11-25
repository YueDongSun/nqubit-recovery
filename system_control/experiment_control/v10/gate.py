"""Target/hash/risk/history gate for V10 proposals."""

from __future__ import annotations

from typing import Dict

from .schemas import V10ActionProposal, V10CampaignAuthority, V10ModelPackage


def evaluate_proposal_gate(
    *,
    proposal: V10ActionProposal,
    model_package: V10ModelPackage,
    authority: V10CampaignAuthority,
    expected_target_hash: str,
    expected_command_table_hash: str,
    maximum_harmful_probability: float = 0.10,
    maximum_abstain_probability: float = 0.50,
    minimum_predicted_gain: float = 0.0,
) -> Dict[str, object]:
    proposal.validate()
    model_package.validate()
    authority.validate()
    reasons = []
    if proposal.target_hash != expected_target_hash:
        reasons.append("target_hash_mismatch")
    if proposal.command_table_hash != expected_command_table_hash:
        reasons.append("command_table_hash_mismatch")
    if proposal.operation_registry_hash != model_package.operation_registry_hash:
        reasons.append("operation_registry_hash_mismatch")
    if proposal.model_checkpoint_sha != model_package.checkpoint_sha:
        reasons.append("model_checkpoint_hash_mismatch")
    if proposal.harmful_probability > maximum_harmful_probability:
        reasons.append("harm_risk_too_high")
    if proposal.abstain_probability > maximum_abstain_probability:
        reasons.append("model_abstained")
    if proposal.predicted_gain_quantiles[1] <= minimum_predicted_gain:
        reasons.append("nonpositive_median_gain")
    if len(proposal.history_hash) != 64:
        reasons.append("history_hash_missing")
    if not authority.simulate_only and not authority.hardware_action_authorized:
        reasons.append("campaign_hardware_authority_missing")
    return {
        "passed": not reasons,
        "reasons": tuple(reasons),
        "simulation_materialization_allowed": not reasons and authority.simulate_only,
        "live_materialization_allowed": (
            not reasons and not authority.simulate_only
            and authority.hardware_action_authorized),
        "upload_allowed": False,
    }
