"""System-control-only V10 campaign authority issuance."""

from __future__ import annotations

from typing import Dict

from .schemas import V10CampaignAuthority, V10ModelPackage


def simulation_authority(campaign_id: str) -> V10CampaignAuthority:
    authority = V10CampaignAuthority(
        campaign_id=campaign_id,
        simulate_only=True,
        operator_approval=False,
        live_gate_results={f"E{index}": False for index in range(6)},
        five_state_protocol_eligible=False,
        hardware_action_authorized=False,
    )
    authority.validate()
    return authority


def issue_live_campaign_authority(
    *,
    campaign_id: str,
    model_package: V10ModelPackage,
    live_gate_results: Dict[str, bool],
    operator_approval: bool,
) -> V10CampaignAuthority:
    """Issue authority only after external live evidence; never called by a model."""
    model_package.validate()
    expected_gates = {f"E{index}" for index in range(6)}
    if set(live_gate_results) != expected_gates or not all(live_gate_results.values()):
        raise ValueError("live campaign authority requires complete passing E0-E5 evidence")
    if not operator_approval:
        raise PermissionError("live campaign authority requires explicit operator approval")
    authority = V10CampaignAuthority(
        campaign_id=campaign_id,
        simulate_only=False,
        operator_approval=bool(operator_approval),
        live_gate_results=dict(live_gate_results),
        five_state_protocol_eligible=True,
        hardware_action_authorized=True,
    )
    authority.validate()
    return authority
