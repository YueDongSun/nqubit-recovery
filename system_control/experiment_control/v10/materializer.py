"""Pure V10 action materialization; no AWG or matrix filesystem writes."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ml_recon.physics_hybrid_v10.hardware_contract import validate_relative_action

from .schemas import (
    V10ActionProposal,
    V10CampaignAuthority,
    V10MaterializedArtifact,
    stable_sha,
)


def materialize_action(
    *,
    proposal: V10ActionProposal,
    authority: V10CampaignAuthority,
    gate_result: Dict[str, object],
    cumulative_log_amp: np.ndarray,
    cumulative_phase: np.ndarray,
    previous_artifact_sha: str,
) -> V10MaterializedArtifact:
    proposal.validate()
    authority.validate()
    if not bool(gate_result.get("passed", False)):
        raise PermissionError("V10 proposal gate did not pass")
    if not authority.simulate_only:
        raise PermissionError(
            "live V10 materialization is intentionally unavailable before validated E0-E5 implementation")
    delta_amp = np.asarray(proposal.delta_log_amp, dtype=np.float64)
    delta_phase = np.asarray(proposal.delta_phase, dtype=np.float64)
    bounds = validate_relative_action(
        delta_amp, delta_phase,
        np.asarray(cumulative_log_amp, dtype=np.float64),
        np.asarray(cumulative_phase, dtype=np.float64))
    guard_results = {
        "proposal_gate": True,
        "relative_action_bounds": bool(bounds["passed"]),
        "simulate_only": True,
        "no_matrix_write": True,
        "no_awg_upload": True,
    }
    if not all(guard_results.values()):
        raise ValueError(f"V10 materialization guard failed: {bounds['reasons']}")
    requested_sha = stable_sha({
        "delta_log_amp": proposal.delta_log_amp,
        "delta_phase": proposal.delta_phase,
    })
    actual_sha = requested_sha
    artifact = V10MaterializedArtifact(
        artifact_id=f"sim_{proposal.proposal_id}",
        proposal_sha=proposal.sha256(),
        campaign_id=authority.campaign_id,
        previous_artifact_sha=previous_artifact_sha,
        requested_action_sha=requested_sha,
        actual_action_sha=actual_sha,
        guard_results=guard_results,
        discrepancy={"max_abs_log_amp": 0.0, "max_abs_phase": 0.0},
        simulate_only=True,
        waveform_hashes={},
    )
    return artifact
