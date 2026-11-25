"""V9 operation action contract binding (charter section 23).

Binds a ``V9OperationPointActionProposalV1`` to the immutable action contract: W_CH1-only
scope, hard bounds, registry/command/checkpoint identity. The binding is research-only:
it validates identity and bounds; it grants no upload authority.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ml_recon.physics_hybrid_v9.contracts import (
    V9OperationPointActionContractV1,
    V9OperationPointActionProposalV1,
)


def bind_proposal_to_contract(
    proposal: V9OperationPointActionProposalV1,
    contract: V9OperationPointActionContractV1,
    *,
    expected_registry_hash: str,
    expected_command_table_hash: str,
    expected_checkpoint_hash: str,
    expected_target_hash: str,
) -> Dict[str, Any]:
    """Validate a proposal against the contract and the frozen identity set.

    Returns a binding record (no authority flags change). Raises on any violation.
    """
    violations: List[str] = []

    # Identity binding (charter 23: proposal carries registry/command/checkpoint/target hash).
    for name, actual, expected in (
        ("target_hash", proposal.target_hash, expected_target_hash),
        ("operation_registry_hash", proposal.operation_registry_hash, expected_registry_hash),
        ("command_table_hash", proposal.command_table_hash, expected_command_table_hash),
        ("model_checkpoint_hash", proposal.model_checkpoint_hash, expected_checkpoint_hash),
    ):
        if actual != expected:
            violations.append(f"{name} mismatch: {actual[:12]} != {expected[:12]}")

    # W_CH1-only scope.
    if not contract.w_ch1_only:
        violations.append("contract w_ch1_only is False")
    if not proposal.research_authority.get("hardware_action_authorized", True):
        pass  # research authority stays shadow; no violation for staying un-authorized.

    # Hard bounds on the per-operation deltas.
    amp = np.asarray(proposal.delta_log_amp_w_ch1, dtype=np.float64)
    phase = np.asarray(proposal.delta_phase_w_ch1, dtype=np.float64)
    if amp.ndim != 2 or phase.ndim != 2 or amp.shape != phase.shape:
        violations.append("delta arrays must be 2D with matching shapes")
    else:
        if np.max(np.abs(amp)) > contract.max_single_amp_fraction + 1e-9:
            violations.append("single-step amplitude bound exceeded")
        if np.max(np.abs(phase)) > contract.max_single_phase_rad + 1e-9:
            violations.append("single-step phase bound exceeded")
        if contract.mean_over_cells_zero:
            amp_means = amp.mean(axis=1)
            phase_means = phase.mean(axis=1)
            if np.max(np.abs(amp_means)) > 1e-9 or np.max(np.abs(phase_means)) > 1e-9:
                violations.append("mean-over-cells zero constraint violated")

    if violations:
        raise ValueError("; ".join(violations))

    return {
        "proposal_id": proposal.proposal_id,
        "system_type": proposal.system_type,
        "bound": True,
        "violations": [],
        "action_stage": proposal.action_stage,
        "no_op_eligible": proposal.no_op_eligible,
        "hardware_action_authorized": False,  # the model never authorizes itself
        "operator_approval_required": True,
        "w_ch1_only": True,
    }
