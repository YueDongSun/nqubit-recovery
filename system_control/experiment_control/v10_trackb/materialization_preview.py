"""Dry-run Track B materialization preview using the existing bounded projector.

The existing feedback action materializer owns amplitude/phase bounds and
engineering diagnostics. This adapter supplies only a proposal-derived action
and returns diagnostics; it does not ask the sequence service to write files or
bind an actual loaded waveform.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from system_control.experiment_control.feedback_action_materializer import FeedbackActionMaterializer
from system_control.experiment_control.models import ControlStateSpecV1, TargetStateSpecV1

from .coordinate_adapter import (
    coordinate_adapter_sha,
    materializer_to_trackb_coordinates,
    trackb_to_materializer_action_payload,
)
from .proposal_reader import (
    ProposalBindingsV1,
    ProposalValidationError,
    ResearchProposalDocumentV1,
    canonical_json_bytes,
    sha256_bytes,
)
from .rf_inverse import NominalRFInverseContractV1, RFInverseResultV1, solve_nominal_rf_inverse
from .support_audit import RecoverableSupportAuditV1, audit_recoverable_support


@dataclass(frozen=True)
class RFMaterializationPreviewV1:
    """A bounded, non-uploaded candidate preview with support evidence."""

    proposal_sha: str
    action_sha: str
    coordinate_adapter_sha: str
    requested_action_hash: str
    projected_action_hash: str
    actual_action_hash: str
    control_hash: str
    system_type: str
    ray_scale: float
    requested_logical_action: Tuple[float, ...]
    inverse_result: RFInverseResultV1
    projected_logical_action: Tuple[float, ...]
    support_audit: RecoverableSupportAuditV1
    materializer_accepted: bool
    accepted: bool
    preview_only: bool
    hardware_action_authorized: bool
    actual_loaded_action: None
    reasons: Tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "proposal_sha", "action_sha", "coordinate_adapter_sha",
            "requested_action_hash", "projected_action_hash",
            "actual_action_hash", "control_hash",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ProposalValidationError(
                    f"{field_name} must be a lowercase SHA-256 digest"
                )
        if (
            not math.isfinite(float(self.ray_scale))
            or not 0.0 <= float(self.ray_scale) <= 1.0
        ):
            raise ProposalValidationError("preview ray scale must be finite in [0,1]")
        if self.preview_only is not True or self.hardware_action_authorized is not False:
            raise ProposalValidationError("Track B previews must remain dry-run-only")
        if self.actual_loaded_action is not None:
            raise ProposalValidationError("a pre-upload preview cannot assert an actual loaded action")

    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_json_dict()))

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": "RFMaterializationPreviewV1",
            "proposal_sha": self.proposal_sha,
            "action_sha": self.action_sha,
            "coordinate_adapter_sha": self.coordinate_adapter_sha,
            "requested_action_hash": self.requested_action_hash,
            "projected_action_hash": self.projected_action_hash,
            "actual_action_hash": self.actual_action_hash,
            "control_hash": self.control_hash,
            "system_type": self.system_type,
            "ray_scale": self.ray_scale,
            "requested_logical_action": list(self.requested_logical_action),
            "inverse_result": self.inverse_result.to_json_dict(),
            "projected_logical_action": list(self.projected_logical_action),
            "support_audit": self.support_audit.to_json_dict(),
            "materializer_accepted": self.materializer_accepted,
            "accepted": self.accepted,
            "preview_only": True,
            "hardware_action_authorized": False,
            "actual_loaded_action": None,
            "reasons": list(self.reasons),
        }


def preview_rf_materialization(
    proposal: ResearchProposalDocumentV1,
    bindings: ProposalBindingsV1,
    inverse_contract: NominalRFInverseContractV1,
    *,
    target_state: TargetStateSpecV1,
    control_state: ControlStateSpecV1,
    waveform_diagnostics: Optional[Mapping[str, Any]] = None,
    fourier_report: Optional[Mapping[str, Any]] = None,
    leakage_tolerance: float = 1e-8,
    ray_shrink_factor: float = 0.5,
    maximum_ray_shrinks: int = 6,
    maximum_requested_to_materialized_error: float = 1.0e-8,
    minimum_direction_cosine: float = 0.999999,
) -> RFMaterializationPreviewV1:
    """Preview a bound action, shrinking only along its original ray."""

    proposal.assert_bindings(bindings)
    proposal.assert_not_expired()
    if proposal.system_type != inverse_contract.system_type:
        raise ProposalValidationError("proposal and inverse contract system_type mismatch")
    if target_state.system_type != proposal.system_type or control_state.system_type != proposal.system_type:
        raise ProposalValidationError("target/control system_type does not match proposal")
    if target_state.target_hash != bindings.desired_target_sha:
        raise ProposalValidationError("target state does not match the bound desired target")
    if inverse_contract.current_control_snapshot_sha != bindings.current_control_snapshot_sha:
        raise ProposalValidationError("inverse contract current snapshot does not match proposal")
    if inverse_contract.bounds_policy_sha != bindings.bounds_policy_sha:
        raise ProposalValidationError("inverse contract bounds policy does not match proposal")
    if inverse_contract.logical_coordinate_layout_sha != bindings.logical_coordinate_layout_sha:
        raise ProposalValidationError("inverse contract logical coordinate layout mismatch")
    if inverse_contract.rf_coordinate_layout_sha != bindings.rf_coordinate_layout_sha:
        raise ProposalValidationError("inverse contract RF coordinate layout mismatch")
    if inverse_contract.rf_command_basis_sha != bindings.rf_command_basis_sha:
        raise ProposalValidationError("inverse contract RF command basis mismatch")
    if inverse_contract.hardware_path_id != bindings.hardware_path_id:
        raise ProposalValidationError("inverse contract hardware path mismatch")
    if inverse_contract.operating_point_id != bindings.operating_point_id:
        raise ProposalValidationError("inverse contract operating point mismatch")
    if not 0.0 < ray_shrink_factor < 1.0:
        raise ProposalValidationError("ray_shrink_factor must lie strictly between zero and one")
    if maximum_ray_shrinks < 0:
        raise ProposalValidationError("maximum_ray_shrinks must be non-negative")
    if (
        maximum_requested_to_materialized_error < 0.0
        or not np.isfinite(maximum_requested_to_materialized_error)
    ):
        raise ProposalValidationError(
            "maximum_requested_to_materialized_error must be finite and non-negative"
        )
    if not -1.0 <= minimum_direction_cosine <= 1.0:
        raise ProposalValidationError("minimum_direction_cosine must lie in [-1,1]")
    requested = np.asarray(proposal.logical_action_vector, dtype=np.float64)
    materializer = FeedbackActionMaterializer()
    latest: RFMaterializationPreviewV1 | None = None
    for attempt in range(maximum_ray_shrinks + 1):
        ray_scale = ray_shrink_factor ** attempt
        scaled = requested * ray_scale
        inverse = solve_nominal_rf_inverse(inverse_contract, scaled)
        action_payload = trackb_to_materializer_action_payload(
            proposal.system_type, inverse.materialized_logical_action,
        )
        result = materializer.materialize(
            target_state, control_state, action_payload,
            waveform_diagnostics=waveform_diagnostics, fourier_report=fourier_report,
        )
        actual_preview = _logical_vector_from_projected_action(
            proposal.system_type, result.projected_action,
        )
        audit = audit_recoverable_support(
            inverse.recoverable_logical_action,
            actual_preview,
            inverse_contract.recoverable_logical_basis,
            leakage_tolerance=leakage_tolerance,
            maximum_requested_to_materialized_error=(
                maximum_requested_to_materialized_error
            ),
            minimum_direction_cosine=minimum_direction_cosine,
        )
        reasons = list(inverse.reasons)
        reasons.extend(result.reasons)
        reasons.extend(audit.reasons)
        if audit.requested_to_materialized_logical_error > maximum_requested_to_materialized_error:
            reasons.append("requested_to_materialized_logical_error_exceeded")
        if audit.direction_cosine < minimum_direction_cosine:
            reasons.append("materialized_direction_cosine_below_minimum")
        accepted = bool(
            inverse.accepted
            and result.accepted
            and audit.passed
            and audit.requested_to_materialized_logical_error
            <= maximum_requested_to_materialized_error
            and audit.direction_cosine >= minimum_direction_cosine
        )
        latest = RFMaterializationPreviewV1(
            proposal_sha=proposal.proposal_sha,
            action_sha=proposal.action_sha,
            coordinate_adapter_sha=coordinate_adapter_sha(proposal.system_type),
            requested_action_hash=result.requested_action_hash,
            projected_action_hash=result.projected_action_hash,
            actual_action_hash=result.actual_action_hash,
            control_hash=result.actual_control_state.control_hash,
            system_type=proposal.system_type,
            ray_scale=float(ray_scale),
            requested_logical_action=tuple(float(value) for value in scaled),
            inverse_result=inverse,
            projected_logical_action=tuple(float(value) for value in actual_preview),
            support_audit=audit,
            materializer_accepted=bool(result.accepted),
            accepted=accepted,
            preview_only=True,
            hardware_action_authorized=False,
            actual_loaded_action=None,
            reasons=tuple(dict.fromkeys(reasons)),
        )
        if accepted or not inverse.accepted:
            return latest
    if latest is None:
        raise RuntimeError("materialization preview did not execute")
    return RFMaterializationPreviewV1(
        proposal_sha=latest.proposal_sha,
        action_sha=latest.action_sha,
        coordinate_adapter_sha=latest.coordinate_adapter_sha,
        requested_action_hash=latest.requested_action_hash,
        projected_action_hash=latest.projected_action_hash,
        actual_action_hash=latest.actual_action_hash,
        control_hash=latest.control_hash,
        system_type=latest.system_type,
        ray_scale=latest.ray_scale,
        requested_logical_action=latest.requested_logical_action,
        inverse_result=latest.inverse_result,
        projected_logical_action=latest.projected_logical_action,
        support_audit=latest.support_audit,
        materializer_accepted=latest.materializer_accepted,
        accepted=False,
        preview_only=True,
        hardware_action_authorized=False,
        actual_loaded_action=None,
        reasons=tuple(dict.fromkeys((*latest.reasons, "ray_shrink_exhausted"))),
    )


def _logical_vector_from_projected_action(
    system_type: str, action: Mapping[str, Any],
) -> np.ndarray:
    try:
        values = list(action["amplitude_coordinates"]) + list(action["phase_coordinates"])
    except (KeyError, TypeError) as exc:
        raise ProposalValidationError("existing materializer returned malformed logical action") from exc
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ProposalValidationError("existing materializer returned non-finite logical action")
    return np.asarray(materializer_to_trackb_coordinates(system_type, vector), dtype=np.float64)


__all__ = ["RFMaterializationPreviewV1", "preview_rf_materialization"]
