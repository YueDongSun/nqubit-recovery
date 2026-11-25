"""System-control dry-run adapter for the shared Track B RF mapping core.

This module revalidates a source-bound transfer contract at the system
boundary, then delegates every numerical mapping decision to the pure research
module ``RegisteredRFMappingCoreV1``. It neither materializes waveform files
nor contacts an instrument.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ml_recon.physics_hybrid_v10.trackb_realfd_v1.rf_mapping_core import (
    CALIBRATION_STATUSES,
    RFCoordinateSpecV1,
    RegisteredRFMappingCoreV1,
    RegisteredSingularSupportV1,
    solve_registered_rf_mapping,
)

from .proposal_reader import (
    ProposalValidationError,
    canonical_json_bytes,
    sha256_bytes,
    validate_sha256,
)


_DRY_RUN_CALIBRATION_STATUSES = {
    "NOMINAL_FILE_BACKED", "STATIC_PRIOR_ONLY", "NOT_AVAILABLE",
}


def _validate_source_bound_support_contract(
    value: Mapping[str, Any], declared_sha: str,
) -> tuple[Tuple[int, ...], float]:
    """Validate the exact support bytes before accepting their SVD axes."""
    if not isinstance(value, Mapping):
        raise ProposalValidationError("singular_support_contract must be a mapping")
    expected_keys = {
        "schema", "effective_matrix", "supported_singular_indices",
        "relative_tolerance",
    }
    if set(value) != expected_keys:
        raise ProposalValidationError("singular_support_contract fields are not exact")
    if (
        value.get("schema") != "TrackBSingularSupportContractV1"
        or value.get("effective_matrix") != "A_NOMINAL_MATMUL_Q_RF"
    ):
        raise ProposalValidationError("singular support contract has an unregistered effective matrix")
    actual_sha = sha256_bytes(canonical_json_bytes(dict(value)))
    if validate_sha256(declared_sha, "singular_support_contract_sha") != actual_sha:
        raise ProposalValidationError("singular support contract SHA does not bind canonical support")
    raw_indices = value.get("supported_singular_indices")
    if not isinstance(raw_indices, (list, tuple)):
        raise ProposalValidationError("singular support contract lacks supported_singular_indices")
    indices = tuple(raw_indices)
    if any(type(index) is not int or index < 0 for index in indices):
        raise ProposalValidationError("singular support indices must be nonnegative native integers")
    if tuple(sorted(set(indices))) != indices:
        raise ProposalValidationError("singular support indices must be unique and ascending")
    try:
        tolerance = float(value.get("relative_tolerance"))
    except (TypeError, ValueError) as error:
        raise ProposalValidationError("singular support relative tolerance is invalid") from error
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ProposalValidationError("singular support relative tolerance is invalid")
    return indices, tolerance


@dataclass(frozen=True)
class NominalRFInverseContractV1:
    """Registered nominal actuator geometry usable only for a dry-run preview.

    The RF coordinate ID/kind pairs are mandatory and remain aligned to the
    transfer columns, covariance axes, bounds, and emitted command delta.
    Empty ``[d, 0]`` support bases plus empty singular support form a legal
    rank-zero/no-support contract.
    """

    system_type: str
    hardware_path_id: str
    operating_point_id: str
    rf_command_basis_sha: str
    logical_coordinate_layout_sha: str
    rf_coordinate_layout_sha: str
    current_control_snapshot_sha: str
    bounds_policy_sha: str
    nominal_transfer_matrix: Tuple[Tuple[float, ...], ...]
    transfer_covariance: Tuple[Tuple[float, ...], ...]
    singular_support_contract: Mapping[str, Any]
    singular_support_contract_sha: str
    supported_singular_indices: Tuple[int, ...]
    supported_singular_values: Tuple[float, ...]
    recoverable_logical_basis: Tuple[Tuple[float, ...], ...]
    recoverable_command_basis: Tuple[Tuple[float, ...], ...]
    ordered_rf_coordinate_ids: Tuple[str, ...]
    rf_coordinate_kinds: Tuple[str, ...]
    lower_bounds: Tuple[float, ...]
    upper_bounds: Tuple[float, ...]
    trust_radius: float
    basis_orientation: str = "LOGICAL_DQ_BY_R__RF_DU_BY_R_V1"
    singular_support_relative_tolerance: float = 1.0e-10
    ridge_lambda: float = 1.0e-8
    covariance_penalty: float = 0.0
    nonlinearity_index: float = 0.0
    calibration_age: str = "NOT_AVAILABLE"
    calibration_source: str = "NOT_AVAILABLE"
    calibration_status: str = "NOMINAL_FILE_BACKED"
    validity_scope: str = "REGISTERED_CANDIDATE_SET_AND_BOUND_RF_MAPPING_ONLY"
    heldout_calibration_pass: bool = False
    hardware_validity: str = "NOT_AVAILABLE"
    hardware_action_authorized: bool = False

    def __post_init__(self) -> None:
        if self.system_type not in {"3qubit", "4qubit"}:
            raise ProposalValidationError("inverse contract system_type is invalid")
        if not str(self.hardware_path_id).strip() or not str(self.operating_point_id).strip():
            raise ProposalValidationError("inverse contract requires hardware_path_id and operating_point_id")
        for field_name in (
            "rf_command_basis_sha", "logical_coordinate_layout_sha",
            "rf_coordinate_layout_sha", "current_control_snapshot_sha",
            "bounds_policy_sha", "singular_support_contract_sha",
        ):
            object.__setattr__(
                self, field_name, validate_sha256(getattr(self, field_name), field_name),
            )
        if self.basis_orientation != "LOGICAL_DQ_BY_R__RF_DU_BY_R_V1":
            raise ProposalValidationError("recoverable basis orientation is not the registered ABI")
        support_indices, support_tolerance = _validate_source_bound_support_contract(
            self.singular_support_contract, self.singular_support_contract_sha,
        )
        if tuple(self.supported_singular_indices) != support_indices:
            raise ProposalValidationError(
                "singular support indices must match the source-bound support contract",
            )
        if float(self.singular_support_relative_tolerance) != support_tolerance:
            raise ProposalValidationError(
                "singular support tolerance must match the source-bound support contract",
            )
        if self.calibration_status not in CALIBRATION_STATUSES:
            raise ProposalValidationError("unknown calibration_status")
        if self.calibration_status not in _DRY_RUN_CALIBRATION_STATUSES:
            raise ProposalValidationError(
                "causal actuator calibration is unavailable to this dry-run bridge",
            )
        if not math.isfinite(float(self.nonlinearity_index)):
            raise ProposalValidationError("nonlinearity_index must be finite")
        if self.heldout_calibration_pass is not False or self.hardware_validity != "NOT_AVAILABLE":
            raise ProposalValidationError(
                "dry-run nominal inverse cannot assert hardware calibration validity",
            )
        if self.hardware_action_authorized is not False:
            raise ProposalValidationError("nominal inverse cannot grant hardware authority")

        try:
            core = self.to_mapping_core()
        except (TypeError, ValueError) as error:
            raise ProposalValidationError(str(error)) from error
        object.__setattr__(self, "nominal_transfer_matrix", core.nominal_transfer_matrix)
        object.__setattr__(self, "transfer_covariance", core.transfer_covariance)
        object.__setattr__(self, "recoverable_logical_basis", core.recoverable_logical_basis)
        object.__setattr__(self, "recoverable_command_basis", core.recoverable_command_basis)
        object.__setattr__(self, "ordered_rf_coordinate_ids", tuple(
            item.coordinate_id for item in core.rf_coordinates
        ))
        object.__setattr__(self, "rf_coordinate_kinds", tuple(
            item.kind for item in core.rf_coordinates
        ))
        object.__setattr__(self, "supported_singular_indices", tuple(
            core.singular_support.supported_singular_indices
        ))
        object.__setattr__(self, "supported_singular_values", core.supported_singular_values)
        object.__setattr__(self, "lower_bounds", core.lower_bounds)
        object.__setattr__(self, "upper_bounds", core.upper_bounds)

    @property
    def logical_dimension(self) -> int:
        return 4 if self.system_type == "3qubit" else 6

    @property
    def rf_dimension(self) -> int:
        return len(self.ordered_rf_coordinate_ids)

    def to_mapping_core(self) -> RegisteredRFMappingCoreV1:
        """Recreate and validate the pure numerical authority at use time."""
        support_indices, support_tolerance = _validate_source_bound_support_contract(
            self.singular_support_contract, self.singular_support_contract_sha,
        )
        if (
            tuple(self.supported_singular_indices) != support_indices
            or float(self.singular_support_relative_tolerance) != support_tolerance
        ):
            raise ProposalValidationError(
                "RF inverse support fields no longer match source-bound support bytes",
            )
        coordinate_ids = tuple(self.ordered_rf_coordinate_ids)
        coordinate_kinds = tuple(self.rf_coordinate_kinds)
        if len(coordinate_ids) != len(coordinate_kinds):
            raise ProposalValidationError("RF coordinate IDs and kinds must have matching explicit order")
        return RegisteredRFMappingCoreV1(
            system_type=self.system_type,
            logical_dimension=self.logical_dimension,
            rf_coordinates=tuple(
                RFCoordinateSpecV1(str(identifier), str(kind))
                for identifier, kind in zip(coordinate_ids, coordinate_kinds, strict=True)
            ),
            nominal_transfer_matrix=tuple(tuple(row) for row in self.nominal_transfer_matrix),
            transfer_covariance=tuple(tuple(row) for row in self.transfer_covariance),
            recoverable_logical_basis=tuple(tuple(row) for row in self.recoverable_logical_basis),
            recoverable_command_basis=tuple(tuple(row) for row in self.recoverable_command_basis),
            singular_support=RegisteredSingularSupportV1(
                support_indices,
                support_tolerance,
            ),
            supported_singular_values=tuple(self.supported_singular_values),
            lower_bounds=tuple(self.lower_bounds),
            upper_bounds=tuple(self.upper_bounds),
            trust_radius=self.trust_radius,
            rf_command_basis_sha=self.rf_command_basis_sha,
            logical_coordinate_layout_sha=self.logical_coordinate_layout_sha,
            rf_coordinate_layout_sha=self.rf_coordinate_layout_sha,
            current_control_snapshot_sha=self.current_control_snapshot_sha,
            bounds_policy_sha=self.bounds_policy_sha,
            calibration_status=self.calibration_status,
            ridge_lambda=self.ridge_lambda,
            covariance_penalty=self.covariance_penalty,
        )


@dataclass(frozen=True)
class RFInverseResultV1:
    """Pure core result rendered as a dry-run system-control diagnostic."""

    requested_logical_action: Tuple[float, ...]
    recoverable_logical_action: Tuple[float, ...]
    rf_delta: Tuple[float, ...]
    amplitude_delta: Tuple[float, ...]
    phase_delta_rad: Tuple[float, ...]
    other_delta: Tuple[float, ...]
    materialized_logical_action: Tuple[float, ...]
    residual_norm: float
    direction_cosine: float
    rf_delta_norm: float
    ray_scale: float
    support_indices: Tuple[int, ...]
    preview_only: bool
    hardware_action_authorized: bool
    accepted: bool
    no_op: bool
    reasons: Tuple[str, ...]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": "RFInverseResultV1",
            "requested_logical_action": list(self.requested_logical_action),
            "recoverable_logical_action": list(self.recoverable_logical_action),
            "rf_delta": list(self.rf_delta),
            "amplitude_delta": list(self.amplitude_delta),
            "phase_delta_rad": list(self.phase_delta_rad),
            "other_delta": list(self.other_delta),
            "materialized_logical_action": list(self.materialized_logical_action),
            "residual_norm": self.residual_norm,
            "direction_cosine": self.direction_cosine,
            "rf_delta_norm": self.rf_delta_norm,
            "ray_scale": self.ray_scale,
            "support_indices": list(self.support_indices),
            "preview_only": True,
            "hardware_action_authorized": False,
            "accepted": self.accepted,
            "no_op": self.no_op,
            "reasons": list(self.reasons),
        }


def solve_nominal_rf_inverse(
    contract: NominalRFInverseContractV1,
    logical_action: Sequence[float],
    *,
    logical_weight: Optional[Sequence[float]] = None,
) -> RFInverseResultV1:
    """Revalidate the contract and call the sole registered mapping core."""
    if logical_weight is not None:
        raise ProposalValidationError(
            "nonuniform logical weights require a separately source-bound weighted-SVD contract",
        )
    try:
        core = contract.to_mapping_core()
        core_result = solve_registered_rf_mapping(core, logical_action)
        requested = tuple(float(value) for value in logical_action)
    except (TypeError, ValueError) as error:
        raise ProposalValidationError(str(error)) from error
    recoverable = (
        requested
        if core_result.accepted
        else tuple(0.0 for _ in range(contract.logical_dimension))
    )
    return RFInverseResultV1(
        requested_logical_action=requested,
        recoverable_logical_action=recoverable,
        rf_delta=core_result.command_delta,
        amplitude_delta=core_result.amplitude_delta,
        phase_delta_rad=core_result.phase_delta_rad,
        other_delta=core_result.other_delta,
        materialized_logical_action=core_result.materialized_logical_action,
        residual_norm=core_result.requested_to_materialized_logical_error,
        direction_cosine=core_result.direction_cosine,
        rf_delta_norm=math.sqrt(sum(value * value for value in core_result.command_delta)),
        ray_scale=core_result.ray_scale,
        support_indices=tuple(contract.supported_singular_indices),
        preview_only=True,
        hardware_action_authorized=False,
        accepted=core_result.accepted,
        no_op=core_result.no_op,
        reasons=_reasons_for_core_status(core_result.status),
    )


def from_actuator_transfer_payload(payload: Mapping[str, Any]) -> NominalRFInverseContractV1:
    """Reread an exported research contract and validate it before core use.

    This adapter accepts only the registered schema and its canonical nested
    singular-support bytes. It does not import a model object or infer missing
    coordinate kinds.
    """

    if not isinstance(payload, Mapping) or payload.get("schema") != "ActuatorTransferContractV2":
        raise ProposalValidationError("unsupported actuator transfer contract schema")
    if payload.get("basis_orientation") != "LOGICAL_DQ_BY_R__RF_DU_BY_R_V1":
        raise ProposalValidationError("recoverable basis orientation is not the registered ABI")
    if payload.get("hardware_validity") is not False:
        raise ProposalValidationError("actuator transfer contract cannot self-assert hardware validity")
    if payload.get("hardware_action_authorized") is not False:
        raise ProposalValidationError("actuator transfer contract cannot grant hardware authority")
    calibration_status = str(payload.get("calibration_status", ""))
    if calibration_status not in _DRY_RUN_CALIBRATION_STATUSES:
        raise ProposalValidationError("causal actuator calibration is not available to this dry-run bridge")
    support_contract = payload.get("singular_support_contract")
    if not isinstance(support_contract, Mapping):
        raise ProposalValidationError("singular_support_contract must expose explicit registered indices")
    if (
        support_contract.get("schema") != "TrackBSingularSupportContractV1"
        or support_contract.get("effective_matrix") != "A_NOMINAL_MATMUL_Q_RF"
    ):
        raise ProposalValidationError("singular support contract has an unregistered effective matrix")
    try:
        relative_tolerance = float(support_contract.get("relative_tolerance"))
    except (TypeError, ValueError) as error:
        raise ProposalValidationError("singular support relative tolerance is invalid") from error
    if not math.isfinite(relative_tolerance) or not 0.0 < relative_tolerance < 1.0:
        raise ProposalValidationError("singular support relative tolerance is invalid")
    raw_indices = support_contract.get("supported_singular_indices")
    if not isinstance(raw_indices, (list, tuple)):
        raise ProposalValidationError("singular support contract lacks supported_singular_indices")
    support_contract_sha = sha256_bytes(canonical_json_bytes(support_contract))
    if payload.get("singular_support_contract_sha") != support_contract_sha:
        raise ProposalValidationError("singular support contract SHA does not bind canonical support")
    if "rf_coordinate_kinds" not in payload:
        raise ProposalValidationError("actuator transfer payload requires explicit rf_coordinate_kinds")
    raw_kinds = payload.get("rf_coordinate_kinds")
    if not isinstance(raw_kinds, (list, tuple)):
        raise ProposalValidationError("rf_coordinate_kinds must be an ordered sequence")
    for field_name in ("ridge_lambda", "covariance_penalty"):
        if field_name not in payload:
            raise ProposalValidationError(f"actuator transfer payload requires {field_name}")
    causal_evidence_sha = payload.get("causal_evidence_sha")
    if causal_evidence_sha is not None:
        validate_sha256(causal_evidence_sha, "causal_evidence_sha")
    if payload.get("causal_evidence_kind") is not None or causal_evidence_sha is not None:
        raise ProposalValidationError("nominal/static actuator contracts cannot carry causal evidence")
    try:
        return NominalRFInverseContractV1(
            system_type=str(payload.get("system_type", "")),
            hardware_path_id=str(payload.get("hardware_path_id", "")),
            operating_point_id=str(payload.get("operating_point_id", "")),
            rf_command_basis_sha=str(payload.get("rf_command_basis_sha", "")),
            logical_coordinate_layout_sha=str(payload.get("logical_coordinate_layout_sha", "")),
            rf_coordinate_layout_sha=str(payload.get("rf_coordinate_layout_sha", "")),
            current_control_snapshot_sha=str(payload.get("current_control_snapshot_sha", "")),
            bounds_policy_sha=str(payload.get("bounds_policy_sha", "")),
            nominal_transfer_matrix=tuple(tuple(row) for row in payload.get("nominal_transfer_matrix", ())),
            transfer_covariance=tuple(tuple(row) for row in payload.get("transfer_covariance", ())),
            singular_support_contract=dict(support_contract),
            singular_support_contract_sha=support_contract_sha,
            supported_singular_indices=tuple(raw_indices),
            supported_singular_values=tuple(payload.get("supported_singular_values", ())),
            recoverable_logical_basis=tuple(tuple(row) for row in payload.get("recoverable_logical_basis", ())),
            recoverable_command_basis=tuple(tuple(row) for row in payload.get("recoverable_command_basis", ())),
            ordered_rf_coordinate_ids=tuple(payload.get("ordered_rf_coordinate_ids", ())),
            rf_coordinate_kinds=tuple(raw_kinds),
            lower_bounds=tuple(payload.get("lower_bounds", ())),
            upper_bounds=tuple(payload.get("upper_bounds", ())),
            trust_radius=float(payload.get("trust_radius", 0.0)),
            basis_orientation=str(payload.get("basis_orientation", "")),
            singular_support_relative_tolerance=relative_tolerance,
            ridge_lambda=float(payload.get("ridge_lambda")),
            covariance_penalty=float(payload.get("covariance_penalty")),
            nonlinearity_index=(
                0.0
                if payload.get("nonlinearity_index") is None
                else float(payload.get("nonlinearity_index"))
            ),
            calibration_age=str(payload.get("calibration_age", "NOT_AVAILABLE")),
            calibration_source=str(payload.get("calibration_source", "NOT_AVAILABLE")),
            calibration_status=calibration_status,
            validity_scope=str(payload.get("validity_scope", "")),
            heldout_calibration_pass=bool(payload.get("heldout_calibration_pass", False)),
            hardware_validity=(
                "NOT_AVAILABLE"
                if payload.get("hardware_validity") is False
                else str(payload.get("hardware_validity", "NOT_AVAILABLE"))
            ),
            hardware_action_authorized=False,
        )
    except (TypeError, ValueError) as error:
        raise ProposalValidationError(str(error)) from error


def _reasons_for_core_status(status: str) -> Tuple[str, ...]:
    mapping = {
        "REGISTERED_RF_PREVIEW": (),
        "NO_RECOVERABLE_RF_SUPPORT": ("no_recoverable_rf_support",),
        "LOGICAL_ACTION_OUTSIDE_REGISTERED_SUPPORT": (
            "logical_action_outside_registered_support",
        ),
    }
    return mapping.get(status, ("unrecognized_registered_rf_core_status",))


__all__ = [
    "NominalRFInverseContractV1", "RFInverseResultV1",
    "from_actuator_transfer_payload", "solve_nominal_rf_inverse",
]
