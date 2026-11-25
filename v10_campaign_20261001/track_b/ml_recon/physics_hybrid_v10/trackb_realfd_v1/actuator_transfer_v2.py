"""Source-bound research adapter for the shared Track B RF mapping core.

This module owns the research-facing contract and proposal descriptor only.
All numerical support validation, regularized inversion, covariance handling,
trust-radius limiting, and ray-bound handling live in
``rf_mapping_core.RegisteredRFMappingCoreV1``. It deliberately has no
``system_control`` dependency and cannot materialize or execute a command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .contracts import RealStaticSupervisionV1, canonical_sha256, require_sha256
from .coordinate_abi import RFCommandDeltaV1, RecoverableLogicalActionV1, system_mode_count
from .rf_mapping_core import (
    CALIBRATION_STATUSES,
    RFCoordinateSpecV1,
    RFMappingSolveResultV1,
    RegisteredRFMappingCoreV1,
    RegisteredSingularSupportV1,
    solve_registered_rf_mapping,
)


@dataclass(frozen=True)
class SingularSupportContractV1:
    """Serialized source seam for the core's ordered effective SVD support.

    An empty sequence is deliberate: it encodes a verified rank-zero contract
    rather than inventing a placeholder singular direction.
    """

    supported_singular_indices: tuple[int, ...]
    relative_tolerance: float = 1.0e-10
    schema: str = field(default="TrackBSingularSupportContractV1", init=False)

    def __post_init__(self) -> None:
        indices = tuple(self.supported_singular_indices)
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("singular support indices must be nonnegative native integers")
        if tuple(sorted(set(indices))) != indices:
            raise ValueError("singular support indices must be unique and ascending")
        if not math.isfinite(float(self.relative_tolerance)) or not 0.0 < float(self.relative_tolerance) < 1.0:
            raise ValueError("singular support relative tolerance must lie in (0,1)")
        object.__setattr__(self, "supported_singular_indices", indices)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "supported_singular_indices": list(self.supported_singular_indices),
            "relative_tolerance": self.relative_tolerance,
            "effective_matrix": "A_NOMINAL_MATMUL_Q_RF",
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_registered_support(self) -> RegisteredSingularSupportV1:
        """Return the pure-core support representation without inferring axes."""
        return RegisteredSingularSupportV1(
            self.supported_singular_indices, self.relative_tolerance,
        )


@dataclass(frozen=True)
class ActuatorTransferContractV2:
    """Source-bound nominal transfer, never a real-static causal calibration.

    ``rf_coordinate_kinds`` is mandatory and positionally aligned with
    ``ordered_rf_coordinate_ids`` and transfer columns. It makes the command
    ABI explicit before the shared numerical core is invoked.
    """

    system_type: str
    hardware_path_id: str
    operating_point_id: str
    rf_command_basis_sha: str
    logical_coordinate_layout_sha: str
    rf_coordinate_layout_sha: str
    nominal_transfer_matrix: tuple[tuple[float, ...], ...]
    transfer_covariance: tuple[tuple[float, ...], ...]
    singular_support_contract: SingularSupportContractV1
    supported_singular_values: tuple[float, ...]
    recoverable_logical_basis: tuple[tuple[float, ...], ...]
    recoverable_command_basis: tuple[tuple[float, ...], ...]
    basis_orientation: str
    trust_radius: float
    nonlinearity_index: float | None
    calibration_age: float | None
    calibration_source: str
    calibration_status: str
    validity_scope: str
    current_control_snapshot_sha: str
    bounds_policy_sha: str
    heldout_calibration_pass: bool
    ordered_rf_coordinate_ids: tuple[str, ...]
    rf_coordinate_kinds: tuple[str, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    ridge_lambda: float = 1.0e-8
    covariance_penalty: float = 0.0
    causal_evidence_kind: str | None = None
    causal_evidence_sha: str | None = None
    hardware_validity: bool = False
    hardware_action_authorized: bool = False
    schema: str = field(default="ActuatorTransferContractV2", init=False)

    def __post_init__(self) -> None:
        logical_dimension = 2 * system_mode_count(self.system_type) - 2
        if self.basis_orientation != "LOGICAL_DQ_BY_R__RF_DU_BY_R_V1":
            raise ValueError("basis orientation must explicitly distinguish logical and RF support axes")
        if type(self.hardware_path_id) is not str or not self.hardware_path_id:
            raise ValueError("actuator contract requires a source-bound hardware path identifier")
        if type(self.operating_point_id) is not str or not self.operating_point_id:
            raise ValueError("actuator contract requires a source-bound operating point identifier")
        if type(self.calibration_source) is not str or not self.calibration_source:
            raise ValueError("actuator contract requires an explicit calibration source")
        if type(self.validity_scope) is not str or not self.validity_scope:
            raise ValueError("actuator contract requires an explicit validity scope")
        for value, name in (
            (self.nonlinearity_index, "nonlinearity_index"),
            (self.calibration_age, "calibration_age"),
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when available")
        for name in (
            "rf_command_basis_sha", "logical_coordinate_layout_sha",
            "rf_coordinate_layout_sha", "current_control_snapshot_sha",
            "bounds_policy_sha",
        ):
            require_sha256(getattr(self, name), name)
        if self.calibration_status not in CALIBRATION_STATUSES:
            raise ValueError("unknown actuator calibration status")
        causal_statuses = {
            "SIGNED_PROBE_BOUND": "SIGNED_PROBE",
            "TRANSITION_BOUND": "TRANSITION",
        }
        if self.calibration_status in causal_statuses:
            required_kind = causal_statuses[self.calibration_status]
            if (
                self.validity_scope != required_kind
                or self.calibration_source != required_kind
                or self.causal_evidence_kind != required_kind
            ):
                raise ValueError("causal actuator status requires matching evidence scope")
            require_sha256(self.causal_evidence_sha, "causal_evidence_sha")
        elif self.causal_evidence_kind is not None or self.causal_evidence_sha is not None:
            raise ValueError("nominal/static actuator contracts cannot carry causal evidence")
        if type(self.heldout_calibration_pass) is not bool:
            raise ValueError("heldout_calibration_pass must be boolean")
        if self.hardware_validity is not False or self.hardware_action_authorized is not False:
            raise ValueError("Track B actuator contracts cannot self-assert hardware validity/authority")

        core = self.to_mapping_core(logical_dimension=logical_dimension)
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
        object.__setattr__(self, "supported_singular_values", core.supported_singular_values)
        object.__setattr__(self, "lower_bounds", core.lower_bounds)
        object.__setattr__(self, "upper_bounds", core.upper_bounds)

    @classmethod
    def static_prior_only(
        cls, *, supervision: RealStaticSupervisionV1, **kwargs: object,
    ) -> "ActuatorTransferContractV2":
        supervision.validate()
        if kwargs.get("calibration_status", "STATIC_PRIOR_ONLY") != "STATIC_PRIOR_ONLY":
            raise ValueError("real_static can construct only STATIC_PRIOR_ONLY actuator contracts")
        if kwargs.get("validity_scope", "REAL_STATIC") != "REAL_STATIC":
            raise ValueError("static prior contract must retain REAL_STATIC scope")
        return cls(calibration_status="STATIC_PRIOR_ONLY", validity_scope="REAL_STATIC", **kwargs)

    def bind_real_static_jacobian(self, supervision: RealStaticSupervisionV1) -> None:
        supervision.validate()
        raise ValueError("real_static rows cannot bind an actuator Jacobian or upgrade calibration status")

    @property
    def logical_dimension(self) -> int:
        return 2 * system_mode_count(self.system_type) - 2

    @property
    def rf_dimension(self) -> int:
        return len(self.ordered_rf_coordinate_ids)

    def to_mapping_core(
        self, *, logical_dimension: int | None = None,
    ) -> RegisteredRFMappingCoreV1:
        """Build the sole numerical authority from this source-bound contract."""
        expected_dimension = self.logical_dimension if logical_dimension is None else logical_dimension
        coordinate_ids = tuple(self.ordered_rf_coordinate_ids)
        coordinate_kinds = tuple(self.rf_coordinate_kinds)
        if len(coordinate_ids) != len(coordinate_kinds):
            raise ValueError("RF coordinate IDs and kinds must have matching explicit order")
        return RegisteredRFMappingCoreV1(
            system_type=self.system_type,
            logical_dimension=expected_dimension,
            rf_coordinates=tuple(
                RFCoordinateSpecV1(str(identifier), str(kind))
                for identifier, kind in zip(coordinate_ids, coordinate_kinds, strict=True)
            ),
            nominal_transfer_matrix=tuple(tuple(row) for row in self.nominal_transfer_matrix),
            transfer_covariance=tuple(tuple(row) for row in self.transfer_covariance),
            recoverable_logical_basis=tuple(tuple(row) for row in self.recoverable_logical_basis),
            recoverable_command_basis=tuple(tuple(row) for row in self.recoverable_command_basis),
            singular_support=self.singular_support_contract.to_registered_support(),
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

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        """Canonical seam payload with explicit logical/RF coordinate axes."""
        return {
            "schema": self.schema,
            "system_type": self.system_type,
            "hardware_path_id": self.hardware_path_id,
            "operating_point_id": self.operating_point_id,
            "rf_command_basis_sha": self.rf_command_basis_sha,
            "logical_coordinate_layout_sha": self.logical_coordinate_layout_sha,
            "rf_coordinate_layout_sha": self.rf_coordinate_layout_sha,
            "nominal_transfer_matrix": [list(row) for row in self.nominal_transfer_matrix],
            "transfer_covariance": [list(row) for row in self.transfer_covariance],
            "singular_support_contract": self.singular_support_contract.to_dict(),
            "singular_support_contract_sha": self.singular_support_contract.sha256(),
            "supported_singular_values": list(self.supported_singular_values),
            "recoverable_logical_basis": [list(row) for row in self.recoverable_logical_basis],
            "recoverable_command_basis": [list(row) for row in self.recoverable_command_basis],
            "basis_orientation": self.basis_orientation,
            "trust_radius": self.trust_radius,
            "ridge_lambda": self.ridge_lambda,
            "covariance_penalty": self.covariance_penalty,
            "nonlinearity_index": self.nonlinearity_index,
            "calibration_age": self.calibration_age,
            "calibration_source": self.calibration_source,
            "calibration_status": self.calibration_status,
            "validity_scope": self.validity_scope,
            "current_control_snapshot_sha": self.current_control_snapshot_sha,
            "bounds_policy_sha": self.bounds_policy_sha,
            "heldout_calibration_pass": self.heldout_calibration_pass,
            "ordered_rf_coordinate_ids": list(self.ordered_rf_coordinate_ids),
            "rf_coordinate_kinds": list(self.rf_coordinate_kinds),
            "lower_bounds": list(self.lower_bounds),
            "upper_bounds": list(self.upper_bounds),
            "causal_evidence_kind": self.causal_evidence_kind,
            "causal_evidence_sha": self.causal_evidence_sha,
            "hardware_validity": False,
            "hardware_action_authorized": False,
        }

    def to_json_dict(self) -> dict[str, object]:
        return self.to_dict()


@dataclass(frozen=True)
class NominalRFInverseResultV1:
    """Research preview translated directly from the shared core result."""

    rf_delta: RFCommandDeltaV1
    command_delta: tuple[float, ...]
    materialized_logical_action: tuple[float, ...]
    support_leakage_norm: float
    inverse_residual_norm: float
    requested_to_materialized_logical_error: float
    direction_cosine: float
    ray_scale: float
    status: str
    accepted: bool
    no_op: bool
    preview_only: bool = True
    hardware_action_authorized: bool = False

    def __post_init__(self) -> None:
        if self.preview_only is not True or self.hardware_action_authorized is not False:
            raise ValueError("nominal RF inverse remains preview-only")
        if not all(math.isfinite(float(value)) for value in (
            *self.command_delta,
            *self.materialized_logical_action,
            self.support_leakage_norm,
            self.inverse_residual_norm,
            self.requested_to_materialized_logical_error,
            self.direction_cosine,
            self.ray_scale,
        )):
            raise ValueError("inverse diagnostics must be finite")


def solve_registered_nominal_inverse(
    contract: ActuatorTransferContractV2, action: RecoverableLogicalActionV1,
) -> NominalRFInverseResultV1:
    """Preview an action through the one registered pure mapping authority."""
    if action.ideal_action.system_type != contract.system_type:
        raise ValueError("logical action and actuator contract system types disagree")
    if action.ideal_action.coordinate_layout_sha != contract.logical_coordinate_layout_sha:
        raise ValueError("actuator contract logical ABI SHA mismatch")
    core_result = solve_registered_rf_mapping(
        contract.to_mapping_core(), action.grouped_coordinates,
    )
    return _from_core_result(contract, action, core_result)


def _from_core_result(
    contract: ActuatorTransferContractV2,
    action: RecoverableLogicalActionV1,
    core_result: RFMappingSolveResultV1,
) -> NominalRFInverseResultV1:
    rf_delta = RFCommandDeltaV1(
        system_type=contract.system_type,
        ordered_tone_ids=contract.ordered_rf_coordinate_ids,
        amplitude_delta=core_result.amplitude_delta,
        phase_delta_rad=core_result.phase_delta_rad,
        other_delta=core_result.other_delta,
        rf_coordinate_layout_sha=contract.rf_coordinate_layout_sha,
        source_logical_action_sha=action.sha256(),
    )
    return NominalRFInverseResultV1(
        rf_delta=rf_delta,
        command_delta=core_result.command_delta,
        materialized_logical_action=core_result.materialized_logical_action,
        support_leakage_norm=core_result.support_leakage_norm,
        inverse_residual_norm=core_result.inverse_residual_norm,
        requested_to_materialized_logical_error=(
            core_result.requested_to_materialized_logical_error
        ),
        direction_cosine=core_result.direction_cosine,
        ray_scale=core_result.ray_scale,
        status=core_result.status,
        accepted=core_result.accepted,
        no_op=core_result.no_op,
    )


__all__ = [
    "ActuatorTransferContractV2", "CALIBRATION_STATUSES",
    "NominalRFInverseResultV1", "SingularSupportContractV1",
    "solve_registered_nominal_inverse",
]
