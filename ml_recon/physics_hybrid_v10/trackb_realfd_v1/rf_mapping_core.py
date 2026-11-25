"""Single pure numerical authority for registered Track B logical-to-RF maps.

This module deliberately contains no filesystem, model-loading, sequence, AWG,
or system-control dependency.  Both research-side counterfactual code and the
system-control dry-run adapter must use this same registered SVD/bounds core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from .contracts import (
    canonical_json_bytes,
    canonical_sha256,
    parse_canonical_json_object,
    require_exact_canonical_json_round_trip,
    require_sha256,
)
from .rf_to_awg_mapping import (
    RegisteredRFToAWGMappingV1,
    registered_rf_to_awg_recipe_sha,
)


RF_COORDINATE_KINDS = frozenset({"AMPLITUDE", "PHASE", "OTHER"})
CALIBRATION_STATUSES = frozenset({
    "NOMINAL_FILE_BACKED", "STATIC_PRIOR_ONLY", "SIGNED_PROBE_BOUND",
    "TRANSITION_BOUND", "NOT_AVAILABLE",
})


def _matrix(
    values: Sequence[Sequence[object]], rows: int, columns: int | None, name: str,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric matrix") from error
    if array.ndim != 2 or array.shape[0] != rows or (
        columns is not None and array.shape[1] != columns
    ):
        raise ValueError(f"{name} shape is not registered")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _orthonormal_columns(values: np.ndarray, name: str) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"{name} must be rank two")
    if values.shape[1] == 0:
        return np.zeros((values.shape[0], 0), dtype=np.float64)
    left, singular, _right = np.linalg.svd(values, full_matrices=False)
    tolerance = max(values.shape) * np.finfo(np.float64).eps * max(
        float(singular[0]), 1.0,
    )
    rank = int(np.sum(singular > tolerance))
    if rank != values.shape[1]:
        raise ValueError(f"{name} must have full registered column rank")
    return left[:, :rank]


def _ray_scale(command: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    scale = 1.0
    for value, minimum, maximum in zip(command, lower, upper, strict=True):
        if value > 0.0:
            scale = min(scale, float(maximum / value))
        elif value < 0.0:
            scale = min(scale, float(minimum / value))
    return max(0.0, min(1.0, scale))


@dataclass(frozen=True)
class RFCoordinateSpecV1:
    """One registered RF coordinate and its physical command kind."""

    coordinate_id: str
    kind: str

    def __post_init__(self) -> None:
        if type(self.coordinate_id) is not str or not self.coordinate_id:
            raise ValueError("RF coordinate_id must be nonempty")
        if self.kind not in RF_COORDINATE_KINDS:
            raise ValueError("RF coordinate kind is not registered")

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RFCoordinateSpecV1":
        return cls(coordinate_id=str(payload["coordinate_id"]), kind=str(payload["kind"]))


@dataclass(frozen=True)
class RegisteredSingularSupportV1:
    """Ordered effective-map SVD support; empty support is an explicit no-op."""

    supported_singular_indices: tuple[int, ...]
    relative_tolerance: float = 1.0e-10
    schema: str = field(default="RegisteredSingularSupportV1", init=False)

    def __post_init__(self) -> None:
        if any(type(index) is not int or index < 0 for index in self.supported_singular_indices):
            raise ValueError("singular support indices must be nonnegative native integers")
        if tuple(sorted(set(self.supported_singular_indices))) != self.supported_singular_indices:
            raise ValueError("singular support indices must be unique and sorted")
        if not math.isfinite(float(self.relative_tolerance)) or not 0.0 < float(self.relative_tolerance) < 1.0:
            raise ValueError("relative singular tolerance must lie in (0,1)")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "effective_matrix": "A_NOMINAL_MATMUL_Q_RF",
            "supported_singular_indices": list(self.supported_singular_indices),
            "relative_tolerance": self.relative_tolerance,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RegisteredSingularSupportV1":
        return cls(
            supported_singular_indices=tuple(int(value) for value in payload["supported_singular_indices"]),
            relative_tolerance=float(payload["relative_tolerance"]),
        )


@dataclass(frozen=True)
class RegisteredRFMappingCoreV1:
    """File-identity-bound, pure registered mapping and constrained inverse.

    Empty logical/RF bases plus an empty singular support are a valid explicit
    no-support contract.  They return a no-op rather than requiring callers to
    invent a rank-one placeholder or invoke an unconstrained pseudoinverse.
    """

    system_type: str
    logical_dimension: int
    rf_coordinates: tuple[RFCoordinateSpecV1, ...]
    nominal_transfer_matrix: tuple[tuple[float, ...], ...]
    transfer_covariance: tuple[tuple[float, ...], ...]
    recoverable_logical_basis: tuple[tuple[float, ...], ...]
    recoverable_command_basis: tuple[tuple[float, ...], ...]
    singular_support: RegisteredSingularSupportV1
    supported_singular_values: tuple[float, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    trust_radius: float
    rf_command_basis_sha: str
    logical_coordinate_layout_sha: str
    rf_coordinate_layout_sha: str
    current_control_snapshot_sha: str
    bounds_policy_sha: str
    calibration_status: str
    ridge_lambda: float = 1.0e-8
    covariance_penalty: float = 0.0
    current_amplitude: tuple[float, ...] = ()
    current_phase_rad: tuple[float, ...] = ()
    phase_wrap_interval: tuple[float, float] = (-math.pi, math.pi)
    operation_registry_sha: str | None = None
    command_table_sha: str | None = None
    hardware_path_id: str | None = None
    operating_point_id: str | None = None
    rf_to_awg_mapping: RegisteredRFToAWGMappingV1 | None = None
    schema: str = field(default="RegisteredRFMappingCoreV1", init=False)

    def __post_init__(self) -> None:
        if self.system_type not in {"3qubit", "4qubit"}:
            raise ValueError("RF mapping system type is unsupported")
        expected_dimension = 4 if self.system_type == "3qubit" else 6
        if self.logical_dimension != expected_dimension:
            raise ValueError("RF mapping logical dimension disagrees with system type")
        if not self.rf_coordinates or len({item.coordinate_id for item in self.rf_coordinates}) != len(self.rf_coordinates):
            raise ValueError("RF coordinates must be explicit and uniquely ordered")
        rf_dimension = len(self.rf_coordinates)
        transfer = _matrix(
            self.nominal_transfer_matrix, self.logical_dimension, rf_dimension,
            "nominal_transfer_matrix",
        )
        covariance = _matrix(
            self.transfer_covariance, rf_dimension, rf_dimension,
            "transfer_covariance",
        )
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-10):
            raise ValueError("transfer covariance must be symmetric")
        if float(np.linalg.eigvalsh(covariance).min()) < -1.0e-10:
            raise ValueError("transfer covariance must be PSD")
        logical_basis = _matrix(
            self.recoverable_logical_basis, self.logical_dimension, None,
            "recoverable_logical_basis",
        )
        command_basis = _matrix(
            self.recoverable_command_basis, rf_dimension, None,
            "recoverable_command_basis",
        )
        if logical_basis.shape[1] != command_basis.shape[1]:
            raise ValueError("logical and RF support ranks must align")
        lower = np.asarray(self.lower_bounds, dtype=np.float64)
        upper = np.asarray(self.upper_bounds, dtype=np.float64)
        if lower.shape != (rf_dimension,) or upper.shape != (rf_dimension,) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("RF bounds must align and be finite")
        if np.any(lower > upper) or np.any(lower > 0.0) or np.any(upper < 0.0):
            raise ValueError("RF bounds must contain no-op and have lower <= upper")
        if not math.isfinite(float(self.trust_radius)) or float(self.trust_radius) <= 0.0:
            raise ValueError("RF trust radius must be positive and finite")
        if not math.isfinite(float(self.covariance_penalty)) or float(self.covariance_penalty) < 0.0:
            raise ValueError("covariance penalty must be finite and nonnegative")
        if not math.isfinite(float(self.ridge_lambda)) or float(self.ridge_lambda) < 0.0:
            raise ValueError("ridge lambda must be finite and nonnegative")
        interval = tuple(float(value) for value in self.phase_wrap_interval)
        if len(interval) != 2 or not all(math.isfinite(value) for value in interval) or interval[1] <= interval[0]:
            raise ValueError("phase wrap interval must be a finite increasing pair")
        current_phase = tuple(float(value) for value in self.current_phase_rad)
        if not current_phase:
            current_phase = tuple(0.0 for _ in range(rf_dimension))
        if len(current_phase) != rf_dimension or not all(math.isfinite(value) for value in current_phase):
            raise ValueError("current phase vector must align to RF coordinates")
        current_amplitude = tuple(float(value) for value in self.current_amplitude)
        if not current_amplitude:
            current_amplitude = tuple(0.0 for _ in range(rf_dimension))
        if len(current_amplitude) != rf_dimension or not all(math.isfinite(value) for value in current_amplitude):
            raise ValueError("current amplitude vector must align to RF coordinates")
        if self.calibration_status not in CALIBRATION_STATUSES:
            raise ValueError("RF calibration status is not registered")
        for name in (
            "rf_command_basis_sha", "logical_coordinate_layout_sha",
            "rf_coordinate_layout_sha", "current_control_snapshot_sha",
            "bounds_policy_sha",
        ):
            require_sha256(getattr(self, name), name)
        for name in ("operation_registry_sha", "command_table_sha"):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, name)
        for name in ("hardware_path_id", "operating_point_id"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value):
                raise ValueError(f"{name} must be a nonempty string when bound")
        if self.rf_to_awg_mapping is not None:
            if type(self.rf_to_awg_mapping) is not RegisteredRFToAWGMappingV1:
                raise TypeError("RF core requires the exact registered RF-to-AWG mapping type")
            if self.command_table_sha is None:
                raise ValueError("RF-to-AWG mapping requires a source-bound command-table SHA")
            self.rf_to_awg_mapping.validate_for_core(
                system_type=self.system_type,
                coordinate_ids=tuple(item.coordinate_id for item in self.rf_coordinates),
                coordinate_kinds=tuple(item.kind for item in self.rf_coordinates),
                command_table_sha=self.command_table_sha,
                current_control_snapshot_sha=self.current_control_snapshot_sha,
            )

        support_rank = logical_basis.shape[1]
        if support_rank == 0:
            if self.singular_support.supported_singular_indices or self.supported_singular_values:
                raise ValueError("rank-zero RF support cannot claim singular components")
        else:
            logical_frame = _orthonormal_columns(logical_basis, "recoverable_logical_basis")
            command_frame = _orthonormal_columns(command_basis, "recoverable_command_basis")
            effective = transfer @ command_frame
            left, singular, _right = np.linalg.svd(effective, full_matrices=False)
            indices = self.singular_support.supported_singular_indices
            values = tuple(float(value) for value in self.supported_singular_values)
            if (
                len(indices) != support_rank
                or len(values) != len(indices)
                or any(index >= singular.size for index in indices)
                or any(not math.isfinite(value) or value <= 0.0 for value in values)
                or not np.allclose(np.asarray(values), singular[list(indices)], rtol=1.0e-9, atol=1.0e-12)
            ):
                raise ValueError("RF singular support does not bind A_nominal @ Q_rf")
            threshold = max(float(singular.max(initial=0.0)) * self.singular_support.relative_tolerance, 1.0e-12)
            if np.any(singular[list(indices)] <= threshold):
                raise ValueError("RF singular support includes a below-threshold component")
            if not np.allclose(
                logical_frame @ logical_frame.T,
                left[:, list(indices)] @ left[:, list(indices)].T,
                rtol=0.0, atol=1.0e-10,
            ):
                raise ValueError("logical support is not the reachable RF support")
        object.__setattr__(self, "nominal_transfer_matrix", tuple(tuple(float(value) for value in row) for row in transfer))
        object.__setattr__(self, "transfer_covariance", tuple(tuple(float(value) for value in row) for row in covariance))
        object.__setattr__(self, "recoverable_logical_basis", tuple(tuple(float(value) for value in row) for row in logical_basis))
        object.__setattr__(self, "recoverable_command_basis", tuple(tuple(float(value) for value in row) for row in command_basis))
        object.__setattr__(self, "lower_bounds", tuple(float(value) for value in lower))
        object.__setattr__(self, "upper_bounds", tuple(float(value) for value in upper))
        object.__setattr__(self, "supported_singular_values", tuple(float(value) for value in self.supported_singular_values))
        object.__setattr__(self, "current_amplitude", current_amplitude)
        object.__setattr__(self, "current_phase_rad", current_phase)
        object.__setattr__(self, "phase_wrap_interval", interval)

    @property
    def rf_dimension(self) -> int:
        return len(self.rf_coordinates)

    @property
    def has_recoverable_support(self) -> bool:
        return bool(self.recoverable_logical_basis[0])

    @property
    def rf_to_awg_recipe_sha(self) -> str | None:
        if self.rf_to_awg_mapping is None:
            return None
        return registered_rf_to_awg_recipe_sha(
            self.rf_to_awg_mapping.sha256(),
            rf_mapping_core_sha=self.sha256(),
            current_amplitude=self.current_amplitude,
            current_phase_rad=self.current_phase_rad,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "system_type": self.system_type,
            "logical_dimension": self.logical_dimension,
            "rf_coordinates": [
                {"coordinate_id": item.coordinate_id, "kind": item.kind}
                for item in self.rf_coordinates
            ],
            "nominal_transfer_matrix": [list(row) for row in self.nominal_transfer_matrix],
            "transfer_covariance": [list(row) for row in self.transfer_covariance],
            "recoverable_logical_basis": [list(row) for row in self.recoverable_logical_basis],
            "recoverable_command_basis": [list(row) for row in self.recoverable_command_basis],
            "singular_support": self.singular_support.to_dict(),
            "singular_support_sha": self.singular_support.sha256(),
            "supported_singular_values": list(self.supported_singular_values),
            "lower_bounds": list(self.lower_bounds),
            "upper_bounds": list(self.upper_bounds),
            "trust_radius": self.trust_radius,
            "rf_command_basis_sha": self.rf_command_basis_sha,
            "logical_coordinate_layout_sha": self.logical_coordinate_layout_sha,
            "rf_coordinate_layout_sha": self.rf_coordinate_layout_sha,
            "current_control_snapshot_sha": self.current_control_snapshot_sha,
            "bounds_policy_sha": self.bounds_policy_sha,
            "calibration_status": self.calibration_status,
            "ridge_lambda": self.ridge_lambda,
            "covariance_penalty": self.covariance_penalty,
            "current_amplitude": list(self.current_amplitude),
            "current_phase_rad": list(self.current_phase_rad),
            "phase_wrap_interval": list(self.phase_wrap_interval),
            "operation_registry_sha": self.operation_registry_sha,
            "command_table_sha": self.command_table_sha,
            "hardware_path_id": self.hardware_path_id,
            "operating_point_id": self.operating_point_id,
            "rf_to_awg_mapping": (
                None if self.rf_to_awg_mapping is None else self.rf_to_awg_mapping.to_dict()
            ),
            "rf_to_awg_mapping_sha": (
                None if self.rf_to_awg_mapping is None else self.rf_to_awg_mapping.sha256()
            ),
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RegisteredRFMappingCoreV1":
        expected = {
            "schema", "system_type", "logical_dimension", "rf_coordinates",
            "nominal_transfer_matrix", "transfer_covariance", "recoverable_logical_basis",
            "recoverable_command_basis", "singular_support", "singular_support_sha",
            "supported_singular_values", "lower_bounds", "upper_bounds", "trust_radius",
            "rf_command_basis_sha", "logical_coordinate_layout_sha", "rf_coordinate_layout_sha",
            "current_control_snapshot_sha", "bounds_policy_sha", "calibration_status",
            "ridge_lambda", "covariance_penalty", "current_amplitude", "current_phase_rad",
            "phase_wrap_interval", "operation_registry_sha", "command_table_sha",
            "hardware_path_id", "operating_point_id", "rf_to_awg_mapping",
            "rf_to_awg_mapping_sha",
        }
        if set(payload) != expected or payload.get("schema") != "RegisteredRFMappingCoreV1":
            raise ValueError("RF mapping core schema is invalid")
        core = cls(
            system_type=str(payload["system_type"]),
            logical_dimension=int(payload["logical_dimension"]),
            rf_coordinates=tuple(
                RFCoordinateSpecV1.from_dict(value) for value in payload["rf_coordinates"]
            ),
            nominal_transfer_matrix=tuple(tuple(row) for row in payload["nominal_transfer_matrix"]),
            transfer_covariance=tuple(tuple(row) for row in payload["transfer_covariance"]),
            recoverable_logical_basis=tuple(tuple(row) for row in payload["recoverable_logical_basis"]),
            recoverable_command_basis=tuple(tuple(row) for row in payload["recoverable_command_basis"]),
            singular_support=RegisteredSingularSupportV1.from_dict(payload["singular_support"]),
            supported_singular_values=tuple(float(value) for value in payload["supported_singular_values"]),
            lower_bounds=tuple(float(value) for value in payload["lower_bounds"]),
            upper_bounds=tuple(float(value) for value in payload["upper_bounds"]),
            trust_radius=float(payload["trust_radius"]),
            rf_command_basis_sha=str(payload["rf_command_basis_sha"]),
            logical_coordinate_layout_sha=str(payload["logical_coordinate_layout_sha"]),
            rf_coordinate_layout_sha=str(payload["rf_coordinate_layout_sha"]),
            current_control_snapshot_sha=str(payload["current_control_snapshot_sha"]),
            bounds_policy_sha=str(payload["bounds_policy_sha"]),
            calibration_status=str(payload["calibration_status"]),
            ridge_lambda=float(payload.get("ridge_lambda", 1.0e-8)),
            covariance_penalty=float(payload.get("covariance_penalty", 0.0)),
            current_amplitude=tuple(float(value) for value in payload.get("current_amplitude", ())),
            current_phase_rad=tuple(float(value) for value in payload.get("current_phase_rad", ())),
            phase_wrap_interval=tuple(float(value) for value in payload.get("phase_wrap_interval", (-math.pi, math.pi))),
            operation_registry_sha=(None if payload.get("operation_registry_sha") is None else str(payload["operation_registry_sha"])),
            command_table_sha=(None if payload.get("command_table_sha") is None else str(payload["command_table_sha"])),
            hardware_path_id=(None if payload.get("hardware_path_id") is None else str(payload["hardware_path_id"])),
            operating_point_id=(None if payload.get("operating_point_id") is None else str(payload["operating_point_id"])),
            rf_to_awg_mapping=(
                None if payload.get("rf_to_awg_mapping") is None
                else RegisteredRFToAWGMappingV1.from_dict(dict(payload["rf_to_awg_mapping"]))
            ),
        )
        observed_mapping_sha = payload["rf_to_awg_mapping_sha"]
        if observed_mapping_sha != (None if core.rf_to_awg_mapping is None else core.rf_to_awg_mapping.sha256()):
            raise ValueError("RF mapping core RF-to-AWG mapping SHA does not match its mapping payload")
        return core

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "RegisteredRFMappingCoreV1":
        decoded = parse_canonical_json_object(
            payload,
            schema="RegisteredRFMappingCoreV1",
            exact_fields=frozenset({
                "schema", "system_type", "logical_dimension", "rf_coordinates",
                "nominal_transfer_matrix", "transfer_covariance", "recoverable_logical_basis",
                "recoverable_command_basis", "singular_support", "singular_support_sha",
                "supported_singular_values", "lower_bounds", "upper_bounds", "trust_radius",
                "rf_command_basis_sha", "logical_coordinate_layout_sha", "rf_coordinate_layout_sha",
                "current_control_snapshot_sha", "bounds_policy_sha", "calibration_status",
                "ridge_lambda", "covariance_penalty", "current_amplitude", "current_phase_rad",
                "phase_wrap_interval", "operation_registry_sha", "command_table_sha",
                "hardware_path_id", "operating_point_id", "rf_to_awg_mapping",
                "rf_to_awg_mapping_sha",
            }),
        )
        core = cls.from_dict(decoded)
        require_exact_canonical_json_round_trip(payload, core)
        return core

    def phase_wrap_valid(self, command_delta: Sequence[float], *, tolerance: float = 1.0e-10) -> bool:
        """Check current phase, wrap interval, and shortest-path delta semantics."""
        command = np.asarray(tuple(command_delta), dtype=np.float64)
        if command.shape != (self.rf_dimension,) or not np.all(np.isfinite(command)):
            return False
        lower, upper = self.phase_wrap_interval
        period = upper - lower
        current = np.asarray(self.current_phase_rad, dtype=np.float64)
        for index, coordinate in enumerate(self.rf_coordinates):
            if coordinate.kind != "PHASE":
                continue
            target = current[index] + command[index]
            wrapped = ((target - lower) % period) + lower
            shortest = ((command[index] + period / 2.0) % period) - period / 2.0
            if abs(shortest - command[index]) > tolerance \
                    or wrapped < lower - tolerance or wrapped > upper + tolerance:
                return False
        return True


@dataclass(frozen=True)
class RFMappingSolveResultV1:
    """Pure inverse result with explicit amplitude/phase/other RF components."""

    command_delta: tuple[float, ...]
    amplitude_delta: tuple[float, ...]
    phase_delta_rad: tuple[float, ...]
    other_delta: tuple[float, ...]
    materialized_logical_action: tuple[float, ...]
    support_leakage_norm: float
    inverse_residual_norm: float
    requested_to_materialized_logical_error: float
    direction_cosine: float
    ray_scale: float
    status: str
    accepted: bool
    no_op: bool
    schema: str = field(default="RFMappingSolveResultV1", init=False)


def solve_registered_rf_mapping(
    core: RegisteredRFMappingCoreV1, logical_action: Sequence[object],
) -> RFMappingSolveResultV1:
    """Solve one registered numerical inverse without coordinate-wise clipping."""
    try:
        requested = np.asarray(logical_action, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("logical action must be finite numeric values") from error
    if requested.shape != (core.logical_dimension,) or not np.all(np.isfinite(requested)):
        raise ValueError("logical action dimension/finiteness is invalid")
    zero_command = np.zeros(core.rf_dimension, dtype=np.float64)
    zero_logical = np.zeros(core.logical_dimension, dtype=np.float64)
    if core.calibration_status == "NOT_AVAILABLE" or not core.has_recoverable_support:
        return _result(core, requested, zero_command, zero_logical, zero_logical, 0.0, "NO_RECOVERABLE_RF_SUPPORT", False)
    logical_basis = np.asarray(core.recoverable_logical_basis, dtype=np.float64)
    command_basis = np.asarray(core.recoverable_command_basis, dtype=np.float64)
    logical_frame = _orthonormal_columns(logical_basis, "recoverable_logical_basis")
    requested_projection = logical_frame @ (logical_frame.T @ requested)
    if not np.allclose(requested, requested_projection, rtol=0.0, atol=1.0e-10):
        return _result(core, requested, zero_command, zero_logical, zero_logical, 0.0, "LOGICAL_ACTION_OUTSIDE_REGISTERED_SUPPORT", False)
    command_frame = _orthonormal_columns(command_basis, "recoverable_command_basis")
    transfer = np.asarray(core.nominal_transfer_matrix, dtype=np.float64)
    effective = transfer @ command_frame
    left, singular, right = np.linalg.svd(effective, full_matrices=False)
    indices = np.asarray(core.singular_support.supported_singular_indices, dtype=np.int64)
    covariance = np.asarray(core.transfer_covariance, dtype=np.float64)
    regularization = core.ridge_lambda + core.covariance_penalty * float(np.trace(covariance))
    reduced = np.zeros(command_frame.shape[1], dtype=np.float64)
    for index in indices:
        value = float(singular[index])
        coefficient = value / (value * value + regularization)
        reduced += coefficient * float(np.dot(left[:, index], requested)) * right[index]
    command = command_frame @ reduced
    norm = float(np.linalg.norm(command))
    if norm > core.trust_radius:
        command *= core.trust_radius / norm
    prebound = transfer @ command
    scale = _ray_scale(command, np.asarray(core.lower_bounds), np.asarray(core.upper_bounds))
    command *= scale
    materialized = transfer @ command
    support_projection = logical_frame @ (logical_frame.T @ materialized)
    return _result(
        core, requested, command, materialized, prebound, scale,
        "REGISTERED_RF_PREVIEW", True,
        support_projection=support_projection,
    )


def _result(
    core: RegisteredRFMappingCoreV1,
    requested: np.ndarray,
    command: np.ndarray,
    materialized: np.ndarray,
    prebound: np.ndarray,
    scale: float,
    status: str,
    accepted: bool,
    *,
    support_projection: np.ndarray | None = None,
) -> RFMappingSolveResultV1:
    projection = materialized if support_projection is None else support_projection
    requested_norm = float(np.linalg.norm(requested))
    materialized_norm = float(np.linalg.norm(materialized))
    if requested_norm <= 1.0e-15 and materialized_norm <= 1.0e-15:
        cosine = 1.0
    elif requested_norm * materialized_norm <= 1.0e-15:
        cosine = 0.0
    else:
        cosine = float(np.clip(requested @ materialized / (requested_norm * materialized_norm), -1.0, 1.0))
    kinds = tuple(item.kind for item in core.rf_coordinates)
    amplitude = tuple(float(value) if kind == "AMPLITUDE" else 0.0 for value, kind in zip(command, kinds, strict=True))
    phase = tuple(float(value) if kind == "PHASE" else 0.0 for value, kind in zip(command, kinds, strict=True))
    other = tuple(float(value) if kind == "OTHER" else 0.0 for value, kind in zip(command, kinds, strict=True))
    return RFMappingSolveResultV1(
        command_delta=tuple(float(value) for value in command),
        amplitude_delta=amplitude,
        phase_delta_rad=phase,
        other_delta=other,
        materialized_logical_action=tuple(float(value) for value in materialized),
        support_leakage_norm=float(np.linalg.norm(materialized - projection)),
        inverse_residual_norm=float(np.linalg.norm(requested - prebound)),
        requested_to_materialized_logical_error=float(np.linalg.norm(requested - materialized)),
        direction_cosine=cosine,
        ray_scale=float(scale),
        status=status,
        accepted=bool(accepted),
        no_op=bool(np.allclose(command, 0.0, atol=1.0e-15)),
    )


__all__ = [
    "CALIBRATION_STATUSES", "RF_COORDINATE_KINDS", "RFCoordinateSpecV1",
    "RFMappingSolveResultV1", "RegisteredRFMappingCoreV1",
    "RegisteredSingularSupportV1", "solve_registered_rf_mapping",
]
