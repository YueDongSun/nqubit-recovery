"""Strict immutable physical-budget contracts for Track A research fixtures.

These payloads distinguish declared synthetic repetitions from source-verified
real repetitions. They never assert that an acquisition or hardware run occurred.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


class Physical_budget_error(ValueError):
    """Raised when a physical-budget payload is not self-consistent."""


SCHEMA = "V10PhysicalMeasurementBudgetV2"
CANONICAL_PROTOCOLS = frozenset(
    {"FIXED_RESOURCE", "FIXED_TIME", "FIXED_QUALITY", "SENSITIVITY"}
)
COMPATIBILITY_PROTOCOL_ALIASES = {
    "FIXED_MODE_BUDGET": "FIXED_RESOURCE",
    "FIXED_TRIAL_BUDGET": "FIXED_TIME",
    "FIXED_TIME_RATE": "FIXED_TIME",
}
SUCCESS_SOURCES = frozenset(
    {"UNAVAILABLE", "SYNTHETIC_FORWARD", "SOURCE_BOUND_FORWARD"}
)
SUCCESS_MODELS = frozenset(
    {"UNAVAILABLE", "POISSON_BINOMIAL", "CORRELATED_MIXTURE", "EMPIRICAL"}
)
REPEAT_STATUSES = frozenset(
    {"SYNTHETIC_DECLARED", "ACTUAL_REPEAT_BOUND", "ACTUAL_REPEAT_UNBOUND"}
)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Physical_budget_error(f"{name} must be an integer >= {minimum}")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if (
        type(value) not in (int, float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise Physical_budget_error(f"{name} must be finite and nonnegative")
    return float(value)


def _probability(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result > 1.0:
        raise Physical_budget_error(f"{name} must lie in [0, 1]")
    return result


def _sha(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Physical_budget_error(f"{name} must be a lowercase SHA-256 digest")
    return value


def _tuple(value: Any, name: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise Physical_budget_error(f"{name} must be an immutable tuple")
    return value


def canonical_budget_protocol(value: Any) -> str:
    if type(value) is not str:
        raise Physical_budget_error("budget_protocol must be a string")
    result = COMPATIBILITY_PROTOCOL_ALIASES.get(value, value)
    if result not in CANONICAL_PROTOCOLS:
        raise Physical_budget_error("budget_protocol is invalid")
    return result


@dataclass(frozen=True)
class Budget_provenance:
    source_kind: str
    source_identity: str
    source_manifest_sha: str
    actual_repeat_status: str
    source_manifest_path: str | None = None

    def validate(self) -> None:
        if self.source_kind not in {"synthetic_fixture", "source_bound_real"}:
            raise Physical_budget_error("source_kind is invalid")
        if type(self.source_identity) is not str or not self.source_identity:
            raise Physical_budget_error("source_identity is required")
        _sha(self.source_manifest_sha, "source_manifest_sha")
        if self.actual_repeat_status not in REPEAT_STATUSES:
            raise Physical_budget_error("actual_repeat_status is invalid")
        if (
            self.source_kind == "source_bound_real"
            and self.actual_repeat_status == "SYNTHETIC_DECLARED"
        ):
            raise Physical_budget_error(
                "real provenance cannot use synthetic repeat status"
            )
        if (
            self.source_kind == "synthetic_fixture"
            and self.actual_repeat_status != "SYNTHETIC_DECLARED"
        ):
            raise Physical_budget_error(
                "synthetic fixtures require declared repeat status"
            )
        if self.source_manifest_path is not None and (
            type(self.source_manifest_path) is not str or not self.source_manifest_path
        ):
            raise Physical_budget_error(
                "source_manifest_path must be a nonempty explicit path"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "source_kind": self.source_kind,
            "source_identity": self.source_identity,
            "source_manifest_sha": self.source_manifest_sha,
            "actual_repeat_status": self.actual_repeat_status,
            "source_manifest_path": self.source_manifest_path,
        }


def verify_actual_repeat_manifest(provenance: Budget_provenance) -> Path:
    """Verify a named manifest before a caller treats repeats as source-bound.

    A matching hash proves byte identity only. It does not create acquisition,
    execution, or hardware evidence.
    """
    provenance.validate()
    if (
        provenance.source_kind != "source_bound_real"
        or provenance.actual_repeat_status != "ACTUAL_REPEAT_BOUND"
    ):
        raise Physical_budget_error(
            "actual repeats require source_bound_real and ACTUAL_REPEAT_BOUND"
        )
    if provenance.source_manifest_path is None:
        raise Physical_budget_error(
            "actual repeats require an explicit source_manifest_path"
        )
    manifest_path = Path(provenance.source_manifest_path)
    if not manifest_path.is_file():
        raise Physical_budget_error("actual-repeat manifest path is not a file")
    if (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        != provenance.source_manifest_sha
    ):
        raise Physical_budget_error(
            "actual-repeat manifest hash does not match source_manifest_sha"
        )
    return manifest_path


@dataclass(frozen=True)
class Operation_registry_binding:
    operation_ids: tuple[str, ...]
    operation_roles: tuple[str, ...]
    n_basis_core: int
    measurement_schedule_sha: str
    operation_registry_sha: str
    command_table_sha: str
    phase_convention: str
    mode_coordinate_schema_sha: str

    def validate(self) -> None:
        _tuple(self.operation_ids, "operation_ids")
        _tuple(self.operation_roles, "operation_roles")
        if (
            not self.operation_ids
            or len(self.operation_ids) != len(self.operation_roles)
            or len(set(self.operation_ids)) != len(self.operation_ids)
        ):
            raise Physical_budget_error(
                "operation registry must align and have unique IDs"
            )
        if any(
            type(item) is not str or not item
            for item in self.operation_ids + self.operation_roles
        ):
            raise Physical_budget_error("operation IDs and roles are required")
        _integer(self.n_basis_core, "n_basis_core")
        if self.n_basis_core > len(self.operation_ids):
            raise Physical_budget_error("n_basis_core exceeds operation window")
        for value, name in (
            (self.measurement_schedule_sha, "measurement_schedule_sha"),
            (self.operation_registry_sha, "operation_registry_sha"),
            (self.command_table_sha, "command_table_sha"),
            (self.mode_coordinate_schema_sha, "mode_coordinate_schema_sha"),
        ):
            _sha(value, name)
        if type(self.phase_convention) is not str or not self.phase_convention:
            raise Physical_budget_error("phase_convention is required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "operation_ids": list(self.operation_ids),
            "operation_roles": list(self.operation_roles),
            "n_basis_core": self.n_basis_core,
            "measurement_schedule_sha": self.measurement_schedule_sha,
            "operation_registry_sha": self.operation_registry_sha,
            "command_table_sha": self.command_table_sha,
            "phase_convention": self.phase_convention,
            "mode_coordinate_schema_sha": self.mode_coordinate_schema_sha,
        }


@dataclass(frozen=True)
class Physical_budget:
    n_mode: int
    registry: Operation_registry_binding
    repeats_by_operation: tuple[int, ...]
    provenance: Budget_provenance
    budget_protocol: str = "FIXED_RESOURCE"
    requested_mode_shots: int | None = None
    repeat_multiplier: int | None = None
    t_window_seconds: float | None = None
    sequence_repetition_rate_hz: float | None = None
    success_probability_by_operation: tuple[float, ...] | None = ()
    success_probability_source: str = "UNAVAILABLE"
    m_succ_realized: int | None = None
    success_multiplicity_model: str = "POISSON_BINOMIAL"
    schema: str = SCHEMA

    def validate(self) -> None:
        if self.schema != SCHEMA:
            raise Physical_budget_error("V2 schema is required")
        _integer(self.n_mode, "n_mode", 1)
        self.registry.validate()
        self.provenance.validate()
        _tuple(self.repeats_by_operation, "repeats_by_operation")
        if self.provenance.actual_repeat_status == "ACTUAL_REPEAT_UNBOUND":
            self._validate_unbound_repeats()
            return
        if len(self.repeats_by_operation) != len(self.registry.operation_ids) or any(
            type(value) is not int or value < 0 for value in self.repeats_by_operation
        ):
            raise Physical_budget_error("repeats must be aligned nonnegative integers")
        canonical_budget_protocol(self.budget_protocol)
        if (
            self.requested_mode_shots is not None
            and _integer(self.requested_mode_shots, "requested_mode_shots")
            < self.b_mode_shot
        ):
            raise Physical_budget_error("realized mode budget exceeds requested budget")
        if self.repeat_multiplier is not None:
            _integer(self.repeat_multiplier, "repeat_multiplier")
            if any(
                repeat != self.repeat_multiplier for repeat in self.repeats_by_operation
            ):
                raise Physical_budget_error(
                    "repeat_multiplier conflicts with repeats_by_operation"
                )
        if (self.t_window_seconds is None) != (
            self.sequence_repetition_rate_hz is None
        ):
            raise Physical_budget_error(
                "time and sequence rate must be supplied together"
            )
        if self.t_window_seconds is not None:
            capacity = math.floor(
                _finite_nonnegative(self.t_window_seconds, "t_window_seconds")
                * _finite_nonnegative(
                    self.sequence_repetition_rate_hz, "sequence_repetition_rate_hz"
                )
            )
            if self.b_trial > capacity:
                raise Physical_budget_error("trial budget exceeds time-rate capacity")
        _tuple(
            self.success_probability_by_operation, "success_probability_by_operation"
        )
        if self.success_probability_source not in SUCCESS_SOURCES:
            raise Physical_budget_error("success_probability_source is invalid")
        if self.success_probability_by_operation:
            if len(self.success_probability_by_operation) != self.n_op_window:
                raise Physical_budget_error(
                    "success probabilities must align with operations"
                )
            for probability in self.success_probability_by_operation:
                _probability(probability, "success probability")
            if self.success_probability_source == "UNAVAILABLE":
                raise Physical_budget_error(
                    "present success probabilities require a declared source"
                )
        elif self.success_probability_source != "UNAVAILABLE":
            raise Physical_budget_error(
                "missing success probabilities must be marked UNAVAILABLE"
            )
        if self.m_succ_realized is not None and (
            type(self.m_succ_realized) is not int
            or not 0 <= self.m_succ_realized <= self.b_trial
        ):
            raise Physical_budget_error("realized success count is invalid")
        if self.success_multiplicity_model not in SUCCESS_MODELS:
            raise Physical_budget_error("success multiplicity model is invalid")

    def _validate_unbound_repeats(self) -> None:
        """Reject every value that would turn unknown real repeats into a count.

        ``n_mode`` and the operation registry remain available because they are
        system/schedule metadata, not evidence of a row-bound repeat count.
        """
        if self.repeats_by_operation != ():
            raise Physical_budget_error("unbound actual repeats must be empty")
        if any(
            value is not None
            for value in (
                self.requested_mode_shots,
                self.repeat_multiplier,
                self.t_window_seconds,
                self.sequence_repetition_rate_hz,
                self.m_succ_realized,
            )
        ):
            raise Physical_budget_error("unbound actual repeats cannot carry budgets")
        if self.success_probability_by_operation is not None:
            raise Physical_budget_error(
                "unbound actual repeats require null success probabilities"
            )
        if self.success_probability_source != "UNAVAILABLE":
            raise Physical_budget_error(
                "unbound actual repeats require unavailable success probabilities"
            )
        if self.success_multiplicity_model != "UNAVAILABLE":
            raise Physical_budget_error(
                "unbound actual repeats require unavailable success multiplicity"
            )

    @property
    def n_op_window(self) -> int:
        return len(self.registry.operation_ids)

    @property
    def n_basis_core(self) -> int:
        return self.registry.n_basis_core

    @property
    def b_trial(self) -> int | None:
        if not self.has_repeat_measurements:
            return None
        return sum(self.repeats_by_operation)

    @property
    def b_mode_shot(self) -> int | None:
        if self.b_trial is None:
            return None
        return self.n_mode * self.b_trial

    @property
    def has_repeat_measurements(self) -> bool:
        """Whether budget-derived quantities are safe to consume."""
        return self.provenance.actual_repeat_status != "ACTUAL_REPEAT_UNBOUND"

    def require_repeat_measurements(self) -> None:
        """Fail closed before a renderer or evaluator consumes repeat counts."""
        self.validate()
        if not self.has_repeat_measurements:
            raise Physical_budget_error(
                "ACTUAL_REPEAT_UNBOUND cannot supply derived physical budgets"
            )

    @property
    def unused_requested_mode_shots(self) -> int | None:
        return (
            None
            if self.requested_mode_shots is None
            else self.requested_mode_shots - self.require_bound_mode_shots()
        )

    def require_bound_mode_shots(self) -> int:
        self.require_repeat_measurements()
        assert self.b_mode_shot is not None
        return self.b_mode_shot

    @property
    def canonical_protocol(self) -> str:
        return canonical_budget_protocol(self.budget_protocol)

    @property
    def mu_succ_predicted(self) -> float | None:
        if (
            not self.has_repeat_measurements
            or not self.success_probability_by_operation
        ):
            return None
        assert self.b_trial is not None
        return float(
            sum(
                probability * repeat
                for probability, repeat in zip(
                    self.success_probability_by_operation, self.repeats_by_operation
                )
            )
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "n_mode": self.n_mode,
            "n_op_window": self.n_op_window,
            "n_basis_core": self.registry.n_basis_core,
            "registry": self.registry.to_dict(),
            "repeats_by_operation": list(self.repeats_by_operation),
            "repeat_multiplier": self.repeat_multiplier,
            "b_trial": self.b_trial,
            "b_mode_shot": self.b_mode_shot,
            "requested_mode_shots": self.requested_mode_shots,
            "unused_requested_mode_shots": self.unused_requested_mode_shots,
            "budget_protocol": self.canonical_protocol,
            "t_window_seconds": self.t_window_seconds,
            "sequence_repetition_rate_hz": self.sequence_repetition_rate_hz,
            "success_probability_by_operation": (
                None
                if self.success_probability_by_operation is None
                else list(self.success_probability_by_operation)
            ),
            "success_probability_source": self.success_probability_source,
            "mu_succ_predicted": self.mu_succ_predicted,
            "m_succ_realized": self.m_succ_realized,
            "success_multiplicity_model": self.success_multiplicity_model,
            "provenance": self.provenance.to_dict(),
        }

    def canonical_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Physical_budget":
        if not isinstance(value, Mapping):
            raise Physical_budget_error("physical budget must be a mapping")
        expected_keys = set(cls._mapping_keys())
        unknown_keys, missing_keys = (
            set(value) - expected_keys,
            expected_keys - set(value),
        )
        if unknown_keys or missing_keys:
            raise Physical_budget_error(
                f"physical budget keys mismatch: missing={sorted(missing_keys)}, unknown={sorted(unknown_keys)}"
            )
        registry_value, provenance_value = value["registry"], value["provenance"]
        if not isinstance(registry_value, Mapping) or not isinstance(
            provenance_value, Mapping
        ):
            raise Physical_budget_error("registry and provenance mappings are required")
        registry = Operation_registry_binding(
            tuple(registry_value["operation_ids"]),
            tuple(registry_value["operation_roles"]),
            registry_value["n_basis_core"],
            registry_value["measurement_schedule_sha"],
            registry_value["operation_registry_sha"],
            registry_value["command_table_sha"],
            registry_value["phase_convention"],
            registry_value["mode_coordinate_schema_sha"],
        )
        provenance = Budget_provenance(
            provenance_value["source_kind"],
            provenance_value["source_identity"],
            provenance_value["source_manifest_sha"],
            provenance_value["actual_repeat_status"],
            provenance_value["source_manifest_path"],
        )
        result = cls(
            value["n_mode"],
            registry,
            tuple(value["repeats_by_operation"]),
            provenance,
            budget_protocol=canonical_budget_protocol(value["budget_protocol"]),
            requested_mode_shots=value["requested_mode_shots"],
            repeat_multiplier=value["repeat_multiplier"],
            t_window_seconds=value["t_window_seconds"],
            sequence_repetition_rate_hz=value["sequence_repetition_rate_hz"],
            success_probability_by_operation=(
                None
                if value["success_probability_by_operation"] is None
                else tuple(value["success_probability_by_operation"])
            ),
            success_probability_source=value["success_probability_source"],
            m_succ_realized=value["m_succ_realized"],
            success_multiplicity_model=value["success_multiplicity_model"],
            schema=value["schema"],
        )
        result.validate()
        for derived_name in (
            "n_op_window",
            "n_basis_core",
            "b_trial",
            "b_mode_shot",
            "unused_requested_mode_shots",
            "mu_succ_predicted",
        ):
            if value[derived_name] != getattr(result, derived_name):
                raise Physical_budget_error(
                    f"derived field {derived_name} does not match authoritative payload"
                )
        if result.to_dict() != dict(value):
            raise Physical_budget_error("physical budget mapping is not canonical")
        return result

    @staticmethod
    def _mapping_keys() -> tuple[str, ...]:
        return (
            "schema",
            "n_mode",
            "n_op_window",
            "n_basis_core",
            "registry",
            "repeats_by_operation",
            "repeat_multiplier",
            "b_trial",
            "b_mode_shot",
            "requested_mode_shots",
            "unused_requested_mode_shots",
            "budget_protocol",
            "t_window_seconds",
            "sequence_repetition_rate_hz",
            "success_probability_by_operation",
            "success_probability_source",
            "mu_succ_predicted",
            "m_succ_realized",
            "success_multiplicity_model",
            "provenance",
        )


Physical_budget_v2 = Physical_budget


def allocate_mode_budget(
    *, n_mode: int, operation_count: int, requested_mode_shots: int
) -> tuple[int, ...]:
    _integer(n_mode, "n_mode", 1)
    _integer(operation_count, "operation_count", 1)
    _integer(requested_mode_shots, "requested_mode_shots")
    base, remainder = divmod(requested_mode_shots // n_mode, operation_count)
    return tuple(base + (index < remainder) for index in range(operation_count))
