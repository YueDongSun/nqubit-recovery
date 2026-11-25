"""Immutable R17 memory extension; validation does not grant scientific authority.

Source JSON verification establishes byte/value/unit consistency only. It does
not establish authenticity, calibration acceptance, or hardware permission.
Evidence locations are excluded from portable parameter semantic identities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any

from .shared_abi import SEMANTIC_ABI_SCHEMA_V2, Semantic_abi_manifest_v2


SCHEMA = "V10MemoryFusionContextV1"
PARAMETER_SCHEMA = "V10MemoryParameterManifestV1"
PARAMETER_SOURCE_SCHEMA = "V10MemoryParameterSourceV1"
EXTENSION_SCHEMA = "V10MemoryFusionExtensionManifestV1"
SOURCE_TAGS = frozenset({"SYNTHETIC", "PUBLISHED_EXPERIMENT", "LOCAL_CALIBRATION"})
CAPACITY_UNITS = frozenset({"LOGICAL_RESOURCE", "PHYSICAL_MEMORY_CELL"})
BUFFER_POLICIES = frozenset({"FIFO", "BEST_QUALITY"})
PARTNER_SCOPES = frozenset(
    {"UNKNOWN", "HELD_RESOURCE_CLASS", "HOMOGENEOUS_ALL_COMPATIBLE"}
)
NUMERIC_UNITS = {
    "storage_efficiency": "dimensionless",
    "tau_eff_seconds": "seconds",
    "maximum_storage_age_s": "seconds",
    "global_qualified_resource_rate_hz": "resources_per_second",
    "held_compatible_partner_rate_hz": "resources_per_second",
}
NUMERIC_FIELDS = tuple(NUMERIC_UNITS)
MAXIMUM_SOURCE_BYTES = 2 * 1024 * 1024


def _sha(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be canonical SHA-256")
    return value


def _name(value: Any, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _time(value: Any, name: str) -> datetime:
    if type(value) is not str:
        raise ValueError(f"{name} must be an ISO timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO timestamp") from error
    if result.tzinfo is None:
        raise ValueError(f"{name} needs an explicit timezone")
    return result.astimezone(timezone.utc)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate source JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite source JSON constant: {value}")


@dataclass(frozen=True)
class Memory_fusion_parameter_manifest:
    source_tag: str
    validity: tuple[bool, ...]
    storage_efficiency: float | None
    tau_eff_seconds: float | None
    maximum_storage_age_s: float | None
    ideal_regime: bool
    global_qualified_resource_rate_hz: float | None
    held_compatible_partner_rate_hz: float | None
    partner_scope: str
    buffer_capacity: int
    capacity_unit: str
    buffer_policy: str
    fusion_order: int
    storage_model_sha: str
    retrieval_model_sha: str
    coherence_model_sha: str
    quality_model_sha: str
    compatibility_contract_sha: str
    consumption_schema_sha: str
    external_reference_id: str | None
    validity_start: str
    validity_end: str
    source_file_sha: str | None = None
    source_file_path: str | None = None
    source_payload_json: str | None = None
    source_available_at: str | None = None
    held_resource_class: str | None = None
    memory_model_id: str = "SYNTHETIC_DECLARED_MEMORY"
    memory_model_sha: str | None = None
    schema: str = PARAMETER_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "validity", tuple(self.validity))

    def validate(self) -> None:
        if self.schema != PARAMETER_SCHEMA or self.source_tag not in SOURCE_TAGS:
            raise ValueError("invalid memory parameter schema or source tag")
        if type(self.ideal_regime) is not bool:
            raise ValueError("ideal_regime must be a strict boolean")
        if len(self.validity) != len(NUMERIC_FIELDS) or any(
            type(flag) is not bool for flag in self.validity
        ):
            raise ValueError("validity must align with the five named numeric fields")
        for name, flag in zip(NUMERIC_FIELDS, self.validity):
            value = getattr(self, name)
            if flag != (value is not None):
                raise ValueError(
                    f"{name} must be None exactly when numeric validity is false"
                )
            if value is not None and (
                type(value) not in (int, float) or not isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} must be finite nonnegative or unknown")
        if self.storage_efficiency is not None and self.storage_efficiency > 1:
            raise ValueError("storage efficiency must be a probability")
        if self.tau_eff_seconds is not None and self.tau_eff_seconds <= 0:
            raise ValueError("finite tau_eff_seconds must be positive")
        if self.ideal_regime and self.tau_eff_seconds is not None:
            raise ValueError(
                "ideal regime has no finite tau value; use None, never Infinity"
            )
        if type(self.buffer_capacity) is not int or self.buffer_capacity < 1:
            raise ValueError("buffer_capacity must be a positive integer")
        if self.buffer_policy == "TARGET_MATCHED":
            raise ValueError(
                "TARGET_MATCHED is unsupported without a source-bound target-ranking contract"
            )
        if self.capacity_unit not in CAPACITY_UNITS or self.buffer_policy not in BUFFER_POLICIES:
            raise ValueError("capacity unit or buffer policy is not registered")
        if type(self.fusion_order) is not int or self.fusion_order < 2:
            raise ValueError("fusion_order must be an integer of at least two")
        if self.partner_scope not in PARTNER_SCOPES:
            raise ValueError("partner rate scope is not registered")
        if self.held_compatible_partner_rate_hz is not None:
            if self.partner_scope == "UNKNOWN":
                raise ValueError("known partner rate requires its conditional scope")
            if self.partner_scope == "HELD_RESOURCE_CLASS":
                _name(self.held_resource_class, "held_resource_class")
            if (
                self.partner_scope == "HOMOGENEOUS_ALL_COMPATIBLE"
                and self.global_qualified_resource_rate_hz
                != self.held_compatible_partner_rate_hz
            ):
                raise ValueError(
                    "homogeneous all-compatible global and partner rates must agree"
                )
        for name in (
            "storage_model_sha",
            "retrieval_model_sha",
            "coherence_model_sha",
            "quality_model_sha",
            "compatibility_contract_sha",
            "consumption_schema_sha",
        ):
            _sha(getattr(self, name), name)
        _name(self.memory_model_id, "memory_model_id")
        if self.memory_model_sha is not None:
            _sha(self.memory_model_sha, "memory_model_sha")
        start, end = (
            _time(self.validity_start, "validity_start"),
            _time(self.validity_end, "validity_end"),
        )
        if start > end:
            raise ValueError("invalid parameter validity interval")
        if self.source_available_at is not None:
            _time(self.source_available_at, "source_available_at")
        if self.source_tag != "SYNTHETIC":
            _name(self.external_reference_id, "external_reference_id")
            if self.source_available_at is None:
                raise ValueError(
                    "external parameters need their knowledge availability timestamp"
                )
        if self.source_tag != "SYNTHETIC" or any(
            value is not None
            for value in (
                self.source_file_sha,
                self.source_file_path,
                self.source_payload_json,
            )
        ):
            verify_parameter_source(self)
        if self.memory_loading_parameter is not None and not isfinite(
            self.memory_loading_parameter
        ):
            raise ValueError("memory loading diagnostic overflowed")

    @property
    def memory_loading_parameter(self) -> float | None:
        if (
            self.ideal_regime
            or self.tau_eff_seconds is None
            or self.held_compatible_partner_rate_hz is None
        ):
            return None
        return self.tau_eff_seconds * self.held_compatible_partner_rate_hz

    def semantic_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("source_file_path")
        result.pop("source_payload_json")
        return result

    def sha256(self) -> str:
        self.validate()
        return _digest(self.semantic_dict())


def verify_parameter_source(parameters: Memory_fusion_parameter_manifest) -> None:
    """Verify a bounded external JSON record; never authenticate a physical model."""
    _sha(parameters.source_file_sha, "source_file_sha")
    _name(parameters.source_file_path, "source_file_path")
    source = Path(parameters.source_file_path)
    try:
        if (
            source.is_symlink()
            or not source.is_file()
            or not 0 < source.stat().st_size <= MAXIMUM_SOURCE_BYTES
        ):
            raise ValueError(
                "parameter source must be a bounded regular local JSON file"
            )
        raw = source.read_bytes()
    except OSError as error:
        raise ValueError("parameter source cannot be read") from error
    if hashlib.sha256(raw).hexdigest() != parameters.source_file_sha:
        raise ValueError("external source file bytes do not match SHA")
    if (
        parameters.source_payload_json is not None
        and parameters.source_payload_json.encode("utf-8") != raw
    ):
        raise ValueError(
            "embedded source payload differs from the exact source file bytes"
        )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as offer:
        raise ValueError("source payload must be strict UTF-8 JSON") from offer
    expected = {
        "schema": PARAMETER_SOURCE_SCHEMA,
        "source_tag": parameters.source_tag,
        "external_reference_id": parameters.external_reference_id,
        "available_at": parameters.source_available_at,
        "validity_start": parameters.validity_start,
        "validity_end": parameters.validity_end,
        "parameters": {name: getattr(parameters, name) for name in NUMERIC_FIELDS},
        "units": NUMERIC_UNITS,
        "validity": dict(zip(NUMERIC_FIELDS, parameters.validity)),
    }
    if payload != expected:
        raise ValueError(
            "source JSON values, units, validity, or provenance differ from the declared parameters"
        )


@dataclass(frozen=True)
class Public_memory_resource:
    resource_id: str
    input_n_modes: int
    mode_id: str
    partition_sha: str
    compatibility_key: str
    logical_resource_units: int
    physical_cell_units: int
    physical_cell_ids: tuple[str, ...]
    allocation_map_sha: str
    generated_at: str
    successful_load_completed_at: str
    qualification_receipt_sha: str
    estimated_quality: float | None
    booked: bool
    reserved: bool
    consumed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "physical_cell_ids", tuple(self.physical_cell_ids))

    def validate(self, as_of: datetime) -> None:
        for name in ("resource_id", "mode_id", "compatibility_key"):
            _name(getattr(self, name), name)
        for name in ("input_n_modes", "logical_resource_units", "physical_cell_units"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(
                    "resource dimensions and units must be positive integers"
                )
        for name in (
            "partition_sha",
            "allocation_map_sha",
            "qualification_receipt_sha",
        ):
            _sha(getattr(self, name), name)
        if len(self.physical_cell_ids) != self.physical_cell_units or len(
            set(self.physical_cell_ids)
        ) != len(self.physical_cell_ids):
            raise ValueError(
                "physical cell IDs must be unique and align with physical units"
            )
        for value in self.physical_cell_ids:
            _name(value, "physical_cell_id")
        generated = _time(self.generated_at, "generated_at")
        loaded = _time(
            self.successful_load_completed_at, "successful_load_completed_at"
        )
        if (
            not isinstance(as_of, datetime)
            or as_of.tzinfo is None
            or generated > loaded
            or loaded > as_of
        ):
            raise ValueError(
                "resource timestamps cannot be future or precede generation"
            )
        if self.estimated_quality is not None and (
            type(self.estimated_quality) not in (int, float)
            or not isfinite(self.estimated_quality)
            or not 0 <= self.estimated_quality <= 1
        ):
            raise ValueError("estimated quality must be finite, in [0,1], or unknown")
        if (
            any(
                type(value) is not bool
                for value in (self.booked, self.reserved, self.consumed)
            )
            or self.reserved
            and not self.booked
            or self.consumed
            and (self.booked or self.reserved)
        ):
            raise ValueError(
                "inconsistent observable booking, reservation or consumption state"
            )


@dataclass(frozen=True)
class Memory_fusion_extension_manifest:
    base_abi_schema: str
    base_abi_sha: str
    memory_parameter_sha: str
    event_sha: str
    queue_policy_sha: str
    compatibility_sha: str
    consumption_sha: str
    memory_schema_sha: str
    parameter_schema_sha: str
    queue_model_sha: str
    schema: str = EXTENSION_SCHEMA

    def validate(self) -> None:
        if (
            self.schema != EXTENSION_SCHEMA
            or self.base_abi_schema != SEMANTIC_ABI_SCHEMA_V2
        ):
            raise ValueError(
                "exact V2 base schema and separate memory extension are required"
            )
        for name in (
            "base_abi_sha",
            "memory_parameter_sha",
            "event_sha",
            "queue_policy_sha",
            "compatibility_sha",
            "consumption_sha",
            "memory_schema_sha",
            "parameter_schema_sha",
            "queue_model_sha",
        ):
            _sha(getattr(self, name), name)


@dataclass(frozen=True)
class V10MemoryFusionContextV1:
    as_of: str
    timestamp_origin: str
    base_abi: Semantic_abi_manifest_v2
    parameters: Memory_fusion_parameter_manifest
    resources: tuple[Public_memory_resource, ...]
    extension: Memory_fusion_extension_manifest
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "resources", tuple(self.resources))

    def validate(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("memory fusion schema mismatch")
        if (
            type(self.base_abi) is not Semantic_abi_manifest_v2
            or type(self.parameters) is not Memory_fusion_parameter_manifest
            or type(self.extension) is not Memory_fusion_extension_manifest
            or any(type(item) is not Public_memory_resource for item in self.resources)
        ):
            raise ValueError("typed immutable memory contracts are required")
        as_of, origin = (
            _time(self.as_of, "as_of"),
            _time(self.timestamp_origin, "timestamp_origin"),
        )
        if origin > as_of:
            raise ValueError("timestamp origin cannot be future")
        self.base_abi.validate()
        self.parameters.validate()
        self.extension.validate()
        if (
            self.extension.base_abi_sha != self.base_abi.sha256()
            or self.extension.memory_parameter_sha != self.parameters.sha256()
            or self.extension.compatibility_sha
            != self.parameters.compatibility_contract_sha
            or self.extension.consumption_sha != self.parameters.consumption_schema_sha
        ):
            raise ValueError(
                "memory extension does not bind the base ABI, parameters, or action contracts"
            )
        if (
            not _time(self.parameters.validity_start, "validity_start")
            <= as_of
            <= _time(self.parameters.validity_end, "validity_end")
        ):
            raise ValueError("decision is outside parameter validity")
        if (
            self.parameters.source_available_at is not None
            and _time(self.parameters.source_available_at, "source_available_at")
            > as_of
        ):
            raise ValueError("future parameter knowledge cannot enter the decision")
        if len({item.resource_id for item in self.resources}) != len(self.resources):
            raise ValueError("public resources must have unique IDs")
        active = []
        for item in self.resources:
            item.validate(as_of)
            if _time(item.generated_at, "generated_at") < origin:
                raise ValueError("resource predates the registered timestamp origin")
            if item.booked:
                active.append(item)
        capacity_field = (
            "logical_resource_units"
            if self.parameters.capacity_unit == "LOGICAL_RESOURCE"
            else "physical_cell_units"
        )
        usage = sum(getattr(item, capacity_field) for item in active)
        cells = [cell for item in active for cell in item.physical_cell_ids]
        if usage > self.parameters.buffer_capacity or len(cells) != len(set(cells)):
            raise ValueError(
                "booked allocation exceeds capacity or reuses a physical memory cell"
            )

    @property
    def current_buffer_occupancy(self) -> int:
        """Known booked logical resources, never unobserved physical survival."""
        return sum(
            item.logical_resource_units for item in self.resources if item.booked
        )

    def to_policy_dict(self) -> dict[str, Any]:
        """Exclude source paths/JSON and any physical audit outcome from decision input."""
        self.validate()
        return {
            "schema": self.schema,
            "decision_as_of_time": self.as_of,
            "timestamp_origin": self.timestamp_origin,
            "base_abi_sha": self.extension.base_abi_sha,
            "memory_parameter_manifest_sha": self.extension.memory_parameter_sha,
            "current_buffer_occupancy": self.current_buffer_occupancy,
            "resources": [asdict(item) for item in self.resources],
            "parameters": self.parameters.semantic_dict(),
            "memory_loading_parameter": self.parameters.memory_loading_parameter,
        }

    def sha256(self) -> str:
        return _digest(self.to_policy_dict())
