"""Source-bound RF-coordinate to legacy AWG command-table mappings.

This module owns the physical adaptation contract used by the isolated AWG
reproduction adapter.  It deliberately has no generator, filesystem, uploader,
or system-control dependency: a mapping artifact is admitted before an AWG
consumer is allowed to materialize it.
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


RF_TO_AWG_KINDS = frozenset({"AMPLITUDE", "PHASE"})
RF_TO_AWG_AXES = frozenset({"row", "column"})
LEGACY_AWG_CHANNELS = (
    "W_CH1", "W_CH2", "W_CH3", "W_CH4",
    "R_CH1", "R_CH2", "R_CH3", "R_CH4",
)
PHASE_AWG_CHANNELS = ("W_CH2", "W_CH4", "R_CH2", "R_CH4")


def registered_rf_to_awg_recipe_sha(
    rf_to_awg_mapping_sha: str, *, rf_mapping_core_sha: str,
    current_amplitude: Sequence[float], current_phase_rad: Sequence[float],
) -> str:
    """Stable recipe identity shared by research proposal and AWG consumer."""
    require_sha256(rf_to_awg_mapping_sha, "rf_to_awg_mapping_sha")
    require_sha256(rf_mapping_core_sha, "rf_mapping_core_sha")
    return canonical_sha256({
        "schema": "TrackBRFToAWGRecipeBindingV1",
        "rf_to_awg_mapping_sha": rf_to_awg_mapping_sha,
        "rf_mapping_core_sha": rf_mapping_core_sha,
        "current_amplitude": [float(value) for value in current_amplitude],
        "current_phase_rad": [float(value) for value in current_phase_rad],
    })


@dataclass(frozen=True)
class RFToAWGCoordinateBindingV1:
    """One explicit physical coordinate-to-generator-field mapping."""

    coordinate_id: str
    kind: str
    write_read_role: str
    awg_channels: tuple[str, ...]
    grid_axis: str
    grid_index: int
    tone_index: int
    generator_field: str
    unit: str
    application_rule: str
    current_control_source_sha: str
    schema: str = field(default="RFToAWGCoordinateBindingV1", init=False)

    def __post_init__(self) -> None:
        if type(self.coordinate_id) is not str or not self.coordinate_id:
            raise ValueError("RF-to-AWG coordinate_id must be nonempty")
        if self.kind not in RF_TO_AWG_KINDS:
            raise ValueError("RF-to-AWG mapping supports only amplitude or phase coordinates")
        if self.write_read_role not in {"write", "read", "shared"}:
            raise ValueError("RF-to-AWG write/read role is not registered")
        if self.grid_axis not in RF_TO_AWG_AXES:
            raise ValueError("RF-to-AWG grid axis is not registered")
        if type(self.grid_index) is not int or self.grid_index < 0 \
                or type(self.tone_index) is not int or self.tone_index < 0:
            raise ValueError("RF-to-AWG grid/tone indices must be nonnegative native integers")
        if not self.awg_channels or any(channel not in LEGACY_AWG_CHANNELS for channel in self.awg_channels):
            raise ValueError("RF-to-AWG channels must be registered legacy channels")
        if len(set(self.awg_channels)) != len(self.awg_channels):
            raise ValueError("RF-to-AWG channels must be unique per coordinate")
        expected = {
            "AMPLITUDE": (
                "write", ("W_CH1",), "correction_matrix", "log_amplitude", "ADD_LOG_AMPLITUDE",
            ),
            "PHASE": (
                "shared", PHASE_AWG_CHANNELS, "phase_vector", "radian", "ADD_PHASE_RAD",
            ),
        }[self.kind]
        if self.write_read_role != expected[0] or self.awg_channels != expected[1] \
                or self.generator_field != expected[2] or self.unit != expected[3] \
                or self.application_rule != expected[4]:
            raise ValueError("RF-to-AWG binding role/channel/field/unit/rule disagrees with coordinate kind")
        if self.tone_index != self.grid_index:
            raise ValueError("RF-to-AWG tone index must equal the legacy command-table grid index")
        require_sha256(self.current_control_source_sha, "current_control_source_sha")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "coordinate_id": self.coordinate_id,
            "kind": self.kind,
            "write_read_role": self.write_read_role,
            "awg_channels": list(self.awg_channels),
            "grid_axis": self.grid_axis,
            "grid_index": self.grid_index,
            "tone_index": self.tone_index,
            "generator_field": self.generator_field,
            "unit": self.unit,
            "application_rule": self.application_rule,
            "current_control_source_sha": self.current_control_source_sha,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RFToAWGCoordinateBindingV1":
        expected = {
            "schema", "coordinate_id", "kind", "write_read_role", "awg_channels",
            "grid_axis", "grid_index", "tone_index", "generator_field", "unit",
            "application_rule", "current_control_source_sha",
        }
        if set(payload) != expected or payload.get("schema") != "RFToAWGCoordinateBindingV1":
            raise ValueError("RF-to-AWG coordinate binding schema/fields are invalid")
        return cls(
            coordinate_id=str(payload["coordinate_id"]),
            kind=str(payload["kind"]),
            write_read_role=str(payload["write_read_role"]),
            awg_channels=tuple(str(value) for value in payload["awg_channels"]),
            grid_axis=str(payload["grid_axis"]),
            grid_index=int(payload["grid_index"]),
            tone_index=int(payload["tone_index"]),
            generator_field=str(payload["generator_field"]),
            unit=str(payload["unit"]),
            application_rule=str(payload["application_rule"]),
            current_control_source_sha=str(payload["current_control_source_sha"]),
        )


@dataclass(frozen=True)
class RegisteredRFToAWGMappingV1:
    """Immutable command-table/channel map with an injective RF adaptation."""

    system_type: str
    command_table_sha: str
    command_table_file_sha: str
    channel_mapping_file_sha: str
    generator_source_sha: str
    frequency_steps: int
    bindings: tuple[RFToAWGCoordinateBindingV1, ...]
    current_control_snapshot_sha: str
    source_artifact_sha: str
    schema: str = field(default="RegisteredRFToAWGMappingV1", init=False)

    def __post_init__(self) -> None:
        if self.system_type not in {"3qubit", "4qubit"}:
            raise ValueError("RF-to-AWG mapping system type is unsupported")
        expected_steps = 15 if self.system_type == "3qubit" else 16
        if type(self.frequency_steps) is not int or self.frequency_steps != expected_steps:
            raise ValueError("RF-to-AWG mapping frequency steps disagree with the legacy system ABI")
        for name in (
            "command_table_sha", "command_table_file_sha", "channel_mapping_file_sha",
            "generator_source_sha", "current_control_snapshot_sha", "source_artifact_sha",
        ):
            require_sha256(getattr(self, name), name)
        if not self.bindings or len({item.coordinate_id for item in self.bindings}) != len(self.bindings):
            raise ValueError("RF-to-AWG mapping requires one unique binding per RF coordinate")
        if any(type(item) is not RFToAWGCoordinateBindingV1 for item in self.bindings):
            raise TypeError("RF-to-AWG mapping requires exact coordinate binding types")
        if any(
            item.current_control_source_sha != self.current_control_snapshot_sha
            for item in self.bindings
        ):
            raise ValueError(
                "RF-to-AWG binding current-control sources must equal the mapping snapshot"
            )
        if any(item.grid_index >= self.frequency_steps or item.tone_index >= self.frequency_steps for item in self.bindings):
            raise ValueError("RF-to-AWG binding exceeds the registered command-table grid")
        targets = [
            (item.generator_field, item.grid_axis, item.grid_index, item.awg_channels)
            for item in self.bindings
        ]
        if len(set(targets)) != len(targets):
            raise ValueError("RF-to-AWG mapping has duplicate physical generator targets")
        matrix = self.coordinate_to_legacy_matrix()
        rank = int(np.linalg.matrix_rank(matrix, tol=1.0e-12))
        if rank != len(self.bindings):
            raise ValueError("RF-to-AWG mapping has a nonzero coordinate nullspace")
        if self.source_artifact_sha != canonical_sha256(self.source_binding_payload()):
            raise ValueError("RF-to-AWG source artifact SHA does not bind its mapping payload")

    def source_binding_payload(self) -> dict[str, object]:
        return {
            "schema": "RegisteredRFToAWGMappingSourceBindingV1",
            "system_type": self.system_type,
            "command_table_sha": self.command_table_sha,
            "command_table_file_sha": self.command_table_file_sha,
            "channel_mapping_file_sha": self.channel_mapping_file_sha,
            "generator_source_sha": self.generator_source_sha,
            "frequency_steps": self.frequency_steps,
            "bindings": [item.to_dict() for item in self.bindings],
            "current_control_snapshot_sha": self.current_control_snapshot_sha,
        }

    def coordinate_to_legacy_matrix(self) -> np.ndarray:
        """Return the source-bound linear coordinate-to-array adaptation matrix."""
        size = self.frequency_steps
        result = np.zeros((2 * size * size, len(self.bindings)), dtype=np.float64)
        for column, binding in enumerate(self.bindings):
            if binding.grid_axis == "row":
                indices = tuple(binding.grid_index * size + index for index in range(size))
            else:
                indices = tuple(index * size + binding.grid_index for index in range(size))
            offset = 0 if binding.generator_field == "correction_matrix" else size * size
            result[np.asarray(indices, dtype=np.int64) + offset, column] = 1.0
        return result

    def validate_for_core(
        self, *, system_type: str, coordinate_ids: Sequence[str], coordinate_kinds: Sequence[str],
        command_table_sha: str, current_control_snapshot_sha: str,
    ) -> None:
        if self.system_type != system_type or self.command_table_sha != command_table_sha \
                or self.current_control_snapshot_sha != current_control_snapshot_sha:
            raise ValueError("RF-to-AWG mapping does not bind the supplied core source identities")
        if tuple(item.coordinate_id for item in self.bindings) != tuple(coordinate_ids) \
                or tuple(item.kind for item in self.bindings) != tuple(coordinate_kinds):
            raise ValueError("RF-to-AWG mapping coordinate order/kinds differ from the RF core")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "system_type": self.system_type,
            "command_table_sha": self.command_table_sha,
            "command_table_file_sha": self.command_table_file_sha,
            "channel_mapping_file_sha": self.channel_mapping_file_sha,
            "generator_source_sha": self.generator_source_sha,
            "frequency_steps": self.frequency_steps,
            "bindings": [item.to_dict() for item in self.bindings],
            "current_control_snapshot_sha": self.current_control_snapshot_sha,
            "source_artifact_sha": self.source_artifact_sha,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RegisteredRFToAWGMappingV1":
        expected = {
            "schema", "system_type", "command_table_sha", "command_table_file_sha",
            "channel_mapping_file_sha", "generator_source_sha", "frequency_steps",
            "bindings", "current_control_snapshot_sha", "source_artifact_sha",
        }
        if set(payload) != expected or payload.get("schema") != "RegisteredRFToAWGMappingV1":
            raise ValueError("RF-to-AWG mapping schema/fields are invalid")
        return cls(
            system_type=str(payload["system_type"]),
            command_table_sha=str(payload["command_table_sha"]),
            command_table_file_sha=str(payload["command_table_file_sha"]),
            channel_mapping_file_sha=str(payload["channel_mapping_file_sha"]),
            generator_source_sha=str(payload["generator_source_sha"]),
            frequency_steps=int(payload["frequency_steps"]),
            bindings=tuple(
                RFToAWGCoordinateBindingV1.from_dict(dict(item))
                for item in payload["bindings"]
            ),
            current_control_snapshot_sha=str(payload["current_control_snapshot_sha"]),
            source_artifact_sha=str(payload["source_artifact_sha"]),
        )

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "RegisteredRFToAWGMappingV1":
        decoded = parse_canonical_json_object(
            payload,
            schema="RegisteredRFToAWGMappingV1",
            exact_fields=frozenset({
                "schema", "system_type", "command_table_sha", "command_table_file_sha",
                "channel_mapping_file_sha", "generator_source_sha", "frequency_steps",
                "bindings", "current_control_snapshot_sha", "source_artifact_sha",
            }),
        )
        mapping = cls.from_dict(decoded)
        require_exact_canonical_json_round_trip(payload, mapping)
        return mapping


__all__ = [
    "LEGACY_AWG_CHANNELS", "PHASE_AWG_CHANNELS", "RFToAWGCoordinateBindingV1",
    "RegisteredRFToAWGMappingV1", "registered_rf_to_awg_recipe_sha",
]
