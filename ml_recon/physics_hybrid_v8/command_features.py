"""Prepared-command feature extraction for the V8 shadow model.

The source is an immutable :class:`PhysicsHybridCommandTableV1`, never the
mutable matrix directory.  Missingness remains feature-local so an absent
timing value cannot erase an otherwise authoritative RF or phase command.
This module only reads and summarizes command facts; it has no hardware-write
authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from ml_recon.physics_hybrid_v5.command_contract import (
    COMMAND_CHANNELS,
    PhysicsHybridCommandTableV1,
)

from .feedback_models import COMMAND_FEATURE_NAMES


COMMAND_FEATURE_SCHEMA = "v8_prepared_command_features_v1"
WRITE_CHANNELS = ("W_CH1", "W_CH2", "W_CH3", "W_CH4")
READ_CHANNELS = ("R_CH1", "R_CH2", "R_CH3", "R_CH4")
WRITE_X_CHANNELS = ("W_CH1", "W_CH3")
READ_X_CHANNELS = ("R_CH1", "R_CH3")
WRITE_Y_CHANNELS = ("W_CH2", "W_CH4")
READ_Y_CHANNELS = ("R_CH2", "R_CH4")
COMMAND_FEATURE_UNITS = (
    "log_relative_amplitude",
    "dimensionless",
    "dimensionless",
    "log_relative_amplitude",
    "dimensionless",
    "dimensionless",
    "relative_amplitude",
    "dimensionless",
    "dimensionless",
    "relative_amplitude",
    "dimensionless",
    "dimensionless",
    "relative_rf_power",
    "relative_rf_power",
    "relative_rf_power",
    "ns",
    "ns",
    "ns",
)


@dataclass(frozen=True)
class PreparedCommandFeatureBatchV1:
    """Per-operation V8 command features and independent availability masks."""

    features: np.ndarray
    feature_mask: np.ndarray
    any_feature_available_mask: np.ndarray
    fully_authoritative_mask: np.ndarray
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        feature = np.asarray(self.features)
        mask = np.asarray(self.feature_mask)
        expected_tail = len(COMMAND_FEATURE_NAMES)
        if feature.ndim != 2 or feature.shape[-1] != expected_tail:
            raise ValueError("prepared command features must be [operation, 18]")
        if mask.shape != feature.shape:
            raise ValueError("prepared command feature mask shape changed")
        if not np.all(np.isfinite(feature)):
            raise ValueError("prepared command features contain non-finite values")
        if not np.all(np.isfinite(mask)) or np.any(mask < 0.0) or np.any(mask > 1.0):
            raise ValueError("prepared command feature mask must lie in [0, 1]")
        for name, value in (
            ("any_feature_available_mask", self.any_feature_available_mask),
            ("fully_authoritative_mask", self.fully_authoritative_mask),
        ):
            array = np.asarray(value)
            if array.shape != (feature.shape[0],):
                raise ValueError(f"{name} shape changed")

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": COMMAND_FEATURE_SCHEMA,
            "operation_count": int(self.features.shape[0]),
            "feature_names": list(COMMAND_FEATURE_NAMES),
            "feature_units": list(COMMAND_FEATURE_UNITS),
            "feature_availability_mean": np.mean(
                np.asarray(self.feature_mask, dtype=np.float64), axis=0
            ).tolist(),
            "any_feature_available_count": int(
                np.sum(np.asarray(self.any_feature_available_mask) > 0.5)
            ),
            "fully_authoritative_count": int(
                np.sum(np.asarray(self.fully_authoritative_mask) > 0.5)
            ),
            "provenance": dict(self.provenance),
        }


def build_prepared_command_features(
    command_table: PhysicsHybridCommandTableV1,
) -> PreparedCommandFeatureBatchV1:
    """Aggregate immutable channel commands into the fixed 18-feature schema."""

    command_table.validate()
    static = command_table.static_arrays
    metadata = command_table.operation_metadata
    masks = command_table.masks
    operation_count = int(np.asarray(metadata["global_index"]).shape[0])
    required_metadata = ("cycle_index", "row_index", "group_index")
    if any(name not in metadata for name in required_metadata):
        raise ValueError("command table lacks operation layout metadata")
    required_static = (
        "x_log_amplitude",
        "x_phase_sin",
        "x_phase_cos",
        "y_amplitude",
        "y_phase_sin",
        "y_phase_cos",
        "per_channel_rf_power",
        "total_rf_power",
        "digital_delay_ns",
        "digital_width_ns",
        "analog_delay_ns",
    )
    if any(name not in static for name in required_static):
        raise ValueError("command table lacks V8 command arrays")
    if "command_missing_mask" not in masks or "timing_missing_mask" not in masks:
        raise ValueError("command table lacks V8 missingness arrays")

    channel_index = {name: index for index, name in enumerate(COMMAND_CHANNELS)}
    channel_groups = {
        "write_x": tuple(channel_index[name] for name in WRITE_X_CHANNELS),
        "read_x": tuple(channel_index[name] for name in READ_X_CHANNELS),
        "write_y": tuple(channel_index[name] for name in WRITE_Y_CHANNELS),
        "read_y": tuple(channel_index[name] for name in READ_Y_CHANNELS),
        "write": tuple(channel_index[name] for name in WRITE_CHANNELS),
        "read": tuple(channel_index[name] for name in READ_CHANNELS),
        "all": tuple(range(len(COMMAND_CHANNELS))),
    }
    feature = np.zeros((operation_count, len(COMMAND_FEATURE_NAMES)), dtype=np.float32)
    feature_mask = np.zeros_like(feature)
    command_missing = np.asarray(masks["command_missing_mask"], dtype=bool)
    timing_missing = np.asarray(masks["timing_missing_mask"], dtype=bool)

    for operation_index in range(operation_count):
        cycle = int(np.asarray(metadata["cycle_index"])[operation_index])
        row = int(np.asarray(metadata["row_index"])[operation_index])
        group = int(np.asarray(metadata["group_index"])[operation_index])
        missing_by_tone_channel = command_missing[cycle, row, group]
        channel_available = 1.0 - np.mean(
            missing_by_tone_channel.astype(np.float64), axis=0
        )

        values_and_masks = []
        for array_name, indices in (
            ("x_log_amplitude", channel_groups["write_x"]),
            ("x_phase_sin", channel_groups["write_x"]),
            ("x_phase_cos", channel_groups["write_x"]),
            ("x_log_amplitude", channel_groups["read_x"]),
            ("x_phase_sin", channel_groups["read_x"]),
            ("x_phase_cos", channel_groups["read_x"]),
        ):
            value = np.asarray(static[array_name])[cycle, row, group][:, indices]
            available = ~missing_by_tone_channel[:, indices]
            values_and_masks.append(_masked_mean(value, available))
        for array_name, indices in (
            ("y_amplitude", channel_groups["write_y"]),
            ("y_phase_sin", channel_groups["write_y"]),
            ("y_phase_cos", channel_groups["write_y"]),
            ("y_amplitude", channel_groups["read_y"]),
            ("y_phase_sin", channel_groups["read_y"]),
            ("y_phase_cos", channel_groups["read_y"]),
        ):
            value = np.asarray(static[array_name])[cycle, row, group][list(indices)]
            available = channel_available[list(indices)] > 0.0
            values_and_masks.append(_masked_mean(value, available))

        per_channel_power = np.asarray(static["per_channel_rf_power"])[
            cycle, row, group
        ]
        total_power = float(np.asarray(static["total_rf_power"])[cycle, row, group])
        total_power_availability = float(np.mean(channel_available))
        write_power = float(np.sum(per_channel_power[list(channel_groups["write"])]))
        read_power = float(np.sum(per_channel_power[list(channel_groups["read"])]))
        write_availability = float(np.mean(channel_available[list(channel_groups["write"])]))
        read_availability = float(np.mean(channel_available[list(channel_groups["read"])]))
        values_and_masks.extend((
            (total_power, total_power_availability),
            (write_power, write_availability),
            (read_power, read_availability),
        ))

        for timing_index, array_name in enumerate(
            ("digital_delay_ns", "digital_width_ns", "analog_delay_ns")
        ):
            value = np.asarray(static[array_name])[cycle, row, group]
            available = ~timing_missing[cycle, row, group, :, timing_index]
            values_and_masks.append(_masked_mean(value, available))

        if len(values_and_masks) != len(COMMAND_FEATURE_NAMES):
            raise AssertionError("V8 prepared command feature contract changed")
        feature[operation_index] = np.asarray(
            [item[0] for item in values_and_masks], dtype=np.float32
        )
        feature_mask[operation_index] = np.asarray(
            [item[1] for item in values_and_masks], dtype=np.float32
        )

    feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
    feature *= (feature_mask > 0.0).astype(np.float32)
    any_available = (np.sum(feature_mask, axis=1) > 0.0).astype(np.float32)
    fully_authoritative = np.all(feature_mask >= 1.0 - 1e-7, axis=1).astype(np.float32)
    provenance_payload = {
        "schema": COMMAND_FEATURE_SCHEMA,
        "source": "immutable_prepared_command_table",
        "command_table_id": command_table.table_id,
        "command_table_hash": command_table.table_hash,
        "operation_registry_hash": command_table.operation_registry_hash,
        "matrix_snapshot_hash": command_table.matrix_snapshot_hash,
        "calibration_snapshot_hash": command_table.calibration_snapshot_hash,
        "awg_state_hash": command_table.awg_state_hash,
        "source_feature_schema_hash": command_table.feature_schema_hash,
        "feature_names": list(COMMAND_FEATURE_NAMES),
        "feature_units": list(COMMAND_FEATURE_UNITS),
    }
    provenance_payload["feature_contract_sha256"] = _stable_json_sha256(
        provenance_payload
    )
    return PreparedCommandFeatureBatchV1(
        features=feature,
        feature_mask=feature_mask,
        any_feature_available_mask=any_available,
        fully_authoritative_mask=fully_authoritative,
        provenance=provenance_payload,
    )


def _masked_mean(value: np.ndarray, available: np.ndarray) -> Tuple[float, float]:
    array = np.asarray(value, dtype=np.float64)
    mask = np.asarray(available, dtype=bool)
    if array.shape != mask.shape:
        raise ValueError("command value and availability shapes differ")
    finite = np.isfinite(array)
    valid = mask & finite
    availability = float(np.mean(valid)) if valid.size else 0.0
    if not np.any(valid):
        return 0.0, 0.0
    return float(np.mean(array[valid])), availability


def _stable_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "COMMAND_FEATURE_SCHEMA",
    "COMMAND_FEATURE_UNITS",
    "PreparedCommandFeatureBatchV1",
    "build_prepared_command_features",
]
