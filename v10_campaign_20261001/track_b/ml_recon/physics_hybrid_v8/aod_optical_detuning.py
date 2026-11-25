"""Nominal AOD optical-detuning features for the V8 DLCZ forward model.

Historical measured rows do not contain an authoritative loaded AOD operation
point.  This module therefore separates the exact operator-provided frequency
law from the repository-reference feature used only as a static weak prior.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Tuple

import numpy as np

from ml_recon.physics_hybrid_v7.geometry import build_operation_frequency_geometry


AOD_OPTICAL_DETUNING_SCHEMA = "physics_hybrid_v8_aod_optical_detuning_v1"
AOD_REFERENCE_OPERATION_POINT_MHZ = (103.0, 103.0)
WRITE_REFERENCE_DETUNING_MHZ = -10.0
READ_REFERENCE_DETUNING_MHZ = 0.0
AOD_DETUNING_SCALE_MHZ = 40.0
MAX_QUBITS = 4
THREE_QUBIT_AWG_OPERATION_ORDER = (
    "basis_config_entries_11_special_then_64_information_row_major"
)
THREE_QUBIT_AWG_SPECIAL_OPERATION_COUNT = 11
THREE_QUBIT_AWG_INFORMATION_OPERATION_COUNT = 64

AOD_OPTICAL_DETUNING_FEATURE_NAMES = (
    "aod_nominal_reference_operation_point_mask",
    "aod_nominal_write_detuning_mean",
    "aod_nominal_write_detuning_std",
    "aod_nominal_read_detuning_mean",
    "aod_nominal_read_detuning_std",
    *(f"aod_nominal_write_detuning_mode_{index}" for index in range(MAX_QUBITS)),
    *(f"aod_nominal_read_detuning_mode_{index}" for index in range(MAX_QUBITS)),
    *(f"aod_nominal_detuning_mode_mask_{index}" for index in range(MAX_QUBITS)),
)


@dataclass(frozen=True)
class AODOpticalDetuningV1:
    delta_aod_mhz: np.ndarray
    write_optical_detuning_mhz: np.ndarray
    read_optical_detuning_mhz: np.ndarray


def aod_optical_detuning(
    operation_point_mhz: np.ndarray | Tuple[float, float],
    cell_frequency_mhz: np.ndarray | Tuple[float, float],
) -> AODOpticalDetuningV1:
    """Apply the operator-locked write-positive/read-negative frequency law."""

    operation = np.asarray(operation_point_mhz, dtype=np.float64)
    cell = np.asarray(cell_frequency_mhz, dtype=np.float64)
    if operation.shape[-1:] != (2,) or cell.shape[-1:] != (2,):
        raise ValueError("operation point and cell frequency must end in x/y pairs")
    try:
        operation, cell = np.broadcast_arrays(operation, cell)
    except ValueError as exc:
        raise ValueError("operation point and cell frequency cannot broadcast") from exc
    if not np.all(np.isfinite(operation)) or not np.all(np.isfinite(cell)):
        raise ValueError("AOD frequencies must be finite")
    delta = operation.sum(axis=-1) - cell.sum(axis=-1)
    return AODOpticalDetuningV1(
        delta_aod_mhz=delta,
        write_optical_detuning_mhz=WRITE_REFERENCE_DETUNING_MHZ + delta,
        read_optical_detuning_mhz=READ_REFERENCE_DETUNING_MHZ - delta,
    )


def build_nominal_aod_detuning_features(system_type: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build outcome-free nominal features with historical command mask zero.

    The fixed 103/103 MHz operation point is a reproducible sensitivity
    reference, not an assertion about the loaded command of any measured row.
    """

    geometry = build_operation_frequency_geometry(system_type)
    operation_count = geometry.operation_count
    qubit_count = geometry.qubit_count
    cell_frequency = np.stack(
        (
            geometry.x_tone_frequencies_mhz,
            np.broadcast_to(
                geometry.y_frequency_mhz[:, None],
                geometry.x_tone_frequencies_mhz.shape,
            ),
        ),
        axis=-1,
    )
    reference = np.broadcast_to(
        np.asarray(AOD_REFERENCE_OPERATION_POINT_MHZ, dtype=np.float64),
        cell_frequency.shape,
    )
    detuning = aod_optical_detuning(reference, cell_frequency)
    mode_mask = np.asarray(geometry.x_tone_mask, dtype=np.float64)
    denominator = np.maximum(mode_mask.sum(axis=-1), 1.0)
    write_mean = (detuning.write_optical_detuning_mhz * mode_mask).sum(axis=-1) / denominator
    read_mean = (detuning.read_optical_detuning_mhz * mode_mask).sum(axis=-1) / denominator
    write_std = np.sqrt(
        (
            np.square(detuning.write_optical_detuning_mhz - write_mean[:, None])
            * mode_mask
        ).sum(axis=-1)
        / denominator
    )
    read_std = np.sqrt(
        (
            np.square(detuning.read_optical_detuning_mhz - read_mean[:, None])
            * mode_mask
        ).sum(axis=-1)
        / denominator
    )
    padded_write = np.zeros((operation_count, MAX_QUBITS), dtype=np.float64)
    padded_read = np.zeros_like(padded_write)
    padded_mask = np.zeros_like(padded_write)
    padded_write[:, :qubit_count] = (
        detuning.write_optical_detuning_mhz / AOD_DETUNING_SCALE_MHZ
    )
    padded_read[:, :qubit_count] = (
        detuning.read_optical_detuning_mhz / AOD_DETUNING_SCALE_MHZ
    )
    padded_mask[:, :qubit_count] = mode_mask
    features = np.concatenate(
        (
            np.zeros((operation_count, 1), dtype=np.float64),
            (write_mean / AOD_DETUNING_SCALE_MHZ)[:, None],
            (write_std / AOD_DETUNING_SCALE_MHZ)[:, None],
            (read_mean / AOD_DETUNING_SCALE_MHZ)[:, None],
            (read_std / AOD_DETUNING_SCALE_MHZ)[:, None],
            padded_write,
            padded_read,
            padded_mask,
        ),
        axis=-1,
    ).astype(np.float32)
    if features.shape != (operation_count, len(AOD_OPTICAL_DETUNING_FEATURE_NAMES)):
        raise AssertionError("nominal AOD detuning feature width changed")
    audit = {
        "schema": AOD_OPTICAL_DETUNING_SCHEMA,
        "system_type": system_type,
        "geometry_hash": geometry.geometry_hash,
        "reference_operation_point_mhz": list(AOD_REFERENCE_OPERATION_POINT_MHZ),
        "write_reference_detuning_mhz": WRITE_REFERENCE_DETUNING_MHZ,
        "read_reference_detuning_mhz": READ_REFERENCE_DETUNING_MHZ,
        "write_diffraction_order_sign": 1.0,
        "read_diffraction_order_sign": -1.0,
        "historical_operation_point_authoritative": False,
        "feature_role": "nominal_static_sensitivity_prior_not_historical_command",
        "action_or_causal_evidence": False,
        "feature_names": list(AOD_OPTICAL_DETUNING_FEATURE_NAMES),
    }
    if system_type == "3qubit":
        special_labels = tuple(geometry.basis_labels[:THREE_QUBIT_AWG_SPECIAL_OPERATION_COUNT])
        information_labels = tuple(geometry.basis_labels[THREE_QUBIT_AWG_SPECIAL_OPERATION_COUNT:])
        if (
            len(special_labels) != THREE_QUBIT_AWG_SPECIAL_OPERATION_COUNT
            or len(information_labels) != THREE_QUBIT_AWG_INFORMATION_OPERATION_COUNT
            or not all(label.startswith("info_") for label in information_labels)
        ):
            raise ValueError("3Q basis registry no longer matches the AWG 11+64 order")
        audit.update(
            {
                "operation_order_authority": "awg_sequential_operation_index",
                "three_qubit_awg_operation_order": THREE_QUBIT_AWG_OPERATION_ORDER,
                "three_qubit_special_operation_range": [0, 11],
                "three_qubit_information_operation_range": [11, 75],
                "ui_64_plus_11_partition_has_physical_authority": False,
                "three_qubit_information_grid_shape": [4, 4, 4],
                "three_qubit_information_grid_flattening": "row_major_within_bins_11_to_74",
            }
        )
    audit["audit_sha256"] = hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return features, audit


def build_nominal_aod_mode_detuning(
    system_type: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return raw-MHz write/read mode detuning and the structural mode mask."""

    features, audit = build_nominal_aod_detuning_features(system_type)
    name_to_index = {
        name: index for index, name in enumerate(AOD_OPTICAL_DETUNING_FEATURE_NAMES)
    }
    write_indices = [
        name_to_index[f"aod_nominal_write_detuning_mode_{index}"]
        for index in range(MAX_QUBITS)
    ]
    read_indices = [
        name_to_index[f"aod_nominal_read_detuning_mode_{index}"]
        for index in range(MAX_QUBITS)
    ]
    mask_indices = [
        name_to_index[f"aod_nominal_detuning_mode_mask_{index}"]
        for index in range(MAX_QUBITS)
    ]
    write = (
        np.asarray(features[:, write_indices], dtype=np.float64)
        * AOD_DETUNING_SCALE_MHZ
    )
    read = (
        np.asarray(features[:, read_indices], dtype=np.float64)
        * AOD_DETUNING_SCALE_MHZ
    )
    mode_mask = np.asarray(features[:, mask_indices], dtype=np.float64)
    return write, read, mode_mask, audit


__all__ = [
    "AOD_OPTICAL_DETUNING_FEATURE_NAMES",
    "AOD_OPTICAL_DETUNING_SCHEMA",
    "AOD_REFERENCE_OPERATION_POINT_MHZ",
    "READ_REFERENCE_DETUNING_MHZ",
    "WRITE_REFERENCE_DETUNING_MHZ",
    "THREE_QUBIT_AWG_INFORMATION_OPERATION_COUNT",
    "THREE_QUBIT_AWG_OPERATION_ORDER",
    "THREE_QUBIT_AWG_SPECIAL_OPERATION_COUNT",
    "AODOpticalDetuningV1",
    "aod_optical_detuning",
    "build_nominal_aod_detuning_features",
    "build_nominal_aod_mode_detuning",
]
