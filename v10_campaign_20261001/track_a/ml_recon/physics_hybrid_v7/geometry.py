"""Nominal operation-point frequency geometry without historical-command invention."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from system_control.experiment_control.operation_points import operation_index_components
from system_control.qubit_recon_files.basis_configs import (
    build_3qubit_basis_config,
    build_4qubit_basis_config,
)
from system_control.qubit_recon_files.system_specs import get_spec


GEOMETRY_SCHEMA = "v7_operation_frequency_geometry_v1"
FREQUENCY_CENTER_MHZ = 105.0
FREQUENCY_SCALE_MHZ = 40.0


@dataclass(frozen=True)
class OperationFrequencyGeometryV1:
    system_type: str
    x_tone_frequencies_mhz: np.ndarray
    x_tone_mask: np.ndarray
    x_mean_frequency_mhz: np.ndarray
    y_frequency_mhz: np.ndarray
    cycle_index: np.ndarray
    row_index: np.ndarray
    group_index: np.ndarray
    basis_amplitudes: np.ndarray
    basis_phases_rad: np.ndarray
    basis_labels: Tuple[str, ...]
    geometry_hash: str

    @property
    def operation_count(self) -> int:
        return int(self.x_mean_frequency_mhz.size)

    @property
    def qubit_count(self) -> int:
        return int(self.basis_amplitudes.shape[1])

    def static_features(self) -> Tuple[np.ndarray, Tuple[str, ...]]:
        x = (self.x_mean_frequency_mhz - FREQUENCY_CENTER_MHZ) / FREQUENCY_SCALE_MHZ
        y = (self.y_frequency_mhz - FREQUENCY_CENTER_MHZ) / FREQUENCY_SCALE_MHZ
        maximum_cycle = max(float(np.max(self.cycle_index)), 1.0)
        cycle_phase = 2.0 * np.pi * self.cycle_index / (maximum_cycle + 1.0)
        amplitude_power = np.sum(np.square(self.basis_amplitudes), axis=1)
        phase_sin = np.mean(np.sin(self.basis_phases_rad), axis=1)
        phase_cos = np.mean(np.cos(self.basis_phases_rad), axis=1)
        features = np.stack((
            x,
            y,
            x * y,
            np.square(x),
            np.square(y),
            np.sin(cycle_phase),
            np.cos(cycle_phase),
            np.mean(self.basis_amplitudes, axis=1),
            np.std(self.basis_amplitudes, axis=1),
            amplitude_power,
            phase_sin,
            phase_cos,
        ), axis=1)
        names = (
            "x_frequency", "y_frequency", "x_y_frequency_interaction",
            "x_frequency_squared", "y_frequency_squared",
            "cycle_sin", "cycle_cos", "basis_amplitude_mean",
            "basis_amplitude_std", "basis_amplitude_power",
            "basis_phase_sin_mean", "basis_phase_cos_mean",
        )
        return features.astype(np.float64), names

    def target_measurement_overlap(self, target: Mapping[str, Any]) -> np.ndarray:
        amplitude = np.asarray(target.get("amplitudes", []), dtype=float).reshape(-1)
        phase = np.asarray(target.get("phases_rad", []), dtype=float).reshape(-1)
        if amplitude.shape != (self.qubit_count,) or phase.shape != amplitude.shape:
            raise ValueError("target state does not match operation geometry")
        target_state = amplitude * np.exp(1j * phase)
        target_state /= max(float(np.linalg.norm(target_state)), 1e-12)
        basis_state = self.basis_amplitudes * np.exp(1j * self.basis_phases_rad)
        basis_state /= np.maximum(np.linalg.norm(basis_state, axis=1, keepdims=True), 1e-12)
        return np.square(np.abs(basis_state.conj() @ target_state))

    def coupling_kernel(
        self,
        *,
        x_length_mhz: float,
        y_length_mhz: float,
        same_local_cross_cycle_weight: float,
    ) -> np.ndarray:
        x_distance = np.abs(
            self.x_mean_frequency_mhz[:, None] - self.x_mean_frequency_mhz[None, :]
        )
        y_distance = np.abs(self.y_frequency_mhz[:, None] - self.y_frequency_mhz[None, :])
        radial = np.exp(-0.5 * (
            np.square(x_distance / max(float(x_length_mhz), 1e-6))
            + np.square(y_distance / max(float(y_length_mhz), 1e-6))
        ))
        left_frequency = self.x_tone_frequencies_mhz[:, :, None]
        right_frequency = self.x_tone_frequencies_mhz.T[None, :, :]
        left_mask = self.x_tone_mask[:, :, None]
        right_mask = self.x_tone_mask.T[None, :, :]
        shared_tone = np.any(
            (np.abs(left_frequency - right_frequency) < 1e-9) & left_mask & right_mask,
            axis=1,
        ).astype(float)
        same_row = (self.row_index[:, None] == self.row_index[None, :]).astype(float)
        same_group = (self.group_index[:, None] == self.group_index[None, :]).astype(float)
        same_local = same_row * same_group
        cross_cycle = (self.cycle_index[:, None] != self.cycle_index[None, :]).astype(float)
        kernel = (
            0.55 * radial
            + 0.20 * shared_tone
            + 0.10 * same_row
            + 0.10 * same_group
            + float(same_local_cross_cycle_weight) * same_local * cross_cycle
        )
        np.fill_diagonal(kernel, 0.0)
        return kernel.astype(np.float64)

    def sparse_knn_kernel(self, *, x_length_mhz: float, y_length_mhz: float,
                          same_local_cross_cycle_weight: float, k: int = 16) -> np.ndarray:
        """V86: k-NN sparse variant of coupling_kernel.

        Keeps only the top-k strongest connections per operation,
        reducing interior-operation over-smoothing while preserving
        the physical coupling structure.
        """
        dense = self.coupling_kernel(
            x_length_mhz=x_length_mhz, y_length_mhz=y_length_mhz,
            same_local_cross_cycle_weight=same_local_cross_cycle_weight,
        )
        n = dense.shape[0]
        sparse = np.zeros_like(dense)
        for i in range(n):
            row = dense[i].copy()
            row[i] = -1.0  # exclude self
            if k < n:
                threshold = np.sort(row)[-(k)]
                row[row < threshold] = 0.0
            sparse[i] = row
        np.fill_diagonal(sparse, 0.0)
        return sparse.astype(np.float64)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": GEOMETRY_SCHEMA,
            "system_type": self.system_type,
            "operation_count": self.operation_count,
            "qubit_count": self.qubit_count,
            "geometry_hash": self.geometry_hash,
            "frequency_source": "system_spec_and_operation_index_components",
            "per_channel_historical_command_source": "unavailable_not_invented",
        }


def _basis_entries(system_type: str) -> List[Mapping[str, Any]]:
    if system_type == "3qubit":
        return list(build_3qubit_basis_config()["entries"])
    cycles = build_4qubit_basis_config()["cycles"]
    return [entry for cycle in cycles for entry in cycle["entries"]]


def build_operation_frequency_geometry(system_type: str) -> OperationFrequencyGeometryV1:
    if system_type not in {"3qubit", "4qubit"}:
        raise ValueError("system_type must be 3qubit or 4qubit")
    spec = get_spec(system_type)
    operation_count = int(spec["total_operations"])
    group_size = int(spec["group_size"])
    entries = _basis_entries(system_type)
    if len(entries) != operation_count:
        raise ValueError("basis entry count and operation geometry disagree")
    x_tones = np.zeros((operation_count, group_size), dtype=np.float64)
    x_mask = np.ones_like(x_tones, dtype=bool)
    x_mean = np.zeros(operation_count, dtype=np.float64)
    y_frequency = np.zeros(operation_count, dtype=np.float64)
    cycle_index = np.zeros(operation_count, dtype=np.int64)
    row_index = np.zeros(operation_count, dtype=np.int64)
    group_index = np.zeros(operation_count, dtype=np.int64)
    for operation_index in range(operation_count):
        cycle, _, row, group, columns = operation_index_components(
            system_type, operation_index,
        )
        tones = np.asarray([
            float(spec["freq_start_mhz"]) + float(spec["freq_step_mhz"]) * column
            for column in columns
        ])
        x_tones[operation_index] = tones
        x_mean[operation_index] = float(np.mean(tones))
        y_frequency[operation_index] = (
            float(spec["freq_start_mhz"]) + float(spec["freq_step_mhz"]) * row
        )
        cycle_index[operation_index] = cycle
        row_index[operation_index] = row
        group_index[operation_index] = group
    basis_amplitudes = np.asarray([entry["amp"] for entry in entries], dtype=np.float64)
    basis_phases = np.asarray([entry["phase"] for entry in entries], dtype=np.float64)
    basis_labels = tuple(str(entry["label"]) for entry in entries)
    hash_payload = {
        "schema": GEOMETRY_SCHEMA,
        "system_type": system_type,
        "x_tone_frequencies_mhz": x_tones.tolist(),
        "y_frequency_mhz": y_frequency.tolist(),
        "cycle_index": cycle_index.tolist(),
        "row_index": row_index.tolist(),
        "group_index": group_index.tolist(),
        "basis_amplitudes": basis_amplitudes.tolist(),
        "basis_phases_rad": basis_phases.tolist(),
        "basis_labels": list(basis_labels),
    }
    geometry_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return OperationFrequencyGeometryV1(
        system_type=system_type,
        x_tone_frequencies_mhz=x_tones,
        x_tone_mask=x_mask,
        x_mean_frequency_mhz=x_mean,
        y_frequency_mhz=y_frequency,
        cycle_index=cycle_index,
        row_index=row_index,
        group_index=group_index,
        basis_amplitudes=basis_amplitudes,
        basis_phases_rad=basis_phases,
        basis_labels=basis_labels,
        geometry_hash=geometry_hash,
    )


__all__ = [
    "FREQUENCY_CENTER_MHZ",
    "FREQUENCY_SCALE_MHZ",
    "GEOMETRY_SCHEMA",
    "OperationFrequencyGeometryV1",
    "build_operation_frequency_geometry",
]
