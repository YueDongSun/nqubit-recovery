"""Versioned calibration store for operation-point reconstruction.

Values absent from a calibration run stay ``None``.  The store deliberately
does not convert missing values to zero because that would turn unknown optics
into a false physical claim.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from system_control.qubit_recon_files.system_specs import get_spec


CALIBRATION_SCHEMA_VERSION = 2


@dataclass
class CellCalibration:
    signal_efficiency: float | None = None
    retrieval_efficiency: float | None = None
    readout_efficiency: float | None = None
    phase_offset_rad: float | None = None
    phase_uncertainty_rad: float | None = None
    background_rate: float | None = None
    leakage_rate: float | None = None
    dark_rate: float | None = None
    confidence: float = 0.0
    sample_count: int = 0
    source: str = "missing"
    updated_at: float = 0.0


@dataclass
class GroupCalibration:
    optical_path_efficiency: float | None = None
    shared_phase_offset_rad: float | None = None
    total_power_compression: float | None = None
    group_background_rate: float | None = None
    crosstalk_strength: float | None = None
    confidence: float = 0.0


@dataclass
class OperationPointCalibration:
    effective_efficiency: float | None = None
    effective_background_rate: float | None = None
    effective_leakage_rate: float | None = None
    variance_scale: float | None = None
    phase_residual_rad: list[float] = field(default_factory=list)
    amplitude_residual: list[float] = field(default_factory=list)
    predicted_projector_fidelity: float | None = None
    confidence: float = 0.0
    source: str = "missing"
    version: int = CALIBRATION_SCHEMA_VERSION


class OperationCalibrationStore:
    """A shape-checked, serializable calibration store with JSON + NPZ payloads."""

    def __init__(self, system_type: str, *, metadata: Optional[Dict[str, Any]] = None):
        self.system_type = system_type
        self.spec = get_spec(system_type)
        n = int(self.spec["freq_steps"])
        groups = int(self.spec["groups_per_row"])
        ops = int(self.spec["total_operations"])
        self.schema_version = CALIBRATION_SCHEMA_VERSION
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.metadata.setdefault("system_type", system_type)
        self.metadata.setdefault("created_at", time.time())
        self.cells = np.empty((n, n), dtype=object)
        self.cells[:] = [[CellCalibration() for _ in range(n)] for _ in range(n)]
        self.groups = np.empty((n, groups), dtype=object)
        self.groups[:] = [[GroupCalibration() for _ in range(groups)] for _ in range(n)]
        self.operation_points = [OperationPointCalibration() for _ in range(ops)]
        self.cycle_group_residuals = (
            np.empty((4, n, groups), dtype=object) if system_type == "4qubit" else None
        )
        if self.cycle_group_residuals is not None:
            self.cycle_group_residuals[:] = [
                [[GroupCalibration() for _ in range(groups)] for _ in range(n)] for _ in range(4)
            ]

    @property
    def cell_shape(self) -> tuple[int, int]:
        return tuple(self.cells.shape)

    @property
    def group_shape(self) -> tuple[int, int]:
        return tuple(self.groups.shape)

    @property
    def operation_shape(self) -> tuple[int, ...]:
        return (len(self.operation_points),) if self.system_type == "3qubit" else (4, 64)

    def cell(self, row: int, col: int) -> CellCalibration:
        return self.cells[row, col]

    def group(self, row: int, group: int, cycle_index: Optional[int] = None) -> GroupCalibration:
        if cycle_index is not None and self.cycle_group_residuals is not None:
            return self.cycle_group_residuals[cycle_index, row, group]
        return self.groups[row, group]

    def operation(self, global_index: int) -> OperationPointCalibration:
        return self.operation_points[global_index]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "system_type": self.system_type,
            "metadata": self.metadata,
            "cells": _object_grid_json(self.cells),
            "groups": _object_grid_json(self.groups),
            "cycle_group_residuals": None if self.cycle_group_residuals is None else _object_grid_json(self.cycle_group_residuals),
            "operation_points": [asdict(value) for value in self.operation_points],
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "OperationCalibrationStore":
        store = cls(str(payload["system_type"]), metadata=dict(payload.get("metadata", {})))
        store.schema_version = int(payload.get("schema_version", CALIBRATION_SCHEMA_VERSION))
        _populate_grid(store.cells, payload.get("cells", []), CellCalibration)
        _populate_grid(store.groups, payload.get("groups", []), GroupCalibration)
        if store.cycle_group_residuals is not None and payload.get("cycle_group_residuals") is not None:
            _populate_grid(store.cycle_group_residuals, payload["cycle_group_residuals"], GroupCalibration)
        points = payload.get("operation_points", [])
        for index, value in enumerate(points[:len(store.operation_points)]):
            store.operation_points[index] = OperationPointCalibration(**dict(value))
        return store

    def save(self, path: str | Path) -> tuple[Path, Path]:
        """Save human-readable metadata and numeric/missing masks side by side."""
        base = Path(path)
        json_path = base if base.suffix == ".json" else base.with_suffix(".json")
        npz_path = json_path.with_suffix(".npz")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")
        np.savez_compressed(npz_path, **self.numeric_arrays())
        return json_path, npz_path

    @classmethod
    def load(cls, path: str | Path) -> "OperationCalibrationStore":
        json_path = Path(path)
        if json_path.suffix != ".json":
            json_path = json_path.with_suffix(".json")
        return cls.from_json_dict(json.loads(json_path.read_text(encoding="utf-8")))

    def numeric_arrays(self) -> Dict[str, np.ndarray]:
        def field_array(grid: np.ndarray, name: str) -> np.ndarray:
            return np.asarray([[np.nan if getattr(item, name) is None else getattr(item, name) for item in row] for row in grid], dtype=float)
        arrays: Dict[str, np.ndarray] = {
            "cell_retrieval_efficiency": field_array(self.cells, "retrieval_efficiency"),
            "cell_confidence": field_array(self.cells, "confidence"),
            "group_optical_path_efficiency": field_array(self.groups, "optical_path_efficiency"),
            "operation_effective_efficiency": _op_field(self.operation_points, "effective_efficiency"),
            "operation_background_rate": _op_field(self.operation_points, "effective_background_rate"),
            "operation_leakage_rate": _op_field(self.operation_points, "effective_leakage_rate"),
            "operation_confidence": _op_field(self.operation_points, "confidence"),
        }
        if self.cycle_group_residuals is not None:
            arrays["cycle_group_confidence"] = np.asarray(
                [[[item.confidence for item in row] for row in cycle] for cycle in self.cycle_group_residuals], dtype=float,
            )
        return arrays

    def load_retrieval_efficiency_map(self, path: str | Path, *, source: str = "legacy_retrieval_map", confidence: float = 0.5) -> None:
        """Import only the known retrieval field from legacy calibration maps."""
        rates = _load_retrieval_efficiency_map(self.system_type, path)
        if rates.shape != self.cells.shape:
            raise ValueError(f"retrieval map shape {rates.shape} does not match {self.cells.shape}")
        now = time.time()
        for row in range(rates.shape[0]):
            for col in range(rates.shape[1]):
                cell = self.cells[row, col]
                cell.retrieval_efficiency = float(rates[row, col])
                cell.confidence = float(confidence)
                cell.source = source
                cell.updated_at = now
        self.metadata["retrieval_efficiency_source"] = str(path)


def _object_grid_json(values: np.ndarray) -> Any:
    if values.ndim == 2:
        return [[asdict(item) for item in row] for row in values]
    return [[[asdict(item) for item in row] for row in cycle] for cycle in values]


def _populate_grid(target: np.ndarray, payload: Any, cls: Any) -> None:
    values = np.asarray(payload, dtype=object)
    if values.shape != target.shape:
        return
    for index in np.ndindex(target.shape):
        target[index] = cls(**dict(values[index]))


def _op_field(values: Iterable[OperationPointCalibration], name: str) -> np.ndarray:
    return np.asarray([np.nan if getattr(item, name) is None else getattr(item, name) for item in values], dtype=float)


def _load_retrieval_efficiency_map(system_type: str, path: str | Path) -> np.ndarray:
    """Use the legacy parser lazily (it retains script-style imports)."""
    recon_dir = Path(__file__).resolve().parents[1] / "qubit_recon_files"
    if str(recon_dir) not in sys.path:
        sys.path.insert(0, str(recon_dir))
    from calibrated_density_reconstruction import load_efficiency_map
    return load_efficiency_map(system_type, path)
