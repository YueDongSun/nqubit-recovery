"""Snapshot-bound compact command tables for Physics Hybrid V5 research.

The contract deliberately accepts an already constructed operation registry.
It never accepts a matrix directory and never reads generated 1 GHz waveform
files.  A table contains cacheable static command state; per-sample targets,
observations, probes, and actions live in :class:`DynamicCommandBatchV1`.

This module is research-only and has no hardware-write authority.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from system_control.experiment_control.operation_points import (
    CHANNELS,
    X_CHANNELS,
    OperationPointSpec,
    operation_registry_hash,
)


COMMAND_SCHEMA = "physics_hybrid_command_table"
COMMAND_SCHEMA_VERSION = 1
COMMAND_TABLE_FORMAT = "npy-directory-v1"
COMMAND_CHANNELS: Tuple[str, ...] = tuple(CHANNELS)


def command_layout(system_type: str) -> Tuple[int, int, int, int, int]:
    """Return ``(operations, cycles, rows, groups, tones)``."""
    if system_type == "3qubit":
        return 75, 1, 15, 5, 3
    if system_type == "4qubit":
        return 256, 4, 16, 4, 4
    raise ValueError(f"unsupported system_type: {system_type!r}")


def _canonical_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind in {"U", "S", "O"}:
        return np.asarray(array, dtype="U")
    if array.dtype.kind == "f":
        return np.asarray(array, dtype=np.float64)
    if array.dtype.kind in {"i", "u"}:
        return np.asarray(array, dtype=np.int64)
    if array.dtype.kind == "b":
        return np.asarray(array, dtype=np.bool_)
    raise TypeError(f"unsupported array dtype {array.dtype}")


def _hash_arrays(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    for name in sorted(arrays):
        array = np.ascontiguousarray(_canonical_array(arrays[name]))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        if array.dtype.kind == "U":
            digest.update("\0".join(array.reshape(-1).tolist()).encode("utf-8"))
        else:
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PhysicsHybridCommandTableV1:
    """Immutable compact representation of one static command context."""

    system_type: str
    operation_registry_hash: str
    matrix_snapshot_hash: str
    calibration_snapshot_hash: str
    awg_state_hash: str
    static_arrays: Mapping[str, np.ndarray]
    operation_metadata: Mapping[str, np.ndarray]
    masks: Mapping[str, np.ndarray]
    schema: str = COMMAND_SCHEMA
    version: int = COMMAND_SCHEMA_VERSION
    table_hash: str = ""

    def __post_init__(self) -> None:
        self.validate()
        expected = self.compute_hash()
        if self.table_hash and self.table_hash != expected:
            raise ValueError("command table hash does not match its contents")
        object.__setattr__(self, "table_hash", expected)

    @property
    def table_id(self) -> str:
        return f"{self.schema}:v{self.version}:{self.table_hash}"

    @property
    def layout(self) -> Tuple[int, int, int, int, int]:
        return command_layout(self.system_type)

    @property
    def command_shape(self) -> Tuple[int, int, int, int]:
        _, cycles, rows, groups, tones = self.layout
        return cycles, rows, groups, tones

    @property
    def feature_schema_hash(self) -> str:
        entries = {
            "static": sorted((key, list(np.asarray(value).shape), str(np.asarray(value).dtype)) for key, value in self.static_arrays.items()),
            "metadata": sorted((key, list(np.asarray(value).shape), str(np.asarray(value).dtype)) for key, value in self.operation_metadata.items()),
            "masks": sorted((key, list(np.asarray(value).shape), str(np.asarray(value).dtype)) for key, value in self.masks.items()),
        }
        return _json_hash(entries)

    def validate(self) -> None:
        if self.schema != COMMAND_SCHEMA or int(self.version) != COMMAND_SCHEMA_VERSION:
            raise ValueError(f"unsupported command schema {self.schema!r} v{self.version}")
        operations, cycles, rows, groups, tones = command_layout(self.system_type)
        expected_shape = (cycles, rows, groups, tones)
        if np.asarray(self.static_arrays.get("absolute_frequency_mhz")).shape != expected_shape:
            raise ValueError(f"absolute_frequency_mhz must have shape {expected_shape}")
        if np.asarray(self.static_arrays.get("basis_amplitude")).shape != expected_shape:
            raise ValueError(f"basis_amplitude must have shape {expected_shape}")
        if np.asarray(self.operation_metadata.get("global_index")).shape != (operations,):
            raise ValueError(f"global_index must have shape ({operations},)")
        if np.asarray(self.masks.get("command_missing_mask")).shape != expected_shape + (len(COMMAND_CHANNELS),):
            raise ValueError("command_missing_mask has the wrong shape")
        for name, value in {**self.static_arrays, **self.operation_metadata, **self.masks}.items():
            array = np.asarray(value)
            if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values")

    def _hash_metadata(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "version": int(self.version),
            "system_type": self.system_type,
            "operation_registry_hash": self.operation_registry_hash,
            "matrix_snapshot_hash": self.matrix_snapshot_hash,
            "calibration_snapshot_hash": self.calibration_snapshot_hash,
            "awg_state_hash": self.awg_state_hash,
            "channels": list(COMMAND_CHANNELS),
        }

    def _all_arrays(self) -> Dict[str, np.ndarray]:
        values: Dict[str, np.ndarray] = {}
        for prefix, source in (("static", self.static_arrays), ("metadata", self.operation_metadata), ("mask", self.masks)):
            for name, array in source.items():
                values[f"{prefix}__{name}"] = _canonical_array(array)
        return values

    def compute_hash(self) -> str:
        return _hash_arrays(self._hash_metadata(), self._all_arrays())

    def manifest(self) -> Dict[str, Any]:
        arrays = self._all_arrays()
        return {
            **self._hash_metadata(),
            "format": COMMAND_TABLE_FORMAT,
            "table_id": self.table_id,
            "table_hash": self.table_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "arrays": {
                name: {"file": f"{name}.npy", "shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in sorted(arrays.items())
            },
        }

    def save(self, root: str | Path) -> Path:
        """Save static arrays exactly once in an mmap-friendly directory."""
        root_path = Path(root)
        table_dir = root_path / self.table_hash
        table_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = table_dir / "command_table.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("table_hash") != self.table_hash:
                raise ValueError(f"conflicting command table at {table_dir}")
            return manifest_path
        for name, array in self._all_arrays().items():
            np.save(table_dir / f"{name}.npy", array, allow_pickle=False)
        manifest_path.write_text(json.dumps(self.manifest(), indent=2, sort_keys=True), encoding="utf-8")
        return manifest_path

    @classmethod
    def load(cls, manifest_path: str | Path, *, mmap_mode: Optional[str] = "r") -> "PhysicsHybridCommandTableV1":
        path = Path(manifest_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        arrays: Dict[str, Dict[str, np.ndarray]] = {"static": {}, "metadata": {}, "mask": {}}
        for combined_name, spec in payload.get("arrays", {}).items():
            prefix, name = combined_name.split("__", 1)
            arrays[prefix][name] = np.load(path.parent / str(spec["file"]), mmap_mode=mmap_mode, allow_pickle=False)
        return cls(
            schema=str(payload["schema"]),
            version=int(payload["version"]),
            system_type=str(payload["system_type"]),
            operation_registry_hash=str(payload["operation_registry_hash"]),
            matrix_snapshot_hash=str(payload["matrix_snapshot_hash"]),
            calibration_snapshot_hash=str(payload["calibration_snapshot_hash"]),
            awg_state_hash=str(payload["awg_state_hash"]),
            static_arrays=arrays["static"],
            operation_metadata=arrays["metadata"],
            masks=arrays["mask"],
            table_hash=str(payload["table_hash"]),
        )


@dataclass(frozen=True)
class CommandTableReferenceV1:
    """The only command-table payload repeated in a sample/shard row."""

    table_index: int
    table_id: str
    table_hash: str
    command_missing: bool = False

    def sample_arrays(self) -> Dict[str, np.ndarray]:
        # Numeric index is tensor-loader safe.  The dataset manifest owns the
        # index -> ID/hash mapping; no static command array is repeated here.
        return {
            "command_table_index": np.asarray(int(self.table_index), dtype=np.int64),
            "command_missing_mask": np.asarray(bool(self.command_missing), dtype=np.bool_),
        }


@dataclass
class DynamicCommandBatchV1:
    """Non-cacheable, per-sample command/observation coordinates."""

    command_table_index: np.ndarray
    target_amplitude_coordinates: Optional[np.ndarray] = None
    target_phase_coordinates: Optional[np.ndarray] = None
    phase_probe: Optional[np.ndarray] = None
    amplitude_action: Optional[np.ndarray] = None
    phase_action: Optional[np.ndarray] = None
    temperature_c: Optional[np.ndarray] = None
    time_s: Optional[np.ndarray] = None
    exposure: Optional[np.ndarray] = None
    counts: Optional[np.ndarray] = None
    masks: Dict[str, np.ndarray] = field(default_factory=dict)

    def as_arrays(self) -> Dict[str, np.ndarray]:
        values: Dict[str, np.ndarray] = {"command_table_index": np.asarray(self.command_table_index, dtype=np.int64)}
        for name in (
            "target_amplitude_coordinates", "target_phase_coordinates", "phase_probe",
            "amplitude_action", "phase_action", "temperature_c", "time_s", "exposure", "counts",
        ):
            value = getattr(self, name)
            if value is not None:
                values[name] = np.asarray(value)
        for name, value in self.masks.items():
            values[f"mask__{name}"] = np.asarray(value, dtype=np.bool_)
        return values


class CommandTableCache:
    """Bounded LRU cache for mmap-backed static command tables."""

    def __init__(self, max_tables: int = 8) -> None:
        if int(max_tables) < 1:
            raise ValueError("max_tables must be positive")
        self.max_tables = int(max_tables)
        self._tables: "OrderedDict[str, PhysicsHybridCommandTableV1]" = OrderedDict()

    def get(self, table_id: str) -> Optional[PhysicsHybridCommandTableV1]:
        table = self._tables.get(str(table_id))
        if table is not None:
            self._tables.move_to_end(str(table_id))
        return table

    def load(self, manifest_path: str | Path) -> PhysicsHybridCommandTableV1:
        path = Path(manifest_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        table_id = str(payload["table_id"])
        cached = self.get(table_id)
        if cached is not None:
            return cached
        table = PhysicsHybridCommandTableV1.load(path, mmap_mode="r")
        self._tables[table.table_id] = table
        self._tables.move_to_end(table.table_id)
        while len(self._tables) > self.max_tables:
            self._tables.popitem(last=False)
        return table

    def close(self) -> None:
        """Release mmap handles deterministically (required on Windows)."""
        for table in self._tables.values():
            for collection in (table.static_arrays, table.operation_metadata, table.masks):
                for value in collection.values():
                    mmap_handle = getattr(value, "_mmap", None)
                    if mmap_handle is not None:
                        mmap_handle.close()
        self._tables.clear()

    def resolve_or_missing(
        self,
        table_id: str,
        table_index: Mapping[str, str | Path],
        *,
        system_type: str,
    ) -> Tuple[Optional[PhysicsHybridCommandTableV1], np.ndarray]:
        """Resolve an exact historical ID or return an all-missing mask.

        There is intentionally no fallback to the current matrix directory.
        """
        _, cycles, rows, groups, tones = command_layout(system_type)
        missing = np.ones((cycles, rows, groups, tones, len(COMMAND_CHANNELS)), dtype=np.bool_)
        cached = self.get(table_id)
        if cached is not None:
            return cached, np.asarray(cached.masks["command_missing_mask"], dtype=np.bool_)
        path = table_index.get(str(table_id))
        if path is None:
            return None, missing
        table = self.load(path)
        if table.table_id != str(table_id):
            raise ValueError("command table index resolved to a different table ID")
        return table, np.asarray(table.masks["command_missing_mask"], dtype=np.bool_)

    @staticmethod
    def broadcast_batch(array: Any, batch_size: int) -> Any:
        """Broadcast static features without material replication when possible."""
        try:
            import torch
        except ImportError:  # pragma: no cover - NumPy branch is normally exercised without torch.
            torch = None
        if torch is not None and isinstance(array, torch.Tensor):
            return array.unsqueeze(0).expand((int(batch_size),) + tuple(array.shape))
        value = np.asarray(array)
        return np.broadcast_to(value, (int(batch_size),) + value.shape)


def build_command_table(
    registry: Sequence[OperationPointSpec],
    *,
    matrix_snapshot_hash: str,
    calibration_snapshot_hash: str,
    awg_state_hash: str,
) -> PhysicsHybridCommandTableV1:
    """Build a compact table solely from a snapshot-bound operation registry."""
    if not registry:
        raise ValueError("registry must not be empty")
    system_type = str(registry[0].system_type)
    operations, cycles, rows, groups, tones = command_layout(system_type)
    if len(registry) != operations:
        raise ValueError(f"{system_type} registry has {len(registry)} operations, expected {operations}")
    if any(item.system_type != system_type for item in registry):
        raise ValueError("registry contains mixed system types")

    shape = (cycles, rows, groups, tones)
    channel_shape = shape + (len(COMMAND_CHANNELS),)
    static: Dict[str, np.ndarray] = {
        "absolute_frequency_mhz": np.zeros(shape, dtype=np.float64),
        "basis_amplitude": np.zeros(shape, dtype=np.float64),
        "basis_phase_sin": np.zeros(shape, dtype=np.float64),
        "basis_phase_cos": np.ones(shape, dtype=np.float64),
        "x_log_amplitude": np.zeros(channel_shape, dtype=np.float64),
        "x_phase_sin": np.zeros(channel_shape, dtype=np.float64),
        "x_phase_cos": np.ones(channel_shape, dtype=np.float64),
        "y_amplitude": np.zeros((cycles, rows, groups, len(COMMAND_CHANNELS)), dtype=np.float64),
        "y_phase_sin": np.zeros((cycles, rows, groups, len(COMMAND_CHANNELS)), dtype=np.float64),
        "y_phase_cos": np.ones((cycles, rows, groups, len(COMMAND_CHANNELS)), dtype=np.float64),
        "y_frequency_mhz": np.zeros((cycles, rows, groups, len(COMMAND_CHANNELS)), dtype=np.float64),
        "digital_delay_ns": np.zeros((cycles, rows, groups, len(COMMAND_CHANNELS)), dtype=np.float64),
        "digital_width_ns": np.zeros((cycles, rows, groups, len(COMMAND_CHANNELS)), dtype=np.float64),
        "analog_delay_ns": np.zeros((cycles, rows, groups, len(COMMAND_CHANNELS)), dtype=np.float64),
        "diffraction_order_sign": np.ones((cycles, rows, groups, len(COMMAND_CHANNELS)), dtype=np.float64),
        "per_channel_rf_power": np.zeros((cycles, rows, groups, len(COMMAND_CHANNELS)), dtype=np.float64),
        "total_rf_power": np.zeros((cycles, rows, groups), dtype=np.float64),
    }
    metadata: Dict[str, np.ndarray] = {
        "global_index": np.zeros((operations,), dtype=np.int64),
        "cycle_index": np.zeros((operations,), dtype=np.int64),
        "local_index": np.zeros((operations,), dtype=np.int64),
        "row_index": np.zeros((operations,), dtype=np.int64),
        "group_index": np.zeros((operations,), dtype=np.int64),
        "column_index": np.zeros(shape, dtype=np.int64),
        "cell_index": np.zeros(shape, dtype=np.int64),
    }
    masks: Dict[str, np.ndarray] = {
        "command_missing_mask": np.zeros(channel_shape, dtype=np.bool_),
        "tone_missing_mask": np.zeros(shape, dtype=np.bool_),
        "timing_missing_mask": np.zeros((cycles, rows, groups, len(COMMAND_CHANNELS), 3), dtype=np.bool_),
    }

    for item in registry:
        index = int(item.global_index)
        cycle, row, group = int(item.cycle_index), int(item.row), int(item.group)
        if index < 0 or index >= operations or cycle >= cycles or row >= rows or group >= groups:
            raise ValueError(f"registry index out of bounds at operation {index}")
        expected_index = cycle * (rows * groups) + row * groups + group
        if index != expected_index:
            raise ValueError(f"registry ordering mismatch at operation {index}; expected {expected_index}")
        metadata["global_index"][index] = index
        metadata["cycle_index"][index] = cycle
        metadata["local_index"][index] = int(item.local_index)
        metadata["row_index"][index] = row
        metadata["group_index"][index] = group
        if len(item.frequencies_mhz) != tones or len(item.cell_indices) != tones:
            raise ValueError(f"operation {index} does not contain {tones} tones")
        static["absolute_frequency_mhz"][cycle, row, group, :] = np.asarray(item.frequencies_mhz, dtype=float)
        metadata["cell_index"][cycle, row, group, :] = np.asarray(item.cell_indices, dtype=np.int64)
        metadata["column_index"][cycle, row, group, :] = np.asarray([coord[1] for coord in item.cell_coordinates], dtype=np.int64)
        basis_amp = np.asarray(item.basis_amplitudes, dtype=float)
        basis_phase = np.asarray(item.basis_phases_rad, dtype=float)
        if basis_amp.shape != (tones,) or basis_phase.shape != (tones,):
            raise ValueError(f"operation {index} basis must have {tones} amplitudes/phases")
        static["basis_amplitude"][cycle, row, group, :] = basis_amp
        static["basis_phase_sin"][cycle, row, group, :] = np.sin(basis_phase)
        static["basis_phase_cos"][cycle, row, group, :] = np.cos(basis_phase)

        for channel_index, channel in enumerate(COMMAND_CHANNELS):
            command = dict(item.channel_commands.get(channel, {}))
            values_amp = list(command.get("amplitudes", []))
            values_phase = list(command.get("phases_rad", []))
            matrix_missing = bool(command.get("matrix_missing", False))
            masks["command_missing_mask"][cycle, row, group, :, channel_index] = matrix_missing
            for timing_index, timing_key in enumerate(("digital_delay_ns", "digital_width_ns", "analog_delay_ns")):
                value = command.get(timing_key)
                masks["timing_missing_mask"][cycle, row, group, channel_index, timing_index] = value is None
                static[timing_key][cycle, row, group, channel_index] = 0.0 if value is None else float(value)
            static["diffraction_order_sign"][cycle, row, group, channel_index] = float(item.diffraction_order_signs.get(channel, 1.0))
            if channel in X_CHANNELS:
                if len(values_amp) != tones or len(values_phase) != tones:
                    matrix_missing = True
                    values_amp = (values_amp + [None] * tones)[:tones]
                    values_phase = (values_phase + [None] * tones)[:tones]
                amp = np.asarray([0.0 if value is None else float(value) for value in values_amp], dtype=float)
                phase = np.asarray([0.0 if value is None else float(value) for value in values_phase], dtype=float)
                missing_tone = np.asarray([a is None or p is None for a, p in zip(values_amp, values_phase)], dtype=bool)
                masks["command_missing_mask"][cycle, row, group, :, channel_index] |= missing_tone
                static["x_log_amplitude"][cycle, row, group, :, channel_index] = np.log(np.maximum(np.abs(amp), 1e-12))
                static["x_phase_sin"][cycle, row, group, :, channel_index] = np.sin(phase)
                static["x_phase_cos"][cycle, row, group, :, channel_index] = np.cos(phase)
                power = float(np.sum(np.square(amp)))
            else:
                amp_value = values_amp[0] if values_amp else None
                phase_value = values_phase[0] if values_phase else None
                missing = matrix_missing or amp_value is None or phase_value is None
                masks["command_missing_mask"][cycle, row, group, :, channel_index] = missing
                amp = 0.0 if amp_value is None else float(amp_value)
                phase = 0.0 if phase_value is None else float(phase_value)
                static["y_amplitude"][cycle, row, group, channel_index] = amp
                static["y_phase_sin"][cycle, row, group, channel_index] = np.sin(phase)
                static["y_phase_cos"][cycle, row, group, channel_index] = np.cos(phase)
                tone_indices = list(command.get("tone_indices", []))
                # The registry frequency grid is fixed and absolute.  Infer its
                # origin/step from the X tones, never by per-sample min-max.
                if tones > 1:
                    step = float(item.frequencies_mhz[1] - item.frequencies_mhz[0])
                    first_col = int(item.cell_coordinates[0][1])
                    origin = float(item.frequencies_mhz[0]) - first_col * step
                    tone_index = int(tone_indices[0]) if tone_indices else row
                    static["y_frequency_mhz"][cycle, row, group, channel_index] = origin + tone_index * step
                power = float(amp * amp)
            static["per_channel_rf_power"][cycle, row, group, channel_index] = power
        static["total_rf_power"][cycle, row, group] = float(np.sum(static["per_channel_rf_power"][cycle, row, group]))
        masks["tone_missing_mask"][cycle, row, group, :] = np.all(
            masks["command_missing_mask"][cycle, row, group, :, :], axis=-1,
        )

    return PhysicsHybridCommandTableV1(
        system_type=system_type,
        operation_registry_hash=operation_registry_hash(registry),
        matrix_snapshot_hash=str(matrix_snapshot_hash),
        calibration_snapshot_hash=str(calibration_snapshot_hash),
        awg_state_hash=str(awg_state_hash),
        static_arrays=static,
        operation_metadata=metadata,
        masks=masks,
    )


__all__ = [
    "COMMAND_SCHEMA",
    "COMMAND_SCHEMA_VERSION",
    "COMMAND_CHANNELS",
    "PhysicsHybridCommandTableV1",
    "CommandTableReferenceV1",
    "DynamicCommandBatchV1",
    "CommandTableCache",
    "build_command_table",
    "command_layout",
]
