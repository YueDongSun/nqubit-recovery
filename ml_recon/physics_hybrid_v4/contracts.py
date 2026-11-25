"""Versioned V4 phase-action, phase-probe, and crosstalk-scan contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .phase_geometry import project_phase_offsets


@dataclass
class OperationPhaseActionV1:
    system_type: str
    operation_registry_hash: str
    phase_posterior_mean_rad: np.ndarray
    phase_posterior_std_rad: np.ndarray
    actionable_mask: np.ndarray
    action_limit_rad: float
    cycle_mapping: Dict[str, Any]
    model_hash: str
    checkpoint_hash: str
    feature_schema_hash: str
    gauge_convention: str = "per_operation_zero_mean_helmert"
    schema_version: int = 1

    def __post_init__(self) -> None:
        expected = (75, 3) if self.system_type == "3qubit" else (256, 4) if self.system_type == "4qubit" else None
        if expected is None:
            raise ValueError("OperationPhaseActionV1 supports only 3qubit and 4qubit")
        self.phase_posterior_mean_rad = np.asarray(self.phase_posterior_mean_rad, dtype=float)
        self.phase_posterior_std_rad = np.asarray(self.phase_posterior_std_rad, dtype=float)
        self.actionable_mask = np.asarray(self.actionable_mask, dtype=bool)
        if self.phase_posterior_mean_rad.shape != expected:
            raise ValueError(f"phase_posterior_mean_rad must have shape {expected}")
        if self.phase_posterior_std_rad.shape != expected or self.actionable_mask.shape != expected:
            raise ValueError("posterior std and actionable mask must match the phase mean shape")
        if not np.allclose(self.phase_posterior_mean_rad.mean(axis=1), 0.0, atol=1e-8):
            raise ValueError("phase posterior mean must use the per-operation zero-mean gauge")
        if np.any(self.phase_posterior_std_rad < 0):
            raise ValueError("phase posterior standard deviation must be non-negative")
        if not 0 < float(self.action_limit_rad) <= 0.20:
            raise ValueError("action_limit_rad must be in (0, 0.20]")
        if int(self.schema_version) != 1:
            raise ValueError("OperationPhaseActionV1 schema_version must be 1")

    @property
    def gauge_coordinates(self) -> np.ndarray:
        return project_phase_offsets(self.phase_posterior_mean_rad)

    def bounded_phase_offsets(self) -> np.ndarray:
        values = np.where(self.actionable_mask, self.phase_posterior_mean_rad, 0.0)
        values = np.clip(values, -float(self.action_limit_rad), float(self.action_limit_rad))
        return values - values.mean(axis=1, keepdims=True)

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["phase_posterior_mean_rad"] = self.phase_posterior_mean_rad.tolist()
        payload["phase_posterior_std_rad"] = self.phase_posterior_std_rad.tolist()
        payload["actionable_mask"] = self.actionable_mask.tolist()
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "OperationPhaseActionV1":
        return cls(**dict(payload))


@dataclass
class OperationPhaseProbeV1:
    system_type: str
    operation_registry_hash: str
    plan_id: str
    anchor_id: str
    repeat_index: int
    code_pair_id: str
    pair_sign: int
    amplitude_rad: float
    direction_seed: int
    phase_tensor_path: str
    phase_tensor_shape: list[int]
    phase_tensor_sha256: str
    requested_awg_matrix_hash: str
    actual_awg_matrix_hash: str
    started_at: float
    ended_at: float
    temperature_context: Dict[str, Any] = field(default_factory=dict)
    gauge_convention: str = "per_operation_zero_mean_helmert"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.system_type != "4qubit":
            raise ValueError("the first V4 phase-probe design is 4qubit only")
        if self.pair_sign not in {-1, 0, 1}:
            raise ValueError("pair_sign must be -1, 0, or 1")
        if float(self.amplitude_rad) not in {0.0, 0.02, 0.05, 0.08}:
            raise ValueError("probe amplitude must be 0, 0.02, 0.05, or 0.08 rad")
        if list(self.phase_tensor_shape) != [256, 4]:
            raise ValueError("4Q phase probe tensor must have shape [256, 4]")
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        if int(self.schema_version) != 1:
            raise ValueError("OperationPhaseProbeV1 schema_version must be 1")

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "OperationPhaseProbeV1":
        """Restore a persisted probe contract from its V1 JSON manifest."""
        return cls(**dict(payload))

    def validate_sidecar(self, root: str | Path) -> None:
        path = Path(root) / self.phase_tensor_path
        values = np.load(path, allow_pickle=False)
        if list(values.shape) != self.phase_tensor_shape:
            raise ValueError("phase probe sidecar shape mismatch")
        if _sha256_file(path) != self.phase_tensor_sha256:
            raise ValueError("phase probe sidecar hash mismatch")
        if not np.allclose(np.mean(values, axis=1), 0.0, atol=1e-8):
            raise ValueError("phase probe sidecar violates the zero-mean gauge")
        if self.requested_awg_matrix_hash != self.actual_awg_matrix_hash:
            raise ValueError("requested and realized AWG hashes do not match")


@dataclass
class CrosstalkScanV1:
    scan_id: str
    system_type: str
    frequency_mhz: np.ndarray
    sweep_rate_mhz_per_s: float
    initial_cells: list[int]
    read_cells: list[int]
    counts: Dict[str, np.ndarray]
    temperature_c: Optional[np.ndarray]
    uncertainty: Dict[str, Any]
    operation_registry_hash: Optional[str] = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        self.frequency_mhz = np.asarray(self.frequency_mhz, dtype=float)
        self.counts = {name: np.asarray(value, dtype=float) for name, value in self.counts.items()}
        if self.frequency_mhz.ndim != 1 or self.frequency_mhz.size < 2:
            raise ValueError("frequency_mhz must be a one-dimensional sweep")
        if self.sweep_rate_mhz_per_s <= 0:
            raise ValueError("sweep_rate_mhz_per_s must be positive")
        if not self.initial_cells or not self.read_cells:
            raise ValueError("initial_cells and read_cells are required")
        for name, values in self.counts.items():
            if values.shape[0] != self.frequency_mhz.size:
                raise ValueError(f"count channel {name} does not match frequency sweep length")
        if self.temperature_c is not None:
            self.temperature_c = np.asarray(self.temperature_c, dtype=float)
            if self.temperature_c.shape != self.frequency_mhz.shape:
                raise ValueError("temperature_c must match frequency_mhz")
        if int(self.schema_version) != 1:
            raise ValueError("CrosstalkScanV1 schema_version must be 1")

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["frequency_mhz"] = self.frequency_mhz.tolist()
        payload["counts"] = {name: values.tolist() for name, values in self.counts.items()}
        payload["temperature_c"] = None if self.temperature_c is None else self.temperature_c.tolist()
        return payload


def stable_payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
