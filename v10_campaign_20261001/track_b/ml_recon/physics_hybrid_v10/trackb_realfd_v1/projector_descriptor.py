"""Ideal projector descriptors kept deliberately separate from raw RF commands."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from .contracts import canonical_sha256, require_sha256


def _canonical_global_phase(vector: np.ndarray) -> np.ndarray:
    """Fix a robust descriptor gauge using the largest occupied mode, not mode zero."""
    value = np.asarray(vector, dtype=np.complex128)
    if value.ndim != 1 or value.size < 2 or not np.all(np.isfinite(value)):
        raise ValueError("projector vector must be a finite one-dimensional complex vector")
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError("projector vector must have nonzero norm")
    # Do not repeatedly renormalize an already-normalized serialized vector:
    # floating roundoff would otherwise change its canonical SHA on re-read.
    if abs(norm - 1.0) > 1.0e-12:
        value = value / norm
    magnitudes = np.abs(value)
    # ``argmax`` selects the first registered maximum deterministically; it never
    # assumes that the first physical mode is a safe gauge reference.
    reference = int(np.argmax(magnitudes))
    if magnitudes[reference] <= 1.0e-12:
        raise ValueError("projector vector has no numerically robust gauge reference")
    value = value * np.exp(-1j * np.angle(value[reference]))
    # Projector descriptors are tolerance-bound geometry artifacts rather than
    # raw floating-point wavefunctions.  Quantize the canonical gauge at a much
    # finer level than any registered support tolerance so a global phase does
    # not perturb its source identity through platform trig roundoff.
    value = np.round(np.real(value), decimals=12) + 1j * np.round(np.imag(value), decimals=12)
    value[np.abs(value) < 0.5e-12] = 0.0
    value[reference] = complex(float(np.real(value[reference])), 0.0)
    return value


def _tuple_real(values: Sequence[object], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class IdealProjectorDescriptorV2:
    """Normalized projector geometry and registry identity for one measurement operation."""

    normalized_vector_real: tuple[float, ...]
    normalized_vector_imag: tuple[float, ...]
    projector_diagonal: tuple[float, ...]
    upper_triangle_real: tuple[float, ...]
    upper_triangle_imag: tuple[float, ...]
    active_mode_mask: tuple[bool, ...]
    operation_id: str
    operation_family: str
    cycle_index: int
    within_cycle_index: int
    phase_sign: int
    projector_sha: str
    operation_registry_sha: str
    command_table_sha: str
    schema: str = field(default="IdealProjectorDescriptorV2", init=False)

    def __post_init__(self) -> None:
        n = len(self.normalized_vector_real)
        if n < 2 or len(self.normalized_vector_imag) != n:
            raise ValueError("normalized projector vector must have matching mode axes")
        if len(self.projector_diagonal) != n or len(self.active_mode_mask) != n:
            raise ValueError("projector diagonal/mask must align to mode count")
        expected_upper = n * (n - 1) // 2
        if len(self.upper_triangle_real) != expected_upper or len(self.upper_triangle_imag) != expected_upper:
            raise ValueError("projector upper triangle has the wrong size")
        real = _tuple_real(self.normalized_vector_real, "normalized_vector_real")
        imag = _tuple_real(self.normalized_vector_imag, "normalized_vector_imag")
        vector = _canonical_global_phase(np.asarray(real) + 1j * np.asarray(imag))
        diagonal = _tuple_real(self.projector_diagonal, "projector_diagonal")
        upper_real = _tuple_real(self.upper_triangle_real, "upper_triangle_real")
        upper_imag = _tuple_real(self.upper_triangle_imag, "upper_triangle_imag")
        if any(type(value) is not bool for value in self.active_mode_mask):
            raise TypeError("active_mode_mask requires exact booleans")
        projector = np.outer(vector, vector.conj())
        upper_indices = np.triu_indices(n, k=1)
        if not (
            np.allclose(diagonal, np.real(np.diag(projector)), atol=1e-12, rtol=0.0)
            and np.allclose(upper_real, np.real(projector[upper_indices]), atol=1e-12, rtol=0.0)
            and np.allclose(upper_imag, np.imag(projector[upper_indices]), atol=1e-12, rtol=0.0)
        ):
            raise ValueError("projector components do not match the normalized vector")
        if tuple(bool(abs(value) > 1.0e-12) for value in vector) != self.active_mode_mask:
            raise ValueError("active-mode mask does not match normalized projector support")
        if type(self.operation_id) is not str or not self.operation_id:
            raise ValueError("operation_id must be a nonempty string")
        if type(self.operation_family) is not str or not self.operation_family:
            raise ValueError("operation_family must be a nonempty string")
        if type(self.cycle_index) is not int or self.cycle_index < 0:
            raise ValueError("cycle_index must be a nonnegative integer")
        if type(self.within_cycle_index) is not int or self.within_cycle_index < 0:
            raise ValueError("within_cycle_index must be a nonnegative integer")
        if self.phase_sign not in (-1, 1):
            raise ValueError("phase_sign must be -1 or 1")
        require_sha256(self.operation_registry_sha, "operation_registry_sha")
        require_sha256(self.command_table_sha, "command_table_sha")
        expected_sha = canonical_sha256(self._projector_payload(vector))
        if self.projector_sha != expected_sha:
            raise ValueError("projector_sha does not bind normalized projector geometry")
        object.__setattr__(self, "normalized_vector_real", tuple(float(value) for value in np.real(vector)))
        object.__setattr__(self, "normalized_vector_imag", tuple(float(value) for value in np.imag(vector)))

    def _projector_payload(self, vector: np.ndarray | None = None) -> dict[str, object]:
        if vector is None:
            vector = np.asarray(self.normalized_vector_real) + 1j * np.asarray(self.normalized_vector_imag)
        projector = np.outer(vector, vector.conj())
        indices = np.triu_indices(vector.size, k=1)
        return {
            "schema": "IdealProjectorGeometryV2",
            "normalized_vector_real": [float(value) for value in np.real(vector)],
            "normalized_vector_imag": [float(value) for value in np.imag(vector)],
            "projector_diagonal": [float(value) for value in np.real(np.diag(projector))],
            "upper_triangle_real": [float(value) for value in np.real(projector[indices])],
            "upper_triangle_imag": [float(value) for value in np.imag(projector[indices])],
            "active_mode_mask": [bool(abs(value) > 1.0e-12) for value in vector],
        }

    @classmethod
    def from_complex_vector(
        cls,
        vector: Sequence[complex],
        *,
        operation_id: str,
        operation_family: str,
        cycle_index: int,
        within_cycle_index: int,
        phase_sign: int,
        operation_registry_sha: str,
        command_table_sha: str,
    ) -> "IdealProjectorDescriptorV2":
        canonical = _canonical_global_phase(np.asarray(vector, dtype=np.complex128))
        projector = np.outer(canonical, canonical.conj())
        indices = np.triu_indices(canonical.size, k=1)
        payload = {
            "schema": "IdealProjectorGeometryV2",
            "normalized_vector_real": [float(value) for value in np.real(canonical)],
            "normalized_vector_imag": [float(value) for value in np.imag(canonical)],
            "projector_diagonal": [float(value) for value in np.real(np.diag(projector))],
            "upper_triangle_real": [float(value) for value in np.real(projector[indices])],
            "upper_triangle_imag": [float(value) for value in np.imag(projector[indices])],
            "active_mode_mask": [bool(abs(value) > 1.0e-12) for value in canonical],
        }
        return cls(
            normalized_vector_real=tuple(payload["normalized_vector_real"]),
            normalized_vector_imag=tuple(payload["normalized_vector_imag"]),
            projector_diagonal=tuple(payload["projector_diagonal"]),
            upper_triangle_real=tuple(payload["upper_triangle_real"]),
            upper_triangle_imag=tuple(payload["upper_triangle_imag"]),
            active_mode_mask=tuple(payload["active_mode_mask"]),
            operation_id=operation_id,
            operation_family=operation_family,
            cycle_index=cycle_index,
            within_cycle_index=within_cycle_index,
            phase_sign=phase_sign,
            projector_sha=canonical_sha256(payload),
            operation_registry_sha=operation_registry_sha,
            command_table_sha=command_table_sha,
        )

    @property
    def mode_count(self) -> int:
        return len(self.normalized_vector_real)

    def complex_vector(self) -> np.ndarray:
        return np.asarray(self.normalized_vector_real) + 1j * np.asarray(self.normalized_vector_imag)

    def projector(self) -> np.ndarray:
        vector = self.complex_vector()
        return np.outer(vector, vector.conj())

    def permute_modes(self, permutation: Sequence[int]) -> "IdealProjectorDescriptorV2":
        index = tuple(int(value) for value in permutation)
        if sorted(index) != list(range(self.mode_count)):
            raise ValueError("mode permutation must contain every registered mode exactly once")
        return self.from_complex_vector(
            self.complex_vector()[list(index)],
            operation_id=self.operation_id,
            operation_family=self.operation_family,
            cycle_index=self.cycle_index,
            within_cycle_index=self.within_cycle_index,
            phase_sign=self.phase_sign,
            operation_registry_sha=self.operation_registry_sha,
            command_table_sha=self.command_table_sha,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "normalized_vector_real": list(self.normalized_vector_real),
            "normalized_vector_imag": list(self.normalized_vector_imag),
            "projector_diagonal": list(self.projector_diagonal),
            "upper_triangle_real": list(self.upper_triangle_real),
            "upper_triangle_imag": list(self.upper_triangle_imag),
            "active_mode_mask": list(self.active_mode_mask),
            "operation_id": self.operation_id,
            "operation_family": self.operation_family,
            "cycle_index": self.cycle_index,
            "within_cycle_index": self.within_cycle_index,
            "phase_sign": self.phase_sign,
            "projector_sha": self.projector_sha,
            "operation_registry_sha": self.operation_registry_sha,
            "command_table_sha": self.command_table_sha,
        }


__all__ = ["IdealProjectorDescriptorV2"]
