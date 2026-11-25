"""Registered control-manifold contracts for V10 CSM geometry.

The CSM layer is deliberately independent from the feedback action answer.  It
describes how a registered logical tangent is embedded in mode-local
coordinates, while the existing feedback renderer remains the source of truth
for finite W actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import numpy as np

from .contracts import sha256_of
from .control_tangent import apply_tangent_action, helmert_matrix


W_LOGAMP_RELATIVE_PHASE_V1 = "W_LOGAMP_RELATIVE_PHASE_V1"
INTERLEAVED_AMP_PHASE = "interleaved_amp_phase"
GROUPED_AMP_PHASE = "grouped_amp_phase"


def _require_n(N: int) -> int:
    value = int(N)
    if value < 3 or value > 256:
        raise ValueError("CSM control manifolds require 3 <= N <= 256")
    return value


def grouped_to_interleaved(values: np.ndarray, N: int | None = None) -> np.ndarray:
    """Convert ``[all amplitude, all phase]`` to interleaved coordinates."""
    array = np.asarray(values)
    inferred = (array.shape[-1] + 2) // 2 if N is None else int(N)
    _require_n(inferred)
    expected = 2 * (inferred - 1)
    if array.shape[-1] != expected:
        raise ValueError(f"expected last dimension {expected}, got {array.shape[-1]}")
    amplitude = array[..., :inferred - 1]
    phase = array[..., inferred - 1:]
    return np.stack((amplitude, phase), axis=-1).reshape(array.shape)


def interleaved_to_grouped(values: np.ndarray, N: int | None = None) -> np.ndarray:
    """Convert interleaved logical coordinates to grouped coordinates."""
    array = np.asarray(values)
    inferred = (array.shape[-1] + 2) // 2 if N is None else int(N)
    _require_n(inferred)
    expected = 2 * (inferred - 1)
    if array.shape[-1] != expected:
        raise ValueError(f"expected last dimension {expected}, got {array.shape[-1]}")
    interleaved = array.reshape(array.shape[:-1] + (inferred - 1, 2))
    return np.concatenate((interleaved[..., :, 0], interleaved[..., :, 1]), axis=-1)


def mode_lift_matrix(N: int) -> np.ndarray:
    """Return ``T_int`` mapping mode-local coordinates to logical coordinates.

    ``T_int`` has shape ``[2N-2, 2N]`` and follows the current physical layout
    ``(a_1, phi_1, ..., a_N, phi_N)``.  Therefore a logical Jacobian uses
    ``J_mode = J_q @ mode_lift_matrix(N)``.
    """
    N = _require_n(N)
    basis = helmert_matrix(N)
    result = np.zeros((2 * N - 2, 2 * N), dtype=np.float64)
    for logical_index in range(N - 1):
        for mode_index in range(N):
            result[2 * logical_index, 2 * mode_index] = basis[logical_index, mode_index]
            result[2 * logical_index + 1, 2 * mode_index + 1] = basis[
                logical_index, mode_index]
    return result


def logical_to_mode_lift_matrix(N: int) -> np.ndarray:
    """Return the transpose mapping logical coordinates to mode-local tangent."""
    return mode_lift_matrix(N).T


def gauge_null_basis(N: int) -> np.ndarray:
    """Return amplitude and phase gauge vectors as columns ``[2N, 2]``."""
    N = _require_n(N)
    amplitude = np.tile(np.asarray((1.0, 0.0)), N)
    phase = np.tile(np.asarray((0.0, 1.0)), N)
    return np.column_stack((amplitude, phase))


def rotated_mode_lift_matrix(N: int, orthogonal_basis_rotation: np.ndarray) -> np.ndarray:
    """Build an interleaved lift after rotating the zero-sum Helmert basis."""
    N = _require_n(N)
    rotation = np.asarray(orthogonal_basis_rotation, dtype=np.float64)
    if rotation.shape != (N - 1, N - 1):
        raise ValueError("basis rotation must have shape [N-1,N-1]")
    if not np.allclose(rotation @ rotation.T, np.eye(N - 1), atol=1e-10):
        raise ValueError("basis rotation must be orthogonal")
    base = mode_lift_matrix(N)
    block = np.zeros((2 * N - 2, 2 * N - 2), dtype=np.float64)
    block[0::2, 0::2] = rotation
    block[1::2, 1::2] = rotation
    return block @ base


class ControlManifold(Protocol):
    """Protocol implemented by registered logical control manifolds."""

    manifold_id: str
    N: int
    logical_dim: int
    mode_local_dim: int
    coordinate_layout: str

    def logical_to_mode_lift(self) -> np.ndarray: ...

    def mode_to_logical_lift(self) -> np.ndarray: ...

    def apply_action(self, coefficients: np.ndarray, logical_action: np.ndarray) -> np.ndarray: ...

    def gauge_null_basis(self) -> np.ndarray: ...

    def validate_action(self, logical_action: np.ndarray, *, bounds: object = None) -> None: ...

    def sha256(self) -> str: ...


@dataclass(frozen=True)
class WLogAmpRelativePhaseManifold:
    """The first registered CSM backend: W relative log-amplitude + phase."""

    N: int
    manifold_id: str = W_LOGAMP_RELATIVE_PHASE_V1
    logical_dim: int = 0
    mode_local_dim: int = 2
    coordinate_layout: str = INTERLEAVED_AMP_PHASE

    def __post_init__(self) -> None:
        N = _require_n(self.N)
        object.__setattr__(self, "N", N)
        object.__setattr__(self, "logical_dim", 2 * N - 2)
        if self.manifold_id != W_LOGAMP_RELATIVE_PHASE_V1:
            raise ValueError("unexpected W manifold id")
        if self.coordinate_layout != INTERLEAVED_AMP_PHASE:
            raise ValueError("W CSM backend requires interleaved_amp_phase")

    def logical_to_mode_lift(self) -> np.ndarray:
        return logical_to_mode_lift_matrix(self.N)

    def mode_to_logical_lift(self) -> np.ndarray:
        return mode_lift_matrix(self.N)

    def gauge_null_basis(self) -> np.ndarray:
        return gauge_null_basis(self.N)

    def validate_action(self, logical_action: np.ndarray, *, bounds: object = None) -> None:
        action = np.asarray(logical_action, dtype=np.float64)
        if action.shape[-1:] != (self.logical_dim,):
            raise ValueError(
                f"logical action must end in {self.logical_dim}, got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("logical action must be finite")
        if bounds is not None:
            lower, upper = bounds
            lower = np.broadcast_to(np.asarray(lower, dtype=np.float64), action.shape)
            upper = np.broadcast_to(np.asarray(upper, dtype=np.float64), action.shape)
            if np.any(lower > upper) or np.any(action < lower) or np.any(action > upper):
                raise ValueError("logical action violates registered bounds")

    def apply_action(self, coefficients: np.ndarray, logical_action: np.ndarray) -> np.ndarray:
        current = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
        if current.shape != (self.N,) or not np.all(np.isfinite(current)):
            raise ValueError("W coefficients must be a finite vector of length N")
        if np.linalg.norm(current) <= 0.0:
            raise ValueError("W coefficients must be non-zero")
        action = np.asarray(logical_action, dtype=np.float64).reshape(-1)
        self.validate_action(action)
        interleaved = action.reshape(self.N - 1, 2)
        return apply_tangent_action(current, interleaved[:, 0], interleaved[:, 1])

    def contract_dict(self) -> Dict[str, object]:
        return {
            "manifold_id": self.manifold_id,
            "N": self.N,
            "logical_dim": self.logical_dim,
            "mode_local_dim": self.mode_local_dim,
            "coordinate_layout": self.coordinate_layout,
            "mode_layout": "a_1,phi_1,...,a_N,phi_N",
            "logical_layout": "q_A1,q_phi1,q_A2,q_phi2,...",
            "gauge_removed": ["global_log_amplitude", "global_phase"],
        }

    def sha256(self) -> str:
        return sha256_of(self.contract_dict())


class ControlManifoldRegistry:
    def __init__(self) -> None:
        self._factories: Dict[str, object] = {}

    def register(self, manifold_id: str, factory: object) -> None:
        if not manifold_id or manifold_id in self._factories:
            raise ValueError("control manifold id must be non-empty and unique")
        self._factories[manifold_id] = factory

    def get(self, manifold_id: str, N: int) -> ControlManifold:
        try:
            factory = self._factories[manifold_id]
        except KeyError as exc:
            raise ValueError(f"unregistered control manifold: {manifold_id}") from exc
        return factory(N)  # type: ignore[misc,operator]

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


CONTROL_MANIFOLD_REGISTRY = ControlManifoldRegistry()
CONTROL_MANIFOLD_REGISTRY.register(
    W_LOGAMP_RELATIVE_PHASE_V1, WLogAmpRelativePhaseManifold)


def get_control_manifold(
    manifold_id: str = W_LOGAMP_RELATIVE_PHASE_V1,
    N: int = 3,
) -> ControlManifold:
    return CONTROL_MANIFOLD_REGISTRY.get(manifold_id, N)


__all__ = [
    "CONTROL_MANIFOLD_REGISTRY", "ControlManifold", "ControlManifoldRegistry",
    "GROUPED_AMP_PHASE", "INTERLEAVED_AMP_PHASE", "W_LOGAMP_RELATIVE_PHASE_V1",
    "WLogAmpRelativePhaseManifold", "get_control_manifold", "gauge_null_basis",
    "grouped_to_interleaved", "interleaved_to_grouped", "logical_to_mode_lift_matrix",
    "mode_lift_matrix", "rotated_mode_lift_matrix",
]
