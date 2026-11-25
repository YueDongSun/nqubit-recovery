"""Joint relative-amplitude and relative-phase projective action geometry.

Brightness is a separate, disabled-by-default coordinate.  Relative amplitude
uses the simplex/Aitchison chart and phase uses the existing Helmert gauge.
This module is research-only and does not materialize or write AWG matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ml_recon.physics_hybrid_v4.phase_geometry import helmert_matrix

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


ACTION_GEOMETRY_SCHEMA = "joint_projective_action"
ACTION_GEOMETRY_VERSION = 1


def qubit_count(system_type: str) -> int:
    if system_type == "3qubit":
        return 3
    if system_type == "4qubit":
        return 4
    raise ValueError(f"unsupported system_type: {system_type!r}")


def action_dimensions(system_type: str) -> Dict[str, int]:
    modes = qubit_count(system_type) - 1
    return {"amplitude": modes, "phase": modes, "total": 2 * modes}


def _is_torch(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _helmert_like(cell_count: int, reference: Any) -> Any:
    basis = helmert_matrix(cell_count)
    if _is_torch(reference):
        return torch.as_tensor(basis, dtype=reference.dtype, device=reference.device)
    return basis


def amplitude_probabilities(amplitudes: Any, *, epsilon: float = 1e-12) -> Any:
    """Return ``p_j=a_j^2/sum_k a_k^2`` with strict positivity."""
    if _is_torch(amplitudes):
        value = amplitudes.abs().clamp_min(float(epsilon))
        power = value.square()
        return power / power.sum(dim=-1, keepdim=True).clamp_min(float(epsilon))
    value = np.maximum(np.abs(np.asarray(amplitudes, dtype=float)), float(epsilon))
    power = np.square(value)
    return power / np.maximum(np.sum(power, axis=-1, keepdims=True), float(epsilon))


def amplitude_to_coordinates(amplitudes: Any, *, epsilon: float = 1e-12) -> Any:
    probability = amplitude_probabilities(amplitudes, epsilon=epsilon)
    basis = _helmert_like(probability.shape[-1], probability)
    if _is_torch(probability):
        return torch.matmul(torch.log(probability.clamp_min(float(epsilon))), basis)
    return np.log(np.maximum(probability, float(epsilon))) @ basis


def coordinates_to_amplitude(coordinates: Any) -> Any:
    cell_count = coordinates.shape[-1] + 1
    basis = _helmert_like(cell_count, coordinates)
    if _is_torch(coordinates):
        logit = torch.matmul(coordinates, basis.T)
        return torch.sqrt(torch.softmax(logit, dim=-1))
    logit = np.asarray(coordinates, dtype=float) @ basis.T
    logit -= np.max(logit, axis=-1, keepdims=True)
    probability = np.exp(logit)
    probability /= np.sum(probability, axis=-1, keepdims=True)
    return np.sqrt(probability)


def phase_to_coordinates(phases_rad: Any) -> Any:
    basis = _helmert_like(phases_rad.shape[-1], phases_rad)
    if _is_torch(phases_rad):
        return torch.matmul(phases_rad, basis)
    return np.asarray(phases_rad, dtype=float) @ basis


def coordinates_to_phase(coordinates: Any) -> Any:
    basis = _helmert_like(coordinates.shape[-1] + 1, coordinates)
    if _is_torch(coordinates):
        return torch.matmul(coordinates, basis.T)
    return np.asarray(coordinates, dtype=float) @ basis.T


def wrap_phase(value: Any) -> Any:
    if _is_torch(value):
        return torch.atan2(torch.sin(value), torch.cos(value))
    return np.angle(np.exp(1j * np.asarray(value, dtype=float)))


def bounded_log_amplitude_residual(raw: Any, *, amplitude_limit: float = 0.03) -> Any:
    limit = float(np.log1p(amplitude_limit))
    return limit * (torch.tanh(raw) if _is_torch(raw) else np.tanh(np.asarray(raw, dtype=float)))


def bounded_phase_residual(raw: Any, *, phase_limit_rad: float = 0.20) -> Any:
    return float(phase_limit_rad) * (torch.tanh(raw) if _is_torch(raw) else np.tanh(np.asarray(raw, dtype=float)))


@dataclass(frozen=True)
class ProjectiveAction:
    system_type: str
    amplitude_coordinates: Any
    phase_coordinates: Any
    brightness_delta: Any = 0.0
    brightness_enabled: bool = False
    shared_spatial_amplitude_residual: Optional[Any] = None
    cycle_amplitude_residual: Optional[Any] = None
    shared_spatial_phase_residual: Optional[Any] = None
    cycle_phase_residual: Optional[Any] = None
    amplitude_actionable_mask: Optional[Any] = None
    phase_actionable_mask: Optional[Any] = None
    schema: str = ACTION_GEOMETRY_SCHEMA
    version: int = ACTION_GEOMETRY_VERSION

    def __post_init__(self) -> None:
        dimensions = action_dimensions(self.system_type)
        if self.amplitude_coordinates.shape[-1] != dimensions["amplitude"]:
            raise ValueError("amplitude coordinate dimension mismatch")
        if self.phase_coordinates.shape[-1] != dimensions["phase"]:
            raise ValueError("phase coordinate dimension mismatch")
        brightness = self.brightness_delta
        nonzero = bool(torch.any(torch.as_tensor(brightness) != 0).item()) if torch is not None and _is_torch(brightness) else bool(np.any(np.asarray(brightness) != 0))
        if nonzero and not self.brightness_enabled:
            raise ValueError("brightness action is disabled; nonzero brightness fails closed")

    @property
    def coordinates(self) -> Any:
        if _is_torch(self.amplitude_coordinates):
            return torch.cat((self.amplitude_coordinates, self.phase_coordinates), dim=-1)
        return np.concatenate((np.asarray(self.amplitude_coordinates), np.asarray(self.phase_coordinates)), axis=-1)

    def diagnostics(self) -> Dict[str, Any]:
        def norm(value: Optional[Any]) -> float:
            if value is None:
                return 0.0
            if _is_torch(value):
                return float(torch.linalg.vector_norm(value).detach().cpu())
            return float(np.linalg.norm(np.asarray(value, dtype=float)))
        return {
            "schema": self.schema,
            "version": int(self.version),
            "amplitude_coordinate_norm": norm(self.amplitude_coordinates),
            "phase_coordinate_norm": norm(self.phase_coordinates),
            "brightness_norm": norm(self.brightness_delta),
            "shared_spatial_amplitude_residual_norm": norm(self.shared_spatial_amplitude_residual),
            "cycle_amplitude_residual_norm": norm(self.cycle_amplitude_residual),
            "shared_spatial_phase_residual_norm": norm(self.shared_spatial_phase_residual),
            "cycle_phase_residual_norm": norm(self.cycle_phase_residual),
        }


def project_action(
    system_type: str,
    raw_amplitude_coordinates: Any,
    raw_phase_coordinates: Any,
    *,
    amplitude_limit: float = 0.03,
    phase_limit_rad: float = 0.20,
    raw_brightness: Any = 0.0,
    brightness_enabled: bool = False,
    brightness_limit: float = 0.0,
) -> ProjectiveAction:
    """Bound every action coordinate; invalid brightness fails closed."""
    amp = bounded_log_amplitude_residual(raw_amplitude_coordinates, amplitude_limit=amplitude_limit)
    phase = bounded_phase_residual(raw_phase_coordinates, phase_limit_rad=phase_limit_rad)
    if brightness_enabled:
        brightness = float(brightness_limit) * (torch.tanh(raw_brightness) if _is_torch(raw_brightness) else np.tanh(np.asarray(raw_brightness, dtype=float)))
    else:
        brightness = torch.zeros_like(raw_brightness) if _is_torch(raw_brightness) else np.zeros_like(np.asarray(raw_brightness, dtype=float))
    return ProjectiveAction(
        system_type=system_type,
        amplitude_coordinates=amp,
        phase_coordinates=phase,
        brightness_delta=brightness,
        brightness_enabled=brightness_enabled,
    )


def apply_projective_action(
    base_amplitude: Any,
    base_phase_rad: Any,
    action: ProjectiveAction,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Apply relative projective coordinates while keeping brightness separate."""
    expected_cells = qubit_count(action.system_type)
    if base_amplitude.shape[-1] != expected_cells or base_phase_rad.shape[-1] != expected_cells:
        raise ValueError("base state cell count does not match system_type")
    base_amp_coordinate = amplitude_to_coordinates(base_amplitude)
    base_phase_coordinate = phase_to_coordinates(base_phase_rad)
    amplitude = coordinates_to_amplitude(base_amp_coordinate + action.amplitude_coordinates)
    phase = wrap_phase(coordinates_to_phase(base_phase_coordinate + action.phase_coordinates))
    if action.brightness_enabled:
        brightness = torch.exp(action.brightness_delta) if _is_torch(action.brightness_delta) else np.exp(action.brightness_delta)
        while brightness.ndim < amplitude.ndim:
            brightness = brightness[..., None]
        amplitude = amplitude * brightness
    diagnostics = action.diagnostics()
    diagnostics.update({
        "relative_amplitude_normalized": True,
        "brightness_separate": True,
    })
    return amplitude, phase, diagnostics


def materialize_local_hardware_residuals(
    action: ProjectiveAction,
    *,
    shared_spatial_amplitude_raw: Any,
    cycle_amplitude_raw: Any,
    shared_spatial_phase_raw: Any,
    cycle_phase_raw: Any,
    amplitude_limit: float = 0.03,
    phase_limit_rad: float = 0.20,
) -> Dict[str, Any]:
    """Keep global projective action and local hardware residuals separate."""
    global_amp = coordinates_to_phase(action.amplitude_coordinates)
    global_phase = coordinates_to_phase(action.phase_coordinates)
    shared_amp = bounded_log_amplitude_residual(shared_spatial_amplitude_raw, amplitude_limit=amplitude_limit)
    cycle_amp = bounded_log_amplitude_residual(cycle_amplitude_raw, amplitude_limit=amplitude_limit)
    shared_phase = bounded_phase_residual(shared_spatial_phase_raw, phase_limit_rad=phase_limit_rad)
    cycle_phase = bounded_phase_residual(cycle_phase_raw, phase_limit_rad=phase_limit_rad)
    return {
        "global_projective_amplitude_log_gain": global_amp,
        "global_projective_phase_rad": global_phase,
        "shared_spatial_amplitude_log_gain": shared_amp,
        "cycle_amplitude_log_gain": cycle_amp,
        "shared_spatial_phase_rad": shared_phase,
        "cycle_phase_rad": cycle_phase,
        "delta_log_amp": global_amp + shared_amp + cycle_amp,
        "delta_phase": global_phase + shared_phase + cycle_phase,
    }


def action_covariance_from_lower_triangular(raw: Any, system_type: str, *, minimum_diagonal: float = 1e-5) -> Any:
    dimension = action_dimensions(system_type)["total"]
    expected = dimension * (dimension + 1) // 2
    if raw.shape[-1] != expected:
        raise ValueError(f"lower-triangular covariance head must emit {expected} values")
    if _is_torch(raw):
        lower = torch.zeros(raw.shape[:-1] + (dimension, dimension), dtype=raw.dtype, device=raw.device)
        indices = torch.tril_indices(dimension, dimension, device=raw.device)
        lower[..., indices[0], indices[1]] = raw
        diagonal = torch.arange(dimension, device=raw.device)
        lower[..., diagonal, diagonal] = torch.nn.functional.softplus(lower[..., diagonal, diagonal]) + float(minimum_diagonal)
        return lower @ lower.transpose(-1, -2)
    lower = np.zeros(np.asarray(raw).shape[:-1] + (dimension, dimension), dtype=float)
    indices = np.tril_indices(dimension)
    lower[..., indices[0], indices[1]] = np.asarray(raw, dtype=float)
    diagonal = np.arange(dimension)
    lower[..., diagonal, diagonal] = np.log1p(np.exp(lower[..., diagonal, diagonal])) + float(minimum_diagonal)
    return lower @ np.swapaxes(lower, -1, -2)


def permute_projective_coordinates(
    amplitude_coordinates: Any,
    phase_coordinates: Any,
    permutation: Sequence[int],
) -> Tuple[Any, Any]:
    """Apply a qubit permutation through physical amplitudes/phases."""
    amplitude = coordinates_to_amplitude(amplitude_coordinates)
    phase = coordinates_to_phase(phase_coordinates)
    if _is_torch(amplitude):
        index = torch.as_tensor(permutation, dtype=torch.long, device=amplitude.device)
        return amplitude_to_coordinates(amplitude.index_select(-1, index)), phase_to_coordinates(phase.index_select(-1, index))
    index = np.asarray(permutation, dtype=int)
    return amplitude_to_coordinates(amplitude[..., index]), phase_to_coordinates(phase[..., index])


__all__ = [
    "ProjectiveAction", "action_dimensions", "amplitude_probabilities",
    "amplitude_to_coordinates", "coordinates_to_amplitude", "phase_to_coordinates",
    "coordinates_to_phase", "project_action", "apply_projective_action",
    "materialize_local_hardware_residuals", "action_covariance_from_lower_triangular",
    "permute_projective_coordinates", "bounded_log_amplitude_residual",
    "bounded_phase_residual", "ACTION_GEOMETRY_SCHEMA", "ACTION_GEOMETRY_VERSION",
]
