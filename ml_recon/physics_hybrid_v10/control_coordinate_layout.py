"""Registered interleaved Helmert control-coordinate ABI."""

from __future__ import annotations
import hashlib
import torch

INTERLEAVED_CONTROL_COORDINATE_SCHEMA = "V10InterleavedHelmertLogAmplitudePhaseV1"
INTERLEAVED_CONTROL_LAYOUT_SHA = hashlib.sha256(
    INTERLEAVED_CONTROL_COORDINATE_SCHEMA.encode()
).hexdigest()


def interleave_control_coordinates(
    amplitude: torch.Tensor, phase: torch.Tensor
) -> torch.Tensor:
    """Flatten matched Helmert amplitude/phase coordinates in registered order."""
    if amplitude.shape != phase.shape:
        raise ValueError("amplitude and phase coordinates must align")
    return torch.stack((amplitude, phase), -1).flatten(-2)


def uninterleave_control_coordinates(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover matched amplitude/phase coordinates from registered interleaved order."""
    if values.ndim < 1 or values.shape[-1] < 2 or values.shape[-1] % 2:
        raise ValueError("interleaved control coordinates require a nonzero even dimension")
    paired = values.reshape(*values.shape[:-1], values.shape[-1] // 2, 2)
    return paired[..., 0], paired[..., 1]


def validate_interleaved_control_covariance(
    residual: torch.Tensor,
    covariance: torch.Tensor,
    *,
    layout_sha: str = INTERLEAVED_CONTROL_LAYOUT_SHA,
    valid_mask: torch.Tensor | None = None,
) -> int:
    """Fail closed unless one registered layout owns residual and covariance axes."""
    if layout_sha != INTERLEAVED_CONTROL_LAYOUT_SHA:
        raise ValueError("control residual layout SHA does not match the registered ABI")
    if (
        residual.ndim != 2
        or covariance.ndim != 3
        or covariance.shape[0] != residual.shape[0]
        or covariance.shape[-2:] != (residual.shape[-1], residual.shape[-1])
        or residual.shape[-1] < 2
        or residual.shape[-1] % 2
    ):
        raise ValueError(
            "interleaved residual and covariance require aligned [batch,2D] and [batch,2D,2D] shapes"
        )
    if residual.device != covariance.device or residual.dtype != covariance.dtype:
        raise ValueError("residual and covariance must share device and floating-point dtype")
    if not torch.is_floating_point(residual) or not torch.is_floating_point(covariance):
        raise ValueError("interleaved residual and covariance must be floating point")
    if valid_mask is None:
        finite_residual, finite_covariance = residual, covariance
    else:
        if (
            not isinstance(valid_mask, torch.Tensor)
            or valid_mask.dtype is not torch.bool
            or valid_mask.shape != (residual.shape[0],)
            or valid_mask.device != residual.device
        ):
            raise ValueError("valid_mask must be strict bool [batch] on the residual device")
        finite_residual, finite_covariance = residual[valid_mask], covariance[valid_mask]
    if not torch.isfinite(finite_residual).all() or not torch.isfinite(finite_covariance).all():
        raise ValueError("valid interleaved residual and covariance rows must be finite")
    return residual.shape[-1]


def interleaved_joint_covariance(
    amplitude: torch.Tensor, phase: torch.Tensor
) -> torch.Tensor:
    if amplitude.shape != phase.shape:
        raise ValueError("coordinate covariance shapes must align")
    batch, dimension, _ = amplitude.shape
    result = amplitude.new_zeros(batch, 2 * dimension, 2 * dimension)
    result[:, 0::2, 0::2] = amplitude
    result[:, 1::2, 1::2] = phase
    return result
