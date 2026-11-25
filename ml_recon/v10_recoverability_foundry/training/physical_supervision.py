"""Synthetic V2 proper supervision for W-control uncertainty outputs."""

from __future__ import annotations

import math
from typing import Mapping

import torch

from ml_recon.physics_hybrid_v10.belief_geometry import belief_residual_details
from ml_recon.physics_hybrid_v10.control_coordinate_layout import (
    validate_interleaved_control_covariance,
)

PHYSICAL_SUPERVISION_SCHEMA = "V10PhysicalSupervisionV3"
JOINT_COVARIANCE_NLL_CONVENTION = (
    "FULL_GAUSSIAN_NLL_MEAN_OVER_VALID_ROWS_V1"
)
PHYSICAL_TARGET_FIELDS = frozenset(
    {
        "target_W_amplitude_error",
        "target_W_phase_error",
        "target_physical_uncertainty_mask",
        "physical_supervision_schema",
    }
)


def _require_bool_mask(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    if (
        not isinstance(mask, torch.Tensor)
        or mask.dtype is not torch.bool
        or mask.shape != (batch_size,)
    ):
        raise ValueError("physical uncertainty mask must be strict bool [batch]")
    return mask


def _masked_gaussian_residual_nll(
    residual: torch.Tensor, variance: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if residual.shape != variance.shape:
        raise ValueError("residual and variance must have identical shapes")
    valid = mask[:, None].expand_as(residual)
    safe_residual = torch.where(valid, residual, torch.zeros_like(residual))
    safe_variance = torch.where(valid, variance, torch.ones_like(variance)).clamp_min(
        1e-6
    )
    score = 0.5 * (safe_variance.log() + safe_residual.square() / safe_variance)
    return score.sum() / valid.sum().to(score.dtype).clamp_min(1.0)


def _masked_squared_residual(
    residual: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    valid = mask[:, None].expand_as(residual)
    safe_residual = torch.where(valid, residual, torch.zeros_like(residual))
    return safe_residual.square().sum() / valid.sum().to(residual.dtype).clamp_min(1.0)


def joint_covariance_gaussian_nll(
    residual: torch.Tensor, predicted_covariance: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Registered full Gaussian NLL, including the dimensional normalizer.

    This is the single numeric convention for registered training, physical E0,
    and held-out calibration. Its value is the full negative log likelihood.
    """
    valid = _require_bool_mask(mask, residual.shape[0])
    validate_interleaved_control_covariance(
        residual, predicted_covariance, valid_mask=valid
    )
    dimension = residual.shape[-1]
    identity = torch.eye(
        dimension, dtype=predicted_covariance.dtype, device=predicted_covariance.device
    ).expand_as(predicted_covariance)
    safe_residual = torch.where(valid[:, None], residual, torch.zeros_like(residual))
    safe_covariance = torch.where(valid[:, None, None], predicted_covariance, identity)
    safe_covariance = 0.5 * (safe_covariance + safe_covariance.transpose(-1, -2))
    chol, info = torch.linalg.cholesky_ex(safe_covariance)
    if bool(torch.any(info[valid] != 0)):
        raise ValueError(
            "predicted joint covariance must be positive definite on valid rows"
        )
    solved = torch.cholesky_solve(safe_residual.unsqueeze(-1), chol).squeeze(-1)
    score = 0.5 * (
        (safe_residual * solved).sum(-1)
        + 2.0 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(-1)
        + dimension * math.log(2.0 * math.pi)
    )
    return (score * valid.to(score.dtype)).sum() / valid.sum().to(
        score.dtype
    ).clamp_min(1.0)


def physical_uncertainty_losses(
    classifier_output: Mapping[str, torch.Tensor], targets: Mapping[str, object]
) -> dict[str, torch.Tensor]:
    """Score W error means/variances in the registered Helmert gauge from synthetic labels."""
    required_output = {
        "amplitude_error",
        "phase_error",
        "R1_re",
        "R1_im",
        "W_amplitude_variance",
        "W_phase_variance",
        "joint_control_covariance",
    }
    required_targets = PHYSICAL_TARGET_FIELDS | {
        "target_R1_re",
        "target_R1_im",
        "mode_mask",
    }
    missing = (required_output - set(classifier_output)) | (required_targets - set(targets))
    if missing:
        raise ValueError(f"physical supervision missing fields: {sorted(missing)}")
    if targets["physical_supervision_schema"] != PHYSICAL_SUPERVISION_SCHEMA:
        raise ValueError("physical supervision schema is not V2")
    amplitude_target = targets["target_W_amplitude_error"]
    phase_target = targets["target_W_phase_error"]
    amplitude_prediction = classifier_output["amplitude_error"]
    phase_prediction = classifier_output["phase_error"]
    if not isinstance(amplitude_target, torch.Tensor) or not isinstance(
        phase_target, torch.Tensor
    ):
        raise ValueError("physical uncertainty targets must be tensors")
    if (
        amplitude_prediction.shape != amplitude_target.shape
        or phase_prediction.shape != phase_target.shape
    ):
        raise ValueError(
            "physical uncertainty means and targets must have identical [batch,N] shapes"
        )
    mask = _require_bool_mask(
        targets["target_physical_uncertainty_mask"], amplitude_prediction.shape[0]
    )
    amplitude_residual = amplitude_prediction - amplitude_target.to(
        amplitude_prediction
    )
    phase_delta = phase_prediction - phase_target.to(phase_prediction)
    phase_residual = torch.atan2(torch.sin(phase_delta), torch.cos(phase_delta))
    amplitude_mean_loss = _masked_squared_residual(amplitude_residual, mask)
    phase_mean_loss = _masked_squared_residual(phase_residual, mask)
    belief = belief_residual_details(
        classifier_output["R1_re"],
        classifier_output["R1_im"],
        targets["target_R1_re"].to(classifier_output["R1_re"]),
        targets["target_R1_im"].to(classifier_output["R1_im"]),
        targets["mode_mask"],
    )
    covariance_mask = mask & belief.valid_mask
    amplitude_variance_loss = _masked_gaussian_residual_nll(
        belief.mode_residual[..., 0],
        classifier_output["W_amplitude_variance"],
        covariance_mask,
    )
    phase_variance_loss = _masked_gaussian_residual_nll(
        belief.mode_residual[..., 1],
        classifier_output["W_phase_variance"],
        covariance_mask,
    )
    joint_loss = joint_covariance_gaussian_nll(
        belief.control_residual,
        classifier_output["joint_control_covariance"],
        covariance_mask,
    )
    return {
        "amplitude_error_mean": amplitude_mean_loss,
        "phase_error_mean": phase_mean_loss,
        "W_amplitude_variance": amplitude_variance_loss,
        "W_phase_variance": phase_variance_loss,
        "joint_control_covariance": joint_loss,
        "physical_uncertainty": (
            amplitude_mean_loss
            + phase_mean_loss
            + amplitude_variance_loss
            + phase_variance_loss
            + joint_loss
        ),
    }


__all__ = [
    "JOINT_COVARIANCE_NLL_CONVENTION",
    "PHYSICAL_SUPERVISION_SCHEMA",
    "PHYSICAL_TARGET_FIELDS",
    "joint_covariance_gaussian_nll",
    "physical_uncertainty_losses",
]
