"""Operation-bound excitation inputs for opt-in synthetic count modeling."""

from __future__ import annotations

import torch


OPERATION_EXCITATION_SCHEMA = "V10OperationExcitationContractV1"
OPERATION_EXCITATION_MODE = "operation_poisson_v1"
MISSING_EXPOSURE_CONVENTION = "UNIT_NORMALIZED_EXPOSURE_DIAGNOSTIC_ONLY"


def validate_operation_excitation_mean(
    operation_excitation_mean: torch.Tensor,
    *,
    batch_size: int,
    operation_count: int,
) -> torch.Tensor:
    """Require source-bound nonnegative [batch, operation] Poisson means."""
    if (
        not isinstance(operation_excitation_mean, torch.Tensor)
        or operation_excitation_mean.shape != (batch_size, operation_count)
        or not torch.is_floating_point(operation_excitation_mean)
        or not bool(torch.isfinite(operation_excitation_mean).all())
        or bool((operation_excitation_mean < 0).any())
    ):
        raise ValueError(
            "operation excitation mean must be finite nonnegative [batch,operation]"
        )
    return operation_excitation_mean


def effective_operation_excitation_mean(
    operation_excitation_rate: torch.Tensor,
    operation_exposure_per_opportunity: torch.Tensor,
    operation_exposure_mask: torch.Tensor,
    operation_mask: torch.Tensor,
) -> torch.Tensor:
    """Construct ``mu_o = E_o * lambda_o`` with a declared missing-E convention."""
    if (
        operation_excitation_rate.ndim != 2
        or operation_exposure_per_opportunity.shape != operation_excitation_rate.shape
        or operation_exposure_mask.shape != operation_excitation_rate.shape
        or operation_mask.shape != operation_excitation_rate.shape
        or operation_exposure_mask.dtype is not torch.bool
        or operation_mask.dtype is not torch.bool
        or not torch.is_floating_point(operation_excitation_rate)
        or not torch.is_floating_point(operation_exposure_per_opportunity)
        or not bool(torch.isfinite(operation_excitation_rate).all())
        or not bool(torch.isfinite(operation_exposure_per_opportunity).all())
        or bool((operation_excitation_rate < 0).any())
        or bool(operation_exposure_mask.logical_and(~operation_mask).any())
        or bool(
            (operation_exposure_per_opportunity[operation_exposure_mask] <= 0).any()
        )
    ):
        raise ValueError(
            "operation rate/exposure fields must be finite, nonnegative, and mask aligned"
        )
    exposure = torch.where(
        operation_exposure_mask,
        operation_exposure_per_opportunity.to(operation_excitation_rate),
        torch.ones_like(operation_excitation_rate),
    )
    return operation_excitation_rate * exposure * operation_mask


def normalized_cell_excitation_mean(
    operation_excitation_mean: torch.Tensor,
    populations: torch.Tensor,
    cell_gain: torch.Tensor | None = None,
) -> torch.Tensor:
    """Allocate each operation mean by rho1 diagonal and an optional normalized gain."""
    operation_excitation_mean = validate_operation_excitation_mean(
        operation_excitation_mean,
        batch_size=populations.shape[0] if populations.ndim == 2 else -1,
        operation_count=(
            operation_excitation_mean.shape[1]
            if operation_excitation_mean.ndim == 2
            else -1
        ),
    )
    if populations.ndim != 2 or not bool(torch.isfinite(populations).all()):
        raise ValueError("cell populations must be finite [batch,mode]")
    if bool((populations < 0).any()) or not torch.allclose(
        populations.sum(-1), torch.ones_like(populations.sum(-1)), atol=1e-6, rtol=1e-6
    ):
        raise ValueError(
            "cell populations must be normalized nonnegative rho1 diagonals"
        )
    batch_size, operation_count = operation_excitation_mean.shape
    if populations.shape[0] != batch_size:
        raise ValueError("population batch does not match operation excitation mean")
    weights = populations[:, None, :].expand(-1, operation_count, -1)
    if cell_gain is not None:
        if (
            cell_gain.shape != weights.shape
            or not torch.is_floating_point(cell_gain)
            or not bool(torch.isfinite(cell_gain).all())
            or bool((cell_gain < 0).any())
        ):
            raise ValueError(
                "cell_gain must be finite nonnegative [batch,operation,mode]"
            )
        weighted = weights * cell_gain
        denominator = weighted.sum(-1, keepdim=True)
        if bool((denominator <= 0).any()):
            raise ValueError("source-bound cell_gain cannot remove all populated cells")
        weights = weighted / denominator
    return operation_excitation_mean[..., None] * weights
