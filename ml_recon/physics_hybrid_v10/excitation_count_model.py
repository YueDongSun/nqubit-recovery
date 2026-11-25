"""Poisson occupation probabilities and distinct local/cross K=2 marks."""

from __future__ import annotations

import torch


def poisson_sector_probabilities(
    mean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        not torch.is_floating_point(mean)
        or not bool(torch.isfinite(mean).all())
        or bool((mean < 0).any())
    ):
        raise ValueError("Poisson mean must be finite and nonnegative")
    p0 = torch.exp(-mean)
    p1 = mean * p0
    small_mean_series = mean.square() * (
        0.5
        + mean * (-1.0 / 3.0 + mean * (1.0 / 8.0 + mean * (-1.0 / 30.0 + mean / 144.0)))
    )
    stable_complement = -torch.expm1(torch.log1p(mean) - mean)
    pge2 = torch.where(mean < 0.1, small_mean_series, stable_complement)
    return p0, p1, pge2.clamp(0.0, 1.0)


def opportunity_weighted_sector_probabilities(
    mean: torch.Tensor, repeats: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        mean.shape != repeats.shape
        or repeats.dtype is torch.bool
        or not bool(torch.isfinite(repeats).all())
        or bool((repeats < 0).any())
        or not bool((repeats == repeats.floor()).all())
    ):
        raise ValueError(
            "means and finite nonnegative integer repeats must share [batch,operation] shape"
        )
    p0, p1, pge2 = poisson_sector_probabilities(mean)
    total = repeats.sum(-1, keepdim=True)
    weights = repeats / total.clamp_min(1)
    mixture = tuple((weights * value).sum(-1) for value in (p0, p1, pge2))
    valid = total.squeeze(-1) > 0
    result = tuple(
        torch.where(valid, value, torch.zeros_like(value)) for value in mixture
    )
    if bool(valid.any()) and not torch.allclose(
        sum(result), valid.to(mean.dtype), atol=1e-7, rtol=1e-7
    ):
        raise RuntimeError("opportunity-weighted sector mixture is not conserved")
    return result


def poisson_k2_mark_probabilities(
    cell_mean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact K=2 local and cross occupation probabilities without merging marks."""
    if (
        cell_mean.ndim < 1
        or not torch.is_floating_point(cell_mean)
        or not bool(torch.isfinite(cell_mean).all())
        or bool((cell_mean < 0).any())
    ):
        raise ValueError("cell means must be finite nonnegative [...,mode]")
    total_mean = cell_mean.sum(-1)
    prefactor = torch.exp(-total_mean)
    local = prefactor[..., None] * cell_mean.square() / 2.0
    cross = (
        prefactor[..., None, None] * cell_mean[..., :, None] * cell_mean[..., None, :]
    )
    diagonal = torch.diagonal(cross, dim1=-2, dim2=-1)
    cross = cross - torch.diag_embed(diagonal)
    k2 = local.sum(-1) + torch.triu(cross, diagonal=1).sum((-1, -2))
    expected = prefactor * total_mean.square() / 2.0
    if not torch.allclose(k2, expected, atol=1e-7, rtol=1e-6):
        raise RuntimeError(
            "local/cross K2 marks do not conserve Poisson K2 probability"
        )
    return local, cross, k2
