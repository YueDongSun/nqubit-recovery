"""Shared set estimators and differentiable W-state preparation for high N.

The input is a matrix of measured or public features per mode. For the
structured generator, encode_observations supplies four count residuals,
log calibration weight, log efficiency, log N, log1p expected coincidences,
and the two physical lattice coordinates scaled to [-1, 1].
"""
from __future__ import annotations

import math

import torch
from torch import nn


FEATURE_DIM = 10
HIDDEN_WIDTH = 48
ANCHORS_PER_AXIS = 4
KERNEL_SIGMA_CELLS = 2.5
GRID_SIDE = 16
ACTION_BOUND = 0.25


def spatial_pair_context(values: torch.Tensor, membership: torch.Tensor, *,
                         second_order: bool = True) -> torch.Tensor:
    """Anchored peer signal, optionally adding half the mean squared difference.

    A[i,r] is the soft assignment of site i to spatial anchor r, normalized
    across anchors. K[i,j] = sum_r A[i,r] A[j,r] / sum_l A[l,r] is a
    row-stochastic kernel on the observed sites. Both arms use the peer
    mean sum_j K[i,j] v_j. The second-order arm adds the local semivariance
    sum_j K[i,j] (v_j-v_i)^2/2, componentwise in learned channels.
    Factoring K through R anchors takes O(N R H) work and O(N H + N R + R H)
    memory per row. No N-by-N tensor is formed.
    """
    anchor_masses = membership.sum(1).unsqueeze(-1)
    moments = torch.cat((values, values.square()), dim=-1) if second_order else values
    anchor_means = (membership.transpose(1, 2) @ moments) / anchor_masses
    site_means = membership @ anchor_means
    if not second_order:
        return site_means
    first, second = site_means.chunk(2, dim=-1)
    return first + 0.5 * (second - 2. * values * first + values.square())


class AmplitudeSetModel(nn.Module):
    """Estimate normalized log population, preserving the supplied mode ordering."""

    def __init__(self, arm: str, feature_dim: int = FEATURE_DIM, width: int = HIDDEN_WIDTH):
        super().__init__()
        if arm not in ("first_order", "second_order") or feature_dim < 2 or width <= 0:
            raise ValueError("Expected a documented arm, two geometry channels and positive width")
        self.arm = arm
        self.feature_dim = feature_dim
        axis = torch.linspace(-1., 1., ANCHORS_PER_AXIS, dtype=torch.float64)
        self.register_buffer("centers", torch.cartesian_prod(axis, axis))
        self.encoder = nn.Sequential(nn.Linear(feature_dim, width), nn.Tanh(),
                                     nn.Linear(width, width), nn.Tanh())
        self.value = nn.Linear(width, width)
        self.decoder = nn.Sequential(nn.Linear(3 * width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("Expected features with shape (rows, modes, feature_dim)")
        hidden = self.encoder(features)
        pooled = hidden.mean(1, keepdim=True).expand_as(hidden)
        values = self.value(hidden).tanh()
        offsets = features[..., None, -2:] - self.centers.to(dtype=features.dtype)
        sigma = 2. * KERNEL_SIGMA_CELLS / (GRID_SIDE - 1)
        membership = (-offsets.square().sum(-1) / (2. * sigma**2)).softmax(-1)
        context = spatial_pair_context(values, membership, second_order=self.arm == "second_order")
        logits = self.decoder(torch.cat((hidden, pooled, context), dim=-1)).squeeze(-1)
        return logits.log_softmax(-1)


def torch_centered_action(log_population: torch.Tensor, bound: float = ACTION_BOUND) -> torch.Tensor:
    """Amplitude-only control on positive support; exp(-inf) is exactly zero.

    Any additive log-population constant cancels. Joint radial contraction
    preserves the centered gauge and the per-mode box. This map is
    differentiable almost everywhere for fixed support. An all-zero support
    is invalid and emits NaNs. Neural log-softmax outputs have full support.
    """
    if log_population.ndim != 2 or not math.isfinite(bound) or bound < 0:
        raise ValueError("Expected a log-population matrix and a finite nonnegative bound")
    supported = log_population != -torch.inf
    raw = torch.where(supported, -0.5 * log_population, 0.)
    counts = supported.sum(-1, keepdim=True)
    centered = torch.where(supported, raw - raw.sum(-1, keepdim=True) / counts, 0.)
    if bound == 0.:
        action = centered * 0.
    else:
        peak = centered.abs().amax(-1, keepdim=True)
        action = centered * (bound / peak.clamp_min(bound))
    return torch.where(counts > 0, action, torch.nan)


def torch_conditional_fidelity(population: torch.Tensor, actions: torch.Tensor,
                               visibility: float | torch.Tensor = 1.) -> torch.Tensor:
    """Aligned uniform-W overlap after synthetic preparation at fixed V and p1.

    Given a valid population row and visibility in [0,1], output per row is
    [1-V + V*(sum_i c'_i)^2]/N for normalized positive amplitudes c'_i.
    Loader boundaries validate input probabilities; zero support is preserved
    without a probability floor.
    """
    if population.ndim != 2 or actions.shape != population.shape:
        raise ValueError("Population and action matrices must have the same shape")
    if isinstance(visibility, torch.Tensor) and visibility.shape not in ((), (len(population),)):
        raise ValueError("Expected a scalar visibility or one visibility per state")
    supported = population > 0.
    log_amplitude = 0.5 * torch.where(supported, population, 1.).log()
    u = actions.masked_fill(~supported, -torch.inf)
    log_amplitude = log_amplitude + (u - u.amax(-1, keepdim=True))
    weights = (log_amplitude - log_amplitude.amax(-1, keepdim=True)).exp()
    coherent_sum_sq = weights.sum(-1).square() / weights.square().sum(-1)
    return (1. - visibility + visibility * coherent_sum_sq) / population.shape[-1]
