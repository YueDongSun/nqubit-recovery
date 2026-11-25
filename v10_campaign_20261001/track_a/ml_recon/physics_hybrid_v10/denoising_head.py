"""Posterior denoising head; outputs probabilities/covariance, never negative counts."""

from __future__ import annotations

import torch
import torch.nn as nn


class V10DenoisingHead(nn.Module):
    def __init__(self, width: int = 192, outcome_dim: int = 4, nuisance_dim: int = 8):
        super().__init__()
        self.probability = nn.Linear(width, outcome_dim)
        self.log_variance = nn.Linear(width, outcome_dim)
        self.nuisance = nn.Linear(width, nuisance_dim * 2)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        probability = torch.softmax(self.probability(tokens), -1)
        variance = torch.nn.functional.softplus(self.log_variance(tokens)) + 1e-8
        nuisance = self.nuisance(tokens.mean(1))
        mean, log_var = nuisance.chunk(2, -1)
        return {"p_clean": probability, "covariance_clean_diag": variance,
                "nuisance_posterior_mean": mean,
                "nuisance_posterior_variance": torch.nn.functional.softplus(log_var) + 1e-8}
