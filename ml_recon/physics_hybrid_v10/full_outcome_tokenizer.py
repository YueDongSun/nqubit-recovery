"""Lossless small-N full global-outcome tokenization for N=3/4."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class V10FullOutcomeTokenizer(nn.Module):
    """Embed the complete 2^N histogram; pair marginals are not a substitute."""

    def __init__(self, width: int, maximum_outcomes: int = 16):
        super().__init__()
        self.maximum_outcomes = int(maximum_outcomes)
        self.net = nn.Sequential(
            nn.Linear(self.maximum_outcomes, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, width))

    def forward(self, histogram: torch.Tensor, outcome_mask: torch.Tensor,
                shots: torch.Tensor, setting_mask: torch.Tensor, N: int) -> torch.Tensor:
        expected_outcomes = 2 ** int(N)
        if N not in (3, 4) or histogram.ndim != 3 \
                or histogram.shape[-1] != expected_outcomes \
                or outcome_mask.shape != histogram.shape:
            raise ValueError("small-N full histogram/mask must have shape [batch,settings,2^N]")
        if histogram.shape[:2] != shots.shape or shots.shape != setting_mask.shape:
            raise ValueError("full histogram setting axes disagree with shots/mask")
        probability = torch.where(
            outcome_mask.bool(), histogram.float(), torch.zeros_like(histogram.float()))
        probability = probability / shots.float().clamp_min(1.0).unsqueeze(-1)
        if expected_outcomes < self.maximum_outcomes:
            probability = F.pad(probability, (0, self.maximum_outcomes - expected_outcomes))
        return self.net(probability) * setting_mask.unsqueeze(-1)
