"""Count/outcome tokenizer using physical descriptors, never fixed operation IDs."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class V10CountTokenizer(nn.Module):
    def __init__(self, count_dim: int = 4, descriptor_dim: int = 15, width: int = 192):
        super().__init__()
        self.count_dim = count_dim
        self.descriptor_dim = descriptor_dim
        self.width = width
        self.net = nn.Sequential(nn.Linear(count_dim + descriptor_dim + 3, width),
                                 nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width))

    def forward(self, counts: torch.Tensor, descriptors: torch.Tensor,
                shots: torch.Tensor, uncertainty: torch.Tensor,
                valid_mask: torch.Tensor) -> torch.Tensor:
        if counts.shape[:-1] != descriptors.shape[:-1]:
            raise ValueError("count and descriptor token shapes must agree")
        if counts.shape[-1] != self.count_dim or descriptors.shape[-1] != self.descriptor_dim:
            raise ValueError("token feature dimension mismatch")
        shots_feature = torch.log1p(shots.float()).unsqueeze(-1)
        x = torch.cat((torch.log1p(counts.float()), descriptors.float(), shots_feature,
                       uncertainty.float().unsqueeze(-1), valid_mask.float().unsqueeze(-1)), dim=-1)
        return self.net(x) * valid_mask.unsqueeze(-1)
