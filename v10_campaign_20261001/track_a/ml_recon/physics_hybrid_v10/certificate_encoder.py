"""Deterministic certificate token encoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class V10CertificateEncoder(nn.Module):
    def __init__(self, input_dim: int = 10, width: int = 192):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(),
                                 nn.Linear(width, width), nn.LayerNorm(width))

    def forward(self, certificate_features: torch.Tensor,
                certificate_mask: torch.Tensor) -> torch.Tensor:
        return self.net(certificate_features) * certificate_mask.unsqueeze(-1)
