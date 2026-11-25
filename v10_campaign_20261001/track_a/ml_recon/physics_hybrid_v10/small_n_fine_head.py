"""Exact project-operational 14/24-class heads for N=3/4."""

from __future__ import annotations

import torch
import torch.nn as nn


class V10SmallNFineHead(nn.Module):
    def __init__(self, width: int = 192):
        super().__init__()
        self.head_3q = nn.Linear(width, 14)
        self.head_4q = nn.Linear(width, 24)

    def forward(self, latent: torch.Tensor, N: int, identifiable: bool) -> dict[str, torch.Tensor | bool]:
        if N == 3:
            logits = self.head_3q(latent)
        elif N == 4:
            logits = self.head_4q(latent)
        else:
            raise ValueError("fine-family head is only defined for N=3,4")
        return {"fine_family_logits": logits, "fine_family_probs": torch.softmax(logits, -1),
                "fine_family_primary_report_allowed": bool(identifiable)}
