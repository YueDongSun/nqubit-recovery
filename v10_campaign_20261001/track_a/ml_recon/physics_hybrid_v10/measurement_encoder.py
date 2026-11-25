"""Variable-length measurement set encoder."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class V10MeasurementEncoder(nn.Module):
    def __init__(self, width: int = 192, blocks: int = 6, heads: int = 8):
        super().__init__()
        # Capacity is placed in content-dependent feed-forward transformations rather
        # than an operation-ID table.  The 36x expansion is what makes C-S/C-M real
        # 15M+/35M-class candidates while preserving variable setting cardinality.
        layer = nn.TransformerEncoderLayer(width, heads, width * 36, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, blocks)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, width))
        self.pool = nn.MultiheadAttention(width, heads, batch_first=True)
        nn.init.normal_(self.pool_query, std=0.02)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.encoder(tokens, src_key_padding_mask=~valid_mask.bool())
        query = self.pool_query.expand(tokens.shape[0], -1, -1)
        # Preserve the trained path; evaluation needs no weights and uses SDPA.
        # A source-bound CUDA replay failed in legacy FP16 bmm after one epoch.
        pooled, _ = self.pool(query, hidden, hidden, key_padding_mask=~valid_mask.bool(),
                              need_weights=self.training)
        return {"measurement_tokens": hidden, "measurement_latent": pooled[:, 0]}
