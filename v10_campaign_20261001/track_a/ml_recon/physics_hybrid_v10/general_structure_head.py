"""General-N ESM posterior head."""

from __future__ import annotations

import torch
import torch.nn as nn


class V10GeneralStructureHead(nn.Module):
    def __init__(self, width: int = 192, max_depth: int = 256):
        super().__init__()
        self.pair_entanglement = nn.Bilinear(width, width, 1)
        self.partition_link = nn.Bilinear(width, width, 1)
        self.depth = nn.Linear(width, max_depth)
        self.intactness = nn.Linear(width, max_depth)
        self.stretchability = nn.Linear(width, 2 * max_depth - 1)
        self.coarse = nn.Linear(width, 10)
        self.abstain = nn.Linear(width, 1)

    def forward(self, mode_tokens: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        b, n, width = mode_tokens.shape
        left = mode_tokens.unsqueeze(2).expand(b, n, n, width)
        right = mode_tokens.unsqueeze(1).expand(b, n, n, width)
        directed_entanglement = self.pair_entanglement(left, right).squeeze(-1)
        entanglement_logits = 0.5 * (
            directed_entanglement + directed_entanglement.transpose(1, 2))
        directed_partition = self.partition_link(left, right).squeeze(-1)
        partition_logits = 0.5 * (
            directed_partition + directed_partition.transpose(1, 2))
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
        diagonal_mask = ~torch.eye(n, device=mode_tokens.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask & diagonal_mask
        pair_probs = torch.sigmoid(entanglement_logits) * pair_mask
        partition_probs = torch.sigmoid(partition_logits) * pair_mask
        pooled = (mode_tokens * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return {"pair_entanglement_probs": pair_probs,
                "pair_entanglement_logits": entanglement_logits,
                "depth_posterior": torch.softmax(self.depth(pooled)[:, :n], -1),
                "intactness_posterior": torch.softmax(self.intactness(pooled)[:, :n], -1),
                "stretchability_posterior": torch.softmax(
                    self.stretchability(pooled)[:, self.stretchability.out_features // 2 - (n - 1):
                                                self.stretchability.out_features // 2 + n], -1),
                "stretchability_values": torch.arange(
                    1 - n, n, device=mode_tokens.device, dtype=torch.long),
                # Keep raw logits and probabilities distinct.  The decoder
                # must receive an explicit input_kind and may not apply a
                # second sigmoid to an already-normalized probability.
                "partition_link_logits": partition_logits,
                "partition_link_probs": partition_probs,
                "coarse_template_posterior": torch.softmax(self.coarse(pooled), -1),
                "abstention_probability": torch.sigmoid(self.abstain(pooled)).squeeze(-1)}
