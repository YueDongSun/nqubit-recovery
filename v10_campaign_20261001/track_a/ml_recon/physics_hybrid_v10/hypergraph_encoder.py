"""Incidence-based hypergraph message passing."""

from __future__ import annotations

import torch
import torch.nn as nn


class V10HypergraphEncoder(nn.Module):
    def __init__(self, width: int = 192, blocks: int = 3):
        super().__init__()
        self.node_updates = nn.ModuleList(nn.Linear(width, width) for _ in range(blocks))
        self.edge_updates = nn.ModuleList(nn.Linear(width, width) for _ in range(blocks))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(blocks))

    def forward(self, nodes: torch.Tensor, incidence: torch.Tensor) -> dict[str, torch.Tensor]:
        h = nodes
        edge = torch.zeros((*incidence.shape[:-2], incidence.shape[-1], h.shape[-1]),
                           dtype=h.dtype, device=h.device)
        for node_map, edge_map, norm in zip(self.node_updates, self.edge_updates, self.norms):
            edge_degree = incidence.sum(dim=-2, keepdim=False).clamp_min(1.0).unsqueeze(-1)
            edge = incidence.transpose(-2, -1) @ h / edge_degree
            node_degree = incidence.sum(dim=-1, keepdim=True).clamp_min(1.0)
            message = incidence @ edge / node_degree
            h = norm(h + torch.nn.functional.gelu(node_map(h) + edge_map(message)))
        return {"node_tokens": h, "hyperedge_tokens": edge}
