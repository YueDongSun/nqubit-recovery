"""Separate physical and correlation graph encoders."""

from __future__ import annotations

import torch
import torch.nn as nn


class GraphMessageBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.self_map = nn.Linear(width, width)
        self.neighbour_map = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighbours = adjacency @ nodes / degree
        return self.norm(nodes + torch.nn.functional.gelu(
            self.self_map(nodes) + self.neighbour_map(neighbours)))


class V10GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, width: int = 192, blocks: int = 3):
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(GraphMessageBlock(width) for _ in range(blocks))

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.input(node_features)
        for block in self.blocks:
            h = block(h, adjacency)
        return h


class V10SeparatedGraphEncoders(nn.Module):
    def __init__(self, input_dim: int, width: int = 192, blocks: int = 3):
        super().__init__()
        self.physical = V10GraphEncoder(input_dim, width, blocks)
        self.correlation = V10GraphEncoder(input_dim, width, blocks)

    def forward(self, features: torch.Tensor, physical_adjacency: torch.Tensor,
                correlation_adjacency: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"physical_tokens": self.physical(features, physical_adjacency),
                "correlation_tokens": self.correlation(features, correlation_adjacency)}
