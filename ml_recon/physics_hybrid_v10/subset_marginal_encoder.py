"""Sparse registered-subset measurement encoder for general-N ESM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class V10SubsetMarginalEncoder(nn.Module):
    def __init__(self, width: int, maximum_subset_order: int = 4,
                 maximum_outcomes: int = 16):
        super().__init__()
        self.maximum_subset_order = int(maximum_subset_order)
        self.maximum_outcomes = int(maximum_outcomes)
        self.axis_descriptor_projection = nn.Linear(4, width, bias=False)
        self.order_descriptor_projection = nn.Linear(
            maximum_subset_order + 1, width, bias=False)
        self.count_projection = nn.Sequential(
            nn.Linear(maximum_outcomes, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, width))
        self.pair_stat_projection = nn.Sequential(
            nn.Linear(11, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, width))

    def _pair_statistics(self, batch: dict[str, torch.Tensor], N: int
                         ) -> torch.Tensor:
        required = {
            "pair_stat_qubits", "pair_stat_axes", "pair_stat_expectation",
            "pair_stat_variance", "pair_stat_shots", "pair_stat_coverage_mask",
        }
        if not required <= set(batch):
            raise ValueError("general-N ESM requires aggregated pair statistics")
        qubits = batch["pair_stat_qubits"].long()
        axes = batch["pair_stat_axes"].long()
        expectation = batch["pair_stat_expectation"].float()
        variance = batch["pair_stat_variance"].float()
        shots = batch["pair_stat_shots"].float()
        coverage = batch["pair_stat_coverage_mask"].bool()
        if qubits.shape != axes.shape or qubits.shape[-1] != 2 \
                or expectation.shape != variance.shape \
                or expectation.shape != shots.shape \
                or expectation.shape != coverage.shape \
                or qubits.shape[:2] != expectation.shape:
            raise ValueError("aggregated pair-statistic tensors have inconsistent shapes")
        if torch.any(qubits < 0) or torch.any(qubits >= N) \
                or torch.any(axes < 0) or torch.any(axes > 2):
            raise ValueError("pair-statistic qubit/axis descriptor is out of range")
        qubit_features = qubits.to(expectation.dtype) / max(N - 1, 1)
        axis_features = functional.one_hot(
            axes, num_classes=3).to(expectation.dtype).flatten(-2)
        numeric = torch.stack((
            expectation,
            torch.sqrt(variance.clamp_min(0.0)),
            torch.log1p(shots) / 16.0,
        ), dim=-1)
        token = self.pair_stat_projection(torch.cat((
            qubit_features, axis_features, numeric), dim=-1))
        token = token * coverage.unsqueeze(-1)
        mode_context = torch.zeros(
            (token.shape[0], N, token.shape[-1]),
            dtype=token.dtype, device=token.device)
        mode_weight = torch.zeros(
            (token.shape[0], N, 1), dtype=token.dtype, device=token.device)
        for position in range(2):
            index = qubits[:, :, position]
            mode_context.scatter_add_(
                1, index.unsqueeze(-1).expand(-1, -1, token.shape[-1]), token)
            mode_weight.scatter_add_(
                1, index.unsqueeze(-1), coverage.unsqueeze(-1).to(token.dtype))
        return mode_context / mode_weight.clamp_min(1.0)

    def forward(self, batch: dict[str, torch.Tensor], N: int) -> torch.Tensor:
        if "pair_stat_expectation" in batch:
            return self._pair_statistics(batch, N)
        required = {
            "subset_qubits", "subset_order", "subset_axes",
            "subset_marginal_counts", "subset_marginal_mask", "subset_shots",
        }
        if not required <= set(batch):
            raise ValueError("general-N ESM requires registered subset marginal inputs")
        qubits = batch["subset_qubits"].long()
        order = batch["subset_order"].long()
        axes = batch["subset_axes"].long()
        counts = batch["subset_marginal_counts"].float()
        outcome_mask = batch["subset_marginal_mask"].bool()
        shots = batch["subset_shots"].float().clamp_min(1.0)
        if qubits.shape != axes.shape or qubits.shape[-1] != self.maximum_subset_order \
                or counts.shape[-1] != self.maximum_outcomes \
                or counts.shape != outcome_mask.shape or counts.shape[:2] != order.shape:
            raise ValueError("registered subset tensors have inconsistent shapes")
        if torch.any(order < 0) or torch.any(order > self.maximum_subset_order) \
                or torch.any(axes > 3):
            raise ValueError("registered subset order/axis descriptor is out of range")
        probability = torch.where(outcome_mask, counts, torch.zeros_like(counts)) \
            / shots.unsqueeze(-1)
        order_descriptor = functional.one_hot(
            order, num_classes=self.maximum_subset_order + 1).to(probability.dtype)
        token = self.count_projection(probability) + self.order_descriptor_projection(
            order_descriptor)
        valid_qubits = qubits >= 0
        axis_descriptor = functional.one_hot(
            axes.clamp_min(0), num_classes=4).to(probability.dtype)
        token = token + (
            self.axis_descriptor_projection(axis_descriptor) * valid_qubits.unsqueeze(-1)
        ).sum(2) / valid_qubits.sum(2, keepdim=True).clamp_min(1)
        mode_context = torch.zeros(
            (counts.shape[0], N, token.shape[-1]), dtype=token.dtype, device=token.device)
        mode_weight = torch.zeros(
            (counts.shape[0], N, 1), dtype=token.dtype, device=token.device)
        for position in range(self.maximum_subset_order):
            valid = valid_qubits[:, :, position]
            index = qubits[:, :, position].clamp_min(0)
            mode_context.scatter_add_(
                1, index.unsqueeze(-1).expand(-1, -1, token.shape[-1]),
                token * valid.unsqueeze(-1))
            mode_weight.scatter_add_(1, index.unsqueeze(-1), valid.unsqueeze(-1).to(token.dtype))
        return mode_context / mode_weight.clamp_min(1.0)
