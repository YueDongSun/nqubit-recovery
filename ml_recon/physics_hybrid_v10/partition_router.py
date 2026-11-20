"""Soft current/target partition and CSM relational routing sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .partition_structure import PartitionStructureV1


@dataclass(frozen=True)
class RelationalRoutingConfig:
    """Engineering defaults; router gains remain open pilot parameters."""

    alpha_partition: float = 0.0
    alpha_csm: float = 0.0
    confidence_temperature: float = 1.0
    engineering_default: bool = True

    def validate(self) -> None:
        if self.confidence_temperature <= 0.0:
            raise ValueError("confidence temperature must be positive")
        if self.alpha_partition < 0.0 or self.alpha_csm < 0.0:
            raise ValueError("router gains must be non-negative")


def structure_confidence(
    *,
    partition_entropy: float,
    candidate_count: int,
    certificate_coverage: float,
    abstention_probability: float,
    temperature: float = 1.0,
) -> float:
    if candidate_count <= 0 or temperature <= 0.0:
        raise ValueError("candidate count and temperature must be positive")
    entropy_normalized = float(partition_entropy) / max(np.log(candidate_count), 1e-12)
    return float(np.clip(
        (1.0 - np.clip(entropy_normalized, 0.0, 1.0))
        * np.clip(certificate_coverage, 0.0, 1.0)
        * (1.0 - np.clip(abstention_probability, 0.0, 1.0)),
        0.0, 1.0)) ** (1.0 / temperature)


def normalized_csm_coupling(coupling: np.ndarray) -> np.ndarray:
    value = np.asarray(coupling, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("CSM coupling must be square")
    value = np.maximum(0.5 * (value + value.T), 0.0)
    np.fill_diagonal(value, 0.0)
    scale = float(value.max(initial=0.0))
    return value / scale if scale > 0.0 else np.zeros_like(value)


def relational_bias(
    current: PartitionStructureV1,
    target: PartitionStructureV1,
    csm_coupling: np.ndarray | None = None,
    *,
    config: RelationalRoutingConfig | None = None,
) -> Dict[str, object]:
    """Build soft relational priors without hard-masking any action dimension."""
    active = config or RelationalRoutingConfig()
    active.validate()
    if current.N != target.N:
        raise ValueError("current and target partition N must match")
    confidence = min(
        structure_confidence(
            partition_entropy=float(current.partition_entropy or 0.0),
            candidate_count=current.candidate_count,
            certificate_coverage=current.certificate_coverage,
            abstention_probability=current.abstention_probability,
            temperature=active.confidence_temperature),
        structure_confidence(
            partition_entropy=float(target.partition_entropy or 0.0),
            candidate_count=target.candidate_count,
            certificate_coverage=target.certificate_coverage,
            abstention_probability=target.abstention_probability,
            temperature=active.confidence_temperature))
    current_membership, current_defined = current.co_membership_views()
    target_membership, target_defined = target.co_membership_views()
    pair_defined = current_defined * target_defined
    keep = current_membership * target_membership * pair_defined
    form = (1.0 - current_membership) * target_membership * pair_defined
    break_relation = current_membership * (1.0 - target_membership) * pair_defined
    for channel in (keep, form, break_relation):
        np.fill_diagonal(channel, 0.0)
    partition_channels = np.stack((keep, form, break_relation), axis=-1)
    partition_prior = keep + 2.0 * form - break_relation
    if csm_coupling is None:
        csm_prior = np.zeros_like(partition_prior)
    else:
        csm_prior = normalized_csm_coupling(csm_coupling)
        if csm_prior.shape != partition_prior.shape:
            raise ValueError("CSM coupling and partition N must match")
    bias = (active.alpha_partition * confidence * partition_prior
            + active.alpha_csm * confidence * csm_prior)
    return {
        "partition_prior": partition_prior,
        "partition_channels": partition_channels,
        "partition_defined": pair_defined,
        "current_partition": current_membership,
        "target_partition": target_membership,
        "csm_prior": csm_prior,
        "relational_bias": bias,
        "structure_confidence": confidence,
        "alpha_partition": active.alpha_partition,
        "alpha_csm": active.alpha_csm,
        "hard_action_mask": np.ones(current.N, dtype=np.bool_),
    }


class V10ZeroInitRelationalRouter(nn.Module):
    """Optional Torch sidecar with exact zero-init baseline behavior.

    The existing six experts are untouched.  When no side inputs are supplied,
    or when the two learnable gains are zero, this module returns its input
    unchanged.  Structure uncertainty scales the prior continuously instead of
    disabling physical action coordinates.
    """

    def __init__(self, width: int, *, node_context_dim: int = 0,
                 global_context_dim: int = 0):
        super().__init__()
        self.alpha_partition = nn.Parameter(torch.zeros(()))
        self.alpha_csm = nn.Parameter(torch.zeros(()))
        self.control_block = RelationalControlBlock(
            width, node_context_dim=node_context_dim,
            global_context_dim=global_context_dim)

    def forward(
        self,
        mode_tokens: torch.Tensor,
        *,
        partition_context: torch.Tensor | None = None,
        partition_current_context: torch.Tensor | None = None,
        partition_target_context: torch.Tensor | None = None,
        partition_current_defined: torch.Tensor | None = None,
        partition_target_defined: torch.Tensor | None = None,
        partition_mask: torch.Tensor | None = None,
        csm_node_context: torch.Tensor | None = None,
        csm_edge_context: torch.Tensor | None = None,
        csm_global_context: torch.Tensor | None = None,
        structure_confidence_value: torch.Tensor | None = None,
        partition_current_confidence: torch.Tensor | None = None,
        partition_target_confidence: torch.Tensor | None = None,
        mode_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if mode_tokens.ndim != 3:
            raise ValueError("mode_tokens must have shape [B,N,width]")
        B, N, _ = mode_tokens.shape
        device, dtype = mode_tokens.device, mode_tokens.dtype
        if partition_context is not None:
            if partition_current_context is not None or partition_target_context is not None:
                raise ValueError("partition_context cannot be combined with current/target contexts")
            partition_current_context = partition_context
            partition_target_context = partition_context
        if (partition_current_context is None) != (partition_target_context is None):
            raise ValueError("current and target partition contexts must be provided together")
        has_structure = partition_current_context is not None
        has_csm = csm_edge_context is not None or csm_node_context is not None \
            or csm_global_context is not None
        if not has_structure and not has_csm:
            return {"mode_tokens": mode_tokens, "relational_bias":
                    torch.zeros((B, N, N), dtype=dtype, device=device),
                    "structure_confidence": torch.ones((B,), dtype=dtype, device=device),
                    "alpha_partition": self.alpha_partition.detach().clone(),
                    "alpha_csm": self.alpha_csm.detach().clone()}
        if structure_confidence_value is not None \
                and (partition_current_confidence is not None
                     or partition_target_confidence is not None):
            raise ValueError(
                "use structure_confidence_value or current/target confidence, not both")
        if (partition_current_confidence is None) != (partition_target_confidence is None):
            raise ValueError("current and target partition confidence must be provided together")
        if partition_current_confidence is not None:
            current_confidence = partition_current_confidence.to(
                device=device, dtype=dtype).reshape(B)
            target_confidence = partition_target_confidence.to(
                device=device, dtype=dtype).reshape(B)
            confidence = torch.minimum(current_confidence, target_confidence)
        else:
            confidence = (torch.ones((B,), dtype=dtype, device=device)
                          if structure_confidence_value is None else
                          structure_confidence_value.to(device=device, dtype=dtype).reshape(B))
        def as_edge(value: torch.Tensor | None, name: str) -> torch.Tensor:
            if value is None:
                return torch.zeros((B, N, N), dtype=dtype, device=device)
            edge = value.to(device=device, dtype=dtype)
            if edge.ndim == 2 and edge.shape == (B, N):
                edge = edge.unsqueeze(1) * edge.unsqueeze(2)
            if edge.ndim == 4:
                raise ValueError(f"{name} must have shape [B,N,N]")
            if edge.shape != (B, N, N):
                raise ValueError(f"{name} must have shape [B,N,N] or [B,N,N,d]")
            return 0.5 * (edge + edge.transpose(1, 2))

        current = as_edge(partition_current_context, "partition_current_context")
        target = as_edge(partition_target_context, "partition_target_context")
        current_defined = as_edge(partition_current_defined, "partition_current_defined")
        target_defined = as_edge(partition_target_defined, "partition_target_defined")
        if partition_current_context is not None and partition_current_defined is None:
            current_defined = torch.ones_like(current)
        if partition_target_context is not None and partition_target_defined is None:
            target_defined = torch.ones_like(target)
        pair_defined = (current_defined.clamp(0.0, 1.0)
                        * target_defined.clamp(0.0, 1.0))
        keep = current.clamp(0.0, 1.0) * target.clamp(0.0, 1.0) * pair_defined
        form = (1.0 - current.clamp(0.0, 1.0)) * target.clamp(0.0, 1.0) * pair_defined
        break_relation = current.clamp(0.0, 1.0) * (1.0 - target.clamp(0.0, 1.0)) * pair_defined
        partition_channels = torch.stack((keep, form, break_relation), dim=-1)
        partition_prior = keep + 2.0 * form - break_relation
        csm = as_edge(csm_edge_context, "csm_edge_context")
        if csm_edge_context is not None and (
                not torch.all(torch.isfinite(csm)) or
                bool(torch.any(csm < 0.0)) or
                bool(torch.any(csm > 1.0))):
            raise ValueError("csm_edge_context must be normalized scalar K_ij in [0,1]")
        off_diagonal = ~torch.eye(N, dtype=torch.bool, device=device)[None]
        partition_channels = partition_channels * off_diagonal.unsqueeze(-1)
        partition_prior = partition_prior * off_diagonal
        csm = csm * off_diagonal
        if partition_mask is not None:
            mask = partition_mask.to(device=device, dtype=dtype)
            if mask.shape != (B, N):
                raise ValueError("partition_mask must have shape [B,N]")
            pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
            partition_channels = partition_channels * pair_mask.unsqueeze(-1)
            partition_prior = partition_prior * pair_mask
            csm = csm * pair_mask
        confidence = confidence.clamp(0.0, 1.0)
        bias = confidence[:, None, None] * (
            self.alpha_partition * partition_prior
            + self.alpha_csm * csm)
        gated_context = self.control_block(
            mode_tokens, partition_channels=partition_channels,
            csm_edge_context=csm, alpha_partition=self.alpha_partition,
            alpha_csm=self.alpha_csm, csm_node_context=csm_node_context,
            csm_global_context=csm_global_context, mode_mask=mode_mask,
            structure_confidence=confidence)
        return {
            "mode_tokens": mode_tokens + gated_context,
            "relational_bias": bias,
            "structure_confidence": confidence,
            "partition_channels": partition_channels,
            "partition_defined": pair_defined,
            "alpha_partition": self.alpha_partition,
            "alpha_csm": self.alpha_csm,
        }


class RelationalControlBlock(nn.Module):
    """Causal relational message passing kept outside the legacy MoE.

    The two scalar gains are the zero-init boundary: when both are zero the
    returned residual is exactly zero, while any non-zero gain makes the
    structure/CSM tensors an actual input to mode representations.
    """

    def __init__(self, width: int, *, node_context_dim: int = 0,
                 global_context_dim: int = 0):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.partition_edge_weights = nn.Parameter(
            torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32))
        self.csm_edge_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.node_projection = (nn.Linear(node_context_dim, width, bias=False)
                                if node_context_dim > 0 else None)
        self.global_projection = (nn.Linear(global_context_dim, width, bias=False)
                                  if global_context_dim > 0 else None)

    def forward(
        self,
        mode_tokens: torch.Tensor,
        *,
        partition_channels: torch.Tensor,
        csm_edge_context: torch.Tensor,
        alpha_partition: torch.Tensor,
        alpha_csm: torch.Tensor,
        csm_node_context: torch.Tensor | None,
        csm_global_context: torch.Tensor | None,
        mode_mask: torch.Tensor | None,
        structure_confidence: torch.Tensor,
    ) -> torch.Tensor:
        B, N, width = mode_tokens.shape
        valid = (torch.ones((B, N), dtype=torch.bool, device=mode_tokens.device)
                 if mode_mask is None else mode_mask.bool())
        q = self.query(mode_tokens)
        k = self.key(mode_tokens)
        base_value = self.value(mode_tokens)
        csm_value = base_value
        if self.node_projection is not None and csm_node_context is not None:
            node = csm_node_context.to(device=mode_tokens.device, dtype=mode_tokens.dtype)
            if node.shape[:2] != (B, N) or node.shape[-1] != self.node_projection.in_features:
                raise ValueError("csm_node_context has an incompatible shape")
            csm_value = csm_value + self.node_projection(node)
        if self.global_projection is not None and csm_global_context is not None:
            global_context = csm_global_context.to(
                device=mode_tokens.device, dtype=mode_tokens.dtype)
            if global_context.shape != (B, self.global_projection.in_features):
                raise ValueError("csm_global_context has an incompatible shape")
            csm_value = csm_value + self.global_projection(global_context)[:, None, :]
        base = torch.matmul(q, k.transpose(1, 2)) / max(width ** 0.5, 1.0)
        key_mask = valid[:, None, :]
        base = base.masked_fill(~key_mask, torch.finfo(base.dtype).min)
        partition_score = torch.tensordot(
            partition_channels, self.partition_edge_weights.to(mode_tokens.dtype), dims=([-1], [0]))
        csm_score = csm_edge_context * self.csm_edge_weight.to(mode_tokens.dtype)

        def message(edge_score: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            logits = base + edge_score
            attention = torch.softmax(logits, dim=-1)
            return torch.matmul(attention, value)

        base_message = message(torch.zeros_like(partition_score), base_value)
        partition_delta = (
            message(partition_score, base_value) - base_message)
        csm_delta = (
            message(csm_score, csm_value) - base_message)
        confidence = structure_confidence.to(
            device=mode_tokens.device, dtype=mode_tokens.dtype).reshape(B, 1, 1)
        residual = confidence * (
            alpha_partition * partition_delta + alpha_csm * csm_delta)
        residual = self.output(residual)
        return residual * valid.to(mode_tokens.dtype).unsqueeze(-1)


__all__ = [
    "RelationalControlBlock", "RelationalRoutingConfig", "V10ZeroInitRelationalRouter",
    "normalized_csm_coupling", "relational_bias", "structure_confidence",
]
