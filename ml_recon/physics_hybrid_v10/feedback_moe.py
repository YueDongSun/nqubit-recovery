"""Soft top-2 V10 W-feedback mixture of experts."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

EXPERT_NAMES = ("phase", "amplitude", "mixed_coherent", "fragmented_partition",
                "dephasing_leakage", "measurement_noise_abstention")


class _Expert(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 4))

    def forward(self, modes: torch.Tensor) -> torch.Tensor:
        return self.net(modes)


class V10FeedbackMoE(nn.Module):
    def __init__(self, width: int = 192, gate_input_dim: int = 64):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(gate_input_dim, width), nn.GELU(),
                                  nn.Linear(width, len(EXPERT_NAMES)))
        self.experts = nn.ModuleList(_Expert(width) for _ in EXPERT_NAMES)

    def forward(self, mode_tokens: torch.Tensor, gate_features: torch.Tensor,
                analytic_action: torch.Tensor, mode_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.gate(gate_features)
        top_values, top_indices = torch.topk(logits, k=2, dim=-1)
        top_weights = torch.softmax(top_values, dim=-1)
        all_experts = torch.stack([expert(mode_tokens) for expert in self.experts], dim=2)
        gather_index = top_indices[:, None, :, None].expand(-1, mode_tokens.shape[1], -1, 4)
        selected = torch.gather(all_experts, 2, gather_index)
        mixed = (selected * top_weights[:, None, :, None]).sum(2)
        damping = torch.sigmoid(mixed[..., 0])
        residual_amp = 0.1 * torch.tanh(mixed[..., 1])
        residual_phase = 0.1 * torch.tanh(mixed[..., 2])
        active_mask = torch.sigmoid(mixed[..., 3]) * mode_mask
        amp = damping * analytic_action[..., 0] + residual_amp * active_mask
        phase = damping * analytic_action[..., 1] + residual_phase * active_mask
        amp = (amp - (amp * mode_mask).sum(-1, keepdim=True) /
               mode_mask.sum(-1, keepdim=True).clamp_min(1)) * mode_mask
        phase = (phase - (phase * mode_mask).sum(-1, keepdim=True) /
                 mode_mask.sum(-1, keepdim=True).clamp_min(1)) * mode_mask
        load = torch.softmax(logits, -1).mean(0)
        balance_loss = ((load - 1.0 / len(EXPERT_NAMES)) ** 2).mean()
        N = mode_tokens.shape[1]
        helmert = torch.zeros((N - 1, N), dtype=amp.dtype, device=amp.device)
        for i in range(1, N):
            helmert[i - 1, :i] = 1.0 / (i * (i + 1.0)) ** 0.5
            helmert[i - 1, i] = -i / (i * (i + 1.0)) ** 0.5
        return {"delta_log_amp": amp, "delta_phase": phase,
                "delta_amplitude_coordinates": torch.einsum("in,bn->bi", helmert, amp),
                "delta_phase_coordinates": torch.einsum("in,bn->bi", helmert, phase),
                "expert_indices": top_indices, "expert_weights": top_weights,
                "routing_probabilities": torch.softmax(logits, -1),
                "damping": damping,
                "residual_amp": residual_amp,
                "residual_phase": residual_phase,
                "moe_load_balance_loss": balance_loss,
                "active_mode_mask": active_mask}
