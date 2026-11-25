"""PSD W-subspace head with ALR/Helmert uncertainty geometry."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _helmert(N: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    matrix = torch.zeros((N - 1, N), device=device, dtype=dtype)
    for i in range(1, N):
        matrix[i - 1, :i] = 1.0 / (i * (i + 1.0)) ** 0.5
        matrix[i - 1, i] = -i / (i * (i + 1.0)) ** 0.5
    return matrix


class V10WStructureHead(nn.Module):
    def __init__(self, width: int = 192, factor_rank: int = 4):
        super().__init__()
        self.factor_rank = int(factor_rank)
        self.mode = nn.Linear(width, 2 * self.factor_rank + 4)
        self.sector_head = nn.Linear(width, 3)
        self.global_head = nn.Linear(width, 3)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if tokens.ndim != 3 or mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("tokens/mask must have shapes [batch,N,width] and [batch,N]")
        batch, N, _ = tokens.shape
        if N < 3 or not torch.all(mask.sum(-1) == N):
            raise ValueError("V10 W head currently requires an unpadded exact-N mode panel")
        mode = self.mode(tokens)
        factor_real = mode[..., :self.factor_rank].float()
        factor_imag = mode[..., self.factor_rank:2 * self.factor_rank].float()
        factor = torch.complex(factor_real, factor_imag) * mask.unsqueeze(-1)
        raw_r1 = factor @ factor.conj().transpose(-2, -1)
        raw_trace = torch.diagonal(raw_r1, dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-12)

        pooled = (tokens * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        sector_probabilities = torch.softmax(self.sector_head(pooled), -1)
        p0, p1, p_ge2 = sector_probabilities.unbind(-1)
        R1 = raw_r1 * (p1 / raw_trace).view(batch, 1, 1)
        diagonal = torch.diagonal(R1, dim1=-2, dim2=-1).real
        conditional_populations = diagonal / p1.unsqueeze(-1).clamp_min(1e-12)
        scale = torch.sqrt(diagonal.unsqueeze(-1) * diagonal.unsqueeze(-2)).clamp_min(1e-12)
        normalized_coherence = R1 / scale

        amplitude_variance = F.softplus(mode[..., 2 * self.factor_rank + 2]) + 1e-6
        phase_variance = F.softplus(mode[..., 2 * self.factor_rank + 3]) + 1e-6
        helmert = _helmert(N, device=tokens.device, dtype=torch.float32)
        amplitude_covariance = torch.einsum(
            "in,bn,jn->bij", helmert, amplitude_variance, helmert)
        phase_covariance = torch.einsum(
            "in,bn,jn->bij", helmert, phase_variance, helmert)
        zeros = torch.zeros_like(amplitude_covariance)
        joint_covariance = torch.cat((
            torch.cat((amplitude_covariance, zeros), -1),
            torch.cat((zeros, phase_covariance), -1),
        ), -2)

        global_values = self.global_head(pooled)
        return {
            "p0": p0,
            "p1": p1,
            "p_ge2": p_ge2,
            "R1_re": R1.real,
            "R1_im": R1.imag,
            "R1_factor_re": factor_real,
            "R1_factor_im": factor_imag,
            "W_populations": conditional_populations,
            "W_coherence_re": normalized_coherence.real,
            "W_coherence_im": normalized_coherence.imag,
            "amplitude_error": mode[..., 2 * self.factor_rank],
            "phase_error": torch.atan2(
                torch.sin(mode[..., 2 * self.factor_rank + 1]),
                torch.cos(mode[..., 2 * self.factor_rank + 1])),
            "amplitude_alr_covariance": amplitude_covariance,
            "phase_helmert_covariance": phase_covariance,
            "joint_control_covariance": joint_covariance,
            "W_amplitude_variance": amplitude_variance,
            "W_phase_variance": phase_variance,
            "leakage": p_ge2,
            "recoverability": torch.sigmoid(global_values[:, 0]),
            "W_uncertainty": F.softplus(global_values[:, 1]),
            "entanglement_depth_score": torch.sigmoid(global_values[:, 2]),
        }
