"""Count-conditioned, gauge-invariant analyzer messages for the V10 main model.

This adapter is for an explicitly declared single-excitation projector readout.
It does not reinterpret the legacy Pauli outcome schema as a photon count law.
Only public commands, analyzers and visible measurement tokens enter this path.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class ProjectorConditioning(nn.Module):
    """Apply factorized analyzer projectors without an [operations,N,N] tensor."""

    def __init__(self, width: int, *, effective: bool = False):
        super().__init__()
        self.mode_gate = nn.Linear(width, width)
        self.observation_gate = nn.Linear(width, width)
        self.update = nn.Sequential(
            nn.Linear(3 * width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.budget = nn.Linear(2, width, bias=False)
        self.calibration = nn.Linear(5, width, bias=False) if effective else None

    def forward(
        self,
        mode_tokens: torch.Tensor,
        measurement_tokens: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        required = {"analyzer_re", "analyzer_im", "prepared_re", "prepared_im", "shots"}
        if not required <= batch.keys():
            raise ValueError(
                f"projector readout missing public fields: {sorted(required - batch.keys())}"
            )
        real = batch["analyzer_re"].to(mode_tokens.dtype)
        imag = batch["analyzer_im"].to(mode_tokens.dtype)
        if real.shape != imag.shape or real.shape[:2] != visible_mask.shape:
            raise ValueError("analyzers must have shape [batch,settings,N]")
        if real.shape[0] != mode_tokens.shape[0] or real.shape[-1] != mode_tokens.shape[1]:
            raise ValueError("analyzer modes and mode tokens disagree")
        norm = real.square().sum(-1) + imag.square().sum(-1)
        if not torch.isfinite(norm).all() or not torch.allclose(
            norm, torch.ones_like(norm), atol=2e-5
        ):
            raise ValueError("public analyzer vectors must be finite and normalized")
        command = torch.complex(
            batch["prepared_re"].to(mode_tokens.dtype),
            batch["prepared_im"].to(mode_tokens.dtype),
        )
        if command.shape != mode_tokens.shape[:2] or not torch.isfinite(command).all():
            raise ValueError("prepared command must be a finite [batch,N] public vector")
        if not torch.allclose(
            command.abs().square().sum(-1), torch.ones_like(norm[:, 0]), atol=2e-5
        ):
            raise ValueError("prepared command must have unit norm")
        analyzer = torch.complex(real, imag)
        hidden = command.unsqueeze(-1) * torch.sigmoid(self.mode_gate(mode_tokens))
        projection = torch.einsum("bon,bnw->bow", analyzer.conj(), hidden)
        observed = self.observation_gate(measurement_tokens)
        calibration_context = None
        if self.calibration is not None:
            calibration = batch["effective_calibration"].to(observed.dtype)
            if not torch.isfinite(calibration).all() or ((calibration < 0) | (calibration > 1)).any():
                raise ValueError("invalid effective calibration probabilities")
            if calibration.shape == (*visible_mask.shape, 5):
                mean = (calibration * visible_mask[..., None]).sum(1) / visible_mask.sum(1, keepdim=True)
                observed = observed + self.calibration(calibration - mean[:, None])
            elif calibration.shape == (command.shape[0], 5):
                mean = calibration
            else:
                raise ValueError("effective calibration must be [B,5] or [B,S,5]")
            calibration_context = self.calibration(mean).unsqueeze(1)
        gate = torch.tanh(observed)
        gate = gate * visible_mask.unsqueeze(-1)
        denominator = visible_mask.sum(-1, keepdim=True).to(gate.dtype)
        if torch.any(denominator <= 0):
            raise ValueError("projector training requires at least one visible operation per row")
        message = torch.einsum("bon,bow->bnw", analyzer, projection * gate)
        message = message / denominator.unsqueeze(-1)
        relative = command.conj().unsqueeze(-1) * message
        features = torch.cat((relative.real, relative.imag, message.abs().square()), -1)
        trials = batch["shots"].to(gate.dtype)
        if (
            trials.shape != visible_mask.shape
            or not torch.isfinite(trials).all()
            or torch.any(trials < 0)
        ):
            raise ValueError("shots must be independently known nonnegative opportunities")
        budget_features = (
            torch.stack(
                (
                    torch.log1p((trials * visible_mask).sum(-1)),
                    torch.log1p(visible_mask.sum(-1).to(gate.dtype)),
                ),
                -1,
            )
            / 16.0
        )
        budget_context = torch.tanh(self.budget(budget_features)).unsqueeze(1)
        if calibration_context is not None:
            budget_context = budget_context + calibration_context
        return self.update(features) + budget_context
