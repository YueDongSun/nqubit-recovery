"""Signed monotonic gain quantiles and their registered pinball objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F


GAIN_QUANTILE_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)


def signed_monotonic_gain_quantiles(raw: torch.Tensor) -> torch.Tensor:
    if raw.shape[-1] != 5:
        raise ValueError("signed gain quantiles require five raw coordinates")
    q50 = raw[..., 0]
    q25 = q50 - F.softplus(raw[..., 1])
    q05 = q25 - F.softplus(raw[..., 2])
    q75 = q50 + F.softplus(raw[..., 3])
    q95 = q75 + F.softplus(raw[..., 4])
    return torch.stack((q05, q25, q50, q75, q95), dim=-1)


def gain_pinball_loss(prediction: torch.Tensor, target: torch.Tensor,
                      reduction: str = "mean") -> torch.Tensor:
    if prediction.shape[-1] != len(GAIN_QUANTILE_LEVELS):
        raise ValueError("pinball loss requires five registered gain quantiles")
    truth = target.unsqueeze(-1) if target.ndim == prediction.ndim - 1 else target
    levels = prediction.new_tensor(GAIN_QUANTILE_LEVELS)
    error = truth - prediction
    loss = torch.maximum(levels * error, (levels - 1.0) * error).mean(-1)
    if reduction == "none":
        return loss
    if reduction != "mean":
        raise ValueError("gain pinball reduction must be mean or none")
    return loss.mean()
