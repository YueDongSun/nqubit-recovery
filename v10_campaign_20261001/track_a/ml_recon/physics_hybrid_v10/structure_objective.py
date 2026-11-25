"""Stable, mask-aware Bernoulli objective for classifier structure logits."""
from __future__ import annotations

import torch
import torch.nn.functional as functional


def masked_structure_logit_loss(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Preserve Bernoulli likelihood gradients without sigmoid roundoff.

    Select the authoritative Boolean mask before nonlinear evaluation, so
    unavailable targets never contribute a loss or an undefined gradient.
    """
    if not logits.dtype.is_floating_point:
        raise TypeError("structure logits must be floating-point tensors")
    if mask.dtype is not torch.bool:
        raise TypeError("structure loss masks must be strict boolean tensors")
    if logits.shape != target.shape or logits.shape != mask.shape:
        raise ValueError("structure logits, targets and mask must have identical shapes")
    if logits.device != target.device or logits.device != mask.device:
        raise ValueError("structure logits, targets and mask must share a device")
    active_logits = logits[mask]
    active_target = target[mask]
    if (not bool(torch.isfinite(active_logits).all())
            or not bool(torch.isfinite(active_target).all())
            or bool(((active_target < 0) | (active_target > 1)).any())):
        raise ValueError("active structure logits must be finite and targets must lie in [0,1]")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        if active_logits.numel() == 0:
            return active_logits.float().sum()
        return functional.binary_cross_entropy_with_logits(
            active_logits.float(), active_target.float(), reduction="mean")
