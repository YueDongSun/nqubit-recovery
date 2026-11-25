"""Support-preserving radial bounds for a previously projected logical action."""
from __future__ import annotations

import math

import torch


bound_method = "recoverable_ray_scale_v1"


def bound_recoverable_action(action: torch.Tensor, *, amplitude_bound: float,
                             phase_bound: float) -> torch.Tensor:
    """Apply one scalar per action; never independently clip its coordinates.

    If action is in the recoverable linear subspace, scalar multiplication
    preserves that membership. This is a conservative feasible bound, not the
    optimizer of a box-constrained control problem.
    """
    if (not isinstance(action, torch.Tensor) or not action.is_floating_point()
            or action.ndim < 1 or action.shape[-1] == 0 or action.shape[-1] % 2):
        raise ValueError("logical actions require a nonempty even floating dimension")
    if any(type(value) not in (float, int) or not math.isfinite(value) or value <= 0
           for value in (amplitude_bound, phase_bound)):
        raise ValueError("action bounds must be finite positive numbers")
    if not bool(torch.isfinite(action).all()):
        raise ValueError("logical action contains nonfinite values")
    values = action if action.dtype in (torch.float32, torch.float64) else action.float()
    limits = values.new_tensor((amplitude_bound, phase_bound)).repeat(values.shape[-1] // 2)
    # max(|u|, b) keeps all ratios in (0, 1], including exact zero actions.
    ratios = limits / torch.maximum(values.abs(), limits)
    scale = ratios.amin(dim=-1, keepdim=True)
    return values * scale
