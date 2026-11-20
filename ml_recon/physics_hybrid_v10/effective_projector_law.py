"""Differentiable K<=1 reference using the effective single-event sector owner.

K2 is absent in this named inference reference. Experiments with nonzero K2
remain separate sensitivity families and cannot silently select this decoder.
"""

from __future__ import annotations

import torch

from nqrec.physics.single_event_mc import (
    calibrate_source_weights,
    four_cell_probabilities,
    k1_event_probability,
)


def effective_probabilities(density, analyzers, calibration):
    """[B,S,4] in 00,10,01,11 order under shared K1 density for S=0 and S=1."""
    calibration = calibration.to(device=density.device, dtype=density.real.dtype)
    batch, settings = analyzers.shape[:2]
    if calibration.shape == (batch, 5):
        calibration = calibration[:, None, :]
    elif calibration.shape != (batch, settings, 5):
        raise ValueError("effective calibration must be [B,5] or [B,S,5]")
    signal, efficiency, signal_background, read_efficiency, background = calibration.unbind(-1)
    source = calibrate_source_weights(
        signal, efficiency, signal_background, double_probability=torch.zeros_like(signal)
    )
    transfer = analyzers.to(density.dtype) * read_efficiency.sqrt().unsqueeze(-1)
    r1 = k1_event_probability(density[:, None], transfer, background)
    response = torch.stack((background.expand_as(r1), r1, torch.zeros_like(r1)), -1)
    return four_cell_probabilities(
        source["joint"], response[..., None].expand(-1, -1, -1, 2)
    )


def effective_jacobians(density, analyzers, calibration):
    """Control and common calibration logit-shift derivatives from the same forward.

    Each tangent is shared across examples only in its input slot. The forward is
    example-separable, so the resulting per-example derivatives remain independent.
    """
    from .training_ideas_geometry import apply_diagonal_control

    calibration = calibration.to(device=density.device, dtype=density.real.dtype)
    if ((calibration <= 0) | (calibration >= 1)).any():
        raise ValueError("profiled logit calibration requires an interior operating point")
    batch, n, _ = density.shape
    zero_action = calibration.new_zeros((batch, 1, n, 2))
    logits = calibration.logit()

    def control(action):
        return effective_probabilities(
            apply_diagonal_control(density, action)[:, 0], analyzers, calibration
        )

    def apparatus(logit_value):
        return effective_probabilities(density, analyzers, logit_value.sigmoid())

    def columns(function, source, directions):
        derivatives = []
        for direction in directions:
            _, derivative = torch.autograd.functional.jvp(
                function, source, direction.expand_as(source)
            )
            derivatives.append(derivative)
        return torch.stack(derivatives, dim=-1)

    control_basis = torch.eye(2 * n, dtype=calibration.dtype, device=density.device).reshape(
        2 * n, 1, 1, n, 2
    )
    nuisance_basis = torch.eye(5, dtype=calibration.dtype, device=density.device).unsqueeze(1)
    return columns(control, zero_action, control_basis), columns(apparatus, logits, nuisance_basis)
