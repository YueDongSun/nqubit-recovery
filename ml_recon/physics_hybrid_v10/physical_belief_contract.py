"""Physical K1 belief constraints shared by calibration and feedback.

The physical route has no registered confidence head. Its K1 probability is a
deterministic trace of the bound subnormalized R1 prediction, not an
independently supplied calibration quantity.
"""

from __future__ import annotations

import torch

from .belief_geometry import Belief_validity_config


def validate_physical_belief_config(config: Belief_validity_config) -> None:
    """Validate the physical subset of the generic belief configuration."""
    if not isinstance(config, Belief_validity_config):
        raise ValueError("physical belief route requires a registered validity configuration")
    config.validate()
    if config.minimum_confidence is not None:
        raise ValueError(
            "physical belief route has no registered confidence output; "
            "minimum_confidence must be None"
        )


def derive_physical_k1_probability(
    r1_re: torch.Tensor,
    aggregate_posterior: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``Tr(R1)`` and validate an optional ``[rest, K1]`` posterior."""
    if (
        not isinstance(r1_re, torch.Tensor)
        or r1_re.ndim != 3
        or r1_re.shape[-1] < 2
        or r1_re.shape[-2] != r1_re.shape[-1]
        or r1_re.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("physical K1 derivation requires float32/float64 [batch,N,N] R1")
    trace = torch.diagonal(r1_re, dim1=-2, dim2=-1).sum(-1)
    tolerance = torch.finfo(r1_re.dtype).eps * r1_re.shape[-1] * 16
    if (
        not bool(torch.isfinite(trace).all())
        or bool((trace < -tolerance).any())
        or bool((trace > 1 + tolerance).any())
    ):
        raise ValueError("physical K1 derivation requires finite R1 traces in [0,1]")
    # Preserve the existing R1 validity convention: only machine-scale endpoint
    # roundoff is projected into the probability simplex.
    trace = trace.clamp(0, 1)
    if aggregate_posterior is None:
        return trace
    if (
        not isinstance(aggregate_posterior, torch.Tensor)
        or aggregate_posterior.shape != (r1_re.shape[0], 2)
        or aggregate_posterior.dtype != r1_re.dtype
        or aggregate_posterior.device != r1_re.device
    ):
        raise ValueError("aggregate K1 posterior requires aligned [batch,2] R1 precision and device")
    expected = torch.stack((1 - trace, trace), dim=-1)
    if (
        not bool(torch.isfinite(aggregate_posterior).all())
        or bool((aggregate_posterior < -tolerance).any())
        or bool((aggregate_posterior > 1 + tolerance).any())
    ):
        raise ValueError("aggregate K1 posterior must be finite and nonnegative")
    bounded_posterior = aggregate_posterior.clamp(0, 1)
    if not torch.allclose(bounded_posterior.sum(-1), torch.ones_like(trace), atol=tolerance, rtol=0):
        raise ValueError("aggregate K1 posterior must be normalized")
    if not torch.allclose(bounded_posterior, expected, atol=tolerance, rtol=0):
        raise ValueError("aggregate K1 posterior must equal the bound R1 trace")
    return trace
