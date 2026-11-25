"""Named V6 objectives and auditable adaptive loss balancing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


LOSS_NAMES = (
    "real_count",
    "synthetic_count",
    "global_click",
    "probe_direction",
    "repeat",
    "intervention",
    "coefficient",
    "sparse",
    "residual",
    "state",
    "action_teacher",
    "uq",
    "graph",
    "hard_projection_audit",
)
BALANCING_MODES = ("static", "uncertainty_weighting", "gradnorm", "residual_decay")


@dataclass(frozen=True)
class LossAuditV1:
    raw_value: float
    weight: float
    gradient_norm: float
    decay_rate: float


def complete_loss_terms(losses: Mapping[str, Any], *, reference: Any) -> Dict[str, Any]:
    """Return the full named objective vector, filling absent terms with graph-safe zeros."""
    unknown = set(losses) - set(LOSS_NAMES)
    if unknown:
        raise ValueError(f"unknown V6 loss terms: {sorted(unknown)}")
    zero = reference.sum() * 0.0
    return {name: losses.get(name, zero) for name in LOSS_NAMES}


def balance_losses(
    losses: Mapping[str, Any],
    *,
    mode: str = "static",
    static_weights: Optional[Mapping[str, float]] = None,
    log_variances: Optional[Mapping[str, Any]] = None,
    gradient_norms: Optional[Mapping[str, float]] = None,
    step: int = 0,
    residual_decay_rate: float = 1e-4,
) -> tuple[Any, Dict[str, LossAuditV1]]:
    """Combine losses and preserve raw values, weights, gradients, and decay evidence."""
    if mode not in BALANCING_MODES:
        raise ValueError(f"unsupported loss balancing mode: {mode}")
    if not losses:
        raise ValueError("at least one loss term is required")
    weights = dict(static_weights or {})
    gradients = dict(gradient_norms or {})
    total = None
    audit: Dict[str, LossAuditV1] = {}
    mean_gradient = sum(max(float(value), 0.0) for value in gradients.values()) / max(
        len(gradients), 1
    )
    for name, raw in losses.items():
        base_weight = float(weights.get(name, 1.0))
        decay = 1.0
        regularizer = raw.sum() * 0.0
        if mode == "uncertainty_weighting":
            if log_variances is None or name not in log_variances:
                raise ValueError(f"missing log variance for loss {name}")
            log_variance = log_variances[name]
            weight = base_weight * torch.exp(-log_variance)
            regularizer = 0.5 * log_variance
        elif mode == "gradnorm":
            gradient = max(float(gradients.get(name, 0.0)), 1e-12)
            weight = base_weight * (mean_gradient / gradient if mean_gradient > 0.0 else 1.0)
        elif mode == "residual_decay" and name == "residual":
            decay = math.exp(-max(int(step), 0) * max(float(residual_decay_rate), 0.0))
            weight = base_weight * decay
        else:
            weight = base_weight
        contribution = raw * weight + regularizer
        total = contribution if total is None else total + contribution
        scalar_weight = float(weight.detach().cpu()) if hasattr(weight, "detach") else float(weight)
        audit[name] = LossAuditV1(
            raw_value=float(raw.detach().cpu()),
            weight=scalar_weight,
            gradient_norm=float(gradients.get(name, 0.0)),
            decay_rate=float(decay),
        )
    return total, audit


__all__ = [
    "BALANCING_MODES",
    "LOSS_NAMES",
    "LossAuditV1",
    "balance_losses",
    "complete_loss_terms",
]
