"""Post-materialization recoverable-support audits for Track B previews."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np

from .proposal_reader import ProposalValidationError


@dataclass(frozen=True)
class RecoverableSupportAuditV1:
    """Numerical evidence that a preview remains in registered logical support."""

    requested_norm: float
    materialized_norm: float
    support_leakage_norm: float
    support_leakage_fraction: float
    requested_to_materialized_logical_error: float
    direction_cosine: float
    recoverable_support_fraction: float
    leakage_tolerance: float
    maximum_requested_to_materialized_error: float
    minimum_direction_cosine: float
    passed: bool
    reasons: Tuple[str, ...]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": "RecoverableSupportAuditV1",
            "requested_norm": self.requested_norm,
            "materialized_norm": self.materialized_norm,
            "support_leakage_norm": self.support_leakage_norm,
            "support_leakage_fraction": self.support_leakage_fraction,
            "requested_to_materialized_logical_error": self.requested_to_materialized_logical_error,
            "direction_cosine": self.direction_cosine,
            "recoverable_support_fraction": self.recoverable_support_fraction,
            "leakage_tolerance": self.leakage_tolerance,
            "maximum_requested_to_materialized_error": self.maximum_requested_to_materialized_error,
            "minimum_direction_cosine": self.minimum_direction_cosine,
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class RayShrinkResultV1:
    """Result of retrying only scalar multiples of one requested action ray."""

    ray_scale: float
    attempts: int
    requested_logical_action: Tuple[float, ...]
    materialized_logical_action: Tuple[float, ...]
    audit: RecoverableSupportAuditV1
    accepted: bool
    reasons: Tuple[str, ...]


def support_projector(
    recoverable_command_basis: Sequence[Sequence[float]],
    logical_dimension: int,
) -> np.ndarray:
    """Build an orthogonal projector from the registered logical support basis."""

    basis = np.asarray(recoverable_command_basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != logical_dimension or basis.shape[1] == 0:
        raise ProposalValidationError("recoverable support basis has the wrong shape")
    if not np.all(np.isfinite(basis)):
        raise ProposalValidationError("recoverable support basis contains non-finite values")
    left, singular_values, _ = np.linalg.svd(basis, full_matrices=False)
    tolerance = max(basis.shape) * np.finfo(np.float64).eps * max(
        float(singular_values[0]), 1.0
    )
    rank = int(np.sum(singular_values > tolerance))
    if rank == 0:
        raise ProposalValidationError("recoverable support basis has no numerical rank")
    vectors = left[:, :rank]
    return vectors @ vectors.T


def audit_recoverable_support(
    requested_logical_action: Sequence[float],
    materialized_logical_action: Sequence[float],
    recoverable_command_basis: Sequence[Sequence[float]],
    *,
    leakage_tolerance: float = 1e-8,
    maximum_requested_to_materialized_error: float = 1e-8,
    minimum_direction_cosine: float = 0.999999,
) -> RecoverableSupportAuditV1:
    """Measure post-materialization leakage instead of trusting clipping."""

    requested = np.asarray(requested_logical_action, dtype=np.float64)
    materialized = np.asarray(materialized_logical_action, dtype=np.float64)
    if requested.ndim != 1 or materialized.shape != requested.shape:
        raise ProposalValidationError("logical actions must be matching 1D vectors")
    if not np.all(np.isfinite(requested)) or not np.all(np.isfinite(materialized)):
        raise ProposalValidationError("logical actions contain non-finite values")
    if leakage_tolerance < 0.0 or not np.isfinite(leakage_tolerance):
        raise ProposalValidationError("leakage_tolerance must be finite and non-negative")
    if (
        maximum_requested_to_materialized_error < 0.0
        or not np.isfinite(maximum_requested_to_materialized_error)
    ):
        raise ProposalValidationError(
            "maximum_requested_to_materialized_error must be finite and non-negative"
        )
    if not -1.0 <= minimum_direction_cosine <= 1.0:
        raise ProposalValidationError("minimum_direction_cosine must lie in [-1,1]")
    projector = support_projector(recoverable_command_basis, requested.size)
    projected = projector @ materialized
    leakage = materialized - projected
    requested_norm = float(np.linalg.norm(requested))
    materialized_norm = float(np.linalg.norm(materialized))
    leakage_norm = float(np.linalg.norm(leakage))
    leakage_fraction = leakage_norm / max(materialized_norm, 1e-12)
    support_fraction = float(np.linalg.norm(projected)) / max(materialized_norm, 1e-12)
    logical_error = float(np.linalg.norm(materialized - requested))
    denominator = requested_norm * materialized_norm
    # A true no-op can be direction-preserving only when it was requested as a
    # no-op.  Treating a nonzero request that got materialized to zero as cosine
    # one would let clipping erase the requested direction while still passing a
    # subspace-only audit.
    if requested_norm <= 1.0e-15 and materialized_norm <= 1.0e-15:
        direction = 1.0
    elif denominator <= 1.0e-15:
        direction = 0.0
    else:
        direction = float(np.clip(np.dot(requested, materialized) / denominator, -1.0, 1.0))
    reasons = []
    if leakage_norm > leakage_tolerance:
        reasons.append("recoverable_support_leakage_exceeded")
    if logical_error > maximum_requested_to_materialized_error:
        reasons.append("requested_to_materialized_logical_error_exceeded")
    if direction < minimum_direction_cosine:
        reasons.append("materialized_direction_cosine_below_minimum")
    return RecoverableSupportAuditV1(
        requested_norm=requested_norm,
        materialized_norm=materialized_norm,
        support_leakage_norm=leakage_norm,
        support_leakage_fraction=leakage_fraction,
        requested_to_materialized_logical_error=logical_error,
        direction_cosine=direction,
            recoverable_support_fraction=support_fraction,
            leakage_tolerance=float(leakage_tolerance),
            maximum_requested_to_materialized_error=float(
                maximum_requested_to_materialized_error
            ),
            minimum_direction_cosine=float(minimum_direction_cosine),
        passed=not reasons,
        reasons=tuple(reasons),
    )


def shrink_action_ray_until_supported(
    requested_logical_action: Sequence[float],
    *,
    recoverable_command_basis: Sequence[Sequence[float]],
    materialize_preview: Callable[[Tuple[float, ...]], Sequence[float]],
    leakage_tolerance: float = 1e-8,
    maximum_requested_to_materialized_error: float = 1e-8,
    minimum_direction_cosine: float = 0.999999,
    shrink_factor: float = 0.5,
    maximum_shrinks: int = 6,
) -> RayShrinkResultV1:
    """Retry a preview on one ray only; no per-coordinate repair is permitted."""

    requested = np.asarray(requested_logical_action, dtype=np.float64)
    if requested.ndim != 1 or not np.all(np.isfinite(requested)):
        raise ProposalValidationError("requested logical action must be finite and 1D")
    if not 0.0 < shrink_factor < 1.0:
        raise ProposalValidationError("shrink_factor must lie strictly between zero and one")
    if maximum_shrinks < 0:
        raise ProposalValidationError("maximum_shrinks must be non-negative")
    latest: RayShrinkResultV1 | None = None
    for attempt in range(maximum_shrinks + 1):
        ray_scale = shrink_factor ** attempt
        scaled = requested * ray_scale
        materialized = np.asarray(
            materialize_preview(tuple(float(value) for value in scaled)), dtype=np.float64,
        )
        audit = audit_recoverable_support(
            scaled, materialized, recoverable_command_basis,
            leakage_tolerance=leakage_tolerance,
            maximum_requested_to_materialized_error=(
                maximum_requested_to_materialized_error
            ),
            minimum_direction_cosine=minimum_direction_cosine,
        )
        latest = RayShrinkResultV1(
            ray_scale=float(ray_scale),
            attempts=attempt + 1,
            requested_logical_action=tuple(float(value) for value in scaled),
            materialized_logical_action=tuple(float(value) for value in materialized),
            audit=audit,
            accepted=audit.passed,
            reasons=audit.reasons,
        )
        if audit.passed:
            return latest
    if latest is None:
        raise RuntimeError("ray shrink did not produce an audit")
    return RayShrinkResultV1(
        ray_scale=latest.ray_scale,
        attempts=latest.attempts,
        requested_logical_action=latest.requested_logical_action,
        materialized_logical_action=latest.materialized_logical_action,
        audit=latest.audit,
        accepted=False,
        reasons=tuple(dict.fromkeys((*latest.reasons, "ray_shrink_exhausted"))),
    )


__all__ = [
    "RayShrinkResultV1",
    "RecoverableSupportAuditV1",
    "audit_recoverable_support",
    "shrink_action_ray_until_supported",
    "support_projector",
]
