"""Immutable, policy-free V10 feedback decision input contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .shared_abi import (
    CSMGeometrySidecarV2,
    ESMPartitionContextV2,
    RSMDecisionContextV2,
    V10MeasurementBudgetContextV2,
    V10SharedBeliefContextV2,
    _freeze,
    _sha,
    sha256_of,
)

FEEDBACK_DECISION_INPUT_SCHEMA = "V10FeedbackDecisionInputV2"
ESM_UNAVAILABLE_REASON = "ESM_UNAVAILABLE_NO_LEGAL_POSTERIOR"


def _finite(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


@dataclass(frozen=True)
class V10FeedbackDecisionInputV2:
    belief: V10SharedBeliefContextV2
    csm: CSMGeometrySidecarV2
    rsm: RSMDecisionContextV2
    budget: V10MeasurementBudgetContextV2
    observed_four_cell_counts: tuple[tuple[float, float, float, float], ...]
    observed_operation_mask: tuple[bool, ...]
    observed_repeats_by_operation: tuple[int, ...]
    observed_mode_coefficients: tuple[tuple[float, float], ...]
    joint_control_covariance: tuple[tuple[float, ...], ...]
    desired_target_coefficients: tuple[tuple[float, float], ...]
    analytic_mode_action: tuple[tuple[float, float], ...]
    esm: ESMPartitionContextV2 | None = None
    esm_unavailable_reason: str = ESM_UNAVAILABLE_REASON
    history: tuple[Any, ...] = ()
    success_fusion: tuple[Any, ...] = ()
    success_fusion_mask: tuple[bool, ...] = ()
    success_fusion_provenance_sha: str | None = None
    schema: str = FEEDBACK_DECISION_INPUT_SCHEMA

    def __post_init__(self) -> None:
        numeric_fields = (
            "observed_four_cell_counts",
            "observed_mode_coefficients",
            "joint_control_covariance",
            "desired_target_coefficients",
            "analytic_mode_action",
            "history",
            "success_fusion",
        )
        for name in numeric_fields:
            object.__setattr__(self, name, _freeze(getattr(self, name), name))
        for name in ("observed_operation_mask", "success_fusion_mask"):
            value = tuple(getattr(self, name))
            if any(type(item) is not bool for item in value):
                raise ValueError(f"{name} must be a strict immutable bool tuple")
            object.__setattr__(self, name, value)
        repeats = tuple(self.observed_repeats_by_operation)
        if any(type(item) is not int or item < 0 for item in repeats):
            raise ValueError("observed repeats must be strict nonnegative integers")
        object.__setattr__(self, "observed_repeats_by_operation", repeats)

    def validate(self) -> None:
        if self.schema != FEEDBACK_DECISION_INPUT_SCHEMA:
            raise ValueError("unsupported feedback decision input schema")
        self.belief.validate()
        self.csm.validate()
        self.rsm.validate()
        self.budget.validate()
        if self.csm.sha256() != self.rsm.csm_context_sha:
            raise ValueError("RSM must bind the exact CSM sidecar")
        if self.csm.forward_instance_sha != self.rsm.same_forward_provenance_sha:
            raise ValueError("CSM and RSM must bind the same forward provenance")
        if self.budget != self.belief.measurement_budget:
            raise ValueError("feedback budget must equal belief measurement budget")
        n_mode = len(self.belief.valid_mode_mask)
        count = len(self.observed_four_cell_counts)
        if (
            count == 0
            or len(self.observed_operation_mask) != count
            or len(self.observed_repeats_by_operation) != count
            or any(type(item) is not bool for item in self.observed_operation_mask)
            or any(
                type(item) is not int or item < 0
                for item in self.observed_repeats_by_operation
            )
        ):
            raise ValueError("measurement arrays and masks must align")
        counts = _finite(
            self.observed_four_cell_counts, (count, 4), "observed_four_cell_counts"
        )
        if (
            np.any(counts < 0)
            or not np.array_equal(counts, np.rint(counts))
            or not np.array_equal(
                counts.sum(axis=1), self.observed_repeats_by_operation
            )
            or any(
                bool(mask) != bool(repeat > 0)
                for mask, repeat in zip(
                    self.observed_operation_mask, self.observed_repeats_by_operation
                )
            )
            or any(
                not mask and bool(np.any(row))
                for mask, row in zip(self.observed_operation_mask, counts)
            )
        ):
            raise ValueError(
                "four-cell integer conservation or observation mask is inconsistent"
            )
        if self.budget.repeats_by_operation and (
            len(self.budget.repeats_by_operation) != count
            or any(
                visible > physical
                for visible, physical in zip(
                    self.observed_repeats_by_operation,
                    self.budget.repeats_by_operation,
                )
            )
        ):
            raise ValueError(
                "visible repeats must be bounded by the full measurement budget"
            )
        if self.budget.n_mode != n_mode:
            raise ValueError("measurement budget N must match feedback mode count")
        observed = _finite(
            self.observed_mode_coefficients, (n_mode, 2), "observed_mode_coefficients"
        )
        if not np.isclose(np.linalg.norm(observed), 1.0, atol=1e-6):
            raise ValueError("observed mode coefficients must be normalized")
        target = _finite(
            self.desired_target_coefficients, (n_mode, 2), "desired_target_coefficients"
        )
        if not np.isclose(np.linalg.norm(target), 1.0, atol=1e-6):
            raise ValueError("desired target coefficients must be normalized")
        if self.rsm.rsm.target_coefficients_sha != sha256_of(
            self.desired_target_coefficients
        ):
            raise ValueError("RSM must bind the canonical desired target coefficients")
        logical = 2 * n_mode - 2
        covariance = _finite(
            self.joint_control_covariance,
            (logical, logical),
            "joint_control_covariance",
        )
        if (
            not np.allclose(covariance, covariance.T, atol=1e-8)
            or np.linalg.eigvalsh(covariance)[0] < -1e-8
        ):
            raise ValueError(
                "joint control covariance must be symmetric positive semidefinite"
            )
        action = _finite(self.analytic_mode_action, (n_mode, 2), "analytic_mode_action")
        active = np.asarray(self.belief.valid_mode_mask, dtype=bool)
        if (
            not active.all()
            or not np.allclose(action[active].sum(0), 0.0, atol=1e-6)
            or np.any(action[~active])
        ):
            raise ValueError(
                "analytic action must be gauge-free and respect active support"
            )
        if self.esm is None:
            if self.esm_unavailable_reason != ESM_UNAVAILABLE_REASON:
                raise ValueError("unavailable ESM requires its explicit reason")
        else:
            self.esm.validate()
            if self.esm_unavailable_reason:
                raise ValueError("available ESM cannot carry an unavailable reason")
        if len(self.success_fusion) != len(self.success_fusion_mask) or any(
            type(item) is not bool for item in self.success_fusion_mask
        ):
            raise ValueError("success/fusion values require aligned immutable masks")
        if self.success_fusion:
            if self.success_fusion_provenance_sha is None:
                raise ValueError("success/fusion values require provenance")
            _sha(self.success_fusion_provenance_sha, "success_fusion_provenance_sha")
        elif self.success_fusion_provenance_sha is not None:
            raise ValueError("masked success/fusion cannot claim provenance")
        for name in ("history", "success_fusion"):
            try:
                values = np.asarray(getattr(self, name), dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{name} must be immutable finite numeric values"
                ) from error
            if values.size and not np.isfinite(values).all():
                raise ValueError(f"{name} must be immutable finite numeric values")

    def sha256(self) -> str:
        self.validate()
        return sha256_of(asdict(self))


__all__ = [
    "ESM_UNAVAILABLE_REASON",
    "FEEDBACK_DECISION_INPUT_SCHEMA",
    "V10FeedbackDecisionInputV2",
]
