"""Metric-bound, target-dependent Track B recovery support.

The RSM is built only after a target-independent CSM has been registered. It
records a typed no-support state rather than fabricating a finite condition or
letting callers replace the numerical reason code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from .contracts import canonical_sha256, require_sha256
from .real_csm import RealCSMSupportV1, array_identity_sha256


RSM_REASON_CODES = frozenset({
    "SHOT_LIMITED", "BASIS_LIMITED", "NUISANCE_CONFOUNDED", "SECTOR_AMBIGUOUS",
    "APPARATUS_MODEL_MISMATCH", "NO_RECOVERABLE_SUPPORT", "RECOVERABLE_SUPPORT_PRESENT",
    "ZERO_TARGET_RESIDUAL",
})
RSM_SUPPORT_STATUSES = frozenset({
    "LEGACY_METRIC_UNBOUND",
    "METRIC_BOUND_RECOVERABLE_SUPPORT",
    "METRIC_BOUND_NO_RECOVERABLE_SUPPORT",
    "METRIC_BOUND_ZERO_TARGET_RESIDUAL",
})


def _metric_matrix(
    values: np.ndarray | Sequence[Sequence[float]], dimension: int,
    *, contract: "MetricProjectionContractV1",
) -> np.ndarray:
    metric = np.asarray(values, dtype=np.float64)
    if metric.shape != (dimension, dimension) or not np.all(np.isfinite(metric)):
        raise ValueError("metric must be finite, square, and align to target residual")
    if not np.allclose(
        metric, metric.T, atol=contract.absolute_tolerance, rtol=contract.relative_tolerance,
    ):
        raise ValueError("metric must be symmetric")
    eigenvalues = np.linalg.eigvalsh(metric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -contract.psd_tolerance * scale:
        raise ValueError("metric must be positive semidefinite")
    return 0.5 * (metric + metric.T)


@dataclass(frozen=True)
class MetricProjectionContractV1:
    """Hash-bound numerical convention for target-dependent metric projection."""

    metric_input_sha: str
    absolute_tolerance: float = 1.0e-12
    relative_tolerance: float = 1.0e-10
    psd_tolerance: float = 1.0e-10
    pinv_rcond: float = 1.0e-10
    zero_energy_tolerance: float = 1.0e-15
    projection_tolerance: float = 1.0e-10
    schema: str = field(default="MetricProjectionContractV1", init=False)

    def __post_init__(self) -> None:
        require_sha256(self.metric_input_sha, "metric_input_sha")
        values = (
            self.absolute_tolerance, self.relative_tolerance, self.psd_tolerance,
            self.pinv_rcond, self.zero_energy_tolerance, self.projection_tolerance,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values):
            raise ValueError("metric projection tolerances must be finite and positive")

    @classmethod
    def from_metric(cls, metric: np.ndarray | Sequence[Sequence[float]]) -> "MetricProjectionContractV1":
        return cls(metric_input_sha=array_identity_sha256(np.asarray(metric, dtype=np.float64)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "metric_input_sha": self.metric_input_sha,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "psd_tolerance": self.psd_tolerance,
            "pinv_rcond": self.pinv_rcond,
            "zero_energy_tolerance": self.zero_energy_tolerance,
            "projection_tolerance": self.projection_tolerance,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _metric_project(
    residual: np.ndarray, basis: np.ndarray, metric: np.ndarray,
    contract: MetricProjectionContractV1,
) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.zeros_like(residual)
    reduced = 0.5 * (basis.T @ metric @ basis + (basis.T @ metric @ basis).T)
    eigenvalues = np.linalg.eigvalsh(reduced)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -contract.psd_tolerance * scale:
        raise ValueError("registered support is not valid under the supplied metric")
    inverse = np.linalg.pinv(reduced, rcond=contract.pinv_rcond)
    return basis @ (inverse @ (basis.T @ metric @ residual))


@dataclass(frozen=True)
class RealRSMSupportV1:
    desired_target_sha: str
    metric_contract_sha: str
    csm_sha: str
    supported_target_direction: tuple[float, ...]
    recoverable_residual_fraction: float
    irrecoverable_floor: float
    reason_code: str
    metric_input_sha: str | None = None
    metric_projection_contract_sha: str | None = None
    metric_bound: bool = False
    support_status: str = "LEGACY_METRIC_UNBOUND"
    schema: str = field(default="RealRSMSupportV1", init=False)

    def __post_init__(self) -> None:
        for name in ("desired_target_sha", "metric_contract_sha", "csm_sha"):
            require_sha256(getattr(self, name), name)
        if self.metric_input_sha is not None:
            require_sha256(self.metric_input_sha, "metric_input_sha")
        if self.metric_projection_contract_sha is not None:
            require_sha256(self.metric_projection_contract_sha, "metric_projection_contract_sha")
        vector = tuple(float(value) for value in self.supported_target_direction)
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("RSM supported direction must be finite and nonempty")
        if not 0.0 <= self.recoverable_residual_fraction <= 1.0 or not 0.0 <= self.irrecoverable_floor <= 1.0:
            raise ValueError("RSM fractions must lie in [0,1]")
        if not math.isclose(
            self.recoverable_residual_fraction + self.irrecoverable_floor,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise ValueError("metric-bound recoverable and irrecoverable fractions must sum to one")
        if self.reason_code not in RSM_REASON_CODES:
            raise ValueError("unknown RSM reason code")
        if self.support_status not in RSM_SUPPORT_STATUSES:
            raise ValueError("unknown RSM support status")
        if type(self.metric_bound) is not bool:
            raise TypeError("metric_bound must be an exact boolean")
        if self.metric_bound:
            if (
                self.metric_input_sha is None
                or self.metric_projection_contract_sha is None
                or self.metric_projection_contract_sha != self.metric_contract_sha
                or self.support_status == "LEGACY_METRIC_UNBOUND"
            ):
                raise ValueError("metric-bound RSM requires the exact typed projection contract")
        elif (
            self.metric_input_sha is not None
            or self.metric_projection_contract_sha is not None
            or self.support_status != "LEGACY_METRIC_UNBOUND"
        ):
            raise ValueError("unbound RSM must use the legacy unbound metric status")
        no_support = self.recoverable_residual_fraction <= 1.0e-12
        if self.support_status == "METRIC_BOUND_NO_RECOVERABLE_SUPPORT" and not no_support:
            raise ValueError("no-support status disagrees with recoverable residual fraction")
        if self.support_status == "METRIC_BOUND_RECOVERABLE_SUPPORT" and no_support:
            raise ValueError("recoverable-support status disagrees with recoverable residual fraction")
        if self.support_status == "METRIC_BOUND_ZERO_TARGET_RESIDUAL" and self.reason_code != "ZERO_TARGET_RESIDUAL":
            raise ValueError("zero-residual status requires the matching typed reason")
        if self.reason_code == "ZERO_TARGET_RESIDUAL" and self.support_status != "METRIC_BOUND_ZERO_TARGET_RESIDUAL":
            raise ValueError("ZERO_TARGET_RESIDUAL reason requires the matching typed status")
        if self.reason_code == "NO_RECOVERABLE_SUPPORT" and not no_support:
            raise ValueError("NO_RECOVERABLE_SUPPORT reason disagrees with numerical fraction")
        if self.reason_code == "RECOVERABLE_SUPPORT_PRESENT" and no_support:
            raise ValueError("RECOVERABLE_SUPPORT_PRESENT reason disagrees with numerical fraction")
        object.__setattr__(self, "supported_target_direction", vector)

    @classmethod
    def from_target_residual(
        cls, target_residual: Sequence[float], *, desired_target_sha: str, metric_contract_sha: str,
        csm: RealCSMSupportV1, metric: np.ndarray | Sequence[Sequence[float]] | None = None,
        reason_code: str | None = None,
    ) -> "RealRSMSupportV1":
        """Project residual with the supplied metric, or label a legacy unbound path.

        Supplying ``metric`` is required for proposal-eligible geometry. The
        identity-only compatibility path remains explicitly unbound and is
        rejected later by the proposal eligibility gate.
        """
        residual = np.asarray(tuple(target_residual), dtype=np.float64)
        if residual.ndim != 1 or residual.size == 0 or not np.all(np.isfinite(residual)):
            raise ValueError("target residual must be a finite explicit vector")
        basis = np.asarray(csm.support_basis, dtype=np.float64)
        if basis.shape[0] != residual.size:
            raise ValueError("target residual does not align to registered CSM support")
        metric_bound = metric is not None
        if metric_bound:
            projection_contract = MetricProjectionContractV1.from_metric(metric)
            actual_metric = _metric_matrix(
                metric, residual.size, contract=projection_contract,
            )
            if metric_contract_sha != projection_contract.sha256():
                raise ValueError(
                    "metric_contract_sha must bind the typed metric projection convention"
                )
        else:
            actual_metric = np.eye(residual.size)
            projection_contract = MetricProjectionContractV1.from_metric(actual_metric)
        projected = _metric_project(residual, basis, actual_metric, projection_contract)
        energy = float(residual @ actual_metric @ residual)
        projected_energy = float(projected @ actual_metric @ projected)
        if energy <= projection_contract.zero_energy_tolerance:
            fraction = 0.0
            status = "METRIC_BOUND_ZERO_TARGET_RESIDUAL" if metric_bound else "LEGACY_METRIC_UNBOUND"
            derived_reason = "ZERO_TARGET_RESIDUAL"
        else:
            fraction = float(np.clip(projected_energy / energy, 0.0, 1.0))
            if fraction <= projection_contract.projection_tolerance:
                status = "METRIC_BOUND_NO_RECOVERABLE_SUPPORT" if metric_bound else "LEGACY_METRIC_UNBOUND"
                derived_reason = "NO_RECOVERABLE_SUPPORT"
            else:
                status = "METRIC_BOUND_RECOVERABLE_SUPPORT" if metric_bound else "LEGACY_METRIC_UNBOUND"
                derived_reason = "RECOVERABLE_SUPPORT_PRESENT"
        if reason_code is not None and reason_code != derived_reason:
            raise ValueError("caller-supplied RSM reason disagrees with the metric projection result")
        return cls(
            desired_target_sha=desired_target_sha,
            metric_contract_sha=metric_contract_sha,
            csm_sha=csm.sha256(),
            supported_target_direction=tuple(float(value) for value in projected),
            recoverable_residual_fraction=fraction,
            irrecoverable_floor=1.0 - fraction,
            reason_code=derived_reason,
            metric_input_sha=array_identity_sha256(actual_metric) if metric_bound else None,
            metric_projection_contract_sha=(
                projection_contract.sha256() if metric_bound else None
            ),
            metric_bound=metric_bound,
            support_status=status,
        )

    @property
    def has_recoverable_support(self) -> bool:
        return self.support_status == "METRIC_BOUND_RECOVERABLE_SUPPORT"

    def sha256(self) -> str:
        return canonical_sha256({
            "schema": self.schema,
            "desired_target_sha": self.desired_target_sha,
            "metric_contract_sha": self.metric_contract_sha,
            "metric_input_sha": self.metric_input_sha,
            "metric_projection_contract_sha": self.metric_projection_contract_sha,
            "metric_bound": self.metric_bound,
            "support_status": self.support_status,
            "csm_sha": self.csm_sha,
            "supported_target_direction": list(self.supported_target_direction),
            "recoverable_residual_fraction": self.recoverable_residual_fraction,
            "irrecoverable_floor": self.irrecoverable_floor,
            "reason_code": self.reason_code,
        })


__all__ = [
    "MetricProjectionContractV1", "RSM_REASON_CODES", "RSM_SUPPORT_STATUSES", "RealRSMSupportV1",
]
