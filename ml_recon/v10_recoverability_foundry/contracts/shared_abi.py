"""Portable immutable V2 semantic ABI shared by Track A and Track B.

Only semantic data crosses this boundary: no policy weights, hardware adapters,
or mutable tensor aliases are accepted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np

from ml_recon.physics_hybrid_v10.contracts import sha256_of


SEMANTIC_ABI_SCHEMA_V2 = "V10SemanticABIManifestV2"
PUBLIC_ADAPTER_DIM = 24
FROZEN_BUNDLE_STATUSES = frozenset({"NOT_READY_STUB", "FROZEN_READY"})
PHYSICAL_BUDGET_STATUSES = frozenset(
    {"SYNTHETIC_DECLARED", "ACTUAL_REPEAT_BOUND", "ACTUAL_REPEAT_UNBOUND"}
)


def _sha(value: Any, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _git_sha(value: Any, name: str) -> None:
    if (
        type(value) is not str
        or len(value) not in (40, 64)
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase Git digest")


def _freeze(value: Any, name: str) -> Any:
    """Snapshot torch/NumPy/list input into nested immutable finite tuples."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item, name) for item in value)
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must contain finite numeric values")
    return float(value)


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        return ()
    if not value:
        return (0,)
    shapes = tuple(_shape(item) for item in value)
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError("array payload must be rectangular")
    return (len(value),) + shapes[0]


def _array(value: Any, name: str, ndim: int | None = None) -> tuple[int, ...]:
    shape = _shape(value)
    if not shape or 0 in shape or (ndim is not None and len(shape) != ndim):
        raise ValueError(f"{name} has an invalid shape")
    return shape


def _flat(value: Any) -> tuple[float, ...]:
    if isinstance(value, tuple):
        return tuple(item for child in value for item in _flat(child))
    return (value,)


def _psd(matrix: Any, name: str) -> None:
    shape = _array(matrix, name, 2)
    if shape[0] != shape[1]:
        raise ValueError(f"{name} must be square")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            if abs(value - matrix[column_index][row_index]) > 1e-8:
                raise ValueError(f"{name} must be symmetric")
        if row[row_index] < -1e-10:
            raise ValueError(f"{name} must be positive semidefinite")
    # Nonnegative diagonal and 2x2 minors do not establish PSD for dimension >=3.
    values = np.asarray(matrix, dtype=np.float64)
    tolerance = 1e-10 * max(1.0, float(np.linalg.norm(values, ord=2)))
    if float(np.linalg.eigvalsh(values)[0]) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")


@dataclass(frozen=True)
class V10MeasurementBudgetContextV2:
    physical_budget_status: str
    source_kind: str
    n_mode: int | None
    b_trial: int | None
    b_mode_shot: int | None
    repeats_by_operation: tuple[int, ...] = ()
    reason: str = ""
    source_manifest_sha: str | None = None

    def validate(self) -> None:
        if (
            self.physical_budget_status not in PHYSICAL_BUDGET_STATUSES
            or self.source_kind
            not in {"synthetic_fixture", "source_bound_real", "unbound"}
            or type(self.reason) is not str
            or not self.reason
        ):
            raise ValueError("invalid measurement budget status, provenance, or reason")
        if (
            self.physical_budget_status == "SYNTHETIC_DECLARED"
            and self.source_kind != "synthetic_fixture"
        ):
            raise ValueError("synthetic budget requires synthetic_fixture provenance")
        if (
            self.physical_budget_status != "SYNTHETIC_DECLARED"
            and self.source_kind != "source_bound_real"
        ):
            raise ValueError("actual budget requires source_bound_real provenance")
        if self.source_manifest_sha is not None:
            _sha(self.source_manifest_sha, "source_manifest_sha")
        if self.physical_budget_status == "ACTUAL_REPEAT_UNBOUND":
            if (
                any(
                    value is not None
                    for value in (self.n_mode, self.b_trial, self.b_mode_shot)
                )
                or self.repeats_by_operation
            ):
                raise ValueError("unbound actual budget must not invent counts")
            return
        if (
            type(self.n_mode) is not int
            or self.n_mode <= 0
            or type(self.b_trial) is not int
            or self.b_trial < 0
            or type(self.b_mode_shot) is not int
            or self.b_mode_shot < 0
        ):
            raise ValueError("bound budget requires integer counts")
        if (
            type(self.repeats_by_operation) is not tuple
            or not self.repeats_by_operation
            or any(
                type(item) is not int or item < 0 for item in self.repeats_by_operation
            )
        ):
            raise ValueError(
                "bound budget requires immutable nonnegative operation repeats"
            )
        if (
            self.b_trial != sum(self.repeats_by_operation)
            or self.b_mode_shot != self.n_mode * self.b_trial
        ):
            raise ValueError("budget violates N * sum(repeats) conservation")


Semantic_budget_v2 = V10MeasurementBudgetContextV2


def semantic_budget_from_physical_budget(
    physical_budget: object,
) -> V10MeasurementBudgetContextV2:
    """Adapt declared synthetic or verifier-confirmed real repeat evidence only."""
    provenance = getattr(physical_budget, "provenance", None)
    if provenance is None:
        raise ValueError("physical budget provenance is required")
    status = getattr(provenance, "actual_repeat_status", None)
    source_kind = getattr(provenance, "source_kind", None)
    manifest_sha = getattr(provenance, "source_manifest_sha", None)
    if status == "ACTUAL_REPEAT_UNBOUND":
        result = V10MeasurementBudgetContextV2(
            status,
            "source_bound_real",
            None,
            None,
            None,
            (),
            "ACTUAL_REPEAT_UNBOUND",
            manifest_sha,
        )
    else:
        if status == "ACTUAL_REPEAT_BOUND":
            from .physical_budget import verify_actual_repeat_manifest

            verify_actual_repeat_manifest(provenance)
        if status not in {"SYNTHETIC_DECLARED", "ACTUAL_REPEAT_BOUND"}:
            raise ValueError("physical budget repeat status is required")
        result = V10MeasurementBudgetContextV2(
            status,
            source_kind,
            getattr(physical_budget, "n_mode", None),
            getattr(physical_budget, "b_trial", None),
            getattr(physical_budget, "b_mode_shot", None),
            tuple(getattr(physical_budget, "repeats_by_operation", ())),
            status,
            manifest_sha,
        )
    result.validate()
    return result


@dataclass(frozen=True)
class V10SharedBeliefContextV2:
    mode_posterior_mean: Any
    mode_posterior_covariance: Any
    sector_posterior: Any
    nuisance_posterior: Any
    measurement_support: Any
    measurement_budget: V10MeasurementBudgetContextV2
    confidence: Any
    provenance_sha: str
    valid_mode_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        for name in (
            "mode_posterior_mean",
            "mode_posterior_covariance",
            "sector_posterior",
            "nuisance_posterior",
            "measurement_support",
            "confidence",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name), name))

    def validate(self) -> None:
        count = _array(self.mode_posterior_mean, "mode_posterior_mean", 1)[0]
        if _array(self.mode_posterior_covariance, "mode_posterior_covariance", 2) != (
            count,
            count,
        ):
            raise ValueError("belief covariance shape must match the mode posterior")
        _psd(self.mode_posterior_covariance, "mode_posterior_covariance")
        for name in (
            "sector_posterior",
            "nuisance_posterior",
            "measurement_support",
            "confidence",
        ):
            _array(getattr(self, name), name)
        if (
            type(self.valid_mode_mask) is not tuple
            or len(self.valid_mode_mask) != count
            or any(type(item) is not bool for item in self.valid_mode_mask)
        ):
            raise ValueError("valid_mode_mask must be immutable and match mode count")
        self.measurement_budget.validate()
        _sha(self.provenance_sha, "belief provenance_sha")

    @property
    def mode_values(self) -> tuple[float, ...]:
        return self.mode_posterior_mean

    @property
    def covariance_sha(self) -> str:
        return sha256_of(self.mode_posterior_covariance)

    @property
    def observation_provenance_sha(self) -> str:
        return self.provenance_sha


@dataclass(frozen=True)
class ESMPartitionContextV2:
    pair_evidence: Any
    hyperedge_evidence: Any
    legal_partition_candidates: Any
    legal_partition_probabilities: Any
    co_membership: Any
    partition_entropy: Any
    current_target_keep_form_break: Any
    certificate_conflict: Any
    confidence: Any
    provenance_sha: str

    def __post_init__(self) -> None:
        for name in (
            "pair_evidence",
            "hyperedge_evidence",
            "legal_partition_candidates",
            "legal_partition_probabilities",
            "co_membership",
            "partition_entropy",
            "current_target_keep_form_break",
            "certificate_conflict",
            "confidence",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name), name))

    def validate(self) -> None:
        _array(self.pair_evidence, "pair_evidence")
        _array(self.hyperedge_evidence, "hyperedge_evidence")
        candidates = _array(
            self.legal_partition_candidates, "legal_partition_candidates", 2
        )
        probabilities = _array(
            self.legal_partition_probabilities, "legal_partition_probabilities", 1
        )
        if (
            candidates[0] != probabilities[0]
            or any(
                value < 0.0 or value > 1.0
                for value in self.legal_partition_probabilities
            )
            or abs(sum(self.legal_partition_probabilities) - 1.0) > 1e-6
        ):
            raise ValueError("legal partitions require a normalized posterior")
        for candidate in self.legal_partition_candidates:
            if any(value < 0.0 or not value.is_integer() for value in candidate):
                raise ValueError(
                    "legal partition candidates must contain integer labels"
                )
            labels = {int(value) for value in candidate}
            if labels != set(range(max(labels) + 1)):
                raise ValueError(
                    "legal partition labels must be canonical and gap-free"
                )
        co_membership = _array(self.co_membership, "co_membership", 2)
        if co_membership[0] != co_membership[1]:
            raise ValueError("co_membership must be square")
        if (
            _array(
                self.current_target_keep_form_break, "current_target_keep_form_break"
            )[-1]
            != 3
        ):
            raise ValueError("keep/form/break requires three channels")
        for name in ("partition_entropy", "certificate_conflict", "confidence"):
            _array(getattr(self, name), name)
        _sha(self.provenance_sha, "ESM provenance_sha")


@dataclass(frozen=True)
class CSM_posterior_v2:
    node_mean: Any
    node_variance: Any
    edge_mean: Any
    edge_variance: Any
    rank_distribution: Any
    condition_distribution: Any
    subspace_uncertainty: Any
    confidence: Any

    def __post_init__(self) -> None:
        for name in (
            "node_mean",
            "node_variance",
            "edge_mean",
            "edge_variance",
            "rank_distribution",
            "condition_distribution",
            "subspace_uncertainty",
            "confidence",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name), name))

    def validate(self) -> None:
        for name in (
            "node_mean",
            "node_variance",
            "edge_mean",
            "edge_variance",
            "rank_distribution",
            "condition_distribution",
            "subspace_uncertainty",
            "confidence",
        ):
            _array(getattr(self, name), name)
        if _shape(self.node_mean) != _shape(self.node_variance) or _shape(
            self.edge_mean
        ) != _shape(self.edge_variance):
            raise ValueError("posterior mean and variance shapes must match")
        if any(
            value < 0.0
            for value in _flat(self.node_variance) + _flat(self.edge_variance)
        ):
            raise ValueError("posterior variance must be nonnegative")
        if (
            _shape(self.rank_distribution)[0] < 2
            or _shape(self.condition_distribution)[0] < 2
        ):
            raise ValueError(
                "posterior distributions need at least two draws; a point estimate is not posterior CSM"
            )


@dataclass(frozen=True)
class CSMGeometrySidecarV2:
    node_context: Any
    edge_context: Any
    global_context: Any
    singular_values: Any
    recoverable_basis: Any
    rank_fraction: Any
    condition: Any
    raw_strength: Any
    per_shot_shape: Any
    confidence: Any
    provenance_sha: str
    forward_instance_sha: str
    posterior: CSM_posterior_v2 | None = None
    target_independent: bool = True

    def __post_init__(self) -> None:
        for name in (
            "node_context",
            "edge_context",
            "global_context",
            "singular_values",
            "recoverable_basis",
            "rank_fraction",
            "condition",
            "raw_strength",
            "per_shot_shape",
            "confidence",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name), name))

    def validate(self) -> None:
        if self.target_independent is not True:
            raise ValueError("CSM must be target independent")
        for name in (
            "node_context",
            "edge_context",
            "global_context",
            "singular_values",
            "recoverable_basis",
            "rank_fraction",
            "condition",
            "raw_strength",
            "per_shot_shape",
            "confidence",
        ):
            _array(getattr(self, name), name)
        _sha(self.provenance_sha, "CSM provenance_sha")
        _sha(self.forward_instance_sha, "CSM forward_instance_sha")
        if self.posterior is not None:
            self.posterior.validate()

    def sha256(self) -> str:
        self.validate()
        return sha256_of(asdict(self))


@dataclass(frozen=True)
class RSM_context_v2:
    kappa_geometry: Any
    epsilon_irrecoverable: Any
    action_snr: Any
    direction_stability: Any
    reason_code: Any
    target_coefficients_sha: str
    target_residual_summary: Any = ()

    def __post_init__(self) -> None:
        for name in (
            "kappa_geometry",
            "epsilon_irrecoverable",
            "action_snr",
            "direction_stability",
            "reason_code",
            "target_residual_summary",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name), name))

    def validate(self) -> None:
        for name in (
            "kappa_geometry",
            "epsilon_irrecoverable",
            "action_snr",
            "direction_stability",
            "reason_code",
        ):
            _array(getattr(self, name), name)
        _sha(self.target_coefficients_sha, "target_coefficients_sha")


@dataclass(frozen=True)
class RSMDecisionContextV2:
    rsm: RSM_context_v2
    csm_context_sha: str
    bounded_action_sha: str
    same_forward_provenance_sha: str

    def validate(self) -> None:
        self.rsm.validate()
        for name in (
            "csm_context_sha",
            "bounded_action_sha",
            "same_forward_provenance_sha",
        ):
            _sha(getattr(self, name), name)


@dataclass(frozen=True)
class Semantic_abi_manifest_v2:
    source_commit: str
    source_tree_sha: str
    coordinate_layout_sha: str
    mode_order_sha: str
    phase_convention_sha: str
    normalization_sha: str
    budget_schema_sha: str
    certificate_schema_sha: str
    esm_schema_sha: str
    csm_schema_sha: str
    rsm_schema_sha: str
    provenance_sha: str
    frozen_bundle_status: str = "NOT_READY_STUB"
    checkpoint_sha: str | None = None
    model_state_sha: str | None = None
    frozen_core_admitted: bool = False
    source_bound_calibration_acceptance_sha: str | None = None
    checkpoint_evidence_sha: str | None = None
    evidence_verifier_path: str | None = None
    public_adapter_dim: int = PUBLIC_ADAPTER_DIM
    schema: str = SEMANTIC_ABI_SCHEMA_V2

    def validate(self) -> None:
        if (
            self.schema != SEMANTIC_ABI_SCHEMA_V2
            or self.frozen_bundle_status not in FROZEN_BUNDLE_STATUSES
            or self.public_adapter_dim != PUBLIC_ADAPTER_DIM
            or type(self.frozen_core_admitted) is not bool
        ):
            raise ValueError("invalid semantic ABI manifest")
        _git_sha(self.source_commit, "source_commit")
        _git_sha(self.source_tree_sha, "source_tree_sha")
        for name in (
            "coordinate_layout_sha",
            "mode_order_sha",
            "phase_convention_sha",
            "normalization_sha",
            "budget_schema_sha",
            "certificate_schema_sha",
            "esm_schema_sha",
            "csm_schema_sha",
            "rsm_schema_sha",
            "provenance_sha",
        ):
            _sha(getattr(self, name), name)
        if self.frozen_bundle_status == "NOT_READY_STUB":
            if self.frozen_core_admitted or any(
                value is not None
                for value in (
                    self.checkpoint_sha,
                    self.model_state_sha,
                    self.source_bound_calibration_acceptance_sha,
                    self.checkpoint_evidence_sha,
                    self.evidence_verifier_path,
                )
            ):
                raise ValueError("not-ready stub must not claim frozen evidence")
            return
        for name in (
            "checkpoint_sha",
            "model_state_sha",
            "source_bound_calibration_acceptance_sha",
            "checkpoint_evidence_sha",
        ):
            _sha(getattr(self, name), name)
        if (
            not self.frozen_core_admitted
            or type(self.evidence_verifier_path) is not str
            or not self.evidence_verifier_path
        ):
            raise ValueError(
                "frozen-ready requires source-bound acceptance and verifier path"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def sha256(self) -> str:
        return sha256_of(self.to_dict())


@dataclass(frozen=True)
class Mainline_bundle_stub:
    manifest: Semantic_abi_manifest_v2
    budget: V10MeasurementBudgetContextV2 = field(
        default_factory=lambda: V10MeasurementBudgetContextV2(
            "ACTUAL_REPEAT_UNBOUND",
            "source_bound_real",
            None,
            None,
            None,
            (),
            "FROZEN_TRACK_A_CORE_NOT_ADMITTED",
        )
    )

    def validate(self) -> None:
        self.manifest.validate()
        self.budget.validate()
        if self.manifest.frozen_bundle_status != "NOT_READY_STUB":
            raise ValueError("Mainline_bundle_stub is always NOT_READY_STUB")


V10SemanticABIManifestV2 = Semantic_abi_manifest_v2
MainlineBundleStub = Mainline_bundle_stub

__all__ = [
    "CSM_posterior_v2",
    "CSMGeometrySidecarV2",
    "ESMPartitionContextV2",
    "FROZEN_BUNDLE_STATUSES",
    "MainlineBundleStub",
    "Mainline_bundle_stub",
    "PHYSICAL_BUDGET_STATUSES",
    "PUBLIC_ADAPTER_DIM",
    "RSM_context_v2",
    "RSMDecisionContextV2",
    "SEMANTIC_ABI_SCHEMA_V2",
    "Semantic_abi_manifest_v2",
    "Semantic_budget_v2",
    "V10MeasurementBudgetContextV2",
    "V10SemanticABIManifestV2",
    "V10SharedBeliefContextV2",
    "semantic_budget_from_physical_budget",
]
