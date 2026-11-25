"""Machine-readable CSM-v1 contracts and provenance bindings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Dict, Tuple

from .contracts import sha256_of
from .control_manifold import (
    INTERLEAVED_AMP_PHASE,
    W_LOGAMP_RELATIVE_PHASE_V1,
    get_control_manifold,
)


CSM_SCHEMA = "V10CSMContractV1"
CSM_CONTRACT_REVISION = "1.4"
WHITENING_SUPPORT_KINDS = (
    "THRESHOLDED_COVARIANCE_EIGEN_SUPPORT",
    "EXACT_POSITIVE_MULTINOMIAL_SUPPORT",
)
OBSERVATION_CONTRACT_KINDS = (
    "CSM_ORACLE_CATEGORICAL_V1",
    "CSM_FEEDBACK_CARTESIAN_STATS_V1",
    "CSM_FEEDBACK_PHASE_PROXY_TOKENS_V1",
)
CARTESIAN_COVARIANCE_KIND = "cartesian_feedback_sufficient_stat_covariance"
GEOMETRY_SOURCES = (
    "BOUND_SAME_FORWARD",
    "PLUGIN_MEASURED_STATE_IDENTITY_FORWARD",
)
PARTITION_SCHEMA = "V10PartitionStructureV1"
ACTUATOR_SCHEMA = "V10ActuatorTransferV1"
FORBIDDEN_CSM_FIELDS = (
    "P_J_r", "JtSigma_r", "u_star", "oracle_action", "action_direction",
    "normal_equation_rhs",
)


@dataclass(frozen=True)
class WhiteningThresholdContract:
    absolute: float = 1e-10
    relative: float = 1e-8
    epsilon: float = 1e-12
    semantics: str = "support_only_strict_greater_than_max_abs_or_rel"

    def validate(self) -> None:
        if (not all(isfinite(value) for value in
                    (self.absolute, self.relative, self.epsilon))
                or self.absolute < 0.0 or self.relative < 0.0 or self.epsilon <= 0.0):
            raise ValueError("whitening thresholds must be non-negative/non-zero")
        if self.absolute == 0.0 and self.relative == 0.0:
            raise ValueError("support whitening needs a positive threshold")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    def sha256(self) -> str:
        return sha256_of(self.to_dict())


@dataclass(frozen=True)
class SingularSupportContract:
    """Effective singular-value support for CSM geometry diagnostics."""

    absolute: float = 1e-10
    relative: float = 1e-8
    semantics: str = "strict_greater_than_max_abs_or_rel_times_smax"

    def validate(self) -> None:
        if (not all(isfinite(value) for value in (self.absolute, self.relative))
                or self.absolute < 0.0 or self.relative < 0.0):
            raise ValueError("singular-support thresholds must be non-negative")
        if self.absolute == 0.0 and self.relative == 0.0:
            raise ValueError("singular support needs a positive threshold")

    def effective_threshold(self, singular_max: float) -> float:
        self.validate()
        if not isfinite(singular_max) or singular_max < 0.0:
            raise ValueError("singular maximum must be finite and non-negative")
        return max(self.absolute, self.relative * max(float(singular_max), 0.0))

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    def sha256(self) -> str:
        return sha256_of(self.to_dict())


@dataclass(frozen=True)
class FiniteDifferenceContract:
    h_amplitude: float = 1e-4
    h_phase: float = 1e-4
    plateau_tolerance: float = 1e-3
    forward_sha: str = ""
    manifold_sha: str = ""
    coordinate_layout: str = INTERLEAVED_AMP_PHASE
    status: str = "engineering_default_not_scientific_freeze"

    def validate(self, *, require_bound: bool = False) -> None:
        if (not all(isfinite(value) for value in
                    (self.h_amplitude, self.h_phase, self.plateau_tolerance))
                or self.h_amplitude <= 0.0 or self.h_phase <= 0.0):
            raise ValueError("finite-difference steps must be positive")
        if self.plateau_tolerance <= 0.0:
            raise ValueError("finite-difference plateau tolerance must be positive")
        for name in ("forward_sha", "manifold_sha"):
            value = getattr(self, name)
            if value and len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest when supplied")
            if require_bound and len(value) != 64:
                raise ValueError(f"{name} must be bound for a scientific CSM artifact")
        if self.coordinate_layout != INTERLEAVED_AMP_PHASE:
            raise ValueError("CSM-v1 finite differences require interleaved coordinates")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    def sha256(self) -> str:
        return sha256_of(self.to_dict())


@dataclass(frozen=True)
class CSMContractV1:
    N: int
    control_manifold_id: str = W_LOGAMP_RELATIVE_PHASE_V1
    logical_dim: int = 0
    mode_local_dim: int = 2
    coordinate_layout: str = INTERLEAVED_AMP_PHASE
    covariance_kind: str = "per_setting_sufficient_stat_blocks"
    geometry_source: str = "BOUND_SAME_FORWARD"
    observation_contract_kind: str = "CSM_ORACLE_CATEGORICAL_V1"
    whitening_support_kind: str = "THRESHOLDED_COVARIANCE_EIGEN_SUPPORT"
    whitening: WhiteningThresholdContract = field(default_factory=WhiteningThresholdContract)
    singular_support: SingularSupportContract = field(default_factory=SingularSupportContract)
    finite_difference: FiniteDifferenceContract = field(default_factory=FiniteDifferenceContract)
    materialization_policy: str = "ephemeral_or_compressed"
    edge_storage: str = "topk_or_runtime"
    forward_sha: str = ""
    jacobian_contract_sha: str = ""
    manifold_contract_sha: str = ""
    forward_instance_sha: str = ""
    feedback_measurement_schedule_sha: str = ""
    feedback_observation_contract_sha: str = ""
    statistic_transform_sha: str = ""
    forbidden_fields: Tuple[str, ...] = FORBIDDEN_CSM_FIELDS
    research_only: bool = True
    hardware_action_authorized: bool = False

    def __post_init__(self) -> None:
        manifold = get_control_manifold(self.control_manifold_id, self.N)
        if self.logical_dim == 0:
            object.__setattr__(self, "logical_dim", manifold.logical_dim)

    def validate(self, *, require_bound: bool = False) -> None:
        manifold = get_control_manifold(self.control_manifold_id, self.N)
        if self.logical_dim != manifold.logical_dim:
            raise ValueError("CSM logical dimension does not match control manifold")
        if self.mode_local_dim != manifold.mode_local_dim:
            raise ValueError("CSM mode-local dimension does not match manifold")
        if self.coordinate_layout != manifold.coordinate_layout:
            raise ValueError("CSM coordinate layout does not match manifold")
        if self.observation_contract_kind not in OBSERVATION_CONTRACT_KINDS:
            raise ValueError("unknown CSM observation contract kind")
        if self.geometry_source not in GEOMETRY_SOURCES:
            raise ValueError("unknown CSM geometry source")
        if self.whitening_support_kind not in WHITENING_SUPPORT_KINDS:
            raise ValueError("unknown CSM whitening support kind")
        if self.materialization_policy not in {"ephemeral_or_compressed", "eval_artifact"}:
            raise ValueError("unknown CSM materialization policy")
        if self.edge_storage not in {"topk_or_runtime", "dense_eval_only"}:
            raise ValueError("unknown CSM edge storage policy")
        self.whitening.validate()
        self.singular_support.validate()
        self.finite_difference.validate(require_bound=require_bound)
        for name in (
            "forward_sha", "jacobian_contract_sha", "manifold_contract_sha",
            "forward_instance_sha", "feedback_measurement_schedule_sha", "feedback_observation_contract_sha",
        ):
            value = getattr(self, name)
            if value and len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest when supplied")
            if require_bound and len(value) != 64:
                raise ValueError(f"{name} must be bound for a scientific CSM artifact")
        if (self.statistic_transform_sha
                and len(self.statistic_transform_sha) != 64):
            raise ValueError("statistic_transform_sha must be a SHA-256 digest when supplied")
        if (require_bound
                and self.observation_contract_kind == "CSM_FEEDBACK_CARTESIAN_STATS_V1"
                and len(self.statistic_transform_sha) != 64):
            raise ValueError(
                "Cartesian feedback CSM requires a bound statistic transform SHA")
        if self.observation_contract_kind == "CSM_FEEDBACK_CARTESIAN_STATS_V1":
            if self.covariance_kind != CARTESIAN_COVARIANCE_KIND:
                raise ValueError(
                    "Cartesian feedback CSM requires Cartesian covariance kind")
            if self.whitening_support_kind != "THRESHOLDED_COVARIANCE_EIGEN_SUPPORT":
                raise ValueError(
                    "Cartesian feedback CSM requires thresholded eigen support")
            if (require_bound
                    and self.geometry_source != "PLUGIN_MEASURED_STATE_IDENTITY_FORWARD"):
                raise ValueError(
                    "Cartesian feedback CSM requires explicit plugin geometry source")
        if tuple(self.forbidden_fields) != FORBIDDEN_CSM_FIELDS:
            raise ValueError("CSM forbidden-field contract was altered")
        if not self.research_only or self.hardware_action_authorized:
            raise ValueError("CSM-v1 is research-only and hardware-action-free")

    def to_dict(self, *, require_bound: bool = False) -> Dict[str, Any]:
        self.validate(require_bound=require_bound)
        value = asdict(self)
        value["schema"] = CSM_SCHEMA
        value["contract_revision"] = CSM_CONTRACT_REVISION
        value["whitening"] = self.whitening.to_dict()
        value["singular_support"] = self.singular_support.to_dict()
        value["finite_difference"] = self.finite_difference.to_dict()
        return value

    def sha256(self, *, require_bound: bool = False) -> str:
        return sha256_of(self.to_dict(require_bound=require_bound))


def make_csm_contract(
    N: int,
    *,
    covariance_kind: str = "per_setting_sufficient_stat_blocks",
    geometry_source: str = "BOUND_SAME_FORWARD",
    forward_sha: str = "",
    jacobian_contract_sha: str = "",
    manifold_contract_sha: str = "",
    forward_instance_sha: str = "",
    feedback_measurement_schedule_sha: str = "",
    feedback_observation_contract_sha: str = "",
    statistic_transform_sha: str = "",
    whitening: WhiteningThresholdContract | None = None,
    whitening_support_kind: str = "THRESHOLDED_COVARIANCE_EIGEN_SUPPORT",
    observation_contract_kind: str = "CSM_ORACLE_CATEGORICAL_V1",
    singular_support: SingularSupportContract | None = None,
    finite_difference: FiniteDifferenceContract | None = None,
) -> CSMContractV1:
    """Create a contract with explicit engineering defaults and hashes."""
    return CSMContractV1(
        covariance_kind=covariance_kind,
        geometry_source=geometry_source,
        N=N,
        whitening=whitening or WhiteningThresholdContract(),
        whitening_support_kind=whitening_support_kind,
        observation_contract_kind=observation_contract_kind,
        singular_support=singular_support or SingularSupportContract(),
        finite_difference=finite_difference or FiniteDifferenceContract(
            forward_sha=forward_sha, manifold_sha=manifold_contract_sha),
        forward_sha=forward_sha,
        jacobian_contract_sha=jacobian_contract_sha,
        manifold_contract_sha=manifold_contract_sha,
        forward_instance_sha=forward_instance_sha,
        feedback_measurement_schedule_sha=feedback_measurement_schedule_sha,
        feedback_observation_contract_sha=feedback_observation_contract_sha,
        statistic_transform_sha=statistic_transform_sha,
    )


__all__ = [
    "ACTUATOR_SCHEMA", "CARTESIAN_COVARIANCE_KIND", "CSM_CONTRACT_REVISION",
    "CSMContractV1", "CSM_SCHEMA",
    "GEOMETRY_SOURCES", "OBSERVATION_CONTRACT_KINDS",
    "FORBIDDEN_CSM_FIELDS", "WHITENING_SUPPORT_KINDS",
    "FiniteDifferenceContract", "PARTITION_SCHEMA", "SingularSupportContract",
    "WhiteningThresholdContract",
    "make_csm_contract",
]
