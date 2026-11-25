"""V9 canonical data and physics contracts.

Research-only dataclasses (no Pydantic, per repo convention). Every contract carries a
``to_dict``/``from_dict`` pair and the research-authority flags required by the V9
charter so that no V9 artifact can silently claim production or hardware authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_text(text: str) -> str:
    return sha256_of_bytes(text.encode("utf-8"))


def stable_json_bytes(obj: Dict[str, Any]) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sha256_of_dict(obj: Dict[str, Any]) -> str:
    return sha256_of_bytes(stable_json_bytes(obj))


class ModelCellStatus:
    """Exact allowed statuses for the four model cells."""

    NOT_STARTED = "NOT_STARTED"
    IMPLEMENTED_NOT_TESTED = "IMPLEMENTED_NOT_TESTED"
    UNIT_TESTED = "UNIT_TESTED"
    SMOKE_PASSED = "SMOKE_PASSED"
    PILOT_DIAGNOSTIC = "PILOT_DIAGNOSTIC"
    INTERMEDIATE_PASSED = "INTERMEDIATE_PASSED"
    FULL_MULTI_SEED_PASSED = "FULL_MULTI_SEED_PASSED"
    FROZEN_OFFLINE_CANDIDATE = "FROZEN_OFFLINE_CANDIDATE"
    REAL_TEST_IN_PROGRESS = "REAL_TEST_IN_PROGRESS"
    REAL_VALIDATED = "REAL_VALIDATED"
    REAL_FAILED = "REAL_FAILED"
    CONSERVATIVE_FALLBACK = "CONSERVATIVE_FALLBACK"

    ALL = {
        NOT_STARTED,
        IMPLEMENTED_NOT_TESTED,
        UNIT_TESTED,
        SMOKE_PASSED,
        PILOT_DIAGNOSTIC,
        INTERMEDIATE_PASSED,
        FULL_MULTI_SEED_PASSED,
        FROZEN_OFFLINE_CANDIDATE,
        REAL_TEST_IN_PROGRESS,
        REAL_VALIDATED,
        REAL_FAILED,
        CONSERVATIVE_FALLBACK,
    }


@dataclass(frozen=True)
class V9ModelCell:
    """One of CLF_3Q / CLF_4Q / FB_3Q / FB_4Q."""

    cell_id: str  # "CLF_3Q" | "CLF_4Q" | "FB_3Q" | "FB_4Q"
    system_type: str  # "3qubit" | "4qubit"
    role: str  # "classifier" | "feedback"
    status: str = ModelCellStatus.NOT_STARTED

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "V9ModelCell":
        return cls(**d)


@dataclass(frozen=True)
class V9ResearchAuthorityV1:
    """Authority flags. Must stay False until the full real-validation gate passes."""

    research_only: bool = True
    shadow_only: bool = True
    promotion_ready: bool = False
    hardware_action_authorized: bool = False

    def as_shadow(self) -> "V9ResearchAuthorityV1":
        return V9ResearchAuthorityV1(
            research_only=True, shadow_only=True,
            promotion_ready=False, hardware_action_authorized=False,
        )

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)


# ---------------------------------------------------------------------------
# rho8 view contract (section 8-10)
# ---------------------------------------------------------------------------

RHO8_ESTIMATOR_KEYS = ("cs_minus_ir", "cs")
RHO8_MAP_KEYS = ("retrieval_raw", "retrieval_sqrt", "idler_raw", "idler_sqrt")

RHO8_CANONICAL_KEYS = tuple(
    "rho_{}_".format(est) + m for est in RHO8_ESTIMATOR_KEYS for m in RHO8_MAP_KEYS
)


@dataclass(frozen=True)
class V9RhoViewProvenanceV1:
    """Provenance of one normalized compensation map.

    ``raw_file_sha256`` hashes the raw source file on disk; ``raw_matrix_sha256``
    hashes the matrix as parsed from it (before normalization); ``normalized_matrix_sha256``
    hashes the canonical normalized map used for reconstruction.
    """

    map_key: str
    system_type: str
    source_path: str
    transform: str  # "raw" | "sqrt"
    raw_file_sha256: str
    raw_matrix_sha256: str
    normalized_matrix_sha256: str
    shape: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "V9RhoViewProvenanceV1":
        d = dict(d)
        d["shape"] = tuple(d["shape"])
        return cls(**d)


@dataclass(frozen=True)
class V9RhoEightViewContractV1:
    """The authoritative definition of the eight rho views.

    Views are deterministic/near-deterministic derivations of one batch of photon
    counts. They are NOT independent observations and NOT ground truth; they serve
    only as weak teachers and for fixed-reconstruction physical evaluation.
    """

    version: str = "V9RhoEightViewContractV1"
    phase_sign: int = -1
    basis_phase_convention: str = "read_aod_negative_order"
    estimators: Tuple[str, ...] = RHO8_ESTIMATOR_KEYS
    maps: Tuple[str, ...] = RHO8_MAP_KEYS
    canonical_keys: Tuple[str, ...] = RHO8_CANONICAL_KEYS
    derived_rho8_independent_evidence: bool = False
    weak_teacher_max_fraction: float = 0.05
    weak_teacher_ablation_fractions: Tuple[float, ...] = (0.01, 0.03, 0.05)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("estimators", "maps", "canonical_keys", "weak_teacher_ablation_fractions"):
            d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "V9RhoEightViewContractV1":
        d = dict(d)
        for k in ("estimators", "maps", "canonical_keys", "weak_teacher_ablation_fractions"):
            d[k] = tuple(d[k])
        return cls(**d)

    def sha256(self) -> str:
        return sha256_of_dict(self.to_dict())


@dataclass(frozen=True)
class V9RhoEightViewBundleV1:
    """Per-sample bundle of the eight views plus provenance and diagnostics."""

    sample_id: str
    system_type: str
    views: Dict[str, Any]  # canonical key -> {"rho_full":..., "rho_W_core":..., ...}
    view_provenance: Dict[str, Any]  # map_key -> V9RhoViewProvenanceV1 dict
    estimator_diagnostics: Dict[str, Any] = field(default_factory=dict)
    reconstruction_diagnostics: Dict[str, Any] = field(default_factory=dict)
    availability_mask: Dict[str, bool] = field(default_factory=dict)
    count_source_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "V9RhoEightViewBundleV1":
        return cls(**d)


# ---------------------------------------------------------------------------
# Count contract (section 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V9FourCellCountsV1:
    """Mutually exclusive four cells for one operation: (n00, n10, n01, n11)."""

    n00: float
    n10: float
    n01: float
    n11: float

    @classmethod
    def from_counts(cls, read_count: float, signal_count: float,
                    idler_count: float, coin_count: float) -> "V9FourCellCountsV1":
        return cls(
            n00=read_count - signal_count - idler_count + coin_count,
            n10=signal_count - coin_count,
            n01=idler_count - coin_count,
            n11=coin_count,
        )

    def is_valid(self, tol: float = 1e-9) -> bool:
        return all(v >= -tol for v in (self.n00, self.n10, self.n01, self.n11))

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class V9CountContractV1:
    """Hard invariants and the primary likelihood rule for one operation."""

    version: str = "V9CountContractV1"
    cells: Tuple[str, ...] = ("n00", "n10", "n01", "n11")
    primary_likelihood: str = "BetaBinomial(S|R) x BetaBinomial(C|S) x BetaBinomial(I-C|R-S)"
    used_exactly_once: bool = True
    standalone_signal_weight: float = 0.0
    standalone_idler_weight: float = 0.0
    standalone_coin_weight: float = 0.0
    four_cell_likelihood_weight: float = 0.0
    raw_coin_nb2_role: str = "diagnostic_non_regression"

    def sha256(self) -> str:
        return sha256_of_dict(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["cells"] = list(self.cells)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "V9CountContractV1":
        d = dict(d)
        d["cells"] = tuple(d["cells"])
        return cls(**d)


@dataclass(frozen=True)
class V9GlobalMultiV1:
    """Native-scope global multi-click observation (section 7)."""

    multi2: float
    pair12: float = 0.0
    pair13: float = 0.0
    pair14: float = 0.0
    pair23: float = 0.0
    pair24: float = 0.0
    pair34: float = 0.0
    multi3: float = 0.0
    multi4: float = 0.0

    def pairs_sum(self) -> float:
        return self.pair12 + self.pair13 + self.pair14 + self.pair23 + self.pair24 + self.pair34

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Operation-point action contract (section 20-23)
# ---------------------------------------------------------------------------

ACTION_STAGE_PHASE_ONLY = "PHASE_ONLY"
ACTION_STAGE_AMPLITUDE_ENABLED = "AMPLITUDE_ENABLED"
ACTION_STAGE_COMBINED_ENABLED = "COMBINED_ENABLED"
ACTION_STAGES = (ACTION_STAGE_PHASE_ONLY, ACTION_STAGE_AMPLITUDE_ENABLED, ACTION_STAGE_COMBINED_ENABLED)


@dataclass(frozen=True)
class V9OperationPointActionProposalV1:
    """A per-operation-point W_CH1 action proposal produced by a V9 model.

    The model package itself has no upload authority; the proposal is only materialized
    after explicit operator approval through system-control.
    """

    proposal_id: str
    system_type: str
    target_hash: str
    control_hash: str
    operation_registry_hash: str
    command_table_hash: str
    model_checkpoint_hash: str
    delta_log_amp_w_ch1: List[List[float]]  # [operation, cell]
    delta_phase_w_ch1: List[List[float]]  # [operation, cell]
    global_component: Dict[str, Any]
    shared_spatial_component: Dict[str, Any]
    cycle_component: Dict[str, Any]
    direct_residual_component: Dict[str, Any]
    per_operation_uncertainty: List[float]
    predicted_gain_quantiles: Dict[str, List[float]]
    harmful_probability: float
    clipping_estimate: Dict[str, float]
    action_stage: str
    no_op_eligible: bool
    research_authority: Dict[str, bool] = field(default_factory=lambda: asdict(V9ResearchAuthorityV1()))

    def sha256(self) -> str:
        return sha256_of_dict(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "V9OperationPointActionProposalV1":
        return cls(**d)


@dataclass(frozen=True)
class V9OperationPointActionContractV1:
    """Contract binding a proposal to registry/command/checkpoint identity."""

    version: str = "V9OperationPointActionContractV1"
    w_ch1_only: bool = True
    forbidden_axes: Tuple[str, ...] = (
        "R_CH1", "W_CH2", "W_CH3", "W_CH4", "R_CH2", "R_CH3", "R_CH4",
        "brightness", "timing", "digital_delay", "local_read_projector",
        "detector_configuration",
    )
    max_single_amp_fraction: float = 0.005
    max_single_phase_rad: float = 0.02
    max_cumulative_amp_fraction: float = 0.03
    max_cumulative_phase_rad: float = 0.10
    emergency_hard_phase_rad: float = 0.20
    direct_residual_max_fraction: float = 0.20
    mean_over_cells_zero: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["forbidden_axes"] = list(self.forbidden_axes)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "V9OperationPointActionContractV1":
        d = dict(d)
        d["forbidden_axes"] = tuple(d["forbidden_axes"])
        return cls(**d)

    def sha256(self) -> str:
        return sha256_of_dict(self.to_dict())


@dataclass(frozen=True)
class V9PreparedOperationActionArtifactV1:
    """Hash-pinned materialized artifact (produced by system-control, not by the model)."""

    artifact_id: str
    proposal_id: str
    staging_workspace: str
    matrix_hashes: Dict[str, str]
    guard_results: Dict[str, Any]
    requested_vs_actual_discrepancy: Dict[str, float]
    materialized_at_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "V9PreparedOperationActionArtifactV1":
        return cls(**d)


# The eight canonical rho keys as a module-level tuple for import convenience.
RHO8_KEYS = list(RHO8_CANONICAL_KEYS)
