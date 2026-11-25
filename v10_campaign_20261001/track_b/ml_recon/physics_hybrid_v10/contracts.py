"""Canonical V10 measurement, data, model, and artifact contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

V10_MODEL_IDS = (
    "C_N_FULL_V10",
    "F_N_RESEARCH_V10",
    "F_3Q_HW_V10",
    "F_4Q_HW_V10",
)
TRACKB_TRAINING_BRANCH = "codex/v10_trackb_v24_realfd_rf_bridge"
V10_AUTHORIZED_BRANCHES = (
    "v10",
    "codex/v10_lowdisk_training_prep",
    "codex/v10_real_data_recovery_gate",
    TRACKB_TRAINING_BRANCH,
)
V10_N_GRID = (3, 4, 6, 8, 12, 16, 24, 32, 64, 128)
V10_STRESS_N = 256


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_of(value: Any) -> str:
    data = value if isinstance(value, bytes) else stable_json_bytes(value)
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class V10ResearchAuthority:
    research_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    full_4q_campaign_authorized: bool = False
    hardware_action_authorized: bool = False

    def validate(self) -> None:
        if self.research_only is not True:
            raise ValueError("V10 authority must remain research-only")
        if any(type(value) is not bool for value in (
            self.selection_evidence,
            self.promotion_ready,
            self.full_4q_campaign_authorized,
            self.hardware_action_authorized,
        )):
            raise TypeError("V10 authority flags must be strict Booleans")
        if self.hardware_action_authorized:
            raise ValueError("V10 model packages never have hardware-write authority")
        if self.selection_evidence or self.promotion_ready or self.full_4q_campaign_authorized:
            raise ValueError("offline and real gates have not been supplied")


@dataclass(frozen=True)
class V10ShotRecord:
    """One preparation -> one fixed setting -> one outcome."""

    measurement_setting_id: str
    shots_attempted: int
    shots_heralded: int
    effective_valid_shots: int
    exposure: float
    outcome: Optional[str] = None

    def validate(self) -> None:
        if not self.measurement_setting_id:
            raise ValueError("measurement_setting_id is required")
        if not (0 <= self.effective_valid_shots <= self.shots_heralded <= self.shots_attempted):
            raise ValueError("shot counts must satisfy valid <= heralded <= attempted")
        if self.exposure < 0.0:
            raise ValueError("exposure must be non-negative")


@dataclass(frozen=True)
class V10MeasurementSnapshot:
    """A window spanning many settings; explicitly not a shot."""

    snapshot_id: str
    N: int
    records: Tuple[V10ShotRecord, ...]
    measurement_contract_sha: str

    @property
    def shots_attempted(self) -> int:
        return sum(r.shots_attempted for r in self.records)

    @property
    def effective_valid_shots(self) -> int:
        return sum(r.effective_valid_shots for r in self.records)


class CertificateStatus(str, Enum):
    CERTIFIED_ENTANGLED = "CERTIFIED_ENTANGLED"
    CERTIFIED_GME = "CERTIFIED_GME"
    NPT_CERTIFIED = "NPT_CERTIFIED"
    PPT_MIXTURE_COMPATIBLE = "PPT_MIXTURE_COMPATIBLE"
    PHYSICALLY_COMPATIBLE = "PHYSICALLY_COMPATIBLE"
    SEPARABLE_COMPATIBLE = "SEPARABLE_COMPATIBLE"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNAVAILABLE = "UNAVAILABLE"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"


# Backwards-compatible import name.  The value set is deliberately expanded: a
# PPT or moment-PSD result is not a separability certificate.
FourStatus = CertificateStatus


@dataclass(frozen=True)
class V10MeasurementContract:
    N: int
    contract_kind: str
    setting_ids: Tuple[str, ...]
    descriptor_schema: str = "V10PhysicalBasisDescriptorV1"
    full_family_identifiable: bool = False
    stress_only: bool = False

    def validate(self) -> None:
        if not 3 <= self.N <= 256:
            raise ValueError("V10 supports 3 <= N <= 256")
        if len(self.setting_ids) != len(set(self.setting_ids)):
            raise ValueError("measurement setting ids must be unique")
        if self.N > 4 and self.full_family_identifiable:
            raise ValueError("general-N contracts cannot claim exhaustive family identifiability")

    def sha256(self) -> str:
        return sha256_of(asdict(self))


@dataclass(frozen=True)
class V10MeasurementSchedule:
    """Immutable registry-bound physical measurement schedule.

    Every Pauli setting is global: ``pauli_axes[i]`` is the basis applied to
    physical qubit ``i``.  Subsets are derived views of these settings, never
    separately prepared pseudo-measurements.
    """

    N: int
    setting_ids: Tuple[str, ...]
    pauli_axes: Tuple[Tuple[str, ...], ...]
    setting_registry_sha: str
    schedule_kind: str
    covered_subsets: Tuple[Tuple[int, ...], ...] = ()
    coverage_sha: str = ""

    def validate(self) -> None:
        if not 3 <= self.N <= 256:
            raise ValueError("V10 schedule N must be in [3, 256]")
        if not self.setting_ids or len(self.setting_ids) != len(set(self.setting_ids)):
            raise ValueError("schedule setting ids must be non-empty and unique")
        if len(self.pauli_axes) != len(self.setting_ids):
            raise ValueError("one global Pauli assignment is required per setting")
        if any(len(axes) != self.N or any(axis not in ("X", "Y", "Z") for axis in axes)
               for axes in self.pauli_axes):
            raise ValueError("every physical Pauli setting must assign X/Y/Z to all N qubits")
        if len(self.setting_registry_sha) != 64:
            raise ValueError("setting_registry_sha must be a SHA-256 digest")

    def sha256(self) -> str:
        self.validate()
        return sha256_of(asdict(self))


@dataclass(frozen=True)
class V10MeasurementObservation:
    """Observed outcomes for one registered global measurement setting."""

    sample_id: str
    setting_id: str
    setting_registry_sha: str
    shot_mode: str
    observation_model: str
    outcomes: np.ndarray
    probabilities: Optional[np.ndarray]
    exposure: float
    requested_shots: int
    realized_shots: int
    valid_shot_mask: np.ndarray

    def validate(self, N: int) -> None:
        import numpy as np

        outcomes = np.asarray(self.outcomes)
        mask = np.asarray(self.valid_shot_mask)
        if outcomes.ndim != 2 or outcomes.shape[1] != N:
            raise ValueError("outcomes must have shape [shots, N]")
        if outcomes.shape[0] != self.realized_shots or mask.shape != (self.realized_shots,):
            raise ValueError("realized_shots/outcomes/valid_shot_mask disagree")
        if outcomes.size and not np.all(np.isin(outcomes, (-1, 1))):
            raise ValueError("Pauli outcomes must be +/-1")
        if mask.dtype != np.bool_:
            raise ValueError("valid_shot_mask must be boolean")
        if self.shot_mode not in ("ideal", "fixed_total", "fixed_per_setting"):
            raise ValueError("unknown V10 shot mode")
        if self.observation_model not in ("ideal", "multinomial", "poisson"):
            raise ValueError("unknown V10 observation model")
        if self.shot_mode == "ideal":
            if self.realized_shots != 0 or self.probabilities is None:
                raise ValueError("ideal observations carry probabilities and no sampled shots")
        if self.probabilities is not None:
            probabilities = np.asarray(self.probabilities, dtype=np.float64)
            if probabilities.ndim != 1 or not np.isclose(probabilities.sum(), 1.0, atol=1e-9):
                raise ValueError("probabilities must be a normalized vector")
        if self.exposure < 0.0 or self.requested_shots < 0 or self.realized_shots < 0:
            raise ValueError("shot counts and exposure must be non-negative")


@dataclass(frozen=True)
class V10SyntheticTruth:
    """Privileged simulation truth; consumers must honor every availability mask."""

    sample_id: str
    N: int
    state_backend: str
    state_sha: str
    family: str
    partition: Tuple[Tuple[int, ...], ...]
    partition_exact: bool
    mechanism_labels: Tuple[str, ...]
    clean_state_available: bool
    action_truth_available: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.sample_id or not 3 <= self.N <= 256:
            raise ValueError("invalid synthetic truth identity")
        flat = [q for block in self.partition for q in block]
        if len(flat) != len(set(flat)) or any(q < 0 or q >= self.N for q in flat):
            raise ValueError("partition contains duplicate or invalid qubits")
        if self.partition_exact and sorted(flat) != list(range(self.N)):
            raise ValueError("exact partitions must cover every qubit exactly once")


@dataclass(frozen=True)
class V10MeasurementBatch:
    """Typed batch joining schedules, observations and optional simulation truth."""

    schedule: V10MeasurementSchedule
    observations: Tuple[V10MeasurementObservation, ...]
    truths: Tuple[Optional[V10SyntheticTruth], ...]
    observation_valid_mask: np.ndarray
    truth_valid_mask: np.ndarray
    setting_registry_sha: str
    measurement_contract_sha: str

    def validate(self) -> None:
        import numpy as np

        self.schedule.validate()
        if self.setting_registry_sha != self.schedule.setting_registry_sha:
            raise ValueError("batch/schedule registry hash mismatch")
        if len(self.observations) != len(self.truths):
            raise ValueError("batch observations/truth slots must align")
        observation_mask = np.asarray(self.observation_valid_mask)
        truth_mask = np.asarray(self.truth_valid_mask)
        expected = (len(self.observations),)
        if observation_mask.shape != expected or truth_mask.shape != expected:
            raise ValueError("batch masks must have shape [batch]")
        if observation_mask.dtype != np.bool_ or truth_mask.dtype != np.bool_:
            raise ValueError("batch masks must be boolean")
        setting_ids = set(self.schedule.setting_ids)
        for index, observation in enumerate(self.observations):
            observation.validate(self.schedule.N)
            if observation.setting_id not in setting_ids:
                raise ValueError("observation references an unregistered setting")
            if observation.setting_registry_sha != self.setting_registry_sha:
                raise ValueError("observation registry hash mismatch")
            truth = self.truths[index]
            if bool(truth is not None) != bool(truth_mask[index]):
                raise ValueError("truth slot and truth_valid_mask disagree")
            if truth is not None:
                truth.validate()
                if truth.sample_id != observation.sample_id or truth.N != self.schedule.N:
                    raise ValueError("truth/observation identity mismatch")


@dataclass(frozen=True)
class V10FourCellCounts:
    read: float
    signal: float
    idler: float
    coin: float

    def cells(self) -> Tuple[float, float, float, float]:
        return (
            self.read - self.signal - self.idler + self.coin,
            self.signal - self.coin,
            self.idler - self.coin,
            self.coin,
        )

    def validate(self, tol: float = 1e-9) -> None:
        if self.coin > self.signal + tol or self.coin > self.idler + tol:
            raise ValueError("coin must not exceed signal or idler")
        if self.signal + self.idler - self.coin > self.read + tol:
            raise ValueError("four-cell inclusion-exclusion exceeds read")
        if any(v < -tol for v in self.cells()):
            raise ValueError("four-cell counts must be non-negative")


@dataclass(frozen=True)
class V10GlobalMultiObservation:
    multi2: int
    pair_counts: Tuple[int, int, int, int, int, int]
    multi3: int
    multi4: int
    scope: str = "global"

    def validate(self) -> None:
        if self.scope != "global":
            raise ValueError("V10 global multi observations cannot be operation-local")
        if sum(self.pair_counts) != self.multi2:
            raise ValueError("pair multinomial cells must sum to multi2")
        if min((self.multi2, self.multi3, self.multi4, *self.pair_counts)) < 0:
            raise ValueError("multi counts must be non-negative")


@dataclass(frozen=True)
class V10ArtifactBinding:
    model_id: str
    branch: str
    source_commit: str
    source_tree_sha: str
    dirty_patch_sha: str
    v23_manifest_sha: str
    synthetic_manifest_sha: str
    measurement_contract_sha: str
    shot_budget_contract_sha: str
    rho8_sha: str
    config_sha: str
    seed_panel: Tuple[int, ...]
    exact_parameter_count: int
    peak_vram_bytes: Optional[int]
    calibration: Dict[str, Any]
    negative_controls: Dict[str, Any]
    checkpoint_sha: str
    report_sha: str
    authority: V10ResearchAuthority = field(default_factory=V10ResearchAuthority)

    def validate(self) -> None:
        if self.model_id not in V10_MODEL_IDS:
            raise ValueError(f"unknown V10 model id: {self.model_id}")
        if self.branch not in V10_AUTHORIZED_BRANCHES:
            raise ValueError("formal V10 artifacts must bind an authorized V10 branch")
        self.authority.validate()

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["artifact_binding_sha"] = sha256_of(result)
        return result


def validate_v23_population(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Validate immutable V2.3 role counts without changing row selection."""
    trusted = [r for r in rows if bool(r.get("trusted", False))]
    counts = {
        "trusted_total": len(trusted),
        "trusted_3q": sum(r.get("system_type") == "3qubit" for r in trusted),
        "trusted_4q": sum(r.get("system_type") == "4qubit" for r in trusted),
    }
    expected = {"trusted_total": 3032, "trusted_3q": 1076, "trusted_4q": 1956}
    if counts != expected:
        raise ValueError(f"V2.3 trusted population mismatch: {counts} != {expected}")
    return counts


def feedback_measurement_contract(
    N: int,
    *,
    shot_mode: str = "fixed_per_setting",
    budget: int = 256,
) -> Tuple[str, str]:
    """Canonical feedback-specific measurement + shot-budget contract SHAs.

    Feedback uses ``registered_feedback_settings(N)`` which is
    {Z⊗N, X₀Xᵢ, X₀Yᵢ}_{i=1}^{N-1} — the phase-sensitive W-state
    schedule.  This function binds that exact schedule (not the generic
    ``V10MeasurementScheduler``) together with the actual shot allocation
    into deterministic SHA-256 digests.

    Returns ``(measurement_contract_sha, shot_budget_contract_sha)``.
    """
    from .noise_interventions import registered_feedback_settings
    from .shot_budget import allocate_measurement_shots

    if not 3 <= N <= 256:
        raise ValueError("feedback measurement contract requires 3 <= N <= 256")
    feedback_settings = registered_feedback_settings(N)
    setting_ids = tuple(f"_fb_{''.join(axes)}" for axes in feedback_settings)
    registry_sha = sha256_of({
        "kind": "feedback_registered_settings",
        "N": N,
        "settings": [list(axes) for axes in feedback_settings],
    })
    schedule = V10MeasurementSchedule(
        N=N,
        setting_ids=setting_ids,
        pauli_axes=feedback_settings,
        setting_registry_sha=registry_sha,
        schedule_kind="phase_sensitive_w_v1",
    )
    schedule.validate()
    measurement_sha = schedule.sha256()
    shot = allocate_measurement_shots(setting_ids, shot_mode, budget)
    return measurement_sha, shot.sha256()
