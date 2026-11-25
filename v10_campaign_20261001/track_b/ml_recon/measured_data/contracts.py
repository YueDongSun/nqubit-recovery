"""Versioned contracts for immutable measured-snapshot indexes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


MANIFEST_SCHEMA = "measured_sample_manifest"
MANIFEST_VERSION = 1
COMMAND_PROVENANCE_VALUES = (
    "exact_prepared_artifact",
    "exact_snapshot_registry",
    "verified_historical_reconstruction",
    "target_only_approximation",
    "missing",
)
COMMAND_SUPERVISION_PROVENANCE = frozenset(COMMAND_PROVENANCE_VALUES[:3])
RHO_TIERS = frozenset({"A_clean_rho", "B_count_only"})


@dataclass
class MeasuredSampleRecordV1:
    """One measured snapshot plus only facts bound to that snapshot."""

    sample_id: str
    system_type: str
    snapshot_path_relative: str
    snapshot_manifest_schema: str
    snapshot_manifest_hash: str
    exact_target_state_params: Dict[str, Any]
    exact_target_hash: str
    base_state_family_id: str
    intervention_id: Optional[str]
    intervention_delta: List[float]
    state_name: str
    state_index: Optional[int]
    plan_id: Optional[str]
    campaign_id: Optional[str]
    automation_run_id: Optional[str]
    acquisition_run_id: str
    timestamp: float
    started_at: float
    ended_at: float
    duration_s: float
    time_block_id: str
    repeat_group_id: str
    repeat_index: int
    repeat_count: int
    signal_total: int
    idler_total: int
    coin_total: int
    count_array_hashes: Dict[str, str]
    raw_count_array_hashes: Dict[str, str]
    read_count: int
    write_count: int
    gate_exposure: float
    multi_click_scope: str
    multi2_count: int
    multi3_count: int
    multi4_count: int
    multi2_pairs: Dict[str, int]
    rho_available: bool
    rho_kind: str
    rho_hash: Optional[str]
    rho_tier: str
    rho_branch_mask: int
    rho_loss_mask: int
    operation_registry_hash: Optional[str]
    matrix_snapshot_hash: Optional[str]
    calibration_snapshot_hash: Optional[str]
    waveform_hash: Optional[str]
    command_table_id: Optional[str]
    command_table_hash: Optional[str]
    command_provenance: str
    command_supervision_mask: int
    quality_status: str
    exclusion_reason: Optional[str]
    hard_fault_kind: Optional[str]
    signal_robust_z_score: float = 0.0
    idler_robust_z_score: float = 0.0
    saturation_flag: bool = False
    low_low_collapse_flag: bool = False
    signal_high_idler_low_flag: bool = False
    sample_weight: float = 1.0
    probe_anchor_id: Optional[str] = None
    probe_pair_id: Optional[str] = None
    guard_source_sample_ids: List[str] = field(default_factory=list)
    guard_directions: List[str] = field(default_factory=list)
    guard_distance: Optional[int] = None
    guard_policy_version: Optional[str] = None
    guard_time_gap_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.system_type not in {"3qubit", "4qubit"}:
            raise ValueError("system_type must be '3qubit' or '4qubit'")
        if not self.sample_id or not self.snapshot_path_relative:
            raise ValueError("sample_id and snapshot_path_relative are required")
        if self.command_provenance not in COMMAND_PROVENANCE_VALUES:
            raise ValueError(f"unsupported command_provenance: {self.command_provenance}")
        if self.rho_tier not in RHO_TIERS:
            raise ValueError(f"unsupported rho_tier: {self.rho_tier}")
        expected_command_mask = int(self.command_provenance in COMMAND_SUPERVISION_PROVENANCE)
        if int(self.command_supervision_mask) != expected_command_mask:
            raise ValueError("command_supervision_mask disagrees with command_provenance")
        expected_rho_mask = int(bool(self.rho_available))
        if self.rho_branch_mask != expected_rho_mask or self.rho_loss_mask != expected_rho_mask:
            raise ValueError("rho masks must exactly reflect rho availability")
        if self.repeat_count < 1 or not 0 <= self.repeat_index < self.repeat_count:
            raise ValueError("repeat index/count are inconsistent")
        if abs(float(self.sample_weight) - 1.0 / float(self.repeat_count)) > 1e-12:
            raise ValueError("sample_weight must equal 1 / repeat_count")
        is_guard = self.quality_status == "excluded_guard_band"
        if is_guard:
            if self.hard_fault_kind is not None:
                raise ValueError("guard-band rows cannot claim a direct hard fault")
            if not self.guard_source_sample_ids or not self.guard_directions:
                raise ValueError("guard-band rows require source IDs and directions")
            if int(self.guard_distance or 0) != 1:
                raise ValueError("V2 guard-band distance must be exactly one")
            if not str(self.guard_policy_version or "").strip():
                raise ValueError("guard-band rows require a policy version")
            if self.guard_time_gap_s is None or float(self.guard_time_gap_s) < 0.0:
                raise ValueError("guard-band rows require a non-negative time gap")
        elif any(
            value not in (None, [], ())
            for value in (
                self.guard_source_sample_ids,
                self.guard_directions,
                self.guard_distance,
                self.guard_policy_version,
                self.guard_time_gap_s,
            )
        ):
            raise ValueError("only excluded_guard_band rows may carry guard provenance")

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "MeasuredSampleRecordV1":
        return cls(**dict(payload))


@dataclass
class MeasuredSampleManifestV1:
    """Deterministic collection of measured sample records."""

    records: List[MeasuredSampleRecordV1]
    data_root: str = "data"
    manifest_hash: str = ""
    schema: str = MANIFEST_SCHEMA
    version: int = MANIFEST_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema != MANIFEST_SCHEMA or self.version != MANIFEST_VERSION:
            raise ValueError("unsupported measured manifest schema/version")
        sample_ids = [item.sample_id for item in self.records]
        paths = [item.snapshot_path_relative for item in self.records]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("measured manifest contains duplicate sample IDs")
        if len(paths) != len(set(paths)):
            raise ValueError("measured manifest contains duplicate snapshot paths")

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["records"] = [item.to_json_dict() for item in self.records]
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "MeasuredSampleManifestV1":
        values = dict(payload)
        values["records"] = [
            MeasuredSampleRecordV1.from_json_dict(item)
            for item in values.get("records", [])
        ]
        return cls(**values)
