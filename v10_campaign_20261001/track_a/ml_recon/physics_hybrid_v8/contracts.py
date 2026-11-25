"""Strict, hash-bound contracts for the Physics Hybrid V8 delivery track.

The contracts in this module describe evidence, not aspirations.  A failed
measured-data audit remains failed even when an explicit waiver allows a run
to proceed, and an ordinary measured snapshot never becomes an action label.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


V8_DATA_CONTRACT_SCHEMA = "physics_hybrid_v8_data_contract_v1"
V8_MEASURED_AUDIT_SCHEMA = "physics_hybrid_v8_measured_audit_v1"
V8_AUDIT_WAIVER_SCHEMA = "physics_hybrid_v8_audit_waiver_v1"
V8_LOCKED_FINAL_TEST_SCHEMA = "physics_hybrid_v8_locked_final_test_v1"
V8_LOCKED_ENSEMBLE_RELEASE_ACCESS_SCHEMA = (
    "v8_locked_ensemble_release_access_declaration_v1"
)
V8_LOCKED_ENSEMBLE_ACCESS_LEDGER_SCHEMA = (
    "v8_locked_ensemble_final_access_ledger_v1"
)
V8_HISTORY_TRANSITION_SCHEMA = "physics_hybrid_v8_history_transition_provenance_v1"
V8_SCHEMA_VERSION = 1

SUPPORTED_SYSTEM_TYPES = frozenset({"3qubit", "4qubit"})
REAL_ACTION_LABEL_POLICY = "paired_transition_or_hash_bound_signed_probe_only"
GLOBAL_MULTI_CLICK_POLICY = "global_interval_upper_bound_only"
MISSING_RHO_POLICY = "none_plus_explicit_mask"
UNKNOWN_EXPOSURE_POLICY = "none_plus_explicit_mask"
HISTORY_ACTION_LABEL_SOURCE = "executed_command_and_upload_receipt"
PAIRED_TRANSITION_ACTION_EVIDENCE = "paired_executed_transition"
SIGNED_PROBE_ACTION_EVIDENCE = "hash_bound_signed_probe"
NO_ACTION_EVIDENCE_KINDS = frozenset(
    {"none_target_only_snapshot", "none_unpaired_snapshot"}
)


class MeasuredDataAuditError(ValueError):
    """Raised when measured evidence is not accepted under the strict audit."""


def stable_json_sha256(payload: Any) -> str:
    """Return a canonical lowercase SHA-256 for a JSON-compatible payload."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    """Bind an unordered final-test membership declaration deterministically."""

    normalized = sorted(str(sample_id) for sample_id in sample_ids)
    if not normalized or any(not sample_id for sample_id in normalized):
        raise ValueError("final-test sample IDs must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError("final-test sample IDs must be unique")
    return stable_json_sha256(normalized)


def canonical_ledger_path_sha256(path_value: str | Path) -> str:
    """Hash one absolute, normalized ledger location.

    The locked-final access budget belongs to the frozen ensemble, not to an
    evaluator output directory.  Requiring one absolute location prevents a
    caller from obtaining a fresh budget merely by changing ``--output-dir``.
    """

    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError("canonical_ledger_path must be absolute")
    canonical_path = str(path.resolve())
    if str(path_value) != canonical_path:
        raise ValueError("canonical_ledger_path must already be normalized")
    return stable_json_sha256({"canonical_ledger_path": canonical_path})


def _require_sha256(value: Any, field_name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return text


def _parse_aware_datetime(value: str, field_name: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _normalized_now(now: Optional[datetime]) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must include a timezone")
    return current.astimezone(timezone.utc)


def _validate_hash_mapping(
    values: Mapping[str, str],
    field_name: str,
    *,
    require_both_systems: bool = True,
) -> None:
    normalized = dict(values)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if require_both_systems and set(normalized) != SUPPORTED_SYSTEM_TYPES:
        raise ValueError(
            f"{field_name} must bind exactly {sorted(SUPPORTED_SYSTEM_TYPES)}"
        )
    for key, value in normalized.items():
        if not str(key).strip():
            raise ValueError(f"{field_name} contains an empty key")
        _require_sha256(value, f"{field_name}[{key!r}]")


@dataclass(frozen=True)
class MeasuredAuditWaiverV1:
    """Explicit, expiring acknowledgement of one exact failed summary.

    The free-form evidence field can record why an expectation drifted and why
    excluded hard-fault rows remain excluded.  A waiver never changes the
    underlying audit result and cannot waive split leakage or hash mismatch.
    """

    reason: str
    expires_at: str
    summary_sha256: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    schema: str = V8_AUDIT_WAIVER_SCHEMA
    schema_version: int = V8_SCHEMA_VERSION

    def validate(
        self,
        *,
        reference_summary_sha256: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> None:
        if self.schema != V8_AUDIT_WAIVER_SCHEMA or self.schema_version != V8_SCHEMA_VERSION:
            raise ValueError("unsupported V8 audit-waiver schema/version")
        if len(self.reason.strip()) < 12:
            raise ValueError("audit waiver reason must be explicit")
        _require_sha256(self.summary_sha256, "summary_sha256")
        if reference_summary_sha256 is not None:
            _require_sha256(reference_summary_sha256, "reference_summary_sha256")
            if self.summary_sha256 != reference_summary_sha256:
                raise MeasuredDataAuditError(
                    "audit waiver is not bound to the current measured summary"
                )
        expires_at = _parse_aware_datetime(self.expires_at, "expires_at")
        if expires_at <= _normalized_now(now):
            raise MeasuredDataAuditError("audit waiver has expired")

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "MeasuredAuditWaiverV1":
        return cls(**dict(payload))

    def sha256(self) -> str:
        self.validate(now=datetime(1970, 1, 1, tzinfo=timezone.utc))
        return stable_json_sha256(self.to_json_dict())


@dataclass(frozen=True)
class LockedFinalTestDeclarationV1:
    """Immutable declaration that keeps final-test rows out of selection."""

    split_hash: str
    final_test_split_names: Tuple[str, ...]
    final_test_sample_ids_sha256: str
    final_test_sample_count: int
    locked_at: str
    training_access_allowed: bool = False
    model_selection_access_allowed: bool = False
    release_evaluation_budget: int = 1
    schema: str = V8_LOCKED_FINAL_TEST_SCHEMA
    schema_version: int = V8_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema != V8_LOCKED_FINAL_TEST_SCHEMA or self.schema_version != V8_SCHEMA_VERSION:
            raise ValueError("unsupported V8 locked-final-test schema/version")
        _require_sha256(self.split_hash, "split_hash")
        _require_sha256(self.final_test_sample_ids_sha256, "final_test_sample_ids_sha256")
        names = tuple(str(name).strip() for name in self.final_test_split_names)
        if not names or any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("final_test_split_names must be non-empty and unique")
        if int(self.final_test_sample_count) <= 0:
            raise ValueError("final_test_sample_count must be positive")
        _parse_aware_datetime(self.locked_at, "locked_at")
        if self.training_access_allowed or self.model_selection_access_allowed:
            raise ValueError("locked final-test rows cannot be used for training or selection")
        if int(self.release_evaluation_budget) != 1:
            raise ValueError("locked final test permits exactly one release evaluation")

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["final_test_split_names"] = list(self.final_test_split_names)
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "LockedFinalTestDeclarationV1":
        values = dict(payload)
        values["final_test_split_names"] = tuple(values.get("final_test_split_names") or ())
        return cls(**values)

    def sha256(self) -> str:
        self.validate()
        return stable_json_sha256(self.to_json_dict())


@dataclass(frozen=True)
class LockedEnsembleReleaseAccessDeclarationV1:
    """Operator approval for one exact frozen ensemble and one ledger.

    This is deliberately ensemble-scoped.  A member checkpoint hash is not a
    substitute for the frozen candidate identity, and the canonical ledger
    path cannot be selected by the evaluator output directory.
    """

    parent_manifest_hash: str
    ensemble_sha256: str
    config_sha256: str
    locked_test_declaration_sha256: str
    canonical_ledger_path: str
    canonical_ledger_path_sha256: str
    operator: str
    declared_at: str
    declaration_sha256: str
    approved_single_use: bool = True
    allowed_for_model_selection: bool = False
    schema: str = V8_LOCKED_ENSEMBLE_RELEASE_ACCESS_SCHEMA
    schema_version: int = V8_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema != V8_LOCKED_ENSEMBLE_RELEASE_ACCESS_SCHEMA
            or int(self.schema_version) != V8_SCHEMA_VERSION
        ):
            raise ValueError("unsupported V8 locked ensemble release schema/version")
        for field_name in (
            "parent_manifest_hash",
            "ensemble_sha256",
            "config_sha256",
            "locked_test_declaration_sha256",
            "canonical_ledger_path_sha256",
            "declaration_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        expected_path_sha256 = canonical_ledger_path_sha256(
            self.canonical_ledger_path
        )
        if self.canonical_ledger_path_sha256 != expected_path_sha256:
            raise ValueError("canonical locked-access ledger path hash mismatch")
        if not self.operator.strip():
            raise ValueError("locked release declaration requires an operator identity")
        _parse_aware_datetime(self.declared_at, "declared_at")
        if self.approved_single_use is not True:
            raise ValueError("locked final evaluation requires explicit single-use approval")
        if self.allowed_for_model_selection is not False:
            raise ValueError("locked final metrics cannot be used for model selection")
        payload = self.to_json_dict()
        payload.pop("declaration_sha256")
        if self.declaration_sha256 != stable_json_sha256(payload):
            raise ValueError("locked release declaration hash mismatch")

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "LockedEnsembleReleaseAccessDeclarationV1":
        return cls(**dict(payload))

    def sha256(self) -> str:
        self.validate()
        return self.declaration_sha256


@dataclass(frozen=True)
class HistoryTransitionProvenanceV1:
    """Proof that a real before/action/after history transition was executed.

    A target state or current matrix-directory reconstruction is not sufficient
    provenance.  All command, action, upload, calibration, tomography, and
    proposing-model identities must be immutable hashes.
    """

    transition_id: str
    episode_id: str
    transition_index: int
    system_type: str
    before_sample_id: str
    after_sample_id: str
    before_snapshot_hash: str
    after_snapshot_hash: str
    actual_command_hash: str
    actual_action_hash: str
    upload_receipt_hash: str
    calibration_hash: str
    tomography_hash: str
    model_hash: str
    executed_at: str
    exposure_s: Optional[float]
    exposure_known_mask: int
    action_label_source: str = HISTORY_ACTION_LABEL_SOURCE
    action_evidence_kind: str = PAIRED_TRANSITION_ACTION_EVIDENCE
    real_action_supervision_mask: int = 1
    execution_verified: bool = True
    schema: str = V8_HISTORY_TRANSITION_SCHEMA
    schema_version: int = V8_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema != V8_HISTORY_TRANSITION_SCHEMA or self.schema_version != V8_SCHEMA_VERSION:
            raise ValueError("unsupported V8 history-transition schema/version")
        for field_name in (
            "transition_id",
            "episode_id",
            "before_sample_id",
            "after_sample_id",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} is required")
        if self.system_type not in SUPPORTED_SYSTEM_TYPES:
            raise ValueError(f"unsupported system_type: {self.system_type}")
        if int(self.transition_index) < 0:
            raise ValueError("transition_index must be non-negative")
        if self.before_sample_id == self.after_sample_id:
            raise ValueError("before and after samples must be distinct")
        for field_name in (
            "before_snapshot_hash",
            "after_snapshot_hash",
            "actual_command_hash",
            "actual_action_hash",
            "upload_receipt_hash",
            "calibration_hash",
            "tomography_hash",
            "model_hash",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if self.before_snapshot_hash == self.after_snapshot_hash:
            raise ValueError("before and after snapshot hashes must be distinct")
        _parse_aware_datetime(self.executed_at, "executed_at")
        if self.action_label_source != HISTORY_ACTION_LABEL_SOURCE:
            raise ValueError("real action labels require executed command/upload provenance")
        if self.action_evidence_kind != PAIRED_TRANSITION_ACTION_EVIDENCE:
            raise ValueError("history action evidence must be a paired executed transition")
        if int(self.real_action_supervision_mask) != 1:
            raise ValueError("verified history transition supervision mask must equal 1")
        if not self.execution_verified:
            raise ValueError("unverified execution cannot supervise a real action transition")
        mask = int(self.exposure_known_mask)
        if mask not in {0, 1}:
            raise ValueError("exposure_known_mask must be 0 or 1")
        if mask == 1 and (self.exposure_s is None or float(self.exposure_s) <= 0.0):
            raise ValueError("known exposure requires a positive exposure_s")
        if mask == 0 and self.exposure_s is not None:
            raise ValueError("unknown exposure must remain None")

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "HistoryTransitionProvenanceV1":
        return cls(**dict(payload))

    def sha256(self) -> str:
        self.validate()
        return stable_json_sha256(self.to_json_dict())


@dataclass(frozen=True)
class MeasuredDataAuditResultV1:
    """Hashable result of the whole measured-manifest audit."""

    accepted: bool
    audit_profile_passed: bool
    waiver_applied: bool
    failed_checks: Tuple[str, ...]
    summary_sha256: str
    manifest_hash: str
    split_hash: str
    trusted_sample_count_by_system: Dict[str, int]
    rho_available_count_by_system: Dict[str, int]
    exposure_known_count_by_system: Dict[str, int]
    exposure_unknown_count_by_system: Dict[str, int]
    real_action_supervision_count_by_system: Dict[str, int]
    excluded_rho_by_fault_by_system: Dict[str, Dict[str, int]]
    waiver_sha256: Optional[str] = None
    schema: str = V8_MEASURED_AUDIT_SCHEMA
    schema_version: int = V8_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema != V8_MEASURED_AUDIT_SCHEMA or self.schema_version != V8_SCHEMA_VERSION:
            raise ValueError("unsupported V8 measured-audit schema/version")
        for field_name in ("summary_sha256", "manifest_hash", "split_hash"):
            _require_sha256(getattr(self, field_name), field_name)
        if set(self.trusted_sample_count_by_system) != SUPPORTED_SYSTEM_TYPES:
            raise ValueError("trusted counts must cover both systems")
        for field_name in (
            "rho_available_count_by_system",
            "exposure_known_count_by_system",
            "exposure_unknown_count_by_system",
            "real_action_supervision_count_by_system",
            "excluded_rho_by_fault_by_system",
        ):
            if set(getattr(self, field_name)) != SUPPORTED_SYSTEM_TYPES:
                raise ValueError(f"{field_name} must cover both systems")
        for system_type in SUPPORTED_SYSTEM_TYPES:
            trusted = int(self.trusted_sample_count_by_system[system_type])
            known = int(self.exposure_known_count_by_system[system_type])
            unknown = int(self.exposure_unknown_count_by_system[system_type])
            rho_count = int(self.rho_available_count_by_system[system_type])
            action_count = int(self.real_action_supervision_count_by_system[system_type])
            if min(trusted, known, unknown, rho_count, action_count) < 0:
                raise ValueError("measured audit counts must be non-negative")
            if known + unknown != trusted:
                raise ValueError("known/unknown exposure counts must partition trusted rows")
            if rho_count > trusted:
                raise ValueError("rho count cannot exceed trusted sample count")
            if action_count > trusted:
                raise ValueError("real action supervision count cannot exceed trusted rows")
        expected_accepted = bool(self.audit_profile_passed or self.waiver_applied)
        if bool(self.accepted) != expected_accepted:
            raise ValueError("accepted must equal audit pass or explicit waiver")
        if self.audit_profile_passed and self.failed_checks:
            raise ValueError("a passed audit cannot contain failed checks")
        if self.waiver_applied:
            _require_sha256(self.waiver_sha256, "waiver_sha256")
        elif self.waiver_sha256 is not None:
            raise ValueError("waiver_sha256 requires waiver_applied=true")

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["failed_checks"] = list(self.failed_checks)
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "MeasuredDataAuditResultV1":
        values = dict(payload)
        values["failed_checks"] = tuple(values.get("failed_checks") or ())
        return cls(**values)

    def sha256(self) -> str:
        self.validate()
        return stable_json_sha256(self.to_json_dict())


@dataclass(frozen=True)
class PhysicsHybridV8DataContractV1:
    """Complete data identity for one reproducible V8 training/evaluation run."""

    contract_id: str
    manifest_hash: str
    manifest_summary_sha256: str
    profile_hashes_by_system: Dict[str, str]
    geometry_hashes_by_system: Dict[str, str]
    split_hash: str
    source_tree_hash: str
    config_hash: str
    audit_result_hash: str
    locked_final_test: LockedFinalTestDeclarationV1
    real_action_label_policy: str = REAL_ACTION_LABEL_POLICY
    global_multi_click_policy: str = GLOBAL_MULTI_CLICK_POLICY
    missing_rho_policy: str = MISSING_RHO_POLICY
    unknown_exposure_policy: str = UNKNOWN_EXPOSURE_POLICY
    hardware_write_authority: bool = False
    schema: str = V8_DATA_CONTRACT_SCHEMA
    schema_version: int = V8_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema != V8_DATA_CONTRACT_SCHEMA or self.schema_version != V8_SCHEMA_VERSION:
            raise ValueError("unsupported V8 data-contract schema/version")
        if not self.contract_id.strip():
            raise ValueError("contract_id is required")
        for field_name in (
            "manifest_hash",
            "manifest_summary_sha256",
            "split_hash",
            "source_tree_hash",
            "config_hash",
            "audit_result_hash",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        _validate_hash_mapping(self.profile_hashes_by_system, "profile_hashes_by_system")
        _validate_hash_mapping(self.geometry_hashes_by_system, "geometry_hashes_by_system")
        self.locked_final_test.validate()
        if self.locked_final_test.split_hash != self.split_hash:
            raise ValueError("locked final-test split hash disagrees with data contract")
        policies = {
            "real_action_label_policy": (
                self.real_action_label_policy,
                REAL_ACTION_LABEL_POLICY,
            ),
            "global_multi_click_policy": (
                self.global_multi_click_policy,
                GLOBAL_MULTI_CLICK_POLICY,
            ),
            "missing_rho_policy": (self.missing_rho_policy, MISSING_RHO_POLICY),
            "unknown_exposure_policy": (
                self.unknown_exposure_policy,
                UNKNOWN_EXPOSURE_POLICY,
            ),
        }
        for field_name, (actual, expected) in policies.items():
            if actual != expected:
                raise ValueError(f"{field_name} must be {expected!r}")
        if self.hardware_write_authority:
            raise ValueError("V8 training data contracts never grant hardware-write authority")

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["locked_final_test"] = self.locked_final_test.to_json_dict()
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "PhysicsHybridV8DataContractV1":
        values = dict(payload)
        locked = values.get("locked_final_test")
        if isinstance(locked, Mapping):
            values["locked_final_test"] = LockedFinalTestDeclarationV1.from_json_dict(locked)
        return cls(**values)

    def sha256(self) -> str:
        self.validate()
        return stable_json_sha256(self.to_json_dict())


__all__ = [
    "GLOBAL_MULTI_CLICK_POLICY",
    "HISTORY_ACTION_LABEL_SOURCE",
    "HistoryTransitionProvenanceV1",
    "LockedEnsembleReleaseAccessDeclarationV1",
    "LockedFinalTestDeclarationV1",
    "MISSING_RHO_POLICY",
    "MeasuredAuditWaiverV1",
    "MeasuredDataAuditError",
    "MeasuredDataAuditResultV1",
    "PhysicsHybridV8DataContractV1",
    "NO_ACTION_EVIDENCE_KINDS",
    "PAIRED_TRANSITION_ACTION_EVIDENCE",
    "REAL_ACTION_LABEL_POLICY",
    "SIGNED_PROBE_ACTION_EVIDENCE",
    "SUPPORTED_SYSTEM_TYPES",
    "UNKNOWN_EXPOSURE_POLICY",
    "V8_AUDIT_WAIVER_SCHEMA",
    "V8_DATA_CONTRACT_SCHEMA",
    "V8_HISTORY_TRANSITION_SCHEMA",
    "V8_LOCKED_ENSEMBLE_ACCESS_LEDGER_SCHEMA",
    "V8_LOCKED_ENSEMBLE_RELEASE_ACCESS_SCHEMA",
    "V8_LOCKED_FINAL_TEST_SCHEMA",
    "V8_MEASURED_AUDIT_SCHEMA",
    "canonical_ledger_path_sha256",
    "sample_ids_sha256",
    "stable_json_sha256",
]
