"""Fail-closed measured-data loading for the V8 feedback delivery track.

This layer reads the immutable measured manifests and their summaries without
rewriting them.  It preserves unknown exposure, missing rho, and global
multi-click semantics explicitly.  Real action supervision is available only
through a separately validated :class:`HistoryTransitionProvenanceV1` record.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ml_recon.measured_data.contracts import MeasuredSampleRecordV1
from ml_recon.measured_data.datasets import load_measured_records
from ml_recon.measured_data.quality import HARD_FAULT_MULTIPLIER, classify_population

from .contracts import (
    HistoryTransitionProvenanceV1,
    LockedFinalTestDeclarationV1,
    MeasuredAuditWaiverV1,
    MeasuredDataAuditError,
    MeasuredDataAuditResultV1,
    NO_ACTION_EVIDENCE_KINDS,
    SUPPORTED_SYSTEM_TYPES,
    sample_ids_sha256,
    stable_json_sha256,
)


V8_MEASURED_BUNDLE_SCHEMA = "physics_hybrid_v8_measured_bundle_v1"
V8_MEASURED_ROW_SCHEMA = "physics_hybrid_v8_measured_row_v1"
V8_MEASURED_BUNDLE_VERSION = 1

SUMMARY_FILE_NAME = "measured_data_summary.json"
SPLIT_FILE_NAME = "measured_splits_v1.json"
SPLIT_AUDIT_FILE_NAME = "split_audit.json"
COMMAND_PROVENANCE_FILE_NAME = "command_provenance_summary.json"
HARD_FAULT_FILE_NAME = "hard_fault_samples.jsonl"
GUARD_BAND_FILE_NAME = "guard_band_samples.jsonl"
V2_PROVENANCE_FILE_NAME = "operator_curated_corpus_v2.json"
TRUSTED_FILE_NAMES = {
    "3qubit": "trusted_samples_3q.jsonl",
    "4qubit": "trusted_samples_4q.jsonl",
}
EXPECTED_SUMMARY_SCHEMA = "measured_data_summary_v1"
EXPECTED_SUMMARY_SCHEMA_V2 = "measured_data_summary_v2"
EXPECTED_SPLIT_SCHEMA = "measured_splits_v1"
EXPECTED_COMMAND_PROVENANCE_SCHEMA = "command_provenance_summary_v1"
OPERATOR_CURATED_CORPUS_SCHEMA = "physics_hybrid_v8_operator_curated_corpus_v1"
OPERATOR_CURATED_CORPUS_SCHEMA_V2 = "physics_hybrid_v8_operator_curated_corpus_v2"
OPERATOR_CURATED_CORPUS_AUDIT_SCHEMA = (
    "physics_hybrid_v8_operator_curated_corpus_audit_v1"
)
OPERATOR_CURATED_CORPUS_AUDIT_SCHEMA_V2 = (
    "physics_hybrid_v8_operator_curated_corpus_audit_v2"
)
HARD_FAULT_KINDS = (
    "fpga_total_multi_exceeds_idler_fraction_contract",
    "fpga_read_idler_counter_stuck",
    "idler_low",
    "signal_high",
    "signal_high_idler_low",
)


@dataclass(frozen=True)
class V8MeasuredRowV1:
    """One trustworthy measured row with missingness kept out of value tensors."""

    record: MeasuredSampleRecordV1
    split_name: str
    exposure_s: Optional[float]
    exposure_known_mask: int
    calibration_hash: Optional[str]
    calibration_known_mask: int
    rho_hash: Optional[str]
    rho_kind: Optional[str]
    rho_available_mask: int
    global_multi_click_upper_bounds: Tuple[int, int, int]
    real_action_label: None = None
    real_action_label_mask: int = 0
    real_action_supervision_mask: int = 0
    action_evidence_kind: str = "none_target_only_snapshot"
    paired_episode_group_key: None = None
    operation_local_multi_click_counts: None = None
    source_id: str = "real"
    schema: str = V8_MEASURED_ROW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != V8_MEASURED_ROW_SCHEMA:
            raise ValueError("unsupported V8 measured-row schema")
        if self.source_id != "real":
            raise ValueError("V8 measured rows must retain source_id='real'")
        if not self.split_name:
            raise ValueError("split_name is required")
        exposure_mask = int(self.exposure_known_mask)
        if exposure_mask not in {0, 1}:
            raise ValueError("exposure_known_mask must be 0 or 1")
        if exposure_mask == 1 and (self.exposure_s is None or float(self.exposure_s) <= 0.0):
            raise ValueError("known exposure requires a positive exposure_s")
        if exposure_mask == 0 and self.exposure_s is not None:
            raise ValueError("unknown exposure must remain None")
        calibration_mask = int(self.calibration_known_mask)
        if calibration_mask not in {0, 1}:
            raise ValueError("calibration_known_mask must be 0 or 1")
        if calibration_mask != int(self.calibration_hash is not None):
            raise ValueError("calibration mask disagrees with calibration hash")
        if self.calibration_hash is not None:
            _require_sha256(self.calibration_hash, "calibration_hash")
        rho_mask = int(self.rho_available_mask)
        if rho_mask not in {0, 1} or rho_mask != int(bool(self.record.rho_available)):
            raise ValueError("rho_available_mask must exactly reflect the measured record")
        if rho_mask == 1:
            _require_sha256(self.rho_hash, "rho_hash")
            if not self.rho_kind or self.rho_kind == "missing":
                raise ValueError("available rho requires a concrete rho kind")
        elif self.rho_hash is not None or self.rho_kind is not None:
            raise ValueError("missing rho must remain None rather than a zero placeholder")
        if self.record.multi_click_scope != "global":
            raise ValueError("V8 legacy measured rows require explicit global multi-click scope")
        if self.operation_local_multi_click_counts is not None:
            raise ValueError("global multi-click totals cannot become operation-local pseudo-counts")
        if tuple(self.global_multi_click_upper_bounds) != (
            int(self.record.multi2_count),
            int(self.record.multi3_count),
            int(self.record.multi4_count),
        ):
            raise ValueError("global multi-click upper bounds disagree with the manifest")
        if (
            self.real_action_label is not None
            or int(self.real_action_label_mask) != 0
            or int(self.real_action_supervision_mask) != 0
        ):
            raise ValueError("ordinary measured rows cannot fabricate real action labels")
        if self.action_evidence_kind not in NO_ACTION_EVIDENCE_KINDS:
            raise ValueError("ordinary measured rows cannot claim paired/probe action evidence")
        if (
            self.record.command_provenance == "target_only_approximation"
            and self.action_evidence_kind != "none_target_only_snapshot"
        ):
            raise ValueError("target-only snapshots must declare no action evidence")
        if self.paired_episode_group_key is not None:
            raise ValueError("ordinary measured rows are not verified action episodes")

    @property
    def sample_id(self) -> str:
        return self.record.sample_id

    @property
    def system_type(self) -> str:
        return self.record.system_type

    @property
    def measurement_group_key(self) -> str:
        """Leakage-control group, explicitly not an action-transition identity."""

        return self.record.repeat_group_id


@dataclass(frozen=True)
class V8MeasuredDataBundleV1:
    """Read-only audited rows plus immutable summary/split identities."""

    rows: Tuple[V8MeasuredRowV1, ...]
    audit: MeasuredDataAuditResultV1
    summary: Dict[str, Any]
    split_assignments: Dict[str, str]
    curated_corpus_audit: Dict[str, Any]
    schema: str = V8_MEASURED_BUNDLE_SCHEMA
    schema_version: int = V8_MEASURED_BUNDLE_VERSION

    def __post_init__(self) -> None:
        if self.schema != V8_MEASURED_BUNDLE_SCHEMA or self.schema_version != V8_MEASURED_BUNDLE_VERSION:
            raise ValueError("unsupported V8 measured-bundle schema/version")
        self.audit.validate()
        if not self.audit.accepted:
            raise MeasuredDataAuditError("measured bundle was not accepted by its audit")
        sample_ids = tuple(row.sample_id for row in self.rows)
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ValueError("V8 measured bundle sample IDs must be non-empty and unique")
        if set(sample_ids) != set(self.split_assignments):
            raise ValueError("split assignments must exactly cover V8 measured rows")
        for row in self.rows:
            if self.split_assignments[row.sample_id] != row.split_name:
                raise ValueError(f"row split mismatch for {row.sample_id}")
        curation = dict(self.curated_corpus_audit)
        if curation.get("schema") not in {
            OPERATOR_CURATED_CORPUS_AUDIT_SCHEMA,
            OPERATOR_CURATED_CORPUS_AUDIT_SCHEMA_V2,
        }:
            raise ValueError("unsupported operator-curated corpus audit schema")
        if curation.get("manifest_hash") != self.audit.manifest_hash:
            raise ValueError("operator-curated corpus audit manifest hash mismatch")
        if curation.get("summary_sha256") != self.audit.summary_sha256:
            raise ValueError("operator-curated corpus audit summary hash mismatch")
        if int(curation.get("trusted_sample_count", -1)) != len(self.rows):
            raise ValueError("operator-curated corpus audit trusted count mismatch")
        if curation.get("classification_recomputed_from_manifest_totals") is not True:
            raise ValueError("operator-curated corpus classification must be recomputed")

    @property
    def records(self) -> Tuple[MeasuredSampleRecordV1, ...]:
        return tuple(row.record for row in self.rows)

    @property
    def manifest_hash(self) -> str:
        return self.audit.manifest_hash

    @property
    def split_hash(self) -> str:
        return self.audit.split_hash

    @property
    def summary_sha256(self) -> str:
        return self.audit.summary_sha256


def load_v8_measured_data_bundle(
    manifest_dir: str | Path = "ml_recon/reports/measured_data_v2_3",
    *,
    audit_waiver: Optional[MeasuredAuditWaiverV1 | Mapping[str, Any]] = None,
    curated_corpus_contract: Optional[Mapping[str, Any]] = None,
    now: Optional[datetime] = None,
) -> V8MeasuredDataBundleV1:
    """Load every trusted manifest row after a whole-bundle fail-closed audit.

    Only a failed, internally consistent audit-profile expectation may be
    waived.  Hash mismatch, split leakage, malformed records, and inconsistent
    summaries always fail and cannot be waived.
    """

    directory = Path(manifest_dir)
    parsed_waiver: Optional[MeasuredAuditWaiverV1]
    if audit_waiver is None:
        parsed_waiver = None
    elif isinstance(audit_waiver, MeasuredAuditWaiverV1):
        parsed_waiver = audit_waiver
    elif isinstance(audit_waiver, Mapping):
        parsed_waiver = (
            MeasuredAuditWaiverV1.from_json_dict(audit_waiver)
            if dict(audit_waiver)
            else None
        )
    else:
        raise TypeError("audit_waiver must be MeasuredAuditWaiverV1, mapping, or None")
    required_files = (
        SUMMARY_FILE_NAME,
        SPLIT_FILE_NAME,
        SPLIT_AUDIT_FILE_NAME,
        COMMAND_PROVENANCE_FILE_NAME,
        HARD_FAULT_FILE_NAME,
        *TRUSTED_FILE_NAMES.values(),
    )
    for file_name in required_files:
        path = directory / file_name
        if not path.is_file():
            raise FileNotFoundError(f"strict V8 measured-data input missing: {path}")

    summary_path = directory / SUMMARY_FILE_NAME
    summary = _read_json_object(summary_path)
    summary_schema = str(summary.get("schema") or "")
    is_v2 = summary_schema == EXPECTED_SUMMARY_SCHEMA_V2
    if summary_schema not in {EXPECTED_SUMMARY_SCHEMA, EXPECTED_SUMMARY_SCHEMA_V2}:
        raise ValueError("unsupported measured-data summary schema")
    if is_v2:
        for file_name in (GUARD_BAND_FILE_NAME, V2_PROVENANCE_FILE_NAME):
            path = directory / file_name
            if not path.is_file():
                raise FileNotFoundError(f"strict V2 measured-data input missing: {path}")
    summary_sha256 = _sha256_file(summary_path)
    split_payload = _read_json_object(directory / SPLIT_FILE_NAME)
    split_audit = _read_json_object(directory / SPLIT_AUDIT_FILE_NAME)
    command_summary = _read_json_object(directory / COMMAND_PROVENANCE_FILE_NAME)

    trusted_records = load_measured_records(directory, trusted_only=True)
    hard_fault_records = _read_record_jsonl(directory / HARD_FAULT_FILE_NAME)
    guard_band_records = (
        _read_record_jsonl(directory / GUARD_BAND_FILE_NAME) if is_v2 else []
    )
    quality_audit = _validate_record_collections(
        trusted_records,
        hard_fault_records,
        guard_band_records,
    )
    all_records = sorted(
        [*trusted_records, *hard_fault_records, *guard_band_records],
        key=lambda record: record.sample_id,
    )
    manifest_hash = stable_json_sha256([record.to_json_dict() for record in all_records])
    _validate_summary(summary, all_records, manifest_hash)
    curated_corpus_audit = _validate_operator_curated_corpus_binding(
        curated_corpus_contract,
        manifest_hash=manifest_hash,
        summary_sha256=summary_sha256,
        hard_fault_manifest_sha256=_sha256_file(directory / HARD_FAULT_FILE_NAME),
        guard_band_manifest_sha256=(
            _sha256_file(directory / GUARD_BAND_FILE_NAME) if is_v2 else None
        ),
        v2_provenance=(
            _read_json_object(directory / V2_PROVENANCE_FILE_NAME) if is_v2 else None
        ),
        v2_provenance_file_sha256=(
            _sha256_file(directory / V2_PROVENANCE_FILE_NAME) if is_v2 else None
        ),
        quality_audit=quality_audit,
    )
    split_hash, assignments = _validate_splits(
        split_payload,
        split_audit,
        trusted_records,
    )
    _validate_command_provenance_summary(command_summary, trusted_records)
    audit_profile_passed, failed_checks = _validate_audit_profile(summary)

    waiver_applied = False
    waiver_sha256 = None
    if not audit_profile_passed:
        if parsed_waiver is None:
            raise MeasuredDataAuditError(
                "measured-data audit profile failed; an explicit summary-hash-bound waiver "
                f"is required for checks: {', '.join(failed_checks)}"
            )
        parsed_waiver.validate(
            reference_summary_sha256=summary_sha256,
            now=now,
        )
        waiver_applied = True
        waiver_sha256 = parsed_waiver.sha256()
    elif parsed_waiver is not None:
        raise MeasuredDataAuditError("a passed measured audit must not carry a waiver")

    rows = tuple(
        _build_measured_row(record, assignments[record.sample_id])
        for record in sorted(trusted_records, key=lambda item: item.sample_id)
    )
    counts = _audit_counts(trusted_records, hard_fault_records)
    result = MeasuredDataAuditResultV1(
        accepted=bool(audit_profile_passed or waiver_applied),
        audit_profile_passed=audit_profile_passed,
        waiver_applied=waiver_applied,
        failed_checks=failed_checks,
        summary_sha256=summary_sha256,
        manifest_hash=manifest_hash,
        split_hash=split_hash,
        trusted_sample_count_by_system=counts["trusted"],
        rho_available_count_by_system=counts["rho"],
        exposure_known_count_by_system=counts["exposure_known"],
        exposure_unknown_count_by_system=counts["exposure_unknown"],
        real_action_supervision_count_by_system=counts["real_action_supervision"],
        excluded_rho_by_fault_by_system=counts["excluded_rho_by_fault"],
        waiver_sha256=waiver_sha256,
    )
    result.validate()
    return V8MeasuredDataBundleV1(
        rows=rows,
        audit=result,
        summary=summary,
        split_assignments=assignments,
        curated_corpus_audit=curated_corpus_audit,
    )


def declare_locked_final_test(
    bundle: V8MeasuredDataBundleV1,
    *,
    final_test_split_names: Sequence[str],
    locked_at: str,
) -> LockedFinalTestDeclarationV1:
    """Bind the exact final-test membership before model selection begins."""

    names = tuple(str(name) for name in final_test_split_names)
    selected_ids = [
        row.sample_id
        for row in bundle.rows
        if row.split_name in set(names)
    ]
    declaration = LockedFinalTestDeclarationV1(
        split_hash=bundle.split_hash,
        final_test_split_names=names,
        final_test_sample_ids_sha256=sample_ids_sha256(selected_ids),
        final_test_sample_count=len(selected_ids),
        locked_at=locked_at,
    )
    declaration.validate()
    return declaration


def load_history_transition_provenance(
    path: str | Path,
    *,
    bundle: V8MeasuredDataBundleV1,
) -> Tuple[HistoryTransitionProvenanceV1, ...]:
    """Load and bind explicit real action transitions; never infer them."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"history transition provenance missing: {source}")
    if source.suffix.lower() == ".jsonl":
        payloads = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        payloads = payload if isinstance(payload, list) else [payload]
    transitions = tuple(
        HistoryTransitionProvenanceV1.from_json_dict(item)
        for item in payloads
    )
    transition_ids = [item.transition_id for item in transitions]
    if not transitions or len(transition_ids) != len(set(transition_ids)):
        raise ValueError("history transitions must be non-empty and uniquely identified")
    for transition in transitions:
        validate_history_transition_binding(transition, bundle=bundle)
    return transitions


def validate_history_transition_binding(
    transition: HistoryTransitionProvenanceV1,
    *,
    bundle: V8MeasuredDataBundleV1,
) -> None:
    """Require before/after hashes to match the immutable measured records."""

    transition.validate()
    by_id = {row.sample_id: row for row in bundle.rows}
    before = by_id.get(transition.before_sample_id)
    after = by_id.get(transition.after_sample_id)
    if before is None or after is None:
        raise ValueError("history transition references a sample outside the audited bundle")
    if before.system_type != transition.system_type or after.system_type != transition.system_type:
        raise ValueError("history transition crosses system types")
    if before.record.snapshot_manifest_hash != transition.before_snapshot_hash:
        raise ValueError("before snapshot hash disagrees with the measured manifest")
    if after.record.snapshot_manifest_hash != transition.after_snapshot_hash:
        raise ValueError("after snapshot hash disagrees with the measured manifest")
    if after.exposure_known_mask != transition.exposure_known_mask:
        raise ValueError("transition exposure mask disagrees with the after snapshot")
    if transition.exposure_known_mask == 1 and abs(
        float(transition.exposure_s) - float(after.exposure_s)
    ) > 1e-9:
        raise ValueError("transition exposure disagrees with the after snapshot")
    if after.calibration_known_mask == 1:
        if transition.calibration_hash != after.calibration_hash:
            raise ValueError("transition calibration hash disagrees with the after snapshot")
    else:
        raise ValueError("real action transition requires a measured calibration hash")


def source_tree_sha256(
    source_root: str | Path,
    relative_paths: Iterable[str | Path],
) -> str:
    """Hash exact source paths and bytes for a contract-bound training tree."""

    root = Path(source_root).resolve()
    entries: List[Dict[str, str]] = []
    normalized_paths = sorted({Path(path).as_posix() for path in relative_paths})
    if not normalized_paths:
        raise ValueError("source tree must contain at least one file")
    for relative_text in normalized_paths:
        path = (root / relative_text).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("source path escapes source_root") from exc
        if not path.is_file():
            raise FileNotFoundError(f"source-tree file missing: {path}")
        entries.append({"path": relative, "sha256": _sha256_file(path)})
    return stable_json_sha256(entries)


def _read_json_object(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _read_record_jsonl(path: Path) -> List[MeasuredSampleRecordV1]:
    records: List[MeasuredSampleRecordV1] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(MeasuredSampleRecordV1.from_json_dict(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"invalid measured record at {path}:{line_number}: {exc}") from exc
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field_name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return text


def validate_operator_curated_corpus_contract(
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the operator-authorized 2026-08-06 corpus declaration.

    The declaration pins the *result* of the one-time curation.  It does not
    authorize a loader to rediscover snapshots, tune a new outlier threshold,
    or reinstate excluded rows.  The actual manifest/count comparison happens
    in :func:`load_v8_measured_data_bundle`.
    """

    value = dict(contract or {})
    if value.get("schema") == OPERATOR_CURATED_CORPUS_SCHEMA_V2:
        return _validate_operator_curated_corpus_contract_v2(value)
    if value.get("schema") != OPERATOR_CURATED_CORPUS_SCHEMA:
        raise ValueError("unsupported operator-curated corpus contract schema")
    if value.get("curated_at_local_date") != "2026-08-06":
        raise ValueError("operator-curated corpus date must remain 2026-08-06")
    if value.get("curation_tool") != "deepseek":
        raise ValueError("operator-curated corpus must record the declared curation tool")
    if value.get("operator_authorized") is not True:
        raise ValueError("measured corpus curation requires explicit operator authority")
    if not str(value.get("physical_exclusion_reason") or "").strip():
        raise ValueError("operator-curated corpus requires a physical exclusion reason")
    for field_name in (
        "manifest_hash",
        "summary_sha256",
        "hard_fault_manifest_sha256",
    ):
        _require_sha256(value.get(field_name), field_name)

    rule = dict(value.get("hard_fault_rule") or {})
    if rule.get("population_scope") != "per_system":
        raise ValueError("hard-fault population reference must be per system")
    if float(rule.get("hard_fault_multiplier", -1.0)) != HARD_FAULT_MULTIPLIER:
        raise ValueError("hard-fault multiplier disagrees with measured-data quality code")
    if rule.get("exclude_signal_high") is not True or rule.get("exclude_idler_low") is not True:
        raise ValueError("signal-high and idler-low hard faults must both remain excluded")
    if rule.get("ordinary_noise_retained") is not True:
        raise ValueError("ordinary measured noise must remain in the trusted corpus")

    required_policy = {
        "trusted_manifest_only": True,
        "raw_snapshot_rediscovery_allowed": False,
        "trusted_rows_are_final_usable_corpus": True,
        "trusted_rows_may_be_reclassified_or_dropped": False,
        "locked_split_is_evaluation_role_not_quality_exclusion": True,
        "all_trusted_rows_allowed_for_post_selection_refit": True,
        "excluded_hard_fault_rows_allowed_for_gradient": False,
        "excluded_hard_fault_rows_allowed_for_evaluation": False,
        "excluded_hard_fault_rows_allowed_for_all_eligible_refit": False,
    }
    policy = dict(value.get("usage_policy") or {})
    mismatches = [
        name for name, expected in required_policy.items()
        if policy.get(name) is not expected
    ]
    if mismatches:
        raise ValueError(f"operator-curated corpus usage policy changed: {mismatches}")

    trusted_by_system = _validated_system_count_mapping(
        value.get("trusted_sample_count_by_system"),
        "trusted_sample_count_by_system",
    )
    excluded_by_system = _validated_system_count_mapping(
        value.get("excluded_hard_fault_count_by_system"),
        "excluded_hard_fault_count_by_system",
    )
    if int(value.get("trusted_sample_count", -1)) != sum(trusted_by_system.values()):
        raise ValueError("operator-curated trusted total disagrees with per-system counts")
    if int(value.get("excluded_hard_fault_sample_count", -1)) != sum(
        excluded_by_system.values()
    ):
        raise ValueError("operator-curated exclusion total disagrees with per-system counts")

    kinds = dict(value.get("hard_fault_kind_count_by_system") or {})
    if set(kinds) != SUPPORTED_SYSTEM_TYPES:
        raise ValueError("hard-fault kind counts must cover exactly both systems")
    normalized_kinds: Dict[str, Dict[str, int]] = {}
    for system_type in sorted(SUPPORTED_SYSTEM_TYPES):
        raw_counts = dict(kinds[system_type] or {})
        if not set(raw_counts).issubset(HARD_FAULT_KINDS):
            raise ValueError(f"unknown hard-fault kind for {system_type}")
        counts = {str(name): int(count) for name, count in raw_counts.items()}
        if any(count < 0 for count in counts.values()):
            raise ValueError("hard-fault kind counts must be non-negative")
        if sum(counts.values()) != excluded_by_system[system_type]:
            raise ValueError(f"hard-fault kind counts disagree for {system_type}")
        normalized_kinds[system_type] = dict(sorted(counts.items()))

    value["trusted_sample_count_by_system"] = trusted_by_system
    value["excluded_hard_fault_count_by_system"] = excluded_by_system
    value["hard_fault_kind_count_by_system"] = normalized_kinds
    value["hard_fault_rule"] = rule
    value["usage_policy"] = policy
    return value


def _validate_operator_curated_corpus_contract_v2(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the operator-authorized guard-band V2 declaration."""

    parsed = dict(value)
    if parsed.get("operator_authorized_at_local_date") != "2026-08-12":
        raise ValueError("V2 operator authorization date must remain 2026-08-12")
    if parsed.get("timezone") != "Asia/Shanghai":
        raise ValueError("V2 operator authorization timezone must be Asia/Shanghai")
    if parsed.get("operator_authorized") is not True:
        raise ValueError("V2 measured corpus requires explicit operator authority")
    for field_name in (
        "parent_manifest_hash",
        "parent_summary_sha256",
        "parent_split_hash",
        "parent_hard_fault_manifest_sha256",
        "manifest_hash",
        "summary_sha256",
        "hard_fault_manifest_sha256",
        "guard_band_manifest_sha256",
        "v2_provenance_file_sha256",
    ):
        _require_sha256(parsed.get(field_name), field_name)

    hard_rule = dict(parsed.get("hard_fault_rule") or {})
    if (
        hard_rule.get("population_scope") != "per_system"
        or float(hard_rule.get("hard_fault_multiplier", -1.0))
        != HARD_FAULT_MULTIPLIER
        or hard_rule.get("exclude_signal_high") is not True
        or hard_rule.get("exclude_idler_low") is not True
        or hard_rule.get("ordinary_noise_retained") is not True
        or hard_rule.get("changed_from_v1") is not False
    ):
        raise ValueError("V2 changed the operator-authorized direct hard-fault rule")

    from ml_recon.measured_data.build_manifest_v2 import (
        V2_FPGA_ANOMALY_KIND,
        V2_FPGA_COIN_GIVEN_SIGNAL_MIN,
        V2_FPGA_IDLER_WINDOW_RATE_MIN,
        V2_FPGA_MULTI2_COUNT_MIN,
        V2_FPGA_MULTI2_PAIR_SHARE_MIN,
        V2_MULTI_RATIO_ANOMALY_KIND,
        V2_TOTAL_MULTI_TO_IDLER_RATIO_MAX,
        V2_OPERATION_COUNT_BY_SYSTEM,
        V2_POLICY_VERSION,
    )

    hardware_rule = dict(parsed.get("hardware_anomaly_rule") or {})
    required_hardware_rule = {
        "kinds": [V2_FPGA_ANOMALY_KIND, V2_MULTI_RATIO_ANOMALY_KIND],
        "population_scope": "per_row_physical_invariants",
        "operation_count_by_system": V2_OPERATION_COUNT_BY_SYSTEM,
        "idler_window_rate_min": V2_FPGA_IDLER_WINDOW_RATE_MIN,
        "coin_given_signal_min": V2_FPGA_COIN_GIVEN_SIGNAL_MIN,
        "largest_multi2_pair_share_min": V2_FPGA_MULTI2_PAIR_SHARE_MIN,
        "multi2_count_min": V2_FPGA_MULTI2_COUNT_MIN,
        "counter_stuck_all_checks_required": True,
        "total_multi_to_idler_ratio_max": V2_TOTAL_MULTI_TO_IDLER_RATIO_MAX,
        "total_multi_ratio_operator_semantic": (
            "global_multi2_plus_multi3_plus_multi4_must_not_exceed_"
            "twenty_percent_of_global_idler_count"
        ),
        "rule_combination": "counter_stuck_or_total_multi_to_idler_ratio",
        "model_metric_or_fit_used": False,
        "low_count_or_state_rarity_used": False,
        "temperature_used_as_hard_exclusion": False,
    }
    hardware_mismatches = [
        name for name, expected in required_hardware_rule.items()
        if hardware_rule.get(name) != expected
    ]
    if hardware_mismatches:
        raise ValueError(
            "V2 FPGA hardware-anomaly rule changed: "
            f"{hardware_mismatches}"
        )
    matched_sample_ids = tuple(
        str(value) for value in hardware_rule.get("matched_sample_ids") or ()
    )
    matched_samples = tuple(hardware_rule.get("matched_samples") or ())
    if (
        int(hardware_rule.get("matched_sample_count", -1)) != 1
        or len(matched_sample_ids) != 1
        or len(set(matched_sample_ids)) != 1
        or len(matched_samples) != 1
        or str(dict(matched_samples[0]).get("sample_id")) != matched_sample_ids[0]
        or dict(matched_samples[0]).get("matched") is not True
        or dict(matched_samples[0]).get("total_multi_ratio_matched") is not True
        or abs(
            float(dict(matched_samples[0]).get("total_multi_to_idler_ratio", 0.0))
            - ((3904555.0 + 1849.0 + 2.0) / 13024916.0)
        ) > 1.0e-15
    ):
        raise ValueError("V2 FPGA hardware-anomaly evidence set changed")

    guard_policy = dict(parsed.get("guard_policy") or {})
    if (
        guard_policy.get("policy_version") != V2_POLICY_VERSION
        or int(guard_policy.get("distance", 0)) != 1
        or tuple(guard_policy.get("directions") or ()) != ("before", "after")
        or guard_policy.get("guard_rows_are_direct_hard_faults") is not False
        or guard_policy.get("maximum_neighbor_gap_s") is not None
        or guard_policy.get("chronological_adjacency_has_time_gap_limit") is not False
        or tuple(guard_policy.get("chronological_order_fields") or ())
        != ("timestamp", "ended_at", "sample_id")
        or guard_policy.get("time_block_identity_required") is not False
    ):
        raise ValueError("V2 guard-band policy changed")

    required_policy = {
        "trusted_manifest_only": True,
        "raw_snapshot_rediscovery_allowed": False,
        "v1_direct_hard_fault_rows_reclassified": False,
        "v2_trusted_rows_are_final_usable_corpus": True,
        "v2_trusted_rows_may_be_reclassified_or_dropped": False,
        "guard_band_rows_allowed_for_gradient": False,
        "guard_band_rows_allowed_for_evaluation": False,
        "guard_band_rows_allowed_for_all_eligible_refit": False,
        "direct_hard_fault_rows_allowed_for_gradient": False,
        "direct_hard_fault_rows_allowed_for_evaluation": False,
        "direct_hard_fault_rows_allowed_for_all_eligible_refit": False,
        "locked_split_is_evaluation_role_not_quality_exclusion": True,
        "all_v2_trusted_rows_allowed_for_post_selection_refit": True,
    }
    usage_policy = dict(parsed.get("usage_policy") or {})
    mismatches = [
        name for name, expected in required_policy.items()
        if usage_policy.get(name) is not expected
    ]
    if mismatches:
        raise ValueError(f"V2 corpus usage policy changed: {mismatches}")

    trusted_by_system = _validated_system_count_mapping(
        parsed.get("trusted_sample_count_by_system"),
        "trusted_sample_count_by_system",
    )
    direct_by_system = _validated_system_count_mapping(
        parsed.get("excluded_hard_fault_count_by_system"),
        "excluded_hard_fault_count_by_system",
    )
    guard_by_system = _validated_system_count_mapping(
        parsed.get("excluded_guard_band_count_by_system"),
        "excluded_guard_band_count_by_system",
    )
    if int(parsed.get("trusted_sample_count", -1)) != sum(trusted_by_system.values()):
        raise ValueError("V2 trusted total disagrees with per-system counts")
    if int(parsed.get("excluded_hard_fault_sample_count", -1)) != sum(
        direct_by_system.values()
    ):
        raise ValueError("V2 direct hard-fault total disagrees with per-system counts")
    if int(parsed.get("excluded_guard_band_sample_count", -1)) != sum(
        guard_by_system.values()
    ):
        raise ValueError("V2 guard-band total disagrees with per-system counts")

    kinds = dict(parsed.get("hard_fault_kind_count_by_system") or {})
    if set(kinds) != SUPPORTED_SYSTEM_TYPES:
        raise ValueError("V2 hard-fault kind counts must cover both systems")
    normalized_kinds: Dict[str, Dict[str, int]] = {}
    for system_type in sorted(SUPPORTED_SYSTEM_TYPES):
        counts = {
            str(name): int(count)
            for name, count in dict(kinds[system_type] or {}).items()
        }
        if not set(counts).issubset(HARD_FAULT_KINDS):
            raise ValueError(f"unknown V2 hard-fault kind for {system_type}")
        if any(count < 0 for count in counts.values()):
            raise ValueError("V2 hard-fault kind counts must be non-negative")
        if sum(counts.values()) != direct_by_system[system_type]:
            raise ValueError(f"V2 hard-fault kind counts disagree for {system_type}")
        normalized_kinds[system_type] = dict(sorted(counts.items()))

    environment = dict(parsed.get("environment_quality_contract") or {})
    if (
        environment.get("role")
        != "weak_analysis_and_preregistered_training_view_candidate"
        or environment.get("hard_label_or_exclusion_authority") is not False
        or float(environment.get("temperature_threshold_c", -1.0)) != 21.5
        or environment.get("humidity_unit") != "percent_rh"
        or float(environment.get("sensor_distance_from_platform_m_approx", -1.0))
        != 2.0
    ):
        raise ValueError("V2 environment weak-quality contract changed")

    parsed["hard_fault_rule"] = hard_rule
    parsed["hardware_anomaly_rule"] = hardware_rule
    parsed["guard_policy"] = guard_policy
    parsed["usage_policy"] = usage_policy
    parsed["environment_quality_contract"] = environment
    parsed["trusted_sample_count_by_system"] = trusted_by_system
    parsed["excluded_hard_fault_count_by_system"] = direct_by_system
    parsed["excluded_guard_band_count_by_system"] = guard_by_system
    parsed["hard_fault_kind_count_by_system"] = normalized_kinds
    return parsed


def _validated_system_count_mapping(value: Any, field_name: str) -> Dict[str, int]:
    counts = {str(name): int(count) for name, count in dict(value or {}).items()}
    if set(counts) != SUPPORTED_SYSTEM_TYPES or any(count < 0 for count in counts.values()):
        raise ValueError(f"{field_name} must contain non-negative 3Q/4Q counts")
    return dict(sorted(counts.items()))


def _validate_record_collections(
    trusted_records: Sequence[MeasuredSampleRecordV1],
    hard_fault_records: Sequence[MeasuredSampleRecordV1],
    guard_band_records: Sequence[MeasuredSampleRecordV1] = (),
) -> Dict[str, Any]:
    if not trusted_records:
        raise ValueError("trusted measured manifests are empty")
    for record in trusted_records:
        if record.quality_status != "trusted" or record.hard_fault_kind is not None:
            raise ValueError(f"trusted manifest contains a hard-fault row: {record.sample_id}")
    for record in hard_fault_records:
        if record.quality_status == "trusted" or not record.hard_fault_kind:
            raise ValueError(f"hard-fault manifest contains a trusted row: {record.sample_id}")
    for record in guard_band_records:
        if record.quality_status != "excluded_guard_band":
            raise ValueError(f"guard-band manifest has wrong status: {record.sample_id}")
        if record.hard_fault_kind is not None:
            raise ValueError(f"guard-band row claims a direct hard fault: {record.sample_id}")
    if guard_band_records:
        _validate_v2_guard_policy(
            trusted_records,
            hard_fault_records,
            guard_band_records,
        )
    combined = [*trusted_records, *hard_fault_records, *guard_band_records]
    sample_ids = [record.sample_id for record in combined]
    paths = [record.snapshot_path_relative for record in combined]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("measured manifests contain duplicate sample IDs")
    if len(paths) != len(set(paths)):
        raise ValueError("measured manifests contain duplicate snapshot paths")
    if {record.system_type for record in trusted_records} != SUPPORTED_SYSTEM_TYPES:
        raise ValueError("trusted manifests must contain both 3qubit and 4qubit rows")
    for record in combined:
        if float(record.duration_s) < 0.0 or float(record.gate_exposure) < 0.0:
            raise ValueError(f"negative exposure for {record.sample_id}")
        duration_known = float(record.duration_s) > 0.0
        gate_known = float(record.gate_exposure) > 0.0
        if duration_known != gate_known:
            raise ValueError(f"partial exposure provenance for {record.sample_id}")
        if record.rho_available:
            _require_sha256(record.rho_hash, f"rho_hash[{record.sample_id}]")
        elif record.rho_hash is not None:
            raise ValueError(f"missing rho has a non-null hash for {record.sample_id}")
        if record.calibration_snapshot_hash is not None:
            _require_sha256(
                record.calibration_snapshot_hash,
                f"calibration_snapshot_hash[{record.sample_id}]",
            )
        # ``multi2_pairs`` names global detector-pair totals in legacy rows.  It
        # is intentionally not copied into V8MeasuredRowV1, because those six
        # detector totals are not operation-local count supervision.

    trusted_count_by_system: Dict[str, int] = {}
    excluded_count_by_system: Dict[str, int] = {}
    hard_fault_kind_count_by_system: Dict[str, Dict[str, int]] = {}
    for system_type in sorted(SUPPORTED_SYSTEM_TYPES):
        system_records = sorted(
            (record for record in combined if record.system_type == system_type),
            key=lambda record: record.sample_id,
        )
        diagnostics = classify_population(
            (int(record.signal_total), int(record.idler_total))
            for record in system_records
        )
        guard_ids = {record.sample_id for record in guard_band_records}
        hard_ids = {record.sample_id for record in hard_fault_records}
        trusted_ids = {record.sample_id for record in trusted_records}
        for record, diagnostic in zip(system_records, diagnostics):
            if record.sample_id in guard_ids:
                valid = (
                    diagnostic.quality_status == "trusted"
                    and diagnostic.hard_fault_kind is None
                    and record.exclusion_reason
                    == "adjacent_to_operator_confirmed_lock_loss_segment"
                )
            elif record.sample_id in hard_ids:
                if record.hard_fault_kind in {
                    "fpga_read_idler_counter_stuck",
                    "fpga_total_multi_exceeds_idler_fraction_contract",
                }:
                    from ml_recon.measured_data.build_manifest_v2 import (
                        V2_FPGA_ANOMALY_REASON,
                        V2_MULTI_RATIO_ANOMALY_REASON,
                        fpga_hardware_anomaly_evidence,
                    )

                    evidence = fpga_hardware_anomaly_evidence(record)
                    expected_reason = (
                        V2_MULTI_RATIO_ANOMALY_REASON
                        if record.hard_fault_kind
                        == "fpga_total_multi_exceeds_idler_fraction_contract"
                        else V2_FPGA_ANOMALY_REASON
                    )
                    expected_match = (
                        bool(evidence["total_multi_ratio_matched"])
                        if record.hard_fault_kind
                        == "fpga_total_multi_exceeds_idler_fraction_contract"
                        else bool(evidence["counter_stuck_matched"])
                    )
                    valid = bool(
                        record.quality_status == "excluded_hard_fault"
                        and record.exclusion_reason == expected_reason
                        and expected_match
                    )
                else:
                    valid = (
                        diagnostic.quality_status == record.quality_status
                        and diagnostic.exclusion_reason == record.exclusion_reason
                        and diagnostic.hard_fault_kind == record.hard_fault_kind
                    )
            else:
                valid = (
                    record.sample_id in trusted_ids
                    and diagnostic.quality_status == "trusted"
                    and diagnostic.hard_fault_kind is None
                )
            if not valid:
                raise MeasuredDataAuditError(
                    "trusted/excluded manifest disagrees with the operator-authorized "
                    f"signal/idler hard-fault rule for {record.sample_id}"
                )
        trusted_count_by_system[system_type] = sum(
            record.sample_id in trusted_ids for record in system_records
        )
        excluded_count_by_system[system_type] = sum(
            record.sample_id in hard_ids for record in system_records
        )
        hard_fault_kind_count_by_system[system_type] = dict(sorted(Counter(
            record.hard_fault_kind
            for record in system_records
            if record.sample_id in hard_ids and record.hard_fault_kind is not None
        ).items()))
    return {
        "classification_recomputed_from_manifest_totals": True,
        "population_scope": "per_system",
        "hard_fault_multiplier": HARD_FAULT_MULTIPLIER,
        "trusted_sample_count_by_system": trusted_count_by_system,
        "excluded_hard_fault_count_by_system": excluded_count_by_system,
        "excluded_guard_band_count_by_system": {
            system_type: sum(
                record.system_type == system_type for record in guard_band_records
            )
            for system_type in sorted(SUPPORTED_SYSTEM_TYPES)
        },
        "hard_fault_kind_count_by_system": hard_fault_kind_count_by_system,
    }


def _validate_v2_guard_policy(
    trusted_records: Sequence[MeasuredSampleRecordV1],
    hard_fault_records: Sequence[MeasuredSampleRecordV1],
    guard_band_records: Sequence[MeasuredSampleRecordV1],
) -> None:
    from ml_recon.measured_data.build_manifest_v2 import (
        V2_GUARD_REASON,
        V2_POLICY_VERSION,
        derive_v2_records,
    )

    reconstructed_guard_rows = [
        replace(
            record,
            quality_status="trusted",
            exclusion_reason=None,
            guard_source_sample_ids=[],
            guard_directions=[],
            guard_distance=None,
            guard_policy_version=None,
            guard_time_gap_s=None,
        )
        for record in guard_band_records
    ]
    _, expected_guards = derive_v2_records(
        [*trusted_records, *reconstructed_guard_rows],
        hard_fault_records,
    )
    expected_by_id = {record.sample_id: record for record in expected_guards}
    actual_by_id = {record.sample_id: record for record in guard_band_records}
    if set(expected_by_id) != set(actual_by_id):
        raise MeasuredDataAuditError("V2 guard-band membership is not reproducible")
    for sample_id, actual in actual_by_id.items():
        expected = expected_by_id[sample_id]
        if (
            actual.exclusion_reason != V2_GUARD_REASON
            or actual.guard_policy_version != V2_POLICY_VERSION
            or int(actual.guard_distance or 0) != 1
            or tuple(actual.guard_source_sample_ids)
            != tuple(expected.guard_source_sample_ids)
            or tuple(actual.guard_directions) != tuple(expected.guard_directions)
            or abs(float(actual.guard_time_gap_s) - float(expected.guard_time_gap_s))
            > 1e-9
        ):
            raise MeasuredDataAuditError(
                f"V2 guard-band provenance mismatch for {sample_id}"
            )


def _validate_operator_curated_corpus_binding(
    contract: Optional[Mapping[str, Any]],
    *,
    manifest_hash: str,
    summary_sha256: str,
    hard_fault_manifest_sha256: str,
    guard_band_manifest_sha256: Optional[str],
    v2_provenance: Optional[Mapping[str, Any]],
    v2_provenance_file_sha256: Optional[str],
    quality_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    parsed = (
        validate_operator_curated_corpus_contract(contract)
        if contract is not None
        else None
    )
    actual_trusted = dict(quality_audit["trusted_sample_count_by_system"])
    actual_excluded = dict(quality_audit["excluded_hard_fault_count_by_system"])
    actual_guard = dict(quality_audit["excluded_guard_band_count_by_system"])
    actual_kinds = dict(quality_audit["hard_fault_kind_count_by_system"])
    is_v2 = parsed is not None and parsed.get("schema") == OPERATOR_CURATED_CORPUS_SCHEMA_V2
    if parsed is not None:
        expected_actual = {
            "manifest_hash": manifest_hash,
            "summary_sha256": summary_sha256,
            "hard_fault_manifest_sha256": hard_fault_manifest_sha256,
            "trusted_sample_count_by_system": actual_trusted,
            "excluded_hard_fault_count_by_system": actual_excluded,
            "hard_fault_kind_count_by_system": actual_kinds,
        }
        if is_v2:
            if (
                guard_band_manifest_sha256 is None
                or v2_provenance is None
                or v2_provenance_file_sha256 is None
            ):
                raise MeasuredDataAuditError("V2 curation files are missing")
            semantic_provenance = dict(v2_provenance)
            declared_semantic_hash = semantic_provenance.pop("provenance_sha256", None)
            if declared_semantic_hash != stable_json_sha256(semantic_provenance):
                raise MeasuredDataAuditError("V2 provenance semantic hash mismatch")
            if (
                v2_provenance.get("parent_manifest_hash")
                != parsed.get("parent_manifest_hash")
                or v2_provenance.get("parent_summary_sha256")
                != parsed.get("parent_summary_sha256")
                or v2_provenance.get("parent_split_hash")
                != parsed.get("parent_split_hash")
                or v2_provenance.get("parent_hard_fault_manifest_sha256")
                != parsed.get("parent_hard_fault_manifest_sha256")
            ):
                raise MeasuredDataAuditError("V2 parent lineage binding mismatch")
            expected_actual.update({
                "guard_band_manifest_sha256": guard_band_manifest_sha256,
                "v2_provenance_file_sha256": v2_provenance_file_sha256,
                "excluded_guard_band_count_by_system": actual_guard,
            })
        mismatches = [
            name for name, actual in expected_actual.items()
            if parsed.get(name) != actual
        ]
        if mismatches:
            raise MeasuredDataAuditError(
                "operator-curated corpus binding disagrees with measured manifests: "
                f"{mismatches}"
            )
    return {
        "schema": (
            OPERATOR_CURATED_CORPUS_AUDIT_SCHEMA_V2
            if is_v2 else OPERATOR_CURATED_CORPUS_AUDIT_SCHEMA
        ),
        "contract_version": 2 if is_v2 else 1,
        "contract_enforced": parsed is not None,
        "contract_sha256": stable_json_sha256(parsed) if parsed is not None else None,
        "curated_at_local_date": (
            (
                parsed["operator_authorized_at_local_date"]
                if is_v2 else parsed["curated_at_local_date"]
            ) if parsed is not None else None
        ),
        "curation_tool": parsed["curation_tool"] if parsed is not None else None,
        "operator_authorized": (
            bool(parsed["operator_authorized"]) if parsed is not None else False
        ),
        "physical_exclusion_reason": (
            parsed["physical_exclusion_reason"] if parsed is not None else None
        ),
        "classification_recomputed_from_manifest_totals": True,
        "population_scope": "per_system",
        "hard_fault_multiplier": HARD_FAULT_MULTIPLIER,
        "ordinary_noise_retained": True,
        "trusted_manifest_only": True,
        "raw_snapshot_rediscovery_performed": False,
        "trusted_rows_are_final_usable_corpus": (
            bool(parsed["usage_policy"][
                "v2_trusted_rows_are_final_usable_corpus"
                if is_v2 else "trusted_rows_are_final_usable_corpus"
            ])
            if parsed is not None else False
        ),
        "trusted_rows_may_be_reclassified_or_dropped": (
            bool(parsed["usage_policy"][
                "v2_trusted_rows_may_be_reclassified_or_dropped"
                if is_v2 else "trusted_rows_may_be_reclassified_or_dropped"
            ])
            if parsed is not None else False
        ),
        "locked_split_is_evaluation_role_not_quality_exclusion": (
            bool(parsed["usage_policy"][
                "locked_split_is_evaluation_role_not_quality_exclusion"
            ]) if parsed is not None else False
        ),
        "all_trusted_rows_allowed_for_post_selection_refit": (
            bool(parsed["usage_policy"][
                (
                    "all_v2_trusted_rows_allowed_for_post_selection_refit"
                    if is_v2
                    else "all_trusted_rows_allowed_for_post_selection_refit"
                )
            ]) if parsed is not None else False
        ),
        "trusted_sample_count": sum(actual_trusted.values()),
        "excluded_hard_fault_sample_count": sum(actual_excluded.values()),
        "excluded_guard_band_sample_count": sum(actual_guard.values()),
        "trusted_sample_count_by_system": actual_trusted,
        "excluded_hard_fault_count_by_system": actual_excluded,
        "excluded_guard_band_count_by_system": actual_guard,
        "hard_fault_kind_count_by_system": actual_kinds,
        "manifest_hash": manifest_hash,
        "summary_sha256": summary_sha256,
        "hard_fault_manifest_sha256": hard_fault_manifest_sha256,
        "guard_band_manifest_sha256": guard_band_manifest_sha256,
        "v2_provenance_file_sha256": v2_provenance_file_sha256,
        "environment_quality_contract": (
            dict(parsed["environment_quality_contract"])
            if is_v2 else None
        ),
    }


def _system_summary(records: Sequence[MeasuredSampleRecordV1]) -> Dict[str, Any]:
    trusted = [record for record in records if record.quality_status == "trusted"]
    excluded = [record for record in records if record.quality_status != "trusted"]
    rho_count = sum(int(record.rho_available) for record in trusted)
    return {
        "total_snapshot_count": len(records),
        "excluded_snapshot_count": len(excluded),
        "trusted_snapshot_count": len(trusted),
        "distinct_target_count": len(
            {record.state_name or record.exact_target_hash for record in trusted}
        ),
        "distinct_exact_target_hash_count": len(
            {record.exact_target_hash for record in trusted}
        ),
        "rho_available_count": rho_count,
        "clean_rho_count": rho_count,
        "rho_tiers": dict(sorted(Counter(record.rho_tier for record in trusted).items())),
        "command_provenance": dict(
            sorted(Counter(record.command_provenance for record in trusted).items())
        ),
        "quality_status": dict(
            sorted(Counter(record.quality_status for record in records).items())
        ),
        "hard_fault_kind": dict(
            sorted(Counter(record.hard_fault_kind or "none" for record in records).items())
        ),
    }


def _validate_summary(
    summary: Mapping[str, Any],
    all_records: Sequence[MeasuredSampleRecordV1],
    manifest_hash: str,
) -> None:
    if summary.get("schema") not in {
        EXPECTED_SUMMARY_SCHEMA,
        EXPECTED_SUMMARY_SCHEMA_V2,
    }:
        raise ValueError("unsupported measured-data summary schema")
    if summary.get("manifest_hash") != manifest_hash:
        raise MeasuredDataAuditError("measured manifest hash disagrees with summary")
    systems = dict(summary.get("systems") or {})
    if set(systems) != SUPPORTED_SYSTEM_TYPES:
        raise ValueError("measured summary must cover exactly 3qubit and 4qubit")
    for system_type in sorted(SUPPORTED_SYSTEM_TYPES):
        system_records = [
            record for record in all_records if record.system_type == system_type
        ]
        actual = _system_summary(system_records)
        if dict(systems[system_type]) != actual:
            raise MeasuredDataAuditError(
                f"measured summary counters disagree with {system_type} manifests"
            )


def _validate_audit_profile(summary: Mapping[str, Any]) -> Tuple[bool, Tuple[str, ...]]:
    audit = dict(summary.get("audit_profile") or {})
    if not str(audit.get("profile") or "").strip():
        raise MeasuredDataAuditError("V8 requires a named measured-data audit profile")
    checks = dict(audit.get("checks") or {})
    if not checks:
        raise MeasuredDataAuditError("measured-data audit profile has no checks")
    systems = dict(summary["systems"])
    failed: List[str] = []
    for check_name, raw_check in sorted(checks.items()):
        parts = str(check_name).split(".", 1)
        if len(parts) != 2 or parts[0] not in SUPPORTED_SYSTEM_TYPES:
            raise ValueError(f"invalid audit check name: {check_name}")
        system_type, metric_name = parts
        if metric_name not in systems[system_type]:
            raise ValueError(f"audit check references unknown metric: {check_name}")
        check = dict(raw_check or {})
        actual = int(check.get("actual", -1))
        expected = int(check.get("expected", -1))
        if actual != int(systems[system_type][metric_name]):
            raise MeasuredDataAuditError(f"audit check actual value is stale: {check_name}")
        expected_passed = actual == expected
        if bool(check.get("passed")) != expected_passed:
            raise MeasuredDataAuditError(f"audit check pass flag is inconsistent: {check_name}")
        if not expected_passed:
            failed.append(str(check_name))
    declared_passed = bool(audit.get("passed"))
    actual_passed = not failed
    if declared_passed != actual_passed:
        raise MeasuredDataAuditError("audit-profile passed flag is inconsistent with its checks")
    return actual_passed, tuple(failed)


def _validate_splits(
    split_payload: Mapping[str, Any],
    split_audit: Mapping[str, Any],
    trusted_records: Sequence[MeasuredSampleRecordV1],
) -> Tuple[str, Dict[str, str]]:
    if split_payload.get("schema") != EXPECTED_SPLIT_SCHEMA:
        raise ValueError("unsupported measured split schema")
    assignments = {
        str(sample_id): str(split_name)
        for sample_id, split_name in dict(split_payload.get("assignments") or {}).items()
    }
    trusted_ids = {record.sample_id for record in trusted_records}
    if set(assignments) != trusted_ids or any(not name for name in assignments.values()):
        raise MeasuredDataAuditError("split assignments do not exactly cover trusted rows")
    hash_payload = {
        "schema": split_payload["schema"],
        "seed": split_payload.get("seed"),
        "assignments": assignments,
    }
    split_hash = stable_json_sha256(hash_payload)
    if split_payload.get("split_hash") != split_hash:
        raise MeasuredDataAuditError("measured split hash mismatch")
    embedded_audit = dict(split_payload.get("audit") or {})
    if embedded_audit != dict(split_audit):
        raise MeasuredDataAuditError("embedded and standalone split audits disagree")
    checks = dict(split_audit.get("checks") or {})
    if not checks or not bool(split_audit.get("passed")) or not all(
        value is True for value in checks.values()
    ):
        raise MeasuredDataAuditError("split audit failed and cannot be waived")
    split_details = dict(split_payload.get("splits") or {})
    expected_by_split: Dict[str, List[str]] = {}
    for sample_id, split_name in assignments.items():
        expected_by_split.setdefault(split_name, []).append(sample_id)
    if set(split_details) != set(expected_by_split):
        raise MeasuredDataAuditError("split detail names disagree with assignments")
    for split_name, expected_ids in expected_by_split.items():
        detail = dict(split_details[split_name])
        actual_ids = list(detail.get("sample_ids") or ())
        if actual_ids != sorted(expected_ids) or int(detail.get("sample_count", -1)) != len(expected_ids):
            raise MeasuredDataAuditError(f"split details disagree for {split_name}")
    return split_hash, assignments


def _validate_command_provenance_summary(
    summary: Mapping[str, Any],
    trusted_records: Sequence[MeasuredSampleRecordV1],
) -> None:
    expected = {
        "schema": EXPECTED_COMMAND_PROVENANCE_SCHEMA,
        "distribution": dict(
            sorted(Counter(record.command_provenance for record in trusted_records).items())
        ),
        "command_supervision_sample_count": sum(
            int(record.command_supervision_mask) for record in trusted_records
        ),
        "target_only_sample_count": sum(
            record.command_provenance == "target_only_approximation"
            for record in trusted_records
        ),
        "missing_sample_count": sum(
            record.command_provenance == "missing" for record in trusted_records
        ),
    }
    if dict(summary) != expected:
        raise MeasuredDataAuditError("command-provenance summary disagrees with manifests")


def _build_measured_row(record: MeasuredSampleRecordV1, split_name: str) -> V8MeasuredRowV1:
    duration_known = float(record.duration_s) > 0.0 and float(record.gate_exposure) > 0.0
    calibration_hash = record.calibration_snapshot_hash
    rho_hash = record.rho_hash if record.rho_available else None
    rho_kind = record.rho_kind if record.rho_available else None
    return V8MeasuredRowV1(
        record=record,
        split_name=split_name,
        exposure_s=float(record.duration_s) if duration_known else None,
        exposure_known_mask=int(duration_known),
        calibration_hash=calibration_hash,
        calibration_known_mask=int(calibration_hash is not None),
        rho_hash=rho_hash,
        rho_kind=rho_kind,
        rho_available_mask=int(record.rho_available),
        global_multi_click_upper_bounds=(
            int(record.multi2_count),
            int(record.multi3_count),
            int(record.multi4_count),
        ),
        action_evidence_kind=(
            "none_target_only_snapshot"
            if record.command_provenance == "target_only_approximation"
            else "none_unpaired_snapshot"
        ),
    )


def _audit_counts(
    trusted_records: Sequence[MeasuredSampleRecordV1],
    hard_fault_records: Sequence[MeasuredSampleRecordV1],
) -> Dict[str, Any]:
    trusted_count: Dict[str, int] = {}
    rho_count: Dict[str, int] = {}
    known_count: Dict[str, int] = {}
    unknown_count: Dict[str, int] = {}
    excluded_rho: Dict[str, Dict[str, int]] = {}
    real_action_count: Dict[str, int] = {}
    for system_type in sorted(SUPPORTED_SYSTEM_TYPES):
        rows = [record for record in trusted_records if record.system_type == system_type]
        known = sum(
            float(record.duration_s) > 0.0 and float(record.gate_exposure) > 0.0
            for record in rows
        )
        trusted_count[system_type] = len(rows)
        rho_count[system_type] = sum(int(record.rho_available) for record in rows)
        known_count[system_type] = int(known)
        unknown_count[system_type] = len(rows) - int(known)
        # Snapshot manifests contain state observations, not executed
        # before/action/after evidence.  Signed probes and paired transitions
        # enter only through their independent hash-bound provenance loaders.
        real_action_count[system_type] = 0
        excluded_rho[system_type] = dict(
            sorted(
                Counter(
                    record.hard_fault_kind or "unknown"
                    for record in hard_fault_records
                    if record.system_type == system_type and record.rho_available
                ).items()
            )
        )
    return {
        "trusted": trusted_count,
        "rho": rho_count,
        "exposure_known": known_count,
        "exposure_unknown": unknown_count,
        "real_action_supervision": real_action_count,
        "excluded_rho_by_fault": excluded_rho,
    }


__all__ = [
    "OPERATOR_CURATED_CORPUS_AUDIT_SCHEMA",
    "OPERATOR_CURATED_CORPUS_AUDIT_SCHEMA_V2",
    "OPERATOR_CURATED_CORPUS_SCHEMA",
    "OPERATOR_CURATED_CORPUS_SCHEMA_V2",
    "V8_MEASURED_BUNDLE_SCHEMA",
    "V8_MEASURED_ROW_SCHEMA",
    "V8MeasuredDataBundleV1",
    "V8MeasuredRowV1",
    "declare_locked_final_test",
    "load_history_transition_provenance",
    "load_v8_measured_data_bundle",
    "source_tree_sha256",
    "validate_operator_curated_corpus_contract",
    "validate_history_transition_binding",
]
