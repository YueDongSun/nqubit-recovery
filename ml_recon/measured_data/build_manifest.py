"""Build deterministic measured-data manifests without mutating snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .command_provenance import resolve_command_provenance
from .contracts import MeasuredSampleRecordV1
from .quality import QualityDiagnostics, classify_population
from .splits import build_measured_splits
from .target_identity import intervention_identity, safe_target_payload, target_hash


SNAPSHOT_SUFFIXES = {"3qubit": "_3qubit", "4qubit": "_4qubit"}
COUNT_ARRAY_NAMES = ("signal_bins", "idler_bins", "coin_bins")
RAW_COUNT_ARRAY_NAMES = ("signal_bins_raw", "idler_bins_raw", "coin_bins_raw")
RHO_PRIORITY = ("posterior_mean", "calibrated", "raw")
AUDIT_PROFILES: Dict[str, Dict[str, Dict[str, int]]] = {
    "final_20260805": {
        "4qubit": {
            "total_snapshot_count": 2086,
            "excluded_snapshot_count": 88,
            "trusted_snapshot_count": 1998,
            "distinct_target_count": 978,
            "rho_available_count": 1998,
        },
        "3qubit": {
            "total_snapshot_count": 1153,
            "excluded_snapshot_count": 49,
            "trusted_snapshot_count": 1104,
            "distinct_target_count": 992,
            "clean_rho_count": 15,
        },
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _array_total(path: Path) -> int:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    return int(np.asarray(values, dtype=np.int64).sum(dtype=np.int64))


def _array_hashes(
    snapshot_dir: Path,
    names: Iterable[str],
    manifest: Mapping[str, Any],
    *,
    strict: bool,
) -> Dict[str, str]:
    declared = dict(manifest.get("arrays") or {})
    hashes: Dict[str, str] = {}
    for name in names:
        path = snapshot_dir / f"{name}.npy"
        if not path.is_file():
            if strict and name in COUNT_ARRAY_NAMES:
                raise FileNotFoundError(f"required count array missing: {path}")
            continue
        actual_hash = _sha256_file(path)
        expected_hash = (declared.get(name) or {}).get("sha256")
        if strict and expected_hash and actual_hash != str(expected_hash):
            raise ValueError(f"snapshot array hash mismatch: {path}")
        hashes[name] = actual_hash
    return hashes


def _target_payload(manifest: Mapping[str, Any], metadata: Mapping[str, Any]) -> Dict[str, Any]:
    target = manifest.get("target_state_params")
    if not isinstance(target, Mapping) or not target:
        feedback_context = metadata.get("feedback_context") or {}
        target = feedback_context.get("target_state_params") if isinstance(feedback_context, Mapping) else {}
    return dict(target or {})


def _automation_context(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    feedback_context = metadata.get("feedback_context") or {}
    if not isinstance(feedback_context, Mapping):
        return {}
    automation = feedback_context.get("automation") or {}
    return dict(automation) if isinstance(automation, Mapping) else {}


def _basename(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value)
    return PureWindowsPath(text).name or Path(text).name or text


def _time_block(timestamp: float, automation_run_id: Optional[str]) -> str:
    day = int(max(timestamp, 0.0) // 86400)
    return f"{automation_run_id or 'untracked'}:{day}"


def _rho_identity(snapshot_dir: Path, manifest: Mapping[str, Any]) -> tuple[bool, str, Optional[str]]:
    declared = dict(manifest.get("arrays") or {})
    for kind in RHO_PRIORITY:
        path = snapshot_dir / f"rho_{kind}.npy"
        if path.is_file():
            actual_hash = _sha256_file(path)
            expected_hash = (declared.get(f"rho_{kind}") or {}).get("sha256")
            if expected_hash and actual_hash != str(expected_hash):
                raise ValueError(f"rho hash mismatch: {path}")
            return True, kind, actual_hash
    return False, "missing", None


def _probe_identity(manifest: Mapping[str, Any], target: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    probe = manifest.get("probe") or {}
    schedule = manifest.get("schedule") or {}
    anchor = (
        probe.get("anchor_id") if isinstance(probe, Mapping) else None
    ) or (
        schedule.get("anchor_id") if isinstance(schedule, Mapping) else None
    ) or target.get("anchor_id")
    pair = (
        probe.get("pair_split_key") if isinstance(probe, Mapping) else None
    ) or (
        schedule.get("pair_split_key") if isinstance(schedule, Mapping) else None
    ) or target.get("pair_split_key")
    return (None if anchor in (None, "") else str(anchor), None if pair in (None, "") else str(pair))


def _candidate_snapshot_dirs(data_root: Path) -> List[Path]:
    candidates = []
    for path in data_root.iterdir() if data_root.is_dir() else []:
        if path.is_dir() and any(path.name.endswith(suffix) for suffix in SNAPSHOT_SUFFIXES.values()):
            candidates.append(path)
    return sorted(candidates, key=lambda item: item.name)


def _raw_snapshot(snapshot_dir: Path, *, strict: bool) -> Dict[str, Any]:
    metadata_path = snapshot_dir / "metadata.json"
    manifest_path = snapshot_dir / "manifest.json"
    metadata = _read_json(metadata_path)
    manifest = _read_json(manifest_path)
    system_type = str(manifest.get("system_type") or metadata.get("system_type") or "")
    if system_type not in SNAPSHOT_SUFFIXES:
        raise ValueError(f"snapshot has unsupported system_type: {snapshot_dir}")
    count_hashes = _array_hashes(snapshot_dir, COUNT_ARRAY_NAMES, manifest, strict=strict)
    raw_hashes = _array_hashes(snapshot_dir, RAW_COUNT_ARRAY_NAMES, manifest, strict=False)
    if len(count_hashes) != len(COUNT_ARRAY_NAMES):
        raise ValueError(f"snapshot lacks complete count arrays: {snapshot_dir}")
    totals = {
        name.removesuffix("_bins") + "_total": _array_total(snapshot_dir / f"{name}.npy")
        for name in COUNT_ARRAY_NAMES
    }
    target = _target_payload(manifest, metadata)
    automation = _automation_context(metadata)
    timestamp = float(metadata.get("timestamp") or manifest.get("ended_at") or 0.0)
    started_at = float(manifest.get("started_at") or timestamp)
    ended_at = float(manifest.get("ended_at") or timestamp)
    duration_s = float(manifest.get("duration_s") or max(0.0, ended_at - started_at))
    state_name = str(target.get("name") or automation.get("state_name") or "")
    state_index = manifest.get("state_index", automation.get("state_index"))
    plan_reference = str(
        automation.get("source_plan_path")
        or automation.get("source_plan_description")
        or manifest.get("plan_id")
        or ""
    )
    plan_id = manifest.get("plan_id")
    if not plan_id and plan_reference:
        plan_id = _stable_hash({"plan_reference": plan_reference})[:16]
    campaign_id = manifest.get("campaign_id") or automation.get("campaign_id")
    automation_run_id = _basename(automation.get("run_dir"))
    rho_available, rho_kind, rho_hash = _rho_identity(snapshot_dir, manifest)
    provenance = resolve_command_provenance(snapshot_dir, manifest, metadata)
    target_payload = safe_target_payload(target) if target.get("amplitudes") else {}
    exact_hash = target_hash(target_payload) if target_payload else ""
    if target_payload:
        base_family, intervention_id, intervention_delta = intervention_identity(
            system_type=system_type,
            target_state_params=target_payload,
            plan_reference=plan_reference,
        )
    else:
        base_family, intervention_id, intervention_delta = "", None, []
    count_summary = dict(manifest.get("count_summary") or {})
    acquisition = dict(manifest.get("acquisition") or {})
    multi_observation = dict(manifest.get("multi_click_observation") or {})
    probe_anchor_id, probe_pair_id = _probe_identity(manifest, target)
    manifest_source = manifest_path if manifest else metadata_path
    return {
        "snapshot_dir": snapshot_dir,
        "manifest": manifest,
        "metadata": metadata,
        "system_type": system_type,
        "snapshot_manifest_schema": (
            f"snapshot_manifest_v{manifest.get('schema_version', 0)}" if manifest else "legacy_metadata_v1"
        ),
        "snapshot_manifest_hash": _sha256_file(manifest_source),
        "target": target_payload,
        "exact_target_hash": exact_hash,
        "base_state_family_id": base_family,
        "intervention_id": intervention_id,
        "intervention_delta": intervention_delta,
        "state_name": state_name,
        "state_index": None if state_index in (None, "") else int(state_index),
        "plan_id": None if plan_id in (None, "") else str(plan_id),
        "campaign_id": None if campaign_id in (None, "") else str(campaign_id),
        "automation_run_id": automation_run_id,
        "timestamp": timestamp,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_s": duration_s,
        "count_hashes": count_hashes,
        "raw_hashes": raw_hashes,
        **totals,
        "read_count": int(acquisition.get("read_count") or metadata.get("read_count") or 0),
        "write_count": int(acquisition.get("write_count") or metadata.get("write_count") or 0),
        "gate_exposure": float(acquisition.get("gate_exposure") or duration_s),
        "multi_click_scope": str(multi_observation.get("scope") or "global"),
        "multi2_count": int(count_summary.get("multi2_count") or metadata.get("multi2_count") or 0),
        "multi3_count": int(count_summary.get("multi3_count") or metadata.get("multi3_count") or 0),
        "multi4_count": int(count_summary.get("multi4_count") or metadata.get("multi4_count") or 0),
        "multi2_pairs": {
            str(key): int(value)
            for key, value in dict(count_summary.get("multi2_pairs") or metadata.get("multi2_pairs") or {}).items()
        },
        "rho_available": rho_available,
        "rho_kind": rho_kind,
        "rho_hash": rho_hash,
        "calibration_snapshot_hash": manifest.get("calibration_snapshot_hash"),
        "provenance": provenance,
        "probe_anchor_id": probe_anchor_id,
        "probe_pair_id": probe_pair_id,
    }


def _make_record(
    raw: Mapping[str, Any],
    diagnostics: QualityDiagnostics,
    *,
    data_root: Path,
) -> MeasuredSampleRecordV1:
    snapshot_dir = Path(raw["snapshot_dir"])
    missing_target = not raw["exact_target_hash"]
    quality_status = diagnostics.quality_status
    exclusion_reason = diagnostics.exclusion_reason
    if missing_target and quality_status == "trusted":
        quality_status = "excluded_missing_target"
        exclusion_reason = "missing_target_state_params"
    rho_available = bool(raw["rho_available"])
    sample_id = str(
        raw["manifest"].get("sample_id")
        or raw["metadata"].get("sample_id")
        or _stable_hash({
            "snapshot": snapshot_dir.name,
            "manifest_hash": raw["snapshot_manifest_hash"],
        })
    )
    provenance = raw["provenance"]
    return MeasuredSampleRecordV1(
        sample_id=sample_id,
        system_type=str(raw["system_type"]),
        snapshot_path_relative=snapshot_dir.relative_to(data_root).as_posix(),
        snapshot_manifest_schema=str(raw["snapshot_manifest_schema"]),
        snapshot_manifest_hash=str(raw["snapshot_manifest_hash"]),
        exact_target_state_params=dict(raw["target"]),
        exact_target_hash=str(raw["exact_target_hash"]),
        base_state_family_id=str(raw["base_state_family_id"] or raw["exact_target_hash"] or sample_id),
        intervention_id=raw["intervention_id"],
        intervention_delta=list(raw["intervention_delta"]),
        state_name=str(raw["state_name"]),
        state_index=raw["state_index"],
        plan_id=raw["plan_id"],
        campaign_id=raw["campaign_id"],
        automation_run_id=raw["automation_run_id"],
        acquisition_run_id=snapshot_dir.name,
        timestamp=float(raw["timestamp"]),
        started_at=float(raw["started_at"]),
        ended_at=float(raw["ended_at"]),
        duration_s=float(raw["duration_s"]),
        time_block_id=_time_block(float(raw["timestamp"]), raw["automation_run_id"]),
        repeat_group_id=str(raw["exact_target_hash"] or sample_id),
        repeat_index=0,
        repeat_count=1,
        signal_total=int(raw["signal_total"]),
        idler_total=int(raw["idler_total"]),
        coin_total=int(raw["coin_total"]),
        count_array_hashes=dict(raw["count_hashes"]),
        raw_count_array_hashes=dict(raw["raw_hashes"]),
        read_count=int(raw["read_count"]),
        write_count=int(raw["write_count"]),
        gate_exposure=float(raw["gate_exposure"]),
        multi_click_scope=str(raw["multi_click_scope"]),
        multi2_count=int(raw["multi2_count"]),
        multi3_count=int(raw["multi3_count"]),
        multi4_count=int(raw["multi4_count"]),
        multi2_pairs=dict(raw["multi2_pairs"]),
        rho_available=rho_available,
        rho_kind=str(raw["rho_kind"]),
        rho_hash=raw["rho_hash"],
        rho_tier="A_clean_rho" if rho_available else "B_count_only",
        rho_branch_mask=int(rho_available),
        rho_loss_mask=int(rho_available),
        operation_registry_hash=provenance.operation_registry_hash,
        matrix_snapshot_hash=provenance.matrix_snapshot_hash,
        calibration_snapshot_hash=(
            None if raw["calibration_snapshot_hash"] in (None, "")
            else str(raw["calibration_snapshot_hash"])
        ),
        waveform_hash=provenance.waveform_hash,
        command_table_id=provenance.command_table_id,
        command_table_hash=provenance.command_table_hash,
        command_provenance=provenance.provenance,
        command_supervision_mask=provenance.supervision_mask,
        quality_status=quality_status,
        exclusion_reason=exclusion_reason,
        hard_fault_kind=diagnostics.hard_fault_kind,
        signal_robust_z_score=diagnostics.signal_robust_z_score,
        idler_robust_z_score=diagnostics.idler_robust_z_score,
        saturation_flag=diagnostics.saturation_flag,
        low_low_collapse_flag=diagnostics.low_low_collapse_flag,
        signal_high_idler_low_flag=diagnostics.signal_high_idler_low_flag,
        sample_weight=1.0,
        probe_anchor_id=raw["probe_anchor_id"],
        probe_pair_id=raw["probe_pair_id"],
    )


def _attach_repeat_metadata(records: Sequence[MeasuredSampleRecordV1]) -> List[MeasuredSampleRecordV1]:
    grouped: Dict[str, List[MeasuredSampleRecordV1]] = defaultdict(list)
    for record in records:
        grouped[record.repeat_group_id].append(record)
    updated: List[MeasuredSampleRecordV1] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: (item.timestamp, item.sample_id))
        repeat_count = len(ordered)
        for repeat_index, record in enumerate(ordered):
            updated.append(replace(
                record,
                repeat_index=repeat_index,
                repeat_count=repeat_count,
                sample_weight=1.0 / float(repeat_count),
            ))
    return sorted(updated, key=lambda item: (item.system_type, item.snapshot_path_relative))


def build_records(data_root: str | Path, *, strict: bool = False) -> List[MeasuredSampleRecordV1]:
    root = Path(data_root).resolve()
    raw_records = [_raw_snapshot(path, strict=strict) for path in _candidate_snapshot_dirs(root)]
    records: List[MeasuredSampleRecordV1] = []
    for system_type in SNAPSHOT_SUFFIXES:
        system_raw = [item for item in raw_records if item["system_type"] == system_type]
        diagnostics = classify_population(
            (int(item["signal_total"]), int(item["idler_total"])) for item in system_raw
        )
        records.extend(
            _make_record(item, diagnostic, data_root=root)
            for item, diagnostic in zip(system_raw, diagnostics)
        )
    return _attach_repeat_metadata(records)


def _state_resolution(records: Iterable[MeasuredSampleRecordV1]) -> Dict[str, Any]:
    grouped: Dict[str, List[MeasuredSampleRecordV1]] = defaultdict(list)
    for record in records:
        if record.quality_status == "trusted":
            # The legacy audit's "state resolved" count is the number of
            # recorded acquisition-state identities.  Exact physical hashes
            # remain separate and are what split/repeat logic consumes.
            grouped[record.state_name or record.exact_target_hash].append(record)
    return {
        "schema": "state_resolved_v1",
        "distinct_target_count": len(grouped),
        "targets": [
            {
                "state_identity": target,
                "exact_target_hashes": sorted({item.exact_target_hash for item in group}),
                "base_state_family_id": group[0].base_state_family_id,
                "sample_ids": sorted(item.sample_id for item in group),
                "sample_count": len(group),
                "representative_sample_id": min(
                    group,
                    key=lambda item: (item.hard_fault_kind is not None, -item.idler_total, item.sample_id),
                ).sample_id,
            }
            for target, group in sorted(grouped.items())
        ],
    }


def _summary(records: Sequence[MeasuredSampleRecordV1]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"schema": "measured_data_summary_v1", "systems": {}}
    for system_type in SNAPSHOT_SUFFIXES:
        system = [item for item in records if item.system_type == system_type]
        trusted = [item for item in system if item.quality_status == "trusted"]
        excluded = [item for item in system if item.quality_status != "trusted"]
        rho_count = sum(item.rho_available for item in trusted)
        values = {
            "total_snapshot_count": len(system),
            "excluded_snapshot_count": len(excluded),
            "trusted_snapshot_count": len(trusted),
            "distinct_target_count": len({item.state_name or item.exact_target_hash for item in trusted}),
            "distinct_exact_target_hash_count": len({item.exact_target_hash for item in trusted}),
            "rho_available_count": rho_count,
            "clean_rho_count": rho_count,
            "rho_tiers": dict(sorted(Counter(item.rho_tier for item in trusted).items())),
            "command_provenance": dict(sorted(Counter(item.command_provenance for item in trusted).items())),
            "quality_status": dict(sorted(Counter(item.quality_status for item in system).items())),
            "hard_fault_kind": dict(sorted(Counter(item.hard_fault_kind or "none" for item in system).items())),
        }
        result["systems"][system_type] = values
    result["manifest_hash"] = _stable_hash([
        item.to_json_dict() for item in sorted(records, key=lambda record: record.sample_id)
    ])
    return result


def _audit_profile(summary: Mapping[str, Any], profile_name: Optional[str]) -> Dict[str, Any]:
    if profile_name is None:
        return {"profile": None, "passed": True, "checks": {}}
    if profile_name not in AUDIT_PROFILES:
        raise ValueError(f"unknown audit profile: {profile_name}")
    checks: Dict[str, Any] = {}
    passed = True
    for system_type, expectations in AUDIT_PROFILES[profile_name].items():
        actual_values = dict(summary["systems"][system_type])
        for key, expected in expectations.items():
            actual = int(actual_values.get(key, -1))
            check_name = f"{system_type}.{key}"
            checks[check_name] = {
                "expected": int(expected),
                "actual": actual,
                "passed": actual == int(expected),
            }
            passed = passed and actual == int(expected)
    return {"profile": profile_name, "passed": passed, "checks": checks}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[MeasuredSampleRecordV1]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record.to_json_dict(), sort_keys=True, separators=(",", ":"))
        for record in sorted(records, key=lambda item: item.snapshot_path_relative)
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_manifest_outputs(
    records: Sequence[MeasuredSampleRecordV1],
    output_dir: str | Path,
    *,
    audit_profile: Optional[str] = None,
) -> Dict[str, Any]:
    destination = Path(output_dir)
    trusted = [item for item in records if item.quality_status == "trusted"]
    hard_faults = [item for item in records if item.hard_fault_kind is not None]
    summary = _summary(records)
    summary["audit_profile"] = _audit_profile(summary, audit_profile)
    splits = build_measured_splits(trusted)
    for system_type, suffix in (("3qubit", "3q"), ("4qubit", "4q")):
        system_trusted = [item for item in trusted if item.system_type == system_type]
        _write_jsonl(destination / f"trusted_samples_{suffix}.jsonl", system_trusted)
        _write_json(destination / f"state_resolved_{suffix}.json", _state_resolution(system_trusted))
    _write_jsonl(destination / "hard_fault_samples.jsonl", hard_faults)
    _write_json(destination / "measured_data_summary.json", summary)
    _write_json(destination / "measured_splits_v1.json", splits)
    _write_json(destination / "split_audit.json", splits["audit"])
    provenance_summary = {
        "schema": "command_provenance_summary_v1",
        "distribution": dict(sorted(Counter(item.command_provenance for item in trusted).items())),
        "command_supervision_sample_count": sum(item.command_supervision_mask for item in trusted),
        "target_only_sample_count": sum(
            item.command_provenance == "target_only_approximation" for item in trusted
        ),
        "missing_sample_count": sum(item.command_provenance == "missing" for item in trusted),
    }
    _write_json(destination / "command_provenance_summary.json", provenance_summary)
    return {
        "summary": summary,
        "splits": splits,
        "command_provenance": provenance_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-profile", choices=sorted(AUDIT_PROFILES))
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = build_records(args.data_root, strict=bool(args.strict))
    result = write_manifest_outputs(records, args.output_dir, audit_profile=args.audit_profile)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0 if result["summary"]["audit_profile"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_PROFILES",
    "build_records",
    "main",
    "write_manifest_outputs",
]
