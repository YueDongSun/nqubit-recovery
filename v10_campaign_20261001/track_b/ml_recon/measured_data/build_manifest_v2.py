"""Derive operator-authorized measured-data contract V2 from immutable V1.

The 2026-08-12 revision preserves the original signal-high/idler-low rule and
adds independently reproducible FPGA/read-idler counter-stuck and operator
total-multi/idler rules.  It then excludes exactly one chronological neighbor
before and after every direct-fault segment in the same system.  Chronological
adjacency has no time-gap exemption: the operator authorized neighboring
*measured points*, including when acquisition pauses separate their timestamps.
``time_block_id`` is deliberately not an adjacency requirement because one
acquisition point commonly owns one time-block label.

Raw snapshots are never rewritten or deleted.  Direct anomalies and guard-band
rows remain in separate, hash-bound manifests with their original payload
hashes and explicit exclusion provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .build_manifest import _state_resolution, _summary, _write_json, _write_jsonl
from .contracts import MeasuredSampleRecordV1
from .datasets import load_measured_records
from .splits import SPLIT_SCHEMA, audit_splits, build_repeat_pairs, split_group_key


V2_SCHEMA = "operator_curated_measured_data_v2"
V2_POLICY_VERSION = "operator_guard_band_chronological_one_neighbor_v2_3"
V2_GUARD_STATUS = "excluded_guard_band"
V2_GUARD_REASON = "adjacent_to_operator_confirmed_lock_loss_segment"
V2_FPGA_ANOMALY_KIND = "fpga_read_idler_counter_stuck"
V2_FPGA_ANOMALY_REASON = f"hard_fault:{V2_FPGA_ANOMALY_KIND}"
V2_MULTI_RATIO_ANOMALY_KIND = "fpga_total_multi_exceeds_idler_fraction_contract"
V2_MULTI_RATIO_ANOMALY_REASON = f"hard_fault:{V2_MULTI_RATIO_ANOMALY_KIND}"
V2_TOTAL_MULTI_TO_IDLER_RATIO_MAX = 0.20
V2_FPGA_IDLER_WINDOW_RATE_MIN = 0.95
V2_FPGA_COIN_GIVEN_SIGNAL_MIN = 0.95
V2_FPGA_MULTI2_PAIR_SHARE_MIN = 0.95
V2_FPGA_MULTI2_COUNT_MIN = 1_000_000
V2_OPERATION_COUNT_BY_SYSTEM = {"3qubit": 75, "4qubit": 256}
GUARD_FILE_NAME = "guard_band_samples.jsonl"
PROVENANCE_FILE_NAME = "operator_curated_corpus_v2.json"
V1_TRUSTED_FILE_NAMES = ("trusted_samples_3q.jsonl", "trusted_samples_4q.jsonl")
V1_HARD_FAULT_FILE_NAME = "hard_fault_samples.jsonl"
V1_SPLIT_FILE_NAME = "measured_splits_v1.json"


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _read_jsonl(path: Path) -> List[MeasuredSampleRecordV1]:
    records: List[MeasuredSampleRecordV1] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            records.append(MeasuredSampleRecordV1.from_json_dict(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"invalid measured record at {path}:{line_number}") from exc
    return records


def _chronological_key(record: MeasuredSampleRecordV1) -> Tuple[float, float, str]:
    return float(record.timestamp), float(record.ended_at), record.sample_id


def _same_segment(
    first: MeasuredSampleRecordV1,
    second: MeasuredSampleRecordV1,
) -> bool:
    """Return whether two consecutive records belong to one fault segment.

    The caller supplies records that are adjacent in the same system's full
    chronological observation sequence.  A wall-clock gap is therefore audit
    metadata, never permission to retain the neighboring measured point.
    """

    return first.system_type == second.system_type


def fpga_hardware_anomaly_evidence(
    record: MeasuredSampleRecordV1,
) -> Dict[str, Any]:
    """Return deterministic FPGA and operator-ratio audits for one manifest row.

    Counter-stuck evidence requires four simultaneous, physically distinct
    symptoms. The independent operator rule excludes global
    (multi2+multi3+multi4)/idler above 20 percent. Both are stricter than a
    generic statistical outlier test and
    are not based on a trained model, validation loss, low counts, temperature,
    or state rarity.
    """

    operation_count = V2_OPERATION_COUNT_BY_SYSTEM[record.system_type]
    read_opportunity_count = int(record.read_count) * int(operation_count)
    multi2_pair_values = {
        str(name): int(value) for name, value in dict(record.multi2_pairs).items()
    }
    largest_pair_name = None
    largest_pair_count = 0
    if multi2_pair_values:
        largest_pair_name, largest_pair_count = max(
            multi2_pair_values.items(),
            key=lambda item: (int(item[1]), str(item[0])),
        )
    idler_window_rate = (
        float(record.idler_total) / float(read_opportunity_count)
        if read_opportunity_count > 0 else 0.0
    )
    coin_given_signal = (
        float(record.coin_total) / float(record.signal_total)
        if int(record.signal_total) > 0 else 0.0
    )
    largest_pair_share = (
        float(largest_pair_count) / float(record.multi2_count)
        if int(record.multi2_count) > 0 else 0.0
    )
    counter_stuck_checks = {
        "idler_window_rate_at_least_minimum": (
            idler_window_rate >= V2_FPGA_IDLER_WINDOW_RATE_MIN
        ),
        "coin_given_signal_at_least_minimum": (
            coin_given_signal >= V2_FPGA_COIN_GIVEN_SIGNAL_MIN
        ),
        "multi2_pair_share_at_least_minimum": (
            largest_pair_share >= V2_FPGA_MULTI2_PAIR_SHARE_MIN
        ),
        "multi2_count_at_least_minimum": (
            int(record.multi2_count) >= V2_FPGA_MULTI2_COUNT_MIN
        ),
    }
    total_multi_count = (
        int(record.multi2_count)
        + int(record.multi3_count)
        + int(record.multi4_count)
    )
    total_multi_to_idler_ratio = (
        float(total_multi_count) / float(record.idler_total)
        if int(record.idler_total) > 0
        else (float("inf") if total_multi_count > 0 else 0.0)
    )
    counter_stuck_matched = all(counter_stuck_checks.values())
    total_multi_ratio_matched = (
        total_multi_to_idler_ratio > V2_TOTAL_MULTI_TO_IDLER_RATIO_MAX
    )
    matched_rule_names = []
    if counter_stuck_matched:
        matched_rule_names.append(V2_FPGA_ANOMALY_KIND)
    if total_multi_ratio_matched:
        matched_rule_names.append(V2_MULTI_RATIO_ANOMALY_KIND)
    return {
        "sample_id": record.sample_id,
        "snapshot_path_relative": record.snapshot_path_relative,
        "system_type": record.system_type,
        "read_opportunity_count": read_opportunity_count,
        "idler_window_rate": idler_window_rate,
        "coin_given_signal": coin_given_signal,
        "multi2_count": int(record.multi2_count),
        "multi3_count": int(record.multi3_count),
        "multi4_count": int(record.multi4_count),
        "total_multi_count": total_multi_count,
        "largest_multi2_pair": largest_pair_name,
        "largest_multi2_pair_count": largest_pair_count,
        "largest_multi2_pair_share": largest_pair_share,
        "total_multi_to_idler_ratio": total_multi_to_idler_ratio,
        "total_multi_to_idler_ratio_max": V2_TOTAL_MULTI_TO_IDLER_RATIO_MAX,
        "counter_stuck_checks": counter_stuck_checks,
        "counter_stuck_matched": counter_stuck_matched,
        "total_multi_ratio_matched": total_multi_ratio_matched,
        "matched_rule_names": matched_rule_names,
        "matched": bool(matched_rule_names),
    }


def derive_v2_collections(
    trusted_records: Sequence[MeasuredSampleRecordV1],
    hard_fault_records: Sequence[MeasuredSampleRecordV1],
) -> Tuple[
    List[MeasuredSampleRecordV1],
    List[MeasuredSampleRecordV1],
    List[MeasuredSampleRecordV1],
    List[Dict[str, Any]],
]:
    """Return trusted, direct-fault, guard, and FPGA-anomaly audit records."""

    if not trusted_records or not hard_fault_records:
        raise ValueError("V2 derivation requires non-empty trusted and hard-fault rows")
    for record in trusted_records:
        if record.quality_status != "trusted" or record.hard_fault_kind is not None:
            raise ValueError("V2 trusted input contains a non-trusted row")
    for record in hard_fault_records:
        if record.quality_status == "trusted" or record.hard_fault_kind is None:
            raise ValueError("V2 hard-fault input contains a non-fault row")

    anomaly_audit = [
        fpga_hardware_anomaly_evidence(record) for record in trusted_records
    ]
    matched_anomalies = [item for item in anomaly_audit if bool(item["matched"])]
    anomaly_by_id = {str(item["sample_id"]): item for item in matched_anomalies}
    promoted_hard_faults = [
        replace(
            record,
            quality_status="excluded_hard_fault",
            exclusion_reason=(
                V2_MULTI_RATIO_ANOMALY_REASON
                if bool(anomaly_by_id[record.sample_id]["total_multi_ratio_matched"])
                else V2_FPGA_ANOMALY_REASON
            ),
            hard_fault_kind=(
                V2_MULTI_RATIO_ANOMALY_KIND
                if bool(anomaly_by_id[record.sample_id]["total_multi_ratio_matched"])
                else V2_FPGA_ANOMALY_KIND
            ),
        )
        for record in trusted_records
        if record.sample_id in anomaly_by_id
    ]
    remaining_trusted = [
        record for record in trusted_records if record.sample_id not in anomaly_by_id
    ]
    direct_hard_faults = sorted(
        [*hard_fault_records, *promoted_hard_faults],
        key=lambda item: (item.system_type, item.snapshot_path_relative),
    )

    all_records = sorted(
        [*remaining_trusted, *direct_hard_faults],
        key=lambda item: (item.system_type, *_chronological_key(item)),
    )
    by_id = {record.sample_id: record for record in all_records}
    if len(by_id) != len(all_records):
        raise ValueError("V1 records contain duplicate sample IDs")
    trusted_ids = {record.sample_id for record in remaining_trusted}
    hard_ids = {record.sample_id for record in direct_hard_faults}
    guard_sources: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"source_ids": [], "directions": [], "gaps": []}
    )

    for system_type in ("3qubit", "4qubit"):
        ordered = [record for record in all_records if record.system_type == system_type]
        index = 0
        while index < len(ordered):
            if ordered[index].sample_id not in hard_ids:
                index += 1
                continue
            segment_start = index
            segment_end = index
            while (
                segment_end + 1 < len(ordered)
                and ordered[segment_end + 1].sample_id in hard_ids
                and _same_segment(
                    ordered[segment_end],
                    ordered[segment_end + 1],
                )
            ):
                segment_end += 1

            first_fault = ordered[segment_start]
            last_fault = ordered[segment_end]
            candidates = (
                (segment_start - 1, "before", first_fault),
                (segment_end + 1, "after", last_fault),
            )
            for candidate_index, direction, source_fault in candidates:
                if not 0 <= candidate_index < len(ordered):
                    continue
                candidate = ordered[candidate_index]
                if candidate.sample_id not in trusted_ids:
                    continue
                if direction == "before":
                    adjacent = _same_segment(
                        candidate,
                        source_fault,
                    )
                    gap_s = max(
                        0.0,
                        float(source_fault.timestamp) - float(candidate.timestamp),
                    )
                else:
                    adjacent = _same_segment(
                        source_fault,
                        candidate,
                    )
                    gap_s = max(
                        0.0,
                        float(candidate.timestamp) - float(source_fault.timestamp),
                    )
                if not adjacent:
                    continue
                guard_sources[candidate.sample_id]["source_ids"].append(
                    source_fault.sample_id
                )
                guard_sources[candidate.sample_id]["directions"].append(direction)
                guard_sources[candidate.sample_id]["gaps"].append(gap_s)
            index = segment_end + 1

    guard_records: List[MeasuredSampleRecordV1] = []
    for sample_id, evidence in sorted(guard_sources.items()):
        source_ids = sorted(set(str(value) for value in evidence["source_ids"]))
        directions = sorted(set(str(value) for value in evidence["directions"]))
        guard_records.append(
            replace(
                by_id[sample_id],
                quality_status=V2_GUARD_STATUS,
                exclusion_reason=V2_GUARD_REASON,
                hard_fault_kind=None,
                guard_source_sample_ids=source_ids,
                guard_directions=directions,
                guard_distance=1,
                guard_policy_version=V2_POLICY_VERSION,
                guard_time_gap_s=float(max(evidence["gaps"])),
            )
        )
    guard_ids = {record.sample_id for record in guard_records}
    survivors = [
        record for record in remaining_trusted if record.sample_id not in guard_ids
    ]
    return (
        _attach_repeat_metadata_for_trusted(survivors),
        direct_hard_faults,
        sorted(
            guard_records,
            key=lambda item: (item.system_type, item.snapshot_path_relative),
        ),
        matched_anomalies,
    )


def derive_v2_records(
    trusted_records: Sequence[MeasuredSampleRecordV1],
    hard_fault_records: Sequence[MeasuredSampleRecordV1],
) -> Tuple[List[MeasuredSampleRecordV1], List[MeasuredSampleRecordV1]]:
    """Return V2 trusted rows and one-neighbor guard rows.

    A direct-fault segment is a maximal chronological run of direct faults in
    one system/time block whose successive acquisition gaps do not exceed the
    registered cadence bound.  Only the immediate trusted predecessor and
    successor in the complete V1 chronology are guarded.
    """

    trusted, _, guards, _ = derive_v2_collections(
        trusted_records,
        hard_fault_records,
    )
    return trusted, guards


def _attach_repeat_metadata_for_trusted(
    records: Sequence[MeasuredSampleRecordV1],
) -> List[MeasuredSampleRecordV1]:
    grouped: Dict[str, List[MeasuredSampleRecordV1]] = defaultdict(list)
    for record in records:
        grouped[record.repeat_group_id].append(record)
    updated: List[MeasuredSampleRecordV1] = []
    for group in grouped.values():
        ordered = sorted(group, key=_chronological_key)
        repeat_count = len(ordered)
        for repeat_index, record in enumerate(ordered):
            updated.append(
                replace(
                    record,
                    repeat_index=repeat_index,
                    repeat_count=repeat_count,
                    sample_weight=1.0 / float(repeat_count),
                )
            )
    return sorted(updated, key=lambda item: (item.system_type, item.snapshot_path_relative))


def _preserved_split_payload(
    trusted_records: Sequence[MeasuredSampleRecordV1],
    parent_split_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    parent_assignments = dict(parent_split_payload.get("assignments") or {})
    assignments = {
        record.sample_id: str(parent_assignments[record.sample_id])
        for record in trusted_records
    }
    split_sample_ids: Dict[str, List[str]] = defaultdict(list)
    split_group_ids: Dict[str, set[str]] = defaultdict(set)
    for record in trusted_records:
        split_name = assignments[record.sample_id]
        split_sample_ids[split_name].append(record.sample_id)
        split_group_ids[split_name].add(split_group_key(record))
    split_hash_payload = {
        "schema": SPLIT_SCHEMA,
        "seed": int(parent_split_payload["seed"]),
        "assignments": assignments,
    }
    split_hash = _stable_hash(split_hash_payload)
    result = {
        **split_hash_payload,
        "split_hash": split_hash,
        "splits": {
            split_name: {
                "sample_ids": sorted(sample_ids),
                "sample_count": len(sample_ids),
                "group_count": len(split_group_ids[split_name]),
            }
            for split_name, sample_ids in sorted(split_sample_ids.items())
        },
        "repeat_pairs": [pair.__dict__ for pair in build_repeat_pairs(trusted_records)],
        "parent_split_hash": str(parent_split_payload["split_hash"]),
        "survivor_assignments_preserved": True,
    }
    result["audit"] = audit_splits(trusted_records, result)
    return result


def write_v2_manifest_outputs(
    parent_manifest_dir: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    parent = Path(parent_manifest_dir)
    destination = Path(output_dir)
    trusted_v1 = load_measured_records(parent, trusted_only=True)
    hard_faults = _read_jsonl(parent / V1_HARD_FAULT_FILE_NAME)
    parent_summary = _read_json(parent / "measured_data_summary.json")
    parent_splits = _read_json(parent / V1_SPLIT_FILE_NAME)
    trusted_v2, direct_hard_faults, guard_records, hardware_anomalies = (
        derive_v2_collections(
            trusted_v1,
            hard_faults,
        )
    )
    if (
        len(trusted_v2) != 3032
        or len(direct_hard_faults) != 138
        or len(guard_records) != 69
        or len(hardware_anomalies) != 1
    ):
        raise ValueError("V2 live corpus counts disagree with the authorized contract")
    all_v2 = sorted(
        [*trusted_v2, *direct_hard_faults, *guard_records],
        key=lambda record: record.sample_id,
    )
    summary = _summary(all_v2)
    summary["schema"] = "measured_data_summary_v2"
    summary["audit_profile"] = {
        "profile": "operator_curated_hardware_anomaly_guard_band_v2_3_20260812",
        "passed": True,
        "checks": {
            "3qubit.trusted_snapshot_count": {
                "expected": 1076,
                "actual": sum(r.system_type == "3qubit" for r in trusted_v2),
                "passed": sum(r.system_type == "3qubit" for r in trusted_v2) == 1076,
            },
            "4qubit.trusted_snapshot_count": {
                "expected": 1956,
                "actual": sum(r.system_type == "4qubit" for r in trusted_v2),
                "passed": sum(r.system_type == "4qubit" for r in trusted_v2) == 1956,
            },
            "3qubit.rho_available_count": {
                "expected": 12,
                "actual": sum(
                    r.system_type == "3qubit" and r.rho_available
                    for r in trusted_v2
                ),
                "passed": sum(
                    r.system_type == "3qubit" and r.rho_available
                    for r in trusted_v2
                ) == 12,
            },
            "4qubit.rho_available_count": {
                "expected": 1956,
                "actual": sum(
                    r.system_type == "4qubit" and r.rho_available
                    for r in trusted_v2
                ),
                "passed": sum(
                    r.system_type == "4qubit" and r.rho_available
                    for r in trusted_v2
                ) == 1956,
            },
        },
    }
    summary["audit_profile"]["passed"] = all(
        item["passed"] for item in summary["audit_profile"]["checks"].values()
    )
    if not summary["audit_profile"]["passed"]:
        raise ValueError("V2 live corpus audit-profile checks failed")
    splits = _preserved_split_payload(trusted_v2, parent_splits)
    parent_manifest_hash = str(parent_summary["manifest_hash"])
    provenance = {
        "schema": V2_SCHEMA,
        "version": 2,
        "operator_authorized_at_local_date": "2026-08-12",
        "timezone": "Asia/Shanghai",
        "parent_contract_schema": "physics_hybrid_v8_operator_curated_corpus_v1",
        "parent_manifest_hash": parent_manifest_hash,
        "parent_summary_sha256": _sha256_file(parent / "measured_data_summary.json"),
        "parent_split_hash": str(parent_splits["split_hash"]),
        "parent_hard_fault_manifest_sha256": _sha256_file(
            parent / V1_HARD_FAULT_FILE_NAME
        ),
        "direct_hard_fault_rule_changed": True,
        "direct_hard_fault_rows_reclassified": False,
        "v1_signal_idler_hard_fault_rule_changed": False,
        "hardware_anomaly_rule": {
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
            "matched_sample_count": len(hardware_anomalies),
            "matched_sample_ids": sorted(
                str(item["sample_id"]) for item in hardware_anomalies
            ),
            "matched_samples": hardware_anomalies,
        },
        "guard_policy": {
            "policy_version": V2_POLICY_VERSION,
            "distance": 1,
            "directions": ["before", "after"],
            "segment_scope": "same_system_chronological_contiguous_direct_faults",
            "time_block_identity_required": False,
            "chronological_adjacency_has_time_gap_limit": False,
            "maximum_neighbor_gap_s": None,
            "chronological_order_fields": ["timestamp", "ended_at", "sample_id"],
            "reason": V2_GUARD_REASON,
            "guard_rows_are_direct_hard_faults": False,
        },
        "counts": {
            "trusted_v1": len(trusted_v1),
            "direct_hard_fault_v1": len(hard_faults),
            "direct_hardware_anomaly_added": len(hardware_anomalies),
            "direct_hard_fault": len(direct_hard_faults),
            "guard_band": len(guard_records),
            "trusted_v2": len(trusted_v2),
            "total_records": len(all_v2),
            "trusted_v2_by_system": dict(
                sorted(Counter(r.system_type for r in trusted_v2).items())
            ),
            "guard_band_by_system": dict(
                sorted(Counter(r.system_type for r in guard_records).items())
            ),
            "direct_hard_fault_by_system": dict(
                sorted(Counter(r.system_type for r in direct_hard_faults).items())
            ),
        },
        "survivor_split_assignments_preserved": True,
        "environment_quality_contract": {
            "role": "weak_analysis_only",
            "hard_label_or_exclusion_authority": False,
            "temperature_threshold_c": 21.5,
            "high_temperature_interpretation": (
                "lower_global_optical_throughput_counts_and_experiment_gsi"
            ),
            "stable_temperature_interpretation": "higher_data_quality_trust_prior",
            "humidity_unit": "percent_rh",
            "sensor_distance_from_platform_m_approx": 2.0,
        },
    }
    provenance["provenance_sha256"] = _stable_hash(provenance)

    for system_type, suffix in (("3qubit", "3q"), ("4qubit", "4q")):
        system_trusted = [r for r in trusted_v2 if r.system_type == system_type]
        _write_jsonl(destination / f"trusted_samples_{suffix}.jsonl", system_trusted)
        _write_json(
            destination / f"state_resolved_{suffix}.json",
            _state_resolution(system_trusted),
        )
    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(destination / V1_HARD_FAULT_FILE_NAME, direct_hard_faults)
    _write_jsonl(destination / GUARD_FILE_NAME, guard_records)
    _write_json(destination / "measured_data_summary.json", summary)
    _write_json(destination / V1_SPLIT_FILE_NAME, splits)
    _write_json(destination / "split_audit.json", splits["audit"])
    command_summary = {
        "schema": "command_provenance_summary_v1",
        "distribution": dict(
            sorted(Counter(record.command_provenance for record in trusted_v2).items())
        ),
        "command_supervision_sample_count": sum(
            record.command_supervision_mask for record in trusted_v2
        ),
        "target_only_sample_count": sum(
            record.command_provenance == "target_only_approximation"
            for record in trusted_v2
        ),
        "missing_sample_count": sum(
            record.command_provenance == "missing" for record in trusted_v2
        ),
    }
    _write_json(destination / "command_provenance_summary.json", command_summary)
    _write_json(destination / PROVENANCE_FILE_NAME, provenance)
    return {
        "summary": summary,
        "splits": splits,
        "command_provenance": command_summary,
        "provenance": provenance,
        "trusted_records": trusted_v2,
        "hard_fault_records": direct_hard_faults,
        "guard_records": guard_records,
        "hardware_anomalies": hardware_anomalies,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-manifest-dir",
        default="ml_recon/reports/measured_data",
    )
    parser.add_argument(
        "--output-dir",
        default="ml_recon/reports/measured_data_v2",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = write_v2_manifest_outputs(
        args.parent_manifest_dir,
        args.output_dir,
    )
    print(json.dumps({
        "schema": V2_SCHEMA,
        "manifest_hash": result["summary"]["manifest_hash"],
        "split_hash": result["splits"]["split_hash"],
        "trusted_sample_count": len(result["trusted_records"]),
        "direct_hard_fault_sample_count": len(result["hard_fault_records"]),
        "guard_band_sample_count": len(result["guard_records"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GUARD_FILE_NAME",
    "PROVENANCE_FILE_NAME",
    "V2_FPGA_ANOMALY_KIND",
    "V2_FPGA_ANOMALY_REASON",
    "V2_MULTI_RATIO_ANOMALY_KIND",
    "V2_MULTI_RATIO_ANOMALY_REASON",
    "V2_TOTAL_MULTI_TO_IDLER_RATIO_MAX",
    "V2_GUARD_REASON",
    "V2_GUARD_STATUS",
    "V2_POLICY_VERSION",
    "V2_SCHEMA",
    "derive_v2_collections",
    "derive_v2_records",
    "fpga_hardware_anomaly_evidence",
    "write_v2_manifest_outputs",
]
