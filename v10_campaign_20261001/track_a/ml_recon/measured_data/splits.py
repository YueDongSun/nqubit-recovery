"""Deterministic group-disjoint splits for measured snapshots and probes."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .contracts import MeasuredSampleRecordV1


SPLIT_SCHEMA = "measured_splits_v1"
SPLIT_SEED = 20260805
SYSTEM_SPLIT_FRACTIONS: Dict[str, Tuple[Tuple[str, float], ...]] = {
    "3qubit": (("train", 0.70), ("validation", 0.15), ("target_holdout", 0.15)),
    "4qubit": (
        ("train_target_family", 0.70),
        ("validation_target_family", 0.10),
        ("target_ood", 0.10),
        ("time_run_ood", 0.10),
    ),
}
PROBE_SPLIT_FRACTIONS = (
    ("probe_train", 0.70),
    ("probe_validation", 0.15),
    ("probe_holdout", 0.15),
)


@dataclass(frozen=True)
class RepeatPairV1:
    repeat_group_id: str
    first_sample_id: str
    second_sample_id: str
    first_timestamp: float
    second_timestamp: float
    observed_count_difference: Dict[str, int]


def _stable_unit_interval(value: str, seed: int = SPLIT_SEED) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def split_group_key(record: MeasuredSampleRecordV1) -> str:
    """Build a conservative group key that cannot split repeats or probes."""
    intervention_family = record.intervention_id or "base"
    probe_identity = record.probe_anchor_id or record.probe_pair_id or "no_probe"
    payload = {
        "system_type": record.system_type,
        "base_state_family_id": record.base_state_family_id,
        "exact_target_hash": record.exact_target_hash,
        "intervention_family": intervention_family,
        "campaign_or_run": record.campaign_id or record.automation_run_id,
        "calibration_hash": record.calibration_snapshot_hash,
        "time_block_id": record.time_block_id,
        "probe_identity": probe_identity,
        "repeat_group_id": record.repeat_group_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _assign_fraction(value: float, fractions: Sequence[Tuple[str, float]]) -> str:
    cumulative = 0.0
    for name, fraction in fractions:
        cumulative += float(fraction)
        if value < cumulative:
            return name
    return fractions[-1][0]


def _balanced_group_assignments(
    group_keys: Sequence[tuple[str, str]],
    fractions: Sequence[Tuple[str, float]],
    *,
    seed: int,
) -> Dict[tuple[str, str], str]:
    """Assign small group sets deterministically while keeping each split non-empty."""
    ordered = sorted(set(group_keys), key=lambda value: _stable_unit_interval(":".join(value), seed))
    split_names = [name for name, fraction in fractions if float(fraction) > 0.0]
    counts = {name: 0 for name in split_names}
    if len(ordered) >= len(split_names):
        for name in split_names:
            counts[name] = 1
    remaining = len(ordered) - sum(counts.values())
    while remaining > 0:
        name = max(
            split_names,
            key=lambda value: (
                float(dict(fractions)[value]) * len(ordered) - counts[value],
                -split_names.index(value),
            ),
        )
        counts[name] += 1
        remaining -= 1
    assignments: Dict[tuple[str, str], str] = {}
    offset = 0
    for name in split_names:
        for group_key in ordered[offset:offset + counts[name]]:
            assignments[group_key] = name
        offset += counts[name]
    return assignments


def build_measured_splits(
    records: Iterable[MeasuredSampleRecordV1],
    *,
    seed: int = SPLIT_SEED,
) -> Dict[str, Any]:
    """Assign whole target/repeat/probe families, never individual snapshots."""
    trusted = sorted(
        (record for record in records if record.quality_status == "trusted"),
        key=lambda item: item.sample_id,
    )
    intervention_families = {
        (record.system_type, record.base_state_family_id)
        for record in trusted
        if record.intervention_id == "q4_recovery_plus90"
    }
    probe_family_keys = {
        (record.system_type, f"probe:{record.probe_anchor_id or record.probe_pair_id}")
        for record in trusted
        if record.probe_anchor_id or record.probe_pair_id
    }
    probe_family_assignment = _balanced_group_assignments(
        sorted(probe_family_keys), PROBE_SPLIT_FRACTIONS, seed=seed,
    )
    assignments: Dict[str, str] = {}
    split_sample_ids: Dict[str, List[str]] = defaultdict(list)
    split_group_ids: Dict[str, set[str]] = defaultdict(set)
    family_assignment: Dict[tuple[str, str], str] = {}

    for record in trusted:
        if record.probe_anchor_id or record.probe_pair_id:
            anchor = record.probe_anchor_id or record.probe_pair_id or ""
            family_key = (record.system_type, f"probe:{anchor}")
            fractions = PROBE_SPLIT_FRACTIONS
        elif (record.system_type, record.base_state_family_id) in intervention_families:
            # Recovery rows and their original-plan bases form one indivisible
            # intervention family.  The benchmark therefore cannot leak a base
            # target into an ordinary train/validation split.
            family_key = (record.system_type, f"intervention:{record.base_state_family_id}")
            fractions = (("q4_plus90_intervention_test", 1.0),)
        else:
            # All repeated measurements and known interventions sharing the
            # same physical base target are bound to one family assignment.
            family_key = (record.system_type, record.base_state_family_id)
            fractions = SYSTEM_SPLIT_FRACTIONS[record.system_type]
        if family_key in probe_family_assignment:
            family_assignment[family_key] = probe_family_assignment[family_key]
        elif family_key not in family_assignment:
            family_assignment[family_key] = _assign_fraction(
                _stable_unit_interval(":".join(family_key), seed), fractions,
            )
        split_name = family_assignment[family_key]
        assignments[record.sample_id] = split_name
        split_sample_ids[split_name].append(record.sample_id)
        split_group_ids[split_name].add(split_group_key(record))

    repeat_pairs = [pair.__dict__ for pair in build_repeat_pairs(trusted)]
    split_hash_payload = {
        "schema": SPLIT_SCHEMA,
        "seed": seed,
        "assignments": assignments,
    }
    split_hash = hashlib.sha256(
        json.dumps(split_hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = {
        **split_hash_payload,
        "split_hash": split_hash,
        "splits": {
            name: {
                "sample_ids": sorted(sample_ids),
                "sample_count": len(sample_ids),
                "group_count": len(split_group_ids[name]),
            }
            for name, sample_ids in sorted(split_sample_ids.items())
        },
        "repeat_pairs": repeat_pairs,
    }
    result["audit"] = audit_splits(trusted, result)
    return result


def build_repeat_pairs(records: Iterable[MeasuredSampleRecordV1]) -> List[RepeatPairV1]:
    grouped: Dict[str, List[MeasuredSampleRecordV1]] = defaultdict(list)
    for record in records:
        grouped[record.repeat_group_id].append(record)
    pairs: List[RepeatPairV1] = []
    for repeat_group_id, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: (item.timestamp, item.sample_id))
        for first, second in zip(ordered, ordered[1:]):
            pairs.append(RepeatPairV1(
                repeat_group_id=repeat_group_id,
                first_sample_id=first.sample_id,
                second_sample_id=second.sample_id,
                first_timestamp=first.timestamp,
                second_timestamp=second.timestamp,
                observed_count_difference={
                    "signal": second.signal_total - first.signal_total,
                    "idler": second.idler_total - first.idler_total,
                    "coin": second.coin_total - first.coin_total,
                },
            ))
    return pairs


def audit_splits(
    records: Iterable[MeasuredSampleRecordV1],
    split_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    records = list(records)
    assignments = dict(split_payload.get("assignments", {}))
    fields = {
        "target_group": lambda item: item.base_state_family_id,
        "repeat_group": lambda item: item.repeat_group_id,
        "probe_pair": lambda item: item.probe_pair_id,
    }
    overlaps: Dict[str, Dict[str, List[str]]] = {}
    for label, key_function in fields.items():
        locations: Dict[str, set[str]] = defaultdict(set)
        for record in records:
            key = key_function(record)
            if key:
                locations[str(key)].add(str(assignments.get(record.sample_id, "missing")))
        overlaps[label] = {
            key: sorted(splits) for key, splits in locations.items() if len(splits) > 1
        }
    sample_ids = [record.sample_id for record in records]
    paths = [record.snapshot_path_relative for record in records]
    checks = {
        "target_group_no_overlap": not overlaps["target_group"],
        "repeat_group_no_overlap": not overlaps["repeat_group"],
        "probe_pair_no_overlap": not overlaps["probe_pair"],
        "intervention_family_no_unintended_overlap": not overlaps["target_group"],
        "calibration_time_holdout_recorded": all(record.time_block_id for record in records),
        "no_duplicate_path": len(paths) == len(set(paths)),
        "no_duplicate_sample_id": len(sample_ids) == len(set(sample_ids)),
        "all_samples_assigned": set(sample_ids) == set(assignments),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "overlaps": overlaps,
    }


__all__ = [
    "PROBE_SPLIT_FRACTIONS",
    "SPLIT_SCHEMA",
    "SPLIT_SEED",
    "SYSTEM_SPLIT_FRACTIONS",
    "RepeatPairV1",
    "audit_splits",
    "build_measured_splits",
    "build_repeat_pairs",
    "split_group_key",
]
