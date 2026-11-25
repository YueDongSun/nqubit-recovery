"""Compact deterministic range index for regular synthetic training shards."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .hashing import read_canonical_json, sha256_json, write_canonical_json
from .shard_lifecycle import exclusive_file_lock


@dataclass(frozen=True)
class V24RangeIndexEntry:
    cell_id: str
    split: str
    shard_id: str
    start_index: int
    stop_index: int
    observation_path: str
    observation_sha: str
    truth_path: str
    truth_sha: str
    ready_path: str
    ready_sha: str
    cell_static_path: str
    cell_static_sha: str
    cell_static_file_sha: str
    build_id: str
    dataset_family: str
    N: int
    noise_family: str
    eta: float
    kappa: float
    shot_mode: str
    shot_budget: int
    measurement_schedule: str
    global_seed: int
    generator_sha: str
    contract_sha: str
    schema_sha: str
    audit_row_hashes: Tuple[Tuple[int, str], ...] = ()

    @property
    def row_count(self) -> int:
        return self.stop_index - self.start_index

    def validate(self) -> None:
        if self.start_index < 0 or self.stop_index <= self.start_index:
            raise ValueError("invalid V2.4 synthetic range")
        if self.N < 3 or self.N > 256:
            raise ValueError("invalid N in V2.4 range index")
        if self.shot_mode not in ("S_total", "S_setting", "S_ideal") \
                or self.shot_budget <= 0 or not 0.0 <= self.eta <= 1.0 \
                or not 0.0 <= self.kappa <= 1.0:
            raise ValueError("invalid physical cell parameters in V2.4 range index")
        sha_fields = (
            self.observation_sha, self.truth_sha, self.ready_sha,
            self.cell_static_sha, self.cell_static_file_sha,
            self.generator_sha, self.contract_sha, self.schema_sha,
        )
        if any(len(value) != 64 for value in sha_fields):
            raise ValueError("V2.4 range index contains a non-SHA256 identity")
        for index, row_sha in self.audit_row_hashes:
            if not self.start_index <= index < self.stop_index or len(row_sha) != 64:
                raise ValueError("invalid bounded row audit hash")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["audit_row_hashes"] = [list(item) for item in self.audit_row_hashes]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V24RangeIndexEntry":
        payload = dict(value)
        payload["audit_row_hashes"] = tuple(
            (int(item[0]), str(item[1])) for item in payload.get("audit_row_hashes", ()))
        entry = cls(**payload)
        entry.validate()
        return entry


def _validated_entries(entries: Iterable[V24RangeIndexEntry]
                       ) -> Tuple[V24RangeIndexEntry, ...]:
    values = tuple(sorted(entries, key=lambda item: (
        item.cell_id, item.split, item.start_index, item.stop_index)))
    seen_shards = set()
    previous: Dict[Tuple[str, str], int] = {}
    for entry in values:
        entry.validate()
        if entry.shard_id in seen_shards:
            raise ValueError("duplicate shard_id in V2.4 range index")
        seen_shards.add(entry.shard_id)
        key = (entry.cell_id, entry.split)
        expected_start = previous.get(key, entry.start_index)
        if entry.start_index != expected_start:
            raise ValueError("V2.4 synthetic ranges must be contiguous and non-overlapping")
        previous[key] = entry.stop_index
    return values


def write_range_index(path: str | Path, entries: Iterable[V24RangeIndexEntry]) -> Dict[str, Any]:
    values = _validated_entries(entries)
    payload = {
        "index_kind": "v24_synthetic_range_index_v1",
        "entries": [entry.to_dict() for entry in values],
    }
    index_sha = sha256_json(payload)
    file_sha = write_canonical_json(path, payload)
    if file_sha != index_sha:
        raise RuntimeError("V2.4 range index canonical SHA mismatch")
    return {"path": str(path), "range_index_sha": index_sha,
            "shards": len(values), "rows": sum(item.row_count for item in values)}


def read_range_index(path: str | Path) -> Tuple[V24RangeIndexEntry, ...]:
    payload = read_canonical_json(path)
    if payload.get("index_kind") != "v24_synthetic_range_index_v1":
        raise ValueError("invalid V2.4 range index kind")
    return _validated_entries(
        V24RangeIndexEntry.from_dict(item) for item in payload["entries"])


def append_range_entry(path: str | Path, entry: V24RangeIndexEntry) -> Dict[str, Any]:
    destination = Path(path)
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    with exclusive_file_lock(lock_path):
        entries: List[V24RangeIndexEntry] = (
            list(read_range_index(destination)) if destination.exists() else [])
        existing = [item for item in entries if item.shard_id == entry.shard_id]
        if existing:
            if existing != [entry]:
                raise ValueError("range index shard identity changed during resume")
            return write_range_index(destination, entries)
        entries.append(entry)
        return write_range_index(destination, entries)


class V24SyntheticRangeIndex:
    """Small in-memory shard-range table, never a per-sample row dictionary."""

    def __init__(self, path: str | Path, *, cell_id: str, split: str):
        self.entries = tuple(
            entry for entry in read_range_index(path)
            if entry.cell_id == cell_id and entry.split == split)
        if not self.entries:
            raise ValueError("V2.4 range index selected zero shards")
        self.starts = tuple(entry.start_index for entry in self.entries)
        self.base_index = self.entries[0].start_index
        self.stop_index = self.entries[-1].stop_index
        if self.stop_index - self.base_index != sum(entry.row_count for entry in self.entries):
            raise ValueError("selected V2.4 ranges are not contiguous")

    def __len__(self) -> int:
        return self.stop_index - self.base_index

    def locate(self, dataset_index: int) -> Tuple[V24RangeIndexEntry, int, int]:
        if dataset_index < 0 or dataset_index >= len(self):
            raise IndexError(dataset_index)
        sample_index = self.base_index + int(dataset_index)
        position = bisect_right(self.starts, sample_index) - 1
        entry = self.entries[position]
        if sample_index >= entry.stop_index:
            raise RuntimeError("V2.4 range lookup crossed a shard gap")
        return entry, sample_index - entry.start_index, sample_index
