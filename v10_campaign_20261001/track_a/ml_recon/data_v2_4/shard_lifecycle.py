"""Atomic lifecycle ledger and safe-eviction gates for ephemeral V2.4 shards."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping

from .hashing import read_canonical_json, write_canonical_json


class V24ShardState(str, Enum):
    PLANNED = "PLANNED"
    GENERATING = "GENERATING"
    VALIDATED = "VALIDATED"
    READY_LOCAL = "READY_LOCAL"
    CLAIMED_BY_TRAINER = "CLAIMED_BY_TRAINER"
    CONSUMED = "CONSUMED"
    CHECKPOINT_COVERED = "CHECKPOINT_COVERED"
    EVICTABLE = "EVICTABLE"
    EVICTED = "EVICTED"
    FAILED = "FAILED"


ALLOWED_TRANSITIONS = {
    V24ShardState.PLANNED: {V24ShardState.GENERATING, V24ShardState.FAILED},
    V24ShardState.GENERATING: {V24ShardState.VALIDATED, V24ShardState.FAILED},
    V24ShardState.VALIDATED: {V24ShardState.READY_LOCAL, V24ShardState.FAILED},
    V24ShardState.READY_LOCAL: {V24ShardState.CLAIMED_BY_TRAINER, V24ShardState.FAILED},
    V24ShardState.CLAIMED_BY_TRAINER: {V24ShardState.CONSUMED, V24ShardState.FAILED},
    V24ShardState.CONSUMED: {V24ShardState.CHECKPOINT_COVERED, V24ShardState.FAILED},
    V24ShardState.CHECKPOINT_COVERED: {V24ShardState.EVICTABLE, V24ShardState.FAILED},
    V24ShardState.EVICTABLE: {V24ShardState.EVICTED, V24ShardState.FAILED},
    V24ShardState.EVICTED: set(),
    V24ShardState.FAILED: set(),
}


@contextmanager
def exclusive_file_lock(path: str | Path, *, timeout_s: float = 30.0) -> Iterator[None]:
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    descriptor = -1
    while descriptor < 0:
        try:
            descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for V2.4 lock: {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _new_ledger() -> Dict[str, Any]:
    return {
        "ledger_kind": "v24_shard_lifecycle_v1",
        "records": {},
        "events": [],
        "producer_paused": False,
        "pause_reason": "",
    }


class V24ShardLifecycleLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        if not self.path.exists():
            write_canonical_json(self.path, _new_ledger())
        self._validate(self._read())

    def _read(self) -> Dict[str, Any]:
        return dict(read_canonical_json(self.path))

    @staticmethod
    def _validate(value: Mapping[str, Any]) -> None:
        if value.get("ledger_kind") != "v24_shard_lifecycle_v1":
            raise ValueError("invalid V2.4 lifecycle ledger kind")
        for shard_id, record in value.get("records", {}).items():
            if record.get("shard_id") != shard_id:
                raise ValueError("V2.4 lifecycle record key mismatch")
            V24ShardState(record["state"])

    def snapshot(self) -> Dict[str, Any]:
        value = self._read()
        self._validate(value)
        return value

    def register(self, shard_id: str, start_index: int, stop_index: int,
                 *, planned_epochs: int = 1) -> Dict[str, Any]:
        if start_index < 0 or stop_index <= start_index or planned_epochs < 1:
            raise ValueError("invalid planned V2.4 shard range")
        with exclusive_file_lock(self.lock_path):
            value = self._read()
            existing = value["records"].get(shard_id)
            proposed = {
                "shard_id": shard_id,
                "start_index": int(start_index),
                "stop_index": int(stop_index),
                "state": V24ShardState.PLANNED.value,
                "planned_epochs": int(planned_epochs),
                "completed_epochs": 0,
                "checkpoint_sha": "",
                "checkpoint_records_range": False,
                "metrics_flushed": False,
                "pending_five_arm_consumers": 0,
                "regeneration_audit_available": False,
                "observation_path": "",
                "truth_path": "",
                "ready_path": "",
                "cell_static_sha": "",
                "failure": "",
            }
            if existing is not None:
                if (existing["start_index"], existing["stop_index"],
                        existing["planned_epochs"]) != (
                        proposed["start_index"], proposed["stop_index"],
                        proposed["planned_epochs"]):
                    raise ValueError("registered shard range changed")
                return dict(existing)
            value["records"][shard_id] = proposed
            value["events"].append({"event": "REGISTER", "shard_id": shard_id})
            write_canonical_json(self.path, value)
            return proposed

    def transition(self, shard_id: str, state: V24ShardState,
                   **updates: Any) -> Dict[str, Any]:
        with exclusive_file_lock(self.lock_path):
            value = self._read()
            if shard_id not in value["records"]:
                raise KeyError(f"unregistered V2.4 shard: {shard_id}")
            record = value["records"][shard_id]
            current = V24ShardState(record["state"])
            if state == current and current == V24ShardState.GENERATING:
                event = "RESUME_GENERATION"
            elif state not in ALLOWED_TRANSITIONS[current]:
                raise ValueError(f"invalid shard transition {current.value} -> {state.value}")
            else:
                event = "TRANSITION"
            record.update(updates)
            record["state"] = state.value
            value["events"].append({
                "event": event, "shard_id": shard_id,
                "from": current.value, "to": state.value,
            })
            write_canonical_json(self.path, value)
            return dict(record)

    def pause_low_disk(self, *, free_bytes: int, reserve_bytes: int,
                       buffered_bytes: int) -> Dict[str, Any]:
        with exclusive_file_lock(self.lock_path):
            value = self._read()
            value["producer_paused"] = True
            value["pause_reason"] = "LOW_DISK_PAUSE"
            value["events"].append({
                "event": "LOW_DISK_PAUSE", "free_bytes": int(free_bytes),
                "reserve_bytes": int(reserve_bytes),
                "buffered_bytes": int(buffered_bytes),
            })
            write_canonical_json(self.path, value)
            return value

    def resume_producer(self) -> Dict[str, Any]:
        with exclusive_file_lock(self.lock_path):
            value = self._read()
            value["producer_paused"] = False
            value["pause_reason"] = ""
            value["events"].append({"event": "PRODUCER_RESUME"})
            write_canonical_json(self.path, value)
            return value

    def update_consumption_evidence(self, shard_id: str, **updates: Any) -> Dict[str, Any]:
        allowed = {
            "completed_epochs", "checkpoint_sha", "checkpoint_records_range",
            "metrics_flushed", "pending_five_arm_consumers",
            "regeneration_audit_available",
        }
        if set(updates) - allowed:
            raise ValueError("unknown shard consumption evidence")
        with exclusive_file_lock(self.lock_path):
            value = self._read()
            record = value["records"][shard_id]
            record.update(updates)
            value["events"].append({"event": "EVIDENCE_UPDATE", "shard_id": shard_id})
            write_canonical_json(self.path, value)
            return dict(record)

    def require_evictable(self, shard_id: str) -> Dict[str, Any]:
        record = self.snapshot()["records"][shard_id]
        if V24ShardState(record["state"]) != V24ShardState.CHECKPOINT_COVERED:
            raise ValueError("shard is not checkpoint-covered")
        conditions = {
            "all_epochs_complete": record["completed_epochs"] >= record["planned_epochs"],
            "checkpoint_saved": len(record["checkpoint_sha"]) == 64,
            "checkpoint_records_range": bool(record["checkpoint_records_range"]),
            "metrics_flushed": bool(record["metrics_flushed"]),
            "no_pending_five_arm_consumer": record["pending_five_arm_consumers"] == 0,
            "regeneration_audit_available": bool(record["regeneration_audit_available"]),
        }
        failed = sorted(name for name, passed in conditions.items() if not passed)
        if failed:
            raise ValueError(f"shard is not safely evictable: {failed}")
        return conditions
