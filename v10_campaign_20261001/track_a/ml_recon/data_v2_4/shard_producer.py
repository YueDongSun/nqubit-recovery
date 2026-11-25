"""Deterministic bounded-memory producer for ephemeral V2.4 train shards."""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Tuple

import numpy as np

from .cell_static import split_cell_static_fields, write_cell_static_store
from .contracts import NQubitDatasetContractV240
from .hashing import read_canonical_json, sha256_json, write_canonical_json
from .materialization import V24MaterializationProfileV1
from .range_index import V24RangeIndexEntry, append_range_entry
from .sample_keys import make_sample_key
from .schema_registry import schema_sha256
from .shard_lifecycle import (
    V24ShardLifecycleLedger,
    V24ShardState,
    exclusive_file_lock,
)
from .shard_reader import V24Hdf5HandleCache
from .storage_hdf5 import write_hdf5_shard
from .synthetic.generator import _render_dataset_family, _row_sha, _stack
from .version import GENERATOR_SHA


class V24SimulatedInterruption(RuntimeError):
    """Test-only interruption after deterministic rows have been rendered."""


class V24LowDiskPause(RuntimeError):
    pass


def rows_per_shard_from_profile(
    p95_dynamic_bytes_per_sample: int,
    profile: V24MaterializationProfileV1,
) -> int:
    profile.validate()
    if p95_dynamic_bytes_per_sample <= 0:
        raise ValueError("empirical p95 bytes/sample must be positive")
    target_bytes = profile.target_shard_mib * 1024**2
    estimated = target_bytes // int(p95_dynamic_bytes_per_sample)
    return max(profile.minimum_rows_per_shard,
               min(profile.maximum_rows_per_shard, int(estimated)))


@dataclass(frozen=True)
class V24ShardPlan:
    plan_id: str
    rung_plan_sha: str
    materialization_profile_sha: str
    build_id: str
    global_seed: int
    cell_id: str
    dataset_family: str
    N: int
    system_type: str
    measurement_domain: str
    noise_family: str
    eta: float
    kappa: float
    shot_mode: str
    shot_budget: int
    measurement_schedule: str
    split: str
    start_index: int
    stop_index: int
    rows_per_shard: int
    contract_sha: str
    schema_sha: str
    generator_sha: str
    plan_kind: str = "v24_shard_plan_v1"

    def validate(self) -> None:
        if self.plan_kind != "v24_shard_plan_v1" or not self.plan_id:
            raise ValueError("invalid V2.4 shard plan identity")
        if self.start_index < 0 or self.stop_index <= self.start_index:
            raise ValueError("invalid V2.4 shard plan range")
        if not 1 <= self.rows_per_shard <= 65_536:
            raise ValueError("invalid V2.4 rows_per_shard")
        if self.N < 3 or self.N > 256:
            raise ValueError("invalid V2.4 shard plan N")
        if self.shot_mode not in ("S_total", "S_setting", "S_ideal") \
                or self.shot_budget <= 0 or not 0.0 <= self.eta <= 1.0 \
                or not 0.0 <= self.kappa <= 1.0:
            raise ValueError("invalid V2.4 physical cell parameters")
        expected = (
            NQubitDatasetContractV240.default().sha256(), schema_sha256(), GENERATOR_SHA)
        if (self.contract_sha, self.schema_sha, self.generator_sha) != expected:
            raise ValueError("V2.4 shard plan dependency mismatch")
        if any(len(value) != 64 for value in (
                self.rung_plan_sha, self.materialization_profile_sha,
                self.contract_sha, self.schema_sha, self.generator_sha)):
            raise ValueError("V2.4 shard plan contains a non-SHA256 dependency")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    def shard_ranges(self) -> Tuple[Tuple[str, int, int], ...]:
        self.validate()
        values = []
        for start in range(self.start_index, self.stop_index, self.rows_per_shard):
            stop = min(start + self.rows_per_shard, self.stop_index)
            shard_id = "v24_" + sha256_json({
                "plan_id": self.plan_id, "cell_id": self.cell_id,
                "split": self.split, "start": start, "stop": stop,
            })[:24]
            values.append((shard_id, start, stop))
        return tuple(values)


def make_shard_plan(*, rung_plan_sha: str, materialization_profile_sha: str,
                    build_id: str, global_seed: int, cell: Mapping[str, Any],
                    split: str, start_index: int, stop_index: int,
                    rows_per_shard: int) -> V24ShardPlan:
    content = {
        "rung_plan_sha": rung_plan_sha,
        "materialization_profile_sha": materialization_profile_sha,
        "build_id": build_id,
        "global_seed": int(global_seed),
        "cell_id": str(cell["cell_id"]),
        "dataset_family": str(cell["dataset_family"]),
        "N": int(cell["N"]),
        "system_type": str(cell.get("system_type", "nqubit")),
        "measurement_domain": str(cell.get("measurement_domain", "global_pauli_v24")),
        "noise_family": str(cell.get("noise_family", "smoke_mixed")),
        "eta": float(cell.get("eta", 0.08)),
        "kappa": float(cell.get("kappa", 0.5)),
        "shot_mode": str(cell.get("shot_mode", "S_setting")),
        "shot_budget": int(cell.get("shot_budget", 64)),
        "measurement_schedule": str(cell.get(
            "measurement_schedule", "smoke_default")),
        "split": split,
        "start_index": int(start_index),
        "stop_index": int(stop_index),
        "rows_per_shard": int(rows_per_shard),
        "contract_sha": NQubitDatasetContractV240.default().sha256(),
        "schema_sha": schema_sha256(),
        "generator_sha": GENERATOR_SHA,
        "plan_kind": "v24_shard_plan_v1",
    }
    plan = V24ShardPlan(plan_id=sha256_json(content)[:24], **content)
    plan.validate()
    return plan


class V24ShardProducer:
    def __init__(self, local_root: str | Path, plan: V24ShardPlan,
                 profile: V24MaterializationProfileV1, *,
                 max_buffer_bytes: int,
                 free_bytes_provider: Callable[[Path], Tuple[int, int]] | None = None):
        plan.validate()
        profile.validate()
        if plan.materialization_profile_sha != profile.sha256():
            raise ValueError("shard plan/materialization profile mismatch")
        if max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be positive")
        self.root = Path(local_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.plan = plan
        self.profile = profile
        self.max_buffer_bytes = int(max_buffer_bytes)
        self.free_bytes_provider = free_bytes_provider or self._disk_capacity_free
        self.ledger = V24ShardLifecycleLedger(
            self.root / "ledger" / f"{plan.plan_id}.json")
        self.range_index_path = self.root / "index" / f"{plan.plan_id}_ranges.json"
        for shard_id, start, stop in self.plan.shard_ranges():
            self.ledger.register(shard_id, start, stop)

    @staticmethod
    def _disk_capacity_free(path: Path) -> Tuple[int, int]:
        usage = shutil.disk_usage(path)
        return int(usage.total), int(usage.free)

    def _buffered_bytes(self) -> int:
        total = 0
        for record in self.ledger.snapshot()["records"].values():
            if record["state"] in (V24ShardState.EVICTED.value, V24ShardState.FAILED.value):
                continue
            for key in ("observation_path", "truth_path", "ready_path"):
                path = self.root / record[key] if record.get(key) else None
                if path is not None and path.is_file():
                    total += path.stat().st_size
        return total

    def _preflight(self) -> None:
        capacity, free = self.free_bytes_provider(self.root)
        reserve = self.profile.reserve_bytes(capacity)
        buffered = self._buffered_bytes()
        if free <= reserve or buffered >= self.max_buffer_bytes:
            self.ledger.pause_low_disk(
                free_bytes=free, reserve_bytes=reserve, buffered_bytes=buffered)
            raise V24LowDiskPause("LOW_DISK_PAUSE")
        if self.ledger.snapshot()["producer_paused"]:
            self.ledger.resume_producer()

    def ready_queue_depth(self) -> int:
        active = {
            V24ShardState.READY_LOCAL.value,
            V24ShardState.CLAIMED_BY_TRAINER.value,
            V24ShardState.CONSUMED.value,
            V24ShardState.CHECKPOINT_COVERED.value,
            V24ShardState.EVICTABLE.value,
        }
        return sum(record["state"] in active
                   for record in self.ledger.snapshot()["records"].values())

    def produce_shard(self, shard_id: str, *,
                      simulate_interrupt_after_rows: int = 0) -> Dict[str, Any]:
        self._preflight()
        records = self.ledger.snapshot()["records"]
        if shard_id not in records:
            raise KeyError(f"shard is not in plan: {shard_id}")
        record = records[shard_id]
        state = V24ShardState(record["state"])
        if state == V24ShardState.READY_LOCAL:
            ready = read_canonical_json(self.root / record["ready_path"])
            return {**ready, "resumed": True}
        if state not in (V24ShardState.PLANNED, V24ShardState.GENERATING):
            raise ValueError(f"cannot generate shard in state {state.value}")
        lock_path = self.root / "locks" / f"{shard_id}.lock"
        with exclusive_file_lock(lock_path, timeout_s=0.1):
            self.ledger.transition(shard_id, V24ShardState.GENERATING)
            rendered_rows = []
            sample_ids = []
            keys = []
            for sample_index in range(record["start_index"], record["stop_index"]):
                key = make_sample_key(
                    global_seed=self.plan.global_seed, split=self.plan.split,
                    dataset_family=self.plan.dataset_family, N=self.plan.N,
                    index=sample_index, generator_sha=self.plan.generator_sha,
                    cell_id=self.plan.cell_id)
                rendered_rows.append(_render_dataset_family(
                    key, cell=self.plan.to_dict()))
                sample_ids.append(key.sample_id)
                keys.append(key)
                if simulate_interrupt_after_rows \
                        and len(rendered_rows) >= simulate_interrupt_after_rows:
                    raise V24SimulatedInterruption(
                        f"interrupted {shard_id} after {len(rendered_rows)} rows")
            full_observations = _stack(
                [row["observation"] for row in rendered_rows], sample_ids)
            truths = _stack([row["truth"] for row in rendered_rows], sample_ids)
            static_arrays, observations = split_cell_static_fields(full_observations)
            common_metadata = {
                "provenance": "synthetic",
                "dataset_family": self.plan.dataset_family,
                "cell_id": self.plan.cell_id,
                "split": self.plan.split,
                "system_type": self.plan.system_type,
                "N": self.plan.N,
                "measurement_domain": self.plan.measurement_domain,
                "noise_family": self.plan.noise_family,
                "eta": self.plan.eta,
                "kappa": self.plan.kappa,
                "shot_mode": self.plan.shot_mode,
                "shot_budget": self.plan.shot_budget,
                "measurement_schedule": self.plan.measurement_schedule,
                "contract_sha": self.plan.contract_sha,
                "schema_sha": self.plan.schema_sha,
                "generator_sha": self.plan.generator_sha,
                "build_id": self.plan.build_id,
                "shard_id": shard_id,
                "start_index": record["start_index"],
                "stop_index": record["stop_index"],
            }
            static_metadata = {
                key: value for key, value in common_metadata.items()
                if key not in (
                    "split", "shard_id", "start_index", "stop_index", "build_id")
            }
            static_result = write_cell_static_store(
                self.root, static_arrays, static_metadata)
            retention = ("ephemeral" if self.plan.split in self.profile.ephemeral_splits
                         else "persistent")
            shard_dir = self.root / retention / self.plan.cell_id / self.plan.split
            observation_path = shard_dir / f"{shard_id}_observation.h5"
            truth_path = shard_dir / f"{shard_id}_truth.h5"
            observation_metadata = {
                **common_metadata, "store": "observation",
                "storage_tier": "CORE",
                "cell_static_sha": static_result["cell_static_sha"],
            }
            truth_metadata = {
                **common_metadata, "store": "truth",
                "storage_tier": "EXTENDED",
                "cell_static_sha": static_result["cell_static_sha"],
            }
            observation_result = write_hdf5_shard(
                observation_path, observations, observation_metadata)
            truth_result = write_hdf5_shard(truth_path, truths, truth_metadata)
            audit_indices = sorted(set((
                record["start_index"],
                (record["start_index"] + record["stop_index"] - 1) // 2,
                record["stop_index"] - 1,
            )))
            audit_hashes = tuple(
                (sample_index, _row_sha(
                    full_observations, truths,
                    sample_index - record["start_index"]))
                for sample_index in audit_indices)
            ready_path = shard_dir / f"{shard_id}.READY.json"
            ready_payload = {
                "ready_kind": "v24_shard_ready_v1",
                "shard_id": shard_id,
                "start_index": record["start_index"],
                "stop_index": record["stop_index"],
                "row_count": record["stop_index"] - record["start_index"],
                "observation_path": observation_path.relative_to(self.root).as_posix(),
                "observation_sha": observation_result["sha256"],
                "truth_path": truth_path.relative_to(self.root).as_posix(),
                "truth_sha": truth_result["sha256"],
                "cell_static_path": static_result["relative_path"],
                "cell_static_sha": static_result["cell_static_sha"],
                "cell_static_file_sha": static_result["file_sha"],
                "contract_sha": self.plan.contract_sha,
                "schema_sha": self.plan.schema_sha,
                "generator_sha": self.plan.generator_sha,
                "build_id": self.plan.build_id,
                "audit_row_hashes": [list(item) for item in audit_hashes],
            }
            ready_sha = write_canonical_json(ready_path, ready_payload)
            self.ledger.transition(
                shard_id, V24ShardState.VALIDATED,
                observation_path=ready_payload["observation_path"],
                truth_path=ready_payload["truth_path"],
                ready_path=ready_path.relative_to(self.root).as_posix(),
                cell_static_sha=static_result["cell_static_sha"])
            self.ledger.transition(shard_id, V24ShardState.READY_LOCAL)
            entry = V24RangeIndexEntry(
                cell_id=self.plan.cell_id, split=self.plan.split,
                shard_id=shard_id, start_index=record["start_index"],
                stop_index=record["stop_index"],
                observation_path=ready_payload["observation_path"],
                observation_sha=observation_result["sha256"],
                truth_path=ready_payload["truth_path"],
                truth_sha=truth_result["sha256"],
                ready_path=ready_path.relative_to(self.root).as_posix(),
                ready_sha=ready_sha,
                cell_static_path=static_result["relative_path"],
                cell_static_sha=static_result["cell_static_sha"],
                cell_static_file_sha=static_result["file_sha"],
                build_id=self.plan.build_id,
                dataset_family=self.plan.dataset_family, N=self.plan.N,
                noise_family=self.plan.noise_family,
                eta=self.plan.eta, kappa=self.plan.kappa,
                shot_mode=self.plan.shot_mode,
                shot_budget=self.plan.shot_budget,
                measurement_schedule=self.plan.measurement_schedule,
                global_seed=self.plan.global_seed,
                generator_sha=self.plan.generator_sha,
                contract_sha=self.plan.contract_sha,
                schema_sha=self.plan.schema_sha,
                audit_row_hashes=audit_hashes,
            )
            index_result = append_range_entry(self.range_index_path, entry)
            return {**ready_payload, "ready_sha": ready_sha,
                    "range_index_sha": index_result["range_index_sha"],
                    "resumed": state == V24ShardState.GENERATING}

    def produce_until_window_full(self) -> Dict[str, Any]:
        produced = []
        for shard_id, _, _ in self.plan.shard_ranges():
            if self.ready_queue_depth() >= self.profile.shard_window_count:
                break
            state = V24ShardState(
                self.ledger.snapshot()["records"][shard_id]["state"])
            if state in (V24ShardState.PLANNED, V24ShardState.GENERATING):
                produced.append(self.produce_shard(shard_id))
        return {
            "produced_shards": len(produced),
            "ready_queue_depth": self.ready_queue_depth(),
            "window_count": self.profile.shard_window_count,
            "buffered_bytes": self._buffered_bytes(),
        }


def audit_regenerate_range_sample(local_root: str | Path,
                                  entry: V24RangeIndexEntry,
                                  sample_index: int) -> Dict[str, Any]:
    if not entry.start_index <= sample_index < entry.stop_index:
        raise ValueError("sample index is outside the audited shard range")
    key = make_sample_key(
        global_seed=entry.global_seed, split=entry.split,
        dataset_family=entry.dataset_family, N=entry.N, index=sample_index,
        generator_sha=entry.generator_sha, cell_id=entry.cell_id)
    regenerated = _render_dataset_family(key, cell={
        "noise_family": entry.noise_family,
        "eta": entry.eta,
        "kappa": entry.kappa,
        "shot_mode": entry.shot_mode,
        "shot_budget": entry.shot_budget,
        "measurement_schedule": entry.measurement_schedule,
    })
    regenerated_observation = _stack([regenerated["observation"]], [key.sample_id])
    regenerated_truth = _stack([regenerated["truth"]], [key.sample_id])
    regenerated_sha = _row_sha(regenerated_observation, regenerated_truth, 0)
    row = sample_index - entry.start_index
    cache = V24Hdf5HandleCache(maximum_handles=3)
    try:
        dynamic = cache.read_row(
            Path(local_root) / entry.observation_path, row,
            expected_sha=entry.observation_sha)
        static = cache.read_row(
            Path(local_root) / entry.cell_static_path, 0,
            expected_sha=entry.cell_static_file_sha)
        truth = cache.read_row(
            Path(local_root) / entry.truth_path, row,
            expected_sha=entry.truth_sha)
    finally:
        cache.close()
    reconstructed = dict(dynamic)
    reconstructed.update({
        name: value for name, value in static.items()
        if name not in reconstructed and name != "row_valid_mask"
    })
    stored_sha = _row_sha(
        {name: np.asarray(value)[None, ...] for name, value in reconstructed.items()},
        {name: np.asarray(value)[None, ...] for name, value in truth.items()}, 0)
    expected = dict(entry.audit_row_hashes).get(sample_index, "")
    if not expected:
        raise ValueError("sample is not in the bounded regeneration audit subset")
    if regenerated_sha != expected or stored_sha != expected:
        raise ValueError("V2.4 ephemeral regeneration is not bitwise identical")
    return {
        "sample_id": key.sample_id,
        "sample_index": sample_index,
        "expected_row_sha": expected,
        "regenerated_row_sha": regenerated_sha,
        "stored_row_sha": stored_sha,
        "bitwise_equal": True,
    }
