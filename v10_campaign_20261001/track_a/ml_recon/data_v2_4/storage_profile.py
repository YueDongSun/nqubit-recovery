"""Empirical per-cell storage and throughput profiling for V2.4."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import h5py
import numpy as np

from .cell_static import cell_static_logical_sha, split_cell_static_fields
from .contracts import NQubitDatasetContractV240
from .hashing import canonical_json_bytes, sha256_json, write_canonical_json
from .materialization import V24MaterializationProfileV1
from .rung_plan import V24RungPlanV2
from .sample_keys import make_sample_key
from .schema_registry import schema_sha256
from .shard_reader import V24Hdf5HandleCache
from .storage_hdf5 import write_hdf5_shard
from .synthetic.generator import _render_dataset_family, _stack
from .version import GENERATOR_SHA


PROFILE_CODECS = ("gzip", "lzf", "none")
EMPIRICAL_STORAGE_PROFILE_REPORT_KIND = "v24_empirical_storage_profile_v2"


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _read_benchmark(paths: Sequence[Path]) -> Dict[str, float]:
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    bytes_read = 0
    rows = 0
    for path in paths:
        with h5py.File(path, "r", swmr=True) as handle:
            for dataset in handle.values():
                value = np.asarray(dataset[:])
                bytes_read += value.nbytes
            rows += int(next(iter(handle.values())).shape[0])
    wall_s = max(time.perf_counter() - wall_start, 1e-9)
    cpu_s = max(time.process_time() - cpu_start, 0.0)
    cache = V24Hdf5HandleCache(maximum_handles=4)
    random_start = time.perf_counter()
    probes = 0
    for path in paths:
        with h5py.File(path, "r") as handle:
            row_count = int(next(iter(handle.values())).shape[0])
        for row in range(min(row_count, 32)):
            cache.read_row(path, row)
            probes += 1
    random_s = max(time.perf_counter() - random_start, 1e-9)
    cache.close()
    return {
        "sequential_read_mib_s": bytes_read / 1024**2 / wall_s,
        "random_read_samples_s": probes / random_s,
        "decompression_cpu_s": cpu_s,
        "read_wall_s": wall_s,
        "rows_read": rows,
    }


def _profile_cell(cell: Mapping[str, Any], *, samples_per_cell: int,
                  global_seed: int, temporary_root: Path,
                  chunk_rows: int = 128) -> Dict[str, Any]:
    if samples_per_cell < 1:
        raise ValueError("storage profile needs at least one sample per cell")
    cell_id = str(cell["cell_id"])
    family = str(cell["dataset_family"])
    N = int(cell["N"])
    contract_sha = NQubitDatasetContractV240.default().sha256()
    schema_sha = schema_sha256()
    raw_dynamic_per_row = []
    raw_derived_per_row = []
    chunks: list[tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]] = []
    static_sha = ""
    generation_start = time.perf_counter()
    for start in range(0, samples_per_cell, chunk_rows):
        stop = min(start + chunk_rows, samples_per_cell)
        rendered = []
        sample_ids = []
        for index in range(start, stop):
            key = make_sample_key(
                global_seed=global_seed, split="train", dataset_family=family,
                N=N, index=index, generator_sha=GENERATOR_SHA, cell_id=cell_id)
            rendered.append(_render_dataset_family(key, cell=cell))
            sample_ids.append(key.sample_id)
        full_observations = _stack(
            [item["observation"] for item in rendered], sample_ids)
        truths = _stack([item["truth"] for item in rendered], sample_ids)
        static_arrays, observations = split_cell_static_fields(full_observations)
        current_static_sha = cell_static_logical_sha(
            static_arrays, cell_id=cell_id,
            contract_sha=contract_sha, schema_sha=schema_sha)
        if static_sha and static_sha != current_static_sha:
            raise ValueError("cell-static content changed across profile chunks")
        static_sha = current_static_sha
        for row in range(stop - start):
            raw_dynamic_per_row.append(sum(
                np.asarray(value[row]).nbytes
                for value in (*observations.values(), *truths.values())))
            raw_derived_per_row.append(sum(
                np.asarray(value).nbytes
                for cache_name in ("derived_classifier", "derived_control")
                for value in rendered[row].get(cache_name, {}).values()))
        chunks.append((observations, truths, static_arrays))
    generation_s = max(time.perf_counter() - generation_start, 1e-9)
    codec_reports: Dict[str, Dict[str, float | int]] = {}
    for codec in PROFILE_CODECS:
        codec_dir = temporary_root / cell_id / codec
        paths = []
        write_start = time.perf_counter()
        for chunk_index, (observations, truths, _) in enumerate(chunks):
            metadata = {
                "provenance": "synthetic", "dataset_family": family,
                "cell_id": cell_id, "split": "train", "system_type": (
                    f"{N}qubit" if N in (3, 4) else "nqubit"),
                "N": N, "measurement_domain": "global_pauli_v24",
                "contract_sha": contract_sha, "schema_sha": schema_sha,
                "generator_sha": GENERATOR_SHA, "build_id": "storage_profile",
            }
            observation_path = codec_dir / f"chunk_{chunk_index:04d}_observation.h5"
            truth_path = codec_dir / f"chunk_{chunk_index:04d}_truth.h5"
            write_hdf5_shard(
                observation_path, observations,
                {**metadata, "store": "observation"}, compression=codec)
            write_hdf5_shard(
                truth_path, truths, {**metadata, "store": "truth"},
                compression=codec)
            paths.extend((observation_path, truth_path))
        static_path = codec_dir / "cell_static.h5"
        write_hdf5_shard(
            static_path, chunks[0][2], {
                "provenance": "synthetic", "dataset_family": family,
                "cell_id": cell_id, "system_type": (
                    f"{N}qubit" if N in (3, 4) else "nqubit"),
                "N": N, "measurement_domain": "global_pauli_v24",
                "contract_sha": contract_sha, "schema_sha": schema_sha,
                "generator_sha": GENERATOR_SHA, "build_id": "storage_profile",
                "store": "cell_static",
            }, compression=codec)
        write_s = max(time.perf_counter() - write_start, 1e-9)
        dynamic_bytes = sum(path.stat().st_size for path in paths)
        benchmark = _read_benchmark(paths)
        codec_reports[codec] = {
            "compressed_dynamic_bytes": dynamic_bytes,
            "compressed_dynamic_bytes_per_sample": dynamic_bytes / samples_per_cell,
            "static_bytes_per_cell": static_path.stat().st_size,
            "write_mib_s": (dynamic_bytes + static_path.stat().st_size) / 1024**2 / write_s,
            **benchmark,
        }
    minimum_size = min(
        float(value["compressed_dynamic_bytes_per_sample"])
        for value in codec_reports.values())
    maximum_read = max(
        float(value["sequential_read_mib_s"]) for value in codec_reports.values())
    selected_codec = min(PROFILE_CODECS, key=lambda name: (
        float(codec_reports[name]["compressed_dynamic_bytes_per_sample"]) / minimum_size
        + maximum_read / max(float(codec_reports[name]["sequential_read_mib_s"]), 1e-9)))
    selected = codec_reports[selected_codec]
    range_payload = {
        "cell_id": cell_id, "split": "train", "shard_id": "profile",
        "start_index": 0, "stop_index": samples_per_cell,
        "observation_path": "x", "truth_path": "y",
        "cell_static_sha": static_sha,
    }
    return {
        "cell_id": cell_id, "dataset_family": family, "N": N,
        "samples": samples_per_cell,
        "raw_dynamic_bytes_per_sample": _percentiles(raw_dynamic_per_row),
        "derived_bytes_per_sample": _percentiles(raw_derived_per_row),
        "selected_compression": selected_codec,
        "compressed_observation_truth_bytes_per_sample": selected[
            "compressed_dynamic_bytes_per_sample"],
        "static_bytes_per_cell": selected["static_bytes_per_cell"],
        "range_index_bytes_per_sample": len(canonical_json_bytes(range_payload)) / samples_per_cell,
        "generation_temp_peak_bytes": max(raw_dynamic_per_row) * min(
            samples_per_cell, chunk_rows) + int(selected["compressed_dynamic_bytes"]),
        "generation_samples_s": samples_per_cell / generation_s,
        "write_mib_s": selected["write_mib_s"],
        "sequential_read_mib_s": selected["sequential_read_mib_s"],
        "random_read_samples_s": selected["random_read_samples_s"],
        "decompression_cpu_s": selected["decompression_cpu_s"],
        "cell_static_sha": static_sha,
        "compression_trials": codec_reports,
    }


def profile_storage(plan: V24RungPlanV2, *, samples_per_cell: int,
                    output: str | Path, global_seed: int = 20261001,
                    materialization_profile: V24MaterializationProfileV1 | None = None
                    ) -> Dict[str, Any]:
    if type(global_seed) is not int:
        raise ValueError("profile global_seed must be an integer, not a coerced value")
    if type(samples_per_cell) is not int or samples_per_cell <= 0:
        raise ValueError("samples_per_cell must be a positive integer")
    plan.validate()
    profile = materialization_profile or (
        V24MaterializationProfileV1.persistent_eval_ephemeral_train())
    profile.validate()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix="v24_storage_profile_", dir=output_path.parent) as temporary:
        temporary_root = Path(temporary)
        cells = [
            _profile_cell(cell.to_generator_cell(), samples_per_cell=samples_per_cell,
                          global_seed=global_seed, temporary_root=temporary_root)
            for cell in plan.cell_allocations
        ]
    usage = shutil.disk_usage(output_path.parent)
    p95_dynamic = max(
        int(np.ceil(cell["compressed_observation_truth_bytes_per_sample"]))
        for cell in cells)
    persistent_fixed = sum(int(cell["static_bytes_per_cell"]) for cell in cells)
    target_shard_bytes = profile.target_shard_mib * 1024**2
    ephemeral_peak = profile.shard_window_count * target_shard_bytes * 3
    reserve = profile.reserve_bytes(int(usage.total))
    temporary_one_shard = target_shard_bytes * 3
    required_peak = persistent_fixed + ephemeral_peak + temporary_one_shard + reserve
    payload = {
        "report_kind": EMPIRICAL_STORAGE_PROFILE_REPORT_KIND,
        "rung_plan_sha": plan.sha256(),
        "materialization_profile_sha": profile.sha256(),
        "materialization_profile": profile.to_dict(),
        "generator_sha": GENERATOR_SHA,
        "contract_sha": NQubitDatasetContractV240.default().sha256(),
        "schema_sha": schema_sha256(),
        "samples_per_cell": int(samples_per_cell),
        "global_seed": int(global_seed),
        "cells": cells,
        "p95_dynamic_bytes_per_sample": p95_dynamic,
        "persistent_fixed_bytes": persistent_fixed,
        "ephemeral_peak_bytes": ephemeral_peak,
        "temporary_one_shard_bytes": temporary_one_shard,
        "reserve_bytes": reserve,
        "required_peak_bytes": required_peak,
        "filesystem_capacity_bytes": int(usage.total),
        "available_bytes": int(usage.free),
        "within_budget": required_peak <= int(usage.free),
    }
    result = {"profile_sha": sha256_json(payload), **payload}
    write_canonical_json(output_path, result)
    return result
