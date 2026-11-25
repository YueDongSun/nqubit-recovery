"""Range-indexed high-throughput reader for ephemeral synthetic V2.4 shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Dataset = object
    TORCH_AVAILABLE = False

from .contracts import NQubitDatasetContractV240
from ml_recon.physics_hybrid_v10.feedback_state_decode import (
    LEGACY_ANALYTIC_STATE_DECODE,
    PHYSICAL_ANALYTIC_STATE_DECODE,
    decode_analytic_state,
    validate_analytic_state_decode_version,
)
from .hashing import read_canonical_json, sha256_file, sha256_json
from .range_index import (
    V24RangeIndexEntry,
    V24SyntheticRangeIndex,
    read_range_index,
)
from .sample_keys import make_sample_key
from .schema_registry import schema_sha256
from .shard_reader import V24Hdf5HandleCache
from .version import GENERATOR_SHA
from .feedback_gate_contract import (
    FEEDBACK_GATE_BASE_DIM,
    FEEDBACK_GATE_DIM,
    FEEDBACK_GATE_ORACLE_DIM,
    compose_feedback_gate_features,
)
from .views import (
    CLASSIFIER_INPUT,
    CLASSIFIER_TRUTH,
    FEEDBACK_BASE,
    FEEDBACK_CONTEXT,
    FEEDBACK_TARGETS,
    ORACLE_INPUT,
)


def _tensor(value: np.ndarray) -> torch.Tensor:
    array = np.asarray(value)
    if array.dtype.kind in ("S", "U"):
        raise TypeError("string provenance fields are not model tensors")
    if array.dtype == np.float64:
        array = array.astype(np.float32)
    if array.dtype in (np.uint32, np.uint64):
        array = array.astype(np.int64)
    return torch.from_numpy(np.array(array, copy=True))


class V24EphemeralRangeDataset(Dataset):
    """Read shard ranges without materializing a per-sample Python index."""

    def __init__(self, local_root: str | Path, range_index_path: str | Path, *,
                 cell_id: str, split: str, include_truth: bool = True,
                 maximum_handles: int = 8, task: str = "classifier",
                 arm: str = "", classifier_checkpoint_sha: str = "",
                 classifier_cache_manifest_path: str | Path = "",
                 analytic_state_decode_version: str = LEGACY_ANALYTIC_STATE_DECODE):
        analytic_state_decode_version = validate_analytic_state_decode_version(
            analytic_state_decode_version)
        if (analytic_state_decode_version == PHYSICAL_ANALYTIC_STATE_DECODE
                and (task != "feedback" or arm != "A0_RAW")):
            raise ValueError(
                "physical analytic state decoding is registered only for feedback A0_RAW")
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "V24EphemeralRangeDataset requires torch; range identity "
                "inspection remains available without torch"
            )
        self.root = Path(local_root)
        self.range_index = V24SyntheticRangeIndex(
            range_index_path, cell_id=cell_id, split=split)
        self.include_truth = bool(include_truth)
        if task not in ("classifier", "feedback", "feedback_teacher"):
            raise ValueError("ephemeral range task is not registered")
        self.task = task
        self.arm = arm
        self.analytic_state_decode_version = analytic_state_decode_version
        self.classifier_checkpoint_sha = classifier_checkpoint_sha
        self.classifier_cache_entries: Dict[str, Dict[str, Any]] = {}
        if task == "feedback" and arm not in (
                "A0_RAW", "A1_DENOISE", "A2_CLASS",
                "A3_DENOISE_CLASS", "A4_ORACLE"):
            raise ValueError("feedback range dataset requires a registered A0-A4 arm")
        classifier_required = task == "feedback" and arm in (
            "A1_DENOISE", "A2_CLASS", "A3_DENOISE_CLASS")
        if classifier_required:
            if len(classifier_checkpoint_sha) != 64 or not classifier_cache_manifest_path:
                raise ValueError(
                    "A1-A3 range training requires a frozen classifier cache manifest")
            manifest = dict(read_canonical_json(classifier_cache_manifest_path))
            manifest_sha = str(manifest.pop("manifest_sha", ""))
            if sha256_json(manifest) != manifest_sha \
                    or manifest.get("manifest_kind") \
                    != "v24_frozen_classifier_range_cache_v1":
                raise ValueError("invalid frozen classifier range cache manifest")
            if manifest.get("classifier_checkpoint_sha") != classifier_checkpoint_sha \
                    or manifest.get("fixture_heuristic", True) \
                    or manifest.get("generator_kind") != "frozen_classifier_inference" \
                    or len(str(manifest.get("classifier_config_sha", ""))) != 64 \
                    or len(str(manifest.get("normalization_sha", ""))) != 64:
                raise ValueError("classifier range cache is not scientific-training eligible")
            self.classifier_cache_entries = {
                str(item["shard_id"]): dict(item) for item in manifest["entries"]
                if item["cell_id"] == cell_id and item["split"] == split
            }
            expected_shards = {entry.shard_id for entry in self.range_index.entries}
            if set(self.classifier_cache_entries) != expected_shards:
                raise ValueError("classifier range cache does not cover the selected shards")
        self.handle_cache = V24Hdf5HandleCache(maximum_handles)
        self._verified_ready = set()
        expected = (
            NQubitDatasetContractV240.default().sha256(), schema_sha256(), GENERATOR_SHA)
        for entry in self.range_index.entries:
            if (entry.contract_sha, entry.schema_sha, entry.generator_sha) != expected:
                raise ValueError("ephemeral range dependency mismatch")

    def __len__(self) -> int:
        return len(self.range_index)

    @property
    def sample_count(self) -> int:
        """Total number of rows across all range entries."""
        return sum(entry.row_count for entry in self.range_index.entries)

    @property
    def sample_range_sha(self) -> str:
        """Deterministic hash of the range entries defining this dataset's sample set."""
        return sha256_json([entry.to_dict() for entry in self.range_index.entries])

    @property
    def sample_id_manifest_sha(self) -> str:
        """Deterministic hash of all sample IDs in this dataset.

        Computed from the range-index metadata (global_seed, split, N,
        generator_sha, cell_id, start/stop index) without materializing
        the full ID list.
        """
        parts = []
        for entry in self.range_index.entries:
            parts.append({
                "cell_id": entry.cell_id,
                "split": entry.split,
                "global_seed": entry.global_seed,
                "N": entry.N,
                "dataset_family": entry.dataset_family,
                "generator_sha": entry.generator_sha,
                "start_index": entry.start_index,
                "stop_index": entry.stop_index,
            })
        return sha256_json(parts)

    def _verify_ready(self, entry: V24RangeIndexEntry) -> None:
        if entry.shard_id in self._verified_ready:
            return
        ready_path = self.root / entry.ready_path
        if sha256_file(ready_path) != entry.ready_sha:
            raise ValueError("ephemeral shard READY SHA mismatch")
        ready = read_canonical_json(ready_path)
        if ready["shard_id"] != entry.shard_id \
                or ready["observation_sha"] != entry.observation_sha \
                or ready["truth_sha"] != entry.truth_sha \
                or ready["cell_static_sha"] != entry.cell_static_sha:
            raise ValueError("ephemeral shard READY identity mismatch")
        self._verified_ready.add(entry.shard_id)

    def _compose(self, entry: V24RangeIndexEntry, sample_index: int,
                  observation: Dict[str, np.ndarray],
                  static: Dict[str, np.ndarray],
                  truth: Dict[str, np.ndarray] | None,
                  classifier: Dict[str, np.ndarray] | None = None) -> Dict[str, Any]:
        if self.task != "classifier":
            if truth is None:
                raise ValueError("feedback range datasets require truth labels")
            values = {
                name: value for name, value in {**static, **observation}.items()
                if name in FEEDBACK_CONTEXT
            }
            if self.task == "feedback_teacher" or self.arm == "A4_ORACLE":
                # A4/teacher: preserve actual noisy measurement, provide
                # clean measurement in denoised slot, and use oracle gate context.
                values["noisy_measurement_tokens"] = observation[
                    "noisy_measurement_tokens"]
                values["denoised_measurement_tokens"] = truth[
                    "clean_measurement_tokens"]
                values["covariance"] = observation.get(
                    "measurement_covariance",
                    np.zeros_like(observation["noisy_measurement_tokens"]))
                values.update({name: truth[name] for name in ORACLE_INPUT})
                values["esm_mode_tokens"] = truth["true_esm_mode_tokens"]
                values["analytic_action"] = truth["oracle_action"]
                values["gate_features"] = compose_feedback_gate_features(
                    np.asarray(values["gate_features"], dtype=np.float32),
                    arm=self.arm if self.arm == "A4_ORACLE" else "feedback_teacher",
                    true_mechanism=truth.get("true_mechanism"),
                    true_severity=float(truth["true_severity"]) if "true_severity" in truth else None,
                )
                if self.task == "feedback_teacher":
                    # Update with observation fields EXCEPT gate_features
                    # (already composed to 24-dim above).
                    values.update({
                        name: value for name, value in observation.items()
                        if name in FEEDBACK_BASE and name != "gate_features"
                    })
                    values.update({name: truth[name] for name in (
                        ORACLE_INPUT | {"true_next_measurement", "true_kappa",
                                        "epsilon_irr"})})
            else:
                classifier = classifier or {}
                denoised_arm = self.arm in ("A1_DENOISE", "A3_DENOISE_CLASS")
                class_arm = self.arm in ("A2_CLASS", "A3_DENOISE_CLASS")
                active = (classifier["denoised_measurement_tokens"]
                          if denoised_arm else observation["noisy_measurement_tokens"])
                values["noisy_measurement_tokens"] = observation[
                    "noisy_measurement_tokens"]
                covariance = (classifier["covariance"] if denoised_arm
                              else observation["measurement_covariance"])
                values["denoised_measurement_tokens"] = active
                values["covariance"] = covariance
                values["esm_mode_tokens"] = (
                    classifier["esm_mode_tokens"] if class_arm
                    else np.zeros_like(active))
                estimated = decode_analytic_state(
                    active, version=self.analytic_state_decode_version)
                target_tokens = np.asarray(values["target_mode_tokens"])
                target = np.maximum(target_tokens[:, 0], 1e-8).astype(np.complex128)
                target *= np.exp(1j * np.pi * target_tokens[:, 1])
                target /= np.linalg.norm(target)
                from .synthetic.feedback_renderers import _deployable_analytic_action
                values["analytic_action"] = _deployable_analytic_action(
                    estimated, target, covariance)[0].astype(np.float32)
                values["gate_features"] = compose_feedback_gate_features(
                    np.asarray(values["gate_features"], dtype=np.float32),
                    arm=self.arm)
            values.update({name: truth[name] for name in FEEDBACK_TARGETS})
            key = make_sample_key(
                global_seed=entry.global_seed, split=entry.split,
                dataset_family=entry.dataset_family, N=entry.N,
                index=sample_index, generator_sha=entry.generator_sha,
                cell_id=entry.cell_id)
            return {**{name: _tensor(value) for name, value in values.items()},
                    "sample_id": key.sample_id}

        values = {
            name: value for name, value in observation.items()
            if name in CLASSIFIER_INPUT
        }
        values.update({
            name: value for name, value in static.items()
            if name in CLASSIFIER_INPUT and name not in values
        })
        if truth is not None:
            values.update({
                name: value for name, value in truth.items()
                if name in CLASSIFIER_TRUTH
            })
        key = make_sample_key(
            global_seed=entry.global_seed, split=entry.split,
            dataset_family=entry.dataset_family, N=entry.N,
            index=sample_index, generator_sha=entry.generator_sha,
            cell_id=entry.cell_id)
        return {
            **{name: _tensor(value) for name, value in values.items()},
            "sample_id": key.sample_id,
        }

    def _read(self, entry: V24RangeIndexEntry, row: int,
              sample_index: int) -> Dict[str, Any]:
        self._verify_ready(entry)
        observation = self.handle_cache.read_row(
            self.root / entry.observation_path, row,
            expected_sha=entry.observation_sha)
        static = self.handle_cache.read_row(
            self.root / entry.cell_static_path, 0,
            expected_sha=entry.cell_static_file_sha)
        truth = (self.handle_cache.read_row(
            self.root / entry.truth_path, row,
            expected_sha=entry.truth_sha) if self.include_truth else None)
        classifier = None
        if entry.shard_id in self.classifier_cache_entries:
            cache = self.classifier_cache_entries[entry.shard_id]
            classifier = self.handle_cache.read_row(
                self.root / cache["path"], row,
                expected_sha=cache["sha256"])
        return self._compose(
            entry, sample_index, observation, static, truth, classifier)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry, row, sample_index = self.range_index.locate(index)
        return self._read(entry, row, sample_index)

    def __getitems__(self, indices: Sequence[int]) -> list[Dict[str, Any]]:
        """PyTorch batched-fetch hook using one HDF5 block read per shard."""
        locations = [self.range_index.locate(index) for index in indices]
        grouped: Dict[str, list[tuple[int, V24RangeIndexEntry, int, int]]] = {}
        for position, (entry, row, sample_index) in enumerate(locations):
            grouped.setdefault(entry.shard_id, []).append(
                (position, entry, row, sample_index))
        results: list[Dict[str, Any] | None] = [None] * len(indices)
        for group in grouped.values():
            entry = group[0][1]
            self._verify_ready(entry)
            rows = [item[2] for item in group]
            observation_batch = self.handle_cache.read_rows(
                self.root / entry.observation_path, rows,
                expected_sha=entry.observation_sha)
            truth_batch = (self.handle_cache.read_rows(
                self.root / entry.truth_path, rows,
                expected_sha=entry.truth_sha) if self.include_truth else None)
            static = self.handle_cache.read_row(
                self.root / entry.cell_static_path, 0,
                expected_sha=entry.cell_static_file_sha)
            classifier_batch = None
            if entry.shard_id in self.classifier_cache_entries:
                cache = self.classifier_cache_entries[entry.shard_id]
                classifier_batch = self.handle_cache.read_rows(
                    self.root / cache["path"], rows,
                    expected_sha=cache["sha256"])
            for offset, (position, _, _, sample_index) in enumerate(group):
                observation = {
                    name: value[offset] for name, value in observation_batch.items()}
                truth = ({name: value[offset] for name, value in truth_batch.items()}
                         if truth_batch is not None else None)
                classifier = ({name: value[offset]
                               for name, value in classifier_batch.items()}
                              if classifier_batch is not None else None)
                results[position] = self._compose(
                    entry, sample_index, observation, static, truth, classifier)
        return [item for item in results if item is not None]

    def close(self) -> None:
        self.handle_cache.close()


def close_ephemeral_datasets(datasets: Iterable[V24EphemeralRangeDataset]) -> None:
    for dataset in datasets:
        dataset.close()


def discover_range_index_path(
    local_root: str | Path,
    *,
    cell_id: str,
    split: str,
) -> Path:
    """Find the sole range index that owns a requested cell/split."""

    root = Path(local_root)
    matches = []
    for path in sorted((root / "index").glob("*_ranges.json")):
        if any(
            entry.cell_id == cell_id and entry.split == split
            for entry in read_range_index(path)
        ):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {cell_id}:{split} range index, got {len(matches)}")
    return matches[0]


def range_training_identity(
    local_root: str | Path,
    *,
    train_range_index_path: str | Path,
    validation_range_index_path: str | Path,
    cell_id: str,
) -> Dict[str, Any]:
    """Bind one train/validation cell to immutable range-indexed shards."""

    root = Path(local_root)
    expected = (
        NQubitDatasetContractV240.default().sha256(), schema_sha256(), GENERATOR_SHA)
    split_entries: Dict[str, tuple[V24RangeIndexEntry, ...]] = {}
    range_index_hashes: Dict[str, str] = {}
    for split, source in (
        ("train", Path(train_range_index_path)),
        ("validation", Path(validation_range_index_path)),
    ):
        path = source if source.is_absolute() or source.is_file() else root / source
        entries = tuple(
            entry for entry in read_range_index(path)
            if entry.cell_id == cell_id and entry.split == split)
        if not entries:
            raise ValueError(f"range materialization selected zero {split} rows")
        if any(
            (entry.contract_sha, entry.schema_sha, entry.generator_sha) != expected
            for entry in entries
        ):
            raise ValueError("range materialization dependency mismatch")
        split_entries[split] = entries
        range_index_hashes[split] = sha256_file(path)

    cell_split_hashes = {
        split: sha256_json([entry.to_dict() for entry in entries])
        for split, entries in split_entries.items()
    }
    split_sample_counts = {
        split: sum(entry.row_count for entry in entries)
        for split, entries in split_entries.items()
    }
    split_sample_range_shas = {
        split: sha256_json([entry.to_dict() for entry in entries])
        for split, entries in split_entries.items()
    }
    split_sample_id_shas = {
        split: sha256_json([{
            "cell_id": entry.cell_id, "split": entry.split,
            "global_seed": entry.global_seed, "N": entry.N,
            "dataset_family": entry.dataset_family,
            "generator_sha": entry.generator_sha,
            "start_index": entry.start_index, "stop_index": entry.stop_index,
        } for entry in entries])
        for split, entries in split_entries.items()
    }
    materialization_payload = {
        "identity_kind": "v24_range_training_materialization_v1",
        "cell_id": cell_id,
        "contract_sha": expected[0],
        "schema_sha": expected[1],
        "generator_sha": expected[2],
        "range_index_hashes": range_index_hashes,
        "cell_split_hashes": cell_split_hashes,
        "split_sample_counts": split_sample_counts,
        "split_sample_range_shas": split_sample_range_shas,
        "split_sample_id_shas": split_sample_id_shas,
        "split_build_ids": {
            split: sorted({entry.build_id for entry in entries})
            for split, entries in split_entries.items()
        },
        "split_global_seeds": {
            split: sorted({entry.global_seed for entry in entries})
            for split, entries in split_entries.items()
        },
    }
    materialization_sha = sha256_json(materialization_payload)
    return {
        **materialization_payload,
        "materialization_sha": materialization_sha,
        "build_id": f"v24_range_{materialization_sha[:20]}",
        "index_sha": sha256_json(range_index_hashes),
        "split_sha": sha256_json(cell_split_hashes),
        "build_manifest_sha": materialization_sha,
        "train_count": sum(entry.row_count for entry in split_entries["train"]),
        "validation_count": sum(
            entry.row_count for entry in split_entries["validation"]),
    }
