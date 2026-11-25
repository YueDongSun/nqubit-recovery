"""Fail-closed disk datasets for V2.4 observation and training views."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import load_contract
from .feedback_gate_contract import compose_feedback_gate_features
from .hashing import read_canonical_json, sha256_file, sha256_json
from .index_sqlite import query_index, validate_index
from .schema_registry import schema_sha256
from .shard_reader import V24Hdf5HandleCache
from .storage_hdf5 import validate_hdf5_shard
from .views import (
    CLASSIFIER_INPUT,
    CLASSIFIER_TRUTH,
    FEEDBACK_BASE,
    FEEDBACK_CONTEXT,
    FEEDBACK_TARGETS,
    ORACLE_INPUT,
    TEACHER_INPUT,
    classifier_input_for_family,
    fields_for_view,
)


def validate_dataset_root(dataset_root: str | Path, *, build_id: Optional[str] = None,
                          strict_shard_audit: bool = False
                          ) -> Dict[str, Any]:
    root = Path(dataset_root)
    ready_path = root / "READY.json"
    if not ready_path.exists():
        raise ValueError("V2.4 loader requires READY.json")
    contract = load_contract(root)
    ready = read_canonical_json(ready_path)
    if build_id is not None and ready["build_id"] != build_id:
        raise ValueError("V2.4 dataset build_id mismatch")
    if ready["contract_sha"] != contract.sha256() or ready["schema_sha"] != schema_sha256():
        raise ValueError("V2.4 contract/schema SHA mismatch")
    manifest_path = root / "manifests" / f"{ready['build_id']}.json"
    manifest = read_canonical_json(manifest_path)
    if sha256_file(manifest_path) != ready["build_manifest_sha"]:
        raise ValueError("V2.4 build manifest SHA mismatch")
    if manifest["contract_sha"] != contract.sha256() or manifest["schema_sha"] != schema_sha256():
        raise ValueError("V2.4 build manifest is bound to another contract/schema")
    index_path = root / "index" / "index.sqlite"
    index = validate_index(index_path, ready["index_sha"])
    for relative, expected_sha in ready["shard_hashes"].items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"V2.4 READY shard is missing: {relative}")
        if strict_shard_audit:
            if sha256_file(path) != expected_sha:
                raise ValueError(f"V2.4 shard SHA mismatch: {relative}")
            validate_hdf5_shard(path)
    return {"build_id": ready["build_id"], "contract_sha": contract.sha256(),
            "schema_sha": schema_sha256(), "index_sha": index["sha256"],
            "split_sha": ready["split_sha"], "build_manifest_sha": ready["build_manifest_sha"],
            "generator_sha": manifest["generator_sha"],
            "shard_hashes": dict(ready["shard_hashes"]),
            "source_commit": manifest["source_commit"],
            "source_tree": manifest["source_tree_sha"],
            "rows": index["rows"], "ready_sha": sha256_file(ready_path)}


def _to_tensor(value: np.ndarray) -> torch.Tensor:
    array = np.asarray(value)
    if array.dtype.kind in ("S", "U"):
        raise TypeError("string provenance fields are not model tensors")
    if array.dtype == np.float64:
        array = array.astype(np.float32)
    if array.dtype in (np.uint32, np.uint64):
        array = array.astype(np.int64)
    return torch.from_numpy(np.array(array, copy=True))


class V24ObservationDataset(Dataset):
    """Observation-only reader. It never opens the truth or derived stores."""

    def __init__(self, dataset_root: str | Path, *, build_id: str, cell_id: str,
                 split: str, view_name: str = "classifier_input"):
        self.root = Path(dataset_root)
        self.identity = validate_dataset_root(self.root, build_id=build_id)
        self.rows = query_index(self.root / "index" / "index.sqlite",
                                cell_id=cell_id, split=split)
        if not self.rows:
            raise ValueError("V2.4 observation view selected zero rows")
        self.view_name = view_name
        if view_name == "classifier_input":
            families = {str(row["dataset_family"]) for row in self.rows}
            if len(families) != 1:
                raise ValueError("V2.4 observation view mixes dataset families")
            selected_fields = classifier_input_for_family(families.pop())
        else:
            selected_fields = fields_for_view(view_name)
        self.fields = tuple(sorted(selected_fields & (CLASSIFIER_INPUT | FEEDBACK_BASE)))
        if not self.fields:
            raise ValueError("selected view has no observation fields")
        self.sample_ids = tuple(row["sample_id"] for row in self.rows)
        self.handle_cache = V24Hdf5HandleCache(maximum_handles=8)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        relative = row["observation_shard"]
        values = self.handle_cache.read_row(
            self.root / relative, int(row["observation_row"]), self.fields,
            expected_sha=self.identity["shard_hashes"][relative])
        return {**{key: _to_tensor(value) for key, value in values.items()},
                "sample_id": row["sample_id"]}

    def close(self) -> None:
        self.handle_cache.close()


class V24TrainingDataset(Dataset):
    """Trusted composition of named observation/truth/derived views for training."""

    def __init__(self, dataset_root: str | Path, *, build_id: str, cell_id: str,
                 split: str, task: str, arm: str = "A3_DENOISE_CLASS",
                 derived_checkpoint_sha: str = "",
                 require_frozen_classifier_cache: bool = False):
        self.root = Path(dataset_root)
        self.identity = validate_dataset_root(self.root, build_id=build_id)
        self.rows = query_index(self.root / "index" / "index.sqlite",
                                cell_id=cell_id, split=split)
        if not self.rows:
            raise ValueError("V2.4 training view selected zero rows")
        if task not in ("classifier", "feedback"):
            raise ValueError("V2.4 training task must be classifier or feedback")
        self.task = task
        self.arm = arm
        self.split = split
        self.cell_id = cell_id
        self.sample_ids = tuple(row["sample_id"] for row in self.rows)
        self.handle_cache = V24Hdf5HandleCache(maximum_handles=8)
        validation_path = self.root / "reports" / f"{build_id}_validation.json"
        self.validation = read_canonical_json(validation_path)
        self.classifier_checkpoint_sha = self.validation.get("classifier_checkpoint_sha", "")
        self.control_forward_sha = self.validation.get("control_forward_sha", "")
        if task == "feedback":
            if arm not in ("A0_RAW", "A1_DENOISE", "A2_CLASS", "A3_DENOISE_CLASS", "A4_ORACLE"):
                raise ValueError("invalid V2.4 feedback ablation arm")
            classifier_required = arm in (
                "A1_DENOISE", "A2_CLASS", "A3_DENOISE_CLASS")
            if classifier_required and derived_checkpoint_sha != self.classifier_checkpoint_sha:
                raise ValueError("derived classifier checkpoint SHA mismatch")
            if classifier_required and require_frozen_classifier_cache:
                from .storage_hdf5 import validate_hdf5_shard
                path = (self.root / "derived" / "classifier"
                        / self.classifier_checkpoint_sha / f"{cell_id}_{split}.h5")
                metadata = validate_hdf5_shard(path)["metadata"]
                if metadata.get("generator_kind") != "frozen_classifier_inference" \
                        or metadata.get("fixture_heuristic", True) \
                        or len(str(metadata.get("classifier_config_sha", ""))) != 64 \
                        or len(str(metadata.get("normalization_sha", ""))) != 64:
                    raise ValueError(
                        "scientific feedback training requires frozen classifier inference cache")
        elif derived_checkpoint_sha and derived_checkpoint_sha != self.classifier_checkpoint_sha:
            raise ValueError("derived checkpoint SHA mismatch")

    def __len__(self) -> int:
        return len(self.rows)

    def _derived(self, kind: str, identity: str, row_index: int,
                 fields: tuple[str, ...]) -> Dict[str, np.ndarray]:
        path = self.root / "derived" / kind / identity / f"{self.cell_id}_{self.split}.h5"
        if not path.exists():
            raise ValueError(f"required V2.4 derived cache is missing: {kind}/{identity}")
        relative = path.relative_to(self.root).as_posix()
        expected_sha = self.identity["shard_hashes"].get(relative, "")
        if not expected_sha:
            raise ValueError(f"derived shard is absent from READY identity: {relative}")
        return self.handle_cache.read_row(
            path, row_index, fields, expected_sha=expected_sha)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        observation_relative = row["observation_shard"]
        truth_relative = row["truth_shard"]
        values: Dict[str, np.ndarray]
        if self.task == "classifier":
            classifier_fields = classifier_input_for_family(
                str(row["dataset_family"]))
            observation = self.handle_cache.read_row(
                self.root / observation_relative, int(row["observation_row"]),
                tuple(sorted(classifier_fields)),
                expected_sha=self.identity["shard_hashes"][observation_relative])
            truth = self.handle_cache.read_row(
                self.root / truth_relative, int(row["truth_row"]),
                tuple(sorted(CLASSIFIER_TRUTH)),
                expected_sha=self.identity["shard_hashes"][truth_relative])
            values = dict(observation)
            values.update(truth)
        else:
            # Every ablation keeps the same raw measurement context. Derived
            # denoising and oracle clean views are additional fields, never a
            # silent replacement of the observed sample.
            observation_fields = tuple(sorted(FEEDBACK_BASE))
            observation = self.handle_cache.read_row(
                self.root / observation_relative, int(row["observation_row"]),
                observation_fields,
                expected_sha=self.identity["shard_hashes"][observation_relative])
            truth_fields = set(FEEDBACK_TARGETS)
            if self.arm == "A4_ORACLE":
                truth_fields.update(ORACLE_INPUT)
            truth = self.handle_cache.read_row(
                self.root / truth_relative, int(row["truth_row"]),
                tuple(sorted(truth_fields)),
                expected_sha=self.identity["shard_hashes"][truth_relative])
            values = {key: value for key, value in observation.items()
                      if key in FEEDBACK_CONTEXT}
            classifier: Dict[str, np.ndarray] = {}
            if self.arm in ("A1_DENOISE", "A2_CLASS", "A3_DENOISE_CLASS"):
                classifier_fields = []
                if self.arm in ("A1_DENOISE", "A3_DENOISE_CLASS"):
                    classifier_fields.extend(("denoised_measurement_tokens", "covariance"))
                if self.arm in ("A2_CLASS", "A3_DENOISE_CLASS"):
                    classifier_fields.append("esm_mode_tokens")
                classifier = self._derived(
                    "classifier", self.classifier_checkpoint_sha, index,
                    tuple(classifier_fields))
            if self.arm == "A4_ORACLE":
                # A4: preserve actual noisy measurement, provide clean in denoised slot.
                values["noisy_measurement_tokens"] = observation[
                    "noisy_measurement_tokens"]
                values["denoised_measurement_tokens"] = truth[
                    "clean_measurement_tokens"]
                values["covariance"] = observation.get(
                    "measurement_covariance",
                    np.zeros_like(observation["noisy_measurement_tokens"]))
                values["esm_mode_tokens"] = truth["true_esm_mode_tokens"]
                values["analytic_action"] = truth["oracle_action"]
                # Use the SAME canonical gate builder as the ephemeral loader.
                # Base gate is the observed (noisy) gate, NOT clean-derived.
                values["gate_features"] = compose_feedback_gate_features(
                    np.asarray(values["gate_features"], dtype=np.float32),
                    arm="A4_ORACLE",
                    true_mechanism=truth["true_mechanism"],
                    true_severity=float(truth["true_severity"]),
                )
                values.update({key: truth[key] for key in ORACLE_INPUT})
            else:
                active_measurement = (classifier["denoised_measurement_tokens"]
                                      if self.arm in ("A1_DENOISE", "A3_DENOISE_CLASS")
                                      else observation["noisy_measurement_tokens"])
                values["noisy_measurement_tokens"] = observation[
                    "noisy_measurement_tokens"]
                values["denoised_measurement_tokens"] = active_measurement
                values["covariance"] = (classifier["covariance"]
                                         if self.arm in ("A1_DENOISE", "A3_DENOISE_CLASS")
                                         else observation["measurement_covariance"])
                values["esm_mode_tokens"] = (classifier["esm_mode_tokens"]
                    if self.arm in ("A2_CLASS", "A3_DENOISE_CLASS")
                    else np.zeros_like(active_measurement))
                population = np.maximum(active_measurement[:, 0], 1e-8)
                population /= population.sum()
                estimated = np.sqrt(population).astype(np.complex128)
                estimated *= np.exp(1j * np.pi * active_measurement[:, 1])
                target_tokens = np.asarray(values["target_mode_tokens"])
                target = np.maximum(target_tokens[:, 0], 1e-8).astype(np.complex128)
                target *= np.exp(1j * np.pi * target_tokens[:, 1])
                target /= np.linalg.norm(target)
                from .synthetic.feedback_renderers import _deployable_analytic_action
                values["analytic_action"] = _deployable_analytic_action(
                    estimated, target, values["covariance"])[0].astype(np.float32)
                values["gate_features"] = compose_feedback_gate_features(
                    np.asarray(values["gate_features"], dtype=np.float32),
                    arm=self.arm)
            values.update({key: truth[key] for key in FEEDBACK_TARGETS})
        return {**{key: _to_tensor(value) for key, value in values.items()
                    if key != "sample_id"}, "sample_id": row["sample_id"]}

    def close(self) -> None:
        self.handle_cache.close()


class V24FeedbackTeacherDataset(Dataset):
    """Privileged synthetic-only teacher reader; never used by a student path."""

    def __init__(self, dataset_root: str | Path, *, build_id: str,
                 cell_id: str, split: str):
        self.root = Path(dataset_root)
        self.identity = validate_dataset_root(self.root, build_id=build_id)
        self.rows = query_index(
            self.root / "index" / "index.sqlite", cell_id=cell_id, split=split)
        if not self.rows:
            raise ValueError("V2.4 feedback teacher selected zero rows")
        if any(row["provenance"] != "synthetic" for row in self.rows):
            raise ValueError("privileged feedback teacher is synthetic-only")
        self.sample_ids = tuple(row["sample_id"] for row in self.rows)
        self.handle_cache = V24Hdf5HandleCache(maximum_handles=8)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        observation_relative = row["observation_shard"]
        truth_relative = row["truth_shard"]
        observation = self.handle_cache.read_row(
            self.root / observation_relative, int(row["observation_row"]),
            tuple(sorted(FEEDBACK_BASE)),
            expected_sha=self.identity["shard_hashes"][observation_relative])
        truth_fields = tuple(sorted((TEACHER_INPUT - FEEDBACK_BASE) | FEEDBACK_TARGETS))
        truth = self.handle_cache.read_row(
            self.root / truth_relative, int(row["truth_row"]), truth_fields,
            expected_sha=self.identity["shard_hashes"][truth_relative])
        values = {**observation, **truth}
        return {**{key: _to_tensor(value) for key, value in values.items()},
                "sample_id": row["sample_id"]}

    def close(self) -> None:
        self.handle_cache.close()


class V24TeacherSoftTargetDataset(Dataset):
    """Join a deployable A0-A3 view with sanitized teacher outputs only."""

    SOFT_FIELDS = (
        "teacher_soft_action", "teacher_soft_recoverability",
        "teacher_soft_gain_quantiles", "teacher_soft_harm_probability",
        "teacher_soft_next_measurement", "teacher_distillation_mask",
    )

    def __init__(self, student_dataset: Dataset,
                 soft_cache_path: str | Path, *, soft_cache_sha: str,
                 expected_teacher_checkpoint_sha: str = "",
                 expected_sample_set_sha: str = ""):
        if student_dataset.arm == "A4_ORACLE":
            raise ValueError("oracle arm cannot be a deployable student dataset")
        if len(soft_cache_sha) != 64:
            raise ValueError("teacher soft cache requires a SHA-256 identity")
        self.student_dataset = student_dataset
        self.soft_cache_path = Path(soft_cache_path)
        self.soft_cache_sha = soft_cache_sha
        self.sample_ids = getattr(student_dataset, "sample_ids", ())
        self.handle_cache = V24Hdf5HandleCache(maximum_handles=1)
        if self.sample_ids:
            cache_ids = self.handle_cache.read_rows(
                self.soft_cache_path, tuple(range(len(self.sample_ids))), ("sample_id",),
                expected_sha=soft_cache_sha)["sample_id"]
            decoded = tuple(value.decode("utf-8") for value in cache_ids)
            if decoded != self.sample_ids:
                raise ValueError("teacher soft cache sample order does not match student rows")
        else:
            from .storage_hdf5 import validate_hdf5_shard
            if validate_hdf5_shard(self.soft_cache_path)["row_count"] != len(student_dataset):
                raise ValueError("teacher soft cache row count does not match range dataset")
        # Verify teacher checkpoint SHA from the cache if an expectation is given.
        if expected_teacher_checkpoint_sha:
            first_row = self.handle_cache.read_row(
                self.soft_cache_path, 0, ("teacher_checkpoint_sha",),
                expected_sha=soft_cache_sha)
            cached_sha = np.asarray(first_row["teacher_checkpoint_sha"]).item().decode("utf-8")
            if cached_sha != expected_teacher_checkpoint_sha:
                raise ValueError(
                    "teacher soft cache checkpoint SHA mismatch: "
                    f"cached={cached_sha[:16]}... expected={expected_teacher_checkpoint_sha[:16]}...")
        # Verify sample-set SHA from the student dataset if available.
        if expected_sample_set_sha:
            student_sha = getattr(student_dataset, "sample_id_manifest_sha", "")
            if student_sha and student_sha != expected_sample_set_sha:
                raise ValueError(
                    "teacher soft cache sample-set SHA mismatch: "
                    f"student={student_sha[:16]}... expected={expected_sample_set_sha[:16]}...")

    def __len__(self) -> int:
        return len(self.student_dataset)

    @property
    def arm(self) -> str:
        """Forward arm identity from the underlying student dataset."""
        return getattr(self.student_dataset, "arm", "")

    @property
    def sample_count(self) -> int:
        """Forward sample count from the underlying student dataset."""
        return getattr(self.student_dataset, "sample_count",
                       len(self.student_dataset))

    @property
    def sample_id_manifest_sha(self) -> str:
        """Forward sample-set identity SHA from the underlying student dataset."""
        return getattr(self.student_dataset, "sample_id_manifest_sha", "")

    @property
    def sample_range_sha(self) -> str:
        """Forward range SHA from the underlying student dataset."""
        return getattr(self.student_dataset, "sample_range_sha", "")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        values = self.student_dataset[index]
        soft = self.handle_cache.read_row(
            self.soft_cache_path, index, self.SOFT_FIELDS + ("sample_id",),
            expected_sha=self.soft_cache_sha)
        soft_sample_id = np.asarray(soft.pop("sample_id")).item().decode("utf-8")
        if soft_sample_id != values["sample_id"]:
            raise ValueError("teacher soft cache sample order does not match student row")
        values.update({key: _to_tensor(value) for key, value in soft.items()})
        return values

    def close(self) -> None:
        self.student_dataset.close()
        self.handle_cache.close()


def dataset_training_identity(dataset_root: str | Path, *, build_id: str,
                              cell_id: str, split: str) -> Dict[str, str]:
    identity = validate_dataset_root(dataset_root, build_id=build_id)
    rows = query_index(Path(dataset_root) / "index" / "index.sqlite",
                       cell_id=cell_id, split=split)
    identity["cell_id"] = cell_id
    identity["split"] = split
    identity["cell_split_sha"] = sha256_json([
        {"sample_id": row["sample_id"], "row_content_sha": row["row_content_sha"]}
        for row in rows])
    return identity
