"""Fail-closed disk datasets for V2.4 observation and training views."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import (
    LEGACY_V240_DATASET_VERSION,
    contract_schema_sha256,
    is_historical_v240_r6_contract_sha,
    load_contract,
)
from .feedback_gate_contract import compose_feedback_gate_features
from .hashing import read_canonical_json, sha256_array, sha256_file, sha256_json
from .index_sqlite import all_index_rows, query_index, validate_index
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
    REAL_COUNT_INPUT,
    TEACHER_INPUT,
    classifier_input_for_family,
    fields_for_view,
    validate_real_v4_lowdisk_fields,
)


REAL_DEVELOPMENT_SPLITS = frozenset({"train", "train_target_family", "probe_train"})
REAL_LOCKED_SPLITS = frozenset({
    "validation", "validation_target_family", "probe_validation", "test",
    "locked", "locked_ood",
})


def _decode_scalar(value: np.ndarray) -> str:
    item = np.asarray(value).item()
    return item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)


class V24RealObservationDataset(Dataset):
    """Observation-only ``real_static`` reader for count recovery.

    It never opens a truth or derived shard.  Split roles are inherited from
    the sealed V2.3 source: development rows are the only training role and
    all other roles stay locked evaluation rows.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        build_id: str,
        system_type: str,
        role: str = "all",
        task: str = "real_count_recovery",
        expected_operation_registry_sha: str = "",
        expected_command_table_sha: str = "",
        basis_config_root: str | Path | None = None,
    ):
        if system_type not in ("3qubit", "4qubit"):
            raise ValueError("real observation system_type must be 3qubit or 4qubit")
        if role not in ("all", "train", "locked"):
            raise ValueError("real observation role must be all, train, or locked")
        if task != "real_count_recovery":
            raise ValueError("real observation task must be real_count_recovery")
        self.root = Path(dataset_root)
        self.identity = validate_dataset_root(self.root, build_id=build_id)
        if self.identity["dataset_contract_version"] != LEGACY_V240_DATASET_VERSION:
            raise ValueError(
                "real observation requires the immutable V2.4.0 real-static contract"
            )
        candidates = [
            row for row in all_index_rows(self.root / "index" / "index.sqlite")
            if row["dataset_family"] == "real_static"
            and row["system_type"] == system_type
        ]
        if role == "train":
            candidates = [row for row in candidates if row["split"] in REAL_DEVELOPMENT_SPLITS]
        elif role == "locked":
            candidates = [row for row in candidates if row["split"] not in REAL_DEVELOPMENT_SPLITS]
        if not candidates:
            raise ValueError("V2.4 real observation view selected zero rows")
        self.rows = candidates
        self.system_type = system_type
        self.role = role
        self.task = task
        self.fields = tuple(sorted(REAL_COUNT_INPUT))
        self.sample_ids = tuple(str(row["sample_id"]) for row in self.rows)
        self.group_ids = tuple(
            f"{row['acquisition_run_id']}|{row['repeat_group_id']}|{row['trajectory_id']}"
            for row in self.rows)
        self.handle_cache = V24Hdf5HandleCache(maximum_handles=8)
        shard_metadata = {}
        for relative in sorted({str(row["observation_shard"]) for row in self.rows}):
            metadata = validate_hdf5_shard(self.root / relative)["metadata"]
            if metadata.get("dataset_family") != "real_static" \
                    or metadata.get("store") != "observation" \
                    or metadata.get("system_type") != system_type \
                    or metadata.get("dataset_contract_version") \
                    != self.identity["dataset_contract_version"]:
                raise ValueError("real observation shard metadata contract mismatch")
            shard_metadata[relative] = metadata
        registry_values = {str(value["operation_registry_sha"])
                           for value in shard_metadata.values()}
        command_values = {str(value["command_table_sha"])
                          for value in shard_metadata.values()}
        flat_orders = {tuple(value["flat_order"]) for value in shard_metadata.values()}
        descriptor_values = {str(value.get("operation_descriptor_sha", ""))
                             for value in shard_metadata.values()}
        descriptor_schemas = {str(value.get("operation_descriptor_schema", "V1"))
                              for value in shard_metadata.values()}
        descriptor_phase_signs = {int(value.get("operation_descriptor_phase_sign", -1))
                                  for value in shard_metadata.values()}
        descriptors = [np.asarray(value.get("operation_descriptors", ()), dtype=np.float32)
                       for value in shard_metadata.values()]
        if len(registry_values) != 1 or len(command_values) != 1 \
                or len(flat_orders) != 1 or len(descriptor_values) != 1 \
                or len(descriptor_schemas) != 1 or len(descriptor_phase_signs) != 1 \
                or not next(iter(descriptor_values)):
            raise ValueError("real observation shards disagree on operation registry binding")
        self.operation_registry_sha = next(iter(registry_values))
        self.command_table_sha = next(iter(command_values))
        self.operation_ids = next(iter(flat_orders))
        expected_count = 75 if system_type == "3qubit" else 256
        if len(self.operation_ids) != expected_count:
            raise ValueError("real observation operation count is incorrect")
        if expected_operation_registry_sha \
                and self.operation_registry_sha != expected_operation_registry_sha:
            raise ValueError("real observation operation registry SHA mismatch")
        if expected_command_table_sha \
                and self.command_table_sha != expected_command_table_sha:
            raise ValueError("real observation command table SHA mismatch")
        self.operation_descriptor_sha = next(iter(descriptor_values))
        self.operation_descriptor_schema = next(iter(descriptor_schemas))
        self.operation_descriptor_phase_sign = next(iter(descriptor_phase_signs))
        if any(value.ndim != 2 or value.shape[0] != expected_count for value in descriptors) \
                or len({value.tobytes() for value in descriptors}) != 1:
            raise ValueError("real observation shards disagree on physical operation descriptors")
        self.operation_descriptors = descriptors[0]
        if sha256_array(self.operation_descriptors) != self.operation_descriptor_sha:
            raise ValueError("real operation descriptor SHA mismatch")
        if basis_config_root is not None:
            from ml_recon.physics_hybrid_v10.operation_adapters import v10_operation_registry
            registry = v10_operation_registry(
                system_type, basis_config_root=basis_config_root)
            if registry.registry_hash() != self.operation_registry_sha \
                    or registry.command_table_hash != self.command_table_sha:
                raise ValueError("real operation registry does not match authoritative command table")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        relative = str(row["observation_shard"])
        values = self.handle_cache.read_row(
            self.root / relative, int(row["observation_row"]), self.fields,
            expected_sha=self.identity["shard_hashes"][relative])
        training_eligible = bool(np.asarray(values["training_eligible_mask"]).item())
        expected_training = row["split"] in REAL_DEVELOPMENT_SPLITS
        if training_eligible != expected_training:
            raise ValueError("real training eligibility disagrees with sealed split role")
        if bool(np.asarray(values["real_action_supervision_mask"]).item()):
            raise ValueError("real_static rows cannot expose action supervision")
        scope = _decode_scalar(values.pop("global_multi_scope"))
        target_sha = _decode_scalar(values.pop("target_sha"))
        if scope not in ("state", "global"):
            raise ValueError("global multi scope became operation-local")
        tensors = {key: _to_tensor(value) for key, value in values.items()}
        return {
            **tensors,
            "sample_id": str(row["sample_id"]),
            "system_type": self.system_type,
            "N": int(row["N"]),
            "source_split": str(row["split"]),
            "acquisition_run_id": str(row["acquisition_run_id"]),
            "repeat_group_id": str(row["repeat_group_id"]),
            "trajectory_id": str(row["trajectory_id"]),
            "global_multi_scope": scope,
            "target_sha": target_sha,
            "operation_ids": self.operation_ids,
            "operation_descriptors": torch.from_numpy(
                np.array(self.operation_descriptors, copy=True)),
            "operation_registry_sha": self.operation_registry_sha,
            "command_table_sha": self.command_table_sha,
        }

    def close(self) -> None:
        self.handle_cache.close()


class V24RealV2ObservationDataset(V24RealObservationDataset):
    """V2 masked-recovery facade with preparation-command semantics."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if self.operation_descriptor_schema != "V10PhysicalOperationProjectorDescriptorV2":
            raise ValueError("masked V2 loader requires projector descriptor V2")
        if self.operation_descriptor_phase_sign != -1:
            raise ValueError("masked V2 loader requires phase_sign=-1")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = super().__getitem__(index)
        item["prepared_state_command_coefficients_re"] = item.pop("target_coefficients_re")
        item["prepared_state_command_coefficients_im"] = item.pop("target_coefficients_im")
        item["prepared_state_mask"] = item.pop("target_coefficients_mask")
        item["prepared_state_command_sha"] = item.pop("target_sha")
        item["observed_operation_mask"] = item["operation_mask"].clone()
        item["prepared_state_semantics"] = (
            "prepared_state_command_coefficients_not_actual_state_or_feedback_target")
        # Compatibility aliases are intentionally explicit and will be removed
        # only by a versioned loader migration.
        item["prepared_state_coefficients_re"] = item[
            "prepared_state_command_coefficients_re"]
        item["prepared_state_coefficients_im"] = item[
            "prepared_state_command_coefficients_im"]
        return item


class V24RealGroupedDataset(Dataset):
    """Role-checked view that proves acquisition/repeat groups do not cross roles."""

    def __init__(self, dataset: V24RealObservationDataset):
        self.dataset = dataset
        all_rows = [
            row for row in all_index_rows(dataset.root / "index" / "index.sqlite")
            if row["dataset_family"] == "real_static"
            and row["system_type"] == dataset.system_type
        ]
        group_roles: Dict[str, set[str]] = {}
        for row in all_rows:
            group_id = f"{row['acquisition_run_id']}|{row['repeat_group_id']}|{row['trajectory_id']}"
            role = "train" if row["split"] in REAL_DEVELOPMENT_SPLITS else "locked"
            group_roles.setdefault(group_id, set()).add(role)
        leaking = [group_id for group_id, roles in group_roles.items() if len(roles) > 1]
        if leaking:
            raise ValueError(f"real acquisition/repeat groups cross roles: {leaking[:5]}")
        self.group_roles = {key: next(iter(value)) for key, value in group_roles.items()}
        if dataset.role == "train" and any(
                self.group_roles[group_id] != "train" for group_id in dataset.group_ids):
            raise ValueError("locked group entered real training dataset")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.dataset[index]

    def close(self) -> None:
        self.dataset.close()


class V24RealLearningCurveDataset(Dataset):
    """Deterministic nested whole-group subset for one learning-curve rung."""

    def __init__(self, dataset: V24RealGroupedDataset, *, requested_rows: int,
                 seed: int = 20261001):
        if dataset.dataset.role != "train":
            raise ValueError("real learning curves may use only the development training role")
        if requested_rows <= 0:
            raise ValueError("requested_rows must be positive")
        groups: Dict[str, list[int]] = {}
        for index, group_id in enumerate(dataset.dataset.group_ids):
            groups.setdefault(group_id, []).append(index)
        ordered = sorted(
            groups,
            key=lambda group_id: sha256_json({"seed": int(seed), "group_id": group_id}))
        selected: list[int] = []
        for group_id in ordered:
            candidate = groups[group_id]
            if len(selected) + len(candidate) > requested_rows:
                continue
            selected.extend(candidate)
        if not selected:
            raise ValueError("no whole group fits the requested learning-curve rung")
        self.dataset = dataset
        self.indices = tuple(selected)
        self.requested_rows = int(requested_rows)
        self.actual_rows = len(self.indices)
        self.group_ids = tuple(ordered_group for ordered_group in ordered
                               if any(index in self.indices for index in groups[ordered_group]))
        self.subset_sha = sha256_json({
            "seed": int(seed), "requested_rows": requested_rows,
            "sample_ids": [dataset.dataset.sample_ids[index] for index in self.indices],
        })

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.dataset[self.indices[index]]


class V24RealV4CompactContextDataset(Dataset):
    """Hash-bound compact V4 cache; dense draws/covariance are rejected."""

    def __init__(self, base_dataset: V24RealV2ObservationDataset,
                 cache_path: str | Path, *, cache_sha: str):
        from .views import REAL_V4_COMPACT_CONTEXT

        validate_real_v4_lowdisk_fields(set(REAL_V4_COMPACT_CONTEXT))
        if len(cache_sha) != 64:
            raise ValueError("V4 compact cache requires a SHA-256 identity")
        self.base_dataset = base_dataset
        self.cache_path = Path(cache_path)
        self.cache_sha = str(cache_sha)
        self.fields = tuple(sorted(REAL_V4_COMPACT_CONTEXT))
        self.handle_cache = V24Hdf5HandleCache(maximum_handles=1)
        metadata = validate_hdf5_shard(self.cache_path)["metadata"]
        persisted = set(metadata.get("fields", self.fields))
        validate_real_v4_lowdisk_fields(persisted)
        if int(metadata.get("row_count", len(base_dataset))) != len(base_dataset):
            raise ValueError("V4 compact cache row count mismatch")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        base = self.base_dataset[index]
        compact = self.handle_cache.read_row(
            self.cache_path, index, self.fields, expected_sha=self.cache_sha)
        base.update({key: _to_tensor(value) for key, value in compact.items()})
        return base

    def close(self) -> None:
        self.base_dataset.close()
        self.handle_cache.close()


def validate_dataset_root(dataset_root: str | Path, *, build_id: Optional[str] = None,
                          strict_shard_audit: bool = False,
                          allow_historical_r6_read_only: bool = False,
                          ) -> Dict[str, Any]:
    root = Path(dataset_root)
    ready_path = root / "READY.json"
    if not ready_path.exists():
        raise ValueError("V2.4 loader requires READY.json")
    contract = load_contract(root)
    if is_historical_v240_r6_contract_sha(contract.sha256()) \
            and not allow_historical_r6_read_only:
        raise ValueError(
            "historical V2.4/r6 roots require a dedicated read-only admission"
        )
    ready = read_canonical_json(ready_path)
    if build_id is not None and ready["build_id"] != build_id:
        raise ValueError("V2.4 dataset build_id mismatch")
    loaded_schema_sha = contract_schema_sha256(contract)
    if ready["contract_sha"] != contract.sha256() or ready["schema_sha"] != loaded_schema_sha:
        raise ValueError("V2.4 contract/schema SHA mismatch")
    manifest_path = root / "manifests" / f"{ready['build_id']}.json"
    manifest = read_canonical_json(manifest_path)
    if sha256_file(manifest_path) != ready["build_manifest_sha"]:
        raise ValueError("V2.4 build manifest SHA mismatch")
    if manifest["contract_sha"] != contract.sha256() or manifest["schema_sha"] != loaded_schema_sha:
        raise ValueError("V2.4 build manifest is bound to another contract/schema")
    if manifest.get("source_contract_version") != contract.source_contract_version \
            or manifest.get("dataset_contract_version") != contract.dataset_contract_version:
        raise ValueError("V2.4 build manifest version does not match the frozen contract")
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
            "schema_sha": loaded_schema_sha, "index_sha": index["sha256"],
            "split_sha": ready["split_sha"], "build_manifest_sha": ready["build_manifest_sha"],
            "source_contract_version": contract.source_contract_version,
            "dataset_contract_version": contract.dataset_contract_version,
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
