"""Source-bound read-only admission for registered historical V2.4/r6 roots.

This module deliberately does not expose a training dataset, optimizer, pilot,
or hardware action. It seals a detached basis-preimage snapshot and the
in-memory projector descriptor derived from that exact snapshot for later
read-only inspection.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from ml_recon.data_v2_4.contracts import (
    HISTORICAL_V240_R6_GENERATOR_REVISION,
    LEGACY_V240_DATASET_VERSION,
    contract_schema_sha256,
    is_historical_v240_r6_contract_sha,
    load_contract,
)
from ml_recon.data_v2_4.hashing import (
    sha256_array,
    sha256_bytes,
    sha256_file,
    sha256_json,
    write_canonical_json,
)
from ml_recon.data_v2_4.index_sqlite import all_index_rows
from ml_recon.data_v2_4.loaders import validate_dataset_root
from ml_recon.data_v2_4.storage_hdf5 import validate_hdf5_shard

from .contracts import (
    canonical_json_bytes,
    canonical_sha256,
    parse_canonical_json_object,
    require_exact_canonical_json_round_trip,
    require_git_oid,
    require_sha256,
)


HISTORICAL_R6_SIDECAR_SCHEMA = "TrackBV240HistoricalR6ReadOnlySidecarV2"
HISTORICAL_R6_SIDECAR_STATUS = "VERIFIED_HISTORICAL_R6_READ_ONLY"
HISTORICAL_R6_SOURCE_ACCESS_PROFILE = "HISTORICAL_R6_READ_ONLY"
PROJECTOR_DESCRIPTOR_SCHEMA = "V10PhysicalOperationProjectorDescriptorV2"
PROJECTOR_PHASE_SIGN = -1
_BUILDER_RECIPE_SCHEMA = "TrackBV240HistoricalR6BasisPreimageV1"
_ROOT_ARTIFACT_SHA_FIELDS = (
    "ready_sha",
    "build_manifest_sha",
    "contract_file_sha",
    "validation_report_sha",
    "index_sha",
    "split_sha",
    "ready_shard_set_sha",
    "selected_index_rows_sha",
)
_BUILDER_SOURCE_MODULES = (
    (
        "ml_recon/physics_hybrid_v10/trackb_realfd_v1/historical_r6_read_only.py",
        "ml_recon.physics_hybrid_v10.trackb_realfd_v1.historical_r6_read_only",
    ),
    ("ml_recon/data_v2_4/contracts.py", "ml_recon.data_v2_4.contracts"),
    ("ml_recon/data_v2_4/loaders.py", "ml_recon.data_v2_4.loaders"),
    ("ml_recon/data_v2_4/storage_hdf5.py", "ml_recon.data_v2_4.storage_hdf5"),
    ("ml_recon/data_v2_4/index_sqlite.py", "ml_recon.data_v2_4.index_sqlite"),
    ("ml_recon/data_v2_4/hashing.py", "ml_recon.data_v2_4.hashing"),
    ("ml_recon/data_v2_4/schema_types.py", "ml_recon.data_v2_4.schema_types"),
    (
        "ml_recon/physics_hybrid_v10/trackb_realfd_v1/contracts.py",
        "ml_recon.physics_hybrid_v10.trackb_realfd_v1.contracts",
    ),
    (
        "ml_recon/physics_hybrid_v10/operation_adapters.py",
        "ml_recon.physics_hybrid_v10.operation_adapters",
    ),
    ("ml_recon/physics_hybrid_v9/contracts.py", "ml_recon.physics_hybrid_v9.contracts"),
    ("ml_recon/physics_hybrid_v9/operation_registry.py", "ml_recon.physics_hybrid_v9.operation_registry"),
)
_BUILDER_SOURCE_PATHS = tuple(path for path, _ in _BUILDER_SOURCE_MODULES)
_CONVERTER_SOURCE_PATHS = (
    "ml_recon/data_v2_4/real_converter.py",
    "ml_recon/data_v2_4/version.py",
)


def _canonical_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_exact_mapping(value: object, fields: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} has an invalid exact field set")
    return dict(value)


def _require_nonempty_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return str(value)


def _require_relative_path(value: object, name: str) -> str:
    text = _require_nonempty_string(value, name)
    path = Path(text)
    if path.is_absolute() or "\\" in text or text.startswith("../") or "/../" in text:
        raise ValueError(f"{name} must be a portable repository-relative path")
    return text


def _git_text(repository_root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ("git", "-C", str(repository_root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("historical r6 sidecar requires a readable Git checkout") from error


def _loaded_module_repository_root() -> Path:
    """Resolve the only checkout permitted to identify executed builder code."""
    module_path = Path(__file__).resolve()
    repository_root = module_path.parents[3]
    git_root = Path(_git_text(repository_root, "rev-parse", "--show-toplevel")).resolve()
    if git_root != repository_root:
        raise ValueError("historical r6 builder module is not rooted at its Git checkout")
    expected_module_path = (
        repository_root
        / "ml_recon/physics_hybrid_v10/trackb_realfd_v1/historical_r6_read_only.py"
    )
    if module_path != expected_module_path:
        raise ValueError("historical r6 builder module path disagrees with its Git checkout")
    return repository_root


def _resolve_builder_repository_root(repository_root: str | Path | None) -> Path:
    """Reject a caller-supplied checkout that differs from loaded builder code."""
    loaded_root = _loaded_module_repository_root()
    if repository_root is not None and Path(repository_root).resolve() != loaded_root:
        raise ValueError("historical r6 builder checkout differs from the loaded module checkout")
    return loaded_root


def _assert_loaded_builder_sources(repository_root: Path) -> None:
    """Bind every tracked builder dependency to the process-imported module file."""
    for relative_path, module_name in _BUILDER_SOURCE_MODULES:
        module = importlib.import_module(module_name)
        module_path = getattr(module, "__file__", None)
        if type(module_path) is not str:
            raise ValueError("historical r6 builder dependency has no source file")
        expected_path = (repository_root / relative_path).resolve()
        if Path(module_path).resolve() != expected_path:
            raise ValueError("historical r6 builder dependency differs from the loaded checkout")


def _git_bytes(repository_root: Path, revision: str, relative_path: str) -> bytes:
    try:
        return subprocess.run(
            ("git", "-C", str(repository_root), "show", f"{revision}:{relative_path}"),
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("historical r6 source blob is unavailable") from error


def _git_blob_oid(repository_root: Path, revision: str, relative_path: str) -> str:
    output = _git_text(repository_root, "ls-tree", revision, "--", relative_path)
    pieces = output.split()
    if len(pieces) < 3 or pieces[1] != "blob":
        raise ValueError("historical r6 source path is absent from the declared Git object")
    require_git_oid(pieces[2], "historical_r6_source_blob_oid")
    return pieces[2]


def _source_records(
    repository_root: Path,
    source_commit: str,
    paths: Sequence[str],
    *,
    require_current_checkout: bool,
) -> tuple[dict[str, object], ...]:
    records = []
    for relative_path in paths:
        blob_oid = _git_blob_oid(repository_root, source_commit, relative_path)
        if require_current_checkout:
            current_oid = _git_text(repository_root, "hash-object", "--", relative_path)
            if current_oid != blob_oid:
                raise ValueError("historical r6 builder source differs from its declared checkout blob")
        records.append({
            "path": relative_path,
            "git_blob_oid": blob_oid,
            "bytes_sha": sha256_bytes(_git_bytes(repository_root, source_commit, relative_path)),
        })
    return tuple(records)


def _historical_converter_identity(
    repository_root: Path,
    source_commit: str,
    source_tree: str,
    generator_sha: str,
) -> dict[str, object]:
    if _git_text(repository_root, "rev-parse", f"{source_commit}^{{tree}}") != source_tree:
        raise ValueError("historical r6 root source commit/tree identity disagrees")
    records = _source_records(
        repository_root,
        source_commit,
        _CONVERTER_SOURCE_PATHS,
        require_current_checkout=False,
    )
    version_source = _git_bytes(repository_root, source_commit, "ml_recon/data_v2_4/version.py")
    match = re.search(
        rb'^REAL_CONVERTER_REVISION\s*=\s*"([^"]+)"',
        version_source,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError("historical r6 version source lacks REAL_CONVERTER_REVISION")
    converter_revision = match.group(1).decode("ascii")
    expected_generator_sha = canonical_sha256({"converter": converter_revision})
    if generator_sha != expected_generator_sha:
        raise ValueError("historical r6 root generator SHA does not identify its converter revision")
    return {
        "converter_revision": converter_revision,
        "converter_generator_sha": generator_sha,
        "converter_source_records": list(records),
    }


def _read_basis_preimage(path: str | Path, expected_sha: str) -> bytes:
    require_sha256(expected_sha, "expected_basis_preimage_sha")
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise ValueError("historical r6 basis preimage is unavailable") from error
    if sha256_bytes(payload) != expected_sha:
        raise ValueError("historical r6 basis preimage SHA mismatch")
    return payload


def _derive_from_basis_snapshot(
    system_type: str,
    payload: bytes,
) -> tuple[dict[str, object], dict[str, object], np.ndarray]:
    try:
        configuration = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("historical r6 basis preimage is not valid UTF-8 JSON") from error
    if type(configuration) is not dict or configuration.get("system_type") != system_type:
        raise ValueError("historical r6 basis preimage has an invalid system type")
    from ml_recon.physics_hybrid_v10.operation_adapters import (
        v10_operation_registry,
        v10_projector_descriptors,
    )

    with tempfile.TemporaryDirectory(prefix="trackb_historical_r6_basis_") as temporary:
        basis_root = Path(temporary)
        snapshot_path = basis_root / f"basis_config_{system_type}.json"
        snapshot_path.write_bytes(payload)
        registry = v10_operation_registry(system_type, basis_config_root=basis_root)
        descriptors, descriptor_families = v10_projector_descriptors(
            system_type,
            basis_config_root=basis_root,
            phase_sign=PROJECTOR_PHASE_SIGN,
        )
        if snapshot_path.read_bytes() != payload:
            raise RuntimeError("historical r6 private basis snapshot changed during derivation")
    expected_count = 75 if system_type == "3qubit" else 256 if system_type == "4qubit" else 0
    if not expected_count or descriptors.ndim != 2 or descriptors.shape[0] != expected_count:
        raise ValueError("historical r6 projector descriptor shape is invalid")
    operation_ids = tuple(str(value) for value in registry.flat_order)
    if len(operation_ids) != expected_count or len(set(operation_ids)) != expected_count:
        raise ValueError("historical r6 basis registry operation order is invalid")
    families = tuple(str(value) for value in descriptor_families)
    if not families or any(not value for value in families):
        raise ValueError("historical r6 basis descriptor family vocabulary is invalid")
    if system_type == "3qubit":
        operation_families = tuple(
            str(entry["label"]).split("_", 1)[0]
            for entry in configuration.get("entries", ())
            if type(entry) is dict
        )
        cycle_indices = tuple(0 for _ in operation_ids)
        within_cycle_indices = tuple(range(expected_count))
        family_recipe = "registered_basis_label_prefix_v1"
        legacy_window_abi = "LEGACY_REAL_3Q_WINDOW_75"
    else:
        cycles = configuration.get("cycles", ())
        if type(cycles) is not list or len(cycles) != 4:
            raise ValueError("historical r6 4Q basis preimage requires four cycles")
        family_values: list[str] = []
        for cycle_index, cycle in enumerate(cycles):
            if type(cycle) is not dict or type(cycle.get("entries")) is not list:
                raise ValueError("historical r6 4Q basis cycle is invalid")
            family_values.extend(f"cycle_{cycle_index}" for entry in cycle["entries"] if type(entry) is dict)
        operation_families = tuple(family_values)
        cycle_indices = tuple(index // 64 for index in range(expected_count))
        within_cycle_indices = tuple(index % 64 for index in range(expected_count))
        family_recipe = "registered_cycle_family_v1"
        legacy_window_abi = "LEGACY_REAL_4Q_WINDOW_256"
    if len(operation_families) != expected_count:
        raise ValueError("historical r6 basis family metadata is incomplete")
    vocabulary = tuple(sorted(set(operation_families)))
    family_index = {value: index for index, value in enumerate(vocabulary)}
    family_ids = tuple(family_index[value] for value in operation_families)
    if tuple(sorted(set(operation_families))) != families:
        raise ValueError("historical r6 projector family vocabulary disagrees with basis metadata")
    descriptor_array = np.ascontiguousarray(np.asarray(descriptors, dtype=np.float32))
    measurement = {
        "system_type": system_type,
        "legacy_window_abi": legacy_window_abi,
        "operation_count": expected_count,
        "ordered_operation_ids": list(operation_ids),
        "ordered_operation_ids_sha": canonical_sha256({"schema": "TrackBV240OperationIdsV1", "values": list(operation_ids)}),
        "command_table_sha": sha256_bytes(payload),
        "operation_registry_sha": registry.registry_hash(),
        "projector_descriptor_schema": PROJECTOR_DESCRIPTOR_SCHEMA,
        "projector_descriptor_sha": sha256_array(descriptor_array),
        "projector_descriptor_dtype": descriptor_array.dtype.str,
        "projector_descriptor_shape": list(descriptor_array.shape),
        "phase_sign": PROJECTOR_PHASE_SIGN,
    }
    operation_metadata = {
        "operation_family_vocabulary": list(vocabulary),
        "operation_family_ids": list(family_ids),
        "cycle_indices": list(cycle_indices),
        "within_cycle_indices": list(within_cycle_indices),
        "family_derivation_recipe": family_recipe,
    }
    return measurement, operation_metadata, descriptor_array


def _root_identity(
    dataset_root: Path,
    *,
    system_type: str,
    repository_root: Path,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    identity = validate_dataset_root(
        dataset_root,
        strict_shard_audit=True,
        allow_historical_r6_read_only=True,
    )
    contract = load_contract(dataset_root)
    contract_sha = contract.sha256()
    if contract.dataset_contract_version != LEGACY_V240_DATASET_VERSION \
            or contract.generator_revision != HISTORICAL_V240_R6_GENERATOR_REVISION \
            or not is_historical_v240_r6_contract_sha(contract_sha):
        raise ValueError("historical read-only sidecar requires a registered V2.4.0/r6 root")
    manifest_path = dataset_root / "manifests" / f"{identity['build_id']}.json"
    ready_path = dataset_root / "READY.json"
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("historical r6 root metadata is unreadable") from error
    source_commit = str(identity["source_commit"])
    source_tree = str(identity["source_tree"])
    converter = _historical_converter_identity(
        repository_root,
        source_commit,
        source_tree,
        str(identity["generator_sha"]),
    )
    rows = [
        row for row in all_index_rows(dataset_root / "index" / "index.sqlite")
        if row["dataset_family"] == "real_static" and row["system_type"] == system_type
    ]
    if not rows:
        raise ValueError("historical r6 root selected zero real-static rows")
    selected_shards = sorted({str(row["observation_shard"]) for row in rows})
    if any(path not in ready["shard_hashes"] for path in selected_shards):
        raise ValueError("historical r6 root selected a shard outside READY")
    metadata_records = []
    for relative_path in selected_shards:
        path = dataset_root / relative_path
        validation = validate_hdf5_shard(path)
        if validation["sha256"] != ready["shard_hashes"][relative_path]:
            raise ValueError("historical r6 root shard changed during admission")
        metadata = validation["metadata"]
        descriptor_fields = (
            "operation_descriptor_schema",
            "operation_descriptor_phase_sign",
            "operation_descriptor_sha",
            "operation_descriptors",
        )
        if any(metadata.get(name) is not None for name in descriptor_fields):
            raise ValueError("historical r6 root has partial or native projector descriptor metadata")
        metadata_records.append(metadata)
        with h5py.File(path, "r", swmr=True) as handle:
            if "real_action_supervision_mask" not in handle \
                    or bool(np.any(np.asarray(handle["real_action_supervision_mask"][:], dtype=np.bool_))):
                raise ValueError("historical r6 root exposes forbidden action supervision")
            if "global_multi_scope" not in handle:
                raise ValueError("historical r6 root lacks global multi scope")
            scopes = {
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in np.asarray(handle["global_multi_scope"][:]).reshape(-1)
            }
            if not scopes <= {"global", "state"}:
                raise ValueError("historical r6 root converts global multi into operation-local scope")
    root = {
        "build_id": str(identity["build_id"]),
        "contract_sha": contract_sha,
        "schema_sha": contract_schema_sha256(contract),
        "ready_sha": sha256_file(ready_path),
        "build_manifest_sha": sha256_file(manifest_path),
        "contract_file_sha": sha256_file(dataset_root / "CONTRACT.json"),
        "validation_report_sha": str(ready["validation_report_sha"]),
        "index_sha": str(identity["index_sha"]),
        "split_sha": str(identity["split_sha"]),
        "ready_shard_set_sha": sha256_json(ready["shard_hashes"]),
        "selected_index_rows_sha": sha256_json(rows),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "generator_sha": str(identity["generator_sha"]),
        "generator_revision": contract.generator_revision,
        "converter": converter,
        "system_type": system_type,
        "selected_shards": [
            {"path": relative_path, "sha256": ready["shard_hashes"][relative_path]}
            for relative_path in selected_shards
        ],
    }
    return root, tuple(metadata_records)


def _builder_identity(repository_root: Path) -> dict[str, object]:
    _assert_loaded_builder_sources(repository_root)
    source_commit = _git_text(repository_root, "rev-parse", "HEAD")
    source_tree = _git_text(repository_root, "rev-parse", f"{source_commit}^{{tree}}")
    records = _source_records(
        repository_root,
        source_commit,
        _BUILDER_SOURCE_PATHS,
        require_current_checkout=True,
    )
    return {
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_records": list(records),
    }


def _validate_root_against_measurement(
    root: Mapping[str, object],
    metadata_records: Sequence[Mapping[str, object]],
    measurement: Mapping[str, object],
) -> None:
    expected_ids = tuple(str(value) for value in measurement["ordered_operation_ids"])
    for metadata in metadata_records:
        if metadata.get("dataset_family") != "real_static" \
                or metadata.get("system_type") != measurement["system_type"] \
                or metadata.get("dataset_contract_version") != LEGACY_V240_DATASET_VERSION \
                or metadata.get("operation_registry_sha") != measurement["operation_registry_sha"] \
                or metadata.get("command_table_sha") != measurement["command_table_sha"] \
                or tuple(metadata.get("flat_order", ())) != expected_ids:
            raise ValueError("historical r6 basis preimage does not match selected shard metadata")


def _authority_payload() -> dict[str, object]:
    return {
        "source_access_profile": HISTORICAL_R6_SOURCE_ACCESS_PROFILE,
        "research_only": True,
        "training_authorized": False,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }


def _binding_payload(
    payload: bytes,
    descriptor: np.ndarray,
    measurement: Mapping[str, object],
) -> dict[str, object]:
    descriptor_bytes = np.ascontiguousarray(descriptor, dtype=np.float32).tobytes(order="C")
    return {
        "source_access_profile": HISTORICAL_R6_SOURCE_ACCESS_PROFILE,
        "basis_preimage_sha": sha256_bytes(payload),
        "basis_preimage_base64": base64.b64encode(payload).decode("ascii"),
        "projector_descriptor_bytes_sha": sha256_bytes(descriptor_bytes),
        "projector_descriptor_base64": base64.b64encode(descriptor_bytes).decode("ascii"),
        "projector_descriptor_sha": measurement["projector_descriptor_sha"],
        "projector_descriptor_dtype": measurement["projector_descriptor_dtype"],
        "projector_descriptor_shape": measurement["projector_descriptor_shape"],
    }


def _validate_binding(value: object, measurement: Mapping[str, object]) -> dict[str, object]:
    fields = frozenset({
        "source_access_profile", "basis_preimage_sha", "basis_preimage_base64",
        "projector_descriptor_bytes_sha", "projector_descriptor_base64",
        "projector_descriptor_sha", "projector_descriptor_dtype", "projector_descriptor_shape",
    })
    result = _require_exact_mapping(value, fields, "historical r6 binding")
    if result["source_access_profile"] != HISTORICAL_R6_SOURCE_ACCESS_PROFILE:
        raise ValueError("historical r6 binding access profile is invalid")
    for name in (
        "basis_preimage_sha",
        "projector_descriptor_bytes_sha",
        "projector_descriptor_sha",
    ):
        require_sha256(result[name], name)
    if result["projector_descriptor_sha"] != measurement["projector_descriptor_sha"] \
            or result["projector_descriptor_dtype"] != measurement["projector_descriptor_dtype"] \
            or result["projector_descriptor_shape"] != measurement["projector_descriptor_shape"]:
        raise ValueError("historical r6 binding descriptor identity disagrees with measurement")
    try:
        basis_bytes = base64.b64decode(result["basis_preimage_base64"], validate=True)
        descriptor_bytes = base64.b64decode(result["projector_descriptor_base64"], validate=True)
    except (TypeError, ValueError) as error:
        raise ValueError("historical r6 binding base64 payload is invalid") from error
    if sha256_bytes(basis_bytes) != result["basis_preimage_sha"] \
            or sha256_bytes(descriptor_bytes) != result["projector_descriptor_bytes_sha"]:
        raise ValueError("historical r6 binding byte identity is invalid")
    shape = tuple(int(value) for value in measurement["projector_descriptor_shape"])
    if len(shape) != 2 or len(descriptor_bytes) != int(np.prod(shape)) * np.dtype("<f4").itemsize:
        raise ValueError("historical r6 binding descriptor bytes have an invalid shape")
    descriptor = np.frombuffer(descriptor_bytes, dtype=np.dtype("<f4")).reshape(shape)
    if sha256_array(descriptor) != measurement["projector_descriptor_sha"]:
        raise ValueError("historical r6 binding descriptor array SHA is invalid")
    return result


def _validate_root_identity(value: object) -> dict[str, object]:
    fields = frozenset({
        "build_id", "contract_sha", "schema_sha", "ready_sha", "build_manifest_sha",
        "contract_file_sha", "validation_report_sha", "index_sha", "split_sha",
        "ready_shard_set_sha", "selected_index_rows_sha", "source_commit", "source_tree",
        "generator_sha", "generator_revision", "converter", "system_type", "selected_shards",
    })
    result = _require_exact_mapping(value, fields, "historical r6 root identity")
    if result["generator_revision"] != HISTORICAL_V240_R6_GENERATOR_REVISION \
            or not is_historical_v240_r6_contract_sha(result["contract_sha"]):
        raise ValueError("historical r6 root contract identity is invalid")
    require_sha256(result["contract_sha"], "historical_r6_contract_sha")
    require_sha256(result["schema_sha"], "historical_r6_schema_sha")
    for name in _ROOT_ARTIFACT_SHA_FIELDS:
        require_sha256(result[name], name)
    require_git_oid(result["source_commit"], "historical_r6_root_source_commit")
    require_git_oid(result["source_tree"], "historical_r6_root_source_tree")
    require_sha256(result["generator_sha"], "historical_r6_generator_sha")
    if result["system_type"] not in ("3qubit", "4qubit"):
        raise ValueError("historical r6 root system type is invalid")
    converter = _require_exact_mapping(
        result["converter"],
        frozenset({"converter_revision", "converter_generator_sha", "converter_source_records"}),
        "historical r6 converter identity",
    )
    _require_nonempty_string(converter["converter_revision"], "converter_revision")
    require_sha256(converter["converter_generator_sha"], "converter_generator_sha")
    records = converter["converter_source_records"]
    if type(records) is not list or tuple(
            record.get("path") for record in records if type(record) is dict
    ) != _CONVERTER_SOURCE_PATHS:
        raise ValueError("historical r6 converter source records are invalid")
    for record in records:
        item = _require_exact_mapping(
            record,
            frozenset({"path", "git_blob_oid", "bytes_sha"}),
            "historical r6 converter source record",
        )
        _require_relative_path(item["path"], "historical_r6_converter_source_path")
        require_git_oid(item["git_blob_oid"], "historical_r6_converter_source_blob")
        require_sha256(item["bytes_sha"], "historical_r6_converter_source_sha")
    shards = result["selected_shards"]
    if type(shards) is not list or not shards:
        raise ValueError("historical r6 selected shards are invalid")
    for record in shards:
        item = _require_exact_mapping(record, frozenset({"path", "sha256"}), "historical r6 shard")
        _require_relative_path(item["path"], "historical_r6_shard_path")
        require_sha256(item["sha256"], "historical_r6_shard_sha")
    return result


def _validate_derivation(value: object) -> dict[str, object]:
    fields = frozenset({"builder", "basis_recipe_schema", "basis_preimage_sha"})
    result = _require_exact_mapping(value, fields, "historical r6 derivation")
    if result["basis_recipe_schema"] != _BUILDER_RECIPE_SCHEMA:
        raise ValueError("historical r6 basis recipe schema is invalid")
    require_sha256(result["basis_preimage_sha"], "basis_preimage_sha")
    builder = _require_exact_mapping(
        result["builder"],
        frozenset({"source_commit", "source_tree", "source_records"}),
        "historical r6 builder identity",
    )
    require_git_oid(builder["source_commit"], "historical_r6_builder_source_commit")
    require_git_oid(builder["source_tree"], "historical_r6_builder_source_tree")
    records = builder["source_records"]
    if type(records) is not list or tuple(
            record.get("path") for record in records if type(record) is dict
    ) != _BUILDER_SOURCE_PATHS:
        raise ValueError("historical r6 builder source closure is incomplete or reordered")
    for record in records:
        item = _require_exact_mapping(
            record,
            frozenset({"path", "git_blob_oid", "bytes_sha"}),
            "historical r6 builder source record",
        )
        _require_relative_path(item["path"], "historical_r6_builder_source_path")
        require_git_oid(item["git_blob_oid"], "historical_r6_builder_source_blob")
        require_sha256(item["bytes_sha"], "historical_r6_builder_source_sha")
    return result


def _validate_measurement(value: object) -> dict[str, object]:
    fields = frozenset({
        "system_type", "legacy_window_abi", "operation_count", "ordered_operation_ids",
        "ordered_operation_ids_sha", "command_table_sha", "operation_registry_sha",
        "projector_descriptor_schema", "projector_descriptor_sha",
        "projector_descriptor_dtype", "projector_descriptor_shape", "phase_sign",
    })
    result = _require_exact_mapping(value, fields, "historical r6 measurement")
    expected_count = 75 if result["system_type"] == "3qubit" else 256 if result["system_type"] == "4qubit" else 0
    if not expected_count or result["operation_count"] != expected_count:
        raise ValueError("historical r6 measurement operation count is invalid")
    operation_ids = result["ordered_operation_ids"]
    if type(operation_ids) is not list or len(operation_ids) != expected_count \
            or len(set(operation_ids)) != expected_count:
        raise ValueError("historical r6 operation order is invalid")
    if result["ordered_operation_ids_sha"] != canonical_sha256(
            {"schema": "TrackBV240OperationIdsV1", "values": operation_ids}):
        raise ValueError("historical r6 operation order SHA is invalid")
    for name in ("command_table_sha", "operation_registry_sha", "projector_descriptor_sha"):
        require_sha256(result[name], name)
    if result["projector_descriptor_schema"] != PROJECTOR_DESCRIPTOR_SCHEMA \
            or result["projector_descriptor_dtype"] != "<f4" \
            or result["phase_sign"] != PROJECTOR_PHASE_SIGN:
        raise ValueError("historical r6 descriptor ABI is invalid")
    shape = result["projector_descriptor_shape"]
    if type(shape) is not list or len(shape) != 2 or shape[0] != expected_count \
            or any(type(item) is not int or item <= 0 for item in shape):
        raise ValueError("historical r6 descriptor shape is invalid")
    return result


def _validate_operation_metadata(value: object, measurement: Mapping[str, object]) -> dict[str, object]:
    fields = frozenset({
        "operation_family_vocabulary", "operation_family_ids", "cycle_indices",
        "within_cycle_indices", "family_derivation_recipe",
    })
    result = _require_exact_mapping(value, fields, "historical r6 operation metadata")
    count = int(measurement["operation_count"])
    vocabulary = result["operation_family_vocabulary"]
    family_ids = result["operation_family_ids"]
    cycle_indices = result["cycle_indices"]
    within_cycle_indices = result["within_cycle_indices"]
    if type(vocabulary) is not list or not vocabulary or vocabulary != sorted(vocabulary) \
            or any(type(item) is not str or not item for item in vocabulary):
        raise ValueError("historical r6 operation family vocabulary is invalid")
    if type(family_ids) is not list or len(family_ids) != count \
            or any(type(item) is not int or item < 0 or item >= len(vocabulary) for item in family_ids):
        raise ValueError("historical r6 operation family IDs are invalid")
    if type(cycle_indices) is not list or type(within_cycle_indices) is not list \
            or len(cycle_indices) != count or len(within_cycle_indices) != count:
        raise ValueError("historical r6 cycle metadata is invalid")
    return result


def _validate_authority(value: object) -> dict[str, object]:
    result = _require_exact_mapping(value, frozenset(_authority_payload()), "historical r6 authority")
    if result != _authority_payload():
        raise ValueError("historical r6 authority boundary is invalid")
    return result


@dataclass(frozen=True)
class TrackBV240HistoricalR6ReadOnlySidecarV2:
    """Canonical, detached, read-only admission of a registered r6 root."""

    created_utc: str
    root_identity: dict[str, object]
    derivation: dict[str, object]
    measurement: dict[str, object]
    operation_metadata: dict[str, object]
    binding: dict[str, object]
    authority: dict[str, object]
    sidecar_semantic_sha: str
    status: str = HISTORICAL_R6_SIDECAR_STATUS
    schema: str = field(default=HISTORICAL_R6_SIDECAR_SCHEMA, init=False)

    def __post_init__(self) -> None:
        if type(self.created_utc) is not str or not self.created_utc.endswith("Z"):
            raise ValueError("historical r6 sidecar created_utc is invalid")
        try:
            datetime.fromisoformat(self.created_utc[:-1] + "+00:00")
        except ValueError as error:
            raise ValueError("historical r6 sidecar created_utc is invalid") from error
        if self.status != HISTORICAL_R6_SIDECAR_STATUS:
            raise ValueError("historical r6 sidecar status is invalid")
        root = _validate_root_identity(self.root_identity)
        derivation = _validate_derivation(self.derivation)
        measurement = _validate_measurement(self.measurement)
        operation_metadata = _validate_operation_metadata(self.operation_metadata, measurement)
        binding = _validate_binding(self.binding, measurement)
        _validate_authority(self.authority)
        if root["system_type"] != measurement["system_type"] \
                or derivation["basis_preimage_sha"] != measurement["command_table_sha"] \
                or binding["basis_preimage_sha"] != measurement["command_table_sha"]:
            raise ValueError("historical r6 sidecar identities disagree")
        if self.sidecar_semantic_sha != canonical_sha256(self.semantic_payload()):
            raise ValueError("historical r6 sidecar semantic SHA is invalid")
        if operation_metadata["operation_family_ids"] and not binding["projector_descriptor_base64"]:
            raise ValueError("historical r6 sidecar descriptor binding is unavailable")

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "status": self.status,
            "created_utc": self.created_utc,
            "root_identity": self.root_identity,
            "derivation": self.derivation,
            "measurement": self.measurement,
            "operation_metadata": self.operation_metadata,
            "binding": self.binding,
            "authority": self.authority,
        }

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {**self.semantic_payload(), "sidecar_semantic_sha": self.sidecar_semantic_sha}

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def file_sha256(self) -> str:
        return sha256_bytes(self.canonical_json_bytes())

    def projector_descriptors(self) -> np.ndarray:
        binding = _validate_binding(self.binding, self.measurement)
        payload = base64.b64decode(binding["projector_descriptor_base64"], validate=True)
        shape = tuple(int(value) for value in self.measurement["projector_descriptor_shape"])
        result = np.frombuffer(payload, dtype=np.dtype("<f4")).reshape(shape).copy()
        result.setflags(write=False)
        return result

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "TrackBV240HistoricalR6ReadOnlySidecarV2":
        value = parse_canonical_json_object(
            payload,
            schema=HISTORICAL_R6_SIDECAR_SCHEMA,
            exact_fields=frozenset({
                "schema", "status", "created_utc", "root_identity", "derivation",
                "measurement", "operation_metadata", "binding", "authority",
                "sidecar_semantic_sha",
            }),
        )
        result = cls(
            created_utc=value["created_utc"],
            root_identity=value["root_identity"],
            derivation=value["derivation"],
            measurement=value["measurement"],
            operation_metadata=value["operation_metadata"],
            binding=value["binding"],
            authority=value["authority"],
            sidecar_semantic_sha=value["sidecar_semantic_sha"],
            status=value["status"],
        )
        require_exact_canonical_json_round_trip(payload, result)
        return result


def _build_from_snapshot(
    dataset_root: Path,
    *,
    system_type: str,
    basis_preimage: bytes,
    expected_basis_preimage_sha: str,
    repository_root: Path,
    created_utc: str,
) -> TrackBV240HistoricalR6ReadOnlySidecarV2:
    if sha256_bytes(basis_preimage) != expected_basis_preimage_sha:
        raise ValueError("historical r6 basis preimage SHA mismatch")
    root, metadata_records = _root_identity(
        dataset_root,
        system_type=system_type,
        repository_root=repository_root,
    )
    measurement, operation_metadata, descriptor = _derive_from_basis_snapshot(
        system_type,
        basis_preimage,
    )
    _validate_root_against_measurement(root, metadata_records, measurement)
    builder = _builder_identity(repository_root)
    derivation = {
        "builder": builder,
        "basis_recipe_schema": _BUILDER_RECIPE_SCHEMA,
        "basis_preimage_sha": expected_basis_preimage_sha,
    }
    binding = _binding_payload(basis_preimage, descriptor, measurement)
    authority = _authority_payload()
    semantic_payload = {
        "schema": HISTORICAL_R6_SIDECAR_SCHEMA,
        "status": HISTORICAL_R6_SIDECAR_STATUS,
        "created_utc": created_utc,
        "root_identity": root,
        "derivation": derivation,
        "measurement": measurement,
        "operation_metadata": operation_metadata,
        "binding": binding,
        "authority": authority,
    }
    return TrackBV240HistoricalR6ReadOnlySidecarV2(
        created_utc=created_utc,
        root_identity=root,
        derivation=derivation,
        measurement=measurement,
        operation_metadata=operation_metadata,
        binding=binding,
        authority=authority,
        sidecar_semantic_sha=canonical_sha256(semantic_payload),
    )


def build_historical_r6_read_only_sidecar(
    dataset_root: str | Path,
    *,
    system_type: str,
    basis_preimage_path: str | Path,
    expected_basis_preimage_sha: str,
    repository_root: str | Path | None = None,
    created_utc: str | None = None,
) -> TrackBV240HistoricalR6ReadOnlySidecarV2:
    """Build a detached V2 sidecar in memory from one explicit basis snapshot."""
    root = Path(dataset_root)
    repository = _resolve_builder_repository_root(repository_root)
    payload = _read_basis_preimage(basis_preimage_path, expected_basis_preimage_sha)
    return _build_from_snapshot(
        root,
        system_type=system_type,
        basis_preimage=payload,
        expected_basis_preimage_sha=expected_basis_preimage_sha,
        repository_root=repository,
        created_utc=created_utc or _canonical_utc_now(),
    )


def write_historical_r6_read_only_sidecar(
    path: str | Path,
    sidecar: TrackBV240HistoricalR6ReadOnlySidecarV2,
    *,
    immutable_dataset_root: str | Path,
) -> dict[str, str]:
    """Write only a detached sidecar, never a file under the immutable root."""
    destination = Path(path)
    root = Path(immutable_dataset_root).resolve(strict=True)
    try:
        destination.resolve().relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("historical r6 sidecar destination must be outside the immutable dataset root")
    payload = sidecar.to_dict()
    actual_sha = write_canonical_json(destination, payload)
    if actual_sha != sidecar.file_sha256():
        raise RuntimeError("historical r6 sidecar write SHA disagrees with canonical bytes")
    return {"sidecar_path": str(destination), "sidecar_file_sha": actual_sha}


def load_verified_historical_r6_read_only_sidecar(
    path: str | Path,
    *,
    expected_file_sha: str,
    dataset_root: str | Path,
    system_type: str,
    repository_root: str | Path | None = None,
) -> TrackBV240HistoricalR6ReadOnlySidecarV2:
    """Replay an embedded preimage against its immutable root and current builder."""
    require_sha256(expected_file_sha, "expected_historical_r6_sidecar_sha")
    sidecar_path = Path(path).resolve(strict=True)
    root = Path(dataset_root).resolve(strict=True)
    try:
        sidecar_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("historical r6 sidecar must be outside the immutable dataset root")
    try:
        payload = sidecar_path.read_bytes()
    except OSError as error:
        raise ValueError("historical r6 sidecar is unavailable") from error
    if sha256_bytes(payload) != expected_file_sha:
        raise ValueError("historical r6 sidecar file SHA mismatch")
    sidecar = TrackBV240HistoricalR6ReadOnlySidecarV2.from_canonical_json_bytes(payload)
    if sidecar.root_identity["system_type"] != system_type:
        raise ValueError("historical r6 sidecar system type differs from the request")
    preimage = base64.b64decode(sidecar.binding["basis_preimage_base64"], validate=True)
    repository = _resolve_builder_repository_root(repository_root)
    replayed = _build_from_snapshot(
        root,
        system_type=system_type,
        basis_preimage=preimage,
        expected_basis_preimage_sha=sidecar.binding["basis_preimage_sha"],
        repository_root=repository,
        created_utc=sidecar.created_utc,
    )
    if replayed.semantic_payload() != sidecar.semantic_payload():
        raise ValueError("historical r6 sidecar does not replay against root and builder identities")
    return sidecar


__all__ = [
    "HISTORICAL_R6_SIDECAR_SCHEMA",
    "HISTORICAL_R6_SOURCE_ACCESS_PROFILE",
    "TrackBV240HistoricalR6ReadOnlySidecarV2",
    "build_historical_r6_read_only_sidecar",
    "load_verified_historical_r6_read_only_sidecar",
    "write_historical_r6_read_only_sidecar",
]
