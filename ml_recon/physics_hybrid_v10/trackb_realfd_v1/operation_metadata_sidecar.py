"""Detached, source-bound operation metadata for legacy V2.4 real roots.

The historical V2.4.0 real-static roots preserve the measurement command-table,
registry, and projector hashes but can lack operation-family metadata.  This
module never rewrites those roots.  Instead it creates and verifies a separate
canonical sidecar whose values are replayed from the same authoritative basis
preimage and rejected on any identity mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from ml_recon.data_v2_4.contracts import (
    LEGACY_V240_DATASET_VERSION,
    LEGACY_V240_GENERATOR_REVISION,
    contract_schema_sha256,
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
from ml_recon.data_v2_4.loaders import V24RealV2ObservationDataset, validate_dataset_root
from ml_recon.data_v2_4.storage_hdf5 import validate_hdf5_shard

from .contracts import (
    canonical_json_bytes,
    canonical_sha256,
    parse_canonical_json_object,
    require_exact_canonical_json_round_trip,
    require_git_oid,
    require_sha256,
)


SIDECAR_SCHEMA = "TrackBV240OperationMetadataSidecarV1"
SIDECAR_STATUS = "VERIFIED_DETACHED_STATIC_METADATA"
SIDECAR_ORIGIN = "DETACHED_POST_HOC_DETERMINISTIC_DERIVATION"
_BASIS_SOURCE_MODE = "DETERMINISTIC_BUILDER"
_BUILDER_RECIPE_SCHEMA = "TrackBV240BasisBuilderRecipeV1"
_PROJECTOR_SCHEMA = "V10PhysicalOperationProjectorDescriptorV2"
_BASIS_PHASE_CONVENTION = "read_aod_negative_order"
_PROFILE_UNRECORDED = "UNRECORDED_IN_HISTORICAL_MANIFEST"
_PROFILE_ATTRIBUTED = "V24_REAL_CONVERTER"
_PROTECTED_BUILDER_PATHS = (
    "system_control/qubit_recon_files/basis_configs.py",
    "ml_recon/physics_hybrid_v10/operation_adapters.py",
    "ml_recon/physics_hybrid_v9/operation_registry.py",
)
_CONVERTER_SOURCE_PATHS = (
    "ml_recon/data_v2_4/real_converter.py",
    "ml_recon/data_v2_4/version.py",
)
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


def _require_exact_bool(value: object, name: str, expected: bool) -> None:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{name} must be native bool {expected}")


def _require_native_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a native integer >= {minimum}")
    return int(value)


def _require_relative_path(value: object, name: str) -> str:
    text = _require_nonempty_string(value, name)
    path = Path(text)
    if path.is_absolute() or "\\" in text or text.startswith("../") or "/../" in text:
        raise ValueError(f"{name} must be a portable repository-relative path")
    return text


def require_detached_sidecar_path(path: str | Path, dataset_root: str | Path) -> Path:
    """Resolve and reject any sidecar physically located in the immutable root."""

    sidecar = Path(path).resolve(strict=True)
    root = Path(dataset_root).resolve(strict=True)
    try:
        sidecar.relative_to(root)
    except ValueError:
        return sidecar
    raise ValueError("operation metadata sidecar must be outside the immutable dataset root")


def _sha_sequence(values: Sequence[object], schema: str) -> str:
    return canonical_sha256({"schema": schema, "values": list(values)})


def _flat_metadata_sha(
    operation_ids: Sequence[str], operation_families: Sequence[str],
    operation_family_ids: Sequence[int], cycle_indices: Sequence[int],
    within_cycle_indices: Sequence[int],
) -> str:
    return canonical_sha256({
        "schema": "TrackBV240FlatOperationMetadataV1",
        "operation_ids": list(operation_ids),
        "operation_families": list(operation_families),
        "operation_family_ids": list(operation_family_ids),
        "cycle_indices": list(cycle_indices),
        "within_cycle_indices": list(within_cycle_indices),
    })


def _git_text(repository_root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ("git", "-C", str(repository_root), *arguments),
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("sidecar requires a readable Git checkout") from error


def _git_bytes(repository_root: Path, revision: str, relative_path: str) -> bytes:
    try:
        return subprocess.run(
            ("git", "-C", str(repository_root), "show", f"{revision}:{relative_path}"),
            check=True, capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("sidecar source blob is unavailable from the declared Git object") from error


def _git_blob_oid(repository_root: Path, revision: str, relative_path: str) -> str:
    output = _git_text(repository_root, "ls-tree", revision, "--", relative_path)
    pieces = output.split()
    if len(pieces) < 3 or pieces[1] != "blob":
        raise ValueError("sidecar protected source path is absent from the declared Git object")
    require_git_oid(pieces[2], "protected_source_blob_oid")
    return pieces[2]


def _checked_source_records(
    repository_root: Path, source_commit: str, source_tree: str,
) -> tuple[dict[str, object], ...]:
    if _git_text(repository_root, "rev-parse", f"{source_commit}^{{tree}}") != source_tree:
        raise ValueError("sidecar root source commit/tree identity disagrees")
    records = []
    for relative_path in _PROTECTED_BUILDER_PATHS:
        source_blob = _git_blob_oid(repository_root, source_commit, relative_path)
        current_blob = _git_text(repository_root, "hash-object", "--", relative_path)
        head_blob = _git_text(repository_root, "rev-parse", f"HEAD:{relative_path}")
        if source_blob != current_blob or source_blob != head_blob:
            raise ValueError("sidecar builder source differs from its declared historical Git blob")
        records.append({
            "path": relative_path,
            "git_blob_oid": source_blob,
            "bytes_sha": sha256_bytes(_git_bytes(repository_root, source_commit, relative_path)),
        })
    return tuple(records)


def _historical_converter_identity(
    repository_root: Path, source_commit: str, root_generator_sha: str,
) -> dict[str, object]:
    records = []
    version_source = b""
    for relative_path in _CONVERTER_SOURCE_PATHS:
        source = _git_bytes(repository_root, source_commit, relative_path)
        records.append({
            "path": relative_path,
            "git_blob_oid": _git_blob_oid(repository_root, source_commit, relative_path),
            "bytes_sha": sha256_bytes(source),
        })
        if relative_path.endswith("version.py"):
            version_source = source
    match = re.search(
        rb'^REAL_CONVERTER_REVISION\s*=\s*"([^"]+)"', version_source, flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError("historical V2.4 version source lacks REAL_CONVERTER_REVISION")
    revision = match.group(1).decode("ascii")
    expected_generator_sha = canonical_sha256({"converter": revision})
    if root_generator_sha != expected_generator_sha:
        raise ValueError("historical root generator SHA does not identify its real converter revision")
    return {
        "profile_attribution": _PROFILE_ATTRIBUTED,
        "profile_attribution_mode": "SOURCE_BOUND_LEGACY_CONVERTER_REVISION",
        "converter_revision": revision,
        "converter_generator_sha": root_generator_sha,
        "converter_source_records": list(records),
    }


def _basis_raw_bytes(system_type: str) -> tuple[bytes, dict[str, object]]:
    from system_control.qubit_recon_files.basis_configs import (
        build_3qubit_basis_config,
        build_4qubit_basis_config,
    )

    if system_type == "3qubit":
        configuration = build_3qubit_basis_config()
    elif system_type == "4qubit":
        configuration = build_4qubit_basis_config()
    else:
        raise ValueError("sidecar supports only 3qubit or 4qubit")
    # The immutable V2.4.0 command tables were emitted by the Windows writer
    # used by the historical converter.  Emit CRLF explicitly rather than
    # inheriting the current host's text-mode newline behavior.
    raw_bytes = json.dumps(configuration, indent=2).replace("\n", "\r\n").encode("utf-8")
    recipe = {
        "schema": _BUILDER_RECIPE_SCHEMA,
        "encoding": "utf-8",
        "ensure_ascii": True,
        "indent": 2,
        "sort_keys": False,
        "trailing_newline": False,
        "line_ending": "CRLF",
    }
    return raw_bytes, recipe


def _metadata_from_basis_bytes(
    system_type: str, raw_bytes: bytes,
) -> tuple[dict[str, object], dict[str, object]]:
    from ml_recon.physics_hybrid_v10.operation_adapters import (
        v10_operation_registry,
        v10_projector_descriptors,
    )

    configuration = json.loads(raw_bytes.decode("utf-8"))
    if type(configuration) is not dict or configuration.get("system_type") != system_type:
        raise ValueError("sidecar basis builder output has an invalid system type")
    with tempfile.TemporaryDirectory(prefix="trackb_v240_sidecar_") as temporary:
        root = Path(temporary)
        path = root / f"basis_config_{system_type}.json"
        path.write_bytes(raw_bytes)
        registry = v10_operation_registry(system_type, basis_config_root=root)
        descriptors, descriptor_vocabulary = v10_projector_descriptors(
            system_type, basis_config_root=root, phase_sign=-1,
        )
    operation_ids = tuple(str(value) for value in registry.flat_order)
    expected_count = 75 if system_type == "3qubit" else 256
    if len(operation_ids) != expected_count:
        raise ValueError("sidecar registry operation count is invalid")
    if system_type == "3qubit":
        entries = configuration.get("entries")
        if type(entries) is not list or len(entries) != expected_count:
            raise ValueError("3Q sidecar basis has an invalid entry window")
        operation_families = tuple(
            _require_nonempty_string(entry.get("label"), "3Q basis label").split("_", 1)[0]
            for entry in entries if type(entry) is dict
        )
        if len(operation_families) != expected_count:
            raise ValueError("3Q sidecar basis entries are malformed")
        cycle_indices = tuple(0 for _ in operation_ids)
        within_cycle_indices = tuple(range(expected_count))
        legacy_window_abi = "LEGACY_REAL_3Q_WINDOW_75"
        family_recipe = "registered_basis_label_prefix_v1"
    else:
        cycles = configuration.get("cycles")
        if type(cycles) is not list or len(cycles) != 4:
            raise ValueError("4Q sidecar basis must contain four explicit cycles")
        operation_families_list: list[str] = []
        cycle_indices_list: list[int] = []
        within_cycle_indices_list: list[int] = []
        for cycle_index, cycle in enumerate(cycles):
            if type(cycle) is not dict or cycle.get("cycle") != cycle_index + 1:
                raise ValueError("4Q sidecar cycle ordering is invalid")
            entries = cycle.get("entries")
            if type(entries) is not list or len(entries) != 64:
                raise ValueError("4Q sidecar cycle must contain exactly 64 entries")
            operation_families_list.extend(f"cycle_{cycle_index}" for _ in entries)
            cycle_indices_list.extend(cycle_index for _ in entries)
            within_cycle_indices_list.extend(range(64))
        operation_families = tuple(operation_families_list)
        cycle_indices = tuple(cycle_indices_list)
        within_cycle_indices = tuple(within_cycle_indices_list)
        legacy_window_abi = "LEGACY_REAL_4Q_WINDOW_256"
        family_recipe = "registered_cycle_family_v1"
    vocabulary = tuple(sorted(set(operation_families)))
    family_lookup = {value: index for index, value in enumerate(vocabulary)}
    family_ids = tuple(family_lookup[value] for value in operation_families)
    if tuple(descriptor_vocabulary) != vocabulary:
        raise ValueError("projector descriptor family vocabulary disagrees with basis derivation")
    metadata = {
        "operation_family_vocabulary": list(vocabulary),
        "operation_family_vocabulary_sha": _sha_sequence(
            vocabulary, "TrackBV240OperationFamilyVocabularyV1"),
        "operation_family_ids": list(family_ids),
        "operation_family_ids_sha": _sha_sequence(
            family_ids, "TrackBV240OperationFamilyIdsV1"),
        "cycle_indices": list(cycle_indices),
        "cycle_indices_sha": _sha_sequence(cycle_indices, "TrackBV240CycleIndicesV1"),
        "within_cycle_indices": list(within_cycle_indices),
        "within_cycle_indices_sha": _sha_sequence(
            within_cycle_indices, "TrackBV240WithinCycleIndicesV1"),
        "flat_operation_metadata_sha": _flat_metadata_sha(
            operation_ids, operation_families, family_ids, cycle_indices, within_cycle_indices),
        "family_derivation_recipe": family_recipe,
    }
    measurement = {
        "system_type": system_type,
        "legacy_window_abi": legacy_window_abi,
        "operation_count": expected_count,
        "ordered_operation_ids": list(operation_ids),
        "ordered_operation_ids_sha": _sha_sequence(operation_ids, "TrackBV240OperationIdsV1"),
        "command_table_sha": sha256_bytes(raw_bytes),
        "operation_registry_sha": registry.registry_hash(),
        "projector_descriptor_schema": _PROJECTOR_SCHEMA,
        "projector_descriptor_sha": sha256_array(descriptors),
        "projector_descriptor_dtype": descriptors.dtype.str,
        "projector_descriptor_shape": list(descriptors.shape),
        "phase_sign": -1,
        "basis_phase_convention": _BASIS_PHASE_CONVENTION,
    }
    return measurement, metadata


def _authority_payload() -> dict[str, object]:
    return {
        "native_operation_family_metadata_present": False,
        "metadata_origin": SIDECAR_ORIGIN,
        "root_rewritten": False,
        "counts_used_for_derivation": False,
        "operation_ids_used_as_family_heuristic": False,
        "research_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "real_action_supervision_available": False,
        "real_gain_available": False,
        "real_jacobian_available": False,
        "hardware_action_authorized": False,
    }


def _validate_authority(value: object) -> dict[str, object]:
    fields = frozenset(_authority_payload())
    result = _require_exact_mapping(value, fields, "sidecar authority")
    expected = _authority_payload()
    if result != expected:
        raise ValueError("sidecar authority boundary is invalid")
    return result


def _validate_root_identity(value: object) -> dict[str, object]:
    fields = frozenset({
        "dataset_contract_version", "source_contract_version", "generator_revision",
        "root_manifest_generator_profile_id", "converter_profile_attribution",
        "converter_profile_attribution_mode", "converter_revision", "converter_generator_sha",
        "converter_source_records", "build_id", "ready_sha", "build_manifest_sha",
        "contract_file_sha", "contract_semantic_sha", "schema_sha", "index_sha", "split_sha",
        "validation_report_sha", "source_commit", "source_tree", "ready_shard_set_sha",
        "selected_system_type", "selected_index_rows_sha", "selected_observation_shards",
        "selected_truth_shards",
    })
    result = _require_exact_mapping(value, fields, "sidecar root identity")
    if result["dataset_contract_version"] != LEGACY_V240_DATASET_VERSION \
            or result["source_contract_version"] != "2.3" \
            or result["generator_revision"] != LEGACY_V240_GENERATOR_REVISION:
        raise ValueError("sidecar must bind the exact V2.4.0/r9 real contract")
    profile = result["root_manifest_generator_profile_id"]
    if profile not in (None, _PROFILE_ATTRIBUTED):
        raise ValueError("sidecar root profile must preserve the native recorded value")
    if result["converter_profile_attribution"] != _PROFILE_ATTRIBUTED \
            or result["converter_profile_attribution_mode"] \
            not in ("NATIVE_MANIFEST", "SOURCE_BOUND_LEGACY_CONVERTER_REVISION"):
        raise ValueError("sidecar converter profile attribution is invalid")
    _require_nonempty_string(result["converter_revision"], "converter_revision")
    require_sha256(result["converter_generator_sha"], "converter_generator_sha")
    for name in (
        "ready_sha", "build_manifest_sha", "contract_file_sha", "contract_semantic_sha",
        "schema_sha", "index_sha", "split_sha", "validation_report_sha", "ready_shard_set_sha",
        "selected_index_rows_sha",
    ):
        require_sha256(result[name], name)
    require_git_oid(result["source_commit"], "root_source_commit")
    require_git_oid(result["source_tree"], "root_source_tree")
    if result["selected_system_type"] not in ("3qubit", "4qubit"):
        raise ValueError("sidecar root identity has an invalid system type")
    _require_nonempty_string(result["build_id"], "build_id")
    records = result["converter_source_records"]
    if type(records) is not list or not records:
        raise ValueError("sidecar requires converter source records")
    for record in records:
        entry = _require_exact_mapping(record, frozenset({"path", "git_blob_oid", "bytes_sha"}), "converter source record")
        _require_relative_path(entry["path"], "converter source path")
        require_git_oid(entry["git_blob_oid"], "converter source blob")
        require_sha256(entry["bytes_sha"], "converter source bytes")
    for key in ("selected_observation_shards", "selected_truth_shards"):
        shards = result[key]
        if type(shards) is not list:
            raise ValueError(f"{key} must be a JSON list")
        for shard in shards:
            entry = _require_exact_mapping(shard, frozenset({"path", "sha256"}), key)
            _require_relative_path(entry["path"], "sidecar shard path")
            require_sha256(entry["sha256"], "sidecar shard SHA")
    return result


def _validate_derivation(value: object) -> dict[str, object]:
    fields = frozenset({
        "basis_source_mode", "basis_raw_bytes_sha", "builder_source_commit", "builder_source_tree",
        "builder_recipe_schema", "builder_recipe_sha", "builder_recipe", "protected_source_records",
    })
    result = _require_exact_mapping(value, fields, "sidecar derivation")
    if result["basis_source_mode"] != _BASIS_SOURCE_MODE \
            or result["builder_recipe_schema"] != _BUILDER_RECIPE_SCHEMA:
        raise ValueError("sidecar derivation source mode or recipe schema is invalid")
    for name in ("basis_raw_bytes_sha", "builder_recipe_sha"):
        require_sha256(result[name], name)
    require_git_oid(result["builder_source_commit"], "builder_source_commit")
    require_git_oid(result["builder_source_tree"], "builder_source_tree")
    recipe = _require_exact_mapping(
        result["builder_recipe"],
        frozenset({
            "schema", "encoding", "ensure_ascii", "indent", "sort_keys", "trailing_newline",
            "line_ending",
        }),
        "sidecar builder recipe",
    )
    if recipe["schema"] != _BUILDER_RECIPE_SCHEMA or recipe["encoding"] != "utf-8" \
            or recipe["ensure_ascii"] is not True or recipe["indent"] != 2 \
            or recipe["sort_keys"] is not False or recipe["trailing_newline"] is not False:
        raise ValueError("sidecar builder serialization recipe is invalid")
    if recipe["line_ending"] != "CRLF":
        raise ValueError("sidecar builder line-ending recipe is invalid")
    records = result["protected_source_records"]
    if type(records) is not list or tuple(record.get("path") for record in records if type(record) is dict) \
            != _PROTECTED_BUILDER_PATHS:
        raise ValueError("sidecar protected source records are incomplete or reordered")
    for record in records:
        entry = _require_exact_mapping(record, frozenset({"path", "git_blob_oid", "bytes_sha"}), "protected source record")
        _require_relative_path(entry["path"], "protected source path")
        require_git_oid(entry["git_blob_oid"], "protected source blob")
        require_sha256(entry["bytes_sha"], "protected source bytes")
    return result


def _validate_measurement(value: object) -> dict[str, object]:
    fields = frozenset({
        "system_type", "legacy_window_abi", "operation_count", "ordered_operation_ids",
        "ordered_operation_ids_sha", "command_table_sha", "operation_registry_sha",
        "projector_descriptor_schema", "projector_descriptor_sha", "projector_descriptor_dtype",
        "projector_descriptor_shape", "phase_sign", "basis_phase_convention",
    })
    result = _require_exact_mapping(value, fields, "sidecar measurement ABI")
    system_type = result["system_type"]
    expected_count = 75 if system_type == "3qubit" else 256 if system_type == "4qubit" else 0
    expected_abi = "LEGACY_REAL_3Q_WINDOW_75" if system_type == "3qubit" else "LEGACY_REAL_4Q_WINDOW_256"
    if not expected_count or result["legacy_window_abi"] != expected_abi \
            or _require_native_int(result["operation_count"], "operation_count") != expected_count:
        raise ValueError("sidecar measurement ABI is invalid")
    operation_ids = result["ordered_operation_ids"]
    if type(operation_ids) is not list or len(operation_ids) != expected_count \
            or len(set(operation_ids)) != expected_count \
            or any(type(item) is not str or not item for item in operation_ids):
        raise ValueError("sidecar operation order is invalid")
    if result["ordered_operation_ids_sha"] != _sha_sequence(operation_ids, "TrackBV240OperationIdsV1"):
        raise ValueError("sidecar operation ID SHA is invalid")
    for name in ("command_table_sha", "operation_registry_sha", "projector_descriptor_sha"):
        require_sha256(result[name], name)
    if result["projector_descriptor_schema"] != _PROJECTOR_SCHEMA \
            or result["phase_sign"] != -1 \
            or result["basis_phase_convention"] != _BASIS_PHASE_CONVENTION:
        raise ValueError("sidecar projector convention is invalid")
    shape = result["projector_descriptor_shape"]
    if type(shape) is not list or len(shape) != 2 or any(type(item) is not int or item <= 0 for item in shape) \
            or shape[0] != expected_count:
        raise ValueError("sidecar projector shape is invalid")
    if result["projector_descriptor_dtype"] != "<f4":
        raise ValueError("sidecar projector dtype is invalid")
    return result


def _validate_operation_metadata(value: object, measurement: Mapping[str, object]) -> dict[str, object]:
    fields = frozenset({
        "operation_family_vocabulary", "operation_family_vocabulary_sha", "operation_family_ids",
        "operation_family_ids_sha", "cycle_indices", "cycle_indices_sha", "within_cycle_indices",
        "within_cycle_indices_sha", "flat_operation_metadata_sha", "family_derivation_recipe",
    })
    result = _require_exact_mapping(value, fields, "sidecar operation metadata")
    count = int(measurement["operation_count"])
    vocabulary = result["operation_family_vocabulary"]
    family_ids = result["operation_family_ids"]
    cycle_indices = result["cycle_indices"]
    within_cycle_indices = result["within_cycle_indices"]
    if type(vocabulary) is not list or not vocabulary or vocabulary != sorted(vocabulary) \
            or len(set(vocabulary)) != len(vocabulary) \
            or any(type(item) is not str or not item for item in vocabulary):
        raise ValueError("sidecar family vocabulary is invalid")
    if type(family_ids) is not list or len(family_ids) != count \
            or any(type(item) is not int or item < 0 or item >= len(vocabulary) for item in family_ids):
        raise ValueError("sidecar family IDs are invalid")
    if type(cycle_indices) is not list or type(within_cycle_indices) is not list \
            or len(cycle_indices) != count or len(within_cycle_indices) != count \
            or any(type(item) is not int or item < 0 for item in cycle_indices + within_cycle_indices):
        raise ValueError("sidecar cycle metadata is invalid")
    if measurement["system_type"] == "3qubit":
        if any(cycle_indices) or within_cycle_indices != list(range(75)) \
                or result["family_derivation_recipe"] != "registered_basis_label_prefix_v1":
            raise ValueError("3Q sidecar cycle/family derivation is invalid")
    else:
        if cycle_indices != [index // 64 for index in range(256)] \
                or within_cycle_indices != [index % 64 for index in range(256)] \
                or result["family_derivation_recipe"] != "registered_cycle_family_v1":
            raise ValueError("4Q sidecar cycle/family derivation is invalid")
    expected_shas = {
        "operation_family_vocabulary_sha": _sha_sequence(vocabulary, "TrackBV240OperationFamilyVocabularyV1"),
        "operation_family_ids_sha": _sha_sequence(family_ids, "TrackBV240OperationFamilyIdsV1"),
        "cycle_indices_sha": _sha_sequence(cycle_indices, "TrackBV240CycleIndicesV1"),
        "within_cycle_indices_sha": _sha_sequence(within_cycle_indices, "TrackBV240WithinCycleIndicesV1"),
        "flat_operation_metadata_sha": _flat_metadata_sha(
            measurement["ordered_operation_ids"],
            [vocabulary[index] for index in family_ids],
            family_ids, cycle_indices, within_cycle_indices,
        ),
    }
    if any(result[name] != expected for name, expected in expected_shas.items()):
        raise ValueError("sidecar operation metadata SHA is invalid")
    return result


@dataclass(frozen=True)
class VerifiedOperationMetadataV1:
    """Verified static schedule metadata usable by Track B admission only."""

    operation_families: tuple[str, ...]
    operation_family_vocabulary: tuple[str, ...]
    operation_family_ids: tuple[int, ...]
    cycle_indices: tuple[int, ...]
    within_cycle_indices: tuple[int, ...]
    sidecar_file_sha: str
    sidecar_semantic_sha: str
    metadata_sha: str
    root_identity_sha: str
    root_artifact_shas: tuple[tuple[str, str], ...]
    origin: str = SIDECAR_ORIGIN

    def __post_init__(self) -> None:
        if not self.operation_families or not self.operation_family_vocabulary \
                or tuple(sorted(self.operation_family_vocabulary)) != self.operation_family_vocabulary \
                or len(set(self.operation_family_vocabulary)) != len(self.operation_family_vocabulary) \
                or len(self.operation_families) != len(self.operation_family_ids) \
                or len(self.operation_families) != len(self.cycle_indices) \
                or len(self.operation_families) != len(self.within_cycle_indices):
            raise ValueError("verified operation metadata lengths disagree")
        require_sha256(self.sidecar_file_sha, "sidecar_file_sha")
        require_sha256(self.sidecar_semantic_sha, "sidecar_semantic_sha")
        require_sha256(self.metadata_sha, "operation_metadata_sha")
        require_sha256(self.root_identity_sha, "root_identity_sha")
        if tuple(name for name, _sha in self.root_artifact_shas) != _ROOT_ARTIFACT_SHA_FIELDS:
            raise ValueError("verified operation metadata root artifact fields are invalid")
        for name, value in self.root_artifact_shas:
            require_sha256(value, name)
        if self.origin != SIDECAR_ORIGIN:
            raise ValueError("verified operation metadata origin is invalid")


@dataclass(frozen=True)
class TrackBV240OperationMetadataSidecarV1:
    """Canonical detached sidecar with zero action or hardware authority."""

    created_utc: str
    root_identity: dict[str, object]
    derivation: dict[str, object]
    measurement_abi: dict[str, object]
    operation_metadata: dict[str, object]
    authority: dict[str, object]
    sidecar_semantic_sha: str
    status: str = SIDECAR_STATUS
    schema: str = field(default=SIDECAR_SCHEMA, init=False)

    def __post_init__(self) -> None:
        if type(self.created_utc) is not str or not self.created_utc.endswith("Z"):
            raise ValueError("sidecar created_utc must be an ISO-8601 UTC timestamp")
        try:
            datetime.fromisoformat(self.created_utc[:-1] + "+00:00")
        except ValueError as error:
            raise ValueError("sidecar created_utc is invalid") from error
        if self.status != SIDECAR_STATUS:
            raise ValueError("sidecar status is invalid")
        root = _validate_root_identity(self.root_identity)
        derivation = _validate_derivation(self.derivation)
        measurement = _validate_measurement(self.measurement_abi)
        _validate_operation_metadata(self.operation_metadata, measurement)
        _validate_authority(self.authority)
        if root["selected_system_type"] != measurement["system_type"] \
                or derivation["basis_raw_bytes_sha"] != measurement["command_table_sha"]:
            raise ValueError("sidecar root, derivation, and measurement identities disagree")
        require_sha256(self.sidecar_semantic_sha, "sidecar_semantic_sha")
        if self.sidecar_semantic_sha != canonical_sha256(self.semantic_payload()):
            raise ValueError("sidecar semantic SHA is invalid")

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "status": self.status,
            "created_utc": self.created_utc,
            "root_identity": self.root_identity,
            "derivation": self.derivation,
            "measurement_abi": self.measurement_abi,
            "operation_metadata": self.operation_metadata,
            "authority": self.authority,
        }

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {**self.semantic_payload(), "sidecar_semantic_sha": self.sidecar_semantic_sha}

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def file_sha256(self) -> str:
        return sha256_bytes(self.canonical_json_bytes())

    @classmethod
    def from_canonical_json_bytes(cls, payload: bytes) -> "TrackBV240OperationMetadataSidecarV1":
        value = parse_canonical_json_object(
            payload,
            schema=SIDECAR_SCHEMA,
            exact_fields=frozenset({
                "schema", "status", "created_utc", "root_identity", "derivation",
                "measurement_abi", "operation_metadata", "authority", "sidecar_semantic_sha",
            }),
        )
        result = cls(
            created_utc=value["created_utc"], root_identity=value["root_identity"],
            derivation=value["derivation"], measurement_abi=value["measurement_abi"],
            operation_metadata=value["operation_metadata"], authority=value["authority"],
            sidecar_semantic_sha=value["sidecar_semantic_sha"], status=value["status"],
        )
        require_exact_canonical_json_round_trip(payload, result)
        return result


def _root_identity(
    dataset_root: Path, system_type: str, identity: Mapping[str, object],
    repository_root: Path,
) -> dict[str, object]:
    contract = load_contract(dataset_root)
    if contract.dataset_contract_version != LEGACY_V240_DATASET_VERSION \
            or contract.generator_revision != LEGACY_V240_GENERATOR_REVISION:
        raise ValueError("sidecar supports only the immutable V2.4.0/r9 real contract")
    ready_path = dataset_root / "READY.json"
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    manifest_path = dataset_root / "manifests" / f"{identity['build_id']}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = manifest.get("generator_profile_id")
    if profile not in (None, _PROFILE_ATTRIBUTED):
        raise ValueError("historical root has an unsupported generator profile identity")
    source_commit = str(identity["source_commit"])
    source_tree = str(identity["source_tree"])
    converter = _historical_converter_identity(
        repository_root, source_commit, str(identity["generator_sha"]),
    )
    if profile == _PROFILE_ATTRIBUTED:
        converter["profile_attribution_mode"] = "NATIVE_MANIFEST"
    rows = [
        row for row in all_index_rows(dataset_root / "index" / "index.sqlite")
        if row["system_type"] == system_type and row["dataset_family"] == "real_static"
    ]
    if not rows:
        raise ValueError("sidecar root selected zero real-static rows")
    observation_shards = sorted({str(row["observation_shard"]) for row in rows})
    if any(not path or path not in ready["shard_hashes"] for path in observation_shards):
        raise ValueError("sidecar selected an observation shard outside READY")
    truth_shards = sorted({str(row["truth_shard"]) for row in rows if str(row["truth_shard"])})
    if truth_shards:
        raise ValueError("real-static sidecar cannot attach a truth shard")
    selected_rows_sha = sha256_json(rows)
    return {
        "dataset_contract_version": contract.dataset_contract_version,
        "source_contract_version": contract.source_contract_version,
        "generator_revision": contract.generator_revision,
        "root_manifest_generator_profile_id": profile,
        "converter_profile_attribution": converter["profile_attribution"],
        "converter_profile_attribution_mode": converter["profile_attribution_mode"],
        "converter_revision": converter["converter_revision"],
        "converter_generator_sha": converter["converter_generator_sha"],
        "converter_source_records": converter["converter_source_records"],
        "build_id": str(identity["build_id"]),
        "ready_sha": sha256_file(ready_path),
        "build_manifest_sha": sha256_file(manifest_path),
        "contract_file_sha": sha256_file(dataset_root / "CONTRACT.json"),
        "contract_semantic_sha": contract.sha256(),
        "schema_sha": contract_schema_sha256(contract),
        "index_sha": str(identity["index_sha"]),
        "split_sha": str(identity["split_sha"]),
        "validation_report_sha": str(ready["validation_report_sha"]),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "ready_shard_set_sha": sha256_json(ready["shard_hashes"]),
        "selected_system_type": system_type,
        "selected_index_rows_sha": selected_rows_sha,
        "selected_observation_shards": [
            {"path": path, "sha256": ready["shard_hashes"][path]}
            for path in observation_shards
        ],
        "selected_truth_shards": [],
    }


def _metadata_families(
    metadata: Mapping[str, object], count: int, expected_vocabulary: Sequence[str],
) -> tuple[str, ...] | None:
    names = metadata.get("operation_families", ())
    identifiers = metadata.get("operation_family_ids", ())
    names_empty = names is None or (type(names) in (list, tuple) and len(names) == 0)
    identifiers_empty = identifiers is None or (
        type(identifiers) in (list, tuple) and len(identifiers) == 0
    )
    if names_empty and identifiers_empty:
        return None
    if not names_empty and identifiers_empty:
        if type(names) is not list or tuple(names) != tuple(expected_vocabulary):
            raise ValueError("root shard native family vocabulary conflicts with sidecar replay")
        return None
    if type(names) is not list or type(identifiers) is not list \
            or len(names) == 0 or len(identifiers) != count \
            or any(type(value) is not str or not value for value in names) \
            or any(type(value) is not int or value < 0 or value >= len(names) for value in identifiers):
        raise ValueError("root shard has partial or invalid native operation-family metadata")
    return tuple(names[index] for index in identifiers)


def _verify_selected_root_shards(
    dataset_root: Path, root_identity: Mapping[str, object],
    measurement: Mapping[str, object], operation_metadata: Mapping[str, object],
) -> None:
    """Recheck every selected shard without treating the sidecar as root data."""

    expected_ids = tuple(str(value) for value in measurement["ordered_operation_ids"])
    vocabulary = tuple(str(value) for value in operation_metadata["operation_family_vocabulary"])
    expected_families = tuple(
        vocabulary[int(index)] for index in operation_metadata["operation_family_ids"]
    )
    for record in root_identity["selected_observation_shards"]:
        path = dataset_root / str(record["path"])
        validation = validate_hdf5_shard(path)
        if validation["sha256"] != record["sha256"]:
            raise ValueError("selected root shard SHA changed during sidecar verification")
        metadata = validation["metadata"]
        if metadata.get("dataset_family") != "real_static" \
                or metadata.get("system_type") != measurement["system_type"] \
                or metadata.get("dataset_contract_version") != root_identity["dataset_contract_version"] \
                or metadata.get("operation_registry_sha") != measurement["operation_registry_sha"] \
                or metadata.get("command_table_sha") != measurement["command_table_sha"] \
                or metadata.get("operation_descriptor_schema") != _PROJECTOR_SCHEMA \
                or metadata.get("operation_descriptor_phase_sign") != -1 \
                or metadata.get("operation_descriptor_sha") != measurement["projector_descriptor_sha"] \
                or tuple(metadata.get("flat_order", ())) != expected_ids:
            raise ValueError("selected root shard measurement ABI disagrees with sidecar replay")
        native_families = _metadata_families(metadata, len(expected_ids), vocabulary)
        if native_families is not None and native_families != expected_families:
            raise ValueError("native root operation-family metadata conflicts with sidecar replay")
        with h5py.File(path, "r", swmr=True) as handle:
            if "real_action_supervision_mask" not in handle:
                raise ValueError("selected root shard lacks real action supervision mask")
            if bool(np.any(np.asarray(handle["real_action_supervision_mask"][:], dtype=np.bool_))):
                raise ValueError("selected root shard exposes forbidden real action supervision")
            if "global_multi_scope" not in handle:
                raise ValueError("selected root shard lacks global multi scope")
            scopes = {
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in np.asarray(handle["global_multi_scope"][:]).reshape(-1)
            }
            if not scopes <= {"global", "state"}:
                raise ValueError("selected root shard converts global multi into operation-local scope")


def _derive_live_components(
    dataset_root: Path, system_type: str, repository_root: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    identity = validate_dataset_root(dataset_root, strict_shard_audit=True)
    root = _root_identity(dataset_root, system_type, identity, repository_root)
    source_records = _checked_source_records(
        repository_root, str(root["source_commit"]), str(root["source_tree"]),
    )
    raw_bytes, recipe = _basis_raw_bytes(system_type)
    measurement, metadata = _metadata_from_basis_bytes(system_type, raw_bytes)
    _verify_selected_root_shards(dataset_root, root, measurement, metadata)
    derivation = {
        "basis_source_mode": _BASIS_SOURCE_MODE,
        "basis_raw_bytes_sha": sha256_bytes(raw_bytes),
        "builder_source_commit": root["source_commit"],
        "builder_source_tree": root["source_tree"],
        "builder_recipe_schema": _BUILDER_RECIPE_SCHEMA,
        "builder_recipe_sha": canonical_sha256(recipe),
        "builder_recipe": recipe,
        "protected_source_records": list(source_records),
    }
    dataset = V24RealV2ObservationDataset(
        dataset_root, build_id=str(identity["build_id"]), system_type=system_type, role="train",
    )
    try:
        if tuple(measurement["ordered_operation_ids"]) != tuple(dataset.operation_ids) \
                or measurement["command_table_sha"] != dataset.command_table_sha \
                or measurement["operation_registry_sha"] != dataset.operation_registry_sha \
                or measurement["projector_descriptor_sha"] != dataset.operation_descriptor_sha:
            raise ValueError("sidecar fresh derivation does not match the immutable root measurement ABI")
    finally:
        dataset.close()
    return root, derivation, measurement | {"operation_metadata": metadata}


def build_v240_operation_metadata_sidecar(
    dataset_root: str | Path, *, system_type: str,
    repository_root: str | Path | None = None, created_utc: str | None = None,
) -> TrackBV240OperationMetadataSidecarV1:
    """Build a detached sidecar in memory; callers choose a separate output path."""

    root = Path(dataset_root)
    repository = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[3]
    root_identity, derivation, derived = _derive_live_components(root, system_type, repository)
    created = created_utc or _canonical_utc_now()
    semantic_payload = {
        "schema": SIDECAR_SCHEMA,
        "status": SIDECAR_STATUS,
        "created_utc": created,
        "root_identity": root_identity,
        "derivation": derivation,
        "measurement_abi": {key: value for key, value in derived.items() if key != "operation_metadata"},
        "operation_metadata": derived["operation_metadata"],
        "authority": _authority_payload(),
    }
    return TrackBV240OperationMetadataSidecarV1(
        created_utc=created,
        root_identity=root_identity,
        derivation=derivation,
        measurement_abi=semantic_payload["measurement_abi"],
        operation_metadata=semantic_payload["operation_metadata"],
        authority=semantic_payload["authority"],
        sidecar_semantic_sha=canonical_sha256(semantic_payload),
    )


def write_v240_operation_metadata_sidecar(
    path: str | Path, sidecar: TrackBV240OperationMetadataSidecarV1,
    *, immutable_dataset_root: str | Path,
) -> dict[str, str]:
    """Write only the detached sidecar, never a file under the immutable root."""

    destination = Path(path)
    root = Path(immutable_dataset_root).resolve(strict=True)
    try:
        destination.resolve().relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("operation metadata sidecar destination must be outside the immutable dataset root")
    payload = sidecar.to_dict()
    actual = write_canonical_json(destination, payload)
    if actual != sidecar.file_sha256():
        raise RuntimeError("sidecar write SHA disagrees with canonical sidecar bytes")
    return {
        "sidecar_path": str(destination),
        "sidecar_file_sha": actual,
        "sidecar_semantic_sha": sidecar.sidecar_semantic_sha,
    }


def load_verified_v240_operation_metadata_sidecar_bytes(
    payload: bytes, *, expected_file_sha: str,
    dataset_root: str | Path, system_type: str,
    repository_root: str | Path | None = None,
) -> VerifiedOperationMetadataV1:
    """Replay one already-snapshotted sidecar payload against the immutable root."""

    require_sha256(expected_file_sha, "expected_operation_metadata_sidecar_sha")
    if type(payload) is not bytes:
        raise TypeError("operation metadata sidecar payload must be immutable bytes")
    file_sha = sha256_bytes(payload)
    if file_sha != expected_file_sha:
        raise ValueError("operation metadata sidecar file SHA mismatch")
    sidecar = TrackBV240OperationMetadataSidecarV1.from_canonical_json_bytes(payload)
    repository = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[3]
    root_identity, derivation, derived = _derive_live_components(
        Path(dataset_root), system_type, repository,
    )
    if sidecar.root_identity != root_identity or sidecar.derivation != derivation \
            or sidecar.measurement_abi != {key: value for key, value in derived.items() if key != "operation_metadata"} \
            or sidecar.operation_metadata != derived["operation_metadata"]:
        raise ValueError("operation metadata sidecar does not replay against the immutable root")
    metadata = sidecar.operation_metadata
    vocabulary = tuple(str(value) for value in metadata["operation_family_vocabulary"])
    family_ids = tuple(int(value) for value in metadata["operation_family_ids"])
    families = tuple(vocabulary[index] for index in family_ids)
    return VerifiedOperationMetadataV1(
        operation_families=families,
        operation_family_vocabulary=vocabulary,
        operation_family_ids=family_ids,
        cycle_indices=tuple(int(value) for value in metadata["cycle_indices"]),
        within_cycle_indices=tuple(int(value) for value in metadata["within_cycle_indices"]),
        sidecar_file_sha=file_sha,
        sidecar_semantic_sha=sidecar.sidecar_semantic_sha,
        metadata_sha=str(metadata["flat_operation_metadata_sha"]),
        root_identity_sha=canonical_sha256(sidecar.root_identity),
        root_artifact_shas=tuple(
            (name, str(sidecar.root_identity[name])) for name in _ROOT_ARTIFACT_SHA_FIELDS
        ),
    )


def load_verified_v240_operation_metadata_sidecar(
    path: str | Path, *, expected_file_sha: str,
    dataset_root: str | Path, system_type: str,
    repository_root: str | Path | None = None,
) -> VerifiedOperationMetadataV1:
    """Open one detached sidecar once, then replay that same payload strictly."""

    sidecar_path = require_detached_sidecar_path(path, dataset_root)
    return load_verified_v240_operation_metadata_sidecar_bytes(
        sidecar_path.read_bytes(),
        expected_file_sha=expected_file_sha,
        dataset_root=dataset_root,
        system_type=system_type,
        repository_root=repository_root,
    )


__all__ = [
    "SIDECAR_ORIGIN", "TrackBV240OperationMetadataSidecarV1", "VerifiedOperationMetadataV1",
    "build_v240_operation_metadata_sidecar", "write_v240_operation_metadata_sidecar",
    "load_verified_v240_operation_metadata_sidecar",
    "load_verified_v240_operation_metadata_sidecar_bytes", "require_detached_sidecar_path",
]
