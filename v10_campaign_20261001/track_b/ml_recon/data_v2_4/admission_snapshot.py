"""Stable snapshots for V2.4 dataset-root admission metadata."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .contracts import contract_schema_sha256, is_historical_v240_r6_contract_sha
from .hashing import canonical_json_bytes


@dataclass(frozen=True)
class Canonical_json_snapshot:
    payload: bytes
    sha256: str

    def __post_init__(self) -> None:
        if type(self.payload) is not bytes or type(self.sha256) is not str:
            raise TypeError("canonical snapshot requires immutable bytes and a text digest")
        if hashlib.sha256(self.payload).hexdigest() != self.sha256:
            raise ValueError("canonical snapshot bytes differ from their digest")

    def decode(self) -> Dict[str, Any]:
        return json.loads(self.payload.decode("utf-8"))


@dataclass(frozen=True)
class Dataset_root_metadata:
    root: Path
    ready: Canonical_json_snapshot
    manifest: Canonical_json_snapshot
    contract_sha: str
    schema_sha: str
    source_contract_version: str
    dataset_contract_version: str


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def read_stable_bytes(path: str | Path, *, label: str, expected_sha: str | None = None,
                      max_bytes: int | None = None,
                      reject_reparse: bool = False) -> bytes:
    source = Path(path)
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError("max_bytes must be a positive integer")
    if expected_sha is not None and (
            type(expected_sha) is not str or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)):
        raise ValueError("expected_sha must be a lowercase SHA-256 hex digest")
    try:
        before_link = source.lstat() if reject_reparse else None
        before = source.stat()
        if reject_reparse and (stat.S_ISLNK(before_link.st_mode)
                               or getattr(before_link, "st_file_attributes", 0) & 0x400):
            raise ValueError(f"{label} must not be a reparse point")
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        if max_bytes is not None and before.st_size > max_bytes:
            raise ValueError(f"{label} exceeds the {max_bytes}-byte limit")
        with source.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if _stat_identity(before) != _stat_identity(opened):
                raise ValueError(f"{label} changed before it was opened")
            if max_bytes is not None and opened.st_size > max_bytes:
                raise ValueError(f"{label} exceeds the {max_bytes}-byte limit")
            payload = handle.read(max_bytes + 1 if max_bytes is not None else -1)
            after_open = os.fstat(handle.fileno())
        after_link = source.lstat() if reject_reparse else None
        after = source.stat()
        if (_stat_identity(opened) != _stat_identity(after_open)
                or _stat_identity(after_open) != _stat_identity(after)
                or len(payload) != after_open.st_size):
            raise ValueError(f"{label} changed while it was read")
        if reject_reparse and (stat.S_ISLNK(after_link.st_mode)
                               or getattr(after_link, "st_file_attributes", 0) & 0x400):
            raise ValueError(f"{label} must not be a reparse point")
        if (reject_reparse and
                _stat_identity(before_link) != _stat_identity(after_link)):
            raise ValueError(f"{label} path changed while it was read")
        if max_bytes is not None and len(payload) > max_bytes:
            raise ValueError(f"{label} exceeds the {max_bytes}-byte limit")
    except OSError as error:
        raise ValueError(f"unable to read stable {label}: {error}") from error
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"{label} SHA mismatch")
    return payload


def read_canonical_snapshot(path: str | Path, *, label: str,
                            expected_sha: str | None = None,
                            max_bytes: int | None = None,
                            reject_reparse: bool = False) -> Canonical_json_snapshot:
    payload = read_stable_bytes(path, label=label, expected_sha=expected_sha,
                                max_bytes=max_bytes, reject_reparse=reject_reparse)
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    if payload != canonical_json_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return Canonical_json_snapshot(payload, hashlib.sha256(payload).hexdigest())


def read_dataset_root_metadata(root: str | Path, *, build_id: str | None = None,
                               max_manifest_bytes: int | None = None,
                               max_ready_bytes: int | None = None,
                               max_contract_bytes: int | None = None,
                               reject_reparse: bool = False,
                               allow_historical_r6_read_only: bool = False
                               ) -> Dataset_root_metadata:
    from .contracts import NQubitDatasetContractV240

    root_path = Path(root)
    contract_snapshot = read_canonical_snapshot(
        root_path / "CONTRACT.json", label="V2.4 contract",
        max_bytes=max_contract_bytes, reject_reparse=reject_reparse)
    contract = NQubitDatasetContractV240.from_dict(contract_snapshot.decode())
    contract_sha = contract.sha256()
    if is_historical_v240_r6_contract_sha(contract_sha) and not allow_historical_r6_read_only:
        raise ValueError("historical V2.4/r6 roots require a dedicated read-only admission")
    ready = read_canonical_snapshot(root_path / "READY.json", label="V2.4 READY",
                                    max_bytes=max_ready_bytes,
                                    reject_reparse=reject_reparse)
    ready_value = ready.decode()
    ready_build_id = ready_value["build_id"]
    if (type(ready_build_id) is not str or not ready_build_id
            or Path(ready_build_id).name != ready_build_id
            or ready_build_id in (".", "..")):
        raise ValueError("V2.4 READY build_id must be a single path component")
    if build_id is not None and ready_value["build_id"] != build_id:
        raise ValueError("V2.4 dataset build_id mismatch")
    schema_sha = contract_schema_sha256(contract)
    if ready_value["contract_sha"] != contract_sha or ready_value["schema_sha"] != schema_sha:
        raise ValueError("V2.4 contract/schema SHA mismatch")
    manifest_path = root_path / "manifests" / f"{ready_value['build_id']}.json"
    manifest = read_canonical_snapshot(
        manifest_path, label="V2.4 build manifest",
        expected_sha=ready_value["build_manifest_sha"],
        max_bytes=max_manifest_bytes, reject_reparse=reject_reparse)
    manifest_value = manifest.decode()
    if manifest_value["build_id"] != ready_build_id:
        raise ValueError("V2.4 build manifest build_id does not match READY")
    if manifest_value["contract_sha"] != contract_sha or manifest_value["schema_sha"] != schema_sha:
        raise ValueError("V2.4 build manifest is bound to another contract/schema")
    if (manifest_value.get("source_contract_version") != contract.source_contract_version
            or manifest_value.get("dataset_contract_version") != contract.dataset_contract_version):
        raise ValueError("V2.4 build manifest version does not match the frozen contract")
    return Dataset_root_metadata(root_path, ready, manifest, contract_sha, schema_sha,
                                 contract.source_contract_version,
                                 contract.dataset_contract_version)


def build_dataset_identity(metadata: Dataset_root_metadata, *, index_sha: str,
                           index_rows: int) -> Dict[str, Any]:
    ready = metadata.ready.decode()
    manifest = metadata.manifest.decode()
    if ready["build_manifest_sha"] != metadata.manifest.sha256:
        raise ValueError("V2.4 admitted manifest snapshot differs from READY")
    return {"build_id": ready["build_id"], "contract_sha": metadata.contract_sha,
            "schema_sha": metadata.schema_sha, "index_sha": index_sha,
            "split_sha": ready["split_sha"],
            "build_manifest_sha": ready["build_manifest_sha"],
            "source_contract_version": metadata.source_contract_version,
            "dataset_contract_version": metadata.dataset_contract_version,
            "generator_sha": manifest["generator_sha"],
            "shard_hashes": dict(ready["shard_hashes"]),
            "source_commit": manifest["source_commit"],
            "source_tree": manifest["source_tree_sha"], "rows": index_rows,
            "ready_sha": metadata.ready.sha256}
