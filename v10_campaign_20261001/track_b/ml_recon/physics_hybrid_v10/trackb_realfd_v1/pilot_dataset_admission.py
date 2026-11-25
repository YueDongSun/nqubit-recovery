"""Bounded, immutable admission for Track B pilot dataset roots."""

from __future__ import annotations

import json
import sqlite3
import stat
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from ml_recon.data_v2_4.admission_snapshot import (
    build_dataset_identity,
    read_dataset_root_metadata,
    read_stable_bytes,
)
from ml_recon.data_v2_4.hashing import canonical_json_bytes
from ml_recon.data_v2_4.storage_hdf5 import validate_hdf5_handle
from ml_recon.data_v2_4.verified_shard_handle import open_verified_shard
from ml_recon.data_v2_4.manifests import V24ReadyManifest
from ml_recon.data_v2_4.validators import validate_authority_flags, validate_split_groups
from ml_recon.data_v2_4.synthetic.trackb_teacher_policy import (
    TRACKB_TEACHER_POLICY_V1,
    TRACKB_TEACHER_POLICY_V2,
    TRACKB_TEACHER_POLICY_V2_CELL_IDS,
    TRACKB_TEACHER_POLICY_V2_SHA256,
    reject_trackb_v2_intent,
    validate_trackb_v2_cell,
)
from .pilot_observation_identity import (
    validate_observation_identity_fields,
    validate_observation_row_index,
    validate_stored_observation_identity,
)


PILOT_FAMILY = "feedback_history"
PILOT_DOMAIN = "global_pauli_v24"
PILOT_SPLITS = ("train", "validation")
MAX_PILOT_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_PILOT_INDEX_BYTES = 64 * 1024 * 1024
MAX_PILOT_CONTRACT_BYTES = 1024 * 1024
MAX_PILOT_READY_BYTES = 1024 * 1024
MAX_PILOT_VALIDATION_REPORT_BYTES = 16 * 1024 * 1024
_FACTORY_TOKEN = object()


def _decode_copy(payload: bytes) -> Any:
    return json.loads(payload.decode("utf-8"))


@dataclass(frozen=True, init=False)
class Pilot_dataset_admission:
    """Factory-only admitted root whose nested state is held as canonical bytes.

    This prevents accidental mutation through returned mappings; it is not a
    security boundary against code executing inside the same Python process.
    """

    root: Path
    build_id: str
    cell_id: str
    history_policy: str
    history_policy_sha256: str
    _identity_payload: bytes = field(repr=False)
    _manifest_payload: bytes = field(repr=False)
    _validation_payload: bytes = field(repr=False)
    _train_rows_payload: tuple[bytes, ...] = field(repr=False)
    _validation_rows_payload: tuple[bytes, ...] = field(repr=False)

    def __init__(self, token: object, *, root: Path, build_id: str, cell_id: str,
                 history_policy: str, history_policy_sha256: str,
                 identity_payload: bytes, manifest_payload: bytes,
                 validation_payload: bytes, train_rows_payload: tuple[bytes, ...],
                 validation_rows_payload: tuple[bytes, ...]) -> None:
        if token is not _FACTORY_TOKEN:
            raise TypeError("Pilot_dataset_admission instances are factory-only")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "build_id", build_id)
        object.__setattr__(self, "cell_id", cell_id)
        object.__setattr__(self, "history_policy", history_policy)
        object.__setattr__(self, "history_policy_sha256", history_policy_sha256)
        object.__setattr__(self, "_identity_payload", bytes(identity_payload))
        object.__setattr__(self, "_manifest_payload", bytes(manifest_payload))
        object.__setattr__(self, "_validation_payload", bytes(validation_payload))
        object.__setattr__(self, "_train_rows_payload", tuple(bytes(v) for v in train_rows_payload))
        object.__setattr__(self, "_validation_rows_payload", tuple(bytes(v) for v in validation_rows_payload))

    def identity_dict(self) -> dict[str, Any]:
        return _decode_copy(self._identity_payload)

    def manifest_dict(self) -> dict[str, Any]:
        return _decode_copy(self._manifest_payload)

    def validation_dict(self) -> dict[str, Any]:
        return _decode_copy(self._validation_payload)

    def rows_for_split(self, split: str) -> list[dict[str, Any]]:
        if split == "train":
            payloads = self._train_rows_payload
        elif split == "validation":
            payloads = self._validation_rows_payload
        else:
            raise ValueError("Track B pilot admission requires train or validation split")
        return [_decode_copy(payload) for payload in payloads]

    def require_valid(self) -> None:
        """Recheck the sealed in-memory payload structure without disk access."""
        if (not isinstance(self.root, Path) or not self.build_id or not self.cell_id
                or self.history_policy not in (TRACKB_TEACHER_POLICY_V1, TRACKB_TEACHER_POLICY_V2)):
            raise ValueError("Track B pilot admission identity is incomplete")
        identity = self.identity_dict()
        manifest = self.manifest_dict()
        report = self.validation_dict()
        mode_count = _mode_count_for_cell(self.cell_id, self.history_policy)
        if (identity.get("build_id") != self.build_id
                or report.get("build_id") != self.build_id
                or not isinstance(manifest.get("cells"), list)):
            raise ValueError("Track B pilot admission payload identity is inconsistent")
        for split in PILOT_SPLITS:
            rows = self.rows_for_split(split)
            if not rows:
                raise ValueError("Track B pilot admission payload is missing a split")
            for row in rows:
                if (row.get("cell_id") != self.cell_id or row.get("split") != split
                        or row.get("N") != mode_count
                        or row.get("dataset_family") != PILOT_FAMILY
                        or row.get("measurement_domain") != PILOT_DOMAIN):
                    raise ValueError("Track B pilot admission row payload is inconsistent")


def _mode_count_for_cell(cell_id: str, history_policy: str) -> int:
    if history_policy == TRACKB_TEACHER_POLICY_V2:
        matches = [mode for mode, candidate in TRACKB_TEACHER_POLICY_V2_CELL_IDS.items()
                   if candidate == cell_id]
        if len(matches) != 1:
            raise ValueError("Track B V2 pilot cell must be trackb_feedback_history_v2_n3 or n4")
        return matches[0]
    if history_policy != TRACKB_TEACHER_POLICY_V1:
        raise ValueError("Track B pilot history policy is unsupported")
    prefix = "trackb_feedback_history_n"
    if not isinstance(cell_id, str) or not cell_id.startswith(prefix):
        raise ValueError("Track B pilot cell must be trackb_feedback_history_n3 or n4")
    suffix = cell_id[len(prefix):]
    if suffix not in ("3", "4"):
        raise ValueError("Track B pilot supports only n3 and n4 history cells")
    return int(suffix)


def _expected_cell(mode_count: int, history_policy: str) -> str:
    if history_policy == TRACKB_TEACHER_POLICY_V1:
        return f"trackb_feedback_history_n{mode_count}"
    if history_policy == TRACKB_TEACHER_POLICY_V2:
        return TRACKB_TEACHER_POLICY_V2_CELL_IDS[mode_count]
    raise ValueError("Track B pilot history policy is unsupported")


def _admitted_shard_path(root: Path, relative: str) -> Path:
    """Reject escaped or reparse shard paths before any shard read or hash."""
    if (type(relative) is not str or not relative or "\\" in relative
            or ":" in relative or "\x00" in relative):
        raise ValueError("Track B shard path must be a canonical relative path")
    parts = PurePosixPath(relative).parts
    if (PurePosixPath(relative).is_absolute() or not parts
            or any(part in (".", "..") for part in parts)
            or "/".join(parts) != relative):
        raise ValueError("Track B shard path must stay within the admitted root")
    current = root.resolve(strict=True)
    for part in parts:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 0x400:
            raise ValueError("Track B shard path must not contain a reparse component")
    if not current.is_file():
        raise ValueError(f"V2.4 READY shard is missing: {relative}")
    return current


def _query_cell_rows(index_bytes: bytes, *, cell_id: str
                     ) -> tuple[int, dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Validate an index snapshot and select both splits in one connection."""
    connection: sqlite3.Connection | None = None
    try:
        if hasattr(sqlite3.Connection, "deserialize"):
            connection = sqlite3.connect(":memory:")
            connection.deserialize(index_bytes)
        else:
            raise RuntimeError("Track B bounded index admission requires SQLite deserialize support")
        connection.row_factory = sqlite3.Row
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        count = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
        duplicate = int(connection.execute(
            "SELECT COUNT(*) FROM (SELECT sample_id FROM samples GROUP BY sample_id HAVING COUNT(*)>1)"
        ).fetchone()[0])
        if integrity != "ok" or duplicate:
            raise ValueError("Track B pilot SQLite index is corrupt or contains duplicate sample IDs")
        group_rows = connection.execute(
            "SELECT sample_id,split,session_id,acquisition_run_id,repeat_group_id,trajectory_id "
            "FROM samples"
        )
        split_report = validate_split_groups(dict(row) for row in group_rows)
        rows_by_split = {}
        for split in PILOT_SPLITS:
            selected = connection.execute(
                "SELECT * FROM samples WHERE cell_id=? AND split=? ORDER BY sample_id",
                (cell_id, split),
            ).fetchall()
            rows_by_split[split] = [dict(row) for row in selected]
        return count, rows_by_split, split_report
    finally:
        if connection is not None:
            connection.close()


def admit_pilot_dataset(root: str | Path, *, build_id: str | None = None,
                        cell_id: str, history_policy: str,
                        history_policy_sha256: str = "",
                        strict_shard_audit: bool = False) -> Pilot_dataset_admission:
    """Validate a Track B pilot root from bounded READY-bound snapshots."""
    if history_policy == TRACKB_TEACHER_POLICY_V1:
        if type(history_policy_sha256) is not str or history_policy_sha256:
            raise ValueError("Track B V1 admission cannot declare a policy SHA")
    elif history_policy == TRACKB_TEACHER_POLICY_V2:
        if history_policy_sha256 != TRACKB_TEACHER_POLICY_V2_SHA256:
            raise ValueError("Track B V2 admission requires the registered history policy SHA")
    else:
        raise ValueError("Track B pilot history policy is unsupported")
    mode_count = _mode_count_for_cell(cell_id, history_policy)
    expected_cell = _expected_cell(mode_count, history_policy)
    if cell_id != expected_cell:
        raise ValueError("Track B pilot cell identity mismatch")

    metadata = read_dataset_root_metadata(
        root, build_id=build_id, max_manifest_bytes=MAX_PILOT_MANIFEST_BYTES,
        max_ready_bytes=MAX_PILOT_READY_BYTES, max_contract_bytes=MAX_PILOT_CONTRACT_BYTES,
        reject_reparse=True,
    )
    ready = metadata.ready.decode()
    manifest = metadata.manifest.decode()
    validate_authority_flags(ready)
    validate_authority_flags(manifest)
    V24ReadyManifest(**ready).validate()
    shard_paths = {
        relative: _admitted_shard_path(metadata.root, relative)
        for relative in ready["shard_hashes"]
    }
    cells = manifest.get("cells")
    if type(cells) is not list:
        raise ValueError("Track B pilot manifest cells are missing")
    for cell in cells:
        if isinstance(cell, dict):
            reject_trackb_v2_intent(cell)
            if (str(cell.get("cell_id", "")).startswith("trackb_feedback_history")
                    and cell.get("dataset_family") != PILOT_FAMILY):
                raise ValueError("Track B history cell has a conflicting dataset family")
    trackb_cells = [cell for cell in cells if isinstance(cell, dict)
                    and cell.get("dataset_family") == PILOT_FAMILY]
    if len(trackb_cells) != 1:
        raise ValueError("Track B pilot root must contain exactly one authoritative history cell")
    manifest_cell = trackb_cells[0]
    if history_policy == TRACKB_TEACHER_POLICY_V1:
        if (manifest_cell.get("history_policy", TRACKB_TEACHER_POLICY_V1) != TRACKB_TEACHER_POLICY_V1
                or manifest_cell.get("history_policy_sha256") is not None):
            raise ValueError("Track B V1 reader rejects a V2 or mixed-policy root")
    else:
        validate_trackb_v2_cell(manifest_cell)
        if manifest_cell.get("history_policy_sha256") != history_policy_sha256:
            raise ValueError("Track B V2 manifest policy SHA differs from pilot config")
    if (manifest_cell.get("cell_id") != expected_cell
            or manifest_cell.get("N") != mode_count
            or manifest_cell.get("system_type") != f"{mode_count}qubit"):
        raise ValueError("Track B pilot manifest cell differs from policy, N, or system")

    validation_path = metadata.root / "reports" / f"{ready['build_id']}_validation.json"
    validation_bytes = read_stable_bytes(
        validation_path, label="Track B pilot validation report",
        expected_sha=ready["validation_report_sha"],
        max_bytes=MAX_PILOT_VALIDATION_REPORT_BYTES, reject_reparse=True,
    )
    validation = json.loads(validation_bytes.decode("utf-8"))
    if validation_bytes != canonical_json_bytes(validation):
        raise ValueError("Track B pilot validation report is not canonical JSON")
    if validation.get("build_id") != ready["build_id"]:
        raise ValueError("Track B pilot validation report build_id mismatch")
    validate_authority_flags(validation)

    index_path = metadata.root / "index" / "index.sqlite"
    index_bytes = read_stable_bytes(
        index_path, label="Track B pilot SQLite index", expected_sha=ready["index_sha"],
        max_bytes=MAX_PILOT_INDEX_BYTES, reject_reparse=True,
    )
    index_count, rows_by_split, verified_split_report = _query_cell_rows(
        index_bytes, cell_id=expected_cell)
    split_report = validation.get("split")
    if (type(validation.get("rows")) is not int or validation["rows"] != index_count
            or validation.get("truth_visibility") is not True
            or validation.get("physics_validators") is not True
            or type(split_report) is not dict
            or type(split_report.get("rows")) is not int
            or split_report.get("rows") != index_count
            or type(split_report.get("unique_samples")) is not int
            or split_report.get("unique_samples") != index_count
            or type(split_report.get("groups")) is not int or split_report["groups"] < 0):
        raise ValueError("Track B validation report rows or readiness claims differ from the index")
    if canonical_json_bytes(split_report) != canonical_json_bytes(verified_split_report):
        raise ValueError("Track B validation split report differs from the admitted index")
    identity = build_dataset_identity(metadata, index_sha=ready["index_sha"], index_rows=index_count)
    for split in PILOT_SPLITS:
        rows = rows_by_split[split]
        if not rows:
            raise ValueError("Track B pilot root is missing a registered train or validation split")
        for row in rows:
            row_policy = row.get("history_policy", history_policy)
            row_policy_sha = row.get("history_policy_sha256", history_policy_sha256)
            if (row.get("cell_id") != expected_cell or row.get("dataset_family") != PILOT_FAMILY
                    or row.get("measurement_domain") != PILOT_DOMAIN
                    or row.get("N") != mode_count or row.get("system_type") != f"{mode_count}qubit"
                    or row.get("split") != split or row_policy != history_policy
                    or (history_policy == TRACKB_TEACHER_POLICY_V2
                        and row_policy_sha != history_policy_sha256)):
                raise ValueError("Track B pilot index mixes policy, cell, system, or split identity")
            if (row.get("build_id") != ready["build_id"]
                    or row.get("contract_sha") != metadata.contract_sha
                    or row.get("observation_shard") not in shard_paths):
                raise ValueError("Track B pilot index row build, contract, or shard binding differs")

    path_stat_rows: dict[tuple[int, int, int], str] = {}
    selected_rows_by_shard: dict[str, list[dict[str, Any]]] = {}
    for split in PILOT_SPLITS:
        for row in rows_by_split[split]:
            row_index = validate_observation_row_index(row)
            relative = row["observation_shard"]
            shard_stat = shard_paths[relative].stat()
            physical_key = (shard_stat.st_dev, shard_stat.st_ino, row_index)
            previous_split = path_stat_rows.get(physical_key)
            if previous_split is not None and previous_split != split:
                raise ValueError("Track B train and validation splits alias one physical observation row")
            path_stat_rows[physical_key] = split
            selected_rows_by_shard.setdefault(relative, []).append(row)

    verified_owner_rows: dict[tuple[int, int, int], str] = {}
    for relative, expected_sha in ready["shard_hashes"].items():
        shard_path = shard_paths[relative]
        if strict_shard_audit:
            with open_verified_shard(shard_path, expected_sha) as shard:
                if shard.handle is None:
                    raise ValueError("Track B verified observation shard is closed")
                shard_info = validate_hdf5_handle(shard.handle)
                selected_rows = selected_rows_by_shard.get(relative, [])
                if selected_rows:
                    shard_metadata = shard_info["metadata"]
                    strict_file_identity = shard.file_identity
                    for row in selected_rows:
                        row_index = validate_observation_row_index(row)
                        physical_key = (*strict_file_identity, row_index)
                        previous_split = verified_owner_rows.get(physical_key)
                        if previous_split is not None and previous_split != row["split"]:
                            raise ValueError(
                                "Track B train and validation splits alias one verified observation row")
                        verified_owner_rows[physical_key] = row["split"]
                        expected_metadata = {
                            "store": "observation", "provenance": "synthetic",
                            "dataset_family": PILOT_FAMILY, "cell_id": expected_cell,
                            "split": row["split"], "system_type": f"{mode_count}qubit",
                            "N": mode_count, "measurement_domain": PILOT_DOMAIN,
                            "contract_sha": metadata.contract_sha, "build_id": ready["build_id"],
                        }
                        if (any(shard_metadata.get(key) != value
                                for key, value in expected_metadata.items())
                                or type(shard_metadata.get("N")) is not int):
                            raise ValueError("Track B observation shard metadata differs from its index rows")
                    validate_observation_identity_fields(
                        shard.handle, row_count=shard_info["row_count"])
                    for row in selected_rows:
                        row_index = validate_observation_row_index(row)
                        if row_index >= shard_info["row_count"]:
                            raise ValueError("Track B observation_row exceeds the shard row count")
                        validate_stored_observation_identity(
                            shard.handle["sample_id"][row_index],
                            shard.handle["row_valid_mask"][row_index],
                            sample_id=row["sample_id"],
                        )
                shard.verify_unchanged()

    return Pilot_dataset_admission(
        _FACTORY_TOKEN, root=metadata.root, build_id=ready["build_id"], cell_id=expected_cell,
        history_policy=history_policy, history_policy_sha256=history_policy_sha256,
        identity_payload=canonical_json_bytes(identity), manifest_payload=metadata.manifest.payload,
        validation_payload=validation_bytes,
        train_rows_payload=tuple(canonical_json_bytes(row) for row in rows_by_split["train"]),
        validation_rows_payload=tuple(canonical_json_bytes(row) for row in rows_by_split["validation"]),
    )


__all__ = [
    "MAX_PILOT_CONTRACT_BYTES", "MAX_PILOT_INDEX_BYTES", "MAX_PILOT_MANIFEST_BYTES",
    "MAX_PILOT_READY_BYTES", "MAX_PILOT_VALIDATION_REPORT_BYTES",
    "Pilot_dataset_admission", "admit_pilot_dataset",
]
