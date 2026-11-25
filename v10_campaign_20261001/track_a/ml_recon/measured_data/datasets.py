"""Strict manifest-backed measured datasets with explicit rho masking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .contracts import MeasuredSampleRecordV1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_measured_records(
    manifest_dir: str | Path,
    *,
    system_type: Optional[str] = None,
    trusted_only: bool = True,
) -> List[MeasuredSampleRecordV1]:
    directory = Path(manifest_dir)
    paths = []
    systems = (system_type,) if system_type else ("3qubit", "4qubit")
    for system in systems:
        suffix = "3q" if system == "3qubit" else "4q"
        paths.append(directory / f"trusted_samples_{suffix}.jsonl")
    records: List[MeasuredSampleRecordV1] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"strict measured manifest missing: {path}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = MeasuredSampleRecordV1.from_json_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid measured record at {path}:{line_number}: {exc}") from exc
            if trusted_only and record.quality_status != "trusted":
                raise ValueError(f"trusted manifest contains excluded sample {record.sample_id}")
            records.append(record)
    return records


def resolve_snapshot_paths(
    records: Sequence[MeasuredSampleRecordV1],
    *,
    data_root: str | Path,
) -> List[Path]:
    """Resolve and verify snapshot directories referenced by a strict manifest."""
    root = Path(data_root).resolve()
    paths: List[Path] = []
    for record in records:
        snapshot_dir = (root / record.snapshot_path_relative).resolve()
        try:
            snapshot_dir.relative_to(root)
        except ValueError as exc:
            raise ValueError("snapshot path escapes configured data root") from exc
        source_name = "manifest.json" if record.snapshot_manifest_schema.startswith(
            "snapshot_manifest_v"
        ) else "metadata.json"
        source_path = snapshot_dir / source_name
        if not source_path.is_file() or _sha256_file(source_path) != record.snapshot_manifest_hash:
            raise ValueError(
                f"snapshot manifest path/hash mismatch for {record.sample_id}: {source_path}"
            )
        paths.append(snapshot_dir)
    return paths


class MeasuredManifestDataset:
    """NumPy dataset that validates every manifest-bound path and array hash."""

    def __init__(
        self,
        records: Sequence[MeasuredSampleRecordV1],
        *,
        data_root: str | Path,
        split_assignments: Optional[Mapping[str, str]] = None,
        split_name: Optional[str] = None,
        allowed_sources: Optional[Iterable[str]] = None,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        assignments = dict(split_assignments or {})
        selected = list(records)
        if split_name is not None:
            selected = [item for item in selected if assignments.get(item.sample_id) == split_name]
        self.records = sorted(selected, key=lambda item: item.sample_id)
        self.allowed_sources = frozenset(allowed_sources or {"real"})

    def __len__(self) -> int:
        return len(self.records)

    def _snapshot_dir(self, record: MeasuredSampleRecordV1) -> Path:
        path = (self.data_root / record.snapshot_path_relative).resolve()
        try:
            path.relative_to(self.data_root)
        except ValueError as exc:
            raise ValueError("snapshot path escapes configured data root") from exc
        if not path.is_dir():
            raise FileNotFoundError(f"measured snapshot missing: {path}")
        return path

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        snapshot_dir = self._snapshot_dir(record)
        counts: Dict[str, np.ndarray] = {}
        for name, expected_hash in record.count_array_hashes.items():
            path = snapshot_dir / f"{name}.npy"
            if not path.is_file() or _sha256_file(path) != expected_hash:
                raise ValueError(f"count array path/hash mismatch for {record.sample_id}:{name}")
            counts[name] = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)

        rho = None
        if record.rho_available:
            rho_path = snapshot_dir / f"rho_{record.rho_kind}.npy"
            if not rho_path.is_file() or _sha256_file(rho_path) != record.rho_hash:
                raise ValueError(f"rho path/hash mismatch for {record.sample_id}")
            rho = np.asarray(np.load(rho_path, allow_pickle=False))
        # Missing rho is represented by None plus exact zero masks.  No zero
        # density image is constructed, so an encoder cannot accidentally use it.
        return {
            "sample_id": record.sample_id,
            "source_id": "real",
            "counts": counts,
            "target_state": record.exact_target_state_params,
            "sample_weight": np.asarray(record.sample_weight, dtype=np.float32),
            "rho": rho,
            "rho_available_mask": np.asarray(record.rho_branch_mask, dtype=np.float32),
            "rho_loss_mask": np.asarray(record.rho_loss_mask, dtype=np.float32),
            "command_supervision_mask": np.asarray(record.command_supervision_mask, dtype=np.float32),
            "command_table_id": record.command_table_id,
            "command_table_hash": record.command_table_hash,
            "command_provenance": record.command_provenance,
        }


def assert_real_only_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    for index, row in enumerate(rows):
        if str(row.get("source_id")) != "real":
            raise ValueError(f"real-only dataset received non-real row at index {index}")


__all__ = [
    "MeasuredManifestDataset",
    "assert_real_only_rows",
    "load_measured_records",
    "resolve_snapshot_paths",
]
