"""Atomic homogeneous HDF5 shard writer and validating reader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping

import h5py
import numpy as np

from .hashing import canonical_json_bytes, sha256_file
from .schema_registry import validate_persisted_arrays


def _validated_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise ValueError(f"HDF5 object/vlen fields are forbidden: {name}")
    if array.dtype.kind == "U":
        maximum = max((len(item.encode("utf-8")) for item in array.reshape(-1)), default=1)
        array = array.astype(f"S{maximum}")
    if array.dtype.kind == "c":
        raise ValueError(f"complex field {name} must be split into explicit re/im arrays")
    if array.dtype.kind in ("i", "u", "f") and array.dtype.itemsize > 1:
        little_endian = array.dtype.newbyteorder("<")
        array = array.astype(little_endian, copy=False)
    if array.ndim == 0:
        raise ValueError(f"shard field {name} must have a row dimension")
    return np.ascontiguousarray(array)


def validate_hdf5_shard(path: str | Path, expected_metadata: Mapping[str, Any] | None = None
                        ) -> Dict[str, Any]:
    source = Path(path)
    with h5py.File(source, "r") as handle:
        names = sorted(handle.keys())
        if not names:
            raise ValueError("empty V2.4 shard")
        row_counts = {int(handle[name].shape[0]) for name in names}
        if len(row_counts) != 1:
            raise ValueError("all shard arrays must have the same row count")
        for name in names:
            dataset = handle[name]
            if dataset.dtype.kind in ("O", "U", "c"):
                raise ValueError(f"invalid HDF5 dtype for {name}: {dataset.dtype}")
        metadata_json = bytes(handle.attrs["metadata_json"])
        import json
        metadata = json.loads(metadata_json.decode("utf-8"))
        if metadata_json != canonical_json_bytes(metadata):
            raise ValueError("shard metadata is not canonical JSON")
        if expected_metadata is not None and metadata != dict(expected_metadata):
            raise ValueError("shard metadata does not match expected homogeneous cell")
    return {"row_count": next(iter(row_counts)), "fields": names,
            "metadata": metadata, "sha256": sha256_file(source)}


def write_hdf5_shard(path: str | Path, arrays: Mapping[str, np.ndarray],
                     metadata: Mapping[str, Any], compression: str = "gzip") -> Dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    values = {name: _validated_array(name, value) for name, value in arrays.items()}
    validate_persisted_arrays(values, metadata)
    row_counts = {array.shape[0] for array in values.values()}
    if not values or len(row_counts) != 1:
        raise ValueError("homogeneous shard arrays require one non-empty row count")
    with h5py.File(temporary, "w", libver="latest") as handle:
        for name, array in sorted(values.items()):
            use_compression = compression not in ("", "none", None)
            kwargs = ({"compression": compression, "shuffle": True}
                      if array.size and use_compression else {})
            handle.create_dataset(name, data=array, **kwargs)
        handle.attrs["metadata_json"] = np.bytes_(canonical_json_bytes(dict(metadata)))
        handle.flush()
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    validation = validate_hdf5_shard(temporary, metadata)
    os.replace(temporary, destination)
    final_sha = sha256_file(destination)
    if final_sha != validation["sha256"]:
        raise RuntimeError("atomic HDF5 rename changed shard content")
    return {**validation, "path": str(destination), "sha256": final_sha}


def read_hdf5_row(path: str | Path, row: int, fields: tuple[str, ...] | None = None
                  ) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r", swmr=True) as handle:
        names = tuple(sorted(handle.keys())) if fields is None else fields
        missing = set(names) - set(handle.keys())
        if missing:
            raise KeyError(f"shard missing fields: {sorted(missing)}")
        return {name: np.asarray(handle[name][row]) for name in names}
