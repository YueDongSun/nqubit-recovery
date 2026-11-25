"""Deduplicate cell-constant V2.4 observation fields into one static store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .hashing import sha256_array, sha256_json
from .storage_hdf5 import write_hdf5_shard


STATIC_OBSERVATION_FIELDS = frozenset({
    "basis_descriptors",
    "setting_mask",
    "shots",
    "mode_features",
    "mode_mask",
    "physical_adjacency",
    "incidence",
})


def split_cell_static_fields(
    observation_arrays: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Split stacked rows and fail if a declared static field varies by row."""
    static_arrays: Dict[str, np.ndarray] = {
        "row_valid_mask": np.ones(1, dtype=np.bool_),
    }
    dynamic_arrays: Dict[str, np.ndarray] = {}
    for name, value in observation_arrays.items():
        array = np.asarray(value)
        if name not in STATIC_OBSERVATION_FIELDS:
            dynamic_arrays[name] = array
            continue
        if array.shape[0] < 1:
            raise ValueError(f"static field has no sample rows: {name}")
        reference = array[0]
        if not all(np.array_equal(reference, array[index], equal_nan=True)
                   for index in range(1, array.shape[0])):
            raise ValueError(f"declared cell-static field varies by sample: {name}")
        static_arrays[name] = np.asarray(reference)[None, ...]
    return static_arrays, dynamic_arrays


def cell_static_logical_sha(arrays: Mapping[str, np.ndarray], *,
                            cell_id: str, contract_sha: str,
                            schema_sha: str) -> str:
    return sha256_json({
        "cell_id": cell_id,
        "contract_sha": contract_sha,
        "schema_sha": schema_sha,
        "fields": {
            name: sha256_array(np.asarray(value))
            for name, value in sorted(arrays.items())
        },
    })


def write_cell_static_store(
    root: str | Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    root_path = Path(root)
    static_metadata = {**dict(metadata), "store": "cell_static"}
    logical_sha = cell_static_logical_sha(
        arrays, cell_id=str(metadata["cell_id"]),
        contract_sha=str(metadata["contract_sha"]),
        schema_sha=str(metadata["schema_sha"]))
    path = root_path / "cell_static" / f"{logical_sha}.h5"
    if path.exists():
        from .hashing import sha256_file
        from .storage_hdf5 import validate_hdf5_shard
        validate_hdf5_shard(path, static_metadata)
        return {
            "path": str(path),
            "relative_path": path.relative_to(root_path).as_posix(),
            "cell_static_sha": logical_sha,
            "file_sha": sha256_file(path),
            "reused": True,
        }
    result = write_hdf5_shard(path, arrays, static_metadata)
    return {
        "path": str(path),
        "relative_path": path.relative_to(root_path).as_posix(),
        "cell_static_sha": logical_sha,
        "file_sha": result["sha256"],
        "reused": False,
    }
