"""In-memory V2.4 encoders for handle-owned behavior dataset publication.

The existing generic path writers replace destination names. This adapter
instead returns fully validated bytes to the native no-replace publisher;
it leaves their APIs and the persisted V2.4 schemas unchanged.
"""

from __future__ import annotations

import io
import sqlite3
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from ml_recon.data_v2_4.hashing import canonical_json_bytes
from ml_recon.data_v2_4.index_sqlite import INDEX_COLUMNS
from ml_recon.data_v2_4.schema_registry import validate_persisted_arrays
from ml_recon.data_v2_4.storage_hdf5 import _validated_array, validate_hdf5_handle


integer_index_columns = frozenset({"N", "observation_row", "truth_row", "derived_rho8",
                                   "derived_count_mle", "derived_classifier", "derived_control"})
maximum_encoded_bytes = 512 * 1024 * 1024


def encode_behavior_hdf5(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> bytes:
    """Encode one homogeneous schema-validated shard without a filesystem path."""
    values = {name: _validated_array(name, value) for name, value in arrays.items()}
    validate_persisted_arrays(values, metadata)
    row_counts = {array.shape[0] for array in values.values()}
    if not values or len(row_counts) != 1 or next(iter(row_counts)) <= 0:
        raise ValueError("behavior shard requires one nonempty homogeneous row count")
    if sum(array.nbytes for array in values.values()) > maximum_encoded_bytes:
        raise ValueError("behavior shard exceeds the bounded in-memory publication size")
    with io.BytesIO() as buffer:
        with h5py.File(buffer, "w", libver="latest") as handle:
            for name, array in sorted(values.items()):
                options = {"compression": "gzip", "shuffle": True} if array.size else {}
                handle.create_dataset(name, data=array, **options)
            handle.attrs["metadata_json"] = np.bytes_(canonical_json_bytes(dict(metadata)))
            handle.flush()
            validation = validate_hdf5_handle(handle, metadata)
            if validation["row_count"] != next(iter(row_counts)):
                raise RuntimeError("behavior HDF5 row count changed during serialization")
        payload = buffer.getvalue()
    if len(payload) > maximum_encoded_bytes:
        raise ValueError("encoded behavior shard exceeds its byte bound")
    return payload


def encode_behavior_index(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize the existing row-index schema from a private SQLite connection."""
    if not rows or len(rows) > 1200:
        raise ValueError("behavior index requires a bounded nonempty row set")
    if not hasattr(sqlite3.Connection, "serialize"):
        raise RuntimeError("behavior index publication requires SQLite serialize support")
    column_sql = []
    for name in INDEX_COLUMNS:
        kind = "INTEGER" if name in integer_index_columns else "TEXT"
        constraint = "PRIMARY KEY" if name == "sample_id" else "NOT NULL"
        column_sql.append(f"{name} {kind} {constraint}")
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(f"CREATE TABLE samples ({','.join(column_sql)})")
        placeholders = ",".join("?" for _ in INDEX_COLUMNS)
        statement = f"INSERT INTO samples ({','.join(INDEX_COLUMNS)}) VALUES ({placeholders})"
        for row in rows:
            if set(row) != set(INDEX_COLUMNS):
                raise ValueError("behavior index row differs from the persisted V2.4 schema")
            if any(type(row[name]) is not (int if name in integer_index_columns else str)
                   for name in INDEX_COLUMNS):
                raise ValueError("behavior index row has noncanonical scalar types")
            connection.execute(statement, tuple(row[name] for name in INDEX_COLUMNS))
        connection.execute("CREATE INDEX sample_cell_split ON samples(cell_id,split,sample_id)")
        connection.execute("CREATE INDEX sample_groups ON samples(session_id,trajectory_id,repeat_group_id)")
        connection.commit()
        if (connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
                or connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0] != len(rows)):
            raise RuntimeError("behavior index failed SQLite integrity or row-count validation")
        payload = connection.serialize()
    finally:
        connection.close()
    if len(payload) > maximum_encoded_bytes:
        raise ValueError("encoded behavior index exceeds its byte bound")
    return payload


__all__ = ["encode_behavior_hdf5", "encode_behavior_index"]
