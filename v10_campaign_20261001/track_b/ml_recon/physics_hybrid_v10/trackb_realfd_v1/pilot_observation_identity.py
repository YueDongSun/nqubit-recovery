"""Shared identity checks for indexed Track B observation rows."""

from __future__ import annotations

from typing import Any, Mapping

import h5py
import numpy as np


def validate_observation_row_index(row: Mapping[str, Any]) -> int:
    """Require an exact, nonnegative SQLite integer observation row."""
    value = row.get("observation_row")
    if type(value) is not int or value < 0:
        raise ValueError("Track B observation_row must be an exact nonnegative integer")
    return value


def validate_observation_identity_fields(handle: h5py.File, *, row_count: int) -> None:
    """Validate canonical stored sample identity and row-validity arrays."""
    if "sample_id" not in handle or "row_valid_mask" not in handle:
        raise ValueError("Track B observation shard requires sample_id and row_valid_mask")
    sample_ids = handle["sample_id"]
    valid_mask = handle["row_valid_mask"]
    if (sample_ids.shape != (row_count,) or sample_ids.dtype.kind != "S"
            or valid_mask.shape != (row_count,) or valid_mask.dtype != np.dtype(np.bool_)):
        raise ValueError("Track B observation identity or row_valid_mask schema mismatch")


def validate_stored_observation_identity(sample_value: Any, valid_value: Any, *,
                                         sample_id: str) -> None:
    """Match the HDF5 row ID exactly to SQLite and require its valid-row mask."""
    if type(sample_id) is not str or not sample_id:
        raise ValueError("Track B SQLite sample_id must be a non-empty string")
    sample_array = np.asarray(sample_value)
    valid_array = np.asarray(valid_value)
    if sample_array.shape != () or sample_array.dtype.kind != "S":
        raise ValueError("Track B observation sample_id must be fixed-width UTF-8 bytes")
    if valid_array.shape != () or valid_array.dtype != np.dtype(np.bool_):
        raise ValueError("Track B observation row_valid_mask must be a boolean scalar")
    try:
        stored_sample_id = bytes(sample_array.item()).decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("Track B observation sample_id is not valid UTF-8") from error
    if stored_sample_id != sample_id:
        raise ValueError("Track B SQLite sample_id differs from the stored observation row")
    if not bool(valid_array.item()):
        raise ValueError("Track B observation row_valid_mask must be true")


__all__ = [
    "validate_observation_identity_fields",
    "validate_observation_row_index",
    "validate_stored_observation_identity",
]
