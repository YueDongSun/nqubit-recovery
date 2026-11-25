"""Worker-local bounded HDF5 handle cache and contiguous shard reads."""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import h5py
import numpy as np

from .hashing import sha256_file


class V24Hdf5HandleCache:
    """Keep read-only SWMR handles local to one DataLoader worker process."""

    def __init__(self, maximum_handles: int = 8):
        if maximum_handles < 1:
            raise ValueError("maximum_handles must be positive")
        self.maximum_handles = int(maximum_handles)
        self._pid = os.getpid()
        self._handles: OrderedDict[str, h5py.File] = OrderedDict()
        self._verified_sha: Dict[str, str] = {}
        self.open_count = 0

    def _reset_after_process_change(self) -> None:
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
            self._verified_sha = {}
            self.open_count = 0

    def _open(self, path: str | Path, expected_sha: str = "") -> h5py.File:
        self._reset_after_process_change()
        key = str(Path(path).resolve())
        if expected_sha:
            cached_sha = self._verified_sha.get(key)
            if cached_sha != expected_sha:
                actual_sha = sha256_file(key)
                if actual_sha != expected_sha:
                    raise ValueError(f"V2.4 shard SHA mismatch on first open: {path}")
                self._verified_sha[key] = actual_sha
        handle = self._handles.pop(key, None)
        if handle is None:
            handle = h5py.File(key, "r", swmr=True)
            self.open_count += 1
        self._handles[key] = handle
        while len(self._handles) > self.maximum_handles:
            _, evicted = self._handles.popitem(last=False)
            evicted.close()
        return handle

    def read_row(self, path: str | Path, row: int,
                 fields: Sequence[str] | None = None, *,
                 expected_sha: str = "") -> Dict[str, np.ndarray]:
        handle = self._open(path, expected_sha)
        names = tuple(sorted(handle.keys())) if fields is None else tuple(fields)
        missing = set(names) - set(handle.keys())
        if missing:
            raise KeyError(f"shard missing fields: {sorted(missing)}")
        return {name: np.asarray(handle[name][int(row)]) for name in names}

    def read_rows(self, path: str | Path, rows: Sequence[int],
                  fields: Sequence[str] | None = None, *,
                  expected_sha: str = "") -> Dict[str, np.ndarray]:
        """Read one contiguous slice when possible, otherwise one indexed block."""
        if not rows:
            return {}
        handle = self._open(path, expected_sha)
        names = tuple(sorted(handle.keys())) if fields is None else tuple(fields)
        missing = set(names) - set(handle.keys())
        if missing:
            raise KeyError(f"shard missing fields: {sorted(missing)}")
        indices = np.asarray(rows, dtype=np.int64)
        if np.any(indices < 0):
            raise IndexError("negative HDF5 row index")
        contiguous = bool(len(indices) == 1 or np.all(np.diff(indices) == 1))
        increasing_unique = bool(len(indices) == 1 or np.all(np.diff(indices) > 0))
        if contiguous:
            selector = slice(int(indices[0]), int(indices[-1]) + 1)
            return {name: np.asarray(handle[name][selector]) for name in names}
        if increasing_unique:
            return {name: np.asarray(handle[name][indices]) for name in names}
        return {
            name: np.stack([np.asarray(handle[name][int(index)]) for index in indices])
            for name in names
        }

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> Mapping[str, int]:
        self.close()
        return {"maximum_handles": self.maximum_handles}

    def __setstate__(self, state: Mapping[str, int]) -> None:
        self.__init__(int(state["maximum_handles"]))

    def __enter__(self) -> "V24Hdf5HandleCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def read_hdf5_rows_cached(cache: V24Hdf5HandleCache, path: str | Path,
                          rows: Iterable[int], fields: Sequence[str] | None = None,
                          *, expected_sha: str = "") -> Dict[str, np.ndarray]:
    return cache.read_rows(path, tuple(rows), fields, expected_sha=expected_sha)


__all__ = ["V24Hdf5HandleCache", "read_hdf5_rows_cached"]
