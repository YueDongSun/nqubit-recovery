"""Worker-local bounded HDF5 handle cache and contiguous shard reads."""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import h5py
import numpy as np

from .verified_shard_handle import Verified_shard_handle, open_verified_shard


class V24Hdf5HandleCache:
    """Keep read-only handles local to one DataLoader worker process."""

    def __init__(self, maximum_handles: int = 8):
        if maximum_handles < 1:
            raise ValueError("maximum_handles must be positive")
        self.maximum_handles = int(maximum_handles)
        self._pid = os.getpid()
        self._handles: OrderedDict[str, h5py.File | Verified_shard_handle] = OrderedDict()
        self.open_count = 0

    def _reset_after_process_change(self) -> None:
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
            self.open_count = 0

    def _open(self, path: str | Path, expected_sha: str = "") -> h5py.File | Verified_shard_handle:
        self._reset_after_process_change()
        key = str(Path(path).resolve())
        handle = self._handles.pop(key, None)
        if handle is None:
            handle = (open_verified_shard(key, expected_sha) if expected_sha
                      else h5py.File(key, "r", swmr=True))
            self.open_count += 1
        elif isinstance(handle, Verified_shard_handle):
            if handle.expected_sha != expected_sha:
                self._handles[key] = handle
                raise ValueError("cached V2.4 shard was admitted with another expected SHA")
            self._verify_handle(handle)
        elif expected_sha:
            self._handles[key] = handle
            raise ValueError("cached legacy V2.4 shard cannot satisfy a verified SHA request")
        self._handles[key] = handle
        while len(self._handles) > self.maximum_handles:
            _, evicted = self._handles.popitem(last=False)
            evicted.close()
        return handle

    @staticmethod
    def _hdf_handle(handle: h5py.File | Verified_shard_handle) -> h5py.File:
        if isinstance(handle, Verified_shard_handle):
            if handle.handle is None:
                raise ValueError("verified shard HDF5 handle is closed")
            return handle.handle
        return handle

    def _verify_handle(self, handle: h5py.File | Verified_shard_handle) -> None:
        if isinstance(handle, Verified_shard_handle):
            try:
                handle.verify_unchanged()
            except BaseException:
                for key, cached_handle in tuple(self._handles.items()):
                    if cached_handle is handle:
                        self._handles.pop(key, None)
                        break
                try:
                    handle.close()
                except BaseException:
                    pass
                raise

    def read_row(self, path: str | Path, row: int,
                 fields: Sequence[str] | None = None, *,
                 expected_sha: str = "") -> Dict[str, np.ndarray]:
        handle = self._open(path, expected_sha)
        hdf_handle = self._hdf_handle(handle)
        self._verify_handle(handle)
        try:
            names = tuple(sorted(hdf_handle.keys())) if fields is None else tuple(fields)
            missing = set(names) - set(hdf_handle.keys())
            if missing:
                raise KeyError(f"shard missing fields: {sorted(missing)}")
            return {name: np.asarray(hdf_handle[name][int(row)]) for name in names}
        finally:
            self._verify_handle(handle)

    def read_rows(self, path: str | Path, rows: Sequence[int],
                  fields: Sequence[str] | None = None, *,
                  expected_sha: str = "") -> Dict[str, np.ndarray]:
        """Read one contiguous slice when possible, otherwise one indexed block."""
        if not rows:
            return {}
        handle = self._open(path, expected_sha)
        hdf_handle = self._hdf_handle(handle)
        self._verify_handle(handle)
        try:
            names = tuple(sorted(hdf_handle.keys())) if fields is None else tuple(fields)
            missing = set(names) - set(hdf_handle.keys())
            if missing:
                raise KeyError(f"shard missing fields: {sorted(missing)}")
            indices = np.asarray(rows, dtype=np.int64)
            if np.any(indices < 0):
                raise IndexError("negative HDF5 row index")
            contiguous = bool(len(indices) == 1 or np.all(np.diff(indices) == 1))
            increasing_unique = bool(len(indices) == 1 or np.all(np.diff(indices) > 0))
            if contiguous:
                selector = slice(int(indices[0]), int(indices[-1]) + 1)
                return {name: np.asarray(hdf_handle[name][selector]) for name in names}
            if increasing_unique:
                return {name: np.asarray(hdf_handle[name][indices]) for name in names}
            return {
                name: np.stack([np.asarray(hdf_handle[name][int(index)]) for index in indices])
                for name in names
            }
        finally:
            self._verify_handle(handle)

    def close(self) -> None:
        handles = tuple(self._handles.values())
        self._handles.clear()
        close_error: BaseException | None = None
        for handle in handles:
            try:
                handle.close()
            except BaseException as error:
                if close_error is None:
                    close_error = error
        if close_error is not None:
            raise close_error

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
