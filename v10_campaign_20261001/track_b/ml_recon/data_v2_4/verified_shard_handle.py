"""Hash-bound HDF5 access through one stable Python file object."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path
from typing import BinaryIO

import h5py

from .hdf5_storage_contract import validate_self_contained_hdf5


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _hash_open_file(handle: BinaryIO, *, chunk_size: int = 1024 * 1024) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    while chunk := handle.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


class Verified_shard_handle:
    """Own an HDF5 handle and its exact underlying, SHA-verified file object."""

    def __init__(self, path: str | Path, expected_sha: str):
        if type(expected_sha) is not str or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise ValueError("verified shard requires a lowercase SHA-256 digest")
        self.path = Path(path)
        self.expected_sha = expected_sha
        self._raw_handle: BinaryIO | None = None
        self.handle: h5py.File | None = None
        self._identity: tuple[int, int, int, int] | None = None
        try:
            raw_handle = self.path.open("rb")
            self._raw_handle = raw_handle
            before = os.fstat(raw_handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"V2.4 shard is not a regular file: {self.path}")
            digest = _hash_open_file(raw_handle)
            after_hash = os.fstat(raw_handle.fileno())
            if _file_identity(before) != _file_identity(after_hash):
                raise ValueError(f"V2.4 shard changed while hashing: {self.path}")
            if digest != expected_sha:
                raise ValueError(f"V2.4 shard SHA mismatch on first open: {self.path}")
            raw_handle.seek(0)
            hdf_handle = h5py.File(raw_handle, "r")
            self.handle = hdf_handle
            validate_self_contained_hdf5(hdf_handle)
            after_open = os.fstat(raw_handle.fileno())
            if _file_identity(after_hash) != _file_identity(after_open):
                raise ValueError(f"V2.4 shard changed while opening HDF5: {self.path}")
            self._identity = _file_identity(after_open)
            self.verify_unchanged()
        except BaseException:
            self.close()
            raise

    @property
    def file_identity(self) -> tuple[int, int]:
        """Return the opened file object's device and inode identity."""
        if self._identity is None or self._raw_handle is None:
            raise ValueError("verified shard handle is closed")
        return self._identity[0], self._identity[1]

    def verify_unchanged(self) -> None:
        raw_handle = self._raw_handle
        if raw_handle is None or self.handle is None or self._identity is None:
            raise ValueError("verified shard handle is closed")
        before = os.fstat(raw_handle.fileno())
        if _file_identity(before) != self._identity:
            raise ValueError(f"V2.4 shard changed after verification: {self.path}")

    def close(self) -> None:
        hdf_handle = self.handle
        self.handle = None
        raw_handle = self._raw_handle
        self._raw_handle = None
        self._identity = None
        close_error: BaseException | None = None
        if hdf_handle is not None:
            try:
                hdf_handle.close()
            except BaseException as error:
                close_error = error
        if raw_handle is not None:
            try:
                raw_handle.close()
            except BaseException as error:
                if close_error is None:
                    close_error = error
        if close_error is not None:
            raise close_error

    def __enter__(self) -> "Verified_shard_handle":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def open_verified_shard(path: str | Path, expected_sha: str) -> Verified_shard_handle:
    return Verified_shard_handle(path, expected_sha)


__all__ = ["Verified_shard_handle", "open_verified_shard"]
