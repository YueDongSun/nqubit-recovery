"""Publish a verified Track B file through one continuously held Windows handle.

The caller owns the admitted parent-directory lease for the entire lifecycle.
Only the held file object is renamed or discarded; cleanup never unlinks a path.
"""

from __future__ import annotations

import ctypes
import os
import uuid
from pathlib import Path
from typing import Any

from .artifact_file_lease import (
    _identity,
    _kernel32 as _file_kernel32,
    _require_regular_handle,
    _sha256_handle,
)
from .contracts import require_sha256


_generic_read_write_delete = 0x80000000 | 0x40000000 | 0x00010000
_share_read = 1
_create_new = 1
_file_attribute_normal = 0x80
_file_rename_info_class = 3
_file_disposition_info_class = 4
_collision_errors = (80, 183)
_write_chunk_bytes = 1024 * 1024


class _File_rename_info(ctypes.Structure):
    _fields_ = [("replace_if_exists", ctypes.c_ubyte),
                ("root_directory", ctypes.c_void_p),
                ("file_name_length", ctypes.c_uint32),
                ("file_name", ctypes.c_uint16 * 1)]


def _kernel32() -> Any:
    library = _file_kernel32()
    library.WriteFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                                  ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    library.WriteFile.restype = ctypes.c_int
    library.FlushFileBuffers.argtypes = [ctypes.c_void_p]
    library.FlushFileBuffers.restype = ctypes.c_int
    library.SetFileInformationByHandle.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                  ctypes.c_void_p, ctypes.c_uint32]
    library.SetFileInformationByHandle.restype = ctypes.c_int
    return library


def _native_error(path: Path) -> OSError:
    error_code = ctypes.get_last_error()
    error = ctypes.WinError(error_code)
    error.filename = str(path)
    if error_code in _collision_errors:
        return FileExistsError(error.errno, error.strerror, str(path), error_code)
    return error


def _require_basename(name: str) -> None:
    if (type(name) is not str or not name or name in (".", "..")
            or any(character in name for character in ('/', '\\', ':', '\0'))
            or name.endswith((".", " "))):
        raise ValueError("artifact target name must be one canonical basename")


def _add_recovery_note(primary: BaseException, detail: str) -> None:
    note = f"RECOVERY_REQUIRED: {detail}"
    if hasattr(primary, "add_note"):
        primary.add_note(note)
    else:
        primary.args = (*primary.args, note)


class Staging_file_handle:
    """Own one new file from creation through verification and no-replace rename."""

    def __init__(self, path: Path, target: Path, handle: int) -> None:
        self.path = path
        self._target = target
        self._handle: int | None = handle
        self.file_identity: tuple[int, str] | None = None
        self.committed = False
        self.rename_outcome_unknown = False
        self._write_started = False
        self._discarded = False

    @classmethod
    def create(cls, parent: Path, target_name: str) -> "Staging_file_handle":
        if os.name != "nt":
            raise ValueError("Windows staging handle is required for guarded publication")
        _require_basename(target_name)
        parent = Path(parent)
        if not parent.is_absolute():
            raise ValueError("staging parent must be absolute and leased by the caller")
        path = parent / f".{target_name}.staging-{uuid.uuid4().hex}"
        library = _kernel32()
        handle = library.CreateFileW(
            str(path), _generic_read_write_delete, _share_read, None,
            _create_new, _file_attribute_normal, None,
        )
        if handle in (None, ctypes.c_void_p(-1).value):
            raise _native_error(path)
        return cls(path, parent / target_name, int(handle))

    def _require_open(self) -> int:
        if self._handle is None:
            raise RuntimeError("staging file handle was already closed")
        return self._handle

    def write_and_verify(self, payload: bytes, expected_sha: str) -> None:
        handle = self._require_open()
        if self.committed or self._discarded or self._write_started:
            raise RuntimeError("staging bytes may be written only once before publication")
        if type(payload) is not bytes:
            raise TypeError("artifact payload must be bytes")
        require_sha256(expected_sha, "artifact_expected_sha")
        self._write_started = True
        library = _kernel32()
        offset = 0
        while offset < len(payload):
            chunk = payload[offset:offset + _write_chunk_bytes]
            buffer = ctypes.create_string_buffer(chunk)
            count = ctypes.c_uint32()
            if not library.WriteFile(ctypes.c_void_p(handle), buffer, len(chunk),
                                     ctypes.byref(count), None):
                raise _native_error(self.path)
            if not 0 < count.value <= len(chunk):
                raise OSError("staging write did not make valid progress")
            offset += count.value
        if not library.FlushFileBuffers(ctypes.c_void_p(handle)):
            raise _native_error(self.path)
        _require_regular_handle(library, handle)
        if _sha256_handle(library, handle) != expected_sha:
            raise ValueError("staged artifact bytes do not match their expected SHA")
        self.file_identity = _identity(library, handle)

    def commit_no_replace(self, target: Path) -> None:
        handle = self._require_open()
        if self.committed or self._discarded or self.file_identity is None:
            raise RuntimeError("publication requires a verified uncommitted staging file")
        target = Path(target)
        if not target.is_absolute() or target != self._target:
            raise ValueError("publication target differs from the bound parent and basename")
        encoded_name = str(target).encode("utf-16-le")
        name_offset = _File_rename_info.file_name.offset
        # Keep a UTF-16 terminator in the allocation, outside FileNameLength.
        buffer_size = max(ctypes.sizeof(_File_rename_info), name_offset + len(encoded_name) + 2)
        if buffer_size > 0xffffffff:
            raise ValueError("artifact rename information exceeds the Windows buffer limit")
        buffer = ctypes.create_string_buffer(buffer_size)
        rename_info = _File_rename_info.from_buffer(buffer)
        rename_info.replace_if_exists = 0
        rename_info.root_directory = None
        rename_info.file_name_length = len(encoded_name)
        ctypes.memmove(ctypes.addressof(buffer) + name_offset, encoded_name, len(encoded_name))
        # An asynchronous exception can arrive after the native rename has
        # succeeded but before Python marks the object committed. In that
        # interval disposal must not be allowed to delete the target.
        self.rename_outcome_unknown = True
        renamed = _kernel32().SetFileInformationByHandle(
            ctypes.c_void_p(handle), _file_rename_info_class, buffer, buffer_size)
        if not renamed:
            self.rename_outcome_unknown = False
            raise _native_error(target)
        # Native rename is the filesystem commit. This assignment records its
        # known successful outcome; the preceding unknown state protects the
        # interval in which an asynchronous exception could arrive.
        self.committed = True
        self.rename_outcome_unknown = False

    def discard_owned(self) -> None:
        """Delete only the held precommit object, then release its handle."""
        if self.committed or self._handle is None:
            return
        if self.rename_outcome_unknown:
            try:
                self.close()
            except BaseException as error:
                _add_recovery_note(error, "rename outcome unknown; inspect target and staging")
                raise
            raise RuntimeError(
                "RECOVERY_REQUIRED: rename outcome unknown; inspect target and staging")
        disposition_error: BaseException | None = None
        if not self._discarded:
            try:
                delete_file = ctypes.c_ubyte(1)
                if not _kernel32().SetFileInformationByHandle(
                        ctypes.c_void_p(self._handle), _file_disposition_info_class,
                        ctypes.byref(delete_file), ctypes.sizeof(delete_file)):
                    raise _native_error(self.path)
                self._discarded = True
            except BaseException as error:
                disposition_error = error
        close_error: BaseException | None = None
        try:
            self.close()
        except BaseException as error:
            close_error = error
        if disposition_error is not None:
            _add_recovery_note(disposition_error, f"staging disposition failed for {self.path}")
            if close_error is not None:
                _add_recovery_note(disposition_error, f"staging handle close also failed: {close_error}")
            raise disposition_error
        if close_error is not None:
            raise close_error

    def close(self) -> None:
        """Release the native handle without performing pathname cleanup."""
        if self._handle is None:
            return
        if not _kernel32().CloseHandle(ctypes.c_void_p(self._handle)):
            raise _native_error(self.path)
        self._handle = None
