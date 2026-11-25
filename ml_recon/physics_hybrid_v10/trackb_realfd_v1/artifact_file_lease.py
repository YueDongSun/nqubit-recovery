"""Windows file-identity leases for a guarded Track B artifact lifecycle.

The lease protects one already-published regular file against ordinary
concurrent write, rename, or delete. It is acquired only after the publisher
has removed its staging hard link. This is not protection against a malicious
process with the ability to replace all evidence after the lease is released.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .artifact_directory_lease import Directory_lease
from .contracts import require_sha256


_generic_read = 0x80000000
_delete_access = 0x00010000
_share_read = 0x00000001
_share_all = 0x00000001 | 0x00000002 | 0x00000004
_open_existing = 3
_open_reparse_point = 0x00200000
_file_attribute_directory = 0x00000010
_file_attribute_reparse_point = 0x00000400
_file_attribute_tag_info_class = 9
_file_id_info_class = 18
_file_begin = 0
_read_chunk_bytes = 1024 * 1024


class _File_id_info(ctypes.Structure):
    _fields_ = [("volume_serial_number", ctypes.c_uint64),
                ("file_id", ctypes.c_ubyte * 16)]


class _File_attribute_tag_info(ctypes.Structure):
    _fields_ = [("file_attributes", ctypes.c_uint32),
                ("reparse_tag", ctypes.c_uint32)]


def _kernel32():
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    library.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32,
                                    ctypes.c_uint32, ctypes.c_void_p,
                                    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    library.CreateFileW.restype = ctypes.c_void_p
    library.GetFileInformationByHandleEx.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                     ctypes.c_void_p, ctypes.c_uint32]
    library.GetFileInformationByHandleEx.restype = ctypes.c_int
    library.GetFileSizeEx.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)]
    library.GetFileSizeEx.restype = ctypes.c_int
    library.SetFilePointerEx.argtypes = [ctypes.c_void_p, ctypes.c_int64,
                                        ctypes.POINTER(ctypes.c_int64), ctypes.c_uint32]
    library.SetFilePointerEx.restype = ctypes.c_int
    library.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                                 ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    library.ReadFile.restype = ctypes.c_int
    library.CloseHandle.argtypes = [ctypes.c_void_p]
    library.CloseHandle.restype = ctypes.c_int
    return library


def _identity(library, handle: int) -> tuple[int, str]:
    info = _File_id_info()
    if not library.GetFileInformationByHandleEx(ctypes.c_void_p(handle),
                                                _file_id_info_class, ctypes.byref(info),
                                                ctypes.sizeof(info)):
        error = ctypes.get_last_error()
        raise OSError(error, os.strerror(error))
    volume = int(info.volume_serial_number)
    file_id = bytes(info.file_id).hex()
    if volume == 0 or file_id == "00" * 16:
        raise ValueError("file filesystem has no supported stable identity")
    return volume, file_id


def _require_regular_handle(library, handle: int) -> None:
    info = _File_attribute_tag_info()
    if not library.GetFileInformationByHandleEx(ctypes.c_void_p(handle),
                                                _file_attribute_tag_info_class,
                                                ctypes.byref(info), ctypes.sizeof(info)):
        error = ctypes.get_last_error()
        raise OSError(error, os.strerror(error))
    if info.file_attributes & (_file_attribute_reparse_point | _file_attribute_directory):
        raise ValueError("artifact handle is a directory or reparse point")


def _require_regular_path(path: Path) -> None:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("artifact path must be absolute and non-aliased")
    metadata = os.lstat(path)
    if (not stat.S_ISREG(metadata.st_mode)
            or getattr(metadata, "st_file_attributes", 0) & _file_attribute_reparse_point):
        raise ValueError("artifact path must name an ordinary regular file")


def _iter_handle_chunks(library, handle: int, *,
                        max_bytes: int | None = None) -> Iterator[bytes]:
    size = ctypes.c_int64()
    if not library.GetFileSizeEx(ctypes.c_void_p(handle), ctypes.byref(size)):
        error = ctypes.get_last_error()
        raise OSError(error, os.strerror(error))
    if size.value < 0:
        raise ValueError("artifact file size is invalid")
    if max_bytes is not None and size.value > max_bytes:
        raise ValueError("artifact exceeds its bounded read size")
    if not library.SetFilePointerEx(ctypes.c_void_p(handle), 0, None, _file_begin):
        error = ctypes.get_last_error()
        raise OSError(error, os.strerror(error))
    remaining = size.value
    while remaining:
        chunk_size = min(remaining, _read_chunk_bytes)
        buffer = ctypes.create_string_buffer(chunk_size)
        count = ctypes.c_uint32()
        if not library.ReadFile(ctypes.c_void_p(handle), buffer, chunk_size,
                                ctypes.byref(count), None):
            error = ctypes.get_last_error()
            raise OSError(error, os.strerror(error))
        if count.value == 0 or count.value > chunk_size:
            raise OSError("artifact changed or ended during leased read")
        yield buffer.raw[:count.value]
        remaining -= count.value


def _read_handle_bytes(library, handle: int, *, max_bytes: int | None = None) -> bytes:
    return b"".join(_iter_handle_chunks(library, handle, max_bytes=max_bytes))


def _sha256_handle(library, handle: int) -> str:
    digest = hashlib.sha256()
    for chunk in _iter_handle_chunks(library, handle):
        digest.update(chunk)
    return digest.hexdigest()


def file_identity_from_descriptor(descriptor: int) -> tuple[int, str]:
    """Read FILE_ID_INFO from an open publisher-owned file descriptor."""
    if os.name != "nt":
        raise ValueError("Windows file identity is required for guarded publication")
    if type(descriptor) is not int or descriptor < 0:
        raise ValueError("publisher descriptor is invalid")
    import msvcrt

    native_handle = msvcrt.get_osfhandle(descriptor)
    library = _kernel32()
    _require_regular_handle(library, native_handle)
    return _identity(library, native_handle)


@dataclass
class File_lease:
    """Pin one verified file object and its expected bytes until close."""

    path: Path
    parent_lease: Directory_lease
    handle: int | None
    volume_serial_number: int
    file_id: str
    expected_sha: str

    @classmethod
    def open_verified(cls, path: Path, expected_sha: str,
                      parent_lease: Directory_lease, *,
                      expected_identity: tuple[int, str] | None = None) -> "File_lease":
        if os.name != "nt":
            raise ValueError("Windows file lease is required for guarded learner artifacts")
        require_sha256(expected_sha, "leased_artifact_sha")
        if not isinstance(parent_lease, Directory_lease):
            raise TypeError("parent_lease must be a directory lease")
        candidate = Path(path)
        if candidate.parent != parent_lease.path:
            raise ValueError("file lease does not bind the exact parent directory")
        parent_lease.verify()
        _require_regular_path(candidate)
        library = _kernel32()
        invalid_handle = ctypes.c_void_p(-1).value
        handle = library.CreateFileW(str(candidate), _generic_read | _delete_access,
                                     _share_read, None, _open_existing,
                                     _open_reparse_point, None)
        if handle in (None, invalid_handle):
            error = ctypes.get_last_error()
            raise OSError(error, os.strerror(error), str(candidate))
        try:
            volume, file_id = _identity(library, int(handle))
            _require_regular_handle(library, int(handle))
            if expected_identity is not None and (volume, file_id) != expected_identity:
                raise ValueError("published artifact file identity changed before lease")
            lease = cls(candidate, parent_lease, int(handle), volume, file_id,
                        expected_sha)
            lease.verify_binding()
            return lease
        except BaseException as error:
            if not library.CloseHandle(ctypes.c_void_p(handle)) and hasattr(error, "add_note"):
                error.add_note("RECOVERY_REQUIRED: file admission handle close failed")
            raise

    def _verify_path_identity(self) -> None:
        if self.handle is None:
            raise RuntimeError("file lease was already closed")
        self.parent_lease.verify()
        _require_regular_path(self.path)
        library = _kernel32()
        _require_regular_handle(library, self.handle)
        if _identity(library, self.handle) != (self.volume_serial_number, self.file_id):
            raise ValueError("held artifact file identity changed")
        invalid_handle = ctypes.c_void_p(-1).value
        probe = library.CreateFileW(str(self.path), 0, _share_all, None,
                                    _open_existing, _open_reparse_point, None)
        if probe in (None, invalid_handle):
            error = ctypes.get_last_error()
            raise OSError(error, os.strerror(error), str(self.path))
        try:
            _require_regular_handle(library, int(probe))
            current = _identity(library, int(probe))
        finally:
            if not library.CloseHandle(ctypes.c_void_p(probe)):
                error = ctypes.get_last_error()
                raise OSError(error, os.strerror(error), str(self.path))
        if current != (self.volume_serial_number, self.file_id):
            raise ValueError("artifact lexical file binding changed")

    def read_bytes(self, *, max_bytes: int | None = None) -> bytes:
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
            raise ValueError("max_bytes must be a nonnegative integer")
        self._verify_path_identity()
        payload = _read_handle_bytes(_kernel32(), self.handle, max_bytes=max_bytes)
        if hashlib.sha256(payload).hexdigest() != self.expected_sha:
            raise ValueError("leased artifact bytes differ from expected SHA")
        self._verify_path_identity()
        return payload

    def verify_binding(self) -> None:
        self._verify_path_identity()
        if _sha256_handle(_kernel32(), self.handle) != self.expected_sha:
            raise ValueError("leased artifact bytes differ from expected SHA")
        self._verify_path_identity()

    def binding(self) -> dict[str, object]:
        self.verify_binding()
        return {"schema": "TrackBFileBindingV1", "path": str(self.path),
                "volume_serial_number": self.volume_serial_number,
                "file_id": self.file_id, "expected_sha": self.expected_sha}

    def close(self) -> None:
        if self.handle is None:
            return
        if not _kernel32().CloseHandle(ctypes.c_void_p(self.handle)):
            error = ctypes.get_last_error()
            raise OSError(error, os.strerror(error), str(self.path))
        self.handle = None


def close_file_leases(leases: list[File_lease]) -> None:
    """Release every owned file handle while surfacing all close failures."""
    errors: list[OSError] = []
    for lease in reversed(leases):
        try:
            lease.close()
        except OSError as error:
            errors.append(error)
    if errors:
        raise RuntimeError(f"RECOVERY_REQUIRED: file lease close failed: {errors}")
