"""Windows directory identity leases for one research-only artifact lifecycle."""

from __future__ import annotations

import ctypes
import os
import stat
from dataclasses import dataclass
from pathlib import Path


_file_read_attributes = 0x80
_delete_access = 0x10000
_synchronize_access = 0x100000
_share_read_write = 0x01 | 0x02
_share_all = _share_read_write | 0x04
_open_existing = 3
_directory_flags = 0x02000000 | 0x00200000  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
_file_attribute_reparse_point = 0x400
_file_id_info_class = 18
_object_case_insensitive = 0x40
_file_attribute_normal = 0x80
_file_create = 2
_file_created = 2
_file_directory_file = 0x01
_file_synchronous_io_nonalert = 0x20


class _File_id_info(ctypes.Structure):
    _fields_ = [("volume_serial_number", ctypes.c_uint64),
                ("file_id", ctypes.c_ubyte * 16)]


class _Unicode_string(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint16),
                ("maximum_length", ctypes.c_uint16),
                ("buffer", ctypes.c_void_p)]


class _Object_attributes(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint32),
                ("root_directory", ctypes.c_void_p),
                ("object_name", ctypes.POINTER(_Unicode_string)),
                ("attributes", ctypes.c_uint32),
                ("security_descriptor", ctypes.c_void_p),
                ("security_quality_of_service", ctypes.c_void_p)]


class _Io_status_union(ctypes.Union):
    _fields_ = [("status", ctypes.c_int32),
                ("pointer", ctypes.c_void_p)]


class _Io_status_block(ctypes.Structure):
    _anonymous_ = ("result",)
    _fields_ = [("result", _Io_status_union),
                ("information", ctypes.c_size_t)]


def _kernel32():
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    library.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32,
                                    ctypes.c_uint32, ctypes.c_void_p,
                                    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    library.CreateFileW.restype = ctypes.c_void_p
    library.GetFileInformationByHandleEx.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                     ctypes.c_void_p, ctypes.c_uint32]
    library.GetFileInformationByHandleEx.restype = ctypes.c_int
    library.CloseHandle.argtypes = [ctypes.c_void_p]
    library.CloseHandle.restype = ctypes.c_int
    return library


def _ntdll():
    library = ctypes.WinDLL("ntdll")
    library.NtCreateFile.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32,
        ctypes.POINTER(_Object_attributes), ctypes.POINTER(_Io_status_block),
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
    ]
    library.NtCreateFile.restype = ctypes.c_int32
    library.RtlNtStatusToDosError.argtypes = [ctypes.c_int32]
    library.RtlNtStatusToDosError.restype = ctypes.c_uint32
    return library


def _directory_identity(library, handle: int) -> tuple[int, str]:
    info = _File_id_info()
    if not library.GetFileInformationByHandleEx(
            ctypes.c_void_p(handle), _file_id_info_class,
            ctypes.byref(info), ctypes.sizeof(info)):
        error = ctypes.get_last_error()
        raise OSError(error, os.strerror(error))
    volume = int(info.volume_serial_number)
    file_id = bytes(info.file_id).hex()
    if volume == 0 or file_id == "00" * 16:
        raise ValueError("directory filesystem has no supported file identity")
    return volume, file_id


def _require_plain_directory(path: Path) -> None:
    metadata = os.lstat(path)
    if (not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_file_attributes & _file_attribute_reparse_point):
        raise ValueError(f"directory is missing or reparse-point aliased: {path}")


@dataclass
class Directory_lease:
    """An opened, non-delete-shared directory with a checked lexical binding."""

    path: Path
    handle: int
    volume_serial_number: int
    file_id: str

    @classmethod
    def create_child(cls, parent_lease: "Directory_lease", name: str) -> "Directory_lease":
        """Create a fresh child and retain the handle returned by that same call."""
        if os.name != "nt":
            raise ValueError("Windows directory lease is required for learner artifacts")
        if not isinstance(parent_lease, cls):
            raise TypeError("parent_lease must be a directory lease")
        if (not isinstance(name, str) or not name or name in (".", "..")
                or any(character in name for character in ("/", "\\", ":", "\0"))
                or name.endswith((".", " "))):
            raise ValueError("child directory name must be one canonical basename")
        encoded_name = name.encode("utf-16-le")
        if len(encoded_name) + 2 > 0xffff:
            raise ValueError("child directory name exceeds UNICODE_STRING length")
        parent_lease.verify()
        candidate = parent_lease.path / name
        name_buffer = ctypes.create_unicode_buffer(name)
        object_name = _Unicode_string(
            len(encoded_name), len(encoded_name) + 2,
            ctypes.cast(name_buffer, ctypes.c_void_p).value,
        )
        attributes = _Object_attributes(
            ctypes.sizeof(_Object_attributes), ctypes.c_void_p(parent_lease.handle),
            ctypes.pointer(object_name), _object_case_insensitive, None, None,
        )
        io_status = _Io_status_block()
        handle = ctypes.c_void_p()
        library = _ntdll()
        status = library.NtCreateFile(
            ctypes.byref(handle),
            _file_read_attributes | _delete_access | _synchronize_access,
            ctypes.byref(attributes), ctypes.byref(io_status), None,
            _file_attribute_normal, _share_read_write, _file_create,
            _file_directory_file | _file_synchronous_io_nonalert, None, 0,
        )
        if status != 0:
            error_code = int(library.RtlNtStatusToDosError(status))
            error = ctypes.WinError(error_code)
            error.filename = str(candidate)
            if hasattr(error, "add_note"):
                error.add_note(f"NtCreateFile status: 0x{status & 0xffffffff:08x}")
            if handle.value and not _kernel32().CloseHandle(handle) and hasattr(error, "add_note"):
                error.add_note("RECOVERY_REQUIRED: failed creation returned an unclosed handle")
            raise error
        if not handle.value:
            raise RuntimeError("NtCreateFile succeeded without a directory handle")
        try:
            if io_status.information != _file_created:
                raise ValueError("directory creation did not report FILE_CREATED")
            kernel = _kernel32()
            volume, file_id = _directory_identity(kernel, int(handle.value))
            lease = cls(candidate, int(handle.value), volume, file_id)
            lease.verify()
            parent_lease.verify()
            return lease
        except BaseException as error:
            if not _kernel32().CloseHandle(handle) and hasattr(error, "add_note"):
                error.add_note("RECOVERY_REQUIRED: created directory handle close failed")
            raise

    @classmethod
    def open(cls, path: Path) -> "Directory_lease":
        if os.name != "nt":
            raise ValueError("Windows directory lease is required for learner artifacts")
        candidate = Path(path)
        if not candidate.is_absolute() or candidate.resolve(strict=True) != candidate:
            raise ValueError("directory path must be absolute and non-aliased")
        _require_plain_directory(candidate)
        library = _kernel32()
        invalid_handle = ctypes.c_void_p(-1).value
        handle = library.CreateFileW(str(candidate), _file_read_attributes | _delete_access,
                                     _share_read_write, None, _open_existing,
                                     _directory_flags, None)
        if handle in (None, invalid_handle):
            error = ctypes.get_last_error()
            raise OSError(error, os.strerror(error), str(candidate))
        try:
            volume, file_id = _directory_identity(library, int(handle))
            lease = cls(candidate, int(handle), volume, file_id)
            lease.verify()
            return lease
        except BaseException as error:
            if not library.CloseHandle(ctypes.c_void_p(handle)) and hasattr(error, "add_note"):
                error.add_note("RECOVERY_REQUIRED: directory admission handle close failed")
            raise

    def binding(self) -> dict[str, object]:
        return {"schema": "TrackBDirectoryBindingV1", "path": str(self.path),
                "volume_serial_number": self.volume_serial_number, "file_id": self.file_id}

    def verify(self) -> None:
        if self.handle is None:
            raise RuntimeError("directory lease was already closed")
        if self.path.resolve(strict=True) != self.path:
            raise ValueError(f"directory path became aliased: {self.path}")
        _require_plain_directory(self.path)
        library = _kernel32()
        if _directory_identity(library, self.handle) != (self.volume_serial_number, self.file_id):
            raise ValueError("held directory identity differs from its admitted binding")
        invalid_handle = ctypes.c_void_p(-1).value
        # Zero access still names the current lexical directory.  Sharing DELETE
        # makes this probe compatible with other accessors while the held lease
        # denies delete sharing to competing rename/delete requests.
        probe = library.CreateFileW(str(self.path), 0, _share_all, None,
                                    _open_existing, _directory_flags, None)
        if probe in (None, invalid_handle):
            error = ctypes.get_last_error()
            raise OSError(error, os.strerror(error), str(self.path))
        try:
            identity = _directory_identity(library, int(probe))
        finally:
            if not library.CloseHandle(ctypes.c_void_p(probe)):
                error = ctypes.get_last_error()
                raise OSError(error, os.strerror(error), str(self.path))
        if identity != (self.volume_serial_number, self.file_id):
            raise ValueError(f"directory lexical binding changed: {self.path}")

    def close(self) -> None:
        if self.handle is None:
            return
        if not _kernel32().CloseHandle(ctypes.c_void_p(self.handle)):
            error = ctypes.get_last_error()
            raise OSError(error, os.strerror(error), str(self.path))
        self.handle = None


def open_directory_ancestry(path: Path, *, anchor: Path) -> list[Directory_lease]:
    """Pin the exact trusted output parent and each existing run component."""
    candidate = Path(path)
    anchor = Path(anchor)
    if not candidate.is_absolute() or not anchor.is_absolute() or (
            candidate != anchor and anchor not in candidate.parents):
        raise ValueError("directory path must be absolute beneath its fixed anchor")
    components = (anchor,) if candidate == anchor else tuple(reversed(
        (candidate, *candidate.parents[:candidate.parents.index(anchor) + 1])))
    leases: list[Directory_lease] = []
    try:
        for component in components:
            leases.append(Directory_lease.open(component))
        verify_directory_leases(leases)
        return leases
    except BaseException as error:
        close_errors: list[OSError] = []
        for lease in reversed(leases):
            try:
                lease.close()
            except OSError as close_error:
                close_errors.append(close_error)
        if close_errors and hasattr(error, "add_note"):
            error.add_note(f"RECOVERY_REQUIRED: partial ancestry lease close failed: {close_errors}")
        raise


def verify_directory_leases(leases: list[Directory_lease]) -> None:
    for lease in leases:
        lease.verify()


def close_directory_leases(leases: list[Directory_lease]) -> None:
    failures: list[OSError] = []
    for lease in reversed(leases):
        try:
            lease.close()
        except OSError as error:
            failures.append(error)
    if failures:
        raise RuntimeError(f"RECOVERY_REQUIRED: directory lease close failed: {failures}")
