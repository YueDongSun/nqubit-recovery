"""Detached CPU snapshots for portable, immutable training checkpoint storage."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import uuid
import shutil
import stat
import io
import ctypes
import ctypes.wintypes
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_COMMIT_SCHEMA = "V10CheckpointCommitV2"
CHECKPOINT_MANIFEST_SCHEMA = "V10TrainingCheckpointManifestV2"
MAX_CHECKPOINT_GENERATIONS = 2
MAX_DESCRIPTOR_BYTES = 2 * 1024 * 1024
MAX_METRICS_JOURNAL_BYTES = 64 * 1024 * 1024


def _is_reparse_point(path: Path) -> bool:
    """Return whether an existing path is a link or Windows reparse point."""
    try:
        info = path.lstat()
    except OSError as exc:
        raise RuntimeError("checkpoint output root identity is unavailable") from exc
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def validate_output_root(output_dir: Path) -> Path:
    """Reject link/reparse output roots and unsafe parent traversal."""
    requested = Path(output_dir)
    if any(part == ".." for part in requested.parts):
        raise ValueError("checkpoint output root contains unsafe parent traversal")
    absolute = requested.absolute()
    resolved = absolute.resolve(strict=False)
    if resolved != absolute:
        raise ValueError("checkpoint output root or parent is a symlink/reparse point")
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if not current.exists():
            break
        if _is_reparse_point(current):
            raise ValueError("checkpoint output root or parent is a symlink/reparse point")
    if not absolute.parent.exists() or _is_reparse_point(absolute.parent):
        raise ValueError("checkpoint output root parent must be an existing safe directory")
    return resolved


def _output_root_identity_chain(output_dir: Path) -> tuple[tuple[str, int, int], ...]:
    """Capture every existing ancestor identity through the output root."""
    absolute = output_dir.absolute()
    chain = []
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if not current.exists() or _is_reparse_point(current):
            raise RuntimeError("checkpoint output root identity chain changed")
        info = current.stat()
        chain.append((str(current), info.st_dev, info.st_ino))
    return tuple(chain)


class Checkpoint_writer_lease:
    def __init__(self, path: Path, token: str, handle: io.IOBase,
                 identity: tuple[int, int], root_identity: tuple[int, int],
                 root_identity_chain: tuple[tuple[str, int, int], ...],
                 root_directory_fd: int | None,
                 native_root_identity: tuple[int, int] | None = None):
        self.path = path
        self.token = token
        self.handle = handle
        self.identity = identity
        self.root_identity = root_identity
        self.root_identity_chain = root_identity_chain
        self.root_directory_fd = root_directory_fd
        self.native_root_identity = native_root_identity
        self.directory_identities: dict[str, tuple[int, int]] = {}
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        try:
            try:
                self.assert_active()
                if os.name == "nt":
                    import msvcrt
                    self.handle.seek(0)
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                # Closing the descriptor is the final lock release and must
                # remain reliable after root replacement or identity drift.
                pass
        finally:
            try:
                self.handle.close()
            finally:
                if self.root_directory_fd is not None:
                    try:
                        os.close(self.root_directory_fd)
                    except OSError:
                        pass
                    self.root_directory_fd = None
            self.released = True

    def assert_rooted_path(self, relative_path: str, *, allow_missing: bool = False,
                           allow_directory: bool = False) -> Path:
        """Validate a critical artifact path relative to the owned root."""
        if (not isinstance(relative_path, str) or not relative_path
                or relative_path.startswith("/") or "\\" in relative_path
                or any(part in ("", ".", "..") for part in relative_path.split("/"))):
            raise ValueError("critical output path is unsafe")
        self.assert_active()
        candidate = self.path.parent / relative_path
        resolved = candidate.resolve(strict=False)
        root = self.path.parent.resolve()
        try:
            relative_parts = resolved.relative_to(root).parts
        except ValueError as exc:
            raise RuntimeError("critical output path escaped owned root") from exc
        current = root
        for part in relative_parts[:-1]:
            current /= part
            if not current.exists() or _is_reparse_point(current):
                raise RuntimeError("critical output ancestor is a reparse point")
            info = current.stat()
            key = current.relative_to(root).as_posix()
            identity = (info.st_dev, info.st_ino)
            previous = self.directory_identities.setdefault(key, identity)
            if previous != identity:
                raise RuntimeError("critical output ancestor identity changed")
        if _is_reparse_point(candidate.parent):
            raise RuntimeError("critical output path escaped owned root")
        if candidate.exists() and _is_reparse_point(candidate):
            raise RuntimeError("critical output path is a reparse point")
        if (not allow_missing and not candidate.is_file()
                and not (allow_directory and candidate.is_dir())):
            raise RuntimeError("critical output path is unavailable")
        return candidate

    def assert_active(self) -> None:
        if (self.released or _is_reparse_point(self.path) or not self.path.is_file()
                or self.path.stat().st_nlink != 1 or _is_reparse_point(self.path.parent)):
            raise RuntimeError("checkpoint writer lease is not active")
        root_stat = self.path.parent.stat()
        if (root_stat.st_dev, root_stat.st_ino) != self.root_identity:
            raise RuntimeError("checkpoint output root identity changed")
        if _output_root_identity_chain(self.path.parent) != self.root_identity_chain:
            raise RuntimeError("checkpoint output ancestor identity changed")
        if self.root_directory_fd is not None:
            opened_root = os.fstat(self.root_directory_fd)
            if (opened_root.st_dev, opened_root.st_ino) != self.root_identity:
                raise RuntimeError("checkpoint output root handle identity changed")
        current = self.path.stat()
        opened = os.fstat(self.handle.fileno())
        if (current.st_dev, current.st_ino) != self.identity or (opened.st_dev, opened.st_ino) != self.identity:
            raise RuntimeError("checkpoint writer lease path identity changed")
        self.handle.seek(0)
        if self.handle.read() != self.token.encode("utf-8"):
            raise RuntimeError("checkpoint writer lease token changed")

    def __enter__(self):
        self.assert_active()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False


def acquire_checkpoint_writer_lease(output_dir: Path, execution_id: str) -> Checkpoint_writer_lease:
    validated_root = validate_output_root(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rebound_root = validate_output_root(output_dir)
    if rebound_root != validated_root:
        raise RuntimeError("checkpoint output root changed during creation")
    root_stat = output_dir.stat()
    root_identity = (root_stat.st_dev, root_stat.st_ino)
    root_identity_chain = _output_root_identity_chain(output_dir)
    native_root_identity = None
    if os.name == "nt":
        native_kernel32 = _windows_kernel32()
        native_root_handle = _open_windows_directory_handle(native_kernel32, output_dir)
        try:
            native_info = _validate_windows_native_handle(
                native_kernel32, native_root_handle, directory=True, single_link=False)
            native_root_identity = (
                native_info.volume_serial_number,
                (native_info.file_index_high << 32) | native_info.file_index_low)
        finally:
            native_kernel32.CloseHandle(native_root_handle)
    root_directory_fd = None
    if os.name != "nt":
        try:
            root_directory_fd = os.open(
                str(output_dir), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            raise RuntimeError("checkpoint output root directory handle is unavailable") from exc
    path = output_dir / ".checkpoint_writer.lock"
    token = json.dumps({"schema": "V10CheckpointWriterLeaseV1",
                        "execution_id": execution_id, "token": uuid.uuid4().hex},
                       sort_keys=True, separators=(",", ":")) + "\n"
    if os.name == "nt":
        try:
            handle = _open_windows_lease_handle(path)
        except Exception:
            if root_directory_fd is not None:
                os.close(root_directory_fd)
            raise
    else:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            if root_directory_fd is not None:
                file_descriptor = os.open(
                    ".checkpoint_writer.lock", flags, 0o600,
                    dir_fd=root_directory_fd)
            else:
                file_descriptor = os.open(str(path), flags, 0o600)
        except OSError as exc:
            if root_directory_fd is not None:
                os.close(root_directory_fd)
            raise RuntimeError("checkpoint writer lease path is unsafe") from exc
        handle = os.fdopen(file_descriptor, "r+b", closefd=True)
    try:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("checkpoint writer lease must be a regular single-link file")
        if info.st_size == 0:
            handle.write(b"\0"); handle.flush(); os.fsync(handle.fileno())
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.seek(0); handle.truncate(); handle.write(token.encode("utf-8")); handle.flush(); os.fsync(handle.fileno())
    except (OSError, IOError, RuntimeError) as exc:
        handle.close()
        if root_directory_fd is not None:
            os.close(root_directory_fd)
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError("checkpoint output root has a live or unknown competing writer") from exc
    info = os.fstat(handle.fileno())
    lease = Checkpoint_writer_lease(
        path, token, handle, (info.st_dev, info.st_ino), root_identity,
        root_identity_chain, root_directory_fd, native_root_identity)
    try:
        lease.assert_active()
    except Exception:
        lease.release()
        raise
    return lease


def _open_windows_lease_handle(path: Path) -> io.IOBase:
    import msvcrt
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [ctypes.wintypes.LPCWSTR, ctypes.wintypes.DWORD,
                                     ctypes.wintypes.DWORD, ctypes.c_void_p,
                                     ctypes.wintypes.DWORD, ctypes.wintypes.DWORD,
                                     ctypes.wintypes.HANDLE]
    kernel32.CreateFileW.restype = ctypes.wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
    kernel32.CloseHandle.restype = ctypes.wintypes.BOOL
    kernel32.GetFileInformationByHandleEx.argtypes = [ctypes.wintypes.HANDLE,
                                                       ctypes.c_int, ctypes.c_void_p,
                                                       ctypes.wintypes.DWORD]
    kernel32.GetFileInformationByHandleEx.restype = ctypes.wintypes.BOOL
    generic_read_write = 0x80000000 | 0x40000000
    share_all = 0x00000001 | 0x00000002 | 0x00000004
    open_always = 4
    file_flag_open_reparse_point = 0x00200000
    handle = kernel32.CreateFileW(str(path), generic_read_write, share_all, None,
                                  open_always, file_flag_open_reparse_point, None)
    invalid = ctypes.wintypes.HANDLE(-1).value
    if handle == invalid:
        raise RuntimeError("checkpoint writer lease native open failed")
    attributes = ctypes.wintypes.DWORD()
    reparse_tag = ctypes.wintypes.DWORD()
    file_attribute_tag_info = 9
    class File_attribute_tag_info(ctypes.Structure):
        _fields_ = [("file_attributes", ctypes.wintypes.DWORD),
                    ("reparse_tag", ctypes.wintypes.DWORD)]
    info = File_attribute_tag_info()
    if not kernel32.GetFileInformationByHandleEx(handle, file_attribute_tag_info,
                                                  ctypes.byref(info), ctypes.sizeof(info)):
        kernel32.CloseHandle(handle)
        raise RuntimeError("checkpoint writer lease attributes unavailable")
    if info.file_attributes & 0x40000000 or info.reparse_tag:
        kernel32.CloseHandle(handle)
        raise RuntimeError("checkpoint writer lease reparse point rejected")
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
    except OSError:
        kernel32.CloseHandle(handle)
        raise
    return os.fdopen(fd, "r+b", closefd=True)


def _file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _safe_read_bytes(path: Path, *, maximum_bytes: int = MAX_DESCRIPTOR_BYTES) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    with os.fdopen(descriptor, "rb", closefd=True) as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum_bytes:
            raise ValueError("unsafe bounded JSON file")
        payload = stream.read(maximum_bytes + 1)
        after = os.fstat(stream.fileno())
        if (before.st_ino != after.st_ino or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns or len(payload) > maximum_bytes):
            raise ValueError("bounded JSON file changed during read")
    return payload


class _Windows_unicode_string(ctypes.Structure):
    _fields_ = [("length", ctypes.wintypes.USHORT),
                ("maximum_length", ctypes.wintypes.USHORT),
                ("buffer", ctypes.wintypes.LPWSTR)]


class _Windows_object_attributes(ctypes.Structure):
    _fields_ = [("length", ctypes.wintypes.ULONG),
                ("root_directory", ctypes.wintypes.HANDLE),
                ("object_name", ctypes.POINTER(_Windows_unicode_string)),
                ("attributes", ctypes.wintypes.ULONG),
                ("security_descriptor", ctypes.c_void_p),
                ("security_quality_of_service", ctypes.c_void_p)]


class _Windows_io_status_block(ctypes.Structure):
    _fields_ = [("status", ctypes.c_void_p), ("information", ctypes.c_void_p)]


class _Windows_file_disposition_information(ctypes.Structure):
    _fields_ = [("delete_file", ctypes.wintypes.BOOLEAN)]


class _Windows_file_attribute_tag_info(ctypes.Structure):
    _fields_ = [("file_attributes", ctypes.wintypes.DWORD),
                ("reparse_tag", ctypes.wintypes.DWORD)]


class _Windows_by_handle_file_information(ctypes.Structure):
    _fields_ = [("file_attributes", ctypes.wintypes.DWORD),
                ("creation_time_low", ctypes.wintypes.DWORD),
                ("creation_time_high", ctypes.wintypes.DWORD),
                ("last_access_time_low", ctypes.wintypes.DWORD),
                ("last_access_time_high", ctypes.wintypes.DWORD),
                ("last_write_time_low", ctypes.wintypes.DWORD),
                ("last_write_time_high", ctypes.wintypes.DWORD),
                ("volume_serial_number", ctypes.wintypes.DWORD),
                ("file_size_high", ctypes.wintypes.DWORD),
                ("file_size_low", ctypes.wintypes.DWORD),
                ("number_of_links", ctypes.wintypes.DWORD),
                ("file_index_high", ctypes.wintypes.DWORD),
                ("file_index_low", ctypes.wintypes.DWORD)]


def _windows_kernel32() -> ctypes.WinDLL:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [ctypes.wintypes.LPCWSTR, ctypes.wintypes.DWORD,
                                     ctypes.wintypes.DWORD, ctypes.c_void_p,
                                     ctypes.wintypes.DWORD, ctypes.wintypes.DWORD,
                                     ctypes.wintypes.HANDLE]
    kernel32.CreateFileW.restype = ctypes.wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
    kernel32.CloseHandle.restype = ctypes.wintypes.BOOL
    kernel32.GetFileInformationByHandleEx.argtypes = [ctypes.wintypes.HANDLE,
                                                       ctypes.c_int, ctypes.c_void_p,
                                                       ctypes.wintypes.DWORD]
    kernel32.GetFileInformationByHandleEx.restype = ctypes.wintypes.BOOL
    kernel32.GetFileInformationByHandle.argtypes = [ctypes.wintypes.HANDLE,
                                                    ctypes.POINTER(_Windows_by_handle_file_information)]
    kernel32.GetFileInformationByHandle.restype = ctypes.wintypes.BOOL
    return kernel32


def _validate_windows_native_handle(kernel32: ctypes.WinDLL, handle: ctypes.wintypes.HANDLE,
                                    *, directory: bool, single_link: bool) -> _Windows_by_handle_file_information:
    tag_info = _Windows_file_attribute_tag_info()
    if not kernel32.GetFileInformationByHandleEx(handle, 9, ctypes.byref(tag_info), ctypes.sizeof(tag_info)):
        raise RuntimeError("Windows artifact handle attributes unavailable")
    if tag_info.file_attributes & 0x40000000 or tag_info.reparse_tag:
        raise RuntimeError("Windows artifact handle is a reparse point")
    info = _Windows_by_handle_file_information()
    if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(info)):
        raise RuntimeError("Windows artifact handle identity unavailable")
    is_directory = bool(info.file_attributes & 0x10)
    if is_directory is not directory or (single_link and info.number_of_links != 1):
        raise RuntimeError("Windows artifact handle type or link count is invalid")
    return info


def _open_windows_directory_handle(kernel32: ctypes.WinDLL, path: Path) -> ctypes.wintypes.HANDLE:
    handle = kernel32.CreateFileW(
        str(path), 0x0080 | 0x00100000, 0x00000007, None, 3,
        0x02000000 | 0x00200000, None)
    invalid = ctypes.wintypes.HANDLE(-1).value
    if handle == invalid:
        raise OSError(ctypes.get_last_error(), "Windows directory handle open failed")
    try:
        _validate_windows_native_handle(kernel32, handle, directory=True, single_link=False)
        return handle
    except Exception:
        kernel32.CloseHandle(handle)
        raise


def _windows_relative_exclusive_create(parent_handle: ctypes.wintypes.HANDLE,
                                       filename: str) -> ctypes.wintypes.HANDLE:
    return _windows_relative_open(parent_handle, filename, disposition=2,
                                  desired_access=0x40000000 | 0x00100000 | 0x00010000 | 0x00000080)


def _windows_relative_directory_open(parent_handle: ctypes.wintypes.HANDLE,
                                     component: str) -> ctypes.wintypes.HANDLE:
    return _windows_relative_open(
        parent_handle, component, disposition=1,
        desired_access=0x00100000 | 0x00000080,
        create_options=0x00000001 | 0x00000020 | 0x00200000)


def _windows_relative_open(parent_handle: ctypes.wintypes.HANDLE, filename: str,
                           *, disposition: int, desired_access: int,
                           create_options: int = 0x00000040 | 0x00000020 | 0x00200000) -> ctypes.wintypes.HANDLE:
    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtCreateFile.argtypes = [ctypes.POINTER(ctypes.wintypes.HANDLE), ctypes.wintypes.ULONG,
                                   ctypes.POINTER(_Windows_object_attributes),
                                   ctypes.POINTER(_Windows_io_status_block), ctypes.c_void_p,
                                   ctypes.wintypes.ULONG, ctypes.wintypes.ULONG, ctypes.wintypes.ULONG,
                                   ctypes.wintypes.ULONG, ctypes.c_void_p, ctypes.wintypes.ULONG]
    ntdll.NtCreateFile.restype = ctypes.c_long
    filename_buffer = ctypes.create_unicode_buffer(filename)
    unicode_name = _Windows_unicode_string(
        len(filename) * 2, ctypes.sizeof(filename_buffer),
        ctypes.cast(filename_buffer, ctypes.wintypes.LPWSTR))
    attributes = _Windows_object_attributes(
        ctypes.sizeof(_Windows_object_attributes), parent_handle, ctypes.pointer(unicode_name),
        0x00000040, None, None)
    status_block = _Windows_io_status_block()
    handle = ctypes.wintypes.HANDLE()
    status = ntdll.NtCreateFile(
        ctypes.byref(handle), desired_access, ctypes.byref(attributes),
        ctypes.byref(status_block), None, 0x80, 0x00000007, disposition,
        create_options, None, 0)
    if status < 0:
        raise OSError(f"NtCreateFile relative exclusive creation failed: 0x{status & 0xffffffff:08x}")
    return handle


def _windows_delete_relative_exact(kernel32: ctypes.WinDLL,
                                   parent_handle: ctypes.wintypes.HANDLE,
                                   filename: str, expected_info: _Windows_by_handle_file_information) -> None:
    handle = _windows_relative_open(
        parent_handle, filename, disposition=1,
        desired_access=0x00010000 | 0x00100000 | 0x00000080)
    try:
        info = _validate_windows_native_handle(kernel32, handle, directory=False, single_link=True)
        identity = (info.volume_serial_number, info.file_index_high, info.file_index_low)
        expected = (expected_info.volume_serial_number, expected_info.file_index_high,
                    expected_info.file_index_low)
        if identity != expected:
            raise RuntimeError("Windows cleanup target identity changed")
        _windows_mark_handle_deleted(handle)
    finally:
        kernel32.CloseHandle(handle)


def _windows_mark_handle_deleted(handle: ctypes.wintypes.HANDLE) -> None:
    """Mark an already identity-checked Windows file handle for deletion."""
    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtSetInformationFile.argtypes = [ctypes.wintypes.HANDLE,
                                           ctypes.POINTER(_Windows_io_status_block),
                                           ctypes.c_void_p, ctypes.wintypes.ULONG, ctypes.wintypes.ULONG]
    ntdll.NtSetInformationFile.restype = ctypes.c_long
    disposition = _Windows_file_disposition_information(1)
    status_block = _Windows_io_status_block()
    status = ntdll.NtSetInformationFile(
        handle, ctypes.byref(status_block), ctypes.byref(disposition),
        ctypes.sizeof(disposition), 13)
    if status < 0:
        raise OSError(f"NtSetInformationFile delete failed: 0x{status & 0xffffffff:08x}")


def _copy_rooted_artifact_windows(source: Path, relative_target: str,
                                   writer_lease: Checkpoint_writer_lease) -> dict[str, object]:
    kernel32 = _windows_kernel32()
    root = writer_lease.path.parent
    parts = relative_target.split("/")
    source_before = source.stat()
    parent_handles: list[ctypes.wintypes.HANDLE] = []
    created_info = None
    target_created = False
    try:
        root_handle = _open_windows_directory_handle(kernel32, root)
        parent_handles.append(root_handle)
        root_info = _validate_windows_native_handle(kernel32, root_handle, directory=True, single_link=False)
        root_stat = root.stat()
        root_file_index = (root_info.file_index_high << 32) | root_info.file_index_low
        opened_root_identity = (root_info.volume_serial_number, root_file_index)
        if (writer_lease.native_root_identity is not None
                and opened_root_identity != writer_lease.native_root_identity):
            raise RuntimeError("Windows root handle identity differs from leased root")
        if root_file_index != root_stat.st_ino:
            raise RuntimeError("Windows root handle identity differs from leased root")
        for component in parts[:-1]:
            next_handle = _windows_relative_directory_open(parent_handles[-1], component)
            try:
                _validate_windows_native_handle(kernel32, next_handle, directory=True, single_link=False)
            except Exception:
                kernel32.CloseHandle(next_handle)
                raise
            parent_handles.append(next_handle)
        parent_handle = parent_handles[-1]
        native_handle = _windows_relative_exclusive_create(parent_handle, parts[-1])
        try:
            info = _validate_windows_native_handle(kernel32, native_handle, directory=False, single_link=True)
            created_info = info
            target_created = True
            import msvcrt
            descriptor = msvcrt.open_osfhandle(native_handle.value, os.O_WRONLY | os.O_BINARY)
            native_handle = None
            digest = hashlib.sha256()
            total = 0
            target_handle_value = msvcrt.get_osfhandle(descriptor)
            after_info = None
            target_stream = None
            target_deleted = False

            def mark_target_deleted() -> None:
                nonlocal target_deleted
                if target_deleted:
                    return
                _validate_windows_native_handle(
                    kernel32, ctypes.wintypes.HANDLE(target_handle_value),
                    directory=False, single_link=True)
                _windows_mark_handle_deleted(ctypes.wintypes.HANDLE(target_handle_value))
                target_deleted = True

            try:
                target_stream = os.fdopen(descriptor, "wb", closefd=True)
                descriptor = None
                with source.open("rb") as source_stream:
                    for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                        total += len(chunk)
                        target_stream.write(chunk)
                    target_stream.flush()
                    os.fsync(target_stream.fileno())
                    after_info = _validate_windows_native_handle(
                        kernel32, ctypes.wintypes.HANDLE(target_handle_value),
                        directory=False, single_link=True)
            except Exception:
                try:
                    mark_target_deleted()
                except Exception:
                    pass
                raise
            finally:
                if target_stream is not None:
                    try:
                        target_stream.close()
                    except Exception:
                        pass
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
            if ((after_info.volume_serial_number, after_info.file_index_high, after_info.file_index_low)
                    != (info.volume_serial_number, info.file_index_high, info.file_index_low)
                    or total != after_info.file_size_low + (after_info.file_size_high << 32)):
                raise RuntimeError("Windows generation target identity changed during copy")
            source_after = source.stat()
            if ((source_after.st_dev, source_after.st_ino, source_after.st_nlink,
                 source_after.st_size, source_after.st_mtime_ns)
                    != (source_before.st_dev, source_before.st_ino, source_before.st_nlink,
                        source_before.st_size, source_before.st_mtime_ns)):
                raise RuntimeError("Windows generation source artifact changed during copy")
            target_created = False
            return {"sha256": digest.hexdigest(), "bytes": total}
        finally:
            if native_handle is not None:
                kernel32.CloseHandle(native_handle)
    except Exception:
        if target_created and created_info is not None and parent_handles:
            try:
                _windows_delete_relative_exact(kernel32, parent_handles[-1], parts[-1], created_info)
            except Exception:
                pass
        raise
    finally:
        for handle in reversed(parent_handles):
            kernel32.CloseHandle(handle)


def copy_rooted_artifact(source: Path, relative_target: str,
                         writer_lease: Checkpoint_writer_lease) -> dict[str, Any]:
    """Copy a producer artifact into the leased root with exclusive creation."""
    if (not isinstance(relative_target, str) or not relative_target
            or relative_target.startswith("/") or "\\" in relative_target
            or any(part in ("", ".", "..") for part in relative_target.split("/"))):
        raise ValueError("generation target path is unsafe")
    writer_lease.assert_rooted_path(relative_target, allow_missing=True)
    if (not source.is_file() or _is_reparse_point(source) or source.stat().st_nlink != 1):
        raise ValueError("generation source artifact is not a regular single-link file")
    if os.name == "nt":
        return _copy_rooted_artifact_windows(source, relative_target, writer_lease)
    source_before = source.stat()
    target = writer_lease.path.parent / relative_target
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    parent_descriptor = None
    opened_parent_path = None
    opened = None
    try:
        if writer_lease.root_directory_fd is not None:
            parent_descriptor = os.dup(writer_lease.root_directory_fd)
            opened_parent_path = writer_lease.path.parent
            for component in relative_target.split("/")[:-1]:
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_descriptor)
                next_info = os.fstat(next_descriptor)
                opened_parent_path = opened_parent_path / component
                expected_info = opened_parent_path.stat()
                if (not stat.S_ISDIR(next_info.st_mode)
                        or (next_info.st_dev, next_info.st_ino) != (expected_info.st_dev, expected_info.st_ino)
                        or _is_reparse_point(opened_parent_path)):
                    os.close(next_descriptor)
                    raise RuntimeError("generation ancestor identity or reparse state changed")
                os.close(parent_descriptor)
                parent_descriptor = next_descriptor
            descriptor = os.open(relative_target.split("/")[-1], flags, 0o600,
                                 dir_fd=parent_descriptor)
        else:
            descriptor = os.open(str(target), flags, 0o600)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise RuntimeError("generation target is not a regular single-link file")
        with source.open("rb") as source_stream, os.fdopen(descriptor, "wb", closefd=True) as target_stream:
            descriptor = None
            shutil.copyfileobj(source_stream, target_stream)
            target_stream.flush()
            os.fsync(target_stream.fileno())
        source_after = source.stat()
        if ((source_after.st_dev, source_after.st_ino, source_after.st_nlink,
             source_after.st_size, source_after.st_mtime_ns)
                != (source_before.st_dev, source_before.st_ino, source_before.st_nlink,
                    source_before.st_size, source_before.st_mtime_ns)):
            raise RuntimeError("generation source artifact changed during copy")
        writer_lease.assert_rooted_path(relative_target)
        if parent_descriptor is not None:
            final = os.stat(relative_target.split("/")[-1], dir_fd=parent_descriptor,
                            follow_symlinks=False)
        else:
            final = target.stat()
        if (final.st_dev, final.st_ino) != (opened.st_dev, opened.st_ino) or final.st_nlink != 1:
            raise RuntimeError("generation target identity changed during copy")
        identity = _file_identity(target)
        return identity
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        try:
            if opened is not None and parent_descriptor is not None:
                current = os.stat(relative_target.split("/")[-1], dir_fd=parent_descriptor,
                                  follow_symlinks=False)
                if (stat.S_ISREG(current.st_mode) and current.st_nlink == 1
                        and (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino)):
                    os.unlink(relative_target.split("/")[-1], dir_fd=parent_descriptor)
            elif opened is not None:
                current = target.lstat()
                if (stat.S_ISREG(current.st_mode) and current.st_nlink == 1
                        and (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino)):
                    target.unlink()
        except OSError:
            pass
        if parent_descriptor is not None:
            try:
                os.close(parent_descriptor)
            except OSError:
                pass
            parent_descriptor = None
        raise
    finally:
        if parent_descriptor is not None:
            try:
                os.close(parent_descriptor)
            except OSError:
                pass


def _write_json(path: Path, value: object) -> None:
    encoded = canonical_json_bytes(value)
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=True,
                       allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def json_identity(value: object) -> dict[str, object]:
    encoded = canonical_json_bytes(value)
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def checkpoint_commit_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint_commit.json"


def _validate_checkpoint_commit_descriptor(value: dict[str, Any], *,
                                           allow_missing_completion_receipt: bool = False) -> None:
    """Validate the complete V2 descriptor before it becomes resume authority."""
    required = {"generation_id", "execution_id", "checkpoint", "manifest", "run_id",
                "model_id", "model_profile", "model_build_spec_identity",
                "source_identity", "config_sha", "epoch", "global_step",
                "training_complete", "model_state_sha", "completion_receipt",
                "metric_record", "metrics_journal", "schema"}
    if not isinstance(value, dict) or value.get("schema") != CHECKPOINT_COMMIT_SCHEMA:
        raise ValueError("checkpoint commit descriptor schema is invalid")
    if set(value) != required:
        raise ValueError("checkpoint commit descriptor fields are invalid")
    for name in ("generation_id", "execution_id", "run_id", "model_id"):
        if name == "run_id" and value[name] is None:
            continue
        if not isinstance(value[name], str) or len(value[name]) > 256:
            raise ValueError("checkpoint commit descriptor identity fields are invalid")
    if value["model_profile"] is not None and (
            not isinstance(value["model_profile"], str) or not value["model_profile"]):
        raise ValueError("checkpoint commit descriptor model profile is invalid")
    model_spec = value["model_build_spec_identity"]
    if (not isinstance(model_spec, dict) or set(model_spec) != {"schema", "sha256", "bytes"}
            or model_spec.get("schema") != "V10ModelBuildSpecIdentityV1"
            or not isinstance(model_spec.get("sha256"), str) or len(model_spec["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in model_spec["sha256"])
            or type(model_spec.get("bytes")) is not int or model_spec["bytes"] <= 0):
        raise ValueError("checkpoint commit descriptor model build specification identity is invalid")
    if not isinstance(value["source_identity"], dict) or not value["source_identity"]:
        raise ValueError("checkpoint commit descriptor source identity is invalid")
    if (not isinstance(value["config_sha"], str) or len(value["config_sha"]) != 64
            or any(char not in "0123456789abcdef" for char in value["config_sha"])
            or type(value["epoch"]) is not int or value["epoch"] < 0
            or type(value["global_step"]) is not int or value["global_step"] < 0
            or type(value["training_complete"]) is not bool
            or not isinstance(value["model_state_sha"], str) or len(value["model_state_sha"]) != 64
            or any(char not in "0123456789abcdef" for char in value["model_state_sha"])):
        raise ValueError("checkpoint commit descriptor progress or state identity is invalid")

    def validate_identity(identity: object, name: str, *, allow_none: bool = False) -> None:
        if allow_none and identity is None:
            return
        if not isinstance(identity, dict):
            raise ValueError(f"checkpoint commit descriptor {name} identity is invalid")
        expected = {"path", "sha256", "bytes"}
        if name == "completion receipt" and set(identity) == {"path"}:
            expected = {"path"}
        if set(identity) != expected:
            raise ValueError(f"checkpoint commit descriptor {name} fields are invalid")
        path = identity["path"]
        if (not isinstance(path, str) or not path or path.startswith("/") or "\\" in path
                or any(part in ("", ".", "..") for part in path.split("/"))):
            raise ValueError(f"checkpoint commit descriptor {name} path is invalid")
        if expected == {"path"}:
            return
        digest = identity["sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError(f"checkpoint commit descriptor {name} digest is invalid")
        if type(identity["bytes"]) is not int or identity["bytes"] <= 0:
            raise ValueError(f"checkpoint commit descriptor {name} size is invalid")

    validate_identity(value["checkpoint"], "checkpoint")
    validate_identity(value["manifest"], "manifest")
    validate_identity(value["metrics_journal"], "metrics journal")
    validate_identity(value["completion_receipt"], "completion receipt", allow_none=True)
    generation_prefix = f".checkpoint_generations/{value['generation_id']}/"
    for identity in (value["checkpoint"], value["manifest"], value["metrics_journal"],
                     value["metric_record"], value["completion_receipt"]):
        if isinstance(identity, dict) and not identity["path"].startswith(generation_prefix):
            raise ValueError("checkpoint commit descriptor artifact escaped its generation")
    if (value["training_complete"] and value["completion_receipt"] is None
            and not allow_missing_completion_receipt):
        raise ValueError("complete checkpoint commit lacks a completion receipt")
    if value["metric_record"] is not None:
        validate_identity(value["metric_record"], "metric record")


def read_checkpoint_commit(output_dir: Path) -> dict[str, Any] | None:
    path = checkpoint_commit_path(output_dir)
    if not path.exists():
        return None
    if (not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1
            or path.stat().st_size > MAX_DESCRIPTOR_BYTES):
        raise ValueError("checkpoint commit descriptor is unsafe")
    try:
        value = json.loads(_safe_read_bytes(path).decode("utf-8"), object_pairs_hook=_reject_duplicate_keys,
                           parse_constant=_reject_nonfinite_json)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("checkpoint commit descriptor is malformed") from exc
    if isinstance(value, dict) and value.get("schema") == "V10CheckpointCommitV1":
        raise ValueError("checkpoint commit descriptor V1 is retired; reissue a V2 descriptor")
    if not isinstance(value, dict) or value.get("schema") != CHECKPOINT_COMMIT_SCHEMA:
        raise ValueError("checkpoint commit descriptor schema is invalid")
    _validate_checkpoint_commit_descriptor(value)
    return value


def read_predecessor_journal(output_dir: Path) -> tuple[dict[str, Any] | None, bytes]:
    """Read predecessor descriptor and journal with a stable double descriptor check."""
    descriptor = read_checkpoint_commit(output_dir)
    if descriptor is None:
        return None, b""
    binding = descriptor.get("metrics_journal")
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        raise ValueError("previous authoritative descriptor lacks metrics journal")
    relative = binding["path"]
    if relative.startswith("/") or "\\" in relative or ".." in relative.split("/"):
        raise ValueError("previous metrics journal path is unsafe")
    path = (output_dir / relative).resolve()
    generation_root = (output_dir / ".checkpoint_generations" / descriptor["generation_id"]).resolve()
    if path.parent != generation_root:
        raise ValueError("previous metrics journal escaped its generation")
    payload = _safe_read_bytes(path, maximum_bytes=MAX_METRICS_JOURNAL_BYTES)
    identity = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    if identity != {"sha256": binding.get("sha256"), "bytes": binding.get("bytes")}:
        raise ValueError("previous metrics journal identity mismatch")
    manifest_path = output_dir / descriptor["manifest"]["path"]
    manifest = json.loads(_safe_read_bytes(manifest_path), object_pairs_hook=_reject_duplicate_keys,
                          parse_constant=_reject_nonfinite_json)
    if manifest.get("metrics_journal") != identity:
        raise ValueError("previous manifest journal binding mismatch")
    after = read_checkpoint_commit(output_dir)
    if after != descriptor:
        raise ValueError("previous descriptor changed during journal read")
    return descriptor, payload


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def commit_checkpoint(output_dir: Path, checkpoint: dict[str, Any], *,
                      execution_id: str, epoch: int, global_step: int,
                      authoritative: bool = True,
                      writer_lease: Checkpoint_writer_lease | None = None,
                      metric_record: dict[str, Any] | None = None) -> dict[str, Any]:
    """Commit a complete checkpoint generation before changing resume authority."""
    if authoritative and writer_lease is None:
        raise RuntimeError("authoritative checkpoint commit requires an active writer lease")
    if writer_lease is None:
        raise RuntimeError("checkpoint generation staging requires an active writer lease")
    if metric_record is None:
        raise ValueError("checkpoint generation requires a bound metric record")
    writer_lease.assert_active()
    writer_lease.assert_rooted_path(".checkpoint_generations", allow_missing=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir / ".checkpoint_generations"
    root.mkdir(exist_ok=True)
    writer_lease.assert_active()
    writer_lease.assert_rooted_path(".checkpoint_generations", allow_directory=True)
    generation_id = f"{execution_id}_{global_step}_{uuid.uuid4().hex}"
    staging = root / (generation_id + ".staging")
    generation = root / generation_id
    writer_lease.assert_rooted_path(
        ".checkpoint_generations/" + generation_id + ".staging",
        allow_missing=True, allow_directory=True)
    staging.mkdir()
    try:
        writer_lease.assert_active()
        writer_lease.assert_rooted_path(
            ".checkpoint_generations/" + generation_id + ".staging",
            allow_directory=True)
        checkpoint_path = staging / "checkpoint.pt"
        writer_lease.assert_rooted_path(
            ".checkpoint_generations/" + generation_id + ".staging/checkpoint.pt",
            allow_missing=True)
        with checkpoint_path.open("wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        checkpoint_identity = _file_identity(checkpoint_path)
        metadata = checkpoint["metadata"]
        if (type(metadata.get("training_complete")) is not bool
                or type(epoch) is not int or type(global_step) is not int):
            raise ValueError("checkpoint completion and progress types are invalid")
        if (metadata.get("execution_id") != execution_id
                or metadata.get("epoch") != epoch
                or metadata.get("global_step") != global_step):
            raise ValueError("checkpoint metadata does not match commit identity")
        manifest = {
            "schema": CHECKPOINT_MANIFEST_SCHEMA,
            "generation_id": generation_id, "execution_id": execution_id,
            "run_id": metadata.get("run_id"), "model_id": metadata["model_id"],
            "model_profile": metadata.get("model_profile"),
            "model_build_spec_identity": metadata.get("model_build_spec_identity"),
            "source_identity": metadata.get("source_identity", {key: metadata.get(key) for key in (
                "branch", "source_commit", "source_tree_sha", "dirty_patch_sha",
                "config_sha", "dataset_hashes", "split_hash")}),
            "checkpoint_sha256": checkpoint_identity["sha256"],
            "checkpoint_bytes": checkpoint_identity["bytes"],
            "config_sha": metadata["config_sha"],
            "epoch": epoch, "global_step": global_step,
            "training_complete": metadata["training_complete"],
            "model_state_sha": metadata["model_state_sha"],
        }
        if metric_record is not None:
            metric = dict(metric_record)
            metric.update({"execution_id": execution_id, "generation_id": generation_id,
                           "checkpoint_sha256": checkpoint_identity["sha256"],
                           "global_step": global_step, "epoch": epoch,
                           "training_complete": metadata["training_complete"]})
            metric["generation_manifest"] = {
                "execution_id": execution_id, "generation_id": generation_id,
                "checkpoint_sha256": checkpoint_identity["sha256"],
                "checkpoint_bytes": checkpoint_identity["bytes"],
                "model_id": metadata["model_id"], "config_sha": metadata["config_sha"],
                "model_profile": metadata.get("model_profile"),
                "model_build_spec_identity": metadata.get("model_build_spec_identity"),
                "source_identity": manifest["source_identity"], "epoch": epoch,
                "global_step": global_step, "training_complete": metadata["training_complete"],
                "model_state_sha": metadata["model_state_sha"],
            }
            _write_json(staging / "metrics_record.json", metric)
            metric_identity = _file_identity(staging / "metrics_record.json")
        else:
            metric_identity = None
        manifest["metric_record"] = metric_identity
        previous_journal = b""
        previous_descriptor, previous_journal = read_predecessor_journal(output_dir)
        metric_line = canonical_json_bytes(metric)
        journal_bytes = previous_journal.rstrip(b"\n") + (b"\n" if previous_journal.rstrip(b"\n") else b"") + metric_line
        if len(journal_bytes) > MAX_METRICS_JOURNAL_BYTES:
            raise ValueError("metrics journal exceeds the bounded journal limit")
        journal_path = staging / "metrics_journal.jsonl"
        writer_lease.assert_rooted_path(
            ".checkpoint_generations/" + generation_id + ".staging/metrics_journal.jsonl",
            allow_missing=True)
        with journal_path.open("wb") as journal_stream:
            journal_stream.write(journal_bytes); journal_stream.flush(); os.fsync(journal_stream.fileno())
        journal_identity = _file_identity(journal_path)
        manifest["metrics_journal"] = journal_identity
        _write_json(staging / "checkpoint_manifest.json", manifest)
        _fsync_directory(staging)
        manifest_identity = {"path": f".checkpoint_generations/{generation_id}/checkpoint_manifest.json",
                             **_file_identity(staging / "checkpoint_manifest.json")}
        descriptor = {
            "schema": CHECKPOINT_COMMIT_SCHEMA,
            "generation_id": generation_id, "execution_id": execution_id,
            "checkpoint": {"path": f".checkpoint_generations/{generation_id}/checkpoint.pt",
                            **checkpoint_identity},
            "manifest": manifest_identity,
            "run_id": metadata.get("run_id"), "model_id": metadata["model_id"],
            "model_profile": metadata.get("model_profile"),
            "model_build_spec_identity": metadata.get("model_build_spec_identity"),
            "source_identity": manifest["source_identity"],
            "config_sha": metadata["config_sha"], "epoch": epoch,
            "global_step": global_step, "training_complete": metadata["training_complete"],
            "model_state_sha": metadata["model_state_sha"], "completion_receipt": None,
            "metric_record": ({"path": f".checkpoint_generations/{generation_id}/metrics_record.json",
                               **metric_identity} if metric_identity is not None else None),
            "metrics_journal": {"path": f".checkpoint_generations/{generation_id}/metrics_journal.jsonl",
                                 **journal_identity},
        }
        # Validate the exact descriptor before staging becomes an immutable
        # generation. Invalid metadata therefore cannot leave an orphan root.
        _validate_checkpoint_commit_descriptor(
            descriptor, allow_missing_completion_receipt=not authoritative)
        writer_lease.assert_active()
        writer_lease.assert_rooted_path(
            ".checkpoint_generations/" + generation_id + ".staging/checkpoint_manifest.json")
        os.replace(staging, generation)
        writer_lease.assert_rooted_path(
            ".checkpoint_generations/" + generation_id + "/checkpoint_manifest.json")
        _fsync_directory(root)
        if authoritative:
            writer_lease.assert_rooted_path("checkpoint_commit.json", allow_missing=True)
            publish_checkpoint_commit(output_dir, descriptor, writer_lease=writer_lease)
        else:
            descriptor_tmp = output_dir / ("checkpoint_commit_" + generation_id + ".staged.json")
            _write_json(descriptor_tmp, descriptor)
        # Keep the historical pair as a compatibility projection.  The descriptor
        # remains the only authority for new writers and resume readers.
        if authoritative:
            for source, target in ((generation / "checkpoint.pt", output_dir / "checkpoint_latest.pt"),
                                   (generation / "checkpoint_manifest.json", output_dir / "checkpoint_manifest.json"),
                                   (generation / "metrics_journal.jsonl", output_dir / "metrics.jsonl")):
                try:
                    writer_lease.assert_rooted_path(target.name, allow_missing=True)
                    temporary = output_dir / (target.name + "_compat_" + uuid.uuid4().hex + ".tmp")
                    with source.open("rb") as src, temporary.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                        dst.flush(); os.fsync(dst.fileno())
                    os.replace(temporary, target)
                except OSError:
                    if 'temporary' in locals() and temporary.exists():
                        temporary.unlink(missing_ok=True)
            _cleanup_checkpoint_generations(root, generation_id)
        return descriptor
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def publish_checkpoint_commit(output_dir: Path, descriptor: dict[str, Any], *,
                             writer_lease: Checkpoint_writer_lease | None = None) -> None:
    """The sole mandatory authority transition for a staged generation."""
    if writer_lease is None:
        raise RuntimeError("descriptor replacement requires an active writer lease")
    _validate_checkpoint_commit_descriptor(descriptor)
    writer_lease.assert_active()
    writer_lease.assert_rooted_path("checkpoint_commit.json", allow_missing=True)
    generation_id = descriptor.get("generation_id")
    staged = output_dir / ("checkpoint_commit_" + str(generation_id) + ".staged.json")
    if staged.exists():
        payload = json.loads(staged.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
        if payload != descriptor:
            raise ValueError("staged checkpoint descriptor changed")
    temporary = output_dir / ("checkpoint_commit_" + uuid.uuid4().hex + ".tmp")
    _write_json(temporary, descriptor)
    writer_lease.assert_active()
    writer_lease.assert_rooted_path("checkpoint_commit.json", allow_missing=True)
    os.replace(temporary, checkpoint_commit_path(output_dir))
    try:
        staged.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        _fsync_directory(output_dir)
    except OSError:
        pass


def _cleanup_checkpoint_generations(root: Path, current_generation_id: str) -> None:
    try:
        entries = list(root.iterdir())
        for path in entries:
            if path.is_dir() and path.name.endswith(".staging"):
                shutil.rmtree(path, ignore_errors=True)
        candidates = sorted((path for path in root.iterdir()
                             if path.is_dir() and not path.name.endswith(".staging")),
                           key=lambda path: path.stat().st_mtime_ns, reverse=True)
        for path in candidates[MAX_CHECKPOINT_GENERATIONS:]:
            if path.name != current_generation_id:
                shutil.rmtree(path)
    except OSError:
        return


def cleanup_checkpoint_commit_artifacts(output_dir: Path, current_generation_id: str) -> None:
    """Best-effort bounded cleanup after authority has advanced."""
    _cleanup_checkpoint_generations(output_dir / ".checkpoint_generations", current_generation_id)
    try:
        for candidate in output_dir.glob("checkpoint_commit_*.staged.json"):
            candidate.unlink(missing_ok=True)
        for candidate in output_dir.glob("checkpoint_commit_*.tmp"):
            candidate.unlink(missing_ok=True)
    except OSError:
        return


def resolve_checkpoint_resume(output_dir: Path, requested_path: Path) -> Path:
    descriptor = read_checkpoint_commit(output_dir)
    if descriptor is None:
        return requested_path
    relative = descriptor.get("checkpoint", {}).get("path")
    if (not isinstance(relative, str) or relative.startswith("/") or "\\" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError("checkpoint commit descriptor path is invalid")
    resolved = (output_dir / relative).resolve()
    if resolved.parent.parent != (output_dir / ".checkpoint_generations").resolve():
        raise ValueError("checkpoint commit descriptor escaped output root")
    identity = descriptor.get("checkpoint")
    manifest_identity = descriptor.get("manifest")
    if not isinstance(identity, dict) or not isinstance(manifest_identity, dict):
        raise ValueError("checkpoint commit descriptor identities are invalid")
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_nlink != 1:
        raise ValueError("committed checkpoint generation is not a regular file")
    actual = _file_identity(resolved)
    if actual != {"sha256": identity.get("sha256"), "bytes": identity.get("bytes")}:
        raise ValueError("checkpoint commit generation identity mismatch")
    manifest_relative = manifest_identity.get("path")
    if not isinstance(manifest_relative, str) or manifest_relative.startswith("/") or "\\" in manifest_relative:
        raise ValueError("checkpoint manifest descriptor path is invalid")
    manifest_path = (output_dir / manifest_relative).resolve()
    if manifest_path.parent != resolved.parent or not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("checkpoint manifest path is invalid")
    if _file_identity(manifest_path) != {"sha256": manifest_identity.get("sha256"),
                                         "bytes": manifest_identity.get("bytes")}:
        raise ValueError("checkpoint manifest identity mismatch")
    try:
        manifest = json.loads(_safe_read_bytes(manifest_path).decode("utf-8"),
                              object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint generation manifest is malformed") from exc
    bindings = (("generation_id", descriptor.get("generation_id")),
                ("execution_id", descriptor.get("execution_id")),
                ("checkpoint_sha256", identity.get("sha256")),
                ("checkpoint_bytes", identity.get("bytes")),
                ("model_id", descriptor.get("model_id")),
                ("model_profile", descriptor.get("model_profile")),
                ("model_build_spec_identity", descriptor.get("model_build_spec_identity")),
                ("run_id", descriptor.get("run_id")),
                ("source_identity", descriptor.get("source_identity")),
                ("config_sha", descriptor.get("config_sha")),
                ("epoch", descriptor.get("epoch")),
                ("global_step", descriptor.get("global_step")),
                ("training_complete", descriptor.get("training_complete")),
                ("model_state_sha", descriptor.get("model_state_sha")))
    expected_manifest_fields = {"schema", "generation_id", "execution_id", "run_id", "model_id",
                                "model_profile", "model_build_spec_identity",
                                "source_identity", "checkpoint_sha256", "checkpoint_bytes", "config_sha",
                                "epoch", "global_step", "training_complete", "model_state_sha"}
    expected_manifest_fields.update({"metric_record", "metrics_journal"})
    if (not isinstance(manifest, dict) or set(manifest) != expected_manifest_fields
            or manifest.get("schema") != CHECKPOINT_MANIFEST_SCHEMA):
        raise ValueError("checkpoint generation manifest schema is invalid")
    if any(manifest.get(key) != value for key, value in bindings):
        raise ValueError("checkpoint generation manifest binding mismatch")
    for field, filename in (("metric_record", "metrics_record.json"),
                            ("metrics_journal", "metrics_journal.jsonl")):
        descriptor_binding = descriptor.get(field)
        manifest_binding = manifest.get(field)
        if not isinstance(descriptor_binding, dict) or not isinstance(manifest_binding, dict):
            raise ValueError(f"checkpoint {field} binding is missing")
        relative = descriptor_binding.get("path")
        if (not isinstance(relative, str) or relative.startswith("/") or "\\" in relative
                or any(part in ("", ".", "..") for part in relative.split("/"))
                or Path(relative).name != filename):
            raise ValueError(f"checkpoint {field} path is unsafe")
        candidate = (output_dir / relative).resolve()
        if candidate.parent != resolved.parent or not candidate.is_file() or candidate.is_symlink():
            raise ValueError(f"checkpoint {field} is unavailable")
        if candidate.stat().st_nlink != 1 or _file_identity(candidate) != {
                "sha256": descriptor_binding.get("sha256"), "bytes": descriptor_binding.get("bytes")}:
            raise ValueError(f"checkpoint {field} identity mismatch")
        if manifest_binding != {"sha256": descriptor_binding.get("sha256"),
                                "bytes": descriptor_binding.get("bytes")}:
            raise ValueError(f"checkpoint {field} manifest binding mismatch")
    if descriptor.get("training_complete") is True:
        receipt_info = descriptor.get("completion_receipt")
        if not isinstance(receipt_info, dict) or not isinstance(receipt_info.get("path"), str):
            raise ValueError("complete checkpoint lacks a generation completion receipt")
        receipt_relative = receipt_info["path"]
        if receipt_relative.startswith("/") or "\\" in receipt_relative or ".." in receipt_relative.split("/"):
            raise ValueError("completion receipt path is unsafe")
        receipt_path = (output_dir / receipt_relative).resolve()
        generation_root = (output_dir / ".checkpoint_generations" / descriptor["generation_id"]).resolve()
        if receipt_path.parent != generation_root:
            raise ValueError("completion receipt escaped its generation")
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise ValueError("complete checkpoint lacks a verified completion receipt")
        if receipt_path.stat().st_nlink != 1:
            raise ValueError("completion receipt must be a single-link file")
        if "sha256" in receipt_info or "bytes" in receipt_info:
            if _file_identity(receipt_path) != {"sha256": receipt_info.get("sha256"),
                                                "bytes": receipt_info.get("bytes")}:
                raise ValueError("completion receipt identity mismatch")
        try:
            receipt = json.loads(_safe_read_bytes(receipt_path).decode("utf-8"),
                                 object_pairs_hook=_reject_duplicate_keys)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("complete checkpoint receipt is malformed") from exc
        if (receipt.get("schema") != "V10TrainingCompletionV1"
                or receipt.get("status") != "COMPLETED"
                or receipt.get("execution_id") != descriptor.get("execution_id")
                or receipt.get("global_step") != descriptor.get("global_step")
                or receipt.get("model_state_sha") != descriptor.get("model_state_sha")
                or receipt.get("checkpoint_commit") != descriptor):
            raise ValueError("complete checkpoint completion binding mismatch")
        source_identity = receipt.get("source_identity")
        descriptor_source_identity = descriptor.get("source_identity")
        if source_identity != descriptor_source_identity:
            raise ValueError("complete checkpoint source identity mismatch")
        if isinstance(descriptor_source_identity, dict) and descriptor_source_identity.get("input_delivery") == "stream_only":
            recipe_artifact = receipt.get("artifacts", {}).get("stream_recipe.json")
            if not isinstance(recipe_artifact, dict):
                raise ValueError("complete stream checkpoint lacks stream recipe artifact")
            expected_recipe_sha = descriptor_source_identity.get("stream_recipe_file_sha")
            if recipe_artifact.get("sha256") != expected_recipe_sha:
                raise ValueError("complete stream checkpoint recipe binding mismatch")
            recipe_path = output_dir / "stream_recipe.json"
            if (not recipe_path.is_file() or _is_reparse_point(recipe_path)
                    or _file_identity(recipe_path) != {
                        "sha256": recipe_artifact.get("sha256"),
                        "bytes": recipe_artifact.get("bytes")}):
                raise ValueError("complete stream checkpoint recipe identity mismatch")
    return resolved


def cpu_checkpoint_snapshot(value: Any) -> Any:
    """Do not pass live CUDA storages or borrowed model tensors to ZIP writing."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        result = copy.copy(value)
        for key, item in value.items():
            result[key] = cpu_checkpoint_snapshot(item)
        if hasattr(value, "_metadata"):
            result._metadata = copy.deepcopy(value._metadata)
        return result
    if isinstance(value, tuple):
        return tuple(cpu_checkpoint_snapshot(item) for item in value)
    if isinstance(value, list):
        return [cpu_checkpoint_snapshot(item) for item in value]
    return copy.deepcopy(value)
