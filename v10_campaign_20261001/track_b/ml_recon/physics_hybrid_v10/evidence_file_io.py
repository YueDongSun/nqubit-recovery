"""Descriptor-bound reads for source-bound local evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat


def _open_windows_no_reparse(path: Path, name: str) -> int:
    """Open the final Windows object without traversing its reparse record."""
    import ctypes
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    handle = create_file(
        str(path), 0x80000000, 0x00000001 | 0x00000002 | 0x00000004,
        None, 3, 0x00200000, None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise ValueError(f"{name} cannot be opened without following aliases")
    try:
        return msvcrt.open_osfhandle(handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except OSError:
        kernel32.CloseHandle(handle)
        raise


def _open_posix_anchored(path: Path, name: str) -> int:
    """Walk parent directories by descriptor so a parent swap cannot redirect open."""
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.anchor, directory_flags)
    try:
        parts = [part for part in path.parts if part != path.anchor]
        for part in parts[:-1]:
            next_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return os.open(
            parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=descriptor
        )
    except OSError as exc:
        raise ValueError(f"{name} cannot be opened without following aliases") from exc
    finally:
        os.close(descriptor)


def _open_evidence_descriptor(path: Path, name: str) -> int:
    if os.name == "nt":
        return _open_windows_no_reparse(path, name)
    return _open_posix_anchored(path, name)


@dataclass(frozen=True)
class Evidence_file_identity:
    """Stable file metadata captured from the opened descriptor."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "Evidence_file_identity":
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
        )

    def matches_stat(self, value: os.stat_result) -> bool:
        return self == self.from_stat(value)


def _is_reparse_or_symlink(value: os.stat_result) -> bool:
    file_attributes = getattr(value, "st_file_attributes", 0)
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(value.st_mode) or bool(file_attributes & reparse_point)


def _lexical_components(path: Path, name: str) -> tuple[Path, ...]:
    """Inspect each spelling component without resolving a possible alias."""
    raw_path = os.fspath(path)
    if not raw_path:
        raise ValueError(f"{name} requires a nonempty file path")
    supplied = Path(raw_path)
    if ".." in supplied.parts:
        raise ValueError(f"{name} path may not contain parent traversal")
    if os.name == "nt":
        for index, part in enumerate(supplied.parts):
            if index == 0 and part.endswith("\\"):
                continue
            if ":" in part or part.rstrip(". ") != part:
                raise ValueError(f"{name} path has a noncanonical Windows spelling")
    absolute = Path(os.path.abspath(raw_path))
    current = Path(absolute.anchor)
    components: list[Path] = []
    for part in absolute.parts:
        if part == absolute.anchor:
            continue
        current = current / part
        try:
            candidate_stat = os.lstat(current)
        except OSError as exc:
            raise ValueError(f"{name} path component is unavailable: {current}") from exc
        if _is_reparse_or_symlink(candidate_stat):
            raise ValueError(f"{name} path contains a symlink or reparse point")
        components.append(current)
    if not components:
        raise ValueError(f"{name} must name a regular file")
    return tuple(components)


def validate_evidence_path(
    path: str | Path, name: str, *, allow_missing: bool = False
) -> Path:
    """Validate every existing lexical ancestor, retaining a missing final suffix."""
    raw_path = os.fspath(path)
    supplied = Path(raw_path)
    if not raw_path or ".." in supplied.parts:
        raise ValueError(f"{name} path may not contain parent traversal")
    absolute = Path(os.path.abspath(raw_path))
    current = Path(absolute.anchor)
    for part in absolute.parts:
        if part == absolute.anchor:
            continue
        current = current / part
        try:
            component_stat = os.lstat(current)
        except FileNotFoundError:
            if allow_missing:
                return absolute
            raise ValueError(f"{name} path component is unavailable: {current}") from None
        except OSError as exc:
            raise ValueError(f"{name} path component is unavailable: {current}") from exc
        if _is_reparse_or_symlink(component_stat):
            raise ValueError(f"{name} path contains a symlink or reparse point")
    return absolute


def _validate_open_file(
    descriptor: int,
    lexical_path: Path,
    name: str,
    *,
    expected_identity: Evidence_file_identity | None,
) -> Evidence_file_identity:
    descriptor_stat = os.fstat(descriptor)
    if _is_reparse_or_symlink(descriptor_stat) or not stat.S_ISREG(descriptor_stat.st_mode):
        raise ValueError(f"{name} must be a regular file")
    if descriptor_stat.st_nlink != 1:
        raise ValueError(f"{name} must not be hardlinked")
    identity = Evidence_file_identity.from_stat(descriptor_stat)
    if expected_identity is not None and identity != expected_identity:
        raise ValueError(f"{name} file identity changed")
    try:
        lexical_stat = os.lstat(lexical_path)
    except OSError as exc:
        raise ValueError(f"{name} path changed while opening") from exc
    if _is_reparse_or_symlink(lexical_stat) or not os.path.samestat(
        descriptor_stat, lexical_stat
    ):
        raise ValueError(f"{name} path identity changed while opening")
    return identity


def validate_evidence_directory(path: str | Path, name: str) -> Path:
    """Return a lexical directory path only after rejecting alias components."""
    components = _lexical_components(Path(path), name)
    directory_path = components[-1]
    try:
        directory_stat = os.lstat(directory_path)
    except OSError as exc:
        raise ValueError(f"{name} path component is unavailable: {directory_path}") from exc
    if _is_reparse_or_symlink(directory_stat) or not stat.S_ISDIR(directory_stat.st_mode):
        raise ValueError(f"{name} must be a non-aliased directory")
    return directory_path


def read_evidence_file(
    path: str | Path,
    name: str,
    maximum_bytes: int,
    *,
    expected_identity: Evidence_file_identity | None = None,
) -> tuple[Path, bytes, Evidence_file_identity]:
    """Read one bounded evidence file from a stable, identity-checked descriptor.

    This defends local aliases and replacement races at the file boundary. It does
    not claim filesystem-wide atomicity against concurrent writers.
    """
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise ValueError("evidence size limit must be a positive integer")
    components = _lexical_components(Path(path), name)
    lexical_path = components[-1]
    descriptor = _open_evidence_descriptor(lexical_path, name)
    try:
        before_identity = _validate_open_file(
            descriptor, lexical_path, name, expected_identity=expected_identity
        )
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw_bytes = b"".join(chunks)
        if len(raw_bytes) > maximum_bytes:
            raise ValueError(f"{name} exceeds the admission size limit")
        after_identity = _validate_open_file(
            descriptor, lexical_path, name, expected_identity=before_identity
        )
        _lexical_components(lexical_path, name)
        return lexical_path, raw_bytes, after_identity
    finally:
        os.close(descriptor)


def sha256_evidence_file(
    path: str | Path,
    name: str,
    *,
    expected_identity: Evidence_file_identity | None = None,
    chunk_bytes: int = 1024 * 1024,
    maximum_bytes: int | None = None,
) -> tuple[Path, str, Evidence_file_identity]:
    """Stream a hash from one descriptor while preserving the same identity checks."""
    if type(chunk_bytes) is not int or chunk_bytes < 1:
        raise ValueError("evidence hash chunk size must be a positive integer")
    components = _lexical_components(Path(path), name)
    lexical_path = components[-1]
    descriptor = _open_evidence_descriptor(lexical_path, name)
    try:
        before_identity = _validate_open_file(
            descriptor, lexical_path, name, expected_identity=expected_identity
        )
        if maximum_bytes is not None and (
            type(maximum_bytes) is not int or maximum_bytes < 0
        ):
            raise ValueError("evidence hash size limit must be a nonnegative integer")
        if maximum_bytes is not None and before_identity.size > maximum_bytes:
            raise ValueError(f"{name} exceeds the admission size limit")
        digest = hashlib.sha256()
        remaining = before_identity.size + 1
        while remaining:
            chunk = os.read(descriptor, min(chunk_bytes, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
        if remaining == 0:
            raise ValueError(f"{name} grew while hashing")
        after_identity = _validate_open_file(
            descriptor, lexical_path, name, expected_identity=before_identity
        )
        _lexical_components(lexical_path, name)
        return lexical_path, digest.hexdigest(), after_identity
    finally:
        os.close(descriptor)
