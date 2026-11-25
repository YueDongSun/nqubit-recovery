"""Atomic, no-overwrite publication for external Track B artifacts."""

from __future__ import annotations

import hashlib
import os
import stat
import ctypes
from dataclasses import dataclass, field
from pathlib import Path
import tempfile

from .contracts import require_sha256
from .artifact_directory_lease import Directory_lease
from .artifact_file_lease import file_identity_from_descriptor


PUBLICATION_COMMITTED = "COMMITTED"
PUBLICATION_COMMITTED_CLEANUP_INCOMPLETE = "COMMITTED_CLEANUP_INCOMPLETE"


@dataclass(frozen=True)
class TrackBArtifactPublicationV1:
    """Explicit state for a hard-link publication and its staging cleanup."""

    target_path: str
    artifact_sha256: str
    status: str
    target_published: bool
    staging_cleanup_complete: bool
    staging_path: str
    cleanup_error: str | None = None
    # In-process ownership handoff only; durable bindings live in the marker.
    file_identity: tuple[int, str] | None = field(default=None, repr=False, compare=False)
    schema: str = field(default="TrackBArtifactPublicationV1", init=False)

    def __post_init__(self) -> None:
        if type(self.target_path) is not str or not self.target_path:
            raise ValueError("artifact publication target path is invalid")
        require_sha256(self.artifact_sha256, "published_artifact_sha")
        if self.status not in {PUBLICATION_COMMITTED, PUBLICATION_COMMITTED_CLEANUP_INCOMPLETE}:
            raise ValueError("artifact publication status is not registered")
        if type(self.target_published) is not bool or not self.target_published:
            raise ValueError("committed artifact publication must publish its target")
        if type(self.staging_cleanup_complete) is not bool:
            raise TypeError("artifact staging cleanup state must be an exact boolean")
        if self.status == PUBLICATION_COMMITTED and not self.staging_cleanup_complete:
            raise ValueError("complete publication cannot report incomplete staging cleanup")
        if self.status == PUBLICATION_COMMITTED_CLEANUP_INCOMPLETE and self.staging_cleanup_complete:
            raise ValueError("incomplete publication must report incomplete staging cleanup")
        if self.status == PUBLICATION_COMMITTED and self.cleanup_error is not None:
            raise ValueError("complete publication cannot carry a cleanup error")
        if self.status == PUBLICATION_COMMITTED_CLEANUP_INCOMPLETE and (
                type(self.cleanup_error) is not str or not self.cleanup_error
        ):
            raise ValueError("incomplete publication requires a cleanup error")
        if self.file_identity is not None and (
                type(self.file_identity) is not tuple or len(self.file_identity) != 2
                or type(self.file_identity[0]) is not int or self.file_identity[0] <= 0
                or type(self.file_identity[1]) is not str
                or len(self.file_identity[1]) != 32
                or any(character not in "0123456789abcdef"
                       for character in self.file_identity[1])):
            raise ValueError("published file identity is invalid")

    def to_dict(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema": self.schema,
            "target_path": self.target_path,
            "artifact_sha256": self.artifact_sha256,
            "status": self.status,
            "target_published": self.target_published,
            "staging_cleanup_complete": self.staging_cleanup_complete,
            "staging_path": self.staging_path,
            "cleanup_error": self.cleanup_error,
        }


def _parent_identity(path: Path) -> tuple[int, int, int]:
    """Return the admitted directory identity without following aliases."""
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("artifact parent directory is not a stable directory")
    # Windows exposes a stable volume/file-index pair for ordinary local
    # directories.  Refuse filesystems which cannot provide that identity;
    # path-only publication cannot be made fail-closed there.
    if not metadata.st_dev or not metadata.st_ino:
        raise ValueError("artifact parent directory identity is unavailable")
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode)


class _ParentLease:
    """Hold an admitted parent directory open against replacement on Windows."""

    def __init__(self, handle: int, identity: tuple[int, int, int]) -> None:
        self.handle = handle
        self.identity = identity

    def close(self) -> None:
        if self.handle is None:
            return
        if not _close_directory_handle(self.handle):
            error = ctypes.get_last_error()
            raise OSError(error, os.strerror(error))
        self.handle = None

    def __enter__(self) -> "_ParentLease":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _close_directory_handle(handle: int) -> bool:
    """Close a native directory lease handle."""
    if os.name != "nt":
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    return bool(kernel32.CloseHandle(ctypes.c_void_p(handle)))


def _open_parent_lease(parent: Path, identity: tuple[int, int, int]) -> _ParentLease:
    """Open a Windows directory handle without delete sharing.

    Publication relies on this lease to prevent a concurrent rename/delete of
    the admitted parent while staging, linking, and cleanup use pathnames.  A
    platform without an equivalent primitive fails closed instead of claiming
    that pathname identity checks provide the same guarantee.
    """
    if os.name != "nt":
        raise ValueError("artifact parent lease is unsupported on this platform")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
    ]
    kernel32.CreateFileW.restype = ctypes.c_void_p
    # Request DELETE access as well as metadata access.  Omitting
    # FILE_SHARE_DELETE pins the parent against rename/delete until the handle
    # is closed on filesystems honoring Windows sharing semantics.
    delete_access = 0x00010000  # DELETE
    share_mode = 0x00000001 | 0x00000002  # FILE_SHARE_READ | FILE_SHARE_WRITE
    flags = 0x02000000  # FILE_FLAG_BACKUP_SEMANTICS
    invalid_handle = ctypes.c_void_p(-1).value
    handle = kernel32.CreateFileW(
        str(parent), delete_access, share_mode, None, 3, flags, None,
    )
    if handle in {None, invalid_handle}:
        error = ctypes.get_last_error()
        raise OSError(error, os.strerror(error), str(parent))
    return _ParentLease(int(handle), identity)


def _admit_parent(parent: Path) -> _ParentLease:
    """Admit one non-aliased parent and retain a replacement-proof lease."""
    if not parent.is_absolute() or not parent.is_dir() or parent.resolve(strict=True) != parent:
        raise ValueError("artifact target must be a new file in an existing non-aliased directory")
    identity = _parent_identity(parent)
    lease = _open_parent_lease(parent, identity)
    try:
        # Close the check-to-open window: if the pathname changed before the
        # lease was acquired, reject the lease rather than pinning a different
        # directory under the original target name.
        if _parent_identity(parent) != identity:
            raise ValueError("artifact parent directory changed during admission")
    except BaseException:
        try:
            lease.close()
        except OSError:
            pass
        raise
    return lease


def _assert_parent_identity(parent: Path, identity: tuple[int, int, int]) -> None:
    """Fail closed if the admitted parent was replaced or aliased."""
    if _parent_identity(parent) != identity:
        raise ValueError("artifact parent directory changed during publication")


def _cleanup_staging_path(path: Path, parent: Path, identity: tuple[int, int, int]) -> None:
    """Remove one owned staging path without masking the original failure."""
    try:
        _assert_parent_identity(parent, identity)
        path.unlink()
    except OSError:
        pass
    except ValueError:
        # Never resolve a path in a replacement directory during failure
        # cleanup.  Leaving the private staging file is safer than deleting
        # an unrelated file selected by a path race.
        pass


def _lexical_absolute_path(path: Path) -> Path:
    """Make a path absolute without resolving a final symlink alias."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(os.path.abspath(os.fspath(candidate)))
    return candidate


def _validate_new_target(target: Path) -> _ParentLease:
    """Reject aliases and existing targets immediately before staging/linking."""
    parent = target.parent
    if (not target.is_absolute() or os.path.lexists(target)):
        raise ValueError(
            "artifact target must be a new file in an existing non-aliased directory",
        )
    return _admit_parent(parent)


def publish_new_bytes_with_state(
    path: Path, payload: bytes, *, expected_sha: str,
    parent_lease: Directory_lease | None = None,
) -> TrackBArtifactPublicationV1:
    """Publish bytes once and return explicit post-link cleanup state.

    Pre-link failures remain exceptions so existing callers cannot mistake an
    unpublished artifact for success.  Once the hard link exists, a failure to
    remove the private staging name is returned as a committed-but-incomplete
    state; the target link exists, but publication cleanup is not confirmed.
    """
    # Do not resolve the final component here.  The caller's preflight has
    # already selected a canonical parent and basename; resolving again would
    # let a dangling symlink created after preflight redirect publication into
    # a protected input tree.
    target = _lexical_absolute_path(path)
    require_sha256(expected_sha, "artifact_expected_sha")
    if type(payload) is not bytes:
        raise TypeError("artifact payload must be bytes")
    parent = target.parent
    borrowed = parent_lease is not None
    if borrowed:
        if parent_lease.path != parent or os.path.lexists(target):
            raise ValueError("borrowed lease does not bind a new target in its exact parent")
        parent_lease.verify()
        parent_identity = _parent_identity(parent)
    else:
        parent_lease = _validate_new_target(target)
        parent_identity = parent_lease.identity
    temporary: Path | None = None
    published = False
    cleanup_attempted = False
    body_error: BaseException | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.staging-",
            dir=str(target.parent),
        )
        temporary = Path(temporary_name)
        try:
            handle = os.fdopen(descriptor, "w+b")
        except BaseException:
            # fdopen may fail before transferring ownership of the descriptor.
            # Close it explicitly, while tolerating an implementation that did
            # transfer ownership before raising.
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            handle.seek(0)
            actual_sha = hashlib.sha256(handle.read()).hexdigest()
            staged_file_identity = file_identity_from_descriptor(handle.fileno())
        if actual_sha != expected_sha:
            raise ValueError("staged artifact bytes do not match their expected SHA")
        # Re-check the lexical target and its parent after staging.  A target
        # alias appearing during staging must fail closed before the hard link.
        _assert_parent_identity(parent, parent_identity)
        if borrowed:
            parent_lease.verify()
        if os.path.lexists(target):
            raise ValueError(
                "artifact target must be a new file in an existing non-aliased directory",
            )
        # A hard link is atomic and refuses to replace a concurrent/existing
        # target on the same filesystem; os.replace would silently overwrite.
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise ValueError("artifact target appeared during publication") from error
        published = True
        cleanup_attempted = True
        try:
            temporary.unlink()
        except OSError as error:
            return TrackBArtifactPublicationV1(
                target_path=str(target), artifact_sha256=actual_sha,
                status=PUBLICATION_COMMITTED_CLEANUP_INCOMPLETE,
                target_published=True, staging_cleanup_complete=False,
                staging_path=str(temporary), cleanup_error=str(error),
                file_identity=staged_file_identity,
            )
        return TrackBArtifactPublicationV1(
            target_path=str(target), artifact_sha256=actual_sha,
            status=PUBLICATION_COMMITTED, target_published=True,
            staging_cleanup_complete=True, staging_path=str(temporary),
            file_identity=staged_file_identity,
        )
    except BaseException as error:
        body_error = error
        raise
    finally:
        if not published and not cleanup_attempted:
            if temporary is not None:
                _cleanup_staging_path(temporary, parent, parent_identity)
        if not borrowed:
            try:
                parent_lease.close()
            except OSError:
                # A lease-close failure must not replace an earlier publication or
                # staging exception.  Surface it only when the body was otherwise
                # successful.
                if body_error is None:
                    raise


def publish_new_bytes(path: Path, payload: bytes, *, expected_sha: str) -> str:
    """Compatibility wrapper returning the published SHA-256 digest."""
    return publish_new_bytes_with_state(path, payload, expected_sha=expected_sha).artifact_sha256


__all__ = [
    "PUBLICATION_COMMITTED", "PUBLICATION_COMMITTED_CLEANUP_INCOMPLETE",
    "TrackBArtifactPublicationV1", "publish_new_bytes", "publish_new_bytes_with_state",
]
