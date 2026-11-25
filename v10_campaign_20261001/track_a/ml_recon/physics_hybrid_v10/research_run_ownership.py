"""Exclusive run ownership for the research stream trainer.

External review (2026-10-02, round 2): nothing fenced a run's output
directory.  The fresh-launch empty-directory check is a plain TOCTOU (two
concurrent launches both pass before either writes an artifact), and resume
launches had no fencing at all, so a resume started while the original
trainer still runs produces two writers of ``metrics.jsonl`` and the
checkpoints — the duplicate-trainer crash class.

Design (Windows-safe; the trainer runtime has no ``fcntl``):

- claim: atomic ``O_CREAT | O_EXCL`` creation of ``run_owner.json`` inside
  the output directory; exactly one launcher wins.
- an existing record refuses the launch unless the recorded PID is
  verifiably dead on this host (``OpenProcess`` + ``GetExitCodeProcess``,
  fail-closed: access-denied or undeterminable counts as alive, and a
  terminated process that happened to exit with code 259 also reads as
  alive).  A hard kill leaves the record behind; the next resume launch
  archives it as ``run_owner.stale.json`` (latest stale record only).
- takeover races between two resumers are closed by a content check: the
  dead record is renamed to the archive path, the renamed file is re-read,
  and anything other than the record that was verified dead is left in the
  archive while the launch refuses — a live lock is never clobbered.
- the trainer asserts ownership at every epoch boundary (a superseded owner
  stops writing artifacts at the next boundary) and releases the record on
  every exit path via ``try``/``finally`` in the caller.
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

OWNERSHIP_SCHEMA = "V10RunOwnershipV1"
LOCK_NAME = "run_owner.json"
ARCHIVE_NAME = "run_owner.stale.json"

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5


class RunOwnershipLost(RuntimeError):
    """The ownership lock no longer names this process.

    Raised at epoch boundaries when another launcher has legitimately taken
    over (the recorded owner was verified dead); the losing trainer must
    stop writing run artifacts immediately.
    """


@dataclass(frozen=True)
class OwnershipRecord:
    schema: str
    pid: int
    hostname: str
    started_unix: float
    run_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "pid": self.pid,
            "hostname": self.hostname,
            "started_unix": self.started_unix,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "OwnershipRecord":
        if not isinstance(value, dict) or value.get("schema") != OWNERSHIP_SCHEMA:
            raise ValueError("run ownership record has an unknown schema")
        pid = value.get("pid")
        hostname = value.get("hostname")
        started = value.get("started_unix")
        run_id = value.get("run_id")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise ValueError("run ownership record has an invalid pid")
        if not isinstance(hostname, str) or not hostname:
            raise ValueError("run ownership record has an invalid hostname")
        if (not isinstance(started, (int, float)) or isinstance(started, bool)):
            raise ValueError("run ownership record has an invalid start time")
        if not isinstance(run_id, str):
            raise ValueError("run ownership record has an invalid run_id")
        return cls(schema=OWNERSHIP_SCHEMA, pid=int(pid), hostname=hostname,
                   started_unix=float(started), run_id=run_id)


def _process_alive_windows(pid: int) -> bool:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    handle = kernel32.OpenProcess(
        _PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        # ERROR_ACCESS_DENIED is a live process we may not query; any other
        # error means the PID does not exist.
        return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
    try:
        exit_code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        kernel32.GetExitCodeProcess.argtypes = (
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True  # undeterminable -> assume alive (fail closed)
        # A terminated process that exited with code 259 also reads as
        # alive here; the ambiguity always refuses, never corrupts.
        return int(exit_code.value) == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle(handle)


def process_alive(pid: int) -> bool:
    """Fail-closed PID liveness: undeterminable counts as alive."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if os.name == "nt":
        return _process_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_record(path: Path) -> OwnershipRecord:
    return OwnershipRecord.from_dict(
        json.loads(path.read_text(encoding="utf-8")))


def _write_record(lock_path: Path, record: OwnershipRecord) -> None:
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record.to_dict(), handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass
class ExclusiveRunOwnership:
    """Handle for one claimed run; assert at epoch boundaries, release on exit."""

    output_dir: Path
    lock_path: Path
    record: OwnershipRecord
    archived_previous: Optional[OwnershipRecord] = None

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": OWNERSHIP_SCHEMA,
            "lock_path": str(self.lock_path),
            "pid": self.record.pid,
            "hostname": self.record.hostname,
            "started_unix": self.record.started_unix,
            "run_id": self.record.run_id,
            "took_over_from": (self.archived_previous.to_dict()
                               if self.archived_previous is not None else None),
        }

    def assert_owned(self) -> None:
        try:
            current = _read_record(self.lock_path)
        except (OSError, ValueError) as error:
            raise RunOwnershipLost(
                f"run ownership lock {self.lock_path} is no longer readable "
                "by this process; refusing to continue writing run artifacts"
            ) from error
        if (current.pid, current.started_unix) != (
                self.record.pid, self.record.started_unix):
            raise RunOwnershipLost(
                f"run ownership of {self.output_dir} was taken over by pid "
                f"{current.pid}; refusing to continue writing run artifacts")

    def release(self) -> None:
        try:
            current = _read_record(self.lock_path)
        except (OSError, ValueError):
            return
        if (current.pid, current.started_unix) != (
                self.record.pid, self.record.started_unix):
            return  # the lock no longer names this process; leave it alone
        try:
            self.lock_path.unlink()
        except OSError:
            pass


def claim_exclusive_run_ownership(output_dir: Path | str,
                                  run_id: str) -> ExclusiveRunOwnership:
    """Atomically claim a run output directory or refuse with a clear reason.

    Fail-closed on every ambiguity: a live recorded owner, an owner on
    another host, a corrupt record, or a takeover race all raise
    ``RuntimeError`` instead of writing artifacts.
    """
    output = Path(output_dir)
    lock_path = output / LOCK_NAME
    record = OwnershipRecord(
        schema=OWNERSHIP_SCHEMA, pid=os.getpid(), hostname=platform.node(),
        started_unix=time.time(), run_id=str(run_id))
    try:
        _write_record(lock_path, record)
    except FileExistsError:
        pass
    else:
        ownership = ExclusiveRunOwnership(
            output_dir=output, lock_path=lock_path, record=record)
        ownership.assert_owned()
        return ownership

    existing = _read_record(lock_path)
    if existing.pid == record.pid:
        raise RuntimeError(
            f"run ownership lock {lock_path} already names this process "
            f"(pid {record.pid}); a process cannot re-claim its own run")
    if existing.hostname != record.hostname:
        raise RuntimeError(
            f"run ownership lock {lock_path} was left by host "
            f"{existing.hostname!r} (this host is {record.hostname!r}); "
            "cross-host takeover is not supported — remove the lock manually "
            "once that host's trainer is verified stopped")
    if process_alive(existing.pid):
        raise RuntimeError(
            f"output directory is owned by live pid {existing.pid} (host "
            f"{existing.hostname!r}, run_id {existing.run_id!r}, started "
            f"{existing.started_unix}); refusing a second trainer on {output}")
    archive_path = output / ARCHIVE_NAME
    # Re-read immediately before the move: if the lock changed since the
    # dead-PID verification (another launcher completed its takeover in the
    # window), the record at lock_path is a live claim and must not move.
    try:
        current = _read_record(lock_path)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"run ownership takeover raced on {lock_path}; retry the launch"
        ) from error
    if (current.pid, current.started_unix) != (
            existing.pid, existing.started_unix):
        raise RuntimeError(
            f"run ownership takeover raced on {lock_path}; retry the launch")
    try:
        os.replace(lock_path, archive_path)
    except OSError as error:
        raise RuntimeError(
            f"run ownership takeover raced on {lock_path}; retry the launch"
        ) from error
    archived = _read_record(archive_path)
    if (archived.pid, archived.started_unix) != (
            existing.pid, existing.started_unix):
        # Another launcher completed its takeover between the re-read and
        # the rename, so the file just archived is that launcher's LIVE
        # claim.  Put it back before refusing — destroying a live lock would
        # reopen the duplicate-trainer window this fence exists to close
        # (review workflow 2026-10-03).  If a further claim already appeared
        # at lock_path, never clobber it: the stale archive stays as
        # evidence and the superseded owner stops at its next epoch-boundary
        # assert_owned.
        try:
            if not lock_path.exists():
                os.replace(archive_path, lock_path)
        except OSError:
            pass
        raise RuntimeError(
            f"run ownership takeover raced on {lock_path}; retry the launch")
    try:
        _write_record(lock_path, record)
    except FileExistsError as error:
        raise RuntimeError(
            f"run ownership takeover lost the claim race on {lock_path}; "
            "retry the launch") from error
    ownership = ExclusiveRunOwnership(
        output_dir=output, lock_path=lock_path, record=record,
        archived_previous=archived)
    ownership.assert_owned()
    return ownership
