"""Fail-closed W_CH1 amp/phase matrix updates for external automation.

This module applies a pre-materialized, matrix-shaped log-amplitude and phase
adjustment to the final system-control W_CH1 text matrices.  It never creates
missing matrices, changes another channel, generates a waveform, uploads to an
AWG, or authorizes hardware activity.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
from io import StringIO
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np


MATRIX_UPDATE_SCHEMA = "WCh1MatrixAdjustmentV1"
MATRIX_UPDATE_RECEIPT_SCHEMA = "WCh1MatrixUpdateReceiptV1"
MATRIX_BACKUP_MANIFEST_SCHEMA = "WCh1MatrixBackupManifestV1"
MATRIX_FLOAT_FORMAT = "%.17g"
SYSTEM_SUFFIXES = {"3qubit": "", "4qubit": "_4qubit"}
PAIR_LOCK_SUFFIX = ".w_ch1_matrix_update.lock"
PAIR_RECOVERY_SUFFIX = ".w_ch1_matrix_recovery_required.json"
PAIR_VERSION_SUFFIX = ".w_ch1_matrix_pair_version.json"
PAIR_VERSION_SCHEMA = "WCh1MatrixPairVersionV1"


@dataclass(frozen=True)
class WCh1MatrixAdjustmentV1:
    """Pre-materialized matrix deltas from an external logical/RF adapter."""

    system_type: str
    amplitude_log_delta: np.ndarray
    phase_delta: np.ndarray
    source_id: str
    schema: str = MATRIX_UPDATE_SCHEMA

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "WCh1MatrixAdjustmentV1":
        expected_keys = {
            "schema",
            "system_type",
            "amplitude_log_delta",
            "phase_delta",
            "source_id",
        }
        if set(payload) != expected_keys:
            raise ValueError("W_CH1 adjustment JSON has unsupported or missing fields")
        amplitude_log_delta = _strict_json_matrix(
            payload["amplitude_log_delta"], "amplitude_log_delta",
        )
        phase_delta = _strict_json_matrix(payload["phase_delta"], "phase_delta")
        adjustment = cls(
            system_type=payload["system_type"],
            amplitude_log_delta=amplitude_log_delta,
            phase_delta=phase_delta,
            source_id=payload["source_id"],
            schema=payload["schema"],
        )
        adjustment.validate()
        return adjustment

    def validate(self) -> None:
        if self.system_type not in SYSTEM_SUFFIXES:
            raise ValueError("W_CH1 adjustment system_type is invalid")
        if self.schema != MATRIX_UPDATE_SCHEMA:
            raise ValueError("W_CH1 adjustment schema is invalid")
        if type(self.source_id) is not str or not self.source_id.strip():
            raise ValueError("W_CH1 adjustment source_id is invalid")
        for name, value in (
            ("amplitude_log_delta", self.amplitude_log_delta),
            ("phase_delta", self.phase_delta),
        ):
            if not isinstance(value, np.ndarray) or value.ndim != 2 or not value.size:
                raise ValueError(f"W_CH1 {name} must be a nonempty two-dimensional matrix")
            if (np.issubdtype(value.dtype, np.bool_)
                    or np.issubdtype(value.dtype, np.complexfloating)
                    or not np.issubdtype(value.dtype, np.number)):
                raise ValueError(f"W_CH1 {name} must use a real numeric dtype")
            if not bool(np.isfinite(value).all()):
                raise ValueError(f"W_CH1 {name} must be finite")
        if self.amplitude_log_delta.shape != self.phase_delta.shape:
            raise ValueError("W_CH1 amplitude and phase adjustment shapes differ")

    def sha256(self) -> str:
        self.validate()
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        digest.update(self.system_type.encode("utf-8"))
        digest.update(self.source_id.encode("utf-8"))
        for matrix in (self.amplitude_log_delta, self.phase_delta):
            digest.update(np.asarray(matrix, dtype=np.float64).tobytes(order="C"))
            digest.update(str(matrix.shape).encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True)
class WCh1MatrixUpdateReceiptV1:
    """Evidence for a dry-run, apply, or restore operation without hardware authority."""

    operation: str
    system_type: str
    matrix_dir: str
    amp_path: str
    phase_path: str
    backup_dir: str | None
    adjustment_sha256: str | None
    amp_before_sha256: str
    phase_before_sha256: str
    amp_after_sha256: str
    phase_after_sha256: str
    dry_run: bool
    research_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    waveform_generation: bool = False
    hardware_action_authorized: bool = False
    schema: str = MATRIX_UPDATE_RECEIPT_SCHEMA

    def validate(self) -> None:
        if self.operation not in {"apply", "restore"}:
            raise ValueError("W_CH1 receipt operation is invalid")
        if self.system_type not in SYSTEM_SUFFIXES or self.schema != MATRIX_UPDATE_RECEIPT_SCHEMA:
            raise ValueError("W_CH1 receipt identity is invalid")
        if (not self.research_only or self.selection_evidence or self.promotion_ready
                or self.waveform_generation or self.hardware_action_authorized):
            raise ValueError("W_CH1 receipt cannot authorize selection, waveform, or hardware")
        for value in (
            self.amp_before_sha256,
            self.phase_before_sha256,
            self.amp_after_sha256,
            self.phase_after_sha256,
        ):
            if type(value) is not str or len(value) != 64:
                raise ValueError("W_CH1 receipt matrix identity is invalid")
        if self.adjustment_sha256 is not None and (
                type(self.adjustment_sha256) is not str or len(self.adjustment_sha256) != 64):
            raise ValueError("W_CH1 receipt adjustment identity is invalid")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class _MatrixPairSnapshot:
    """One locked, byte-identical source pair used for computation and backup."""

    amp_path: Path
    phase_path: Path
    amp_bytes: bytes
    phase_bytes: bytes
    amp_matrix: np.ndarray
    phase_matrix: np.ndarray


@dataclass(frozen=True)
class _MatrixPairVersion:
    """A durable, monotonically advancing identity for a complete pair commit."""

    generation: int
    amp_sha256: str
    phase_sha256: str
    pair_sha256: str


@dataclass
class _MatrixPairLockHandle:
    """A cooperative writer lock that becomes a fail-closed recovery interlock."""

    lock_path: Path
    retain_lock: bool = False

    def retain_until_pair_verified(self) -> None:
        """Keep the lock through every state in which the pair may be partial."""
        self.retain_lock = True

    def release_after_pair_verified(self) -> None:
        """Allow normal lock cleanup only after a full pair is verified."""
        self.retain_lock = False

    def record_unverified_recovery(
        self,
        snapshot: _MatrixPairSnapshot,
        error: BaseException,
    ) -> None:
        """Write the recovery marker or retain the existing lock as its fallback."""
        self.retain_until_pair_verified()
        try:
            _write_recovery_required_marker(snapshot, error)
        except BaseException:
            # The already-created lock is the durable fallback interlock.
            return


def _strict_json_matrix(value: Any, name: str) -> np.ndarray:
    """Reject coercible JSON values before turning them into numeric arrays."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"W_CH1 {name} must be a nonempty rectangular JSON matrix")
    rows: list[list[float]] = []
    expected_width: int | None = None
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or not row:
            raise ValueError(f"W_CH1 {name}[{row_index}] must be a nonempty JSON row")
        if expected_width is None:
            expected_width = len(row)
        elif len(row) != expected_width:
            raise ValueError(f"W_CH1 {name} must be rectangular")
        parsed_row: list[float] = []
        for column_index, item in enumerate(row):
            if type(item) not in (int, float) or not math.isfinite(float(item)):
                raise ValueError(
                    f"W_CH1 {name}[{row_index}][{column_index}] must be a finite real JSON number",
                )
            parsed_row.append(float(item))
        rows.append(parsed_row)
    return np.asarray(rows, dtype=np.float64)


def default_matrix_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "qubit_recon_files" / "matrix"


def target_matrix_paths(matrix_dir: str | Path, system_type: str) -> tuple[Path, Path]:
    if system_type not in SYSTEM_SUFFIXES:
        raise ValueError("W_CH1 matrix system_type is invalid")
    root = Path(matrix_dir).resolve()
    suffix = SYSTEM_SUFFIXES[system_type]
    return (
        root / f"W_CH1_amp{suffix}.txt",
        root / f"W_CH1_phase{suffix}.txt",
    )


def load_w_ch1_adjustment(path: str | Path) -> WCh1MatrixAdjustmentV1:
    adjustment_path = Path(path)
    try:
        payload = json.loads(adjustment_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read W_CH1 adjustment JSON: {adjustment_path}") from error
    if not isinstance(payload, dict):
        raise ValueError("W_CH1 adjustment JSON must be an object")
    return WCh1MatrixAdjustmentV1.from_mapping(payload)


def apply_w_ch1_matrix_adjustment(
    adjustment: WCh1MatrixAdjustmentV1,
    *,
    matrix_dir: str | Path,
    backup_dir: str | Path | None,
    apply: bool,
    replace_func: Callable[[str | Path, str | Path], None] = os.replace,
    write_temp_func: Callable[[Path, bytes], Path] | None = None,
) -> WCh1MatrixUpdateReceiptV1:
    """Validate both final W_CH1 matrices, then update both or neither."""
    adjustment.validate()
    amp_path, phase_path = target_matrix_paths(matrix_dir, adjustment.system_type)
    resolved_backup_dir = _resolve_backup_dir(backup_dir, apply=apply)
    if not apply:
        preview_snapshot = _capture_preview_pair_snapshot(amp_path, phase_path)
        snapshot, amp_after, phase_after, receipt = _prepare_apply_update(
            adjustment,
            matrix_dir=matrix_dir,
            amp_path=amp_path,
            phase_path=phase_path,
            backup_dir=None,
            snapshot=preview_snapshot,
        )
        return receipt
    with _matrix_pair_lock(amp_path, phase_path) as pair_lock:
        snapshot, amp_after, phase_after, receipt = _prepare_apply_update(
            adjustment,
            matrix_dir=matrix_dir,
            amp_path=amp_path,
            phase_path=phase_path,
            backup_dir=resolved_backup_dir,
        )
        pair_version = _pair_version_for_snapshot(snapshot)

        def write_backup() -> None:
            _assert_snapshot_current(snapshot)
            _write_backup(resolved_backup_dir, snapshot, receipt)

        _replace_pair_atomically(
            snapshot,
            amp_after,
            phase_after,
            before_commit=write_backup,
            replace_func=replace_func,
            write_temp_func=write_temp_func or _write_temp_bytes,
            retain_interlock=pair_lock.retain_until_pair_verified,
            release_interlock=pair_lock.release_after_pair_verified,
            recovery_blocker=pair_lock.record_unverified_recovery,
            pair_version=pair_version,
            track_pair_version=True,
        )
        return receipt


def restore_w_ch1_matrix_backup(
    backup_dir: str | Path,
    *,
    matrix_dir: str | Path,
    system_type: str,
    apply: bool,
    replace_func: Callable[[str | Path, str | Path], None] = os.replace,
    write_temp_func: Callable[[Path, bytes], Path] | None = None,
) -> WCh1MatrixUpdateReceiptV1:
    """Restore a verified W_CH1 backup pair without affecting other files."""
    resolved_backup_dir = Path(backup_dir).resolve()
    amp_path, phase_path = target_matrix_paths(matrix_dir, system_type)
    if not apply:
        preview_snapshot = _capture_preview_pair_snapshot(amp_path, phase_path)
        snapshot, amp_before, phase_before, receipt = _prepare_restore_update(
            resolved_backup_dir,
            matrix_dir=matrix_dir,
            system_type=system_type,
            amp_path=amp_path,
            phase_path=phase_path,
            apply=False,
            snapshot=preview_snapshot,
        )
        return receipt
    with _matrix_pair_lock(amp_path, phase_path) as pair_lock:
        snapshot, amp_before, phase_before, receipt = _prepare_restore_update(
            resolved_backup_dir,
            matrix_dir=matrix_dir,
            system_type=system_type,
            amp_path=amp_path,
            phase_path=phase_path,
            apply=True,
        )
        pair_version = _pair_version_for_snapshot(snapshot)
        _replace_pair_atomically(
            snapshot,
            amp_before,
            phase_before,
            before_commit=None,
            replace_func=replace_func,
            write_temp_func=write_temp_func or _write_temp_bytes,
            retain_interlock=pair_lock.retain_until_pair_verified,
            release_interlock=pair_lock.release_after_pair_verified,
            recovery_blocker=pair_lock.record_unverified_recovery,
            pair_version=pair_version,
            track_pair_version=True,
        )
        return receipt


def _prepare_apply_update(
    adjustment: WCh1MatrixAdjustmentV1,
    *,
    matrix_dir: str | Path,
    amp_path: Path,
    phase_path: Path,
    backup_dir: Path | None,
    snapshot: _MatrixPairSnapshot | None = None,
) -> tuple[_MatrixPairSnapshot, bytes, bytes, WCh1MatrixUpdateReceiptV1]:
    snapshot = snapshot or _capture_matrix_pair_snapshot(amp_path, phase_path)
    if adjustment.amplitude_log_delta.shape != snapshot.amp_matrix.shape:
        raise ValueError("W_CH1 adjustment shape differs from final matrix shape")
    updated_amp, updated_phase = _apply_matrix_deltas(
        snapshot.amp_matrix,
        snapshot.phase_matrix,
        adjustment.amplitude_log_delta,
        adjustment.phase_delta,
    )
    amp_after = _matrix_bytes(updated_amp)
    phase_after = _matrix_bytes(updated_phase)
    receipt = WCh1MatrixUpdateReceiptV1(
        operation="apply",
        system_type=adjustment.system_type,
        matrix_dir=str(Path(matrix_dir).resolve()),
        amp_path=str(amp_path),
        phase_path=str(phase_path),
        backup_dir=str(backup_dir) if backup_dir is not None else None,
        adjustment_sha256=adjustment.sha256(),
        amp_before_sha256=_sha256_bytes(snapshot.amp_bytes),
        phase_before_sha256=_sha256_bytes(snapshot.phase_bytes),
        amp_after_sha256=_sha256_bytes(amp_after),
        phase_after_sha256=_sha256_bytes(phase_after),
        dry_run=backup_dir is None,
    )
    return snapshot, amp_after, phase_after, receipt


def _prepare_restore_update(
    backup_dir: Path,
    *,
    matrix_dir: str | Path,
    system_type: str,
    amp_path: Path,
    phase_path: Path,
    apply: bool,
    snapshot: _MatrixPairSnapshot | None = None,
) -> tuple[_MatrixPairSnapshot, bytes, bytes, WCh1MatrixUpdateReceiptV1]:
    snapshot = snapshot or _capture_matrix_pair_snapshot(amp_path, phase_path)
    manifest = _load_backup_manifest(backup_dir)
    _validate_backup_manifest(
        manifest,
        system_type=system_type,
        matrix_dir=Path(matrix_dir).resolve(),
        amp_path=amp_path,
        phase_path=phase_path,
    )
    amp_before = _read_verified_backup(
        backup_dir / amp_path.name,
        manifest,
        "amp_before_sha256",
    )
    phase_before = _read_verified_backup(
        backup_dir / phase_path.name,
        manifest,
        "phase_before_sha256",
    )
    _validate_backup_matrix_shapes(amp_before, phase_before, manifest, snapshot)
    receipt = WCh1MatrixUpdateReceiptV1(
        operation="restore",
        system_type=system_type,
        matrix_dir=str(Path(matrix_dir).resolve()),
        amp_path=str(amp_path),
        phase_path=str(phase_path),
        backup_dir=str(backup_dir),
        adjustment_sha256=None,
        amp_before_sha256=_sha256_bytes(snapshot.amp_bytes),
        phase_before_sha256=_sha256_bytes(snapshot.phase_bytes),
        amp_after_sha256=_sha256_bytes(amp_before),
        phase_after_sha256=_sha256_bytes(phase_before),
        dry_run=not apply,
    )
    return snapshot, amp_before, phase_before, receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-type", choices=tuple(SYSTEM_SUFFIXES), required=True)
    parser.add_argument("--matrix-dir", default=str(default_matrix_dir()))
    parser.add_argument("--adjustment-json")
    parser.add_argument("--restore-from")
    parser.add_argument("--backup-dir")
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.adjustment_json) == bool(args.restore_from):
        raise SystemExit("provide exactly one of --adjustment-json or --restore-from")
    if args.adjustment_json:
        adjustment = load_w_ch1_adjustment(args.adjustment_json)
        if adjustment.system_type != args.system_type:
            raise SystemExit("adjustment system_type differs from --system-type")
        receipt = apply_w_ch1_matrix_adjustment(
            adjustment,
            matrix_dir=args.matrix_dir,
            backup_dir=args.backup_dir,
            apply=args.apply,
        )
    else:
        receipt = restore_w_ch1_matrix_backup(
            args.restore_from,
            matrix_dir=args.matrix_dir,
            system_type=args.system_type,
            apply=args.apply,
        )
    print(json.dumps(receipt.to_dict(), sort_keys=True))
    return 0


@contextmanager
def _matrix_pair_lock(amp_path: Path, phase_path: Path) -> Iterator[_MatrixPairLockHandle]:
    """Serialize cooperating apply/restore writers for one final matrix pair."""
    lock_path = _pair_control_path(amp_path, phase_path, PAIR_LOCK_SUFFIX)
    _assert_not_recovery_blocked(amp_path, phase_path)
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise RuntimeError("W_CH1 matrix pair is already locked by another writer") from error
    pair_lock = _MatrixPairLockHandle(lock_path=lock_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
            json.dump(
                {
                    "schema": "WCh1MatrixPairLockV1",
                    "pid": os.getpid(),
                    "amp_path": str(amp_path),
                    "phase_path": str(phase_path),
                    "retained_on_unverified_recovery": True,
                },
                lock_file,
                sort_keys=True,
            )
        _assert_not_recovery_blocked(amp_path, phase_path)
        yield pair_lock
    finally:
        if not pair_lock.retain_lock:
            lock_path.unlink(missing_ok=True)


def _pair_control_path(amp_path: Path, phase_path: Path, suffix: str) -> Path:
    return amp_path.parent / f".{amp_path.stem}__{phase_path.stem}{suffix}"


def _assert_not_recovery_blocked(amp_path: Path, phase_path: Path) -> None:
    recovery_path = _pair_control_path(amp_path, phase_path, PAIR_RECOVERY_SUFFIX)
    if recovery_path.exists():
        raise RuntimeError("W_CH1 matrix pair is recovery-blocked; inspect the recovery record first")


def _assert_preview_pair_available(amp_path: Path, phase_path: Path) -> None:
    """Reject a preview while a cooperative writer owns or retains the pair lock."""
    _assert_not_recovery_blocked(amp_path, phase_path)
    lock_path = _pair_control_path(amp_path, phase_path, PAIR_LOCK_SUFFIX)
    if lock_path.exists():
        raise RuntimeError("W_CH1 matrix pair is locked; preview rejected")
    _assert_not_recovery_blocked(amp_path, phase_path)


def _read_pair_version(amp_path: Path, phase_path: Path) -> _MatrixPairVersion | None:
    """Read one complete writer-published pair version without creating state."""
    version_path = _pair_control_path(amp_path, phase_path, PAIR_VERSION_SUFFIX)
    if not version_path.exists():
        return None
    try:
        payload = json.loads(version_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("W_CH1 matrix pair version is unreadable") from error
    expected_keys = {
        "schema",
        "generation",
        "amp_sha256",
        "phase_sha256",
        "pair_sha256",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
        raise RuntimeError("W_CH1 matrix pair version is invalid")
    generation = payload["generation"]
    if type(generation) is not int or generation < 0:
        raise RuntimeError("W_CH1 matrix pair version generation is invalid")
    hashes = (payload["amp_sha256"], payload["phase_sha256"], payload["pair_sha256"])
    if any(type(value) is not str or len(value) != 64 for value in hashes):
        raise RuntimeError("W_CH1 matrix pair version identity is invalid")
    if payload["schema"] != PAIR_VERSION_SCHEMA:
        raise RuntimeError("W_CH1 matrix pair version schema is invalid")
    return _MatrixPairVersion(
        generation=generation,
        amp_sha256=payload["amp_sha256"],
        phase_sha256=payload["phase_sha256"],
        pair_sha256=payload["pair_sha256"],
    )


def _pair_version_for_snapshot(snapshot: _MatrixPairSnapshot) -> _MatrixPairVersion | None:
    """Read the latest generation while an exclusive writer owns the pair lock."""
    return _read_pair_version(snapshot.amp_path, snapshot.phase_path)


def _assert_pair_version_matches_snapshot(
    version: _MatrixPairVersion,
    snapshot: _MatrixPairSnapshot,
) -> None:
    expected_amp_sha = _sha256_bytes(snapshot.amp_bytes)
    expected_phase_sha = _sha256_bytes(snapshot.phase_bytes)
    if (
        version.amp_sha256 != expected_amp_sha
        or version.phase_sha256 != expected_phase_sha
        or version.pair_sha256 != _pair_sha256(snapshot.amp_bytes, snapshot.phase_bytes)
    ):
        raise RuntimeError("W_CH1 matrix pair version does not bind the captured snapshot")


def _capture_preview_pair_snapshot(amp_path: Path, phase_path: Path) -> _MatrixPairSnapshot:
    """Take a read-only snapshot that is bound to one writer commit generation."""
    _assert_preview_pair_available(amp_path, phase_path)
    version_before = _read_pair_version(amp_path, phase_path)
    snapshot = _capture_matrix_pair_snapshot(amp_path, phase_path)
    _assert_preview_pair_available(amp_path, phase_path)
    version_after = _read_pair_version(amp_path, phase_path)
    if version_before != version_after:
        raise RuntimeError("W_CH1 matrix pair version changed during preview")
    if version_after is not None:
        _assert_pair_version_matches_snapshot(version_after, snapshot)
    return snapshot


def _next_pair_version(
    previous_version: _MatrixPairVersion | None,
    amp_bytes: bytes,
    phase_bytes: bytes,
) -> _MatrixPairVersion:
    generation = 1 if previous_version is None else previous_version.generation + 1
    amp_sha = _sha256_bytes(amp_bytes)
    phase_sha = _sha256_bytes(phase_bytes)
    return _MatrixPairVersion(
        generation=generation,
        amp_sha256=amp_sha,
        phase_sha256=phase_sha,
        pair_sha256=_pair_sha256(amp_bytes, phase_bytes),
    )


def _write_pair_version(snapshot: _MatrixPairSnapshot, version: _MatrixPairVersion) -> None:
    """Atomically publish the identity of a complete pair after its bytes verify."""
    payload = json.dumps(
        {
            "schema": PAIR_VERSION_SCHEMA,
            "generation": version.generation,
            "amp_sha256": version.amp_sha256,
            "phase_sha256": version.phase_sha256,
            "pair_sha256": version.pair_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    _atomic_write_control_bytes(
        _pair_control_path(snapshot.amp_path, snapshot.phase_path, PAIR_VERSION_SUFFIX),
        payload,
    )


def _restore_pair_version_after_failed_commit(
    snapshot: _MatrixPairSnapshot,
    previous_version: _MatrixPairVersion | None,
) -> None:
    """Publish a fresh non-reusable generation for a verified rollback result."""
    current_version = _read_pair_version(snapshot.amp_path, snapshot.phase_path)
    current_generation = 0 if current_version is None else current_version.generation
    previous_generation = 0 if previous_version is None else previous_version.generation
    rollback_generation = max(current_generation, previous_generation) + 1
    version = _MatrixPairVersion(
        generation=rollback_generation,
        amp_sha256=_sha256_bytes(snapshot.amp_bytes),
        phase_sha256=_sha256_bytes(snapshot.phase_bytes),
        pair_sha256=_pair_sha256(snapshot.amp_bytes, snapshot.phase_bytes),
    )
    _write_pair_version(snapshot, version)


def _atomic_write_control_bytes(target_path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=f".{target_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_recovery_required_marker(snapshot: _MatrixPairSnapshot, error: BaseException) -> None:
    marker_path = _pair_control_path(
        snapshot.amp_path,
        snapshot.phase_path,
        PAIR_RECOVERY_SUFFIX,
    )
    marker_path.write_text(
        json.dumps(
            {
                "schema": "WCh1MatrixRecoveryRequiredV1",
                "amp_path": str(snapshot.amp_path),
                "phase_path": str(snapshot.phase_path),
                "amp_before_sha256": _sha256_bytes(snapshot.amp_bytes),
                "phase_before_sha256": _sha256_bytes(snapshot.phase_bytes),
                "error_type": type(error).__name__,
            },
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def _capture_matrix_pair_snapshot(amp_path: Path, phase_path: Path) -> _MatrixPairSnapshot:
    if not amp_path.is_file() or not phase_path.is_file():
        missing_path = amp_path if not amp_path.is_file() else phase_path
        raise FileNotFoundError(f"final W_CH1 matrix is missing: {missing_path}")
    amp_bytes = amp_path.read_bytes()
    phase_bytes = phase_path.read_bytes()
    amp_matrix = _load_matrix_bytes(amp_path, amp_bytes)
    phase_matrix = _load_matrix_bytes(phase_path, phase_bytes)
    if amp_matrix.shape != phase_matrix.shape:
        raise ValueError("final W_CH1 amp and phase matrix shapes differ")
    return _MatrixPairSnapshot(
        amp_path=amp_path,
        phase_path=phase_path,
        amp_bytes=amp_bytes,
        phase_bytes=phase_bytes,
        amp_matrix=amp_matrix,
        phase_matrix=phase_matrix,
    )


def _assert_snapshot_current(snapshot: _MatrixPairSnapshot) -> None:
    if (snapshot.amp_path.read_bytes() != snapshot.amp_bytes
            or snapshot.phase_path.read_bytes() != snapshot.phase_bytes):
        raise RuntimeError("W_CH1 matrix pair changed during the guarded update")


def _assert_pair_bytes(
    amp_path: Path,
    phase_path: Path,
    expected_amp: bytes,
    expected_phase: bytes,
) -> None:
    if amp_path.read_bytes() != expected_amp or phase_path.read_bytes() != expected_phase:
        raise RuntimeError("W_CH1 matrix pair changed during the guarded commit")


def _load_matrix_bytes(path: Path, payload: bytes) -> np.ndarray:
    try:
        matrix = np.asarray(np.loadtxt(StringIO(payload.decode("utf-8"))), dtype=np.float64)
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"final W_CH1 matrix cannot be parsed: {path}") from error
    if matrix.ndim != 2 or not matrix.size or not bool(np.isfinite(matrix).all()):
        raise ValueError(f"final W_CH1 matrix is invalid: {path}")
    return matrix


def _apply_matrix_deltas(
    amp_matrix: np.ndarray,
    phase_matrix: np.ndarray,
    amplitude_log_delta: np.ndarray,
    phase_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if bool((amp_matrix < 0.0).any()):
        raise ValueError("final W_CH1 amplitude matrix contains a negative value")
    try:
        with np.errstate(over="raise", invalid="raise"):
            updated_amp = amp_matrix * np.exp(amplitude_log_delta)
            updated_phase = phase_matrix + phase_delta
    except FloatingPointError as error:
        raise ValueError("W_CH1 adjustment produced a non-finite value") from error
    if not bool(np.isfinite(updated_amp).all()) or not bool(np.isfinite(updated_phase).all()):
        raise ValueError("W_CH1 adjustment produced a non-finite value")
    if bool((updated_amp < 0.0).any()):
        raise ValueError("W_CH1 adjustment produced a negative amplitude")
    return updated_amp, updated_phase


def _resolve_backup_dir(backup_dir: str | Path | None, *, apply: bool) -> Path | None:
    if not apply:
        return None
    if backup_dir is None:
        raise ValueError("--backup-dir is required with --apply")
    resolved_backup_dir = Path(backup_dir)
    if not resolved_backup_dir.is_absolute():
        raise ValueError("W_CH1 backup_dir must be absolute")
    resolved_backup_dir = resolved_backup_dir.resolve()
    if resolved_backup_dir.exists():
        raise FileExistsError("W_CH1 backup_dir must not already exist")
    return resolved_backup_dir


def _write_backup(
    backup_dir: Path,
    snapshot: _MatrixPairSnapshot,
    receipt: WCh1MatrixUpdateReceiptV1,
) -> None:
    backup_dir.mkdir(parents=True, exist_ok=False)
    try:
        (backup_dir / snapshot.amp_path.name).write_bytes(snapshot.amp_bytes)
        (backup_dir / snapshot.phase_path.name).write_bytes(snapshot.phase_bytes)
        manifest = {
            "schema": MATRIX_BACKUP_MANIFEST_SCHEMA,
            "system_type": receipt.system_type,
            "matrix_dir": receipt.matrix_dir,
            "amp_path": receipt.amp_path,
            "phase_path": receipt.phase_path,
            "amp_before_sha256": receipt.amp_before_sha256,
            "phase_before_sha256": receipt.phase_before_sha256,
            "amp_shape": list(snapshot.amp_matrix.shape),
            "phase_shape": list(snapshot.phase_matrix.shape),
            "pair_before_sha256": _pair_sha256(snapshot.amp_bytes, snapshot.phase_bytes),
            "adjustment_sha256": receipt.adjustment_sha256,
        }
        (backup_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        for child_path in backup_dir.iterdir():
            child_path.unlink(missing_ok=True)
        backup_dir.rmdir()
        raise


def _load_backup_manifest(backup_dir: Path) -> Mapping[str, Any]:
    try:
        manifest = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("W_CH1 backup manifest is unavailable") from error
    expected_keys = {
        "schema",
        "system_type",
        "matrix_dir",
        "amp_path",
        "phase_path",
        "amp_before_sha256",
        "phase_before_sha256",
        "amp_shape",
        "phase_shape",
        "pair_before_sha256",
        "adjustment_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise ValueError("W_CH1 backup manifest is invalid")
    return manifest


def _validate_backup_manifest(
    manifest: Mapping[str, Any],
    *,
    system_type: str,
    matrix_dir: Path,
    amp_path: Path,
    phase_path: Path,
) -> None:
    if manifest["schema"] != MATRIX_BACKUP_MANIFEST_SCHEMA:
        raise ValueError("W_CH1 backup manifest schema is invalid")
    if manifest["system_type"] != system_type:
        raise ValueError("W_CH1 backup system_type differs from restore request")
    expected_paths = {
        "matrix_dir": str(matrix_dir),
        "amp_path": str(amp_path),
        "phase_path": str(phase_path),
    }
    if any(manifest[name] != value for name, value in expected_paths.items()):
        raise ValueError("W_CH1 backup targets differ from restore request")
    for name in ("amp_before_sha256", "phase_before_sha256", "pair_before_sha256"):
        value = manifest[name]
        if type(value) is not str or len(value) != 64:
            raise ValueError("W_CH1 backup manifest hash is invalid")
    for name in ("amp_shape", "phase_shape"):
        shape = manifest[name]
        if (not isinstance(shape, list) or len(shape) != 2
                or any(type(value) is not int or value <= 0 for value in shape)):
            raise ValueError("W_CH1 backup manifest shape is invalid")
    if manifest["adjustment_sha256"] is not None and (
            type(manifest["adjustment_sha256"]) is not str
            or len(manifest["adjustment_sha256"]) != 64):
        raise ValueError("W_CH1 backup manifest adjustment identity is invalid")


def _replace_pair_atomically(
    snapshot: _MatrixPairSnapshot,
    amp_after: bytes,
    phase_after: bytes,
    *,
    before_commit: Callable[[], None] | None,
    replace_func: Callable[[str | Path, str | Path], None],
    write_temp_func: Callable[[Path, bytes], Path],
    retain_interlock: Callable[[], None] | None = None,
    release_interlock: Callable[[], None] | None = None,
    recovery_blocker: Callable[[_MatrixPairSnapshot, BaseException], None] | None = None,
    pair_version: _MatrixPairVersion | None = None,
    track_pair_version: bool = False,
) -> None:
    temporary_paths: list[Path] = []
    commit_started = False
    try:
        amp_temp = write_temp_func(snapshot.amp_path, amp_after)
        temporary_paths.append(amp_temp)
        phase_temp = write_temp_func(snapshot.phase_path, phase_after)
        temporary_paths.append(phase_temp)
        if before_commit is not None:
            before_commit()
        _assert_snapshot_current(snapshot)
        commit_started = True
        if retain_interlock is not None:
            retain_interlock()
        replace_func(amp_temp, snapshot.amp_path)
        temporary_paths.remove(amp_temp)
        _assert_pair_bytes(
            snapshot.amp_path,
            snapshot.phase_path,
            amp_after,
            snapshot.phase_bytes,
        )
        replace_func(phase_temp, snapshot.phase_path)
        temporary_paths.remove(phase_temp)
        _assert_pair_bytes(
            snapshot.amp_path,
            snapshot.phase_path,
            amp_after,
            phase_after,
        )
        if track_pair_version:
            _write_pair_version(
                snapshot,
                _next_pair_version(pair_version, amp_after, phase_after),
            )
    except BaseException as error:
        if commit_started:
            if retain_interlock is not None:
                retain_interlock()
            try:
                _restore_pair_from_snapshot(
                    snapshot,
                    replace_func,
                    write_temp_func,
                )
                if track_pair_version:
                    _restore_pair_version_after_failed_commit(snapshot, pair_version)
            except BaseException as restore_error:
                if recovery_blocker is None:
                    _write_recovery_required_marker(snapshot, restore_error)
                else:
                    recovery_blocker(snapshot, restore_error)
                raise RuntimeError("W_CH1 matrix pair update failed and rollback failed") from restore_error
            if release_interlock is not None:
                release_interlock()
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise RuntimeError("W_CH1 matrix pair update failed and was restored") from error
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise RuntimeError("W_CH1 matrix pair update failed before completion") from error
    else:
        if release_interlock is not None:
            release_interlock()
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def _restore_pair_from_snapshot(
    snapshot: _MatrixPairSnapshot,
    replace_func: Callable[[str | Path, str | Path], None],
    write_temp_func: Callable[[Path, bytes], Path],
) -> None:
    temporary_paths: list[Path] = []
    try:
        amp_temp = write_temp_func(snapshot.amp_path, snapshot.amp_bytes)
        temporary_paths.append(amp_temp)
        phase_temp = write_temp_func(snapshot.phase_path, snapshot.phase_bytes)
        temporary_paths.append(phase_temp)
        replace_func(amp_temp, snapshot.amp_path)
        temporary_paths.remove(amp_temp)
        replace_func(phase_temp, snapshot.phase_path)
        temporary_paths.remove(phase_temp)
        _assert_snapshot_current(snapshot)
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def _write_temp_bytes(target_path: Path, payload: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=f".{target_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(payload)
        return temporary_path
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _matrix_bytes(matrix: np.ndarray) -> bytes:
    buffer = StringIO()
    np.savetxt(buffer, matrix, fmt=MATRIX_FLOAT_FORMAT)
    return buffer.getvalue().encode("utf-8")


def _read_verified_backup(backup_path: Path, manifest: Mapping[str, Any], sha_key: str) -> bytes:
    if not backup_path.is_file():
        raise ValueError(f"W_CH1 backup file is missing: {backup_path.name}")
    payload = backup_path.read_bytes()
    if manifest.get(sha_key) != _sha256_bytes(payload):
        raise ValueError(f"W_CH1 backup file hash differs: {backup_path.name}")
    return payload


def _validate_backup_matrix_shapes(
    amp_backup: bytes,
    phase_backup: bytes,
    manifest: Mapping[str, Any],
    snapshot: _MatrixPairSnapshot,
) -> None:
    amp_matrix = _load_matrix_bytes(snapshot.amp_path, amp_backup)
    phase_matrix = _load_matrix_bytes(snapshot.phase_path, phase_backup)
    if (list(amp_matrix.shape) != manifest["amp_shape"]
            or list(phase_matrix.shape) != manifest["phase_shape"]
            or amp_matrix.shape != phase_matrix.shape
            or amp_matrix.shape != snapshot.amp_matrix.shape):
        raise ValueError("W_CH1 backup matrix shape differs from final matrix pair")
    if _pair_sha256(amp_backup, phase_backup) != manifest["pair_before_sha256"]:
        raise ValueError("W_CH1 backup pair identity differs from its manifest")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pair_sha256(amp_bytes: bytes, phase_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"WCh1MatrixPairV1\x00")
    digest.update(amp_bytes)
    digest.update(b"\x00")
    digest.update(phase_bytes)
    return digest.hexdigest()


__all__ = [
    "WCh1MatrixAdjustmentV1",
    "WCh1MatrixUpdateReceiptV1",
    "apply_w_ch1_matrix_adjustment",
    "default_matrix_dir",
    "load_w_ch1_adjustment",
    "main",
    "restore_w_ch1_matrix_backup",
    "target_matrix_paths",
]
