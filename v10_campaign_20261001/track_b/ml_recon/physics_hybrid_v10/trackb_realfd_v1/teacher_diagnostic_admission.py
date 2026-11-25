"""Inspect persisted teacher evidence under independently pinned native identities.

Inspection verifies complete durable content, not the original process exit or
historical handle-close outcome. It never creates, repairs, deletes or resumes a
root. Handles are closed before return: the immutable snapshots are evidence at
inspection time, not continuous filesystem locks or unforgeable authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat
from typing import Any

from ml_recon.data_v2_4.admission_snapshot import Canonical_json_snapshot
from ml_recon.data_v2_4.hashing import canonical_json_bytes, sha256_bytes

from .artifact_directory_lease import Directory_lease, close_directory_leases, verify_directory_leases
from .artifact_file_lease import File_lease, close_file_leases
from .contracts import require_git_oid, require_sha256
from .teacher_diagnostic_preflight import require_unchanged_teacher_diagnostic_preflight
from .teacher_diagnostic_run_contract import (
    Teacher_diagnostic_run_plan, artifact_max_bytes, decode_canonical_teacher_object,
    decode_teacher_diagnostic_result, plan_filename, ready_filename, result_filename,
    validate_teacher_diagnostic_result,
)


_factory_token = object()
_file_id_pattern = re.compile(r"[0-9a-f]{32}")


def validate_native_identity(value: tuple[int, str]) -> None:
    if (type(value) is not tuple or len(value) != 2 or type(value[0]) is not int or value[0] <= 0
            or type(value[1]) is not str or not _file_id_pattern.fullmatch(value[1]) or value[1] == "0" * 32):
        raise ValueError("native identity must be a nonzero volume/file-ID tuple")


@dataclass(frozen=True)
class Teacher_evidence_pin:
    """Caller-retained expectations; never construct these from candidate claims."""

    plan: Teacher_diagnostic_run_plan
    expected_plan_sha256: str
    expected_source: tuple[str, str]
    expected_root_identity: tuple[int, str]
    expected_ready_sha256: str
    expected_ready_identity: tuple[int, str]

    @property
    def root(self) -> Path:
        return self.plan.output_root

    def validate(self) -> None:
        if type(self.plan) is not Teacher_diagnostic_run_plan:
            raise ValueError("teacher evidence requires its complete independently pinned plan")
        self.plan.validate_identity()
        require_sha256(self.expected_plan_sha256, "teacher_plan_sha")
        require_sha256(self.expected_ready_sha256, "teacher_ready_sha")
        if self.expected_source != self.plan.expected_source or self.expected_plan_sha256 != self.plan.sha256():
            raise ValueError("teacher evidence source or plan differs from its independent pins")
        validate_native_identity(self.expected_root_identity)
        validate_native_identity(self.expected_ready_identity)

    def identity_summary(self) -> dict[str, Any]:
        self.validate()
        return {"schema": "TrackBTeacherEvidencePinV1", "run_id": self.plan.run_id,
                "plan_sha256": self.expected_plan_sha256, "ready_sha256": self.expected_ready_sha256,
                "source_commit": self.expected_source[0], "source_tree": self.expected_source[1],
                "root_identity": {"volume_serial_number": self.expected_root_identity[0],
                                  "file_id": self.expected_root_identity[1]},
                "ready_identity": {"volume_serial_number": self.expected_ready_identity[0],
                                   "file_id": self.expected_ready_identity[1]}}


@dataclass(frozen=True, init=False)
class Teacher_evidence_admission:
    """Factory-only immutable content snapshot, consumed by the Stage 2 entry."""

    source: tuple[str, str]
    plan_sha256: str
    root_identity: tuple[int, str]
    ready_identity: tuple[int, str]
    ready: Canonical_json_snapshot
    plan_record: Canonical_json_snapshot
    results: tuple[Canonical_json_snapshot, ...]

    def __init__(self, token: object, *, expected: Teacher_evidence_pin, ready: Canonical_json_snapshot,
                 plan_record: Canonical_json_snapshot, results: tuple[Canonical_json_snapshot, ...]) -> None:
        if token is not _factory_token:
            raise TypeError("teacher admissions are created only by complete persisted inspection")
        for name, value in (("source", expected.expected_source), ("plan_sha256", expected.expected_plan_sha256),
                            ("root_identity", expected.expected_root_identity), ("ready_identity", expected.expected_ready_identity),
                            ("ready", ready), ("plan_record", plan_record), ("results", results)):
            object.__setattr__(self, name, value)

    def identity_summary(self) -> dict[str, Any]:
        return {"schema": "TrackBTeacherEvidenceAdmissionV1", "inspection_kind": "INSPECTED_CONTENT_COMPLETION",
                "source_commit": self.source[0], "source_tree": self.source[1], "plan_sha256": self.plan_sha256,
                "ready_sha256": self.ready.sha256, "plan_record_sha256": self.plan_record.sha256,
                "result_sha256": [result.sha256 for result in self.results],
                "root_identity": {"volume_serial_number": self.root_identity[0], "file_id": self.root_identity[1]},
                "ready_identity": {"volume_serial_number": self.ready_identity[0], "file_id": self.ready_identity[1]},
                "original_process_exit": "NOT_PROVEN", "historical_handle_cleanup": "NOT_PROVEN",
                "continuous_filesystem_lock": False, "research_only": True, "training_authorized": False,
                "model_gain_measured": False, "gate_decision": "NOT_PERFORMED", "selection_evidence": False,
                "promotion_ready": False, "hardware_action_authorized": False}


def _revalidate_physics(expected: Teacher_evidence_pin) -> None:
    for key, binding in zip(expected.plan.sample_keys, expected.plan.preflight_bindings):
        require_unchanged_teacher_diagnostic_preflight(
            binding, sample_key=key, config=expected.plan.config, expected_source=expected.expected_source)


def _open_bounded_file(path: Path, expected_sha: str, root: Directory_lease,
                       expected_identity: tuple[int, str], *, expected_size: int | None = None) -> File_lease:
    """Reject oversized/nonregular path entries before native verification reads.

    The native lease checks the held handle's size before its first hash chunk,
    as well as bounding the later read. A path precheck alone is not a lock.
    """
    metadata = path.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 0x400
            or not 0 < metadata.st_size <= artifact_max_bytes
            or (expected_size is not None and metadata.st_size != expected_size)):
        raise ValueError("teacher artifact type or bounded byte size is invalid")
    return File_lease.open_verified(path, expected_sha, root, expected_identity=expected_identity,
                                    max_bytes=artifact_max_bytes if expected_size is None else expected_size)


def _expected_ready_fields(expected: Teacher_evidence_pin) -> dict[str, Any]:
    return {
        "schema": "TrackBTeacherDiagnosticReadyV1", "run_id": expected.plan.run_id,
        "plan_sha256": expected.expected_plan_sha256, "plan": expected.plan.portable_payload(),
        "root_identity": {"basename": expected.plan.run_id, "volume_serial_number": expected.expected_root_identity[0],
                          "file_id": expected.expected_root_identity[1]},
        "execution_phase": "TEACHER_KERNEL_COMPLETED_OUTPUTS_VERIFIED",
        "source_physical_preflight": "CAPTURED_AND_REVALIDATED_BEFORE_EXECUTION_AND_EACH_PUBLICATION",
        "teacher_kernel_completed": True, "completed_episode_count": len(expected.plan.sample_keys),
        "research_only": True, "training_authorized": False, "model_gain_measured": False,
        "empirical_model_gain": "NOT_PERFORMED", "scientific_gate": "NOT_PERFORMED",
        "gate_decision": "NOT_PERFORMED", "selection_evidence": False, "promotion_ready": False,
        "hardware_action_authorized": False, "external_source_review": "NOT_CERTIFIED_BY_RUNNER",
        "registered_study_approval": "EXTERNAL_WORKFLOW_REQUIRED",
        "ready_scope": "TEACHER_DIAGNOSTIC_ONLY_NOT_DATASET_OR_TRAINING_ADMISSION",
        "completion_boundary": "REQUIRES_SUCCESSFUL_RUN_RETURN_OR_INDEPENDENT_COMPLETION_INSPECTION",
    }


def _borrowed_parent_binding(parent: Directory_lease, expected_path: Path) -> tuple[int, Path, int, str]:
    """Verify the exact caller-owned live parent without acquiring another lease."""
    if (type(parent) is not Directory_lease or type(parent.handle) is not int or parent.handle <= 0
            or parent.path != expected_path):
        raise ValueError("borrowed teacher parent must be the exact live trusted-parent lease")
    parent.verify()  # Checks canonical path and native lexical identity.
    return parent.handle, parent.path, parent.volume_serial_number, parent.file_id


def inspect_teacher_diagnostic(
    expected: Teacher_evidence_pin, *, expected_source: tuple[str, str], expected_plan_sha256: str,
    borrowed_parent: Directory_lease | None = None,
) -> Teacher_evidence_admission:
    """Inspect pinned evidence, optionally borrowing the caller's exact parent.

    A borrowed parent is verified before/after inspection and is never closed or
    reacquired here. Only the teacher root and files are owned in that mode.
    No caller-supplied success flag substitutes for evidence checks.
    """
    if os.name != "nt":
        raise ValueError("persisted teacher inspection requires Windows native leases")
    if type(expected) is not Teacher_evidence_pin:
        raise ValueError("independent teacher evidence pins are required")
    expected.validate()
    if type(expected_source) is not tuple or len(expected_source) != 2:
        raise ValueError("independently pinned inspection source is required")
    for value in expected_source:
        require_git_oid(value, "teacher_inspection_source")
    require_sha256(expected_plan_sha256, "teacher_inspection_plan")
    if expected_source != expected.expected_source or expected_plan_sha256 != expected.expected_plan_sha256:
        raise ValueError("candidate teacher pins differ from independently supplied source/plan")
    parent_binding = (None if borrowed_parent is None else
                      _borrowed_parent_binding(borrowed_parent, expected.plan.trusted_output_parent))
    _revalidate_physics(expected)
    directories: list[Directory_lease] = []
    files: list[File_lease] = []
    body_error = None
    try:
        if borrowed_parent is None:
            parent = Directory_lease.open(expected.plan.trusted_output_parent)
            directories.append(parent)
        elif _borrowed_parent_binding(borrowed_parent, expected.plan.trusted_output_parent) != parent_binding:
            raise ValueError("borrowed teacher parent binding changed before inspection")
        root = Directory_lease.open(expected.root)
        directories.append(root)
        if (root.volume_serial_number, root.file_id) != expected.expected_root_identity:
            raise ValueError("teacher root native identity differs from its independent pin")
        ready_lease = _open_bounded_file(
            root.path / ready_filename, expected.expected_ready_sha256, root,
            expected_identity=expected.expected_ready_identity)
        files.append(ready_lease)
        ready_bytes = ready_lease.read_bytes(max_bytes=artifact_max_bytes)
        ready = decode_canonical_teacher_object(ready_bytes, label="teacher ready")
        fixed = _expected_ready_fields(expected)
        if (set(ready) != set(fixed) | {"inventory"}
                or canonical_json_bytes({name: ready[name] for name in fixed}) != canonical_json_bytes(fixed)):
            raise ValueError("teacher ready metadata, authority or plan does not match independent expectations")
        expected_names = (plan_filename,) + tuple(result_filename(index) for index in range(len(expected.plan.sample_keys)))
        inventory = ready["inventory"]
        if type(inventory) is not list or len(inventory) != len(expected_names):
            raise ValueError("teacher inventory is incomplete")
        if {path.name for path in root.path.iterdir()} != set(expected_names) | {ready_filename}:
            raise ValueError("teacher root has missing, extra or partial inventory entries")
        snapshots = []
        for index, (name, entry) in enumerate(zip(expected_names, inventory)):
            if (type(entry) is not dict or set(entry) != {"name", "sha256", "size_bytes", "volume_serial_number", "file_id"}
                    or entry["name"] != name or type(entry["size_bytes"]) is not int
                    or not 0 < entry["size_bytes"] <= artifact_max_bytes):
                raise ValueError("teacher inventory entry shape, order or size is invalid")
            identity = (entry["volume_serial_number"], entry["file_id"])
            validate_native_identity(identity)
            require_sha256(entry["sha256"], "teacher_inventory_sha")
            lease = _open_bounded_file(root.path / name, entry["sha256"], root, identity,
                                       expected_size=entry["size_bytes"])
            files.append(lease)
            payload = lease.read_bytes(max_bytes=artifact_max_bytes)
            if len(payload) != entry["size_bytes"]:
                raise ValueError("teacher inventory size differs from leased bytes")
            if index == 0:
                plan_payload = {"plan": expected.plan.portable_payload(), "plan_sha256": expected.expected_plan_sha256,
                                "root_identity": fixed["root_identity"]}
                decode_canonical_teacher_object(payload, label="teacher plan")
                if payload != canonical_json_bytes(plan_payload):
                    raise ValueError("teacher plan record differs from independently supplied plan")
            else:
                parsed = decode_teacher_diagnostic_result(payload)
                if validate_teacher_diagnostic_result(parsed, expected.plan, index - 1) != payload:
                    raise ValueError("teacher result differs from its strict typed representation")
            snapshots.append(Canonical_json_snapshot(payload, sha256_bytes(payload)))
        _revalidate_physics(expected)
        verify_directory_leases(directories)
        for lease in files:
            lease.verify_binding()
        if {path.name for path in root.path.iterdir()} != set(expected_names) | {ready_filename}:
            raise ValueError("teacher inventory changed during inspection")
        if (borrowed_parent is not None
                and _borrowed_parent_binding(borrowed_parent, expected.plan.trusted_output_parent) != parent_binding):
            raise ValueError("borrowed teacher parent binding changed during inspection")
        admission = Teacher_evidence_admission(
            _factory_token, expected=expected,
            ready=Canonical_json_snapshot(ready_bytes, expected.expected_ready_sha256),
            plan_record=snapshots[0], results=tuple(snapshots[1:]))
    except BaseException as error:
        body_error = error
        raise
    finally:
        failures = []
        for close, leases in ((close_file_leases, files), (close_directory_leases, directories)):
            try:
                close(leases)
            except BaseException as error:
                failures.append(error)
        if borrowed_parent is not None:
            try:
                if _borrowed_parent_binding(borrowed_parent, expected.plan.trusted_output_parent) != parent_binding:
                    raise ValueError("borrowed teacher parent binding changed after inspection")
            except BaseException as error:
                failures.append(error)
        if failures:
            if body_error is not None and hasattr(body_error, "add_note"):
                body_error.add_note(f"RECOVERY_REQUIRED: teacher inspection lease cleanup failed: {failures!r}")
            elif body_error is None:
                raise RuntimeError("RECOVERY_REQUIRED: teacher inspection leases did not close") from failures[0]
    return admission


def require_unchanged_teacher_evidence(
    admission: Teacher_evidence_admission, expected: Teacher_evidence_pin, *,
    expected_source: tuple[str, str], expected_plan_sha256: str,
    borrowed_parent: Directory_lease | None = None,
) -> Teacher_evidence_admission:
    if type(admission) is not Teacher_evidence_admission:
        raise ValueError("a complete teacher evidence admission is required")
    fresh = inspect_teacher_diagnostic(expected, expected_source=expected_source,
                                       expected_plan_sha256=expected_plan_sha256, borrowed_parent=borrowed_parent)
    if fresh != admission:
        raise ValueError("teacher evidence changed since the admitted content snapshot")
    return fresh


__all__ = ["Teacher_evidence_pin", "Teacher_evidence_admission", "inspect_teacher_diagnostic",
           "require_unchanged_teacher_evidence", "validate_native_identity"]
