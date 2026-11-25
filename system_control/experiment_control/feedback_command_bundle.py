"""Immutable V6 command bundles with atomic activation and exact rollback.

Only this engineering-side transaction layer may call an injected AWG uploader.
It never writes ``matrix/ml_delta`` and has no checkpoint-promotion authority.
Bundle payloads stay byte-immutable; lifecycle state and the active pointer are
stored as separate atomic records.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .awg_uploader import AwgUploader, REQUIRED_SEQUENCE_CHANNELS
from .data_protocol import stable_json_hash
from .sequence_service import PreparedSequenceArtifactV1


BUNDLE_STATUSES = (
    "staged",
    "operator_approved",
    "activated_for_trial",
    "measurement_completed",
    "accepted",
    "rolled_back",
    "rejected",
)
_STATUS_TRANSITIONS = {
    "staged": {"operator_approved", "rejected"},
    "operator_approved": {"activated_for_trial", "rejected"},
    "activated_for_trial": {"measurement_completed", "rolled_back"},
    "measurement_completed": {"accepted", "rolled_back"},
    "accepted": set(),
    "rolled_back": set(),
    "rejected": set(),
}
_SEQUENCE_FILE_RE = re.compile(
    r"^(?P<channel>[WR]_CH[1-4])_T(?P<duration>\d+)us\.txt$", re.IGNORECASE,
)
_MISSING = object()


@dataclass(frozen=True)
class CommandBundleManifestV1:
    bundle_id: str
    bundle_hash: str
    parent_bundle_id: Optional[str]
    artifact_id: str
    system_type: str
    target_hash: str
    control_hash: str
    action_hashes: Mapping[str, str]
    matrix_hashes: Mapping[str, Any]
    command_table_hashes: Mapping[str, str]
    waveform_hashes: Mapping[str, Any]
    phase_cal_manifest_hash: str
    fourier_report: Mapping[str, Any]
    status: str
    created_at: str
    parent_bundle_hash: str
    upload_receipt: Optional[Mapping[str, Any]] = None
    upload_receipt_hash: Optional[str] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": "CommandBundleManifestV1",
            "bundle_id": self.bundle_id,
            "bundle_hash": self.bundle_hash,
            "parent_bundle_id": self.parent_bundle_id,
            "parent_bundle_hash": self.parent_bundle_hash,
            "artifact_id": self.artifact_id,
            "system_type": self.system_type,
            "target_hash": self.target_hash,
            "control_hash": self.control_hash,
            "action_hashes": dict(self.action_hashes),
            "matrix_hashes": dict(self.matrix_hashes),
            "command_table_hashes": dict(self.command_table_hashes),
            "waveform_hashes": dict(self.waveform_hashes),
            "phase_cal_manifest_hash": self.phase_cal_manifest_hash,
            "fourier_report": dict(self.fourier_report),
            "status": self.status,
            "created_at": self.created_at,
            "upload_receipt": None if self.upload_receipt is None else dict(self.upload_receipt),
            "upload_receipt_hash": self.upload_receipt_hash,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "CommandBundleManifestV1":
        if payload.get("schema") != "CommandBundleManifestV1":
            raise ValueError("unsupported command bundle manifest")
        return cls(
            bundle_id=str(payload["bundle_id"]),
            bundle_hash=str(payload["bundle_hash"]),
            parent_bundle_id=payload.get("parent_bundle_id"),
            artifact_id=str(payload["artifact_id"]),
            system_type=str(payload["system_type"]),
            target_hash=str(payload["target_hash"]),
            control_hash=str(payload["control_hash"]),
            action_hashes=dict(payload["action_hashes"]),
            matrix_hashes=dict(payload["matrix_hashes"]),
            command_table_hashes=dict(payload["command_table_hashes"]),
            waveform_hashes=dict(payload["waveform_hashes"]),
            phase_cal_manifest_hash=str(payload["phase_cal_manifest_hash"]),
            fourier_report=dict(payload.get("fourier_report") or {}),
            status=str(payload["status"]),
            created_at=str(payload["created_at"]),
            parent_bundle_hash=str(
                payload.get("parent_bundle_hash")
                or stable_json_hash({
                    "schema": "CommandBundleRootV1",
                    "target_hash": str(payload["target_hash"]),
                })
            ),
            upload_receipt=payload.get("upload_receipt"),
            upload_receipt_hash=payload.get("upload_receipt_hash"),
        )


class FeedbackCommandBundleStore:
    """Manage one feedback session's immutable bundle tree and active pointer."""

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir).resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.active_pointer_path = self.session_dir / "active_bundle.json"
        self.bundle_index_path = self.session_dir / "bundle_index.json"
        self.pending_transaction_path = self.session_dir / "pending_bundle_transaction.json"
        self.receipts_dir = self.session_dir / "receipts"
        self.receipts_dir.mkdir(exist_ok=True)

    def stage_prepared_artifact(
        self,
        artifact: PreparedSequenceArtifactV1,
        relative_bundle_dir: str | Path,
        *,
        parent_bundle_id: str | None = None,
    ) -> CommandBundleManifestV1:
        materialization = dict(artifact.materialization)
        if not bool(materialization.get("accepted", False)):
            raise ValueError("rejected materialization cannot be staged as a command bundle")
        if artifact.upload_receipt is not None:
            raise ValueError("prepared artifact must be pre-upload when staged")
        parent = None
        if parent_bundle_id is not None:
            parent = self.load_manifest(parent_bundle_id)
            if parent.status != "accepted":
                raise ValueError("candidate parent must be the latest accepted bundle")
            if parent.target_hash != artifact.target_hash:
                raise ValueError("candidate target hash differs from accepted parent")
            if parent.control_hash != artifact.parent_control_hash:
                raise ValueError("candidate parent control hash mismatch")
            pointer = self.active_pointer()
            if pointer is None or pointer.get("bundle_id") != parent_bundle_id:
                raise ValueError("candidate parent must be the active accepted bundle")

        destination = (self.session_dir / Path(relative_bundle_dir)).resolve()
        if self.session_dir not in destination.parents:
            raise ValueError("bundle directory must stay inside its feedback session")
        if destination.exists():
            raise FileExistsError(f"immutable bundle destination already exists: {destination}")
        identity = {
            "schema": "CommandBundleManifestV1",
            "parent_bundle_id": parent_bundle_id,
            "parent_bundle_hash": (
                parent.bundle_hash
                if parent is not None
                else stable_json_hash({
                    "schema": "CommandBundleRootV1",
                    "target_hash": artifact.target_hash,
                })
            ),
            "artifact_id": artifact.artifact_id,
            "system_type": artifact.system_type,
            "target_hash": artifact.target_hash,
            "control_hash": artifact.control_hash,
            "action_hashes": {
                "requested": artifact.requested_action_hash,
                "projected": artifact.projected_action_hash,
                "actual": artifact.actual_action_hash,
            },
            "matrix_hashes": {
                "bundle_hash": artifact.matrix_hash,
                "files": dict(artifact.matrix_files),
            },
            "command_table_hashes": {
                "table_id": artifact.command_table_id,
                "table_hash": artifact.command_table_hash,
                "registry_hash": artifact.registry_hash,
            },
            "waveform_hashes": {
                "bundle_hash": artifact.waveform_hash,
                "files": dict(artifact.waveform_files),
            },
            "phase_cal_manifest_hash": artifact.phase_cal_manifest_hash,
            "fourier_report": dict(artifact.fourier_report),
        }
        bundle_hash = stable_json_hash(identity)
        bundle_id = f"bundle_{bundle_hash[:20]}"
        manifest = CommandBundleManifestV1(
            bundle_id=bundle_id,
            bundle_hash=bundle_hash,
            parent_bundle_id=parent_bundle_id,
            artifact_id=artifact.artifact_id,
            system_type=artifact.system_type,
            target_hash=artifact.target_hash,
            control_hash=artifact.control_hash,
            action_hashes=dict(identity["action_hashes"]),
            matrix_hashes=dict(identity["matrix_hashes"]),
            command_table_hashes=dict(identity["command_table_hashes"]),
            waveform_hashes=dict(identity["waveform_hashes"]),
            phase_cal_manifest_hash=artifact.phase_cal_manifest_hash,
            fourier_report=dict(artifact.fourier_report),
            status="staged",
            created_at=self._now(),
            parent_bundle_hash=str(identity["parent_bundle_hash"]),
        )

        temporary = destination.with_name(destination.name + f"_staging_{bundle_hash[:12]}")
        if temporary.exists():
            raise FileExistsError(f"stale bundle staging directory exists: {temporary}")
        temporary.mkdir(parents=True)
        try:
            shutil.copytree(Path(artifact.matrix_workspace), temporary / "matrices")
            shutil.copytree(Path(artifact.sequence_dir), temporary / "sequence")
            shutil.copytree(Path(artifact.command_table_directory), temporary / "command_table")
            registry_source = Path(artifact.command_table_directory).parent / "operation_registry.json"
            if registry_source.is_file():
                shutil.copy2(registry_source, temporary / "operation_registry.json")
            shutil.copy2(Path(artifact.phase_cal_manifest_path), temporary / "phase_cal_manifest.json")
            self._atomic_write_json(
                temporary / "prepared_sequence_artifact.json",
                artifact.to_json_dict(),
            )
            self._atomic_write_json(temporary / "manifest.json", manifest.to_json_dict())
            self._atomic_write_json(
                temporary / "bundle_status.json",
                {
                    "schema": "CommandBundleStatusV1",
                    "bundle_id": bundle_id,
                    "status": "staged",
                    "history": [{"status": "staged", "at": manifest.created_at}],
                },
            )
            self._verify_bundle_at(temporary, manifest)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(destination)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        index = self._read_json(self.bundle_index_path, {"schema": "CommandBundleIndexV1", "bundles": {}})
        bundles = dict(index.get("bundles") or {})
        if bundle_id in bundles:
            raise ValueError(f"bundle ID already exists in session: {bundle_id}")
        bundles[bundle_id] = str(destination.relative_to(self.session_dir))
        self._atomic_write_json(
            self.bundle_index_path, {"schema": "CommandBundleIndexV1", "bundles": bundles},
        )
        return manifest

    def approve(
        self,
        bundle_id: str,
        *,
        operator: str,
        approval_note: str = "",
    ) -> CommandBundleManifestV1:
        if not str(operator).strip():
            raise ValueError("operator identity is required")
        return self._transition(
            bundle_id,
            "operator_approved",
            {"operator": str(operator), "approval_note": str(approval_note)},
        )

    def activate_for_trial(
        self,
        bundle_id: str,
        uploader: AwgUploader,
        *,
        fault_injector: Callable[[str], None] | None = None,
    ) -> CommandBundleManifestV1:
        manifest = self.load_manifest(bundle_id)
        if manifest.status != "operator_approved":
            raise ValueError("bundle must be operator_approved before trial activation")
        self.verify_bundle(bundle_id)
        current = self.active_pointer()
        previous_bundle_id = None if current is None else str(current["bundle_id"])
        if manifest.parent_bundle_id != previous_bundle_id and previous_bundle_id is not None:
            raise ValueError("candidate parent is not the currently active bundle")
        pending = {
            "schema": "PendingBundleTransactionV1",
            "kind": "activate",
            "bundle_id": bundle_id,
            "previous_bundle_id": previous_bundle_id,
            "started_at": self._now(),
        }
        self._atomic_write_json(self.pending_transaction_path, pending)
        receipt = self._upload_and_verify(manifest, uploader, purpose="activate_for_trial")
        if fault_injector is not None:
            fault_injector("before_pointer_swap")
        self._write_active_pointer(manifest, previous_bundle_id, receipt)
        if fault_injector is not None:
            fault_injector("after_pointer_swap")
        transitioned = self._transition(
            bundle_id,
            "activated_for_trial",
            {"upload_receipt": receipt, "upload_receipt_hash": receipt["receipt_hash"]},
        )
        self.pending_transaction_path.unlink(missing_ok=True)
        return transitioned

    def mark_measurement_completed(
        self, bundle_id: str, *, snapshot_hash: str,
    ) -> CommandBundleManifestV1:
        if not str(snapshot_hash):
            raise ValueError("authoritative snapshot hash is required")
        return self._transition(
            bundle_id, "measurement_completed", {"snapshot_hash": str(snapshot_hash)},
        )

    def accept(self, bundle_id: str) -> CommandBundleManifestV1:
        pointer = self.active_pointer()
        if pointer is None or pointer.get("bundle_id") != bundle_id:
            raise ValueError("only the active measured bundle can be accepted")
        return self._transition(bundle_id, "accepted", {})

    def reject(self, bundle_id: str, *, reason: str) -> CommandBundleManifestV1:
        pointer = self.active_pointer()
        if pointer is not None and pointer.get("bundle_id") == bundle_id:
            raise ValueError("an active bundle must be rolled back, not rejected")
        return self._transition(bundle_id, "rejected", {"reason": str(reason)})

    def rollback(
        self,
        bundle_id: str,
        uploader: AwgUploader,
        *,
        fault_injector: Callable[[str], None] | None = None,
    ) -> CommandBundleManifestV1:
        candidate = self.load_manifest(bundle_id)
        pointer = self.active_pointer()
        if pointer is None or pointer.get("bundle_id") != bundle_id:
            raise ValueError("rollback candidate is not the active bundle")
        if candidate.status not in {"activated_for_trial", "measurement_completed"}:
            raise ValueError("only an active trial bundle can be rolled back")
        if not candidate.parent_bundle_id:
            raise ValueError("baseline bundle has no rollback parent")
        parent = self.load_manifest(candidate.parent_bundle_id)
        if parent.status != "accepted":
            raise ValueError("rollback target must be the previous accepted bundle")
        self.verify_bundle(parent.bundle_id)
        pending = {
            "schema": "PendingBundleTransactionV1",
            "kind": "rollback",
            "bundle_id": parent.bundle_id,
            "previous_bundle_id": bundle_id,
            "rolled_back_bundle_id": bundle_id,
            "started_at": self._now(),
        }
        self._atomic_write_json(self.pending_transaction_path, pending)
        receipt = self._upload_and_verify(parent, uploader, purpose="rollback")
        if fault_injector is not None:
            fault_injector("before_pointer_swap")
        self._write_active_pointer(parent, bundle_id, receipt)
        if fault_injector is not None:
            fault_injector("after_pointer_swap")
        rolled_back = self._transition(
            bundle_id,
            "rolled_back",
            {
                "rollback_bundle_id": parent.bundle_id,
                "rollback_receipt": receipt,
                "rollback_receipt_hash": receipt["receipt_hash"],
            },
        )
        self.pending_transaction_path.unlink(missing_ok=True)
        return rolled_back

    def recover_active_bundle(
        self, uploader: AwgUploader,
    ) -> Optional[CommandBundleManifestV1]:
        pointer = self.active_pointer()
        if pointer is None:
            return None
        active = self.load_manifest(str(pointer["bundle_id"]))
        self.verify_bundle(active.bundle_id)
        receipt = self._upload_and_verify(active, uploader, purpose="crash_recovery")
        self._write_active_pointer(
            active, pointer.get("previous_bundle_id"), receipt,
        )
        pending = self._read_json(self.pending_transaction_path, None)
        if pending and pending.get("kind") == "activate" and pending.get("bundle_id") == active.bundle_id:
            if active.status == "operator_approved":
                active = self._transition(
                    active.bundle_id,
                    "activated_for_trial",
                    {
                        "upload_receipt": receipt,
                        "upload_receipt_hash": receipt["receipt_hash"],
                        "recovery_receipt": receipt,
                        "recovered_after_pointer_swap": True,
                    },
                )
        elif pending and pending.get("kind") == "rollback" and pending.get("bundle_id") == active.bundle_id:
            rolled_back_id = str(pending.get("rolled_back_bundle_id") or "")
            if rolled_back_id:
                rolled_back = self.load_manifest(rolled_back_id)
                if rolled_back.status in {"activated_for_trial", "measurement_completed"}:
                    self._transition(
                        rolled_back_id,
                        "rolled_back",
                        {
                            "rollback_bundle_id": active.bundle_id,
                            "recovery_receipt": receipt,
                            "recovered_after_pointer_swap": True,
                        },
                    )
        self.pending_transaction_path.unlink(missing_ok=True)
        return active

    def verify_bundle(self, bundle_id: str) -> CommandBundleManifestV1:
        manifest = self.load_manifest(bundle_id)
        self._verify_bundle_at(self.bundle_path(bundle_id), manifest)
        return manifest

    def load_manifest(self, bundle_id: str) -> CommandBundleManifestV1:
        path = self.bundle_path(bundle_id)
        manifest = CommandBundleManifestV1.from_json_dict(
            self._read_json(path / "manifest.json")
        )
        status_record = self._read_json(path / "bundle_status.json")
        status = str(status_record["status"])
        if status not in BUNDLE_STATUSES:
            raise ValueError(f"invalid bundle status: {status}")
        receipt = None
        receipt_hash = None
        for event in reversed(list(status_record.get("history") or [])):
            if isinstance(event.get("upload_receipt"), Mapping):
                receipt = dict(event["upload_receipt"])
                receipt_hash = event.get("upload_receipt_hash")
                break
        return replace(
            manifest,
            status=status,
            upload_receipt=receipt,
            upload_receipt_hash=None if receipt_hash is None else str(receipt_hash),
        )

    def bundle_path(self, bundle_id: str) -> Path:
        index = self._read_json(self.bundle_index_path)
        relative = dict(index.get("bundles") or {}).get(str(bundle_id))
        if relative is None:
            raise KeyError(f"unknown bundle ID: {bundle_id}")
        path = (self.session_dir / str(relative)).resolve()
        if self.session_dir not in path.parents:
            raise ValueError("bundle index escapes the session directory")
        return path

    def active_pointer(self) -> Optional[Dict[str, Any]]:
        return self._read_json(self.active_pointer_path, None)

    def _transition(
        self,
        bundle_id: str,
        next_status: str,
        details: Mapping[str, Any],
    ) -> CommandBundleManifestV1:
        manifest = self.load_manifest(bundle_id)
        if next_status not in _STATUS_TRANSITIONS.get(manifest.status, set()):
            raise ValueError(f"invalid bundle transition {manifest.status} -> {next_status}")
        path = self.bundle_path(bundle_id) / "bundle_status.json"
        record = self._read_json(path)
        history = list(record.get("history") or [])
        event = {"status": next_status, "at": self._now(), **dict(details)}
        history.append(event)
        self._atomic_write_json(
            path,
            {
                "schema": "CommandBundleStatusV1",
                "bundle_id": bundle_id,
                "status": next_status,
                "history": history,
            },
        )
        receipt = details.get("upload_receipt")
        return replace(
            manifest,
            status=next_status,
            upload_receipt=receipt if isinstance(receipt, Mapping) else manifest.upload_receipt,
            upload_receipt_hash=(
                str(details["upload_receipt_hash"])
                if details.get("upload_receipt_hash") is not None
                else manifest.upload_receipt_hash
            ),
        )

    def _upload_and_verify(
        self,
        manifest: CommandBundleManifestV1,
        uploader: AwgUploader,
        *,
        purpose: str,
    ) -> Dict[str, Any]:
        self._verify_bundle_at(self.bundle_path(manifest.bundle_id), manifest)
        sequence_dir = self.bundle_path(manifest.bundle_id) / "sequence"
        raw_receipt = uploader.upload(
            str(sequence_dir),
            {
                "system_type": manifest.system_type,
                "strict_hardware_mode": True,
                "bundle_id": manifest.bundle_id,
                "bundle_hash": manifest.bundle_hash,
                "waveform_hash": manifest.waveform_hashes["bundle_hash"],
                "purpose": purpose,
            },
        )
        if not isinstance(raw_receipt, Mapping) or not bool(raw_receipt.get("uploaded", False)):
            raise RuntimeError("AWG uploader did not return a successful upload receipt")
        observed_waveform_hash = raw_receipt.get("observed_waveform_hash")
        if (
            observed_waveform_hash is not None
            and str(observed_waveform_hash) != str(manifest.waveform_hashes["bundle_hash"])
        ):
            raise RuntimeError("AWG upload receipt waveform hash mismatch")
        self._verify_bundle_at(self.bundle_path(manifest.bundle_id), manifest)
        receipt_payload = {
            "schema": "CommandBundleUploadReceiptV1",
            "bundle_id": manifest.bundle_id,
            "bundle_hash": manifest.bundle_hash,
            "waveform_hash": manifest.waveform_hashes["bundle_hash"],
            "purpose": purpose,
            "uploaded_at": self._now(),
            "uploader_receipt": dict(raw_receipt),
            "uploader_receipt_hash": stable_json_hash(dict(raw_receipt)),
        }
        receipt_payload["receipt_hash"] = stable_json_hash(receipt_payload)
        receipt_path = self.receipts_dir / (
            f"{len(list(self.receipts_dir.glob('*.json'))) + 1:04d}_{purpose}_{manifest.bundle_id}.json"
        )
        self._atomic_write_json(receipt_path, receipt_payload)
        return receipt_payload

    def _write_active_pointer(
        self,
        manifest: CommandBundleManifestV1,
        previous_bundle_id: Any,
        receipt: Mapping[str, Any],
    ) -> None:
        pointer = {
            "schema": "ActiveCommandBundlePointerV1",
            "bundle_id": manifest.bundle_id,
            "bundle_hash": manifest.bundle_hash,
            "previous_bundle_id": previous_bundle_id,
            "target_hash": manifest.target_hash,
            "control_hash": manifest.control_hash,
            "waveform_hash": manifest.waveform_hashes["bundle_hash"],
            "upload_receipt_hash": receipt["receipt_hash"],
            "changed_at": self._now(),
        }
        self._atomic_write_json(self.active_pointer_path, pointer)

    def _verify_bundle_at(
        self, path: Path, manifest: CommandBundleManifestV1,
    ) -> None:
        immutable_payload = self._read_json(path / "manifest.json")
        immutable_manifest = CommandBundleManifestV1.from_json_dict(immutable_payload)
        if immutable_manifest.bundle_hash != manifest.bundle_hash:
            raise ValueError("immutable bundle manifest hash changed")
        identity = dict(immutable_payload)
        for name in (
            "bundle_id", "bundle_hash", "status", "created_at", "upload_receipt", "upload_receipt_hash",
        ):
            identity.pop(name, None)
        if stable_json_hash(identity) != manifest.bundle_hash:
            raise ValueError("command bundle identity hash mismatch")
        matrix_files = dict(manifest.matrix_hashes["files"])
        waveform_files = dict(manifest.waveform_hashes["files"])
        current_matrix = self._file_hashes(path / "matrices", matrix_files)
        current_waveform = self._file_hashes(path / "sequence", waveform_files)
        actual_waveform_names = {
            item.name for item in (path / "sequence").glob("*.txt") if item.is_file()
        }
        if actual_waveform_names != set(waveform_files):
            raise ValueError("immutable bundle contains unmanifested or missing waveform files")
        actual_matrix_names = {
            item.name
            for item in (path / "matrices").glob("*.txt")
            if item.is_file()
            and item.name.startswith(("W_", "R_"))
            and ("_amp" in item.name or "_phase" in item.name)
        }
        if actual_matrix_names != set(matrix_files):
            raise ValueError("immutable bundle contains unmanifested or missing command matrices")
        if current_matrix != matrix_files:
            raise ValueError("immutable bundle matrix bytes changed")
        if current_waveform != waveform_files:
            raise ValueError("immutable bundle waveform bytes changed")
        if stable_json_hash(current_matrix) != manifest.matrix_hashes["bundle_hash"]:
            raise ValueError("matrix bundle hash mismatch")
        if stable_json_hash(current_waveform) != manifest.waveform_hashes["bundle_hash"]:
            raise ValueError("waveform bundle hash mismatch")
        phase_manifest = path / "phase_cal_manifest.json"
        if self._sha256_file(phase_manifest) != manifest.phase_cal_manifest_hash:
            raise ValueError("phase calculation manifest hash mismatch")
        table_manifest = self._read_json(path / "command_table" / "command_table.json")
        if table_manifest.get("table_hash") != manifest.command_table_hashes["table_hash"]:
            raise ValueError("command table hash mismatch")
        if table_manifest.get("table_id") != manifest.command_table_hashes["table_id"]:
            raise ValueError("command table ID mismatch")
        registry_path = path / "operation_registry.json"
        if registry_path.is_file():
            registry = self._read_json(registry_path)
            if registry.get("registry_hash") != manifest.command_table_hashes["registry_hash"]:
                raise ValueError("operation registry hash mismatch")
        prepared_payload = self._read_json(path / "prepared_sequence_artifact.json")
        expected_artifact_values = {
            "artifact_id": manifest.artifact_id,
            "system_type": manifest.system_type,
            "target_hash": manifest.target_hash,
            "control_hash": manifest.control_hash,
            "requested_action_hash": manifest.action_hashes["requested"],
            "projected_action_hash": manifest.action_hashes["projected"],
            "actual_action_hash": manifest.action_hashes["actual"],
            "matrix_hash": manifest.matrix_hashes["bundle_hash"],
            "command_table_id": manifest.command_table_hashes["table_id"],
            "command_table_hash": manifest.command_table_hashes["table_hash"],
            "waveform_hash": manifest.waveform_hashes["bundle_hash"],
            "phase_cal_manifest_hash": manifest.phase_cal_manifest_hash,
        }
        for field_name, expected_value in expected_artifact_values.items():
            if str(prepared_payload.get(field_name) or "") != str(expected_value):
                raise ValueError(f"prepared sequence artifact binding changed: {field_name}")
        self._validate_complete_waveform_set(manifest.system_type, waveform_files, matrix_files)

    @staticmethod
    def _validate_complete_waveform_set(
        system_type: str,
        waveform_files: Mapping[str, str],
        matrix_files: Mapping[str, str],
    ) -> None:
        groups: Dict[str, set[str]] = {}
        for name in waveform_files:
            match = _SEQUENCE_FILE_RE.match(name)
            if match is None:
                raise ValueError(f"unexpected waveform file in immutable bundle: {name}")
            groups.setdefault(match.group("duration"), set()).add(match.group("channel").upper())
        expected = set(REQUIRED_SEQUENCE_CHANNELS)
        if len(groups) != 1 or next(iter(groups.values()), set()) != expected:
            raise ValueError("bundle does not contain one complete eight-channel waveform set")
        if system_type == "4qubit":
            required = {
                f"{channel}_{kind}_{cycle}_4qubit.txt"
                for channel in ("W_CH1", "R_CH1", "R_CH2")
                for kind in ("amp", "phase")
                for cycle in range(1, 5)
            }
            missing = sorted(required - set(matrix_files))
            if missing:
                raise ValueError(
                    "4Q immutable bundle is missing cycle matrices: " + ", ".join(missing)
                )

    @staticmethod
    def _file_hashes(root: Path, expected: Mapping[str, str]) -> Dict[str, str]:
        values: Dict[str, str] = {}
        for name in expected:
            path = root / str(name)
            if not path.is_file():
                raise FileNotFoundError(f"immutable bundle file is missing: {path}")
            values[str(name)] = FeedbackCommandBundleStore._sha256_file(path)
        return values

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _read_json(path: Path, default: Any = _MISSING) -> Any:
        if not path.is_file():
            if default is _MISSING:
                raise FileNotFoundError(path)
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


__all__ = [
    "BUNDLE_STATUSES",
    "CommandBundleManifestV1",
    "FeedbackCommandBundleStore",
]
