"""No-overwrite pilot artifact publication under borrowed lifecycle leases."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from collections.abc import Callable, Mapping

import torch

from .artifact_directory_lease import Directory_lease, verify_directory_leases
from .artifact_file_lease import File_lease
from .artifact_io import PUBLICATION_COMMITTED, publish_new_bytes_with_state
from .contracts import canonical_json_bytes


class Learner_pilot_artifact_writer:
    """Own the model lease while borrowing the runner's root and pilot leases."""

    def __init__(self, root_lease: Directory_lease, pilot_lease: Directory_lease,
                 lifecycle_leases: list[Directory_lease],
                 file_leases: list[File_lease] | None = None) -> None:
        if pilot_lease.path != root_lease.path / "pilot":
            raise ValueError("pilot lease is outside the exact run root")
        self.root_lease = root_lease
        self.pilot_lease = pilot_lease
        self.lifecycle_leases = lifecycle_leases
        self.file_leases = file_leases if file_leases is not None else []
        self.model_lease: Directory_lease | None = None
        self.checkpoint_lease: File_lease | None = None
        self.receipt_lease: File_lease | None = None

    def verify(self) -> None:
        verify_directory_leases(self.lifecycle_leases)
        for lease in self.file_leases:
            lease.verify_binding()

    def admit_model_directory(self) -> Directory_lease:
        self.verify()
        model_path = self.pilot_lease.path / "model"
        if self.model_lease is None:
            self.model_lease = Directory_lease.open(model_path)
            self.lifecycle_leases.append(self.model_lease)
        self.verify()
        return self.model_lease

    def create_model_directory(self) -> Directory_lease:
        self.verify()
        if self.model_lease is not None:
            raise ValueError("model directory was already admitted")
        self.model_lease = Directory_lease.create_child(self.pilot_lease, "model")
        self.lifecycle_leases.append(self.model_lease)
        self.verify()
        return self.model_lease

    def _publish(self, path: Path, payload: bytes, lease: Directory_lease,
                 before_publish: Callable[[], object] | None = None) -> str:
        self.verify()
        digest = hashlib.sha256(payload).hexdigest()
        publication = publish_new_bytes_with_state(
            path, payload, expected_sha=digest, parent_lease=lease,
            before_publish=before_publish,
        )
        if publication.status != PUBLICATION_COMMITTED or not publication.staging_cleanup_complete:
            raise RuntimeError(f"RECOVERY_REQUIRED: pilot artifact publication: {publication.to_dict()}")
        if publication.file_identity is None:
            raise RuntimeError("RECOVERY_REQUIRED: publisher omitted the staged file identity")
        artifact_lease = File_lease.open_verified(
            path, digest, lease, expected_identity=publication.file_identity)
        self.file_leases.append(artifact_lease)
        if path.name == "checkpoint.pt":
            self.checkpoint_lease = artifact_lease
        elif path.name == "pilot_receipt.json":
            self.receipt_lease = artifact_lease
        self.verify()
        return digest

    def write_checkpoint(self, payload: Mapping[str, object]) -> str:
        buffer = io.BytesIO()
        torch.save(dict(payload), buffer)
        return self.write_checkpoint_bytes(buffer.getvalue())

    def write_checkpoint_bytes(self, payload: bytes, *,
                               before_publish: Callable[[], object] | None = None) -> str:
        if type(payload) is not bytes:
            raise TypeError("checkpoint payload must be bytes")
        model_lease = self.create_model_directory()
        return self._publish(model_lease.path / "checkpoint.pt", payload, model_lease,
                             before_publish)

    def write_receipt(self, payload: Mapping[str, object]) -> str:
        return self.write_receipt_bytes(canonical_json_bytes(dict(payload)))

    def write_receipt_bytes(self, payload: bytes, *,
                            before_publish: Callable[[], object] | None = None) -> str:
        if type(payload) is not bytes:
            raise TypeError("receipt payload must be bytes")
        model_lease = self.admit_model_directory()
        self.verify()
        if (self.checkpoint_lease is None
                or self.checkpoint_lease.path != model_lease.path / "checkpoint.pt"):
            raise ValueError("guarded pilot checkpoint is missing before receipt")
        self.checkpoint_lease.verify_binding()
        return self._publish(self.pilot_lease.path / "pilot_receipt.json",
                             payload, self.pilot_lease, before_publish)
