"""One native owner for a fresh behavior-stage tree; no reuse or recovery writes.

This composes the existing handle-owned, no-replace publisher. A caller supplies
its already-admitted source/evidence check; ownership does not grant scientific
or execution authority. Partial roots are preserved on every failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import stat
from typing import Callable

from ml_recon.data_v2_4.hashing import sha256_bytes

from .artifact_directory_lease import Directory_lease, close_directory_leases
from .artifact_file_lease import File_lease, close_file_leases
from .artifact_io import PUBLICATION_COMMITTED, _add_cleanup_note, publish_new_bytes_with_state


maximum_artifact_bytes = 512 * 1024 * 1024
reparse_attribute = 0x400


def _relative_parts(relative: str) -> tuple[str, ...]:
    if (type(relative) is not str or not relative or "\\" in relative or ":" in relative
            or "\0" in relative or PurePosixPath(relative).is_absolute()
            or PurePosixPath(relative).as_posix() != relative):
        raise ValueError("behavior artifact path must be canonical and relative")
    parts = PurePosixPath(relative).parts
    if not parts or any(part in (".", "..") or part.endswith((".", " ")) for part in parts):
        raise ValueError("behavior artifact path contains an aliased component")
    return parts


@dataclass(frozen=True)
class Behavior_artifact_binding:
    relative_path: str
    sha256: str
    size_bytes: int
    volume_serial_number: int
    file_id: str

    def to_dict(self) -> dict[str, object]:
        return {"relative_path": self.relative_path, "sha256": self.sha256,
                "size_bytes": self.size_bytes, "volume_serial_number": self.volume_serial_number,
                "file_id": self.file_id}


@dataclass
class Behavior_artifact_owner:
    """Live single-writer capability borrowed by phase helpers in one process."""

    root: Directory_lease
    check_inputs: Callable[[], None]
    parent_lease: Directory_lease
    directories: dict[str, Directory_lease] = field(default_factory=dict)
    files: dict[str, File_lease] = field(default_factory=dict)
    bindings: dict[str, Behavior_artifact_binding] = field(default_factory=dict)
    _closed: bool = False
    _failed: bool = False

    @classmethod
    def create_child(cls, parent: Directory_lease, name: str, *,
                     check_inputs: Callable[[], None]) -> "Behavior_artifact_owner":
        if len(_relative_parts(name)) != 1:
            raise ValueError("behavior stage must be one fresh child")
        check_inputs()
        root = Directory_lease.create_child(parent, name)
        owner = cls(root, check_inputs, parent, directories={"": root})
        try:
            owner.verify()
            return owner
        except BaseException as error:
            owner._failed = True
            try:
                owner.close()
            except BaseException as cleanup_error:
                _add_cleanup_note(error, "behavior owner creation cleanup", cleanup_error)
            raise

    def verify(self, *, staged_relative: str | None = None) -> None:
        if self._closed or self._failed:
            raise RuntimeError("RECOVERY_REQUIRED: behavior owner is closed or failed")
        if (type(self.parent_lease) is not Directory_lease
                or self.parent_lease.path != self.root.path.parent):
            raise ValueError("behavior owner must retain its exact borrowed parent lease")
        self.parent_lease.verify()
        if set(self.files) != set(self.bindings) or self.directories.get("") is not self.root:
            raise ValueError("behavior owner inventory is inconsistent")
        self.check_inputs()
        for relative, directory in self.directories.items():
            expected_path = self.root.path if relative == "" else self.root.path.joinpath(*_relative_parts(relative))
            if directory.path != expected_path:
                raise ValueError("behavior directory lease differs from its inventory path")
            directory.verify()
        for relative, lease in self.files.items():
            binding = self.bindings[relative]
            if (binding.relative_path != relative or type(binding.size_bytes) is not int
                    or not 0 < binding.size_bytes <= maximum_artifact_bytes
                    or lease.path != self.root.path.joinpath(*_relative_parts(relative))
                    or lease.expected_sha != binding.sha256
                    or (lease.volume_serial_number, lease.file_id) !=
                    (binding.volume_serial_number, binding.file_id)):
                raise ValueError("behavior artifact lease differs from publisher identity")
            if len(lease.read_bytes(max_bytes=maximum_artifact_bytes)) != binding.size_bytes:
                raise ValueError("behavior artifact size differs from publisher binding")
        staged_parts = _relative_parts(staged_relative) if staged_relative is not None else None
        for relative, directory in self.directories.items():
            expected: dict[str, bool] = {}
            for candidate in self.directories:
                parent = PurePosixPath(candidate).parent.as_posix()
                if candidate and ("" if parent == "." else parent) == relative:
                    expected[PurePosixPath(candidate).name] = True
            for candidate in self.files:
                parent = PurePosixPath(candidate).parent.as_posix()
                if ("" if parent == "." else parent) == relative:
                    expected[PurePosixPath(candidate).name] = False
            actual = tuple(directory.path.iterdir())
            extra = [path for path in actual if path.name not in expected]
            staging_parent = "/".join(staged_parts[:-1]) if staged_parts is not None else None
            if staged_parts is not None and relative == staging_parent:
                if len(extra) != 1 or not extra[0].name.startswith(f".{staged_parts[-1]}.staging-"):
                    raise ValueError("unexpected behavior publication staging inventory")
            elif extra:
                raise ValueError("behavior tree contains unowned entries")
            if len(actual) != len(expected) + len(extra):
                raise ValueError("behavior inventory has missing entries")
            for path in actual:
                metadata = path.lstat()
                if getattr(metadata, "st_file_attributes", 0) & reparse_attribute:
                    raise ValueError("behavior tree contains a reparse point")
                wants_directory = expected.get(path.name, False)
                if not (stat.S_ISDIR(metadata.st_mode) if wants_directory else stat.S_ISREG(metadata.st_mode)):
                    raise ValueError("behavior artifact kind changed")

    def create_directory(self, relative: str, *, before_create: Callable[[], None] | None = None) -> Directory_lease:
        parts = _relative_parts(relative)
        try:
            self.verify()
            for index in range(1, len(parts) + 1):
                key = "/".join(parts[:index])
                if key not in self.directories:
                    parent_key = "/".join(parts[:index - 1])
                    self.check_inputs()
                    if before_create is not None:
                        before_create()
                    self.directories[key] = Directory_lease.create_child(self.directories[parent_key], parts[index - 1])
            self.verify()
            return self.directories[relative]
        except BaseException:
            self._failed = True
            raise

    def publish(self, relative: str, payload: bytes, *,
                before_publish: Callable[[], None] | None = None) -> Behavior_artifact_binding:
        parts = _relative_parts(relative)
        if type(payload) is not bytes or not payload or len(payload) > maximum_artifact_bytes:
            raise ValueError("behavior publication requires bounded nonempty bytes")
        if relative in self.files or relative in self.directories:
            raise ValueError("behavior artifact already exists; replacement is prohibited")
        parent_key = "/".join(parts[:-1])
        if parent_key not in self.directories:
            raise ValueError("behavior artifact parent must be explicitly owned before publication")
        parent = self.directories[parent_key]
        digest = sha256_bytes(payload)
        try:
            self.verify()
            if before_publish is not None:
                before_publish()

            def check_publication() -> None:
                self.verify(staged_relative=relative)
                if before_publish is not None:
                    before_publish()

            publication = publish_new_bytes_with_state(
                parent.path / parts[-1], payload, expected_sha=digest, parent_lease=parent,
                before_publish=check_publication)
            if (publication.status != PUBLICATION_COMMITTED or not publication.staging_cleanup_complete
                    or publication.file_identity is None or publication.artifact_sha256 != digest
                    or publication.target_path != str(parent.path / parts[-1])):
                raise RuntimeError("RECOVERY_REQUIRED: behavior artifact publication is uncertain")
            lease = File_lease.open_verified(
                parent.path / parts[-1], digest, parent, expected_identity=publication.file_identity,
                allow_readonly_consumers=True, max_bytes=len(payload))
            self.files[relative] = lease
            binding = Behavior_artifact_binding(relative, digest, len(payload), *publication.file_identity)
            self.bindings[relative] = binding
            self.verify()
            return binding
        except BaseException:
            self._failed = True
            raise

    def read(self, relative: str) -> bytes:
        self.verify()
        if relative not in self.files:
            raise ValueError("behavior artifact is not owned")
        payload = self.files[relative].read_bytes(max_bytes=maximum_artifact_bytes)
        if len(payload) != self.bindings[relative].size_bytes:
            raise ValueError("behavior artifact size changed")
        return payload

    def inventory(self) -> tuple[Behavior_artifact_binding, ...]:
        self.verify()
        return tuple(self.bindings[name] for name in sorted(self.bindings))

    def close(self) -> None:
        """Release owned handles, never the caller's borrowed parent lease."""
        if self._closed:
            return
        errors = []
        for closer, leases in ((close_file_leases, list(self.files.values())),
                               (close_directory_leases, list(self.directories.values()))):
            try:
                closer(leases)
            except BaseException as error:
                errors.append(error)
        if errors:
            self._failed = True
            raise RuntimeError("RECOVERY_REQUIRED: behavior owner cleanup failed") from errors[0]
        self._closed = True


__all__ = ["Behavior_artifact_binding", "Behavior_artifact_owner"]
