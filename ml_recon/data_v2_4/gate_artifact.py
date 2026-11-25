"""Canonical, dependency-bound authorization gate artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .hashing import read_canonical_json, sha256_json, write_canonical_json


GATE_KINDS = ("phase_1_to_7", "phase_8_pilot_negative_control")


@dataclass(frozen=True)
class V24GateArtifact:
    gate_kind: str
    source_commit: str
    source_tree: str
    contract_sha: str
    schema_sha: str
    build_program_sha: str
    test_report_sha: str
    physics_validation_sha: str
    leakage_report_sha: str
    negative_control_report_sha: str
    disk_budget_report_sha: str
    operator_identity: str
    authorization_scope: tuple[int, ...]
    creation_time: str
    expiry: Optional[str]
    gate_content_sha: str = ""

    def content_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value.pop("gate_content_sha")
        value["authorization_scope"] = list(self.authorization_scope)
        return value

    def calculated_sha(self) -> str:
        return sha256_json(self.content_dict())

    def to_dict(self) -> Dict[str, Any]:
        value = self.content_dict()
        value["gate_content_sha"] = self.gate_content_sha or self.calculated_sha()
        return value

    def validate(self, *, requested_rung: Optional[int] = None,
                 expected_kind: Optional[str] = None,
                 expected_dependencies: Optional[Dict[str, str]] = None) -> None:
        if self.gate_kind not in GATE_KINDS or (expected_kind and self.gate_kind != expected_kind):
            raise ValueError("gate kind mismatch")
        hashes = (self.source_commit, self.source_tree, self.contract_sha, self.schema_sha,
                  self.build_program_sha, self.test_report_sha, self.physics_validation_sha,
                  self.leakage_report_sha, self.negative_control_report_sha,
                  self.disk_budget_report_sha)
        if any(len(value) not in (40, 64) for value in hashes):
            raise ValueError("gate dependency hashes are invalid")
        if not self.operator_identity or self.gate_content_sha != self.calculated_sha():
            raise ValueError("gate content SHA/operator identity is invalid")
        if requested_rung is not None and requested_rung not in self.authorization_scope:
            raise ValueError("gate does not authorize the requested sample rung")
        if self.expiry is not None:
            expiry = datetime.fromisoformat(self.expiry.replace("Z", "+00:00"))
            if expiry <= datetime.now(timezone.utc):
                raise ValueError("gate artifact has expired")
        if expected_dependencies:
            mismatches = [key for key, value in expected_dependencies.items()
                          if getattr(self, key) != value]
            if mismatches:
                raise ValueError(f"gate dependency mismatch: {mismatches}")

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "V24GateArtifact":
        data = dict(value)
        data["authorization_scope"] = tuple(int(item) for item in data["authorization_scope"])
        return cls(**data)


def write_gate_artifact(path: str | Path, **kwargs: Any) -> Dict[str, str]:
    artifact = V24GateArtifact(**kwargs)
    value = artifact.to_dict()
    artifact = V24GateArtifact.from_dict(value)
    artifact.validate()
    file_sha = write_canonical_json(path, value)
    return {"path": str(path), "gate_content_sha": artifact.gate_content_sha,
            "file_sha": file_sha}


def validate_gate_artifact(path: str | Path, *, requested_rung: int,
                           expected_kind: str,
                           expected_dependencies: Optional[Dict[str, str]] = None
                           ) -> V24GateArtifact:
    source = Path(path)
    if not source.is_file():
        raise ValueError("authorization requires a canonical gate artifact file")
    artifact = V24GateArtifact.from_dict(read_canonical_json(source))
    artifact.validate(requested_rung=requested_rung, expected_kind=expected_kind,
                      expected_dependencies=expected_dependencies)
    return artifact
