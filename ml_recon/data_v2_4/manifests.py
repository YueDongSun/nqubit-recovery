"""Canonical build and READY manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

from .hashing import sha256_json
from .validators import validate_authority_flags
from .version import DATASET_CONTRACT_VERSION, SOURCE_CONTRACT_VERSION


@dataclass(frozen=True)
class V24BuildManifest:
    build_id: str
    contract_sha: str
    schema_sha: str
    generator_sha: str
    source_commit: str
    source_tree_sha: str
    global_seed: int
    cells: Tuple[Dict[str, Any], ...]
    generator_profile_id: str = "UNSPECIFIED"
    source_contract_version: str = SOURCE_CONTRACT_VERSION
    dataset_contract_version: str = DATASET_CONTRACT_VERSION
    research_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["cells"] = list(self.cells)
        return value

    def validate(self) -> None:
        validate_authority_flags(self.to_dict())
        if any(len(value) not in (40, 64) for value in (
                self.contract_sha, self.schema_sha, self.generator_sha,
                self.source_commit, self.source_tree_sha)):
            raise ValueError("build manifest hashes are invalid")
        if len({cell["cell_id"] for cell in self.cells}) != len(self.cells):
            raise ValueError("build manifest cell IDs must be unique")
        if not isinstance(self.generator_profile_id, str) or self.generator_profile_id in ("", "UNSPECIFIED"):
            raise ValueError("build manifest generator profile is required")

    def sha256(self) -> str:
        self.validate()
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class V24ReadyManifest:
    build_id: str
    build_manifest_sha: str
    contract_sha: str
    schema_sha: str
    index_sha: str
    split_sha: str
    shard_hashes: Dict[str, str]
    validation_report_sha: str
    research_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False
    readiness_checks: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["readiness_checks"] = list(self.readiness_checks)
        return value

    def validate(self) -> None:
        validate_authority_flags(self.to_dict())
        required = {
            "shard_hashes", "sqlite_integrity", "split_disjoint", "group_isolation",
            "truth_visibility", "physics_validators",
        }
        if set(self.readiness_checks) != required:
            raise ValueError("READY requires every V2.4 validation gate")
        hashes = (self.build_manifest_sha, self.contract_sha, self.schema_sha,
                  self.index_sha, self.split_sha, self.validation_report_sha,
                  *self.shard_hashes.values())
        if any(len(value) != 64 for value in hashes):
            raise ValueError("READY contains an invalid SHA-256 identity")

    def sha256(self) -> str:
        self.validate()
        return sha256_json(self.to_dict())
