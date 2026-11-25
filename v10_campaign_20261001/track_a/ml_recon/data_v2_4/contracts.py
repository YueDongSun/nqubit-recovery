"""Canonical V2.4 contract and immutable freeze/load operations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

from .enums import DATASET_FAMILIES, N_PANEL
from .field_catalog import all_field_specs
from .hashing import read_canonical_json, sha256_json, write_canonical_json
from .schema_types import FieldSpec
from .version import (
    CONTRACT_KIND,
    DATASET_CONTRACT_VERSION,
    GENERATOR_REVISION,
    SOURCE_CONTRACT_VERSION,
)


@dataclass(frozen=True)
class NQubitDatasetContractV240:
    contract_kind: str = CONTRACT_KIND
    source_contract_version: str = SOURCE_CONTRACT_VERSION
    dataset_contract_version: str = DATASET_CONTRACT_VERSION
    generator_revision: str = GENERATOR_REVISION
    fields: Tuple[FieldSpec, ...] = field(default_factory=tuple)
    dataset_families: Tuple[str, ...] = DATASET_FAMILIES
    n_panel: Tuple[int, ...] = N_PANEL
    research_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False

    @classmethod
    def default(cls) -> "NQubitDatasetContractV240":
        return cls(fields=all_field_specs())

    def validate(self) -> None:
        if self.contract_kind != CONTRACT_KIND \
                or self.source_contract_version != SOURCE_CONTRACT_VERSION \
                or self.dataset_contract_version != DATASET_CONTRACT_VERSION \
                or self.generator_revision != GENERATOR_REVISION:
            raise ValueError("V2.4 contract version identity is immutable")
        if not self.research_only or self.selection_evidence or self.promotion_ready \
                or self.hardware_action_authorized:
            raise ValueError("V2.4 authority must remain research-only and hardware-disabled")
        if set(self.dataset_families) != set(DATASET_FAMILIES) or tuple(self.n_panel) != N_PANEL:
            raise ValueError("V2.4 family/N panels are incomplete")
        paths = [item.field_path for item in self.fields]
        if len(paths) != len(set(paths)):
            raise ValueError("V2.4 field paths must be unique")
        for item in self.fields:
            item.validate()

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["fields"] = [asdict(item) for item in self.fields]
        value["dataset_families"] = list(self.dataset_families)
        value["n_panel"] = list(self.n_panel)
        return value

    def sha256(self) -> str:
        self.validate()
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "NQubitDatasetContractV240":
        data = dict(value)
        for item in data["fields"]:
            item["provenance_applicability"] = tuple(item["provenance_applicability"])
        data["fields"] = tuple(FieldSpec(**item) for item in data["fields"])
        data["dataset_families"] = tuple(data["dataset_families"])
        data["n_panel"] = tuple(data["n_panel"])
        result = cls(**data)
        result.validate()
        return result


def freeze_contract(dataset_root: str | Path) -> Dict[str, str]:
    root = Path(dataset_root)
    destination = root / "CONTRACT.json"
    contract = NQubitDatasetContractV240.default()
    contract_sha = contract.sha256()
    if destination.exists():
        existing = load_contract(root)
        if existing.sha256() != contract_sha:
            raise ValueError("V2.4 contract is immutable; use a new contract version")
    file_sha = write_canonical_json(destination, contract.to_dict())
    if file_sha != contract_sha:
        raise RuntimeError("canonical contract file SHA disagrees with contract SHA")
    return {"contract_path": str(destination), "contract_sha": contract_sha}


def load_contract(dataset_root: str | Path) -> NQubitDatasetContractV240:
    value = read_canonical_json(Path(dataset_root) / "CONTRACT.json")
    return NQubitDatasetContractV240.from_dict(value)
