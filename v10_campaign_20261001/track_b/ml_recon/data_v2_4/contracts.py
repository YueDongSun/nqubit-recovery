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


LEGACY_V240_CONTRACT_KIND = "NQubit Dataset Contract V2.4.0"
LEGACY_V240_DATASET_VERSION = "2.4.0"
LEGACY_V240_GENERATOR_REVISION = "data_v2_4_generator_r9_physics_closure_v2"
LEGACY_V240_DATASET_FAMILIES = tuple(
    family for family in DATASET_FAMILIES if family != "track_a_dlcz_event"
)
LEGACY_V240_TRACK_A_FIELD_PATH = "truth/track_a_dlcz_provenance"
LEGACY_V240_SCHEMA_SHA256 = "11658187dc25b4d398bb05e0bc590b2a26abf06e5195c25880d1aa990b9dcde6"
LEGACY_V240_CONTRACT_SHA256 = "eb103af6931651068a9e0d674aae198381b2421add3787f41338ac2b8c854409"
HISTORICAL_V240_R6_GENERATOR_REVISION = "data_v2_4_generator_r6"
HISTORICAL_V240_R6_CONTRACT_SHA256S = frozenset({
    "2cf29d88289ed7bf274dfd46a0377b2b2c7da10da64b5e40b572121993d14c1f",
    "f0ead456999176c421fd7022eaac4ba92f5a08fcc45c65a804aed28f6afc3aa7",
})


def _field_schema_map(fields: Tuple[FieldSpec, ...]) -> Dict[str, Dict[str, Any]]:
    return {item.field_path: asdict(item) for item in fields}


def _field_schema_sha256(fields: Tuple[FieldSpec, ...]) -> str:
    return sha256_json({
        item.field_path: item.__dict__
        for item in sorted(fields, key=lambda item: item.field_path)
    })


def _contract_payload_sha256(contract: "NQubitDatasetContractV240") -> str:
    """Hash a parsed contract without recursively invoking its validation gate."""
    return sha256_json(contract.to_dict())


def is_historical_v240_r6_contract_sha(contract_sha: object) -> bool:
    """Return whether an exact registered r6 real-static contract was supplied."""
    return type(contract_sha) is str and contract_sha in HISTORICAL_V240_R6_CONTRACT_SHA256S


def _legacy_v240_field_specs() -> Tuple[FieldSpec, ...]:
    """Return the exact field catalog frozen into V2.4.0 real-static roots."""

    result = tuple(
        item for item in all_field_specs()
        if item.field_path != LEGACY_V240_TRACK_A_FIELD_PATH
    )
    if _field_schema_sha256(result) != LEGACY_V240_SCHEMA_SHA256:
        raise RuntimeError(
            "current field catalog no longer reproduces the frozen V2.4.0 schema"
        )
    return result


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

    @classmethod
    def real_static_v240(cls) -> "NQubitDatasetContractV240":
        """Return the immutable V2.4.0 contract reserved for real observations.

        V2.4.1 adds Track A DLCZ Monte Carlo provenance and must remain a
        synthetic-only contract path.  A real-data writer therefore freezes
        this historical schema verbatim instead of adapting a current contract.
        """

        result = cls(
            contract_kind=LEGACY_V240_CONTRACT_KIND,
            source_contract_version=SOURCE_CONTRACT_VERSION,
            dataset_contract_version=LEGACY_V240_DATASET_VERSION,
            generator_revision=LEGACY_V240_GENERATOR_REVISION,
            fields=_legacy_v240_field_specs(),
            dataset_families=LEGACY_V240_DATASET_FAMILIES,
        )
        if result.sha256() != LEGACY_V240_CONTRACT_SHA256:
            raise RuntimeError(
                "current contract construction no longer reproduces frozen V2.4.0"
            )
        return result

    def validate(self) -> None:
        current_identity = (
            self.contract_kind == CONTRACT_KIND
            and self.source_contract_version == SOURCE_CONTRACT_VERSION
            and self.dataset_contract_version == DATASET_CONTRACT_VERSION
            and self.generator_revision == GENERATOR_REVISION
        )
        legacy_identity = (
            self.contract_kind == LEGACY_V240_CONTRACT_KIND
            and self.source_contract_version == SOURCE_CONTRACT_VERSION
            and self.dataset_contract_version == LEGACY_V240_DATASET_VERSION
            and self.generator_revision == LEGACY_V240_GENERATOR_REVISION
        )
        historical_r6_identity = (
            self.contract_kind == LEGACY_V240_CONTRACT_KIND
            and self.source_contract_version == SOURCE_CONTRACT_VERSION
            and self.dataset_contract_version == LEGACY_V240_DATASET_VERSION
            and self.generator_revision == HISTORICAL_V240_R6_GENERATOR_REVISION
        )
        if not (current_identity or legacy_identity or historical_r6_identity):
            raise ValueError("V2.4 contract version identity is immutable")
        if (historical_r6_identity
                and _contract_payload_sha256(self) not in HISTORICAL_V240_R6_CONTRACT_SHA256S):
            raise ValueError("V2.4 historical r6 contract is not registered")
        if not self.research_only or self.selection_evidence or self.promotion_ready \
                or self.hardware_action_authorized:
            raise ValueError("V2.4 authority must remain research-only and hardware-disabled")
        expected_families = DATASET_FAMILIES if current_identity else LEGACY_V240_DATASET_FAMILIES
        if set(self.dataset_families) != set(expected_families) or tuple(self.n_panel) != N_PANEL:
            raise ValueError("V2.4 family/N panels are incomplete")
        paths = [item.field_path for item in self.fields]
        if len(paths) != len(set(paths)):
            raise ValueError("V2.4 field paths must be unique")
        expected_fields = (
            all_field_specs() if current_identity
            else _legacy_v240_field_specs() if legacy_identity
            else None
        )
        if expected_fields is not None and _field_schema_map(self.fields) != _field_schema_map(expected_fields):
            raise ValueError("V2.4 field catalog does not match its immutable contract identity")
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
        return _contract_payload_sha256(self)

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


def _freeze_exact_contract(
    dataset_root: str | Path,
    contract: NQubitDatasetContractV240,
) -> Dict[str, str]:
    root = Path(dataset_root)
    destination = root / "CONTRACT.json"
    contract_sha = contract.sha256()
    if destination.exists():
        existing = load_contract(root)
        if existing.sha256() != contract_sha:
            raise ValueError("V2.4 contract is immutable; use a new contract version")
        return {"contract_path": str(destination), "contract_sha": contract_sha}
    file_sha = write_canonical_json(destination, contract.to_dict())
    if file_sha != contract_sha:
        raise RuntimeError("canonical contract file SHA disagrees with contract SHA")
    return {"contract_path": str(destination), "contract_sha": contract_sha}


def freeze_contract(dataset_root: str | Path) -> Dict[str, str]:
    """Freeze the current V2.4.1 synthetic contract into a new dataset root."""

    return _freeze_exact_contract(dataset_root, NQubitDatasetContractV240.default())


def freeze_real_contract(dataset_root: str | Path) -> Dict[str, str]:
    """Freeze the V2.4.0 real-static contract without promoting an old root."""

    return _freeze_exact_contract(dataset_root, NQubitDatasetContractV240.real_static_v240())


def contract_schema_sha256(contract: NQubitDatasetContractV240) -> str:
    """Return the exact schema identity carried by an immutable V2.4 contract."""

    contract.validate()
    return _field_schema_sha256(contract.fields)


def load_contract(dataset_root: str | Path) -> NQubitDatasetContractV240:
    value = read_canonical_json(Path(dataset_root) / "CONTRACT.json")
    return NQubitDatasetContractV240.from_dict(value)
