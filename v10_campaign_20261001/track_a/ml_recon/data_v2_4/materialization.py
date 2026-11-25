"""Versioned V2.4 storage materialization profiles.

Profiles change retention and placement only. They never change field physics,
authority, visibility, identifiability, or sample identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Tuple

from .hashing import sha256_json, write_canonical_json
from .version import MATERIALIZATION_PROFILE_VERSION


class MaterializationKind(str, Enum):
    PERSISTENT_FULL = "PERSISTENT_FULL"
    PERSISTENT_EVAL_EPHEMERAL_TRAIN = "PERSISTENT_EVAL_EPHEMERAL_TRAIN"
    CELL_LOCAL_PERSISTENT = "CELL_LOCAL_PERSISTENT"


@dataclass(frozen=True)
class V24MaterializationProfileV1:
    profile_kind: str
    persistent_splits: Tuple[str, ...]
    ephemeral_splits: Tuple[str, ...]
    target_shard_mib: int = 384
    shard_window_count: int = 4
    minimum_rows_per_shard: int = 128
    maximum_rows_per_shard: int = 65_536
    minimum_reserve_bytes: int = 20 * 1000**3
    checkpoint_reserve_bytes: int = 10 * 1000**3
    filesystem_reserve_fraction: float = 0.20
    profile_version: str = MATERIALIZATION_PROFILE_VERSION
    field_semantics_unchanged: bool = True

    @classmethod
    def persistent_full(cls) -> "V24MaterializationProfileV1":
        return cls(
            MaterializationKind.PERSISTENT_FULL.value,
            ("train", "validation", "test", "locked_ood", "negative_control", "audit"),
            (),
        )

    @classmethod
    def persistent_eval_ephemeral_train(cls) -> "V24MaterializationProfileV1":
        return cls(
            MaterializationKind.PERSISTENT_EVAL_EPHEMERAL_TRAIN.value,
            ("validation", "test", "locked_ood", "negative_control", "audit", "real"),
            ("train",),
        )

    @classmethod
    def cell_local_persistent(cls) -> "V24MaterializationProfileV1":
        return cls(
            MaterializationKind.CELL_LOCAL_PERSISTENT.value,
            ("train", "validation", "test", "locked_ood", "negative_control", "audit"),
            (),
        )

    def validate(self) -> None:
        MaterializationKind(self.profile_kind)
        if self.profile_version != MATERIALIZATION_PROFILE_VERSION \
                or not self.field_semantics_unchanged:
            raise ValueError("V2.4 materialization profile identity/semantics changed")
        if set(self.persistent_splits) & set(self.ephemeral_splits):
            raise ValueError("a split cannot be both persistent and ephemeral")
        if not 256 <= self.target_shard_mib <= 512 or not 3 <= self.shard_window_count <= 6:
            raise ValueError("V2.4 shard target/window must stay in the registered range")
        if not 1 <= self.minimum_rows_per_shard <= self.maximum_rows_per_shard:
            raise ValueError("invalid V2.4 shard row bounds")
        if not 0.0 < self.filesystem_reserve_fraction < 1.0 \
                or self.minimum_reserve_bytes < 20 * 1000**3:
            raise ValueError("invalid V2.4 disk reserve")
        if self.checkpoint_reserve_bytes < 0:
            raise ValueError("checkpoint reserve cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["persistent_splits"] = list(self.persistent_splits)
        value["ephemeral_splits"] = list(self.ephemeral_splits)
        return value

    def sha256(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "V24MaterializationProfileV1":
        payload = dict(value)
        payload["persistent_splits"] = tuple(payload["persistent_splits"])
        payload["ephemeral_splits"] = tuple(payload["ephemeral_splits"])
        profile = cls(**payload)
        profile.validate()
        return profile

    def reserve_bytes(self, filesystem_capacity_bytes: int) -> int:
        self.validate()
        target_shard_bytes = self.target_shard_mib * 1024**2
        return max(
            self.minimum_reserve_bytes,
            int(filesystem_capacity_bytes * self.filesystem_reserve_fraction),
            2 * target_shard_bytes,
            self.checkpoint_reserve_bytes,
        )


def write_materialization_profile(path: str | Path,
                                  profile: V24MaterializationProfileV1) -> Dict[str, str]:
    profile_sha = profile.sha256()
    file_sha = write_canonical_json(path, profile.to_dict())
    if file_sha != profile_sha:
        raise RuntimeError("materialization profile canonical SHA mismatch")
    return {"path": str(path), "materialization_profile_sha": profile_sha}
