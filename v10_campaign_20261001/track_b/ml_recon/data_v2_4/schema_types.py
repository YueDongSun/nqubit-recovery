"""Canonical field-schema types for V2.4 persisted stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from .enums import Authority, Identifiability, StorageTier, Visibility


@dataclass(frozen=True)
class FieldSpec:
    field_path: str
    store: str
    dtype: str
    shape_rule: str
    unit: str
    required: bool
    mask_path: str
    authority: str
    visibility: str
    identifiability: str
    storage_tier: str
    static_or_dynamic: str
    compression_policy: str
    provenance_applicability: Tuple[str, ...]
    authority_by_provenance: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.field_path or self.store not in {
                "observation", "truth", "derived", "cell_static", "provenance"}:
            raise ValueError("invalid V2.4 field path/store")
        if not self.field_path.startswith(self.store + "/"):
            raise ValueError("V2.4 field path must be rooted in its store")
        if not self.dtype or not self.shape_rule or not self.unit or not self.mask_path:
            raise ValueError("V2.4 field requires dtype/shape/unit/mask")
        Authority(self.authority)
        Visibility(self.visibility)
        Identifiability(self.identifiability)
        StorageTier(self.storage_tier)
        if self.static_or_dynamic not in ("static", "dynamic"):
            raise ValueError("V2.4 field must declare static_or_dynamic")
        if self.compression_policy not in ("gzip", "lzf", "zstd_optional", "none", "profiled"):
            raise ValueError("invalid V2.4 compression policy")
        if not self.provenance_applicability \
                or set(self.provenance_applicability) - {"real", "synthetic", "both"}:
            raise ValueError("invalid V2.4 provenance applicability")
        for provenance, authority in self.authority_by_provenance.items():
            if provenance not in ("real", "synthetic"):
                raise ValueError("invalid authority_by_provenance key")
            Authority(authority)
