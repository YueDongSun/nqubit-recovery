"""Track B transition-dataset declaration with no fabricated rows.

Real-static observations are not transitions. The current bridge can create
only the explicit empty manifest; recording a before/action/after row remains
disabled until an independently authorized acquisition transaction exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from .proposal_reader import ProposalValidationError, canonical_json_bytes, sha256_bytes


@dataclass(frozen=True)
class TrackBTransitionManifestV1:
    """The initial real-feedback transition manifest required by the Track B ABI."""

    dataset_family: str = "real_feedback_transition"
    row_count: int = 0
    status: str = "NOT_AVAILABLE"
    real_static_rows_converted: bool = False
    actuator_transfer_update_allowed: bool = False

    def __post_init__(self) -> None:
        if self.dataset_family != "real_feedback_transition":
            raise ProposalValidationError("transition dataset family is fixed")
        if self.row_count != 0 or self.status != "NOT_AVAILABLE":
            raise ProposalValidationError("initial transition manifest must remain empty and unavailable")
        if self.real_static_rows_converted or self.actuator_transfer_update_allowed:
            raise ProposalValidationError("real_static rows cannot become Track B transitions")

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "schema": "TrackBTransitionManifestV1",
            "dataset_family": self.dataset_family,
            "row_count": 0,
            "status": "NOT_AVAILABLE",
            "real_static_rows_converted": False,
            "actuator_transfer_update_allowed": False,
        }

    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_json_dict()))


class TrackBTransitionLogger:
    """Own the empty-manifest file only; no row writer is enabled in this branch."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "transition_manifest_empty.json"

    def initialize_empty_manifest(self) -> TrackBTransitionManifestV1:
        """Create exactly one canonical empty manifest, or verify the existing one."""

        manifest = TrackBTransitionManifestV1()
        encoded = canonical_json_bytes(manifest.to_json_dict())
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            with self.manifest_path.open("xb") as handle:
                handle.write(encoded)
        except FileExistsError:
            raw = self.manifest_path.read_bytes()
            if raw != encoded:
                raise ProposalValidationError("transition manifest is not the canonical empty manifest")
        return manifest

    def record_transition(self, transition: Mapping[str, Any]) -> None:
        """Fail closed: a live transition cannot be fabricated from static data."""

        del transition
        raise PermissionError(
            "Track B transition logging is NOT_AVAILABLE until a separately authorized external trial exists"
        )


__all__ = ["TrackBTransitionLogger", "TrackBTransitionManifestV1"]
