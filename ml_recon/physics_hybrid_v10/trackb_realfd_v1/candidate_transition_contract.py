"""Typed source binding for synthetic Track B candidate-transition labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import math

from .contracts import canonical_sha256


MATERIAL_HARM_TOLERANCE = 1e-10


@dataclass(frozen=True)
class TrackBCandidateTransitionLabelV1:
    """One action-specific synthetic transition label; latent state stays private."""

    system_type: str
    split: str
    v24_contract_sha: str
    v24_schema_sha: str
    v24_generator_sha: str
    forward_source_commit: str
    forward_source_tree: str
    actor_checkpoint_sha: str
    ancestor_trajectory_id: str
    round_index: int
    observation_sha: str
    pre_state_sha: str
    candidate_action_sha: str
    after_state_sha: str
    gain: float
    harm: bool
    valid_mask: bool
    research_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False
    schema: str = "TrackBCandidateTransitionLabelV1"

    def validate(self) -> None:
        if self.system_type not in ("3qubit", "4qubit") or self.split not in ("train", "validation", "test"):
            raise ValueError("candidate transition system/split is invalid")
        if type(self.ancestor_trajectory_id) is not str or not self.ancestor_trajectory_id:
            raise ValueError("candidate transition ancestor identity is invalid")
        if type(self.round_index) is not int or not 0 <= self.round_index < 10:
            raise ValueError("candidate transition round index is invalid")
        for name in (
            "v24_contract_sha",
            "v24_schema_sha",
            "v24_generator_sha",
            "actor_checkpoint_sha",
            "observation_sha",
            "pre_state_sha",
            "candidate_action_sha",
            "after_state_sha",
        ):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64:
                raise ValueError(f"candidate transition {name} is invalid")
        for name in ("forward_source_commit", "forward_source_tree"):
            value = getattr(self, name)
            if type(value) is not str or len(value) not in (40, 64):
                raise ValueError(f"candidate transition {name} is invalid")
        if not math.isfinite(float(self.gain)):
            raise ValueError("candidate transition gain is non-finite")
        if type(self.harm) is not bool or type(self.valid_mask) is not bool:
            raise TypeError("candidate transition harm and valid_mask must be strict booleans")
        if self.valid_mask and self.harm != (self.gain < -MATERIAL_HARM_TOLERANCE):
            raise ValueError("candidate transition harm label differs from its action-specific gain")
        if (not self.research_only or self.selection_evidence or self.promotion_ready
                or self.hardware_action_authorized):
            raise ValueError("candidate transition labels cannot authorize selection, promotion, or hardware")
        if self.schema != "TrackBCandidateTransitionLabelV1":
            raise ValueError("unsupported candidate transition label schema")

    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


__all__ = [
    "MATERIAL_HARM_TOLERANCE",
    "TrackBCandidateTransitionLabelV1",
]
