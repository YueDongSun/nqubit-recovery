"""Deployable V10 feedback envelope with an optional, separately versioned memory view."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from ml_recon.physics_hybrid_v10.contracts import sha256_of

from .feedback_input import V10FeedbackDecisionInputV2
from .memory_fusion import Memory_fusion_extension_manifest, V10MemoryFusionContextV1
from .shared_abi import Semantic_abi_manifest_v2


FEEDBACK_DECISION_ENVELOPE_SCHEMA = "V10FeedbackDecisionEnvelopeV1"
MEMORY_POLICY_DISABLED = "MEMORY_POLICY_DISABLED_LEGACY_CORE_EMPTY"
MEMORY_POLICY_NOT_AVAILABLE = "MEMORY_POLICY_NOT_AVAILABLE"


def _sha(value: object, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _extension_manifest_sha(extension: Memory_fusion_extension_manifest) -> str:
    extension.validate()
    return sha256_of(asdict(extension))


@dataclass(frozen=True)
class V10FeedbackDecisionEnvelopeV1:
    """Keep V2 core bytes/semantics separate from the R17 memory extension.

    The current Track A policy does not implement a memory-aware fusion map.
    It may therefore validate this envelope, but must fail closed for a present
    memory extension instead of silently treating it as an unused side channel.
    """

    core: V10FeedbackDecisionInputV2
    memory_extension: V10MemoryFusionContextV1 | None
    extension_manifest_sha: str | None
    base_abi: Semantic_abi_manifest_v2
    memory_policy_status: Literal[
        "MEMORY_POLICY_DISABLED_LEGACY_CORE_EMPTY", "MEMORY_POLICY_NOT_AVAILABLE"
    ] = MEMORY_POLICY_NOT_AVAILABLE
    schema: str = FEEDBACK_DECISION_ENVELOPE_SCHEMA

    def validate(self) -> None:
        if self.schema != FEEDBACK_DECISION_ENVELOPE_SCHEMA:
            raise ValueError("unsupported feedback decision envelope schema")
        if type(self.core) is not V10FeedbackDecisionInputV2:
            raise ValueError(
                "envelope requires an exact V10FeedbackDecisionInputV2 core"
            )
        if type(self.base_abi) is not Semantic_abi_manifest_v2:
            raise ValueError(
                "envelope requires an externally pinned semantic ABI manifest"
            )
        self.base_abi.validate()
        self.core.validate()
        if (
            self.core.success_fusion
            or self.core.success_fusion_mask
            or self.core.success_fusion_provenance_sha is not None
        ):
            raise ValueError(
                "legacy core success/fusion data is forbidden in the deployable envelope"
            )
        if self.memory_extension is None:
            if self.extension_manifest_sha is not None:
                raise ValueError(
                    "disabled memory extension cannot claim a manifest SHA"
                )
            if self.memory_policy_status != MEMORY_POLICY_DISABLED:
                raise ValueError(
                    "empty memory data must be explicitly labeled disabled"
                )
            return
        if type(self.memory_extension) is not V10MemoryFusionContextV1:
            raise ValueError("memory extension requires V10MemoryFusionContextV1")
        self.memory_extension.validate()
        if self.memory_extension.base_abi != self.base_abi:
            raise ValueError(
                "memory extension must equal the externally pinned semantic ABI"
            )
        _sha(self.extension_manifest_sha, "extension_manifest_sha")
        if self.extension_manifest_sha != _extension_manifest_sha(
            self.memory_extension.extension
        ):
            raise ValueError("envelope manifest SHA does not bind the memory extension")
        core_budget = self.core.budget
        if core_budget.physical_budget_status == "ACTUAL_REPEAT_UNBOUND":
            raise ValueError(
                "unbound core budget cannot enter a memory decision envelope"
            )
        if core_budget.n_mode != len(self.core.belief.valid_mode_mask):
            raise ValueError("core belief and budget mode dimensions disagree")
        if any(
            resource.input_n_modes != core_budget.n_mode
            for resource in self.memory_extension.resources
        ):
            raise ValueError("memory resources must bind the exact core mode dimension")
        if self.memory_policy_status != MEMORY_POLICY_NOT_AVAILABLE:
            raise ValueError(
                "memory extension requires an explicit unavailable policy status"
            )

    def sha256(self) -> str:
        self.validate()
        return sha256_of(asdict(self))


def adapt_legacy_core_for_deployable_policy(
    core: V10FeedbackDecisionInputV2,
    base_abi: Semantic_abi_manifest_v2,
) -> V10FeedbackDecisionEnvelopeV1:
    """Make the legacy path explicit; only a wholly empty fusion payload is legal."""
    result = V10FeedbackDecisionEnvelopeV1(
        core=core,
        memory_extension=None,
        extension_manifest_sha=None,
        base_abi=base_abi,
        memory_policy_status=MEMORY_POLICY_DISABLED,
    )
    result.validate()
    return result


__all__ = [
    "FEEDBACK_DECISION_ENVELOPE_SCHEMA",
    "MEMORY_POLICY_DISABLED",
    "MEMORY_POLICY_NOT_AVAILABLE",
    "V10FeedbackDecisionEnvelopeV1",
    "adapt_legacy_core_for_deployable_policy",
]
