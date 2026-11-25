"""Versioned ideal-action providers with fail-closed Track A fallback handling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .contracts import canonical_sha256, require_git_oid, require_sha256
from .coordinate_abi import IdealLogicalActionV1


PROVIDER_TYPES = frozenset({
    "ANALYTIC_V24", "FROZEN_V24", "TRACK_A_FROZEN_V10", "TRACKB_LOCAL_ANALYTIC",
})


def _sha256_file(path: str | Path) -> str:
    """Hash a file without parsing or importing its model payload."""
    value = Path(path)
    if not value.is_file():
        raise ValueError("file-backed action prior admission requires an existing regular file")
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ActionPriorCompatibilityContextV1:
    """Complete Track A ABI/identity context, never a two-field compatibility check."""

    system_type: str
    source_commit: str
    source_tree: str
    config_sha: str
    normalization_sha: str
    semantic_abi_sha: str
    operation_registry_sha: str
    projector_descriptor_sha: str
    coordinate_layout_sha: str
    target_schema_sha: str
    action_bounds_sha: str
    checkpoint_sha: str
    state_dict_sha: str
    model_build_spec_sha: str

    def validate(self) -> None:
        if self.system_type not in ("3qubit", "4qubit"):
            raise ValueError("Track A compatibility context has an unsupported system type")
        require_git_oid(self.source_commit, "source_commit")
        require_git_oid(self.source_tree, "source_tree")
        for name in (
            "config_sha", "normalization_sha", "semantic_abi_sha", "operation_registry_sha",
            "projector_descriptor_sha", "coordinate_layout_sha", "target_schema_sha",
            "action_bounds_sha", "checkpoint_sha", "state_dict_sha", "model_build_spec_sha",
        ):
            require_sha256(getattr(self, name), name)

    def matches(self, bundle: "ActionPriorBundleV1") -> bool:
        """Compare every identity that defines Track A action compatibility."""
        try:
            self.validate()
            bundle.validate()
        except (TypeError, ValueError):
            return False
        return (
            bundle.action.system_type == self.system_type
            and bundle.source_commit == self.source_commit
            and bundle.source_tree == self.source_tree
            and bundle.config_sha == self.config_sha
            and bundle.normalization_sha == self.normalization_sha
            and bundle.semantic_abi_sha == self.semantic_abi_sha
            and bundle.operation_registry_sha == self.operation_registry_sha
            and bundle.projector_descriptor_sha == self.projector_descriptor_sha
            and bundle.coordinate_layout_sha == self.coordinate_layout_sha
            and bundle.target_schema_sha == self.target_schema_sha
            and bundle.action_bounds_sha == self.action_bounds_sha
            and bundle.checkpoint_sha == self.checkpoint_sha
            and bundle.state_dict_sha == self.state_dict_sha
            and bundle.model_build_spec_sha == self.model_build_spec_sha
        )


@dataclass(frozen=True)
class ActionPriorFileBackedAdmissionV1:
    """Raw-byte rehash receipt for a frozen Track A action provider."""

    checkpoint_sha: str
    state_dict_sha: str
    model_build_spec_sha: str
    config_sha: str
    normalization_sha: str
    schema: str = "TrackBActionPriorFileBackedAdmissionV1"

    def validate(self) -> None:
        if self.schema != "TrackBActionPriorFileBackedAdmissionV1":
            raise ValueError("unknown action-prior file admission schema")
        for name in (
            "checkpoint_sha", "state_dict_sha", "model_build_spec_sha", "config_sha", "normalization_sha",
        ):
            require_sha256(getattr(self, name), name)

    @classmethod
    def rehash_files(
        cls,
        *,
        checkpoint_path: str | Path,
        state_dict_path: str | Path,
        model_build_spec_path: str | Path,
        config_path: str | Path,
        normalization_path: str | Path,
    ) -> "ActionPriorFileBackedAdmissionV1":
        """Create a portable receipt without serializing the local paths."""
        return cls(
            checkpoint_sha=_sha256_file(checkpoint_path),
            state_dict_sha=_sha256_file(state_dict_path),
            model_build_spec_sha=_sha256_file(model_build_spec_path),
            config_sha=_sha256_file(config_path),
            normalization_sha=_sha256_file(normalization_path),
        )

    def matches(self, bundle: "ActionPriorBundleV1") -> bool:
        try:
            self.validate()
            bundle.validate()
        except (TypeError, ValueError):
            return False
        return (
            bundle.checkpoint_sha == self.checkpoint_sha
            and bundle.state_dict_sha == self.state_dict_sha
            and bundle.model_build_spec_sha == self.model_build_spec_sha
            and bundle.config_sha == self.config_sha
            and bundle.normalization_sha == self.normalization_sha
        )


@dataclass(frozen=True)
class ActionPriorBundleV1:
    """An exact ABI/source-bound action provider bundle, never a random checkpoint alias."""

    provider_type: str
    source_commit: str
    source_tree: str
    checkpoint_sha: str | None
    state_dict_sha: str | None
    model_build_spec_sha: str | None
    config_sha: str
    normalization_sha: str
    semantic_abi_sha: str
    operation_registry_sha: str
    projector_descriptor_sha: str
    coordinate_layout_sha: str
    target_schema_sha: str
    action_bounds_sha: str
    action: IdealLogicalActionV1
    frozen_checkpoint_verified: bool = False
    random_weights: bool = False

    def validate(self) -> None:
        if self.provider_type not in PROVIDER_TYPES:
            raise ValueError("unknown ideal action provider type")
        require_git_oid(self.source_commit, "source_commit")
        require_git_oid(self.source_tree, "source_tree")
        for name in (
            "config_sha", "normalization_sha", "semantic_abi_sha", "operation_registry_sha",
            "projector_descriptor_sha", "coordinate_layout_sha", "target_schema_sha",
            "action_bounds_sha",
        ):
            require_sha256(getattr(self, name), name)
        self.action.__post_init__()
        if self.action.provider_type != self.provider_type:
            raise ValueError("ideal action provider type does not match its bundle")
        if self.action.provider_source_commit != self.source_commit or self.action.provider_source_tree != self.source_tree:
            raise ValueError("ideal action source does not match its provider bundle")
        if self.action.coordinate_layout_sha != self.coordinate_layout_sha:
            raise ValueError("ideal action coordinate ABI does not match its provider bundle")
        frozen = self.provider_type in {"FROZEN_V24", "TRACK_A_FROZEN_V10"}
        if frozen:
            if self.random_weights:
                raise ValueError("random weights cannot claim a frozen action provider")
            if self.frozen_checkpoint_verified is not True:
                raise ValueError("frozen provider requires exact checkpoint verification")
            for name in ("checkpoint_sha", "state_dict_sha", "model_build_spec_sha"):
                value = getattr(self, name)
                require_sha256(value, name)
            if self.action.provider_checkpoint_sha != self.checkpoint_sha:
                raise ValueError("frozen action checkpoint identity differs from provider bundle")
        elif any(value is not None for value in (self.checkpoint_sha, self.state_dict_sha, self.model_build_spec_sha)):
            # An analytic action can bind a file-backed derivation separately, but
            # must not masquerade as a loaded checkpoint.
            raise ValueError("analytic provider must not carry checkpoint/state-dict identities")

    def compatible_with(
        self,
        *,
        semantic_abi_sha: str,
        system_type: str,
        compatibility_context: ActionPriorCompatibilityContextV1 | None = None,
        file_admission: ActionPriorFileBackedAdmissionV1 | None = None,
        require_full_context: bool = False,
    ) -> bool:
        """Check legacy ABI fields plus optional full Track A admission evidence."""
        try:
            self.validate()
        except (TypeError, ValueError):
            return False
        basic = self.semantic_abi_sha == semantic_abi_sha and self.action.system_type == system_type
        if not basic:
            return False
        frozen = self.provider_type in {"FROZEN_V24", "TRACK_A_FROZEN_V10"}
        if compatibility_context is None:
            return not (require_full_context or frozen)
        if not compatibility_context.matches(self):
            return False
        if frozen and file_admission is None:
            return False
        return file_admission is None or file_admission.matches(self)

    def sha256(self) -> str:
        self.validate()
        return canonical_sha256({
            "provider_type": self.provider_type,
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "checkpoint_sha": self.checkpoint_sha,
            "state_dict_sha": self.state_dict_sha,
            "model_build_spec_sha": self.model_build_spec_sha,
            "config_sha": self.config_sha,
            "normalization_sha": self.normalization_sha,
            "semantic_abi_sha": self.semantic_abi_sha,
            "operation_registry_sha": self.operation_registry_sha,
            "projector_descriptor_sha": self.projector_descriptor_sha,
            "coordinate_layout_sha": self.coordinate_layout_sha,
            "target_schema_sha": self.target_schema_sha,
            "action_bounds_sha": self.action_bounds_sha,
            "action_sha": self.action.sha256(),
            "frozen_checkpoint_verified": self.frozen_checkpoint_verified,
        })


@dataclass(frozen=True)
class ActionPriorResolutionV1:
    selected_action: IdealLogicalActionV1 | None
    selected_provider_type: str | None
    local_analytic_action: IdealLogicalActionV1
    fallback_reason: str
    track_a_abi_compatible: bool

    @property
    def no_op_required(self) -> bool:
        return self.selected_action is None


@dataclass(frozen=True)
class DirectionComparisonV1:
    cosine: float
    conflict: bool
    abstain: bool
    reason_code: str


class ActionPriorProviderV1:
    """Resolve exact provider fallbacks without averaging Track A and local actions."""

    def __init__(
        self,
        *,
        semantic_abi_sha: str,
        local_analytic_bundle: ActionPriorBundleV1,
        track_a_bundle: ActionPriorBundleV1 | None = None,
        frozen_v24_bundle: ActionPriorBundleV1 | None = None,
        analytic_v24_bundle: ActionPriorBundleV1 | None = None,
        track_a_compatibility_context: ActionPriorCompatibilityContextV1 | None = None,
        track_a_file_admission: ActionPriorFileBackedAdmissionV1 | None = None,
        frozen_v24_compatibility_context: ActionPriorCompatibilityContextV1 | None = None,
        frozen_v24_file_admission: ActionPriorFileBackedAdmissionV1 | None = None,
    ) -> None:
        require_sha256(semantic_abi_sha, "semantic_abi_sha")
        local_analytic_bundle.validate()
        if local_analytic_bundle.provider_type != "TRACKB_LOCAL_ANALYTIC":
            raise ValueError("Track B requires an independent TRACKB_LOCAL_ANALYTIC baseline")
        self.semantic_abi_sha = semantic_abi_sha
        self.local_analytic_bundle = local_analytic_bundle
        self.track_a_bundle = track_a_bundle
        self.frozen_v24_bundle = frozen_v24_bundle
        self.analytic_v24_bundle = analytic_v24_bundle
        self.track_a_compatibility_context = track_a_compatibility_context
        self.track_a_file_admission = track_a_file_admission
        self.frozen_v24_compatibility_context = frozen_v24_compatibility_context
        self.frozen_v24_file_admission = frozen_v24_file_admission

    def resolve(self, system_type: str) -> ActionPriorResolutionV1:
        local = self.local_analytic_bundle
        if not local.compatible_with(semantic_abi_sha=self.semantic_abi_sha, system_type=system_type):
            raise ValueError("independent Track B local analytic action lost ABI compatibility")
        track_a_ok = self.track_a_bundle is not None and self.track_a_bundle.compatible_with(
            semantic_abi_sha=self.semantic_abi_sha,
            system_type=system_type,
            compatibility_context=self.track_a_compatibility_context,
            file_admission=self.track_a_file_admission,
            require_full_context=True,
        )
        frozen_v24_ok = self.frozen_v24_bundle is not None and self.frozen_v24_bundle.compatible_with(
            semantic_abi_sha=self.semantic_abi_sha,
            system_type=system_type,
            compatibility_context=self.frozen_v24_compatibility_context,
            file_admission=self.frozen_v24_file_admission,
            require_full_context=True,
        )
        candidates = (
            (self.track_a_bundle if track_a_ok else None, "TRACK_A_FROZEN_V10"),
            (self.frozen_v24_bundle if frozen_v24_ok else None, "FROZEN_V24"),
            (self.analytic_v24_bundle, "ANALYTIC_V24"),
        )
        for bundle, label in candidates:
            if bundle is not None and (
                label in {"TRACK_A_FROZEN_V10", "FROZEN_V24"}
                or bundle.compatible_with(
                    semantic_abi_sha=self.semantic_abi_sha, system_type=system_type,
                )
            ):
                return ActionPriorResolutionV1(bundle.action, label, local.action, "EXACT_PROVIDER_VALID", track_a_ok)
        return ActionPriorResolutionV1(None, None, local.action, "NO_VALID_PRIMARY_PROVIDER_NO_OP", track_a_ok)

    @staticmethod
    def compare_directions(
        track_a_action: IdealLogicalActionV1 | None,
        local_action: IdealLogicalActionV1,
        *,
        cosine_threshold: float = 0.8,
        research_comparison_mode: bool = False,
    ) -> DirectionComparisonV1:
        if track_a_action is None:
            return DirectionComparisonV1(1.0, False, False, "TRACK_A_OPTIONAL_INPUT_MISSING")
        if track_a_action.system_type != local_action.system_type or (
            track_a_action.coordinate_layout_sha != local_action.coordinate_layout_sha
        ):
            return DirectionComparisonV1(-1.0, True, True, "TRACK_A_ABI_MISMATCH")
        left = np.asarray(track_a_action.grouped_coordinates, dtype=np.float64)
        right = np.asarray(local_action.grouped_coordinates, dtype=np.float64)
        norm = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1.0e-15)
        cosine = float(np.clip(left @ right / norm, -1.0, 1.0))
        conflict = cosine < float(cosine_threshold)
        return DirectionComparisonV1(
            cosine=cosine,
            conflict=conflict,
            abstain=conflict and not research_comparison_mode,
            reason_code="TRACK_A_TRACK_B_DIRECTION_CONFLICT" if conflict else "TRACK_A_TRACK_B_DIRECTION_AGREE",
        )


__all__ = [
    "ActionPriorBundleV1", "ActionPriorCompatibilityContextV1",
    "ActionPriorFileBackedAdmissionV1", "ActionPriorProviderV1", "ActionPriorResolutionV1",
    "DirectionComparisonV1", "PROVIDER_TYPES",
]
