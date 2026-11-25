"""Typed parent-checkpoint binding for the unequal-concentration overlay.

This is a provenance record, not a checkpoint and not a training resume path.
It binds a distribution overlay to the exact frozen native point and to the
checkpoint or family-refinement replay that produced that point. Legacy V2/V3
checkpoint readers remain unchanged and never consume this schema.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

from .contracts import canonical_json_bytes, parse_canonical_json_object, require_git_oid, require_sha256
from .effective_checkpoint_reader import Effective_checkpoint_replay
from .effective_inference import _model_artifact_binding
from .effective_inference_artifact import Effective_point_inference, validate_effective_point_artifact
from .family_refinement_reader import Family_refinement_replay
from .unequal_concentration_point import (
    Unequal_concentration_point_overlay,
    validate_unequal_concentration_point_artifact,
)


checkpoint_binding_schema = "TrackBUnequalConcentrationCheckpointBindingV1"
checkpoint_binding_scope = "BASE_CHECKPOINT_BINDING_FOR_DISTRIBUTION_OVERLAY_ONLY"
binding_fields = frozenset({
    "schema", "scope", "source_commit", "source_tree", "overlay_file_sha",
    "overlay_source_commit", "overlay_source_tree", "base_point_artifact_sha",
    "base_model_artifact_sha", "base_model_artifact_kind", "base_model_state_sha",
    "base_model_config_sha", "training_checkpoint_file_sha", "training_identity_sha",
    "training_source_commit", "training_source_tree", "optimizer_constructed",
    "parent_training_resumed", "scientific_gate_issued", "hardware_action_authorized",
})


def _training_replay(replay: Effective_checkpoint_replay | Family_refinement_replay) -> Effective_checkpoint_replay:
    if type(replay) is Effective_checkpoint_replay:
        return replay
    if type(replay) is Family_refinement_replay:
        return replay.parent
    raise TypeError("unequal concentration binding requires a typed checkpoint replay")


def _binding_payload(
    base_artifact: Effective_point_inference,
    overlay_artifact: Unequal_concentration_point_overlay,
    replay: Effective_checkpoint_replay | Family_refinement_replay,
    *, source_commit: str, source_tree: str,
) -> dict[str, Any]:
    require_git_oid(source_commit, "binding_source_commit")
    require_git_oid(source_tree, "binding_source_tree")
    if type(base_artifact) is not Effective_point_inference \
            or type(overlay_artifact) is not Unequal_concentration_point_overlay:
        raise TypeError("unequal concentration binding requires typed point and overlay artifacts")
    training = _training_replay(replay)
    replay.summary()
    base_payload = base_artifact.payload()
    overlay_payload = overlay_artifact.payload()
    validate_effective_point_artifact(base_payload)
    validate_unequal_concentration_point_artifact(overlay_payload, base_payload=base_payload)
    model = base_payload["model_artifact"]
    expected_model = _model_artifact_binding(replay)
    if model != expected_model or overlay_payload["base_model_artifact_sha"] != expected_model["file_sha"]:
        raise ValueError("unequal concentration overlay does not bind to the supplied replay")
    return {
        "schema": checkpoint_binding_schema, "scope": checkpoint_binding_scope,
        "source_commit": source_commit, "source_tree": source_tree,
        "overlay_file_sha": overlay_artifact.file_sha,
        "overlay_source_commit": overlay_payload["source_commit"],
        "overlay_source_tree": overlay_payload["source_tree"],
        "base_point_artifact_sha": base_artifact.file_sha,
        "base_model_artifact_sha": model["file_sha"],
        "base_model_artifact_kind": model["kind"],
        "base_model_state_sha": model["model_state_sha"],
        "base_model_config_sha": model["model_config_sha"],
        "training_checkpoint_file_sha": training.file_sha,
        "training_identity_sha": training.identity.sha256(),
        "training_source_commit": training.identity.source_commit,
        "training_source_tree": training.identity.source_tree,
        "optimizer_constructed": False, "parent_training_resumed": False,
        "scientific_gate_issued": False, "hardware_action_authorized": False,
    }


def validate_unequal_concentration_checkpoint_binding(payload: Mapping[str, Any]) -> None:
    """Validate a binding without treating it as a loadable model checkpoint."""
    if type(payload) is not dict or set(payload) != binding_fields \
            or payload["schema"] != checkpoint_binding_schema \
            or payload["scope"] != checkpoint_binding_scope:
        raise ValueError("unequal concentration checkpoint binding schema differs")
    for name in ("source_commit", "source_tree", "overlay_source_commit", "overlay_source_tree",
                 "training_source_commit", "training_source_tree"):
        require_git_oid(payload[name], name)
    for name in ("overlay_file_sha", "base_point_artifact_sha", "base_model_artifact_sha",
                 "base_model_state_sha", "base_model_config_sha", "training_checkpoint_file_sha",
                 "training_identity_sha"):
        require_sha256(payload[name], name)
    if payload["base_model_artifact_kind"] not in {"EFFECTIVE_CPU_CHECKPOINT", "FAMILY_REFINEMENT_OVERLAY"}:
        raise ValueError("unequal concentration checkpoint binding model kind differs")
    for name in (
        "optimizer_constructed", "parent_training_resumed", "scientific_gate_issued",
        "hardware_action_authorized",
    ):
        if payload[name] is not False:
            raise ValueError("unequal concentration binding authority flags must remain false")


@dataclass(frozen=True)
class Unequal_concentration_checkpoint_binding:
    payload_bytes: bytes
    file_sha: str

    def payload(self) -> dict[str, Any]:
        require_sha256(self.file_sha, "unequal_concentration_binding_file_sha")
        if type(self.payload_bytes) is not bytes \
                or hashlib.sha256(self.payload_bytes).hexdigest() != self.file_sha:
            raise ValueError("unequal concentration binding byte identity differs")
        payload = parse_canonical_json_object(
            self.payload_bytes, schema=checkpoint_binding_schema, exact_fields=binding_fields,
        )
        validate_unequal_concentration_checkpoint_binding(payload)
        return payload


def build_unequal_concentration_checkpoint_binding(
    base_artifact: Effective_point_inference,
    overlay_artifact: Unequal_concentration_point_overlay,
    replay: Effective_checkpoint_replay | Family_refinement_replay, *,
    source_commit: str, source_tree: str,
) -> Unequal_concentration_checkpoint_binding:
    payload = _binding_payload(
        base_artifact, overlay_artifact, replay,
        source_commit=source_commit, source_tree=source_tree,
    )
    encoded = canonical_json_bytes(payload)
    artifact = Unequal_concentration_checkpoint_binding(
        encoded, hashlib.sha256(encoded).hexdigest(),
    )
    artifact.payload()
    return artifact


def load_unequal_concentration_checkpoint_binding_bytes(
    payload: bytes, base_artifact: Effective_point_inference,
    overlay_artifact: Unequal_concentration_point_overlay,
    replay: Effective_checkpoint_replay | Family_refinement_replay, *,
    expected_file_sha: str, expected_source_commit: str, expected_source_tree: str,
) -> Unequal_concentration_checkpoint_binding:
    """Read a binding only when the exact point, overlay and replay agree."""
    require_sha256(expected_file_sha, "expected_unequal_concentration_binding_sha")
    require_git_oid(expected_source_commit, "expected_binding_source_commit")
    require_git_oid(expected_source_tree, "expected_binding_source_tree")
    artifact = Unequal_concentration_checkpoint_binding(payload, expected_file_sha)
    observed = artifact.payload()
    expected = build_unequal_concentration_checkpoint_binding(
        base_artifact, overlay_artifact, replay,
        source_commit=expected_source_commit, source_tree=expected_source_tree,
    ).payload()
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise ValueError("unequal concentration binding differs from the supplied parent replay")
    return artifact


def verify_unequal_concentration_checkpoint_binding(
    artifact: Unequal_concentration_checkpoint_binding,
    base_artifact: Effective_point_inference,
    overlay_artifact: Unequal_concentration_point_overlay,
    replay: Effective_checkpoint_replay | Family_refinement_replay, *,
    source_commit: str, source_tree: str,
) -> dict[str, Any]:
    """Rebuild the provenance record; no model forward or optimizer is used."""
    if type(artifact) is not Unequal_concentration_checkpoint_binding:
        raise TypeError("unequal concentration binding verification requires its typed artifact")
    original = artifact.payload()
    repeated = build_unequal_concentration_checkpoint_binding(
        base_artifact, overlay_artifact, replay,
        source_commit=source_commit, source_tree=source_tree,
    ).payload()
    compare_fields = tuple(name for name in binding_fields if name not in {"source_commit", "source_tree"})
    if any(canonical_json_bytes(original[name]) != canonical_json_bytes(repeated[name]) for name in compare_fields):
        raise ValueError("unequal concentration checkpoint binding does not replay")
    return {
        "schema": "TrackBUnequalConcentrationCheckpointBindingVerificationV1",
        "binding_file_sha": artifact.file_sha,
        "base_point_artifact_sha": original["base_point_artifact_sha"],
        "replay_source_commit": source_commit,
        "replay_source_tree": source_tree,
        "model_forward_evaluations": 0,
        "optimizer_constructed": False,
        "parent_training_resumed": False,
        "scientific_gate_issued": False,
        "hardware_action_authorized": False,
    }
