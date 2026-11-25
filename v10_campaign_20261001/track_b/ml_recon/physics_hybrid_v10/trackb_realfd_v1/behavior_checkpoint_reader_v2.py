"""Read-only, receipt-bound loader for a Track B V2 behavior checkpoint."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
import hashlib
import io
import math
from threading import RLock
from typing import Any
from weakref import WeakKeyDictionary

import torch

from ml_recon.data_v2_4.hashing import sha256_json

from .inverse_transformer import (
    TrackBInverseTransformerConfigV1,
    TrackBInverseTransformerV1,
    require_action_decode_scale_binding,
    require_gain_parameterization_binding,
)
from .inverse_transformer_pilot import TrackBTargetNormalizationV1
from .learner_recipe_contract_v2 import (
    Behavior_pilot_receipt_pin_v2,
    verify_behavior_pilot_receipt_bytes_v2,
)
from .training import trackb_model_state_sha


BEHAVIOR_CHECKPOINT_SCHEMA_V2 = "TrackBBehaviorCheckpointV2"
BEHAVIOR_OPTIMIZER_STEPS_V2 = 191
MAX_BEHAVIOR_CHECKPOINT_BYTES_V2 = 512 * 1024 * 1024
_CHECKPOINT_FIELDS_V2 = frozenset({
    "schema", "stage_plan_sha256", "source_commit", "source_tree", "history_policy",
    "history_policy_sha256", "source_view_kind", "model_config", "optimizer_steps",
    "target_normalization", "normalization_source", "round_schedule_sha256",
    "batch_schedule_sha256", "train_source", "validation_source", "dataset_receipt_sha256",
    "recipe_sha256", "research_only", "selection_evidence", "promotion_ready",
    "hardware_action_authorized", "model_state_dict",
})


@dataclass(frozen=True)
class Behavior_checkpoint_evidence_v2:
    """Verified artifact identities; this is not Stage 3 or scientific admission."""

    checkpoint_sha256: str
    receipt_sha256: str
    source_commit: str
    source_tree: str
    stage_plan_sha256: str
    recipe_sha256: str
    dataset_receipt_sha256: str
    round_schedule_sha256: str
    batch_schedule_sha256: str
    optimizer_steps: int
    model_config_sha256: str
    model_state_sha256: str
    schema: str = BEHAVIOR_CHECKPOINT_SCHEMA_V2
    learner_views_admitted: bool = False
    training_admitted: bool = False
    scientific_gate: str = "NOT_PERFORMED"
    hardware_action_authorized: bool = False


@dataclass(frozen=True, eq=False)
class Behavior_checkpoint_model_v2:
    """Frozen model context whose reader issuance is checked out-of-band."""

    model: TrackBInverseTransformerV1
    evidence: Behavior_checkpoint_evidence_v2


_checkpoint_context_lock = RLock()
_issued_checkpoint_contexts: WeakKeyDictionary[
    Behavior_checkpoint_model_v2,
    tuple[TrackBInverseTransformerV1, Behavior_checkpoint_evidence_v2,
          tuple[tuple[str, type, object], ...]],
] = WeakKeyDictionary()


def _checkpoint_evidence_snapshot(
    evidence: Behavior_checkpoint_evidence_v2,
) -> tuple[tuple[str, type, object], ...]:
    return tuple(
        (item.name, type(value), value)
        for item in fields(Behavior_checkpoint_evidence_v2)
        for value in (getattr(evidence, item.name),)
    )


def _issue_checkpoint_context_v2(
    model: TrackBInverseTransformerV1,
    evidence: Behavior_checkpoint_evidence_v2,
) -> Behavior_checkpoint_model_v2:
    context = Behavior_checkpoint_model_v2(model, evidence)
    record = (model, evidence, _checkpoint_evidence_snapshot(evidence))
    with _checkpoint_context_lock:
        _issued_checkpoint_contexts[context] = record
    return context


def _require_object(value: object, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"behavior checkpoint {name} must be an object")
    return value


def _require_exact_fields(value: dict[str, Any], fields: frozenset[str], name: str) -> None:
    if set(value) != fields:
        raise ValueError(f"behavior checkpoint {name} has missing or unknown fields")


def _require_sha(value: object, name: str) -> str:
    if (type(value) is not str or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"behavior checkpoint {name} must be a lowercase SHA-256")
    return value


def _same_typed_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (set(actual) == set(expected)
                and all(_same_typed_value(actual[key], expected[key]) for key in expected))
    if type(expected) in (list, tuple):
        return len(actual) == len(expected) and all(
            _same_typed_value(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _normalization_from_receipt(value: object) -> TrackBTargetNormalizationV1:
    payload = _require_object(value, "target_normalization")
    fields = frozenset({
        "amplitude_scale", "phase_scale", "gain_center", "gain_scale",
        "action_loss_normalization_kind", "amplitude_scale_by_round",
        "phase_scale_by_round", "schema",
    })
    _require_exact_fields(payload, fields, "target_normalization")
    normalized = dict(payload)
    for name in ("amplitude_scale", "phase_scale", "gain_center", "gain_scale"):
        if type(normalized[name]) not in (int, float) or not math.isfinite(float(normalized[name])):
            raise ValueError(f"behavior checkpoint normalization {name} must be a finite real scalar")
    for name in ("amplitude_scale_by_round", "phase_scale_by_round"):
        if type(normalized[name]) not in (list, tuple):
            raise ValueError(f"behavior checkpoint {name} must be a numeric sequence")
        if any(type(item) not in (int, float) or not math.isfinite(float(item)) for item in normalized[name]):
            raise ValueError(f"behavior checkpoint {name} contains a non-real or non-finite value")
        normalized[name] = tuple(normalized[name])
    result = TrackBTargetNormalizationV1(**normalized)
    result.validate()
    return result


def _model_config_from_receipt(value: object, normalization: TrackBTargetNormalizationV1) -> TrackBInverseTransformerConfigV1:
    payload = _require_object(value, "model_config")
    config_values = dict(payload)
    for name in ("amplitude_decode_scale_by_round", "phase_decode_scale_by_round"):
        if name in config_values:
            if type(config_values[name]) is not list:
                raise ValueError(f"behavior checkpoint model config {name} must be a JSON array")
            config_values[name] = tuple(config_values[name])
    try:
        config = TrackBInverseTransformerConfigV1(**config_values)
    except TypeError as error:
        raise ValueError("behavior checkpoint model config has missing or unknown fields") from error
    config.validate()
    if config.schema != "TrackBInverseTransformerV1":
        raise ValueError("behavior checkpoint model config schema must remain V1")
    require_gain_parameterization_binding(config, normalization.to_dict())
    require_action_decode_scale_binding(config, normalization.to_dict())
    if config.gain_parameterization == "standardized_latent_gain_v1":
        if config.gain_center != normalization.gain_center or config.gain_scale != normalization.gain_scale:
            raise ValueError("behavior checkpoint gain normalization differs from receipt")
    if config.action_projection_kind == "train_rms_centered_bounded_gauge_v3":
        if (tuple(config.amplitude_decode_scale_by_round) != tuple(normalization.amplitude_scale_by_round)
                or tuple(config.phase_decode_scale_by_round) != tuple(normalization.phase_scale_by_round)):
            raise ValueError("behavior checkpoint action normalization differs from receipt")
    return config


def _require_registered_model_config(
    config: TrackBInverseTransformerConfigV1,
    recipe: dict[str, Any],
    normalization: TrackBTargetNormalizationV1,
) -> None:
    expected = TrackBInverseTransformerConfigV1(
        system_type=recipe["system_type"], mode_count=recipe["mode_count"],
        width=recipe["width"], heads=recipe["heads"], blocks=recipe["blocks"],
        action_value_feature_kind=recipe["action_value_feature_kind"],
        action_projection_kind=recipe["action_projection_kind"],
        gain_parameterization=recipe["gain_parameterization"],
        gain_center=(normalization.gain_center
                     if recipe["gain_parameterization"] == "standardized_latent_gain_v1" else 0.0),
        gain_scale=(normalization.gain_scale
                    if recipe["gain_parameterization"] == "standardized_latent_gain_v1" else 1.0),
        amplitude_decode_scale_by_round=(
            normalization.amplitude_scale_by_round
            if recipe["action_projection_kind"] == "train_rms_centered_bounded_gauge_v3" else ()),
        phase_decode_scale_by_round=(
            normalization.phase_scale_by_round
            if recipe["action_projection_kind"] == "train_rms_centered_bounded_gauge_v3" else ()),
        history_gain_input_kind=recipe["history_gain_input_kind"],
    )
    if not _same_typed_value(config.to_dict(), expected.to_dict()):
        raise ValueError("behavior checkpoint model config differs from the registered V2 recipe binding")


def read_behavior_checkpoint_v2(
    receipt_bytes: bytes,
    checkpoint_bytes: bytes,
    pin: Behavior_pilot_receipt_pin_v2,
) -> Behavior_checkpoint_model_v2:
    """Verify immutable V2 receipt/checkpoint bytes and load the registered model strictly."""
    if type(checkpoint_bytes) is not bytes or not checkpoint_bytes:
        raise ValueError("behavior checkpoint must be nonempty immutable bytes")
    if len(checkpoint_bytes) > MAX_BEHAVIOR_CHECKPOINT_BYTES_V2:
        raise ValueError("behavior checkpoint exceeds the bounded reader limit")
    receipt = verify_behavior_pilot_receipt_bytes_v2(receipt_bytes, pin)
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    if checkpoint_sha256 != pin.expected_checkpoint_sha256:
        raise ValueError("behavior checkpoint bytes differ from the independently pinned SHA")
    receipt_checkpoint_sha = receipt["model"]["checkpoint"]["sha256"]
    if checkpoint_sha256 != receipt_checkpoint_sha:
        raise ValueError("behavior checkpoint SHA differs from the verified receipt")
    try:
        with io.BytesIO(checkpoint_bytes) as buffer:
            payload = torch.load(buffer, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("behavior checkpoint is not a safe weights-only PyTorch artifact") from error
    checkpoint = _require_object(payload, "root")
    _require_exact_fields(checkpoint, _CHECKPOINT_FIELDS_V2, "root")
    receipt_recipe = receipt["recipe"]
    expected_metadata = {
        "schema": BEHAVIOR_CHECKPOINT_SCHEMA_V2,
        "stage_plan_sha256": receipt["stage_plan_sha256"],
        "source_commit": receipt["source_commit"],
        "source_tree": receipt["source_tree"],
        "history_policy": receipt_recipe["history_policy"],
        "history_policy_sha256": receipt_recipe["history_policy_sha256"],
        "source_view_kind": receipt_recipe["source_view_kind"],
        "optimizer_steps": BEHAVIOR_OPTIMIZER_STEPS_V2,
        "normalization_source": "FRESH_CURRENT_DATASET_TRAIN_ONLY",
        "round_schedule_sha256": receipt["round_schedule_sha256"],
        "batch_schedule_sha256": receipt["batch_schedule_sha256"],
        "train_source": receipt["train_source"],
        "validation_source": receipt["validation_source"],
        "dataset_receipt_sha256": receipt["dataset_receipt_sha256"],
        "recipe_sha256": receipt["recipe_sha256"],
        "research_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }
    for name, expected in expected_metadata.items():
        if not _same_typed_value(checkpoint.get(name), expected):
            raise ValueError(f"behavior checkpoint metadata differs from verified receipt at {name}")
    for name in ("round_schedule_sha256", "batch_schedule_sha256"):
        _require_sha(checkpoint[name], name)
    if type(checkpoint["optimizer_steps"]) is not int or checkpoint["optimizer_steps"] != BEHAVIOR_OPTIMIZER_STEPS_V2:
        raise ValueError("behavior checkpoint must bind exactly 191 optimizer updates")
    model_config_payload = _require_object(checkpoint["model_config"], "model_config")
    if not _same_typed_value(model_config_payload, receipt["model"]["config"]):
        raise ValueError("behavior checkpoint model config differs from verified receipt")
    normalization = _normalization_from_receipt(receipt["model"]["target_normalization"])
    checkpoint_normalization = _normalization_from_receipt(checkpoint["target_normalization"])
    if not _same_typed_value(checkpoint_normalization.to_dict(), normalization.to_dict()):
        raise ValueError("behavior checkpoint target normalization differs from verified receipt")
    config = _model_config_from_receipt(model_config_payload, normalization)
    _require_registered_model_config(config, receipt_recipe, normalization)
    raw_state_dict = checkpoint["model_state_dict"]
    if type(raw_state_dict) not in (dict, OrderedDict):
        raise ValueError("behavior checkpoint model_state_dict must be an ordered tensor mapping")
    state_dict = dict(raw_state_dict)
    model = TrackBInverseTransformerV1(config)
    expected_state = model.state_dict()
    if set(state_dict) != set(expected_state):
        raise ValueError("behavior checkpoint state dict has missing or unknown tensor keys")
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"behavior checkpoint tensor {name} is not a tensor")
        if tensor.shape != expected_state[name].shape or tensor.dtype != expected_state[name].dtype:
            raise ValueError(f"behavior checkpoint tensor {name} has a mismatched shape or dtype")
        if tensor.dtype.is_floating_point and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"behavior checkpoint tensor {name} contains non-finite values")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("behavior checkpoint state dict failed strict registered-model loading") from error
    model.eval()
    evidence = Behavior_checkpoint_evidence_v2(
        checkpoint_sha256=checkpoint_sha256,
        receipt_sha256=pin.expected_receipt_sha256,
        source_commit=receipt["source_commit"], source_tree=receipt["source_tree"],
        stage_plan_sha256=receipt["stage_plan_sha256"], recipe_sha256=receipt["recipe_sha256"],
        dataset_receipt_sha256=receipt["dataset_receipt_sha256"],
        round_schedule_sha256=receipt["round_schedule_sha256"],
        batch_schedule_sha256=receipt["batch_schedule_sha256"],
        optimizer_steps=BEHAVIOR_OPTIMIZER_STEPS_V2,
        model_config_sha256=sha256_json(model.config.to_dict()),
        model_state_sha256=trackb_model_state_sha(model),
    )
    return _issue_checkpoint_context_v2(model, evidence)


def verify_loaded_behavior_model_v2(
    checkpoint_model: Behavior_checkpoint_model_v2,
) -> None:
    """Recheck that a frozen model object still matches reader-verified weights."""
    if type(checkpoint_model) is not Behavior_checkpoint_model_v2:
        raise ValueError("behavior model binding requires a reader-issued V2 checkpoint context")
    with _checkpoint_context_lock:
        issuance = _issued_checkpoint_contexts.get(checkpoint_model)
    if issuance is None:
        raise ValueError("behavior model binding requires a reader-issued V2 checkpoint context")
    issued_model, issued_evidence, issued_evidence_snapshot = issuance
    model = checkpoint_model.model
    evidence = checkpoint_model.evidence
    if (model is not issued_model or evidence is not issued_evidence
            or type(evidence) is not Behavior_checkpoint_evidence_v2
            or _checkpoint_evidence_snapshot(evidence) != issued_evidence_snapshot):
        raise ValueError("reader-issued behavior checkpoint identities changed after issuance")
    if not isinstance(model, TrackBInverseTransformerV1) or model.training:
        raise ValueError("behavior model binding requires the frozen evaluation model")
    model.config.validate()
    if sha256_json(model.config.to_dict()) != evidence.model_config_sha256:
        raise ValueError("behavior model config changed after V2 checkpoint loading")
    if trackb_model_state_sha(model) != evidence.model_state_sha256:
        raise ValueError("behavior model weights changed after V2 checkpoint loading")


__all__ = [
    "Behavior_checkpoint_evidence_v2",
    "Behavior_checkpoint_model_v2",
    "read_behavior_checkpoint_v2",
    "verify_loaded_behavior_model_v2",
]
