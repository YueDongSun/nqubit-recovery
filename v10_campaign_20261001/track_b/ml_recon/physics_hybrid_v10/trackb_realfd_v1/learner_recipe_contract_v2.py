"""Verify an independently pinned V2 behavior-pilot receipt's metadata only."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping

from ml_recon.data_v2_4.synthetic.trackb_teacher_policy import (
    TRACKB_TEACHER_POLICY_V2,
    TRACKB_TEACHER_POLICY_V2_SHA256,
)

from .behavior_v2_contract import Behavior_recipe_v2
from .contracts import canonical_json_bytes


MAX_BEHAVIOR_RECEIPT_BYTES_V2 = 1024 * 1024
BEHAVIOR_PILOT_RECEIPT_SCHEMA_V2 = "TrackBBehaviorPilotReceiptV2"
BEHAVIOR_PILOT_RECIPE_SCHEMA_V2 = "TrackBBehaviorRecipeV2"
BEHAVIOR_MODEL_CONFIG_SCHEMA_V1 = "TrackBInverseTransformerV1"
BEHAVIOR_COMPLETED_OPTIMIZER_UPDATES_V2 = 191
BEHAVIOR_SOURCE_VIEW_V2 = "v24_oracle_history_v1"
BEHAVIOR_SEVERITY_MATCHING_V2 = "evaluated_midpoint_v2"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_RECEIPT_FIELDS = frozenset({
    "batch_schedule_sha256", "completed_optimizer_updates", "dataset_identity",
    "dataset_receipt_sha256", "hardware_action_authorized", "inverse_transformer_dependency",
    "model", "model_self_rollout", "promotion_ready", "recipe", "recipe_sha256",
    "research_only", "round_schedule_sha256", "scientific_gate", "selection_evidence",
    "schema", "source_commit", "source_tree", "stage_plan_sha256", "teacher_forced_evaluation",
    "ten_round_guidance", "train_source", "training_progress", "validation_source",
})
_DATASET_IDENTITY_FIELDS = frozenset({
    "build_id", "build_manifest_sha", "contract_sha", "dataset_contract_version",
    "generator_sha", "index_sha", "ready_sha", "rows", "schema_sha", "shard_hashes",
    "source_commit", "source_contract_version", "source_tree", "split_sha",
})
_SOURCE_FIELDS = frozenset({
    "build_id", "build_manifest_sha", "cell_id", "contract_sha", "generator_sha",
    "history_policy", "history_policy_sha256", "index_sha", "ready_sha", "sample_set_sha",
    "schema_sha", "source_commit", "source_tree", "split", "split_sha",
})
_CHECKPOINT_FIELDS = frozenset({
    "file_id", "relative_path", "sha256", "size_bytes", "volume_serial_number",
})
_MODEL_CONFIG_FIELDS = frozenset({
    "action_projection_kind", "action_value_feature_kind", "amplitude_bound",
    "amplitude_decode_scale_by_round", "blocks", "gain_center", "gain_parameterization",
    "gain_scale", "heads", "history_gain_input_kind", "mode_count", "phase_bound",
    "phase_decode_scale_by_round", "risk_qualification_state", "schema", "system_type", "width",
})
_DATASET_SOURCE_BINDING_FIELDS = (
    "build_id", "build_manifest_sha", "contract_sha", "generator_sha", "index_sha",
    "ready_sha", "schema_sha", "split_sha", "source_commit", "source_tree",
)
_TRAINING_PROGRESS_FIELDS_V2 = frozenset({
    "optimizer_step", "round_index", "action_normalized_mse",
    "amplitude_normalized_mse", "phase_normalized_mse", "gain_standardized_huber",
    "action_mse", "gain_mse", "loss", "gradient_norm_before_clip",
})
_TRAINING_PROGRESS_METRIC_FIELDS_V2 = _TRAINING_PROGRESS_FIELDS_V2 - {
    "optimizer_step", "round_index",
}
_TRAINING_PROGRESS_STEPS_V2 = (
    1, *range(10, BEHAVIOR_COMPLETED_OPTIMIZER_UPDATES_V2, 10),
    BEHAVIOR_COMPLETED_OPTIMIZER_UPDATES_V2,
)


@dataclass(frozen=True)
class Behavior_pilot_receipt_pin_v2:
    """Caller-owned identities and admitted V2 recipe/policy for one receipt."""

    expected_receipt_sha256: str
    expected_source_commit: str
    expected_source_tree: str
    expected_stage_plan_sha256: str
    expected_dataset_receipt_sha256: str
    expected_checkpoint_sha256: str
    expected_recipe_sha256: str
    admitted_recipe: Mapping[str, Any]
    admitted_policy_id: str
    admitted_policy_sha256: str

    def validate(self) -> None:
        for name in (
            "expected_receipt_sha256", "expected_stage_plan_sha256",
            "expected_dataset_receipt_sha256", "expected_checkpoint_sha256",
            "expected_recipe_sha256", "admitted_policy_sha256",
        ):
            value = getattr(self, name)
            if type(value) is not str or not _SHA256_PATTERN.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        for name in ("expected_source_commit", "expected_source_tree"):
            value = getattr(self, name)
            if type(value) is not str or not _GIT_OID_PATTERN.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase Git object id")
        registered_recipe = Behavior_recipe_v2().to_dict()
        if (type(self.admitted_recipe) is not dict
                or not _same_typed_value(dict(self.admitted_recipe), registered_recipe)):
            raise ValueError("pin must carry the exact registered TrackBBehaviorRecipeV2")
        registered_recipe_sha = hashlib.sha256(canonical_json_bytes(registered_recipe)).hexdigest()
        if self.expected_recipe_sha256 != registered_recipe_sha:
            raise ValueError("pin recipe SHA differs from its admitted V2 recipe")
        if (self.admitted_policy_id != TRACKB_TEACHER_POLICY_V2
                or self.admitted_policy_sha256 != TRACKB_TEACHER_POLICY_V2_SHA256):
            raise ValueError("pin must identify the registered Track B teacher policy V2")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("behavior receipt JSON contains a duplicate key")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"behavior receipt JSON contains non-finite number {value}")


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"behavior receipt {name} must be an object")
    return value


def _require_exact_fields(value: dict[str, Any], expected: frozenset[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"behavior receipt {name} has missing or unknown fields")


def _same_typed_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (set(actual) == set(expected)
                and all(_same_typed_value(actual[key], expected[key]) for key in expected))
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _same_typed_value(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _verify_training_schedule_v2(receipt: dict[str, Any], recipe: dict[str, Any]) -> None:
    """Recreate the registered training schedule and verify its sparse trace."""
    from ml_recon.data_v2_4.hashing import sha256_json
    from .inverse_transformer_pilot import _round_coverage_batch_schedule

    batch_schedule, round_schedule = _round_coverage_batch_schedule(
        dataset_size=recipe["train_samples"],
        batch_size=recipe["batch_size"],
        optimizer_steps=recipe["optimizer_steps"],
        seed=recipe["model_seed"],
    )
    if (len(round_schedule) != BEHAVIOR_COMPLETED_OPTIMIZER_UPDATES_V2
            or len(batch_schedule) != BEHAVIOR_COMPLETED_OPTIMIZER_UPDATES_V2):
        raise ValueError("behavior receipt registered training schedule has an invalid length")
    if (sha256_json(round_schedule) != receipt["round_schedule_sha256"]
            or sha256_json(batch_schedule) != receipt["batch_schedule_sha256"]):
        raise ValueError("behavior receipt schedule hashes differ from the registered recipe")

    dataset_size = recipe["train_samples"]
    batch_size = recipe["batch_size"]
    if any(type(round_index) is not int or not 0 <= round_index < 10
           for round_index in round_schedule):
        raise ValueError("registered behavior round schedule contains an invalid round")
    if any(type(batch) is not tuple or len(batch) != batch_size
           or any(type(index) is not int or not 0 <= index < dataset_size for index in batch)
           for batch in batch_schedule):
        raise ValueError("registered behavior batch schedule has invalid fixed-size TRAIN indexes")

    for round_index in range(10):
        covered_indices = {
            index
            for scheduled_round, batch in zip(round_schedule[:160], batch_schedule[:160])
            if scheduled_round == round_index
            for index in batch
        }
        if covered_indices != set(range(dataset_size)):
            raise ValueError("first 160 behavior updates do not cover every TRAIN row per round")

    progress = receipt["training_progress"]
    if type(progress) is not list or len(progress) != len(_TRAINING_PROGRESS_STEPS_V2):
        raise ValueError("behavior receipt training_progress must contain the 21 sparse updates")
    for record, expected_step in zip(progress, _TRAINING_PROGRESS_STEPS_V2):
        if type(record) is not dict:
            raise ValueError("behavior receipt training_progress record must be an object")
        _require_exact_fields(record, _TRAINING_PROGRESS_FIELDS_V2, "training_progress record")
        step = record["optimizer_step"]
        round_index = record["round_index"]
        if type(step) is not int or step != expected_step:
            raise ValueError("behavior receipt training_progress steps are missing, duplicate, or reordered")
        if (type(round_index) is not int or not 1 <= round_index <= 10
                or round_index != round_schedule[step - 1] + 1):
            raise ValueError("behavior receipt training_progress round differs from its schedule")
        for metric_name in _TRAINING_PROGRESS_METRIC_FIELDS_V2:
            metric = record[metric_name]
            if (type(metric) not in (int, float) or not math.isfinite(metric)
                    or metric < 0):
                raise ValueError(
                    f"behavior receipt training_progress {metric_name} must be finite and nonnegative"
                )


def verify_behavior_pilot_receipt_bytes_v2(
    payload: bytes,
    pin: Behavior_pilot_receipt_pin_v2,
) -> dict[str, Any]:
    """Verify canonical receipt metadata against a separate run-specific pin.

    This does not read dataset shards or checkpoint tensors; those remain the
    responsibility of the downstream Stage 3 evidence reader.
    """
    if type(payload) is not bytes or len(payload) > MAX_BEHAVIOR_RECEIPT_BYTES_V2:
        raise ValueError("behavior receipt must be bounded immutable bytes")
    if type(pin) is not Behavior_pilot_receipt_pin_v2:
        raise ValueError("behavior receipt verification requires its explicit V2 pin")
    pin.validate()
    if hashlib.sha256(payload).hexdigest() != pin.expected_receipt_sha256:
        raise ValueError("behavior receipt bytes differ from the independently pinned SHA")
    try:
        receipt = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("behavior receipt must be valid canonical UTF-8 JSON") from error
    if type(receipt) is not dict or canonical_json_bytes(receipt) != payload:
        raise ValueError("behavior receipt must be one canonical JSON object")
    _require_exact_fields(receipt, _RECEIPT_FIELDS, "root")
    if receipt["schema"] != BEHAVIOR_PILOT_RECEIPT_SCHEMA_V2:
        raise ValueError("receipt is not a TrackBBehaviorPilotReceiptV2")

    for field_name, expected in (
        ("source_commit", pin.expected_source_commit),
        ("source_tree", pin.expected_source_tree),
        ("stage_plan_sha256", pin.expected_stage_plan_sha256),
        ("dataset_receipt_sha256", pin.expected_dataset_receipt_sha256),
        ("recipe_sha256", pin.expected_recipe_sha256),
    ):
        if receipt[field_name] != expected:
            raise ValueError(f"behavior receipt {field_name} differs from its independent pin")
    if type(receipt["completed_optimizer_updates"]) is not int or receipt[
        "completed_optimizer_updates"
    ] != BEHAVIOR_COMPLETED_OPTIMIZER_UPDATES_V2:
        raise ValueError("behavior receipt must record exactly 191 completed optimizer updates")
    if (receipt["research_only"] is not True or receipt["selection_evidence"] is not False
            or receipt["promotion_ready"] is not False
            or receipt["hardware_action_authorized"] is not False
            or receipt["scientific_gate"] != "NOT_PERFORMED"):
        raise ValueError("behavior receipt violates the research-only authority boundary")
    if receipt["model_self_rollout"] != "NOT_PERFORMED_IN_TRAINING_PHASE":
        raise ValueError("behavior receipt has an unexpected training-phase rollout claim")

    recipe = _require_mapping(receipt["recipe"], "recipe")
    expected_recipe = dict(pin.admitted_recipe)
    _require_exact_fields(recipe, frozenset(expected_recipe), "recipe")
    if (not _same_typed_value(recipe, expected_recipe)
            or recipe.get("schema") != BEHAVIOR_PILOT_RECIPE_SCHEMA_V2):
        raise ValueError("behavior receipt recipe is not the exactly admitted V2 recipe")
    if (recipe.get("history_policy") != pin.admitted_policy_id
            or recipe.get("history_policy_sha256") != pin.admitted_policy_sha256):
        raise ValueError("behavior receipt does not bind the admitted V2 history policy")
    if (recipe.get("source_view_kind") != BEHAVIOR_SOURCE_VIEW_V2
            or recipe.get("severity_matching_kind") != BEHAVIOR_SEVERITY_MATCHING_V2):
        raise ValueError("behavior receipt uses an unsupported source view or severity matcher")
    _verify_training_schedule_v2(receipt, recipe)

    dataset_identity = _require_mapping(receipt["dataset_identity"], "dataset_identity")
    _require_exact_fields(dataset_identity, _DATASET_IDENTITY_FIELDS, "dataset_identity")
    if (dataset_identity.get("source_commit") != pin.expected_source_commit
            or dataset_identity.get("source_tree") != pin.expected_source_tree
            or type(dataset_identity.get("rows")) is not int or dataset_identity["rows"] != 1200):
        raise ValueError("behavior receipt dataset identity is not the pinned 1,000/200 source")
    shard_hashes = _require_mapping(dataset_identity["shard_hashes"], "dataset_identity.shard_hashes")
    if (set(shard_hashes) != {
            "observations/synthetic/trackb_feedback_history_v2_n4_train.h5",
            "observations/synthetic/trackb_feedback_history_v2_n4_validation.h5",
            "truth/synthetic/trackb_feedback_history_v2_n4_train.h5",
            "truth/synthetic/trackb_feedback_history_v2_n4_validation.h5",
        } or any(type(value) is not str or not _SHA256_PATTERN.fullmatch(value)
                 for value in shard_hashes.values())):
        raise ValueError("behavior receipt dataset shard metadata has unsupported names or hashes")
    for split_name in ("train_source", "validation_source"):
        source = _require_mapping(receipt[split_name], split_name)
        _require_exact_fields(source, _SOURCE_FIELDS, split_name)
        if any(not _same_typed_value(source[name], dataset_identity[name])
               for name in _DATASET_SOURCE_BINDING_FIELDS):
            raise ValueError(f"behavior receipt {split_name} differs from dataset identity")
        if (source.get("source_commit") != pin.expected_source_commit
                or source.get("source_tree") != pin.expected_source_tree
                or source.get("cell_id") != recipe["cell_id"]
                or source.get("history_policy") != recipe["history_policy"]
                or source.get("history_policy_sha256") != recipe["history_policy_sha256"]
                or type(source.get("sample_set_sha")) is not str
                or not _SHA256_PATTERN.fullmatch(source["sample_set_sha"])
                or source.get("history_policy") != pin.admitted_policy_id
                or source.get("history_policy_sha256") != pin.admitted_policy_sha256
                or source.get("split") != ("train" if split_name == "train_source" else "validation")):
            raise ValueError(f"behavior receipt {split_name} is not bound to the pinned V2 source and policy")

    model = _require_mapping(receipt["model"], "model")
    _require_exact_fields(model, frozenset({"config", "target_normalization", "checkpoint",
                                            "checkpoint_roundtrip_verified"}), "model")
    checkpoint = _require_mapping(model["checkpoint"], "model.checkpoint")
    _require_exact_fields(checkpoint, _CHECKPOINT_FIELDS, "model.checkpoint")
    if checkpoint.get("sha256") != pin.expected_checkpoint_sha256:
        raise ValueError("behavior receipt checkpoint SHA differs from its independent pin")
    if model["checkpoint_roundtrip_verified"] is not True:
        raise ValueError("behavior receipt does not record checkpoint roundtrip verification")
    config = _require_mapping(model["config"], "model.config")
    _require_exact_fields(config, _MODEL_CONFIG_FIELDS, "model.config")
    expected_config = {
        "schema": BEHAVIOR_MODEL_CONFIG_SCHEMA_V1,
        "system_type": "4qubit", "mode_count": 4, "width": 64, "heads": 4, "blocks": 2,
        "history_gain_input_kind": "unobserved_zero_v1",
        "gain_parameterization": recipe["gain_parameterization"],
        "action_projection_kind": recipe["action_projection_kind"],
        "action_value_feature_kind": recipe["action_value_feature_kind"],
    }
    for name, expected in expected_config.items():
        if not _same_typed_value(config.get(name), expected):
            raise ValueError(f"behavior receipt model config differs at {name}")

    return receipt


__all__ = [
    "Behavior_pilot_receipt_pin_v2",
    "verify_behavior_pilot_receipt_bytes_v2",
]
