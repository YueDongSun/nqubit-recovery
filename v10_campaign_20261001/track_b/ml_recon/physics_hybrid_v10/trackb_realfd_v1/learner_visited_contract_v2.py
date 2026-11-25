"""Pure V2 learner-row integrity contract, separate from frozen V1 rows."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from ml_recon.data_v2_4.hashing import sha256_array
from ml_recon.data_v2_4.synthetic.feedback_renderers import (
    CONTROL_AMPLITUDE_BOUND,
    CONTROL_PHASE_BOUND,
)
from ml_recon.data_v2_4.synthetic.trackb_teacher_policy import (
    TRACKB_TEACHER_POLICY_V2,
    TRACKB_TEACHER_POLICY_V2_SHA256,
)

from .contracts import canonical_json_bytes, canonical_sha256, require_git_oid, require_sha256
from .learner_stage_contract_v2 import (
    LEARNER_HISTORY_KIND_V2,
    LEARNER_LABEL_KIND_V2,
    LEARNER_TEACHER_GAIN_KIND_V2,
    LEARNER_VISITED_ROW_SCHEMA_V2,
)


ROW_TENSOR_NAMES_V2 = (
    "history_tokens", "history_mask", "noisy_measurement_tokens",
    "target_mode_tokens", "mode_mask", "teacher_action_tokens",
    "teacher_transition_gain",
)
ROW_IDENTITY_FIELDS_V2 = frozenset({
    "schema", "sample_id", "split", "sample_key_sha",
    "behavior_pilot_receipt_sha256", "checkpoint_sha256",
    "behavior_source_commit", "behavior_source_tree", "behavior_stage_plan_sha256",
    "behavior_dataset_receipt_sha256", "behavior_model_config_sha256",
    "behavior_model_state_sha256", "recipe_sha256", "learner_view_plan_sha256",
    "history_policy", "history_policy_sha256", "initial_context_sha256",
    "history_kind", "label_kind", "teacher_gain_kind",
    *(f"{name}_sha" for name in ROW_TENSOR_NAMES_V2),
    "teacher_pre_state_sha_by_round", "executed_action_sha_by_round",
    "observation_record_ids", "action_execution", "research_only",
    "selection_evidence", "promotion_ready", "hardware_action_authorized",
    "deployment_validation", "held_out",
})
_action_execution_v2 = "FLOAT32_COMMON_MODEL_ZERO_TEACHER_V1"


def validate_learner_visited_tensors_v2(
    identity_json: str,
    row_sha: str,
    tensors: Mapping[str, Any],
) -> dict[str, Any]:
    """Rehash one complete V2 row and enforce its physical/action tensor shape."""
    if type(identity_json) is not str:
        raise ValueError("learner-visited V2 identity must be canonical JSON text")
    try:
        identity = json.loads(identity_json)
    except json.JSONDecodeError as error:
        raise ValueError("learner-visited V2 identity is not JSON") from error
    if (type(identity) is not dict or set(identity) != ROW_IDENTITY_FIELDS_V2
            or canonical_json_bytes(identity).decode("utf-8") != identity_json
            or canonical_sha256(identity) != require_sha256(row_sha, "learner_visited_row_sha_v2")):
        raise ValueError("learner-visited V2 canonical identity or row hash differs")
    if (identity["schema"] != LEARNER_VISITED_ROW_SCHEMA_V2
            or identity["split"] not in ("train", "validation")
            or type(identity["sample_id"]) is not str or not identity["sample_id"]
            or identity["history_kind"] != LEARNER_HISTORY_KIND_V2
            or identity["label_kind"] != LEARNER_LABEL_KIND_V2
            or identity["teacher_gain_kind"] != LEARNER_TEACHER_GAIN_KIND_V2
            or identity["history_policy"] != TRACKB_TEACHER_POLICY_V2
            or identity["history_policy_sha256"] != TRACKB_TEACHER_POLICY_V2_SHA256
            or identity["action_execution"] != _action_execution_v2
            or identity["research_only"] is not True
            or identity["selection_evidence"] is not False
            or identity["promotion_ready"] is not False
            or identity["hardware_action_authorized"] is not False
            or identity["deployment_validation"] is not False
            or identity["held_out"] is not False):
        raise ValueError("learner-visited V2 schema, labels, or authority differs")
    for name in (
        "sample_key_sha", "behavior_pilot_receipt_sha256", "checkpoint_sha256",
        "behavior_stage_plan_sha256", "behavior_dataset_receipt_sha256",
        "behavior_model_config_sha256", "behavior_model_state_sha256", "recipe_sha256",
        "learner_view_plan_sha256", "history_policy_sha256", "initial_context_sha256",
    ):
        require_sha256(identity[name], name)
    for name in ("behavior_source_commit", "behavior_source_tree"):
        require_git_oid(identity[name], name)
    for name, count in (
        ("teacher_pre_state_sha_by_round", 10),
        ("executed_action_sha_by_round", 10),
        ("observation_record_ids", 11),
    ):
        values = identity[name]
        if type(values) is not list or len(values) != count:
            raise ValueError(f"learner-visited V2 {name} has the wrong length")
        for value in values:
            require_sha256(value, name)
    if len(set(identity["observation_record_ids"])) != 11:
        raise ValueError("learner-visited V2 observation ids must be unique")
    if type(tensors) not in (dict,):
        raise ValueError("learner-visited V2 tensors must be an exact mapping")
    if set(tensors) != set(ROW_TENSOR_NAMES_V2):
        raise ValueError("learner-visited V2 tensors are incomplete or unknown")
    arrays = {name: np.asarray(tensors[name]) for name in ROW_TENSOR_NAMES_V2}
    history = arrays["history_tokens"]
    if history.ndim != 3 or history.shape != (10, 4, 7):
        raise ValueError("learner-visited V2 history must be ten complete 4Q rounds")
    expected_shapes = {
        "history_tokens": (10, 4, 7),
        "history_mask": (10,),
        "noisy_measurement_tokens": (4, 2),
        "target_mode_tokens": (4, 2),
        "mode_mask": (4,),
        "teacher_action_tokens": (10, 4, 2),
        "teacher_transition_gain": (10,),
    }
    for name, array in arrays.items():
        expected_dtype = np.dtype(np.bool_ if name in ("history_mask", "mode_mask") else np.float32)
        if array.shape != expected_shapes[name] or array.dtype != expected_dtype:
            raise ValueError(f"learner-visited V2 {name} shape or dtype differs")
        if name not in ("history_mask", "mode_mask") and not np.isfinite(array).all():
            raise ValueError(f"learner-visited V2 {name} contains non-finite values")
        if identity[f"{name}_sha"] != sha256_array(array):
            raise ValueError(f"learner-visited V2 {name} differs from row identity")
    if (not arrays["history_mask"].all() or not arrays["mode_mask"].all()
            or not np.array_equal(arrays["noisy_measurement_tokens"], history[0, :, :2])
            or not np.array_equal(history[:-1, :, 4:6], history[1:, :, :2])
            or not np.all(history[..., 6] == 0.0)):
        raise ValueError("learner-visited V2 observable history contract differs")
    bounds = np.asarray((CONTROL_AMPLITUDE_BOUND, CONTROL_PHASE_BOUND), dtype=np.float64)
    for name, actions in (
        ("executed", history[..., 2:4]),
        ("teacher", arrays["teacher_action_tokens"]),
    ):
        if (np.any(np.abs(actions) > bounds)
                or not np.allclose(actions.sum(axis=1), 0.0, atol=1e-6, rtol=0)):
            raise ValueError(f"learner-visited V2 {name} action violates gauge or bounds")
    for round_index, action_sha in enumerate(identity["executed_action_sha_by_round"]):
        if action_sha != sha256_array(history[round_index, :, 2:4].astype(np.float64)):
            raise ValueError("learner-visited V2 executed action identity differs")
    return identity


__all__ = [
    "ROW_TENSOR_NAMES_V2",
    "ROW_IDENTITY_FIELDS_V2",
    "validate_learner_visited_tensors_v2",
]
