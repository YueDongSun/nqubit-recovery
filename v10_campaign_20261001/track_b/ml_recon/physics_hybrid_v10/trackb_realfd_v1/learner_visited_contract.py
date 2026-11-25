"""Pure integrity contract for derived learner-visited Track B rows."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from ml_recon.data_v2_4.hashing import sha256_array
from ml_recon.data_v2_4.synthetic.feedback_renderers import (
    CONTROL_AMPLITUDE_BOUND,
    CONTROL_PHASE_BOUND,
)

from .contracts import (
    LEARNER_VISITED_ROW_SCHEMA,
    canonical_json_bytes,
    canonical_sha256,
    require_sha256,
)


LEARNER_HISTORY_KIND = "MODEL_BRANCH_EXECUTED_ACTIONS_UNOBSERVED_ZERO_GAIN_V1"
LEARNER_LABEL_KIND = "TEACHER_ACTION_AT_LEARNER_PRE_STATE_V1"
TEACHER_GAIN_KIND = "TEACHER_ACTION_TRANSITION_GAIN_NOT_MODEL_ACTION_GAIN"
ROW_TENSOR_NAMES = (
    "history_tokens", "history_mask", "noisy_measurement_tokens",
    "target_mode_tokens", "mode_mask", "teacher_action_tokens",
    "teacher_transition_gain",
)
ROW_IDENTITY_FIELDS = frozenset({
    "schema", "sample_id", "split", "sample_key_sha", "round_zero_binding_sha",
    "history_kind", "label_kind", "teacher_gain_kind",
    *(f"{name}_sha" for name in ROW_TENSOR_NAMES),
    "teacher_pre_state_sha_by_round", "executed_action_sha_by_round",
    "observation_record_ids", "research_only", "selection_evidence",
    "promotion_ready", "hardware_action_authorized", "deployment_validation",
    "held_out",
})


def validate_learner_visited_tensors(
    identity_json: str,
    row_sha: str,
    tensors: Mapping[str, Any],
) -> dict[str, Any]:
    """Rehash every tensor and require complete ten-round row semantics."""
    if type(identity_json) is not str:
        raise ValueError("learner-visited row identity must be canonical JSON text")
    try:
        identity = json.loads(identity_json)
    except json.JSONDecodeError as error:
        raise ValueError("learner-visited row identity is not JSON") from error
    if (type(identity) is not dict or set(identity) != ROW_IDENTITY_FIELDS
            or canonical_json_bytes(identity).decode("utf-8") != identity_json
            or canonical_sha256(identity) != require_sha256(row_sha, "learner_visited_row_sha")):
        raise ValueError("learner-visited row canonical identity or hash differs")
    if (identity["schema"] != LEARNER_VISITED_ROW_SCHEMA
            or identity["split"] not in ("train", "validation")
            or type(identity["sample_id"]) is not str or not identity["sample_id"]
            or identity["history_kind"] != LEARNER_HISTORY_KIND
            or identity["label_kind"] != LEARNER_LABEL_KIND
            or identity["teacher_gain_kind"] != TEACHER_GAIN_KIND
            or identity["research_only"] is not True
            or identity["selection_evidence"] is not False
            or identity["promotion_ready"] is not False
            or identity["hardware_action_authorized"] is not False
            or identity["deployment_validation"] is not False
            or identity["held_out"] is not False):
        raise ValueError("learner-visited row schema, labels or authority differs")
    for name in ("sample_key_sha", "round_zero_binding_sha"):
        require_sha256(identity[name], name)
    for name, count in (
        ("teacher_pre_state_sha_by_round", 10),
        ("executed_action_sha_by_round", 10),
        ("observation_record_ids", 11),
    ):
        values = identity[name]
        if type(values) is not list or len(values) != count:
            raise ValueError(f"learner-visited {name} has the wrong length")
        for value in values:
            require_sha256(value, name)
    if set(tensors) != set(ROW_TENSOR_NAMES):
        raise ValueError("learner-visited row tensors are incomplete or unknown")
    arrays = {name: np.asarray(tensors[name]) for name in ROW_TENSOR_NAMES}
    history = arrays["history_tokens"]
    if history.ndim != 3 or history.shape[0] != 10 or history.shape[1] not in (3, 4) or history.shape[2] != 7:
        raise ValueError("learner-visited history shape is invalid")
    mode_count = history.shape[1]
    expected_shapes = {
        "history_tokens": (10, mode_count, 7),
        "history_mask": (10,),
        "noisy_measurement_tokens": (mode_count, 2),
        "target_mode_tokens": (mode_count, 2),
        "mode_mask": (mode_count,),
        "teacher_action_tokens": (10, mode_count, 2),
        "teacher_transition_gain": (10,),
    }
    for name, array in arrays.items():
        expected_dtype = np.dtype(np.bool_ if name in ("history_mask", "mode_mask") else np.float32)
        if array.shape != expected_shapes[name] or array.dtype != expected_dtype:
            raise ValueError(f"learner-visited {name} shape or dtype differs")
        if name not in ("history_mask", "mode_mask") and not np.isfinite(array).all():
            raise ValueError(f"learner-visited {name} contains non-finite values")
        if identity[f"{name}_sha"] != sha256_array(array):
            raise ValueError(f"learner-visited {name} differs from row identity")
    if (not arrays["history_mask"].all() or not arrays["mode_mask"].all()
            or not np.array_equal(arrays["noisy_measurement_tokens"], history[0, :, :2])
            or not np.array_equal(history[:-1, :, 4:6], history[1:, :, :2])
            or not np.all(history[..., 6] == 0)):
        raise ValueError("learner-visited observable history contract differs")
    bounds = np.asarray((CONTROL_AMPLITUDE_BOUND, CONTROL_PHASE_BOUND), dtype=np.float64)
    for name, actions in (
        ("executed", history[..., 2:4]),
        ("teacher", arrays["teacher_action_tokens"]),
    ):
        if (np.any(np.abs(actions) > bounds)
                or not np.allclose(actions.sum(axis=1), 0.0, atol=1e-6, rtol=0)):
            raise ValueError(f"learner-visited {name} action violates gauge or bounds")
    for round_index, action_sha in enumerate(identity["executed_action_sha_by_round"]):
        if action_sha != sha256_array(history[round_index, :, 2:4].astype(np.float64)):
            raise ValueError("learner-visited executed action identity differs")
    return identity


__all__ = [
    "LEARNER_HISTORY_KIND", "LEARNER_LABEL_KIND", "TEACHER_GAIN_KIND",
    "ROW_TENSOR_NAMES", "validate_learner_visited_tensors",
]
