"""Derive observable Track B histories and oracle labels at learner states.

The registered synthetic state stays inside this label producer. Historical
tokens contain actions actually executed by the learner; oracle actions and
teacher-transition gains are separate supervision targets.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import math
from typing import Any, Mapping

import numpy as np

from ml_recon.data_v2_4.hashing import sha256_array
from ml_recon.data_v2_4.sample_keys import V24SampleKey
from ml_recon.data_v2_4.synthetic.feedback_renderers import _trackb_receding_horizon_action

from .contracts import LEARNER_VISITED_ROW_SCHEMA, canonical_json_bytes, canonical_sha256
from .learner_visited_contract import (
    LEARNER_HISTORY_KIND,
    LEARNER_LABEL_KIND,
    ROW_TENSOR_NAMES,
    TEACHER_GAIN_KIND,
    validate_learner_visited_tensors,
)
from .self_rollout_evaluator import (
    Self_rollout_config_v1,
    _bind_round_zero,
    _score_transition,
    _validate_action,
    round_count,
)
from .self_rollout_observation import branch_kinds


def _required_array(value: Any, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"learner-visited {label} has invalid shape or values")
    return array


def _required_float(value: Any, *, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"learner-visited {label} must be finite")
    return float(value)


def derive_learner_visited_row(
    sample: Mapping[str, Any],
    sample_key: V24SampleKey,
    *,
    config: Self_rollout_config_v1,
) -> dict[str, Any]:
    """Recompute student-state teacher targets from one bound synthetic rollout."""
    config.validate()
    if (sample.get("sample_id") != sample_key.sample_id
            or sample.get("sample_key_sha") != sample_key.sha256()
            or sample.get("replicate_index") != config.replicate_index
            or sample.get("pairing_kind") != config.pairing_kind
            or not isinstance(sample.get("config"), Mapping)
            or canonical_sha256(sample["config"]) != canonical_sha256(asdict(config))):
        raise ValueError("learner-visited row differs from its sample/config identity")
    if (sample.get("research_only") is not True
            or sample.get("selection_evidence") is not False
            or sample.get("promotion_ready") is not False
            or sample.get("hardware_action_authorized") is not False):
        raise ValueError("learner-visited row violates research-only authority")
    branches = sample.get("branches")
    if (not isinstance(branches, (tuple, list))
            or [branch.get("branch_kind") for branch in branches if isinstance(branch, Mapping)]
            != list(branch_kinds)):
        raise ValueError("learner-visited row lacks the registered branch order")
    model_branch = branches[0]
    steps = model_branch.get("steps")
    record_ids = model_branch.get("observation_record_ids")
    records = model_branch.get("observation_records")
    if (not isinstance(steps, (tuple, list)) or len(steps) != round_count
            or not isinstance(record_ids, (tuple, list)) or len(record_ids) != round_count + 1
            or not isinstance(records, (tuple, list)) or len(records) != round_count + 1):
        raise ValueError("learner-visited model branch lacks ten complete steps")
    context, binding_sha = _bind_round_zero(sample_key, None, config)
    if sample.get("round_zero_binding_sha") != binding_sha:
        raise ValueError("learner-visited round-zero binding differs from registered replay")
    mode_count = len(context.before_coefficients)
    record_tokens: list[np.ndarray] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("record_id") != record_ids[record_index]:
            raise ValueError("learner-visited acquired observation record identity differs")
        payload = {name: value for name, value in record.items() if name != "record_id"}
        tokens = _required_array(
            record.get("measurement_tokens"), shape=(mode_count, 2), label="acquired observation",
        ).astype(np.float32)
        if (record.get("record_id") != canonical_sha256(payload)
                or record.get("measurement_token_sha") != sha256_array(tokens)):
            raise ValueError("learner-visited acquired observation record bytes differ")
        record_tokens.append(tokens)
    target = context.target_coefficients
    target_tokens = np.stack((
        np.abs(target), np.angle(target * target[0].conj()) / np.pi,
    ), axis=-1).astype(np.float32)
    history_records: list[np.ndarray] = []
    teacher_actions: list[np.ndarray] = []
    teacher_gains: list[np.float32] = []
    teacher_pre_state_shas: list[str] = []
    executed_action_shas: list[str] = []
    previous_post_observation: np.ndarray | None = None
    for round_index, step in enumerate(steps):
        if not isinstance(step, Mapping) or step.get("round_number") != round_index + 1:
            raise ValueError("learner-visited step order differs from ten-round replay")
        if (step.get("pre_state_sha") != context.pre_state_sha
                or step.get("pre_observation_record_id") != record_ids[round_index]
                or step.get("post_observation_record_id") != record_ids[round_index + 1]):
            raise ValueError("learner-visited step state or observation lineage differs")
        pre_observation = _required_array(
            step.get("pre_observation"), shape=(mode_count, 2), label="pre-observation")
        post_observation = _required_array(
            step.get("post_observation"), shape=(mode_count, 2), label="post-observation")
        pre_observation = pre_observation.astype(np.float32)
        post_observation = post_observation.astype(np.float32)
        if (step.get("pre_observation_sha") != sha256_array(pre_observation)
                or step.get("post_observation_sha") != sha256_array(post_observation)
                or not np.array_equal(pre_observation, record_tokens[round_index])
                or not np.array_equal(post_observation, record_tokens[round_index + 1])
                or (previous_post_observation is not None
                    and not np.array_equal(pre_observation, previous_post_observation))):
            raise ValueError("learner-visited observation bytes or chain differ")
        executed_action = _validate_action(
            _required_array(step.get("action"), shape=(mode_count, 2), label="executed action"),
            mode_count, config)
        if step.get("action_sha") != sha256_array(executed_action):
            raise ValueError("learner-visited executed action bytes differ")
        teacher_action, _ = _trackb_receding_horizon_action(
            context.before_coefficients, context.target_coefficients)
        teacher_action = _validate_action(teacher_action, mode_count, config)
        _teacher_post, _teacher_before, _teacher_after, teacher_gain = _score_transition(
            context, teacher_action)
        post_context, before_distance, after_distance, executed_gain = _score_transition(
            context, executed_action)
        if (step.get("post_state_sha") != post_context.pre_state_sha
                or not math.isclose(_required_float(step.get("before_distance"), label="before distance"), before_distance,
                                    abs_tol=1e-12, rel_tol=1e-10)
                or not math.isclose(_required_float(step.get("after_distance"), label="after distance"), after_distance,
                                    abs_tol=1e-12, rel_tol=1e-10)
                or not math.isclose(_required_float(step.get("synthetic_gain"), label="executed gain"), executed_gain,
                                    abs_tol=1e-12, rel_tol=1e-10)):
            raise ValueError("learner-visited executed transition differs from registered forward")
        history_records.append(np.concatenate((
            pre_observation, executed_action.astype(np.float32), post_observation,
            np.zeros((mode_count, 1), dtype=np.float32),
        ), axis=-1))
        teacher_actions.append(teacher_action.astype(np.float32))
        teacher_gains.append(np.float32(teacher_gain))
        teacher_pre_state_shas.append(context.pre_state_sha)
        executed_action_shas.append(step["action_sha"])
        previous_post_observation = post_observation
        context = replace(post_context, round_index=min(round_index + 1, round_count - 1))
    history_tokens = np.stack(history_records)
    teacher_action_tokens = np.stack(teacher_actions)
    teacher_transition_gain = np.asarray(teacher_gains, dtype=np.float32)
    history_mask = np.ones(round_count, dtype=np.bool_)
    noisy_measurement_tokens = history_tokens[0, :, :2].copy()
    mode_mask = np.ones(mode_count, dtype=np.bool_)
    row_identity = {
        "schema": LEARNER_VISITED_ROW_SCHEMA,
        "sample_id": sample_key.sample_id,
        "split": sample_key.split,
        "sample_key_sha": sample_key.sha256(),
        "round_zero_binding_sha": binding_sha,
        "history_kind": LEARNER_HISTORY_KIND,
        "label_kind": LEARNER_LABEL_KIND,
        "teacher_gain_kind": TEACHER_GAIN_KIND,
        "history_tokens_sha": sha256_array(history_tokens),
        "history_mask_sha": sha256_array(history_mask),
        "noisy_measurement_tokens_sha": sha256_array(noisy_measurement_tokens),
        "target_mode_tokens_sha": sha256_array(target_tokens),
        "mode_mask_sha": sha256_array(mode_mask),
        "teacher_action_tokens_sha": sha256_array(teacher_action_tokens),
        "teacher_transition_gain_sha": sha256_array(teacher_transition_gain),
        "teacher_pre_state_sha_by_round": teacher_pre_state_shas,
        "executed_action_sha_by_round": executed_action_shas,
        "observation_record_ids": list(record_ids),
        "research_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
        "deployment_validation": False,
        "held_out": False,
    }
    row = {
        **row_identity,
        "row_sha": canonical_sha256(row_identity),
        "row_identity_json": canonical_json_bytes(row_identity).decode("utf-8"),
        "history_tokens": history_tokens,
        "history_mask": history_mask,
        "noisy_measurement_tokens": noisy_measurement_tokens,
        "target_mode_tokens": target_tokens,
        "mode_mask": mode_mask,
        "teacher_action_tokens": teacher_action_tokens,
        "teacher_transition_gain": teacher_transition_gain,
    }
    validate_learner_visited_tensors(
        row["row_identity_json"], row["row_sha"],
        {name: row[name] for name in ROW_TENSOR_NAMES},
    )
    return row


__all__ = [
    "LEARNER_VISITED_ROW_SCHEMA",
    "LEARNER_HISTORY_KIND",
    "LEARNER_LABEL_KIND",
    "TEACHER_GAIN_KIND",
    "derive_learner_visited_row",
]
