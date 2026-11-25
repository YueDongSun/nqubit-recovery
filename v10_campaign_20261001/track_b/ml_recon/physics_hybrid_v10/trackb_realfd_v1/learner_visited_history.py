"""Derive observable Track B histories and oracle labels at learner states.

The registered synthetic state stays inside this label producer. Historical
tokens contain actions actually executed by the learner; oracle actions and
teacher-transition gains are separate supervision targets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from threading import RLock
from typing import Any, Mapping
from weakref import WeakKeyDictionary

import numpy as np

from ml_recon.data_v2_4.hashing import sha256_array
from ml_recon.data_v2_4.sample_keys import V24SampleKey
from ml_recon.data_v2_4.synthetic.feedback_renderers import _trackb_receding_horizon_action
from ml_recon.data_v2_4.synthetic.trackb_phase_lift import trackb_receding_horizon_action_v2
from ml_recon.data_v2_4.synthetic.trackb_teacher_policy import (
    TRACKB_TEACHER_POLICY_V2,
    TRACKB_TEACHER_POLICY_V2_SHA256,
)

from .contracts import LEARNER_VISITED_ROW_SCHEMA, canonical_json_bytes, canonical_sha256, require_sha256
from .behavior_checkpoint_reader_v2 import (
    Behavior_checkpoint_model_v2,
    verify_loaded_behavior_model_v2,
)
from .learner_visited_contract import (
    LEARNER_HISTORY_KIND,
    LEARNER_LABEL_KIND,
    ROW_TENSOR_NAMES,
    TEACHER_GAIN_KIND,
    validate_learner_visited_tensors,
)
from .learner_visited_contract_v2 import (
    ROW_TENSOR_NAMES_V2,
    validate_learner_visited_tensors_v2,
)
from .learner_stage_contract_v2 import (
    LEARNER_HISTORY_KIND_V2,
    LEARNER_LABEL_KIND_V2,
    LEARNER_TEACHER_GAIN_KIND_V2,
    LEARNER_VISITED_ROW_SCHEMA_V2,
)
from .self_rollout_evaluator import (
    Rollout_branch_v1,
    Self_rollout_config_v1,
    _bind_round_zero,
    _score_transition,
    _validate_action,
    round_count,
)
from .self_rollout_observation import branch_kinds
from .behavior_v2_contract import Behavior_recipe_v2
from .behavior_self_rollout_v2 import replay_behavior_model_branch_v2
from .teacher_diagnostic_v2 import _context_identity as _context_identity_v2
from .teacher_diagnostic_v2 import _render_initial_context as _render_initial_context_v2


TEACHER_GAIN_KIND_V2 = LEARNER_TEACHER_GAIN_KIND_V2
BEHAVIOR_SAMPLE_SCHEMA_V2 = "TrackBBehaviorSelfRolloutV2"
BEHAVIOR_ACTION_EXECUTION_V2 = "FLOAT32_COMMON_MODEL_ZERO_TEACHER_V1"


@dataclass(frozen=True, eq=False)
class Behavior_rollout_binding_v2:
    """One-branch replay evidence verified against a module-owned issuance record."""

    sample_key: V24SampleKey
    recipe: Behavior_recipe_v2
    checkpoint_model: Behavior_checkpoint_model_v2
    learner_view_plan_sha256: str
    initial_context_sha256: str
    replicate_index: int
    model_branch: Rollout_branch_v1
    model_branch_sha256: str


_behavior_binding_lock = RLock()
_issued_behavior_bindings: WeakKeyDictionary[
    Behavior_rollout_binding_v2,
    tuple[tuple[type, bytes], tuple[type, bytes], Behavior_checkpoint_model_v2,
          str, str, tuple[type, int], tuple[type, bytes], str],
] = WeakKeyDictionary()


def _behavior_binding_snapshot(
    binding: Behavior_rollout_binding_v2,
) -> tuple[tuple[type, bytes], tuple[type, bytes], Behavior_checkpoint_model_v2,
           str, str, tuple[type, int], tuple[type, bytes], str]:
    return (
        (type(binding.sample_key), canonical_json_bytes(asdict(binding.sample_key))),
        (type(binding.recipe), canonical_json_bytes(asdict(binding.recipe))),
        binding.checkpoint_model,
        binding.learner_view_plan_sha256,
        binding.initial_context_sha256,
        (type(binding.replicate_index), binding.replicate_index),
        (type(binding.model_branch), canonical_json_bytes(asdict(binding.model_branch))),
        binding.model_branch_sha256,
    )


def _issue_behavior_binding_v2(
    sample_key: V24SampleKey,
    recipe: Behavior_recipe_v2,
    checkpoint_model: Behavior_checkpoint_model_v2,
    learner_view_plan_sha256: str,
    initial_context_sha256: str,
    replicate_index: int,
    model_branch: Rollout_branch_v1,
    model_branch_sha256: str,
) -> Behavior_rollout_binding_v2:
    binding = Behavior_rollout_binding_v2(
        sample_key, recipe, checkpoint_model, learner_view_plan_sha256,
        initial_context_sha256, replicate_index, model_branch, model_branch_sha256,
    )
    with _behavior_binding_lock:
        _issued_behavior_bindings[binding] = _behavior_binding_snapshot(binding)
    return binding


def _require_issued_behavior_binding_v2(binding: object) -> Behavior_rollout_binding_v2:
    if type(binding) is not Behavior_rollout_binding_v2:
        raise ValueError("learner-visited V2 requires a replay-verified behavior binding")
    with _behavior_binding_lock:
        issued_snapshot = _issued_behavior_bindings.get(binding)
    if issued_snapshot is None or _behavior_binding_snapshot(binding) != issued_snapshot:
        raise ValueError("learner-visited V2 requires an unchanged replay-verified behavior binding")
    return binding


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


def bind_behavior_rollout_v2(
    sample: Mapping[str, Any],
    sample_key: V24SampleKey,
    *,
    recipe: Behavior_recipe_v2,
    checkpoint_model: Behavior_checkpoint_model_v2,
    learner_view_plan_sha256: str,
) -> Behavior_rollout_binding_v2:
    """Replay one admitted checkpoint's model branch before binding a stored unit."""
    if not isinstance(sample, Mapping):
        raise ValueError("learner-visited V2 sample must be an object")
    if type(recipe) is not Behavior_recipe_v2:
        raise ValueError("learner-visited V2 requires the fixed Behavior_recipe_v2")
    if type(sample_key) is not V24SampleKey or sample_key.split != "validation":
        raise ValueError("stored Stage2 behavior units bind registered validation keys only")
    recipe.validate()
    config = recipe.teacher_config()
    config.validate(sample_key)
    require_sha256(learner_view_plan_sha256, "learner_view_plan_sha256")
    verify_loaded_behavior_model_v2(checkpoint_model)
    evidence = checkpoint_model.evidence
    if evidence.recipe_sha256 != recipe.sha256():
        raise ValueError("loaded behavior checkpoint recipe differs from the learner-view recipe")
    if (recipe.history_policy != TRACKB_TEACHER_POLICY_V2
            or recipe.history_policy_sha256 != TRACKB_TEACHER_POLICY_V2_SHA256):
        raise ValueError("learner-visited V2 policy identity differs from registered V2")
    if (sample.get("schema") != BEHAVIOR_SAMPLE_SCHEMA_V2
            or canonical_json_bytes(sample.get("sample_key")) != canonical_json_bytes(asdict(sample_key))
            or sample.get("recipe_sha256") != recipe.sha256()
            or sample.get("target_source") != "NORMALIZED_RENDERED_TARGET_MODE_TOKENS_CONVERTED_TO_FLOAT64"
            or sample.get("pairing_kind") != "common_random_variates_v1"
            or sample.get("history_distribution_shift") != recipe.history_distribution_shift
            or sample.get("action_execution") != BEHAVIOR_ACTION_EXECUTION_V2
            or sample.get("actions_per_branch") != round_count
            or sample.get("observation_windows_per_branch") != round_count + 1
            or sample.get("incidental_renderer_observations_used") != 0
            or sample.get("research_only") is not True
            or sample.get("selection_evidence") is not False
            or sample.get("promotion_ready") is not False
            or sample.get("hardware_action_authorized") is not False
            or sample.get("scientific_gate") != "NOT_PERFORMED"
            or sample.get("execution_admission") != "NOT_GRANTED_IN_MEMORY_KERNEL_ONLY"):
        raise ValueError("learner-visited V2 sample schema, recipe, or authority differs")
    if (sample.get("model_config_sha256") != evidence.model_config_sha256
            or type(sample.get("replicate_index")) is not int
            or sample.get("replicate_index") != config.replicate_index):
        raise ValueError("learner-visited V2 sample model or replicate identity differs from reader evidence")
    context = _render_initial_context_v2(sample_key, config)
    initial_context_sha = _context_identity_v2(context)
    if sample.get("initial_context_sha256") != initial_context_sha:
        raise ValueError("learner-visited V2 initial context differs from registered rendering")

    branches = sample.get("branches")
    if (not isinstance(branches, (tuple, list)) or len(branches) != 3
            or {branch.get("branch_kind") for branch in branches if isinstance(branch, Mapping)}
            != {"model", "zero", "teacher_v2"}):
        raise ValueError("learner-visited V2 requires the complete three-branch sample")
    model_branches = [branch for branch in branches if isinstance(branch, Mapping)
                      and branch.get("branch_kind") == "model"]
    if len(model_branches) != 1:
        raise ValueError("learner-visited V2 model branch identity is missing or duplicated")

    replay = replay_behavior_model_branch_v2(
        checkpoint_model.model, sample_key, recipe=recipe, replicate_index=config.replicate_index,
    )
    if (replay.initial_context_sha256 != initial_context_sha
            or replay.model_config_sha256 != evidence.model_config_sha256
            or canonical_json_bytes(asdict(replay.branch)) != canonical_json_bytes(model_branches[0])):
        raise ValueError("stored learner rollout model branch differs from the pinned checkpoint replay")
    branch_sha256 = canonical_sha256(asdict(replay.branch))
    return _issue_behavior_binding_v2(
        sample_key, recipe, checkpoint_model, learner_view_plan_sha256,
        initial_context_sha, config.replicate_index, replay.branch, branch_sha256,
    )


def derive_learner_visited_row_v2(
    binding: Behavior_rollout_binding_v2,
) -> dict[str, Any]:
    """Derive V2 teacher labels from one reader-bound behavior model replay."""
    binding = _require_issued_behavior_binding_v2(binding)
    verify_loaded_behavior_model_v2(binding.checkpoint_model)
    recipe = binding.recipe
    sample_key = binding.sample_key
    if type(recipe) is not Behavior_recipe_v2 or sample_key.split not in ("train", "validation"):
        raise ValueError("learner-visited V2 binding is not a registered learner-view split")
    recipe.validate()
    config = recipe.teacher_config()
    config.validate(sample_key)
    require_sha256(binding.learner_view_plan_sha256, "learner_view_plan_sha256")
    evidence = binding.checkpoint_model.evidence
    context = _render_initial_context_v2(sample_key, config)
    initial_context_sha = _context_identity_v2(context)
    if (evidence.recipe_sha256 != recipe.sha256()
            or initial_context_sha != binding.initial_context_sha256
            or canonical_sha256(asdict(binding.model_branch)) != binding.model_branch_sha256):
        raise ValueError("learner-visited V2 replay binding changed after verification")
    model_branch = asdict(binding.model_branch)
    steps = model_branch.get("steps")
    records = model_branch.get("observation_records")
    record_ids = model_branch.get("observation_record_ids")
    if (not isinstance(steps, (tuple, list)) or len(steps) != round_count
            or not isinstance(records, (tuple, list)) or len(records) != round_count + 1
            or not isinstance(record_ids, (tuple, list)) or len(record_ids) != round_count + 1):
        raise ValueError("learner-visited V2 model branch lacks unique ten-action/eleven-observation chain")
    if (any(not isinstance(value, str) or not value for value in record_ids)
            or len(set(record_ids)) != round_count + 1):
        raise ValueError("learner-visited V2 observation record ids are invalid")

    mode_count = sample_key.N
    record_tokens: list[np.ndarray] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("record_id") != record_ids[record_index]:
            raise ValueError("learner-visited V2 observation record identity differs")
        payload = {name: value for name, value in record.items() if name != "record_id"}
        raw_tokens = _required_array(
            record.get("measurement_tokens"), shape=(mode_count, 2), label="V2 observation",
        )
        tokens = raw_tokens.astype(np.float32)
        if (record.get("record_id") != canonical_sha256(payload)
                or not np.array_equal(raw_tokens, tokens.astype(np.float64))
                or record.get("measurement_token_sha") != sha256_array(tokens)):
            raise ValueError("learner-visited V2 observation record bytes differ")
        record_tokens.append(tokens)

    target = context.target_coefficients
    target_tokens = np.stack((np.abs(target), np.angle(target * target[0].conj()) / np.pi), axis=-1).astype(np.float32)
    rollout_config = Self_rollout_config_v1(
        system_type=recipe.system_type, shot_mode=recipe.shot_mode,
        shot_budget=recipe.shot_budget, noise_family=recipe.noise_family,
        eta=recipe.eta, kappa=recipe.kappa, include_teacher_state_control=False,
    )
    history_records: list[np.ndarray] = []
    teacher_actions: list[np.ndarray] = []
    teacher_gains: list[np.float32] = []
    teacher_pre_state_shas: list[str] = []
    executed_action_shas: list[str] = []
    previous_post_observation: np.ndarray | None = None
    for round_index, step in enumerate(steps):
        if not isinstance(step, Mapping) or step.get("round_number") != round_index + 1:
            raise ValueError("learner-visited V2 step order differs from ten-round replay")
        if (step.get("pre_state_sha") != context.pre_state_sha
                or step.get("pre_observation_record_id") != record_ids[round_index]
                or step.get("post_observation_record_id") != record_ids[round_index + 1]):
            raise ValueError("learner-visited V2 state or observation lineage differs")
        raw_pre_observation = _required_array(
            step.get("pre_observation"), shape=(mode_count, 2), label="V2 pre-observation",
        )
        raw_post_observation = _required_array(
            step.get("post_observation"), shape=(mode_count, 2), label="V2 post-observation",
        )
        pre_observation = raw_pre_observation.astype(np.float32)
        post_observation = raw_post_observation.astype(np.float32)
        if (step.get("pre_observation_sha") != sha256_array(pre_observation)
                or step.get("post_observation_sha") != sha256_array(post_observation)
                or not np.array_equal(raw_pre_observation, pre_observation.astype(np.float64))
                or not np.array_equal(raw_post_observation, post_observation.astype(np.float64))
                or not np.array_equal(pre_observation, record_tokens[round_index])
                or not np.array_equal(post_observation, record_tokens[round_index + 1])
                or (previous_post_observation is not None
                    and not np.array_equal(pre_observation, previous_post_observation))):
            raise ValueError("learner-visited V2 observation bytes or chain differ")

        raw_action = _required_array(step.get("action"), shape=(mode_count, 2), label="V2 executed action")
        executed_action = _validate_action(raw_action, mode_count, rollout_config)
        if not np.array_equal(raw_action, executed_action):
            raise ValueError("learner-visited V2 action precision differs from float32 rollout")
        if step.get("action_sha") != sha256_array(executed_action):
            raise ValueError("learner-visited V2 executed action bytes differ")

        teacher_action, _ = trackb_receding_horizon_action_v2(
            context.before_coefficients.copy(), context.target_coefficients.copy(),
        )
        teacher_action = _validate_action(teacher_action, mode_count, rollout_config)
        _teacher_post, _before, _after, teacher_gain = _score_transition(context, teacher_action)
        post_context, before_distance, after_distance, executed_gain = _score_transition(context, executed_action)
        if (step.get("post_state_sha") != post_context.pre_state_sha
                or not math.isclose(_required_float(step.get("before_distance"), label="before distance"),
                                    before_distance, abs_tol=1e-12, rel_tol=1e-10)
                or not math.isclose(_required_float(step.get("after_distance"), label="after distance"),
                                    after_distance, abs_tol=1e-12, rel_tol=1e-10)
                or not math.isclose(_required_float(step.get("synthetic_gain"), label="executed gain"),
                                    executed_gain, abs_tol=1e-12, rel_tol=1e-10)):
            raise ValueError("learner-visited V2 executed transition differs from registered forward")
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
        "schema": LEARNER_VISITED_ROW_SCHEMA_V2,
        "sample_id": sample_key.sample_id, "split": sample_key.split,
        "sample_key_sha": sample_key.sha256(),
        "behavior_pilot_receipt_sha256": evidence.receipt_sha256,
        "checkpoint_sha256": evidence.checkpoint_sha256,
        "behavior_source_commit": evidence.source_commit,
        "behavior_source_tree": evidence.source_tree,
        "behavior_stage_plan_sha256": evidence.stage_plan_sha256,
        "behavior_dataset_receipt_sha256": evidence.dataset_receipt_sha256,
        "behavior_model_config_sha256": evidence.model_config_sha256,
        "behavior_model_state_sha256": evidence.model_state_sha256,
        "recipe_sha256": recipe.sha256(),
        "learner_view_plan_sha256": binding.learner_view_plan_sha256,
        "history_policy": recipe.history_policy,
        "history_policy_sha256": recipe.history_policy_sha256,
        "initial_context_sha256": initial_context_sha,
        "history_kind": LEARNER_HISTORY_KIND_V2, "label_kind": LEARNER_LABEL_KIND_V2,
        "teacher_gain_kind": TEACHER_GAIN_KIND_V2,
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
        "action_execution": BEHAVIOR_ACTION_EXECUTION_V2,
        "research_only": True, "selection_evidence": False,
        "promotion_ready": False, "hardware_action_authorized": False,
        "deployment_validation": False, "held_out": False,
    }
    row = {
        **row_identity,
        "row_sha": canonical_sha256(row_identity),
        "row_identity_json": canonical_json_bytes(row_identity).decode("utf-8"),
        "history_tokens": history_tokens, "history_mask": history_mask,
        "noisy_measurement_tokens": noisy_measurement_tokens,
        "target_mode_tokens": target_tokens, "mode_mask": mode_mask,
        "teacher_action_tokens": teacher_action_tokens,
        "teacher_transition_gain": teacher_transition_gain,
    }
    validate_learner_visited_tensors_v2(
        row["row_identity_json"], row["row_sha"],
        {name: row[name] for name in ROW_TENSOR_NAMES_V2},
    )
    return row


__all__ = [
    "LEARNER_VISITED_ROW_SCHEMA",
    "LEARNER_HISTORY_KIND",
    "LEARNER_LABEL_KIND",
    "TEACHER_GAIN_KIND",
    "LEARNER_VISITED_ROW_SCHEMA_V2",
    "LEARNER_HISTORY_KIND_V2",
    "LEARNER_LABEL_KIND_V2",
    "TEACHER_GAIN_KIND_V2",
    "Behavior_rollout_binding_v2",
    "bind_behavior_rollout_v2",
    "derive_learner_visited_row",
    "derive_learner_visited_row_v2",
]
