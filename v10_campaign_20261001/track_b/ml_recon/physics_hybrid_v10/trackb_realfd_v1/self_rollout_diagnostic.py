"""Source-bound publication for frozen, observable-only Track B self-rollouts.

This development diagnostic never trains, selects a policy, changes a gate, or
creates hardware authority. V2.4 artifacts and checkpoint bytes are read-only.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

import numpy as np

from ml_recon.data_v2_4.hashing import sha256_array, sha256_bytes
from ml_recon.data_v2_4.loaders import V24TrainingDataset
from ml_recon.data_v2_4.sample_keys import V24SampleKey, make_sample_key
from ml_recon.data_v2_4.synthetic.feedback_renderers import (
    CONTROL_AMPLITUDE_BOUND,
    CONTROL_PHASE_BOUND,
    _feedback_shots,
    replay_trackb_ten_round_forward_contexts,
)
from ml_recon.data_v2_4.version import GENERATOR_SHA

from .artifact_io import publish_new_bytes_with_state
from .contracts import canonical_json_bytes, canonical_sha256, require_sha256
from .diagnostic_runtime import preserve_python_numpy_torch_rng
from .inverse_transformer import (
    LEGACY_ORACLE_HISTORY_GAIN_INPUT,
)
from .policy_ablation import (
    _bound_v24_source,
    _dataset_source_identity,
    _evaluation_kwargs,
    _load_bound_model,
    _load_split_rows,
    _matching_trackb_cell,
    _require_matching_source_identity,
)
from .self_rollout_evaluator import (
    Self_rollout_config_v1,
    _bind_round_zero,
    _score_transition,
    evaluate_trackb_model_self_rollout,
    history_gain_input_kind,
    is_materially_negative_trackb_gain,
    round_count,
)
from .self_rollout_shift_contract import (
    LEGACY_ORACLE_GAIN_SHIFT, UNOBSERVED_ZERO_GAIN_SHIFT,
    history_gain_distribution_shift_kind, report_history_distribution_shift_kind,
)
from .self_rollout_observation import (
    build_observation_panel,
    common_pairing,
    observation_stream,
    pairing_kinds,
)
from .self_rollout_outcomes import build_self_rollout_failure, summarize_self_rollout_samples
from .ten_round_guidance import _require_git_identity
from .train_fit_diagnostic import (
    _checkpoint_path_from_receipt,
    _prepare_new_output_path,
    _read_verified_pilot_receipt,
    _require_isolated_report_output_path,
)


self_rollout_diagnostic_schema = "TrackBSelfRolloutDiagnosticV1"
default_sample_count = 32
development_splits = ("train", "validation")
self_rollout_source_paths = (
    "ml_recon/physics_hybrid_v10/trackb_realfd_v1/self_rollout_diagnostic.py",
    "ml_recon/physics_hybrid_v10/trackb_realfd_v1/self_rollout_evaluator.py",
    "ml_recon/physics_hybrid_v10/trackb_realfd_v1/self_rollout_shift_contract.py",
    "ml_recon/physics_hybrid_v10/trackb_realfd_v1/self_rollout_observation.py",
    "ml_recon/physics_hybrid_v10/trackb_realfd_v1/self_rollout_outcomes.py",
)


def _history_distribution_shift_kind(model_config: Any) -> str:
    """Describe gain-feature parity and retain the separate learner-state caveat."""
    input_kind = getattr(model_config, "history_gain_input_kind", LEGACY_ORACLE_HISTORY_GAIN_INPUT)
    return report_history_distribution_shift_kind(input_kind)


@dataclass(frozen=True)
class TrackBSelfRolloutDiagnosticConfigV1:
    """An immutable receipt and a declared development sample budget."""

    pilot_receipt_path: str
    expected_pilot_receipt_sha: str
    output_path: str
    split: str = "train"
    sample_count: int = default_sample_count
    replicate_index: int = 0
    pairing_kind: str = common_pairing
    schema: str = self_rollout_diagnostic_schema

    def validate(self) -> None:
        for name in ("pilot_receipt_path", "output_path"):
            value = getattr(self, name)
            if type(value) is not str or not value or not Path(value).is_absolute():
                raise ValueError(f"{name} must be an absolute path")
        require_sha256(self.expected_pilot_receipt_sha, "expected_pilot_receipt_sha")
        if self.split not in development_splits:
            raise ValueError("self-rollout diagnostic accepts development splits only")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("self-rollout sample_count must be a positive integer")
        if type(self.replicate_index) is not int or self.replicate_index < 0:
            raise ValueError("self-rollout replicate_index must be a nonnegative integer")
        if self.pairing_kind not in pairing_kinds:
            raise ValueError("self-rollout pairing kind is unsupported")
        if self.schema != self_rollout_diagnostic_schema:
            raise ValueError("unsupported self-rollout diagnostic schema")


def _committed_self_rollout_identity() -> tuple[str, str]:
    """Reject dirty or untracked source before claiming a committed evaluator."""
    repository_root = Path(__file__).resolve().parents[3]
    status = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=repository_root, text=True,
    ).strip()
    if status:
        raise ValueError("self-rollout diagnostic requires clean committed source")
    # A newly added module must be in the Git tree, even if an ignore rule hides it.
    subprocess.check_output(
        ("git", "ls-files", "--error-unmatch", "--", *self_rollout_source_paths),
        cwd=repository_root, text=True,
    )
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=repository_root, text=True,
    ).strip()
    tree = subprocess.check_output(
        ("git", "rev-parse", "HEAD^{tree}"), cwd=repository_root, text=True,
    ).strip()
    _require_git_identity(commit, "evaluator_commit")
    _require_git_identity(tree, "evaluator_tree")
    return commit, tree


def _bound_development_rows(
    receipt: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    dataset: V24TrainingDataset,
    *,
    sample_count: int,
) -> tuple[dict[str, object], dict[int, Mapping[str, Any]]]:
    """Admit the whole split before taking an exact deterministic prefix."""
    dataset_record = receipt.get("dataset")
    checkpoint_source = checkpoint.get(f"{dataset.split}_source")
    receipt_source = dataset_record.get(dataset.split) if isinstance(dataset_record, Mapping) else None
    if not isinstance(checkpoint_source, Mapping) or not isinstance(receipt_source, Mapping):
        raise ValueError("self-rollout pilot lacks the requested split source identity")
    source = _dataset_source_identity(dataset)
    _require_matching_source_identity(checkpoint_source, receipt_source, label="checkpoint/receipt")
    _require_matching_source_identity(checkpoint_source, source, label="checkpoint/opened")
    if source["generator_sha"] != GENERATOR_SHA:
        raise ValueError("self-rollout source generator differs from the loaded replay generator")
    rows = _load_split_rows(dataset, sample_count=sample_count)
    if len(rows) != sample_count:
        raise ValueError("self-rollout did not load the exact requested sample count")
    return source, rows


def _verified_replay_key(
    source_index: int,
    row: Mapping[str, Any],
    *,
    mode_count: int,
    replay: Mapping[str, Any],
) -> V24SampleKey:
    """Check source row identity, target and round-zero action before inference."""
    key = make_sample_key(
        global_seed=replay["global_seed"], split=replay["split"],
        dataset_family="feedback_history", N=mode_count, index=source_index,
        generator_sha=replay["generator_sha"], cell_id=replay["cell_id"],
    )
    if row.get("sample_id") != key.sample_id:
        raise ValueError("self-rollout row identity differs from its replay key")
    context = replay_trackb_ten_round_forward_contexts(
        key, **{name: replay[name] for name in (
            "shot_mode", "shot_budget", "noise_family", "eta", "kappa",
        )},
    )[0]
    history = row["history_tokens"].detach().cpu().numpy()
    mask = row["history_mask"].detach().cpu().numpy()
    mode_mask = row["mode_mask"].detach().cpu().numpy()
    target = row["target_mode_tokens"].detach().cpu().numpy()
    if (history.shape != (10, mode_count, 7) or mask.shape != (10,)
            or mask.dtype != np.bool_ or not bool(mask.all())
            or mode_mask.shape != (mode_count,) or mode_mask.dtype != np.bool_
            or not bool(mode_mask.all()) or target.shape != (mode_count, 2)
            or not np.isfinite(history).all() or not np.isfinite(target).all()):
        raise ValueError("self-rollout source row has invalid registered history or masks")
    coefficients = target[:, 0] * np.exp(1j * np.pi * target[:, 1])
    norm = float(np.linalg.norm(coefficients))
    if not np.isclose(norm, 1.0, atol=1e-6, rtol=0) or np.any(target[:, 0] < 0):
        raise ValueError("self-rollout target amplitude is invalid")
    if not np.allclose(coefficients / norm, context.target_coefficients, atol=1e-6, rtol=0):
        raise ValueError("self-rollout target differs from registered replay")
    if not np.allclose(history[0, :, 2:4], context.teacher_action, atol=1e-6, rtol=0):
        raise ValueError("self-rollout first teacher action differs from registered replay")
    return key


def _publish_failure_receipt(path: Path, failure: Mapping[str, object], error: Exception) -> None:
    """Preserve the original failure even if its separate audit publication fails."""
    try:
        payload = canonical_json_bytes(failure)
        publication = publish_new_bytes_with_state(path, payload, expected_sha=sha256_bytes(payload))
    except Exception as publication_error:
        note = f"Track B failure receipt publication failed: {type(publication_error).__name__}"
    else:
        note = f"Track B failure receipt: {publication.target_path} ({publication.status})"
    if hasattr(error, "add_note"):
        error.add_note(note)


def _validate_self_rollout_result(
    result: Mapping[str, Any],
    *,
    key: V24SampleKey,
    model: Any,
    replay: Mapping[str, Any],
    replicate_index: int,
    pairing_kind: str,
    allow_legacy_v2: bool = False,
) -> None:
    """Admit one complete evaluator result before exposing it as completed.

    The evaluator already validates its inputs while producing a result.  This
    second boundary validates the serialized result itself, including every
    acquisition schedule and the matched teacher control.  It therefore catches
    stale, mixed, or mutated result dictionaries before progress accounting or
    publication can treat them as completed samples.
    """
    if not isinstance(result, Mapping):
        raise ValueError("self-rollout result contract mismatch: result is not a mapping")

    model_config = getattr(model, "config", None)
    system_type = getattr(model_config, "system_type", None)
    if not isinstance(system_type, str):
        raise ValueError("self-rollout result contract mismatch: requested system is unavailable")
    amplitude_bound = getattr(model_config, "amplitude_bound", CONTROL_AMPLITUDE_BOUND)
    phase_bound = getattr(model_config, "phase_bound", CONTROL_PHASE_BOUND)
    model_gain_input_kind = getattr(
        model_config, "history_gain_input_kind", LEGACY_ORACLE_HISTORY_GAIN_INPUT,
    )
    schema = result.get("schema")
    is_legacy_v2 = schema == "TrackBSelfRolloutEvaluationV2"
    if is_legacy_v2 and not allow_legacy_v2:
        raise ValueError("self-rollout result contract mismatch: legacy V2 is not admitted here")
    if schema not in ("TrackBSelfRolloutEvaluationV3", "TrackBSelfRolloutEvaluationV2"):
        raise ValueError("self-rollout result contract mismatch: schema")
    if is_legacy_v2:
        if result.get("input_distribution_shift") != LEGACY_ORACLE_GAIN_SHIFT:
            raise ValueError("self-rollout result contract mismatch: legacy V2 input distribution shift")
    else:
        expected_shift = history_gain_distribution_shift_kind(model_gain_input_kind)
        if result.get("input_distribution_shift") != expected_shift:
            raise ValueError("self-rollout result contract mismatch: input distribution shift")
    expected_config = Self_rollout_config_v1(
        system_type=system_type,
        shot_mode=replay["shot_mode"],
        shot_budget=replay["shot_budget"],
        noise_family=replay["noise_family"],
        eta=replay["eta"],
        kappa=replay["kappa"],
        amplitude_bound=amplitude_bound,
        phase_bound=phase_bound,
        # This is the evaluation-time zero-gain input recipe. Training provenance
        # comes from the bound model config and is reflected only in the shift field.
        history_gain_input_kind=history_gain_input_kind,
        replicate_index=replicate_index,
        pairing_kind=pairing_kind,
        include_teacher_state_control=True,
    )
    expected_config.validate()
    expected_context, expected_binding_sha = _bind_round_zero(key, None, expected_config)
    expected_settings = tuple(tuple(setting) for setting in expected_context.settings)
    expected_shots = tuple(int(value) for value in _feedback_shots(
        len(expected_settings), expected_config.shot_mode, expected_config.shot_budget,
    ))

    def json_shape(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): json_shape(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [json_shape(item) for item in value]
        return value

    def require_equal(actual: Any, expected: Any, label: str) -> None:
        if json_shape(actual) != json_shape(expected):
            raise ValueError(f"self-rollout result contract mismatch: {label}")

    expected_schema = "TrackBSelfRolloutEvaluationV2" if is_legacy_v2 else "TrackBSelfRolloutEvaluationV3"
    require_equal(result.get("schema"), expected_schema, "schema")
    require_equal(result.get("sample_id"), key.sample_id, "sample_id")
    require_equal(result.get("sample_key_sha"), key.sha256(), "sample_key_sha")
    require_equal(result.get("round_zero_binding_sha"), expected_binding_sha, "round-zero binding")
    require_equal(result.get("config"), asdict(expected_config), "declared evaluation recipe")
    require_equal(result.get("history_gain_input_kind"), history_gain_input_kind, "history gain input")
    require_equal(result.get("observation_stream"), observation_stream, "observation stream")
    require_equal(result.get("replicate_index"), replicate_index, "declared evaluation recipe replicate")
    require_equal(result.get("pairing_kind"), pairing_kind, "declared evaluation recipe pairing")
    require_equal(result.get("shots_by_setting"), list(expected_shots), "shots by setting")
    require_equal(result.get("shots_per_observation"), sum(expected_shots), "shots per observation")
    expected_sampled_shots = 0 if expected_config.shot_mode == "S_ideal" else (round_count + 1) * sum(expected_shots)
    require_equal(result.get("sampled_shots_per_branch"), expected_sampled_shots, "sampled shots")
    require_equal(result.get("observation_substreams"), list(range(round_count + 1)), "observation windows")
    require_equal(
        result.get("observation_substreams_scope"),
        "WINDOW_INDICES_ONLY_SETTING_SUBSTREAMS_ARE_IN_BRANCH_SCHEDULES",
        "observation substream scope",
    )
    require_equal(result.get("crn_schedule"), "PER_BRANCH_SAMPLE_REPLICATE_WINDOW_SETTING_SCHEDULE_V2", "CRN schedule")
    require_equal(
        result.get("observation_panel_shas_scope"),
        "MODEL_BRANCH_COMPATIBILITY_VIEW_USE_BRANCH_ACQUISITION_SCHEDULE",
        "observation panel scope",
    )
    require_equal(result.get("statistical_unit"), "PAIRED_SOURCE_EPISODE_REPLICATE_NOT_THREE_INDEPENDENT_SAMPLES", "statistical unit")
    require_equal(result.get("history_gain_consumed_value"), 0.0, "history gain consumed value")
    require_equal(result.get("history_gain_consumed_dtype"), "float32", "history gain consumed dtype")
    require_equal(
        result.get("history_gain_preprocessing"),
        "DIRECT_FLOAT32_ZERO_NO_CENTERING_OR_SCALING_NOT_A_MISSINGNESS_MASK",
        "history gain preprocessing",
    )

    branches = result.get("branches")
    if not isinstance(branches, (list, tuple)) or [branch.get("branch_kind") for branch in branches if isinstance(branch, Mapping)] != ["model", "zero", "teacher"]:
        raise ValueError("self-rollout result contract mismatch: branch order")
    if len(branches) != 3:
        raise ValueError("self-rollout result contract mismatch: branch count")

    model_panel_shas: list[str] = []
    for branch in branches:
        if not isinstance(branch, Mapping):
            raise ValueError("self-rollout result contract mismatch: branch is not a mapping")
        branch_kind = branch["branch_kind"]
        steps = branch.get("steps")
        records = branch.get("observation_records")
        record_ids = branch.get("observation_record_ids")
        schedules = branch.get("acquisition_schedule")
        if not (isinstance(steps, (list, tuple)) and len(steps) == round_count
                and isinstance(records, (list, tuple)) and len(records) == round_count + 1
                and isinstance(record_ids, (list, tuple)) and len(record_ids) == round_count + 1
                and isinstance(schedules, (list, tuple)) and len(schedules) == round_count + 1):
            raise ValueError("self-rollout result contract mismatch: branch observation count")
        require_equal(branch.get("selected_round"), round_count, f"{branch_kind} selected round")
        require_equal(branch.get("observation_count"), round_count + 1, f"{branch_kind} observation count")
        for window_index, (record, schedule, record_id) in enumerate(zip(records, schedules, record_ids)):
            if not isinstance(record, Mapping) or not isinstance(schedule, Mapping):
                raise ValueError("self-rollout result contract mismatch: acquisition record")
            _, expected_schedule_object = build_observation_panel(
                key, expected_context.settings, np.asarray(expected_shots, dtype=np.int64),
                window_index=window_index, branch_kind=branch_kind,
                replicate_index=replicate_index, pairing_kind=pairing_kind,
            )
            expected_schedule = asdict(expected_schedule_object)
            require_equal(schedule, expected_schedule, f"{branch_kind} acquisition schedule")
            require_equal(record.get("schedule"), schedule, f"{branch_kind} record schedule")
            require_equal(record.get("record_id"), record_id, f"{branch_kind} record identity")
            record_payload = {name: value for name, value in record.items() if name != "record_id"}
            require_equal(record_id, canonical_sha256(record_payload), f"{branch_kind} record bytes")
            try:
                serialized_tokens = np.asarray(record.get("measurement_tokens"), dtype=np.float32)
            except (TypeError, ValueError) as error:
                raise ValueError("self-rollout result contract mismatch: measurement token bytes") from error
            if serialized_tokens.shape != (key.N, 2) or not np.isfinite(serialized_tokens).all():
                raise ValueError("self-rollout result contract mismatch: measurement token bytes")
            require_equal(
                record.get("measurement_token_sha"),
                sha256_array(serialized_tokens),
                f"{branch_kind} measurement token SHA",
            )
            require_equal(record.get("schema"), "TrackBSelfRolloutAcquiredObservationV2", f"{branch_kind} record schema")
            require_equal(record.get("shot_mode"), expected_config.shot_mode, f"{branch_kind} shot mode")
            if branch_kind == "model":
                model_panel_shas.append(schedule["panel_sha"])
        branch_context = deepcopy(expected_context)
        for step_index, step in enumerate(steps, start=1):
            if not isinstance(step, Mapping):
                raise ValueError("self-rollout result contract mismatch: branch step")
            require_equal(step.get("round_number"), step_index, f"{branch_kind} round number")
            require_equal(step.get("pre_observation_record_id"), record_ids[step_index - 1], f"{branch_kind} pre-observation identity")
            require_equal(step.get("post_observation_record_id"), record_ids[step_index], f"{branch_kind} post-observation identity")
            pre_record = records[step_index - 1]
            post_record = records[step_index]
            require_equal(step.get("pre_observation_sha"), pre_record.get("measurement_token_sha"), f"{branch_kind} pre-observation SHA")
            require_equal(step.get("post_observation_sha"), post_record.get("measurement_token_sha"), f"{branch_kind} post-observation SHA")
            require_equal(step.get("pre_observation"), pre_record.get("measurement_tokens"), f"{branch_kind} pre-observation bytes")
            require_equal(step.get("post_observation"), post_record.get("measurement_tokens"), f"{branch_kind} post-observation bytes")
            action = step.get("action")
            if not isinstance(action, (list, tuple)):
                raise ValueError("self-rollout result contract mismatch: action bytes")
            action_array = np.asarray(action, dtype=np.float64)
            require_equal(step.get("action_sha"), sha256_array(action_array), f"{branch_kind} action SHA")
            require_equal(step.get("pre_state_sha"), branch_context.pre_state_sha, f"{branch_kind} pre-state lineage")
            recomputed_context, recomputed_before, recomputed_after, recomputed_gain = _score_transition(
                branch_context, action_array,
            )
            require_equal(step.get("post_state_sha"), recomputed_context.pre_state_sha, f"{branch_kind} post-state lineage")
            before_distance = step.get("before_distance")
            after_distance = step.get("after_distance")
            gain = step.get("synthetic_gain")
            if (not all(isinstance(value, (int, float)) and np.isfinite(value)
                        for value in (before_distance, after_distance, gain))):
                raise ValueError("self-rollout result contract mismatch: non-finite step score")
            if not np.isclose(float(before_distance), recomputed_before, atol=1e-12, rtol=1e-10):
                raise ValueError("self-rollout result contract mismatch: step before distance")
            if not np.isclose(float(after_distance), recomputed_after, atol=1e-12, rtol=1e-10):
                raise ValueError("self-rollout result contract mismatch: step after distance")
            if not np.isclose(float(gain), recomputed_gain, atol=1e-12, rtol=1e-10):
                raise ValueError("self-rollout result contract mismatch: step transition gain")
            if not np.isclose(float(gain), float(before_distance) - float(after_distance), atol=1e-12, rtol=1e-10):
                raise ValueError("self-rollout result contract mismatch: step gain")
            require_equal(step.get("harmful"), is_materially_negative_trackb_gain(recomputed_gain), f"{branch_kind} harmful flag")
            branch_context = replace(
                recomputed_context,
                round_index=min(step_index, round_count - 1),
            )
        require_equal(branch.get("initial_distance"), steps[0].get("before_distance"), f"{branch_kind} initial distance")
        require_equal(branch.get("final_distance"), steps[-1].get("after_distance"), f"{branch_kind} final distance")
        telescoping_gain = float(steps[0]["before_distance"]) - float(steps[-1]["after_distance"])
        summed_gain = math.fsum(float(step["synthetic_gain"]) for step in steps)
        if not np.isclose(summed_gain, telescoping_gain, atol=1e-12, rtol=1e-10):
            raise ValueError(f"self-rollout result contract mismatch: {branch_kind} gain telescoping")
        require_equal(branch.get("telescoping_gain"), telescoping_gain, f"{branch_kind} telescoping gain")
        require_equal(branch.get("summed_step_gain"), summed_gain, f"{branch_kind} summed step gain")
        require_equal(branch.get("final_state_sha"), steps[-1].get("post_state_sha"), f"{branch_kind} final state")
    require_equal(branches[0]["steps"][0].get("pre_state_sha"), expected_context.pre_state_sha, "round-zero state")

    require_equal(result.get("observation_panel_shas"), model_panel_shas, "observation panel SHAs")
    control = result.get("teacher_state_zero_history_gain_control")
    if not isinstance(control, Mapping):
        raise ValueError("self-rollout result contract mismatch: teacher control missing")
    teacher_records = branches[2]["observation_record_ids"]
    require_equal(control.get("is_true_rollout"), False, "teacher control rollout kind")
    require_equal(control.get("control_kind"), "teacher_state_zero_history_gain_candidate_v1", "teacher control kind")
    require_equal(control.get("evaluation_kind"), "COUNTERFACTUAL_ONE_STEP_ACTIONS_AT_NEW_TEACHER_BRANCH_STATES", "teacher control evaluation")
    require_equal(control.get("additional_observation_count"), 0, "teacher control observation count")
    require_equal(control.get("history_action_source"), "NEW_TEACHER_BRANCH_EXECUTED_ACTIONS", "teacher control history source")
    require_equal(control.get("history_gain_input_kind"), history_gain_input_kind,
                  "teacher control history input")
    require_equal(control.get("history_gain_consumed_value"), 0.0, "teacher control history value")
    require_equal(control.get("history_gain_consumed_dtype"), "float32", "teacher control history dtype")
    require_equal(
        control.get("history_gain_preprocessing"),
        "DIRECT_FLOAT32_ZERO_NO_CENTERING_OR_SCALING_NOT_A_MISSINGNESS_MASK",
        "teacher control preprocessing",
    )
    control_steps = control.get("steps")
    require_equal(control.get("teacher_observation_record_ids"), teacher_records, "teacher control observation identities")
    if not isinstance(control_steps, (list, tuple)) or len(control_steps) != round_count:
        raise ValueError("self-rollout result contract mismatch: teacher control steps")
    teacher_contexts: list[Any] = []
    teacher_context = deepcopy(expected_context)
    for teacher_step in branches[2]["steps"]:
        teacher_contexts.append(deepcopy(teacher_context))
        main_action = np.asarray(teacher_step.get("action"), dtype=np.float64)
        teacher_context, _before, _after, _gain = _score_transition(teacher_context, main_action)
        teacher_context = replace(
            teacher_context,
            round_index=min(int(teacher_step["round_number"]), round_count - 1),
        )
    for step_index, step in enumerate(control_steps, start=1):
        if not isinstance(step, Mapping):
            raise ValueError("self-rollout result contract mismatch: teacher control step")
        require_equal(step.get("round_number"), step_index, f"teacher control round {step_index}")
        require_equal(step.get("pre_observation_record_id"), teacher_records[step_index - 1], f"teacher control pre-observation {step_index}")
        require_equal(step.get("teacher_history_observation_record_ids"), teacher_records[:step_index], f"teacher control history {step_index}")
        teacher_context = teacher_contexts[step_index - 1]
        require_equal(step.get("pre_state_sha"), teacher_context.pre_state_sha, f"teacher control pre-state {step_index}")
        action = step.get("action")
        if not isinstance(action, (list, tuple)):
            raise ValueError("self-rollout result contract mismatch: teacher control action")
        action_array = np.asarray(action, dtype=np.float64)
        require_equal(step.get("action_sha"), sha256_array(action_array), f"teacher control action SHA {step_index}")
        candidate_context, recomputed_before, recomputed_after, recomputed_gain = _score_transition(
            teacher_context, action_array,
        )
        require_equal(step.get("candidate_after_state_sha"), candidate_context.pre_state_sha, f"teacher control candidate state {step_index}")
        before_distance = step.get("before_distance")
        after_distance = step.get("candidate_after_distance")
        gain = step.get("candidate_gain")
        if (not all(isinstance(value, (int, float)) and np.isfinite(value)
                    for value in (before_distance, after_distance, gain))):
            raise ValueError("self-rollout result contract mismatch: teacher control score")
        if not np.isclose(float(before_distance), recomputed_before, atol=1e-12, rtol=1e-10):
            raise ValueError("self-rollout result contract mismatch: teacher control before distance")
        if not np.isclose(float(after_distance), recomputed_after, atol=1e-12, rtol=1e-10):
            raise ValueError("self-rollout result contract mismatch: teacher control after distance")
        if not np.isclose(float(gain), recomputed_gain, atol=1e-12, rtol=1e-10):
            raise ValueError("self-rollout result contract mismatch: teacher control transition gain")
        if not np.isclose(float(gain), float(before_distance) - float(after_distance), atol=1e-12, rtol=1e-10):
            raise ValueError("self-rollout result contract mismatch: teacher control gain")
        require_equal(step.get("harmful"), is_materially_negative_trackb_gain(recomputed_gain), f"teacher control harmful {step_index}")


def run_trackb_self_rollout_diagnostic(
    config: TrackBSelfRolloutDiagnosticConfigV1,
    *,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Evaluate a frozen checkpoint and publish a new development-only report."""
    config.validate()
    output_path = _prepare_new_output_path(Path(config.output_path))
    failure_path = _prepare_new_output_path(output_path.with_name(f"{output_path.stem}_failure.json"))
    evaluator_identity = _committed_self_rollout_identity()
    with preserve_python_numpy_torch_rng():
        receipt_path = Path(config.pilot_receipt_path).resolve()
        receipt = _read_verified_pilot_receipt(receipt_path, config.expected_pilot_receipt_sha)
        root, build_id, _identity, manifest = _bound_v24_source(receipt)
        _require_isolated_report_output_path(
            output_path, dataset_root=root, receipt_path=receipt_path,
            checkpoint_path=_checkpoint_path_from_receipt(receipt),
        )
        _require_isolated_report_output_path(
            failure_path, dataset_root=root, receipt_path=receipt_path,
            checkpoint_path=_checkpoint_path_from_receipt(receipt),
        )
        model, model_identity, checkpoint = _load_bound_model(receipt)
        for field in ("source_commit", "source_tree"):
            _require_git_identity(model_identity[field], f"model_{field}")
        cell = _matching_trackb_cell(manifest, system_type=model.config.system_type)
        dataset = V24TrainingDataset(
            root, build_id=build_id, cell_id=cell["cell_id"],
            split=config.split, task="feedback", arm="A0_RAW",
        )
        try:
            source, rows = _bound_development_rows(
                receipt, checkpoint, dataset, sample_count=config.sample_count,
            )
        finally:
            dataset.close()
        replay = _evaluation_kwargs(manifest=manifest, cell=cell, split=config.split, device="cpu")
        binding = {
            "config": asdict(config), "evaluator_source_commit": evaluator_identity[0],
            "evaluator_source_tree": evaluator_identity[1],
            "pilot_receipt_sha": config.expected_pilot_receipt_sha,
            "model": model_identity, "source": source, "replay": replay,
            "evaluation_recipe": "COMPLETE_ACQUISITION_SELF_ROLLOUT_WITH_MATCHED_TEACHER_CONTROL_V2",
        }
        intended_sample_ids = [str(rows[index]["sample_id"]) for index in sorted(rows)]
        attempted_sample_ids: list[str] = []
        samples = []
        sample_ids = []
        failure_stage = "sample_admission"
        source_verification = "NOT_RECHECKED"
        try:
            for source_index in sorted(rows):
                attempted_sample_ids.append(str(rows[source_index]["sample_id"]))
                failure_stage = "sample_admission"
                key = _verified_replay_key(
                    source_index, rows[source_index], mode_count=model.config.mode_count, replay=replay,
                )
                failure_stage = "sample_evaluation"
                result = evaluate_trackb_model_self_rollout(
                    model, key, replicate_index=config.replicate_index, pairing_kind=config.pairing_kind,
                    include_teacher_state_control=True,
                    **{name: replay[name] for name in (
                        "shot_mode", "shot_budget", "noise_family", "eta", "kappa",
                    )},
                )
                _validate_self_rollout_result(
                    result, key=key, model=model, replay=replay,
                    replicate_index=config.replicate_index, pairing_kind=config.pairing_kind,
                )
                samples.append(result)
                sample_ids.append(key.sample_id)
                failure_stage = "progress_callback"
                if progress is not None:
                    progress({"event": "sample_completed", "completed": len(samples),
                              "total": config.sample_count, "sample_id": key.sample_id})
            failure_stage = "evaluator_source_recheck"
            source_verification = "MISMATCH_OR_UNAVAILABLE"
            if _committed_self_rollout_identity() != evaluator_identity:
                raise ValueError("self-rollout evaluator source changed during execution")
            source_verification = "MATCHED"
            failure_stage = "outcome_validation"
            output = {
                "schema": self_rollout_diagnostic_schema, **binding,
                "sample_ids": sample_ids, "sample_set_sha": canonical_sha256(sample_ids),
                "sample_count": len(samples), "samples": samples,
                "branch_summaries": summarize_self_rollout_samples(samples),
                "episode_accounting": {"requested_count": config.sample_count,
                                       "attempted_count": len(attempted_sample_ids),
                                       "completed_count": len(sample_ids), "failed_count": 0,
                                       "not_attempted_count": 0},
                "status": "COMPLETE", "evaluation_role": "DEVELOPMENT_DIAGNOSTIC",
                "history_gain_input_kind": history_gain_input_kind,
                "history_distribution_shift": _history_distribution_shift_kind(model.config),
                "selected_final_round": 10, "gate_decision": "NOT_PERFORMED",
                "diagnostic_only": True, "research_only": True,
                "selection_evidence": False, "promotion_ready": False,
                "hardware_action_authorized": False, "external_waveform_write": "NOT_IMPLEMENTED",
            }
            failure_stage = "artifact_publication"
            payload = canonical_json_bytes(output)
            publication = publish_new_bytes_with_state(output_path, payload, expected_sha=sha256_bytes(payload))
        except Exception as error:
            try:
                failure = build_self_rollout_failure(
                    binding, intended_sample_ids=intended_sample_ids,
                    attempted_sample_ids=attempted_sample_ids, completed_sample_ids=sample_ids,
                    failure_stage=failure_stage, error_type=type(error).__name__,
                    source_verification_at_completion=source_verification,
                )
                _publish_failure_receipt(failure_path, failure, error)
            except Exception as accounting_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"Track B failure accounting unavailable: {type(accounting_error).__name__}")
            raise
    return {**output, "publication": publication.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen Track B ten-round self-rollout diagnostic")
    parser.add_argument("--pilot-receipt", required=True)
    parser.add_argument("--expected-pilot-receipt-sha", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=development_splits, default="train")
    parser.add_argument("--sample-count", type=int, default=default_sample_count)
    parser.add_argument("--replicate-index", type=int, default=0)
    parser.add_argument("--pairing-kind", choices=pairing_kinds, default=common_pairing)
    arguments = parser.parse_args()
    config = TrackBSelfRolloutDiagnosticConfigV1(
        pilot_receipt_path=str(Path(arguments.pilot_receipt).resolve()),
        expected_pilot_receipt_sha=arguments.expected_pilot_receipt_sha,
        output_path=str(Path(arguments.output).resolve()), split=arguments.split,
        sample_count=arguments.sample_count,
        replicate_index=arguments.replicate_index, pairing_kind=arguments.pairing_kind,
    )
    result = run_trackb_self_rollout_diagnostic(
        config, progress=lambda event: print(json.dumps(event, sort_keys=True), flush=True),
    )
    print(json.dumps({"event": "published", "sample_count": result["sample_count"],
                      "publication": result["publication"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
