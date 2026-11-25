"""Explicit finite plans for the separate teacher-only diagnostic lifecycle.

Plans freeze caller choices; they do not choose a study size, authenticate an
external review, or admit model training. A runner additionally requires an
independently retained plan digest and source identity. All output is private
teacher evidence, never a generic V2.4 training root.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import re
import stat
from types import UnionType
from typing import Any, get_args, get_origin, get_type_hints

import numpy as np

from ml_recon.data_v2_4.admission_snapshot import Canonical_json_snapshot
from ml_recon.data_v2_4.hashing import canonical_json_bytes, sha256_array, sha256_json
from ml_recon.data_v2_4.sample_keys import V24SampleKey

from ml_recon.data_v2_4.synthetic.feedback_renderers import (
    CONTROL_AMPLITUDE_BOUND, CONTROL_PHASE_BOUND, TRACKB_RECEDING_HORIZON_DAMPING, _feedback_shots,
)
from ml_recon.data_v2_4.synthetic.trackb_teacher_policy import TRACKB_TEACHER_POLICY_V1, TRACKB_TEACHER_POLICY_V2
from ml_recon.physics_hybrid_v10.control_tangent import helmert_matrix
from ml_recon.physics_hybrid_v10.noise_interventions import registered_feedback_settings

from . import pilot_runtime_checks
from .contracts import require_git_oid, require_sha256
from .teacher_diagnostic_preflight import (
    Teacher_diagnostic_preflight_binding, capture_teacher_diagnostic_preflight,
)
from .self_rollout_observation import Acquired_observation_v2, Observation_schedule_v2
from .teacher_diagnostic_v2 import (
    Teacher_diagnostic_branch_v2, Teacher_diagnostic_config_v2,
    Teacher_diagnostic_result_v2, Teacher_diagnostic_step_v2,
)


plan_filename = "teacher_diagnostic_plan.json"
ready_filename = "teacher_diagnostic_ready.json"
artifact_max_bytes = 64 * 1024 * 1024
run_id_pattern = re.compile(r"[a-z][a-z0-9_]{0,63}")
reserved_run_ids = frozenset(("con", "prn", "aux", "nul", *(f"com{index}" for index in range(1, 10)),
                              *(f"lpt{index}" for index in range(1, 10))))


_branch_policies = (TRACKB_TEACHER_POLICY_V1, TRACKB_TEACHER_POLICY_V2, "zero_action")
_branch_names = ("teacher_v1", "teacher_v2", "zero")
_result_contract = {
    "schema": "TrackBTeacherDiagnosticKernelV2", "research_only": True,
    "model_gain_measured": False, "pure_state_fidelity_measured": False,
    "gate_decision": "NOT_PERFORMED", "selection_evidence": False,
    "promotion_ready": False, "hardware_action_authorized": False,
    "execution_admission": "NOT_GRANTED_IN_MEMORY_KERNEL_ONLY",
    "source_physical_contract_preflight": "NOT_PERFORMED",
    "action_execution_dtype": "float64", "state_dtype": "complex128",
    "target_source": "NORMALIZED_RENDERED_TARGET_MODE_TOKENS_CONVERTED_TO_FLOAT64",
    "endpoint": "D0_MINUS_D10_EQUAL_SETTING_WEIGHT_HELLINGER_SQUARED_AFTER_FROZEN_NUISANCE",
    "preparation_assumption": "STATIC_STATE_REPREPARED_FOR_EACH_SETTING_AND_ATTEMPT",
    "backaction_assumption": "NO_DYNAMIC_MEASUREMENT_BACKACTION_MODELED",
    "nuisance_assumption": "FROZEN_TIME_INVARIANT_COMPLETE_INTERVENTION",
    "policy_information": "LATENT_STATE_TARGET_ORACLES_NOT_OBSERVATION_DRIVEN_CONTROL",
    "pairing_kind": "common_random_variates_v1", "observation_windows_per_branch": 11,
    "actions_per_branch": 10, "incidental_renderer_observations_used": 0,
}


def result_filename(index: int) -> str:
    return f"teacher_diagnostic_result_{index:06d}.json"


def _canonical_directory(path: Path) -> None:
    if (not isinstance(path, Path) or not re.fullmatch(r"[A-Za-z]:", path.drive)
            or str(path).startswith(("\\\\", "//")) or path.parent == path):
        raise ValueError("only explicit local Windows drive paths are supported")
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("run directories must be explicit absolute canonical paths")
    for component in (*reversed(path.parents), path):
        metadata = component.lstat()
        if (not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)
                or getattr(metadata, "st_file_attributes", 0) & 0x400):
            raise ValueError("run directory ancestry contains a non-directory or reparse point")


def _validate_identity_choices(
    sample_keys: tuple[V24SampleKey, ...], config: Teacher_diagnostic_config_v2,
    expected_source: tuple[str, str], run_id: str, trusted_output_parent: Path,
    output_root: Path, protected_input_roots: tuple[Path, ...],
) -> None:
    if type(sample_keys) is not tuple or not sample_keys:
        raise ValueError("sample_keys must be an explicit finite nonempty tuple")
    if type(config) is not Teacher_diagnostic_config_v2:
        raise ValueError("an immutable teacher diagnostic recipe is required")
    for key in sample_keys:
        config.validate(key)
    if len({key.sample_id for key in sample_keys}) != len(sample_keys):
        raise ValueError("planned episode sample identities must be unique")
    if len({key.sha256() for key in sample_keys}) != len(sample_keys):
        raise ValueError("planned canonical sample keys must be unique")
    if type(expected_source) is not tuple or len(expected_source) != 2:
        raise ValueError("expected_source must pin commit and tree")
    require_git_oid(expected_source[0], "run_source_commit")
    require_git_oid(expected_source[1], "run_source_tree")
    if type(run_id) is not str or not run_id_pattern.fullmatch(run_id) or run_id in reserved_run_ids:
        raise ValueError("run_id must be a bounded snake_case identifier")
    if type(protected_input_roots) is not tuple:
        raise ValueError("protected_input_roots must be an explicit tuple")
    for path in (trusted_output_parent, output_root, *protected_input_roots):
        if (not isinstance(path, Path) or not path.is_absolute() or not re.fullmatch(r"[A-Za-z]:", path.drive)
                or str(path).startswith(("\\\\", "//")) or ".." in path.parts):
            raise ValueError("plan paths must be explicit canonical local Windows paths")
    if trusted_output_parent.parent == trusted_output_parent or trusted_output_parent == Path.home():
        raise ValueError("trusted output parent must be a task-owned directory, not a broad root")
    if (not isinstance(output_root, Path) or not output_root.is_absolute()
            or output_root != trusted_output_parent / run_id):
        raise ValueError("output root must be the exact run_id child of the trusted parent")
    for source in (pilot_runtime_checks.PILOT_REPOSITORY_ROOT, *protected_input_roots):
        if output_root == source or source in output_root.parents or output_root in source.parents:
            raise ValueError("output root overlaps source or protected input trees")


def _validate_choices(
    sample_keys: tuple[V24SampleKey, ...], config: Teacher_diagnostic_config_v2,
    expected_source: tuple[str, str], run_id: str, trusted_output_parent: Path,
    output_root: Path, protected_input_roots: tuple[Path, ...],
) -> None:
    _validate_identity_choices(sample_keys, config, expected_source, run_id, trusted_output_parent,
                               output_root, protected_input_roots)
    _canonical_directory(trusted_output_parent)
    for source in (pilot_runtime_checks.PILOT_REPOSITORY_ROOT, *protected_input_roots):
        _canonical_directory(source)
    if os.path.lexists(output_root):
        raise ValueError("diagnostic output root already exists; reuse and resume are prohibited")


@dataclass(frozen=True)
class Teacher_diagnostic_run_plan:
    sample_keys: tuple[V24SampleKey, ...]
    config: Teacher_diagnostic_config_v2
    expected_source: tuple[str, str]
    run_id: str
    trusted_output_parent: Path
    output_root: Path
    protected_input_roots: tuple[Path, ...]
    preflight_bindings: tuple[Teacher_diagnostic_preflight_binding, ...]

    def validate(self) -> None:
        _validate_choices(self.sample_keys, self.config, self.expected_source, self.run_id,
                          self.trusted_output_parent, self.output_root, self.protected_input_roots)
        self.validate_identity()

    def validate_identity(self) -> None:
        """Check immutable choices only; this does not permit creation or reuse."""
        _validate_identity_choices(self.sample_keys, self.config, self.expected_source, self.run_id,
                                   self.trusted_output_parent, self.output_root, self.protected_input_roots)
        if (type(self.preflight_bindings) is not tuple
                or len(self.preflight_bindings) != len(self.sample_keys)
                or any(type(binding) is not Teacher_diagnostic_preflight_binding
                       for binding in self.preflight_bindings)):
            raise ValueError("plan must retain one exact preflight binding per episode")

    def portable_payload(self) -> dict[str, Any]:
        """Describe every scientific choice without exporting private host paths."""
        location_binding = sha256_json({
            "trusted_output_parent": str(self.trusted_output_parent), "output_root": str(self.output_root),
            "protected_input_roots": [str(path) for path in self.protected_input_roots],
        })
        return {
            "schema": "TrackBTeacherDiagnosticRunPlanV1", "run_id": self.run_id,
            "output_root_basename": self.output_root.name, "location_binding_sha256": location_binding,
            "sample_keys": [asdict(key) for key in self.sample_keys], "recipe": asdict(self.config),
            "source_commit": self.expected_source[0], "source_tree": self.expected_source[1],
            "policy_id": self.config.history_policy, "policy_sha256": self.config.history_policy_sha256,
            "preflight_bindings": [binding.identity_summary() for binding in self.preflight_bindings],
            "episode_count": len(self.sample_keys), "actions_per_branch": 10, "windows_per_branch": 11,
            "sample_selection": "EXPLICIT_CALLER_REGISTERED_KEYS_NO_AUTOMATIC_EXPANSION",
            "research_only": True, "training_authorized": False, "model_gain_measured": False,
            "external_source_review": "NOT_CERTIFIED_BY_PLAN_OR_PREFLIGHT",
            "registered_study_approval": "EXTERNAL_WORKFLOW_REQUIRED",
            "gate_decision": "NOT_PERFORMED", "selection_evidence": False,
            "promotion_ready": False, "hardware_action_authorized": False,
        }

    def sha256(self) -> str:
        return sha256_json(self.portable_payload())


@dataclass(frozen=True)
class Teacher_diagnostic_run_receipt:
    """Returned only after ready publication, verified inventory and lease cleanup."""

    ready: Canonical_json_snapshot
    ready_file_identity: tuple[int, str]
    schema: str = "TrackBTeacherDiagnosticRunReturnV1"


def prepare_teacher_diagnostic_run_plan(
    *, sample_keys: tuple[V24SampleKey, ...], config: Teacher_diagnostic_config_v2,
    expected_source: tuple[str, str], run_id: str, trusted_output_parent: Path,
    output_root: Path, protected_input_roots: tuple[Path, ...] = (),
) -> Teacher_diagnostic_run_plan:
    """Freeze explicit caller choices and read all preflights without any write."""
    _validate_choices(sample_keys, config, expected_source, run_id, trusted_output_parent,
                      output_root, protected_input_roots)
    bindings = tuple(capture_teacher_diagnostic_preflight(
        key, config=config, expected_source=expected_source) for key in sample_keys)
    plan = Teacher_diagnostic_run_plan(sample_keys, config, expected_source, run_id,
                                      trusted_output_parent, output_root, protected_input_roots, bindings)
    if len(canonical_json_bytes(plan.portable_payload())) > artifact_max_bytes:
        raise ValueError("explicit plan exceeds the artifact byte limit")
    return plan


def _float64_tuple_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    """Validate the kernel's immutable Python-float tuples before NumPy conversion.

    Checking dtype after conversion is insufficient: mixed bool/float tuples
    are silently promoted. Validate original scalars because the original
    dataclass, not a normalized replacement, is what publication serializes.
    """
    if type(value) is not tuple or len(value) != shape[0]:
        raise ValueError(f"teacher {label} must be an immutable float64 tuple with registered shape")
    rows = (value,) if len(shape) == 1 else value
    row_width = shape[-1]
    if any(type(row) is not tuple or len(row) != row_width
           or any(type(item) is not float for item in row) for row in rows):
        raise ValueError(f"teacher {label} must contain exact Python float64 tuple scalars")
    array = np.asarray(value)
    if array.dtype != np.float64 or array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"teacher {label} must have finite float64 values and registered shape")
    return array


def _complex_coefficients(real: Any, imaginary: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Rebuild exact complex128 bytes, including signed zero, from typed parts."""
    real_array = _float64_tuple_array(real, shape, "coefficient real parts")
    imaginary_array = _float64_tuple_array(imaginary, shape, "coefficient imaginary parts")
    coefficients = np.empty(shape, dtype=np.complex128)
    coefficients.real, coefficients.imag = real_array, imaginary_array
    return coefficients


def validate_teacher_diagnostic_result(result: Teacher_diagnostic_result_v2, plan: Teacher_diagnostic_run_plan, index: int) -> bytes:
    """Check typed unit/trajectory relationships before accepting output bytes."""
    key, config = plan.sample_keys[index], plan.config
    if type(result) is not Teacher_diagnostic_result_v2 or result.sample_key != key or result.config != config:
        raise ValueError("teacher kernel returned a missing, foreign or wrong-unit result")
    for name, expected in _result_contract.items():
        actual = getattr(result, name)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"teacher kernel result violates {name}")
    require_sha256(result.initial_context_sha, "initial_context_sha")
    recipe = asdict(config)
    recipe.pop("replicate_index")
    if result.episode_sha != sha256_json({"sample_key_sha": key.sha256(), "recipe": recipe,
                                        "initial_context_sha": result.initial_context_sha}):
        raise ValueError("teacher episode identity does not match the planned key and recipe")
    settings = registered_feedback_settings(key.N)
    shots = tuple(int(value) for value in _feedback_shots(len(settings), config.shot_mode, config.shot_budget))
    attempts = 0 if config.shot_mode == "S_ideal" else 11 * sum(shots)
    if (result.settings != settings or result.shots_by_setting != shots
            or result.sampled_attempts_per_branch != attempts or len(result.observation_panel_shas) != 11
            or len(result.branches) != 3):
        raise ValueError("teacher kernel result has an invalid registered observation budget")
    target = _complex_coefficients(result.target_coefficients_re, result.target_coefficients_im, (key.N,))
    if (not math.isclose(float(np.linalg.norm(target)), 1.0, abs_tol=1e-10, rel_tol=0)
            or type(result.physical_parameter) is not float or not math.isfinite(result.physical_parameter)
            or type(result.intervention_type) is not str or not result.intervention_type):
        raise ValueError("teacher result target or complete nuisance identity is invalid")
    intervention = json.loads(result.intervention_descriptor_json)
    if (type(intervention) is not dict or not intervention
            or json.dumps(intervention, sort_keys=True, allow_nan=False) != result.intervention_descriptor_json):
        raise ValueError("teacher intervention descriptor is not its complete canonical object")
    action_bounds = TRACKB_RECEDING_HORIZON_DAMPING * np.array([CONTROL_AMPLITUDE_BOUND, CONTROL_PHASE_BOUND])
    initial_shas = []
    for branch, name, policy in zip(result.branches, _branch_names, _branch_policies):
        if (branch.branch_kind != name or branch.policy_id != policy
                or branch.initial_context_sha != result.initial_context_sha
                or len(branch.steps) != 10 or len(branch.observation_records) != 11):
            raise ValueError("teacher kernel result has missing rounds, windows or policy identity")
        states = _complex_coefficients(branch.state_coefficients_re, branch.state_coefficients_im, (11, key.N))
        if not np.allclose(np.linalg.norm(states, axis=1), 1.0, atol=1e-10, rtol=0):
            raise ValueError("teacher state trajectory is malformed")
        initial_shas.append(sha256_array(states[0]))
        context_sha = sha256_json({
            "sample_id": key.sample_id, "initial_coefficients": sha256_array(states[0]),
            "target_coefficients": sha256_array(target), "intervention_type": result.intervention_type,
            "intervention": intervention, "physical_parameter": result.physical_parameter, "settings": settings,
        })
        if context_sha != result.initial_context_sha:
            raise ValueError("teacher initial context identity does not bind its full serialized values")
        if branch.observation_records[0] != result.branches[0].observation_records[0]:
            raise ValueError("teacher branches do not share the initial acquired record")
        for window, observation in enumerate(branch.observation_records):
            schedule = observation.schedule
            if (observation.shot_mode != config.shot_mode or schedule.sample_key_sha != key.sha256()
                    or schedule.replicate_index != config.replicate_index or schedule.window_index != window
                    or schedule.settings != settings or schedule.shots_by_setting != shots
                    or schedule.panel_sha != result.observation_panel_shas[window]
                    or schedule != result.branches[0].observation_records[window].schedule):
                raise ValueError("teacher observation is not bound to the planned common panel")
        for step_index, step in enumerate(branch.steps):
            action = _float64_tuple_array(step.action, (key.N, 2), "action")
            if (step.round_number != step_index + 1 or step.action_sha != sha256_array(action)
                    or not np.allclose(action.sum(axis=0), 0, atol=1e-12, rtol=0)
                    or np.any(np.abs(helmert_matrix(key.N) @ action) > action_bounds + 1e-12)
                    or step.pre_state_sha != sha256_array(states[step_index])
                    or step.post_state_sha != sha256_array(states[step_index + 1])
                    or step.pre_observation_record_id != branch.observation_records[step_index].record_id
                    or step.post_observation_record_id != branch.observation_records[step_index + 1].record_id
                    or not all(math.isfinite(value) for value in (
                        step.before_distance, step.after_distance, step.latent_gain))
                    or not 0 <= step.before_distance <= 1 or not 0 <= step.after_distance <= 1
                    or not math.isclose(step.latent_gain, step.before_distance - step.after_distance,
                                        abs_tol=1e-12, rel_tol=0)
                    or (step_index and step.before_distance != branch.steps[step_index - 1].after_distance)):
                raise ValueError("teacher result action, state or continuous observation identity is inconsistent")
        total = branch.steps[0].before_distance - branch.steps[-1].after_distance
        if (branch.initial_distance != branch.steps[0].before_distance
                or branch.final_distance != branch.steps[-1].after_distance
                or not math.isclose(branch.terminal_improvement, total, abs_tol=1e-12, rel_tol=0)
                or not math.isclose(branch.summed_step_gain, total, abs_tol=1e-12, rel_tol=0)
                or not math.isclose(math.fsum(step.latent_gain for step in branch.steps), total, abs_tol=1e-12, rel_tol=0)):
            raise ValueError("teacher terminal gain does not match its ten-round trajectory")
        if name == "zero" and (any(np.any(step.action) or step.latent_gain != 0 for step in branch.steps)
                               or any(state.tobytes() != states[0].tobytes() for state in states)):
            raise ValueError("zero branch does not preserve exact state and gain")
    if len(set(initial_shas)) != 1:
        raise ValueError("teacher branches did not start from the same episode")
    payload = canonical_json_bytes(asdict(result))
    if len(payload) > artifact_max_bytes:
        raise ValueError("teacher result exceeds the artifact byte limit")
    return payload


def decode_canonical_teacher_object(payload: bytes, *, label: str) -> dict[str, Any]:
    """Parse bounded exact canonical JSON without coercion or duplicate keys."""
    if type(payload) is not bytes or not payload or len(payload) > artifact_max_bytes:
        raise ValueError(f"{label} must be bounded immutable JSON bytes")

    def unique_object(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"{label} contains duplicate JSON keys")
            result[name] = value
        return result

    def reject_constant(value):
        raise ValueError(f"{label} contains nonfinite JSON")

    try:
        result = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object,
                            parse_constant=reject_constant)
        if type(result) is not dict or canonical_json_bytes(result) != payload:
            raise ValueError(f"{label} must be an exact canonical JSON object")
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"{label} is malformed JSON") from error
    return result


_result_types = (V24SampleKey, Teacher_diagnostic_config_v2, Teacher_diagnostic_step_v2,
                 Teacher_diagnostic_branch_v2, Teacher_diagnostic_result_v2,
                 Observation_schedule_v2, Acquired_observation_v2)


def _decode_teacher_value(value: Any, expected_type: Any) -> Any:
    """Decode only this explicit dataclass whitelist and its scalar/tuple fields."""
    if expected_type in (str, int, float, bool, type(None)):
        if type(value) is not expected_type or (expected_type is float and not math.isfinite(value)):
            raise ValueError("teacher result JSON scalar type or finiteness is invalid")
        return value
    origin, arguments = get_origin(expected_type), get_args(expected_type)
    if origin is UnionType:
        for alternative in arguments:
            try:
                return _decode_teacher_value(value, alternative)
            except ValueError:
                pass
        raise ValueError("teacher result JSON union field has no valid exact type")
    if origin is tuple:
        if type(value) is not list:
            raise ValueError("teacher result JSON tuple must be an array")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_decode_teacher_value(item, arguments[0]) for item in value)
        if len(value) != len(arguments):
            raise ValueError("teacher result JSON tuple has an invalid length")
        return tuple(_decode_teacher_value(item, item_type) for item, item_type in zip(value, arguments))
    if expected_type not in _result_types or type(value) is not dict:
        raise ValueError("teacher result JSON contains an unsupported object type")
    declared_fields = fields(expected_type)
    if set(value) != {field.name for field in declared_fields}:
        raise ValueError("teacher result JSON dataclass fields are missing or unknown")
    hints = get_type_hints(expected_type)
    decoded = {field.name: _decode_teacher_value(value[field.name], hints[field.name]) for field in declared_fields}
    result = expected_type(**{field.name: decoded[field.name] for field in declared_fields if field.init})
    for field in declared_fields:
        if not field.init and getattr(result, field.name) != decoded[field.name]:
            raise ValueError("teacher acquired record identity differs from its complete payload")
        if (expected_type in (Teacher_diagnostic_result_v2, Observation_schedule_v2, Acquired_observation_v2)
                and field.default is not MISSING and decoded[field.name] != field.default):
            raise ValueError("teacher result fixed schema or observation semantics changed")
    return result


def decode_teacher_diagnostic_result(payload: bytes) -> Teacher_diagnostic_result_v2:
    """Strict typed decoding only; callers still validate the independently pinned plan."""
    value = decode_canonical_teacher_object(payload, label="teacher result")
    result = _decode_teacher_value(value, Teacher_diagnostic_result_v2)
    if canonical_json_bytes(asdict(result)) != payload:
        raise ValueError("teacher result round-trip changed serialized bytes")
    return result


__all__ = ["Teacher_diagnostic_run_plan", "Teacher_diagnostic_run_receipt", "prepare_teacher_diagnostic_run_plan",
           "validate_teacher_diagnostic_result", "decode_teacher_diagnostic_result",
           "decode_canonical_teacher_object"]
