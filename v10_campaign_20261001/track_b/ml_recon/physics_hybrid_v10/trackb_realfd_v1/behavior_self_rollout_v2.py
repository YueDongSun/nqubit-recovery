"""Actual model/zero/V2-teacher ten-action computation for the fixed 4Q recipe.

All branches execute float32 logical actions on their own evolving states. The
separate float64 teacher diagnostic is not reused as model measurement. This is
an in-memory kernel: source, data, checkpoint and study admission belong to the
stage orchestrator, not to a caller-supplied completion or approval flag.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ml_recon.data_v2_4.hashing import sha256_json
from ml_recon.data_v2_4.sample_keys import V24SampleKey
from ml_recon.data_v2_4.synthetic.feedback_renderers import _feedback_shots
from ml_recon.data_v2_4.synthetic.trackb_phase_lift import trackb_receding_horizon_action_v2

from .behavior_v2_contract import Behavior_recipe_v2
from .inverse_transformer import TrackBInverseTransformerV1
from .model_self_rollout_core import (
    Rollout_branch_spec, execute_rollout_branch, execute_rollout_branches,
    isolated_model_policy, preserve_process_rng,
)
from .self_rollout_evaluator import Rollout_branch_v1, Self_rollout_config_v1, rollout_core_adapters
from .teacher_diagnostic_v2 import _context_identity, _render_initial_context


branch_specs = (Rollout_branch_spec("model", "model", "model"),
                Rollout_branch_spec("zero", "zero", "zero"),
                Rollout_branch_spec("teacher_v2", "teacher", "teacher"))


@dataclass(frozen=True)
class Behavior_self_rollout_result_v2:
    sample_key: V24SampleKey
    recipe_sha256: str
    initial_context_sha256: str
    model_config_sha256: str
    replicate_index: int
    shots_by_setting: tuple[int, ...]
    branches: tuple[Rollout_branch_v1, ...]
    schema: str = "TrackBBehaviorSelfRolloutV2"
    endpoint: str = "D0_MINUS_D10_EQUAL_SETTING_WEIGHT_HELLINGER_SQUARED_AFTER_FROZEN_NUISANCE"
    action_execution: str = "FLOAT32_COMMON_MODEL_ZERO_TEACHER_V1"
    target_source: str = "NORMALIZED_RENDERED_TARGET_MODE_TOKENS_CONVERTED_TO_FLOAT64"
    history_distribution_shift: str = "UNOBSERVED_ZERO_GAIN_INPUT_MATCHED_LEARNER_STATE_SHIFT_REMAINS"
    pairing_kind: str = "common_random_variates_v1"
    statistical_unit: str = "PAIRED_SOURCE_EPISODE_REPLICATE_NOT_THREE_INDEPENDENT_SAMPLES"
    observation_windows_per_branch: int = 11
    actions_per_branch: int = 10
    incidental_renderer_observations_used: int = 0
    preparation_assumption: str = "STATIC_STATE_REPREPARED_FOR_EACH_SETTING_AND_ATTEMPT"
    backaction_assumption: str = "NO_DYNAMIC_MEASUREMENT_BACKACTION_MODELED"
    nuisance_assumption: str = "FROZEN_TIME_INVARIANT_COMPLETE_INTERVENTION"
    research_only: bool = True
    scientific_gate: str = "NOT_PERFORMED"
    learner_minus_behavior_gate: str = "NOT_PERFORMED_NO_LEARNER_TRAINED"
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False
    execution_admission: str = "NOT_GRANTED_IN_MEMORY_KERNEL_ONLY"


@dataclass(frozen=True)
class Behavior_model_branch_replay_v2:
    """One deterministic model branch replay, bound to a fixed V2 checkpoint."""

    sample_key: V24SampleKey
    recipe_sha256: str
    initial_context_sha256: str
    model_config_sha256: str
    replicate_index: int
    shots_by_setting: tuple[int, ...]
    branch: Rollout_branch_v1
    schema: str = "TrackBBehaviorModelBranchReplayV2"
    action_execution: str = "FLOAT32_COMMON_MODEL_ZERO_TEACHER_V1"
    research_only: bool = True
    scientific_gate: str = "NOT_PERFORMED"
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False


def replay_behavior_model_branch_v2(
    model: TrackBInverseTransformerV1,
    sample_key: V24SampleKey,
    *,
    recipe: Behavior_recipe_v2,
    replicate_index: int,
) -> Behavior_model_branch_replay_v2:
    """Re-execute only the frozen behavior model branch for provenance checks.

    The public three-branch evaluator remains validation-only. This replay is a
    narrow verification seam for a stored validation unit; it never opens the
    TRAIN split or writes an artifact.
    """
    if type(recipe) is not Behavior_recipe_v2:
        raise ValueError("behavior branch replay requires the fixed V2 recipe")
    recipe.validate()
    diagnostic_config = recipe.teacher_config()
    diagnostic_config.validate(sample_key)
    if sample_key.split != "validation" or sample_key.global_seed != recipe.dataset_seed:
        raise ValueError("behavior branch replay accepts registered validation keys only")
    if type(replicate_index) is not int or replicate_index < 0:
        raise ValueError("behavior observation replicate must be a nonnegative integer")
    if (not isinstance(model, TrackBInverseTransformerV1)
            or model.config.history_gain_input_kind != recipe.history_gain_input_kind):
        raise ValueError("behavior model must use the matched unobserved-zero history input")
    model.config.validate()
    for name in ("system_type", "mode_count", "width", "heads", "blocks", "action_projection_kind",
                 "action_value_feature_kind", "gain_parameterization", "history_gain_input_kind"):
        if getattr(model.config, name) != getattr(recipe, name):
            raise ValueError(f"behavior model differs from the fixed recipe at {name}")
    config = Self_rollout_config_v1(
        system_type=recipe.system_type, shot_mode=recipe.shot_mode, shot_budget=recipe.shot_budget,
        noise_family=recipe.noise_family, eta=recipe.eta, kappa=recipe.kappa,
        replicate_index=replicate_index, include_teacher_state_control=False)
    config.validate()
    with preserve_process_rng():
        policy = isolated_model_policy(model, config=config)
        context = _render_initial_context(sample_key, diagnostic_config)
        context_sha = _context_identity(context)
        shots = _feedback_shots(len(context.settings), config.shot_mode, config.shot_budget)
        branch, control = execute_rollout_branch(
            context, sample_key, shots, policy, branch_spec=branch_specs[0], config=config,
            teacher_policy=lambda current: trackb_receding_horizon_action_v2(
                current.before_coefficients, current.target_coefficients)[0],
            adapters=rollout_core_adapters(),
        )
    if (control is not None or branch.branch_kind != "model"
            or len(branch.steps) != 10 or len(branch.observation_records) != 11
            or not math.isfinite(branch.telescoping_gain)
            or not math.isclose(branch.telescoping_gain, branch.summed_step_gain,
                                abs_tol=1e-12, rel_tol=0)):
        raise ValueError("behavior model branch replay is incomplete or misbound")
    return Behavior_model_branch_replay_v2(
        sample_key, recipe.sha256(), context_sha, sha256_json(model.config.to_dict()),
        replicate_index, tuple(int(value) for value in shots), branch,
    )


def evaluate_behavior_model_v2(
    model: TrackBInverseTransformerV1, sample_key: V24SampleKey, *, recipe: Behavior_recipe_v2,
    replicate_index: int, branch_order: tuple[str, ...] | None = None,
) -> Behavior_self_rollout_result_v2:
    """Measure three paired branches without mutating the caller-owned model."""
    if type(recipe) is not Behavior_recipe_v2:
        raise ValueError("behavior rollout requires the fixed V2 recipe")
    recipe.validate()
    diagnostic_config = recipe.teacher_config()
    diagnostic_config.validate(sample_key)
    if sample_key.split != "validation" or sample_key.global_seed != recipe.dataset_seed:
        raise ValueError("behavior measurement requires the explicitly registered validation seed/split")
    if type(replicate_index) is not int or replicate_index < 0:
        raise ValueError("behavior observation replicate must be a nonnegative integer")
    if (not isinstance(model, TrackBInverseTransformerV1)
            or model.config.history_gain_input_kind != recipe.history_gain_input_kind):
        raise ValueError("behavior model must use the matched unobserved-zero history input")
    model.config.validate()
    for name in ("system_type", "mode_count", "width", "heads", "blocks", "action_projection_kind",
                 "action_value_feature_kind", "gain_parameterization", "history_gain_input_kind"):
        if getattr(model.config, name) != getattr(recipe, name):
            raise ValueError(f"behavior model differs from the fixed recipe at {name}")
    config = Self_rollout_config_v1(
        system_type=recipe.system_type, shot_mode=recipe.shot_mode, shot_budget=recipe.shot_budget,
        noise_family=recipe.noise_family, eta=recipe.eta, kappa=recipe.kappa,
        replicate_index=replicate_index, include_teacher_state_control=False)
    config.validate()
    with preserve_process_rng():
        policy = isolated_model_policy(model, config=config)
        context = _render_initial_context(sample_key, diagnostic_config)
        context_sha = _context_identity(context)
        shots = _feedback_shots(len(context.settings), config.shot_mode, config.shot_budget)
        computation = execute_rollout_branches(
            context, sample_key, shots, policy, config=config,
            teacher_policy=lambda current: trackb_receding_horizon_action_v2(
                current.before_coefficients, current.target_coefficients)[0],
            adapters=rollout_core_adapters(), branch_specs=branch_specs, branch_order=branch_order)
    branches = computation.branches
    if (len(branches) != 3 or tuple(branch.branch_kind for branch in branches) != ("model", "zero", "teacher_v2")
            or computation.teacher_state_control is not None):
        raise ValueError("behavior rollout returned an unregistered branch inventory")
    for branch in branches:
        if (len(branch.steps) != 10 or len(branch.observation_records) != 11
                or not math.isfinite(branch.telescoping_gain)
                or not math.isclose(branch.telescoping_gain, branch.summed_step_gain, abs_tol=1e-12, rel_tol=0)):
            raise ValueError("behavior rollout is incomplete or its endpoint does not telescope")
    if len({branch.observation_records[0].record_id for branch in branches}) != 1:
        raise ValueError("behavior branches do not share the same round-zero observation")
    return Behavior_self_rollout_result_v2(
        sample_key, recipe.sha256(), context_sha, sha256_json(model.config.to_dict()), replicate_index,
        tuple(int(value) for value in shots), branches)


def summarize_behavior_measurements(results: tuple[Behavior_self_rollout_result_v2, ...]) -> dict[str, object]:
    """Descriptive paired endpoints, explicitly not the later learner gain gate."""
    if not results or len({result.sample_key.sample_id for result in results}) != len(results):
        raise ValueError("behavior measurement units must be nonempty and unique")
    summary = {}
    for branch_index, name in enumerate(("model", "zero", "teacher_v2")):
        branches = [result.branches[branch_index] for result in results]
        gains = [branch.telescoping_gain for branch in branches]
        summary[name] = {"mean_terminal_gain": math.fsum(gains) / len(gains),
                         "minimum_terminal_gain": min(gains), "maximum_terminal_gain": max(gains),
                         "terminal_negative_fraction": sum(value < 0 for value in gains) / len(gains),
                         "harmful_action_fraction": sum(step.harmful for branch in branches for step in branch.steps)
                         / (10 * len(branches))}
    paired = [result.branches[0].telescoping_gain - result.branches[1].telescoping_gain for result in results]
    return {"schema": "TrackBBehaviorMeasurementSummaryV2", "unit_count": len(results), "branches": summary,
            "model_minus_zero_mean": math.fsum(paired) / len(paired),
            "model_minus_zero_minimum": min(paired), "paired_units_not_independent_branches": True,
            "confidence_interval": "NOT_COMPUTED_DESCRIPTIVE_STAGE_ONLY",
            "scientific_gate": "NOT_PERFORMED", "learner_minus_behavior_gate": "NOT_PERFORMED_NO_LEARNER_TRAINED",
            "model_selection_or_promotion": False, "hardware_action_authorized": False}


__all__ = [
    "Behavior_self_rollout_result_v2",
    "Behavior_model_branch_replay_v2",
    "replay_behavior_model_branch_v2",
    "evaluate_behavior_model_v2",
    "summarize_behavior_measurements",
]
