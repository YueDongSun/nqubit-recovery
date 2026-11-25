"""In-memory V1/V2/zero target-oracle comparison on one canonical V2 episode.

This kernel grants no execution admission and publishes no dataset or receipt.
Its global-Pauli synthetic forward is the registered renderer's forward, not a
qualification of the full DLCZ physical contract. Incidental renderer samples
are discarded; each branch has eleven continuous acquisitions and ten actions.
Teachers observe latent state, not sampled records, and cannot measure model gain.

The narrow private renderer adapters preserve the frozen V1 policy and shot
allocation. Teacher-returned states retain the renderer's float64 action path;
the model evaluator's float32 quantization is deliberately not used here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import math

import numpy as np

from ml_recon.data_v2_4.hashing import sha256_array, sha256_json
from ml_recon.data_v2_4.sample_keys import V24SampleKey
from ml_recon.data_v2_4.synthetic.feedback_renderers import (
    CONTROL_AMPLITUDE_BOUND,
    CONTROL_PHASE_BOUND,
    TRACKB_RECEDING_HORIZON_DAMPING,
    TrackBRecedingHorizonForwardContextV1,
    _feedback_shots,
    _trackb_receding_horizon_action,
    render_feedback_sample,
)
from ml_recon.data_v2_4.synthetic.trackb_phase_lift import trackb_receding_horizon_action_v2
from ml_recon.data_v2_4.synthetic.trackb_teacher_policy import (
    TRACKB_TEACHER_POLICY_V1,
    TRACKB_TEACHER_POLICY_V2,
    TRACKB_TEACHER_POLICY_V2_DESCRIPTOR,
    TRACKB_TEACHER_POLICY_V2_SHA256,
    TRACKB_TEN_ROUND_COUNT,
    validate_trackb_v2_cell,
    validate_trackb_v2_sample_key,
)
from ml_recon.physics_hybrid_v10.control_tangent import apply_tangent_action, helmert_matrix
from ml_recon.physics_hybrid_v10.noise_interventions import (
    V24NoiseIntervention, registered_feedback_settings, w_density_matrix,
)
from ml_recon.physics_hybrid_v10.noise_matching import evaluated_severity_matching_kind, hellinger_squared

from .self_rollout_observation import Acquired_observation_v2, acquire_observation, common_pairing


branch_kinds = ("teacher_v1", "teacher_v2", "zero")
round_count = TRACKB_TEN_ROUND_COUNT
numerical_tolerance = 1e-12
# Same physical no-op equivalence tolerance as the registered self-rollout.
zero_equivalence_tolerance = 1e-10
Float_rows = tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class Teacher_diagnostic_config_v2:
    """One explicit renderer recipe; observation replicate affects only uniforms."""

    system_type: str
    shot_mode: str = "S_setting"
    shot_budget: int = 256
    noise_family: str = "feedback_mixed"
    eta: float = 0.08
    kappa: float = 0.15
    replicate_index: int = 0
    history_policy: str = TRACKB_TEACHER_POLICY_V2
    history_policy_sha256: str = TRACKB_TEACHER_POLICY_V2_SHA256
    severity_matching_kind: str = evaluated_severity_matching_kind

    def validate(self, sample_key: V24SampleKey) -> None:
        if not isinstance(sample_key, V24SampleKey) or type(sample_key.global_seed) is not int:
            raise ValueError("a canonical V24SampleKey with integer seed is required")
        cell = {
            "N": sample_key.N, "cell_id": sample_key.cell_id,
            "dataset_family": sample_key.dataset_family, "system_type": self.system_type,
            "history_rounds": round_count, "history_policy": self.history_policy,
            "history_policy_sha256": self.history_policy_sha256,
            "severity_matching_kind": self.severity_matching_kind,
        }
        mode_count = validate_trackb_v2_cell(cell)
        validate_trackb_v2_sample_key(
            sample_key, N=mode_count, dataset_family="feedback_history", cell_id=cell["cell_id"])
        if sha256_json(TRACKB_TEACHER_POLICY_V2_DESCRIPTOR) != self.history_policy_sha256:
            raise ValueError("V2 descriptor no longer matches its registered SHA")
        action_contract = TRACKB_TEACHER_POLICY_V2_DESCRIPTOR["action"]
        for name, value in (("amplitude_bound", CONTROL_AMPLITUDE_BOUND),
                            ("phase_bound", CONTROL_PHASE_BOUND),
                            ("damping", TRACKB_RECEDING_HORIZON_DAMPING)):
            if action_contract[name] != value:
                raise ValueError("frozen teachers no longer share the registered action contract")
        if type(self.shot_budget) is not int or self.shot_budget <= 0:
            raise ValueError("shot_budget must be a positive integer")
        _feedback_shots(2 * mode_count - 1, self.shot_mode, self.shot_budget)
        if self.noise_family != "feedback_mixed":
            raise ValueError("diagnostic requires the registered feedback_mixed recipe")
        for name in ("eta", "kappa"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite float fraction")
        if type(self.replicate_index) is not int or self.replicate_index < 0:
            raise ValueError("replicate_index must be a nonnegative integer")


@dataclass(frozen=True)
class Teacher_diagnostic_step_v2:
    round_number: int
    pre_state_sha: str
    post_state_sha: str
    action: Float_rows
    action_sha: str
    pre_observation_record_id: str
    post_observation_record_id: str
    before_distance: float
    after_distance: float
    latent_gain: float


@dataclass(frozen=True)
class Teacher_diagnostic_branch_v2:
    branch_kind: str
    policy_id: str
    initial_context_sha: str
    steps: tuple[Teacher_diagnostic_step_v2, ...]
    state_coefficients_re: Float_rows
    state_coefficients_im: Float_rows
    observation_records: tuple[Acquired_observation_v2, ...]
    initial_distance: float
    final_distance: float
    terminal_improvement: float
    summed_step_gain: float


@dataclass(frozen=True)
class Teacher_diagnostic_result_v2:
    sample_key: V24SampleKey
    config: Teacher_diagnostic_config_v2
    episode_sha: str
    initial_context_sha: str
    target_coefficients_re: tuple[float, ...]
    target_coefficients_im: tuple[float, ...]
    intervention_descriptor_json: str
    intervention_type: str
    physical_parameter: float
    settings: tuple[tuple[str, ...], ...]
    shots_by_setting: tuple[int, ...]
    sampled_attempts_per_branch: int
    observation_panel_shas: tuple[str, ...]
    branches: tuple[Teacher_diagnostic_branch_v2, ...]
    schema: str = "TrackBTeacherDiagnosticKernelV2"
    target_source: str = "NORMALIZED_RENDERED_TARGET_MODE_TOKENS_CONVERTED_TO_FLOAT64"
    endpoint: str = "D0_MINUS_D10_EQUAL_SETTING_WEIGHT_HELLINGER_SQUARED_AFTER_FROZEN_NUISANCE"
    action_execution_dtype: str = "float64"
    state_dtype: str = "complex128"
    policy_information: str = "LATENT_STATE_TARGET_ORACLES_NOT_OBSERVATION_DRIVEN_CONTROL"
    preparation_assumption: str = "STATIC_STATE_REPREPARED_FOR_EACH_SETTING_AND_ATTEMPT"
    backaction_assumption: str = "NO_DYNAMIC_MEASUREMENT_BACKACTION_MODELED"
    nuisance_assumption: str = "FROZEN_TIME_INVARIANT_COMPLETE_INTERVENTION"
    pairing_kind: str = common_pairing
    observation_windows_per_branch: int = round_count + 1
    actions_per_branch: int = round_count
    incidental_renderer_observations_used: int = 0
    research_only: bool = True
    model_gain_measured: bool = False
    pure_state_fidelity_measured: bool = False
    gate_decision: str = "NOT_PERFORMED"
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False
    execution_admission: str = "NOT_GRANTED_IN_MEMORY_KERNEL_ONLY"
    source_physical_contract_preflight: str = "NOT_PERFORMED"


def _context_identity(context: TrackBRecedingHorizonForwardContextV1) -> str:
    """Bind full nuisance values, not just its scalar or mechanism name."""
    return sha256_json({
        "sample_id": context.sample_id,
        "initial_coefficients": sha256_array(context.before_coefficients),
        "target_coefficients": sha256_array(context.target_coefficients),
        "intervention_type": type(context.intervention).__qualname__,
        "intervention": asdict(context.intervention),
        "physical_parameter": context.physical_parameter, "settings": context.settings,
    })


def _render_initial_context(
    sample_key: V24SampleKey, config: Teacher_diagnostic_config_v2,
) -> TrackBRecedingHorizonForwardContextV1:
    rendered = render_feedback_sample(
        sample_key, shot_mode=config.shot_mode, shot_budget=config.shot_budget,
        noise_family=config.noise_family, eta=config.eta, kappa=config.kappa, composite_noise=True,
        severity_matching_kind=config.severity_matching_kind)
    truth, internal = rendered["truth"], rendered["internal"]
    # This is the exact target reconstruction used by ten-round forward replay.
    before = (np.array(truth["current_coefficients_re"], dtype=np.float64, copy=True)
              + 1j * np.array(truth["current_coefficients_im"], dtype=np.float64, copy=True))
    tokens = np.array(rendered["observation"]["target_mode_tokens"], dtype=np.float64, copy=True)
    if before.shape != (sample_key.N,) or tokens.shape != (sample_key.N, 2):
        raise ValueError("renderer returned an invalid state or target shape")
    target = tokens[:, 0] * np.exp(1j * np.pi * tokens[:, 1])
    target_norm = float(np.linalg.norm(target))
    if not math.isfinite(target_norm) or target_norm <= 0:
        raise ValueError("renderer returned a nonfinite or zero target")
    target /= target_norm
    if (not np.isfinite(before).all()
            or not math.isclose(float(np.linalg.norm(before)), 1.0, abs_tol=zero_equivalence_tolerance,
                                rel_tol=0.0)):
        raise ValueError("renderer returned an invalid normalized initial state")
    context = TrackBRecedingHorizonForwardContextV1(
        sample_id=sample_key.sample_id, round_index=0,
        before_coefficients=before, target_coefficients=target,
        intervention=deepcopy(internal["intervention"]),
        physical_parameter=float(internal["physical_parameter"]),
        settings=registered_feedback_settings(sample_key.N),
        teacher_action=np.zeros((sample_key.N, 2), dtype=np.float64), teacher_gain=0.0,
        pre_state_sha=sha256_array(before))
    context.validate()
    if not math.isfinite(context.physical_parameter):
        raise ValueError("renderer returned a nonfinite nuisance parameter")
    return context


def _latent_distance(context: TrackBRecedingHorizonForwardContextV1, target_law: np.ndarray) -> float:
    # w_density_matrix normalizes in place: never pass a ledger-owned array.
    law = context.intervention.measurement_law(
        w_density_matrix(context.before_coefficients.copy()), context.settings, context.physical_parameter)
    if not np.isfinite(law).all():
        raise ValueError("registered nuisance produced a nonfinite measurement law")
    distance = float(hellinger_squared(law, target_law))
    if not math.isfinite(distance) or not 0 <= distance <= 1:
        raise ValueError("registered Hellinger distance is invalid")
    return distance


def _teacher_transition(
    context: TrackBRecedingHorizonForwardContextV1, branch_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve frozen teacher arithmetic and check its Helmert action contract."""
    before, target = context.before_coefficients.copy(), context.target_coefficients.copy()
    mode_count = len(before)
    if branch_kind == "teacher_v1":
        action, after = _trackb_receding_horizon_action(before, target)
    elif branch_kind == "teacher_v2":
        action, after = trackb_receding_horizon_action_v2(before, target)
    else:
        action = np.zeros((mode_count, 2), dtype=np.float64)
        after = apply_tangent_action(before, np.zeros(mode_count - 1), np.zeros(mode_count - 1))
    if action.dtype != np.float64 or action.shape != (mode_count, 2) or not np.isfinite(action).all():
        raise ValueError("teacher must return a finite float64 [N,2] action")
    if not np.allclose(action.sum(axis=0), 0, atol=numerical_tolerance, rtol=0):
        raise ValueError("teacher action is not gauge centered")
    coordinates = helmert_matrix(mode_count) @ action
    bounds = TRACKB_RECEDING_HORIZON_DAMPING * np.array([CONTROL_AMPLITUDE_BOUND, CONTROL_PHASE_BOUND])
    if np.any(np.abs(coordinates) > bounds + numerical_tolerance):
        raise ValueError("teacher action exceeds the damped Helmert bounds")
    if (after.dtype != np.complex128 or after.shape != before.shape or not np.isfinite(after).all()
            or not math.isclose(float(np.linalg.norm(after)), 1.0,
                                abs_tol=zero_equivalence_tolerance, rel_tol=0)):
        raise ValueError("teacher returned an invalid complex128 normalized state")
    executed = apply_tangent_action(context.before_coefficients.copy(), coordinates[:, 0], coordinates[:, 1])
    if not np.allclose(after, executed, atol=numerical_tolerance, rtol=0):
        raise ValueError("teacher state differs from its float64 action transition")
    return action.copy(), after.copy()


def _run_branch(
    initial: TrackBRecedingHorizonForwardContextV1, sample_key: V24SampleKey,
    config: Teacher_diagnostic_config_v2, shots: np.ndarray, branch_kind: str,
) -> Teacher_diagnostic_branch_v2:
    context = deepcopy(initial)
    initial_context_sha = _context_identity(context)
    identity = V24NoiseIntervention("identity", "none", "none", 0.0, "none")
    target_law = identity.measurement_law(w_density_matrix(context.target_coefficients.copy()), context.settings, 0.0)
    acquisition = dict(shot_mode=config.shot_mode, replicate_index=config.replicate_index,
                       pairing_kind=common_pairing, branch_kind="zero" if branch_kind == "zero" else "teacher")
    observation = acquire_observation(context, sample_key, shots, window_index=0, **acquisition)
    records, states, steps = [observation], [context.before_coefficients.copy()], []
    before_distance = _latent_distance(context, target_law)
    for index in range(round_count):
        action, after = _teacher_transition(context, branch_kind)
        post = replace(context, before_coefficients=after, pre_state_sha=sha256_array(after))
        after_distance = _latent_distance(post, target_law)
        if branch_kind == "zero":
            phase = np.angle(np.vdot(context.before_coefficients, after))
            if (not np.allclose(after * np.exp(-1j * phase), context.before_coefficients,
                                atol=zero_equivalence_tolerance, rtol=0)
                    or abs(before_distance - after_distance) > zero_equivalence_tolerance):
                raise ValueError("zero action did not preserve physical state and distance")
            # Verify the stabilized transition first, then retain exact no-op bytes.
            post = replace(context, before_coefficients=context.before_coefficients.copy())
            after_distance = before_distance
        post_observation = acquire_observation(post, sample_key, shots, window_index=index + 1, **acquisition)
        steps.append(Teacher_diagnostic_step_v2(
            round_number=index + 1, pre_state_sha=context.pre_state_sha, post_state_sha=post.pre_state_sha,
            action=tuple(tuple(float(value) for value in row) for row in action),
            action_sha=sha256_array(action), pre_observation_record_id=observation.record_id,
            post_observation_record_id=post_observation.record_id, before_distance=before_distance,
            after_distance=after_distance, latent_gain=before_distance - after_distance))
        records.append(post_observation)
        states.append(post.before_coefficients.copy())
        context = replace(post, round_index=min(index + 1, round_count - 1))
        observation, before_distance = post_observation, after_distance
    improvement = steps[0].before_distance - steps[-1].after_distance
    summed_gain = math.fsum(step.latent_gain for step in steps)
    if not math.isclose(improvement, summed_gain, abs_tol=numerical_tolerance, rel_tol=0):
        raise ValueError("ten-round latent gains do not telescope")
    policy_id = {"teacher_v1": TRACKB_TEACHER_POLICY_V1, "teacher_v2": TRACKB_TEACHER_POLICY_V2,
                 "zero": "zero_action"}[branch_kind]
    return Teacher_diagnostic_branch_v2(
        branch_kind=branch_kind, policy_id=policy_id, initial_context_sha=initial_context_sha,
        steps=tuple(steps), state_coefficients_re=tuple(tuple(float(x) for x in state.real) for state in states),
        state_coefficients_im=tuple(tuple(float(x) for x in state.imag) for state in states),
        observation_records=tuple(records), initial_distance=steps[0].before_distance,
        final_distance=steps[-1].after_distance, terminal_improvement=improvement, summed_step_gain=summed_gain)


def evaluate_teacher_diagnostic_v2(
    sample_key: V24SampleKey, *, config: Teacher_diagnostic_config_v2,
    branch_order: tuple[str, ...] = branch_kinds,
) -> Teacher_diagnostic_result_v2:
    """Run one paired synthetic oracle unit, with no file, model, or admission I/O.

    Validation precedes the sole initial renderer call. All randomness comes
    from key-local Philox streams; no process RNG state is read or changed.
    Source/physical preflight and eventual publication require a separate entry.
    """
    if not isinstance(config, Teacher_diagnostic_config_v2):
        raise ValueError("a teacher diagnostic V2 configuration is required")
    config.validate(sample_key)
    if len(branch_order) != len(branch_kinds) or set(branch_order) != set(branch_kinds):
        raise ValueError("branch_order must contain teacher_v1, teacher_v2 and zero exactly once")
    initial = _render_initial_context(sample_key, config)
    initial_context_sha = _context_identity(initial)
    shots = _feedback_shots(len(initial.settings), config.shot_mode, config.shot_budget)
    results = {}
    for branch_kind in branch_order:
        results[branch_kind] = _run_branch(initial, sample_key, config, shots.copy(), branch_kind)
        if _context_identity(initial) != initial_context_sha:
            raise ValueError("branch evaluation mutated the common initial context")
    branches = tuple(results[name] for name in branch_kinds)
    schedules = tuple(record.schedule for record in branches[0].observation_records)
    if any(branch.initial_context_sha != initial_context_sha
           or tuple(record.schedule for record in branch.observation_records) != schedules
           or branch.observation_records[0] != branches[0].observation_records[0] for branch in branches):
        raise ValueError("branches do not share the complete episode and observation panels")
    recipe = asdict(config)
    recipe.pop("replicate_index")  # Replicates couple observation noise only.
    return Teacher_diagnostic_result_v2(
        sample_key=sample_key, config=config,
        episode_sha=sha256_json({"sample_key_sha": sample_key.sha256(), "recipe": recipe,
                                "initial_context_sha": initial_context_sha}),
        initial_context_sha=initial_context_sha,
        target_coefficients_re=tuple(float(x) for x in initial.target_coefficients.real),
        target_coefficients_im=tuple(float(x) for x in initial.target_coefficients.imag),
        intervention_descriptor_json=json.dumps(asdict(initial.intervention), sort_keys=True, allow_nan=False),
        intervention_type=type(initial.intervention).__qualname__,
        physical_parameter=initial.physical_parameter, settings=initial.settings,
        shots_by_setting=tuple(int(value) for value in shots),
        sampled_attempts_per_branch=0 if config.shot_mode == "S_ideal" else (round_count + 1) * int(shots.sum()),
        observation_panel_shas=tuple(schedule.panel_sha for schedule in schedules), branches=branches)


__all__ = ["Teacher_diagnostic_config_v2", "Teacher_diagnostic_step_v2", "Teacher_diagnostic_branch_v2",
           "Teacher_diagnostic_result_v2", "evaluate_teacher_diagnostic_v2"]
