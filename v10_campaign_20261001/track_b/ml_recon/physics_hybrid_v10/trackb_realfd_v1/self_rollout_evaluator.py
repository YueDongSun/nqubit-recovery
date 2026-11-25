"""Observable-only, frozen CPU diagnostics on actual ten-action V2.4 trajectories.

The registered round-zero context is regenerated and compared exactly before use.
Only that initial context is retained; teacher replay states never drive a model
branch. ``_feedback_shots`` and ``_trackb_receding_horizon_action`` are narrow
private imports from the registered renderer: respectively its shot allocation
and its oracle reference policy. State transitions and observations use public
registered helpers. No source, checkpoint, dataset or report is written here.

Callbacks receive detached copies of observable inputs only. Their own captured
objects remain their responsibility; use ``evaluate_model_self_rollout`` to
isolate a caller-owned model. Run synchronously: restoring process RNG is not a
concurrency guard for other threads consuming those same RNGs.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math
import random
from typing import Callable, Iterator, Mapping

import numpy as np
from numpy.typing import NDArray
import torch

from ml_recon.data_v2_4.hashing import sha256_array, sha256_json
from ml_recon.data_v2_4.sample_keys import V24SampleKey, make_sample_key
from ml_recon.data_v2_4.synthetic.feedback_renderers import (
    CONTROL_AMPLITUDE_BOUND,
    CONTROL_PHASE_BOUND,
    TRACKB_TEN_ROUND_HISTORY,
    TrackBRecedingHorizonForwardContextV1,
    _feedback_shots,
    _trackb_receding_horizon_action,
    evaluate_trackb_candidate_transition,
    replay_trackb_ten_round_forward_contexts,
)
from ml_recon.data_v2_4.version import GENERATOR_SHA

from .candidate_action_evaluator import (
    MATERIAL_NEGATIVE_GAIN_TOLERANCE,
    is_materially_negative_trackb_gain,
)
from .inverse_transformer import (
    LEGACY_ORACLE_HISTORY_GAIN_INPUT,
    TrackBInverseTransformerV1,
    UNOBSERVED_ZERO_HISTORY_GAIN_INPUT,
)
from .model_self_rollout_core import (
    Rollout_branch_spec, Rollout_core_adapters, execute_rollout_branch,
)
from .self_rollout_observation import (
    Acquired_observation_v2, Observation_schedule_v2, acquire_observation,
    branch_kinds, common_pairing, observation_stream, pairing_kinds,
)
from .self_rollout_shift_contract import (
    LEGACY_ORACLE_GAIN_SHIFT, TRAINING_GAIN_INPUT_UNKNOWN,
    UNOBSERVED_ZERO_GAIN_SHIFT, history_gain_distribution_shift_kind,
)


history_gain_input_kind = UNOBSERVED_ZERO_HISTORY_GAIN_INPUT
gauge_tolerance = 1e-6  # Registered candidate-action gauge tolerance.
round_count = TRACKB_TEN_ROUND_HISTORY
Token_rows = tuple[tuple[float, ...], ...]
Rollout_policy = Callable[[Mapping[str, torch.Tensor]], NDArray[np.float64]]


@dataclass(frozen=True)
class Self_rollout_config_v1:
    """V1 is CPU-only, ten actions, and uses the registered feedback mixture."""

    system_type: str
    shot_mode: str = "S_setting"
    shot_budget: int = 256
    noise_family: str = "feedback_mixed"
    eta: float = 0.08
    kappa: float = 0.15
    amplitude_bound: float = CONTROL_AMPLITUDE_BOUND
    phase_bound: float = CONTROL_PHASE_BOUND
    history_gain_input_kind: str = history_gain_input_kind
    observation_stream: str = observation_stream
    replicate_index: int = 0
    pairing_kind: str = common_pairing
    include_teacher_state_control: bool = False

    def validate(self) -> None:
        if self.system_type not in ("3qubit", "4qubit"):
            raise ValueError("self rollout requires 3qubit or 4qubit")
        if self.shot_mode not in ("S_setting", "S_total", "S_ideal"):
            raise ValueError("unsupported registered shot mode")
        if type(self.shot_budget) is not int or self.shot_budget <= 0:
            raise ValueError("shot budget must be a positive integer")
        if self.noise_family != "feedback_mixed":
            raise ValueError("V1 requires registered feedback_mixed noise")
        for name in ("eta", "kappa"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite fraction")
        for name, maximum in (("amplitude_bound", CONTROL_AMPLITUDE_BOUND),
                              ("phase_bound", CONTROL_PHASE_BOUND)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"{name} exceeds the registered logical bound")
        if self.history_gain_input_kind != history_gain_input_kind:
            raise ValueError("only unobserved_zero_v1 evaluator history is supported")
        if self.observation_stream != observation_stream:
            raise ValueError("unregistered observation RNG stream")
        if type(self.replicate_index) is not int or self.replicate_index < 0:
            raise ValueError("replicate_index must be a nonnegative integer")
        if self.pairing_kind not in pairing_kinds:
            raise ValueError("unregistered observation pairing kind")
        if type(self.include_teacher_state_control) is not bool:
            raise TypeError("include_teacher_state_control must be boolean")
        _feedback_shots(2 * int(self.system_type[0]) - 1, self.shot_mode, self.shot_budget)


@dataclass(frozen=True)
class Rollout_step_v1:
    round_number: int
    pre_state_sha: str
    post_state_sha: str
    action_sha: str
    pre_observation_sha: str
    post_observation_sha: str
    pre_observation_record_id: str
    post_observation_record_id: str
    pre_observation: Token_rows
    post_observation: Token_rows
    action: Token_rows
    before_distance: float
    after_distance: float
    synthetic_gain: float
    harmful: bool


@dataclass(frozen=True)
class Rollout_branch_v1:
    branch_kind: str
    steps: tuple[Rollout_step_v1, ...]
    initial_distance: float
    final_distance: float
    telescoping_gain: float
    summed_step_gain: float
    final_state_sha: str
    observation_records: tuple[Acquired_observation_v2, ...]
    observation_record_ids: tuple[str, ...]
    acquisition_schedule: tuple[Observation_schedule_v2, ...]
    selected_round: int = round_count
    observation_count: int = round_count + 1


@dataclass(frozen=True)
class Self_rollout_result_v1:
    sample_id: str
    sample_key_sha: str
    round_zero_binding_sha: str
    config: Self_rollout_config_v1
    branches: tuple[Rollout_branch_v1, ...]
    shots_by_setting: tuple[int, ...]
    shots_per_observation: int
    sampled_shots_per_branch: int
    observation_panel_shas: tuple[str, ...]
    replicate_index: int
    pairing_kind: str
    teacher_state_zero_history_gain_control: Teacher_state_candidate_control_v1 | None
    history_gain_input_kind: str = history_gain_input_kind
    observation_stream: str = observation_stream
    observation_substreams: tuple[int, ...] = tuple(range(round_count + 1))
    observation_substreams_scope: str = "WINDOW_INDICES_ONLY_SETTING_SUBSTREAMS_ARE_IN_BRANCH_SCHEDULES"
    crn_schedule: str = "PER_BRANCH_SAMPLE_REPLICATE_WINDOW_SETTING_SCHEDULE_V2"
    observation_panel_shas_scope: str = "MODEL_BRANCH_COMPATIBILITY_VIEW_USE_BRANCH_ACQUISITION_SCHEDULE"
    statistical_unit: str = "PAIRED_SOURCE_EPISODE_REPLICATE_NOT_THREE_INDEPENDENT_SAMPLES"
    history_gain_consumed_value: float = 0.0
    history_gain_consumed_dtype: str = "float32"
    history_gain_preprocessing: str = "DIRECT_FLOAT32_ZERO_NO_CENTERING_OR_SCALING_NOT_A_MISSINGNESS_MASK"
    input_distribution_shift: str = TRAINING_GAIN_INPUT_UNKNOWN
    score_kind: str = "SYNTHETIC_LATENT_MEASUREMENT_LAW_HELLINGER_SQUARED"
    action_representation: str = "FLOAT32_LOGICAL_ACTION_EXECUTED_AND_RECORDED_V1"
    schema: str = "TrackBSelfRolloutEvaluationV3"
    research_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False
    deployment_validation: bool = False


@dataclass(frozen=True)
class Teacher_state_candidate_step_v1:
    round_number: int
    pre_state_sha: str
    pre_observation_record_id: str
    teacher_history_observation_record_ids: tuple[str, ...]
    action: Token_rows
    action_sha: str
    candidate_after_state_sha: str
    before_distance: float
    candidate_after_distance: float
    candidate_gain: float
    harmful: bool


@dataclass(frozen=True)
class Teacher_state_candidate_control_v1:
    steps: tuple[Teacher_state_candidate_step_v1, ...]
    teacher_observation_record_ids: tuple[str, ...]
    control_kind: str = "teacher_state_zero_history_gain_candidate_v1"
    evaluation_kind: str = "COUNTERFACTUAL_ONE_STEP_ACTIONS_AT_NEW_TEACHER_BRANCH_STATES"
    is_true_rollout: bool = False
    additional_observation_count: int = 0
    history_action_source: str = "NEW_TEACHER_BRANCH_EXECUTED_ACTIONS"
    history_gain_input_kind: str = history_gain_input_kind
    history_gain_consumed_value: float = 0.0
    history_gain_consumed_dtype: str = "float32"
    history_gain_preprocessing: str = "DIRECT_FLOAT32_ZERO_NO_CENTERING_OR_SCALING_NOT_A_MISSINGNESS_MASK"


@contextmanager
def _preserve_rng() -> Iterator[None]:
    """Preserve initialized RNGs without initializing CUDA for a CPU evaluator."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _context_identity(context: TrackBRecedingHorizonForwardContextV1) -> dict[str, object]:
    context.validate()
    if context.pre_state_sha != sha256_array(context.before_coefficients):
        raise ValueError("round-zero state bytes do not match the bound pre-state SHA")
    return {
        "sample_id": context.sample_id,
        "round_index": context.round_index,
        "pre_state_sha": context.pre_state_sha,
        "before_coefficients_sha": sha256_array(context.before_coefficients),
        "target_coefficients_sha": sha256_array(context.target_coefficients),
        "teacher_action_sha": sha256_array(context.teacher_action),
        "teacher_gain": context.teacher_gain,
        "settings": context.settings,
        "physical_parameter": context.physical_parameter,
        "intervention_type": type(context.intervention).__qualname__,
        "intervention": asdict(context.intervention),
    }


def _bind_round_zero(
    sample_key: V24SampleKey,
    context: TrackBRecedingHorizonForwardContextV1 | None,
    config: Self_rollout_config_v1,
) -> tuple[TrackBRecedingHorizonForwardContextV1, str]:
    config.validate()
    if (not isinstance(sample_key, V24SampleKey)
            or (context is not None and not isinstance(context, TrackBRecedingHorizonForwardContextV1))):
        raise TypeError("registered sample key and round-zero context are required")
    sample_key.validate()
    if (type(sample_key.N) is not int or sample_key.N != int(config.system_type[0])
            or type(sample_key.global_seed) is not int
            or sample_key.generator_sha != GENERATOR_SHA
            or sample_key.dataset_family != "feedback_history"
            or sample_key.cell_id != f"trackb_feedback_history_n{sample_key.N}"):
        raise ValueError("sample key differs from the current registered Track B source")
    try:
        source_index = int(sample_key.sample_id.rsplit(":", 1)[-1])
    except (ValueError, AttributeError) as error:
        raise ValueError("sample identity lacks its canonical source index") from error
    canonical_key = make_sample_key(
        global_seed=sample_key.global_seed, split=sample_key.split,
        dataset_family=sample_key.dataset_family, N=sample_key.N, index=source_index,
        generator_sha=sample_key.generator_sha, cell_id=sample_key.cell_id,
    )
    if source_index < 0 or canonical_key != sample_key:
        raise ValueError("sample identity is not canonical")
    if context is not None and (type(context.round_index) is not int
                                or context.round_index != 0 or context.sample_id != sample_key.sample_id):
        raise ValueError("context must be the bound sample's round zero")
    canonical = replay_trackb_ten_round_forward_contexts(
        sample_key, shot_mode=config.shot_mode, shot_budget=config.shot_budget,
        noise_family=config.noise_family, eta=config.eta, kappa=config.kappa,
    )[0]
    expected = _context_identity(canonical)
    if context is not None and _context_identity(context) != expected:
        raise ValueError("round-zero context does not exactly match registered source replay")
    return canonical, sha256_json({
        "sample_key_sha": sample_key.sha256(), "config": asdict(config), "context": expected,
    })


def _validate_action(action: NDArray[np.float64], mode_count: int,
                     config: Self_rollout_config_v1) -> NDArray[np.float64]:
    value = np.asarray(action)
    if value.shape != (mode_count, 2) or value.dtype.kind not in "fiu":
        raise ValueError("policy action must be a real [modes,2] array")
    value = value.astype(np.float64, copy=True)
    if not np.isfinite(value).all():
        raise ValueError("policy action must be finite")
    if np.any(np.abs(value.sum(axis=0)) > gauge_tolerance):
        raise ValueError("policy action must be gauge centered")
    if np.any(np.abs(value) > np.array([config.amplitude_bound, config.phase_bound])):
        raise ValueError("policy action exceeds configured logical bounds")
    # Execute the same float32 values stored in V2.4 history, including references.
    value = value.astype(np.float32).astype(np.float64)
    if (np.any(np.abs(value.sum(axis=0)) > gauge_tolerance)
            or np.any(np.abs(value) > np.array([config.amplitude_bound, config.phase_bound]))):
        raise ValueError("float32 action exceeds the configured gauge or logical bounds")
    return value


def _token_rows(value: NDArray[np.floating]) -> Token_rows:
    return tuple(tuple(float(item) for item in row) for row in value)


def _observable_inputs(
    history: list[NDArray[np.float32]],
    observation: Acquired_observation_v2,
    target_tokens: NDArray[np.float64],
) -> dict[str, torch.Tensor]:
    """Only established observable tensors cross into callbacks, without record metadata."""
    mode_count = len(observation.measurement_tokens)
    index = len(history)
    return {
        "history_tokens": torch.tensor(np.asarray(history, dtype=np.float32).reshape(
            1, index, mode_count, 7), device="cpu"),
        "history_mask": torch.ones((1, index), dtype=torch.bool, device="cpu"),
        "noisy_measurement_tokens": torch.tensor([observation.measurement_tokens], dtype=torch.float32, device="cpu"),
        "target_mode_tokens": torch.tensor(target_tokens[None], dtype=torch.float32, device="cpu"),
        "mode_mask": torch.ones((1, mode_count), dtype=torch.bool, device="cpu"),
    }


def _score_transition(
    context: TrackBRecedingHorizonForwardContextV1,
    action: NDArray[np.float64],
) -> tuple[TrackBRecedingHorizonForwardContextV1, float, float, float]:
    # Registered density normalization mutates arrays, so isolate it from the ledger.
    score = evaluate_trackb_candidate_transition(deepcopy(context), action)
    before_distance, after_distance, gain = (float(score[name]) for name in (
        "before_distance", "after_distance", "gain"))
    after = np.asarray(score["after_coefficients"], dtype=np.complex128).copy()
    if (not all(math.isfinite(value) for value in (before_distance, after_distance, gain))
            or min(before_distance, after_distance) < 0
            or after.shape != context.before_coefficients.shape or not np.isfinite(after).all()
            or not math.isclose(float(np.linalg.norm(after)), 1.0, abs_tol=1e-10)):
        raise ValueError("registered forward returned an invalid state or score")
    if not math.isclose(gain, before_distance - after_distance, abs_tol=1e-12, rel_tol=1e-10):
        raise ValueError("registered forward gain does not match its distances")
    if not np.any(action):
        # Registered log-amplitude stabilization adds epsilon even for zero.
        # Verify physical equivalence, then retain the exact no-op state bytes.
        phase = np.angle(np.vdot(context.before_coefficients, after))
        if (not np.allclose(after * np.exp(-1j * phase), context.before_coefficients,
                            atol=MATERIAL_NEGATIVE_GAIN_TOLERANCE, rtol=0)
                or abs(gain) > MATERIAL_NEGATIVE_GAIN_TOLERANCE):
            raise ValueError("registered zero action did not preserve physical state")
        after = context.before_coefficients.copy()
        after_distance, gain = before_distance, 0.0
    post_context = replace(context, before_coefficients=after, pre_state_sha=sha256_array(after))
    return post_context, before_distance, after_distance, gain


def _run_branch(
    initial_context: TrackBRecedingHorizonForwardContextV1,
    sample_key: V24SampleKey,
    shots: NDArray[np.int64],
    policy: Rollout_policy,
    branch_kind: str,
    config: Self_rollout_config_v1,
    control_policy: Rollout_policy | None,
) -> tuple[Rollout_branch_v1, Teacher_state_candidate_control_v1 | None]:
    return execute_rollout_branch(
        initial_context, sample_key, shots, policy,
        branch_spec=Rollout_branch_spec(branch_kind, branch_kind, branch_kind),
        config=config,
        teacher_policy=lambda context: _trackb_receding_horizon_action(
            context.before_coefficients, context.target_coefficients)[0],
        adapters=rollout_core_adapters(), control_policy=control_policy,
    )


def rollout_core_adapters() -> Rollout_core_adapters:
    """Bind current helper objects so V1 monkeypatch seams remain observable."""
    return Rollout_core_adapters(
        acquire_observation=acquire_observation, observable_inputs=_observable_inputs,
        validate_action=_validate_action, score_transition=_score_transition,
        token_rows=_token_rows, harmful_gain=is_materially_negative_trackb_gain,
        step_type=Rollout_step_v1, branch_type=Rollout_branch_v1,
        control_step_type=Teacher_state_candidate_step_v1,
        control_type=Teacher_state_candidate_control_v1,
    )


def evaluate_self_rollout(
    sample_key: V24SampleKey,
    round_zero_context: TrackBRecedingHorizonForwardContextV1 | None,
    policy: Rollout_policy,
    *,
    config: Self_rollout_config_v1,
    teacher_state_policy: Rollout_policy | None = None,
    branch_order: tuple[str, ...] = branch_kinds,
) -> Self_rollout_result_v1:
    """Evaluate model/zero/teacher branches, ten actions each, with observable callbacks.

    The callback receives exactly the five inverse-model input tensors on CPU,
    with a batch size of one, and returns a physical logical action [modes,2].
    Synthetic truth never crosses this boundary. This is a diagnostic input
    shift, not validation of an oracle-gain-trained model's deployment inputs.
    A supplied initial context must match replay exactly; None uses replay alone.
    """
    with _preserve_rng():
        if not callable(policy):
            raise TypeError("policy must be callable")
        if len(branch_order) != 3 or set(branch_order) != set(branch_kinds):
            raise ValueError("branch_order must contain model, zero and teacher exactly once")
        if teacher_state_policy is not None and not callable(teacher_state_policy):
            raise TypeError("teacher_state_policy must be callable")
        if teacher_state_policy is not None and not config.include_teacher_state_control:
            raise ValueError("teacher-state callback requires explicitly enabled control")
        initial_context, binding_sha = _bind_round_zero(sample_key, round_zero_context, config)
        shots = _feedback_shots(len(initial_context.settings), config.shot_mode, config.shot_budget)
        results = {}
        control = None
        control_policy = (teacher_state_policy if teacher_state_policy is not None else policy) \
            if config.include_teacher_state_control else None
        for branch_kind in branch_order:
            # A callback's process RNG consumption must not couple branch order.
            # Captured mutable callback objects remain caller-owned; use pure callbacks.
            with _preserve_rng():
                branch, candidate_control = _run_branch(
                    initial_context, sample_key, shots, policy, branch_kind, config, control_policy)
            results[branch_kind] = branch
            if candidate_control is not None:
                control = candidate_control
        branches = tuple(results[name] for name in branch_kinds)
        return Self_rollout_result_v1(
            sample_id=sample_key.sample_id, sample_key_sha=sample_key.sha256(),
            round_zero_binding_sha=binding_sha, config=config, branches=branches,
            shots_by_setting=tuple(int(value) for value in shots),
            shots_per_observation=int(shots.sum()),
            sampled_shots_per_branch=(0 if config.shot_mode == "S_ideal" else
                                      (round_count + 1) * int(shots.sum())),
            observation_panel_shas=tuple(schedule.panel_sha for schedule in branches[0].acquisition_schedule),
            replicate_index=config.replicate_index, pairing_kind=config.pairing_kind,
            teacher_state_zero_history_gain_control=control,
        )


def evaluate_model_self_rollout(
    model: TrackBInverseTransformerV1,
    sample_key: V24SampleKey,
    round_zero_context: TrackBRecedingHorizonForwardContextV1 | None,
    *,
    config: Self_rollout_config_v1,
) -> Self_rollout_result_v1:
    """Infer on an eval-mode CPU deepcopy; preserve caller tensors, grads and modes."""
    with _preserve_rng():
        config.validate()
        if not isinstance(model, TrackBInverseTransformerV1):
            raise TypeError("a bound TrackBInverseTransformerV1 is required")
        model.config.validate()
        if (model.config.system_type != config.system_type
                or model.config.amplitude_bound != config.amplitude_bound
                or model.config.phase_bound != config.phase_bound):
            raise ValueError("model system and logical bounds must match rollout configuration")
        tensors = tuple(model.parameters()) + tuple(model.buffers())
        if any(value.device.type != "cpu" for value in tensors):
            raise ValueError("self rollout V1 requires an already CPU-bound model")
        if any(value.dtype != torch.float32 for value in model.parameters()):
            raise ValueError("self rollout V1 requires float32 model parameters")
        if any(not bool(torch.isfinite(value).all()) for value in tensors if value.is_floating_point()):
            raise ValueError("bound model contains non-finite tensors")
        def isolated_policy() -> Rollout_policy:
            frozen_model = deepcopy(model)
            frozen_model.eval()

            def policy(inputs: Mapping[str, torch.Tensor]) -> NDArray[np.float64]:
                output = frozen_model(inputs)
                return torch.stack((output["amplitude_delta"][0], output["phase_delta"][0]),
                                   dim=-1).detach().cpu().numpy().astype(np.float64)

            return policy

        result = evaluate_self_rollout(
            sample_key, round_zero_context, isolated_policy(), config=config,
            teacher_state_policy=isolated_policy() if config.include_teacher_state_control else None,
        )
        return replace(
            result,
            input_distribution_shift=history_gain_distribution_shift_kind(
                model.config.history_gain_input_kind,
            ),
        )


def evaluate_trackb_model_self_rollout(
    model: TrackBInverseTransformerV1,
    key: V24SampleKey,
    *,
    shot_mode: str,
    shot_budget: int,
    noise_family: str,
    eta: float,
    kappa: float,
    replicate_index: int = 0,
    pairing_kind: str = common_pairing,
    include_teacher_state_control: bool = True,
) -> dict[str, object]:
    """Return JSON-compatible sample diagnostics; external lineage belongs to caller.

    Replays the registered source once. The caller binds the selected key and
    model to its receipt, checkpoint, data root and producing Git revision.
    This function verifies the current generator and canonical Track B sample
    identity, but does not claim to open or verify those external artifacts.
    """
    if not isinstance(model, TrackBInverseTransformerV1):
        raise TypeError("a bound TrackBInverseTransformerV1 is required")
    config = Self_rollout_config_v1(
        system_type=model.config.system_type, shot_mode=shot_mode, shot_budget=shot_budget,
        noise_family=noise_family, eta=eta, kappa=kappa,
        amplitude_bound=model.config.amplitude_bound, phase_bound=model.config.phase_bound,
        replicate_index=replicate_index, pairing_kind=pairing_kind,
        include_teacher_state_control=include_teacher_state_control,
    )
    return asdict(evaluate_model_self_rollout(model, key, None, config=config))


__all__ = [
    "Self_rollout_config_v1", "Rollout_policy", "Rollout_step_v1", "Rollout_branch_v1",
    "Self_rollout_result_v1", "evaluate_self_rollout", "evaluate_model_self_rollout",
    "evaluate_trackb_model_self_rollout", "Teacher_state_candidate_step_v1", "Teacher_state_candidate_control_v1",
    "TRAINING_GAIN_INPUT_UNKNOWN", "LEGACY_ORACLE_GAIN_SHIFT", "UNOBSERVED_ZERO_GAIN_SHIFT",
    "history_gain_distribution_shift_kind",
    "rollout_core_adapters",
]
