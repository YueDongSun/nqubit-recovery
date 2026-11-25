"""Pure ten-action model/zero/teacher execution after external source admission.

The caller supplies one admitted initial context and a teacher policy. Nothing
here renders a source episode, reads artifacts, trains a model, or publishes
evidence. The teacher receives its own current state on every action. All roles
use the same injected float32 action validator and latent-distance transition.

Output constructors and narrow compute adapters keep the established V1 data
classes and monkeypatch boundaries at their owning module. There is no runtime
import of that module. Call synchronously: process RNG restoration does not
coordinate other threads, and callback-captured mutable objects remain owned by
the caller.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
import math
import random
from typing import Any, Callable, Iterator, Mapping, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import torch

from ml_recon.data_v2_4.hashing import sha256_array
from ml_recon.data_v2_4.sample_keys import V24SampleKey
from ml_recon.data_v2_4.synthetic.feedback_renderers import (
    TRACKB_TEN_ROUND_HISTORY,
    TrackBRecedingHorizonForwardContextV1,
)

from .inverse_transformer import TrackBInverseTransformerV1
from .self_rollout_observation import Acquired_observation_v2

if TYPE_CHECKING:
    from .self_rollout_evaluator import (
        Rollout_branch_v1,
        Self_rollout_config_v1,
        Teacher_state_candidate_control_v1,
    )


round_count = TRACKB_TEN_ROUND_HISTORY
Rollout_policy = Callable[[Mapping[str, torch.Tensor]], NDArray[np.float64]]
Teacher_policy = Callable[[TrackBRecedingHorizonForwardContextV1], NDArray[np.float64]]


@dataclass(frozen=True)
class Rollout_branch_spec:
    """Separate output identity, policy role, and registered observation role."""

    branch_kind: str
    policy_kind: str
    observation_branch_kind: str

    def validate(self) -> None:
        if not isinstance(self.branch_kind, str) or not self.branch_kind:
            raise ValueError("rollout branch name must be nonempty")
        if self.policy_kind not in ("model", "zero", "teacher"):
            raise ValueError("rollout policy role must be model, zero or teacher")
        if self.observation_branch_kind != self.policy_kind:
            raise ValueError("observation role must match the explicit policy role")


@dataclass(frozen=True)
class Rollout_core_adapters:
    """Explicit V1-compatible compute seams and result constructors."""

    acquire_observation: Callable[..., Acquired_observation_v2]
    observable_inputs: Callable[..., Mapping[str, torch.Tensor]]
    validate_action: Callable[..., NDArray[np.float64]]
    score_transition: Callable[..., tuple[TrackBRecedingHorizonForwardContextV1, float, float, float]]
    token_rows: Callable[..., tuple[tuple[float, ...], ...]]
    harmful_gain: Callable[[float], bool]
    step_type: Callable[..., Any]
    branch_type: Callable[..., Rollout_branch_v1]
    control_step_type: Callable[..., Any]
    control_type: Callable[..., Teacher_state_candidate_control_v1]


@dataclass(frozen=True)
class Rollout_computation:
    branches: tuple[Rollout_branch_v1, ...]
    teacher_state_control: Teacher_state_candidate_control_v1 | None


@contextmanager
def preserve_process_rng() -> Iterator[None]:
    """Restore initialized RNGs, without initializing CUDA in this CPU path."""
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


def isolated_model_policy(
    model: TrackBInverseTransformerV1,
    *,
    config: Self_rollout_config_v1,
) -> Rollout_policy:
    """Validate and clone an already CPU-bound float32 model for evaluation."""
    with preserve_process_rng():
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
        frozen_model = deepcopy(model)
        frozen_model.eval()

    def policy(inputs: Mapping[str, torch.Tensor]) -> NDArray[np.float64]:
        output = frozen_model(inputs)
        return torch.stack((output["amplitude_delta"][0], output["phase_delta"][0]),
                           dim=-1).detach().cpu().numpy().astype(np.float64)

    return policy


def execute_rollout_branch(
    initial_context: TrackBRecedingHorizonForwardContextV1,
    sample_key: V24SampleKey,
    shots: NDArray[np.int64],
    policy: Rollout_policy,
    *,
    branch_spec: Rollout_branch_spec,
    config: Self_rollout_config_v1,
    teacher_policy: Teacher_policy,
    adapters: Rollout_core_adapters,
    control_policy: Rollout_policy | None = None,
) -> tuple[Rollout_branch_v1, Teacher_state_candidate_control_v1 | None]:
    """Execute ten actions and reuse each acquired post record as the next pre."""
    branch_spec.validate()
    context = deepcopy(initial_context)
    acquisition = dict(shot_mode=config.shot_mode, branch_kind=branch_spec.observation_branch_kind,
                       replicate_index=config.replicate_index, pairing_kind=config.pairing_kind)
    observation = adapters.acquire_observation(context, sample_key, shots, window_index=0, **acquisition)
    mode_count = len(context.before_coefficients)
    target = context.target_coefficients
    target_tokens = np.stack((np.abs(target), np.angle(target * target[0].conj()) / np.pi), axis=-1)
    history: list[NDArray[np.float32]] = []
    records = [observation]
    steps = []
    control_steps = []
    for index in range(round_count):
        if branch_spec.policy_kind == "teacher" and control_policy is not None:
            with torch.no_grad():
                candidate = control_policy(adapters.observable_inputs(history, observation, target_tokens))
            candidate = adapters.validate_action(candidate, mode_count, config)
            counterfactual, before, after, gain = adapters.score_transition(context, candidate)
            control_steps.append(adapters.control_step_type(
                round_number=index + 1, pre_state_sha=context.pre_state_sha,
                pre_observation_record_id=observation.record_id,
                teacher_history_observation_record_ids=tuple(record.record_id for record in records),
                action=adapters.token_rows(candidate), action_sha=sha256_array(candidate),
                candidate_after_state_sha=counterfactual.pre_state_sha,
                before_distance=before, candidate_after_distance=after, candidate_gain=gain,
                harmful=adapters.harmful_gain(gain),
            ))
        if branch_spec.policy_kind == "model":
            with torch.no_grad():
                action = policy(adapters.observable_inputs(history, observation, target_tokens))
        elif branch_spec.policy_kind == "zero":
            action = np.zeros((mode_count, 2), dtype=np.float64)
        else:
            action = teacher_policy(deepcopy(context))
        action = adapters.validate_action(action, mode_count, config)
        post_context, before_distance, after_distance, gain = adapters.score_transition(context, action)
        if steps and not math.isclose(steps[-1].after_distance, before_distance, abs_tol=1e-12, rel_tol=1e-10):
            raise ValueError("trajectory distances are not continuous")
        post_observation = adapters.acquire_observation(
            post_context, sample_key, shots, window_index=index + 1, **acquisition)
        steps.append(adapters.step_type(
            round_number=index + 1, pre_state_sha=context.pre_state_sha,
            post_state_sha=post_context.pre_state_sha, action_sha=sha256_array(action),
            pre_observation_sha=observation.measurement_token_sha,
            post_observation_sha=post_observation.measurement_token_sha,
            pre_observation_record_id=observation.record_id,
            post_observation_record_id=post_observation.record_id,
            pre_observation=observation.measurement_tokens, post_observation=post_observation.measurement_tokens,
            action=adapters.token_rows(action), before_distance=before_distance,
            after_distance=after_distance, synthetic_gain=gain, harmful=adapters.harmful_gain(gain),
        ))
        history.append(np.concatenate((np.asarray(observation.measurement_tokens, dtype=np.float32),
                                       action.astype(np.float32),
                                       np.asarray(post_observation.measurement_tokens, dtype=np.float32),
                                       np.zeros((mode_count, 1), dtype=np.float32)), axis=-1))
        records.append(post_observation)
        observation = post_observation
        context = replace(post_context, round_index=min(index + 1, round_count - 1))
    telescoping_gain = steps[0].before_distance - steps[-1].after_distance
    summed_step_gain = math.fsum(step.synthetic_gain for step in steps)
    if not math.isclose(summed_step_gain, telescoping_gain, abs_tol=1e-12, rel_tol=1e-10):
        raise ValueError("self rollout gains do not telescope")
    record_ids = tuple(record.record_id for record in records)
    branch = adapters.branch_type(
        branch_kind=branch_spec.branch_kind, steps=tuple(steps), initial_distance=steps[0].before_distance,
        final_distance=steps[-1].after_distance, telescoping_gain=telescoping_gain,
        summed_step_gain=summed_step_gain, final_state_sha=steps[-1].post_state_sha,
        observation_records=tuple(records), observation_record_ids=record_ids,
        acquisition_schedule=tuple(record.schedule for record in records),
    )
    control = adapters.control_type(tuple(control_steps), record_ids) if control_steps else None
    return branch, control


def execute_rollout_branches(
    initial_context: TrackBRecedingHorizonForwardContextV1,
    sample_key: V24SampleKey,
    shots: NDArray[np.int64],
    policy: Rollout_policy,
    *,
    config: Self_rollout_config_v1,
    teacher_policy: Teacher_policy,
    adapters: Rollout_core_adapters,
    branch_specs: tuple[Rollout_branch_spec, ...],
    branch_order: tuple[str, ...] | None = None,
    teacher_state_policy: Rollout_policy | None = None,
) -> Rollout_computation:
    """Run admitted branches with RNG isolation and stable declared output order."""
    with preserve_process_rng():
        config.validate()
        if not callable(policy) or not callable(teacher_policy):
            raise TypeError("model and teacher policies must be callable")
        if teacher_state_policy is not None and not callable(teacher_state_policy):
            raise TypeError("teacher_state_policy must be callable")
        if teacher_state_policy is not None and not config.include_teacher_state_control:
            raise ValueError("teacher-state callback requires explicitly enabled control")
        for spec in branch_specs:
            spec.validate()
        names = tuple(spec.branch_kind for spec in branch_specs)
        if (len(names) != 3 or len(set(names)) != 3
                or {spec.policy_kind for spec in branch_specs} != {"model", "zero", "teacher"}):
            raise ValueError("branch specs require model, zero and teacher exactly once")
        order = names if branch_order is None else branch_order
        if len(order) != 3 or set(order) != set(names):
            raise ValueError("branch_order must contain the three declared branches exactly once")
        specs_by_name = {spec.branch_kind: spec for spec in branch_specs}
        control_policy = (teacher_state_policy if teacher_state_policy is not None else policy) \
            if config.include_teacher_state_control else None
        results = {}
        control = None
        for name in order:
            with preserve_process_rng():
                branch, candidate_control = execute_rollout_branch(
                    initial_context, sample_key, shots, policy, branch_spec=specs_by_name[name],
                    config=config, teacher_policy=teacher_policy, adapters=adapters,
                    control_policy=control_policy,
                )
            results[name] = branch
            if candidate_control is not None:
                control = candidate_control
        return Rollout_computation(tuple(results[name] for name in names), control)


__all__ = [
    "Rollout_branch_spec", "Rollout_core_adapters", "Rollout_computation",
    "Rollout_policy", "Teacher_policy", "execute_rollout_branch", "execute_rollout_branches",
    "isolated_model_policy", "preserve_process_rng",
]
