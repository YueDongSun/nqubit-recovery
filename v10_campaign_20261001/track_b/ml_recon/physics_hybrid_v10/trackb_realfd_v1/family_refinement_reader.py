"""Independent frozen-parent plus nine-parameter refinement artifact reader."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from typing import Any

import torch

from .contracts import canonical_json_bytes, parse_canonical_json_object, require_git_oid, require_sha256
from .effective_checkpoint_reader import Effective_checkpoint_replay
from .effective_observation_v2 import Effective_singleton_observation, validate_global_initialization_buffers
from .family_block_derivatives import family_parameter_name
from .family_refinement_contract import checked_vector, validate_family_refinement_result
from .training import trackb_model_state_sha


artifact_schema = 'TrackBFamilyBlockRefinementArtifactV1'
maximum_artifact_bytes = 2 * 1024 * 1024
train_objective_replay_absolute_tolerance = 1e-8
artifact_fields = frozenset({'schema', 'source_commit', 'source_tree', 'solver_result', 'refined_model_state_sha',
    'runtime', 'research_only', 'hardware_action_authorized', 'scientific_gate_issued'})


def _checked_artifact(payload):
    artifact = parse_canonical_json_object(payload, schema=artifact_schema, exact_fields=artifact_fields)
    require_git_oid(artifact['source_commit'], 'refinement_source_commit')
    require_git_oid(artifact['source_tree'], 'refinement_source_tree')
    require_sha256(artifact['refined_model_state_sha'], 'refined_model_state_sha')
    if artifact['research_only'] is not True or artifact['hardware_action_authorized'] is not False \
            or artifact['scientific_gate_issued'] is not False:
        raise ValueError('Refinement authority fields are invalid')
    runtime = artifact['runtime']
    if type(runtime) is not dict or set(runtime) != {'device', 'theta_precision', 'linear_algebra_precision', 'torch_version', 'cpu_threads'} \
            or runtime['device'] != 'cpu' or runtime['theta_precision'] != 'fp32' or runtime['linear_algebra_precision'] != 'fp64' \
            or type(runtime['cpu_threads']) is not int or runtime['cpu_threads'] < 1 \
            or type(runtime['torch_version']) is not str or not runtime['torch_version']:
        raise ValueError('Refinement runtime is not the registered CPU precision contract')
    return artifact


def _same_tensor_bytes(first, second):
    return first.shape == second.shape and first.dtype == second.dtype and torch.equal(
        first.detach().contiguous().reshape(-1).view(torch.uint8),
        second.detach().contiguous().reshape(-1).view(torch.uint8))


def materialize_refined_family_model(parent: Effective_checkpoint_replay, theta) -> Effective_singleton_observation:
    parent.summary()
    value = checked_vector(theta, 'refined theta', fp32=True)
    model = copy.deepcopy(parent.final_model)
    if model.config.measurement_context is None:
        raise ValueError('Refinement requires a family-context parent')
    with torch.no_grad():
        model.family_context.raw_coefficients.copy_(value.reshape(3, 3))
    model.eval().requires_grad_(False)
    reference = parent.final_model.state_dict()
    state = model.state_dict()
    if set(state) != set(reference) or any(not _same_tensor_bytes(state[name], reference[name])
            for name in reference if name != family_parameter_name):
        raise ValueError('Refinement changed a non-family tensor')
    validate_global_initialization_buffers(model)
    parent.summary()
    return model


def build_family_refinement_artifact(result, parent: Effective_checkpoint_replay, *, source_commit: str, source_tree: str):
    require_git_oid(source_commit, 'refinement_source_commit')
    require_git_oid(source_tree, 'refinement_source_tree')
    validate_family_refinement_result(result, parent)
    final = materialize_refined_family_model(parent, result['final_theta'])
    return {'schema': artifact_schema, 'source_commit': source_commit, 'source_tree': source_tree,
        'solver_result': copy.deepcopy(result), 'refined_model_state_sha': trackb_model_state_sha(final),
        'runtime': {'device': 'cpu', 'theta_precision': 'fp32', 'linear_algebra_precision': 'fp64',
            'torch_version': str(torch.__version__), 'cpu_threads': torch.get_num_threads()},
        'research_only': True, 'hardware_action_authorized': False, 'scientific_gate_issued': False}


@dataclass(frozen=True)
class Family_refinement_replay:
    """Two-stage provenance; never an original AdamW runtime/checkpoint identity."""
    parent: Effective_checkpoint_replay
    final_model: Effective_singleton_observation
    artifact_bytes: bytes
    file_sha: str
    final_model_state_sha: str

    @property
    def identity(self):
        """Original training provenance only, for admitted data/query matching."""
        return self.parent.identity

    @property
    def initial_model(self):
        """Original training M0, not the starting point of the refinement."""
        return self.parent.initial_model

    def artifact(self):
        if hashlib.sha256(self.artifact_bytes).hexdigest() != self.file_sha:
            raise ValueError('Refinement in-memory artifact identity changed')
        return _checked_artifact(self.artifact_bytes)

    def summary(self) -> dict[str, Any]:
        self.parent.summary()
        artifact = self.artifact()
        result = artifact['solver_result']
        validate_family_refinement_result(result, self.parent)
        if self.final_model.config.sha256() != self.parent.final_model.config.sha256() \
                or any(module.training for module in self.final_model.modules()) \
                or any(parameter.requires_grad or parameter.grad is not None or parameter.device.type != 'cpu' for parameter in self.final_model.parameters()) \
                or any(buffer.device.type != 'cpu' for buffer in self.final_model.buffers()) \
                or trackb_model_state_sha(self.final_model) != self.final_model_state_sha \
                or self.final_model_state_sha != artifact['refined_model_state_sha']:
            raise ValueError('Refinement model state, configuration or frozen CPU mode changed')
        validate_global_initialization_buffers(self.final_model)
        reference = self.parent.final_model.state_dict()
        current = self.final_model.state_dict()
        if set(current) != set(reference) or any(not _same_tensor_bytes(current[name], reference[name])
                for name in reference if name != family_parameter_name):
            raise ValueError('Refinement no longer preserves every non-family tensor')
        return {'schema': 'TrackBFamilyRefinementInspectionV1',
            'artifact_kind': 'FROZEN_PARENT_PLUS_FAMILY_OVERLAY_NOT_ADAMW_CHECKPOINT',
            'checkpoint_file_sha': self.file_sha, 'parent_checkpoint_file_sha': self.parent.file_sha,
            'parent_identity_sha': self.identity.sha256(), 'source_commit': artifact['source_commit'],
            'source_tree': artifact['source_tree'], 'parent_producer_commit': self.identity.source_commit,
            'parent_producer_tree': self.identity.source_tree, 'system_type': self.identity.system_type,
            'model_config_sha': self.identity.model_config_sha, 'training_initial_model_state_sha': self.identity.initial_model_state_sha,
            'refinement_initial_model_state_sha': self.parent.final_model_state_sha,
            'final_model_state_sha': self.final_model_state_sha, 'accepted_steps': result['accepted_steps'],
            'solver_config': result['solver_config'], 'stop_reason': result['stop_reason'],
            'objective_evaluations': result['objective_evaluations'], 'forward_evaluations': result['forward_evaluations'],
            'optimizer_constructed_by_reader': False, 'parent_training_resumed': False,
            'scientific_gate_issued': False,
            'scope': 'ARTIFACT_STRUCTURE_AND_NINE_PARAMETER_RELOAD_ONLY; TRAIN_OBJECTIVE_VALUES_REQUIRE_SEPARATE_REPLAY'}


def load_family_refinement_bytes(
    payload: bytes, parent: Effective_checkpoint_replay, *, expected_file_sha: str,
    expected_source_commit: str, expected_source_tree: str,
) -> Family_refinement_replay:
    require_sha256(expected_file_sha, 'refinement_file_sha')
    require_git_oid(expected_source_commit, 'refinement_source_commit')
    require_git_oid(expected_source_tree, 'refinement_source_tree')
    if type(payload) is not bytes or len(payload) > maximum_artifact_bytes or hashlib.sha256(payload).hexdigest() != expected_file_sha:
        raise ValueError('Refinement artifact bytes differ from the expected bounded input')
    artifact = _checked_artifact(payload)
    if (artifact['source_commit'], artifact['source_tree']) != (expected_source_commit, expected_source_tree) \
            or artifact['research_only'] is not True or artifact['hardware_action_authorized'] is not False \
            or artifact['scientific_gate_issued'] is not False:
        raise ValueError('Refinement source or authority fields are invalid')
    validate_family_refinement_result(artifact['solver_result'], parent)
    model = materialize_refined_family_model(parent, artifact['solver_result']['final_theta'])
    observed = trackb_model_state_sha(model)
    if observed != artifact['refined_model_state_sha']:
        raise ValueError('Refined model state does not match the artifact')
    replay = Family_refinement_replay(parent, model, payload, expected_file_sha, observed)
    replay.summary()
    return replay


def validate_refinement_parent_comparison(refined: Family_refinement_replay, parent: Effective_checkpoint_replay):
    if type(refined) is not Family_refinement_replay or type(parent) is not Effective_checkpoint_replay:
        raise TypeError('Refinement comparison requires its explicit two-stage replay types')
    refined.summary()
    parent.summary()
    if parent.file_sha != refined.parent.file_sha or parent.identity != refined.parent.identity \
            or parent.final_model_state_sha != refined.parent.final_model_state_sha:
        raise ValueError('Refinement comparison is not against its exact frozen parent')


def verify_refinement_train_objective(refined: Family_refinement_replay, dataset):
    """Independently re-evaluate both recorded TRAIN endpoints; never fit them."""
    from .family_block_objective import Frozen_family_training_objective

    refined.summary()
    result = refined.artifact()['solver_result']
    objective = Frozen_family_training_objective(refined.parent, dataset)
    if objective.binding != result['training_protocol']:
        raise ValueError('Refinement TRAIN protocol cannot be reconstructed from admitted data')
    maximum_error = 0.0
    observed = {}
    for name in ('initial', 'final'):
        actual = objective.evaluate(checked_vector(result[name + '_theta'], name + '_theta', fp32=True), derivatives=False)
        expected = result[name + '_loss_components']
        for component, value in actual['loss_components'].items():
            error = abs(value - expected[component])
            maximum_error = max(maximum_error, error)
            if error > train_objective_replay_absolute_tolerance:
                raise ValueError('Refinement endpoint native TRAIN objective does not reproduce')
        observed[name + '_loss_components'] = actual['loss_components']
    refined.summary()
    return {'schema': 'TrackBFamilyRefinementTrainReplayV1', 'refinement_file_sha': refined.file_sha,
        'parent_checkpoint_file_sha': refined.parent.file_sha, 'training_protocol': objective.binding,
        'maximum_objective_error': maximum_error, 'absolute_tolerance': train_objective_replay_absolute_tolerance,
        'objective_evaluations': objective.evaluations, 'forward_evaluations': objective.forward_evaluations,
        'optimizer_constructed': False, 'scientific_gate_issued': False, **observed}
