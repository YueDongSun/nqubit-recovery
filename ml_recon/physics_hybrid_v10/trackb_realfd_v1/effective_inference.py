"""Execute actual frozen V2/V3 native point inference and independent replay.

No optimizer, loss scoring, posterior fabrication or hardware action is used.
The caller owns source acceptance, checkpoint loading and dataset lifetime.
This is an observation-model producer, not the old hash-only receipt emitter.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping

import torch

from .contracts import canonical_json_bytes, canonical_sha256, require_git_oid
from .effective_checkpoint_reader import Effective_checkpoint_replay
from .family_refinement_reader import Family_refinement_replay
from .effective_inference_context import Effective_inference_config, build_effective_inference_plan
from .effective_inference_artifact import (
    Effective_point_inference, point_inference_schema, point_scope, sampling_covariance_scope,
    unavailable_inference_quantities, native_point_observation_semantics,
)
from .effective_point_values import extract_effective_point_values


def _assert_frozen_inference_model(replay) -> None:
    replay.summary()
    if any(parameter.grad is not None for parameter in replay.final_model.parameters()):
        raise ValueError('Native inference requires a frozen model without residual parameter gradients')


def _model_artifact_binding(replay) -> dict[str, Any]:
    refined = type(replay) is Family_refinement_replay
    parent = replay.parent if refined else replay
    summary = replay.summary()
    return {'kind': 'FAMILY_REFINEMENT_OVERLAY' if refined else 'EFFECTIVE_CPU_CHECKPOINT',
        'file_sha': replay.file_sha, 'model_state_sha': replay.final_model_state_sha,
        'model_config_sha': replay.identity.model_config_sha,
        'producer_commit': summary['source_commit'], 'producer_tree': summary['source_tree'],
        'training_checkpoint_file_sha': parent.file_sha, 'training_identity_sha': parent.identity.sha256(),
        'training_source_commit': parent.identity.source_commit, 'training_source_tree': parent.identity.source_tree,
        'training_final_model_state_sha': parent.final_model_state_sha}


@torch.no_grad()
def run_effective_point_inference(
    replay: Effective_checkpoint_replay | Family_refinement_replay, dataset: Any, *,
    source_commit: str, source_tree: str, config: Effective_inference_config = Effective_inference_config(),
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> Effective_point_inference:
    """Execute every selected inference row once with its explicit visible view."""
    require_git_oid(source_commit, 'inference_source_commit')
    require_git_oid(source_tree, 'inference_source_tree')
    plan = build_effective_inference_plan(replay, dataset, config=config)
    _assert_frozen_inference_model(replay)
    binding = _model_artifact_binding(replay)
    runtime = {'device': 'cpu', 'parameter_precision': 'fp32', 'count_precision': 'fp64',
        'torch_version': str(torch.__version__), 'cpu_threads': torch.get_num_threads()}
    semantics = native_point_observation_semantics()
    rows, schemas = [], set()
    completed_batches = 0
    try:
        for batch_index in range(len(plan.batches)):
            batch = plan.checked_batch(batch_index)
            if any(value.device.type != 'cpu' for value in batch.values() if isinstance(value, torch.Tensor)):
                raise ValueError('Native inference supports admitted CPU tensors only')
            prediction = replay.final_model(batch)
            completed_batches += 1
            if any(prediction.get(name) != semantics[name] for name in ('response_scope', 'idler_event',
                    'parameter_interpretation', 'read_trials_semantics')) or any(batch.get(name) != semantics[name]
                    for name in ('native_read_count_origin', 'prepared_command_semantics')):
                raise ValueError('Actual native inference forward or input observation semantics differ')
            plan.checked_batch(batch_index)
            _assert_frozen_inference_model(replay)
            schemas.add(prediction['model_schema'])
            for index, sample in enumerate(batch['sample_ids']):
                rows.append({'sample_id': sample, 'batch_index': batch_index,
                    'input_sha': plan.batch_input_shas[batch_index],
                    'operation_ids': list(batch['operation_ids']), 'operation_families': list(batch['operation_families']),
                    'cycle_indices': list(batch['cycle_indices']),
                    **{name: batch[name][index].tolist() for name in ('operation_mask', 'visible_operation_mask', 'query_mask')},
                    'values': extract_effective_point_values(prediction, batch, index)})
            if progress is not None:
                progress({'completed_batches': completed_batches, 'completed_rows': len(rows),
                    'batch_input_sha': plan.batch_input_shas[batch_index], 'model_parameters_updated': False})
    finally:
        _assert_frozen_inference_model(replay)
    if len(schemas) != 1 or runtime['cpu_threads'] != torch.get_num_threads() or binding != _model_artifact_binding(replay):
        raise ValueError('Native inference model schema, runtime or binding changed during execution')
    protocol = json.loads(plan.protocol_bytes)
    payload = {'schema': point_inference_schema, 'source_commit': source_commit, 'source_tree': source_tree,
        'system_type': replay.identity.system_type, 'model_schema': next(iter(schemas)), 'model_artifact': binding,
        'inference_protocol': protocol, 'inference_protocol_sha': canonical_sha256(protocol), 'rows': rows,
        'runtime': runtime, 'forward_evaluations': completed_batches, 'scope': point_scope,
        'sampling_covariance_scope': sampling_covariance_scope, 'unavailable': unavailable_inference_quantities(),
        'observation_semantics': semantics,
        'model_forward_executed': True, 'optimizer_constructed': False, 'model_parameters_updated': False,
        'scientific_gate_issued': False, 'hardware_action_authorized': False}
    encoded = canonical_json_bytes(payload)
    artifact = Effective_point_inference(encoded, hashlib.sha256(encoded).hexdigest())
    artifact.payload()
    return artifact


def verify_effective_point_replay(
    artifact: Effective_point_inference, replay: Effective_checkpoint_replay | Family_refinement_replay,
    dataset: Any, *, source_commit: str, source_tree: str,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Independently re-execute native predictions, retaining both source IDs."""
    if type(artifact) is not Effective_point_inference:
        raise TypeError('Independent replay requires a validated native point artifact')
    original = artifact.payload()
    repeated = run_effective_point_inference(replay, dataset, source_commit=source_commit, source_tree=source_tree,
        config=Effective_inference_config(**original['inference_protocol']['query_recipe']), progress=progress).payload()
    compare_fields = ('system_type', 'model_schema', 'model_artifact', 'inference_protocol', 'inference_protocol_sha',
        'rows', 'scope', 'sampling_covariance_scope', 'unavailable', 'observation_semantics', 'forward_evaluations')
    if any(canonical_json_bytes(original[name]) != canonical_json_bytes(repeated[name]) for name in compare_fields):
        raise ValueError('Independent native point inference does not exactly reproduce its source-bound outputs')
    return {'schema': 'TrackBEffectivePointReplayVerificationV1', 'point_artifact_file_sha': artifact.file_sha,
        'original_source_commit': original['source_commit'], 'original_source_tree': original['source_tree'],
        'replay_source_commit': source_commit, 'replay_source_tree': source_tree,
        'model_artifact_file_sha': repeated['model_artifact']['file_sha'],
        'inference_protocol_sha': repeated['inference_protocol_sha'], 'row_count': len(repeated['rows']),
        'model_forward_evaluations': repeated['forward_evaluations'], 'prediction_values_identical': True,
        'original_runtime': original['runtime'], 'replay_runtime': repeated['runtime'],
        'runtime_identical': original['runtime'] == repeated['runtime'], 'optimizer_constructed': False,
        'scientific_gate_issued': False, 'hardware_action_authorized': False}
